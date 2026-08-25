"""Shared plain base-training loop for every Phase-4 model adapter.

The loop is intentionally conventional and readable: sample a deterministic
mini-batch, build the common observation contract, compute the adapter's native
data/generative loss, update, and checkpoint atomically. `max_steps` truncates
only integration checks; normal configs run their declared epochs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ..data.factory import open_field_dataset
from ..data.manifest import dataset_fingerprint, manifest_from_batch
from ..data.training_batches import build_training_batch_source, dataset_field_bytes
from ..evaluation import reconstruction_metrics
from ..models import build_model
from ..registry import MODEL_REGISTRY
from ..utils.reproducibility import seed_everything
from .benchmark_telemetry import BenchmarkTelemetry
from .checkpointing import PeriodicCheckpointManager
from .common import (
    iter_unique_batch_indices,
)
from .gradients import stable_clip_grad_norm_
from .monitoring import TrainingMonitor
from .preview import TrainingReconstructionPreview
from .run_store import RunStore, checkpoint_model_state, file_sha256, load_model_state_strict


def _build_dataset(config: Mapping[str, Any]):
    data = config["dataset"]
    if not bool(data.get("benchmark_eligible", True)) and not bool(
        data.get("allow_nonbenchmark", False)
    ):
        raise ValueError(
            "this dataset is an integration fixture, not a formal benchmark; set "
            "dataset.allow_nonbenchmark=true only for an acknowledged integration run"
        )
    return open_field_dataset(data)


def run_base_training(
    config: Mapping[str, Any],
    *,
    case_dir: str | Path,
    max_steps: int | None = None,
    resume: str | Path | None = None,
) -> Path:
    if config["stage"] != "base_training":
        raise ValueError("run_base_training accepts only stage=base_training")
    seed = int(config["runtime"].get("seed", 42))
    seed_everything(seed, bool(config["runtime"].get("deterministic", True)))
    device = torch.device(config["runtime"].get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    dataset = _build_dataset(config)
    if not len(dataset):
        raise ValueError(f"training split is empty for {config['dataset']['path']}")
    model = build_model(config["model"], dataset.data_spec).to(device)
    registry_entry = MODEL_REGISTRY.get(config["model"]["name"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    telemetry_settings = config.get("benchmark_telemetry", {})
    if not isinstance(telemetry_settings, Mapping):
        raise TypeError("benchmark_telemetry must be a mapping")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimization"].get("lr", 1e-4)),
        weight_decay=float(config["optimization"].get("weight_decay", 0.0)),
    )

    if resume is None:
        experiment = config["output"].get("experiment_name", f"base_{config['model']['name']}")
        store = RunStore.create(case_dir, experiment, config)
        start_step = 0
    else:
        store = RunStore.resume(resume, config)
        checkpoint = store.load_checkpoint("last")
        load_model_state_strict(model, checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["global_step"])

    batch_size = int(config["optimization"].get("batch_size", 1))
    epochs = int(config["optimization"].get("epochs", 1))
    steps_per_epoch = math.ceil(len(dataset) / batch_size)
    configured_steps = epochs * steps_per_epoch
    if start_step >= configured_steps:
        raise ValueError(f"run already reached configured_steps={configured_steps}")
    final_step = (
        min(configured_steps, start_step + max_steps) if max_steps is not None else configured_steps
    )
    query_points = (
        None if model.capabilities.structured_grid_required else config["model"].get("query_points")
    )
    index_generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    if resume is not None:
        torch.set_rng_state(checkpoint["rng_state"]["torch_cpu"])
        if device.type == "cuda" and checkpoint["rng_state"].get("torch_cuda"):
            torch.cuda.set_rng_state_all(checkpoint["rng_state"]["torch_cuda"])
        index_generator.set_state(checkpoint["rng_state"]["index_generator"])
    store.update_manifest(
        dataset_path=str(dataset.path),
        dataset_fingerprint=dataset_fingerprint(dataset.path),
        dataset_split=dataset.split_name,
        split_strategy=dataset.selection.strategy,
        model_name=registry_entry.name,
        model_version=registry_entry.version,
        model_registry_metadata=registry_entry.metadata,
        benchmark_telemetry_enabled=bool(telemetry_settings.get("enabled", False)),
    )
    store.save_artifact("normalization.pt", dataset.normalizer.state_dict())
    store.write_json(
        "artifacts/split_manifest.json",
        {
            "split": dataset.selection.split,
            "strategy": dataset.selection.strategy,
            "trajectory_indices": list(dataset.selection.trajectory_indices),
            "frame_indices": list(dataset.selection.frame_indices),
        },
    )
    store.set_status("running", start_step=start_step, final_step=final_step)

    step_count = final_step - start_step
    sampled_indices = iter_unique_batch_indices(
        len(dataset),
        step_count,
        batch_size,
        generator=index_generator,
    )
    batch_source = build_training_batch_source(
        dataset,
        sampled_indices,
        config,
        query_points=query_points,
        device=device,
        start_step=start_step,
    )
    store.update_manifest(
        training_data_strategy=batch_source.strategy,
        training_dataset_logical_bytes=dataset_field_bytes(dataset),
    )

    monitor = TrainingMonitor(
        store.run_dir,
        start_step=start_step,
        final_step=final_step,
        configured_steps=configured_steps,
        steps_per_epoch=steps_per_epoch,
        description=f"base:{config['model']['name']}",
        enabled=bool(config["runtime"].get("progress", True)),
        plot_every_steps=int(config["runtime"].get("plot_every_steps", 10)),
    )
    preview = TrainingReconstructionPreview(
        config,
        store=store,
        steps_per_epoch=steps_per_epoch,
        device=device,
    )
    checkpoint_manager = PeriodicCheckpointManager(
        config,
        store=store,
        steps_per_epoch=steps_per_epoch,
    )
    telemetry = BenchmarkTelemetry.from_config(
        config,
        store.run_dir,
        device=device,
        steps_per_epoch=steps_per_epoch,
        parameter_count=parameter_count,
        trainable_parameter_count=trainable_parameter_count,
    )
    last_batch = None
    for step_offset, batch in enumerate(batch_source):
        global_step = start_step + step_offset
        telemetry.start_step(global_step)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        telemetry.start_phase("forward_native_loss")
        losses = model.training_loss(batch)
        telemetry.end_phase("forward_native_loss")
        if not torch.isfinite(losses.total):
            raise FloatingPointError(f"non-finite loss at step {global_step}: {losses.total}")
        telemetry.start_phase("backward")
        losses.total.backward()
        telemetry.end_phase("backward")
        gradient_norm = stable_clip_grad_norm_(
            model.parameters(), float(config["optimization"].get("grad_clip", 1.0))
        )
        telemetry.start_phase("optimizer")
        optimizer.step()
        telemetry.end_phase("optimizer")
        telemetry.finish_step()
        if (global_step + 1) % steps_per_epoch == 0:
            telemetry.finish_epoch((global_step + 1) // steps_per_epoch)
        row = {
            "step": global_step + 1,
            "total": float(losses.total.detach().cpu()),
            "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
            **{name: float(value.detach().cpu()) for name, value in losses.components.items()},
        }
        store.append_history(row)
        monitor.record(row, lr=optimizer.param_groups[0]["lr"])
        if checkpoint_manager.due_for_preview_or_checkpoint(global_step + 1, preview):
            checkpoint_manager.save(
                _base_checkpoint_payload(
                    model,
                    optimizer,
                    global_step=global_step + 1,
                    config=config,
                    dataset=dataset,
                    evaluation_mse=None,
                    parameter_count=parameter_count,
                    trainable_parameter_count=trainable_parameter_count,
                    index_generator=index_generator,
                    device=device,
                    config_sha256=store.config_hash,
                ),
                model=model,
                preview=preview,
                global_step=global_step + 1,
                fallback_metric=row["total"],
            )
        last_batch = batch
    monitor.close()
    batch_source.close()
    telemetry.close()

    if last_batch is None:
        raise ValueError("training performed no optimizer steps")
    model.eval()
    with torch.no_grad():
        evaluation_generator = torch.Generator(device=device).manual_seed(seed + 1_000_003)
        reconstruction = model.reconstruct(
            last_batch,
            steps=int(config.get("evaluation", {}).get("generation_steps", 2)),
            generator=evaluation_generator,
        )
        target = last_batch.target_fields
        metrics = (
            reconstruction_metrics(
                reconstruction.prediction, target, last_batch, dataset.field_names
            )
            if target is not None
            else {}
        )
        evaluation_mse = metrics.get("mse_normalized")

    training_manifest = manifest_from_batch(last_batch, dataset.path, dataset.split_name)
    training_manifest.save(store.run_dir / "artifacts" / "training_sensor_manifest.json")
    store.update_manifest(training_sensor_manifest_sha256=training_manifest.digest())
    store.write_json(
        "evaluation/integration.json",
        {
            **metrics,
            "sample_ids": list(last_batch.sample_ids),
            "sensor_manifest_sha256": training_manifest.digest(),
            "scope": "last_training_batch_integration_check"
            if max_steps is not None
            else "training_endpoint",
        },
    )

    checkpoint = _base_checkpoint_payload(
        model,
        optimizer,
        global_step=final_step,
        config=config,
        dataset=dataset,
        evaluation_mse=evaluation_mse,
        parameter_count=parameter_count,
        trainable_parameter_count=trainable_parameter_count,
        index_generator=index_generator,
        device=device,
        config_sha256=store.config_hash,
    )
    saved = checkpoint_manager.save(
        checkpoint,
        model=model,
        preview=preview,
        global_step=final_step,
        fallback_metric=float(row["total"]),
        force=True,
    )
    if saved is None:
        raise RuntimeError("final checkpoint save was unexpectedly disabled")
    last_checkpoint, _ = saved
    best_checkpoint = store.run_dir / "checkpoints" / "best.pt"
    if not best_checkpoint.is_file():
        best_checkpoint = store.save_checkpoint("best", checkpoint)
    preview.close()
    store.update_manifest(
        checkpoint_hashes={
            "last": file_sha256(last_checkpoint),
            "latest": file_sha256(last_checkpoint),
            "best": file_sha256(best_checkpoint),
        },
    )
    status = "completed" if final_step >= configured_steps else "integration_truncated"
    store.set_status(
        status,
        global_step=final_step,
        configured_steps=configured_steps,
        evaluation_mse=evaluation_mse,
        parameter_count=parameter_count,
        trainable_parameter_count=trainable_parameter_count,
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
    )
    dataset.close()
    return store.run_dir


def _base_checkpoint_payload(
    model,
    optimizer,
    *,
    global_step: int,
    config: Mapping[str, Any],
    dataset,
    evaluation_mse: float | None,
    parameter_count: int,
    trainable_parameter_count: int,
    index_generator: torch.Generator,
    device: torch.device,
    config_sha256: str,
) -> dict[str, Any]:
    """Build the same resumable payload for periodic and terminal saves."""
    return {
        "model": checkpoint_model_state(model),
        "optimizer": optimizer.state_dict(),
        "global_step": int(global_step),
        "model_name": config["model"]["name"],
        "model_config": dict(config["model"]),
        "data_spec": asdict(dataset.data_spec),
        "normalization": dataset.normalizer.state_dict(),
        "evaluation_mse": evaluation_mse,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "config_sha256": config_sha256,
        "rng_state": {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "index_generator": index_generator.get_state(),
        },
    }
