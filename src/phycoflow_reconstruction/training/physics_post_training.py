"""Physics-informed post-training of an immutable plain-model child run.

The objective may retain a weighted data term while the case-owned physics
provider supplies the refinement signal.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from ..data.manifest import dataset_fingerprint, manifest_from_batch
from ..data.sensor_protocols import SensorProtocol, build_observation_batch
from ..data.training_batches import build_training_batch_source, dataset_field_bytes
from ..evaluation import reconstruction_metrics
from ..physics import build_case_physics
from ..utils.reproducibility import seed_everything
from .checkpointing import PeriodicCheckpointManager
from .common import (
    iter_unique_batch_indices,
    sensor_protocol_from_config,
)
from .gradient_balance import two_objective_update
from .monitoring import TrainingMonitor
from .preview import TrainingReconstructionPreview
from .rollout import differentiable_reconstruction
from .run_store import RunStore, checkpoint_model_state, file_sha256
from .source import (
    load_source_model,
    set_trainable_scope,
    source_hashes,
)


def _protocol(config: Mapping[str, Any], step: int = 0) -> SensorProtocol:
    return sensor_protocol_from_config(config, seed_offset=step)


def _evaluate(model, physics, batch, fields, config, seed: int) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        generator = torch.Generator(device=batch.query_coords.device).manual_seed(seed)
        reconstruction = model.reconstruct(
            batch,
            steps=int(config.get("rollout", {}).get("steps", 2)),
            generator=generator,
        )
        metrics = reconstruction_metrics(
            reconstruction.prediction, batch.target_fields, batch, fields
        )
        metrics["physics"] = physics.evaluate(reconstruction.prediction, batch)
    return metrics


def run_physics_post_training(
    config: Mapping[str, Any],
    *,
    case_dir: str | Path,
    max_steps: int | None = None,
    resume: str | Path | None = None,
) -> Path:
    if resume is not None:
        raise ValueError("physics post-training resume is not implemented")
    seed = int(config["runtime"].get("seed", 42))
    seed_everything(seed, bool(config["runtime"].get("deterministic", True)))
    device = torch.device(config["runtime"].get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    source_before = source_hashes(config)
    model, dataset, source_metadata = load_source_model(config, device)
    if not len(dataset):
        raise ValueError(f"training split is empty for {config['dataset']['path']}")
    if not model.capabilities.differentiable_rollout:
        raise ValueError("source model cannot drive differentiable physics post-training")
    trainable = set_trainable_scope(model, config.get("trainable", {"scope": "full_model"}))
    physics = build_case_physics(
        config["case"], config["physics"], dataset.data_spec, dataset.normalizer
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["optimization"].get("lr", 5e-5)),
        weight_decay=float(config["optimization"].get("weight_decay", 0.0)),
    )
    store = RunStore.create(
        case_dir,
        config["output"].get("experiment_name", "physics_posttrain"),
        config,
        parent_run=str(config["source_run"]),
    )
    batch_size = int(config["optimization"].get("batch_size", 1))
    steps_per_epoch = math.ceil(len(dataset) / batch_size)
    configured_steps = int(config["optimization"].get("epochs", 1)) * steps_per_epoch
    final_step = (
        min(configured_steps, int(max_steps)) if max_steps is not None else configured_steps
    )
    if final_step < 1:
        raise ValueError("physics post-training would perform no optimizer steps")
    index_generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    indices = iter_unique_batch_indices(
        len(dataset), final_step, batch_size, generator=index_generator
    )
    batch_source = build_training_batch_source(
        dataset,
        indices,
        config,
        query_points=None,
        device=device,
        start_step=0,
    )
    evaluation_batch = build_observation_batch([dataset[0]], _protocol(config)).to(device)
    manifest = manifest_from_batch(evaluation_batch, dataset.path, dataset.split_name)
    manifest_path = store.run_dir / "artifacts" / "evaluation_sensor_manifest.json"
    manifest.save(manifest_path)
    before = _evaluate(model, physics, evaluation_batch, dataset.field_names, config, seed + 99)
    store.write_json("evaluation/before.json", before)
    store.update_manifest(
        source_hashes=source_before,
        source_metadata=source_metadata,
        source_checkpoint=str(config["source_checkpoint"]),
        dataset_fingerprint=dataset_fingerprint(dataset.path),
        trainable_parameter_names=list(trainable),
        sensor_manifest_sha256=manifest.digest(),
        training_data_strategy=batch_source.strategy,
        training_dataset_logical_bytes=dataset_field_bytes(dataset),
    )
    store.set_status("running", final_step=final_step)
    started = perf_counter()
    monitor = TrainingMonitor(
        store.run_dir,
        start_step=0,
        final_step=final_step,
        configured_steps=configured_steps,
        steps_per_epoch=steps_per_epoch,
        description=f"physics-post:{config['model']['name']}",
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
    last_physics = None
    for step, batch in enumerate(batch_source):
        model.train()
        generator = torch.Generator(device=device).manual_seed(seed + 1_000_003 + step)
        prediction = differentiable_reconstruction(
            model,
            batch,
            steps=int(config.get("rollout", {}).get("steps", 2)),
            solver=config.get("rollout", {}).get("solver", "euler"),
            generator=generator,
            observation_config=dict(config.get("observation_consistency", {"mode": "none"})),
        )
        data_loss = (prediction - batch.target_fields).square().mean()
        physics_loss = physics.loss(prediction, batch)
        data_settings = config.get("objectives", {}).get("data_retention", {})
        physics_settings = config.get("objectives", {}).get("physics", {})
        gradient = two_objective_update(
            model,
            optimizer,
            data_loss,
            physics_loss.total,
            mode=config["optimization"].get("gradient_balance", "weighted_sum"),
            data_weight=(
                float(data_settings.get("weight", 0.1))
                if bool(data_settings.get("enabled", True))
                else 0.0
            ),
            coherence_weight=(
                float(physics_settings.get("weight", 1.0))
                if bool(physics_settings.get("enabled", True))
                else 0.0
            ),
            grad_clip=config["optimization"].get("grad_clip"),
            config_missing_behavior=config["optimization"].get("config_missing_behavior", "error"),
        )
        row = {
                "step": step + 1,
                "data_loss": float(data_loss.detach().cpu()),
                "physics_loss": float(physics_loss.total.detach().cpu()),
                **{
                    name: float(value.detach().cpu())
                    for name, value in physics_loss.components.items()
                },
                **gradient,
            }
        store.append_history(row)
        monitor.record(row, lr=optimizer.param_groups[0]["lr"])
        if checkpoint_manager.due_for_preview_or_checkpoint(step + 1, preview):
            checkpoint_manager.save(
                _physics_post_checkpoint_payload(
                    model,
                    optimizer,
                    global_step=step + 1,
                    config=config,
                    dataset=dataset,
                    source_hashes_before=source_before,
                    config_sha256=store.config_hash,
                ),
                model=model,
                preview=preview,
                global_step=step + 1,
                fallback_metric=row["physics_loss"],
            )
        last_physics = physics_loss.total
    monitor.close()
    batch_source.close()
    after = _evaluate(model, physics, evaluation_batch, dataset.field_names, config, seed + 99)
    store.write_json("evaluation/after.json", after)
    payload = _physics_post_checkpoint_payload(
        model,
        optimizer,
        global_step=final_step,
        config=config,
        dataset=dataset,
        source_hashes_before=source_before,
        config_sha256=store.config_hash,
    )
    saved = checkpoint_manager.save(
        payload,
        model=model,
        preview=preview,
        global_step=final_step,
        fallback_metric=float(last_physics.detach().cpu()),
        force=True,
    )
    if saved is None:
        raise RuntimeError("final checkpoint save was unexpectedly skipped")
    last_path, _ = saved
    best_path = store.run_dir / "checkpoints" / "best.pt"
    if not best_path.is_file():
        best_path = store.save_checkpoint("best", payload)
    preview.close()
    source_after = source_hashes(config)
    if source_after != source_before:
        raise RuntimeError("source run changed during physics post-training")
    store.update_manifest(
        checkpoint_hashes={
            "last": file_sha256(last_path),
            "latest": file_sha256(last_path),
            "best": file_sha256(best_path),
        },
        source_hashes_after=source_after,
        source_immutable_verified=True,
    )
    status = "completed" if final_step >= configured_steps else "integration_truncated"
    store.set_status(
        status,
        global_step=final_step,
        configured_steps=configured_steps,
        post_training_seconds=perf_counter() - started,
        final_physics_loss=float(last_physics.detach().cpu()),
        source_immutable_verified=True,
    )
    dataset.close()
    return store.run_dir


def _physics_post_checkpoint_payload(
    model,
    optimizer,
    *,
    global_step: int,
    config: Mapping[str, Any],
    dataset,
    source_hashes_before: Mapping[str, Any],
    config_sha256: str,
) -> dict[str, Any]:
    """Build the common periodic/terminal physics-refinement checkpoint."""
    return {
        "model": checkpoint_model_state(model),
        "optimizer": optimizer.state_dict(),
        "global_step": int(global_step),
        "model_name": config["model"]["name"],
        "model_config": dict(config["model"]),
        "data_spec": asdict(dataset.data_spec),
        "normalization": dataset.normalizer.state_dict(),
        "source_run": str(config["source_run"]),
        "source_hashes": dict(source_hashes_before),
        "config_sha256": config_sha256,
    }
