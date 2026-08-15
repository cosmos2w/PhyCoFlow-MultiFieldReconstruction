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
from torch.utils.data import DataLoader

from ..data.manifest import dataset_fingerprint, manifest_from_batch
from ..data.sensor_protocols import SensorProtocol, build_observation_batch
from ..evaluation import reconstruction_metrics
from ..physics import build_case_physics
from ..utils.reproducibility import seed_everything
from .common import collate_field_samples, sensor_protocol_from_config
from .gradient_balance import two_objective_update
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
    configured_steps = int(config["optimization"].get("epochs", 1)) * math.ceil(
        len(dataset) / batch_size
    )
    final_step = (
        min(configured_steps, int(max_steps)) if max_steps is not None else configured_steps
    )
    if final_step < 1:
        raise ValueError("physics post-training would perform no optimizer steps")
    index_generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    indices = torch.randint(
        0, len(dataset), (final_step, batch_size), generator=index_generator
    ).tolist()
    loader = DataLoader(dataset, batch_sampler=indices, collate_fn=collate_field_samples)
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
    )
    store.set_status("running", final_step=final_step)
    started = perf_counter()
    last_physics = None
    for step, samples in enumerate(loader):
        batch = build_observation_batch(samples, _protocol(config, step)).to(device)
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
        store.append_history(
            {
                "step": step + 1,
                "data_loss": float(data_loss.detach().cpu()),
                "physics_loss": float(physics_loss.total.detach().cpu()),
                **{
                    name: float(value.detach().cpu())
                    for name, value in physics_loss.components.items()
                },
                **gradient,
            }
        )
        last_physics = physics_loss.total
    after = _evaluate(model, physics, evaluation_batch, dataset.field_names, config, seed + 99)
    store.write_json("evaluation/after.json", after)
    payload = {
        "model": checkpoint_model_state(model),
        "optimizer": optimizer.state_dict(),
        "global_step": final_step,
        "model_name": config["model"]["name"],
        "model_config": dict(config["model"]),
        "data_spec": asdict(dataset.data_spec),
        "normalization": dataset.normalizer.state_dict(),
        "source_run": str(config["source_run"]),
        "source_hashes": source_before,
        "config_sha256": store.config_hash,
    }
    last_path = store.save_checkpoint("last", payload)
    best_path = store.save_checkpoint("best", payload)
    source_after = source_hashes(config)
    if source_after != source_before:
        raise RuntimeError("source run changed during physics post-training")
    store.update_manifest(
        checkpoint_hashes={"last": file_sha256(last_path), "best": file_sha256(best_path)},
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
