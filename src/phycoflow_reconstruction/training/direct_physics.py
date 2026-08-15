"""Direct physics-informed training with a verified case PhysicsProvider."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..data.h5_dataset import H5FieldDataset
from ..data.manifest import dataset_fingerprint, manifest_from_batch
from ..data.sensor_protocols import build_observation_batch
from ..evaluation import reconstruction_metrics
from ..models import build_model
from ..physics import build_case_physics
from ..utils.reproducibility import seed_everything
from .common import collate_field_samples, sensor_protocol_from_config
from .run_store import RunStore, checkpoint_model_state, file_sha256


def _dataset(config: Mapping[str, Any]) -> H5FieldDataset:
    data = config["dataset"]
    return H5FieldDataset(
        data["path"],
        split=data.get("split", "train"),
        reconstruction_unit=data.get("reconstruction_unit", "snapshot"),
        field_names=data.get("field_names"),
        field_units=data.get("field_units"),
        normalization=data.get("normalization", "auto"),
        coordinate_dim=data.get("coordinate_dim"),
        grid_shape=data.get("grid_shape"),
        coordinate_reorder=data.get("coordinate_reorder", "stored"),
        include_temporal_derivative=True,
    )


def run_direct_physics_training(
    config: Mapping[str, Any],
    *,
    case_dir: str | Path,
    max_steps: int | None = None,
    resume: str | Path | None = None,
) -> Path:
    if config["stage"] != "direct_physics":
        raise ValueError("direct physics trainer requires stage=direct_physics")
    if resume is not None:
        raise ValueError("direct-physics resume is not implemented; start an immutable new run")
    seed = int(config["runtime"].get("seed", 42))
    seed_everything(seed, bool(config["runtime"].get("deterministic", True)))
    device = torch.device(config["runtime"].get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dataset = _dataset(config)
    if not len(dataset):
        raise ValueError(f"training split is empty for {config['dataset']['path']}")
    physics = build_case_physics(
        config["case"], config["physics"], dataset.data_spec, dataset.normalizer
    )
    model = build_model(config["model"], dataset.data_spec, physics_provider=physics).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimization"].get("lr", 1e-4)),
        weight_decay=float(config["optimization"].get("weight_decay", 0.0)),
    )
    store = RunStore.create(
        case_dir,
        config["output"].get("experiment_name", "direct_physics"),
        config,
    )
    batch_size = int(config["optimization"].get("batch_size", 1))
    configured_steps = int(config["optimization"].get("epochs", 1)) * math.ceil(
        len(dataset) / batch_size
    )
    final_step = (
        min(configured_steps, int(max_steps)) if max_steps is not None else configured_steps
    )
    if final_step < 1:
        raise ValueError("direct-physics training would perform no optimizer steps")
    generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    indices = torch.randint(0, len(dataset), (final_step, batch_size), generator=generator).tolist()
    loader = DataLoader(
        dataset,
        batch_sampler=indices,
        collate_fn=collate_field_samples,
        num_workers=int(config["runtime"].get("num_workers", 0)),
    )
    protocol = sensor_protocol_from_config(config)
    store.update_manifest(
        dataset_path=str(dataset.path),
        dataset_fingerprint=dataset_fingerprint(dataset.path),
        physics_provider=config["physics"].get("provider", config["case"]),
        derivative_provenance="paired_finite_difference",
    )
    store.set_status("running", final_step=final_step)
    started = perf_counter()
    last_batch = None
    last_losses = None
    for step, samples in enumerate(loader):
        batch = build_observation_batch(samples, protocol).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = model.training_loss(batch)
        if not torch.isfinite(losses.total):
            raise FloatingPointError("direct-physics objective is non-finite")
        losses.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["optimization"].get("grad_clip", 1.0))
        )
        optimizer.step()
        store.append_history(
            {
                "step": step + 1,
                "total": float(losses.total.detach().cpu()),
                "gradient_norm": float(torch.as_tensor(gradient_norm).cpu()),
                **{name: float(value.detach().cpu()) for name, value in losses.components.items()},
            }
        )
        last_batch, last_losses = batch, losses
    if last_batch is None or last_losses is None:
        raise ValueError("direct-physics training performed no update")
    model.eval()
    with torch.no_grad():
        reconstruction = model.reconstruct(last_batch)
        metrics = reconstruction_metrics(
            reconstruction.prediction,
            last_batch.target_fields,
            last_batch,
            dataset.field_names,
        )
        metrics["physics"] = physics.evaluate(reconstruction.prediction, last_batch)
    manifest = manifest_from_batch(last_batch, dataset.path, dataset.split_name)
    manifest_path = store.run_dir / "artifacts" / "training_sensor_manifest.json"
    manifest.save(manifest_path)
    store.write_json("evaluation/integration.json", metrics)
    payload = {
        "model": checkpoint_model_state(model),
        "optimizer": optimizer.state_dict(),
        "global_step": final_step,
        "model_name": config["model"]["name"],
        "model_config": dict(config["model"]),
        "data_spec": asdict(dataset.data_spec),
        "normalization": dataset.normalizer.state_dict(),
        "config_sha256": store.config_hash,
    }
    last_path = store.save_checkpoint("last", payload)
    best_path = store.save_checkpoint("best", payload)
    store.update_manifest(
        checkpoint_hashes={
            "last": file_sha256(last_path),
            "best": file_sha256(best_path),
        },
        sensor_manifest_sha256=manifest.digest(),
    )
    status = "completed" if final_step >= configured_steps else "integration_truncated"
    store.set_status(
        status,
        global_step=final_step,
        configured_steps=configured_steps,
        training_seconds=perf_counter() - started,
        physics_total=float(last_losses.total.detach().cpu()),
    )
    dataset.close()
    return store.run_dir
