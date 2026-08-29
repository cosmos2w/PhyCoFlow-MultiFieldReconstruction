"""Traceable checkpoint evaluation shared by every case-local launcher."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from ..config import load_config
from ..data.factory import FieldDataset, open_field_dataset
from ..data.manifest import (
    SensorManifest,
    build_batch_from_manifest,
    dataset_fingerprint,
    manifest_from_batch,
)
from ..data.normalization import FieldNormalizer
from ..data.sensor_protocols import build_observation_batch
from ..models import build_model
from ..physics import build_case_diagnostics, build_case_physics
from ..training.common import sensor_protocol_from_config
from ..training.model_lifecycle import (
    load_training_aux_state,
    selected_evaluation_weight_context,
)
from ..training.run_store import file_sha256, load_model_state_strict, load_project_checkpoint
from .metrics import reconstruction_metrics


@dataclass
class EvaluationRuntime:
    """Loaded checkpoint, dataset, and policy shared by evaluation workflows."""

    config: dict[str, Any]
    device: torch.device
    dataset: FieldDataset
    model: torch.nn.Module
    checkpoint_path: Path
    generation_steps: int
    seed: int


def _checkpoint_path(run_dir: Path, checkpoint: str) -> Path:
    candidate = Path(checkpoint)
    if candidate.is_absolute():
        return candidate
    name = candidate.name if candidate.suffix == ".pt" else f"{candidate.name}.pt"
    return run_dir / "checkpoints" / name


def _physical_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    dataset: FieldDataset,
) -> dict[str, Any]:
    physical_prediction = dataset.normalizer.decode(prediction)
    physical_target = dataset.normalizer.decode(target)
    squared = (physical_prediction - physical_target).square()
    return {
        "mse_physical": float(squared.mean().cpu()),
        "per_field_mse_physical": {
            name: float(squared[..., index].mean().cpu())
            for index, name in enumerate(dataset.field_names)
        },
    }


def _uncertainty_metrics(samples: torch.Tensor | None, target: torch.Tensor) -> dict[str, Any]:
    if samples is None:
        return {"available": False, "reason": "model returned no ensemble samples"}
    variance = samples.var(dim=0, unbiased=False)
    ensemble_mean = samples.mean(dim=0)
    return {
        "available": True,
        "sample_count": int(samples.shape[0]),
        "mean_predictive_variance": float(variance.mean().cpu()),
        "ensemble_mean_mse_normalized": float((ensemble_mean - target).square().mean().cpu()),
    }


def load_evaluation_runtime(
    run_dir: str | Path,
    *,
    split: str,
    checkpoint: str,
    sensor_config: str | Path | None,
    generation_steps: int | None,
    device_name: str | None,
    include_temporal_derivative: bool = True,
) -> EvaluationRuntime:
    """Load one verified model and dataset for single- or multi-snapshot evaluation."""
    run_dir = Path(run_dir).resolve()
    config = load_config(run_dir / "resolved_config.yaml")
    if sensor_config is not None:
        sensor_settings = load_config(sensor_config)
        if "observations" not in sensor_settings:
            raise ValueError("sensor config must contain an observations section")
        config["observations"] = sensor_settings["observations"]
    device = torch.device(device_name or config.get("runtime", {}).get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    dataset_config = dict(config["dataset"])
    if include_temporal_derivative and Path(dataset_config["path"]).suffix.lower() in {
        ".h5",
        ".hdf5",
    }:
        dataset_config["include_temporal_derivative"] = True
    dataset = open_field_dataset(dataset_config, split=split)

    physics = None
    if config["stage"] == "direct_physics" or config["model"]["name"] == "pinn":
        physics = build_case_physics(
            config["case"], config["physics"], dataset.data_spec, dataset.normalizer
        )
    model = build_model(config["model"], dataset.data_spec, physics_provider=physics).to(device)
    checkpoint_path = _checkpoint_path(run_dir, checkpoint)
    payload = load_project_checkpoint(checkpoint_path)
    checkpoint_normalizer = FieldNormalizer(**payload["normalization"])
    if checkpoint_normalizer.digest() != dataset.normalizer.digest():
        dataset.close()
        raise ValueError("checkpoint normalization disagrees with the configured dataset")
    load_model_state_strict(model, payload["model"])
    load_training_aux_state(model, payload)
    model.eval()
    steps = int(
        generation_steps
        if generation_steps is not None
        else config.get("evaluation", {}).get("generation_steps", 2)
    )
    return EvaluationRuntime(
        config=config,
        device=device,
        dataset=dataset,
        model=model,
        checkpoint_path=checkpoint_path,
        generation_steps=steps,
        seed=int(config.get("evaluation", {}).get("seed", 2027)),
    )


def evaluate_run(
    run_dir: str | Path,
    *,
    case_dir: str | Path,
    split: str = "validation",
    sample_index: int = 0,
    max_samples: int = 1,
    checkpoint: str = "best",
    sensor_config: str | Path | None = None,
    sensor_manifest: str | Path | None = None,
    query_points: int | None = None,
    generation_steps: int | None = None,
    device_name: str | None = None,
    report_name: str = "benchmark",
    weight_selection: str = "configured",
) -> Path:
    """Evaluate a native project run and persist metrics plus a plotting payload."""
    run_dir = Path(run_dir).resolve()
    case_dir = Path(case_dir).resolve()
    runtime = load_evaluation_runtime(
        run_dir,
        split=split,
        checkpoint=checkpoint,
        sensor_config=sensor_config,
        generation_steps=generation_steps,
        device_name=device_name,
    )
    config = runtime.config
    device = runtime.device
    dataset = runtime.dataset
    model = runtime.model
    checkpoint_path = runtime.checkpoint_path
    sample_index = int(sample_index)
    if not 0 <= sample_index < len(dataset):
        dataset.close()
        raise IndexError(
            f"sample_index={sample_index} is outside {split} split with {len(dataset)} samples"
        )
    sample_count = min(int(max_samples), len(dataset) - sample_index)
    if sample_count < 1:
        dataset.close()
        raise ValueError(f"split {split!r} contains no samples")
    samples = [dataset[index] for index in range(sample_index, sample_index + sample_count)]
    if sensor_manifest is None:
        batch = build_observation_batch(
            samples, sensor_protocol_from_config(config), query_points=query_points
        )
        manifest = manifest_from_batch(batch, dataset.path, split)
    else:
        manifest = SensorManifest.load(sensor_manifest)
        batch = build_batch_from_manifest(
            samples, manifest, dataset.path, query_points=query_points
        )
    batch = batch.to(device)

    steps = runtime.generation_steps
    seed = runtime.seed

    with selected_evaluation_weight_context(model, weight_selection), torch.no_grad():
        warmup = torch.Generator(device=device).manual_seed(seed)
        model.reconstruct(batch, steps=steps, generator=warmup)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = perf_counter()
        generator = torch.Generator(device=device).manual_seed(seed)
        reconstruction = model.reconstruct(batch, steps=steps, generator=generator)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = perf_counter() - started

    target = batch.target_fields
    if target is None:
        raise ValueError("checkpoint evaluation requires dense targets")
    report: dict[str, Any] = {
        **reconstruction_metrics(reconstruction.prediction, target, batch, dataset.field_names),
        **_physical_metrics(reconstruction.prediction, target, dataset),
        "uncertainty": _uncertainty_metrics(reconstruction.samples, target),
        "compute": {
            "seconds": seconds,
            "seconds_per_sample": seconds / sample_count,
            "query_points_per_sample": int(batch.query_valid_mask[0].sum().cpu()),
            "peak_cuda_memory_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
        },
    }
    diagnostics = build_case_diagnostics(config["case"], dataset.data_spec, dataset.normalizer)
    if diagnostics is not None:
        report["case_diagnostics"] = diagnostics.evaluate(reconstruction.prediction, batch)

    output_dir = run_dir / "evaluation" / report_name
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "sensor_manifest.json"
    manifest.save(manifest_path)
    query_path = output_dir / "query_indices.pt"
    torch.save({"query_indices": batch.metadata.get("query_indices")}, query_path)
    # The compressed NumPy payload supersedes the former duplicate PyTorch plotting payload.
    # Remove a stale copy when an evaluation directory is regenerated by newer code.
    (output_dir / "reconstruction.pt").unlink(missing_ok=True)
    portable_plot_path = output_dir / "reconstruction.npz"
    query_indices = batch.metadata.get("query_indices")
    if not isinstance(query_indices, torch.Tensor):
        raise TypeError("evaluation plotting requires serialized query indices")
    first_query_indices = query_indices.cpu()[0, batch.query_valid_mask[0].cpu()]
    np.savez_compressed(
        portable_plot_path,
        prediction_physical=dataset.normalizer.decode(reconstruction.prediction[:1]).cpu().numpy(),
        target_physical=dataset.normalizer.decode(target[:1]).cpu().numpy(),
        query_coords=batch.query_coords[:1].cpu().numpy(),
        query_coords_physical=samples[0].coordinates_raw[first_query_indices].unsqueeze(0).numpy(),
        obs_coords=batch.obs_coords[:1].cpu().numpy(),
        obs_values_physical=(
            batch.obs_values[:1, :, 0].cpu()
            * dataset.normalizer.scale[batch.obs_field_ids[:1].cpu()]
            + dataset.normalizer.offset[batch.obs_field_ids[:1].cpu()]
        ).numpy(),
        obs_field_ids=batch.obs_field_ids[:1].cpu().numpy(),
        obs_valid_mask=batch.obs_valid_mask[:1].cpu().numpy(),
        obs_indices=batch.obs_indices[:1].cpu().numpy(),
        logical_shape=np.asarray(dataset.data_spec.logical_shape, dtype=np.int64),
        field_names=np.asarray(dataset.field_names),
        sample_id=np.asarray(batch.sample_ids[0]),
    )
    report["trace"] = {
        "run": str(run_dir.relative_to(case_dir)),
        "resolved_config_sha256": file_sha256(run_dir / "resolved_config.yaml"),
        "checkpoint": str(checkpoint_path.relative_to(run_dir)),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "sensor_manifest": str(manifest_path.relative_to(run_dir)),
        "sensor_manifest_sha256": manifest.digest(),
        "query_indices": str(query_path.relative_to(run_dir)),
        "query_indices_sha256": file_sha256(query_path),
        "dataset_fingerprint": dataset_fingerprint(dataset.path),
        "dataset_catalog_name": dataset.path.name,
        "split": split,
        "sample_index": sample_index,
        "weight_selection": weight_selection,
        "sample_ids": list(batch.sample_ids),
        "portable_plot_payload": str(portable_plot_path.relative_to(run_dir)),
        "portable_plot_payload_sha256": file_sha256(portable_plot_path),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    dataset.close()
    return report_path
