"""Opt-in paired native-grid topology audit hooks for post-training runs.

The audit owns no optimizer or selector state.  It evaluates the live source
anchor and live child on one separately selected, complete validation lattice,
then writes identity-bound JSON records beneath the child RunStore.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from ..coherence.families.topology.geometry import coordinate_digest
from ..coherence.families.topology.persistence import (
    cubical_diagrams,
    sliced_diagram_distance,
)
from ..coherence.families.topology.spatial_persistence import spatial_diagram_distance
from ..contracts import ObservationBatch
from ..data.manifest import dataset_fingerprint
from ..data.sensor_protocols import SensorProtocol, build_observation_batch
from ..evaluation.topology_set import TopologySetAccumulator
from .run_store import file_sha256

AUDIT_SUBDIR = "evaluation/native_topology_audit"


def audit_settings(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    settings = config.get("evaluation", {}).get("native_topology_audit", {})
    if not isinstance(settings, Mapping):
        raise TypeError("evaluation.native_topology_audit must be a mapping")
    return settings if bool(settings.get("enabled", False)) else None


def audit_due(
    committed_step: int,
    every_steps: int,
    *,
    durable: bool,
    update_accepted: bool = True,
    initial: bool = False,
    terminal: bool = False,
) -> bool:
    """Cadence is based on committed updates and never runs ahead of a save."""
    if not durable:
        return False
    if initial:
        return committed_step == 0
    if terminal:
        return True
    return update_accepted and committed_step > 0 and committed_step % every_steps == 0


def _sample_id(sample: Any) -> str:
    time_index = getattr(sample, "time_index", None)
    return f"{sample.trajectory_id}:{time_index if time_index is not None else 'all'}"


def _sample_id_for_index(dataset: Any, index: int) -> str:
    """Read the canonical lightweight index-to-ID mapping without loading fields."""
    items = getattr(dataset, "_items", None)
    trajectory_ids = getattr(dataset, "trajectory_ids", None)
    if items is None or trajectory_ids is None:
        raise TypeError("native topology audit requires the dataset index/sample-ID contract")
    trajectory_index, time_index = items[int(index)]
    suffix = time_index if time_index is not None else "all"
    return f"{trajectory_ids[int(trajectory_index)]}:{suffix}"


def build_disjoint_validation_batch(
    dataset: Any,
    protocol: SensorProtocol,
    *,
    selector_sample_ids: tuple[str, ...] | list[str],
    max_samples: int,
    seed: int,
    device: torch.device,
) -> tuple[ObservationBatch, dict[str, Any]]:
    """Build one reproducible full-query validation batch outside the selector."""
    excluded = set(map(str, selector_sample_ids))
    candidates = [
        index
        for index in range(len(dataset))
        if _sample_id_for_index(dataset, index) not in excluded
    ]
    if len(candidates) < max_samples:
        raise ValueError(
            "native topology audit requires enough validation snapshots disjoint from "
            f"the selector (need {max_samples}, found {len(candidates)})"
        )
    rng = np.random.default_rng(int(seed))
    positions = np.sort(rng.choice(len(candidates), size=max_samples, replace=False))
    selected_indices = [candidates[int(position)] for position in positions]
    selected_samples = [dataset[index] for index in selected_indices]
    metadata_sample_ids = [_sample_id_for_index(dataset, index) for index in selected_indices]
    loaded_sample_ids = [_sample_id(sample) for sample in selected_samples]
    if metadata_sample_ids != loaded_sample_ids:
        raise ValueError(
            "dataset index/sample-ID contract changed while selecting native audit data"
        )
    batch = build_observation_batch(selected_samples, protocol, query_points=None).to(device)
    metadata = {
        "dataset_indices": [int(index) for index in selected_indices],
        "sample_ids": list(batch.sample_ids),
        "selector_sample_ids": sorted(excluded),
        "audit_seed": int(seed),
        "protocol": protocol.to_dict(),
    }
    return batch, metadata


def validate_complete_native_batch(
    batch: ObservationBatch,
    grid_shape: tuple[int, int],
    *,
    excluded_sample_ids: tuple[str, ...] | list[str] = (),
) -> None:
    """Reject partial, repeated, selector-overlapping, or malformed native grids."""
    height, width = map(int, grid_shape)
    expected_count = height * width
    if batch.target_fields is None or batch.target_fields.shape[1] != expected_count:
        raise ValueError("native topology audit requires a complete dense target field")
    if batch.query_coords.shape[1] != expected_count:
        raise ValueError("native topology audit requires every native query coordinate")
    if not bool(batch.query_valid_mask.all()):
        raise ValueError("native topology audit does not allow padded or masked queries")
    query_indices = batch.metadata.get("query_indices")
    if not isinstance(query_indices, torch.Tensor) or query_indices.shape != (
        len(batch.sample_ids),
        expected_count,
    ):
        raise ValueError("native topology audit requires saved complete query indices")
    expected = torch.arange(expected_count, device=query_indices.device).expand_as(query_indices)
    if not torch.equal(query_indices, expected):
        raise ValueError("native topology audit query indices must cover the full lattice once")
    if len(set(batch.sample_ids)) != len(batch.sample_ids):
        raise ValueError("native topology audit sample IDs must be unique")
    overlap = set(batch.sample_ids) & set(map(str, excluded_sample_ids))
    if overlap:
        raise ValueError(f"native topology audit overlaps selector samples: {sorted(overlap)}")
    if (
        not torch.isfinite(batch.query_coords).all()
        or not torch.isfinite(batch.target_fields).all()
    ):
        raise FloatingPointError(
            "native topology audit query fields and coordinates must be finite"
        )


def build_native_topology_family(
    config: Mapping[str, Any],
    dataset: Any,
    normalizer: Any,
    run_dir: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Restore the child's exact bank, then build only its native raster geometry."""
    runtime = SimpleNamespace(
        config=config,
        dataset=dataset,
        device=device,
        run_dir=Path(run_dir),
        coherence_artifact_run_dir=Path(run_dir),
    )
    restored = TopologySetAccumulator.build(runtime)
    saved_family = restored.family
    if getattr(saved_family, "strategy", None) != "cubical_persistence":
        raise ValueError("periodic native topology audit requires cubical_persistence")
    saved_objective = saved_family.spatial_objective
    if not saved_objective.line_sampling_enabled or not saved_objective.groups:
        raise ValueError("periodic native topology audit requires a serialized master line bank")
    grid_shape = tuple(map(int, config["dataset"]["grid_shape"]))
    if len(grid_shape) != 2 or tuple(dataset.data_spec.logical_shape) != grid_shape:
        raise ValueError("native topology audit grid must match the structured dataset shape")
    native_config = deepcopy(saved_family.config)
    native_config["geometry"] = dict(native_config.get("geometry", {}))
    native_config["geometry"]["grid_shape"] = list(grid_shape)
    native_config["geometry"]["antialias_downsample"] = True
    native_config["filtration"] = dict(native_config.get("filtration", {}))
    native_config["filtration"]["smoothing_sigma"] = 0.0
    data_spec = replace(dataset.data_spec, logical_shape=grid_shape)
    native_family = type(saved_family)(native_config, data_spec, normalizer).to(device)
    native_objective = native_family.spatial_objective
    with torch.no_grad():
        for index in range(len(saved_objective.groups)):
            for suffix in ("directions", "offsets"):
                name = f"line_{suffix}_{index}"
                getattr(native_objective, name).copy_(getattr(saved_objective, name))
    saved_sampling = saved_objective.sampling_artifact()
    native_sampling = native_objective.sampling_artifact()
    if saved_sampling != native_sampling:
        raise ValueError("native audit family did not restore the exact saved master line bank")
    full_lines = tuple(range(native_objective.line_bank_size))
    selected = native_objective.selected_line_indices(None, "evaluation")
    if selected != tuple(full_lines for _ in native_objective.groups):
        raise ValueError("native audit evaluation must use the complete saved line bank")
    native_family.eval()
    provenance = {
        **restored.family_artifact_provenance,
        "sampling_artifact": native_sampling,
        "configured_grid_shape": list(saved_family.grid_shape),
        "native_grid_shape": list(grid_shape),
        "geometry_sha256": native_family.geometry_sha256,
        "geometry_diagnostics": dict(native_family.geometry_diagnostics),
        "line_indices_evaluation": [list(indices) for indices in selected],
    }
    return native_family, provenance


def initialize_native_geometry(
    family: torch.nn.Module,
    batch: ObservationBatch,
    grid_shape: tuple[int, int],
    *,
    excluded_sample_ids: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    validate_complete_native_batch(batch, grid_shape, excluded_sample_ids=excluded_sample_ids)
    family._raster_map(batch.query_coords)
    diagnostics = dict(family.geometry_diagnostics)
    if (
        int(diagnostics.get("complete_native_cartesian_grid", 0)) != 1
        or int(diagnostics.get("native_gather_applied", 0)) != 1
        or int(diagnostics.get("native_area_average_applied", 0)) != 0
    ):
        raise ValueError("native audit requires a complete native gather/reshape geometry")
    if tuple(family.grid_shape) != tuple(grid_shape):
        raise ValueError("native audit family raster does not match the dataset grid")
    return {
        "grid_shape": list(grid_shape),
        "geometry_sha256": family.geometry_sha256,
        "query_coordinate_sha256": coordinate_digest(batch.query_coords[0]),
        "geometry_diagnostics": diagnostics,
        "smoothing_sigma": float(family.smoothing_sigma),
        "units": family.units,
    }


def _tensor_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        if torch.is_tensor(value):
            tensor = value.detach().to(device="cpu").contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, default=repr).encode("utf-8"))
    return digest.hexdigest()


@contextmanager
def _preserve_audit_state(models: tuple[torch.nn.Module, ...]):
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    training = [(module, bool(module.training)) for model in models for module in model.modules()]
    ema_settings = []
    for model in models:
        for module in model.modules():
            value = module.__dict__.get("_ema_eval")
            if isinstance(value, bool):
                ema_settings.append((module, value))
                module.__dict__["_ema_eval"] = False
    model_states = [_tensor_state_sha256(model) for model in models]
    try:
        yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.random.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        for module, was_training in training:
            module.training = was_training
        for module, value in ema_settings:
            module.__dict__["_ema_eval"] = value
        changed = [
            index
            for index, (model, before) in enumerate(zip(models, model_states))
            if _tensor_state_sha256(model) != before
        ]
        if changed:
            raise RuntimeError(f"native topology audit mutated model state(s): {changed}")


def _capture_forward_inputs(family: torch.nn.Module, capture: dict[str, torch.Tensor]):
    objective = family.spatial_objective

    def hook(_module, args, _kwargs):
        if len(args) < 2:
            raise RuntimeError("persistence objective input hook did not receive field tensors")
        capture["generated_grid"] = args[0].detach()
        capture["reference_grid"] = args[1].detach()

    return objective.register_forward_pre_hook(hook, with_kwargs=True)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _evaluate_pair_member(
    model: torch.nn.Module,
    family: torch.nn.Module,
    batch: ObservationBatch,
    config: Mapping[str, Any],
    field_names: tuple[str, ...],
    normalizer: Any,
    family_scales: Mapping[str, float],
    evaluate_fn: Callable[..., dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, torch.Tensor], float]:
    capture: dict[str, torch.Tensor] = {}
    hook = _capture_forward_inputs(family, capture)
    device = batch.query_coords.device
    _sync(device)
    started = time.perf_counter()
    try:
        report = evaluate_fn(
            model,
            batch,
            batch,
            {"topology": family},
            {"topology": None},
            field_names,
            config,
            normalizer=normalizer,
            family_scales=family_scales,
        )
    finally:
        hook.remove()
    _sync(device)
    elapsed = time.perf_counter() - started
    if set(capture) != {"generated_grid", "reference_grid"}:
        raise RuntimeError("native topology evaluation did not reach the persistence objective")
    for name, value in capture.items():
        if value.ndim != 4 or not torch.isfinite(value).all():
            raise FloatingPointError(f"native topology captured {name} is malformed or non-finite")
    return report, capture, elapsed


def _diagram_distance_parts(left: Any, right: Any, objective: Any, height: int, width: int):
    essential = (left.essential.sort().values - right.essential.sort().values).abs().sum()
    compare = (
        sliced_diagram_distance
        if objective.distance == "sliced_wasserstein"
        else spatial_diagram_distance
    )
    kwargs = (
        {"projections": objective.projections}
        if objective.distance == "sliced_wasserstein"
        else {
            "periodic": objective.periodic,
            "spatial_weight": objective.spatial_weight,
            "spatial_mode": objective.spatial_mode,
            "max_assignment_size": objective.max_assignment_size,
        }
    )
    total = compare(
        left,
        right,
        essential_weight=objective.essential_weight,
        normalization=height * width,
        **kwargs,
    )
    weighted_essential = objective.essential_weight * essential
    finite_normalized = total - weighted_essential
    return float(finite_normalized.cpu()), float(essential.cpu()), float(total.cpu())


def paired_bar_diagnostics(
    family: torch.nn.Module,
    source_capture: Mapping[str, torch.Tensor],
    candidate_capture: Mapping[str, torch.Tensor],
    sample_ids: tuple[str, ...],
    *,
    max_snapshots: int = 2,
) -> tuple[dict[str, Any], float]:
    """Compare source/child bars to the same detached GT bars, at most twice."""
    summary_started = time.perf_counter()
    source_reference = source_capture["reference_grid"]
    candidate_reference = candidate_capture["reference_grid"]
    if not torch.equal(source_reference, candidate_reference):
        raise ValueError("paired native source and child evaluations changed the GT reference grid")
    objective = family.spatial_objective
    count = min(int(max_snapshots), len(sample_ids), source_reference.shape[0])
    if count < 1:
        raise ValueError("bar diagnostics require at least one validation snapshot")
    reference = source_reference[:count].detach().to(device="cpu", dtype=torch.float32)
    reference_filtration_started = time.perf_counter()
    with torch.no_grad():
        descriptor_reference = objective._descriptor_fields(reference)
        mean = descriptor_reference.mean((-2, -1), keepdim=True)
        scale = descriptor_reference.std((-2, -1), keepdim=True, unbiased=False).clamp_min(
            objective.scale_floor
        )
        target_x = (descriptor_reference - mean) / scale
        target_bank, labels, line_weights, metric_labels = objective._filtrations(target_x)
        height, width = target_bank.shape[-2:]
    reference_filtration_seconds = time.perf_counter() - reference_filtration_started
    started = time.perf_counter()
    target_diagrams = cubical_diagrams(
        target_bank.flatten(0, 1),
        periodic=objective.periodic,
        dimensions=objective.dimensions,
        locations=objective.distance == "spatial_wasserstein",
        cacheable=True,
    )
    gt_seconds = time.perf_counter() - started
    roles = {}
    generated_banks: dict[str, torch.Tensor] = {}
    for role, capture in (("source_live", source_capture), ("candidate_live", candidate_capture)):
        generated = capture["generated_grid"][:count].detach().to(device="cpu", dtype=torch.float32)
        prediction_filtration_started = time.perf_counter()
        with torch.no_grad():
            descriptor_generated = objective._descriptor_fields(generated)
            prediction_x = (descriptor_generated - mean) / scale
            prediction_bank, pred_labels, pred_weights, pred_metric_labels = objective._filtrations(
                prediction_x
            )
        prediction_filtration_seconds = time.perf_counter() - prediction_filtration_started
        if (pred_labels, pred_metric_labels) != (labels, metric_labels) or not torch.equal(
            torch.as_tensor(pred_weights), torch.as_tensor(line_weights)
        ):
            raise RuntimeError("native bar diagnostic filtration rows differ from the saved family")
        started = time.perf_counter()
        prediction_diagrams = cubical_diagrams(
            prediction_bank.flatten(0, 1),
            periodic=objective.periodic,
            dimensions=objective.dimensions,
            locations=objective.distance == "spatial_wasserstein",
        )
        ph_seconds = time.perf_counter() - started
        rows = []
        distance_started = time.perf_counter()
        for filtration_index, (label, metric, line_weight) in enumerate(
            zip(labels, metric_labels, line_weights)
        ):
            for sample_index in range(count):
                row_index = filtration_index * count + sample_index
                for dimension in objective.dimensions:
                    generated_diagram = prediction_diagrams[row_index][dimension]
                    reference_diagram = target_diagrams[row_index][dimension]
                    finite, essential, total = _diagram_distance_parts(
                        generated_diagram,
                        reference_diagram,
                        objective,
                        height,
                        width,
                    )
                    rows.append(
                        {
                            "sample_id": sample_ids[sample_index],
                            "filtration_index": filtration_index,
                            "filtration": label,
                            "metric": metric,
                            "homology_dimension": int(dimension),
                            "line_weight": float(line_weight),
                            "reference_finite_bar_count": int(reference_diagram.finite.shape[0]),
                            "generated_finite_bar_count": int(generated_diagram.finite.shape[0]),
                            "reference_essential_class_count": int(
                                reference_diagram.essential.numel()
                            ),
                            "generated_essential_class_count": int(
                                generated_diagram.essential.numel()
                            ),
                            "finite_distance_normalized_by_hw": finite,
                            "essential_birth_l1_raw": essential,
                            "essential_distance_weight": float(objective.essential_weight),
                            "weighted_essential_distance_not_hw_normalized": float(
                                objective.essential_weight
                            )
                            * essential,
                            "total_distance_before_line_weight": total,
                            "total_distance_after_line_weight": total * float(line_weight),
                        }
                    )
        distance_seconds = time.perf_counter() - distance_started
        roles[role] = {
            "rows": rows,
            "diagnostic_snapshot_count": count,
            "prediction_filtration_seconds": prediction_filtration_seconds,
            "prediction_ph_seconds": ph_seconds,
            "diagram_distance_comparisons_seconds": distance_seconds,
        }
        generated_banks[role] = prediction_bank
    complete_seconds = time.perf_counter() - summary_started
    pairing_seconds = gt_seconds + sum(item["prediction_ph_seconds"] for item in roles.values())
    payload = {
        "reference_descriptor_scaling": "detached GT mean/std per sample and descriptor field; std clamped to configured scale_floor",
        "finite_essential_formula": "finite distance is normalized by H*W; essential birth L1 is raw and multiplied by essential_weight without H*W normalization",
        "line_bank": objective.sampling_artifact(),
        "dimensions": list(objective.dimensions),
        "filtration_rows": list(labels),
        "metric_rows": list(metric_labels),
        "diagnostic_snapshot_ids": list(sample_ids[:count]),
        "ground_truth_filtration_seconds_once": reference_filtration_seconds,
        "ground_truth_ph_seconds_once": gt_seconds,
        "pairing_ph_seconds_total": pairing_seconds,
        "complete_bar_summary_seconds": complete_seconds,
        "roles": roles,
        "extra_ph_note": "Separate bounded bar-summary pairing pass; no persistence-cache reuse is claimed.",
    }
    return payload, complete_seconds


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_idempotent_report(
    store: Any, relative_path: str, payload: Mapping[str, Any]
) -> tuple[Path, bool]:
    """Atomically write once; a same-step retry must match the exact identity."""
    path = store.run_dir / relative_path
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise TypeError("native topology audit report needs an identity mapping")
    identity_sha256 = _canonical_sha256(identity)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("identity_sha256") != identity_sha256:
            raise ValueError(f"native topology audit provenance changed for {relative_path}")
        return path, False
    record = {**dict(payload), "identity_sha256": identity_sha256}
    store.write_json(relative_path, record)
    written = json.loads(path.read_text(encoding="utf-8"))
    if written.get("identity_sha256") != identity_sha256:
        raise RuntimeError("native topology audit atomic output failed identity verification")
    return path, True


def verify_saved_family_hash(store: Any) -> str:
    path = store.run_dir / "artifacts/topology_family.pt"
    manifest_path = store.run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("coherence_family_state_sha256s", {}).get("topology")
    actual = file_sha256(path) if path.is_file() else None
    if not expected or actual != expected:
        raise ValueError("native topology audit child family artifact hash mismatch")
    return actual


def contract_payload(
    *,
    config: Mapping[str, Any],
    dataset: Any,
    batch: ObservationBatch,
    batch_metadata: Mapping[str, Any],
    sensor_manifest: Any,
    family_provenance: Mapping[str, Any],
    config_sha256: str,
    selector_manifest_sha256: str,
) -> dict[str, Any]:
    query_indices = batch.metadata["query_indices"].detach().to(device="cpu").contiguous()
    query_coords = batch.query_coords.detach().to(device="cpu").contiguous()
    observations = []
    for index, sample_id in enumerate(batch.sample_ids):
        mask = batch.obs_valid_mask[index].detach().to(device="cpu")
        observations.append(
            {
                "sample_id": sample_id,
                "point_field_indices": torch.stack(
                    (batch.obs_indices[index, mask].cpu(), batch.obs_field_ids[index, mask].cpu()),
                    dim=1,
                ).tolist(),
            }
        )
    selector = config["evaluation"]["native_topology_audit"]
    identity = {
        "config_sha256": config_sha256,
        "dataset_fingerprint": dataset_fingerprint(dataset.path),
        "evaluation_split": config["evaluation"].get("split", "validation"),
        "selector_manifest_sha256": selector_manifest_sha256,
        "sample_ids": list(batch.sample_ids),
        "sample_indices": list(batch_metadata["dataset_indices"]),
        "sensor_protocol": batch_metadata["protocol"],
        "audit_seed": int(selector.get("seed", 2027)),
        "query_indices_sha256": hashlib.sha256(query_indices.numpy().tobytes()).hexdigest(),
        "query_coordinates_sha256": hashlib.sha256(query_coords.numpy().tobytes()).hexdigest(),
        "family_sampling_artifact": family_provenance["sampling_artifact"],
        "source_checkpoint_sha256": family_provenance.get("source_checkpoint_sha256"),
    }
    return {
        "identity": identity,
        "split": "validation",
        "selector_sample_ids": list(batch_metadata["selector_sample_ids"]),
        "sample_ids": list(batch.sample_ids),
        "dataset_indices": list(batch_metadata["dataset_indices"]),
        "sensor_protocol": batch_metadata["protocol"],
        "sensor_manifest_sha256": sensor_manifest.digest(),
        "observation_point_field_indices": observations,
        "query_indices_sha256": identity["query_indices_sha256"],
        "query_coordinates_sha256": identity["query_coordinates_sha256"],
        "query_count_per_sample": int(batch.query_coords.shape[1]),
        "native_grid_shape": list(config["dataset"]["grid_shape"]),
        "family_provenance": dict(family_provenance),
        "audit_sampling": {
            key: selector.get(key) for key in ("every_steps", "max_samples", "seed")
        },
    }


def identity_digest(identity: Mapping[str, Any]) -> str:
    return _canonical_sha256(identity)


def load_existing_report(store: Any, relative_path: str, identity: Mapping[str, Any]):
    """Return a matching durable record, or reject a conflicting same-step record."""
    path = store.run_dir / relative_path
    if not path.is_file():
        return None
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing.get("identity_sha256") != identity_digest(identity):
        raise ValueError(f"native topology audit provenance changed for {relative_path}")
    return existing


def _report_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    topology = report.get("coherence", {}).get("families", {}).get("topology", {})
    keys = (
        "mse_normalized",
        "mse_physical",
        "mean_relative_l2",
        "mean_relative_l2_physical",
        "worst_field_relative_l2",
        "per_field_mse_normalized",
        "per_field_mse_physical",
        "per_field_relative_l2",
        "per_field_relative_l2_physical",
        "per_field_unobserved_mse_normalized",
        "observed_entry_mse_normalized",
        "unobserved_entry_mse_normalized",
    )
    return {
        "metrics": {key: report.get(key) for key in keys if key in report},
        "topology": topology,
        "native_topology": report.get("native_topology"),
    }


def _numeric_delta(candidate: Any, source: Any):
    if isinstance(candidate, Mapping) and isinstance(source, Mapping):
        return {
            str(key): _numeric_delta(value, source[key])
            for key, value in candidate.items()
            if key in source
        }
    if isinstance(candidate, (int, float)) and isinstance(source, (int, float)):
        return float(candidate) - float(source)
    return None


def execute_native_audit(
    *,
    store: Any,
    config: Mapping[str, Any],
    current_model: torch.nn.Module,
    source_model: torch.nn.Module,
    native_family: torch.nn.Module,
    batch: ObservationBatch,
    field_names: tuple[str, ...],
    normalizer: Any,
    family_scales: Mapping[str, float],
    evaluate_fn: Callable[..., dict[str, Any]],
    contract: Mapping[str, Any],
    family_provenance: Mapping[str, Any],
    geometry_provenance: Mapping[str, Any],
    committed_step: int,
    attempt_step: int,
    durable_references: list[Mapping[str, Any]],
    terminal: bool = False,
) -> tuple[Path, bool, dict[str, Any] | None]:
    """Run paired LIVE evaluations once per committed state and persist evidence."""
    audit_started = time.perf_counter()
    artifact_sha256 = verify_saved_family_hash(store)
    objective = native_family.spatial_objective
    source_state_sha256 = _tensor_state_sha256(source_model)
    current_state_sha256 = _tensor_state_sha256(current_model)
    contract_identity_sha256 = identity_digest(contract["identity"])
    identity = {
        "schema_version": 1,
        "committed_step": int(committed_step),
        "contract_identity_sha256": contract_identity_sha256,
        "source_checkpoint_sha256": family_provenance.get("source_checkpoint_sha256"),
        "source_live_state_sha256": source_state_sha256,
        "candidate_live_state_sha256": current_state_sha256,
        "family_sampling_artifact": objective.sampling_artifact(),
        "family_config": native_family.config,
        "native_geometry": dict(geometry_provenance),
        "evaluation_seed": int(config.get("evaluation", {}).get("seed", 2027)),
        "generation_steps": int(config.get("evaluation", {}).get("generation_steps", 2)),
    }
    relative_path = f"{AUDIT_SUBDIR}/step_{int(committed_step):09d}.json"
    existing = load_existing_report(store, relative_path, identity)
    if existing is not None:
        terminal_record = None
        if terminal:
            terminal_relative = f"{AUDIT_SUBDIR}/terminal_attempt_{int(attempt_step):09d}.json"
            terminal_record = {
                "identity": {
                    "committed_step": int(committed_step),
                    "attempt_step": int(attempt_step),
                    "step_report_identity_sha256": identity_digest(identity),
                    "durable_references": [dict(item) for item in durable_references],
                },
                "committed_step": int(committed_step),
                "attempt_step": int(attempt_step),
                "step_report": str((store.run_dir / relative_path).relative_to(store.run_dir)),
                "durable_references": [dict(item) for item in durable_references],
            }
            write_idempotent_report(store, terminal_relative, terminal_record)
        return store.run_dir / relative_path, False, terminal_record

    evaluation_seconds = {}
    captures = {}
    audit_eval_config = deepcopy(dict(config))
    audit_eval_config["evaluation"] = dict(audit_eval_config.get("evaluation", {}))
    # The explicitly supplied full-grid family is the audit path. Keep the
    # unrelated legacy sparse-native diagnostic from building a second family.
    audit_eval_config["evaluation"]["native_topology"] = False
    with _preserve_audit_state((source_model, current_model)):
        source_report, captures["source_live"], evaluation_seconds["source_live"] = (
            _evaluate_pair_member(
                source_model,
                native_family,
                batch,
                audit_eval_config,
                field_names,
                normalizer,
                family_scales,
                evaluate_fn,
            )
        )
        candidate_report, captures["candidate_live"], evaluation_seconds["candidate_live"] = (
            _evaluate_pair_member(
                current_model,
                native_family,
                batch,
                audit_eval_config,
                field_names,
                normalizer,
                family_scales,
                evaluate_fn,
            )
        )
        bars, bar_seconds = paired_bar_diagnostics(
            native_family,
            captures["source_live"],
            captures["candidate_live"],
            tuple(batch.sample_ids),
            max_snapshots=2,
        )
    if (
        _tensor_state_sha256(source_model) != source_state_sha256
        or _tensor_state_sha256(current_model) != current_state_sha256
    ):
        raise RuntimeError("native topology audit changed live source or candidate weights")

    full_lines = tuple(range(objective.line_bank_size))
    expected_lines = [list(full_lines) for _ in objective.groups]
    for role, report in (
        ("source_live", source_report.get("coherence", {}).get("families", {}).get("topology", {})),
        (
            "candidate_live",
            candidate_report.get("coherence", {}).get("families", {}).get("topology", {}),
        ),
    ):
        observed = report.get("diagnostics", {}).get("line_sampling", {}).get("selected_indices")
        if observed != expected_lines:
            raise RuntimeError(f"{role} native evaluation did not use the complete saved line bank")
    generated_field_sha256 = {
        role: hashlib.sha256(
            capture["generated_grid"]
            .detach()
            .to(device="cpu")
            .contiguous()
            .view(torch.uint8)
            .numpy()
            .tobytes()
        ).hexdigest()
        for role, capture in captures.items()
    }
    payload = {
        "identity": identity,
        "committed_step": int(committed_step),
        "attempt_step": int(attempt_step),
        "terminal_evaluation": bool(terminal),
        "durable_references": [dict(item) for item in durable_references],
        "source_checkpoint_sha256": family_provenance.get("source_checkpoint_sha256"),
        "source_live_state_sha256": source_state_sha256,
        "candidate_live_state_sha256": current_state_sha256,
        "child_family_artifact_sha256": artifact_sha256,
        "family_provenance": dict(family_provenance),
        "native_geometry": dict(geometry_provenance),
        "selector_contract_sha256": contract_identity_sha256,
        "split": "validation",
        "selector_sample_ids": contract.get("selector_sample_ids", []),
        "sample_ids": list(batch.sample_ids),
        "query_count_per_sample": int(batch.query_coords.shape[1]),
        "native_grid_shape": list(native_family.grid_shape),
        "generation": {
            "seed": int(config.get("evaluation", {}).get("seed", 2027)),
            "steps": int(config.get("evaluation", {}).get("generation_steps", 2)),
            "sensor_protocol": contract.get("sensor_protocol"),
            "sensor_manifest_sha256": contract.get("sensor_manifest_sha256"),
            "source_and_candidate_share_batch": True,
            "source_and_candidate_share_local_generation_seed": True,
            "weight_selection": "live; EMA evaluation swap disabled and restored",
        },
        "full_bank_evaluation": {
            "line_bank": objective.sampling_artifact(),
            "selected_line_indices": expected_lines,
            "subset_estimator": "not_applicable_full_bank",
        },
        "source_to_reference": _report_metrics(source_report),
        "candidate_to_reference": _report_metrics(candidate_report),
        "paired_metric_differences_candidate_minus_source": {
            "metrics": _numeric_delta(
                _report_metrics(candidate_report)["metrics"],
                _report_metrics(source_report)["metrics"],
            ),
            "topology_component_scalars": _numeric_delta(
                _report_metrics(candidate_report)["topology"].get("component_scalars", {}),
                _report_metrics(source_report)["topology"].get("component_scalars", {}),
            ),
        },
        "native_generated_field_sha256": generated_field_sha256,
        "bar_diagnostics": bars,
        "cost_seconds": {
            "source_evaluation_including_full_bank_native_ph": evaluation_seconds["source_live"],
            "candidate_evaluation_including_full_bank_native_ph": evaluation_seconds[
                "candidate_live"
            ],
            "bounded_bar_summary_complete_including_filtration_and_comparisons": bar_seconds,
            "bounded_bar_summary_ground_truth_ph_pairing": bars["ground_truth_ph_seconds_once"],
            "bounded_bar_summary_generated_ph_pairing_total": sum(
                item["prediction_ph_seconds"] for item in bars["roles"].values()
            ),
            "bounded_bar_summary_diagram_distance_comparisons": sum(
                item["diagram_distance_comparisons_seconds"] for item in bars["roles"].values()
            ),
        },
        "timing_scope": "Complete audit wall includes saved-bank verification, source/candidate live-state and identity hashing, both LIVE evaluations with full-bank native PH, bounded separate GT-relative bar summary, and diagram-distance comparisons. It excludes native-family/batch setup before execute_native_audit and the final RunStore JSON serialization/write. No CUDA-peak reset is performed, so process peak memory remains run-wide since the trainer reset.",
        "cuda_memory_bytes": (
            {
                "currently_allocated": int(torch.cuda.memory_allocated(batch.query_coords.device)),
                "process_peak_since_last_reset": int(
                    torch.cuda.max_memory_allocated(batch.query_coords.device)
                ),
            }
            if batch.query_coords.device.type == "cuda"
            else None
        ),
    }
    payload["cost_seconds"]["complete_audit_wall_before_json_write"] = (
        time.perf_counter() - audit_started
    )
    path, written = write_idempotent_report(store, relative_path, payload)
    terminal_record = None
    if terminal:
        terminal_relative = f"{AUDIT_SUBDIR}/terminal_attempt_{int(attempt_step):09d}.json"
        terminal_record = {
            "identity": {
                "committed_step": int(committed_step),
                "attempt_step": int(attempt_step),
                "step_report_identity_sha256": identity_digest(identity),
                "durable_references": [dict(item) for item in durable_references],
            },
            "committed_step": int(committed_step),
            "attempt_step": int(attempt_step),
            "step_report": str(path.relative_to(store.run_dir)),
            "durable_references": [dict(item) for item in durable_references],
        }
        write_idempotent_report(store, terminal_relative, terminal_record)
    return path, written, terminal_record
