"""Small, prospectively frozen panel and reporting helpers for the R2 study.

No helper fits descriptors, changes model parameters, or reads the test split.
The training evaluator remains the single endpoint/family implementation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import fields
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..contracts import ObservationBatch
from ..data.sensor_protocols import build_observation_batch


def selection_indices(
    dataset_size: int,
    count: int = 64,
    strata: int = 16,
    excluded_indices: Sequence[int] = (),
    *,
    stratify_available: bool = False,
) -> list[int]:
    """Equal counts per chronological stratum, spread across its available frames."""
    if dataset_size < strata or count < strata or count % strata:
        raise ValueError("panel requires a positive integral count per stratum")
    excluded = set(map(int, excluded_indices))
    if any(index < 0 or index >= dataset_size for index in excluded):
        raise ValueError("excluded index is outside the split")
    candidates = [index for index in range(dataset_size) if index not in excluded]
    boundaries = np.linspace(
        0, len(candidates) if stratify_available else dataset_size, strata + 1, dtype=int
    )
    result: list[int] = []
    per_stratum = count // strata
    for start, stop in pairwise(boundaries):
        available = (
            candidates[int(start) : int(stop)]
            if stratify_available
            else [index for index in range(int(start), int(stop)) if index not in excluded]
        )
        if len(available) < per_stratum:
            raise ValueError("insufficient fresh snapshots in a chronological stratum")
        positions = np.linspace(0, len(available) - 1, per_stratum).round().astype(int)
        result.extend(available[int(position)] for position in positions)
    return result


def declare_panels(dataset_size: int, *, previously_used_indices: Sequence[int] = ()) -> dict:
    """Selection64, nested extended128, and disjoint fresh audit64; test stays locked."""
    selection = selection_indices(dataset_size)
    # Add four distinct points per stratum; preserve all selection points.
    extra = selection_indices(dataset_size, excluded_indices=selection)
    extended = sorted(selection + extra)
    audit = selection_indices(
        dataset_size,
        excluded_indices=tuple(extended) + tuple(previously_used_indices),
        stratify_available=True,
    )

    def panel(indices, sensor_offset, noise_seed):
        return {
            "dataset_indices": indices,
            "groups": [indices[start : start + 32] for start in range(0, len(indices), 32)],
            "strata": 16,
            "equal_sample_weights": True,
            "sensor_seed_offset": sensor_offset,
            "generation_seed": noise_seed,
            "native_seeds": [2027, 3027] if not sensor_offset else [4027, 5027],
            "weight_selection": "live",
            "topology_bank": "full_16",
        }

    declaration = {
        "schema": "phycoflow.upgrade_1002_r2_panels.v1",
        "split": "validation",
        "dataset_size": dataset_size,
        "test_locked": True,
        "previously_used_validation_indices": sorted(set(map(int, previously_used_indices))),
        "selection": panel(selection, 0, 2027),
        "extended": panel(extended, 0, 2027),
        "audit": panel(audit, 1000, 4027),
    }
    declaration["audit"]["stratification"] = "16_chronological_strata_of_remaining_fresh_snapshots"
    return declaration


def build_panel_batch(dataset, protocol, indices: Sequence[int], device) -> ObservationBatch:
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(dataset) for i in indices):
        raise ValueError("panel indices must be unique and within the split")
    return build_observation_batch(
        [dataset[int(index)] for index in indices], protocol, query_points=None
    ).to(device)


def slice_panel(batch: ObservationBatch, start: int, stop: int) -> ObservationBatch:
    """Slice sample axes including nested adapter contexts, without slicing fixed geometry."""
    size = len(batch.sample_ids)
    if not 0 <= start < stop <= size:
        raise ValueError("invalid panel slice")

    def sliced(value):
        if isinstance(value, torch.Tensor):
            return value[start:stop] if value.ndim and value.shape[0] == size else value
        if isinstance(value, dict):
            return {key: sliced(item) for key, item in value.items()}
        return value

    values = {field.name: getattr(batch, field.name) for field in fields(batch)}
    for name, value in values.items():
        if name in {"sample_ids", "logical_shapes"}:
            values[name] = value[start:stop]
        else:
            values[name] = sliced(value)
    return ObservationBatch(**values)


def matched_native_monitor(
    model,
    source_model,
    batches: Mapping[str, ObservationBatch],
    seeds: Sequence[int] = (2027, 3027),
) -> dict:
    """Frozen common-random native observations; preserve public RNG, modes and weights."""
    from ..training.fidelity_controller import matched_native_losses
    from ..training.native_topology_audit import _preserve_audit_state

    if not seeds:
        raise ValueError("native monitoring needs predeclared random draws")
    reports = {}
    with _preserve_audit_state((model, source_model)):
        model.eval()
        source_model.eval()
        for role, batch in batches.items():
            records = []
            for seed in seeds:
                import random

                random.seed(int(seed))
                np.random.seed(int(seed))
                torch.manual_seed(int(seed))
                if batch.query_coords.device.type == "cuda":
                    torch.cuda.manual_seed_all(int(seed))
                candidate, source = matched_native_losses(
                    model, source_model, batch, live_graph=False
                )
                # The helper may return native loss containers or their total tensors.
                candidate = getattr(candidate, "total", candidate)
                source = getattr(source, "total", source)
                records.append(
                    {"seed": int(seed), "candidate": float(candidate), "source": float(source)}
                )
            candidate_mean = float(np.mean([record["candidate"] for record in records]))
            source_mean = float(np.mean([record["source"] for record in records]))
            reports[role] = {
                "candidate": candidate_mean,
                "source": source_mean,
                "ratio": candidate_mean / source_mean if source_mean > 0 else None,
                "draws": records,
                "sample_ids": list(batch.sample_ids),
                "source_weight_selection": "live",
                "candidate_weight_selection": "live",
                "role": "monitor_only_no_calibration_or_gradient",
            }
    return reports


def grouped_native_monitor(model, source_model, batches, seeds=(2027, 3027)) -> dict:
    """Audit-only native draws in exact groups32, without altering training monitors."""
    reports = {}
    for role, batch in batches.items():
        count = len(batch.sample_ids)
        if count < 32 or count % 32:
            raise ValueError("grouped native audit requires complete groups of 32")
        groups = []
        for start in range(0, count, 32):
            group = matched_native_monitor(
                model, source_model, {role: slice_panel(batch, start, start + 32)}, seeds
            )[role]
            groups.append({"group_index": start // 32, **group})
        draws = []
        for index, seed in enumerate(seeds):
            candidate = float(np.mean([group["draws"][index]["candidate"] for group in groups]))
            source = float(np.mean([group["draws"][index]["source"] for group in groups]))
            draws.append(
                {
                    "seed": int(seed),
                    "candidate": candidate,
                    "source": source,
                    "ratio": candidate / source if source > 0 else None,
                }
            )
        candidate = float(np.mean([draw["candidate"] for draw in draws]))
        source = float(np.mean([draw["source"] for draw in draws]))
        reports[role] = {
            "candidate": candidate,
            "source": source,
            "ratio": candidate / source if source > 0 else None,
            "draws": draws,
            "groups": groups,
            "sample_ids": list(batch.sample_ids),
            "group_size": 32,
            "aggregation": "equal_group_mean_then_equal_draw_mean",
            "rng_policy": "reset_each_declared_seed_for_each_group_source_candidate_matched",
            "source_weight_selection": "live",
            "candidate_weight_selection": "live",
            "role": "audit_only_no_calibration_or_gradient",
        }
    return reports


def error_decomposition(prediction, target, valid_mask=None) -> dict:
    """Per-snapshot mean-offset and centered MSE; retain original production error."""
    error = (prediction.detach() - target.detach()).double()
    valid = (
        torch.ones_like(error, dtype=torch.bool)
        if valid_mask is None
        else valid_mask[..., None].expand_as(error)
    )
    counts = valid.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("error decomposition requires valid points in every field")
    mean_error = (error * valid).sum(dim=1) / counts
    mse = (error.square() * valid).sum(dim=1) / counts
    offset = mean_error.square()
    centered = ((error - mean_error[:, None]).square() * valid).sum(dim=1) / counts
    return {
        "per_sample_mean_error": mean_error.cpu().tolist(),
        "per_sample_mse": mse.cpu().tolist(),
        "per_sample_mean_offset_squared": offset.cpu().tolist(),
        "per_sample_centered_mse": centered.cpu().tolist(),
        "mse": mse.mean(dim=0).cpu().tolist(),
        "mean_offset_squared": offset.mean(dim=0).cpu().tolist(),
        "centered_mse": centered.mean(dim=0).cpu().tolist(),
        "identity_max_abs_error": float((mse - offset - centered).abs().max()),
        "production_metric_gauge_subtracted": False,
    }


def _average_tree(values):
    first = values[0]
    if isinstance(first, Mapping):
        return {key: _average_tree([value[key] for value in values]) for key in first}
    if isinstance(first, (float, int)) and not isinstance(first, bool):
        return float(np.mean(values))
    return first


def grouped_evaluate(
    model,
    complete_batch,
    comparison_batch,
    families,
    banks,
    field_names,
    config,
    *,
    normalizer=None,
    family_scales=None,
    group_size=32,
    endpoint_callback=None,
) -> dict:
    """Reuse the trainer evaluator in exact frozen groups; separately pool B statistics."""
    from ..training.native_topology_audit import _preserve_audit_state

    with _preserve_audit_state((model,)):
        return _grouped_evaluate_live(
            model,
            complete_batch,
            comparison_batch,
            families,
            banks,
            field_names,
            config,
            normalizer=normalizer,
            family_scales=family_scales,
            group_size=group_size,
            endpoint_callback=endpoint_callback,
        )


def _grouped_evaluate_live(
    model,
    complete_batch,
    comparison_batch,
    families,
    banks,
    field_names,
    config,
    *,
    normalizer=None,
    family_scales=None,
    group_size=32,
    endpoint_callback=None,
) -> dict:
    from ..coherence.families.cross_spectrum.covariance_blocks import (
        LinearCoefficientCovarianceAccumulator,
        second_order_covariance_block_losses,
    )
    from ..coherence.families.cross_spectrum.statistics import graph_fourier
    from ..training.post_training import _evaluate, _gather_prediction

    count = len(comparison_batch.sample_ids)
    if group_size != 32 or count < 32 or count % group_size:
        raise ValueError("R2 evaluation requires complete declared spectral groups")
    family_map = (
        dict(families) if isinstance(families, Mapping) else {families.family_name: families}
    )
    cross = family_map.get("cross_spectrum")
    coefficients = []
    decompositions = []

    def capture_cross(_module, args, kwargs):
        generated, reference = cross._units_and_fields(args[0], args[1])
        basis = cross._basis(kwargs["coordinates"], generated.dtype)
        coefficients.append(
            (graph_fourier(generated, basis).detach(), graph_fourier(reference, basis).detach())
        )

    class Capture:
        def __getattr__(self, name):
            return getattr(model, name)

        def reconstruct(self, *args, **kwargs):
            result = model.reconstruct(*args, **kwargs)
            self.prediction = result.prediction
            return result

    capture = Capture()
    hook = (
        cross.register_forward_pre_hook(capture_cross, with_kwargs=True)
        if cross is not None and cross.definition == "second_order_blocks_v4"
        else None
    )
    reports = []
    try:
        for start in range(0, count, group_size):
            complete = slice_panel(complete_batch, start, start + group_size)
            comparison = slice_panel(comparison_batch, start, start + group_size)
            reports.append(
                _evaluate(
                    capture,
                    complete,
                    comparison,
                    family_map,
                    banks,
                    field_names,
                    config,
                    normalizer=normalizer,
                    family_scales=family_scales,
                )
            )
            prediction = capture.prediction
            if model.capabilities.structured_grid_required:
                prediction = _gather_prediction(prediction, complete, comparison)
            decompositions.append(
                error_decomposition(
                    prediction, comparison.target_fields, comparison.query_valid_mask
                )
            )
            if endpoint_callback is not None:
                endpoint_callback(prediction.detach(), comparison)
    finally:
        if hook is not None:
            hook.remove()
    report = _average_tree(reports)
    report["sample_ids"] = list(comparison_batch.sample_ids)
    report["query_ids"] = comparison_batch.metadata["query_indices"].detach().cpu().tolist()
    report["inference"]["samples"] = count
    report["inference"]["seconds"] = sum(item["inference"]["seconds"] for item in reports)
    report["panel_groups"] = [item["sample_ids"] for item in reports]
    compact_groups = []
    for item in reports:
        compact = {
            key: value for key, value in item.items() if key not in {"query_ids", "coherence"}
        }
        compact["query_ids_sha256"] = hashlib.sha256(
            np.asarray(item["query_ids"], dtype=np.int64).tobytes()
        ).hexdigest()
        compact["coherence"] = {
            "total": item["coherence"].get("total"),
            "components": item["coherence"].get("components", {}),
            "families": {
                name: {
                    key: value
                    for key, value in payload.items()
                    if key not in {"components", "diagnostics"}
                }
                for name, payload in item["coherence"]["families"].items()
            },
        }
        compact_groups.append(compact)
    report["group_reports"] = compact_groups
    report["aggregation"] = "equal_sample_group32_mean"
    report["evaluation_weight_source"] = "live_checkpoint_model_weights"
    report["generation_seed_policy"] = (
        "reset_declared_seed_per_group32_identically_for_source_and_candidate"
    )
    for name, payload in report["coherence"]["families"].items():
        payload["reference_ids"] = [
            sample
            for item in reports
            for sample in item["coherence"]["families"][name]["reference_ids"]
        ]
        # Components retain the exact group reports; diagnostic lists are not pooled estimators.
        payload["diagnostics"] = {"aggregation": "see_exact_group_reports"}
    if decompositions:
        report["error_decomposition"] = {
            key: np.mean([item[key] for item in decompositions], axis=0).tolist()
            for key in ("mse", "mean_offset_squared", "centered_mse")
        }
    if coefficients:
        generated_stats, reference_stats = (
            LinearCoefficientCovarianceAccumulator(),
            LinearCoefficientCovarianceAccumulator(),
        )
        for generated, reference in coefficients:
            generated_stats.update(generated)
            reference_stats.update(reference)
        generated, reference = (
            torch.cat([pair[index] for pair in coefficients]) for index in (0, 1)
        )
        pooled = second_order_covariance_block_losses(
            generated,
            reference,
            cross.band_ids,
            cross.pairs,
            relative_floor=cross.relative_floor,
            absolute_floor=cross.absolute_floor,
            minimum_reference_band_fraction=cross.minimum_reference_band_fraction,
            energy_floor_policy=cross.energy_floor_policy,
            calibration_reference_energies=cross.calibrated_band_energies,
            calibration_ensemble_size=cross.calibration_ensemble_size,
            generated_covariance=generated_stats.covariance(),
            reference_covariance=reference_stats.covariance(),
        )
        report["pooled_B"] = {
            "estimator": "mergeable_sufficient_statistics",
            "samples": count,
            "same_frequency": float(pooled.same_frequency),
            "cross_frequency": float(pooled.cross_frequency),
            "total": float(
                cross.component_weights["same_frequency"] * pooled.same_frequency
                + cross.component_weights["cross_frequency"] * pooled.cross_frequency
            ),
        }
        report["aggregation"] = "equal_sample_group32_mean_with_separate_pooled_B"
    return report


def mature_checkpoint_summary(records: Sequence[Mapping[str, Any]]) -> dict:
    """Checkpoint evidence by recorded child epoch, preserving source fallback identity."""
    mature = [
        item
        for item in records
        if int(item["epoch"]) >= 100 and not item.get("is_source_baseline", False)
    ]
    eligible = [item for item in mature if item.get("eligible", False)]
    latest = max(records, key=lambda item: int(item["epoch"]), default=None)
    last_epoch = int(latest["epoch"]) if latest else 0
    recent = [item for item in records if last_epoch - 50 < int(item["epoch"]) <= last_epoch]
    best = min(eligible, key=lambda item: float(item["metric"]), default=None)
    center = (
        best if best is not None else max(mature, key=lambda item: int(item["epoch"]), default=None)
    )
    late_mature = [item for item in mature if last_epoch - 50 < int(item["epoch"]) <= last_epoch]
    neighbors = late_mature if center in late_mature else mature
    nearby = (
        sorted(
            neighbors,
            key=lambda item: (abs(int(item["epoch"]) - int(center["epoch"])), -int(item["epoch"])),
        )[:3]
        if center is not None
        else []
    )
    return {
        "best_eligible_mature": best,
        "latest": latest,
        "nearby_mature": nearby,
        "neighbor_center": center,
        "center_role": "best_eligible_mature"
        if best is not None
        else "latest_mature"
        if center is not None
        else None,
        "center_eligible": bool(center.get("eligible", False)) if center is not None else None,
        "eligible_fraction_last_50_epochs": sum(bool(item.get("eligible")) for item in recent)
        / len(recent)
        if recent
        else None,
        "recorded_evaluations_last_50_epochs": len(recent),
        "minimum_mature_epoch": 100,
    }


def save_declaration(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Immutable once published: reruns must match the prospective manifest exactly."""
    path = Path(path)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() and path.read_text() != text:
        raise ValueError("refusing to change a previously declared R2 panel")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def epoch_window_summary(records: Sequence[Mapping[str, Any]], value_key: str) -> dict:
    """Observed epoch windows and robust last-50 slope, with missing evidence explicit."""
    points = sorted(
        (int(record["epoch"]), float(record[value_key]))
        for record in records
        if record.get(value_key) is not None and np.isfinite(float(record[value_key]))
    )
    windows = {}
    for start, stop in ((1, 25), (26, 50), (51, 100), (101, 150), (151, 200)):
        observed = [value for epoch, value in points if start <= epoch <= stop]
        windows[f"epochs_{start}_{stop}"] = {
            "observed": len(observed),
            "median": float(np.median(observed)) if observed else None,
            "minimum": min(observed) if observed else None,
            "maximum": max(observed) if observed else None,
        }
    last_epoch = max((epoch for epoch, _ in points), default=0)
    recent = [(epoch, value) for epoch, value in points if epoch > last_epoch - 50]
    slopes = [
        (right[1] - left[1]) / (right[0] - left[0])
        for i, left in enumerate(recent)
        for right in recent[i + 1 :]
        if right[0] > left[0]
    ]
    return {
        "windows": windows,
        "latest_epoch": last_epoch,
        "robust_slope_last_50_epochs": float(np.median(slopes)) if slopes else None,
        "observations_last_50_epochs": len(recent),
    }


def render_pressure_comparison(
    coordinates_raw, ground_truth, source, candidate, output_path, *, sample_id: str, epoch: int
) -> Path:
    """One PDF with raw-coordinate GT/LIVE fields and matched signed-error scales."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output_path)
    if path.suffix.lower() != ".pdf":
        raise ValueError("R2 scientific figures must be PDF")
    coords = np.asarray(coordinates_raw)
    gt, base, child = map(np.asarray, (ground_truth, source, candidate))
    if (
        coords.ndim != 2
        or coords.shape[1] != 2
        or any(value.shape != (len(coords),) for value in (gt, base, child))
    ):
        raise ValueError("pressure contours require aligned raw [points,2] coordinates")
    values = np.concatenate((gt, base, child))
    low, high = float(values.min()), float(values.max())
    if low == high:
        high = low + max(abs(low) * 1e-8, 1e-8)
    error_limit = max(float(np.abs(base - gt).max()), float(np.abs(child - gt).max()), 1e-12)
    figure, axes = plt.subplots(1, 5, figsize=(15, 3.5), layout="constrained")
    scalar_levels = np.linspace(low, high, 33)
    error_levels = np.linspace(-error_limit, error_limit, 33)
    for axis, value, title in zip(
        axes,
        (gt, base, child, base - gt, child - gt),
        (
            "GT p",
            "Source LIVE p",
            f"Candidate LIVE p, epoch {epoch}",
            "Source − GT",
            "Candidate − GT",
        ),
    ):
        signed = "GT" in title and title != "GT p"
        levels = error_levels if signed else scalar_levels
        contour = axis.tricontourf(
            coords[:, 0], coords[:, 1], value, levels=levels, cmap="RdBu_r" if signed else "viridis"
        )
        axis.set_aspect("equal")
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("raw x (units unspecified)", fontsize=8)
        axis.set_ylabel("raw y (units unspecified)", fontsize=8)
        figure.colorbar(contour, ax=axis, shrink=0.75)
    figure.suptitle(
        f"Pressure and signed errors · {sample_id} · decoded dataset values; units unspecified",
        fontsize=10,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path
