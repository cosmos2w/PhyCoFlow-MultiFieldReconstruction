"""Streaming whole-split reconstruction evaluation and distribution visualization."""

from __future__ import annotations

import csv
import json
import re
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from ..config import load_config
from ..data.manifest import SensorManifest, build_batch_from_manifest, dataset_fingerprint
from ..data.sensor_protocols import build_observation_batch
from ..training.common import sensor_protocol_from_config
from ..training.model_lifecycle import selected_evaluation_weight_context
from ..training.run_store import file_sha256
from ..training.source import source_checkpoint_path
from .checkpoint import load_evaluation_runtime
from .coherence_set import (
    adaptive_coherence_score_limits,
    build_coherence_accumulators,
    render_coherence_distribution,
    render_cross_spectrum_band_profiles,
    render_cross_spectrum_score_bars,
)
from .reconstruction_visualization import warn_if_cuda_memory_tight
from .topology_set import (
    render_configured_grid_topology,
    render_topology_betti_curves,
    render_topology_distance_distributions,
)


@dataclass(frozen=True)
class _SetEvaluationResult:
    run_dir: Path
    output_dir: Path
    figure_path: Path
    payload_path: Path
    report_path: Path
    manifest_path: Path
    checkpoint_path: Path
    checkpoint_label: str
    run_label: str
    split: str
    sample_ids: tuple[str, ...]
    field_names: tuple[str, ...]
    dataset_fingerprint: str
    generation_steps: int
    evaluation_seed: int
    coherence_outputs: dict[str, Any]
    coherence_accumulators: dict[str, Any]


def _finite_summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {
            "count": 0,
            "mean": None,
            "standard_deviation": None,
            "minimum": None,
            "quartile_25": None,
            "median": None,
            "quartile_75": None,
            "maximum": None,
        }
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "standard_deviation": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "minimum": float(finite.min()),
        "quartile_25": float(np.quantile(finite, 0.25)),
        "median": float(np.median(finite)),
        "quartile_75": float(np.quantile(finite, 0.75)),
        "maximum": float(finite.max()),
    }


def _physical_relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    normalizer,
) -> torch.Tensor:
    prediction_physical = normalizer.decode(prediction)
    target_physical = normalizer.decode(target)
    error_sq = (prediction_physical - target_physical).square().sum(dim=1)
    target_sq = target_physical.square().sum(dim=1)
    eps = torch.finfo(prediction.dtype).eps
    relative = torch.sqrt(error_sq / target_sq.clamp_min(eps))
    return torch.where(target_sq > eps, relative, torch.full_like(relative, torch.nan))


def _select_sample_indices(dataset_size: int, max_samples: int | None) -> np.ndarray:
    """Select a deterministic split-wide subset, or every sample when uncapped."""
    if dataset_size < 1:
        raise ValueError("dataset_size must be positive")
    if max_samples is None or max_samples >= dataset_size:
        return np.arange(dataset_size, dtype=np.int64)
    if max_samples < 1:
        raise ValueError("max_samples must be positive or None for the full split")
    return np.linspace(0, dataset_size - 1, num=max_samples, dtype=np.int64)


def _distribution_limits(values: np.ndarray, scale: str) -> tuple[float, float]:
    """Return stable limits suitable for one or several matched distributions."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if scale == "log":
        finite = finite[finite > 0.0]
        if not finite.size:
            raise ValueError("log-scale statistical plot contains no positive values")
        return float(finite.min()) * 0.75, float(finite.max()) * 1.25
    if scale != "linear":
        raise ValueError("statistical plot scale must be 'log' or 'linear'")
    upper = float(finite.max()) * 1.08 if finite.size else 1.0
    return 0.0, max(upper, np.finfo(np.float64).eps)


def _relative_l2_distribution_limits(values: np.ndarray, scale: str) -> tuple[float, float]:
    """Limits for nonnegative relative errors, including an all-zero log case."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if np.any(finite < 0.0):
        raise ValueError("relative L2 errors must be nonnegative")
    if scale == "log" and finite.size and not np.any(finite > 0.0):
        return 1.0e-4, 1.0
    return _distribution_limits(finite, scale)


def _merge_coherence_evaluation_override(
    base_config: dict[str, Any],
    current_coherence: dict[str, Any] | None,
    requested_families: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Apply only the current run's requested coherence evaluation contracts.

    The source model keeps its own observations, runtime, and other resolved
    configuration. Family settings and the small shared compute-budget subset
    needed for identical graph/query construction are aligned for comparison.
    """
    if not isinstance(current_coherence, dict):
        current_coherence = {}
    existing = base_config.get("coherence")
    merged = deepcopy(dict(existing)) if isinstance(existing, dict) else {}
    current_families = current_coherence.get("families", {})
    merged_families = deepcopy(merged.get("families", {}))
    if not isinstance(merged_families, dict):
        merged_families = {}
    for name in requested_families:
        family_settings = (
            current_families.get(name) if isinstance(current_families, dict) else None
        )
        if isinstance(family_settings, dict):
            merged_families[name] = deepcopy(family_settings)
        else:
            # Reproduce the current run's default family settings when it did
            # not resolve an explicit block; retaining a source-only override
            # would make the two figures describe different contracts.
            merged_families.pop(name, None)
    if merged_families:
        merged["families"] = merged_families

    compute_fields: set[str] = set()
    if "topology" in requested_families:
        compute_fields.update(("query_policy", "point_count", "query_seed"))
    if "cross_spectrum" in requested_families:
        compute_fields.update(("batch_size", "point_count", "query_seed"))
    if compute_fields:
        current_budget = current_coherence.get("compute_budget", {})
        merged_budget = deepcopy(merged.get("compute_budget", {}))
        if not isinstance(merged_budget, dict):
            merged_budget = {}
        defaults = {
            "query_policy": "fixed_shared",
            "point_count": 4096,
            "query_seed": 100045,
            "batch_size": 16,
        }
        for key in compute_fields:
            merged_budget[key] = deepcopy(
                current_budget.get(key, defaults[key])
                if isinstance(current_budget, dict)
                else defaults[key]
            )
        merged["compute_budget"] = merged_budget
    result = deepcopy(base_config)
    result["coherence"] = merged
    return result


def _publication_field_label(name: str) -> str:
    indexed = re.fullmatch(r"([A-Za-z]+)_([0-9]+)", name)
    if indexed:
        return rf"${indexed.group(1)}_{{{indexed.group(2)}}}$"
    trailing_index = re.fullmatch(r"(.+?)([0-9]+)", name)
    if trailing_index:
        return rf"{trailing_index.group(1)}$_{{{trailing_index.group(2)}}}$"
    return name


def _trace_path(path: Path, run_dir: Path) -> str:
    """Prefer compact run-relative artifact paths without rejecting external outputs."""
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


def render_reconstruction_set_distribution(
    payload_path: str | Path,
    output_path: str | Path,
    *,
    title: str | None = None,
    dpi: int = 300,
    scale: str = "log",
    value_limits: tuple[float, float] | None = None,
) -> Path:
    """Render per-field violin distributions overlaid with per-sample scatter points."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    payload_path = Path(payload_path)
    output_path = Path(output_path)
    with np.load(payload_path, allow_pickle=False) as payload:
        errors = np.asarray(payload["per_field_relative_l2_physical"], dtype=np.float64)
        field_names = tuple(str(value) for value in payload["field_names"])
        split = str(payload["split"])

    if errors.ndim != 2 or errors.shape[1] != len(field_names):
        raise ValueError("set-evaluation payload must have shape [samples, fields]")
    if errors.shape[0] < 1:
        raise ValueError("set-evaluation payload contains no samples")
    if scale not in {"log", "linear"}:
        raise ValueError("statistical plot scale must be 'log' or 'linear'")

    if np.any(errors[np.isfinite(errors)] < 0.0):
        raise ValueError("relative L2 errors must be nonnegative")

    figure_width = max(7.2, 1.05 * len(field_names) + 2.1)
    figure, axis = plt.subplots(figsize=(figure_width, 4.25), layout="constrained")
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    positions = np.arange(1, len(field_names) + 1, dtype=np.float64)
    rng = np.random.default_rng(2027)
    color = "#2878A5"
    finite_by_field = []
    finite_positions = []
    zero_counts = []
    for field_index in range(len(field_names)):
        field_values = errors[:, field_index]
        finite = field_values[np.isfinite(field_values)]
        zero_count = int(np.count_nonzero(finite == 0.0))
        zero_counts.append(zero_count)
        if scale == "log":
            plot_values = finite[finite > 0.0]
        else:
            plot_values = finite
        if plot_values.size >= 2:
            density_values = np.log10(plot_values) if scale == "log" else plot_values
            finite_by_field.append(density_values)
            finite_positions.append(positions[field_index])
        jitter = rng.uniform(-0.105, 0.105, size=plot_values.size)
        axis.scatter(
            np.full(plot_values.size, positions[field_index]) + jitter,
            plot_values,
            s=12,
            alpha=0.42,
            color=color,
            edgecolors="none",
            zorder=3,
            rasterized=True,
        )
        if not finite.size:
            axis.text(
                positions[field_index],
                0.5,
                "N/A",
                ha="center",
                va="center",
                transform=axis.get_xaxis_transform(),
            )

    if finite_by_field:
        violin = axis.violinplot(
            finite_by_field,
            positions=finite_positions,
            widths=0.64,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in violin["bodies"]:
            if scale == "log":
                for path in body.get_paths():
                    path.vertices[:, 1] = np.power(10.0, path.vertices[:, 1])
            body.set_facecolor("#9CC7DA")
            body.set_edgecolor(color)
            body.set_alpha(0.48)
            body.set_linewidth(0.9)

    limits = value_limits or _relative_l2_distribution_limits(errors, scale)
    if scale == "log" and limits[0] <= 0.0:
        raise ValueError("log-scale reconstruction limits must be positive")
    if not np.isfinite(limits).all() or limits[0] >= limits[1]:
        raise ValueError("reconstruction plot limits must be finite and increasing")
    axis.set_ylim(*limits)

    # Show robust summaries directly, without making the violin shape carry
    # the only location/spread information.
    for field_index, position in enumerate(positions):
        finite = errors[:, field_index]
        finite = finite[np.isfinite(finite)]
        if scale == "log":
            finite = finite[finite > 0.0]
        if finite.size:
            q25, median, q75 = np.quantile(finite, (0.25, 0.5, 0.75))
            axis.vlines(position, q25, q75, color="#1D2933", linewidth=1.25, zorder=4)
            axis.scatter(
                position,
                median,
                marker="D",
                s=28,
                linewidths=0.8,
                facecolors="white",
                edgecolors="#1D2933",
                zorder=5,
            )
        if scale == "log" and zero_counts[field_index]:
            axis.scatter(
                position + rng.uniform(-0.08, 0.08, size=zero_counts[field_index]),
                np.full(zero_counts[field_index], limits[0] * 1.06),
                marker="v",
                s=23,
                color="#1D2933",
                edgecolors="white",
                linewidths=0.35,
                zorder=5,
                clip_on=False,
            )

    axis.set_xticks(
        positions,
        tuple(
            f"{_publication_field_label(name)}\n(n={count})"
            for name, count in zip(field_names, np.isfinite(errors).sum(axis=0))
        ),
        fontsize=9.5,
    )
    axis.set_xlim(0.35, len(field_names) + 0.65)
    axis.set_yscale(scale)
    axis.set_ylim(*limits)
    axis.set_ylabel(
        r"Per-sample relative $L_2$ error" "\n(dimensionless; decoded physical fields)",
        fontsize=10.0,
        labelpad=7.0,
    )
    axis.set_title(
        title or f"Per-field reconstruction error — {split} set",
        fontsize=10.8,
        fontweight="medium",
        pad=8.0,
    )
    axis.tick_params(axis="y", labelsize=9.5, width=0.8, length=4)
    axis.tick_params(axis="x", width=0.8, length=4, pad=5)
    axis.grid(axis="y", color="0.88", linewidth=0.6, linestyle="-", zorder=0)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.legend(
        handles=(
            Patch(
                facecolor="#9CC7DA",
                edgecolor=color,
                alpha=0.48,
                label="Density in log space" if scale == "log" else "Density",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=4.0,
                markerfacecolor=color,
                markeredgecolor=color,
                label="Sample",
            ),
            Line2D(
                [],
                [],
                marker="D",
                linestyle="none",
                markersize=4.5,
                markerfacecolor="white",
                markeredgecolor="#1D2933",
                markeredgewidth=0.8,
                label="Median; line = IQR",
            ),
            *(
                (
                    Line2D(
                        [],
                        [],
                        marker="v",
                        linestyle="none",
                        markersize=5,
                        markerfacecolor="#1D2933",
                        markeredgecolor="white",
                        markeredgewidth=0.3,
                        label="Zero at plot floor",
                    ),
                )
                if any(zero_counts) and scale == "log"
                else ()
            ),
        ),
        loc="upper left",
        ncol=3,
        frameon=False,
        fontsize=8.0,
        handlelength=1.2,
        columnspacing=1.1,
        borderaxespad=0.5,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    for vector_suffix in (".pdf", ".svg"):
        figure.savefig(output_path.with_suffix(vector_suffix), bbox_inches="tight")
    plt.close(figure)
    return output_path


def _evaluate_reconstruction_set_once(
    run_dir: str | Path,
    *,
    case_dir: str | Path,
    split: str,
    checkpoint: str = "best",
    sensor_config: str | Path | None = None,
    sensor_manifest: str | Path | None = None,
    generation_steps: int | None = None,
    device_name: str | None = None,
    output_path: str | Path | None = None,
    weight_selection: str = "configured",
    max_samples: int | None = 200,
    coherence_families: tuple[str, ...] | list[str] | None = None,
    extra_coherence_views: bool = False,
    cross_spectrum_aggregation: str = "training_aligned",
    statistic_scale: str = "log",
    output_dir_override: str | Path | None = None,
    coherence_config_override: dict[str, Any] | None = None,
    coherence_artifact_run_dir: str | Path | None = None,
    evaluation_seed_override: int | None = None,
) -> _SetEvaluationResult:
    """Stream a complete split through one loaded model and plot field-wise error statistics."""
    run_dir = Path(run_dir).resolve()
    case_dir = Path(case_dir).resolve()
    checkpoint_label = Path(checkpoint).stem
    output_dir = (
        Path(output_dir_override).resolve()
        if output_dir_override is not None
        else run_dir / "evaluation" / f"reconstruction_set_{split}_{checkpoint_label}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = (
        Path(output_path).resolve()
        if output_path is not None
        else output_dir / "relative_l2_violin.png"
    )
    warn_if_cuda_memory_tight(run_dir, checkpoint=checkpoint, device_name=device_name)
    runtime = load_evaluation_runtime(
        run_dir,
        split=split,
        checkpoint=checkpoint,
        sensor_config=sensor_config,
        generation_steps=generation_steps,
        device_name=device_name,
        include_temporal_derivative=False,
    )
    if coherence_config_override is not None:
        runtime.config = _merge_coherence_evaluation_override(
            runtime.config,
            coherence_config_override,
            tuple(coherence_families or ()),
        )
    if coherence_artifact_run_dir is not None:
        runtime.coherence_artifact_run_dir = Path(coherence_artifact_run_dir).resolve()
    if evaluation_seed_override is not None:
        runtime.seed = int(evaluation_seed_override)
    dataset = runtime.dataset
    evaluated_dataset_fingerprint = dataset_fingerprint(dataset.path)
    available_sample_count = len(dataset)
    if available_sample_count < 1:
        dataset.close()
        raise ValueError(f"split {split!r} contains no samples")
    selected_indices = _select_sample_indices(available_sample_count, max_samples)
    sample_count = int(selected_indices.size)
    input_manifest = SensorManifest.load(sensor_manifest) if sensor_manifest is not None else None
    field_names = tuple(dataset.field_names)
    coherence = build_coherence_accumulators(
        coherence_families or (),
        runtime,
        extra_views=extra_coherence_views,
        evaluated_sample_count=sample_count,
        cross_spectrum_aggregation=cross_spectrum_aggregation,
    )
    errors = np.full((sample_count, len(field_names)), np.nan, dtype=np.float64)
    sample_ids: list[str] = []
    manifest_path = output_dir / "sensor_manifest.jsonl"
    csv_path = output_dir / "relative_l2.csv"
    started = perf_counter()
    inference_seconds = 0.0
    if runtime.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.device)

    try:
        with (
            manifest_path.open("w", encoding="utf-8") as manifest_stream,
            csv_path.open("w", encoding="utf-8", newline="") as csv_stream,
            selected_evaluation_weight_context(runtime.model, weight_selection),
            torch.no_grad(),
        ):
            base_protocol = sensor_protocol_from_config(runtime.config)
            manifest_stream.write(
                json.dumps(
                    {
                        "type": "metadata",
                        "version": "1",
                        "dataset_path": dataset.path.name,
                        "dataset_fingerprint": evaluated_dataset_fingerprint,
                        "split": split,
                        "protocol": base_protocol.to_dict(),
                        "sensor_seed_policy": "base_seed_plus_split_relative_sample_index",
                        "query_policy": "full_grid",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            writer = csv.writer(csv_stream)
            writer.writerow(("sample_index", "sample_id", *field_names))
            for output_index, sample_index_value in enumerate(
                tqdm(
                    selected_indices,
                    desc=f"evaluate {split} set",
                    unit="sample",
                )
            ):
                sample_index = int(sample_index_value)
                sample = dataset[sample_index]
                if input_manifest is None:
                    batch = build_observation_batch(
                        [sample],
                        sensor_protocol_from_config(runtime.config, seed_offset=sample_index),
                        query_points=None,
                    )
                else:
                    batch = build_batch_from_manifest(
                        [sample], input_manifest, dataset.path, query_points=None
                    )
                if batch.obs_indices is None:
                    raise ValueError("set evaluation requires observation indices")
                valid_obs = batch.obs_valid_mask[0]
                manifest_stream.write(
                    json.dumps(
                        {
                            "type": "sample",
                            "sample_index": sample_index,
                            "sample_id": batch.sample_ids[0],
                            "sensor_seed": int(batch.metadata["protocol"]["seed"]),
                            "indices": [
                                [int(point), int(field)]
                                for point, field in zip(
                                    batch.obs_indices[0, valid_obs].tolist(),
                                    batch.obs_field_ids[0, valid_obs].tolist(),
                                )
                            ],
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                batch = batch.to(runtime.device)
                if output_index == 0:
                    warmup_generator = torch.Generator(device=runtime.device).manual_seed(
                        runtime.seed
                    )
                    runtime.model.reconstruct(
                        batch,
                        steps=runtime.generation_steps,
                        generator=warmup_generator,
                    )
                    if runtime.device.type == "cuda":
                        torch.cuda.synchronize(runtime.device)
                        torch.cuda.reset_peak_memory_stats(runtime.device)
                generator = torch.Generator(device=runtime.device).manual_seed(
                    runtime.seed + sample_index
                )
                inference_started = perf_counter()
                reconstruction = runtime.model.reconstruct(
                    batch,
                    steps=runtime.generation_steps,
                    generator=generator,
                )
                if runtime.device.type == "cuda":
                    torch.cuda.synchronize(runtime.device)
                inference_seconds += perf_counter() - inference_started
                target = batch.target_fields
                if target is None:
                    raise ValueError("set evaluation requires dense targets")
                sample_errors = (
                    _physical_relative_l2(
                        reconstruction.prediction,
                        target,
                        dataset.normalizer,
                    )[0]
                    .cpu()
                    .numpy()
                )
                errors[output_index] = sample_errors
                sample_ids.append(batch.sample_ids[0])
                valid_query = batch.query_valid_mask[0]
                for accumulator in coherence.values():
                    accumulator.update(
                        reconstruction.prediction[:, valid_query],
                        target[:, valid_query],
                        batch.query_coords[:, valid_query],
                        batch.sample_ids[0],
                    )
                writer.writerow((sample_index, batch.sample_ids[0], *sample_errors.tolist()))
                if (output_index + 1) % 100 == 0:
                    manifest_stream.flush()
                    csv_stream.flush()
    finally:
        dataset.close()

    payload_path = output_dir / "relative_l2.npz"
    np.savez_compressed(
        payload_path,
        per_field_relative_l2_physical=errors,
        field_names=np.asarray(field_names),
        sample_ids=np.asarray(sample_ids),
        split=np.asarray(split),
    )
    render_reconstruction_set_distribution(
        payload_path,
        figure_path,
        title=(
            f"{run_dir.parent.name} — {split} set relative $L_2$ "
            f"({checkpoint_label}.pt, n={sample_count})"
        ),
        dpi=300,
        scale=statistic_scale,
    )
    coherence_outputs = {
        name: accumulator.finalize(
            output_dir,
            split=split,
            checkpoint_label=checkpoint_label,
            run_label=run_dir.parent.name,
            scale=statistic_scale,
        )
        for name, accumulator in coherence.items()
    }
    report = {
        "metric": "per_sample_per_field_relative_l2_physical",
        "split": split,
        "available_sample_count": available_sample_count,
        "sample_count": sample_count,
        "selection": {
            "policy": (
                "full_split"
                if sample_count == available_sample_count
                else "evenly_spaced_split_subset"
            ),
            "requested_max_samples": max_samples,
            "split_relative_indices": selected_indices.tolist(),
        },
        "field_names": list(field_names),
        "statistic_scale": statistic_scale,
        "coherence": coherence_outputs,
        "per_field_statistics": {
            field: _finite_summary(errors[:, field_index])
            for field_index, field in enumerate(field_names)
        },
        "compute": {
            "total_seconds": perf_counter() - started,
            "inference_seconds": inference_seconds,
            "inference_seconds_per_sample": inference_seconds / sample_count,
            "peak_cuda_memory_bytes": (
                torch.cuda.max_memory_allocated(runtime.device)
                if runtime.device.type == "cuda"
                else 0
            ),
        },
        "trace": {
            "run": str(run_dir.relative_to(case_dir)),
            "resolved_config_sha256": file_sha256(run_dir / "resolved_config.yaml"),
            "checkpoint": _trace_path(runtime.checkpoint_path, run_dir),
            "checkpoint_sha256": file_sha256(runtime.checkpoint_path),
            "dataset_fingerprint": evaluated_dataset_fingerprint,
            "weight_selection": weight_selection,
            "generation_steps": runtime.generation_steps,
            "generation_seed_policy": "base_seed_plus_split_relative_sample_index",
            "sensor_manifest": _trace_path(manifest_path, run_dir),
            "sensor_manifest_sha256": file_sha256(manifest_path),
            "metrics_csv": _trace_path(csv_path, run_dir),
            "metrics_payload": _trace_path(payload_path, run_dir),
            "figure": str(figure_path),
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _SetEvaluationResult(
        run_dir=run_dir,
        output_dir=output_dir,
        figure_path=figure_path,
        payload_path=payload_path,
        report_path=report_path,
        manifest_path=manifest_path,
        checkpoint_path=runtime.checkpoint_path,
        checkpoint_label=checkpoint_label,
        run_label=run_dir.parent.name,
        split=split,
        sample_ids=tuple(sample_ids),
        field_names=field_names,
        dataset_fingerprint=evaluated_dataset_fingerprint,
        generation_steps=runtime.generation_steps,
        evaluation_seed=runtime.seed,
        coherence_outputs=coherence_outputs,
        coherence_accumulators=coherence,
    )


def _load_relative_l2_payload(path: Path) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    with np.load(path, allow_pickle=False) as payload:
        return (
            np.asarray(payload["per_field_relative_l2_physical"], dtype=np.float64),
            tuple(str(value) for value in payload["field_names"]),
            tuple(str(value) for value in payload["sample_ids"]),
        )


def _global_distribution_specs(
    path: Path,
) -> dict[str, tuple[np.ndarray, tuple[str, ...], tuple[str, ...], str, str]]:
    with np.load(path, allow_pickle=False) as payload:
        if "marginal_per_field_w2" in payload.files:
            marginal = np.asarray(payload["marginal_per_field_w2"], dtype=np.float64)
            joint = np.asarray(payload["joint_copula_smooth_cvar"], dtype=np.float64)
            direction_mean = np.asarray(payload["mean_directional_w2"], dtype=np.float64)
            smooth_tail = np.asarray(
                payload["identity_calibrated_smooth_cvar"], dtype=np.float64
            )
            hard_tail = np.asarray(
                payload["hard_empirical_cvar_diagnostic"], dtype=np.float64
            )
            exact_rank = np.asarray(
                payload["exact_midranks_projected_w2_diagnostic"], dtype=np.float64
            )
            direction_costs = np.asarray(payload["per_direction_w2"], dtype=np.float64)
            component_totals = np.asarray(
                payload["weighted_component_totals"], dtype=np.float64
            )
            family_total = np.asarray(payload["family_total"], dtype=np.float64)
            field_names = tuple(str(value) for value in payload["field_names"])
        else:
            marginal = joint = direction_mean = smooth_tail = hard_tail = exact_rank = None
            direction_costs = component_totals = family_total = None
            field_names = ()

        if marginal is not None:
            if marginal.ndim != 2 or component_totals.shape != (len(marginal), 2):
                raise ValueError("global-distribution v2 marginal metrics are misaligned")
            if family_total.shape != (len(marginal), 1) or joint.shape != (len(marginal), 1):
                raise ValueError("global-distribution v2 family metrics are misaligned")
            if direction_costs.ndim != 2 or direction_costs.shape[0] != len(marginal):
                raise ValueError("global-distribution v2 direction spectrum is misaligned")
            if not all(
                value.shape == (len(marginal), 1)
                for value in (direction_mean, smooth_tail, hard_tail, exact_rank)
            ):
                raise ValueError("global-distribution v2 tail diagnostics are misaligned")
            directions = tuple(
                f"Projection {index + 1:02d}"
                for index in range(direction_costs.shape[1])
            )
            marginal_plot = np.column_stack(
                (marginal, component_totals[:, 0], family_total[:, 0])
            )
            direction_plot = np.column_stack(
                (
                    direction_costs,
                    direction_mean[:, 0],
                    smooth_tail[:, 0],
                    joint[:, 0],
                    family_total[:, 0],
                )
            )
            tail_plot = np.column_stack(
                (
                    hard_tail[:, 0],
                    smooth_tail[:, 0],
                    exact_rank[:, 0],
                    joint[:, 0],
                    family_total[:, 0],
                )
            )
            return {
                "marginal": (
                    marginal_plot,
                    (*field_names, "Marginal total", "Family total"),
                    (*("detail" for _ in field_names), "component_total", "family_total"),
                    "marginal_copula_v2_marginal_distributions.png",
                    "Marginal field-distribution coherence · marginal-copula v2",
                ),
                "direction_spectrum": (
                    direction_plot,
                    (
                        *directions,
                        "Direction mean",
                        "Identity-calibrated smooth CVaR",
                        "Joint objective total",
                        "Family total",
                    ),
                    (
                        *("detail" for _ in directions),
                        "detail",
                        "detail",
                        "component_total",
                        "family_total",
                    ),
                    "marginal_copula_v2_direction_spectrum.png",
                    "Frozen projection direction spectrum · marginal-copula v2",
                ),
                "tail_diagnostics": (
                    tail_plot,
                    (
                        "Hard empirical CVaR",
                        "Identity-calibrated smooth CVaR",
                        "Exact midrank diagnostic",
                        "Joint objective total",
                        "Family total",
                    ),
                    ("detail", "detail", "detail", "component_total", "family_total"),
                    "marginal_copula_v2_tail_diagnostics.png",
                    "Tail and exact-rank diagnostics · marginal-copula v2",
                ),
            }

        marginal = np.asarray(payload["marginal_per_field"], dtype=np.float64)
        pairwise = np.asarray(payload["pairwise_per_field_pair"], dtype=np.float64)
        joint = np.asarray(payload["joint_top_tail"], dtype=np.float64)
        component_totals = np.asarray(payload["weighted_component_totals"], dtype=np.float64)
        family_total = np.asarray(payload["family_total"], dtype=np.float64)
        field_names = tuple(str(value) for value in payload["field_names"])
        pair_labels = tuple(str(value) for value in payload["pair_labels"])
    return {
        "marginal": (
            np.column_stack((marginal, component_totals[:, 0], family_total[:, 0])),
            (*field_names, "Marginal total", "Family total"),
            (*("detail" for _ in field_names), "component_total", "family_total"),
            "marginal_field_distributions.png",
            "Marginal field-distribution coherence",
        ),
        "pairwise": (
            np.column_stack((pairwise, component_totals[:, 1], family_total[:, 0])),
            (*pair_labels, "Pairwise total", "Family total"),
            (*("detail" for _ in pair_labels), "component_total", "family_total"),
            "pairwise_field_distributions.png",
            "Pairwise field-distribution coherence",
        ),
        "joint_top_tail": (
            np.column_stack((joint[:, 0], component_totals[:, 2], family_total[:, 0])),
            ("Joint/top-tail", "Joint total", "Family total"),
            ("detail", "component_total", "family_total"),
            "joint_top_tail_distributions.png",
            "Joint/top-tail distribution coherence",
        ),
    }


def _global_distribution_pair_contract(
    base_payload: Path,
    current_payload: Path,
    base_report: dict[str, Any],
    current_report: dict[str, Any],
    *,
    base_sample_ids: tuple[str, ...],
    current_sample_ids: tuple[str, ...],
    base_field_names: tuple[str, ...],
    current_field_names: tuple[str, ...],
    base_weight_selection: str | None,
    current_weight_selection: str | None,
) -> dict[str, Any]:
    """Validate the saved contracts underlying a paired global-family report."""
    base_definition = str(base_report.get("definition", "legacy_v1"))
    current_definition = str(current_report.get("definition", "legacy_v1"))
    if base_definition != current_definition:
        raise ValueError("base and post-training global-distribution definitions differ")
    if base_sample_ids != current_sample_ids:
        raise ValueError("base and post-training global-distribution sample identities differ")
    if base_field_names != current_field_names:
        raise ValueError("base and post-training global-distribution field orders differ")

    def read_identity(path: Path) -> dict[str, Any]:
        with np.load(path, allow_pickle=False) as payload:
            sample_ids = tuple(str(value) for value in payload["sample_ids"])
            fields = tuple(str(value) for value in payload["field_names"])
            units = str(payload["units"].item())
            version_two = "marginal_per_field_w2" in payload.files
            identity: dict[str, Any] = {
                "sample_ids": sample_ids,
                "field_names": fields,
                "units": units,
                "definition": "marginal_copula_v2" if version_two else "legacy_v1",
            }
            if version_two:
                identity.update(
                    projection_directions=np.asarray(payload["projection_directions"]),
                    projection_direction_sha256=str(
                        payload["projection_direction_sha256"].item()
                    ),
                    pairwise_projection_directions=np.asarray(
                        payload["pairwise_projection_directions"]
                    ),
                    pairwise_projection_direction_sha256=str(
                        payload["pairwise_projection_direction_sha256"].item()
                    ),
                )
            return identity

    base_identity = read_identity(base_payload)
    current_identity = read_identity(current_payload)
    for name, identity, expected_ids, expected_fields, report in (
        ("base", base_identity, base_sample_ids, base_field_names, base_report),
        (
            "post-training",
            current_identity,
            current_sample_ids,
            current_field_names,
            current_report,
        ),
    ):
        if identity["sample_ids"] != expected_ids or identity["field_names"] != expected_fields:
            raise ValueError(f"{name} global-distribution metrics do not match evaluated samples")
        if identity["definition"] != base_definition:
            raise ValueError(f"{name} global-distribution metrics do not match its report definition")
        if report.get("units") != identity["units"]:
            raise ValueError(f"{name} global-distribution report and metric units differ")
    if base_identity["units"] != current_identity["units"]:
        raise ValueError("base and post-training global-distribution units differ")

    if base_definition != "marginal_copula_v2":
        return {
            "definition": base_definition,
            "sample_ids": list(current_sample_ids),
            "field_names": list(current_field_names),
            "units": current_identity["units"],
        }

    import hashlib

    for bank_name in ("projection_directions", "pairwise_projection_directions"):
        base_bank = base_identity[bank_name]
        current_bank = current_identity[bank_name]
        if not np.array_equal(base_bank, current_bank):
            raise ValueError(f"base and post-training global-distribution {bank_name} differ")
        digest_key = (
            "projection_direction_sha256"
            if bank_name == "projection_directions"
            else "pairwise_projection_direction_sha256"
        )
        for identity, report in (
            (base_identity, base_report),
            (current_identity, current_report),
        ):
            actual_digest = hashlib.sha256(
                np.ascontiguousarray(identity[bank_name]).tobytes()
            ).hexdigest()
            if identity[digest_key] != actual_digest:
                raise ValueError(f"{bank_name} checksum does not match saved direction bank")
            report_digest = (
                report.get("projection_bank", {}).get("sha256")
                if bank_name == "projection_directions"
                else report.get("pairwise_diagnostics", {}).get(
                    "projection_direction_sha256"
                )
            )
            if report_digest != actual_digest:
                raise ValueError(f"{bank_name} checksum does not match its saved report")

    for name, report in (("base", base_report), ("post-training", current_report)):
        if report.get("projection_bank", {}).get("artifact_restored") is not True:
            raise ValueError(f"{name} global-distribution projection bank was not restored")
        artifact = report.get("family_artifact", {})
        if not artifact.get("artifact_path") or not artifact.get("artifact_sha256"):
            raise ValueError(f"{name} global-distribution family artifact provenance is missing")
        if not artifact.get("artifact_run_dir"):
            raise ValueError(f"{name} global-distribution artifact run directory is missing")
    base_artifact = base_report["family_artifact"]
    current_artifact = current_report["family_artifact"]
    for key in (
        "artifact_path",
        "artifact_sha256",
        "artifact_run_dir",
        "source_run_id",
        "source_checkpoint_sha256",
    ):
        if base_artifact.get(key) != current_artifact.get(key):
            raise ValueError("base and post-training runs did not restore the same frozen family artifact")

    if base_report.get("settings") != current_report.get("settings"):
        raise ValueError("base and post-training global-distribution settings differ")
    if base_report.get("family_weight") != current_report.get("family_weight"):
        raise ValueError("base and post-training global-distribution family weights differ")
    if base_report.get("component_weights") != current_report.get("component_weights"):
        raise ValueError("base and post-training global-distribution component weights differ")
    base_tail = base_report.get("objective", {}).get("mean_tail_mixture", {})
    current_tail = current_report.get("objective", {}).get("mean_tail_mixture", {})
    tail_parameters = {name: base_tail.get(name) for name in ("alpha", "rho", "temperature")}
    if tail_parameters != {
        name: current_tail.get(name) for name in ("alpha", "rho", "temperature")
    }:
        raise ValueError("base and post-training global-distribution tail parameters differ")

    if base_weight_selection is None or current_weight_selection is None:
        raise ValueError("paired global-distribution model weight-selection policy is unavailable")
    if base_weight_selection != current_weight_selection:
        raise ValueError("base and post-training model weight-selection policies differ")

    projection = current_report["projection_bank"]
    artifact_contract = {
        key: current_artifact.get(key)
        for key in (
            "artifact_run_dir",
            "artifact_sha256",
            "source_run_id",
            "source_checkpoint_sha256",
        )
    }
    return {
        "definition": base_definition,
        "sample_ids": list(current_sample_ids),
        "field_names": list(current_field_names),
        "units": current_identity["units"],
        "weight_selection": current_weight_selection,
        "family_weight": current_report["family_weight"],
        "component_weights": current_report["component_weights"],
        "family_artifact": artifact_contract,
        "projection_bank": {
            key: projection.get(key)
            for key in ("kind", "directions", "dimension", "include_axes", "qmc", "seed", "sha256")
        },
        "tail_parameters": tail_parameters,
        "canonicalizer": current_report.get("objective", {}).get("canonicalizer", {}),
    }


def _cross_spectrum_specs(
    path: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...], str, str]]:
    with np.load(path, allow_pickle=False) as payload:
        component_names = tuple(str(value) for value in payload["component_names"])
        component_scores = {
            name: float(payload["component_coherence_scores"][index])
            for index, name in enumerate(component_names)
        }
        component_score_std = {
            name: float(payload["component_coherence_score_std"][index])
            for index, name in enumerate(component_names)
        }
        family_score = float(payload["family_coherence_score"])
        family_score_std = float(payload["family_coherence_score_std"])
        field_names = tuple(str(value) for value in payload["field_names"])
        pair_labels = tuple(str(value) for value in payload["pair_labels"])
        band_labels = tuple(str(value) for value in payload["band_field_labels"])
        arrays = {
            "self_spectrum": np.asarray(payload["self_spectrum_coherence_score"], dtype=np.float64),
            "same_frequency": np.asarray(
                payload["same_frequency_coherence_score"], dtype=np.float64
            ),
            "cross_frequency": np.asarray(
                payload["cross_frequency_coherence_score"], dtype=np.float64
            ),
            "band_energy": np.asarray(payload["band_energy_coherence_score"], dtype=np.float64),
        }
        spread_arrays = {
            "self_spectrum": np.asarray(
                payload["self_spectrum_coherence_score_std"], dtype=np.float64
            ),
            "same_frequency": np.asarray(
                payload["same_frequency_coherence_score_std"], dtype=np.float64
            ),
            "cross_frequency": np.asarray(
                payload["cross_frequency_coherence_score_std"], dtype=np.float64
            ),
            "band_energy": np.asarray(payload["band_energy_coherence_score_std"], dtype=np.float64),
        }
    definitions = {
        "self_spectrum": (
            field_names,
            "self_spectrum_coherence.png",
            "Self-spectrum coherence",
            "Field mean",
        ),
        "same_frequency": (
            pair_labels,
            "same_frequency_coherence.png",
            "Same-frequency spectral coherence",
            "Pair mean",
        ),
        "cross_frequency": (
            pair_labels,
            "cross_frequency_coherence.png",
            "Cross-frequency spectral coherence",
            "Pair mean",
        ),
        "band_energy": (
            band_labels,
            "band_energy_coherence.png",
            "Spectral-band energy coherence",
            "Band–field mean",
        ),
    }
    specs = {}
    for name, scores in arrays.items():
        if not scores.size:
            continue
        labels, filename, title, component_label = definitions[name]
        specs[name] = (
            np.concatenate((scores, [component_scores[name], family_score])),
            np.concatenate((spread_arrays[name], [component_score_std[name], family_score_std])),
            (*labels, component_label, "Overall score"),
            (*("detail" for _ in labels), "component_total", "family_total"),
            filename,
            title,
        )
    return specs


def _cross_spectrum_comparison_subtitle(
    role: str,
    result: _SetEvaluationResult,
    payload_path: Path,
) -> str:
    with np.load(payload_path, allow_pickle=False) as payload:
        aggregation = str(payload["aggregation"])
        ensemble_count = int(payload["ensemble_count"])
        ensemble_size = int(payload["ensemble_size"])
        used_count = int(payload["sample_ids"].size)
        selected_count = int(payload["selected_sample_ids"].size)
    if aggregation == "training_aligned":
        ensemble_summary = (
            f"{ensemble_count}×{ensemble_size} ensembles · n={used_count}/{selected_count}"
        )
        estimate_summary = "bars mean; whiskers ±1 SD"
    else:
        ensemble_summary = f"one pooled ensemble · n={used_count}"
        estimate_summary = "single pooled estimate"
    return (
        f"{role} · {result.split} · "
        f"{result.checkpoint_label}.pt · {ensemble_summary} · {estimate_summary}"
    )


def _comparison_subtitle(role: str, result: _SetEvaluationResult, *, scale: str) -> str:
    return (
        f"{role} · {result.run_label.replace('_', ' ')} · {result.split} · "
        f"{result.checkpoint_label}.pt · "
        f"n={len(result.sample_ids)} · shared {scale} axis"
    )


def _base_figure_path(path: Path) -> Path:
    """Return the adjacent source-run figure name for a comparison artifact."""
    return path.with_name(f"{path.stem}-base{path.suffix}")


def _report_artifact_path(path: Path, output_dir: Path) -> str:
    """Prefer compact output-relative paths while supporting custom figure paths."""
    try:
        return str(path.relative_to(output_dir))
    except ValueError:
        return str(path)


def _load_topology_comparison_data(payload_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = payload_path.with_name("report.json")
    if not payload_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("topology comparison payload or report is unavailable")
    keys = (
        "sample_ids",
        "field_names",
        "source_grid_shape",
        "configured_grid_shape",
        "query_indices",
        "query_coordinates",
        "query_seed",
        "query_point_count",
        "quantiles",
        "objective_component_names",
        "objective_component_distances",
        "family_total_distance",
        "filtration_metric",
        "filtration_kind",
        "filtration_direction",
        "filtration_line_index",
        "filtration_line_weight",
        "reference_betti",
        "reconstruction_betti",
        "grid_coordinates",
    )
    with np.load(payload_path, allow_pickle=False) as payload:
        missing = [key for key in keys if key not in payload]
        if missing:
            raise ValueError(f"topology comparison payload is missing {missing}")
        data = {key: np.asarray(payload[key]).copy() for key in keys}
    data["objective_component_names"] = tuple(str(value) for value in data["objective_component_names"])
    data["field_names"] = tuple(str(value) for value in data["field_names"])
    data["sample_ids"] = tuple(str(value) for value in data["sample_ids"])
    data["filtration_metric"] = tuple(str(value) for value in data["filtration_metric"])
    data["filtration_kind"] = tuple(str(value) for value in data["filtration_kind"])
    data["filtration_direction"] = tuple(str(value) for value in data["filtration_direction"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return data, report


def _validate_topology_comparison(
    base: dict[str, Any],
    current: dict[str, Any],
    base_report: dict[str, Any],
    current_report: dict[str, Any],
) -> None:
    for key in (
        "sample_ids",
        "field_names",
        "source_grid_shape",
        "configured_grid_shape",
        "query_indices",
        "query_seed",
        "query_point_count",
        "quantiles",
        "objective_component_names",
        "filtration_metric",
        "filtration_kind",
        "filtration_direction",
        "filtration_line_index",
        "filtration_line_weight",
    ):
        if not np.array_equal(base[key], current[key]):
            raise ValueError(f"base and post-training topology {key} do not match")
    for key in ("query_coordinates", "grid_coordinates", "reference_betti"):
        if not np.array_equal(base[key], current[key]):
            raise ValueError(f"base and post-training topology {key} do not match")
    if not np.array_equal(base["objective_component_names"], current["objective_component_names"]):
        raise ValueError("base and post-training topology distance labels do not match")
    if base_report.get("units") != current_report.get("units"):
        raise ValueError("base and post-training topology units do not match")
    base_query = base_report.get("query_contract", {})
    current_query = current_report.get("query_contract", {})
    base_raster = base_report.get("raster", {})
    current_raster = current_report.get("raster", {})
    base_objective = base_report.get("objective", {})
    current_objective = current_report.get("objective", {})
    if base_query != current_query:
        raise ValueError("base and post-training topology fixed-query contracts do not match")
    for key in ("grid_shape", "source_grid_shape", "periodic", "geometry_sha256"):
        if base_raster.get(key) != current_raster.get(key):
            raise ValueError(f"base and post-training topology raster {key} do not match")
    if base_objective.get("settings") != current_objective.get("settings"):
        raise ValueError("base and post-training topology family settings do not match")


def _load_band_profile_comparison_data(
    payload_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = payload_path.with_name("report.json")
    if not payload_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("cross-spectrum comparison payload or report is unavailable")
    keys = (
        "field_names",
        "graph_band_names",
        "band_field_labels",
        "graph_band_energy_fraction_reference_mean",
        "graph_band_energy_fraction_reference_std",
        "graph_band_energy_fraction_reconstruction_mean",
        "graph_band_energy_fraction_reconstruction_std",
        "sample_ids",
        "selected_sample_ids",
        "ensemble_sample_ids",
        "aggregation",
        "ensemble_size",
        "ensemble_count",
    )
    with np.load(payload_path, allow_pickle=False) as payload:
        missing = [key for key in keys if key not in payload]
        if missing:
            raise ValueError(f"cross-spectrum comparison payload is missing {missing}")
        data = {key: np.asarray(payload[key]).copy() for key in keys}
    for key in ("field_names", "graph_band_names", "band_field_labels", "sample_ids", "selected_sample_ids"):
        data[key] = tuple(str(value) for value in data[key].reshape(-1))
    data["aggregation"] = str(data["aggregation"].item())
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return data, report


def _validate_band_profile_comparison(
    base: dict[str, Any],
    current: dict[str, Any],
    base_report: dict[str, Any],
    current_report: dict[str, Any],
) -> None:
    for key in (
        "field_names",
        "graph_band_names",
        "band_field_labels",
        "sample_ids",
        "selected_sample_ids",
        "aggregation",
        "ensemble_size",
        "ensemble_count",
    ):
        if not np.array_equal(base[key], current[key]):
            raise ValueError(f"base and post-training graph-band profile {key} do not match")
    if not np.array_equal(base["ensemble_sample_ids"], current["ensemble_sample_ids"]):
        raise ValueError("base and post-training graph-band ensemble memberships do not match")
    if base_report.get("units") != current_report.get("units"):
        raise ValueError("base and post-training cross-spectrum field units do not match")
    base_graph = base_report.get("graph", {})
    current_graph = current_report.get("graph", {})
    for key in (
        "query_policy",
        "query_seed",
        "query_point_count",
        "geometry_sha256",
        "k_neighbors",
        "resolved_sigma",
        "num_modes",
        "eigenvalues",
        "band_mode_ids",
    ):
        if base_graph.get(key) != current_graph.get(key):
            raise ValueError(f"base and post-training cross-spectrum graph contract {key} does not match")
    for role in ("reference", "reconstruction"):
        for statistic in ("mean", "std"):
            key = f"graph_band_energy_fraction_{role}_{statistic}"
            if base[key].shape != current[key].shape:
                raise ValueError(f"base and post-training graph-band profile {key} shape differs")


def _load_second_order_spectrum_artifacts(
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, Path]:
    cross_dir = output_dir / "coherence" / "cross_spectrum"
    report_path = cross_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"second-order cross-spectrum report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("definition") != "second_order_blocks_v4":
        raise ValueError("paired v4 cross-spectrum comparison requires second_order_blocks_v4 reports")
    artifact_paths = report.get("artifacts", {})
    blocks_path = cross_dir / str(artifact_paths.get("signed_covariance_blocks_json", ""))
    metrics_path = cross_dir / str(artifact_paths.get("covariance_metrics_csv", ""))
    profile_path = cross_dir / str(artifact_paths.get("covariance_band_energy_profile_csv", ""))
    if not blocks_path.is_file() or not metrics_path.is_file() or not profile_path.is_file():
        raise FileNotFoundError("second-order cross-spectrum raw artifacts are incomplete")
    blocks = json.loads(blocks_path.read_text(encoding="utf-8"))
    if blocks.get("definition") != "second_order_blocks_v4":
        raise ValueError("signed covariance artifact definition does not match the report")
    return report, blocks, blocks_path, metrics_path, profile_path


def _validate_second_order_spectrum_pair(
    base_report: dict[str, Any],
    current_report: dict[str, Any],
    base_blocks: dict[str, Any],
    current_blocks: dict[str, Any],
) -> None:
    """Require one graph, mask, normalization and ensemble contract on both sides."""
    for key in (
        "version",
        "definition",
        "target_use",
        "units",
        "field_names",
        "field_pairs",
        "bands",
        "family_weight",
        "component_weights",
        "settings",
        "aggregation",
        "selected_sample_ids",
    ):
        if base_report.get(key) != current_report.get(key):
            raise ValueError(f"base and post-training v4 cross-spectrum {key} do not match")
    base_graph = base_report.get("graph", {})
    current_graph = current_report.get("graph", {})
    for key in (
        "query_policy",
        "query_seed",
        "query_point_count",
        "query_indices",
        "geometry_sha256",
        "k_neighbors",
        "resolved_sigma",
        "num_modes",
        "eigenvalues",
        "band_intervals",
        "band_mode_ids",
    ):
        if base_graph.get(key) != current_graph.get(key):
            raise ValueError(f"base and post-training v4 cross-spectrum graph {key} does not match")
    for key in (
        "basis_sha256",
        "family_artifact_sha256",
        "source_checkpoint_sha256",
        "geometry_sha256",
    ):
        if base_report.get("training_artifact", {}).get(key) != current_report.get(
            "training_artifact", {}
        ).get(key):
            raise ValueError(f"base and post-training v4 cross-spectrum training artifact {key} differs")
    base_calibration = base_report.get("calibration", {})
    current_calibration = current_report.get("calibration", {})
    for key in (
        "source",
        "sha256",
        "ensemble_size",
        "band_energies",
        "relative_floor",
        "absolute_floor",
        "minimum_reference_band_fraction",
    ):
        if base_calibration.get(key) != current_calibration.get(key):
            raise ValueError(f"base and post-training v4 cross-spectrum calibration {key} differs")
    if base_report.get("covariance_estimator") != current_report.get("covariance_estimator"):
        raise ValueError("base and post-training v4 covariance normalization definitions differ")
    if base_blocks.get("basis_sha256") != current_blocks.get("basis_sha256"):
        raise ValueError("base and post-training signed covariance basis hashes differ")
    if base_blocks.get("calibration_sha256") != current_blocks.get("calibration_sha256"):
        raise ValueError("base and post-training signed covariance calibration hashes differ")
    if base_blocks.get("primary_aggregation") != current_blocks.get("primary_aggregation"):
        raise ValueError("base and post-training covariance aggregation differs")
    for report_payload, block_payload in (
        (base_report, base_blocks),
        (current_report, current_blocks),
    ):
        if block_payload.get("field_names") != report_payload.get("field_names"):
            raise ValueError("signed covariance field names do not match the v4 report")
        if block_payload.get("band_names") != report_payload.get("bands"):
            raise ValueError("signed covariance band names do not match the v4 report")
        if block_payload.get("basis_sha256") != report_payload.get("training_artifact", {}).get(
            "basis_sha256"
        ):
            raise ValueError("signed covariance basis hash does not match the v4 report")
        if block_payload.get("calibration_sha256") != report_payload.get("calibration", {}).get(
            "sha256"
        ):
            raise ValueError("signed covariance calibration hash does not match the v4 report")
        if block_payload.get("primary_aggregation") != report_payload.get("aggregation"):
            raise ValueError("signed covariance aggregation does not match the v4 report")

    for aggregation in ("pooled", "training_aligned"):
        base_summary = base_report.get("covariance_reports", {}).get(aggregation, {})
        current_summary = current_report.get("covariance_reports", {}).get(aggregation, {})
        base_records = base_blocks.get("reports", {}).get(aggregation, {}).get("ensembles", [])
        current_records = current_blocks.get("reports", {}).get(aggregation, {}).get("ensembles", [])
        if base_summary.get("sample_ids_by_ensemble") != current_summary.get("sample_ids_by_ensemble"):
            raise ValueError(f"base and post-training {aggregation} covariance memberships differ")
        if len(base_records) != len(current_records):
            raise ValueError(f"base and post-training {aggregation} covariance ensemble counts differ")
        if base_summary.get("sample_ids_by_ensemble") != [
            record.get("sample_ids") for record in base_records
        ]:
            raise ValueError(f"base {aggregation} report and signed-block memberships disagree")
        if current_summary.get("sample_ids_by_ensemble") != [
            record.get("sample_ids") for record in current_records
        ]:
            raise ValueError(f"post-training {aggregation} report and signed-block memberships disagree")
        for base_record, current_record in zip(base_records, current_records):
            if base_record.get("sample_ids") != current_record.get("sample_ids"):
                raise ValueError(f"base and post-training {aggregation} ensemble membership differs")
            for component, block_key in (
                ("same_frequency", "same_frequency_blocks"),
                ("cross_frequency", "cross_frequency_blocks"),
            ):
                base_diagnostics = base_record.get("diagnostics", {})
                current_diagnostics = current_record.get("diagnostics", {})
                for count_key in (
                    f"{component}_eligible_blocks",
                    f"{component}_skipped_blocks",
                ):
                    if base_diagnostics.get(count_key) != current_diagnostics.get(count_key):
                        raise ValueError(
                            f"base and post-training {aggregation} {component} reference masks differ"
                        )
                base_blocks_for_term = base_diagnostics.get(block_key, [])
                current_blocks_for_term = current_diagnostics.get(block_key, [])
                if [
                    (block.get("fields"), block.get("bands"))
                    for block in base_blocks_for_term
                ] != [
                    (block.get("fields"), block.get("bands"))
                    for block in current_blocks_for_term
                ]:
                    raise ValueError(
                        f"base and post-training {aggregation} {component} eligible blocks differ"
                    )
                for base_block, current_block in zip(
                    base_blocks_for_term, current_blocks_for_term
                ):
                    base_fractions = np.asarray(
                        base_block.get("reference_energy_fractions"), dtype=np.float64
                    )
                    current_fractions = np.asarray(
                        current_block.get("reference_energy_fractions"), dtype=np.float64
                    )
                    if base_fractions.shape != current_fractions.shape or not np.allclose(
                        base_fractions, current_fractions, rtol=1.0e-7, atol=1.0e-9
                    ):
                        raise ValueError(
                            f"base and post-training {aggregation} {component} reference masks differ"
                        )
                    if not np.isclose(
                        float(base_block.get("shared_floor")),
                        float(current_block.get("shared_floor")),
                        rtol=1.0e-10,
                        atol=0.0,
                    ):
                        raise ValueError(
                            f"base and post-training {aggregation} {component} stabilization floors differ"
                        )
                    base_reference = base_block.get("reference_normalized_block")
                    current_reference = current_block.get("reference_normalized_block")
                    if isinstance(base_reference, dict) and isinstance(current_reference, dict):
                        base_matrix = np.asarray(base_reference.get("real"), dtype=np.float64) + 1j * np.asarray(
                            base_reference.get("imag", 0.0), dtype=np.float64
                        )
                        current_matrix = np.asarray(current_reference.get("real"), dtype=np.float64) + 1j * np.asarray(
                            current_reference.get("imag", 0.0), dtype=np.float64
                        )
                    else:
                        base_matrix = np.asarray(base_reference, dtype=np.float64)
                        current_matrix = np.asarray(current_reference, dtype=np.float64)
                    if base_matrix.shape != current_matrix.shape or not np.allclose(
                        base_matrix, current_matrix, rtol=1.0e-7, atol=1.0e-9
                    ):
                        raise ValueError(
                            f"base and post-training {aggregation} {component} reference covariance differs"
                        )


def _v4_component_pair_values(
    report: dict[str, Any], aggregation: str, component: str
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    values = report["covariance_reports"][aggregation]["components"][component]["per_field_pair"]
    labels = tuple(values)
    means = np.asarray(
        [np.nan if values[label]["raw_block_loss_mean"] is None else values[label]["raw_block_loss_mean"] for label in labels],
        dtype=np.float64,
    )
    spreads = np.asarray(
        [np.nan if values[label]["raw_block_loss_sample_sd"] is None else values[label]["raw_block_loss_sample_sd"] for label in labels],
        dtype=np.float64,
    )
    return labels, means, spreads


def _render_v4_covariance_loss(
    report: dict[str, Any],
    component: str,
    output_path: Path,
    *,
    role: str,
    upper_limit: float,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .coherence_set import _save_publication_figure

    labels, means, spreads = _v4_component_pair_values(
        report, report["aggregation"], component
    )
    if not labels or not np.isfinite(means[np.isfinite(means)]).all():
        raise ValueError(f"{component} covariance report has no finite pair losses")
    positions = np.arange(len(labels), dtype=np.float64)
    valid = np.isfinite(means)
    figure, axis = plt.subplots(figsize=(7.6, max(3.8, 0.42 * len(labels) + 1.8)))
    axis.set_facecolor("white")
    figure.patch.set_facecolor("white")
    axis.barh(
        positions[valid],
        means[valid],
        xerr=np.nan_to_num(spreads[valid], nan=0.0),
        color="#687786" if role == "Base source" else "#16647F",
        capsize=3,
        zorder=3,
    )
    for position in positions[~valid]:
        axis.text(0.015 * upper_limit, position, "no eligible blocks", va="center", fontsize=8.5)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0.0, upper_limit)
    axis.set_xlabel("Mean squared Frobenius covariance-block difference")
    axis.set_title(
        f"{role} · {component.replace('_', ' ')} covariance blocks",
        loc="left",
        fontsize=11.0,
    )
    axis.grid(axis="x", color="#DCE3E8", linewidth=0.65)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    figure.tight_layout()
    result = _save_publication_figure(figure, output_path, dpi=300)
    plt.close(figure)
    return result


def _v4_energy_profile(
    blocks: dict[str, Any], aggregation: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ensembles = blocks["reports"][aggregation]["ensembles"]
    if not ensembles:
        raise ValueError(f"{aggregation} covariance report has no energy profile ensembles")
    references = np.asarray(
        [record["reference_energy_fractions"] for record in ensembles], dtype=np.float64
    )
    generated = np.asarray(
        [record["generated_energy_fractions"] for record in ensembles], dtype=np.float64
    )
    return (
        references.mean(axis=0),
        references.std(axis=0, ddof=1) if len(ensembles) > 1 else np.zeros_like(references[0]),
        generated.mean(axis=0),
        generated.std(axis=0, ddof=1) if len(ensembles) > 1 else np.zeros_like(generated[0]),
    )


def _render_v4_energy_profile(
    report: dict[str, Any],
    profile: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    output_path: Path,
    *,
    role: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .coherence_set import _save_publication_figure

    reference, reference_sd, generated, generated_sd = profile
    band_names = tuple(report["bands"])
    field_names = tuple(report["field_names"])
    expected_shape = (len(band_names), len(field_names))
    if reference.shape != expected_shape or generated.shape != expected_shape:
        raise ValueError("v4 covariance-energy profile does not align with bands and fields")
    columns = min(3, len(field_names))
    rows = int(np.ceil(len(field_names) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.0 * columns, 2.8 * rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    figure.patch.set_facecolor("white")
    x = np.arange(len(band_names), dtype=np.float64)
    for index, field_name in enumerate(field_names):
        row, column = divmod(index, columns)
        axis = axes[row, column]
        axis.errorbar(
            x,
            reference[:, index],
            yerr=reference_sd[:, index],
            color="#4B5B6B",
            marker="o",
            markersize=4,
            linewidth=1.5,
            capsize=2,
            label="Reference",
        )
        axis.errorbar(
            x,
            generated[:, index],
            yerr=generated_sd[:, index],
            color="#28769B",
            marker="s",
            markersize=3.8,
            linewidth=1.5,
            linestyle="--",
            capsize=2,
            label="Reconstruction",
        )
        axis.set_title(str(field_name), loc="left", fontsize=9.2, fontweight="semibold")
        axis.set_xticks(x, band_names)
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", color="#E2E6E9", linewidth=0.6, linestyle="--")
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    for index in range(len(field_names), rows * columns):
        figure.delaxes(axes.flat[index])
    axes[0, 0].set_ylabel("Centered covariance-energy fraction")
    axes[0, 0].legend(frameon=False, fontsize=7.7)
    figure.suptitle(f"{role} · centered covariance energy by graph band", fontsize=11)
    figure.tight_layout()
    result = _save_publication_figure(figure, output_path, dpi=300)
    plt.close(figure)
    return result


def _render_second_order_spectrum_comparison(
    current: _SetEvaluationResult,
    base: _SetEvaluationResult,
    *,
    shared_limits: dict[str, Any],
    artifacts: dict[str, Any],
    matched_coherence_contracts: dict[str, Any],
) -> None:
    current_report, current_blocks, current_blocks_path, current_csv_path, current_profile_csv = (
        _load_second_order_spectrum_artifacts(current.output_dir)
    )
    base_report, base_blocks, base_blocks_path, base_csv_path, base_profile_csv = (
        _load_second_order_spectrum_artifacts(base.output_dir)
    )
    _validate_second_order_spectrum_pair(
        base_report, current_report, base_blocks, current_blocks
    )
    cross_dir = current_blocks_path.parent
    base_blocks_copy = cross_dir / "covariance_blocks_v4-base.json"
    base_csv_copy = cross_dir / "covariance_metrics_v4-base.csv"
    base_profile_copy = cross_dir / "covariance_band_energy_profile_v4-base.csv"
    base_report_copy = cross_dir / "report-base.json"
    shutil.copy2(base_blocks_path, base_blocks_copy)
    shutil.copy2(base_csv_path, base_csv_copy)
    shutil.copy2(base_profile_csv, base_profile_copy)

    aggregation = current_report["aggregation"]
    upper_limits: dict[str, float] = {}
    for component in ("same_frequency", "cross_frequency"):
        values = []
        for report_payload in (base_report, current_report):
            _, means, spreads = _v4_component_pair_values(
                report_payload, aggregation, component
            )
            finite = np.isfinite(means)
            if finite.any():
                values.extend((means[finite] + np.nan_to_num(spreads[finite], nan=0.0)).tolist())
        upper = max(values) if values else 0.0
        upper_limits[component] = upper * 1.12 if upper > 0.0 else 1.0

    paired_figures: dict[str, dict[str, Path]] = {"base": {}, "post_training": {}}
    for component in ("same_frequency", "cross_frequency"):
        current_path = cross_dir / f"{component}_covariance_block_loss_v4.png"
        base_path = _base_figure_path(current_path)
        paired_figures["base"][component] = _render_v4_covariance_loss(
            base_report,
            component,
            base_path,
            role="Base source",
            upper_limit=upper_limits[component],
        )
        paired_figures["post_training"][component] = _render_v4_covariance_loss(
            current_report,
            component,
            current_path,
            role="Post-training",
            upper_limit=upper_limits[component],
        )

    base_profile = _v4_energy_profile(base_blocks, aggregation)
    current_profile = _v4_energy_profile(current_blocks, aggregation)
    if not np.allclose(base_profile[0], current_profile[0], rtol=0.0, atol=1.0e-6):
        raise ValueError("base and post-training v4 covariance reference energy profiles differ")
    current_profile_path = cross_dir / "covariance_band_energy_profile_v4.png"
    base_profile_path = _base_figure_path(current_profile_path)
    paired_figures["base"]["covariance_band_energy_profile"] = _render_v4_energy_profile(
        base_report,
        base_profile,
        base_profile_path,
        role="Base source",
    )
    paired_figures["post_training"]["covariance_band_energy_profile"] = _render_v4_energy_profile(
        current_report,
        current_profile,
        current_profile_path,
        role="Post-training",
    )

    def update_run_report(
        report_payload: dict[str, Any], *, role: str, base_role: bool
    ) -> dict[str, Any]:
        report_payload["paired_source_comparison"] = {
            "role": "base_source" if base_role else "post_training",
            "aggregation": aggregation,
            "selected_sample_ids": current_report["selected_sample_ids"],
            "training_aligned_sample_ids_by_ensemble": current_report[
                "covariance_reports"
            ]["training_aligned"]["sample_ids_by_ensemble"],
            "raw_block_loss_by_component": {
                component: report_payload["covariance_reports"][aggregation][
                    "components"
                ][component]
                for component in ("same_frequency", "cross_frequency")
            },
            "shared_raw_loss_limits": {
                name: [0.0, upper] for name, upper in upper_limits.items()
            },
            "shared_covariance_energy_fraction_limits": [0.0, 1.0],
        }
        figure_names = {
            component: path.name
            for component, path in paired_figures[role].items()
        }
        report_payload.setdefault("artifacts", {})["figures"] = figure_names
        report_payload["artifacts"]["vector_figures"] = {
            name: {"pdf": Path(filename).with_suffix(".pdf").name,
                   "svg": Path(filename).with_suffix(".svg").name}
            for name, filename in figure_names.items()
        }
        report_payload["artifacts"]["covariance_metrics_csv"] = (
            base_csv_copy.name if base_role else current_csv_path.name
        )
        report_payload["artifacts"]["signed_covariance_blocks_json"] = (
            base_blocks_copy.name if base_role else current_blocks_path.name
        )
        report_payload["artifacts"]["covariance_band_energy_profile_csv"] = (
            base_profile_copy.name if base_role else current_profile_csv.name
        )
        return report_payload

    base_report_copy.write_text(
        json.dumps(update_run_report(base_report, role="base", base_role=True), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    current_report_path = cross_dir / "report.json"
    current_report_path.write_text(
        json.dumps(
            update_run_report(current_report, role="post_training", base_role=False),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    shared_limits["cross_spectrum"] = {
        "definition": "second_order_blocks_v4",
        "aggregation": aggregation,
        "raw_covariance_block_loss_by_component": {
            name: [0.0, upper] for name, upper in upper_limits.items()
        },
        "covariance_energy_fraction": [0.0, 1.0],
    }
    artifacts["base"]["cross_spectrum"] = {
        **{
            name: _report_artifact_path(path, current.output_dir)
            for name, path in paired_figures["base"].items()
        },
        "covariance_metrics_csv": _report_artifact_path(base_csv_copy, current.output_dir),
        "signed_covariance_blocks_json": _report_artifact_path(base_blocks_copy, current.output_dir),
        "covariance_band_energy_profile_csv": _report_artifact_path(base_profile_copy, current.output_dir),
        "report": _report_artifact_path(base_report_copy, current.output_dir),
    }
    artifacts["post_training"]["cross_spectrum"] = {
        **{
            name: _report_artifact_path(path, current.output_dir)
            for name, path in paired_figures["post_training"].items()
        },
        "covariance_metrics_csv": _report_artifact_path(current_csv_path, current.output_dir),
        "signed_covariance_blocks_json": _report_artifact_path(current_blocks_path, current.output_dir),
        "covariance_band_energy_profile_csv": _report_artifact_path(current_profile_csv, current.output_dir),
        "report": _report_artifact_path(current_report_path, current.output_dir),
    }
    def raw_losses_by_aggregation(report_payload: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for aggregation_name in ("pooled", "training_aligned"):
            aggregation_payload = report_payload["covariance_reports"][aggregation_name]
            result[aggregation_name] = {
                component: aggregation_payload.get("components", {}).get(component)
                for component in ("same_frequency", "cross_frequency")
            }
        return result

    matched_coherence_contracts["cross_spectrum"] = {
        "definition": "second_order_blocks_v4",
        "field_names": current_report["field_names"],
        "field_pairs": current_report["field_pairs"],
        "bands": current_report["bands"],
        "selected_sample_ids": current_report["selected_sample_ids"],
        "pooled_sample_ids_by_ensemble": current_report["covariance_reports"]["pooled"][
            "sample_ids_by_ensemble"
        ],
        "training_aligned_sample_ids_by_ensemble": current_report["covariance_reports"][
            "training_aligned"
        ]["sample_ids_by_ensemble"],
        "units": current_report["units"],
        "graph": current_report["graph"],
        "basis_sha256": current_report["training_artifact"]["basis_sha256"],
        "calibration_sha256": current_report["calibration"]["sha256"],
        "calibration": current_report["calibration"],
        "normalization": current_report["covariance_estimator"],
        "reference_mask": {
            "source": current_report["covariance_estimator"]["mask_source"],
            "minimum_reference_band_fraction": current_report["calibration"][
                "minimum_reference_band_fraction"
            ],
            "stabilization_floors": {
                "relative": current_report["calibration"]["relative_floor"],
                "absolute": current_report["calibration"]["absolute_floor"],
            },
        },
        "raw_block_loss_by_role_and_aggregation": {
            "base": raw_losses_by_aggregation(base_report),
            "post_training": raw_losses_by_aggregation(current_report),
        },
    }


def _render_posttraining_comparison(
    current: _SetEvaluationResult,
    base: _SetEvaluationResult,
    *,
    coherence_families: tuple[str, ...],
    statistic_scale: str,
) -> Path:
    """Render adjacent base figures and matched current figures with shared axes."""
    if current.dataset_fingerprint != base.dataset_fingerprint:
        raise ValueError("base and post-training comparison datasets do not match")
    if current.field_names != base.field_names:
        raise ValueError("base and post-training comparison field orders do not match")
    if current.sample_ids != base.sample_ids:
        raise ValueError("base and post-training comparison sample identities do not match")
    current_manifest_sha = file_sha256(current.manifest_path)
    base_manifest_sha = file_sha256(base.manifest_path)
    if current_manifest_sha != base_manifest_sha:
        raise ValueError("base and post-training comparison sensor selections do not match")

    current_errors, current_fields, current_ids = _load_relative_l2_payload(current.payload_path)
    base_errors, base_fields, base_ids = _load_relative_l2_payload(base.payload_path)
    if (current_fields, current_ids) != (base_fields, base_ids):
        raise ValueError("base and post-training reconstruction payloads are not aligned")
    reconstruction_limits = _relative_l2_distribution_limits(
        np.concatenate((base_errors.reshape(-1), current_errors.reshape(-1))),
        statistic_scale,
    )
    post_reconstruction = current.figure_path
    base_reconstruction = _base_figure_path(post_reconstruction)
    render_reconstruction_set_distribution(
        base.payload_path,
        base_reconstruction,
        title=(
            f"Base source — {base.run_label.replace('_', ' ')} — {base.split} relative $L_2$ "
            f"({base.checkpoint_label}.pt, n={len(base.sample_ids)})"
        ),
        scale=statistic_scale,
        value_limits=reconstruction_limits,
    )
    render_reconstruction_set_distribution(
        current.payload_path,
        post_reconstruction,
        title=(
            f"Post-training — {current.run_label.replace('_', ' ')} — "
            f"{current.split} relative $L_2$ "
            f"({current.checkpoint_label}.pt, n={len(current.sample_ids)})"
        ),
        scale=statistic_scale,
        value_limits=reconstruction_limits,
    )

    shared_limits: dict[str, Any] = {
        "relative_l2": list(reconstruction_limits),
    }
    artifacts: dict[str, Any] = {
        "base": {"relative_l2": _report_artifact_path(base_reconstruction, current.output_dir)},
        "post_training": {
            "relative_l2": _report_artifact_path(post_reconstruction, current.output_dir)
        },
    }
    matched_coherence_contracts: dict[str, Any] = {}
    if "global_distribution" in coherence_families:
        base_payload = base.output_dir / "coherence" / "global_distribution" / "metrics.npz"
        current_payload = current.output_dir / "coherence" / "global_distribution" / "metrics.npz"
        base_family_report_path = base_payload.with_name("report.json")
        current_family_report_path = current_payload.with_name("report.json")
        base_global_report = json.loads(
            base_family_report_path.read_text(encoding="utf-8")
        )
        current_global_report = json.loads(
            current_family_report_path.read_text(encoding="utf-8")
        )
        base_set_report = json.loads(base.report_path.read_text(encoding="utf-8"))
        current_set_report = json.loads(current.report_path.read_text(encoding="utf-8"))
        global_contract = _global_distribution_pair_contract(
            base_payload,
            current_payload,
            base_global_report,
            current_global_report,
            base_sample_ids=base.sample_ids,
            current_sample_ids=current.sample_ids,
            base_field_names=base.field_names,
            current_field_names=current.field_names,
            base_weight_selection=base_set_report.get("trace", {}).get("weight_selection"),
            current_weight_selection=current_set_report.get("trace", {}).get("weight_selection"),
        )
        if (
            current.generation_steps != base.generation_steps
            or current.evaluation_seed != base.evaluation_seed
        ):
            raise ValueError("base and post-training global-distribution geometry seeds differ")
        global_contract.update(
            dataset_fingerprint=current.dataset_fingerprint,
            generation_steps=current.generation_steps,
            evaluation_seed=current.evaluation_seed,
            sensor_manifest_sha256=current_manifest_sha,
        )
        base_specs = _global_distribution_specs(base_payload)
        current_specs = _global_distribution_specs(current_payload)
        if base_specs.keys() != current_specs.keys():
            raise ValueError("base and post-training global-distribution metrics differ")
        family_limits = {}
        family_scales = {}
        base_figure_paths = {}
        for name, base_spec in base_specs.items():
            current_spec = current_specs[name]
            if base_spec[1:3] != current_spec[1:3]:
                raise ValueError(f"global-distribution {name} comparison labels do not match")
            combined = np.concatenate(
                (base_spec[0].reshape(-1), current_spec[0].reshape(-1))
            )
            plot_scale = statistic_scale
            if (
                global_contract["definition"] == "marginal_copula_v2"
                and statistic_scale == "log"
                and not np.any(np.isfinite(combined) & (combined > 0.0))
            ):
                plot_scale = "linear"
            limits = _distribution_limits(
                combined,
                plot_scale,
            )
            family_limits[name] = list(limits)
            family_scales[name] = plot_scale
            post_path = current.output_dir / "coherence" / "global_distribution" / current_spec[3]
            base_path = _base_figure_path(post_path)
            render_coherence_distribution(
                base_spec[0],
                base_spec[1],
                base_spec[2],
                base_path,
                title=f"Base source — {base_spec[4]}",
                subtitle=_comparison_subtitle("Base source", base, scale=plot_scale),
                scale=plot_scale,
                value_limits=limits,
            )
            render_coherence_distribution(
                current_spec[0],
                current_spec[1],
                current_spec[2],
                post_path,
                title=f"Post-training — {current_spec[4]}",
                subtitle=_comparison_subtitle("Post-training", current, scale=plot_scale),
                scale=plot_scale,
                value_limits=limits,
            )
            base_figure_paths[name] = base_path
            artifacts["base"].setdefault("global_distribution", {})[name] = str(
                base_path.relative_to(current.output_dir)
            )
            artifacts["post_training"].setdefault("global_distribution", {})[name] = str(
                post_path.relative_to(current.output_dir)
            )
        shared_limits["global_distribution"] = family_limits
        shared_limits["global_distribution_scales"] = family_scales
        matched_coherence_contracts["global_distribution"] = global_contract

        # Retain the matched source's numerical evidence alongside its figures.
        # The source evaluation otherwise lives only in a TemporaryDirectory.
        global_destination = current.output_dir / "coherence" / "global_distribution"
        base_global_files = {
            "metrics_payload": (base_payload, global_destination / "metrics-base.npz"),
            "metrics_csv": (
                base.output_dir / "coherence" / "global_distribution" / "metrics.csv",
                global_destination / "metrics-base.csv",
            ),
            "report": (
                base.output_dir / "coherence" / "global_distribution" / "report.json",
                global_destination / "report-base.json",
            ),
        }
        for source_path, destination_path in base_global_files.values():
            shutil.copy2(source_path, destination_path)
        artifacts["base"]["global_distribution_metrics"] = {
            label: str(destination.relative_to(current.output_dir))
            for label, (_, destination) in base_global_files.items()
        }

        current_extra = current.coherence_accumulators.get("global_distribution")
        base_extra = base.coherence_accumulators.get("global_distribution")
        if current_extra is not None and getattr(current_extra, "extra_view", False):
            if base_extra is None or not getattr(base_extra, "extra_view", False):
                raise ValueError("base global-distribution extra view is unavailable")
            extra_ranges = current_extra.extra_value_ranges(base_extra)
            extra_density_limits = current_extra.extra_density_limits(extra_ranges, base_extra)
            extra_destination = (
                current.output_dir
                / "coherence"
                / "global_distribution"
                / "global_distribution_extra"
            )
            base_extra_output = base_extra.render_extra(
                extra_destination,
                split=base.split,
                checkpoint_label=base.checkpoint_label,
                run_label=base.run_label,
                value_ranges=extra_ranges,
                density_limits=extra_density_limits,
                filename_suffix="-base",
                role_label="Base source",
            )
            current_extra_output = current_extra.render_extra(
                extra_destination,
                split=current.split,
                checkpoint_label=current.checkpoint_label,
                run_label=current.run_label,
                value_ranges=extra_ranges,
                density_limits=extra_density_limits,
                role_label="Post-training",
            )
            artifacts["base"]["global_distribution_extra"] = {
                label: str(Path(path).relative_to(current.output_dir))
                for label, path in base_extra_output["figures"].items()
            }
            artifacts["post_training"]["global_distribution_extra"] = {
                label: str(Path(path).relative_to(current.output_dir))
                for label, path in current_extra_output["figures"].items()
            }
            shared_limits["global_distribution_extra"] = {
                label: {
                    "x": list(limits[0]),
                    "y": list(limits[1]),
                    "density": list(extra_density_limits[label]),
                }
                for label, limits in extra_ranges.items()
            }

        if global_contract["definition"] == "marginal_copula_v2":
            base_global_report_payload = deepcopy(base_global_report)
            base_artifacts = base_global_report_payload.setdefault("artifacts", {})
            base_artifacts["metrics_payload"] = "metrics-base.npz"
            base_artifacts["metrics_csv"] = "metrics-base.csv"
            base_artifacts["figures"] = {
                name: path.name for name, path in base_figure_paths.items()
            }
            base_artifacts["vector_figures"] = {
                name: {
                    fmt: path.with_suffix(f".{fmt}").name
                    for fmt in ("pdf", "svg")
                }
                for name, path in base_figure_paths.items()
            }
            if base_extra is not None and getattr(base_extra, "extra_view", False):
                base_artifacts["extra"] = {
                    "directory": _report_artifact_path(
                        Path(base_extra_output["directory"]), current.output_dir
                    ),
                    "report": _report_artifact_path(
                        Path(base_extra_output["report"]), current.output_dir
                    ),
                    "metrics_csv": _report_artifact_path(
                        Path(base_extra_output["metrics_csv"]), current.output_dir
                    ),
                    "metrics_payload": _report_artifact_path(
                        Path(base_extra_output["metrics_payload"]), current.output_dir
                    ),
                    "figures": {
                        label: _report_artifact_path(Path(path), current.output_dir)
                        for label, path in base_extra_output["figures"].items()
                    },
                    "jensen_shannon_divergence_bits": base_extra_output[
                        "jensen_shannon_divergence_bits"
                    ],
                }
            base_global_report_payload["paired_source_comparison"] = {
                "role": "base_source",
                "matched_contract": global_contract,
                "shared_limits": family_limits,
                "shared_scales": family_scales,
            }
            (global_destination / "report-base.json").write_text(
                json.dumps(base_global_report_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            current_global_report["paired_source_comparison"] = {
                "role": "post_training",
                "matched_contract": global_contract,
                "shared_limits": family_limits,
                "shared_scales": family_scales,
            }
            current_family_report_path.write_text(
                json.dumps(current_global_report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    if "topology" in coherence_families:
        base_topology_payload = base.output_dir / "coherence" / "topology" / "metrics.npz"
        current_topology_payload = current.output_dir / "coherence" / "topology" / "metrics.npz"
        base_topology, base_topology_report = _load_topology_comparison_data(
            base_topology_payload
        )
        current_topology, current_topology_report = _load_topology_comparison_data(
            current_topology_payload
        )
        _validate_topology_comparison(
            base_topology,
            current_topology,
            base_topology_report,
            current_topology_report,
        )

        topology_dir = current_topology_payload.parent
        base_topology_copy = topology_dir / "metrics-base.npz"
        base_topology_csv_copy = topology_dir / "metrics-base.csv"
        base_topology_report_copy = topology_dir / "report-base.json"
        shutil.copy2(base_topology_payload, base_topology_copy)
        shutil.copy2(base_topology_payload.with_name("metrics.csv"), base_topology_csv_copy)

        distance_values = np.column_stack(
            (
                base_topology["objective_component_distances"],
                base_topology["family_total_distance"].reshape(-1),
            )
        )
        current_distance_values = np.column_stack(
            (
                current_topology["objective_component_distances"],
                current_topology["family_total_distance"].reshape(-1),
            )
        )
        objective_labels = base_topology["objective_component_names"]
        distance_labels = tuple(name.removeprefix("topology.") for name in objective_labels) + (
            "component-weighted total before outer family weight",
        )
        distance_roles = (*("component" for _ in objective_labels), "component_weighted_total")
        if (
            current_topology["objective_component_names"] != objective_labels
            or distance_values.shape[1] != current_distance_values.shape[1]
        ):
            raise ValueError("base and post-training topology distance components do not match")
        distance_upper = max(float(distance_values.max()), float(current_distance_values.max()))
        distance_limits = (0.0, distance_upper * 1.13 if distance_upper > 0.0 else 1.0)
        current_topology_accumulator = current.coherence_accumulators.get("topology")
        base_topology_accumulator = base.coherence_accumulators.get("topology")
        if current_topology_accumulator is None or base_topology_accumulator is None:
            raise RuntimeError("topology accumulators are unavailable for paired visualization")
        if (
            tuple(current_topology_accumulator.sample_ids) != current_topology["sample_ids"]
            or tuple(base_topology_accumulator.sample_ids) != base_topology["sample_ids"]
        ):
            raise ValueError("topology accumulator samples do not match their saved metrics")
        base_cost = base_topology["family_total_distance"].reshape(-1)
        current_cost = current_topology["family_total_distance"].reshape(-1)
        base_ranks = np.empty(len(base_cost), dtype=np.int64)
        current_ranks = np.empty(len(current_cost), dtype=np.int64)
        base_ranks[np.argsort(base_cost, kind="stable")] = np.arange(len(base_cost))
        current_ranks[np.argsort(current_cost, kind="stable")] = np.arange(len(current_cost))
        representative_index = int(
            np.argsort(base_ranks + current_ranks, kind="stable")[len(base_cost) // 2]
        )
        representative_id = current_topology["sample_ids"][representative_index]
        reference_maps = np.asarray(
            current_topology_accumulator.reference_maps[representative_index]
        )
        base_reference_maps = np.asarray(
            base_topology_accumulator.reference_maps[representative_index]
        )
        if not np.array_equal(reference_maps, base_reference_maps):
            raise ValueError("base and post-training topology references differ for paired sample")
        fields = tuple(current_topology_accumulator.field_names)
        if fields != current_topology["field_names"]:
            raise ValueError("topology accumulator field order does not match saved metrics")
        betti_payloads = (
            base_topology["reference_betti"],
            base_topology["reconstruction_betti"],
            current_topology["reference_betti"],
            current_topology["reconstruction_betti"],
        )
        betti_limits_by_dimension = tuple(
            (
                0.0,
                max(
                    1.0,
                    float(
                        np.ceil(
                            max(float(values[:, :, dimension, :].max()) for values in betti_payloads)
                            * 1.08
                        )
                    ),
                ),
            )
            for dimension in range(2)
        )
        betti_limits_report = {
            f"h{dimension}": list(limits)
            for dimension, limits in enumerate(betti_limits_by_dimension)
        }
        filtration_metadata = [
            {
                "metric": metric,
                "kind": kind,
                "direction": direction,
                "line_index": None if int(line_index) < 0 else int(line_index),
                "line_weight": float(line_weight),
            }
            for metric, kind, direction, line_index, line_weight in zip(
                current_topology["filtration_metric"],
                current_topology["filtration_kind"],
                current_topology["filtration_direction"],
                current_topology["filtration_line_index"],
                current_topology["filtration_line_weight"],
            )
        ]
        metric_labels = tuple(dict.fromkeys(current_topology["filtration_metric"]))
        directions = tuple(dict.fromkeys(current_topology["filtration_direction"]))
        betti_path = topology_dir / "betti_curves.png"
        betti_base_path = _base_figure_path(betti_path)
        count_subtitle = (
            f"{current.split} · n={len(current.sample_ids)} · mean ±1 SD across snapshots · "
            f"fixed_shared query={int(current_topology['query_point_count']):,} · "
            f"{int(current_topology['configured_grid_shape'][0])}×"
            f"{int(current_topology['configured_grid_shape'][1])} configured raster"
        )
        for role, payload, destination in (
            ("Base source", base_topology, betti_base_path),
            ("Post-training", current_topology, betti_path),
        ):
            render_topology_betti_curves(
                payload["reference_betti"],
                payload["reconstruction_betti"],
                filtration_metadata,
                metric_labels,
                directions,
                payload["quantiles"],
                destination,
                title=f"{role} · exact Betti curves",
                subtitle=count_subtitle,
                dimension_y_limits=betti_limits_by_dimension,
            )

        topology_figure_path = topology_dir / "configured_grid_topology.png"
        topology_base_figure_path = _base_figure_path(topology_figure_path)
        paired_cost_summary = (
            f"paired median objective rank · source={base_cost[representative_index]:.4g} · "
            f"post={current_cost[representative_index]:.4g}"
        )
        for role, accumulator, destination, prediction_maps, cost in (
            (
                "Base source",
                base_topology_accumulator,
                topology_base_figure_path,
                base_topology_accumulator.reconstruction_maps[representative_index],
                base_cost[representative_index],
            ),
            (
                "Post-training",
                current_topology_accumulator,
                topology_figure_path,
                current_topology_accumulator.reconstruction_maps[representative_index],
                current_cost[representative_index],
            ),
        ):
            render_configured_grid_topology(
                reference_maps,
                prediction_maps,
                accumulator.grid_coordinates,
                fields,
                representative_id,
                units=str(current_topology_report["units"]),
                sample_epoch=f"{role} · {paired_cost_summary} · cost={cost:.4g}",
                output_path=destination,
                objective_distance=float(cost),
                role=role,
            )

        persistence_path = topology_dir / "persistence_term_distributions.png"
        persistence_base_path = _base_figure_path(persistence_path)
        for role, values, destination in (
            ("Base source", distance_values, persistence_base_path),
            ("Post-training", current_distance_values, persistence_path),
        ):
            render_topology_distance_distributions(
                values,
                distance_labels,
                distance_roles,
                destination,
                title=f"{role} · configured persistence distances",
                subtitle=(
                    f"{current.split} · n={len(current.sample_ids)} · paired {statistic_scale} "
                    "relative-$L_2$ sample selection"
                ),
                y_limits=distance_limits,
            )

        base_topology_report = deepcopy(base_topology_report)
        base_topology_report.setdefault("artifacts", {})["metrics_csv"] = base_topology_csv_copy.name
        base_topology_report["artifacts"]["metrics_payload"] = base_topology_copy.name
        base_topology_report["artifacts"]["figures"] = {
            key: _base_figure_path(topology_dir / filename).name
            for key, filename in base_topology_report["artifacts"].get("figures", {}).items()
        }
        base_topology_report["artifacts"]["vector_figures"] = {
            key: {
                "pdf": _base_figure_path(topology_dir / filename["pdf"]).name,
                "svg": _base_figure_path(topology_dir / filename["svg"]).name,
            }
            for key, filename in base_topology_report["artifacts"].get(
                "vector_figures", {}
            ).items()
        }
        base_topology_report["paired_source_comparison"] = {
            "role": "base_source",
            "representative_sample_id": representative_id,
            "shared_limits": {
                "betti_count_by_dimension": betti_limits_report,
                "persistence_distance": list(distance_limits),
            },
        }
        base_topology_report_copy.write_text(
            json.dumps(base_topology_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        current_topology_report["paired_source_comparison"] = {
            "role": "post_training",
            "representative_sample_id": representative_id,
            "shared_limits": {
                "betti_count_by_dimension": betti_limits_report,
                "persistence_distance": list(distance_limits),
            },
        }
        (topology_dir / "report.json").write_text(
            json.dumps(current_topology_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        artifacts["base"]["topology"] = {
            "betti_curves": str(betti_base_path.relative_to(current.output_dir)),
            "configured_grid_topology": str(
                topology_base_figure_path.relative_to(current.output_dir)
            ),
            "persistence_term_distributions": str(
                persistence_base_path.relative_to(current.output_dir)
            ),
            "metrics_csv": str(base_topology_csv_copy.relative_to(current.output_dir)),
            "metrics_payload": str(base_topology_copy.relative_to(current.output_dir)),
            "report": str(base_topology_report_copy.relative_to(current.output_dir)),
        }
        artifacts["post_training"]["topology"] = {
            key: str((topology_dir / filename).relative_to(current.output_dir))
            for key, filename in current_topology_report["artifacts"]["figures"].items()
        }
        artifacts["post_training"]["topology"].update(
            {
                "metrics_csv": str((topology_dir / "metrics.csv").relative_to(current.output_dir)),
                "metrics_payload": str(current_topology_payload.relative_to(current.output_dir)),
                "report": str((topology_dir / "report.json").relative_to(current.output_dir)),
            }
        )
        shared_limits["topology"] = {
            "betti_count_by_dimension": betti_limits_report,
            "persistence_distance": list(distance_limits),
            "representative_sample_id": representative_id,
        }
        matched_coherence_contracts["topology"] = {
            "sample_ids": list(current_topology["sample_ids"]),
            "field_names": list(current_topology["field_names"]),
            "query_contract": current_topology_report["query_contract"],
            "raster": current_topology_report["raster"],
            "settings": current_topology_report["objective"]["settings"],
        }

    cross_spectrum_v4 = False
    if "cross_spectrum" in coherence_families:
        base_cross_report_path = base.output_dir / "coherence" / "cross_spectrum" / "report.json"
        current_cross_report_path = (
            current.output_dir / "coherence" / "cross_spectrum" / "report.json"
        )
        base_cross_report = json.loads(base_cross_report_path.read_text(encoding="utf-8"))
        current_cross_report = json.loads(
            current_cross_report_path.read_text(encoding="utf-8")
        )
        base_definition = base_cross_report.get("definition", "legacy_v3")
        current_definition = current_cross_report.get("definition", "legacy_v3")
        if base_definition != current_definition:
            raise ValueError("base and post-training cross-spectrum definitions do not match")
        cross_spectrum_v4 = current_definition == "second_order_blocks_v4"
        if cross_spectrum_v4:
            _render_second_order_spectrum_comparison(
                current,
                base,
                shared_limits=shared_limits,
                artifacts=artifacts,
                matched_coherence_contracts=matched_coherence_contracts,
            )

    if "cross_spectrum" in coherence_families and not cross_spectrum_v4:
        base_payload = base.output_dir / "coherence" / "cross_spectrum" / "metrics.npz"
        current_payload = current.output_dir / "coherence" / "cross_spectrum" / "metrics.npz"
        current_cross_dir = current_payload.parent
        base_csv = base_payload.with_name("metrics.csv")
        base_report_path = base_payload.with_name("report.json")
        base_payload_copy = current_cross_dir / "metrics-base.npz"
        base_csv_copy = current_cross_dir / "metrics-base.csv"
        base_report_copy = current_cross_dir / "report-base.json"
        shutil.copy2(base_payload, base_payload_copy)
        shutil.copy2(base_csv, base_csv_copy)
        base_report_payload = json.loads(base_report_path.read_text(encoding="utf-8"))
        base_report_payload["artifacts"]["metrics_payload"] = base_payload_copy.name
        base_report_payload["artifacts"]["metrics_csv"] = base_csv_copy.name
        base_report_payload["artifacts"]["figures"] = {
            key: _base_figure_path(current_cross_dir / filename).name
            for key, filename in base_report_payload["artifacts"]["figures"].items()
        }
        base_report_payload["artifacts"]["vector_figures"] = {
            key: {
                format_name: _base_figure_path(current_cross_dir / filename).name
                for format_name, filename in formats.items()
            }
            for key, formats in base_report_payload["artifacts"].get(
                "vector_figures", {}
            ).items()
        }
        base_report_copy.write_text(
            json.dumps(base_report_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        base_specs = _cross_spectrum_specs(base_payload)
        current_specs = _cross_spectrum_specs(current_payload)
        displayed_score_ranges = {}
        for name, base_spec in base_specs.items():
            current_spec = current_specs[name]
            if base_spec[2:4] != current_spec[2:4]:
                raise ValueError(f"cross-spectrum {name} comparison labels do not match")
            score_limits = adaptive_coherence_score_limits(
                np.concatenate((base_spec[0], current_spec[0])),
                np.concatenate((base_spec[1], current_spec[1])),
            )
            displayed_score_ranges[name] = list(score_limits)
            post_path = current.output_dir / "coherence" / "cross_spectrum" / current_spec[4]
            base_path = _base_figure_path(post_path)
            render_cross_spectrum_score_bars(
                base_spec[0],
                base_spec[2],
                base_spec[3],
                base_path,
                title=f"Base source — {base_spec[5]}",
                subtitle=_cross_spectrum_comparison_subtitle("Base source", base, base_payload),
                score_std=base_spec[1],
                score_limits=score_limits,
            )
            render_cross_spectrum_score_bars(
                current_spec[0],
                current_spec[2],
                current_spec[3],
                post_path,
                title=f"Post-training — {current_spec[5]}",
                subtitle=_cross_spectrum_comparison_subtitle(
                    "Post-training", current, current_payload
                ),
                score_std=current_spec[1],
                score_limits=score_limits,
            )
            artifacts["base"].setdefault("cross_spectrum", {})[name] = str(
                base_path.relative_to(current.output_dir)
            )
            artifacts["post_training"].setdefault("cross_spectrum", {})[name] = str(
                post_path.relative_to(current.output_dir)
            )
        shared_limits["cross_spectrum"] = {
            "coherence_score": [0.0, 1.0],
            "displayed_coherence_score_by_component": displayed_score_ranges,
        }
        artifacts["base"].setdefault("cross_spectrum", {}).update(
            {
                "metrics_csv": str(base_csv_copy.relative_to(current.output_dir)),
                "metrics_payload": str(base_payload_copy.relative_to(current.output_dir)),
                "report": str(base_report_copy.relative_to(current.output_dir)),
            }
        )

        base_profile, base_profile_report = _load_band_profile_comparison_data(base_payload)
        current_profile, current_profile_report = _load_band_profile_comparison_data(
            current_payload
        )
        _validate_band_profile_comparison(
            base_profile,
            current_profile,
            base_profile_report,
            current_profile_report,
        )
        base_profile_csv_copy = current_cross_dir / "band_energy_profile-base.csv"
        shutil.copy2(base_payload.with_name("band_energy_profile.csv"), base_profile_csv_copy)
        profile_path = current_cross_dir / "spectral_band_profiles.png"
        profile_base_path = _base_figure_path(profile_path)
        profile_subtitle = (
            f"{current.split} · n={len(base_profile['sample_ids'])} paired used samples · "
            f"{base_profile['aggregation']} ensembles · mean ±1 SD · shared fraction scale 0–1"
        )
        for role, profile, destination, report_payload in (
            ("Base source", base_profile, profile_base_path, base_profile_report),
            ("Post-training", current_profile, profile_path, current_profile_report),
        ):
            render_cross_spectrum_band_profiles(
                profile["graph_band_energy_fraction_reference_mean"],
                profile["graph_band_energy_fraction_reconstruction_mean"],
                profile["graph_band_names"],
                profile["field_names"],
                destination,
                reference_std=profile["graph_band_energy_fraction_reference_std"],
                reconstruction_std=profile[
                    "graph_band_energy_fraction_reconstruction_std"
                ],
                title=f"{role} · graph spectral energy by band",
                subtitle=profile_subtitle,
            )
            if role == "Base source":
                report_payload["paired_source_comparison"] = {
                    "role": "base_source",
                    "shared_fraction_limits": [0.0, 1.0],
                }
            else:
                report_path = current_payload.with_name("report.json")
                current_cross_report = json.loads(report_path.read_text(encoding="utf-8"))
                current_cross_report["paired_source_comparison"] = {
                    "role": "post_training",
                    "shared_fraction_limits": [0.0, 1.0],
                }
                report_path.write_text(
                    json.dumps(current_cross_report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        base_report_payload["paired_source_comparison"] = {
            "role": "base_source",
            "shared_fraction_limits": [0.0, 1.0],
        }
        base_report_payload.setdefault("artifacts", {})[
            "band_energy_profile_csv"
        ] = base_profile_csv_copy.name
        base_report_copy.write_text(
            json.dumps(base_report_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts["base"]["cross_spectrum"].update(
            {
                "band_energy_profile": str(profile_base_path.relative_to(current.output_dir)),
                "band_energy_profile_csv": str(
                    base_profile_csv_copy.relative_to(current.output_dir)
                ),
            }
        )
        artifacts["post_training"]["cross_spectrum"]["band_energy_profile"] = str(
            profile_path.relative_to(current.output_dir)
        )
        shared_limits["cross_spectrum"]["graph_band_energy_fraction"] = [0.0, 1.0]
        matched_coherence_contracts["cross_spectrum"] = {
            "field_names": list(current_profile["field_names"]),
            "bands": list(current_profile["graph_band_names"]),
            "band_field_labels": list(current_profile["band_field_labels"]),
            "sample_ids": list(current_profile["sample_ids"]),
            "selected_sample_ids": list(current_profile["selected_sample_ids"]),
            "units": current_profile_report.get("units"),
            "graph": current_profile_report.get("graph", {}),
        }

    report = {
        "kind": "post_training_source_comparison",
        "split": current.split,
        "sample_count": len(current.sample_ids),
        "statistic_scale": statistic_scale,
        "matched_inputs": {
            "dataset_fingerprint": current.dataset_fingerprint,
            "sample_ids": list(current.sample_ids),
            "sensor_manifest_sha256": current_manifest_sha,
            "generation_steps": current.generation_steps,
            "evaluation_seed": current.evaluation_seed,
        },
        "runs": {
            "base": {
                "run": str(base.run_dir),
                "checkpoint": str(base.checkpoint_path),
            },
            "post_training": {
                "run": str(current.run_dir),
                "checkpoint": str(current.checkpoint_path),
            },
        },
        "shared_axis_limits": shared_limits,
        "matched_coherence_contracts": matched_coherence_contracts,
        "artifacts": artifacts,
    }
    report_path = current.output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    current_report = json.loads(current.report_path.read_text(encoding="utf-8"))
    current_report["comparison"] = {
        "enabled": True,
        "source_run": str(base.run_dir),
        "source_checkpoint": str(base.checkpoint_path),
        "report": str(report_path.relative_to(current.run_dir)),
    }
    current.report_path.write_text(
        json.dumps(current_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report_path


def _posttraining_source(config: dict[str, Any], run_dir: Path) -> tuple[Path, Path] | None:
    if config.get("stage") != "post_training":
        return None
    if not config.get("source_run") or not config.get("source_checkpoint"):
        raise ValueError("post-training statistical comparison requires source lineage")
    source_run = Path(str(config["source_run"])).resolve()
    checkpoint_path = source_checkpoint_path(config).resolve()
    if not source_run.is_dir() or not checkpoint_path.is_file():
        raise FileNotFoundError("post-training source run or checkpoint is unavailable")
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if Path(str(manifest.get("parent_run", source_run))).resolve() != source_run:
            raise ValueError("post-training run manifest parent disagrees with resolved config")
        if (
            Path(str(manifest.get("source_checkpoint", checkpoint_path))).resolve()
            != checkpoint_path
        ):
            raise ValueError("post-training source checkpoint lineage is inconsistent")
        expected_sha = manifest.get("source_hashes", {}).get("checkpoint")
        if expected_sha and file_sha256(checkpoint_path) != expected_sha:
            raise ValueError("post-training source checkpoint hash has changed")
    return source_run, checkpoint_path


def evaluate_reconstruction_set(
    run_dir: str | Path,
    *,
    case_dir: str | Path,
    split: str,
    checkpoint: str = "best",
    sensor_config: str | Path | None = None,
    sensor_manifest: str | Path | None = None,
    generation_steps: int | None = None,
    device_name: str | None = None,
    output_path: str | Path | None = None,
    weight_selection: str = "configured",
    max_samples: int | None = 200,
    coherence_families: tuple[str, ...] | list[str] | None = None,
    extra_coherence_views: bool = False,
    cross_spectrum_aggregation: str = "training_aligned",
    statistic_scale: str = "log",
    compare_source: bool = True,
) -> Path:
    """Evaluate one run and automatically compare post-training runs with their source."""
    run_dir = Path(run_dir).resolve()
    case_dir = Path(case_dir).resolve()
    run_config = load_config(run_dir / "resolved_config.yaml")
    if (
        run_config.get("optimization", {}).get("update_policy") == "coherence_primal_dual"
        and weight_selection == "configured"
    ):
        # The parent may have an EMA evaluation default; this recipe is live/live.
        weight_selection = "live"
    families = tuple(
        dict.fromkeys(str(name).strip().lower() for name in (coherence_families or ()))
    )
    source = _posttraining_source(run_config, run_dir) if compare_source else None
    current = _evaluate_reconstruction_set_once(
        run_dir,
        case_dir=case_dir,
        split=split,
        checkpoint=checkpoint,
        sensor_config=sensor_config,
        sensor_manifest=sensor_manifest,
        generation_steps=generation_steps,
        device_name=device_name,
        output_path=output_path,
        weight_selection=weight_selection,
        max_samples=max_samples,
        coherence_families=families,
        extra_coherence_views=extra_coherence_views,
        cross_spectrum_aggregation=cross_spectrum_aggregation,
        statistic_scale=statistic_scale,
    )
    if source is None:
        if "topology" in families:
            from .explanatory_coherence import render_saved_coherence_explanatory

            render_saved_coherence_explanatory(current.output_dir, families=families)
        return current.figure_path

    source_run, source_checkpoint = source
    with TemporaryDirectory(
        prefix="phycoflow-base-comparison-", dir=current.output_dir
    ) as temporary_dir:
        base = _evaluate_reconstruction_set_once(
            source_run,
            case_dir=case_dir,
            split=split,
            checkpoint=str(source_checkpoint),
            sensor_config=sensor_config,
            sensor_manifest=sensor_manifest,
            generation_steps=current.generation_steps,
            device_name=device_name,
            output_path=None,
            weight_selection=weight_selection,
            max_samples=max_samples,
            coherence_families=families,
            extra_coherence_views=extra_coherence_views,
            cross_spectrum_aggregation=cross_spectrum_aggregation,
            statistic_scale=statistic_scale,
            output_dir_override=Path(temporary_dir),
            coherence_config_override=run_config.get("coherence"),
            coherence_artifact_run_dir=run_dir,
            evaluation_seed_override=current.evaluation_seed,
        )
        _render_posttraining_comparison(
            current,
            base,
            coherence_families=families,
            statistic_scale=statistic_scale,
        )
        if {"cross_spectrum", "topology"} & set(families):
            from .explanatory_coherence import render_saved_coherence_explanatory

            render_saved_coherence_explanatory(current.output_dir, families=families)
    return current.figure_path
