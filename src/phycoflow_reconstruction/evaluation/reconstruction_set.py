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
    build_coherence_accumulators,
    render_coherence_distribution,
    render_cross_spectrum_score_bars,
)
from .reconstruction_visualization import warn_if_cuda_memory_tight


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

    figure_width = max(7.2, 1.15 * len(field_names) + 1.9)
    figure, axis = plt.subplots(figsize=(figure_width, 4.7), layout="constrained")
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    positions = np.arange(1, len(field_names) + 1, dtype=np.float64)
    rng = np.random.default_rng(2027)
    okabe_ito = (
        "#0072B2",
        "#E69F00",
        "#009E73",
        "#CC79A7",
        "#56B4E9",
        "#D55E00",
        "#F0E442",
        "#000000",
    )
    colors = tuple(okabe_ito[index % len(okabe_ito)] for index in range(len(field_names)))
    finite_by_field = []
    finite_positions = []
    for field_index in range(len(field_names)):
        finite = errors[:, field_index]
        finite = finite[np.isfinite(finite)]
        if scale == "log":
            finite = finite[finite > 0.0]
        if finite.size >= 2:
            finite_by_field.append(finite)
            finite_positions.append(positions[field_index])
        jitter = rng.uniform(-0.12, 0.12, size=finite.size)
        axis.scatter(
            np.full(finite.size, positions[field_index]) + jitter,
            finite,
            s=11,
            alpha=0.48,
            color=colors[field_index],
            edgecolors="white",
            linewidths=0.25,
            zorder=3,
            rasterized=True,
        )
        if finite.size:
            axis.scatter(
                positions[field_index],
                np.median(finite),
                marker="D",
                s=31,
                linewidths=0.9,
                facecolors="white",
                edgecolors="black",
                zorder=4,
            )
        else:
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
            widths=0.68,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, position in zip(violin["bodies"], finite_positions):
            field_index = int(position - 1)
            body.set_facecolor(colors[field_index])
            body.set_edgecolor(colors[field_index])
            body.set_alpha(0.22)
            body.set_linewidth(1.25)

    axis.set_xticks(
        positions,
        tuple(_publication_field_label(name) for name in field_names),
        fontsize=10.5,
    )
    axis.set_xlim(0.35, len(field_names) + 0.65)
    axis.set_yscale(scale)
    limits = value_limits or _distribution_limits(errors, scale)
    if scale == "log" and limits[0] <= 0.0:
        raise ValueError("log-scale reconstruction limits must be positive")
    if not np.isfinite(limits).all() or limits[0] >= limits[1]:
        raise ValueError("reconstruction plot limits must be finite and increasing")
    axis.set_ylim(*limits)
    axis.set_ylabel(
        r"Relative $L_2$ error after physical-unit decoding",
        fontsize=10.5,
        labelpad=7.0,
    )
    axis.set_title(
        title or f"Reconstruction quality distribution — {split} set",
        fontsize=12.0,
        fontweight="medium",
        pad=9.0,
    )
    axis.tick_params(axis="y", labelsize=9.5, width=0.8, length=4)
    axis.tick_params(axis="x", width=0.8, length=4, pad=5)
    axis.grid(axis="y", color="0.86", linewidth=0.65, linestyle="--", dashes=(3, 2))
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.legend(
        handles=(
            Patch(facecolor="0.55", edgecolor="0.35", alpha=0.22, label="Density"),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=4.2,
                markerfacecolor="0.45",
                markeredgecolor="white",
                markeredgewidth=0.3,
                label="Sample",
            ),
            Line2D(
                [],
                [],
                marker="D",
                linestyle="none",
                markersize=4.5,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.8,
                label="Median",
            ),
        ),
        loc="upper left",
        ncol=3,
        frameon=False,
        fontsize=8.5,
        handlelength=1.2,
        columnspacing=1.1,
        borderaxespad=0.5,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
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
        runtime.config["coherence"] = deepcopy(coherence_config_override)
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
        pair_labels = tuple(str(value) for value in payload["pair_labels"])
        band_labels = tuple(str(value) for value in payload["band_field_labels"])
        arrays = {
            "same_frequency": np.asarray(
                payload["same_frequency_coherence_score"], dtype=np.float64
            ),
            "cross_frequency": np.asarray(
                payload["cross_frequency_coherence_score"], dtype=np.float64
            ),
            "band_energy": np.asarray(payload["band_energy_coherence_score"], dtype=np.float64),
        }
        spread_arrays = {
            "same_frequency": np.asarray(
                payload["same_frequency_coherence_score_std"], dtype=np.float64
            ),
            "cross_frequency": np.asarray(
                payload["cross_frequency_coherence_score_std"], dtype=np.float64
            ),
            "band_energy": np.asarray(payload["band_energy_coherence_score_std"], dtype=np.float64),
        }
    definitions = {
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
        f"{role} · {result.run_label.replace('_', ' ')} · {result.split} · "
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
    reconstruction_limits = _distribution_limits(
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
    if "global_distribution" in coherence_families:
        base_payload = base.output_dir / "coherence" / "global_distribution" / "metrics.npz"
        current_payload = current.output_dir / "coherence" / "global_distribution" / "metrics.npz"
        base_specs = _global_distribution_specs(base_payload)
        current_specs = _global_distribution_specs(current_payload)
        family_limits = {}
        for name, base_spec in base_specs.items():
            current_spec = current_specs[name]
            if base_spec[1:3] != current_spec[1:3]:
                raise ValueError(f"global-distribution {name} comparison labels do not match")
            limits = _distribution_limits(
                np.concatenate((base_spec[0].reshape(-1), current_spec[0].reshape(-1))),
                statistic_scale,
            )
            family_limits[name] = list(limits)
            post_path = current.output_dir / "coherence" / "global_distribution" / current_spec[3]
            base_path = _base_figure_path(post_path)
            render_coherence_distribution(
                base_spec[0],
                base_spec[1],
                base_spec[2],
                base_path,
                title=f"Base source — {base_spec[4]}",
                subtitle=_comparison_subtitle("Base source", base, scale=statistic_scale),
                scale=statistic_scale,
                value_limits=limits,
            )
            render_coherence_distribution(
                current_spec[0],
                current_spec[1],
                current_spec[2],
                post_path,
                title=f"Post-training — {current_spec[4]}",
                subtitle=_comparison_subtitle("Post-training", current, scale=statistic_scale),
                scale=statistic_scale,
                value_limits=limits,
            )
            artifacts["base"].setdefault("global_distribution", {})[name] = str(
                base_path.relative_to(current.output_dir)
            )
            artifacts["post_training"].setdefault("global_distribution", {})[name] = str(
                post_path.relative_to(current.output_dir)
            )
        shared_limits["global_distribution"] = family_limits

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

    if "cross_spectrum" in coherence_families:
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
        base_report_copy.write_text(
            json.dumps(base_report_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        base_specs = _cross_spectrum_specs(base_payload)
        current_specs = _cross_spectrum_specs(current_payload)
        for name, base_spec in base_specs.items():
            current_spec = current_specs[name]
            if base_spec[2:4] != current_spec[2:4]:
                raise ValueError(f"cross-spectrum {name} comparison labels do not match")
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
            )
            artifacts["base"].setdefault("cross_spectrum", {})[name] = str(
                base_path.relative_to(current.output_dir)
            )
            artifacts["post_training"].setdefault("cross_spectrum", {})[name] = str(
                post_path.relative_to(current.output_dir)
            )
        shared_limits["cross_spectrum"] = {"coherence_score": [0.0, 1.0]}
        artifacts["base"].setdefault("cross_spectrum", {}).update(
            {
                "metrics_csv": str(base_csv_copy.relative_to(current.output_dir)),
                "metrics_payload": str(base_payload_copy.relative_to(current.output_dir)),
                "report": str(base_report_copy.relative_to(current.output_dir)),
            }
        )

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
        return current.figure_path

    source_run, source_checkpoint = source
    with TemporaryDirectory(prefix="phycoflow-base-comparison-") as temporary_dir:
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
            evaluation_seed_override=current.evaluation_seed,
        )
        _render_posttraining_comparison(
            current,
            base,
            coherence_families=families,
            statistic_scale=statistic_scale,
        )
    return current.figure_path
