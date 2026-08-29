"""Streaming whole-split reconstruction evaluation and distribution visualization."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from tqdm.auto import tqdm

from ..data.manifest import SensorManifest, build_batch_from_manifest, dataset_fingerprint
from ..data.sensor_protocols import build_observation_batch
from ..training.common import sensor_protocol_from_config
from ..training.model_lifecycle import selected_evaluation_weight_context
from ..training.run_store import file_sha256
from .checkpoint import load_evaluation_runtime
from .reconstruction_visualization import warn_if_cuda_memory_tight


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


def _publication_field_label(name: str) -> str:
    indexed = re.fullmatch(r"([A-Za-z]+)_([0-9]+)", name)
    if indexed:
        return rf"${indexed.group(1)}_{{{indexed.group(2)}}}$"
    trailing_index = re.fullmatch(r"(.+?)([0-9]+)", name)
    if trailing_index:
        return rf"{trailing_index.group(1)}$_{{{trailing_index.group(2)}}}$"
    return name


def render_reconstruction_set_distribution(
    payload_path: str | Path,
    output_path: str | Path,
    *,
    title: str | None = None,
    dpi: int = 300,
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
    finite_errors = errors[np.isfinite(errors)]
    upper = float(finite_errors.max()) * 1.07 if finite_errors.size else 1.0
    axis.set_ylim(0.0, max(upper, np.finfo(np.float64).eps))
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
) -> Path:
    """Stream a complete split through one loaded model and plot field-wise error statistics."""
    run_dir = Path(run_dir).resolve()
    case_dir = Path(case_dir).resolve()
    checkpoint_label = Path(checkpoint).stem
    output_dir = run_dir / "evaluation" / f"reconstruction_set_{split}_{checkpoint_label}"
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
    dataset = runtime.dataset
    available_sample_count = len(dataset)
    if available_sample_count < 1:
        dataset.close()
        raise ValueError(f"split {split!r} contains no samples")
    selected_indices = _select_sample_indices(available_sample_count, max_samples)
    sample_count = int(selected_indices.size)
    input_manifest = SensorManifest.load(sensor_manifest) if sensor_manifest is not None else None
    field_names = tuple(dataset.field_names)
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
                        "dataset_fingerprint": dataset_fingerprint(dataset.path),
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
                sample_errors = _physical_relative_l2(
                    reconstruction.prediction,
                    target,
                    dataset.normalizer,
                )[0].cpu().numpy()
                errors[output_index] = sample_errors
                sample_ids.append(batch.sample_ids[0])
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
    )
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
            "checkpoint": str(runtime.checkpoint_path.relative_to(run_dir)),
            "checkpoint_sha256": file_sha256(runtime.checkpoint_path),
            "dataset_fingerprint": dataset_fingerprint(dataset.path),
            "weight_selection": weight_selection,
            "generation_steps": runtime.generation_steps,
            "generation_seed_policy": "base_seed_plus_split_relative_sample_index",
            "sensor_manifest": str(manifest_path.relative_to(run_dir)),
            "sensor_manifest_sha256": file_sha256(manifest_path),
            "metrics_csv": str(csv_path.relative_to(run_dir)),
            "metrics_payload": str(payload_path.relative_to(run_dir)),
            "figure": str(figure_path),
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return figure_path
