"""One-command checkpoint reconstruction visualization."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from ..config import load_config
from .checkpoint import evaluate_run


def _relative_l2_error(estimate: np.ndarray, truth: np.ndarray) -> float | None:
    estimate64 = np.asarray(estimate, dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    denominator = float(np.linalg.norm(truth64.ravel()))
    if denominator == 0.0 or not np.isfinite(denominator):
        return None
    value = float(np.linalg.norm((estimate64 - truth64).ravel()) / denominator)
    return value if np.isfinite(value) else None


def _error_title(relative_l2: float | None) -> str:
    metric = "N/A" if relative_l2 is None else f"{relative_l2:.3e}"
    return f"Absolute error\nRelative $L_2$ = {metric}"


def _nondegenerate_range(low: float, high: float) -> tuple[float, float]:
    if high > low:
        return low, high
    padding = max(abs(low), 1.0) * 1.0e-6
    return low - padding, high + padding


def _checkpoint_path(run_dir: Path, checkpoint: str) -> Path:
    candidate = Path(checkpoint)
    if candidate.is_absolute():
        return candidate
    name = candidate.name if candidate.suffix == ".pt" else f"{candidate.name}.pt"
    return run_dir / "checkpoints" / name


def warn_if_cuda_memory_tight(
    run_dir: str | Path,
    *,
    checkpoint: str,
    device_name: str | None,
) -> bool:
    """Warn before inference when available CUDA memory is below a conservative estimate."""
    run_dir = Path(run_dir).resolve()
    config = load_config(run_dir / "resolved_config.yaml")
    device = torch.device(device_name or config.get("runtime", {}).get("device", "cpu"))
    if device.type != "cuda" or not torch.cuda.is_available():
        return False

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    checkpoint_path = _checkpoint_path(run_dir, checkpoint)
    checkpoint_bytes = checkpoint_path.stat().st_size if checkpoint_path.is_file() else 0
    fallback_bytes = 2 * 1024**3 + 4 * checkpoint_bytes
    estimate_bytes = fallback_bytes
    estimate_source = "checkpoint-size inference reserve"
    status_path = run_dir / "status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        recorded_peak = int(status.get("peak_cuda_memory_bytes", 0) or 0)
        if recorded_peak > estimate_bytes:
            estimate_bytes = recorded_peak
            estimate_source = "recorded training peak (conservative for inference)"

    required_free = int(estimate_bytes * 1.10)
    if free_bytes >= required_free:
        return False

    gib = 1024**3
    print(
        "WARNING: CUDA memory may be tight before full-grid reconstruction: "
        f"{free_bytes / gib:.2f} GiB free of {total_bytes / gib:.2f} GiB on {device}; "
        f"estimated safe free memory is {required_free / gib:.2f} GiB from {estimate_source}. "
        "Consider rerunning with --device cuda:<index> for a less-loaded GPU or --device cpu.",
        file=sys.stderr,
        flush=True,
    )
    return True


def render_reconstruction_payload(
    payload_path: str | Path,
    output_path: str | Path,
    *,
    title: str | None = None,
    dpi: int = 300,
    contour_levels: int = 20,
) -> Path:
    """Render physical target, reconstruction, and error fields from an evaluation payload."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    payload_path = Path(payload_path)
    output_path = Path(output_path)
    with np.load(payload_path, allow_pickle=False) as payload:
        prediction = np.asarray(payload["prediction_physical"])[0]
        target = np.asarray(payload["target_physical"])[0]
        query_coords_physical = np.asarray(payload["query_coords_physical"])[0]
        logical_shape = tuple(int(value) for value in payload["logical_shape"])
        field_names = tuple(str(value) for value in payload["field_names"])
        obs_indices = np.asarray(payload["obs_indices"])[0]
        obs_fields = np.asarray(payload["obs_field_ids"])[0]
        obs_valid = np.asarray(payload["obs_valid_mask"])[0].astype(bool)
        sample_id = str(payload["sample_id"])

    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes do not match")
    if prediction.shape[0] != math.prod(logical_shape):
        raise ValueError(
            "reconstruction visualization requires full-grid query points; "
            f"got {prediction.shape[0]} points for logical shape {logical_shape}"
        )
    if len(logical_shape) != 2:
        raise ValueError("reconstruction visualization currently requires a two-dimensional grid")
    if query_coords_physical.shape != (prediction.shape[0], 2):
        raise ValueError("physical query coordinates must have shape [grid_points, 2]")
    if contour_levels < 2:
        raise ValueError("contour_levels must be at least 2")

    x_grid = query_coords_physical[:, 0].reshape(logical_shape)
    y_grid = query_coords_physical[:, 1].reshape(logical_shape)
    x_span = float(x_grid.max() - x_grid.min())
    y_span = float(y_grid.max() - y_grid.min())
    if x_span <= 0.0 or y_span <= 0.0:
        raise ValueError("physical query coordinates must span both grid axes")
    domain_aspect = x_span / y_span
    panel_width = 4.0
    panel_height = panel_width / domain_aspect
    row_text_allowance = 0.52
    figure_width = 3.0 * panel_width + 2.8
    figure_height = max(
        3.0,
        len(field_names) * (panel_height + row_text_allowance) + 0.75,
    )
    font_scale = max(0.82, min(1.0, 6.0 / max(len(field_names), 1)))
    title_fontsize = 10.5 * font_scale
    label_fontsize = 10.0 * font_scale
    tick_fontsize = 8.0 * font_scale

    plt.rcParams["svg.fonttype"] = "none"
    figure = plt.figure(figsize=(figure_width, figure_height), layout="constrained")
    grid = figure.add_gridspec(
        len(field_names),
        5,
        width_ratios=(1.0, 1.0, 0.045, 1.0, 0.045),
        hspace=0.12,
        wspace=0.16,
    )
    axes = np.empty((len(field_names), 3), dtype=object)
    field_colorbar_axes = []
    error_colorbar_axes = []
    for field_index in range(len(field_names)):
        axes[field_index, 0] = figure.add_subplot(grid[field_index, 0])
        axes[field_index, 1] = figure.add_subplot(grid[field_index, 1])
        field_colorbar_axes.append(figure.add_subplot(grid[field_index, 2]))
        axes[field_index, 2] = figure.add_subplot(grid[field_index, 3])
        error_colorbar_axes.append(figure.add_subplot(grid[field_index, 4]))

    for field_index, field_name in enumerate(field_names):
        truth = target[:, field_index]
        estimate = prediction[:, field_index]
        error = np.abs(estimate - truth)
        low = float(min(truth.min(), estimate.min()))
        high = float(max(truth.max(), estimate.max()))
        low, high = _nondegenerate_range(low, high)
        error_high = max(float(error.max()), np.finfo(np.float64).eps)
        field_contour_levels = np.linspace(low, high, contour_levels)
        error_contour_levels = np.linspace(0.0, error_high, contour_levels)
        field_colorbar_ticks = np.linspace(low, high, 4)
        error_colorbar_ticks = np.linspace(0.0, error_high, 4)
        field_norm = Normalize(vmin=low, vmax=high)
        error_norm = Normalize(vmin=0.0, vmax=error_high)
        panels = (
            (
                truth.reshape(logical_shape),
                "Ground truth",
                "viridis",
                field_contour_levels,
                field_norm,
            ),
            (
                estimate.reshape(logical_shape),
                "Reconstruction",
                "viridis",
                field_contour_levels,
                field_norm,
            ),
            (
                error.reshape(logical_shape),
                _error_title(_relative_l2_error(estimate, truth)),
                "magma",
                error_contour_levels,
                error_norm,
            ),
        )
        for column_index, (values, panel_title, cmap, fill_levels, norm) in enumerate(panels):
            axis = axes[field_index, column_index]
            axis.contourf(
                x_grid,
                y_grid,
                values,
                levels=fill_levels,
                cmap=cmap,
                norm=norm,
                extend="max" if column_index == 2 else "both",
            )
            line_low = float(values.min())
            line_high = float(values.max())
            line_low, line_high = _nondegenerate_range(line_low, line_high)
            axis.contour(
                x_grid,
                y_grid,
                values,
                levels=np.linspace(line_low, line_high, contour_levels),
                colors="lightgrey",
                linewidths=0.35,
                alpha=0.75,
            )
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlim(float(x_grid.min()), float(x_grid.max()))
            axis.set_ylim(float(y_grid.min()), float(y_grid.max()))
            axis.set_title(panel_title, fontsize=title_fontsize, pad=4.0)
            axis.tick_params(labelsize=tick_fontsize, pad=2.0)
        field_colorbar = figure.colorbar(
            ScalarMappable(norm=field_norm, cmap="viridis"),
            cax=field_colorbar_axes[field_index],
            ticks=field_colorbar_ticks,
        )
        error_colorbar = figure.colorbar(
            ScalarMappable(norm=error_norm, cmap="magma"),
            cax=error_colorbar_axes[field_index],
            ticks=error_colorbar_ticks,
        )
        field_colorbar.ax.tick_params(labelsize=tick_fontsize, pad=2.0)
        error_colorbar.ax.tick_params(labelsize=tick_fontsize, pad=2.0)
        axes[field_index, 0].set_ylabel(field_name, fontsize=label_fontsize, labelpad=4.0)

        sensor_mask = obs_valid & (obs_fields == field_index)
        if sensor_mask.any():
            point_ids = obs_indices[sensor_mask]
            axes[field_index, 0].scatter(
                query_coords_physical[point_ids, 0],
                query_coords_physical[point_ids, 1],
                s=8,
                facecolors="none",
                edgecolors="white",
                linewidths=0.5,
                label="observations",
            )
            axes[field_index, 0].legend(
                loc="best",
                frameon=False,
                fontsize=max(6.5, 7.5 * font_scale),
            )

    figure.suptitle(
        title or f"Sparse reconstruction — {sample_id}",
        fontsize=12.5 * font_scale,
    )
    # Constrained layout reserves dedicated colorbar columns, while equal physical
    # aspect may shorten the contour axes inside each row. Align colorbars to the
    # final contour boxes after layout so their visible heights match exactly.
    figure.canvas.draw()
    figure.set_layout_engine(None)
    for colorbar_axis, parent_axis in (
        *zip(field_colorbar_axes, axes[:, 1]),
        *zip(error_colorbar_axes, axes[:, 2]),
    ):
        colorbar_box = colorbar_axis.get_position()
        parent_box = parent_axis.get_position()
        colorbar_axis.set_in_layout(False)
        colorbar_axis.set_position(
            [colorbar_box.x0, parent_box.y0, colorbar_box.width, parent_box.height]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def visualize_run(
    run_dir: str | Path,
    *,
    case_dir: str | Path,
    checkpoint: str = "best",
    split: str = "test",
    sample_index: int = 0,
    sensor_config: str | Path | None = None,
    sensor_manifest: str | Path | None = None,
    generation_steps: int | None = None,
    device_name: str | None = None,
    output_path: str | Path | None = None,
    weight_selection: str = "configured",
    contour_levels: int = 20,
) -> Path:
    """Evaluate one full-grid snapshot and render its 300-DPI PNG."""
    run_dir = Path(run_dir).resolve()
    checkpoint_label = Path(checkpoint).stem
    report_name = f"reconstruction_{split}_{sample_index:04d}_{checkpoint_label}"
    warn_if_cuda_memory_tight(
        run_dir,
        checkpoint=checkpoint,
        device_name=device_name,
    )
    report_path = evaluate_run(
        run_dir,
        case_dir=case_dir,
        split=split,
        sample_index=sample_index,
        max_samples=1,
        checkpoint=checkpoint,
        sensor_config=sensor_config,
        sensor_manifest=sensor_manifest,
        query_points=None,
        generation_steps=generation_steps,
        device_name=device_name,
        report_name=report_name,
        weight_selection=weight_selection,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    payload_path = run_dir / report["trace"]["portable_plot_payload"]
    figure_path = (
        Path(output_path).resolve()
        if output_path is not None
        else report_path.parent / "reconstruction.png"
    )
    render_reconstruction_payload(
        payload_path,
        figure_path,
        title=(
            f"{Path(run_dir).parent.name} — {split} snapshot {sample_index} "
            f"({checkpoint_label}.pt)"
        ),
        dpi=300,
        contour_levels=contour_levels,
    )
    report["visualization"] = {
        "png": str(figure_path),
        "dpi": 300,
        "panels": ["ground_truth", "reconstruction", "absolute_error"],
        "error_annotation": "per_field_relative_l2_physical",
        "filled_contour_levels": contour_levels,
        "colorbar_ticks": 4,
        "line_contour_levels": contour_levels,
        "coordinate_space": "physical",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return figure_path
