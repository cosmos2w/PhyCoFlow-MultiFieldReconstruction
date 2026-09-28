"""Fixed-panel topology and source-relative fidelity history figures."""

from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path
from typing import Any

from ..config import load_config
from .history_plotting import HISTORY_MUTED_TEXT_COLOR, HISTORY_TEXT_COLOR, style_history_axis

_FIELD_COLORS = {
    "total": "#202733",
    "CH4": "#3B6EA8",
    "CO": "#D95F59",
    "T": "#B58900",
    "U_1": "#5B8E7D",
    "p": "#8C6BB1",
}
_FALLBACK_COLORS = ("#3B6EA8", "#D95F59", "#B58900", "#5B8E7D", "#8C6BB1", "#667085")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                warnings.warn(
                    f"ignored incomplete history row {line_number} in {path}",
                    stacklevel=2,
                )
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _observations(
    validation_rows: list[dict[str, Any]], training_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], str]:
    epoch_by_step = {
        int(row["step"]): float(row["epoch"])
        for row in training_rows
        if _finite(row.get("step")) is not None and _finite(row.get("epoch")) is not None
    }
    records = []
    epoch_matches = 0
    for row in validation_rows:
        step = _finite(row.get("step"))
        metric = _finite(row.get("metric"))
        if step is None or metric is None:
            continue
        step = int(step)
        epoch = epoch_by_step.get(step, 0.0 if step == 0 else None)
        if epoch is not None:
            epoch_matches += 1
        records.append(
            {
                "step": step,
                "x": float(epoch if epoch is not None else step),
                "metric": metric,
                "eligible": bool(row.get("eligible", False)),
                "relative_mse_increase": row.get("relative_mse_increase", {}),
                "failed_fidelity_fields": set(row.get("failed_fidelity_fields", ())),
            }
        )
    records.sort(key=lambda record: record["step"])
    x_label = "Training epoch" if records and epoch_matches == len(records) else "Optimizer step"
    return records, x_label


def build_fidelity_history_figure(
    validation_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    pyplot,
    *,
    total_budget: float,
    field_budget: float,
    selected_step: int | None = None,
):
    """Build decision-focused score and relative-MSE panels from saved reports."""
    records, x_label = _observations(validation_rows, training_rows)
    if not records:
        return None

    field_names = list(records[0]["relative_mse_increase"])
    if "total" in field_names:
        field_names = ["total", *(name for name in field_names if name != "total")]
    epoch_max = max(record["x"] for record in records)
    score_values = [record["metric"] for record in records]
    figure = pyplot.figure(figsize=(12.4, 10.2), constrained_layout=True, facecolor="white")
    grid = figure.add_gridspec(4, 1, height_ratios=(0.78, 1.0, 1.05, 0.13))
    score_axis = figure.add_subplot(grid[0, 0])
    full_range_axis = figure.add_subplot(grid[1, 0], sharex=score_axis)
    gate_axis = figure.add_subplot(grid[2, 0], sharex=score_axis)
    legend_axis = figure.add_subplot(grid[3, 0])
    legend_axis.set_axis_off()
    figure.suptitle("Checkpoint fidelity and topology score", color=HISTORY_TEXT_COLOR, fontsize=14, fontweight="medium")

    x = [record["x"] for record in records]
    metric = [record["metric"] for record in records]
    score_axis.plot(x, metric, color="#667085", linewidth=0.9, alpha=0.65, zorder=1)
    eligible = [record["eligible"] for record in records]
    score_axis.scatter(
        [value for value, passed in zip(x, eligible) if passed],
        [value for value, passed in zip(metric, eligible) if passed],
        marker="o",
        s=25,
        facecolors="#3B6EA8",
        edgecolors="white",
        linewidths=0.45,
        zorder=3,
        label="Eligible candidate",
    )
    score_axis.scatter(
        [value for value, passed in zip(x, eligible) if not passed],
        [value for value, passed in zip(metric, eligible) if not passed],
        marker="x",
        s=34,
        color="#D95F59",
        linewidths=1.3,
        zorder=4,
        label="Rejected by eligibility gates",
    )
    selected = next((record for record in records if record["step"] == selected_step), None)
    if selected is not None:
        score_axis.scatter(
            [selected["x"]],
            [selected["metric"]],
            marker="*",
            s=150,
            facecolors="#E0A526",
            edgecolors="#202733",
            linewidths=0.65,
            zorder=5,
            label="Selected checkpoint",
        )
        for axis in (score_axis, full_range_axis, gate_axis):
            axis.axvline(selected["x"], color="#B58900", linewidth=0.9, linestyle=":", alpha=0.85, zorder=0)
    score_axis.set_title("Fixed-panel topology score · lower is better", loc="left", color=HISTORY_TEXT_COLOR, fontsize=11.2)
    score_axis.set_ylabel("Selection score")
    style_history_axis(score_axis, score_values, x_max=epoch_max)
    score_axis.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#D4D9E2", framealpha=0.96, fontsize=8.2, ncol=3)
    all_relative_values: list[float] = []
    field_series = []
    for field_index, field in enumerate(field_names):
        color = _FIELD_COLORS.get(field, _FALLBACK_COLORS[field_index % len(_FALLBACK_COLORS)])
        points = []
        failed_points = []
        for record in records:
            raw_value = record["relative_mse_increase"].get(field)
            value = _finite(raw_value)
            if value is None:
                continue
            points.append((record["x"], value))
            all_relative_values.append(value)
            threshold = total_budget if field == "total" else field_budget
            if value > threshold:
                failed_points.append((record["x"], value))
        if not points:
            continue
        field_series.append((field, color, points, failed_points))

    from matplotlib.ticker import PercentFormatter

    relative_values = [*all_relative_values, 0.0, total_budget, field_budget]
    low, high = min(relative_values), max(relative_values)
    padding = max((high - low) * 0.06, 0.01)
    full_limits = (low - padding, high + padding)
    gate_low = min(0.0, -abs(field_budget))
    gate_high = max(total_budget, field_budget) + max(0.5 * max(abs(total_budget), abs(field_budget)), 0.01)

    def draw_relative_panel(axis, *, limits: tuple[float, float], show_zoom_edges: bool) -> list[Any]:
        handles = []
        for field, color, points, failed_points in field_series:
            if show_zoom_edges:
                visible = [point for point in points if limits[0] <= point[1] <= limits[1]]
                axis.scatter(
                    [point[0] for point in visible],
                    [point[1] for point in visible],
                    marker="o",
                    s=8,
                    color=color,
                    alpha=0.58,
                    linewidths=0,
                    zorder=2,
                )
            else:
                (line,) = axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    color=color,
                    linewidth=1.25,
                    alpha=0.92,
                    label=field if field != "total" else "Total",
                )
                handles.append(line)
            if failed_points:
                failed_visible = [
                    point
                    for point in failed_points
                    if not show_zoom_edges or limits[0] <= point[1] <= limits[1]
                ]
                axis.scatter(
                    [point[0] for point in failed_visible],
                    [point[1] for point in failed_visible],
                    marker="x",
                    s=24 if show_zoom_edges else 30,
                    color=color,
                    linewidths=1.2,
                    zorder=4,
                )
            if show_zoom_edges:
                span = limits[1] - limits[0]
                edge_pad = span * 0.012
                above = [point for point in points if point[1] > limits[1]]
                below = [point for point in points if point[1] < limits[0]]
                for outside, marker, y in (
                    (above, "^", limits[1] - edge_pad),
                    (below, "v", limits[0] + edge_pad),
                ):
                    if outside:
                        axis.scatter(
                            [point[0] for point in outside],
                            [y] * len(outside),
                            marker=marker,
                            s=10,
                            facecolors=color,
                            alpha=0.34,
                            linewidths=0,
                            zorder=5,
                        )
        axis.axhline(0.0, color="#667085", linewidth=0.85, linestyle="-", alpha=0.8, zorder=0)
        axis.axhline(total_budget, color="#202733", linewidth=0.95, linestyle="--", alpha=0.72, zorder=0)
        if not math.isclose(total_budget, field_budget, rel_tol=1e-9, abs_tol=1e-12):
            axis.axhline(field_budget, color="#667085", linewidth=0.95, linestyle=":", alpha=0.8, zorder=0)
        style_history_axis(axis, [*all_relative_values, 0.0, total_budget, field_budget], x_max=epoch_max)
        axis.set_yscale("linear")
        axis.set_ylim(*limits)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        return handles

    field_handles = draw_relative_panel(full_range_axis, limits=full_limits, show_zoom_edges=False)
    full_range_axis.set_title("Source-relative normalized MSE · full range", loc="left", color=HISTORY_TEXT_COLOR, fontsize=11.2)
    full_range_axis.set_ylabel("Relative MSE change")
    draw_relative_panel(gate_axis, limits=(gate_low, gate_high), show_zoom_edges=True)
    gate_axis.set_title("Fidelity gates · focused range", loc="left", color=HISTORY_TEXT_COLOR, fontsize=11.2)
    gate_axis.set_ylabel("Relative MSE change")
    gate_axis.set_xlabel(x_label)
    gate_note = (
        f"Dashed: +{total_budget:.0%} total / field limits"
        if math.isclose(total_budget, field_budget, rel_tol=1e-9, abs_tol=1e-12)
        else f"Dashed: total +{total_budget:.0%} · dotted: field +{field_budget:.0%}"
    )
    full_range_axis.text(0.99, 1.015, gate_note + " · crosses mark field-limit violations", transform=full_range_axis.transAxes, ha="right", va="bottom", color=HISTORY_MUTED_TEXT_COLOR, fontsize=7.8)
    gate_axis.text(0.99, 1.015, "Dots are in-view observations; triangles mark values outside this view", transform=gate_axis.transAxes, ha="right", va="bottom", color=HISTORY_MUTED_TEXT_COLOR, fontsize=7.8)
    legend_axis.legend(
        field_handles,
        [handle.get_label() for handle in field_handles],
        loc="center",
        frameon=False,
        fontsize=8.6,
        ncol=min(6, max(1, len(field_handles))),
        handlelength=2.0,
        columnspacing=1.5,
    )
    return figure


def render_fidelity_history(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    pyplot=None,
) -> Path | None:
    """Render ``checkpoint_fidelity.png`` without loading a model or dataset."""
    run_dir = Path(run_dir)
    validation_rows = _read_jsonl(run_dir / "metrics" / "topology_validation.jsonl")
    if not validation_rows:
        return None
    training_rows = _read_jsonl(run_dir / "metrics" / "history.jsonl")
    config_path = run_dir / "resolved_config.yaml"
    config = load_config(config_path) if config_path.is_file() else {}
    fidelity = config.get("posttrain_fidelity", {})
    total_budget = float(fidelity.get("max_relative_mse_increase", 0.05))
    field_budget = float(fidelity.get("max_relative_field_mse_increase", total_budget))
    selected_path = run_dir / "evaluation" / "selected.json"
    selected_step = None
    if selected_path.is_file():
        try:
            selected = json.loads(selected_path.read_text(encoding="utf-8"))
            selected_step = int(selected["global_step"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            selected_step = None
    if pyplot is None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            from matplotlib import pyplot
        except ImportError:
            warnings.warn("matplotlib is unavailable; checkpoint_fidelity.png cannot be generated", stacklevel=2)
            return None
    pyplot.rcParams["svg.fonttype"] = "none"
    figure = build_fidelity_history_figure(
        validation_rows,
        training_rows,
        pyplot,
        total_budget=total_budget,
        field_budget=field_budget,
        selected_step=selected_step,
    )
    if figure is None:
        return None
    destination = Path(output_path) if output_path is not None else run_dir / "checkpoint_fidelity.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    figure.savefig(temporary, dpi=180, format="png")
    pyplot.close(figure)
    os.replace(temporary, destination)
    return destination
