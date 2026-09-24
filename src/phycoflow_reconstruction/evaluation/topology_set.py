"""Set-level evaluation figures for the configured cubical-persistence family.

The evaluator reuses the training family's fixed query selector, raster map,
normalization and filtration construction. Exact H0/H1 counts are then measured
on those same signed filtration fields, separately from the differentiable
training objective.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..coherence.families.topology.betti_curves import gaussian_blur
from ..coherence.families.topology.exact import exact_betti_curves, reference_levels
from ..coherence.families.topology.geometry import rasterize_fields
from ..coherence.registry import build_coherence_family
from ..data.training_batches import fixed_query_indices

TOPOLOGY_COLOR = "#C48400"
TOPOLOGY_DARK = "#76500A"
REFERENCE_COLOR = "#3F5365"
PREDICTION_COLOR = TOPOLOGY_COLOR
DISAGREEMENT_COLOR = "#D96855"


def _publication_label(value: str) -> str:
    return str(value).replace("_", " ")


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


def _save_publication_figure(figure: Any, output_path: str | Path, *, dpi: int = 300) -> Path:
    """Write a familiar PNG plus vector PDF and SVG with matching base names."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(figure)
    return output_path


def _filtration_metadata(objective: Any) -> list[dict[str, Any]]:
    """Describe rows in PersistenceTopologyObjective._filtrations output."""
    rows: list[dict[str, Any]] = []
    for direction in objective.directions:
        if objective.weights.get("self.persistence", 0.0) > 0:
            for name in objective.self_fields:
                rows.append(
                    {
                        "metric": f"self.{name}",
                        "kind": "self",
                        "direction": direction,
                        "line_index": None,
                        "line_weight": 1.0,
                    }
                )
        if objective.weights.get("mutual.persistence", 0.0) > 0:
            for group_index, group in enumerate(objective.groups):
                directions = getattr(objective, f"line_directions_{group_index}")
                weights = directions.min(dim=1).values.detach().cpu().numpy()
                label = "mutual." + "+".join(objective.fields[index] for index in group)
                for line_index, line_weight in enumerate(weights):
                    rows.append(
                        {
                            "metric": label,
                            "kind": "mutual",
                            "direction": direction,
                            "line_index": line_index,
                            "line_weight": float(line_weight),
                        }
                    )
    return rows


def _aggregate_rows(
    values: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    direction: str,
) -> np.ndarray:
    indices = [
        index
        for index, row in enumerate(metadata)
        if row["metric"] == metric and row["direction"] == direction
    ]
    if not indices:
        raise KeyError(f"topology filtration rows are missing {metric} / {direction}")
    # A Betti number remains an unweighted count. For mutual views this averages
    # the configured line restrictions; the exact line-weighted distance remains
    # available as a separate objective metric.
    return values[:, indices].mean(axis=1)


def exact_betti_from_filtration_banks(
    reference_bank: torch.Tensor,
    generated_bank: torch.Tensor,
    quantiles: Sequence[float],
    *,
    periodic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return generated/reference exact H0/H1 counts for objective filtrations.

    Inputs follow ``PersistenceTopologyObjective._filtrations``: ``[rows, B,
    H, W]``. Thresholds come from each paired reference filtration using the
    same float64 CPU quantile helper as native topology evaluation. Negating
    fields and thresholds converts exact superlevel counts to lower-star
    sublevel counts, matching the scored cubical diagrams.
    """
    if reference_bank.ndim != 4 or generated_bank.shape != reference_bank.shape:
        raise ValueError("topology filtration banks must align as [rows,B,H,W]")
    if reference_bank.shape[1] < 1:
        raise ValueError("topology filtration banks must contain a sample")
    quantile_values = np.asarray(quantiles, dtype=np.float64)
    if quantile_values.ndim != 1 or quantile_values.size < 1 or not np.isfinite(quantile_values).all():
        raise ValueError("topology reference quantiles must be a nonempty finite sequence")
    levels = reference_levels(
        reference_bank[:, 0], tuple(float(value) for value in quantile_values)
    )
    counts = exact_betti_curves(
        torch.cat((-generated_bank[:, 0], -reference_bank[:, 0])),
        torch.cat((-levels, -levels)),
        periodic=periodic,
    )
    generated_counts, reference_counts = counts.chunk(2)
    expected_shape = (reference_bank.shape[0], 2, quantile_values.size)
    if tuple(generated_counts.shape) != expected_shape or tuple(reference_counts.shape) != expected_shape:
        raise RuntimeError("exact topology Betti count shape does not match filtration bank")
    return generated_counts, reference_counts


def render_topology_distance_distributions(
    values: np.ndarray,
    labels: Sequence[str],
    roles: Sequence[str],
    output_path: str | Path,
    *,
    title: str,
    subtitle: str,
    ylabel: str = "Persistence distance / raster vertex",
    y_limits: tuple[float, float] | None = None,
    dpi: int = 300,
) -> Path:
    """Show per-snapshot configured persistence distances on their natural scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    values = np.asarray(values, dtype=np.float64)
    labels = tuple(str(value) for value in labels)
    roles = tuple(str(value) for value in roles)
    if values.ndim != 2 or values.shape[1] != len(labels) or len(labels) != len(roles):
        raise ValueError("topology distance values, labels, and roles must align")
    if values.shape[0] < 1 or not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("topology distance plot requires finite non-negative samples")
    colors = {
        "component": TOPOLOGY_COLOR,
        "component_weighted_total": "#354052",
    }
    figure, axis = plt.subplots(figsize=(7.2, 4.55))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    positions = np.arange(1, len(labels) + 1, dtype=np.float64)
    rng = np.random.default_rng(2718)
    violin_values: list[np.ndarray] = []
    violin_positions: list[float] = []
    for index, position in enumerate(positions):
        sample = values[:, index]
        color = colors[roles[index]]
        if sample.size >= 2 and not np.allclose(sample, sample[0]):
            violin_values.append(sample)
            violin_positions.append(position)
        axis.scatter(
            position + rng.uniform(-0.075, 0.075, size=sample.size),
            sample,
            s=18,
            color=color,
            alpha=0.60,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
        axis.scatter(
            position,
            np.median(sample),
            marker="D",
            s=38,
            facecolors="white",
            edgecolors=color,
            linewidths=1.0,
            zorder=4,
        )
    if violin_values:
        violins = axis.violinplot(
            violin_values,
            positions=violin_positions,
            widths=0.65,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, position in zip(violins["bodies"], violin_positions):
            color = colors[roles[int(position - 1)]]
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.17)
            body.set_linewidth(1.0)
    short_labels = {
        "self.persistence": "Self\npersistence",
        "mutual.persistence": "Mutual\npersistence",
        "component-weighted total*": "Component-weighted\ntotal*",
    }
    plot_labels = tuple(
        short_labels.get(label, _publication_label(label)) for label in labels
    )
    axis.set_xticks(positions, plot_labels)
    axis.tick_params(axis="x", labelsize=8.5, pad=6)
    axis.tick_params(axis="y", labelsize=8.7, width=0.75, length=3.5)
    axis.set_xlim(0.45, len(labels) + 0.55)
    if y_limits is None:
        upper = float(values.max())
        limits = (0.0, upper * 1.13 if upper > 0 else 1.0)
    else:
        limits = tuple(float(value) for value in y_limits)
        if len(limits) != 2 or not np.isfinite(limits).all() or limits[0] < 0.0 or limits[0] >= limits[1]:
            raise ValueError("topology distance y-limits must be finite, non-negative, and increasing")
    axis.set_ylim(*limits)
    axis.set_ylabel(ylabel, fontsize=9.8, labelpad=7)
    figure.text(0.02, 0.98, title, ha="left", va="top", fontsize=11.6, fontweight="semibold", color="#1F2937")
    figure.text(0.02, 0.91, subtitle, ha="left", va="top", fontsize=8.1, color="#65717D")
    axis.grid(axis="y", color="#E1E5E9", linewidth=0.65, linestyle="--")
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#8B95A1")
    axis.spines["bottom"].set_color("#8B95A1")
    axis.spines["left"].set_linewidth(0.75)
    axis.spines["bottom"].set_linewidth(0.75)
    figure.legend(
        handles=(
            Line2D([], [], marker="o", linestyle="none", color=TOPOLOGY_COLOR, label="Configured component"),
            Line2D([], [], marker="o", linestyle="none", color="#354052", label="Component-weighted total*"),
            Line2D([], [], marker="D", linestyle="none", markerfacecolor="white", markeredgecolor="#333333", label="Median"),
        ),
        loc="upper center",
        bbox_to_anchor=(0.54, 0.865),
        frameon=False,
        ncol=3,
        fontsize=8.0,
        handletextpad=0.4,
        columnspacing=0.9,
    )
    figure.subplots_adjust(left=0.13, right=0.98, top=0.78, bottom=0.24)
    figure.text(
        0.98,
        0.035,
        "*Total is before the outer topology family weight.",
        ha="right",
        va="bottom",
        fontsize=7.6,
        color="#66717C",
    )
    return _save_publication_figure(figure, output_path, dpi=dpi)


def render_topology_betti_curves(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    metric_labels: Sequence[str],
    directions: Sequence[str],
    quantiles: Sequence[float],
    output_path: str | Path,
    *,
    title: str,
    subtitle: str,
    y_limits: tuple[float, float] | None = None,
    dimension_y_limits: Sequence[tuple[float, float]] | None = None,
    dpi: int = 300,
) -> Path:
    """Compare exact H0/H1 curves for the same objective filtrations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    reference = np.asarray(reference, dtype=np.float64)
    reconstruction = np.asarray(reconstruction, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if reference.shape != reconstruction.shape or reference.ndim != 4:
        raise ValueError("topology Betti arrays must align as [sample,row,dimension,quantile]")
    if reference.shape[1] != len(metadata) or reference.shape[2] != 2:
        raise ValueError("topology Betti arrays and filtration metadata do not align")
    if reference.shape[3] != len(quantiles) or len(quantiles) < 2:
        raise ValueError("topology Betti quantiles do not align")
    if not np.isfinite(reference).all() or not np.isfinite(reconstruction).all():
        raise ValueError("topology Betti curves must be finite")

    if not directions:
        raise ValueError("topology Betti plot requires at least one filtration direction")
    if y_limits is not None and dimension_y_limits is not None:
        raise ValueError("set either shared y_limits or dimension_y_limits, not both")
    if dimension_y_limits is not None:
        dimension_y_limits = tuple(
            (float(lower), float(upper)) for lower, upper in dimension_y_limits
        )
        if len(dimension_y_limits) != 2 or any(
            not np.isfinite((lower, upper)).all() or lower < 0.0 or lower >= upper
            for lower, upper in dimension_y_limits
        ):
            raise ValueError("dimension_y_limits must contain valid non-negative H0/H1 bounds")
    colors = ("#B87600", "#43657A", "#48545E", "#8B6D52", "#6C7580")
    figure, axes = plt.subplots(
        2,
        len(directions),
        figsize=(4.4 * len(directions), 5.6),
        squeeze=False,
        sharex=True,
    )
    figure.patch.set_facecolor("white")
    for dimension in range(2):
        for direction_index, direction in enumerate(directions):
            axis = axes[dimension, direction_index]
            axis.set_facecolor("white")
            for metric_index, metric in enumerate(metric_labels):
                color = colors[metric_index % len(colors)]
                reference_curves = _aggregate_rows(
                    reference[:, :, dimension, :], metadata, metric=metric, direction=direction
                )
                reconstruction_curves = _aggregate_rows(
                    reconstruction[:, :, dimension, :], metadata, metric=metric, direction=direction
                )
                ref_mean = reference_curves.mean(axis=0)
                pred_mean = reconstruction_curves.mean(axis=0)
                ref_std = reference_curves.std(axis=0, ddof=1) if len(reference_curves) > 1 else np.zeros_like(ref_mean)
                pred_std = reconstruction_curves.std(axis=0, ddof=1) if len(reconstruction_curves) > 1 else np.zeros_like(pred_mean)
                axis.plot(quantiles, ref_mean, color=color, linewidth=1.7, marker="o", markersize=3.6)
                axis.plot(quantiles, pred_mean, color=color, linewidth=1.55, marker="s", markersize=3.4, linestyle="--")
                if len(reference_curves) > 1:
                    axis.fill_between(quantiles, ref_mean - ref_std, ref_mean + ref_std, color=color, alpha=0.10, linewidth=0)
                    axis.fill_between(quantiles, pred_mean - pred_std, pred_mean + pred_std, color=color, alpha=0.08, linewidth=0)
            axis.set_title(f"$\\beta_{dimension}$ · {direction}", loc="left", fontsize=9.8, fontweight="semibold", color="#26323D")
            axis.set_ylabel("Betti count", fontsize=8.7, color="#374151")
            axis.set_xticks(quantiles, [f"{value:.2g}" for value in quantiles])
            axis.tick_params(axis="both", labelsize=8.2, colors="#56616D", width=0.65, length=3)
            axis.grid(axis="y", color="#E2E5E8", linewidth=0.6, linestyle="--")
            axis.set_axisbelow(True)
            if dimension_y_limits is not None:
                axis.set_ylim(*dimension_y_limits[dimension])
            elif y_limits is not None:
                axis.set_ylim(*y_limits)
            else:
                maximum = float(max(reference[:, :, dimension].max(), reconstruction[:, :, dimension].max()))
                axis.set_ylim(0.0, max(1.0, maximum * 1.10))
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_color("#9AA2AA")
            axis.spines["bottom"].set_color("#9AA2AA")
            axis.spines["left"].set_linewidth(0.7)
            axis.spines["bottom"].set_linewidth(0.7)
    for axis in axes[1]:
        axis.set_xlabel("Reference filtration quantile", fontsize=8.8, labelpad=6)
    figure.text(0.01, 0.985, title, ha="left", va="top", fontsize=11.6, fontweight="semibold", color="#1F2937")
    figure.text(
        0.01,
        0.94,
        f"{subtitle} · solid circles: reference; dashed squares: reconstruction",
        ha="left",
        va="top",
        fontsize=7.9,
        color="#66717C",
    )
    legend = [
        Line2D([], [], color=colors[index % len(colors)], marker="o", linewidth=1.7, label=_publication_label(metric))
        for index, metric in enumerate(metric_labels)
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=min(4, len(legend)),
        frameon=False,
        fontsize=7.8,
        handlelength=1.6,
        columnspacing=1.0,
    )
    figure.subplots_adjust(left=0.105, right=0.99, top=0.86, bottom=0.23, wspace=0.22, hspace=0.28)
    return _save_publication_figure(figure, output_path, dpi=dpi)


def render_configured_grid_topology(
    reference_fields: np.ndarray,
    reconstruction_fields: np.ndarray,
    grid_coordinates: np.ndarray,
    field_names: Sequence[str],
    sample_id: str,
    *,
    units: str,
    sample_epoch: str,
    output_path: str | Path,
    dpi: int = 300,
) -> Path:
    """Render paired q50 masks and contour disagreement on the configured raster."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    reference_fields = np.asarray(reference_fields, dtype=np.float64)
    reconstruction_fields = np.asarray(reconstruction_fields, dtype=np.float64)
    grid_coordinates = np.asarray(grid_coordinates, dtype=np.float64)
    field_names = tuple(str(value) for value in field_names)
    if reference_fields.shape != reconstruction_fields.shape or reference_fields.ndim != 3:
        raise ValueError("configured topology maps must align as [field,height,width]")
    if reference_fields.shape[0] != len(field_names):
        raise ValueError("configured topology field labels do not align")
    if grid_coordinates.shape != (*reference_fields.shape[-2:], 2):
        raise ValueError("configured topology coordinates do not align with raster maps")
    if not np.isfinite(reference_fields).all() or not np.isfinite(reconstruction_fields).all():
        raise ValueError("configured topology maps must be finite")

    figure, axes = plt.subplots(
        len(field_names), 3, figsize=(9.2, max(3.0, 2.65 * len(field_names))),
        squeeze=False, sharex=True, sharey=True,
    )
    figure.patch.set_facecolor("white")
    x = grid_coordinates[..., 0]
    y = grid_coordinates[..., 1]
    x_extent = (float(x.min()), float(x.max()))
    y_extent = (float(y.min()), float(y.max()))
    extent = (*x_extent, *y_extent)
    panel_titles = ("Reference", "Reconstruction", "Mask disagreement")
    for field_index, field_name in enumerate(field_names):
        ref = reference_fields[field_index]
        pred = reconstruction_fields[field_index]
        level = float(np.quantile(ref, 0.5))
        ref_mask = ref >= level
        pred_mask = pred >= level
        xor = np.logical_xor(ref_mask, pred_mask)
        masks = (ref_mask, pred_mask)
        for panel in range(3):
            axis = axes[field_index, panel]
            axis.set_facecolor("white")
            if panel < 2:
                mask = masks[panel]
                color = REFERENCE_COLOR if panel == 0 else PREDICTION_COLOR
                cmap = ListedColormap(("#F4F5F6", color))
                axis.imshow(
                    mask,
                    origin="lower",
                    interpolation="nearest",
                    extent=extent,
                    cmap=cmap,
                    vmin=0,
                    vmax=1,
                    aspect="equal",
                )
                boundary = mask
                if boundary.any() and not boundary.all():
                    axis.contour(x, y, boundary.astype(np.float32), levels=(0.5,), colors=("#26323D",), linewidths=0.7)
            else:
                axis.imshow(
                    xor,
                    origin="lower",
                    interpolation="nearest",
                    extent=extent,
                    cmap=ListedColormap(("#F4F5F6", DISAGREEMENT_COLOR)),
                    vmin=0,
                    vmax=1,
                    aspect="equal",
                    alpha=0.82,
                )
                if ref_mask.any() and not ref_mask.all():
                    axis.contour(x, y, ref_mask.astype(np.float32), levels=(0.5,), colors=(REFERENCE_COLOR,), linewidths=0.9, linestyles="solid")
                if pred_mask.any() and not pred_mask.all():
                    axis.contour(x, y, pred_mask.astype(np.float32), levels=(0.5,), colors=(PREDICTION_COLOR,), linewidths=0.9, linestyles="dashed")
            if field_index == 0:
                axis.set_title(
                    panel_titles[panel],
                    fontsize=8.8,
                    fontweight="semibold",
                    color="#2C3742",
                    pad=7,
                )
            axis.tick_params(axis="both", labelsize=7.2, colors="#68737E", length=2.5, width=0.6)
            axis.set_aspect("equal", adjustable="box")
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_color("#A1A8AF")
            axis.spines["bottom"].set_color("#A1A8AF")
            axis.spines["left"].set_linewidth(0.65)
            axis.spines["bottom"].set_linewidth(0.65)
            if panel == 0:
                axis.set_ylabel("y-coordinate", fontsize=7.8, labelpad=5)
                figure.text(
                    0.012,
                    0.91 - 0.70 * (field_index + 0.5) / len(field_names),
                    f"{_publication_label(field_name)}\nq50={level:.3g}",
                    ha="left",
                    va="center",
                    fontsize=8.0,
                    fontweight="semibold",
                    color="#2C3742",
                )
            if field_index == len(field_names) - 1:
                axis.set_xlabel("x-coordinate", fontsize=8.0, labelpad=5)
    handles = (
        Line2D([], [], color=REFERENCE_COLOR, linewidth=1.1, linestyle="solid", label="Reference boundary"),
        Line2D([], [], color=PREDICTION_COLOR, linewidth=1.1, linestyle="dashed", label="Reconstruction boundary"),
        Line2D([], [], color=DISAGREEMENT_COLOR, linewidth=4.0, label="Mask disagreement"),
    )
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=3, frameon=False, fontsize=7.8)
    figure.text(
        0.01,
        0.985,
        f"Configured topology raster · {reference_fields.shape[-2]}×{reference_fields.shape[-1]} · {units}",
        ha="left", va="top", fontsize=11.4, fontweight="semibold", color="#1F2937",
    )
    # Provenance and interpretation are recorded in report.json/README so that
    # the image header can stay clear at journal-column size.
    del sample_id, units, sample_epoch
    figure.subplots_adjust(left=0.18, right=0.99, top=0.91, bottom=0.21, wspace=0.08, hspace=0.25)
    return _save_publication_figure(figure, output_path, dpi=dpi)


@dataclass
class TopologySetAccumulator:
    """Stream paired predictions through the resolved training topology contract."""

    family: Any
    config: dict[str, Any]
    dataset_field_names: tuple[str, ...]
    normalization_method: str
    source_grid_shape: tuple[int, ...]
    query_point_count: int
    query_seed: int
    query_policy: str = "fixed_shared"
    query_selector: str = "phycoflow_reconstruction.data.training_batches.fixed_query_indices"
    matches_training_fixed_shared_selector: bool = True
    query_indices: torch.Tensor | None = None
    fixed_coordinates: torch.Tensor | None = None
    sample_ids: list[str] = field(default_factory=list)
    family_total: list[float] = field(default_factory=list)
    objective_component_names: list[str] = field(default_factory=list)
    objective_component_values: list[list[float]] = field(default_factory=list)
    diagnostic_names: list[str] = field(default_factory=list)
    diagnostic_values: list[list[float]] = field(default_factory=list)
    filtration_metadata: list[dict[str, Any]] = field(default_factory=list)
    reference_betti: list[np.ndarray] = field(default_factory=list)
    reconstruction_betti: list[np.ndarray] = field(default_factory=list)
    reference_maps: list[np.ndarray] = field(default_factory=list)
    reconstruction_maps: list[np.ndarray] = field(default_factory=list)
    grid_coordinates: np.ndarray | None = None

    @classmethod
    def build(cls, runtime: Any) -> TopologySetAccumulator:
        coherence = runtime.config.get("coherence", {})
        configured = coherence.get("families", {}).get("topology") if isinstance(coherence, Mapping) else None
        if configured is None:
            raise ValueError("topology set evaluation requires a resolved topology family config")
        if not isinstance(configured, Mapping):
            raise TypeError("topology set family configuration must be a mapping")
        settings = dict(configured)
        settings["enabled"] = True
        settings["target_use"] = "paired_supervised"
        settings["reference_bank"] = {"enabled": False}
        family = build_coherence_family(
            "topology", settings, runtime.dataset.data_spec, runtime.dataset.normalizer
        ).to(runtime.device)
        family.eval()
        if getattr(family, "strategy", None) != "cubical_persistence":
            raise ValueError(
                "set-level topology diagnostics currently require strategy=cubical_persistence"
            )
        compute = coherence.get("compute_budget", {}) if isinstance(coherence, Mapping) else {}
        policy = str(compute.get("query_policy", "fixed_shared")).strip().lower()
        if policy != "fixed_shared":
            raise ValueError(
                "topology set evaluation requires the training fixed_shared query policy"
            )
        query_point_count = int(compute.get("point_count", 4096))
        if query_point_count < 2:
            raise ValueError("topology set evaluation requires at least two fixed shared points")
        objective = family.spatial_objective
        if objective is None or not hasattr(objective, "_filtrations"):
            raise RuntimeError("cubical persistence objective does not expose its filtration contract")
        dimensions = tuple(int(value) for value in objective.dimensions)
        if dimensions != (0, 1):
            raise ValueError(
                "topology set figures require configured H0/H1 persistence dimensions"
            )
        filtration_metadata = _filtration_metadata(objective)
        objective_components = tuple(
            f"topology.{name}" for name, weight in objective.weights.items() if weight > 0.0
        )
        diagnostic_names = tuple(
            f"topology.{metric}.h{dimension}"
            for metric in dict.fromkeys(row["metric"] for row in filtration_metadata)
            for dimension in dimensions
        )
        return cls(
            family=family,
            config=settings,
            dataset_field_names=tuple(runtime.dataset.field_names),
            normalization_method=str(runtime.dataset.normalizer.method),
            source_grid_shape=tuple(int(value) for value in runtime.dataset.data_spec.logical_shape),
            query_point_count=query_point_count,
            query_seed=int(compute.get("query_seed", 100045)),
            objective_component_names=objective_components,
            diagnostic_names=diagnostic_names,
            filtration_metadata=filtration_metadata,
        )

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self.family.field_names)

    @property
    def quantiles(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.family.quantiles)

    def use_preselected_query_points(self, *, seed: int, description: str) -> None:
        """Use an already sampled query array, clearly marking it auxiliary.

        This is for rerendering immutable prediction previews that retain query
        coordinates but not full-grid outputs. Such plots must not be reported
        as reproducing the training fixed-shared estimator.
        """
        if self.sample_ids or self.query_indices is not None:
            raise RuntimeError("preselected queries must be configured before the first update")
        if not description.strip():
            raise ValueError("preselected query provenance must be described")
        self.query_policy = "preselected_saved_queries"
        self.query_selector = description.strip()
        self.matches_training_fixed_shared_selector = False
        self.query_seed = int(seed)

    def _select_queries(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if prediction.ndim != 3 or target.shape != prediction.shape:
            raise ValueError("topology set predictions and targets must align as [B,N,C]")
        if coordinates.ndim != 3 or coordinates.shape[:2] != prediction.shape[:2]:
            raise ValueError("topology set coordinates must align with [B,N]")
        if prediction.shape[0] != 1:
            raise ValueError("topology set accumulator expects one snapshot per update")
        point_count = prediction.shape[1]
        if self.query_indices is None:
            if self.query_policy == "fixed_shared":
                self.query_indices = fixed_query_indices(
                    point_count, self.query_point_count, seed=self.query_seed
                )
            elif self.query_policy == "preselected_saved_queries":
                if point_count != self.query_point_count:
                    raise ValueError(
                        "saved preview query count does not match the retained coordinate array"
                    )
                self.query_indices = torch.arange(point_count, dtype=torch.long)
            else:
                raise ValueError(f"unsupported topology query policy {self.query_policy!r}")
            if self.query_indices is None:
                raise RuntimeError("fixed shared topology queries were not created")
            self.fixed_coordinates = coordinates[0, self.query_indices].detach().clone()
        indices = self.query_indices.to(prediction.device)
        selected_coordinates = coordinates[:, indices]
        if self.fixed_coordinates is None or not torch.equal(
            selected_coordinates[0], self.fixed_coordinates.to(selected_coordinates.device)
        ):
            raise ValueError(
                "topology set evaluation requires the recorded fixed shared query coordinates"
            )
        return prediction[:, indices], target[:, indices], selected_coordinates

    @torch.no_grad()
    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        coordinates: torch.Tensor,
        sample_id: str,
    ) -> None:
        prediction, target, coordinates = self._select_queries(prediction, target, coordinates)
        result = self.family(prediction, target, coordinates=coordinates)
        if result.per_sample_cost is None or result.per_sample_cost.shape != (1,):
            raise ValueError("topology family did not return one per-snapshot total cost")
        self.family_total.append(float(result.per_sample_cost[0].detach().cpu()))

        objective_names = tuple(
            key
            for key in result.component_results
            if key in self.objective_component_names
        )
        if not self.objective_component_names:
            self.objective_component_names = list(objective_names)
        if tuple(self.objective_component_names) != objective_names:
            raise ValueError("topology objective components changed during set evaluation")
        self.objective_component_values.append(
            [float(result.component_results[key].per_sample_cost[0].detach().cpu()) for key in objective_names]
        )

        objective = self.family.spatial_objective
        generated_units, reference_units = self.family._units_and_fields(prediction, target)
        generated_grid = rasterize_fields(
            generated_units,
            self.family.neighbor_indices,
            self.family.neighbor_weights.to(generated_units),
            self.family.grid_shape,
        )
        reference_grid = rasterize_fields(
            reference_units,
            self.family.neighbor_indices,
            self.family.neighbor_weights.to(reference_units),
            self.family.grid_shape,
        )
        generated_grid = gaussian_blur(
            generated_grid,
            self.family.smoothing_sigma,
            self.family.periodic,
            radius=self.family.smoothing_radius,
        )
        reference_grid = gaussian_blur(
            reference_grid,
            self.family.smoothing_sigma,
            self.family.periodic,
            radius=self.family.smoothing_radius,
        )

        reference_descriptors = objective._descriptor_fields(reference_grid.detach())
        mean = reference_descriptors.mean((-2, -1), keepdim=True)
        scale = reference_descriptors.std((-2, -1), keepdim=True, unbiased=False).clamp_min(
            objective.scale_floor
        )
        reference_standardized = (reference_descriptors - mean) / scale
        generated_standardized = (objective._descriptor_fields(generated_grid) - mean) / scale
        reference_bank, _, _, _ = objective._filtrations(reference_standardized)
        generated_bank, _, _, _ = objective._filtrations(generated_standardized)
        if reference_bank.shape != generated_bank.shape:
            raise RuntimeError("topology reference and generated filtration banks do not align")
        if reference_bank.shape[0] != len(self.filtration_metadata):
            raise RuntimeError("topology filtration metadata no longer matches the family")

        generated_counts, reference_counts = exact_betti_from_filtration_banks(
            reference_bank,
            generated_bank,
            self.quantiles,
            periodic=self.family.periodic,
        )
        self.reconstruction_betti.append(generated_counts.cpu().numpy())
        self.reference_betti.append(reference_counts.cpu().numpy())

        field_lookup = {name: index for index, name in enumerate(self.dataset_field_names)}
        channel_ids = [field_lookup[name] for name in self.field_names]
        self.reconstruction_maps.append(
            generated_grid[0, channel_ids].detach().cpu().numpy().astype(np.float32)
        )
        self.reference_maps.append(
            reference_grid[0, channel_ids].detach().cpu().numpy().astype(np.float32)
        )
        grid_coordinates = self.family.grid_coordinates.detach().cpu().numpy().astype(np.float64)
        if self.grid_coordinates is None:
            self.grid_coordinates = grid_coordinates
        elif not np.array_equal(grid_coordinates, self.grid_coordinates):
            raise ValueError("topology configured raster coordinates changed during set evaluation")
        if self.diagnostic_names:
            diagnostic_by_name = {
                key: term
                for key, term in result.component_results.items()
                if key not in self.objective_component_names
                and key.startswith("topology.")
                and ".h" in key
                and term.per_sample_cost is not None
                and term.diagnostics.get("evaluation_only")
            }
            row_values = []
            for key in self.diagnostic_names:
                if key not in diagnostic_by_name:
                    raise ValueError(f"topology evaluation-only metric {key!r} is missing")
                row_values.append(float(diagnostic_by_name[key].per_sample_cost[0].detach().cpu()))
            self.diagnostic_values.append(row_values)
        self.sample_ids.append(str(sample_id))

    def _curve_groups(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(row["metric"]) for row in self.filtration_metadata))

    def finalize(
        self,
        output_root: str | Path,
        *,
        split: str,
        checkpoint_label: str,
        run_label: str,
        scale: str,
    ) -> dict[str, Any]:
        del scale  # Betti counts and persistence distances use natural linear scales.
        if not self.sample_ids:
            raise ValueError("topology set evaluation contains no snapshots")
        if self.query_indices is None or self.fixed_coordinates is None or self.grid_coordinates is None:
            raise RuntimeError("topology set geometry/query contract was not initialized")
        destination = Path(output_root) / "coherence" / "topology"
        destination.mkdir(parents=True, exist_ok=True)
        family_values = np.asarray(self.family_total, dtype=np.float64)
        objective_values = np.asarray(self.objective_component_values, dtype=np.float64)
        diagnostic_values = np.asarray(self.diagnostic_values, dtype=np.float64)
        reference_betti = np.stack(self.reference_betti).astype(np.int64)
        reconstruction_betti = np.stack(self.reconstruction_betti).astype(np.int64)
        reference_maps = np.stack(self.reference_maps).astype(np.float32)
        reconstruction_maps = np.stack(self.reconstruction_maps).astype(np.float32)
        quantiles = np.asarray(self.quantiles, dtype=np.float64)
        objective_labels = tuple(self.objective_component_names)
        diagnostic_labels = tuple(self.diagnostic_names)

        # Aggregate exact count errors by objective filtration family, direction,
        # homology dimension and reference quantile. Mutual restrictions are
        # averaged as counts for interpretation; their weighted exact distances
        # remain unaggregated in the objective diagnostics above.
        curve_statistics: dict[str, dict[str, Any]] = {}
        for metric in self._curve_groups():
            for direction in self.family.spatial_objective.directions:
                rows = [
                    index
                    for index, meta in enumerate(self.filtration_metadata)
                    if meta["metric"] == metric and meta["direction"] == direction
                ]
                if not rows:
                    continue
                row_key = f"{metric}|{direction}"
                curve_statistics[row_key] = {}
                ref_curves = reference_betti[:, rows].mean(axis=1)
                pred_curves = reconstruction_betti[:, rows].mean(axis=1)
                for dimension in range(2):
                    ref_dim = ref_curves[:, dimension, :].astype(np.float64)
                    pred_dim = pred_curves[:, dimension, :].astype(np.float64)
                    abs_error = np.abs(pred_dim - ref_dim)
                    curve_l1 = abs_error.sum(axis=1) / np.maximum(ref_dim.sum(axis=1), 1.0)
                    curve_statistics[row_key][f"h{dimension}"] = {
                        "mean_absolute_count_error_by_quantile": abs_error.mean(axis=0).tolist(),
                        "mean_absolute_count_error": float(abs_error.mean()),
                        "normalized_curve_l1_mean": float(curve_l1.mean()),
                        "normalized_curve_l1_standard_deviation": float(
                            curve_l1.std(ddof=1) if len(curve_l1) > 1 else 0.0
                        ),
                    }

        # Select the median-scoring snapshot for the qualitative panel so the
        # example is representative of this evaluated set rather than a cherry pick.
        representative_index = int(np.argsort(family_values, kind="stable")[len(family_values) // 2])
        representative_metric = float(family_values[representative_index])
        representative_id = self.sample_ids[representative_index]
        units = str(self.family.units)
        units_label = (
            f"model units ({self.normalization_method} normalized)"
            if units == "model_units"
            else "physical units"
        )
        if self.matches_training_fixed_shared_selector:
            query_summary = (
                f"fixed_shared query={self.query_indices.numel():,}/"
                f"{int(np.prod(self.source_grid_shape)):,} (seed {self.query_seed})"
            )
        else:
            query_summary = (
                f"saved preview queries={self.query_indices.numel():,} (selection seed {self.query_seed}; "
                "auxiliary estimator)"
            )
        subtitle = (
            f"{split} · {checkpoint_label} · n={len(self.sample_ids)} · "
            f"{query_summary} · {self.family.grid_shape[0]}×{self.family.grid_shape[1]} raster"
        )

        figures = {
            "persistence_term_distributions": destination / "persistence_term_distributions.png",
            "betti_curves": destination / "betti_curves.png",
            "configured_grid_topology": destination / "configured_grid_topology.png",
        }
        distance_plot_values = np.column_stack((objective_values, family_values))
        distance_plot_labels = tuple(label.removeprefix("topology.") for label in objective_labels) + (
            "component-weighted total*",
        )
        distance_plot_roles = ("component" for _ in objective_labels)
        distance_name = str(self.family.spatial_objective.distance)
        render_topology_distance_distributions(
            distance_plot_values,
            distance_plot_labels,
            (*distance_plot_roles, "component_weighted_total"),
            figures["persistence_term_distributions"],
            title="Persistence distance by objective term",
            subtitle=subtitle,
            ylabel=f"{_publication_label(distance_name)} distance / raster vertex",
        )

        render_topology_betti_curves(
            reference_betti,
            reconstruction_betti,
            self.filtration_metadata,
            self._curve_groups(),
            tuple(self.family.spatial_objective.directions),
            quantiles,
            figures["betti_curves"],
            title="Exact Betti curves on scored filtrations",
            subtitle=(
                f"{split} · n={len(self.sample_ids)} · mean ±1 SD across snapshots · "
                f"reference filtration quantiles · {self.family.grid_shape[0]}×{self.family.grid_shape[1]} raster"
            ),
        )

        field_lookup = {name: index for index, name in enumerate(self.field_names)}
        representative_fields = [field_lookup[name] for name in self.field_names]
        render_configured_grid_topology(
            reference_maps[representative_index, representative_fields],
            reconstruction_maps[representative_index, representative_fields],
            self.grid_coordinates,
            self.field_names,
            representative_id,
            units=units_label,
            sample_epoch=f"{checkpoint_label} · q50 level-set interpretation",
            output_path=figures["configured_grid_topology"],
        )

        payload_path = destination / "metrics.npz"
        np.savez_compressed(
            payload_path,
            sample_ids=np.asarray(self.sample_ids),
            field_names=np.asarray(self.field_names),
            source_grid_shape=np.asarray(self.source_grid_shape, dtype=np.int64),
            configured_grid_shape=np.asarray(self.family.grid_shape, dtype=np.int64),
            query_indices=self.query_indices.cpu().numpy(),
            query_coordinates=self.fixed_coordinates.cpu().numpy(),
            query_seed=np.asarray(self.query_seed),
            query_policy=np.asarray(self.query_policy),
            query_selector=np.asarray(self.query_selector),
            matches_training_fixed_shared_selector=np.asarray(
                self.matches_training_fixed_shared_selector
            ),
            query_point_count=np.asarray(self.query_indices.numel()),
            quantiles=quantiles,
            objective_component_names=np.asarray(objective_labels),
            objective_component_distances=objective_values,
            # Keep the original key for the reconstruction-set comparison API;
            # both entries have the same value and are explicitly documented.
            family_total_distance=family_values,
            component_weighted_total_before_outer_family_weight=family_values,
            diagnostic_names=np.asarray(diagnostic_labels),
            diagnostic_distances=diagnostic_values,
            filtration_metric=np.asarray([row["metric"] for row in self.filtration_metadata]),
            filtration_kind=np.asarray([row["kind"] for row in self.filtration_metadata]),
            filtration_direction=np.asarray([row["direction"] for row in self.filtration_metadata]),
            filtration_line_index=np.asarray(
                [-1 if row["line_index"] is None else row["line_index"] for row in self.filtration_metadata],
                dtype=np.int64,
            ),
            filtration_line_weight=np.asarray([row["line_weight"] for row in self.filtration_metadata]),
            reference_betti=reference_betti,
            reconstruction_betti=reconstruction_betti,
            grid_coordinates=self.grid_coordinates,
            representative_index=np.asarray(representative_index),
            representative_reference_fields=reference_maps[representative_index, representative_fields],
            representative_reconstruction_fields=reconstruction_maps[representative_index, representative_fields],
            representative_family_total=np.asarray(representative_metric),
        )

        csv_path = destination / "metrics.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow((
                "sample_id",
                "metric_kind",
                "metric",
                "direction",
                "dimension",
                "reference_quantile",
                "reference_betti",
                "reconstruction_betti",
                "absolute_count_error",
                "value",
            ))
            for sample_index, sample_id in enumerate(self.sample_ids):
                for row_index, meta in enumerate(self.filtration_metadata):
                    for dimension in range(2):
                        for quantile_index, quantile in enumerate(quantiles):
                            ref_count = int(reference_betti[sample_index, row_index, dimension, quantile_index])
                            pred_count = int(reconstruction_betti[sample_index, row_index, dimension, quantile_index])
                            writer.writerow((sample_id, meta["kind"], meta["metric"], meta["direction"], f"h{dimension}", quantile, ref_count, pred_count, abs(pred_count - ref_count), ""))
                for label, value in zip(objective_labels, objective_values[sample_index]):
                    writer.writerow((sample_id, "persistence_distance", label, "", "", "", "", "", "", value))
                writer.writerow((sample_id, "persistence_distance", "topology.component_weighted_total_before_outer_family_weight", "", "", "", "", "", "", family_values[sample_index]))
                for label, value in zip(diagnostic_labels, diagnostic_values[sample_index]):
                    writer.writerow((sample_id, "dimension_distance_diagnostic", label, "", "", "", "", "", "", value))

        objective_statistics = {
            label: _finite_summary(objective_values[:, index])
            for index, label in enumerate(objective_labels)
        }
        diagnostic_statistics = {
            label: _finite_summary(diagnostic_values[:, index])
            for index, label in enumerate(diagnostic_labels)
        }
        if self.family.spatial_objective.distance == "sliced_wasserstein":
            distance_definition = (
                "cubical persistence diagrams compared with finite-angle sliced-Wasserstein quadrature; "
                "essential classes are weighted as configured and distances are normalized by raster vertex count"
            )
        else:
            distance_definition = (
                "cubical persistence diagrams compared with configured spatial-Wasserstein assignment; "
                "essential classes and spatial terms are weighted as configured, with distance normalized by raster vertex count"
            )
        report = {
            "family": "topology",
            "metric": "paired_reconstruction_to_ground_truth_cubical_persistence_and_exact_betti_curves",
            "split": split,
            "run_label": run_label,
            "checkpoint_label": checkpoint_label,
            "sample_count": len(self.sample_ids),
            "target_use": "paired_supervised",
            "units": units,
            "normalization_method": self.normalization_method,
            "query_contract": {
                "selector": self.query_selector,
                "policy": self.query_policy,
                "seed": self.query_seed,
                "selected_point_count": int(self.query_indices.numel()),
                "input_point_count": int(self.query_indices.numel())
                if self.query_policy == "preselected_saved_queries"
                else int(np.prod(self.source_grid_shape)),
                "source_point_count": int(np.prod(self.source_grid_shape)),
                "selected_indices": self.query_indices.cpu().tolist()
                if self.matches_training_fixed_shared_selector
                else None,
                "coordinates_sha256": hashlib.sha256(
                    self.fixed_coordinates.detach().cpu().contiguous().numpy().tobytes()
                ).hexdigest(),
                "matches_training_fixed_shared_selector": self.matches_training_fixed_shared_selector,
            },
            "raster": {
                "grid_shape": list(self.family.grid_shape),
                "source_grid_shape": list(self.source_grid_shape),
                "periodic": bool(self.family.periodic),
                "antialias_downsample_configured": bool(self.family.antialias_downsample),
                "antialias_downsample_applied": bool(
                    self.family.geometry_diagnostics.get("antialias_applied", 0)
                ),
                "geometry_sha256": self.family.geometry_sha256,
                "geometry_diagnostics": dict(self.family.geometry_diagnostics),
                "interpretation": (
                    "configured topology raster produced from the recorded fixed_shared query point cloud; not the dataset-native grid"
                    if self.matches_training_fixed_shared_selector
                    else "configured topology raster produced from already sampled preview query coordinates; not the training fixed_shared estimator or dataset-native grid"
                ),
            },
            "objective": {
                "strategy": self.family.strategy,
                "family_weight": float(self.family.family_weight),
                "component_weights": dict(self.family.component_weights),
                "objective_component_weights": dict(self.family.spatial_objective.weights),
                "settings": self.config,
                "distance_definition": distance_definition,
                "component_weighted_total_before_outer_family_weight": _finite_summary(family_values),
                "outer_family_weight_applied": False,
                "component_distances_before_family_weight": objective_statistics,
                "dimension_distance_diagnostics": diagnostic_statistics,
            },
            "evaluation_mode": "set_level_fixed_shared_training_selector"
            if self.matches_training_fixed_shared_selector
            else "auxiliary_saved_preview_queries_not_training_selector",
            "betti_curves": {
                "definition": "exact H0/H1 counts from the same signed, reference-standardized scalar and mutual line filtrations used by the cubical persistence objective",
                "reference_quantiles": quantiles.tolist(),
                "thresholds": "reference-filtration quantiles listed above; the persistence score itself integrates the full diagram",
                "periodic_boundary": bool(self.family.periodic),
                "aggregation": "per-snapshot exact counts; mutual curves average the fixed line restrictions for visualization; distance metrics preserve the configured line weights",
                "statistics": curve_statistics,
            },
            "visualization_scale": "linear natural scale; statistical log/linear scale option does not apply",
            "representative_visualization": {
                "policy": "stable median of per-snapshot component-weighted total before outer family weight",
                "sample_id": representative_id,
                "component_weighted_total_before_outer_family_weight": representative_metric,
                "level_set": "superlevel at the paired reference median for each configured self field",
                "reference_median_thresholds": {
                    name: float(np.quantile(reference_maps[representative_index, field_lookup[name]], 0.5))
                    for name in self.field_names
                },
                "value_units": units_label,
            },
            "artifacts": {
                "metrics_csv": csv_path.name,
                "metrics_payload": payload_path.name,
                "figures": {name: path.name for name, path in figures.items()},
                "vector_figures": {
                    name: {
                        "pdf": path.with_suffix(".pdf").name,
                        "svg": path.with_suffix(".svg").name,
                    }
                    for name, path in figures.items()
                },
            },
        }
        report_path = destination / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "family": "topology",
            "directory": str(destination),
            "report": str(report_path),
            "metrics_csv": str(csv_path),
            "metrics_payload": str(payload_path),
            "figures": {name: str(path) for name, path in figures.items()},
        }
