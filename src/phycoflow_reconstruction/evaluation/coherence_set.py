"""Statistical coherence evaluation and publication-ready distribution figures."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..coherence.families.cross_spectrum.statistics import (
    band_energies,
    graph_fourier,
    normalized_cross_band_coupling,
    off_diagonal_pair_mean_square_values,
    off_diagonal_pair_symmetric_coherence_scores,
    pair_mean_square_values,
    pair_symmetric_coherence_scores,
    spectral_coherence,
)
from ..coherence.registry import build_coherence_family
from ..data.training_batches import fixed_query_indices

SUPPORTED_COHERENCE_FAMILIES = ("global_distribution", "cross_spectrum")


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


def _default_global_distribution_config(field_names: Sequence[str]) -> dict[str, Any]:
    """Return the case-independent paired-reference statistical contract."""
    return {
        "enabled": True,
        "weight": 1.0,
        "target_use": "paired_supervised",
        "units": "model_units",
        "fields": list(field_names),
        "reference_bank": {"enabled": False},
        "components": {
            "self": {"enabled": True, "weight": 1.0},
            "mutual": {
                "enabled": len(field_names) >= 2,
                "weight": 1.0,
                "directions": 8,
                "seed": 1234,
            },
            "cross": {
                "enabled": len(field_names) >= 2,
                "weight": 1.0,
                "directions": 16,
                "top_fraction": 0.1,
                "seed": 1234,
                "include_axes": True,
                "qmc": True,
            },
        },
    }


def _default_cross_spectrum_config(field_names: Sequence[str]) -> dict[str, Any]:
    """Return the canonical fixed-graph, ensemble-level spectral contract."""
    if len(field_names) < 2:
        raise ValueError("cross-spectrum evaluation requires at least two fields")
    return {
        "enabled": True,
        "weight": 1.0,
        "target_use": "paired_supervised",
        "units": "model_units",
        "fields": list(field_names),
        "pairs": [list(pair) for pair in combinations(field_names, 2)],
        "reference_bank": {"enabled": False},
        "graph": {
            "k_neighbors": 16,
            "sigma": None,
            "num_modes": 48,
            "exclude_zero": True,
            "bands": ["low", "mid", "high"],
        },
        "eps": 1.0e-8,
        "components": {
            "same_frequency": {"enabled": True, "weight": 1.0},
            "cross_frequency": {"enabled": True, "weight": 1.0},
            "band_energy": {"enabled": False, "weight": 0.0},
        },
    }


def _evaluation_family_config(
    run_config: Mapping[str, Any], family_name: str, field_names: Sequence[str]
) -> dict[str, Any]:
    if family_name not in SUPPORTED_COHERENCE_FAMILIES:
        raise ValueError(
            f"coherence set evaluation does not yet support {family_name!r}; "
            f"supported families: {', '.join(SUPPORTED_COHERENCE_FAMILIES)}"
        )
    configured = (
        run_config.get("coherence", {}).get("families", {}).get(family_name)
        if isinstance(run_config.get("coherence", {}), Mapping)
        else None
    )
    defaults = (
        _default_global_distribution_config(field_names)
        if family_name == "global_distribution"
        else _default_cross_spectrum_config(field_names)
    )
    settings = deepcopy(dict(configured)) if isinstance(configured, Mapping) else defaults
    settings["enabled"] = True
    # Statistical checkpoint evaluation has a natural sample-aligned reference:
    # the dense ground truth already used by reconstruction evaluation.
    settings["target_use"] = "paired_supervised"
    settings["reference_bank"] = {"enabled": False}
    settings.setdefault("units", "model_units")
    settings.setdefault("fields", list(field_names))
    return settings


def _publication_label(name: str) -> str:
    name = str(name)
    if "–" in name:
        return "–".join(_publication_label(part) for part in name.split("–"))
    indexed = re.fullmatch(r"([A-Za-z]+)_([0-9]+)", name)
    if indexed:
        return rf"${indexed.group(1)}_{{{indexed.group(2)}}}$"
    trailing_index = re.fullmatch(r"(.+?)([0-9]+)", name)
    if trailing_index:
        return rf"{trailing_index.group(1)}$_{{{trailing_index.group(2)}}}$"
    return name.replace("_", " ")


def render_coherence_distribution(
    values: np.ndarray,
    labels: Sequence[str],
    roles: Sequence[str],
    output_path: str | Path,
    *,
    title: str,
    subtitle: str,
    ylabel: str = r"Squared distribution discrepancy ($W_2^2$)",
    dpi: int = 300,
    scale: str = "log",
    value_limits: tuple[float, float] | None = None,
) -> Path:
    """Render violin densities overlaid with samples and explicit medians."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    values = np.asarray(values, dtype=np.float64)
    labels = tuple(str(label) for label in labels)
    roles = tuple(str(role) for role in roles)
    if values.ndim != 2 or values.shape[1] != len(labels) or len(labels) != len(roles):
        raise ValueError("coherence plot values, labels, and roles must align")
    if values.shape[0] < 1:
        raise ValueError("coherence plot contains no samples")
    if scale not in {"log", "linear"}:
        raise ValueError("statistical plot scale must be 'log' or 'linear'")

    role_colors = {
        "detail": "#0072B2",
        "component_total": "#E69F00",
        "family_total": "#3F3F46",
    }
    colors = tuple(role_colors[role] for role in roles)
    width = max(7.6, min(18.0, 0.82 * len(labels) + 2.7))
    figure, axis = plt.subplots(figsize=(width, 5.2), layout="constrained")
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    positions = np.arange(1, len(labels) + 1, dtype=np.float64)
    rng = np.random.default_rng(2027)
    violin_values: list[np.ndarray] = []
    violin_positions: list[float] = []

    for index, position in enumerate(positions):
        finite = values[:, index]
        finite = finite[np.isfinite(finite)]
        if scale == "log":
            finite = finite[finite > 0.0]
        if finite.size >= 2 and not np.allclose(finite, finite[0]):
            violin_values.append(finite)
            violin_positions.append(position)
        jitter = rng.uniform(-0.11, 0.11, size=finite.size)
        axis.scatter(
            np.full(finite.size, position) + jitter,
            finite,
            s=11,
            alpha=0.42,
            color=colors[index],
            edgecolors="white",
            linewidths=0.25,
            zorder=3,
            rasterized=True,
        )
        if finite.size:
            axis.scatter(
                position,
                np.median(finite),
                marker="D",
                s=32,
                linewidths=0.9,
                facecolors="white",
                edgecolors=colors[index],
                zorder=4,
            )

    if violin_values:
        violin = axis.violinplot(
            violin_values,
            positions=violin_positions,
            widths=0.66,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, position in zip(violin["bodies"], violin_positions):
            category = int(position - 1)
            body.set_facecolor(colors[category])
            body.set_edgecolor(colors[category])
            body.set_alpha(0.20)
            body.set_linewidth(1.15)

    axis.set_xticks(positions, tuple(_publication_label(label) for label in labels))
    rotation = 28 if len(labels) > 6 or max(map(len, labels), default=0) > 10 else 0
    axis.tick_params(axis="x", labelsize=9.2, rotation=rotation, pad=5)
    if rotation:
        for tick in axis.get_xticklabels():
            tick.set_horizontalalignment("right")
    axis.tick_params(axis="y", labelsize=9.3, width=0.8, length=4)
    axis.set_xlim(0.35, len(labels) + 0.65)
    finite_values = values[np.isfinite(values)]
    axis.set_yscale(scale)
    if value_limits is None:
        if scale == "log":
            finite_values = finite_values[finite_values > 0.0]
            if not finite_values.size:
                raise ValueError("log-scale coherence plot contains no positive values")
            limits = (
                float(finite_values.min()) * 0.75,
                float(finite_values.max()) * 1.25,
            )
        else:
            upper = float(finite_values.max()) * 1.08 if finite_values.size else 1.0
            limits = (0.0, max(upper, np.finfo(np.float64).eps))
    else:
        limits = value_limits
    if scale == "log" and limits[0] <= 0.0:
        raise ValueError("log-scale coherence limits must be positive")
    if not np.isfinite(limits).all() or limits[0] >= limits[1]:
        raise ValueError("coherence plot limits must be finite and increasing")
    axis.set_ylim(*limits)
    axis.set_ylabel(ylabel, fontsize=10.3)
    axis.set_title(title, fontsize=12.0, fontweight="medium", pad=21)
    axis.text(
        0.5,
        1.015,
        subtitle,
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.0,
        color="0.32",
    )
    axis.grid(axis="y", color="0.86", linewidth=0.65, linestyle="--", dashes=(3, 2))
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.legend(
        handles=(
            Patch(facecolor=role_colors["detail"], alpha=0.20, label="Sub-term density"),
            Patch(
                facecolor=role_colors["component_total"],
                alpha=0.35,
                label="Weighted component total",
            ),
            Patch(
                facecolor=role_colors["family_total"],
                alpha=0.35,
                label="Family total",
            ),
            Line2D(
                [],
                [],
                marker="D",
                linestyle="none",
                markersize=4.5,
                markerfacecolor="white",
                markeredgecolor="0.2",
                label="Median",
            ),
        ),
        loc="upper left",
        ncol=4,
        frameon=False,
        fontsize=8.2,
        handlelength=1.1,
        columnspacing=1.0,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def render_cross_spectrum_score_bars(
    scores: np.ndarray,
    labels: Sequence[str],
    roles: Sequence[str],
    output_path: str | Path,
    *,
    title: str,
    subtitle: str,
    score_std: np.ndarray | None = None,
    dpi: int = 300,
) -> Path:
    """Render mean bounded spectral-agreement scores with optional ensemble spread."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = tuple(str(label) for label in labels)
    roles = tuple(str(role) for role in roles)
    if scores.size != len(labels) or len(labels) != len(roles):
        raise ValueError("cross-spectrum scores, labels, and roles must align")
    if not scores.size or not np.isfinite(scores).all():
        raise ValueError("cross-spectrum score chart requires finite values")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("cross-spectrum coherence scores must lie in [0, 1]")
    if score_std is None:
        score_std = np.zeros_like(scores)
    else:
        score_std = np.asarray(score_std, dtype=np.float64).reshape(-1)
        if score_std.shape != scores.shape or not np.isfinite(score_std).all():
            raise ValueError("cross-spectrum score standard deviations must align and be finite")
        if np.any(score_std < 0.0):
            raise ValueError("cross-spectrum score standard deviations must be non-negative")

    role_colors = {
        "detail": "#4C78A8",
        "component_total": "#D99100",
        "family_total": "#354052",
    }
    role_edges = {
        "detail": "#315D83",
        "component_total": "#9A6500",
        "family_total": "#202938",
    }
    detail_count = roles.count("detail")
    positions = np.arange(len(labels), dtype=np.float64)
    if detail_count and detail_count < len(labels):
        positions[detail_count:] += 0.30
    height = max(4.2, min(18.0, 0.39 * len(labels) + 1.65))
    figure, axis = plt.subplots(figsize=(8.35, height), layout="constrained")
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    for position, score, spread, role in zip(positions, scores, score_std, roles):
        axis.barh(
            position,
            score,
            height=0.36 if role == "detail" else 0.44,
            color=role_colors[role],
            edgecolor=role_edges[role],
            linewidth=0.55,
            zorder=3,
        )
        axis.scatter(
            score,
            position,
            s=24 if role == "detail" else 31,
            color=role_colors[role],
            edgecolors=role_edges[role],
            linewidths=0.7,
            zorder=4,
        )
        if spread > 0.0:
            axis.errorbar(
                score,
                position,
                xerr=np.asarray(
                    [[min(spread, score)], [min(spread, 1.0 - score)]], dtype=np.float64
                ),
                fmt="none",
                ecolor=role_edges[role],
                elinewidth=0.9,
                capsize=2.5,
                capthick=0.9,
                zorder=4.5,
            )
        lower_whisker = max(0.0, score - spread)
        upper_whisker = min(1.0, score + spread)
        inside = upper_whisker > 0.925
        axis.text(
            lower_whisker - 0.014 if inside else upper_whisker + 0.014,
            position,
            f"{score:.1%}",
            ha="right" if inside else "left",
            va="center",
            fontsize=8.7,
            fontweight="medium",
            color="white" if inside else "#27313D",
            zorder=5,
        )

    axis.set_yticks(positions, tuple(_publication_label(label) for label in labels))
    for tick, role in zip(axis.get_yticklabels(), roles):
        tick.set_fontweight("semibold" if role != "detail" else "normal")
        tick.set_color("#27313D")
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.045)
    axis.set_xticks(np.linspace(0.0, 1.0, 5))
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.tick_params(axis="x", labelsize=8.8, colors="#56606D", width=0.7, length=3.5)
    axis.tick_params(axis="y", labelsize=9.1, width=0.0, length=0, pad=7)
    axis.set_xlabel("Coherence score", fontsize=9.6, color="#374151", labelpad=8)
    axis.set_title(
        title,
        fontsize=12.2,
        fontweight="semibold",
        color="#1F2937",
        loc="left",
        pad=24,
    )
    axis.text(
        0.0,
        1.014,
        subtitle,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        color="#66707C",
    )
    axis.axvline(1.0, color="#68717D", linewidth=0.85, linestyle=(0, (3, 3)), zorder=2)
    if detail_count and detail_count < len(labels):
        separator = (positions[detail_count - 1] + positions[detail_count]) / 2
        axis.axhline(separator, color="#D4D8DE", linewidth=0.75, zorder=1)
    axis.grid(axis="x", color="#E4E7EB", linewidth=0.65, linestyle="-")
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color("#9AA1AA")
    axis.spines["bottom"].set_linewidth(0.75)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _padded_finite_range(*values: np.ndarray) -> tuple[float, float]:
    finite_parts = [
        np.asarray(value, dtype=np.float64).reshape(-1)
        for value in values
        if np.asarray(value).size
    ]
    finite = np.concatenate(finite_parts) if finite_parts else np.empty(0)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("joint-PDF range contains no finite values")
    lower = float(finite.min())
    upper = float(finite.max())
    if lower == upper:
        padding = max(abs(lower) * 0.02, 1.0e-6)
    else:
        padding = 0.02 * (upper - lower)
    return lower - padding, upper + padding


def _joint_probability_mass(
    values: np.ndarray,
    pair: tuple[int, int],
    value_range: tuple[tuple[float, float], tuple[float, float]],
    bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = np.asarray(values, dtype=np.float64)[:, pair]
    finite = np.isfinite(selected).all(axis=1)
    histogram, x_edges, y_edges = np.histogram2d(
        selected[finite, 0],
        selected[finite, 1],
        bins=bins,
        range=value_range,
    )
    total = float(histogram.sum())
    if total <= 0.0:
        raise ValueError("joint-PDF histogram contains no samples inside its shared range")
    return histogram / total, x_edges, y_edges


def _jensen_shannon_divergence_bits(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or np.any(left < 0.0) or np.any(right < 0.0):
        raise ValueError("Jensen-Shannon inputs must be aligned non-negative arrays")
    left_total = float(left.sum())
    right_total = float(right.sum())
    if left_total <= 0.0 or right_total <= 0.0:
        raise ValueError("Jensen-Shannon inputs must each have positive mass")
    left = left / left_total
    right = right / right_total
    midpoint = 0.5 * (left + right)

    def relative_entropy(values: np.ndarray) -> float:
        positive = values > 0.0
        return float(np.sum(values[positive] * np.log2(values[positive] / midpoint[positive])))

    return 0.5 * (relative_entropy(left) + relative_entropy(right))


def render_global_distribution_joint_pdf(
    reference_mass: np.ndarray,
    generated_mass: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    output_path: str | Path,
    *,
    left_field: str,
    right_field: str,
    title_prefix: str,
    js_divergence_bits: float,
    density_limits: tuple[float, float] | None = None,
    dpi: int = 300,
) -> Path:
    """Render matched ground-truth and reconstruction joint probability densities."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

    reference_mass = np.asarray(reference_mass, dtype=np.float64)
    generated_mass = np.asarray(generated_mass, dtype=np.float64)
    if reference_mass.shape != generated_mass.shape or reference_mass.ndim != 2:
        raise ValueError("joint-PDF ground-truth and reconstruction grids must align")
    dx = np.diff(np.asarray(x_edges, dtype=np.float64))[:, None]
    dy = np.diff(np.asarray(y_edges, dtype=np.float64))[None, :]
    reference_density = reference_mass / (dx * dy)
    generated_density = generated_mass / (dx * dy)
    positive = np.concatenate(
        (
            reference_density[reference_density > 0.0],
            generated_density[generated_density > 0.0],
        )
    )
    if not positive.size:
        raise ValueError("joint-PDF visualization contains no positive density")
    if density_limits is None:
        upper = float(positive.max())
        lower = max(float(positive.min()), upper * 1.0e-5)
    else:
        lower, upper = (float(value) for value in density_limits)
        if lower <= 0.0 or lower >= upper:
            raise ValueError("joint-PDF density limits must be positive and increasing")
    norm = (
        Normalize(vmin=0.0, vmax=upper * 1.05)
        if upper / lower < 1.05
        else LogNorm(vmin=lower, vmax=upper)
    )
    colormap = LinearSegmentedColormap.from_list(
        "phycoflow_joint_pdf",
        ("#EEF5F8", "#C9E2EA", "#79B8C8", "#2D8197", "#15536C", "#082F49"),
    )
    colormap.set_bad("white")

    figure, axes = plt.subplots(
        1, 2, figsize=(9.2, 4.35), layout="constrained", sharex=True, sharey=True
    )
    figure.patch.set_facecolor("white")
    meshes = []
    for axis, density, panel_title in zip(
        axes,
        (reference_density, generated_density),
        ("Ground truth", "Reconstruction"),
    ):
        axis.set_facecolor("white")
        mesh = axis.pcolormesh(
            x_edges,
            y_edges,
            np.ma.masked_less_equal(density.T, 0.0),
            cmap=colormap,
            norm=norm,
            shading="auto",
            rasterized=True,
        )
        meshes.append(mesh)
        axis.set_title(panel_title, fontsize=11.2, fontweight="semibold", color="#1F2937", pad=9)
        axis.set_xlabel(_publication_label(left_field), fontsize=10.5, color="#27313D", labelpad=6)
        axis.tick_params(labelsize=9.2, colors="#4B5563", width=0.75, length=3.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#8B95A1")
        axis.spines["bottom"].set_color("#8B95A1")
        axis.spines["left"].set_linewidth(0.8)
        axis.spines["bottom"].set_linewidth(0.8)
    axes[0].set_ylabel(_publication_label(right_field), fontsize=10.5, color="#27313D", labelpad=7)
    axes[1].text(
        0.975,
        0.965,
        f"JSD = {js_divergence_bits:.4f} bits",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9.4,
        fontweight="semibold",
        color="#16384A",
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "white",
            "edgecolor": "#B8C8D1",
            "linewidth": 0.7,
            "alpha": 0.92,
        },
        zorder=5,
    )
    colorbar = figure.colorbar(meshes[-1], ax=axes, location="right", pad=0.02, aspect=28)
    colorbar.set_label("Probability density", fontsize=10.0, color="#27313D", labelpad=8)
    colorbar.ax.tick_params(labelsize=8.8, colors="#4B5563", width=0.7, length=3.2)
    colorbar.outline.set_linewidth(0.7)
    colorbar.outline.set_edgecolor("#8B95A1")
    figure.suptitle(
        f"{title_prefix} — {left_field}–{right_field} joint probability density",
        fontsize=13.0,
        fontweight="semibold",
        color="#1F2937",
        y=1.025,
        va="bottom",
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


@dataclass
class GlobalDistributionAccumulator:
    """Collect per-snapshot global-distribution values without retaining fields."""

    family: torch.nn.Module
    data_spec: Any
    config: dict[str, Any]
    sample_ids: list[str] = field(default_factory=list)
    marginal: list[np.ndarray] = field(default_factory=list)
    pairwise: list[np.ndarray] = field(default_factory=list)
    joint: list[float] = field(default_factory=list)
    marginal_total: list[float] = field(default_factory=list)
    pairwise_total: list[float] = field(default_factory=list)
    joint_total: list[float] = field(default_factory=list)
    family_total: list[float] = field(default_factory=list)
    extra_view: bool = False
    extra_query_point_count: int = 0
    extra_query_seed: int = 100045
    extra_generated: list[np.ndarray] = field(default_factory=list)
    extra_reference: list[np.ndarray] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        runtime: Any,
        *,
        extra_view: bool = False,
        evaluated_sample_count: int | None = None,
    ) -> GlobalDistributionAccumulator:
        settings = _evaluation_family_config(
            runtime.config, "global_distribution", runtime.dataset.field_names
        )
        family = build_coherence_family(
            "global_distribution", settings, runtime.dataset.data_spec, runtime.dataset.normalizer
        ).to(runtime.device)
        family.eval()
        coherence = runtime.config.get("coherence", {})
        compute = coherence.get("compute_budget", {}) if isinstance(coherence, Mapping) else {}
        configured_points = int(compute.get("point_count", 4096))
        sample_count = max(1, int(evaluated_sample_count or 1))
        bounded_points = max(64, 1_024_000 // sample_count)
        return cls(
            family=family,
            data_spec=runtime.dataset.data_spec,
            config=settings,
            extra_view=extra_view,
            extra_query_point_count=min(configured_points, bounded_points),
            extra_query_seed=int(compute.get("query_seed", 100045)),
        )

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self.family.field_names)

    @property
    def pair_labels(self) -> tuple[str, ...]:
        if "mutual_pairwise_swd" not in self.family.components_by_key:
            return ()
        component = self.family.components_by_key["mutual_pairwise_swd"]
        names = tuple(self.data_spec.field_names)
        return tuple(f"{names[left]}–{names[right]}" for left, right in component.pairs)

    @property
    def pair_indices(self) -> tuple[tuple[int, int], ...]:
        if "mutual_pairwise_swd" not in self.family.components_by_key:
            return ()
        component = self.family.components_by_key["mutual_pairwise_swd"]
        return tuple(tuple(int(value) for value in pair) for pair in component.pairs)

    def _extra_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.extra_view or not self.extra_generated or not self.extra_reference:
            raise ValueError("global-distribution extra-view samples are unavailable")
        return np.concatenate(self.extra_generated), np.concatenate(self.extra_reference)

    def extra_value_ranges(
        self, *others: GlobalDistributionAccumulator
    ) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
        """Return shared pair domains across this run and optional comparison runs."""
        accumulators = (self, *others)
        if any(accumulator.pair_indices != self.pair_indices for accumulator in accumulators):
            raise ValueError("global-distribution extra-view field pairs do not match")
        arrays = [accumulator._extra_arrays() for accumulator in accumulators]
        ranges = {}
        for label, (left, right) in zip(self.pair_labels, self.pair_indices):
            ranges[label] = (
                _padded_finite_range(*(value[:, left] for pair in arrays for value in pair)),
                _padded_finite_range(*(value[:, right] for pair in arrays for value in pair)),
            )
        return ranges

    def extra_density_limits(
        self,
        value_ranges: Mapping[str, tuple[tuple[float, float], tuple[float, float]]],
        *others: GlobalDistributionAccumulator,
        bins: int = 64,
    ) -> dict[str, tuple[float, float]]:
        """Return shared logarithmic density limits for matched run figures."""
        accumulators = (self, *others)
        if any(accumulator.pair_indices != self.pair_indices for accumulator in accumulators):
            raise ValueError("global-distribution extra-view field pairs do not match")
        limits = {}
        for label, pair in zip(self.pair_labels, self.pair_indices):
            positive_parts = []
            for accumulator in accumulators:
                generated, reference = accumulator._extra_arrays()
                for values in (generated, reference):
                    mass, x_edges, y_edges = _joint_probability_mass(
                        values, pair, value_ranges[label], bins
                    )
                    density = mass / (np.diff(x_edges)[:, None] * np.diff(y_edges)[None, :])
                    positive_parts.append(density[density > 0.0])
            positive = np.concatenate(positive_parts)
            upper = float(positive.max())
            limits[label] = max(float(positive.min()), upper * 1.0e-5), upper
        return limits

    def render_extra(
        self,
        destination: str | Path,
        *,
        split: str,
        checkpoint_label: str,
        run_label: str,
        value_ranges: Mapping[str, tuple[tuple[float, float], tuple[float, float]]] | None = None,
        density_limits: Mapping[str, tuple[float, float]] | None = None,
        filename_suffix: str = "",
        role_label: str | None = None,
        bins: int = 64,
    ) -> dict[str, Any]:
        """Render configured pairwise joint PDFs and persist their JSD values."""
        if bins < 8:
            raise ValueError("global-distribution joint PDFs require at least eight bins")
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        generated, reference = self._extra_arrays()
        ranges = dict(value_ranges or self.extra_value_ranges())
        field_names = tuple(self.data_spec.field_names)
        figures: dict[str, Path] = {}
        rows = []
        reference_masses = []
        generated_masses = []
        x_edges_all = []
        y_edges_all = []
        for label, pair in zip(self.pair_labels, self.pair_indices):
            if label not in ranges:
                raise KeyError(f"missing shared joint-PDF range for {label}")
            reference_mass, x_edges, y_edges = _joint_probability_mass(
                reference, pair, ranges[label], bins
            )
            generated_mass, generated_x_edges, generated_y_edges = _joint_probability_mass(
                generated, pair, ranges[label], bins
            )
            if not np.array_equal(x_edges, generated_x_edges) or not np.array_equal(
                y_edges, generated_y_edges
            ):
                raise RuntimeError("joint-PDF histogram edges are not shared")
            divergence = _jensen_shannon_divergence_bits(reference_mass, generated_mass)
            left_name, right_name = field_names[pair[0]], field_names[pair[1]]
            safe_pair = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{left_name}-{right_name}")
            figure_path = destination / f"joint_pdf_{safe_pair}{filename_suffix}.png"
            render_global_distribution_joint_pdf(
                reference_mass,
                generated_mass,
                x_edges,
                y_edges,
                figure_path,
                left_field=left_name,
                right_field=right_name,
                title_prefix=role_label or run_label.replace("_", " "),
                js_divergence_bits=divergence,
                density_limits=(None if density_limits is None else density_limits[label]),
            )
            figures[label] = figure_path
            rows.append(
                {
                    "pair": label,
                    "left_field": left_name,
                    "right_field": right_name,
                    "jensen_shannon_divergence_bits": divergence,
                    "x_range": list(ranges[label][0]),
                    "y_range": list(ranges[label][1]),
                    "density_range": (
                        None if density_limits is None else list(density_limits[label])
                    ),
                    "figure": figure_path.name,
                }
            )
            reference_masses.append(reference_mass)
            generated_masses.append(generated_mass)
            x_edges_all.append(x_edges)
            y_edges_all.append(y_edges)

        if not rows:
            raise ValueError("global-distribution extra-view requires at least one field pair")
        metrics_stem = f"joint_pdf_metrics{filename_suffix}"
        csv_path = destination / f"{metrics_stem}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        payload_path = destination / f"{metrics_stem}.npz"
        np.savez_compressed(
            payload_path,
            pair_labels=np.asarray(self.pair_labels),
            reference_probability_mass=np.stack(reference_masses),
            reconstruction_probability_mass=np.stack(generated_masses),
            x_edges=np.stack(x_edges_all),
            y_edges=np.stack(y_edges_all),
            jensen_shannon_divergence_bits=np.asarray(
                [row["jensen_shannon_divergence_bits"] for row in rows]
            ),
            sample_ids=np.asarray(self.sample_ids),
            sampled_points_per_snapshot=np.asarray(len(self.extra_generated[0])),
            pooled_point_count=np.asarray(generated.shape[0]),
            units=np.asarray(self.family.units),
        )
        report_path = destination / f"report{filename_suffix}.json"
        report = {
            "family": "global_distribution",
            "view": "pairwise_joint_probability_density",
            "split": split,
            "checkpoint": f"{checkpoint_label}.pt",
            "sample_count": len(self.sample_ids),
            "sample_ids": list(self.sample_ids),
            "sampled_points_per_snapshot": len(self.extra_generated[0]),
            "pooled_point_count": int(generated.shape[0]),
            "point_selection": {
                "policy": "fixed_shared_deterministic_subset",
                "seed": self.extra_query_seed,
                "configured_maximum_points_per_snapshot": self.extra_query_point_count,
            },
            "histogram": {
                "bins_per_axis": bins,
                "normalization": "probability_mass_then_density_by_bin_area",
                "range_policy": "shared_ground_truth_and_reconstruction_finite_extrema_with_2pct_padding",
            },
            "jensen_shannon_divergence": {
                "logarithm_base": 2,
                "units": "bits",
                "range": [0.0, 1.0],
                "perfect_agreement": 0.0,
            },
            "pairs": rows,
            "artifacts": {
                "metrics_csv": csv_path.name,
                "metrics_payload": payload_path.name,
                "figures": {label: path.name for label, path in figures.items()},
            },
        }
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return {
            "directory": str(destination),
            "report": str(report_path),
            "metrics_csv": str(csv_path),
            "metrics_payload": str(payload_path),
            "figures": {label: str(path) for label, path in figures.items()},
            "jensen_shannon_divergence_bits": {
                row["pair"]: row["jensen_shannon_divergence_bits"] for row in rows
            },
        }

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        coordinates: torch.Tensor,
        sample_id: str,
    ) -> None:
        result = self.family(
            prediction,
            target,
            coordinates=coordinates,
            context={"sample_ids": [sample_id], "reference_ids": [sample_id]},
        )
        paths = {
            "marginal": "global_distribution.self.marginal_w2",
            "pairwise": "global_distribution.mutual.pairwise_swd",
            "joint": "global_distribution.cross.joint_topk_swd",
        }
        marginal = result.component_results.get(paths["marginal"])
        pairwise = result.component_results.get(paths["pairwise"])
        joint = result.component_results.get(paths["joint"])
        if marginal is None or pairwise is None or joint is None:
            raise ValueError(
                "global-distribution statistical evaluation requires self, mutual, and cross components"
            )
        marginal_values = marginal.diagnostics["per_field_w2"][0].detach().cpu().numpy()
        pairwise_values = pairwise.diagnostics["per_pair_swd"][0].detach().cpu().numpy()
        weights = self.family.component_weights
        self.sample_ids.append(sample_id)
        self.marginal.append(marginal_values)
        self.pairwise.append(pairwise_values)
        self.joint.append(float(joint.per_sample_cost[0].detach().cpu()))
        self.marginal_total.append(
            weights["self_marginal_w2"] * float(marginal.per_sample_cost[0].detach().cpu())
        )
        self.pairwise_total.append(
            weights["mutual_pairwise_swd"] * float(pairwise.per_sample_cost[0].detach().cpu())
        )
        self.joint_total.append(
            weights["cross_joint_topk_swd"] * float(joint.per_sample_cost[0].detach().cpu())
        )
        self.family_total.append(float(result.per_sample_cost[0].detach().cpu()))
        if self.extra_view:
            indices = fixed_query_indices(
                prediction.shape[1], self.extra_query_point_count, seed=self.extra_query_seed
            )
            if indices is None:
                raise RuntimeError("global-distribution extra-view indices are unavailable")
            generated, reference = self.family._in_declared_units(prediction, target)
            device_indices = indices.to(generated.device)
            self.extra_generated.append(
                generated[0, device_indices].detach().cpu().numpy().astype(np.float32)
            )
            self.extra_reference.append(
                reference[0, device_indices].detach().cpu().numpy().astype(np.float32)
            )

    def finalize(
        self,
        output_root: str | Path,
        *,
        split: str,
        checkpoint_label: str,
        run_label: str,
        scale: str,
    ) -> dict[str, Any]:
        destination = Path(output_root) / "coherence" / "global_distribution"
        destination.mkdir(parents=True, exist_ok=True)
        marginal = np.asarray(self.marginal, dtype=np.float64)
        pairwise = np.asarray(self.pairwise, dtype=np.float64)
        joint = np.asarray(self.joint, dtype=np.float64)[:, None]
        component_totals = np.column_stack(
            (self.marginal_total, self.pairwise_total, self.joint_total)
        )
        family_total = np.asarray(self.family_total, dtype=np.float64)[:, None]
        payload_path = destination / "metrics.npz"
        np.savez_compressed(
            payload_path,
            marginal_per_field=marginal,
            pairwise_per_field_pair=pairwise,
            joint_top_tail=joint,
            weighted_component_totals=component_totals,
            family_total=family_total,
            field_names=np.asarray(self.field_names),
            pair_labels=np.asarray(self.pair_labels),
            component_names=np.asarray(("marginal", "pairwise", "joint_top_tail")),
            sample_ids=np.asarray(self.sample_ids),
            split=np.asarray(split),
            units=np.asarray(self.family.units),
        )

        csv_path = destination / "metrics.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("sample_id", "component", "sub_term", "value"))
            for row, sample_id in enumerate(self.sample_ids):
                for column, name in enumerate(self.field_names):
                    writer.writerow((sample_id, "marginal", name, marginal[row, column]))
                for column, name in enumerate(self.pair_labels):
                    writer.writerow((sample_id, "pairwise", name, pairwise[row, column]))
                writer.writerow((sample_id, "joint_top_tail", "all_fields", joint[row, 0]))
                for column, name in enumerate(("marginal", "pairwise", "joint_top_tail")):
                    writer.writerow(
                        (sample_id, name, "weighted_component_total", component_totals[row, column])
                    )
                writer.writerow(
                    (sample_id, "global_distribution", "family_total", family_total[row, 0])
                )

        subtitle = (
            f"{split} set · {checkpoint_label}.pt · n={len(self.sample_ids)} · "
            f"paired ground truth · {self.family.units.replace('_', ' ')} · {scale} scale"
        )
        figures = {
            "marginal": destination / "marginal_field_distributions.png",
            "pairwise": destination / "pairwise_field_distributions.png",
            "joint_top_tail": destination / "joint_top_tail_distributions.png",
        }
        marginal_plot = np.column_stack((marginal, component_totals[:, 0], family_total[:, 0]))
        render_coherence_distribution(
            marginal_plot,
            (*self.field_names, "Marginal total", "Family total"),
            (*("detail" for _ in self.field_names), "component_total", "family_total"),
            figures["marginal"],
            title=f"{run_label} — marginal field-distribution coherence",
            subtitle=subtitle,
            scale=scale,
        )

        extra_output = None
        if self.extra_view:
            extra_output = self.render_extra(
                destination / "global_distribution_extra",
                split=split,
                checkpoint_label=checkpoint_label,
                run_label=run_label,
            )
        pairwise_plot = np.column_stack((pairwise, component_totals[:, 1], family_total[:, 0]))
        render_coherence_distribution(
            pairwise_plot,
            (*self.pair_labels, "Pairwise total", "Family total"),
            (*("detail" for _ in self.pair_labels), "component_total", "family_total"),
            figures["pairwise"],
            title=f"{run_label} — pairwise field-distribution coherence",
            subtitle=subtitle,
            scale=scale,
        )
        joint_plot = np.column_stack((joint[:, 0], component_totals[:, 2], family_total[:, 0]))
        render_coherence_distribution(
            joint_plot,
            ("Joint/top-tail", "Joint total", "Family total"),
            ("detail", "component_total", "family_total"),
            figures["joint_top_tail"],
            title=f"{run_label} — joint/top-tail distribution coherence",
            subtitle=subtitle,
            scale=scale,
        )

        statistics = {
            "marginal_per_field": {
                name: _finite_summary(marginal[:, index])
                for index, name in enumerate(self.field_names)
            },
            "pairwise_per_field_pair": {
                name: _finite_summary(pairwise[:, index])
                for index, name in enumerate(self.pair_labels)
            },
            "joint_top_tail": _finite_summary(joint[:, 0]),
            "weighted_component_totals": {
                name: _finite_summary(component_totals[:, index])
                for index, name in enumerate(("marginal", "pairwise", "joint_top_tail"))
            },
            "family_total": _finite_summary(family_total[:, 0]),
        }
        report = {
            "family": "global_distribution",
            "metric": "paired_reconstruction_to_ground_truth_squared_distribution_discrepancy",
            "split": split,
            "sample_count": len(self.sample_ids),
            "target_use": "paired_supervised",
            "units": self.family.units,
            "statistic_scale": scale,
            "field_names": list(self.field_names),
            "field_pairs": list(self.pair_labels),
            "family_weight": float(self.family.family_weight),
            "component_weights": dict(self.family.component_weights),
            "settings": self.config,
            "statistics": statistics,
            "artifacts": {
                "metrics_csv": csv_path.name,
                "metrics_payload": payload_path.name,
                "figures": {name: path.name for name, path in figures.items()},
                "extra": extra_output,
            },
        }
        report_path = destination / "report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return {
            "family": "global_distribution",
            "directory": str(destination),
            "report": str(report_path),
            "figures": {name: str(path) for name, path in figures.items()},
            "extra": extra_output,
        }


@dataclass
class CrossSpectrumAccumulator:
    """Collect compact coefficients for pooled or training-aligned ensemble estimates."""

    family: torch.nn.Module
    config: dict[str, Any]
    query_point_count: int
    query_seed: int
    aggregation: str
    ensemble_size: int
    ensemble_seed: int
    generated_coefficients: list[torch.Tensor] = field(default_factory=list)
    reference_coefficients: list[torch.Tensor] = field(default_factory=list)
    sample_ids: list[str] = field(default_factory=list)
    query_indices: torch.Tensor | None = None
    fixed_coordinates: torch.Tensor | None = None

    @classmethod
    def build(
        cls,
        runtime: Any,
        *,
        aggregation: str = "training_aligned",
    ) -> CrossSpectrumAccumulator:
        settings = _evaluation_family_config(
            runtime.config, "cross_spectrum", runtime.dataset.field_names
        )
        family = build_coherence_family(
            "cross_spectrum", settings, runtime.dataset.data_spec, runtime.dataset.normalizer
        ).to(runtime.device)
        family.eval()
        coherence = runtime.config.get("coherence", {})
        compute = coherence.get("compute_budget", {}) if isinstance(coherence, Mapping) else {}
        aggregation = str(aggregation).strip().lower()
        if aggregation not in {"training_aligned", "pooled"}:
            raise ValueError("cross-spectrum aggregation must be 'training_aligned' or 'pooled'")
        ensemble_size = int(compute.get("batch_size", 16))
        if ensemble_size < family.required_batch_size:
            raise ValueError(
                "cross-spectrum coherence batch size is smaller than the enabled components require"
            )
        query_point_count = int(compute.get("point_count", 4096))
        query_seed = int(compute.get("query_seed", 100045))
        return cls(
            family=family,
            config=settings,
            query_point_count=query_point_count,
            query_seed=query_seed,
            aggregation=aggregation,
            ensemble_size=ensemble_size,
            ensemble_seed=int(runtime.seed),
        )

    @property
    def pair_labels(self) -> tuple[str, ...]:
        names = tuple(self.family.field_names)
        return tuple(f"{names[left]}–{names[right]}" for left, right in self.family.pairs)

    @property
    def band_field_labels(self) -> tuple[str, ...]:
        return tuple(
            f"{band} {field_name}"
            for band in self.family.band_names
            for field_name in self.family.field_names
        )

    def _select_fixed_queries(
        self, prediction: torch.Tensor, target: torch.Tensor, coordinates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.query_indices is None:
            self.query_indices = fixed_query_indices(
                prediction.shape[1], self.query_point_count, seed=self.query_seed
            )
            if self.query_indices is None:
                raise RuntimeError("fixed cross-spectrum query selection was not created")
            self.fixed_coordinates = coordinates[0, self.query_indices].detach().clone()
        indices = self.query_indices.to(prediction.device)
        selected_coordinates = coordinates[0, indices]
        if self.fixed_coordinates is None or not torch.equal(
            selected_coordinates, self.fixed_coordinates.to(selected_coordinates.device)
        ):
            raise ValueError(
                "cross-spectrum set evaluation requires identical fixed coordinates for every snapshot"
            )
        return prediction[:, indices], target[:, indices], selected_coordinates.unsqueeze(0)

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        coordinates: torch.Tensor,
        sample_id: str,
    ) -> None:
        prediction, target, _ = self._select_fixed_queries(prediction, target, coordinates)
        generated, reference = self.family._units_and_fields(prediction, target)
        if self.fixed_coordinates is None:
            raise RuntimeError("cross-spectrum fixed coordinates are unavailable")
        basis = self.family._basis(
            self.fixed_coordinates.to(generated.device).unsqueeze(0), generated.dtype
        )
        self.generated_coefficients.append(graph_fourier(generated, basis)[0].detach().cpu())
        self.reference_coefficients.append(graph_fourier(reference, basis)[0].detach().cpu())
        self.sample_ids.append(sample_id)

    def finalize(
        self,
        output_root: str | Path,
        *,
        split: str,
        checkpoint_label: str,
        run_label: str,
        scale: str,
    ) -> dict[str, Any]:
        del scale  # Cross-spectrum scores always use their natural bounded linear scale.
        if len(self.sample_ids) < self.family.required_batch_size:
            raise ValueError(
                f"cross-spectrum evaluation requires at least {self.family.required_batch_size} "
                "selected snapshots"
            )
        destination = Path(output_root) / "coherence" / "cross_spectrum"
        destination.mkdir(parents=True, exist_ok=True)
        if self.query_indices is None:
            raise RuntimeError("cross-spectrum query indices are unavailable")

        generated = torch.stack(self.generated_coefficients)
        reference = torch.stack(self.reference_coefficients)
        selected_sample_count = len(self.sample_ids)
        if self.aggregation == "training_aligned":
            order = np.random.default_rng(self.ensemble_seed).permutation(selected_sample_count)
            generated = generated[torch.as_tensor(order, dtype=torch.long)]
            reference = reference[torch.as_tensor(order, dtype=torch.long)]
            ordered_sample_ids = [self.sample_ids[index] for index in order]
            ensemble_count = selected_sample_count // self.ensemble_size
            if ensemble_count < 1:
                raise ValueError(
                    "training-aligned cross-spectrum evaluation requires at least "
                    f"{self.ensemble_size} selected snapshots; received {selected_sample_count}"
                )
            used_sample_count = ensemble_count * self.ensemble_size
            ensemble_slices = tuple(
                slice(start, start + self.ensemble_size)
                for start in range(0, used_sample_count, self.ensemble_size)
            )
            ensemble_policy = "training_aligned_fixed_size_nonoverlapping"
        else:
            ordered_sample_ids = list(self.sample_ids)
            used_sample_count = selected_sample_count
            ensemble_count = 1
            ensemble_slices = (slice(0, selected_sample_count),)
            ensemble_policy = "single_pooled_selected_snapshot_ensemble"

        used_sample_ids = ordered_sample_ids[:used_sample_count]
        dropped_sample_ids = ordered_sample_ids[used_sample_count:]
        absolute_lists: dict[str, list[np.ndarray]] = {}
        score_lists: dict[str, list[np.ndarray]] = {}
        band_ids = self.family.band_ids.detach().cpu()
        for ensemble_slice in ensemble_slices:
            ensemble_generated = generated[ensemble_slice]
            ensemble_reference = reference[ensemble_slice]
            if self.family.component_weights.get("same_frequency", 0.0) > 0:
                generated_coherence = spectral_coherence(ensemble_generated, self.family.eps)
                reference_coherence = spectral_coherence(ensemble_reference, self.family.eps)
                absolute_lists.setdefault("same_frequency", []).append(
                    pair_mean_square_values(
                        generated_coherence, reference_coherence, self.family.pairs
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                score_lists.setdefault("same_frequency", []).append(
                    pair_symmetric_coherence_scores(
                        generated_coherence,
                        reference_coherence,
                        self.family.pairs,
                        self.family.eps,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

            energies_generated = energies_reference = None
            if (
                self.family.component_weights.get("cross_frequency", 0.0) > 0
                or self.family.component_weights.get("band_energy", 0.0) > 0
            ):
                energies_generated = band_energies(ensemble_generated, band_ids)
                energies_reference = band_energies(ensemble_reference, band_ids)
            if self.family.component_weights.get("cross_frequency", 0.0) > 0:
                assert energies_generated is not None and energies_reference is not None
                generated_coupling = normalized_cross_band_coupling(
                    energies_generated, self.family.eps
                )
                reference_coupling = normalized_cross_band_coupling(
                    energies_reference, self.family.eps
                )
                absolute_lists.setdefault("cross_frequency", []).append(
                    off_diagonal_pair_mean_square_values(
                        generated_coupling, reference_coupling, self.family.pairs
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                score_lists.setdefault("cross_frequency", []).append(
                    off_diagonal_pair_symmetric_coherence_scores(
                        generated_coupling,
                        reference_coupling,
                        self.family.pairs,
                        self.family.eps,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
            if self.family.component_weights.get("band_energy", 0.0) > 0:
                assert energies_generated is not None and energies_reference is not None
                generated_log_power = (energies_generated.mean(dim=0) + self.family.eps).log()
                reference_log_power = (energies_reference.mean(dim=0) + self.family.eps).log()
                difference = generated_log_power - reference_log_power
                absolute_lists.setdefault("band_energy", []).append(
                    difference.square().reshape(-1).numpy()
                )
                denominator = (generated_log_power.abs() + reference_log_power.abs()).clamp_min(
                    self.family.eps
                )
                score_lists.setdefault("band_energy", []).append(
                    (1.0 - difference.abs() / denominator).clamp(0.0, 1.0).reshape(-1).numpy()
                )

        absolute_by_ensemble = {key: np.stack(values) for key, values in absolute_lists.items()}
        scores_by_ensemble = {key: np.stack(values) for key, values in score_lists.items()}

        def ensemble_std(values: np.ndarray) -> np.ndarray:
            if values.shape[0] < 2:
                return np.zeros(values.shape[1:], dtype=np.float64)
            return np.asarray(values).std(axis=0, ddof=1)

        absolute_discrepancies = {
            key: values.mean(axis=0) for key, values in absolute_by_ensemble.items()
        }
        absolute_discrepancy_std = {
            key: ensemble_std(values) for key, values in absolute_by_ensemble.items()
        }
        coherence_scores = {key: values.mean(axis=0) for key, values in scores_by_ensemble.items()}
        coherence_score_std = {
            key: ensemble_std(values) for key, values in scores_by_ensemble.items()
        }

        labels_by_component = {
            "same_frequency": self.pair_labels,
            "cross_frequency": self.pair_labels,
            "band_energy": self.band_field_labels,
        }
        component_absolute_by_ensemble = {
            key: self.family.component_weights[key] * values.mean(axis=1)
            for key, values in absolute_by_ensemble.items()
        }
        component_score_by_ensemble = {
            key: values.mean(axis=1) for key, values in scores_by_ensemble.items()
        }
        component_absolute_totals = {
            key: float(values.mean()) for key, values in component_absolute_by_ensemble.items()
        }
        component_absolute_std = {
            key: float(ensemble_std(values[:, None])[0])
            for key, values in component_absolute_by_ensemble.items()
        }
        component_scores = {
            key: float(values.mean()) for key, values in component_score_by_ensemble.items()
        }
        component_score_std = {
            key: float(ensemble_std(values[:, None])[0])
            for key, values in component_score_by_ensemble.items()
        }
        family_absolute_by_ensemble = np.stack(
            tuple(component_absolute_by_ensemble.values()), axis=1
        ).sum(axis=1)
        family_absolute_total = float(family_absolute_by_ensemble.mean())
        family_absolute_std = float(ensemble_std(family_absolute_by_ensemble[:, None])[0])
        positive_weight = sum(
            self.family.component_weights[key]
            for key in component_score_by_ensemble
            if self.family.component_weights[key] > 0
        )
        family_score_by_ensemble = (
            sum(
                self.family.component_weights[key] * values
                for key, values in component_score_by_ensemble.items()
            )
            / positive_weight
        )
        family_score = float(family_score_by_ensemble.mean())
        family_score_std = float(ensemble_std(family_score_by_ensemble[:, None])[0])

        payload_path = destination / "metrics.npz"
        np.savez_compressed(
            payload_path,
            same_frequency_absolute_discrepancy=absolute_discrepancies.get(
                "same_frequency", np.empty(0)
            ),
            same_frequency_absolute_discrepancy_std=absolute_discrepancy_std.get(
                "same_frequency", np.empty(0)
            ),
            same_frequency_absolute_discrepancy_by_ensemble=absolute_by_ensemble.get(
                "same_frequency", np.empty((ensemble_count, 0))
            ),
            same_frequency_coherence_score=coherence_scores.get("same_frequency", np.empty(0)),
            same_frequency_coherence_score_std=coherence_score_std.get(
                "same_frequency", np.empty(0)
            ),
            same_frequency_coherence_score_by_ensemble=scores_by_ensemble.get(
                "same_frequency", np.empty((ensemble_count, 0))
            ),
            cross_frequency_absolute_discrepancy=absolute_discrepancies.get(
                "cross_frequency", np.empty(0)
            ),
            cross_frequency_absolute_discrepancy_std=absolute_discrepancy_std.get(
                "cross_frequency", np.empty(0)
            ),
            cross_frequency_absolute_discrepancy_by_ensemble=absolute_by_ensemble.get(
                "cross_frequency", np.empty((ensemble_count, 0))
            ),
            cross_frequency_coherence_score=coherence_scores.get("cross_frequency", np.empty(0)),
            cross_frequency_coherence_score_std=coherence_score_std.get(
                "cross_frequency", np.empty(0)
            ),
            cross_frequency_coherence_score_by_ensemble=scores_by_ensemble.get(
                "cross_frequency", np.empty((ensemble_count, 0))
            ),
            band_energy_absolute_discrepancy=absolute_discrepancies.get("band_energy", np.empty(0)),
            band_energy_absolute_discrepancy_std=absolute_discrepancy_std.get(
                "band_energy", np.empty(0)
            ),
            band_energy_absolute_discrepancy_by_ensemble=absolute_by_ensemble.get(
                "band_energy", np.empty((ensemble_count, 0))
            ),
            band_energy_coherence_score=coherence_scores.get("band_energy", np.empty(0)),
            band_energy_coherence_score_std=coherence_score_std.get("band_energy", np.empty(0)),
            band_energy_coherence_score_by_ensemble=scores_by_ensemble.get(
                "band_energy", np.empty((ensemble_count, 0))
            ),
            component_absolute_totals=np.asarray(
                [component_absolute_totals[key] for key in component_absolute_totals]
            ),
            component_absolute_total_std=np.asarray(
                [component_absolute_std[key] for key in component_absolute_totals]
            ),
            component_coherence_scores=np.asarray(
                [component_scores[key] for key in component_scores]
            ),
            component_coherence_score_std=np.asarray(
                [component_score_std[key] for key in component_scores]
            ),
            component_names=np.asarray(list(component_scores)),
            family_absolute_total=np.asarray(family_absolute_total),
            family_absolute_total_std=np.asarray(family_absolute_std),
            family_coherence_score=np.asarray(family_score),
            family_coherence_score_std=np.asarray(family_score_std),
            pair_labels=np.asarray(self.pair_labels),
            band_field_labels=np.asarray(self.band_field_labels),
            sample_ids=np.asarray(used_sample_ids),
            selected_sample_ids=np.asarray(self.sample_ids),
            dropped_sample_ids=np.asarray(dropped_sample_ids),
            ensemble_sample_ids=np.asarray(used_sample_ids).reshape(ensemble_count, -1),
            aggregation=np.asarray(self.aggregation),
            ensemble_size=np.asarray(
                self.ensemble_size if self.aggregation == "training_aligned" else used_sample_count
            ),
            ensemble_count=np.asarray(ensemble_count),
            query_indices=self.query_indices.cpu().numpy(),
            split=np.asarray(split),
            units=np.asarray(self.family.units),
        )

        csv_path = destination / "metrics.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                (
                    "component",
                    "sub_term",
                    "absolute_discrepancy_mean",
                    "absolute_discrepancy_std",
                    "coherence_score_mean",
                    "coherence_score_std",
                )
            )
            for key, values in absolute_discrepancies.items():
                for index, label in enumerate(labels_by_component[key]):
                    writer.writerow(
                        (
                            key,
                            label,
                            values[index],
                            absolute_discrepancy_std[key][index],
                            coherence_scores[key][index],
                            coherence_score_std[key][index],
                        )
                    )
                writer.writerow(
                    (
                        key,
                        "component_total",
                        component_absolute_totals[key],
                        component_absolute_std[key],
                        component_scores[key],
                        component_score_std[key],
                    )
                )
            writer.writerow(
                (
                    "cross_spectrum",
                    "family_total",
                    family_absolute_total,
                    family_absolute_std,
                    family_score,
                    family_score_std,
                )
            )

        if self.aggregation == "training_aligned":
            ensemble_summary = (
                f"{ensemble_count}×{self.ensemble_size} ensembles · "
                f"n={used_sample_count}/{selected_sample_count}"
            )
            estimate_summary = "bars mean; whiskers ±1 SD"
        else:
            ensemble_summary = f"one pooled ensemble · n={used_sample_count}"
            estimate_summary = "single pooled estimate"
        subtitle = (
            f"{run_label} · {split} · {checkpoint_label}.pt · {ensemble_summary} · "
            f"{self.query_indices.numel():,} graph points · {estimate_summary}"
        )
        definitions = {
            "same_frequency": (
                "same_frequency_coherence.png",
                "Same-frequency spectral coherence",
                self.pair_labels,
            ),
            "cross_frequency": (
                "cross_frequency_coherence.png",
                "Cross-frequency spectral coherence",
                self.pair_labels,
            ),
            "band_energy": (
                "band_energy_coherence.png",
                "Spectral-band energy coherence",
                self.band_field_labels,
            ),
        }
        figures: dict[str, Path] = {}
        for key, scores in coherence_scores.items():
            filename, title, labels = definitions[key]
            figures[key] = destination / filename
            component_label = "Pair mean" if key != "band_energy" else "Band–field mean"
            render_cross_spectrum_score_bars(
                np.concatenate((scores, [component_scores[key], family_score])),
                (*labels, component_label, "Overall score"),
                (*("detail" for _ in labels), "component_total", "family_total"),
                figures[key],
                title=title,
                subtitle=subtitle,
                score_std=np.concatenate(
                    (
                        coherence_score_std[key],
                        [component_score_std[key], family_score_std],
                    )
                ),
            )

        statistics = {
            key: {
                "sub_terms": {
                    label: {
                        "absolute_discrepancy": float(values[index]),
                        "absolute_discrepancy_std": float(absolute_discrepancy_std[key][index]),
                        "coherence_score": float(coherence_scores[key][index]),
                        "coherence_score_std": float(coherence_score_std[key][index]),
                    }
                    for index, label in enumerate(labels_by_component[key])
                },
                "weighted_absolute_discrepancy": component_absolute_totals[key],
                "weighted_absolute_discrepancy_std": component_absolute_std[key],
                "coherence_score": component_scores[key],
                "coherence_score_std": component_score_std[key],
            }
            for key, values in absolute_discrepancies.items()
        }
        statistics["family_total"] = {
            "weighted_absolute_discrepancy": family_absolute_total,
            "weighted_absolute_discrepancy_std": family_absolute_std,
            "coherence_score": family_score,
            "coherence_score_std": family_score_std,
        }
        report = {
            "family": "cross_spectrum",
            "metric": "paired_reconstruction_to_ground_truth_graph_cross_spectrum_coherence",
            "coherence_score": {
                "range": [0.0, 1.0],
                "perfect_agreement": 1.0,
                "definition": "1 - ||S_reconstruction - S_ground_truth||_2 / "
                "max(||S_reconstruction||_2 + ||S_ground_truth||_2, epsilon)",
                "absolute_discrepancy": "mean squared spectral difference",
            },
            "split": split,
            "selected_sample_count": selected_sample_count,
            "used_sample_count": used_sample_count,
            "dropped_sample_count": len(dropped_sample_ids),
            "dropped_sample_ids": dropped_sample_ids,
            "ensemble": {
                "policy": ensemble_policy,
                "aggregation": self.aggregation,
                "minimum_size": self.family.required_batch_size,
                "configured_training_batch_size": self.ensemble_size,
                "grouping_seed": self.ensemble_seed,
                "grouping_policy": (
                    "deterministic_seeded_permutation_then_complete_chunks"
                    if self.aggregation == "training_aligned"
                    else "selected_order_single_group"
                ),
                "ensemble_size": (
                    self.ensemble_size
                    if self.aggregation == "training_aligned"
                    else used_sample_count
                ),
                "ensemble_count": ensemble_count,
                "sample_count": used_sample_count,
                "sample_ids": used_sample_ids,
                "sample_ids_by_ensemble": np.asarray(used_sample_ids)
                .reshape(ensemble_count, -1)
                .tolist(),
                "summary_statistics": (
                    "mean and sample standard deviation across ensembles"
                    if self.aggregation == "training_aligned"
                    else "single pooled estimate with zero ensemble spread"
                ),
                "stored_representation": "graph_fourier_coefficients",
            },
            "target_use": "paired_supervised",
            "units": self.family.units,
            "field_names": list(self.family.field_names),
            "field_pairs": list(self.pair_labels),
            "bands": list(self.family.band_names),
            "family_weight": float(self.family.family_weight),
            "component_weights": dict(self.family.component_weights),
            "statistic_scale": "bounded_linear_0_to_1",
            "graph": {
                "query_policy": "fixed_shared",
                "query_seed": self.query_seed,
                "query_point_count": int(self.query_indices.numel()),
                "geometry_sha256": self.family.geometry_sha256,
                "k_neighbors": self.family.k_neighbors,
                "resolved_sigma": self.family.resolved_sigma,
                "num_modes": int(self.family.eigenvalues.numel()),
                "eigenvalues": self.family.eigenvalues.detach().cpu().tolist(),
                "band_mode_ids": self.family.band_ids.detach().cpu().tolist(),
            },
            "settings": self.config,
            "statistics": statistics,
            "artifacts": {
                "metrics_csv": csv_path.name,
                "metrics_payload": payload_path.name,
                "figures": {name: path.name for name, path in figures.items()},
            },
        }
        report_path = destination / "report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return {
            "family": "cross_spectrum",
            "directory": str(destination),
            "report": str(report_path),
            "figures": {name: str(path) for name, path in figures.items()},
        }


def build_coherence_accumulators(
    requested: Sequence[str],
    runtime: Any,
    *,
    extra_views: bool = False,
    evaluated_sample_count: int | None = None,
    cross_spectrum_aggregation: str = "training_aligned",
) -> dict[str, Any]:
    normalized = tuple(dict.fromkeys(str(name).strip().lower() for name in requested))
    unsupported = sorted(set(normalized) - set(SUPPORTED_COHERENCE_FAMILIES))
    if unsupported:
        raise ValueError(
            f"unsupported coherence families: {unsupported}; "
            f"supported families: {list(SUPPORTED_COHERENCE_FAMILIES)}"
        )
    accumulators = {}
    for name in normalized:
        if name == "global_distribution":
            accumulators[name] = GlobalDistributionAccumulator.build(
                runtime,
                extra_view=extra_views,
                evaluated_sample_count=evaluated_sample_count,
            )
        else:
            accumulators[name] = CrossSpectrumAccumulator.build(
                runtime,
                aggregation=cross_spectrum_aggregation,
            )
    return accumulators
