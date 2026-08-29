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
    if scale == "log":
        finite_values = finite_values[finite_values > 0.0]
        if not finite_values.size:
            raise ValueError("log-scale coherence plot contains no positive values")
        axis.set_ylim(float(finite_values.min()) * 0.75, float(finite_values.max()) * 1.25)
    else:
        upper = float(finite_values.max()) * 1.08 if finite_values.size else 1.0
        axis.set_ylim(0.0, max(upper, np.finfo(np.float64).eps))
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
    dpi: int = 300,
) -> Path:
    """Render bounded spectral-agreement scores as an adaptive horizontal bar chart."""
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
    for position, score, role in zip(positions, scores, roles):
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
        inside = score > 0.925
        axis.text(
            score - 0.014 if inside else score + 0.014,
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

    @classmethod
    def build(cls, runtime: Any) -> GlobalDistributionAccumulator:
        settings = _evaluation_family_config(
            runtime.config, "global_distribution", runtime.dataset.field_names
        )
        family = build_coherence_family(
            "global_distribution", settings, runtime.dataset.data_spec, runtime.dataset.normalizer
        ).to(runtime.device)
        family.eval()
        return cls(family=family, data_spec=runtime.dataset.data_spec, config=settings)

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
        }


@dataclass
class CrossSpectrumAccumulator:
    """Collect compact graph-Fourier coefficients for one pooled split estimate."""

    family: torch.nn.Module
    config: dict[str, Any]
    query_point_count: int
    query_seed: int
    generated_coefficients: list[torch.Tensor] = field(default_factory=list)
    reference_coefficients: list[torch.Tensor] = field(default_factory=list)
    sample_ids: list[str] = field(default_factory=list)
    query_indices: torch.Tensor | None = None
    fixed_coordinates: torch.Tensor | None = None

    @classmethod
    def build(cls, runtime: Any) -> CrossSpectrumAccumulator:
        settings = _evaluation_family_config(
            runtime.config, "cross_spectrum", runtime.dataset.field_names
        )
        family = build_coherence_family(
            "cross_spectrum", settings, runtime.dataset.data_spec, runtime.dataset.normalizer
        ).to(runtime.device)
        family.eval()
        coherence = runtime.config.get("coherence", {})
        compute = coherence.get("compute_budget", {}) if isinstance(coherence, Mapping) else {}
        query_point_count = int(compute.get("point_count", 4096))
        query_seed = int(compute.get("query_seed", 100045))
        return cls(
            family=family,
            config=settings,
            query_point_count=query_point_count,
            query_seed=query_seed,
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
        absolute_discrepancies: dict[str, np.ndarray] = {}
        coherence_scores: dict[str, np.ndarray] = {}
        if self.family.component_weights.get("same_frequency", 0.0) > 0:
            generated_coherence = spectral_coherence(generated, self.family.eps)
            reference_coherence = spectral_coherence(reference, self.family.eps)
            absolute_discrepancies["same_frequency"] = (
                pair_mean_square_values(generated_coherence, reference_coherence, self.family.pairs)
                .detach()
                .cpu()
                .numpy()
            )
            coherence_scores["same_frequency"] = (
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
            band_ids = self.family.band_ids.detach().cpu()
            energies_generated = band_energies(generated, band_ids)
            energies_reference = band_energies(reference, band_ids)
        if self.family.component_weights.get("cross_frequency", 0.0) > 0:
            assert energies_generated is not None and energies_reference is not None
            generated_coupling = normalized_cross_band_coupling(energies_generated, self.family.eps)
            reference_coupling = normalized_cross_band_coupling(energies_reference, self.family.eps)
            absolute_discrepancies["cross_frequency"] = (
                off_diagonal_pair_mean_square_values(
                    generated_coupling, reference_coupling, self.family.pairs
                )
                .detach()
                .cpu()
                .numpy()
            )
            coherence_scores["cross_frequency"] = (
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
            absolute_discrepancies["band_energy"] = difference.square().reshape(-1).numpy()
            denominator = (generated_log_power.abs() + reference_log_power.abs()).clamp_min(
                self.family.eps
            )
            coherence_scores["band_energy"] = (
                (1.0 - difference.abs() / denominator).clamp(0.0, 1.0).reshape(-1).numpy()
            )

        labels_by_component = {
            "same_frequency": self.pair_labels,
            "cross_frequency": self.pair_labels,
            "band_energy": self.band_field_labels,
        }
        component_absolute_totals = {
            key: float(self.family.component_weights[key] * values.mean())
            for key, values in absolute_discrepancies.items()
        }
        component_scores = {key: float(values.mean()) for key, values in coherence_scores.items()}
        family_absolute_total = float(sum(component_absolute_totals.values()))
        positive_weight = sum(
            self.family.component_weights[key]
            for key in component_scores
            if self.family.component_weights[key] > 0
        )
        family_score = float(
            sum(
                self.family.component_weights[key] * component_scores[key]
                for key in component_scores
            )
            / positive_weight
        )

        payload_path = destination / "metrics.npz"
        np.savez_compressed(
            payload_path,
            same_frequency_absolute_discrepancy=absolute_discrepancies.get(
                "same_frequency", np.empty(0)
            ),
            same_frequency_coherence_score=coherence_scores.get("same_frequency", np.empty(0)),
            cross_frequency_absolute_discrepancy=absolute_discrepancies.get(
                "cross_frequency", np.empty(0)
            ),
            cross_frequency_coherence_score=coherence_scores.get("cross_frequency", np.empty(0)),
            band_energy_absolute_discrepancy=absolute_discrepancies.get("band_energy", np.empty(0)),
            band_energy_coherence_score=coherence_scores.get("band_energy", np.empty(0)),
            component_absolute_totals=np.asarray(
                [component_absolute_totals[key] for key in component_absolute_totals]
            ),
            component_coherence_scores=np.asarray(
                [component_scores[key] for key in component_scores]
            ),
            component_names=np.asarray(list(component_scores)),
            family_absolute_total=np.asarray(family_absolute_total),
            family_coherence_score=np.asarray(family_score),
            pair_labels=np.asarray(self.pair_labels),
            band_field_labels=np.asarray(self.band_field_labels),
            sample_ids=np.asarray(self.sample_ids),
            query_indices=self.query_indices.cpu().numpy(),
            split=np.asarray(split),
            units=np.asarray(self.family.units),
        )

        csv_path = destination / "metrics.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("component", "sub_term", "absolute_discrepancy", "coherence_score"))
            for key, values in absolute_discrepancies.items():
                for index, label in enumerate(labels_by_component[key]):
                    writer.writerow((key, label, values[index], coherence_scores[key][index]))
                writer.writerow(
                    (
                        key,
                        "component_total",
                        component_absolute_totals[key],
                        component_scores[key],
                    )
                )
            writer.writerow(("cross_spectrum", "family_total", family_absolute_total, family_score))

        subtitle = (
            f"{run_label} · {split} · {checkpoint_label}.pt · n={len(self.sample_ids)} pooled · "
            f"{self.query_indices.numel():,} graph points · 100% = exact agreement"
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
            )

        statistics = {
            key: {
                "sub_terms": {
                    label: {
                        "absolute_discrepancy": float(values[index]),
                        "coherence_score": float(coherence_scores[key][index]),
                    }
                    for index, label in enumerate(labels_by_component[key])
                },
                "weighted_absolute_discrepancy": component_absolute_totals[key],
                "coherence_score": component_scores[key],
            }
            for key, values in absolute_discrepancies.items()
        }
        statistics["family_total"] = {
            "weighted_absolute_discrepancy": family_absolute_total,
            "coherence_score": family_score,
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
            "selected_sample_count": len(self.sample_ids),
            "used_sample_count": len(self.sample_ids),
            "dropped_sample_count": 0,
            "dropped_sample_ids": [],
            "ensemble": {
                "policy": "single_pooled_selected_snapshot_ensemble",
                "minimum_size": self.family.required_batch_size,
                "sample_count": len(self.sample_ids),
                "sample_ids": self.sample_ids,
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


def build_coherence_accumulators(requested: Sequence[str], runtime: Any) -> dict[str, Any]:
    normalized = tuple(dict.fromkeys(str(name).strip().lower() for name in requested))
    unsupported = sorted(set(normalized) - set(SUPPORTED_COHERENCE_FAMILIES))
    if unsupported:
        raise ValueError(
            f"unsupported coherence families: {unsupported}; "
            f"supported families: {list(SUPPORTED_COHERENCE_FAMILIES)}"
        )
    builders = {
        "global_distribution": GlobalDistributionAccumulator.build,
        "cross_spectrum": CrossSpectrumAccumulator.build,
    }
    return {name: builders[name](runtime) for name in normalized}
