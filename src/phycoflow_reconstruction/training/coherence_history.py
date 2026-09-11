"""Generic extraction and rendering of coherence-component training histories."""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import load_config
from .history_plotting import (
    HISTORY_FAMILY_COLORS,
    HISTORY_FAMILY_LINESTYLES,
    HISTORY_MUTED_TEXT_COLOR,
    HISTORY_TEXT_COLOR,
    style_history_axis,
)

_COMPONENT_PREFIX = "coherence_component/"
_FAMILY_PREFIX = "coherence_family/"
_FAMILY_WEIGHTED_SUFFIX = "/weighted_contribution"
_SUMMARY_COLOR = "#16827C"


@dataclass(frozen=True)
class CoherenceComponentHistory:
    """One family-owned component history on its observed epoch coordinates."""

    family: str
    component: str
    epochs: tuple[float, ...]
    raw: tuple[float, ...]
    weighted: tuple[float, ...]
    partial_epochs: tuple[float, ...]


@dataclass(frozen=True)
class CoherenceHistoryData:
    """Renderer-independent coherence history grouped by configured family."""

    components: tuple[CoherenceComponentHistory, ...]
    family_order: tuple[str, ...]
    total_epochs: tuple[float, ...]
    total_values: tuple[float, ...]
    family_totals: dict[str, tuple[tuple[float, ...], tuple[float, ...]]]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                warnings.warn(
                    f"ignored incomplete history row {line_number} in {path}",
                    stacklevel=2,
                )
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _epoch(row: dict[str, Any]) -> float:
    value = _finite_number(row.get("epoch"))
    if value is not None:
        return value
    return float(int(row.get("step", 0)))


def _ordered_points(points: dict[float, float]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    ordered = sorted(points.items())
    return tuple(epoch for epoch, _ in ordered), tuple(value for _, value in ordered)


def _configured_families(config: dict[str, Any]) -> tuple[str, ...]:
    families = config.get("coherence", {}).get("families", {})
    return tuple(
        str(name)
        for name, settings in families.items()
        if isinstance(settings, dict) and bool(settings.get("enabled", True))
    )


def _effective_multiplier(
    row: dict[str, Any], config: dict[str, Any], family: str, component: str
) -> float:
    family_config = config.get("coherence", {}).get("families", {}).get(family, {})
    component_key = component.split(".", 1)[0]
    component_config = family_config.get("components", {}).get(component_key, {})
    inner = float(component_config.get("weight", 1.0))
    outer = float(family_config.get("weight", 1.0))
    calibration = float(row.get(f"{_FAMILY_PREFIX}{family}/calibration_scale", 1.0))
    return inner * outer * calibration


def extract_coherence_history(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> CoherenceHistoryData:
    """Extract new namespaced metrics with a fallback for historical flat keys."""
    configured = _configured_families(config)
    raw_points: dict[tuple[str, str], dict[float, float]] = {}
    weighted_points: dict[tuple[str, str], dict[float, float]] = {}
    standardized: set[tuple[str, str, float]] = set()
    partial: dict[tuple[str, str], set[float]] = {}

    for row in rows:
        epoch = _epoch(row)
        is_partial = row.get("epoch_complete") is False
        for key, value in row.items():
            if not key.startswith(_COMPONENT_PREFIX):
                continue
            parts = key.split("/", 3)
            if len(parts) != 4 or parts[3] not in {"raw", "weighted_contribution"}:
                continue
            numeric = _finite_number(value)
            if numeric is None:
                continue
            identity = (parts[1], parts[2])
            standardized.add((*identity, epoch))
            destination = raw_points if parts[3] == "raw" else weighted_points
            destination.setdefault(identity, {})[epoch] = numeric
            if is_partial:
                partial.setdefault(identity, set()).add(epoch)

    # Rows written before the namespaced metric contract already contain full
    # family.component paths.  Recover them without requiring model loading.
    for row in rows:
        epoch = _epoch(row)
        is_partial = row.get("epoch_complete") is False
        for family in configured:
            prefix = f"{family}."
            for key, value in row.items():
                if not key.startswith(prefix):
                    continue
                component = key[len(prefix) :]
                identity = (family, component)
                if (*identity, epoch) in standardized:
                    continue
                numeric = _finite_number(value)
                if numeric is None:
                    continue
                raw_points.setdefault(identity, {})[epoch] = numeric
                weighted_points.setdefault(identity, {})[epoch] = numeric * _effective_multiplier(
                    row, config, family, component
                )
                if is_partial:
                    partial.setdefault(identity, set()).add(epoch)

    discovered = tuple(dict.fromkeys(family for family, _ in raw_points))
    family_order = (*configured, *(name for name in discovered if name not in configured))
    family_rank = {name: index for index, name in enumerate(family_order)}
    configured_component_order = {
        family: {
            str(name): index
            for index, name in enumerate(
                config.get("coherence", {})
                .get("families", {})
                .get(family, {})
                .get("components", {})
            )
        }
        for family in family_order
    }
    components = []
    for identity, values in sorted(
        raw_points.items(),
        key=lambda item: (
            family_rank.get(item[0][0], 10_000),
            configured_component_order.get(item[0][0], {}).get(item[0][1].split(".", 1)[0], 10_000),
            item[0][1],
        ),
    ):
        epochs, raw = _ordered_points(values)
        weighted_lookup = weighted_points.get(identity, {})
        weighted = tuple(weighted_lookup.get(epoch, value) for epoch, value in zip(epochs, raw))
        components.append(
            CoherenceComponentHistory(
                family=identity[0],
                component=identity[1],
                epochs=epochs,
                raw=raw,
                weighted=weighted,
                partial_epochs=tuple(sorted(partial.get(identity, set()))),
            )
        )

    total = {
        _epoch(row): numeric
        for row in rows
        if (numeric := _finite_number(row.get("coherence_loss"))) is not None
    }
    total_epochs, total_values = _ordered_points(total)
    family_totals = {}
    for family in family_order:
        key = f"{_FAMILY_PREFIX}{family}{_FAMILY_WEIGHTED_SUFFIX}"
        points = {
            _epoch(row): numeric
            for row in rows
            if (numeric := _finite_number(row.get(key))) is not None
        }
        if points:
            family_totals[family] = _ordered_points(points)
    return CoherenceHistoryData(
        components=tuple(components),
        family_order=tuple(family_order),
        total_epochs=total_epochs,
        total_values=total_values,
        family_totals=family_totals,
    )


def _display_name(value: str) -> str:
    replacements = {"w2": "W2", "swd": "SWD", "rbf": "RBF", "qmc": "QMC"}
    words = []
    for word in value.replace("_", " ").split():
        words.append(replacements.get(word.lower(), word))
    label = " ".join(words)
    return f"{label[:1].upper()}{label[1:]}"


def _component_label(component: str) -> str:
    return " · ".join(_display_name(part) for part in component.split("."))


def build_coherence_history_figure(data: CoherenceHistoryData, plt, *, description: str):
    """Build an adaptive family-grouped small-multiples coherence figure."""
    by_family = {
        family: tuple(component for component in data.components if component.family == family)
        for family in data.family_order
    }
    by_family = {family: components for family, components in by_family.items() if components}
    family_rows = {
        family: math.ceil(len(components) / 2) for family, components in by_family.items()
    }
    total_component_rows = sum(family_rows.values())
    figure = plt.figure(
        figsize=(12.4, 3.4 + 3.0 * total_component_rows),
        constrained_layout=True,
        facecolor="white",
    )
    subfigures = figure.subfigures(
        1 + len(by_family),
        1,
        height_ratios=[1.15, *(family_rows.values())],
        squeeze=False,
    ).ravel()
    figure.suptitle(
        f"{description.replace(':', ' · ').replace('_', ' ')} coherence component history",
        color=HISTORY_TEXT_COLOR,
        fontsize=15,
        fontweight="medium",
    )

    summary = subfigures[0].subplots()
    summary_values = list(data.total_values)
    if data.total_values:
        summary.plot(
            data.total_epochs,
            data.total_values,
            color=_SUMMARY_COLOR,
            linewidth=2.2,
            label="Total coherence",
        )
    if len(data.family_totals) > 1:
        for index, family in enumerate(data.family_order):
            if family not in data.family_totals:
                continue
            epochs, values = data.family_totals[family]
            summary_values.extend(values)
            summary.plot(
                epochs,
                values,
                color=HISTORY_FAMILY_COLORS[index % len(HISTORY_FAMILY_COLORS)],
                linestyle=HISTORY_FAMILY_LINESTYLES[index % len(HISTORY_FAMILY_LINESTYLES)],
                linewidth=1.65,
                label=_display_name(family),
            )
    summary.set_title(
        "Weighted coherence objective",
        loc="left",
        color=HISTORY_TEXT_COLOR,
        fontsize=12.5,
        fontweight="medium",
        pad=10,
    )
    summary.set_xlabel("Training epoch")
    summary.set_ylabel("Objective value")
    style_history_axis(summary, summary_values)
    if summary.lines:
        summary.legend(
            loc="best",
            frameon=True,
            facecolor="white",
            edgecolor="#D4D9E2",
            framealpha=0.94,
            fontsize=8.6,
            ncol=min(3, len(summary.lines)),
        )

    for family_index, (family, components) in enumerate(by_family.items()):
        subfigure = subfigures[1 + family_index]
        subfigure.suptitle(
            _display_name(family),
            x=0.01,
            ha="left",
            color=HISTORY_TEXT_COLOR,
            fontsize=13.0,
            fontweight="medium",
        )
        grid = subfigure.add_gridspec(family_rows[family], 2)
        axes = []
        for component_index in range(len(components)):
            row = component_index // 2
            column = component_index % 2
            cell = (
                grid[row, :]
                if component_index == len(components) - 1 and len(components) % 2
                else grid[row, column]
            )
            axes.append(subfigure.add_subplot(cell))
        color_index = data.family_order.index(family)
        color = HISTORY_FAMILY_COLORS[color_index % len(HISTORY_FAMILY_COLORS)]
        for axis, component in zip(axes, components):
            marker = "o" if len(component.epochs) <= 12 else None
            axis.plot(
                component.epochs,
                component.raw,
                color=color,
                linewidth=1.9,
                marker=marker,
                markersize=3.8 if marker else None,
            )
            if component.partial_epochs:
                lookup = dict(zip(component.epochs, component.raw))
                visible = [epoch for epoch in component.partial_epochs if epoch in lookup]
                axis.scatter(
                    visible,
                    [lookup[epoch] for epoch in visible],
                    facecolors="white",
                    edgecolors=color,
                    linewidths=1.2,
                    s=28,
                    zorder=3,
                )
            ratios = [
                weighted / raw
                for raw, weighted in zip(component.raw, component.weighted)
                if raw != 0.0
            ]
            multiplier = ratios[-1] if ratios else 1.0
            axis.set_title(
                _component_label(component.component),
                loc="left",
                color=HISTORY_TEXT_COLOR,
                fontsize=11.5,
                fontweight="medium",
                pad=9,
            )
            axis.text(
                1.0,
                1.015,
                f"latest {component.raw[-1]:.3e} · effective ×{multiplier:.3g}",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                color=HISTORY_MUTED_TEXT_COLOR,
                fontsize=8.2,
            )
            axis.set_xlabel("Training epoch")
            axis.set_ylabel("Raw component loss")
            style_history_axis(axis, component.raw)
    return figure


def render_coherence_history(
    run_dir: str | Path,
    *,
    description: str | None = None,
    output_path: str | Path | None = None,
    pyplot=None,
) -> Path | None:
    """Render ``coherence_history.png`` from run metadata without loading a model."""
    run_dir = Path(run_dir)
    rows = _read_jsonl(run_dir / "metrics" / "history.jsonl")
    config_path = run_dir / "resolved_config.yaml"
    config = load_config(config_path) if config_path.is_file() else {}
    data = extract_coherence_history(rows, config)
    if not data.components:
        return None
    if pyplot is None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            from matplotlib import pyplot
        except ImportError:
            warnings.warn(
                "matplotlib is unavailable; coherence_history.png cannot be generated",
                stacklevel=2,
            )
            return None
    pyplot.rcParams["svg.fonttype"] = "none"
    if description is None:
        stage = str(config.get("stage", "training")).replace("_training", "")
        model = str(config.get("model", {}).get("name", "model"))
        description = f"{stage}:{model}"
    figure = build_coherence_history_figure(data, pyplot, description=description)
    destination = (
        Path(output_path) if output_path is not None else run_dir / "coherence_history.png"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    figure.savefig(temporary, dpi=180, format="png")
    pyplot.close(figure)
    os.replace(temporary, destination)
    return destination
