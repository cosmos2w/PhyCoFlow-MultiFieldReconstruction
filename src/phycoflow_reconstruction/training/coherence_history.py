"""Generic extraction and rendering of coherence-component training histories."""

from __future__ import annotations

import json
import math
import os
import statistics
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import load_config
from .history_plotting import (
    HISTORY_MUTED_TEXT_COLOR,
    HISTORY_TEXT_COLOR,
    history_family_style,
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


def _rolling_median(values: tuple[float, ...]) -> tuple[float, ...]:
    """Return a centered, two-percent-window median for visual trend context."""
    if len(values) < 8:
        return values
    window = max(3, round(len(values) * 0.02))
    if window % 2 == 0:
        window += 1
    radius = window // 2
    return tuple(
        statistics.median(values[max(0, index - radius) : index + radius + 1])
        for index in range(len(values))
    )


def _stage_label(description: str) -> str:
    stage = description.split(":", 1)[0].replace("_", " ").replace("-", " ").strip()
    return stage.title() or "Training"


def _family_totals_from_evaluation(payload: Mapping[str, Any] | None) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    coherence = payload.get("coherence", {})
    families = coherence.get("families", {}) if isinstance(coherence, Mapping) else {}
    if not isinstance(families, Mapping):
        return {}
    return {
        str(name): numeric
        for name, value in families.items()
        if isinstance(value, Mapping)
        and (numeric := _finite_number(value.get("total"))) is not None
    }


def _family_scores_from_row(
    row: Mapping[str, Any], source_totals: Mapping[str, float]
) -> dict[str, float | None]:
    configured = row.get("family_source_normalized_scores")
    if isinstance(configured, Mapping):
        return {
            str(name): _finite_number(value)
            for name, value in configured.items()
        }
    metrics = row.get("metrics")
    candidate_totals = _family_totals_from_evaluation(metrics)
    scores = {}
    for family, source in source_totals.items():
        candidate = candidate_totals.get(family)
        scores[family] = candidate / source if candidate is not None and source != 0 else None
    return scores


def extract_adaptive_coherence_history(
    training_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    before: Mapping[str, Any] | None = None,
    selected: Mapping[str, Any] | None = None,
    after: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    steps_per_epoch: int | None = None,
) -> dict[str, Any]:
    """Extract adaptive fidelity and feasible-selector histories without a model.

    Training endpoint risks are paired to their same-row source risks. Selector
    scores remain dimensionless raw-family candidate/source ratios. Missing,
    null, or non-finite diagnostics stay missing instead of becoming zero.
    """
    config = config or {}
    epoch_axis = config.get("runtime", {}).get("plot_format") == "pdf"
    if epoch_axis and (steps_per_epoch is None or steps_per_epoch <= 0):
        raise ValueError("epoch reporting requires the saved steps_per_epoch mapping")
    divisor = int(steps_per_epoch) if epoch_axis else 1
    checkpointing = config.get("checkpointing", {})
    controller_settings = config.get("fidelity_controller", {})
    if not isinstance(controller_settings, Mapping):
        controller_settings = {}
    if not isinstance(checkpointing, Mapping):
        checkpointing = {}
    family_settings = config.get("coherence", {}).get("families", {})
    if not isinstance(family_settings, Mapping):
        family_settings = {}
    family_order = tuple(
        str(name)
        for name, settings in family_settings.items()
        if isinstance(settings, Mapping) and bool(settings.get("enabled", True))
    )
    source_totals = _family_totals_from_evaluation(before)
    field_names = tuple(
        str(name)
        for name in (before or {}).get("per_field_mse_normalized", {})
    )
    if not field_names:
        telemetry_names = {
            key[len("fidelity/") :].split("/", 1)[0]
            for row in training_rows
            for key in row
            if key.startswith("fidelity/") and key.endswith("/source_risk")
        }
        field_names = tuple(sorted(telemetry_names - {"total", "native"}))
    constraints = ("total", *field_names)
    if config.get("fidelity_controller", {}).get("native_loss_role") == "constraint":
        constraints = (*constraints, "native")

    ordered_training = sorted(
        (row for row in training_rows if _finite_number(row.get("step")) is not None),
        key=lambda row: int(float(row["step"])),
    )
    epoch_by_step: dict[int, float] = {}
    training_records = []
    for row in ordered_training:
        step = int(float(row["step"]))
        epoch = _finite_number(row.get("epoch"))
        if epoch is not None:
            epoch_by_step[step] = epoch
        # Rounded epoch indices can collapse separate training windows.
        x_value = float(step) / divisor
        fidelity = {}
        for name in constraints:
            prefix = f"fidelity/{name}/"
            source_risk = _finite_number(row.get(prefix + "source_risk"))
            live_risk = _finite_number(row.get(prefix + "live_risk"))
            relative_change = (
                live_risk / source_risk - 1.0
                if source_risk is not None and source_risk > 0.0 and live_risk is not None
                else None
            )
            cap_value = row.get(prefix + "cap_saturated")
            cap_fraction = _finite_number(
                row.get(prefix + "cap_saturated_fraction")
            )
            fidelity[name] = {
                "source_risk": source_risk,
                "live_risk": live_risk,
                "relative_change": (
                    relative_change if relative_change is not None and math.isfinite(relative_change) else None
                ),
                "violation": _finite_number(row.get(prefix + "violation")),
                "ema_violation": _finite_number(row.get(prefix + "ema_violation")),
                "multiplier": _finite_number(row.get(prefix + "multiplier")),
                "effective_primal_coefficient": _finite_number(
                    row.get(prefix + "effective_primal_coefficient")
                ),
                "cap_saturated": cap_value if isinstance(cap_value, bool) else None,
                "cap_saturated_fraction": (
                    cap_fraction
                    if cap_fraction is not None and 0.0 <= cap_fraction <= 1.0
                    else None
                ),
            }

        gradient_cosines = {}
        actual_dots = {}
        for key, value in row.items():
            if key.startswith("gradient/") and key.endswith("/cosine"):
                parts = key.split("/")
                if len(parts) == 4 and parts[1] in family_order and parts[2] in family_order:
                    gradient_cosines[f"{parts[1]}|{parts[2]}"] = _finite_number(value)
            elif key.startswith("update/actual_dot/"):
                name = key[len("update/actual_dot/") :]
                actual_dots[name] = _finite_number(value)
        training_records.append(
            {
                "step": step,
                "x": float(x_value),
                "epoch": epoch,
                "fidelity": fidelity,
                "gradient_cosines": gradient_cosines,
                "actual_dots": actual_dots,
            }
        )

    validation_records = []
    for row in validation_rows:
        step_value = _finite_number(row.get("step"))
        if step_value is None:
            continue
        step = int(step_value)
        epoch = epoch_by_step.get(step)
        if epoch is None:
            epoch = _finite_number(row.get("training_epoch"))
        validation_records.append(
            {
                "step": step,
                "x": float(step) / divisor,
                "epoch": epoch,
                "metric": _finite_number(row.get("metric")),
                "eligible": bool(row.get("eligible", False)),
                "failed_fidelity_fields": [
                    str(value) for value in row.get("failed_fidelity_fields", ())
                ],
                "family_source_normalized_scores": _family_scores_from_row(
                    row, source_totals
                ),
            }
        )
    validation_records.sort(key=lambda row: row["step"])

    selected_summary = None
    if isinstance(selected, Mapping):
        selected_step_value = _finite_number(selected.get("global_step"))
        selected_step = None if selected_step_value is None else int(selected_step_value)
        selected_summary = {
            "step": selected_step,
            "x": None if selected_step is None else float(selected_step) / divisor,
            "eligible": bool(selected.get("eligible", False)),
            "metric": _finite_number(selected.get("metric")),
            "family_source_normalized_scores": _family_scores_from_row(
                selected, source_totals
            ),
            "failed_fidelity_fields": [
                str(value) for value in selected.get("failed_fidelity_fields", ())
            ],
        }

    latest_cosine_record = next(
        (record for record in reversed(training_records) if record["gradient_cosines"]),
        None,
    )
    latest_matrix: dict[str, dict[str, float | None]] = {
        left: {right: None for right in family_order} for left in family_order
    }
    if latest_cosine_record is not None:
        for pair, value in latest_cosine_record["gradient_cosines"].items():
            left, right = pair.split("|", 1)
            latest_matrix[left][right] = value
            latest_matrix[right][left] = value

    return {
        "family_order": list(family_order),
        "constraints": list(constraints),
        "x_label": "Epoch" if epoch_axis else "Optimizer updates (run step)",
        "controller_settings": dict(controller_settings),
        "selector_settings": {
            "selection_metric": checkpointing.get("selection_metric"),
            "total_relative_budget": _finite_number(
                controller_settings.get("relative_budget_total", 0.05)
            ),
            "per_field_relative_budget": _finite_number(
                controller_settings.get("relative_budget_per_field", 0.05)
            ),
        },
        "source_family_totals": source_totals,
        "training_records": training_records,
        "validation_records": validation_records,
        "selected": selected_summary,
        "gradient_cosine_matrix": {
            "step": None if latest_cosine_record is None else latest_cosine_record["step"],
            "x": None if latest_cosine_record is None else latest_cosine_record["x"],
            "families": list(family_order),
            "values": latest_matrix,
        },
        "evaluation_endpoints": {
            "before": {
                "mse_normalized": _finite_number((before or {}).get("mse_normalized")),
                "per_field_mse_normalized": {
                    str(name): _finite_number(value)
                    for name, value in (before or {}).get(
                        "per_field_mse_normalized", {}
                    ).items()
                },
                "raw_family_totals": _family_totals_from_evaluation(before),
            },
            "selected": None if selected_summary is None else {
                "step": selected_summary["step"],
                "eligible": selected_summary["eligible"],
                "metric": selected_summary["metric"],
            },
            "after": {
                "mse_normalized": _finite_number((after or {}).get("mse_normalized")),
                "per_field_mse_normalized": {
                    str(name): _finite_number(value)
                    for name, value in (after or {}).get(
                        "per_field_mse_normalized", {}
                    ).items()
                },
                "raw_family_totals": _family_totals_from_evaluation(after),
            },
        },
    }


def _series_summary(values: list[Any]) -> dict[str, float | int | None]:
    finite = [value for item in values if (value := _finite_number(item)) is not None]
    if not finite:
        return {"count": 0, "minimum": None, "median": None, "maximum": None, "last": None}
    return {
        "count": len(finite),
        "minimum": min(finite),
        "median": statistics.median(finite),
        "maximum": max(finite),
        "last": finite[-1],
    }


def _adaptive_history_summaries(data: Mapping[str, Any]) -> dict[str, Any]:
    training = data["training_records"]
    constraints = data["constraints"]
    risk = {}
    controller = {}
    for name in constraints:
        risk[name] = _series_summary(
            [record["fidelity"][name]["relative_change"] for record in training]
        )
        controller[name] = {
            metric: _series_summary(
                [record["fidelity"][name][metric] for record in training]
            )
            for metric in (
                "violation",
                "ema_violation",
                "multiplier",
                "effective_primal_coefficient",
            )
        }
        controller[name]["cap_saturated_count"] = sum(
            record["fidelity"][name]["cap_saturated"] is True for record in training
        )
        controller[name]["cap_saturated_batch_fraction"] = _series_summary(
            [
                record["fidelity"][name]["cap_saturated_fraction"]
                for record in training
            ]
        )
    validation = data["validation_records"]
    family_scores = {
        family: _series_summary(
            [
                record["family_source_normalized_scores"].get(family)
                for record in validation
            ]
        )
        for family in data["family_order"]
    }
    actual_dot_names = sorted(
        {name for record in training for name in record["actual_dots"]}
    )
    actual_dots = {
        name: _series_summary(
            [record["actual_dots"].get(name) for record in training]
        )
        for name in actual_dot_names
    }
    return {
        "training_row_count": len(training),
        "controller_rows_with_any_fidelity_telemetry": sum(
            any(value["source_risk"] is not None for value in record["fidelity"].values())
            for record in training
        ),
        "validation_candidate_count": len(validation),
        "eligible_candidate_count": sum(record["eligible"] for record in validation),
        "ineligible_candidate_count": sum(not record["eligible"] for record in validation),
        "relative_fidelity_risk": risk,
        "controller": controller,
        "source_normalized_raw_family_validation": family_scores,
        "gradient_cosine_diagnostic_row_count": sum(
            bool(record["gradient_cosines"]) for record in training
        ),
        "latest_gradient_cosine_matrix_step": data["gradient_cosine_matrix"]["step"],
        "actual_adamw_gradient_dot": {
            "sign_interpretation": "negative dot predicts first-order local decrease of the named objective along the actual AdamW parameter displacement",
            "series": actual_dots,
        },
        "telemetry_ordering": {
            "pre_step": [
                "same-batch source/live fidelity risks",
                "normalized primal violations",
                "effective primal coefficients",
            ],
            "post_step": [
                "EMA violations",
                "dual multipliers and cap states",
            ],
            "actual_adamw_dots": "objective gradients dotted with the post-step actual parameter displacement",
        },
        "selected_candidate": data["selected"],
        "evaluation_endpoints": data["evaluation_endpoints"],
    }


def _adaptive_color(index: int) -> str:
    colors = ("#3B6EA8", "#D95F59", "#B58900", "#5B8E7D", "#8C6BB1", "#667085")
    return colors[index % len(colors)]


def build_adaptive_coherence_figures(data: Mapping[str, Any], plt) -> dict[str, Any]:
    """Build opt-in fidelity, controller, selector, and gradient audit plots."""
    from matplotlib.ticker import PercentFormatter

    records = data["training_records"]
    constraints = data["constraints"]
    family_order = data["family_order"]
    x_label = data["x_label"]
    total_budget = data["selector_settings"]["total_relative_budget"]
    field_budget = data["selector_settings"]["per_field_relative_budget"]
    total_budget = total_budget if total_budget is not None else 0.05
    field_budget = field_budget if field_budget is not None else 0.05
    x = [record["x"] for record in records]
    risk_figure, risk_axis = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    plotted_risk = False
    for index, name in enumerate(constraints):
        color = _adaptive_color(index)
        points = [record["fidelity"][name]["relative_change"] for record in records]
        visible = [(x_value, value) for x_value, value in zip(x, points) if value is not None]
        if visible:
            risk_axis.plot(
                [point[0] for point in visible],
                [point[1] for point in visible],
                color=color,
                linewidth=1.8 if name == "total" else 1.25,
                label="Total" if name == "total" else name,
            )
            plotted_risk = True
    risk_axis.axhline(0.0, color="#667085", linewidth=0.8)
    risk_axis.axhline(
        total_budget,
        color="#202733",
        linewidth=1.0,
        linestyle="--",
        label=f"Total +{total_budget:.0%} limit",
    )
    risk_axis.axhline(
        field_budget,
        color="#667085",
        linewidth=1.0,
        linestyle=":",
        label=f"Per-field +{field_budget:.0%} limit",
    )
    risk_axis.set_title("Paired same-batch source/live MSE change (pre-step)")
    risk_axis.set_xlabel(x_label)
    risk_axis.set_ylabel("Relative MSE change")
    risk_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    if plotted_risk:
        risk_axis.legend(fontsize=8.2, ncol=min(4, max(1, len(risk_axis.lines))))
    else:
        _empty_adaptive_axis(risk_axis, "No finite paired source/live risk telemetry")

    control_figure, axes = plt.subplots(
        4, 1, figsize=(12.0, max(9.0, 1.5 * len(constraints) + 6.0)),
        sharex=True, constrained_layout=True,
    )
    violation_axis, multiplier_axis, pressure_axis, cap_axis = axes
    for index, name in enumerate(constraints):
        color = _adaptive_color(index)
        violation = [record["fidelity"][name]["violation"] for record in records]
        ema = [record["fidelity"][name]["ema_violation"] for record in records]
        multiplier = [record["fidelity"][name]["multiplier"] for record in records]
        pressure = [
            record["fidelity"][name]["effective_primal_coefficient"]
            for record in records
        ]
        violation_points = [
            (x_value, value) for x_value, value in zip(x, violation) if value is not None
        ]
        ema_points = [(x_value, value) for x_value, value in zip(x, ema) if value is not None]
        multiplier_points = [
            (x_value, value) for x_value, value in zip(x, multiplier) if value is not None
        ]
        if violation_points:
            violation_axis.plot(
                [point[0] for point in violation_points],
                [point[1] for point in violation_points],
                color=color,
                linewidth=0.9,
                alpha=0.55,
                label=f"{name} violation",
            )
        if ema_points:
            violation_axis.plot(
                [point[0] for point in ema_points],
                [point[1] for point in ema_points],
                color=color,
                linewidth=1.8,
                linestyle="--",
                label=f"{name} EMA",
            )
        if multiplier_points:
            multiplier_axis.plot(
                [point[0] for point in multiplier_points],
                [point[1] for point in multiplier_points],
                color=color,
                linewidth=1.5,
                label=name,
            )
        pressure_points = [
            (x_value, value)
            for x_value, value in zip(x, pressure)
            if value is not None
        ]
        if pressure_points:
            pressure_axis.plot(
                [point[0] for point in pressure_points],
                [point[1] for point in pressure_points],
                color=color,
                linewidth=1.5,
                label=name,
            )
        cap_points = [
            (
                x_value,
                index,
                record["fidelity"][name]["cap_saturated_fraction"],
            )
            for x_value, record in zip(x, records)
            if record["fidelity"][name]["cap_saturated"] is True
        ]
        if cap_points:
            cap_axis.scatter(
                [point[0] for point in cap_points],
                [point[1] for point in cap_points],
                marker="x",
                color=color,
                s=[
                    36.0 + 90.0 * (1.0 if point[2] is None else point[2])
                    for point in cap_points
                ],
                label=name,
            )

    violation_axis.axhline(0.0, color="#202733", linewidth=0.8)
    violation_axis.set_title(
        "Pre-step primal violations · dashed lines are post-step EMA"
    )
    violation_axis.set_ylabel("Violation / source calibration scale")
    multiplier_axis.axhline(
        _finite_number(data["controller_settings"].get("multiplier_max")) or 100.0,
        color="#667085",
        linewidth=0.9,
        linestyle=":",
        label="Multiplier cap",
    )
    multiplier_axis.set_title("Dual multipliers (post-step)")
    multiplier_axis.set_ylabel("Multiplier")
    pressure_axis.set_title("Effective primal coefficients (pre-step)")
    pressure_axis.set_ylabel("Coefficient")
    cap_axis.set_title("Multiplier cap states (post-step; marker size = capped batch fraction)")
    cap_axis.set_ylabel("Constraint")
    cap_axis.set_yticks(range(len(constraints)), constraints)
    cap_axis.set_ylim(-0.6, max(0.6, len(constraints) - 0.4))
    cap_axis.set_xlabel(x_label)
    for axis in axes[:3]:
        if axis.lines:
            axis.legend(fontsize=7.5, ncol=3, loc="upper left")
    if cap_axis.collections:
        cap_axis.legend(fontsize=8.0, loc="upper right")
    if not any(record["fidelity"][name]["violation"] is not None for record in records for name in constraints):
        _empty_adaptive_axis(violation_axis, "No finite controller violations")
    if not any(record["fidelity"][name]["multiplier"] is not None for record in records for name in constraints):
        _empty_adaptive_axis(multiplier_axis, "No finite multipliers")
    if not any(
        record["fidelity"][name]["effective_primal_coefficient"] is not None
        for record in records
        for name in constraints
    ):
        _empty_adaptive_axis(pressure_axis, "No effective primal-coefficient telemetry")
    if not any(record["fidelity"][name]["cap_saturated"] is not None for record in records for name in constraints):
        _empty_adaptive_axis(cap_axis, "No cap-state telemetry")

    validation = data["validation_records"]
    selector_figure, selector_axis = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    plotted_selector = False
    for index, family in enumerate(family_order):
        color = _adaptive_color(index)
        points = [
            (record["x"], record["family_source_normalized_scores"].get(family), record["eligible"])
            for record in validation
        ]
        points = [point for point in points if point[1] is not None]
        if not points:
            continue
        selector_axis.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            color=color,
            linewidth=1.0,
            alpha=0.6,
            label=family,
        )
        for eligible, marker, marker_color, label in (
            (True, "o", color, "Eligible candidate"),
            (False, "x", "#D95F59", "Ineligible candidate"),
        ):
            subset = [point for point in points if point[2] is eligible]
            if subset:
                selector_axis.scatter(
                    [point[0] for point in subset],
                    [point[1] for point in subset],
                    marker=marker,
                    color=marker_color,
                    edgecolors="white" if marker == "o" else None,
                    linewidths=0.7 if marker == "o" else 1.2,
                    s=32,
                    label=label if family == next(iter(family_order), family) else None,
                    zorder=3,
                )
        plotted_selector = True
    selector_axis.axhline(1.0, color="#202733", linewidth=0.9, linestyle="--", label="Source parity")
    selected = data["selected"]
    if selected is not None and selected["x"] is not None:
        for family in family_order:
            value = selected["family_source_normalized_scores"].get(family)
            if value is not None:
                plotted_selector = True
                selector_axis.scatter(
                    [selected["x"]], [value], marker="*", s=135,
                    facecolor="#E0A526", edgecolor="#202733", linewidth=0.65,
                    zorder=5, label="Selected checkpoint" if family == next(iter(family_order), family) else None,
                )
        selector_axis.axvline(selected["x"], color="#B58900", linestyle=":", linewidth=0.9)
    selector_axis.set_title("Source-normalized raw-family validation trajectories")
    selector_axis.set_xlabel(x_label)
    selector_axis.set_ylabel("Raw family total · candidate / source")
    if plotted_selector:
        selector_axis.legend(fontsize=8, ncol=3)
    else:
        _empty_adaptive_axis(selector_axis, "No source-normalized family validation scores")

    gradient_figure, (matrix_axis, dot_axis) = plt.subplots(
        1, 2, figsize=(13.2, 6.0), constrained_layout=True,
        gridspec_kw={"width_ratios": [0.95, 1.65]},
    )
    matrix_data = data["gradient_cosine_matrix"]
    matrix_families = matrix_data["families"]
    matrix = np.asarray(
        [
            [matrix_data["values"][left].get(right) for right in matrix_families]
            for left in matrix_families
        ],
        dtype=np.float64,
    )
    if matrix.size:
        masked = np.ma.masked_invalid(matrix)
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad("#F2F4F7")
        image = matrix_axis.imshow(masked, vmin=-1.0, vmax=1.0, cmap=cmap)
        matrix_axis.set_xticks(range(len(matrix_families)), matrix_families, rotation=35, ha="right")
        matrix_axis.set_yticks(range(len(matrix_families)), matrix_families)
        for row_index in range(len(matrix_families)):
            for column_index in range(len(matrix_families)):
                value = matrix[row_index, column_index]
                label = "—" if not np.isfinite(value) else f"{value:+.2f}"
                matrix_axis.text(column_index, row_index, label, ha="center", va="center", fontsize=8)
        gradient_figure.colorbar(image, ax=matrix_axis, fraction=0.046, pad=0.04, label="Gradient cosine")
        step = matrix_data["step"]
        matrix_axis.set_title(
            "Family gradient-cosine matrix"
            if step is None else f"Family gradient-cosine matrix · epoch {matrix_data['x']:g}"
            if data["x_label"] == "Epoch" else f"Family gradient-cosine matrix · step {step}"
        )
    else:
        _empty_adaptive_axis(matrix_axis, "No configured family gradients")
    observed_dot_names = {
        name for record in records for name in record["actual_dots"]
    }
    dot_names = [name for name in family_order if name in observed_dot_names]
    dot_names.extend(
        sorted(observed_dot_names.difference(dot_names))
    )
    plotted_dots = False
    for index, name in enumerate(dot_names):
        points = [
            (record["x"], record["actual_dots"].get(name)) for record in records
        ]
        points = [point for point in points if point[1] is not None]
        if points:
            dot_axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o" if len(points) <= 8 else None,
                markersize=3,
                linewidth=1.2,
                color=_adaptive_color(index),
                label=name,
            )
            plotted_dots = True
    dot_axis.axhline(0.0, color="#202733", linewidth=0.9, linestyle="--")
    dot_axis.set_title("Actual AdamW displacement dot products")
    dot_axis.set_xlabel(x_label)
    dot_axis.set_ylabel(r"Objective gradient $\cdot$ actual $\Delta\theta$")
    dot_axis.text(
        0.99,
        0.02,
        "Negative predicts local first-order decrease",
        transform=dot_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=HISTORY_MUTED_TEXT_COLOR,
    )
    if plotted_dots:
        dot_axis.legend(fontsize=8, ncol=2)
    else:
        _empty_adaptive_axis(dot_axis, "Actual optimizer-dot diagnostics were not recorded")

    return {
        "fidelity_risks": risk_figure,
        "controller_state": control_figure,
        "selector_trajectories": selector_figure,
        "gradient_geometry": gradient_figure,
    }


def _empty_adaptive_axis(axis, text: str) -> None:
    axis.text(0.5, 0.5, text, ha="center", va="center", transform=axis.transAxes)
    axis.set_xticks([])
    axis.set_yticks([])


def render_adaptive_coherence_history(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    pyplot=None,
) -> dict[str, Any] | None:
    """Render opt-in adaptive fidelity/controller/selector plots and a JSON audit."""
    run_dir = Path(run_dir)
    metrics_dir = run_dir / "metrics"
    training_rows = _read_jsonl(metrics_dir / "history.jsonl")
    validation_rows = _read_jsonl(metrics_dir / "coherence_validation.jsonl")
    before_path = run_dir / "evaluation" / "before.json"
    selected_path = run_dir / "evaluation" / "selected.json"
    after_path = run_dir / "evaluation" / "after.json"

    def read_json(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    config_path = run_dir / "resolved_config.yaml"
    config = load_config(config_path) if config_path.is_file() else {}
    before, selected, after = (
        read_json(before_path),
        read_json(selected_path),
        read_json(after_path),
    )
    has_adaptive_training = any(
        any(key.startswith("fidelity/") for key in row)
        or any(key.startswith("update/actual_dot/") for key in row)
        for row in training_rows
    )
    checkpointing = config.get("checkpointing", {})
    if not isinstance(checkpointing, Mapping):
        checkpointing = {}
    selected_has_family_scores = isinstance(
        (selected or {}).get("family_source_normalized_scores"), Mapping
    )
    has_family_selection = any(
        isinstance(row.get("family_source_normalized_scores"), Mapping)
        for row in validation_rows
    ) or selected_has_family_scores or checkpointing.get("selection_metric") == "coherence_with_fidelity"
    if not has_adaptive_training and not has_family_selection:
        return None
    data = extract_adaptive_coherence_history(
        training_rows,
        validation_rows,
        before=before,
        selected=selected,
        after=after,
        config=config,
        steps_per_epoch=(read_json(run_dir / "run_manifest.json") or {}).get("steps_per_epoch")
        if (run_dir / "run_manifest.json").is_file() else None,
    )
    if pyplot is None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            from matplotlib import pyplot
        except ImportError:
            warnings.warn(
                "matplotlib is unavailable; adaptive coherence history figures cannot be generated",
                stacklevel=2,
            )
            return None
    pyplot.rcParams["svg.fonttype"] = "none"
    destination = Path(output_dir) if output_dir is not None else run_dir / "visualization"
    destination.mkdir(parents=True, exist_ok=True)
    figures = build_adaptive_coherence_figures(data, pyplot)
    artifact_paths: dict[str, dict[str, str]] = {}
    for name, figure in figures.items():
        stem = f"adaptive_{name}"
        paths = {}
        for fmt in (("pdf",) if config.get("runtime", {}).get("plot_format") == "pdf"
                    else ("png", "pdf", "svg")):
            path = destination / f"{stem}.{fmt}"
            figure.savefig(path, dpi=180 if fmt == "png" else None, format=fmt)
            paths[fmt] = path.name
        pyplot.close(figure)
        artifact_paths[name] = paths

    summary = {
        "run_dir": str(run_dir.resolve()),
        "controller_settings": data["controller_settings"],
        "selector_settings": data["selector_settings"],
        "family_order": data["family_order"],
        "constraints": data["constraints"],
        "x_label": data["x_label"],
        "inputs": {
            "training_history": {
                "path": "metrics/history.jsonl",
                "rows": len(training_rows),
                "present": (metrics_dir / "history.jsonl").is_file(),
            },
            "coherence_validation": {
                "path": "metrics/coherence_validation.jsonl",
                "rows": len(validation_rows),
                "present": (metrics_dir / "coherence_validation.jsonl").is_file(),
            },
            "evaluation_before": {"path": "evaluation/before.json", "present": before is not None},
            "evaluation_selected": {"path": "evaluation/selected.json", "present": selected is not None},
            "evaluation_after": {"path": "evaluation/after.json", "present": after is not None},
        },
        "artifacts": artifact_paths,
        **_adaptive_history_summaries(data),
    }
    summary_path = destination / "adaptive_coherence_history.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "directory": str(destination.resolve()),
        "summary": str(summary_path),
        "figures": {
            name: {fmt: str(destination / filename) for fmt, filename in formats.items()}
            for name, formats in artifact_paths.items()
        },
        "statistics": _adaptive_history_summaries(data),
    }


def build_coherence_history_figure(data: CoherenceHistoryData, plt, *, description: str):
    """Build a compact family-grouped figure of weighted and raw diagnostics."""
    by_family = {
        family: tuple(component for component in data.components if component.family == family)
        for family in data.family_order
    }
    by_family = {family: components for family, components in by_family.items() if components}
    family_columns = {family: min(3, len(components)) for family, components in by_family.items()}
    family_rows = {
        family: math.ceil(len(components) / family_columns[family])
        for family, components in by_family.items()
    }
    total_component_rows = sum(family_rows.values())
    all_epochs = [*data.total_epochs]
    for epochs, _ in data.family_totals.values():
        all_epochs.extend(epochs)
    for component in data.components:
        all_epochs.extend(component.epochs)
    epoch_max = max(all_epochs, default=1.0)
    figure = plt.figure(
        figsize=(12.6, 2.5 + 1.9 * total_component_rows + 0.45 * len(by_family)),
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
        f"{_stage_label(description)} coherence history",
        color=HISTORY_TEXT_COLOR,
        fontsize=14,
        fontweight="medium",
    )

    total_axis, family_axis = subfigures[0].subplots(
        1, 2, gridspec_kw={"width_ratios": [1.0, 1.55], "wspace": 0.22}
    )
    if data.total_values:
        total_axis.plot(
            data.total_epochs,
            data.total_values,
            color=_SUMMARY_COLOR,
            linewidth=2.2,
            label="Total coherence",
        )
    total_axis.set_title(
        "Total coherence", loc="left", color=HISTORY_TEXT_COLOR,
        fontsize=11.5, fontweight="medium", pad=8,
    )
    total_axis.set_xlabel("Training epoch")
    total_axis.set_ylabel("Weighted objective")
    style_history_axis(total_axis, data.total_values, x_max=epoch_max)

    family_values: list[float] = []
    if data.family_totals:
        for index, family in enumerate(data.family_order):
            if family not in data.family_totals:
                continue
            epochs, values = data.family_totals[family]
            family_values.extend(values)
            color, linestyle = history_family_style(family, index)
            family_axis.plot(
                epochs,
                values,
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                label=_display_name(family),
            )
    family_axis.set_title(
        "Weighted family contributions",
        loc="left",
        color=HISTORY_TEXT_COLOR,
        fontsize=11.5,
        fontweight="medium",
        pad=8,
    )
    family_axis.set_xlabel("Training epoch")
    family_axis.set_ylabel("Weighted objective")
    style_history_axis(family_axis, family_values, x_max=epoch_max)
    if family_axis.lines:
        family_axis.legend(
            loc="upper right",
            frameon=True,
            facecolor="white",
            edgecolor="#D4D9E2",
            framealpha=0.96,
            fontsize=8.2,
            ncol=min(3, len(family_axis.lines)),
            handlelength=2.4,
            columnspacing=1.2,
        )

    for family_index, (family, components) in enumerate(by_family.items()):
        subfigure = subfigures[1 + family_index]
        subfigure.suptitle(
            _display_name(family),
            x=0.01,
            ha="left",
            color=history_family_style(family, family_index)[0],
            fontsize=11.5,
            fontweight="medium",
        )
        columns = family_columns[family]
        grid = subfigure.add_gridspec(family_rows[family], columns)
        axes = []
        for component_index in range(len(components)):
            row = component_index // columns
            column = component_index % columns
            cell = (
                grid[row, :]
                if component_index == len(components) - 1
                and len(components) % columns == 1
                else grid[row, column]
            )
            axes.append(subfigure.add_subplot(cell))
        color, _ = history_family_style(family, family_index)
        for axis, component in zip(axes, components):
            marker = "o" if len(component.epochs) <= 12 else None
            axis.plot(
                component.epochs,
                component.raw,
                color=color,
                linewidth=0.8,
                alpha=0.42 if marker is None else 0.85,
                marker=marker,
                markersize=3.2 if marker else None,
            )
            trend = _rolling_median(component.raw)
            if trend is not component.raw:
                axis.plot(
                    component.epochs,
                    trend,
                    color=color,
                    linewidth=1.55,
                    alpha=0.96,
                    zorder=3,
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
            axis.set_title(
                _component_label(component.component),
                loc="left",
                color=HISTORY_TEXT_COLOR,
                fontsize=9.4,
                fontweight="medium",
                pad=6,
            )
            axis.text(
                1.0,
                0.97,
                f"raw {component.raw[-1]:.2e} · weighted {component.weighted[-1]:.2e}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                color=HISTORY_MUTED_TEXT_COLOR,
                fontsize=7.3,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
            )
            axis.set_xlabel("Epoch", labelpad=2)
            axis.set_ylabel("Raw loss", labelpad=2)
            style_history_axis(axis, component.raw, x_max=epoch_max)
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
    figure.savefig(temporary, dpi=180, format=destination.suffix.lstrip("."))
    pyplot.close(figure)
    os.replace(temporary, destination)
    return destination
