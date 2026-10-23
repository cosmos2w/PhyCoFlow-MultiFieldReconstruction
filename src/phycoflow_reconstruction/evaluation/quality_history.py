"""Prospective mature-window assessment for fixed full-descriptor checks.

The input is already-computed validation evidence. This module never opens a
dataset or invokes a validator. Ratios are formed from pooled raw numerators
and their matched SOURCE numerators; B is pooled only across exact groups of 32.
"""

from __future__ import annotations

import itertools
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class QualityHistoryError(ValueError):
    """Raised when history evidence cannot support the declared assessment."""


@dataclass(frozen=True)
class RawMeasurement:
    """Unnormalized metric and matching SOURCE metric with their raw counts."""

    numerator: float
    count: float
    source_numerator: float
    source_count: float

    def __post_init__(self) -> None:
        values = (self.numerator, self.count, self.source_numerator, self.source_count)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            raise QualityHistoryError("raw measurements must be finite")
        if self.numerator < 0 or self.source_numerator <= 0:
            raise QualityHistoryError("metric numerators must be nonnegative and SOURCE positive")
        if self.count <= 0 or self.source_count <= 0:
            raise QualityHistoryError("raw metric counts must be positive")

    @property
    def ratio(self) -> float:
        return (self.numerator / self.count) / (self.source_numerator / self.source_count)

    def to_dict(self) -> dict[str, float]:
        return {
            "numerator": float(self.numerator),
            "count": float(self.count),
            "source_numerator": float(self.source_numerator),
            "source_count": float(self.source_count),
            "ratio": self.ratio,
        }


def pool_measurements(measurements: Sequence[RawMeasurement]) -> RawMeasurement:
    """Pool sums and counts before dividing, preserving the SOURCE weighting."""
    if not measurements:
        raise QualityHistoryError("cannot pool an empty set of measurements")
    return RawMeasurement(
        numerator=sum(item.numerator for item in measurements),
        count=sum(item.count for item in measurements),
        source_numerator=sum(item.source_numerator for item in measurements),
        source_count=sum(item.source_count for item in measurements),
    )


@dataclass(frozen=True)
class BGroupMeasurement:
    """One independently computed B(32) score; groups are never merged first."""

    group_id: str
    snapshot_count: int
    measurement: RawMeasurement

    def __post_init__(self) -> None:
        if not self.group_id:
            raise QualityHistoryError("B group_id must be nonempty")
        if self.snapshot_count != 32:
            raise QualityHistoryError(
                f"B group {self.group_id!r} has {self.snapshot_count} snapshots; expected exactly 32"
            )


@dataclass(frozen=True)
class QualityCheck:
    """One full validation check ending at an accepted reporting epoch."""

    epoch: int
    completion_seconds: float
    coherence_sum: RawMeasurement
    families: Mapping[str, RawMeasurement]
    b_groups: tuple[BGroupMeasurement, ...]
    endpoint_aggregate: RawMeasurement
    endpoint_fields: Mapping[str, RawMeasurement]
    split: str = "validation"

    def __post_init__(self) -> None:
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int) or self.epoch < 0:
            raise QualityHistoryError("check epoch must be a nonnegative integer")
        if (
            isinstance(self.completion_seconds, bool)
            or not isinstance(self.completion_seconds, (int, float))
            or not math.isfinite(float(self.completion_seconds))
            or self.completion_seconds < 0
        ):
            raise QualityHistoryError("check completion_seconds must be finite and nonnegative")
        if self.split != "validation":
            raise QualityHistoryError("quality histories accept validation checks only; TEST is forbidden")
        if set(self.families) != {"A", "T"}:
            raise QualityHistoryError("families must contain raw A and T measurements; B comes from B(32) groups")
        if not self.endpoint_fields:
            raise QualityHistoryError("at least one per-field endpoint measurement is required")
        if len(self.b_groups) != 4:
            raise QualityHistoryError("each check must contain exactly four B(32) groups")
        group_ids = [group.group_id for group in self.b_groups]
        if len(set(group_ids)) != len(group_ids):
            raise QualityHistoryError("B group IDs must be unique within a check")


@dataclass(frozen=True)
class QualityContract:
    """Frozen quality thresholds and reporting cadence for one task."""

    field_names: tuple[str, ...]
    coherence_sum_limit: float = 0.90
    family_limits: Mapping[str, float] = field(
        default_factory=lambda: {"A": 1.05, "B": 1.05, "T": 1.05}
    )
    endpoint_aggregate_limit: float = 1.05
    endpoint_field_limit: float = 1.05
    mature_start_epoch: int = 60
    window_epochs: int = 40
    check_interval_epochs: int = 10
    maximum_epochs: int = 100

    def __post_init__(self) -> None:
        if not self.field_names or len(set(self.field_names)) != len(self.field_names):
            raise QualityHistoryError("field_names must be nonempty and unique")
        if set(self.family_limits) != {"A", "B", "T"}:
            raise QualityHistoryError("family_limits must declare A, B, and T")
        if self.window_epochs <= 0 or self.check_interval_epochs <= 0:
            raise QualityHistoryError("window and check interval must be positive")
        if self.window_epochs % self.check_interval_epochs:
            raise QualityHistoryError("window length must be divisible by the check interval")
        if self.mature_start_epoch < self.window_epochs + self.check_interval_epochs:
            raise QualityHistoryError("mature_start_epoch must allow a complete scheduled window")
        if self.maximum_epochs < self.mature_start_epoch:
            raise QualityHistoryError("maximum_epochs must reach the mature assessment period")
        thresholds = [
            self.coherence_sum_limit,
            self.endpoint_aggregate_limit,
            self.endpoint_field_limit,
            *self.family_limits.values(),
        ]
        if any(not math.isfinite(float(limit)) or limit <= 0 for limit in thresholds):
            raise QualityHistoryError("quality limits must be finite and positive")

    def limits(self) -> dict[str, float]:
        result = {
            "coherence_sum": float(self.coherence_sum_limit),
            "family_A": float(self.family_limits["A"]),
            "family_B": float(self.family_limits["B"]),
            "family_T": float(self.family_limits["T"]),
            "endpoint_aggregate": float(self.endpoint_aggregate_limit),
        }
        result.update(
            {f"endpoint_field:{name}": float(self.endpoint_field_limit) for name in self.field_names}
        )
        return result


def quality_check_from_mapping(payload: Mapping[str, Any]) -> QualityCheck:
    """Parse the compact JSON representation used by campaign artifacts."""

    def measurement(value: Mapping[str, Any]) -> RawMeasurement:
        return RawMeasurement(
            numerator=float(value["numerator"]),
            count=float(value["count"]),
            source_numerator=float(value["source_numerator"]),
            source_count=float(value["source_count"]),
        )

    raw = payload.get("raw", payload)
    family_payload = raw.get("families", {})
    epoch = payload.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise QualityHistoryError("quality-history epoch must be an integer")
    completion_seconds = payload.get("completion_seconds")
    if isinstance(completion_seconds, bool) or not isinstance(completion_seconds, (int, float)):
        raise QualityHistoryError("quality-history completion_seconds must be numeric")
    if "split" not in payload:
        raise QualityHistoryError("quality-history check must explicitly identify its validation split")
    groups = raw.get("b_groups")
    if not isinstance(groups, list):
        raise QualityHistoryError("raw B groups must be an array")
    for group in groups:
        snapshot_count = group.get("snapshot_count") if isinstance(group, Mapping) else None
        if isinstance(snapshot_count, bool) or not isinstance(snapshot_count, int):
            raise QualityHistoryError("B group snapshot_count must be an integer")
    return QualityCheck(
        epoch=epoch,
        completion_seconds=float(payload["completion_seconds"]),
        coherence_sum=measurement(raw["coherence_sum"]),
        families={name: measurement(value) for name, value in family_payload.items()},
        b_groups=tuple(
            BGroupMeasurement(
                group_id=str(group["group_id"]),
                snapshot_count=int(group["snapshot_count"]),
                measurement=measurement(group["measurement"]),
            )
            for group in groups
        ),
        endpoint_aggregate=measurement(raw["endpoint_aggregate"]),
        endpoint_fields={
            name: measurement(value) for name, value in raw["endpoint_fields"].items()
        },
        split=str(payload["split"]),
    )


def load_quality_checks(path: str | Path) -> list[QualityCheck]:
    """Load a JSON array of precomputed checks; no dataset access is performed."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QualityHistoryError(f"cannot read quality-history JSON: {error}") from error
    checks_payload = payload.get("checks") if isinstance(payload, dict) else payload
    if not isinstance(checks_payload, list):
        raise QualityHistoryError("quality-history JSON must contain a checks array")
    return [quality_check_from_mapping(item) for item in checks_payload]


def _check_measurements(check: QualityCheck) -> dict[str, RawMeasurement]:
    b_measurement = pool_measurements([group.measurement for group in check.b_groups])
    result = {
        "coherence_sum": check.coherence_sum,
        "family_A": check.families["A"],
        "family_B": b_measurement,
        "family_T": check.families["T"],
        "endpoint_aggregate": check.endpoint_aggregate,
    }
    result.update(
        {f"endpoint_field:{name}": check.endpoint_fields[name] for name in check.endpoint_fields}
    )
    return result


def _least_squares_slope(points: Sequence[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    x_mean = sum(point[0] for point in points) / len(points)
    y_mean = sum(point[1] for point in points) / len(points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    if denominator == 0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator


def _window_summary(
    endpoint: int,
    checks: Mapping[int, QualityCheck],
    contract: QualityContract,
) -> dict[str, Any]:
    interval = contract.check_interval_epochs
    expected_epochs = tuple(
        range(endpoint - contract.window_epochs + interval, endpoint + 1, interval)
    )
    missing = [epoch for epoch in expected_epochs if epoch not in checks]
    observed = [checks[epoch] for epoch in expected_epochs if epoch in checks]
    check_values = {check.epoch: _check_measurements(check) for check in observed}
    metric_names = tuple(contract.limits())
    metrics: dict[str, Any] = {}

    for name in metric_names:
        raw_items: list[RawMeasurement] = []
        points: list[tuple[float, float]] = []
        violations = 0
        limit = contract.limits()[name]
        for check in observed:
            item = check_values[check.epoch][name]
            raw_items.append(item)
            points.append((float(check.epoch), item.ratio))
            violations += item.ratio > limit
        pooled = pool_measurements(raw_items) if raw_items else None
        ratios = [value for _, value in points]
        metrics[name] = {
            "pooled_raw": pooled.to_dict() if pooled is not None else None,
            "pooled_ratio": pooled.ratio if pooled is not None else None,
            "mean_check_ratio": statistics.fmean(ratios) if ratios else None,
            "median_check_ratio": statistics.median(ratios) if ratios else None,
            "slope_per_epoch": _least_squares_slope(points),
            "range": [min(ratios), max(ratios)] if ratios else None,
            "raw_check_violations": violations,
            "limit": limit,
        }

    complete = not missing
    meets = complete and all(
        metrics[name]["pooled_ratio"] is not None
        and metrics[name]["pooled_ratio"] <= limit
        for name, limit in contract.limits().items()
    )
    check_target: dict[int, bool] = {}
    for check in observed:
        values = check_values[check.epoch]
        check_target[check.epoch] = all(
            values[name].ratio <= limit for name, limit in contract.limits().items()
        )
    return {
        "epoch": endpoint,
        "window_open_epoch": endpoint - contract.window_epochs,
        "window_interval": "(e-40,e]",
        "expected_check_epochs": list(expected_epochs),
        "observed_check_epochs": [check.epoch for check in observed],
        "missing_check_epochs": missing,
        "complete": complete,
        "metrics": metrics,
        "raw_check_target": {str(epoch): value for epoch, value in check_target.items()},
        "raw_target_checks": sum(check_target.values()),
        "raw_check_count": len(check_target),
        "meets_target": bool(meets),
    }


def _excursion_windows(
    windows: Sequence[Mapping[str, Any]], check_interval_epochs: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize complete failed windows and report incomplete censoring spans."""
    excursions: list[dict[str, Any]] = []
    censored: list[dict[str, Any]] = []
    active: list[Mapping[str, Any]] = []
    active_censored: list[Mapping[str, Any]] = []

    def flush() -> None:
        if not active:
            return
        excesses: dict[str, float] = {}
        for window in active:
            for name, metric in window["metrics"].items():
                ratio = metric["pooled_ratio"]
                if ratio is not None and ratio > metric["limit"]:
                    excesses[name] = max(excesses.get(name, 0.0), ratio - metric["limit"])
        excursions.append(
            {
                "start_window_epoch": int(active[0]["epoch"]),
                "end_window_epoch": int(active[-1]["epoch"]),
                "window_count": len(active),
                "duration_epochs": len(active) * check_interval_epochs,
                "max_excess_by_metric": excesses,
            }
        )
        active.clear()

    def flush_censored() -> None:
        if not active_censored:
            return
        censored.append(
            {
                "start_window_epoch": int(active_censored[0]["epoch"]),
                "end_window_epoch": int(active_censored[-1]["epoch"]),
                "window_epochs": [int(window["epoch"]) for window in active_censored],
                "missing_check_epochs": sorted(
                    {
                        int(epoch)
                        for window in active_censored
                        for epoch in window["missing_check_epochs"]
                    }
                ),
            }
        )
        active_censored.clear()

    for window in windows:
        if not window["complete"]:
            flush()
            active_censored.append(window)
        elif window["meets_target"]:
            flush()
            flush_censored()
        else:
            flush_censored()
            active.append(window)
    flush()
    flush_censored()
    return excursions, censored


def assess_quality_history(
    checks: Sequence[QualityCheck],
    *,
    actual_end_epoch: int,
    end_completion_seconds: float,
    contract: QualityContract,
) -> dict[str, Any]:
    """Assess mature windows, confirmation, retention and right-censoring.

    The reported attainment time is the completion time of the second passing
    window check. A missing scheduled check makes its window incomplete and
    cannot count as target attainment.
    """
    if (
        isinstance(actual_end_epoch, bool)
        or not isinstance(actual_end_epoch, int)
        or not (0 <= actual_end_epoch <= contract.maximum_epochs)
    ):
        raise QualityHistoryError("actual_end_epoch must be within the frozen horizon")
    if not math.isfinite(float(end_completion_seconds)) or end_completion_seconds < 0:
        raise QualityHistoryError("end_completion_seconds must be finite and nonnegative")
    if not checks:
        raise QualityHistoryError("at least one full validation check is required")
    ordered = sorted(checks, key=lambda item: item.epoch)
    epochs = [item.epoch for item in ordered]
    if len(epochs) != len(set(epochs)):
        raise QualityHistoryError("duplicate check epochs are not allowed")
    if epochs[-1] > actual_end_epoch:
        raise QualityHistoryError("a validation check lies beyond the actual end epoch")
    if epochs[-1] % contract.check_interval_epochs:
        raise QualityHistoryError("validation checks must follow the frozen reporting cadence")
    if any(epoch % contract.check_interval_epochs for epoch in epochs):
        raise QualityHistoryError("validation checks must follow the frozen reporting cadence")
    if any(
        left.completion_seconds > right.completion_seconds
        for left, right in itertools.pairwise(ordered)
    ):
        raise QualityHistoryError("check completion times must be monotone with epoch")
    if ordered[-1].completion_seconds > end_completion_seconds:
        raise QualityHistoryError("end completion time precedes the final supplied validation check")
    for check in ordered:
        if set(check.endpoint_fields) != set(contract.field_names):
            raise QualityHistoryError(
                f"epoch {check.epoch} endpoint fields differ from the frozen field contract"
            )
    group_ids = {tuple(group.group_id for group in check.b_groups) for check in ordered}
    if len(group_ids) != 1:
        raise QualityHistoryError("B(32) group identities must remain fixed across checks")

    by_epoch = {check.epoch: check for check in ordered}
    raw_checks: list[dict[str, Any]] = []
    limits = contract.limits()
    for check in ordered:
        values = _check_measurements(check)
        ratios = {name: item.ratio for name, item in values.items()}
        violations = {
            name: ratios[name] > limit for name, limit in limits.items()
        }
        raw_checks.append(
            {
                "epoch": check.epoch,
                "completion_seconds": check.completion_seconds,
                "metrics": {name: item.to_dict() for name, item in values.items()},
                "violations": violations,
                "meets_target": not any(violations.values()),
            }
        )

    mature_endpoints = range(
        contract.mature_start_epoch,
        actual_end_epoch + 1,
        contract.check_interval_epochs,
    )
    windows = [_window_summary(endpoint, by_epoch, contract) for endpoint in mature_endpoints]
    passing_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for first, second in itertools.pairwise(windows):
        if (
            second["epoch"] == first["epoch"] + contract.check_interval_epochs
            and first["meets_target"]
            and second["meets_target"]
        ):
            passing_pairs.append((first, second))

    first_raw = next((row for row in raw_checks if row["epoch"] > 0 and row["meets_target"]), None)
    confirmation: dict[str, Any] | None = None
    retention: dict[str, Any] | None = None
    if passing_pairs:
        first_window, confirmation_window = passing_pairs[0]
        confirmation_check = by_epoch[confirmation_window["epoch"]]
        confirmation = {
            "status": "confirmed",
            "first_window_epoch": first_window["epoch"],
            "confirmation_epoch": confirmation_window["epoch"],
            "completion_seconds": confirmation_check.completion_seconds,
            "completion_basis": "second passing window check completion",
        }
        post_confirmation = [
            item for item in windows if item["epoch"] > confirmation_window["epoch"]
        ]
        later_checks = [row for row in raw_checks if row["epoch"] > confirmation_window["epoch"]]
        failed_later = [
            item for item in post_confirmation if item["complete"] and not item["meets_target"]
        ]
        incomplete_later = [item for item in post_confirmation if not item["complete"]]
        excursions, censored_spans = _excursion_windows(
            post_confirmation, contract.check_interval_epochs
        )
        if failed_later:
            retention_status = "lost"
            censor_epoch = None
        elif incomplete_later:
            retention_status = "right_censored"
            censor_epoch = incomplete_later[0]["epoch"]
        elif not post_confirmation or actual_end_epoch < contract.maximum_epochs:
            retention_status = "right_censored"
            censor_epoch = actual_end_epoch
        else:
            retention_status = "retained"
            censor_epoch = None
        retention = {
            "status": retention_status,
            "censor_epoch": censor_epoch,
            "through_observed_end_epoch": actual_end_epoch,
            "full_horizon_observed": actual_end_epoch >= contract.maximum_epochs,
            "window_epochs": [item["epoch"] for item in post_confirmation],
            "window_target_count": sum(item["meets_target"] for item in post_confirmation),
            "window_count": len(post_confirmation),
            "window_target_fraction": (
                sum(item["meets_target"] for item in post_confirmation) / len(post_confirmation)
                if post_confirmation
                else None
            ),
            "raw_check_target_count": sum(row["meets_target"] for row in later_checks),
            "raw_check_count": len(later_checks),
            "raw_check_target_fraction": (
                sum(row["meets_target"] for row in later_checks) / len(later_checks)
                if later_checks
                else None
            ),
            "lost_after_confirmation": bool(failed_later),
            "retained_through_observed_end": retention_status == "retained",
            "retained_through_epoch_100": (
                retention_status == "retained" and actual_end_epoch >= contract.maximum_epochs
            ),
            "final_window": windows[-1] if windows else None,
            "excursions": excursions,
            "censored_spans": censored_spans,
        }
    else:
        confirmation = {
            "status": "right_censored",
            "censor_epoch": actual_end_epoch,
            "completion_seconds": float(end_completion_seconds),
            "reason": "no pair of consecutive complete mature windows met the frozen target",
        }

    return {
        "schema_version": "quality_history.mature_window.v1",
        "contract": {
            **asdict(contract),
            "limits": limits,
            "window_convention": "(e-40,e]",
            "b_group_size": 32,
            "b_groups_per_check": 4,
        },
        "actual_end_epoch": actual_end_epoch,
        "end_completion_seconds": float(end_completion_seconds),
        "first_raw_crossing": (
            {
                "epoch": first_raw["epoch"],
                "completion_seconds": first_raw["completion_seconds"],
            }
            if first_raw
            else None
        ),
        "confirmation": confirmation,
        "retention": retention,
        "raw_checks": raw_checks,
        "mature_windows": windows,
        "censored": confirmation["status"] == "right_censored",
    }


def write_quality_assessment(path: str | Path, assessment: Mapping[str, Any]) -> None:
    """Write a compact, deterministic JSON assessment of supplied metrics."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(assessment), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
