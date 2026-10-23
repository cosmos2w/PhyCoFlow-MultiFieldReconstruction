"""Focused tests for pooled mature-window quality assessment."""

from __future__ import annotations

import pytest

from phycoflow_reconstruction.evaluation.quality_history import (
    BGroupMeasurement,
    QualityCheck,
    QualityContract,
    QualityHistoryError,
    RawMeasurement,
    assess_quality_history,
    pool_measurements,
    quality_check_from_mapping,
)

FIELDS = ("CH4", "CO", "T", "U_1", "p")
CONTRACT = QualityContract(field_names=FIELDS)


def raw_ratio(ratio: float, *, count: float = 1.0, source_count: float = 1.0) -> RawMeasurement:
    return RawMeasurement(
        numerator=ratio * count,
        count=count,
        source_numerator=source_count,
        source_count=source_count,
    )


def make_check(
    epoch: int,
    *,
    coherence: float = 0.85,
    a: float = 0.99,
    t: float = 0.98,
    b: float = 0.97,
    endpoint: float = 0.99,
    fields: dict[str, float] | None = None,
    completion: float | None = None,
    b_group_size: int = 32,
) -> QualityCheck:
    completion = float(epoch * 10 if completion is None else completion)
    values = fields or {name: 1.0 for name in FIELDS}
    return QualityCheck(
        epoch=epoch,
        completion_seconds=completion,
        coherence_sum=raw_ratio(coherence, count=128, source_count=128),
        families={"A": raw_ratio(a, count=128, source_count=128), "T": raw_ratio(t, count=128, source_count=128)},
        b_groups=tuple(
            BGroupMeasurement(
                group_id=f"g{index}",
                snapshot_count=b_group_size,
                measurement=raw_ratio(b),
            )
            for index in range(4)
        ),
        endpoint_aggregate=raw_ratio(endpoint, count=128, source_count=128),
        endpoint_fields={
            name: raw_ratio(value, count=128, source_count=128)
            for name, value in values.items()
        },
    )


def full_history(**overrides: dict[int, dict[str, float]]) -> list[QualityCheck]:
    adjustments = overrides.get("adjustments", {})
    return [make_check(epoch, **adjustments.get(epoch, {})) for epoch in range(0, 101, 10)]


def assess(checks: list[QualityCheck], *, end: int = 100) -> dict:
    return assess_quality_history(
        checks,
        actual_end_epoch=end,
        end_completion_seconds=float(end * 10 + 1),
        contract=CONTRACT,
    )


def test_pooling_uses_raw_numerators_and_counts_before_ratio():
    measurements = [
        RawMeasurement(90, 100, 100, 100),
        RawMeasurement(30, 10, 20, 10),
    ]
    pooled = pool_measurements(measurements)
    assert pooled.ratio == pytest.approx((120 / 110) / (120 / 110))
    assert pooled.ratio != pytest.approx((0.9 + 1.5) / 2)


def test_confirms_on_second_mature_window_completion_and_reports_retention():
    result = assess(full_history())
    assert result["first_raw_crossing"]["epoch"] == 10
    assert result["confirmation"] == {
        "status": "confirmed",
        "first_window_epoch": 60,
        "confirmation_epoch": 70,
        "completion_seconds": 700.0,
        "completion_basis": "second passing window check completion",
    }
    assert result["retention"]["retained_through_epoch_100"] is True
    assert result["retention"]["window_target_fraction"] == 1.0
    assert [window["epoch"] for window in result["mature_windows"]] == [60, 70, 80, 90, 100]


def test_family_b_uses_exact_b32_scores_and_pooled_raw_ratio():
    checks = full_history()
    check = checks[-1]
    groups = tuple(
        BGroupMeasurement(
            group_id=f"g{index}", snapshot_count=32, measurement=raw_ratio(value)
        )
        for index, value in enumerate((0.50, 1.50, 1.50, 1.50))
    )
    checks[-1] = QualityCheck(
        epoch=check.epoch,
        completion_seconds=check.completion_seconds,
        coherence_sum=check.coherence_sum,
        families=check.families,
        b_groups=groups,
        endpoint_aggregate=check.endpoint_aggregate,
        endpoint_fields=check.endpoint_fields,
    )
    result = assess(checks)
    epoch100 = result["mature_windows"][-1]
    assert epoch100["metrics"]["family_B"]["pooled_ratio"] == pytest.approx(1.04)
    assert epoch100["metrics"]["family_B"]["raw_check_violations"] == 1
    assert epoch100["meets_target"] is True


def test_later_loss_is_reported_with_magnitude_and_duration():
    checks = full_history(
        adjustments={80: {"t": 1.4}, 90: {"t": 1.4}, 100: {"t": 1.4}}
    )
    result = assess(checks)
    assert result["confirmation"]["confirmation_epoch"] == 70
    retention = result["retention"]
    assert retention["lost_after_confirmation"] is True
    assert retention["retained_through_epoch_100"] is False
    assert retention["raw_check_target_fraction"] == pytest.approx(0.0)
    assert retention["excursions"] == [
        {
            "start_window_epoch": 80,
            "end_window_epoch": 100,
            "window_count": 3,
            "duration_epochs": 30,
            "max_excess_by_metric": {"family_T": pytest.approx(0.245)},
        }
    ]


def test_missing_scheduled_check_makes_window_incomplete_and_censors():
    checks = [make_check(epoch) for epoch in range(0, 101, 10) if epoch != 70]
    result = assess(checks)
    assert result["mature_windows"][1]["epoch"] == 70
    assert result["mature_windows"][1]["complete"] is False
    assert result["mature_windows"][1]["missing_check_epochs"] == [70]
    assert result["confirmation"]["status"] == "right_censored"
    assert result["confirmation"]["censor_epoch"] == 100


def test_missing_post_confirmation_checks_censor_retention_without_counting_as_loss():
    checks = [make_check(epoch) for epoch in range(0, 101, 10) if epoch != 90]
    result = assess(checks)
    retention = result["retention"]
    assert result["confirmation"]["confirmation_epoch"] == 70
    assert retention["status"] == "right_censored"
    assert retention["censor_epoch"] == 90
    assert retention["lost_after_confirmation"] is False
    assert retention["retained_through_observed_end"] is False
    assert retention["censored_spans"] == [
        {
            "start_window_epoch": 90,
            "end_window_epoch": 100,
            "window_epochs": [90, 100],
            "missing_check_epochs": [90],
        }
    ]


def test_no_target_hit_is_right_censored_at_actual_end():
    checks = [make_check(epoch, coherence=0.94) for epoch in range(0, 81, 10)]
    result = assess(checks, end=80)
    assert result["confirmation"]["status"] == "right_censored"
    assert result["confirmation"]["censor_epoch"] == 80
    assert result["confirmation"]["completion_seconds"] == 801.0
    assert result["retention"] is None


def test_b_group_size_and_test_split_are_rejected():
    with pytest.raises(QualityHistoryError, match="exactly 32"):
        make_check(60, b_group_size=128)
    valid = make_check(60)
    with pytest.raises(QualityHistoryError, match="validation checks only"):
        QualityCheck(
            epoch=valid.epoch,
            completion_seconds=valid.completion_seconds,
            coherence_sum=valid.coherence_sum,
            families=valid.families,
            b_groups=valid.b_groups,
            endpoint_aggregate=valid.endpoint_aggregate,
            endpoint_fields=valid.endpoint_fields,
            split="test",
        )


def test_actual_end_epoch_must_be_an_integer():
    with pytest.raises(QualityHistoryError, match="within the frozen horizon"):
        assess_quality_history(
            full_history(),
            actual_end_epoch=100.0,
            end_completion_seconds=1001.0,
            contract=CONTRACT,
        )


def test_confirmation_at_horizon_does_not_claim_post_confirmation_retention():
    contract = QualityContract(field_names=FIELDS, maximum_epochs=70)
    result = assess_quality_history(
        [make_check(epoch) for epoch in range(0, 71, 10)],
        actual_end_epoch=70,
        end_completion_seconds=701.0,
        contract=contract,
    )
    assert result["confirmation"]["status"] == "confirmed"
    assert result["retention"]["status"] == "right_censored"
    assert result["retention"]["censor_epoch"] == 70
    assert result["retention"]["window_count"] == 0
    assert result["retention"]["retained_through_epoch_100"] is False


def test_json_history_requires_explicit_validation_split_and_exact_b32_sizes():
    measurement = {
        "numerator": 1.0,
        "count": 1.0,
        "source_numerator": 1.0,
        "source_count": 1.0,
    }
    payload = {
        "epoch": 60,
        "completion_seconds": 60.0,
        "split": "validation",
        "raw": {
            "coherence_sum": measurement,
            "families": {"A": measurement, "T": measurement},
            "b_groups": [
                {"group_id": f"g{index}", "snapshot_count": 32, "measurement": measurement}
                for index in range(4)
            ],
            "endpoint_aggregate": measurement,
            "endpoint_fields": {name: measurement for name in FIELDS},
        },
    }
    parsed = quality_check_from_mapping(payload)
    assert parsed.split == "validation"
    with pytest.raises(QualityHistoryError, match="explicitly identify"):
        quality_check_from_mapping({key: value for key, value in payload.items() if key != "split"})
    malformed = {**payload, "raw": {**payload["raw"], "b_groups": [
        {"group_id": f"g{index}", "snapshot_count": 32.5, "measurement": measurement}
        for index in range(4)
    ]}}
    with pytest.raises(QualityHistoryError, match="must be an integer"):
        quality_check_from_mapping(malformed)
