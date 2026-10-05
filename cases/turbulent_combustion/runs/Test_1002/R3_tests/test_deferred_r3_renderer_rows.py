"""R3 reports retain every saved observation without changing the legacy reader."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]
TEST_ROOT = ROOT / "cases/turbulent_combustion/runs/Test_1002"
SPEC = importlib.util.spec_from_file_location(
    "r3_deferred_rows", ROOT / "scripts/visualization/upgrade_1002_r3_campaign_report.py"
)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def test_different_metrics_at_same_epoch_remain_distinct(monkeypatch, tmp_path):
    before = {"per_field_mse_normalized": {"p": 2.0}}
    run = {"manifest": {"steps_per_epoch": 38}, "before": before, "selection": []}
    rows = [
        {
            "step": 3800,
            "metric": metric,
            "mse": mse,
            "eligible": eligible,
            "failed_fidelity_fields": failed,
            "family_source_normalized_scores": {
                "global_distribution": ratio, "cross_spectrum": 1.0, "topology": ratio
            },
            "metrics": {"per_field_mse_normalized": {"p": mse}},
            "distinct_saved_evidence": {"tag": tag},
        }
        for metric, mse, eligible, failed, ratio, tag in (
            (.9, 1.8, True, [], .9, "first"),
            (1.1, 2.2, False, ["p"], 1.1, "second"),
        )
    ]
    monkeypatch.setattr(REPORT.READER, "read_run", lambda *_: run)
    monkeypatch.setattr(REPORT.READER, "_read_jsonl", lambda *_: rows)
    loaded = REPORT.read_r3_run("fixture", tmp_path)["selection"]
    assert [row["epoch"] for row in loaded] == [100.0, 100.0]
    assert [row["metric"] for row in loaded] == [.9, 1.1]
    assert [row["mse"] for row in loaded] == [1.8, 2.2]
    assert [row["eligible"] for row in loaded] == [True, False]
    assert loaded[1]["failed_fidelity_fields"] == ["p"]
    assert [row["per_field_relative_change"]["p"] for row in loaded] == pytest.approx([-.1, .1])
    assert [row["raw_record"] for row in loaded] == rows
    assert [row["observation_index"] for row in loaded] == [0, 1]
    values = REPORT.windows(loaded, lambda row: row["family_ratios"]["topology"])
    assert values["rolling50"]["observations"] == 2
    assert values["rolling50"]["mean"] == pytest.approx(1.0)
    assert values["rolling50"]["median"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("run_relative", "low", "high", "expected_epochs"),
    [
        (
            "R3_10_corridor_control/20261004T145443Z_ffdf2f0b",
            100, 150, [110, 120, 130, 140, 150, 150],
        ),
        (
            "R3_20_finite_primary/20261004T191607Z_254fbc47",
            150, 200, [160, 170, 180, 190, 200, 200],
        ),
        (
            "R3_30_confirmation/20261005T013624Z_afa006a8",
            100, 150, [110, 120, 130, 140, 150, 150],
        ),
    ],
)
def test_actual_completed_windows_preserve_all_six_saved_records(
    run_relative, low, high, expected_epochs
):
    run_path = TEST_ROOT / run_relative
    if not (run_path / "run_manifest.json").exists():
        pytest.skip("Local measured R3 acceptance artifacts are unavailable")
    manifest = json.loads((run_path / "run_manifest.json").read_text())
    divisor = manifest["steps_per_epoch"]
    raw = [
        json.loads(line)
        for line in (run_path / "metrics/coherence_validation.jsonl").read_text().splitlines()
        if line.strip()
    ]
    expected = [row for row in raw if low < row["step"] / divisor <= high]
    loaded = REPORT.read_r3_run("saved", run_path)
    actual = [row for row in loaded["selection"] if low < row["epoch"] <= high]
    assert [row["epoch"] for row in actual] == expected_epochs
    assert len(expected) == len(actual) == 6
    assert [row["raw_record"] for row in actual] == expected
    for item, row in zip(actual, expected):
        assert item["family_ratios"] == row["family_source_normalized_scores"]
        assert item["metric"] == row["metric"]
        assert item["mse"] == row["mse"]
    legacy = REPORT.READER.read_run("saved", run_path)["selection"]
    assert len([row for row in legacy if low < row["epoch"] <= high]) == 5
    assert REPORT.windows(actual, lambda row: row["family_ratios"]["topology"])["rolling50"]["observations"] == 6
