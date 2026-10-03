"""R2 epoch accounting extends the historical ledger without resetting age."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "pilot_r2_epochs", Path(__file__).resolve().parents[1] / "scripts/training/run_upgrade_1002_pilot.py"
)
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)


def test_resumed_epoch_arithmetic_uses_child_age():
    assert pilot.epoch_segment_limit(
        until_epoch=100, additional_epochs=None, start_step=38 * 75, configured_epochs=200
    ) == 38 * 25
    assert pilot.epoch_segment_limit(
        until_epoch=None, additional_epochs=25, start_step=38 * 100, configured_epochs=200
    ) == 38 * 25


@pytest.mark.parametrize("target,start,configured,match", [
    (250, 0, 249, "strictly below"),
    (100, 38 * 100, 200, "already been reached"),
    (201, 38 * 150, 200, "configured horizon"),
    (0, 0, 200, "positive integers"),
])
def test_epoch_segments_preserve_age_and_frozen_horizon(target, start, configured, match):
    with pytest.raises(pilot.PilotContractError, match=match):
        pilot.epoch_segment_limit(
            until_epoch=target, additional_epochs=None, start_step=start,
            configured_epochs=configured,
        )


def test_r2_output_stage_remains_nested_and_formal_is_rejected():
    assert pilot._experiment_parts("Test_1002/R2_20_native_config")[1] == "R2_20"
    with pytest.raises(pilot.PilotContractError, match="formal"):
        pilot._experiment_parts("Test_1002/R2_20_formal_5000ep")


def test_r2_scientific_horizon_is_distinct_from_software_smoke():
    config = {
        "stage": "post_training", "case": "turbulent_combustion",
        "output": {"experiment_name": "Test_1002/R2_20_native"},
        "optimization": {"epochs": 20, "batch_size": 32, "train_fraction": .15},
        "runtime": {"device": "cuda:0"},
    }
    planned = {"maximum_initial_epochs": {"R2_20": 200, "R2_90": 10},
               "every_lineage_epoch_cap": 249, "non_scientific_stages": ["R2_90"]}
    with pytest.raises(pilot.PilotContractError, match="at least 100"):
        pilot.validate_test_envelope(config, planned_runs=planned, train_count=8000)
    config["output"]["experiment_name"] = "Test_1002/R2_90_smoke"
    config["optimization"]["epochs"] = 2
    assert pilot.validate_test_envelope(config, planned_runs=planned, train_count=8000)[
        "configured_epochs"
    ] == 2


def test_interrupted_partial_epoch_stream_is_charged_without_summary_duplication(tmp_path):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (metrics / "history.jsonl").write_text(json.dumps({"step": 76, "batches": 38}) + "\n")
    (metrics / "coherence_updates.jsonl").write_text("".join(
        json.dumps({"step": step, "update_accepted": True}) + "\n"
        for step in range(39, 82)
    ))
    assert pilot._pending_history_counters(tmp_path, 38, include_update_stream=True) == {
        "attempted": 43, "accepted": 43,
    }
    # The historical R1 accounting remains unchanged.
    assert pilot._pending_history_counters(tmp_path, 38) == {"attempted": 38, "accepted": 38}
