"""Saved-history summaries preserve exposure and do not invent missing timing."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/posttraining/summarize_runs.py"
SPEC = importlib.util.spec_from_file_location("posttraining_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _run(tmp_path, rows, cursor=38):
    run = tmp_path / "child"
    (run / "metrics").mkdir(parents=True)
    (run / "run_manifest.json").write_text(json.dumps({"last_checkpoint_step": cursor}))
    (run / "status.json").write_text(json.dumps({"status": "interrupted", "global_step": 99}))
    (run / "resolved_config.yaml").write_text("optimization: {epochs: 1000}\n")
    (run / "metrics/history.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    return run


def test_partial_epoch_rows_use_metric_specific_counts_and_committed_cursor(tmp_path):
    run = _run(tmp_path, [
        {"epoch": 1.0, "step": 1, "batches": 1, "update_seconds": 10.0, "data_loss": 2.0,
         "finite_counts": {"update_seconds": 1, "data_loss": 1}},
        {"epoch": 1.0, "step": 38, "batches": 37, "update_seconds": 2.0, "data_loss": 4.0,
         "finite_counts": {"update_seconds": 37, "data_loss": 2}},
        {"epoch": 2.0, "step": 39, "update_seconds": 1000.0,
         "finite_counts": {"update_seconds": 1}},
    ])
    result = MODULE.summarize_run("primary", run)
    assert result["committed_updates"] == 38
    assert result["metrics"]["update_seconds"] == {"count": 38, "mean": 84 / 38}
    assert result["metrics"]["data_loss"] == {"count": 3, "mean": 10 / 3}
    assert result["timed_training_seconds"] == 84
    assert "ignored 1 history rows" in result["warnings"][0]
    assert {row["epoch"] for row in result["epochs"]} == {1}


def test_missing_finite_counts_remains_unmeasured(tmp_path):
    run = _run(tmp_path, [{"epoch": 1, "step": 38, "update_seconds": 2.0}])
    result = MODULE.summarize_run("primary", run)
    assert result["metrics"]["update_seconds"] == {"count": 0, "mean": None}
    assert result["timed_training_seconds"] is None
    assert "0 of 38" in result["warnings"][0]


def test_overlapping_metric_rows_cannot_double_count_a_replayed_update(tmp_path):
    row = {"epoch": 1, "step": 38, "update_seconds": 2.0,
           "finite_counts": {"update_seconds": 38}}
    run = _run(tmp_path, [row, row])
    with pytest.raises(ValueError, match="overlapping metric coverage"):
        MODULE.summarize_run("primary", run)


def test_counted_nonfinite_metrics_fail_instead_of_dropping_evidence(tmp_path):
    run = _run(tmp_path, [{"epoch": 1, "step": 38, "update_seconds": float("nan"),
                          "finite_counts": {"update_seconds": 38}}])
    with pytest.raises(ValueError, match="nonfinite counted metric"):
        MODULE.summarize_run("primary", run)


def test_finite_counts_cannot_exceed_the_rows_observed_batches(tmp_path):
    run = _run(tmp_path, [{"epoch": 1, "step": 38, "batches": 1, "update_seconds": 2.0,
                          "finite_counts": {"update_seconds": 38}}])
    with pytest.raises(ValueError, match="finite count exceeds batches"):
        MODULE.summarize_run("primary", run)


def test_cli_preserves_inputs_and_refuses_to_overwrite_an_existing_summary(tmp_path):
    run = _run(tmp_path, [{"epoch": 1, "step": 38, "update_seconds": 2.0,
                          "finite_counts": {"update_seconds": 38}}])
    before = {str(path): path.read_bytes() for path in run.rglob("*") if path.is_file()}
    output = tmp_path / "report"
    command = [sys.executable, str(SCRIPT), "--run", f"primary={run}", "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True)
    summary = output / "summary.json"
    generated = summary.read_bytes()
    assert json.loads(generated)["runs"][0]["metrics"]["update_seconds"]["mean"] == 2.0
    assert "no checkpoint or validation assessment" in json.loads(generated)["scope"]
    assert {str(path): path.read_bytes() for path in run.rglob("*") if path.is_file()} == before
    repeat = subprocess.run(command, capture_output=True, text=True, check=False)
    assert repeat.returncode != 0 and "output already exists" in repeat.stderr
    assert summary.read_bytes() == generated
    inside = subprocess.run(
        command[:-1] + [str(run / "report")], capture_output=True, text=True, check=False,
    )
    assert inside.returncode != 0 and "outside every input run" in inside.stderr
    assert not (run / "report").exists()
