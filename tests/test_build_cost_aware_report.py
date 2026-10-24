from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.posttraining import build_cost_aware_report as report


def _task_record(arms: list[str]) -> dict:
    return {
        "role": "mandatory",
        "arms": arms,
        "horizon": 100,
        "frozen_task_config": {
            "dataset": {"field_names": ["f1", "f2", "f3", "f4", "f5"]}
        },
    }


def _inventory() -> dict:
    arms = ["dense", "heuristic", "strategy"]
    return {
        "tasks": {"G": _task_record(arms), "D": _task_record(arms)},
        "quality_contract": {"monitor_epochs": list(range(0, 101, 10))},
    }


def _complete_run(*, missing_epoch: int | None = None, assessed_end: int = 100) -> dict:
    checks = [
        {
            "epoch": epoch,
            "status": "native_primary",
            "panel_kind": "uniform",
            "measurement_source": "native_topology_history",
            "completion_seconds": 500.0 + epoch,
            "cold_process_seconds_including_profile_acquisition": 500.0 + epoch,
            "wall_seconds_from_first_training_command": 200.0 + epoch,
            "ratios": {
                "coherence_sum": 0.8,
                "family_A": 0.9,
                "family_B": 0.95,
                "family_T": 0.9,
                "endpoint_aggregate": 0.92,
                "endpoint_fields": {f"f{i}": 0.9 + i / 100 for i in range(1, 6)},
            },
        }
        for epoch in range(10, 101, 10)
        if epoch != missing_epoch
    ]
    return {
        "run_id": "synthetic-run",
        "status": "completed",
        "accepted_end_epoch": 100,
        "quality_assessment_status": "assessed",
        "native_history": {"integrity_status": "ok"},
        "uniform_evaluations": checks,
        "quality_assessment": {
            "actual_end_epoch": assessed_end,
            "contract": {"maximum_epochs": 100},
            "confirmation": {"status": "confirmed", "confirmation_epoch": 80},
            "retention": {"status": "retained", "through_observed_end_epoch": 100},
            "mature_windows": [],
        },
    }


def _history_for_run(run: dict) -> dict:
    return {
        "tasks": {
            task: {
                "arms": {
                    arm: {"runs": [run]}
                    for arm in ("dense", "heuristic", "strategy")
                }
            }
            for task in ("G", "D")
        }
    }


def test_portable_audit_table_lists_both_panels_all_fields_and_pending_arms(tmp_path: Path) -> None:
    uniform_path = tmp_path / "analysis" / "G" / "dense" / "child-001" / "portable" / "uniform.json"
    disjoint_path = tmp_path / "analysis" / "G" / "dense" / "child-001" / "portable" / "disjoint.json"
    uniform_path.parent.mkdir(parents=True)
    uniform_path.write_text("{}", encoding="utf-8")
    disjoint_path.write_text("{}", encoding="utf-8")
    uniform = {
        "path": str(uniform_path),
        "status": "valid_metrics",
        "checkpoint": "best",
        "epoch": 80,
        "panel_kind": "uniform",
        "ratios": {
            "coherence_sum": 0.88,
            "family_A": 0.95,
            "family_B": 0.96,
            "family_T": 0.97,
            "endpoint_aggregate": 1.01,
            "endpoint_fields": {"u": 1.02, "v": 1.03},
        },
    }
    disjoint = {
        "path": str(disjoint_path),
        "status": "supplementary_disjoint_panel",
        "checkpoint": "last",
        "epoch": 100,
        "panel_kind": "disjoint",
        "ratios": {
            "coherence_sum": 0.89,
            "family_A": 0.95,
            "family_B": 1.06,
            "family_T": 0.97,
            "endpoint_aggregate": 1.01,
            "endpoint_fields": {"u": 1.02, "v": 1.03},
        },
    }
    inputs = {
        "root": tmp_path,
        "inventory": {
            "tasks": {
                "G": {
                    "arms": ["dense", "heuristic"],
                    "frozen_task_config": {"dataset": {"field_names": ["u", "v"]}},
                }
            }
        },
        "history": {
            "tasks": {
                "G": {
                    "arms": {
                        "dense": {
                            "runs": [
                                {
                                    "run_id": "child-001",
                                    "portable_evaluations": [uniform],
                                    "supplementary_evaluations": [disjoint],
                                }
                            ]
                        },
                        "heuristic": {"runs": []},
                    }
                }
            }
        },
    }

    table = "\n".join(report._portable_audit_table_lines(inputs))
    assert "best / E80 | uniform (portable)" in table
    assert "last / E100 | disjoint (supplementary)" in table
    assert "full A+B+T; 128 snapshots; 4 groups of 32; 4096 points; six topology filtrations" in table
    assert "1.0600 !" in table
    assert "u 1.0200, v 1.0300" in table
    assert "[artifact](../analysis/G/dense/child-001/portable/uniform.json)" in table
    assert "G / heuristic | pending" in table


def test_mandatory_gate_requires_all_runs_and_every_scheduled_native_check() -> None:
    run = _complete_run()
    inputs = {"inventory": _inventory(), "history": _history_for_run(run)}
    complete, records = report._all_mandatory_complete(inputs)
    assert complete
    assert len(records) == 6
    assert {item["status"] for item in records} == {"complete"}

    inputs["history"] = _history_for_run(_complete_run(missing_epoch=100))
    complete, records = report._all_mandatory_complete(inputs)
    assert not complete
    assert all(item["status"] == "quality_pending" for item in records)
    assert all("100" in item["detail"] for item in records)


def test_mandatory_gate_rejects_assessor_cursor_mismatch_and_duplicate_lineages() -> None:
    inputs = {"inventory": _inventory(), "history": _history_for_run(_complete_run(assessed_end=90))}
    complete, records = report._all_mandatory_complete(inputs)
    assert not complete
    assert all(item["status"] == "quality_ambiguous" for item in records)

    run = _complete_run()
    inputs["history"] = _history_for_run(run)
    inputs["history"]["tasks"]["G"]["arms"]["dense"]["runs"].append(
        {**run, "run_id": "second-lineage"}
    )
    complete, records = report._all_mandatory_complete(inputs)
    assert not complete
    assert next(item for item in records if item["task_id"] == "G" and item["arm"] == "dense")["status"] == "ambiguous_multiple_runs"


def test_report_builder_emits_only_three_pdf_masters_and_keeps_empty_campaign_pending(tmp_path: Path) -> None:
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "task_inventory.json").write_text(
        json.dumps(_inventory()), encoding="utf-8"
    )

    output = report.build_report(tmp_path)

    assert output == tmp_path / "reports" / "COST_AWARE_LAGRANGIAN_REPORT.md"
    text = output.read_text(encoding="utf-8")
    assert "INCOMPLETE:" in text
    assert "no campaign completion or verdict is available" in text
    assert "Analyzer output is absent" in text
    assert "All mandatory horizons and native assessor inputs are present." not in text
    assert "Portable full A+B+T endpoint audits" in text
    assert "topology_with_fidelity" in text
    assert "pending full A+B+T validation audit" in text
    assert (tmp_path / "analysis" / "decision_cost_quality.json").is_file()
    assert (tmp_path / "analysis" / "decision_cost_quality.csv").is_file()
    assert (tmp_path / "analysis" / "mature_window_metrics.csv").is_file()
    machine_rows = json.loads((tmp_path / "analysis" / "decision_cost_quality.json").read_text())
    assert len(machine_rows["records"]) == 6
    pdfs = sorted((tmp_path / "reports").rglob("*.pdf"))
    assert [path.name for path in pdfs] == [
        "figure_1_cost_decision.pdf",
        "figure_2_quality_history.pdf",
        "figure_3_mature_windows_transfer.pdf",
    ]
    assert all(path.stat().st_size > 1000 for path in pdfs)


def test_quality_axis_prioritizes_cold_process_and_reports_wall_separately() -> None:
    rows = [
        {
            "epoch": 10,
            "cold_process_seconds_including_profile_acquisition": 900.0,
            "completion_seconds_from_first_training_command": 500.0,
            "wall_seconds_from_first_training_command": 1200.0,
        },
        {
            "epoch": 20,
            "cold_process_seconds_including_profile_acquisition": 1800.0,
            "completion_seconds_from_first_training_command": 1400.0,
            "wall_seconds_from_first_training_command": 2400.0,
        },
    ]
    x, label = report._elapsed_axis(rows)
    assert x == [0.25, 0.5]
    assert label == "cold check-completion hours including profile acquisition"
    summary = report._arm_metric_summary({"uniform_evaluations": rows})
    assert summary["cold_seconds"] == 1800.0
    assert summary["wall_seconds"] == 2400.0


def test_observed_cold_cost_adds_terminal_evaluations_without_inventing_audits(tmp_path: Path) -> None:
    run = _complete_run()
    run.update(
        {
            "allocated_profile_acquisition_measured_seconds": 20.0,
            "allocated_profile_acquisition_estimated_seconds": 1.0,
            "allocated_profile_acquisition_seconds": 21.0,
        }
    )
    inputs = {
        "root": tmp_path,
        "history_path": tmp_path / "analysis" / "quality_history.json",
        "inventory": {
            "quality_contract": {"monitor_epochs": list(range(0, 101, 10))},
            "tasks": {"G": {**_task_record(["strategy"]), "horizon": 100}},
        },
        "history": {"tasks": {"G": {"arms": {"strategy": {"runs": [run]}}}}},
        "profiles": {task: {} for task in ("G", "D", "X")},
        "raw_profiles": {task: {} for task in ("G", "D", "X")},
        "plans": {task: {} for task in ("G", "D", "X")},
        "calibrations": {task: None for task in ("G", "D", "X")},
        "contracts": {task: None for task in ("G", "D", "X")},
        "recovery_probe": None,
    }
    attempts = [
        {"task_id": "G", "arm": "strategy", "action": "launch", "status": "completed", "charged_process_seconds": 100.0},
        {"task_id": "G", "arm": "strategy", "action": "resume", "status": "completed", "charged_process_seconds": 40.0},
        {"task_id": "G", "arm": "strategy", "action": "evaluate", "status": "completed", "charged_process_seconds": 12.0},
        {"task_id": "G", "arm": "strategy", "action": "evaluate", "status": "failed_or_stopped", "charged_process_seconds": 8.0},
    ]

    report._write_analysis_tables(inputs, attempts)

    row = inputs["decision_cost_records"][0]
    assert row["terminal_training_process_seconds"] == 140.0
    assert row["charged_final_evaluation_process_seconds"] == 20.0
    assert row["final_evaluation_attempt_count"] == 2
    assert row["final_evaluation_charge_status"] == "terminal_charges_complete_for_recorded_evaluations"
    assert row["final_evaluation_scope"] == "all recorded evaluate commands; final-audit completeness is separate"
    assert row["observed_cold_task_total_seconds"] == 181.0

    machine = json.loads((tmp_path / "analysis" / "decision_cost_quality.json").read_text(encoding="utf-8"))
    assert machine["records"][0]["observed_cold_task_total_seconds"] == 181.0
    with (tmp_path / "analysis" / "decision_cost_quality.csv").open(newline="", encoding="utf-8") as handle:
        csv_row = next(csv.DictReader(handle))
    assert csv_row["charged_final_evaluation_process_seconds"] == "20.0"
    assert csv_row["observed_cold_task_total_seconds"] == "181.0"
    table = "\n".join(report._arm_cost_table_lines(inputs))
    assert "G / strategy" in table and "20.0" in table and "181.0" in table

    attempts.append(
        {"task_id": "G", "arm": "strategy", "action": "evaluate", "status": "running", "charged_process_seconds": None}
    )
    records, _ = report._decision_cost_quality_records(inputs, attempts)
    assert records[0]["charged_final_evaluation_process_seconds"] == 20.0
    assert records[0]["final_evaluation_charge_status"] == "recorded_evaluations_incomplete_or_unmeasured"
    assert records[0]["observed_cold_task_total_seconds"] is None


def test_final_interpretation_requires_current_hashes_and_complete_horizons(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis"
    manifests = tmp_path / "manifests"
    analysis.mkdir()
    manifests.mkdir()
    history_path = analysis / "quality_history.json"
    ledger_path = manifests / "campaign_runner_ledger.jsonl"
    history_path.write_text('{"revision": 1}\n', encoding="utf-8")
    ledger_path.write_text('{"attempt_id":"attempt-1"}\n', encoding="utf-8")
    paragraphs = ["The reviewed finding must not survive a stale evidence binding."]
    review_path = analysis / "final_interpretation.json"
    review = {
        "schema": 1,
        "status": "reviewed",
        "paragraphs": paragraphs,
        "evidence_sha256": {
            "quality_history.json": "0" * 64,
            "campaign_runner_ledger.jsonl": report._sha256(ledger_path),
        },
    }
    review_path.write_text(json.dumps(review), encoding="utf-8")
    inputs = {"root": tmp_path, "history_path": history_path}

    stale = report._final_interpretation_state(inputs, all_mandatory_complete=True)
    assert stale["status"] == "stale"
    stale_lines = report._final_interpretation_lines(stale)
    assert "STALE" in " ".join(stale_lines)
    assert paragraphs[0] not in " ".join(stale_lines)

    review["evidence_sha256"]["quality_history.json"] = report._sha256(history_path)
    review_path.write_text(json.dumps(review), encoding="utf-8")
    pending = report._final_interpretation_state(inputs, all_mandatory_complete=False)
    assert pending["status"] == "pending"
    assert paragraphs[0] not in " ".join(report._final_interpretation_lines(pending))

    reviewed = report._final_interpretation_state(inputs, all_mandatory_complete=True)
    assert reviewed["status"] == "reviewed"
    assert paragraphs[0] in " ".join(report._final_interpretation_lines(reviewed))


def test_runtime_policy_summary_links_actual_strategy_coverage_and_keeps_missing_pending(tmp_path: Path) -> None:
    (tmp_path / "tasks").mkdir()
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    (tmp_path / "manifests").mkdir()
    (tmp_path / "tasks" / "task_inventory.json").write_text(
        json.dumps(
            {
                "tasks": {
                    "G": {"role": "mandatory", "arms": ["strategy"]},
                    "D": {"role": "mandatory", "arms": ["strategy"]},
                }
            }
        ),
        encoding="utf-8",
    )
    (analysis / "full_train_audit.json").write_text('{"source":"TRAIN"}\n', encoding="utf-8")
    runtime_summary = {
        "schema": 1,
        "records": [
            {
                "task_id": "G",
                "arm": "strategy",
                "checkpoint_step": 380,
                "checkpoint_sha256": "a" * 64,
                "initial_plan_id": "full",
                "final_plan_id": "nested",
                "accepted_cursor": 380,
                "plan_step_counts": {"full": 120, "nested": 260},
                "topology_included_updates": 260,
                "max_importance_multiplier_observed": 4.0,
                "monitor_overhead_seconds": 12.5,
                "accepted_training_seconds": 125.0,
                "fallback_reason": "timing_guard",
                "full_recovery_intervals": [{"start": 120, "end": 139}],
                "train_audit_epochs": [25, 50, 75, 100],
                "noise_audit_epochs": [25, 50, 75, 100],
                "full_train_audit_artifact": "analysis/full_train_audit.json",
            }
        ],
    }
    summary_path = analysis / "runtime_policy_summary.json"
    summary_path.write_text(json.dumps(runtime_summary), encoding="utf-8")

    inputs = report._load_inputs(tmp_path)
    rendered = "\n".join(report._runtime_policy_table_lines(inputs))

    assert inputs["runtime_policy_summary"] == runtime_summary
    assert f"SHA-256 `{report._sha256(summary_path)}`" in rendered
    assert "G / strategy | recorded" in rendered
    assert "`full` → `nested`" in rendered
    assert "plans `{\"full\":120,\"nested\":260}`" in rendered
    assert "12.50 s / 125.00 s (10.0%)" in rendered
    assert "timing_guard" in rendered
    assert "train `[25,50,75,100]`" in rendered
    assert "[`analysis/full_train_audit.json`](<../analysis/full_train_audit.json>)" in rendered
    assert "D / strategy | pending: no strategy record" in rendered

    missing_root = tmp_path / "missing"
    (missing_root / "tasks").mkdir(parents=True)
    (missing_root / "tasks" / "task_inventory.json").write_text(
        json.dumps({"tasks": {"G": {"arms": ["strategy"]}}}), encoding="utf-8"
    )
    missing_inputs = report._load_inputs(missing_root)
    missing_rendered = "\n".join(report._runtime_policy_table_lines(missing_inputs))
    assert "Pending: `analysis/runtime_policy_summary.json` is absent." in missing_rendered
    assert "G / strategy | pending: schema 1 summary is unavailable" in missing_rendered


def _optional_pair_decision_inputs(tmp_path: Path, *, missing_epoch: int | None = None) -> dict:
    root = tmp_path / "campaign"
    analysis = root / "analysis"
    manifests = root / "manifests"
    tasks_dir = root / "tasks"
    analysis.mkdir(parents=True)
    manifests.mkdir()
    tasks_dir.mkdir()
    source_path = manifests / "science_source_binding_v1.json"
    source_path.write_text('{"frozen":"binding"}\n', encoding="utf-8")
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    ledger_path = manifests / "campaign_runner_ledger.jsonl"
    ledger_prefix = b'{"action":"launch","task_id":"G"}\n'
    ledger_path.write_bytes(ledger_prefix + b'{"action":"evaluate","task_id":"D"}\n')
    decision_path = analysis / "optional_pair_decision.json"
    receipt = {
        "schema_version": 1,
        "status": "omitted_budget",
        "source_binding_sha256": source_sha,
        "ledger_prefix_bytes": len(ledger_prefix),
        "ledger_prefix_sha256": hashlib.sha256(ledger_prefix).hexdigest(),
        "mandatory_science_epochs": 600,
        "available_seconds": 900.0,
        "required_seconds": 1500.0,
        "closeout_reserve_seconds": 3600,
        "reason": "The remaining budget is below the required paired transfer run cost.",
    }
    decision_path.write_text(json.dumps(receipt), encoding="utf-8")
    inventory = _inventory()
    inventory["tasks"]["X"] = {
        "role": "optional",
        "arms": ["dense", "strategy"],
        "horizon": 100,
        "frozen_task_config": {"dataset": {"field_names": ["f1", "f2", "f3", "f4", "f5"]}},
    }
    (tasks_dir / "task_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
    history_tasks = _history_for_run(_complete_run(missing_epoch=missing_epoch))["tasks"]
    history_tasks["X"] = {"arms": {"dense": {"runs": []}, "strategy": {"runs": []}}}
    history = {"tasks": history_tasks}
    history_path = analysis / "quality_history.json"
    history_path.write_text(json.dumps(history), encoding="utf-8")
    return report._load_inputs(root)


def test_valid_optional_pair_budget_omission_is_terminal_and_keeps_x_rows_empty(tmp_path: Path) -> None:
    inputs = _optional_pair_decision_inputs(tmp_path)
    summary = inputs["optional_pair_decision_summary"]
    assert summary["status"] == "omitted_budget"
    assert summary["validated"] is True
    assert summary["mandatory_science_complete"] is True
    assert summary["available_seconds"] == 900.0
    assert summary["required_seconds"] == 1500.0
    assert summary["closeout_reserve_seconds"] == 3600.0
    assert "not run: resource gate; physical transfer is unestablished" in report._optional_pair_decision_text(inputs)
    assert "[validated resource decision](../analysis/optional_pair_decision.json)" in report._optional_pair_decision_text(inputs)

    _write_analysis_tables_for_optional_decision(inputs)
    compact = json.loads((inputs["root"] / "analysis/decision_cost_quality.json").read_text(encoding="utf-8"))
    assert compact["optional_pair_decision"]["status"] == "omitted_budget"
    x_rows = [row for row in compact["records"] if row["task_id"] == "X"]
    assert [(row["arm"], row["arm_evidence_status"], row["run_id"]) for row in x_rows] == [
        ("dense", "pending_run", None),
        ("strategy", "pending_run", None),
    ]
    evidence_text = "\n".join(report._study_dimension_lines(inputs, []))
    assert "not run: resource gate; physical transfer is unestablished" in evidence_text
    assert "900.0 s available versus 1500.0 s required" in evidence_text


def _write_analysis_tables_for_optional_decision(inputs: dict) -> None:
    report._write_analysis_tables(inputs, [])


def test_optional_pair_decision_with_stale_ledger_prefix_stays_pending(tmp_path: Path) -> None:
    inputs = _optional_pair_decision_inputs(tmp_path)
    ledger_path = inputs["root"] / "manifests/campaign_runner_ledger.jsonl"
    current = ledger_path.read_bytes()
    ledger_path.write_bytes(b"X" + current[1:])

    summary = report._optional_pair_decision_summary(inputs)
    assert summary["status"] == "pending"
    assert summary["validated"] is False
    assert "campaign ledger prefix SHA-256 is stale" in summary["validation_errors"]
    assert report._optional_pair_decision_text({**inputs, "optional_pair_decision_summary": summary}) is None


def test_optional_pair_decision_before_all_mandatory_horizons_stays_pending(tmp_path: Path) -> None:
    inputs = _optional_pair_decision_inputs(tmp_path, missing_epoch=90)

    summary = inputs["optional_pair_decision_summary"]
    assert summary["status"] == "pending"
    assert summary["mandatory_science_complete"] is False
    assert "not all six mandatory 100-epoch histories and assessments are complete" in summary["validation_errors"]


def test_machine_tables_preserve_cost_split_censoring_and_raw_target_excursions(tmp_path: Path) -> None:
    fields = ["f1", "f2", "f3", "f4", "f5"]
    ratios = {
        "coherence_sum": 0.8,
        "family_A": 0.9,
        "family_B": 0.95,
        "family_T": 0.9,
        "endpoint_aggregate": 0.92,
        "endpoint_fields": {field: 0.9 for field in fields},
    }
    violating = {**ratios, "family_B": 1.06}
    run = {
        "run_id": "one-run",
        "status": "completed",
        "accepted_end_epoch": 100,
        "quality_assessment_status": "assessed",
        "native_history": {"integrity_status": "ok"},
        "task_profile_acquisition_seconds_measured": 12.0,
        "task_profile_acquisition_seconds_estimated": 1.0,
        "task_profile_acquisition_seconds": None,
        "allocated_profile_acquisition_seconds_measured": 12.0,
        "allocated_profile_acquisition_seconds_estimated": 1.0,
        "allocated_profile_acquisition_seconds": None,
        "uniform_evaluations": [
            {"epoch": 90, "completion_seconds": 450.0, "ratios": violating},
            {"epoch": 100, "completion_seconds": 500.0, "cold_process_seconds_including_profile_acquisition": None, "ratios": ratios},
        ],
        "quality_assessment": {
            "actual_end_epoch": 100,
            "contract": {"maximum_epochs": 100},
            "confirmation": {"status": "right_censored", "censor_epoch": 100, "completion_seconds": 500.0},
            "retention": None,
            "mature_windows": [
                {
                    "epoch": 60,
                    "window_open_epoch": 20,
                    "window_interval": "(e-40,e]",
                    "complete": True,
                    "meets_target": False,
                    "raw_target_checks": 3,
                    "raw_check_count": 5,
                    "metrics": {
                        "family_B": {
                            "pooled_raw": {"numerator": 11.0, "count": 10, "source_numerator": 10.0, "source_count": 10},
                            "pooled_ratio": 1.1,
                            "mean_check_ratio": 1.08,
                            "median_check_ratio": 1.07,
                            "slope_per_epoch": 0.001,
                            "range": [1.02, 1.14],
                            "raw_check_violations": 2,
                            "limit": 1.05,
                        }
                    },
                }
            ],
        },
    }
    inputs = {
        "root": tmp_path,
        "history_path": tmp_path / "analysis" / "quality_history.json",
        "inventory": {
            "quality_contract": {"monitor_epochs": list(range(0, 101, 10))},
            "tasks": {"G": {**_task_record(["strategy"]), "frozen_task_config": {"dataset": {"field_names": fields}}}},
        },
        "history": {"tasks": {"G": {"arms": {"strategy": {"runs": [run]}}}}},
        "profiles": {task: {} for task in ("G", "D", "X")},
        "raw_profiles": {task: {} for task in ("G", "D", "X")},
        "plans": {task: {} for task in ("G", "D", "X")},
        "calibrations": {task: None for task in ("G", "D", "X")},
        "contracts": {task: None for task in ("G", "D", "X")},
        "recovery_probe": None,
    }

    report._write_analysis_tables(inputs, [])

    rows = inputs["decision_cost_records"]
    assert len(rows) == 1
    row = rows[0]
    assert row["task_profile_acquisition_measured_seconds"] == 12.0
    assert row["task_profile_acquisition_estimated_seconds"] == 1.0
    assert row["task_profile_acquisition_total_seconds"] is None
    assert row["allocated_profile_acquisition_total_seconds"] is None
    assert row["charged_final_evaluation_process_seconds"] == 0.0
    assert row["final_evaluation_charge_status"] == "no_recorded_evaluate_attempts"
    assert row["observed_cold_task_total_seconds"] is None
    assert row["latest_accounted_cold_process_seconds"] == 500.0
    assert row["confirmation_status"] == "right_censored"
    assert row["retention_status"] == "not_established_no_confirmed_attainment"
    assert row["raw_whole_target_occupancy"] == {"target_checks": 1, "check_count": 2, "fraction": 0.5}
    assert row["worst_raw_excursions"]["family_B"] == {
        "max_ratio": 1.06,
        "limit": 1.05,
        "max_excess": pytest.approx(0.01),
        "raw_violation_count": 1,
        "raw_check_count": 2,
    }
    metric = inputs["mature_window_metric_records"][0]
    assert (metric["pooled_numerator"], metric["pooled_count"], metric["pooled_source_count"]) == (11.0, 10.0, 10.0)
    with (tmp_path / "analysis" / "mature_window_metrics.csv").open(newline="", encoding="utf-8") as handle:
        csv_row = next(csv.DictReader(handle))
    assert csv_row["window_end_epoch"] == "60"
    assert csv_row["raw_check_violations"] == "2"
    assert csv_row["slope_per_epoch"] == "0.001"

    final_window = {
        "epoch": 100,
        "metrics": {
            "family_A": {"pooled_ratio": 1.01},
            "family_B": {"pooled_ratio": 1.08},
        },
    }
    retention = {
        "status": "lost",
        "through_observed_end_epoch": 100,
        "full_horizon_observed": True,
        "window_epochs": [80, 90, 100],
        "window_target_count": 1,
        "window_count": 3,
        "window_target_fraction": 1 / 3,
        "raw_check_target_count": 1,
        "raw_check_count": 3,
        "raw_check_target_fraction": 1 / 3,
        "final_window": final_window,
        "excursions": [
            {
                "start_window_epoch": 80,
                "end_window_epoch": 90,
                "window_count": 2,
                "duration_epochs": 20,
                "max_excess_by_metric": {"family_B": 0.03},
            }
        ],
        "censored_spans": [
            {
                "start_window_epoch": 100,
                "end_window_epoch": 100,
                "window_epochs": [100],
                "missing_check_epochs": [90],
            }
        ],
    }
    run["quality_assessment"]["confirmation"] = {
        "status": "confirmed",
        "confirmation_epoch": 70,
        "completion_seconds": 350.0,
    }
    run["quality_assessment"]["retention"] = retention
    report._write_analysis_tables(inputs, [])

    row = inputs["decision_cost_records"][0]
    assert row["retention_window_epochs"] == [80, 90, 100]
    assert row["retention_window_target_count"] == 1
    assert row["retention_window_count"] == 3
    assert row["retention_window_target_fraction"] == pytest.approx(1 / 3)
    assert row["retention_raw_check_target_count"] == 1
    assert row["retention_raw_check_count"] == 3
    assert row["retention_raw_check_target_fraction"] == pytest.approx(1 / 3)
    assert row["retention_final_window"] == final_window
    assert row["retention_final_window_pooled_ratios"] == {"family_A": 1.01, "family_B": 1.08}
    assert row["retention_excursions"] == retention["excursions"]
    assert row["retention_censored_spans"] == retention["censored_spans"]

    machine = json.loads((tmp_path / "analysis" / "decision_cost_quality.json").read_text(encoding="utf-8"))
    machine_row = machine["records"][0]
    assert machine_row["retention_excursions"] == retention["excursions"]
    assert machine_row["retention_censored_spans"] == retention["censored_spans"]
    assert machine_row["retention_final_window_pooled_ratios"] == {"family_A": 1.01, "family_B": 1.08}
    with (tmp_path / "analysis" / "decision_cost_quality.csv").open(newline="", encoding="utf-8") as handle:
        decision_row = next(csv.DictReader(handle))
    assert json.loads(decision_row["retention_excursions_json"]) == retention["excursions"]
    assert json.loads(decision_row["retention_censored_spans_json"]) == retention["censored_spans"]
    assert decision_row["retention_window_count"] == "3"
    assert json.loads(decision_row["retention_final_window_pooled_ratios_json"]) == {"family_A": 1.01, "family_B": 1.08}

    retention_table = "\n".join(report._retention_excursion_table_lines(inputs))
    assert "E80–E90, 20 epochs; family_B +0.0300" in retention_table
    assert "E100–E100; missing checks 90" in retention_table


def test_fixed_query_and_profiler_text_require_their_artifact_contracts(tmp_path: Path) -> None:
    inputs = {
        "root": tmp_path,
        "fixed_query_attestation_path": tmp_path / "analysis" / "fixed_query_count_attestation.json",
        "fixed_query_attestation": {
            "status": "static_invariant_under_frozen_fixed_shape_H5_contract",
            "authorized_split": "validation",
            "queries_per_snapshot": 4096,
            "tasks": ["G", "D", "X"],
        },
        "bottleneck_scope_limitations_path": tmp_path / "analysis" / "bottleneck_scope_limitations.json",
        "bottleneck_scope_limitations": {
            "status": "diagnostic_trace_with_incomplete_phase_attribution",
            "tasks": {task: {} for task in ("G", "D", "X")},
        },
    }
    query_text = report._fixed_query_count_text(inputs)
    scope_text = report._bottleneck_scope_text(inputs)
    assert "native JSONL omits valid-mask counts" in query_text
    assert "masked or variable-length future loaders require serialized valid counts" in query_text
    assert "no numeric Amdahl affected fraction" in scope_text

    inputs["fixed_query_attestation"]["queries_per_snapshot"] = 2048
    assert "not independently verified" in report._fixed_query_count_text(inputs)

    erratum_path = tmp_path / "analysis" / "profile_proxy_label_erratum.json"
    erratum_path.parent.mkdir(parents=True, exist_ok=True)
    erratum_path.write_text("{}", encoding="utf-8")
    inputs["profile_proxy_label_erratum_path"] = erratum_path
    inputs["profile_proxy_label_erratum"] = {"schema_version": 1}
    erratum = report._profile_proxy_erratum_text(inputs)
    assert "unnormalized gradients with respect to reconstructed endpoint predictions" in erratum
    assert "Selection uses the separate weighted/C0-normalized model-parameter gradient arrays" in erratum

    inputs["raw_profiles"] = {
        task: {"batches": [{"parameter_expectation_relative_error": value} for value in values]}
        for task, values in {
            "G": [9.0872e-7, 3.6e-7, 3.4e-7, 6.7e-7],
            "D": [3.9e-7, 3.95075e-7, 3.8e-7, 3.7e-7],
            "X": [6.0e-7, 5.1e-7, 6.1e-7, 7.35691e-7],
        }.items()
    }
    summary = report._parameter_expectation_summary(inputs)
    assert "G: max relative error 9.087e-07" in summary
    assert "D: max relative error 3.951e-07" in summary
    assert "X: max relative error 7.357e-07" in summary
    assert "X is software-only" in summary


def test_component_series_pools_same_metric_by_source_weight_and_requires_all_groups(tmp_path: Path) -> None:
    path = tmp_path / "quality_components.csv"
    columns = ("task", "arm", "run_id", "epoch", "panel", "group", "metric_path", "source_value", "child_value")
    leaf = "coherence.components.topology.self.persistence.h0"
    parent = "coherence.components.topology.self.persistence"
    incomplete = "coherence.components.topology.mutual.incomplete_leaf"
    zero_weight = "coherence.components.topology.mutual.zero_weight_leaf"
    rows = []
    for group, source, child in ((0, 1.0, 0.5), (1, 9.0, 9.0), (2, 1.0, 0.5), (3, 9.0, 9.0)):
        rows.append({"task": "G", "arm": "dense", "run_id": "r", "epoch": 10, "panel": "uniform", "group": group, "metric_path": leaf, "source_value": source, "child_value": child})
        rows.append({"task": "G", "arm": "dense", "run_id": "r", "epoch": 10, "panel": "uniform", "group": group, "metric_path": parent, "source_value": source + 100, "child_value": child + 100})
        rows.append({"task": "G", "arm": "dense", "run_id": "r", "epoch": 10, "panel": "uniform", "group": group, "metric_path": zero_weight, "source_value": source, "child_value": child})
        if group < 3:
            rows.append({"task": "G", "arm": "dense", "run_id": "r", "epoch": 10, "panel": "uniform", "group": group, "metric_path": incomplete, "source_value": source, "child_value": child})
    for group in range(4):
        rows.append({"task": "G", "arm": "dense", "run_id": "r", "epoch": 20, "panel": "uniform", "group": group, "metric_path": leaf, "source_value": 0.0, "child_value": 1.0})
    rows.append({"task": "G", "arm": "dense", "run_id": "r", "epoch": 30, "panel": "disjoint", "group": 0, "metric_path": leaf, "source_value": 1.0, "child_value": 99.0})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    series = report._self_mutual_component_series(
        {
            "components_path": path,
            "inventory": {
                "tasks": {
                    "G": {
                        "frozen_task_config": {
                            "coherence": {
                                "families": {
                                    "topology": {
                                        "components": {
                                            "self": {"enabled": True, "weight": 1.0},
                                            "mutual": {"enabled": True, "weight": 0.0},
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
        }
    )

    points = series["G"]["dense"][leaf]
    assert points[0][0] == 10
    assert points[0][1] == pytest.approx(19.0 / 20.0)
    assert points[0][1] != pytest.approx((0.5 + 1.0 + 0.5 + 1.0) / 4.0)
    assert points[0][2:] == (20.0, 19.0)
    assert points[1] == (20, None, 0.0, 4.0)
    assert parent not in series["G"]["dense"]
    assert incomplete not in series["G"]["dense"]
    assert zero_weight not in series["G"]["dense"]
