from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.posttraining import finalize_cost_aware_records as finalize


def test_ledger_dedup_uses_one_latest_charge_per_attempt() -> None:
    rows = [
        {
            "attempt_id": "a",
            "task_id": "D",
            "arm": "strategy",
            "action": "launch",
            "status": "running",
            "started_utc": "start",
            "reserved_process_seconds": 100.0,
        },
        {
            "attempt_id": "a",
            "task_id": "D",
            "arm": "strategy",
            "action": "launch",
            "status": "completed",
            "started_utc": "start",
            "finished_utc": "finish",
            "charged_process_seconds": 12.0,
            "charged_science_epochs": 100.0,
        },
        {
            "attempt_id": "a",
            "task_id": "D",
            "arm": "strategy",
            "action": "launch",
            "status": "completed",
            "accounting_reconciliation": {"note": "same attempt, updated metadata"},
            "charged_process_seconds": 12.0,
            "charged_science_epochs": 100.0,
        },
        {
            "attempt_id": "b",
            "task_id": "G",
            "arm": "profile",
            "action": "profile",
            "status": "running",
            "reserved_process_seconds": 20.0,
        },
    ]

    attempts = finalize.deduplicate_ledger_attempts(rows)
    assert len(attempts) == 2
    attempt_a = next(item for item in attempts if item["attempt_id"] == "a")
    assert attempt_a["event_count"] == 3
    assert attempt_a["accounting"]["charged_process_seconds"] == 12.0
    assert attempt_a["accounting"]["charged_science_epochs"] == 100.0
    assert attempt_a["finished_utc"] == "finish"

    totals = finalize._resource_totals(attempts)["recorded_charges"]
    assert totals["charged_process_seconds"]["recorded_sum"] == 12.0
    assert totals["charged_process_seconds"]["attempts_with_recorded_value"] == 1
    assert totals["charged_process_seconds"]["attempts_missing_value"] == 1
    assert totals["charged_process_seconds"]["complete_for_latest_attempts"] is False


def test_profile_compaction_excludes_support_points_and_raw_vectors() -> None:
    compact = finalize.compact_profile(
        "D",
        {
            "schema_version": 1,
            "task_hash": "task-hash",
            "profiled_epoch": 0,
            "intrinsic_noise_variance": 2.5,
            "measurements": [
                {
                    "plan_id": "nested",
                    "mean_epoch_cost_s": 12.0,
                    "family_support": [{"raw_vector": [1, 2, 3]}],
                    "gradient_vector": [4, 5, 6],
                    "max_joint_importance_multiplier": 4.0,
                }
            ],
        },
        {"path": "profiles/D_profile.json", "sha256": "digest"},
    )
    measurement = compact["measurements"][0]
    assert measurement["plan_id"] == "nested"
    assert measurement["max_joint_importance_multiplier"] == 4.0
    assert "family_support" not in measurement
    assert "gradient_vector" not in measurement


def test_owned_json_writer_preserves_bytes_when_only_generation_time_changes(tmp_path: Path) -> None:
    path = tmp_path / "owned.json"
    first = {
        "schema": "test-owned.v1",
        "root": "campaign",
        "generated_at_utc": "2026-10-24T01:00:00Z",
        "payload": {"profile_sha256": "abc123", "nested": {"generated_at_utc": "stable"}},
    }
    finalize._write_json_owned(path, first, "test-owned.v1")
    original_bytes = path.read_bytes()
    original_sha256 = finalize._sha256(path)

    rerendered = {**first, "generated_at_utc": "2026-10-24T02:00:00Z"}
    finalize._write_json_owned(path, rerendered, "test-owned.v1")
    assert path.read_bytes() == original_bytes
    assert finalize._sha256(path) == original_sha256
    assert json.loads(path.read_text(encoding="utf-8"))["generated_at_utc"] == first["generated_at_utc"]

    changed_data = {
        **rerendered,
        "payload": {"profile_sha256": "def456", "nested": {"generated_at_utc": "stable"}},
    }
    finalize._write_json_owned(path, changed_data, "test-owned.v1")
    assert path.read_bytes() != original_bytes
    assert json.loads(path.read_text(encoding="utf-8")) == changed_data


def test_aggregate_outputs_have_compact_profiles_and_separate_readiness_estimates(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    for directory in ("analysis", "manifests", "plans", "profiles", "tasks"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    for task_id in finalize.TASK_IDS:
        (root / "tasks" / f"{task_id}_task_contract.json").write_text(
            json.dumps({"task_id": task_id, "task_hash": f"{task_id}-hash"}), encoding="utf-8"
        )
        (root / "tasks" / f"{task_id}_action_registry.json").write_text(
            json.dumps([{"action_id": "full"}]), encoding="utf-8"
        )
        (root / "profiles" / f"{task_id}_profile.json").write_text(
            json.dumps(
                {
                    "task_hash": f"{task_id}-hash",
                    "profile_overhead_s": 3.0,
                    "raw_parameter_vectors": [[1.0, 2.0]],
                    "measurements": [
                        {
                            "plan_id": "full",
                            "mean_epoch_cost_s": 1.0,
                            "family_support": [{"point": 1}],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (root / "plans" / f"{task_id}_plan_decision.json").write_text(
            json.dumps({"plan_id": "full", "selected_actions": []}), encoding="utf-8"
        )
    (root / "manifests" / "campaign_runner_ledger.jsonl").write_text(
        '\n'.join(
            json.dumps(row)
            for row in [
                {"attempt_id": "a", "task_id": "G", "action": "profile", "status": "running"},
                {
                    "attempt_id": "a",
                    "task_id": "G",
                    "action": "profile",
                    "status": "completed",
                    "charged_process_seconds": 2.0,
                    "charged_profile_seconds": 2.0,
                    "charged_science_epochs": 0.0,
                    "charged_aux_epochs": 0.0,
                    "charged_closeout_seconds": 0.0,
                },
            ]
        )
        + '\n',
        encoding="utf-8",
    )
    (root / "analysis" / "readiness_attempts.jsonl").write_text(
        json.dumps(
            {
                "phase": "G eval preflight",
                "outcome": "rejected before GPU admission",
                "conservative_command_seconds": 1,
                "gpu_process_seconds": 0,
                "optimizer_exposures": 0,
            }
        )
        + "\n"
        + json.dumps(
            {
                "phase": "D profile preflight",
                "outcome": "rejected before GPU admission",
                "conservative_command_seconds": 1,
                "gpu_process_seconds": 0,
                "optimizer_exposures": 0,
            }
        )
        + '\n',
        encoding="utf-8",
    )

    records = finalize.aggregate_campaign(root)
    assert set(records) == {
        "task_contracts.json",
        "action_registry.json",
        "profiles.json",
        "plan_decisions.jsonl",
        "resource_summary.json",
    }
    profile = records["profiles.json"]["tasks"]["G"]
    assert profile["profile_overhead_s"] == 3.0
    assert "raw_parameter_vectors" not in profile
    assert "family_support" not in profile["measurements"][0]

    resources = records["resource_summary.json"]
    assert resources["ledger"]["event_row_count"] == 2
    assert resources["ledger"]["unique_attempt_count"] == 1
    assert resources["ledger"]["recorded_charges"]["charged_process_seconds"]["recorded_sum"] == 2.0
    readiness = resources["cpu_readiness_estimates"]
    assert readiness["recorded_estimated_command_seconds"] == 2.0
    assert readiness["allocated_to_strategy_profile_estimate_seconds"] == 1.0
    assert readiness["counted_in_runner_actuals"] is False
    assert readiness["counted_as_measured_profile_acquisition"] is False
    assert readiness["counted_as_gpu_process_time"] is False
    records_by_phase = {row["phase"]: row for row in readiness["records"]}
    assert records_by_phase["D profile preflight"]["allocated_to_strategy_profile_estimate"] is True
    assert records_by_phase["G eval preflight"]["allocated_to_strategy_profile_estimate"] is False
    assert records_by_phase["D profile preflight"]["counted_as_measured_profile_acquisition"] is False

    assert resources["measured_totals"] == {
        "science_epochs": 0.0,
        "auxiliary_epoch_exposures": 0.0,
        "combined_epoch_exposures": 0.0,
        "process_seconds": 2.0,
        "profile_calibration_seconds": 2.0,
        "closeout_seconds": 0.0,
    }
    assert resources["accounted_process_plus_cpu_readiness_estimates_seconds"] == 4.0
    assert resources["resource_accounting_complete"] is True
    assert resources["within_caps"]["all"] is True

    output_paths = finalize.write_aggregates(root, records)
    assert {path.name for path in output_paths} == set(records)


def test_resource_summary_matches_deduplicated_campaign_totals_and_keeps_estimates_separate() -> None:
    attempts = [
        {
            "attempt_id": "complete-campaign",
            "action": "evaluate",
            "accounting": {
                "charged_science_epochs": 600.0,
                "charged_aux_epochs": 6.657894736842105,
                "charged_process_seconds": 33473.099595753985,
                "charged_profile_seconds": 1032.3274224350098,
                "charged_closeout_seconds": 146.13364308398013,
            },
        }
    ]
    readiness = {"recorded_estimated_command_seconds": 2.0}

    summary = finalize.resource_budget_summary(attempts, readiness)

    assert summary["measured_totals"] == {
        "science_epochs": 600.0,
        "auxiliary_epoch_exposures": 6.657894736842105,
        "combined_epoch_exposures": 606.6578947368421,
        "process_seconds": 33473.099595753985,
        "profile_calibration_seconds": 1032.3274224350098,
        "closeout_seconds": 146.13364308398013,
    }
    assert summary["recorded_cpu_readiness_estimates_seconds"] == 2.0
    assert summary["accounted_process_plus_cpu_readiness_estimates_seconds"] == 33475.099595753985
    assert summary["caps"] == {
        "science_epochs": 800.0,
        "auxiliary_epoch_exposures": 50.0,
        "combined_epoch_exposures": 850.0,
        "process_seconds": 57600.0,
        "profile_calibration_seconds": 5400.0,
        "closeout_seconds": 3600.0,
    }
    assert summary["within_caps"]["all"] is True
    assert "not added again" in summary["scope_note"]


def test_final_resource_accounting_rejects_active_or_missing_process_charges(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    (root / "manifests").mkdir(parents=True)
    active_attempt = {
        "attempt_id": "active",
        "status": "completed",
        "action": "launch",
        "accounting": {
            "charged_process_seconds": 1.0,
            "charged_profile_seconds": 0.0,
            "charged_science_epochs": 0.0,
            "charged_aux_epochs": 0.0,
        },
    }
    (root / "manifests" / "active_process.json").write_text("{}", encoding="utf-8")
    with pytest.raises(finalize.FinalizeError, match="active-process receipt"):
        finalize._assert_final_resource_accounting(root, [active_attempt])

    (root / "manifests" / "active_process.json").unlink()
    missing_process = {
        **active_attempt,
        "attempt_id": "missing-process",
        "accounting": {**active_attempt["accounting"], "charged_process_seconds": None},
    }
    with pytest.raises(finalize.FinalizeError, match="missing or invalid charged_process_seconds"):
        finalize._assert_final_resource_accounting(root, [missing_process])


def test_relative_markdown_links_and_images_rebase_with_query_and_fragment(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    source_report = root / "reports" / "report.md"
    source_report.parent.mkdir(parents=True)
    source_report.write_text("", encoding="utf-8")
    text = (
        '[data](../analysis/metrics.json?view=short#row-1 "Metrics") '
        '![figure](figures/figure%201.pdf "PDF figure") '
        '[section](#method) [remote](https://example.org/a.pdf) '
        '<https://example.org/plain.pdf>'
    )

    rewritten = finalize.rebase_markdown_links(text, source_report, root, "reports/campaign_evidence")
    assert (
        '[data](reports/campaign_evidence/analysis/metrics.json?view=short#row-1 "Metrics")'
        in rewritten
    )
    assert (
        '![figure](reports/campaign_evidence/reports/figures/figure%201.pdf "PDF figure")'
        in rewritten
    )
    assert "[section](#method)" in rewritten
    assert "[remote](https://example.org/a.pdf)" in rewritten
    assert "<https://example.org/plain.pdf>" in rewritten


def test_export_creates_owned_ignored_alias_and_only_overwrites_own_report(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    root = tmp_path / "owned_campaign"
    (repo / "reports").mkdir(parents=True)
    (root / "reports" / "figures").mkdir(parents=True)
    source = root / "reports" / finalize.REPORT_NAME
    source.write_text("# Report\n\n![figure](figures/f1.pdf)\n", encoding="utf-8")
    (root / "reports" / "figures" / "f1.pdf").write_bytes(b"pdf")

    destination = repo / finalize.REPORT_NAME
    destination.write_text("unrelated user report\n", encoding="utf-8")
    with pytest.raises(finalize.FinalizeError, match="ownership marker"):
        finalize.export_repository_report(root, repo)
    assert not (repo / "reports" / "owned_campaign_evidence").exists()

    destination.unlink()
    exported = finalize.export_repository_report(root, repo)
    alias = repo / "reports" / "owned_campaign_evidence"
    assert alias.is_symlink()
    assert alias.resolve() == root.resolve()
    report_text = exported.read_text(encoding="utf-8")
    assert report_text.startswith(f"{finalize.REPORT_MARKER_PREFIX}{root.resolve()} -->\n")
    assert "![figure](reports/owned_campaign_evidence/reports/figures/f1.pdf)" in report_text

    source.write_text("# Updated report\n", encoding="utf-8")
    finalize.export_repository_report(root, repo)
    assert "# Updated report" in destination.read_text(encoding="utf-8")


def test_evidence_alias_rejects_wrong_target_and_non_symlink_occupant(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    report_dir = repo / "reports"
    root = tmp_path / "campaign"
    other = tmp_path / "other"
    report_dir.mkdir(parents=True)
    root.mkdir()
    other.mkdir()
    alias = report_dir / "campaign_evidence"
    alias.symlink_to(other, target_is_directory=True)

    with pytest.raises(finalize.FinalizeError, match="points elsewhere"):
        finalize.ensure_evidence_alias(repo, root)

    alias.unlink()
    alias.mkdir()
    with pytest.raises(finalize.FinalizeError, match="occupied by a non-symlink"):
        finalize.ensure_evidence_alias(repo, root)
