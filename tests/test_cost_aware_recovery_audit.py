from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import torch

from scripts.posttraining.audit_cost_aware_recovery import (
    RecoveryAuditError,
    capture_probe,
    compare_probe,
    confirm_stopped_probe,
    tensor_tree_sha256,
)

TASK_HASH = "a" * 64
CONFIG_HASH = "b" * 64
SOURCE_HASH = "c" * 64


def _protocol_files(tmp_path: Path) -> tuple[Path, Path, dict[str, dict]]:
    protocol = {
        "schema_version": 1,
        "task": "D",
        "arm": "strategy",
        "checkpoint_step": 76,
        "tolerance": {"numeric_relative": 1e-6, "numeric_absolute": 1e-10},
    }
    decisions = {
        str(cursor): {
            "task_hash": TASK_HASH,
            "plan_id": "nested_topology",
            "selected_actions": ["whole_family_sampling", "generalized_filtration_sampling"],
            "family_inclusions": {
                "global_distribution": True,
                "cross_spectrum": True,
                "topology": cursor in (76, 77),
            },
            "family_probabilities": {
                "global_distribution": 1.0,
                "cross_spectrum": 1.0,
                "topology": 0.5,
            },
            "within_family": {
                "topology": {
                    "provider_id": "topology.complete_filtration.v1",
                    "estimator": "polarity",
                    "conditional_inclusion_probability": 0.5,
                    "seed": 10_000 + cursor,
                }
            },
            "accepted_cursor": cursor,
            "execution_rng_seed": 700211,
            "max_joint_importance_multiplier": 4.0,
            "dense_fidelity_required": True,
        }
        for cursor in (76, 77, 78)
    }
    protocol_path = tmp_path / "protocol.json"
    decisions_path = tmp_path / "decisions.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    decisions_path.write_text(json.dumps(decisions), encoding="utf-8")
    return protocol_path, decisions_path, decisions


def _run_tree(tmp_path: Path) -> Path:
    root = tmp_path / "campaign"
    run_dir = root / "runs" / "D" / "strategy" / "run-001"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "metrics").mkdir()
    (root / "manifests").mkdir()
    (root / "tasks").mkdir()
    (root / "configs").mkdir()
    source_snapshot = root / "source_snapshots/science_v1"
    source_snapshot.mkdir(parents=True)
    (run_dir / "resolved_config.yaml").write_text(
        "stage: post_training\ncase: turbulent_combustion\n", encoding="utf-8"
    )
    strategy_config = root / "configs/D_strategy.yaml"
    strategy_config.write_text("task: D\narm: strategy\n", encoding="utf-8")
    strategy_config_sha = hashlib.sha256(strategy_config.read_bytes()).hexdigest()
    source_binding = {
        "source_snapshot": str(source_snapshot),
        "configs": {"configs/D_strategy.yaml": strategy_config_sha},
    }
    (root / "manifests/science_source_binding_v1.json").write_text(
        json.dumps(source_binding), encoding="utf-8"
    )
    task_contract = {
        "task_id": "D",
        "case_id": "turbulent_combustion",
        "task_hash": TASK_HASH,
        "source_identity": SOURCE_HASH,
        "metadata": {"source_hashes": {"checkpoint": SOURCE_HASH}},
    }
    (root / "tasks/D_task_contract.json").write_text(
        json.dumps(task_contract), encoding="utf-8"
    )
    manifest = {
        "stage": "post_training",
        "case": "turbulent_combustion",
        "experiment_name": f"_CostAwareLPC/{root.name}/D/strategy",
        "run_id": "run-001",
        "parent_run": "/source/D",
        "config_sha256": CONFIG_HASH,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    launch = {
        "schema_version": 1,
        "attempt_id": "launch-001",
        "task_id": "D",
        "arm": "strategy",
        "action": "launch",
        "status": "running",
        "started_utc": _utc(-10),
        "command_sha256": "launch-command-sha",
        "command": ["python", "cost_aware_campaign.py", "_child"],
    }
    (root / "manifests/campaign_runner_ledger.jsonl").write_text(
        json.dumps(launch) + "\n", encoding="utf-8"
    )
    return run_dir


def _utc(seconds_offset: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds_offset)).isoformat().replace(
        "+00:00", "Z"
    )


def _checkpoint(step: int) -> dict:
    return {
        "global_step": step,
        "model": {"weight": torch.tensor([1.0, 2.0])},
        "optimizer": {"state": {0: {"exp_avg": torch.tensor([0.1, 0.2])}}},
        "family_states": {"topology": {"alpha": torch.tensor([0.3])}},
        "rng_state": {"torch_cpu": torch.tensor([7, 8], dtype=torch.uint8)},
        "source_run": "/source/D",
        "source_checkpoint": "/source/D/checkpoints/best.pt",
        "source_hashes": {"checkpoint": SOURCE_HASH},
        "config_sha256": CONFIG_HASH,
        "coherence_calibration_sha256": "d" * 64,
        "family_scales": {"global_distribution": 1.0, "cross_spectrum": 2.0},
        "execution_strategy": {
            "schema_version": 1,
            "task_hash": TASK_HASH,
            "profile_identity": "profile-v1",
            "profiled_epoch": 0,
            "current_plan": {"plan_id": "nested_topology"},
            "execution_state": {
                "task_hash": TASK_HASH,
                "plan_id": "nested_topology",
                "rng_seed": 700211,
                "rng_identity": "rng-v1",
                "accepted_cursor": step,
            },
        },
        "execution_strategy_implementation": {"runtime.py": "e" * 64},
    }


def _write_checkpoint(run_dir: Path, step: int) -> None:
    torch.save(_checkpoint(step), run_dir / "checkpoints/last.pt")


def _row(step: int, decision: dict, *, scalar: float = 1.25, clock: float = 2.0) -> dict:
    return {
        "epoch": 2,
        "step": step,
        "execution": decision,
        "T_estimate": scalar,
        "T_importance_contribution": 2.5,
        "families": {
            "topology": {"scalar_loss": scalar, "weighted_contribution": scalar * 2.0},
            "global_distribution": {"scalar_loss": 0.75},
        },
        "gradient": {
            "optimized_scalar": scalar + 0.5,
            "model_grad_norm": 0.125,
            "update_accepted": True,
            "update_seconds_before_reporting": clock,
        },
    }


def _write_rows(run_dir: Path, decisions: dict[str, dict], *, mutate_step77=None, clock=2.0) -> None:
    rows = [_row(77, decisions["76"], clock=clock), _row(78, decisions["77"], scalar=2.0, clock=clock)]
    if mutate_step77 is not None:
        mutate_step77(rows[0])
    path = run_dir / "metrics/execution_strategy_updates.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _capture_fixture(tmp_path: Path):
    run_dir = _run_tree(tmp_path)
    protocol, decisions_path, decisions = _protocol_files(tmp_path)
    _write_checkpoint(run_dir, 76)
    _write_rows(run_dir, decisions)
    copy_path = tmp_path / "checkpoint_step76.pt"
    captured = capture_probe(
        run_dir,
        protocol_path=protocol,
        decisions_path=decisions_path,
        checkpoint_copy_path=copy_path,
    )
    return run_dir, protocol, decisions_path, decisions, captured


def _stop_capture_fixture(tmp_path: Path):
    run_dir, protocol, decisions_path, decisions, captured = _capture_fixture(tmp_path)
    root = run_dir.parents[3]
    ledger_path = root / "manifests/campaign_runner_ledger.jsonl"
    launch = json.loads(ledger_path.read_text(encoding="utf-8").splitlines()[0])
    terminal = {
        **launch,
        "status": "failed_or_stopped",
        "return_code": -2,
        "finished_utc": _utc(-5),
        "actual_science_epoch_observation": {
            "run_dir": str(run_dir.resolve()),
            "start_epoch": 0.0,
            "accepted_end_epoch": 2.0,
            "observed_history_end_epoch": 2.0526315789473686,
        },
    }
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(terminal) + "\n")
    stopped = confirm_stopped_probe(captured, run_dir, ledger_path=ledger_path)
    return run_dir, protocol, decisions_path, decisions, captured, stopped, ledger_path


def _append_resume_attempt(ledger_path: Path, run_dir: Path) -> None:
    resume_id = "resume-001"
    started = _utc(-4)
    command = ["python", "cost_aware_campaign.py", "_train", "--resume", str(run_dir.resolve())]
    launch = {
        "schema_version": 1,
        "attempt_id": resume_id,
        "task_id": "D",
        "arm": "strategy",
        "action": "resume",
        "status": "running",
        "started_utc": started,
        "command": command,
    }
    terminal = {
        **launch,
        "status": "completed",
        "finished_utc": _utc(2),
        "actual_science_epoch_observation": {
            "run_dir": str(run_dir.resolve()),
            "start_epoch": 2.0,
            "accepted_end_epoch": 3.0,
            "observed_history_end_epoch": 3.0,
            "training_complete": False,
        },
    }
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(launch) + "\n")
        handle.write(json.dumps(terminal) + "\n")


def test_capture_preserves_checkpoint_and_rows_with_cursor_offset(tmp_path: Path) -> None:
    run_dir, _protocol, _decisions_path, decisions, captured = _capture_fixture(tmp_path)

    assert captured["status"] == "pending_replay"
    assert captured["checkpoint"]["global_step"] == 76
    assert captured["checkpoint"]["state_digests"]["model"]
    preserved = captured["speculative_update_log"]["rows"]
    assert set(preserved) == {"77", "78"}
    assert preserved["77"]["decision_cursor"] == 76
    assert preserved["78"]["decision_cursor"] == 77
    assert json.loads(preserved["77"]["raw_jsonl_line"])["execution"] == decisions["76"]
    assert Path(captured["checkpoint"]["byte_copy"]["path"]).is_file()
    assert captured["checkpoint"]["byte_copy"]["sha256"] == captured["checkpoint"]["sha256"]
    assert captured["run_identity"]["case"] == "turbulent_combustion"
    assert captured["run_identity"]["experiment_name"].endswith("/D/strategy")
    assert (run_dir / "checkpoints/last.pt").is_file()


def test_capture_rejects_off_by_one_execution_decision(tmp_path: Path) -> None:
    run_dir = _run_tree(tmp_path)
    protocol, decisions_path, decisions = _protocol_files(tmp_path)
    _write_checkpoint(run_dir, 76)
    _write_rows(run_dir, decisions, mutate_step77=lambda row: row.update(execution=decisions["77"]))

    with pytest.raises(RecoveryAuditError, match="speculative update 77"):
        capture_probe(run_dir, protocol_path=protocol, decisions_path=decisions_path)


def test_compare_verifies_replay_with_tolerance_and_ignores_clock_fields(tmp_path: Path) -> None:
    run_dir, protocol, decisions_path, decisions, captured, stopped, ledger_path = (
        _stop_capture_fixture(tmp_path)
    )
    _write_checkpoint(run_dir, 79)

    def within_tolerance(row: dict) -> None:
        row["gradient"]["model_grad_norm"] += 5e-8

    _write_rows(run_dir, decisions, mutate_step77=within_tolerance, clock=90.0)
    _append_resume_attempt(ledger_path, run_dir)
    result = compare_probe(
        captured,
        run_dir,
        stop_receipt=stopped,
        protocol_path=protocol,
        decisions_path=decisions_path,
        ledger_path=ledger_path,
    )

    assert result["status"] == "verified"
    assert all(result["checks"].values())
    assert result["next_execution_mapping_match"] is True
    assert result["replay_update_diagnostics_match"] is True
    assert result["source_and_config_bindings_unchanged"] is True
    assert result["resumed_same_lineage"] is True
    assert result["replay_rows"]["77"]["expected_decision_cursor"] == 76
    assert result["replay_rows"]["78"]["expected_decision_cursor"] == 77


def test_compare_fails_diagnostic_mismatch_outside_declared_tolerance(tmp_path: Path) -> None:
    run_dir, protocol, decisions_path, decisions, captured, stopped, ledger_path = (
        _stop_capture_fixture(tmp_path)
    )
    _write_checkpoint(run_dir, 79)

    def outside_tolerance(row: dict) -> None:
        row["gradient"]["model_grad_norm"] += 1e-5

    _write_rows(run_dir, decisions, mutate_step77=outside_tolerance)
    _append_resume_attempt(ledger_path, run_dir)
    result = compare_probe(
        captured,
        run_dir,
        stop_receipt=stopped,
        protocol_path=protocol,
        decisions_path=decisions_path,
        ledger_path=ledger_path,
    )

    assert result["status"] == "failed"
    assert result["checks"]["replay_update_diagnostics_match"] is False
    assert any("model_grad_norm" in item["path"] for item in result["numeric_mismatches"])


def test_compare_does_not_verify_changed_source_binding(tmp_path: Path) -> None:
    run_dir, protocol, decisions_path, decisions, captured, stopped, ledger_path = (
        _stop_capture_fixture(tmp_path)
    )
    changed = _checkpoint(79)
    changed["source_hashes"] = {"checkpoint": "f" * 64}
    torch.save(changed, run_dir / "checkpoints/last.pt")
    _write_rows(run_dir, decisions)
    _append_resume_attempt(ledger_path, run_dir)

    result = compare_probe(
        captured,
        run_dir,
        stop_receipt=stopped,
        protocol_path=protocol,
        decisions_path=decisions_path,
        ledger_path=ledger_path,
    )

    assert result["status"] == "failed"
    assert result["checks"]["source_and_config_bindings_unchanged"] is False
    assert result["checks"]["resumed_same_lineage"] is True


def test_compare_requires_checkpoint_to_advance_past_both_replayed_rows(tmp_path: Path) -> None:
    run_dir, protocol, decisions_path, decisions, captured, stopped, ledger_path = (
        _stop_capture_fixture(tmp_path)
    )
    # The ledger and rewritten diagnostic rows prove a real resume attempt,
    # but the checkpoint remains at the stopped step 76.
    _write_rows(run_dir, decisions)
    _append_resume_attempt(ledger_path, run_dir)

    result = compare_probe(
        captured,
        run_dir,
        stop_receipt=stopped,
        protocol_path=protocol,
        decisions_path=decisions_path,
        ledger_path=ledger_path,
    )

    assert result["status"] == "failed"
    assert result["post_resume_checkpoint_advanced"] is False
    assert result["checks"]["post_resume_checkpoint_advanced"] is False


@pytest.mark.parametrize("binding", ["strategy_config", "task_contract", "source_manifest"])
def test_compare_requires_frozen_campaign_bindings_to_remain_unchanged(
    tmp_path: Path, binding: str
) -> None:
    run_dir, protocol, decisions_path, decisions, captured, stopped, ledger_path = (
        _stop_capture_fixture(tmp_path)
    )
    _write_checkpoint(run_dir, 79)
    _write_rows(run_dir, decisions)
    _append_resume_attempt(ledger_path, run_dir)
    campaign_root = run_dir.parents[3]

    if binding == "strategy_config":
        path = campaign_root / "configs/D_strategy.yaml"
        path.write_text(path.read_text(encoding="utf-8") + "changed: true\n", encoding="utf-8")
    elif binding == "task_contract":
        path = campaign_root / "tasks/D_task_contract.json"
        contract = json.loads(path.read_text(encoding="utf-8"))
        contract["source_identity"] = "f" * 64
        path.write_text(json.dumps(contract), encoding="utf-8")
    else:
        path = campaign_root / "manifests/science_source_binding_v1.json"
        source_binding = json.loads(path.read_text(encoding="utf-8"))
        source_binding["source_snapshot"] = str(campaign_root / "source_snapshots/changed")
        path.write_text(json.dumps(source_binding), encoding="utf-8")

    result = compare_probe(
        captured,
        run_dir,
        stop_receipt=stopped,
        protocol_path=protocol,
        decisions_path=decisions_path,
        ledger_path=ledger_path,
    )

    assert result["status"] == "failed"
    assert result["checks"]["resumed_same_lineage"] is True
    assert result["campaign_source_bindings_unchanged"] is False
    assert result["checks"]["campaign_source_bindings_unchanged"] is False


def test_compare_marks_missing_replay_as_not_established(tmp_path: Path) -> None:
    run_dir, protocol, decisions_path, decisions, captured, stopped, ledger_path = (
        _stop_capture_fixture(tmp_path)
    )
    # Matching rows and an advanced checkpoint alone cannot establish that a
    # stop/resume occurred; the runner ledger must show a post-stop resume.
    _write_checkpoint(run_dir, 79)
    _write_rows(run_dir, decisions)
    result = compare_probe(captured, run_dir, stop_receipt=stopped,
        protocol_path=protocol, decisions_path=decisions_path, ledger_path=ledger_path)

    assert result["status"] == "not_established"
    assert result["checks"]["resumed_same_lineage"] is False
    assert result["post_resume_checkpoint_advanced"] is True


def test_stopped_receipt_requires_the_preserved_step76_bytes(tmp_path: Path) -> None:
    run_dir, _protocol, _decisions_path, _decisions, captured = _capture_fixture(tmp_path)
    root = run_dir.parents[3]
    ledger_path = root / "manifests/campaign_runner_ledger.jsonl"
    launch = json.loads(ledger_path.read_text(encoding="utf-8").splitlines()[0])
    terminal = {
        **launch,
        "status": "failed_or_stopped",
        "return_code": -2,
        "finished_utc": _utc(-5),
        "actual_science_epoch_observation": {
            "run_dir": str(run_dir.resolve()),
            "accepted_end_epoch": 2.0,
        },
    }
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(terminal) + "\n")
    _write_checkpoint(run_dir, 79)

    with pytest.raises(RecoveryAuditError, match="global_step and accepted_cursor"):
        confirm_stopped_probe(captured, run_dir, ledger_path=ledger_path)


def test_tensor_tree_digest_detects_state_mutation_and_load_is_cpu(tmp_path: Path) -> None:
    original = {"model": torch.tensor([1.0, 2.0])}
    changed = {"model": torch.tensor([1.0, 2.0001])}
    assert tensor_tree_sha256(original) != tensor_tree_sha256(changed)

    run_dir = _run_tree(tmp_path)
    _write_checkpoint(run_dir, 76)
    from scripts.posttraining.audit_cost_aware_recovery import _load_checkpoint_bytes_cpu

    payload = _load_checkpoint_bytes_cpu(
        (run_dir / "checkpoints/last.pt").read_bytes(), str(run_dir / "checkpoints/last.pt")
    )
    assert payload["model"]["weight"].device.type == "cpu"
