from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_script = Path(__file__).resolve().parents[1] / "scripts/training/run_upgrade_1002_pilot.py"
_spec = importlib.util.spec_from_file_location("upgrade_1002_pilot_test", _script)
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)
PINNED_SOURCE_SHA256 = pilot.PINNED_SOURCE_SHA256
PilotContractError = pilot.PilotContractError
_apply_attempt = pilot._apply_attempt
_attempt_delta = pilot._attempt_delta
_effective_step_cap = pilot._effective_step_cap
_empty_ledger = pilot._empty_ledger
_register_stage_config = pilot._register_stage_config
_stage_entry = pilot._stage_entry
_validate_visible_device = pilot._validate_visible_device
_verify_resume_identity = pilot._verify_resume_identity
validate_source_identity = pilot.validate_source_identity
validate_test_envelope = pilot.validate_test_envelope


def _pilot_config(**overrides):
    config = {
        "stage": "post_training",
        "case": "turbulent_combustion",
        "source_run": "/pinned/source/run",
        "source_checkpoint": "last.pt",
        "model": {"model_ema_eval": False},
        "dataset": {"path": "/pinned/data.h5"},
        "optimization": {"epochs": 50, "batch_size": 32, "train_fraction": 0.15},
        "runtime": {"device": "cuda:0"},
        "evaluation": {"split": "validation", "preview": {"enabled": True, "split": "validation"}},
        "checkpointing": {"selection_metric": "coherence_with_fidelity"},
        "output": {"experiment_name": "Test_1002/T20_A_copula"},
    }
    for key, value in overrides.items():
        config[key] = value
    return config


def _planned():
    return {
        "maximum_initial_epochs": {"T00": 75, "T10": 100, "T20": 50},
        "every_lineage_epoch_cap": 249,
        "global_accepted_update_cap": 40000,
    }


def test_test_mode_rejects_250_epochs_even_when_caller_has_a_step_limit():
    config = _pilot_config()
    config["optimization"]["epochs"] = 250
    proposed_max_steps = 1
    assert proposed_max_steps < 250
    with pytest.raises(PilotContractError, match="epochs >= 250"):
        validate_test_envelope(config, planned_runs=_planned(), train_count=8000)


@pytest.mark.parametrize(
    "experiment_name",
    [
        "/tmp/outside",
        "Test_1002/../formal",
        "Test_1002/T20_A/formal_run",
        "Test_1002/not_a_stage/experiment",
    ],
)
def test_output_paths_cannot_escape_test_root_or_name_a_formal_run(experiment_name):
    config = _pilot_config()
    config["output"]["experiment_name"] = experiment_name
    with pytest.raises(PilotContractError):
        validate_test_envelope(config, planned_runs=_planned(), train_count=8000)


def test_test_split_selection_and_changed_update_semantics_are_rejected():
    config = _pilot_config()
    config["evaluation"]["split"] = "test"
    with pytest.raises(PilotContractError, match="validation split"):
        validate_test_envelope(config, planned_runs=_planned(), train_count=8000)

    config = _pilot_config()
    config["optimization"]["batch_size"] = 8
    with pytest.raises(PilotContractError, match="batch_size"):
        validate_test_envelope(config, planned_runs=_planned(), train_count=8000)


def test_physical_gpu_mapping_requires_gpu_zero_and_logical_cuda_zero():
    config = _pilot_config()
    _validate_visible_device(config, "0")
    with pytest.raises(PilotContractError, match="CUDA_VISIBLE_DEVICES=0"):
        _validate_visible_device(config, "1")
    config["runtime"]["device"] = "cuda:1"
    with pytest.raises(PilotContractError, match="runtime.device"):
        _validate_visible_device(config, "0")


def test_identical_gpu_names_do_not_override_a_uuid_mismatch(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Identical GPU")
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda index: SimpleNamespace(uuid="gpu-one"))
    monkeypatch.setattr(pilot.subprocess, "check_output",
                        lambda *args, **kwargs: "GPU-gpu-zero, Identical GPU")
    with pytest.raises(PilotContractError, match="UUID"):
        pilot._verify_gpu0()
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda index: SimpleNamespace(uuid="gpu-zero"))
    assert pilot._verify_gpu0()["uuid"] == "GPU-gpu-zero"


def test_source_contract_requires_live_pinned_checkpoint_and_non_ema_model(tmp_path):
    source_run = tmp_path / "source"
    (source_run / "checkpoints").mkdir(parents=True)
    checkpoint = source_run / "checkpoints" / "last.pt"
    checkpoint.write_bytes(b"synthetic source")
    source_contract = {
        "source_weight_selection": "live",
        "source_checkpoint_sha256": PINNED_SOURCE_SHA256,
        "source_global_step": 315000,
        "dataset_path": "/pinned/data.h5",
        "dataset_fingerprint": "fixture",
        "runs": {"source": {"path": str(source_run)}},
    }
    config = _pilot_config()
    config["source_run"] = str(source_run)
    source = validate_source_identity(
        config,
        source_contract,
        sha256_fn=lambda path: PINNED_SOURCE_SHA256,
    )
    assert source["source_checkpoint_sha256"] == PINNED_SOURCE_SHA256
    assert source["source_weight_selection"] == "live"

    config["model"]["model_ema_eval"] = True
    with pytest.raises(PilotContractError, match="live weights"):
        validate_source_identity(
            config,
            source_contract,
            sha256_fn=lambda path: PINNED_SOURCE_SHA256,
        )


def test_stage_config_revision_cap_records_two_targeted_fixes():
    entry = _stage_entry(_empty_ledger(40000), "T20")
    assert _register_stage_config(entry, "base", "Test_1002/T20_A") == "initial"
    assert _register_stage_config(entry, "fix-1", "Test_1002/T20_A_v2") == "targeted_fix_1"
    assert _register_stage_config(entry, "fix-2", "Test_1002/T20_A_v3") == "targeted_fix_2"
    with pytest.raises(PilotContractError, match="targeted fixes"):
        _register_stage_config(entry, "fix-3", "Test_1002/T20_A_v4")
    assert entry["targeted_fix_count"] == 2


def test_stage_initial_epoch_cap_is_stricter_than_global_lineage_cap():
    config = _pilot_config()
    config["optimization"]["epochs"] = 51
    with pytest.raises(PilotContractError, match="T20 allows at most 50"):
        validate_test_envelope(config, planned_runs=_planned(), train_count=8000)


def test_resume_step_reservation_obeys_stage_lineage_and_global_remaining_budgets():
    first_segment = _effective_step_cap(
        requested_max_steps=50,
        configured_steps=1900,
        start_step=0,
        stage_remaining=1800,
        global_remaining=40,
        lineage_remaining=1900,
        is_resume=False,
    )
    assert first_segment == 40

    # A resumed invocation sees the spent stage/global/lineage budgets and may
    # continue only within all three remaining limits.
    resumed_segment = _effective_step_cap(
        requested_max_steps=100,
        configured_steps=1900,
        start_step=40,
        stage_remaining=25,
        global_remaining=30,
        lineage_remaining=60,
        is_resume=True,
    )
    assert resumed_segment == 25


def test_resume_requires_exact_resolved_config_hash_and_saved_stage(tmp_path):
    run_dir = tmp_path / "Test_1002" / "T20_A" / "fixture_run"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "config_sha256": "same-config",
                "experiment_name": "Test_1002/T20_A_copula",
            }
        ),
        encoding="utf-8",
    )
    key = str(run_dir.resolve())
    ledger = {
        "lineages": {
            key: {
                "config_sha256": "same-config",
                "stage": "T20",
                "attempted_updates": 5,
            }
        }
    }
    lineage, counters = _verify_resume_identity(
        run_dir,
        _pilot_config(),
        "same-config",
        ledger,
        "T20",
    )
    assert lineage == key
    assert counters["step"] == 0
    with pytest.raises(PilotContractError, match="resolved config hash"):
        _verify_resume_identity(
            run_dir,
            _pilot_config(),
            "changed-config",
            ledger,
            "T20",
        )


def test_interrupted_resume_ledger_counts_checkpoint_and_uncommitted_history(tmp_path):
    run_dir = tmp_path / "Test_1002" / "T20_A" / "20261002T120000Z_fixture"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "metrics").mkdir(parents=True)
    torch.save(
        {
            "global_step": 5,
            "component_controller": {"attempted": 5, "accepted": 4, "rejected": 1},
        },
        run_dir / "checkpoints" / "last.pt",
    )
    (run_dir / "metrics" / "history.jsonl").write_text(
        json.dumps(
            {
                "step": 7,
                "batches": 2,
                "update_accepted_fraction": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    active = {
        "start_step": 3,
        "start_attempted": 3,
        "start_accepted": 2,
    }
    delta = _attempt_delta(run_dir, active)
    assert delta == {
        "attempted": 4,
        "accepted": 3,
        "end_step": 5,
        "committed_accepted_total": 4,
        "committed_attempted_total": 5,
    }


def test_failed_resume_attempt_updates_global_stage_and_lineage_totals(tmp_path):
    run_dir = tmp_path / "Test_1002" / "T20_A" / "20261002T120000Z_fixture"
    (run_dir / "checkpoints").mkdir(parents=True)
    torch.save({"global_step": 4}, run_dir / "checkpoints" / "last.pt")
    ledger = _empty_ledger(40000)
    active = {
        "stage": "T20",
        "run_dir": str(run_dir),
        "lineage_key": str(run_dir),
        "config_sha256": "fixture-config-hash",
        "implementation_commit": "repair-commit",
        "experiment_name": "Test_1002/T20_A",
        "configured_epochs": 50,
        "steps_per_epoch": 38,
        "start_step": 2,
        "start_attempted": 2,
        "start_accepted": 2,
        "started_utc": "2026-10-02T12:00:00+00:00",
        "revision_type": "initial",
        "kind": "resume",
    }
    _apply_attempt(ledger, active, run_dir=run_dir, status="failed")
    assert ledger["accepted_updates"] == 2
    assert ledger["attempted_updates"] == 2
    assert ledger["stages"]["T20"]["attempted_updates"] == 2
    assert ledger["lineages"][str(run_dir.resolve())]["attempted_updates"] == 2
    assert ledger["runs"][-1]["status"] == "failed"
    assert ledger["runs"][-1]["implementation_commit"] == "repair-commit"
    assert ledger["lineages"][str(run_dir.resolve())]["last_implementation_commit"] == "repair-commit"


def test_dry_run_prints_full_contract_without_model_launch_or_ledger_mutation(
    tmp_path, monkeypatch, capsys
):
    config = _pilot_config()
    source_run = tmp_path / "source"
    (source_run / "checkpoints").mkdir(parents=True)
    (source_run / "checkpoints/last.pt").write_bytes(b"fixture source")
    source_contract = {"source_weight_selection": "live", "source_checkpoint_sha256": PINNED_SOURCE_SHA256,
        "source_global_step": 315000, "train_count": 8000, "dataset_path": "/pinned/data.h5",
        "runs": {"source": {"path": str(source_run)}}}
    source_contract_path = tmp_path / "source_contract.json"
    source_contract_path.write_text(json.dumps(source_contract))
    planned_path = tmp_path / "planned_runs.json"
    planned_path.write_text(json.dumps(_planned()))
    monkeypatch.setattr(pilot, "SOURCE_CONTRACT_PATH", source_contract_path)
    monkeypatch.setattr(pilot, "PLANNED_RUNS_PATH", planned_path)
    monkeypatch.setattr(pilot, "_git_identity", lambda: {"branch": pilot.EXPECTED_BRANCH, "commit": "fixture"})
    monkeypatch.setattr(pilot, "validate_source_identity", lambda config, contract:
        validate_source_identity(config, contract, sha256_fn=lambda path: PINNED_SOURCE_SHA256))
    config["source_run"] = source_contract["runs"]["source"]["path"]
    config["dataset"]["path"] = source_contract["dataset_path"]
    output_config = tmp_path / "resolved.yaml"
    monkeypatch.setattr(pilot, "_resolved_config", lambda path: config)
    ledger_path = tmp_path / "stage_ledger.json"
    lock_path = tmp_path / "launcher.lock"
    monkeypatch.setattr(pilot, "LEDGER_PATH", ledger_path)
    monkeypatch.setattr(pilot, "LOCK_PATH", lock_path)

    result = pilot.run_pilot(
        config_path=output_config,
        max_steps=1,
        dry_run=True,
        env={"CUDA_VISIBLE_DEVICES": "0"},
    )
    output = capsys.readouterr().out
    payload = json.loads(output.split("\nOutput path:", 1)[0])
    assert result is None
    assert payload["resolved_config"] == config
    assert payload["effective_max_steps"] == 1
    assert payload["source"]["source_checkpoint_sha256"] == PINNED_SOURCE_SHA256
    assert payload["formal_launch_authorized"] is False
    assert payload["implementation_commit"] == "fixture"
    assert "Test_1002/T20_A_copula" in output
    assert not ledger_path.exists()
