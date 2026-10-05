"""Pilot-only R4 launcher rejects formal, TEST and exposure-contract violations."""

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def launcher(monkeypatch):
    path = Path(__file__).parents[1] / "scripts/training/run_upgrade_1002_r4.py"
    spec = importlib.util.spec_from_file_location("r4_launcher_safety", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(module, "sha", lambda path: module.SOURCE_SHA)
    return module


def _config(launcher):
    return {"stage": "post_training", "case": "turbulent_combustion",
        "output": {"experiment_name": "Test_1002/R4_simple_reference"},
        "optimization": {"epochs": 100, "batch_size": 32, "train_fraction": .15},
        "runtime": {"device": "cuda:0"}, "evaluation": {"split": "validation"},
        "checkpointing": {"selection_metric": "coherence_with_fidelity"},
        "source_run": str(launcher.SOURCE), "source_checkpoint": "last.pt",
        "model": {"model_ema_eval": False}}


@pytest.mark.parametrize("name", ["Test_1002/R4_formal", "Test_1002/R4_candidate_5000ep",
    "Test_1002/R3_old", "../Test_1002/R4_candidate", "/tmp/R4_candidate",
    "Test_1002/R4_candidate/nested"])
def test_formal_names_and_output_escape_are_rejected(launcher, name):
    config = _config(launcher)
    config["output"]["experiment_name"] = name
    with pytest.raises(ValueError):
        launcher.validate(config)


@pytest.mark.parametrize("section,key,value", [
    ("optimization", "epochs", 250), ("optimization", "epochs", 5000),
    ("optimization", "epochs", True), ("optimization", "batch_size", 16),
    ("optimization", "train_fraction", .1), ("optimization", "steps_per_epoch", 20),
    ("optimization", "sampling", "full_pass"), ("runtime", "device", "cuda:1"),
    ("evaluation", "split", "test"), ("checkpointing", "selection_split", "test"),
    ("checkpointing", "selection_metric", "test_loss"), ("model", "model_ema_eval", True)])
def test_fixed_contract_changes_are_rejected(launcher, section, key, value):
    config = _config(launcher)
    config[section][key] = value
    with pytest.raises(ValueError):
        launcher.validate(config)


def test_test_preview_source_hash_and_physical_gpu_mapping_are_rejected(launcher, monkeypatch):
    config = _config(launcher)
    config["evaluation"]["preview"] = {"split": "test"}
    with pytest.raises(ValueError, match="TEST"):
        launcher.validate(config)
    config = _config(launcher)
    monkeypatch.setattr(launcher, "sha", lambda path: "changed")
    with pytest.raises(ValueError, match="checksum"):
        launcher.validate(config)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(ValueError, match="physical GPU0"):
        launcher.validate(config)


def test_exposure_reservation_uses_complete_historical_epochs(launcher):
    config = _config(launcher)
    result = launcher.validate(config, resume_epoch=50)
    assert result["reserved_exposures"] == 50
    result = launcher.validate(config, max_steps=2, resume_epoch=50)
    assert result["reserved_exposures"] == pytest.approx(2 / 38)
    with pytest.raises(ValueError, match="no new exposure"):
        launcher.validate(config, resume_epoch=100)


def test_dry_run_never_spawns_training(launcher, monkeypatch, tmp_path, capsys):
    import phycoflow_reconstruction.cli as cli
    monkeypatch.setattr(cli, "_load_case_config", lambda *args: _config(launcher))
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs:
                        pytest.fail("dry validation launched training"))
    monkeypatch.setattr(sys, "argv", ["launcher", "--config", str(tmp_path / "pilot.yaml"),
                                      "--role", "simple_reference"])
    assert launcher.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["execute"] is False and report["TEST_locked"] is True


@pytest.mark.parametrize("entry,role", [
    ({"reserved_exposures": 620, "role": "simple_reference"}, "simple_reference"),
    ({"reserved_exposures": 0, "gpu0_wall_seconds": 86400, "role": "profile"}, "profile"),
    ({"reserved_exposures": 10, "role": "profile"}, "profile")])
def test_campaign_caps_fail_before_process_creation(launcher, monkeypatch, tmp_path, entry, role):
    import phycoflow_reconstruction.cli as cli
    monkeypatch.setattr(cli, "_load_case_config", lambda *args: _config(launcher))
    monkeypatch.setattr(launcher, "AUDIT", tmp_path)
    monkeypatch.setattr(launcher, "load_ledger", lambda: {"entries": [copy.deepcopy(entry)]})
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs:
                        pytest.fail("exhausted campaign launched training"))
    monkeypatch.setattr(sys, "argv", ["launcher", "--config", str(tmp_path / "pilot.yaml"),
                                      "--role", role, "--execute"])
    with pytest.raises(ValueError, match="cap|hours"):
        launcher.main()
