"""Scope and early-failure checks for real J1 truncated qualification."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts.qualification import car_round8_production_resume as qualification


def test_short_scalable_horizon_is_rejected_before_worker_or_attempt(tmp_path, monkeypatch):
    cfg = {"optimization": {"epochs": 2},
           "coherence": {"scalable_anchored_coherence": {"enabled": True}}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="at least 10 epochs"):
        qualification.worker(SimpleNamespace(resume=None, config=path))
    monkeypatch.setenv("CAR_ROUND8_LEASE_TOKEN", "test")
    args = SimpleNamespace(config=path, name="Senseiver_Qualification", output=tmp_path / "attempt")
    with pytest.raises(ValueError, match="at least 10 epochs"):
        qualification.orchestrate(args)
    assert not args.output.exists()
    cfg["optimization"]["epochs"] = 10
    qualification.validate_qualification_horizon(cfg)


def test_j1_evidence_distinguishes_active_and_inactive_field_actions():
    row = {"update_mode": "coherence_config_lagrangian_improve", "family_gradient_evaluations": 3,
           "update_accepted": True, "field_signed_mean_risk": None, "constraint_gradient_norm": 0.}
    checkpoint = {"scalable_runtime": {"calibration": {"coefficients": None}}}
    active = [deepcopy(row), {**row, "constraint_gradient_norm": .25}]
    result = qualification.j1_action_evidence(active, checkpoint)
    assert result["fresh_config_actions"] == 2 and result["active_field_actions"] == 1
    assert not result["calibration_frozen"] and "online unit-coefficient" in result["scope"]
    assert qualification.j1_action_evidence([row, row], checkpoint)["active_field_actions"] == 0
    with pytest.raises(ValueError, match="fresh three-family"):
        qualification.j1_action_evidence([{**row, "family_gradient_evaluations": 0}], checkpoint)
    with pytest.raises(ValueError, match="finite"):
        qualification.j1_action_evidence([{**row, "constraint_gradient_norm": float("nan")}], checkpoint)


def test_epoch_one_boundary_retains_trajectory_and_ignores_checkpoint_metadata(tmp_path):
    left, right = tmp_path / "continuous", tmp_path / "resumed"
    for run in (left, right):
        (run / "checkpoints").mkdir(parents=True)
    first = {"global_step": 1, "model": {"parameter": torch.tensor([1.])},
             "optimizer": {"moment": torch.tensor([.2])}, "rng_state": {"torch_cpu": torch.tensor([1, 2])},
             "checkpoint_metric": {"name": "unavailable_external_assessment", "value": None},
             "best_metric_value": None, "best_metric_name": None,
             "scalable_runtime": {"policy_seconds": 1., "occupied_seconds": 2., "sampler_seconds": 3.}}
    second = deepcopy(first)
    for key in ("checkpoint_metric", "best_metric_value", "best_metric_name"):
        second.pop(key)
    second["scalable_runtime"]["occupied_seconds"] = 8.
    torch.save(first, left / "checkpoints/epoch_001.pt")
    torch.save(second, right / "checkpoints/epoch_001.pt")
    matched = qualification.compare_epoch_one(left, right, steps_per_epoch=1)
    assert matched["status"] == "matched" and matched["differences"] == []
    second["model"]["parameter"].add_(.01)
    torch.save(second, right / "checkpoints/epoch_001.pt")
    changed = qualification.compare_epoch_one(left, right, steps_per_epoch=1)
    assert changed["status"] == "different"
    assert any("model/parameter" in diff for diff in changed["differences"])
    (right / "checkpoints/epoch_001.pt").unlink()
    assert qualification.compare_epoch_one(left, right, steps_per_epoch=1)["status"] == "unavailable"


def test_boundary_diagnostic_survives_final_comparison_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("CAR_ROUND8_LEASE_TOKEN", "test")
    cfg = {"optimization": {"epochs": 10, "multitask": {"external_policy": {"lagrangian": {
        "epoch_calibrated_pi": {"steps_per_epoch": 1}}}}},
        "coherence": {"scalable_anchored_coherence": {"enabled": True}}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    args = SimpleNamespace(config=path, name="Senseiver_Qualification", output=tmp_path / "attempt",
                           members=1, device="cpu")
    count = []
    def worker(command, **_kwargs):
        index = len(count)
        count.append(command)
        output = command[command.index("--output") + 1]
        run_dir = "continuous" if index == 0 else "paused"
        with open(output, "w") as stream:
            json.dump({"pid": index + 1, "run_dir": run_dir}, stream)
    monkeypatch.setattr(qualification.subprocess, "run", worker)
    boundary = {"status": "different", "differences": ["epoch_one/model: values differ"]}
    monkeypatch.setattr(qualification, "compare_epoch_one", lambda *_args, **_kw: boundary)
    def fail(*_args, **_kwargs):
        raise ValueError("final mismatch")
    monkeypatch.setattr(qualification, "compare", fail)
    with pytest.raises(ValueError, match="final mismatch"):
        qualification.orchestrate(args)
    receipt = json.loads((args.output / "receipt.json").read_text())
    assert not receipt["qualified"] and receipt["epoch_one_boundary"] == boundary


def test_strict_runtime_resolution_and_config_validation(tmp_path):
    from helpers.coherence import _post_config
    from phycoflow_reconstruction.config.validate import validate_config
    from phycoflow_reconstruction.training.post_training import strict_post_training_algorithms

    cfg = _post_config(tmp_path / "fields.h5", tmp_path / "source")
    assert not strict_post_training_algorithms(cfg)
    cfg["runtime"]["strict_algorithms"] = True
    validate_config(cfg)
    assert strict_post_training_algorithms(cfg)
    cfg["runtime"]["strict_algorithms"] = "true"
    with pytest.raises(TypeError, match="must be boolean"):
        validate_config(cfg)
    cfg["runtime"]["strict_algorithms"] = True
    cfg["runtime"]["deterministic"] = False
    with pytest.raises(ValueError, match="requires deterministic"):
        validate_config(cfg)
    assert strict_post_training_algorithms({"runtime": {"strict_algorithms": False},
        "coherence": {"scalable_anchored_coherence": {"common_prefit": {"sha256": "existing"}}}})


def test_strict_option_reaches_production_seed_before_source_loading(tmp_path, monkeypatch):
    import phycoflow_reconstruction.training.post_training as post
    seen = []
    monkeypatch.setattr(post, "_configure_persistence_workers", lambda _cfg: None)
    monkeypatch.setattr(post, "seed_everything", lambda seed, deterministic, *, strict_algorithms:
                        seen.append((seed, deterministic, strict_algorithms)))
    def stop(_cfg):
        raise RuntimeError("source boundary")
    monkeypatch.setattr(post, "source_hashes", stop)
    cfg = {"stage": "post_training", "optimization": {}, "runtime": {"seed": 47, "device": "cpu",
            "deterministic": True, "strict_algorithms": True}, "coherence": {}}
    with pytest.raises(RuntimeError, match="source boundary"):
        post.run_post_training(cfg, case_dir=tmp_path)
    assert seen == [(47, True, True)]


def test_qualification_sets_cublas_workspace_before_cuda_init_and_records_mode(tmp_path, monkeypatch):
    import os
    import phycoflow_reconstruction.training.post_training as post
    cfg = {"runtime": {"seed": 47, "device": "cuda:0", "deterministic": True, "strict_algorithms": True},
        "optimization": {"epochs": 10, "multitask": {"fidelity_budget": {
            "metric": "physical_member_mse_model_units", "rotating_complement": {}},
            "external_policy": {"lagrangian": {"epoch_calibrated_pi": {}}}}},
        "coherence": {"scalable_anchored_coherence": {"enabled": True}}, "output": {}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    args = SimpleNamespace(config=path, resume=None, output=tmp_path / "worker.json", name="Senseiver_Test",
                           device=None, stop_action=2)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    events = []
    def initialize():
        events.append(("init", os.environ.get("CUBLAS_WORKSPACE_CONFIG")))
    monkeypatch.setattr(torch.cuda, "init", initialize)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: events.append(("reset", None)))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 123)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    monkeypatch.setattr(post, "run_post_training", lambda *_args, **_kwargs: tmp_path / "run")
    qualification.worker(args)
    receipt = json.loads(args.output.read_text())
    assert events == [("init", ":4096:8"), ("reset", None)]
    assert receipt["peak_allocated_bytes"] == 123
    assert receipt["numerical_runtime"]["strict_requested"] and receipt["numerical_runtime"]["strict_algorithms_enabled"]
    assert receipt["numerical_runtime"]["cublas_workspace_config"] == ":4096:8"
