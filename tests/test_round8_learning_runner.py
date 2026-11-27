"""Matched learning preparation and the frozen positional observer seam."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from scripts.posttraining.run_car_round8 import BASE, SPATIAL, matched_configs


def test_matched_arms_differ_only_in_position_and_run_name(monkeypatch):
    import scripts.posttraining.run_car_round8 as runner
    import phycoflow_reconstruction.config.validate as validation

    monkeypatch.setattr(runner, "validate_common", lambda _config: None)
    monkeypatch.setattr(validation, "validate_config", lambda _config: None)
    base = {"optimization": {"epochs": 500, "multitask": {"fidelity_budget": {
        "metric": "physical_member_mse_model_units", "relative_increase": .1,
        "absolute_allowance": 0., "reference_member_mse_from_panel": True,
        "reference_artifact": {"path": "/reference", "sha256": "a" * 64}}}},
        "coherence": {"scalable_anchored_coherence": {"scale_objective": "composite",
            "common_prefit": {"path": "/prefit", "sha256": "b" * 64}}},
        "runtime": {"seed": 47}, "checkpointing": {}, "output": {}}
    position = {"path": "/position", "sha256": "c" * 64}
    arms = matched_configs(base, position)
    assert set(arms) == {BASE, SPATIAL}
    assert arms[SPATIAL]["coherence"]["scalable_anchored_coherence"].pop("position_artifact") == position
    for name in arms:
        assert arms[name]["runtime"]["plot_every_steps"] == 10
        arms[name]["output"].pop("experiment_name")
    assert arms[BASE] == arms[SPATIAL]
    assert list(matched_configs(base, None)) == [BASE]
    assert "position_artifact" not in base["coherence"]["scalable_anchored_coherence"]


def test_position_binding_keeps_ph_and_member_reference_unchanged(tmp_path):
    pytest.importorskip("gudhi")
    from test_frozen_coherence_observer import make_observer
    from phycoflow_reconstruction.coherence.positional_shape import PositionalShapeFitter
    from phycoflow_reconstruction.coherence.query_runtime import _shape_grid
    from phycoflow_reconstruction.coherence.query_sampling import full_target_query_batch
    from phycoflow_reconstruction.coherence.shape_compute import LazyPooledShape
    from phycoflow_reconstruction.training.post_training import _fidelity_reference_binding
    from phycoflow_reconstruction.training.run_store import file_sha256
    from phycoflow_reconstruction.training.scalable_fitting import _digest
    from phycoflow_reconstruction.training.scalable_prefit import attach_position_artifact

    observer, members, families, original = make_observer("conditional")
    shape = LazyPooledShape.from_state_dict(original["shape_artifact"])
    family = families["topology"]
    target = _shape_grid(family, observer.reference, observer.reference)
    source = _shape_grid(family, members[0], observer.reference)
    queries = shape.compile_queries((full_target_query_batch(2),), exhaustive=True)
    fitter = PositionalShapeFitter(shape, family.grid_coordinates, torch.ones(target.shape[-2:]))
    fitter.observe_reference(target, queries)
    fitter.freeze_thresholds()
    fitter.observe_pair(source, target, queries)
    shape.set_position(fitter.freeze())
    candidate = deepcopy(original)
    candidate["shape_artifact"] = shape.state_dict()
    candidate["sha256"] = _digest({key: value for key, value in candidate.items() if key != "sha256"})
    path = tmp_path / "position.pt"
    torch.save(candidate, path)
    declaration = {"path": str(path), "sha256": file_sha256(path)}
    effective = attach_position_artifact(original, declaration)
    assert "position" in effective["shape_artifact"] and "position" not in original["shape_artifact"]
    assert effective["shape_artifact"]["field_mean"] == original["shape_artifact"]["field_mean"]
    assert effective["sha256"] != original["sha256"]

    dataset_file = tmp_path / "dataset"
    dataset_file.write_bytes(b"fixture")
    dataset = SimpleNamespace(path=dataset_file, normalizer=SimpleNamespace(digest=lambda: "b" * 64),
                              field_names=["CO", "T"])
    cfg = {"optimization": {"multitask": {"fidelity_budget": {}}}, "rollout": {},
           "observations": {}, "observation_consistency": {},
           "coherence": {"families": {}, "compute_budget": {}}}
    params = (cfg, dataset, {"checkpoint": "a" * 64}, {},)
    first = _fidelity_reference_binding(*params, SimpleNamespace(artifacts=original), {})
    second = _fidelity_reference_binding(*params, SimpleNamespace(artifacts=effective), {})
    assert first == second  # C0 and member-risk calibration remain common.
    candidate["shape_artifact"]["field_scale"][0] *= 2
    candidate["sha256"] = _digest({key: value for key, value in candidate.items() if key != "sha256"})
    torch.save(candidate, path)
    with pytest.raises(ValueError, match="field transform"):
        attach_position_artifact(original, {"path": str(path), "sha256": file_sha256(path)})


def test_member_recording_uses_actual_clock_and_preserves_earlier_schema(tmp_path):
    import json
    import yaml
    from phycoflow_reconstruction.training.monitoring import TrainingMonitor

    (tmp_path / "resolved_config.yaml").write_text(yaml.safe_dump({
        "coherence": {"families": {"state": {}, "scale": {}, "shape": {}},
                      "scalable_anchored_coherence": {"common_prefit": True}},
        "optimization": {"batch_size": 32, "multitask": {"fidelity_budget": {
            "metric": "physical_member_mse_model_units"}}}}))
    def monitor(start):
        value = TrainingMonitor(tmp_path, start_step=start, final_step=2, configured_steps=2,
            steps_per_epoch=1, description="member-recording", enabled=False)
        value._plot = lambda: None
        return value
    row = {"step": 1, "coherence_loss": 1., "fidelity_estimator": {"members": 3},
           "field_violation": {"one": .1}, "field_signed_mean_risk": {"one": .2}}
    first = monitor(0)
    first.record(row)
    first.close()
    path = tmp_path / "logs/recording_definitions.json"
    definitions = json.loads(path.read_text())
    columns = {column["name"]: column for column in definitions["columns"]}
    assert "32 conditions" in columns["step"]["meaning"]
    assert "1 updates per reporting epoch" in columns["step"]["meaning"]
    assert "last accepted batch" in columns["field_violation"]["meaning"]
    assert "last accepted batch" in columns["field_signed_mean_risk"]["meaning"]
    columns["step"]["meaning"] = "Completed clock or count in the saved B64/12 reporting convention."
    columns["epoch"]["meaning"] = columns["batches"]["meaning"] = columns["step"]["meaning"]
    columns["field_violation"]["meaning"] = "Field-ordered member-MSE excess above its frozen bound divided by the declared positive field scale."
    columns["field_signed_mean_risk"]["meaning"] = "Field-ordered signed mean-risk diagnostic on the same training action, unavailable for a single member."
    path.write_text(json.dumps(definitions))
    previous = path.read_bytes()
    resumed = monitor(1)
    resumed.record({**row, "step": 2})
    resumed.close()
    assert path.read_bytes() == previous
    assert len((tmp_path / "logs/history.jsonl").read_text().splitlines()) == 2


def test_training_requires_a_bound_admitted_arm_and_stops_at_epoch_100(tmp_path, monkeypatch):
    import json
    import yaml
    import scripts.posttraining.run_car_round8 as runner
    import phycoflow_reconstruction.training.post_training as training
    import phycoflow_reconstruction.training.run_store as store

    campaign = tmp_path / "campaign"
    (campaign / "configs").mkdir(parents=True)
    (campaign / "operations").mkdir()
    cfg_path = campaign / "configs/arm.yaml"
    cfg_path.write_text(yaml.safe_dump({"runtime": {"device": "cpu"}}))
    declaration = runner.ref(cfg_path)
    prepared_path = campaign / "configs/conditional_learning_prepared.json"
    prepared_path.write_text(json.dumps({"arms": {BASE: declaration, SPATIAL: declaration},
                                         "assessment": {"split": "validation"}}))
    decision_path = campaign / "operations/learning_decision.json"
    decision = {"status": "sealed", "admitted": True, "test_locked": True,
        "total_gpu_ceiling_seconds": 86400, "question": "question", "forecast": "forecast",
        "admitted_arms": [BASE]}
    args = SimpleNamespace(campaign=campaign, arm=BASE, device="cpu", stop_epoch=100,
                           resume=None, result=campaign / "operations/result.json")
    monkeypatch.setenv("CAR_ROUND8_LEASE_TOKEN", "test-lease")
    monkeypatch.delenv("CAR_ROUND8_LEASE_DEADLINE", raising=False)
    monkeypatch.setattr(runner.signal, "signal", lambda *_args: None)
    decision_path.write_text(json.dumps(decision))
    with pytest.raises(ValueError, match="prepared comparison"):
        runner.train(args)
    decision["prepared_sha256"] = runner.digest(prepared_path)
    decision["admitted_arms"] = [SPATIAL]
    decision_path.write_text(json.dumps(decision))
    with pytest.raises(ValueError, match="explicitly admitted"):
        runner.train(args)

    decision["admitted_arms"] = [BASE]
    decision_path.write_text(json.dumps(decision))
    run = campaign / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints/last.pt").write_bytes(b"saved-checkpoint")
    (run / "resolved_config.yaml").write_text(cfg_path.read_text())
    (run / "status.json").write_text(json.dumps({"global_step": 1200}))
    def fake_training(config, *, case_dir, resume, budget_stop_requested):
        assert config["runtime"]["device"] == "cpu"
        assert not budget_stop_requested(1199, run)
        assert budget_stop_requested(1200, run)
        return run
    monkeypatch.setattr(training, "run_post_training", fake_training)
    monkeypatch.setattr(store, "load_project_checkpoint", lambda _path: {"global_step": 1200})
    runner.train(args)
    result = json.loads(args.result.read_text())
    assert result["target_reached"] and result["reporting_epoch"] == 100
    assert result["checkpoint"] == runner.ref(run / "checkpoints/last.pt")


def test_formal_runner_prepares_existing_prefit_workspace_before_cuda_init(tmp_path, monkeypatch):
    import json
    import os
    import yaml
    import scripts.posttraining.run_car_round8 as runner
    import phycoflow_reconstruction.training.post_training as training
    import phycoflow_reconstruction.training.run_store as store

    campaign = tmp_path / "campaign"
    (campaign / "configs").mkdir(parents=True)
    (campaign / "operations").mkdir()
    cfg_path = campaign / "configs/arm.yaml"
    cfg_path.write_text(yaml.safe_dump({"runtime": {"device": "cpu"}, "coherence": {
        "scalable_anchored_coherence": {"common_prefit": {"sha256": "existing"}}}}))
    prepared = campaign / "configs/conditional_learning_prepared.json"
    prepared.write_text(json.dumps({"arms": {BASE: runner.ref(cfg_path)}, "assessment": {"split": "validation"}}))
    (campaign / "operations/learning_decision.json").write_text(json.dumps({"status": "sealed", "admitted": True,
        "test_locked": True, "total_gpu_ceiling_seconds": 86400, "question": "question", "forecast": "forecast",
        "admitted_arms": [BASE], "prepared_sha256": runner.digest(prepared)}))
    run = campaign / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints/last.pt").write_bytes(b"checkpoint")
    (run / "resolved_config.yaml").write_text(cfg_path.read_text())
    (run / "status.json").write_text(json.dumps({"global_step": 1200}))
    args = SimpleNamespace(campaign=campaign, arm=BASE, device="cuda:0", stop_epoch=100,
                           resume=None, result=campaign / "operations/result.json")
    monkeypatch.setenv("CAR_ROUND8_LEASE_TOKEN", "test")
    monkeypatch.delenv("CAR_ROUND8_LEASE_DEADLINE", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(runner.signal, "signal", lambda *_args: None)
    events = []
    monkeypatch.setattr(torch.cuda, "init", lambda: events.append(("init", os.environ.get("CUBLAS_WORKSPACE_CONFIG"))))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: events.append(("reset", None)))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 123)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 456)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    monkeypatch.setattr(training, "run_post_training", lambda *_args, **_kwargs: run)
    monkeypatch.setattr(store, "load_project_checkpoint", lambda _path: {"global_step": 1200})
    runner.train(args)
    result = json.loads(args.result.read_text())
    assert events == [("init", ":4096:8"), ("reset", None)]
    assert result["numerical_runtime"]["strict_requested"] and result["numerical_runtime"]["strict_algorithms_enabled"]
    assert result["torch_memory"]["peak_allocated_bytes"] == 123
