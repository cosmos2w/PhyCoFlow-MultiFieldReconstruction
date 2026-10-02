"""Exact interrupted/resumed child lifecycle, with controller and line bank."""

import json
from copy import deepcopy

import pytest
import torch
from helpers.coherence import _base_config, _post_config, _write_fixture

from phycoflow_reconstruction.config.validate import validate_config
from phycoflow_reconstruction.training.base_training import run_base_training
from phycoflow_reconstruction.training.checkpointing import PeriodicCheckpointManager
from phycoflow_reconstruction.training.post_training import run_post_training
from phycoflow_reconstruction.training.run_store import (
    RunStore,
    file_sha256,
    load_project_checkpoint,
)


def upgraded_config(path, source):
    post = _post_config(path, source)
    post["dataset"]["grid_shape"] = [4, 4]
    post["objectives"]["data_retention"] = {"enabled": False, "weight": 0.}
    post["optimization"].update(epochs=4, batch_size=2, update_policy="coherence_primal_dual",
                               coherence_gradient_method="config", gradient_balance="weighted_sum")
    post["fidelity_controller"] = {"calibration_batches": 2, "diagnostics_every_steps": 1}
    post["checkpointing"] = {"every_epochs": 1, "validation_every_epochs": 1,
                             "selection_metric": "coherence_with_fidelity"}
    post["evaluation"].update(max_samples=2, query_points=16, preview={"enabled": False})
    post["observation_consistency"] = {"mode": "hard", "final_clamp": True}
    post["posttrain_fidelity"] = {"max_relative_mse_increase": .05,
                                 "max_relative_field_mse_increase": .05}
    distribution = post["coherence"]["families"]["global_distribution"]
    distribution.update(definition="marginal_copula_v2", target_use="paired_supervised",
                        reference_bank={"enabled": False})
    distribution["components"]["mutual"] = {"enabled": False, "weight": 0}
    distribution["components"]["cross"].pop("top_fraction")
    distribution["components"]["cross"].update(include_axes=False,
        copula={"landmarks": 8, "bandwidth": .1, "chunk_size": 8},
        tail={"temperature": .001, "rho": .1, "alpha": .25})
    post["coherence"]["families"]["topology"] = {
        "strategy": "cubical_persistence", "target_use": "paired_supervised",
        "units": "model_units", "fields": ["u", "v"],
        "geometry": {"grid_shape": [4, 4], "neighbors": 1, "periodic": False},
        "filtration": {"dimensions": [0, 1], "smoothing_sigma": 0},
        "persistence": {"projections": 4, "distance": "sliced_wasserstein"},
        "components": {"self": {"enabled": True},
                       "mutual": {"enabled": True, "groups": [["u", "v"]],
                                  "lines": 2, "line_bank_size": 4, "training_subset_size": 2}},
    }
    post["coherence"]["compute_budget"] = {"batch_size": 2, "point_count": 16,
                                              "query_policy": "fixed_shared", "query_seed": 17}
    post["coherence"]["family_balance"] = {"mode": "initial_grad_norm", "calibration_batches": 2}
    return post


def test_upgrade_interrupted_resume_exact_controller_banks_and_optimizer(tmp_path):
    path = tmp_path / "fixture.h5"
    _write_fixture(path)
    source = run_base_training(_base_config(path), case_dir=tmp_path / "case")
    source_hash = file_sha256(source / "checkpoints/last.pt")
    post = upgraded_config(path, source)
    validate_config(post)
    child = run_post_training(post, case_dir=tmp_path / "case", max_steps=2)
    run_post_training(post, case_dir=tmp_path / "case", max_steps=2, resume=child)
    whole = run_post_training(post, case_dir=tmp_path / "whole_case", max_steps=4)
    a = load_project_checkpoint(child / "checkpoints/last.pt")
    b = load_project_checkpoint(whole / "checkpoints/last.pt")
    assert a["global_step"] == b["global_step"] == 4
    assert a["fidelity_controller"]["updates"] == b["fidelity_controller"]["updates"] == 4
    for key in ("multipliers", "ema"):
        assert torch.equal(a["fidelity_controller"][key], b["fidelity_controller"][key])
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key]), key
    for key in a["family_states"]["topology"]:
        assert torch.equal(a["family_states"]["topology"][key], b["family_states"]["topology"][key])
    for key, state in a["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, b["optimizer"]["state"][key][name])
    assert file_sha256(source / "checkpoints/last.pt") == source_hash
    selected = json.loads((child / "evaluation/selected.json").read_text())
    assert selected["eligible"]
    rows = [json.loads(line) for line in (child / "metrics/history.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2, 3, 4]
    assert all(row["data_update_weight"] == 0 for row in rows)
    assert all("update/actual_dot/global_distribution" in row for row in rows)


@pytest.mark.parametrize("change,expected", [
    (lambda c: c["objectives"]["data_retention"].update(enabled=True, weight=.1), "manual"),
    (lambda c: c["model"].update(model_ema_eval=True), "EMA"),
    (lambda c: c["evaluation"].update(split="test"), "test data"),
    (lambda c: c["fidelity_controller"].update(dual_lr=float("nan")), "finite"),
    (lambda c: c["optimization"].update(coherence_gradient_method="unknown"), "method"),
    (lambda c: c["output"].update(experiment_name="../escape"), "relative"),
    (lambda c: (c["output"].update(experiment_name="Test_1002/oops"), c["optimization"].update(epochs=250)), "250"),
])
def test_upgrade_rejects_unsupported_contracts(change, expected):
    post = upgraded_config("not_opened.h5", "not_opened_source")
    candidate = deepcopy(post)
    change(candidate)
    with pytest.raises((ValueError, TypeError), match=expected):
        validate_config(candidate)


def test_selector_recovery_uses_committed_archive_and_retains_previous_files(tmp_path):
    config = {"stage": "post_training", "checkpointing": {
        "every_epochs": 10, "validation_every_epochs": 1,
        "selection_metric": "coherence_with_fidelity"}}
    store = RunStore.create(tmp_path, "selector_recovery", config)
    manager = PeriodicCheckpointManager(config, store=store, steps_per_epoch=1)
    model = torch.nn.Linear(1, 1)

    class Preview:
        enabled = False
        def due(self, step):
            return False
        def update(self, *args, **kwargs):
            return None

    manager.panel_evaluator = lambda step: {
        "eligible": True, "metric": 1. / (step + 1), "mse": 1., "metrics": {},
        "family_source_normalized_scores": {"A": 1. / (step + 1)}}
    manager.save({"model": model.state_dict()}, model=model, preview=Preview(),
                 global_step=0, fallback_metric=0., force=True)
    committed = store.load_checkpoint("last")
    assert committed["coherence_selector_state"]["feasible_archive"][0]["step"] == 0
    for step in range(1, 5):
        manager.save({"model": model.state_dict()}, model=model, preview=Preview(),
                     global_step=step, fallback_metric=0.)
    # Later panel saves prune the in-memory top three but retain committed
    # files until a new rolling checkpoint exists.
    assert (store.run_dir / "checkpoints/feasible_0000000.pt").is_file()
    assert store.load_checkpoint("best")["global_step"] == 4
    recovered = PeriodicCheckpointManager(config, store=store, steps_per_epoch=1)
    recovered.restore_coherence_selector(committed)
    assert recovered.best_value == 1.
    assert [item["step"] for item in recovered.feasible_archive] == [0]
    assert store.load_checkpoint("best")["global_step"] == 0
    selected = json.loads((store.run_dir / "evaluation/selected.json").read_text())
    assert selected["global_step"] == 0
    assert selected["is_source_baseline"]
    assert selected["family_source_normalized_scores"] == {"A": 1.}
    manifest = json.loads((store.run_dir / "run_manifest.json").read_text())
    assert manifest["checkpoint_hashes"]["best"] == file_sha256(
        store.run_dir / "checkpoints/best.pt"
    )


def test_off_panel_recovery_and_milestone_preserve_committed_selector(tmp_path):
    config = {"stage": "post_training", "checkpointing": {
        "every_epochs": 1, "validation_every_epochs": 5, "epochs": [1],
        "selection_metric": "coherence_with_fidelity"}}
    store = RunStore.create(tmp_path, "off_panel_recovery", config)
    manager = PeriodicCheckpointManager(config, store=store, steps_per_epoch=1)
    model = torch.nn.Linear(1, 1)
    panels = []

    class Preview:
        enabled = False
        def due(self, step):
            return False
        def update(self, *args, **kwargs):
            return None

    def panel(step):
        panels.append(step)
        return {"eligible": True, "metric": 1. / (step + 1), "mse": 1.,
                "metrics": {}, "family_source_normalized_scores": {"A": 1.}}

    manager.panel_evaluator = panel
    manager.save({"model": model.state_dict()}, model=model, preview=Preview(),
                 global_step=0, fallback_metric=0., force=True)
    selected_hash = file_sha256(store.run_dir / "checkpoints/best.pt")
    manager.save({"model": model.state_dict()}, model=model, preview=Preview(),
                 global_step=1, fallback_metric=.5)
    assert panels == [0]
    committed = store.load_checkpoint("last")
    assert committed["global_step"] == 1
    milestone = store.load_checkpoint("epoch_001")
    assert milestone["coherence_selector_state"] == committed["coherence_selector_state"]
    assert committed["coherence_selector_state"]["feasible_archive"][0]["step"] == 0
    assert file_sha256(store.run_dir / "checkpoints/best.pt") == selected_hash
    # A newer manifest/panel must not cause recovery from step one to select
    # weights or an archive that did not exist in the committed checkpoint.
    manager.save({"model": model.state_dict()}, model=model, preview=Preview(),
                 global_step=5, fallback_metric=.1)
    recovered = PeriodicCheckpointManager(config, store=store, steps_per_epoch=1)
    recovered.restore_coherence_selector(committed)
    assert recovered.best_value == 1.
    assert [item["step"] for item in recovered.feasible_archive] == [0]
    assert store.load_checkpoint("best")["global_step"] == 0
    with pytest.raises(TypeError, match="missing selector state"):
        recovered.restore_coherence_selector({"global_step": 1})
