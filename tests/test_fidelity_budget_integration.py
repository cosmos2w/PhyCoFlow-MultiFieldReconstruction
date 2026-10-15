"""Exercise shared reconstruction, replay omission and ordinary child recovery."""

import json
import random
from dataclasses import replace

import numpy as np
import pytest
import torch
from helpers.coherence import _base_config, _post_config, _write_fixture
from helpers.multifamily import _global_config

from phycoflow_reconstruction.training.base_training import run_base_training
from phycoflow_reconstruction.training.post_training import run_post_training
from phycoflow_reconstruction.training.run_store import load_project_checkpoint


@pytest.mark.parametrize("deterministic", [False, True])
def test_shared_endpoint_no_native_replay_and_resume(tmp_path, monkeypatch, deterministic):
    pytest.importorskip("cooper")
    import phycoflow_reconstruction.training.post_training as trainer

    path = tmp_path / "fixture.h5"
    _write_fixture(path)
    base = _base_config(path)
    if deterministic:
        base["model"] = {
            "name": "senseiver", "width": 16, "num_latents": 4, "heads": 2, "depth": 1,
        }
    case_dir = tmp_path / "case"
    source = run_base_training(base, case_dir=case_dir)
    post = _post_config(path, source)
    post["model"] = base["model"]
    post["optimization"].update(
        epochs=2, steps_per_epoch=1, sampling="full_pass",
        multitask={
            "family_scalarization": {"method": "dwa"},
            "outer_aggregation": None,
            "external_policy": {"method": "lagrangian", "vector_reduction": "sum"},
            "fidelity_budget": {"calibration_batches": 2, "calibration_draws": 2},
        },
    )
    post["objectives"]["data_retention"] = {"enabled": False, "weight": 0.0}
    post["coherence"]["compute_budget"]["batch_size"] = 2
    post["coherence"]["families"] = {"global_distribution": _global_config()}
    post["coherence"]["family_balance"] = {
        "mode": "initial_grad_norm", "calibration_batches": 2,
    }
    post["checkpointing"] = {"every_epochs": 1, "selection_metric": "reconstruction_mse"}
    post["evaluation"]["preview"] = {"enabled": False}
    post["runtime"]["progress"] = False

    native = 0
    original_load = trainer.load_source_model

    def load(*args, **kwargs):
        model, dataset, metadata = original_load(*args, **kwargs)

        def forbidden(*args, **kwargs):
            nonlocal native
            native += 1
            raise AssertionError("external path replayed native training loss")

        model.training_loss = forbidden
        return model, dataset, metadata

    monkeypatch.setattr(trainer, "load_source_model", load)
    original_calibrate = trainer._calibrate_fidelity_budget

    def calibrate(*args, **kwargs):
        cpu_rng = torch.get_rng_state().clone()
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        result = original_calibrate(*args, **kwargs)
        assert torch.equal(torch.get_rng_state(), cpu_rng)
        assert random.getstate() == python_rng
        restored_numpy = np.random.get_state()
        assert restored_numpy[0] == numpy_rng[0]
        assert np.array_equal(restored_numpy[1], numpy_rng[1])
        assert restored_numpy[2:] == numpy_rng[2:]
        return result

    monkeypatch.setattr(trainer, "_calibrate_fidelity_budget", calibrate)
    original_objective = trainer._coherence_objective
    ordinary = []

    def objective(model, batch, *args, **kwargs):
        calls = []
        reconstruction = trainer.differentiable_reconstruction

        def tracked(candidate, candidate_batch, **options):
            assert candidate is model
            assert candidate_batch.target_fields is None
            calls.append(candidate)
            return reconstruction(candidate, candidate_batch, **options)

        with monkeypatch.context() as context:
            context.setattr(trainer, "differentiable_reconstruction", tracked)
            result = original_objective(model, batch, *args, **kwargs)
        endpoint = kwargs.get("endpoint_context")
        if endpoint is not None and torch.is_grad_enabled():
            record = {"reconstructions": len(calls), "backwards": 0}

            def backward(gradient):
                record["backwards"] += 1
                return gradient

            endpoint["prediction"].register_hook(backward)
            ordinary.append(record)
        return result

    monkeypatch.setattr(trainer, "_coherence_objective", objective)
    child = run_post_training(post, case_dir=case_dir, max_steps=1)
    first = load_project_checkpoint(child / "checkpoints/last.pt")
    reference = json.loads((child / "artifacts/fidelity_budget_reference.json").read_text())
    assert reference["effective_draws"] == (1 if deterministic else 2)
    assert first["family_scalarizer"]["epochs_completed"] == 1
    run_post_training(post, case_dir=case_dir, resume=child, max_steps=1)
    second = load_project_checkpoint(child / "checkpoints/last.pt")
    assert first["global_step"] == 1 and second["global_step"] == 2
    assert first["external_policy"]["reference"] == second["external_policy"]["reference"]
    assert second["family_scalarizer"]["epochs_completed"] == 2
    assert all(float(state["step"]) == 2 for state in second["optimizer"]["state"].values())
    assert any(not torch.equal(first["model"][key], value)
               for key, value in second["model"].items())
    assert native == 0
    assert ordinary == [{"reconstructions": 1, "backwards": 1}] * 2
    rows = [json.loads(line) for line in (child / "metrics/history.jsonl").read_text().splitlines()]
    assert all(row["native_data_loss"] is None for row in rows)
    manifest = json.loads((child / "run_manifest.json").read_text())
    assert manifest["best_metric"]["name"] == "fixed_validation_reconstruction_mse"
    evaluation = json.loads((child / "evaluation/after.json").read_text())
    assert "validation_endpoint_source_ratios" in evaluation


def test_external_direct_endpoint_contract_uses_declared_observation_processing():
    from helpers.physical import sample

    from phycoflow_reconstruction.contracts import DataSpec
    from phycoflow_reconstruction.data.sensor_protocols import (
        SensorProtocol,
        build_observation_batch,
    )
    from phycoflow_reconstruction.models import build_model
    from phycoflow_reconstruction.training.rollout_contract import verify_rollout_contract

    field = sample()
    batch = build_observation_batch([field], SensorProtocol(field_counts={"phi": 4}, seed=3))
    batch = replace(batch, target_fields=None)
    model = build_model(
        {"name": "senseiver", "width": 16, "num_latents": 4, "heads": 2, "depth": 1},
        DataSpec(field.field_names, ("1",) * 3, 3, (8, 8)),
    ).eval()
    config = {
        "rollout": {"steps": 1, "solver": "euler"},
        "evaluation": {"generation_steps": 1},
        "observation_consistency": {"mode": "hard", "final_clamp": True},
        "optimization": {"multitask": {"external_policy": {"method": "rectify"}}},
    }
    assert verify_rollout_contract(model, batch, config)["passed"]
    # Preserve the old contract: raw direct inference differs from a clamped
    # training endpoint if its configuration has not selected the new route.
    config["optimization"] = {}
    with pytest.raises(ValueError, match="training/inference rollout mismatch"):
        verify_rollout_contract(model, batch, config)
