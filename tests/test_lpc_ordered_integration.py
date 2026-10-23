"""Ordered objective shares one endpoint and recovers fixed references and duals."""

import json

import pytest
import torch
from helpers.coherence import _base_config, _post_config, _write_fixture
from helpers.multifamily import _cross_config, _global_config

from phycoflow_reconstruction.training.base_training import run_base_training
from phycoflow_reconstruction.training.post_training import run_post_training
from phycoflow_reconstruction.training.run_store import file_sha256, load_project_checkpoint


@pytest.mark.parametrize("deterministic", [False, True])
def test_ordered_shared_endpoint_calibration_and_recovery(tmp_path, monkeypatch, deterministic):
    pytest.importorskip("cooper")
    import phycoflow_reconstruction.training.post_training as trainer

    path = tmp_path / "fixture.h5"
    _write_fixture(path)
    base = _base_config(path)
    if deterministic:
        base["model"] = {
            "name": "senseiver", "width": 16, "num_latents": 4, "heads": 2, "depth": 1,
        }
    case = tmp_path / "case"
    source = run_base_training(base, case_dir=case)
    config = _post_config(path, source)
    config["model"] = base["model"]
    config["dataset"]["grid_shape"] = [4, 4]
    config["objectives"]["data_retention"] = {"enabled": False, "weight": 0.0}
    config["optimization"].update(
        epochs=2, steps_per_epoch=1, sampling="full_pass",
        lpc_ordered={"dual_lr": 0.01, "denominator_floor": 1e-8},
        fast_lagrangian={"topology_estimator": "full", "family_probability": 1.0,
                         "coefficient_interval": 1, "seed": 700211},
        multitask={
            "family_scalarization": {"method": "fixed_sum"},
            "outer_aggregation": None,
            "external_policy": {
                "method": "lagrangian", "vector_reduction": "sum",
                "lagrangian": {"dual_lr": 0.01, "penalty_interval_updates": 1},
            },
            "fidelity_budget": {"calibration_batches": 2, "calibration_draws": 2},
        },
    )
    cross = _cross_config()
    cross["components"]["cross_frequency"]["enabled"] = False
    config["coherence"].update(
        family_balance={"mode": "none"},
        compute_budget={"batch_size": 2, "point_count": 16, "query_policy": "fixed_shared"},
        families={
            "global_distribution": _global_config(),
            "cross_spectrum": cross,
            "topology": {
                "enabled": True, "strategy": "cubical_persistence",
                "target_use": "paired_supervised", "units": "model_units", "fields": ["u", "v"],
                "geometry": {"grid_shape": [4, 4], "neighbors": 1,
                             "antialias_downsample": True, "periodic": False},
                "filtration": {"dimensions": [0, 1], "directions": ["sublevel", "superlevel"],
                               "smoothing_sigma": 0.0},
                "persistence": {"projections": 2},
                "components": {"self": {"enabled": True},
                               "mutual": {"enabled": True, "groups": [["u", "v"]], "lines": 1}},
            },
        },
    )
    config["checkpointing"] = {"every_epochs": 1, "selection_metric": "reconstruction_mse"}
    config["evaluation"].update(max_samples=2, query_points=16, preview={"enabled": False})
    config["runtime"].update(progress=False, plot_every_steps=1000)
    original_calibrate = trainer._calibrate_fidelity_budget
    calibration_calls = []

    def calibrate(*args, **kwargs):
        # Same full SOURCE draws: collecting family means must leave C0, risks,
        # model/data RNG, and all legacy reference fields unchanged.
        before = torch.get_rng_state().clone()
        legacy_reference, _ = original_calibrate(*args, **{**kwargs, "family_means": None})
        assert torch.equal(before, torch.get_rng_state())
        reference, report = original_calibrate(*args, **kwargs)
        assert torch.equal(before, torch.get_rng_state())
        assert reference.state_dict() == legacy_reference.state_dict()
        assert set(kwargs["family_means"]) == {"global_distribution", "cross_spectrum", "topology"}
        assert sum(kwargs["family_means"].values()) == pytest.approx(reference.c0, rel=2e-7)
        calibration_calls.append(True)
        return reference, report

    monkeypatch.setattr(trainer, "_calibrate_fidelity_budget", calibrate)
    original_objective = trainer._coherence_objective
    records = []

    def objective(model, *args, **kwargs):
        calls = []
        reconstruct = trainer.differentiable_reconstruction

        def tracked(candidate, model_batch, **options):
            assert candidate is model and model_batch.target_fields is None
            calls.append(True)
            return reconstruct(candidate, model_batch, **options)

        with monkeypatch.context() as context:
            context.setattr(trainer, "differentiable_reconstruction", tracked)
            result = original_objective(model, *args, **kwargs)
        endpoint = kwargs.get("endpoint_context")
        if endpoint is not None and torch.is_grad_enabled():
            record = {"forwards": len(calls), "backwards": 0}

            def backward(gradient):
                record["backwards"] += 1
                return gradient

            endpoint["prediction"].register_hook(backward)
            records.append(record)
        return result

    monkeypatch.setattr(trainer, "_coherence_objective", objective)
    child = run_post_training(config, case_dir=case, max_steps=1)
    first = load_project_checkpoint(child / "checkpoints/last.pt")
    artifact = child / "artifacts/lpc_ordered_reference.json"
    reference_hash = file_sha256(artifact)
    log = child / "metrics/lpc_ordered_updates.jsonl"
    first_row = json.loads(log.read_text().splitlines()[0])
    # Speculative writes and torn rows must be removed before replay.
    with log.open("a") as handle:
        handle.write(json.dumps({"step": 2, "speculative": True}) + "\n{")
    run_post_training(config, case_dir=case, resume=child, max_steps=1)
    second = load_project_checkpoint(child / "checkpoints/last.pt")
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2]
    assert rows[0] == first_row and all("speculative" not in row for row in rows)
    assert calibration_calls == [True]
    assert reference_hash == file_sha256(artifact) == second["lpc_ordered_reference_sha256"]
    assert first["lpc_ordered_implementation"] == second["lpc_ordered_implementation"]
    assert first["lpc_ordered_state"]["reference"] == second["lpc_ordered_state"]["reference"]
    assert second["external_policy"]["completed_updates"] == second["global_step"] == 2
    assert second["lpc_ordered_state"]["completed_updates"] == 2
    assert records == [{"forwards": 1, "backwards": 1}] * 2
    history = [json.loads(line) for line in (child / "metrics/history.jsonl").read_text().splitlines()]
    assert all(row["total"] == row["optimized_scalar"] for row in history)
    assert all(row["coherence_loss"] != row["primary_scalar"] for row in history)
    assert all(row["native_data_loss"] is None for row in history)
    assert any(not torch.equal(first["model"][key], value) for key, value in second["model"].items())
