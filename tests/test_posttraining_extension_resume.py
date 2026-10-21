"""Exercise warmup and fidelity-policy recovery across an epoch extension."""

import json

import pytest
import torch
import yaml
from helpers.coherence import _base_config, _post_config, _write_fixture
from helpers.multifamily import _global_config

from phycoflow_reconstruction.training.base_training import run_base_training
from phycoflow_reconstruction.training.post_training import run_post_training
from phycoflow_reconstruction.training.run_store import RunStore, load_project_checkpoint


def test_posttraining_epoch_extension_resumes_warmup_and_fidelity_policy(
    tmp_path, monkeypatch
):
    pytest.importorskip("cooper")
    import phycoflow_reconstruction.training.post_training as trainer

    path = tmp_path / "fixture.h5"
    _write_fixture(path)
    base = _base_config(path)
    case_dir = tmp_path / "case"
    source = run_base_training(base, case_dir=case_dir)
    post = _post_config(path, source)
    post["optimization"].update(
        epochs=1,
        steps_per_epoch=1,
        sampling="full_pass",
        lr=1.0e-3,
        lr_warmup={"enabled": True, "start_lr": 1.0e-4, "updates": 4},
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
        "mode": "initial_grad_norm",
        "calibration_batches": 2,
    }
    post["checkpointing"] = {"every_epochs": 1, "selection_metric": "reconstruction_mse"}
    post["evaluation"]["preview"] = {"enabled": False}
    post["runtime"]["progress"] = False

    native_calls = 0
    original_load = trainer.load_source_model

    def load_without_native_loss(*args, **kwargs):
        nonlocal native_calls
        model, dataset, metadata = original_load(*args, **kwargs)

        def forbidden(*args, **kwargs):
            nonlocal native_calls
            native_calls += 1
            raise AssertionError("external path replayed native training loss")

        model.training_loss = forbidden
        return model, dataset, metadata

    monkeypatch.setattr(trainer, "load_source_model", load_without_native_loss)
    original_objective = trainer._coherence_objective
    shared_updates = []

    def track_shared_update(model, batch, *args, **kwargs):
        reconstruction_calls = []
        reconstruction = trainer.differentiable_reconstruction

        def tracked_reconstruction(candidate, candidate_batch, **options):
            assert candidate is model
            assert candidate_batch.target_fields is None
            reconstruction_calls.append(candidate)
            return reconstruction(candidate, candidate_batch, **options)

        with monkeypatch.context() as context:
            context.setattr(trainer, "differentiable_reconstruction", tracked_reconstruction)
            result = original_objective(model, batch, *args, **kwargs)
        endpoint = kwargs.get("endpoint_context")
        if endpoint is not None and torch.is_grad_enabled():
            update = {"reconstructions": len(reconstruction_calls), "backwards": 0}

            def count_backward(gradient):
                update["backwards"] += 1
                return gradient

            endpoint["prediction"].register_hook(count_backward)
            shared_updates.append(update)
        return result

    monkeypatch.setattr(trainer, "_coherence_objective", track_shared_update)

    child = run_post_training(post, case_dir=case_dir)
    first = load_project_checkpoint(child / "checkpoints/last.pt")
    reference_path = child / "artifacts/fidelity_budget_reference.json"
    reference_bytes = reference_path.read_bytes()
    reference_state = first["external_policy"]["reference"]
    assert first["global_step"] == 1
    assert first["external_policy"]["completed_updates"] == 1
    assert all(float(state["step"]) == 1 for state in first["optimizer"]["state"].values())
    first_lr_rows = [
        json.loads(line)
        for line in (child / "metrics/lr_warmup_updates.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in first_lr_rows] == [1]
    assert first_lr_rows[0]["optimizer_lr"] == pytest.approx(1.0e-4)

    RunStore.extend_epochs(child, 2)
    updated_config = yaml.safe_load((child / "resolved_config.yaml").read_text())
    assert updated_config["optimization"]["epochs"] == 2
    resumed_child = run_post_training(
        updated_config,
        case_dir=case_dir,
        resume=child,
        max_steps=1,
    )

    second = load_project_checkpoint(resumed_child / "checkpoints/last.pt")
    assert resumed_child == child
    assert second["global_step"] == 2
    assert second["external_policy"]["completed_updates"] == 2
    assert second["external_policy"]["reference"] == reference_state
    assert reference_path.read_bytes() == reference_bytes
    assert all(float(state["step"]) == 2 for state in second["optimizer"]["state"].values())
    lr_rows = [
        json.loads(line)
        for line in (child / "metrics/lr_warmup_updates.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in lr_rows] == [1, 2]
    assert [row["optimizer_lr"] for row in lr_rows] == pytest.approx([1.0e-4, 4.0e-4])
    history = [
        json.loads(line) for line in (child / "metrics/history.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in history] == [1, 2]
    assert all(row["native_data_loss"] is None for row in history)
    assert native_calls == 0
    assert shared_updates == [{"reconstructions": 1, "backwards": 1}] * 2
