"""Exercise topology through the existing source/child training lifecycle."""

import json

import pytest
import torch
from helpers.coherence import _base_config, _post_config, _write_fixture
from helpers.multifamily import _cross_config, _global_config

from phycoflow_reconstruction.config.validate import validate_config
from phycoflow_reconstruction.training.base_training import run_base_training
from phycoflow_reconstruction.training.post_training import run_post_training
from phycoflow_reconstruction.training.run_store import file_sha256


def test_spatial_topology_trains_with_other_families_and_preserves_source(tmp_path):
    dataset_path = tmp_path / "fixture.h5"
    _write_fixture(dataset_path)
    case_dir = tmp_path / "case"
    source = run_base_training(_base_config(dataset_path), case_dir=case_dir)
    checkpoint = source / "checkpoints" / "last.pt"
    original_hash = file_sha256(checkpoint)
    post = _post_config(dataset_path, source)
    cross = _cross_config()
    cross["components"]["cross_frequency"]["enabled"] = False  # Fixture has two train samples.
    post["coherence"].update(
        {
            "compute_budget": {"batch_size": 2, "point_count": 16, "query_policy": "fixed_shared"},
            "families": {
                "global_distribution": _global_config(),
                "cross_spectrum": cross,
                "topology": {
                    "strategy": "spatial_self_mutual",
                    "target_use": "paired_supervised",
                    "units": "model_units",
                    "fields": ["u"],
                    "geometry": {
                        "grid_shape": [4, 4],
                        "neighbors": 1,
                        "periodic": False,
                        "antialias_downsample": True,
                    },
                    "filtration": {"smoothing_sigma": 0.0, "quantiles": [0.3, 0.5, 0.9]},
                    "anchor": {"provider": "raw", "fields": ["v"]},
                    "components": {
                        "self": {"enabled": True},
                        "anchor_self": {"enabled": True, "weight": 0.2},
                        "mutual": {"enabled": True, "carrier_field": "u", "lines": 4},
                    },
                },
            },
        }
    )
    post["evaluation"].update(max_samples=2, query_points=16)
    validate_config(post)
    child = run_post_training(post, case_dir=case_dir)
    assert file_sha256(checkpoint) == original_hash
    assert (child / "checkpoints" / "last.pt").is_file()
    manifest = json.loads((child / "run_manifest.json").read_text())
    assert manifest["source_immutable_verified"] is True
    history = (child / "metrics" / "history.jsonl").read_text()
    for term in ("self.region", "self.connectivity", "anchor_self.spatial", "mutual.spatial"):
        assert f"topology.{term}" in history
    for family in ("global_distribution", "cross_spectrum", "topology"):
        assert family in history
    evaluation = (child / "evaluation" / "after.json").read_text()
    assert "topology.self.h0_nmae" in evaluation
    assert "line_error_mass" in evaluation


@pytest.mark.parametrize("constrained", [False, "topology", "endpoint"])
@pytest.mark.parametrize("spatial", [False, True])
def test_persistence_retention_panel_selection_and_resume(tmp_path, constrained, spatial):
    pytest.importorskip("gudhi")
    from phycoflow_reconstruction.training.run_store import load_project_checkpoint

    dataset_path = tmp_path / "fixture.h5"
    _write_fixture(dataset_path)
    source = run_base_training(_base_config(dataset_path), case_dir=tmp_path / "case")
    post = _post_config(dataset_path, source)
    post["coherence"]["families"] = {
        "topology": {
            "strategy": "cubical_persistence",
            "target_use": "paired_supervised",
            "units": "model_units",
            "fields": ["u", "v"],
            "geometry": {"grid_shape": [4, 4], "periodic": True, "neighbors": 1},
            "filtration": {"smoothing_sigma": 0.0},
            "components": {
                "self": {"enabled": True},
                "mutual": {"enabled": True, "groups": [["u", "v"]], "lines": 2},
            },
        }
    }
    if spatial:
        post["coherence"]["families"]["topology"]["persistence"] = {
            "distance": "spatial_wasserstein",
            "spatial_weight": 1.0,
            "spatial_mode": "additive",
        }
    post["coherence"]["compute_budget"] = {
        "batch_size": 2,
        "point_count": 16,
        "query_policy": "fixed_shared",
    }
    post["objectives"]["endpoint"] = {"enabled": True, "weight": 1.0}
    post["dataset"]["grid_shape"] = [4, 4]
    # This fixture's native one-step sampler clamps only its final endpoint.
    post["observation_consistency"] = {"mode": "hard", "final_clamp": True}
    post["objectives"]["source_anchor"] = {"enabled": True, "weight": 0.2}
    post["optimization"].update(epochs=4, batch_size=2, model_mode="eval")
    post["evaluation"].update(
        max_samples=2,
        query_points=16,
        sample_selection="uniform",
        native_topology=True,
        preview={"enabled": False},
    )
    post["checkpointing"] = {"every_epochs": 2, "selection_metric": "topology_with_fidelity"}
    post["posttrain_fidelity"] = {
        "max_relative_mse_increase": 0.05,
        "max_relative_field_mse_increase": 0.05,
        "max_relative_native_topology_increase": 0.0,
    }
    if constrained:
        post["optimization"].update(
            gradient_balance="component_constrained",
            component_constraints={"calibration_batches": 2, "proposal_objective": constrained},
        )
        for name in ("data_retention", "endpoint", "source_anchor"):
            post["objectives"][name]["enabled"] = False
        post["coherence"]["compute_budget"]["batch_size"] = 2
        post["coherence"]["families"]["topology"]["geometry"]["antialias_downsample"] = True
        post["coherence"]["schedule"].update(every_n_steps=1, start_epoch=1, weight_warmup_epochs=0)
        post["checkpointing"].update(every_epochs=1, validation_every_epochs=4)
    validate_config(post)
    child = run_post_training(post, case_dir=tmp_path / "case", max_steps=2)
    selected = json.loads((child / "evaluation/selected.json").read_text())
    assert selected["eligible"]
    assert selected["metrics"]["native_topology"]["grid_shape"] == [4, 4]
    resumed = run_post_training(post, case_dir=tmp_path / "case", max_steps=2, resume=child)
    whole = run_post_training(post, case_dir=tmp_path / "case", max_steps=4)
    a = load_project_checkpoint(resumed / "checkpoints/last.pt")
    b = load_project_checkpoint(whole / "checkpoints/last.pt")
    assert a["global_step"] == b["global_step"] == 4
    for name, value in a["model"].items():
        torch.testing.assert_close(value, b["model"][name], rtol=0, atol=0)
    # The final checkpoint may survive preemption while final reporting does
    # not. Recovery must finish reports without taking another optimizer step.
    history_before = (child / "metrics/history.jsonl").read_bytes()
    (child / "evaluation/after.json").unlink()
    (child / "status.json").write_text(json.dumps({"status": "running", "global_step": 4}))
    run_post_training(post, case_dir=tmp_path / "case", max_steps=0, resume=child)
    recovered = load_project_checkpoint(child / "checkpoints/last.pt")
    assert recovered["global_step"] == 4
    for name, value in a["model"].items():
        torch.testing.assert_close(value, recovered["model"][name], rtol=0, atol=0)
    assert (child / "metrics/history.jsonl").read_bytes() == history_before
    assert (child / "evaluation/after.json").is_file()
    recovered_status = json.loads((child / "status.json").read_text())
    assert recovered_status["status"] == "completed"
    assert recovered_status["segment_steps"] == 0
    history = [
        json.loads(line) for line in (child / "metrics/history.jsonl").read_text().splitlines()
    ]
    if constrained:
        assert a["component_controller"] == b["component_controller"]
        assert a["component_controller"]["attempted"] == 4
        assert all("update_accepted" in row for row in history)
        topology = selected["metrics"]["coherence"]["families"]["topology"]
        assert topology["selection_score"] == pytest.approx(
            sum(
                value["weighted_contribution"]
                for value in topology["selection_components"].values()
            )
        )
        assert topology["total"] == selected["metrics"]["native_topology"]["total"]
        audit = [
            json.loads(line)
            for line in (child / "metrics/constraint_updates.jsonl").read_text().splitlines()
        ]
        assert [row["step"] for row in audit] == [1, 2, 3, 4]
        for row in audit:
            if row["update_accepted"]:
                assert all(
                    value <= row["constraint_bounds"][name] + row["constraint_tolerances"][name]
                    for name, value in row["constraint_candidate"].items()
                )
        return
    assert any("endpoint_retention_loss" in row for row in history)
    assert any("source_anchor_retention_loss" in row for row in history)
    # PyTorch can choose a fused no-grad attention kernel for the frozen anchor.
    assert history[0]["source_anchor_retention_loss"] < 1e-12
    assert json.loads((child / "evaluation/rollout_contract.json").read_text())["passed"]
