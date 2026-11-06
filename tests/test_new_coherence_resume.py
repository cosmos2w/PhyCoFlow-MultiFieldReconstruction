"""End-to-end checkpoint continuity for the three new coherence definitions."""

import copy
import json
import random
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml
from helpers.coherence import _base_config, _post_config, _write_fixture

from phycoflow_reconstruction.config.validate import validate_config
from phycoflow_reconstruction.training.base_training import run_base_training
from phycoflow_reconstruction.training.post_training import run_post_training
from phycoflow_reconstruction.training.posttrain_rescore import restore_evaluation_panel
from phycoflow_reconstruction.training.run_store import load_project_checkpoint


def _assert_exact(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype and torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            _assert_exact(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for item, other in zip(left, right):
            _assert_exact(item, other)
    else:
        assert left == right


def _new_families(weighted=False):
    filename = "state_mixed_weighted.yaml" if weighted else "state_mixed_mean.yaml"
    fragment = Path(__file__).resolve().parents[1] / "configs/coherence" / filename
    state = yaml.safe_load(fragment.read_text())["coherence"]["families"]["global_distribution"]
    state["fields"] = ["u", "v"]
    families = {
        "global_distribution": state,
        "multiscale_spectrum": {
            "fields": ["u", "v"],
            "units": "model_units",
            "geometry": {"operator": "fem_p1"},
            "bands": {"count": 2, "min_wavelength": 1.5},
            "calibration": {"samples": 2, "seed": 1729, "field_unit_floor": 0.01},
            "numerics": {"dtype": "float64"},
            "objective": {
                "kind": "signed_band_energy_score",
                "score_definition_version": "signed_spectral_mean_augmented_v4",
                "mean_weight": 1.0,
                "conditional_draws": 2,
                "almost_fair_alpha": 1.0,
                "joint_energy_score_weight": 0.25,
                "coherence_crps_weight": 0.0,
            },
        },
        "topology": {
            "enabled": True,
            "strategy": "cubical_persistence",
            "target_use": "paired_supervised",
            "units": "model_units",
            "fields": ["u", "v"],
            "geometry": {"grid_shape": [4, 4], "neighbors": 1, "periodic": False},
            "filtration": {
                "dimensions": [0, 1],
                "directions": ["sublevel", "superlevel"],
                "smoothing_sigma": 0.0,
            },
            "persistence": {
                "definition": "finite_primary_v1",
                "projections": 2,
                "scale_floor": 1e-4,
                "calibration": {"finite_floor": 1e-5, "essential_floor": 1e-3},
            },
            "components": {
                "self": {"enabled": True},
                "mutual": {"enabled": True, "groups": [["u", "v"]], "lines": 3},
            },
        },
    }
    if weighted:
        families["topology"]["geometry"].update(
            grid_shape=[3, 3], raster_mapping="physical_cell_overlap_v1",
        )
    return families


@pytest.mark.parametrize("weighted", [False, True], ids=["legacy-state", "weighted-physical-state"])
def test_new_definitions_config_lagrangian_resume_and_frozen_state(tmp_path, monkeypatch, request, weighted):
    pytest.importorskip("torchjd")
    pytest.importorskip("cooper")
    pytest.importorskip("gudhi")
    import phycoflow_reconstruction.training.post_training as trainer
    from phycoflow_reconstruction.data import training_batches

    # Keep the real compact HDF5 dataset and batching while avoiding worker
    # socket IPC, which is unavailable in some local sandboxes.
    data_loader = training_batches.DataLoader

    def in_process_loader(*args, **kwargs):
        kwargs.update(num_workers=0, persistent_workers=False)
        kwargs.pop("prefetch_factor", None)
        return data_loader(*args, **kwargs)

    monkeypatch.setattr(training_batches, "DataLoader", in_process_loader)
    previous_threads = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(previous_threads))
    torch.set_num_threads(1)
    data_path = tmp_path / "fixture.h5"
    _write_fixture(data_path)
    if weighted:
        with h5py.File(data_path, "r+") as handle:
            coords = handle["coordinates"][:]
            coords[:, 0, 0, 0] = np.tile(np.array([0., .5, 2., 5.]), 4)
            coords[:, 0, 0, 1] = np.repeat(np.array([0., .2, 1., 3.]), 4)
            handle["coordinates"][:] = coords
    source = run_base_training(_base_config(data_path), case_dir=tmp_path / "source")
    from phycoflow_reconstruction.models.flows.pointcloud.adapters.pointcloud_ffm_adapter import (
        PointCloudFFM,
    )

    def forbid_native_loss(*args, **kwargs):
        raise AssertionError("external fidelity policy called the native training loss")

    monkeypatch.setattr(PointCloudFFM, "training_loss", forbid_native_loss)
    config = _post_config(data_path, source)
    config["dataset"]["grid_shape"] = [4, 4]
    if weighted:
        config["dataset"]["physical_measure"] = "native_rectilinear_area"
    config["objectives"]["data_retention"] = {"enabled": False, "weight": 0.0}
    config["optimization"].update(
        epochs=2,
        steps_per_epoch=1,
        sampling="full_pass",
        multitask={
            "family_scalarization": {"method": "none"},
            "outer_aggregation": None,
            "coherence_aggregation": {"method": "torchjd_config"},
            "external_policy": {
                "method": "lagrangian",
                "vector_reduction": "sum",
                "lagrangian": {"penalty_interval_updates": 1},
            },
            "fidelity_budget": {"calibration_batches": 2, "calibration_draws": 2},
        },
    )
    config["coherence"].update(
        compute_budget={"batch_size": 2, "point_count": 16, "query_policy": "fixed_shared"},
        families=_new_families(weighted=weighted),
    )
    config["checkpointing"] = {"every_epochs": 1, "selection_metric": "reconstruction_mse"}
    config["evaluation"].update(max_samples=2, query_points=16, preview={"enabled": False})
    config["runtime"].update(progress=False, plot_every_steps=1000)
    validate_config(config)  # auto is the supported CPU data strategy; legacy is not accepted.

    batches, draw_streams, callbacks = [], [], []
    in_tracked_update = False
    original = trainer._coherence_objective
    original_reconstruction = trainer.differentiable_reconstruction

    def track_reconstruction(model, batch, *args, **kwargs):
        value = original_reconstruction(model, batch, *args, **kwargs)
        if in_tracked_update:
            draw_streams.append((kwargs["generator"].initial_seed(), value.detach().clone()))
        return value

    def track(model, batch, *args, **kwargs):
        nonlocal in_tracked_update
        tracking = kwargs.get("endpoint_context") is not None and torch.is_grad_enabled()
        if tracking:
            batches.append(tuple(batch.sample_ids))
        in_tracked_update = tracking
        try:
            return original(model, batch, *args, **kwargs)
        finally:
            in_tracked_update = False

    def consume_randomness(run_dir, saved_path, step):
        assert saved_path.is_file() and run_dir.is_dir()
        callbacks.append(step)
        torch.rand(20)
        np.random.random(20)
        random.random()

    monkeypatch.setattr(trainer, "_coherence_objective", track)
    monkeypatch.setattr(trainer, "differentiable_reconstruction", track_reconstruction)
    child = run_post_training(
        config, case_dir=tmp_path / "resumed", max_steps=1,
        checkpoint_callback=consume_randomness,
    )
    checkpoint_path = child / "checkpoints/last.pt"
    original_bytes = checkpoint_path.read_bytes()
    first = load_project_checkpoint(checkpoint_path)
    assert first["global_step"] == first["external_policy"]["completed_updates"] == 1
    assert first["samples_seen"] == 2 and first["dataset_passes"] == 1
    assert set(first["family_states"]) == set(config["coherence"]["families"])
    state_name = "global_distribution"
    bank_key = "components_by_key.cross_joint_topk_swd.directions"
    assert first["family_states"][state_name][bank_key].shape == (32, 2)
    assert first["family_states"]["multiscale_spectrum"]["piece_mean_scale"].numel() == 2
    assert any("leaf_scales" in key for key in first["family_states"]["topology"])
    shape_artifact = torch.load(child / "artifacts/topology_family.pt", weights_only=True)
    assert shape_artifact["calibration"]["provenance"]["split"] == "train"
    if weighted:
        state_artifact = torch.load(child / "artifacts/global_distribution_family.pt", weights_only=True)
        assert state_artifact["measure"]["metadata"]["provider"] == "native_rectilinear_area"
        assert state_artifact["measure_support"]["weights"].shape == (16,)
        assert shape_artifact["geometry_diagnostics"]["mapping"] == "physical_cell_overlap_v1"

    panel = restore_evaluation_panel(child, evaluation_overrides={}, device_name="cpu")
    assert panel.load_checkpoint("last") == 1
    history_path = child / "metrics/history.jsonl"
    history_before = history_path.read_bytes()

    def forbid_update(*args, **kwargs):
        raise AssertionError("invalid checkpoint attempted an optimizer update")

    def rejects(tampered, message):
        torch.save(tampered, checkpoint_path)
        corrupt_bytes = checkpoint_path.read_bytes()
        with pytest.raises(ValueError, match=message):
            panel.load_checkpoint("last")
        with monkeypatch.context() as context:
            context.setattr(torch.optim.AdamW, "step", forbid_update)
            with pytest.raises(ValueError, match=message):
                run_post_training(config, case_dir=tmp_path / "resumed", resume=child, max_steps=1)
        assert checkpoint_path.read_bytes() == corrupt_bytes
        assert history_path.read_bytes() == history_before
        checkpoint_path.write_bytes(original_bytes)

    for name in config["coherence"]["families"]:
        missing = copy.deepcopy(first)
        del missing["family_states"][name]
        rejects(missing, "missing frozen new-family state")
    changed = copy.deepcopy(first)
    changed["family_states"][state_name][bank_key][0, 0] += 1.0
    rejects(changed, "differs from the frozen family artifact")

    run_post_training(
        config, case_dir=tmp_path / "resumed", resume=child, max_steps=1,
        checkpoint_callback=consume_randomness,
    )
    resumed = load_project_checkpoint(checkpoint_path)
    resumed_batches = batches.copy()
    resumed_draw_streams = draw_streams.copy()
    batches.clear()
    draw_streams.clear()
    uninterrupted_dir = run_post_training(config, case_dir=tmp_path / "uninterrupted", max_steps=2)
    uninterrupted = load_project_checkpoint(uninterrupted_dir / "checkpoints/last.pt")
    assert resumed["global_step"] == uninterrupted["global_step"] == 2
    assert resumed["external_policy"]["completed_updates"] == 2
    assert resumed["optimizer"]["state"]
    assert all(float(value["step"]) == 2 for value in resumed["optimizer"]["state"].values())
    assert resumed["samples_seen"] == uninterrupted["samples_seen"] == 4
    assert resumed["dataset_passes"] == uninterrupted["dataset_passes"] == 2
    assert len(resumed_batches) == len(batches) == 2
    assert resumed_batches == batches
    assert callbacks == [1, 2]
    assert len(resumed_draw_streams) == len(draw_streams) == 4
    _assert_exact(resumed_draw_streams, draw_streams)
    assert resumed_draw_streams[0][0] != resumed_draw_streams[1][0]
    assert resumed_draw_streams[2][0] != resumed_draw_streams[3][0]
    assert all(set(batch) == {"train:0", "train:1"} for batch in batches)
    for key in ("model", "optimizer", "rng_state", "family_states", "external_policy"):
        _assert_exact(resumed[key], uninterrupted[key])
    _assert_exact(first["family_states"], resumed["family_states"])
    for name in config["coherence"]["families"]:
        resumed_artifact = torch.load(child / f"artifacts/{name}_family.pt", weights_only=True)
        continuous_artifact = torch.load(
            uninterrupted_dir / f"artifacts/{name}_family.pt", weights_only=True
        )
        _assert_exact(resumed_artifact, continuous_artifact)
    rows = [json.loads(line) for line in history_path.read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2]
    assert all(
        "multiscale_spectrum.joint_spectral_mean_energy_score.conditional" in row for row in rows
    )
    evaluation = json.loads((child / "evaluation/after.json").read_text())
    families = evaluation["coherence"]["families"]
    assert families["global_distribution"]["diagnostics"]["score_definition"] == (
        "marginal_mixed_weighted_v3" if weighted else "marginal_mixed_mean_v2"
    )
    assert (
        families["multiscale_spectrum"]["diagnostics"]["signed_band"]["score_definition_version"]
        == "signed_spectral_mean_augmented_v4"
    )
    assert families["topology"]["diagnostics"]["definition"] == "finite_primary_v1"
    assert families["topology"]["diagnostics"]["scalar_filtrations"] == 16
