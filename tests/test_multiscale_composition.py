"""All three families share conditional draws under the current external policy."""

import json

import pytest
import torch
from helpers.coherence import _base_config, _post_config, _write_fixture
from helpers.multifamily import _global_config

from phycoflow_reconstruction.config.validate import validate_config
from phycoflow_reconstruction.training.base_training import run_base_training
from phycoflow_reconstruction.training.post_training import run_post_training
from phycoflow_reconstruction.training.run_store import load_project_checkpoint


@pytest.mark.parametrize(
    ("method", "penalty_interval", "masked"),
    [("lagrangian", 1, False), ("rectify", 1, False), ("lagrangian", 1, True)],
)
def test_spectral_composition_shared_draws_and_recovery(
    tmp_path, monkeypatch, method, penalty_interval, masked
):
    pytest.importorskip("torchjd")
    if method == "lagrangian":
        pytest.importorskip("cooper")
    import phycoflow_reconstruction.training.post_training as trainer

    path = tmp_path / "fixture.h5"
    _write_fixture(path)
    if masked:
        import h5py
        import numpy as np

        rows, columns = np.meshgrid(np.arange(4), np.arange(5), indexing="ij")
        keep = np.ones((4, 5), dtype=bool)
        keep[[0, 1, 2, 3], [1, 2, 2, 3]] = False
        rc = np.stack((rows[keep], columns[keep]), -1)
        coords = np.stack(
            (rc[:, 1] + 0.03 * rc[:, 0] ** 2, rc[:, 0] + 0.02 * rc[:, 1] ** 2, np.zeros(16)), -1
        )
        with h5py.File(path, "r+") as handle:
            handle["coordinates"][:] = coords[:, None, None].astype("float32")
            handle.create_dataset("auxiliary/native_grid_indices", data=rc)
    case_dir = tmp_path / "case"
    source = run_base_training(_base_config(path), case_dir=case_dir)
    config = _post_config(path, source)
    config["dataset"]["grid_shape"] = [4, 4]
    config["objectives"]["data_retention"] = {"enabled": False, "weight": 0.0}
    config["optimization"].update(
        epochs=2 if method == "lagrangian" else 1,
        steps_per_epoch=1,
        sampling="full_pass",
        multitask={
            "family_scalarization": {"method": "none"},
            "outer_aggregation": None,
            "coherence_aggregation": {"method": "torchjd_config"},
            "external_policy": {"method": method, "vector_reduction": "sum"},
            "fidelity_budget": {"calibration_batches": 2, "calibration_draws": 2},
        },
    )
    if method == "lagrangian":
        config["optimization"]["multitask"]["external_policy"]["lagrangian"] = {
            "penalty_interval_updates": penalty_interval,
        }
    config["coherence"].update(
        compute_budget={"batch_size": 2, "point_count": 16, "query_policy": "fixed_shared"},
        families={
            "global_distribution": _global_config(),
            "multiscale_spectrum": {
                "fields": ["u", "v"],
                "geometry": {"operator": "fem_p1"},
                "bands": {"count": 2, "min_wavelength": 1.5},
                "calibration": {"samples": 2},
                "objective": {
                    "kind": "signed_band_energy_score",
                    "conditional_draws": 2,
                    "almost_fair_alpha": 1.0,
                    "joint_energy_score_weight": 0.25,
                },
            },
            "topology": {
                "enabled": True,
                "strategy": "cubical_persistence",
                "target_use": "paired_supervised",
                "units": "model_units",
                "fields": ["u", "v"],
                "geometry": {
                    "grid_shape": [4, 4],
                    "neighbors": 1,
                    "antialias_downsample": True,
                    "periodic": False,
                },
                "filtration": {
                    "dimensions": [0, 1],
                    "directions": ["sublevel", "superlevel"],
                    "smoothing_sigma": 0.0,
                },
                "persistence": {"projections": 2},
                "components": {
                    "self": {"enabled": True},
                    "mutual": {"enabled": True, "groups": [["u", "v"]], "lines": 1},
                },
            },
        },
    )
    if masked:
        families = config["coherence"]["families"]
        families["topology"]["geometry"].update(
            grid_shape=[4, 5],
            native_masked=True,
            structured_indices="native_grid_indices",
            antialias_downsample=False,
        )
        families["multiscale_spectrum"]["geometry"].update(
            triangulation="structured_indices",
            structured_indices="native_grid_indices",
            connectivity="components",
        )
    config["checkpointing"] = {"every_epochs": 1, "selection_metric": "reconstruction_mse"}
    config["evaluation"].update(max_samples=2, query_points=16, preview={"enabled": False})
    config["runtime"].update(progress=False, plot_every_steps=1000)
    validate_config(config)
    original_load = trainer.load_source_model
    original_objective = trainer._coherence_objective
    records = []

    def load(*args, **kwargs):
        model, dataset, metadata = original_load(*args, **kwargs)

        def forbidden(*args, **kwargs):
            raise AssertionError("internal ConFIG replayed native loss")

        model.training_loss = forbidden
        return model, dataset, metadata

    def objective(model, *args, **kwargs):
        calls = []
        reconstruct = trainer.differentiable_reconstruction

        def tracked(candidate, model_batch, **options):
            assert candidate is model and model_batch.target_fields is None
            calls.append(candidate)
            return reconstruct(candidate, model_batch, **options)

        with monkeypatch.context() as context:
            context.setattr(trainer, "differentiable_reconstruction", tracked)
            result = original_objective(model, *args, **kwargs)
        endpoint = kwargs.get("endpoint_context")
        if endpoint is not None and torch.is_grad_enabled():
            record = {"reconstructions": len(calls), "traversals": 0}

            def backward(gradient):
                record["traversals"] += 1
                return gradient

            endpoint["prediction"].register_hook(backward)
            records.append(record)
        return result

    monkeypatch.setattr(trainer, "load_source_model", load)
    monkeypatch.setattr(trainer, "_coherence_objective", objective)
    child = run_post_training(config, case_dir=case_dir, max_steps=1)
    first = load_project_checkpoint(child / "checkpoints/last.pt")
    if method == "lagrangian":
        run_post_training(config, case_dir=case_dir, resume=child, max_steps=1)
        restored = load_project_checkpoint(child / "checkpoints/last.pt")
        assert restored["global_step"] == 2
        assert restored["external_policy"]["reference"] == first["external_policy"]["reference"]
        assert restored["external_policy"]["completed_updates"] == 2
        assert all(float(state["step"]) == 2 for state in restored["optimizer"]["state"].values())
    artifact = torch.load(child / "artifacts/multiscale_spectrum_family.pt", weights_only=True)
    assert artifact["calibration"]["split"] == "train"
    assert artifact["calibration"]["states"] == 2
    after = json.loads((child / "evaluation/after.json").read_text())
    assert set(after["coherence"]["families"]) == {
        "global_distribution",
        "multiscale_spectrum",
        "topology",
    }
    rows = [json.loads(line) for line in (child / "metrics/history.jsonl").read_text().splitlines()]
    assert len(records) == len(rows)
    for record, row in zip(records, rows):
        assert row["native_data_loss"] is None and row["total"] is None
        assert row["total_semantics"] == "parameter_direction_without_scalar_objective"
        assert row["family_scalarization"]["method"] == "none"
        assert "multiscale_spectrum" in row["family_scalarization"]["raw_losses"]
        assert "multiscale_spectrum.joint_spectral_energy_score.conditional" in row
        assert record["reconstructions"] == 2
        if method == "lagrangian":
            assert row["fidelity_penalty_active"] == (row["step"] % penalty_interval == 0)
        if row.get("phase") == "repair":
            assert record["traversals"] == 1
        else:
            assert 3 <= record["traversals"] <= 4
