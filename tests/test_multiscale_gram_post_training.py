"""Synthetic CPU integration for the Gram candidate and shared SOURCE reference."""

import copy
import json

import pytest
import torch
from helpers.coherence import _base_config, _post_config, _write_fixture
from helpers.multifamily import _global_config

from phycoflow_reconstruction.coherence.families.multiscale_gram import (
    ANCHOR_PATH,
    MUTUAL_PATH,
    SELF_PATH,
)
from phycoflow_reconstruction.config.validate import validate_config
from phycoflow_reconstruction.data.factory import open_field_dataset
from phycoflow_reconstruction.data.manifest import dataset_fingerprint
from phycoflow_reconstruction.training.base_training import run_base_training
from phycoflow_reconstruction.training.fidelity_reference import (
    create_shared_reference,
    serialize_shared_reference,
)
from phycoflow_reconstruction.training.gradient_balance import _flat_gradient
from phycoflow_reconstruction.training.post_training import run_post_training
from phycoflow_reconstruction.training.run_store import file_sha256, load_project_checkpoint
from phycoflow_reconstruction.training.source import source_hashes


def _matched_config(path, source):
    config = _post_config(path, source)
    config["dataset"]["grid_shape"] = [4, 4]
    config["objectives"]["data_retention"] = {"enabled": False, "weight": 0.0}
    config["optimization"].update(
        epochs=2, steps_per_epoch=1, sampling="full_pass",
        multitask={
            "family_scalarization": {"method": "none"},
            "outer_aggregation": None,
            "coherence_aggregation": {"method": "torchjd_config"},
            "external_policy": {"method": "lagrangian", "vector_reduction": "sum"},
            "fidelity_budget": {"calibration_batches": 2, "calibration_draws": 2},
        },
    )
    config["coherence"].update(
        compute_budget={"batch_size": 2, "point_count": 16, "query_policy": "fixed_shared"},
        families={
            "global_distribution": _global_config(),
            "multiscale_spectrum": {
                "fields": ["u", "v"],
                "geometry": {"operator": "fem_p1"},
                "bands": {"count": 2, "min_wavelength": 1.5},
                "calibration": {"samples": 2, "field_unit_floor": 0.01},
                "numerics": {"dtype": "float64"},
                "objective": {
                    "kind": "signed_band_energy_score",
                    "conditional_draws": 2,
                    "almost_fair_alpha": 1.0,
                    "joint_energy_score_weight": 0.25,
                    "score_definition_version": "signed_spectral_mean_augmented_v4",
                },
            },
            "topology": {
                "enabled": True,
                "strategy": "cubical_persistence",
                "target_use": "paired_supervised",
                "units": "model_units",
                "fields": ["u", "v"],
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
    return config


def _make_shared_reference(config, baseline, destination):
    dataset = open_field_dataset(config["dataset"], split="train")
    try:
        selection = dataset.selection
        train_selection = {
            "split": selection.split,
            "strategy": selection.strategy,
            "trajectory_indices": [int(value) for value in selection.trajectory_indices],
            "frame_indices": [int(value) for value in selection.frame_indices],
            "sample_count": len(dataset),
            "training_subset_sha256": None,
        }
        scales = json.loads((baseline / "artifacts/coherence_calibration.json").read_text())["resolved_scales"]
        report = baseline / "artifacts/fidelity_budget_reference.json"
        artifact = create_shared_reference(
            report, file_sha256(report), config,
            field_names=dataset.field_names, normalizer_sha256=dataset.normalizer.digest(),
            source_hashes=source_hashes(config), dataset_fingerprint=dataset_fingerprint(dataset.path),
            train_selection=train_selection, family_scales=scales,
        )
        destination.write_bytes(serialize_shared_reference(artifact))
        return artifact
    finally:
        dataset.close()


def test_candidate_trainer_uses_shared_reference_and_restores_exact_state(tmp_path, monkeypatch):
    pytest.importorskip("torchjd")
    pytest.importorskip("cooper")
    import phycoflow_reconstruction.data.training_batches as batches
    import phycoflow_reconstruction.training.post_training as trainer

    # Run the same compact HDF5 loader synchronously for this tiny fixture;
    # multiprocessing transport sockets can be unavailable in test sandboxes.
    data_loader = batches.DataLoader

    def synchronous_loader(*args, **kwargs):
        kwargs.update(num_workers=0, persistent_workers=False, prefetch_factor=None)
        return data_loader(*args, **kwargs)

    monkeypatch.setattr(batches, "DataLoader", synchronous_loader)
    path = tmp_path / "fixture.h5"
    _write_fixture(path)
    case_dir = tmp_path / "case"
    source = run_base_training(_base_config(path), case_dir=case_dir)
    baseline_config = _matched_config(path, source)
    baseline_config["output"]["experiment_name"] = "baseline"
    validate_config(baseline_config)
    baseline = run_post_training(baseline_config, case_dir=case_dir, max_steps=1)
    shared_path = tmp_path / "shared_source_reference.json"
    shared = _make_shared_reference(baseline_config, baseline, shared_path)
    shared_hash = file_sha256(shared_path)

    config = copy.deepcopy(baseline_config)
    config["output"]["experiment_name"] = "candidate"
    weight = 0.06287622600055182
    config["coherence"]["families"]["multiscale_spectrum"]["gram_extension"] = {
        "definition": "self_mutual_gram_v1", "weight": weight,
        "std_floor": 0.001, "relative_std_floor": 0.05,
    }
    config["optimization"]["multitask"]["fidelity_budget"].update(
        reference_artifact=str(shared_path), reference_sha256=shared_hash
    )
    validate_config(config)

    original_objective = trainer._coherence_objective
    checked = []

    def verify_training_graph(model, *args, **kwargs):
        result, ids = original_objective(model, *args, **kwargs)
        family_losses = kwargs.get("family_losses")
        if torch.is_grad_enabled() and kwargs.get("endpoint_context") is not None and family_losses:
            params = [parameter for parameter in model.parameters() if parameter.requires_grad]
            total = _flat_gradient(family_losses["multiscale_spectrum"], params)
            anchor = _flat_gradient(result.component_results[ANCHOR_PATH].scalar_loss, params)
            self_grad = _flat_gradient(result.component_results[SELF_PATH].scalar_loss, params)
            mutual_grad = _flat_gradient(result.component_results[MUTUAL_PATH].scalar_loss, params)
            auxiliary = weight * (self_grad + mutual_grad) / 2
            torch.testing.assert_close(total, anchor + auxiliary, rtol=2e-5, atol=2e-7)
            assert torch.isfinite(auxiliary).all() and auxiliary.norm() > 0
            checked.append(float(auxiliary.norm()))
        return result, ids

    monkeypatch.setattr(trainer, "_coherence_objective", verify_training_graph)
    child = run_post_training(config, case_dir=case_dir, max_steps=1)
    first = load_project_checkpoint(child / "checkpoints/last.pt")
    artifact = torch.load(child / "artifacts/multiscale_spectrum_family.pt", weights_only=True)
    assert first["global_step"] == 1 and first["shared_fidelity_reference_sha256"] == shared_hash
    assert first["external_policy"]["reference"] == shared["reference"]
    assert first["family_configs"]["multiscale_spectrum"] == config["coherence"]["families"]["multiscale_spectrum"]
    assert first["family_states"]["multiscale_spectrum"].keys() == artifact["state_dict"].keys()
    for name, value in artifact["state_dict"].items():
        assert torch.equal(value, first["family_states"]["multiscale_spectrum"][name]), name

    run_post_training(config, case_dir=case_dir, resume=child, max_steps=1)
    resumed = load_project_checkpoint(child / "checkpoints/last.pt")
    assert resumed["global_step"] == 2 and resumed["shared_fidelity_reference_sha256"] == shared_hash
    assert resumed["external_policy"]["reference"] == shared["reference"]
    assert resumed["family_configs"]["multiscale_spectrum"] == first["family_configs"]["multiscale_spectrum"]
    for name, value in first["family_states"]["multiscale_spectrum"].items():
        assert torch.equal(value, resumed["family_states"]["multiscale_spectrum"][name]), name
    assert (child / "artifacts/shared_fidelity_reference.json").read_bytes() == shared_path.read_bytes()
    assert len(checked) == 2
