"""CPU-only acceptance tests for the opt-in periodic native topology audit."""

from __future__ import annotations

import copy
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch
from helpers.coherence import _post_config

from phycoflow_reconstruction.coherence.families.topology.family import TopologyFamily
from phycoflow_reconstruction.config.validate import validate_config
from phycoflow_reconstruction.contracts import DataSpec, FieldSample
from phycoflow_reconstruction.data.manifest import manifest_from_batch
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.data.sensor_protocols import SensorProtocol
from phycoflow_reconstruction.training.native_topology_audit import (
    _numeric_delta,
    _preserve_audit_state,
    _report_metrics,
    audit_due,
    audit_settings,
    build_disjoint_validation_batch,
    build_native_topology_family,
    contract_payload,
    execute_native_audit,
    initialize_native_geometry,
    paired_bar_diagnostics,
    verify_saved_family_hash,
    write_idempotent_report,
)
from phycoflow_reconstruction.training.run_store import file_sha256


def _topology_config() -> dict:
    return {
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
        "components": {
            "self": {"enabled": False},
            "mutual": {
                "enabled": True,
                "groups": [["u", "v"]],
                "line_bank_size": 4,
                "training_subset_size": 2,
                "seed": 37,
            },
        },
    }


def _audit_config(tmp_path: Path, enabled: bool = True) -> dict:
    config = _post_config(tmp_path / "unused.h5", tmp_path / "source")
    config["dataset"]["grid_shape"] = [4, 4]
    config["optimization"].update(epochs=2, batch_size=1)
    config["coherence"] = {
        "schedule": {
            "start_epoch": 1,
            "every_n_steps": 1,
            "weight_warmup_epochs": 0,
            "interval_rescale": False,
        },
        "compute_budget": {
            "batch_size": 1,
            "point_count": 16,
            "query_policy": "fixed_shared",
            "query_seed": 773,
        },
        "families": {"topology": _topology_config()},
    }
    config["evaluation"].update(
        split="validation",
        max_samples=1,
        query_points=8,
        native_topology_audit={
            "enabled": enabled,
            "every_steps": 2,
            "max_samples": 2,
            "seed": 2027,
        },
    )
    config["checkpointing"] = {
        "enabled": True,
        "every_steps": 2,
        "selection_metric": "native_validation_loss",
    }
    return config


def _grid_sample(index: int, spec: DataSpec) -> FieldSample:
    axis = torch.linspace(0.0, 1.0, 4)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    coordinates = torch.stack((xx, yy), dim=-1).reshape(16, 2)
    values = torch.stack(
        (
            xx.reshape(-1) + 0.1 * yy.reshape(-1) + index * 0.01,
            torch.sin(2 * torch.pi * xx.reshape(-1))
            + 0.3 * torch.cos(2 * torch.pi * yy.reshape(-1))
            + index * 0.02,
        ),
        dim=1,
    )
    return FieldSample(
        values=values,
        coordinates=coordinates,
        coordinates_raw=coordinates.clone(),
        time=torch.tensor([float(index)]),
        trajectory_id=f"validation_{index:03d}",
        time_index=0,
        conditions=torch.empty(0),
        field_names=spec.field_names,
        logical_shape=(4, 4),
    )


class _Dataset:
    def __init__(self, path: Path, samples: list[FieldSample], spec: DataSpec, normalizer):
        self.path = path
        self.samples = samples
        self.data_spec = spec
        self.field_names = spec.field_names
        self.normalizer = normalizer
        self.trajectory_ids = tuple(sample.trajectory_id for sample in samples)
        self._items = [(index, sample.time_index) for index, sample in enumerate(samples)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class _Store:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir

    def write_json(self, relative_path, payload):
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(dict(payload), sort_keys=True), encoding="utf-8")
        temporary.replace(path)
        return path


def _runtime(tmp_path):
    pytest.importorskip("gudhi")
    fields = ("u", "v")
    normalizer = FieldNormalizer.identity(2)
    spec = DataSpec(fields, ("1", "1"), 2, (4, 4), mesh_type="structured")
    data_path = tmp_path / "fixture.h5"
    data_path.write_bytes(b"synthetic validation fixture")
    dataset = _Dataset(data_path, [_grid_sample(i, spec) for i in range(7)], spec, normalizer)
    config = {
        "dataset": {"grid_shape": [4, 4]},
        "coherence": {
            "compute_budget": {"query_policy": "fixed_shared", "point_count": 16},
            "families": {"topology": _topology_config()},
        },
    }
    run_dir = tmp_path / "child"
    (run_dir / "artifacts").mkdir(parents=True)
    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"immutable source fixture")
    saved_family = TopologyFamily(_topology_config(), spec, normalizer)
    coordinates = torch.stack([sample.coordinates for sample in dataset.samples[:1]])
    saved_family._raster_map(coordinates)
    artifact_path = run_dir / "artifacts/topology_family.pt"
    torch.save(saved_family.state_artifact(), artifact_path)
    artifact_sha = file_sha256(artifact_path)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "stage": "post_training",
                "parent_run": str(tmp_path / "source_run"),
                "source_checkpoint": str(source_checkpoint),
                "source_hashes": {"checkpoint": file_sha256(source_checkpoint)},
                "coherence_family_state_sha256s": {"topology": artifact_sha},
            }
        ),
        encoding="utf-8",
    )
    return config, dataset, normalizer, run_dir


def test_disabled_audit_keeps_default_path_and_config_valid(tmp_path):
    config = _audit_config(tmp_path, enabled=False)
    validate_config(config)
    assert audit_settings(config) is None
    without_key = copy.deepcopy(config)
    del without_key["evaluation"]["native_topology_audit"]
    validate_config(without_key)
    assert audit_settings(without_key) is None
    assert not audit_due(2, 2, durable=False)


def test_enabled_legacy_audit_preserves_configured_ema_eval_policy(tmp_path):
    config = _audit_config(tmp_path, enabled=True)
    config["model"]["model_ema_eval"] = True
    validate_config(config)
    assert config["model"]["model_ema_eval"] is True


def test_report_metrics_preserve_per_field_diagnostics_for_all_case_fields():
    field_names = ("CH4", "CO", "T", "U_1", "p")
    candidate_fields = {name: float(index + 1) for index, name in enumerate(field_names)}
    source_fields = {name: 0.0 for name in field_names}
    candidate_report = {
        "mse_normalized": 3.0,
        "mse_physical": 5.0,
        "per_field_mse_normalized": candidate_fields,
        "per_field_mse_physical": {name: 10.0 * value for name, value in candidate_fields.items()},
        "per_field_relative_l2": {name: value / 10.0 for name, value in candidate_fields.items()},
        "per_field_unobserved_mse_normalized": {
            name: 0.5 * value for name, value in candidate_fields.items()
        },
        "observed_entry_mse_normalized": 2.0,
        "unobserved_entry_mse_normalized": 4.0,
    }
    source_report = {
        "mse_normalized": 0.0,
        "mse_physical": 0.0,
        "per_field_mse_normalized": source_fields,
        "per_field_mse_physical": source_fields,
        "per_field_relative_l2": source_fields,
        "per_field_unobserved_mse_normalized": source_fields,
        "observed_entry_mse_normalized": 0.0,
        "unobserved_entry_mse_normalized": 0.0,
    }
    candidate_metrics = _report_metrics(candidate_report)["metrics"]
    source_metrics = _report_metrics(source_report)["metrics"]
    delta = _numeric_delta(candidate_metrics, source_metrics)

    for key in (
        "per_field_mse_normalized",
        "per_field_mse_physical",
        "per_field_relative_l2",
        "per_field_unobserved_mse_normalized",
    ):
        assert set(candidate_metrics[key]) == set(field_names)
        assert delta[key] == pytest.approx(candidate_metrics[key])
    assert delta["observed_entry_mse_normalized"] == pytest.approx(2.0)
    assert delta["unobserved_entry_mse_normalized"] == pytest.approx(4.0)


def test_native_audit_config_rejects_non_validation_or_unsaved_cadence(tmp_path):
    config = _audit_config(tmp_path)
    validate_config(config)
    broken = copy.deepcopy(config)
    broken["evaluation"]["split"] = "test"
    with pytest.raises(ValueError, match="validation split"):
        validate_config(broken)
    broken = copy.deepcopy(config)
    broken["checkpointing"]["every_steps"] = 3
    with pytest.raises(ValueError, match="multiple of checkpointing.every_steps"):
        validate_config(broken)


def test_disjoint_batch_is_complete_and_native_geometry_is_a_gather(tmp_path):
    config, dataset, normalizer, run_dir = _runtime(tmp_path)
    selector_ids = ("validation_000:0",)
    batch, metadata = build_disjoint_validation_batch(
        dataset,
        SensorProtocol(name="random_uniform", field_counts={"u": 3}, seed=61),
        selector_sample_ids=selector_ids,
        max_samples=2,
        seed=2027,
        device=torch.device("cpu"),
    )
    family, provenance = build_native_topology_family(
        config, dataset, normalizer, run_dir, torch.device("cpu")
    )
    geometry = initialize_native_geometry(
        family,
        batch,
        (4, 4),
        excluded_sample_ids=selector_ids,
    )
    assert set(batch.sample_ids).isdisjoint(selector_ids)
    assert batch.query_coords.shape[1] == batch.target_fields.shape[1] == 16
    assert geometry["geometry_diagnostics"]["native_gather_applied"] == 1
    assert geometry["geometry_diagnostics"]["native_area_average_applied"] == 0
    assert geometry["geometry_diagnostics"]["complete_native_cartesian_grid"] == 1
    assert provenance["sampling_artifact"]["line_bank_size"] == 4
    assert provenance["line_indices_evaluation"] == [[0, 1, 2, 3]]
    source_objective = TopologyFamily(
        _topology_config(), dataset.data_spec, normalizer
    ).spatial_objective
    assert provenance["sampling_artifact"] == source_objective.sampling_artifact()
    for suffix in ("directions", "offsets"):
        torch.testing.assert_close(
            getattr(family.spatial_objective, f"line_{suffix}_0"),
            getattr(source_objective, f"line_{suffix}_0"),
            rtol=0,
            atol=0,
        )
    assert verify_saved_family_hash(_Store(run_dir)) == provenance["artifact_sha256"]

    manifest = manifest_from_batch(batch, dataset.path, "validation")
    contract = contract_payload(
        config={
            "dataset": config["dataset"],
            "evaluation": {"split": "validation", "native_topology_audit": {"seed": 2027}},
        },
        dataset=dataset,
        batch=batch,
        batch_metadata=metadata,
        sensor_manifest=manifest,
        family_provenance=provenance,
        config_sha256="resolved-config-hash",
        selector_manifest_sha256="selector-hash",
    )
    assert contract["split"] == "validation"
    assert contract["query_count_per_sample"] == 16
    assert contract["sensor_manifest_sha256"] == manifest.digest()

    artifact = run_dir / "artifacts/topology_family.pt"
    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        verify_saved_family_hash(_Store(run_dir))


def test_disjoint_selection_loads_dense_fields_only_for_selected_indices():
    spec = DataSpec(("u", "v"), ("1", "1"), 2, (4, 4), mesh_type="structured")
    selector_ids = ("validation_000:0", "validation_001:0")
    seed = 2027
    max_samples = 3
    candidate_indices = list(range(2, 5000))
    positions = np.sort(
        np.random.default_rng(seed).choice(len(candidate_indices), size=max_samples, replace=False)
    )
    selected_indices = [candidate_indices[int(position)] for position in positions]

    class MetadataOnlyDataset:
        def __init__(self):
            self.data_spec = spec
            self.field_names = spec.field_names
            self.normalizer = FieldNormalizer.identity(2)
            self.path = Path("metadata-only-fixture")
            self.trajectory_ids = tuple(f"validation_{index:03d}" for index in range(5000))
            self._items = [(index, 0) for index in range(5000)]
            self.allowed_reads = set(selected_indices)
            self.read_indices = []

        def __len__(self):
            return len(self._items)

        def __getitem__(self, index):
            if index not in self.allowed_reads:
                raise AssertionError(f"unselected dense snapshot {index} was loaded")
            self.read_indices.append(index)
            return _grid_sample(index, self.data_spec)

    dataset = MetadataOnlyDataset()
    batch, metadata = build_disjoint_validation_batch(
        dataset,
        SensorProtocol(name="random_uniform", field_counts={"u": 3}, seed=61),
        selector_sample_ids=selector_ids,
        max_samples=max_samples,
        seed=seed,
        device=torch.device("cpu"),
    )
    assert dataset.read_indices == selected_indices
    assert metadata["dataset_indices"] == selected_indices
    assert set(batch.sample_ids).isdisjoint(selector_ids)
    assert len(batch.sample_ids) == max_samples


def test_audit_cadence_is_committed_and_outputs_are_resume_idempotent(tmp_path):
    assert audit_due(0, 10, durable=True, initial=True)
    assert audit_due(20, 10, durable=True, update_accepted=True)
    assert not audit_due(20, 10, durable=True, update_accepted=False)
    assert not audit_due(20, 10, durable=False, update_accepted=True)
    assert audit_due(17, 10, durable=True, terminal=True)
    store = _Store(tmp_path)
    payload = {"identity": {"committed_step": 20, "samples": ["v1", "v2"]}, "score": 1.25}
    path, created = write_idempotent_report(store, "evaluation/native/step_20.json", payload)
    assert created and path.is_file()
    same_path, created_again = write_idempotent_report(
        store, "evaluation/native/step_20.json", payload
    )
    assert same_path == path and not created_again
    with pytest.raises(ValueError, match="provenance changed"):
        write_idempotent_report(
            store,
            "evaluation/native/step_20.json",
            {"identity": {"committed_step": 20, "samples": ["different"]}},
        )


def test_audit_restores_rng_module_modes_ema_flag_and_live_parameters():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(3, 2)
            self.layer.eval()
            self._ema_eval = True

    model = Model().train()
    model.layer.eval()
    before_training = [module.training for module in model.modules()]
    before_state = {name: value.clone() for name, value in model.state_dict().items()}
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    with _preserve_audit_state((model,)):
        assert model._ema_eval is False
        model.eval()
        random.random()
        np.random.random()
        torch.rand(3)
    assert model._ema_eval is True
    assert [module.training for module in model.modules()] == before_training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before_state[name], rtol=0, atol=0)
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    assert np.array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)


def test_bar_distance_decomposition_matches_native_objective_oracle():
    pytest.importorskip("gudhi")
    spec = DataSpec(("u", "v"), ("1", "1"), 2, (4, 4), mesh_type="structured")
    family = TopologyFamily(_topology_config(), spec, FieldNormalizer.identity(2))
    axis = torch.linspace(0.0, 1.0, 4)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    reference = torch.stack(
        (xx + 0.2 * yy, torch.sin(2 * torch.pi * xx) + 0.3 * yy), dim=0
    ).unsqueeze(0)
    generated = reference + torch.stack(
        (0.08 * torch.sin(torch.pi * yy), 0.04 * torch.cos(torch.pi * xx)), dim=0
    ).unsqueeze(0)
    capture = {"generated_grid": generated, "reference_grid": reference}
    diagnostics, complete_seconds = paired_bar_diagnostics(
        family,
        capture,
        capture,
        ("validation_003:0",),
        max_snapshots=2,
    )
    result = family.spatial_objective(
        generated,
        reference,
        phase="evaluation",
        global_step=0,
    )
    rows = diagnostics["roles"]["source_live"]["rows"]
    from_bar_rows = np.mean([row["total_distance_after_line_weight"] for row in rows])
    assert from_bar_rows == pytest.approx(float(result.scalar_loss), rel=1e-6, abs=1e-7)
    for row in rows:
        assert row["total_distance_before_line_weight"] == pytest.approx(
            row["finite_distance_normalized_by_hw"]
            + row["weighted_essential_distance_not_hw_normalized"],
            rel=1e-6,
            abs=1e-8,
        )
    assert diagnostics["ground_truth_ph_seconds_once"] >= 0
    assert diagnostics["ground_truth_filtration_seconds_once"] >= 0
    assert diagnostics["pairing_ph_seconds_total"] >= diagnostics["ground_truth_ph_seconds_once"]
    assert diagnostics["complete_bar_summary_seconds"] == pytest.approx(complete_seconds)
    assert complete_seconds >= diagnostics["pairing_ph_seconds_total"]
    assert diagnostics["roles"]["source_live"]["prediction_filtration_seconds"] >= 0
    assert diagnostics["roles"]["source_live"]["diagram_distance_comparisons_seconds"] >= 0
    assert diagnostics["extra_ph_note"].startswith("Separate bounded")


def test_paired_audit_runs_live_full_bank_and_resumes_by_identity(tmp_path):
    pytest.importorskip("gudhi")
    config, dataset, normalizer, run_dir = _runtime(tmp_path)
    config["evaluation"] = {
        "seed": 77,
        "generation_steps": 1,
        "native_topology_audit": {
            "enabled": True,
            "every_steps": 2,
            "max_samples": 2,
            "seed": 2027,
        },
    }
    # The audit must use LIVE weights internally without rewriting the legacy
    # evaluation policy used by selection and ordinary validation.
    config["model"] = {"model_ema_eval": True}
    selector_ids = ("validation_000:0",)
    protocol = SensorProtocol(name="random_uniform", field_counts={"u": 3}, seed=61)
    batch, batch_metadata = build_disjoint_validation_batch(
        dataset,
        protocol,
        selector_sample_ids=selector_ids,
        max_samples=2,
        seed=2027,
        device=torch.device("cpu"),
    )
    family, family_provenance = build_native_topology_family(
        config, dataset, normalizer, run_dir, torch.device("cpu")
    )
    geometry = initialize_native_geometry(family, batch, (4, 4), excluded_sample_ids=selector_ids)
    family_provenance["native_geometry"] = geometry
    sensor_manifest = manifest_from_batch(batch, dataset.path, "validation")
    contract = contract_payload(
        config=config,
        dataset=dataset,
        batch=batch,
        batch_metadata=batch_metadata,
        sensor_manifest=sensor_manifest,
        family_provenance=family_provenance,
        config_sha256="resolved-config-hash",
        selector_manifest_sha256="fixed-selector-hash",
    )

    class ToyModel(torch.nn.Module):
        def __init__(self, delta):
            super().__init__()
            self.delta = torch.nn.Parameter(
                torch.tensor(delta, dtype=torch.float32).reshape(1, 1, 2)
            )
            self._ema_eval = True
            self.weight_context_modes = []

        def evaluation_weight_context(self):
            self.weight_context_modes.append(self._ema_eval)
            return nullcontext()

    source = ToyModel([0.0, 0.0])
    candidate = ToyModel([0.03, -0.02])
    evaluate_calls = []
    assert config["model"]["model_ema_eval"] is True
    with source.evaluation_weight_context():
        pass
    with candidate.evaluation_weight_context():
        pass

    def evaluate_fn(
        model, _complete, comparison, families, _banks, _field_names, _config, **_kwargs
    ):
        evaluate_calls.append(model)
        with model.evaluation_weight_context():
            target = comparison.target_fields
            prediction = target + model.delta
            result = families["topology"](
                prediction,
                target,
                coordinates=comparison.query_coords,
                context={"phase": "evaluation", "global_step": 0},
            )
            error = (prediction - target).square().mean(dim=(0, 1))
            mse = float((prediction - target).square().mean())
            relative = torch.sqrt(error / target.square().mean(dim=(0, 1)).clamp_min(1.0e-8))
            per_field_mse = {name: float(error[index]) for index, name in enumerate(("u", "v"))}
            per_field_relative = {
                name: float(relative[index]) for index, name in enumerate(("u", "v"))
            }
            return {
                "mse_normalized": mse,
                "mse_physical": mse,
                "mean_relative_l2": float(relative.mean()),
                "mean_relative_l2_physical": float(relative.mean()),
                "worst_field_relative_l2": float(relative.max()),
                "per_field_mse_normalized": per_field_mse,
                "per_field_mse_physical": per_field_mse,
                "per_field_relative_l2": per_field_relative,
                "per_field_relative_l2_physical": per_field_relative,
                "per_field_unobserved_mse_normalized": {
                    name: value * 0.75 for name, value in per_field_mse.items()
                },
                "observed_entry_mse_normalized": mse * 0.8,
                "unobserved_entry_mse_normalized": mse * 1.2,
                "coherence": {
                    "families": {
                        "topology": {
                            "total": float(result.scalar_loss),
                            "component_scalars": {
                                name: float(term.scalar_loss)
                                for name, term in result.component_results.items()
                            },
                            "diagnostics": result.diagnostics,
                        }
                    }
                },
            }

    store = _Store(run_dir)
    kwargs = {
        "store": store,
        "config": config,
        "current_model": candidate,
        "source_model": source,
        "native_family": family,
        "batch": batch,
        "field_names": dataset.field_names,
        "normalizer": normalizer,
        "family_scales": {"topology": 1.0},
        "evaluate_fn": evaluate_fn,
        "contract": contract,
        "family_provenance": family_provenance,
        "geometry_provenance": geometry,
        "committed_step": 2,
        "attempt_step": 2,
        "durable_references": [
            {"kind": "last", "path": "checkpoints/last.pt", "sha256": "durable"}
        ],
    }
    path, created, _ = execute_native_audit(**kwargs)
    assert created and path.is_file()
    assert len(evaluate_calls) == 2
    assert source.weight_context_modes == [True, False]
    assert candidate.weight_context_modes == [True, False]
    assert config["model"]["model_ema_eval"] is True
    assert source._ema_eval is True and candidate._ema_eval is True
    with source.evaluation_weight_context():
        pass
    with candidate.evaluation_weight_context():
        pass
    assert source.weight_context_modes == [True, False, True]
    assert candidate.weight_context_modes == [True, False, True]
    report = json.loads(path.read_text())
    assert report["split"] == "validation"
    assert report["sample_ids"] == list(batch.sample_ids)
    assert report["full_bank_evaluation"]["selected_line_indices"] == [[0, 1, 2, 3]]
    assert report["generation"]["weight_selection"].startswith("live;")
    assert report["bar_diagnostics"]["diagnostic_snapshot_ids"] == list(batch.sample_ids)
    expected_metric_keys = {
        "mse_normalized",
        "mse_physical",
        "mean_relative_l2",
        "mean_relative_l2_physical",
        "worst_field_relative_l2",
        "per_field_mse_normalized",
        "per_field_mse_physical",
        "per_field_relative_l2",
        "per_field_relative_l2_physical",
        "per_field_unobserved_mse_normalized",
        "observed_entry_mse_normalized",
        "unobserved_entry_mse_normalized",
    }
    source_metrics = report["source_to_reference"]["metrics"]
    candidate_metrics = report["candidate_to_reference"]["metrics"]
    assert expected_metric_keys <= source_metrics.keys()
    assert expected_metric_keys <= candidate_metrics.keys()
    assert candidate_metrics["per_field_mse_physical"] == pytest.approx({"u": 0.0009, "v": 0.0004})
    metric_delta = report["paired_metric_differences_candidate_minus_source"]["metrics"]
    assert metric_delta["per_field_mse_normalized"] == pytest.approx({"u": 0.0009, "v": 0.0004})
    assert metric_delta["per_field_unobserved_mse_normalized"] == pytest.approx(
        {"u": 0.000675, "v": 0.0003}
    )
    assert metric_delta["observed_entry_mse_normalized"] == pytest.approx(0.00052)
    assert metric_delta["unobserved_entry_mse_normalized"] == pytest.approx(0.00078)
    costs = report["cost_seconds"]
    assert costs["complete_audit_wall_before_json_write"] >= sum(
        costs[key]
        for key in (
            "source_evaluation_including_full_bank_native_ph",
            "candidate_evaluation_including_full_bank_native_ph",
            "bounded_bar_summary_complete_including_filtration_and_comparisons",
        )
    )
    assert costs["bounded_bar_summary_ground_truth_ph_pairing"] >= 0
    assert costs["bounded_bar_summary_generated_ph_pairing_total"] >= 0
    assert costs["bounded_bar_summary_diagram_distance_comparisons"] >= 0
    assert "final RunStore JSON serialization/write" in report["timing_scope"]
    assert "No CUDA-peak reset" in report["timing_scope"]
    second_path, created_again, _ = execute_native_audit(**kwargs)
    assert second_path == path and not created_again
    assert len(evaluate_calls) == 2
    with torch.no_grad():
        candidate.delta.add_(0.01)
    with pytest.raises(ValueError, match="provenance changed"):
        execute_native_audit(**kwargs)
