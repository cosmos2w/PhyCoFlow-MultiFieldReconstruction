"""Selection, leakage protection and exact mid-epoch sample coverage."""

import functools
import math
import operator
from collections import Counter

import numpy as np
import pytest

from phycoflow_reconstruction.data.topology_subset import (
    iter_full_pass_indices,
    select_topology_subset,
)


def rows():
    return [
        {
            "category": f"category{regime}",
            "parameter": m,
            "trajectory_id": f"{regime}_{m}_{run}",
            "frame": frame,
            "time": frame * 10.0,
            "split": "train",
        }
        for regime in range(3)
        for m in [0.0, -0.1]
        for run in range(3)
        for frame in range(16 + run * 4)
    ]


def test_subset_budget_repeatability_rare_regimes_trajectories_and_time():
    source = rows()
    chosen, strata = select_topology_subset(source, strata_keys=["category", "parameter"])
    assert chosen == select_topology_subset(source, strata_keys=["category", "parameter"])[0]
    assert (
        chosen != select_topology_subset(source, seed=43, strata_keys=["category", "parameter"])[0]
    )
    assert len(chosen) == len(set(chosen)) == math.ceil(0.3 * len(source))
    selected = [source[i] for i in chosen]
    assert {(r["category"], r["parameter"]) for r in selected} == {
        (r["category"], r["parameter"]) for r in source
    }
    counts = Counter(r["trajectory_id"] for r in selected)
    assert set(counts) == {r["trajectory_id"] for r in source}
    assert min(counts.values()) >= 4
    for name, count in counts.items():
        times = [r["frame"] for r in source if r["trajectory_id"] == name]
        actual = {r["frame"] for r in selected if r["trajectory_id"] == name}
        assert all(len(actual.intersection(chunk)) == 1 for chunk in np.array_split(times, count))
    assert sum(s["selected"] for s in strata) == len(chosen)
    assert select_topology_subset(source, fraction=1.0)[0] == list(range(len(source)))


def test_subset_rejects_holdout_and_impossible_coverage():
    source = rows()
    with pytest.raises(ValueError, match="coverage"):
        select_topology_subset(source, fraction=0.01)
    source[5]["split"] = "validation"
    with pytest.raises(ValueError, match="training rows only"):
        select_topology_subset(source, strata_keys=["category", "parameter"])


def test_full_pass_covers_each_sample_once_and_resumes_exactly():
    full = list(iter_full_pass_indices(27, 12, 8, seed=42))
    for i in range(3):
        epoch = full[i * 4 : (i + 1) * 4]
        assert sorted(functools.reduce(operator.iadd, epoch, [])) == list(range(27))
        assert list(map(len, epoch)) == [8, 8, 8, 3]
    assert full[:4] != full[4:8]
    assert full[3:] == list(iter_full_pass_indices(27, 9, 8, seed=42, start_step=3))


def test_subset_keeps_resident_loader_selection_consistent(tmp_path):
    from types import SimpleNamespace

    from phycoflow_reconstruction.data.splits import SplitSelection
    from phycoflow_reconstruction.data.topology_subset import apply_training_subset

    path = tmp_path / "fingerprint_fixture"
    path.write_bytes(b"unchanged source payload")
    samples = rows()
    original_selection = SplitSelection(
        "train", tuple(range(len(samples))), (0,), "stored_trajectory"
    )
    dataset = SimpleNamespace(
        path=path,
        split_name="train",
        reconstruction_unit="snapshot",
        dataset_metadata={"samples": samples},
        _items=[(i, 0) for i in range(len(samples))],
        selection=original_selection,
    )
    manifest = apply_training_subset(dataset, {"fraction": 0.3})
    # The resident path reads selection, whereas the compact path reads _items.
    # They must enumerate the same reduced snapshots in the same order.
    assert [
        (i, t)
        for i in dataset.selection.trajectory_indices
        for t in dataset.selection.frame_indices
    ] == dataset._items
    assert len(dataset._items) == manifest["selected_count"]
    assert len(original_selection.trajectory_indices) == len(samples)
    assert path.read_bytes() == b"unchanged source payload"


@pytest.mark.parametrize("distance", ["sliced_wasserstein", "spatial_wasserstein"])
@pytest.mark.parametrize("mode", ["component_constrained", "topology_regularized"])
def test_subset_training_resumes_mid_epoch_with_short_final_batch(tmp_path, distance, mode):
    import json

    import h5py
    import torch
    from helpers.coherence import _base_config, _post_config, _write_fixture

    from phycoflow_reconstruction.config.validate import validate_config
    from phycoflow_reconstruction.training.base_training import run_base_training
    from phycoflow_reconstruction.training.post_training import run_post_training
    from phycoflow_reconstruction.training.run_store import load_project_checkpoint

    pytest.importorskip("gudhi")
    path = tmp_path / "snapshot_fixture.h5"
    _write_fixture(path)
    with h5py.File(path, "a") as handle:
        fields = handle["fields"][:].reshape(6, 1, 16, 1, 1, 2)
        fields = np.concatenate([fields, fields[-1:]], axis=0)
        for key in ["fields", "time", "conditions", "trajectory_id", "splits"]:
            del handle[key]
        handle["fields"] = fields
        handle["time"] = [0.0]
        handle["conditions"] = np.empty((7, 0), dtype=np.float32)
        handle.create_dataset(
            "trajectory_id", data=[f"snapshot{i}" for i in range(7)], dtype=h5py.string_dtype()
        )
        for name, ids in {"train": [0, 1, 2, 3], "validation": [4, 5], "test": [6]}.items():
            handle[f"splits/{name}"] = ids
        metadata = [
            {
                "trajectory_id": "simulation",
                "frame": i,
                "time": float(i),
                "split": "train" if i < 4 else "validation" if i < 6 else "test",
            }
            for i in range(7)
        ]
        handle["metadata/json"] = json.dumps({"samples": metadata})
    source = run_base_training(_base_config(path), case_dir=tmp_path / "case")
    config = _post_config(path, source)
    config["dataset"]["grid_shape"] = [4, 4]
    config["optimization"].update(
        epochs=2,
        batch_size=2,
        sampling="full_pass",
        model_mode="eval",
        training_subset={"fraction": 0.75, "seed": 42, "min_frames_per_trajectory": 1},
        gradient_balance=mode,
        component_constraints={
            "calibration_batches": 2,
            "correction_selection": "largest_violation",
            "topology_reduction": "batch_mean",
        },
    )
    if mode == "topology_regularized":
        config["optimization"].update(epochs=4, steps_per_epoch=1)
    config["coherence"]["families"] = {
        "topology": {
            "strategy": "cubical_persistence",
            "target_use": "paired_supervised",
            "units": "model_units",
            "fields": ["u", "v"],
            "persistence": {"distance": distance},
            "geometry": {
                "grid_shape": [4, 4],
                "periodic": True,
                "neighbors": 1,
                "antialias_downsample": True,
            },
            "filtration": {"smoothing_sigma": 0.0},
            "components": {
                "self": {"enabled": True},
                "mutual": {"enabled": True, "groups": [["u", "v"]], "lines": 2},
            },
        }
    }
    config["coherence"]["compute_budget"] = {
        "batch_size": 2,
        "point_count": 16,
        "query_policy": "fixed_shared",
    }
    config["objectives"]["data_retention"]["enabled"] = False
    config["observation_consistency"] = {"mode": "hard", "final_clamp": True}
    config["evaluation"].update(
        max_samples=2,
        query_points=16,
        sample_selection="uniform",
        native_topology=True,
        preview={"enabled": False},
    )
    config["checkpointing"] = {
        "every_epochs": 1,
        "every_steps": 1,
        "validation_every_epochs": 2,
        "selection_metric": "topology_with_fidelity",
    }
    config["posttrain_fidelity"] = {
        "max_relative_mse_increase": 0.05,
        "max_relative_field_mse_increase": 0.05,
        "max_relative_native_topology_increase": 0.0,
    }
    validate_config(config)
    child = run_post_training(config, case_dir=tmp_path / "case", max_steps=1)
    run_post_training(config, case_dir=tmp_path / "case", max_steps=3, resume=child)
    whole = run_post_training(config, case_dir=tmp_path / "case", max_steps=4)
    resumed = load_project_checkpoint(child / "checkpoints/last.pt")
    expected = load_project_checkpoint(whole / "checkpoints/last.pt")
    assert resumed["global_step"] == expected["global_step"] == 4
    assert resumed["training_subset_sha256"] == expected["training_subset_sha256"]
    assert resumed["component_controller"] == expected["component_controller"]
    audit = [
        json.loads(line)
        for line in (child / "metrics/constraint_updates.jsonl").read_text().splitlines()
    ]
    assert [r["constraint_batch_size"] for r in audit] == [2, 1, 2, 1]
    assert all(
        r["topology_reduction"] == "batch_mean" and r["topology_constraint_count"] == 6
        for r in audit
    )
    assert [r["fidelity_constraint_count"] for r in audit] == [8, 4, 8, 4]
    for name, value in expected["model"].items():
        torch.testing.assert_close(resumed["model"][name], value, rtol=0, atol=0)
    manifest = json.loads((child / "run_manifest.json").read_text())
    assert manifest["steps_per_epoch"] == (1 if mode == "topology_regularized" else 2)
    assert manifest["training_subset_count"] == 3 and manifest["batches_per_dataset_pass"] == 2
    assert resumed["samples_seen"] == 6 and resumed["dataset_passes"] == 2.0
    artifact = child / "artifacts/training_subset.json"
    selected = json.loads(artifact.read_text())
    assert all(row["split"] == "train" and row["dataset_index"] < 4 for row in selected["samples"])
    selected["original_indices"][0] = 99
    artifact.write_text(json.dumps(selected))
    with pytest.raises(ValueError, match="resume training subset differs"):
        run_post_training(config, case_dir=tmp_path / "case", max_steps=0, resume=child)


def test_subset_metadata_mapping_preserves_selection_and_split_checks():
    source = rows()
    expected, _ = select_topology_subset(source, strata_keys=["category", "parameter"])
    mapping = {
        "category": "flow_type",
        "parameter": "control",
        "trajectory_id": "simulation",
        "time": "timestamp",
        "frame": "snapshot",
        "split": "partition",
    }
    renamed = [{mapping[key]: value for key, value in row.items()} for row in source]
    settings = {
        "strata_keys": ["flow_type", "control"],
        "trajectory_key": "simulation",
        "time_key": "timestamp",
        "frame_key": "snapshot",
        "split_key": "partition",
    }
    actual, strata = select_topology_subset(renamed, **settings)
    assert actual == expected
    assert all(set(row["labels"]) == {"flow_type", "control"} for row in strata)
    renamed[0]["partition"] = "test"
    with pytest.raises(ValueError, match="training rows only"):
        select_topology_subset(renamed, **settings)


def test_subset_reports_missing_metadata_column():
    source = rows()
    del source[0]["trajectory_id"]
    with pytest.raises(ValueError, match="missing columns.*trajectory_id"):
        select_topology_subset(source)
