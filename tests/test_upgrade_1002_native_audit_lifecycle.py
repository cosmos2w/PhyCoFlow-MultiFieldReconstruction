"""Exercise the default-off audit through real CPU child-run lifecycle calls."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from helpers.coherence import _base_config, _post_config

from phycoflow_reconstruction.config.validate import validate_config
from phycoflow_reconstruction.training import native_topology_audit as native_audit_module
from phycoflow_reconstruction.training.base_training import run_base_training
from phycoflow_reconstruction.training.post_training import run_post_training
from phycoflow_reconstruction.training.run_store import file_sha256, load_project_checkpoint


def _write_lifecycle_fixture(path: Path) -> None:
    """A tiny paired dataset with enough validation snapshots for disjoint audit."""
    rng = np.random.default_rng(108)
    trajectories, times, side = 8, 2, 4
    fields = rng.normal(size=(trajectories, times, side * side, 1, 1, 2)).astype("float32")
    y, x = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    coordinates = np.stack((x, y, np.zeros_like(x)), axis=-1).reshape(side * side, 1, 1, 3)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("fields", data=fields)
        handle.create_dataset("coordinates", data=coordinates.astype("float32"))
        handle.create_dataset("time", data=np.asarray([0.0, 1.0], dtype="float32"))
        handle.create_dataset("conditions", data=np.empty((trajectories, 0), dtype="float32"))
        handle.create_dataset(
            "trajectory_id",
            data=np.asarray(
                [f"trajectory_{i:02d}" for i in range(trajectories)], dtype=h5py.string_dtype()
            ),
        )
        splits = handle.create_group("splits")
        splits.create_dataset("train", data=np.asarray([0], dtype="int64"))
        splits.create_dataset("validation", data=np.arange(1, 7, dtype="int64"))
        splits.create_dataset("test", data=np.asarray([7], dtype="int64"))
        statistics = handle.create_group("statistics")
        statistics.create_dataset("train_mean", data=np.zeros(2, dtype="float32"))
        statistics.create_dataset("train_std", data=np.ones(2, dtype="float32"))
        handle.attrs["field_names"] = '["u", "v"]'
        handle.attrs["field_units"] = '["1", "1"]'
        handle.attrs["grid_shape"] = "[4, 4]"
        handle.attrs["schema_version"] = "1.0"


def _lifecycle_config(dataset_path: Path, source_run: Path, name: str, *, enabled: bool) -> dict:
    config = _post_config(dataset_path, source_run)
    config["dataset"]["grid_shape"] = [4, 4]
    config["optimization"].update(epochs=2, batch_size=1, train_fraction=1.0, lr=1.0e-4)
    config["runtime"].update(
        progress=False,
        num_workers=0,
        device="cpu",
    )
    # This lifecycle test exercises save/resume/audit ordering, not the
    # independent asynchronous HDF5 worker. The valid structured protocol
    # selects the compatibility loader and keeps the synthetic CPU path in
    # process so a worker fork cannot obscure audit failures.
    config["observations"]["protocol"] = "structured_stride"
    config["coherence"]["compute_budget"] = {
        "batch_size": 1,
        "point_count": 16,
        "query_policy": "fixed_shared",
        "query_seed": 773,
    }
    config["coherence"]["families"] = {
        "topology": {
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
                "self": {"enabled": True},
                "mutual": {
                    "enabled": True,
                    "groups": [["u", "v"]],
                    "line_bank_size": 4,
                    "training_subset_size": 2,
                    "seed": 37,
                },
            },
        }
    }
    config["evaluation"].update(
        split="validation",
        max_samples=1,
        query_points=16,
        sample_selection="first",
        generation_steps=1,
        preview={"enabled": False},
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
    config["rollout"] = {"steps": 1, "solver": "euler"}
    config["observation_consistency"] = {"mode": "hard", "final_clamp": True}
    config["output"]["experiment_name"] = name
    return config


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_nested_exact(actual, expected, path="state"):
    if torch.is_tensor(actual):
        assert torch.is_tensor(expected), path
        assert torch.equal(actual, expected), path
    elif isinstance(actual, dict):
        assert isinstance(expected, dict) and actual.keys() == expected.keys(), path
        for key in actual:
            _assert_nested_exact(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, (tuple, list)):
        assert isinstance(expected, type(actual)) and len(actual) == len(expected), path
        for index, (actual_value, expected_value) in enumerate(zip(actual, expected)):
            _assert_nested_exact(actual_value, expected_value, f"{path}[{index}]")
    else:
        assert actual == expected, path


def test_run_post_training_audit_off_on_and_resumed_after_durable_save(tmp_path, monkeypatch):
    pytest.importorskip("gudhi")
    dataset_path = tmp_path / "lifecycle.h5"
    _write_lifecycle_fixture(dataset_path)
    case_dir = tmp_path / "case"
    source_config = _base_config(dataset_path)
    source_config["observations"]["protocol"] = "structured_stride"
    source = run_base_training(source_config, case_dir=case_dir)

    disabled = _lifecycle_config(dataset_path, source, "audit_disabled", enabled=False)
    validate_config(disabled)
    disabled_run = run_post_training(disabled, case_dir=case_dir, max_steps=3)
    assert not (disabled_run / "evaluation" / "native_topology_audit").exists()

    enabled = _lifecycle_config(dataset_path, source, "audit_enabled", enabled=True)
    validate_config(enabled)
    checkpoint_observations = []
    original_execute = native_audit_module.execute_native_audit

    def observe_checkpoint_boundary(**kwargs):
        if kwargs["committed_step"] == 2 and not kwargs.get("terminal", False):
            last_path = kwargs["store"].run_dir / "checkpoints" / "last.pt"
            last_hash = file_sha256(last_path)
            last_reference = next(
                item for item in kwargs["durable_references"] if item["kind"] == "last"
            )
            artifact_path = kwargs["store"].run_dir / "artifacts" / "topology_family.pt"
            artifact_hash = file_sha256(artifact_path)
            assert last_reference["sha256"] == last_hash
            result = original_execute(**kwargs)
            assert file_sha256(last_path) == last_hash
            checkpoint_observations.append((last_hash, artifact_hash))
            return result
        return original_execute(**kwargs)

    monkeypatch.setattr(native_audit_module, "execute_native_audit", observe_checkpoint_boundary)
    child = run_post_training(enabled, case_dir=case_dir, max_steps=3)
    audit_dir = child / "evaluation" / "native_topology_audit"
    assert len(checkpoint_observations) == 1
    step_two_hash, step_two_artifact_hash = checkpoint_observations[0]
    step_two_report = _read_json(audit_dir / "step_000000002.json")
    assert step_two_report["committed_step"] == 2
    assert step_two_report["terminal_evaluation"] is False
    assert step_two_report["durable_references"][0]["sha256"] == step_two_hash
    assert step_two_report["child_family_artifact_sha256"] == step_two_artifact_hash

    final_checkpoint_path = child / "checkpoints" / "last.pt"
    final_checkpoint_hash = file_sha256(final_checkpoint_path)
    enabled_three = load_project_checkpoint(final_checkpoint_path)
    disabled_three = load_project_checkpoint(disabled_run / "checkpoints" / "last.pt")
    assert enabled_three["global_step"] == disabled_three["global_step"] == 3
    assert final_checkpoint_hash != step_two_hash
    for key in (
        "model",
        "optimizer",
        "rng_state",
        "family_states",
        "family_configs",
        "family_scales",
        "coherence_calibration_sha256",
    ):
        _assert_nested_exact(enabled_three[key], disabled_three[key], key)
    for optional_key in (
        "component_controller",
        "fidelity_controller",
        "coherence_selector_state",
    ):
        assert (optional_key in enabled_three) == (optional_key in disabled_three)
        if optional_key in enabled_three:
            _assert_nested_exact(enabled_three[optional_key], disabled_three[optional_key])

    contract = _read_json(audit_dir / "contract.json")
    source_report = _read_json(audit_dir / "step_000000000.json")
    terminal_three = _read_json(audit_dir / "terminal_attempt_000000003.json")
    assert source_report["committed_step"] == 0
    assert source_report["durable_references"][0]["kind"] == "immutable_source_checkpoint"
    assert terminal_three["committed_step"] == 3
    assert terminal_three["durable_references"][0]["sha256"] == final_checkpoint_hash
    assert step_two_report["full_bank_evaluation"]["selected_line_indices"] == [[0, 1, 2, 3]]
    assert set(step_two_report["sample_ids"]).isdisjoint(contract["selector_sample_ids"])

    disabled_run = run_post_training(disabled, case_dir=case_dir, max_steps=1, resume=disabled_run)
    child = run_post_training(enabled, case_dir=case_dir, max_steps=1, resume=child)
    enabled_four = load_project_checkpoint(child / "checkpoints" / "last.pt")
    disabled_four = load_project_checkpoint(disabled_run / "checkpoints" / "last.pt")
    assert enabled_four["global_step"] == disabled_four["global_step"] == 4
    for key in ("model", "optimizer", "rng_state", "family_states"):
        _assert_nested_exact(enabled_four[key], disabled_four[key], key)
    assert _read_json(audit_dir / "contract.json")["identity_sha256"] == contract["identity_sha256"]
    terminal_four = _read_json(audit_dir / "terminal_attempt_000000004.json")
    assert terminal_four["durable_references"][0]["sha256"] == file_sha256(
        child / "checkpoints" / "last.pt"
    )
