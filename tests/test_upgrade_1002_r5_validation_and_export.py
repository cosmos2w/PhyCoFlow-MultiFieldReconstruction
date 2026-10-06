"""Locked HDF5 validation and the normal evaluator's LIVE-only export path."""

import copy
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from phycoflow_reconstruction.training.update_budget import post_training_final_step
import torch

from phycoflow_reconstruction.data.validation import validate_h5_dataset
from phycoflow_reconstruction.evaluation.checkpoint import _load_evaluation_checkpoint_state
from phycoflow_reconstruction.training.parameter_interpolation import inference_only_checkpoint


def test_frozen_retention_artifact_path_is_relative_to_case(tmp_path):
    from phycoflow_reconstruction.cli import _load_case_config

    path = tmp_path / "profile.yaml"
    path.write_text("case: fixture\ndataset: {path: dataset.h5}\nobjectives:\n  parameter_retention:\n    calibration_path: configs/frozen_train_calibration.json\n")
    config = _load_case_config(path, tmp_path, "fixture", [], validate_stage=False)
    assert config["objectives"]["parameter_retention"]["calibration_path"] == str(
        tmp_path / "configs" / "frozen_train_calibration.json")


def test_epoch_limits_preserve_recipe_and_resume_boundary():
    assert post_training_final_step(7600, 0, 38, until_epoch=150) == 5700
    assert post_training_final_step(7600, 5700, 38, additional_epochs=50) == 7600
    assert post_training_final_step(7600, 5700, 38, until_epoch=200) == 7600
    assert post_training_final_step(7600, 5700, 38, max_steps=1900) == 7600
    with pytest.raises(ValueError, match="immutable"):
        post_training_final_step(7600, 5700, 38, until_epoch=201)
    with pytest.raises(ValueError, match="epoch-boundary"):
        post_training_final_step(7600, 5701, 38, additional_epochs=1)
    with pytest.raises(ValueError, match="only one"):
        post_training_final_step(7600, 0, 38, max_steps=38, until_epoch=1)
def test_hdf5_validation_does_not_read_any_field_payload(tmp_path, monkeypatch):
    path = tmp_path / "chronological.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("fields", data=np.full((1, 10, 2, 1, 1, 1), np.nan))
        handle.create_dataset("coordinates", data=np.zeros((2, 1, 1, 3)))
        handle.create_dataset("time", data=np.arange(10))
        handle.create_dataset("conditions", data=np.zeros((1, 0)))
        handle.attrs["field_names"] = '["T"]'
    original = h5py.Dataset.__getitem__

    def locked(dataset, key):
        if dataset.name == "/fields":
            raise AssertionError("locked field values were read")
        return original(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", locked)
    result = validate_h5_dataset(path, ["T"])
    assert result["valid"]
    assert result["field_payload_read"] is False
    assert result["validation_scope"] == "structural_metadata_only"
    assert "sampled_endpoints" not in result
    assert any("finiteness was not evaluated" in text for text in result["warnings"])
    assert not validate_h5_dataset(path, ["U", "T"])["valid"]


def _export():
    model = torch.nn.Linear(2, 1)
    config = {"model": {"name": "fixture", "model_ema_eval": False},
              "rollout": {"steps": 2, "solver": "euler"}}
    normalizer = SimpleNamespace(digest=lambda: "normalizer")
    dataset = SimpleNamespace(field_names=("T",), normalizer=normalizer)
    provenance = {"alpha": .5, "semantics": {"fields": ["T"],
        "normalizer_digest": "normalizer", "model_config": config["model"],
        "rollout": config["rollout"]},
        **{key: "a" * 64 for key in ("source_checkpoint_sha256", "child_checkpoint_sha256",
                                   "source_parameter_sha256", "child_parameter_sha256")}}
    payload = inference_only_checkpoint(model.state_dict(), normalization={}, provenance=provenance)
    return model, config, dataset, payload


def test_normal_evaluator_loads_live_export_without_training_aux(monkeypatch):
    model, config, dataset, payload = _export()
    with torch.no_grad():
        model.weight.zero_()

    def forbidden(*args):
        raise AssertionError("deploy-only checkpoint requested training aux state")

    monkeypatch.setattr("phycoflow_reconstruction.evaluation.checkpoint.load_training_aux_state", forbidden)
    _load_evaluation_checkpoint_state(model, payload, config, dataset)
    assert torch.equal(model.weight, payload["model"]["weight"])


def test_segmented_clean_stop_finishes_segment_then_resumes_own_child(tmp_path, monkeypatch):
    from phycoflow_reconstruction.training import segmented

    case = tmp_path / "case"
    case.mkdir()
    (case / "run.py").write_text("# fixture entrypoint\n")
    run = case / "runs" / "fixture" / "child"
    (run / "metrics").mkdir(parents=True)
    (run / "status.json").write_text('{"post_training_seconds": 1, "peak_cuda_memory_bytes": 0}')
    stop = tmp_path / "requested.stop"
    config = {"case": "fixture", "stage": "post_training", "output": {"experiment_name": "fixture"},
              "optimization": {"epochs": 5000}}
    monkeypatch.setattr(segmented, "load_config", lambda *args: config)
    monkeypatch.setattr(segmented, "_load_case_config", lambda *args: config)
    monkeypatch.setattr(segmented, "configured_epoch_steps", lambda *args: 38)
    age = [0]
    commands = []

    def selection(*args):
        return (run if age[0] else None), age[0], False

    def execute(command, **kwargs):
        commands.append(command)
        age[0] += int(command[command.index("--max-steps") + 1])
        stop.touch()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(segmented, "select_run", selection)
    monkeypatch.setattr(segmented.subprocess, "run", execute)
    argv = ["--config", str(tmp_path / "fixture.yaml"), "--case-dir", str(case),
            "--segment-epochs", "25", "--allocation-hours", "100", "--stop-file", str(stop)]
    segmented.main(argv)
    assert age[0] == 38 and len(commands) == 1
    assert "--resume" not in commands[0]
    stop.unlink()
    segmented.main(argv)
    assert age[0] == 25 * 38 and len(commands) == 2
    assert commands[1][commands[1].index("--resume") + 1] == str(run)


@pytest.mark.parametrize("change", ["optimizer", "ema", "field", "normalizer", "alpha", "kind"])
def test_normal_evaluator_rejects_incompatible_export(change):
    model, config, dataset, payload = _export()
    if change == "optimizer":
        payload["optimizer"] = {}
    elif change == "ema":
        config = copy.deepcopy(config)
        config["model"]["model_ema_eval"] = True
        payload["interpolation"]["semantics"]["model_config"] = config["model"]
    elif change == "field":
        payload["interpolation"]["semantics"]["fields"] = ["p"]
    elif change == "normalizer":
        payload["interpolation"]["semantics"]["normalizer_digest"] = "other"
    elif change == "alpha":
        payload["interpolation"]["alpha"] = float("nan")
    else:
        payload["checkpoint_kind"] = "unknown"
    with pytest.raises(ValueError):
        _load_evaluation_checkpoint_state(model, payload, config, dataset)
