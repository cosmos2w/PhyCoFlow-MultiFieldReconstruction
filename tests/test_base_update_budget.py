"""Fixed base-training budgets retain full-split sampling and resumable epochs."""

import json

import h5py
import numpy as np
import pytest
import torch

from phycoflow_reconstruction.config.validate import _validate_base_training
from phycoflow_reconstruction.training.base_training import run_base_training
from phycoflow_reconstruction.training.common import iter_unique_batch_indices
from phycoflow_reconstruction.training.update_budget import base_training_steps_per_epoch


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "54"])
def test_invalid_base_budget(value):
    with pytest.raises(ValueError, match="positive integer"):
        _validate_base_training({"optimization": {"steps_per_epoch": value}})


def test_budget_does_not_restrict_sampling_to_first_half_and_rng_resumes():
    config = {"optimization": {"batch_size": 128, "steps_per_epoch": 54}}
    assert base_training_steps_per_epoch(config, 13824) == 54
    assert base_training_steps_per_epoch({"optimization": {"batch_size": 128}}, 13824) == 108
    generator = torch.Generator().manual_seed(42)
    first = list(iter_unique_batch_indices(13824, 54, 128, generator=generator))
    state = generator.get_state()
    second = list(iter_unique_batch_indices(13824, 54, 128, generator=generator))
    restored = torch.Generator().set_state(state)
    assert second == list(iter_unique_batch_indices(13824, 54, 128, generator=restored))
    assert any(i >= 6912 for batch in first for i in batch)
    assert all(len(set(batch)) == 128 for batch in first + second)
    assert first != second


def test_base_loop_and_resume_use_explicit_reporting_budget(tmp_path):
    path = tmp_path / "tiny.h5"
    with h5py.File(path, "w") as h:
        rng = np.random.default_rng(42)
        h.create_dataset("fields", data=rng.normal(size=(3, 8, 8, 1, 1, 2)).astype("f4"))
        coords = np.zeros((8, 1, 1, 3), dtype="f4")
        coords[:, 0, 0, 0] = np.arange(8)
        h.create_dataset("coordinates", data=coords)
        h.create_dataset("time", data=np.arange(8))
        h.create_dataset("conditions", data=np.zeros((3, 1), dtype="f4"))
        h.create_dataset("trajectory_id", data=["a", "b", "c"], dtype=h5py.string_dtype())
        for i, split in enumerate(("train", "validation", "test")):
            h.create_dataset(f"splits/{split}", data=[i])
        h.create_dataset("statistics/train_mean", data=np.zeros(2))
        h.create_dataset("statistics/train_std", data=np.ones(2))
        h.attrs["field_names"] = '["u", "v"]'
        h.attrs["grid_shape"] = "[8]"
    config = {
        "stage": "base_training",
        "case": "fixture",
        "dataset": {
            "path": str(path),
            "split": "train",
            "field_names": ["u", "v"],
            "normalization": "mean_std",
            "coordinate_dim": 1,
        },
        "model": {"name": "mlp_rbf", "hidden_dim": 8, "fourier_bands": 1, "query_points": 4},
        "observations": {"protocol": "random_uniform", "seed": 42, "fields": {"u": {"count": 2}}},
        "optimization": {"epochs": 2, "steps_per_epoch": 2, "batch_size": 2, "lr": 1e-3},
        "runtime": {
            "seed": 42,
            "device": "cpu",
            "progress": False,
            "data_strategy": "async_cpu",
            "num_workers": 0,
        },
        "evaluation": {
            "generation_steps": 1,
            "preview": {
                "enabled": True,
                "loss_every_epochs": 1,
                "reconstruct_every_epochs": 100,
                "query_points": 4,
            },
        },
        "checkpointing": {"enabled": True, "every_epochs": 1, "save_epoch_one": True},
        "output": {"experiment_name": "budget"},
    }
    run = run_base_training(config, case_dir=tmp_path / "case", max_steps=2)
    assert json.loads((run / "status.json").read_text())["global_step"] == 2
    run_base_training(config, case_dir=tmp_path / "case", resume=run)
    status = json.loads((run / "status.json").read_text())
    assert status["global_step"] == status["configured_steps"] == 4
    assert status["status"] == "completed"
    rows = [json.loads(line) for line in (run / "metrics/history.jsonl").read_text().splitlines()]
    assert [(r["epoch"], r["batches"]) for r in rows] == [(1, 2), (2, 2)]
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert manifest["training_epoch_budget"]["available_training_snapshots"] == 8
    assert manifest["training_epoch_budget"]["sample_presentations_per_epoch"] == 4
    assert manifest["training_validation"]["training_epoch"] == 2
