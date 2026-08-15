"""Run-store tests cover atomic checkpoints, resume, and child lineage metadata."""

import pytest
import torch

from phycoflow_reconstruction.training.run_store import RunStore


def _config():
    return {"stage": "base_training", "case": "fixture", "output": {}}


def test_create_checkpoint_and_resume(tmp_path):
    store = RunStore.create(tmp_path, "exp", _config())
    store.save_checkpoint("last", {"step": 1, "tensor": torch.ones(2)})
    resumed = RunStore.resume(store.run_dir, _config())
    assert resumed.load_checkpoint()["step"] == 1
    with pytest.raises(ValueError, match="config hash"):
        RunStore.resume(store.run_dir, {**_config(), "case": "changed"})
