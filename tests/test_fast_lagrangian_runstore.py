"""Atomic recovery through a campaign-style local link; no model training."""

import json

import pytest
import torch

from phycoflow_reconstruction.training.run_store import RunStore


def test_link_runstore_atomic_checkpoint_and_history_recovery(tmp_path):
    payload = tmp_path / "payload"
    payload.mkdir()
    case = tmp_path / "case"
    (case / "runs").mkdir(parents=True)
    (case / "runs" / "campaign").symlink_to(payload, target_is_directory=True)
    config = {"stage": "post_training", "optimization": {"epochs": 10}}
    store = RunStore.create(case, "campaign/T0", config)
    assert store.run_dir.is_relative_to(payload)
    store.save_checkpoint("last", {"global_step": 2, "state": torch.tensor([3.0])})
    history = store.run_dir / "metrics/fast_lagrangian_updates.jsonl"
    history.write_text("".join(json.dumps({"step": i}) + "\n" for i in [1, 2, 3]) + "{")
    restored = RunStore.resume(store.run_dir, config)
    assert restored.load_checkpoint()["global_step"] == 2
    restored.recover_metric_histories(2)
    assert [json.loads(x)["step"] for x in history.read_text().splitlines()] == [1, 2]
    assert not list(store.run_dir.rglob("*.tmp"))
    with pytest.raises(ValueError, match="config hash"):
        RunStore.resume(store.run_dir, {"stage": "post_training", "optimization": {"epochs": 100}})
