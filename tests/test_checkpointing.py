"""Periodic recovery checkpoint cadence and alias contracts."""

import json

import torch

from phycoflow_reconstruction.training.checkpointing import PeriodicCheckpointManager
from phycoflow_reconstruction.training.run_store import RunStore


class _Preview:
    enabled = True

    def due(self, global_step):
        return global_step == 1

    def update(self, _model, *, global_step, force=False, checkpoint_path=None):
        if global_step == 1 or force:
            return {"metrics": {"mse_normalized": 0.25}}
        return None


def test_periodic_checkpoint_refreshes_last_alias_and_fixed_validation_best(tmp_path):
    config = {
        "stage": "base_training",
        "case": "fixture",
        "output": {},
        "checkpointing": {
            "enabled": True,
            "every_epochs": 5,
            "save_epoch_one": True,
        },
    }
    store = RunStore.create(tmp_path, "periodic", config)
    manager = PeriodicCheckpointManager(config, store=store, steps_per_epoch=1)
    model = torch.nn.Linear(2, 1)
    preview = _Preview()

    assert manager.due(1)
    assert not manager.due(2)
    assert manager.due(5)
    manager.save(
        {"model": model.state_dict()},
        model=model,
        preview=preview,
        global_step=1,
        fallback_metric=9.0,
    )
    best_step = store.load_checkpoint("best")["global_step"]
    manager.save(
        {"model": model.state_dict()},
        model=model,
        preview=preview,
        global_step=5,
        fallback_metric=0.01,
    )

    assert store.load_checkpoint("last")["global_step"] == 5
    assert store.load_checkpoint("latest")["global_step"] == 5
    assert store.load_checkpoint("best")["global_step"] == best_step == 1
    report = json.loads((store.run_dir / "evaluation/latest_checkpoint.json").read_text())
    assert report["global_step"] == 5
    assert report["best_updated"] is False


def test_forced_terminal_save_is_not_suppressed_when_periodic_saves_are_disabled(tmp_path):
    config = {
        "stage": "base_training",
        "case": "fixture",
        "output": {},
        "checkpointing": {"enabled": False},
    }
    store = RunStore.create(tmp_path, "terminal", config)
    manager = PeriodicCheckpointManager(config, store=store, steps_per_epoch=4)
    model = torch.nn.Linear(2, 1)
    preview = _Preview()

    assert not manager.due(4)
    manager.save(
        {"model": model.state_dict()},
        model=model,
        preview=preview,
        global_step=3,
        fallback_metric=1.0,
        force=True,
    )
    assert store.load_checkpoint("last")["global_step"] == 3
    assert store.load_checkpoint("best")["global_step"] == 3
