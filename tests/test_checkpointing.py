"""Periodic recovery checkpoint cadence and alias contracts."""

import json

import torch

from phycoflow_reconstruction.training.checkpointing import PeriodicCheckpointManager
from phycoflow_reconstruction.training.run_store import RunStore


class _Preview:
    enabled = True
    loss_steps = frozenset((1, 2, 3, 4, 5, 10))

    def due(self, global_step):
        return global_step in self.loss_steps

    def update(self, _model, *, global_step, force=False, checkpoint_path=None):
        if self.due(global_step) or force:
            value = {1: 0.25, 2: 0.5, 3: 0.75, 4: 0.5, 5: 0.5, 10: 0.2}.get(
                global_step, 0.25
            )
            return {
                "validation": {
                    "global_step": global_step,
                    "training_epoch": float(global_step),
                    "loss": value,
                    "components": {"data_mse": value},
                },
                "reconstruction": None,
            }
        return None


def test_periodic_checkpoint_refreshes_last_and_fixed_validation_best(tmp_path):
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
    assert store.load_checkpoint("best")["global_step"] == best_step == 1
    assert not (store.run_dir / "checkpoints/latest.pt").exists()
    report = json.loads((store.run_dir / "evaluation/checkpoint_status.json").read_text())
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


def test_explicit_epoch_schedule_keeps_immutable_milestone_checkpoints(tmp_path):
    config = {
        "stage": "base_training",
        "case": "fixture",
        "output": {},
        "checkpointing": {
            "enabled": True,
            "every_epochs": 20,
            "epochs": [1, 2],
            "save_epoch_one": True,
        },
    }
    store = RunStore.create(tmp_path, "milestones", config)
    manager = PeriodicCheckpointManager(config, store=store, steps_per_epoch=2)
    model = torch.nn.Linear(2, 1)
    preview = _Preview()

    manager.save(
        {"model": model.state_dict()},
        model=model,
        preview=preview,
        global_step=2,
        fallback_metric=1.0,
    )
    manager.save(
        {"model": model.state_dict()},
        model=model,
        preview=preview,
        global_step=4,
        fallback_metric=0.5,
    )
    assert store.load_checkpoint("epoch_001")["global_step"] == 2
    assert store.load_checkpoint("epoch_002")["global_step"] == 4
    assert store.load_checkpoint("last")["global_step"] == 2
    manifest = json.loads((store.run_dir / "run_manifest.json").read_text())
    assert "epoch_001" in manifest["checkpoint_hashes"]
    assert "epoch_002" in manifest["checkpoint_hashes"]


def test_validation_can_update_best_without_refreshing_last(tmp_path):
    config = {
        "stage": "base_training",
        "case": "fixture",
        "output": {},
        "checkpointing": {
            "enabled": True,
            "every_epochs": 10,
            "save_epoch_one": False,
        },
    }
    store = RunStore.create(tmp_path, "independent", config)
    manager = PeriodicCheckpointManager(config, store=store, steps_per_epoch=1)
    model = torch.nn.Linear(2, 1)

    saved = manager.save(
        {"model": model.state_dict()},
        model=model,
        preview=_Preview(),
        global_step=1,
        fallback_metric=9.0,
    )

    assert saved is not None and saved[0] is None and saved[1] is not None
    assert store.load_checkpoint("best")["global_step"] == 1
    assert not (store.run_dir / "checkpoints/last.pt").exists()
