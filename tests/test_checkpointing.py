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


class _FidelityPreview(_Preview):
    def update(self, _model, *, global_step, force=False, checkpoint_path=None):
        if self.due(global_step) or force:
            mse = {1: 0.4, 2: 0.2}.get(global_step, 0.3)
            return {
                "validation": {
                    "global_step": global_step,
                    "training_epoch": float(global_step),
                    "loss": 10.0 - global_step,
                    "components": {},
                },
                "reconstruction": {"metrics": {"mse_normalized": mse}},
            }
        return None


def test_step_recovery_cadence_does_not_trigger_extra_topology_validation(tmp_path):
    config = {'stage': 'post_training', 'checkpointing': {'every_epochs': 1,
        'every_steps': 3, 'validation_every_epochs': 1, 'selection_metric': 'topology_with_fidelity'}}
    store = RunStore.create(tmp_path, 'step_recovery', config)
    manager = PeriodicCheckpointManager(config, store=store, steps_per_epoch=10)
    assert manager.due(3) and manager.last_due(3)
    assert not manager.due(7)
    assert manager.due(10)
    manager.panel_evaluator = lambda step: (_ for _ in ()).throw(AssertionError('unexpected validation'))
    model = torch.nn.Linear(1, 1)
    manager.save({'model': model.state_dict()}, model=model, preview=_Preview(),
                 global_step=3, fallback_metric=0.)
    assert torch.load(store.run_dir / 'checkpoints/last.pt', weights_only=True)['global_step'] == 3
    assert not manager.last_best_checked


def test_topology_panel_gate_source_fallback_and_independent_fidelity_resume(tmp_path):
    config = {"stage": "post_training", "checkpointing": {
        "every_epochs": 1, "selection_metric": "topology_with_fidelity"}}
    store = RunStore.create(tmp_path, "topology_panel", config)
    manager = PeriodicCheckpointManager(config, store=store, steps_per_epoch=1)
    model = torch.nn.Linear(1, 1)
    rows = {0: (1., 1., True), 1: (.5, 1.2, False), 2: (.8, .95, True),
            3: (.7, .9, False), 4: (.85, 1., True)}
    def evaluate(step):
        topology, mse, eligible = rows[step]
        return {"metric": topology, "mse": mse, "eligible": eligible, "metrics": {}}
    manager.panel_evaluator = evaluate
    for step in range(4):
        manager.save({"model": model.state_dict()}, model=model, preview=_Preview(),
                     global_step=step, fallback_metric=0., force=step == 0)
        assert manager.last_best_checked
        best = torch.load(store.run_dir / "checkpoints/best.pt", weights_only=True)
        assert best["global_step"] == (0 if step < 2 else 2)
    assert torch.load(store.run_dir / "checkpoints/best_fidelity.pt", weights_only=True)["global_step"] == 3
    restored = PeriodicCheckpointManager(config, store=store, steps_per_epoch=1)
    assert restored.best_value == .8
    assert restored.best_fidelity_value == .9
    restored.panel_evaluator = evaluate
    restored.save({"model": model.state_dict()}, model=model, preview=_Preview(),
                  global_step=4, fallback_metric=0.)
    assert json.loads((store.run_dir / "evaluation/selected.json").read_text())["global_step"] == 2


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


def test_reconstruction_fidelity_selection_writes_explicit_best_alias(tmp_path):
    config = {
        "stage": "post_training",
        "case": "fixture",
        "output": {},
        "checkpointing": {
            "enabled": True,
            "every_epochs": 1,
            "selection_metric": "reconstruction_mse",
        },
    }
    store = RunStore.create(tmp_path, "fidelity", config)
    manager = PeriodicCheckpointManager(config, store=store, steps_per_epoch=1)
    model = torch.nn.Linear(2, 1)
    preview = _FidelityPreview()

    for step in (1, 2):
        manager.save(
            {"model": model.state_dict()},
            model=model,
            preview=preview,
            global_step=step,
            fallback_metric=0.01,
        )

    assert store.load_checkpoint("best")["global_step"] == 2
    assert store.load_checkpoint("best_fidelity")["global_step"] == 2
    status = json.loads((store.run_dir / "evaluation/checkpoint_status.json").read_text())
    assert status["selection_metric"]["name"] == "fixed_validation_reconstruction_mse"
    assert status["best_fidelity_updated"] is True


def test_topology_recovery_and_panel_cadences_are_independent(tmp_path):
    config = {"stage": "post_training", "checkpointing": {"every_epochs": 1,
        "validation_every_epochs": 3, "selection_metric": "topology_with_fidelity"}}
    store = RunStore.create(tmp_path, "separate_cadence", config)
    manager = PeriodicCheckpointManager(config, store=store, steps_per_epoch=1)
    model = torch.nn.Linear(1, 1)
    calls = []
    def panel(step):
        calls.append(step)
        return {"metric": 1 / (1 + step), "mse": 1., "eligible": True, "metrics": {}}
    manager.panel_evaluator = panel
    for step in range(5):
        manager.save({"model": model.state_dict()}, model=model, preview=_Preview(),
                     global_step=step, fallback_metric=0., force=step in {0, 4})
        assert store.load_checkpoint("last")["global_step"] == step
        assert manager.last_best_checked == (step in {0, 3, 4})
    assert calls == [0, 3, 4]
    # A validation event also triggers a save/check when recovery isn't due.
    config["checkpointing"]["every_epochs"] = 5
    manager = PeriodicCheckpointManager(config, store=store, steps_per_epoch=1)
    assert not manager.last_due(6)
    assert manager.due_for_preview_or_checkpoint(6, _Preview())
