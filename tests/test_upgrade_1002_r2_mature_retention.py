"""Bounded mature recovery retention without changing global best selection."""

import json

import pytest
import torch

from phycoflow_reconstruction.training.checkpointing import PeriodicCheckpointManager
from phycoflow_reconstruction.training.run_store import RunStore, file_sha256


class DisabledPreview:
    enabled = False

    def due(self, step):
        return False

    def update(self, *args, **kwargs):
        return None


def setup_manager(path, *, r2=True, every_epochs=5):
    config = {"stage": "post_training", "checkpointing": {
        "every_epochs": every_epochs, "validation_every_epochs": 5,
        "selection_metric": "coherence_with_fidelity"}}
    if r2:
        config["evaluation"] = {"r2_protocol": {"enabled": True}}
    store = RunStore.create(path, "mature_retention", config)
    manager = PeriodicCheckpointManager(config, store=store, steps_per_epoch=38)
    model = torch.nn.Linear(1, 1)
    scores = {0: 1., 5: .4, 10: .5, 15: .6, 95: .7, 100: .9, 105: .95, 110: .8, 115: .3}
    manager.panel_evaluator = lambda step: {
        "eligible": True, "metric": scores[step // 38], "mse": 1., "metrics": {},
        "family_source_normalized_scores": {"A": scores[step // 38]}}
    return config, store, manager, model


def save_epoch(manager, model, epoch):
    manager.save({"model": model.state_dict()}, model=model, preview=DisabledPreview(),
                 global_step=epoch * 38, fallback_metric=0., force=epoch == 0)


def archive_epochs(manager):
    return [entry["step"] // 38 for entry in manager.feasible_archive]


def test_r2_archive_keeps_global_early_best_two_and_best_eligible_mature(tmp_path):
    _, store, manager, model = setup_manager(tmp_path)
    for epoch in (0, 5, 10, 15, 95):
        save_epoch(manager, model, epoch)
    assert archive_epochs(manager) == [5, 10, 15]
    save_epoch(manager, model, 100)
    assert archive_epochs(manager) == [5, 10, 100]
    assert store.load_checkpoint("best")["global_step"] == 5 * 38
    assert json.loads((store.run_dir / "evaluation/selected.json").read_text())["global_step"] == 5 * 38
    assert (store.run_dir / "checkpoints/feasible_0003800.pt").is_file()
    assert not (store.run_dir / "checkpoints/feasible_0000570.pt").exists()
    save_epoch(manager, model, 105)
    assert archive_epochs(manager) == [5, 10, 100]
    assert not (store.run_dir / "checkpoints/feasible_0003990.pt").exists()
    save_epoch(manager, model, 110)
    assert archive_epochs(manager) == [5, 10, 110]
    assert not (store.run_dir / "checkpoints/feasible_0003800.pt").exists()
    save_epoch(manager, model, 115)
    assert archive_epochs(manager) == [115, 5, 10]
    assert store.load_checkpoint("best")["global_step"] == 115 * 38
    assert len(list((store.run_dir / "checkpoints").glob("feasible_*.pt"))) == 3
    assert all(set(entry) == {"step", "metric", "mse", "family_scores", "checkpoint"}
               for entry in manager.feasible_archive)


def test_r2_mature_archive_recovers_committed_state_after_newer_unsaved_panel(tmp_path):
    config, store, manager, model = setup_manager(tmp_path, every_epochs=100)
    for epoch in (0, 5, 10, 15, 100):
        save_epoch(manager, model, epoch)
    committed = store.load_checkpoint("last")
    assert archive_epochs(manager) == [5, 10, 100]
    for epoch in (105, 110):
        save_epoch(manager, model, epoch)
    assert archive_epochs(manager) == [5, 10, 110]
    recovered = PeriodicCheckpointManager(config, store=store, steps_per_epoch=38)
    recovered.restore_coherence_selector(committed)
    assert archive_epochs(recovered) == [5, 10, 100]
    assert recovered.best_value == .4
    assert store.load_checkpoint("best")["global_step"] == 5 * 38
    assert len(recovered.feasible_archive) == 3
    assert store.load_checkpoint("feasible_0003800")["global_step"] == 100 * 38


@pytest.mark.parametrize("r2", [False, True])
def test_archive_global_ranking_unchanged_before_maturity(tmp_path, r2):
    _, _, manager, model = setup_manager(tmp_path, r2=r2)
    for epoch in (0, 5, 10, 15, 95):
        save_epoch(manager, model, epoch)
    assert archive_epochs(manager) == [5, 10, 15]
    assert manager.best_value == .4


def test_legacy_archive_stays_global_top_three_after_maturity(tmp_path):
    config, store, manager, model = setup_manager(tmp_path, r2=False)
    for epoch in (0, 5, 10, 15, 100, 105, 110):
        save_epoch(manager, model, epoch)
    assert archive_epochs(manager) == [5, 10, 15]
    assert manager.best_value == .4
    assert not (store.run_dir / "checkpoints/feasible_0003800.pt").exists()
    restored = PeriodicCheckpointManager(config, store=store, steps_per_epoch=38)
    restored.restore_coherence_selector(store.load_checkpoint("last"))
    assert archive_epochs(restored) == [5, 10, 15]
    assert restored.best_value == .4
    assert not (store.run_dir / "checkpoints/feasible_0004180.pt").exists()


def test_r2_restore_migrates_current_mature_weights_idempotently_without_pruning(tmp_path):
    config, store, old, model = setup_manager(tmp_path)
    old.retain_mature = False  # The already-running pre-repair R2 process.
    for epoch in (0, 5, 10, 15, 100):
        with torch.no_grad(): model.weight.fill_(epoch)
        save_epoch(old, model, epoch)
    committed = store.load_checkpoint("last")
    assert archive_epochs(old) == [5, 10, 15]
    early_path = store.run_dir / "checkpoints/feasible_0000570.pt"
    assert early_path.is_file()
    original = {"best_value": old.best_value, "best_fidelity_value": old.best_fidelity_value,
                "best_name": old.best_name}
    recovered = PeriodicCheckpointManager(config, store=store, steps_per_epoch=38)
    recovered.restore_coherence_selector(committed)
    assert archive_epochs(recovered) == [5, 10, 100]
    assert early_path.is_file()  # Only a later atomic last save may prune it.
    assert all(getattr(recovered, key) == value for key, value in original.items())
    assert store.load_checkpoint("best")["global_step"] == 5 * 38
    assert json.loads((store.run_dir / "evaluation/selected.json").read_text())["global_step"] == 5 * 38
    mature_path = store.run_dir / "checkpoints/feasible_0003800.pt"
    mature = store.load_checkpoint("feasible_0003800")
    assert torch.equal(mature["model"]["weight"], committed["model"]["weight"])
    assert mature["coherence_selector_state"]["feasible_archive"] == recovered.feasible_archive
    first_hash = file_sha256(mature_path)
    recovered.restore_coherence_selector(committed)
    assert archive_epochs(recovered) == [5, 10, 100]
    assert file_sha256(mature_path) == first_hash
    assert early_path.is_file()


@pytest.mark.parametrize("change", [
    lambda checkpoint: checkpoint.update(global_step=100 * 38 + 1),
    lambda checkpoint: checkpoint["coherence_selection_report"].update(eligible=False),
    lambda checkpoint: checkpoint["coherence_selection_report"].update(metric=float("inf")),
    lambda checkpoint: checkpoint["checkpoint_metric"].update(name="training_loss"),
])
def test_r2_restore_does_not_migrate_off_epoch_ineligible_nonfinite_or_stale_report(tmp_path, change):
    config, store, old, model = setup_manager(tmp_path)
    old.retain_mature = False
    for epoch in (0, 5, 10, 15, 100):
        save_epoch(old, model, epoch)
    committed = store.load_checkpoint("last")
    change(committed)
    recovered = PeriodicCheckpointManager(config, store=store, steps_per_epoch=38)
    recovered.restore_coherence_selector(committed)
    assert archive_epochs(recovered) == [5, 10, 15]
    assert not (store.run_dir / "checkpoints/feasible_0003800.pt").exists()
