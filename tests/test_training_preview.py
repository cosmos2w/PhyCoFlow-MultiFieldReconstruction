"""Unit checks for quantitative training-preview annotations."""

from types import SimpleNamespace

import numpy as np
import torch

from phycoflow_reconstruction.contracts import LossBundle
from phycoflow_reconstruction.training.preview import (
    TrainingReconstructionPreview,
    _absolute_error_title,
    _relative_l2_error,
)
from phycoflow_reconstruction.training.run_store import RunStore


def test_disabled_preview_is_safe_between_sparse_checkpoints(tmp_path):
    from phycoflow_reconstruction.training.checkpointing import PeriodicCheckpointManager
    config = {"evaluation": {"preview": {"enabled": False}},
              "checkpointing": {"every_epochs": 2, "save_epoch_one": True}}
    store = SimpleNamespace(run_dir=tmp_path)
    preview = TrainingReconstructionPreview(config, store=store, steps_per_epoch=9,
                                            device=torch.device("cpu"))
    manager = PeriodicCheckpointManager(config, store=store, steps_per_epoch=9)
    for step in (1, 9, 10, 18, 27):
        assert not preview.due_loss(step)
        assert not preview.due_reconstruction(step)
        assert not preview.due(step)
        assert manager.due_for_preview_or_checkpoint(step, preview) == (step in (9, 18))
    assert preview.update(torch.nn.Linear(1, 1), global_step=18, force=True) is None
    assert preview.last_validation_report is None
    assert preview.dataset is None and preview.batch is None
    assert not preview.output_dir.exists()
    preview.close()


def test_relative_l2_error_uses_field_reference_norm():
    truth = np.asarray([3.0, 4.0])
    estimate = np.asarray([0.0, 0.0])

    value = _relative_l2_error(estimate, truth)

    assert value == 1.0
    assert _absolute_error_title(value) == "Absolute error\nRelative $L_2$ = 1.000e+00"


def test_relative_l2_error_marks_zero_reference_as_unavailable():
    value = _relative_l2_error(np.ones(4), np.zeros(4))

    assert value is None
    assert _absolute_error_title(value) == "Absolute error\nRelative $L_2$ = N/A"


def test_validation_loss_and_reconstruction_have_independent_cadences():
    preview = TrainingReconstructionPreview.__new__(TrainingReconstructionPreview)
    preview.enabled = True
    preview.steps_per_epoch = 2
    preview.loss_every_epochs = 10
    preview.reconstruct_every_epochs = 500

    assert preview.due_loss(20)
    assert not preview.due_reconstruction(20)
    assert preview.due_reconstruction(1000)
    assert preview.due(20)
    assert not preview.due(18)


def test_validation_loss_is_seeded_and_does_not_advance_training_rng(tmp_path):
    class RandomLossModel(torch.nn.Module):
        def training_loss(self, _batch):
            value = torch.rand(())
            return LossBundle(value, {"random_component": value})

    config = {"stage": "base_training", "case": "fixture", "output": {}}
    preview = TrainingReconstructionPreview.__new__(TrainingReconstructionPreview)
    preview.batch = SimpleNamespace(sample_ids=("validation:0",))
    preview.device = torch.device("cpu")
    preview.settings = {"seed": 123}
    preview.steps_per_epoch = 4
    preview.store = RunStore.create(tmp_path, "validation", config)
    model = RandomLossModel()

    torch.manual_seed(999)
    expected_next = torch.rand(())
    torch.manual_seed(999)
    first = preview._validation_loss(model, global_step=40)
    actual_next = torch.rand(())
    second = preview._validation_loss(model, global_step=80)

    assert first["loss"] == second["loss"]
    assert first["components"] == second["components"]
    assert torch.equal(actual_next, expected_next)
