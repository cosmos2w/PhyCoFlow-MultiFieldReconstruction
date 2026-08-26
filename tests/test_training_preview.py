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
