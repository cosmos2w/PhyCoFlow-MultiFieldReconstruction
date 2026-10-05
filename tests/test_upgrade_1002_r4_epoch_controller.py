"""Epoch PI must hold pressure, release saturation and recover partial epochs."""

from copy import deepcopy

import pytest
import torch

from phycoflow_reconstruction.training.fidelity_controller import (
    EPOCH_NATIVE_VERSION,
    FidelityController,
)


def make_controller(state=None, **settings):
    calibration = {
        "version": "endpoint_native_source_calibration_v2", "split": "train",
        "field_names": ["u"], "source_risks": [1., 1.],
        "native": {"version": "native_source_calibration_v2", "split": "train", "source_scale": .2},
    }
    return FidelityController(
        ["u"], {"version": EPOCH_NATIVE_VERSION, "native_budget": .02, **settings},
        calibration, state=state, steps_per_epoch=4,
    )


def test_pressure_fixed_inside_epoch_with_native_floor_and_exact_primal_derivative():
    c = make_controller()
    violations = torch.tensor([.3, .2, .4], dtype=torch.float64, requires_grad=True)
    initial = c.multipliers.clone()
    assert initial.tolist() == pytest.approx([0., 0., .02])
    for _ in range(3):
        loss, _, pressure = c.primal(violations)
        assert torch.equal(torch.autograd.grad(loss, violations)[0], initial)
        assert torch.equal(pressure, initial)
        c.advance(violations)
        assert torch.equal(c.multipliers, initial)
    c.advance(violations)
    assert c.completed_epochs == 1 and c.partial_count == 0
    assert (c.multipliers > initial).all()


def test_partial_epoch_recovery_and_coefficient_saturation_release_have_no_hidden_integrator():
    c = make_controller(epoch_ema_decay=0.)
    for _ in range(7):
        c.advance(torch.full((3,), 10.))
    restored = make_controller(c.state_dict(), epoch_ema_decay=0.)
    for values in ([10.] * 3, [-10.] * 3, [-10.] * 3, [-10.] * 3, [-10.] * 3, [.1] * 3):
        c.advance(torch.tensor(values))
        restored.advance(torch.tensor(values))
        assert torch.equal(c.multipliers, restored.multipliers)
        assert torch.equal(c.ema, restored.ema)
        assert torch.equal(c.partial_sum, restored.partial_sum)
        assert c.partial_count == restored.partial_count
    assert torch.equal(c.multipliers, c.minimum)
    assert c.multipliers[-1] > 0
    saved = deepcopy(c.state_dict())
    saved["epoch_pi"]["partial_count"] += 1
    with pytest.raises(ValueError, match="partial-epoch"):
        make_controller(saved, epoch_ema_decay=0.)


def test_v3_rejects_old_instantaneous_pressure_or_incompatible_epoch_recovery():
    with pytest.raises(ValueError, match="dual/rho"):
        make_controller(augmented_rho=1.)
    c = make_controller()
    saved = c.state_dict()
    saved["epoch_pi"]["steps_per_epoch"] = 3
    with pytest.raises(ValueError, match="exposure"):
        make_controller(saved)
    with pytest.raises(ValueError, match="cap"):
        make_controller(native_min_weight=.5, native_max_weight=.1)
