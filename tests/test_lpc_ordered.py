"""Focused contracts for the default-off LPC-ordered scalar helper."""

import copy
import json

import pytest
import torch
from torch import nn

from phycoflow_reconstruction.training.fidelity_budget import ExternalPolicy, FidelityBudget
from phycoflow_reconstruction.training.lpc_ordered import (
    LPCOrderedReference,
    LPCOrderedState,
    validate_lpc_ordered,
)

_SOURCE_SHA = "0123456789abcdef" * 4


class _ParameterModel(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.value = parameter


def _reference(*, a0=2.0, b0=4.0, t0=8.0, floor=1e-8, source_sha=_SOURCE_SHA):
    return LPCOrderedReference.from_means(
        {
            "global_distribution": a0,
            "cross_spectrum": b0,
            "topology": t0,
        },
        source_sha256=source_sha,
        denominator_floor=floor,
    )


def _losses(parameter, *, a=None, b=None, t=None):
    return {
        "global_distribution": 2.0 + parameter if a is None else a,
        "cross_spectrum": 4.0 + 2.0 * parameter if b is None else b,
        "topology": 8.0 + 3.0 * parameter if t is None else t,
    }


def test_lpc_primary_preserves_negative_signed_surrogate_and_external_c0_scale():
    parameter = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    reference = _reference()
    state = LPCOrderedState(reference, dual_lr=0.01, device="cpu")
    state.mu_a = 1.0
    losses = _losses(
        parameter,
        a=2.0 - parameter,
        b=4.0 + parameter * 0.0,
        t=0.2 + 0.2 * parameter,
    )
    budget = FidelityBudget.from_reference(("field",), (1.0,), 5.0)
    surrogate = state.primary(losses, objective_scale=budget.objective_scale)
    assert surrogate.dtype == torch.float64
    assert surrogate.item() == pytest.approx(-2.25)

    policy = ExternalPolicy({"method": "lagrangian"}, budget, "cpu")
    optimizer = torch.optim.SGD((parameter,), lr=0.0)
    risks = parameter.square().reshape(1) * 0.0
    diagnostics = policy.update(_ParameterModel(parameter), optimizer, surrogate, risks, None)

    # J = T/T0 + mu_A * (A-A0)/A0 = -0.45 at p=1.
    # dJ/dp = 0.2/8 - 1/2 = -0.475. C0 is removed once by ExternalPolicy.
    assert diagnostics["scaled_primary_scalar"] == pytest.approx(-0.45)
    assert diagnostics["optimized_scalar"] == pytest.approx(-0.45)
    assert parameter.grad.item() == pytest.approx(-0.475)
    assert state.mu_a == 1.0  # the primary graph does not attach multiplier gradients


def test_reference_floor_only_scales_h_values_and_does_not_raise_bounds():
    reference = _reference(a0=1e-10, b0=0.0, t0=0.0, floor=1e-8)
    assert reference.a_denominator == reference.b_denominator == reference.t_denominator == 1e-8
    saved = reference.state_dict()
    assert saved["no_regression_bounds"] == {
        "global_distribution": 1e-10,
        "cross_spectrum": 0.0,
    }

    parameter = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    losses = _losses(
        parameter,
        a=parameter + 1.1e-10,
        b=parameter + 1.0e-10,
        t=parameter + 2.0e-8,
    )
    state = LPCOrderedState(reference)
    diagnostics = state.accept(losses)
    assert diagnostics["violations"]["global_distribution"] == pytest.approx(0.001)
    assert diagnostics["violations"]["cross_spectrum"] == pytest.approx(0.01)
    assert state.mu_a == pytest.approx(1e-5)
    assert state.mu_b == pytest.approx(1e-4)


def test_ordered_objective_keeps_generic_nonzero_gradient_without_rank_shortcut():
    parameter = nn.Parameter(torch.tensor(0.3, dtype=torch.float64))
    losses = _losses(parameter)
    state = LPCOrderedState(_reference())
    state.mu_a = 0.2
    state.mu_b = 0.1
    objective = state.primary(losses, objective_scale=3.0)
    objective.backward()
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad)
    assert parameter.grad.item() != 0.0
    assert not hasattr(state, "_parameter_rank")


def test_accept_projects_duals_after_successful_action_and_reports_json_safe_state():
    parameter = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    state = LPCOrderedState(_reference(), dual_lr=0.5)
    first = state.accept(
        _losses(parameter, a=3.0 + parameter * 0.0, b=2.0 + parameter * 0.0)
    )
    assert first["accepted_update"] == 1
    assert first["multipliers_before"] == {"global_distribution": 0.0, "cross_spectrum": 0.0}
    assert first["multipliers_after"] == pytest.approx(
        {"global_distribution": 0.25, "cross_spectrum": 0.0}
    )
    assert first["violations"] == pytest.approx(
        {"global_distribution": 0.5, "cross_spectrum": -0.5}
    )
    restored = LPCOrderedState(_reference(), dual_lr=0.5)
    restored.load_state_dict(json.loads(json.dumps(state.state_dict())))
    assert restored.state_dict() == state.state_dict()
    assert restored.completed_updates == 1


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda state: state.update(reference=_reference(source_sha="f" * 64).state_dict()), "SOURCE reference"),
        (lambda state: state["multipliers"].__setitem__("global_distribution", float("nan")), "finite and non-negative"),
        (lambda state: state.__setitem__("completed_updates", True), "completed_updates is invalid"),
    ],
)
def test_resume_rejects_wrong_reference_nonfinite_dual_and_invalid_cursor(mutation, message):
    source = LPCOrderedState(_reference()).state_dict()
    bad = copy.deepcopy(source)
    if mutation.__code__.co_argcount == 1:
        mutation(bad)
    with pytest.raises(ValueError, match=message):
        LPCOrderedState(_reference()).load_state_dict(bad)


def test_default_off_validation_is_a_noop_and_ordered_profile_is_narrow():
    validate_lpc_ordered({})
    validate_lpc_ordered({"optimization": {"lr": 1e-4}})
    validate_lpc_ordered({"optimization": {"lpc_ordered": None, "lr": 1e-4}})

    valid = {
        "optimization": {
            "lpc_ordered": {"dual_lr": 0.01, "denominator_floor": 1e-8},
            "multitask": {
                "family_scalarization": {"method": "fixed_sum"},
                "external_policy": {
                    "method": "lagrangian",
                    "lagrangian": {"penalty_interval_updates": 1},
                },
            },
            "fast_lagrangian": {"family_probability": 1.0, "coefficient_interval": 1},
        },
        "coherence": {
            "schedule": {"start_epoch": 1, "every_n_steps": 1, "weight_warmup_epochs": 0},
            "family_balance": {"mode": "none", "gradient_diagnostics_every_epochs": 0},
        },
    }
    validate_lpc_ordered(valid)
    for change, message in (
        (lambda cfg: cfg["optimization"].update(fast_lagrangian={"family_probability": 0.5}), "family_probability=1"),
        (lambda cfg: cfg["optimization"]["multitask"].update(coherence_aggregation={"method": "torchjd_config"}), "without internal aggregation"),
        (lambda cfg: cfg["coherence"]["family_balance"].update(mode="initial_grad_norm"), "dynamic family calibration"),
        (lambda cfg: cfg["optimization"]["lpc_ordered"].update(extra=True), "only dual_lr and denominator_floor"),
    ):
        candidate = copy.deepcopy(valid)
        change(candidate)
        with pytest.raises(ValueError, match=message):
            validate_lpc_ordered(candidate)


def test_reference_round_trip_rejects_changed_bounds_and_source_identity():
    original = _reference()
    state = json.loads(json.dumps(original.state_dict()))
    assert LPCOrderedReference.from_state_dict(state) == original
    state["no_regression_bounds"]["global_distribution"] = 99.0
    with pytest.raises(ValueError, match="inconsistent fixed bounds"):
        LPCOrderedReference.from_state_dict(state)
