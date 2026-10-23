from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch import nn

from phycoflow_reconstruction.training.execution_strategy.audit import (
    _flat_parameter_gradient,
    nested_parameter_variance,
    preserve_rng_and_modes,
)
from phycoflow_reconstruction.training.execution_strategy.bottleneck import profile_one_update


def test_nested_parameter_variance_matches_exact_correlated_outer_and_inner_enumeration():
    gradients = {
        "A": torch.tensor([1.0, 0.0], dtype=torch.float64),
        "B": torch.tensor([0.0, 2.0], dtype=torch.float64),
        "T": torch.tensor([1.0, 1.0], dtype=torch.float64),
    }
    delta_t = torch.tensor([0.3, 0.4], dtype=torch.float64)
    inner_variance = float(delta_t.square().sum())
    probabilities = {"A": 0.5, "B": 0.5, "T": 1.0}
    support = ((0.5, ("A", "T")), (0.5, ("B", "T")))

    full_gradient = sum(gradients.values())
    exact_variance = 0.0
    for mass, included in support:
        for sign in (-1.0, 1.0):
            estimate = gradients["T"] + sign * delta_t
            if "A" in included:
                estimate = estimate + 2.0 * gradients["A"]
            if "B" in included:
                estimate = estimate + 2.0 * gradients["B"]
            exact_variance += 0.5 * mass * float((estimate - full_gradient).square().sum())

    computed = nested_parameter_variance(
        gradients,
        {"T": inner_variance},
        probabilities,
        support=support,
    )
    assert computed == pytest.approx(exact_variance, rel=1e-12, abs=1e-12)
    assert computed == pytest.approx(5.25)


def test_nested_parameter_variance_matches_outer_times_inner_probability_enumeration():
    gradients = {
        "A": torch.tensor([1.0, 0.0], dtype=torch.float64),
        "B": torch.tensor([0.0, 2.0], dtype=torch.float64),
        "T": torch.tensor([1.0, 1.0], dtype=torch.float64),
    }
    inner_units = (
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        torch.tensor([0.0, 1.0], dtype=torch.float64),
    )
    inner_variance = sum(
        0.5 * float((2.0 * unit - gradients["T"]).square().sum()) for unit in inner_units
    )
    probabilities = {"A": 0.5, "B": 0.5, "T": 0.5}
    support = ((0.5, ("A", "T")), (0.5, ("B",)))
    full_gradient = sum(gradients.values())

    exact_variance = 0.0
    for outer_mass, included in support:
        if "A" in included:
            for unit in inner_units:
                estimate = 2.0 * gradients["A"] + 4.0 * unit
                exact_variance += (
                    outer_mass * 0.5 * float((estimate - full_gradient).square().sum())
                )
        else:
            estimate = 2.0 * gradients["B"]
            exact_variance += outer_mass * float((estimate - full_gradient).square().sum())

    computed = nested_parameter_variance(
        gradients,
        {"T": inner_variance},
        probabilities,
        support=support,
    )
    assert computed == pytest.approx(exact_variance, rel=1e-12, abs=1e-12)
    assert computed == pytest.approx(9.0)


def test_two_family_variance_uses_outer_weights_for_full_and_inner_gradients():
    parameter = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    parameters = (parameter,)
    family_weights = {"A": 0.25, "B": 2.0}

    # A has a two-point conditional estimator (0 or 4*parameter, each with
    # probability 1/2) for its full scalar loss 2*parameter. B is sampled as
    # an outer family with p=1/2. The unequal outer weights are part of the
    # optimized scalar objective and therefore must also scale both variance
    # calculations.
    full = {
        "A": _flat_parameter_gradient(
            2.0 * parameter, parameters, multiplier=family_weights["A"]
        ),
        "B": _flat_parameter_gradient(
            3.0 * parameter, parameters, multiplier=family_weights["B"]
        ),
    }
    conditional_a = [
        _flat_parameter_gradient(
            estimate * parameter, parameters, multiplier=family_weights["A"]
        )
        for estimate in (0.0, 4.0)
    ]
    inner_variance_a = sum(
        0.5 * float((estimate - full["A"]).square().sum())
        for estimate in conditional_a
    )

    full_objective_gradient = torch.autograd.grad(
        family_weights["A"] * 2.0 * parameter
        + family_weights["B"] * 3.0 * parameter,
        parameters,
    )[0]
    assert torch.equal(full["A"] + full["B"], full_objective_gradient)

    exact_variance = 0.0
    for a_gradient in conditional_a:
        for included_b in (False, True):
            estimate = a_gradient + (2.0 * full["B"] if included_b else 0.0)
            exact_variance += 0.25 * float((estimate - full_objective_gradient).square().sum())
    computed = nested_parameter_variance(
        full,
        {"A": inner_variance_a},
        {"B": 0.5},
    )
    assert inner_variance_a == pytest.approx(0.25)
    assert computed == pytest.approx(exact_variance, rel=1e-12, abs=1e-12)
    assert computed == pytest.approx(36.25)


class _NoiseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(8)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.batch_norm(values))


def test_preserve_rng_and_modes_restores_real_cpu_streams_and_mixed_module_modes():
    model = _NoiseModel()
    model.train()
    model.dropout.eval()
    expected_modes = [module.training for module in model.modules()]
    random.seed(17)
    np.random.seed(19)
    torch.manual_seed(23)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    buffers = {name: value.clone() for name, value in model.named_buffers()}

    with preserve_rng_and_modes(model):
        model.train()
        torch.rand(3)
        random.random()
        np.random.random()
        model(torch.arange(32, dtype=torch.float32).reshape(4, 8))

    assert [module.training for module in model.modules()] == expected_modes
    assert all(torch.equal(value, dict(model.named_buffers())[name]) for name, value in buffers.items())
    assert random.getstate() == python_state
    restored_numpy = np.random.get_state()
    assert restored_numpy[0] == numpy_state[0]
    assert np.array_equal(restored_numpy[1], numpy_state[1])
    assert restored_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_nested_parameter_variance_rejects_negative_inner_variance():
    with pytest.raises(ValueError, match="inner variances"):
        nested_parameter_variance(
            {"A": torch.tensor([1.0], dtype=torch.float64)},
            {"A": -0.25},
            {"A": 1.0},
        )


class _LinearFamily(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values).square().mean()


def test_profile_one_update_returns_compact_real_cpu_operator_and_callback_scopes():
    model = nn.Linear(4, 4)
    family = _LinearFamily()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = torch.ones(8, 4)
    calls = 0

    def update(*, scope):
        nonlocal calls
        calls += 1
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)
        with scope("dense_fidelity"):
            fidelity = (prediction - 1.0).square().mean()
        family_loss = family(prediction)
        with scope("policy_backward"):
            (fidelity + family_loss).backward()
        with scope("adamw_step"):
            optimizer.step()
        with scope("dual_update"):
            _ = float(fidelity.detach())
        return fidelity.detach()

    profile = profile_one_update(model, {"topology": family}, update, "cpu")

    assert calls == 1
    assert profile["device"] == "cpu"
    assert profile["measured_update_wall_seconds"] > 0
    assert profile["measurement_limits"]["profiled_update_count"] == 1
    assert profile["measurement_limits"]["trace_exported"] is False
    assert profile["measurement_limits"]["scope_times_are_inclusive_and_must_not_be_summed"]
    scopes = profile["scopes"]
    assert scopes["shared_model_rollout_forward"]["calls"] == 1
    assert scopes["family.topology"]["calls"] == 1
    for name in ("dense_fidelity", "policy_backward", "adamw_step", "dual_update"):
        assert scopes[f"scope.{name}"]["calls"] == 1
        assert scopes[f"scope.{name}"]["inclusive_cpu_seconds"] >= 0
    assert profile["top_operators_self_cpu"]
    assert profile["top_operators_self_cuda"] == []
    assert len(profile["top_operators_self_cpu"]) <= 12
