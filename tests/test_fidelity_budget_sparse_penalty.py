"""Sparse Lagrangian penalty scheduling and resume contracts."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.config.validate import validate_config
from phycoflow_reconstruction.training.fidelity_budget import ExternalPolicy, FidelityBudget
from phycoflow_reconstruction.training.multitask import OuterAggregator

_RUN11_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "cases/turbulent_combustion/configs/posttrain/fidelity_budget/run11_config_lagrangian_1000ep.yaml"
)
_SPARSE_INTERVAL = 380


class _ScalarPair(nn.Module):
    def __init__(self, x: float = 1.5, y: float = 2.0):
        super().__init__()
        self.x = nn.Parameter(torch.tensor(x, dtype=torch.float64))
        self.y = nn.Parameter(torch.tensor(y, dtype=torch.float64))


class _ConfigPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = nn.Parameter(torch.tensor(0.8, dtype=torch.float64))
        self.y = nn.Parameter(torch.tensor(0.4, dtype=torch.float64))


def _lagrangian_settings(interval: int | None = _SPARSE_INTERVAL) -> dict[str, Any]:
    lagrangian: dict[str, Any] = {"dual_lr": 0.01}
    if interval is not None:
        lagrangian["penalty_interval_updates"] = interval
    return {
        "method": "lagrangian",
        "vector_reduction": "sum",
        "objective_scale": "source_mean",
        "lagrangian": lagrangian,
    }


def _scalar_reference() -> FidelityBudget:
    return FidelityBudget.from_reference(("field",), (1.0,), 1.0)


def _three_field_reference() -> FidelityBudget:
    return FidelityBudget.from_reference(("A", "B", "T"), (0.6, 0.1, 1.0), 2.0)


def _set_cursor_and_multiplier(policy: ExternalPolicy, completed_updates: int, values: list[float]) -> None:
    state = policy.state_dict()
    state["completed_updates"] = completed_updates
    state["multipliers"] = values
    policy.load_state_dict(state)


def _scalar_step(
    policy: ExternalPolicy,
    model: _ScalarPair,
    optimizer: torch.optim.Optimizer,
    risk_hook: list[torch.Tensor] | None = None,
) -> dict[str, Any]:
    risk = model.y.square().reshape(1)
    if risk_hook is not None:
        risk.register_hook(lambda gradient: risk_hook.append(gradient.detach().clone()) or gradient)
    return policy.update(model, optimizer, model.x.square(), risk, None)


def _config_losses(model: _ConfigPair) -> dict[str, torch.Tensor]:
    x, y = model.x, model.y
    return {
        "A": x.square() + 0.1 * y,
        "B": y.square() + 0.2 * x,
        "T": (x - y).square() + 0.1 * x * y,
    }


def _config_risks(model: _ConfigPair) -> torch.Tensor:
    return torch.stack((model.x.square(), model.y.square(), (model.x + model.y).square()))


def _config_step(
    policy: ExternalPolicy,
    model: _ConfigPair,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    return policy.update_with_coherence_aggregation(
        model,
        optimizer,
        _config_losses(model),
        _config_risks(model),
        None,
        OuterAggregator({"method": "torchjd_config"}),
    )


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    elif isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


@pytest.mark.parametrize("bad_value", [0, -1, 1.5, 380.0, True, "380"])
def test_sparse_penalty_interval_config_requires_a_positive_integer(bad_value):
    config = deepcopy(load_config(_RUN11_CONFIG))
    config["optimization"]["multitask"]["external_policy"]["lagrangian"][
        "penalty_interval_updates"
    ] = bad_value

    with pytest.raises((TypeError, ValueError), match="penalty_interval_updates"):
        validate_config(config)


def test_sparse_penalty_has_only_one_interval_setting():
    config = deepcopy(load_config(_RUN11_CONFIG))
    config["optimization"]["multitask"]["external_policy"]["lagrangian"][
        "penalty_active_updates"
    ] = 1

    with pytest.raises(ValueError, match="unknown.*penalty_active_updates"):
        validate_config(config)


@pytest.mark.parametrize(
    ("next_update", "expected_active"),
    [
        (379, False),
        (380, True),
        (381, False),
        (759, False),
        (760, True),
        (761, False),
    ],
)
def test_scalar_penalty_and_dual_update_boundaries(next_update, expected_active):
    pytest.importorskip("cooper")

    policy = ExternalPolicy(_lagrangian_settings(), _scalar_reference(), "cpu")
    _set_cursor_and_multiplier(policy, next_update - 1, [0.2])
    model = _ScalarPair()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    risk_traversals: list[torch.Tensor] = []
    initial_parameters = torch.stack((model.x.detach(), model.y.detach()))
    model_steps: list[bool] = []
    dual_steps: list[bool] = []
    original_model_step = optimizer.step
    original_dual_step = policy._dual_optimizer.step

    def model_step(*args, **kwargs):
        result = original_model_step(*args, **kwargs)
        model_steps.append(True)
        return result

    def dual_step(*args, **kwargs):
        assert len(model_steps) == len(dual_steps) + 1
        assert not torch.equal(torch.stack((model.x.detach(), model.y.detach())), initial_parameters)
        dual_steps.append(True)
        return original_dual_step(*args, **kwargs)

    optimizer.step = model_step
    policy._dual_optimizer.step = dual_step
    result = _scalar_step(policy, model, optimizer, risk_traversals)

    assert result["fidelity_penalty_active"] is expected_active
    assert result["fidelity_penalty_interval_updates"] == _SPARSE_INTERVAL
    assert result["completed_updates"] == next_update
    assert len(risk_traversals) == int(expected_active)
    assert model_steps == [True]
    assert dual_steps == [True]
    assert result["multipliers_after"][0] > result["multipliers_before"][0]


@pytest.mark.parametrize(
    ("completed_updates", "expected_active"),
    [(378, False), (379, True), (380, False)],
)
def test_config_penalty_gradient_is_skipped_off_interval_but_dual_steps_each_update(
    completed_updates, expected_active
):
    pytest.importorskip("torchjd")
    pytest.importorskip("cooper")

    policy = ExternalPolicy(_lagrangian_settings(), _three_field_reference(), "cpu")
    _set_cursor_and_multiplier(policy, completed_updates, [0.2, 0.3, 0.4])
    model = _ConfigPair()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    model_steps: list[bool] = []
    dual_steps: list[bool] = []
    before_update = torch.stack((model.x.detach(), model.y.detach()))
    original_model_step = optimizer.step
    original_dual_step = policy._dual_optimizer.step

    def model_step(*args, **kwargs):
        result = original_model_step(*args, **kwargs)
        model_steps.append(True)
        return result

    def dual_step(*args, **kwargs):
        assert len(model_steps) == len(dual_steps) + 1
        assert not torch.equal(torch.stack((model.x.detach(), model.y.detach())), before_update)
        dual_steps.append(True)
        return original_dual_step(*args, **kwargs)

    optimizer.step = model_step
    policy._dual_optimizer.step = dual_step
    result = _config_step(policy, model, optimizer)

    expected_traversals = int(expected_active)
    assert result["fidelity_penalty_active"] is expected_active
    assert result["fidelity_penalty_interval_updates"] == _SPARSE_INTERVAL
    assert result["constraint_gradient_evaluations"] == expected_traversals
    assert result["gradient_traversal_counts"]["constraint"] == expected_traversals
    assert result["family_gradient_evaluations"] == 3
    assert model_steps == [True]
    assert dual_steps == [True]
    assert all(after > before for before, after in zip(result["multipliers_before"], result["multipliers_after"]))


def test_omitted_interval_matches_explicit_every_update_default():
    pytest.importorskip("cooper")

    models = [_ScalarPair(), _ScalarPair()]
    optimizers = [
        torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1.0e-4)
        for model in models
    ]
    policies = [
        ExternalPolicy(_lagrangian_settings(None), _scalar_reference(), "cpu"),
        ExternalPolicy(_lagrangian_settings(1), _scalar_reference(), "cpu"),
    ]

    for _ in range(3):
        results = [
            _scalar_step(policy, model, optimizer)
            for policy, model, optimizer in zip(policies, models, optimizers)
        ]
        assert [result["fidelity_penalty_active"] for result in results] == [True, True]
        assert [result["fidelity_penalty_interval_updates"] for result in results] == [1, 1]
        assert results[0]["optimized_scalar"] == results[1]["optimized_scalar"]
        assert results[0]["multipliers_after"] == results[1]["multipliers_after"]

    _assert_nested_equal(models[0].state_dict(), models[1].state_dict())
    _assert_nested_equal(optimizers[0].state_dict(), optimizers[1].state_dict())
    assert policies[0].completed_updates == policies[1].completed_updates == 3
    assert policies[0]._multipliers() == policies[1]._multipliers()


def test_sparse_penalty_resume_at_379_and_380_matches_uninterrupted_full_state():
    pytest.importorskip("cooper")

    def make_run():
        model = _ScalarPair()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1.0e-4)
        policy = ExternalPolicy(_lagrangian_settings(), _scalar_reference(), "cpu")
        _set_cursor_and_multiplier(policy, 378, [0.2])
        return model, optimizer, policy

    continuous_model, continuous_optimizer, continuous_policy = make_run()
    continuous_results = [
        _scalar_step(continuous_policy, continuous_model, continuous_optimizer)
        for _ in range(3)
    ]
    assert [result["fidelity_penalty_active"] for result in continuous_results] == [False, True, False]

    for split_after in (1, 2):
        model, optimizer, policy = make_run()
        prefix = [
            _scalar_step(policy, model, optimizer)
            for _ in range(split_after)
        ]
        checkpoint = {
            "global_step": policy.completed_updates,
            "model": deepcopy(model.state_dict()),
            "optimizer": deepcopy(optimizer.state_dict()),
            "external_policy": deepcopy(policy.state_dict()),
        }
        assert checkpoint["global_step"] == 378 + split_after

        resumed_model, resumed_optimizer, resumed_policy = make_run()
        resumed_model.load_state_dict(checkpoint["model"])
        resumed_optimizer.load_state_dict(checkpoint["optimizer"])
        resumed_policy.load_state_dict(checkpoint["external_policy"])
        suffix = [
            _scalar_step(resumed_policy, resumed_model, resumed_optimizer)
            for _ in range(3 - split_after)
        ]

        assert [result["fidelity_penalty_active"] for result in prefix + suffix] == [
            result["fidelity_penalty_active"] for result in continuous_results
        ]
        assert resumed_policy.completed_updates == continuous_policy.completed_updates == 381
        _assert_nested_equal(resumed_model.state_dict(), continuous_model.state_dict())
        _assert_nested_equal(resumed_optimizer.state_dict(), continuous_optimizer.state_dict())
        _assert_nested_equal(resumed_policy.state_dict(), continuous_policy.state_dict())
