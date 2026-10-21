"""Focused algebra and update-order tests for internal coherence ConFIG."""

import pytest
import torch
from torch import nn

from phycoflow_reconstruction.training.fidelity_budget import (
    ExternalPolicy,
    FidelityBudget,
    barrier_schedule,
    extended_log_barrier,
    extended_log_barrier_derivative,
)
from phycoflow_reconstruction.training.multitask import OuterAggregator


@pytest.mark.parametrize("updates", [0, 760, 2280, 3800, 4900, 38000])
def test_formal_barrier_continuation_and_late_pressure_are_finite(updates):
    t = barrier_schedule(updates)
    join = -1 / t**2
    h = torch.tensor(
        [-10.0, join - 1e-10, join, join + 1e-10, -1e-8, 0.0, 0.05, 1000.0],
        dtype=torch.float64, requires_grad=True,
    )
    values = extended_log_barrier(h, t)
    actual, = torch.autograd.grad(values.sum(), h)
    expected = extended_log_barrier_derivative(h.detach(), t)
    assert torch.isfinite(values).all() and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
    assert (actual > 0).all() and (actual <= t + 1e-10).all()
    assert values[1].item() == pytest.approx(values[3].item(), abs=1e-6)
    if updates >= 4900:
        assert t == 100.0


class _TwoParameterModel(nn.Module):
    def __init__(self, x=0.8, y=0.4):
        super().__init__()
        self.x = nn.Parameter(torch.tensor(float(x), dtype=torch.float64))
        self.y = nn.Parameter(torch.tensor(float(y), dtype=torch.float64))


def _family_losses(model):
    x, y = model.x, model.y
    return {
        "global_distribution": x.square() + 0.1 * y,
        "cross_spectrum": y.square() + 0.2 * x,
        "topology": (x - y).square() + 0.1 * x * y,
    }


def _risk_vector(model):
    x, y = model.x, model.y
    return torch.stack((x.square(), y.square(), (x + y).square()))


def _risk_gradient_rows(x=0.8, y=0.4):
    return torch.tensor(
        [[2 * x, 0.0], [0.0, 2 * y], [2 * (x + y), 2 * (x + y)]],
        dtype=torch.float64,
    )


def _family_gradient_rows(x=0.8, y=0.4):
    return torch.tensor(
        [[2 * x, 0.1], [0.2, 2 * y], [2 * (x - y) + 0.1 * y, -2 * (x - y) + 0.1 * x]],
        dtype=torch.float64,
    )


def _reference(risks=(0.6, 0.1, 1.0), c0=2.0):
    return FidelityBudget.from_reference(("CH4", "CO", "T"), risks, c0)


def _count_steps(optimizer):
    steps = []
    original_step = optimizer.step

    def counted_step(*args, **kwargs):
        steps.append(True)
        return original_step(*args, **kwargs)

    optimizer.step = counted_step
    return steps


def test_lagrangian_uses_ordered_individual_config_rows_and_steps_dual_after_model():
    pytest.importorskip("torchjd")
    pytest.importorskip("cooper")

    model = _TwoParameterModel()
    policy = ExternalPolicy(
        {"method": "lagrangian", "lagrangian": {"dual_lr": 0.01}}, _reference(), "cpu"
    )
    state = policy.state_dict()
    state["multipliers"] = [0.2, 0.3, 0.4]
    policy.load_state_dict(state)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    model_steps = _count_steps(optimizer)
    original_model = torch.stack((model.x.detach(), model.y.detach()))
    original_dual_step = policy._dual_optimizer.step

    def dual_step_after_model(*args, **kwargs):
        assert model_steps == [True]
        assert not torch.equal(torch.stack((model.x.detach(), model.y.detach())), original_model)
        return original_dual_step(*args, **kwargs)

    policy._dual_optimizer.step = dual_step_after_model
    rows = _family_gradient_rows()
    config_direction, _ = OuterAggregator({"method": "torchjd_config"})(rows)
    risk_gradients = _risk_gradient_rows()
    multipliers = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float64)
    denominators = torch.tensor([0.6, 0.1, 1.0], dtype=torch.float64)
    constraint_direction = (multipliers[:, None] * risk_gradients / denominators[:, None]).sum(dim=0)
    expected_direction = config_direction / 2.0 + constraint_direction
    expected_violations = torch.tensor(
        [(0.64 - 0.63) / 0.6, (0.16 - 0.105) / 0.1, (1.44 - 1.05) / 1.0],
        dtype=torch.float64,
    )

    result = policy.update_with_coherence_aggregation(
        model,
        optimizer,
        _family_losses(model),
        _risk_vector(model),
        None,
        OuterAggregator({"method": "torchjd_config"}),
    )

    actual_direction = (original_model - torch.stack((model.x.detach(), model.y.detach()))) / 0.001
    assert torch.allclose(actual_direction, expected_direction, atol=1e-10, rtol=1e-10)
    assert model_steps == [True]
    assert policy._multipliers() == pytest.approx((multipliers + 0.01 * expected_violations).tolist())
    assert result["family_order"] == ["global_distribution", "cross_spectrum", "topology"]
    assert result["family_gradient_evaluations"] == 3
    assert result["constraint_gradient_evaluations"] == 1
    assert result["coherence_aggregation"]["gradient_matrix_shape"] == [3, 2]
    assert result["coherence_aggregation"]["method"] == "torchjd_config"
    assert result["coherence_aggregation"]["directional_products_by_family"].keys() == set(
        result["family_order"]
    )
    assert result["coherence_value"] == pytest.approx(sum(float(value.detach()) for value in _family_losses(_TwoParameterModel()).values()))
    assert "optimized_scalar" not in result
    assert result["policy_state"]["completed_updates"] == 1


def test_rectify_repair_skips_family_gradients_and_config(monkeypatch):
    model = _TwoParameterModel()
    policy = ExternalPolicy({"method": "rectify"}, _reference(), "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model_steps = _count_steps(optimizer)
    losses = _family_losses(model)
    traversed = []
    for name, loss in losses.items():
        loss.register_hook(lambda gradient, family=name: traversed.append(family) or gradient)

    def fail_if_config_called(*_args, **_kwargs):
        raise AssertionError("repair must not invoke ConFIG")

    monkeypatch.setattr(OuterAggregator, "__call__", fail_if_config_called)
    result = policy.update_with_coherence_aggregation(
        model,
        optimizer,
        losses,
        _risk_vector(model),
        None,
        OuterAggregator({"method": "torchjd_config"}),
    )

    assert result["phase"] == "repair"
    assert result["repair_field"] == "CO"
    assert result["update_mode"] == "coherence_config_rectify_repair"
    assert result["family_gradient_evaluations"] == 0
    assert result["constraint_gradient_evaluations"] == 1
    assert result["coherence_aggregation"] is None
    assert result["rectify_repair_field/CO"] is True
    assert result["rectify_repair_field/CH4"] is False
    assert traversed == []
    assert model_steps == [True]
    assert model.x.item() == pytest.approx(0.8)
    assert model.y.item() == pytest.approx(0.32)
    assert "optimized_scalar" not in result


def test_rectify_improve_uses_config_only_and_rank_deficiency_is_reported():
    pytest.importorskip("torchjd")

    model = _TwoParameterModel(x=0.5, y=0.25)
    policy = ExternalPolicy(
        {"method": "rectify"}, _reference(risks=(10.0, 10.0, 10.0), c0=1.0), "cpu"
    )
    losses = {"A": model.x, "B": 2.0 * model.x, "T": 3.0 * model.x}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    rows = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=torch.float64)
    expected, _ = OuterAggregator({"method": "torchjd_config"})(rows)

    result = policy.update_with_coherence_aggregation(
        model,
        optimizer,
        losses,
        torch.stack((model.x.square(), model.x.square(), model.x.square())),
        None,
        OuterAggregator({"method": "torchjd_config"}),
    )

    assert model.x.item() == pytest.approx(0.5 - 0.01 * expected[0].item())
    assert model.y.item() == pytest.approx(0.25)
    assert result["phase"] == "improve"
    assert result["family_gradient_evaluations"] == 3
    assert result["constraint_gradient_evaluations"] == 0
    assert result["coherence_aggregation"]["matrix_rank"] == 1
    assert result["coherence_aggregation"]["matrix_rank_deficient"] is True
    assert result["coherence_aggregation"]["zero_gradient_flags"] == {
        "A": False,
        "B": False,
        "T": False,
    }


def test_float32_family_gradients_enter_torchjd_pinv_in_float64(monkeypatch):
    pytest.importorskip("torchjd")

    class _Float32Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.value = nn.Parameter(torch.tensor([0.2, 0.3, 0.4], dtype=torch.float32))

    model = _Float32Model()
    families = {
        "A": model.value[0],
        "B": -0.8 * model.value[0] + 0.6 * model.value[1],
        "T": model.value[2],
    }
    raw_rows = torch.tensor([[1.0, 0.0, 0.0], [-0.8, 0.6, 0.0], [0.0, 0.0, 1.0]])
    expected, _ = OuterAggregator({"method": "torchjd_config"})(raw_rows.double())
    seen_dtypes = []
    original_pinv = torch.linalg.pinv

    def record_pinv_dtype(matrix, *args, **kwargs):
        seen_dtypes.append(matrix.dtype)
        return original_pinv(matrix, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "pinv", record_pinv_dtype)
    policy = ExternalPolicy(
        {"method": "rectify"}, FidelityBudget.from_reference(("f0", "f1", "f2"), (10, 10, 10), 1), "cpu"
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    initial = model.value.detach().clone()
    result = policy.update_with_coherence_aggregation(
        model,
        optimizer,
        families,
        model.value.square(),
        None,
        OuterAggregator({"method": "torchjd_config"}),
    )

    assert seen_dtypes == [torch.float64]
    assert result["coherence_aggregation"]["source_gradient_dtype"] == "torch.float32"
    assert result["coherence_aggregation"]["aggregation_dtype"] == "torch.float64"
    assert torch.allclose((initial - model.value.detach()) / 0.001, expected.float(), atol=1e-5)


def test_extended_barrier_matches_config_plus_single_penalty_gradient():
    pytest.importorskip("torchjd")

    model = _TwoParameterModel()
    reference = _reference(risks=(0.8, 0.2, 2.0), c0=2.0)
    policy = ExternalPolicy({"method": "extended_log_barrier"}, reference, "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    initial = torch.stack((model.x.detach(), model.y.detach()))
    config_direction, _ = OuterAggregator({"method": "torchjd_config"})(_family_gradient_rows())
    risk_gradients = _risk_gradient_rows()
    denominators = torch.tensor([0.8, 0.2, 2.0], dtype=torch.float64)
    risks = torch.tensor([0.64, 0.16, 1.44], dtype=torch.float64)
    upper = torch.tensor([0.84, 0.21, 2.1], dtype=torch.float64)
    violations = (risks - upper) / denominators
    barrier_weights = extended_log_barrier_derivative(violations, 1.0)
    expected_direction = config_direction / 2.0 + (
        barrier_weights[:, None] * risk_gradients / denominators[:, None]
    ).sum(dim=0)

    result = policy.update_with_coherence_aggregation(
        model,
        optimizer,
        _family_losses(model),
        _risk_vector(model),
        None,
        OuterAggregator({"method": "torchjd_config"}),
    )

    actual_direction = (initial - torch.stack((model.x.detach(), model.y.detach()))) / 0.001
    assert torch.allclose(actual_direction, expected_direction, atol=1e-10, rtol=1e-10)
    assert result["barrier_t"] == 1.0
    assert result["family_gradient_evaluations"] == 3
    assert result["constraint_gradient_evaluations"] == 1


def test_nonfinite_and_zero_family_gradients_reject_without_optimizer_step():
    model = _TwoParameterModel()
    policy = ExternalPolicy({"method": "rectify"}, _reference(), "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model_steps = _count_steps(optimizer)
    aggregator = OuterAggregator({"method": "torchjd_config"})

    with pytest.raises(FloatingPointError, match="risk is non-finite"):
        policy.update_with_coherence_aggregation(
            model,
            optimizer,
            _family_losses(model),
            torch.tensor([float("nan"), 0.0, 0.0], dtype=torch.float64),
            None,
            aggregator,
        )
    assert model_steps == []
    assert policy.completed_updates == 0

    zero_gradient_losses = {
        "A": model.x,
        "B": model.x * 0.0,
        "T": model.y,
    }
    with pytest.raises(FloatingPointError, match="family gradient is zero.*B"):
        policy.update_with_coherence_aggregation(
            model,
            optimizer,
            zero_gradient_losses,
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
            None,
            aggregator,
        )
    assert model_steps == []
    assert policy.completed_updates == 0


def test_config_zero_direction_from_three_way_conflict_rejects_without_fallback():
    pytest.importorskip("torchjd")

    model = _TwoParameterModel()
    root_three_over_two = torch.sqrt(model.x.new_tensor(3.0)) / 2.0
    losses = {
        "A": model.x,
        "B": -0.5 * model.x + root_three_over_two * model.y,
        "T": -0.5 * model.x - root_three_over_two * model.y,
    }
    policy = ExternalPolicy(
        {"method": "rectify"}, _reference(risks=(10.0, 10.0, 10.0), c0=1.0), "cpu"
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model_steps = _count_steps(optimizer)

    with pytest.raises(FloatingPointError, match="ConFIG returned a zero"):
        policy.update_with_coherence_aggregation(
            model,
            optimizer,
            losses,
            torch.stack((model.x.square(), model.y.square(), (model.x + model.y).square())),
            None,
            OuterAggregator({"method": "torchjd_config"}),
        )

    assert model_steps == []
    assert policy.completed_updates == 0


def test_scalar_external_policy_api_still_returns_optimized_scalar():
    model = _TwoParameterModel(x=0.5, y=0.25)
    policy = ExternalPolicy(
        {"method": "rectify"}, FidelityBudget.from_reference(("f",), (1.0,), 1.0), "cpu"
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    result = policy.update(
        model,
        optimizer,
        model.x.square() + model.y.square(),
        torch.tensor([0.25], dtype=torch.float64),
        None,
    )

    assert result["optimized_scalar"] == pytest.approx(0.3125)
    assert result["policy"] == "rectify"
    assert "optimization_route" not in result
