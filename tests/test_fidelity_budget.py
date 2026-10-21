"""Numerical contracts for pooled endpoint fidelity and external policies."""

import json

import pytest
import torch
from torch import nn

from phycoflow_reconstruction.training.fidelity_budget import (
    ExternalPolicy,
    FidelityBudget,
    barrier_schedule,
    extended_log_barrier,
    extended_log_barrier_derivative,
    field_error_sums,
    field_mse,
    reduce_primary,
)
from phycoflow_reconstruction.training.multitask import FamilyScalarizer


class _ScalarModel(nn.Module):
    def __init__(self, value=2.0):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value), dtype=torch.float64))


def _budget(names=("first",), risks=(1.0,), *, epsilon=0.0, c0=1.0):
    return FidelityBudget.from_reference(names, risks, c0, relative_increase=epsilon)


def test_field_mse_pools_valid_points_instead_of_averaging_sample_means():
    prediction = torch.tensor(
        [
            [[1.0, 4.0], [3.0, 4.0], [5.0, 4.0]],
            [[7.0, 8.0], [100.0, 100.0], [100.0, 100.0]],
        ],
        dtype=torch.float64,
    )
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, True, True], [True, False, False]])

    sums, counts = field_error_sums(prediction, target, mask)
    risks = field_mse(prediction, target, mask)

    assert torch.equal(sums, torch.tensor([84.0, 112.0], dtype=torch.float64))
    assert torch.equal(counts, torch.tensor([4.0, 4.0], dtype=torch.float64))
    assert torch.equal(risks, torch.tensor([21.0, 28.0], dtype=torch.float64))
    # A mean of the two snapshot means would be 25.33 and 34, respectively.
    assert not torch.allclose(risks, torch.tensor([76.0 / 3, 34.0], dtype=torch.float64))


def test_field_mse_pools_generative_draws_and_broadcasts_targets_and_mask():
    predictions = torch.tensor([[[[1.0], [3.0]]], [[[5.0], [7.0]]]])
    target = torch.zeros((1, 2, 1))
    mask = torch.tensor([[True, True]])

    sums, counts = field_error_sums(predictions, target, mask)

    assert torch.equal(sums, torch.tensor([84.0], dtype=torch.float64))
    assert torch.equal(counts, torch.tensor([4.0], dtype=torch.float64))
    assert field_mse(predictions, target, mask).item() == pytest.approx(21.0)


def test_invalid_nan_padding_is_masked_before_squared_error_backward():
    prediction = torch.tensor([[[2.0], [float("nan")]]], requires_grad=True)
    target = torch.tensor([[[1.0], [float("nan")]]], requires_grad=True)
    risk = field_mse(prediction, target, torch.tensor([[True, False]]))
    risk.sum().backward()
    assert risk.item() == 1.0
    torch.testing.assert_close(prediction.grad, torch.tensor([[[2.0], [0.0]]]))
    assert target.grad is None


def test_field_mse_rejects_empty_support_and_masks_invalid_query_gradients():
    prediction = torch.tensor([[[1.0, 2.0], [100.0, 100.0]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    with pytest.raises(ValueError, match="every field"):
        field_mse(prediction, target, torch.zeros((1, 2), dtype=torch.bool))

    risk = field_mse(prediction, target, torch.tensor([[True, False]]))
    risk.sum().backward()
    assert torch.equal(risk, torch.tensor([1.0, 4.0], dtype=torch.float64))
    assert torch.equal(prediction.grad, torch.tensor([[[2.0, 4.0], [0.0, 0.0]]]))


def test_budget_floor_changes_only_scaling_and_preserves_near_zero_thresholds():
    reference = FidelityBudget.from_reference(
        ("ordinary", "tiny", "zero"),
        (2.0, 1.0e-10, 0.0),
        0.0,
        relative_increase=0.05,
    )

    assert reference.upper_bounds == pytest.approx((2.1, 1.05e-10, 0.0))
    assert reference.denominators == (2.0, 1.0e-8, 1.0e-8)
    assert reference.denominator_floor_activated == (False, True, True)
    assert reference.objective_scale == 1.0e-8
    assert reference.objective_floor_activated
    violation = reference.violations(torch.tensor([2.1, 1.0e-9, 1.0e-10], dtype=torch.float64))
    assert violation[0].item() == pytest.approx(0.0, abs=1e-15)
    assert violation[1].item() == pytest.approx((1.0e-9 - 1.05e-10) / 1.0e-8)
    assert violation[2].item() == pytest.approx(0.01)


def test_budget_state_is_json_friendly_and_restores_reference_exactly():
    reference = FidelityBudget.from_reference(
        ("CO", "T"), (0.25, 1.5), 0.75, relative_increase=0.05, absolute_allowance=0.01
    )
    state = json.loads(json.dumps(reference.state_dict()))

    assert FidelityBudget.from_state_dict(state) == reference


def test_none_external_sum_matches_fixed_sum_value_and_gradient():
    losses_none = {"A": torch.tensor(2.0, requires_grad=True) ** 2, "B": torch.tensor(3.0, requires_grad=True) * 4}
    losses_fixed = {"A": torch.tensor(2.0, requires_grad=True) ** 2, "B": torch.tensor(3.0, requires_grad=True) * 4}
    scales = {"A": 0.5, "B": 1.25}
    weights = {"A": 1.0, "B": 0.8}
    separate, _ = FamilyScalarizer({"method": "none"})(losses_none, scales=scales, weights=weights)
    fixed, _ = FamilyScalarizer({"method": "fixed_sum"})(losses_fixed, scales=scales, weights=weights)
    external = reduce_primary(separate)

    assert torch.equal(external, fixed)
    none_grads = torch.autograd.grad(external, tuple(losses_none.values()))
    fixed_grads = torch.autograd.grad(fixed, tuple(losses_fixed.values()))
    assert all(torch.equal(left, right) for left, right in zip(none_grads, fixed_grads))


def test_cooper_policy_matches_analytic_and_high_level_simultaneous_update():
    pytest.importorskip("cooper")
    from cooper import CMPState, ConstrainedMinimizationProblem, Constraint, ConstraintState
    from cooper.formulations import Lagrangian
    from cooper.multipliers import DenseMultiplier
    from cooper.optim import SimultaneousOptimizer
    from cooper.utils import ConstraintType

    model = _ScalarModel(2.0)
    primary = model.value.square()
    risk = model.value.reshape(1)
    reference = _budget()
    policy = ExternalPolicy(
        {"method": "lagrangian", "lagrangian": {"dual_lr": 0.25}}, reference, "cpu"
    )
    primal_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    backward_count = []
    model.value.register_hook(lambda gradient: backward_count.append(gradient.clone()) or gradient)
    model_step_count = 0
    original_step = primal_optimizer.step

    def model_step(*args, **kwargs):
        nonlocal model_step_count
        model_step_count += 1
        return original_step(*args, **kwargs)

    primal_optimizer.step = model_step
    original_dual_step = policy._dual_optimizer.step

    def ordered_dual_step(*args, **kwargs):
        assert model_step_count == 1
        return original_dual_step(*args, **kwargs)

    policy._dual_optimizer.step = ordered_dual_step
    result = policy.update(model, primal_optimizer, primary, risk, None)

    assert model.value.item() == pytest.approx(1.6)
    assert policy._multipliers() == pytest.approx([0.25])
    assert result["optimized_scalar"] == pytest.approx(4.0)
    assert result["multipliers_before"] == [0.0]
    assert result["multipliers_after"] == pytest.approx([0.25])
    assert len(backward_count) == 1  # detached Cooper dual update did not revisit the model graph

    class _AnalyticProblem(ConstrainedMinimizationProblem):
        def __init__(self, variable, multiplier, violation):
            super().__init__()
            self.constraint = Constraint(
                ConstraintType.INEQUALITY, formulation_type=Lagrangian, multiplier=multiplier
            )
            self.variable = variable
            self.violation = violation

        def compute_cmp_state(self, **kwargs):
            del kwargs
            return CMPState(
                loss=self.variable.square(),
                observed_constraints={
                    self.constraint: ConstraintState(violation=self.violation),
                },
            )

    comparison_model = _ScalarModel(2.0)
    multiplier = DenseMultiplier(
        num_constraints=1, init=torch.zeros(1, dtype=torch.float64), dtype=torch.float64
    )
    problem = _AnalyticProblem(comparison_model.value, multiplier, comparison_model.value.reshape(1) - 1.0)
    high_level = SimultaneousOptimizer(
        problem,
        torch.optim.SGD(comparison_model.parameters(), lr=0.1),
        torch.optim.SGD(multiplier.parameters(), lr=0.25, maximize=True),
    )
    high_level.roll()
    assert comparison_model.value.item() == pytest.approx(model.value.item())
    assert multiplier().item() == pytest.approx(policy._multipliers()[0])


def test_cooper_projection_clamps_a_negative_multiplier_to_zero():
    pytest.importorskip("cooper")
    policy = ExternalPolicy(
        {"method": "lagrangian", "lagrangian": {"dual_lr": 0.5}}, _budget(), "cpu"
    )
    state = policy.state_dict()
    state["multipliers"] = [0.1]
    policy.load_state_dict(state)
    model = _ScalarModel(2.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    policy.update(model, optimizer, model.value.square(), torch.tensor([0.0]), None)

    assert policy._multipliers() == [0.0]


def test_rectify_improve_and_tied_repair_choose_first_field_without_mixing():
    reference = _budget(("A", "B"), (1.0, 1.0))
    repair = ExternalPolicy({"method": "rectify"}, reference, "cpu")
    model = _ScalarModel(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    risks = torch.stack((model.value + 1.0, 2.0 * model.value))
    result = repair.update(model, optimizer, model.value.square() * 100.0, risks, None)

    assert result["phase"] == "repair"
    assert result["repair_field"] == "A"
    assert result["optimized_scalar"] == pytest.approx(1.0)
    assert model.value.item() == pytest.approx(0.9)

    improve = ExternalPolicy({"method": "rectify"}, reference, "cpu")
    model = _ScalarModel(2.0)
    result = improve.update(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        model.value.square(),
        torch.tensor([0.5, 0.75], dtype=torch.float64),
        None,
    )
    assert result["phase"] == "improve"
    assert result["repair_field"] is None
    assert result["optimized_scalar"] == pytest.approx(4.0)


def test_barrier_value_gradient_join_schedule_and_resume():
    t = 2.0
    join = -1.0 / (t * t)
    violation = torch.tensor([-0.5, join, -0.1, 0.0, 0.1], dtype=torch.float64, requires_grad=True)
    values = extended_log_barrier(violation, t)
    values.sum().backward()
    expected = extended_log_barrier_derivative(violation.detach(), t)

    assert torch.isfinite(values).all()
    assert torch.isfinite(violation.grad).all()
    assert torch.allclose(violation.grad, expected)
    assert extended_log_barrier(torch.tensor([join - 1e-10], dtype=torch.float64), t).item() == pytest.approx(
        extended_log_barrier(torch.tensor([join + 1e-10], dtype=torch.float64), t).item(), abs=1e-9
    )
    assert barrier_schedule(99) == 1.0
    assert barrier_schedule(100) == pytest.approx(1.1)
    assert barrier_schedule(10**1000) == 100.0

    settings = {"method": "extended_log_barrier"}
    policy = ExternalPolicy(settings, _budget(), "cpu")
    model = _ScalarModel(2.0)
    state = policy.state_dict()
    state["completed_updates"] = 99
    policy.load_state_dict(state)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    first = policy.update(model, optimizer, model.value.square(), torch.tensor([0.5]), None)
    assert first["barrier_t"] == 1.0
    assert policy.completed_updates == 100
    restored = ExternalPolicy(settings, _budget(), "cpu")
    restored.load_state_dict(policy.state_dict())
    assert restored.completed_updates == 100
    next_result = restored.update(model, optimizer, model.value.square(), torch.tensor([0.5]), None)
    assert next_result["barrier_t"] == pytest.approx(1.1)
    assert restored.completed_updates == 101


@pytest.mark.parametrize("failure", ["risk", "gradient"])
def test_nonfinite_attempt_does_not_step_model_dual_or_policy_state(failure):
    pytest.importorskip("cooper")
    policy = ExternalPolicy({"method": "lagrangian"}, _budget(), "cpu")
    model = _ScalarModel(0.0 if failure == "gradient" else 2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    before_parameter = model.value.detach().clone()
    before_policy = policy.state_dict()

    with pytest.raises(FloatingPointError, match="(risk|gradient) is non-finite"):
        policy.update(
            model,
            optimizer,
            model.value.sqrt() if failure == "gradient" else model.value.square(),
            torch.tensor([0.5 if failure == "gradient" else float("nan")], dtype=torch.float64),
            None,
        )

    assert torch.equal(model.value, before_parameter)
    assert policy.state_dict() == before_policy
    assert optimizer.state == {}


@pytest.mark.parametrize("clip", [None, 0.5])
def test_unused_and_complex_model_parameters_keep_flat_gradient_adamw_semantics(clip):
    class _ComplexModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.used = nn.Parameter(torch.tensor(1.0 + 2.0j))
            self.unused = nn.Parameter(torch.tensor(3.0))

    model = _ComplexModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.1)
    policy = ExternalPolicy({"method": "rectify"}, _budget(), "cpu")
    before_unused = model.unused.detach().clone()

    policy.update(model, optimizer, model.used.abs().square(), torch.tensor([0.5]), clip)

    assert model.used.grad is not None and torch.is_complex(model.used.grad)
    assert model.used.grad.imag.item() != 0
    assert model.used.grad.imag.item() == pytest.approx(2 * model.used.grad.real.item())
    if clip is not None:
        assert model.used.grad.abs().item() == pytest.approx(clip, rel=1e-5)
    assert torch.equal(model.unused.grad, torch.zeros_like(model.unused))
    assert model.unused.item() == pytest.approx(float(before_unused) * (1.0 - 0.05 * 0.1))
