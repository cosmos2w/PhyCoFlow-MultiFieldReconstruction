"""Numerical and failure-order contracts for synchronized external-policy updates."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import pytest
import torch
from torch import nn

from phycoflow_reconstruction.training.fidelity_budget import (
    ExternalPolicy,
    FidelityBudget,
)
from phycoflow_reconstruction.training.multitask import (
    OuterAggregator,
    _gradient_diagnostics,
)


class _WarmupOracleModel(nn.Module):
    """Small real/complex model with one intentionally unused trainable parameter."""

    def __init__(self, device: torch.device):
        super().__init__()
        self.head = nn.Parameter(torch.tensor([0.8, 0.2], dtype=torch.float64, device=device))
        self.tail = nn.Parameter(torch.tensor([0.7], dtype=torch.float64, device=device))
        self.unused = nn.Parameter(torch.tensor([1.2, -0.4], dtype=torch.float64, device=device))
        self.complex_mode = nn.Parameter(
            torch.tensor([0.4 + 0.3j], dtype=torch.complex128, device=device)
        )

    def objectives(self) -> tuple[torch.Tensor, torch.Tensor]:
        primary = (
            (self.head - self.head.new_tensor([0.2, 0.5])).square().sum()
            + 0.3 * (self.tail - 1.0).square().sum()
            + 0.07 * self.complex_mode.abs().square().sum()
        )
        risks = torch.stack(
            (
                self.head.square().mean() + 0.2 * self.complex_mode.abs().square().mean(),
                (self.tail - 0.15 * self.head[:1]).square().mean(),
            )
        )
        return primary, risks


def _assert_tree_close(left, right, *, atol: float = 1.0e-12, rtol: float = 1.0e-12) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, atol=atol, rtol=rtol)
    elif isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_close(left[key], right[key], atol=atol, rtol=rtol)
    elif isinstance(left, (tuple, list)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right):
            _assert_tree_close(left_value, right_value, atol=atol, rtol=rtol)
    else:
        if isinstance(left, float):
            assert left == pytest.approx(right)
        else:
            assert left == right


def _warmup_lr(completed_updates: int) -> float:
    # Same three-update linear warmup contract used by retained post-training.
    start_lr, base_lr, last_warmup_index = 0.002, 0.01, 2
    if completed_updates >= last_warmup_index:
        return base_lr
    return start_lr + (base_lr - start_lr) * completed_updates / last_warmup_index


def _policy_settings() -> dict:
    return {
        "method": "lagrangian",
        "vector_reduction": "sum",
        "objective_scale": "source_mean",
        "lagrangian": {"dual_lr": 0.01},
    }


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_run11_lagrangian_update_matches_independent_primal_dual_oracle_with_warmup(device_name):
    pytest.importorskip("cooper")

    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_name)
    reference = FidelityBudget.from_reference(
        ("head_risk", "tail_risk"),
        (0.45, 0.50),
        2.0,
        relative_increase=0.05,
    )
    actual_model = _WarmupOracleModel(device)
    oracle_model = deepcopy(actual_model)
    actual_optimizer = torch.optim.AdamW(
        actual_model.parameters(), lr=0.01, weight_decay=0.03, eps=1.0e-8
    )
    oracle_optimizer = torch.optim.AdamW(
        oracle_model.parameters(), lr=0.01, weight_decay=0.03, eps=1.0e-8
    )
    policy = ExternalPolicy(_policy_settings(), reference, device)
    policy_state = policy.state_dict()
    policy_state["multipliers"] = [0.2, 0.001]
    policy.load_state_dict(policy_state)
    oracle_multipliers = torch.tensor([0.2, 0.001], dtype=torch.float64, device=device)
    dual_lr = 0.01
    bounds = torch.tensor(reference.upper_bounds, dtype=torch.float64, device=device)
    denominators = torch.tensor(reference.denominators, dtype=torch.float64, device=device)

    # Three concrete optimizer updates cover the warmup start, interpolation, and
    # return to base LR while retaining every-update primal and dual contributions.
    for expected_update in range(1, 4):
        lr = _warmup_lr(policy.completed_updates)
        actual_optimizer.param_groups[0]["lr"] = lr
        oracle_optimizer.param_groups[0]["lr"] = lr

        primary, risks = oracle_model.objectives()
        violations = (risks - bounds) / denominators
        oracle_multipliers_before = oracle_multipliers.clone()
        oracle_objective = primary / reference.objective_scale + torch.dot(
            oracle_multipliers, violations
        )
        oracle_optimizer.zero_grad(set_to_none=True)
        oracle_objective.backward()
        for parameter in oracle_model.parameters():
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
        oracle_gradient_norm = torch.sqrt(
            sum(
                parameter.grad.detach().abs().to(torch.float64).square().sum()
                for parameter in oracle_model.parameters()
            )
        )
        oracle_optimizer.step()
        oracle_multipliers = (oracle_multipliers + dual_lr * violations.detach()).clamp_min(0.0)

        actual_primary, actual_risks = actual_model.objectives()
        result = policy.update(
            actual_model,
            actual_optimizer,
            actual_primary,
            actual_risks,
            grad_clip=None,
        )

        _assert_tree_close(actual_model.state_dict(), oracle_model.state_dict())
        _assert_tree_close(actual_optimizer.state_dict(), oracle_optimizer.state_dict())
        assert policy._multipliers() == pytest.approx(oracle_multipliers.cpu().tolist(), abs=1.0e-12)
        assert result["multipliers_before"] == pytest.approx(
            oracle_multipliers_before.cpu().tolist(), abs=1.0e-12
        )
        assert result["completed_updates"] == expected_update
        assert result["fidelity_penalty_active"] is True
        assert result["fidelity_penalty_interval_updates"] == 1
        assert result["optimized_scalar"] == pytest.approx(float(oracle_objective.detach().cpu()), abs=1.0e-12)
        assert result["model_grad_norm"] == pytest.approx(float(oracle_gradient_norm.cpu()), abs=1.0e-12)
        assert result["field_violation"] == pytest.approx(
            dict(zip(reference.field_names, violations.detach().cpu().tolist())), abs=1.0e-12
        )
        assert actual_optimizer.param_groups[0]["lr"] == lr


class _TwoScalarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.early = nn.Parameter(torch.tensor(0.8, dtype=torch.float64))
        self.late = nn.Parameter(torch.tensor(0.4, dtype=torch.float64))


class _FiniteForwardNanBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.new_zeros(())

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("nan"))


def _one_field_policy() -> ExternalPolicy:
    pytest.importorskip("cooper")
    reference = FidelityBudget.from_reference(("field",), (0.5,), 1.0)
    policy = ExternalPolicy(_policy_settings(), reference, "cpu")
    state = policy.state_dict()
    state["multipliers"] = [0.3]
    policy.load_state_dict(state)
    return policy


def test_nonfinite_late_model_gradient_rejects_before_model_or_dual_action():
    model = _TwoScalarModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.02)
    policy = _one_field_policy()
    before_model = deepcopy(model.state_dict())
    before_optimizer = deepcopy(optimizer.state_dict())
    before_policy = deepcopy(policy.state_dict())
    model_steps: list[bool] = []
    original_step = optimizer.step

    def count_step(*args, **kwargs):
        model_steps.append(True)
        return original_step(*args, **kwargs)

    optimizer.step = count_step
    primary = model.early.square() + _FiniteForwardNanBackward.apply(model.late)
    risks = model.early.square().reshape(1)

    with pytest.raises(FloatingPointError, match="model gradient is non-finite; refusing update"):
        policy.update(model, optimizer, primary, risks, grad_clip=None)

    assert model_steps == []
    _assert_tree_close(model.state_dict(), before_model)
    _assert_tree_close(optimizer.state_dict(), before_optimizer)
    _assert_tree_close(policy.state_dict(), before_policy)
    assert model.early.grad is None and model.late.grad is None
    assert policy.completed_updates == 0


def test_nonfinite_late_parameter_is_detected_after_model_step_and_before_dual_step():
    pytest.importorskip("cooper")

    model = _TwoScalarModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.02)
    policy = _one_field_policy()
    before_model = deepcopy(model.state_dict())
    before_multipliers = policy._multipliers()
    before_dual_optimizer = deepcopy(policy._dual_optimizer.state_dict())
    model_steps: list[bool] = []
    dual_steps: list[bool] = []
    original_model_step = optimizer.step
    original_dual_step = policy._dual_optimizer.step

    def corrupt_late_parameter_after_step(*args, **kwargs):
        result = original_model_step(*args, **kwargs)
        model_steps.append(True)
        with torch.no_grad():
            model.late.fill_(float("inf"))
        return result

    def count_dual_step(*args, **kwargs):
        dual_steps.append(True)
        return original_dual_step(*args, **kwargs)

    optimizer.step = corrupt_late_parameter_after_step
    policy._dual_optimizer.step = count_dual_step
    primary = model.early.square() + 0.5 * model.late.square()
    risks = model.early.square().reshape(1)

    with pytest.raises(FloatingPointError, match="model optimizer produced non-finite parameters"):
        policy.update(model, optimizer, primary, risks, grad_clip=None)

    assert model_steps == [True]
    assert dual_steps == []
    assert model.early.detach() != before_model["early"]
    assert torch.isinf(model.late.detach())
    assert policy._multipliers() == before_multipliers
    _assert_tree_close(policy._dual_optimizer.state_dict(), before_dual_optimizer)
    assert policy.completed_updates == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("reuse_gram", [False, True])
def test_gradient_diagnostics_match_original_zero_row_cosine_and_product_formula(dtype, reuse_gram):
    gradients = torch.tensor([[3.0, 4.0], [0.0, 0.0], [-3.0, 4.0]], dtype=dtype)
    direction = torch.tensor([1.0, 2.0], dtype=dtype)

    kwargs = {}
    if reuse_gram:
        matrix64 = gradients.double()
        kwargs = {"row_norms": torch.linalg.vector_norm(matrix64, dim=1), "row_gram": matrix64 @ matrix64.T}
    report = _gradient_diagnostics(gradients, direction, method="reference", **kwargs)

    assert report["gradient_norms"] == pytest.approx([5.0, 0.0, 5.0])
    assert report["gradient_cosine"] == pytest.approx(0.28)
    torch.testing.assert_close(
        torch.tensor(report["pairwise_gradient_cosines"]),
        torch.tensor([[1.0, 0.0, 0.28], [0.0, 0.0, 0.0], [0.28, 0.0, 1.0]]),
        atol=1.0e-7,
        rtol=0.0,
    )
    assert report["direction_norm"] == pytest.approx(5.0**0.5)
    assert report["directional_products"] == pytest.approx([11.0, 0.0, 5.0])


@pytest.mark.parametrize("reuse_gram", [False, True])
def test_gradient_diagnostics_zero_reference_row_keeps_finite_zero_cosine(reuse_gram):
    gradients = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.float64)
    direction = gradients[1].clone()
    kwargs = {}
    if reuse_gram:
        kwargs = {"row_norms": torch.linalg.vector_norm(gradients, dim=1), "row_gram": gradients @ gradients.T}
    result = _gradient_diagnostics(gradients, direction, method="reference", **kwargs)
    assert result["gradient_cosine"] == 0.0
    assert result["pairwise_gradient_cosines"][0] == [0.0, 0.0]


def test_zero_family_row_reports_existing_error_and_preserves_model_optimizer_and_controller():
    model = _TwoScalarModel()
    reference = FidelityBudget.from_reference(("field_a", "field_b"), (10.0, 10.0), 1.0)
    policy = ExternalPolicy({"method": "rectify"}, reference, "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    before_model = deepcopy(model.state_dict())
    before_optimizer = deepcopy(optimizer.state_dict())
    before_policy = deepcopy(policy.state_dict())
    model_steps: list[bool] = []
    original_step = optimizer.step

    def count_step(*args, **kwargs):
        model_steps.append(True)
        return original_step(*args, **kwargs)

    optimizer.step = count_step
    losses = {"A": model.early, "B": 2.0 * model.early, "T": model.late * 0.0}
    risks = torch.stack((model.early.square(), model.late.square()))

    with pytest.raises(
        FloatingPointError,
        match=r"ConFIG family gradient is zero for \['T'\]; refusing update",
    ):
        policy.update_with_coherence_aggregation(
            model,
            optimizer,
            losses,
            risks,
            grad_clip=None,
            aggregator=OuterAggregator({"method": "torchjd_config"}),
        )

    assert model_steps == []
    _assert_tree_close(model.state_dict(), before_model)
    _assert_tree_close(optimizer.state_dict(), before_optimizer)
    _assert_tree_close(policy.state_dict(), before_policy)
    assert policy.completed_updates == 0


def test_rank_deficient_config_diagnostics_keep_flags_and_torchjd_direction():
    torchjd = pytest.importorskip("torchjd.aggregation")
    model = _TwoScalarModel()
    reference = FidelityBudget.from_reference(("field_a", "field_b"), (10.0, 10.0), 1.0)
    policy = ExternalPolicy({"method": "rectify"}, reference, "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    losses = {"A": model.early, "B": 2.0 * model.early, "T": 3.0 * model.early}
    gradient_rows = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=torch.float64)
    torchjd_direction = torchjd.ConFIG()(gradient_rows)
    before = model.early.detach().clone()

    result = policy.update_with_coherence_aggregation(
        model,
        optimizer,
        losses,
        torch.stack((model.early.square(), model.late.square())),
        grad_clip=None,
        aggregator=OuterAggregator({"method": "torchjd_config"}),
    )

    torch.testing.assert_close(model.early.detach(), before - 0.01 * torchjd_direction[0])
    assert model.late.item() == pytest.approx(0.4)
    aggregation = result["coherence_aggregation"]
    assert aggregation["matrix_rank"] == 1
    assert aggregation["matrix_rank_deficient"] is True
    assert aggregation["zero_gradient_flags"] == {"A": False, "B": False, "T": False}
    torch.testing.assert_close(
        torch.tensor(aggregation["pairwise_gradient_cosines"], dtype=torch.float64),
        torch.ones((3, 3), dtype=torch.float64),
        atol=1.0e-12,
        rtol=0.0,
    )
