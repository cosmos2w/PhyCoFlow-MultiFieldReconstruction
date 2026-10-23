from __future__ import annotations

import copy
from importlib.metadata import version

import pytest
import torch
from torch import nn

from phycoflow_reconstruction.training.periodic_config import (
    PeriodicConfigState,
    extract_config_coefficients,
    residual_gradient_diagnostic,
)


def _oracle():
    pytest.importorskip("torchjd")
    from torchjd.aggregation import ConFIG

    assert version("torchjd") == "0.17.1"
    return ConFIG()


def _gradient_case(name: str, *, width: int = 2048) -> torch.Tensor:
    dtype = torch.float64
    if name == "aligned":
        generator = torch.Generator().manual_seed(101)
        base = torch.randn(width, generator=generator, dtype=dtype)
        return torch.stack((base, 2.0 * base, 0.5 * base))
    if name == "conflicting":
        generator = torch.Generator().manual_seed(202)
        return torch.randn((3, width), generator=generator, dtype=dtype)
    if name == "nearly_dependent":
        generator = torch.Generator().manual_seed(303)
        base = torch.randn(width, generator=generator, dtype=dtype)
        perturbation = torch.randn(width, generator=generator, dtype=dtype)
        return torch.stack(
            (base, base + 1.0e-15 * perturbation, 0.7 * base - 0.2 * perturbation)
        )
    if name == "signed":
        generator = torch.Generator().manual_seed(31)
        compact = torch.randn((3, 12), generator=generator, dtype=dtype)
        result = torch.zeros((3, width), dtype=dtype)
        result[:, :12] = compact
        return result
    raise AssertionError(f"unknown fixture {name}")


@pytest.mark.parametrize(
    ("name", "expected_rank"),
    [
        ("aligned", 1),
        ("conflicting", 3),
        ("nearly_dependent", 2),
        ("signed", 3),
    ],
)
def test_coefficients_reconstruct_pinned_full_width_float64_config(
    name: str, expected_rank: int
) -> None:
    oracle = _oracle()
    gradients = _gradient_case(name)
    oracle_direction = oracle(gradients)

    recovered = extract_config_coefficients(
        gradients, oracle_direction=oracle_direction
    )

    torch.testing.assert_close(
        recovered.reconstructed_direction,
        oracle_direction,
        atol=2.0e-11,
        rtol=2.0e-11,
    )
    assert recovered.rank == expected_rank
    assert recovered.relative_reconstruction_error < 2.0e-11
    if name == "signed":
        assert any(sign < 0 for sign in recovered.coefficient_signs)
        assert any(sign > 0 for sign in recovered.coefficient_signs)
    else:
        assert all(sign >= 0 for sign in recovered.coefficient_signs)


def test_coefficient_extraction_can_call_the_pinned_oracle() -> None:
    gradients = _gradient_case("conflicting", width=96)
    recovered = extract_config_coefficients(gradients)
    expected = _oracle()(gradients)
    torch.testing.assert_close(recovered.oracle_direction, expected)


def test_zero_rows_and_rank_zero_are_explicit() -> None:
    oracle = _oracle()
    one_zero_row = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, -2.0, 0.5]], dtype=torch.float64
    )
    recovered = extract_config_coefficients(one_zero_row, oracle_direction=oracle(one_zero_row))
    assert recovered.rank == 1
    assert recovered.coefficients[0].item() == 0.0
    assert recovered.relative_reconstruction_error < 1.0e-12

    all_zero = torch.zeros((3, 7), dtype=torch.float64)
    recovered_zero = extract_config_coefficients(all_zero, oracle_direction=oracle(all_zero))
    assert recovered_zero.rank == 0
    assert torch.equal(recovered_zero.coefficients, torch.zeros_like(recovered_zero.coefficients))
    assert torch.equal(
        recovered_zero.reconstructed_direction,
        torch.zeros_like(recovered_zero.reconstructed_direction),
    )


def test_coefficient_recovery_rejects_direction_outside_gradient_row_space() -> None:
    gradients = torch.tensor([[1.0, 2.0], [2.0, 4.0]], dtype=torch.float64)
    with pytest.raises(FloatingPointError, match="exceeded tolerance"):
        extract_config_coefficients(
            gradients,
            oracle_direction=torch.tensor([2.0, -1.0], dtype=torch.float64),
        )


def test_interval_one_scalar_reuse_matches_oracle_adamw_displacement() -> None:
    oracle = _oracle()
    gradients = _gradient_case("signed", width=12)
    direction = oracle(gradients)
    recovered = extract_config_coefficients(gradients, oracle_direction=direction)
    names = ("A", "B", "T")

    initial = torch.linspace(-0.8, 0.9, 12, dtype=torch.float64)
    reused_parameter = nn.Parameter(initial.clone())
    dense_parameter = nn.Parameter(initial.clone())
    state = PeriodicConfigState(interval=1, compatibility_key="unit-objective-v1")
    assert state.refresh_due
    state.refresh(names, recovered.coefficients)
    assert not state.refresh_due

    current_losses = {
        name: (row * reused_parameter).sum()
        for name, row in zip(names, gradients)
    }
    state.reuse_objective(current_losses, scale=1.0).backward()
    dense_parameter.grad = direction.clone()

    reused_optimizer = torch.optim.AdamW(
        [reused_parameter], lr=0.025, weight_decay=0.11, betas=(0.8, 0.95)
    )
    dense_optimizer = torch.optim.AdamW(
        [dense_parameter], lr=0.025, weight_decay=0.11, betas=(0.8, 0.95)
    )
    reused_optimizer.step()
    dense_optimizer.step()
    torch.testing.assert_close(
        reused_parameter.detach(), dense_parameter.detach(), atol=2.0e-11, rtol=2.0e-11
    )

    state.advance()
    assert state.refresh_due


def test_reuse_uses_fresh_loss_graph_and_divides_by_scale_once() -> None:
    state = PeriodicConfigState(interval=4, compatibility_key="fresh-losses-v1")
    signed = torch.tensor([1.25, -0.4, 0.75], dtype=torch.float64)
    state.refresh(("A", "B", "T"), signed)
    state.advance()

    parameter = nn.Parameter(torch.tensor([1.4, -0.7], dtype=torch.float64))
    losses = {
        "A": (parameter - parameter.new_tensor([0.1, 0.5])).square().sum(),
        "B": (2.0 * parameter + 0.3).square().sum(),
        "T": (parameter[0] - 3.0 * parameter[1]).square(),
    }
    expected = sum(
        (signed[index] * torch.autograd.grad(loss, parameter, retain_graph=True)[0]
         for index, loss in enumerate(losses.values())),
        torch.zeros_like(parameter),
    ) / 2.5
    objective = state.reuse_objective(losses, scale=2.5)
    objective.backward()
    torch.testing.assert_close(parameter.grad, expected)
    assert objective.item() == pytest.approx(
        sum(float(signed[index]) * float(loss.detach()) for index, loss in enumerate(losses.values()))
        / 2.5
    )


def test_periodic_state_resume_preserves_sign_cursor_and_compatibility() -> None:
    state = PeriodicConfigState(interval=4, compatibility_key="A-B-T/source/hash")
    state.refresh(("A", "B", "T"), (1.5, -0.375, 0.25))
    state.advance()
    state.advance()
    state.advance()
    assert not state.refresh_due
    assert state.age == 3

    checkpoint = copy.deepcopy(state.state_dict())
    restored = PeriodicConfigState(interval=4, compatibility_key="A-B-T/source/hash")
    restored.load_state_dict(checkpoint)
    assert restored.objective_names == ("A", "B", "T")
    assert restored.coefficients == (1.5, -0.375, 0.25)
    assert restored.age == 3
    assert not restored.refresh_due

    restored.advance()
    assert restored.age == 4
    assert restored.refresh_due
    with pytest.raises(RuntimeError, match="absent or due"):
        restored.reuse_objective(
            {name: torch.tensor(1.0, requires_grad=True) for name in ("A", "B", "T")},
            scale=1.0,
        )

    with pytest.raises(ValueError, match="incompatible"):
        PeriodicConfigState(interval=4, compatibility_key="different/source").load_state_dict(
            checkpoint
        )
    with pytest.raises(ValueError, match="interval"):
        PeriodicConfigState(interval=2, compatibility_key="A-B-T/source/hash").load_state_dict(
            checkpoint
        )


def test_periodic_interval_four_refreshes_after_four_accepted_steps() -> None:
    state = PeriodicConfigState(interval=4, compatibility_key="cadence-v1")
    assert state.refresh_due
    state.refresh(("A", "B", "T"), (1.0, -0.5, 0.25))
    for age in (1, 2, 3):
        state.advance()
        assert state.age == age
        assert not state.refresh_due
    state.advance()
    assert state.age == 4
    assert state.refresh_due


def test_residual_gradient_diagnostic_projects_only_lower_row_space() -> None:
    lower = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([2.0, -3.0, 4.0], dtype=torch.float64)

    diagnostic = residual_gradient_diagnostic(lower, target)

    torch.testing.assert_close(
        diagnostic.residual,
        torch.tensor([0.0, -3.0, 4.0], dtype=torch.float64),
    )
    assert diagnostic.rank == 1
    assert diagnostic.residual_norm_fraction == pytest.approx(5.0 / (29.0**0.5))
    assert diagnostic.lower_target_cosines == pytest.approx((2.0 / (29.0**0.5),) * 2)
    assert diagnostic.improves_target_to_first_order


def test_residual_gradient_diagnostic_reports_zero_tangent_for_dependent_target() -> None:
    lower = torch.tensor([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([3.0, 3.0, 0.0], dtype=torch.float64)
    diagnostic = residual_gradient_diagnostic(lower, target)

    assert diagnostic.rank == 1
    assert diagnostic.residual_norm_fraction < 1.0e-14
    assert not diagnostic.improves_target_to_first_order


def test_residual_gradient_diagnostic_without_lower_rows_keeps_full_target() -> None:
    lower = torch.empty((0, 4), dtype=torch.float64)
    target = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float64)

    diagnostic = residual_gradient_diagnostic(lower, target)

    torch.testing.assert_close(diagnostic.residual, target)
    assert diagnostic.rank == 0
    assert diagnostic.residual_norm_fraction == pytest.approx(1.0)
