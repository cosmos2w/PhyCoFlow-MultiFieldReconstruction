"""Stable global gradient clipping contracts."""

import pytest
import torch

from phycoflow_reconstruction.training.gradients import (
    adaptive_backward_and_clip_,
    stable_clip_grad_norm_,
)


def test_stable_clip_matches_torch_for_ordinary_finite_gradients():
    first = torch.nn.Parameter(torch.zeros(3))
    second = torch.nn.Parameter(torch.zeros(2))
    reference_first = torch.nn.Parameter(torch.zeros(3))
    reference_second = torch.nn.Parameter(torch.zeros(2))
    first.grad = torch.tensor([3.0, 4.0, 0.5])
    second.grad = torch.tensor([-2.0, 1.0])
    reference_first.grad = first.grad.clone()
    reference_second.grad = second.grad.clone()

    expected = torch.nn.utils.clip_grad_norm_([reference_first, reference_second], 1.25)
    actual = stable_clip_grad_norm_([first, second], 1.25)

    assert float(actual) == pytest.approx(float(expected), rel=1.0e-6)
    assert torch.allclose(first.grad, reference_first.grad, rtol=1.0e-6, atol=1.0e-7)
    assert torch.allclose(second.grad, reference_second.grad, rtol=1.0e-6, atol=1.0e-7)


def test_stable_clip_handles_huge_finite_float32_gradients():
    parameter = torch.nn.Parameter(torch.zeros(8, dtype=torch.float32))
    parameter.grad = torch.full_like(parameter, 1.0e30)

    norm = stable_clip_grad_norm_([parameter], 1.0)

    assert torch.isfinite(norm)
    assert float(norm) == pytest.approx(8**0.5 * 1.0e30, rel=1.0e-6)
    assert torch.isfinite(parameter.grad).all()
    clipped_norm = torch.linalg.vector_norm(parameter.grad.to(torch.float64))
    assert float(clipped_norm) <= 1.0 + 1.0e-6


def test_scaled_backward_avoids_overflow_and_unscales_before_clipping():
    parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    loss_scale = 2.0**-140
    loss = ((parameter * 1.0e20) * 1.0e20).sum()

    (loss * loss_scale).backward()
    norm = stable_clip_grad_norm_([parameter], 1.0, gradient_scale=loss_scale)

    assert torch.isfinite(norm)
    assert float(norm) == pytest.approx(2.0e40, rel=1.0e-6)
    assert torch.isfinite(parameter.grad).all()
    clipped_norm = torch.linalg.vector_norm(parameter.grad.to(torch.float64))
    assert float(clipped_norm) <= 1.0 + 1.0e-6


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_stable_clip_rejects_nonfinite_individual_gradients(value):
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([value, 1.0])

    with pytest.raises(FloatingPointError, match="global gradient norm is non-finite"):
        stable_clip_grad_norm_([parameter], 1.0)


def test_stable_clip_accepts_an_empty_gradient_set():
    parameter = torch.nn.Parameter(torch.ones(1))
    assert float(stable_clip_grad_norm_([parameter], 1.0)) == 0.0


def test_adaptive_backward_retries_overflow_with_identical_rng_draw():
    parameter = torch.nn.Parameter(torch.zeros(()))
    draws = []

    class Losses:
        def __init__(self, total):
            self.total = total

    def closure():
        draw = torch.rand(())
        draws.append(draw.detach().clone())
        return Losses(((parameter * 2.0e19) * 2.0e19) + draw * 0.0)

    _, norm, scale, retries = adaptive_backward_and_clip_(
        closure,
        parameter,
        1.0,
        adaptive=True,
    )

    assert retries == 1
    assert scale == 2.0**-1
    assert torch.equal(draws[0], draws[1])
    assert torch.isfinite(norm)
    assert torch.isfinite(parameter.grad)


def test_adaptive_backward_grows_scale_after_complete_underflow():
    parameter = torch.nn.Parameter(torch.ones(()))

    class Losses:
        def __init__(self, total):
            self.total = total

    _, norm, scale, retries = adaptive_backward_and_clip_(
        lambda: Losses(parameter * 1.0e-20),
        parameter,
        1.0,
        initial_scale=2.0**-128,
        adaptive=True,
    )

    assert retries > 0
    assert scale > 2.0**-128
    assert float(norm) > 0.0
    assert torch.isfinite(parameter.grad)
