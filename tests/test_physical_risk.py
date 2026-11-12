import itertools

import pytest
import torch

from phycoflow_reconstruction.coherence.physical_risk import (
    blend_risk_quadratic,
    physical_field_mse,
)


def test_without_replacement_matches_native_in_expectation_and_gradient():
    x = torch.tensor(
        [[1.0, -2.0], [3.0, 1.0], [-1.0, 4.0]], dtype=torch.float64, requires_grad=True
    )
    w = x.new_tensor([1.0, 2.0, 7.0])
    full = physical_field_mse(x, torch.zeros_like(x), w)
    draws = [
        physical_field_mse(
            x[list(ids)],
            torch.zeros_like(x[list(ids)]),
            w[list(ids)],
            total_mass=w.sum(),
            sampling="without_replacement",
            probabilities=2 / 3,
        )
        for ids in itertools.combinations(range(3), 2)
    ]
    estimate = torch.stack(draws).mean(0)
    torch.testing.assert_close(estimate, full)
    torch.testing.assert_close(
        torch.autograd.grad(estimate.sum(), x, retain_graph=True)[0],
        torch.autograd.grad(full.sum(), x)[0],
    )


def test_with_replacement_probability_and_duplicates():
    x = torch.tensor([[1.0], [3.0]], dtype=torch.float64)
    w, p = x.new_tensor([1.0, 3.0]), x.new_tensor([0.8, 0.2])
    expectation = sum(
        p[a]
        * p[b]
        * physical_field_mse(
            x[[a, b]],
            torch.zeros_like(x[[a, b]]),
            w[[a, b]],
            total_mass=w.sum(),
            sampling="with_replacement",
            probabilities=p[[a, b]],
        )
        for a, b in itertools.product(range(2), repeat=2)
    )
    torch.testing.assert_close(expectation, physical_field_mse(x, torch.zeros_like(x), w))


def test_native_total_cannot_be_replaced_by_sampled_mass():
    x = torch.ones(2, 3, 4)
    with pytest.raises(ValueError, match="native total"):
        physical_field_mse(x, x, torch.ones(3), sampling="without_replacement", probabilities=0.5)


def test_blend_risk_quadratic_matches_all_alphas_and_gradients():
    source = torch.tensor([[1.0, 3.0], [2.0, -1.0]], dtype=torch.float64)
    child = torch.tensor([[2.0, -1.0], [1.0, 2.0]], dtype=torch.float64, requires_grad=True)
    target, mass = torch.zeros_like(source), source.new_tensor([1.0, 3.0])
    r0, b, a = blend_risk_quadratic(source, child, target, mass)
    for alpha in (0.0, 0.2, 0.5, 1.0):
        polynomial = r0 + 2 * alpha * b + alpha**2 * a
        direct = physical_field_mse(torch.lerp(source, child, alpha), target, mass)
        torch.testing.assert_close(polynomial, direct)
        torch.testing.assert_close(
            torch.autograd.grad(polynomial.sum(), child, retain_graph=True)[0],
            torch.autograd.grad(direct.sum(), child, retain_graph=True)[0],
        )
