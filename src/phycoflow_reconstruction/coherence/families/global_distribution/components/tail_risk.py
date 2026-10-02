"""Identity-calibrated smooth upper-tail risk for projection discrepancies."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F


class SmoothCVaRResult(NamedTuple):
    value: torch.Tensor
    eta: torch.Tensor
    eta_residual: torch.Tensor
    direction_weights: torch.Tensor


def identity_calibrated_smooth_cvar(
    values: torch.Tensor,
    *,
    rho: float,
    temperature: float,
    tolerance: float = 1e-6,
    max_iterations: int = 80,
) -> SmoothCVaRResult:
    """Compute a smooth CVaR with the zero-input baseline removed.

    Eta is solved independently for this vector and detached. The returned
    value differentiates through the softplus terms only, which is the
    envelope gradient of the scalar convex eta minimization.
    """
    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("smooth CVaR requires a non-empty vector of direction costs")
    if not torch.isfinite(values).all():
        raise FloatingPointError("smooth CVaR costs must be finite")
    if not 0.0 < float(rho) <= 1.0:
        raise ValueError("smooth CVaR rho must lie in (0,1]")
    if float(temperature) <= 0.0:
        raise ValueError("smooth CVaR temperature must be positive")
    if tolerance <= 0.0:
        raise ValueError("smooth CVaR tolerance must be positive")
    if max_iterations < 1:
        raise ValueError("smooth CVaR max_iterations must be positive")

    count = values.numel()
    if rho == 1.0:
        mean = values.mean()
        weights = torch.full_like(values, 1.0 / count)
        zero = values.new_zeros(())
        return SmoothCVaRResult(mean, zero, zero, weights)

    tau = float(temperature)
    # Eta has no gradient under the envelope theorem. Transfer this small
    # direction vector once, then solve the scalar root on CPU; repeated
    # device-side scalar reads would synchronize the accelerator every
    # bisection iteration.
    host_values = values.detach().to(device="cpu", dtype=torch.float64).tolist()
    minimum = min(host_values)
    maximum = max(host_values)
    span = maximum - minimum
    margin = span + max(32.0 * tau, tau * abs(math.log((1.0 - rho) / rho)))
    low = minimum - margin
    high = maximum + margin

    # The average sigmoid is strictly decreasing in eta. This bisection is
    # deterministic and needs only O(R) storage; there is no persistent
    # position-dependent eta state.
    def sigmoid(value: float) -> float:
        if value >= 0.0:
            return 1.0 / (1.0 + math.exp(-value))
        exponential = math.exp(value)
        return exponential / (1.0 + exponential)

    eta_value = (low + high) / 2.0
    residual_value = math.inf
    for _ in range(max_iterations):
        eta_value = (low + high) / 2.0
        probability_mass = sum(
            sigmoid((value - eta_value) / tau) for value in host_values
        ) / count
        residual_value = abs(probability_mass - rho)
        if residual_value <= tolerance:
            break
        if probability_mass > rho:
            low = eta_value
        else:
            high = eta_value

    eta = values.new_tensor(eta_value, dtype=torch.float64).detach()
    wide_values = values.to(dtype=torch.float64)
    logits = (wide_values - eta) / tau
    smooth_value = eta + tau * F.softplus(logits).sum() / (float(rho) * count)
    baseline = values.new_tensor(
        -tau * math.log(rho)
        - tau * (1.0 - rho) / rho * math.log1p(-rho),
        dtype=torch.float64,
    )
    value = (smooth_value - baseline).to(dtype=values.dtype)
    weights = (
        torch.sigmoid(logits) / (float(rho) * count)
    ).to(dtype=values.dtype)
    return SmoothCVaRResult(
        value,
        eta,
        values.new_tensor(residual_value, dtype=torch.float64),
        weights,
    )
