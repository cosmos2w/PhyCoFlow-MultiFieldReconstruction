"""Numerically stable shared gradient utilities."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn


def stable_clip_grad_norm_(
    parameters: Iterable[nn.Parameter] | nn.Parameter,
    max_norm: float,
    *,
    gradient_scale: float = 1.0,
    error_if_nonfinite: bool = True,
) -> torch.Tensor:
    """Clip a global L2 gradient norm without float32 reduction overflow.

    PyTorch's standard helper reduces in the gradient dtype. Large, otherwise
    finite float32 gradients can therefore produce an infinite aggregate norm
    before clipping. Accumulating squared magnitudes in float64 preserves the
    intended global-norm clipping semantics for those gradients.
    """

    limit = float(max_norm)
    if not math.isfinite(limit) or limit < 0:
        raise ValueError("max_norm must be finite and non-negative")
    scale = float(gradient_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("gradient_scale must be finite and positive")
    if isinstance(parameters, nn.Parameter):
        parameters = [parameters]
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return torch.tensor(0.0, dtype=torch.float64)

    reference_device = gradients[0].device
    total_squared = torch.zeros((), dtype=torch.float64, device=reference_device)
    for gradient in gradients:
        values = gradient.coalesce().values() if gradient.is_sparse else gradient
        contribution = values.detach().abs().to(dtype=torch.float64).square().sum()
        total_squared.add_(contribution.to(reference_device))
    total_norm = total_squared.sqrt() / scale

    if not bool(torch.isfinite(total_norm)):
        if error_if_nonfinite:
            raise FloatingPointError(
                "global gradient norm is non-finite; refusing to corrupt optimizer state"
            )
        return total_norm

    coefficient = (limit / (total_norm + 1.0e-6)).clamp(max=1.0) / scale
    with torch.no_grad():
        for gradient in gradients:
            values = gradient._values() if gradient.is_sparse else gradient
            values.mul_(coefficient.to(device=values.device, dtype=values.dtype))
    return total_norm
