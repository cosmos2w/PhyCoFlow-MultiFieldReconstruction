"""Numerically stable shared gradient utilities."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import nn

_MIN_FLOAT32_POWER_OF_TWO = 2.0**-149
_ADAPTIVE_SCALE_FACTOR = 2.0


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
            # Apply potentially large unscale factors in float64, then copy the
            # final clipped value back. This keeps very small power-of-two loss
            # scales safe even when the unclipped norm later falls below one.
            clipped = values.to(dtype=torch.float64) * coefficient.to(values.device)
            values.copy_(clipped.to(dtype=values.dtype))
    return total_norm


def adaptive_backward_and_clip_(
    loss_closure: Callable[[], Any],
    parameters: Iterable[nn.Parameter] | nn.Parameter,
    max_norm: float,
    *,
    initial_scale: float = 1.0,
    adaptive: bool = False,
    device: torch.device | str = "cpu",
    start_phase: Callable[[str], None] | None = None,
    end_phase: Callable[[str], None] | None = None,
) -> tuple[Any, torch.Tensor, float, int]:
    """Run an exact-RNG backward retry and return clipped finite gradients.

    A failed scaled backward never reaches the optimizer.  When adaptation is
    enabled, the closure is recomputed after restoring the CPU and device RNG
    states captured before the logical step.  This preserves stochastic model
    draws while allowing a power-of-two scale to move down on overflow or up
    on complete float32 underflow.
    """

    scale = float(initial_scale)
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("initial_scale must be finite and in (0, 1]")
    if (start_phase is None) != (end_phase is None):
        raise ValueError("start_phase and end_phase must be provided together")
    parameter_list = [parameters] if isinstance(parameters, nn.Parameter) else list(parameters)
    target_device = torch.device(device)
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state(target_device) if target_device.type == "cuda" else None
    )
    retries = 0
    attempted_scales = {scale}

    while True:
        if retries:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, target_device)
        for parameter in parameter_list:
            parameter.grad = None

        if start_phase is not None:
            start_phase("forward_native_loss")
        losses = loss_closure()
        if end_phase is not None:
            end_phase("forward_native_loss")
        total = losses.total
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f"non-finite loss before backward: {total}")

        if start_phase is not None:
            start_phase("backward")
        (total * scale).backward()
        if end_phase is not None:
            end_phase("backward")
        try:
            norm = stable_clip_grad_norm_(
                parameter_list,
                max_norm,
                gradient_scale=scale,
            )
        except FloatingPointError:
            if not adaptive or scale <= _MIN_FLOAT32_POWER_OF_TWO:
                raise
            next_scale = max(
                _MIN_FLOAT32_POWER_OF_TWO, scale / _ADAPTIVE_SCALE_FACTOR
            )
            if next_scale in attempted_scales:
                raise FloatingPointError(
                    "no float32 loss scale avoids both gradient overflow and underflow"
                )
            scale = next_scale
            attempted_scales.add(scale)
            retries += 1
            continue

        if adaptive and float(norm) == 0.0 and scale < 1.0:
            next_scale = min(1.0, scale * _ADAPTIVE_SCALE_FACTOR)
            if next_scale in attempted_scales:
                raise FloatingPointError(
                    "no float32 loss scale avoids both gradient overflow and underflow"
                )
            scale = next_scale
            attempted_scales.add(scale)
            retries += 1
            continue
        return losses, norm, scale, retries
