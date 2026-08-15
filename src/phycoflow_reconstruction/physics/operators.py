"""Small differentiable spatial operators shared by case-owned providers."""

from __future__ import annotations

import math

import torch


def reshape_fields(values: torch.Tensor, logical_shape: tuple[int, ...]) -> torch.Tensor:
    """Convert `[B,N,C]` values to `[B,C,*logical_shape]`."""
    point_count = math.prod(logical_shape)
    if values.ndim != 3 or values.shape[1] != point_count:
        raise ValueError(
            f"physics requires the complete logical grid {logical_shape}, got {tuple(values.shape)}"
        )
    return values.transpose(1, 2).reshape(values.shape[0], values.shape[2], *logical_shape)


def periodic_derivative(
    field: torch.Tensor,
    *,
    axis: int,
    domain_length: float,
    order: int = 1,
) -> torch.Tensor:
    """Spectral derivative along a periodic spatial tensor axis."""
    if order not in {1, 2}:
        raise ValueError("periodic_derivative supports first or second order")
    size = field.shape[axis]
    spacing = float(domain_length) / size
    frequencies = (
        2.0 * math.pi * torch.fft.fftfreq(size, d=spacing, device=field.device, dtype=field.dtype)
    )
    shape = [1] * field.ndim
    shape[axis] = size
    multiplier = (1j * frequencies.reshape(shape)) ** order
    transformed = torch.fft.fft(field, dim=axis)
    return torch.fft.ifft(transformed * multiplier, dim=axis).real


def periodic_laplacian_2d(field: torch.Tensor, domain_length: float) -> torch.Tensor:
    return periodic_derivative(
        field, axis=-1, domain_length=domain_length, order=2
    ) + periodic_derivative(field, axis=-2, domain_length=domain_length, order=2)


def nonperiodic_gradient_2d(
    field: torch.Tensor, spacing_y: float, spacing_x: float
) -> tuple[torch.Tensor, torch.Tensor]:
    derivative_y, derivative_x = torch.gradient(
        field, spacing=(float(spacing_y), float(spacing_x)), dim=(-2, -1)
    )
    return derivative_x, derivative_y


def relative_rms(residual: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    numerator = residual.square().mean().sqrt()
    denominator = scale.square().mean().sqrt().clamp_min(1e-8)
    return numerator / denominator
