"""Empirical copula coordinates for global-distribution coherence.

The exact transform is a detached diagnostic. The training transform is a
quantile-landmark smooth-CDF quadrature whose generated side remains
differentiable through centering, scale normalization, quantile landmarks, and
the sigmoid CDF evaluation. It approximates ranks and is not exactly invariant
to nonlinear monotone transforms.
"""

from __future__ import annotations

import torch

from .point_masks import select_valid_points


def exact_midrank_uniform(
    values: torch.Tensor,
    point_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return exact empirical midrank coordinates for `[N,C]` values.

    Ties receive their average one-based rank before conversion to the uniform
    cube. If a point mask is provided, masked rows are excluded and only valid
    rows are returned. The diagnostic deliberately has no gradient.
    """
    if values.ndim != 2:
        raise ValueError("exact midranks require values with shape [N,C]")
    values = select_valid_points(values, point_mask)
    if not torch.isfinite(values).all():
        raise FloatingPointError("exact midranks require finite values")
    values = values.detach()
    point_count, channel_count = values.shape
    sorted_values, order = values.sort(dim=0, stable=True)
    positions = torch.arange(point_count, device=values.device).unsqueeze(1)
    positions = positions.expand(point_count, channel_count)

    starts = torch.ones_like(sorted_values, dtype=torch.bool)
    starts[1:] = sorted_values[1:] != sorted_values[:-1]
    start_positions = torch.where(starts, positions, 0).cummax(dim=0).values

    ends = torch.ones_like(sorted_values, dtype=torch.bool)
    ends[:-1] = sorted_values[:-1] != sorted_values[1:]
    end_positions = torch.where(ends, positions, point_count - 1)
    end_positions = end_positions.flip(0).cummin(dim=0).values.flip(0)

    midranks = (start_positions.to(values.dtype) + end_positions.to(values.dtype)) / 2.0 + 1.0
    sorted_uniform = (midranks - 0.5) / point_count
    uniform = torch.empty_like(sorted_uniform)
    uniform.scatter_(dim=0, index=order, src=sorted_uniform)
    return uniform


def quantile_landmark_smooth_cdf(
    values: torch.Tensor,
    *,
    landmarks: int = 64,
    bandwidth: float = 0.1,
    scale_floor: float = 1e-6,
    chunk_size: int = 512,
    point_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Approximate per-channel ranks using a smooth quantile-landmark CDF.

    The operation has linear temporary memory in the point count because it
    evaluates only `chunk_size × landmarks` comparisons at a time. For
    generated values, autograd follows the normalization and quantile
    landmarks. Call with a detached reference when no target gradient is
    wanted.
    """
    if values.ndim != 2:
        raise ValueError("smooth copula coordinates require values with shape [N,C]")
    if landmarks < 2:
        raise ValueError("copula landmarks must be at least two")
    if bandwidth <= 0.0:
        raise ValueError("copula bandwidth must be positive")
    if scale_floor <= 0.0:
        raise ValueError("copula scale_floor must be positive")
    if chunk_size < 1:
        raise ValueError("copula chunk_size must be positive")
    values = select_valid_points(values, point_mask)
    if not torch.isfinite(values).all():
        raise FloatingPointError("smooth copula coordinates require finite values")

    centered = values - values.mean(dim=0, keepdim=True)
    # The smooth floor avoids the undefined derivative of sqrt(mean(x^2)) at
    # an exactly constant channel while asymptoting to the RMS scale above it.
    scale = torch.sqrt(
        centered.square().mean(dim=0, keepdim=True) + scale_floor**2
    )
    normalized = centered / scale
    quantile_levels = torch.linspace(
        0.5 / landmarks,
        1.0 - 0.5 / landmarks,
        landmarks,
        dtype=values.dtype,
        device=values.device,
    )
    quantile_landmarks = torch.quantile(normalized, quantile_levels, dim=0).transpose(0, 1)

    chunks = []
    for start in range(0, values.shape[0], chunk_size):
        current = normalized[start : start + chunk_size]
        logits = (
            current.unsqueeze(-1) - quantile_landmarks.unsqueeze(0)
        ) / bandwidth
        chunks.append(torch.sigmoid(logits).mean(dim=-1))
    return torch.cat(chunks, dim=0)
