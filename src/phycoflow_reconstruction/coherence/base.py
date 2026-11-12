"""Shared numerical primitives for data-driven coherence families.

These helpers contain no case names, target-selection policy, or hidden
subsampling. Families receive already selected tensors in declared units.
"""

from __future__ import annotations

import math

import torch


def require_field_tensor(name: str, value: torch.Tensor) -> None:
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [B,N,C], got {tuple(value.shape)}")
    if value.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two empirical points")
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains non-finite values")


def empirical_w2_columns(generated: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Squared empirical 1-D Wasserstein distance for columns `[N,K]`."""
    if generated.shape != reference.shape or generated.ndim != 2:
        raise ValueError("empirical Wasserstein inputs must share shape [N,K]")
    generated_sorted = generated.sort(dim=0).values
    reference_sorted = reference.sort(dim=0).values
    return (generated_sorted - reference_sorted).square().mean(dim=0)


def weighted_w2_columns(
    generated: torch.Tensor,
    reference: torch.Tensor,
    generated_weights: torch.Tensor,
    reference_weights: torch.Tensor,
) -> torch.Tensor:
    """Integrate the exact weighted 1-D quantile gap for columns `[N,K]`.

    Endpoint masses are fixed, nonnegative and independently normalized. Each
    set of values carries its masses through its own sort. The merged CDF
    breakpoints define intervals on which both quantiles are constant. Values
    are differentiable away from sort ties; the fixed masses have no gradient.
    """
    if generated.ndim != 2 or reference.ndim != 2 or generated.shape[1] != reference.shape[1]:
        raise ValueError("weighted Wasserstein inputs must have shapes [N,K] and [M,K]")
    if generated.device != reference.device or generated.dtype != reference.dtype:
        raise ValueError("weighted Wasserstein inputs must share device and dtype")
    if not generated.is_floating_point() or generated.shape[0] == 0 or reference.shape[0] == 0:
        raise ValueError("weighted Wasserstein inputs must be nonempty floating-point tensors")
    if not torch.isfinite(generated).all() or not torch.isfinite(reference).all():
        raise ValueError("weighted Wasserstein values must be finite")

    def mass(value: torch.Tensor, count: int) -> torch.Tensor:
        value = torch.as_tensor(value, device=generated.device, dtype=generated.dtype).detach()
        if value.ndim != 1 or value.numel() != count:
            raise ValueError("each weighted Wasserstein mass must align with its endpoint points")
        if not torch.isfinite(value).all() or torch.any(value < 0) or not bool(value.sum() > 0):
            raise ValueError("weighted Wasserstein masses must be finite, nonnegative and nonzero")
        return value / value.sum()

    left = mass(generated_weights, generated.shape[0])
    right = mass(reference_weights, reference.shape[0])
    # Reproduce the historical equal-node calculation bit for bit.
    if (
        generated.shape == reference.shape
        and bool(torch.all(left == left[0]))
        and bool(torch.all(right == right[0]))
    ):
        return empirical_w2_columns(generated, reference)

    generated_sorted, generated_order = generated.sort(dim=0, stable=True)
    reference_sorted, reference_order = reference.sort(dim=0, stable=True)
    left_sorted = left[:, None].expand_as(generated).gather(0, generated_order)
    right_sorted = right[:, None].expand_as(reference).gather(0, reference_order)

    def cumulative(weights: torch.Tensor) -> torch.Tensor:
        # The final breakpoint is exactly one even after floating-point sums.
        cdf = weights.cumsum(dim=0).clamp(max=1.0)
        return torch.cat((cdf[:-1], torch.ones_like(cdf[-1:])), dim=0)

    left_cdf, right_cdf = cumulative(left_sorted), cumulative(right_sorted)
    breaks = torch.cat((left_cdf, right_cdf), dim=0).sort(dim=0).values
    widths = torch.diff(breaks, dim=0, prepend=torch.zeros_like(breaks[:1]))
    # `searchsorted` works row-wise, hence the transpose to [K,N].
    query = breaks.T.contiguous()
    left_index = torch.searchsorted(left_cdf.T.contiguous(), query).T
    right_index = torch.searchsorted(right_cdf.T.contiguous(), query).T
    left_values = generated_sorted.gather(0, left_index.clamp_max(generated.shape[0] - 1))
    right_values = reference_sorted.gather(0, right_index.clamp_max(reference.shape[0] - 1))
    return (widths * (left_values - right_values).square()).sum(dim=0)


def normalize_directions(directions: torch.Tensor) -> torch.Tensor:
    return directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def projection_bank(
    dimensions: int,
    count: int,
    *,
    seed: int,
    include_axes: bool = False,
    qmc: bool = True,
) -> torch.Tensor:
    """Create a deterministic CPU bank, later serialized with family state."""
    if dimensions < 1 or count < 1:
        raise ValueError("projection dimensions and count must be positive")
    parts: list[torch.Tensor] = []
    if include_axes:
        parts.append(torch.eye(dimensions)[: min(dimensions, count)])
    remaining = count - sum(part.shape[0] for part in parts)
    random_part = None
    if remaining > 0 and qmc:
        engine = torch.quasirandom.SobolEngine(dimensions, scramble=True, seed=int(seed))
        uniform = engine.draw(remaining).clamp(torch.finfo(torch.float32).eps, 1.0 - 1e-7)
        random_part = math.sqrt(2.0) * torch.erfinv(2.0 * uniform - 1.0)
    if remaining > 0 and random_part is None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        random_part = torch.randn(remaining, dimensions, generator=generator)
    if random_part is not None:
        parts.append(random_part)
    return normalize_directions(torch.cat(parts, dim=0)[:count].float())
