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
