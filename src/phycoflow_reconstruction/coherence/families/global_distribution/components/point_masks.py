"""Small point-mask helpers shared by distribution components."""

from __future__ import annotations

import torch


def validate_point_mask(
    point_mask: torch.Tensor | None,
    *,
    batch_size: int,
    point_count: int,
) -> None:
    if point_mask is None:
        return
    if point_mask.dtype != torch.bool:
        raise TypeError("point_mask must use boolean dtype")
    if point_mask.shape not in {(point_count,), (batch_size, point_count)}:
        raise ValueError(
            "point_mask must have shape [N] or [B,N] and align with the input tensors"
        )


def point_mask_for_batch(
    point_mask: torch.Tensor | None,
    batch_index: int,
    *,
    batch_size: int,
    point_count: int,
    device: torch.device,
) -> torch.Tensor | None:
    validate_point_mask(point_mask, batch_size=batch_size, point_count=point_count)
    if point_mask is None:
        return None
    current = point_mask if point_mask.ndim == 1 else point_mask[batch_index]
    return current.to(device=device)


def select_valid_points(
    values: torch.Tensor,
    point_mask: torch.Tensor | None,
    *,
    minimum_count: int = 2,
) -> torch.Tensor:
    if point_mask is None:
        selected = values
    else:
        selected = values[point_mask]
    if selected.shape[0] < minimum_count:
        raise ValueError(
            f"point_mask must retain at least {minimum_count} points per snapshot"
        )
    return selected
