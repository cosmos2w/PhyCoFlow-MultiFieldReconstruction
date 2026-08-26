"""Tensor-level rectified-flow training helpers for downstream integrations."""

from __future__ import annotations

from typing import Any, Protocol

import torch


class RectifiedFlowModel(Protocol):
    """Minimal training protocol implemented by :class:`PointCloudFFM`."""

    def training_loss(self, **kwargs: Any) -> tuple[torch.Tensor, dict[str, float]]: ...

    def training_loss_microbatched(
        self, **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, float]]: ...


def rectified_flow_loss(
    model: RectifiedFlowModel,
    *,
    x1: torch.Tensor,
    coords: torch.Tensor,
    obs_coords: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_field_ids: torch.Tensor,
    obs_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Evaluate the unchanged RF objective on downstream-owned tensors."""
    return model.training_loss(
        x1=x1,
        coords=coords,
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_mask=obs_mask,
        obs_field_ids=obs_field_ids,
        obs_indices=obs_indices,
    )


def rectified_flow_loss_microbatched(
    model: RectifiedFlowModel,
    *,
    x1: torch.Tensor,
    coords: torch.Tensor,
    obs_coords: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_field_ids: torch.Tensor,
    obs_indices: torch.Tensor | None = None,
    query_microbatch_size: int,
    backward: bool = False,
    reuse_condition_context: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Evaluate one coherent RF bridge with bounded query activations."""
    return model.training_loss_microbatched(
        x1=x1,
        coords=coords,
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_mask=obs_mask,
        obs_field_ids=obs_field_ids,
        obs_indices=obs_indices,
        query_microbatch_size=query_microbatch_size,
        backward=backward,
        reuse_condition_context=reuse_condition_context,
    )


__all__ = [
    "RectifiedFlowModel",
    "rectified_flow_loss",
    "rectified_flow_loss_microbatched",
]
