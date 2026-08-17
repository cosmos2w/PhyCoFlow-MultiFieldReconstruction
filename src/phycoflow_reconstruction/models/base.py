"""Shared model helpers and masked data-loss behavior for all adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from ..contracts import LossBundle, ModelCapabilities, ObservationBatch, ReconstructionBatch


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(prediction.dtype)
    return ((prediction - target).square() * weights).sum() / (
        weights.sum() * prediction.shape[-1]
    ).clamp_min(1)


def observation_summary(batch: ObservationBatch, num_fields: int) -> torch.Tensor:
    """Return per-field observed means and support flags, never hidden targets."""
    bsz = batch.obs_coords.shape[0]
    sums = torch.zeros(
        bsz, num_fields, device=batch.obs_coords.device, dtype=batch.obs_values.dtype
    )
    counts = torch.zeros_like(sums)
    fields = batch.obs_field_ids.clamp(0, num_fields - 1)
    valid = batch.obs_valid_mask.to(batch.obs_values.dtype)
    sums.scatter_add_(1, fields, batch.obs_values[:, :, 0] * valid)
    counts.scatter_add_(1, fields, valid)
    means = sums / counts.clamp_min(1)
    return torch.cat([means, (counts > 0).to(means.dtype)], dim=-1)


class BaseReconstructionModel(nn.Module, ABC):
    capabilities = ModelCapabilities(
        "point", False, True, False, False, ("base_training", "post_training")
    )

    @abstractmethod
    def forward_batch(self, batch: ObservationBatch) -> torch.Tensor:
        """Return `[B,Q,C]` predictions for the batch query points."""

    def training_loss(self, batch: ObservationBatch) -> LossBundle:
        if batch.target_fields is None:
            raise ValueError("plain base training requires target_fields")
        prediction = self.forward_batch(batch)
        loss = masked_mse(prediction, batch.target_fields, batch.query_valid_mask)
        return LossBundle(loss, {"data_mse": loss})

    def differentiable_reconstruct(
        self,
        batch: ObservationBatch,
        *,
        steps: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Direct regressors/operators need no sampling loop, but retain autograd."""
        del steps, generator
        return self.forward_batch(batch)

    @torch.no_grad()
    def reconstruct(self, batch: ObservationBatch, **_: Any) -> ReconstructionBatch:
        return ReconstructionBatch(self.forward_batch(batch))
