"""Observation-conditioned, point-local correction of a frozen source model."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn

from ..contracts import ObservationBatch


class FrozenSourceResidual(nn.Module):
    """Keep the source function intact and learn a bounded fieldwise displacement.

    ``field_scales`` are fixed TRAIN-derived scales in the source output units.
    The head never aggregates over queries, so its result for a given source
    value does not depend on the other query points present in the call.
    """

    def __init__(self, source: nn.Module, *, coordinate_dim: int, field_scales,
                 width: int = 64, max_fraction: float = 0.25) -> None:
        super().__init__()
        scales = torch.as_tensor(field_scales, dtype=torch.float32)
        if scales.ndim != 1 or scales.numel() == 0 or not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("field_scales must be a nonempty vector of finite positive TRAIN scales")
        if coordinate_dim < 1 or width < 1 or not 0 < max_fraction <= 1:
            raise ValueError("coordinate_dim and width must be positive; max_fraction must be in (0, 1]")
        self.source = source
        self.capabilities = source.capabilities
        self.num_fields = scales.numel()
        self.coordinate_dim = coordinate_dim
        self.max_fraction = float(max_fraction)
        self.register_buffer("field_scales", scales.clone())
        self.observation_encoder = nn.Sequential(
            nn.Linear(coordinate_dim + 1 + self.num_fields, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(coordinate_dim + self.num_fields + width + 1, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, self.num_fields),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.freeze_source()

    def freeze_source(self) -> None:
        self.source.requires_grad_(False)
        self.source.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.source.eval()
        return self

    def correction(self, batch: ObservationBatch, source_prediction: torch.Tensor) -> torch.Tensor:
        """Calculate a sensor-null correction from observations and source output only."""
        valid = batch.obs_valid_mask
        ids = batch.obs_field_ids
        if batch.obs_indices is None or not isinstance(batch.metadata.get("query_indices"), torch.Tensor):
            raise ValueError("residual sensor nulling requires observation and query point IDs")
        if torch.any((ids[valid] < 0) | (ids[valid] >= self.num_fields)):
            raise ValueError("valid observation field ID is out of range")
        if source_prediction.shape != (*batch.query_coords.shape[:2], self.num_fields):
            raise ValueError("source prediction must align with query points and fields")
        safe_ids = ids.masked_fill(~valid, 0).long()
        safe_coords = torch.where(valid[..., None], batch.obs_coords, torch.zeros_like(batch.obs_coords))
        safe_values = torch.where(valid[..., None], batch.obs_values, torch.zeros_like(batch.obs_values))
        values = safe_values / self.field_scales[safe_ids].unsqueeze(-1)
        tokens = torch.cat((safe_coords, values, F.one_hot(safe_ids, self.num_fields).to(values)), dim=-1)
        encoded = self.observation_encoder(tokens)
        counts = valid.sum(dim=1, keepdim=True)
        pooled = (encoded * valid[..., None]).sum(dim=1) / counts.clamp_min(1)
        count_feature = torch.log1p(counts.to(pooled.dtype))
        context = torch.cat((pooled, count_feature), dim=-1)
        head_input = torch.cat((batch.query_coords, source_prediction / self.field_scales,
                                context[:, None, :].expand(-1, source_prediction.shape[1], -1)), dim=-1)
        delta = torch.tanh(self.head(head_input)) * self.field_scales * self.max_fraction
        query_ids = batch.metadata["query_indices"]
        if query_ids.shape != batch.query_coords.shape[:2]:
            raise ValueError("query point IDs must align with query coordinates")
        null = torch.zeros_like(delta, dtype=torch.bool)
        for row in range(delta.shape[0]):
            for field in range(self.num_fields):
                observed = batch.obs_indices[row, valid[row] & (ids[row] == field)]
                if observed.numel():
                    null[row, :, field] = torch.isin(query_ids[row], observed)
        null |= ~batch.query_valid_mask[..., None]
        return delta.masked_fill(null, 0)

    def reconstruct_with_observation_config(self, batch: ObservationBatch, *, steps: int,
                                            solver: str, generator: torch.Generator,
                                            observation_config: dict) -> torch.Tensor:
        from ..training.rollout import differentiable_reconstruction

        self.freeze_source()
        inference_batch = replace(batch, target_fields=None)
        with torch.no_grad():
            prediction = differentiable_reconstruction(
                self.source, inference_batch, steps=steps, solver=solver,
                generator=generator, observation_config=observation_config,
            )
        return prediction + self.correction(inference_batch, prediction)
