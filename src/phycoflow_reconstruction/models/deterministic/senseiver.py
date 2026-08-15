"""Compact Perceiver-IO/Senseiver regressor for sparse sensor tokens."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ...contracts import ModelCapabilities, ObservationBatch
from ..base import BaseReconstructionModel
from ..common import AttentionBlock, FourierFeatures, make_mlp


class SenseiverRegressor(BaseReconstructionModel):
    capabilities = ModelCapabilities(
        "point", False, True, False, False, ("base_training", "post_training")
    )

    def __init__(
        self,
        coordinate_dim: int,
        num_fields: int,
        width: int = 128,
        num_latents: int = 32,
        heads: int = 4,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.num_fields = num_fields
        self.position = FourierFeatures(coordinate_dim, 12)
        self.sensor_projection = make_mlp(self.position.out_dim + 1 + num_fields, width, width, 3)
        self.query_projection = make_mlp(self.position.out_dim, width, width, 2)
        self.latents = nn.Parameter(torch.randn(num_latents, width) / width**0.5)
        self.input_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.blocks = nn.ModuleList(AttentionBlock(width, heads) for _ in range(depth))
        self.output_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.head = make_mlp(width, width, num_fields, 2)

    def forward_batch(self, batch: ObservationBatch) -> torch.Tensor:
        field_one_hot = F.one_hot(batch.obs_field_ids, self.num_fields).to(batch.obs_values.dtype)
        sensor_tokens = self.sensor_projection(
            torch.cat((self.position(batch.obs_coords), batch.obs_values, field_one_hot), -1)
        )
        latents = self.latents.unsqueeze(0).expand(batch.obs_coords.shape[0], -1, -1)
        latents = (
            latents
            + self.input_attention(
                latents,
                sensor_tokens,
                sensor_tokens,
                key_padding_mask=~batch.obs_valid_mask,
                need_weights=False,
            )[0]
        )
        for block in self.blocks:
            latents = block(latents)
        queries = self.query_projection(self.position(batch.query_coords))
        decoded = queries + self.output_attention(queries, latents, latents, need_weights=False)[0]
        return self.head(decoded)
