"""Enhanced global-local point backbone with the sole new gather mode `topk_rbf`.

The model tokenizes sparse sensors, repeatedly injects them into a global latent
array, reads latents at each query, and combines that global feature with a
top-k local RBF aggregate. Historical gather variants are intentionally absent.
"""

from __future__ import annotations

import torch
from torch import nn

from ...contracts import ObservationBatch
from ..common import AttentionBlock, FourierFeatures, make_mlp


class EnhancedGLRBFTopK(nn.Module):
    def __init__(
        self,
        coordinate_dim: int,
        num_fields: int,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        num_latents: int = 32,
        heads: int = 4,
        latent_blocks: int = 2,
        gather_topk: int = 16,
        rbf_sigma: float = 0.08,
        fourier_bands: int = 16,
        query_chunk_size: int = 2048,
    ) -> None:
        super().__init__()
        self.num_fields = num_fields
        self.gather_topk = int(gather_topk)
        self.query_chunk_size = int(query_chunk_size)
        if self.query_chunk_size < 1:
            raise ValueError("query_chunk_size must be positive")
        self.log_rbf_sigma = nn.Parameter(torch.log(torch.tensor(float(rbf_sigma))))
        self.position = FourierFeatures(coordinate_dim, fourier_bands)
        self.field_embedding = nn.Embedding(num_fields, 24)

        self.sensor_projection = make_mlp(self.position.out_dim + 1 + 24, latent_dim, latent_dim, 3)
        self.point_projection = make_mlp(
            self.position.out_dim + num_fields + 1, hidden_dim, hidden_dim, 3
        )
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) / latent_dim**0.5)
        self.input_attention = nn.MultiheadAttention(latent_dim, heads, batch_first=True)
        self.latent_blocks = nn.ModuleList(
            AttentionBlock(latent_dim, heads) for _ in range(latent_blocks)
        )
        self.reinjection = nn.ModuleList(
            nn.MultiheadAttention(latent_dim, heads, batch_first=True) for _ in range(latent_blocks)
        )
        self.sensor_readback = nn.MultiheadAttention(latent_dim, heads, batch_first=True)
        self.query_projection = make_mlp(self.position.out_dim, latent_dim, latent_dim, 2)
        self.query_readout = nn.MultiheadAttention(latent_dim, heads, batch_first=True)
        self.head = make_mlp(hidden_dim + latent_dim * 3, hidden_dim, num_fields, 3)
        self.head_norm = nn.LayerNorm(hidden_dim + latent_dim * 3)

    @staticmethod
    def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        expanded = values[:, None].expand(-1, indices.shape[1], -1, -1)
        gather_indices = indices.unsqueeze(-1).expand(-1, -1, -1, values.shape[-1])
        return torch.gather(expanded, 2, gather_indices)

    def _local_features(
        self,
        batch: ObservationBatch,
        sensor_tokens: torch.Tensor,
    ) -> torch.Tensor:
        sigma = self.log_rbf_sigma.exp().clamp_min(1e-4)
        k = min(self.gather_topk, batch.obs_coords.shape[1])
        outputs: list[torch.Tensor] = []
        # Chunk the O(Q*M) distance calculation so a full KS (t,x) state or
        # combustion point cloud does not allocate one global distance matrix.
        for start in range(0, batch.query_coords.shape[1], self.query_chunk_size):
            stop = min(start + self.query_chunk_size, batch.query_coords.shape[1])
            distances = torch.cdist(batch.query_coords[:, start:stop], batch.obs_coords).square()
            distances = distances.masked_fill(~batch.obs_valid_mask[:, None, :], torch.inf)
            distance_topk, indices = torch.topk(distances, k=k, dim=-1, largest=False)
            weights = torch.exp(-distance_topk / (2 * sigma.square()))
            gathered = self._gather(sensor_tokens, indices)
            outputs.append(
                (gathered * weights.unsqueeze(-1)).sum(2)
                / weights.sum(2, keepdim=True).clamp_min(1e-8)
            )
        return torch.cat(outputs, dim=1)

    def _global_readout(self, query_tokens: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for start in range(0, query_tokens.shape[1], self.query_chunk_size):
            stop = min(start + self.query_chunk_size, query_tokens.shape[1])
            chunk = query_tokens[:, start:stop]
            outputs.append(
                chunk + self.query_readout(chunk, latents, latents, need_weights=False)[0]
            )
        return torch.cat(outputs, dim=1)

    def forward(
        self,
        batch: ObservationBatch,
        state: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        # Sparse sensor encoding carries coordinate, value, and measured field identity.
        sensor_tokens = self.sensor_projection(
            torch.cat(
                (
                    self.position(batch.obs_coords),
                    batch.obs_values,
                    self.field_embedding(batch.obs_field_ids),
                ),
                dim=-1,
            )
        )
        latents = self.latents.unsqueeze(0).expand(state.shape[0], -1, -1)
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

        # Enhanced backbone: global processing plus sensor-to-latent re-injection at every block.
        for block, reinject in zip(self.latent_blocks, self.reinjection):
            latents = block(latents)
            latents = (
                latents
                + reinject(
                    latents,
                    sensor_tokens,
                    sensor_tokens,
                    key_padding_mask=~batch.obs_valid_mask,
                    need_weights=False,
                )[0]
            )
        enriched_sensors = (
            sensor_tokens
            + self.sensor_readback(sensor_tokens, latents, latents, need_weights=False)[0]
        )

        time_feature = time[:, None, None].expand(-1, state.shape[1], 1)
        point_features = self.point_projection(
            torch.cat((self.position(batch.query_coords), state, time_feature), dim=-1)
        )
        query_tokens = self.query_projection(self.position(batch.query_coords))
        query_global = self._global_readout(query_tokens, latents)
        local = self._local_features(batch, enriched_sensors)
        global_summary = latents.mean(1, keepdim=True).expand(-1, state.shape[1], -1)
        fused = self.head_norm(
            torch.cat((point_features, query_global, local, global_summary), dim=-1)
        )
        return self.head(fused)
