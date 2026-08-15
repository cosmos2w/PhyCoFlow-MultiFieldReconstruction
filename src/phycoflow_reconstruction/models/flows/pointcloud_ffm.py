"""Narrow rectified-flow reconstruction wrapper with GL-RBF-ENH or FNO.

The data path follows `x_t=(1-t)z+t*x` with target velocity `x-z`. The default
backbone is the sole new point mode `GL_rbf_ENH/topk_rbf`; FNO is optional for
complete regular-grid queries.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ...contracts import LossBundle, ModelCapabilities, ObservationBatch, ReconstructionBatch
from ..base import masked_mse
from .fno_backbone import FNOFlowBackbone
from .gl_rbf_enh_topk import EnhancedGLRBFTopK
from .priors import IIDGaussianPrior, RFFGaussianPrior


class PointCloudFFM(nn.Module):
    capabilities = ModelCapabilities(
        "point", True, True, False, True, ("base_training", "post_training")
    )

    def __init__(
        self,
        coordinate_dim: int,
        num_fields: int,
        logical_shape: tuple[int, ...],
        *,
        backbone: str = "gl_rbf_enh",
        prior: str = "rff",
        hidden_dim: int = 128,
        latent_dim: int = 128,
        num_latents: int = 32,
        heads: int = 4,
        latent_blocks: int = 2,
        gather_topk: int = 16,
        rbf_sigma: float = 0.08,
        fno_hidden_channels: int = 32,
        query_chunk_size: int = 2048,
    ) -> None:
        super().__init__()
        self.num_fields = num_fields
        self.backbone_name = backbone
        if backbone == "gl_rbf_enh":
            self.velocity_model = EnhancedGLRBFTopK(
                coordinate_dim,
                num_fields,
                hidden_dim,
                latent_dim,
                num_latents,
                heads,
                latent_blocks,
                gather_topk,
                rbf_sigma,
                query_chunk_size=query_chunk_size,
            )
        elif backbone == "fno":
            self.velocity_model = FNOFlowBackbone(num_fields, logical_shape, fno_hidden_channels)
            self.capabilities = ModelCapabilities(
                "grid", True, True, True, True, ("base_training", "post_training")
            )
        else:
            raise ValueError("PointCloudFFM supports only gl_rbf_enh or fno")

        if prior == "rff":
            self.prior = RFFGaussianPrior(coordinate_dim)
        elif prior == "iid":
            self.prior = IIDGaussianPrior()
        else:
            raise ValueError("prior must be rff or iid")

    def _velocity(
        self, batch: ObservationBatch, state: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        return self.velocity_model(batch, state, time)

    def sample_source(
        self,
        batch: ObservationBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Public differentiable-rollout hook used by the common post-trainer."""
        return self.prior.sample(batch.query_coords, self.num_fields, generator=generator)

    def velocity(
        self,
        batch: ObservationBatch,
        state: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the learned RF velocity without detaching autograd."""
        return self._velocity(batch, state, time)

    def training_loss(self, batch: ObservationBatch) -> LossBundle:
        if batch.target_fields is None:
            raise ValueError("rectified-flow training requires target_fields")
        target = batch.target_fields
        source = self.sample_source(batch)
        time = torch.rand(target.shape[0], device=target.device)
        state = (1 - time[:, None, None]) * source + time[:, None, None] * target
        predicted_velocity = self._velocity(batch, state, time)
        loss = masked_mse(predicted_velocity, target - source, batch.query_valid_mask)
        return LossBundle(loss, {"rectified_flow_mse": loss})

    def reconstruct(
        self,
        batch: ObservationBatch,
        *,
        steps: int = 4,
        generator: torch.Generator | None = None,
        **_: Any,
    ) -> ReconstructionBatch:
        if steps < 1:
            raise ValueError("reconstruction steps must be at least one")
        state = self.prior.sample(batch.query_coords, self.num_fields, generator=generator)
        step_size = 1.0 / steps
        for step in range(steps):
            time = torch.full(
                (state.shape[0],), step / steps, device=state.device, dtype=state.dtype
            )
            state = state + step_size * self.velocity(batch, state, time)

        # Exact sensor clamping is valid only when complete query indices are present.
        query_indices = batch.metadata.get("query_indices")
        if batch.obs_indices is not None and isinstance(query_indices, torch.Tensor):
            query_indices = query_indices.to(state.device)
            for batch_index in range(state.shape[0]):
                mapping = {
                    int(index): position
                    for position, index in enumerate(query_indices[batch_index].tolist())
                    if index >= 0
                }
                valid = batch.obs_valid_mask[batch_index]
                for point, field, value in zip(
                    batch.obs_indices[batch_index, valid].tolist(),
                    batch.obs_field_ids[batch_index, valid].tolist(),
                    batch.obs_values[batch_index, valid, 0],
                ):
                    if int(point) in mapping:
                        state[batch_index, mapping[int(point)], int(field)] = value
        return ReconstructionBatch(
            state, diagnostics={"sampling_steps": steps, "backbone": self.backbone_name}
        )
