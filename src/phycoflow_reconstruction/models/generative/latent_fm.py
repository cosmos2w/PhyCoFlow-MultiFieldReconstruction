"""Two-stage convolutional autoencoder and latent rectified-flow model.

Stage 1 learns reconstruction. Stage 2 explicitly loads and freezes the Stage-1
autoencoder, then learns conditional latent flow; checkpoint lineage records
that prerequisite source. The compact implementation supports 2-D physical or
`(time,x)` grids.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ...contracts import LossBundle, ModelCapabilities, ObservationBatch, ReconstructionBatch
from ...data.observations import rasterize_observations, reshape_full_target


class _Autoencoder(nn.Module):
    def __init__(self, channels: int, latent_channels: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, latent_channels, 4, stride=2, padding=1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, channels, 3, padding=1),
        )


class _LatentVelocity(nn.Module):
    def __init__(self, latent_channels: int, condition_channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(latent_channels + condition_channels + 1, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, latent_channels, 3, padding=1),
        )

    def forward(
        self, state: torch.Tensor, condition: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        time_map = time[:, None, None, None].expand(-1, 1, state.shape[-2], state.shape[-1])
        return self.network(torch.cat((state, condition, time_map), 1))


class LatentFlowModel(nn.Module):
    capabilities = ModelCapabilities(
        "grid",
        True,
        True,
        True,
        True,
        ("base_training", "latent_stage1", "latent_stage2", "post_training"),
    )

    def __init__(
        self,
        num_fields: int,
        logical_shape: tuple[int, ...],
        latent_channels: int = 16,
        stage: int = 1,
        stage1_checkpoint: str | None = None,
    ) -> None:
        super().__init__()
        if len(logical_shape) != 2:
            raise ValueError("LatentFlowModel currently requires a 2-D logical grid")
        self.num_fields = num_fields
        self.logical_shape = logical_shape
        self.stage = int(stage)
        if self.stage not in {1, 2}:
            raise ValueError("latent flow stage must be 1 or 2")
        if self.stage == 1:
            self.capabilities = ModelCapabilities(
                "grid",
                True,
                False,
                True,
                False,
                ("base_training", "latent_stage1"),
            )
        self.autoencoder = _Autoencoder(num_fields, latent_channels)
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(2 * num_fields, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 2 * num_fields, 4, stride=2, padding=1),
            nn.GELU(),
        )
        self.velocity = _LatentVelocity(latent_channels, 2 * num_fields)
        self.stage1_checkpoint = stage1_checkpoint
        if self.stage == 2:
            if stage1_checkpoint is None:
                raise ValueError("latent_fm stage 2 requires model.stage1_checkpoint")
            self._load_stage1_autoencoder(stage1_checkpoint)
            for parameter in self.autoencoder.parameters():
                parameter.requires_grad_(False)

    def _load_stage1_autoencoder(self, checkpoint_path: str | Path) -> None:
        """Load only the Stage-1 autoencoder and reject incompatible lineage."""
        path = Path(checkpoint_path).expanduser().resolve()
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch before weights_only support
            payload = torch.load(path, map_location="cpu")
        if (
            payload.get("model_name") != "latent_fm"
            or int(payload.get("model_config", {}).get("stage", 0)) != 1
        ):
            raise ValueError(f"{path} is not a latent_fm Stage-1 checkpoint")
        prefix = "autoencoder."
        source = {
            key.removeprefix(prefix): value
            for key, value in payload["model"].items()
            if key.startswith(prefix)
        }
        result = self.autoencoder.load_state_dict(source, strict=True)
        if result.missing_keys or result.unexpected_keys:  # defensive; strict=True already raises
            raise RuntimeError(f"Stage-1 autoencoder mismatch: {result}")

    def training_loss(self, batch: ObservationBatch) -> LossBundle:
        target = reshape_full_target(batch)
        latent = self.autoencoder.encoder(target)
        reconstruction = self.autoencoder.decoder(latent)
        reconstruction = reconstruction[..., : target.shape[-2], : target.shape[-1]]
        reconstruction_loss = F.mse_loss(reconstruction, target)
        if self.stage == 1:
            return LossBundle(reconstruction_loss, {"autoencoder_mse": reconstruction_loss})

        values, mask = rasterize_observations(batch, self.num_fields)
        condition = self.condition_encoder(torch.cat((values, mask), 1))
        noise = torch.randn_like(latent)
        time = torch.rand(latent.shape[0], device=latent.device)
        state = (1 - time[:, None, None, None]) * noise + time[
            :, None, None, None
        ] * latent.detach()
        velocity = self.velocity(state, condition, time)
        flow_loss = F.mse_loss(velocity, latent.detach() - noise)
        return LossBundle(
            flow_loss,
            {"latent_flow_mse": flow_loss, "autoencoder_mse": reconstruction_loss.detach()},
        )

    def differentiable_reconstruct(
        self,
        batch: ObservationBatch,
        *,
        steps: int = 8,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if self.stage == 1:
            raise ValueError(
                "latent_fm stage 1 is an autoencoder prerequisite, not a sparse-reconstruction source"
            )
        if steps < 1:
            raise ValueError("reconstruction steps must be at least one")
        values, mask = rasterize_observations(batch, self.num_fields)
        condition = self.condition_encoder(torch.cat((values, mask), 1))
        latent_shape = condition.shape[-2:]
        latent_channels = self.autoencoder.encoder(values).shape[1]
        state_latent = torch.randn(
            values.shape[0],
            latent_channels,
            *latent_shape,
            device=values.device,
            dtype=values.dtype,
            generator=generator,
        )
        for step in range(steps):
            time = torch.full(
                (values.shape[0],),
                step / steps,
                device=values.device,
                dtype=values.dtype,
            )
            state_latent = state_latent + self.velocity(state_latent, condition, time) / steps
        state = self.autoencoder.decoder(state_latent)
        state = state[..., : self.logical_shape[-2], : self.logical_shape[-1]]
        state = state * (1 - mask) + values * mask
        point_count = math.prod(self.logical_shape)
        return state.reshape(state.shape[0], self.num_fields, point_count).transpose(1, 2)

    @torch.no_grad()
    def reconstruct(
        self,
        batch: ObservationBatch,
        *,
        steps: int = 8,
        generator: torch.Generator | None = None,
        **_: Any,
    ) -> ReconstructionBatch:
        if self.stage == 1:
            values, mask = rasterize_observations(batch, self.num_fields)
            state = self.autoencoder.decoder(self.autoencoder.encoder(values))
            state = state[..., : self.logical_shape[-2], : self.logical_shape[-1]]
            state = state * (1 - mask) + values * mask
            point_count = math.prod(self.logical_shape)
            prediction = state.reshape(state.shape[0], self.num_fields, point_count).transpose(1, 2)
        else:
            prediction = self.differentiable_reconstruct(batch, steps=steps, generator=generator)
        return ReconstructionBatch(
            prediction, diagnostics={"stage": self.stage, "sampling_steps": steps}
        )
