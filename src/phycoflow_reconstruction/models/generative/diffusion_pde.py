"""Clean local DDPM/DDIM implementation for partial-observation PDE fields.

This is an independently organized conditional denoising adapter; it does not
copy the archived DiffusionPDE repository. Sparse value/mask rasters condition
a compact convolutional noise predictor. Training uses a cosine cumulative
noise schedule; reconstruction uses deterministic DDIM steps and hard sensor
conditioning. Phase-4 checks use very few sampling steps.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ...contracts import LossBundle, ModelCapabilities, ObservationBatch, ReconstructionBatch
from ...data.observations import rasterize_observations, reshape_full_target


class _Denoiser(nn.Module):
    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels * 3 + 1, hidden, 3, padding=1),
            nn.GroupNorm(4, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(4, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 3, padding=1),
        )

    def forward(
        self, noisy: torch.Tensor, values: torch.Tensor, mask: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        time_map = time[:, None, None, None].expand(-1, 1, noisy.shape[-2], noisy.shape[-1])
        return self.network(torch.cat((noisy, values, mask, time_map), dim=1))


def _cosine_alpha_bar(timesteps: int, offset: float = 0.008) -> torch.Tensor:
    positions = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    values = torch.cos(((positions / timesteps + offset) / (1 + offset)) * math.pi / 2).square()
    values = values / values[0]
    return values[1:].float().clamp_min(1e-5)


class DiffusionPDEModel(nn.Module):
    capabilities = ModelCapabilities(
        "grid", True, True, True, True, ("base_training", "post_training")
    )

    def __init__(
        self,
        num_fields: int,
        logical_shape: tuple[int, ...],
        hidden_channels: int = 32,
        training_timesteps: int = 1000,
    ) -> None:
        super().__init__()
        if len(logical_shape) != 2:
            raise ValueError("DiffusionPDEModel currently requires a 2-D logical grid")
        if hidden_channels < 4 or hidden_channels % 4:
            raise ValueError("hidden_channels must be a positive multiple of four")
        self.num_fields = num_fields
        self.logical_shape = logical_shape
        if training_timesteps < 2:
            raise ValueError("training_timesteps must be at least two")
        self.training_timesteps = int(training_timesteps)
        # This schedule is derived exactly from training_timesteps and is not
        # learned checkpoint state. Keeping it non-persistent preserves strict
        # compatibility with Phase-4 checkpoints while still moving it with the model.
        self.register_buffer(
            "alpha_bar",
            _cosine_alpha_bar(self.training_timesteps),
            persistent=False,
        )
        self.denoiser = _Denoiser(num_fields, hidden_channels)

    def training_loss(self, batch: ObservationBatch) -> LossBundle:
        target = reshape_full_target(batch)
        values, mask = rasterize_observations(batch, self.num_fields)
        timestep = torch.randint(
            0, self.training_timesteps, (target.shape[0],), device=target.device
        )
        alpha = self.alpha_bar[timestep].to(target.dtype)
        time = timestep.to(target.dtype) / max(self.training_timesteps - 1, 1)
        noise = torch.randn_like(target)
        noisy = (
            alpha[:, None, None, None].sqrt() * target
            + (1 - alpha[:, None, None, None]).sqrt() * noise
        )
        predicted_noise = self.denoiser(noisy, values, mask, time)
        loss = F.mse_loss(predicted_noise, noise)
        return LossBundle(loss, {"diffusion_noise_mse": loss})

    def differentiable_reconstruct(
        self,
        batch: ObservationBatch,
        *,
        steps: int = 8,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Run deterministic DDIM sampling without disabling parameter gradients."""
        if steps < 1:
            raise ValueError("reconstruction steps must be at least one")
        values, mask = rasterize_observations(batch, self.num_fields)
        state = torch.randn(
            values.shape,
            device=values.device,
            dtype=values.dtype,
            generator=generator,
        )
        schedule = (
            torch.linspace(self.training_timesteps - 1, 0, steps, device=state.device)
            .round()
            .long()
            .unique_consecutive()
        )
        for index, timestep in enumerate(schedule):
            time = torch.full(
                (state.shape[0],),
                float(timestep) / max(self.training_timesteps - 1, 1),
                device=state.device,
            )
            predicted_noise = self.denoiser(state, values, mask, time)
            alpha = self.alpha_bar[timestep].to(state.dtype)
            clean = (state - (1 - alpha).sqrt() * predicted_noise) / alpha.sqrt().clamp_min(1e-5)
            if index + 1 < schedule.numel():
                alpha_previous = self.alpha_bar[schedule[index + 1]].to(state.dtype)
                state = (
                    alpha_previous.sqrt() * clean + (1 - alpha_previous).sqrt() * predicted_noise
                )
            else:
                state = clean
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
        prediction = self.differentiable_reconstruct(batch, steps=steps, generator=generator)
        return ReconstructionBatch(prediction, diagnostics={"sampling_steps": steps})
