"""Clean local DDPM/DDIM implementation for partial-observation PDE fields.

This is an independently organized conditional denoising adapter; it does not
copy the archived DiffusionPDE repository. Sparse value/mask rasters condition
either a legacy compact CNN or a multiscale U-Net noise predictor. Training
uses a cosine cumulative noise schedule; reconstruction uses deterministic
DDIM steps and hard sensor conditioning.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ...contracts import LossBundle, ModelCapabilities, ObservationBatch, ReconstructionBatch
from ...data.observations import rasterize_observations, reshape_full_target


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class _PlainCNNDenoiser(nn.Module):
    """Legacy three-convolution denoiser retained for checkpoint compatibility."""

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


class _SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequency = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=time.device, dtype=time.dtype)
            / max(half - 1, 1)
        )
        angles = time[:, None] * frequency[None, :] * 1_000.0
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < self.dimension:
            embedding = F.pad(embedding, (0, self.dimension - embedding.shape[-1]))
        return embedding


class _TimeResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_embed_dim, out_channels)
        self.norm2 = _group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, inputs: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = hidden + self.time_projection(F.silu(time_embedding))[:, :, None, None]
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return self.skip(inputs) + hidden


class _SpatialAttention(nn.Module):
    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.norm = _group_norm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = inputs.shape
        tokens = self.norm(inputs).reshape(batch, channels, height * width).transpose(1, 2)
        attended = self.attention(tokens, tokens, tokens, need_weights=False)[0]
        return inputs + attended.transpose(1, 2).reshape(batch, channels, height, width)


class _UNetLevel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        num_res_blocks: int,
        dropout: float,
        attention_heads: int | None,
    ) -> None:
        super().__init__()
        blocks = [_TimeResidualBlock(in_channels, out_channels, time_embed_dim, dropout)]
        blocks.extend(
            _TimeResidualBlock(out_channels, out_channels, time_embed_dim, dropout)
            for _ in range(num_res_blocks - 1)
        )
        self.blocks = nn.ModuleList(blocks)
        self.attention = (
            _SpatialAttention(out_channels, attention_heads)
            if attention_heads is not None
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for block in self.blocks:
            hidden = block(hidden, time_embedding)
        return self.attention(hidden)


class _ConditionalUNetDenoiser(nn.Module):
    """Multiscale value/mask-conditioned U-Net with learned timestep features."""

    def __init__(
        self,
        channels: int,
        base_channels: int,
        channel_multipliers: Sequence[int],
        num_res_blocks: int,
        time_embed_dim: int,
        attention_levels: Sequence[int],
        attention_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        level_channels = [base_channels * int(multiplier) for multiplier in channel_multipliers]
        attention_level_set = {int(level) for level in attention_levels}
        self.time_embedding = nn.Sequential(
            _SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.input_projection = nn.Conv2d(channels * 3, level_channels[0], 3, padding=1)

        self.down_levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_channels = level_channels[0]
        for level, out_channels in enumerate(level_channels):
            heads = attention_heads if level in attention_level_set else None
            self.down_levels.append(
                _UNetLevel(
                    current_channels,
                    out_channels,
                    time_embed_dim,
                    num_res_blocks,
                    dropout,
                    heads,
                )
            )
            current_channels = out_channels
            if level + 1 < len(level_channels):
                next_channels = level_channels[level + 1]
                self.downsamples.append(
                    nn.Conv2d(current_channels, next_channels, 4, stride=2, padding=1)
                )
                current_channels = next_channels

        deepest = level_channels[-1]
        self.middle1 = _TimeResidualBlock(deepest, deepest, time_embed_dim, dropout)
        self.middle_attention = (
            _SpatialAttention(deepest, attention_heads)
            if len(level_channels) - 1 in attention_level_set
            else nn.Identity()
        )
        self.middle2 = _TimeResidualBlock(deepest, deepest, time_embed_dim, dropout)

        self.up_levels = nn.ModuleList()
        current_channels = deepest
        for level in reversed(range(len(level_channels))):
            out_channels = level_channels[level]
            heads = attention_heads if level in attention_level_set else None
            self.up_levels.append(
                _UNetLevel(
                    current_channels + out_channels,
                    out_channels,
                    time_embed_dim,
                    num_res_blocks,
                    dropout,
                    heads,
                )
            )
            current_channels = out_channels

        self.output_norm = _group_norm(level_channels[0])
        self.output_projection = nn.Conv2d(level_channels[0], channels, 3, padding=1)

    def forward(
        self, noisy: torch.Tensor, values: torch.Tensor, mask: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        time_embedding = self.time_embedding(time)
        hidden = self.input_projection(torch.cat((noisy, values, mask), dim=1))
        skips: list[torch.Tensor] = []
        for level, down_level in enumerate(self.down_levels):
            hidden = down_level(hidden, time_embedding)
            skips.append(hidden)
            if level < len(self.downsamples):
                hidden = self.downsamples[level](hidden)

        hidden = self.middle1(hidden, time_embedding)
        hidden = self.middle_attention(hidden)
        hidden = self.middle2(hidden, time_embedding)

        for up_level, skip in zip(self.up_levels, reversed(skips)):
            if hidden.shape[-2:] != skip.shape[-2:]:
                hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
            hidden = up_level(torch.cat((hidden, skip), dim=1), time_embedding)
        return self.output_projection(F.silu(self.output_norm(hidden)))


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
        backbone: str = "plain_cnn",
        hidden_channels: int = 32,
        base_channels: int = 64,
        channel_multipliers: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_embed_dim: int = 256,
        attention_levels: Sequence[int] = (2, 3),
        attention_heads: int = 4,
        dropout: float = 0.0,
        training_timesteps: int = 1000,
    ) -> None:
        super().__init__()
        if len(logical_shape) != 2:
            raise ValueError("DiffusionPDEModel currently requires a 2-D logical grid")
        self.num_fields = num_fields
        self.logical_shape = logical_shape
        self.backbone = str(backbone).lower()
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
        if self.backbone == "plain_cnn":
            if hidden_channels < 4 or hidden_channels % 4:
                raise ValueError("hidden_channels must be a positive multiple of four")
            self.denoiser = _PlainCNNDenoiser(num_fields, hidden_channels)
        elif self.backbone == "conditional_unet":
            multipliers = tuple(int(multiplier) for multiplier in channel_multipliers)
            levels = tuple(int(level) for level in attention_levels)
            if base_channels < 4:
                raise ValueError("base_channels must be at least four")
            if not multipliers or any(multiplier < 1 for multiplier in multipliers):
                raise ValueError("channel_multipliers must contain positive integers")
            if min(logical_shape) < 2 ** (len(multipliers) - 1):
                raise ValueError("logical grid is too small for the configured U-Net levels")
            if num_res_blocks < 1:
                raise ValueError("num_res_blocks must be positive")
            if time_embed_dim < 4:
                raise ValueError("time_embed_dim must be at least four")
            if attention_heads < 1:
                raise ValueError("attention_heads must be positive")
            if not 0.0 <= dropout < 1.0:
                raise ValueError("dropout must lie in [0, 1)")
            if any(level < 0 or level >= len(multipliers) for level in levels):
                raise ValueError("attention_levels contains an invalid U-Net level")
            if any(base_channels * multipliers[level] % attention_heads for level in levels):
                raise ValueError("attention level channels must be divisible by attention_heads")
            self.denoiser = _ConditionalUNetDenoiser(
                num_fields,
                base_channels,
                multipliers,
                num_res_blocks,
                time_embed_dim,
                levels,
                attention_heads,
                dropout,
            )
        else:
            raise ValueError("diffusion_pde backbone must be plain_cnn or conditional_unet")

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
