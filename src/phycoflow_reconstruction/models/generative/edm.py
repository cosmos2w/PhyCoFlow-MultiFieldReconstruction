"""Conditional EDM for the shared grid/sparse-observation contract.

EDM preconditioning, log-normal training noise and rho/Heun integration follow
the equations used by the official DiffusionPDE (1e2bc8b9) / NVIDIA EDM.
The independently implemented conditional U-Net and multiscale value/mask
injection adapt those equations to arbitrary observed field subsets. This is
a conditional adaptation, not the paper's unconditional PDE-guided solver.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ...contracts import LossBundle, ModelCapabilities, ObservationBatch, ReconstructionBatch
from ...data.observations import rasterize_observations, reshape_full_target
from ..flows.pointcloud.runtime.ema import ModelEMA
from .diffusion_pde import _ConditionalUNetDenoiser, _TimeResidualBlock


class _ConditionedEDMUNet(_ConditionalUNetDenoiser):
    """Pad/crop odd grids; inject mask-normalized observations at every scale."""

    def __init__(self, channels: int, **kwargs: Any) -> None:
        super().__init__(channels=channels, **kwargs)
        widths = [kwargs["base_channels"] * m for m in kwargs["channel_multipliers"]]
        self.grid_multiple = 2 ** (len(widths) - 1)
        self.condition_projections = nn.ModuleList(
            nn.Conv2d(channels * 2, width, 1) for width in widths
        )
        self.upsample_projections = nn.ModuleList(
            nn.Conv2d(widths[i + 1], widths[i + 1], 3, padding=1)
            for i in reversed(range(len(widths) - 1))
        )
        for block in self.modules():
            if isinstance(block, _TimeResidualBlock):
                block.residual_scale = math.sqrt(0.5)
                nn.init.normal_(block.conv2.weight, std=1e-5)
                nn.init.zeros_(block.conv2.bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, noisy, values, mask, time):
        height, width = noisy.shape[-2:]
        pad = (0, -width % self.grid_multiple, 0, -height % self.grid_multiple)
        noisy, values, mask = (F.pad(x, pad) for x in (noisy, values, mask))
        embedding = self.time_embedding(time)
        hidden = self.input_projection(torch.cat((noisy, values, mask), dim=1))
        skips = []
        for level, block in enumerate(self.down_levels):
            size = hidden.shape[-2:]
            density = F.adaptive_avg_pool2d(mask, size)
            pooled = F.adaptive_avg_pool2d(values, size) / density.clamp_min(1e-8)
            support = (density > 0).to(values.dtype)
            context = torch.cat((pooled * support, support), dim=1)
            hidden = block(hidden + self.condition_projections[level](context), embedding)
            skips.append(hidden)
            if level < len(self.downsamples):
                hidden = self.downsamples[level](hidden)
        hidden = self.middle2(self.middle_attention(self.middle1(hidden, embedding)), embedding)
        for level, (block, skip) in enumerate(zip(self.up_levels, reversed(skips))):
            if level:
                hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
                hidden = self.upsample_projections[level - 1](hidden)
            hidden = block(torch.cat((hidden, skip), dim=1), embedding)
        output = self.output_projection(F.silu(self.output_norm(hidden)))
        return output[..., :height, :width]


class EDMDiffusionPDEModel(nn.Module):
    capabilities = ModelCapabilities(
        "grid", True, True, True, True, ("base_training", "post_training")
    )

    def __init__(
        self,
        num_fields: int,
        logical_shape: tuple[int, ...],
        backbone: str = "conditional_unet",
        base_channels: int = 64,
        channel_multipliers: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_embed_dim: int = 256,
        attention_levels: Sequence[int] = (2, 3),
        attention_heads: int = 4,
        dropout: float = 0.0,
        sigma_data: float = 1.0,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        p_mean: float = -0.5,
        p_std: float = 1.2,
        high_noise_fraction: float = 0.0,
        train_microbatch_size: int = 32,
        precision: str = "fp32",
        model_ema_enabled: bool = True,
        model_ema_decay: float = 0.999,
        model_ema_eval: bool = True,
    ) -> None:
        super().__init__()
        if len(logical_shape) != 2 or min(logical_shape) < 1:
            raise ValueError("EDM requires a non-empty 2-D logical grid")
        if backbone != "conditional_unet":
            raise ValueError("EDM requires backbone=conditional_unet")
        widths = tuple(int(m) for m in channel_multipliers)
        levels = tuple(int(i) for i in attention_levels)
        if base_channels < 4 or not widths or min(widths) < 1 or num_res_blocks < 1:
            raise ValueError("invalid EDM U-Net widths or block count")
        if time_embed_dim < 4 or attention_heads < 1:
            raise ValueError("invalid EDM time embedding or attention heads")
        if not 0 <= dropout < 1 or any(i < 0 or i >= len(widths) for i in levels):
            raise ValueError("invalid EDM dropout or attention level")
        if any(base_channels * widths[i] % attention_heads for i in levels):
            raise ValueError("EDM attention channels must be divisible by attention_heads")
        if not all(
            math.isfinite(x) for x in (sigma_data, sigma_min, sigma_max, rho, p_mean, p_std)
        ):
            raise ValueError("EDM noise settings must be finite")
        if sigma_data <= 0 or not 0 < sigma_min < sigma_max or rho <= 0 or p_std <= 0:
            raise ValueError("invalid EDM noise scales")
        if (
            not 0 <= high_noise_fraction <= 1
            or isinstance(train_microbatch_size, bool)
            or not isinstance(train_microbatch_size, int)
            or train_microbatch_size < 1
        ):
            raise ValueError("invalid EDM high_noise_fraction or train_microbatch_size")
        if precision not in {"fp32", "bf16"}:
            raise ValueError("EDM precision must be fp32 or bf16")
        if not 0 <= model_ema_decay < 1:
            raise ValueError("EDM EMA decay must be in [0, 1)")
        self.num_fields = num_fields
        self.logical_shape = tuple(logical_shape)
        self.sigma_data, self.sigma_min, self.sigma_max = sigma_data, sigma_min, sigma_max
        self.rho, self.p_mean, self.p_std = rho, p_mean, p_std
        self.high_noise_fraction = high_noise_fraction
        self.train_microbatch_size, self.precision = train_microbatch_size, precision
        self.model_ema_enabled = model_ema_enabled
        self.model_ema_decay, self._ema_eval = model_ema_decay, model_ema_eval
        # Created lazily after .to(device); EMA is auxiliary state, not a submodule.
        self._ema: ModelEMA | None = None
        self.denoiser = _ConditionedEDMUNet(
            channels=num_fields,
            base_channels=base_channels,
            channel_multipliers=widths,
            num_res_blocks=num_res_blocks,
            time_embed_dim=time_embed_dim,
            attention_levels=levels,
            attention_heads=attention_heads,
            dropout=dropout,
        )

    def _autocast(self, device: torch.device):
        if self.precision == "bf16" and device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def denoise(self, noisy, sigma, values, mask):
        """EDM x0 estimate; exact data consistency acts on x0, not noisy state."""
        sigma = torch.as_tensor(sigma, device=noisy.device, dtype=torch.float32).reshape(
            -1, 1, 1, 1
        )
        noisy = noisy.float()
        sd = self.sigma_data
        c_skip = sd**2 / (sigma.square() + sd**2)
        c_out = sigma * sd / (sigma.square() + sd**2).sqrt()
        c_in = (sigma.square() + sd**2).rsqrt()
        # Existing sinusoidal embedding multiplies its input by 1000.
        time = sigma.flatten().log() / 4 / 1000
        if time.numel() == 1:
            time = time.expand(noisy.shape[0])
        with self._autocast(noisy.device):
            residual = self.denoiser(c_in * noisy, values / sd, mask, time)
        clean = c_skip * noisy + c_out * residual.float()
        return clean * (1 - mask) + values * mask

    def _training_inputs(self, batch):
        target = reshape_full_target(batch)
        values, mask = self._conditions(batch)
        sigma = (
            torch.randn(target.shape[0], 1, 1, 1, device=target.device) * self.p_std + self.p_mean
        ).exp()
        if self.high_noise_fraction:
            high = torch.rand_like(sigma) < self.high_noise_fraction
            sigma = torch.where(high, self.sigma_max, sigma)
        return target, values, mask, sigma, torch.randn_like(target)

    def _conditions(self, batch):
        if batch.metadata.get("protocol", {}).get("name") == "structured_block_mean":
            raise ValueError("EDM currently requires point observations, not block means")
        if batch.query_coords.shape[1] != math.prod(self.logical_shape) or any(
            tuple(shape) != self.logical_shape for shape in batch.logical_shapes
        ):
            raise ValueError("EDM requires the complete configured logical query grid")
        return rasterize_observations(batch, self.num_fields)

    def _loss(self, target, values, mask, sigma, noise):
        prediction = self.denoise(target + sigma * noise, sigma, values, mask)
        weight = (sigma.square() + self.sigma_data**2) / (sigma * self.sigma_data).square()
        unknown = 1 - mask
        # Normalize per sample so observation density does not change sample weight.
        denominator = unknown.flatten(1).sum(1).clamp_min(1)
        error = ((prediction - target).square() * unknown).flatten(1).sum(1) / denominator
        weighted = (weight.flatten() * error).mean()
        return weighted, error.mean()

    def training_loss(self, batch: ObservationBatch) -> LossBundle:
        loss, clean_mse = self._loss(*self._training_inputs(batch))
        return LossBundle(loss, {"edm_weighted_mse": loss, "edm_clean_mse": clean_mse})

    def training_backward(
        self, batch, *, loss_scale=1.0, start_phase=None, end_phase=None
    ) -> LossBundle:
        """Whole-sample accumulation: one optimizer update per effective batch."""
        if loss_scale != 1:
            raise ValueError("EDM microbatch backward requires loss_scale=1")
        if start_phase:
            start_phase("forward_native_loss")
        inputs = self._training_inputs(batch)
        count = inputs[0].shape[0]
        total = torch.zeros((), device=inputs[0].device)
        clean = torch.zeros_like(total)
        if end_phase:
            end_phase("forward_native_loss")
        for start in range(0, count, self.train_microbatch_size):
            stop = min(count, start + self.train_microbatch_size)
            fraction = (stop - start) / count
            if start_phase:
                start_phase("forward_native_loss")
            loss, mse = self._loss(*(x[start:stop] for x in inputs))
            if end_phase:
                end_phase("forward_native_loss")
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite EDM microbatch loss")
            if start_phase:
                start_phase("backward")
            (loss * fraction).backward()
            if end_phase:
                end_phase("backward")
            total += loss.detach() * fraction
            clean += mse.detach() * fraction
        return LossBundle(total, {"edm_weighted_mse": total, "edm_clean_mse": clean})

    def differentiable_reconstruct(self, batch, *, steps=32, generator=None):
        if steps < 2:
            raise ValueError("EDM Heun sampling requires at least two steps")
        values, mask = self._conditions(batch)
        indices = torch.linspace(0, 1, steps, device=values.device, dtype=torch.float64)
        sigmas = (
            self.sigma_max ** (1 / self.rho)
            + indices * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho
        sigmas = F.pad(sigmas.float(), (0, 1))  # final sigma = 0
        noise = torch.randn(values.shape, device=values.device, generator=generator)
        # Observed entries follow q(x_sigma | known clean observation).
        state = sigmas[0] * noise + mask * values
        for index in range(steps):
            sigma, next_sigma = sigmas[index], sigmas[index + 1]
            clean = self.denoise(state, sigma, values, mask)
            direction = (state - clean) / sigma
            proposed = state + (next_sigma - sigma) * direction
            if index < steps - 1:
                next_clean = self.denoise(proposed, next_sigma, values, mask)
                next_direction = (proposed - next_clean) / next_sigma
                proposed = state + (next_sigma - sigma) * (direction + next_direction) / 2
            state = proposed
        state = state * (1 - mask) + values * mask
        return state.flatten(2).transpose(1, 2)

    @torch.no_grad()
    def reconstruct(self, batch, *, steps=32, generator=None, **_: Any):
        prediction = self.differentiable_reconstruct(batch, steps=steps, generator=generator)
        return ReconstructionBatch(
            prediction,
            diagnostics={
                "sampling_steps": steps,
                "denoiser_calls": 2 * steps - 1,
                "sampler": "edm_heun",
            },
        )

    def after_optimizer_step(self) -> None:
        if not self.model_ema_enabled:
            return
        if self._ema is None:
            self._ema = ModelEMA(self, decay=self.model_ema_decay)
            self._ema.num_updates = 1
        else:
            # Short-run warmup avoids evaluating an almost-untrained EMA.
            self._ema.decay = min(
                self.model_ema_decay, (1 + self._ema.num_updates) / (10 + self._ema.num_updates)
            )
            self._ema.update(self)

    @contextmanager
    def evaluation_weight_context(self) -> Iterator[None]:
        if self._ema is None or not self._ema_eval:
            yield
        else:
            with self._ema.average_parameters(self):
                yield

    def training_aux_state_dict(self) -> dict[str, Any]:
        return {"model_ema": None if self._ema is None else self._ema.state_dict()}

    def load_training_aux_state_dict(self, state: Mapping[str, Any]) -> None:
        payload = state["model_ema"]
        if payload is None:
            self._ema = None
            return
        if not self.model_ema_enabled:
            raise ValueError("EDM checkpoint has EMA but model_ema_enabled=false")
        self._ema = ModelEMA(self, decay=self.model_ema_decay)
        self._ema.load_state_dict(payload)
