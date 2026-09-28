"""Expose a transformed-coordinate CQ checkpoint through physical field contracts.

The learned RF, its prior, and its data loss stay in the checkpoint's original
coordinates. Only observations/targets entering the model and endpoints leaving
it are transformed. In particular, Euler integration never runs in sinh space.
"""

from collections.abc import Mapping
from dataclasses import replace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ...contracts import ReconstructionBatch
from ..flows.pointcloud.adapters.gl_rbf_cq_adapter import GLRbfCQ


class PhysicalGLRbfCQ(nn.Module):
    capabilities = GLRbfCQ.capabilities

    def __init__(
        self,
        *,
        physical_field_transform: Mapping,
        rollout_checkpointing: bool = False,
        rollout_context_cache: str = "none",
        data_query_points: int | None = None,
        **config,
    ):
        super().__init__()
        self.core = GLRbfCQ(**config)
        transform = dict(physical_field_transform)
        if set(transform) != {"offset", "scale", "asinh_scale", "linear_mask"}:
            raise ValueError(
                "physical_field_transform requires offset, scale, asinh_scale, linear_mask"
            )
        for name in ("offset", "scale", "asinh_scale", "linear_mask"):
            value = torch.as_tensor(
                transform[name], dtype=(torch.bool if name == "linear_mask" else torch.float32)
            )
            if value.shape != (self.core.num_fields,) or not torch.isfinite(value).all():
                raise ValueError(f"physical_field_transform.{name} must align with finite fields")
            if name in {"scale", "asinh_scale"} and not (value > 0).all():
                raise ValueError(f"physical_field_transform.{name} must be positive")
            # Config owns these constants; preserve the imported neural state keys exactly.
            self.register_buffer(name, value, persistent=False)
        self.rollout_checkpointing = bool(rollout_checkpointing)
        if rollout_context_cache not in {"none", "condition", "geometry", "static_features"}:
            raise ValueError("invalid rollout_context_cache")
        self.rollout_context_cache = rollout_context_cache
        self.data_query_points = None if data_query_points is None else int(data_query_points)
        if self.data_query_points is not None and self.data_query_points < 1:
            raise ValueError("data_query_points must be positive")
        if self.core.obs_consistency_mode != "none" or self.core.obs_consistency_final_clamp:
            raise ValueError(
                "physical CQ adapter requires observation consistency none without clamping"
            )

    def encode(self, values):
        transformed = torch.where(
            self.linear_mask, values, torch.asinh(values / self.asinh_scale.to(values))
        )
        return (transformed - self.offset.to(values)) / self.scale.to(values)

    def decode(self, values):
        transformed = values * self.scale.to(values) + self.offset.to(values)
        return torch.where(
            self.linear_mask, transformed, self.asinh_scale.to(values) * torch.sinh(transformed)
        )

    def _encoded_batch(self, batch):
        ids = batch.obs_field_ids
        values = batch.obs_values[..., 0]
        transformed = torch.where(
            self.linear_mask[ids], values, torch.asinh(values / self.asinh_scale[ids].to(values))
        )
        observations = (transformed - self.offset[ids].to(values)) / self.scale[ids].to(values)
        return replace(
            batch,
            obs_values=observations.unsqueeze(-1),
            target_fields=(
                None if batch.target_fields is None else self.encode(batch.target_fields)
            ),
        )

    def _data_batch(self, batch):
        """Keep sparse RF supervision independent of the full coherence grid."""
        count = self.data_query_points
        if count is None or count >= batch.query_coords.shape[1]:
            return batch
        indices = (
            torch.randperm(batch.query_coords.shape[1], device=batch.query_coords.device)[:count]
            .sort()
            .values
        )
        metadata = dict(batch.metadata)
        if isinstance(metadata.get("query_indices"), torch.Tensor):
            metadata["query_indices"] = metadata["query_indices"][:, indices]
        return replace(
            batch,
            query_coords=batch.query_coords[:, indices],
            query_valid_mask=batch.query_valid_mask[:, indices],
            target_fields=None if batch.target_fields is None else batch.target_fields[:, indices],
            metadata=metadata,
        )

    def training_loss(self, batch):
        return self.core.training_loss(self._encoded_batch(self._data_batch(batch)))

    def training_backward(self, batch, **kwargs):
        return self.core.training_backward(self._encoded_batch(self._data_batch(batch)), **kwargs)

    def differentiable_reconstruct(self, batch, *, steps, generator=None):
        if int(steps) < 1:
            raise ValueError("rollout requires positive steps")
        encoded = self._encoded_batch(replace(batch, target_fields=None))
        state = self.core.sample_source(encoded, generator=generator)
        times = torch.linspace(0, 1, steps + 1, device=state.device, dtype=state.dtype)

        # These graphs belong to this rollout only. Rebuild after every update,
        # and independently for the frozen source and the trainable model.
        condition = query = None
        if self.rollout_context_cache != "none":
            if self.training:
                raise ValueError("shared rollout context requires eval mode (dropout disabled)")
            backbone = self.core.model
            condition = backbone.prepare_condition_context(
                encoded.obs_coords,
                encoded.obs_values,
                encoded.obs_valid_mask,
                encoded.obs_field_ids,
            )
            if self.rollout_context_cache != "condition":
                query = backbone.prepare_rollout_query_context(
                    encoded.query_coords, condition, cache_level=self.rollout_context_cache
                )

        def velocity(value, time):
            if condition is not None:
                return self.core.model.forward_query_chunk(
                    time, value, encoded.query_coords, condition, query
                )
            return self.core.velocity(encoded, value, time)

        def evaluate(value, time):
            if self.rollout_checkpointing and torch.is_grad_enabled():
                return checkpoint(velocity, value, time, use_reentrant=False)
            return velocity(value, time)

        for index in range(steps):
            time = times[index].expand(state.shape[0])
            delta = times[index + 1] - times[index]
            first = evaluate(state, time)
            if self.core.ode_solver == "heun":
                second = evaluate(state + delta * first, times[index + 1].expand_as(time))
                state = state + 0.5 * delta * (first + second)
            else:
                state = state + delta * first
        return self.decode(state)

    @torch.no_grad()
    def reconstruct(self, batch, *, steps=16, generator=None, **kwargs):
        if kwargs.get("obs_consistency_mode", "none") != "none" or kwargs.get(
            "obs_consistency_final_clamp", False
        ):
            raise ValueError("physical block means cannot be used as pointwise clamps")
        output = self.core.reconstruct(
            self._encoded_batch(batch), steps=steps, generator=generator, **kwargs
        )
        return ReconstructionBatch(self.decode(output.prediction), diagnostics=output.diagnostics)

    def state_dict(self, *args, **kwargs):
        return self.core.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        return self.core.load_state_dict(state_dict, strict=strict, **kwargs)

    def after_optimizer_step(self):
        self.core.after_optimizer_step()

    def evaluation_weight_context(self):
        return self.core.evaluation_weight_context()

    def training_aux_state_dict(self):
        return self.core.training_aux_state_dict()

    def load_training_aux_state_dict(self, state):
        self.core.load_training_aux_state_dict(state)
