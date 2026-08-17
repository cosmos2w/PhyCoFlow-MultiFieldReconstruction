"""Differentiable rectified-flow rollout shared by post-training sources."""

from __future__ import annotations

import torch

from ..coherence.observation import (
    MODES,
    clamp_observations,
    guide_endpoint_velocity,
    pointwise_maps,
    smooth_maps,
)
from ..contracts import ObservationBatch


def subset_query_batch(
    batch: ObservationBatch,
    point_count: int,
    *,
    generator: torch.Generator,
    indices: torch.Tensor | None = None,
    shared: bool = False,
) -> ObservationBatch:
    """Select identical query/target entries while retaining original point IDs."""
    if point_count >= batch.query_coords.shape[1]:
        return batch
    if point_count < 2:
        raise ValueError("coherence query subsets require at least two points")
    batch_size, query_count, coordinate_dim = batch.query_coords.shape
    if indices is not None:
        selected_once = torch.as_tensor(
            indices, device=batch.query_coords.device, dtype=torch.long
        )
        if selected_once.ndim != 1 or selected_once.numel() != point_count:
            raise ValueError("explicit query indices must be one-dimensional and match point_count")
        if torch.any(selected_once < 0) or torch.any(selected_once >= query_count):
            raise ValueError("explicit query indices contain an out-of-range position")
        selected = selected_once[None, :].expand(batch_size, -1)
    elif shared:
        selected_once = torch.randperm(
            query_count, device=batch.query_coords.device, generator=generator
        )[:point_count].sort().values
        selected = selected_once[None, :].expand(batch_size, -1)
    else:
        selected = torch.stack(
            [
                torch.randperm(query_count, device=batch.query_coords.device, generator=generator)[
                    :point_count
                ]
                .sort()
                .values
                for _ in range(batch_size)
            ]
        )
    coordinate_indices = selected.unsqueeze(-1).expand(-1, -1, coordinate_dim)
    query_coords = torch.gather(batch.query_coords, 1, coordinate_indices)
    query_mask = torch.gather(batch.query_valid_mask, 1, selected)
    target = None
    if batch.target_fields is not None:
        field_indices = selected.unsqueeze(-1).expand(-1, -1, batch.target_fields.shape[-1])
        target = torch.gather(batch.target_fields, 1, field_indices)
    metadata = dict(batch.metadata)
    original_query_indices = metadata.get("query_indices")
    if isinstance(original_query_indices, torch.Tensor):
        metadata["query_indices"] = torch.gather(
            original_query_indices.to(selected.device), 1, selected
        )
    context = metadata.get("sample_context")
    if isinstance(context, dict):
        context = dict(context)
        for group_name in ("physics", "auxiliary"):
            group = context.get(group_name)
            if not isinstance(group, dict):
                continue
            group = dict(group)
            for name, value in group.items():
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim >= 2
                    and value.shape[:2] == batch.query_coords.shape[:2]
                ):
                    gather_shape = (selected.shape[0], selected.shape[1], *value.shape[2:])
                    indices = selected.reshape(*selected.shape, *((1,) * (value.ndim - 2))).expand(
                        gather_shape
                    )
                    group[name] = torch.gather(value, 1, indices)
            context[group_name] = group
        metadata["sample_context"] = context
    return ObservationBatch(
        obs_coords=batch.obs_coords,
        obs_values=batch.obs_values,
        obs_field_ids=batch.obs_field_ids,
        obs_valid_mask=batch.obs_valid_mask,
        query_coords=query_coords,
        query_valid_mask=query_mask,
        target_fields=target,
        sample_ids=batch.sample_ids,
        obs_indices=batch.obs_indices,
        logical_shapes=batch.logical_shapes,
        metadata=metadata,
    )


def differentiable_rf_rollout(
    model,
    batch: ObservationBatch,
    *,
    steps: int,
    solver: str,
    generator: torch.Generator,
    observation_config: dict,
) -> torch.Tensor:
    """Integrate a clean endpoint without entering a no-grad sampling path."""
    if not getattr(model.capabilities, "differentiable_rollout", False):
        raise ValueError("post-training source does not expose a differentiable rollout")
    if steps < 1 or solver not in {"euler", "heun"}:
        raise ValueError("rollout requires steps>=1 and solver=euler or heun")
    mode = str(observation_config.get("mode", "endpoint_smooth"))
    if mode not in MODES:
        raise ValueError(f"unknown observation consistency mode {mode!r}")
    state = model.sample_source(batch, generator=generator)
    num_fields = state.shape[-1]
    value_map = mask_map = None
    if mode == "endpoint":
        value_map, mask_map = pointwise_maps(batch, num_fields)
    elif mode == "endpoint_smooth":
        value_map, mask_map = smooth_maps(
            batch,
            num_fields,
            sigma=float(observation_config.get("sigma", 0.05)),
            chunk_size=int(observation_config.get("chunk_size", 4096)),
        )

    times = torch.linspace(0.0, 1.0, steps + 1, device=state.device, dtype=state.dtype)
    for step in range(steps):
        time0 = times[step].expand(state.shape[0])
        delta = times[step + 1] - times[step]
        velocity0 = model.velocity(batch, state, time0)
        if mode in {"endpoint", "endpoint_smooth"}:
            velocity0 = guide_endpoint_velocity(
                state,
                velocity0,
                time0,
                value_map,
                mask_map,
                strength=float(observation_config.get("strength", 1.0)),
                schedule_power=float(observation_config.get("schedule_power", 2.0)),
            )
        if solver == "heun":
            euler_state = state + delta * velocity0
            time1 = times[step + 1].expand(state.shape[0])
            velocity1 = model.velocity(batch, euler_state, time1)
            if mode in {"endpoint", "endpoint_smooth"} and step + 1 < steps:
                velocity1 = guide_endpoint_velocity(
                    euler_state,
                    velocity1,
                    time1,
                    value_map,
                    mask_map,
                    strength=float(observation_config.get("strength", 1.0)),
                    schedule_power=float(observation_config.get("schedule_power", 2.0)),
                )
            state = state + 0.5 * delta * (velocity0 + velocity1)
        else:
            state = state + delta * velocity0
        if mode == "hard":
            state = clamp_observations(state, batch)

    if bool(observation_config.get("final_clamp", True)) and mode != "none":
        state = clamp_observations(state, batch)
    return state


def _endpoint_consistency(
    prediction: torch.Tensor,
    batch: ObservationBatch,
    observation_config: dict,
) -> torch.Tensor:
    """Apply the shared endpoint constraint to non-RF differentiable inference."""
    mode = str(observation_config.get("mode", "endpoint_smooth"))
    if mode not in MODES:
        raise ValueError(f"unknown observation consistency mode {mode!r}")
    if mode == "none":
        return prediction
    if mode == "endpoint_smooth":
        values, mask = smooth_maps(
            batch,
            prediction.shape[-1],
            sigma=float(observation_config.get("sigma", 0.05)),
            chunk_size=int(observation_config.get("chunk_size", 4096)),
        )
        blend = (float(observation_config.get("strength", 1.0)) * mask).clamp(0.0, 1.0)
        prediction = prediction * (1.0 - blend) + values * blend
    else:
        prediction = clamp_observations(
            prediction,
            batch,
            strength=float(observation_config.get("strength", 1.0)),
        )
    if bool(observation_config.get("final_clamp", True)):
        prediction = clamp_observations(prediction, batch)
    return prediction


def differentiable_reconstruction(
    model,
    batch: ObservationBatch,
    *,
    steps: int,
    solver: str,
    generator: torch.Generator,
    observation_config: dict,
) -> torch.Tensor:
    """Dispatch one common coherence path without inspecting model names."""
    if hasattr(model, "sample_source") and hasattr(model, "velocity"):
        return differentiable_rf_rollout(
            model,
            batch,
            steps=steps,
            solver=solver,
            generator=generator,
            observation_config=observation_config,
        )
    native = getattr(model, "differentiable_reconstruct", None)
    if native is None:
        raise ValueError(
            "post-training source declares differentiable rollout but exposes neither "
            "flow hooks nor differentiable_reconstruct"
        )
    prediction = native(batch, steps=steps, generator=generator)
    return _endpoint_consistency(prediction, batch, observation_config)
