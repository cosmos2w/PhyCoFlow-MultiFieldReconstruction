"""Observation-consistency constraints kept outside coherence-family taxonomy."""

from __future__ import annotations

import torch

from ..contracts import ObservationBatch

MODES = {"none", "hard", "endpoint", "endpoint_smooth"}


def _query_positions(batch: ObservationBatch, batch_index: int) -> dict[int, int]:
    query_indices = batch.metadata.get("query_indices")
    if not isinstance(query_indices, torch.Tensor):
        return {index: index for index in range(batch.query_coords.shape[1])}
    indices = query_indices[batch_index].tolist()
    return {int(point): position for position, point in enumerate(indices) if int(point) >= 0}


def clamp_observations(
    state: torch.Tensor, batch: ObservationBatch, strength: float = 1.0
) -> torch.Tensor:
    """Blend observations into query entries only where exact index mappings exist."""
    if batch.obs_indices is None:
        return state
    output = state.clone()
    for batch_index in range(state.shape[0]):
        positions = _query_positions(batch, batch_index)
        valid = batch.obs_valid_mask[batch_index]
        for point, field, value in zip(
            batch.obs_indices[batch_index, valid].tolist(),
            batch.obs_field_ids[batch_index, valid].tolist(),
            batch.obs_values[batch_index, valid, 0],
        ):
            position = positions.get(int(point))
            if position is not None:
                output[batch_index, position, int(field)] = (1.0 - float(strength)) * output[
                    batch_index, position, int(field)
                ] + float(strength) * value
    return output


def pointwise_maps(batch: ObservationBatch, num_fields: int) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.zeros(
        *batch.query_coords.shape[:2],
        num_fields,
        device=batch.query_coords.device,
        dtype=batch.obs_values.dtype,
    )
    mask = torch.zeros_like(values)
    if batch.obs_indices is None:
        return values, mask
    for batch_index in range(batch.query_coords.shape[0]):
        positions = _query_positions(batch, batch_index)
        valid = batch.obs_valid_mask[batch_index]
        for point, field, value in zip(
            batch.obs_indices[batch_index, valid].tolist(),
            batch.obs_field_ids[batch_index, valid].tolist(),
            batch.obs_values[batch_index, valid, 0],
        ):
            position = positions.get(int(point))
            if position is not None:
                values[batch_index, position, int(field)] = value
                mask[batch_index, position, int(field)] = 1.0
    return values, mask


def smooth_maps(
    batch: ObservationBatch,
    num_fields: int,
    *,
    sigma: float,
    chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build chunked Gaussian sensor maps without attaching map construction to autograd."""
    if sigma <= 0:
        raise ValueError("endpoint_smooth sigma must be positive")
    values = torch.zeros(
        *batch.query_coords.shape[:2],
        num_fields,
        device=batch.query_coords.device,
        dtype=batch.obs_values.dtype,
    )
    mask = torch.zeros_like(values)
    sigma_squared = float(sigma) ** 2
    with torch.no_grad():
        for batch_index in range(batch.query_coords.shape[0]):
            for field in range(num_fields):
                valid = batch.obs_valid_mask[batch_index] & (
                    batch.obs_field_ids[batch_index] == field
                )
                if not torch.any(valid):
                    continue
                sensor_coords = batch.obs_coords[batch_index, valid]
                sensor_values = batch.obs_values[batch_index, valid, 0]
                for start in range(0, batch.query_coords.shape[1], chunk_size):
                    stop = min(start + chunk_size, batch.query_coords.shape[1])
                    distances = torch.cdist(
                        batch.query_coords[batch_index, start:stop], sensor_coords
                    ).square()
                    weights = torch.exp(-distances / (2.0 * sigma_squared))
                    weight_sum = weights.sum(dim=1)
                    values[batch_index, start:stop, field] = (
                        weights @ sensor_values
                    ) / weight_sum.clamp_min(1e-12)
                    mask[batch_index, start:stop, field] = weights.max(dim=1).values
    return values, mask


def guide_endpoint_velocity(
    state: torch.Tensor,
    velocity: torch.Tensor,
    time: torch.Tensor,
    value_map: torch.Tensor,
    mask_map: torch.Tensor,
    *,
    strength: float,
    schedule_power: float,
) -> torch.Tensor:
    remaining = (1.0 - time).clamp_min(1e-5).view(-1, 1, 1)
    guide = float(strength) * remaining.pow(float(schedule_power)) * mask_map
    endpoint = state + remaining * velocity
    consistent_endpoint = endpoint * (1.0 - guide) + value_map * guide
    return (consistent_endpoint - state) / remaining
