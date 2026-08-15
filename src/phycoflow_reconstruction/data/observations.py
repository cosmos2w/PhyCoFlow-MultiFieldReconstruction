"""Observation rasterization for grid adapters; hidden target values are never used."""

from __future__ import annotations

import math

import torch

from ..contracts import ObservationBatch


def rasterize_observations(
    batch: ObservationBatch, num_fields: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch.obs_indices is None:
        raise ValueError("grid rasterization requires observation point indices")
    if not batch.logical_shapes or len(set(batch.logical_shapes)) != 1:
        raise ValueError("grid batches require one shared logical shape")
    logical_shape = batch.logical_shapes[0]
    point_count = math.prod(logical_shape)
    value_map = torch.zeros(
        batch.obs_coords.shape[0], num_fields, point_count, device=batch.obs_coords.device
    )
    mask_map = torch.zeros_like(value_map)
    for batch_index in range(batch.obs_coords.shape[0]):
        valid = batch.obs_valid_mask[batch_index]
        points = batch.obs_indices[batch_index, valid]
        fields = batch.obs_field_ids[batch_index, valid]
        values = batch.obs_values[batch_index, valid, 0]
        if torch.any(points < 0) or torch.any(points >= point_count):
            raise ValueError("observation index falls outside the logical grid")
        value_map[batch_index, fields, points] = values
        mask_map[batch_index, fields, points] = 1.0
    return value_map.reshape(
        batch.obs_coords.shape[0], num_fields, *logical_shape
    ), mask_map.reshape(batch.obs_coords.shape[0], num_fields, *logical_shape)


def reshape_full_target(batch: ObservationBatch) -> torch.Tensor:
    if batch.target_fields is None:
        raise ValueError("training a grid model requires target_fields")
    if not batch.logical_shapes or len(set(batch.logical_shapes)) != 1:
        raise ValueError("grid batches require one shared logical shape")
    logical_shape = batch.logical_shapes[0]
    if math.prod(logical_shape) != batch.target_fields.shape[1] or not batch.query_valid_mask.all():
        raise ValueError("grid models require the complete unpadded query grid")
    return batch.target_fields.transpose(1, 2).reshape(
        batch.target_fields.shape[0], -1, *logical_shape
    )
