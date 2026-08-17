"""Fixed differentiable point-cloud rasterization for topology coherence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
from scipy.spatial import cKDTree


def coordinate_digest(coordinates: torch.Tensor) -> str:
    array = coordinates.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


@dataclass(frozen=True)
class RasterMap:
    neighbor_indices: torch.Tensor
    neighbor_weights: torch.Tensor
    grid_coordinates: torch.Tensor
    grid_shape: tuple[int, int]
    coordinate_sha256: str


def build_raster_map(
    coordinates: torch.Tensor,
    *,
    grid_shape: tuple[int, int],
    axes: tuple[int, int] = (0, 1),
    neighbors: int = 4,
    power: float = 2.0,
    periodic: bool = False,
) -> RasterMap:
    """Precompute inverse-distance weights; gradients flow only through field values."""
    if coordinates.ndim != 2 or coordinates.shape[0] < 4:
        raise ValueError("topology coordinates must have shape [N,D] with N>=4")
    if len(set(axes)) != 2 or min(axes) < 0 or max(axes) >= coordinates.shape[1]:
        raise ValueError("topology geometry.axes must name two distinct coordinate columns")
    height, width = (int(value) for value in grid_shape)
    if height < 2 or width < 2:
        raise ValueError("topology grid_shape must contain dimensions >=2")
    coords = coordinates.detach().to(device="cpu", dtype=torch.float64)[:, axes].numpy()
    minimum = coords.min(axis=0)
    maximum = coords.max(axis=0)
    span = maximum - minimum
    if np.any(span <= 0):
        raise ValueError("topology axes must both have nonzero coordinate span")
    y = np.linspace(minimum[1], maximum[1], height, endpoint=not periodic)
    x = np.linspace(minimum[0], maximum[0], width, endpoint=not periodic)
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    grid = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2)

    source = coords
    source_ids = np.arange(coords.shape[0])
    if periodic:
        copies = []
        copy_ids = []
        for shift_x, shift_y in product((-1, 0, 1), repeat=2):
            copies.append(coords + np.asarray((shift_x * span[0], shift_y * span[1])))
            copy_ids.append(source_ids)
        source = np.concatenate(copies, axis=0)
        source_ids = np.concatenate(copy_ids, axis=0)
    count = min(max(int(neighbors), 1), coords.shape[0])
    distances, indices = cKDTree(source).query(grid, k=count)
    if count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    indices = source_ids[indices]
    exact = distances <= 1e-12
    weights = 1.0 / np.maximum(distances, 1e-12) ** float(power)
    exact_rows = exact.any(axis=1)
    if exact_rows.any():
        weights[exact_rows] = exact[exact_rows].astype(np.float64)
    weights /= weights.sum(axis=1, keepdims=True)
    return RasterMap(
        neighbor_indices=torch.from_numpy(indices.astype(np.int64)),
        neighbor_weights=torch.from_numpy(weights).float(),
        grid_coordinates=torch.from_numpy(grid).float().reshape(height, width, 2),
        grid_shape=(height, width),
        coordinate_sha256=coordinate_digest(coordinates),
    )


def rasterize_fields(
    fields: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_weights: torch.Tensor,
    grid_shape: tuple[int, int],
) -> torch.Tensor:
    """Map `[B,N,C]` values to `[B,C,H,W]` with a fixed linear operator."""
    if fields.ndim != 3:
        raise ValueError("rasterize_fields expects [B,N,C]")
    selected = fields[:, neighbor_indices]
    values = (selected * neighbor_weights[None, :, :, None]).sum(dim=2)
    height, width = grid_shape
    return values.reshape(fields.shape[0], height, width, fields.shape[2]).permute(0, 3, 1, 2)
