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
    periods: tuple[float, float] | None
    diagnostics: dict[str, float | int]


def _infer_period(values: np.ndarray, axis_name: str) -> float:
    unique = np.unique(values)
    if unique.size < 2:
        raise ValueError(f"cannot infer topology period for {axis_name}: fewer than two coordinates")
    gaps = np.diff(unique)
    pitch = float(np.median(gaps))
    tolerance = max(1e-10, abs(pitch) * 1e-5)
    if pitch <= 0 or not np.allclose(gaps, pitch, rtol=1e-5, atol=tolerance):
        raise ValueError(
            f"cannot infer topology period for {axis_name}: projected coordinates are not an "
            "approximately uniform endpoint-free lattice; configure geometry.periods"
        )
    return float(unique[-1] - unique[0] + pitch)


def build_raster_map(
    coordinates: torch.Tensor,
    *,
    grid_shape: tuple[int, int],
    axes: tuple[int, int] = (0, 1),
    neighbors: int = 4,
    power: float = 2.0,
    periodic: bool = False,
    periods: tuple[float, float] | None = None,
    allow_projected_collisions: bool = False,
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
    unique_coords = np.unique(coords, axis=0)
    unique_count = int(unique_coords.shape[0])
    collision_fraction = 1.0 - unique_count / float(coords.shape[0])
    if unique_count != coords.shape[0] and not allow_projected_collisions:
        raise ValueError(
            "topology projection contains coordinate collisions; select physically unique axes "
            "or explicitly set geometry.allow_projected_collisions=true"
        )
    nearest = cKDTree(unique_coords).query(unique_coords, k=2)[0][:, 1]
    finite_nearest = nearest[np.isfinite(nearest)]
    if not finite_nearest.size:
        raise ValueError("topology projection has insufficient distinct coordinates")
    minimum = coords.min(axis=0)
    maximum = coords.max(axis=0)
    span = maximum - minimum
    if np.any(span <= 0):
        raise ValueError("topology axes must both have nonzero coordinate span")
    resolved_periods: tuple[float, float] | None = None
    if periodic:
        if periods is None:
            x_count = np.unique(coords[:, 0]).size
            y_count = np.unique(coords[:, 1]).size
            if unique_count != x_count * y_count:
                raise ValueError(
                    "cannot infer topology periods: projected coordinates do not form a complete "
                    "endpoint-free Cartesian lattice; configure geometry.periods"
                )
            resolved_periods = (
                _infer_period(coords[:, 0], "x"),
                _infer_period(coords[:, 1], "y"),
            )
        else:
            if len(periods) != 2 or any(not np.isfinite(value) or value <= 0 for value in periods):
                raise ValueError("topology geometry.periods must contain two positive finite values")
            resolved_periods = (float(periods[0]), float(periods[1]))
        if any(resolved_periods[index] <= span[index] for index in range(2)):
            raise ValueError("topology periodic periods must be greater than projected coordinate spans")
    if periodic:
        x = minimum[0] + np.arange(width) * resolved_periods[0] / width
        y = minimum[1] + np.arange(height) * resolved_periods[1] / height
    else:
        x = np.linspace(minimum[0], maximum[0], width)
        y = np.linspace(minimum[1], maximum[1], height)
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    grid = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2)

    source = coords
    source_ids = np.arange(coords.shape[0])
    if periodic:
        copies = []
        copy_ids = []
        for shift_x, shift_y in product((-1, 0, 1), repeat=2):
            copies.append(
                coords
                + np.asarray(
                    (shift_x * resolved_periods[0], shift_y * resolved_periods[1])
                )
            )
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
        periods=resolved_periods,
        diagnostics={
            "point_count": int(coords.shape[0]),
            "unique_projected_count": unique_count,
            "collision_fraction": collision_fraction,
            "nearest_spacing_min": float(finite_nearest.min()),
            "nearest_spacing_median": float(np.median(finite_nearest)),
            "nearest_spacing_max": float(finite_nearest.max()),
        },
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
