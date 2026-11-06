"""Fixed differentiable point-cloud rasterization for topology coherence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
from scipy import sparse
from scipy.spatial import Delaunay, QhullError, cKDTree


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
    diagnostics: dict[str, float | int | str]
    valid_mask: torch.Tensor | None = None


def _infer_period(values: np.ndarray, axis_name: str) -> float:
    unique = np.unique(values)
    if unique.size < 2:
        raise ValueError(
            f"cannot infer topology period for {axis_name}: fewer than two coordinates"
        )
    gaps = np.diff(unique)
    pitch = float(np.median(gaps))
    tolerance = max(1e-10, abs(pitch) * 1e-5)
    if pitch <= 0 or not np.allclose(gaps, pitch, rtol=1e-5, atol=tolerance):
        raise ValueError(
            f"cannot infer topology period for {axis_name}: projected coordinates are not an "
            "approximately uniform endpoint-free lattice; configure geometry.periods"
        )
    return float(unique[-1] - unique[0] + pitch)


def _physical_axis_overlap(axis: np.ndarray, size: int):
    """Overlap of native nodal control cells with uniform physical raster cells.

    The domain ends at the endpoint nodes; interior edges are midpoints. Thus
    source cell widths are the standard trapezoidal quadrature weights.
    """
    edges = np.concatenate(([axis[0]], (axis[:-1] + axis[1:]) / 2, [axis[-1]]))
    output_edges = np.linspace(axis[0], axis[-1], size + 1)
    lengths = np.maximum(
        0.0,
        np.minimum(output_edges[1:, None], edges[None, 1:])
        - np.maximum(output_edges[:-1, None], edges[None, :-1]),
    )
    return sparse.csr_matrix(lengths / np.diff(output_edges)[:, None]), output_edges


def build_physical_cell_overlap_map(
    native_coordinates: torch.Tensor,
    query_indices: torch.Tensor,
    *,
    grid_shape: tuple[int, int],
    query_coordinates: torch.Tensor | None = None,
    axes: tuple[int, int] = (0, 1),
    neighbors: int = 4,
    power: float = 2.0,
) -> RasterMap:
    """Build a fixed physical control-cell average or an explicit sparse surrogate.

    The native support must be a complete nonperiodic rectilinear grid. With
    all native points, the operator exactly integrates the piecewise-constant
    nodal control cells. For a strict subset, native values are first estimated
    by raw-coordinate inverse-distance interpolation, and that reconstructed
    field is averaged. This is a sparse surrogate, not native antialiasing.
    `query_coordinates` binds later model-coordinate identity independently of
    the raw physical geometry used for the overlap and interpolation.
    """
    if native_coordinates.ndim != 2 or native_coordinates.shape[0] < 4:
        raise ValueError("physical overlap requires native coordinates [N,D], N>=4")
    if len(set(axes)) != 2 or min(axes) < 0 or max(axes) >= native_coordinates.shape[1]:
        raise ValueError("physical overlap axes must name two distinct coordinate columns")
    height, width = map(int, grid_shape)
    if min(height, width) < 2:
        raise ValueError("physical overlap raster dimensions must be >=2")
    coords = native_coordinates.detach().to(device="cpu", dtype=torch.float64)[:, axes].numpy()
    if not np.isfinite(coords).all():
        raise ValueError("physical overlap requires finite native coordinates")
    xs, ys = np.unique(coords[:, 0]), np.unique(coords[:, 1])
    if min(len(xs), len(ys)) < 2 or len(xs) * len(ys) != len(coords):
        raise ValueError("physical overlap requires a complete rectilinear native grid")
    order = np.lexsort((coords[:, 0], coords[:, 1]))
    expected = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    if not np.array_equal(coords[order], expected):
        raise ValueError("physical overlap requires unique complete rectilinear coordinates")
    selected = torch.as_tensor(query_indices).detach().cpu()
    if selected.ndim != 1 or selected.numel() < 4 or selected.dtype == torch.bool:
        raise ValueError("physical overlap query_indices must be a unique integer vector")
    if selected.is_floating_point() and not torch.equal(selected, selected.round()):
        raise ValueError("physical overlap query_indices must be integers")
    selected = selected.long().numpy()
    if np.any(selected < 0) or np.any(selected >= len(coords)) or len(np.unique(selected)) != len(selected):
        raise ValueError("physical overlap query_indices must be unique native indices")
    if query_coordinates is None:
        query_coordinates = native_coordinates[selected]
    if query_coordinates.ndim != 2 or query_coordinates.shape[0] != len(selected):
        raise ValueError("query coordinates must align with query indices")
    if not bool(torch.isfinite(query_coordinates).all()):
        raise ValueError("query coordinates must be finite")
    if neighbors < 1 or not np.isfinite(power) or power <= 0:
        raise ValueError("physical overlap interpolation needs positive neighbors and power")

    wx, x_edges = _physical_axis_overlap(xs, width)
    wy, y_edges = _physical_axis_overlap(ys, height)
    # Columns of the Kronecker product follow sorted native (row, column).
    native_to_raster = sparse.kron(wy, wx, format="csr")
    complete = len(selected) == len(coords)
    if complete:
        inverse = np.empty(len(coords), dtype=np.int64)
        inverse[selected] = np.arange(len(selected))
        final = native_to_raster[:, np.argsort(inverse[order])]
        nearest_max = 0.0
        outside_count = 0
        outside_area_fraction = 0.0
        mapping_name = "physical_cell_overlap_v1"
    else:
        try:
            outside = Delaunay(coords[selected]).find_simplex(coords, tol=1e-10) < 0
        except QhullError as error:
            raise ValueError("sparse physical raster queries must span a 2D domain") from error
        x_native_edges = np.concatenate(([xs[0]], (xs[:-1] + xs[1:]) / 2, [xs[-1]]))
        y_native_edges = np.concatenate(([ys[0]], (ys[:-1] + ys[1:]) / 2, [ys[-1]]))
        sorted_area = np.outer(np.diff(y_native_edges), np.diff(x_native_edges)).ravel()
        native_area = np.empty(len(coords), dtype=np.float64)
        native_area[order] = sorted_area
        outside_count = int(outside.sum())
        outside_area_fraction = float(native_area[outside].sum() / native_area.sum())
        neighbor_count = min(int(neighbors), len(selected))
        distances, nearest = cKDTree(coords[selected]).query(coords, k=neighbor_count)
        if neighbor_count == 1:
            distances, nearest = distances[:, None], nearest[:, None]
        exact = distances <= 1e-12
        weights = 1.0 / np.maximum(distances, 1e-12) ** float(power)
        exact_rows = exact.any(axis=1)
        weights[exact_rows] = exact[exact_rows].astype(np.float64)
        weights /= weights.sum(axis=1, keepdims=True)
        interpolation = sparse.csr_matrix(
            (weights.reshape(-1), (np.repeat(np.arange(len(coords)), neighbor_count), nearest.reshape(-1))),
            shape=(len(coords), len(selected)),
        )
        final = native_to_raster @ interpolation[order]
        nearest_max = float(distances[:, 0].max())
        mapping_name = "sparse_idw_then_physical_cell_overlap_v1"
    final = final.tocsr()
    final.eliminate_zeros()
    if not np.allclose(np.asarray(final.sum(axis=1)).ravel(), 1.0, rtol=1e-10, atol=1e-10):
        raise RuntimeError("physical overlap raster does not cover its declared domain")
    max_support = int(np.diff(final.indptr).max())
    indices = np.zeros((height * width, max_support), dtype=np.int64)
    weights = np.zeros_like(indices, dtype=np.float64)
    for row in range(height * width):
        start, end = final.indptr[row : row + 2]
        row_size = end - start
        indices[row, :row_size] = final.indices[start:end]
        weights[row, :row_size] = final.data[start:end]
    centers_x = (x_edges[:-1] + x_edges[1:]) / 2
    centers_y = (y_edges[:-1] + y_edges[1:]) / 2
    grid = np.stack(np.meshgrid(centers_x, centers_y, indexing="xy"), axis=-1)
    return RasterMap(
        neighbor_indices=torch.from_numpy(indices),
        neighbor_weights=torch.from_numpy(weights).float(),
        grid_coordinates=torch.from_numpy(grid).float(),
        grid_shape=(height, width),
        coordinate_sha256=coordinate_digest(query_coordinates),
        periods=None,
        diagnostics={
            "mapping": mapping_name,
            "physical_coordinates_sha256": coordinate_digest(native_coordinates),
            "query_coordinates_sha256": coordinate_digest(query_coordinates),
            "source_edge_rule": "native_endpoints_and_interior_midpoints",
            "target_edge_rule": "uniform_physical_edges_over_native_extent",
            "native_shape": f"{len(ys)}x{len(xs)}",
            "native_points": len(coords),
            "query_points": len(selected),
            "sparse_interpolation": int(not complete),
            "sparse_neighbors": 0 if complete else neighbor_count,
            "sparse_power": 0.0 if complete else float(power),
            "sparse_max_nearest_raw_distance": nearest_max,
            "sparse_outside_convex_hull_native_points": outside_count,
            "sparse_outside_convex_hull_area_fraction": outside_area_fraction,
            "antialias_applied": int(complete),
        },
    )


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
    antialias_downsample: bool = False,
    native_masked: bool = False,
    native_grid_indices: torch.Tensor | None = None,
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
    if not np.isfinite(coords).all():
        raise ValueError("topology coordinates must be finite")
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
                raise ValueError(
                    "topology geometry.periods must contain two positive finite values"
                )
            resolved_periods = (float(periods[0]), float(periods[1]))
        if any(resolved_periods[index] <= span[index] for index in range(2)):
            raise ValueError(
                "topology periodic periods must be greater than projected coordinate spans"
            )
    if periodic:
        x = minimum[0] + np.arange(width) * resolved_periods[0] / width
        y = minimum[1] + np.arange(height) * resolved_periods[1] / height
    else:
        x = np.linspace(minimum[0], maximum[0], width)
        y = np.linspace(minimum[1], maximum[1], height)
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    grid = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2)

    if native_masked:
        if periodic or antialias_downsample or unique_count != len(coords):
            raise ValueError("native_masked topology requires a unique nonperiodic native lattice")
        if native_grid_indices is None:
            xs, ys = np.unique(coords[:, 0]), np.unique(coords[:, 1])
            if len(xs) == width and len(ys) == height and len(coords) == height * width:
                cells = np.stack((np.searchsorted(xs, coords[:, 0]), np.searchsorted(ys, coords[:, 1])), axis=-1)
                error = np.zeros_like(coords)
            else:
                pitch = span / np.asarray((width - 1, height - 1))
                cells = np.rint((coords - minimum) / pitch).astype(np.int64)
                error = np.abs(coords - (minimum + cells * pitch)) / pitch
        else:
            rc = native_grid_indices.detach().cpu().numpy()
            if (
                rc.shape != (len(coords), 2)
                or not np.isfinite(rc).all()
                or not np.equal(rc, np.rint(rc)).all()
            ):
                raise ValueError("native grid indices must be integer [N,2] row/column indices")
            rc = rc.astype(np.int64) - rc.min(axis=0).astype(np.int64)
            if tuple(rc.max(axis=0) + 1) != (height, width):
                raise ValueError("native grid indices extent must match grid_shape")
            cells, error = rc[:, ::-1], np.zeros_like(rc)
        flat_ids = cells[:, 1] * width + cells[:, 0]
        if error.max() > 1e-3 or len(np.unique(flat_ids)) != len(coords):
            raise ValueError("native_masked grid_shape does not match the source lattice")
        indices = np.zeros((height * width, 1), dtype=np.int64)
        weights = np.zeros_like(indices, dtype=np.float64)
        indices[flat_ids, 0] = np.arange(len(coords))
        weights[flat_ids, 0] = 1.0
        grid[flat_ids] = coords
        valid = weights[:, 0].astype(bool).reshape(height, width)
        return RasterMap(
            neighbor_indices=torch.from_numpy(indices),
            neighbor_weights=torch.from_numpy(weights).float(),
            grid_coordinates=torch.from_numpy(grid).float().reshape(height, width, 2),
            grid_shape=(height, width),
            coordinate_sha256=coordinate_digest(coordinates),
            periods=None,
            valid_mask=torch.from_numpy(valid),
            diagnostics={
                "point_count": len(coords),
                "native_masked": 1,
                "valid_vertices": int(valid.sum()),
                "excluded_vertices": int((~valid).sum()),
                "max_lattice_error_cells": float(error.max()),
                "antialias_applied": 0,
            },
        )

    # Use every native lattice value when reducing a complete Cartesian grid.
    # Express block averaging in the existing fixed linear-map contract so
    # shuffled point order, checkpointing and autograd need no special path.
    xs, ys = np.unique(coords[:, 0]), np.unique(coords[:, 1])
    native_h, native_w = len(ys), len(xs)
    area_indices = None
    if (
        antialias_downsample
        and unique_count == len(coords) == native_h * native_w
        and native_h >= height
        and native_w >= width
        and native_h % height == 0
        and native_w % width == 0
    ):
        order = np.lexsort((coords[:, 0], coords[:, 1])).reshape(native_h, native_w)
        factor_h, factor_w = native_h // height, native_w // width
        area_indices = (
            order.reshape(height, factor_h, width, factor_w)
            .transpose(0, 2, 1, 3)
            .reshape(height * width, factor_h * factor_w)
        )

    source = coords
    source_ids = np.arange(coords.shape[0])
    if periodic:
        copies = []
        copy_ids = []
        for shift_x, shift_y in product((-1, 0, 1), repeat=2):
            copies.append(
                coords + np.asarray((shift_x * resolved_periods[0], shift_y * resolved_periods[1]))
            )
            copy_ids.append(source_ids)
        source = np.concatenate(copies, axis=0)
        source_ids = np.concatenate(copy_ids, axis=0)
    if area_indices is not None:
        indices = area_indices
        weights = np.full(indices.shape, 1.0 / indices.shape[1])
        grid = coords[indices].mean(axis=1)
    else:
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
            "antialias_applied": int(area_indices is not None),
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
