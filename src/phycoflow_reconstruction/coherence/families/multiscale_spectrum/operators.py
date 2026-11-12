"""Spatial measure, self-adjoint operator, eigenbasis, filters, and regions.

Every quantity here is fixed geometry: it depends on the shared query
coordinates and declared options only, never on fields or targets.

The generalized eigenproblem is ``K phi = lambda M phi`` with ``K`` symmetric
positive semidefinite, ``K 1 = 0``, and ``M = diag(m)``, ``sum(m) = 1``. Both
matrices are divided by the same total measure, so normalizing ``M`` does not
change the eigenvalues. Eigenvectors are ``M``-orthonormal and the constant
null vector is removed; the spatial mean is handled separately.

``fem_p1`` assembles the P1 finite-element stiffness (cotangent weights) on a
Delaunay triangulation of the points with a mixed Voronoi lumped mass. Without
periodic axes this is the Neumann Laplacian of the triangulated domain, so
``sqrt(lambda)`` is a wavenumber in the coordinate length unit. Mixed Voronoi
areas and cotangent weights do not depend on how co-circular points are split,
so regular lattices give a unique operator. ``knn_graph`` is a fallback for any
dimension; its eigenvalues are graph scales, not physical wavenumbers.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.spatial import Delaunay, cKDTree

EARTH_RADIUS_KM = 6371.0088
DENSE_EIGEN_LIMIT = 4096
SLIVER_QUALITY = 1.0e-6


@dataclass(frozen=True)
class SpatialOperator:
    stiffness: sparse.csr_matrix
    mass: np.ndarray
    kind: str
    diagnostics: dict
    # Indices of the input points that form the analysis domain.
    nodes: np.ndarray
    # Connected piece (0 = largest) of every analysis node.
    components: np.ndarray


@dataclass(frozen=True)
class SpectralBasis:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    diagnostics: dict


def coordinate_digest(coordinates: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(coordinates, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def active_axes(coordinates: np.ndarray, tolerance: float = 1.0e-12) -> tuple[int, ...]:
    span = coordinates.max(axis=0) - coordinates.min(axis=0)
    axes = tuple(int(axis) for axis in np.flatnonzero(span > tolerance))
    if not axes:
        raise ValueError("multiscale geometry needs a coordinate axis with nonzero span")
    return axes


def lonlat_axis_lengths(
    minimum: np.ndarray, maximum: np.ndarray, mean_latitude: float
) -> tuple[float, float]:
    """Equirectangular km extents of (longitude, latitude) degree spans."""
    scale = math.pi / 180.0 * EARTH_RADIUS_KM
    return (
        float((maximum[0] - minimum[0]) * scale * math.cos(math.radians(mean_latitude))),
        float((maximum[1] - minimum[1]) * scale),
    )


def _local_spacing(points: np.ndarray, neighbors: int, boxsize: np.ndarray | None) -> np.ndarray:
    count = min(int(neighbors), points.shape[0] - 1)
    tree = cKDTree(points, boxsize=boxsize)
    distances, _ = tree.query(points, k=count + 1)
    return np.asarray(distances[:, count], dtype=np.float64)


def _periodic_box(
    points: np.ndarray, periodic: tuple[bool, ...], periods: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Shift points into a cKDTree box; non-periodic axes get a box wide enough not to wrap."""
    if not any(periodic):
        return points, None
    minimum = points.min(axis=0)
    shifted = points - minimum
    span = shifted.max(axis=0)
    boxsize = np.where(periodic, periods, 3.0 * span + 1.0)
    if np.any(shifted >= boxsize) and any(periodic):
        raise ValueError("periodic coordinate span must be smaller than the declared period")
    return shifted, boxsize


def _fem_1d(
    points: np.ndarray, periodic: bool, period: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    x = points[:, 0]
    order = np.argsort(x, kind="stable")
    sorted_x = x[order]
    lengths = np.diff(sorted_x)
    if np.any(lengths <= 0):
        raise ValueError("fem_p1 coordinates contain duplicate points")
    left, right = order[:-1], order[1:]
    if periodic:
        wrap = sorted_x[0] + float(period) - sorted_x[-1]
        if wrap <= 0:
            raise ValueError("periodic coordinate span must be smaller than the declared period")
        left = np.append(left, order[-1])
        right = np.append(right, order[0])
        lengths = np.append(lengths, wrap)
    rows = np.concatenate([left, right, left, right])
    cols = np.concatenate([left, right, right, left])
    values = np.concatenate([1.0 / lengths, 1.0 / lengths, -1.0 / lengths, -1.0 / lengths])
    mass = np.zeros(x.size)
    np.add.at(mass, left, 0.5 * lengths)
    np.add.at(mass, right, 0.5 * lengths)
    box_points, boxsize = _periodic_box(
        points, (periodic,), None if period is None else np.array([period])
    )
    nearest = _local_spacing(box_points, 1, boxsize)
    return (
        rows,
        cols,
        values,
        {
            "dimension": 1,
            "elements": int(lengths.size),
            "total_measure": float(lengths.sum()),
            "max_element_length": float(lengths.max()),
            "nearest_spacing_median": float(np.median(nearest)),
            "nearest_spacing_p90": float(np.percentile(nearest, 90)),
            "mass": mass,
            "edges": np.stack((left, right), axis=1),
        },
    )


def _periodic_triangulation(
    points: np.ndarray, periodic: tuple[bool, bool], periods: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a periodic image set and keep one image of each triangle."""
    count = points.shape[0]
    shifts = [
        np.array([sx, sy], dtype=np.float64)
        for sx in ((-1, 0, 1) if periodic[0] else (0,))
        for sy in ((-1, 0, 1) if periodic[1] else (0,))
    ]
    offsets = np.stack([shift * np.where(periodic, periods, 0.0) for shift in shifts])
    tiled = np.concatenate([points + offset for offset in offsets])
    # Regular lattices are co-circular. One tiny jitter copied to every image
    # makes the image set generic, so its Delaunay triangulation is unique and
    # identical across images. Assembly uses the unperturbed coordinates.
    spacing = float(np.median(_local_spacing(points, 1, None)))
    jitter = np.random.default_rng(seed).uniform(-1.0, 1.0, points.shape) * 1.0e-6 * spacing
    triangulation = Delaunay(tiled + np.tile(jitter, (len(offsets), 1)))
    simplices = triangulation.simplices
    centroids = tiled[simplices].mean(axis=1)
    minimum = points.min(axis=0)
    inside = np.ones(simplices.shape[0], dtype=bool)
    for axis in range(2):
        if periodic[axis]:
            inside &= (centroids[:, axis] >= minimum[axis]) & (
                centroids[:, axis] < minimum[axis] + periods[axis]
            )
    selected = simplices[inside]
    # Triangles that join a point to its own image are checked after the sliver
    # filter: along a non-periodic boundary they are degenerate hull slivers.
    return selected % count, tiled[selected]


def _structured_triangles(
    points: np.ndarray, grid_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Elements of a masked structured grid from per-point logical (i, j) indices.

    Every logical cell with four present corners is split along its shorter
    physical diagonal; a cell with exactly three present corners contributes
    that triangle. Absent (masked) points never enter an element, so the mesh
    follows the grid's own coastline instead of a convex hull.
    """
    indices = np.asarray(grid_indices)
    if indices.shape != (points.shape[0], 2) or not np.allclose(indices, np.round(indices)):
        raise ValueError("structured_indices must hold one integer (i, j) pair per point")
    indices = np.round(indices).astype(np.int64)
    lower = indices.min(axis=0)
    shape = indices.max(axis=0) - lower + 2
    lookup = np.full(tuple(shape), -1, dtype=np.int64)
    local = indices - lower
    if np.unique(local, axis=0).shape[0] != local.shape[0]:
        raise ValueError("structured_indices contain duplicate grid positions")
    lookup[local[:, 0], local[:, 1]] = np.arange(points.shape[0])
    a, b = lookup[:-1, :-1], lookup[1:, :-1]
    c, d = lookup[1:, 1:], lookup[:-1, 1:]
    corners = np.stack((a, b, c, d), axis=-1).reshape(-1, 4)
    present = corners >= 0
    triangles = []
    full = corners[present.all(axis=1)]
    if full.size:
        first = np.linalg.norm(points[full[:, 0]] - points[full[:, 2]], axis=1)
        second = np.linalg.norm(points[full[:, 1]] - points[full[:, 3]], axis=1)
        short = first <= second
        triangles += [
            full[short][:, [0, 1, 2]],
            full[short][:, [0, 2, 3]],
            full[~short][:, [0, 1, 3]],
            full[~short][:, [1, 2, 3]],
        ]
    partial = corners[present.sum(axis=1) == 3]
    if partial.size:
        triangles.append(np.stack([row[row >= 0] for row in partial]))
    if not triangles:
        raise ValueError("structured_indices define no grid cell with three present corners")
    vertex_ids = np.concatenate(triangles)
    return vertex_ids, points[vertex_ids]


def _fem_2d(
    points: np.ndarray,
    periodic: tuple[bool, bool],
    periods: np.ndarray | None,
    *,
    max_edge_factor: float | None,
    spacing_neighbors: int,
    seed: int,
    grid_indices: np.ndarray | None = None,
    support_points: np.ndarray | None = None,
    support_factor: float | None = None,
    support_neighbors: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    count = points.shape[0]
    if count < 3:
        raise ValueError("fem_p1 needs at least three points in two dimensions")
    if grid_indices is not None:
        vertex_ids, corners = _structured_triangles(points, grid_indices)
    elif any(periodic):
        vertex_ids, corners = _periodic_triangulation(points, periodic, periods, seed)
    else:
        vertex_ids = Delaunay(points).simplices
        corners = points[vertex_ids]
    edges = np.stack(
        (
            corners[:, 2] - corners[:, 1],
            corners[:, 0] - corners[:, 2],
            corners[:, 1] - corners[:, 0],
        ),
        axis=1,
    )
    lengths_squared = np.einsum("tid,tid->ti", edges, edges)
    cross = edges[:, 2, 0] * (-edges[:, 1, 1]) - edges[:, 2, 1] * (-edges[:, 1, 0])
    area = 0.5 * np.abs(cross)
    quality = 4.0 * math.sqrt(3.0) * area / np.maximum(lengths_squared.sum(axis=1), 1.0e-300)
    keep = quality >= SLIVER_QUALITY
    removed_slivers = int((~keep).sum())
    self_image = (
        (vertex_ids[:, 0] == vertex_ids[:, 1])
        | (vertex_ids[:, 1] == vertex_ids[:, 2])
        | (vertex_ids[:, 0] == vertex_ids[:, 2])
    )
    if np.any(self_image & keep):
        raise ValueError("periodic triangulation is too coarse: a triangle touches two images")
    keep &= ~self_image
    removed_long = 0
    box_points, boxsize = _periodic_box(points, periodic, periods)
    spacing = _local_spacing(box_points, spacing_neighbors, boxsize)
    if max_edge_factor is not None:
        # The median of the three vertex spacings: one isolated (sparse) vertex cannot
        # license a bridge between two dense regions.
        local = np.median(spacing[vertex_ids], axis=1)
        long = np.sqrt(lengths_squared.max(axis=1)) > float(max_edge_factor) * local
        removed_long = int((long & keep).sum())
        keep &= ~long
    removed_unsupported = 0
    if support_factor is not None:
        # A triangle belongs to the domain when its centroid and edge midpoints lie
        # within support_factor local spacings of the dataset's own point set.
        tree = cKDTree(support_points)
        support_spacing = tree.query(support_points, k=support_neighbors + 1)[0][
            :, support_neighbors
        ]
        samples = np.concatenate(
            (corners.mean(axis=1, keepdims=True), 0.5 * (corners + np.roll(corners, -1, axis=1))),
            axis=1,
        )
        distance, nearest_support = tree.query(samples.reshape(-1, 2))
        ratio = (
            (distance / support_spacing[nearest_support]).reshape(-1, samples.shape[1]).max(axis=1)
        )
        unsupported = ratio > float(support_factor)
        removed_unsupported = int((unsupported & keep).sum())
        keep &= ~unsupported
    vertex_ids, edges, lengths_squared, area = (
        vertex_ids[keep],
        edges[keep],
        lengths_squared[keep],
        area[keep],
    )
    if vertex_ids.size == 0:
        raise ValueError("fem_p1 triangulation retained no triangles")
    # Local P1 stiffness with cyclic opposite-edge vectors: K_ij = e_i.e_j/(4A).
    gram = np.einsum("tid,tjd->tij", edges, edges) / (4.0 * area[:, None, None])
    rows = np.repeat(vertex_ids, 3, axis=1).reshape(-1)
    cols = np.tile(vertex_ids, (1, 3)).reshape(-1)
    values = gram.reshape(-1)
    # Mixed Voronoi lumped mass (Meyer et al. 2003). cot(theta_k) = -e_i.e_j/(2A).
    cot = np.stack(
        (
            -gram[:, 1, 2] * 2.0,
            -gram[:, 0, 2] * 2.0,
            -gram[:, 0, 1] * 2.0,
        ),
        axis=1,
    )
    voronoi = np.stack(
        (
            (lengths_squared[:, 1] * cot[:, 1] + lengths_squared[:, 2] * cot[:, 2]) / 8.0,
            (lengths_squared[:, 0] * cot[:, 0] + lengths_squared[:, 2] * cot[:, 2]) / 8.0,
            (lengths_squared[:, 0] * cot[:, 0] + lengths_squared[:, 1] * cot[:, 1]) / 8.0,
        ),
        axis=1,
    )
    obtuse = cot < 0.0
    any_obtuse = obtuse.any(axis=1)
    mixed = np.where(
        any_obtuse[:, None],
        np.where(obtuse, area[:, None] / 2.0, area[:, None] / 4.0),
        voronoi,
    )
    mass = np.zeros(count)
    np.add.at(mass, vertex_ids.reshape(-1), mixed.reshape(-1))
    undirected = np.sort(
        np.concatenate((vertex_ids[:, [0, 1]], vertex_ids[:, [1, 2]], vertex_ids[:, [0, 2]])),
        axis=1,
    )
    unique_edges, multiplicity = np.unique(undirected, axis=0, return_counts=True)
    euler = int(np.unique(vertex_ids).size - unique_edges.shape[0] + vertex_ids.shape[0])
    if any(periodic) and not (removed_long or removed_unsupported) and euler != 0:
        # An unfiltered torus or cylinder has Euler characteristic zero; anything
        # else means the image selection produced overlaps or gaps.
        raise ValueError(f"periodic triangulation is inconsistent (Euler characteristic {euler})")
    diagnostics = {
        "dimension": 2,
        "triangulation": "structured_indices" if grid_indices is not None else "delaunay",
        "triangles": int(vertex_ids.shape[0]),
        "removed_sliver_triangles": removed_slivers,
        "removed_long_edge_triangles": removed_long,
        "removed_unsupported_triangles": removed_unsupported,
        "max_edge_factor": max_edge_factor,
        "support_factor": support_factor,
        "spacing_neighbors": int(spacing_neighbors),
        "triangulated_measure": float(area.sum()),
        "boundary_edges": int((multiplicity == 1).sum()),
        "euler_characteristic": euler,
        "obtuse_triangle_fraction": float(any_obtuse.mean()),
        "mass": mass,
        "edges": unique_edges,
    }
    return rows, cols, values, diagnostics


def fem_p1_operator(
    points: np.ndarray,
    *,
    periodic: tuple[bool, ...],
    periods: np.ndarray | None,
    max_edge_factor: float | None,
    spacing_neighbors: int,
    seed: int = 0,
    grid_indices: np.ndarray | None = None,
    support_points: np.ndarray | None = None,
    support_factor: float | None = None,
    support_neighbors: int = 8,
    connectivity: str = "require_connected",
    min_component_fraction: float = 0.01,
) -> SpatialOperator:
    """Mass-weighted P1 Laplacian on 1-D or 2-D points, in coordinate units.

    ``connectivity="components"`` keeps every connected piece holding at least
    ``min_component_fraction`` of the points; the stiffness is then block
    diagonal and each piece contributes its own constant null mode. Points in
    smaller pieces or in no element are excluded. ``SpatialOperator.nodes``
    lists the retained points, ``components`` their piece, and the diagnostics
    record what was excluded.
    """
    dimension = points.shape[1]
    if dimension == 1:
        rows, cols, values, details = _fem_1d(
            points, periodic[0], None if periods is None else float(periods[0])
        )
        details["triangulated_measure"] = details.pop("total_measure")
    elif dimension == 2:
        rows, cols, values, details = _fem_2d(
            points,
            (periodic[0], periodic[1]),
            periods,
            max_edge_factor=max_edge_factor,
            spacing_neighbors=spacing_neighbors,
            seed=seed,
            grid_indices=grid_indices,
            support_points=support_points,
            support_factor=support_factor,
            support_neighbors=support_neighbors,
        )
    else:
        raise ValueError(
            "fem_p1 supports one or two active coordinate axes; use geometry.operator=knn_graph"
        )
    edges = details.pop("edges")
    mass = details.pop("mass")
    count = points.shape[0]
    adjacency = sparse.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(count, count)
    )
    components, labels = connected_components(adjacency, directed=False)
    sizes = np.bincount(labels)
    if connectivity == "require_connected":
        if np.any(mass <= 0):
            raise ValueError(
                f"fem_p1 left {int((mass <= 0).sum())} query points outside every element; relax "
                "the triangle filters, use a denser query set, or set "
                "geometry.connectivity=components"
            )
        if components != 1:
            raise ValueError(
                f"fem_p1 analysis domain is disconnected ({components} components); set "
                "geometry.connectivity=components to analyze each piece with its own null mode"
            )
        kept = np.array([0])
    elif connectivity == "components":
        # Points in no element form singleton components and are always excluded.
        kept = np.flatnonzero(sizes >= max(3, min_component_fraction * count))
        kept = kept[np.argsort(-sizes[kept], kind="stable")]
        if kept.size == 0:
            raise ValueError("no connected mesh component reaches geometry.min_component_fraction")
    else:
        raise ValueError(f"unknown fem_p1 connectivity policy {connectivity!r}")
    relabel = np.full(components, -1)
    relabel[kept] = np.arange(kept.size)
    nodes = np.flatnonzero(relabel[labels] >= 0)
    piece = relabel[labels[nodes]]
    total_triangulated = float(mass.sum())
    stiffness = sparse.coo_matrix((values, (rows, cols)), shape=(count, count)).tocsr()
    stiffness = stiffness[nodes][:, nodes]
    mass = mass[nodes]
    total = float(mass.sum())
    stiffness = ((stiffness + stiffness.T) * (0.5 / total)).tocsr()
    box_points, boxsize = _periodic_box(points[nodes], periodic, periods)
    nearest = _local_spacing(box_points, 1, boxsize)
    details.update(
        {
            "connectivity": connectivity,
            "min_component_fraction": min_component_fraction
            if connectivity == "components"
            else None,
            "mesh_components": int(components),
            "analysis_components": int(kept.size),
            "component_points": sizes[kept].tolist(),
            "component_measure_fractions": (np.bincount(piece, weights=mass) / total).tolist(),
            "analysis_points": int(nodes.size),
            "excluded_points": int(count - nodes.size),
            "total_measure": total,
            "excluded_measure_fraction": float(1.0 - total / total_triangulated),
            "nearest_spacing_median": float(np.median(nearest)),
            "nearest_spacing_p90": float(np.percentile(nearest, 90)),
            "mass_ratio_max_to_min": float(mass.max() / mass.min()),
        }
    )
    return SpatialOperator(stiffness, mass / total, "fem_p1", details, nodes, piece)


def knn_graph_operator(
    points: np.ndarray,
    *,
    k_neighbors: int,
    sigma: float | None,
    periodic: tuple[bool, ...],
    periods: np.ndarray | None,
) -> SpatialOperator:
    """Combinatorial Gaussian kNN Laplacian with the uniform point measure."""
    count = points.shape[0]
    if count < 3:
        raise ValueError("knn_graph needs at least three points")
    box_points, boxsize = _periodic_box(points, periodic, periods)
    neighbors = min(max(int(k_neighbors), 1), count - 1)
    distances, indices = cKDTree(box_points, boxsize=boxsize).query(box_points, k=neighbors + 1)
    distances = distances[:, 1:].reshape(-1)
    columns = indices[:, 1:].reshape(-1)
    rows = np.repeat(np.arange(count), neighbors)
    positive = distances[distances > 0]
    resolved_sigma = float(np.median(positive)) if sigma is None else float(sigma)
    if not math.isfinite(resolved_sigma) or resolved_sigma <= 0:
        raise ValueError("knn_graph sigma must be positive (coordinates may contain duplicates)")
    weights = np.exp(-(distances**2) / (2.0 * resolved_sigma**2))
    adjacency = sparse.coo_matrix((weights, (rows, columns)), shape=(count, count)).tocsr()
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    components, _ = connected_components(adjacency, directed=False)
    if components != 1:
        raise ValueError(
            f"knn_graph is disconnected ({components} components); increase geometry.k_neighbors"
        )
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    # With M = I/N, (D - W)/N phi = lambda M phi has the combinatorial eigenvalues.
    stiffness = ((sparse.diags(degrees) - adjacency) / count).tocsr()
    return SpatialOperator(
        stiffness,
        np.full(count, 1.0 / count),
        "knn_graph",
        {
            "k_neighbors": neighbors,
            "resolved_sigma": resolved_sigma,
            "scale": "graph",
            "analysis_points": count,
            "excluded_points": 0,
            "analysis_components": 1,
        },
        np.arange(count),
        np.zeros(count, dtype=np.int64),
    )


def _deterministic_signs(vectors: np.ndarray) -> np.ndarray:
    pivots = np.abs(vectors).argmax(axis=0)
    signs = np.sign(vectors[pivots, np.arange(vectors.shape[1])])
    signs[signs == 0] = 1.0
    return vectors * signs[None, :]


def _iterative_eigenpairs(
    matrix: sparse.csr_matrix,
    *,
    cutoff: float | None,
    required: int,
    cap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Smallest eigenpairs by shift-invert Lanczos, growing until the cutoff is covered."""
    count = matrix.shape[0]
    shift = -1.0e-8 * float(matrix.diagonal().mean())
    start = np.random.default_rng(seed).standard_normal(count)
    # A count truncation needs every requested pair; a cutoff grows from a small solve.
    wanted = min(required if cutoff is None else 64, cap, count - 2)
    while True:
        values, vectors = eigsh(matrix, k=wanted, sigma=shift, which="LM", v0=start)
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]
        if cutoff is None or values[-1] > cutoff:
            return values, vectors
        if wanted >= min(cap, count - 2):
            raise ValueError(
                "the requested spectral cutoff needs more than geometry.max_modes modes; raise "
                "geometry.max_modes or the finest analyzed wavelength"
            )
        wanted = min(math.ceil(1.5 * wanted), cap, count - 2)


def spectral_basis(
    operator: SpatialOperator,
    *,
    max_modes: int,
    cutoff_eigenvalue: float | None,
    seed: int = 0,
    dense_limit: int = DENSE_EIGEN_LIMIT,
) -> SpectralBasis:
    """M-orthonormal eigenpairs orthogonal to every piecewise constant.

    The null space has one constant mode per connected piece; those modes are
    removed (the means are handled separately), and the remaining modes are
    kept up to a cutoff eigenvalue or a gap-checked count.
    """
    mass = operator.mass
    count = mass.size
    pieces = operator.components
    piece_count = int(pieces.max()) + 1
    indicator = np.zeros((count, piece_count))
    indicator[np.arange(count), pieces] = 1.0
    piece_mass = mass @ indicator
    inverse_sqrt = 1.0 / np.sqrt(mass)
    scaled = sparse.diags(inverse_sqrt) @ operator.stiffness @ sparse.diags(inverse_sqrt)
    scaled = ((scaled + scaled.T) * 0.5).tocsr()
    if count <= dense_limit:
        values, vectors = linalg.eigh(scaled.toarray())
    else:
        # The null modes plus one extra pair for the truncation-gap check.
        values, vectors = _iterative_eigenpairs(
            scaled,
            cutoff=cutoff_eigenvalue,
            required=max_modes + piece_count + 1,
            cap=max_modes + piece_count + 1,
            seed=seed,
        )
    scale = max(float(np.abs(values).max()), 1.0e-300)
    if np.any(np.abs(values[:piece_count]) > 1.0e-8 * scale):
        raise ValueError(
            "operator has fewer null modes than connected pieces; it is not a Laplacian"
        )
    if values.size <= piece_count or float(values[piece_count]) <= 1.0e-10 * scale:
        raise ValueError("operator has more null modes than connected pieces")
    phi = inverse_sqrt[:, None] * vectors

    def piece_means(modes: np.ndarray) -> np.ndarray:
        return (indicator.T @ (mass[:, None] * modes)) / piece_mass[:, None]

    null = phi[:, :piece_count]
    if np.abs(null - indicator @ piece_means(null)).max() > 1.0e-6 * np.abs(null).max():
        raise ValueError("operator null vectors are not piecewise constant; K 1 = 0 is violated")
    values, phi = values[piece_count:], phi[:, piece_count:]
    # Remove round-off piecewise constants so filtered fields are exactly mean-free per piece.
    phi = phi - indicator @ piece_means(phi)
    phi = phi / np.sqrt(np.einsum("n,nk,nk->k", mass, phi, phi))[None, :]
    truncated_by = "all_modes"
    if cutoff_eigenvalue is not None:
        # A relative tolerance keeps an eigenspace that sits exactly on the cutoff whole;
        # its filter weight is zero there, so this only fixes the basis size.
        retained = int(np.searchsorted(values, cutoff_eigenvalue * (1 + 1e-9), side="right"))
        if retained > max_modes:
            raise ValueError(
                f"{retained} modes lie below the spectral cutoff, above geometry.max_modes={max_modes}"
            )
        if retained < 1:
            raise ValueError("no nonconstant mode lies below the spectral cutoff")
        truncated_by = "cutoff_eigenvalue"
    else:
        retained = min(max_modes, values.size)
        if retained < values.size:
            truncated_by = "max_modes"
            # Never split an exactly degenerate eigenspace at the truncation.
            while (
                retained > 0
                and values[retained] - values[retained - 1] <= 1.0e-9 * values[retained]
            ):
                retained -= 1
            if retained == 0:
                raise ValueError(
                    "geometry.max_modes truncates inside the first degenerate eigenspace; "
                    "raise geometry.max_modes"
                )
    values, phi = values[:retained], _deterministic_signs(phi[:, :retained])
    gaps = np.diff(values)
    return SpectralBasis(
        values,
        phi,
        {
            "retained_modes": int(retained),
            "truncated_by": truncated_by,
            "first_eigenvalue": float(values[0]),
            "last_eigenvalue": float(values[-1]),
            "minimum_relative_gap": None if gaps.size == 0 else float((gaps / values[1:]).min()),
            "eigen_solver": "dense" if count <= dense_limit else "shift_invert_lanczos",
        },
    )


def tight_filter_bank(wavenumbers: np.ndarray, centers: np.ndarray, cutoff: float) -> np.ndarray:
    """Real band multipliers that, with one discarded unresolved band, form a tight frame.

    Adjacent bands cross over with cos/sin windows in log wavenumber. The
    coarsest band is one below its center. The finest band crosses over in
    the same way into an unresolved band that ends at ``cutoff``, so
    ``sum_m g_m(k)^2 = 1`` up to the finest center, decays smoothly to zero at
    the cutoff, and the missing part is exactly the unresolved energy. Equal
    eigenvalues receive equal weights by construction.
    """
    if np.any(wavenumbers <= 0) or np.any(wavenumbers > cutoff * (1 + 1e-9)):
        raise ValueError("filter bank wavenumbers must be positive and not above the cutoff")
    if centers.ndim != 1 or centers.size < 1 or np.any(np.diff(centers) <= 0):
        raise ValueError("band centers must be strictly increasing")
    if centers[-1] >= cutoff:
        raise ValueError("the finest band center must lie below the resolution cutoff")
    anchors = np.log(np.append(centers, cutoff))
    position = np.minimum(np.log(wavenumbers), anchors[-1])
    windows = np.zeros((anchors.size, wavenumbers.size))
    windows[0, position < anchors[0]] = 1.0
    for band in range(anchors.size - 1):
        inside = (position >= anchors[band]) & (position <= anchors[band + 1])
        fraction = (position[inside] - anchors[band]) / (anchors[band + 1] - anchors[band])
        windows[band, inside] = np.cos(0.5 * math.pi * fraction)
        windows[band + 1, inside] = np.sin(0.5 * math.pi * fraction)
    coverage = np.square(windows).sum(axis=0)
    if not np.allclose(coverage, 1.0, rtol=0.0, atol=1.0e-12):
        raise AssertionError("filter bank with its unresolved band is not tight")
    return windows[:-1]


def region_measures(
    coordinates: np.ndarray,
    mass: np.ndarray,
    *,
    grid: tuple[int, ...],
    periodic: tuple[bool, ...],
    periods: np.ndarray | None,
    min_mass_fraction: float,
) -> tuple[np.ndarray, tuple[str, ...], dict]:
    """Fixed overlapping raised-cosine windows as normalized regional measures."""
    if len(grid) != coordinates.shape[1]:
        raise ValueError("regions.grid must give one window count per active coordinate axis")
    windows = []
    for axis, count in enumerate(grid):
        values = coordinates[:, axis]
        if count == 1:
            windows.append(np.ones((1, values.size)))
            continue
        minimum = values.min()
        if periodic[axis]:
            width = float(periods[axis]) / count
            centers = minimum + width * np.arange(count)
            offset = values[None, :] - centers[:, None] + 0.5 * float(periods[axis])
            distance = np.abs(np.mod(offset, float(periods[axis])) - 0.5 * float(periods[axis]))
        else:
            span = values.max() - minimum
            width = span / (count - 1)
            centers = minimum + width * np.arange(count)
            distance = np.abs(values[None, :] - centers[:, None])
        windows.append(
            np.where(distance < width, np.cos(0.5 * math.pi * distance / width) ** 2, 0.0)
        )
    weights = windows[0]
    labels = [(index,) for index in range(grid[0])]
    for axis_windows in windows[1:]:
        weights = (weights[:, None, :] * axis_windows[None, :, :]).reshape(-1, weights.shape[1])
        labels = [label + (index,) for label in labels for index in range(axis_windows.shape[0])]
    region_mass = weights @ mass
    keep = region_mass >= min_mass_fraction
    if not keep.any():
        raise ValueError("every region falls below regions.min_mass_fraction")
    measures = weights[keep] * mass[None, :] / region_mass[keep, None]
    names = tuple(
        "r" + "_".join(str(item) for item in label) for label, flag in zip(labels, keep) if flag
    )
    return (
        measures,
        names,
        {
            "configured_regions": int(weights.shape[0]),
            "retained_regions": int(keep.sum()),
            "dropped_regions": [
                "r" + "_".join(str(item) for item in label)
                for label, flag in zip(labels, keep)
                if not flag
            ],
            "region_mass": region_mass[keep].tolist(),
        },
    )
