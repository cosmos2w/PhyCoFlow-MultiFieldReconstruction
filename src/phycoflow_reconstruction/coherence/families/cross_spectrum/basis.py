"""Deterministic graph Fourier basis construction for fixed point sets.

Adapted from Joseph Castro's MIT-licensed
PhyCoFlowModel-Cross-Spectral-Coherence (revision add1b1a6422c). The adapter
adds connectivity checks, deterministic eigenvector signs, and geometry
fingerprints required by the reconstruction run contract.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import isclose, isfinite

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class GraphBasis:
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    band_ids: torch.Tensor
    band_names: tuple[str, ...]
    band_intervals: tuple[tuple[str, float, float], ...] | None
    coordinate_sha256: str
    sigma: float
    k_neighbors: int
    zero_mode_eigenvalue: float
    first_retained_eigengap: float | None


def coordinate_digest(coordinates: torch.Tensor) -> str:
    array = coordinates.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def basis_digest(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    band_ids: torch.Tensor,
    *,
    band_names: tuple[str, ...],
    band_intervals: tuple[tuple[str, float, float], ...] | None,
    geometry_sha256: str,
    exclude_zero: bool,
) -> str:
    """Hash the realized basis and its spectral partition for artifact checks."""
    digest = hashlib.sha256()
    digest.update(geometry_sha256.encode("ascii"))
    digest.update(str(bool(exclude_zero)).encode("ascii"))
    digest.update(repr(band_names).encode("utf-8"))
    digest.update(repr(band_intervals).encode("utf-8"))
    for name, tensor in (
        ("eigenvalues", eigenvalues),
        ("eigenvectors", eigenvectors),
        ("band_ids", band_ids),
    ):
        array = tensor.detach().to(device="cpu").contiguous().numpy()
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _deterministic_signs(eigenvectors: np.ndarray) -> np.ndarray:
    result = eigenvectors.copy()
    pivots = np.abs(result).argmax(axis=0)
    signs = np.sign(result[pivots, np.arange(result.shape[1])])
    signs[signs == 0] = 1.0
    return result * signs[None, :]


def interval_band_ids(
    eigenvalues: np.ndarray,
    band_intervals: tuple[tuple[str, float, float], ...],
    degeneracy_tolerance: float,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Assign retained eigenvalues to explicit half-open spectral intervals.

    All intervals except the final one are ``[lower, upper)``; the final
    interval also includes its upper endpoint.  Near-degenerate eigenvalue
    clusters are kept intact: a cluster touching two bands is rejected.
    """
    if not band_intervals:
        raise ValueError("second-order covariance blocks require graph.band_intervals")
    if not isfinite(degeneracy_tolerance) or degeneracy_tolerance < 0:
        raise ValueError("graph.degeneracy_tolerance must be finite and non-negative")
    names = tuple(item[0] for item in band_intervals)
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("graph.band_intervals names must be non-empty and unique")
    if eigenvalues.ndim != 1 or not np.isfinite(eigenvalues).all():
        raise ValueError("retained eigenvalues must be a finite one-dimensional array")
    if eigenvalues.size > 1 and np.any(np.diff(eigenvalues) < 0):
        raise ValueError("retained eigenvalues must be sorted")
    for index, (name, lower, upper) in enumerate(band_intervals):
        if not isfinite(lower) or not isfinite(upper) or lower >= upper:
            raise ValueError("graph.band_intervals bounds must be finite and increasing")
        if index and not isclose(band_intervals[index - 1][2], lower, abs_tol=1.0e-12):
            raise ValueError("graph.band_intervals must be ordered, contiguous, and non-overlapping")
    if eigenvalues.size == 0:
        raise ValueError("the graph basis retained no modes")
    ids = np.full(eigenvalues.shape, -1, dtype=np.int64)
    for band_id, (_, lower, upper) in enumerate(band_intervals):
        if band_id == len(band_intervals) - 1:
            mask = (eigenvalues >= lower) & (eigenvalues <= upper)
        else:
            mask = (eigenvalues >= lower) & (eigenvalues < upper)
        ids[mask] = band_id
    if np.any(ids < 0):
        raise ValueError(
            "graph.band_intervals must cover every retained eigenvalue without gaps"
        )
    if np.any(ids >= len(names)):
        raise ValueError("graph.band_intervals assign an invalid band id")
    empty = [names[index] for index in range(len(names)) if not np.any(ids == index)]
    if empty:
        raise ValueError(f"every graph spectral band must contain a retained mode: {empty}")
    if eigenvalues.size > 1 and degeneracy_tolerance > 0:
        for index, gap in enumerate(np.diff(eigenvalues)):
            if gap <= degeneracy_tolerance and ids[index] != ids[index + 1]:
                raise ValueError(
                    "graph.band_intervals split a near-degenerate eigenvalue cluster; "
                    "move the boundary or increase graph.degeneracy_tolerance"
                )
    return ids, names


def build_graph_basis(
    coordinates: torch.Tensor,
    *,
    k_neighbors: int,
    sigma: float | None,
    num_modes: int,
    band_names: tuple[str, ...],
    exclude_zero: bool = True,
    band_intervals: tuple[tuple[str, float, float], ...] | None = None,
    degeneracy_tolerance: float = 1.0e-6,
) -> GraphBasis:
    """Build the symmetric-normalized kNN Laplacian eigensystem."""
    if coordinates.ndim != 2 or coordinates.shape[0] < 3:
        raise ValueError("graph coordinates must have shape [N,D] with N>=3")
    if not torch.isfinite(coordinates).all():
        raise FloatingPointError("graph coordinates contain non-finite values")
    if num_modes < 1 or not band_names:
        raise ValueError("num_modes and band_names must be non-empty")
    coords = coordinates.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()
    count = coords.shape[0]
    neighbors = min(max(int(k_neighbors), 1), count - 1)
    distances, indices = cKDTree(coords).query(coords, k=neighbors + 1)
    distances = distances[:, 1:].reshape(-1)
    columns = indices[:, 1:].reshape(-1)
    rows = np.repeat(np.arange(count), neighbors)
    positive = distances[distances > 0]
    resolved_sigma = float(np.median(positive)) if sigma is None else float(sigma)
    if not np.isfinite(resolved_sigma) or resolved_sigma <= 0:
        raise ValueError("graph sigma must be positive (coordinates may contain duplicates)")
    weights = np.exp(-(distances**2) / (2.0 * resolved_sigma**2 + 1e-12))
    adjacency = sparse.coo_matrix((weights, (rows, columns)), shape=(count, count)).tocsr()
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    components, _ = connected_components(adjacency, directed=False)
    if components != 1:
        raise ValueError(
            f"kNN coherence graph is disconnected ({components} components); "
            "increase graph.k_neighbors"
        )
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse_sqrt = sparse.diags(1.0 / np.sqrt(np.maximum(degrees, 1e-12)))
    laplacian = sparse.eye(count, format="csr") - inverse_sqrt @ adjacency @ inverse_sqrt

    requested = min(int(num_modes) + int(exclude_zero), count)
    if requested >= count:
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian.toarray())
    else:
        eigenvalues, eigenvectors = eigsh(
            laplacian,
            k=requested,
            which="SM",
            v0=np.ones(count, dtype=np.float64),
        )
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    zero_mode_eigenvalue = float(eigenvalues[0])
    first_retained_eigengap = (
        float(eigenvalues[1] - eigenvalues[0]) if exclude_zero and eigenvalues.size > 1 else None
    )
    if exclude_zero:
        eigenvalues = eigenvalues[1:]
        eigenvectors = eigenvectors[:, 1:]
    eigenvalues = eigenvalues[:num_modes]
    eigenvectors = _deterministic_signs(eigenvectors[:, :num_modes])
    if band_intervals is None:
        if eigenvalues.size < len(band_names):
            raise ValueError("retained graph modes must be at least the number of frequency bands")
        band_ids = np.empty(eigenvalues.size, dtype=np.int64)
        for band_id, mode_ids in enumerate(
            np.array_split(np.arange(eigenvalues.size), len(band_names))
        ):
            band_ids[mode_ids] = band_id
        resolved_band_names = band_names
    else:
        band_ids, resolved_band_names = interval_band_ids(
            eigenvalues, band_intervals, degeneracy_tolerance
        )
    return GraphBasis(
        eigenvalues=torch.from_numpy(eigenvalues).float(),
        eigenvectors=torch.from_numpy(eigenvectors).float(),
        band_ids=torch.from_numpy(band_ids),
        band_names=resolved_band_names,
        band_intervals=band_intervals,
        coordinate_sha256=coordinate_digest(coordinates),
        sigma=resolved_sigma,
        k_neighbors=neighbors,
        zero_mode_eigenvalue=zero_mode_eigenvalue,
        first_retained_eigengap=first_retained_eigengap,
    )
