"""Deterministic graph Fourier basis construction for fixed point sets.

Adapted from Joseph Castro's MIT-licensed
PhyCoFlowModel-Cross-Spectral-Coherence (revision add1b1a6422c). The adapter
adds connectivity checks, deterministic eigenvector signs, and geometry
fingerprints required by the reconstruction run contract.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

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
    coordinate_sha256: str
    sigma: float
    k_neighbors: int


def coordinate_digest(coordinates: torch.Tensor) -> str:
    array = coordinates.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _deterministic_signs(eigenvectors: np.ndarray) -> np.ndarray:
    result = eigenvectors.copy()
    pivots = np.abs(result).argmax(axis=0)
    signs = np.sign(result[pivots, np.arange(result.shape[1])])
    signs[signs == 0] = 1.0
    return result * signs[None, :]


def build_graph_basis(
    coordinates: torch.Tensor,
    *,
    k_neighbors: int,
    sigma: float | None,
    num_modes: int,
    band_names: tuple[str, ...],
    exclude_zero: bool = True,
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
    if exclude_zero:
        eigenvalues = eigenvalues[1:]
        eigenvectors = eigenvectors[:, 1:]
    eigenvalues = eigenvalues[:num_modes]
    eigenvectors = _deterministic_signs(eigenvectors[:, :num_modes])
    if eigenvalues.size < len(band_names):
        raise ValueError("retained graph modes must be at least the number of frequency bands")
    band_ids = np.empty(eigenvalues.size, dtype=np.int64)
    for band_id, mode_ids in enumerate(
        np.array_split(np.arange(eigenvalues.size), len(band_names))
    ):
        band_ids[mode_ids] = band_id
    return GraphBasis(
        eigenvalues=torch.from_numpy(eigenvalues).float(),
        eigenvectors=torch.from_numpy(eigenvectors).float(),
        band_ids=torch.from_numpy(band_ids),
        band_names=band_names,
        coordinate_sha256=coordinate_digest(coordinates),
        sigma=resolved_sigma,
        k_neighbors=neighbors,
    )
