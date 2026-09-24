"""Exact finite-diagram assignment with optional spatial creator costs.

This is a repository-local L1 matching experiment inspired by SATLoss, not a
reproduction of its squared, spatially weighted segmentation objective.
All finite bars are included. Essential births keep the v3 sorted comparison:
a representative creator of a noncontractible torus cycle is not a canonical
spatial location for that cycle.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .persistence import Diagram


def spatial_diagram_distance(
    left: Diagram,
    right: Diagram,
    *,
    periodic: bool,
    spatial_weight: float = 1.0,
    spatial_mode: str = "additive",
    essential_weight: float = 0.1,
    normalization: float = 1.0,
    max_assignment_size: int = 4096,
) -> torch.Tensor:
    """L1 Wasserstein-1 assignment, including deletion/insertion via the diagonal.

    Finite pair cost is ``L1(bd_i, bd_j) + lambda * distance(creator_i, creator_j)``
    (additive), or ``L1(bd_i, bd_j) * (1 + lambda * distance(...))``
    (multiplicative). Spatial distances are Euclidean distances in domain-axis
    fractions, divided by sqrt(2); periodic axes use shortest wrapped distances.
    Diagonal cost is persistence, with no spatial charge. lambda=0 is the exact
    nonspatial assignment control, NOT the original sliced approximation.

    Assignment and creator coordinates are discrete and detached. Backward uses
    live critical values, so an additive spatial cost alone supplies no location
    gradient while the pairing is fixed. Dense reconstruction supervision is
    still required. Multiplicative matching is blind to an exactly translated
    copy with identical birth/death values.
    """
    if spatial_mode not in {"additive", "multiplicative"}:
        raise ValueError("spatial_mode must be additive or multiplicative")
    if (
        not math.isfinite(spatial_weight)
        or spatial_weight < 0
        or not math.isfinite(essential_weight)
        or essential_weight < 0
        or not math.isfinite(normalization)
        or normalization <= 0
    ):
        raise ValueError("invalid spatial persistence weights or normalization")
    if int(max_assignment_size) != max_assignment_size or max_assignment_size < 1:
        raise ValueError("max_assignment_size must be a positive integer")
    if left.essential.numel() != right.essential.numel():
        raise ValueError("essential class counts differ: check domain/boundary conventions")
    x, y = left.finite, right.finite
    n, m = len(x), len(y)
    if n + m > max_assignment_size:
        raise ValueError(
            f"assignment has {n + m} finite bars; exceeds max_assignment_size="
            f"{max_assignment_size}. Increase the explicit memory limit; "
            "no bars are silently truncated."
        )
    # Build the detached dense assignment only on CPU; the differentiable part
    # below gathers O(n+m) selected values, not the full pair-cost matrix.
    a, b = x.detach().double().cpu().numpy(), y.detach().double().cpu().numpy()
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise FloatingPointError("diagram contains nonfinite values")
    spatial = np.zeros((n, m), dtype=np.float64)
    if spatial_weight and n and m:
        if left.finite_locations is None or right.finite_locations is None:
            raise ValueError("spatial matching requires finite creator locations")
        u = left.finite_locations.detach().double().cpu().numpy()
        v = right.finite_locations.detach().double().cpu().numpy()
        if u.shape != (n, 2) or v.shape != (m, 2):
            raise ValueError("creator locations must have shape [finite_bars, 2]")
        if not np.isfinite(u).all() or not np.isfinite(v).all():
            raise FloatingPointError("creator locations must be finite")
        delta = np.abs(u[:, None] - v[None])
        if periodic:
            delta %= 1.0
            delta = np.minimum(delta, 1.0 - delta)
        spatial = spatial_weight * np.linalg.norm(delta, axis=-1) / math.sqrt(2)
    pair_cost = np.abs(a[:, None] - b[None]).sum(axis=-1)
    pair_cost = pair_cost + spatial if spatial_mode == "additive" else pair_cost * (1 + spatial)
    # Start with every bar on the diagonal, then choose disjoint improvements.
    # This n*m rectangular problem has the same optimum as (n+m)^2 augmentation.
    diagonal_a, diagonal_b = np.abs(a[:, 1] - a[:, 0]), np.abs(b[:, 1] - b[:, 0])
    reduced = np.minimum(pair_cost - diagonal_a[:, None] - diagonal_b[None, :], 0.0)
    rows, cols = linear_sum_assignment(reduced)
    improved = reduced[rows, cols] < 0.0
    rows, cols = rows[improved], cols[improved]
    pi = torch.as_tensor(rows, device=x.device)
    pj = torch.as_tensor(cols, device=y.device)
    live_pairs = (x[pi] - y[pj]).abs().sum(-1)
    location_cost = x.new_tensor(spatial[rows, cols])
    live_pairs = (
        live_pairs + location_cost
        if spatial_mode == "additive"
        else live_pairs * (1 + location_cost)
    )
    unmatched_a, unmatched_b = np.ones(n, dtype=bool), np.ones(m, dtype=bool)
    unmatched_a[rows], unmatched_b[cols] = False, False
    deleted = torch.as_tensor(np.flatnonzero(unmatched_a), device=x.device)
    inserted = torch.as_tensor(np.flatnonzero(unmatched_b), device=y.device)
    finite = (
        live_pairs.sum()
        + (x[deleted, 1] - x[deleted, 0]).abs().sum()
        + (y[inserted, 1] - y[inserted, 0]).abs().sum()
    ) / normalization
    essential = (left.essential.sort().values - right.essential.sort().values).abs().sum()
    return finite + essential_weight * essential
