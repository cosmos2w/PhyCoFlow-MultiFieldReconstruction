"""Explicit physical measures for coherence queries.

Geometry providers choose the physical metric and sampling design; these
helpers do not infer a physical measure from normalized query coordinates.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch


def rectilinear_nodal_weights(
    coordinates: torch.Tensor,
    logical_shape: Sequence[int],
) -> torch.Tensor:
    """Return nonperiodic physical nodal masses on a full rectilinear grid.

    Trapezoidal masses integrate multilinear nodal fields over the rectangle
    bounded by the outermost coordinates. Coordinate columns may be ordered
    differently from logical axes, and either axis direction is supported.
    Masked, periodic and unstructured domains need their own case provider.
    The returned masses are unnormalized and use the coordinate-product unit.
    """
    coords = torch.as_tensor(coordinates).detach()
    shape = tuple(int(size) for size in logical_shape)
    if not coords.is_floating_point() or coords.ndim != 2:
        raise ValueError("rectilinear coordinates must be a floating tensor [N,D]")
    if len(shape) != coords.shape[1] or any(size < 2 for size in shape):
        raise ValueError("logical_shape must have one size >= 2 per coordinate dimension")
    if math.prod(shape) != coords.shape[0] or not torch.isfinite(coords).all():
        raise ValueError("rectilinear coordinates must be finite and match logical_shape")
    grid = coords.reshape(*shape, coords.shape[1])
    used_axes: set[int] = set()
    factors: dict[int, torch.Tensor] = {}
    for component in range(coords.shape[1]):
        plane = grid[..., component]
        found: list[tuple[int, torch.Tensor]] = []
        for axis in range(len(shape)):
            index = tuple(slice(None) if position == axis else 0 for position in range(len(shape)))
            values = plane[index]
            broadcast_shape = [1] * len(shape)
            broadcast_shape[axis] = shape[axis]
            if torch.allclose(plane, values.reshape(broadcast_shape).expand_as(plane)):
                delta = values[1:] - values[:-1]
                if bool(torch.all(delta > 0)) or bool(torch.all(delta < 0)):
                    found.append((axis, delta.abs()))
        if len(found) != 1 or found[0][0] in used_axes:
            raise ValueError("coordinates must form a complete strictly monotone rectilinear grid")
        axis, widths = found[0]
        used_axes.add(axis)
        # The corresponding hat-function integral: half each adjacent cell.
        factors[axis] = torch.cat((widths[:1] / 2, (widths[:-1] + widths[1:]) / 2, widths[-1:] / 2))
    weights = torch.ones(shape, device=coords.device, dtype=coords.dtype)
    for axis, values in factors.items():
        broadcast_shape = [1] * len(shape)
        broadcast_shape[axis] = shape[axis]
        weights = weights * values.reshape(broadcast_shape)
    return weights.reshape(-1)


def subset_inclusion_weights(
    native_weights: torch.Tensor,
    indices: torch.Tensor,
    inclusion_probability: float | torch.Tensor,
) -> torch.Tensor:
    """Select native masses and apply known point inclusion probabilities.

    Dividing by inclusion probability yields a Horvitz-Thompson mass estimate;
    after normalization by its random total, occupation is a self-normalized
    sparse approximation, not exact native quadrature. Sampling is without
    replacement; repeated point IDs are rejected rather than turned into a mesh.
    """
    native = torch.as_tensor(native_weights).detach()
    if native.ndim != 1 or not native.is_floating_point():
        raise ValueError("native_weights must be a floating tensor [N]")
    if not torch.isfinite(native).all() or torch.any(native < 0) or not bool(native.sum() > 0):
        raise ValueError("native_weights must be finite, nonnegative and nonzero")
    index = torch.as_tensor(indices, device=native.device)
    if index.dtype not in (torch.int32, torch.int64) or index.ndim not in (1, 2):
        raise ValueError("indices must be integer [Q] or [B,Q]")
    if index.shape[-1] == 0 or torch.any(index < 0) or torch.any(index >= native.numel()):
        raise ValueError("subset indices must be nonempty and lie within the native support")
    sorted_index = index.sort(dim=-1).values
    if index.shape[-1] > 1 and bool(torch.any(sorted_index[..., 1:] == sorted_index[..., :-1])):
        raise ValueError("subset indices must be unique within each sample")
    probability = torch.as_tensor(
        inclusion_probability, device=native.device, dtype=native.dtype
    ).detach()
    try:
        probability = torch.broadcast_to(probability, index.shape)
    except RuntimeError as exc:
        raise ValueError("inclusion_probability must broadcast to the selected points") from exc
    if (
        not torch.isfinite(probability).all()
        or torch.any(probability <= 0)
        or torch.any(probability > 1)
    ):
        raise ValueError("inclusion probabilities must be finite and lie in (0,1]")
    return native[index.long()] / probability


def normalize_measure_weights(
    weights: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    batch_size: int,
    point_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Validate and normalize fixed [N] or per-sample [B,N] masses."""
    value = torch.as_tensor(weights, device=device, dtype=dtype).detach()
    if value.shape == (point_count,):
        value = value.unsqueeze(0).expand(batch_size, -1)
    if value.shape != (batch_size, point_count):
        raise ValueError("measure weights must have shape [N] or [B,N] aligned with their endpoint")
    if not torch.isfinite(value).all() or torch.any(value < 0):
        raise ValueError("measure weights must be finite and nonnegative")
    if mask is not None:
        valid = torch.as_tensor(mask, device=device)
        if valid.dtype != torch.bool:
            raise TypeError("measure mask must be boolean")
        if valid.shape == (point_count,):
            valid = valid.unsqueeze(0).expand(batch_size, -1)
        if valid.shape != value.shape:
            raise ValueError("measure mask must have shape [N] or [B,N] aligned with its endpoint")
        value = value * valid
    totals = value.sum(dim=1, keepdim=True)
    if not torch.isfinite(totals).all() or torch.any(totals <= 0):
        raise ValueError(
            "each endpoint measure must have positive finite mass on its valid support"
        )
    return value / totals


def measure_fingerprint(
    weights: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    point_indices: torch.Tensor | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an inspectable SHA-256 receipt for ordered weights and support.

    Include physical coordinate units, provider, sampling policy and native
    support identity in `metadata`; all must be JSON serializable. Hashes are
    invariant to original numeric dtype and device, but not to point ordering.
    """
    mass = torch.as_tensor(weights).detach().cpu()
    if (
        mass.ndim != 1
        or not torch.isfinite(mass).all()
        or torch.any(mass < 0)
        or not bool(mass.sum() > 0)
    ):
        raise ValueError("fingerprinted weights must be finite nonnegative [N] with positive total")
    if mask is None:
        valid = torch.ones_like(mass, dtype=torch.bool)
    else:
        valid = torch.as_tensor(mask).detach().cpu()
        if valid.dtype != torch.bool or valid.shape != mass.shape:
            raise ValueError("fingerprint mask must be boolean [N]")
    if not bool((mass * valid).sum() > 0):
        raise ValueError("fingerprint mask leaves no positive mass")
    if point_indices is None:
        indices = torch.arange(mass.numel(), dtype=torch.int64)
    else:
        indices = torch.as_tensor(point_indices).detach().cpu()
        if indices.dtype not in (torch.int32, torch.int64) or indices.shape != mass.shape:
            raise ValueError("fingerprint point indices must be integer [N]")
    info = dict(metadata or {})
    encoded = json.dumps(info, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    digest = hashlib.sha256(b"phycoflow-measure-v1\0" + encoded)
    digest.update(np.asarray(mass, dtype="<f8").tobytes())
    digest.update(np.asarray(valid, dtype=np.uint8).tobytes())
    digest.update(np.asarray(indices, dtype="<i8").tobytes())
    return {
        "version": "physical_measure_v1",
        "sha256": digest.hexdigest(),
        "point_count": int(mass.numel()),
        "valid_point_count": int(valid.sum()),
        "total_mass": float((mass * valid).sum()),
        "metadata": info,
    }
