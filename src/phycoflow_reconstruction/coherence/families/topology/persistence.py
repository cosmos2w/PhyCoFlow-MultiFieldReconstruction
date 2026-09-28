"""Exact cubical pairings with autograd through their critical vertex values.

GUDHI computes combinatorics on the CPU. Only integer indices return to the
input device; no straight-through Betti estimator is used. Gradients are valid
on each stratum with fixed pairings (ties are nonsmooth). Arrays use GUDHI's
Fortran vertex order, including for periodic complexes.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
import torch

_POOL = None
_POOL_PID = None
_REFERENCE_PAIRS = OrderedDict()
_REFERENCE_LOCK = Lock()


def _pairing_pool():
    global _POOL, _POOL_PID
    workers = max(1, int(os.environ.get("PHYCOFLOW_TOPOLOGY_WORKERS", "1")))
    if workers == 1:
        return None
    if _POOL is None or _POOL_PID != os.getpid() or _POOL._max_workers != workers:
        if _POOL is not None and _POOL_PID == os.getpid():
            _POOL.shutdown()
        _POOL = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="periodic-persistence")
        _POOL_PID = os.getpid()
    return _POOL


def _cpu_pairing(values, periodic, cacheable):
    import gudhi

    key = None
    if cacheable:
        key = (values.shape, periodic, hashlib.blake2b(values.tobytes(), digest_size=16).digest())
        with _REFERENCE_LOCK:
            if key in _REFERENCE_PAIRS:
                _REFERENCE_PAIRS.move_to_end(key)
                return _REFERENCE_PAIRS[key]
    complex_ = gudhi.PeriodicCubicalComplex(
        vertices=values, periodic_dimensions=[periodic, periodic]
    )
    complex_.compute_persistence(homology_coeff_field=2, min_persistence=0)
    pairs, essential = complex_.vertices_of_persistence_pairs()
    result = (
        [np.array(p, dtype=np.int64, copy=True) for p in pairs],
        [np.array(p, dtype=np.int64, copy=True) for p in essential],
    )
    if key is not None:
        with _REFERENCE_LOCK:
            _REFERENCE_PAIRS[key] = result
            capacity = max(0, int(os.environ.get("PHYCOFLOW_TOPOLOGY_REFERENCE_CACHE", "4096")))
            while len(_REFERENCE_PAIRS) > capacity:
                _REFERENCE_PAIRS.popitem(last=False)
    return result


@dataclass
class Diagram:
    finite: torch.Tensor
    essential: torch.Tensor
    # Creator coordinates in fractions of each domain axis, in (x, y) order.
    # Optional because the original sliced loss does not use spatial locations.
    finite_locations: torch.Tensor | None = None


def persistence_source() -> dict:
    """Bind saved family artifacts to the implemented definition and backend."""
    import gudhi

    names = (
        "persistence.py",
        "persistence_objective.py",
        "family.py",
        "geometry.py",
        "betti_curves.py",
        "spatial.py",
        "spatial_persistence.py",
    )
    return {
        "implementation": "repository-local cubical persistence v3",
        "pairing_backend": "GUDHI",
        "gudhi_version": gudhi.__version__,
        "boundary": "vertex lower-star",
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in names
        },
    }


def cubical_diagrams(
    fields: torch.Tensor,
    *,
    periodic: bool,
    dimensions: tuple[int, ...] = (0, 1),
    locations: bool = False,
    cacheable: bool = False,
) -> list[dict[int, Diagram]]:
    """Compute all positive-persistence pairs for [N,H,W], without truncation."""
    try:
        importlib.import_module("gudhi")
    except ImportError as error:
        raise ImportError("cubical_persistence requires pip install '.[topology]'") from error
    if fields.ndim != 3 or min(fields.shape[-2:]) < 2:
        raise ValueError("cubical persistence requires [N,H,W], H,W >= 2")
    # One device synchronization and transfer for the complete bank of slices.
    arrays = fields.detach().to(device="cpu", dtype=torch.float64).numpy()
    if not np.isfinite(arrays).all():
        raise FloatingPointError("cubical persistence input must be finite")
    pool = _pairing_pool()
    reduce = lambda values: _cpu_pairing(values, periodic, cacheable)
    reductions = list(map(reduce, arrays) if pool is None else pool.map(reduce, arrays))
    result = [{} for _ in reductions]
    height, width = fields.shape[-2:]
    vertices = height * width
    flat = fields.transpose(-2, -1).reshape(-1)  # Each grid remains Fortran ordered.
    for dim in dimensions:
        pair_rows, birth_rows, positions = [], [], []
        for row, (pairs, essential) in enumerate(reductions):
            pair_ids = pairs[dim] if dim < len(pairs) else np.empty((0, 2), dtype=np.int64)
            birth_ids = essential[dim] if dim < len(essential) else np.empty(0, dtype=np.int64)
            pair_rows.append(pair_ids + row * vertices)
            birth_rows.append(birth_ids + row * vertices)
            if locations:
                creators = pair_ids[:, 0]
                xy = np.stack((creators // height, creators % height), axis=-1)
                extent = (width, height) if periodic else (width - 1, height - 1)
                positions.append(xy / np.asarray(extent))
        # Upload one packed index array per degree/side, instead of per diagram.
        pair_values = flat[torch.as_tensor(np.concatenate(pair_rows), device=fields.device)]
        birth_values = flat[torch.as_tensor(np.concatenate(birth_rows), device=fields.device)]
        pair_chunks = pair_values.split([len(p) for p in pair_rows])
        birth_chunks = birth_values.split([len(p) for p in birth_rows])
        location_chunks = (
            torch.as_tensor(
                np.concatenate(positions), device=fields.device, dtype=fields.dtype
            ).split([len(p) for p in pair_rows])
            if locations
            else [None] * len(result)
        )
        for row, (pairs, births, xy) in enumerate(zip(pair_chunks, birth_chunks, location_chunks)):
            result[row][dim] = Diagram(pairs, births, xy)
    return result


def sliced_diagram_distance(
    left: Diagram,
    right: Diagram,
    *,
    projections: int = 32,
    essential_weight: float = 0.1,
    normalization: float = 1.0,
) -> torch.Tensor:
    """Finite-angle SW1 with cross-diagonal augmentation, plus essential births.

    This is a quadrature approximation of Carriere/Cuturi/Oudot (ICML 2017),
    not an exact Wasserstein assignment. Sum over points, average over angles;
    normalization must be fixed independently of prediction cardinality.
    Essential classes cannot match the diagonal; compare their sorted births.
    """
    x, y = left.finite, right.finite
    if projections < 2 or normalization <= 0:
        raise ValueError("projections >= 2 and normalization > 0 required")
    if left.essential.numel() != right.essential.numel():
        raise ValueError("essential class counts differ: check domain/boundary conventions")
    theta = (torch.arange(projections, device=x.device, dtype=x.dtype) + 0.5) * (
        math.pi / projections
    )
    directions = torch.stack((theta.cos(), theta.sin()))
    x_diag = x.mean(-1, keepdim=True).expand(-1, 2)
    y_diag = y.mean(-1, keepdim=True).expand(-1, 2)
    a = (torch.cat((x, y_diag)) @ directions).sort(dim=0).values
    b = (torch.cat((y, x_diag)) @ directions).sort(dim=0).values
    finite = (a - b).abs().sum(dim=0).mean() / normalization
    essential = (left.essential.sort().values - right.essential.sort().values).abs().sum()
    return finite + essential_weight * essential


def sliced_diagram_distances(
    left, right, *, projections=32, essential_weight=0.1, normalization=1.0
):
    """Exact batching of the existing finite-angle distance, with bounded padding.

    The reference scalar function above remains the value/gradient oracle. Padding
    is excluded before subtraction, so empty diagrams never produce inf-inf.
    """
    if len(left) != len(right) or not left or projections < 2 or normalization <= 0:
        raise ValueError("invalid batched diagram inputs")
    if os.environ.get("PHYCOFLOW_TOPOLOGY_BATCHED", "1") == "0":
        return torch.stack(
            [
                sliced_diagram_distance(
                    a,
                    b,
                    projections=projections,
                    essential_weight=essential_weight,
                    normalization=normalization,
                )
                for a, b in zip(left, right)
            ]
        )
    device, dtype = left[0].finite.device, left[0].finite.dtype
    theta = (torch.arange(projections, device=device, dtype=dtype) + 0.5) * (math.pi / projections)
    directions = torch.stack((theta.cos(), theta.sin()))
    lengths = [len(a.finite) + len(b.finite) for a, b in zip(left, right)]
    order = sorted(range(len(left)), key=lambda i: lengths[i])
    result = [None] * len(left)
    limit = max(1, int(os.environ.get("PHYCOFLOW_TOPOLOGY_DISTANCE_ELEMENTS", "2000000")))
    start = 0
    while start < len(order):
        stop = start + 1
        while (
            stop < len(order)
            and (stop - start + 1) * max(lengths[order[stop]], 1) * projections <= limit
        ):
            stop += 1
        ids = order[start:stop]
        start = stop
        arows, brows, essentials = [], [], []
        for i in ids:
            a, b = left[i], right[i]
            if a.essential.numel() != b.essential.numel():
                raise ValueError("essential class counts differ: check domain/boundary conventions")
            arows.append(torch.cat((a.finite, b.finite.mean(-1, keepdim=True).expand(-1, 2))))
            brows.append(torch.cat((b.finite, a.finite.mean(-1, keepdim=True).expand(-1, 2))))
            essentials.append((a.essential.sort().values - b.essential.sort().values).abs().sum())
        a = torch.nn.utils.rnn.pad_sequence(arows, batch_first=True) @ directions
        b = torch.nn.utils.rnn.pad_sequence(brows, batch_first=True) @ directions
        valid = (
            torch.arange(a.shape[1], device=device)[None, :, None]
            < torch.tensor([lengths[i] for i in ids], device=device)[:, None, None]
        )
        a = a.masked_fill(~valid, float("inf")).sort(dim=1).values
        b = b.masked_fill(~valid, float("inf")).sort(dim=1).values
        delta = a.masked_fill(~valid, 0.0) - b.masked_fill(~valid, 0.0)
        costs = delta.abs().sum(1).mean(1) / normalization + essential_weight * torch.stack(
            essentials
        )
        for i, cost in zip(ids, costs.unbind()):
            result[i] = cost
    return torch.stack(result)
