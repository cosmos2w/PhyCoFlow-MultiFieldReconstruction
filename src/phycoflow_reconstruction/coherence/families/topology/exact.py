"""Held-out H0/H1 counts and additive errors on the raster's cubical complex.

Evaluation uses NumPy's float64 quantile oracle, including repeated thresholds.
SciPy labels the four-neighbor foreground; periodic seam unions identify the
torus exactly. No persistence padding or straight-through arithmetic is used.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import label


def reference_levels(reference: torch.Tensor, quantiles: tuple[float, ...]) -> torch.Tensor:
    """Return CPU float64 [B,T] thresholds without rounding them to field dtype."""
    values = reference.detach().cpu().numpy().reshape(reference.shape[0], -1)
    return torch.from_numpy(np.quantile(values, quantiles, axis=1).T.copy())


def _counts(mask: np.ndarray, periodic: bool) -> tuple[int, int]:
    labels, components = label(mask)  # Default connectivity is four-neighbor in 2-D.
    if periodic:
        parent = np.arange(components + 1)

        def root(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        # Only open-grid components joined across a seam need to be merged.
        for left, right in zip(
            np.concatenate((labels[0], labels[:, 0])),
            np.concatenate((labels[-1], labels[:, -1])),
        ):
            if left and right:
                first, second = root(left), root(right)
                if first != second:
                    parent[second] = first
                    components -= 1
        right = np.roll(mask, -1, axis=1)
        down = np.roll(mask, -1, axis=0)
        diagonal = np.roll(down, -1, axis=1)
        edges = (mask & right).sum() + (mask & down).sum()
        faces = (mask & right & down & diagonal).sum()
        b2 = int(mask.all())
    else:
        edges = (mask[:, :-1] & mask[:, 1:]).sum() + (mask[:-1] & mask[1:]).sum()
        faces = (mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]).sum()
        b2 = 0
    chi = int(mask.sum() - edges + faces)
    return int(components), int(components - chi + b2)


@torch.no_grad()
def exact_betti_curves(
    fields: torch.Tensor, levels: torch.Tensor, *, periodic: bool
) -> torch.Tensor:
    """Return CPU integer [B,2,T] superlevel counts, with inclusive birth ties."""
    if fields.ndim != 3 or min(fields.shape) < 1:
        raise ValueError("exact topology fields must have nonempty [B,H,W] shape")
    values = fields.detach().double().cpu().numpy()
    thresholds = levels.detach().double().cpu().numpy()
    if thresholds.ndim == 1:
        thresholds = np.broadcast_to(thresholds, (len(values), len(thresholds)))
    if thresholds.ndim != 2 or thresholds.shape[0] != len(values) or not thresholds.shape[1]:
        raise ValueError("exact topology levels must have shape [T] or [B,T]")
    if not np.isfinite(values).all() or not np.isfinite(thresholds).all():
        raise ValueError("exact topology fields and levels must be finite")
    counts = np.empty((len(values), 2, thresholds.shape[1]), dtype=np.int64)
    for batch, (field, row) in enumerate(zip(values, thresholds)):
        for index, threshold in enumerate(row):
            counts[batch, :, index] = _counts(field >= threshold, periodic)
    return torch.from_numpy(counts)


@torch.no_grad()
def curve_statistics(
    pairs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], *, periodic: bool
) -> dict[str, torch.Tensor]:
    """Pool error/reference masses over fields, directions and thresholds.

    The slice masses [B,S,2] also allow full-bank mutual line guards and pooling
    across batches. Averaging individually normalized slice errors is different.
    """
    predictions, references, levels = zip(*pairs)
    batch_size = predictions[0].shape[0]
    pred = torch.cat(predictions).detach()
    ref = torch.cat(references).detach()
    thresholds = torch.cat([row.cpu() for row in levels])
    # Transfer generated/reference fields together rather than per sample or line.
    counts = exact_betti_curves(torch.cat((pred, ref)), thresholds.repeat(2, 1), periodic=periodic)
    pc, rc = counts.chunk(2)
    error = (pc - rc).abs().sum(-1).reshape(len(pairs), batch_size, 2).transpose(0, 1)
    mass = rc.abs().clamp_min(1).sum(-1).reshape(len(pairs), batch_size, 2).transpose(0, 1)
    return {
        "slice_error_mass": error,
        "slice_reference_mass": mass,
        "error_mass": error.sum(1),
        "reference_mass": mass.sum(1),
    }
