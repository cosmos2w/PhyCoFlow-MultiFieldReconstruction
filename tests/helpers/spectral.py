"""Deterministic synthetic geometries and reference states for spectral tests."""

from __future__ import annotations

import math

import numpy as np
import torch

from phycoflow_reconstruction.coherence.families.multiscale_spectrum import (
    MultiscaleSpectrumFamily,
)
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer


def _grid(size: int, *, jitter: float = 0.0, seed: int = 0) -> torch.Tensor:
    axis = np.linspace(0.0, 1.0, size)
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    if jitter:
        points = points + np.random.default_rng(seed).uniform(-jitter, jitter, points.shape) / size
    return torch.from_numpy(points)


def _geometry(**settings) -> dict:
    # The synthetic coordinates are already in their physical units.
    return {"operator": "fem_p1", "coordinate_scaling": "none", **settings}


def _config(**overrides) -> dict:
    config = {
        "fields": ["u", "v", "w"],
        "geometry": _geometry(),
        "bands": {"count": 3},
        "components": {
            "log_band_power": {"enabled": True, "weight": 1.0},
            "signed_band_cospectrum": {"enabled": True, "weight": 1.0},
        },
    }
    config.update(overrides)
    return config


def _third(tuples, **settings) -> dict:
    return {"enabled": True, "weight": 1.0, "tuples": tuples, **settings}


def _family(config: dict, points: int, fields=("u", "v", "w"), dimension: int = 2):
    spec = DataSpec(tuple(fields), ("1",) * len(fields), dimension, (points,))
    return MultiscaleSpectrumFamily(config, spec, FieldNormalizer.identity(len(fields)))


def _smooth_states(coordinates: torch.Tensor, count: int, fields: int = 3, seed: int = 0):
    """Random superpositions of smooth modes, so every band carries power."""
    generator = torch.Generator().manual_seed(seed)
    x, y = coordinates[:, 0], coordinates[:, 1]
    basis = torch.stack(
        [
            torch.cos(math.pi * kx * x) * torch.cos(math.pi * ky * y)
            for kx in range(5)
            for ky in range(5)
        ],
        dim=-1,
    ).to(torch.float64)
    weights = torch.randn(count, basis.shape[-1], fields, generator=generator, dtype=torch.float64)
    weights = weights / (1.0 + torch.arange(basis.shape[-1], dtype=torch.float64))[None, :, None]
    return torch.einsum("nk,skc->snc", basis, weights) + 0.3 * torch.randn(
        count, 1, fields, generator=generator, dtype=torch.float64
    )


def _fitted(config: dict, coordinates: torch.Tensor, states: torch.Tensor, **kwargs):
    family = _family(config, coordinates.shape[0], **kwargs)
    family.fit_reference_tensors(
        coordinates, states, sample_ids=[f"s{index}" for index in range(states.shape[0])]
    )
    return family


def _descriptor_blocks(family, fields):
    with torch.no_grad():
        return family._descriptors(family._statistics(fields))


def _u_shaped_grid():
    index = np.stack(np.meshgrid(np.arange(21), np.arange(21), indexing="ij"), axis=-1).reshape(
        -1, 2
    )
    keep = ~((index[:, 0] > 6) & (index[:, 0] < 14) & (index[:, 1] > 6))
    return index[keep], index[keep] / 20.0


def _signed_config(**objective):
    """Exercise legacy alpha explicitly; fair/joint tests override it with one."""
    settings = {
        "kind": "signed_band_energy_score",
        "conditional_draws": 2,
        "almost_fair_alpha": 0.95,
    }
    settings.update(objective)
    return _config(objective=settings)


def _band_noise(family, coordinates, count, band, seed):
    """Independent random fields confined to one band of the fitted operator."""
    generator = torch.Generator().manual_seed(seed)
    coefficients = torch.randn(
        count, family.basis.shape[1], len(family.field_names), generator=generator
    )
    coefficients = coefficients.to(family.basis) * family.filters[band][None, :, None]
    fields = torch.einsum("nk,bkc->bnc", family.basis, coefficients)
    full = torch.zeros(count, coordinates.shape[0], len(family.field_names), dtype=fields.dtype)
    full[:, family.analysis_index] = fields
    return full / full.std()
