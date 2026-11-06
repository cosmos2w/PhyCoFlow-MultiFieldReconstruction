"""Differentiable per-state multiscale statistics.

All functions act on one fixed geometry: a spatial measure ``mass`` [N] that
sums to one, an M-orthonormal eigenbasis ``basis`` [N,K] that is mean-free on
every connected piece, and real filters ``filters`` [S,K]. Fields have shape
[B,N,C]. Every statistic is a function of one state; nothing is estimated
across the batch axis.

Averages divide by the stored measure's own sum, so a float32 copy of a unit
measure still removes a constant field exactly up to round-off.
"""

from __future__ import annotations

import torch


def _normalized(measure: torch.Tensor) -> torch.Tensor:
    return measure / measure.sum(dim=-1, keepdim=True)


def piecewise_fluctuations(
    fields: torch.Tensor, piece_measures: torch.Tensor, pieces: torch.Tensor
) -> torch.Tensor:
    """Remove each connected piece's M-weighted mean (the operator null space); [B,N,C]."""
    means = torch.einsum("pn,bnc->bpc", _normalized(piece_measures), fields)
    return fields - means[:, pieces]


def mean_square(values: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    """``<X_i, X_i>_M`` with shape [B,C]."""
    return torch.einsum("n,bnc->bc", _normalized(mass), values.square())


def spectral_coefficients(fluctuations: torch.Tensor, weighted_basis: torch.Tensor) -> torch.Tensor:
    """``a_ki = <phi_k, q_i>_M`` with ``weighted_basis = M Phi``; shape [B,K,C]."""
    return torch.einsum("nk,bnc->bkc", weighted_basis, fluctuations)


def band_gram(coefficients: torch.Tensor, squared_filters: torch.Tensor) -> torch.Tensor:
    """Band cross-field Gram matrices ``G^m_ij = <F_m q_i, F_m q_j>_M``; [B,S,C,C].

    M-orthonormality gives ``G^m_ij = sum_k g_m(lambda_k)^2 a_ki a_kj`` exactly:
    the diagonal is the band (auto) power, the off-diagonal the band co-spectrum,
    and each matrix is positive semidefinite.
    """
    return torch.einsum("sk,bki,bkj->bsij", squared_filters, coefficients, coefficients)


def band_fields(
    coefficients: torch.Tensor, basis: torch.Tensor, filters: torch.Tensor
) -> torch.Tensor:
    """Synthesize every band-filtered field ``F_m q_i`` on the analysis nodes; [B,S,N,C].

    ``F_m q = Phi g_m(Lambda) Phi^T M q`` with the M-orthonormal basis, so the signed
    band fields carry the phases that the Gram matrices discard. Mean-free by
    construction on every connected piece (the basis excludes the null modes).
    """
    return torch.einsum("nk,sk,bkc->bsnc", basis, filters, coefficients)


def regional_centered(series: torch.Tensor, measures: torch.Tensor) -> torch.Tensor:
    """Center each series within each normalized measure; [B,F,N] -> [B,F,R,N]."""
    means = torch.einsum("bfn,rn->bfr", series, _normalized(measures))
    return series[:, :, None, :] - means[..., None]


def regional_variance(series: torch.Tensor, measures: torch.Tensor) -> torch.Tensor:
    """Region-centered second moments ``<(f - <f>_r)^2>_r``; [B,F,N] -> [B,F,R]."""
    centered = regional_centered(series, measures)
    return torch.einsum("bfrn,rn->bfr", centered.square(), _normalized(measures))


def third_central_moments(
    normalized: torch.Tensor, measures: torch.Tensor, index: torch.Tensor
) -> torch.Tensor:
    """Third central moments of centered, scaled factors; [B,F,R,N] -> [B,T,R].

    ``index`` [T,3] selects factor series. For the global measure and mean-free
    band fields this is ``<f g, h>_M``; the statistic is symmetric in its
    factors and its sign is retained.
    """
    product = normalized[:, index[:, 0]] * normalized[:, index[:, 1]] * normalized[:, index[:, 2]]
    return torch.einsum("btrn,rn->btr", product, _normalized(measures))
