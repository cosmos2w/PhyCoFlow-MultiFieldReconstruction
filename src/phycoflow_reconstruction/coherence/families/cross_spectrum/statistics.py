"""Vectorized differentiable graph cross-spectrum statistics."""

from __future__ import annotations

import torch


def graph_fourier(fields: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if fields.ndim != 3 or basis.ndim != 2 or fields.shape[1] != basis.shape[0]:
        raise ValueError("graph Fourier inputs must align as [B,N,C] and [N,K]")
    return torch.einsum("nk,bnc->bkc", basis, fields)


def spectral_coherence(coefficients: torch.Tensor, eps: float) -> torch.Tensor:
    auto = coefficients.abs().square().mean(dim=0)
    cross = torch.einsum("bki,bkj->kij", coefficients, coefficients.conj()) / coefficients.shape[0]
    denominator = torch.einsum("ki,kj->kij", auto, auto)
    return cross.abs().square() / (denominator + eps)


def auto_spectrum(coefficients: torch.Tensor) -> torch.Tensor:
    """Return the ensemble-mean modewise power for every field.

    Coefficients have shape ``[batch, mode, field]`` and the returned
    auto-spectrum has shape ``[mode, field]``.  Keeping this estimator
    separate from the coherence estimator makes the absolute spectral-power
    term usable without changing the existing pairwise coherence terms.
    """
    if coefficients.ndim != 3:
        raise ValueError("auto-spectrum coefficients must have shape [B,K,C]")
    return coefficients.abs().square().mean(dim=0)


def auto_spectrum_mean_square_values(
    generated: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    """Return one modewise auto-spectrum MSE value per field."""
    if generated.ndim != 3 or reference.shape != generated.shape:
        raise ValueError("auto-spectrum coefficients must align as [B,K,C]")
    return (auto_spectrum(generated) - auto_spectrum(reference)).square().mean(dim=0)


def band_energies(coefficients: torch.Tensor, band_ids: torch.Tensor) -> torch.Tensor:
    count = int(band_ids.max().item()) + 1
    energies = []
    power = coefficients.abs().square()
    for band_id in range(count):
        energies.append(power[:, band_ids == band_id].sum(dim=1))
    return torch.stack(energies, dim=1)


def normalized_cross_band_coupling(energies: torch.Tensor, eps: float) -> torch.Tensor:
    centered = energies - energies.mean(dim=0, keepdim=True)
    covariance = torch.einsum("bmi,bnj->mnij", centered, centered) / energies.shape[0]
    variances = torch.stack(
        [torch.diagonal(covariance[index, index]) for index in range(covariance.shape[0])]
    )
    denominator = torch.einsum("mi,nj->mnij", variances, variances)
    return covariance.abs().square() / (denominator + eps)


def pair_mean_square(
    generated: torch.Tensor,
    reference: torch.Tensor,
    pairs: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    return pair_mean_square_values(generated, reference, pairs).mean()


def pair_mean_square_values(
    generated: torch.Tensor,
    reference: torch.Tensor,
    pairs: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    """Return one same-frequency mean-square discrepancy per field pair."""
    return torch.stack(
        [
            (generated[..., left, right] - reference[..., left, right]).square().mean()
            for left, right in pairs
        ]
    )


def pair_symmetric_coherence_scores(
    generated: torch.Tensor,
    reference: torch.Tensor,
    pairs: tuple[tuple[int, int], ...],
    eps: float,
) -> torch.Tensor:
    """Return bounded same-frequency agreement scores, where one is exact agreement."""
    return torch.stack(
        [
            symmetric_relative_coherence_score(
                generated[..., left, right], reference[..., left, right], eps
            )
            for left, right in pairs
        ]
    )


def off_diagonal_pair_mean_square(
    generated: torch.Tensor,
    reference: torch.Tensor,
    pairs: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    return off_diagonal_pair_mean_square_values(generated, reference, pairs).mean()


def off_diagonal_pair_mean_square_values(
    generated: torch.Tensor,
    reference: torch.Tensor,
    pairs: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    """Return one off-diagonal cross-band discrepancy per field pair."""
    mask = ~torch.eye(generated.shape[0], device=generated.device, dtype=torch.bool)
    return torch.stack(
        [
            (generated[:, :, left, right] - reference[:, :, left, right])[mask].square().mean()
            for left, right in pairs
        ]
    )


def off_diagonal_pair_symmetric_coherence_scores(
    generated: torch.Tensor,
    reference: torch.Tensor,
    pairs: tuple[tuple[int, int], ...],
    eps: float,
) -> torch.Tensor:
    """Return bounded cross-frequency agreement scores over off-diagonal bands."""
    mask = ~torch.eye(generated.shape[0], device=generated.device, dtype=torch.bool)
    return torch.stack(
        [
            symmetric_relative_coherence_score(
                generated[:, :, left, right][mask],
                reference[:, :, left, right][mask],
                eps,
            )
            for left, right in pairs
        ]
    )


def symmetric_relative_coherence_score(
    generated: torch.Tensor, reference: torch.Tensor, eps: float
) -> torch.Tensor:
    """Map symmetric relative L2 discrepancy to a bounded [0, 1] agreement score."""
    difference = torch.linalg.vector_norm(generated - reference)
    scale = torch.linalg.vector_norm(generated) + torch.linalg.vector_norm(reference)
    return (1.0 - difference / scale.clamp_min(eps)).clamp(0.0, 1.0)
