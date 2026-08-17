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
    costs = [
        (generated[..., left, right] - reference[..., left, right]).square().mean()
        for left, right in pairs
    ]
    return torch.stack(costs).mean()


def off_diagonal_pair_mean_square(
    generated: torch.Tensor,
    reference: torch.Tensor,
    pairs: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    mask = ~torch.eye(generated.shape[0], device=generated.device, dtype=torch.bool)
    costs = [
        (generated[:, :, left, right] - reference[:, :, left, right])[mask].square().mean()
        for left, right in pairs
    ]
    return torch.stack(costs).mean()
