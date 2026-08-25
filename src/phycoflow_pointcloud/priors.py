"""Random-field priors used by the point-cloud flow-matching wrapper."""

from __future__ import annotations

import math

import torch
from torch import nn


class IIDGaussianPrior(nn.Module):
    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        bsz, n_pts, _ = coords.shape
        return torch.randn(
            bsz, n_pts, n_channels, device=coords.device, dtype=coords.dtype
        )


class RFFGaussianPrior(nn.Module):
    """Scalable smooth Gaussian-field approximation via random Fourier features."""

    def __init__(
        self, coord_dim: int = 3, n_features: int = 256, lengthscale: float = 0.15
    ):
        super().__init__()
        self.coord_dim = coord_dim
        self.n_features = n_features
        self.lengthscale = lengthscale
        self.register_buffer(
            "omega", torch.randn(coord_dim, n_features) / max(lengthscale, 1e-6)
        )
        self.register_buffer("phase", 2 * math.pi * torch.rand(n_features))

    def _features(self, coords: torch.Tensor) -> torch.Tensor:
        z = coords @ self.omega + self.phase
        return math.sqrt(2.0 / self.n_features) * torch.cos(z)

    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        phi = self._features(coords)
        bsz, _, n_feat = phi.shape
        weights = torch.randn(
            bsz, n_channels, n_feat, device=coords.device, dtype=coords.dtype
        )
        return torch.einsum("bnf,bcf->bnc", phi, weights)
