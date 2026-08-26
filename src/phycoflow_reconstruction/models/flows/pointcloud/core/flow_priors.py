"""IID and smooth random-Fourier source priors for rectified flow."""

from __future__ import annotations

import math

import torch
from torch import nn


class IIDGaussianPrior(nn.Module):
    def sample(
        self,
        coordinates: torch.Tensor,
        channels: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return torch.randn(
            *coordinates.shape[:2],
            channels,
            device=coordinates.device,
            dtype=coordinates.dtype,
            generator=generator,
        )


class RFFGaussianPrior(nn.Module):
    def __init__(self, coordinate_dim: int, features: int = 64, lengthscale: float = 0.15) -> None:
        super().__init__()
        self.coordinate_dim = coordinate_dim
        self.features = features
        frequencies = torch.randn(features, coordinate_dim) / max(float(lengthscale), 1e-6)
        phases = 2 * math.pi * torch.rand(features)
        self.register_buffer("frequencies", frequencies)
        self.register_buffer("phases", phases)

    def sample(
        self,
        coordinates: torch.Tensor,
        channels: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        features = math.sqrt(2.0 / self.features) * torch.cos(
            torch.einsum("bqd,fd->bqf", coordinates[..., : self.coordinate_dim], self.frequencies)
            + self.phases
        )
        weights = torch.randn(
            coordinates.shape[0],
            self.features,
            channels,
            device=coordinates.device,
            dtype=coordinates.dtype,
            generator=generator,
        )
        return torch.einsum("bqf,bfc->bqc", features, weights)
