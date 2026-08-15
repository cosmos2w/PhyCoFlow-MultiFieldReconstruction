"""Readable neural building blocks reused across point and generative models."""

from __future__ import annotations

import math

import torch
from torch import nn


def make_mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int = 3) -> nn.Sequential:
    if depth < 2:
        raise ValueError("MLP depth must be at least two")
    layers: list[nn.Module] = []
    current = in_dim
    for _ in range(depth - 1):
        layers.extend((nn.Linear(current, hidden_dim), nn.GELU()))
        current = hidden_dim
    layers.append(nn.Linear(current, out_dim))
    return nn.Sequential(*layers)


class FourierFeatures(nn.Module):
    def __init__(self, coordinate_dim: int, bands: int = 16, max_frequency: float = 32.0) -> None:
        super().__init__()
        self.coordinate_dim = coordinate_dim
        frequencies = torch.linspace(1.0, max_frequency, bands)
        self.register_buffer("frequencies", frequencies)
        self.out_dim = coordinate_dim * bands * 2

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        angles = coordinates[..., : self.coordinate_dim].unsqueeze(-1) * self.frequencies * math.pi
        return torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(-2)


class AttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, ff_multiplier: int = 2) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, width * ff_multiplier),
            nn.GELU(),
            nn.Linear(width * ff_multiplier, width),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_attention(values)
        values = values + self.attention(normalized, normalized, normalized, need_weights=False)[0]
        return values + self.feed_forward(self.norm_ff(values))
