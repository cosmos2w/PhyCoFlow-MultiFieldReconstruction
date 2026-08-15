"""Sparse-to-grid neural-operator regressor using maintained `neuraloperator`.

Sparse values and binary supports are rasterized without interpolation from
hidden targets. The adapter supports one- and two-dimensional logical grids.
"""

from __future__ import annotations

import math

import torch

from ...contracts import ModelCapabilities, ObservationBatch
from ...data.observations import rasterize_observations
from ..base import BaseReconstructionModel

try:
    from neuralop.models import FNO as NeuralOperatorFNO
except ImportError:  # pragma: no cover - exercised through actionable runtime error
    NeuralOperatorFNO = None


class GeoFNORegressor(BaseReconstructionModel):
    capabilities = ModelCapabilities(
        "grid", False, True, True, False, ("base_training", "post_training")
    )

    def __init__(
        self,
        num_fields: int,
        logical_shape: tuple[int, ...],
        hidden_channels: int = 32,
        modes: tuple[int, ...] | None = None,
        layers: int = 3,
    ) -> None:
        super().__init__()
        if NeuralOperatorFNO is None:
            raise ImportError("GeoFNO requires the optional 'neuraloperator' dependency")
        if len(logical_shape) not in {1, 2}:
            raise ValueError("GeoFNO currently supports 1-D or 2-D logical grids")
        self.num_fields = num_fields
        self.logical_shape = logical_shape
        modes = modes or tuple(max(2, min(12, size // 4)) for size in logical_shape)
        self.operator = NeuralOperatorFNO(
            n_modes=tuple(int(v) for v in modes),
            in_channels=2 * num_fields,
            out_channels=num_fields,
            hidden_channels=hidden_channels,
            n_layers=layers,
        )

    def forward_batch(self, batch: ObservationBatch) -> torch.Tensor:
        values, mask = rasterize_observations(batch, self.num_fields)
        prediction = self.operator(torch.cat((values, mask), dim=1))
        point_count = math.prod(self.logical_shape)
        return prediction.reshape(prediction.shape[0], self.num_fields, point_count).transpose(1, 2)
