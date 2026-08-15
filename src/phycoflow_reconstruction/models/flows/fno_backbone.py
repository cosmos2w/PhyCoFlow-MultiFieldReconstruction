"""Regular-grid FNO velocity backbone for the PointCloudFFM wrapper."""

from __future__ import annotations

import math

import torch
from torch import nn

from ...contracts import ObservationBatch
from ...data.observations import rasterize_observations

try:
    from neuralop.models import FNO as NeuralOperatorFNO
except ImportError:  # pragma: no cover
    NeuralOperatorFNO = None


class FNOFlowBackbone(nn.Module):
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
            raise ImportError("FNO PointCloudFFM backbone requires neuraloperator")
        if len(logical_shape) not in {1, 2}:
            raise ValueError("FNO flow backbone supports 1-D or 2-D grids")
        self.num_fields = num_fields
        self.logical_shape = logical_shape
        modes = modes or tuple(max(2, min(12, size // 4)) for size in logical_shape)
        self.operator = NeuralOperatorFNO(
            n_modes=modes,
            in_channels=3 * num_fields + 1,
            out_channels=num_fields,
            hidden_channels=hidden_channels,
            n_layers=layers,
        )

    def forward(
        self, batch: ObservationBatch, state: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        values, mask = rasterize_observations(batch, self.num_fields)
        point_count = math.prod(self.logical_shape)
        if state.shape[1] != point_count:
            raise ValueError("FNO flow backbone requires complete-grid query points")
        state_grid = state.transpose(1, 2).reshape(
            state.shape[0], self.num_fields, *self.logical_shape
        )
        time_grid = time.reshape(time.shape[0], 1, *([1] * len(self.logical_shape))).expand(
            -1, 1, *self.logical_shape
        )
        velocity = self.operator(torch.cat((state_grid, values, mask, time_grid), dim=1))
        return velocity.reshape(state.shape[0], self.num_fields, point_count).transpose(1, 2)
