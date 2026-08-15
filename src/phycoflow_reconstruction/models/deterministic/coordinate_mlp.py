"""Coordinate regression baseline and its explicitly physics-enabled PINN form."""

from __future__ import annotations

from typing import Any

import torch

from ...contracts import LossBundle, ModelCapabilities, ObservationBatch
from ..base import BaseReconstructionModel, observation_summary
from ..common import FourierFeatures, make_mlp


class CoordinateMLP(BaseReconstructionModel):
    capabilities = ModelCapabilities(
        "point", False, True, False, False, ("base_training", "post_training")
    )

    def __init__(
        self, coordinate_dim: int, num_fields: int, hidden_dim: int = 128, fourier_bands: int = 16
    ) -> None:
        super().__init__()
        self.num_fields = num_fields
        self.position = FourierFeatures(coordinate_dim, fourier_bands)
        self.network = make_mlp(
            self.position.out_dim + 2 * num_fields, hidden_dim, num_fields, depth=4
        )

    def forward_batch(self, batch: ObservationBatch) -> torch.Tensor:
        condition = observation_summary(batch, self.num_fields)
        condition = condition[:, None, :].expand(-1, batch.query_coords.shape[1], -1)
        return self.network(torch.cat((self.position(batch.query_coords), condition), dim=-1))


class PINNRegressor(CoordinateMLP):
    """Coordinate MLP called PINN only when a case physics provider is active."""

    capabilities = ModelCapabilities("point", False, True, False, False, ("direct_physics",))

    def __init__(
        self, *args: Any, physics_provider: Any = None, physics_weight: float = 1.0, **kwargs: Any
    ) -> None:
        if physics_provider is None:
            raise ValueError("PINNRegressor requires an active case physics_provider")
        super().__init__(*args, **kwargs)
        self.physics_provider = physics_provider
        self.physics_weight = float(physics_weight)

    def training_loss(self, batch: ObservationBatch) -> LossBundle:
        data = super().training_loss(batch)
        prediction = self.forward_batch(batch)
        physics = self.physics_provider.loss(prediction, batch)
        total = data.total + self.physics_weight * physics.total
        return LossBundle(total, {**data.components, **physics.components})
