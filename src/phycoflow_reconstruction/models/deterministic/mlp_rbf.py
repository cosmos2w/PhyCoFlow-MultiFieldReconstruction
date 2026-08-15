"""Direct point regressor using per-field local RBF aggregation of sparse sensors."""

from __future__ import annotations

import torch

from ...contracts import ModelCapabilities, ObservationBatch
from ..base import BaseReconstructionModel, observation_summary
from ..common import FourierFeatures, make_mlp


class MLPRBFRegressor(BaseReconstructionModel):
    capabilities = ModelCapabilities(
        "point", False, True, False, False, ("base_training", "post_training")
    )

    def __init__(
        self,
        coordinate_dim: int,
        num_fields: int,
        hidden_dim: int = 128,
        rbf_sigma: float = 0.08,
        fourier_bands: int = 16,
    ) -> None:
        super().__init__()
        self.num_fields = num_fields
        self.rbf_sigma = float(rbf_sigma)
        self.position = FourierFeatures(coordinate_dim, fourier_bands)
        self.network = make_mlp(
            self.position.out_dim + 2 * num_fields + 2 * num_fields, hidden_dim, num_fields, 4
        )

    def _local_features(self, batch: ObservationBatch) -> torch.Tensor:
        distances = torch.cdist(batch.query_coords, batch.obs_coords).square()
        weights = torch.exp(-distances / (2 * self.rbf_sigma**2)) * batch.obs_valid_mask[:, None, :]
        local_values = []
        local_support = []
        for field_id in range(self.num_fields):
            field_mask = (batch.obs_field_ids == field_id)[:, None, :]
            field_weights = weights * field_mask
            denominator = field_weights.sum(-1).clamp_min(1e-8)
            local_values.append(
                (field_weights * batch.obs_values[:, None, :, 0]).sum(-1) / denominator
            )
            local_support.append(field_weights.sum(-1))
        return torch.cat((torch.stack(local_values, -1), torch.stack(local_support, -1)), dim=-1)

    def forward_batch(self, batch: ObservationBatch) -> torch.Tensor:
        global_condition = observation_summary(batch, self.num_fields)
        global_condition = global_condition[:, None, :].expand(-1, batch.query_coords.shape[1], -1)
        features = torch.cat(
            (self.position(batch.query_coords), global_condition, self._local_features(batch)), -1
        )
        return self.network(features)
