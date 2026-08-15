"""Sparse-observation DeepONet with a token branch and coordinate trunk."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ...contracts import ModelCapabilities, ObservationBatch
from ..base import BaseReconstructionModel
from ..common import FourierFeatures, make_mlp


class SparseDeepONet(BaseReconstructionModel):
    capabilities = ModelCapabilities(
        "point", False, True, False, False, ("base_training", "post_training")
    )

    def __init__(
        self, coordinate_dim: int, num_fields: int, width: int = 128, basis_dim: int = 64
    ) -> None:
        super().__init__()
        self.num_fields = num_fields
        self.position = FourierFeatures(coordinate_dim, 12)
        token_dim = coordinate_dim + 1 + num_fields
        self.branch_token = make_mlp(token_dim, width, basis_dim * num_fields, 3)
        self.trunk = make_mlp(self.position.out_dim, width, basis_dim * num_fields, 3)
        self.bias = torch.nn.Parameter(torch.zeros(num_fields))
        self.basis_dim = basis_dim

    def forward_batch(self, batch: ObservationBatch) -> torch.Tensor:
        field_one_hot = F.one_hot(batch.obs_field_ids, self.num_fields).to(batch.obs_values.dtype)
        tokens = torch.cat((batch.obs_coords, batch.obs_values, field_one_hot), -1)
        encoded = self.branch_token(tokens) * batch.obs_valid_mask.unsqueeze(-1)
        branch = encoded.sum(1) / batch.obs_valid_mask.sum(1, keepdim=True).clamp_min(1)
        branch = branch.reshape(branch.shape[0], self.num_fields, self.basis_dim)
        trunk = self.trunk(self.position(batch.query_coords)).reshape(
            batch.query_coords.shape[0],
            batch.query_coords.shape[1],
            self.num_fields,
            self.basis_dim,
        )
        return torch.einsum("bcf,bqcf->bqc", branch, trunk) + self.bias
