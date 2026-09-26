"""MIMONet branch--trunk operator adapted to the shared observation contract.

The released model uses one FCN for measured values, one for their locations,
a multiplicative merge of the two branch vectors, and a coordinate FCN whose
output is contracted with that shared vector for every requested field.  This
adapter keeps that computation and supplies fixed, field-specific sensor slots
from :class:`ObservationBatch` without reading target fields.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from ...contracts import ModelCapabilities, ObservationBatch
from ..base import BaseReconstructionModel


class _FCN(nn.Module):
    """Released MIMONet FCN: three ReLU hidden layers and a linear output."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class MIMONetOperator(BaseReconstructionModel):
    """Fixed-capacity multi-input, multi-output neural operator.

    ``conditioning_fields`` declares exactly which measured fields may enter
    the branches. ``sensor_capacities`` defines a fixed segment for each field;
    valid observations retain their within-field order and unused slots carry
    a zero value, zero coordinate, and zero validity flag. The trunk sees only
    the query coordinates. The five combustion outputs, for example, all share
    the same 256-dimensional branch vector when ``basis_dim=256``.
    """

    capabilities = ModelCapabilities(
        "point", False, True, False, False, ("base_training", "post_training")
    )

    def __init__(
        self,
        coordinate_dim: int,
        num_fields: int,
        field_names: Sequence[str],
        conditioning_fields: Sequence[str],
        sensor_capacities: Sequence[int],
        basis_dim: int = 256,
        branch_hidden_dim: int = 512,
        trunk_hidden_dim: int = 256,
        merge_type: str = "mul",
    ) -> None:
        super().__init__()
        if coordinate_dim < 1 or num_fields < 1:
            raise ValueError("coordinate_dim and num_fields must be positive")
        if len(field_names) != num_fields or len(set(field_names)) != num_fields:
            raise ValueError("field_names must list each output field exactly once")
        if not conditioning_fields or len(conditioning_fields) != len(sensor_capacities):
            raise ValueError("each conditioning field needs one sensor capacity")
        if len(set(conditioning_fields)) != len(conditioning_fields):
            raise ValueError("conditioning_fields must be unique")
        if set(conditioning_fields) - set(field_names):
            raise ValueError("conditioning_fields must belong to the dataset")
        if any(int(value) < 1 for value in sensor_capacities):
            raise ValueError("sensor capacities must be positive")
        if min(basis_dim, branch_hidden_dim, trunk_hidden_dim) < 1:
            raise ValueError("MIMONet network dimensions must be positive")
        if merge_type not in {"mul", "sum"}:
            raise ValueError("merge_type must be 'mul' or 'sum'")

        self.num_fields = num_fields
        self.coordinate_dim = coordinate_dim
        self.conditioning_field_ids = tuple(field_names.index(name) for name in conditioning_fields)
        self.sensor_capacities = tuple(int(value) for value in sensor_capacities)
        self.max_sensors = sum(self.sensor_capacities)
        self.basis_dim = int(basis_dim)
        self.merge_type = merge_type

        # Keep the released parameter/module structure for the two FCN branches,
        # the FCN trunk, and one learned bias per output field.
        self.branch_nets = nn.ModuleList(
            (
                _FCN(2 * self.max_sensors, branch_hidden_dim, basis_dim),
                _FCN((coordinate_dim + 1) * self.max_sensors, branch_hidden_dim, basis_dim),
            )
        )
        self.trunk_net = _FCN(coordinate_dim, trunk_hidden_dim, basis_dim * num_fields)
        self.bias = nn.Parameter(torch.zeros(1, 1, num_fields))

    def _packed_branches(self, batch: ObservationBatch) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, observed, coordinate_dim = batch.obs_coords.shape
        if coordinate_dim != self.coordinate_dim:
            raise ValueError("observation coordinates have the wrong dimension")
        if batch.query_coords.shape[-1] != self.coordinate_dim:
            raise ValueError("query coordinates have the wrong dimension")
        allowed = torch.zeros_like(batch.obs_valid_mask)
        for field_id in self.conditioning_field_ids:
            allowed |= batch.obs_field_ids == field_id
        if bool((batch.obs_valid_mask & ~allowed).any()):
            raise ValueError("observations contain an undeclared conditioning field")

        packed_values = batch.obs_values.new_zeros(batch_size, self.max_sensors, 1)
        packed_coords = batch.obs_coords.new_zeros(
            batch_size, self.max_sensors, self.coordinate_dim
        )
        packed_valid = batch.obs_valid_mask.new_zeros(batch_size, self.max_sensors)
        start = 0
        for field_id, capacity in zip(self.conditioning_field_ids, self.sensor_capacities):
            matches = (batch.obs_field_ids == field_id) & batch.obs_valid_mask
            counts = matches.sum(dim=1)
            if bool((counts > capacity).any()):
                raise ValueError(f"conditioning field {field_id} exceeds its sensor capacity")
            take = min(capacity, observed)
            # Sort valid entries first while retaining their original order.
            positions = torch.arange(observed, device=matches.device).expand(batch_size, -1)
            order = torch.argsort((~matches).long() * observed + positions, dim=1)[:, :take]
            valid = torch.arange(take, device=matches.device)[None, :] < counts[:, None]
            packed_valid[:, start : start + take] = valid
            packed_values[:, start : start + take] = batch.obs_values.gather(
                1, order.unsqueeze(-1)
            ) * valid.unsqueeze(-1)
            packed_coords[:, start : start + take] = batch.obs_coords.gather(
                1, order.unsqueeze(-1).expand(-1, -1, coordinate_dim)
            ) * valid.unsqueeze(-1)
            start += capacity

        mask = packed_valid.unsqueeze(-1)
        value_input = torch.cat((packed_values * mask, mask.to(packed_values.dtype)), dim=-1)
        location_input = torch.cat(
            (packed_coords * mask, mask.to(packed_coords.dtype)), dim=-1
        )
        return value_input.flatten(1), location_input.flatten(1)

    def forward_batch(self, batch: ObservationBatch) -> torch.Tensor:
        value_input, location_input = self._packed_branches(batch)
        value_branch = self.branch_nets[0](value_input)
        location_branch = self.branch_nets[1](location_input)
        if self.merge_type == "mul":
            combined = value_branch * location_branch
        else:
            combined = value_branch + location_branch
        trunk = self.trunk_net(batch.query_coords).reshape(
            batch.query_coords.shape[0],
            batch.query_coords.shape[1],
            self.basis_dim,
            self.num_fields,
        )
        return torch.einsum("bi,bqic->bqc", combined, trunk) + self.bias
