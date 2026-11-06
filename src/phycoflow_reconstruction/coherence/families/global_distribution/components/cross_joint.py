"""All-field fixed-bank top-fraction sliced-Wasserstein component."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from .....contracts import CoherenceComponentSpec, TermResult
from ....base import empirical_w2_columns, projection_bank, weighted_w2_columns


class CrossJointTopKSWD(nn.Module):
    def __init__(
        self,
        field_ids: Sequence[int],
        directions: int,
        top_fraction: float,
        seed: int,
        include_axes: bool,
        qmc: bool,
        target_use: str = "training_reference",
        units: str = "model_units",
        mean_reduction: bool = False,
    ) -> None:
        super().__init__()
        self.spec = CoherenceComponentSpec("cross.joint_topk_swd", target_use, units, True)
        self.spec.validate()
        self.field_ids = tuple(int(value) for value in field_ids)
        if len(self.field_ids) < 2:
            self.register_buffer("directions", torch.empty(0, len(self.field_ids)))
        else:
            self.register_buffer(
                "directions",
                projection_bank(
                    len(self.field_ids), directions, seed=seed, include_axes=include_axes, qmc=qmc
                ),
            )
        if not 0.0 < float(top_fraction) <= 1.0:
            raise ValueError("cross top_fraction must lie in (0,1]")
        self.top_fraction = float(top_fraction)
        self.mean_reduction = bool(mean_reduction)

    def forward(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor,
        *,
        generated_weights: torch.Tensor | None = None,
        reference_weights: torch.Tensor | None = None,
    ) -> TermResult:
        if len(self.field_ids) < 2:
            zero = generated.sum(dim=(1, 2)) * 0.0
            return TermResult(zero, zero.mean(), reason="fewer than two configured fields")
        directions = self.directions.to(device=generated.device, dtype=generated.dtype)
        per_batch = []
        per_direction = []
        top_indices = []
        for index, (generated_item, reference_item) in enumerate(zip(generated, reference)):
            generated_projection = generated_item[:, self.field_ids] @ directions.T
            reference_projection = reference_item[:, self.field_ids] @ directions.T
            if generated_weights is None and reference_weights is None:
                costs = empirical_w2_columns(generated_projection, reference_projection)
            elif generated_weights is not None and reference_weights is not None:
                costs = weighted_w2_columns(
                    generated_projection, reference_projection,
                    generated_weights[index], reference_weights[index],
                )
            else:
                raise ValueError("both endpoint measures are required")
            count = max(1, min(costs.numel(), math.ceil(self.top_fraction * costs.numel())))
            if self.mean_reduction:
                values, indices = costs, torch.arange(costs.numel(), device=costs.device)
            else:
                values, indices = costs.topk(count, largest=True, sorted=True)
            per_batch.append(values.mean())
            per_direction.append(costs)
            top_indices.append(indices)
        per_sample = torch.stack(per_batch)
        return TermResult(
            per_sample_cost=per_sample,
            scalar_loss=per_sample.mean(),
            diagnostics={
                "per_direction_w2": torch.stack(per_direction),
                "top_indices": torch.stack(top_indices),
            },
        )
