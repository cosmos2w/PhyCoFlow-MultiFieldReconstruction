"""Per-field empirical marginal Wasserstein component."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .....contracts import CoherenceComponentSpec, TermResult
from ....base import empirical_w2_columns


class SelfMarginalW2(nn.Module):
    def __init__(
        self,
        field_ids: Sequence[int],
        channel_weights: Sequence[float] | None = None,
        *,
        target_use: str = "training_reference",
        units: str = "model_units",
    ) -> None:
        super().__init__()
        self.spec = CoherenceComponentSpec("self.marginal_w2", target_use, units, True)
        self.spec.validate()
        self.field_ids = tuple(int(value) for value in field_ids)
        if not self.field_ids:
            raise ValueError("self.marginal_w2 requires at least one field")
        weights = (
            torch.ones(len(self.field_ids))
            if channel_weights is None
            else torch.as_tensor(channel_weights)
        )
        if (
            weights.numel() != len(self.field_ids)
            or torch.any(weights < 0)
            or float(weights.sum()) <= 0
        ):
            raise ValueError(
                "self channel weights must be non-negative and align with selected fields"
            )
        self.register_buffer("channel_weights", weights.float() / weights.float().sum())

    def forward(self, generated: torch.Tensor, reference: torch.Tensor) -> TermResult:
        per_batch = []
        per_field = []
        weights = self.channel_weights.to(device=generated.device, dtype=generated.dtype)
        for generated_item, reference_item in zip(generated, reference):
            costs = empirical_w2_columns(
                generated_item[:, self.field_ids], reference_item[:, self.field_ids]
            )
            per_field.append(costs)
            per_batch.append((costs * weights).sum())
        per_sample = torch.stack(per_batch)
        return TermResult(
            per_sample_cost=per_sample,
            scalar_loss=per_sample.mean(),
            diagnostics={"per_field_w2": torch.stack(per_field)},
        )
