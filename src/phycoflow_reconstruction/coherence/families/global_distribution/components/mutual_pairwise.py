"""Pairwise two-field sliced-Wasserstein component."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .....contracts import CoherenceComponentSpec, TermResult
from ....base import empirical_w2_columns, projection_bank


class MutualPairwiseSWD(nn.Module):
    def __init__(
        self,
        pairs: Sequence[tuple[int, int]],
        directions: int,
        seed: int,
        *,
        target_use: str = "training_reference",
        units: str = "model_units",
    ) -> None:
        super().__init__()
        self.spec = CoherenceComponentSpec("mutual.pairwise_swd", target_use, units, True)
        self.spec.validate()
        self.pairs = tuple((int(left), int(right)) for left, right in pairs)
        if any(left == right for left, right in self.pairs):
            raise ValueError("mutual field pairs must contain distinct fields")
        # Preserve the historical pair-specific seed convention so migrated
        # Demo50 losses remain numerically comparable.
        banks = [
            projection_bank(2, directions, seed=seed + left * 1009 + right, qmc=False)
            for left, right in self.pairs
        ]
        self.register_buffer(
            "directions", torch.stack(banks) if banks else torch.empty(0, directions, 2)
        )

    def forward(self, generated: torch.Tensor, reference: torch.Tensor) -> TermResult:
        if not self.pairs:
            zero = generated.sum(dim=(1, 2)) * 0.0
            return TermResult(zero, zero.mean(), reason="fewer than two configured fields")
        per_batch = []
        per_pair = []
        banks = self.directions.to(device=generated.device, dtype=generated.dtype)
        for generated_item, reference_item in zip(generated, reference):
            costs = []
            for pair_index, pair in enumerate(self.pairs):
                generated_projection = generated_item[:, pair] @ banks[pair_index].T
                reference_projection = reference_item[:, pair] @ banks[pair_index].T
                costs.append(
                    empirical_w2_columns(generated_projection, reference_projection).mean()
                )
            pair_costs = torch.stack(costs)
            per_pair.append(pair_costs)
            per_batch.append(pair_costs.mean())
        per_sample = torch.stack(per_batch)
        return TermResult(
            per_sample_cost=per_sample,
            scalar_loss=per_sample.mean(),
            diagnostics={"per_pair_swd": torch.stack(per_pair)},
        )
