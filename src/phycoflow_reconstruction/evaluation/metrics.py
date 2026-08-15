"""Small representation-independent reconstruction metric set.

Phase-4 runs record normalized full-field and per-field errors plus observed and
unobserved entry errors. Later benchmark aggregation can extend this module
without changing what each model adapter returns.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..contracts import ObservationBatch


def reconstruction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: ObservationBatch,
    field_names: Sequence[str],
) -> dict[str, float | dict[str, float] | None]:
    squared = (prediction - target).square()
    valid = batch.query_valid_mask.unsqueeze(-1).expand_as(squared)
    total = squared[valid].mean()
    per_field = {
        str(name): float(squared[..., field][batch.query_valid_mask].mean().cpu())
        for field, name in enumerate(field_names)
    }

    observed = torch.zeros_like(valid)
    query_indices = batch.metadata.get("query_indices")
    if batch.obs_indices is not None and isinstance(query_indices, torch.Tensor):
        for batch_index in range(prediction.shape[0]):
            lookup = {
                int(point): position
                for position, point in enumerate(query_indices[batch_index].tolist())
                if point >= 0
            }
            obs_valid = batch.obs_valid_mask[batch_index]
            for point, field in zip(
                batch.obs_indices[batch_index, obs_valid].tolist(),
                batch.obs_field_ids[batch_index, obs_valid].tolist(),
            ):
                position = lookup.get(int(point))
                if position is not None:
                    observed[batch_index, position, int(field)] = True
    observed &= valid
    unobserved = valid & ~observed
    return {
        "mse_normalized": float(total.cpu()),
        "per_field_mse_normalized": per_field,
        "observed_entry_mse_normalized": (
            float(squared[observed].mean().cpu()) if observed.any() else None
        ),
        "unobserved_entry_mse_normalized": (
            float(squared[unobserved].mean().cpu()) if unobserved.any() else None
        ),
    }
