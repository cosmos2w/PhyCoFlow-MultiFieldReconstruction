"""Finite-ensemble metrics for conditionally independent field reconstructions."""

from collections.abc import Sequence

import torch


def ensemble_crps(draws: Sequence[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    """Fair K-member marginal CRPS, per field, for conditionally IID [B,N,C] draws.

    Unbiased for the expected CRPS at any K >= 2 and equal to ``paired_draw_crps``
    for two draws; extra members only lower the estimator variance.
    """
    count = len(draws)
    if count < 2 or any(draw.shape != target.shape or draw.ndim != 3 for draw in draws):
        raise ValueError("ensemble CRPS expects at least two [B,N,C] draws matching the target")
    target = target.detach()
    accuracy = sum((draw - target).abs() for draw in draws) / count
    spread = sum(
        (draws[left] - draws[right]).abs()
        for left in range(count)
        for right in range(left + 1, count)
    ) / (count * (count - 1))
    return (accuracy - spread).mean(dim=(0, 1))
