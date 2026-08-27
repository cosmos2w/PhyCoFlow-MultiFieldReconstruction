"""Fixed source-checkpoint gradient calibration for coherence families."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

import torch

Gradients = tuple[torch.Tensor | None, ...]


def loss_gradients(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool = False,
) -> Gradients:
    """Return detached gradients for one scalar loss without updating parameters."""
    gradients = torch.autograd.grad(
        loss,
        parameters,
        allow_unused=True,
        retain_graph=retain_graph,
        create_graph=False,
    )
    return tuple(None if gradient is None else gradient.detach() for gradient in gradients)


def gradient_norm(gradients: Gradients) -> float:
    """Compute a stable Euclidean norm across a possibly sparse gradient tuple."""
    squared: torch.Tensor | None = None
    for gradient in gradients:
        if gradient is None:
            continue
        contribution = gradient.double().square().sum()
        squared = contribution if squared is None else squared + contribution
    return 0.0 if squared is None else float(torch.sqrt(squared).cpu())


def gradient_cosine(left: Gradients, right: Gradients, *, epsilon: float) -> float:
    """Cosine between two sparse gradient tuples with aligned parameters."""
    if len(left) != len(right):
        raise ValueError("gradient tuples must have the same parameter layout")
    dot = torch.zeros((), dtype=torch.float64)
    left_sq = torch.zeros((), dtype=torch.float64)
    right_sq = torch.zeros((), dtype=torch.float64)
    for left_gradient, right_gradient in zip(left, right, strict=True):
        if left_gradient is not None:
            left_sq = left_sq + left_gradient.double().square().sum().cpu()
        if right_gradient is not None:
            right_sq = right_sq + right_gradient.double().square().sum().cpu()
        if left_gradient is not None and right_gradient is not None:
            dot = dot + (left_gradient.double() * right_gradient.double()).sum().cpu()
    denominator = torch.sqrt(left_sq * right_sq)
    if not torch.isfinite(denominator) or float(denominator) <= epsilon:
        raise FloatingPointError("cannot compute a cosine for an effectively zero gradient")
    value = float(dot / denominator)
    if not math.isfinite(value):
        raise FloatingPointError("gradient cosine is non-finite")
    return max(-1.0, min(1.0, value))


def gradient_diagnostics(
    data_gradients: Gradients,
    family_gradients: Mapping[str, Gradients],
    *,
    epsilon: float,
) -> dict[str, Any]:
    """Return norms and the complete data/family cosine diagnostics."""
    names = tuple(family_gradients)
    norms = {name: gradient_norm(family_gradients[name]) for name in names}
    data_norm = gradient_norm(data_gradients)
    return {
        "native_data_gradient_norm": data_norm,
        "family_gradient_norms": norms,
        "family_family_cosines": {
            left: {
                right: (
                    1.0
                    if left == right
                    else gradient_cosine(
                        family_gradients[left], family_gradients[right], epsilon=epsilon
                    )
                )
                for right in names
            }
            for left in names
        },
        "data_family_cosines": {
            name: gradient_cosine(data_gradients, family_gradients[name], epsilon=epsilon)
            for name in names
        },
    }


def resolve_family_scales(
    batch_gradient_norms: Sequence[Mapping[str, float]],
    settings: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    """Robustly aggregate calibration batches and resolve fixed family scales."""
    if not batch_gradient_norms:
        raise ValueError("family calibration requires at least one batch")
    names = tuple(batch_gradient_norms[0])
    if not names or any(tuple(record) != names for record in batch_gradient_norms):
        raise ValueError("calibration batches must contain the same ordered families")
    epsilon = float(settings.get("epsilon", 1.0e-12))
    scale_min = float(settings.get("scale_min", 1.0e-2))
    scale_max = float(settings.get("scale_max", 1.0e2))
    max_batch_ratio = float(settings.get("max_batch_ratio", 1.0e2))
    if epsilon <= 0 or scale_min <= 0 or scale_max < scale_min or max_batch_ratio < 1:
        raise ValueError("invalid family-balance calibration bounds")
    if settings.get("reference", "median") != "median":
        raise ValueError("family_balance.reference currently supports only median")

    aggregated: dict[str, float] = {}
    for name in names:
        values = [float(record[name]) for record in batch_gradient_norms]
        if any(not math.isfinite(value) or value <= epsilon for value in values):
            raise FloatingPointError(
                f"family {name} has a non-finite or effectively zero calibration gradient"
            )
        if max(values) / min(values) > max_batch_ratio:
            raise FloatingPointError(
                f"family {name} calibration gradients are unstable across batches"
            )
        aggregated[name] = float(statistics.median(values))
    reference_norm = float(statistics.median(aggregated.values()))
    scales = {
        name: min(scale_max, max(scale_min, reference_norm / max(norm, epsilon)))
        for name, norm in aggregated.items()
    }
    return aggregated, scales
