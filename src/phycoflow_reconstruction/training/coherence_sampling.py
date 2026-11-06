"""Frozen TRAIN calibration and conditional-draw contracts for coherence families."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from ..data.factory import FieldDataset
from ..data.training_batches import fixed_query_indices


def _coherence_query_seed(config: Mapping[str, Any]) -> int:
    """Seed of the fixed shared coherence queries, as used by every batch builder."""
    compute = config["coherence"]["compute_budget"]
    default = config["observations"].get("seed", config["runtime"].get("seed", 42)) + 100_003
    return int(compute.get("query_seed", default))


def _conditional_draw_count(families: Mapping[str, Any]) -> int:
    return max(
        (int(getattr(family, "conditional_draws", 1)) for family in families.values()), default=1
    )


def _conditional_draw_seed(base_seed: int, draw: int) -> int:
    # A golden-ratio stride keeps extra-draw seeds clear of the per-step rollout seeds.
    return (int(base_seed) + int(draw) * 0x9E3779B97F4A7C15) % (1 << 63)


def _family_context(
    family,
    context: Mapping[str, Any],
    conditional_draws: list[torch.Tensor],
    *,
    report_detail: bool = False,
) -> dict[str, Any]:
    """Add the extra conditional draws and report requests a family declares it uses."""
    payload = dict(context)
    needed = int(getattr(family, "conditional_draws", 1))
    if needed > 1:
        payload["conditional_draws"] = tuple(conditional_draws[: needed - 1])
    if report_detail and getattr(family, "supports_report_detail", False):
        payload["report_detail"] = True
        if getattr(family, "supports_ensemble_report", False):
            # Every available independent draw, for per-band ensemble diagnostics only.
            payload["ensemble_draws"] = tuple(conditional_draws)
    return payload


def _fit_training_references(
    families: Mapping[str, Any],
    train_dataset: FieldDataset,
    config: Mapping[str, Any],
    *,
    device: torch.device,
    structured_grid_required: bool,
) -> None:
    """Freeze training-only geometry and reference scales before any model evaluation."""
    total = math.prod(train_dataset.data_spec.logical_shape)
    point_count = min(int(config["coherence"]["compute_budget"]["point_count"]), total)
    for name, family in families.items():
        fit = getattr(family, "fit_training_reference", None)
        if fit is None:
            continue
        if structured_grid_required and point_count < total:
            raise ValueError(
                f"{name} with a structured-grid source needs coherence.compute_budget.point_count "
                f"to cover all {total} grid points"
            )
        query_indices = fixed_query_indices(total, point_count, seed=_coherence_query_seed(config))
        with _preserved_rng(device):
            fit(train_dataset, query_indices=query_indices, device=device)


@contextmanager
def _preserved_rng(device: torch.device) -> Iterator[None]:
    """Keep calibration draws out of the training random stream."""
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    try:
        yield
    finally:
        torch.set_rng_state(torch_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
