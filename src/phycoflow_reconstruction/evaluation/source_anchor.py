"""Explicit, TRAIN-qualified blending of matched SOURCE and child outputs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch


def blend_matched_outputs(
    source: torch.Tensor,
    child: torch.Tensor,
    *,
    alpha: float,
    source_sample_ids: Sequence[str],
    child_sample_ids: Sequence[str],
    source_support_id: str,
    child_support_id: str,
    source_checkpoint_sha256: str,
    child_checkpoint_sha256: str,
    source_coupling: Mapping,
    child_coupling: Mapping,
    qualification: Mapping,
) -> torch.Tensor:
    """Blend aligned outputs after checking the declared TRAIN qualification.

    ``alpha`` is the child contribution; 0 returns SOURCE and 1 the child.
    Qualification identifies both checkpoints, the TRAIN panel and its measured
    output cost. Callers perform inference and supply its matched outputs.
    """
    if not isinstance(source, torch.Tensor) or not isinstance(child, torch.Tensor):
        raise TypeError("source and child outputs must be tensors")
    if source.shape != child.shape or source.dtype != child.dtype or source.device != child.device:
        raise ValueError("outputs must have matching shape, dtype and device")
    if (
        source.ndim < 1
        or len(source_sample_ids) != source.shape[0]
        or tuple(source_sample_ids) != tuple(child_sample_ids)
    ):
        raise ValueError("outputs must have matching ordered sample IDs")
    if not source_support_id or source_support_id != child_support_id:
        raise ValueError("outputs must have matching support identity")
    if not torch.isfinite(source).all() or not torch.isfinite(child).all():
        raise ValueError("outputs must be finite")
    if isinstance(alpha, bool):
        raise TypeError("alpha must be numeric, not bool")
    keys = ("observations", "field_order", "latent_prior", "solver", "normalizer", "clamp")
    if not isinstance(source_coupling, Mapping) or not isinstance(child_coupling, Mapping):
        raise TypeError("matched coupling identities are required")
    if any(
        k not in source_coupling
        or not source_coupling[k]
        or source_coupling[k] != child_coupling.get(k)
        for k in keys
    ):
        raise ValueError("SOURCE and child coupling identities differ or are missing")
    if not isinstance(qualification, Mapping) or qualification.get("coupling") != {
        k: source_coupling[k] for k in keys if k != "observations"
    }:
        raise ValueError("qualification coupling differs from output coupling")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1]")
    if not isinstance(qualification, Mapping) or qualification.get("split") != "train":
        raise ValueError("a TRAIN qualification is required")
    for key in ("source_checkpoint_sha256", "child_checkpoint_sha256", "support_id", "sample_ids"):
        if not qualification.get(key):
            raise ValueError(f"qualification requires {key}")
    for key, identity in (
        ("source_checkpoint_sha256", source_checkpoint_sha256),
        ("child_checkpoint_sha256", child_checkpoint_sha256),
    ):
        if (
            not isinstance(identity, str)
            or len(identity) != 64
            or any(c not in "0123456789abcdef" for c in identity)
            or qualification[key] != identity
        ):
            raise ValueError(f"qualification {key} differs from output identity")
    ids = qualification["sample_ids"]
    if (
        not isinstance(ids, (list, tuple))
        or any(not isinstance(x, str) or not x for x in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("TRAIN qualification sample IDs must be a unique sequence of strings")
    if qualification["support_id"] != source_support_id:
        raise ValueError("qualification support differs from output support")
    if float(qualification.get("alpha", float("nan"))) != alpha:
        raise ValueError("alpha differs from TRAIN qualification")
    cost = qualification.get("cost")
    if not isinstance(cost, Mapping) or any(
        not math.isfinite(float(cost.get(k, float("nan")))) or float(cost[k]) < 0
        for k in ("source_seconds", "child_seconds")
    ):
        raise ValueError("qualification requires measured SOURCE and child cost in seconds")
    return torch.lerp(source, child, alpha)
