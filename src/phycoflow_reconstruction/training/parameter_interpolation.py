"""Strict LIVE parameter interpolation for inference-only source refinements."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def interpolate_live_parameters(
    source: Mapping[str, Any],
    child: Mapping[str, Any],
    *,
    trainable_names: Sequence[str],
    alpha: float,
    source_semantics: Mapping[str, Any],
    child_semantics: Mapping[str, Any],
) -> dict[str, Any]:
    """Blend only declared trainable tensors; fail on changed frozen state.

    The caller obtains ``trainable_names`` from the unchanged model's named
    parameters. Semantic identities include fields, prior, coordinates,
    normalization and descriptor calibrations. No EMA or optimizer is accepted.
    Endpoint branches clone their inputs exactly, avoiding rounding at 0/1.
    """
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("interpolation alpha must be finite and in [0,1]")
    if dict(source_semantics) != dict(child_semantics):
        raise ValueError("source/child semantic identities disagree")
    if source.keys() != child.keys():
        raise ValueError("source/child LIVE state keys disagree")
    names = tuple(trainable_names)
    if len(set(names)) != len(names) or not names or set(names) - source.keys():
        raise ValueError("trainable parameter layout is empty, duplicate or missing")
    trainable = set(names)
    result = {}
    for name, left in source.items():
        right = child[name]
        if isinstance(left, torch.Tensor):
            if not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype:
                raise ValueError(f"incompatible LIVE tensor layout: {name}")
            right = right.to(left.device)
            if name in trainable:
                if not (left.is_floating_point() or left.is_complex()):
                    raise ValueError(f"trainable tensor must be real or complex: {name}")
                if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
                    raise ValueError(f"nonfinite trainable tensor: {name}")
                value = left if alpha == 0 else right if alpha == 1 else (1 - alpha) * left + alpha * right
                result[name] = value.detach().clone()
            else:
                if not torch.equal(left, right):
                    raise ValueError(f"frozen parameter/buffer differs: {name}")
                result[name] = left.detach().clone()
        else:
            if name in trainable or left != right:
                raise ValueError(f"non-tensor LIVE metadata differs: {name}")
            result[name] = copy.deepcopy(left)
    return result


def inference_only_checkpoint(
    live_state: Mapping[str, Any],
    *,
    normalization: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an evaluator-compatible payload without any training recovery."""
    required = {"alpha", "source_checkpoint_sha256", "child_checkpoint_sha256",
                "source_parameter_sha256", "child_parameter_sha256", "semantics"}
    if not required <= provenance.keys() or not 0 < float(provenance["alpha"]) <= 1:
        raise ValueError("nonzero blend with complete immutable provenance required")
    return {"model": copy.deepcopy(dict(live_state)),
            "normalization": copy.deepcopy(dict(normalization)),
            "inference_only": True, "checkpoint_kind": "source_parameter_interpolation_v1",
            "interpolation": copy.deepcopy(dict(provenance))}


def require_training_checkpoint(payload: Mapping[str, Any]) -> None:
    """Reject deploy-only blends before constructing or restoring an optimizer."""
    if payload.get("inference_only") or payload.get("checkpoint_kind") == "source_parameter_interpolation_v1":
        raise ValueError("inference-only interpolated checkpoint cannot resume training")
