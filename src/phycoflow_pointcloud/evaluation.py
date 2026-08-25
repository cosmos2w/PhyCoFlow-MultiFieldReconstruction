"""Dataset-independent schema and numerical comparison utilities."""

from __future__ import annotations

import hashlib

import torch
from torch import nn


def model_schema_digest(model: nn.Module) -> str:
    """Hash state keys, dtypes, and shapes without depending on parameter values."""
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
    return digest.hexdigest()


def tensor_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    """Return portable absolute and relative L2 equivalence diagnostics."""
    delta = candidate.detach() - reference.detach()
    return {
        "max_abs": float(delta.abs().max().cpu()),
        "relative_l2": float(
            torch.linalg.vector_norm(delta)
            / torch.linalg.vector_norm(reference.detach()).clamp_min(1.0e-12)
        ),
    }


__all__ = ["model_schema_digest", "tensor_error"]
