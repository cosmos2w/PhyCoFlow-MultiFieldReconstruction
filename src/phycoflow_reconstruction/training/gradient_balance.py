"""Two-objective weighted-sum and optional ConFIG optimizer updates."""

from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import nn

from .gradients import stable_clip_grad_norm_


def _trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("post-training selected no trainable parameters")
    return parameters


def _flat_gradient(loss: torch.Tensor, parameters: list[nn.Parameter]) -> torch.Tensor:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    flattened = []
    for parameter, gradient in zip(parameters, gradients):
        value = torch.zeros_like(parameter) if gradient is None else gradient
        if torch.is_complex(value):
            value = torch.view_as_real(value)
        flattened.append(value.reshape(-1))
    return torch.cat(flattened)


def _assign_flat_gradient(parameters: list[nn.Parameter], gradient: torch.Tensor) -> None:
    offset = 0
    for parameter in parameters:
        count = parameter.numel() * (2 if torch.is_complex(parameter) else 1)
        value = gradient[offset : offset + count]
        if torch.is_complex(parameter):
            real_dtype = parameter.real.dtype
            value = torch.view_as_complex(
                value.to(real_dtype).clone().reshape(*parameter.shape, 2).contiguous()
            )
        else:
            value = value.to(parameter.dtype).view_as(parameter)
        parameter.grad = value.clone()
        offset += count
    if offset != gradient.numel():
        raise ValueError("combined gradient length does not match trainable parameters")


def data_only_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    *,
    weight: float,
    grad_clip: float | None,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    (float(weight) * loss).backward()
    norm = (
        stable_clip_grad_norm_(model.parameters(), grad_clip)
        if grad_clip
        else torch.tensor(float("nan"))
    )
    optimizer.step()
    return {
        "update_mode": "data_only",
        "data_grad_norm": float(norm),
        "coherence_grad_norm": float("nan"),
        "gradient_cosine": float("nan"),
        "gradient_conflict": False,
        "combined_grad_norm": float(norm),
        "config_fallback_used": False,
    }


def two_objective_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loss: torch.Tensor,
    coherence_loss: torch.Tensor,
    *,
    mode: str,
    data_weight: float,
    coherence_weight: float,
    grad_clip: float | None,
    config_missing_behavior: str = "error",
) -> dict[str, Any]:
    """Update once while recording the relationship between both gradients."""
    parameters = _trainable_parameters(model)
    weighted_data = float(data_weight) * data_loss
    weighted_coherence = float(coherence_weight) * coherence_loss
    data_gradient = _flat_gradient(weighted_data, parameters)
    coherence_gradient = _flat_gradient(weighted_coherence, parameters)
    if not torch.isfinite(data_gradient).all() or not torch.isfinite(coherence_gradient).all():
        raise FloatingPointError("post-training objective gradient contains non-finite values")
    data_norm = torch.linalg.vector_norm(data_gradient)
    coherence_norm = torch.linalg.vector_norm(coherence_gradient)
    denominator = (data_norm * coherence_norm).clamp_min(1e-12)
    cosine = torch.dot(data_gradient, coherence_gradient) / denominator
    weighted_sum = data_gradient + coherence_gradient
    selected = weighted_sum
    update_mode = "weighted_sum"
    fallback = False

    normalized_mode = str(mode).lower()
    if normalized_mode not in {"weighted_sum", "config"}:
        raise ValueError("gradient balance mode must be weighted_sum or config")
    if normalized_mode == "config" and float(coherence_norm) > 0:
        try:
            from conflictfree.grad_operator import ConFIG_update
        except ImportError as error:
            if config_missing_behavior != "weighted_sum":
                raise ImportError(
                    "gradient balance mode=config requires the optional conflictfree package"
                ) from error
            warnings.warn(
                "conflictfree unavailable; using weighted_sum", RuntimeWarning, stacklevel=2
            )
            fallback = True
            update_mode = "weighted_sum_missing_config"
        else:
            if float(cosine) >= 0:
                update_mode = "weighted_sum_aligned"
            else:
                candidate = ConFIG_update([data_gradient, coherence_gradient])
                descends_data = torch.dot(candidate, data_gradient) > 0
                descends_coherence = torch.dot(candidate, coherence_gradient) > 0
                if torch.isfinite(candidate).all() and descends_data and descends_coherence:
                    selected = candidate
                    update_mode = "config"
                else:
                    fallback = True
                    update_mode = "weighted_sum_nondescent_config"

    optimizer.zero_grad(set_to_none=True)
    _assign_flat_gradient(parameters, selected)
    if grad_clip:
        stable_clip_grad_norm_(parameters, float(grad_clip))
    optimizer.step()
    return {
        "update_mode": update_mode,
        "data_grad_norm": float(data_norm.detach().cpu()),
        "coherence_grad_norm": float(coherence_norm.detach().cpu()),
        "gradient_cosine": float(cosine.detach().cpu()),
        "gradient_conflict": bool(float(cosine.detach().cpu()) < 0),
        "combined_grad_norm": float(torch.linalg.vector_norm(selected).detach().cpu()),
        "config_fallback_used": fallback,
    }
