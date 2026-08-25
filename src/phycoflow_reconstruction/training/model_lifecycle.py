"""Optional model-owned training and evaluation lifecycle hooks.

Models that do not implement these hooks stay on the existing trainer paths.
The hooks let adapters own operations that cannot be expressed as a single
loss tensor, such as query-microbatched backward, without teaching the generic
trainers about individual model families.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
from torch import nn

from .gradients import adaptive_backward_and_clip_, stable_clip_grad_norm_


def backward_and_clip_model_loss(
    model: nn.Module,
    batch: Any,
    parameters: Iterable[nn.Parameter] | nn.Parameter,
    max_norm: float,
    *,
    initial_scale: float = 1.0,
    adaptive: bool = False,
    device: torch.device | str = "cpu",
    start_phase: Callable[[str], None] | None = None,
    end_phase: Callable[[str], None] | None = None,
) -> tuple[Any, torch.Tensor, float, int]:
    """Run either the unchanged loss closure or an optional model-owned backward.

    ``training_backward`` is deliberately a combined forward/backward hook: a
    query-microbatched implementation releases each query chunk after its own
    backward pass and therefore cannot return a live monolithic loss graph to
    the trainer. The trainer never calls ``backward`` a second time on this
    path.

    The optional path currently supports only the benchmark's exact scale-1,
    non-adaptive policy. A future adapter that needs retryable scaled backward
    should expose that capability through an explicit extension rather than
    silently weakening the common exact-RNG retry contract.
    """
    hook = getattr(model, "training_backward", None)
    if not callable(hook):
        return adaptive_backward_and_clip_(
            lambda: model.training_loss(batch),
            parameters,
            max_norm,
            initial_scale=initial_scale,
            adaptive=adaptive,
            device=device,
            start_phase=start_phase,
            end_phase=end_phase,
        )

    if adaptive:
        raise ValueError("model-owned training_backward does not support adaptive scaling")
    if float(initial_scale) != 1.0:
        raise ValueError("model-owned training_backward currently requires loss_scale=1")
    if (start_phase is None) != (end_phase is None):
        raise ValueError("start_phase and end_phase must be provided together")

    parameter_list = [parameters] if isinstance(parameters, nn.Parameter) else list(parameters)
    for parameter in parameter_list:
        parameter.grad = None
    losses = hook(
        batch,
        loss_scale=1.0,
        start_phase=start_phase,
        end_phase=end_phase,
    )
    if not bool(torch.isfinite(losses.total)):
        raise FloatingPointError(f"non-finite loss before optimizer step: {losses.total}")
    norm = stable_clip_grad_norm_(parameter_list, max_norm)
    return losses, norm, 1.0, 0


def after_optimizer_step(model: nn.Module) -> None:
    """Notify a model after one successful optimizer update, when supported."""
    hook = getattr(model, "after_optimizer_step", None)
    if callable(hook):
        hook()


def evaluation_weight_context(model: nn.Module) -> AbstractContextManager[Any]:
    """Return the model's temporary evaluation-weight context, or a no-op."""
    hook = getattr(model, "evaluation_weight_context", None)
    return hook() if callable(hook) else nullcontext()


def add_training_aux_state(payload: dict[str, Any], model: nn.Module) -> dict[str, Any]:
    """Add optional resumable model state without changing hookless payloads."""
    hook = getattr(model, "training_aux_state_dict", None)
    if not callable(hook):
        return payload
    state = hook()
    if not isinstance(state, Mapping):
        raise TypeError("training_aux_state_dict() must return a mapping")
    payload["training_aux_state"] = dict(state)
    return payload


def load_training_aux_state(model: nn.Module, payload: Mapping[str, Any]) -> None:
    """Strictly restore optional model lifecycle state from a checkpoint."""
    hook = getattr(model, "load_training_aux_state_dict", None)
    has_state = "training_aux_state" in payload
    if not callable(hook):
        if has_state:
            raise TypeError(
                "checkpoint contains training_aux_state but model cannot restore it"
            )
        return
    if not has_state:
        raise KeyError("checkpoint is missing required training_aux_state")
    state = payload["training_aux_state"]
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint training_aux_state must be a mapping")
    hook(state)
