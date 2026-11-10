"""Bounded, scalar-only measurements of actual and detached AdamW proposals."""
from __future__ import annotations

from copy import deepcopy
from time import perf_counter
from typing import Sequence

import torch
from torch import Tensor, nn

from .gradient_balance import _assign_flat_gradient
from .gradient_balance import _flat_gradient
from .gradients import stable_clip_grad_norm_


def component_gradients(components: dict[str, Tensor], parameters: Sequence[nn.Parameter],
                        family_gradient: Tensor) -> dict[str, Tensor]:
    """Measure the four already weighted live terms and verify their sum."""
    expected = {"signed", "joint", "self", "rho", "joint_attraction", "joint_separation"}
    if set(components) != expected:
        raise ValueError("component probe requires four terms and both signed joint parts")
    gradients = {name: _flat_gradient(value, list(parameters)).detach().double()
                 for name, value in components.items()}
    total = sum(gradients[name] for name in ("signed", "joint", "self", "rho"))
    def close(a, b):
        return bool(torch.linalg.vector_norm(a - b) <= 1e-9 + 2e-5 * torch.linalg.vector_norm(b))
    if not close(total, family_gradient.double()):
        raise FloatingPointError("weighted Scale component gradients do not sum to family gradient")
    if not close(gradients["joint_attraction"] + gradients["joint_separation"], gradients["joint"]):
        raise FloatingPointError("signed joint attraction and separation gradients do not sum")
    return gradients


def snapshot(parameters: Sequence[nn.Parameter]) -> list[Tensor]:
    return [p.detach().clone() for p in parameters]


def displacement(parameters: Sequence[nn.Parameter], before: Sequence[Tensor]) -> Tensor:
    if len(parameters) != len(before):
        raise ValueError("parameter snapshot length differs")
    return torch.cat([(p.detach() - b).reshape(-1).double() for p, b in zip(parameters, before)])


def projected_benefits(gradients: Tensor, delta: Tensor, scales: Sequence[float]) -> list[float]:
    s = gradients.new_tensor(scales, dtype=torch.float64)
    if s.shape != (gradients.shape[0],) or not bool(torch.isfinite(s).all()) or bool((s <= 0).any()):
        raise ValueError("projection scales must be finite and positive")
    values = -(gradients.double() @ delta.double()) / s
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("nonfinite actual-step projection")
    return values.cpu().tolist()


def shadow_adamw(optimizer: torch.optim.Optimizer, parameters: Sequence[nn.Parameter],
                 direction: Tensor, grad_clip: float | None) -> tuple[Tensor, float]:
    """Replay a proposal with copied parameters, optimizer state and global clipping.

    AdamW itself performs all FP32 arithmetic/rounding. No model, optimizer or RNG
    state is modified. Parameter order must match the optimizer group order.
    """
    if type(optimizer) is not torch.optim.AdamW:
        raise TypeError("same-state shadows require torch.optim.AdamW")
    original = [p for group in optimizer.param_groups for p in group['params']]
    if len(original) != len(parameters) or any(a is not b for a, b in zip(original, parameters)):
        raise ValueError("shadow parameter order must equal optimizer group order")
    start = perf_counter()
    clones = [nn.Parameter(p.detach().clone(), requires_grad=True) for p in parameters]
    groups = []
    offset = 0
    for group in optimizer.param_groups:
        n = len(group['params'])
        settings = {k: deepcopy(v) for k, v in group.items() if k != 'params'}
        groups.append({**settings, 'params': clones[offset:offset+n]})
        offset += n
    shadow = torch.optim.AdamW(groups)
    shadow.load_state_dict(deepcopy(optimizer.state_dict()))
    before = snapshot(clones)
    _assign_flat_gradient(clones, direction)
    if grad_clip is not None and float(grad_clip) > 0:
        stable_clip_grad_norm_(clones, float(grad_clip))
    shadow.step()
    return displacement(clones, before), perf_counter() - start
