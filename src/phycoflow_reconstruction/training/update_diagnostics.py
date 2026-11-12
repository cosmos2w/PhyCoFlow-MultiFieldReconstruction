"""Optional scalar diagnostics of objectives against an actual parameter update."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class UpdateProbe:
    """Capture live objective gradients once, then compare with the actual step.

    ``objectives`` may be family losses or coherent semantic channels, including
    a fair score's attraction and separation together. Units are fixed positive
    TRAIN scales. This probe performs no optimizer update or model forward.
    Flat snapshots are retained only until ``measure``; enable at sparse chosen
    updates rather than keeping one probe for every recorded training row.
    """

    parameters: tuple[nn.Parameter, ...]
    before: tuple[Tensor, ...]
    gradients: dict[str, Tensor]
    units: dict[str, float]

    @classmethod
    def capture(
        cls,
        objectives: Mapping[str, Tensor],
        parameters: Sequence[nn.Parameter],
        *,
        units: Mapping[str, float] | None = None,
    ) -> UpdateProbe:
        import math

        parameters = tuple(parameters)
        if not parameters or not objectives:
            raise ValueError("objectives and trainable parameters must be nonempty")
        if any(not p.requires_grad for p in parameters):
            raise ValueError("probe parameter set must be trainable")
        resolved = {key: float(1.0 if units is None else units[key]) for key in objectives}
        if any(not math.isfinite(value) or value <= 0 for value in resolved.values()):
            raise ValueError("probe units must be finite and positive")
        gradients = {}
        for name, loss in objectives.items():
            if loss.numel() != 1 or not torch.isfinite(loss).all():
                raise ValueError("each probe objective must be one finite scalar")
            raw = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            gradients[name] = torch.cat(
                [
                    (torch.zeros_like(p) if g is None else g).detach().reshape(-1).double()
                    for p, g in zip(parameters, raw)
                ]
            )
        return cls(parameters, tuple(p.detach().clone() for p in parameters), gradients, resolved)

    def measure(self) -> dict[str, float]:
        """Report negative gradient-dot-step in declared units; consume the probe.

        A positive value is local first-order improvement for this particular
        update. It is not a measured finite loss change or a convergence claim.
        Momentum, clipping and weight decay are included in the displacement.
        """
        if not self.before:
            raise RuntimeError("update probe already consumed")
        delta = torch.cat(
            [
                (p.detach() - before).reshape(-1).double()
                for p, before in zip(self.parameters, self.before)
            ]
        )
        if not torch.isfinite(delta).all():
            raise FloatingPointError("nonfinite parameter displacement")
        norm = float(delta.norm())
        result = {"parameter_displacement_norm": norm}
        for name, gradient in self.gradients.items():
            if not torch.isfinite(gradient).all():
                raise FloatingPointError("nonfinite objective gradient")
            size = float(gradient.norm())
            benefit = float(-(gradient @ delta))
            result[f"{name}/gradient_norm"] = size
            result[f"{name}/first_order_benefit"] = benefit / self.units[name]
            result[f"{name}/descent_cosine"] = benefit / (size * norm) if size * norm else 0.0
        self.before = ()
        self.gradients.clear()
        return result
