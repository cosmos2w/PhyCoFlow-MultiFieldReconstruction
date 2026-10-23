"""Default-off LPC-ordered scalar objective and fixed dual state.

This helper only reorders the existing A/B/T scalar losses. It does not alter
descriptor definitions or the endpoint fidelity policy. The A/B constraints
are relative non-regression constraints against exact SOURCE means; numerical
floors affect scaling only and never relax those bounds.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

_FAMILY_NAMES = ("global_distribution", "cross_spectrum", "topology")
_STATE_VERSION = 1


def _finite_positive(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def validate_lpc_ordered(config: Mapping[str, Any]) -> None:
    """Validate the narrow, optional scalar profile used by LPC ordering.

    The absent (or explicit ``None``) setting is a true no-op so existing
    post-training configurations retain their prior validation and behavior.
    """
    if not isinstance(config, Mapping):
        raise TypeError("configuration must be a mapping")
    optimization = config.get("optimization")
    if not isinstance(optimization, Mapping):
        return
    settings = optimization.get("lpc_ordered")
    if settings is None:
        return
    if not isinstance(settings, Mapping) or set(settings) - {"dual_lr", "denominator_floor"}:
        raise ValueError("optimization.lpc_ordered accepts only dual_lr and denominator_floor")
    _finite_positive(settings.get("dual_lr", 0.01), name="optimization.lpc_ordered.dual_lr")
    _finite_positive(
        settings.get("denominator_floor", 1.0e-8),
        name="optimization.lpc_ordered.denominator_floor",
    )

    multitask = optimization.get("multitask")
    if not isinstance(multitask, Mapping):
        raise TypeError("LPC-ordered requires optimization.multitask")
    scalarization = multitask.get("family_scalarization")
    if not isinstance(scalarization, Mapping) or scalarization.get("method") != "fixed_sum":
        raise ValueError("LPC-ordered requires family_scalarization.method=fixed_sum")
    if "coherence_aggregation" in multitask:
        raise ValueError("LPC-ordered requires scalar family losses without internal aggregation")

    external = multitask.get("external_policy")
    if not isinstance(external, Mapping) or external.get("method") != "lagrangian":
        raise ValueError("LPC-ordered requires the dense external Lagrangian policy")
    lagrangian = external.get("lagrangian", {})
    if not isinstance(lagrangian, Mapping):
        raise TypeError("external_policy.lagrangian must be a mapping")
    interval = lagrangian.get("penalty_interval_updates", 1)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval != 1:
        raise ValueError("LPC-ordered requires dense Lagrangian penalties on every update")

    fast = optimization.get("fast_lagrangian")
    if not isinstance(fast, Mapping):
        raise TypeError("LPC-ordered requires fast_lagrangian with all families evaluated")
    probability = fast.get("family_probability", 1.0)
    coefficient_interval = fast.get("coefficient_interval", 1)
    if isinstance(probability, bool) or probability != 1.0:
        raise ValueError("LPC-ordered requires family_probability=1")
    if (
        isinstance(coefficient_interval, bool)
        or not isinstance(coefficient_interval, int)
        or coefficient_interval != 1
    ):
        raise ValueError("LPC-ordered does not support periodic coefficient reuse")

    coherence = config.get("coherence", {})
    if not isinstance(coherence, Mapping):
        raise TypeError("coherence must be a mapping")
    schedule = coherence.get("schedule", {})
    if not isinstance(schedule, Mapping):
        raise TypeError("coherence.schedule must be a mapping")
    start_epoch = schedule.get("start_epoch", 1)
    every_n_steps = schedule.get("every_n_steps", 1)
    warmup_epochs = schedule.get("weight_warmup_epochs", 0)
    if (
        isinstance(start_epoch, bool)
        or start_epoch != 1
        or isinstance(every_n_steps, bool)
        or every_n_steps != 1
        or isinstance(warmup_epochs, bool)
        or warmup_epochs != 0
        or schedule.get("interval_rescale", False) is not False
    ):
        raise ValueError("LPC-ordered requires unscheduled, every-step family losses")
    family_balance = coherence.get("family_balance", {})
    if not isinstance(family_balance, Mapping):
        raise TypeError("coherence.family_balance must be a mapping")
    if family_balance.get("mode", "none") != "none":
        raise ValueError("LPC-ordered does not support dynamic family calibration")
    diagnostic_interval = family_balance.get("gradient_diagnostics_every_epochs", 0)
    if (
        isinstance(diagnostic_interval, bool)
        or not isinstance(diagnostic_interval, (int, float))
        or not math.isfinite(float(diagnostic_interval))
        or float(diagnostic_interval) != 0.0
    ):
        raise ValueError("LPC-ordered requires fixed family scales without scheduled diagnostics")


@dataclass(frozen=True)
class LPCOrderedReference:
    """Immutable SOURCE A/B/T means, scaling floors, and source identity."""

    a0: float
    b0: float
    t0: float
    a_denominator: float
    b_denominator: float
    t_denominator: float
    denominator_floor: float
    source_sha256: str

    @classmethod
    def from_means(
        cls,
        family_means: Mapping[str, Tensor | float],
        *,
        source_sha256: str,
        denominator_floor: float = 1.0e-8,
    ) -> LPCOrderedReference:
        floor = _finite_positive(denominator_floor, name="denominator_floor")
        if not isinstance(family_means, Mapping) or set(family_means) != set(_FAMILY_NAMES):
            raise ValueError(f"family_means must contain exactly {_FAMILY_NAMES}")
        if not isinstance(source_sha256, str) or len(source_sha256) != 64:
            raise ValueError("source_sha256 must be a 64-character SHA256 hex digest")
        try:
            int(source_sha256, 16)
        except ValueError as error:
            raise ValueError("source_sha256 must be a 64-character SHA256 hex digest") from error

        means: list[float] = []
        for name in _FAMILY_NAMES:
            value = family_means[name]
            if isinstance(value, Tensor):
                if value.numel() != 1 or not value.is_floating_point():
                    raise TypeError(f"SOURCE mean {name!r} must be a real scalar")
                value = float(value.detach().to(device="cpu", dtype=torch.float64).reshape(()))
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"SOURCE mean {name!r} must be a real scalar")
            else:
                value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"SOURCE mean {name!r} must be finite and non-negative")
            means.append(value)
        a0, b0, t0 = means
        return cls(
            a0=a0,
            b0=b0,
            t0=t0,
            a_denominator=max(a0, floor),
            b_denominator=max(b0, floor),
            t_denominator=max(t0, floor),
            denominator_floor=floor,
            source_sha256=source_sha256.lower(),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": _STATE_VERSION,
            "family_names": list(_FAMILY_NAMES),
            "means": {"global_distribution": self.a0, "cross_spectrum": self.b0, "topology": self.t0},
            "denominators": {
                "global_distribution": self.a_denominator,
                "cross_spectrum": self.b_denominator,
                "topology": self.t_denominator,
            },
            "denominator_floor": self.denominator_floor,
            "source_sha256": self.source_sha256,
            # Bounds are the exact SOURCE means; floors never raise them.
            "no_regression_bounds": {"global_distribution": self.a0, "cross_spectrum": self.b0},
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> LPCOrderedReference:
        if not isinstance(state, Mapping):
            raise TypeError("LPC-ordered reference checkpoint must be a mapping")
        if isinstance(state.get("version"), bool) or state.get("version") != _STATE_VERSION:
            raise ValueError("unsupported LPC-ordered reference state")
        if tuple(state.get("family_names", ())) != _FAMILY_NAMES:
            raise ValueError("LPC-ordered reference family identity or order changed")
        means = state.get("means")
        if not isinstance(means, Mapping) or set(means) != set(_FAMILY_NAMES):
            raise ValueError("LPC-ordered reference means are invalid")
        reference = cls.from_means(
            means,
            source_sha256=state.get("source_sha256"),
            denominator_floor=state.get("denominator_floor"),
        )
        if reference.state_dict() != dict(state):
            raise ValueError("LPC-ordered reference has inconsistent fixed bounds or denominators")
        return reference


class LPCOrderedState:
    """Detached nonnegative A/B duals with an accepted-update recovery cursor."""

    def __init__(
        self,
        reference: LPCOrderedReference,
        *,
        dual_lr: float = 0.01,
        device: torch.device | str = "cpu",
    ) -> None:
        if not isinstance(reference, LPCOrderedReference):
            raise TypeError("reference must be LPCOrderedReference")
        self.reference = reference
        self.dual_lr = _finite_positive(dual_lr, name="dual_lr")
        self.device = torch.device(device)
        self.mu_a = 0.0
        self.mu_b = 0.0
        self.completed_updates = 0

    @staticmethod
    def _checked_family_losses(family_losses: Mapping[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(family_losses, Mapping) or tuple(family_losses) != _FAMILY_NAMES:
            raise ValueError(f"family_losses must be ordered as {_FAMILY_NAMES}")
        values: list[Tensor] = []
        for name in _FAMILY_NAMES:
            loss = family_losses[name]
            if not isinstance(loss, Tensor) or loss.numel() != 1 or not loss.is_floating_point():
                raise TypeError(f"family loss {name!r} must be a real scalar tensor")
            scalar = loss.reshape(())
            if not scalar.requires_grad:
                raise ValueError(f"family loss {name!r} must retain its current gradient graph")
            if not bool(torch.isfinite(scalar.detach())):
                raise FloatingPointError(f"family loss {name!r} is non-finite")
            if float(scalar.detach()) < 0.0:
                raise ValueError(f"family loss {name!r} must be non-negative")
            values.append(scalar)
        if any(value.device != values[0].device for value in values[1:]):
            raise ValueError("family losses must share a device")
        return values[0], values[1], values[2]

    def _violations(self, family_losses: Mapping[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        a, b, t = self._checked_family_losses(family_losses)
        if a.device.type != self.device.type or (
            self.device.index is not None and a.device.index != self.device.index
        ):
            raise ValueError("family losses must be on the LPC-ordered state device")
        a0 = a.new_tensor(self.reference.a0, dtype=torch.float64)
        b0 = b.new_tensor(self.reference.b0, dtype=torch.float64)
        da = a.new_tensor(self.reference.a_denominator, dtype=torch.float64)
        db = b.new_tensor(self.reference.b_denominator, dtype=torch.float64)
        dt = t.new_tensor(self.reference.t_denominator, dtype=torch.float64)
        return (a.to(torch.float64) - a0) / da, (b.to(torch.float64) - b0) / db, t.to(torch.float64) / dt

    def primary(
        self,
        family_losses: Mapping[str, Tensor],
        *,
        objective_scale: float,
    ) -> Tensor:
        """Return ``C0 * (T/T0 + mu_A h_A + mu_B h_B)`` in float64.

        The external fidelity policy divides this scalar by the same C0 once.
        Multipliers are Python floats and are therefore detached from autograd.
        """
        scale = _finite_positive(objective_scale, name="objective_scale")
        h_a, h_b, normalized_t = self._violations(family_losses)
        objective = normalized_t + self.mu_a * h_a + self.mu_b * h_b
        return objective * objective.new_tensor(scale, dtype=torch.float64)

    def accept(self, family_losses: Mapping[str, Tensor]) -> dict[str, Any]:
        """Project A/B multipliers after one accepted model action."""
        h_a, h_b, _ = self._violations(family_losses)
        violations = (float(h_a.detach().cpu()), float(h_b.detach().cpu()))
        if not all(math.isfinite(value) for value in violations):
            raise FloatingPointError("LPC-ordered dual violations are non-finite")
        before = {"global_distribution": self.mu_a, "cross_spectrum": self.mu_b}
        after = {
            "global_distribution": max(0.0, self.mu_a + self.dual_lr * violations[0]),
            "cross_spectrum": max(0.0, self.mu_b + self.dual_lr * violations[1]),
        }
        if not all(math.isfinite(value) for value in after.values()):
            raise FloatingPointError("LPC-ordered dual update is non-finite")
        self.mu_a = after["global_distribution"]
        self.mu_b = after["cross_spectrum"]
        self.completed_updates += 1
        return {
            "multipliers_before": before,
            "multipliers_after": dict(after),
            "violations": {
                "global_distribution": violations[0],
                "cross_spectrum": violations[1],
            },
            "accepted_update": self.completed_updates,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": _STATE_VERSION,
            "reference": self.reference.state_dict(),
            "dual_lr": self.dual_lr,
            "multipliers": {
                "global_distribution": self.mu_a,
                "cross_spectrum": self.mu_b,
            },
            "completed_updates": self.completed_updates,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("LPC-ordered checkpoint state must be a mapping")
        if isinstance(state.get("version"), bool) or state.get("version") != _STATE_VERSION:
            raise ValueError("unsupported LPC-ordered checkpoint state")
        if state.get("reference") != self.reference.state_dict():
            raise ValueError("LPC-ordered checkpoint SOURCE reference does not match")
        saved_lr = state.get("dual_lr")
        if isinstance(saved_lr, bool) or not isinstance(saved_lr, (int, float)):
            raise TypeError("LPC-ordered checkpoint dual_lr is invalid")
        if not math.isfinite(float(saved_lr)) or float(saved_lr) != self.dual_lr:
            raise ValueError("LPC-ordered checkpoint dual_lr does not match")
        multipliers = state.get("multipliers")
        if not isinstance(multipliers, Mapping) or set(multipliers) != {
            "global_distribution", "cross_spectrum"
        }:
            raise ValueError("LPC-ordered checkpoint multipliers are invalid")
        mu_a = multipliers["global_distribution"]
        mu_b = multipliers["cross_spectrum"]
        values = (mu_a, mu_b)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ValueError("LPC-ordered checkpoint multipliers must be numeric")
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values):
            raise ValueError("LPC-ordered checkpoint multipliers must be finite and non-negative")
        completed = state.get("completed_updates")
        if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
            raise ValueError("LPC-ordered checkpoint completed_updates is invalid")
        self.mu_a, self.mu_b = float(mu_a), float(mu_b)
        self.completed_updates = completed


def implementation_binding() -> dict[str, str]:
    """Hash the ordered helper and its trainer/config/recovery integration."""
    package = Path(__file__).resolve().parents[1]
    repo = package.parents[1]
    paths = (
        package / "training/lpc_ordered.py",
        package / "training/post_training.py",
        package / "training/run_store.py",
        package / "config/validate.py",
    )
    return {
        str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
