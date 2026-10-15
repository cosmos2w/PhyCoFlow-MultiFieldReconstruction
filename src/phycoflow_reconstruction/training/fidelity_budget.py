"""Pooled endpoint fidelity budgets and external policy update math."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .gradients import stable_clip_grad_norm_


def field_error_sums(prediction: Tensor, target: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Pool masked squared errors and counts over all leading dimensions, in field order."""
    if not all(isinstance(value, Tensor) for value in (prediction, target, mask)):
        raise TypeError("prediction, target, and mask must be tensors")
    if prediction.ndim < 2 or prediction.shape[-1] < 1:
        raise ValueError("prediction must have sample/query dimensions and a nonempty field dimension")
    if any(not (value.is_floating_point() or value.is_complex()) for value in (prediction, target)):
        raise TypeError("prediction and target must have floating or complex dtypes")
    if prediction.device != target.device or mask.device != prediction.device:
        raise ValueError("prediction, target, and mask must share a device")
    try:
        target = torch.broadcast_to(target, prediction.shape)
    except RuntimeError as error:
        raise ValueError("target must broadcast to prediction shape") from error

    if mask.ndim == prediction.ndim and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    try:
        mask = torch.broadcast_to(mask, prediction.shape[:-1]).unsqueeze(-1)
    except RuntimeError:
        try:
            mask = torch.broadcast_to(mask, prediction.shape)
        except RuntimeError as error:
            raise ValueError("mask must broadcast to query positions or prediction fields") from error
    if mask.is_complex():
        raise TypeError("mask must be real-valued")
    if mask.is_floating_point() and not bool(torch.isfinite(mask.detach()).all()):
        raise FloatingPointError("mask contains non-finite values")
    if mask.dtype != torch.bool and bool(((mask != 0) & (mask != 1)).any()):
        raise ValueError("mask must contain only boolean or zero/one values")

    valid = torch.broadcast_to(mask.to(dtype=torch.bool), prediction.shape)
    # Mask before nonlinear arithmetic so invalid padding cannot produce a
    # zero-times-NaN derivative during the model backward.
    difference = torch.where(valid, prediction - target.detach(), torch.zeros_like(prediction))
    error = difference.abs().square() if difference.is_complex() else difference.square()
    dimensions = tuple(range(prediction.ndim - 1))
    return (
        error.to(torch.float64).sum(dim=dimensions),
        valid.to(torch.float64).sum(dim=dimensions),
    )


def field_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Return pooled masked MSE per field; inputs are already in model units."""
    sums, counts = field_error_sums(prediction, target, mask)
    if bool((counts <= 0).any()):
        raise ValueError("every field must have at least one valid endpoint reference")
    risks = sums / counts
    if not bool(torch.isfinite(risks.detach()).all()):
        raise FloatingPointError("field endpoint risk is non-finite")
    return risks


def reduce_primary(primary: Tensor | Mapping[str, Tensor]) -> Tensor:
    """Explicitly sum the scalarized family mapping for an external policy."""
    values = list(primary.values()) if isinstance(primary, Mapping) else [primary]
    if not values:
        raise ValueError("primary objective must not be empty")
    for value in values:
        if not isinstance(value, Tensor) or value.numel() != 1 or not value.is_floating_point():
            raise TypeError("primary objectives must be real scalar tensors")
        if not bool(torch.isfinite(value.detach()).all()):
            raise FloatingPointError("primary objective is non-finite")
    result = sum((value.reshape(()) for value in values), values[0].new_zeros(()))
    return result


def extended_log_barrier(violation: Tensor, t: float) -> Tensor:
    """Finite extended-log-barrier value with a safe logarithm on inactive branches."""
    if not math.isfinite(float(t)) or t <= 0 or not violation.is_floating_point():
        raise ValueError("barrier requires a real tensor and finite t > 0")
    boundary = -1.0 / (float(t) * float(t))
    logarithmic = violation <= boundary
    argument = torch.where(logarithmic, -violation, torch.ones_like(violation))
    log_value = -torch.log(argument) / t
    linear_value = t * violation + (2.0 * math.log(t) + 1.0) / t
    return torch.where(logarithmic, log_value, linear_value)


def extended_log_barrier_derivative(violation: Tensor, t: float) -> Tensor:
    """Derivative of the extended-log-barrier with respect to each violation."""
    boundary = -1.0 / (float(t) * float(t))
    logarithmic = violation <= boundary
    safe_violation = torch.where(logarithmic, violation, -torch.ones_like(violation))
    return torch.where(logarithmic, -1.0 / (t * safe_violation), torch.full_like(violation, t))


def barrier_schedule(
    completed_updates: int,
    *,
    t_initial: float = 1.0,
    growth_factor: float = 1.1,
    update_interval: int = 100,
    t_max: float = 100.0,
) -> float:
    """Capped update-based continuation schedule, safe even for huge update counts."""
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (completed_updates, update_interval)):
        raise TypeError("completed_updates and update_interval must be integers")
    if completed_updates < 0 or update_interval < 1 or growth_factor < 1:
        raise ValueError("invalid barrier schedule settings")
    if not all(math.isfinite(float(value)) for value in (t_initial, growth_factor, t_max)):
        raise ValueError("barrier schedule settings must be finite")
    if t_initial <= 0 or t_max < t_initial:
        raise ValueError("barrier schedule requires 0 < t_initial <= t_max")
    intervals = completed_updates // update_interval
    if t_initial == t_max or growth_factor == 1:
        return min(t_initial, t_max)
    log_initial, log_growth, log_max = math.log(t_initial), math.log(growth_factor), math.log(t_max)
    if intervals >= (log_max - log_initial) / log_growth:
        return t_max
    return math.exp(log_initial + intervals * log_growth)


@dataclass(frozen=True)
class FidelityBudget:
    """Fixed SOURCE risks, per-field limits, and objective scale needed for recovery."""

    field_names: tuple[str, ...]
    reference_risks: tuple[float, ...]
    denominators: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    relative_increase: float
    absolute_allowance: float
    denominator_floor: float
    denominator_floor_activated: tuple[bool, ...]
    c0: float
    objective_scale: float
    objective_floor_activated: bool

    @classmethod
    def from_reference(
        cls,
        field_names: Sequence[str],
        reference_risks: Tensor | Sequence[float],
        c0: Tensor | float,
        *,
        relative_increase: float = 0.05,
        absolute_allowance: float = 0.0,
        denominator_floor: float = 1.0e-8,
    ) -> FidelityBudget:
        names = tuple(field_names)
        values = torch.as_tensor(reference_risks, dtype=torch.float64).detach().cpu().reshape(-1)
        c0_value = float(c0.detach().cpu()) if isinstance(c0, Tensor) else float(c0)
        epsilon, delta, floor = map(float, (relative_increase, absolute_allowance, denominator_floor))
        if not names or len(set(names)) != len(names) or values.numel() != len(names):
            raise ValueError("field names and reference risks must have matching nonempty order")
        if any(not isinstance(name, str) or not name for name in names) or not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
            raise ValueError("reference risks and field names must be valid")
        if not math.isfinite(epsilon + delta + floor + c0_value) or epsilon < 0 or delta < 0 or floor <= 0 or c0_value < 0:
            raise ValueError("budget settings and C0 must be finite and non-negative; floor must be positive")
        references = tuple(float(value) for value in values)
        denominators = tuple(max(value, floor) for value in references)
        # The scaling floor never appears in the accepted upper bound.
        bounds = tuple((1.0 + epsilon) * value + delta for value in references)
        if not all(math.isfinite(value) for value in (*denominators, *bounds)):
            raise ValueError("derived fidelity bounds must be finite")
        return cls(
            names,
            references,
            denominators,
            bounds,
            epsilon,
            delta,
            floor,
            tuple(value < floor for value in references),
            c0_value,
            max(c0_value, 1.0e-8),
            c0_value < 1.0e-8,
        )

    def violations(self, risks: Tensor) -> Tensor:
        if risks.numel() != len(self.field_names) or not risks.is_floating_point() or risks.is_complex():
            raise ValueError("risks must be a real floating vector matching field_names")
        risks = risks.reshape(-1).to(dtype=torch.float64)
        if not bool(torch.isfinite(risks.detach()).all()):
            raise FloatingPointError("field endpoint risk is non-finite")
        if bool((risks.detach() < 0).any()):
            raise ValueError("field endpoint risks must be non-negative")
        return (risks - risks.new_tensor(self.upper_bounds)) / risks.new_tensor(self.denominators)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "field_names": list(self.field_names),
            "reference_risks": list(self.reference_risks),
            "denominators": list(self.denominators),
            "upper_bounds": list(self.upper_bounds),
            "relative_increase": self.relative_increase,
            "absolute_allowance": self.absolute_allowance,
            "denominator_floor": self.denominator_floor,
            "denominator_floor_activated": list(self.denominator_floor_activated),
            "c0": self.c0,
            "objective_scale": self.objective_scale,
            "objective_floor_activated": self.objective_floor_activated,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> FidelityBudget:
        if state.get("version") != 1:
            raise ValueError("unsupported fidelity budget checkpoint state")
        reference = cls.from_reference(
            state["field_names"], state["reference_risks"], state["c0"],
            relative_increase=state["relative_increase"],
            absolute_allowance=state["absolute_allowance"],
            denominator_floor=state["denominator_floor"],
        )
        if reference.state_dict() != dict(state):
            raise ValueError("fidelity budget checkpoint contains inconsistent derived values")
        return reference


class ExternalPolicy:
    """Build one external objective and apply one model backward/optimizer step."""

    def __init__(
        self, settings: Mapping[str, Any], reference: FidelityBudget | Mapping[str, Any], device: torch.device | str
    ) -> None:
        self.settings = dict(settings)
        self.method = self.settings["method"]
        if self.method not in {"lagrangian", "rectify", "extended_log_barrier"}:
            raise ValueError("unknown external policy method")
        self.reference = reference if isinstance(reference, FidelityBudget) else FidelityBudget.from_state_dict(reference)
        self.device = torch.device(device)
        self.completed_updates = 0
        if self.settings.get("vector_reduction", "sum") != "sum" or self.settings.get("objective_scale", "source_mean") != "source_mean":
            raise ValueError("external policy requires vector_reduction=sum and objective_scale=source_mean")
        self._multiplier = self._constraint = self._dual_optimizer = None
        self._constraint_state_type = None
        self._barrier_settings = None
        if self.method == "lagrangian":
            self._init_lagrangian()
        elif self.method == "extended_log_barrier":
            self._barrier_settings = {
                "t_initial": float(self.settings.get("extended_log_barrier", {}).get("t_initial", 1.0)),
                "growth_factor": float(self.settings.get("extended_log_barrier", {}).get("growth_factor", 1.1)),
                "update_interval": int(self.settings.get("extended_log_barrier", {}).get("update_interval", 100)),
                "t_max": float(self.settings.get("extended_log_barrier", {}).get("t_max", 100.0)),
            }
            barrier_schedule(0, **self._barrier_settings)

    def _init_lagrangian(self) -> None:
        try:
            from cooper.constraints import Constraint, ConstraintState
            from cooper.formulations import Lagrangian
            from cooper.multipliers import DenseMultiplier
            from cooper.utils import ConstraintType
        except ImportError as error:
            raise ImportError("lagrangian policy requires cooper-optim==1.0.1; install posttrain_constraints") from error
        dual_lr = float(self.settings.get("lagrangian", {}).get("dual_lr", 0.01))
        if not math.isfinite(dual_lr) or dual_lr <= 0:
            raise ValueError("lagrangian dual_lr must be finite and positive")
        zeros = torch.zeros(len(self.reference.field_names), dtype=torch.float64, device=self.device)
        self._multiplier = DenseMultiplier(
            num_constraints=len(self.reference.field_names), init=zeros, device=self.device, dtype=torch.float64
        )
        self._constraint = Constraint(
            ConstraintType.INEQUALITY, formulation_type=Lagrangian, multiplier=self._multiplier
        )
        self._constraint.name = "field_fidelity"
        self._dual_optimizer = torch.optim.SGD([self._multiplier.weight], lr=dual_lr, maximize=True)
        self._constraint_state_type = ConstraintState

    def update(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        primary: Tensor | Mapping[str, Tensor],
        risks: Tensor,
        grad_clip: float | None,
    ) -> dict[str, Any]:
        primary_value = reduce_primary(primary)
        violations = self.reference.violations(risks)
        if primary_value.device != violations.device:
            raise ValueError("primary objective and risks must share a device")
        scaled_primary = primary_value / self.reference.objective_scale
        diagnostics: dict[str, Any] = {}
        if self.method == "lagrangian":
            objective, diagnostics = self._lagrangian_objective(scaled_primary, violations)
        elif self.method == "rectify":
            objective, diagnostics = self._rectify_objective(scaled_primary, violations)
        else:
            objective, diagnostics = self._barrier_objective(scaled_primary, violations)
        if not objective.requires_grad or not bool(torch.isfinite(objective.detach()).all()):
            raise FloatingPointError("external objective is non-finite or has no model gradient")

        parameters = _model_optimizer_parameters(model, optimizer)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        for parameter in parameters:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
        if any(not bool(torch.isfinite(parameter.grad).all()) for parameter in parameters):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("model gradient is non-finite; refusing update")
        clip = None if grad_clip is None or float(grad_clip) == 0 else float(grad_clip)
        if clip is not None and (not math.isfinite(clip) or clip < 0):
            raise ValueError("grad_clip must be finite and non-negative")
        norm = stable_clip_grad_norm_(parameters, clip) if clip is not None else _grad_norm(parameters)
        if not bool(torch.isfinite(norm)):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("model gradient norm is non-finite; refusing update")
        clip_coefficient = min(1.0, clip / (float(norm) + 1.0e-6)) if clip else 1.0

        if self.method == "lagrangian":
            self._prepare_dual_gradient(violations)
        optimizer.step()
        if any(not bool(torch.isfinite(parameter.detach()).all()) for parameter in parameters):
            raise FloatingPointError("model optimizer produced non-finite parameters")
        if self.method == "lagrangian":
            self._dual_optimizer.step()
            self._multiplier.post_step_()
            diagnostics["multipliers_after"] = self._multipliers()
        self.completed_updates += 1
        diagnostics.update(
            policy=self.method,
            optimized_scalar=float(objective.detach().cpu()),
            primary_scalar=float(primary_value.detach().cpu()),
            scaled_primary_scalar=float(scaled_primary.detach().cpu()),
            field_risk=dict(zip(self.reference.field_names, (float(x) for x in risks.detach().cpu().reshape(-1)))),
            field_reference_risk=dict(zip(self.reference.field_names, self.reference.reference_risks)),
            field_upper_bound=dict(zip(self.reference.field_names, self.reference.upper_bounds)),
            field_violation=dict(zip(self.reference.field_names, (float(x) for x in violations.detach().cpu()))),
            denominator_floor_activated=dict(zip(self.reference.field_names, self.reference.denominator_floor_activated)),
            objective_floor_activated=self.reference.objective_floor_activated,
            model_grad_norm=float(norm.detach().cpu()),
            gradient_clip_coefficient=clip_coefficient,
            completed_updates=self.completed_updates,
            update_accepted=True,
        )
        return diagnostics

    def _lagrangian_objective(self, primary: Tensor, violations: Tensor) -> tuple[Tensor, dict[str, Any]]:
        state = self._constraint_state_type(violation=violations, strict_violation=violations.detach())
        # Cooper's primal contribution detaches multiplier state; the dual contribution
        # detaches violations, so neither graph crosses into the other variable set.
        contribution = self._constraint.compute_contribution_to_lagrangian(state, "primal")
        return primary + contribution.lagrangian_contribution, {"multipliers_before": self._multipliers()}

    def _prepare_dual_gradient(self, violations: Tensor) -> None:
        self._dual_optimizer.zero_grad(set_to_none=True)
        state = self._constraint_state_type(violation=violations, strict_violation=violations.detach())
        contribution = self._constraint.compute_contribution_to_lagrangian(state, "dual")
        contribution.lagrangian_contribution.backward()
        gradient = self._multiplier.weight.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("Cooper dual gradient is missing or non-finite")

    def _rectify_objective(self, primary: Tensor, violations: Tensor) -> tuple[Tensor, dict[str, Any]]:
        maximum, index = torch.max(violations.detach(), dim=0)
        if float(maximum) <= 0:
            return primary, {"phase": "improve", "repair_field": None}
        return violations[index], {"phase": "repair", "repair_field": self.reference.field_names[int(index)]}

    def _barrier_objective(self, primary: Tensor, violations: Tensor) -> tuple[Tensor, dict[str, Any]]:
        t = barrier_schedule(self.completed_updates, **self._barrier_settings)
        values = extended_log_barrier(violations, t)
        weights = extended_log_barrier_derivative(violations.detach(), t)
        return primary + values.sum(), {
            "barrier_t": t,
            "barrier_derivative_weights": dict(zip(self.reference.field_names, (float(x) for x in weights.cpu()))),
        }

    def _multipliers(self) -> list[float] | None:
        return None if self._multiplier is None else [float(x) for x in self._multiplier().detach().cpu()]

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "method": self.method,
            "settings": self.settings,
            "reference": self.reference.state_dict(),
            "multipliers": self._multipliers(),
            "dual_optimizer": None if self._dual_optimizer is None else self._dual_optimizer.state_dict(),
            "completed_updates": self.completed_updates,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != 1 or state.get("method") != self.method or state.get("settings") != self.settings:
            raise ValueError("external policy checkpoint does not match active settings")
        reference = FidelityBudget.from_state_dict(state["reference"])
        if reference.field_names != self.reference.field_names:
            raise ValueError("external policy checkpoint field order differs")
        self.reference = reference
        updates = state["completed_updates"]
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
            raise ValueError("invalid completed update count")
        self.completed_updates = updates
        if self.method == "lagrangian":
            values = torch.as_tensor(state["multipliers"], dtype=torch.float64, device=self.device)
            if values.numel() != len(self.reference.field_names) or not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
                raise ValueError("invalid Lagrangian multipliers")
            with torch.no_grad():
                self._multiplier.weight.copy_(values.reshape(-1))
            self._dual_optimizer.load_state_dict(state["dual_optimizer"])


def _model_optimizer_parameters(model: nn.Module, optimizer: torch.optim.Optimizer) -> list[nn.Parameter]:
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"] if p.requires_grad}
    if not parameters or any(id(p) not in optimized for p in parameters):
        raise ValueError("existing optimizer must own every trainable model parameter")
    return parameters


def _grad_norm(parameters: Sequence[nn.Parameter]) -> Tensor:
    device = parameters[0].device
    total = torch.zeros((), dtype=torch.float64, device=device)
    for parameter in parameters:
        total += parameter.grad.detach().abs().to(torch.float64).square().sum()
    return total.sqrt()
