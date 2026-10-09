"""Minimal adapters for family scalarization and two-objective gradient aggregation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar

import torch
from torch import Tensor, nn

from .gradient_balance import (
    _assign_flat_gradient,
    _flat_gradient,
    _trainable_parameters,
)
from .gradients import stable_clip_grad_norm_


def _setting(
    settings: Mapping[str, Any] | None,
    *,
    default: str,
    supported: set[str],
) -> tuple[str, Mapping[str, Any]]:
    if settings is None:
        settings = {}
    if not isinstance(settings, Mapping):
        raise TypeError("multitask settings must be a mapping")
    method = str(settings.get("method", default)).lower()
    if method not in supported:
        raise ValueError(f"unsupported multitask method {method!r}; expected one of {sorted(supported)}")
    options = settings.get(method, {})
    if not isinstance(options, Mapping):
        raise TypeError(f"settings for {method!r} must be a mapping")
    return method, options


def _finite_number(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite scalar") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


class FamilyScalarizer:
    """Combine ordered, SOURCE-calibrated family losses into one differentiable scalar."""

    _METHODS: ClassVar[set[str]] = {"fixed_sum", "stch", "dwa"}

    def __init__(self, settings: Mapping[str, Any] | None = None) -> None:
        self.method, options = _setting(settings, default="fixed_sum", supported=self._METHODS)
        self._torchjd_scalarizer = None
        self._mu: float | None = None
        self._temperature = 2.0
        self._family_names: tuple[str, ...] | None = None
        self._previous_averages: list[list[float]] = []
        self._epoch_sum: list[float] | None = None
        self._epoch_batches = 0
        self._epochs_completed = 0

        if self.method == "stch":
            self._mu = _finite_number(options.get("mu", 0.1), name="stch.mu")
            if self._mu <= 0:
                raise ValueError("stch.mu must be strictly positive")
            try:
                from torchjd.scalarization import STCH
            except ImportError as error:
                raise ImportError(
                    "family scalarization method='stch' requires TorchJD 0.17.1; "
                    "install phycoflow-reconstruction[multitask]"
                ) from error
            self._torchjd_scalarizer = STCH(mu=self._mu, weights=None, reference=None)
        elif self.method == "dwa":
            self._temperature = _finite_number(
                options.get("temperature", 2.0), name="dwa.temperature"
            )
            if self._temperature <= 0:
                raise ValueError("dwa.temperature must be strictly positive")

    def __call__(
        self,
        raw_losses: Mapping[str, Tensor],
        *,
        scales: Mapping[str, float],
        weights: Mapping[str, float],
    ) -> tuple[Tensor, dict[str, Any]]:
        if not isinstance(raw_losses, Mapping):
            raise TypeError("raw_losses must be an ordered mapping")
        if not raw_losses:
            raise ValueError("raw_losses must be a non-empty ordered mapping")
        names = tuple(raw_losses)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("family names must be non-empty strings")
        if set(scales) != set(names) or set(weights) != set(names):
            raise ValueError("scales and weights must have exactly the raw-loss family keys")
        if self._family_names is not None and names != self._family_names:
            raise ValueError("family names or order changed for this scalarizer")
        self._family_names = names

        calibrated_values: list[Tensor] = []
        raw_diagnostics: dict[str, float] = {}
        calibrated_diagnostics: dict[str, float] = {}
        for name, loss in raw_losses.items():
            if not isinstance(loss, Tensor) or loss.numel() != 1 or not loss.is_floating_point():
                raise TypeError(f"family loss {name!r} must be a real scalar tensor")
            if not loss.requires_grad:
                raise ValueError(f"family loss {name!r} must retain its gradient graph")
            if not bool(torch.isfinite(loss.detach()).all()):
                raise FloatingPointError(f"family loss {name!r} is non-finite")
            scale = _finite_number(scales[name], name=f"scales[{name!r}]")
            weight = _finite_number(weights[name], name=f"weights[{name!r}]")
            raw = loss.reshape(())
            calibrated = raw * (scale * weight)
            if not bool(torch.isfinite(calibrated.detach()).all()):
                raise FloatingPointError(f"calibrated family loss {name!r} is non-finite")
            calibrated_values.append(calibrated)
            raw_diagnostics[name] = float(raw.detach().cpu())
            calibrated_diagnostics[name] = float(calibrated.detach().cpu())

        values = torch.stack(calibrated_values)
        effective_weights = self._effective_weights(values)
        if self.method == "fixed_sum":
            scalar = sum(calibrated_values)
        elif self.method == "stch":
            scalar = self._torchjd_scalarizer(values)
        else:
            scalar = torch.dot(effective_weights, values)
            self._accumulate_epoch(values)
        if not bool(torch.isfinite(scalar.detach()).all()):
            raise FloatingPointError(f"family scalarizer method {self.method!r} returned non-finite loss")

        diagnostics = {
            "method": self.method,
            "raw_losses": raw_diagnostics,
            "calibrated_losses": calibrated_diagnostics,
            "effective_weights": {
                name: float(value.detach().cpu())
                for name, value in zip(names, effective_weights)
            },
            "scalarized_loss": float(scalar.detach().cpu()),
        }
        return scalar, diagnostics

    def _effective_weights(self, values: Tensor) -> Tensor:
        if self.method == "fixed_sum":
            return torch.ones_like(values)
        if self.method == "stch":
            # TorchJD STCH defaults to the uniform preference vector (1 / K, ...).
            return torch.softmax(values.detach() / (values.numel() * self._mu), dim=0) / values.numel()
        if len(self._previous_averages) < 2:
            return torch.ones_like(values)
        older = values.new_tensor(self._previous_averages[0])
        newer = values.new_tensor(self._previous_averages[1])
        if bool((older == 0).any()):
            raise FloatingPointError("DWA requires nonzero family means in its two completed epochs")
        rates = newer / older
        if not bool(torch.isfinite(rates).all()):
            raise FloatingPointError("DWA relative loss rates are non-finite")
        return values.numel() * torch.softmax(rates / self._temperature, dim=0)

    def _accumulate_epoch(self, values: Tensor) -> None:
        detached = [float(value) for value in values.detach().cpu().tolist()]
        if self._epoch_sum is None:
            self._epoch_sum = detached
        else:
            self._epoch_sum = [total + value for total, value in zip(self._epoch_sum, detached)]
        self._epoch_batches += 1

    def step(self) -> None:
        """Finalize a DWA reporting epoch; empty epochs leave its history unchanged."""
        if self.method != "dwa" or self._epoch_batches == 0:
            return
        if self._epoch_sum is None:
            raise RuntimeError("DWA epoch state is inconsistent")
        average = [value / self._epoch_batches for value in self._epoch_sum]
        self._previous_averages = [*self._previous_averages, average][-2:]
        self._epoch_sum = None
        self._epoch_batches = 0
        self._epochs_completed += 1

    def state_dict(self) -> dict[str, Any]:
        """Return only the small DWA state needed to reproduce its next epoch weights."""
        if self.method != "dwa":
            return {}
        return {
            "version": 1,
            "method": "dwa",
            "temperature": self._temperature,
            "family_names": list(self._family_names) if self._family_names is not None else None,
            "previous_averages": [list(values) for values in self._previous_averages],
            "epoch_sum": list(self._epoch_sum) if self._epoch_sum is not None else None,
            "epoch_batches": self._epoch_batches,
            "epochs_completed": self._epochs_completed,
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        if self.method != "dwa":
            if state not in (None, {}):
                raise ValueError("only DWA scalarizers accept scalarizer checkpoint state")
            return
        if not isinstance(state, Mapping):
            raise TypeError("DWA resume requires saved scalarizer state")
        if state.get("method") != "dwa" or state.get("version") != 1:
            raise ValueError("checkpoint does not contain supported DWA scalarizer state")
        temperature = _finite_number(state.get("temperature"), name="DWA state temperature")
        if temperature != self._temperature:
            raise ValueError("DWA checkpoint temperature does not match the active configuration")

        names = state.get("family_names")
        if names is not None:
            if not isinstance(names, (tuple, list)) or any(not isinstance(name, str) for name in names):
                raise ValueError("DWA checkpoint family_names must be a string sequence")
            self._family_names = tuple(names)
        previous = state.get("previous_averages", [])
        if not isinstance(previous, list) or len(previous) > 2:
            raise ValueError("DWA checkpoint must contain at most two completed epoch means")
        self._previous_averages = [
            [_finite_number(value, name="DWA history mean") for value in row]
            for row in previous
        ]
        if self._family_names is not None and any(
            len(row) != len(self._family_names) for row in self._previous_averages
        ):
            raise ValueError("DWA checkpoint history does not match its family names")

        epoch_batches = state.get("epoch_batches", 0)
        if not isinstance(epoch_batches, int) or epoch_batches < 0:
            raise ValueError("DWA checkpoint epoch_batches must be a non-negative integer")
        epoch_sum = state.get("epoch_sum")
        if epoch_batches == 0:
            if epoch_sum is not None:
                raise ValueError("DWA checkpoint has an epoch sum without any epoch batches")
            self._epoch_sum = None
        else:
            if not isinstance(epoch_sum, (tuple, list)):
                raise ValueError("DWA checkpoint is missing its partial epoch sum")
            self._epoch_sum = [
                _finite_number(value, name="DWA partial epoch sum") for value in epoch_sum
            ]
            if self._family_names is None or len(self._epoch_sum) != len(self._family_names):
                raise ValueError("DWA partial epoch sum does not match its family names")
        self._epoch_batches = epoch_batches
        epochs_completed = state.get("epochs_completed", 0)
        if not isinstance(epochs_completed, int) or epochs_completed < len(self._previous_averages):
            raise ValueError("DWA checkpoint epochs_completed is invalid")
        self._epochs_completed = epochs_completed


class OuterAggregator:
    """Select the update direction for exactly two objective gradients."""

    _METHODS: ClassVar[set[str]] = {"sum", "torchjd_config", "upgrad", "cagrad"}

    def __init__(self, settings: Mapping[str, Any] | None = None) -> None:
        self.method, options = _setting(settings, default="sum", supported=self._METHODS)
        self._c = 0.5
        if self.method == "cagrad":
            self._c = _finite_number(options.get("c", 0.5), name="cagrad.c")
            if self._c < 0:
                raise ValueError("cagrad.c must be non-negative")

    def __call__(self, gradients: Tensor) -> tuple[Tensor, dict[str, Any]]:
        if not isinstance(gradients, Tensor) or gradients.ndim != 2 or gradients.shape[0] != 2:
            raise ValueError("outer aggregation expects a tensor with shape [2, P]")
        if gradients.shape[1] == 0 or not gradients.is_floating_point():
            raise ValueError("outer gradients must be non-empty real vectors")
        if not bool(torch.isfinite(gradients).all()):
            raise FloatingPointError("outer objective gradient contains non-finite values")

        if self.method == "sum":
            direction = gradients.sum(dim=0)
        elif self.method == "torchjd_config":
            try:
                from torchjd.aggregation import ConFIG
            except ImportError as error:
                raise ImportError(
                    "outer method='torchjd_config' requires TorchJD 0.17.1; "
                    "install phycoflow-reconstruction[multitask]"
                ) from error
            direction = ConFIG()(gradients)
        elif self.method == "upgrad":
            try:
                from torchjd.aggregation import UPGrad
            except ImportError as error:
                raise ImportError(
                    "outer method='upgrad' requires TorchJD and its quadprog projector extra; "
                    "install phycoflow-reconstruction[multitask]"
                ) from error
            direction = UPGrad()(gradients)
        else:
            try:
                from torchjd.aggregation import CAGrad
            except ImportError as error:
                raise ImportError(
                    "outer method='cagrad' requires TorchJD and its CAGrad extra; "
                    "install phycoflow-reconstruction[multitask]"
                ) from error
            direction = CAGrad(c=self._c)(gradients)

        if not isinstance(direction, Tensor) or direction.shape != (gradients.shape[1],):
            raise ValueError("outer aggregator returned a direction with the wrong shape")
        if not bool(torch.isfinite(direction).all()):
            raise FloatingPointError(
                f"outer method {self.method!r} returned a non-finite direction; refusing update"
            )

        measured = gradients.detach().to(dtype=torch.float64)
        direction64 = direction.detach().to(dtype=torch.float64)
        norms = torch.linalg.vector_norm(measured, dim=1)
        denominator = (norms[0] * norms[1]).clamp_min(1e-12)
        cosine = torch.dot(measured[0], measured[1]) / denominator
        diagnostics = {
            "method": self.method,
            "gradient_norms": [float(value.cpu()) for value in norms],
            "gradient_cosine": float(cosine.cpu()),
            "direction_norm": float(torch.linalg.vector_norm(direction64).cpu()),
            "directional_products": [
                float(torch.dot(direction64, row).cpu()) for row in measured
            ],
        }
        return direction, diagnostics


def modular_objective_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loss: Tensor,
    coherence_loss: Tensor,
    *,
    aggregator: OuterAggregator,
    data_weight: float,
    coherence_weight: float,
    grad_clip: float | None,
) -> dict[str, Any]:
    """Aggregate D and C' gradients, assign one direction, clip, and update once."""
    data_coefficient = _finite_number(data_weight, name="data_weight")
    coherence_coefficient = _finite_number(coherence_weight, name="coherence_weight")
    for name, loss in (("data_loss", data_loss), ("coherence_loss", coherence_loss)):
        if not isinstance(loss, Tensor) or loss.numel() != 1 or not loss.is_floating_point():
            raise TypeError(f"{name} must be a real scalar tensor")
        if not bool(torch.isfinite(loss.detach()).all()):
            raise FloatingPointError(f"{name} is non-finite")

    parameters = _trainable_parameters(model)
    optimizer.zero_grad(set_to_none=True)
    weighted_data = data_coefficient * data_loss
    weighted_coherence = coherence_coefficient * coherence_loss
    data_gradient = _flat_gradient(weighted_data, parameters)
    coherence_gradient = _flat_gradient(weighted_coherence, parameters)
    gradients = torch.stack((data_gradient, coherence_gradient))
    if not bool(torch.isfinite(gradients).all()):
        raise FloatingPointError("post-training objective gradient contains non-finite values")

    direction, aggregation_diagnostics = aggregator(gradients)
    if direction.shape != (gradients.shape[1],) or not bool(torch.isfinite(direction).all()):
        raise FloatingPointError("outer aggregator returned an invalid direction; refusing update")

    clipped_norm = None
    assigned_direction = direction
    if grad_clip:
        # The flat vector stores complex gradients as real/imaginary coordinates. Clipping it
        # preserves the global norm while avoiding a real cast of complex parameter gradients.
        clip_parameter = nn.Parameter(direction.detach(), requires_grad=False)
        clip_parameter.grad = direction.detach()
        clipped_norm = float(
            stable_clip_grad_norm_(clip_parameter, float(grad_clip)).detach().cpu()
        )
        assigned_direction = clip_parameter.grad
    _assign_flat_gradient(parameters, assigned_direction)
    optimizer.step()

    measured = gradients.detach().to(dtype=torch.float64)
    norms = torch.linalg.vector_norm(measured, dim=1)
    data_norm, coherence_norm = (float(value.cpu()) for value in norms)
    return {
        "update_mode": "multitask_modular",
        "optimization_route": "modular",
        "outer_method": aggregator.method,
        "data_grad_norm": data_norm,
        "coherence_grad_norm": coherence_norm,
        "gradient_cosine": aggregation_diagnostics["gradient_cosine"],
        "gradient_conflict": aggregation_diagnostics["gradient_cosine"] < 0.0,
        "combined_grad_norm": aggregation_diagnostics["direction_norm"],
        "directional_products": aggregation_diagnostics["directional_products"],
        "clip_input_norm": clipped_norm,
        "config_fallback_used": False,
        "outer_aggregation": aggregation_diagnostics,
    }
