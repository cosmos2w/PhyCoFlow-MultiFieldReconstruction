"""Family scalarization and aggregation of hierarchical or joint objective gradients."""

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
    """Calibrate families, then scalarize or keep them separate with method ``none``."""

    _METHODS: ClassVar[set[str]] = {"none", "fixed_sum", "stch", "dwa"}

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
    ) -> tuple[Tensor | dict[str, Tensor], dict[str, Any]]:
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
        diagnostic_values = []
        for name, loss in raw_losses.items():
            if not isinstance(loss, Tensor) or loss.numel() != 1 or not loss.is_floating_point():
                raise TypeError(f"family loss {name!r} must be a real scalar tensor")
            if not loss.requires_grad:
                raise ValueError(f"family loss {name!r} must retain its gradient graph")
            scale = _finite_number(scales[name], name=f"scales[{name!r}]")
            weight = _finite_number(weights[name], name=f"weights[{name!r}]")
            raw = loss.reshape(())
            calibrated = raw * (scale * weight)
            calibrated_values.append(calibrated)
            diagnostic_values.extend((raw.detach(), calibrated.detach()))

        # Transfer the tiny diagnostic vector once; validation still precedes
        # scalarizer state changes and every model backward/optimizer action.
        observed = torch.stack(diagnostic_values).to(torch.float64).cpu().tolist()
        for index, name in enumerate(names):
            raw_value, calibrated_value = observed[2 * index : 2 * index + 2]
            if not math.isfinite(raw_value):
                raise FloatingPointError(f"family loss {name!r} is non-finite")
            if not math.isfinite(calibrated_value):
                raise FloatingPointError(f"calibrated family loss {name!r} is non-finite")
            raw_diagnostics[name] = raw_value
            calibrated_diagnostics[name] = calibrated_value

        diagnostics = {
            "method": self.method,
            "raw_losses": raw_diagnostics,
            "calibrated_losses": calibrated_diagnostics,
        }
        if self.method == "none":
            # A zero coefficient disables a task; a genuinely stationary active task stays.
            separate = {
                name: value
                for name, value in zip(names, calibrated_values)
                if float(scales[name]) * float(weights[name]) != 0.0
            }
            diagnostics["effective_weights"] = {
                name: float(name in separate) for name in names
            }
            return separate, diagnostics

        values = torch.stack(calibrated_values)
        effective_weights = self._effective_weights(values)
        if self.method == "fixed_sum":
            scalar = sum(calibrated_values)
        elif self.method == "stch":
            scalar = self._torchjd_scalarizer(values)
        else:
            scalar = torch.dot(effective_weights, values)
            self._accumulate_epoch(values)
        observed = torch.cat((effective_weights.detach().to(torch.float64), scalar.detach().reshape(1).to(torch.float64))).cpu().tolist()
        if not math.isfinite(observed[-1]):
            raise FloatingPointError(f"family scalarizer method {self.method!r} returned non-finite loss")

        diagnostics.update(
            effective_weights=dict(zip(names, observed[:-1])),
            scalarized_loss=observed[-1],
        )
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
        detached = values.detach().cpu().tolist()
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
    """Select one update direction from an ordered matrix of objective gradients."""

    _METHODS: ClassVar[set[str]] = {"sum", "torchjd_config", "upgrad", "cagrad"}

    def __init__(self, settings: Mapping[str, Any] | None = None) -> None:
        self.method, options = _setting(settings, default="sum", supported=self._METHODS)
        self._c = 0.5
        if self.method == "cagrad":
            self._c = _finite_number(options.get("c", 0.5), name="cagrad.c")
            if self._c < 0:
                raise ValueError("cagrad.c must be non-negative")

    def __call__(self, gradients: Tensor) -> tuple[Tensor, dict[str, Any]]:
        return self._aggregate(gradients)

    def _aggregate(
        self, gradients: Tensor, *, row_norms: Tensor | None = None, row_gram: Tensor | None = None
    ) -> tuple[Tensor, dict[str, Any]]:
        if not isinstance(gradients, Tensor) or gradients.ndim != 2 or gradients.shape[0] == 0:
            raise ValueError("outer aggregation expects a tensor with shape [N, P], N >= 1")
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

        return direction, _gradient_diagnostics(
            gradients, direction, method=self.method, row_norms=row_norms, row_gram=row_gram
        )


def _gradient_diagnostics(
    gradients: Tensor, direction: Tensor, *, method: str,
    row_norms: Tensor | None = None, row_gram: Tensor | None = None,
) -> dict[str, Any]:
    measured = gradients.detach().to(dtype=torch.float64)
    direction64 = direction.detach().to(dtype=torch.float64)
    norms = torch.linalg.vector_norm(measured, dim=1) if row_norms is None else row_norms
    safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
    if row_gram is None:
        units = measured / safe_norms[:, None]
        pairwise_products = units @ units.T
    else:
        # Only diagnostics use this identity. TorchJD still receives the
        # original FP64 rows and performs its original wide pseudoinverse.
        pairwise_products = row_gram / (safe_norms[:, None] * safe_norms[None, :])
    # For a joint update this summary compares the first task with the sum of the rest.
    # Individual conflicts are retained in the ordered pairwise matrix.
    remaining = measured[1:].sum(dim=0)
    denominator = norms[0] * torch.linalg.vector_norm(remaining)
    positive = denominator > 0
    cosine = torch.where(
        positive,
        torch.dot(measured[0], remaining) / torch.where(positive, denominator, torch.ones_like(denominator)),
        torch.zeros_like(denominator),
    )
    count = gradients.shape[0]
    observed = torch.cat((
        norms,
        cosine.reshape(1),
        pairwise_products.reshape(-1),
        torch.linalg.vector_norm(direction64).reshape(1),
        measured @ direction64,
    )).cpu().tolist()
    pairwise_end = count + 1 + count * count
    pairwise = observed[count + 1 : pairwise_end]
    return {
        "method": method,
        "gradient_norms": observed[:count],
        "gradient_cosine": observed[count],
        "pairwise_gradient_cosines": [pairwise[i * count : (i + 1) * count] for i in range(count)],
        "direction_norm": observed[pairwise_end],
        "directional_products": observed[pairwise_end + 1 :],
    }


def modular_objective_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loss: Tensor,
    coherence_loss: Tensor | Mapping[str, Tensor],
    *,
    aggregator: OuterAggregator,
    data_weight: float,
    coherence_weight: float,
    grad_clip: float | None,
) -> dict[str, Any]:
    """Aggregate RF with a coherence scalar or separate families, then update once."""
    data_coefficient = _finite_number(data_weight, name="data_weight")
    coherence_coefficient = _finite_number(coherence_weight, name="coherence_weight")
    joint = isinstance(coherence_loss, Mapping)
    coherence_losses = dict(coherence_loss) if joint else {"coherence": coherence_loss}
    if any(
        not isinstance(name, str) or not name or name == "data_retention"
        for name in coherence_losses
    ):
        raise ValueError("coherence objective names must be non-empty and distinct from data_retention")
    losses = {"data_retention": data_loss, **coherence_losses}
    for name, loss in losses.items():
        if not isinstance(loss, Tensor) or loss.numel() != 1 or not loss.is_floating_point():
            raise TypeError(f"{name} must be a real scalar tensor")
        if not bool(torch.isfinite(loss.detach()).all()):
            raise FloatingPointError(f"{name} is non-finite")

    parameters = _trainable_parameters(model)
    optimizer.zero_grad(set_to_none=True)
    active: dict[str, Tensor] = {}
    for name, loss in losses.items():
        coefficient = data_coefficient if name == "data_retention" else coherence_coefficient
        if coefficient != 0.0:
            active[name] = _flat_gradient(coefficient * loss, parameters)
    if not active:
        raise ValueError("modular update requires at least one active objective")
    gradients = torch.stack(list(active.values()))
    if not bool(torch.isfinite(gradients).all()):
        raise FloatingPointError("post-training objective gradient contains non-finite values")

    single_objective = len(active) == 1
    if single_objective:
        direction = gradients[0]
        aggregation_diagnostics = _gradient_diagnostics(
            gradients, direction, method="single_objective"
        )
    else:
        direction, aggregation_diagnostics = aggregator(gradients)
    if direction.shape != (gradients.shape[1],) or not bool(torch.isfinite(direction).all()):
        raise FloatingPointError("outer aggregator returned an invalid direction; refusing update")
    aggregation_diagnostics["objective_names"] = list(active)

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

    zero = torch.zeros_like(direction)
    data_gradient = active.get("data_retention", zero).detach().to(dtype=torch.float64)
    coherence_gradient = sum(
        (
            gradient.detach().to(dtype=torch.float64)
            for name, gradient in active.items()
            if name != "data_retention"
        ),
        torch.zeros_like(data_gradient),
    )
    data_norm = float(torch.linalg.vector_norm(data_gradient).cpu())
    coherence_norm = float(torch.linalg.vector_norm(coherence_gradient).cpu())
    denominator = data_norm * coherence_norm
    cosine = (
        float(torch.dot(data_gradient, coherence_gradient).cpu()) / denominator
        if denominator > 0 else 0.0
    )
    return {
        "update_mode": "modular_single_objective" if single_objective else "multitask_modular",
        "optimization_route": "modular",
        "aggregation_scope": "joint" if joint else "hierarchical",
        "outer_method": aggregator.method,
        "data_grad_norm": data_norm,
        "coherence_grad_norm": coherence_norm,
        "gradient_cosine": cosine,
        "gradient_conflict": cosine < 0.0,
        "combined_grad_norm": aggregation_diagnostics["direction_norm"],
        "directional_products": aggregation_diagnostics["directional_products"],
        "clip_input_norm": clipped_norm,
        "config_fallback_used": False,
        "outer_aggregation": aggregation_diagnostics,
    }
