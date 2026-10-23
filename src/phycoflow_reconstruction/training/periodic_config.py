"""Stable signed ConFIG coefficient reuse and LPC residual diagnostics.

The coefficient helper is deliberately separate from optimizer and training
dispatch. It recovers a scalar representation of a supplied ConFIG direction;
``PeriodicConfigState`` only owns the resumable refresh cursor and the signed
coefficients needed to scalarize current losses.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import torch
from torch import Tensor


def _real_matrix(value: Tensor, *, name: str, allow_zero_rows: bool = False) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 2:
        raise ValueError(f"{name} must be a rank-two tensor")
    if (
        (value.shape[0] == 0 and not allow_zero_rows)
        or value.shape[1] == 0
        or not value.is_floating_point()
    ):
        raise ValueError(f"{name} must be a non-empty real floating-point matrix")
    result = value.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError(f"{name} contains non-finite values")
    return result


def _vector(value: Tensor, *, name: str, width: int, device: torch.device) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 1 or value.numel() != width:
        raise ValueError(f"{name} must be a vector with length {width}")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a real floating-point dtype")
    result = value.detach().to(device=device, dtype=torch.float64)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError(f"{name} contains non-finite values")
    return result


def _matrix_rank_tolerance(shape: tuple[int, int], singular_values: Tensor) -> tuple[float, float]:
    """Use the default ``torch.linalg.pinv`` relative cutoff for the full shape."""
    if singular_values.numel() == 0:
        return 0.0, 0
    largest = float(singular_values[0].detach().cpu())
    relative = max(shape) * torch.finfo(torch.float64).eps
    return relative * largest, relative


def _thin_row_svd(matrix: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return the thin SVD of a short-row matrix using QR then a small SVD.

    If ``matrix.T = Q R`` and ``R.T = L S V.T``, then
    ``matrix = L S (Q V).T``. Only the parameter-sized thin Q is retained; the
    rank decision uses singular values from the small square factor and the
    default full-matrix pseudoinverse cutoff.
    """
    q, r = torch.linalg.qr(matrix.T, mode="reduced")
    left, singular_values, right_h_small = torch.linalg.svd(r.T, full_matrices=False)
    right = q @ right_h_small.T
    return left, singular_values, right


@dataclass(frozen=True)
class ConfigCoefficientResult:
    """Coefficient recovery checked against the supplied full-width oracle."""

    coefficients: Tensor
    oracle_direction: Tensor
    reconstructed_direction: Tensor
    rank: int
    rank_threshold: float
    relative_reconstruction_error: float
    coefficient_signs: tuple[int, ...]

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "rank_threshold": self.rank_threshold,
            "relative_reconstruction_error": self.relative_reconstruction_error,
            "coefficient_signs": list(self.coefficient_signs),
            "coefficients": self.coefficients.detach().cpu().tolist(),
        }


def extract_config_coefficients(
    gradients: Tensor,
    *,
    oracle_direction: Tensor | None = None,
    max_relative_reconstruction_error: float = 1.0e-8,
) -> ConfigCoefficientResult:
    """Recover signed ``alpha`` such that ``gradients.T @ alpha`` is ConFIG.

    ``gradients`` must be the same full-width FP64 matrix passed to TorchJD
    ``ConFIG``. Passing its already-computed direction avoids invoking the
    expensive oracle twice. If omitted, the pinned TorchJD implementation is
    called here. The reduced QR/SVD is rank-revealing and avoids normal
    equations, while its singular-value cutoff matches ``pinv`` for the full
    matrix dimensions.
    """
    matrix = _real_matrix(gradients, name="gradients")
    if isinstance(max_relative_reconstruction_error, bool):
        raise TypeError("max_relative_reconstruction_error must be finite and positive")
    try:
        reconstruction_tolerance = float(max_relative_reconstruction_error)
    except (TypeError, ValueError) as error:
        raise TypeError("max_relative_reconstruction_error must be finite and positive") from error
    if not math.isfinite(reconstruction_tolerance) or reconstruction_tolerance <= 0.0:
        raise ValueError("max_relative_reconstruction_error must be finite and positive")
    objective_count, parameter_count = matrix.shape
    norms = torch.linalg.vector_norm(matrix, dim=1)
    if not bool(torch.isfinite(norms).all()):
        raise FloatingPointError("gradient row norm is non-finite")
    units = torch.where(norms[:, None] > 0, matrix / norms.clamp_min(1.0e-300)[:, None], 0.0)

    if oracle_direction is None:
        try:
            from torchjd.aggregation import ConFIG
        except ImportError as error:
            raise ImportError(
                "ConFIG coefficient extraction requires TorchJD 0.17.1; "
                "install phycoflow-reconstruction[multitask]"
            ) from error
        if version("torchjd") != "0.17.1":
            raise RuntimeError("ConFIG coefficient extraction is pinned to TorchJD 0.17.1")
        direction = ConFIG()(matrix)
    else:
        direction = _vector(
            oracle_direction,
            name="oracle_direction",
            width=parameter_count,
            device=matrix.device,
        )
    if not bool(torch.isfinite(direction).all()):
        raise FloatingPointError("TorchJD ConFIG oracle returned a non-finite direction")

    left, singular_values, right = _thin_row_svd(units)
    threshold, _ = _matrix_rank_tolerance((objective_count, parameter_count), singular_values)
    retained = singular_values > threshold
    rank = int(retained.sum().detach().cpu())

    coefficients = torch.zeros(objective_count, dtype=torch.float64, device=matrix.device)
    if rank:
        # U = L S V.T. For d in row(U), the minimum-norm beta satisfying
        # U.T beta = d is L S^-1 V.T d. Since G = diag(norms) U, alpha is
        # beta / norms. Zero rows have no identifiable coefficient and retain 0.
        left_kept = left[:, retained]
        singular_kept = singular_values[retained]
        right_kept = right[:, retained]
        beta = left_kept @ ((right_kept.T @ direction) / singular_kept)
        nonzero = norms > 0
        coefficients = torch.where(nonzero, beta / norms.clamp_min(1.0e-300), 0.0)

    reconstructed = matrix.T @ coefficients
    error_norm = torch.linalg.vector_norm(reconstructed - direction)
    direction_norm = torch.linalg.vector_norm(direction)
    relative_error = (
        float(error_norm.detach().cpu()) / float(direction_norm.detach().cpu())
        if float(direction_norm.detach().cpu()) > 0.0
        else float(error_norm.detach().cpu())
    )
    if not bool(torch.isfinite(coefficients).all()) or not math.isfinite(relative_error):
        raise FloatingPointError("ConFIG coefficient recovery produced a non-finite result")
    if relative_error > reconstruction_tolerance:
        raise FloatingPointError(
            "ConFIG coefficient reconstruction exceeded tolerance "
            f"({relative_error:.3e} > {reconstruction_tolerance:.3e})"
        )
    signs = tuple(
        1 if float(value) > 0.0 else -1 if float(value) < 0.0 else 0
        for value in coefficients.detach().cpu().tolist()
    )
    return ConfigCoefficientResult(
        coefficients=coefficients,
        oracle_direction=direction,
        reconstructed_direction=reconstructed,
        rank=rank,
        rank_threshold=threshold,
        relative_reconstruction_error=relative_error,
        coefficient_signs=signs,
    )


class PeriodicConfigState:
    """Versioned coefficient and refresh-cursor state for periodic reuse.

    The compatibility key must identify the exact objective/configuration
    contract that produced the coefficients. ``advance`` is called only after
    an accepted optimizer update. Coefficients remain signed Python floats in
    state, independent of the parameter device and scalar-loss dtype.
    """

    STATE_VERSION = 1

    def __init__(self, interval: int = 4, *, compatibility_key: str) -> None:
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError("periodic ConFIG interval must be a positive integer")
        if not isinstance(compatibility_key, str) or not compatibility_key:
            raise ValueError("periodic ConFIG compatibility_key must be a non-empty string")
        self.interval = interval
        self.compatibility_key = compatibility_key
        self._names: tuple[str, ...] | None = None
        self._coefficients: tuple[float, ...] | None = None
        self.completed_steps = 0
        self.last_refresh_step: int | None = None

    @property
    def refresh_due(self) -> bool:
        if self._coefficients is None or self.last_refresh_step is None:
            return True
        return self.completed_steps - self.last_refresh_step >= self.interval

    @property
    def age(self) -> int | None:
        if self.last_refresh_step is None:
            return None
        return self.completed_steps - self.last_refresh_step

    @property
    def objective_names(self) -> tuple[str, ...] | None:
        return self._names

    @property
    def coefficients(self) -> tuple[float, ...] | None:
        return self._coefficients

    def refresh(
        self,
        objective_names: Sequence[str],
        coefficients: Tensor | Sequence[float] | Mapping[str, float],
        *,
        force: bool = False,
    ) -> None:
        """Install a fresh signed coefficient vector at the current cursor."""
        names = tuple(objective_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("objective_names must contain non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("objective_names must be unique")
        if not force and not self.refresh_due:
            raise RuntimeError("periodic ConFIG refresh was requested before its interval")

        if isinstance(coefficients, Mapping):
            if tuple(coefficients) != names:
                raise ValueError("coefficient mapping order must match objective_names")
            values = tuple(float(coefficients[name]) for name in names)
        elif isinstance(coefficients, Tensor):
            if coefficients.ndim != 1 or coefficients.numel() != len(names):
                raise ValueError("coefficient tensor length must match objective_names")
            values = tuple(float(value) for value in coefficients.detach().to("cpu", torch.float64))
        else:
            values = tuple(float(value) for value in coefficients)
        if len(values) != len(names):
            raise ValueError("coefficient length must match objective_names")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("periodic ConFIG coefficients must be finite")
        if self._names is not None and names != self._names:
            raise ValueError("periodic ConFIG objective names or order changed")
        self._names = names
        self._coefficients = values
        self.last_refresh_step = self.completed_steps

    def reuse_objective(self, current_losses: Mapping[str, Tensor], *, scale: float) -> Tensor:
        """Build the fresh-loss signed scalar with exactly one scale division."""
        if self.refresh_due:
            raise RuntimeError("periodic ConFIG coefficients are absent or due for refresh")
        if not isinstance(current_losses, Mapping) or tuple(current_losses) != self._names:
            raise ValueError("current loss names and order must match refreshed objectives")
        if isinstance(scale, bool):
            raise TypeError("periodic ConFIG scalar scale must be finite and positive")
        try:
            denominator = float(scale)
        except (TypeError, ValueError) as error:
            raise TypeError("periodic ConFIG scalar scale must be finite and positive") from error
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("periodic ConFIG scalar scale must be finite and positive")
        if self._coefficients is None or self._names is None:
            raise RuntimeError("periodic ConFIG coefficients have not been initialized")

        terms: list[Tensor] = []
        reference: Tensor | None = None
        for name, coefficient in zip(self._names, self._coefficients):
            loss = current_losses[name]
            if not isinstance(loss, Tensor) or loss.numel() != 1 or not loss.is_floating_point():
                raise TypeError(f"current loss {name!r} must be a real scalar tensor")
            scalar = loss.reshape(())
            if not scalar.requires_grad:
                raise ValueError(f"current loss {name!r} must retain its fresh gradient graph")
            if not bool(torch.isfinite(scalar.detach())):
                raise FloatingPointError(f"current loss {name!r} is non-finite")
            if reference is not None and (scalar.device != reference.device or scalar.dtype != reference.dtype):
                raise ValueError("current losses must share a device and dtype")
            reference = scalar
            # A scalar tensor constant makes the detachment explicit and follows
            # the current loss dtype/device, preserving the existing backward route.
            terms.append(scalar * scalar.new_tensor(coefficient).detach())
        assert reference is not None
        return torch.stack(terms).sum() / denominator

    def advance(self) -> None:
        """Advance the cursor after one accepted optimizer update."""
        if self.refresh_due:
            raise RuntimeError("cannot advance periodic ConFIG state before refreshing")
        self.completed_steps += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "interval": self.interval,
            "compatibility_key": self.compatibility_key,
            "objective_names": None if self._names is None else list(self._names),
            "coefficients": None if self._coefficients is None else list(self._coefficients),
            "completed_steps": self.completed_steps,
            "last_refresh_step": self.last_refresh_step,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state only under the same coefficient and cadence contract."""
        if not isinstance(state, Mapping):
            raise TypeError("periodic ConFIG checkpoint must be a mapping")
        version = state.get("version")
        if isinstance(version, bool) or version != self.STATE_VERSION:
            raise ValueError("unsupported periodic ConFIG checkpoint version")
        saved_interval = state.get("interval")
        if (
            isinstance(saved_interval, bool)
            or not isinstance(saved_interval, int)
            or saved_interval != self.interval
        ):
            raise ValueError("periodic ConFIG checkpoint interval is incompatible")
        if state.get("compatibility_key") != self.compatibility_key:
            raise ValueError("periodic ConFIG checkpoint objective is incompatible")

        names_value = state.get("objective_names")
        coefficients_value = state.get("coefficients")
        if (names_value is None) != (coefficients_value is None):
            raise ValueError("periodic ConFIG checkpoint has incomplete coefficient state")
        if names_value is None:
            names = None
            coefficients = None
            if state.get("last_refresh_step") is not None:
                raise ValueError("periodic ConFIG checkpoint has a refresh cursor without coefficients")
        else:
            if not isinstance(names_value, (tuple, list)) or not isinstance(
                coefficients_value, (tuple, list)
            ):
                raise ValueError("periodic ConFIG checkpoint names and coefficients must be lists")
            names = tuple(names_value)
            coefficients = tuple(float(value) for value in coefficients_value)
            if (
                not names
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)
                or len(coefficients) != len(names)
                or not all(math.isfinite(value) for value in coefficients)
            ):
                raise ValueError("periodic ConFIG checkpoint coefficient state is invalid")

        completed = state.get("completed_steps")
        last_refresh = state.get("last_refresh_step")
        if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
            raise ValueError("periodic ConFIG checkpoint completed_steps is invalid")
        if last_refresh is not None and (
            isinstance(last_refresh, bool)
            or not isinstance(last_refresh, int)
            or last_refresh < 0
            or last_refresh > completed
        ):
            raise ValueError("periodic ConFIG checkpoint last_refresh_step is invalid")
        if coefficients is not None and last_refresh is None:
            raise ValueError("periodic ConFIG checkpoint is missing its coefficient refresh cursor")
        if last_refresh is not None and completed - last_refresh > self.interval:
            raise ValueError("periodic ConFIG checkpoint refresh cursor is stale beyond one interval")

        self._names = names
        self._coefficients = coefficients
        self.completed_steps = completed
        self.last_refresh_step = last_refresh


@dataclass(frozen=True)
class ResidualGradientDiagnostic:
    """First-order target-gradient component tangent to lower-gradient rows."""

    residual: Tensor
    rank: int
    residual_norm: float
    residual_norm_fraction: float
    lower_target_cosines: tuple[float, ...]
    improves_target_to_first_order: bool


def residual_gradient_diagnostic(
    lower_gradients: Tensor,
    target_gradient: Tensor,
) -> ResidualGradientDiagnostic:
    """Project a target gradient out of the lower-sector row span by QR/SVD.

    This is a parameter-gradient diagnostic only. It does not define a loss
    residual, a conditional distribution, or a production optimizer step.
    """
    lower = _real_matrix(lower_gradients, name="lower_gradients", allow_zero_rows=True)
    target = _vector(
        target_gradient,
        name="target_gradient",
        width=lower.shape[1],
        device=lower.device,
    )
    if not bool(torch.isfinite(lower).all()):
        raise FloatingPointError("lower_gradients contains non-finite values")

    if lower.shape[0] == 0:
        residual = target.clone()
        rank = 0
        singular_values = lower.new_empty((0,))
        basis = lower.new_empty((lower.shape[1], 0))
    else:
        _, singular_values, right = _thin_row_svd(lower)
        threshold, _ = _matrix_rank_tolerance(lower.shape, singular_values)
        retained = singular_values > threshold
        rank = int(retained.sum().detach().cpu())
        basis = right[:, retained]
        residual = target - basis @ (basis.T @ target)

    target_norm = float(torch.linalg.vector_norm(target).detach().cpu())
    residual_norm = float(torch.linalg.vector_norm(residual).detach().cpu())
    fraction = residual_norm / target_norm if target_norm > 0.0 else 0.0
    lower_norms = torch.linalg.vector_norm(lower, dim=1)
    denominator = lower_norms * torch.linalg.vector_norm(target)
    safe_denominator = torch.where(denominator > 0.0, denominator, torch.ones_like(denominator))
    cosines = (lower @ target) / safe_denominator
    observed_cosines = tuple(float(value) for value in cosines.detach().cpu().tolist())
    tolerance = max(lower.shape) * torch.finfo(torch.float64).eps * max(target_norm, 1.0)
    return ResidualGradientDiagnostic(
        residual=residual,
        rank=rank,
        residual_norm=residual_norm,
        residual_norm_fraction=fraction,
        lower_target_cosines=observed_cosines,
        improves_target_to_first_order=residual_norm > tolerance,
    )

