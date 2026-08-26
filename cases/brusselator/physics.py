"""Verified periodic Brusselator residuals and reconstruction diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from phycoflow_reconstruction.contracts import DataSpec, LossBundle, ObservationBatch
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.physics.operators import (
    periodic_laplacian_2d,
    relative_rms,
    reshape_fields,
)


class BrusselatorPhysics:
    """Reaction-diffusion residual using paired finite-difference time context."""

    required_query_layout = "full_grid"

    def __init__(
        self,
        settings: Mapping[str, Any],
        data_spec: DataSpec,
        normalizer: FieldNormalizer,
    ) -> None:
        if tuple(data_spec.field_names) != ("u", "v") or len(data_spec.logical_shape) != 2:
            raise ValueError("Brusselator physics requires full 2-D fields ordered as u,v")
        self.settings = dict(settings)
        self.data_spec = data_spec
        self.normalizer = normalizer
        self.domain_length = float(settings.get("domain_length", 20.0))
        self.weights = {
            "pde_u": float(settings.get("weights", {}).get("pde_u", 1.0)),
            "pde_v": float(settings.get("weights", {}).get("pde_v", 1.0)),
            "positivity": float(settings.get("weights", {}).get("positivity", 0.01)),
        }
        source = settings.get("temporal_derivative_source", "paired_finite_difference")
        if source != "paired_finite_difference":
            raise ValueError(
                "Brusselator snapshot residual currently requires explicitly labeled "
                "paired_finite_difference temporal context"
            )

    def _terms(
        self, prediction: torch.Tensor, batch: ObservationBatch
    ) -> dict[str, torch.Tensor]:
        context = batch.metadata.get("sample_context", {})
        physics = context.get("physics", {}) if isinstance(context, dict) else {}
        derivative = physics.get("temporal_derivative") if isinstance(physics, dict) else None
        conditions = context.get("conditions") if isinstance(context, dict) else None
        if not isinstance(derivative, torch.Tensor) or not isinstance(conditions, torch.Tensor):
            raise TypeError(
                "Brusselator physics requires temporal_derivative and conditions in batch context"
            )
        physical = self.normalizer.decode(prediction)
        fields = reshape_fields(physical, self.data_spec.logical_shape)
        time_derivative = reshape_fields(
            derivative.to(prediction.device, prediction.dtype), self.data_spec.logical_shape
        )
        u, v = fields[:, 0], fields[:, 1]
        u_t, v_t = time_derivative[:, 0], time_derivative[:, 1]
        conditions = conditions.to(prediction.device, prediction.dtype)
        a, b, diffusivity_u, diffusivity_v = (
            conditions[:, index, None, None] for index in range(4)
        )
        reaction_u = a - (b + 1.0) * u + u.square() * v
        reaction_v = b * u - u.square() * v
        residual_u = (
            u_t
            - diffusivity_u * periodic_laplacian_2d(u, self.domain_length)
            - reaction_u
        )
        residual_v = (
            v_t
            - diffusivity_v * periodic_laplacian_2d(v, self.domain_length)
            - reaction_v
        )
        scale_u = u_t.square().mean().sqrt().detach().clamp_min(1e-6)
        scale_v = v_t.square().mean().sqrt().detach().clamp_min(1e-6)
        return {
            "pde_u": (residual_u / scale_u).square().mean(),
            "pde_v": (residual_v / scale_v).square().mean(),
            "positivity": torch.relu(-fields).square().mean(),
            "relative_residual_u": relative_rms(residual_u, u_t),
            "relative_residual_v": relative_rms(residual_v, v_t),
        }

    def loss(self, prediction: torch.Tensor, batch: ObservationBatch) -> LossBundle:
        terms = self._terms(prediction, batch)
        total = sum(self.weights[name] * terms[name] for name in self.weights)
        return LossBundle(
            total,
            {
                "physics_pde_u": terms["pde_u"],
                "physics_pde_v": terms["pde_v"],
                "physics_positivity": terms["positivity"],
            },
            diagnostics={
                "relative_residual_u": terms["relative_residual_u"].detach(),
                "relative_residual_v": terms["relative_residual_v"].detach(),
                "boundary_operator": "periodic_spectral",
                "temporal_derivative_source": "paired_finite_difference",
            },
        )

    @torch.no_grad()
    def evaluate(
        self,
        prediction: torch.Tensor,
        batch: ObservationBatch,
    ) -> dict[str, float | str]:
        terms = self._terms(prediction, batch)
        return {
            "relative_pde_residual_u": float(terms["relative_residual_u"].cpu()),
            "relative_pde_residual_v": float(terms["relative_residual_v"].cpu()),
            "positivity_violation_mse": float(terms["positivity"].cpu()),
            "boundary_operator": "periodic_spectral",
            "temporal_derivative_source": "paired_finite_difference",
        }


def build_physics_provider(
    settings: Mapping[str, Any],
    data_spec: DataSpec,
    normalizer: FieldNormalizer,
) -> BrusselatorPhysics:
    return BrusselatorPhysics(settings, data_spec, normalizer)


def build_diagnostics_provider(
    data_spec: DataSpec, normalizer: FieldNormalizer
) -> BrusselatorPhysics:
    return BrusselatorPhysics(
        {"temporal_derivative_source": "paired_finite_difference"},
        data_spec,
        normalizer,
    )
