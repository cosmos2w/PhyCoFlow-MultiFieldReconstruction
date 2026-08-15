"""Periodic flow diagnostics for reconstructed Kolmogorov snapshots."""

from __future__ import annotations

import math

import torch

from phycoflow_reconstruction.contracts import DataSpec, ObservationBatch
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.physics.operators import (
    periodic_derivative,
    periodic_laplacian_2d,
    relative_rms,
    reshape_fields,
)


class KolmogorovDiagnostics:
    def __init__(self, data_spec: DataSpec, normalizer: FieldNormalizer) -> None:
        if tuple(data_spec.field_names) != ("u", "v", "p"):
            raise ValueError("Kolmogorov diagnostics require u,v,p field order")
        self.data_spec = data_spec
        self.normalizer = normalizer
        self.domain_length = 2.0 * math.pi

    @torch.no_grad()
    def evaluate(
        self, prediction: torch.Tensor, batch: ObservationBatch
    ) -> dict[str, float | str]:
        physical = self.normalizer.decode(prediction)
        fields = reshape_fields(physical, self.data_spec.logical_shape)
        u, v, pressure = fields[:, 0], fields[:, 1], fields[:, 2]
        ux = periodic_derivative(u, axis=-1, domain_length=self.domain_length)
        uy = periodic_derivative(u, axis=-2, domain_length=self.domain_length)
        vx = periodic_derivative(v, axis=-1, domain_length=self.domain_length)
        vy = periodic_derivative(v, axis=-2, domain_length=self.domain_length)
        vorticity = vx - uy
        divergence = ux + vy
        result: dict[str, float | str] = {
            "divergence_rms": float(divergence.square().mean().sqrt().cpu()),
            "pressure_mean_abs": float(pressure.mean(dim=(-2, -1)).abs().mean().cpu()),
            "kinetic_energy": float((0.5 * (u.square() + v.square()).mean()).cpu()),
            "enstrophy": float((0.5 * vorticity.square().mean()).cpu()),
            "boundary_operator": "periodic_spectral",
            "pressure_gauge": "zero_spatial_mean",
        }
        context = batch.metadata.get("sample_context", {})
        auxiliary = context.get("auxiliary", {}) if isinstance(context, dict) else {}
        stored_vorticity = (
            auxiliary.get("vorticity") if isinstance(auxiliary, dict) else None
        )
        if isinstance(stored_vorticity, torch.Tensor):
            target = stored_vorticity.to(prediction.device).reshape_as(vorticity)
            result["vorticity_mse"] = float((vorticity - target).square().mean().cpu())

        physics = context.get("physics", {}) if isinstance(context, dict) else {}
        conditions = context.get("conditions") if isinstance(context, dict) else None
        derivative = physics.get("temporal_derivative") if isinstance(physics, dict) else None
        if isinstance(derivative, torch.Tensor) and isinstance(conditions, torch.Tensor):
            time_derivative = reshape_fields(
                derivative.to(prediction.device, prediction.dtype),
                self.data_spec.logical_shape,
            )
            u_t, v_t = time_derivative[:, 0], time_derivative[:, 1]
            conditions = conditions.to(prediction.device, prediction.dtype)
            reynolds = conditions[:, 0, None, None]
            amplitude = conditions[:, 1, None, None]
            wavenumber = conditions[:, 2, None, None]
            y = torch.arange(
                u.shape[-2], device=u.device, dtype=u.dtype
            ) * (self.domain_length / u.shape[-2])
            force_x = amplitude * torch.sin(wavenumber * y[None, :, None])
            px = periodic_derivative(
                pressure, axis=-1, domain_length=self.domain_length
            )
            py = periodic_derivative(
                pressure, axis=-2, domain_length=self.domain_length
            )
            residual_u = (
                u_t
                + u * ux
                + v * uy
                + px
                - periodic_laplacian_2d(u, self.domain_length) / reynolds
                - force_x
            )
            residual_v = (
                v_t
                + u * vx
                + v * vy
                + py
                - periodic_laplacian_2d(v, self.domain_length) / reynolds
            )
            result["relative_momentum_residual_u"] = float(
                relative_rms(residual_u, u_t).cpu()
            )
            result["relative_momentum_residual_v"] = float(
                relative_rms(residual_v, v_t).cpu()
            )
            result["temporal_derivative_source"] = "paired_finite_difference"
        else:
            result["momentum_residual_availability"] = (
                "requires paired_finite_difference temporal context"
            )
        return result


def build_diagnostics_provider(
    data_spec: DataSpec, normalizer: FieldNormalizer
) -> KolmogorovDiagnostics:
    return KolmogorovDiagnostics(data_spec, normalizer)
