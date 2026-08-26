"""Nonperiodic flow/transport diagnostics for the integration-only fixture."""

from __future__ import annotations

import torch

from phycoflow_reconstruction.contracts import DataSpec, ObservationBatch
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.physics.operators import (
    nonperiodic_gradient_2d,
    reshape_fields,
)


class MassTransportDiagnostics:
    def __init__(self, data_spec: DataSpec, normalizer: FieldNormalizer) -> None:
        if tuple(data_spec.field_names) != ("u_x", "u_y", "concentration"):
            raise ValueError("mass-transport diagnostics require u_x,u_y,concentration")
        self.data_spec = data_spec
        self.normalizer = normalizer

    @torch.no_grad()
    def evaluate(
        self, prediction: torch.Tensor, batch: ObservationBatch
    ) -> dict[str, float | str | bool]:
        fields = reshape_fields(
            self.normalizer.decode(prediction), self.data_spec.logical_shape
        )
        velocity_x, velocity_y, concentration = fields[:, 0], fields[:, 1], fields[:, 2]
        spacing = 300.0 / (self.data_spec.logical_shape[-1] - 1)
        du_dx, _ = nonperiodic_gradient_2d(velocity_x, spacing, spacing)
        _, dv_dy = nonperiodic_gradient_2d(velocity_y, spacing, spacing)
        boundary_flux = torch.cat(
            (
                velocity_x[..., :, 0].reshape(-1),
                velocity_x[..., :, -1].reshape(-1),
                velocity_y[..., 0, :].reshape(-1),
                velocity_y[..., -1, :].reshape(-1),
            )
        )
        result: dict[str, float | str | bool] = {
            "benchmark_eligible": False,
            "boundary_condition": "nonperiodic_no_flow",
            "divergence_rms": float((du_dx + dv_dy).square().mean().sqrt().cpu()),
            "boundary_normal_flux_rms": float(boundary_flux.square().mean().sqrt().cpu()),
            "concentration_minimum": float(concentration.min().cpu()),
            "concentration_maximum": float(concentration.max().cpu()),
            "concentration_spatial_mean": float(concentration.mean().cpu()),
        }
        context = batch.metadata.get("sample_context", {})
        auxiliary = context.get("auxiliary", {}) if isinstance(context, dict) else {}
        pressure = auxiliary.get("pressure") if isinstance(auxiliary, dict) else None
        source = auxiliary.get("source_field") if isinstance(auxiliary, dict) else None
        if isinstance(pressure, torch.Tensor):
            result["reference_pressure_mean_abs"] = float(pressure.mean().abs().cpu())
        if isinstance(source, torch.Tensor):
            result["reference_source_integral"] = float(source.sum().cpu())
        return result


def build_diagnostics_provider(
    data_spec: DataSpec, normalizer: FieldNormalizer
) -> MassTransportDiagnostics:
    return MassTransportDiagnostics(data_spec, normalizer)
