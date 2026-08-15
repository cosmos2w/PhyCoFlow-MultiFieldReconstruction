"""Space/time-resolved diagnostics for KS quasi-super-resolution."""

from __future__ import annotations

import torch

from phycoflow_reconstruction.contracts import DataSpec, ObservationBatch
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.physics.operators import periodic_derivative, reshape_fields


class KSSpaceTimeDiagnostics:
    def __init__(self, data_spec: DataSpec, normalizer: FieldNormalizer) -> None:
        if data_spec.reconstruction_unit != "space_time_trajectory":
            raise ValueError("KS diagnostics require a complete space-time trajectory")
        if tuple(data_spec.logical_shape) != (401, 256):
            raise ValueError("KS diagnostics require reversible logical shape (401,256)")
        self.data_spec = data_spec
        self.normalizer = normalizer

    @torch.no_grad()
    def evaluate(
        self, prediction: torch.Tensor, batch: ObservationBatch
    ) -> dict[str, float | bool | list[int]]:
        if batch.target_fields is None:
            raise ValueError("KS evaluation diagnostics require the held-out dense target")
        physical_prediction = self.normalizer.decode(prediction)
        physical_target = self.normalizer.decode(batch.target_fields)
        predicted = reshape_fields(
            physical_prediction, self.data_spec.logical_shape
        )[:, 0]
        target = reshape_fields(physical_target, self.data_spec.logical_shape)[:, 0]
        error = (predicted - target).square()
        per_time = error.mean(dim=-1)
        per_space = error.mean(dim=-2)
        times = batch.metadata.get("sample_context", {}).get("time", ())
        if not times or not isinstance(times[0], torch.Tensor):
            raise ValueError("KS diagnostics require stored time coordinates")
        time_values = times[0].to(prediction.device, prediction.dtype)
        temporal_prediction = torch.gradient(
            predicted, spacing=(time_values,), dim=(-2,)
        )[0]
        temporal_target = torch.gradient(target, spacing=(time_values,), dim=(-2,))[0]
        spatial_prediction = periodic_derivative(
            predicted, axis=-1, domain_length=60.0
        )
        spatial_target = periodic_derivative(target, axis=-1, domain_length=60.0)
        flattened = predicted.reshape(predicted.shape[0], -1, 1)
        return {
            "logical_shape": [401, 256],
            "layout_roundtrip_exact": bool(torch.equal(flattened, physical_prediction)),
            "per_time_mse_mean": float(per_time.mean().cpu()),
            "per_time_mse_max": float(per_time.max().cpu()),
            "per_time_mse_std": float(per_time.std().cpu()),
            "per_space_mse_mean": float(per_space.mean().cpu()),
            "per_space_mse_max": float(per_space.max().cpu()),
            "temporal_derivative_mse": float(
                (temporal_prediction - temporal_target).square().mean().cpu()
            ),
            "spatial_derivative_mse": float(
                (spatial_prediction - spatial_target).square().mean().cpu()
            ),
        }


def build_diagnostics_provider(
    data_spec: DataSpec, normalizer: FieldNormalizer
) -> KSSpaceTimeDiagnostics:
    return KSSpaceTimeDiagnostics(data_spec, normalizer)
