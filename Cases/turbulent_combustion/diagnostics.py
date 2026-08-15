"""Combustion topology verification and multi-field reconstruction diagnostics."""

from __future__ import annotations

import torch

from phycoflow_reconstruction.contracts import DataSpec, ObservationBatch
from phycoflow_reconstruction.data.normalization import FieldNormalizer


def verify_cartesian_topology(
    coordinates: torch.Tensor, expected_shape: tuple[int, int] = (100, 403)
) -> dict[str, object]:
    coordinates = coordinates.reshape(-1, 3).cpu()
    unique_x = torch.unique(coordinates[:, 0])
    unique_y = torch.unique(coordinates[:, 1])
    unique_z = torch.unique(coordinates[:, 2])
    pairs = torch.unique(coordinates[:, :2], dim=0)
    complete = (
        (unique_y.numel(), unique_x.numel()) == expected_shape
        and pairs.shape[0] == expected_shape[0] * expected_shape[1]
        and coordinates.shape[0] == pairs.shape[0]
    )
    return {
        "verified": bool(complete),
        "logical_shape": list(expected_shape),
        "unique_x": int(unique_x.numel()),
        "unique_y": int(unique_y.numel()),
        "unique_z": int(unique_z.numel()),
        "constant_z": bool(unique_z.numel() == 1),
        "canonical_order": "ascending y, then ascending x",
        "stored_order": "permuted Cartesian points",
    }


class CombustionDiagnostics:
    def __init__(self, data_spec: DataSpec, normalizer: FieldNormalizer) -> None:
        self.data_spec = data_spec
        self.normalizer = normalizer

    @torch.no_grad()
    def evaluate(
        self, prediction: torch.Tensor, batch: ObservationBatch
    ) -> dict[str, object]:
        physical = self.normalizer.decode(prediction)
        per_field = {}
        observed_fields = {
            int(value)
            for value in batch.obs_field_ids[batch.obs_valid_mask].unique().tolist()
        }
        for index, name in enumerate(self.data_spec.field_names):
            values = physical[..., index]
            per_field[name] = {
                "mean": float(values.mean().cpu()),
                "std": float(values.std().cpu()),
                "minimum": float(values.min().cpu()),
                "maximum": float(values.max().cpu()),
                "observed_input_field": index in observed_fields,
            }
        return {
            "topology": {
                "verified": self.data_spec.logical_shape == (100, 403),
                "logical_shape": list(self.data_spec.logical_shape),
                "coordinate_reorder": "lexicographic_yx",
            },
            "per_field_physical": per_field,
        }


def build_diagnostics_provider(
    data_spec: DataSpec, normalizer: FieldNormalizer
) -> CombustionDiagnostics:
    return CombustionDiagnostics(data_spec, normalizer)
