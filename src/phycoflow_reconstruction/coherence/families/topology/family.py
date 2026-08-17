"""Differentiable topology coherence on a fixed two-dimensional raster."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from math import cos, pi, sin
from typing import Any

import torch
from torch import nn

from ....contracts import (
    CoherenceComponentSpec,
    CoherenceFamilySpec,
    DataSpec,
    FamilyResult,
    TermResult,
)
from ....data.normalization import FieldNormalizer
from ...base import require_field_tensor
from .betti_curves import betti_curves, gaussian_blur
from .geometry import build_raster_map, coordinate_digest, rasterize_fields


class TopologyFamily(nn.Module):
    family_name = "topology"
    version = "1"

    def __init__(
        self,
        config: Mapping[str, Any],
        data_spec: DataSpec,
        normalizer: FieldNormalizer,
    ) -> None:
        super().__init__()
        self.config = dict(config)
        self.target_use = str(config.get("target_use", "paired_supervised"))
        self.units = str(config.get("units", "model_units"))
        self.family_weight = float(config.get("weight", 1.0))
        if self.target_use not in {"training_reference", "paired_supervised"}:
            raise ValueError("topology.target_use is invalid")
        if self.units not in {"model_units", "physical_units"}:
            raise ValueError("topology.units is invalid")
        lookup = {name: index for index, name in enumerate(data_spec.field_names)}
        self.field_names = tuple(config.get("fields") or data_spec.field_names)
        if len(set(self.field_names)) != len(self.field_names):
            raise ValueError("topology fields must be unique")
        unknown = sorted(set(self.field_names) - set(lookup))
        if unknown:
            raise KeyError(f"unknown topology fields: {unknown}")
        self.field_ids = tuple(lookup[name] for name in self.field_names)
        self.register_buffer("normalization_offset", normalizer.offset.clone())
        self.register_buffer("normalization_scale", normalizer.scale.clone())

        geometry = config.get("geometry", {})
        shape = geometry.get("grid_shape", (32, 32))
        self.grid_shape = (int(shape[0]), int(shape[1]))
        axes = geometry.get("axes", (0, 1))
        self.axes = (int(axes[0]), int(axes[1]))
        self.raster_neighbors = int(geometry.get("neighbors", 4))
        self.raster_power = float(geometry.get("power", 2.0))
        self.periodic = bool(geometry.get("periodic", False))
        self.register_buffer("neighbor_indices", torch.empty(0, 0, dtype=torch.long))
        self.register_buffer("neighbor_weights", torch.empty(0, 0))
        self.register_buffer("grid_coordinates", torch.empty(0, 0, 2))
        self.geometry_sha256: str | None = None

        filtration = config.get("filtration", {})
        quantiles = filtration.get("quantiles", (0.1, 0.25, 0.5, 0.75, 0.9))
        self.quantiles = tuple(float(value) for value in quantiles)
        if (
            not self.quantiles
            or any(not 0 < value < 1 for value in self.quantiles)
            or tuple(sorted(set(self.quantiles))) != self.quantiles
        ):
            raise ValueError("topology filtration quantiles must lie in (0,1)")
        self.dimensions = tuple(int(value) for value in filtration.get("dimensions", (0, 1)))
        if (
            not self.dimensions
            or not set(self.dimensions) <= {0, 1}
            or len(set(self.dimensions)) != len(self.dimensions)
        ):
            raise ValueError("topology filtration dimensions must be drawn from {0,1}")
        self.directions = tuple(filtration.get("directions", ("superlevel", "sublevel")))
        if (
            not self.directions
            or not set(self.directions) <= {"superlevel", "sublevel"}
            or len(set(self.directions)) != len(self.directions)
        ):
            raise ValueError("topology filtration directions are invalid")
        self.sharpness = float(filtration.get("sharpness", 12.0))
        self.smoothing_sigma = float(filtration.get("smoothing_sigma", 0.8))
        if self.sharpness <= 0 or self.smoothing_sigma < 0:
            raise ValueError("topology sharpness must be positive and smoothing non-negative")

        components = config.get("components", {})
        self_settings = components.get("self", {})
        mutual_settings = components.get("mutual", {})
        self.component_weights: dict[str, float] = {}
        specs = []
        if bool(self_settings.get("enabled", True)):
            self.component_weights["self"] = float(self_settings.get("weight", 1.0))
            specs.append(
                CoherenceComponentSpec(
                    "self.betti_curves",
                    self.target_use,
                    self.units,
                    True,
                    required_geometry="fixed_2d_raster",
                    aggregation="per_sample",
                    metadata={"homology_dimensions": self.dimensions},
                )
            )
        if bool(mutual_settings.get("enabled", len(self.field_names) >= 2)):
            self.component_weights["mutual"] = float(mutual_settings.get("weight", 1.0))
            configured_pairs = mutual_settings.get("pairs") or list(combinations(self.field_names, 2))
            try:
                self.mutual_pairs = tuple(
                    (self.field_names.index(left), self.field_names.index(right))
                    for left, right in configured_pairs
                )
            except ValueError as error:
                raise KeyError("topology mutual pair contains a field outside fields") from error
            self.fibered_lines = int(mutual_settings.get("lines", 3))
            self.theta_min_degrees = float(mutual_settings.get("theta_min_degrees", 15.0))
            if (
                not self.mutual_pairs
                or any(left == right for left, right in self.mutual_pairs)
                or len({tuple(sorted(pair)) for pair in self.mutual_pairs})
                != len(self.mutual_pairs)
                or self.fibered_lines < 1
                or not 0.0 < self.theta_min_degrees < 45.0
            ):
                raise ValueError("topology mutual component requires pairs and lines>=1")
            specs.append(
                CoherenceComponentSpec(
                    "mutual.fibered_betti_curves",
                    self.target_use,
                    self.units,
                    True,
                    required_geometry="fixed_2d_raster",
                    aggregation="per_sample",
                    metadata={"homology_dimensions": self.dimensions},
                )
            )
        else:
            self.mutual_pairs = ()
            self.fibered_lines = 0
            self.theta_min_degrees = 15.0
        if not self.component_weights or any(value < 0 for value in self.component_weights.values()):
            raise ValueError("topology must enable non-negative component weights")
        if not any(self.component_weights.values()):
            raise ValueError("topology must have a positive-weight component")
        self.spec = CoherenceFamilySpec(
            self.family_name,
            self.version,
            tuple(specs),
            metadata={"aggregation": "per_sample", "target_use": self.target_use},
        )
        self.spec.validate()

    def _raster_map(self, coordinates: torch.Tensor) -> None:
        if coordinates.ndim != 3:
            raise ValueError("topology coordinates must have shape [B,N,D]")
        if not torch.equal(coordinates, coordinates[:1].expand_as(coordinates)):
            raise ValueError("topology requires identical coordinates for every sample")
        digest = coordinate_digest(coordinates[0])
        if not self.neighbor_indices.numel():
            mapping = build_raster_map(
                coordinates[0],
                grid_shape=self.grid_shape,
                axes=self.axes,
                neighbors=self.raster_neighbors,
                power=self.raster_power,
                periodic=self.periodic,
            )
            self.neighbor_indices = mapping.neighbor_indices.to(coordinates.device)
            self.neighbor_weights = mapping.neighbor_weights.to(coordinates.device)
            self.grid_coordinates = mapping.grid_coordinates.to(coordinates.device)
            self.geometry_sha256 = mapping.coordinate_sha256
        elif digest != self.geometry_sha256:
            raise ValueError("topology coordinates changed after raster-map construction")

    def _units_and_fields(
        self, generated: torch.Tensor, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.units == "physical_units":
            offset = self.normalization_offset.to(generated)
            scale = self.normalization_scale.to(generated)
            generated = generated * scale + offset
            reference = reference * scale + offset
        return generated[..., self.field_ids], reference[..., self.field_ids]

    def _curve_cost(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-sample curve MSE for `[B,H,W]` fields."""
        costs = []
        quantiles = torch.as_tensor(self.quantiles, device=generated.device, dtype=generated.dtype)
        for batch_index in range(generated.shape[0]):
            sample_costs = []
            for direction in self.directions:
                sign = 1.0 if direction == "superlevel" else -1.0
                generated_field = sign * generated[batch_index]
                reference_field = sign * reference[batch_index]
                levels = torch.quantile(reference_field.detach().reshape(-1), quantiles)
                generated_curves = betti_curves(
                    generated_field,
                    levels,
                    self.dimensions,
                    sharpness=self.sharpness,
                    periodic=self.periodic,
                )
                reference_curves = betti_curves(
                    reference_field,
                    levels,
                    self.dimensions,
                    sharpness=self.sharpness,
                    periodic=self.periodic,
                )
                for dimension in self.dimensions:
                    sample_costs.append(
                        (generated_curves[dimension] - reference_curves[dimension]).square().mean()
                    )
            costs.append(torch.stack(sample_costs).mean())
        return torch.stack(costs)

    def _self_cost(self, generated: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [self._curve_cost(generated[:, field], reference[:, field])
             for field in range(generated.shape[1])]
        ).mean(dim=0)

    def _mutual_cost(self, generated: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        theta_min = self.theta_min_degrees * pi / 180.0
        angle_step = (pi / 2.0 - 2.0 * theta_min) / (self.fibered_lines + 1.0)
        angles = tuple(theta_min + angle_step * (index + 1) for index in range(self.fibered_lines))
        all_costs = []
        for left, right in self.mutual_pairs:
            reference_left = reference[:, left]
            reference_right = reference[:, right]
            mean_left = reference_left.detach().mean(dim=(1, 2), keepdim=True)
            mean_right = reference_right.detach().mean(dim=(1, 2), keepdim=True)
            std_left = reference_left.detach().std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            std_right = reference_right.detach().std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            generated_left = (generated[:, left] - mean_left) / std_left
            generated_right = (generated[:, right] - mean_right) / std_right
            reference_left = (reference_left - mean_left) / std_left
            reference_right = (reference_right - mean_right) / std_right
            line_costs = []
            for line_index, angle in enumerate(angles):
                fraction = (line_index + 1.0) / (self.fibered_lines + 1.0)
                offset_left = torch.quantile(reference_left.detach().flatten(1), fraction, dim=1)
                offset_right = torch.quantile(
                    reference_right.detach().flatten(1), 1.0 - fraction, dim=1
                )
                velocity_left = max(cos(angle), 1e-6)
                velocity_right = max(sin(angle), 1e-6)
                generated_push = torch.minimum(
                    (generated_left - offset_left[:, None, None]) / velocity_left,
                    (generated_right - offset_right[:, None, None]) / velocity_right,
                )
                reference_push = torch.minimum(
                    (reference_left - offset_left[:, None, None]) / velocity_left,
                    (reference_right - offset_right[:, None, None]) / velocity_right,
                )
                line_costs.append(self._curve_cost(generated_push, reference_push))
            all_costs.append(torch.stack(line_costs).mean(dim=0))
        return torch.stack(all_costs).mean(dim=0)

    def forward(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor,
        *,
        coordinates: torch.Tensor | None = None,
        context: Any | None = None,
    ) -> FamilyResult:
        require_field_tensor("generated", generated)
        require_field_tensor("reference", reference)
        if generated.shape != reference.shape:
            raise ValueError("topology generated/reference shapes differ")
        if coordinates is None or coordinates.shape[:2] != generated.shape[:2]:
            raise ValueError("topology requires coordinates aligned with [B,N]")
        self._raster_map(coordinates)
        generated, reference = self._units_and_fields(generated, reference)
        generated_grid = rasterize_fields(
            generated,
            self.neighbor_indices,
            self.neighbor_weights.to(generated),
            self.grid_shape,
        )
        reference_grid = rasterize_fields(
            reference,
            self.neighbor_indices,
            self.neighbor_weights.to(reference),
            self.grid_shape,
        )
        generated_grid = gaussian_blur(generated_grid, self.smoothing_sigma, self.periodic)
        reference_grid = gaussian_blur(reference_grid, self.smoothing_sigma, self.periodic)
        results: dict[str, TermResult] = {}
        per_sample = generated.sum(dim=(1, 2)) * 0.0
        if "self" in self.component_weights:
            cost = self._self_cost(generated_grid, reference_grid)
            results[f"{self.family_name}.self.betti_curves"] = TermResult(
                cost,
                cost.mean(),
                diagnostics={"dimensions": self.dimensions, "directions": self.directions},
            )
            per_sample = per_sample + self.component_weights["self"] * cost
        if "mutual" in self.component_weights:
            cost = self._mutual_cost(generated_grid, reference_grid)
            results[f"{self.family_name}.mutual.fibered_betti_curves"] = TermResult(
                cost,
                cost.mean(),
                diagnostics={"pairs": self.mutual_pairs, "lines": self.fibered_lines},
            )
            per_sample = per_sample + self.component_weights["mutual"] * cost
        if not torch.isfinite(per_sample).all():
            raise FloatingPointError("topology family produced a non-finite cost")
        return FamilyResult(
            component_results=results,
            per_sample_cost=per_sample,
            scalar_loss=per_sample.mean(),
            diagnostics={
                "family": self.family_name,
                "version": self.version,
                "aggregation": "per_sample",
                "target_use": self.target_use,
                "units": self.units,
                "fields": self.field_names,
                "grid_shape": self.grid_shape,
                "periodic": self.periodic,
                "geometry_sha256": self.geometry_sha256,
                "component_weights": dict(self.component_weights),
            },
        )

    def state_artifact(self) -> dict[str, Any]:
        return {
            "family": self.family_name,
            "version": self.version,
            "scientific_source": {
                "repository": "https://github.com/jachen25/PhyCoFlow_dev/tree/main/src",
                "revision": "ab49ea37a",
                "license_status": "not_declared; independent implementation",
            },
            "config": self.config,
            "field_names": self.field_names,
            "target_use": self.target_use,
            "units": self.units,
            "geometry_sha256": self.geometry_sha256,
            "state_dict": self.state_dict(),
        }

    def load_state_artifact(self, artifact: Mapping[str, Any]) -> None:
        if artifact.get("family") != self.family_name or artifact.get("version") != self.version:
            raise ValueError("topology family artifact identity mismatch")
        if dict(artifact.get("config", {})) != self.config:
            raise ValueError("topology family artifact config mismatch")
        state = artifact["state_dict"]
        device = self.normalization_offset.device
        self.neighbor_indices = state["neighbor_indices"].to(device)
        self.neighbor_weights = state["neighbor_weights"].to(device)
        self.grid_coordinates = state["grid_coordinates"].to(device)
        self.geometry_sha256 = artifact.get("geometry_sha256")
        self.load_state_dict(state, strict=True)
