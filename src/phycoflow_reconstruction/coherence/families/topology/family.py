"""Differentiable topology coherence on a fixed two-dimensional raster."""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import nullcontext
from copy import deepcopy
from itertools import combinations
from math import ceil, cos, pi, sin
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
from .finite_primary import FinitePrimaryTopologyObjective
from .geometry import (
    build_physical_cell_overlap_map,
    build_raster_map,
    coordinate_digest,
    rasterize_fields,
)
from .persistence import persistence_source
from .persistence_objective import PersistenceTopologyObjective
from .provenance import SOURCE_SNAPSHOT
from .spatial_objective import SpatialTopologyObjective

_LEGACY_GEOMETRY_SOURCE = {
    "family.py": "0c80a1a350939dc143b594990216fa454f28526f467adb354befd140543edfda",
    "geometry.py": "eed9bb67a0aac7114982d5d1af1055ea5780a1d365a0584611a317b0dc9370c7",
}


def _compatible_persistence_source(actual, expected, *, legacy_geometry):
    """Accept the immediately preceding legacy map while keeping other bindings strict."""
    if actual == expected:
        return True
    if not legacy_geometry or not isinstance(actual, Mapping):
        return False
    previous = deepcopy(expected)
    previous["source_sha256"].update(_LEGACY_GEOMETRY_SOURCE)
    return actual == previous


class TopologyFamily(nn.Module):
    family_name = "topology"
    version = "1"

    @property
    def needs_source_calibration(self) -> bool:
        """Whether the new leaf scales still need frozen TRAIN/SOURCE fitting."""
        return self.version == "4" and not bool(self.spatial_objective.calibrated)

    def __init__(
        self,
        config: Mapping[str, Any],
        data_spec: DataSpec,
        normalizer: FieldNormalizer,
    ) -> None:
        super().__init__()
        self.config = deepcopy(dict(config))
        config = self.config
        # Old resolved configurations retain their scientific definition and v1 artifacts.
        self.strategy = str(config.get("strategy", "betti_curves"))
        if self.strategy not in {"betti_curves", "spatial_self_mutual", "cubical_persistence"}:
            raise ValueError("unknown topology.strategy")
        self.version = {
            "betti_curves": "1",
            "spatial_self_mutual": "2",
            "cubical_persistence": "3",
        }[self.strategy]
        if (
            self.strategy == "cubical_persistence"
            and config.get("persistence", {}).get("definition", "legacy_v3") == "finite_primary_v1"
        ):
            self.version = "4"
        self.target_use = str(config.get("target_use", "paired_supervised"))
        self.units = str(config.get("units", "model_units"))
        self.family_weight = float(config.get("weight", 1.0))
        if not math.isfinite(self.family_weight) or self.family_weight <= 0:
            raise ValueError("topology.weight must be positive")
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
        configured_periods = geometry.get("periods")
        if configured_periods is not None and (
            not isinstance(configured_periods, (list, tuple)) or len(configured_periods) != 2
        ):
            raise ValueError("topology geometry.periods must contain exactly two values")
        self.periods = None if configured_periods is None else tuple(map(float, configured_periods))
        if not self.periodic and self.periods is not None:
            raise ValueError("topology geometry.periods requires periodic=true")
        self.allow_projected_collisions = bool(geometry.get("allow_projected_collisions", False))
        self.antialias_downsample = geometry.get("antialias_downsample", False)
        self.raster_mapping = geometry.get("raster_mapping", "legacy_v1")
        self.native_masked = geometry.get("native_masked", False)
        self.structured_indices = geometry.get("structured_indices")
        if self.structured_indices is not None and (
            not self.native_masked or not isinstance(self.structured_indices, str)
        ):
            raise ValueError(
                "topology structured_indices requires native_masked and an auxiliary array name"
            )
        if not isinstance(self.native_masked, bool):
            raise TypeError("topology geometry.native_masked must be boolean")
        if self.native_masked:
            if (
                self.strategy != "cubical_persistence"
                or self.periodic
                or self.antialias_downsample
                or float(config.get("filtration", {}).get("smoothing_sigma", 0.8)) != 0
                or config.get("persistence", {}).get("descriptors")
            ):
                raise ValueError(
                    "native_masked requires nonperiodic cubical persistence, zero smoothing, no downsampling or derived descriptors"
                )
            self.register_buffer("domain_mask", torch.empty(0, 0, dtype=torch.bool))
        if not isinstance(self.antialias_downsample, bool):
            raise TypeError("topology geometry.antialias_downsample must be boolean")
        if self.raster_mapping not in {"legacy_v1", "physical_cell_overlap_v1"}:
            raise ValueError("unknown topology geometry.raster_mapping")
        if self.raster_mapping == "physical_cell_overlap_v1" and (
            self.periodic or self.native_masked or self.antialias_downsample
        ):
            raise ValueError(
                "physical_cell_overlap_v1 requires a complete nonperiodic grid and excludes legacy antialias"
            )
        if self.antialias_downsample and self.strategy == "betti_curves":
            raise ValueError("topology antialias_downsample requires topology v2 or v3")
        self.register_buffer("neighbor_indices", torch.empty(0, 0, dtype=torch.long))
        self.register_buffer("neighbor_weights", torch.empty(0, 0))
        self.register_buffer("grid_coordinates", torch.empty(0, 0, 2))
        self.geometry_sha256: str | None = None
        self.geometry_diagnostics: dict[str, float | int | str] = {}

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
        if (
            not math.isfinite(self.sharpness)
            or not math.isfinite(self.smoothing_sigma)
            or self.sharpness <= 0
            or self.smoothing_sigma < 0
        ):
            raise ValueError("topology sharpness must be positive and smoothing non-negative")
        self.smoothing_radius = (
            max(1, round(3.0 * self.smoothing_sigma))
            if self.strategy == "spatial_self_mutual"
            else None
        )
        smoothing_radius = self.smoothing_radius or ceil(3.0 * self.smoothing_sigma)
        if (
            not self.periodic
            and self.smoothing_sigma > 0
            and (smoothing_radius >= self.grid_shape[0] or smoothing_radius >= self.grid_shape[1])
        ):
            raise ValueError(
                "topology nonperiodic smoothing radius must be smaller than both grid dimensions"
            )

        self.spatial_objective = None
        if self.strategy == "cubical_persistence":
            objective_type = (
                FinitePrimaryTopologyObjective
                if self.version == "4"
                else PersistenceTopologyObjective
            )
            self.spatial_objective = objective_type(config, data_spec.field_names)
            self.component_weights = dict(self.spatial_objective.weights)
            self.spec = self.spatial_objective.spec
            return
        if self.strategy == "spatial_self_mutual":
            self.spatial_objective = SpatialTopologyObjective(config, data_spec.field_names)
            self.component_weights = dict(self.spatial_objective.weights)
            self.spec = self.spatial_objective.spec
            return

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
            configured_pairs = mutual_settings.get("pairs") or list(
                combinations(self.field_names, 2)
            )
            try:
                self.mutual_pairs = tuple(
                    (self.field_names.index(left), self.field_names.index(right))
                    for left, right in configured_pairs
                )
            except ValueError as error:
                raise KeyError("topology mutual pair contains a field outside fields") from error
            self.fibered_lines = int(mutual_settings.get("lines", 3))
            self.theta_min_degrees = float(mutual_settings.get("theta_min_degrees", 15.0))
            self.mutual_axis_tolerance = float(mutual_settings.get("axis_tolerance", 1e-6))
            if (
                not self.mutual_pairs
                or any(left == right for left, right in self.mutual_pairs)
                or len({tuple(sorted(pair)) for pair in self.mutual_pairs})
                != len(self.mutual_pairs)
                or self.fibered_lines < 1
                or not 0.0 < self.theta_min_degrees < 45.0
                or self.mutual_axis_tolerance <= 0
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
            self.mutual_axis_tolerance = 1e-6
        if not self.component_weights or any(
            value < 0 for value in self.component_weights.values()
        ):
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

    @torch.no_grad()
    def fit_calibration(self, prediction, reference, *, coordinates, provenance):
        """Freeze raw leaf scales from a paired TRAIN/SOURCE batch, once."""
        if self.version != "4":
            raise ValueError("leaf calibration requires finite_primary_v1")
        self.forward(
            prediction,
            reference,
            coordinates=coordinates,
            context={"_finite_primary_calibration": provenance},
        )
        return deepcopy(self.spatial_objective.calibration_record)

    def fit_training_reference(self, dataset, *, query_indices, device=None) -> None:
        """Freeze configured topology geometry from the training support."""
        if not self.native_masked and self.raster_mapping == "legacy_v1":
            return
        if dataset.split_name != "train" or dataset.selection.split != "train":
            raise ValueError("native topology geometry must be fitted on the training split")
        sample = dataset[0]
        if sample.valid_points is not None and not bool(sample.valid_points.all()):
            raise ValueError(
                "native_masked topology requires a compact all-valid provider; "
                "full-grid invalid nodes are unsupported"
            )
        query = torch.as_tensor(query_indices, dtype=torch.long).cpu()
        if device is not None:
            self.to(device)
        if self.raster_mapping == "physical_cell_overlap_v1":
            self.fit_physical_raster(
                sample.coordinates_raw,
                query,
                query_coordinates=sample.coordinates[query],
            )
            return
        if self.structured_indices is None:
            return
        indices = sample.metadata.get("auxiliary", {})[self.structured_indices][query]
        self._raster_map(
            sample.coordinates[query][None].to(self.normalization_offset.device),
            native_grid_indices=indices,
        )

    def fit_physical_raster(self, native_coordinates, query_indices, *, query_coordinates):
        """Freeze a raw-coordinate native overlap map for the selected model queries."""
        if self.raster_mapping != "physical_cell_overlap_v1":
            raise ValueError("physical raster fitting requires physical_cell_overlap_v1")
        if self.neighbor_indices.numel():
            raise ValueError("topology raster mapping is already frozen")
        mapping = build_physical_cell_overlap_map(
            native_coordinates,
            query_indices,
            grid_shape=self.grid_shape,
            query_coordinates=query_coordinates,
            axes=self.axes,
            neighbors=self.raster_neighbors,
            power=self.raster_power,
        )
        device = self.normalization_offset.device
        self.neighbor_indices = mapping.neighbor_indices.to(device)
        self.neighbor_weights = mapping.neighbor_weights.to(device)
        self.grid_coordinates = mapping.grid_coordinates.to(device)
        self.geometry_sha256 = mapping.coordinate_sha256
        self.geometry_diagnostics = dict(mapping.diagnostics)

    def _raster_map(self, coordinates: torch.Tensor, *, native_grid_indices=None) -> None:
        if coordinates.ndim != 3:
            raise ValueError("topology coordinates must have shape [B,N,D]")
        if not torch.equal(coordinates, coordinates[:1].expand_as(coordinates)):
            raise ValueError("topology requires identical coordinates for every sample")
        digest = coordinate_digest(coordinates[0])
        if not self.neighbor_indices.numel():
            if self.raster_mapping == "physical_cell_overlap_v1":
                raise ValueError(
                    "fit physical topology raster from native coordinates before scoring"
                )
            if self.structured_indices is not None and native_grid_indices is None:
                raise ValueError("fit native topology geometry or load its artifact before scoring")
            mapping = build_raster_map(
                coordinates[0],
                grid_shape=self.grid_shape,
                axes=self.axes,
                neighbors=self.raster_neighbors,
                power=self.raster_power,
                periodic=self.periodic,
                periods=self.periods,
                allow_projected_collisions=self.allow_projected_collisions,
                antialias_downsample=self.antialias_downsample,
                native_masked=self.native_masked,
                native_grid_indices=native_grid_indices,
            )
            self.neighbor_indices = mapping.neighbor_indices.to(coordinates.device)
            self.neighbor_weights = mapping.neighbor_weights.to(coordinates.device)
            self.grid_coordinates = mapping.grid_coordinates.to(coordinates.device)
            self.geometry_sha256 = mapping.coordinate_sha256
            self.periods = mapping.periods
            self.geometry_diagnostics = dict(mapping.diagnostics)
            if self.native_masked:
                self.domain_mask = mapping.valid_mask.to(coordinates.device)
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
        if self.spatial_objective is not None:
            # Self fields, carrier and derived anchor can use different dataset channels.
            return generated, reference
        return generated[..., self.field_ids], reference[..., self.field_ids]

    def _curve_cost(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, int]]:
        """Return per-sample curve MSE for `[B,H,W]` fields."""
        costs = []
        requested_level_count = 0
        unique_level_count = 0
        quantiles = torch.as_tensor(self.quantiles, device=generated.device, dtype=generated.dtype)
        for batch_index in range(generated.shape[0]):
            sample_costs = []
            for direction in self.directions:
                sign = 1.0 if direction == "superlevel" else -1.0
                generated_field = sign * generated[batch_index]
                reference_field = sign * reference[batch_index]
                levels = torch.quantile(reference_field.detach().reshape(-1), quantiles)
                requested_level_count += int(levels.numel())
                # v1 rule: collapse identical thresholds and weight every unique level equally.
                levels = torch.unique(levels, sorted=True)
                unique_level_count += int(levels.numel())
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
        return torch.stack(costs), {
            "requested_threshold_count": requested_level_count,
            "unique_threshold_count": unique_level_count,
        }

    def _self_cost(
        self, generated: torch.Tensor, reference: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, int]]:
        outputs = [
            self._curve_cost(generated[:, field], reference[:, field])
            for field in range(generated.shape[1])
        ]
        return torch.stack([output[0] for output in outputs]).mean(dim=0), {
            key: sum(output[1][key] for output in outputs) for key in outputs[0][1]
        }

    def _mutual_cost(
        self, generated: torch.Tensor, reference: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, int]]:
        theta_min = self.theta_min_degrees * pi / 180.0
        angle_step = (pi / 2.0 - 2.0 * theta_min) / (self.fibered_lines + 1.0)
        angles = tuple(theta_min + angle_step * (index + 1) for index in range(self.fibered_lines))
        all_costs = []
        threshold_counts = {"requested_threshold_count": 0, "unique_threshold_count": 0}
        for left, right in self.mutual_pairs:
            reference_left = reference[:, left]
            reference_right = reference[:, right]
            mean_left = reference_left.detach().mean(dim=(1, 2), keepdim=True)
            mean_right = reference_right.detach().mean(dim=(1, 2), keepdim=True)
            std_left = reference_left.detach().std(dim=(1, 2), keepdim=True)
            std_right = reference_right.detach().std(dim=(1, 2), keepdim=True)
            if (std_left <= self.mutual_axis_tolerance).any() or (
                std_right <= self.mutual_axis_tolerance
            ).any():
                raise ValueError(
                    "topology mutual axis is constant or near-constant in a reference sample"
                )
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
                line_cost, counts = self._curve_cost(generated_push, reference_push)
                line_costs.append(line_cost)
                for key, value in counts.items():
                    threshold_counts[key] += value
            all_costs.append(torch.stack(line_costs).mean(dim=0))
        return torch.stack(all_costs).mean(dim=0), threshold_counts

    def forward(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor,
        *,
        coordinates: torch.Tensor | None = None,
        context: Any | None = None,
    ) -> FamilyResult:
        reference_cache = (context or {}).get("persistence_cache")
        if reference_cache is not None:
            if self.strategy != "cubical_persistence":
                raise ValueError("reference caching requires cubical_persistence")
            signature = (
                id(self),
                id(reference),
                reference._version,
                id(coordinates),
                coordinates._version if coordinates is not None else None,
                self.normalization_offset._version,
                self.normalization_scale._version,
            )
            reference_cache.setdefault("reference_identity", (reference, coordinates))
            if reference_cache.setdefault("signature", signature) != signature:
                raise ValueError("persistence reference cache target/coordinates changed")
        require_field_tensor("generated", generated)
        require_field_tensor("reference", reference)
        if generated.shape != reference.shape:
            raise ValueError("topology generated/reference shapes differ")
        if coordinates is None or coordinates.shape[:2] != generated.shape[:2]:
            raise ValueError("topology requires coordinates aligned with [B,N]")
        if self.spatial_objective is not None:
            if not torch.isfinite(generated).all() or not torch.isfinite(reference).all():
                raise FloatingPointError("topology generated/reference fields must be finite")
            reference = reference.detach()
            # Quantiles and topology arithmetic remain FP32 under mixed-precision rollouts.
            if generated.dtype in {torch.float16, torch.bfloat16}:
                generated = generated.float()
            reference = reference.to(generated)
        self._raster_map(coordinates)
        generated, reference = self._units_and_fields(generated, reference)
        precision = (
            torch.autocast(device_type=generated.device.type, enabled=False)
            if self.spatial_objective is not None
            else nullcontext()
        )
        with precision:
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
            generated_grid = gaussian_blur(
                generated_grid, self.smoothing_sigma, self.periodic, radius=self.smoothing_radius
            )
            reference_grid = gaussian_blur(
                reference_grid, self.smoothing_sigma, self.periodic, radius=self.smoothing_radius
            )
        if self.spatial_objective is not None:
            with torch.autocast(device_type=generated.device.type, enabled=False):
                if self.version == "4" and context and "_finite_primary_calibration" in context:
                    self.spatial_objective.fit_calibration(
                        generated_grid.float(),
                        reference_grid.float(),
                        valid_mask=self.domain_mask if self.native_masked else None,
                        provenance=context["_finite_primary_calibration"],
                    )
                kwargs = (
                    {
                        "reference_cache": reference_cache,
                        "valid_mask": self.domain_mask if self.native_masked else None,
                    }
                    if self.strategy == "cubical_persistence"
                    else {}
                )
                if context and context.get("query_selection") is not None:
                    if self.version != "4":
                        raise ValueError("query selection requires finite_primary_v1")
                    kwargs["query_selection"] = context["query_selection"]
                result = self.spatial_objective(
                    generated_grid.float(), reference_grid.float(), **kwargs
                )
            result.diagnostics.update(
                {
                    "family": self.family_name,
                    "version": self.version,
                    "aggregation": "per_sample",
                    "target_use": self.target_use,
                    "units": self.units,
                    "fields": self.field_names,
                    "grid_shape": self.grid_shape,
                    "periodic": self.periodic,
                    "periods": self.periods,
                    "geometry_sha256": self.geometry_sha256,
                    "geometry": dict(self.geometry_diagnostics),
                    "component_weights": dict(self.component_weights),
                }
            )
            return result
        results: dict[str, TermResult] = {}
        per_sample = generated.sum(dim=(1, 2)) * 0.0
        if self.component_weights.get("self", 0.0) > 0:
            cost, threshold_counts = self._self_cost(generated_grid, reference_grid)
            collapsed = 1.0 - (
                threshold_counts["unique_threshold_count"]
                / threshold_counts["requested_threshold_count"]
            )
            results[f"{self.family_name}.self.betti_curves"] = TermResult(
                cost,
                cost.mean(),
                diagnostics={
                    "dimensions": self.dimensions,
                    "directions": self.directions,
                    **threshold_counts,
                    "collapsed_threshold_fraction": collapsed,
                    "threshold_rule": "deduplicate_equal_then_equal_weight_unique",
                },
            )
            per_sample = per_sample + self.component_weights["self"] * cost
        if self.component_weights.get("mutual", 0.0) > 0:
            cost, threshold_counts = self._mutual_cost(generated_grid, reference_grid)
            collapsed = 1.0 - (
                threshold_counts["unique_threshold_count"]
                / threshold_counts["requested_threshold_count"]
            )
            results[f"{self.family_name}.mutual.fibered_betti_curves"] = TermResult(
                cost,
                cost.mean(),
                diagnostics={
                    "pairs": self.mutual_pairs,
                    "lines": self.fibered_lines,
                    **threshold_counts,
                    "collapsed_threshold_fraction": collapsed,
                    "threshold_rule": "deduplicate_equal_then_equal_weight_unique",
                },
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
                "periods": self.periods,
                "geometry_sha256": self.geometry_sha256,
                "geometry": dict(self.geometry_diagnostics),
                "component_weights": dict(self.component_weights),
            },
        )

    def state_artifact(self) -> dict[str, Any]:
        return {
            "family": self.family_name,
            "version": self.version,
            "scientific_source": persistence_source(finite_primary=self.version == "4")
            if self.version in {"3", "4"}
            else dict(SOURCE_SNAPSHOT)
            if self.version == "2"
            else {
                "repository": "https://github.com/jachen25/PhyCoFlow_dev/tree/main/src",
                "revision": "ab49ea37a",
                "license_status": "Project-supervised contribution; see docs/provenance.md",
            },
            "config": self.config,
            "field_names": self.field_names,
            "target_use": self.target_use,
            "units": self.units,
            "geometry_sha256": self.geometry_sha256,
            "periods": self.periods,
            "geometry_diagnostics": dict(self.geometry_diagnostics),
            **(
                {"calibration": deepcopy(self.spatial_objective.calibration_record)}
                if self.version == "4"
                else {}
            ),
            "state_dict": self.state_dict(),
        }

    def load_state_artifact(self, artifact: Mapping[str, Any]) -> None:
        if artifact.get("family") != self.family_name or artifact.get("version") != self.version:
            raise ValueError("topology family artifact identity mismatch")
        if dict(artifact.get("config", {})) != self.config:
            raise ValueError("topology family artifact config mismatch")
        if self.version == "2" and artifact.get("scientific_source") != SOURCE_SNAPSHOT:
            raise ValueError(
                "topology family artifact scientific source mismatch; rebuild the artifact"
            )
        if self.version in {"3", "4"} and not _compatible_persistence_source(
            artifact.get("scientific_source"),
            persistence_source(finite_primary=self.version == "4"),
            legacy_geometry=self.raster_mapping == "legacy_v1",
        ):
            raise ValueError("persistence family source/backend mismatch; start a new experiment")
        state = artifact["state_dict"]
        device = self.normalization_offset.device
        self.neighbor_indices = state["neighbor_indices"].to(device)
        self.neighbor_weights = state["neighbor_weights"].to(device)
        self.grid_coordinates = state["grid_coordinates"].to(device)
        if self.native_masked:
            self.domain_mask = state["domain_mask"].to(device)
        self.geometry_sha256 = artifact.get("geometry_sha256")
        self.periods = artifact.get("periods", self.periods)
        self.geometry_diagnostics = dict(artifact.get("geometry_diagnostics", {}))
        self.load_state_dict(state, strict=True)
        if self.version == "4":
            calibration = artifact.get("calibration")
            if bool(self.spatial_objective.calibrated) != (calibration is not None):
                raise ValueError("finite_primary_v1 calibration artifact mismatch")
            if calibration is not None:
                objective = self.spatial_objective
                try:
                    source = calibration["provenance"]["source_checkpoint_sha256"]
                    ids = calibration["provenance"]["sample_ids"]
                    scales = torch.as_tensor(
                        calibration["scales"], dtype=objective.leaf_scales.dtype
                    )
                    presence = torch.as_tensor(calibration["essential_present"], dtype=torch.bool)
                    counts = torch.as_tensor(calibration["essential_counts"], dtype=torch.long)
                    raw = torch.as_tensor(
                        calibration["raw_mean"], dtype=objective.leaf_scales.dtype
                    )
                    floors = raw.new_tensor([objective.finite_floor, objective.essential_floor])
                    valid = (
                        calibration["definition"] == objective.definition
                        and calibration["provenance"]["split"] == "train"
                        and isinstance(ids, (list, tuple))
                        and bool(ids)
                        and isinstance(source, str)
                        and len(source) == 64
                        and all(c in "0123456789abcdef" for c in source)
                        and calibration["units"] == [unit[1] for unit in objective.units]
                        and calibration["dimensions"] == list(objective.dimensions)
                        and calibration["point_normalization"]
                        == (
                            int(self.domain_mask.sum())
                            if self.native_masked
                            else math.prod(self.grid_shape)
                        )
                        and raw.shape == scales.shape == objective.leaf_scales.shape
                        and torch.isfinite(raw).all()
                        and bool((raw >= 0).all())
                        and torch.equal(scales, torch.maximum(raw, floors))
                        and torch.equal(scales, objective.leaf_scales.cpu())
                        and torch.equal(presence, objective.essential_present.cpu())
                        and torch.equal(counts, objective.essential_counts.cpu())
                        and torch.equal(presence, counts > 0)
                        and bool((counts >= 0).all())
                    )
                except (KeyError, TypeError, ValueError, RuntimeError):
                    valid = False
                if not valid:
                    raise ValueError("finite_primary_v1 calibration artifact contents mismatch")
            self.spatial_objective.calibration_record = deepcopy(calibration)
