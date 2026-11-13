"""Differentiable topology coherence on a fixed two-dimensional raster."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from contextlib import nullcontext
from copy import deepcopy
from itertools import combinations
from math import ceil, cos, pi, sin
from pathlib import Path
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
from .conditional_fair import ConditionalFairTopologyObjective
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

# The complete original-main source snapshot for v3 is the sole historical
# exception: its unmasked, legacy-raster scoring path is numerically unchanged.
_MAIN_V3_SOURCE = {
    "family.py": "def608594481b1a411af9180e27143c67ce8856076ca5062e953760be8ff90c9",
    "geometry.py": "3f453236ff68e4bd3e466117fb6f79f28b4c8eecb8e191c6218050d2c6ed0d3d",
    "persistence.py": "a79375a229aba57d52ece0e17719ebac8f42880204b2e53976cc0c07e7d72643",
    "persistence_objective.py": "9632489eeaf7b9543b80c8878adf4cdeff14f404289562ccb690d5397ce198a0",
    "betti_curves.py": "6b356f1e1fc15876f84f8c56834dbf943a592ca9b0e8dd70d6ba9192a9a0fdea",
    "spatial.py": "0c2c9615024204f4bcd05a3ae13715027dd17c1b53fbd528c92254a3f354f2cc",
    "spatial_persistence.py": "eda6d46150214ebdc506ac9908c1735322c19dbb82bba710d1aaba94e43c00e3",
}


def _fair_source():
    result = persistence_source(finite_primary=True)
    result["implementation"] = "repository-local conditional fair cubical persistence v5"
    result["source_sha256"]["conditional_fair.py"] = hashlib.sha256(
        Path(__file__).with_name("conditional_fair.py").read_bytes()
    ).hexdigest()
    return result


def _compatible_persistence_source(actual, expected, *, original_main_v3):
    if actual == expected:
        return True
    if not original_main_v3 or not isinstance(actual, Mapping):
        return False
    previous = deepcopy(expected)
    previous["source_sha256"] = _MAIN_V3_SOURCE
    return actual == previous


def _pooled_source():
    root = Path(__file__).resolve()
    files = {
        "family.py": root,
        "geometry.py": root.with_name("geometry.py"),
        "persistence.py": root.with_name("persistence.py"),
        "shape_compute.py": root.parents[2] / "shape_compute.py",
        "query_sampling.py": root.parents[2] / "query_sampling.py",
    }
    return {
        "definition": "scalable_pooled_v1",
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()
        },
    }


class _PooledGeometryObjective(nn.Module):
    """Linear-size geometry contract; pooled scoring owns all query materialization."""

    def __init__(self, config, fields):
        super().__init__()
        if config.get("target_use", "paired_supervised") != "paired_supervised":
            raise ValueError("scalable pooled Shape requires paired_supervised")
        settings = config.get("persistence", {})
        if (
            set(settings) - {"definition", "projections", "distance"}
            or settings.get("distance", "sliced_wasserstein") != "sliced_wasserstein"
        ):
            raise ValueError(
                "scalable pooled Shape accepts only sliced persistence and projection count"
            )
        count = settings.get("projections", 32)
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise ValueError("scalable pooled Shape projections must be an integer >= 2")
        self.projections = count
        filtration = config.get("filtration", {})
        if set(filtration.get("dimensions", (0, 1))) != {0, 1} or set(
            filtration.get("directions", ("sublevel", "superlevel"))
        ) != {"sublevel", "superlevel"}:
            raise ValueError("scalable pooled Shape requires H0/H1 and both signs")
        components = config.get("components", {})
        if set(components) - {"self", "mutual"}:
            raise ValueError("unknown scalable pooled Shape component")
        self.weights = {}
        for name in ("self", "mutual"):
            settings = components.get(name, {})
            if set(settings) - {"enabled", "weight"}:
                raise ValueError(
                    "scalable pooled Shape uses implicit all-field and all-pair populations"
                )
            if settings.get("enabled", True) and (name == "self" or len(fields) > 1):
                value = float(settings.get("weight", 1.0))
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("scalable pooled Shape weights must be positive finite")
                self.weights[name + ".persistence"] = value
        if "self.persistence" not in self.weights or (
            len(fields) > 1 and "mutual.persistence" not in self.weights
        ):
            raise ValueError(
                "scalable pooled Shape requires all available self and mutual populations"
            )
        self.spec = CoherenceFamilySpec(
            "topology",
            "6",
            tuple(
                CoherenceComponentSpec(
                    name,
                    "paired_supervised",
                    config.get("units", "model_units"),
                    True,
                    required_geometry="fixed_2d_raster",
                    aggregation="per_sample",
                )
                for name in self.weights
            ),
            metadata={"definition": "scalable_pooled_v1"},
        )
        self.spec.validate()


class TopologyFamily(nn.Module):
    family_name = "topology"
    version = "1"

    @property
    def needs_source_calibration(self) -> bool:
        """Whether the new leaf scales still need frozen TRAIN/SOURCE fitting."""
        return self.version in {"4", "5"} and not bool(self.spatial_objective.calibrated)

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
        if (
            self.strategy == "cubical_persistence"
            and config.get("persistence", {}).get("definition")
            == "conditional_fixed_transform_fair_v1"
        ):
            self.version = "5"
        if (
            self.strategy == "cubical_persistence"
            and config.get("persistence", {}).get("definition") == "scalable_pooled_v1"
        ):
            self.version = "6"
        self.conditional_draws = 2 if self.version in {"5", "6"} else 1
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
                _PooledGeometryObjective
                if self.version == "6"
                else ConditionalFairTopologyObjective
                if self.version == "5"
                else FinitePrimaryTopologyObjective
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
    def fair_prepare_grid(self, fields, *, coordinates):
        """Apply the family units, raster and smoothing to a streaming calibration batch."""
        if self.version != "5":
            raise ValueError("fair preparation requires version 5")
        self._raster_map(coordinates)
        values, _ = self._units_and_fields(fields, fields)
        grid = rasterize_fields(
            values, self.neighbor_indices, self.neighbor_weights.to(values), self.grid_shape
        )
        return gaussian_blur(
            grid, self.smoothing_sigma, self.periodic, radius=self.smoothing_radius
        ).float()

    @torch.no_grad()
    def fair_accumulate_transform(self, reference, *, coordinates):
        grid = self.fair_prepare_grid(reference, coordinates=coordinates)
        self.spatial_objective.accumulate_transform(
            grid, valid_mask=self.domain_mask if self.native_masked else None
        )

    @torch.no_grad()
    def fair_freeze_transform(self, *, floors, provenance):
        return self.spatial_objective.freeze_transform(floors=floors, provenance=provenance)

    @torch.no_grad()
    def fair_accumulate_calibration(self, prediction, reference, *, coordinates):
        source = self.fair_prepare_grid(prediction, coordinates=coordinates)
        target = self.fair_prepare_grid(reference, coordinates=coordinates)
        self.spatial_objective.accumulate_calibration(
            source, target, valid_mask=self.domain_mask if self.native_masked else None
        )

    @torch.no_grad()
    def fair_freeze_calibration(self, *, provenance):
        return self.spatial_objective.freeze_calibration(provenance=provenance)

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
        if self.version == "6":
            raise ValueError("scalable pooled Shape requires ScalableQueryAdapter")
        original_reference = reference
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
        reference_cache = (context or {}).get("persistence_cache")
        if reference_cache is not None:
            if self.strategy != "cubical_persistence":
                raise ValueError("reference caching requires cubical_persistence")
            objective = self.spatial_objective
            dependencies = [self.neighbor_indices, self.neighbor_weights]
            if hasattr(self, "domain_mask"):
                dependencies.append(self.domain_mask)
            dependencies.extend(
                getattr(objective, name)
                for name in ("descriptor_mean", "descriptor_scale", "transform_frozen")
                if hasattr(objective, name)
            )
            signature = (
                id(self),
                tuple((id(value), value._version) for value in dependencies),
                id(original_reference),
                original_reference._version,
                id(coordinates),
                coordinates._version if coordinates is not None else None,
                self.normalization_offset._version,
                self.normalization_scale._version,
            )
            reference_cache.setdefault("reference_identity", (original_reference, coordinates))
            if reference_cache.setdefault("signature", signature) != signature:
                raise ValueError("persistence reference cache target/coordinates changed")
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
                    if self.version not in {"4", "5"}:
                        raise ValueError("query selection requires finite-primary Shape")
                    kwargs["query_selection"] = context["query_selection"]
                if self.version == "5":
                    second = (context or {}).get("second_generated")
                    if second is None and (context or {}).get("conditional_draws"):
                        second = context["conditional_draws"][0]
                    if second is None or second.shape != generated.shape:
                        raise ValueError(
                            "fair Shape requires second_generated aligned with generated"
                        )
                    second, _ = self._units_and_fields(second, reference)
                    second_grid = gaussian_blur(
                        rasterize_fields(
                            second,
                            self.neighbor_indices,
                            self.neighbor_weights.to(second),
                            self.grid_shape,
                        ),
                        self.smoothing_sigma,
                        self.periodic,
                        radius=self.smoothing_radius,
                    )
                    kwargs["second_generated"] = second_grid.float()
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
            "scientific_source": _pooled_source()
            if self.version == "6"
            else _fair_source()
            if self.version == "5"
            else persistence_source(finite_primary=self.version == "4")
            if self.version in {"3", "4", "5"}
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
                if self.version in {"4", "5"}
                else {}
            ),
            **(
                {"transform": deepcopy(self.spatial_objective.transform_record)}
                if self.version == "5"
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
        if self.version == "6" and artifact.get("scientific_source") != _pooled_source():
            raise ValueError("pooled Shape source mismatch; rebuild the artifact")
        if self.version in {"3", "4", "5"} and not _compatible_persistence_source(
            artifact.get("scientific_source"),
            _fair_source()
            if self.version == "5"
            else persistence_source(finite_primary=self.version == "4"),
            original_main_v3=self.version == "3" and self.raster_mapping == "legacy_v1",
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
        if self.version == "5":
            transform = artifact.get("transform")
            obj = self.spatial_objective
            if bool(obj.transform_frozen) != (transform is not None):
                raise ValueError("fair Shape transform artifact mismatch")
            if transform is not None:
                try:
                    ids = transform["provenance"]["sample_ids"]
                    floor = torch.as_tensor(transform["floors"], dtype=obj.descriptor_scale.dtype)
                    expected_mask = (
                        hashlib.sha256(self.domain_mask.cpu().numpy().tobytes()).hexdigest()
                        if self.native_masked
                        else None
                    )
                    valid = (
                        transform["definition"] == obj.definition
                        and transform["provenance"]["split"] == "train"
                        and bool(ids)
                        and len(set(ids)) == len(ids) == transform["visits"]
                        and transform["raster_measure"] == "uniform_equal_valid_pixel_area"
                        and transform["mask_sha256"] == expected_mask
                        and transform["support_count"]
                        == len(ids)
                        * (
                            int(self.domain_mask.sum())
                            if self.native_masked
                            else math.prod(self.grid_shape)
                        )
                        and torch.equal(
                            torch.as_tensor(transform["mean"], dtype=obj.descriptor_mean.dtype),
                            obj.descriptor_mean.cpu(),
                        )
                        and torch.equal(
                            torch.as_tensor(transform["scale"], dtype=obj.descriptor_scale.dtype),
                            obj.descriptor_scale.cpu(),
                        )
                        and floor.shape == obj.descriptor_scale.shape
                        and bool(torch.isfinite(floor).all())
                        and bool((floor > 0).all())
                        and bool((obj.descriptor_scale.cpu() >= floor).all())
                        and bool(torch.isfinite(obj.descriptor_mean).all())
                        and bool(torch.isfinite(obj.descriptor_scale).all())
                    )
                except (KeyError, TypeError, ValueError, RuntimeError):
                    valid = False
                if not valid:
                    raise ValueError("fair Shape transform artifact contents mismatch")
            obj.transform_record = deepcopy(transform)
        if self.version in {"4", "5"}:
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
                        (
                            self.version != "5"
                            or list(ids)
                            == list(
                                self.spatial_objective.transform_record["provenance"]["sample_ids"]
                            )
                        )
                        and calibration["definition"] == objective.definition
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
