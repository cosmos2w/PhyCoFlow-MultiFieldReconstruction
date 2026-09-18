"""Spatially-aware cubical persistence coherence on a fixed two-dimensional raster.

This is the persistence-diagram counterpart of the Betti-curve `topology_betti` family:
the same fixed point-set-to-raster map, but the per-field ("self") and per-pair
("mutual", fibered) terms compare full lower-star persistence diagrams through an
optimal partial matching whose costs are reweighted by creator separation. It is a
detached implementation of the historical `persistence_satloss_cubical` and
`fibered_satloss_cubical` modes, and is not wired into post-training.

Both terms run in three phases over every unit of a call at once. Phase 1 builds
the compared fields as one live block per side. Phase 2 reduces both blocks on the
device (GUDHI on the host when the device backend is not selected or declines),
builds every unit's cost matrices on the device, and solves the assignments exactly
on the shared thread pool. Phase 3 rebuilds the differentiable distances in one set
of kernels.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Mapping
from itertools import combinations
from math import ceil
from typing import Any

import numpy as np
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
from .execution import parallel_map, tensor_pairing_selected, worker_count
from .geometry import build_raster_map, coordinate_digest, gaussian_blur, rasterize_fields
from .persistence import (
    LINE_SAMPLINGS,
    SPATIAL_MODES,
    Generators,
    MergeTreeUnavailable,
    SolvedBatch,
    apply_units_batched,
    cubical_generators,
    cubical_generators_on_device,
    domain_scale,
    gudhi_available,
    gudhi_module,
    pad_generators,
    plan_units,
    reference_minmax,
    slice_lines,
)

DIRECTION_SIGNS = {"sublevel": 1.0, "superlevel": -1.0}

#: Reference generators are a few small index arrays per field, so the bound holds
#: a whole training split's worth; a larger split just stops hitting, and the
#: `reference_cache` diagnostics make that visible.
_REFERENCE_CACHE_CAPACITY = 16384


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
        gudhi_module()  # fail at construction, with the reason, rather than mid-run
        self.config = dict(config)
        self.target_use = str(config.get("target_use", "paired_supervised"))
        self.units = str(config.get("units", "model_units"))
        self.family_weight = float(config.get("weight", 1.0))
        if self.family_weight <= 0:
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
        if bool(geometry.get("periodic", False)):
            # The cubical complex here is the plain one and creator separations are
            # measured in the box, so a periodic raster would be silently wrong.
            raise ValueError("topology supports only nonperiodic geometry")
        self.allow_projected_collisions = bool(
            geometry.get("allow_projected_collisions", False)
        )
        self.register_buffer("neighbor_indices", torch.empty(0, 0, dtype=torch.long))
        self.register_buffer("neighbor_weights", torch.empty(0, 0))
        self.register_buffer("grid_coordinates", torch.empty(0, 0, 2))
        self.geometry_sha256: str | None = None
        self.geometry_diagnostics: dict[str, float | int] = {}
        # Reference generators are a pure function of the normalized reference row,
        # so the cache is keyed on those bytes rather than on a sample id: correct
        # under `paired_supervised` and a rotating `training_reference` bank alike.
        self._reference_cache: OrderedDict[tuple[bytes, str], list[Generators]] = OrderedDict()
        self._reference_cache_hits = 0
        self._reference_cache_misses = 0
        # Host grid coordinates and domain scale per dtype; fixed once the raster map is.
        self._coordinate_cache: dict[tuple[str, str], tuple[torch.Tensor, float]] = {}

        filtration = config.get("filtration", {})
        self.dimensions = tuple(int(value) for value in filtration.get("dimensions", (0, 1)))
        if (
            not self.dimensions
            or not set(self.dimensions) <= {0, 1}
            or len(set(self.dimensions)) != len(self.dimensions)
        ):
            raise ValueError("topology filtration dimensions must be drawn from {0,1}")
        self.directions = tuple(filtration.get("directions", ("sublevel", "superlevel")))
        if (
            not self.directions
            or not set(self.directions) <= set(DIRECTION_SIGNS)
            or len(set(self.directions)) != len(self.directions)
        ):
            raise ValueError("topology filtration directions are invalid")
        self.smoothing_sigma = float(filtration.get("smoothing_sigma", 0.0))
        if self.smoothing_sigma < 0:
            raise ValueError("topology smoothing must be non-negative")
        smoothing_radius = ceil(3.0 * self.smoothing_sigma)
        if self.smoothing_sigma > 0 and (
            smoothing_radius >= self.grid_shape[0] or smoothing_radius >= self.grid_shape[1]
        ):
            raise ValueError(
                "topology smoothing radius must be smaller than both grid dimensions"
            )

        matching = config.get("matching", {})
        removed = sorted({"solver", "solver_tolerance"} & set(matching))
        if removed:
            raise ValueError(
                f"topology matching keys {removed} were removed: the assignment "
                "is always solved exactly from device-built costs"
            )
        self.order = float(matching.get("order", 1.0))
        self.lambda_spatial = float(matching.get("lambda_spatial", 1.0))
        self.spatial_mode = str(matching.get("spatial_mode", "multiplicative"))
        # Bars shorter than this fraction of the reference's value range are dropped at
        # the reduction. A smoothed field's diagram is mostly such bars; they only ever
        # match the diagonal, so they add a noise floor to the value and a cubic cost to
        # the assignment. Zero keeps every bar of positive persistence.
        self.min_persistence = float(matching.get("min_persistence", 0.01))
        if (
            self.order <= 0
            or self.lambda_spatial < 0
            or self.spatial_mode not in SPATIAL_MODES
            or not 0.0 <= self.min_persistence < 1.0
        ):
            raise ValueError(
                "topology matching requires order>0, lambda_spatial>=0, a "
                f"spatial_mode in {SPATIAL_MODES} and min_persistence in [0, 1)"
            )
        # Resolved per call once the fields' device is known: the host reduction is
        # the right answer until then, and what a declined batch falls back to.
        self._reduction_backend = "gudhi"

        components = config.get("components", {})
        self_settings = components.get("self", {})
        mutual_settings = components.get("mutual", {})
        self.component_weights: dict[str, float] = {}
        component_metadata = {
            "homology_dimensions": self.dimensions,
            "complex": "cubical",
            "spatial_mode": self.spatial_mode,
            "lambda_spatial": self.lambda_spatial,
            "min_persistence": self.min_persistence,
        }
        specs = []
        if bool(self_settings.get("enabled", True)):
            self.component_weights["self"] = float(self_settings.get("weight", 1.0))
            specs.append(
                CoherenceComponentSpec(
                    "self.persistence_matching",
                    self.target_use,
                    self.units,
                    True,
                    required_geometry="fixed_2d_raster",
                    aggregation="per_sample",
                    metadata=dict(component_metadata),
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
                raise KeyError(
                    "topology mutual pair contains a field outside fields"
                ) from error
            self.fibered_lines = int(mutual_settings.get("lines", 8))
            self.angle_margin = float(mutual_settings.get("angle_margin", 0.12))
            self.line_seed = int(mutual_settings.get("seed", 0))
            self.line_sampling = str(mutual_settings.get("line_sampling", "stratified"))
            if (
                not self.mutual_pairs
                or any(left == right for left, right in self.mutual_pairs)
                or len({tuple(sorted(pair)) for pair in self.mutual_pairs})
                != len(self.mutual_pairs)
                or self.fibered_lines < 1
                or not 0.0 < self.angle_margin < 0.5
                or self.line_sampling not in LINE_SAMPLINGS
            ):
                raise ValueError(
                    "topology mutual component requires distinct pairs, lines>=1, "
                    f"angle_margin in (0,0.5) and line_sampling in {LINE_SAMPLINGS}"
                )
            specs.append(
                CoherenceComponentSpec(
                    "mutual.fibered_matching",
                    self.target_use,
                    self.units,
                    True,
                    required_geometry="fixed_2d_raster",
                    aggregation="per_sample",
                    metadata=dict(component_metadata),
                )
            )
        else:
            self.mutual_pairs = ()
            self.fibered_lines = 0
            self.angle_margin = 0.12
            self.line_seed = 0
            self.line_sampling = "stratified"
        # Slice lines are re-drawn on every call from this explicit, seeded stream;
        # its state travels with the family artifact so a reload continues it.
        self._line_rng = np.random.default_rng(self.line_seed)
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

    # ------------------------------------------------------------------ geometry
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
                periodic=False,
                allow_projected_collisions=self.allow_projected_collisions,
            )
            self.neighbor_indices = mapping.neighbor_indices.to(coordinates.device)
            self.neighbor_weights = mapping.neighbor_weights.to(coordinates.device)
            self.grid_coordinates = mapping.grid_coordinates.to(coordinates.device)
            self.geometry_sha256 = mapping.coordinate_sha256
            self.geometry_diagnostics = dict(mapping.diagnostics)
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

    # -------------------------------------------------------------- unit engine
    @staticmethod
    def _new_counts() -> dict[str, int]:
        return {
            "generated_finite_bars": 0,
            "generated_essential_bars": 0,
            "reference_finite_bars": 0,
            "reference_essential_bars": 0,
            "matchings": 0,
            "merge_rounds": 0,
        }

    def _cached_generators(self, key: tuple[bytes, str]) -> list[Generators] | None:
        entry = self._reference_cache.get(key)
        if entry is None:
            self._reference_cache_misses += 1
            return None
        self._reference_cache.move_to_end(key)
        self._reference_cache_hits += 1
        return entry

    def _store_generators(self, key: tuple[bytes, str], generators: list[Generators]) -> None:
        self._reference_cache[key] = generators
        while len(self._reference_cache) > _REFERENCE_CACHE_CAPACITY:
            self._reference_cache.popitem(last=False)

    def _geometry(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, float]:
        """Grid coordinates on `device` in `dtype` and the domain scale, cached per dtype."""
        key = (str(device), str(dtype))
        cached = self._coordinate_cache.get(key)
        if cached is None:
            coordinates = (
                self.grid_coordinates.reshape(-1, 2).detach().to(device=device, dtype=dtype)
            )
            cached = (coordinates, domain_scale(coordinates.cpu().numpy()))
            self._coordinate_cache[key] = cached
        return cached

    def _reduce(
        self,
        gen_block: torch.Tensor,
        ref_block: torch.Tensor,
        cacheable: bool,
        counts: dict[str, int],
    ) -> tuple[list[list[Generators]], list[list[Generators]]]:
        """Generators of both blocks, per unit and degree; the reference side may be cached.

        The device backend reduces the generated rows and the uncached reference rows
        as one stacked block; GUDHI on the pool takes over when that backend is not
        selected or declines the batch. Cache keys are digests of the normalized
        reference rows, which is the one thing that needs a host copy of a block.
        """
        height, width = self.grid_shape
        count = gen_block.shape[0]
        keys: list[tuple[bytes, str] | None] = [None] * count
        cached: list[list[Generators] | None] = [None] * count
        host_ref: np.ndarray | None = None
        if cacheable:
            host_ref = ref_block.cpu().numpy()
            for row in range(count):
                keys[row] = (
                    hashlib.blake2b(host_ref[row].tobytes(), digest_size=16).digest(),
                    str(host_ref.dtype),
                )
                cached[row] = self._cached_generators(keys[row])
        missing = [row for row in range(count) if cached[row] is None]
        reductions: list[list[Generators]] | None = None
        self._reduction_backend = "gudhi"
        if tensor_pairing_selected(gen_block.device):
            blocks = [gen_block]
            if missing:
                index = torch.as_tensor(missing, device=ref_block.device)
                blocks.append(ref_block.index_select(0, index))
            try:
                reductions, rounds = cubical_generators_on_device(
                    torch.cat(blocks), height, width, self.dimensions, self.min_persistence
                )
            except MergeTreeUnavailable:
                reductions = None
            else:
                self._reduction_backend = "tensor"
                counts["merge_rounds"] = max(counts["merge_rounds"], rounds)
        if reductions is None:
            host_gen = gen_block.cpu().numpy()
            if host_ref is None:
                host_ref = ref_block.cpu().numpy()
            rows = [host_gen[row] for row in range(count)] + [host_ref[row] for row in missing]
            floor = self.min_persistence
            reductions = parallel_map(
                lambda values: cubical_generators(values, height, width, self.dimensions, floor),
                rows,
            )
        fresh = iter(reductions[count:])
        generators_ref = [
            cached[row] if cached[row] is not None else next(fresh) for row in range(count)
        ]
        for row in missing:
            if keys[row] is not None:
                self._store_generators(keys[row], generators_ref[row])
        return reductions[:count], generators_ref

    def _solve_units(
        self,
        gen_block: torch.Tensor,
        ref_block: torch.Tensor,
        *,
        cacheable: bool,
        counts: dict[str, int],
    ) -> SolvedBatch:
        """Phase 2: reductions, cost matrices and exact assignments for every unit."""
        device = gen_block.device
        generators_gen, generators_ref = self._reduce(gen_block, ref_block, cacheable, counts)
        coordinates, scale = self._geometry(device, gen_block.dtype)
        cap = torch.maximum(gen_block.amax(dim=1), ref_block.amax(dim=1))
        padded = [
            (
                pad_generators([gens[degree] for gens in generators_gen], device),
                pad_generators([refs[degree] for refs in generators_ref], device),
            )
            for degree in range(len(self.dimensions))
        ]
        plans = [
            plan_units(
                gen_block,
                ref_block,
                padded_gen,
                padded_ref,
                coordinates,
                scale,
                cap,
                order=self.order,
                lambda_spatial=self.lambda_spatial,
                spatial_mode=self.spatial_mode,
            )
            for padded_gen, padded_ref in padded
        ]
        for padded_gen, padded_ref in padded:
            essential_gen = int(padded_gen.essential.sum())
            essential_ref = int(padded_ref.essential.sum())
            counts["generated_finite_bars"] += int(padded_gen.valid.sum()) - essential_gen
            counts["generated_essential_bars"] += essential_gen
            counts["reference_finite_bars"] += int(padded_ref.valid.sum()) - essential_ref
            counts["reference_essential_bars"] += essential_ref
        counts["matchings"] += len(self.dimensions) * gen_block.shape[0]
        return SolvedBatch(plans, padded, cap, coordinates, scale)

    def _normalized_blocks(
        self, generated: torch.Tensor, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Signed, reference-min-max-normalized live fields as `[B, D, C, N]` blocks."""
        count, channels = generated.shape[:2]
        signs = torch.tensor(
            [DIRECTION_SIGNS[direction] for direction in self.directions],
            device=generated.device,
            dtype=generated.dtype,
        ).reshape(1, -1, 1, 1)
        gen = generated.reshape(count, 1, channels, -1) * signs
        ref = reference.reshape(count, 1, channels, -1) * signs
        lo, span = reference_minmax(ref, torch.finfo(generated.dtype).eps)
        return (gen - lo) / span, (ref - lo) / span

    # --------------------------------------------------------------------- terms
    def _self_cost(
        self, generated: torch.Tensor, reference: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, int]]:
        """Per-sample mean over directions, fields and degrees for `[B,C,H,W]` grids."""
        count = generated.shape[0]
        counts = self._new_counts()
        live_gen, live_ref = self._normalized_blocks(generated, reference)
        live_gen = live_gen.reshape(-1, live_gen.shape[-1])
        live_ref = live_ref.reshape(-1, live_ref.shape[-1])
        solved = self._solve_units(
            live_gen.detach(), live_ref.detach(), cacheable=True, counts=counts
        )
        distances = apply_units_batched(
            live_gen,
            live_ref,
            solved,
            order=self.order,
            lambda_spatial=self.lambda_spatial,
            spatial_mode=self.spatial_mode,
        )
        return distances.reshape(count, -1).mean(dim=1), counts

    def _mutual_cost(
        self, generated: torch.Tensor, reference: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, int]]:
        """Per-sample mean over directions, pairs and degrees of the line-averaged distance.

        The slice lines of the whole call are drawn in one request from the seeded
        stream, in (sample, direction, pair) order, and the line reduction is one
        broadcast over `[units, lines, N]`.
        """
        count = generated.shape[0]
        lines = self.fibered_lines
        counts = self._new_counts()
        options = {"dtype": generated.dtype, "device": generated.device}
        live_gen, live_ref = self._normalized_blocks(generated, reference)
        points = live_gen.shape[-1]
        left = [pair[0] for pair in self.mutual_pairs]
        right = [pair[1] for pair in self.mutual_pairs]
        # Lower-left corner of each field's combined value box, where every slice
        # line starts: `[B, D, C]` in one synchronization.
        field_lo = (
            torch.minimum(live_gen.detach().amin(dim=-1), live_ref.detach().amin(dim=-1))
            .cpu()
            .numpy()
        )
        basepoints, directions = slice_lines(
            field_lo[:, :, left].reshape(-1),
            field_lo[:, :, right].reshape(-1),
            lines,
            self.angle_margin,
            self._line_rng,
            self.line_sampling,
        )
        base = torch.as_tensor(basepoints, **options)[..., None]  # [K, lines, 2, 1]
        step = torch.as_tensor(directions, **options)[..., None]

        def reduce(block: torch.Tensor) -> torch.Tensor:
            # The sublevel set of the line reduction is the intersection of the two
            # fields' sublevel sets at the line's position.
            block_left = block[:, :, left].reshape(-1, 1, points)
            block_right = block[:, :, right].reshape(-1, 1, points)
            reduced = torch.maximum(
                (block_left - base[:, :, 0]) / step[:, :, 0],
                (block_right - base[:, :, 1]) / step[:, :, 1],
            )
            return reduced.reshape(-1, points)

        line_gen = reduce(live_gen)
        line_ref = reduce(live_ref)
        # Lines are fresh every call, so nothing on the reference side repeats.
        solved = self._solve_units(
            line_gen.detach(), line_ref.detach(), cacheable=False, counts=counts
        )
        distances = apply_units_batched(
            line_gen,
            line_ref,
            solved,
            order=self.order,
            lambda_spatial=self.lambda_spatial,
            spatial_mode=self.spatial_mode,
        )
        per_line = distances.reshape(count, -1, lines, len(self.dimensions)).mean(dim=2)
        return per_line.reshape(count, -1).mean(dim=1), counts

    # ------------------------------------------------------------------- forward
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
            generated, self.neighbor_indices, self.neighbor_weights.to(generated), self.grid_shape
        )
        reference_grid = rasterize_fields(
            reference, self.neighbor_indices, self.neighbor_weights.to(reference), self.grid_shape
        )
        generated_grid = gaussian_blur(generated_grid, self.smoothing_sigma, False)
        reference_grid = gaussian_blur(reference_grid, self.smoothing_sigma, False)
        shared_diagnostics = {
            "dimensions": self.dimensions,
            "directions": self.directions,
            "order": self.order,
            "lambda_spatial": self.lambda_spatial,
            "spatial_mode": self.spatial_mode,
            "min_persistence": self.min_persistence,
        }
        results: dict[str, TermResult] = {}
        per_sample = generated.sum(dim=(1, 2)) * 0.0
        if self.component_weights.get("self", 0.0) > 0:
            cost, counts = self._self_cost(generated_grid, reference_grid)
            results[f"{self.family_name}.self.persistence_matching"] = TermResult(
                cost, cost.mean(), diagnostics={**shared_diagnostics, **counts}
            )
            per_sample = per_sample + self.component_weights["self"] * cost
        if self.component_weights.get("mutual", 0.0) > 0:
            cost, counts = self._mutual_cost(generated_grid, reference_grid)
            results[f"{self.family_name}.mutual.fibered_matching"] = TermResult(
                cost,
                cost.mean(),
                diagnostics={
                    **shared_diagnostics,
                    "pairs": self.mutual_pairs,
                    "lines": self.fibered_lines,
                    "angle_margin": self.angle_margin,
                    "line_sampling": f"{self.line_sampling}_random_per_call",
                    "line_seed": self.line_seed,
                    **counts,
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
                "geometry_sha256": self.geometry_sha256,
                "geometry": dict(self.geometry_diagnostics),
                "component_weights": dict(self.component_weights),
                "backend": "gudhi" if gudhi_available() else "unavailable",
                "reduction_backend": self._reduction_backend,
                "matching_formulation": "reduced_rectangular",
                "solver": "exact_hungarian",
                "workers": worker_count(),
                "reference_cache": {
                    "hits": self._reference_cache_hits,
                    "misses": self._reference_cache_misses,
                    "entries": len(self._reference_cache),
                    "capacity": _REFERENCE_CACHE_CAPACITY,
                },
            },
        )

    # ----------------------------------------------------------------- artifacts
    def state_artifact(self) -> dict[str, Any]:
        return {
            "family": self.family_name,
            "version": self.version,
            "scientific_source": {
                "repository": "https://github.com/jachen25/PhyCoFlow_dev/tree/main/src",
                "modes": ("persistence_satloss_cubical", "fibered_satloss_cubical"),
                "revision": None,
                "license_status": "not_declared; independent implementation",
            },
            "config": self.config,
            "field_names": self.field_names,
            "target_use": self.target_use,
            "units": self.units,
            "geometry_sha256": self.geometry_sha256,
            "geometry_diagnostics": dict(self.geometry_diagnostics),
            "line_rng_state": self._line_rng.bit_generator.state,
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
        self.geometry_diagnostics = dict(artifact.get("geometry_diagnostics", {}))
        rng_state = artifact.get("line_rng_state")
        if rng_state is not None:
            self._line_rng.bit_generator.state = rng_state
        # Generators cached against a previous geometry must not survive a reload.
        self._reference_cache.clear()
        self._coordinate_cache.clear()
        self.load_state_dict(state, strict=True)
