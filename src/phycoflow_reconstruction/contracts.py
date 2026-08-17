"""Typed boundaries shared by datasets, models, cases, and trainers.

The dataclasses keep model implementations independent of file formats and
case names. Shape checks fail early with messages that are useful in case-level
launches and small integration tests.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from math import prod
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class DataSpec:
    field_names: tuple[str, ...]
    field_units: tuple[str, ...]
    coordinate_dim: int
    logical_shape: tuple[int, ...]
    reconstruction_unit: str = "snapshot"
    mesh_type: str = "point"

    @property
    def num_fields(self) -> int:
        return len(self.field_names)

    def validate(self) -> None:
        if not self.field_names or len(set(self.field_names)) != len(self.field_names):
            raise ValueError("field_names must be non-empty and unique")
        if len(self.field_units) != self.num_fields:
            raise ValueError("field_units must align with field_names")
        if self.coordinate_dim < 1:
            raise ValueError("coordinate_dim must be positive")
        if not self.logical_shape or any(int(size) < 1 for size in self.logical_shape):
            raise ValueError("logical_shape must contain positive axis sizes")
        if self.reconstruction_unit not in {"snapshot", "space_time_trajectory"}:
            raise ValueError(f"unsupported reconstruction_unit={self.reconstruction_unit!r}")
        if not self.mesh_type:
            raise ValueError("mesh_type cannot be empty")


@dataclass
class FieldSample:
    values: torch.Tensor
    coordinates: torch.Tensor
    coordinates_raw: torch.Tensor
    time: torch.Tensor
    trajectory_id: str
    time_index: int | None
    conditions: torch.Tensor
    field_names: tuple[str, ...]
    logical_shape: tuple[int, ...]
    reconstruction_unit: str = "snapshot"
    valid_points: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(f"values must be [N,C], got {tuple(self.values.shape)}")
        if self.coordinates.ndim != 2 or self.coordinates_raw.ndim != 2:
            raise ValueError("coordinates and coordinates_raw must be [N,D]")
        if self.values.shape[0] != self.coordinates.shape[0]:
            raise ValueError("value and coordinate point counts differ")
        if self.coordinates.shape != self.coordinates_raw.shape:
            raise ValueError("normalized and raw coordinate shapes differ")
        if self.values.shape[1] != len(self.field_names):
            raise ValueError("field_names do not align with value channels")
        if not self.field_names or len(set(self.field_names)) != len(self.field_names):
            raise ValueError("field_names must be non-empty and unique")
        if not self.logical_shape or any(int(size) < 1 for size in self.logical_shape):
            raise ValueError("logical_shape must contain positive axis sizes")
        if prod(self.logical_shape) != self.values.shape[0]:
            raise ValueError("logical_shape point count does not align with values")
        if self.reconstruction_unit not in {"snapshot", "space_time_trajectory"}:
            raise ValueError(f"unsupported reconstruction_unit={self.reconstruction_unit!r}")
        if self.reconstruction_unit == "space_time_trajectory" and (
            self.time.ndim != 1 or self.time.numel() != self.logical_shape[0]
        ):
            raise ValueError("space-time time coordinates must align with the first logical axis")
        if self.conditions.ndim != 1:
            raise ValueError("conditions must be a one-dimensional feature vector")
        if self.valid_points is not None:
            if self.valid_points.shape != self.values.shape[:1]:
                raise ValueError("valid_points must have shape [N]")
            if self.valid_points.dtype != torch.bool:
                raise TypeError("valid_points must use boolean dtype")


@dataclass
class ObservationBatch:
    obs_coords: torch.Tensor
    obs_values: torch.Tensor
    obs_field_ids: torch.Tensor
    obs_valid_mask: torch.Tensor
    query_coords: torch.Tensor
    query_valid_mask: torch.Tensor
    target_fields: torch.Tensor | None
    sample_ids: tuple[str, ...]
    obs_indices: torch.Tensor | None = None
    logical_shapes: tuple[tuple[int, ...], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.obs_coords.ndim != 3 or self.query_coords.ndim != 3:
            raise ValueError("obs_coords and query_coords must be [B,N,D]")
        bsz, obs_count, _ = self.obs_coords.shape
        if self.query_coords.shape[0] != bsz:
            raise ValueError("observation and query batch sizes differ")
        if self.query_coords.shape[2] != self.obs_coords.shape[2]:
            raise ValueError("observation and query coordinate dimensions differ")
        if self.obs_values.shape != (bsz, obs_count, 1):
            raise ValueError("obs_values must be [B,M,1]")
        if self.obs_field_ids.shape != (bsz, obs_count):
            raise ValueError("obs_field_ids must be [B,M]")
        if self.obs_valid_mask.shape != (bsz, obs_count):
            raise ValueError("obs_valid_mask must be [B,M]")
        if self.obs_valid_mask.dtype != torch.bool:
            raise TypeError("obs_valid_mask must use boolean dtype")
        if self.query_valid_mask.shape != self.query_coords.shape[:2]:
            raise ValueError("query_valid_mask must align with query points")
        if self.query_valid_mask.dtype != torch.bool:
            raise TypeError("query_valid_mask must use boolean dtype")
        if self.target_fields is not None and (
            self.target_fields.ndim != 3
            or self.target_fields.shape[:2] != self.query_coords.shape[:2]
        ):
            raise ValueError("target_fields must be [B,Q,C]")
        if len(self.sample_ids) != bsz:
            raise ValueError("sample_ids must contain one id per batch item")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids must be unique within a batch")
        if self.logical_shapes and len(self.logical_shapes) != bsz:
            raise ValueError("logical_shapes must contain one shape per batch item")
        if self.obs_indices is not None:
            if self.obs_indices.shape != (bsz, obs_count):
                raise ValueError("obs_indices must be [B,M]")
            if self.obs_indices.dtype not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                raise TypeError("obs_indices must use an integer dtype")
        if self.obs_field_ids.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("obs_field_ids must use an integer dtype")
        if torch.any(self.obs_valid_mask.sum(dim=1) == 0):
            raise ValueError("every batch item must contain at least one valid observation")
        if torch.any(self.query_valid_mask.sum(dim=1) == 0):
            raise ValueError("every batch item must contain at least one valid query")
        valid_fields = self.obs_field_ids[self.obs_valid_mask]
        if torch.any(valid_fields < 0):
            raise ValueError("valid observations cannot use negative field IDs")
        if self.target_fields is not None and torch.any(
            valid_fields >= self.target_fields.shape[-1]
        ):
            raise ValueError("observation field ID exceeds target channel count")
        if self.obs_indices is not None and torch.any(self.obs_indices[self.obs_valid_mask] < 0):
            raise ValueError("valid observations cannot use negative point indices")

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> ObservationBatch:
        def move(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.to(device, non_blocking=non_blocking)

        def move_nested(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.to(device, non_blocking=non_blocking)
            if isinstance(value, dict):
                return {key: move_nested(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return tuple(move_nested(item) for item in value)
            if isinstance(value, list):
                return [move_nested(item) for item in value]
            return value

        return ObservationBatch(
            obs_coords=self.obs_coords.to(device, non_blocking=non_blocking),
            obs_values=self.obs_values.to(device, non_blocking=non_blocking),
            obs_field_ids=self.obs_field_ids.to(device, non_blocking=non_blocking),
            obs_valid_mask=self.obs_valid_mask.to(device, non_blocking=non_blocking),
            query_coords=self.query_coords.to(device, non_blocking=non_blocking),
            query_valid_mask=self.query_valid_mask.to(device, non_blocking=non_blocking),
            target_fields=move(self.target_fields),
            sample_ids=self.sample_ids,
            obs_indices=move(self.obs_indices),
            logical_shapes=self.logical_shapes,
            metadata=move_nested(self.metadata),
        )

    def pin_memory(self) -> ObservationBatch:
        """Pin compact batch tensors so asynchronous CUDA copies can overlap loading."""

        def pin_tensor(value: torch.Tensor) -> torch.Tensor:
            # Expanded shared-query views have overlapping zero-stride storage,
            # which CUDA pinning cannot write into. Materialize only those views.
            return (value if value.is_contiguous() else value.contiguous()).pin_memory()

        def pin(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else pin_tensor(value)

        def pin_nested(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return pin_tensor(value)
            if isinstance(value, dict):
                return {key: pin_nested(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return tuple(pin_nested(item) for item in value)
            if isinstance(value, list):
                return [pin_nested(item) for item in value]
            return value

        return ObservationBatch(
            obs_coords=pin_tensor(self.obs_coords),
            obs_values=pin_tensor(self.obs_values),
            obs_field_ids=pin_tensor(self.obs_field_ids),
            obs_valid_mask=pin_tensor(self.obs_valid_mask),
            query_coords=pin_tensor(self.query_coords),
            query_valid_mask=pin_tensor(self.query_valid_mask),
            target_fields=pin(self.target_fields),
            sample_ids=self.sample_ids,
            obs_indices=pin(self.obs_indices),
            logical_shapes=self.logical_shapes,
            metadata=pin_nested(self.metadata),
        )


@dataclass
class ReconstructionBatch:
    prediction: torch.Tensor
    samples: torch.Tensor | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossBundle:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelCapabilities:
    representation: str
    generative: bool
    differentiable_rollout: bool
    structured_grid_required: bool
    uncertainty_samples: bool
    stages: tuple[str, ...] = ("base_training",)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    display_name: str
    field_names: tuple[str, ...]
    field_units: tuple[str, ...]
    reconstruction_unit: str
    mesh_type: str
    grid_shape: tuple[int, ...]
    benchmark_eligible: bool = True
    physics_factory: Callable[..., Any] | None = None
    diagnostics_factory: Callable[..., Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name or not self.field_names:
            raise ValueError("case name and fields are required")
        if len(self.field_names) != len(self.field_units):
            raise ValueError("case field names and units must align")


@dataclass(frozen=True)
class CoherenceComponentSpec:
    """Scientific disclosure required from every future component."""

    name: str
    target_use: str
    units: str
    differentiable: bool
    required_geometry: str = "none"
    aggregation: str = "per_sample"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("coherence component name cannot be empty")
        if self.target_use not in {"training_reference", "paired_supervised", "none"}:
            raise ValueError(f"invalid coherence target_use={self.target_use!r}")
        if self.units not in {"model_units", "physical_units"}:
            raise ValueError(f"invalid coherence units={self.units!r}")
        if self.aggregation not in {"per_sample", "ensemble"}:
            raise ValueError(f"invalid coherence aggregation={self.aggregation!r}")


@dataclass(frozen=True)
class CoherenceFamilySpec:
    """Top-level family contract; component taxonomies remain family-owned."""

    name: str
    version: str
    components: tuple[CoherenceComponentSpec, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name or not self.version:
            raise ValueError("coherence family name and version are required")
        names = [component.name for component in self.components]
        if len(names) != len(set(names)):
            raise ValueError("coherence component names must be unique within a family")
        for component in self.components:
            component.validate()


@dataclass(frozen=True)
class CoherenceContext:
    """Reference provenance supplied to a coherence family at runtime."""

    split: str
    reference_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class TermResult:
    """One coherence component result with explicit availability semantics."""

    per_sample_cost: torch.Tensor | None
    scalar_loss: torch.Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)
    valid_mask: torch.Tensor | None = None
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.reason is None


@dataclass
class FamilyResult:
    """Aggregated result while preserving family-owned component paths."""

    component_results: dict[str, TermResult]
    per_sample_cost: torch.Tensor | None
    scalar_loss: torch.Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)


class PhysicsProvider(Protocol):
    """Case-owned differentiable or diagnostic physics interface."""

    def loss(self, prediction: torch.Tensor, batch: ObservationBatch) -> LossBundle: ...


class CoherenceComponent(Protocol):
    spec: CoherenceComponentSpec

    def __call__(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor,
    ) -> TermResult: ...


class CoherenceFamily(Protocol):
    spec: CoherenceFamilySpec
    family_name: str
    target_use: str
    units: str

    def __call__(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor,
        *,
        coordinates: torch.Tensor | None = None,
        context: CoherenceContext | Mapping[str, Any] | None = None,
    ) -> FamilyResult: ...


class ReconstructionModel(Protocol):
    capabilities: ModelCapabilities

    def training_loss(self, batch: ObservationBatch) -> LossBundle: ...

    def reconstruct(self, batch: ObservationBatch, **kwargs: Any) -> ReconstructionBatch: ...


class DifferentiableReconstructionModel(ReconstructionModel, Protocol):
    """Model-owned differentiable inference used by coherence post-training."""

    def differentiable_reconstruct(
        self,
        batch: ObservationBatch,
        *,
        steps: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor: ...


class DifferentiableFlowModel(ReconstructionModel, Protocol):
    """Extra hooks required by the common rectified-flow post-trainer."""

    def sample_source(
        self,
        batch: ObservationBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor: ...

    def velocity(
        self,
        batch: ObservationBatch,
        state: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor: ...
