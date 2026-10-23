"""Default-off, unbiased sampling over complete persistence filtrations.

The historical :class:`PersistenceTopologyObjective` remains the value and
gradient oracle. This helper reuses its descriptor normalization and distance
primitives while constructing persistence diagrams only for selected complete
filtrations. A work unit always includes every configured homology dimension.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from ....contracts import FamilyResult, TermResult
from ...base import require_field_tensor
from .betti_curves import gaussian_blur
from .geometry import rasterize_fields
from .persistence import (
    cubical_diagrams,
    sliced_diagram_distances,
)
from .spatial_persistence import spatial_diagram_distance


@dataclass(frozen=True)
class FiltrationWorkUnit:
    """One complete scalar filtration and its coefficient in the full score."""

    index: int
    identity: str
    component: str
    direction: str
    field: str | None
    group_index: int | None
    line_index: int | None
    line_weight: float
    coefficient: float


@dataclass(frozen=True)
class FiltrationChoice:
    """A sampled work-unit set and each unit's marginal inclusion probability."""

    strategy: str
    label: str
    indices: tuple[int, ...]
    inclusion_probabilities: tuple[float, ...]


@dataclass(frozen=True)
class FiltrationContribution:
    identity: str
    inclusion_probability: float
    original_weight: float
    per_sample: torch.Tensor


@dataclass(frozen=True)
class FiltrationEstimate:
    """An unbiased estimate of the existing per-sample topology scalar."""

    scalar_loss: torch.Tensor
    per_sample: torch.Tensor
    choice: FiltrationChoice
    contributions: tuple[FiltrationContribution, ...]
    full_scalar_filtrations: int

    @property
    def sampled_units(self) -> tuple[str, ...]:
        return tuple(item.identity for item in self.contributions)

    @property
    def diagnostics(self) -> dict:
        return {
            "estimator_value": "T_estimate" if self.choice.strategy != "full" else "T_full",
            "estimator": self.choice.strategy,
            "topology_units": self.sampled_units,
            "inclusion_probabilities": self.choice.inclusion_probabilities,
            "original_weights": tuple(item.original_weight for item in self.contributions),
            "scalar_filtrations": len(self.contributions),
            "full_scalar_filtrations": self.full_scalar_filtrations,
        }


def _objective_fingerprint(objective) -> str:
    """Bind sampler state to the mathematical settings and fixed line bank."""
    line_state = []
    for index in range(len(objective.groups)):
        for prefix in ("line_directions", "line_offsets"):
            tensor = getattr(objective, f"{prefix}_{index}")
            line_state.append(
                (
                    prefix,
                    index,
                    tuple(tensor.detach().to(device="cpu").reshape(-1).tolist()),
                )
            )
    spec = (
        tuple(objective.fields),
        tuple(objective.ids),
        tuple(
            (provider, tuple(objective.fields[index] for index in indices))
            for provider, indices in objective.descriptors
        ),
        tuple(objective.self_ids),
        tuple(objective.self_fields),
        tuple(objective.directions),
        tuple(objective.dimensions),
        tuple(tuple(objective.fields[i] for i in group) for group in objective.groups),
        tuple(sorted((name, float(value)) for name, value in objective.weights.items())),
        bool(objective.periodic),
        tuple(float(value) for value in objective.periods),
        float(objective.scale_floor),
        int(objective.projections),
        float(objective.essential_weight),
        str(objective.distance),
        float(objective.spatial_weight),
        str(objective.spatial_mode),
        int(objective.max_assignment_size),
        tuple(line_state),
    )
    return hashlib.sha256(repr(spec).encode("utf-8")).hexdigest()


def _work_units(objective) -> tuple[FiltrationWorkUnit, ...]:
    """Enumerate units in exactly the same direction/component order as v3."""
    records = []
    for direction in objective.directions:
        if direction not in {"sublevel", "superlevel"}:
            raise ValueError(f"unsupported persistence direction: {direction!r}")
        if objective.weights.get("self.persistence", 0.0) > 0:
            for field in objective.self_fields:
                records.append(
                    {
                        "component": "self",
                        "direction": direction,
                        "field": field,
                        "group_index": None,
                        "line_index": None,
                        "line_weight": 1.0,
                        "identity": f"self:{field}:{direction}",
                    }
                )
        if objective.weights.get("mutual.persistence", 0.0) > 0:
            for group_index, group in enumerate(objective.groups):
                line_directions = getattr(objective, f"line_directions_{group_index}")
                group_name = "+".join(objective.fields[i] for i in group)
                for line_index, direction_vector in enumerate(line_directions):
                    line_weight = float(direction_vector.min().detach().cpu())
                    records.append(
                        {
                            "component": "mutual",
                            "direction": direction,
                            "field": None,
                            "group_index": group_index,
                            "line_index": line_index,
                            "line_weight": line_weight,
                            "identity": (f"mutual:{group_name}:line{line_index}:{direction}"),
                        }
                    )

    counts = Counter(record["component"] for record in records)
    component_weights = {
        "self": float(objective.weights.get("self.persistence", 0.0)),
        "mutual": float(objective.weights.get("mutual.persistence", 0.0)),
    }
    result = []
    for index, record in enumerate(records):
        count = counts[record["component"]]
        if count < 1:
            raise ValueError(f"no work units for {record['component']} persistence")
        coefficient = component_weights[record["component"]] / count
        coefficient *= record["line_weight"]
        result.append(FiltrationWorkUnit(index=index, coefficient=coefficient, **record))
    return tuple(result)


def _filtration_values(objective, fields: torch.Tensor, unit: FiltrationWorkUnit):
    """Build one selected raster using the existing objective's fixed geometry."""
    sign = 1.0 if unit.direction == "sublevel" else -1.0
    if unit.component == "self":
        return sign * fields[:, objective.self_fields.index(unit.field)]

    group = objective.groups[unit.group_index]
    direction = getattr(objective, f"line_directions_{unit.group_index}")[unit.line_index].to(
        fields
    )
    offset = getattr(objective, f"line_offsets_{unit.group_index}")[unit.line_index].to(fields)
    return (
        (sign * fields[:, group] - offset[None, :, None, None]) / direction[None, :, None, None]
    ).amax(1)


class TopologyFiltrationSampler:
    """Sample P3 or S2 complete filtrations with a private, resumable RNG.

    ``objective`` is an existing ``PersistenceTopologyObjective``. The sampled
    modes deliberately accept only the Run-11 profile of two self fields, one
    mutual group with one line, and both polarities. ``full`` is provided for
    direct comparison and leaves the legacy objective available as the oracle.
    """

    STATE_VERSION = 1

    def __init__(self, objective, strategy: str = "p3", *, seed: int = 0):
        if strategy not in {"full", "p3", "s2"}:
            raise ValueError("strategy must be 'full', 'p3', or 's2'")
        if isinstance(seed, bool) or int(seed) != seed or not 0 <= int(seed) < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        self.objective = objective
        self.strategy = strategy
        self.units = _work_units(objective)
        self._fingerprint = _objective_fingerprint(objective)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(int(seed))
        if strategy != "full":
            self._validate_run11_profile()
        self._choices = self._enumerate_choices()
        self._choice_keys = {
            (choice.indices, choice.inclusion_probabilities): choice for choice in self._choices
        }

    def _validate_run11_profile(self) -> None:
        if (
            len(self.objective.self_fields) != 2
            or len(self.objective.groups) != 1
            or len(self.objective.directions) != 2
            or set(self.objective.directions) != {"sublevel", "superlevel"}
            or len(self.objective.line_directions_0) != 1
            or self.objective.weights.get("self.persistence", 0.0) <= 0
            or self.objective.weights.get("mutual.persistence", 0.0) <= 0
        ):
            raise ValueError(
                "P3/S2 require two self fields, one mutual group/line, both polarities, "
                "and positive self/mutual weights"
            )
        if tuple(self.objective.dimensions) != (0, 1):
            raise ValueError("P3/S2 keep the complete H0/H1 filtration output")
        if self.objective.distance != "sliced_wasserstein" or self.objective.projections != 32:
            raise ValueError("P3/S2 require the existing 32-direction sliced-Wasserstein metric")
        per_direction = Counter((unit.direction, unit.component) for unit in self.units)
        if any(
            per_direction[(direction, component)] != expected
            for direction in ("sublevel", "superlevel")
            for component, expected in (("self", 2), ("mutual", 1))
        ):
            raise ValueError("P3/S2 require exactly three complete filtrations per polarity")

    def _enumerate_choices(self) -> tuple[FiltrationChoice, ...]:
        if self.strategy == "full":
            indices = tuple(unit.index for unit in self.units)
            return (FiltrationChoice("full", "full", indices, (1.0,) * len(indices)),)
        if self.strategy == "p3":
            return tuple(
                FiltrationChoice(
                    "p3",
                    direction,
                    tuple(unit.index for unit in self.units if unit.direction == direction),
                    (0.5, 0.5, 0.5),
                )
                for direction in ("sublevel", "superlevel")
            )

        self_units = [unit for unit in self.units if unit.component == "self"]
        mutual_units = [unit for unit in self.units if unit.component == "mutual"]
        choices = []
        for self_unit in self_units:
            for mutual_unit in mutual_units:
                selected = tuple(sorted((self_unit.index, mutual_unit.index)))
                probabilities = tuple(
                    0.25 if self.units[index].component == "self" else 0.5 for index in selected
                )
                label = "+".join(self.units[index].identity for index in selected)
                choices.append(FiltrationChoice("s2", label, selected, probabilities))
        return tuple(choices)

    def enumerate_choices(self) -> tuple[FiltrationChoice, ...]:
        """Return all exact support points for oracle expectation checks."""
        return self._choices

    def draw(self) -> FiltrationChoice:
        """Draw one choice without consuming or changing PyTorch's global RNG."""
        choice_index = int(
            torch.randint(len(self._choices), (1,), generator=self._generator).item()
        )
        return self._choices[choice_index]

    def state_dict(self) -> dict:
        return {
            "version": self.STATE_VERSION,
            "strategy": self.strategy,
            "objective_fingerprint": self._fingerprint,
            "unit_identities": tuple(unit.identity for unit in self.units),
            "generator_state": self._generator.get_state().clone(),
        }

    def load_state_dict(self, state: Mapping) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("sampler state must be a mapping")
        if state.get("version") != self.STATE_VERSION:
            raise ValueError("unsupported topology sampler state version")
        if state.get("strategy") != self.strategy:
            raise ValueError("topology sampler strategy changed across resume")
        if state.get("objective_fingerprint") != self._fingerprint:
            raise ValueError("topology objective changed across sampler resume")
        if tuple(state.get("unit_identities", ())) != tuple(unit.identity for unit in self.units):
            raise ValueError("topology filtration identities changed across sampler resume")
        generator_state = state.get("generator_state")
        if not isinstance(generator_state, torch.Tensor) or generator_state.device.type != "cpu":
            raise ValueError("topology sampler RNG state must be a CPU tensor")
        try:
            self._generator.set_state(generator_state)
        except RuntimeError as error:
            raise ValueError("invalid topology sampler RNG state") from error

    def _validated_choice(self, choice: FiltrationChoice) -> FiltrationChoice:
        if not isinstance(choice, FiltrationChoice) or choice.strategy != self.strategy:
            raise ValueError("choice does not belong to this topology sampler")
        key = (tuple(choice.indices), tuple(choice.inclusion_probabilities))
        try:
            return self._choice_keys[key]
        except KeyError as error:
            raise ValueError("choice is not in the sampler's exact support") from error

    def _reference_state(self, reference: torch.Tensor, reference_cache: dict | None) -> dict:
        cache = {} if reference_cache is None else reference_cache
        cache_key = ("topology_filtration_sampler", id(self))
        state = cache.get(cache_key)
        if state is not None:
            if (
                state["reference"] is not reference
                or state["reference_version"] != reference._version
                or state["reference_shape"] != tuple(reference.shape)
            ):
                raise ValueError("topology sampler reference cache target changed")
            return state

        with torch.no_grad():
            descriptors = self.objective._descriptor_fields(reference.detach())
            mean = descriptors.mean((-2, -1), keepdim=True)
            scale = descriptors.std((-2, -1), keepdim=True, unbiased=False).clamp_min(
                self.objective.scale_floor
            )
            normalized = (descriptors - mean) / scale
        state = {
            "reference": reference,
            "reference_version": reference._version,
            "reference_shape": tuple(reference.shape),
            "mean": mean,
            "scale": scale,
            "normalized": normalized,
            "diagrams": {},
        }
        cache[cache_key] = state
        return state

    def evaluate(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor,
        *,
        choice: FiltrationChoice | None = None,
        reference_cache: dict | None = None,
    ) -> FiltrationEstimate:
        """Evaluate selected complete units with Horvitz-Thompson weights."""
        if choice is None:
            choice = self.draw()
        choice = self._validated_choice(choice)
        if generated.ndim != 4 or reference.ndim != 4:
            raise ValueError("generated and reference must both have shape [B,C,H,W]")
        if tuple(generated.shape) != tuple(reference.shape):
            raise ValueError("generated and reference topology tensors must have equal shape")
        if generated.device != reference.device:
            raise ValueError("generated and reference topology tensors must share a device")
        if generated.shape[-2] < 2 or generated.shape[-1] < 2:
            raise ValueError("cubical persistence requires raster dimensions >= 2")

        prepared = self._reference_state(reference, reference_cache)
        normalized_generated = (
            self.objective._descriptor_fields(generated) - prepared["mean"]
        ) / prepared["scale"]
        selected_units = [self.units[index] for index in choice.indices]
        generated_filtrations = [
            _filtration_values(self.objective, normalized_generated, unit)
            for unit in selected_units
        ]
        generated_bank = torch.stack(generated_filtrations)
        batch, height, width = generated.shape[0], generated.shape[-2], generated.shape[-1]
        generated_diagrams = cubical_diagrams(
            generated_bank.flatten(0, 1),
            periodic=self.objective.periodic,
            dimensions=self.objective.dimensions,
            locations=self.objective.distance == "spatial_wasserstein",
        )

        target_diagrams = []
        for unit in selected_units:
            cached = prepared["diagrams"].get(unit.index)
            if cached is None:
                target_filtration = _filtration_values(self.objective, prepared["normalized"], unit)
                cached = cubical_diagrams(
                    target_filtration,
                    periodic=self.objective.periodic,
                    dimensions=self.objective.dimensions,
                    locations=self.objective.distance == "spatial_wasserstein",
                    cacheable=True,
                )
                prepared["diagrams"][unit.index] = cached
            target_diagrams.append(cached)

        if self.objective.distance == "sliced_wasserstein":
            all_costs = sliced_diagram_distances(
                [item[dim] for item in generated_diagrams for dim in self.objective.dimensions],
                [
                    item[dim]
                    for unit_diagrams in target_diagrams
                    for item in unit_diagrams
                    for dim in self.objective.dimensions
                ],
                projections=self.objective.projections,
                essential_weight=self.objective.essential_weight,
                normalization=height * width,
            ).reshape(len(selected_units), batch, len(self.objective.dimensions))
        else:
            all_costs = torch.stack(
                [
                    torch.stack(
                        [
                            torch.stack(
                                [
                                    spatial_diagram_distance(
                                        generated_diagrams[unit_index * batch + batch_index][
                                            dimension
                                        ],
                                        target_diagrams[unit_index][batch_index][dimension],
                                        essential_weight=self.objective.essential_weight,
                                        normalization=height * width,
                                        periodic=self.objective.periodic,
                                        spatial_weight=self.objective.spatial_weight,
                                        spatial_mode=self.objective.spatial_mode,
                                        max_assignment_size=self.objective.max_assignment_size,
                                    )
                                    for batch_index in range(batch)
                                ]
                            )
                            for dimension in self.objective.dimensions
                        ]
                    ).transpose(0, 1)
                    for unit_index in range(len(selected_units))
                ],
                dim=0,
            )

        contributions = []
        for position, (unit, inclusion_probability) in enumerate(
            zip(selected_units, choice.inclusion_probabilities)
        ):
            unit_cost = all_costs[position].mean(-1)
            estimate = unit_cost * (unit.coefficient / inclusion_probability)
            contributions.append(
                FiltrationContribution(
                    unit.identity,
                    inclusion_probability,
                    unit.coefficient,
                    estimate,
                )
            )
        per_sample = torch.stack([item.per_sample for item in contributions]).sum(0)
        return FiltrationEstimate(
            scalar_loss=per_sample.mean(),
            per_sample=per_sample,
            choice=choice,
            contributions=tuple(contributions),
            full_scalar_filtrations=len(self.units),
        )


def sampled_family_forward(
    family,
    prediction: torch.Tensor,
    reference: torch.Tensor,
    coordinates: torch.Tensor,
    sampler: TopologyFiltrationSampler | None = None,
    choice: FiltrationChoice | None = None,
    *,
    strategy: str | None = None,
    seed: int | None = None,
    context: Mapping | None = None,
) -> FamilyResult:
    """Run a cubical-persistence family through its unchanged raster prelude.

    This adapter mirrors ``TopologyFamily.forward`` through rasterization and
    smoothing, then substitutes only the final complete-filtration evaluator.
    The family module and its hash-bound persistence implementation stay intact.
    """
    if getattr(family, "strategy", None) != "cubical_persistence":
        raise ValueError("sampled topology forwarding requires cubical_persistence")
    if sampler is None:
        if seed is None:
            raise ValueError("sampled topology forwarding requires a dedicated integer seed")
        sampler = TopologyFiltrationSampler(
            family.spatial_objective,
            strategy="p3" if strategy is None else strategy,
            seed=seed,
        )
    elif sampler.objective is not getattr(family, "spatial_objective", None):
        raise ValueError("topology sampler must wrap this family's persistence objective")
    elif strategy is not None and sampler.strategy != strategy:
        raise ValueError("topology sampler strategy differs from requested adapter strategy")
    if getattr(family, "spatial_objective", None) is not sampler.objective:
        raise ValueError("topology sampler must wrap this family's persistence objective")
    if context is None:
        context = {}
    reference_cache = context.get("persistence_cache")
    if reference_cache is not None:
        signature = (
            id(family),
            id(reference),
            reference._version,
            id(coordinates),
            coordinates._version if coordinates is not None else None,
        )
        reference_cache.setdefault("reference_identity", (reference, coordinates))
        if reference_cache.setdefault("signature", signature) != signature:
            raise ValueError("persistence reference cache target/coordinates changed")

    require_field_tensor("generated", prediction)
    require_field_tensor("reference", reference)
    if prediction.shape != reference.shape:
        raise ValueError("topology generated/reference shapes differ")
    if coordinates is None or coordinates.shape[:2] != prediction.shape[:2]:
        raise ValueError("topology requires coordinates aligned with [B,N]")
    if not torch.isfinite(prediction).all() or not torch.isfinite(reference).all():
        raise FloatingPointError("topology generated/reference fields must be finite")

    reference = reference.detach()
    if prediction.dtype in {torch.float16, torch.bfloat16}:
        prediction = prediction.float()
    reference = reference.to(prediction)
    family._raster_map(coordinates)
    prediction, reference = family._units_and_fields(prediction, reference)
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        generated_grid = rasterize_fields(
            prediction,
            family.neighbor_indices,
            family.neighbor_weights.to(prediction),
            family.grid_shape,
        )
        reference_grid = rasterize_fields(
            reference,
            family.neighbor_indices,
            family.neighbor_weights.to(reference),
            family.grid_shape,
        )
        generated_grid = gaussian_blur(
            generated_grid,
            family.smoothing_sigma,
            family.periodic,
            radius=family.smoothing_radius,
        )
        reference_grid = gaussian_blur(
            reference_grid,
            family.smoothing_sigma,
            family.periodic,
            radius=family.smoothing_radius,
        )
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        estimate = sampler.evaluate(
            generated_grid.float(),
            reference_grid.float(),
            choice=choice,
            reference_cache=reference_cache,
        )

    component_results = {}
    for component, key in (
        ("self", "self.persistence"),
        ("mutual", "mutual.persistence"),
    ):
        component_weight = float(sampler.objective.weights.get(key, 0.0))
        if component_weight <= 0:
            continue
        selected = [
            contribution
            for unit, contribution in zip(
                (sampler.units[index] for index in estimate.choice.indices),
                estimate.contributions,
            )
            if unit.component == component
        ]
        if not selected:
            raise RuntimeError(f"sampled topology choice omitted the {component} category")
        per_sample = torch.stack([item.per_sample for item in selected]).sum(0) / component_weight
        component_results[f"topology.{key}"] = TermResult(
            per_sample,
            per_sample.mean(),
            diagnostics={
                "estimator_value": "T_estimate" if sampler.strategy != "full" else "T_full",
                "estimator": sampler.strategy,
                "sampled_units": tuple(item.identity for item in selected),
                "inclusion_probabilities": tuple(item.inclusion_probability for item in selected),
                "original_weights": tuple(item.original_weight for item in selected),
            },
        )

    if not torch.isfinite(estimate.per_sample).all():
        raise FloatingPointError("sampled topology family produced a non-finite cost")
    diagnostics = estimate.diagnostics
    diagnostics.update(
        {
            "backend": "gudhi_cpu_pairing_torch_device_distance",
            "diagram_projections": sampler.objective.projections,
            "finite_slice_approximation": True,
            "essential_classes_included": True,
            "point_normalization": generated_grid.shape[-2] * generated_grid.shape[-1],
            "descriptor_fields": sampler.objective.fields,
            "self_descriptor_fields": sampler.objective.self_fields,
            "diagram_distance": sampler.objective.distance,
            "family": family.family_name,
            "version": family.version,
            "aggregation": "per_sample",
            "target_use": family.target_use,
            "units": family.units,
            "fields": family.field_names,
            "grid_shape": family.grid_shape,
            "periodic": family.periodic,
            "periods": family.periods,
            "geometry_sha256": family.geometry_sha256,
            "geometry": dict(family.geometry_diagnostics),
            "component_weights": dict(family.component_weights),
        }
    )
    return FamilyResult(
        component_results=component_results,
        per_sample_cost=estimate.per_sample,
        scalar_loss=estimate.scalar_loss,
        diagnostics=diagnostics,
    )
