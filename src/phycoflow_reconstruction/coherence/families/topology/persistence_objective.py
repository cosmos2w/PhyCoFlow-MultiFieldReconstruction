"""Self persistence and finite positive-line restrictions of joint filtrations."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn

from ....contracts import CoherenceComponentSpec, CoherenceFamilySpec, FamilyResult, TermResult
from .persistence import cubical_diagrams, sliced_diagram_distance, sliced_diagram_distances
from .spatial import central_gradient
from .spatial_persistence import spatial_diagram_distance

PERSISTENCE_COMPONENT_KEYS = {
    "self": {"enabled", "weight", "fields"},
    "mutual": {"enabled", "weight", "groups", "lines", "seed", "direction_floor", "offset_range"},
}
PERSISTENCE_KEYS = {
    "projections",
    "essential_weight",
    "scale_floor",
    "descriptors",
    "distance",
    "spatial_weight",
    "spatial_mode",
    "max_assignment_size",
}


def validate_persistence_config(config: Mapping, field_names=None) -> None:
    if config.get("target_use") != "paired_supervised":
        raise ValueError("cubical_persistence requires paired_supervised targets")
    if "anchor" in config or "evaluation" in config:
        raise ValueError("cubical_persistence uses explicit fields and mutual.groups")
    fields = tuple(config.get("fields") or field_names or ())
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("persistence fields must be nonempty and unique")
    if field_names is not None and not set(fields) <= set(field_names):
        raise ValueError("persistence fields must belong to the dataset")
    settings = config.get("persistence", {})
    if set(settings) - PERSISTENCE_KEYS:
        raise ValueError("unknown topology.persistence keys")
    distance = settings.get("distance", "sliced_wasserstein")
    if distance not in {"sliced_wasserstein", "spatial_wasserstein"}:
        raise ValueError("unknown persistence.distance")
    spatial_keys = {"spatial_weight", "spatial_mode", "max_assignment_size"}
    if distance == "sliced_wasserstein" and spatial_keys & settings.keys():
        raise ValueError("spatial persistence settings require distance=spatial_wasserstein")
    if settings.get("spatial_mode", "additive") not in {"additive", "multiplicative"}:
        raise ValueError("invalid persistence.spatial_mode")
    spatial_weight = float(settings.get("spatial_weight", 1.0))
    if not math.isfinite(spatial_weight) or spatial_weight < 0:
        raise ValueError("persistence.spatial_weight must be finite and nonnegative")
    limit = settings.get("max_assignment_size", 4096)
    if int(limit) != limit or limit < 1:
        raise ValueError("persistence.max_assignment_size must be a positive integer")
    descriptors = settings.get("descriptors", {})
    if not isinstance(descriptors, Mapping):
        raise TypeError("persistence.descriptors must map names to providers and fields")
    for name, descriptor in descriptors.items():
        provider = descriptor.get("provider")
        names = descriptor.get("fields", [])
        if (
            name in (field_names or fields)
            or set(descriptor) != {"provider", "fields"}
            or provider not in {"signed_vorticity", "gradient_magnitude", "vector_magnitude"}
            or not names
            or len(set(names)) != len(names)
            or not set(names) <= set(field_names or fields)
        ):
            raise ValueError("invalid persistence descriptor provider/fields or name collision")
        if provider != "vector_magnitude" and len(names) != (
            2 if provider == "signed_vorticity" else 1
        ):
            raise ValueError("invalid persistence derivative descriptor arity")
        if provider != "vector_magnitude" and (
            config.get("units") != "physical_units"
            or not config.get("geometry", {}).get("periodic", False)
            or config.get("geometry", {}).get("periods") is None
        ):
            raise ValueError(
                "derivative persistence descriptors require physical units and explicit periodic lengths"
            )
    fields = (*fields, *descriptors)
    projections = settings.get("projections", 32)
    if int(projections) != projections or projections < 2:
        raise ValueError("persistence.projections must be an integer >= 2")
    for key, default, positive in (("scale_floor", 1e-4, True), ("essential_weight", 0.1, False)):
        value = float(settings.get(key, default))
        if not math.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(f"invalid persistence.{key}")
    components = config.get("components", {})
    if set(components) - PERSISTENCE_COMPONENT_KEYS.keys():
        raise ValueError("unknown persistence components")
    active = False
    for name, allowed in PERSISTENCE_COMPONENT_KEYS.items():
        term = components.get(name, {})
        if set(term) - allowed:
            raise ValueError(f"unknown persistence {name} settings")
        weight = float(term.get("weight", 1))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("persistence weights must be finite and nonnegative")
        active |= bool(term.get("enabled", name == "self")) and weight > 0
    if not active:
        raise ValueError("persistence requires a positive component")
    self_fields = components.get("self", {}).get("fields", fields)
    if (
        not isinstance(self_fields, (list, tuple))
        or not self_fields
        or len(set(self_fields)) != len(self_fields)
        or not set(self_fields) <= set(fields)
    ):
        raise ValueError("persistence self.fields must be unique selected fields or descriptors")
    mutual = components.get("mutual", {})
    if mutual.get("enabled", False):
        groups = mutual.get("groups", [fields])
        canonical = [tuple(sorted(group)) for group in groups]
        if (
            not groups
            or len(set(canonical)) != len(groups)
            or any(
                len(group) < 2 or len(set(group)) != len(group) or not set(group) <= set(fields)
                for group in groups
            )
        ):
            raise ValueError("mutual.groups requires unique groups of >=2 selected fields")
        lines = mutual.get("lines", 8)
        if int(lines) != lines or lines < 1:
            raise ValueError("mutual.lines must be a positive integer")
        floor, extent = (
            float(mutual.get("direction_floor", 0.25)),
            float(mutual.get("offset_range", 1)),
        )
        if not 0 < floor <= 1 or not math.isfinite(extent) or extent < 0:
            raise ValueError("mutual directions must be positive and offsets finite")


class PersistenceTopologyObjective(nn.Module):
    """All finite bars and essential H0/H1; no threshold sampling or stop-grad axes.

    Shared reference scaling, signed sub/superlevel fields, and fixed Sobol line
    parameters define comparable filtrations. For k fields, a positive line
    (b + t a) restricts the sublevel module to max_i((f_i-b_i)/a_i).
    Finite slices summarize, but do not fully characterize, a k-parameter module.
    """

    def __init__(self, config: Mapping, field_names: tuple[str, ...]):
        super().__init__()
        validate_persistence_config(config, field_names)
        base_fields = tuple(config.get("fields") or field_names)
        self.ids = tuple(field_names.index(name) for name in base_fields)
        self.periodic = bool(config.get("geometry", {}).get("periodic", False))
        filtration = config.get("filtration", {})
        self.dimensions = tuple(filtration.get("dimensions", (0, 1)))
        self.directions = tuple(filtration.get("directions", ("sublevel", "superlevel")))
        settings = config.get("persistence", {})
        descriptors = settings.get("descriptors", {})
        self.descriptors = tuple(
            (item["provider"], tuple(field_names.index(name) for name in item["fields"]))
            for item in descriptors.values()
        )
        self.fields = (*base_fields, *descriptors)
        self.periods = tuple(config.get("geometry", {}).get("periods") or (1.0, 1.0))
        self.projections = int(settings.get("projections", 32))
        self.essential_weight = float(settings.get("essential_weight", 0.1))
        self.scale_floor = float(settings.get("scale_floor", 1e-4))
        self.distance = settings.get("distance", "sliced_wasserstein")
        self.spatial_weight = float(settings.get("spatial_weight", 1.0))
        self.spatial_mode = settings.get("spatial_mode", "additive")
        self.max_assignment_size = int(settings.get("max_assignment_size", 4096))
        components = config.get("components", {})
        self.self_fields = tuple(components.get("self", {}).get("fields", self.fields))
        self.self_ids = tuple(self.fields.index(name) for name in self.self_fields)
        self.weights = {
            f"{name}.persistence": float(components.get(name, {}).get("weight", 1))
            for name in ("self", "mutual")
            if components.get(name, {}).get("enabled", name == "self")
        }
        mutual = components.get("mutual", {})
        self.groups = (
            tuple(
                tuple(self.fields.index(name) for name in group)
                for group in mutual.get("groups", [self.fields])
            )
            if ("mutual.persistence" in self.weights)
            else ()
        )
        lines = int(mutual.get("lines", 8))
        for i, group in enumerate(self.groups):
            k = len(group)
            q = torch.quasirandom.SobolEngine(
                2 * k, scramble=True, seed=int(mutual.get("seed", 1729)) + i
            ).draw(lines)
            a = float(mutual.get("direction_floor", 0.25)) + q[:, :k]
            a /= a.norm(dim=1, keepdim=True)
            b = (2 * q[:, k:] - 1) * float(mutual.get("offset_range", 1))
            b -= b.mean(dim=1, keepdim=True)
            self.register_buffer(f"line_directions_{i}", a)
            self.register_buffer(f"line_offsets_{i}", b)
        metric_groups = [name for name in self.weights if self.weights[name] > 0]
        if self.weights.get("self.persistence", 0) > 0:
            metric_groups += [f"self.{field}" for field in self.self_fields]
        if self.weights.get("mutual.persistence", 0) > 0:
            metric_groups += [
                "mutual." + "+".join(self.fields[i] for i in group) for group in self.groups
            ]
        specs = [
            CoherenceComponentSpec(
                name,
                "paired_supervised",
                config.get("units", "model_units"),
                True,
                required_geometry="fixed_2d_raster",
                aggregation="per_sample",
            )
            for name in self.weights
        ]
        specs += [
            CoherenceComponentSpec(
                f"{name}.h{dim}",
                "paired_supervised",
                config.get("units", "model_units"),
                False,
                required_geometry="fixed_2d_raster",
                aggregation="per_sample",
                metadata={"evaluation_only": True, "weight": 0.0},
            )
            for name in metric_groups
            for dim in self.dimensions
        ]
        self.spec = CoherenceFamilySpec(
            "topology",
            "3",
            tuple(specs),
            metadata={
                "strategy": "cubical_persistence",
                "joint_invariant": "finite_positive_line_restrictions",
            },
        )
        self.spec.validate()

    def _descriptor_fields(self, grid):
        fields = [grid[:, i] for i in self.ids]
        h, w = grid.shape[-2:]
        for provider, ids in self.descriptors:
            if provider == "vector_magnitude":
                fields.append((grid[:, ids].square().sum(1) + 1e-12).sqrt())
                continue
            dx, dy = central_gradient(grid[:, ids[0]], self.periodic)
            dx, dy = dx / (self.periods[0] / w), dy / (self.periods[1] / h)
            if provider == "gradient_magnitude":
                fields.append((dx.square() + dy.square() + 1e-12).sqrt())
            else:
                dv_dx, _ = central_gradient(grid[:, ids[1]], self.periodic)
                fields.append(dv_dx / (self.periods[0] / w) - dy)
        return torch.stack(fields, dim=1)

    def _filtrations(self, x):
        banks, labels, weights, metrics = [], [], [], []
        for direction in self.directions:
            sign = 1 if direction == "sublevel" else -1
            if self.weights.get("self.persistence", 0) > 0:
                for i, field in zip(self.self_ids, self.self_fields):
                    banks.append(sign * x[:, i])
                    labels.append("self.persistence")
                    weights.append(1.0)
                    metrics.append(f"self.{field}")
            if self.weights.get("mutual.persistence", 0) > 0:
                for g, group in enumerate(self.groups):
                    a = getattr(self, f"line_directions_{g}").to(x)
                    b = getattr(self, f"line_offsets_{g}").to(x)
                    for v, offset in zip(a, b):
                        banks.append(
                            (
                                (sign * x[:, group] - offset[None, :, None, None])
                                / v[None, :, None, None]
                            ).amax(1)
                        )
                        labels.append("mutual.persistence")
                        weights.append(v.min())
                        metrics.append("mutual." + "+".join(self.fields[i] for i in group))
        return torch.stack(banks), labels, weights, metrics

    def forward(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor,
        *,
        reference_cache: dict | None = None,
    ) -> FamilyResult:
        # Cache lifetime is one transactional update. The family validates the
        # original target and coordinate tensor identities/versions before reuse.
        if reference_cache is not None and "signature" not in reference_cache:
            original, version = reference_cache.setdefault(
                "reference_identity", (reference, reference._version)
            )
            if original is not reference or version != reference._version:
                raise ValueError("persistence reference cache target changed")
        prepared = None if reference_cache is None else reference_cache.get("prepared")
        if prepared is None:
            y = self._descriptor_fields(reference.detach())
            mean = y.mean((-2, -1), keepdim=True)
            scale = y.std((-2, -1), keepdim=True, unbiased=False).clamp_min(self.scale_floor)
            target, _, _, _ = self._filtrations((y - mean) / scale)
            target_diagrams = cubical_diagrams(
                target.flatten(0, 1),
                periodic=self.periodic,
                dimensions=self.dimensions,
                locations=self.distance == "spatial_wasserstein",
                cacheable=True,
            )
            prepared = (mean, scale, target_diagrams, tuple(reference.shape), id(self))
            if reference_cache is not None:
                reference_cache["prepared"] = prepared
        mean, scale, target_diagrams, shape, owner = prepared
        if shape != tuple(reference.shape) or owner != id(self):
            raise ValueError("persistence reference cache belongs to another objective or shape")
        x = (self._descriptor_fields(generated) - mean) / scale
        bank, labels, line_weights, metric_labels = self._filtrations(x)
        n, batch, h, w = bank.shape
        diagrams = cubical_diagrams(
            bank.flatten(0, 1),
            periodic=self.periodic,
            dimensions=self.dimensions,
            locations=self.distance == "spatial_wasserstein",
        )
        compare = (
            sliced_diagram_distance
            if self.distance == "sliced_wasserstein"
            else spatial_diagram_distance
        )
        distance_kwargs = (
            {"projections": self.projections}
            if self.distance == "sliced_wasserstein"
            else {
                "periodic": self.periodic,
                "spatial_weight": self.spatial_weight,
                "spatial_mode": self.spatial_mode,
                "max_assignment_size": self.max_assignment_size,
            }
        )
        terms = {name: [] for name in self.weights if self.weights[name] > 0}
        by_dimension = {f"{name}.h{d}": [] for name in terms for d in self.dimensions}
        batched_costs = None
        if self.distance == "sliced_wasserstein":
            batched_costs = sliced_diagram_distances(
                [item[d] for item in diagrams for d in self.dimensions],
                [item[d] for item in target_diagrams for d in self.dimensions],
                projections=self.projections,
                essential_weight=self.essential_weight,
                normalization=h * w,
            ).reshape(n, batch, len(self.dimensions))
        for i, (label, weight) in enumerate(zip(labels, line_weights)):
            dim_costs = []
            for dim_index, dim in enumerate(self.dimensions):
                cost = (
                    batched_costs[i, :, dim_index]
                    if batched_costs is not None
                    else torch.stack(
                        [
                            compare(
                                diagrams[i * batch + j][dim],
                                target_diagrams[i * batch + j][dim],
                                essential_weight=self.essential_weight,
                                normalization=h * w,
                                **distance_kwargs,
                            )
                            for j in range(batch)
                        ]
                    )
                ) * weight
                by_dimension[f"{label}.h{dim}"].append(cost)
                by_dimension.setdefault(f"{metric_labels[i]}.h{dim}", []).append(cost)
                dim_costs.append(cost)
            terms[label].append(torch.stack(dim_costs).mean(0))
        results, total = {}, x.sum((1, 2, 3)) * 0
        for name, costs in terms.items():
            value = torch.stack(costs).mean(0)
            results[f"topology.{name}"] = TermResult(value, value.mean())
            total = total + self.weights[name] * value
        for name, costs in by_dimension.items():
            value = torch.stack(costs).mean(0)
            results[f"topology.{name}"] = TermResult(
                value, value.mean(), diagnostics={"evaluation_only": True}
            )
        return FamilyResult(
            results,
            total,
            total.mean(),
            diagnostics={
                "backend": (
                    "gudhi_cpu_pairing_torch_device_distance"
                    if self.distance == "sliced_wasserstein"
                    else "gudhi_cpu_pairing_scipy_cpu_assignment_torch_device_cost"
                ),
                "scalar_filtrations": n,
                "diagram_projections": self.projections
                if self.distance == "sliced_wasserstein"
                else None,
                "diagram_assignment": "exact_linear_sum_assignment"
                if self.distance == "spatial_wasserstein"
                else None,
                "finite_slice_approximation": True,
                "essential_classes_included": True,
                "point_normalization": h * w,
                "descriptor_fields": self.fields,
                "self_descriptor_fields": self.self_fields,
                "diagram_distance": self.distance,
                "spatial_mode": self.spatial_mode
                if self.distance == "spatial_wasserstein"
                else None,
                "spatial_weight": self.spatial_weight
                if self.distance == "spatial_wasserstein"
                else 0.0,
                "spatial_scope": "finite_creators_only"
                if self.distance == "spatial_wasserstein"
                else None,
            },
        )
