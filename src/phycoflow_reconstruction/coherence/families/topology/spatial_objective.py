"""Versioned spatial objective and separate exact topology diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ....contracts import CoherenceComponentSpec, CoherenceFamilySpec, FamilyResult, TermResult
from .exact import curve_statistics, reference_levels
from .spatial import (
    anchor_spatial_cost,
    descriptor,
    mutual_spatial_cost,
    reference_standardize,
    self_spatial_costs,
)

SPATIAL_COMPONENT_KEYS = {
    "self": {
        "enabled",
        "weight",
        "dice_weight",
        "cldice_weight",
        "skeleton_sharpness",
        "skeleton_iterations",
    },
    "anchor_self": {"enabled", "weight"},
    "mutual": {
        "enabled",
        "weight",
        "carrier_field",
        "carrier_gauge",
        "detach_carrier",
        "gradient_scale",
        "lines",
        "theta_min_degrees",
        "offset_quantiles",
    },
}
ANCHOR_KEYS = {"provider", "fields", "quantiles", "sharpness"}
EVALUATION_KEYS = {"exact_betti"}


def _finite(value: Any, name: str, *, minimum: float = 0, positive: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or value < minimum or (positive and value == minimum):
        raise ValueError(
            f"topology {name} must be finite and {'>' if positive else '>='} {minimum}"
        )
    return value


def _quantiles(values: Any, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if (
        not result
        or any(not 0 < value < 1 for value in result)
        or tuple(sorted(set(result))) != result
    ):
        raise ValueError(f"topology {name} must be sorted unique quantiles in (0,1)")
    return result


def validate_spatial_config(
    config: Mapping[str, Any], field_names: tuple[str, ...] | None = None
) -> None:
    """Shared constructor/config checks; the spatial inverse requires paired truth."""
    if config.get("target_use", "paired_supervised") != "paired_supervised":
        raise ValueError("spatial_self_mutual topology requires paired_supervised targets")
    filtration = config.get("filtration", {})
    mode = filtration.get("level_mode", "reference_quantile")
    if mode not in {"physical", "reference_quantile"}:
        raise ValueError("topology filtration.level_mode is invalid")
    if mode == "physical":
        if config.get("units", "model_units") != "physical_units":
            raise ValueError("physical topology levels require physical_units")
        levels = tuple(float(value) for value in filtration.get("physical_levels", ()))
        if (
            not levels
            or any(not math.isfinite(value) for value in levels)
            or tuple(sorted(set(levels))) != levels
        ):
            raise ValueError("topology physical_levels must be finite, sorted and unique")
    elif "physical_levels" in filtration:
        raise ValueError("topology physical_levels requires level_mode=physical")
    _quantiles(filtration.get("quantiles", (0.3, 0.5, 0.7, 0.9)), "filtration.quantiles")
    _finite(filtration.get("sharpness", 12), "sharpness", positive=True)
    components = config.get("components", {})
    unknown = set(components) - SPATIAL_COMPONENT_KEYS.keys()
    if unknown:
        raise ValueError(f"unknown spatial topology components: {sorted(unknown)}")
    active = {}
    for name, allowed in SPATIAL_COMPONENT_KEYS.items():
        settings = components.get(name, {})
        if set(settings) - allowed:
            raise ValueError(
                f"unknown topology components.{name} keys: {sorted(set(settings) - allowed)}"
            )
        weight = _finite(settings.get("weight", 0.2 if name == "anchor_self" else 1), name)
        active[name] = bool(settings.get("enabled", name == "self")) and weight > 0
    if not any(active.values()):
        raise ValueError("spatial topology requires an active positive-weight component")
    self_config = components.get("self", {})
    dice = _finite(self_config.get("dice_weight", 1), "dice_weight")
    cldice = _finite(self_config.get("cldice_weight", 0.25), "cldice_weight")
    if active["self"] and dice + cldice == 0:
        raise ValueError("topology self requires a positive Dice or clDice weight")
    _finite(self_config.get("skeleton_sharpness", 16), "skeleton_sharpness", positive=True)
    iterations = self_config.get("skeleton_iterations", 8)
    if int(iterations) != iterations or iterations < 0:
        raise ValueError("topology skeleton_iterations must be a non-negative integer")
    anchor = config.get("anchor", {})
    if set(anchor) - ANCHOR_KEYS:
        raise ValueError("unknown topology.anchor keys")
    evaluation = config.get("evaluation", {})
    if set(evaluation) - EVALUATION_KEYS:
        raise ValueError("unknown topology.evaluation keys")
    if not isinstance(evaluation.get("exact_betti", True), bool):
        raise TypeError("topology evaluation.exact_betti must be boolean")
    if active["anchor_self"] or active["mutual"]:
        provider = anchor.get("provider", "raw")
        arities = {
            "raw": 1,
            "abs_channel": 1,
            "gradient_magnitude": 1,
            "vorticity": 2,
            "strain_rate": 2,
            "vector_magnitude": None,
        }
        names = tuple(anchor.get("fields", ()))
        if (
            provider not in arities
            or not names
            or len(set(names)) != len(names)
            or (arities[provider] is not None and len(names) != arities[provider])
        ):
            raise ValueError("topology anchor provider/fields have invalid arity")
        if field_names is not None and not set(names) <= set(field_names):
            raise ValueError("topology anchor fields are outside the dataset fields")
        if provider in {"vorticity", "strain_rate", "gradient_magnitude"} and (
            config.get("units", "model_units") != "physical_units"
        ):
            raise ValueError("derivative topology anchors require physical_units")
        _quantiles(
            anchor.get("quantiles", filtration.get("quantiles", (0.3, 0.5, 0.7, 0.9))),
            "anchor.quantiles",
        )
        _finite(anchor.get("sharpness", 12), "anchor.sharpness", positive=True)
    if active["mutual"]:
        mutual = components["mutual"]
        carrier = mutual.get("carrier_field")
        if not carrier or (field_names is not None and carrier not in field_names):
            raise ValueError("topology mutual requires a dataset carrier_field")
        if mutual.get("carrier_gauge", "signed") not in {"signed", "interface"}:
            raise ValueError("topology mutual carrier_gauge must be signed or interface")
        if not isinstance(mutual.get("detach_carrier", True), bool):
            raise ValueError("topology mutual detach_carrier must be boolean")
        _finite(mutual.get("gradient_scale", 0.2), "mutual.gradient_scale")
        lines = mutual.get("lines", 16)
        if int(lines) != lines or lines < 1:
            raise ValueError("topology mutual lines must be a positive integer")
        theta = float(mutual.get("theta_min_degrees", 15))
        if not 0 < theta < 45:
            raise ValueError("topology mutual theta_min_degrees must lie in (0,45)")
        offsets = _quantiles(mutual.get("offset_quantiles", (0.05, 0.95)), "offset_quantiles")
        if len(offsets) != 2:
            raise ValueError("topology mutual offset_quantiles requires two endpoints")


class SpatialTopologyObjective(nn.Module):
    """Three paired spatial terms; exact H0/H1 are no-grad diagnostics only."""

    def __init__(self, config: Mapping[str, Any], field_names: tuple[str, ...]) -> None:
        super().__init__()
        validate_spatial_config(config, field_names)
        self.config = config
        self.field_ids = tuple(
            field_names.index(name) for name in (config.get("fields") or field_names)
        )
        self.periodic = bool(config.get("geometry", {}).get("periodic", False))
        filtration = config.get("filtration", {})
        self.quantiles = tuple(filtration.get("quantiles", (0.3, 0.5, 0.7, 0.9)))
        self.directions = tuple(filtration.get("directions", ("superlevel", "sublevel")))
        self.dimensions = tuple(filtration.get("dimensions", (0, 1)))
        self.sharpness = float(filtration.get("sharpness", 12))
        self.physical_levels = (
            tuple(filtration["physical_levels"])
            if filtration.get("level_mode") == "physical"
            else None
        )
        self.exact_betti = bool(config.get("evaluation", {}).get("exact_betti", True))
        components = config.get("components", {})
        self.self_config = components.get("self", {})
        self.mutual_config = components.get("mutual", {})
        self.weights = {}
        for name in ("self", "anchor_self", "mutual"):
            settings = components.get(name, {})
            if settings.get("enabled", name == "self"):
                weight = float(settings.get("weight", 0.2 if name == "anchor_self" else 1))
                if name == "self":
                    self.weights["self.region"] = weight * float(settings.get("dice_weight", 1))
                    self.weights["self.connectivity"] = weight * float(
                        settings.get("cldice_weight", 0.25)
                    )
                else:
                    self.weights[f"{name}.spatial"] = weight
        self.weights = {name: value for name, value in self.weights.items() if value > 0}
        self.anchor_config = config.get("anchor", {})
        needs_anchor = any(
            name in self.weights for name in ("anchor_self.spatial", "mutual.spatial")
        )
        self.anchor_ids = (
            tuple(field_names.index(name) for name in self.anchor_config.get("fields", ()))
            if needs_anchor
            else ()
        )
        self.anchor_quantiles = tuple(self.anchor_config.get("quantiles", self.quantiles))
        self.anchor_sharpness = float(self.anchor_config.get("sharpness", 12))
        carrier_name = self.mutual_config.get("carrier_field")
        self.carrier_id = (
            field_names.index(carrier_name) if "mutual.spatial" in self.weights else None
        )
        self.metric_groups = set()
        if any(name.startswith("self.") for name in self.weights):
            self.metric_groups.add("self")
        if "anchor_self.spatial" in self.weights:
            self.metric_groups.add("anchor_self")
        if "mutual.spatial" in self.weights:
            self.metric_groups.add("mutual")
        units = str(config.get("units", "model_units"))
        specs = [
            CoherenceComponentSpec(
                name,
                "paired_supervised",
                units,
                True,
                required_geometry="fixed_2d_raster",
                metadata={"objective": "spatial_surrogate", "weight": weight},
            )
            for name, weight in self.weights.items()
        ]
        if self.exact_betti:
            for group in sorted(self.metric_groups):
                for dimension in self.dimensions:
                    name = f"{group}.h{dimension}_nmae"
                    self.weights[name] = 0.0
                    specs.append(
                        CoherenceComponentSpec(
                            name,
                            "paired_supervised",
                            units,
                            False,
                            required_geometry="fixed_2d_raster",
                            metadata={"evaluation_only": True, "weight": 0.0},
                        )
                    )
        self.spec = CoherenceFamilySpec(
            "topology",
            "2",
            tuple(specs),
            metadata={
                "aggregation": "per_sample",
                "strategy": "spatial_self_mutual",
                "exact_counts_are_objectives": False,
            },
        )
        self.spec.validate()

    def _mutual_curves(
        self, cg: torch.Tensor, ag: torch.Tensor, ct: torch.Tensor, at: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # Match the exact validation axis map in PhyCoFlow_dev, including epsilon.
        cg, ct, _ = reference_standardize(cg, ct, unbiased=True, scale_epsilon=1e-8)
        ag, at, _ = reference_standardize(ag, at, unbiased=True, scale_epsilon=1e-8)
        settings = self.mutual_config
        count = int(settings.get("lines", 16))
        # Fixed full bank only in evaluation: no stochastic state or training CPU pairing.
        bank = torch.quasirandom.SobolEngine(2, scramble=False).draw(count + 1)[1:].to(ct)
        minimum = ct.new_tensor(float(settings.get("theta_min_degrees", 15)) * math.pi / 180)
        theta = minimum + (math.pi / 2 - minimum - minimum) * bank[:, 0]
        cos, sin = theta.cos(), theta.sin()
        speed = torch.minimum(cos, sin)
        lower, upper = settings.get("offset_quantiles", (0.05, 0.95))
        qs = lower + (upper - lower) * bank[:, 1]
        qt = lower + (upper - lower) * (1 - bank[:, 1])
        pushes = []
        for direction in self.directions:
            sign = 1 if direction == "superlevel" else -1
            left_offsets = torch.quantile((sign * ct).flatten(1), qs, dim=1).T
            right_offsets = torch.quantile(at.flatten(1), qt, dim=1).T
            for index in range(count):
                left, right = (
                    left_offsets[:, index, None, None],
                    right_offsets[:, index, None, None],
                )
                pred = torch.minimum(
                    (sign * cg - left) / (cos[index] / speed[index]),
                    (ag - right) / (sin[index] / speed[index]),
                )
                ref = torch.minimum(
                    (sign * ct - left) / (cos[index] / speed[index]),
                    (at - right) / (sin[index] / speed[index]),
                )
                pushes.append((pred, ref))
        # Transfer the entire bank together, as in development validation.
        # Computing host quantiles inside the line loop would synchronize CUDA
        # once per line and direction before doing the actual count work.
        predicted, reference = zip(*pushes)
        host = torch.stack((torch.cat(predicted), torch.cat(reference))).detach().cpu()
        levels = reference_levels(host[1], self.anchor_quantiles)
        batch_size = cg.shape[0]
        pairs = list(
            zip(host[0].split(batch_size), host[1].split(batch_size), levels.split(batch_size))
        )
        return curve_statistics(pairs, periodic=self.periodic)

    def forward(self, generated: torch.Tensor, reference: torch.Tensor) -> FamilyResult:
        reference = reference.detach()
        results = {}
        batch_size = generated.shape[0]
        per_sample = generated.sum((1, 2, 3)) * 0

        def record(
            name: str,
            cost: torch.Tensor,
            valid: torch.Tensor | None = None,
            gradient_scale: float = 1,
            statistics: dict | None = None,
        ) -> None:
            nonlocal per_sample
            if valid is None:
                valid = torch.ones(batch_size, device=cost.device, dtype=torch.bool)
            weight = self.weights.get(name, 0.0)
            valid_count = int(valid.sum())
            # Missing references are disclosed, never presented as zero-error successes.
            masked = torch.where(valid, cost, torch.zeros_like(cost))
            scalar = masked.sum() / max(valid_count, 1)
            if statistics is not None:
                scalar = cost.new_tensor(statistics["error_mass"]) / max(
                    statistics["reference_mass"], 1
                )
            results[f"topology.{name}"] = TermResult(
                masked,
                scalar,
                valid_mask=valid,
                reason=None if valid_count else "no nonconstant reference descriptor",
                diagnostics={
                    "weight": weight,
                    "gradient_scale": gradient_scale,
                    "evaluation_only": weight == 0,
                    "valid_count": valid_count,
                    "valid_fraction": valid_count / batch_size,
                    **(statistics or {}),
                },
            )
            if weight > 0:
                # Match the development valid-sample mean while retaining additive per-sample costs.
                applied = masked.detach() + gradient_scale * (masked - masked.detach())
                per_sample = per_sample + weight * applied * batch_size / max(valid_count, 1)

        def record_curves(
            group: str, stats: dict[str, torch.Tensor], valid: torch.Tensor | None = None
        ) -> None:
            if valid is None:
                valid = torch.ones(batch_size, dtype=torch.bool, device=generated.device)
            for dimension in self.dimensions:
                error, mass = (
                    stats["error_mass"][:, dimension],
                    stats["reference_mass"][:, dimension],
                )
                if error.shape[0] == batch_size:
                    error, mass = error[valid.cpu()], mass[valid.cpu()]
                cost = generated.new_zeros(batch_size)
                cost[valid] = (error / mass.clamp_min(1)).to(cost)
                diagnostics = {
                    "aggregation": "pooled_error_over_reference_mass",
                    "error_mass": int(error.sum()),
                    "reference_mass": int(mass.sum()),
                    "per_valid_sample_error_mass": error.tolist(),
                    "per_valid_sample_reference_mass": mass.tolist(),
                    "threshold_oracle": "numpy_float64_quantile",
                }
                if group == "mutual":
                    lines = int(self.mutual_config.get("lines", 16))
                    # Slices are ordered by direction then bank index. Pool samples
                    # and directions first, retaining one numerator/denominator per line.
                    line_error = (
                        stats["slice_error_mass"][..., dimension]
                        .reshape(-1, len(self.directions), lines)
                        .sum((0, 1))
                    )
                    line_mass = (
                        stats["slice_reference_mass"][..., dimension]
                        .reshape(-1, len(self.directions), lines)
                        .sum((0, 1))
                    )
                    line_ratio = line_error.double() / line_mass.clamp_min(1)
                    diagnostics.update(
                        {
                            "line_error_mass": line_error.tolist(),
                            "line_reference_mass": line_mass.tolist(),
                            "line_max_nmae": float(line_ratio.max()),
                            "line_p90_nmae": float(torch.quantile(line_ratio, 0.9)),
                            "lines_evaluated": lines,
                        }
                    )
                record(f"{group}.h{dimension}_nmae", cost, valid, statistics=diagnostics)

        if "self" in self.metric_groups:
            pred, ref = generated[:, self.field_ids], reference[:, self.field_ids]
            region, connectivity = self_spatial_costs(
                pred,
                ref,
                directions=self.directions,
                quantiles=self.quantiles,
                physical_levels=self.physical_levels,
                sharpness=self.sharpness,
                skeleton_sharpness=float(self.self_config.get("skeleton_sharpness", 16)),
                skeleton_iterations=int(self.self_config.get("skeleton_iterations", 8)),
                periodic=self.periodic,
                compute_connectivity="self.connectivity" in self.weights,
            )
            for name, cost in (("self.region", region), ("self.connectivity", connectivity)):
                if name in self.weights:
                    record(name, cost)
            if self.exact_betti and not torch.is_grad_enabled():
                pairs = []
                for p, r in zip(pred.unbind(1), ref.unbind(1)):
                    levels = (
                        torch.tensor(self.physical_levels, dtype=torch.float64).expand(
                            batch_size, -1
                        )
                        if self.physical_levels is not None
                        else reference_levels(r, self.quantiles)
                    )
                    for direction in self.directions:
                        sign = 1 if direction == "superlevel" else -1
                        pairs.append((sign * p, sign * r, sign * levels))
                record_curves("self", curve_statistics(pairs, periodic=self.periodic))
        if self.metric_groups & {"anchor_self", "mutual"}:
            provider = str(self.anchor_config.get("provider", "raw"))
            ag = descriptor(generated, provider, self.anchor_ids, self.periodic)
            at = descriptor(reference, provider, self.anchor_ids, self.periodic)
            if "anchor_self" in self.metric_groups:
                cost, valid = anchor_spatial_cost(
                    ag,
                    at,
                    quantiles=self.anchor_quantiles,
                    sharpness=self.anchor_sharpness,
                )
                record("anchor_self.spatial", cost, valid)
                if self.exact_betti and not torch.is_grad_enabled():
                    pairs = [(ag, at, reference_levels(at, self.anchor_quantiles))]
                    record_curves(
                        "anchor_self", curve_statistics(pairs, periodic=self.periodic), valid
                    )
            if "mutual" in self.metric_groups:
                gauge = self.mutual_config.get("carrier_gauge", "signed")
                carrier_provider = "gradient_magnitude" if gauge == "interface" else "raw"
                cg = descriptor(generated, carrier_provider, (self.carrier_id,), self.periodic)
                ct = descriptor(reference, carrier_provider, (self.carrier_id,), self.periodic)
                cost, valid = mutual_spatial_cost(
                    cg,
                    ag,
                    ct,
                    at,
                    quantiles=self.anchor_quantiles,
                    sharpness=self.anchor_sharpness,
                    directions=self.directions,
                    detach_carrier=bool(self.mutual_config.get("detach_carrier", True)),
                )
                if not valid.any():
                    raise ValueError("topology mutual has no nonconstant valid reference samples")
                record(
                    "mutual.spatial",
                    cost,
                    valid,
                    gradient_scale=float(self.mutual_config.get("gradient_scale", 0.2)),
                )
                if self.exact_betti and not torch.is_grad_enabled():
                    record_curves(
                        "mutual",
                        self._mutual_curves(cg[valid], ag[valid], ct[valid], at[valid]),
                        valid,
                    )
        if not torch.isfinite(per_sample).all():
            raise FloatingPointError("topology spatial objective produced a non-finite cost")
        return FamilyResult(
            results,
            per_sample,
            per_sample.mean(),
            diagnostics={
                "strategy": "spatial_self_mutual",
                "exact_counts_are_objectives": False,
                "exact_counts_evaluated": self.exact_betti and not torch.is_grad_enabled(),
                "threshold_rule": "retain_reference_threshold_bank",
                "derivative_units": "grid_index",
                "anchor_provider": self.anchor_config.get("provider"),
                "mutual_carrier_detached": bool(self.mutual_config.get("detach_carrier", True)),
            },
        )
