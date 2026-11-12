"""Explicit integration adapter for scalable selected Scale and pooled Shape."""

from __future__ import annotations

import math

import torch

from ..contracts import FamilyResult, TermResult
from .families.multiscale_spectrum.statistics import (
    piecewise_fluctuations,
    piecewise_means,
    spectral_coefficients,
)
from .families.topology.betti_curves import gaussian_blur
from .families.topology.geometry import rasterize_fields
from .scale_compute import sampled_scale


def _scale_statistics(family, fields):
    selected = family._select(fields)
    g = family._geometry_tensors(family.compute_dtype)
    fluctuations = piecewise_fluctuations(selected, g["piece_measures"], g["pieces"])
    coef = spectral_coefficients(fluctuations, g["basis"] * g["mass"][:, None])
    means = piecewise_means(selected, g["piece_measures"])
    power = torch.einsum("sk,bkc->bsc", g["filters"].square(), coef.square())
    logpower = torch.log1p(power / family.power_floor.to(power)).reshape(len(fields), -1)
    denominator = family.relation_self_scale.to(logpower) * math.sqrt(
        family.relation_self_scale.numel()
    )
    return coef, means, logpower / denominator


def _draws(context, first, count):
    supplied = tuple((context or {}).get("conditional_draws", ()))
    if len(supplied) < count - 1:
        raise ValueError("missing independent conditional draws")
    all_draws = (
        supplied
        if (context or {}).get("score_all_conditional_draws", False)
        else supplied[: count - 1]
    )
    if any(x.shape != first.shape for x in all_draws):
        raise ValueError("conditional draw shape mismatch")
    return (first, *all_draws)


def _result(blocks, weights, family):
    result = {}
    for name, value in blocks.items():
        if value is not None:
            result[f"{family}.{name}"] = TermResult(value, value.mean())
    total = sum(weights[name] * value for name, value in blocks.items() if value is not None)
    if not torch.isfinite(total).all():
        raise FloatingPointError("nonfinite scalable query objective")
    components = {
        f"{family}.{name}": {
            "weight": weights[name],
            "executed": True,
            "raw_scalar_loss": float(value.mean().detach()),
            "weighted_scalar_contribution": float((weights[name] * value.mean()).detach()),
        }
        for name, value in blocks.items()
        if value is not None
    }
    return FamilyResult(
        result,
        total,
        total.mean(),
        diagnostics={
            "definition": "round8_query_adapter_v1",
            "weights": weights,
            "components": components,
        },
    )


def score(
    name, original_family, prediction, reference, coordinates, context, sampler_batch, coefficients
):
    """Return (FamilyResult, scalar semantic blocks).

    ``coefficients`` supplies positive frozen ``signed/joint/log_power_self/rho``
    or ``self/mutual`` multipliers. Shape also requires ``shape_object`` fitted on
    the exact raster, field ordering and TRAIN transform. Family physical weights
    are applied before these multipliers. This function never draws a batch.
    """
    if name in ("multiscale_spectrum", "scale"):
        f = original_family
        if not f.fitted or not hasattr(f, "relation_extension") or not f.mean_augmented:
            raise ValueError("fitted mean-augmented relation Scale family required")
        if f.settings.almost_fair_alpha != 1 or f.settings.coherence_crps_weight != 0:
            raise ValueError("fair alpha one and relation rho configuration required")
        f._require_ready(prediction, coordinates)
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            with torch.no_grad():
                target, target_means, target_desc = _scale_statistics(f, reference)
            deterministic = (context or {}).get("scoring_mode") == "deterministic"
            draws = (
                (prediction,) if deterministic else _draws(context, prediction, f.conditional_draws)
            )
            stats = [_scale_statistics(f, x) for x in draws]
            draw_coef, draw_means, desc = (
                torch.stack([item[i] for item in stats]) for i in range(3)
            )
            blocks = sampled_scale(
                draw_coef,
                target,
                sampler_batch,
                filters=f.filters,
                scales=f.band_scale,
                mass=f.mass,
                power_floor=f.power_floor,
                band_weights=f._band_weights(f.compute_dtype, prediction.device),
                self_descriptors=desc,
                target_self_descriptor=target_desc,
                mean_kwargs={
                    "draw_piece_means": draw_means,
                    "target_piece_means": target_means,
                    "piece_weights": f.piece_measures.sum(dim=1),
                    "mean_scales": f.piece_mean_scale,
                    "mean_weight": f.settings.mean_weight,
                },
                mode="paired" if deterministic else "conditional",
            )
        score_version = blocks.pop("version")
        raw_leaves = blocks.pop("leaves")
        unit = coefficients.get("_difficulty_units", {})
        signed_unit, rho_unit = (
            float(unit.get("scale.signed", 1.0)),
            float(unit.get("scale.rho", 1.0)),
        )
        leaves = {
            "self": {i: v / signed_unit for i, v in raw_leaves["self"].items()},
            "pair": {
                i: raw_leaves["pair_signed"][i] / signed_unit
                + f.relation_extension["rho_weight"] * v / rho_unit
                for i, v in raw_leaves["pair_rho"].items()
            },
        }
        physical = {
            "signed": 1.0,
            "joint": f.settings.joint_energy_score_weight,
            "log_power_self": f.relation_extension["self_weight"],
            "rho": f.relation_extension["rho_weight"],
        }
    elif name in ("topology", "shape"):
        f = original_family
        shape = coefficients["shape_object"]
        if tuple(f.field_names) != tuple((context or {}).get("field_order", f.field_names)):
            raise ValueError("Shape field order differs from sampler")
        f._raster_map(coordinates)
        deterministic = (context or {}).get("scoring_mode") == "deterministic"
        draws = (prediction,) if deterministic else _draws(context, prediction, 2)
        if not deterministic and len(draws) != 2:
            raise ValueError("Shape adapter requires exactly two draws")

        def grid(x):
            x, _ = f._units_and_fields(x, reference)
            return gaussian_blur(
                rasterize_fields(x, f.neighbor_indices, f.neighbor_weights.to(x), f.grid_shape),
                f.smoothing_sigma,
                f.periodic,
                radius=f.smoothing_radius,
            ).float()

        with torch.autocast(device_type=prediction.device.type, enabled=False):
            with torch.no_grad():
                target = grid(reference.detach())
            blocks, leaves = shape.score(
                grid(draws[0]),
                None if deterministic else grid(draws[1]),
                target,
                sampler_batch,
                valid_mask=f.domain_mask if f.native_masked else None,
                return_leaves=True,
                mode="paired" if deterministic else "conditional",
            )
        physical = {
            "self": f.spatial_objective.weights.get("self.persistence", 0.0),
            "mutual": f.spatial_objective.weights.get("mutual.persistence", 0.0),
        }
        unit = coefficients.get("_difficulty_units", {})
        leaves = {
            "self": {
                i: physical["self"] * v / float(unit.get("shape.self", 1.0))
                for i, v in leaves["self"].items()
            },
            "pair": {
                i: physical["mutual"] * v / float(unit.get("shape.mutual", 1.0))
                for i, v in leaves["pair"].items()
            },
        }
    else:
        raise ValueError("unsupported scalable query family")
    weights = {}
    for key in blocks:
        prefix = "scale" if name in ("multiscale_spectrum", "scale") else "shape"
        value = float(coefficients.get(f"{prefix}.{key}", coefficients.get(key, 1.0)))
        if not math.isfinite(value) or value <= 0:
            raise ValueError("positive finite frozen block coefficient required")
        weights[key] = physical[key] * value
    family = "multiscale_spectrum" if name in ("multiscale_spectrum", "scale") else "topology"
    result = _result(blocks, weights, family)
    if family == "multiscale_spectrum":
        result.diagnostics["score_definition"] = score_version
        result.diagnostics["scoring_mode"] = (
            "deterministic" if deterministic else "conditional_fair"
        )
    if family == "topology":
        from .query_sampling import SHAPE_SAMPLING_VERSION
        from .shape_compute import PAIRED_VERSION as PAIRED_SHAPE_VERSION
        from .shape_compute import VERSION as SHAPE_VERSION

        result.diagnostics["score_definition"] = (
            PAIRED_SHAPE_VERSION if deterministic else SHAPE_VERSION
        )
        result.diagnostics["shape_sampling"] = SHAPE_SAMPLING_VERSION
        result.diagnostics["scoring_mode"] = (
            "deterministic" if deterministic else "conditional_fair"
        )
    semantic = {
        key: physical[key] * value.mean() for key, value in blocks.items() if value is not None
    }
    return result, semantic, leaves


class ScalableQueryAdapter:
    def __init__(self, families, shape_artifact):
        from .shape_compute import LazyPooledShape

        self.families = families
        self.shape = LazyPooledShape.from_state_dict(shape_artifact)

    def score(self, name, prediction, reference, *, coordinates, context, batch, coefficients):
        key = "multiscale_spectrum" if name in ("scale", "multiscale_spectrum") else "topology"
        config = dict(coefficients)
        if key == "topology":
            config["shape_object"] = self.shape
        result, blocks, leaves = score(
            key, self.families[key], prediction, reference, coordinates, context, batch, config
        )
        prefix = "scale" if key == "multiscale_spectrum" else "shape"
        return (
            result,
            {f"{prefix}.{k}": v for k, v in blocks.items()},
            {
                kind: {i: float(v.detach().mean()) for i, v in values.items()}
                for kind, values in leaves.items()
            },
        )


def fit_pooled_shape(
    family,
    source,
    reference,
    coordinates,
    batches,
    *,
    field_mean,
    field_scale,
    provenance,
    dimensions=(0, 1),
    projections=32,
    floors=(1e-6, 1e-6),
):
    """Fit from bounded preselected query batches on cached matched TRAIN tensors.

    ``field_mean`` and ``field_scale`` are the already fitted TRAIN raster units;
    query batches are reproducible and never require enumerating field pairs.
    """
    from .shape_compute import LazyPooledShape

    if source.shape != reference.shape or len(provenance.get("sample_ids", ())) != len(source):
        raise ValueError("TRAIN source/reference and sample IDs must align")
    family._raster_map(coordinates)

    def grid(value):
        value, _ = family._units_and_fields(value, reference)
        return gaussian_blur(
            rasterize_fields(
                value, family.neighbor_indices, family.neighbor_weights.to(value), family.grid_shape
            ),
            family.smoothing_sigma,
            family.periodic,
            radius=family.smoothing_radius,
        ).float()

    with torch.no_grad():
        a, b = grid(source), grid(reference)
        obj = LazyPooledShape(
            field_mean,
            field_scale,
            dimensions=dimensions,
            projections=projections,
            periodic=family.periodic,
            floors=floors,
        )
        for batch in batches:
            obj.accumulate(
                a, b, batch, valid_mask=family.domain_mask if family.native_masked else None
            )
    return obj.freeze(provenance=provenance)
