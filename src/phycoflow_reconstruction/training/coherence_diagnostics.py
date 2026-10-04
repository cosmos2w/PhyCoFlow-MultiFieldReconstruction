"""Small, non-mutating R3 component and cached-estimator diagnostics.

The caller owns reconstruction and its fixed TRAIN panel. No model forwards,
dataset access, optimization, or persistence recomputation occurs here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import torch

from phycoflow_reconstruction.coherence.families.cross_spectrum.covariance_blocks import (
    second_order_covariance_block_losses,
)


def _gradient(loss, parameters):
    if not loss.requires_grad:
        return [None] * len(parameters)
    return torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)


def _norm(gradients):
    return math.sqrt(
        sum(
            float(value.detach().abs().double().square().sum())
            for value in gradients
            if value is not None
        )
    )


def component_gradient_diagnostics(
    results: Mapping,
    families: Mapping,
    parameters,
    *,
    split="train",
    representation="training_raster",
    batch_index=0,
    sample_ids=(),
    source_checkpoint_sha256=None,
    family_scales=None,
    native_loss=None,
    endpoint_loss=None,
    native_pressure=None,
    endpoint_pressure=None,
    abc_direction=None,
    coherence_direction_scale=1.0,
):
    """Inspect live component graphs while preserving subsequent backward.

    Norms refer to model parameters, never field derivatives. Raw and weighted
    component contributions are both explicit. ``abc_direction`` may supply
    the actual ConFIG direction for the norm-share denominator; otherwise the
    existing combiner computes that direction without assigning gradients.
    """
    if split != "train" or representation != "training_raster":
        raise ValueError("R3 calibration gradients require TRAIN training_raster")
    if not 0 <= batch_index < 8:
        raise ValueError("R3 gradient diagnostic permits at most eight fixed batches")
    parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not parameters:
        raise ValueError("R3 diagnostics require trainable model parameters")
    if not math.isfinite(coherence_direction_scale) or coherence_direction_scale <= 0:
        raise ValueError("coherence direction scale must be finite and positive")
    scales = family_scales or {}
    records = {}
    losses = {}

    def record(name, raw, weight=1.0, source_mean=None, gradient=True):
        raw = raw.mean()
        weighted = weight * raw
        raw_norm = _norm(_gradient(raw, parameters)) if gradient else None
        weighted_norm = abs(weight) * raw_norm if gradient else None
        records[name] = {
            "raw_value": float(raw.detach()),
            "weight": float(weight),
            "weighted_value": float(weighted.detach()),
            "raw_parameter_gradient_norm": raw_norm,
            "weighted_parameter_gradient_norm": weighted_norm,
            "source_mean": source_mean,
            "source_ratio": float(raw.detach()) / source_mean
            if source_mean is not None and source_mean > 0
            else None,
        }
        losses[name] = weighted

    for name, result in results.items():
        family = families[name]
        outer = float(scales.get(name, 1.0)) * float(getattr(family, "family_weight", 1.0))
        record(f"{name}.family", result.scalar_loss, outer)
        component_meta = result.diagnostics.get("components", {})
        if name in {"global_distribution", "cross_spectrum"}:
            for path, term in result.component_results.items():
                if not term.available or term.diagnostics.get("evaluation_only"):
                    continue
                weight = component_meta.get(path, {}).get("weight")
                if weight is None:
                    keys = getattr(family, "component_weights", {})
                    weight = next((value for key, value in keys.items() if f".{key}" in path), 1.0)
                record(path, term.scalar_loss, outer * float(weight))
        elif name == "topology":
            objective = getattr(family, "spatial_objective", family)
            calibration = getattr(objective, "source_calibration", None)
            groups = objective.reporting_group_weights()
            zero = result.scalar_loss * 0
            finite_raw, essential_raw = zero, zero
            finite_normalized, essential_normalized = zero, zero
            for group, weight in groups.items():
                for part in ("finite", "essential"):
                    path = f"topology.{group}.{part}"
                    if path not in result.component_results:
                        continue
                    value = result.component_results[path].scalar_loss
                    mean = calibration["source_means"][part][group] if calibration else None
                    record(path, value, weight, mean)
                    if part == "finite":
                        finite_raw = finite_raw + weight * value
                    else:
                        essential_raw = essential_raw + weight * value
                    if calibration:
                        scaled = weight * value / calibration["scales"][part][group]
                        if part == "finite":
                            finite_normalized = finite_normalized + scaled
                        else:
                            essential_normalized = essential_normalized + scaled
            historical_weight = sum(objective.weights.values())
            finite_raw = finite_raw * historical_weight
            essential_raw = essential_raw * historical_weight
            record("topology.finite_raw", finite_raw)
            record("topology.essential_raw", essential_raw)
            record("topology.legacy_finite_contribution", finite_raw, outer)
            record(
                "topology.legacy_essential_contribution",
                essential_raw,
                outer * objective.essential_weight,
            )
            if calibration:
                record("topology.finite_primary_finite", finite_normalized, outer * 0.9)
                record("topology.finite_primary_essential", essential_normalized, outer * 0.1)
    for name, loss in (
        ("native.raw", native_loss),
        ("endpoint.raw", endpoint_loss),
        ("native.pressure", native_pressure),
        ("endpoint.pressure", endpoint_pressure),
    ):
        if loss is not None:
            record(name, loss)
    if abc_direction is None:
        from .gradient_balance import _flat_gradient, combine_coherence_gradients

        gradients = {
            name: _flat_gradient(losses[f"{name}.family"], list(parameters)) for name in results
        }
        direction, combiner = combine_coherence_gradients(gradients, method="config")
        denominator = (
            float(torch.linalg.vector_norm(direction.double())) * coherence_direction_scale
        )
        denominator_role = "existing_ConFIG_primary_direction"
    else:
        denominator = _norm(
            (abc_direction,) if isinstance(abc_direction, torch.Tensor) else abc_direction
        )
        denominator_role = "provided_ConFIG_primary_direction"
        combiner = None
    for item in records.values():
        item["norm_relative_to_ABC"] = (
            item["weighted_parameter_gradient_norm"] / denominator
            if denominator > 0 and item["weighted_parameter_gradient_norm"] is not None
            else None
        )
    return {
        "schema": "phycoflow.upgrade_1002_r3_component_gradients.v1",
        "split": split,
        "representation": representation,
        "source_weight_selection": "live",
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "batch_index": batch_index,
        "sample_ids": list(sample_ids),
        "ABC_norm": denominator,
        "ABC_norm_role": denominator_role,
        "coherence_direction_scale": coherence_direction_scale,
        "combiner": combiner,
        "components": records,
        "updates_parameters": False,
        "mutates_parameter_grad": False,
        "retains_graph": True,
    }


def pressure_error_decomposition(prediction, target):
    """Original pressure MSE = offset squared + centered error MSE."""
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    if error.ndim < 1 or not np.isfinite(error).all():
        raise ValueError("pressure decomposition requires finite matched arrays")
    error = error.reshape(error.shape[0], -1)
    means = error.mean(axis=1)
    mse = (error * error).mean(axis=1)
    centered = ((error - means[:, None]) ** 2).mean(axis=1)
    return {
        "uncorrected_mse": float(mse.mean()),
        "mean_error_squared": float((means * means).mean()),
        "centered_error_mse": float(centered.mean()),
        "identity_absolute_error": float(np.max(np.abs(mse - means * means - centered))),
        "metric_unchanged": True,
        "gauge_removed": False,
    }


def cached_B_regrouping(
    source, candidate, reference, family, *, seeds=tuple(range(100200, 100208))
):
    """Eight fixed grouped32 recomputations and pooled estimator from coefficients."""
    tensors = tuple(
        torch.as_tensor(value).detach().cpu() for value in (source, candidate, reference)
    )
    size = tensors[0].shape[0]
    if len(seeds) != 8 or len(set(seeds)) != 8 or size < 64 or size % 32:
        raise ValueError(
            "eight unique regroupings require >=64 cached samples in complete32 groups"
        )
    if any(value.shape != tensors[0].shape for value in tensors):
        raise ValueError("cached coefficient shapes must match exactly")

    def evaluate(values, targets):
        block = second_order_covariance_block_losses(
            values,
            targets,
            family.band_ids.cpu(),
            family.pairs,
            relative_floor=family.relative_floor,
            absolute_floor=family.absolute_floor,
            minimum_reference_band_fraction=family.minimum_reference_band_fraction,
            energy_floor_policy=family.energy_floor_policy,
            calibration_reference_energies=family.calibrated_band_energies.cpu(),
            calibration_ensemble_size=family.calibration_ensemble_size,
        )
        return float(
            family.component_weights.get("same_frequency", 0) * block.same_frequency
            + family.component_weights.get("cross_frequency", 0) * block.cross_frequency
        )

    rows = []
    for seed in seeds:
        indices = np.random.default_rng(seed).permutation(size)
        losses = [
            [evaluate(role[group], tensors[2][group]) for group in indices.reshape(-1, 32)]
            for role in tensors[:2]
        ]
        source_mean, child_mean = (float(np.mean(items)) for items in losses)
        rows.append(
            {
                "seed": seed,
                "source_mean": source_mean,
                "candidate_mean": child_mean,
                "candidate_source_ratio": child_mean / source_mean if source_mean > 0 else None,
            }
        )
    ratios = [
        item["candidate_source_ratio"]
        for item in rows
        if item["candidate_source_ratio"] is not None
    ]
    pooled = [evaluate(role, tensors[2]) for role in tensors[:2]]
    return {
        "sample_count": size,
        "group_size": 32,
        "regroupings": rows,
        "ratio_range": [min(ratios), max(ratios)] if ratios else None,
        "ratio_median": float(np.median(ratios)) if ratios else None,
        "pooled_source": pooled[0],
        "pooled_candidate": pooled[1],
        "pooled_ratio": pooled[1] / pooled[0] if pooled[0] > 0 else None,
        "interpretation": "fixed-panel grouping sensitivity; not independent-data CI",
        "regenerates_fields": False,
    }
