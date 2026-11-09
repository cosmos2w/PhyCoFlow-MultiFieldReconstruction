"""Compose signed-band, joint and optional coherence scores into family terms.

The tensor kernels live in losses.py. This module assigns stable component names,
weights and diagnostics; it neither fits geometry nor draws reconstructions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from ....contracts import TermResult
from .config import MEAN_AUGMENTED_SCORE_VERSION, MultiscaleSettings
from .losses import (
    almost_fair_crps,
    joint_spectral_energy_score,
    paired_joint_spectral_mean_distance,
    paired_signed_band_distance,
    selected_band_coherence,
    signed_band_coefficient_energy_score,
)
from .reporting import finite_values

FAMILY_NAME = "multiscale_spectrum"
ENERGY_SCORE_PATH = "descriptor_energy_score.conditional"
SIGNED_BAND_PATH = "signed_band_energy_score.conditional"
BAND_COHERENCE_PATH = "band_coherence_crps.conditional"
JOINT_ENERGY_PATH = "joint_spectral_energy_score.conditional"
MEAN_JOINT_ENERGY_PATH = "joint_spectral_mean_energy_score.conditional"
PAIRED_BAND_PATH = "paired_signed_band_distance.paired"
PAIRED_MEAN_JOINT_PATH = "paired_joint_spectral_mean_distance.paired"


@dataclass(frozen=True)
class SignedBandTerms:
    """Per-state cost and unchanged serialized component/report contracts."""

    per_sample_cost: torch.Tensor
    component_results: dict[str, TermResult]
    component_diagnostics: dict[str, dict[str, Any]]
    report: dict[str, Any]


def paired_signed_band_terms(
    prediction: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    *,
    settings: MultiscaleSettings,
    mass: torch.Tensor,
    filters: torch.Tensor,
    band_scale: torch.Tensor,
    projections: torch.Tensor,
    band_weights: torch.Tensor,
    piece_measures: torch.Tensor,
    mean_scale: torch.Tensor,
) -> SignedBandTerms:
    """One paired point and target, with no distributional or repulsion claim."""
    per_band = paired_signed_band_distance(
        prediction["coefficients"],
        target["coefficients"],
        filters=filters,
        mass=mass,
        scales=band_scale,
        projections=projections,
    )
    primary = (per_band * band_weights).sum(dim=1)
    joint = paired_joint_spectral_mean_distance(
        prediction["coefficients"],
        target["coefficients"],
        filters=filters,
        scales=band_scale,
        mass=mass,
        prediction_piece_means=prediction["piece_means"],
        target_piece_means=target["piece_means"],
        piece_weights=piece_measures.sum(dim=1),
        mean_scales=mean_scale,
        mean_weight=settings.mean_weight,
    )
    primary_path = f"{FAMILY_NAME}.{PAIRED_BAND_PATH}"
    joint_path = f"{FAMILY_NAME}.{PAIRED_MEAN_JOINT_PATH}"
    mean_error = prediction["piece_means"] - target["piece_means"]
    primary_diagnostics = {
        "per_band": finite_values(per_band.mean(dim=0).detach().cpu()),
        "band_weights": band_weights.tolist(),
        "projections": settings.projections,
        "projection_pairs": (
            None
            if settings.projection_pairs is None
            else [
                [settings.fields[left], settings.fields[right]]
                for left, right in settings.projection_pairs
            ]
        ),
    }
    joint_diagnostics = {
        "mean_weight": settings.mean_weight,
        "piece_mass_fractions": finite_values(
            (piece_measures.sum(dim=1) / piece_measures.sum()).detach().cpu()
        ),
        "mean_scale": finite_values(mean_scale.detach().cpu()),
        "mean_bias_by_piece_field": finite_values(mean_error.mean(dim=0).detach().cpu()),
        "mean_absolute_error_by_piece_field": finite_values(
            mean_error.abs().mean(dim=0).detach().cpu()
        ),
        "mean_error_units": settings.units,
    }
    results = {
        primary_path: TermResult(primary, primary.mean(), diagnostics=primary_diagnostics),
        joint_path: TermResult(joint, joint.mean(), diagnostics=joint_diagnostics),
    }
    component_diagnostics = {}
    for path, loss, weight in (
        (primary_path, primary, 1.0),
        (joint_path, joint, settings.joint_distance_weight),
    ):
        raw = float(loss.mean().detach())
        component_diagnostics[path] = {
            "weight": weight,
            "executed": True,
            "raw_scalar_loss": raw,
            "weighted_scalar_contribution": weight * raw,
        }
    report = {
        "score_definition_version": settings.score_definition_version,
        "interpretation": "paired_point_distance",
        "spatial_weighting": "normalized_analysis_mass",
        "units": settings.units,
        "band_names": list(settings.bands.names),
        "distance_per_band": primary_diagnostics["per_band"],
        "joint_distance_weight": settings.joint_distance_weight,
        "joint_distance": joint_diagnostics,
    }
    return SignedBandTerms(
        primary + settings.joint_distance_weight * joint, results, component_diagnostics, report
    )


def signed_band_terms(
    prediction: Mapping[str, torch.Tensor],
    draws: list[Mapping[str, torch.Tensor]],
    target: Mapping[str, torch.Tensor],
    *,
    settings: MultiscaleSettings,
    mass: torch.Tensor,
    filters: torch.Tensor,
    band_scale: torch.Tensor,
    power_floor: torch.Tensor,
    projections: torch.Tensor,
    band_weights: torch.Tensor,
    piece_measures: torch.Tensor | None = None,
    mean_scale: torch.Tensor | None = None,
) -> SignedBandTerms:
    """Signed-band score, optional fair joint score and band-coherence auxiliary."""
    results: dict[str, TermResult] = {}
    component_diagnostics: dict[str, dict[str, Any]] = {}
    draw_coefficients = torch.stack(
        [prediction["coefficients"], *[item["coefficients"] for item in draws]]
    )
    conditional_draws = len(draws) + 1
    dtype = draw_coefficients.dtype
    score, attraction, repulsion = signed_band_coefficient_energy_score(
        draw_coefficients,
        target["coefficients"],
        filters=filters,
        mass=mass.to(dtype),
        scales=band_scale.to(dtype),
        projections=projections,
        alpha=settings.almost_fair_alpha,
    )
    primary = (score * band_weights).sum(dim=1)
    path = f"{FAMILY_NAME}.{SIGNED_BAND_PATH}"
    results[path] = TermResult(
        primary,
        primary.mean(),
        diagnostics={
            "conditional_draws": conditional_draws,
            "almost_fair_alpha": settings.almost_fair_alpha,
            "projections": settings.projections,
            **(
                {
                    "projection_pairs": [
                        [settings.fields[left], settings.fields[right]]
                        for left, right in settings.projection_pairs
                    ]
                }
                if settings.projection_pairs is not None
                else {}
            ),
            "band_weights": band_weights.tolist(),
            "per_band": finite_values(score.mean(dim=0).detach().cpu()),
            "attraction_per_band": finite_values(attraction.mean(dim=0).detach().cpu()),
            "repulsion_per_band": finite_values(repulsion.mean(dim=0).detach().cpu()),
        },
    )
    raw = float(primary.mean().detach())
    component_diagnostics[path] = {
        "weight": 1.0,
        "executed": True,
        "raw_scalar_loss": raw,
        "weighted_scalar_contribution": raw,
    }
    floor = power_floor.to(dtype)
    rho_draws = torch.stack([selected_band_coherence(item, floor) for item in (prediction, *draws)])
    with torch.no_grad():
        rho_target = selected_band_coherence(target, floor)
    crps = almost_fair_crps(rho_draws, rho_target, settings.almost_fair_alpha)  # [B,S,P2]
    has_pairs = crps.shape[-1] > 0
    coherence = crps.mean(dim=(1, 2)) if has_pairs else torch.zeros_like(primary)
    weight = settings.coherence_crps_weight
    count = len(settings.fields)
    pairs = (prediction["pair_indices"].T.cpu() if "pair_indices" in prediction
             else torch.triu_indices(count, count, offset=1))
    path = f"{FAMILY_NAME}.{BAND_COHERENCE_PATH}"
    coherence_diagnostics = {
        "conditional_draws": conditional_draws,
        "almost_fair_alpha": settings.almost_fair_alpha,
        "pairs": [
            [settings.fields[int(i)], settings.fields[int(j)]] for i, j in zip(pairs[0], pairs[1])
        ],
        "available": has_pairs,
        "per_band": finite_values(crps.mean(dim=(0, 2)).detach().cpu()) if has_pairs else None,
        "per_pair": finite_values(crps.mean(dim=(0, 1)).detach().cpu()),
    }
    if weight <= 0:
        coherence = coherence.detach()
        coherence_diagnostics["evaluation_only"] = True
    results[path] = TermResult(coherence, coherence.mean(), diagnostics=coherence_diagnostics)
    raw_coherence = float(coherence.mean().detach())
    component_diagnostics[path] = {
        "weight": weight,
        "executed": has_pairs,
        "raw_scalar_loss": raw_coherence,
        "weighted_scalar_contribution": weight * raw_coherence,
    }
    per_sample = primary + weight * coherence if weight > 0 else primary
    joint_weight = settings.joint_energy_score_weight
    joint_report = None
    if joint_weight > 0:
        augmented = settings.score_definition_version == MEAN_AUGMENTED_SCORE_VERSION
        mean_args = {}
        if augmented:
            if piece_measures is None or mean_scale is None:
                raise ValueError("mean-augmented score needs frozen piece measures and scales")
            mean_args = {
                "draw_piece_means": torch.stack(
                    [item["piece_means"] for item in (prediction, *draws)]
                ),
                "target_piece_means": target["piece_means"],
                "piece_weights": piece_measures.sum(dim=1),
                "mean_scales": mean_scale,
                "mean_weight": settings.mean_weight,
            }
        joint, joint_attraction, joint_repulsion = joint_spectral_energy_score(
            torch.stack([item["coefficients"] for item in (prediction, *draws)]),
            target["coefficients"],
            filters=filters,
            scales=band_scale,
            mass=mass,
            **mean_args,
        )
        path = f"{FAMILY_NAME}.{MEAN_JOINT_ENERGY_PATH if augmented else JOINT_ENERGY_PATH}"
        joint_report = {
            "conditional_draws": conditional_draws,
            "almost_fair_alpha": 1.0,
            "attraction": float(joint_attraction.mean().detach()),
            "repulsion": float(joint_repulsion.mean().detach()),
        }
        if augmented:
            mean_error = prediction["piece_means"] - target["piece_means"]
            joint_report.update(
                {
                    "mean_weight": settings.mean_weight,
                    "piece_mass_fractions": finite_values(
                        (piece_measures.sum(dim=1) / piece_measures.sum()).detach().cpu()
                    ),
                    "mean_scale": finite_values(mean_scale.detach().cpu()),
                    "first_draw_mean_bias_by_piece_field": finite_values(
                        mean_error.mean(dim=0).detach().cpu()
                    ),
                    "first_draw_mean_absolute_error_by_piece_field": finite_values(
                        mean_error.abs().mean(dim=0).detach().cpu()
                    ),
                    "mean_error_units": settings.units,
                }
            )
        results[path] = TermResult(joint, joint.mean(), diagnostics=joint_report)
        raw_joint = float(joint.mean().detach())
        component_diagnostics[path] = {
            "weight": joint_weight,
            "executed": True,
            "raw_scalar_loss": raw_joint,
            "weighted_scalar_contribution": joint_weight * raw_joint,
        }
        per_sample = per_sample + joint_weight * joint
    report = {
        "score_definition_version": settings.score_definition_version,
        "almost_fair_alpha": settings.almost_fair_alpha,
        "conditional_draws": conditional_draws,
        "proper_for_scored_representations": settings.almost_fair_alpha == 1.0,
        "joint_identification": joint_weight > 0,
        "spatial_weighting": "normalized_analysis_mass",
        "units": settings.units,
        "band_names": list(settings.bands.names),
        "energy_score_per_band": results[f"{FAMILY_NAME}.{SIGNED_BAND_PATH}"].diagnostics[
            "per_band"
        ],
        "coherence_crps_per_band": coherence_diagnostics["per_band"],
        "coherence_crps_per_pair": coherence_diagnostics["per_pair"],
        "pairs": coherence_diagnostics["pairs"],
        "coherence_crps_weight": weight,
        "joint_energy_score_weight": joint_weight,
        "joint_energy_score": joint_report,
    }
    return SignedBandTerms(per_sample, results, component_diagnostics, report)
