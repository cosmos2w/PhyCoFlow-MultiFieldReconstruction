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
from .config import MultiscaleSettings
from .losses import (
    SCORE_DEFINITION_VERSION,
    almost_fair_crps,
    band_coherence,
    joint_spectral_energy_score,
    signed_band_energy_score,
)
from .reporting import finite_values

FAMILY_NAME = "multiscale_spectrum"
ENERGY_SCORE_PATH = "descriptor_energy_score.conditional"
SIGNED_BAND_PATH = "signed_band_energy_score.conditional"
BAND_COHERENCE_PATH = "band_coherence_crps.conditional"
JOINT_ENERGY_PATH = "joint_spectral_energy_score.conditional"


@dataclass(frozen=True)
class SignedBandTerms:
    """Per-state cost and unchanged serialized component/report contracts."""

    per_sample_cost: torch.Tensor
    component_results: dict[str, TermResult]
    component_diagnostics: dict[str, dict[str, Any]]
    report: dict[str, Any]


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
) -> SignedBandTerms:
    """Signed-band score, optional fair joint score and band-coherence auxiliary."""
    results: dict[str, TermResult] = {}
    component_diagnostics: dict[str, dict[str, Any]] = {}
    draw_fields = torch.stack([prediction["band_fields"], *[item["band_fields"] for item in draws]])
    dtype = draw_fields.dtype
    score, attraction, repulsion = signed_band_energy_score(
        draw_fields,
        target["band_fields"],
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
            "conditional_draws": settings.conditional_draws,
            "almost_fair_alpha": settings.almost_fair_alpha,
            "projections": settings.projections,
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
    rho_draws = torch.stack([band_coherence(item["gram"], floor) for item in (prediction, *draws)])
    with torch.no_grad():
        rho_target = band_coherence(target["gram"], floor)
    crps = almost_fair_crps(rho_draws, rho_target, settings.almost_fair_alpha)  # [B,S,P2]
    has_pairs = crps.shape[-1] > 0
    coherence = crps.mean(dim=(1, 2)) if has_pairs else torch.zeros_like(primary)
    weight = settings.coherence_crps_weight
    count = len(settings.fields)
    pairs = torch.triu_indices(count, count, offset=1)
    path = f"{FAMILY_NAME}.{BAND_COHERENCE_PATH}"
    coherence_diagnostics = {
        "conditional_draws": settings.conditional_draws,
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
        joint, joint_attraction, joint_repulsion = joint_spectral_energy_score(
            torch.stack([item["coefficients"] for item in (prediction, *draws)]),
            target["coefficients"],
            filters=filters,
            scales=band_scale,
            mass=mass,
        )
        path = f"{FAMILY_NAME}.{JOINT_ENERGY_PATH}"
        joint_report = {
            "conditional_draws": settings.conditional_draws,
            "almost_fair_alpha": 1.0,
            "attraction": float(joint_attraction.mean().detach()),
            "repulsion": float(joint_repulsion.mean().detach()),
        }
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
        "score_definition_version": SCORE_DEFINITION_VERSION,
        "almost_fair_alpha": settings.almost_fair_alpha,
        "conditional_draws": settings.conditional_draws,
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
