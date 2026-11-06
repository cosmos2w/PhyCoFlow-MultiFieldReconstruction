"""Finite, explicitly labelled spectral calibration diagnostics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import torch

from .features import DescriptorLayout


def positive_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Undefined where the denominator is nonpositive or either input is nonfinite.

    Debiased error estimates can be negative. Clamping them to a tiny positive
    number would turn an undefined calibration ratio into an enormous fake value.
    """
    valid = torch.isfinite(numerator) & torch.isfinite(denominator) & (denominator > 0)
    safe_denominator = torch.where(valid, denominator, torch.ones_like(denominator))
    return torch.where(valid, numerator / safe_denominator, torch.nan)


def finite_values(value: Any) -> Any:
    """Convert tensors/lists to JSON data, representing undefined numbers as null."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: finite_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_values(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def band_ensemble_report(
    draw_fields: Iterable[torch.Tensor],
    truth: torch.Tensor,
    *,
    mass: torch.Tensor,
    count: int,
    band_names: tuple[str, ...],
    field_names: tuple[str, ...],
) -> dict[str, Any]:
    """Per band and field: member, shared (debiased mean) and varying power vs the truth.

    With K independent draws F_k of the band field and the truth F_y, all in the
    mass inner product: member power = mean_k <F_k,F_k>; the unbiased spread
    V = sum_k <F_k - Fbar, F_k - Fbar> / (K - 1); shared power = <Fbar,Fbar> - V/K;
    mean error E = <Fbar - F_y, Fbar - F_y> - V/K; spread-skill = V / E; and the
    correlation of the K-draw mean with the truth. The plain keys average the per-state
    ratios (dominated by quiet states); the ``_pooled`` keys are ratios of state means, the
    conventional pooled calibration numbers that the offline yardstick reports.
    """
    with torch.no_grad():
        q = mass / mass.sum()

        def power(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return torch.einsum("n,bsnc->bsc", q, left * right)

        member_power = None
        running_sum = None
        for fields in draw_fields:
            member_power = (
                power(fields, fields)
                if member_power is None
                else member_power + power(fields, fields)
            )
            running_sum = fields.clone() if running_sum is None else running_sum + fields
        member_power = member_power / count
        mean = running_sum / count
        mean_power = power(mean, mean)
        spread = ((count * (member_power - mean_power)) / (count - 1)).clamp_min(0)
        shared = mean_power - spread / count
        truth_power = power(truth, truth)
        mean_error = power(mean - truth, mean - truth) - spread / count
        correlation = positive_ratio(power(mean, truth), (mean_power * truth_power).sqrt())
        variance_ratio = positive_ratio(spread, mean_error)

        def summary(value: torch.Tensor) -> list:
            return finite_values(value.nanmean(dim=0).detach().cpu())

        def pooled(numerator: torch.Tensor, denominator: torch.Tensor) -> list:
            # Ratio of state means (the conventional pooled spread-skill); the per-state
            # mean of ratios above is dominated by quiet states whose band power is tiny.
            return finite_values(
                positive_ratio(numerator.mean(dim=0), denominator.mean(dim=0)).detach().cpu()
            )

        def median(value: torch.Tensor) -> list:
            return finite_values(value.nanmedian(dim=0).values.detach().cpu())

        return {
            "members": count,
            "band_names": list(band_names),
            "fields": list(field_names),
            "spread_skill_definition": "variance_over_debiased_mean_error (legacy alias)",
            "member_power_over_truth": summary(positive_ratio(member_power, truth_power)),
            "shared_power_over_truth": summary(positive_ratio(shared, truth_power)),
            "varying_power_over_truth": summary(positive_ratio(spread, truth_power)),
            "shared_power_over_rho2_truth": summary(
                positive_ratio(shared, correlation.square() * truth_power)
            ),
            "spread_skill": summary(variance_ratio),
            "variance_over_debiased_error": summary(variance_ratio),
            "variance_ratio_valid_states": torch.isfinite(variance_ratio).sum(dim=0).tolist(),
            "mean_correlation": summary(correlation),
            "mean_error_over_truth": summary(positive_ratio(mean_error, truth_power)),
            "member_power_over_truth_pooled": pooled(member_power, truth_power),
            "shared_power_over_truth_pooled": pooled(shared, truth_power),
            "varying_power_over_truth_pooled": pooled(spread, truth_power),
            "spread_skill_pooled": pooled(spread, mean_error),
            "variance_over_debiased_error_pooled": pooled(spread, mean_error),
            "rms_spread_over_rmse_pooled": finite_values(
                positive_ratio(spread.mean(dim=0), mean_error.mean(dim=0)).sqrt().cpu()
            ),
            "spread_skill_median": median(variance_ratio),
        }


def spectral_report(
    prediction: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    prediction_blocks: Mapping[str, torch.Tensor],
    target_blocks: Mapping[str, torch.Tensor],
    *,
    power_floor: torch.Tensor,
    field_names: tuple[str, ...],
    band_names: tuple[str, ...],
    layout: DescriptorLayout,
) -> dict[str, Any]:
    """Interpretable panel summaries: power curves, signed coherence, descriptor errors."""
    with torch.no_grad():
        floor = power_floor

        def summary(statistics: Mapping[str, torch.Tensor]) -> dict[str, Any]:
            gram = statistics["gram"].detach()
            power = torch.diagonal(gram, dim1=-2, dim2=-1)
            total = statistics["fluctuation_power"].detach()
            unresolved = torch.where(
                total > 0, 1.0 - power.sum(dim=1) / total.clamp_min(1e-300), 0.0
            )
            left, right = torch.triu_indices(power.shape[-1], power.shape[-1], offset=1)
            valid = (power[:, :, left] > floor[:, left]) & (power[:, :, right] > floor[:, right])
            rho = (
                gram[:, :, left, right]
                / (power[:, :, left] * power[:, :, right]).clamp_min(1e-300).sqrt()
            )
            counts = valid.sum(dim=0)
            mean_rho = torch.where(
                counts > 0,
                torch.where(valid, rho, 0.0).sum(dim=0) / counts.clamp_min(1),
                torch.nan,
            )
            return {
                "band_power_mean": finite_values(power.mean(dim=0).cpu()),
                "fluctuation_power_mean": finite_values(total.mean(dim=0).cpu()),
                "unresolved_fraction_mean": finite_values(unresolved.mean(dim=0).cpu()),
                "normalized_band_coherence_mean": finite_values(mean_rho.cpu()),
                "normalized_band_coherence_valid_fraction": finite_values(
                    valid.to(power.dtype).mean(dim=0).cpu()
                ),
            }

        pairs = torch.triu_indices(len(field_names), len(field_names), offset=1)
        blocks = {}
        for key, names in zip(layout.keys, layout.names):
            difference = (prediction_blocks[key] - target_blocks[key]).detach()
            blocks[key] = {
                "names": list(names),
                "reference_mean": finite_values(target_blocks[key].mean(dim=0).cpu()),
                "prediction_mean": finite_values(prediction_blocks[key].detach().mean(dim=0).cpu()),
                "mean_signed_error": finite_values(difference.mean(dim=0).cpu()),
                "rms_error": finite_values(difference.square().mean(dim=0).sqrt().cpu()),
            }
        return {
            "states": int(prediction["gram"].shape[0]),
            "fields": list(field_names),
            "bands": list(band_names),
            "coherence_pairs": [
                [field_names[int(i)], field_names[int(j)]] for i, j in zip(pairs[0], pairs[1])
            ],
            "reference": summary(target),
            "prediction": summary(prediction),
            "descriptor_blocks": blocks,
        }
