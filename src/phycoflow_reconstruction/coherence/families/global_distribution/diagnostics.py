"""Named-pair occupation diagnostics, independent of training components."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _weighted_midrank(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Within-endpoint weighted midranks, assigning equal values equal rank."""
    sorted_values, order = values.sort(stable=True)
    _, group = torch.unique_consecutive(sorted_values, return_inverse=True)
    group_mass = torch.zeros(int(group[-1]) + 1, device=weights.device, dtype=weights.dtype)
    group_mass.scatter_add_(0, group, weights[order])
    centers = group_mass.cumsum(0) - 0.5 * group_mass
    result = torch.empty_like(weights)
    result[order] = centers[group]
    return result


def _histogram2d(
    left: torch.Tensor,
    right: torch.Tensor,
    weights: torch.Tensor,
    left_min: torch.Tensor,
    left_max: torch.Tensor,
    right_min: torch.Tensor,
    right_max: torch.Tensor,
    bins: int,
) -> torch.Tensor:
    # A constant axis is put in the middle bin, retaining a stable histogram.
    def locate(values: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        if bool(high == low):
            return torch.full_like(values, bins // 2, dtype=torch.long)
        return torch.floor((values - low) / (high - low) * bins).long().clamp(0, bins - 1)

    index = locate(left, left_min, left_max) * bins + locate(right, right_min, right_max)
    result = torch.zeros(bins * bins, device=weights.device, dtype=weights.dtype)
    result.scatter_add_(0, index, weights)
    return result.reshape(bins, bins)


@torch.no_grad()
def named_pair_occupations(
    generated: torch.Tensor,
    reference: torch.Tensor,
    generated_weights: torch.Tensor,
    reference_weights: torch.Tensor,
    pairs: Sequence[tuple[str, str, int, int]],
    *,
    bins: int = 16,
) -> dict[str, dict]:
    """Return weighted raw-value and within-endpoint rank occupation grids.

    Inputs have already been validated and masses normalized by the family.
    Each raw-value grid has the same endpoint range. Rank grids use each
    endpoint's own weighted midranks, so they diagnose dependence separately
    from marginal value shifts. Ties share their entire weighted midrank.
    """
    if bins < 2:
        raise ValueError("pair diagnostic bins must be >= 2")
    diagnostics = {}
    for left_name, right_name, left_id, right_id in pairs:
        raw_generated, raw_reference = [], []
        rank_generated, rank_reference, limits = [], [], []
        for sample in range(generated.shape[0]):
            left = generated[sample, :, left_id]
            right = generated[sample, :, right_id]
            ref_left = reference[sample, :, left_id]
            ref_right = reference[sample, :, right_id]
            gen_mass, ref_mass = generated_weights[sample], reference_weights[sample]
            left_both = torch.cat((left[gen_mass > 0], ref_left[ref_mass > 0]))
            right_both = torch.cat((right[gen_mass > 0], ref_right[ref_mass > 0]))
            bounds = (left_both.min(), left_both.max(), right_both.min(), right_both.max())
            raw_generated.append(_histogram2d(left, right, gen_mass, *bounds, bins))
            raw_reference.append(_histogram2d(ref_left, ref_right, ref_mass, *bounds, bins))
            rank_bounds = tuple(left.new_tensor(value) for value in (0.0, 1.0, 0.0, 1.0))
            rank_generated.append(
                _histogram2d(
                    _weighted_midrank(left, gen_mass),
                    _weighted_midrank(right, gen_mass),
                    gen_mass,
                    *rank_bounds,
                    bins,
                )
            )
            rank_reference.append(
                _histogram2d(
                    _weighted_midrank(ref_left, ref_mass),
                    _weighted_midrank(ref_right, ref_mass),
                    ref_mass,
                    *rank_bounds,
                    bins,
                )
            )
            limits.append(torch.stack(bounds))
        diagnostics[f"{left_name}::{right_name}"] = {
            "fields": (left_name, right_name),
            "bins": bins,
            "raw_value_bounds": torch.stack(limits),
            "generated_occupation": torch.stack(raw_generated),
            "reference_occupation": torch.stack(raw_reference),
            "generated_own_rank_occupation": torch.stack(rank_generated),
            "reference_own_rank_occupation": torch.stack(rank_reference),
            "rank_definition": "endpoint_weighted_midrank_ties_share_rank",
        }
    return diagnostics
