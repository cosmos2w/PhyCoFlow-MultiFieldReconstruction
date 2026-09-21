"""Paired spatial topology surrogates adapted to the common raster contract.

These terms provide spatial gradients, not differentiable Betti counts. Reference
statistics are shared by prediction and truth; derivatives use grid-index units
as in the development recipe (see README.md for geometry/units limitations).
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def central_gradient(field: torch.Tensor, periodic: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Return column/x and row/y central differences, with zero boundary derivatives."""
    if periodic:
        dx = (field.roll(-1, -1) - field.roll(1, -1)) * 0.5
        dy = (field.roll(-1, -2) - field.roll(1, -2)) * 0.5
    else:
        dx, dy = torch.zeros_like(field), torch.zeros_like(field)
        dx[..., 1:-1] = (field[..., 2:] - field[..., :-2]) * 0.5
        dy[..., 1:-1, :] = (field[..., 2:, :] - field[..., :-2, :]) * 0.5
    return dx, dy


def descriptor(
    grids: torch.Tensor, provider: str, fields: tuple[int, ...], periodic: bool
) -> torch.Tensor:
    """Build a scalar from generated or reference [B,C,H,W] fields."""
    selected = grids[:, fields]
    if provider == "raw":
        return selected[:, 0]
    if provider == "abs_channel":
        return selected[:, 0].abs()
    if provider == "vector_magnitude":
        return (selected.square().sum(1) + 1e-12).sqrt()
    dx, dy = central_gradient(selected[:, 0], periodic)
    if provider == "gradient_magnitude":
        return (dx.square() + dy.square() + 1e-12).sqrt()
    vx, vy = central_gradient(selected[:, 1], periodic)
    if provider == "vorticity":
        return ((vx - dy).square() + 1e-12).sqrt()
    if provider == "strain_rate":
        return (2 * (dx.square() + vy.square() + 0.5 * (dy + vx).square()) + 1e-12).sqrt()
    raise ValueError(f"unknown topology anchor provider: {provider}")


def reference_standardize(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    *,
    unbiased: bool,
    scale_epsilon: float = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reference = reference.detach()
    mean = reference.mean((-2, -1), keepdim=True)
    std = reference.std((-2, -1), keepdim=True, unbiased=unbiased)
    scale = std.clamp_min(1e-7) + scale_epsilon
    return (prediction - mean) / scale, (reference - mean) / scale, std.flatten() > 1e-7


def quantile_levels(reference: torch.Tensor, quantiles: tuple[float, ...]) -> torch.Tensor:
    q = reference.new_tensor(quantiles)
    return torch.quantile(reference.detach().flatten(1), q, dim=1).T


def masks(field: torch.Tensor, levels: torch.Tensor, sharpness: float) -> torch.Tensor:
    return torch.sigmoid(sharpness * (field[:, None] - levels[..., None, None]))


def dice_cost(prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Squared Dice error per sample/level; exactly zero for identical masks."""
    numerator = (prediction - reference).square().sum((-2, -1))
    denominator = (prediction.square() + reference.square()).sum((-2, -1)) + 1e-6
    return numerator / denominator


def _dilate(mask: torch.Tensor, periodic: bool) -> torch.Tensor:
    return F.max_pool2d(
        F.pad(mask, (1, 1, 1, 1), mode="circular" if periodic else "replicate"), 3, 1
    )


def _skeleton(mask: torch.Tensor, iterations: int, periodic: bool) -> torch.Tensor:
    eroded = -_dilate(-mask, periodic)
    skeleton = F.relu(mask - _dilate(eroded, periodic))
    for _ in range(iterations):
        mask = eroded
        eroded = -_dilate(-mask, periodic)
        delta = F.relu(mask - _dilate(eroded, periodic))
        skeleton = skeleton + (1 - skeleton).clamp(0, 1) * delta
    return skeleton.clamp(0, 1)


def cldice_cost(
    prediction: torch.Tensor, reference: torch.Tensor, *, iterations: int, periodic: bool
) -> torch.Tensor:
    pred_skeleton = _skeleton(prediction, iterations, periodic)
    ref_skeleton = _skeleton(reference, iterations, periodic)
    precision = ((pred_skeleton * reference).flatten(1).sum(1) + 1e-6) / (
        pred_skeleton.flatten(1).sum(1) + 1e-6
    )
    sensitivity = ((ref_skeleton * prediction).flatten(1).sum(1) + 1e-6) / (
        ref_skeleton.flatten(1).sum(1) + 1e-6
    )
    return 1 - 2 * precision * sensitivity / (precision + sensitivity + 1e-6)


def self_spatial_costs(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    *,
    directions: tuple[str, ...],
    quantiles: tuple[float, ...],
    physical_levels: tuple[float, ...] | None,
    sharpness: float,
    skeleton_sharpness: float,
    skeleton_iterations: int,
    periodic: bool,
    compute_connectivity: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Phase/selected-field masks, plus clDice at the middle reference level."""
    region, connectivity = [], []
    for channel in range(prediction.shape[1]):
        pred, ref = prediction[:, channel], reference[:, channel].detach()
        if physical_levels is None:
            pred, ref, _ = reference_standardize(pred, ref, unbiased=False)
            levels = quantile_levels(ref, quantiles)
        else:
            levels = pred.new_tensor(physical_levels).expand(pred.shape[0], -1)
        for direction in directions:
            sign = 1 if direction == "superlevel" else -1
            region.append(
                dice_cost(
                    masks(sign * pred, sign * levels, sharpness),
                    masks(sign * ref, sign * levels, sharpness),
                ).mean(1)
            )
            if compute_connectivity:
                middle = sign * levels[:, levels.shape[1] // 2 : levels.shape[1] // 2 + 1]
                connectivity.append(
                    cldice_cost(
                        masks(sign * pred, middle, skeleton_sharpness),
                        masks(sign * ref, middle, skeleton_sharpness),
                        iterations=skeleton_iterations,
                        periodic=periodic,
                    )
                )
    zero = prediction.sum((1, 2, 3)) * 0
    return torch.stack(region).mean(0), (
        torch.stack(connectivity).mean(0) if connectivity else zero
    )


def anchor_spatial_cost(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    *,
    quantiles: tuple[float, ...],
    sharpness: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred, ref, valid = reference_standardize(prediction, reference, unbiased=False)
    levels = quantile_levels(ref, quantiles)
    return dice_cost(masks(pred, levels, sharpness), masks(ref, levels, sharpness)).mean(1), valid


def mutual_spatial_cost(
    carrier: torch.Tensor,
    anchor: torch.Tensor,
    carrier_ref: torch.Tensor,
    anchor_ref: torch.Tensor,
    *,
    quantiles: tuple[float, ...],
    sharpness: float,
    directions: tuple[str, ...],
    detach_carrier: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    carrier = carrier.detach() if detach_carrier else carrier
    cg, ct, carrier_valid = reference_standardize(carrier, carrier_ref, unbiased=True)
    ag, at, anchor_valid = reference_standardize(anchor, anchor_ref, unbiased=True)
    a_levels = quantile_levels(at, quantiles)
    pa, ra = masks(ag, a_levels, sharpness), masks(at, a_levels, sharpness)
    costs = []
    for direction in directions:
        sign = 1 if direction == "superlevel" else -1
        levels = quantile_levels(sign * ct, quantiles)
        pc, rc = masks(sign * cg, levels, sharpness), masks(sign * ct, levels, sharpness)
        costs.extend((dice_cost(pc * pa, rc * ra), dice_cost(pc * (1 - pa), rc * (1 - ra))))
    return torch.stack(costs).mean((0, 2)), carrier_valid & anchor_valid
