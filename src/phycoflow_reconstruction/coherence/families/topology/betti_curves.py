"""Exact-forward H0/H1 Betti curves with straight-through gradients.

This is an independent implementation of the mathematical method documented
by the topology co-worker repository. No source from that unlicensed repository
is vendored into the package.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from torch.nn import functional


def _straight_through(soft: torch.Tensor, hard: torch.Tensor) -> torch.Tensor:
    return soft + (hard.to(soft.dtype) - soft).detach()


@lru_cache(maxsize=32)
def _neighbors(height: int, width: int, periodic: bool) -> tuple[tuple[int, ...], ...]:
    adjacency = [[] for _ in range(height * width)]
    for row in range(height):
        for column in range(width):
            node = row * width + column
            for delta_row, delta_column in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                next_row, next_column = row + delta_row, column + delta_column
                if periodic:
                    next_row %= height
                    next_column %= width
                elif not (0 <= next_row < height and 0 <= next_column < width):
                    continue
                adjacency[node].append(next_row * width + next_column)
    return tuple(tuple(values) for values in adjacency)


def _superlevel_pairs(field: torch.Tensor, periodic: bool) -> tuple[list[int], list[int], int]:
    """Return finite birth/death vertex IDs and the essential birth ID."""
    height, width = field.shape
    values = field.detach().to(device="cpu", dtype=torch.float64).reshape(-1).numpy()
    order = np.lexsort((np.arange(values.size), -values))
    adjacency = _neighbors(height, width, periodic)
    parent = np.arange(values.size)
    birth = np.arange(values.size)
    active = np.zeros(values.size, dtype=bool)
    finite_births: list[int] = []
    finite_deaths: list[int] = []

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for node_value in order:
        node = int(node_value)
        active[node] = True
        roots = sorted({find(other) for other in adjacency[node] if active[other]})
        if not roots:
            parent[node] = node
            birth[node] = node
            continue
        candidates = roots + [node]
        elder = min(candidates, key=lambda root: (-values[birth[root]], birth[root]))
        parent[node] = elder
        for root in roots:
            root = find(root)
            if root == elder:
                continue
            finite_births.append(int(birth[root]))
            finite_deaths.append(node)
            parent[root] = elder
        parent[elder] = elder
    root = find(int(order[-1]))
    return finite_births, finite_deaths, int(birth[root])


def betti0_curve(
    field: torch.Tensor,
    levels: torch.Tensor,
    *,
    sharpness: float,
    periodic: bool,
) -> torch.Tensor:
    births_ids, deaths_ids, essential_id = _superlevel_pairs(field, periodic)
    flat = field.reshape(-1)
    essential = flat[essential_id]
    curve = _straight_through(
        torch.sigmoid(float(sharpness) * (essential - levels)),
        essential >= levels,
    )
    if births_ids:
        births = flat[torch.as_tensor(births_ids, device=field.device)]
        deaths = flat[torch.as_tensor(deaths_ids, device=field.device)]
        threshold = levels[:, None]
        born = _straight_through(
            torch.sigmoid(float(sharpness) * (births[None] - threshold)),
            births[None] >= threshold,
        )
        alive = _straight_through(
            torch.sigmoid(float(sharpness) * (threshold - deaths[None])),
            threshold > deaths[None],
        )
        curve = curve + (born * alive).sum(dim=1)
    return curve


def euler_curve(
    field: torch.Tensor,
    levels: torch.Tensor,
    *,
    sharpness: float,
    periodic: bool,
) -> torch.Tensor:
    difference = field[None] - levels[:, None, None]
    mask = _straight_through(
        torch.sigmoid(float(sharpness) * difference), difference >= 0
    )
    vertices = mask.sum(dim=(1, 2))
    if periodic:
        right = torch.roll(mask, -1, dims=2)
        down = torch.roll(mask, -1, dims=1)
        diagonal = torch.roll(mask, shifts=(-1, -1), dims=(1, 2))
        edges = (mask * right).sum(dim=(1, 2)) + (mask * down).sum(dim=(1, 2))
        faces = (mask * right * down * diagonal).sum(dim=(1, 2))
    else:
        edges = (mask[:, :, :-1] * mask[:, :, 1:]).sum(dim=(1, 2))
        edges = edges + (mask[:, :-1, :] * mask[:, 1:, :]).sum(dim=(1, 2))
        faces = (
            mask[:, :-1, :-1]
            * mask[:, :-1, 1:]
            * mask[:, 1:, :-1]
            * mask[:, 1:, 1:]
        ).sum(dim=(1, 2))
    return vertices - edges + faces


def betti_curve(
    field: torch.Tensor,
    levels: torch.Tensor,
    dimension: int,
    *,
    sharpness: float,
    periodic: bool,
) -> torch.Tensor:
    b0 = betti0_curve(field, levels, sharpness=sharpness, periodic=periodic)
    if dimension == 0:
        return b0
    if dimension != 1:
        raise ValueError("topology coherence supports only H0 and H1")
    chi = euler_curve(field, levels, sharpness=sharpness, periodic=periodic)
    if periodic:
        minimum = field.min()
        b2 = _straight_through(
            torch.sigmoid(float(sharpness) * (minimum - levels)), minimum >= levels
        )
    else:
        b2 = torch.zeros_like(levels)
    return b0 - chi + b2


def betti_curves(
    field: torch.Tensor,
    levels: torch.Tensor,
    dimensions: tuple[int, ...],
    *,
    sharpness: float,
    periodic: bool,
) -> dict[int, torch.Tensor]:
    """Compute requested dimensions while reusing the expensive H0 pairing."""
    if not set(dimensions) <= {0, 1}:
        raise ValueError("topology coherence supports only H0 and H1")
    b0 = betti0_curve(field, levels, sharpness=sharpness, periodic=periodic)
    result = {0: b0} if 0 in dimensions else {}
    if 1 in dimensions:
        chi = euler_curve(field, levels, sharpness=sharpness, periodic=periodic)
        if periodic:
            minimum = field.min()
            b2 = _straight_through(
                torch.sigmoid(float(sharpness) * (minimum - levels)), minimum >= levels
            )
        else:
            b2 = torch.zeros_like(levels)
        result[1] = b0 - chi + b2
    return result


def gaussian_blur(fields: torch.Tensor, sigma: float, periodic: bool) -> torch.Tensor:
    if sigma <= 0:
        return fields
    radius = max(1, int(np.ceil(3.0 * sigma)))
    axis = torch.arange(-radius, radius + 1, device=fields.device, dtype=fields.dtype)
    kernel = torch.exp(-0.5 * (axis / float(sigma)).square())
    kernel = kernel / kernel.sum()
    channels = fields.shape[1]
    horizontal = kernel.reshape(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.reshape(1, 1, -1, 1).expand(channels, 1, -1, 1)
    mode = "circular" if periodic else "reflect"
    result = functional.conv2d(
        functional.pad(fields, (radius, radius, 0, 0), mode=mode),
        horizontal,
        groups=channels,
    )
    return functional.conv2d(
        functional.pad(result, (0, 0, radius, radius), mode=mode),
        vertical,
        groups=channels,
    )
