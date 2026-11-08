"""Exact endpoint-cached weighted State distances for CPU native observation.

The cache owns no model calls and does not change the historical v3 family. It
reuses projections, value orders and cumulative masses of each endpoint;
merged-CDF integration remains exact for each endpoint pair. Endpoint tensors
remain connected to autograd, including every conditional member.
"""
from __future__ import annotations

from itertools import combinations

import torch

from ..coherence.base import require_field_tensor


def _sorted_endpoint(values, weights, uniform):
    # Historical equal-mass transport uses the default (unstable) torch sort;
    # nonuniform transport uses stable sorting to define tie subgradients.
    sorted_values, order = values.sort(dim=0, stable=not uniform)
    if uniform:
        return sorted_values, None
    masses = weights[:, None].expand_as(values).gather(0, order)
    cumulative = masses.cumsum(dim=0).clamp(max=1.0)
    cdf = torch.cat((cumulative[:-1], torch.ones_like(cumulative[-1:])), dim=0)
    return sorted_values, cdf


def _pair_columns(left, right, uniform):
    left_values, left_cdf = left
    right_values, right_cdf = right
    if uniform:
        return (left_values - right_values).square().mean(dim=0)
    breaks = torch.cat((left_cdf, right_cdf), dim=0).sort(dim=0).values
    widths = torch.diff(breaks, dim=0, prepend=torch.zeros_like(breaks[:1]))
    query = breaks.T.contiguous()
    li = torch.searchsorted(left_cdf.T.contiguous(), query).T
    ri = torch.searchsorted(right_cdf.T.contiguous(), query).T
    left_at = left_values.gather(0, li.clamp_max(left_values.shape[0] - 1))
    right_at = right_values.gather(0, ri.clamp_max(right_values.shape[0] - 1))
    return (widths * (left_at - right_at).square()).sum(dim=0)


def cached_state_matrices(family, members, target, *, weights, mask=None):
    """Compute joint and leaf squared distances for reference + members on CPU.

    `members` is [J,B,N,C], `target` is [B,N,C], with a common physical
    measure. Outputs match CommonStateObserver's exact v3 pair/leaf matrices.
    """
    if getattr(family, 'score_definition', None) != 'marginal_mixed_weighted_v3':
        raise ValueError('cached State requires marginal_mixed_weighted_v3')
    if members.ndim != 4 or members.shape[1:] != target.shape or members.shape[0] < 2:
        raise ValueError('members [J,B,N,C] must match target [B,N,C], J>=2')
    if target.device.type != 'cpu' or members.device.type != 'cpu':
        raise ValueError('cached State observer supports CPU only')
    if target.dtype != members.dtype or target.device != members.device:
        raise ValueError('State endpoints must share dtype and device')
    require_field_tensor('target', target)
    for member in members.unbind(0): require_field_tensor('member', member)
    context = {'generated_weights': weights, 'reference_weights': weights,
               'generated_mask': mask, 'reference_mask': mask}
    mass, _ = family._endpoint_measures(target, target, context, required=True)
    endpoints = [target, *members.unbind(0)]
    endpoints = [family._in_declared_units(x, x)[0] for x in endpoints]
    component = family.components_by_key['self_marginal_w2']
    channel_weights = component.channel_weights.to(device=target.device, dtype=target.dtype)
    self_weight = family.component_weights['self_marginal_w2']
    cross = family.components_by_key['cross_joint_topk_swd'] if 'cross_joint_topk_swd' in family.components_by_key else None
    cross_weight = family.component_weights.get('cross_joint_topk_swd', 0.)
    if cross is not None and (not cross.mean_reduction or cross.top_fraction != 1.0):
        raise ValueError('cached State requires the full mean direction bank')
    directions = None if cross is None else cross.directions.to(device=target.device, dtype=target.dtype)
    count, batch = len(endpoints), target.shape[0]
    matrix = target.new_zeros((count, count, batch))
    field_key = 'global_distribution.self.marginal_w2.per_field_w2'
    direction_key = 'global_distribution.cross.joint_topk_swd.per_direction_w2'
    fields = target.new_zeros((count, count, batch, len(component.field_ids)))
    projected = None if directions is None or cross_weight == 0 else target.new_zeros((count, count, batch, len(directions)))
    for b in range(batch):
        # weighted_w2_columns normalizes its already normalized endpoint mass a
        # second time; preserve that operation and the uniform shortcut exactly.
        w = mass[b].detach()
        w = w / w.sum()
        uniform = bool(torch.all(w == w[0]))
        self_cache = [_sorted_endpoint(x[b, :, component.field_ids], w, uniform) for x in endpoints]
        cross_cache = None
        if projected is not None:
            cross_cache = [_sorted_endpoint(x[b, :, cross.field_ids] @ directions.T, w, uniform) for x in endpoints]
        for i, j in combinations(range(count), 2):
            f = _pair_columns(self_cache[i], self_cache[j], uniform)
            total = self_weight * (f * channel_weights).sum()
            fields[i, j, b] = fields[j, i, b] = f
            if cross_cache is not None:
                p = _pair_columns(cross_cache[i], cross_cache[j], uniform)
                projected[i, j, b] = projected[j, i, b] = p
                total = total + cross_weight * p.mean()
            matrix[i, j, b] = matrix[j, i, b] = total
    leaves = {field_key: fields}
    if projected is not None: leaves[direction_key] = projected
    return {'squared_distances': matrix, 'leaf_squared_distances': leaves}
