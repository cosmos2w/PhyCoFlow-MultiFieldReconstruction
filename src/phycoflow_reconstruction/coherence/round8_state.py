"""Frozen hard/diverse State directions and common Hilbert energy score."""
from __future__ import annotations
import hashlib
import torch

VERSION = 'round8_state_hard_diverse_hilbert_v1'


def select_directions(candidates, hardness, field_order, seed):
    """Four angular farthest-first anchors, then four ranked-hardness anchors.

    The first direction is lowest original ID. Sign-equivalent duplicates are
    removed in original order. Hardness must be mean projected weighted W2^2
    measured from fixed TRAIN/SOURCE, supplied by the caller.
    """
    q = torch.as_tensor(candidates).detach().cpu().to(torch.float64)
    h = torch.as_tensor(hardness).detach().cpu().to(torch.float64)
    if q.ndim != 2 or h.shape != (q.shape[0],) or q.shape[1] != len(field_order) or not torch.isfinite(q).all() or not torch.isfinite(h).all() or (h < 0).any():
        raise ValueError('invalid directions, field order or hardness')
    if len(field_order) == 1:
        ids = []
    else:
        norms = q.norm(dim=1)
        if (norms <= 0).any():
            raise ValueError('directions must be nonzero')
        q = q / norms[:, None]
        unique = []
        for i in range(len(q)):
            if not any(abs(float(q[i] @ q[j])) >= 1 - 1e-10 for j in unique):
                unique.append(i)
        ids = unique[:1]
        while len(ids) < min(4, len(unique)):
            remaining = [i for i in unique if i not in ids]
            ids.append(min(remaining, key=lambda i: (max(abs(float(q[i] @ q[j])) for j in ids), i)))
        # Average tied ranks to avoid assigning different hardness to equal values.
        rank = {}
        for i in unique:
            less = sum(h[j] < h[i] for j in unique)
            equal = sum(h[j] == h[i] for j in unique)
            rank[i] = (float(less) + (float(equal)-1)/2) / max(len(unique)-1, 1)
        while len(ids) < min(8, len(unique)):
            remaining = [i for i in unique if i not in ids]
            ids.append(min(remaining, key=lambda i: (-(rank[i] - .25 * max(abs(float(q[i] @ q[j])) for j in ids)), i)))
    selected = q[ids].clone()
    digest = hashlib.sha256(selected.contiguous().numpy().astype('<f8').tobytes()).hexdigest()
    return {'version': VERSION, 'field_order': tuple(field_order), 'seed': int(seed), 'candidate_directions': q.clone(), 'hardness': h.clone(), 'selected_ids': tuple(ids), 'directions': selected, 'weights': torch.full((len(ids),), 1 / len(ids), dtype=torch.float64) if ids else torch.empty(0, dtype=torch.float64), 'sha256': digest, 'mixed_active': bool(ids)}


def common_distance(marginal_x, mixed_x, marginal_y, mixed_y, *, marginal_weight=1., mixed_weight=1., marginal_coefficient=1., mixed_coefficient=1.):
    """Distance from the same fixed feature embedding for all score pairs."""
    if min(marginal_weight, marginal_coefficient) <= 0 or mixed_weight < 0 or mixed_coefficient <= 0:
        raise ValueError('invalid metric weights')
    s = ((marginal_x - marginal_y) ** 2).sum(-1) * marginal_weight * marginal_coefficient
    if mixed_x is not None or mixed_y is not None:
        if mixed_x is None or mixed_y is None:
            raise ValueError('both mixed representations are required')
        s = s + ((mixed_x - mixed_y) ** 2).sum(-1) * mixed_weight * mixed_coefficient
    positive = s > 0
    return torch.where(positive, torch.sqrt(torch.where(positive, s, torch.ones_like(s))), torch.zeros_like(s))


def fair_score(marginal_members, marginal_reference, mixed_members=None, mixed_reference=None, **weights):
    """Score cached per-endpoint quantile features; member dimension is first."""
    j = marginal_members.shape[0]
    if j < 2 or (mixed_members is None) != (mixed_reference is None):
        raise ValueError('at least two members and matching mixed features required')
    def distance(i, right, reference=False):
        mx = None if mixed_members is None else mixed_members[i]
        my = None if mixed_members is None else (mixed_reference if reference else mixed_members[right])
        return common_distance(marginal_members[i], mx, marginal_reference if reference else marginal_members[right], my, **weights)
    attraction = torch.stack([distance(i, 0, True) for i in range(j)]).mean(0)
    separation = sum((distance(i, k) for i in range(j) for k in range(i+1,j)), torch.zeros_like(attraction)) / (j*(j-1))
    return attraction - separation, attraction, separation


def fit_source_hardness(batches, candidates):
    """Stream matched TRAIN/SOURCE batches, averaging exact projected W2².

    Each batch is (source [B,J,N,C], reference [B,N,C], source masses
    [B,N], reference masses [B,N]); endpoint scales/field order are already
    frozen and applied by the caller. Supports unequal endpoint measures.
    """
    from .base import weighted_w2_columns
    directions = torch.as_tensor(candidates).detach()
    if directions.ndim != 2 or (directions.norm(dim=-1) == 0).any():
        raise ValueError('invalid direction bank')
    directions = directions / directions.norm(dim=-1, keepdim=True)
    total = torch.zeros(len(directions), dtype=torch.float64)
    count = 0
    for source, reference, source_weights, reference_weights in batches:
        if source.ndim != 4 or reference.ndim != 3 or source.shape[0] != reference.shape[0] or source.shape[-1] != directions.shape[1]:
            raise ValueError('TRAIN source/reference batch shape mismatch')
        d = directions.to(source)
        for b in range(source.shape[0]):
            for j in range(source.shape[1]):
                value = weighted_w2_columns(source[b,j] @ d.T, reference[b] @ d.T, source_weights[b], reference_weights[b])
                total += value.detach().cpu().double()
                count += 1
    if not count:
        raise ValueError('TRAIN stream must be nonempty')
    return total / count


def exact_common_embeddings(endpoints, masses, directions, channel_weights=None):
    """Sort each endpoint once; embed on all endpoints' merged CDF intervals.

    Endpoints are [B,N,C]; masses may differ by endpoint. Every pair then has
    exact weighted projected W2² in one common finite Hilbert space.
    """
    if not endpoints or len(endpoints) != len(masses):
        raise ValueError('each endpoint needs its own masses')
    bsz, _, fields = endpoints[0].shape
    d = torch.as_tensor(directions, device=endpoints[0].device, dtype=endpoints[0].dtype)
    if d.ndim != 2 or d.shape[1] != fields:
        raise ValueError('invalid directions')
    channel_weights = torch.ones(fields, device=endpoints[0].device, dtype=endpoints[0].dtype) if channel_weights is None else torch.as_tensor(channel_weights, device=endpoints[0].device, dtype=endpoints[0].dtype).detach()
    if channel_weights.shape != (fields,) or not torch.isfinite(channel_weights).all() or (channel_weights < 0).any() or channel_weights.sum() <= 0:
        raise ValueError('invalid marginal channel weights')
    channel_weights = channel_weights / channel_weights.sum()
    projected, cdfs = [], []
    for x, raw_mass in zip(endpoints,masses):
        if x.ndim != 3 or x.shape[0] != bsz or x.shape[2] != fields or not torch.isfinite(x).all():
            raise ValueError('invalid endpoint')
        mass = torch.as_tensor(raw_mass,device=x.device,dtype=x.dtype).detach()
        if mass.ndim == 1: mass = mass.expand(bsz,-1)
        if mass.shape != x.shape[:2] or not torch.isfinite(mass).all() or (mass < 0).any() or (mass.sum(-1) <= 0).any():
            raise ValueError('invalid endpoint masses')
        mass = mass / mass.sum(-1,keepdim=True)
        p = torch.cat((x,x @ d.T),-1)
        val, idx = p.sort(dim=1)
        cdf = mass.unsqueeze(-1).expand_as(p).gather(1,idx).cumsum(1).clamp(max=1)
        projected.append(val.transpose(1,2).contiguous())
        cdfs.append(cdf.transpose(1,2).contiguous())
    all_breaks = torch.cat([torch.zeros_like(cdfs[0][...,:1]), *cdfs],dim=-1).sort(dim=-1).values
    lengths = (all_breaks[...,1:] - all_breaks[...,:-1]).clamp_min(0)
    mid = ((all_breaks[...,1:] + all_breaks[...,:-1]) / 2).contiguous()
    result = []
    for value,cdf in zip(projected,cdfs):
        index = torch.searchsorted(cdf,mid).clamp_max(value.shape[-1]-1)
        features = value.gather(-1,index) * lengths.sqrt()
        marginal = (features[:,:fields] * channel_weights.sqrt()[None,:,None]).reshape(bsz,-1)
        mixed = None if not len(d) else (features[:,fields:] / len(d)**.5).reshape(bsz,-1)
        result.append((marginal,mixed))
    return result


def score_family(members, reference, member_weights, reference_weights, bank, quantile_count=None, *, marginal_weight=1., mixed_weight=1., marginal_coefficient=1., mixed_coefficient=1., channel_weights=None):
    """FamilyResult and exact autograd semantic channel probe tensors.

    Members [J,B,N,C] are embedded exactly once each; reference is [B,N,C].
    Semantic probes sum channel chain-rule terms over identical reference and
    member pairs, keeping attraction and separation together.
    """
    from ..contracts import FamilyResult, TermResult
    from ..training.round8_calibration import state_channel_contribution
    if bank['version'] != VERSION or members.ndim != 4 or members.shape[0] < 2:
        raise ValueError('invalid State bank or members')
    if marginal_weight <= 0 or marginal_coefficient <= 0 or mixed_coefficient <= 0 or mixed_weight < 0 or (bank['mixed_active'] and mixed_weight <= 0):
        raise ValueError('active metric weights and coefficients must be positive')
    if quantile_count is not None:
        raise ValueError('exact State representation does not accept quantile_count')
    embeddings = exact_common_embeddings([*members,reference], [member_weights]*len(members)+[reference_weights], bank['directions'], channel_weights)
    ref = embeddings.pop()
    j = len(embeddings)
    metrics = (marginal_weight*marginal_coefficient, mixed_weight*mixed_coefficient)
    def pair(left, right):
        sm = (left[0]-right[0]).square().sum(-1)
        sx = torch.zeros_like(sm) if left[1] is None else (left[1]-right[1]).square().sum(-1)
        sq = metrics[0]*sm + metrics[1]*sx
        positive = sq > 0
        dist = torch.where(positive, torch.sqrt(torch.where(positive,sq,torch.ones_like(sq))),torch.zeros_like(sq))
        return dist, state_channel_contribution(sm,sq,metrics[0]), state_channel_contribution(sx,sq,metrics[1])
    attract = [pair(e,ref) for e in embeddings]
    separate = [pair(embeddings[i],embeddings[k]) for i in range(j) for k in range(i+1,j)]
    vals = [sum((p[n] for p in attract), torch.zeros_like(attract[0][n]))/j - sum((p[n] for p in separate), torch.zeros_like(attract[0][n]))/(j*(j-1)) for n in range(3)]
    score = vals[0]
    result = FamilyResult({'global_distribution.round8_fair': TermResult(score,score.mean())},score,score.mean(),{'score_definition':VERSION,'directions_sha256':bank['sha256'],'quantile_representation':'exact_merged_weighted_cdf_v1','conditional_draws':j})
    return result, {'state.marginal': vals[1].mean(), 'state.mixed': vals[2].mean() if bank['mixed_active'] else None}
