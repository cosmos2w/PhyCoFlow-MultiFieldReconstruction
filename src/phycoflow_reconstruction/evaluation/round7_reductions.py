"""Round 7 reusable J2 reductions and strict observer cache identities.

The caller supplies already generated matched outputs; these functions never infer.
Q4096 fidelity is unweighted and separate from any physical-measure reduction.
"""
from __future__ import annotations
import hashlib
import json
from typing import Mapping
import torch


def identity_hash(identity: Mapping) -> str:
    required = {'checkpoint_sha256', 'manifest_sha256', 'support_sha256',
                'draw_seed', 'query_seed', 'observer_sha256', 'observer_version'}
    if not required <= identity.keys():
        raise ValueError('incomplete observer cache identity')
    for key in ('checkpoint_sha256', 'manifest_sha256', 'support_sha256', 'observer_sha256'):
        value = identity[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'invalid {key}')
    if not identity['observer_version']:
        raise ValueError('observer version required')
    return hashlib.sha256(json.dumps(dict(identity), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_cache(cache: Mapping, identity: Mapping) -> None:
    if cache.get('identity_sha256') != identity_hash(identity):
        raise ValueError('incompatible observer, model, support, or draws')


def j2_field_reductions(x1: torch.Tensor, x2: torch.Tensor, target: torch.Tensor,
                        mask: torch.Tensor, weights: torch.Tensor | None = None) -> dict:
    """Fieldwise paired MSE, generated-variance and bias-plus-noise proxy.

    Arrays end in [sites, fields], with any leading condition dimensions. A
    nonnegative physical measure may be supplied for explanatory reductions.
    The signed bias proxy is not clipped and is not isolated conditional bias.
    """
    if x1.shape != x2.shape or x1.shape != target.shape or x1.ndim < 2 or mask.shape != x1.shape[:-1]:
        raise ValueError('draw, target and mask shapes disagree')
    if not all(bool(torch.isfinite(x).all()) for x in (x1, x2, target)):
        raise ValueError('nonfinite prediction or reference')
    w = mask.to(dtype=x1.dtype, device=x1.device)
    if weights is not None:
        if weights.shape != mask.shape or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError('invalid physical weights')
        w = w * weights.to(dtype=x1.dtype, device=x1.device)
    count = w.sum()
    if not bool(count > 0):
        raise ValueError('empty reduction support')
    axes = tuple(range(x1.ndim - 1))
    def reduce(z):
        return ((z * w.unsqueeze(-1)).sum(dim=axes) / count).detach().cpu().tolist()
    e1, e2 = (x1 - target).square(), (x2 - target).square()
    variance = .5 * (x1 - x2).square()
    return {'primary_mse': reduce(e1), 'secondary_mse': reduce(e2),
            'variance_proxy': reduce(variance),
            'bias_plus_reference_variance_proxy': reduce(.5 * (e1 + e2) - variance)}
