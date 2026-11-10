"""Explicit parent-linked SOURCE budget binding for a Round 7 child recipe."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping

from .fidelity_reference import load_shared_reference

SOURCE_SHA = '2d1da713538f6e9613560f3a7eb4a9df2a95d1ad1f56dd11a1a8a4f9ef65050c'
C0 = 1.9418972484616779
ORDER = ['global_distribution', 'multiscale_spectrum', 'topology']


def digest(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def bind_child(parent_path: str, parent_sha256: str, parent_binding: Mapping[str, Any],
               child_binding: Mapping[str, Any], *, arm: str, preference: Mapping[str, Any] | None):
    """Validate the unmodified parent and bind the complete permitted child identity.

    Callers persist this record and its digest in both run manifest and checkpoint;
    on recovery regenerate it from the active config and compare the digest.
    """
    reference, raw = load_shared_reference(parent_path, parent_sha256, parent_binding)
    if reference.c0 != C0 or parent_binding['source_hashes']['checkpoint'] != SOURCE_SHA:
        raise ValueError('Round 7 requires the pinned SOURCE and original C0')
    parent = deepcopy(dict(parent_binding)); child = deepcopy(dict(child_binding))
    if arm == 'P':
        if parent != child or preference != {'family_order': ORDER, 'vector': [1., 1.2, 1.], 'norm_match': True}:
            raise ValueError('P child must retain parent baseline and exact preference')
    elif arm == 'T':
        if preference is not None:
            raise ValueError('T child must retain uniform preference')
        ptop = parent['baseline_config_anchor']['coherence']['families'].pop('topology')
        ctop = child['baseline_config_anchor']['coherence']['families'].pop('topology')
        parent['family_scales'].pop('topology')
        cscale = child['family_scales'].pop('topology')
        if (parent != child or ptop == ctop or
                ctop.get('persistence', {}).get('definition') != 'conditional_fixed_transform_fair_v1'):
            raise ValueError('T child may change only the declared Shape recipe and its scale')
        normalized_topology = deepcopy(ctop)
        normalized_topology['persistence']['definition'] = 'finite_primary_v1'
        if ptop != normalized_topology:
            raise ValueError('T child changed Shape geometry, weights, queries, or non-definition settings')
        if not isinstance(cscale, (int, float)) or isinstance(cscale, bool) or not 0 < cscale < float('inf'):
            raise ValueError('T child Shape scale must be positive and finite')
    else:
        raise ValueError('child arm must be P or T')
    record = {'schema': 'round7_source_child_recipe_v1', 'arm': arm,
              'parent_shared_reference_sha256': parent_sha256,
              'parent_binding_sha256': digest(parent_binding), 'child_binding': dict(child_binding),
              'preference': preference, 'reference': reference.state_dict(),
              'policy': 'original_source_risks_bounds_and_c0_without_recalibration'}
    return reference, raw, record
