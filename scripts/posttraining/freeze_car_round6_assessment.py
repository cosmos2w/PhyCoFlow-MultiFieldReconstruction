#!/usr/bin/env python3
"""Choose Round 6 illustration strata using only the cached SOURCE panel.

This uses the existing matched VALIDATION100 bank, with no prediction generation
or candidate inspection. The reused event definition is never recalibrated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]


def main():
    import torch
    from phycoflow_reconstruction.evaluation.adaptive_campaign import digest, immutable_json
    from scripts.posttraining.assess_car_round3 import context, verify
    from scripts.posttraining.freeze_car_event import FrozenRedesignedEvent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    torch.set_num_threads(4)
    ctx = context(campaign.parent / 'round_3')
    baseline = next(row for row in ctx['baseline']['rows'] if row['alpha'] == 0.)
    ids = ctx['source']['sample_ids']
    scales = torch.tensor(baseline['risk']['member_mse'], dtype=torch.float64)
    errors = torch.tensor(baseline['per_condition_member_mse'], dtype=torch.float64)
    if len(ids) != 100 or errors.shape != (100, len(scales)) or not bool((scales > 0).all()):
        raise ValueError('matched SOURCE VALIDATION100 and positive field scales required')
    normalized = (errors / scales).mean(1).tolist()
    ranked = sorted(range(100), key=lambda i: (normalized[i], ids[i]))
    event_path = verify(ctx['freeze']['event'])
    event = FrozenRedesignedEvent(json.loads(event_path.read_text()))
    support = []
    for i, sample_id in enumerate(ids):
        path = ctx['source_dir'] / f'{i:04d}.pt'
        if digest(path) != ctx['source']['files_sha256'][i]:
            raise ValueError('cached SOURCE bank row changed')
        row = torch.load(path, map_location='cpu', weights_only=True)
        if row['sample_id'] != sample_id:
            raise ValueError('SOURCE sample order changed')
        truth = row['truth_model'] * row['normalizer_scale'] + row['normalizer_offset']
        mask = event.masks(truth)[-1].double()
        mass = row['native_masses'].double()
        support.append(float((mask * mass).sum() / mass.sum()))
    excluded = {ranked[50], ranked[-1]}
    supported = [i for i, area in enumerate(support) if .005 <= area <= .5 and i not in excluded]
    if not supported:
        raise ValueError('no separate illustrative condition has a nontrivial frozen event support')
    supported.sort(key=lambda i: (support[i], ids[i]))
    choices = [('typical_field_error', ranked[50]), ('high_field_error', ranked[-1]),
               ('supported_joint_event', supported[len(supported)//2])]
    immutable_json(campaign / 'configs/visual_selection.json', {
        'schema': 'car_round6_source_defined_visual_cases_v1', 'created_unix': time.time(),
        'candidates_used': False, 'source_bank_sha256': digest(ctx['source_receipt']),
        'panel': {'path': str(ctx['panel_path']), 'sha256': digest(ctx['panel_path'])},
        'source_bank': {'path': str(ctx['source_receipt']), 'sha256': digest(ctx['source_receipt'])},
        'event': {'path': str(event_path), 'sha256': digest(event_path)},
        'regional_task': ctx['freeze']['reserved_task'],
        'selection_rule': 'Median and maximum SOURCE member-MSE normalized per field; separate median supported truth-event condition with mass fraction between 0.005 and 0.5.',
        'selection_use': 'reused development VALIDATION100 illustrations, not model calibration or independent confirmation',
        'cases': [{'stratum': name, 'ordinal': i, 'sample_id': ids[i],
                   'source_normalized_field_error': normalized[i],
                   'truth_event_mass_fraction': support[i]} for name, i in choices],
        'test_locked': True,
    })
    print(campaign / 'configs/visual_selection.json')


if __name__ == '__main__':
    main()
