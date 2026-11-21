#!/usr/bin/env python3
"""Freeze SOURCE-defined representative cases before any Round 4 candidates."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    args = p.parse_args()
    from scripts.posttraining.assess_car_round3 import context
    from phycoflow_reconstruction.evaluation.adaptive_campaign import digest, immutable_json
    campaign = args.campaign.resolve(strict=True)
    ctx = context(campaign.parent / 'round_3')
    baseline = next(row for row in ctx['baseline']['rows'] if row['alpha'] == 0.)
    errors = baseline['per_condition_member_mse']
    scales = baseline['risk']['member_mse']
    ids = ctx['source']['sample_ids']
    if len(errors) != len(ids) or len(ids) != 100 or any(v <= 0 for v in scales):
        raise ValueError('positive SOURCE field scales and matched VALIDATION100 rows required')
    values = [sum(v/s for v,s in zip(row, scales, strict=True))/len(scales) for row in errors]
    if not all(math.isfinite(value) for value in values):
        raise ValueError('nonfinite SOURCE case-selection errors')
    ranked = sorted(range(len(ids)), key=lambda i:(values[i], ids[i]))
    selection = dict(representative=ranked[len(ranked)//2], source_tail=ranked[-1])
    immutable_json(campaign / 'configs/visual_selection.json', dict(
        schema='car_round4_source_defined_visual_cases_v1',
        selection_rule='median and maximum SOURCE per-condition member MSE, normalized by each field population SOURCE member MSE, equally averaged across fields',
        candidates_used=False, selection_use='previously accessed development VALIDATION100',
        panel=dict(path=str(ctx['panel_path']),sha256=digest(ctx['panel_path'])),
        source_bank=dict(path=str(ctx['source_receipt']),sha256=digest(ctx['source_receipt'])),
        baseline=dict(path=str(ctx['baseline_path']),sha256=digest(ctx['baseline_path'])),
        cases={key:dict(ordinal=i,sample_id=ids[i],source_normalized_field_error=values[i]) for key,i in selection.items()},
        test_locked=True))
    print(campaign / 'configs/visual_selection.json')


if __name__ == '__main__':
    main()
