#!/usr/bin/env python3
"""Freeze the TRAIN-only replacement for the degenerate Round 1 event.

Uses exactly the existing TRAIN panel and original task receipt. Never reads
VALIDATION/TEST or generated endpoints. The original evidence is unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get('CAR_REPO_ROOT', Path(__file__).resolve().parents[2])).resolve()
CODE_ROOT = Path(os.environ.get('CAR_CODE_ROOT', Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(CODE_ROOT / 'src'))
import torch
import yaml

from phycoflow_reconstruction.data.factory import open_field_dataset
from phycoflow_reconstruction.data.manifest import dataset_fingerprint
from phycoflow_reconstruction.evaluation.adaptive_assessment import (
    FrozenEventTask,
    _identity,
    _tensor_identity,
)
from phycoflow_reconstruction.evaluation.adaptive_campaign import digest, immutable_json


class FrozenRedesignedEvent:
    """Replay a hash-verified frozen event with the shared J4 scoring kernel."""

    def __init__(self, receipt):
        self._receipt = json.loads(json.dumps(receipt))
        payload = {k: v for k, v in self._receipt.items() if k != 'sha256'}
        if self._receipt.get('schema') != 'car_event_task_redesign_v1' or self._receipt.get('split') != 'train' or self._receipt.get('sha256') != _identity(payload):
            raise ValueError('invalid frozen redesigned event receipt')
        definition = self._receipt['definition']
        if [definition[k]['operator'] for k in ('T', 'CO', 'CH4')] != ['>', '>', '<']:
            raise ValueError('unsupported frozen event operators')

    def receipt(self):
        return json.loads(json.dumps(self._receipt))

    def masks(self, fields):
        fields = torch.as_tensor(fields)
        names = self._receipt['field_names']
        if fields.shape[-1] != len(names) or not torch.isfinite(fields).all():
            raise ValueError('fields must be finite in frozen field order')
        d = self._receipt['definition']
        hot = fields[..., names.index('T')] > d['T']['threshold']
        co = fields[..., names.index('CO')] > d['CO']['threshold']
        low_ch4 = fields[..., names.index('CH4')] < d['CH4']['threshold']
        return hot, co, low_ch4, hot & co & low_ch4

    def score(self, members, truth, masses, coordinates):
        return FrozenEventTask.score(self, members, truth, masses, coordinates)


def redesign(truth, masses, original, ids, population, *, data_identity):
    """Select the first 5-point quantile above a small TRAIN-relative floor.

    Retain T>q75, CO>q90, CH4<q and all original physical support. Starting at
    q30, choose the first q on a 0.05 grid whose CH4 value is at least 0.1% of
    its positive q50. This avoids a zero atom and a threshold at float32 decode
    resolution; selection uses only TRAIN truths, never models or task scores.
    """
    truth = torch.as_tensor(truth)
    mass = torch.as_tensor(masses).double()
    if truth.ndim != 3 or truth.shape[-1] != len(original['field_names']) or truth.shape[1] != mass.numel():
        raise ValueError('TRAIN native truth/mass shape mismatch')
    if original.get('split') != 'train' or list(ids) != original['sample_ids'] or population != original['population'] or len(ids) != 800 or population != 8000:
        raise ValueError('exact frozen TRAIN800/8000 panel required')
    if original['data_identity'] != data_identity or _tensor_identity(truth) != original['truth_sha256']:
        raise ValueError('TRAIN source identity mismatch')
    if _tensor_identity(mass.expand(len(ids), -1)) != original['native_masses_sha256']:
        raise ValueError('native physical measure mismatch')
    if not torch.isfinite(truth).all() or not torch.isfinite(mass).all() or (mass < 0).any() or mass.sum() <= 0:
        raise ValueError('finite truth and valid physical masses required')
    names = original['field_names']
    ch4 = truth[..., names.index('CH4')]
    sorted_values, order = ch4.flatten().sort()
    weights = mass.expand(len(ids), -1).reshape(-1)[order].cumsum(0)
    def quantile(q):
        return float(sorted_values[torch.searchsorted(weights, q * weights[-1]).clamp(max=len(sorted_values) - 1)])
    grid = (0.30, 0.35, 0.40, 0.45, 0.50)
    qs = {f'{q:.2f}': quantile(q) for q in grid}
    median = qs['0.50']
    if median <= 0:
        raise ValueError('CH4 q50 must be positive for relative threshold rule')
    chosen = next((q for q in grid if qs[f'{q:.2f}'] >= median * .001), None)
    if chosen is None:
        raise ValueError('no nondegenerate CH4 threshold')
    temperature = truth[..., names.index('T')] > original['temperature_q75']
    co = truth[..., names.index('CO')] > original['co_q90']
    original_event = temperature & co & (ch4 < original['ch4_q25'])
    if bool(original_event.any()):
        raise ValueError('original task is not degenerate on frozen TRAIN truths')
    event = temperature & co & (ch4 < qs[f'{chosen:.2f}'])
    area = (event * mass).sum(-1) / mass.sum()
    positive = area[area > 0]
    if not len(positive):
        raise ValueError('replacement remains empty on TRAIN')
    record = {
        'schema': 'car_event_task_redesign_v1', 'split': 'train',
        'field_names': names, 'sample_ids': list(ids), 'population': population,
        'data_identity': data_identity, 'truth_sha256': original['truth_sha256'],
        'native_masses_sha256': original['native_masses_sha256'], 'units': original['units'],
        'original_task_sha256': original['sha256'],
        'definition': {'T': {'operator': '>', 'quantile': .75, 'threshold': original['temperature_q75']},
                       'CO': {'operator': '>', 'quantile': .90, 'threshold': original['co_q90']},
                       'CH4': {'operator': '<', 'quantile': chosen, 'threshold': qs[f'{chosen:.2f}']}},
        'selection_rule': 'first CH4 quantile in [0.30,0.35,0.40,0.45,0.50] >= 0.001 times positive TRAIN CH4 q50',
        'rationale': 'Original strict CH4<q25 is empty because q25 is zero; retain strict physical cuts and choose a low-fuel threshold above decode-resolution-scale values.',
        'train_ch4_quantiles': qs,
        'train_support': {'samples': len(ids), 'samples_nonzero': int((area > 0).sum()),
                          'mean_area_fraction': float(area.mean()), 'max_area_fraction': float(area.max()),
                          'minimum_positive_area_fraction': float(positive.min())},
        'scope': 'frozen independent final assessment only; excluded from discovery, gradient, priorities, alpha and candidate selection',
    }
    return {**record, 'sha256': _identity(record)}, area


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=ROOT / 'cases/turbulent_combustion/runs/coherence_adaptive_repair/round_1')
    args = parser.parse_args()
    root = args.campaign.resolve()
    original_path = root / 'assessment/native_event_task.json'
    support_path = root / 'assessment/native_event_support.json'
    original, support = json.loads(original_path.read_text()), json.loads(support_path.read_text())
    if support.get('status') != 'requires_train_redesign' or support.get('task_sha256') != original.get('sha256') or support['TRAIN_support']['samples_with_nonzero_event'] != 0:
        raise ValueError('original zero-support receipt required')
    panel = json.loads((root / 'panels/train.json').read_text())
    if panel.get('split') != 'train' or panel['sample_ids'] != original['sample_ids']:
        raise ValueError('original task and TRAIN panel mismatch')
    config = yaml.safe_load((root / 'sources/ffm/resolved_config.yaml').read_text())['dataset']
    config['physical_measure'] = 'native_rectilinear_area'
    torch.set_num_threads(4)
    dataset = open_field_dataset(config, split='train')
    try:
        lookup = {f'{dataset.trajectory_ids[b]}:{t}': i for i, (b, t) in enumerate(dataset._items)}
        ids = panel['sample_ids']
        truth = torch.stack([dataset.normalizer.decode(dataset[lookup[x]].values) for x in ids])
        receipt, _area = redesign(truth, dataset.physical_weights, original, ids, len(dataset), data_identity=dataset_fingerprint(dataset.path))
        receipt['original_task_file_sha256'] = digest(original_path)
        receipt['original_support_file_sha256'] = digest(support_path)
        receipt['sha256'] = _identity({k: v for k, v in receipt.items() if k != 'sha256'})
        out = root / 'assessment/native_event_redesigned.json'
        immutable_json(out, receipt)
        print(json.dumps({'path': str(out), 'sha256': receipt['sha256'], 'definition': receipt['definition'], 'support': receipt['train_support']}, indent=2))
    finally:
        dataset.close()


if __name__ == '__main__':
    main()
