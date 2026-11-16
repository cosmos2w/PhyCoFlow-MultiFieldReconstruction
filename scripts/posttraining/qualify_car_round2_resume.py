#!/usr/bin/env python3
"""Compare a completed 12-epoch run with a fresh-process 9+3 resume.

This is execution equivalence evidence, not source, fidelity, or scientific
qualification. It neither launches a run nor opens TEST.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from phycoflow_reconstruction.evaluation.adaptive_campaign import digest, immutable_json
from phycoflow_reconstruction.training.run_store import load_project_checkpoint


def differences(left, right, path='', *, atol=1e-7, rtol=1e-6):
    import torch
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            return [f'{path}: tensor shape or dtype differs']
        if left.is_floating_point() or left.is_complex():
            ok = torch.allclose(left, right, rtol=rtol, atol=atol, equal_nan=False)
        else:
            ok = torch.equal(left, right)
        return [] if ok else [f'{path}: tensor values differ']
    if isinstance(left, dict) and isinstance(right, dict):
        out = []
        for key in sorted(left.keys() | right.keys(), key=str):
            if key not in left or key not in right:
                out.append(f'{path}/{key}: missing on one side')
            else:
                out.extend(differences(left[key], right[key], f'{path}/{key}', atol=atol, rtol=rtol))
        return out
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        if len(left) != len(right):
            return [f'{path}: length differs']
        return [item for i, (a, b) in enumerate(zip(left, right)) for item in differences(a, b, f'{path}/{i}', atol=atol, rtol=rtol)]
    if isinstance(left, float) and isinstance(right, float):
        return [] if math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=rtol, abs_tol=atol) else [f'{path}: values differ']
    return [] if type(left) is type(right) and left == right else [f'{path}: values differ']


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def scientific_checkpoint_view(checkpoint):
    """Retain optimizer/RNG/target state; separate measured accounting clocks."""
    result = copy.deepcopy(checkpoint)
    result.pop('config_sha256', None)
    runtime = result.get('scalable_runtime', {})
    for key in ('policy_seconds', 'occupied_seconds', 'sampler_seconds'):
        value = runtime.pop(key, None)
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f'invalid saved runtime timer: {key}')
    return result


def scientific_history_view(history, definitions):
    timers = [column['name'] for column in definitions['columns'] if column['units'] == 'seconds']
    result = copy.deepcopy(history)
    for row in result:
        for key in timers:
            value = row.pop(key, None)
            if value is not None and (not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0):
                raise ValueError(f'invalid measured history timer: {key}')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', required=True)
    p.add_argument('--formal-config', type=Path, required=True)
    p.add_argument('--uninterrupted', type=Path, required=True)
    p.add_argument('--resumed', type=Path, required=True)
    p.add_argument('--pause-launch', type=Path, required=True)
    p.add_argument('--pause-completion', type=Path, required=True)
    p.add_argument('--resume-launch', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    pause, resume = (json.loads(p.read_text()) for p in (a.pause_launch, a.resume_launch))
    completion = json.loads(a.pause_completion.read_text())
    if pause['pid'] == resume['pid'] or resume.get('resume') is None:
        raise ValueError('distinct fresh processes and a real resume launch are required')
    if Path(resume['resume']).resolve() != a.resumed.resolve() or pause['arm'] != a.arm or resume['arm'] != a.arm:
        raise ValueError('launch receipts must identify the same resumed arm/run')
    if not pause.get('pause_at_epoch') == 9:
        raise ValueError('launch receipt must bind the epoch 9 pause')
    if (completion.get('token') != pause.get('token') or completion.get('status') != 'budget_paused'
            or completion.get('global_step') != 108
            or Path(completion.get('run_dir', '')).resolve() != a.resumed.resolve()):
        raise ValueError('pause completion must record step 108 in the resumed run')
    configs = []
    import yaml
    for directory in (a.uninterrupted, a.resumed):
        for name in ('loss_history.png', 'loss_history.pdf'):
            if not (directory / 'logs' / name).is_file():
                raise ValueError(f'missing loss curve {directory / "logs" / name}')
        configs.append(yaml.safe_load((directory / 'resolved_config.yaml').read_text()))
    formal = yaml.safe_load(a.formal_config.read_text())
    for cfg in configs:
        cfg['optimization']['epochs'] = formal['optimization']['epochs']
        cfg['checkpointing']['epochs'] = formal['checkpointing']['epochs']
        cfg['output']['experiment_name'] = formal['output']['experiment_name']
        cfg['runtime']['device'] = formal['runtime']['device']
        diffs = differences(cfg, formal)
        if diffs:
            raise ValueError('qualification config differs scientifically from formal: ' + '; '.join(diffs[:10]))
    checkpoints = [load_project_checkpoint(d / 'checkpoints/last.pt') for d in (a.uninterrupted, a.resumed)]
    for ckpt in checkpoints:
        if ckpt['global_step'] != 144:
            raise ValueError('both runs must complete 144 updates')
    diff = differences(*(scientific_checkpoint_view(ckpt) for ckpt in checkpoints))
    definitions = [json.loads((d / 'logs/recording_definitions.json').read_text()) for d in (a.uninterrupted, a.resumed)]
    for filename in ('history.jsonl', 'recording_definitions.json'):
        paths = [d / 'logs' / filename for d in (a.uninterrupted, a.resumed)]
        if not all(path.is_file() for path in paths):
            raise ValueError(f'missing required recording artifact {filename}')
        values = [rows(path) if filename.endswith('.jsonl') else json.loads(path.read_text()) for path in paths]
        if filename == 'history.jsonl':
            values = [scientific_history_view(value, definition) for value, definition in zip(values, definitions)]
        diff.extend(differences(*values, path=filename))
    for relative in ('metrics/training_coupling.jsonl', 'logs/scalable_updates.jsonl'):
        values = [rows(d / relative) for d in (a.uninterrupted, a.resumed)]
        if any(len(value) != 144 for value in values):
            raise ValueError(f'qualification requires all 144 accepted update records: {relative}')
        diff.extend(differences(*values, path=relative))
        if relative.endswith('training_coupling.jsonl'):
            for value in values:
                if any(len(row['sample_ids']) != 64 or not row['accepted'] for row in value):
                    raise ValueError('all 64 conditions must bear the objective at each accepted update')
                population = len({item for row in value[:125] for item in row['sample_ids']})
                if population != 8000:
                    raise ValueError('qualification must cross the complete TRAIN8000 permutation boundary')
    if diff:
        raise ValueError('fresh-process equivalence failed: ' + '; '.join(diff[:20]))
    immutable_json(a.output, {'schema': 'car_round2_qualification_v1', 'arm': a.arm, 'passed': True,
                             'pause_step': 108, 'final_step': 144, 'formal_config_sha256': digest(a.formal_config),
                             'uninterrupted': str(a.uninterrupted.resolve()), 'resumed': str(a.resumed.resolve()),
                             'checkpoint_sha256': [digest(d / 'checkpoints/last.pt') for d in (a.uninterrupted, a.resumed)],
                             'pause_launch_sha256': digest(a.pause_launch), 'resume_launch_sha256': digest(a.resume_launch),
                             'pause_completion_sha256': digest(a.pause_completion),
                             'objective_visits': 144 * 64, 'verified_first_pass_unique_conditions': 8000,
                             'accounting_timers_compared_separately': ['scalable_runtime.policy_seconds', 'scalable_runtime.occupied_seconds', 'scalable_runtime.sampler_seconds', 'history columns with saved units seconds'],
                             'comparison_atol': 1e-7, 'comparison_rtol': 1e-6})
    print(a.output)

if __name__ == '__main__':
    main()
