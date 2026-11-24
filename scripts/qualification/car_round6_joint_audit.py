#!/usr/bin/env python3
"""Round 6 TRAIN-only audit: historical controller profile and paired-response math.

This module contains the deterministic parts shared by the real-action audit.
No model evaluations or calibration are performed by the history command.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def coarse_profile(history, fields, allowances, scales, *, epochs=100, steps=12):
    """Off-policy profile: each last-batch epoch sample represents twelve actions."""
    if len(history) < epochs or steps <= 0:
        raise ValueError('complete epoch history and positive action count required')
    z = -np.asarray(allowances, dtype=np.float64) / np.asarray(scales, dtype=np.float64)
    multipliers = np.zeros_like(z)
    nu = math.exp(-1.0 / steps)
    for index, row in enumerate(history[:epochs]):
        if row.get('epoch') != index + 1 or not row.get('epoch_complete'):
            raise ValueError('history must contain consecutive completed epochs')
        errors = np.array([row['field_violation'][f] for f in fields], dtype=np.float64)
        if not np.isfinite(errors).all():
            raise ValueError('nonfinite recorded field violation')
        for _ in range(steps):
            previous = z.copy()
            z = nu * z + (1 - nu) * errors
            multipliers = np.maximum(0, multipliers + 0.0012 / steps * errors + 0.012 * (z - previous))
    return multipliers, z


def weighted_mean(rows, counts):
    x = np.asarray(rows, dtype=np.float64)
    weights = np.asarray(counts, dtype=np.float64)
    if x.ndim < 1 or len(x) != len(weights) or np.any(weights <= 0) or not np.isfinite(x).all():
        raise ValueError('finite response rows and positive matched sample weights required')
    return np.average(x, axis=0, weights=weights)


def fit_gain(pre, zero, probe, counts, gamma_probe, allowance, scale, *, steps=12, horizon=10):
    if not math.isfinite(gamma_probe) or gamma_probe <= 0:
        raise ValueError('probe amplitude must be positive and finite')
    allowance, scale = np.asarray(allowance, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    if np.any(allowance <= 0) or np.any(scale <= 0):
        raise ValueError('positive physical allowance required; zero needs an explicit reference unit')
    pre, zero, probe = (weighted_mean(x, counts) for x in (pre, zero, probe))
    delta = 1 / steps
    baseline = (zero - pre) / delta
    response = (probe - zero) / (gamma_probe * delta)
    target = -np.maximum(pre, 0) / horizon
    weight = scale / allowance
    wr = weight * response
    denom = float(wr @ wr)
    if not math.isfinite(denom) or denom == 0:
        raise ValueError('zero or nonfinite measured response; no gain can be nominated')
    gamma = float(np.clip(np.dot(wr, weight * (target - baseline)) / denom, 0, gamma_probe))
    return {'gamma_star': gamma, 'gamma_probe': float(gamma_probe), 'pre': pre.tolist(),
            'zero': zero.tolist(), 'probe': probe.tolist(), 'baseline_drift': baseline.tolist(),
            'response_per_gain': response.tolist(), 'target_drift': target.tolist(),
            'baseline_weighted_residual_squared': float(np.sum((weight * (baseline - target)) ** 2))}


def history_audit(run):
    run = Path(run)
    history_path = run / 'logs/history.jsonl'
    history = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    reference_path = run / 'artifacts/fidelity_budget_reference.json'
    reference = json.loads(reference_path.read_text())['reference']
    lam, z = coarse_profile(history, reference['field_names'], reference['absolute_allowance'], reference['risk_scales'])
    keys = sorted(k for k in history[0] if k.startswith('coherence_component/') and k.endswith(('/inner_weight', '/outer_weight', '/calibration_scale')))
    if not keys:
        raise ValueError('missing coefficient recording columns')
    coefficient_changes = {k: sorted({float(row[k]) for row in history if k in row}) for k in keys}
    missing = {k: sum(k not in row for row in history) for k in keys}
    lr_path = run / 'metrics/lr_warmup_updates.jsonl'
    lr = [json.loads(line) for line in lr_path.read_text().splitlines() if line.strip()]
    return {'history_sha256': sha256(history_path), 'reference_sha256': sha256(reference_path),
            'history_epochs': len(history), 'profile_epochs': 100, 'actions_per_saved_epoch': 12,
            'profile_interpretation': 'off-policy coarse reconstruction; saved last batch repeated 12 times, not an epoch mean or trained counterfactual',
            'field_names': reference['field_names'], 'profile_lambda': lam.tolist(), 'profile_z': z.tolist(),
            'reference': reference, 'coefficient_values': coefficient_changes, 'coefficient_missing_counts': missing,
            'coefficients_constant': all(len(v) == 1 for v in coefficient_changes.values()) and not any(missing.values()),
            'lr_updates_sha256': sha256(lr_path), 'lr_update_count': len(lr),
            'lr_first': lr[0] if lr else None, 'lr_last': lr[-1] if lr else None,
            'epoch_lr_first': history[0].get('optimizer_lr'), 'epoch_lr_100': history[99].get('optimizer_lr')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = history_audit(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')

if __name__ == '__main__':
    main()
