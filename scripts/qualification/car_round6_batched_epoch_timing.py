#!/usr/bin/env python3
"""Matched fresh-SOURCE complete trainer epochs for optional family VJP execution."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import torch
import yaml
from phycoflow_reconstruction.training.post_training import run_post_training


def initial_state_digest(path):
    """Hash actual ordered initial model, optimizer, RNG, dual and sampler state."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['global_step'] != 0:
        raise ValueError('expected fresh SOURCE epoch-zero checkpoint')
    runtime = state['scalable_runtime']
    selected = {key:state[key] for key in ('model','optimizer','rng_state','external_policy','family_scales','source_hashes')}
    selected['sampler'] = {key:runtime[key] for key in ('step','sampler','group_samplers','probe_sampler')}
    digest = hashlib.sha256()
    def add(value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().contiguous().cpu()
            digest.update(f'tensor:{tensor.dtype}:{tuple(tensor.shape)}:'.encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(b'dict{')
            for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
                add(key); add(value[key])
            digest.update(b'}')
        elif isinstance(value, (list,tuple)):
            digest.update(type(value).__name__.encode()+b'[')
            for item in value: add(item)
            digest.update(b']')
        elif value is None or isinstance(value, (str,int,float,bool)):
            digest.update(f'{type(value).__name__}:{value!r};'.encode())
        else:
            raise TypeError(f'unsupported initial-state value {type(value).__name__}')
    add(selected)
    return digest.hexdigest()


def final_model_comparison(reference, candidate, parameter_names):
    """Compare finite trainable final parameters; this is not an efficacy measure."""
    left = torch.load(reference, map_location='cpu', weights_only=False)
    right = torch.load(candidate, map_location='cpu', weights_only=False)
    if left['global_step'] != right['global_step'] or set(left['model']) != set(right['model']):
        raise ValueError('final checkpoint steps or model keys differ')
    squared = 0.0
    baseline_squared = 0.0
    maximum = 0.0
    if not parameter_names or not set(parameter_names).issubset(left['model']):
        raise ValueError('trainable parameter names missing from final checkpoint')
    for name in parameter_names:
        a, b = left['model'][name], right['model'][name]
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f'final model tensor differs in shape or is nonfinite: {name}')
        if not (a.is_floating_point() or a.is_complex()):
            if not torch.equal(a, b): raise ValueError(f'final model metadata differs: {name}')
            continue
        delta = (a-b).abs().double()
        squared += float(delta.square().sum())
        baseline_squared += float(a.abs().double().square().sum())
        maximum = max(maximum, float(delta.max()) if delta.numel() else 0.0)
    return {'max_abs':maximum, 'relative_l2':(squared/baseline_squared)**0.5 if baseline_squared else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--case-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experiment-prefix', required=True)
    parser.add_argument('--device', type=torch.device, required=True)
    args = parser.parse_args()
    if not os.environ.get('CAR_ROUND6_LEASE_TOKEN') or args.device.type != 'cuda':
        parser.error('a measured Round 6 CUDA preparation lease is required')
    started_job = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    base = yaml.safe_load(args.config.read_text())
    steps = int(base['optimization']['steps_per_epoch'])
    if steps != 12:
        raise ValueError('this matched Round 6 screening requires 12 updates per reporting epoch')
    if base['optimization'].get('direct_supervision'):
        raise ValueError('this comparison requires the existing SSS-only policy')
    torch.set_num_threads(4)
    device = args.device
    receipt = {'schema':'car_round6_batched_epoch_timing_v1','status':'running',
        'config':str(args.config),'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'helper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'device':str(device),'device_name':torch.cuda.get_device_name(device),
        'scope':'fresh SOURCE, fresh AdamW/controller, matched configured RNG and sampler; two one-update warmups then four independent complete 12-update epochs with callbacks; no learning efficacy claim',
        'timing_limits':'Trainer epoch training time includes data waits and update work; callback/reporting time is separate. The outer lease records occupied GPU and CPU work; it is not additional to whole-job wall time.',
        'order':['separate','batched','batched','separate'],'attempts':[]}

    def write():
        temporary = args.output / 'receipt.tmp'
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
        os.replace(temporary, args.output/'receipt.json')

    def trial(method, label, count):
        cfg = deepcopy(base)
        cfg['runtime']['device'] = str(device)
        cfg['optimization']['multitask']['coherence_aggregation']['family_vjp'] = method
        cfg['output']['experiment_name'] = f'{args.experiment_prefix}/{label}'
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        run = run_post_training(cfg, case_dir=args.case_dir, max_steps=count)
        torch.cuda.synchronize(device)
        item = {'label':label,'family_vjp':method,'run':str(run),
            'attempted_updates':count,'enclosing_wall_seconds':time.monotonic()-started,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
            'initial_state_sha256':initial_state_digest(Path(run)/'checkpoints'/'epoch_000.pt')}
        if count == steps:
            path = Path(run)/'logs'/'epoch_timing.jsonl'
            rows = [json.loads(row) for row in path.read_text().splitlines() if row.strip()]
            if not rows or rows[-1]['attempted_updates'] != steps or rows[-1]['accepted_updates'] != steps:
                raise RuntimeError('trial did not complete all accepted updates')
            item['epoch_timing'] = rows[-1]
        return item

    write()
    try:
        for method in ('separate','batched'):
            receipt['attempts'].append(trial(method, f'Warmup_{method}', 1)); write()
        for index, method in enumerate(receipt['order'], 1):
            receipt['attempts'].append(trial(method, f'MatchedEpoch_{index:02d}_{method}', steps)); write()
        fingerprints = {row['initial_state_sha256'] for row in receipt['attempts']}
        if len(fingerprints) != 1:
            raise ValueError('initial model, optimizer, controller, RNG, or sampler state differs across attempts')
        epochs = [row for row in receipt['attempts'] if 'epoch_timing' in row]
        reference = Path(epochs[0]['run'])/'checkpoints'/'last.pt'
        names = json.loads((Path(epochs[0]['run'])/'artifacts'/'trainable_parameters.json').read_text())['names']
        for row in epochs[1:]:
            candidate_names = json.loads((Path(row['run'])/'artifacts'/'trainable_parameters.json').read_text())['names']
            if candidate_names != names:
                raise ValueError('trainable parameter order differs across trials')
        receipt['final_model_vs_first_separate'] = {
            row['label']:final_model_comparison(reference,Path(row['run'])/'checkpoints'/'last.pt',names)
            for row in epochs[1:]}
        times = {method:[entry['epoch_timing']['wall_seconds'] for entry in receipt['attempts']
                         if entry['family_vjp']==method and 'epoch_timing' in entry]
                 for method in ('separate','batched')}
        receipt.update(status='completed', epoch_wall_seconds=times,
            identical_initial_state_sha256=fingerprints.pop(),
            mean_wall_seconds={k:sum(v)/len(v) for k,v in times.items()},
            interpretation='two paired-order complete-epoch timings per method; compare within-method spread and memory; initial SOURCE qualification, not a mature trajectory or efficacy result')
    except BaseException as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt['whole_job_wall_seconds'] = time.monotonic() - started_job
        write()


if __name__ == '__main__':
    main()
