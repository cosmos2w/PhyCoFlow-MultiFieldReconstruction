#!/usr/bin/env python3
"""Replay independently reset mature-state complete epochs for executor timing.

Uses disposable case-owned forks. A fork retains parent model, optimizer, RNG,
query sampler and PI state, substituting the explicitly supplied common new-law
reference for this diagnostic. It is not a scientific training continuation.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
import torch
import yaml
from phycoflow_reconstruction.training.run_store import RunStore, file_sha256


def worker(args):
    from phycoflow_reconstruction.training.post_training import run_post_training
    torch.set_num_threads(4)
    cfg = yaml.safe_load(args.config.read_text())
    if cfg['optimization']['multitask']['fidelity_budget'].get('reference_law') != 'training_selected_complement_v1':
        raise ValueError('new selected-complement reference law required')
    reference_decl = cfg['optimization']['multitask']['fidelity_budget']['reference_artifact']
    if file_sha256(reference_decl['path']) != reference_decl['sha256']:
        raise ValueError('reference artifact hash mismatch')
    reference = json.loads(Path(reference_decl['path']).read_text())
    parent = args.parent.resolve(strict=True)
    checkpoint_path = parent/'checkpoints/epoch_500.pt'
    if file_sha256(checkpoint_path) != args.parent_sha256:
        raise ValueError('parent checkpoint identity mismatch')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    origin = int(checkpoint['global_step'])
    if origin != 6000 or checkpoint['external_policy']['completed_updates'] != origin or checkpoint['scalable_runtime']['step'] != origin:
        raise ValueError('expected mature update-6000 checkpoint')
    if checkpoint['training_monitor']['step'] != origin or checkpoint['training_monitor']['batch_count'] != 0:
        raise ValueError('parent checkpoint must be an epoch boundary')
    agg = cfg['optimization']['multitask']['coherence_aggregation']
    agg['family_vjp'] = args.method
    if args.method == 'factored': agg['vjp_chunk_size'] = args.chunk
    else: agg.pop('vjp_chunk_size', None)
    cfg['optimization'].setdefault('rollout_execution', {})['context_cache'] = args.context_cache
    cfg['runtime']['device'] = str(args.device)
    cfg['output']['experiment_name'] = f'{args.experiment_prefix}/{args.label}'
    settings = cfg['coherence']['scalable_anchored_coherence']
    checkpoint['scalable_runtime']['scale_objective'] = settings.get('scale_objective', 'composite')
    checkpoint['scalable_runtime']['scale_joint_kappa'] = settings.get('scale_joint_kappa', 1.0)
    checkpoint['external_policy']['reference'] = reference['reference']
    # Initialize artifacts under the actual diagnostic config instead of
    # copying the parent's differently bound provenance into the new run.
    run = run_post_training(cfg,case_dir=args.case_dir,prepare_only=True)
    store = RunStore.resume(run,cfg)
    prepared_manifest = json.loads((store.run_dir/'run_manifest.json').read_text())
    core = torch.load(store.run_dir/'artifacts/scalable_coherence.pt',map_location='cpu',weights_only=True)
    parent_core = torch.load(parent/'artifacts/scalable_coherence.pt',map_location='cpu',weights_only=True)
    from phycoflow_reconstruction.training.scalable_fitting import _digest
    def frozen_core(artifact):
        selected = {k:deepcopy(artifact[k]) for k in ('state_bank','shape_artifact','priorities','units','active','adaptive_bank') if k in artifact}
        selected['shape_artifact']['provenance'].pop('config_sha256',None)
        return selected
    if _digest(frozen_core(core)) != _digest(frozen_core(parent_core)):
        raise ValueError('prepared diagnostic changed fitted coherence values')
    if checkpoint['family_scales'] != prepared_manifest['coherence_family_scales']:
        raise ValueError('prepared diagnostic changed family scales')
    checkpoint['scalable_runtime']['provenance'] = deepcopy(core['provenance'])
    checkpoint['scalable_runtime']['artifact_sha256'] = core['sha256']
    checkpoint['coherence_calibration_sha256'] = prepared_manifest['coherence_calibration_sha256']
    checkpoint['config_sha256'] = store.config_hash
    checkpoint['fork_origin'] = {'parent_run':str(parent), 'parent_checkpoint_sha256':args.parent_sha256,
        'absolute_step':origin, 'scientific_transition':'round7_executor_cost_qualification_only'}
    store.save_checkpoint('last', checkpoint)
    store.update_manifest(fork_origin=checkpoint['fork_origin'])
    store.set_status('fork_initialized', global_step=origin, reporting_epoch=0, absolute_reporting_epoch=origin/12)
    start = time.monotonic()
    run = run_post_training(cfg, case_dir=args.case_dir, resume=store.run_dir,
                            budget_stop_requested=lambda step, _run: step >= origin+12)
    timing = [json.loads(line) for line in (run/'logs/epoch_timing.jsonl').read_text().splitlines() if line.strip()]
    if len(timing) != 1 or timing[0]['step'] != origin+12 or timing[0]['timed_updates'] != 12 or not timing[0]['complete_epoch_duration']:
        raise ValueError('qualification fork did not complete the 12-update epoch')
    result = {'label':args.label, 'method':args.method, 'chunk':args.chunk,
        'context_cache':args.context_cache, 'run':str(run),
        'wall_seconds':time.monotonic()-start, 'epoch_timing':timing[0],
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(args.device),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(args.device)}
    args.output.write_text(json.dumps(result, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', required=True, type=Path)
    p.add_argument('--parent-sha256', required=True)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--case-dir', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--experiment-prefix', required=True)
    p.add_argument('--device', required=True, type=torch.device)
    p.add_argument('--finalist', choices=('batched','factored'), default='factored')
    p.add_argument('--finalist-context-cache', choices=('none','condition','static_features'), default='none')
    p.add_argument('--chunk', type=int, choices=(1,2,3), default=1)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--label')
    p.add_argument('--method', choices=('separate','batched','factored'))
    p.add_argument('--context-cache', choices=('none','condition','static_features'), default='none', help=argparse.SUPPRESS)
    a=p.parse_args()
    if not os.environ.get('CAR_ROUND7_LEASE_TOKEN') or a.device.type != 'cuda': p.error('Round 7 CUDA preparation lease required')
    if a.worker: return worker(a)
    a.output.mkdir(parents=True, exist_ok=False)
    started=time.monotonic()
    receipt={'schema':'car_round7_executor_epoch_timing_v1', 'status':'failed',
        'scope':'balanced duplicate 12-update parent-state epochs including normal callbacks; new-law reference substituted for all trials',
        'parent_sha256':a.parent_sha256, 'config_sha256':file_sha256(a.config),
        'baseline_context_cache':'none', 'finalist_context_cache':a.finalist_context_cache, 'attempts':[]}
    try:
        for index, method in enumerate(('separate',a.finalist,a.finalist,'separate'),1):
            label=f'Epoch_{index:02d}_{method}'
            result=a.output/f'{label}.json'
            cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--parent',str(a.parent),
                '--parent-sha256',a.parent_sha256,'--config',str(a.config),'--case-dir',str(a.case_dir),
                '--output',str(result),'--experiment-prefix',a.experiment_prefix,'--device',str(a.device),
                '--label',label,'--method',method,'--chunk',str(a.chunk),
                '--context-cache',a.finalist_context_cache if method == a.finalist else 'none']
            with (a.output/f'{label}.log').open('x') as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            receipt['attempts'].append(json.loads(result.read_text()))
        keys=('step','sample_ids','observation_sha256','query_sha256','member_seeds',
              'first_member_generator_state_sha256','field_complement_sha256')
        records=[]
        for item in receipt['attempts']:
            path=Path(item['run'])/'metrics/training_coupling.jsonl'
            rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if len(rows)!=12: raise ValueError('expected exactly twelve coupling records')
            records.append([{key:row.get(key) for key in keys} for row in rows])
        if any(rows != records[0] for rows in records[1:]):
            raise ValueError('restored sample/query/observation/member/complement streams differ')
        if file_sha256(a.parent/'checkpoints/epoch_500.pt') != a.parent_sha256:
            raise ValueError('parent checkpoint changed during screen')
        receipt.update(status='completed', matched_coupling=True, parent_checkpoint_unchanged=True)
    except BaseException as error:
        receipt.update(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        receipt['wall_seconds']=time.monotonic()-started
        (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__': main()
