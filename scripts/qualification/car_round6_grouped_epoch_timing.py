#!/usr/bin/env python3
"""Compare matched fresh-SOURCE complete epochs with shared and eight-group queries."""
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
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
import torch
import yaml
from phycoflow_reconstruction.training.post_training import run_post_training
from scripts.qualification.car_round6_batched_epoch_timing import final_model_comparison


def semantic_digest(state):
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
        elif isinstance(value, (list, tuple)):
            digest.update(type(value).__name__.encode()+b'[')
            for item in value: add(item)
            digest.update(b']')
        elif value is None or isinstance(value, (str, int, float, bool)):
            digest.update(f'{type(value).__name__}:{value!r};'.encode())
        else:
            raise TypeError(f'unsupported initial-state value {type(value).__name__}')
    add(state)
    return digest.hexdigest()


def common_initial_state_digest(path):
    """Hash epoch-zero state except query-law and derived hash metadata."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['global_step'] != 0:
        raise ValueError('expected fresh SOURCE epoch-zero checkpoint')
    state = deepcopy(state)
    state.pop('config_sha256')  # Configs differ in query law and attempt output path.
    runtime = state['scalable_runtime']
    runtime.pop('condition_query_groups')
    runtime.pop('group_samplers')
    runtime.pop('artifact_sha256')  # Compare normalized artifact content separately.
    runtime['provenance'].pop('config_sha256')
    return semantic_digest(state)


def common_artifact_digest(path):
    """Compare all scalable scientific artifact contents, apart from derived hashes."""
    artifact = deepcopy(torch.load(path, map_location='cpu', weights_only=False))
    artifact.pop('sha256')
    artifact['provenance'].pop('config_sha256')
    # The two provenance dictionaries may share identity in the saved artifact.
    artifact['shape_artifact']['provenance'].pop('config_sha256', None)
    return semantic_digest(artifact)


def read_coupling(run, expected_count):
    """Read existing trainer evidence for paired conditions, observations and member RNG."""
    path = Path(run)/'metrics'/'training_coupling.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != expected_count:
        raise ValueError('training coupling record count does not match accepted updates')
    fields = ('step','sample_ids','observation_sha256','query_sha256',
              'member_seeds','first_member_generator_state_sha256','accepted')
    result = [{key:row[key] for key in fields} for row in rows]
    if any(not row['accepted'] or len(row['member_seeds']) != 2 for row in result):
        raise ValueError('all actions and exactly two member RNG identities are required')
    return result


def completed_epoch_record(rows, steps):
    if (not rows or rows[-1]['attempted_updates'] != steps or rows[-1]['accepted_updates'] != steps
        or rows[-1].get('complete_epoch_duration') is False):
        raise RuntimeError('trial did not record a complete epoch duration with all accepted updates')
    return rows[-1]


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
    if steps != 12 or base['optimization'].get('direct_supervision'):
        raise ValueError('12-step SSS-only Round 6 configuration required')
    family_vjp = base['optimization']['multitask']['coherence_aggregation'].get('family_vjp','separate')
    torch.set_num_threads(4)
    device = args.device
    receipt = {'schema':'car_round6_grouped_epoch_timing_v1','status':'running',
        'config':str(args.config),'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'helper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'device':str(device),'device_name':torch.cuda.get_device_name(device),
        'cpu_threads':torch.get_num_threads(),
        'cpu_affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        'thread_environment':{key:os.environ.get(key) for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS')},
        'family_vjp':family_vjp,'order':['shared','condition_groups_8','condition_groups_8','shared'],
        'scope':'fresh SOURCE, fresh AdamW and controller; two one-update warmups then four independent 12-update epochs, including optimizer, logging and callbacks',
        'state_fingerprint_excludes':['config_sha256 (query law and output path differ)','condition_query_groups','group_samplers','scalable_runtime.provenance.config_sha256','scalable_runtime.artifact_sha256'],
        'artifact_fingerprint_excludes':['sha256','provenance.config_sha256','shape_artifact.provenance.config_sha256'],
        'coupling_scope':'trainer records sample IDs, observation and spatial-query hashes, two member seeds and initial first-member RNG state; member predictions and final model trajectories may differ under the query laws',
        'timing_limits':'training time includes data waits and update work; callback/reporting time is separate. Whole-job and enclosing lease times overlap and must not be added.',
        'attempts':[]}

    def write():
        tmp=args.output/'receipt.tmp'
        tmp.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
        os.replace(tmp,args.output/'receipt.json')

    def trial(law,label,count):
        cfg=deepcopy(base)
        cfg['runtime']['device']=str(device)
        cfg['coherence']['scalable_anchored_coherence']['condition_query_groups'] = 1 if law == 'shared' else 8
        cfg['output']['experiment_name']=f'{args.experiment_prefix}/{label}'
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started=time.monotonic()
        run=run_post_training(cfg,case_dir=args.case_dir,max_steps=count)
        torch.cuda.synchronize(device)
        item={'label':label,'query_law':law,'run':str(run),'attempted_updates':count,
              'enclosing_wall_seconds':time.monotonic()-started,
              'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
              'common_initial_state_sha256':common_initial_state_digest(Path(run)/'checkpoints'/'epoch_000.pt'),
              'common_artifact_sha256':common_artifact_digest(Path(run)/'artifacts'/'scalable_coherence.pt'),
              'coupling':read_coupling(run,count)}
        if count==steps:
            rows=[json.loads(line) for line in (Path(run)/'logs'/'epoch_timing.jsonl').read_text().splitlines() if line.strip()]
            item['epoch_timing']=completed_epoch_record(rows,steps)
        return item

    write()
    try:
        def add_attempt(row):
            previous=receipt['attempts']
            error=None
            if previous:
                if row['common_initial_state_sha256'] != previous[0]['common_initial_state_sha256']:
                    error='common epoch-zero state differs across attempts'
                if row['common_artifact_sha256'] != previous[0]['common_artifact_sha256']:
                    error='common scalable scientific artifact contents differ across attempts'
                baseline=max(previous,key=lambda item:len(item['coupling']))['coupling']
                count=min(len(baseline),len(row['coupling']))
                if row['coupling'][:count] != baseline[:count]:
                    error='condition, observation, spatial query, or member RNG coupling differs'
            previous.append(row)
            write()
            if error:
                raise ValueError(error)
        for law in ('shared','condition_groups_8'):
            add_attempt(trial(law,f'Warmup_{law}',1))
        for index,law in enumerate(receipt['order'],1):
            add_attempt(trial(law,f'MatchedEpoch_{index:02d}_{law}',steps))
        fingerprints={row['common_initial_state_sha256'] for row in receipt['attempts']}
        if len(fingerprints)!=1:
            raise ValueError('common epoch-zero state differs across attempts')
        epochs=[row for row in receipt['attempts'] if 'epoch_timing' in row]
        reference_coupling=epochs[0]['coupling']
        for row in receipt['attempts']:
            if row['coupling'] != reference_coupling[:len(row['coupling'])]:
                raise ValueError('condition, observation, spatial query, or member RNG coupling differs')
        reference=Path(epochs[0]['run'])/'checkpoints'/'last.pt'
        names=json.loads((Path(epochs[0]['run'])/'artifacts'/'trainable_parameters.json').read_text())['names']
        for row in epochs[1:]:
            other=json.loads((Path(row['run'])/'artifacts'/'trainable_parameters.json').read_text())['names']
            if names!=other: raise ValueError('trainable parameter order differs')
        receipt['final_model_vs_first_shared']={
            row['label']:final_model_comparison(reference,Path(row['run'])/'checkpoints'/'last.pt',names)
            for row in epochs[1:]}
        times={law:[row['epoch_timing']['wall_seconds'] for row in epochs if row['query_law']==law]
               for law in ('shared','condition_groups_8')}
        receipt.update(status='completed',identical_common_initial_state_sha256=fingerprints.pop(),
            matching_training_coupling=True,epoch_wall_seconds=times,
            mean_wall_seconds={law:sum(values)/len(values) for law,values in times.items()},
            interpretation='two alternating complete epochs per query law; report variation and peak memory. Different query laws can change trajectories; this is a cost comparison, not an efficacy test.')
    except BaseException as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        receipt['whole_job_wall_seconds']=time.monotonic()-started_job
        write()


if __name__=='__main__':
    main()
