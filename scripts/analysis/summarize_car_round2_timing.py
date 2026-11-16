#!/usr/bin/env python3
"""Create a measured B64 update-time receipt from a qualified resumed run.

Epoch timers span datasource waits, sampler, rollout, ConFIG, clipping and
AdamW. Reporting/checkpoint callbacks are recorded separately. This CPU-only
summarizer never opens VALIDATION or TEST and never polls an ongoing run.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / 'src'))
from phycoflow_reconstruction.evaluation.adaptive_campaign import digest, immutable_json


def summarize(*, formal_config, run, qualification, pause_launch, resume_launch, pause_completion, resume_completion):
    import yaml
    started, cpu = time.time(), time.process_time()
    run = Path(run).resolve(strict=True)
    q = json.loads(Path(qualification).read_text())
    pauses = json.loads(Path(pause_launch).read_text())
    resumed = json.loads(Path(resume_launch).read_text())
    completions = [json.loads(Path(path).read_text()) for path in (pause_completion, resume_completion)]
    config = yaml.safe_load(Path(formal_config).read_text())
    if (q.get('schema') != 'car_round2_qualification_v1' or q.get('passed') is not True
            or q.get('formal_config_sha256') != digest(formal_config)
            or Path(q['resumed']).resolve() != run
            or q.get('pause_step') != 108 or q.get('final_step') != 144
            or q.get('pause_launch_sha256') != digest(pause_launch)
            or q.get('pause_completion_sha256') != digest(pause_completion)
            or q.get('resume_launch_sha256') != digest(resume_launch)
            or q.get('checkpoint_sha256', [None, None])[1] != digest(run/'checkpoints/last.pt')):
        raise ValueError('passing matching fresh-process 9+3 qualification is required')
    if (pauses.get('gpu_uuid') != resumed.get('gpu_uuid') or not pauses.get('gpu_uuid')
            or pauses.get('arm') != q.get('arm') or resumed.get('arm') != q.get('arm')
            or Path(resumed.get('resume','')).resolve() != run):
        raise ValueError('pause and fresh-process resume must use the same physical GPU and arm')
    walls = []
    for launch, completion, expected in zip((pauses, resumed), completions,
                                             (('budget_paused', 108), ('completed', 144)), strict=True):
        start, end = launch.get('started_unix'), completion.get('finished_unix')
        if (completion.get('token') != launch.get('token') or
                (completion.get('status'), completion.get('global_step')) != expected or
                Path(completion.get('run_dir', '')).resolve() != run or
                not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or
                not math.isfinite(start) or not math.isfinite(end) or end <= start):
            raise ValueError('matching completed pause and resume lease intervals required')
        walls.append(end - start)
    if (config.get('optimization',{}).get('batch_size') != 64
            or config['optimization'].get('steps_per_epoch') != 12
            or config['optimization'].get('epochs') != 1000):
        raise ValueError('formal config must declare the mature B64/12 protocol')
    parameterization = 'residual_head' if config.get('residual',{}).get('enabled') else 'full_model'
    if q['arm'].endswith('Residual_Repair') != (parameterization == 'residual_head'):
        raise ValueError('qualification arm and formal parameterization differ')
    status = json.loads((run/'status.json').read_text())
    if status.get('status') != 'completed' or status.get('global_step') != 144:
        raise ValueError('qualified timing run must be terminal at 144 accepted updates')
    timing_path = run/'logs/epoch_timing.jsonl'
    records = [json.loads(line) for line in timing_path.read_text().splitlines() if line.strip()]
    if len(records) != 12 or [row.get('epoch') for row in records] != list(range(1,13)):
        raise ValueError('complete ordered epoch timing journal is required')
    for row in records:
        if (row.get('version') != 'complete_epoch_timing_v1' or row.get('step') != row['epoch']*12
                or row.get('accepted_updates') != 12):
            raise ValueError('each epoch timing must cover exactly 12 accepted updates')
        values = [row.get(key) for key in ('train_seconds','wall_seconds','callback_and_reporting_seconds')]
        if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v < 0 for v in values) or values[0] <= 0 or values[1] < values[0] or not math.isclose(values[1]-values[0], values[2], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError('invalid measured training/callback timing')
    selected = [row for row in records if row['epoch'] in (*range(2,9),11)]
    # Omit cold first epoch, pause, resume initialization and final closeout;
    # use the maximum measured steady epoch for a conservative forecast.
    update_seconds = max(row['train_seconds']/row['accepted_updates'] for row in selected)
    lower_observed_seconds = min(row['train_seconds']/row['accepted_updates'] for row in selected)
    outside_epoch_seconds = max(0., sum(walls) - sum(row['wall_seconds'] for row in records))
    cold_train_excess = sum(max(0., records[epoch-1]['train_seconds'] - 12*update_seconds) for epoch in (1, 10))
    source = Path(config['source_run'])/'checkpoints/last.pt'
    return {'schema':'car_round2_complete_update_timing_v1',
            'source_checkpoint_sha256':digest(source), 'config_sha256':digest(formal_config),
            'gpu_uuid':pauses['gpu_uuid'], 'parameterization':parameterization, 'batch_size':64,
            'measured_updates':sum(row['accepted_updates'] for row in selected),
            'complete_update_seconds':update_seconds, 'selected_epochs':[row['epoch'] for row in selected],
            'minimum_observed_steady_update_seconds':lower_observed_seconds,
            'outside_epoch_seconds':outside_epoch_seconds, 'cold_train_excess_seconds':cold_train_excess,
            'lease_wall_seconds':walls,
            'per_epoch':records, 'callback_seconds_by_epoch':[row['callback_and_reporting_seconds'] for row in records],
            'qualification_receipt_sha256':digest(qualification), 'pause_launch_sha256':digest(pause_launch),
            'resume_launch_sha256':digest(resume_launch), 'epoch_timing_sha256':digest(timing_path),
            'pause_completion_sha256':digest(pause_completion), 'resume_completion_sha256':digest(resume_completion),
            'qualification_receipt_path':str(Path(qualification).resolve()),
            'pause_launch_path':str(Path(pause_launch).resolve()), 'resume_launch_path':str(Path(resume_launch).resolve()),
            'pause_completion_path':str(Path(pause_completion).resolve()), 'resume_completion_path':str(Path(resume_completion).resolve()),
            'epoch_timing_path':str(timing_path.resolve()),
            'scope':'maximum steady epoch train wall divided by accepted updates; includes data wait, sampler, source/head, ConFIG, clipping and AdamW; excludes callbacks',
            'callback_scope':'per-epoch additional reporting, plotting, validation and checkpoint wall; not silently charged to update timer',
            'outside_epoch_scope':'sum of both measured GPU lease walls minus the twelve epoch walls, floored at zero; includes both startup/closure processes',
            'start_unix':started, 'end_unix':time.time(), 'cpu_seconds':time.process_time()-cpu}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--formal-config',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--pause-launch',type=Path,required=True)
    p.add_argument('--resume-launch',type=Path,required=True)
    p.add_argument('--pause-completion',type=Path,required=True)
    p.add_argument('--resume-completion',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    result = summarize(formal_config=a.formal_config, run=a.run, qualification=a.qualification,
                       pause_launch=a.pause_launch,resume_launch=a.resume_launch,
                       pause_completion=a.pause_completion,resume_completion=a.resume_completion)
    immutable_json(a.output,result)
    print(a.output)

if __name__ == '__main__':
    main()
