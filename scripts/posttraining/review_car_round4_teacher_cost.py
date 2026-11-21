#!/usr/bin/env python3
"""Create one immutable warm teacher cost review after every GPU lease closes."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import statistics
import time
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def bind(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha(path)}


def read(path):
    return json.loads(Path(path).read_text())


def number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{label} must be finite and nonnegative')
    return float(value)


def calculate(history, writes, completed, resumed_cpu, teacher_wall, teacher_cpu, other_candidate, other_whole, committed):
    """Forecast using the last five epochs of exactly one completed resumed lease."""
    if not 5 <= len(writes) <= completed < 100 or len(history) != completed:
        raise ValueError('need at least five completed epochs in the resumed lease and a partial 100-epoch teacher')
    if [r.get('epoch') for r in history] != list(range(1, completed + 1)):
        raise ValueError('teacher history is not a complete ordered epoch sequence')
    lease_rows = history[completed-len(writes):]
    epoch_cpu = [number(r['cpu_seconds'], 'epoch CPU') for r in lease_rows]
    epoch_wall = [number(r['elapsed_seconds'], 'epoch wall') for r in lease_rows]
    writes = [number(x, 'checkpoint write wall') for x in writes]
    resumed_cpu = number(resumed_cpu, 'resumed lease CPU')
    teacher_wall = number(teacher_wall, 'teacher GPU wall')
    teacher_cpu = number(teacher_cpu, 'teacher CPU')
    other_candidate = number(other_candidate, 'other candidate GPU')
    other_whole = number(other_whole, 'other whole GPU')
    committed = number(committed, 'reserve committed elsewhere')
    if committed > 18000 or other_whole < other_candidate or teacher_cpu < resumed_cpu:
        raise ValueError('inconsistent reserve, other costs or teacher CPU')
    overhead = max(0., resumed_cpu - sum(epoch_cpu)) / len(lease_rows)
    remaining = 100-completed
    mw, mc, mp = map(statistics.mean, (epoch_wall[-5:], epoch_cpu[-5:], writes[-5:]))
    projected_gpu = teacher_wall + 60 + 1.2*remaining*(mw+mp) + 1599
    projected_cpu = teacher_cpu + 60 + 1.2*remaining*(mc+overhead) + 1698
    effective = min(14400+18000-committed, 35317-other_candidate, 86400-other_whole)
    return {'schema':'car_round4_teacher_warm_cache_forecast_v1',
            'completed_epochs':completed, 'remaining_epochs':remaining, 'safety_factor':1.2,
            'base_teacher_gpu_allowance_seconds':14400, 'variation_reserve_gpu_seconds':18000,
            'reserve_committed_elsewhere_gpu_seconds':committed, 'candidate_gpu_cap_seconds':35317,
            'whole_gpu_cap_seconds':86400, 'teacher_cpu_cap_seconds':21600,
            'candidate_other_gpu_seconds':other_candidate, 'whole_other_gpu_seconds':other_whole,
            'teacher_gpu_cap_seconds':effective, 'projected_teacher_gpu_seconds':projected_gpu,
            'projected_teacher_cpu_worker_seconds':projected_cpu,
            'projected_candidate_gpu_seconds':other_candidate+projected_gpu,
            'projected_whole_gpu_seconds':other_whole+projected_gpu,
            'training_window_epochs':[r['epoch'] for r in lease_rows[-5:]],
            'window':{'epochs':[r['epoch'] for r in lease_rows[-5:]], 'update_wall_seconds':epoch_wall[-5:],
                      'epoch_cpu_seconds':epoch_cpu[-5:], 'checkpoint_write_wall_seconds':writes[-5:],
                      'mean_update_wall_seconds':mw, 'mean_epoch_cpu_seconds':mc,
                      'mean_checkpoint_write_wall_seconds':mp, 'resumed_lease_cpu_seconds':resumed_cpu,
                      'resumed_lease_epoch_cpu_seconds':sum(epoch_cpu), 'resumed_lease_epochs':len(lease_rows),
                      'cpu_persistence_overhead_per_epoch_seconds':overhead},
            'closed_teacher_gpu_seconds':teacher_wall, 'closed_teacher_cpu_worker_seconds':teacher_cpu,
            'future_startup_allowance_seconds':60, 'future_closeout_wall_seconds':1599,
            'future_closeout_cpu_seconds':1698,
            'interpretation':'Planning forecast only; no efficacy, stopping, resource admission or adoption decision.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--attempt', required=True)
    p.add_argument('--resumed-token', required=True)
    p.add_argument('--review-name', required=True)
    p.add_argument('--execution-code-receipt', type=Path, required=True,
                   help='Reviewed operational continuation capsule receipt')
    p.add_argument('--reserve-committed-elsewhere', type=float, required=True)
    args=p.parse_args()
    import re
    for value in (args.attempt, args.resumed_token, args.review_name):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{1,127}', value):
            p.error('attempt, token and review name must be descriptive path components')
    campaign=args.campaign.resolve(strict=True)
    with (campaign/'operations/gpu_ledger.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        review(campaign,args)


def review(c, args):
    started_wall, started_cpu = time.time(), time.process_time()
    ledger_path=c/'operations/gpu_ledger.json'
    ledger=read(ledger_path)
    if ledger.get('schema') != 'car_gpu_ledger_v1' or ledger.get('active'):
        raise ValueError('closed-phase review requires every GPU lease to be closed')
    # The checks above intentionally precede all checkpoint, history and job reads.
    initial_path=c/'operations/feasibility_gate.json'
    launch_path=c/'operations/teacher_launch_forecast.json'
    initial, launch=read(initial_path), read(launch_path)
    manifest_path=c/'configs/train_manifest.json'
    manifest_hash=sha(manifest_path)
    if (initial.get('schema') != 'car_round4_feasibility_gate_v1' or initial.get('admitted') is not True
            or launch.get('schema') != 'car_round4_teacher_launch_forecast_v1' or launch.get('admitted') is not True
            or initial.get('manifest_sha256') != manifest_hash or launch.get('manifest_sha256') != manifest_hash
            or launch['bindings']['initial_gate']['sha256'] != sha(initial_path)):
        raise ValueError('initial and teacher-launch forecasts must be admitted and manifest-bound')
    tokens=set(); rows=[]; teacher_receipts=[]
    for entry in ledger['closed']:
        token=entry['token']
        if token in tokens: raise ValueError('duplicate closed lease token')
        tokens.add(token)
        if entry['phase'] != 'endpoint_teacher': continue
        path=c/'operations/teacher_jobs'/f'{token}.json'
        receipt=read(path)
        if (receipt.get('schema') != 'car_round4_teacher_job_v1' or receipt.get('token') != token
                or receipt.get('manifest_sha256') != manifest_hash or receipt.get('status') not in ('failed','paused','completed')
                or abs(number(receipt['wall_seconds'],'receipt wall')-number(entry['seconds'],'ledger wall')) > .01
                or 'cpu_seconds' not in entry):
            raise ValueError('closed teacher receipt or timer does not match ledger')
        rows.append(entry); teacher_receipts.append((path,receipt))
    chosen=[(e,p,r) for e,(p,r) in zip(rows,teacher_receipts) if e['token']==args.resumed_token]
    if len(chosen)!=1 or chosen[0][2]['status']!='paused':
        raise ValueError('named resumed teacher lease must be closed and paused')
    selected,selected_path,selected_receipt=chosen[0]
    if selected['token'] != rows[-1]['token']:
        raise ValueError('named resumed lease is not the latest teacher lease')
    run=c/'runs/formal/SSS_Empirical_Endpoint_Teacher'/args.attempt
    run=run.resolve(strict=True)
    if not run.is_relative_to((c/'runs/formal/SSS_Empirical_Endpoint_Teacher').resolve()):
        raise ValueError('attempt outside teacher campaign')
    checkpoint=run/'checkpoints/teacher_latest.pt'
    history=run/'logs/teacher_history.jsonl'
    config=run/'configs/resolved_scientific_config.json'
    if Path(selected_receipt['checkpoint']).resolve() != checkpoint.resolve():
        raise ValueError('paused lease checkpoint does not match the selected attempt')
    review_dir=c/'assessment/teacher_cost_reviews'/args.review_name
    review_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_copy=review_dir/'teacher_checkpoint.pt'
    history_copy=review_dir/'teacher_history.jsonl'
    config_copy=review_dir/'resolved_scientific_config.json'
    # Atomic checkpoint replacement leaves the linked inode immutable for review.
    os.link(checkpoint, checkpoint_copy)
    shutil.copyfile(history, history_copy)
    shutil.copyfile(config, config_copy)
    if not os.path.samefile(checkpoint_copy, checkpoint) or sha(history_copy)!=sha(history) or sha(config_copy)!=sha(config):
        raise ValueError('review snapshot differs from closed teacher files')
    scientific=read(config_copy)
    identity=scientific['identity']
    if scientific.get('schema')!='car_round4_teacher_scientific_config_v1' or identity.get('train_manifest_sha256')!=manifest_hash:
        raise ValueError('saved scientific definitions differ from manifest')
    import torch
    saved=torch.load(checkpoint_copy,map_location='cpu',weights_only=True,mmap=True)
    completed=saved['completed_epochs']
    history_rows=[json.loads(line) for line in history_copy.read_text().splitlines()]
    if (saved.get('schema')!='car_round4_teacher_checkpoint_v1' or completed!=selected_receipt.get('completed_epochs')
            or saved.get('history') != history_rows or any(saved.get(k)!=v for k,v in identity.items())
            or saved.get('normalizers')!=scientific.get('normalizers')):
        raise ValueError('checkpoint, history, paused receipt or scientific definitions disagree')
    # The reviewed launch forecast charges each nonteacher cost once; require
    # its measured closed leases to be unchanged before replacing teacher cost.
    nonteacher=sum(number(e['seconds'],'nonteacher ledger wall') for e in ledger['closed'] if e['phase']!='endpoint_teacher')
    previous=number(launch['measured_qualification_gpu_seconds'],'qualification')+number(launch['measured_generation_gpu_seconds'],'generation')
    if abs(nonteacher-previous)>.01:
        raise ValueError('nonteacher ledger differs from launch forecast; refresh costs before review')
    old_failed = {Path(ref['path']).resolve():ref for ref in launch['bindings']['failed_startups']}
    failed = {p.resolve() for p in (c/'operations/failed_launches').glob('*.json')}
    if failed != set(old_failed) or any(sha(path) != ref['sha256'] for path, ref in old_failed.items()):
        raise ValueError('failed startup costs changed since the teacher launch forecast')
    original_teacher=number(launch['remaining_gpu_forecast_seconds']['teacher'],'initial teacher forecast')
    other_candidate=number(launch['candidate_inclusive_gpu_seconds'],'candidate forecast')-original_teacher
    other_whole=number(launch['whole_round_gpu_seconds'],'whole forecast')-original_teacher
    result=calculate(history_rows, selected_receipt['checkpoint_write_wall_seconds'], completed,
                     selected['cpu_seconds'], sum(number(x['seconds'],'teacher wall') for x in rows),
                     sum(number(x['cpu_seconds'],'teacher CPU') for x in rows), other_candidate, other_whole,
                     args.reserve_committed_elsewhere)
    capsules=[]
    execution_receipt = args.execution_code_receipt.resolve(strict=True)
    if not execution_receipt.is_relative_to(c):
        raise ValueError('execution code receipt must belong to campaign')
    next_execution = read(execution_receipt)
    next_code = Path(next_execution['execution_code_root']).resolve(strict=True)
    if not next_code.is_relative_to((c/'operations/code').resolve()):
        raise ValueError('continuation code receipt has an invalid capsule')
    next_inventory = read(next_code/'execution_receipt.json')
    if (next_inventory.get('schema') != 'car_execution_code_v1' or next_inventory.get('sha256') != next_code.name
            or hashlib.sha256(json.dumps(next_inventory['files'],sort_keys=True).encode()).hexdigest() != next_code.name
            or sha(next_code/'scripts/posttraining/run_car_round4_teacher.py') != next_inventory['files']['scripts/posttraining/run_car_round4_teacher.py']):
        raise ValueError('continuation code capsule identity differs')
    for _, receipt in teacher_receipts:
        code=Path(receipt['code_root']).resolve(strict=True)
        inventory=read(code/'execution_receipt.json')
        if (not code.is_relative_to((c/'operations/code').resolve()) or inventory.get('schema')!='car_execution_code_v1'
                or inventory.get('sha256')!=code.name or hashlib.sha256(json.dumps(inventory['files'],sort_keys=True).encode()).hexdigest()!=code.name
                or inventory['files'].get('scripts/posttraining/run_car_round4_teacher.py')!=receipt['script_sha256']
                or sha(code/'scripts/posttraining/run_car_round4_teacher.py')!=receipt['script_sha256']):
            raise ValueError('teacher execution capsule identity differs')
        capsules.append(bind(code/'execution_receipt.json'))
    checkpoint_binding, history_binding = bind(checkpoint_copy), bind(history_copy)
    result.update(train_manifest_sha256=manifest_hash, teacher_checkpoint_sha256=checkpoint_binding['sha256'],
                  teacher_history_sha256=history_binding['sha256'], attempt=args.attempt, resumed_token=args.resumed_token,
                  review_tool_sha256=sha(__file__),
                  bindings={'initial_gate':bind(initial_path), 'teacher_launch_forecast':bind(launch_path),
                            'manifest':bind(manifest_path), 'ledger':bind(ledger_path),
                            'checkpoint':checkpoint_binding, 'history':history_binding,
                            'scientific_config':bind(config_copy), 'teacher_receipts':[bind(p) for p,_ in teacher_receipts],
                            'execution_capsules':capsules, 'continuation_code_receipt':bind(execution_receipt),
                            'continuation_capsule_receipt':bind(next_code/'execution_receipt.json')})
    result['review_timer']={'start_unix':started_wall, 'end_unix':time.time(),
                            'process_cpu_seconds':time.process_time()-started_cpu,
                            'scope':'standalone CPU review from ledger read through input verification; excludes output serialization; outside teacher lease'}
    out=review_dir/'warm_forecast.json'
    with out.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    print(out)


if __name__=='__main__': main()
