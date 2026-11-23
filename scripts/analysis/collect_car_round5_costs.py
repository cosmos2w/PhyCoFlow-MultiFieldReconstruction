#!/usr/bin/env python3
"""Collect terminal Round 5 costs from closed leases and existing journals.

Input JSON schema car_round5_cost_inputs_v1: arms maps each evaluated arm to
{"run_dir": "...", "lease_tokens": ["..."]} (lease_tokens is optional;
when absent, uniquely infer it from --arm arguments in completion commands).
Optional cpu_jobs is a list of {"path":"...", "kind":"assessment" or
"independent", "inside_gpu_lease":false}. These are separate process receipts,
not child CPU reported inside GPU leases. Optional assessment_windows declares
[{"gpu_tokens":[...], "cpu_paths":[...]}] for each review/terminal assessment.
It enables wall-span and final queue-tail accounting without counting the training
interval between checkpoints as assessment time. This script performs no profiling.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from phycoflow_reconstruction.evaluation.adaptive_campaign import immutable_json

ARMS=('Native_GT_Continuation','Native_GT_SSS_Safeguarded_Continuation')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def ref(path):
    p=Path(path).resolve(strict=True)
    return {'path':str(p),'sha256':sha(p)}


def read(path):return json.loads(Path(path).read_text())
def jsonl(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def measured(x):
    return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>=0


def sum_known(values):
    return sum(float(x) for x in values) if values and all(measured(x) for x in values) else None


def command_arms(command):
    """Find explicit arm identities even inside a nested frozen-driver command."""
    def walk(value):
        if isinstance(value,str):
            for arm in ARMS:
                if value==arm:yield arm
        elif isinstance(value,list):
            for item in value:yield from walk(item)
        elif isinstance(value,dict):
            for item in value.values():yield from walk(item)
    return set(walk(command))


def closed_leases(campaign):
    ledger_path=campaign/'operations/gpu_ledger.json';ledger=read(ledger_path)
    if ledger.get('active'):raise ValueError('GPU ledger has active leases; terminal costs require closed leases')
    records={};evidence=[ref(ledger_path)]
    for row in ledger['closed']:
        token=row.get('token')
        if not token or token in records or not measured(row.get('seconds')):
            raise ValueError('closed ledger has duplicate token or invalid elapsed seconds')
        receipt_path=campaign/'operations/completions'/f'{token}.json'
        receipt=read(receipt_path) if receipt_path.exists() else None
        if receipt is not None:
            if receipt.get('token')!=token or not measured(receipt.get('seconds')) or not math.isclose(receipt['seconds'],row['seconds'],rel_tol=1e-8,abs_tol=1e-5):
                raise ValueError(f'completion receipt differs from ledger: {token}')
            evidence.append(ref(receipt_path))
        records[token]={'ledger':row,'receipt':receipt}
    return records,evidence


def assign_arms(records,arms):
    result={};unassigned=[];explicit={}
    for arm,item in arms.items():
        if arm not in ARMS:raise ValueError(f'unknown arm: {arm}')
        for token in item.get('lease_tokens',[]):
            if token in explicit:raise ValueError('one lease cannot belong to two arms')
            if token not in records or records[token]['ledger'].get('phase')!='formal_training':
                raise ValueError(f'explicit token is not a formal training lease: {token}')
            explicit[token]=arm
    for token,entry in records.items():
        if entry['ledger'].get('phase')!='formal_training':continue
        receipt=entry['receipt']; inferred=command_arms(receipt.get('command',[])) if receipt else set()
        if token in explicit:
            arm=explicit[token]
            if inferred and inferred!={arm}:raise ValueError(f'explicit arm conflicts with command for {token}')
        elif len(inferred)==1:arm=next(iter(inferred))
        else:
            unassigned.append(token);continue
        if arm not in arms:unassigned.append(token);continue
        result.setdefault(arm,[]).append(token)
    return result,unassigned


def training(run, tokens, records, attributable):
    run=Path(run).resolve(strict=True)
    paths={k:run/'logs'/f'{k}.jsonl' for k in ('history','epoch_timing','direct_updates')}
    evidence=[ref(p) for p in paths.values() if p.exists()]
    history=jsonl(paths['history']) if paths['history'].exists() else []
    timing=jsonl(paths['epoch_timing']) if paths['epoch_timing'].exists() else []
    updates=jsonl(paths['direct_updates']) if paths['direct_updates'].exists() else []
    epochs=[r.get('epoch') for r in history]
    if epochs!=list(range(1,len(epochs)+1)) or any(r.get('epoch_complete') is False for r in history):
        raise ValueError(f'incomplete or noncontiguous completed history: {run}')
    if [r.get('epoch') for r in timing]!=epochs:
        raise ValueError(f'epoch timing does not match completed history: {run}')
    steps=[r.get('step') for r in updates]
    if steps!=list(range(1,len(steps)+1)):raise ValueError(f'direct attempt journal is incomplete: {run}')
    complete_steps=timing[-1]['step'] if timing else 0
    if updates and len(updates)<complete_steps:raise ValueError('completed epoch exceeds attempt journal')
    full=[r for r in updates if r['step']<=complete_steps]
    partial=len(updates)>complete_steps
    walls=[r.get('wall_seconds') for r in timing]
    whole_epoch_total=sum_known(walls)
    train=sum_known([r.get('train_seconds') for r in timing])
    callback=sum_known([r.get('callback_and_reporting_seconds') for r in timing])
    updates_all=sum_known([r.get('update_seconds') for r in updates])
    updates_complete=sum_known([r.get('update_seconds') for r in full])
    waits=sum_known([r.get('data_wait_seconds') for r in updates])
    loop=None
    if train is not None and updates_complete is not None and not partial:
        residual=train-updates_complete
        if residual < -1e-5:raise ValueError('measured update time exceeds complete epoch train time')
        loop=max(0.,residual)
    gpu=sum_known([records[t]['ledger']['seconds'] for t in tokens]) if attributable else None
    cpu_user=sum_known([(records[t]['receipt'] or {}).get('child_user_cpu_seconds') for t in tokens]) if attributable else None
    cpu_system=sum_known([(records[t]['receipt'] or {}).get('child_system_cpu_seconds') for t in tokens]) if attributable else None
    cpu_total=cpu_user+cpu_system if cpu_user is not None and cpu_system is not None else None
    # A fully complete history can still have missing failed startup attempts;
    # occupied lease costs come only from the ledger, never from epoch durations.
    return {'seconds_per_epoch':whole_epoch_total/len(timing) if whole_epoch_total is not None else None,
        'occupied_gpu_seconds':gpu,'update_work_seconds':updates_all,
        'inside_lease_child_user_cpu_seconds':cpu_user,'inside_lease_child_system_cpu_seconds':cpu_system,
        'inside_lease_child_cpu_seconds':cpu_total,
        'data_wait_and_loop_overhead_seconds':loop,
        'callback_and_reporting_seconds':callback if not partial else None,
        'data_wait_seconds':waits,'completed_epochs':len(history) if paths['history'].exists() else None,
        'recorded_attempts':len(updates) if paths['direct_updates'].exists() else None,
        'partial_epoch_attempts':max(0,len(updates)-complete_steps) if paths['direct_updates'].exists() and paths['epoch_timing'].exists() else None,
        'runtime_devices':sorted({str((records[t]['receipt'] or {}).get('device',{}).get('name', records[t]['ledger'].get('gpu_uuid','unknown'))) for t in tokens}),
        'complete_epoch_train_seconds':train,'complete_epoch_wall_seconds':whole_epoch_total,
        'complete_epoch_callback_seconds':callback,'lease_tokens':tokens,
        'definitions':'update work includes accepted and rejected attempts; residual includes data waits and loop overhead; epoch callbacks are separate; GPU occupancy includes whole assigned leases',
        'evidence':evidence}


def cpu_jobs(items):
    output=[];seen=set()
    for item in items:
        path=Path(item['path']).resolve(strict=True)
        if path in seen:raise ValueError('CPU job receipt supplied twice')
        seen.add(path)
        if item.get('kind') not in ('assessment','independent') or item.get('inside_gpu_lease') is not False:
            raise ValueError('CPU receipts must identify a separate assessment or independent process')
        obj=read(path)
        cpu=obj.get('cpu_seconds',obj.get('process_cpu_seconds'))
        if cpu is None and measured(obj.get('user_cpu_seconds')) and measured(obj.get('system_cpu_seconds')):
            cpu=obj['user_cpu_seconds']+obj['system_cpu_seconds']
        start=obj.get('start_unix',obj.get('started_unix'));end=obj.get('end_unix')
        output.append({'kind':item['kind'],'status':obj.get('status'),'cpu_seconds':cpu if measured(cpu) else None,
            'start_unix':start if measured(start) else None,'end_unix':end if measured(end) else None,
            'receipt':ref(path)})
    return output


def assessment_spans(windows, records, jobs):
    """Sum separate assessment windows; their wall span includes any GPU/CPU overlap."""
    if not windows:
        return None, None
    gpu_tokens={t for t,v in records.items() if v['ledger'].get('phase')=='assessment'}
    cpu_paths={j['receipt']['path'] for j in jobs if j['kind']=='assessment'}
    seen_gpu=set();seen_cpu=set();spans=[]
    by_path={j['receipt']['path']:j for j in jobs if j['kind']=='assessment'}
    for window in windows:
        tokens=window.get('gpu_tokens',[])
        paths=[str(Path(p).resolve(strict=True)) for p in window.get('cpu_paths',[])]
        if not tokens or not paths or set(tokens)&seen_gpu or set(paths)&seen_cpu:
            raise ValueError('assessment windows require distinct GPU tokens and CPU receipts')
        if not set(tokens)<=gpu_tokens or not set(paths)<=cpu_paths:
            raise ValueError('assessment window references an unrelated lease or CPU job')
        seen_gpu.update(tokens);seen_cpu.update(paths)
        gpu=[records[t]['ledger'] for t in tokens];cpu=[by_path[p] for p in paths]
        starts=[r.get('start_unix') for r in gpu]+[j['start_unix'] for j in cpu]
        ends=[r.get('end_unix') for r in gpu]+[j['end_unix'] for j in cpu]
        if not all(measured(v) for v in starts+ends):return None,None
        if any(end<start for start,end in zip(starts,ends,strict=True)):
            raise ValueError('assessment window ends before it starts')
        start=min(starts);end=max(ends)
        tail=max(0.,max(j['end_unix'] for j in cpu)-max(r['end_unix'] for r in gpu))
        spans.append((start,end,tail))
    if seen_gpu!=gpu_tokens or seen_cpu!=cpu_paths:
        raise ValueError('assessment windows must account for every assessment GPU lease and CPU job')
    spans.sort()
    if any(a[1]>b[0] for a,b in zip(spans,spans[1:])):
        raise ValueError('assessment windows overlap; summing their wall spans would double count')
    return sum(end-start for start,end,_ in spans),spans[-1][2]


def collect(campaign,input_path):
    campaign=Path(campaign).resolve(strict=True);input_path=Path(input_path).resolve(strict=True)
    inputs=read(input_path)
    if inputs.get('schema')!='car_round5_cost_inputs_v1' or not isinstance(inputs.get('arms'),dict):
        raise ValueError('Round 5 cost input manifest with named arms required')
    frozen_path=campaign/'configs/formal_freeze.json';frozen=read(frozen_path)
    if frozen.get('schema')!='car_round5_freeze_v1' or not set(inputs['arms']).issubset(set(frozen['active_arms'])):
        raise ValueError('cost arms differ from the frozen study')
    leases,evidence=closed_leases(campaign)
    assigned,unassigned=assign_arms(leases,inputs['arms'])
    arms={arm:training(value['run_dir'],assigned.get(arm,[]),leases,
                       attributable=bool(assigned.get(arm)) and not unassigned)
          for arm,value in inputs['arms'].items()}
    by_phase={}
    for entry in leases.values():
        row=entry['ledger'];phase=row.get('phase','unclassified')
        by_phase[phase]=by_phase.get(phase,0.)+row['seconds']
    gpu_assessment=[v['ledger'] for v in leases.values() if v['ledger'].get('phase')=='assessment']
    jobs=cpu_jobs(inputs.get('cpu_jobs',[]));assessment_jobs=[j for j in jobs if j['kind']=='assessment']
    assessment_gpu=sum_known([x['seconds'] for x in gpu_assessment])
    assessment_cpu=sum_known([j['cpu_seconds'] for j in assessment_jobs])
    wall,tail=assessment_spans(inputs.get('assessment_windows',[]),leases,jobs)
    independent=[j for j in jobs if j['kind']=='independent']
    return {'schema':'car_round5_costs_v1','training':arms,
        'campaign_gpu_seconds':sum_known([v['ledger']['seconds'] for v in leases.values()]),
        'gpu_seconds_by_phase':by_phase,'failed_attempt_gpu_seconds':(sum_known([v['ledger']['seconds'] for v in leases.values() if v['receipt'] is not None and v['receipt'].get('status')=='failed']) if all(v['receipt'] is not None for v in leases.values()) else None),
        'unattributed_formal_tokens':unassigned,'missing_completion_tokens':[t for t,v in leases.items() if v['receipt'] is None],
        'assessment':{'gpu_seconds':assessment_gpu,'cpu_seconds':assessment_cpu,'wall_seconds':wall,'queue_tail_seconds':tail},
        'independent_cpu_seconds':sum_known([j['cpu_seconds'] for j in independent]),'cpu_jobs':jobs,
        'forecast':frozen.get('forecast'),'forecast_is_measured':False,
        'accounting_note':'CPU process times are separate from occupied GPU lease wall times; do not add overlapping CPU and GPU work to elapsed wall time. Null indicates unavailable measurements.',
        'inputs':ref(input_path),'frozen_manifest':ref(frozen_path),'ledger_and_completion_evidence':evidence}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();value=collect(args.campaign,args.inputs);immutable_json(args.output,value)
    print(json.dumps(value,indent=2,allow_nan=False))

if __name__=='__main__':main()
