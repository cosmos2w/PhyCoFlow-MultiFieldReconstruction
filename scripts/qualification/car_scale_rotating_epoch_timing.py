#!/usr/bin/env python3
"""Matched actual-parent complete-epoch cost; no model efficacy assessment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(os.environ.get('CAR_EXECUTION_ROOT', os.environ.get('CAR_REPO_ROOT', Path(__file__).resolve().parents[2]))).resolve()
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from phycoflow_reconstruction.evaluation.adaptive_campaign import atomic_json, digest
from scripts.qualification.car_scale_fork_resume_qualification import parent_and_source_identity


def worker(args):
    import torch
    import yaml
    from phycoflow_reconstruction.training.fork_control import create_control_fork, create_fidelity_quadrature_fork
    from phycoflow_reconstruction.training.post_training import run_post_training
    torch.set_num_threads(4)
    cfg=yaml.safe_load(args.config.read_text())
    cfg['output']['experiment_name']='qualification/RotatingComplement_CompleteEpoch/'+args.variant
    create=create_fidelity_quadrature_fork if args.variant=='RotatingComplement' else create_control_fork
    start=time.time()
    run=create(args.case_dir,cfg,args.parent,parent_checkpoint_sha256=args.parent_sha256)
    run=run_post_training(cfg,case_dir=args.case_dir,resume=run,budget_stop_requested=lambda step,_run:step>=6012)
    timing=[json.loads(line) for line in (run/'logs/epoch_timing.jsonl').read_text().splitlines() if line.strip()]
    if len(timing)!=1 or timing[0]['epoch']!=1 or timing[0]['step']!=6012 or timing[0]['timed_updates']!=12 or not timing[0]['complete_epoch_duration']:
        raise ValueError('one complete phase epoch timing required')
    if json.loads((run/'status.json').read_text())['global_step']!=6012: raise ValueError('wrong completed action cursor')
    atomic_json(args.output,{'variant':args.variant,'run':str(run),'pid':os.getpid(),
        'wall_seconds':time.time()-start,'training_epoch':timing[0],
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(), 'device_name':torch.cuda.get_device_name(0)})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--parent-sha256',required=True)
    p.add_argument('--control-config',type=Path,required=True);p.add_argument('--candidate-config',type=Path,required=True)
    p.add_argument('--case-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--variant',choices=['Control','RotatingComplement']);p.add_argument('--config',type=Path)
    a=p.parse_args()
    if a.worker: return worker(a)
    if not os.environ.get('CAR_ROUND6_LEASE_TOKEN'):raise ValueError('bounded follow-up GPU lease required')
    a.output.mkdir(parents=True,exist_ok=False);start=time.time();receipt={'schema':'car_scale_rotating_complete_epoch_v1','status':'failed','parent_sha256':a.parent_sha256,'start_unix':start,'scope':'one independently reset actual-parent control and candidate epoch; diagnostic cost only','attempts':[]}
    before=None
    try:
        before=parent_and_source_identity(a.parent,a.control_config);receipt['parent_source_before']=before
        if before['parent_files']['checkpoints/epoch_500.pt']!=a.parent_sha256:raise ValueError('parent checkpoint mismatch')
        for name,config in [('Control',a.control_config),('RotatingComplement',a.candidate_config)]:
            result=a.output/f'{name}.json'
            command=[sys.executable,str(Path(__file__).resolve()),'--worker','--variant',name,'--config',str(config),'--output',str(result),'--case-dir',str(a.case_dir),'--parent',str(a.parent),'--parent-sha256',a.parent_sha256,'--control-config',str(a.control_config),'--candidate-config',str(a.candidate_config)]
            with (a.output/f'{name}.log').open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            receipt['attempts'].append(json.loads(result.read_text()))
        runs=[Path(r['run']) for r in receipt['attempts']]
        rows=[[json.loads(x) for x in (run/'metrics/training_coupling.jsonl').read_text().splitlines() if x.strip()] for run in runs]
        keys=('step','sample_ids','observation_sha256','query_sha256','member_seeds','first_member_generator_state_sha256')
        if len(rows[0])!=12 or len(rows[1])!=12 or any(any(x.get(k)!=y.get(k) for k in keys) for x,y in zip(*rows,strict=True)):
            raise ValueError('matched scientific condition/observation/SSS query/member coupling differs')
        receipt['parent_source_after']=parent_and_source_identity(a.parent,a.control_config)
        if before!=receipt['parent_source_after']:raise ValueError('parent or SOURCE changed')
        receipt['matched_sss_coupling']=True;receipt['parent_source_immutable_verified']=True
        receipt['status']='completed'
    except BaseException as exc:
        receipt.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:
        if before is not None and 'parent_source_after' not in receipt:
            try:
                receipt['parent_source_after']=parent_and_source_identity(a.parent,a.control_config)
                receipt['parent_source_immutable_verified']=receipt['parent_source_after']==before
            except Exception as exc:receipt['parent_source_verification_error']=str(exc)
        receipt.update(end_unix=time.time(),wall_seconds=time.time()-start);atomic_json(a.output/'receipt.json',receipt)
if __name__=='__main__':main()
