#!/usr/bin/env python3
"""Run one leased phase of a bound epoch-500 Scale continuation fork."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path(os.environ.get('CAR_EXECUTION_ROOT',os.environ.get('CAR_REPO_ROOT',Path(__file__).resolve().parents[2]))).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
ORIGIN=6000
STEPS_PER_EPOCH=12
CONTROL='FFM_SSS_Unchanged_Scale_Continuation'
CANDIDATE='FFM_SSS_RotatingComplement_FieldFeedback'


def ref(path):
    from phycoflow_reconstruction.training.run_store import file_sha256
    path=Path(path).resolve(strict=True)
    return {'path':str(path),'sha256':file_sha256(path)}


def target_step(phase_epoch):
    if phase_epoch not in (100,500):raise ValueError('phase endpoint must be 100 or 500')
    return ORIGIN+STEPS_PER_EPOCH*phase_epoch


def validate_config(cfg,parent_cfg,variant):
    """Keep the original SOURCE risk and allow only the nominated quadrature."""
    if variant not in (CONTROL,CANDIDATE):raise ValueError('unrecognized descriptive fork variant')
    if cfg['optimization']['epochs']!=500 or cfg['optimization']['steps_per_epoch']!=STEPS_PER_EPOCH:
        raise ValueError('500-epoch maximum and 12 actions per epoch required')
    if cfg['runtime'].get('plot_every_steps',10)!=10:raise ValueError('loss curves must refresh every 10 epochs')
    if cfg['objectives']['data_retention']['enabled'] or cfg['objectives']['data_retention']['weight']!=0 or cfg['optimization'].get('direct_supervision'):
        raise ValueError('native data loss must remain disabled')
    current=deepcopy(cfg['optimization']['multitask']['fidelity_budget'])
    parent=deepcopy(parent_cfg['optimization']['multitask']['fidelity_budget'])
    complement=current.pop('rotating_complement',None)
    expected=None if variant==CONTROL else {'points':512,'seed':20261125}
    if current!=parent or complement!=expected:raise ValueError('SOURCE field reference or authorized quadrature differs')
    declaration=current['reference_artifact']
    if ref(declaration['path'])['sha256']!=declaration['sha256']:raise ValueError('SOURCE field reference artifact changed')
    return declaration


def verify_origin(manifest,state,parent,expected_hash,variant):
    expected={'parent_run':str(parent.resolve(strict=True)),'parent_checkpoint_sha256':expected_hash,'absolute_step':ORIGIN}
    if variant==CANDIDATE:expected['scientific_transition']='rotating_complement_field_quadrature_512_v1'
    origin=manifest.get('fork_origin')
    if origin!=expected or state.get('fork_origin')!=origin:
        raise ValueError('fork parent/checkpoint/origin identity differs')
    return origin


def completed_epoch_evidence(run,phase_epoch,step):
    """Describe recorded timing honestly, and verify the requested boundary when reached."""
    def read(name):
        path=run/'logs'/name
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    history=read('history.jsonl');journal=read('epoch_calibrated_pi_updates.jsonl');timing=read('epoch_timing.jsonl')
    complete=[r for r in timing if r.get('complete_epoch_duration',True) is True]
    values=[float(r['wall_seconds']) for r in complete]
    if any(not math.isfinite(v) or v<0 for v in values):raise ValueError('invalid epoch wall duration')
    evidence={'history_rows':len(history),'pi_action_rows':len(journal),'timing_rows':len(timing),
              'complete_duration_rows':len(complete),'partial_duration_rows':len(timing)-len(complete),
              'complete_epoch_mean_wall_seconds':sum(values)/len(values) if values else None}
    if step==target_step(phase_epoch):
        if ([int(r['epoch']) for r in history]!=list(range(1,phase_epoch+1))
                or any(r.get('epoch_complete') is not True for r in history)
                or [r['step'] for r in journal]!=list(range(ORIGIN+1,step+1))):
            raise ValueError('phase endpoint history or action journal incomplete')
        checkpoint=run/'checkpoints'/f'epoch_{phase_epoch:03d}.pt'
        evidence.update(checkpoint=ref(checkpoint),history=ref(run/'logs/history.jsonl'),
                        pi_journal=ref(run/'logs/epoch_calibrated_pi_updates.jsonl'),
                        epoch_timing=ref(run/'logs/epoch_timing.jsonl'),
                        recording_definitions=ref(run/'logs/recording_definitions.json'))
        for name in ('loss_history.png','loss_history.pdf'):
            if not (run/'logs'/name).is_file():raise ValueError('required loss curve missing')
        evidence['loss_curves']=[ref(run/'logs'/name) for name in ('loss_history.png','loss_history.pdf')]
    return evidence


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--config-sha256',required=True)
    p.add_argument('--parent-snapshot',type=Path,required=True);p.add_argument('--parent-snapshot-sha256',required=True)
    p.add_argument('--variant',choices=(CONTROL,CANDIDATE),required=True);p.add_argument('--phase-epoch',type=int,choices=(100,500),required=True)
    p.add_argument('--resume',type=Path);p.add_argument('--device',required=True);p.add_argument('--result',type=Path,required=True)
    args=p.parse_args()
    if not os.environ.get('CAR_ROUND6_LEASE_TOKEN'):raise ValueError('training requires an enclosing campaign GPU lease')
    import torch
    import yaml
    from phycoflow_reconstruction.training.post_training import run_post_training
    from phycoflow_reconstruction.training.run_store import load_project_checkpoint
    from phycoflow_reconstruction.evaluation.adaptive_campaign import atomic_json
    campaign=args.campaign.resolve(strict=True);result=args.result.resolve();target=target_step(args.phase_epoch)
    if campaign/'operations' not in result.parents or result.exists():raise ValueError('new campaign-owned result receipt required')
    config_ref=ref(args.config);snapshot_ref=ref(args.parent_snapshot)
    if config_ref['sha256']!=args.config_sha256 or snapshot_ref['sha256']!=args.parent_snapshot_sha256:raise ValueError('config or parent snapshot hash differs')
    expected_snapshot=(campaign.parent/'round_6/assessment/Epoch500_Validation100_J4_20261125T042223Z/snapshot/snapshot.json').resolve(strict=True)
    if Path(snapshot_ref['path'])!=expected_snapshot:raise ValueError('parent must be the original epoch500 snapshot')
    snapshot=json.loads(args.parent_snapshot.read_text());parent=Path(snapshot['run']).resolve(strict=True)
    parent_checkpoint=ref(snapshot['checkpoint']['path'])
    if parent_checkpoint!=snapshot['checkpoint'] or parent_checkpoint['path']!=str(parent/'checkpoints/epoch_500.pt'):raise ValueError('original epoch500 parent identity differs')
    original_files=[parent/'resolved_config.yaml',parent/'run_manifest.json',parent/'checkpoints/epoch_500.pt']
    before=[ref(path) for path in original_files];parent_cfg=yaml.safe_load((parent/'resolved_config.yaml').read_text());cfg=yaml.safe_load(args.config.read_text())
    source_ref=validate_config(cfg,parent_cfg,args.variant)
    cfg=deepcopy(cfg);cfg['output']['experiment_name']='formal/'+args.variant;cfg['runtime']['device']=args.device
    parent_state=load_project_checkpoint(parent/'checkpoints/epoch_500.pt')
    if parent_state['global_step']!=ORIGIN:raise ValueError('parent must be at action 6000')
    result.parent.mkdir(parents=True,exist_ok=True)
    receipt={'schema':'car_scale_followup_training_v1','status':'running','variant':args.variant,'phase_target_epoch':args.phase_epoch,
        'phase_origin_step':ORIGIN,'target_global_step':target,'lease_token':os.environ['CAR_ROUND6_LEASE_TOKEN'],
        'selected_config':config_ref,'parent_snapshot':snapshot_ref,'parent_checkpoint':parent_checkpoint,'source_reference':source_ref,
        'execution_root':str(ROOT),'driver':ref(Path(__file__)),'start_unix':time.time(),'test_locked':True}
    with result.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    started=time.time();run=None;requested=False
    def stop_signal(_sig,_frame):
        nonlocal requested
        requested=True
    previous={sig:signal.signal(sig,stop_signal) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        if args.resume is None:
            if args.phase_epoch!=100:raise ValueError('new fork must start with phase100 review')
            if args.variant==CONTROL:
                from phycoflow_reconstruction.training.fork_control import create_control_fork as creator
            else:
                from phycoflow_reconstruction.training.fork_control import create_fidelity_quadrature_fork as creator
            run=Path(creator(campaign,cfg,parent,parent_checkpoint_sha256=parent_checkpoint['sha256'])).resolve(strict=True)
        else:
            run=args.resume.resolve(strict=True)
            if campaign/'runs'/'formal'/args.variant not in run.parents:raise ValueError('resume must use the named followup fork')
            saved=yaml.safe_load((run/'resolved_config.yaml').read_text())
            expected=deepcopy(cfg);observed=deepcopy(saved)
            expected['runtime'].pop('device',None);observed['runtime'].pop('device',None)
            if expected!=observed:raise ValueError('saved fork science differs from selected config')
        state=load_project_checkpoint(run/'checkpoints/last.pt');manifest=json.loads((run/'run_manifest.json').read_text())
        origin=verify_origin(manifest,state,parent,parent_checkpoint['sha256'],args.variant)
        if state['external_policy']['reference']!=parent_state['external_policy']['reference']:
            raise ValueError('fork changed original field risks, scales, allowances or C0')
        if state['global_step']<ORIGIN or state['global_step']>target:raise ValueError('resume is outside target phase')
        def stop(step,_run):
            deadline=float(os.environ.get('CAR_ROUND6_LEASE_DEADLINE','inf'))
            return requested or step>=target or time.time()>=deadline-180
        actual=Path(run_post_training(cfg,case_dir=campaign,resume=run,budget_stop_requested=stop)).resolve(strict=True)
        if actual!=run:raise ValueError('trainer returned a different fork')
        status=json.loads((run/'status.json').read_text());step=int(status['global_step'])
        if step>target:raise ValueError('trainer crossed the requested phase endpoint')
        final_state=load_project_checkpoint(run/'checkpoints/last.pt');verify_origin(json.loads((run/'run_manifest.json').read_text()),final_state,parent,parent_checkpoint['sha256'],args.variant)
        if final_state.get('global_step')!=step or status.get('source_immutable_verified') is not True:
            raise ValueError('trainer status, checkpoint or SOURCE immutability differs')
        if final_state['external_policy']['reference']!=parent_state['external_policy']['reference']:
            raise ValueError('training changed the bound SOURCE field reference')
        after=[ref(path) for path in original_files]
        if before!=after:raise ValueError('original parent artifacts changed')
        receipt.update(status='completed_phase' if step==target else 'paused_before_target',run=str(run),fork_origin=origin,
            trainer_status=status,global_step=step,phase_epoch_completed=(step-ORIGIN)/STEPS_PER_EPOCH,
            complete_epoch_evidence=completed_epoch_evidence(run,args.phase_epoch,step),original_parent_files=before,
            original_parent_unchanged=True)
    except BaseException as exc:
        receipt.update(status='failed',error_type=type(exc).__name__,error=str(exc),run=str(run) if run else None);raise
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
        receipt.update(end_unix=time.time(),wall_seconds=time.time()-started)
        atomic_json(result,receipt)
    print(json.dumps(receipt,indent=2,allow_nan=False))

if __name__=='__main__':main()
