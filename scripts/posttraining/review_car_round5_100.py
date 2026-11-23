#!/usr/bin/env python3
"""Sequential GPU0 native/direct 100-epoch review bundle; no continuation.

This is a launch bundle, not a scheduler. Run only after the freeze is reviewed
and launch is authorized. It runs each training arm to exactly 1200 attempted
steps and checks its durable checkpoint and history. It then generates both raw
J4 banks consecutively and performs one task-free VALIDATION100 CPU score. Failed stages stop the
sequence and leave their original leases and output intact.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT=Path(os.environ.get('CAR_REPO_ROOT', Path(__file__).resolve().parents[2])).resolve()
ARMS=('Native_GT_Continuation','Native_GT_SSS_Safeguarded_Continuation')
LIMITS={ARMS[0]:1200,ARMS[1]:12000}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def stage(command, env, logfile, receipt_path):
    started=time.time(); usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    result={'command':command,'start_unix':started,'status':'failed','log':str(logfile)}
    try:
        with logfile.open('x') as log:
            completed=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
        result['exit_code']=completed.returncode
        if completed.returncode:raise subprocess.CalledProcessError(completed.returncode,command)
        result['status']='completed'
        return result
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        used=resource.getrusage(resource.RUSAGE_CHILDREN);ended=time.time()
        result.update(end_unix=ended,wall_seconds=ended-started,
                      user_cpu_seconds=used.ru_utime-usage.ru_utime,
                      system_cpu_seconds=used.ru_stime-usage.ru_stime)
        receipt_path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


def completed_run(log, campaign, arm):
    allowed=(campaign/'runs/formal'/arm).resolve()
    for line in reversed(log.read_text().splitlines()):
        text=line.strip()
        if not text.startswith('/'):continue
        path=Path(text)
        if not path.is_dir():continue
        path=path.resolve(strict=True)
        if allowed not in path.parents:continue
        status=json.loads((path/'status.json').read_text())
        if status.get('global_step')!=1200 or status.get('steps_per_epoch')!=12:
            raise ValueError(f'{arm}: training stopped before the 100-epoch review')
        checkpoint=path/'checkpoints/epoch_100.pt'
        if not checkpoint.is_file():raise ValueError(f'{arm}: epoch_100 checkpoint missing')
        updates=[json.loads(line) for line in (path/'logs/direct_updates.jsonl').read_text().splitlines() if line.strip()]
        if [row.get('step') for row in updates]!=list(range(1,1201)):
            raise ValueError(f'{arm}: expected exactly 1200 recorded attempts')
        history=[json.loads(line) for line in (path/'logs/history.jsonl').read_text().splitlines() if line.strip()]
        if [row.get('epoch') for row in history]!=list(range(1,101)) or any(row.get('epoch_complete') is not True for row in history):
            raise ValueError(f'{arm}: expected 100 complete recorded epochs')
        return path,checkpoint
    raise ValueError(f'{arm}: completed formal run path not found in trainer output')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='new campaign-owned operations directory')
    args=parser.parse_args()
    campaign=args.campaign.resolve(strict=True);output=args.output.resolve()
    if campaign/'operations' not in output.parents:raise ValueError('bundle output must be campaign-owned operations storage')
    freeze_path=campaign/'configs/formal_freeze.json';freeze=json.loads(freeze_path.read_text())
    if (freeze.get('schema')!='car_round5_freeze_v1' or freeze.get('test_locked') is not True
            or freeze.get('active_arms')!=list(ARMS) or freeze.get('final_epoch',0)<=100):
        raise ValueError('expected frozen two-arm study with task-free 100-epoch review')
    selected=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not selected or ',' in selected or not os.environ.get('KEOPS_CACHE_FOLDER'):
        raise ValueError('select one runtime CUDA_VISIBLE_DEVICES entry and set KEOPS_CACHE_FOLDER')
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    code=Path(freeze['code_root']).resolve(strict=True)
    driver=code/'scripts/posttraining/run_car_round5.py';assessment=code/'scripts/posttraining/assess_car_round5.py'
    env=dict(os.environ,CAR_REPO_ROOT=str(ROOT),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PHYCOFLOW_TOPOLOGY_WORKERS='4')
    result={'schema':'car_round5_review100_bundle_v1','status':'failed','start_unix':time.time(),
        'freeze':str(freeze_path),'freeze_sha256':digest(freeze_path),'code_root':str(code),
        'bundle_code_sha256':digest(Path(__file__)),
        'decision_scope':'task-free VALIDATION100 review; no continuation or terminal task scoring',
        'arms':{}}
    try:
        for arm in ARMS:
            name='native' if arm==ARMS[0] else 'direct'
            log=output/f'{name}_train.log'
            command=[sys.executable,str(driver),'--campaign',str(campaign),'train','--arm',arm,
                     '--stop-epoch','100','--max-seconds',str(LIMITS[arm])]
            stage(command,env,log,output/f'{name}_train_receipt.json')
            run,checkpoint=completed_run(log,campaign,arm)
            result['arms'][arm]={'run_dir':str(run),'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),
                'training_receipt':str(output/f'{name}_train_receipt.json')}
        for arm in ARMS:
            name='native' if arm==ARMS[0] else 'direct'
            run=Path(result['arms'][arm]['run_dir'])
            bank=campaign/'assessment/epoch_100/banks'/name
            gen=[sys.executable,str(driver),'--campaign',str(campaign),'lease','--phase','assessment',
                 '--max-seconds','600','--',sys.executable,str(assessment),'--campaign',str(campaign),
                 'generate','--arm',arm,'--epoch','100','--run',str(run),'--device','cuda:0','--output',str(bank)]
            generation=stage(gen,env,output/f'{name}_generation.log',output/f'{name}_generation_receipt.json')
            result['arms'][arm].update(bank=str(bank),
                generation_receipt=str(output/f'{name}_generation_receipt.json'),
                generation_start_unix=generation['start_unix'],generation_end_unix=generation['end_unix'])
        scoredir=campaign/'assessment/epoch_100/scores'
        score=[sys.executable,str(assessment),'--campaign',str(campaign),'score','--epoch','100',
            '--raw-bank',result['arms'][ARMS[0]]['bank'],'--scaled-bank',result['arms'][ARMS[1]]['bank'],
            '--output',str(scoredir)]
        score_result=stage(score,env,output/'cpu_score.log',output/'cpu_score_receipt.json')
        review=json.loads((scoredir/'assessment.json').read_text())
        if review.get('epoch')!=100 or review.get('task_scope')!='both tasks excluded':
            raise ValueError('CPU output is not the required task-free 100-epoch review')
        starts=[v['generation_start_unix'] for v in result['arms'].values()]
        ends=[v['generation_end_unix'] for v in result['arms'].values()]
        gpu_wall=sum(v['generation_end_unix']-v['generation_start_unix'] for v in result['arms'].values())
        cpu_wall=score_result['wall_seconds']
        result.update(status='completed',review=str(scoredir/'assessment.json'),
            review_sha256=digest(scoredir/'assessment.json'),cpu_score_receipt=str(output/'cpu_score_receipt.json'),
            assessment_generation_wall_seconds=gpu_wall,assessment_cpu_wall_seconds=cpu_wall,
            assessment_added_wall_seconds=score_result['end_unix']-min(starts),
            final_cpu_queue_tail_seconds=max(0.,score_result['end_unix']-max(ends)),
            cpu_user_seconds=score_result['user_cpu_seconds'],cpu_system_seconds=score_result['system_cpu_seconds'])
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        result['end_unix']=time.time()
        (output/'bundle_receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
