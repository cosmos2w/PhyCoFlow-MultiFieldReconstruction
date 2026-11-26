#!/usr/bin/env python3
"""Score one completed epoch-100 or epoch-500 arm's four banks, without GPU or polling.

The input is the immutable ready.json emitted by run_car_round7_review.py.
A new DATA2 assessment attempt retains per-split logs, outputs and failures.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.posttraining.assess_car_round7 import baseline_binding, checked, digest, load, ref, validate_spec, verify_bank, write
from scripts.posttraining.run_car_round7_review import ARMS

SPLITS=('train','validation')
SUPPORTS=('native','selected')


def prepare(ready_path):
    """Verify terminal receipt and bind both splits to checkpoint, panels and SOURCE baselines."""
    ready_path=Path(ready_path).resolve(strict=True)
    ready=load(ready_path)
    if (ready.get('schema')!='car_round7_review_arm_ready_v1' or ready.get('arm') not in ARMS or
        set(ready.get('specs',{}))!=set(SPLITS) or len(ready.get('banks',[]))!=4):
        raise ValueError('one complete endpoint arm ready receipt is required')
    if ready_path.parent.name != ready['arm']:
        raise ValueError('ready receipt must remain in its named arm attempt')
    checkpoint=checked(ready['checkpoint'])
    training_path=ready_path.parent/'training_result.json'
    training=load(training_path)
    epoch=training.get('target_epoch')
    if (training.get('schema')!='car_round7_training_result_v1' or training.get('arm')!=ready['arm'] or
        type(epoch) is not int or epoch not in (100,500) or training.get('target_reached') is not True or
        training.get('global_step')!=12*epoch or
        Path(ready['checkpoint']['path']).absolute()!=Path(training['run_dir']).absolute()/f'checkpoints/epoch_{epoch:03d}.pt' or
        (Path(training['run_dir']).resolve(strict=True)/f'checkpoints/epoch_{epoch:03d}.pt').resolve(strict=True)!=checkpoint):
        raise ValueError('ready checkpoint must belong to the completed canonical endpoint arm run')
    saved_config=checked(training['saved_config'])
    specs={split:checked(ready['specs'][split]) for split in SPLITS}
    contexts={}; campaign=None
    for split in SPLITS:
        spec=load(specs[split]); panel, source=validate_spec(spec)
        if (spec['split']!=split or spec.get('reporting_epoch')!=epoch or
            spec.get('checkpoint')!=ready['checkpoint'] or checked(spec['checkpoint'])!=checkpoint or
            checked(spec['run_config'])!=saved_config):
            raise ValueError('split spec does not bind the matched endpoint checkpoint')
        current=Path(spec['campaign_root']).resolve(strict=True)
        if campaign is not None and current!=campaign: raise ValueError('split campaigns differ')
        campaign=current
        baseline_path=campaign/'assessment'/f'SOURCE_Baseline_{"TRAIN800" if split=="train" else "VALIDATION100"}_v1'/'baseline.json'
        baseline=load(baseline_path)
        source_selected=checked(baseline['source_selected'])
        verify_bank(source_selected,spec,panel,source,'selected')
        if (baseline.get('schema')!='car_round7_source_baseline_v1' or baseline.get('status')!='completed' or
            baseline.get('binding')!=baseline_binding(spec) or baseline.get('sample_ids')!=panel['sample_ids'] or
            baseline.get('source_native')!=spec['source_native_receipt'] or len(baseline.get('files_sha256',[]))!=len(panel['sample_ids'])):
            raise ValueError('frozen SOURCE baseline is incompatible')
        contexts[split]={'spec':spec,'spec_path':specs[split], 'panel':panel, 'source':source,
                         'baseline':baseline_path,'source_selected':source_selected,'banks':{},'epoch':epoch}
    if (campaign/'assessment') not in ready_path.parents:
        raise ValueError('ready receipt must be in the same campaign assessment tree')
    for bank_ref in ready['banks']:
        path=checked(bank_ref); bank=load(path)
        split=bank.get('split');support=bank.get('support')
        if split not in SPLITS or support not in SUPPORTS or support in contexts[split]['banks']:
            raise ValueError('exactly one bank for each split and support is required')
        ctx=contexts[split]
        verify_bank(path,ctx['spec'],ctx['panel'],ctx['source'],support)
        if (bank.get('checkpoint_sha256')!=ready['checkpoint']['sha256'] or
            bank.get('run_config_sha256')!=ctx['spec']['run_config']['sha256'] or
            bank.get('spec_sha256')!=ready['specs'][split]['sha256']):
            raise ValueError('bank checkpoint, run config or exact split spec differs')
        ctx['banks'][support]=path
    if any(set(contexts[split]['banks'])!=set(SUPPORTS) for split in SPLITS):
        raise ValueError('all four completed matched banks are required')
    if digest(ready_path)!=ref(ready_path)['sha256']:
        raise ValueError('ready receipt changed during preparation')
    return ready, campaign, contexts


def frozen_scorer(campaign):
    """Bind the actual subprocess to the sealed scientific code and scorer hash."""
    freeze_path=Path(campaign)/'configs/formal_freeze.json'
    freeze=load(freeze_path)
    if freeze.get('schema')!='car_round7_formal_freeze_v1' or freeze.get('status')!='sealed':
        raise ValueError('sealed formal freeze required')
    code=Path(freeze['code_root']).resolve(strict=True)
    inventory_path=checked(freeze['execution_code'])
    inventory=load(inventory_path)
    relative='scripts/posttraining/assess_car_round7.py'
    scorer=code/relative
    if (inventory_path!=code/'execution_receipt.json' or inventory.get('sha256')!=code.name or
        inventory.get('files',{}).get(relative)!=digest(scorer) or digest(ROOT/relative)!=digest(scorer)):
        raise ValueError('frozen and current assessment scorer identities differ')
    return code, ref(scorer), ref(freeze_path)


def commands(contexts, output, *, python=sys.executable, code=ROOT):
    """Construct the only two allowed CPU scoring commands in declared order."""
    output=Path(output); code=Path(code); result=[]
    for split in SPLITS:
        item=contexts[split]
        result.append({'split':split,'command':[str(python),str(code/'scripts/posttraining/assess_car_round7.py'),
            'score','--spec',str(item['spec_path']),'--child-native',str(item['banks']['native']),
            '--source-selected',str(item['source_selected']),'--child-selected',str(item['banks']['selected']),
            '--baseline',str(item['baseline']),'--threads','2','--output',str(output/split)]})
    return result


def run(args):
    if os.environ.get('CAR_ROUND7_LEASE_TOKEN'):
        raise ValueError('CPU scoring must run outside a GPU lease')
    ready_path=args.ready.resolve(strict=True); ready_bytes=ready_path.read_bytes(); ready_hash=digest(ready_path)
    # Only inspect the supplied terminal ready document; no run/status polling.
    initial=json.loads(ready_bytes)
    if set(initial.get('specs',{}))!=set(SPLITS): raise ValueError('terminal ready receipt required')
    preliminary=load(checked(initial['specs']['train']))
    campaign=Path(preliminary['campaign_root']).resolve(strict=True)
    output=args.output.resolve()
    data_root=Path('/data2/wanglz/PhyCoFlow-MultiFieldReconstruction').resolve(strict=True)
    if data_root not in campaign.parents or campaign/'assessment' not in output.parents:
        raise ValueError('new output must be in the DATA2 campaign assessment tree')
    output.mkdir(parents=True,exist_ok=False)
    started=time.time(); self_cpu=time.process_time(); child_cpu=resource.getrusage(resource.RUSAGE_CHILDREN)
    result={'schema':'car_round7_review_cpu_scoring_v1','status':'failed','ready':{'path':str(ready_path),'sha256':ready_hash},
            'arm':initial.get('arm'),'reporting_epoch':None,'threads':2,'test_locked':True,
            'scope':'CPU scoring of cached endpoints only; no prediction generation, task scoring or TEST',
            'start_unix':started,'splits':{}}
    try:
        ready,campaign,contexts=prepare(ready_path)
        if digest(ready_path)!=ready_hash: raise ValueError('ready receipt changed during validation')
        epoch=contexts['train']['epoch']
        code,scorer,freeze=frozen_scorer(campaign)
        result.update(checkpoint=ready['checkpoint'],reporting_epoch=epoch,scorer=scorer,formal_freeze=freeze,code_root=str(code))
        result['training_result']=ref(ready_path.parent/'training_result.json')
        result['driver']=ref(Path(__file__))
        plan=commands(contexts,output,code=code);result['commands']=plan
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        env.pop('CAR_ROUND7_LEASE_TOKEN',None)
        for item in plan:
            split=item['split']; start=time.time(); log=output/f'{split}.log'
            with log.open('x') as stream:
                completed=subprocess.run(item['command'],stdout=stream,stderr=subprocess.STDOUT,env=env)
            stage={'start_unix':start,'end_unix':time.time(),'exit_code':completed.returncode,'log':str(log)}
            result['splits'][split]=stage
            if completed.returncode:
                raise RuntimeError(f'{split} scoring failed with exit {completed.returncode}; see {log}')
            folder=output/split; receipt=load(folder/'receipt.json'); assessment=load(folder/'assessment.json')
            if (receipt.get('status')!='completed' or receipt.get('threads')!=2 or
                assessment.get('status')!='completed' or assessment.get('split')!=split or
                assessment.get('spec')!=ready['specs'][split] or
                assessment['child_native']['sha256']!=digest(contexts[split]['banks']['native']) or
                assessment['child_selected']['sha256']!=digest(contexts[split]['banks']['selected'])):
                raise ValueError(f'{split} scoring output identity differs')
            stage.update(status='completed',assessment=ref(folder/'assessment.json'),
                         scale_summary=ref(folder/'scale_summary.json'),decomposition=ref(folder/'decomposition.json'),
                         receipt=ref(folder/'receipt.json'))
            write(output/f'{split}_completed.json',{'schema':'car_round7_review_split_score_v1','arm':ready['arm'],
                'epoch':epoch,'reporting_epoch':epoch,'checkpoint':ready['checkpoint'],
                'checkpoint_sha256':ready['checkpoint']['sha256'],'ready_sha256':ready_hash,'split':split,**stage})
        result['status']='completed'
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        child_after=resource.getrusage(resource.RUSAGE_CHILDREN)
        result.update(end_unix=time.time(),wall_seconds=time.time()-started,
            driver_cpu_seconds=time.process_time()-self_cpu,
            child_cpu_seconds=(child_after.ru_utime-child_cpu.ru_utime)+(child_after.ru_stime-child_cpu.ru_stime))
        write(output/'score_result.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ready',type=Path,required=True,help='terminal four-bank ready.json for one arm')
    p.add_argument('--output',type=Path,required=True,help='new named DATA2 assessment attempt')
    run(p.parse_args())


if __name__=='__main__':main()
