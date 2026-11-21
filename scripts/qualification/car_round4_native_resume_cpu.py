#!/usr/bin/env python3
"""Bounded CPU execution parity: actual FFM SOURCE, two TRAIN truths, no efficacy claim."""
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

PROCESS_START = time.time()
CPU_START = time.process_time()

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT), str(ROOT/'scripts/posttraining')]


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): h.update(block)
    return h.hexdigest()


def child(args):
    import torch
    from prepare_car_round4 import resolve_config
    from phycoflow_reconstruction.data.manifest import SensorManifest
    from phycoflow_reconstruction.training.native_amortization import run_native_amortization
    from phycoflow_reconstruction.training.scalable_fitting import _digest
    from phycoflow_reconstruction.training.source import load_source_model, source_checkpoint_path
    from phycoflow_reconstruction.utils.reproducibility import seed_everything
    start, cpu = PROCESS_START, CPU_START
    result={'status':'failed','stage':args.child,'start_unix':start}
    try:
        torch.set_num_threads(4)
        cfg,_=resolve_config(args.round3)
        seed=int(cfg['runtime'].get('seed',42))
        deterministic=bool(cfg['runtime'].get('deterministic',True))
        strict=bool(cfg.get('coherence',{}).get('scalable_anchored_coherence',{}).get('common_prefit'))
        seed_everything(seed,deterministic,strict_algorithms=strict)
        model,data,metadata=load_source_model(cfg,torch.device('cpu'))
        try:
            source_digest=_digest(model.state_dict())
            assert callable(getattr(model,'training_backward',None))
            model.train_query_microbatch_size=512
            class TwoConditions:
                path=data.path
                normalizer=data.normalizer
                data_spec=data.data_spec
                trajectory_ids=data.trajectory_ids
                _items=data._items[:2]
                def __len__(self): return 2
                def __getitem__(self,index):
                    if index >= 2: raise IndexError(index)
                    return data[index]
            dataset=TwoConditions()
            ids=[f"{data.trajectory_ids[t]}:{f if f is not None else 'all'}" for t,f in dataset._items]
            original=SensorManifest.load(args.campaign/'configs/train_sensor_manifest.json')
            manifest=SensorManifest(dataset_path=original.dataset_path,dataset_fingerprint=original.dataset_fingerprint,
                split=original.split,protocol=original.protocol,indices={sid:original.indices[sid] for sid in ids})
            config={'seed':seed,'epochs':args.epochs,'device':'cpu','deterministic':deterministic,
                'strict_algorithms':strict,'query_points':1024,'temperature_field_index':list(data.field_names).index('T'),
                'identity':{'target_kind':'truth','source_sha256':digest(source_checkpoint_path(cfg)),
                    'membership_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
                    'manifest_sha256':manifest.digest()}}
            if args.resume:
                saved=json.loads((args.output/args.child/'resolved_config.json').read_text())
                saved['epochs']=args.epochs
                config=saved
            run_native_amortization(model=model,dataset=dataset,manifest=manifest,config=config,
                run_dir=args.output/args.child,resume=args.resume,inference_config=cfg)
            result.update(status='completed', sample_ids=ids, source_checkpoint_sha256=digest(source_checkpoint_path(cfg)),
                source_parameter_buffer_sha256=source_digest, source_metadata=metadata,
                query_points=1024, query_microbatch_size=512, batches_per_epoch=12, deterministic=deterministic,
                strict_algorithms=strict, resumed=args.resume, epochs=args.epochs,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        finally: data.close()
    except BaseException as exc:
        result.update(error_type=type(exc).__name__, error=str(exc)); raise
    finally:
        result.update(end_unix=time.time(), wall_seconds=time.time()-start, process_cpu_seconds=time.process_time()-cpu)
        name=f'{args.child}_epoch_{args.epochs}.json'
        (args.output/name).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


def compare(left,right,path='root'):
    import numpy as np
    import torch
    if isinstance(left,torch.Tensor):
        if not isinstance(right,torch.Tensor) or not torch.equal(left,right): raise AssertionError(path)
    elif isinstance(left,np.ndarray):
        if not isinstance(right,np.ndarray) or not np.array_equal(left,right): raise AssertionError(path)
    elif isinstance(left,dict):
        if set(left)!=set(right): raise AssertionError(path+' keys')
        for key in left:
            if key not in ('epoch_seconds','data_seconds','update_seconds'): compare(left[key],right[key],f'{path}.{key}')
    elif isinstance(left,(tuple,list)):
        if type(left)!=type(right) or len(left)!=len(right): raise AssertionError(path)
        for i,(a,b) in enumerate(zip(left,right)): compare(a,b,f'{path}[{i}]')
    elif left!=right: raise AssertionError(f'{path}: {left!r} != {right!r}')


def parent(args):
    import torch
    args.output.mkdir(parents=True,exist_ok=False)
    started,cpu=PROCESS_START,CPU_START
    result={'schema':'car_round4_native_real_ffm_resume_cpu_v1','status':'failed',
        'start_unix':started,'budget_wall_seconds':300,'gpu_used':False,'children':[],
        'purpose':'execution parity only; two original TRAIN truths, no model-efficacy conclusion',
        'script_sha256':digest(__file__)}
    try:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        for name,epochs,resume in [('continuous',2,False),('split',1,False),('split',2,True)]:
            remaining=295-(time.time()-started)
            if remaining<=0: raise TimeoutError('300 second parent budget exhausted')
            command=[sys.executable,__file__,'--campaign',str(args.campaign),'--round3',str(args.round3),
                '--output',str(args.output),'--child',name,'--epochs',str(epochs)]
            if resume: command.append('--resume')
            outcome=subprocess.run(command,env=env,timeout=remaining,check=False)
            path=args.output/f'{name}_epoch_{epochs}.json'
            result['children'].append({'stage':name,'epoch':epochs,'exit_code':outcome.returncode,
                                       'receipt':str(path),'sha256':digest(path) if path.is_file() else None})
            if outcome.returncode: raise RuntimeError(f'{name} epoch {epochs} exited {outcome.returncode}')
        continuous=torch.load(args.output/'continuous/checkpoint.pt',map_location='cpu',weights_only=False)
        resumed=torch.load(args.output/'split/checkpoint.pt',map_location='cpu',weights_only=False)
        compare(continuous,resumed)
        hashes=[json.loads(Path(row['receipt']).read_text())['source_parameter_buffer_sha256'] for row in result['children']]
        if len(set(hashes)) != 1: raise ValueError('SOURCE weight view differs across processes')
        result.update(status='completed',comparison='exact nested state equality excluding epoch/data/update timings',
            source_parameter_buffer_sha256=hashes[0],
            child_process_cpu_seconds=sum(json.loads(Path(row['receipt']).read_text())['process_cpu_seconds'] for row in result['children']))
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error=str(exc)); raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        result.update(end_unix=time.time(),wall_seconds=time.time()-started,parent_process_cpu_seconds=time.process_time()-cpu,
                      waited_children_cpu_seconds=usage.ru_utime+usage.ru_stime)
        (args.output/'receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--round3',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--child',choices=('continuous','split'))
    p.add_argument('--epochs',type=int,choices=(1,2))
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': p.error('CUDA_VISIBLE_DEVICES must be explicitly empty')
    if a.child: child(a)
    else: parent(a)
if __name__=='__main__': main()
