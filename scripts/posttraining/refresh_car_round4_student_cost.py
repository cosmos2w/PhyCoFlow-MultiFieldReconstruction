#!/usr/bin/env python3
"""Closed-phase CPU-only native target I/O measurement and cost forecast.

Requires an admitted terminal teacher. This produces evidence, not a launch or
adoption decision, and does not invoke a model or alter the frozen learner.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import fcntl
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def bind(path):
    path=Path(path).resolve(strict=True)
    return {'path':str(path),'sha256':sha(path)}


def read(path): return json.loads(Path(path).read_text())


def finite(x, label):
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0:
        raise ValueError(f'{label} must be finite and nonnegative')
    return float(x)


def forecast(*, prior_gpu, teacher_gpu, update_per_batch, qualified_batch_input,
             measured_batch_input, measured_target_io, assessment_gpu, assessment_cpu,
             assessment_wall, safety=1.2):
    """Forecast serial occupied time, charging target CPU I/O once per visit."""
    values=locals().copy()
    for key,value in values.items(): finite(value,key)
    if safety < 1: raise ValueError('safety factor must be at least one')
    base_input=max(measured_batch_input, 12*qualified_batch_input)
    truth_epoch=safety*(12*update_per_batch+base_input+2.)
    teacher_epoch=safety*(12*update_per_batch+base_input+measured_target_io+2.)
    startup_each=60.
    def horizon(epochs):
        startup_total=startup_each*(1 if epochs==100 else 2)
        candidate_student=startup_total+epochs*teacher_epoch
        control_student=startup_total+epochs*truth_epoch
        candidate=prior_gpu+teacher_gpu+candidate_student+assessment_gpu
        whole=candidate+control_student
        return {'epochs':epochs,'startup_allowance_per_student_seconds':startup_total,
                'candidate_student_gpu_seconds':candidate_student,
                'control_student_gpu_seconds':control_student,'candidate_inclusive_gpu_seconds':candidate,
                'whole_round_gpu_seconds':whole,'candidate_twofold_budget_fits':candidate<=35317,
                'whole_24_gpu_hour_budget_fits':whole<=86400,
                'student_10_gpu_hour_budget_fits':candidate_student+control_student<=36000,
                'assessment_wall_fraction_of_teacher_and_students':assessment_wall/(teacher_gpu+candidate_student+control_student)}
    return {'safety_factor':safety,'prior_gpu_seconds':prior_gpu,'teacher_gpu_seconds':teacher_gpu,
            'qualified_native_update_seconds_per_batch':update_per_batch,
            'qualified_input_seconds_per_batch':qualified_batch_input,
            'measured_batch_input_epoch_seconds':measured_batch_input,
            'measured_repaired_target_epoch_seconds':measured_target_io,
            'charged_base_input_epoch_seconds':base_input,
            'student_epoch_forecast_seconds':{'teacher_targets':teacher_epoch,'dense_truth':truth_epoch},
            'startup_allowance_per_student_seconds':startup_each,'persistence_allowance_per_epoch_seconds':2.,
            'assessment_gpu_seconds':assessment_gpu,'assessment_cpu_worker_seconds':assessment_cpu,
            'assessment_wall_seconds_including_cpu_tail':assessment_wall,
            'review_100':horizon(100),'conditional_endpoint_500':horizon(500)}


def first_epoch_stream(seed, population, torch):
    conditions=torch.Generator().manual_seed(seed+17)
    queries=torch.Generator().manual_seed(seed+37)
    atoms=torch.Generator().manual_seed(seed+59)
    if population < 768: raise ValueError('first reporting epoch requires 768 conditions')
    order=torch.randperm(population,generator=conditions).tolist()
    for batch_number in range(12):
        selected=order[64*batch_number:64*(batch_number+1)]
        query_seed=int(torch.randint(0,2**31-1,(),generator=queries))
        atom=torch.randint(0,4,(64,),generator=atoms)
        yield selected,query_seed,atom


def decoded_target(row, index, atom, query, support_lookup, corrections, torch):
    """The frozen native reader's sparse model correction and physical decode."""
    values=row['members_model'][int(atom),0,query].to(torch.float64)
    locations=support_lookup[query]; included=locations>=0
    values[included]+=corrections[index,int(atom),locations[included]]
    return values*row['normalizer_scale'].double()+row['normalizer_offset'].double()


def closed_ledger(c):
    path=c/'operations/gpu_ledger.json'; ledger=read(path)
    if ledger.get('schema')!='car_gpu_ledger_v1' or ledger.get('active'):
        raise ValueError('all GPU leases must be closed before student cost refresh')
    return path,ledger


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--learning-code-receipt',type=Path,required=True)
    args=p.parse_args()
    c=args.campaign.resolve(strict=True)
    with (c/'operations/gpu_ledger.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        execute(c,args)


def closed_costs(c, ledger, launch, manifest_hash, teacher_path):
    # Enclosing leases are charged once; receipt CPU values are not nested into GPU work.
    seen=set(); teacher_gpu=other_gpu=0.; cpu_clocks=[]; job_refs=[]; complete=False
    for entry in ledger['closed']:
        token=entry['token']; phase=entry['phase']
        if token in seen: raise ValueError('duplicate ledger token')
        seen.add(token); seconds=finite(entry['seconds'],'closed lease')
        if phase=='endpoint_teacher':
            teacher_gpu+=seconds; path=c/'operations/teacher_jobs'/f'{token}.json'; doc=read(path)
            if (doc.get('token')!=token or doc.get('manifest_sha256')!=manifest_hash or doc.get('status') not in ('failed','paused','completed')
                    or abs(finite(doc['wall_seconds'],'teacher receipt')-seconds)>.01):
                raise ValueError('teacher ledger/receipt mismatch')
            if doc.get('status')=='completed':
                if doc.get('completed_epochs')!=100 or Path(doc['checkpoint']).resolve()!=teacher_path:
                    raise ValueError('completed teacher lease does not match qualified checkpoint')
                complete=True
            job_refs.append(bind(path))
        elif phase in ('qualify','teacher_numerical_qualification','current_route_timing','generate'):
            other_gpu+=seconds
            directory={'qualify':'preparation_jobs','generate':'preparation_jobs',
                       'teacher_numerical_qualification':'teacher_jobs','current_route_timing':'current_route_timing'}[phase]
            path=c/'operations'/directory/f'{token}.json'; doc=read(path)
            if doc.get('token')!=token or abs(finite(doc['wall_seconds'],'job receipt')-seconds)>.01:
                raise ValueError('preparation ledger/receipt mismatch')
            job_refs.append(bind(path))
        else: raise ValueError(f'unexpected pre-student phase: {phase}')
        cpu=entry.get('cpu_seconds',doc.get('process_cpu_seconds'))
        cpu_clocks.append({'token':token,'phase':phase,'cpu_seconds':None if cpu is None else finite(cpu,'process CPU')})
    if not complete: raise ValueError('completed terminal teacher lease missing')
    expected_other=finite(launch['measured_qualification_gpu_seconds'],'qualification')+finite(launch['measured_generation_gpu_seconds'],'generation')
    if abs(other_gpu-expected_other)>.01: raise ValueError('preparation ledger differs from launch forecast')
    failed_refs=launch['bindings']['failed_startups']
    failed_paths={Path(x['path']).resolve():x for x in failed_refs}
    if set(failed_paths)!={p.resolve() for p in (c/'operations/failed_launches').glob('*.json')}:
        raise ValueError('failed startup costs differ from launch forecast')
    failed_gpu=0.
    for path,ref in failed_paths.items():
        if sha(path)!=ref['sha256']: raise ValueError('failed startup receipt changed')
        failed_gpu+=finite(read(path)['conservative_gpu_seconds_charge'],'failed startup charge')
    prior=other_gpu+failed_gpu+finite(launch['unmeasured_startup_allowance_gpu_seconds'],'startup allowance')
    return prior,teacher_gpu,cpu_clocks,job_refs,failed_paths


def execute(c,args):
    wall_start,cpu_start=time.time(),time.process_time()
    ledger_path,ledger=closed_ledger(c)  # No model, job, bank or teacher reads before this guard.
    timing_path=c/'operations/student_target_io_timing.json'
    output_path=c/'operations/student_cost_forecast.json'
    if timing_path.exists() or output_path.exists(): raise FileExistsError('student cost evidence already exists')
    manifest_path=c/'configs/train_manifest.json'; manifest=read(manifest_path); mh=sha(manifest_path)
    sensor_path=c/'configs/train_sensor_manifest.json'
    bank_path=c/'artifacts/source_bank/receipt.json'; bank=read(bank_path)
    gate_path=c/'operations/feasibility_gate.json'; gate=read(gate_path)
    launch_path=c/'operations/teacher_launch_forecast.json'; launch=read(launch_path)
    teacher_gate_path=c/'operations/teacher_qualification.json'; teacher_gate=read(teacher_gate_path)
    if (gate.get('admitted') is not True or launch.get('admitted') is not True
            or gate.get('manifest_sha256')!=mh or launch.get('manifest_sha256')!=mh
            or launch['bindings']['initial_gate']['sha256']!=sha(gate_path)
            or teacher_gate.get('schema')!='car_round4_teacher_qualification_v1'
            or teacher_gate.get('admitted') is not True or teacher_gate.get('completed_epochs')!=100
            or teacher_gate.get('train_manifest_sha256')!=mh):
        raise ValueError('manifest-bound admitted initial forecasts and completed qualified teacher required')
    teacher_path=(c/teacher_gate['checkpoint']).resolve(strict=True)
    if not teacher_path.is_relative_to(c) or sha(teacher_path)!=teacher_gate['teacher_checkpoint_sha256']:
        raise ValueError('teacher checkpoint identity differs')
    if (bank.get('schema')!='car_round4_source_bank_v1' or bank.get('status')!='completed'
            or bank.get('manifest_sha256')!=mh or bank.get('population')!=8000
            or bank.get('members')!=4 or bank.get('native_points')!=40300 or bank.get('units')!='model field units'
            or bank.get('source_checkpoint_sha256')!=manifest.get('source_checkpoint_sha256')
            or len(bank.get('rows_sha256',[]))!=8000):
        raise ValueError('complete matched native SOURCE bank required')
    learning_receipt=Path(args.learning_code_receipt).resolve(strict=True)
    if not learning_receipt.is_relative_to(c): raise ValueError('learning code receipt outside campaign')
    learning=read(learning_receipt)
    code=Path(learning['code_root']).resolve(strict=True)
    inventory=read(code/'execution_receipt.json'); files=inventory['files']
    if (not code.is_relative_to((c/'operations/code').resolve()) or code.name!=learning.get('code_sha256')
            or inventory.get('schema')!='car_execution_code_v1' or inventory.get('sha256')!=code.name
            or hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()!=code.name):
        raise ValueError('learning capsule identity differs')
    for name in ('scripts/posttraining/run_car_round4_native.py','src/phycoflow_reconstruction/training/native_amortization.py'):
        if sha(code/name)!=files[name]: raise ValueError(f'frozen learner changed: {name}')
    prior,teacher_gpu,cpu_clocks,job_refs,failed_paths=closed_costs(c,ledger,launch,mh,teacher_path)
    source_ref=gate['bindings']['source_qualification']; source_path=Path(source_ref['path'])
    if sha(source_path)!=source_ref['sha256']: raise ValueError('native qualification changed')
    qual=read(source_path)
    sys.path[:0]=[str(code/'src'),str(code),str(code/'scripts/posttraining')]
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    from phycoflow_reconstruction.data.factory import open_field_dataset
    from phycoflow_reconstruction.data.manifest import SensorManifest, dataset_fingerprint
    from phycoflow_reconstruction.data.sensor_protocols import SensorProtocol, build_observation_batch
    from phycoflow_reconstruction.training.source import source_checkpoint_path
    from prepare_car_round4 import resolve_config
    torch.set_num_threads(4)
    cfg, config_binding=resolve_config(c.parent/'round_3')
    if config_binding!=manifest['config'] or sha(source_checkpoint_path(cfg))!=manifest['source_checkpoint_sha256']:
        raise ValueError('frozen SOURCE or inherited config differs')
    sensor=SensorManifest.load(sensor_path)
    field_index=manifest['field_order'].index('T')
    if (sensor.split.lower()!='train' or sensor.query_indices is not None
            or sensor.dataset_fingerprint!=manifest['dataset_fingerprint']
            or sensor.indices!={x['sample_id']:[[p,field_index] for p in x['observation_indices']] for x in manifest['items']}):
        raise ValueError('fixed observation manifest differs')
    data=open_field_dataset(cfg['dataset'],split='train')
    try:
        ids=[f"{data.trajectory_ids[t]}:{frame if frame is not None else 'all'}" for t,frame in data._items]
        if (len(data)!=8000 or ids!=[x['sample_id'] for x in manifest['items']]
                or data.normalizer.digest()!=manifest['normalizer_sha256']
                or dataset_fingerprint(data.path)!=manifest['dataset_fingerprint']):
            raise ValueError('live TRAIN data identity differs')
        teacher=torch.load(teacher_path,map_location='cpu',weights_only=True,mmap=True)
        source_digest=bank['source_parameter_buffer_sha256']
        corrections=teacher['corrections']
        support=torch.as_tensor(manifest['query_indices'],dtype=torch.long)
        if (teacher.get('schema')!='car_round4_teacher_checkpoint_v1' or teacher.get('completed_epochs')!=100
                or teacher.get('train_manifest_sha256')!=mh or teacher.get('source_parameter_buffer_sha256')!=source_digest
                or teacher.get('query_indices')!=manifest['query_indices'] or corrections.dtype!=torch.float64
                or corrections.shape!=(8000,4,len(support),len(manifest['field_order']))):
            raise ValueError('teacher checkpoint differs from native target contract')
        support_lookup=torch.full((int(manifest['native_points']),),-1,dtype=torch.long)
        support_lookup[support]=torch.arange(len(support))
        id_index={sid:i for i,sid in enumerate(ids)}
        bank_dir=bank_path.parent
        # Match the native learner's reader: hash once per row, then compare its
        # sidecar with the receipt-bound digest, without a redundant second read.
        def exact_reader(sample_ids, atoms, queries):
            outputs=[]
            for sid,atom,query in zip(sample_ids,atoms,queries):
                index=id_index[sid]; row_path=bank_dir/f'{index:04d}.pt'
                digest=sha(row_path)
                if row_path.with_suffix('.sha256').read_text().strip()!=digest or digest!=bank['rows_sha256'][index]:
                    raise ValueError('SOURCE bank row checksum mismatch')
                row=torch.load(row_path,map_location='cpu',weights_only=True)
                if (row['manifest_sha256']!=mh or row['sample_id']!=sid
                        or row['source_parameter_buffer_sha256']!=source_digest
                        or not torch.equal(row['normalizer_offset'],data.normalizer.offset)
                        or not torch.equal(row['normalizer_scale'],data.normalizer.scale)):
                    raise ValueError('SOURCE row identity mismatch')
                outputs.append(decoded_target(row,index,atom,query,support_lookup,corrections,torch))
            return torch.stack(outputs)
        rows=[]; seed=int(cfg['runtime'].get('seed',42))
        for indices,query_seed,atoms in first_epoch_stream(seed,len(data),torch):
            a,b=time.perf_counter(),time.process_time()
            protocol=replace(SensorProtocol(**sensor.protocol),seed=query_seed)
            batch=build_observation_batch([data[i] for i in indices],protocol,
                                          query_points=int(cfg['model']['query_points']),manifest_indices=sensor.indices)
            if tuple(ids[i] for i in indices)!=batch.sample_ids: raise ValueError('batch sample order differs')
            base_wall,base_cpu=time.perf_counter()-a,time.process_time()-b
            a,b=time.perf_counter(),time.process_time()
            target=torch.as_tensor(exact_reader(batch.sample_ids,atoms,batch.metadata['query_indices']))
            if target.shape!=batch.target_fields.shape or not torch.isfinite(target).all():
                raise ValueError('native physical targets invalid')
            encoded=data.normalizer.encode(target).to(dtype=batch.target_fields.dtype)
            if not torch.isfinite(encoded).all(): raise ValueError('native encoded targets invalid')
            rows.append({'condition_indices':indices,'query_seed':query_seed,'teacher_atom_indices':atoms.tolist(),
                         'batch_input_wall_seconds':base_wall,'batch_input_cpu_seconds':base_cpu,
                         'target_wall_seconds':time.perf_counter()-a,'target_cpu_seconds':time.process_time()-b})
        del teacher,corrections
    finally:
        data.close()
    if (qual.get('status')!='completed' or qual.get('manifest_sha256')!=mh or qual.get('native_batch')!=64
            or qual.get('native_query_points')!=int(cfg['model']['query_points'])):
        raise ValueError('qualified native timing differs')
    input_wall=sum(x['batch_input_wall_seconds'] for x in rows)
    target_wall=sum(x['target_wall_seconds'] for x in rows)
    result=forecast(prior_gpu=prior,teacher_gpu=teacher_gpu,update_per_batch=max(qual['native_update_seconds']),
                    qualified_batch_input=qual['native_batch_input_seconds'],measured_batch_input=input_wall,
                    measured_target_io=target_wall,assessment_gpu=gate['forecast_gpu_seconds']['assessment'],
                    assessment_cpu=gate['assessment_cpu_worker_seconds'],assessment_wall=gate['assessment_added_wall_seconds'],
                    safety=gate['safety_factor'])
    timing={'schema':'car_round4_student_target_io_timing_v1','status':'completed','gpu_used':False,'scope':'one native first reporting epoch, CPU-only data/target preparation; no model or optimizer',
            'train_manifest_sha256':mh,'teacher_checkpoint_sha256':teacher_gate['teacher_checkpoint_sha256'],
            'seed':seed,'batch_count':12,'batch_size':64,'condition_rng_seed':seed+17,'query_rng_seed':seed+37,'atom_rng_seed':seed+59,
            'batches':rows,'batch_input_wall_seconds':input_wall,'target_wall_seconds':target_wall,
            'batch_input_cpu_seconds':sum(x['batch_input_cpu_seconds'] for x in rows),
            'target_cpu_seconds':sum(x['target_cpu_seconds'] for x in rows),
            'tool_sha256':sha(__file__),'bindings':{'manifest':bind(manifest_path),'sensor_manifest':bind(sensor_path),
            'source_bank_receipt':bind(bank_path),'teacher_gate':bind(teacher_gate_path),'teacher_checkpoint':bind(teacher_path),
            'learning_code_receipt':bind(learning_receipt),'learning_capsule_receipt':bind(code/'execution_receipt.json')}}
    timing['start_unix']=wall_start
    timing['end_unix']=time.time()
    timing['wall_seconds']=timing['end_unix']-wall_start
    timing['process_cpu_seconds']=time.process_time()-cpu_start
    timing['timer_scope']='standalone CPU measurement and input binding; excludes receipt serialization; nested batch clocks are not additive'
    with timing_path.open('x') as f: json.dump(timing,f,indent=2,allow_nan=False); f.write('\n')
    result.update(schema='car_round4_student_cost_forecast_v1',train_manifest_sha256=mh,
                  interpretation='Bounded planning evidence only; assessment remains a forecast and CPU I/O measurements include cache-state effects; no student launch or adoption decision.',
                  standalone_cpu_measurement_seconds=timing['process_cpu_seconds'],
                  measured_batch_input_cpu_seconds=timing['batch_input_cpu_seconds'],measured_repaired_target_cpu_seconds=timing['target_cpu_seconds'],
                  student_data_cpu_only_forecast_seconds={str(n):{
                      'teacher_targets':n*(timing['batch_input_cpu_seconds']+timing['target_cpu_seconds']),
                      'dense_truth':n*timing['batch_input_cpu_seconds']} for n in (100,500)},
                  ledger_cpu_clocks=cpu_clocks,unknown_cpu_clocks=[x['token'] for x in cpu_clocks if x['cpu_seconds'] is None],
                  failed_startup_cpu_clocks=[{'receipt':str(path),
                      'cpu_seconds':read(path).get('process_cpu_seconds')} for path in failed_paths],
                  teacher_cpu_worker_seconds=(sum(x['cpu_seconds'] for x in cpu_clocks if x['phase']=='endpoint_teacher')
                      if all(x['cpu_seconds'] is not None for x in cpu_clocks if x['phase']=='endpoint_teacher') else None),
                  student_model_update_cpu_seconds='unknown; GPU qualification does not isolate CPU update cost',
                  bindings={'timing':bind(timing_path),'ledger':bind(ledger_path),'initial_gate':bind(gate_path),
                            'teacher_launch_forecast':bind(launch_path),'closed_job_receipts':job_refs,
                            'failed_startups':[bind(x) for x in failed_paths],'source_qualification':bind(source_path),
                            'tool':bind(Path(__file__))})
    with output_path.open('x') as f: json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    print(output_path)


if __name__=='__main__': main()
