#!/usr/bin/env python3
"""Matched native VALIDATION100/J4 assessment of an origin-6000 Scale fork."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(os.environ.get('CAR_EXECUTION_ROOT',os.environ.get('CAR_REPO_ROOT',Path(__file__).resolve().parents[2]))).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.posttraining.assess_car_round2_checkpoints import hash_file,matched_rows
from scripts.posttraining import assess_car_round6 as legacy

ARM='FFM_SSS_EpochCalibrated_PI_Repair'


def reference(path):
    path=Path(path).resolve(strict=True)
    return {'path':str(path),'sha256':hash_file(path)}


def checked(ref):
    path=Path(ref['path']).resolve(strict=True)
    if hash_file(path)!=ref['sha256']:raise ValueError(f'bound evidence changed: {path}')
    return path


def validate_clock(spec,manifest,state,run,config_path,checkpoint_path,parent_snapshot):
    """Enforce phase age and the unchanged parent checkpoint lineage."""
    import yaml
    from phycoflow_reconstruction.training.run_store import config_digest
    phase=spec['phase_epoch']; expected=6000+12*phase
    if phase not in (100,225,500) or spec.get('phase_origin_step')!=6000 or spec.get('global_step')!=expected:
        raise ValueError('phase and global reporting clocks differ')
    if not isinstance(spec.get('variant'),str) or len(spec['variant'].strip())<3:
        raise ValueError('descriptive fork variant is required')
    if checked(spec['run_config'])!=config_path or checked(spec['checkpoint'])!=checkpoint_path:
        raise ValueError('run configuration or selected checkpoint differs')
    parent=parent_snapshot['checkpoint']
    checked(parent)
    origin={'parent_run':parent_snapshot['run'],'parent_checkpoint_sha256':parent['sha256'],'absolute_step':6000}
    if spec['variant']=='FFM_SSS_RotatingComplement_FieldFeedback':
        origin['scientific_transition']='rotating_complement_field_quadrature_512_v1'
    if (manifest.get('fork_origin')!=origin or state.get('fork_origin')!=origin
            or state.get('global_step')!=expected or manifest.get('run_id')!=run.name
            or state.get('config_sha256')!=config_digest(yaml.safe_load(config_path.read_text()))
            or manifest.get('config_sha256')!=state.get('config_sha256')
            or checkpoint_path != run/'checkpoints'/f'epoch_{phase:03d}.pt'):
        raise ValueError('fork checkpoint, configuration or source lineage differs')
    return expected


def verify_execution(spec):
    """Require the assessment to use the declared frozen scientific capsule."""
    root=Path(spec['execution_code_root']).resolve(strict=True)
    inventory_path=checked(spec['execution_inventory'])
    if root!=ROOT or inventory_path!=root/'execution_receipt.json':raise ValueError('execution capsule identity differs')
    inventory=json.loads(inventory_path.read_text())
    if inventory.get('sha256')!=root.name:raise ValueError('execution capsule provenance differs')
    required=('scripts/posttraining/assess_car_round6.py','scripts/posttraining/assess_car_round5.py','scripts/posttraining/assess_car_round2_checkpoints.py')
    if any(k not in inventory['files'] for k in required):raise ValueError('assessment helpers absent from capsule inventory')
    for name,expected in inventory['files'].items():
        if name.startswith('src/') or name in required:
            if hash_file(root/name)!=expected:raise ValueError(f'execution file changed: {name}')
    return inventory


def make_context(campaign,spec_path,spec):
    campaign=campaign.resolve(strict=True);round6=(campaign.parent/'round_6').resolve(strict=True)
    if spec.get('schema')!='car_scale_fork_validation_spec_v1' or spec.get('test_locked') is not True:
        raise ValueError('locked fork assessment specification required')
    if checked(spec['parent_snapshot'])!=round6/'assessment/Epoch500_Validation100_J4_20261125T042223Z/snapshot/snapshot.json':
        raise ValueError('parent snapshot differs from original epoch500')
    verify_execution(spec)
    ctx=legacy.context(round6)
    # Compatibility adapter: legacy scoring uses the original arm key internally;
    # the variant and fork identity are retained in this spec and receipts.
    original=ctx['prepared'];ctx['campaign']=campaign;ctx['prepared_path']=spec_path
    ctx['prepared']={'arms':{ARM:spec['run_config']},'source_reference':original['source_reference']}
    if spec['source_bank']!=reference(ctx['source_receipt']) or spec['panel']!=reference(ctx['panel_path']):
        raise ValueError('SOURCE bank or frozen validation panel differs')
    return ctx


def validate_fork(campaign,spec):
    import torch
    run=Path(spec['run']).resolve(strict=True)
    if (campaign/'runs').resolve() not in run.parents:raise ValueError('fork run must belong to followup campaign')
    config=(run/'resolved_config.yaml').resolve(strict=True)
    checkpoint=(run/'checkpoints'/f"epoch_{spec['phase_epoch']:03d}.pt").resolve(strict=True)
    manifest_path=checked(spec['run_manifest'])
    manifest=json.loads(manifest_path.read_text());state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    step=validate_clock(spec,manifest,state,run,config,checkpoint,json.loads(checked(spec['parent_snapshot']).read_text()))
    return run,config,checkpoint,state,step


def generate(spec_path,spec,ctx,device,output):
    import torch
    import yaml
    from phycoflow_reconstruction.data.factory import open_field_dataset
    from phycoflow_reconstruction.data.manifest import dataset_fingerprint
    from phycoflow_reconstruction.data.sensor_protocols import SensorProtocol,build_observation_batch
    from phycoflow_reconstruction.training.model_lifecycle import load_training_aux_state
    from phycoflow_reconstruction.training.rollout import differentiable_reconstruction
    from phycoflow_reconstruction.training.run_store import load_model_state_strict
    from phycoflow_reconstruction.training.source import load_source_model
    from phycoflow_reconstruction.training.scalable_fitting import _digest
    if not os.environ.get('CAR_ROUND6_LEASE_TOKEN') or device.type!='cuda':raise ValueError('GPU generation requires followup lease')
    campaign=ctx['campaign'];run,config,checkpoint,state,step=validate_fork(campaign,spec)
    if state.get('source_hashes',{}).get('checkpoint')!=ctx['source']['source_sha256']:raise ValueError('fork SOURCE identity differs')
    cfg=yaml.safe_load(config.read_text());output=output.resolve()
    if (campaign/'assessment' not in output.parents or output.exists()
            or output!=Path(spec['bank_output']).resolve()):raise ValueError('declared new followup assessment bank required')
    output.mkdir(parents=True);started=time.time();cpu=time.process_time();train=dataset=None
    receipt={'schema':'car_round3_native_bank_v1','status':'failed','start_unix':started,'arm':ARM,'variant':spec['variant'],
        'epoch':spec['phase_epoch'],'phase_origin_step':6000,'global_step':step,'fork_spec':reference(spec_path),
        'fork_origin':state['fork_origin'],'checkpoint':str(checkpoint),'checkpoint_sha256':hash_file(checkpoint),
        'config_path':str(config),'config_sha256':hash_file(config),'prepared_sha256':hash_file(spec_path),
        'source_bank_sha256':hash_file(ctx['source_receipt']),'panel_sha256':hash_file(ctx['panel_path']),
        'source_checkpoint_sha256':ctx['source']['source_sha256'],'members':4,'sample_ids':ctx['source']['sample_ids'],
        'lease_token':os.environ['CAR_ROUND6_LEASE_TOKEN'],'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'driver_sha256':hash_file(Path(__file__)),'execution_inventory':spec['execution_inventory'],
        'test_locked':True,'raw_checkpoint_output':True}
    try:
        model,train,_=load_source_model(cfg,device)
        if _digest(model.state_dict())!=ctx['source']['source_parameter_buffer_sha256']:raise ValueError('SOURCE materialization differs')
        load_model_state_strict(model,state['model']);load_training_aux_state(model,state);model.eval()
        dataset=open_field_dataset(cfg['dataset'],split='validation',normalizer=train.normalizer);panel=ctx['panel']
        if (len(dataset)!=panel['population'] or dataset.normalizer.digest()!=panel['normalizer_sha256']
                or dataset_fingerprint(dataset.path)!=panel['dataset_fingerprint']):raise ValueError('validation dataset or normalizer differs')
        protocol=SensorProtocol(field_counts={'T':256},seed=2047);files=[]
        with torch.no_grad():
            for ordinal,(index,sample_id) in enumerate(zip(panel['sample_indices'],panel['sample_ids'],strict=True)):
                source_path=ctx['source_dir']/f'{ordinal:04d}.pt'
                if hash_file(source_path)!=ctx['source']['files_sha256'][ordinal]:raise ValueError('SOURCE bank row changed')
                source=torch.load(source_path,map_location='cpu',weights_only=True);sample=dataset[index]
                actual=f'{dataset.trajectory_ids[dataset._items[index][0]]}:{dataset._items[index][1]}'
                if actual!=sample_id or source['sample_id']!=sample_id or sample.values.shape[0]!=40300:raise ValueError('matched native sample differs')
                obs={sample_id:list(zip(source['obs_indices'][0].tolist(),source['obs_fields'][0].tolist(),strict=True))}
                batch=build_observation_batch([sample],protocol,query_indices=None,manifest_indices=obs).to(device)
                members=[differentiable_reconstruction(model,replace(batch,target_fields=None),steps=int(cfg['rollout']['steps']),solver=cfg['rollout']['solver'],generator=torch.Generator(device=device).manual_seed(seed),observation_config=cfg['observation_consistency']).detach().cpu() for seed in source['member_seeds']]
                row={'sample_id':sample_id,'sample_index':index,'members_model':torch.stack(members),'truth_model':batch.target_fields.detach().cpu(),
                    'coordinates_model':batch.query_coords.detach().cpu(),'coordinates_raw':sample.coordinates_raw.cpu(),'native_masses':dataset.physical_weights.cpu(),
                    'obs_indices':batch.obs_indices.detach().cpu(),'obs_values':batch.obs_values.detach().cpu(),'obs_fields':batch.obs_field_ids.detach().cpu(),
                    'member_seeds':source['member_seeds'],'normalizer_offset':dataset.normalizer.offset.cpu(),'normalizer_scale':dataset.normalizer.scale.cpu()}
                matched_rows(source,row);path=output/f'{ordinal:04d}.pt';torch.save(row,path);files.append(hash_file(path))
        receipt.update(status='completed',files_sha256=files,device_name=torch.cuda.get_device_name(device))
    except BaseException as exc:
        receipt.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:
        if dataset is not None:dataset.close()
        if train is not None:train.close()
        receipt.update(wall_seconds=time.time()-started,cpu_seconds=time.process_time()-cpu)
        (output/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')


def score(spec_path,spec,ctx,bank,output):
    import argparse
    if bank.resolve()!=Path(spec['bank_output']).resolve():raise ValueError('declared bank identity differs')
    validate_fork(ctx['campaign'],spec)
    legacy.check_bank(bank,ctx,spec['phase_epoch'],ARM)
    legacy.review(argparse.Namespace(bank=bank,epoch=spec['phase_epoch'],output=output),ctx)
    assessment=output/'assessment.json';result=json.loads(assessment.read_text())
    # Preserve legacy score bytes; bind phase identity in a companion, not by rewriting it.
    (output/'fork_identity.json').write_text(json.dumps({'schema':'car_scale_fork_assessment_identity_v1','spec':reference(spec_path),
        'assessment':reference(assessment),'bank_receipt':reference(bank/'receipt.json'),'variant':spec['variant'],
        'phase_epoch':spec['phase_epoch'],'phase_origin_step':6000,'global_step':spec['global_step'],'test_locked':True},indent=2)+'\n')


def authorize_endpoint(campaign,spec_path,spec,review_path,bank,decision,rationale,output):
    """Record an explicitly chosen terminal task authorization for the fork phase."""
    validate_fork(campaign,spec)
    if bank.resolve()!=Path(spec['bank_output']).resolve():raise ValueError('declared bank identity differs')
    review=json.loads(review_path.read_text());phase=spec['phase_epoch']
    if (not rationale.strip() or decision!={100:'stop_at_epoch_100',225:'endpoint_at_epoch_225',500:'endpoint_at_epoch_500'}[phase]
            or review.get('epoch')!=phase or review.get('task_scope')!='both tasks excluded'
            or review.get('bank_receipt_sha256')!=hash_file(bank/'receipt.json')
            or review.get('prepared_sha256')!=hash_file(spec_path)
            or review.get('checkpoint_sha256')!=spec['checkpoint']['sha256']):
        raise ValueError('explicit terminal choice must bind the task-free fork and bank')
    output=output.resolve()
    if campaign/'assessment' not in output.parents or output.exists():raise ValueError('new followup task decision required')
    output.parent.mkdir(parents=True,exist_ok=True)
    value={'schema':'car_scale_fork_terminal_task_decision_v1','status':'completed','epoch':phase,
           'phase_origin_step':6000,'global_step':spec['global_step'],'variant':spec['variant'],'decision':decision,
           'authorize_endpoint_tasks':True,'review_sha256':hash_file(review_path),'bank_receipt_sha256':hash_file(bank/'receipt.json'),
           'spec':reference(spec_path),'rationale':rationale.strip(),'test_locked':True}
    with output.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def compare(campaign,left_path,right_path,output):
    """Paired cached per-condition differences, right minus left; no inference."""
    left=json.loads(left_path.read_text());right=json.loads(right_path.read_text())
    identities=('split','population','members','field_order','sample_ids','source_checkpoint_sha256',
                'source_bank_sha256','panel_sha256','contract_sha256','core_sha256','source_baseline_sha256','normalizers','native_allowance')
    if any(left.get(k)!=right.get(k) for k in identities) or left.get('split')!='validation' or len(left.get('sample_ids',[]))!=100 or left.get('members')!=4:
        raise ValueError('paired validation identities differ')
    if left.get('task_scope')!='both tasks excluded' or right.get('task_scope')!='both tasks excluded':raise ValueError('task-free cached assessments required')
    for item in (left,right):
        if len(item['per_condition_fields']['child'])!=100:raise ValueError('complete per-condition child records required')
    a=left['per_condition_fields']['child'];b=right['per_condition_fields']['child'];norm=right['normalizers']
    core={k:legacy.intervals([x['scores'][k] for x in a],[x['scores'][k] for x in b],scale=norm[k]) for k in ('state','scale','shape')}
    fields={k:legacy.intervals([x[k] for x in a],[x[k] for x in b]) for k in ('signed_mean_risk','member_mse','ensemble_mean_mse','unbiased_member_variance')}
    output=output.resolve()
    if campaign/'assessment' not in output.parents or output.exists():raise ValueError('new followup comparison path required')
    output.parent.mkdir(parents=True,exist_ok=True)
    result={'schema':'car_scale_fork_paired_validation_v1','status':'completed','left':reference(left_path),'right':reference(right_path),
            'difference':'right minus left','sample_count':100,'members':4,'core_in_frozen_train_units':core,'physical_fields':fields,
            'interpretation':'paired condition bootstrap on reused same-trajectory development VALIDATION; package comparison, not independent trajectory evidence',
            'test_locked':True}
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--spec',type=Path,required=True)
    sub=p.add_subparsers(dest='command',required=True)
    x=sub.add_parser('generate');x.add_argument('--device',type=__import__('torch').device,required=True);x.add_argument('--output',type=Path,required=True)
    x=sub.add_parser('score');x.add_argument('--bank',type=Path,required=True);x.add_argument('--output',type=Path,required=True)
    x=sub.add_parser('compare');x.add_argument('--left',type=Path,required=True);x.add_argument('--right',type=Path,required=True);x.add_argument('--output',type=Path,required=True)
    x=sub.add_parser('authorize-endpoint');x.add_argument('--bank',type=Path,required=True);x.add_argument('--review',type=Path,required=True);x.add_argument('--decision',choices=('stop_at_epoch_100','endpoint_at_epoch_225','endpoint_at_epoch_500'),required=True);x.add_argument('--rationale',required=True);x.add_argument('--output',type=Path,required=True)
    x=sub.add_parser('endpoint');x.add_argument('--bank',type=Path,required=True);x.add_argument('--review',type=Path,required=True);x.add_argument('--decision',type=Path,required=True);x.add_argument('--output',type=Path,required=True)
    a=p.parse_args();campaign=a.campaign.resolve(strict=True);spec_path=a.spec.resolve(strict=True);spec=json.loads(spec_path.read_text());ctx=make_context(campaign,spec_path,spec)
    if a.command=='generate':generate(spec_path,spec,ctx,a.device,a.output)
    elif a.command=='compare':
        validate_fork(campaign,spec)
        right=json.loads(a.right.read_text())
        if (right.get('prepared_sha256')!=hash_file(spec_path) or right.get('checkpoint_sha256')!=spec['checkpoint']['sha256']
                or right.get('epoch')!=spec['phase_epoch']):raise ValueError('right assessment differs from declared fork')
        compare(campaign,a.left.resolve(strict=True),a.right.resolve(strict=True),a.output)
    elif a.command=='score':score(spec_path,spec,ctx,a.bank.resolve(strict=True),a.output)
    elif a.command=='authorize-endpoint':authorize_endpoint(campaign,spec_path,spec,a.review.resolve(strict=True),a.bank.resolve(strict=True),a.decision,a.rationale,a.output)
    else:
        validate_fork(campaign,spec)
        if a.bank.resolve()!=Path(spec['bank_output']).resolve():raise ValueError('declared bank identity differs')
        legacy.endpoint(argparse.Namespace(bank=a.bank,review=a.review,decision=a.decision,output=a.output),ctx)

if __name__=='__main__':main()
