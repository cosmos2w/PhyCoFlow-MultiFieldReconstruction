#!/usr/bin/env python3
"""Prepare immutable Scale fork assessment specs and paired cached comparisons."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(os.environ.get('CAR_EXECUTION_ROOT',os.environ.get('CAR_REPO_ROOT',Path(__file__).resolve().parents[2]))).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.qualification.car_scale_fork_validation import reference,checked,validate_clock,verify_execution
from scripts.posttraining import assess_car_round6 as legacy
from scripts.posttraining.assess_car_round2_checkpoints import hash_file,matched_rows

DEFINITIONS=('Published_Core_Default','Complete_Core_Prefit','Complete_Expanded_Prefit')


def write_new(path,value):
    path=Path(path).resolve()
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    return path


def base_context(campaign):
    round6=(campaign.parent/'round_6').resolve(strict=True);ctx=legacy.context(round6)
    start_bank=round6/'assessment/Epoch500_Validation100_J4_20261125T042223Z/bank/receipt.json'
    start=json.loads(start_bank.read_text());source=ctx['source'];panel=ctx['panel']
    if (start.get('status')!='completed' or start.get('epoch')!=500 or start.get('members')!=4
            or start.get('sample_ids')!=source['sample_ids'] or start.get('panel_sha256')!=hash_file(ctx['panel_path'])
            or start.get('source_bank_sha256')!=hash_file(ctx['source_receipt'])
            or start.get('source_checkpoint_sha256')!=source['source_sha256']
            or panel.get('sensor_count')!=256 or panel.get('native_points')!=40300
            or len(panel.get('query_indices',[]))!=4096):
        raise ValueError('cached epoch500 and SOURCE panel identities differ')
    return ctx,start_bank,start


def check_cached_rows(ctx,start_bank,start,indices):
    import torch
    for i in indices:
        source_file=ctx['source_dir']/f'{i:04d}.pt';candidate_file=start_bank.parent/f'{i:04d}.pt'
        if (hash_file(source_file)!=ctx['source']['files_sha256'][i] or hash_file(candidate_file)!=start['files_sha256'][i]):
            raise ValueError(f'cached source or start row changed: {i}')
        source=torch.load(source_file,map_location='cpu',weights_only=True);candidate=torch.load(candidate_file,map_location='cpu',weights_only=True)
        if source['sample_id']!=ctx['source']['sample_ids'][i] or len(source['member_seeds'])!=4 or source['obs_indices'].shape[1]!=256:
            raise ValueError(f'cached member/observation manifest differs: {i}')
        matched_rows(source,candidate)


def context_check(campaign,output):
    start=time.time();cpu=time.process_time();ctx,bank,receipt=base_context(campaign)
    check_cached_rows(ctx,bank,receipt,(0,99))
    report={'schema':'car_scale_validation_context_check_v1','status':'completed','source_bank':reference(ctx['source_receipt']),
        'start_bank':reference(bank),'panel':reference(ctx['panel_path']),'sample_count':100,'members':4,'native_points':40300,
        'observations_per_condition':256,'selected_support_points':4096,'row_identity_checks':[0,99],
        'scope':'static context and two boundary cached rows; no scoring or inference',
        'wall_seconds':time.time()-start,'cpu_seconds':time.process_time()-cpu,'test_locked':True}
    write_new(output,report);return report


def fork_spec(campaign,run,phase,variant,bank_output,output):
    import torch
    ctx,start_bank,start=base_context(campaign);run=run.resolve(strict=True);output=output.resolve()
    if campaign/'configs' not in output.parents or campaign/'runs' not in run.parents:raise ValueError('campaign-owned config and fork run required')
    if phase not in (100,500):raise ValueError('phase epoch must be 100 or 500')
    if variant not in ('FFM_SSS_Unchanged_Scale_Continuation','FFM_SSS_RotatingComplement_FieldFeedback') or run.parent.name!=variant:
        raise ValueError('named fork variant and run path differ')
    bank_output=bank_output.resolve()
    if campaign/'assessment' not in bank_output.parents or bank_output.exists():raise ValueError('new declared bank output required')
    snap=(campaign.parent/'round_6/assessment/Epoch500_Validation100_J4_20261125T042223Z/snapshot/snapshot.json').resolve(strict=True)
    checkpoint=run/'checkpoints'/f'epoch_{phase:03d}.pt';cfg=run/'resolved_config.yaml';manifest_file=run/'run_manifest.json'
    manifest=json.loads(manifest_file.read_text());state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    spec={'schema':'car_scale_fork_validation_spec_v1','test_locked':True,'variant':variant,'run':str(run),'phase_epoch':phase,
        'phase_origin_step':6000,'global_step':6000+12*phase,'run_config':reference(cfg),'checkpoint':reference(checkpoint),
        'run_manifest':reference(manifest_file),'parent_snapshot':reference(snap),'source_bank':reference(ctx['source_receipt']),
        'panel':reference(ctx['panel_path']),'execution_code_root':str(ROOT),'execution_inventory':reference(ROOT/'execution_receipt.json'),
        'bank_output':str(bank_output),'start_bank':reference(start_bank)}
    verify_execution(spec)
    validate_clock(spec,manifest,state,run,cfg.resolve(),checkpoint.resolve(),json.loads(snap.read_text()))
    if state.get('source_hashes',{}).get('checkpoint')!=ctx['source']['source_sha256']:raise ValueError('fork SOURCE checkpoint differs')
    check_cached_rows(ctx,start_bank,start,range(100))
    snapshot_path=output.with_name(output.stem+'_run_manifest.json')
    if output.exists() or snapshot_path.exists():raise FileExistsError('spec or manifest snapshot already exists')
    # Copy a verified completed manifest; later training cannot mutate this binding.
    content=manifest_file.read_bytes()
    if hash_file(manifest_file)!=spec['run_manifest']['sha256']:raise ValueError('run manifest changed during preparation')
    with snapshot_path.open('xb') as f:f.write(content)
    spec['run_manifest']=reference(snapshot_path)
    write_new(output,spec);return spec


def scale_spec(campaign,children,output):
    base=campaign/'configs/Validation100_ScaleDecomposition_20261125_v2.json'
    spec=json.loads(base.read_text());ctx,start_bank,start=base_context(campaign)
    if Path(spec['campaign']).resolve()!=campaign or spec['source_receipt']!=reference(ctx['source_receipt']) or spec['panel']!=reference(ctx['panel_path']):
        raise ValueError('original Scale spec SOURCE or panel differs')
    # Keep only original epoch500 common start; add one or more descriptive fork children.
    spec['children']={'original_epoch500':spec['children']['epoch500']}
    for name,bank_path,assessment_path in children:
        if not name.startswith('FFM_SSS_') or name in spec['children']:raise ValueError('unique descriptive fork child name required')
        bank=Path(bank_path).resolve(strict=True);assessment=Path(assessment_path).resolve(strict=True)
        b=json.loads(bank.read_text());a=json.loads(assessment.read_text())
        identity=assessment.parent/'fork_identity.json';i=json.loads(identity.read_text())
        if (b.get('status')!='completed' or b.get('sample_ids')!=ctx['source']['sample_ids'] or b.get('members')!=4
                or b.get('source_bank_sha256')!=hash_file(ctx['source_receipt']) or b.get('panel_sha256')!=hash_file(ctx['panel_path'])
                or a.get('bank_receipt_sha256')!=hash_file(bank) or a.get('checkpoint_sha256')!=b['checkpoint_sha256']
                or a.get('normalizers',{}).get('scale')!=spec['frozen_train_scale_normalizer']
                or checked(i['assessment'])!=assessment or checked(i['bank_receipt'])!=bank
                or not checked(i['spec']).is_file() or b.get('fork_spec')!=i['spec']
                or b.get('phase_origin_step')!=i.get('phase_origin_step') or b.get('global_step')!=i.get('global_step')
                or b.get('epoch')!=i.get('phase_epoch')
                or name not in (i.get('variant'),f"{i.get('variant')}_Phase{i.get('phase_epoch')}")
                or i.get('phase_origin_step')!=6000 or i.get('phase_epoch') not in (100,500)
                or i.get('global_step')!=6000+12*i['phase_epoch']):
            raise ValueError(f'fork cached assessment identity differs: {name}')
        spec['children'][name]={'receipt':reference(bank),'assessment':reference(assessment)}
    if len(spec['children'])<2:raise ValueError('at least one completed fork child required')
    for key in ('source_receipt','panel','expanded_bank','prefit','formal_freeze'):checked(spec[key])
    if campaign/'configs' not in output.resolve().parents:raise ValueError('new campaign config required')
    write_new(output,spec);return spec


def scale_compare(campaign,path,unit,pairs,output):
    doc=json.loads(path.read_text());records=doc['records'];ids=doc['sample_ids']
    if doc.get('status')!='completed' or doc.get('split')!='validation' or doc.get('members')!=4 or len(ids)!=100 or [r['sample_id'] for r in records]!=ids or not math.isfinite(unit) or unit<=0:
        raise ValueError('completed matched VALIDATION100 J4 decomposition and positive unit required')
    decomposition_spec=json.loads(checked(doc['spec']).read_text())
    if unit!=decomposition_spec.get('frozen_train_scale_normalizer'):
        raise ValueError('comparison unit differs from bound frozen TRAIN Scale unit')
    comparisons={}
    for left,right in pairs:
        if left==right:raise ValueError('paired comparison needs different models')
        values={}
        for key in DEFINITIONS:
            a=[r['models'][left]['definitions'][key] for r in records];b=[r['models'][right]['definitions'][key] for r in records]
            values[key]=legacy.intervals(a,b,scale=unit)
        comparisons[f'{right}_minus_{left}']=values
    if not comparisons:raise ValueError('at least one explicit pair required')
    result={'schema':'car_scale_aligned_fork_comparisons_v1','status':'completed','decomposition':reference(path),'scale_unit':unit,
        'sample_count':100,'comparisons':comparisons,'interpretation':'paired condition bootstrap on one reused development trajectory; not independent-trajectory confirmation',
        'test_locked':True}
    if campaign/'assessment' not in output.resolve().parents:raise ValueError('campaign assessment output required')
    write_new(output,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);sub=p.add_subparsers(dest='command',required=True)
    x=sub.add_parser('context-check');x.add_argument('--output',type=Path,required=True)
    x=sub.add_parser('fork-spec');x.add_argument('--run',type=Path,required=True);x.add_argument('--phase-epoch',type=int,choices=(100,500),required=True);x.add_argument('--variant',required=True);x.add_argument('--bank-output',type=Path,required=True);x.add_argument('--output',type=Path,required=True)
    x=sub.add_parser('scale-spec');x.add_argument('--child',nargs=3,action='append',required=True,metavar=('VARIANT','BANK_RECEIPT','ASSESSMENT'));x.add_argument('--output',type=Path,required=True)
    x=sub.add_parser('scale-compare');x.add_argument('--decomposition',type=Path,required=True);x.add_argument('--scale-unit',type=float,required=True);x.add_argument('--pair',nargs=2,action='append',required=True,metavar=('LEFT','RIGHT'));x.add_argument('--output',type=Path,required=True)
    a=p.parse_args();campaign=a.campaign.resolve(strict=True)
    if a.command=='context-check':result=context_check(campaign,a.output)
    elif a.command=='fork-spec':result=fork_spec(campaign,a.run,a.phase_epoch,a.variant,a.bank_output,a.output)
    elif a.command=='scale-spec':result=scale_spec(campaign,a.child,a.output)
    else:result=scale_compare(campaign,a.decomposition.resolve(strict=True),a.scale_unit,a.pair,a.output)
    print(json.dumps({'status':'completed','output':reference(a.output)},indent=2))

if __name__=='__main__':main()
