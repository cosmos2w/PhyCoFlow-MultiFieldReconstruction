#!/usr/bin/env python3
"""Assemble completed matched-100 evidence without inference or selecting an arm."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
import numpy as np
import yaml
from scripts.analysis.car_round7_paired_summary import paired_summary, summarize_scale
from scripts.analysis.report_car_round7 import check_snapshot

CONTROL = 'FFM_Round7_Composite_Scale_Control'
JOINT = 'FFM_Round7_Joint_Anchor_Scale'


def ref(path):
    path = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return {'path': str(path), 'sha256': h.hexdigest()}


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def training(run):
    files = {'history':run/'logs/history.jsonl', 'journal':run/'logs/epoch_calibrated_pi_updates.jsonl',
             'coupling':run/'metrics/training_coupling.jsonl', 'query':run/'logs/scalable_updates.jsonl',
             'definitions':run/'logs/recording_definitions.json', 'status':run/'status.json',
             'config':run/'resolved_config.yaml', 'checkpoint':run/'checkpoints/epoch_100.pt'}
    identity = {key:ref(path) for key,path in files.items()}
    hist, journal, coupling, query = [rows(files[key]) for key in ('history','journal','coupling','query')]
    status=load(files['status'])
    config=yaml.safe_load(files['config'].read_text())
    reference=config['optimization']['multitask']['fidelity_budget']['reference_artifact']
    if ref(reference['path'])['sha256'] != reference['sha256']:
        raise ValueError('saved field reference hash differs')
    fields=load(reference['path'])['reference']['field_names']
    if not fields or len(set(fields)) != len(fields):
        raise ValueError('saved field reference order must be nonempty and unique')
    identity['field_reference']=reference
    identity['field_order']=fields
    if status.get('global_step') != 1200 or len(hist) != 100 or [r['epoch'] for r in hist] != list(range(1,101)) or any(r.get('epoch_complete') is not True for r in hist):
        raise ValueError('terminal epoch-100 status and complete 100-epoch history required')
    for name, records in (('journal',journal),('coupling',coupling),('query',query)):
        if len(records) != 1200 or [r['step'] for r in records] != list(range(1,1201)):
            raise ValueError(f'{name} requires exactly 1200 ordered actions')
    if any(r.get('accepted') is not True or len(r.get('sample_ids', [])) != 64 for r in coupling):
        raise ValueError('each action must accept the full B64 objective-bearing conditions')
    if any(r.get('update_accepted') is not True for r in query):
        raise ValueError('all query actions must be accepted')
    return identity,hist,journal,coupling,query


def capture_training_evidence(identity, directory):
    """Retain terminal-100 mutable records before an authorized continuation."""
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=False)
    result=copy.deepcopy(identity)
    for key in ('history','journal','coupling','query','status'):
        original=identity[key]
        source=Path(original['path']).resolve(strict=True)
        if ref(source)['sha256']!=original['sha256']:
            raise ValueError(f'terminal training evidence changed before snapshot: {key}')
        destination=directory/source.name
        with source.open('rb') as reader, destination.open('xb') as writer:
            shutil.copyfileobj(reader,writer,1024*1024)
        saved=ref(destination)
        if saved['sha256']!=original['sha256'] or ref(source)['sha256']!=original['sha256']:
            raise ValueError(f'terminal training evidence changed during snapshot: {key}')
        result[key]={**saved,'original':original}
    with (directory/'snapshot_receipt.json').open('x') as stream:
        json.dump({'schema':'car_round7_review100_training_snapshot_v1','status':'completed',
                   'files':{key:result[key] for key in ('history','journal','coupling','query','status')}},stream,indent=2,allow_nan=False)
        stream.write('\n')
    result['snapshot_receipt']=ref(directory/'snapshot_receipt.json')
    return result


def streams(control, joint):
    cc,cq=control[3],control[4]; jc,jq=joint[3],joint[4]
    if any(len(value)!=1200 for value in (cc,cq,jc,jq)):
        raise ValueError('both arms require 1200 coupled and query actions')
    keys=('sample_ids','observation_sha256','query_sha256','member_seeds',
          'first_member_generator_state_sha256','field_complement_sha256','field_complement_seed')
    for left,right in zip(cc,jc,strict=True):
        if any(key not in left or left[key] != right.get(key) for key in keys):
            raise ValueError(f'sample/observation/member/support/complement stream mismatch at action {left["step"]}')
    for left,right in zip(cq,jq,strict=True):
        if (not left.get('exposure') or left['exposure'] != right.get('exposure')
                or {entry.get('condition_group') for entry in left['exposure']} != set(range(8))):
            raise ValueError(f'query exposure mismatch at action {left["step"]}')
    return {'matched_actions':1200,'conditions_per_action':64,'objective_condition_visits':76800,
            'dataset_pass_equivalents':9.6,'matched_keys':list(keys)+['query_exposure'],
            'sampling_definition':'equal ordered condition IDs, padded observation hashes, support, independent member seeds and first-generator states, complement and grouped query exposures'}


def matched_configs(control_path,joint_path):
    control,joint=[yaml.safe_load(Path(path).read_text()) for path in (control_path,joint_path)]
    for cfg, objective in ((control,'composite'),(joint,'joint_anchor')):
        scalable=cfg['coherence']['scalable_anchored_coherence']
        if (scalable.get('scale_objective')!=objective or cfg['optimization']['epochs']!=500 or
            cfg['optimization']['steps_per_epoch']!=12 or cfg['optimization']['batch_size']!=64):
            raise ValueError('arm objective or 500-epoch/B64 scientific schedule differs')
    left,right=copy.deepcopy(control),copy.deepcopy(joint)
    for cfg in (left,right):
        cfg['output'].pop('experiment_name',None)
        cfg['runtime'].pop('device',None)
        cfg['coherence']['scalable_anchored_coherence'].pop('scale_objective',None)
        cfg['coherence']['scalable_anchored_coherence'].pop('scale_joint_kappa',None)
    if left!=right:
        raise ValueError('arm scientific configurations differ beyond declared Scale objective and kappa')
    return {'common_scientific_configuration':True,'candidate_kappa':joint['coherence']['scalable_anchored_coherence']['scale_joint_kappa'],
            'reporting_horizon_epochs':500,'control_objective':'composite','candidate_objective':'joint_anchor'}


def window(hist,journal,start,end,fields):
    values=hist[start-1:end]
    actions=journal[(start-1)*12:end*12]
    prefixes=('coherence_component/multiscale_spectrum/', 'coherence_family/', 'round7_scale_kappa',
              'round7_active_field_row_fraction','optimizer_lr')
    keys=sorted({key for row in values for key in row if key.startswith(prefixes)})
    means={}
    for key in keys:
        vector=[row.get(key) for row in values]
        if all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in vector):
            means[key]=sum(vector)/len(vector)
    def vectors(key):
        values=[]
        for action in actions:
            value=action[key]
            if key == 'field_violation':
                if not isinstance(value,dict) or set(value)!=set(fields):
                    raise ValueError('field violation keys differ from saved reference order')
                value=[value[field] for field in fields]
            elif not isinstance(value,list) or len(value)!=len(fields):
                raise ValueError(f'{key} must follow the saved reference field order')
            values.append(value)
        raw=np.asarray(values,dtype=float)
        if not np.isfinite(raw).all(): raise ValueError(f'nonfinite controller journal: {key}')
        return {'mean':dict(zip(fields,raw.mean(axis=0).tolist(),strict=True)),
                'last':dict(zip(fields,raw[-1].tolist(),strict=True))}
    mult=np.asarray([a['multipliers_before'] for a in actions],dtype=float)
    return {'epochs':[start,end], 'field_order':fields, 'history_means':means,
            'field_violation':vectors('field_violation'),
            'multipliers_before':vectors('multipliers_before'),
            'multipliers_after':vectors('multipliers_after'), 'smoothed_violation_z':vectors('z_after'),
            'active_field_row_fraction':float(np.any(mult!=0,axis=1).mean())}


def snapshot(directory,arm,checkpoint):
    directory=Path(directory).resolve(strict=True)
    assessment=directory/'assessment.json'; scale=directory/'scale_summary.json'; receipt=directory/'receipt.json'
    job=load(receipt)
    if job.get('status')!='completed' or job.get('schema')!='car_round7_score_job_v1':
        raise ValueError('completed scoring receipt required')
    data=load(assessment)
    entry={'assessment':ref(assessment),'scale_summary':ref(scale),'split':data['split'],
           'epoch':100,'arm':arm,'checkpoint_sha256':checkpoint['sha256']}
    data, summary=check_snapshot(entry)
    decomp=directory/'decomposition.json'
    doc=load(decomp)
    if doc.get('status')!='completed' or doc['sample_ids']!=data['sample_ids'] or doc.get('split')!=data['split']:
        raise ValueError('decomposition must match assessment panel')
    if doc['child_receipt']!=data['child_selected'] or doc['source_receipt']!=data['source_selected']:
        raise ValueError('decomposition bank identity differs from assessment')
    return {'assessment':data,'scale':summary,'decomposition':doc,
            'evidence':{**entry,'score_receipt':ref(receipt),'decomposition':ref(decomp)}}


def matched_assessment(control,joint):
    a,b=control['assessment'],joint['assessment']
    if (a['sample_ids']!=b['sample_ids'] or a['field_order']!=b['field_order'] or
        a['field_units']!=b['field_units'] or a['physical_mask']!=b['physical_mask'] or
        a['source_baseline']!=b['source_baseline'] or a['source_native']!=b['source_native'] or
        a['source_selected']!=b['source_selected'] or a['native_allowance_physical']!=b['native_allowance_physical'] or
        a['per_condition_fields']['SOURCE']!=b['per_condition_fields']['SOURCE'] or
        a['per_condition_scores']['SOURCE']!=b['per_condition_scores']['SOURCE']):
        raise ValueError('paired assessments do not share exact panel, units and SOURCE baseline')
    ca,ja=control['decomposition'],joint['decomposition']
    if ca['prefit_coefficients']!=ja['prefit_coefficients'] or control['scale']['source_unit']!=joint['scale']['source_unit']:
        raise ValueError('Scale definitions or SOURCE units differ')
    contract=load(a['spec']['path'])
    contract=load(contract['assessment_contract']['path'])
    indices=np.random.default_rng(6042026).integers(0,len(a['sample_ids']),size=(2000,len(a['sample_ids'])))
    direct={}
    for key in ('state','scale','shape'):
        direct[key]=paired_summary([r[key] for r in a['per_condition_scores']['candidate']],
            [r[key] for r in b['per_condition_scores']['candidate']], indices, contract['normalizers'][key])
    shape_direct={key:paired_summary([r[key] for r in a['per_condition_scores']['candidate']],
        [r[key] for r in b['per_condition_scores']['candidate']],indices,contract['normalizers']['shape'])
        for key in a['paired_shape_components']}
    fields={key:paired_summary([r[key] for r in a['per_condition_fields']['candidate']],
              [r[key] for r in b['per_condition_fields']['candidate']],indices)
            for key in ('member_mse','ensemble_mean_mse','signed_mean_risk','unbiased_member_variance','sensor_mse')}
    control_rows=ca['records']; joint_rows=ja['records']
    combined=[]
    for left,right in zip(control_rows,joint_rows,strict=True):
        if left['sample_id']!=right['sample_id'] or left['member_seeds']!=right['member_seeds'] or left['models']['SOURCE']!=right['models']['SOURCE']:
            raise ValueError('Scale decomposition SOURCE, ID or member pairing differs')
        combined.append({'sample_id':left['sample_id'], 'models':{'SOURCE':left['models']['candidate'],
                                                                   'joint':right['models']['candidate']}})
    direct_scale=summarize_scale({'records':combined,'sample_ids':a['sample_ids'],'members':4,
        'split':a['split'],'prefit_coefficients':ca['prefit_coefficients']},'joint',control['scale']['source_unit'])
    return {'split':a['split'],'sample_count':len(a['sample_ids']),'field_order':a['field_order'],
        'field_units':a['field_units'],'native_allowance_physical':a['native_allowance_physical'],
        'selection_use':a['selection_use'], 'control_vs_source':{'paired_fields':a['paired_fields'],
        'paired_core':a['paired_core'],'paired_shape_components':a['paired_shape_components'],
        'native_safeguards':a['native_safeguards'],'scale':control['scale']},
        'joint_vs_source':{'paired_fields':b['paired_fields'],'paired_core':b['paired_core'],
        'paired_shape_components':b['paired_shape_components'],'native_safeguards':b['native_safeguards'],
        'scale':joint['scale']},'joint_minus_control':{'paired_core':direct,'paired_fields_physical':fields,
        'paired_shape_components':shape_direct,'scale':direct_scale},
        'evidence':{'control':control['evidence'],'joint':joint['evidence']}}


def fmt(x):
    return f'{float(x):+.4g}'


def markdown(doc):
    lines=['# Round 7 matched-100 decision evidence','','Both arms completed 100 reporting epochs on matched streams. VALIDATION is reused for development and selection; unoptimized tasks remain withheld. The figures below are paired effects; no continuation or adoption decision is automated.','',
           '| Split | Comparison | State | Complete expanded Scale | Raw joint Scale | Shape |','| --- | --- | ---: | ---: | ---: | ---: |']
    for split,data in doc['assessment'].items():
        for key,label in (('control_vs_source','Composite − SOURCE'),('joint_vs_source','Joint anchor − SOURCE'),('joint_minus_control','Joint anchor − composite')):
            result=data[key]; scale=result['scale']; core=result['paired_core']
            lines.append(f"| {split} | {label} | {fmt(core['state']['mean'])} | {fmt(scale['definitions']['Complete_Expanded_Prefit']['mean'])} | {fmt(scale['raw_joint']['score']['mean'])} | {fmt(core['shape']['mean'])} |")
    lines.extend(['','Values use the corresponding frozen TRAIN units. Full paired condition-bootstrap intervals, all five Scale attraction/separation/score blocks, native field and sensor effects, controller windows, definitions and hash-bound evidence are in the JSON. A/R terms use positive separation and score = attraction − separation.','',
        '| Arm | Epoch window | Mean raw sampled joint Scale | Mean weighted sampled joint contribution | Active field-row fraction |', '| --- | --- | ---: | ---: | ---: |'])
    for arm,data in doc['training'].items():
        for window_name,window_data in data['late_windows'].items():
            values=window_data['history_means'];prefix='coherence_component/multiscale_spectrum/joint/'
            raw=values.get(prefix+'raw'); weighted=values.get(prefix+'weighted_contribution')
            lines.append(f"| {arm} | {window_name} | {fmt(raw) if raw is not None else 'unavailable'} | {fmt(weighted) if weighted is not None else 'unavailable'} | {window_data['active_field_row_fraction']:.3f} |")
    lines.extend(['','Sampled training scores describe optimization exposure and do not replace complete TRAIN800 or held-out VALIDATION100 assessments. No task results are used.','', '[Full evidence and intervals](readout.json)',''])
    return '\n'.join(lines)


def build(args):
    left=training(args.control_run); right=training(args.joint_run)
    left=(capture_training_evidence(left[0],args.output/'training_evidence'/CONTROL),*left[1:])
    right=(capture_training_evidence(right[0],args.output/'training_evidence'/JOINT),*right[1:])
    if left[0]['field_order'] != right[0]['field_order'] or left[0]['field_reference'] != right[0]['field_reference']:
        raise ValueError('both arms must use the same saved field reference and order')
    match=streams(left,right)
    train={}
    for name,observed in ((CONTROL,left),(JOINT,right)):
        train[name]={'evidence':observed[0], 'late_windows':{
            '51–75':window(observed[1],observed[2],51,75,observed[0]['field_order']),
            '76–100':window(observed[1],observed[2],76,100,observed[0]['field_order'])}}
    assessments={}
    for split,ca,ja in (('train',args.control_train,args.joint_train),('validation',args.control_validation,args.joint_validation)):
        control=snapshot(ca,CONTROL,left[0]['checkpoint']);joint=snapshot(ja,JOINT,right[0]['checkpoint'])
        if control['assessment']['split']!=split or joint['assessment']['split']!=split:
            raise ValueError('assessment split does not match input')
        assessments[split]=matched_assessment(control,joint)
    result={'schema':'car_round7_review100_readout_v1','status':'completed',
        'training_streams':match,'config_comparison':matched_configs(left[0]['config']['path'],right[0]['config']['path']),
        'training':train,'assessment':assessments,
        'decision':'not selected; evidence for an explicit continuation or closure decision',
        'tasks':'withheld until decision is sealed','test_locked':True}
    with (args.output/'readout.json').open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    with (args.output/'readout.md').open('x') as stream: stream.write(markdown(result))
    print(args.output/'readout.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('control-run','joint-run','control-train','control-validation','joint-train','joint-validation','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    args.output=args.output.resolve()
    data_root=Path('/data2/wanglz/PhyCoFlow-MultiFieldReconstruction').resolve(strict=True)
    if data_root not in args.output.parents or 'assessment' not in args.output.relative_to(data_root).parts:
        raise ValueError('new output must be inside a DATA2 campaign assessment directory')
    args.output.mkdir(parents=True,exist_ok=False)
    receipt={'schema':'car_round7_review100_readout_attempt_v1','status':'failed'}
    try:
        build(args)
        receipt.update(status='completed',readout=ref(args.output/'readout.json'))
    except BaseException as error:
        receipt.update(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        with (args.output/'attempt_receipt.json').open('x') as stream:
            json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')

if __name__=='__main__': main()
