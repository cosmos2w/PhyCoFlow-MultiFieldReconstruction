#!/usr/bin/env python3
"""Decide a bounded Round 5 startup from real saved replay and trace evidence.

This CPU-only analyzer does not run a model. It requires full TRAIN800 replay,
complete matched candidate/native traces, an old-route timing, a measured
scalar-precision receipt, a fresh-process resume/rollback receipt, and the
existing Round 3 assessment timing receipts. It reports one evidence-directed
correction when indicated, or a startup qualification for the formal launcher.
See --help for inputs; every input and the consulted run journals are hashed.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from phycoflow_reconstruction.evaluation.adaptive_campaign import digest, immutable_json


def ref(path):
    path=Path(path).resolve(strict=True)
    return {'path':str(path),'sha256':digest(path)}


def read(path): return json.loads(Path(path).read_text())
def jsonl(path): return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def positive(value,name,zero=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or (value<0 if zero else value<=0):
        raise ValueError(f'{name} must be finite and '+('nonnegative' if zero else 'positive'))
    return float(value)


def analyze_replay(rows, ids, lrs):
    results={}; overshoot=False; all_same=[]; all_independent=[]; all_tolerances=[]
    for lr in lrs:
        chosen=[row for row in rows if row.get('method')=='Direct_Scalar_AdamW' and math.isclose(float(row.get('lr',-1)),lr,rel_tol=1e-10,abs_tol=0)]
        seen=[sid for row in chosen for sid in row['sample_ids']]
        if seen!=ids or any(row.get('sample_count')!=len(row['sample_ids']) for row in chosen):
            raise ValueError(f'full matched TRAIN800 replay at LR {lr} required')
        actions=[row for row in chosen if row.get('update_accepted') is True and positive(row['committed_displacement_norm'],'committed displacement',True)>0]
        if not actions:
            results[str(lr)]={'accepted_nonzero':0,'descent_fraction':None,'median_decrease':None,'independent_median_decrease':None}
            continue
        same=[float(row['before']['direct_scalar'])-float(row['same_query_after']['direct_scalar']) for row in actions]
        independent=[float(row['independent_query_before']['direct_scalar'])-float(row['independent_query_after']['direct_scalar']) for row in actions]
        tolerances=[positive(row['scalar_descent_tolerance'],'measured scalar tolerance',True) for row in actions]
        if not all(math.isfinite(x) for x in same+independent):raise ValueError('nonfinite replay objective')
        for row,tolerance in zip(actions,tolerances,strict=True):
            precision=positive(row['scalar_replay_precision'],'scalar replay precision',True)
            if tolerance<5*precision or not math.isfinite(float(row['scalar_replay_before'])):
                raise ValueError('scalar descent tolerance does not cover measured replay precision')
        fraction=sum(x>tol for x,tol in zip(same,tolerances,strict=True))/len(same)
        overshoot |= any(float(r['scalar_displacement_dot']) < 0 and decrease < -tol for r,decrease,tol in zip(actions,same,tolerances,strict=True))
        results[str(lr)]={'accepted_nonzero':len(actions),'descent_fraction':fraction,'median_decrease':statistics.median(same),
            'independent_median_decrease':statistics.median(independent),
            'independent_descent_fraction':sum(x>tol for x,tol in zip(independent,tolerances,strict=True))/len(independent)}
        all_same.extend(same);all_independent.extend(independent);all_tolerances.extend(tolerances)
    if not all_same:
        aggregate={'accepted_nonzero':0,'descent_fraction':0.,'median_decrease':None,'tolerance':None,'passed':False}
    else:
        fraction=sum(x>tol for x,tol in zip(all_same,all_tolerances,strict=True))/len(all_same)
        median=statistics.median(all_same); tolerance=max(all_tolerances)
        aggregate={'accepted_nonzero':len(all_same),'descent_fraction':fraction,'median_decrease':median,
            'tolerance':tolerance,'independent_median_decrease':statistics.median(all_independent),
            'passed':fraction>=.8 and median>tolerance}
    return results,aggregate,overshoot


def trace(receipt_path, lease_path, role):
    import yaml
    receipt=read(receipt_path);lease=read(lease_path)
    if receipt.get('schema')!='car_round5_training_trace_v1' or receipt.get('status')!='completed':
        raise ValueError(f'{role}: completed production trace receipt required')
    if lease.get('schema')!='car_round5_lease_v1' or lease.get('status')!='completed':
        raise ValueError(f'{role}: completed measured GPU lease required')
    if not receipt.get('lease_token') or receipt['lease_token'] != lease.get('token'):
        raise ValueError(f'{role}: trace is not bound to the supplied lease')
    if not (float(lease['start_unix']) <= float(receipt['started_unix']) <= float(lease['end_unix'])):
        raise ValueError(f'{role}: trace start is outside its measured lease')
    run=Path(receipt['run_dir']).resolve(strict=True)
    config=yaml.safe_load(Path(receipt['config']).read_text())
    mode=config['optimization'].get('direct_supervision',{}).get('mode')
    expected={'candidate':'safeguarded','native':'native','historical':None}[role]
    if mode!=expected or bool(receipt.get('historical_timing'))!=(role=='historical'):
        raise ValueError(f'{role}: trace mode mismatch')
    epochs=receipt['reporting_epochs'];start=receipt['start_step'];target=receipt['target_step']
    if epochs not in (1,2) or target != epochs*12 or (role=='historical' and epochs!=1):
        raise ValueError(f'{role}: bounded complete epoch trace required')
    timings=jsonl(run/'logs/epoch_timing.jsonl')
    if [r['epoch'] for r in timings]!=list(range(1,epochs+1)) or any(r['step']!=r['epoch']*12 for r in timings):
        raise ValueError(f'{role}: incomplete epoch timings')
    wall=[positive(r['wall_seconds'],f'{role} complete epoch wall time') for r in timings]
    updates=[] if role=='historical' else jsonl(run/'logs/direct_updates.jsonl')
    if role!='historical' and ([r['step'] for r in updates]!=list(range(1,target+1)) or len(updates)!=target):
        raise ValueError(f'{role}: missing or duplicate attempt records')
    if role!='historical':
        for row in updates:
            positive(row.get('native_data_loss'),f'{role} native loss',True)
            positive(row.get('optimizer_lr'),f'{role} optimizer LR')
            if role=='candidate' and not math.isfinite(float(row['direct_scalar'])):
                raise ValueError('nonfinite direct objective')
    return {'receipt':receipt,'lease':lease,'run':run,'config':config,'updates':updates,
            'epoch_seconds':statistics.mean(wall),'timings':timings,'start_step':start}


def coupling(candidate,native):
    if len(candidate)!=len(native):raise ValueError('candidate and native traces must cover the same attempts')
    keys=('sample_ids','observation_sha256','query_sha256','native_seed','native_generator_state_after_sha256')
    for i,(a,b) in enumerate(zip(candidate,native,strict=True),1):
        ca,cb=a.get('native_coupling'),b.get('native_coupling')
        if not isinstance(ca,dict) or not isinstance(cb,dict) or any(k not in ca or ca[k]!=cb.get(k) for k in keys):
            raise ValueError(f'native coupled inputs differ at attempt {i}')
    return True


def candidate_actions(rows,dot_tolerance):
    nonzero=[];accepted=0;rejected=0;alpha_counts={'full':0,'half':0,'rejected':0}
    for r in rows:
        action=positive(r.get('action_norm'),'candidate action norm',True)
        dot=float(r['action_dot'])
        if not math.isfinite(dot):raise ValueError('nonfinite actual proposal dot')
        if action>0:nonzero.append(dot)
        ok=r.get('update_accepted')
        if not isinstance(ok,bool):raise ValueError('candidate acceptance missing')
        alpha=r.get('action_alpha')
        if ok and alpha==1.:accepted+=1;alpha_counts['full']+=1
        elif ok and alpha==.5:accepted+=1;alpha_counts['half']+=1
        elif not ok and alpha==0.:rejected+=1;alpha_counts['rejected']+=1
        else:raise ValueError('candidate acceptance and displacement fraction disagree')
        if r.get('field_source_ensemble_checks')!=1 or r.get('field_post_ensemble_checks') not in (0,1,2):
            raise ValueError('paired field check counts missing or invalid')
    return {'attempts':len(rows),'acceptance_fraction':accepted/len(rows),
            'ascending_proposal_fraction':sum(x>dot_tolerance for x in nonzero)/len(nonzero) if nonzero else 0.,
            'nonzero_proposals':len(nonzero),'action_counts':alpha_counts}


def validate_replay_optimizer(action, optimization, preparation):
    """A corrected trace must use actions replayed with its corrected AdamW."""
    lr_start = float(optimization['lr_warmup']['start_lr'])
    lr_target = float(optimization['lr'])
    beta1, beta2 = map(float, optimization['adam_betas'])
    if preparation.get('kind') != 'correction':
        return
    optimizer = action.get('optimizer', {})
    evidence = action.get('correction_evidence')
    if (action.get('candidate_only') is not True
            or action.get('learning_rates') != [lr_start, lr_target]
            or optimizer.get('method') != 'AdamW'
            or optimizer.get('beta1') != beta1 or optimizer.get('beta2') != beta2
            or optimizer.get('weight_decay') != float(optimization.get('weight_decay', 0.))
            or optimizer.get('grad_clip') != optimization.get('grad_clip')
            or not isinstance(evidence, dict)
            or evidence.get('sha256') != preparation.get('correction_evidence_sha256')):
        raise ValueError('correction requires candidate-only action replay bound to the selected AdamW and correction evidence')


def assessment_forecast(round3):
    receipts=[]
    for epoch in (100,500):
        path=round3/'assessment'/f'epoch_{epoch}'/'scores/receipt.json'; obj=read(path)
        if obj.get('status')!='completed' or obj.get('epoch')!=epoch:raise ValueError('completed historical CPU scorer timings required')
        positive(obj['wall_seconds'],'historical scorer wall time');positive(obj['cpu_seconds'],'historical scorer CPU time')
        receipts.append((path,obj))
    generation=[]
    for path in (round3/'operations/completions').glob('*.json'):
        obj=read(path)
        if obj.get('schema')=='car_round3_assessment_launch_v1' and obj.get('status')=='completed' and obj.get('epoch') in (100,500):
            generation.append((path,obj))
    if len(generation)!=4 or {r['epoch'] for _,r in generation}!={100,500}:
        raise ValueError('four historical generation timing receipts required')
    for _,row in generation:positive(row['seconds'],'historical generation time')
    wall=sum(r['wall_seconds'] for _,r in receipts);cpu=sum(r['cpu_seconds'] for _,r in receipts)
    return {'worker_hours':2*cpu/3600,'wall_seconds':2*wall,
            'tail_seconds':2*max(r['wall_seconds'] for _,r in receipts),
            'basis':'two times observed Round 3 matched two-arm review and endpoint CPU scoring; tail is twice the slower scoring wall time; historical hardware and scheduler may differ',
            'evidence':ref(receipts[0][0]),'cpu_timing_receipts':[ref(p) for p,_ in receipts],
            'generation_receipts':[ref(p) for p,_ in generation],
            'historical_generation_gpu_seconds':sum(r['seconds'] for _,r in generation)}


def resume_check(path):
    value=read(path)
    required=('fresh_process','saved_config_reload','resume_and_append',
              'forced_rejection_then_acceptance','state_rng_sampler_and_recording_match')
    if (value.get('schema')!='car_round5_resume_qualification_v1' or value.get('status') not in ('completed','passed')
            or value.get('exit_code')!=0 or any(value.get(key) is not True for key in required)
            or not value.get('command') or not isinstance(value.get('test_file'),dict)):
        raise ValueError('completed fresh-process save/reload/resume/append and forced rejection qualification required')
    test=value['test_file']; actual=ref(test['path'])
    if actual['sha256']!=test.get('sha256'):raise ValueError('resume qualification test file changed')
    return value


def assess(args):
    units=read(args.units);action=read(args.action_replay);preparation=read(args.preparation)
    if (units.get('schema')!='car_round5_source_units_v1' or units.get('status')!='completed'
            or units.get('sample_count')!=800 or units.get('split')!='train' or units.get('members')!=2
            or any(units.get('floor_active',{}).values())):
        raise ValueError('completed SOURCE TRAIN800 J2 unit fit required')
    panel_path=args.units.parent/'panel.json';panel=read(panel_path)
    ids=panel['sample_ids']
    if len(ids)!=800 or len(set(ids))!=800 or digest(panel_path)!=units['panel_sha256']:
        raise ValueError('SOURCE TRAIN800 panel identity differs')
    if (action.get('schema')!='car_round5_action_replay_v1' or action.get('status')!='completed'
            or action.get('sample_count')!=800 or action.get('units_sha256')!=digest(args.units)):
        raise ValueError('complete unit-bound TRAIN800 action replay required')
    if (preparation.get('schema')!='car_round5_startup_preparation_v1'
            or preparation.get('units_sha256')!=digest(args.units)
            or preparation.get('action_replay_sha256')!=digest(args.action_replay)
            or preparation.get('kind') not in ('initial','correction')):
        raise ValueError('startup preparation must bind the SOURCE units and action replay')
    replay_path=args.action_replay.parent/'actions.jsonl';rows=jsonl(replay_path)
    candidate=trace(args.candidate_trace,args.candidate_lease,'candidate')
    prefix_path = getattr(args, 'candidate_prefix_trace', None)
    if candidate['start_step']:
        if prefix_path is None:
            raise ValueError('resumed candidate requires its epoch-one trace receipt')
        prefix = read(prefix_path)
        resumed = candidate['receipt']
        if (prefix.get('schema') != 'car_round5_training_trace_v1' or prefix.get('status') != 'completed'
                or prefix.get('reporting_epochs') != 1 or prefix.get('start_step') != 0
                or prefix.get('target_step') != candidate['start_step'] or candidate['start_step'] != 12
                or prefix.get('run_dir') != resumed.get('run_dir') or resumed.get('resume') != prefix.get('run_dir')
                or prefix.get('config') != resumed.get('config') or prefix.get('lease_token') != resumed.get('lease_token')
                or resumed.get('saved_config_reloaded') is not True
                or resumed.get('saved_config_sha256') != digest(candidate['run']/'resolved_config.yaml')
                or not (float(candidate['lease']['start_unix']) <= float(prefix.get('started_unix', 0)) <= float(candidate['lease']['end_unix']))):
            raise ValueError('candidate epoch-one trace must bind the same run, configuration and lease before resume')
    elif prefix_path is not None:
        raise ValueError('candidate prefix receipt requires a resumed candidate trace')
    native=trace(args.native_trace,args.native_lease,'native')
    historical=trace(args.historical_trace,args.historical_lease,'historical')
    if candidate['receipt']['reporting_epochs']!=2 or native['receipt']['reporting_epochs']!=2:
        raise ValueError('candidate and native need complete two-epoch startup traces')
    for role,info,key in [('candidate',candidate,'Native_GT_SSS_Safeguarded_Continuation'),
                          ('native',native,'Native_GT_Continuation'),
                          ('historical',historical,'Historical_SourceScaledRisk_Timing')]:
        expected=preparation['configs'][key]
        if digest(Path(info['receipt']['config']))!=expected['sha256']:
            raise ValueError(f'{role}: startup configuration differs from prepared binding')
    devices=[v['lease'].get('device',{}).get('uuid') for v in (candidate,native,historical)]
    if not devices[0] or len(set(devices))!=1:
        raise ValueError('matched old-route and direct timing require same runtime device')
    coupling(candidate['updates'],native['updates'])
    resume_check(args.resume_receipt)
    opt=candidate['config']['optimization'];nopt=native['config']['optimization']; direct=opt['direct_supervision']
    for key in ('lr','lr_warmup','adam_betas','batch_size','steps_per_epoch','sampling'):
        if opt.get(key)!=nopt.get(key):raise ValueError(f'candidate and native config differ at {key}')
    if direct['units']!=units['units'] or nopt['direct_supervision']['units']!=units['units']:
        raise ValueError('trace units do not match SOURCE fit')
    validate_replay_optimizer(action, opt, preparation)
    lr_start=float(opt['lr_warmup']['start_lr']);lr_target=float(opt['lr'])
    replay,aggregate,overshoot=analyze_replay(rows,ids,(lr_start,lr_target))
    actual=candidate_actions(candidate['updates'],float(direct.get('dot_tolerance',0)))
    if actual['nonzero_proposals']==0:raise ValueError('no candidate-owned nonzero proposal to qualify')
    ascent=actual['ascending_proposal_fraction']>.1
    must_evaluate_beta_zero=ascent and float(opt['adam_betas'][0])!=0
    candidate_pass=aggregate['passed'] and actual['acceptance_fraction']>=.5 and not must_evaluate_beta_zero
    corrections=[]
    if not candidate_pass and overshoot and (lr_start,lr_target)==(5e-6,5e-5):corrections.append('halve both LR endpoints')
    if not candidate_pass and ascent and float(opt['adam_betas'][0])!=0:corrections.append('use beta1=0 with unchanged beta2')
    terminal=not candidate_pass and preparation['kind']=='correction'
    reason=[]
    if not aggregate['passed']:reason.append('fixed-query replay descent threshold failed')
    if actual['acceptance_fraction']<.5:reason.append('startup acceptance below 50%')
    if must_evaluate_beta_zero:reason.append('more than 10% of nonzero proposals ascend')
    if not candidate_pass and not corrections and not terminal:reason.append('diagnose the demonstrated failure within remaining startup budget')
    tolerance=[float(v) for v in action.get('field_tolerance_max',[])]
    if len(tolerance)!=5 or any(not math.isfinite(v) or v<0 for v in tolerance):raise ValueError('five field replay tolerances required')
    if list(direct.get('field_tolerance', [])) != tolerance:
        raise ValueError('candidate trace must use the measured frozen field tolerances')
    forecast=assessment_forecast(args.round3)
    evidence={name:ref(path) for name,path in {
        'units':args.units,'unit_panel':panel_path,'action_replay':args.action_replay,'replay_actions':replay_path,
        'startup_preparation':args.preparation,'candidate_trace':args.candidate_trace,'native_trace':args.native_trace,
        'historical_trace':args.historical_trace,'candidate_lease':args.candidate_lease,'native_lease':args.native_lease,
        'historical_lease':args.historical_lease,'resume':args.resume_receipt}.items()}
    if prefix_path is not None:
        evidence['candidate_prefix_trace'] = ref(prefix_path)
    for role,info in [('candidate',candidate),('native',native),('historical',historical)]:
        evidence[role+'_timing']=ref(info['run']/'logs/epoch_timing.jsonl')
        if role!='historical':evidence[role+'_updates']=ref(info['run']/'logs/direct_updates.jsonl')
    result={'schema':'car_round5_startup_qualification_v1','qualified':candidate_pass,
        'native_qualified':True,'candidate_rejection_terminal':terminal,
        'candidate_rejection_reason':'; '.join(reason) if reason else None,
        'units_sha256':digest(args.units),'field_tolerance':tolerance,'dot_tolerance':float(direct.get('dot_tolerance',0)),
        'lr_start':lr_start,'lr_target':lr_target,'beta1':float(opt['adam_betas'][0]),'beta2':float(opt['adam_betas'][1]),
        'native_epoch_seconds':native['epoch_seconds'],'direct_epoch_seconds':candidate['epoch_seconds'],
        'historical_epoch_seconds':historical['epoch_seconds'],
        'matched_epoch_speed_reduction':1-candidate['epoch_seconds']/historical['epoch_seconds'],
        'device':devices[0],'per_lr_replay':replay,'aggregate_replay':aggregate,'candidate_actions':actual,
        'finite_step_overshoot_demonstrated':overshoot,'ascending_proposal_fraction':actual['ascending_proposal_fraction'],
        'native_input_identity_matched':True,'resume_qualified':True,'cpu_assessment_forecast':forecast,
        'demonstrated_failure':not candidate_pass,'correction':'; '.join(corrections) if corrections else None,
        'disposition':('qualified' if candidate_pass else 'terminal_candidate_rejection' if terminal else 'correction_required' if corrections else 'diagnosis_required'),
        'independent_query_interpretation':'reported separately; an unreproducible improvement is not convergence evidence',
        'evidence':evidence}
    immutable_json(args.output,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('units','action-replay','preparation','candidate-trace','native-trace','historical-trace',
                'candidate-lease','native-lease','historical-lease','resume-receipt','round3','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--candidate-prefix-trace', type=Path)
    args=p.parse_args();print(json.dumps(assess(args),indent=2,allow_nan=False))

if __name__=='__main__':main()
