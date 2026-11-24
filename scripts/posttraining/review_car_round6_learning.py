#!/usr/bin/env python3
"""Record explicit Round 6 learning and terminal decisions from matched evidence.

Original thresholds are reported as references. This helper does not select a
checkpoint, run a model, infer a continuation from one scalar, or unlock TEST.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

FAMILIES=('state','scale','shape')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def binding(path):
    path=Path(path).resolve(strict=True)
    return {'path':str(path),'sha256':digest(path)}


def read(path):return json.loads(Path(path).read_text())


def require_text(text,label):
    if not isinstance(text,str) or not text.strip():raise ValueError(f'{label} requires a concrete explanation')
    return text.strip()


def assessment_identity(doc, *, terminal=False):
    if (doc.get('schema')!='car_round6_matched_validation_v1' or doc.get('split')!='validation'
        or doc.get('population')!=1000 or doc.get('members')!=4 or len(doc.get('sample_ids',[]))!=100
        or len(set(doc['sample_ids']))!=100 or doc.get('epoch') not in (100,500)):
        raise ValueError('matched approximately 10% VALIDATION100 J4 assessment required')
    required='reused development tasks at selected endpoint' if terminal else 'both tasks excluded'
    if doc.get('task_scope')!=required:raise ValueError('assessment task scope does not match decision stage')
    arm=doc.get('arm');candidate=doc.get('candidates',{}).get(arm)
    if not isinstance(candidate,dict):raise ValueError('one matched raw candidate required')
    if not terminal and ('reused_tasks' in candidate or 'reused_tasks' in doc.get('source',{})):
        raise ValueError('task-free review cannot consume endpoint task outcomes')
    if terminal and ('reused_tasks' not in candidate or 'reused_tasks' not in doc.get('source',{})):
        raise ValueError('terminal assessment requires both reused tasks')
    fields=doc.get('field_order',[]);fidelity=candidate.get('fidelity',{})
    if not fields or len(set(fields))!=len(fields):raise ValueError('unique ordered fields required')
    differences=candidate.get('normalized_core_differences',{})
    if set(differences)!=set(FAMILIES) or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in differences.values()):
        raise ValueError('all three finite normalized core differences required')
    differences={key:float(differences[key]) for key in FAMILIES}
    field_diff=fidelity.get('field_difference',[]);allow=fidelity.get('field_allowance',[])
    if len(field_diff)!=len(fields) or len(allow)!=len(fields) or any(not math.isfinite(float(x)) for x in field_diff+allow) or any(float(x)<=0 for x in allow):
        raise ValueError('finite field differences and positive field allowances required')
    flags={key:fidelity.get(key) for key in ('primary_field_pass','sensor_pass','primary_pass','legacy_member_pass')}
    if any(type(value) is not bool for value in flags.values()):raise ValueError('explicit field, sensor and member safeguards required')
    if flags['primary_pass']!=(flags['primary_field_pass'] and flags['sensor_pass']):
        raise ValueError('aggregate primary guard is inconsistent')
    return {'arm':arm,'fields':list(fields),'normalized_core_differences':differences,
        'field_difference':list(field_diff),'field_allowance':list(allow),
        'field_difference_over_allowance':[float(x)/float(y) for x,y in zip(field_diff,allow,strict=True)],
        'field_excess_over_allowance':[name for name,x,y in zip(fields,field_diff,allow,strict=True) if x>y],
        'physical_safeguards':flags,
        'original_reference':{'all_core_regressions_at_most_0_02':all(v<=.02 for v in differences.values()),
          'any_core_improvement_at_least_0_05':any(v<=-.05 for v in differences.values()),
          'replacement_eligible':candidate.get('replacement_eligible') if terminal else None}}


def check_history(history,epoch):
    if len(history)<epoch:raise ValueError('complete history through review epoch required')
    rows=history[:epoch]
    for index,row in enumerate(rows,1):
        if row.get('epoch')!=index or row.get('epoch_complete') is not True or row.get('step')!=12*index:
            raise ValueError('history must contain consecutive complete 12-action reporting epochs')
    return rows


def sampled_constraint_windows(rows,fields):
    """Descriptive last-batch samples, not complete-epoch constraint means."""
    if len(rows)<100:return None
    result={}
    for label,start,end in (('epochs_51_75',51,75),('epochs_76_100',76,100)):
        selected=rows[start-1:end]
        if any(not isinstance(row.get('field_violation'),dict) or set(row['field_violation'])!=set(fields) for row in selected):
            return None
        values={name:[float(row['field_violation'][name]) for row in selected] for name in fields}
        if any(not math.isfinite(x) for entries in values.values() for x in entries):
            raise ValueError('recorded constraint samples must be finite')
        result[label]={name:sum(v)/len(v) for name,v in values.items()}
    return {'definition':'mean of 25 saved last-batch signed normalized errors per window; not epoch means',
            'samples_per_window':25,'field_means':result}


def historical_comparison(doc, historical, summary):
    for k in ('split','population','members','field_order','sample_ids','source_checkpoint_sha256','source_bank_sha256','panel_sha256','contract_sha256','core_sha256','source_baseline_sha256'):
        if doc.get(k)!=historical.get(k):raise ValueError(f'historical comparison identity differs: {k}')
    if any(doc.get('normalizers',{}).get(k)!=historical.get('normalizers',{}).get(k) for k in FAMILIES):
        raise ValueError('historical comparison core normalizers differ')
    if historical.get('epoch')!=doc['epoch']:
        raise ValueError('historical comparison must match reporting age')
    arm='FFM_SourceScaledRisk_Uniform_Repair'
    hist=historical['candidates'][arm];f=hist['fidelity']
    diff=hist['normalized_core_differences']
    if set(diff)!=set(FAMILIES):raise ValueError('historical comparison lacks core families')
    return {'arm':arm,'epoch':historical['epoch'],'package_comparison':True,
        'limitation':'matched age and assessment identities; controller differs, so effects are not isolated',
        'normalized_core_differences':diff,
        'field_difference_over_allowance':[x/y for x,y in zip(f['field_difference'],f['field_allowance'],strict=True)],
        'sample_ids_match':True}


def decide_review(args):
    doc=read(args.assessment);summary=assessment_identity(doc)
    rows=check_history([json.loads(line) for line in args.history.read_text().splitlines() if line.strip()],doc['epoch'])
    expected={'bank_receipt_sha256':args.bank_receipt,'checkpoint_sha256':args.checkpoint,'prepared_sha256':args.prepared}
    for name,path in expected.items():
        if digest(path)!=doc.get(name):raise ValueError(f'assessment binding differs: {name}')
    bank=read(args.bank_receipt)
    if (bank.get('status')!='completed' or bank.get('epoch')!=doc['epoch'] or bank.get('arm')!=summary['arm']
        or bank.get('members')!=4 or bank.get('sample_ids')!=doc['sample_ids']
        or bank.get('source_bank_sha256')!=doc.get('source_bank_sha256')
        or bank.get('panel_sha256')!=doc.get('panel_sha256')
        or bank.get('checkpoint_sha256')!=doc['checkpoint_sha256']
        or bank.get('prepared_sha256')!=doc['prepared_sha256']
        or bank.get('config_sha256')!=digest(args.resolved_config)):
        raise ValueError('bank does not bind the same panel, candidate, checkpoint and resolved configuration')
    decision=args.decision;epoch=doc['epoch']
    if (epoch==100 and decision not in ('continue_to_500','stop_at_epoch_100')) or (epoch==500 and decision!='endpoint_at_epoch_500'):
        raise ValueError('decision must match reviewed epoch')
    if decision=='continue_to_500':
        if args.remaining_gpu_hours is None or args.forecast_remaining_gpu_hours is None:
            raise ValueError('continuation requires explicit remaining and forecast occupied GPU hours')
        if (not math.isfinite(args.remaining_gpu_hours) or not math.isfinite(args.forecast_remaining_gpu_hours)
            or args.remaining_gpu_hours<0 or args.forecast_remaining_gpu_hours<=0
            or args.forecast_remaining_gpu_hours>args.remaining_gpu_hours):
            raise ValueError('continuation must fit the stated remaining GPU budget')
    rationale=require_text(args.rationale,'decision rationale')
    progress=require_text(args.scientific_progress,'scientific progress')
    review_ref=binding(args.assessment);terminal=decision!='continue_to_500'
    result={'schema':'car_round6_learning_decision_v1','status':'completed','epoch':epoch,'decision':decision,
        'continue_to_500':decision=='continue_to_500','authorize_endpoint_tasks':terminal,
        'review_sha256':review_ref['sha256'],'bank_receipt_sha256':doc['bank_receipt_sha256'],
        'assessment':review_ref,'selection_use':'previously accessed development VALIDATION100; used for learning decisions and not independent confirmation',
        'decision_rationale':rationale,'scientific_progress':progress,
        'operational_model_replacement':'pending selected-endpoint tasks and explicit decision',
        'metrics':summary,'reference_thresholds_interpretation':'predeclared numeric margins remain visible references; trajectory and scientific progress require evidence and explicit judgment',
        'history_complete_through_epoch':epoch,'history_step':rows[-1]['step'],
        'training_constraint_samples_51_75_76_100':sampled_constraint_windows(rows,summary['fields']),
        'remaining_gpu_hours':args.remaining_gpu_hours,'forecast_remaining_gpu_hours':args.forecast_remaining_gpu_hours,
        'evidence':{'history':binding(args.history),'checkpoint':binding(args.checkpoint),'bank_receipt':binding(args.bank_receipt),
                    'prepared':binding(args.prepared),'resolved_config':binding(args.resolved_config),'resumption_protocol':binding(args.resumption_protocol)},
        'test_locked':True}
    if args.historical_review:
        hist=read(args.historical_review);result['historical_comparison']=historical_comparison(doc,hist,summary)
        result['evidence']['historical_review']=binding(args.historical_review)
    return result


def decide_disposition(args):
    doc=read(args.assessment);summary=assessment_identity(doc,terminal=True)
    prior=read(args.review_decision)
    if (prior.get('status')!='completed' or prior.get('authorize_endpoint_tasks') is not True
        or digest(args.review_decision)!=doc.get('task_decision_sha256')
        or prior.get('review_sha256')!=doc.get('review_sha256')
        or prior.get('bank_receipt_sha256')!=doc.get('bank_receipt_sha256') or prior.get('epoch')!=doc['epoch']):
        raise ValueError('terminal assessment must reuse the bank and task-free review explicitly selected')
    if digest(args.bank_receipt)!=doc['bank_receipt_sha256']:
        raise ValueError('terminal prediction bank receipt mismatch')
    protocol=binding(args.resumption_protocol)
    if prior.get('evidence',{}).get('resumption_protocol')!=protocol:
        raise ValueError('resumption protocol identity changed')
    arm=summary['arm'];candidate=doc['candidates'][arm];task=candidate['reused_tasks']
    for key in ('regional_normalized_improvement','event_normalized_difference'):
        if not isinstance(task.get(key),(int,float)) or not math.isfinite(task[key]):raise ValueError('finite terminal task scores required')
    if args.recommendation=='replace' and not (summary['physical_safeguards']['primary_pass'] and summary['physical_safeguards']['legacy_member_pass']):
        raise ValueError('automatic replacement cannot override field, sensor or member physical safeguard failure')
    result={'schema':'car_round6_terminal_disposition_v1','status':'completed','epoch':doc['epoch'],
        'recommendation':arm if args.recommendation=='replace' else 'retain SOURCE/C8',
        'reason':require_text(args.rationale,'operational decision rationale'),
        'scientific_progress':require_text(args.scientific_progress,'scientific progress'),
        'assessment_sha256':digest(args.assessment),'assessment':binding(args.assessment),
        'selection_use':'reused development VALIDATION100, not independent confirmation',
        'metrics':summary,'reused_tasks':{'regional_normalized_improvement':task['regional_normalized_improvement'],
           'event_normalized_difference':task['event_normalized_difference']},
        'original_task_reference':{'regional_improvement_at_least_0_05':task['regional_normalized_improvement']>=.05,
           'event_regression_at_most_0_05':task['event_normalized_difference']<=.05},
        'evidence':{'review_decision':binding(args.review_decision),'bank_receipt':binding(args.bank_receipt),
                    'resumption_protocol':protocol},'test_locked':True}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('review');r.add_argument('--assessment',type=Path,required=True);r.add_argument('--history',type=Path,required=True)
    r.add_argument('--checkpoint',type=Path,required=True);r.add_argument('--bank-receipt',type=Path,required=True)
    r.add_argument('--prepared',type=Path,required=True);r.add_argument('--resolved-config',type=Path,required=True)
    r.add_argument('--resumption-protocol',type=Path,required=True);r.add_argument('--historical-review',type=Path)
    r.add_argument('--decision',choices=('continue_to_500','stop_at_epoch_100','endpoint_at_epoch_500'),required=True)
    r.add_argument('--remaining-gpu-hours',type=float);r.add_argument('--forecast-remaining-gpu-hours',type=float)
    r.add_argument('--rationale',required=True);r.add_argument('--scientific-progress',required=True);r.add_argument('--output',type=Path,required=True)
    d=sub.add_parser('disposition');d.add_argument('--assessment',type=Path,required=True);d.add_argument('--review-decision',type=Path,required=True)
    d.add_argument('--bank-receipt',type=Path,required=True);d.add_argument('--resumption-protocol',type=Path,required=True)
    d.add_argument('--recommendation',choices=('replace','retain'),required=True);d.add_argument('--rationale',required=True)
    d.add_argument('--scientific-progress',required=True);d.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=decide_review(args) if args.command=='review' else decide_disposition(args)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(args.output)
if __name__=='__main__':main()
