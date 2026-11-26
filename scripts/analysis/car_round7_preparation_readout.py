#!/usr/bin/env python3
"""Summarize a terminal Round 7 executor/resume/pilot package on CPU.

No model is loaded or evaluated. Inputs must be explicit, completed receipts;
this does not poll a run, infer efficacy, or decide executor adoption.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

import yaml
from PIL import Image

OBJECTIVES=('composite','joint_anchor')
FAMILIES=('global_distribution','multiscale_spectrum','topology')
METRICS=('values','rows','field','direction','displacement','risk')


def ref(path):
    path=Path(path).resolve(strict=True)
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return {'path':str(path),'sha256':digest.hexdigest()}


def document(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def finite(value, label):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    return float(value)


def finite_tree(value,label):
    if isinstance(value,dict):
        for key,item in value.items():finite_tree(item,label+'.'+str(key))
    elif isinstance(value,(tuple,list)):
        for index,item in enumerate(value):finite_tree(item,label+f'[{index}]')
    else:
        finite(value,label)


def numeric_difference(value,label,*,relative_nullable=False):
    for key in ('max_abs','l2_abs'):
        if finite(value[key],label+'.'+key)<0: raise ValueError('negative absolute error')
    relative=value['relative_l2']
    if relative is None:
        if not relative_nullable: raise ValueError(label+' relative error is missing')
    elif finite(relative,label+'.relative_l2')<0: raise ValueError('negative relative error')
    return {key:value[key] for key in ('max_abs','l2_abs','relative_l2')}


def summarize_screen(path,expected_state):
    doc=document(path)
    if doc.get('schema')!='car_round7_executor_screen_v1' or doc.get('status')!='completed_screen' or doc.get('state')!=expected_state:
        raise ValueError('completed expected-state per-family screen required')
    if tuple(doc.get('family_order',())) != FAMILIES:
        raise ValueError('unexpected parameter-family order')
    if doc.get('field_row_active') != (expected_state=='mature_checkpoint'):
        raise ValueError('fresh SOURCE must have inactive and mature must have active field feedback')
    policy=doc['backend_policy']
    if not (policy.get('deterministic') and policy.get('strict_algorithms') and policy.get('deterministic_algorithms_enabled')
            and policy.get('cudnn_deterministic') and not policy.get('cudnn_benchmark')
            and policy.get('cublas_workspace_config') in (':4096:8',':16:8')):
        raise ValueError('screen backend policy differs from strict production policy')
    attempts=[];selected_queries={}
    for objective in OBJECTIVES:
        selected=[item for item in doc['attempts'] if item['objective']==objective]
        expected=('separate_repeat_1','separate_repeat_2','factored_static_features')
        selected_by_name={item['setting']:item for item in selected}
        if len(selected_by_name)!=len(selected) or any(name not in selected_by_name for name in expected):
            raise ValueError('each objective needs two distinct separate baselines and cached factored candidate')
        selected=[selected_by_name[name] for name in expected]
        selected_queries[objective]=selected[0]['selected_query_sha256']
        if any(item.get('profiler_enabled') for item in selected): raise ValueError('unprofiled parity screens required')
        for index,item in enumerate(selected):
            desired=('separate','none') if index<2 else ('factored','static_features')
            if (item['method'],item['context_cache'])!=desired or not item['checkpointing'] or (index==2 and item['chunk']!=2):
                raise ValueError('screen executor policy differs')
            if item['selected_query_sha256']!=selected[0]['selected_query_sha256'] or item['complement_sha256']!=selected[0]['complement_sha256']:
                raise ValueError('screen query or complement identity differs')
            per_family=item['per_family_row_comparison_to_first_separate']
            if set(per_family) != set(FAMILIES): raise ValueError('family row summaries missing or unexpected')
            output={'objective':objective,'setting':item['setting'],'parameter_row_dtype':item['parameter_row_dtype'],
                'checkpointing':item['checkpointing'],'context_cache':item['context_cache'],'chunk':item['chunk'],
                'comparison_to_first_separate':{},'per_family_row_comparison_to_first_separate':{},
                'weighted_field_row_norms':item['weighted_field_row_norms'],
                'clip_input_norm':finite(item['clip_input_norm'],'clip_input_norm'),
                'forward_seconds':finite(item['forward_seconds'],'forward_seconds'),
                'reverse_seconds':finite(item['reverse_seconds'],'reverse_seconds'),
                'solver_optimizer_seconds':finite(item['solver_optimizer_seconds'],'solver_optimizer_seconds'),
                'peak_allocated_bytes':item['peak_allocated_bytes'],'peak_reserved_bytes':item['peak_reserved_bytes']}
            for metric in METRICS:
                output['comparison_to_first_separate'][metric]=numeric_difference(
                    item['comparison_to_first_separate'][metric],metric,relative_nullable=metric in ('field','displacement','values','risk'))
            for name in FAMILIES:
                value=per_family[name]
                result=numeric_difference(value,name,relative_nullable=True)
                result.update(baseline_norm=finite(value['baseline_norm'],name+' baseline norm'),
                              candidate_norm=finite(value['candidate_norm'],name+' candidate norm'))
                output['per_family_row_comparison_to_first_separate'][name]=result
            for name,value in output['weighted_field_row_norms'].items():finite(value,'weighted field norm '+name)
            attempts.append(output)
    return {'source':ref(path),'state':expected_state,'backend_policy':policy,
        'reference_sha256':doc['reference_sha256'],'config_sha256':doc['config_sha256'],
        'field_row_active':doc['field_row_active'],'source_identity':doc.get('source_identity'),
        'source_materialized_state_sha256':doc.get('source_materialized_state_sha256'),
        'checkpoint_sha256':doc.get('checkpoint_sha256'),'family_order':list(FAMILIES),
        'sample_ids':doc['sample_ids'],'selected_query_sha256_by_objective':selected_queries,
        'joint_kappa':doc.get('kappa'),
        'kappa_limitation':'screen receipt does not record kappa' if 'kappa' not in doc else None,
        'absolute_family_loss_values':'not stored in screen receipt; only their differences are recorded',
        'attempts':attempts,'wall_seconds':finite(doc['wall_seconds'],'screen wall_seconds')}


def summarize_resume(path):
    doc=document(path)
    if (doc.get('schema')!='car_round7_production_resume_v1' or doc.get('status')!='completed'
        or doc.get('qualified') is not True or doc.get('pause_action')!=12
        or doc.get('final_reporting_epoch')!=2 or doc.get('saved_config_reloaded') is not True):
        raise ValueError('completed one-epoch save plus one-epoch resume receipt required')
    evidence=[];configs=[]
    for index,key in enumerate(('uninterrupted','resumed')):
        run=Path(doc[key]);checkpoint=run/'checkpoints/last.pt';definitions=run/'logs/recording_definitions.json'
        if ref(checkpoint)['sha256']!=doc['checkpoint_sha256'][index] or ref(definitions)['sha256']!=doc['recording_definitions_sha256'][index]:
            raise ValueError('resume checkpoint or recording definitions hash changed')
        for name in ('resolved_config.yaml','logs/history.jsonl','logs/epoch_calibrated_pi_updates.jsonl','metrics/training_coupling.jsonl'):
            evidence.append(ref(run/name))
        evidence += [ref(checkpoint),ref(definitions)]
        cfg=yaml.safe_load((run/'resolved_config.yaml').read_text())
        configs.append(cfg['coherence']['scalable_anchored_coherence'].get('scale_objective','composite'))
    if configs[0]!=configs[1] or configs[0] not in OBJECTIVES: raise ValueError('resume objectives differ')
    return {'source':ref(path),'objective':configs[0],'qualified':True,'pause_action':12,
        'final_reporting_epoch':2,'saved_config_reloaded':True,'evidence':evidence,
        'wall_seconds':finite(doc['wall_seconds'],'resume wall_seconds')}


def decode_curve(path):
    path=Path(path)
    if path.suffix=='.png':
        with Image.open(path) as image:
            image.load(); dimensions=list(image.size)
    elif path.suffix=='.pdf':
        with tempfile.TemporaryDirectory(prefix='phycoflow-round7-curve-') as folder:
            prefix=Path(folder)/'page'
            subprocess.run(['pdftoppm','-f','1','-l','1','-singlefile','-scale-to','512','-png',str(path),str(prefix)],
                           check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            with Image.open(prefix.with_suffix('.png')) as image:
                image.load();dimensions=list(image.size)
    else: raise ValueError('unsupported curve type')
    if min(dimensions)<=0: raise ValueError('empty curve image')
    return {**ref(path),'decoded_dimensions':dimensions}


def summarize_action_products(journal, journal_path):
    def stats(values):
        values=[finite(value,'actual displacement product') for value in values]
        return {'count':len(values),'negative':sum(value < 0 for value in values),
                'zero':sum(value == 0 for value in values),'positive':sum(value > 0 for value in values),
                'minimum':min(values),'maximum':max(values),'mean':sum(values)/len(values)}
    result={name:stats([row['action_family_products']['actual_displacement'][name] for row in journal])
            for name in FAMILIES}
    result['weighted_field']=stats([row['action_constraint_product'] for row in journal])
    return {'source':ref(journal_path),'products':result,
            'interpretation':'first-order pre-action gradient dot actual post-action Adam displacement, not finite score change'}


def action_product_lines(summary):
    lines=['| Actual displacement product | Negative | Zero | Positive | Minimum | Maximum | Mean |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, item in summary['products'].items():
        lines.append(f"| {name} | {item['negative']} | {item['zero']} | {item['positive']} | {item['minimum']:.6g} | {item['maximum']:.6g} | {item['mean']:.6g} |")
    lines.extend(['', 'These are first-order products of pre-action gradients with actual Adam displacement, not finite score changes. Different batch, member and query realizations during warmup mean the small sampled joint rise does not establish a defect or an efficacy gate; the products do not establish its cause.', ''])
    return lines


def summarize_pilot(path):
    doc=document(path)
    if (doc.get('schema')!='car_round7_training_result_v1' or doc.get('target_epoch')!=10
        or doc.get('global_step')!=120 or doc.get('target_reached') is not True
        or doc.get('disposition')!='target_reached' or doc.get('arm')!='FFM_Round7_Joint_Anchor_Scale'):
        raise ValueError('completed 10-epoch joint pilot required')
    for key in ('checkpoint','saved_config'):
        if ref(doc[key]['path'])['sha256']!=doc[key]['sha256']:raise ValueError('pilot '+key+' hash changed')
    run=Path(doc['run_dir']);history_path=run/'logs/history.jsonl';journal_path=run/'logs/epoch_calibrated_pi_updates.jsonl'
    coupling_path=run/'metrics/training_coupling.jsonl';timing_path=run/'logs/epoch_timing.jsonl'
    scalable_path=run/'logs/scalable_updates.jsonl'
    history=rows(history_path);journal=rows(journal_path);coupling=rows(coupling_path);timing=rows(timing_path);scalable=rows(scalable_path)
    if (len(history)!=10 or [int(row['epoch']) for row in history]!=list(range(1,11))
        or any(row.get('epoch_complete') is not True or row['step']!=12*(i+1) for i,row in enumerate(history))):
        raise ValueError('ten contiguous completed 12-action reporting epochs required')
    for values,label in ((journal,'PI'),(coupling,'coupling')):
        if [row['step'] for row in values]!=list(range(1,121)):raise ValueError(label+' requires 120 contiguous actions')
    if any(not row.get('accepted',False) for row in coupling): raise ValueError('pilot includes unaccepted action')
    if [row['step'] for row in scalable]!=list(range(1,121)) or any(not row.get('update_accepted') for row in scalable):
        raise ValueError('120 contiguous accepted scalable updates required')
    exposure={str(group):{'self':0,'pair':0} for group in range(8)}
    for row in scalable:
        for draw in row['exposure']:
            group=str(draw['condition_group']);kind=draw['kind']
            if group not in exposure or kind not in exposure[group]:raise ValueError('unexpected query group or kind')
            exposure[group][kind]+=int(draw['count'])
    if any(not values['self'] or not values['pair'] for values in exposure.values()):raise ValueError('eight query groups require self and pair exposure')
    coupling_identities=[{key:row[key] for key in ('step','sample_ids','observation_sha256','query_sha256','member_seeds','first_member_generator_state_sha256','field_complement_sha256')} for row in coupling]
    if (len(timing)!=10 or [int(row['epoch']) for row in timing]!=list(range(1,11))
        or any(row.get('complete_epoch_duration') is not True or row.get('timed_updates')!=12 or row.get('accepted_updates')!=12 for row in timing)):
        raise ValueError('ten complete timed epochs required')
    needed=('optimizer_lr','round7_scale_kappa','topology.self','topology.mutual',
        'coherence_component/multiscale_spectrum/joint/raw',
        'coherence_component/multiscale_spectrum/joint/weighted_contribution',
        *(f'coherence_family/{name}/{part}' for name in FAMILIES for part in ('raw','weighted_contribution')))
    observed=[]
    for row in history:
        selected={key:finite(row[key],f'epoch {row["epoch"]} {key}') for key in needed}
        if selected['round7_scale_kappa']<=0 or selected['optimizer_lr']<0:raise ValueError('invalid kappa or learning rate')
        observed.append({'epoch':int(row['epoch']),**selected,
            'active_field_fraction':finite(row['round7_active_field_row_fraction'],'active field fraction')})
    for entry in journal:
        for key in ('field_violation','multipliers_before','multipliers_after','z_after',
                    'action_family_products','action_constraint_product','action_displacement_norm'):
            finite_tree(entry[key],'PI '+key)
    for index,item in enumerate(observed):
        first,last=journal[index*12],journal[index*12+11]
        item['pi_first_multipliers']=first['multipliers_before']
        item['pi_last_multipliers']=last['multipliers_after']
        item['pi_last_smoothed_error']=last['z_after']
        item['pi_last_field_violation']=last['field_violation']
        item['pi_last_action_displacement_norm']=last['action_displacement_norm']
    cfg=yaml.safe_load(Path(doc['saved_config']['path']).read_text())
    if cfg['coherence']['scalable_anchored_coherence'].get('scale_objective')!='joint_anchor':
        raise ValueError('pilot configuration is not joint-anchor')
    kappa=finite(cfg['coherence']['scalable_anchored_coherence']['scale_joint_kappa'],'pilot kappa')
    if any(not math.isclose(item['round7_scale_kappa'],kappa,rel_tol=1e-14,abs_tol=0.) for item in observed):
        raise ValueError('pilot recorded kappa differs from saved config')
    curves=[decode_curve(run/'logs'/name) for name in ('loss_history.png','loss_history.pdf','coherence_history.png','coherence_history.pdf')]
    files=[ref(p) for p in (history_path,journal_path,coupling_path,timing_path,scalable_path,run/'logs/recording_definitions.json')]
    return {'source':ref(path),'checkpoint':doc['checkpoint'],'saved_config':doc['saved_config'],
        'reporting_epochs':10,'contiguous_actions':120,'kappa':kappa,
        'reference_sha256':cfg['optimization']['multitask']['fidelity_budget']['reference_artifact']['sha256'],
        'epoch_observations':observed,
        'last_pi':journal[-1], 'epoch_timing':timing,
        'actual_displacement_products':summarize_action_products(journal,journal_path),
        'scalable_calibration':{'coefficients_first':scalable[0]['coefficients'],
            'coefficients_last':scalable[-1]['coefficients'],
            'coefficients_constant':all(row['coefficients']==scalable[0]['coefficients'] for row in scalable),
            'calibration_frozen_first':scalable[0]['calibration_frozen'],
            'calibration_frozen_last':scalable[-1]['calibration_frozen'],
            'calibration_frozen_constant':all(row['calibration_frozen']==scalable[0]['calibration_frozen'] for row in scalable)},
        'query_group_exposure':exposure,'coupling_identities':coupling_identities,
        'recording_semantics':'family and joint scores are sampled pre-action epoch averages; LR is averaged per accepted action; multipliers are before action and displacement is actual post-action Adam displacement','curves_decoded':curves,'evidence':files,
        'interpretation':'TRAIN sampled early execution/recording diagnostics only; no efficacy or held-out claim'}


def markdown_readout(out, base_dir):
    def number(value): return 'unavailable' if value is None else f'{value:.6g}'
    lines=['# Round 7 terminal preparation readout','',
        'This is an execution and recording qualification on TRAIN data. It does not decide executor adoption or establish model efficacy. Full hash-bound evidence and per-attempt timings are in the adjacent JSON.','',
        '| State | Objective | Setting | Family | Dtype | Baseline norm | Candidate norm | Max absolute error | Relative L2 error |',
        '| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |']
    for state in ('fresh_source','mature_checkpoint'):
        for item in out[state]['attempts']:
            for family,value in item['per_family_row_comparison_to_first_separate'].items():
                lines.append(f"| {state} | {item['objective']} | {item['setting']} | {family} | {item['parameter_row_dtype']} | {number(value['baseline_norm'])} | {number(value['candidate_norm'])} | {number(value['max_abs'])} | {number(value['relative_l2'])} |")
    lines.extend(['','| State | Objective | Setting | Observable | Max absolute error | L2 absolute error | Relative L2 error |',
                  '| --- | --- | --- | --- | ---: | ---: | ---: |'])
    for state in ('fresh_source','mature_checkpoint'):
        for item in out[state]['attempts']:
            for metric in ('values','field','direction','displacement','risk'):
                value=item['comparison_to_first_separate'][metric]
                lines.append(f"| {state} | {item['objective']} | {item['setting']} | {metric} | {number(value['max_abs'])} | {number(value['l2_abs'])} | {number(value['relative_l2'])} |")
    lines.extend(['','The field entry is the single multiplier-weighted parameter row, inactive at fresh SOURCE. Zero reference norms make relative errors unavailable; absolute errors and norms remain visible. The screen records differences in family values, not the absolute values, and its receipt does not record joint κ.','',
        'Both objective-specific fresh-process resume receipts report successful one-epoch save plus one-epoch resume. The separately supplied zero-tolerance comparison reports no differences in scientific checkpoints, recording definitions, scientific histories, PI journals, coupling or scalable-update logs; its embedded receipt and artifact hashes were verified. The stated accounting exclusions are retained in the adjacent JSON.','',
        '| Pilot epoch | Learning rate | κ | Raw joint | Weighted joint | Shape self | Shape mutual | Active field fraction |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for item in out['joint_pilot']['epoch_observations']:
        lines.append(f"| {item['epoch']} | {number(item['optimizer_lr'])} | {number(item['round7_scale_kappa'])} | {number(item['coherence_component/multiscale_spectrum/joint/raw'])} | {number(item['coherence_component/multiscale_spectrum/joint/weighted_contribution'])} | {number(item['topology.self'])} | {number(item['topology.mutual'])} | {number(item['active_field_fraction'])} |")
    lines.extend(['','The ten-epoch pilot has 120 contiguous accepted actions and decoded loss and coherence curves in both PNG and PDF formats. PI state and epoch timings are recorded in the adjacent JSON. Sampled Shape and joint losses are training diagnostics, not complete panel scores.','',
        '## Figure index','', '- [Pilot loss history](#pilot-loss-history)', '- [Pilot coherence history](#pilot-coherence-history)'])
    lines.extend(['','## First-order action products',''])
    lines.extend(action_product_lines(out['joint_pilot']['actual_displacement_products']))
    curves={Path(item['path']).name:item['path'] for item in out['joint_pilot']['curves_decoded']}
    for title,name in (('Pilot loss history','loss_history'),('Pilot coherence history','coherence_history')):
        png=os.path.relpath(curves[name+'.png'],base_dir)
        pdf=os.path.relpath(curves[name+'.pdf'],base_dir)
        lines.extend(['',f'### {title}','',f'![{title} across ten reporting epochs]({png})','',
            f'{title}: recorded sampled training diagnostics across ten reporting epochs. [PDF master]({pdf})'])
    return '\n'.join(lines)+'\n'


def run(args):
    started=time.time();cpu=time.process_time()
    fresh=summarize_screen(args.fresh_screen,'fresh_source')
    mature=summarize_screen(args.mature_screen,'mature_checkpoint')
    if fresh['reference_sha256']!=mature['reference_sha256']:raise ValueError('screen field references differ')
    if fresh['sample_ids']!=mature['sample_ids'] or fresh['selected_query_sha256_by_objective']!=mature['selected_query_sha256_by_objective']:
        raise ValueError('screen condition IDs or selected support differ')
    resumes=[summarize_resume(path) for path in args.resume_receipts]
    if {item['objective'] for item in resumes}!=set(OBJECTIVES):raise ValueError('one resume receipt per objective required')
    strict=document(args.strict_resume_evidence)
    expected_names={Path(path).parent.name for path in args.resume_receipts}
    required=('strict_scientific_checkpoint_differences','strict_recording_definition_differences',
              'strict_scientific_history_differences','logs/epoch_calibrated_pi_updates.jsonl',
              'metrics/training_coupling.jsonl','logs/scalable_updates.jsonl')
    if (not isinstance(strict,list) or {row.get('name') for row in strict}!=expected_names
        or len(strict)!=2 or any(row.get(key)!=[] for row in strict for key in required)):
        raise ValueError('separate zero-tolerance resume comparison must report no differences for both named runs')
    for entry in strict:
        if entry.get('comparison_atol')!=0 or entry.get('comparison_rtol')!=0 or not isinstance(entry.get('accounting_exclusions'),list):
            raise ValueError('explicit zero comparison tolerances and accounting exclusions required')
        qualification=entry['qualification_receipt']
        actual_qualification=ref(qualification['path'])
        if (actual_qualification != qualification or Path(qualification['path']).parent.name!=entry['name']
            or actual_qualification['path'] not in {str(Path(p).resolve()) for p in args.resume_receipts}):
            raise ValueError('strict comparison qualification receipt differs')
        qualification_doc=document(qualification['path'])
        expected_files={'resolved_config.yaml','checkpoints/last.pt','logs/recording_definitions.json',
                        'logs/history.jsonl','logs/epoch_calibrated_pi_updates.jsonl',
                        'metrics/training_coupling.jsonl','logs/scalable_updates.jsonl'}
        for branch in ('uninterrupted','resumed'):
            evidence=entry['evidence'][branch]
            run_dir=Path(qualification_doc[branch]).resolve()
            if Path(evidence['run_dir']).resolve()!=run_dir or set(evidence['files'])!=expected_files:
                raise ValueError('strict comparison run or file set differs')
            for relative,binding in evidence['files'].items():
                if Path(binding['path']).resolve()!=run_dir/relative or ref(binding['path'])!=binding:
                    raise ValueError('strict comparison artifact hash differs: '+relative)
    pilot=summarize_pilot(args.pilot_result)
    if pilot['reference_sha256']!=fresh['reference_sha256']:raise ValueError('pilot and screens use different field references')
    out={'schema':'car_round7_terminal_preparation_readout_v1','status':'completed',
         'scope':'terminal preparation receipts only; no independent model evaluation, efficacy claim or executor decision',
         'fresh_source':fresh,'mature_checkpoint':mature,'production_resume':resumes,
         'strict_resume_comparison':{'source':ref(args.strict_resume_evidence),'results':strict,
             'binding':'embedded qualification receipts and every comparison artifact hash verified'},
         'joint_pilot':pilot,
         'timing':{'wall_seconds':time.time()-started,'cpu_seconds':time.process_time()-cpu}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    markdown=args.output.with_suffix('.md')
    if args.output.exists() or markdown.exists():raise FileExistsError('readout output already exists')
    with args.output.open('x') as stream:json.dump(out,stream,indent=2,allow_nan=False);stream.write('\n')
    with markdown.open('x') as stream:stream.write(markdown_readout(out,markdown.parent))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fresh-screen',type=Path,required=True);p.add_argument('--mature-screen',type=Path,required=True)
    p.add_argument('--resume-receipts',type=Path,nargs=2,required=True);p.add_argument('--pilot-result',type=Path,required=True)
    p.add_argument('--strict-resume-evidence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())


if __name__=='__main__':main()
