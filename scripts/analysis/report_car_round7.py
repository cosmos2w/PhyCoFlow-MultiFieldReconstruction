#!/usr/bin/env python3
"""Build a DATA2-only Round 7 report from hash-bound completed evidence.

Run `template` for the manifest contract. This tool never performs model inference,
chooses a checkpoint, releases deferred tasks, publishes, or edits source evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from scripts.analysis.report_car_round5 import bound, doc, sha, save_figure, rows_for_case
from scripts.analysis.car_round7_render import render
from scripts.analysis.report_car_round6 import regional
from scripts.analysis.car_round7_preparation_report import validate_preparation, preparation_lines

ARMS = ('FFM_Round7_Composite_Scale_Control','FFM_Round7_Joint_Anchor_Scale')
BLOCKS = ('signed','joint','log_power_self','rho','third_order')
ARM_LABELS = {ARMS[0]:'Composite control', ARMS[1]:'Joint-anchor candidate'}


def compact_preparation_lines(data):
    lines=preparation_lines(data)
    start=next(i for i,line in enumerate(lines) if line.startswith('| State | Objective | Compared run |'))
    end=start+2
    while end<len(lines) and lines[end].startswith('|'):end+=1
    rows=[[cell.strip() for cell in line.strip('|').split('|')] for line in lines[start+2:end]]
    sections=[]
    for title,indices in (('Family parameter rows',(3,4,5)),('Field, direction and displacement',(6,7,8))):
        headings=['State','Objective','Compared run']+[lines[start].strip('|').split('|')[i].strip() for i in indices]
        sections.extend([f'### {title}','', '| '+' | '.join(headings)+' |','| '+' | '.join(['---']*len(headings))+' |'])
        for row in rows:
            short=[row[0].replace('fresh_source','SOURCE').replace('mature_checkpoint','Mature'),row[1].replace('joint_anchor','Joint anchor').capitalize(),row[2].replace('separate_repeat_2 (torch.float32)','Separate repeat').replace('factored_static_features (torch.float32)','Factored static')]
            sections.append('| '+' | '.join(short+[row[i] for i in indices])+' |')
        sections.append('')
    return lines[:start]+sections+lines[end:]


def finite(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('finite numerical evidence required')
    return float(value)


def refs(value):
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value:
            bound(value); yield value
        else:
            for child in value.values(): yield from refs(child)
    elif isinstance(value,list):
        for child in value: yield from refs(child)


def read_lines(ref):
    path=bound(ref)
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows: raise ValueError(f'empty training history: {path}')
    return rows


def validate_terminal_history(history,arm,terminal_epoch):
    if [row['epoch'] for row in history]!=list(range(1,len(history)+1)) or len(history)<100:
        raise ValueError('complete ordered history from epoch 1 through at least 100 required')
    if len(history)!=terminal_epoch:
        raise ValueError(f'{arm} history must cover its entire declared terminal epoch {terminal_epoch}')
    if any(row.get('epoch_complete') is not True for row in history):
        raise ValueError('incomplete training epoch in history')


def calibration_intervals(calibration, uncertainty, reference_sha256):
    """Use the saved TRAIN condition-bootstrap intervals without refitting."""
    if (uncertainty.get('schema') != 'car_round7_source_training_law_uncertainty_v1' or
        uncertainty.get('sample_count') != 800 or uncertainty.get('replicates') != 2000 or
        uncertainty.get('reference', {}).get('sha256') != reference_sha256 or
        uncertainty.get('field_order') != [row['field'] for row in calibration['fields']]):
        raise ValueError('training-law uncertainty receipt does not match the frozen reference and fields')
    result=[]
    for index,row in enumerate(calibration['fields']):
        intervals=[]
        for key,readout in (('source_signed_risk','risk0'),('source_member_mse','member_mse0')):
            item=uncertainty[key]; mean=finite(item['mean'][index]);lo=finite(item['interval_95'][0][index]);hi=finite(item['interval_95'][1][index])
            if lo>hi or not math.isclose(mean,finite(row[readout]),rel_tol=1e-10,abs_tol=1e-14):
                raise ValueError(f'calibration readout disagrees with uncertainty receipt: {row["field"]} {key}')
            intervals.append((lo,hi))
        result.append(tuple(intervals))
    return result


def task_interval_text(task, name):
    item=task['paired'][name]
    mean=finite(item['mean']); low=finite(item['interval_95'][0]); high=finite(item['interval_95'][1])
    if low > high: raise ValueError('invalid saved paired task interval')
    return f'{mean:+.5g} [{low:+.5g}, {high:+.5g}]'


def check_snapshot(entry):
    assessment=doc(entry['assessment']); scale=doc(entry['scale_summary'])
    split=entry['split']; epoch=entry['epoch']; arm=entry['arm']
    if arm not in ARMS or split not in ('train','validation') or not isinstance(epoch,int) or epoch < 1:
        raise ValueError('descriptive arm, split and reporting epoch required')
    expected=800 if split=='train' else 100
    if (assessment.get('schema')!='car_round7_matched_assessment_v1' or assessment.get('status')!='completed' or
        assessment.get('split')!=split or assessment.get('members')!=4 or len(assessment.get('sample_ids',[]))!=expected or
        assessment.get('test_locked') is not True or scale.get('sample_ids')!=assessment['sample_ids'] or
        scale.get('split')!=split or scale.get('child')!='candidate'):
        raise ValueError('completed matched Round 7 assessment and Scale summary required')
    # Both files are generated together from the same checkpoint and panel.
    ap=bound(entry['assessment']); sp=bound(entry['scale_summary'])
    if ap.parent != sp.parent: raise ValueError('Scale summary must be from the matching assessment directory')
    child=doc(assessment['child_native']); selected=doc(assessment['child_selected'])
    if child['checkpoint_sha256']!=selected['checkpoint_sha256']:
        raise ValueError('native and Q4096 checkpoint mismatch')
    spec=doc(assessment['spec'])
    if spec['checkpoint']['sha256'] != child['checkpoint_sha256']:
        raise ValueError('assessment checkpoint does not match its specification')
    if spec.get('reporting_epoch') != epoch:
        raise ValueError('assessment specification must declare the checkpoint reporting epoch')
    if entry.get('checkpoint_sha256') != child['checkpoint_sha256']:
        raise ValueError('declared snapshot checkpoint differs')
    for name in BLOCKS:
        terms=scale['weighted_blocks'][name]
        if not math.isclose(finite(terms['attraction']['mean'])-finite(terms['separation']['mean']),
                            finite(terms['score']['mean']),rel_tol=2e-6,abs_tol=1e-8):
            raise ValueError(f'Scale A/R block mismatch: {name}')
    if not math.isclose(sum(scale['weighted_blocks'][k]['score']['mean'] for k in BLOCKS),
                        scale['definitions']['Complete_Expanded_Prefit']['mean'],rel_tol=2e-6,abs_tol=1e-8):
        raise ValueError('Scale composite blocks do not sum')
    if entry.get('tasks'):
        tasks=doc(entry['tasks']); decision=doc(tasks['decision'])
        if (split!='validation' or tasks.get('status')!='completed' or tasks['sample_ids']!=assessment['sample_ids'] or
            tasks['spec']!=assessment['spec'] or
            tasks['child_native']['sha256']!=assessment['child_native']['sha256'] or decision.get('status')!='sealed'):
            raise ValueError('tasks must be sealed and bound to the matching validation bank')
    return assessment,scale


def validate(inputs):
    if inputs.get('schema')!='car_round7_report_inputs_v1': raise ValueError('Round 7 report input schema required')
    list(refs(inputs))
    for name in ('formal_freeze','matched100_readout','calibration','calibration_readout','calibration_uncertainty','engineering','action_audit','action_readout','continuation_decision','terminal_decisions','visual_policy','event','preparation_readout'):
        doc(inputs[name])
    calibration=doc(inputs['calibration_readout']); audit=doc(inputs['action_readout'])
    if calibration.get('source_calibration_sha256') != inputs['calibration']['sha256'] or audit.get('source_action_audit_sha256') != inputs['action_audit']['sha256']:
        raise ValueError('calibration and action readouts must bind their raw evidence')
    if len(calibration.get('fields',[])) < 1 or finite(calibration.get('kappa')) <= 0 or not audit.get('comparisons'):
        raise ValueError('complete calibration and paired action readouts required')
    for row in calibration['fields']:
        member=finite(row['member_mse0']); allowance=finite(row['allowance']); scale=finite(row['risk_scale'])
        if member < 0 or not math.isclose(allowance,0.1*member,rel_tol=1e-10,abs_tol=1e-15) or not math.isclose(scale,max(member,1e-8),rel_tol=1e-10,abs_tol=1e-15):
            raise ValueError('calibration allowance or risk scale differs from the frozen training law')
    policy=doc(inputs['visual_policy']); uncertainty=doc(inputs['calibration_uncertainty'])
    bound(uncertainty['reference'])
    calibration_intervals(calibration,uncertainty,policy['training_reference']['sha256'])
    validate_preparation(doc(inputs['preparation_readout']), policy['training_reference']['sha256'], finite(calibration['kappa']), refs, doc)
    snapshots=inputs['snapshots']
    if len({(e['arm'],e['epoch'],e['split']) for e in snapshots})!=len(snapshots): raise ValueError('duplicate snapshot')
    loaded={(e['arm'],e['epoch'],e['split']):check_snapshot(e) for e in snapshots}
    for arm in ARMS:
        for split in ('train','validation'):
            if (arm,100,split) not in loaded: raise ValueError('both matched epoch-100 arms and splits required')
    for split in ('train','validation'):
        ids=[loaded[(arm,100,split)][0]['sample_ids'] for arm in ARMS]
        if ids[0]!=ids[1]: raise ValueError('epoch-100 arms must share the same ordered conditions')
    validate_matched100(inputs,loaded)
    terminal=inputs['terminal_snapshots']
    if {item['arm'] for item in terminal}!=set(ARMS): raise ValueError('each initial arm needs an explicit closure snapshot')
    for item in terminal:
        key=(item['arm'],item['epoch'],'validation')
        entry=next((e for e in snapshots if (e['arm'],e['epoch'],e['split'])==key),None)
        if not entry or not entry.get('tasks'): raise ValueError('sealed endpoint tasks required for each closed arm')
    decision=doc(inputs['terminal_decisions'])
    if decision.get('status')!='sealed' or not all(isinstance(decision.get(k),dict) and decision[k].get('outcome') and decision[k].get('rationale') for k in ('executor','hypothesis','model')):
        raise ValueError('sealed executor, hypothesis and model decisions with rationale required')
    if decision['model']['outcome'] not in ('retain SOURCE/C8',*ARMS): raise ValueError('invalid model decision')
    if decision['model']['outcome'] in ARMS:
        final=next(item for item in terminal if item['arm']==decision['model']['outcome'])
        safeguard=loaded[(final['arm'],final['epoch'],'validation')][0]['native_safeguards']
        if not (safeguard['primary_field_pass'] and safeguard['sensor_pass'] and safeguard['legacy_member_pass']):
            raise ValueError('replacement fails a required native field, sensor or member safeguard')
    for arm in ARMS:
        run=next((x for x in inputs['training'] if x['arm']==arm),None)
        if not run: raise ValueError('training evidence required for each arm')
        history=read_lines(run['history']); journal=read_lines(run['journal']); doc(run['recording_definitions']);doc(run['resume_receipt'])
        terminal_epoch=next(item['epoch'] for item in terminal if item['arm']==arm)
        validate_terminal_history(history,arm,terminal_epoch)
        if len(journal) != 12*len(history) or [r['step'] for r in journal]!=list(range(1,12*len(history)+1)):
            raise ValueError('complete ordered action journal required')
        for key in ('curves_pdf','curves_png'): bound(run[key])
    return loaded


def validate_matched100(inputs,loaded):
    """Bind saved direct paired effects to the declared matched100 snapshots."""
    readout=doc(inputs['matched100_readout'])
    if (readout.get('schema')!='car_round7_review100_readout_v1' or
        readout.get('status')!='completed' or readout.get('test_locked') is not True):
        raise ValueError('completed TEST-locked matched100 readout required')
    for split in ('train','validation'):
        data=readout['assessment'][split]
        panels=[]
        for name,arm in (('control',ARMS[0]),('joint',ARMS[1])):
            entry=next(item for item in inputs['snapshots'] if
                       (item['arm'],item['epoch'],item['split'])==(arm,100,split))
            evidence=data['evidence'][name]
            assessment,_=loaded[(arm,100,split)]
            spec=doc(assessment['spec']); panel=doc(spec['panel'])
            if (evidence.get('arm')!=arm or evidence.get('epoch')!=100 or evidence.get('split')!=split or
                evidence.get('checkpoint_sha256')!=entry['checkpoint_sha256'] or
                evidence.get('assessment',{}).get('sha256')!=entry['assessment']['sha256'] or
                evidence.get('scale_summary',{}).get('sha256')!=entry['scale_summary']['sha256'] or
                readout['training'][arm]['evidence']['checkpoint']['sha256']!=entry['checkpoint_sha256'] or
                spec['checkpoint']['sha256']!=entry['checkpoint_sha256'] or
                panel['sample_ids']!=assessment['sample_ids'] or
                data['sample_count']!=len(panel['sample_ids'])):
                raise ValueError('matched100 readout assessment, checkpoint or panel identity differs')
            panels.append(spec['panel']['sha256'])
        if panels[0]!=panels[1] or loaded[(ARMS[0],100,split)][0]['sample_ids']!=loaded[(ARMS[1],100,split)][0]['sample_ids']:
            raise ValueError('matched100 readout requires the same exact panel for both arms')
    return readout


def direct_interval(value):
    mean=finite(value['mean']); interval=value['interval_95']
    if len(interval)!=2: raise ValueError('paired 95% interval requires two endpoints')
    low,high=map(finite,interval)
    return f'{mean:+.5g} [{low:+.5g}, {high:+.5g}]'


def matched100_tables(readout):
    lines=['## Direct matched-100 comparison','',
        'Values are joint-anchor minus composite-control effects in the corresponding frozen TRAIN units, with saved paired 95% condition-bootstrap intervals. Both arms use the matched initial recipe and reporting age; VALIDATION is reused for development and selection. Later revised phases are not isolated same-age treatment comparisons.','',
        '| Split | State | Complete expanded Scale | Raw joint Scale | Shape |',
        '| --- | ---: | ---: | ---: | ---: |']
    for split in ('train','validation'):
        effect=readout['assessment'][split]['joint_minus_control']; scale=effect['scale'];core=effect['paired_core']
        values=(core['state'],scale['definitions']['Complete_Expanded_Prefit'],scale['raw_joint']['score'],core['shape'])
        lines.append(f'| {split} | '+' | '.join(direct_interval(value) for value in values)+' |')
    for split in ('train','validation'):
        blocks=readout['assessment'][split]['joint_minus_control']['scale']['weighted_blocks']
        lines.extend(['',f'**{split}: direct complete Scale block changes.**','',
            '| Weighted Scale block | Attraction change | Positive separation change | Score change |',
            '| --- | ---: | ---: | ---: |'])
        for block in BLOCKS:
            lines.append(f'| {block} | '+' | '.join(direct_interval(blocks[block][term]) for term in ('attraction','separation','score'))+' |')
    lines.extend(['','Each score is attraction minus positive separation; all intervals are the saved paired condition-bootstrap intervals.',''])
    return lines


def continuation_context_lines(decision,readout,diagnosis):
    """Explain the sealed epoch-100 decision without implying endpoint results."""
    validation=readout['assessment']['validation']['joint_minus_control']['scale']
    expanded=validation['definitions']['Complete_Expanded_Prefit']
    joint=validation['raw_joint']['score']
    spans=lambda value: finite(value['interval_95'][0]) <= 0 <= finite(value['interval_95'][1])
    uncertainty=('Both intervals include zero, so the point advantages are uncertain.'
                 if spans(expanded) and spans(joint) else
                 'The saved paired intervals quantify uncertainty in these initial effects.')
    windows=diagnosis['arms'][ARMS[1]]['windows']
    first=windows['51-75']; last=windows['76-100']
    early=finite(first['epoch_means']['constraint_gradient_norm'])
    late=finite(last['epoch_means']['constraint_gradient_norm'])
    positive=sum(finite(value)>0 for value in last['field_violation'].values())
    total=len(last['field_violation'])
    return ['## Sealed epoch-100 continuation context','',
        f"On reused development VALIDATION100, joint-anchor minus composite was {direct_interval(expanded)} for complete-expanded Scale and {direct_interval(joint)} for raw joint Scale, in frozen TRAIN units. Lower is better. {uncertainty}",'',
        f"For the joint arm, the mean weighted field-gradient norm was {early:.4g} in epochs 51–75 and {late:.4g} in epochs 76–100; {positive} of {total} mean training-law field violations remained positive in the latter window. The active PI feedback and remaining violations motivated the bounded recovery question, not a claim that retention had already recovered.",'',
        'The two epoch-100 arms are matched in age and exposure. If the selected arm reaches epoch 500, its comparisons with SOURCE and its own epoch-100 checkpoint describe package progress; its comparison with composite at epoch 100 is not an isolated treatment effect. VALIDATION is development and selection evidence, not independent confirmation.','']


def protocol_evidence(inputs):
    """Read the already-bound formal recipe and common calibration for reporting."""
    freeze=doc(inputs['formal_freeze'])
    prepared=doc(freeze['prepared'])
    configs={arm:yaml.safe_load(bound(prepared['arms'][arm]).read_text()) for arm in ARMS}
    return freeze,prepared,configs,doc(prepared['reference']),doc(inputs['calibration_readout'])


def methods_lines(inputs):
    _,_,configs,reference,calibration=protocol_evidence(inputs)
    cfg=configs[ARMS[0]]; candidate=configs[ARMS[1]]
    opt=cfg['optimization']; scalable=cfg['coherence']['scalable_anchored_coherence']
    budget=opt['multitask']['fidelity_budget']; rollout=cfg['rollout']
    pi=opt['multitask']['external_policy']['lagrangian']['epoch_calibrated_pi']
    c0=finite(reference['reference']['objective_scale']); kappa=finite(calibration['kappa'])
    control_scale=scalable.get('scale_objective','composite')
    joint_scale=candidate['coherence']['scalable_anchored_coherence']['scale_objective']
    return ['## Training and assessment methods','',
        f"Both arms restarted from the immutable {cfg['source']['weight_view']} SOURCE with fresh AdamW and PI state. Training used {opt['sampling'].replace('_',' ')} TRAIN8000 sampling, batch {opt['batch_size']}, {opt['steps_per_epoch']} actions per reporting epoch, {scalable['condition_query_groups']} query groups, Q{cfg['coherence']['compute_budget']['point_count']} selected support and {budget['rotating_complement']['points']} rotating complement points. Two independent members were reconstructed with {rollout['steps']} {rollout['solver']} steps; the AdamW learning rate warmed up over {opt['lr_warmup']['epochs']} epochs. Each reporting epoch therefore visits {opt['batch_size']*opt['steps_per_epoch']} conditions. Training curves were refreshed every {cfg['runtime']['plot_every_steps']} epochs and at pauses and completion.",'',
        f"The {control_scale} control optimized the expanded prefit Scale score, including sampled signed, joint, log-power, correlation and frozen triple terms. The {joint_scale} candidate optimized the full mean-augmented joint energy score alone, scaled by TRAIN-fitted kappa = {kappa:.7g}. Both kept State and Shape. Fresh parameter-space ConFIG combined the three coherence gradients; its direction was divided by C0 = {c0:.10g}, then the signed SOURCE-relative Lagrangian field gradient was added, followed by clipping and AdamW. The PI feedback used gamma = {finite(pi['gamma']):.10g}, smoothing time {finite(pi['tau_epochs']):g} epoch and horizon {finite(pi['horizon_epochs']):g} epochs. No native data loss entered these updates.",'',
        'The fixed field allowance was 0.1 times SOURCE member MSE, with the feedback scale floored at 1e-8; signed mean risk retained its sign. Evaluation used matched TRAIN800 and VALIDATION100 conditions, fixed T256 observations and four independent members. Native field predictions and the Q4096 coherence support were reconstructed separately. VALIDATION is reused for development and selection; TEST remains locked.','']


def calibration_tables(calibration,uncertainty_rows,prior):
    panel=prior.get('risk_reference_panel',{})
    names=panel.get('field_order',[]); values=panel.get('source_member_mse')
    old_mse=dict(zip(names,values,strict=True)) if isinstance(values,list) else {}
    lines=['| Field | New signed R0 [95% interval] | Prior signed R0 | New member M0 [95% interval] | Prior member M0 |',
           '| --- | ---: | ---: | ---: | ---: |']
    for row,(risk_ci,mse_ci) in zip(calibration['fields'],uncertainty_rows,strict=True):
        prior_member=f"{finite(old_mse[row['field']]):.6g}" if row['field'] in old_mse else 'unrecorded'
        lines.append(f"| {row['field']} | {finite(row['risk0']):.6g} [{risk_ci[0]:.6g}, {risk_ci[1]:.6g}] | {finite(row['prior_risk0']):.6g} | {finite(row['member_mse0']):.6g} [{mse_ci[0]:.6g}, {mse_ci[1]:.6g}] | {prior_member} |")
    lines.extend(['','| Field | New risk scale | New allowance | Prior allowance |',
                  '| --- | ---: | ---: | ---: |'])
    for row in calibration['fields']:
        lines.append(f"| {row['field']} | {finite(row['risk_scale']):.6g} | {finite(row['allowance']):.6g} | {finite(row['prior_allowance']):.6g} |")
    lines.append('')
    return lines


def field_interval(paired,index,*,scale=1.0):
    value=finite(paired['mean'][index])*scale
    lower=finite(paired['interval_95'][0][index])*scale
    upper=finite(paired['interval_95'][1][index])*scale
    return f'{value:+.5g} [{lower:+.5g}, {upper:+.5g}]'


def native_field_tables(snapshots,loaded,calibration):
    by_field={row['field']:row for row in calibration['fields']}
    lines=['## Native field comparisons','',
        'Each cell is the paired child-minus-SOURCE mean and 95% condition-bootstrap interval, except member MSE ratios, which are child divided by SOURCE. Signed-risk ratios use the new TRAIN-law allowance; the prior-allowance ratio is numerical historical context only and does not rewrite historical outcomes or select a model.','',
        'For the four native members, signed risk equals ensemble-mean MSE minus one quarter of unbiased member variance; member MSE equals ensemble-mean MSE plus three quarters of that variance. Feature-space energy-score separation is a different measure: its decrease does not establish a decrease in physical member spread.','']
    for entry in snapshots:
        assessment,_=loaded[(entry['arm'],entry['epoch'],entry['split'])]
        fields=assessment['field_order']; units=assessment['field_units']; data=assessment['paired_fields']
        lines.extend([f"### {ARM_LABELS[entry['arm']]} · epoch {entry['epoch']} · {entry['split']}",'',
            '| Field | Units | Signed risk / new allowance | Signed risk / prior allowance | Member MSE ratio |',
            '| --- | --- | ---: | ---: | ---: |'])
        for i,(field,unit) in enumerate(zip(fields,units,strict=True)):
            row=by_field[field]; previous=finite(row['prior_allowance']); new=finite(row['allowance'])
            old_ratio=field_interval(data['signed_risk_difference_over_allowance'],i,scale=new/previous) if previous>0 else 'unavailable (prior allowance zero)'
            lines.append(f"| {field} | {unit} | {field_interval(data['signed_risk_difference_over_allowance'],i)} | {old_ratio} | {field_interval(data['member_mse_ratio'],i)} |")
        lines.extend(['','| Field | Ensemble-mean MSE change | Member variance change | Sensor MSE change |',
                      '| --- | ---: | ---: | ---: |'])
        for i,field in enumerate(fields):
            count=sum(row['sensor_counts'][i] for row in assessment['per_condition_fields']['SOURCE'])
            sensor=field_interval(data['sensor_mse'],i) if count else 'unavailable (no sensors)'
            lines.append(f"| {field} | {field_interval(data['ensemble_mean_mse'],i)} | {field_interval(data['unbiased_member_variance'],i)} | {sensor} |")
        lines.extend(['',f"Physical support: {assessment['physical_mask']}. MSE, variance and sensor changes use squared declared field units. Member variance describes physical ensemble spread; it is distinct from ensemble-mean error and signed risk.",''])
    return lines


def named_evidence(inputs,loaded,campaign,logical_campaign,output):
    """Describe raw identities by role; retain portable publication separately."""
    freeze,prepared,_,reference,_=protocol_evidence(inputs)
    amendment=doc(freeze['code_amendment']) if freeze.get('code_amendment') else {}
    records=[]
    for entry in inputs['snapshots']:
        assessment,_=loaded[(entry['arm'],entry['epoch'],entry['split'])]
        spec=doc(assessment['spec']); panel=doc(spec['panel']); ids=panel['sample_ids']
        records.append({'arm':entry['arm'],'reporting_epoch':entry['epoch'],'split':entry['split'],
            'checkpoint':spec['checkpoint'],'assessment':entry['assessment'],
            'scale_summary':entry['scale_summary'],'native_bank_receipt':assessment['child_native'],
            'selected_bank_receipt':assessment['child_selected'],
            'source_native_bank_receipt':assessment['source_native'],
            'source_selected_bank_receipt':assessment['source_selected'],
            'source_baseline':assessment['source_baseline'],
            'panel':spec['panel'],'panel_sample_count':len(ids),
            'ordered_sample_ids_sha256':hashlib.sha256(json.dumps(ids,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()})
    entry=inputs['snapshots'][0]
    first=doc(loaded[(entry['arm'],entry['epoch'],entry['split'])][0]['spec'])
    return {'raw_data2_campaign_root':str(campaign),'logical_campaign_root':str(logical_campaign),
        'raw_report_build_root':str(output),
        'portable_publication_root':'UpdatePlans/_Reports/Coherence_Adaptive_Repair/Round_7',
        'portable_publication_contents':'selected ordinary-file Markdown/PDF, relative PNG/PDF figure pairs and compact evidence index',
        'formal_freeze':inputs['formal_freeze'],'formal_code_root':freeze['code_root'],
        'formal_code_capsule_id':Path(freeze['code_root']).name,
        'formal_execution_receipt':freeze.get('execution_code'),
        'qualified_code_root':amendment.get('qualified_code_root'),
        'qualified_code_capsule_id':Path(amendment['qualified_code_root']).name if amendment.get('qualified_code_root') else None,
        'qualified_execution_receipt':amendment.get('qualified_execution_receipt'),
        'code_amendment':freeze.get('code_amendment'),'qualification_decision':freeze.get('qualification_decision'),
        'source_checkpoint':first['source_checkpoint'],
        'source_weight_view':reference['binding']['weight_view'],
        'source_materialized_view_sha256':reference['binding'].get('materialized_view_sha256'),
        'common_training_reference':prepared['reference'],'prepared_configurations':freeze['prepared'],
        'prior_training_reference':reference.get('c0_preserved_from'),
        'arm_configs':prepared['arms'],'training_calibration':inputs['calibration'],
        'calibration_readout':inputs['calibration_readout'],'calibration_uncertainty':inputs['calibration_uncertainty'],
        'preparation_readout':inputs['preparation_readout'],'action_audit':inputs['action_audit'],
        'action_readout':inputs['action_readout'],'engineering_readout':inputs['engineering'],
        'training_histories_and_qualifications':inputs['training'],
        'matched100_readout':inputs['matched100_readout'],
        'matched100_curves':inputs.get('matched100_curves'),
        'matched100_diagnosis':inputs.get('matched100_diagnosis'),
        'continuation_decision':inputs['continuation_decision'],'terminal_decisions':inputs['terminal_decisions'],
        'snapshots':records}


def scale_figures(snapshots,loaded,figures):
    import numpy as np
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    captions={}
    for entry in snapshots:
        arm,epoch,split=entry['arm'],entry['epoch'],entry['split']; _,summary=loaded[(arm,epoch,split)]
        fig,grid=plt.subplots(2,2,figsize=(12,7),constrained_layout=True);axes=grid.ravel();x=np.arange(len(BLOCKS))
        for ax,term,title in zip(axes,('attraction','separation','score'),('Attraction A','Positive separation R','Fair score S = A - R')):
            values=[summary['weighted_blocks'][k][term]['mean'] for k in BLOCKS]
            ci=np.asarray([summary['weighted_blocks'][k][term]['interval_95'] for k in BLOCKS])
            ax.bar(x,values);ax.vlines(x,ci[:,0],ci[:,1],color='black');ax.axhline(0,color='black',lw=.6)
            ax.set_xticks(x,['Signed','Joint','Log-power','Correlation','Triples'],rotation=35,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.2)
        changes=np.asarray([summary['weighted_blocks'][key]['score']['mean'] for key in BLOCKS])
        running=np.concatenate(([0.],np.cumsum(changes)))
        span=float(np.max(running)-np.min(running)); pad=0.1*span if span>0 else 0.1
        axes[3].set_ylim(float(np.min(running))-pad,float(np.max(running))+pad)
        axes[3].bar(x,changes,bottom=running[:-1],color=['tab:blue' if value < 0 else 'tab:orange' for value in changes])
        axes[3].scatter([len(BLOCKS)],[running[-1]],marker='_',s=180,color='black')
        axes[3].axhline(0,color='black',lw=.6);axes[3].set_xticks([*x,len(BLOCKS)],['Signed','Joint','Log-power','Correlation','Triples','Total'],rotation=35,ha='right')
        axes[3].set_title('Additive score waterfall');axes[3].grid(axis='y',alpha=.2)
        axes[0].set_ylabel('Child minus SOURCE / frozen TRAIN Scale unit')
        name=f'scale_blocks_{arm}_epoch{epoch}_{split}'.lower();save_figure(fig,figures,name)
        captions[name]=f'{arm}, epoch {epoch}, {split}: all complete expanded Q4096 prefit Scale blocks; bars are weighted paired changes and whiskers are condition-bootstrap 95% intervals. Positive separation R is shown separately; score S = A - R.'
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for axis,kind in zip(axes,('Complete_Expanded_Prefit','joint')):
        for arm in ARMS:
            for split,style in (('train','-'),('validation','--')):
                entries=sorted([e for e in snapshots if e['arm']==arm and e['split']==split],key=lambda e:e['epoch'])
                vals=[];lo=[];hi=[]
                for entry in entries:
                    summary=loaded[(arm,entry['epoch'],split)][1]
                    v=summary['raw_joint']['score'] if kind=='joint' else summary['definitions'][kind]
                    vals.append(v['mean']);lo.append(v['interval_95'][0]);hi.append(v['interval_95'][1])
                xx=[e['epoch'] for e in entries];axis.plot(xx,vals,marker='o',ls=style,label=f'{ARM_LABELS[arm]} · {split}')
                axis.vlines(xx,lo,hi,alpha=.7)
        axis.axhline(0,color='black',lw=.6);axis.grid(alpha=.2);axis.set(xlabel='Reporting epoch',ylabel='Change / frozen TRAIN Scale unit',title='Raw joint' if kind=='joint' else 'Complete expanded prefit')
    axes[0].legend(fontsize=8);save_figure(fig,figures,'scale_split_trajectories')
    captions['scale_split_trajectories']='Declared checkpoint effects on TRAIN800 and reused development VALIDATION100. Whiskers are paired condition-bootstrap 95% intervals; lower scores favor the child. Intermediate points are not implied.'
    return captions


def field_figure(snapshots,loaded,figures):
    import numpy as np
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    entries=[e for e in snapshots if e['split']=='validation'];fig,axes=plt.subplots(len(entries),2,figsize=(12,3.3*len(entries)),squeeze=False,constrained_layout=True)
    for row,entry in enumerate(entries):
        a,_=loaded[(entry['arm'],entry['epoch'],'validation')];names=a['field_order'];x=np.arange(len(names));p=a['paired_fields'];risk=p['signed_risk_difference_over_allowance'];ratio=p['member_mse_ratio']
        for ax,v,title,limit in ((axes[row,0],risk,'Signed risk difference / TRAIN allowance',1.),(axes[row,1],ratio,'Member MSE ratio',1.1)):
            ax.bar(x,v['mean']);bounds=np.asarray(v['interval_95']);ax.vlines(x,bounds[0],bounds[1],color='black');ax.axhline(limit,color='red',ls='--');ax.set_xticks(x,names);ax.set_title(f"{ARM_LABELS[entry['arm']]} · epoch {entry['epoch']}: {title}");ax.grid(axis='y',alpha=.2)
    save_figure(fig,figures,'native_field_safeguards')
    return 'Native physical field signed-risk allowance ratios and member MSE ratios on development VALIDATION100. Whiskers are paired condition-bootstrap 95% intervals; dashed lines show the frozen allowances and 1.10 member threshold.'


def localization_figures(snapshots, loaded, figures):
    """Show every saved localized Scale view without recomputing scores."""
    import numpy as np
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import yaml
    captions = {}; co_rows = []
    for entry in snapshots:
        arm, epoch, split = entry['arm'], entry['epoch'], entry['split']
        assessment, summary = loaded[(arm, epoch, split)]
        spec = doc(assessment['spec']); config = yaml.safe_load(bound(spec['core_config']).read_text())
        prefit = doc(spec['prefit'])['coefficients']; bank = doc(spec['expanded_bank'])
        fields = assessment['field_order']; bands = config['coherence']['families']['multiscale_spectrum']['bands']['wavelengths']
        if bank['field_order'] != fields or 'CO' not in fields:
            raise ValueError('localized Scale labels must match the frozen field order and include CO')
        pairs = list(itertools.combinations(fields, 2)); local = summary['localization_raw']
        expected = {'joint_spectral_band_field_attribution':(len(bands),len(fields)),
                    'joint_piece_field_attribution':(None,len(fields)),
                    'signed_self_band_field':(len(bands),len(fields)),
                    'signed_pair_band':(len(bands),len(pairs)), 'rho_pair_band':(len(bands),len(pairs)),
                    'log_power_band_field_attribution':(len(bands),len(fields)),
                    'third_order_leaf':(len(bank['triples']),)}
        arrays = {}
        for key, shape in expected.items():
            arrays[key] = {}
            for term in ('attraction','separation','score'):
                value = np.asarray(local[key][term]['mean'], dtype=float)
                if value.ndim != len(shape) or any(n is not None and value.shape[i] != n for i,n in enumerate(shape)) or not np.isfinite(value).all():
                    raise ValueError(f'bad localized Scale array {key}/{term}: {value.shape}')
                arrays[key][term] = value
        for term in ('attraction','separation','score'):
            joint = arrays['joint_spectral_band_field_attribution'][term].sum() + arrays['joint_piece_field_attribution'][term].sum()
            expected_joint = finite(summary['raw_joint'][term]['mean'])
            logpower = arrays['log_power_band_field_attribution'][term].sum()
            expected_logpower = finite(summary['weighted_blocks']['log_power_self'][term]['mean']) / finite(prefit['scale.log_power_self'])
            triple = float(np.dot(arrays['third_order_leaf'][term], bank['target_weights']['scale.third_order']))
            expected_triple = finite(summary['weighted_blocks']['third_order'][term]['mean']) / finite(prefit['scale.third_order'])
            if not all(np.isclose(left,right,rtol=2e-6,atol=1e-8) for left,right in
                       ((joint,expected_joint),(logpower,expected_logpower),(triple,expected_triple))):
                raise ValueError(f'localized Scale attribution does not sum: {term}')
        ci = fields.index('CO')
        co = [float(arrays['joint_spectral_band_field_attribution'][term][:,ci].sum() +
                    arrays['joint_piece_field_attribution'][term][:,ci].sum()) for term in ('attraction','separation','score')]
        co_rows.append((arm,epoch,split,*co))
        field_fig, field_axes = plt.subplots(2,2,figsize=(14,8),constrained_layout=True)
        pair_fig, pair_axes = plt.subplots(2,2,figsize=(15,8.5),constrained_layout=True)
        for figure in (field_fig,pair_fig):
            figure.suptitle(f'{ARM_LABELS[arm]} · epoch {epoch} · {split}', fontsize=14)
        band_names = [f'Band {index+1} (λ {wavelength:g})' for index,wavelength in enumerate(bands)]
        def heat(fig, ax, matrix, title, columns, ylabels):
            limit = max(float(np.abs(matrix).max()),1e-15)
            image = ax.imshow(matrix, cmap='coolwarm', vmin=-limit, vmax=limit, aspect='auto')
            ax.set_title(title,fontsize=12);ax.set_xticks(range(len(columns)),columns,rotation=40,ha='right',fontsize=10)
            ax.set_yticks(range(len(ylabels)),ylabels,fontsize=10)
            for label in ax.get_xticklabels():
                if 'CO' in label.get_text(): label.set_fontweight('bold')
            fig.colorbar(image,ax=ax,shrink=.8,label='ΔS / TRAIN Scale unit')
        heat(field_fig,field_axes[0,0],arrays['joint_spectral_band_field_attribution']['score'],'Joint spectral norm allocation',fields,band_names)
        heat(field_fig,field_axes[0,1],arrays['joint_piece_field_attribution']['score'],'Joint connected-piece mean allocation',fields,
             [f'Piece {i+1}' for i in range(arrays['joint_piece_field_attribution']['score'].shape[0])])
        heat(field_fig,field_axes[1,0],arrays['signed_self_band_field']['score'],'Signed self projections',fields,band_names)
        heat(field_fig,field_axes[1,1],arrays['log_power_band_field_attribution']['score'],'Log-power norm allocation',fields,band_names)
        pair_labels=[f'{left} / {right}' for left,right in pairs]
        heat(pair_fig,pair_axes[0,0],arrays['signed_pair_band']['score'],'Signed pair projections',pair_labels,band_names)
        heat(pair_fig,pair_axes[0,1],arrays['rho_pair_band']['score'],'Correlation pair projections',pair_labels,band_names)
        field_total = arrays['joint_spectral_band_field_attribution']['score'].sum(0) + arrays['joint_piece_field_attribution']['score'].sum(0)
        pair_axes[1,0].bar(range(len(fields)),field_total);pair_axes[1,0].axhline(0,color='black',lw=.6)
        pair_axes[1,0].set_xticks(range(len(fields)),fields);pair_axes[1,0].set_title('Joint allocation summed by field');pair_axes[1,0].set_ylabel('ΔS / TRAIN Scale unit')
        for label in pair_axes[1,0].get_xticklabels():
            if label.get_text() == 'CO': label.set_fontweight('bold')
        triple_labels = [f"Q{i+1}: "+' · '.join(f'{fields[field]} B{band+1}' for band,field in terms)
                         for i,(_,terms) in enumerate(bank['triples'])]
        pair_axes[1,1].barh(range(len(triple_labels)),arrays['third_order_leaf']['score'])
        pair_axes[1,1].set_yticks(range(len(triple_labels)),triple_labels,fontsize=10)
        pair_axes[1,1].axvline(0,color='black',lw=.6);pair_axes[1,1].set_title('Frozen triple queries · raw score changes');pair_axes[1,1].set_xlabel('ΔS / TRAIN Scale unit')
        stem=f'scale_localization_{arm}_epoch{epoch}_{split}'.lower()
        shared=(f'{ARM_LABELS[arm]}, epoch {epoch}, {split}: localized Scale views. '
            'Values are raw block changes divided by the frozen TRAIN Scale unit, before prefit block weights; color scales are specific to each panel. '
            'Joint and log-power allocations partition a vector norm and are descriptive, not independently proper scores. '
            'Band wavelengths use the original unspecified coordinate units; CO labels are bold. The joint and log-power allocations and weighted triple leaves sum to their reported blocks.')
        for figure,suffix in ((field_fig,'fields'),(pair_fig,'pairs')):
            name=f'{stem}_{suffix}';save_figure(figure,figures,name);captions[name]=shared
    return captions, co_rows


def shape_component_figures(snapshots, loaded, figures):
    import numpy as np
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    captions = {}
    for entry in snapshots:
        arm,epoch,split=entry['arm'],entry['epoch'],entry['split']
        assessment,_=loaded[(arm,epoch,split)]; components=assessment['paired_shape_components']
        labels=['Self total','Self finite','Self essential','Mutual total','Mutual finite','Mutual essential']
        x=np.arange(len(labels)); attr=[]; sep=[]; score=[]; totals=[]
        for kind in ('self','mutual'):
            current=[]
            for category in ('finite','essential'):
                values=[components[f'shape.{kind}.{category}.{term}'] for term in ('attraction','separation')]
                a,r=(finite(value['mean']) for value in values)
                current.append((a,r,a-r,values))
            whole=components[f'shape.{kind}']; reconstructed=sum(v[2] for v in current)
            if not math.isclose(reconstructed,finite(whole['mean']),rel_tol=2e-6,abs_tol=1e-7+1e-6*sum(abs(v) for row in current for v in row[:2])):
                raise ValueError(f'Shape component reconstruction failed: {kind}')
            attr.extend([sum(v[0] for v in current),*(v[0] for v in current)])
            sep.extend([sum(v[1] for v in current),*(v[1] for v in current)])
            score.extend([finite(whole['mean']),*(v[2] for v in current)])
            totals.extend([(whole,current),*[(None,[v]) for v in current]])
        if not math.isclose(score[0]+score[3],finite(assessment['paired_core']['shape']['mean']),rel_tol=2e-6,abs_tol=1e-7):
            raise ValueError('Shape self/mutual does not reconstruct complete Shape')
        fig,grid=plt.subplots(2,2,figsize=(12,7.5),constrained_layout=True);axes=grid.ravel();axes[3].remove()
        fig.suptitle(f'{ARM_LABELS[arm]} · epoch {epoch} · {split}',fontsize=14)
        for ax,values,title in zip(axes,(attr,sep,score),('Attraction A','Positive separation R','Fair score S = A - R')):
            ax.bar(x,values);ax.axhline(0,color='black',lw=.6);ax.set_xticks(x,labels,rotation=35,ha='right',fontsize=9)
            ax.set_title(title);ax.grid(axis='y',alpha=.2)
        # Intervals exist for each primitive A/R component and whole self/mutual score.
        # Category score intervals cannot be obtained by subtracting separate interval endpoints.
        for j,(_,parts) in enumerate(totals):
            if len(parts)==1:
                for ax,term in ((axes[0],0),(axes[1],1)):
                    lo,hi=parts[0][3][term]['interval_95'];ax.vlines(j,lo,hi,color='black')
            elif totals[j][0] is not None:
                lo,hi=totals[j][0]['interval_95'];axes[2].vlines(j,lo,hi,color='black')
        axes[0].set_ylabel('Change / frozen TRAIN Shape unit')
        name=f'shape_components_{arm}_epoch{epoch}_{split}'.lower();save_figure(fig,figures,name)
        captions[name]=(f'{ARM_LABELS[arm]}, epoch {epoch}, {split}: complete Shape self and mutual contributions with finite and essential attraction and positive separation. '
            'Bars are paired child-minus-SOURCE changes in the frozen TRAIN Shape unit; available paired 95% condition-bootstrap intervals are shown. '
            'Primitive finite/essential score intervals and total A/R intervals are not inferred from separate interval endpoints; all terms sum to the frozen Shape score.')
    return captions


def resolve_visual_cases(policy_ref, assessment, task_ref):
    """Apply the frozen presentation rule only to the already nominated endpoint."""
    import numpy as np
    import torch
    policy = doc(policy_ref); tasks = doc(task_ref); spec = doc(assessment['spec'])
    if (policy.get('schema') != 'car_round7_visual_example_policy_v1' or
        policy.get('status') != 'frozen_before_formal_learning' or policy.get('split') != 'validation' or
        policy.get('test_locked') is not True or policy['panel'] != spec['panel'] or
        policy['source_native'] != assessment['source_native'] or policy['training_reference'] != spec['training_law_reference'] or
        policy['field_order'] != assessment['field_order'] or policy['field_units'] != assessment['field_units'] or
        policy['physical_mask'] != assessment['physical_mask']):
        raise ValueError('visual policy, assessment panel, reference or physical support differ')
    if (tasks.get('status') != 'completed' or tasks['child_native'] != assessment['child_native'] or
        doc(tasks['decision']).get('status') != 'sealed'):
        raise ValueError('visual example selection requires the sealed task release for this endpoint')
    child = doc(assessment['child_native']); source = doc(assessment['source_native']); panel = doc(policy['panel'])
    if policy['created_unix'] >= child['start_unix']:
        raise ValueError('visual policy must predate endpoint generation')
    source_row_path = bound(policy['normalizer_source_row'])
    if (source_row_path != bound(policy['source_native']).parent/'0000.pt' or
        sha(source_row_path) != source['files_sha256'][0]):
        raise ValueError('visual policy normalizer row differs from the SOURCE bank')
    source_row = torch.load(source_row_path,map_location='cpu',weights_only=True)
    reference = doc(policy['training_reference'])['reference']
    denominators = np.asarray(policy['representative_case_rule']['field_denominators_physical'],dtype=float)
    expected = np.asarray(reference['absolute_allowance'],dtype=float) * 10 * source_row['normalizer_scale'].double().numpy()**2
    if (reference['field_names'] != policy['field_order'] or not np.isfinite(denominators).all() or
        np.any(denominators <= 0) or not np.allclose(denominators,expected,rtol=1e-10,atol=0)):
        raise ValueError('visual policy field denominators do not match TRAIN M0 in physical units')
    ids=assessment['sample_ids']; rows=assessment['per_condition_fields']['candidate']
    if ids != panel['sample_ids'] or child['sample_ids'] != ids or len(rows) != len(ids):
        raise ValueError('visual policy requires the complete matched validation panel')
    values=np.asarray([row['ensemble_mean_mse'] for row in rows],dtype=float)
    if values.shape != (len(ids),len(denominators)) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('visual selection needs finite nonnegative per-condition ensemble-mean MSE')
    rank=(values/denominators).mean(axis=1); worst=int(np.argmax(rank))
    fixed=policy['fixed_cases']
    if len(fixed)!=2 or [x['ordinal'] for x in fixed] != [0,1] or any(ids[x['ordinal']] != x['sample_id'] for x in fixed):
        raise ValueError('visual policy requires the first two declared panel cases')
    cases=[{'ordinal':x['ordinal'],'sample_id':x['sample_id'],'label':f'Fixed panel case {i+1}',
            'normalized_mean_field_mse':float(rank[x['ordinal']])} for i,x in enumerate(fixed)]
    overlap=[i+1 for i,x in enumerate(fixed) if x['ordinal']==worst]
    cases.append({'ordinal':worst,'sample_id':ids[worst],
                  'label':'Representative worst' + (f' (also fixed case {overlap[0]})' if overlap else ''),
                  'normalized_mean_field_mse':float(rank[worst]),'overlap_fixed_cases':overlap})
    return {'schema':'car_round7_resolved_visual_examples_v1','policy':policy_ref,
            'assessment_child_native':assessment['child_native'],'checkpoint_sha256':child['checkpoint_sha256'],
            'task_release':tasks['decision'],'cases':cases,'test_locked':True}


def round7_case_figures(assessment, visual, figures):
    import numpy as np
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source_path=bound(assessment['source_native']);source=doc(assessment['source_native'])
    child_path=bound(assessment['child_native']);child=doc(assessment['child_native'])
    captions={}; fields=assessment['field_order']; units=assessment['field_units']
    for case_index,case in enumerate(visual['cases'],1):
        source_row, children=rows_for_case(source_path,source,{'repair':(child_path,child)},case['ordinal'],case['sample_id'])
        candidate=children['repair'];coords=source_row['coordinates_raw'].numpy()
        mass=source_row['native_masses'].reshape(-1).double().numpy()
        mask=(mass>0)&np.isfinite(coords[:,:2]).all(1)
        if not mask.any() or np.any(mass<0) or not np.isfinite(mass).all(): raise ValueError('invalid physical support')
        weight=mass[mask]/mass[mask].sum(); scale=source_row['normalizer_scale'];offset=source_row['normalizer_offset']
        truth=(source_row['truth_model'][0]*scale+offset).numpy()
        source_mean=(source_row['members_model'][:,0]*scale+offset).mean(0).numpy()
        child_mean=(candidate['members_model'][:,0]*scale+offset).mean(0).numpy()
        for first in range(0,len(fields),2):
            names=fields[first:first+2];fig,axes=plt.subplots(len(names),5,figsize=(16,3.4*len(names)),squeeze=False,constrained_layout=True)
            fig.suptitle(f"{case['label']} · {case['sample_id']}",fontsize=14)
            for local,field in enumerate(names):
                index=first+local; arrays=[truth[:,index],source_mean[:,index],child_mean[:,index]]
                low=min(a[mask].min() for a in arrays); high=max(a[mask].max() for a in arrays)
                residuals=[arrays[1]-arrays[0],arrays[2]-arrays[0]]
                limit=max(float(np.max(np.abs(a[mask]))) for a in residuals) or 1.
                for j,values in enumerate(arrays+residuals):
                    ax=axes[local,j];img=ax.scatter(coords[mask,0],coords[mask,1],c=values[mask],s=.5,
                        cmap='viridis' if j<3 else 'coolwarm',vmin=low if j<3 else -limit,vmax=high if j<3 else limit,rasterized=True)
                    ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
                    if j>=3:
                        rmse=np.sqrt(np.sum(weight*values[mask]**2))
                        ax.text(.02,.02,f'RMSE {rmse:.4g}',transform=ax.transAxes,color='white',fontsize=10,
                                bbox={'facecolor':'black','alpha':.7})
                    if local==0:ax.set_title(('Reference','SOURCE mean','Repair mean','SOURCE residual','Repair residual')[j])
                    if j==0:ax.set_ylabel(f'{field} ({units[index]})')
                fig.colorbar(img,ax=list(axes[local,3:]),shrink=.6)
                fig.colorbar(axes[local,2].collections[0],ax=list(axes[local,:3]),shrink=.6)
            name=f'field_example_{case_index}_part_{first//2+1}';save_figure(fig,figures,name)
            captions[name]=(f"{case['label']}, {case['sample_id']}, fields {', '.join(names)}: reference, SOURCE and repair use shared physical color limits. "
                'Residual badges show mass-weighted ensemble-mean RMSE on the displayed native positive-mass mask, in the declared field units; units marked unknown are unchanged from the data.')
    return captions


def training_figures(training,field_names,figures):
    import numpy as np
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    captions={}
    for run in training:
        arm=run['arm'];history=read_lines(run['history']);journal=read_lines(run['journal']);x=np.asarray([r['epoch'] for r in history]);jx=np.asarray([r['step']/12 for r in journal])
        fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
        for key in ('coherence_direction_norm','constraint_gradient_norm'):
            axes[0,0].plot(x,[finite(r[key]) for r in history],label=key)
        axes[0,1].plot(x,[finite(r['optimizer_lr']) for r in history],label='AdamW learning rate');axes[0,1].axvline(50,color='grey',ls=':',label='50-epoch warmup')
        for i,field in enumerate(field_names):
            axes[1,0].plot(jx,[finite(r['field_violation'][field]) for r in journal],label=field)
            axes[1,1].plot(jx,[finite(r['multipliers_after'][i]) for r in journal],label=field)
        keys=[key for key in history[0] if key.startswith('coherence_family/') and key.endswith('/raw')]
        if len(keys)<3:raise ValueError('all sampled family histories are required')
        for key in keys:axes[2,0].plot(x,[finite(r[key]) for r in history],label=key.removeprefix('coherence_family/'))
        actions=journal[0]['action_family_products']['actual_displacement']
        for key in actions:axes[2,1].plot(jx,[finite(r['action_family_products']['actual_displacement'][key]) for r in journal],label=key)
        axes[2,1].plot(jx,[finite(r['action_constraint_product']) for r in journal],label='field feedback')
        titles=('Parameter-direction norms','Learning rate','Signed pre-action field constraints','Post-action multipliers','Sampled family losses','Gradient dot actual AdamW displacement')
        fig.suptitle(ARM_LABELS[arm],fontsize=13)
        for ax,title in zip(axes.flat,titles):ax.set(title=title,xlabel='Reporting epoch');ax.grid(alpha=.2);ax.legend(fontsize=8)
        name=f'training_{arm}'.lower();save_figure(fig,figures,name)
        captions[name]=f'{arm}: complete recorded training history through epoch {len(history)}. The family losses are sampled training signals, the field feedback and actual displacement products are separate diagnostics, and no scalar ConFIG objective is implied.'
        leaves=[key for key in history[0] if key.startswith('coherence_component/') and key.endswith('/raw')]
        if not leaves:raise ValueError('complete sampled component histories required')
        for first in range(0,len(leaves),6):
            group=leaves[first:first+6];fig,axes=plt.subplots(len(group),1,figsize=(11,2.2*len(group)),squeeze=False,constrained_layout=True)
            for ax,key in zip(axes.flat,group):
                ax.plot(x,[finite(r[key]) for r in history]);ax.set(title=key.removeprefix('coherence_component/'),xlabel='Reporting epoch');ax.grid(alpha=.2)
            name=f'training_components_{arm}_{first//6+1}'.lower();save_figure(fig,figures,name)
            captions[name]=f'{arm}: sampled training component histories through epoch {len(history)}. These are optimization diagnostics, not complete panel assessments.'
    return captions


def engineering_figure(engineering,figures):
    import numpy as np
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    measurements=engineering['complete_epoch_measurements'];breakdown=engineering['measured_breakdown']
    if len(measurements)<2 or not breakdown: raise ValueError('measured full-epoch comparisons and cost breakdown required')
    devices={m['device_name'] for m in measurements}
    if len(devices)!=1:raise ValueError('executor timing comparison must use the same measured device')
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    labels=[m['method'] for m in measurements];x=np.arange(len(labels))
    for ax,key,title in ((axes[0],'seconds','Complete epoch wall time (s)'),(axes[1],'peak_allocated_gib','Peak allocated GPU memory (GiB)')):
        ax.bar(x,[finite(m[key]) for m in measurements]);ax.set_xticks(x,labels,rotation=30,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[1].scatter(x,[finite(m['peak_reserved_gib']) for m in measurements],marker='_',s=160,color='black',label='reserved');axes[1].legend(fontsize=7)
    axes[2].barh([r['scope'] for r in breakdown],[finite(r['seconds']) for r in breakdown]);axes[2].set_title('Measured scopes (s)');axes[2].grid(axis='x',alpha=.2)
    if any(finite(m['seconds'])<=0 or finite(m['peak_allocated_gib'])<=0 or finite(m['peak_reserved_gib'])<=0 for m in measurements):raise ValueError('positive measured epoch time and memory required')
    save_figure(fig,figures,'executor_time_memory_breakdown')
    return f"Measured executor epoch wall times and allocated/reserved GPU memory on {next(iter(devices))}; scope measurements are shown separately and may overlap, so they must not be added as exclusive costs. Measurements use the referenced full-epoch and trace evidence."


def action_figure(audit,figures):
    import numpy as np
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=audit['comparisons'];fig,axes=plt.subplots(len(rows),1,figsize=(10,2.0*len(rows)),squeeze=False,constrained_layout=True)
    for ax,row in zip(axes.flat,rows):
        values=[finite(row['composite']),finite(row['joint'])]
        ax.barh([0,1],values);ax.set_yticks([0,1],['Composite Scale','Joint-anchor Scale'])
        ax.axvline(0,color='black',lw=.6);ax.set_title(f"{row['checkpoint']}: {row['metric']} ({row['unit']})");ax.grid(axis='x',alpha=.2)
    save_figure(fig,figures,'same_state_action_audit')
    return 'Independently reset same-state composite and joint-anchored action measurements from the bound TRAIN action audit. Each panel uses the recorded units and checkpoint; these local actions do not establish learning or transfer.'


def relink_pdf(output,captions):
    # Run PyMuPDF in the local figure environment when the training env lacks it.
    worker = """
import fitz,json,pathlib,sys
output=pathlib.Path(sys.argv[1]); expected=set(json.loads(sys.argv[2])); document=fitz.open(output/'report.pdf'); found=set()
for page in document:
    for link in page.get_links():
        target=link.get('file','')
        if '/figures/' not in target: continue
        name=target.rsplit('/figures/',1)[1]
        if name not in expected or '/' in name: raise ValueError('unexpected figure link in PDF')
        document.xref_set_key(link['xref'],'A','<</S /URI /URI (figures/'+name+')>>');found.add(name)
if found!=expected: raise ValueError('missing PDF master links')
document.set_metadata({**document.metadata,'title':'Coherence Adaptive Repair: Round 7'})
temp=output/'report_relinked.pdf';document.save(temp,garbage=3,deflate=True);document.close();temp.replace(output/'report.pdf')
"""
    interpreter=os.environ.get('CAR_REPORT_PDF_PYTHON','/home/wanglz/miniconda3/envs/fig/bin/python')
    expected=[f'{key}.pdf' for key in captions]
    subprocess.run([interpreter,'-c',worker,str(output),json.dumps(expected)],check=True,capture_output=True,text=True)
    urls=subprocess.run(['pdfinfo','-url',str(output/'report.pdf')],check=True,capture_output=True,text=True).stdout.splitlines()[1:]
    actual=[row.split(None,2)[2] for row in urls if row.strip()]
    if set(actual)!=set('figures/'+name for name in expected):raise ValueError('PDF figure links are not portable')


def produce(inputs_path,campaign,output):
    started=time.time();logical_campaign=campaign.absolute();inputs_path=inputs_path.resolve(strict=True);campaign=campaign.resolve(strict=True);output=output.resolve()
    data_root=Path('/data2/wanglz/PhyCoFlow-MultiFieldReconstruction').resolve(strict=True)
    if data_root not in campaign.parents or campaign/'reports' not in output.parents:raise ValueError('new DATA2 campaign reports directory required')
    inputs=json.loads(inputs_path.read_text());loaded=validate(inputs)
    engineering=doc(inputs['engineering']);decision=doc(inputs['terminal_decisions'])
    calibration=doc(inputs['calibration_readout']);action=doc(inputs['action_readout'])
    uncertainty=doc(inputs['calibration_uncertainty']);policy=doc(inputs['visual_policy'])
    uncertainty_rows=calibration_intervals(calibration,uncertainty,policy['training_reference']['sha256'])
    # The compact engineering readout must bind its measured raw inputs explicitly.
    for key in ('screen_evidence','complete_epoch_evidence','trace_evidence'):
        if not engineering.get(key):raise ValueError(f'missing measured engineering evidence: {key}')
        for reference in engineering[key]:bound(reference)
    output.mkdir(parents=True,exist_ok=False);figures=output/'figures';figures.mkdir()
    captions=scale_figures(inputs['snapshots'],loaded,figures)
    localization_captions,co_rows=localization_figures(inputs['snapshots'],loaded,figures)
    captions.update(localization_captions)
    captions.update(shape_component_figures(inputs['snapshots'],loaded,figures))
    captions['native_field_safeguards']=field_figure(inputs['snapshots'],loaded,figures)
    captions.update(training_figures(inputs['training'],loaded[(ARMS[0],100,'validation')][0]['field_order'],figures))
    captions['executor_time_memory_breakdown']=engineering_figure(engineering,figures)
    captions['same_state_action_audit']=action_figure(action,figures)
    visual=inputs['visual_snapshot'];key=(visual['arm'],visual['epoch'],'validation')
    if key not in loaded:raise ValueError('visual snapshot must be a declared validation checkpoint')
    assessment,_=loaded[key]
    entry=next(x for x in inputs['snapshots'] if (x['arm'],x['epoch'],x['split'])==key)
    if not entry.get('tasks') or not any(x['arm']==visual['arm'] and x['epoch']==visual['epoch'] for x in inputs['terminal_snapshots']):
        raise ValueError('visual endpoint must be a nominated closed snapshot with sealed tasks')
    tasks=doc(entry['tasks'])
    if tasks['event'] != inputs['event']:raise ValueError('visual event differs from the sealed task')
    event_definition=doc(tasks['event'])['definition'];regional_definition=doc(tasks['regional_task'])
    task_units=doc(tasks['spec'])['task_normalizers']
    visual_selection=resolve_visual_cases(inputs['visual_policy'],assessment,entry['tasks'])
    (output/'visual_selection.json').write_text(json.dumps(visual_selection,indent=2,allow_nan=False)+'\n')
    captions.update(round7_case_figures(assessment,visual_selection,figures))
    regional({'per_condition_tasks':{'source':tasks['rows']['SOURCE'],'child':tasks['rows']['candidate']}},figures)
    captions['regional_and_event']='Frozen CO–U_1 regional covariance and the joint event task on the reused development validation panel. The covariance is normalized by TRAIN-truth scales; event values use the frozen area fair score.'
    lines=['# Coherence Adaptive Repair: Round 7','',
        'TRAIN800 and VALIDATION100 use matched condition IDs, fixed T256 observations and four independent members. The Q4096 coherence support is reconstructed separately from native field support. VALIDATION is a reused development and selection panel; TEST remains locked.','',
        '## Decisions','']
    for label,key in (('Exact executor','executor'),('Joint-Scale hypothesis','hypothesis'),('Raw model','model')):
        rationale=decision[key]['rationale']
        rationale=re.sub(r'(?<=[a-z])(?=\d)|(?<=\d)(?=[a-zA-Z])|(?<=[a-z])(?=[+-]\d)|(?<=[\d%])(?=\[)|(?<=,)(?=\S)', ' ', rationale)
        lines.extend([f"**{label}: {decision[key]['outcome']}.** {rationale}",''])
    lines.extend(methods_lines(inputs))
    lines.extend(compact_preparation_lines(doc(inputs['preparation_readout'])))
    lines.extend(['## Training-law calibration','',f"The fixed joint-anchor magnitude is kappa = {finite(calibration['kappa']):.7g}, estimated on the declared TRAIN800 law. The field references below are model-unit quantities; physical assessment uses the saved field normalization.",''])
    _,_,_,reference,_=protocol_evidence(inputs)
    lines.extend(calibration_tables(calibration,uncertainty_rows,doc(reference['c0_preserved_from'])))
    lines.extend(['','Intervals are the saved 2,000-replicate condition bootstrap, retaining dependence between the two members within each condition. Kappa is the fitted gradient-norm ratio; no kappa uncertainty was recorded.',''])
    lines.extend(['','## Matched results','', '| Arm | Epoch | Split | Complete expanded Scale | Raw joint Scale | State | Shape |', '| --- | ---: | --- | ---: | ---: | ---: | ---: |'])
    for entry in inputs['snapshots']:
        a,scale=loaded[(entry['arm'],entry['epoch'],entry['split'])]
        vals=[scale['definitions']['Complete_Expanded_Prefit']['mean'],scale['raw_joint']['score']['mean'],a['paired_core']['state']['mean'],a['paired_core']['shape']['mean']]
        lines.append(f"| {ARM_LABELS[entry['arm']]} | {entry['epoch']} | {entry['split']} | "+' | '.join(f'{finite(v):+.5g}' for v in vals)+' |')
    lines.extend(['','All values are paired child-minus-SOURCE changes divided by their frozen TRAIN SOURCE units; the figures show paired condition-bootstrap intervals. The earlier 0.02 regression and 0.05 improvement references are context, not an automatic combined veto.','',
        *matched100_tables(doc(inputs['matched100_readout'])),
        *continuation_context_lines(doc(inputs['continuation_decision']),doc(inputs['matched100_readout']),
                                    doc(doc(inputs['continuation_decision'])['focused_diagnosis']))])
    if inputs.get('lineage_progress'):
        progress=doc(inputs['lineage_progress'])
        if progress.get('schema')!='car_round7_selected500_lineage_progress_v1':
            raise ValueError('selected-500 lineage progress receipt required')
        lines.extend(['## Joint-anchor progression from epoch 100 to 500','','Paired changes below are epoch 500 minus epoch 100 for the same joint-anchor lineage and conditions, in frozen TRAIN units; lower scores are better. VALIDATION was reused for development, and this is not a matched-age comparison with the composite control.','',
            '| Split | Measure | Paired change [95% interval] |','| --- | --- | ---: |'])
        for split in ('train','validation'):
            item=progress['assessment'][split]
            for label,side,epoch in (('control','control',100),('joint','joint',500)):
                entry=next(e for e in inputs['snapshots'] if e['arm']==ARMS[1] and e['epoch']==epoch and e['split']==split)
                if any(item['evidence'][side][key]!=entry[key] for key in ('assessment','scale_summary')):
                    raise ValueError('lineage progress evidence differs from declared snapshots')
            delta=item['joint_minus_control']
            for label,value in (('Complete expanded Scale',delta['scale']['definitions']['Complete_Expanded_Prefit']),('Raw joint Scale',delta['scale']['raw_joint']['score']),('State',delta['paired_core']['state']),('Shape',delta['paired_core']['shape'])):
                mean=finite(value['mean']);lo,hi=map(finite,value['interval_95'])
                if lo>hi:raise ValueError('invalid lineage interval')
                lines.append(f'| {split} | {label} | {mean:+.5g} [{lo:+.5g}, {hi:+.5g}] |')
        lines.append('')
    lines.extend(['## CO contribution to the joint Scale norm','',
        'The CO entries sum its spectral-band and connected-piece allocations. They partition the joint vector norm and are descriptive allocations, not independent proper field scores. A is attraction and R is positive separation.','',
        '| Arm | Epoch | Split | CO ΔA | CO ΔR | CO ΔS |','| --- | ---: | --- | ---: | ---: | ---: |'])
    for arm,epoch,split,a,r,s in co_rows:
        lines.append(f'| {ARM_LABELS[arm]} | {epoch} | {split} | {a:+.5g} | {r:+.5g} | {s:+.5g} |')
    if inputs.get('training_readout'):
        training_readout=doc(inputs['training_readout'])
        window=training_readout['late_windows']['476-500']
        values=window['fields']['field_violation']
        lines.extend(['','## Late training-law field violations','','The selected joint-anchor run has the following mean training-law field violations over epochs 476–500. These sampled training-law quantities are distinct from the native panel safeguards below.','',
            '| Field | Mean training-law violation |','| --- | ---: |'])
        for name in ('CH4','CO','T','U_1','p'):
            lines.append(f'| {name} | {finite(values[name]):+.5g} |')
    lines.extend(['',*native_field_tables(inputs['snapshots'],loaded,calibration)])
    lines.extend(['','## Visual examples','',
        f"The frozen policy fixes validation ordinals 0 and 1. After endpoint nomination and task release, the representative worst is the maximum across the panel of five-field mean physical ensemble-mean MSE divided by the corresponding TRAIN M0, with ties resolved in panel order. The selected condition is {visual_selection['cases'][2]['sample_id']}; any overlap with a fixed case is retained and labeled. This selection is for presentation only and does not establish a field-budget violation.",'',
        '## Physical safeguards and tasks','', 'Native field-risk differences use the new fixed TRAIN-law allowances converted to physical units. Member MSE, ensemble-mean MSE, member spread, and sensor errors are distinct measures. The endpoint tasks were released only after the corresponding continuation or closure decision.',''])
    event_cuts=' and '.join(f"{name} {cut['operator']} {finite(cut['threshold']):.7g}" for name,cut in event_definition.items())
    region_fields=', '.join(regional_definition['field_names'])
    lines.extend([f"The frozen event is the native positive-mass intersection `{event_cuts}` in the declared field units; its score is the four-member fair score of event area fraction, divided by frozen TRAIN unit {finite(task_units['event']):.7g}.", '',
        f"The frozen regional task is {region_fields} covariance over `{regional_definition['region_rule']}`, with median x = {finite(regional_definition['median_x']):.7g} in the declared coordinate units and TRAIN-truth scales {', '.join(f'{finite(x):.7g}' for x in regional_definition['truth_scales'])}. The scored scalar is `{regional_definition['score']}`; its four-member fair score is divided by frozen TRAIN unit {finite(task_units['regional']):.7g}.",''])
    for item in inputs['terminal_snapshots']:
        a,_=loaded[(item['arm'],item['epoch'],'validation')];entry=next(e for e in inputs['snapshots'] if (e['arm'],e['epoch'],e['split'])==(item['arm'],item['epoch'],'validation'));t=doc(entry['tasks'])
        if t['event']!=tasks['event'] or t['regional_task']!=tasks['regional_task'] or doc(t['spec'])['task_normalizers']!=task_units:
            raise ValueError('terminal task definitions and frozen units must agree')
        lines.extend([f"{item['arm']} at epoch {item['epoch']}: native signed-field pass = {a['native_safeguards']['primary_field_pass']}, sensor pass = {a['native_safeguards']['sensor_pass']}, member-MSE pass = {a['native_safeguards']['legacy_member_pass']}. Event change [paired 95% interval] = {task_interval_text(t,'event')}; regional covariance change [paired 95% interval] = {task_interval_text(t,'regional')} in the respective frozen TRAIN units. A positive event-score change is worse.",''])
    costs=engineering['costs']
    lines.extend(['## Measured cost','', '| Cost measure | Recorded value | Units |', '| --- | ---: | --- |'])
    for row in costs:
        value='unavailable' if row['value'] is None else f"{finite(row['value']):.6g}"
        lines.append(f"| {row['measure']} | {value} | {row['units']} |")
    lines.extend(['','Cost entries include failures and CPU scoring or delivery only to the extent recorded in the bound accounting; unavailable values remain unavailable.',''])
    if inputs.get('endpoint_score_launcher'):
        launcher=doc(inputs['endpoint_score_launcher']);previous=launcher['previous_attempt']
        if previous['cpu_cost']=='unknown':
            lines.extend(['The first selected-500 CPU scoring attempt stopped without a terminal receipt; its consumed CPU time and exit cause are unknown. The successful replacement is separately timed in the bound launcher receipt. Neither attempt used a GPU.',''])
    lines.extend(['## Figure index',''])
    for name in captions:lines.append(f'- [Figure {name.replace("_"," ")}](#figure-{name.replace("_","-")})')
    lines.append('')
    for name,caption in captions.items():lines.extend([f'## Figure {name.replace("_"," ")}','',f'![{caption}](figures/{name}.png)','',f'{caption} [PDF master](figures/{name}.pdf)',''])
    (output/'report.md').write_text('\n'.join(lines))
    pages=render(output,captions);relink_pdf(output,captions)
    evidence={'schema':'car_round7_report_evidence_v1','input_manifest_sha256':sha(inputs_path),
        'report_generator_sha256':sha(Path(__file__)),
        'identity_map':named_evidence(inputs,loaded,campaign,logical_campaign,output),
        'engineering_evidence':{key:engineering[key] for key in ('screen_evidence','complete_epoch_evidence','trace_evidence')},
        'visual_policy_sha256':inputs['visual_policy']['sha256'],
        'visual_selection_sha256':sha(output/'visual_selection.json'),
        'visual_child_native_sha256':visual_selection['assessment_child_native']['sha256'],
        'visual_checkpoint_sha256':visual_selection['checkpoint_sha256'],
        'visual_cases':visual_selection['cases'],
        'artifacts':{'report_md_sha256':sha(output/'report.md'),'report_pdf_sha256':sha(output/'report.pdf'),
        'figures':{name:{'png_sha256':sha(figures/f'{name}.png'),'pdf_sha256':sha(figures/f'{name}.pdf')} for name in captions}},
        'test_locked':True}
    (output/'evidence_index.json').write_text(json.dumps(evidence,indent=2)+'\n')
    (output/'report_receipt.json').write_text(json.dumps({'schema':'car_round7_report_v1','status':'rendered_pending_visual_inspection',
        'pages':pages,'elapsed_seconds':time.time()-started,'evidence_index_sha256':sha(output/'evidence_index.json')},indent=2)+'\n')


def template():
    r={'path':'ABSOLUTE_EXISTING_FILE','sha256':'SHA256'}
    return {'schema':'car_round7_report_inputs_v1',
        'formal_freeze':dict(r),'matched100_readout':dict(r),'calibration':dict(r),'calibration_readout':dict(r),'calibration_uncertainty':dict(r),'engineering':dict(r),'action_audit':dict(r),'action_readout':dict(r),
        'continuation_decision':dict(r),'terminal_decisions':dict(r),'visual_policy':dict(r),'event':dict(r),'preparation_readout':dict(r),
        'training':[{'arm':arm,'history':dict(r),'journal':dict(r),'recording_definitions':dict(r),'resume_receipt':dict(r),
                     'curves_pdf':dict(r),'curves_png':dict(r)} for arm in ARMS],
        'snapshots':[{'arm':arm,'epoch':100,'split':split,'checkpoint_sha256':'CHECKPOINT_SHA256',
                      'assessment':dict(r),'scale_summary':dict(r),'tasks':None}
                     for arm in ARMS for split in ('train','validation')],
        'terminal_snapshots':[{'arm':arm,'epoch':'CLOSED_EPOCH_WITH_VALIDATION_TASKS'} for arm in ARMS],
        'visual_snapshot':{'arm':'TERMINAL_ARM','epoch':'CLOSED_EPOCH_WITH_TASKS'}}


def readout_templates():
    """Small measured-data adapters; every raw evidence reference is hash-bound."""
    r={'path':'ABSOLUTE_EXISTING_FILE','sha256':'SHA256'}
    return {
        'calibration_readout':{'source_calibration_sha256':'CALIBRATION_SHA256','kappa':'MEASURED_VALUE',
            'fields':[{'field':'FIELD','risk0':'NEW_R0','member_mse0':'NEW_M0','risk_scale':'MAX_M0_1E-8',
                       'allowance':'POINT_ONE_M0','prior_risk0':'PRIOR_R0','prior_allowance':'PRIOR_ALLOWANCE'}]},
        'action_readout':{'source_action_audit_sha256':'ACTION_AUDIT_SHA256',
            'comparisons':[{'checkpoint':'SOURCE_OR_MATURE','metric':'MEASURED_METRIC','unit':'UNITS',
                            'composite':'MEASURED_VALUE','joint':'MEASURED_VALUE'}]},
        'engineering':{'screen_evidence':[dict(r)],'complete_epoch_evidence':[dict(r)],'trace_evidence':[dict(r)],
            'complete_epoch_measurements':[{'method':'EXECUTOR','seconds':'MEASURED','peak_allocated_gib':'MEASURED',
                                            'peak_reserved_gib':'MEASURED','device_name':'RECORDED_DEVICE'}],
            'measured_breakdown':[{'scope':'RECORDED_SCOPE','seconds':'MEASURED'}],
            'costs':[{'measure':'OCCUPIED_GPU_TIME_INCLUDING_FAILURES','value':'MEASURED_OR_NULL','units':'hours'},
                     {'measure':'CPU_SCORING_AND_DELIVERY_TAIL','value':'MEASURED_OR_NULL','units':'seconds'}]},
        'terminal_decisions':{'status':'sealed','executor':{'outcome':'MEASURED_CHOICE','rationale':'EVIDENCE_BASED_REASON'},
            'hypothesis':{'outcome':'SUPPORTED_REJECTED_OR_UNRESOLVED','rationale':'EVIDENCE_BASED_REASON'},
            'model':{'outcome':'retain SOURCE/C8 OR EXACT_ARM','rationale':'EVIDENCE_BASED_REASON'}}}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('template')
    sub.add_parser('readout-templates')
    v=sub.add_parser('validate');v.add_argument('--inputs',type=Path,required=True)
    b=sub.add_parser('build');b.add_argument('--inputs',type=Path,required=True);b.add_argument('--campaign',type=Path,required=True);b.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='template':print(json.dumps(template(),indent=2));return
    if a.command=='readout-templates':print(json.dumps(readout_templates(),indent=2));return
    if a.command=='validate':
        loaded=validate(json.loads(a.inputs.read_text()));print(json.dumps({'status':'validated','snapshots':len(loaded),'input_sha256':sha(a.inputs)}));return
    produce(a.inputs,a.campaign,a.output)


if __name__=='__main__':main()
