#!/usr/bin/env python3
"""Render the selected Round 6 endpoint from hash-bound cached evidence."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import time

CODE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(CODE/'src'), str(CODE)]
from scripts.analysis.report_car_round5 import bound, doc, sha, save_figure, render, rows_for_case


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_model_decision(decision, candidate, arm):
    """Keep physical safeguards while allowing the authorized measured judgment."""
    if decision['recommendation'] not in (arm, 'retain SOURCE/C8'):
        raise ValueError('decision must identify the raw candidate or retain SOURCE/C8')
    resumed = decision.get('schema') == 'car_round6_terminal_disposition_v1'
    if resumed:
        if (decision.get('status') != 'completed' or decision.get('test_locked') is not True
                or not isinstance(decision.get('scientific_progress'), str)
                or not decision['scientific_progress'].strip()):
            raise ValueError('completed disposition with explicit scientific progress required')
        bound(decision.get('evidence', {}).get('resumption_protocol'))
        flags = decision.get('metrics', {}).get('physical_safeguards')
        fidelity = candidate['fidelity']
        actual = {key: fidelity[key] for key in ('primary_field_pass', 'sensor_pass',
                                                'primary_pass', 'legacy_member_pass')}
        if flags != actual:
            raise ValueError('disposition physical safeguards differ from the assessment')
        if decision['recommendation'] == arm and not (actual['primary_pass'] and actual['legacy_member_pass']):
            raise ValueError('candidate replacement requires field, sensor and member safeguards')
    elif decision['recommendation'] == arm and not candidate['replacement_eligible']:
        raise ValueError('candidate cannot be adopted without the original guards or an authorized disposition')
    return resumed


def series(ax, x, y, label):
    import numpy as np
    y = np.asarray(y, dtype=float); x = np.asarray(x, dtype=float)
    if not np.isfinite(y).all(): raise ValueError(f'nonfinite training history: {label}')
    ax.plot(x, y, alpha=.25, linewidth=.7)
    for i in range(0, len(y), 10):
        ax.plot([x[i], x[min(i+9,len(y)-1)]], [y[i:i+10].mean()]*2, linewidth=1.5, label=label if i==0 else None)


def learning(history, journal, names, coefficients, directory):
    import numpy as np
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    x=[r['epoch'] for r in history]
    keys=[k for k in history[0] if k.startswith('coherence_family/') and k.endswith('/raw') and isinstance(history[0][k],(int,float))]
    if not keys: raise ValueError('sampled family and leaf histories are missing')
    for key in ('coherence_direction_norm','constraint_gradient_norm'):
        series(axes[0,0],x,[r[key] for r in history],key)
    series(axes[0,1],x,[r['optimizer_lr'] for r in history],'AdamW learning rate')
    jx=np.asarray([r['step']/12 for r in journal])
    for field in names:
        values = [r['field_violation'][field] for r in journal] if journal else [r['field_violation'][field] for r in history]
        axes[1,0].plot(jx if journal else x,values,alpha=.55,label=field)
    for i,field in enumerate(names):
        values = [r['multipliers_after'][i] for r in journal] if journal else [r['policy_state']['multipliers'][i] for r in history]
        axes[1,1].plot(jx if journal else x,values,label=field)
    ratio=[]
    for r in history:
        denom=float(r['coherence_direction_norm']); numer=float(r['constraint_gradient_norm'])
        ratio.append(numer/denom if denom else float('nan'))
    series(axes[2,0],x,ratio,'field / coherence')
    if journal:
        products=journal[0]['action_family_products']['actual_displacement']
        for family in products:
            axes[2,1].plot(jx,[r['action_family_products']['actual_displacement'][family] for r in journal],alpha=.6,label=family)
        axes[2,1].plot(jx,[r['action_constraint_product'] for r in journal],alpha=.6,label='field feedback')
    else:
        axes[2,1].text(.5,.5,'Actual displacement products unavailable',ha='center',va='center',transform=axes[2,1].transAxes)
    titles=('Parameter-direction norms','Learning rate','Signed pre-action constraint','Post-action multiplier','Field/coherence gradient norm ratio','Gradient dot actual AdamW displacement')
    for ax,title in zip(axes.flat,titles):
        ax.set(title=title,xlabel='Reporting epoch'); ax.grid(alpha=.2)
        if ax.lines: ax.legend(fontsize=6)
    save_figure(fig,directory,'learning_and_action')
    fig, family_axes=plt.subplots(len(keys),1,figsize=(11,2.5*len(keys)),constrained_layout=True,squeeze=False)
    for ax,key in zip(family_axes.flat,keys):
        series(ax,x,[r[key] for r in history],key.removeprefix('coherence_family/'))
        ax.set(xlabel='Reporting epoch',title=key.removeprefix('coherence_family/'));ax.grid(alpha=.2)
    save_figure(fig,directory,'family_history')
    leaves=[k for k in history[0] if k.startswith('coherence_component/') and k.endswith('/raw') and isinstance(history[0][k],(int,float))]
    if not leaves: raise ValueError('sampled coherence leaf histories are missing')
    for start in range(0,len(leaves),6):
        group=leaves[start:start+6]; fig, axes=plt.subplots(len(group),1,figsize=(11,2.3*len(group)),constrained_layout=True,squeeze=False)
        for ax,key in zip(axes.flat,group):
            series(ax,x,[r[key] for r in history],key.removeprefix('coherence_component/'))
            ax.set(xlabel='Reporting epoch',title=key.removeprefix('coherence_component/')); ax.grid(alpha=.2)
        save_figure(fig,directory,f'leaf_history_{start//6+1}')
    fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    for name,value in coefficients.items():ax.plot(x,[value]*len(x),label=name)
    ax.set(xlabel='Reporting epoch',ylabel='Fixed coefficient',title='SOURCE-prefit coefficients');ax.grid(alpha=.2);ax.legend(fontsize=6,ncol=3)
    save_figure(fig,directory,'fixed_coefficients')
    return [f'leaf_history_{i//6+1}' for i in range(0,len(leaves),6)]


def outcomes(a, directory):
    import numpy as np
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arm=a['arm']; child=a['candidates'][arm]; fields=a['field_order']; x=np.arange(len(fields)); src=a['source']['risk']; risk=child['risk']
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    allowance=np.asarray(a['native_allowance']); excess=np.asarray(child['fidelity']['field_difference'])
    if (allowance <= 0).any(): raise ValueError('zero allowance needs explicit physical unit for normalized excess')
    bounds=np.asarray(a['paired_fields']['signed_mean_risk']['interval_95'])/allowance
    axes[0,0].bar(x,excess/allowance);axes[0,0].vlines(x,bounds[0],bounds[1],color='black');axes[0,0].axhline(1,color='red',linestyle='--');axes[0,0].set_title('Signed risk change / allowance; paired 95% CI')
    for key,label in (('ensemble_mean_mse','ensemble mean MSE'),('member_mse','member MSE'),('unbiased_member_variance','member variance')):
        base=np.asarray(src[key]); val=np.asarray(risk[key]); valid=base>0
        axes[0,1].plot(x[valid],val[valid]/base[valid],marker='o',label=label)
    axes[0,1].axhline(1,color='black',lw=.5);axes[0,1].legend(fontsize=7);axes[0,1].set_title('Native field error and spread ratios')
    core=a['paired_core']; keys=['state','scale','shape']; positions=np.arange(3)
    means=np.array([core[k]['mean'] for k in keys]); lo=np.array([core[k]['interval_95'][0] for k in keys]);hi=np.array([core[k]['interval_95'][1] for k in keys])
    axes[1,0].scatter(positions,means);axes[1,0].vlines(positions,lo,hi);axes[1,0].axhline(0,color='black',lw=.5);axes[1,0].set_xticks(positions,keys);axes[1,0].set_title('Core change / TRAIN SOURCE unit')
    task=a['paired_tasks']; keys=['regional','event']; means=np.array([task[k]['mean'] for k in keys]);lo=np.array([task[k]['interval_95'][0] for k in keys]);hi=np.array([task[k]['interval_95'][1] for k in keys])
    axes[1,1].scatter(np.arange(2),means);axes[1,1].vlines(np.arange(2),lo,hi);axes[1,1].axhline(0,color='black',lw=.5);axes[1,1].set_xticks(np.arange(2),keys);axes[1,1].set_title('Task score change / TRAIN unit')
    for ax in axes[0]:ax.set_xticks(x,fields);ax.grid(alpha=.2)
    for ax in axes[1]:ax.set_ylabel('Child minus SOURCE; negative is better');ax.grid(alpha=.2)
    save_figure(fig,directory,'field_core_task')


def controller_response(nomination, directory):
    """Show the measured nomination separately from subsequent learning evidence."""
    import numpy as np
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if (nomination.get('schema') != 'car_round6_measured_response_qualification_v1'
            or nomination.get('status') != 'completed'
            or nomination.get('panel_sample_count') != 800
            or nomination.get('dataset_split') != 'train'):
        raise ValueError('completed matched TRAIN800 response nomination required')
    fields = nomination['field_order']
    zero = np.asarray(nomination['panel_mean_excess_zero'], dtype=float)
    probe = np.asarray(nomination['panel_mean_excess_probe'], dtype=float)
    norms = np.asarray([nomination['positive_excess_norm_zero'],
                        nomination['positive_excess_norm_probe']], dtype=float)
    ratios = np.asarray([nomination['shape_finite_change'][key]['probe_fraction_of_zero_decrease']
                         for key in ('fixed', 'independent')], dtype=float)
    if (zero.shape != (len(fields),) or probe.shape != zero.shape
            or not all(np.isfinite(x).all() for x in (zero, probe, norms, ratios))):
        raise ValueError('finite fieldwise and query-response evidence required')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    x = np.arange(len(fields)); width = .36
    axes[0].bar(x-width/2, zero, width, label='Zero current feedback')
    axes[0].bar(x+width/2, probe, width, label='Measured PI probe')
    axes[0].axhline(0, color='black', linewidth=.7)
    axes[0].set(xticks=x, xticklabels=fields, ylabel='Pooled signed excess e',
                title='One mature AdamW action')
    axes[0].legend(fontsize=7)
    bars = axes[1].bar(['Zero current\nfeedback', 'Measured\nprobe'], norms)
    for bar, value in zip(bars, norms, strict=True):
        axes[1].text(bar.get_x()+bar.get_width()/2, value, f'{value:.4g}',
                     ha='center', va='bottom')
    axes[1].set(ylabel='norm((s/a) positive_part(mean e))',
                title='Remaining mean field violation')
    axes[1].set_ylim(0, max(norms)*1.2)
    axes[2].bar(['Fixed query', 'Independent query'], ratios)
    axes[2].axhline(1, color='black', linewidth=.7)
    axes[2].set(ylabel='Probe / zero-feedback Shape decrease',
                title='Measured local Shape tradeoff', ylim=(0, max(1.1, float(max(ratios))*1.1)))
    for axis in axes:
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
    save_figure(fig, directory, 'measured_controller_response')
    return ('Matched TRAIN800 mature-state actions nominate the already measured positive PI probe: '
            'pool signed field errors before taking their positive part. Both Shape decreases remain positive; '
            'the independent-query decrease is weaker under the probe. Actual saved Adam moments contain '
            'historical feedback. This is local nomination evidence, not a fresh-state trajectory or held-out efficacy result.')


def case_figures(a, selection, source_path, source, candidate_path, candidate, event, directory):
    import numpy as np
    import torch
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arm=a['arm']; bank={arm:(candidate_path,candidate)}; captions={}
    if selection['source_bank_sha256']!=sha(source_path) or selection.get('candidates_used') is not False or selection['created_unix']>=candidate['start_unix']:
        raise ValueError('visual cases must be frozen from SOURCE before child predictions')
    if len(selection['cases'])!=3: raise ValueError('three SOURCE-selected conditions required')
    for index,case in enumerate(selection['cases'],1):
        sid=case['sample_id']; ordinal=case['ordinal']
        source_row,children=rows_for_case(source_path,source,bank,ordinal,sid); child=children[arm]
        coords=source_row['coordinates_raw'].numpy();mass=source_row['native_masses'].reshape(-1).double().numpy();mask=(mass>0)&np.isfinite(coords[:,:2]).all(1)
        if not mask.any() or np.any(mass<0) or not np.isfinite(mass).all(): raise ValueError('invalid physical support')
        weights=mass[mask]/mass[mask].sum();scale=source_row['normalizer_scale'];offset=source_row['normalizer_offset']
        truth=(source_row['truth_model'][0]*scale+offset).numpy()
        src=(source_row['members_model'][:,0]*scale+offset).numpy(); pred=(child['members_model'][:,0]*scale+offset).numpy()
        for first in range(0, len(a['field_order']), 2):
            fields=a['field_order'][first:first+2]
            fig,axes=plt.subplots(len(fields),5,figsize=(16,3.4*len(fields)),constrained_layout=True,squeeze=False)
            for local,field in enumerate(fields):
                fi=first+local
                array=[truth[:,fi],src[:,:,fi].mean(0),pred[:,:,fi].mean(0)];low=min(v[mask].min() for v in array);high=max(v[mask].max() for v in array)
                residual=[array[1]-array[0],array[2]-array[0]];limit=max(float(np.max(np.abs(v[mask]))) for v in residual) or 1.
                for j,values in enumerate(array+residual):
                    ax=axes[local,j];img=ax.scatter(coords[mask,0],coords[mask,1],c=values[mask],s=.5,cmap='viridis' if j<3 else 'coolwarm',vmin=low if j<3 else -limit,vmax=high if j<3 else limit,rasterized=True)
                    ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
                    if j in (3,4):
                        rmse=np.sqrt(np.sum(weights*values[mask]**2));ax.text(.02,.02,f'RMSE {rmse:.4g}',transform=ax.transAxes,color='white',fontsize=10,bbox={'facecolor':'black','alpha':.7})
                    if j==2:fig.colorbar(img,ax=list(axes[local,:3]),shrink=.6)
                    if local==0:ax.set_title(['Reference','SOURCE mean','Repair mean','SOURCE residual','Repair residual'][j])
                    if j==0:ax.set_ylabel(field+' (dataset units)')
                fig.colorbar(img,ax=list(axes[local,3:]),shrink=.6)
            name=f'fields_case_{index}_part_{first//2+1}';save_figure(fig,directory,name)
            captions[name]=f'{sid}, fields {", ".join(fields)}: native positive-mass field reconstructions and residuals; residual badges are mass-weighted mean-field RMSE in dataset field units.'
        truth_t=torch.as_tensor(truth); src_t=torch.as_tensor(src); pred_t=torch.as_tensor(pred)
        truth_masks=[m.numpy() for m in event.masks(truth_t)]; src_masks=[m.numpy() for m in event.masks(src_t)]; pred_masks=[m.numpy() for m in event.masks(pred_t)]
        fig,axes=plt.subplots(3,5,figsize=(15,8),constrained_layout=True)
        for i,(label,masks) in enumerate((('Reference',truth_masks),('SOURCE members',src_masks),('Repair members',pred_masks))):
            for j in range(3):
                values=masks[j].astype(float) if i==0 else masks[j].mean(0)
                axes[i,j].scatter(coords[mask,0],coords[mask,1],c=values[mask],s=.5,cmap='viridis',vmin=0,vmax=1,rasterized=True)
            if i==0:
                values=truth_masks[-1].astype(float);prob=values
            else:
                joint=masks[-1]; mean_joint=joint.mean(0); values=np.where(truth_masks[-1],-1,0)*(1-mean_joint)+np.where(~truth_masks[-1],1,0)*mean_joint;prob=mean_joint
            axes[i,3].scatter(coords[mask,0],coords[mask,1],c=values[mask],s=.5,cmap='coolwarm',vmin=-1,vmax=1,rasterized=True)
            axes[i,4].scatter(coords[mask,0],coords[mask,1],c=prob[mask],s=.5,cmap='viridis',vmin=0,vmax=1,rasterized=True)
            axes[i,0].set_ylabel(label)
            for ax in axes[i]:ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
        for ax,title in zip(axes[0],('T support','CO support','CH4 support','Joint missing (-) / excess (+)','Joint member probability')):ax.set_title(title,fontsize=8)
        name=f'event_case_{index}';save_figure(fig,directory,name)
        captions[name]=f'{sid}: frozen unary event supports, signed joint missing/excess and independent-member joint-event probability on native positive-mass support.'
    return captions


def regional(a,directory):
    import numpy as np
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=a['per_condition_tasks'];fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for label in ('source','child'):
        r=rows[label]; truth=np.asarray([v['regional_truth'] for v in r]);pred=np.asarray([np.mean(v['regional_members']) for v in r]);area=np.asarray([v['event_target_area'] for v in r]);event=np.asarray([v['event_score'] for v in r])
        axes[0].scatter(truth,pred,s=12,alpha=.6,label=label);axes[1].scatter(truth,pred-truth,s=12,alpha=.6,label=label);axes[2].scatter(area,event,s=12,alpha=.6,label=label)
    lo=min(axes[0].get_xlim()[0],axes[0].get_ylim()[0]);hi=max(axes[0].get_xlim()[1],axes[0].get_ylim()[1]);axes[0].plot([lo,hi],[lo,hi],color='black');axes[1].axhline(0,color='black')
    for ax,title,xlab,ylab in zip(axes,('Regional covariance','Regional error','Joint event area fair CRPS'),('Reference covariance','Reference covariance','Reference area fraction'),('Mean member covariance','Prediction minus reference','Fair CRPS, area fraction')):
        ax.set(title=title,xlabel=xlab,ylabel=ylab);ax.grid(alpha=.2);ax.legend(fontsize=7)
    save_figure(fig,directory,'regional_and_event')


def produce(inputs_path,campaign,output):
    from scripts.posttraining.freeze_car_event import FrozenRedesignedEvent
    started=time.time(); inputs_path=inputs_path.resolve(strict=True);campaign=campaign.resolve(strict=True);output=output.resolve()
    if campaign/'reports' not in output.parents or output.exists(): raise ValueError('new campaign-owned report directory required')
    inputs=json.loads(inputs_path.read_text())
    if inputs.get('schema')!='car_round6_report_inputs_v1': raise ValueError('Round 6 report input manifest required')
    assessment=doc(inputs['assessment']);decision=doc(inputs['decision']);cost=doc(inputs['costs']);engineering=doc(inputs['engineering'])
    if (assessment.get('schema')!='car_round6_matched_validation_v1' or assessment.get('task_scope')!='reused development tasks at selected endpoint'
        or assessment.get('split')!='validation' or len(assessment.get('sample_ids',[]))!=100 or assessment.get('members')!=4
        or decision.get('assessment_sha256')!=inputs['assessment']['sha256'] or not decision.get('recommendation') or not decision.get('reason')):
        raise ValueError('terminal matched assessment and explicit bound decision required')
    arm=assessment['arm'];candidate_result=assessment['candidates'][arm]
    resumed_disposition = validate_model_decision(decision, candidate_result, arm)
    source_path=bound(inputs['source_bank']);source=json.loads(source_path.read_text());candidate_path=bound(inputs['candidate_bank']);candidate=json.loads(candidate_path.read_text())
    if (sha(source_path)!=assessment['source_bank_sha256'] or sha(candidate_path)!=assessment['bank_receipt_sha256']
        or source['sample_ids']!=assessment['sample_ids'] or candidate['sample_ids']!=assessment['sample_ids']
        or candidate['arm']!=arm or candidate['epoch']!=assessment['epoch'] or candidate['source_bank_sha256']!=sha(source_path)):
        raise ValueError('assessment and prediction bank identities differ')
    selection=doc(inputs['visual_selection']);event=FrozenRedesignedEvent(doc(inputs['event']))
    if selection['event']['sha256']!=inputs['event']['sha256']:raise ValueError('visual event freeze differs')
    history=lines(bound(inputs['history']));journal=lines(bound(inputs['pi_journal'])) if inputs.get('pi_journal') else []
    if [r.get('epoch') for r in history]!=list(range(1,assessment['epoch']+1)) or any(r.get('epoch_complete') is False for r in history):
        raise ValueError('complete epoch history through assessed checkpoint required')
    if (arm=='FFM_SSS_EpochCalibrated_PI_Repair' and not journal) or (journal and [r.get('step') for r in journal]!=list(range(1,12*assessment['epoch']+1))):
        raise ValueError('complete per-action PI journal required')
    coefficient_doc=doc(inputs['coefficients']);coefficients=coefficient_doc['coefficients']
    if not coefficients or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in coefficients.values()):raise ValueError('finite fixed prefit coefficients required')
    if not isinstance(engineering.get('summary'),str) or not engineering['summary'].strip():raise ValueError('engineering disposition summary required')
    output.mkdir(parents=True);figures=output/'figures';figures.mkdir()
    leaf_figures=learning(history,journal,assessment['field_order'],coefficients,figures)
    outcomes(assessment,figures);regional(assessment,figures)
    captions={'learning_and_action':'Parameter-direction norms, learning rate, pre-action signed violations, post-action multipliers, gradient norm ratio and actual optimizer-displacement products; translucent curves show recorded values and solid bars show ten-epoch means where applicable.',
        'family_history':'Complete raw sampled family histories in their respective declared representation units, with translucent epoch records and descriptive ten-epoch means.',
        'fixed_coefficients':'SOURCE-prefit coefficients remain fixed throughout the recorded optimization.',
        'field_core_task':'Native positive-mass field retention, member and ensemble errors and spread, with paired 95% condition-bootstrap intervals for complete State, Scale, Shape and endpoint regional and event score changes; lower score changes are better.',
        'regional_and_event':'Matched VALIDATION conditions: right-half mass-weighted CO–U1 covariance prediction and error, and memberwise joint-event area fair CRPS. These are frozen reused development tasks.'}
    if not journal:
        captions['learning_and_action']='Parameter-direction norms, learning rate and the last recorded batch constraint and multiplier at each epoch; actual optimizer-displacement products are unavailable for this run.'
    for name in leaf_figures: captions[name]='Complete raw sampled leaf histories in their declared representation units; translucent values are epoch records and solid bars are descriptive ten-epoch means.'
    response = doc(inputs['controller_response']) if inputs.get('controller_response') else None
    if response is not None:
        captions['measured_controller_response'] = controller_response(response, figures)
    captions.update(case_figures(assessment,selection,source_path,source,candidate_path,candidate,event,figures))
    f=candidate_result['fidelity']; scores=candidate_result['normalized_core_differences'];task=candidate_result['reused_tasks']
    body=['# Coherence Adaptive Repair: Round 6','',f"**Decision:** {decision['recommendation']}. {decision['reason']}",'',
        f"The raw {arm} model is compared with matched SOURCE on {len(assessment['sample_ids'])} development VALIDATION conditions, each with four independent members, at reporting epoch {assessment['epoch']}. TEST remains locked. No native reconstruction loss was added to the SSS update.",'',
        '## Field retention and coherence','',f"Signed field and sensor guard: {f['primary_pass']}; member-MSE guard: {f['legacy_member_pass']}. State, Scale and Shape normalized changes: "+', '.join(f'{k} {v:+.5g}' for k,v in scores.items())+'.','',
        f"Regional task improvement in frozen TRAIN units: {task['regional_normalized_improvement']:+.5g}; joint-event score change: {task['event_normalized_difference']:+.5g}. Positive regional improvement and negative event change favor the repair. These are reused development outcomes, not independent confirmation.",'',
        '| Field | Signed risk change | TRAIN allowance | Change / allowance | SOURCE sensor MSE | Repair sensor MSE | Sensor allowance |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    if resumed_disposition:
        body[4:4] = [f"**Scientific progress:** {decision['scientific_progress']}", '',
            f"The user-authorized resumption treats the original core/task percentage margins as reported references. Their combined replacement flag is {candidate_result['replacement_eligible']}; the operational recommendation above is an explicit measured judgment, with field, sensor and member safeguards retained.", '']
    for i,name in enumerate(assessment['field_order']):
        def fmt(v):return 'unavailable' if v is None else f'{v:.6g}'
        allowance=f['field_allowance'][i]
        body.append('| '+' | '.join([name,fmt(f['field_difference'][i]),fmt(allowance),fmt(f['field_difference'][i]/allowance if allowance else None),fmt(assessment['source']['sensor_mse'][i]),fmt(candidate_result['sensor_mse'][i]),fmt(f['sensor_allowance'][i])])+' |')
    body += ['', '| Field | SOURCE mean MSE | Repair mean MSE | SOURCE member MSE | Repair member MSE | SOURCE spread | Repair spread |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for i,name in enumerate(assessment['field_order']):
        body.append('| '+' | '.join([name,*[fmt(view['risk'][key][i]) for key in ('ensemble_mean_mse','member_mse','unbiased_member_variance') for view in (assessment['source'],candidate_result)]])+' |')
    if response is not None:
        body += ['', '## Controller nomination and learning evidence', '',
            f"The user-authorized resumption nominated the measured norm-balanced probe gamma = {float(response['gamma']):.8g} for a fresh SOURCE trajectory. The prior exact ten-epoch drift fit selected zero because the saved mature AdamW action already corrected faster than that target. The measured probe lowered the remaining pooled field-violation norm from {float(response['positive_excess_norm_zero']):.6g} to {float(response['positive_excess_norm_probe']):.6g}, with a weaker but positive independent-query Shape decrease. This supports the bounded learning test; it does not establish an optimal gain or model replacement.", '']
    if inputs.get('audit'):
        audit=doc(inputs['audit']); qualification=audit.get('controller_qualification',audit)
        gamma=qualification.get('gamma_star',qualification.get('calibration',{}).get('gamma_star'))
        if gamma is not None:
            body += ['', '## Controller diagnosis', '', f'The TRAIN-only controller uses delta = 1/12 reporting epoch, tau = 1 epoch, horizon = 10 epochs and measured gain gamma = {float(gamma):.6g}; its signed error is captured before the action and the combined PI increment is projected after the action.', '']
        for key,label in (('baseline_weighted_residual_squared','Zero-current-feedback weighted response residual squared'),('star_weighted_residual_squared','Calibrated weighted response residual squared'),('residual_norm_improvement','Measured residual norm improvement'),('replay_discrepancy_norm','Repeated numerical replay discrepancy')):
            if qualification.get(key) is not None: body += [f"{label}: {float(qualification[key]):.6g}.", '']
        if isinstance(audit.get('summary'),str):body += [audit['summary'],'']
    body+=['','## Engineering and cost','',engineering['summary'],'',cost.get('summary','Measured cost details follow; unavailable values are shown as unavailable.'),'', '| Measurement | Value |','| --- | ---: |']
    measurements=cost.get('measurements')
    if not isinstance(measurements,dict) or not measurements:raise ValueError('measured cost and memory mapping required')
    for name,value in measurements.items():
        if value is None: shown='unavailable'
        elif isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value):shown=f'{value:.6g}'
        else: raise ValueError('cost measurements must be finite numbers or null')
        body.append(f'| {name} | {shown} |')
    body+=['','## Figure index','']+[f'- [Figure {name.replace("_"," ")}](#figure-{name.replace("_","-")})' for name in captions]+['']
    for name,caption in captions.items():body += [f'## Figure {name.replace("_"," ")}','',f'![{caption}](figures/{name}.png)','',f'{caption} [PDF master](figures/{name}.pdf)','']
    (output/'report.md').write_text('\n'.join(body),encoding='utf-8')
    pages=render(output,captions)
    receipt={'schema':'car_round6_report_v1','status':'rendered_pending_visual_inspection','pages':pages,'input_manifest_sha256':sha(inputs_path),
        'report_md_sha256':sha(output/'report.md'),'report_pdf_sha256':sha(output/'report.pdf'),
        'figures':{name:{'png_sha256':sha(figures/f'{name}.png'),'pdf_sha256':sha(figures/f'{name}.pdf')} for name in captions},'elapsed_seconds':time.time()-started}
    (output/'report_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    return receipt


def preparation_figures(audit, diagnosis, profile, history, directory):
    import numpy as np
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    captions={}
    fields=audit['field_order'];q=audit['controller_qualification'];x=np.arange(len(fields))
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    axes[0].bar(x-.2,q['baseline_drift'],.4,label='zero-current-feedback drift');axes[0].bar(x+.2,q['target_drift'],.4,label='H=10 target drift')
    axes[0].set_xticks(x,fields);axes[0].set_ylabel('normalized signed error / reporting epoch');axes[0].legend(fontsize=7);axes[0].set_title('Finite paired field response')
    axes[1].bar(x,q['baseline_drift_over_target']);axes[1].set_xticks(x,fields);axes[1].axhline(1,color='black');axes[1].set_ylabel('drift magnitude / target magnitude');axes[1].set_title('Observed recovery versus target')
    for ax in axes:ax.grid(alpha=.2)
    save_figure(fig,directory,'controller_response')
    captions['controller_response']='TRAIN800 mature one-step paired response with saved Adam moments: zero-current-feedback drift and the H=10 target, and the corresponding magnitudes. This is a local diagnostic rather than a trained PI trajectory.'
    ev=audit['action_evidence']; family=audit['family_order']; products=ev['zero_family_actual_products']['mean']; finite=[ev['zero_fixed_query_shape_changes']['mean'],ev['zero_independent_query_shape_changes']['mean']]
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    axes[0].bar(np.arange(len(family)),products);axes[0].set_xticks(np.arange(len(family)),['State','Scale','Shape']);axes[0].set_ylabel('gradient dot AdamW displacement');axes[0].set_title('First-order action products')
    axes[1].bar(np.arange(2),finite);axes[1].set_xticks(np.arange(2),['Same query','Independent query']);axes[1].set_ylabel('post-minus-pre Shape score');axes[1].set_title('Finite Shape response')
    for ax in axes:ax.axhline(0,color='black',lw=.7);ax.grid(alpha=.2)
    save_figure(fig,directory,'action_and_shape')
    captions['action_and_shape']='Sample-weighted mature TRAIN800 action evidence: family-gradient products with the actual zero-current-feedback AdamW displacement and finite Shape changes on matched versus independent queries. Products are first-order diagnostics; only Shape has finite score reassessment here.'
    var=audit['source_shape_variation'];comp=audit['shape_components']['components'];names=list(comp)
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    axes[0].bar(np.arange(3),[var['condition_variance'],var['paired_member_half_mean_square'],var['paired_query_half_mean_square']]);axes[0].set_xticks(np.arange(3),['Condition','Member pair','Query pair'],rotation=20);axes[0].set_ylabel('variance or paired half mean-square');axes[0].set_title('Cached SOURCE J4 Shape variation')
    axes[1].barh(np.arange(len(names)),[comp[n]['parameter_gradient_rms'] for n in names]);axes[1].set_yticks(np.arange(len(names)),names,fontsize=7);axes[1].set_xlabel('parameter-gradient RMS');axes[1].set_title('Mature one-batch Shape components')
    save_figure(fig,directory,'shape_variability')
    captions['shape_variability']='Cached SOURCE J4 variation across 800 TRAIN conditions and paired member/query draws; semantic-component gradient RMS comes from one separate real mature B64 fixture and is not a population variance estimate.'
    support=audit['source_support'];fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,kind,title in ((axes[0],'member_mse','SOURCE member MSE'),(axes[1],'risk','SOURCE signed mean risk')):
        ax.bar(x-.2,support[kind+'_native'],.4,label='native physical support');ax.bar(x+.2,support[kind+'_selected'],.4,label='selected support');ax.set_xticks(x,fields);ax.set_title(title);ax.set_ylabel('model-field units squared');ax.axhline(0,color='black',lw=.5);ax.legend(fontsize=7)
    save_figure(fig,directory,'source_support')
    captions['source_support']='Cached TRAIN800 SOURCE J4 native and selected-support risks are in model-field units squared and use the same predictions and declared physical measures; these are support comparisons, not an experiment on support-induced training bias.'
    timing=profile['epoch_timing'];scopes=profile['scopes'];fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    axes[0].bar(['train','epoch wall'],[timing['train_seconds'],timing['wall_seconds']]);axes[0].set_ylabel('seconds');axes[0].set_title('Instrumented complete epoch')
    names=['wide_pseudoinverse','shape_pairing_and_transfers','family_and_constraint_backward'];axes[1].barh(names,[scopes[k]['seconds'] for k in names]);axes[1].set_xlabel('seconds');axes[1].set_title('Inclusive overlapping scopes')
    save_figure(fig,directory,'engineering_profile')
    captions['engineering_profile']='One synchronized 12-action qualification epoch and selected inclusive profiling scopes. Scope durations overlap, include instrumentation effects, and do not establish an uninstrumented speedup.'
    epochs=[r['epoch'] for r in history];family_keys=[k for k in history[0] if k.startswith('coherence_family/') and k.endswith('/raw')];leaf_keys=[k for k in history[0] if k.startswith('coherence_component/') and k.endswith('/raw')]
    for label,keys in [('family',family_keys),('leaf',leaf_keys)]:
        if not keys:raise ValueError(f'historical {label} history absent')
        for start in range(0,len(keys),4):
            group=keys[start:start+4];fig,axes=plt.subplots(len(group),1,figsize=(11,2.4*len(group)),constrained_layout=True,squeeze=False)
            for ax,key in zip(axes.flat,group):series(ax,epochs,[r[key] for r in history],key);ax.set(title=key,xlabel='Historical R3 reporting epoch');ax.grid(alpha=.2)
            name=f'historical_{label}_{start//4+1}';save_figure(fig,directory,name)
            captions[name]='Historical Round 3 SourceScaledRisk complete 500-epoch sampled history; translucent epoch values and descriptive ten-epoch means are shown in each raw family or leaf unit. This is not Round 6 training.'
    fig,axes=plt.subplots(3,1,figsize=(11,8),constrained_layout=True)
    series(axes[0],epochs,[r['optimizer_lr'] for r in history],'learning rate');axes[0].set_title('Historical learning rate')
    for field in fields:axes[1].plot(epochs,[r['field_violation'][field] for r in history],label=field,alpha=.7)
    for i,field in enumerate(fields):axes[2].plot(epochs,[r['policy_state']['multipliers'][i] for r in history],label=field,alpha=.7)
    axes[1].set_title('Historical last-batch signed constraint');axes[2].set_title('Historical last-batch multiplier')
    for ax in axes:ax.set_xlabel('Historical R3 reporting epoch');ax.legend(fontsize=7);ax.grid(alpha=.2)
    save_figure(fig,directory,'historical_constraints')
    captions['historical_constraints']='Historical Round 3 SourceScaledRisk 500-epoch learning rate and last-batch field constraints and multipliers; the nested values are not epoch means and do not describe a Round 6 run.'
    return captions


def produce_preparation(inputs_path,campaign,output):
    from phycoflow_reconstruction.evaluation.adaptive_campaign import digest
    import resource
    start=time.time();inputs_path=inputs_path.resolve(strict=True);campaign=campaign.resolve(strict=True);output=output.resolve()
    own_cpu=resource.getrusage(resource.RUSAGE_SELF);child_cpu=resource.getrusage(resource.RUSAGE_CHILDREN)
    if campaign/'reports' not in output.parents or output.exists():raise ValueError('new campaign-owned report directory required')
    inputs=json.loads(inputs_path.read_text());audit=doc(inputs['decision']);history_audit=doc(inputs['history_audit']);profile=doc(inputs['profile']);engineering=doc(inputs['engineering']);batched=doc(inputs['batched_vjp']);cost=doc(inputs['costs'])
    if audit.get('schema')!='car_round6_joint_audit_decision_v1' or audit.get('status')!='completed' or audit.get('decision')!='unqualified_pi_retain_source' or audit['controller_qualification'].get('qualified') is not False:
        raise ValueError('completed terminal unqualified-PI joint decision required')
    required={'diagnosis','parity','shape','support','pairing','variance','prefit'}
    if not required.issubset(audit['evidence']):raise ValueError('complete bound joint evidence required')
    evidence={name:doc(audit['evidence'][name]) for name in required}
    for name,ref in audit['evidence'].items():
        if name not in required:bound(ref)
    if any(x.get('status')!='completed' for x in evidence.values()):raise ValueError('joint audit evidence incomplete')
    if any(evidence[k].get('sample_count')!=800 for k in ('diagnosis','parity','support','pairing','variance')):raise ValueError('complete TRAIN800 evidence required')
    if (cost.get('status')!='completed' or not isinstance(cost.get('summary'),str) or not isinstance(cost.get('measurements'),dict) or not cost['measurements']):raise ValueError('completed cost summary and measurements required')
    ledger=doc(cost['ledger'])
    if ledger.get('active'):raise ValueError('cannot close preparation while GPU leases are active')
    if any(not isinstance(row.get('seconds'),(int,float)) or not math.isfinite(row['seconds']) or row['seconds']<0 for row in ledger.get('closed',[])):raise ValueError('invalid GPU lease duration')
    ledger_seconds=sum(row['seconds'] for row in ledger.get('closed',[]))
    if not math.isclose(ledger_seconds,cost['measurements'].get('occupied_gpu_seconds',float('nan')),rel_tol=1e-9,abs_tol=1e-6):raise ValueError('GPU ledger and total occupied seconds differ')
    history_path=bound(inputs['historical_history']);history=lines(history_path)
    if sha(history_path)!=history_audit['history_sha256'] or [r.get('epoch') for r in history]!=list(range(1,501)) or not history_audit['coefficients_constant']:
        raise ValueError('complete hash-bound historical 500-epoch history with fixed coefficients required')
    if profile.get('status')!='completed' or engineering.get('schema')!='car_round6_engineering_decision_v1' or batched.get('status') not in ('qualified','rejected'):
        raise ValueError('completed engineering evidence required')
    if engineering['complete_epoch_profile']['receipt']['sha256'] != inputs['profile']['sha256'] or engineering['batched_family_vjp']['receipt']['sha256'] != inputs['batched_vjp']['sha256']:raise ValueError('engineering decision and profiling receipts differ')
    output.mkdir(parents=True);figures=output/'figures';figures.mkdir()
    captions=preparation_figures(audit,evidence['diagnosis'],profile,history,figures);q=audit['controller_qualification'];shape=audit['shape_components'];support=audit['source_support']
    body=['# Coherence Adaptive Repair: Round 6 preparation close','',f"**Decision:** {audit['disposition']}",'',
        'The proposed PI repair did not qualify. SOURCE/C8 is retained. The 20-epoch pilot and conditional 100/500-epoch candidate training, held-out candidate assessment, and replacement decision were not performed. TEST remained locked. The one-epoch instrumented computation profile is an engineering qualification, not a Round 6 learning experiment.','',
        '## Mathematical and empirical diagnosis','',
        f"The audited method uses SSS-only primary objectives. Three fresh full-parameter State, Scale and Shape gradients enter the pinned TorchJD ConFIG operation; the pre-clipping direction is d = d_C/C0 + sum_f lambda_f grad(e_f), followed by clipping and AdamW. The fixed SOURCE objective scale is C0 = {audit['constraint_reference']['objective_scale']!r}. The original native RF/MSE loss is not replayed. The mature audit uses J2 independent members, four Euler steps, batch64, Q4096 and 12 actions per reporting epoch.",'',
        'Signed independent-member mean risk is mean-field squared error minus the finite-member spread correction. It estimates conditional-mean risk without equating that quantity to member MSE, which is the average error of individual predictions. A finite-sample signed estimate can be negative.','',
        'The signed normalized error is e_f = (R_f - R_0,f - a_f) / s_f. The candidate would update z_t = nu z_(t-1) + (1 - nu)e_t and lambda_(t+1) = projection_positive(lambda_t + k_I delta e_t + H k_I (z_t - z_(t-1))), with delta = 1/12 epoch, tau = 1 epoch, nu = exp(-delta/tau), H = 10 epochs and k_I = 0.0012 gamma per epoch. The current multiplier acts before this update.','',
        'For paired pre-action, zero-feedback and probe errors, b = (e_zero - e_pre)/delta, r = (e_probe - e_zero)/(gamma_probe delta), t_f = -max(e_pre,f, 0)/H, and W_ff = s_f/a_f. The predeclared fit is gamma_star = clip(dot(Wr, W(t-b)) / ||Wr||², 0, gamma_probe); the production combined-VJP replay independently remeasures r at the same probe. Workplan section 3 specifies closing preparation and retaining SOURCE when no justified correction remains.','',
        f"The predeclared weighted response fit selected gamma = {q['gamma_star']:.6g} (unconstrained fit {q['unclipped_gain']:.6g}); the independent production combined-VJP probe at gamma = {q['gamma_probe']:.6g} gave unclipped fit {q['production_unclipped_gain']:.6g} and projected gain {q['production_recalculated_gain']:.6g}. Zero-current-feedback weighted residual norm was {q['zero_weighted_residual_norm']:.6g}; production probe residual norm was {q['production_probe_weighted_residual_norm']:.6g}; production-versus-separate response discrepancy was {q['production_vs_separate_response_norm']:.6g} and zero-action replay discrepancy was {q['zero_replay_discrepancy_norm']:.6g}.",'',
        'The mature zero-current-feedback action already reduced each field excess much faster than the prescribed target, so a positive probe moved the measured response farther from that particular target. The saved mature Adam moments include previous training feedback; zero current feedback is not an independently trained integral-control baseline. A zero fitted gain does not establish failure of Lagrangian methods or efficacy of an alternative.','',
        f"The frozen SOURCE prefit matched the saved mature checkpoint and logged coefficients stayed constant: {audit['source_prefit_identity']['logged_coefficients_constant']}. A proposed epoch-10 coefficient switch is therefore not supported by these records. The SOURCE support and pairing checks used {evidence['support']['sample_count']} matched TRAIN conditions; sensor-pair and member-seed lists matched {support['sensor_pairs_match_count']} and {support['member_seed_list_match_count']} conditions respectively. {evidence['pairing']['interpretation']}",'',
        f"The Shape semantic fixture reconstructed the sampled family value to absolute difference {shape['absolute_difference']:.6g} on one mature B64 graph. Its component gradients and cached SOURCE J4 member/query variation describe different quantities. State and Scale have first-order products only; finite paired reassessment was performed for Shape. No evidence here separates Adam moment carryover from preconditioning.",'',
        '## Engineering and cost','',engineering['decision']+'. '+engineering['normalized_gram']['reason']+' '+engineering['context_reuse']['decision']+'. '+engineering['context_reuse']['limits']+' '+engineering['grouped_queries']['reason'],'',
        f"The mature batched-family-VJP fixture was {batched['status']}; its measured peak allocations were {batched['separate']['peak_allocated_bytes']} bytes separately and {batched['batched']['peak_allocated_bytes']} bytes batched. It was a one-batch compatibility check, not a complete-epoch speed qualification. No new computation route is adopted solely from that check.",'',cost['summary'],'','| Measurement | Value |','| --- | ---: |']
    import numpy as np
    parity=evidence['parity']; batches=parity['per_batch']; counts=np.asarray([len(r['sample_ids']) for r in batches],dtype=float)
    if counts.sum()!=800 or list(parity['field_order'])!=audit['field_order'] or any(len(r['pre'])!=len(audit['field_order']) for r in batches):
        raise ValueError('production probe pre-action field coverage differs')
    pre=np.average(np.asarray([r['pre'] for r in batches],dtype=float),axis=0,weights=counts)
    baseline=np.asarray(q['baseline_drift'],dtype=float);zero=pre+baseline/12;probe=parity['combined_post_mean'];target=q['target_drift'];ref=audit['constraint_reference']
    field_table=['## Fieldwise controller response','','R0, a and s are in model-field units squared. Errors are dimensionless and drifts are per reporting epoch; the pre-action mean is condition-weighted, and the zero-post value is reconstructed exactly from the saved baseline drift.','',
        '| Field | R0 | a | s | e pre | e zero post | e probe post | b | target t |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for i,name in enumerate(audit['field_order']):
        field_table.append('| '+' | '.join([name,*[f'{float(v):.6g}' for v in (ref['reference_risks'][i],ref['absolute_allowance'][i],ref['risk_scales'][i],pre[i],zero[i],probe[i],baseline[i],target[i])]])+' |')
    field_table.append('')
    position=body.index('## Engineering and cost');body[position:position]=field_table
    for name,value in cost['measurements'].items():
        if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)):raise ValueError('cost values must be finite numbers or null')
        body.append(f'| {name} | {"unavailable" if value is None else format(value,".6g")} |')
    body += ['','## Figure index','']+[f'- [Figure {name.replace("_"," ")}](#figure-{name.replace("_","-")})' for name in captions]+['']
    for name,caption in captions.items():body += [f'## Figure {name.replace("_"," ")}','',f'![{caption}](figures/{name}.png)','',f'{caption} [PDF master](figures/{name}.pdf)','']
    (output/'report.md').write_text('\n'.join(body),encoding='utf-8');pages=render(output,captions)
    bindings={key:ref for key,ref in inputs.items() if isinstance(ref,dict) and 'sha256' in ref};bindings['decision_evidence']=audit['evidence'];bindings['generator']={'path':str(Path(__file__).resolve()),'sha256':sha(Path(__file__))}
    (output/'evidence_index.json').write_text(json.dumps({'schema':'car_round6_preparation_evidence_v1','bindings':bindings,'test_locked':True},indent=2,allow_nan=False)+'\n')
    own_after=resource.getrusage(resource.RUSAGE_SELF);child_after=resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt={'schema':'car_round6_preparation_report_v1','status':'rendered_pending_visual_inspection','pages':pages,'input_manifest_sha256':sha(inputs_path),'report_md_sha256':sha(output/'report.md'),'report_pdf_sha256':sha(output/'report.pdf'),'evidence_index_sha256':sha(output/'evidence_index.json'),'figures':{name:{'png_sha256':sha(figures/f'{name}.png'),'pdf_sha256':sha(figures/f'{name}.pdf')} for name in captions},'elapsed_seconds':time.time()-start,
        'cpu_user_seconds':own_after.ru_utime-own_cpu.ru_utime+child_after.ru_utime-child_cpu.ru_utime,
        'cpu_system_seconds':own_after.ru_stime-own_cpu.ru_stime+child_after.ru_stime-child_cpu.ru_stime,
        'cost_scope':'CPU-only report build including completed child processes; separate from GPU lease occupation'}
    (output/'report_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');return receipt

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();kind=json.loads(args.inputs.read_text()).get('schema'); function=produce_preparation if kind=='car_round6_preparation_report_inputs_v1' else produce;print(json.dumps(function(args.inputs,args.campaign,args.output),indent=2))

if __name__=='__main__':main()
