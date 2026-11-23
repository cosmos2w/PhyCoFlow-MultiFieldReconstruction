#!/usr/bin/env python3
"""Render a terminal Round 5 report exclusively from hash-bound saved evidence.

Manifest schema car_round5_report_inputs_v1: decision, costs, engineering and
(for model outcomes) assessment, source_bank, visual_selection, regional_task,
historical_history, histories[arm] and candidate_banks[arm] are {path,sha256} bindings. Bank
bindings point to their receipt.json; history bindings point to complete JSONL.
Decision supplies scope (model or engineering), recommendation, reason, and for
model scope assessment_sha256. Costs schema car_round5_costs_v1 supplies
training[arm] with seconds_per_epoch, occupied_gpu_seconds, update_work_seconds,
data_wait_and_loop_overhead_seconds, callback_and_reporting_seconds (nullable
when genuinely unavailable), assessment {gpu_seconds,cpu_seconds,wall_seconds,
queue_tail_seconds}, and campaign_gpu_seconds including failed attempts. Every
cost is measured or null; never replace missing measurements by zero. Engineering
supplies a human-readable summary and may bind query_gradient_review and
query_actions for the optional SOURCE query-engineering figure. Visual selection supplies source_bank_sha256,
created_unix, and cases [{sample_id, ordinal}]; it must predate candidate banks.
For differing terminal ages, assessment.epoch_by_arm maps every candidate arm
to its assessed integer epoch; when omitted, assessment.epoch applies to all.
The frozen regional task is the existing car_round2_reserved_task_v1 artifact.
No training, prediction generation, model evaluation, or TEST access occurs.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import math
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

ARMS = ('Native_GT_Continuation', 'Native_GT_SSS_Safeguarded_Continuation')
LABELS = {ARMS[0]:'Native continuation', ARMS[1]:'Native + SSS safeguard'}
FAMILY = {'State':'global_distribution', 'Scale':'multiscale_spectrum', 'Shape':'topology'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()


def bound(ref):
    if not isinstance(ref,dict) or set(('path','sha256')) - ref.keys():
        raise ValueError('required hash-bound evidence is missing')
    path=Path(ref['path']).resolve(strict=True)
    if sha(path)!=ref['sha256']: raise ValueError(f'evidence changed: {path}')
    return path


def doc(ref): return json.loads(bound(ref).read_text())


def number(x):
    if x is None: return 'unavailable'
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):
        raise ValueError('nonfinite or nonnumeric measurement')
    return f'{x:.5g}'


def save_figure(fig,directory,name):
    import matplotlib.pyplot as plt
    fig.savefig(directory/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(directory/f'{name}.png',bbox_inches='tight',dpi=150)
    plt.close(fig)


QUERY_CAPTION = ('Finite query replay at fixed SOURCE on TRAIN800: the parameter-gradient comparison uses four paired query seeds with matched observations, native draws and J2 endpoints; the finite-action comparison uses 26 actions from 13 batches at two learning rates. Grouped queries reduce parameter-gradient variance and improve independent-query behavior, but their lower fixed-query median decrease fails the predeclared no-worse gate. Shared queries are retained; full attempted-epoch cost was not measured after that failure. All 26 actions were accepted in each query law; no confidence intervals are inferred.')


def query_engineering(gradient, actions, figdir):
    """Plot the completed, matched SOURCE engineering review without model work."""
    import numpy as np
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if (gradient.get('schema') != 'car_round5_estimator_decision_v1'
            or actions.get('schema') != 'car_round5_query_finite_action_comparison_v1'
            or actions.get('status') != 'completed' or actions.get('sample_count') != 800
            or actions.get('matched_inputs') is not True or actions.get('finite_action_no_worse') is not False
            or actions.get('gates',{}).get('same_query_median_no_worse') is not False
            or gradient.get('query_law') != 'shared'):
        raise ValueError('completed matched TRAIN800 query review with the declared shared-query disposition required')
    stats=actions['statistics'];shared=stats['shared'];grouped=stats['grouped']
    if any(stats[name]['proposals'] != 26 or stats[name]['accepted_nonzero'] != 26 for name in ('shared','grouped')):
        raise ValueError('the figure requires the completed 26 paired accepted actions per query law')
    ratios=[float(gradient['shape_parameter_gradient']['centered_variance_ratio']),
            float(gradient['scalar_parameter_gradient']['centered_variance_ratio'])]
    medians=[[float(stats[name][key]) for name in ('shared','grouped')]
             for key in ('median_same_query_decrease','median_independent_query_decrease')]
    counts=[[int(stats[name][key]) for name in ('shared','grouped')]
            for key in ('descending_beyond_precision','independent_descending_beyond_precision')]
    if (not all(np.isfinite(r) and r>=0 for r in ratios)
            or not all(np.isfinite(v) for pair in medians for v in pair)
            or any(not 0<=v<=26 for pair in counts for v in pair)):
        raise ValueError('invalid recorded query-comparison metrics')
    fig,axes=plt.subplots(1,3,figsize=(16.5,5.5),constrained_layout=True)
    color_shared='#315b86';color_grouped='#c16730';width=.34;x=np.arange(2)
    def paired(axis, first, second, fmt, offset):
        for j,(values,color,hatch,label) in enumerate(((first,color_shared,'','Shared'),(second,color_grouped,'///','8 groups'))):
            bars=axis.bar(x+(j-.5)*width,values,width,color=color,edgecolor='#24303b',hatch=hatch,linewidth=.8,label=label)
            for bar,value in zip(bars,values,strict=True):
                axis.text(bar.get_x()+bar.get_width()/2,bar.get_height()+offset,fmt(value),ha='center',va='bottom',fontsize=10)
    paired(axes[0],[1.,1.],ratios,lambda v:f'{v:.5f}',.035)
    axes[0].axhline(.8,color='#8b2727',linestyle='--',linewidth=1.5,label='0.8 variance gate')
    axes[0].set_xticks(x,['Shape','Direct scalar']);axes[0].set_ylim(0,1.24)
    axes[0].set(ylabel='Centered gradient variance / shared',title='Parameter-gradient variation\n4 paired query seeds')
    paired(axes[1],[pair[0] for pair in medians],[pair[1] for pair in medians],lambda v:f'{v:.8f}',.0022)
    axes[1].set_xticks(x,['Fixed query','Independent query']);axes[1].set_ylim(0,max(v for pair in medians for v in pair)*1.30)
    axes[1].set(ylabel='Median scalar decrease',title='Finite-action median decrease\n26 actions: 13 batches × 2 LRs')
    paired(axes[2],[pair[0] for pair in counts],[pair[1] for pair in counts],lambda v:f'{int(v)}/26',.6)
    axes[2].set_xticks(x,['Fixed query','Independent query']);axes[2].set_ylim(0,31)
    axes[2].set(ylabel='Actions descending beyond precision',title='Strict finite-action descent\n26 actions: 13 batches × 2 LRs')
    for ax in axes:
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.tick_params(axis='both',labelsize=10)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.12),ncol=3,fontsize=10)
    fig.suptitle('SOURCE query engineering · TRAIN800',fontsize=15,fontweight='bold')
    save_figure(fig,figdir,'query_engineering')


def render_query_only(inputs_path,campaign,output):
    import time
    from PIL import Image
    started=time.time();inputs_path=inputs_path.resolve(strict=True);campaign=campaign.resolve(strict=True);output=output.resolve()
    if campaign/'assessment'/'engineering' not in output.parents:
        raise ValueError('standalone figure output must be inside campaign assessment/engineering')
    inputs=json.loads(inputs_path.read_text())
    gradient=doc(inputs['query_gradient_review']);actions=doc(inputs['query_actions'])
    output.mkdir(parents=True,exist_ok=True)
    names=[output/'query_engineering.png',output/'query_engineering.pdf',output/'query_engineering_receipt.json']
    if any(p.exists() for p in names):raise FileExistsError('standalone engineering figure already exists')
    query_engineering(gradient,actions,output)
    with Image.open(names[0]) as im:im.verify()
    subprocess.run(['pdfinfo',str(names[1])],check=True,capture_output=True,text=True)
    receipt={'schema':'car_round5_query_figure_v1','status':'rendered_pending_visual_inspection',
        'inputs':{'query_gradient_review':inputs['query_gradient_review'],'query_actions':inputs['query_actions']},
        'input_manifest_sha256':sha(inputs_path),'renderer_sha256':sha(Path(__file__)),
        'figure_png_sha256':sha(names[0]),'figure_pdf_sha256':sha(names[1]),
        'caption':QUERY_CAPTION,'wall_seconds':time.time()-started}
    names[2].write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    return receipt


def panel_coverage(assessment):
    population=assessment.get('population')
    ids=assessment.get('sample_ids',[])
    if type(population) is not int or population<1 or len(ids)!=100 or len(set(ids))!=100 or population<len(ids):
        raise ValueError('VALIDATION100 requires 100 unique matched IDs and a positive declared population')
    return (f"Matched VALIDATION100 comprises {len(ids)} of {population} validation conditions "
            f"({100*len(ids)/population:.1f}%); each prediction uses J4 members. This previously accessed "
            'same-trajectory validation panel is used for development selection and is not independent confirmation.')


def assessed_ages(assessment):
    """Resolve terminal learning ages without implying that arms are matched."""
    candidates=assessment['candidates']
    epochs=assessment.get('epoch_by_arm')
    if epochs is None:
        epochs={arm:assessment.get('epoch') for arm in candidates}
    if not isinstance(epochs,dict) or set(epochs)!=set(candidates):
        raise ValueError('epoch_by_arm must name each assessed arm exactly once')
    if any(type(epoch) is not int or epoch<1 for epoch in epochs.values()):
        raise ValueError('assessed arm epochs must be positive integers')
    return epochs


def methods_lines(freeze,inputs,arms):
    import yaml
    if freeze.get('schema')!='car_round5_freeze_v1':
        raise ValueError('frozen Round 5 scientific contract required')
    fitted=doc(freeze['units'])
    if fitted.get('status')!='completed' or fitted.get('sample_count')!=800:
        raise ValueError('completed SOURCE TRAIN800 units required for methods')
    units=fitted['units']
    if set(units)!={'native','state','scale','shape'} or any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in units.values()):
        raise ValueError('invalid frozen direct-objective units')
    for arm in arms:
        cfg=yaml.safe_load(bound(freeze['arms'][arm]).read_text())
        opt=cfg['optimization']; direct=opt['direct_supervision']
        if (direct.get('units')!=units or direct.get('mode')!=('native' if arm==ARMS[0] else 'safeguarded')
                or cfg['source']['weight_view']!='ema_materialized' or opt['batch_size']!=64 or opt['steps_per_epoch']!=12):
            raise ValueError(f'{arm}: frozen native objective or training exposure mismatch')
        manifest_path=bound(inputs['histories'][arm]).parent.parent/'run_manifest.json'
        if json.loads(manifest_path.read_text()).get('effective_child_ema_enabled') is not False:
            raise ValueError(f'{arm}: effective child EMA state is unavailable or enabled')
    values=', '.join(f'{key}={number(units[key])}' for key in ('native','state','scale','shape'))
    return ['## Method and training exposure','',
        'Both arms start from the EMA-materialized SOURCE with child EMA disabled. Both use the original native bridge-velocity objective divided by the same fixed native unit. The safeguarded candidate adds (State / u_state + Scale / u_scale + Shape / u_shape) / 3, accumulated in the order native, State, Scale, Shape, and applies a paired local finite field-action filter.', '',
        f'The fixed SOURCE TRAIN800 units are {values}. The native unit is the mean native loss; the three SSS units are means of absolute signed per-condition scores using fresh observations. Each reporting epoch contains 12 attempted batches of 64 conditions, or 768 attempted visits. Candidate training uses J2 endpoints on selected Q4096 support; assessment uses the native VALIDATION100 panel and J4 members.', '']


def age_and_exposure_lines(rows, ages):
    lines=['## Assessed learning ages and exposure','',
           '| Path | Assessed epoch | Attempted condition visits | Accepted objective-bearing visits |',
           '| --- | ---: | ---: | ---: |']
    for arm,history in rows.items():
        totals=[]
        for key in ('attempted_condition_visits','accepted_objective_condition_visits'):
            values=[row.get(key) for row in history]
            totals.append(str(sum(values)) if all(type(v) is int and v>=0 for v in values) else 'unavailable')
        lines.append('| '+' | '.join([LABELS[arm],str(ages[arm]),*totals])+' |')
    if len(set(ages.values()))>1:
        lines+=['','The terminal outputs use the same condition and member panel at different learning ages and exposures. The terminal comparison does not establish matched-age superiority.']
    else:
        lines+=['','The terminal outputs use the same condition and member panel and share the same assessed epoch; recorded attempted and accepted exposure is shown separately.']
    return lines+['']


def histories(refs, assessment):
    result={}
    ages=assessed_ages(assessment)
    for arm in assessment['candidates']:
        if arm not in refs: raise ValueError(f'missing complete history for {arm}')
        rows=[json.loads(x) for x in bound(refs[arm]).read_text().splitlines() if x.strip()]
        if not rows or [r.get('epoch') for r in rows] != list(range(1,len(rows)+1)):
            raise ValueError(f'{arm}: complete ordered epoch history required')
        if len(rows)!=ages[arm] or any(r.get('epoch_complete') is False for r in rows):
            raise ValueError(f'{arm}: history must end at its assessed epoch')
        result[arm]=rows
    return result


def learning(rows,figdir,historical=None):
    import numpy as np
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(11,11),constrained_layout=True)
    panels=[('native_data_loss','Native flow loss'),
        *[(f'coherence_family/{key}/raw',name+' sampled score') for name,key in FAMILY.items()],
        ('optimizer_lr','Learning rate'),('update_accepted_fraction','Accepted action fraction'),
        ('action_dot','Actual action directional product')]
    for axis,(key,title) in zip(axes.flat,panels):
        for arm,history in rows.items():
            # Native control has no sampled SSS and no finite-action filter.
            if arm==ARMS[0] and (key.startswith('coherence_family/') or key=='action_dot'): continue
            if arm==ARMS[0] and key=='update_accepted_fraction' and all(r.get(key) is None for r in history): continue
            vals=[r.get(key) for r in history]
            if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in vals):
                raise ValueError(f'{arm}: missing required history column {key}')
            x=np.arange(1,len(vals)+1); y=np.asarray(vals,dtype=float)
            color={ARMS[0]:'#1f77b4',ARMS[1]:'#d56b2d'}[arm]
            axis.plot(x,y,alpha=.25,linewidth=.8,color=color)
            for start in range(0,len(y),10):
                stop=min(start+10,len(y))
                axis.plot([start+1,stop],[y[start:stop].mean()]*2,linewidth=1.8,color=color,
                    label=LABELS[arm] if start==0 else None)
        if historical is not None and (key.startswith('coherence_family/') or key == 'optimizer_lr'):
            length=max(len(history) for history in rows.values())
            if len(historical)<length: raise ValueError('historical history does not cover the common learning window')
            values=[r.get(key) for r in historical[:length]]
            if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values):
                raise ValueError(f'historical ConFIG: missing {key}')
            x=np.arange(1,length+1);y=np.asarray(values,dtype=float)
            axis.plot(x,y,alpha=.25,linewidth=.8,color='black')
            for start in range(0,len(y),10):
                stop=min(start+10,len(y))
                axis.plot([start+1,stop],[y[start:stop].mean()]*2,linewidth=1.8,color='black',linestyle='--',
                          label='Historical ConFIG' if start==0 else None)
        axis.set(title=title,xlabel='Reporting epoch',ylabel='Recorded sampled value')
        axis.set_xlim(1,max(2,max(len(history) for history in rows.values())))
        axis.grid(alpha=.2)
        if axis.lines: axis.legend(fontsize=7)
    axes.flat[-1].set_visible(False)
    save_figure(fig,figdir,'learning')


def historical_history(ref, required_epochs):
    path=bound(ref)
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows)<required_epochs or [r.get('epoch') for r in rows] != list(range(1,len(rows)+1)):
        raise ValueError('complete ordered historical ConFIG history covering the learning window is required')
    return rows


def action_summary(rows):
    if ARMS[1] not in rows: return []
    counts={'attempted':0,'accepted':0,'shrunk':0,'rejected':0,'source_checks':0,'post_checks':0}
    for row in rows[ARMS[1]]:
        batches=row.get('batches');accepted=row.get('accepted_actions')
        if type(batches) is not int or type(accepted) is not int or not 0<=accepted<=batches:
            raise ValueError('candidate action attempt and acceptance counts required')
        observed=row.get('finite_counts',{})
        for key in ('action_alpha','field_source_ensemble_checks','field_post_ensemble_checks'):
            if observed.get(key)!=batches:
                raise ValueError(f'complete per-attempt {key} recording required')
        alpha=float(row['action_alpha'])*batches
        shrunk=round(2*(accepted-alpha))
        if not 0<=shrunk<=accepted or abs(shrunk-2*(accepted-alpha))>1e-5:
            raise ValueError('accepted, rejected, and half-action history is inconsistent')
        source=row['field_source_ensemble_checks']*batches
        post=row['field_post_ensemble_checks']*batches
        if (abs(source-round(source))>1e-5 or abs(post-round(post))>1e-5
                or round(source)!=batches or not 0<=round(post)<=2*batches):
            raise ValueError('invalid recorded field ensemble-check counts')
        counts['attempted']+=batches;counts['accepted']+=accepted
        counts['shrunk']+=shrunk;counts['rejected']+=batches-accepted
        counts['source_checks']+=round(source);counts['post_checks']+=round(post)
    return ['## Sampled finite actions','',
        f"The safeguarded path attempted {counts['attempted']} actions: {counts['accepted']} accepted, including {counts['shrunk']} half-displacement actions, and {counts['rejected']} rejected. It recorded {counts['source_checks']} SOURCE and {counts['post_checks']} post-proposal field ensemble checks, including probes of rejected proposals.", '',
        'These sampled checks are part of the complete attempted-epoch training cost reported below; they are not standalone profiling or a held-out feasibility guarantee. Half actions are recovered from recorded mean displacement fractions and acceptance counts.', '']


def rows_for_case(source_path, source_receipt, banks, ordinal, sample_id):
    import torch
    def read(receipt_path,receipt):
        path=receipt_path.parent/f'{ordinal:04d}.pt'
        if sha(path)!=receipt['files_sha256'][ordinal]:raise ValueError(f'bank row changed: {path}')
        row=torch.load(path,map_location='cpu',weights_only=True)
        if row['sample_id']!=sample_id:raise ValueError('bank sample ID mismatch')
        return row
    source=read(source_path,source_receipt)
    children={arm:read(path,receipt) for arm,(path,receipt) in banks.items()}
    for child in children.values():
        for key in ('truth_model','coordinates_model','coordinates_raw','native_masses','normalizer_offset','normalizer_scale','obs_indices','obs_values','obs_fields'):
            if not torch.equal(source[key],child[key]):raise ValueError(f'bank support differs: {key}')
        if source['member_seeds']!=child['member_seeds'] or source['sample_index']!=child['sample_index']:
            raise ValueError('bank member/condition coupling differs')
    return source,children


def bank_context(inputs,assessment):
    source_path=bound(inputs['source_bank']);source=json.loads(source_path.read_text())
    if (sha(source_path)!=assessment['source_bank_sha256'] or source['sample_ids']!=assessment['sample_ids']
            or source['field_order']!=assessment['field_order'] or source['members']!=4
            or source.get('panel_sha256')!=assessment['panel_sha256']
            or source.get('source_sha256')!=assessment['source_checkpoint_sha256']):
        raise ValueError('SOURCE bank does not match terminal assessment')
    banks={}
    ages=assessed_ages(assessment)
    for arm in assessment['candidates']:
        path=bound(inputs['candidate_banks'][arm]); receipt=json.loads(path.read_text())
        if (sha(path)!=assessment['bank_receipt_sha256'][arm] or receipt.get('status')!='completed'
            or receipt.get('epoch')!=ages[arm] or receipt.get('arm')!=arm
            or receipt.get('sample_ids')!=source['sample_ids'] or receipt.get('members')!=4
            or receipt.get('source_bank_sha256')!=sha(source_path)
            or receipt.get('panel_sha256')!=assessment['panel_sha256']
            or receipt.get('source_checkpoint_sha256')!=assessment['source_checkpoint_sha256']):
            raise ValueError(f'candidate bank mismatch: {arm}')
        banks[arm]=(path,receipt)
    return source_path,source,banks


def spatial(assessment,selection,source_path,source,banks,figdir):
    import numpy as np
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases=selection.get('cases')
    if not isinstance(cases,list) or not cases: raise ValueError('predeclared visual cases required')
    if selection.get('source_bank_sha256')!=sha(source_path):raise ValueError('visual selection SOURCE differs')
    selected_at=selection.get('created_unix')
    if not isinstance(selected_at,(int,float)) or any(selected_at>=receipt['start_unix'] for _,receipt in banks.values()):
        raise ValueError('visual selection must predate every candidate prediction bank')
    figures=[]
    for index,case in enumerate(cases):
        ordinal=case['ordinal'];sid=case['sample_id']
        if source['sample_ids'][ordinal]!=sid:raise ValueError('selected sample ID changed')
        row,children=rows_for_case(source_path,source,banks,ordinal,sid)
        coords=row['coordinates_raw'].numpy();mass=row['native_masses'].reshape(-1).double().numpy()
        mask=(mass>0)&np.isfinite(coords[:,:2]).all(axis=1)
        if not mask.any() or np.any(~np.isfinite(mass)) or np.any(mass<0):raise ValueError('invalid physical mask or masses')
        w=mass[mask]/mass[mask].sum();scale=row['normalizer_scale'].numpy();offset=row['normalizer_offset'].numpy()
        truth=row['truth_model'][0].numpy()*scale+offset
        preds={'SOURCE':row['members_model'][:,0].mean(0).numpy()*scale+offset}
        preds.update({LABELS[a]:child['members_model'][:,0].mean(0).numpy()*scale+offset for a,child in children.items()})
        cols=1+len(preds);fig,axes=plt.subplots(len(assessment['field_order']),cols,figsize=(4*cols,2.8*len(assessment['field_order'])),squeeze=False,constrained_layout=True)
        for field_idx,field in enumerate(assessment['field_order']):
            arrays=[truth[:,field_idx]]+[x[:,field_idx] for x in preds.values()]
            if any(not np.isfinite(v[mask]).all() for v in arrays):raise ValueError('nonfinite displayed field')
            low=min(v[mask].min() for v in arrays);high=max(v[mask].max() for v in arrays)
            for j,(ax,array) in enumerate(zip(axes[field_idx],arrays,strict=True)):
                img=ax.scatter(coords[mask,0],coords[mask,1],c=array[mask],s=.5,cmap='viridis',vmin=low,vmax=high,rasterized=True)
                ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
                if j:
                    rmse=float(np.sqrt(np.sum(w*(array[mask]-truth[mask,field_idx])**2)))
                    ax.text(.02,.02,f'RMSE {rmse:.4g} (field units)',transform=ax.transAxes,color='white',fontsize=15,bbox={'facecolor':'black','alpha':.7})
                if field_idx==0:ax.set_title(['Reference',*preds.keys()][j],fontsize=15)
                if j==0:ax.set_ylabel(field,fontsize=16)
            bar=fig.colorbar(img,ax=list(axes[field_idx]),shrink=.6,label=f'{field} (dataset units)')
            bar.ax.tick_params(labelsize=12);bar.set_label(f'{field} (dataset units)',fontsize=14)
        name=f'spatial_case_{index+1}';fig.suptitle(f'Condition {sid} · native positive-mass physical support',fontsize=18)
        save_figure(fig,figdir,name);figures.append((name,f'Condition {sid}: reference and ensemble-mean reconstructions at common fieldwise color scales; annotations are mass-weighted RMSE on the displayed positive-mass native support in each field’s dataset units.'))
    return figures


def regional(assessment,source_path,source,banks,task,figdir):
    import numpy as np
    import torch
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from phycoflow_reconstruction.evaluation.round2_task import regional_covariance
    values={'SOURCE':([],[])}; values.update({LABELS[a]:([],[]) for a in banks})
    for i,sid in enumerate(source['sample_ids']):
        row,children=rows_for_case(source_path,source,banks,i,sid)
        scale=row['normalizer_scale'].double();offset=row['normalizer_offset'].double();mass=row['native_masses'].double()
        ref=regional_covariance(row['truth_model'].double()*scale+offset,mass,task,tuple(assessment['field_order']))
        for label,draws in [('SOURCE',row['members_model']),*((LABELS[a],x['members_model']) for a,x in children.items())]:
            physical=draws[:,0].double()*scale+offset
            pred=regional_covariance(physical,mass,task,tuple(assessment['field_order']))
            values[label][0].append(float(ref[0]));values[label][1].append(float(pred.mean()))
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for label,(ref,pred) in values.items():
        axes[0].scatter(ref,pred,s=12,alpha=.65,label=label)
        axes[1].scatter(ref,np.array(pred)-ref,s=12,alpha=.65,label=label)
    allv=np.asarray([v for pair in values.values() for arr in pair for v in arr]);lo,hi=allv.min(),allv.max()
    axes[0].plot([lo,hi],[lo,hi],color='black',lw=.8);axes[1].axhline(0,color='black',lw=.8)
    axes[0].set(xlabel='Reference regional covariance',ylabel='Mean member regional covariance')
    axes[1].set(xlabel='Reference regional covariance',ylabel='Prediction minus reference')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
    save_figure(fig,figdir,'regional_relationship')


def costs_table(costs,arms,timing=None):
    if costs.get('schema')!='car_round5_costs_v1':raise ValueError('Round 5 measured costs receipt required')
    headers=['Path','Seconds / completed epoch','Occupied GPU s','Update work s','Data wait and loop overhead s','Callback / reporting s']
    keys=['seconds_per_epoch','occupied_gpu_seconds','update_work_seconds','data_wait_and_loop_overhead_seconds','callback_and_reporting_seconds']
    lines=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    for arm in arms:
        entry=costs['training'][arm]
        if any(k not in entry for k in keys):raise ValueError(f'missing measured cost field for {arm}')
        lines.append('| '+' | '.join([LABELS[arm],*(number(entry[k]) for k in keys)])+' |')
    assessment=costs['assessment']
    for key in ('gpu_seconds','cpu_seconds','wall_seconds','queue_tail_seconds'):
        if key not in assessment:raise ValueError(f'missing assessment cost {key}')
    if 'campaign_gpu_seconds' not in costs:raise ValueError('campaign GPU total including failed attempts required')
    lines += ['',f"Campaign occupied GPU seconds including failed attempts: {number(costs['campaign_gpu_seconds'])}. Assessment GPU seconds: {number(assessment['gpu_seconds'])}; assessment child CPU seconds: {number(assessment['cpu_seconds'])}; full assessment elapsed seconds: {number(assessment['wall_seconds'])}; assessment tail after last GPU generation seconds: {number(assessment['queue_tail_seconds'])}.", '', 'CPU work inside a GPU lease is included in occupied GPU time and is not added again to wall time. Missing measured values are marked unavailable.']
    for arm in arms:
        child=costs['training'][arm].get('inside_lease_child_cpu_seconds')
        lines += ['',f"{LABELS[arm]} recorded child CPU time inside its GPU leases: {number(child)} seconds; this CPU time overlaps the occupied lease and is not additional wall time."]
    if timing is not None:
        if timing.get('schema')!='car_round5_assessment_timing_note_v1':
            raise ValueError('invalid hash-bound assessment timing note')
        elapsed=timing.get('assessment_window_elapsed_seconds')
        actual=timing.get('sequential_gpu_lease_and_cpu_job_wall_sum_seconds')
        wait=timing.get('between_review_cpu_end_and_terminal_cpu_start_seconds')
        tail=timing.get('assessment_cpu_queue_tail_seconds')
        for value in (elapsed,actual,wait,tail):
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value<0:
                raise ValueError('assessment timing note has unavailable or invalid wall measurements')
        if not math.isclose(elapsed,assessment['wall_seconds'],rel_tol=1e-9,abs_tol=1e-5) or not math.isclose(tail,assessment['queue_tail_seconds'],rel_tol=1e-9,abs_tol=1e-5):
            raise ValueError('assessment timing note disagrees with measured cost receipt')
        training_wall=sum(costs['training'][arm]['complete_epoch_wall_seconds'] for arm in arms)
        if not math.isfinite(training_wall) or training_wall<=0:raise ValueError('complete-epoch training wall required for assessment comparison')
        lines += ['',f"The full assessment path elapsed {elapsed:.2f} seconds ({100*elapsed/training_wall:.1f}% of the completed training wall); the summed GPU-lease and CPU-job wall times were {actual:.2f} seconds ({100*actual/training_wall:.1f}%). The {tail:.2f}-second tail includes {wait:.2f} seconds between the review and terminal CPU jobs for the scheduled monitoring and stop decision; it is not CPU computation. The 20% assessment-wall target is a planning target, not an adoption gate."]
    phases=costs.get('gpu_seconds_by_phase')
    preparation=(sum(phases.get(k,0.) for k in ('startup_initial','startup_correction'))
        if isinstance(phases,dict) else None)
    lines += ['',f"Preparation occupied GPU seconds: {number(preparation)}; failed-attempt GPU seconds across the campaign: {number(costs.get('failed_attempt_gpu_seconds'))}; separately recorded preparation CPU seconds: {number(costs.get('independent_cpu_seconds'))}."]
    return lines


def evidence_lines(assessment):
    source=assessment['source']; arms=list(assessment['candidates']);fields=assessment['field_order']
    lines=['## Field retention and scientific relations','',
      'Field values are native-unit squared mass-weighted statistics; signed-risk changes are relative to the matched VALIDATION SOURCE and use the frozen TRAIN allowance converted to native units.','']
    for i,field in enumerate(fields):
        if i in (0,3):
            lines += [f'### Field table {1 if i == 0 else 2}', '',
                '| Field | Path | Member MSE | Ensemble mean MSE | Member variance | Signed risk change | Allowance | Sensor MSE |',
                '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for arm,row in [('SOURCE',source),*((LABELS[a],assessment['candidates'][a]) for a in arms)]:
            fidelity=row.get('fidelity',{});risk=row['risk'];sensor=row.get('sensor_mse') or [None]*len(fields)
            diff=fidelity.get('field_difference');allowance=fidelity.get('field_allowance')
            cells=[field,arm,number(risk['member_mse'][i]),number(risk['ensemble_mean_mse'][i]),
                number(risk['unbiased_member_variance'][i]),number(None if diff is None else diff[i]),
                number(None if allowance is None else allowance[i]),number(sensor[i])]
            lines.append('| '+' | '.join(cells)+' |')
        if i==2: lines.append('')
    lines += ['','| Path | State change | Scale change | Shape change | Field / sensor gate | Member gate | Regional improvement | Event change |',
              '| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |']
    for arm in arms:
        row=assessment['candidates'][arm]; scores=row['normalized_core_differences'];f=row['fidelity'];t=row['reused_tasks']
        values=[LABELS[arm],*(number(scores[k]) for k in ('state','scale','shape')),
            str(f['primary_pass']),str(f['legacy_member_pass']),number(t['regional_normalized_improvement']),number(t['event_normalized_difference'])]
        lines.append('| '+' | '.join(values)+' |')
    lines += ['', 'SSS and task changes use frozen TRAIN SOURCE units; negative SSS and event changes improve scores, while positive regional improvement is better. The event and regional tasks are reused development tasks, not independent confirmation.', '']
    pairs=assessment.get('paired_condition_differences')
    if not isinstance(pairs,dict):raise ValueError('paired condition differences and intervals required')
    lines+=['The paired 95% percentile intervals resample conditions on this finite development panel; they do not cover temporal dependence or training-seed uncertainty.', '', '| Comparison | Score | Mean change | Paired 95% interval |','| --- | --- | ---: | ---: |']
    comparison_labels={ARMS[0]:'Native minus SOURCE',ARMS[1]:'Safeguarded minus SOURCE','direct_minus_native':'Safeguarded minus native'}
    metric_labels={'state':'State','scale':'Scale','shape':'Shape',
        'event_score_difference_negative_better':'Event score change',
        'regional_score_difference_negative_better':'Regional score change'}
    for comparison,metrics in pairs.items():
        for metric,item in metrics.items():
            interval=item['interval_95']
            lines.append(f"| {comparison_labels.get(comparison,comparison.replace('_',' '))} | {metric_labels.get(metric,metric.replace('_',' '))} | {number(item['mean'])} | [{number(interval[0])}, {number(interval[1])}] |")
    return lines


def render(output,captions):
    import mistune
    from PIL import Image
    class Renderer(mistune.HTMLRenderer):
        def heading(self,text,level,**attrs):
            anchor=re.sub(r'[^a-z0-9 -]','',re.sub(r'<[^>]*>','',text).lower()).replace(' ','-')
            return f'<h{level} id="{anchor}">{text}</h{level}>\n'
    markdown=(output/'report.md').read_text()
    rendered=mistune.create_markdown(renderer=Renderer(),plugins=['table'])(markdown)
    css='img {max-width: 100%;height:auto} table {border-collapse:collapse;font-size:9pt} td,th {padding:3px;border-bottom:1px solid #bbb} h2 {page-break-after:avoid} img {page-break-inside:avoid}'
    html_text='<html><head><meta charset="utf-8"><title>Round 5 scientific report</title><style>'+css+'</style></head><body>'+rendered+'</body></html>'
    (output/'report.html').write_text(html_text)
    for name in captions:
        image=output/'figures'/f'{name}.png';pdf=output/'figures'/f'{name}.pdf'
        with Image.open(image) as im:im.verify()
        subprocess.run(['pdfinfo',str(pdf)],check=True,capture_output=True,text=True)
        anchor='figure-'+name.replace('_','-')
        if (f'id="{anchor}"' not in rendered or f'href="#{anchor}"' not in rendered
                or f'figures/{name}.png' not in rendered or f'figures/{name}.pdf' not in rendered):
            raise ValueError(f'figure preview, index or PDF link missing: {name}')
    if re.search(r'!\[[^\]]*\]\([^)]*\.pdf\)',markdown):raise ValueError('PDF masters must be links')
    # Writer ignores CSS max-width: size each raster explicitly for one A4 page.
    # Group each figure heading, raster and caption in one table row so Writer
    # keeps them together when the row fits, as established in the Round 2 packager.
    def size_image(match):
        tag=match.group(0)
        found=re.search(r'src="([^"]+)"',tag)
        if not found:raise ValueError('rendered figure has no source')
        src=html.unescape(found.group(1));path=(output/src).resolve(strict=True)
        if output not in path.parents:raise ValueError('figure path escapes report')
        with Image.open(path) as im:width,height=im.size
        scale=min(1.,650/width,590/height)
        tag=re.sub(r'src="[^"]+"','src="'+path.as_uri()+'"',tag)
        return re.sub(r'\s*/?>$',f' width="{max(1,round(width*scale))}" height="{max(1,round(height*scale))}">',tag)
    absolute=re.sub(r'<img\b[^>]*>',size_image,html_text)
    absolute=re.sub(r'href="((?:figures/|\.\./)[^\"]+)"',lambda m:'href="'+(output/m.group(1)).resolve(strict=True).as_uri()+'"',absolute)
    # First body title is a one-cell table to avoid Writer's empty first page.
    title=re.search(r'<h1\b[^>]*>.*?</h1>',absolute,flags=re.DOTALL|re.IGNORECASE)
    if title is None:raise ValueError('report title missing')
    absolute=absolute[:title.start()]+'<table><tr><td>'+title.group(0)+'</td></tr></table>'+absolute[title.end():]
    for name in captions:
        anchor='figure-'+name.replace('_','-')
        pattern=r'(<h2 id="'+re.escape(anchor)+r'">.*?</h2>\s*<p><img\b[^>]*>\s*</p>\s*<p>.*?</p>)'
        absolute,count=re.subn(pattern,lambda m:'<table width="100%" style="page-break-inside:avoid"><tr><td>'+m.group(1)+'</td></tr></table>',absolute,flags=re.DOTALL,count=1)
        if count!=1:raise ValueError(f'cannot group complete figure section: {name}')
    absolute=absolute.replace('<table>', '<table width="100%" cellspacing="0" cellpadding="3">')
    absolute=absolute.replace('<tr>', '<tr style="page-break-inside:avoid">')
    for index in (1,2):
        marker=f'<h3 id="field-table-{index}">'
        absolute=absolute.replace(marker,'<p style="page-break-before:always"><br></p>'+marker)
    with tempfile.TemporaryDirectory(prefix='car_round5_render_') as temp:
        tmp=Path(temp);source=tmp/'report_render_source.html';source.write_text(absolute)
        subprocess.run(['libreoffice','-env:UserInstallation='+ (tmp/'profile').as_uri(), '--headless','--convert-to','pdf:writer_pdf_Export','--outdir',str(tmp),str(source)],capture_output=True,text=True,check=True,timeout=180)
        pdf=tmp/'report_render_source.pdf'
        if not pdf.is_file():raise ValueError('LibreOffice did not produce the report PDF')
        (output/'report.pdf').write_bytes(pdf.read_bytes())
    info=subprocess.run(['pdfinfo',str(output/'report.pdf')],check=True,capture_output=True,text=True).stdout
    found=re.search(r'^Pages:\s*(\d+)',info,re.MULTILINE)
    if not found or int(found.group(1))<1:raise ValueError('invalid rendered PDF')
    pages=int(found.group(1)); preview=output/'page_previews';preview.mkdir()
    subprocess.run(['pdftoppm','-png','-r','60',str(output/'report.pdf'),str(preview/'page')],check=True,capture_output=True,timeout=180)
    files=sorted(preview.glob('page-*.png'))
    if len(files)!=pages:raise ValueError('rendered page preview count differs')
    for path in files:
        with Image.open(path) as im:im.verify()
    if captions:
        urls=subprocess.run(['pdfinfo','-url',str(output/'report.pdf')],check=True,capture_output=True,text=True).stdout
        for name in captions:
            if f'{name}.pdf' not in urls: raise ValueError(f'PDF master link missing from rendered PDF: {name}')
        listing=subprocess.run(['pdfimages','-list',str(output/'report.pdf')],check=True,capture_output=True,text=True).stdout
        if len([line for line in listing.splitlines()[2:] if line.strip()]) < len(captions):
            raise ValueError('rendered PDF does not contain every preview image')
    report_text=subprocess.run(['pdftotext',str(output/'report.pdf'),'-'],check=True,capture_output=True,text=True).stdout
    if 'Coherence Adaptive Repair' not in report_text.split('\f',1)[0]:raise ValueError('report title missing from first PDF page')
    return pages


def produce(inputs_path,campaign,output):
    import sys
    root=Path(__file__).resolve().parents[2];sys.path[:0]=[str(root/'src'),str(root)]
    start=time.time();inputs_path=inputs_path.resolve(strict=True);campaign=campaign.resolve(strict=True);output=output.resolve()
    if campaign/'reports' not in output.parents:raise ValueError('report must be a new campaign-owned directory')
    inputs=json.loads(inputs_path.read_text())
    if inputs.get('schema')!='car_round5_report_inputs_v1':raise ValueError('Round 5 report input manifest required')
    decision=doc(inputs['decision']);costs=doc(inputs['costs']);engineering=doc(inputs['engineering'])
    timing=doc(inputs['assessment_timing_note']) if inputs.get('assessment_timing_note') else None
    if timing is not None and timing.get('costs',{}).get('sha256')!=inputs['costs']['sha256']:
        raise ValueError('assessment timing note does not bind this cost receipt')
    if decision.get('scope') not in ('model','engineering') or not decision.get('recommendation') or not decision.get('reason'):
        raise ValueError('terminal decision scope, recommendation and reason required')
    if not isinstance(engineering.get('summary'),str) or not engineering['summary'].strip():raise ValueError('engineering disposition summary required')
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);figdir=output/'figures';figdir.mkdir();captions={}
    lines=['# Coherence Adaptive Repair: Round 5','',f"**Decision:** {decision['recommendation']}. {decision['reason']}",'',
        'TEST remains locked. SOURCE/C8 and the historical Round 3 and Round 4 artifacts remain unchanged; those rounds remain closed. The SSS package compares a complete optimization package against native continuation, rather than isolating the contribution of each component.','',
        '## Engineering disposition','',engineering['summary'],'']
    if inputs.get('learning_review'):
        review=doc(inputs['learning_review']); stats=doc(review['statistics'])
        recent=stats['arms'][ARMS[1]]['sampled_field_ten_epoch_summaries'][-1]
        if recent['epochs'] != [91,100] or review['continued_arms']:
            raise ValueError('expected stopped 100-epoch learning review')
        ch4,co=recent['pre_excess_mean_over_allowance'][:2]
        lines[4:4]=[f"Both routes stopped at the 100-epoch review instead of invoking the conditional 500-epoch maximum because of material multi-field and Scale failures. In the safeguarded candidate's final ten epochs, mean pre-proposal CH4 and CO excesses were {ch4:.6f} and {co:.6f} times their allowances; {recent['pre_outside_any']}/120 proposals were pre-infeasible and {recent['post_proposal_outside_any']}/120 were post-proposal infeasible. Recent sampled field risks and Scale showed no recovery trend, leaving no evidence-based unchanged-recipe recovery question. This rejects the routes within the executed budget; it does not establish that later recovery is impossible.",'']
    has_gradient='query_gradient_review' in engineering
    has_actions='query_actions' in engineering
    if has_gradient != has_actions:
        raise ValueError('query engineering figure requires both hash-bound inputs')
    if has_gradient:
        query_engineering(doc(engineering['query_gradient_review']),doc(engineering['query_actions']),figdir)
        captions['query_engineering']=QUERY_CAPTION
    if decision['scope']=='engineering':
        arms=[]
        if decision.get('model_assessment_performed') is not False:raise ValueError('engineering-only closure must state that no model assessment was performed')
        lines+=['No learned reconstruction is adopted from this engineering-only disposition.','']
    else:
        assessment=doc(inputs['assessment'])
        if (decision.get('assessment_sha256')!=inputs['assessment']['sha256'] or assessment.get('split')!='validation'
                or len(assessment.get('sample_ids',[]))!=100 or assessment.get('members')!=4
                or 'both tasks excluded' in assessment.get('task_scope','')):
            raise ValueError('terminal decision requires matching VALIDATION100 J4 endpoint with task scores')
        lines += [panel_coverage(assessment),'']
        arms=list(assessment['candidates'])
        if not arms or any(a not in ARMS for a in arms):raise ValueError('invalid terminal candidate set')
        if any('reused_tasks' not in assessment['candidates'][a] for a in arms):raise ValueError('terminal task results required')
        eligible=[a for a in arms if assessment['candidates'][a].get('replacement_eligible') is True]
        if len(eligible) < 2:
            expected=eligible[0] if eligible else 'retain SOURCE/C8'
        else:
            advantage=assessment.get('direct_regional_advantage')
            if not isinstance(advantage,(int,float)) or not math.isfinite(advantage):
                raise ValueError('matched regional advantage is required for two qualifying arms')
            if advantage >= .02 and not isinstance(decision.get('extra_cost_worthwhile'),bool):
                raise ValueError('both qualifying paths require an explicit extra-cost decision')
            expected=ARMS[1] if advantage >= .02 and decision['extra_cost_worthwhile'] else ARMS[0]
        if decision['recommendation'] != expected:
            raise ValueError('terminal recommendation violates the declared scientific and cost decision')
        ages=assessed_ages(assessment)
        freeze=doc(inputs['formal_freeze'])
        lines += methods_lines(freeze,inputs,arms)
        history=histories(inputs['histories'],assessment)
        lines += age_and_exposure_lines(history,ages)
        age_caption='Assessed epochs: '+', '.join(f'{LABELS[arm]} {ages[arm]}' for arm in arms)+'.'
        if len(set(ages.values()))>1:
            age_caption+=' Learning ages differ.'
        historical=historical_history(inputs['historical_history'],max(len(v) for v in history.values()))
        learning(history,figdir,historical)
        lines += action_summary(history)
        captions['learning']='Raw sampled epoch histories and ten-epoch means for native and SSS training quantities, learning rate, actual action dot product and proposal acceptance. Historical ConFIG appears only for the same frozen State, Scale, Shape and learning-rate definitions on the matching epoch window; query grouping, when adopted, changes cross-condition coupling. These are training diagnostics, not held-out scientific metrics.'
        source_path,source,banks=bank_context(inputs,assessment)
        selection=doc(inputs['visual_selection']);task=doc(inputs['regional_task'])
        for name,caption in spatial(assessment,selection,source_path,source,banks,figdir):captions[name]=caption
        regional(assessment,source_path,source,banks,task,figdir)
        captions['regional_relationship']='Matched condition-level reference versus mean-member regional CO–U1 covariance and corresponding signed condition differences, computed on the frozen right-half mass-weighted task from saved predictions.'
        captions={name: caption if name=='query_engineering' else caption+' '+age_caption for name,caption in captions.items()}
        lines += evidence_lines(assessment)+['']
        lines += ['Fields and reconstruction errors are decoded to native dataset values. The saved case configuration labels all five physical units as unknown, and the HDF5 attributes supply no unit labels; figures therefore state native dataset units without assigning SI units.','']
    lines += ['## Training and assessment cost','']+costs_table(costs,arms,timing)+['']
    if inputs.get('prior_report_cpu_receipts'):
        lines += ['Reporting CPU is separate from scientific assessment costs. Earlier report render attempts:']
        for label,ref in inputs['prior_report_cpu_receipts'].items():
            receipt=doc(ref); cpu=receipt['user_cpu_seconds']+receipt['system_cpu_seconds']
            rel=os.path.relpath(bound(ref),output)
            lines.append(f"- [{label}]({rel}): {cpu:.3f} CPU seconds; {receipt['wall_seconds']:.3f} wall seconds.")
        lines += ['', 'The current report render has a separate companion CPU receipt.','']
    if inputs.get('learning_review'):
        refs=[('Final decision',inputs['decision']),('Task-free learning review and stop decision',inputs['learning_review']),('Terminal assessment',inputs['assessment']),('Measured costs',inputs['costs'])]
        lines += ['## Evidence links','']+[f'- [{label}]({os.path.relpath(bound(ref),output)})' for label,ref in refs]+['']
    if captions:
        lines += ['## Figure index','']+[f'- [Figure {name.replace("_"," ")}](#figure-{name.replace("_","-")})' for name in captions]+['']
        for name,caption in captions.items():
            lines += [f'## Figure {name.replace("_"," ")}','',f'![{html.escape(caption)}](figures/{name}.png)','',caption+f' [PDF master](figures/{name}.pdf)','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    pages=render(output,captions)
    receipt={'schema':'car_round5_report_v1','status':'rendered_pending_visual_inspection','pages':pages,
        'inputs_sha256':sha(inputs_path),'report_md_sha256':sha(output/'report.md'),
        'report_html_sha256':sha(output/'report.html'),'report_pdf_sha256':sha(output/'report.pdf'),
        'figures':{name:{'png_sha256':sha(figdir/f'{name}.png'),'pdf_sha256':sha(figdir/f'{name}.pdf')} for name in captions},
        'rendered_page_previews':str(output/'page_previews'),'elapsed_seconds':time.time()-start}
    (output/'report_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--query-only',action='store_true',help='render a standalone query figure from two bound receipts')
    args=p.parse_args();function=render_query_only if args.query_only else produce
    print(json.dumps(function(args.inputs,args.campaign,args.output),indent=2))

if __name__=='__main__':main()
