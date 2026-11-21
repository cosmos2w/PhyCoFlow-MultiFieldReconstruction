#!/usr/bin/env python3
"""Build an English Round 4 report exclusively from hash-bound saved evidence.

Inputs use car_round4_report_inputs_v1. Every binding is {path, sha256}. Required
bindings: train_manifest, teacher_gate, teacher_checkpoint, teacher_history and
forecast. Student terminal additionally requires assessment, decision, costs,
visual_selection, source_bank_receipt and student_bank_receipts keyed by both
full arm names. Teacher-rejected inputs must not include student/task evidence.
A partial cost stop uses teacher_cost_rejected with a bound budget decision,
scientific configuration and complete partial-cost receipt, but no utility gate.
A teacher-only measured cost receipt, when available, uses
car_round4_teacher_closeout_cost_v1 with complete=true and gpu_seconds mapping.
No model inference, population scoring or task fair-score recomputation is
invoked. Fixed-case panels transform cached arrays with the frozen definitions
and compute their plotted-field RMSE and descriptive regional covariance.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from scripts.analysis.report_car_round3 import save_figure

DATA_ROOT = Path('/data2/wanglz/PhyCoFlow-MultiFieldReconstruction')
TEACHER = 'FFM_SSS_Target_Amortization'
NATIVE = 'FFM_Native_Supervised_Continuation'
ARMS = (TEACHER, NATIVE)
LABELS = {TEACHER:'SSS target amortization', NATIVE:'Native supervised continuation'}
FAMILIES = ('global_distribution', 'multiscale_spectrum', 'topology')
FAMILY_NAMES = ('State', 'Scale', 'Shape')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def bound(binding, *, required=True):
    if binding is None and not required:
        return None
    if not isinstance(binding, dict) or set(binding) != {'path', 'sha256'}:
        raise ValueError('Evidence must be an explicit path and SHA-256 binding')
    path = Path(binding['path']).resolve(strict=True)
    if sha(path) != binding['sha256']:
        raise ValueError(f'Evidence hash mismatch: {path}')
    return path


def document(binding, schema=None, *, required=True):
    path = bound(binding, required=required)
    if path is None: return None
    result = json.loads(path.read_text())
    if schema is not None and result.get('schema') != schema:
        raise ValueError(f'{path} is not {schema}')
    return result


def finite(value):
    return isinstance(value, (int,float)) and not isinstance(value,bool) and math.isfinite(value)


def teacher_preparation_cpu(cost):
    """Read only the job receipts already bound by the inclusive cost receipt."""
    values={'qualify':[], 'generate':[], 'current_route_timing':[], 'teacher_numerical_qualification':[]}
    for job in cost.get('included_jobs',[]):
        phase=job.get('phase')
        if phase in values:
            values[phase].append(document(job['receipt']).get('process_cpu_seconds'))
    def total(phases):
        entries=[v for phase in phases for v in values[phase]]
        return sum(entries) if all(values[phase] for phase in phases) and all(finite(v) and v>=0 for v in entries) else None
    return total(('qualify','generate')),total(('current_route_timing',)),total(('teacher_numerical_qualification',))


def history_rows(path, expected, population):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if [row.get('epoch') for row in rows] != list(range(1, expected+1)):
        raise ValueError('Teacher history must cover every completed epoch in order')
    cumulative = 0
    for row in rows:
        batch = row['batch_stream']
        if len(batch) != 12 or any(item['batch_size'] != len(item['condition_indices'])
                or item['batch_size'] != len(item['accepted_alpha']) or item['batch_size'] < 1
                or any(not isinstance(i,int) or isinstance(i,bool) or not 0 <= i < population for i in item['condition_indices'])
                or len(set(item['condition_indices'])) != item['batch_size']
                or any(not finite(a) or not 0 <= a <= 1 for a in item['accepted_alpha']) for item in batch):
            raise ValueError('Teacher visit stream or accepted-ray history is incomplete')
        visits = sum(item['batch_size'] for item in batch); cumulative += visits
        if (row['visits'] != visits or row['cumulative_visits'] != cumulative
                or not math.isclose(row['dataset_pass_equivalents'], cumulative/population, rel_tol=1e-12)):
            raise ValueError('Teacher visit totals disagree with recorded stream')
        values = list(row['family_v'].values()) + row['field_displacement'] + row['field_member_spread'] + row['min_field_slack'] + [row['mean_alpha'], row['elapsed_seconds'], row['cpu_seconds']]
        if set(row['family_v']) != set(FAMILIES) or not all(finite(v) for v in values):
            raise ValueError('Teacher history contains missing or nonfinite diagnostics')
    return rows



def validate_teacher_field_diagnostics(gate, checkpoint, manifest):
    """Check saved TRAIN800 field evidence, without rerunning a metric."""
    fields=manifest['field_order']; nfields=len(fields)
    diagnostics=gate.get('train800_field_diagnostics', {})
    if set(diagnostics) != {'source','teacher'}:
        raise ValueError('Teacher gate lacks SOURCE and teacher TRAIN800 field diagnostics')
    epsilon_model=checkpoint['epsilon_model_squared']; epsilon_physical=checkpoint['epsilon_physical_squared']
    if len(epsilon_model)!=nfields or len(epsilon_physical)!=nfields or any(not finite(a) or a<=0 or not finite(b) or b<=0 for a,b in zip(epsilon_model,epsilon_physical,strict=True)):
        raise ValueError('Invalid frozen field-unit conversion')
    factors=[b/a for a,b in zip(epsilon_model,epsilon_physical,strict=True)]
    expected_names={'finite_law_mean_mse','member_mse','member_spread','source_displacement','observed_entry_mse'}
    expected_counts=[800*int(manifest['sensor_count'])*4 if name==manifest['sensor_field'] else 0 for name in fields]
    for label,record in diagnostics.items():
        if (record.get('schema')!='car_round4_teacher_train800_fields_v1' or record.get('split')!='TRAIN'
                or record.get('sample_count')!=800 or record.get('members')!=4 or record.get('field_order')!=fields
                or len(record.get('physical_field_units',[]))!=nfields or record.get('observation_entry_counts')!=expected_counts):
            raise ValueError(f'{label} TRAIN800 field diagnostic identity differs')
        model=record['model_field_units_squared']; physical=record['decoded_field_units_squared']
        if set(model)!=expected_names or set(physical)!=expected_names:
            raise ValueError('Incomplete finite-law field measurements')
        for name in expected_names:
            if len(model[name])!=nfields or len(physical[name])!=nfields:
                raise ValueError('Field diagnostic lengths differ')
            for i,(a,b) in enumerate(zip(model[name],physical[name],strict=True)):
                if name=='observed_entry_mse' and expected_counts[i]==0:
                    if a is not None or b is not None: raise ValueError('Unobserved field has a sensor residual')
                elif not finite(a) or a<0 or not finite(b) or b<0 or not math.isclose(b,a*factors[i],rel_tol=1e-9,abs_tol=1e-12):
                    raise ValueError('Field diagnostic unit conversion or value is invalid')
        for mean,member,spread in zip(model['finite_law_mean_mse'],model['member_mse'],model['member_spread'],strict=True):
            if not math.isclose(mean+spread,member,rel_tol=1e-8,abs_tol=1e-10):
                raise ValueError('Empirical finite-law mean and spread do not recover member MSE')
        if label=='source' and any(value != 0 for value in model['source_displacement']):
            raise ValueError('SOURCE displacement must be exactly zero')
    if diagnostics['source']['physical_field_units']!=diagnostics['teacher']['physical_field_units']:
        raise ValueError('SOURCE and teacher field units differ')
    return diagnostics


def teacher_field_tables(diagnostics, fields):
    def pair(a,b):
        if a is None or b is None: return 'unobserved'
        return f'{math.sqrt(a):.5g} / {math.sqrt(b):.5g}'
    lines=['## TRAIN800 teacher field diagnostics','',
           'These are in-sample finite-law errors on the fixed TRAIN800 replay. The mean is the exact average of the four jointly optimized atoms; it has no independent-draw variance subtraction. Each pair is SOURCE / teacher. Physical-mass weighting applies to field RMS values; sensor RMS averages observed member entries.','']
    for key,title in (('model_field_units_squared','Model field units'),('decoded_field_units_squared','Decoded dataset field units')):
        source=diagnostics['source'][key]; teacher=diagnostics['teacher'][key]
        lines += [f'### {title}','', '| Field | Mean RMSE SOURCE / teacher | Member RMSE SOURCE / teacher | Member spread RMS SOURCE / teacher | Teacher displacement RMS | Sensor RMSE SOURCE / teacher |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for i,field in enumerate(fields):
            unit=diagnostics['source']['physical_field_units'][i]
            name=field if key=='model_field_units_squared' else f'{field} ({unit if unit is not None else "unknown"})'
            lines.append(f"| {name} | {pair(source['finite_law_mean_mse'][i],teacher['finite_law_mean_mse'][i])} | {pair(source['member_mse'][i],teacher['member_mse'][i])} | {pair(source['member_spread'][i],teacher['member_spread'][i])} | {math.sqrt(teacher['source_displacement'][i]):.5g} | {pair(source['observed_entry_mse'][i],teacher['observed_entry_mse'][i])} |")
        lines.append('')
    return lines

def figure_history(rows, fields, figdir):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3,2,figsize=(11,10),constrained_layout=True)
    epochs = [r['epoch'] for r in rows]
    marker={'marker':'o'} if len(rows)==1 else {}
    for key,label in zip(FAMILIES,FAMILY_NAMES,strict=True):
        ax[0,0].plot(epochs,[r['family_v'][key] for r in rows],label=label,**marker)
    ax[0,0].set_ylabel('Sampled teacher V / SOURCE unit'); ax[0,0].legend()
    ray_values=[[a for batch in row['batch_stream'] for a in batch['accepted_alpha']] for row in rows]
    ax[0,1].plot(epochs,[r['mean_alpha'] for r in rows],label='Mean ray factor',**marker)
    ax[0,1].fill_between(epochs,[min(v) for v in ray_values],[max(v) for v in ray_values],alpha=.15,label='Observed range')
    ax[0,1].plot(epochs,[sum(a == 0 for a in values)/len(values) for values in ray_values],label='Zero-ray fraction',**marker)
    ax[0,1].set_ylabel('Ray factor / visit fraction'); ax[0,1].legend(fontsize=7)
    for axis,key,title in ((ax[1,0],'field_displacement','SOURCE displacement'),
                           (ax[1,1],'field_member_spread','Finite-law member spread'),
                           (ax[2,0],'min_field_slack','Minimum field feasibility slack')):
        for i,name in enumerate(fields): axis.plot(epochs,[r[key][i] for r in rows],label=name,**marker)
        axis.set_ylabel(title+' (squared model units)'); axis.legend(fontsize=7)
    ax[2,1].plot(epochs,[r['elapsed_seconds'] for r in rows],label='Update wall',**marker)
    ax[2,1].plot(epochs,[r['cpu_seconds'] for r in rows],label='Process CPU',**marker)
    ax[2,1].set_ylabel('Seconds per reporting epoch'); ax[2,1].legend()
    for axis in ax.flat: axis.set_xlabel('Reporting epoch'); axis.grid(alpha=.15)
    save_figure(fig,figdir,'teacher_histories')


def figure_teacher_gate(gate, figdir):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values = [gate['normalized_improvement'][name] for name in FAMILIES]
    if not all(finite(v) for v in values): raise ValueError('invalid teacher replay improvement')
    fig, ax = plt.subplots(figsize=(7,3.7),constrained_layout=True)
    ax.bar(FAMILY_NAMES,values,color='#286a8e'); ax.axhline(0,color='black',lw=.8)
    ax.axhline(.05,color='#40853f',ls='--',label='One-family improvement margin')
    ax.axhline(-.02,color='#b34b42',ls=':',label='Maximum regression')
    ax.set_ylabel('(SOURCE V − teacher V) / SOURCE V unit')
    ax.set_title('TRAIN800 independent-query teacher replay'); ax.legend(fontsize=8)
    save_figure(fig,figdir,'teacher_replay')


def figure_students(assessment, figdir):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    positions=np.arange(3)
    for i, arm in enumerate(ARMS):
        row=assessment['candidates'][arm]
        vals=[row['normalized_core_differences'][x] for x in ('state','scale','shape')]
        ax[0].bar(positions+(i-.5)*.35,vals,.35,label=LABELS[arm])
        losses=assessment['native_learning_histories'][arm]['native_rf_loss']
        if len(losses) != assessment['epoch'] or not all(finite(x) for x in losses):
            raise ValueError('Native learning history is incomplete')
        ax[1].plot(range(1,len(losses)+1),losses,label=LABELS[arm])
    ax[0].set_xticks(positions,FAMILY_NAMES); ax[0].axhline(0,color='black',lw=.8)
    ax[0].axhline(.02,color='#b34b42',ls=':'); ax[0].axhline(-.05,color='#40853f',ls='--')
    ax[0].set_ylabel('(Student fair − SOURCE fair) / SOURCE unit');ax[0].legend(fontsize=7)
    ax[1].set(xlabel='Reporting epoch',ylabel='Sampled native RF loss');ax[1].legend(fontsize=7)
    save_figure(fig,figdir,'student_fair_and_learning')


def figure_costs(forecast, cost, figdir, *, teacher_only=False):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    forecast_values=forecast['forecast_gpu_seconds']
    measured = cost.get('candidate_gpu_seconds', cost.get('gpu_seconds', {})) if cost else {}
    measured_label = 'Accounted upper bound' if cost and cost.get('cost_basis') == 'conservative_upper_bound' else 'Measured'
    values=[('Qualification / failed work', forecast_values.get('qualification_measured_plus_upper_bound'), cost.get('candidate_overhead_gpu_seconds') if cost else None),
            ('SOURCE preparation*', forecast_values.get('full_train_target_preparation_remaining'), measured.get('source_generation')),
            ('Teacher', forecast_values.get('teacher_including_calibration_replay_audit_and_persistence'), measured.get('teacher')),
            ('Amortized student', forecast_values.get('native_amortization_500_epochs'), measured.get('student')),
            ('Assessment', forecast_values.get('assessment'), measured.get('assessment'))]
    fig, ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    import numpy as np
    x=np.arange(len(values))
    ax.bar(x-.2,[v/3600 if finite(v) else 0 for _,v,_ in values],.4,label='Prelaunch forecast')
    for i,(_,_,value) in enumerate(values):
        if finite(value): ax.bar(x[i]+.2,value/3600,.4,color='#b65c39',label=measured_label if i==0 else None)
        else: ax.text(x[i]+.2,.02,'not run' if teacher_only and i in (3,4) else 'unknown',ha='center',va='bottom',rotation=90,fontsize=7)
    ax.set_xlim(-.6,len(values)-.4)
    ax.set_xticks(x,[name for name,_,_ in values],rotation=15,ha='right');ax.set_ylabel('Occupied GPU hours')
    ax.set_title('Phase costs; forecast includes planned work');ax.legend()
    save_figure(fig,figdir,'inclusive_costs')



def paired_condition_figure(assessment, selection, figdir):
    """Plot paired saved score differences in chronological sampled blocks."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    baseline_path=bound(selection['baseline'])
    if sha(baseline_path) != assessment['source_baseline_sha256']:
        raise ValueError('Paired regional baseline differs from the assessment')
    baseline=json.loads(baseline_path.read_text())
    ids=assessment['sample_ids']
    if baseline['sample_ids'] != ids or len(ids) != 100:
        raise ValueError('Paired SOURCE and student conditions differ')
    def order_key(item):
        try:
            trajectory, clock=item[1].rsplit(':',1)
            return trajectory, int(clock)
        except (ValueError,AttributeError):
            raise ValueError('Chronological condition IDs must include trajectory and integer time')
    order=sorted(enumerate(ids),key=order_key)
    if len({item[1] for item in order}) != len(order): raise ValueError('Repeated condition IDs')
    groups=[]
    for trajectory in dict.fromkeys(order_key(item)[0] for item in order):
        positions=[i for i,item in enumerate(order) if order_key(item)[0] == trajectory]
        groups.extend(positions[start:start+20] for start in range(0,len(positions),20))
    source=assessment['source']['per_condition_scores']
    source_regional=baseline['reserved_task']['source_per_condition']
    if len(source) != 100 or len(source_regional) != 100: raise ValueError('Missing matched SOURCE per-condition scores')
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for axis,key,title in zip(axes.flat,('state','scale','shape','regional'),('State','Scale','Shape','Regional task'),strict=True):
        for arm in ARMS:
            row=assessment['candidates'][arm]
            if key=='regional':
                candidate=row['reused_tasks']['regional_per_condition']; reference=source_regional
                normalizer=assessment['task_normalizers']['regional']
            else:
                candidate=[entry[key] for entry in row['per_condition_scores']]
                reference=[entry[key] for entry in source]; normalizer=assessment['normalizers'][key]
            if len(candidate)!=100 or not finite(normalizer) or normalizer<=0 or not all(finite(x) for x in candidate+reference):
                raise ValueError('Incomplete paired per-condition scores')
            differences=np.asarray([(candidate[index]-reference[index])/normalizer for index,_ in order])
            x=np.arange(100)
            axis.scatter(x,differences,s=8,alpha=.25,label=LABELS[arm])
            axis.plot([np.mean(block) for block in groups],[differences[block].mean() for block in groups],marker='o',lw=1)
        axis.axhline(0,color='black',lw=.7);axis.set_title(title)
        axis.set_xlabel('Chronologically ordered sampled condition');axis.set_ylabel('(Student − SOURCE) / frozen unit')
        axis.grid(alpha=.15)
    axes[0,0].legend(fontsize=7)
    save_figure(fig,figdir,'paired_condition_blocks')


def relationship_context(freeze_ref, selection, assessment):
    import torch
    freeze_path=bound(freeze_ref)
    if sha(freeze_path) != assessment['freeze_sha256']:
        raise ValueError('Relationship observer freeze differs from terminal scoring')
    freeze=json.loads(freeze_path.read_text())
    task_path=bound(freeze['reserved_task']); task=json.loads(task_path.read_text())
    baseline_path=bound(selection['baseline']); baseline=json.loads(baseline_path.read_text())
    panel_path=bound(selection['panel']); panel=json.loads(panel_path.read_text())
    if (sha(panel_path)!=assessment['panel_sha256'] or sha(baseline_path)!=assessment['source_baseline_sha256']
            or panel['sample_ids'] != assessment['sample_ids'] or baseline['sample_ids'] != assessment['sample_ids']
            or baseline['reserved_task']['task_sha256'] != sha(task_path)):
        raise ValueError('Relationship panel, regional task or baseline identity differs')
    core_ref=freeze['artifacts']['ffm']['core']; core_path=bound(core_ref)
    if sha(core_path)!=assessment['core_sha256']: raise ValueError('Relationship core differs from observer')
    core=torch.load(core_path,map_location='cpu',weights_only=True)
    run=Path(freeze['artifacts']['ffm']['core_run']).resolve(strict=True)
    run_manifest=json.loads((run/'run_manifest.json').read_text())
    families={}
    for name in ('multiscale_spectrum','topology'):
        path=run/'artifacts'/f'{name}_family.pt'
        if sha(path)!=run_manifest['coherence_family_state_sha256s'][name]: raise ValueError('Frozen relationship family changed')
        families[name]=torch.load(path,map_location='cpu',weights_only=True)
    query=torch.as_tensor(panel['query_indices'],dtype=torch.long)
    if query.numel()!=4096 or query.unique().numel()!=4096:
        raise ValueError('Frozen Q4096 support required for relationship figures')
    return {'task':task,'baseline':baseline,'query':query,'core':core,'families':families}


def plot_case_relationship(case_name, sample_id, ordinal, source, candidates, assessment, context, figdir, *, labels_override=None):
    """Only selected cached cases; no population assessment or model inference."""
    import numpy as np
    import torch
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    from phycoflow_reconstruction.evaluation.round2_task import regional_covariance
    from phycoflow_reconstruction.coherence.families.multiscale_spectrum.statistics import piecewise_fluctuations, spectral_coefficients, band_gram
    from phycoflow_reconstruction.coherence.families.topology.geometry import coordinate_digest, rasterize_fields
    from phycoflow_reconstruction.coherence.families.topology.betti_curves import gaussian_blur
    fields=assessment['field_order']; q=context['query']; task=context['task']
    if task.get('field_names') != ['CO','U_1'] or any(name not in fields for name in task['field_names']):
        raise ValueError('Frozen regional task must identify CO and U_1')
    scale_family=context['families']['multiscale_spectrum']; shape_family=context['families']['topology']
    if list(scale_family['field_names']) != fields or list(shape_family['field_names']) != fields or scale_family['units'] != 'model_units' or shape_family['units'] != 'model_units':
        raise ValueError('Relationship observer field order or units changed')
    if coordinate_digest(source['coordinates_model'][0,q]) != shape_family['geometry_sha256']:
        raise ValueError('Relationship raster support geometry changed')
    scale=source['normalizer_scale'].double(); offset=source['normalizer_offset'].double()
    truth=source['truth_model'][0].double().unsqueeze(0)
    model_draws=[truth, source['members_model'][:,0].double()]
    model_draws.extend(candidates[arm]['members_model'][:,0].double() for arm in ARMS)
    physical=[x*scale+offset for x in model_draws]
    weights=source['native_masses'].double(); region=torch.as_tensor(task['region_point_indices'],dtype=torch.long)
    # The primitive verifies the immutable region and native physical measure.
    cov=[regional_covariance(x,weights,task,tuple(fields)).cpu().numpy() for x in physical]
    wc=weights[region].numpy();wc=wc/wc.sum(); fi=[fields.index('CO'),fields.index('U_1')]
    xy=[x[:,region][:,:,fi].reshape(-1,2).numpy() for x in physical]
    pair_weights=[np.tile(wc/len(x),len(x)) for x in physical]
    if not all(np.isfinite(x).all() for x in xy): raise ValueError('Nonfinite regional fields')
    low=np.min(np.concatenate(xy,axis=0),axis=0);high=np.max(np.concatenate(xy,axis=0),axis=0)
    high=np.where(high>low,high,low+1.)
    bins=[np.linspace(low[i],high[i],33) for i in range(2)]
    hist=[np.histogram2d(v[:,0],v[:,1],bins=bins,weights=w)[0] for v,w in zip(xy,pair_weights,strict=True)]
    maximum=max(float(a.max()) for a in hist) or 1.
    state=scale_family['state_dict']; xq=torch.cat([v[:,q] for v in model_draws]).to(torch.float64)
    if not torch.equal(state['analysis_index'].cpu(),torch.arange(len(q))): raise ValueError('Scale support ordering changed')
    m=state['mass'].double(); basis=state['basis'].double(); filters=state['filters'].double()
    fluctuations=piecewise_fluctuations(xq,state['piece_measures'].double(),state['piece_index'].long())
    coef=spectral_coefficients(fluctuations,basis*m[:,None]); gram=band_gram(coef,filters.square())
    ia,ib=fi; floors=state['power_floor'].double()
    all_rho=gram[:,:,ia,ib]/((gram[:,:,ia,ia]+floors[:,ia])*(gram[:,:,ib,ib]+floors[:,ib])).sqrt()
    rho=torch.stack((all_rho[0],all_rho[1:5].mean(0),all_rho[5:9].mean(0),all_rho[9:13].mean(0)))
    if not torch.isfinite(rho).all(): raise ValueError('Nonfinite frozen scale organization')
    topo=shape_family['state_dict']; grid=tuple(shape_family['config']['geometry']['grid_shape'])
    raster_inputs=torch.stack([draws[0,q] for draws in model_draws]).float()
    raster=rasterize_fields(raster_inputs,topo['neighbor_indices'].long(),topo['neighbor_weights'].float(),grid)
    cfg=shape_family['config']; sigma=float(cfg['filtration']['smoothing_sigma'])
    raster=gaussian_blur(raster,sigma,bool(cfg['geometry'].get('periodic',False)),radius=cfg['filtration'].get('smoothing_radius'))
    shape=context['core']['shape_artifact']; idx=fi[0]
    mean=float(shape['field_mean'][idx]); std=float(shape['field_scale'][idx])
    if not math.isfinite(std) or std <= 0: raise ValueError('Invalid frozen Shape scale')
    excursion=((raster[:,idx]-mean)/std >= 0).cpu().numpy()
    if excursion.shape[1:] != grid: raise ValueError('Shape raster changed')
    domain=shape.get('domain')
    if not isinstance(domain,(list,tuple)) or len(domain)!=3 or tuple(domain[0])!=grid or domain[2] != bool(cfg['geometry'].get('periodic',False)):
        raise ValueError('Frozen Shape domain or periodicity changed')
    weights_raster=topo['neighbor_weights'].double()
    indices_raster=topo['neighbor_indices'].long()
    if (weights_raster.shape != indices_raster.shape or not torch.isfinite(weights_raster).all()
            or (weights_raster < 0).any() or (indices_raster < 0).any() or (indices_raster >= len(q)).any()):
        raise ValueError('Invalid frozen Shape raster operator')
    if domain[1] is None:
        if topo.get('domain_mask') is not None or not torch.allclose(weights_raster.sum(-1),torch.ones(len(weights_raster),dtype=torch.double),rtol=1e-5,atol=1e-6):
            raise ValueError('Unmasked Shape domain requires every raster cell to be valid')
        if not (weights[q][indices_raster][weights_raster > 0] > 0).all():
            raise ValueError('Shape raster reaches invalid native mass support')
        valid_mask=np.ones(grid,dtype=bool)
    else:
        raster_mask=topo.get('domain_mask')
        if raster_mask is None or tuple(raster_mask.shape) != grid or hashlib.sha256(raster_mask.cpu().numpy().tobytes()).hexdigest()!=domain[1]:
            raise ValueError('Frozen Shape raster mask changed')
        valid_mask=raster_mask.bool().cpu().numpy()
        if not valid_mask.any(): raise ValueError('Shape raster mask is empty')
    baseline=context['baseline']['reserved_task']['source_per_condition']
    task_values=[None,baseline[ordinal]]+[assessment['candidates'][arm]['reused_tasks']['regional_per_condition'][ordinal] for arm in ARMS]
    if not all(finite(v) for v in task_values[1:]): raise ValueError('Missing saved per-condition regional task scores')
    labels=['Reference','SOURCE','SSS target student','Native student'] if labels_override is None else labels_override
    if len(labels) != 4: raise ValueError('Four relationship labels required')
    fig=plt.figure(figsize=(14,9),constrained_layout=True);gs=fig.add_gridspec(3,4)
    occupancy_axes=[]
    for i in range(4):
        axis=fig.add_subplot(gs[0,i]); occupancy_axes.append(axis)
        img=axis.imshow(hist[i].T,origin='lower',extent=(low[0],high[0],low[1],high[1]),aspect='auto',norm=PowerNorm(gamma=.35,vmin=0,vmax=maximum),cmap='magma')
        task_label='' if i==0 else f'\nregional fair score {task_values[i]:.3g}'
        axis.set_title(f'{labels[i]}\nmean member regional covariance {cov[i].mean():.3g}'+task_label,fontsize=8)
        axis.set_xlabel('CO (dataset units)');axis.set_ylabel('U_1 (dataset units)')
    fig.colorbar(img,ax=occupancy_axes,shrink=.7,label='Regional mass fraction per bin')
    axis=fig.add_subplot(gs[1,:]);band_x=np.arange(rho.shape[1])
    for i in range(4):axis.plot(band_x,rho[i].cpu().numpy(),marker='o',label=labels[i])
    axis.axhline(0,color='black',lw=.7);axis.set_xticks(band_x,[f'Frozen band {i+1}' for i in band_x]);axis.set_ylabel('Mean member signed CO–U_1 band rho');axis.legend(ncol=4,fontsize=8);axis.grid(alpha=.15)
    for i in range(4):
        axis=fig.add_subplot(gs[2,i]);axis.imshow(np.ma.masked_where(~valid_mask,excursion[i]),origin='lower',cmap='gray_r',vmin=0,vmax=1,aspect='auto')
        axis.set_xticks([]);axis.set_yticks([]);axis.set_title(labels[i] + ('' if i == 0 else ' · member 1'),fontsize=8)
        if i:
            rmse=float(np.sqrt(np.mean((excursion[i][valid_mask].astype(float)-excursion[0][valid_mask].astype(float))**2)))
            axis.text(.02,.02,f'Binary-mask RMSE {rmse:.3g}',transform=axis.transAxes,fontsize=7,color='white',bbox={'facecolor':'black','alpha':.7,'pad':2})
    fig.suptitle(f'{case_name}: {sample_id} · fixed-case relationship evidence',fontsize=12)
    save_figure(fig,figdir,f'relationship_{case_name}')
    return {'regional_covariance_members':[values.tolist() for values in cov],
            'mean_member_band_rho':rho.detach().cpu().tolist(),
            'occupation_mass_sums':[float(values.sum()) for values in hist]}

def atlas(selection, source_path, candidate_paths, assessment, figdir, relationship=None):
    import numpy as np
    import torch
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source_receipt=json.loads(source_path.read_text())
    if (set(selection.get('cases', {})) != {'representative', 'source_tail'}
            or selection['source_bank']['sha256'] != sha(source_path) or selection.get('candidates_used') is not False
            or selection.get('test_locked') is not True or source_receipt.get('status') != 'completed'
            or source_receipt.get('kind') != 'source' or source_receipt.get('split') != 'validation'
            or source_receipt.get('members') != 4 or len(source_receipt.get('sample_ids', [])) != 100):
        raise ValueError('SOURCE-defined visual selection does not match the source bank')
    if (assessment['source_bank_sha256'] != sha(source_path) or assessment['visual_selection_sha256'] != sha(selection['_path'])
            or selection['panel']['sha256'] != assessment['panel_sha256']
            or source_receipt['sample_ids'] != assessment['sample_ids']
            or source_receipt['field_order'] != assessment['field_order']
            or source_receipt['source_sha256'] != assessment['source_checkpoint_sha256']):
        raise ValueError('Atlas source or selection differs from terminal assessment')
    receipts={arm:json.loads(path.read_text()) for arm,path in candidate_paths.items()}
    for arm,path in candidate_paths.items():
        r=receipts[arm]
        if (sha(path) != assessment['bank_receipt_sha256'][arm] or r.get('status') != 'completed'
                or r.get('epoch') != assessment['epoch'] or r.get('arm') != arm
                or r.get('source_bank_sha256') != sha(source_path) or r.get('sample_ids') != source_receipt.get('sample_ids')
                or r.get('members') != 4 or r.get('source_checkpoint_sha256') != source_receipt['source_sha256']):
            raise ValueError(f'Atlas candidate bank differs: {arm}')
    fields=assessment['field_order']
    for case_name,case in selection['cases'].items():
        if case_name not in ('representative','source_tail'): raise ValueError('unexpected atlas selection')
        ordinal=case['ordinal']; sid=case['sample_id']
        if source_receipt['sample_ids'][ordinal] != sid: raise ValueError('SOURCE atlas sample ID changed')
        def read_row(path,receipt):
            file=path.parent/f'{ordinal:04d}.pt'
            if sha(file)!=receipt['files_sha256'][ordinal]: raise ValueError('Cached atlas row changed')
            row=torch.load(file,map_location='cpu',weights_only=True)
            if row['sample_id']!=sid: raise ValueError('Atlas row sample mismatch')
            return row
        source=read_row(source_path,source_receipt)
        candidates={arm:read_row(path,receipts[arm]) for arm,path in candidate_paths.items()}
        if len(source['member_seeds']) != 4 or len(set(source['member_seeds'])) != 4:
            raise ValueError('Atlas SOURCE member identities are not independent')
        for child in candidates.values():
            if child.get('member_seeds') != source['member_seeds'] or child.get('sample_index') != source.get('sample_index'):
                raise ValueError('Atlas generation coupling identity changed')
            for key in ('truth_model','coordinates_model','coordinates_raw','native_masses','normalizer_offset','normalizer_scale','obs_indices','obs_values','obs_fields'):
                if not torch.equal(source[key],child[key]): raise ValueError(f'Atlas row changed {key}')
        coords=source['coordinates_raw'].numpy(); weights=source['native_masses'].reshape(-1).double().numpy()
        if coords.ndim != 2 or coords.shape[1]<2 or len(coords)!=len(weights) or np.any(weights<0):
            raise ValueError('Invalid native atlas coordinates or mass')
        mask=(weights>0) & np.isfinite(coords[:,:2]).all(1)
        if not mask.any(): raise ValueError('No valid atlas support')
        w=weights[mask]/weights[mask].sum();scale=source['normalizer_scale'].numpy();offset=source['normalizer_offset'].numpy()
        truth=source['truth_model'][0].numpy()*scale+offset
        predictions=[(source['members_model'][:,0].mean(0).numpy()*scale+offset)]
        predictions.extend(candidates[arm]['members_model'][:,0].mean(0).numpy()*scale+offset for arm in ARMS)
        fig, axes=plt.subplots(len(fields),7,figsize=(22,2.5*len(fields)),constrained_layout=True,squeeze=False)
        for c,field in enumerate(fields):
            arrays=[truth[:,c]]+[p[:,c] for p in predictions]
            if not all(np.isfinite(a[mask]).all() for a in arrays): raise ValueError('Nonfinite atlas field')
            low=min(a[mask].min() for a in arrays); high=max(a[mask].max() for a in arrays)
            errors=[a-truth[:,c] for a in arrays[1:]]
            error_limit=max(float(np.max(np.abs(e[mask]))) for e in errors) or 1.
            rmses=[float(np.sqrt(np.sum(w*e[mask]**2))) for e in errors]
            for j,(axis,values) in enumerate(zip(axes[c,:4],arrays,strict=True)):
                artist=axis.scatter(coords[mask,0],coords[mask,1],c=values[mask],s=.55,cmap='viridis',vmin=low,vmax=high,rasterized=True)
                axis.set_aspect('equal'); axis.set_xticks([]); axis.set_yticks([])
                if j:
                    axis.text(.02,.02,f'RMSE {rmses[j-1]:.4g}',transform=axis.transAxes,fontsize=7,color='white',bbox={'facecolor':'black','alpha':.65,'pad':2})
                if c==0: axis.set_title(('Reference','SOURCE','SSS target student','Native student')[j],fontsize=9)
                if j==0: axis.set_ylabel(field)
            for j,(axis,error) in enumerate(zip(axes[c,4:],errors,strict=True)):
                error_artist=axis.scatter(coords[mask,0],coords[mask,1],c=error[mask],s=.55,cmap='coolwarm',vmin=-error_limit,vmax=error_limit,rasterized=True)
                axis.set_aspect('equal'); axis.set_xticks([]); axis.set_yticks([])
                axis.text(.02,.02,f'RMSE {rmses[j]:.4g}',transform=axis.transAxes,fontsize=7,color='white',bbox={'facecolor':'black','alpha':.65,'pad':2})
                if c==0: axis.set_title(('SOURCE error','SSS student error','Native student error')[j],fontsize=9)
            fig.colorbar(artist,ax=list(axes[c,:4]),shrink=.65,label='Dataset field units')
            fig.colorbar(error_artist,ax=list(axes[c,4:]),shrink=.65,label='Dataset field units')
        fig.suptitle(f'{case_name}: {sid} · ensemble means · native positive-mass support')
        save_figure(fig,figdir,f'atlas_{case_name}')
        if relationship is not None:
            plot_case_relationship(case_name,sid,ordinal,source,candidates,assessment,relationship,figdir)


def render_validate(output, names):
    import mistune
    from PIL import Image
    class Renderer(mistune.HTMLRenderer):
        def heading(self,text,level,**attrs):
            anchor=re.sub(r'[^a-z0-9 -]','',re.sub(r'<[^>]*>','',text).lower()).replace(' ','-')
            return f'<h{level} id="{anchor}">{text}</h{level}>\n'
    markdown=(output/'report.md').read_text()
    rendered=mistune.create_markdown(renderer=Renderer(),plugins=['table'])(markdown)
    (output/'report.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Round 4 report</title><style>img {max-width:100%;height:auto} table {border-collapse:collapse} td,th {padding:.35rem;border-bottom:1px solid #ddd}</style></head><body>'+rendered+'</body></html>')
    for name in names:
        png=output/'figures'/f'{name}.png'; pdf=output/'figures'/f'{name}.pdf'
        with Image.open(png) as im: im.verify()
        info=subprocess.run(['pdfinfo',str(pdf)],check=True,capture_output=True,text=True).stdout
        anchor=f'figure-{name.replace("_","-")}'
        if not re.search(r'^Pages:\s+[1-9][0-9]*$',info,re.M) or f'href="#{anchor}"' not in rendered or f'id="{anchor}"' not in rendered:
            raise ValueError(f'Invalid figure or index: {name}')
        if html.escape(f'figures/{name}.png') not in rendered or html.escape(f'figures/{name}.pdf') not in rendered:
            raise ValueError(f'Invalid relative figure links: {name}')
    if re.search(r'!\[[^\]]*\]\([^)]*\.pdf\)',markdown): raise ValueError('PDF embedded instead of linked')


def produce_cost_rejection(inputs, input_path, campaign, output):
    """Report a documented budget stop without constructing an efficacy gate."""
    import torch
    from scripts.posttraining.run_car_round4_teacher import validate_resume_identity, _semantic_equal
    prohibited=('teacher_gate','assessment','decision','visual_selection','source_bank_receipt',
                'student_bank_receipts','observer_freeze')
    if any(inputs.get(key) is not None for key in prohibited):
        raise ValueError('Partial cost rejection cannot contain teacher utility gate, student or task evidence')
    required={'schema','mode','train_manifest','teacher_checkpoint','teacher_history','teacher_scientific_config',
              'budget_stop_decision','forecast','costs'}
    if set(inputs)!=required:
        raise ValueError('Partial cost rejection inputs must contain only the declared bound evidence')
    manifest=document(inputs['train_manifest'],'car_round4_training_manifest_v1')
    forecast=document(inputs['forecast'],'car_round4_feasibility_gate_v1')
    config=document(inputs['teacher_scientific_config'],'car_round4_teacher_scientific_config_v1')
    decision=document(inputs['budget_stop_decision'],'car_round4_teacher_budget_stop_v1')
    cost=document(inputs['costs'],'car_round4_teacher_cost_rejected_closeout_v1')
    checkpoint_path=bound(inputs['teacher_checkpoint']); history_path=bound(inputs['teacher_history'])
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True,mmap=True)
    age=checkpoint.get('completed_epochs')
    if not isinstance(age,int) or isinstance(age,bool) or not 1<=age<100:
        raise ValueError('Cost rejection requires a partial completed teacher age below 100')
    expected={'train_manifest_sha256':inputs['train_manifest']['sha256'], 'completed_epochs':age}
    for key,input_key in (('checkpoint','teacher_checkpoint'),('history','teacher_history'),
                          ('scientific_config','teacher_scientific_config')):
        if (decision.get(key)!=inputs[input_key] or not Path(decision[key]['path']).is_absolute()
                or not bound(decision[key]).is_relative_to(campaign)):
            raise ValueError('Budget-stop decision and partial evidence bindings differ')
    if (not isinstance(decision.get('forecast'),dict) or not Path(decision['forecast'].get('path','')).is_absolute()
            or not bound(decision['forecast']).is_relative_to(campaign)):
        raise ValueError('Warm-cache budget forecast must be campaign-owned')
    warm=document(decision['forecast'],'car_round4_teacher_warm_cache_forecast_v1')
    if (decision.get('decision')!='stop_cost_infeasible' or any(decision.get(k)!=v for k,v in expected.items())
            or forecast.get('manifest_sha256')!=expected['train_manifest_sha256']):
        raise ValueError('Budget-stop decision and partial evidence bindings differ')
    from scripts.posttraining.closeout_car_round4_cost import validate_budget_stop
    validate_budget_stop(decision,warm,expected['train_manifest_sha256'])
    gpu=decision['projected_teacher_gpu_seconds']; cpu=decision['projected_teacher_cpu_worker_seconds']
    warm_expected={'train_manifest_sha256':expected['train_manifest_sha256'], 'completed_epochs':age,
                   'teacher_checkpoint_sha256':inputs['teacher_checkpoint']['sha256'],
                   'teacher_history_sha256':inputs['teacher_history']['sha256'], 'remaining_epochs':100-age,
                   'safety_factor':1.2, 'projected_teacher_gpu_seconds':gpu, 'projected_teacher_cpu_worker_seconds':cpu,
                   'teacher_gpu_cap_seconds':decision['teacher_gpu_cap_seconds'], 'teacher_cpu_cap_seconds':21600}
    if any(warm.get(k)!=v for k,v in warm_expected.items()):
        raise ValueError('Warm-cache forecast differs from the bound budget decision')
    window=warm.get('training_window_epochs')
    if window!=list(range(max(1,age-4),age+1)):
        raise ValueError('Warm-cache forecast must identify the last available five completed epochs')
    if (cost.get('complete') is not True or cost.get('budget_stop_decision_sha256')!=inputs['budget_stop_decision']['sha256']
            or cost.get('checkpoint_sha256')!=inputs['teacher_checkpoint']['sha256']
            or cost.get('history_sha256')!=inputs['teacher_history']['sha256']
            or cost.get('scientific_config_sha256')!=inputs['teacher_scientific_config']['sha256']
            or cost.get('forecast_sha256')!=decision['forecast']['sha256']
            or any(cost.get(k)!=decision[k] for k in ('projected_teacher_gpu_seconds','projected_teacher_cpu_worker_seconds',
                                                       'teacher_gpu_cap_seconds','teacher_cpu_cap_seconds'))
            or cost.get('train_manifest_sha256')!=expected['train_manifest_sha256'] or cost.get('partial_epoch')!=age
            or cost.get('cost_basis') not in ('measured','conservative_upper_bound')):
        raise ValueError('Partial teacher cost receipt and bound budget stop differ')
    pieces=cost.get('gpu_seconds',{})
    if (set(pieces)!={'source_generation','teacher'} or any(not finite(v) or v<0 for v in pieces.values())
            or not finite(cost.get('candidate_overhead_gpu_seconds')) or cost['candidate_overhead_gpu_seconds']<0
            or not finite(cost.get('teacher_cpu_worker_seconds')) or cost['teacher_cpu_worker_seconds']<0):
        raise ValueError('Partial cost receipt lacks measured teacher/source costs')
    if cost['cost_basis']=='conservative_upper_bound' and not cost.get('bounded_unknown_components'):
        raise ValueError('Partial upper-bound costs require bounded unknown components')
    rows=history_rows(history_path,age,manifest['population'])
    identity=config.get('identity',{}); saved_identity={k:checkpoint[k] for k in identity if k in checkpoint}
    validate_resume_identity(saved_identity,identity,origin='partial checkpoint')
    if (checkpoint.get('schema')!='car_round4_teacher_checkpoint_v1'
            or checkpoint.get('train_manifest_sha256')!=expected['train_manifest_sha256']
            or checkpoint.get('history')!=rows or not _semantic_equal(checkpoint.get('normalizers'),config.get('normalizers'))
            or checkpoint.get('runtime',{}).get('step')!=12*age):
        raise ValueError('Partial teacher checkpoint, configuration and history differ')
    fields=manifest['field_order']
    if any(len(row[k])!=len(fields) for row in rows for k in ('field_displacement','field_member_spread','min_field_slack')):
        raise ValueError('Partial teacher history field dimensions differ')
    visits=torch.zeros(manifest['population'],dtype=torch.long)
    for row in rows:
        for batch in row['batch_stream']:
            visits[torch.tensor(batch['condition_indices'],dtype=torch.long)]+=1
    if not torch.equal(checkpoint.get('visits'),visits) or not torch.equal(checkpoint.get('stream',{}).get('visits'),visits):
        raise ValueError('Partial checkpoint visits differ from the saved condition stream')
    output.mkdir(parents=True,exist_ok=False); figdir=output/'figures'; figdir.mkdir()
    figure_history(rows,fields,figdir); figure_costs(forecast,cost,figdir,teacher_only=True)
    figures=[('teacher_histories','Sampled teacher training diagnostics and accepted rays through the saved partial age; these are not a final replay or efficacy assessment.'),
             ('inclusive_costs','Initial forecast and accounted partial costs; the planned student and assessment phases were not run. Unknown costs are not zero.')]
    elapsed=cost.get('elapsed_wall_seconds')
    elapsed_text=f'{elapsed:.4g} seconds' if finite(elapsed) else 'unavailable'
    lines=['# Round 4: teacher cost feasibility','',
           '**Decision: retain SOURCE/C8.** The teacher route was stopped for forecast cost feasibility before its 100-epoch endpoint; no efficacy failure is asserted. No student or task assessment was run.','',
           f"The saved teacher completed {age} reporting epochs, {rows[-1]['cumulative_visits']} objective-bearing visits and {rows[-1]['dataset_pass_equivalents']:.3g} TRAIN passes. Curves show sampled training diagnostics only; the final TRAIN800 replay and full-bank feasibility audit were not performed.",'',
           f"The budget-stop forecast projects {gpu/3600:.3g} teacher GPU hours against the {decision['teacher_gpu_cap_seconds']/3600:.3g}-hour effective limit after the declared reserve and candidate/whole budget constraints, and {cpu/3600:.3g} teacher CPU-worker hours against the unchanged 6-hour limit. These are forecasts based on the saved decision, not final measured costs.",'',
           f"The forecast uses measured warm-cache training epochs {json.dumps(window)}, a 1.2 safety factor, and the saved startup, persistence and closing-work allowances. Extrapolation is uncertain and depends on those measured epochs and allowances.",'',
           f"The initial candidate forecast was {forecast['candidate_inclusive_gpu_seconds']/3600:.3g} GPU hours at the planned 500-epoch student endpoint; its initial teacher CPU forecast was {forecast['teacher_cpu_worker_seconds']/3600:.3g} hours. Neither is a measured speedup.",'',
           '## Accounted partial costs','',
           'GPU phase costs (seconds): '+', '.join(f'{k}: {v:.4g}' for k,v in pieces.items())+f"; qualification/failure overhead: {cost['candidate_overhead_gpu_seconds']:.4g}. Cost basis: {cost['cost_basis']}.",'',
           f"Teacher CPU worker time: {cost['teacher_cpu_worker_seconds']:.4g} seconds. Elapsed wall: {elapsed_text}; basis: {cost.get('elapsed_wall_basis','unavailable')}.",'']
    if cost.get('bounded_unknown_components'):
        lines += ['Bounded unknown cost components: '+json.dumps(cost['bounded_unknown_components'],sort_keys=True)+'.','']
    for key,label in (('auxiliary_cpu_work','Auxiliary CPU work'),('auxiliary_cpu_unknown','Unavailable auxiliary clocks')):
        if cost.get(key): lines += [label+': '+json.dumps(cost[key],sort_keys=True)+'. These scopes do not form a complete CPU sum.','']
    lines += ['## Figure index','']+[f'- [Figure: {name.replace("_"," ").title()}](#figure-{name.replace("_","-")})' for name,_ in figures]+['']
    for name,caption in figures:
        lines += [f'## Figure: {name.replace("_"," ").title()}','',f'![{caption}](figures/{name}.png)','','[PDF master](figures/'+name+'.pdf)','',''+caption,'']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    render_validate(output,[name for name,_ in figures])
    result={'schema':'car_round4_report_v1','mode':'teacher_cost_rejected','status':'terminal_evidence_assembled',
            'pending':[],'figures':[name for name,_ in figures],'html_rendered':True,
            'relative_links_and_png_decode_validated':True,'visual_inspection_required':True,'input_manifest_sha256':sha(input_path)}
    (output/'report_receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def produce(input_path, campaign, output):
    import torch
    campaign=campaign.resolve(strict=True); data_root=DATA_ROOT.resolve(strict=True)
    if not campaign.is_relative_to(data_root) or not output.resolve().is_relative_to(campaign/'reports'):
        raise ValueError('Reports must be stored under the DATA2 case campaign reports directory')
    inputs=json.loads(input_path.read_text())
    if inputs.get('schema') != 'car_round4_report_inputs_v1' or inputs.get('mode') not in ('teacher_rejected','student_terminal','teacher_cost_rejected'):
        raise ValueError('Expected a declared terminal Round 4 report input manifest')
    if inputs['mode']=='teacher_cost_rejected':
        return produce_cost_rejection(inputs,input_path,campaign,output)
    mode=inputs['mode']; manifest=document(inputs['train_manifest'],'car_round4_training_manifest_v1')
    gate=document(inputs['teacher_gate'],'car_round4_teacher_qualification_v1')
    checkpoint_path=bound(inputs['teacher_checkpoint']); history_path=bound(inputs['teacher_history'])
    forecast=document(inputs['forecast'],'car_round4_feasibility_gate_v1')
    if (gate.get('completed_epochs') != 100 or gate.get('teacher_checkpoint_sha256') != sha(checkpoint_path)
            or gate.get('train_manifest_sha256') != inputs['train_manifest']['sha256']
            or forecast.get('manifest_sha256') != inputs['train_manifest']['sha256']):
        raise ValueError('Teacher, manifest and forecast identities differ')
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True,mmap=True)
    rows=history_rows(history_path,100,manifest['population'])
    if (checkpoint.get('schema')!='car_round4_teacher_checkpoint_v1' or checkpoint.get('completed_epochs')!=100
            or checkpoint.get('train_manifest_sha256')!=gate['train_manifest_sha256'] or checkpoint.get('history')!=rows):
        raise ValueError('Teacher checkpoint and complete history differ')
    fields=manifest['field_order']
    units=checkpoint.get('normalizers', {})
    panel=checkpoint.get('panel_ids', [])
    if (set(units) != set(FAMILIES) or any(not finite(units[key]) or units[key] <= 0 for key in FAMILIES)
            or len(panel) != 800 or len(set(panel)) != 800 or gate.get('panel_ids') != panel
            or gate.get('replay_seed') != 819103
            or gate.get('feasibility_population') != manifest.get('population')
            or not isinstance(gate.get('violations'), int) or gate['violations'] < 0
            or not finite(gate.get('zero_correction_fraction')) or not 0 <= gate['zero_correction_fraction'] <= 1):
        raise ValueError('Teacher normalizers, replay panel or native feasibility audit is incomplete')
    for family in FAMILIES:
        source, teacher = gate['source_scores'][family], gate['teacher_scores'][family]
        improvement = gate['normalized_improvement'][family]
        if not all(finite(x) for x in (source, teacher, improvement)) or not math.isclose(improvement, (source-teacher)/units[family], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError('Teacher replay score does not match frozen SOURCE normalizer')
    eligible = (gate['violations'] == 0 and all(value >= -.02 for value in gate['normalized_improvement'].values())
                and any(value >= .05 for value in gate['normalized_improvement'].values()))
    if gate.get('admitted') is not eligible:
        raise ValueError('Teacher admission does not match the frozen construction margins')
    field_diagnostics=validate_teacher_field_diagnostics(gate,checkpoint,manifest)
    if any(len(row[key])!=len(fields) for row in rows for key in ('field_displacement','field_member_spread','min_field_slack')):
        raise ValueError('Teacher field order or history dimensions differ')
    if mode=='teacher_rejected':
        if gate.get('admitted') is not False or any(inputs.get(k) is not None for k in ('assessment','decision','visual_selection','source_bank_receipt','student_bank_receipts')):
            raise ValueError('Teacher rejection cannot contain student or task evidence')
    else:
        if gate.get('admitted') is not True: raise ValueError('Student terminal requires an admitted teacher')
    cost=document(inputs.get('costs'), required=False)
    assessment=decision=selection=None; candidate_paths={}; source_path=None; relationship=None
    if mode=='student_terminal':
        assessment=document(inputs['assessment'],'car_round3_matched_validation_v1')
        decision=document(inputs['decision'],'car_round3_matched_validation_v1')
        selection=document(inputs['visual_selection'],'car_round4_source_defined_visual_cases_v1')
        if cost is None or cost.get('schema')!='car_round4_inclusive_cost_v1' or cost.get('complete') is not True:
            raise ValueError('Complete measured candidate cost receipt required')
        if (assessment.get('epoch') not in (100,500) or assessment.get('split') != 'validation'
                or assessment.get('members') != 4 or len(assessment.get('sample_ids', [])) != 100
                or not assessment.get('task_scope','').startswith('reused development tasks')
                or decision.get('round4_decision',{}).get('assessment_sha256') != inputs['assessment']['sha256']
                or decision['round4_decision'].get('cost_receipt_sha256') != inputs['costs']['sha256']
                or cost.get('assessment_sha256') != inputs['assessment']['sha256']
                or cost.get('reference_epoch') != assessment['epoch']
                or cost.get('cost_basis') not in ('measured','conservative_upper_bound')
                or set(assessment.get('candidates',{})) != set(ARMS)):
            raise ValueError('Terminal task assessment, cost and decision do not match')
        pieces=cost.get('candidate_gpu_seconds', {})
        if set(pieces) != {'source_generation','teacher','student','assessment'} or any(not finite(v) or v < 0 for v in pieces.values()):
            raise ValueError('Complete disjoint measured candidate GPU costs required')
        reference_path=bound(cost['reference_receipt'])
        historical_path=(campaign.parent/'round_3/assessment/epoch_500/costs.json').resolve(strict=True)
        if reference_path != historical_path:
            raise ValueError('Matched-age reference cost must be the retained Round 3 cost receipt')
        history=json.loads(reference_path.read_text())['training']['FFM_SourceScaledRisk_Uniform_Repair']['segments']
        reference_seconds=sum(item['occupied_gpu_seconds'] for item in history if item['through_epoch'] <= assessment['epoch'])
        if not finite(reference_seconds) or reference_seconds <= 0 or not math.isclose(reference_seconds,cost['reference_gpu_seconds'],rel_tol=1e-12):
            raise ValueError('Matched-age reference GPU cost differs from the historical receipt')
        from scripts.posttraining.assess_car_round4 import decision as recompute_decision
        overhead=cost.get('candidate_overhead_gpu_seconds')
        if not finite(overhead) or overhead < 0: raise ValueError('Explicit candidate qualification and failure overhead required')
        expected_result=recompute_decision(assessment,sum(pieces.values())+overhead,reference_seconds)
        expected=expected_result['round4_decision']
        if any(decision['candidates'][arm].get('scientific_eligible') != expected_result['candidates'][arm]['scientific_eligible'] for arm in ARMS):
            raise ValueError('Saved candidate scientific eligibility differs from measured gates')
        if any(decision['round4_decision'].get(key) != value for key,value in expected.items()):
            raise ValueError('Saved recommendation differs from its bound measured gates')
        source_path=bound(inputs.get('source_bank_receipt'),required=False)
        candidate_paths={arm:bound(ref) for arm,ref in inputs.get('student_bank_receipts',{}).items()}
        selection['_path']=str(bound(inputs['visual_selection']))
        if inputs.get('observer_freeze') is not None:
            relationship=relationship_context(inputs['observer_freeze'],selection,assessment)
    elif cost is not None and (cost.get('schema')!='car_round4_teacher_closeout_cost_v1' or cost.get('train_manifest_sha256')!=gate['train_manifest_sha256']):
        raise ValueError('Teacher-only measured cost receipt does not match')
    output.mkdir(parents=True,exist_ok=False);figdir=output/'figures';figdir.mkdir()
    figure_history(rows,fields,figdir);figure_teacher_gate(gate,figdir);figure_costs(forecast,cost,figdir,teacher_only=mode=='teacher_rejected')
    figures=[('teacher_histories','Complete sampled teacher V, native field constraints, finite-law spread and accepted-ray histories.'),
             ('teacher_replay','Independent fixed-query TRAIN800 empirical-law teacher replay; positive values improve on SOURCE.'),
             ('inclusive_costs','Prelaunch forecast assumes 500 student epochs and remaining SOURCE preparation; accounted costs use the actual terminal age and declared SOURCE total. An upper bound includes bounded unknown work; unknown quantities are not treated as zero.')]
    pending=[]
    if cost is None or cost.get('complete') is not True: pending.append('complete measured inclusive cost receipt')
    elif mode == 'teacher_rejected':
        if cost.get('cost_basis') not in ('measured','conservative_upper_bound') or any(not finite(cost.get('gpu_seconds', {}).get(k)) or cost['gpu_seconds'][k] < 0 for k in ('source_generation','teacher')) or not finite(cost.get('candidate_overhead_gpu_seconds')) or not finite(cost.get('teacher_cpu_worker_seconds')) or not finite(cost.get('elapsed_wall_seconds')):
            pending.append('measured teacher CPU, wall and source/teacher GPU costs')
    if mode=='student_terminal':
        figure_students(assessment,figdir)
        figures.append(('student_fair_and_learning','Held-out independent-J4 fair State, Scale and Shape changes and complete native RF learning histories. Negative fair-score changes improve on SOURCE.'))
        paired_condition_figure(assessment,selection,figdir)
        figures.append(('paired_condition_blocks','Paired condition differences from SOURCE in chronological order, with means for consecutive blocks of up to 20 sampled frames within each trajectory. Points and block means are descriptive and are not IID confidence intervals.'))
        if source_path is not None and set(candidate_paths)==set(ARMS):
            atlas(selection,source_path,candidate_paths,assessment,figdir,relationship=relationship)
            for key in selection['cases']:
                figures.append((f'atlas_{key}',f'SOURCE-selected {key} case on native positive-mass support. Reconstructions are ensemble means and error panels are prediction minus reference. Displayed RMSE is computed from each plotted field and its reference in dataset field units.'))
                if relationship is not None:
                    figures.append((f'relationship_{key}',f'SOURCE-selected {key} case: regional CO–U_1 occupation is a four-member physical-mass mixture and titles pair mean regional covariance with the previously saved fair task score. Signed band rho uses frozen Scale filters and averages per-member values. The lower panels show member 1 at the zero level of the frozen Shape-standardized CO filtration; binary-mask RMSE is on the declared raster. This is descriptive case evidence, not a population score.'))
            if relationship is None: pending.append('bound frozen observer and regional-task definitions for fixed-case relationship panels')
        else: pending.append('SOURCE and both terminal student bank receipts for fixed-case atlas and relationship panels')
    evidence_scope=('Teacher results are TRAIN-only finite empirical-law construction. TEST remains locked.' if mode=='teacher_rejected'
                    else 'Teacher results are TRAIN-only finite empirical-law construction; student results, where present, use the independent-J4 fair observer on previously accessed VALIDATION100. TEST remains locked.')
    lines=['# Round 4: constrained teacher and native amortization','',
           '**Evidence status:** '+('Terminal evidence assembled; rendered assets require visual inspection.' if not pending else 'Pending: '+', '.join(pending)+'.'),'',
           evidence_scope,'',
           '## Decision','']
    if mode=='teacher_rejected':
        lines += ['The 100-epoch teacher did not meet its declared construction gate. Close this local mechanism and retain SOURCE/C8; students were not qualified to start. No measured 500-epoch speed ratio is claimed because students were not run.','']
    else:
        choice=decision['round4_decision']['recommendation']
        if choice not in (*ARMS,'retain SOURCE/C8'): raise ValueError('Unknown terminal recommendation')
        d=decision['round4_decision']
        lines += [f"Recommendation: **{LABELS.get(choice,choice)}**. The candidate inclusive speed ratio is {'at least ' if cost['cost_basis']=='conservative_upper_bound' else ''}{d['inclusive_speed_ratio']:.3g} at reporting epoch {assessment['epoch']}; the scientific and cost decision follows the saved terminal receipt.",'',
                  f"The teacher student's regional improvement relative to native continuation is {d['teacher_regional_improvement_vs_native']:.4g} normalized units; positive values favor the teacher student. The teacher package {'meets' if d['teacher_scientifically_useful'] else 'does not meet'} the scientific utility requirement and {'meets' if d['acceleration_pass'] else 'does not meet'} the twofold inclusive cost target. Native continuation {'meets' if decision['candidates'][NATIVE]['scientific_eligible'] else 'does not meet'} its scientific gate.",'',
                  'The regional relationship and event are reused, unoptimized development tasks, not independent confirmation. The native control compares the whole teacher package rather than isolating SSS.','']
    lines += [f"The teacher completed {rows[-1]['cumulative_visits']} objective-bearing visits ({rows[-1]['dataset_pass_equivalents']:.3g} TRAIN passes). The final construction audit reported {gate['violations']} native field/sensor violations and zero correction for {gate['zero_correction_fraction']:.2%} of audited conditions.",'',
              f"Forecast inclusive candidate GPU cost: {forecast['candidate_inclusive_gpu_seconds']/3600:.3g} hours at the planned 500-epoch student endpoint; forecast teacher CPU worker cost: {forecast['teacher_cpu_worker_seconds']/3600:.3g} hours. Forecasts are not measured costs.",'']
    lines += teacher_field_tables(field_diagnostics,fields)
    lines += ['## Teacher construction','', '| Family | SOURCE empirical V | Teacher empirical V | SOURCE − teacher / frozen unit |', '| --- | ---: | ---: | ---: |']
    for name,label in zip(FAMILIES,FAMILY_NAMES,strict=True):
        lines.append(f"| {label} | {gate['source_scores'][name]:.5g} | {gate['teacher_scores'][name]:.5g} | {gate['normalized_improvement'][name]:.5g} |")
    lines.append('')
    if mode == 'student_terminal':
        lines += ['## Held-out fields and reused tasks','',
                  'Field values are square roots of the saved physical member MSE averaged over matched conditions and members, in dataset field units. Individual reconstruction panels carry their own plotted-field RMSE.','',
                  '| Field | SOURCE member RMSE | SSS target student | Native student |', '| --- | ---: | ---: | ---: |']
        for index,field in enumerate(fields):
            values=[assessment['source']['risk']['member_mse'][index]] + [assessment['candidates'][arm]['risk']['member_mse'][index] for arm in ARMS]
            if any(not finite(value) or value < 0 for value in values): raise ValueError('Invalid saved field MSE')
            lines.append('| '+field+' | '+' | '.join(f'{math.sqrt(value):.5g}' for value in values)+' |')
        lines += ['', '| Path | Signed-field and sensor pass | Member guard | Regional improvement | Event difference |', '| --- | --- | --- | ---: | ---: |']
        for arm in ARMS:
            row=assessment['candidates'][arm]; fidelity=row['fidelity']; task=row['reused_tasks']
            lines.append(f"| {LABELS[arm]} | {'pass' if fidelity['primary_pass'] else 'fail'} | {'pass' if fidelity['legacy_member_pass'] else 'fail'} | {task['regional_normalized_improvement']:.5g} | {task['event_normalized_difference']:.5g} |")
        lines += ['', 'Regional improvement is relative to SOURCE and positive is better; event difference is relative to SOURCE and positive is worse. The field, core and task margins are development decisions, not significance tests.','']
    if cost and cost.get('complete'):
        measured=cost.get('candidate_gpu_seconds',cost.get('gpu_seconds',{}))
        cost_label='Accounted GPU component upper bounds' if cost.get('cost_basis')=='conservative_upper_bound' else 'Measured GPU components'
        lines += [cost_label+' (seconds): '+', '.join(f'{k}: {v:.4g}' if finite(v) else f'{k}: unavailable' for k,v in measured.items())+'.','']
        overhead=cost.get('candidate_overhead_gpu_seconds')
        lines += [f"Qualification and failed-attempt overhead: {f'{overhead:.4g} seconds' if finite(overhead) else 'unavailable'}, charged once to the candidate. SOURCE preparation includes the declared inseparable SOURCE qualification and RF smoke.",'']
        if cost.get('cost_basis')=='conservative_upper_bound':
            bounds=cost.get('bounded_unknown_components')
            if not bounds: raise ValueError('Conservative cost needs its bounded unknown components')
            cost_interpretation=('The accounted GPU cost is an upper bound.' if mode=='teacher_rejected'
                                 else 'The inclusive GPU cost is an upper bound; the corresponding speed ratio is a lower bound.')
            lines += [cost_interpretation+' Bounded unknown components: '+json.dumps(bounds,sort_keys=True)+'.','']
        cpu=cost.get('teacher_cpu_worker_seconds'); native_cpu=cost.get('native_cpu_worker_seconds'); assessment_cpu=cost.get('assessment_cpu_worker_seconds'); tail=cost.get('assessment_cpu_wall_seconds'); elapsed=cost.get('elapsed_wall_seconds')
        if mode=='teacher_rejected':
            lines += [f"Teacher CPU worker time: {f'{cpu:.4g} seconds' if finite(cpu) else 'unavailable'}; elapsed campaign wall: {f'{elapsed:.4g} seconds' if finite(elapsed) else 'unavailable'}. Native student/control training and held-out assessment costs were not incurred. TRAIN800 endpoint replays and the full-bank feasibility audit are included in the teacher leases; their separate subphase timers were not recorded.",'']
            preparation_cpu,current_cpu,numerical_cpu=teacher_preparation_cpu(cost)
            fmt=lambda value: f'{value:.4g} seconds' if finite(value) else 'unavailable'
            lines += [f"SOURCE qualification and full preparation CPU: {fmt(preparation_cpu)}; current-route timing qualification CPU: {fmt(current_cpu)}; teacher numerical-qualification total CPU: {fmt(numerical_cpu)}. These are scoped separately from teacher CPU worker time.",'']
        else:
            lines += [f"Teacher CPU worker time: {f'{cpu:.4g} seconds' if finite(cpu) else 'unavailable'}; native CPU worker time: {f'{native_cpu:.4g} seconds' if finite(native_cpu) else 'unavailable'}; assessment CPU worker time: {f'{assessment_cpu:.4g} seconds' if finite(assessment_cpu) else 'unavailable'}; assessment CPU scoring wall: {f'{tail:.4g} seconds' if finite(tail) else 'unavailable'}; elapsed campaign wall: {f'{elapsed:.4g} seconds' if finite(elapsed) else 'unavailable'}.",'']
        if cost.get('elapsed_wall_basis'):
            lines += ['Elapsed wall basis: '+str(cost['elapsed_wall_basis'])+'.','']
        auxiliary=cost.get('auxiliary_cpu_work', [])
        if auxiliary:
            entries=[]
            for item in auxiliary:
                value=item.get('process_cpu_seconds')
                entries.append(str(item.get('scope','unspecified'))+': '+(f'{value:.4g} CPU seconds' if finite(value) else 'CPU time unavailable'))
            lines += ['Auxiliary CPU work: '+'; '.join(entries)+'. These scoped amounts are not a complete CPU sum.','']
        unknown=cost.get('auxiliary_cpu_unknown', [])
        if unknown:
            lines += ['Additional unavailable auxiliary clocks: '+'; '.join(str(item.get('scope','unspecified'))+' ('+str(item.get('timer','unknown'))+')' for item in unknown)+'.','']
    else: lines += ['Measured inclusive costs are pending; no speedup conclusion is made.','']
    lines += ['## Figure index',''] + [f'- [Figure: {name.replace("_"," ").title()}](#figure-{name.replace("_","-")})' for name,_ in figures] + ['']
    for name,caption in figures:
        lines += [f'## Figure: {name.replace("_"," ").title()}','',f'![{caption}](figures/{name}.png)','','[PDF master](figures/'+name+'.pdf)','',''+caption,'']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    render_validate(output,[name for name,_ in figures])
    result={'schema':'car_round4_report_v1','mode':mode,'status':'terminal_evidence_assembled' if not pending else 'pending',
            'pending':pending,'figures':[name for name,_ in figures], 'html_rendered':True,
            'relative_links_and_png_decode_validated':True, 'visual_inspection_required':True,
            'input_manifest_sha256':sha(input_path)}
    (output/'report_receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(produce(args.inputs,args.campaign,args.output),indent=2))

if __name__=='__main__': main()
