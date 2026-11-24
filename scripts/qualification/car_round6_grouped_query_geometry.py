#!/usr/bin/env python3
"""Paired mature B64 shared/eight-group query geometry; diagnostic, not efficacy."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import torch
from phycoflow_reconstruction.coherence.compose import three_family_order
from phycoflow_reconstruction.data.physical_measure import batch_measure_context
from phycoflow_reconstruction.training.fidelity_budget import MeanRiskDifferenceBudget
from phycoflow_reconstruction.training.gradient_balance import _assign_flat_gradient,_flat_gradient
from phycoflow_reconstruction.training.gradients import stable_clip_grad_norm_
from phycoflow_reconstruction.training.multitask import OuterAggregator
from phycoflow_reconstruction.training.post_training import _coherence_objective,_physical_endpoint_risk
from phycoflow_reconstruction.training.retention import post_training_mode
from phycoflow_reconstruction.training.source import set_trainable_scope
from scripts.qualification.car_round5_panel_replay import load_context,batches,replay_members
from scripts.qualification.car_round5_query_parameter_replay import plan
from scripts.qualification.car_round5_query_endpoint_replay import gram_summary
from scripts.qualification.car_round6_joint_audit import sha256,weighted_mean
from scripts.qualification.car_round6_mature_first_batch import flattened


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--parity',type=Path,required=True);p.add_argument('--units',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=torch.device,required=True)
    p.add_argument('--seconds-limit',type=float,required=True);p.add_argument('--max-batches',type=int,choices=(1,13),default=13)
    a=p.parse_args()
    if not os.environ.get('CAR_ROUND6_LEASE_TOKEN') or a.device.type!='cuda':p.error('GPU preparation lease required')
    if a.seconds_limit<=0:p.error('positive time limit required')
    start=time.monotonic();torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=False)
    audit=json.loads(a.audit.read_text());parity=json.loads(a.parity.read_text());unit_record=json.loads(a.units.read_text())
    if (audit['status']!='completed' or not audit['gain_response_qualified'] or audit['calibration']['gamma_star']<=0
        or parity['status']!='completed' or not parity['production_action_improves_beyond_replay_noise']
        or parity['audit_sha256']!=sha256(a.audit)):
        raise ValueError('matched production-verified positive star required; do not force an invalid candidate')
    if unit_record.get('status')!='completed' or unit_record.get('sample_count')!=800 or any(unit_record.get('floor_active',{}).values()):
        raise ValueError('complete floor-free SOURCE TRAIN800 normalization required')
    units=[float(unit_record['units'][x]) for x in ('state','scale','shape')]
    if any(not 0<x<float('inf') for x in units):raise ValueError('positive SSS units required')
    cfg,model,dataset,panel,runtime,scales=load_context(a.run/'resolved_config.yaml',a.device)
    ckpt_path=a.run/'checkpoints/epoch_100.pt'
    panel_ref=cfg['coherence']['scalable_anchored_coherence']['common_core']
    if (sha256(ckpt_path)!=audit['checkpoint_sha256'] or panel['provenance']['source_checkpoint_sha256']!=unit_record['source_checkpoint_sha256']
        or unit_record.get('source_panel_sha256')!=panel_ref['source_panel_sha256']):
        raise ValueError('checkpoint or SOURCE unit identity mismatch')
    ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=False)
    set_trainable_scope(model,cfg['trainable']);model.load_state_dict(ckpt['model'],strict=True);post_training_mode(model,cfg)
    params=[x for x in model.parameters() if x.requires_grad]
    names=[n for n,x in model.named_parameters() if x.requires_grad]
    if names!=json.loads((a.run/'artifacts/trainable_parameters.json').read_text())['names']:raise ValueError('trainable order mismatch')
    opt=torch.optim.AdamW(params,lr=float(cfg['optimization']['lr']),weight_decay=float(cfg['optimization'].get('weight_decay',0)),betas=tuple(cfg['optimization'].get('adam_betas',(0.9,0.999))))
    ref=MeanRiskDifferenceBudget.from_state_dict(ckpt['external_policy']['reference'])
    multiplier=torch.tensor(audit['profile_lambda'],device=a.device,dtype=torch.float64)*audit['calibration']['gamma_star']
    order=three_family_order(runtime.families);counts=[];rows=[];grad_sums=None;batch_grams=[]
    source=batches(dataset,panel,cfg,a.device)
    progress={'status':'running','batches_completed':0,'max_batches':a.max_batches,'checkpoint_sha256':sha256(ckpt_path),
        'audit_sha256':sha256(a.audit),'parity_sha256':sha256(a.parity),'units_sha256':sha256(a.units),
        'seeds':[1,2,3,4],'laws':['shared','condition_groups_8'],'gamma':audit['calibration']['gamma_star'],
        'interpretation':'four common query-seed repetitions on one matched panel, not four datasets',
        'shape_gradient_definition':'weighted Shape family parameter gradient before C0 scaling; identical fixed units across laws',
        'sss_normalization':'positive TRAIN SOURCE per-family absolute-score units from supplied receipt; no native loss is evaluated'}
    def write_progress():
        (a.output/'progress.json').write_text(json.dumps(progress,indent=2,allow_nan=False)+'\n')
    write_progress()
    try:
        for index,batch in enumerate(source):
            if index>=a.max_batches:break
            if time.monotonic()-start>a.seconds_limit:raise TimeoutError('bounded geometry time expired')
            expected=panel['sample_ids'][sum(counts):sum(counts)+len(batch.sample_ids)]
            if list(batch.sample_ids)!=expected:raise ValueError('matched panel order changed')
            batch_start=time.monotonic();runtime.begin_step(index);batch_grad=[];batch_rows=[]
            for seed in (1,2,3,4):
                shared,grouped=plan(runtime,seed,index)
                independent,_=plan(runtime,seed+1000,index)
                laws=[('shared',shared),('condition_groups_8',grouped)]
                if seed==1:laws.append(('shared_replay',shared))
                for law,query in laws:
                    model.load_state_dict(ckpt['model'],strict=True);opt.load_state_dict(deepcopy(ckpt['optimizer']))
                    rollout_seed=47+1_000_003+index;initial,endpoint={},{}
                    _coherence_objective(model,batch,runtime.families,{n:None for n in runtime.families},cfg,step=index,
                        generator=torch.Generator(device=a.device).manual_seed(rollout_seed),family_scales=scales,
                        family_losses=initial,endpoint_context=endpoint,scalable_runtime=runtime,scalable_probe=False)
                    selected=endpoint['coupling_selected_batch'];target=endpoint['target']
                    def score(members,query_draw):
                        context={'sample_ids':batch.sample_ids,'reference_ids':batch.sample_ids,
                            **batch_measure_context(selected),'conditional_draws':(members[1],)}
                        result={}
                        for name in order:
                            value,_,_=runtime._score(name,runtime.families[name],members[0],target,selected.query_coords,context,query_draw)
                            result[name]=value.scalar_loss * scales[name] * float(getattr(runtime.families[name],'family_weight',1.0))
                        return result
                    losses={order[0]:initial[order[0]] * scales[order[0]] * float(getattr(runtime.families[order[0]],'family_weight',1.0))}
                    qloss=score(endpoint['members'],query)
                    losses.update({n:qloss[n] for n in order[1:]})
                    risk=_physical_endpoint_risk(endpoint,dataset,mean_risk=True).mean(0);pre=ref.violations(risk)
                    with torch.no_grad():independent_pre=score(endpoint['members'],independent)
                    family_rows=torch.stack([_flat_gradient(losses[n],params) for n in order]).double()
                    direction,_=OuterAggregator({'method':'torchjd_config'})(family_rows);direction/=ref.objective_scale
                    field=_flat_gradient(torch.dot(multiplier,pre),params).double();composed=direction+field
                    before=flattened(params);_assign_flat_gradient(params,composed)
                    norm=stable_clip_grad_norm_(params,float(cfg['optimization']['grad_clip']))
                    clipped=flattened([p.grad for p in params]);opt.step();displacement=flattened(params)-before
                    with torch.no_grad():
                        members=replay_members(model,endpoint,cfg,rollout_seed)
                        replay=dict(endpoint,prediction=members[0],members=members)
                        post=ref.violations(_physical_endpoint_risk(replay,dataset,mean_risk=True).mean(0))
                        independent_post=score(members,independent)
                    family_products=(family_rows@displacement).cpu().tolist()
                    per_family=[(float(independent_pre[n])-float(independent_post[n]))/units[i] for i,n in enumerate(order)]
                    batch_rows.append({'seed':seed,'law':law,'sample_ids':list(batch.sample_ids),
                        'pre':pre.detach().cpu().tolist(),'post':post.cpu().tolist(),
                        'independent_pre':[float(independent_pre[n]) for n in order],
                        'independent_post':[float(independent_post[n]) for n in order],
                        'independent_normalized_decreases':per_family,'independent_normalized_mean_decrease':float(np.mean(per_family)),
                        'family_products_config':(family_rows@direction).cpu().tolist(),
                        'family_products_composed':(family_rows@composed).cpu().tolist(),
                        'family_products_clipped':(family_rows@clipped).cpu().tolist(),
                        'family_products_actual_displacement':family_products,
                        'field_weighted_actual_product':float(field@displacement),
                        'family_action_ascent':[x>0 for x in family_products],
                        'field_finite_increase':(post>pre.detach()).cpu().tolist(), 'clip_pre_norm':float(norm)})
                    if law!='shared_replay':batch_grad.append(family_rows[2].float().cpu())
                    del endpoint,initial,qloss,losses,pre,risk,family_rows,direction,field,composed,displacement,members
            runtime.finish_step(True)
            grad=torch.stack(batch_grad);gram=(grad.double()@grad.double().T)
            batch_grams.append(gram.tolist())
            if grad_sums is None:grad_sums=torch.zeros_like(grad)
            grad_sums.add_(grad,alpha=len(batch.sample_ids)/800)
            rows.extend(batch_rows);counts.append(len(batch.sample_ids))
            progress.update(batches_completed=index+1,conditions_completed=sum(counts),elapsed_seconds=time.monotonic()-start,
                first_batch_seconds=(time.monotonic()-batch_start if index==0 else progress.get('first_batch_seconds')))
            if index==0:progress['forecast_13_batches_seconds']=progress['elapsed_seconds']+12*progress['first_batch_seconds']
            write_progress()
        if len(counts)!=a.max_batches:raise ValueError('incomplete requested diagnostic panel')
        n=sum(counts);g=(grad_sums.double()@grad_sums.double().T)
        summaries={}
        shared_rows=[r for r in rows if r['law']=='shared']
        shared_counts=[len(r['sample_ids']) for r in shared_rows]
        fixed_pre=weighted_mean([r['pre'] for r in shared_rows],shared_counts)
        fixed_target=-np.maximum(fixed_pre,0)/10
        shared_seed_one=[r for r in shared_rows if r['seed']==1]
        replay=[r for r in rows if r['law']=='shared_replay']
        replay_counts=[len(r['sample_ids']) for r in replay]
        first_mean=float(weighted_mean([r['independent_normalized_mean_decrease'] for r in shared_seed_one],replay_counts))
        replay_mean=float(weighted_mean([r['independent_normalized_mean_decrease'] for r in replay],replay_counts))
        sss_numerical_noise=abs(first_mean-replay_mean)
        for law,indices in (('shared',[0,2,4,6]),('condition_groups_8',[1,3,5,7])):
            chosen=[r for r in rows if r['law']==law]
            law_counts=[len(r['sample_ids']) for r in chosen]
            pre=weighted_mean([r['pre'] for r in chosen],law_counts);post=weighted_mean([r['post'] for r in chosen],law_counts)
            target=fixed_target;weight=np.asarray(ref.risk_scales)/np.asarray(ref.absolute_allowance)
            batch_variance=[gram_summary(torch.tensor(gram,dtype=torch.float64)[indices][:,indices])['centered_gradient_squared_norm'] for gram in batch_grams]
            summaries[law]={'per_batch_shape_gradient_variance_sample_weighted':float(np.average(batch_variance,weights=counts)),
                'panel_mean_shape_gradient_variance':gram_summary(g[indices][:,indices])['centered_gradient_squared_norm'] * (800/n)**2,
                'independent_normalized_sss_decrease':float(weighted_mean([r['independent_normalized_mean_decrease'] for r in chosen],law_counts)),
                'independent_normalized_family_decreases':weighted_mean([r['independent_normalized_decreases'] for r in chosen],law_counts).tolist(),
                'field_response_residual_norm':float(np.linalg.norm(weight*(12*(post-pre)-target))),
                'field_pre':pre.tolist(),'field_post':post.tolist(),'target_drift':target.tolist(),
                'family_action_ascent_counts':np.sum([r['family_action_ascent'] for r in chosen],axis=0).tolist(),
                'field_finite_increase_counts':np.sum([r['field_finite_increase'] for r in chosen],axis=0).tolist()}
        shared_result=summaries['shared'];grouped_result=summaries['condition_groups_8']
        shared_decrease=shared_result['independent_normalized_sss_decrease']
        shared_residual=shared_result['field_response_residual_norm']
        field_noise=float(parity['repeat_replay_noise_norm'])
        nondegenerate=shared_decrease>sss_numerical_noise and shared_residual>field_noise
        geometry_pass=(nondegenerate and grouped_result['per_batch_shape_gradient_variance_sample_weighted']<shared_result['per_batch_shape_gradient_variance_sample_weighted']
            and grouped_result['independent_normalized_sss_decrease']>=0.95*shared_decrease-sss_numerical_noise
            and grouped_result['field_response_residual_norm']<=1.05*shared_residual+field_noise)
        out={**progress,'status' :'completed' if n==800 else 'completed_execution_fixture_only',
            'sample_count':n,'family_order':list(order),'field_order':list(ref.field_names),'units':units,
            'summaries':summaries,'shared_fixed_target_drift':fixed_target.tolist(),
            'sss_replay_discrepancy':sss_numerical_noise,'field_replay_discrepancy':field_noise,
            'geometry_qualification':'pass' if geometry_pass and n==800 else ('inconclusive' if not nondegenerate or n!=800 else 'fail'),
            'geometry_qualification_scope':'required variance and paired-action checks only; cost and learning efficacy are separate',
            'batch_gradient_grams':batch_grams,'panel_gradient_gram':g.tolist(), 'actions':rows,
            'limitation':'fixed mature model and optimizer; paired query seeds are repeated estimator draws, not independent datasets; no efficacy claim',
            'wall_seconds':time.monotonic()-start}
        with (a.output/'geometry.json').open('x') as stream:json.dump(out,stream,indent=2,allow_nan=False);stream.write('\n')
        progress.update(status=out['status'],elapsed_seconds=out['wall_seconds']);write_progress()
    except BaseException as exc:
        progress.update(status='failed',error_type=type(exc).__name__,error=str(exc),elapsed_seconds=time.monotonic()-start);write_progress();raise
    finally:source.close();dataset.close()
if __name__=='__main__':main()
