#!/usr/bin/env python3
"""Verify nominated star action with production's one weighted field VJP on TRAIN800."""
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
from phycoflow_reconstruction.training.fidelity_budget import MeanRiskDifferenceBudget
from phycoflow_reconstruction.training.gradient_balance import _assign_flat_gradient,_flat_gradient
from phycoflow_reconstruction.training.gradients import stable_clip_grad_norm_
from phycoflow_reconstruction.training.multitask import FamilyScalarizer,OuterAggregator
from phycoflow_reconstruction.training.post_training import _coherence_objective,_physical_endpoint_risk
from phycoflow_reconstruction.training.retention import post_training_mode
from phycoflow_reconstruction.training.source import set_trainable_scope
from scripts.qualification.car_round5_panel_replay import load_context,batches,replay_members
from scripts.qualification.car_round6_joint_audit import sha256,weighted_mean
from scripts.qualification.car_round6_mature_first_batch import flattened


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=torch.device,required=True)
    p.add_argument('--seconds-limit',type=float,required=True);p.add_argument('--amplitude',choices=('star','probe'),default='star');a=p.parse_args()
    if not os.environ.get('CAR_ROUND6_LEASE_TOKEN') or a.device.type!='cuda':p.error('GPU preparation lease required')
    started=time.monotonic();torch.set_num_threads(4)
    audit=json.loads(a.audit.read_text())
    if audit['status']!='completed':raise ValueError('completed calibration required')
    if a.amplitude=='star' and (not audit['gain_response_qualified'] or audit['calibration']['gamma_star']<=0):
        raise ValueError('positive response-qualified star required')
    gamma=float(audit['calibration']['gamma_star' if a.amplitude=='star' else 'gamma_probe'])
    if gamma<=0:raise ValueError('positive nominated amplitude required')
    comparison_key='star_actions' if a.amplitude=='star' else 'probe_actions'
    cfg,model,dataset,panel,runtime,scales=load_context(a.run/'resolved_config.yaml',a.device)
    ckpt_path=a.run/'checkpoints/epoch_100.pt'
    if sha256(ckpt_path)!=audit['checkpoint_sha256']:raise ValueError('checkpoint mismatch')
    ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=False)
    set_trainable_scope(model,cfg['trainable']);model.load_state_dict(ckpt['model'],strict=True);post_training_mode(model,cfg)
    params=[p for p in model.parameters() if p.requires_grad]
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    if names!=json.loads((a.run/'artifacts/trainable_parameters.json').read_text())['names']:raise ValueError('trainable order mismatch')
    opt=torch.optim.AdamW(params,lr=float(cfg['optimization']['lr']),weight_decay=float(cfg['optimization'].get('weight_decay',0)),betas=tuple(cfg['optimization'].get('adam_betas',(0.9,0.999))))
    ref=MeanRiskDifferenceBudget.from_state_dict(ckpt['external_policy']['reference'])
    multiplier=torch.tensor(audit['profile_lambda'],device=a.device,dtype=torch.float64)*gamma
    order=three_family_order(runtime.families);counts=[];results=[]
    source=batches(dataset,panel,cfg,a.device)
    try:
        for index,batch in enumerate(source):
            if time.monotonic()-started>a.seconds_limit:raise TimeoutError('bounded parity run expired')
            prior=audit[comparison_key][index]
            if list(batch.sample_ids)!=prior['sample_ids']:raise ValueError('matched panel order changed')
            model.load_state_dict(ckpt['model'],strict=True);opt.load_state_dict(deepcopy(ckpt['optimizer']))
            runtime.begin_step(index);seed=47+1_000_003+index
            losses,endpoint={},{}
            _coherence_objective(model,batch,runtime.families,{n:None for n in runtime.families},cfg,step=index,
                generator=torch.Generator(device=a.device).manual_seed(seed),family_scales=scales,
                family_losses=losses,endpoint_context=endpoint,scalable_runtime=runtime,scalable_probe=False)
            scalar,_=FamilyScalarizer(cfg['optimization']['multitask'].get('family_scalarization',{}))(
                losses,scales=scales,weights={n:float(getattr(f,'family_weight',1.0)) for n,f in runtime.families.items()})
            risk=_physical_endpoint_risk(endpoint,dataset,mean_risk=True).mean(0);e=ref.violations(risk)
            expected=torch.tensor(audit['paired_pre'][index],dtype=torch.float64,device=a.device)
            if not torch.allclose(e,expected,rtol=2e-5,atol=2e-6):raise ValueError('paired pre-action risk mismatch')
            rows=torch.stack([_flat_gradient(scalar[name],params) for name in order]).double()
            direction,_=OuterAggregator({'method':'torchjd_config'})(rows)
            direction/=ref.objective_scale
            weighted=_flat_gradient(torch.dot(multiplier,e),params).double()
            combined=direction+weighted
            saved=torch.load(a.audit.parent/f'batch_{index:02d}.pt',map_location='cpu',weights_only=True)
            separately=(multiplier.cpu() @ saved['field_rows']).to(a.device)
            original_rows=saved['family_rows'].to(a.device)
            family_row_delta=torch.linalg.vector_norm(rows-original_rows,dim=1)
            family_row_norm=torch.linalg.vector_norm(original_rows,dim=1)
            before=flattened(params);_assign_flat_gradient(params,combined)
            norm=stable_clip_grad_norm_(params,float(cfg['optimization']['grad_clip']));opt.step()
            displacement=flattened(params)-before
            with torch.no_grad():
                members=replay_members(model,endpoint,cfg,seed)
                replay=dict(endpoint,prediction=members[0],members=members)
                post=ref.violations(_physical_endpoint_risk(replay,dataset,mean_risk=True).mean(0))
            results.append({'sample_ids':list(batch.sample_ids),'pre':e.detach().cpu().tolist(),'post':post.cpu().tolist(),
                'field_vjp_difference_norm':float(torch.linalg.vector_norm(weighted-separately)),
                'combined_vjp_norm':float(torch.linalg.vector_norm(weighted)),
                'family_row_difference_norms':family_row_delta.cpu().tolist(),
                'original_family_row_norms':family_row_norm.cpu().tolist(),
                'family_actual_products':(rows@displacement).cpu().tolist(),
                'field_actual_products':(saved['field_rows'].to(a.device)@displacement).cpu().tolist(),
                'separate_family_actual_products':prior['family_products_actual_displacement'],
                'separate_field_actual_products':prior['field_products_actual_displacement'],
                'clip_pre_norm':float(norm)})
            counts.append(len(batch.sample_ids));runtime.finish_step(True)
        if sum(counts)!=800:raise ValueError('incomplete matched panel')
        actual=weighted_mean([r['post'] for r in results],counts)
        separate=weighted_mean([r['post'] for r in audit[comparison_key]],counts)
        pre=weighted_mean(audit['paired_pre'],counts)
        zero=np.asarray(audit['calibration']['zero']);target=np.asarray(audit['calibration']['target_drift'])
        weights=np.asarray(ref.risk_scales)/np.asarray(ref.absolute_allowance)
        residual=float(np.linalg.norm(weights*(12*(actual-pre)-target)))
        separate_residual=float(np.linalg.norm(weights*(12*(separate-pre)-target)))
        baseline_residual=float(np.linalg.norm(weights*(12*(zero-pre)-target)))
        response_difference=float(np.linalg.norm(weights*12*(actual-separate)))
        replay_noise=float(audit['replay_discrepancy_norm'])
        drift_baseline=12*(zero-pre)
        probe_response=12*(actual-zero)/gamma
        wr=weights*probe_response
        denominator=float(wr@wr)
        if not np.isfinite(denominator) or denominator==0:raise ValueError('degenerate production response')
        unbounded_gain=float(np.dot(wr,weights*(target-drift_baseline))/denominator)
        recalculated_gain=float(np.clip(unbounded_gain,0,gamma))
        result={'status':'completed','sample_count':800,'audit_sha256':sha256(a.audit),'checkpoint_sha256':sha256(ckpt_path),
            'amplitude':a.amplitude,'gamma':gamma,'field_order':list(ref.field_names),
            'combined_post_mean':actual.tolist(),'separate_post_mean':separate.tolist(),
            'baseline_residual_norm':baseline_residual,'combined_residual_norm':residual,
            'separate_residual_norm':separate_residual,'combined_vs_separate_weighted_response_norm':response_difference,
            'repeat_replay_noise_norm':replay_noise,
            'production_response_per_gain':probe_response.tolist(), 'production_unclipped_gain':unbounded_gain,
            'production_recalculated_gain':recalculated_gain,
            'production_action_improves_beyond_replay_noise':bool(baseline_residual-residual>replay_noise),
            'per_batch':results,'wall_seconds':time.monotonic()-started,
            'interpretation':'replay the already measured amplitude with one multiplier-weighted field VJP and one clipped real AdamW action per independently reset mature checkpoint; no new gain trial'}
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    finally:source.close();dataset.close()
if __name__=='__main__':main()
