#!/usr/bin/env python3
"""Independently reset epoch-500 J2 production actions on a fixed TRAIN800 panel."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ.get('CAR_REPO_ROOT', Path(__file__).resolve().parents[2])).resolve(strict=True)
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
import torch
from phycoflow_reconstruction.coherence.compose import three_family_order
from phycoflow_reconstruction.coherence.query_sampling import full_target_query_batch
from phycoflow_reconstruction.data.physical_measure import batch_measure_context
from phycoflow_reconstruction.data.training_batches import build_training_batch_source
from phycoflow_reconstruction.training.fidelity_budget import MeanRiskDifferenceBudget
from phycoflow_reconstruction.training.gradient_balance import _assign_flat_gradient, _flat_gradient
from phycoflow_reconstruction.training.gradients import stable_clip_grad_norm_
from phycoflow_reconstruction.training.model_lifecycle import load_training_aux_state
from phycoflow_reconstruction.training.multitask import OuterAggregator, batched_family_gradients
from phycoflow_reconstruction.training.post_training import _coherence_objective, _physical_endpoint_risk
from phycoflow_reconstruction.training.retention import post_training_mode
from phycoflow_reconstruction.training.scalable_runtime import ScalableRuntime
from phycoflow_reconstruction.training.source import set_trainable_scope
from scripts.qualification.car_round5_panel_replay import load_context, replay_members, independent_query
from scripts.qualification.car_round6_joint_audit import sha256
from scripts.qualification.car_round6_mature_first_batch import flattened
from phycoflow_reconstruction.training.scalable_fitting import _digest


def validate_source(source):
    if (source.get('schema') != 'car_round2_native_bank_v1' or source.get('status') != 'completed'
            or source.get('split') != 'train' or source.get('population') != 8000
            or source.get('members') != 4 or source.get('native_points') != 40300
            or len(source.get('sample_ids', [])) != 800 or len(set(source['sample_ids'])) != 800):
        raise ValueError('expected completed native SOURCE TRAIN800 J4 bank')


def helper_bindings():
    """Capture imported local helper identity, including transitive helpers."""
    result={}
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if not filename: continue
        path=Path(filename).resolve()
        if path.suffix=='.py' and (ROOT/'scripts/qualification') in path.parents:
            result[str(path.relative_to(ROOT))]=sha256(path)
    return dict(sorted(result.items()))


def full_core_scale(runtime, members, target, selected, sample_ids, *, prefit=False):
    """Exhaustive unary/pair core with unit or fixed prefit coefficients, no triples."""
    family=runtime.families['multiscale_spectrum']
    context={'sample_ids':sample_ids,'reference_ids':sample_ids,**batch_measure_context(selected),
             'conditional_draws':(members[1],),'scoring_mode':runtime.scoring_mode,'field_order':runtime.fields}
    coeff=runtime.coefficients if prefit else {k:1.0 for k in ('scale.signed','scale.joint','scale.log_power_self','scale.rho')}
    result,_,_=runtime.adapter.score('multiscale_spectrum',members[0],target,
        coordinates=selected.query_coords,context=context,batch=full_target_query_batch(len(runtime.fields)),coefficients=coeff)
    return result.scalar_loss


def trained_scale(runtime, members, target, selected, sample_ids, query):
    family=runtime.families['multiscale_spectrum']
    context={'sample_ids':sample_ids,'reference_ids':sample_ids,**batch_measure_context(selected),
             'conditional_draws':(members[1],)}
    result,_,_=runtime._score('multiscale_spectrum',family,members[0],target,selected.query_coords,context,query)
    return result.scalar_loss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',type=torch.device,required=True);p.add_argument('--seconds-limit',type=float,required=True)
    p.add_argument('--max-batches',type=int,choices=(1,13),default=13);a=p.parse_args()
    campaign=a.campaign.resolve(strict=True);round6=(campaign.parent/'round_6').resolve(strict=True)
    if campaign/'assessment' not in a.output.resolve().parents or a.output.exists():raise ValueError('new followup assessment attempt required')
    if not os.environ.get('CAR_ROUND6_LEASE_TOKEN') or a.device.type!='cuda' or a.seconds_limit<=0:raise ValueError('bounded GPU lease required')
    source_path=campaign.parent/'round_2/assessment/source_train_ffm_v1/receipt.json'
    panel_path=campaign.parent/'round_2/configs/panels/train.json'
    source=json.loads(source_path.read_text());panel=json.loads(panel_path.read_text());validate_source(source)
    if sha256(panel_path)!=source['panel_sha256'] or panel['sample_ids']!=source['sample_ids'] or len(panel['sample_indices'])!=800:raise ValueError('fixed TRAIN800 identity differs')
    snap_path=round6/'assessment/Epoch500_Validation100_J4_20261125T042223Z/snapshot/snapshot.json';snap=json.loads(snap_path.read_text())
    run=Path(snap['run']).resolve(strict=True);ckpt_path=Path(snap['checkpoint']['path']).resolve(strict=True)
    if sha256(ckpt_path)!=snap['checkpoint']['sha256']:raise ValueError('epoch500 checkpoint changed')
    freeze_path=round6/'configs/formal_freeze.json';freeze=json.loads(freeze_path.read_text());code=Path(freeze['code_root']);inventory=json.loads((code/'execution_receipt.json').read_text())
    if inventory['sha256']!=freeze['code_sha256'] or sha256(run/'resolved_config.yaml')!=freeze['arms']['FFM_SSS_EpochCalibrated_PI_Repair']['sha256']:raise ValueError('frozen package identity differs')
    for relative,digest in inventory['files'].items():
        if relative.startswith('src/') and sha256(ROOT/relative)!=digest:raise ValueError(f'active scientific code differs from freeze: {relative}')
    for relative,digest in helper_bindings().items():
        if Path(ROOT/relative).resolve()!=Path(__file__).resolve() and inventory['files'].get(relative)!=digest:
            raise ValueError(f'qualification helper differs from freeze: {relative}')
    ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=False)
    if (ckpt['global_step']!=6000 or ckpt['scalable_runtime']['step']!=6000
            or ckpt['external_policy']['completed_updates']!=6000
            or ckpt['external_policy']['method']!='lagrangian'
            or ckpt['external_policy']['settings']['lagrangian']['penalty_interval_updates']!=1):
        raise ValueError('completed epoch500 and active every-action Lagrangian state required')
    torch.set_num_threads(4);start=time.monotonic();a.output.mkdir(parents=True)
    status={'schema':'car_scale_epoch500_actual_adam_action_v1','status':'failed','scope':'independently reset fixed-model fixed-Adam J2 selected-support TRAIN shadows; primal update only, with no controller feedback commit or EMA advancement; raw post-action parameters; not native J4 or a continuation trajectory',
        'checkpoint':{'path':str(ckpt_path),'sha256':sha256(ckpt_path)},'snapshot_sha256':sha256(snap_path),'source_receipt_sha256':sha256(source_path),'panel_sha256':sha256(panel_path),
        'frozen_code_sha256':freeze['code_sha256'],'driver_sha256':sha256(Path(__file__)),'imported_qualification_helpers_sha256':helper_bindings(),'sample_count_planned':800,'max_batches':a.max_batches,'lease_token':os.environ['CAR_ROUND6_LEASE_TOKEN'],
        'field_penalty_convention':'detached current nonnegative multipliers dot signed violations; same primal sign as Cooper inequality Lagrangian',
        'query_law':'saved grouped sampler state for production query; independently seeded same-law grouped query for replay',
        'support_scope':'J2, selected support Q4096 and training-law observations; not native J4 physical support or cached T256 observations','test_locked':True}
    dataset=None;stream=None;rows=[]
    try:
        cfg,model,dataset,_,initial,scales=load_context(run/'resolved_config.yaml',a.device)
        if cfg['rollout']['steps']!=4 or cfg['rollout']['solver']!='euler':raise ValueError('replay helper requires the saved four-step Euler rollout')
        model.load_state_dict(ckpt['model'],strict=True);load_training_aux_state(model,ckpt);set_trainable_scope(model,cfg['trainable']);post_training_mode(model,cfg)
        params=[v for v in model.parameters() if v.requires_grad]
        names=[n for n,v in model.named_parameters() if v.requires_grad]
        if names!=json.loads((run/'artifacts/trainable_parameters.json').read_text())['names']:raise ValueError('parameter order mismatch')
        settings=cfg['coherence']['scalable_anchored_coherence'];alloc=settings['allocation']
        runtime=ScalableRuntime(initial.artifacts,initial.families,steps_per_epoch=12,expected_provenance=initial.artifacts['provenance'],seed=int(cfg['runtime'].get('seed',42))+800048,
             scoring_mode=ckpt['scalable_runtime']['scoring_mode'],allocation_mode=alloc['mode'],uniform_epochs=int(alloc.get('uniform_epochs',0)),condition_query_groups=8)
        prefit_path=Path(settings['common_prefit']['path'])
        if sha256(prefit_path)!=settings['common_prefit']['sha256']:raise ValueError('declared prefit changed')
        runtime.calibration.load_prefit(json.loads(prefit_path.read_text()))
        runtime.load_state_dict(ckpt['scalable_runtime'])
        wanted={'scale.signed':2.0,'scale.joint':1.2835023021293974,'scale.log_power_self':0.5,'scale.rho':0.8190801090444172,'scale.third_order':0.5}
        if (len(runtime.bank['triples'])!=8 or any(not math.isclose(runtime.coefficients[k],v,rel_tol=1e-12) for k,v in wanted.items())
                or runtime.families['multiscale_spectrum'].settings.joint_energy_score_weight!=0.25):
            raise ValueError('checkpoint Scale prefit, joint weight or triple bank differs')
        ref=MeanRiskDifferenceBudget.from_state_dict(ckpt['external_policy']['reference']);mult=torch.tensor(ckpt['external_policy']['multipliers'],device=a.device,dtype=torch.float64)
        order=three_family_order(runtime.families);scale_name='multiscale_spectrum';scales=ckpt['family_scales']
        optimizer=torch.optim.AdamW(params,lr=float(cfg['optimization']['lr']),weight_decay=float(cfg['optimization'].get('weight_decay',0)),betas=tuple(cfg['optimization'].get('adam_betas',(0.9,0.999))))
        index=panel['sample_indices'];stream=build_training_batch_source(dataset,[index[i:i+64] for i in range(0,800,64)],cfg,query_points=4096,device=a.device,start_step=6000)
        torch.cuda.synchronize(a.device);torch.cuda.reset_peak_memory_stats(a.device)
        status['checkpoint_optimizer_param_groups']=[{k:g[k] for k in ('lr','betas','eps','weight_decay','amsgrad','maximize')} for g in ckpt['optimizer']['param_groups']]
        status['checkpoint_optimizer_step_counters']=sorted({float(v['step']) for v in ckpt['optimizer']['state'].values() if 'step' in v})
        status['rollout_validation']='saved four Euler steps match replay_members helper'
        status['data_rng_scope']='batch observations and support materialized by production private per-step generators; restore checkpoint global RNG after materialization'
        for batch_index,batch in enumerate(stream):
            if batch_index>=a.max_batches:break
            batch_started=time.monotonic()
            if time.monotonic()-start>a.seconds_limit:raise TimeoutError('bounded action audit expired')
            ids=list(batch.sample_ids);expected=panel['sample_ids'][batch_index*64:batch_index*64+len(ids)]
            if ids!=expected:raise ValueError('fixed TRAIN800 order differs')
            model.load_state_dict(ckpt['model'],strict=True);load_training_aux_state(model,ckpt);optimizer.load_state_dict(deepcopy(ckpt['optimizer']))
            torch.set_rng_state(ckpt['rng_state']['torch_cpu']);torch.cuda.set_rng_state(ckpt['rng_state']['torch_cuda'][0],device=a.device)
            runtime.load_state_dict(ckpt['scalable_runtime']);runtime.begin_step(6000)
            same=deepcopy(runtime.batch);independent=independent_query(runtime,batch_index)
            losses={};endpoint={};rollout_seed=int(cfg['runtime'].get('seed',42))+1_000_003+6000
            _coherence_objective(model,batch,runtime.families,{n:None for n in runtime.families},cfg,step=6000,
                generator=torch.Generator(device=a.device).manual_seed(rollout_seed),family_scales=scales,
                family_losses=losses,endpoint_context=endpoint,scalable_runtime=runtime,scalable_probe=False)
            selected=endpoint['coupling_selected_batch'];target=endpoint['target'];members=endpoint['members']
            ordered=[losses[n]*scales[n]*float(getattr(runtime.families[n],'family_weight',1.0)) for n in order]
            with torch.no_grad():
                sampled_same=trained_scale(runtime,members,target,selected,ids,same);sampled_ind=trained_scale(runtime,members,target,selected,ids,independent)
            full=full_core_scale(runtime,members,target,selected,ids)
            full_query=full_target_query_batch(len(runtime.fields))
            with torch.no_grad():
                prefit_core=full_core_scale(runtime,members,target,selected,ids,prefit=True)
                prefit_expanded=trained_scale(runtime,members,target,selected,ids,full_query)
            if not torch.allclose(sampled_same,losses[scale_name],rtol=1e-6,atol=1e-7):raise ValueError('production query replay scalar mismatch')
            core_grad=_flat_gradient(full,params).detach().double()
            family_rows=batched_family_gradients(ordered,params).detach().double()
            direction,_=OuterAggregator({'method':'torchjd_config'})(family_rows);direction/=ref.objective_scale
            risk=_physical_endpoint_risk(endpoint,dataset,mean_risk=True).mean(0);error=ref.violations(risk)
            field=_flat_gradient(torch.dot(mult,error),params).detach().double();combined=direction+field
            before=flattened(params);_assign_flat_gradient(params,combined);clip=stable_clip_grad_norm_(params,float(cfg['optimization']['grad_clip']))
            optimizer.step();displacement=flattened(params)-before
            with torch.no_grad():
                after=replay_members(model,endpoint,cfg,rollout_seed);post_endpoint=dict(endpoint,prediction=after[0],members=after)
                risk_after=_physical_endpoint_risk(post_endpoint,dataset,mean_risk=True).mean(0)
                same_after=trained_scale(runtime,after,target,selected,ids,same);ind_after=trained_scale(runtime,after,target,selected,ids,independent)
                full_after=full_core_scale(runtime,after,target,selected,ids)
                prefit_core_after=full_core_scale(runtime,after,target,selected,ids,prefit=True)
                prefit_expanded_after=trained_scale(runtime,after,target,selected,ids,full_query)
            torch.cuda.synchronize(a.device)
            rows.append({'sample_ids':ids,'sample_count':len(ids),'data_step':6000+batch_index,'batch_wall_seconds':time.monotonic()-batch_started,
                'observation_sha256':_digest([selected.obs_indices,selected.obs_field_ids,selected.obs_values,selected.obs_valid_mask]),
                'spatial_query_sha256':_digest([selected.metadata.get('query_indices'),selected.query_coords,selected.query_valid_mask]),'rollout_seed':rollout_seed,'same_query_train_scale_pre':float(sampled_same),'same_query_train_scale_post':float(same_after),
                'independent_query_train_scale_pre':float(sampled_ind),'independent_query_train_scale_post':float(ind_after),
                'Published_Core_Default':{'pre':float(full),'post':float(full_after)},
                'Complete_Core_Prefit':{'pre':float(prefit_core),'post':float(prefit_core_after)},
                'Complete_Expanded_Prefit':{'pre':float(prefit_expanded),'post':float(prefit_expanded_after)},
                'field_risk_pre':risk.detach().cpu().tolist(),'field_risk_post':risk_after.cpu().tolist(),
                'field_violation_pre':error.detach().cpu().tolist(),'field_violation_post':ref.violations(risk_after).cpu().tolist(),
                'full_core_gradient_dot_actual_displacement':float(core_grad@displacement),'family_gradient_dot_actual_displacement':(family_rows@displacement).cpu().tolist(),
                'weighted_field_gradient_dot_actual_displacement':float(field@displacement),'displacement_l2_norm':float(torch.linalg.vector_norm(displacement.double())),'clip_pre_norm':float(clip)})
            del losses,endpoint,members,ordered,full,core_grad,family_rows,direction,error,field,combined,displacement,after
        count=sum(len(x['sample_ids']) for x in rows);assert count==(800 if a.max_batches==13 else 64)
        status.update(status='completed' if count==800 else 'completed_fixture_only',sample_count=count,batches_completed=len(rows),
            field_order=list(ref.field_names),family_order=list(order),actual_multipliers=mult.cpu().tolist(),
            learning_rate=optimizer.param_groups[0]['lr'],results=rows,peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device))
    except BaseException as exc:
        status.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:
        if stream is not None:stream.close()
        if dataset is not None:dataset.close()
        status['elapsed_seconds']=time.monotonic()-start
        (a.output/'receipt.json').write_text(json.dumps(status,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
