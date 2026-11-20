#!/usr/bin/env python3
"""Paired real-input FFM execution comparison against the historical implementation."""
import argparse, copy, hashlib, json, subprocess, time, types
from pathlib import Path
import torch, yaml
from phycoflow_reconstruction.coherence import build_enabled_families
from phycoflow_reconstruction.coherence.compose import three_family_order
from phycoflow_reconstruction.data.topology_subset import iter_full_pass_indices
from phycoflow_reconstruction.data.training_batches import build_training_batch_source
from phycoflow_reconstruction.training.fidelity_budget import ExternalPolicy, _budget_from_state
from phycoflow_reconstruction.training.multitask import FamilyScalarizer, OuterAggregator
from phycoflow_reconstruction.training.post_training import _coherence_objective, _physical_endpoint_risk
from phycoflow_reconstruction.training.retention import post_training_mode
from phycoflow_reconstruction.training.run_store import load_model_state_strict, load_project_checkpoint
from phycoflow_reconstruction.training.scalable_runtime import ScalableRuntime
from phycoflow_reconstruction.training.source import load_source_model, set_trainable_scope
import phycoflow_reconstruction.training.scalable_runtime as sr
import phycoflow_reconstruction.coherence.query_runtime as qr
import phycoflow_reconstruction.training.fidelity_budget as fb

p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--device',default='cuda:0'); p.add_argument('--steps',type=int,default=12); p.add_argument('--baseline-ref',default='e1366b9'); args=p.parse_args()
run=args.run.resolve(strict=True); cfg=yaml.safe_load((run/'resolved_config.yaml').read_text()); ck=load_project_checkpoint(run/'checkpoints/epoch_000.pt'); device=torch.device(args.device)
if cfg['model']['name'] != 'gl_rbf_cq' or ck['global_step'] != 0: raise ValueError('FFM epoch zero required')
old_source=subprocess.check_output(['git','show',f'{args.baseline_ref}:src/phycoflow_reconstruction/coherence/state_compute.py'],text=True)
old=types.ModuleType('phycoflow_reconstruction.coherence._historical_state'); old.__package__='phycoflow_reconstruction.coherence'; exec(compile(old_source,'historical_state.py','exec'),old.__dict__)
new_group=sr.group_projection_scores; new_score=qr.score; assign=fb._assign_flat_gradient
model,dataset,identity=load_source_model(cfg,device); set_trainable_scope(model,cfg['trainable']); model.rollout_checkpointing=True; params=tuple(p for p in model.parameters() if p.requires_grad)
families=build_enabled_families(cfg['coherence'],dataset.data_spec,dataset.normalizer)
for name,family in families.items():
    family.to(device).load_state_artifact(torch.load(run/'artifacts'/f'{name}_family.pt',map_location=device,weights_only=True)); family.load_state_dict(ck['family_states'][name])
artifact=torch.load(run/'artifacts/scalable_coherence.pt',map_location='cpu',weights_only=True); saved=ck['scalable_runtime']; seed=int(cfg['runtime']['seed']); epoch_steps=int(saved['steps_per_epoch'])
if not 1 <= args.steps <= epoch_steps: raise ValueError('qualification steps must fit one reporting epoch')
scalarizer=FamilyScalarizer(cfg['optimization']['multitask']['family_scalarization']); aggregator=OuterAggregator(cfg['optimization']['multitask']['coherence_aggregation']); names=list(three_family_order(families))
def arm(legacy):
    sr.group_projection_scores=old.group_projection_scores if legacy else new_group
    def score(*a,**kw):
        if legacy: a[5].pop('_reuse_scale_for_expanded',None)
        return new_score(*a,**kw)
    qr.score=score
    load_model_state_strict(model,ck['model']); post_training_mode(model,cfg)
    optimizer=torch.optim.AdamW(params,lr=float(cfg['optimization']['lr']),weight_decay=float(cfg['optimization'].get('weight_decay',0))); optimizer.load_state_dict(copy.deepcopy(ck['optimizer']))
    policy=ExternalPolicy(cfg['optimization']['multitask']['external_policy'],_budget_from_state(ck['external_policy']['reference']),device); policy.load_state_dict(copy.deepcopy(ck['external_policy']))
    rt=ScalableRuntime(artifact,families,steps_per_epoch=epoch_steps,expected_provenance=artifact['provenance'],seed=seed+800048,scoring_mode=saved['scoring_mode'],core_only=saved['core_only'],allocation_mode=saved['allocation_mode'],uniform_epochs=saved['uniform_epochs']); rt.calibration.load_prefit(saved['calibration']['artifact']); rt.load_state_dict(copy.deepcopy(saved))
    indices=iter_full_pass_indices(len(dataset),args.steps,64,seed=seed+17,start_step=0); source=build_training_batch_source(dataset,indices,cfg,query_points=4096,device=device,start_step=0)
    torch.set_rng_state(ck['rng_state']['torch_cpu']); torch.cuda.set_rng_state_all(ck['rng_state']['torch_cuda']); capture={}; durations=[]; first=None
    def save_direction(parameters,direction):
        if first is None: capture['direction']=direction.detach().cpu().clone()
        return assign(parameters,direction)
    fb._assign_flat_gradient=save_direction
    arm_started=time.perf_counter()
    try:
        for step,batch in enumerate(source):
            if step>=args.steps: break
            torch.cuda.synchronize(); start=time.perf_counter(); rt.begin_step(step); losses={}; endpoint={}
            _coherence_objective(model,batch,families,{k:None for k in families},cfg,step=step,generator=torch.Generator(device=device).manual_seed(seed+1_000_003+step),family_scales=ck['family_scales'],family_losses=losses,endpoint_context=endpoint,scalable_runtime=rt)
            objectives,_=scalarizer(losses,scales=ck['family_scales'],weights={n:float(getattr(f,'family_weight',1.0)) for n,f in families.items()})
            risks=_physical_endpoint_risk(endpoint,dataset,mean_risk=True).mean(0)
            before=[x.detach().cpu().clone() for x in params] if step==0 else None
            values={n:float(objectives[n].detach()) for n in names} if step==0 else None
            if step == 0:
                torch.cuda.synchronize(); extra_started=time.perf_counter()
                endpoint_grads={n:torch.stack([g.detach().cpu() for g in torch.autograd.grad(objectives[n],endpoint['members'],retain_graph=True)]) for n in ('global_distribution','multiscale_spectrum')}
                blocks={n:float(v.detach()) for n,v in rt.blocks.items() if n in ('state.group_projection','scale.third_order')}
                torch.cuda.synchronize(); start += time.perf_counter()-extra_started
            report=policy.update_with_coherence_aggregation(model,optimizer,{n:objectives[n] for n in names},risks,cfg['optimization'].get('grad_clip'),aggregator)
            rt.finish_step(report['update_accepted']); torch.cuda.synchronize(); durations.append(time.perf_counter()-start)
            if step==0: first={'values':values,'blocks':blocks,'endpoint_grads':endpoint_grads,'direction':capture['direction'],'action':torch.cat([(p.detach().cpu()-b).flatten() for p,b in zip(params,before)]),'risks':risks.detach().cpu()}
    finally: source.close(); fb._assign_flat_gradient=assign
    if len(durations) != args.steps:
        raise ValueError("incomplete reporting epoch")
    return first,durations,time.perf_counter()-arm_started
try:
    baseline,bt,baseline_wall=arm(True); optimized,nt,optimized_wall=arm(False)
    for key in ('direction','action','risks'): torch.testing.assert_close(optimized[key],baseline[key],rtol=3e-4,atol=3e-6)
    for key in baseline['values']: torch.testing.assert_close(torch.tensor(optimized['values'][key]),torch.tensor(baseline['values'][key]),rtol=3e-4,atol=3e-6)
    for key in baseline['blocks']: torch.testing.assert_close(torch.tensor(optimized['blocks'][key]),torch.tensor(baseline['blocks'][key]),rtol=3e-4,atol=3e-6)
    for key in baseline['endpoint_grads']: torch.testing.assert_close(optimized['endpoint_grads'][key],baseline['endpoint_grads'][key],rtol=3e-4,atol=3e-6)
    result={'passed':True,'run':str(run),'device':torch.cuda.get_device_name(device),'steps_per_epoch':epoch_steps,'steps':args.steps,'scope':'real-input training updates including loader waits; excludes epoch callbacks and includes first-update equivalence measurements in enclosing times','baseline_seconds':sum(bt),'optimized_seconds':sum(nt),'baseline_enclosing_seconds':baseline_wall,'optimized_enclosing_seconds':optimized_wall,'baseline_update_seconds':bt,'optimized_update_seconds':nt,'family_values_baseline':baseline['values'],'family_values_optimized':optimized['values'],'max_direction_abs':float((baseline['direction']-optimized['direction']).abs().max()),'relative_direction_l2':float((baseline['direction']-optimized['direction']).norm()/baseline['direction'].norm()),'max_adamw_action_abs':float((baseline['action']-optimized['action']).abs().max()),'max_risk_abs':float((baseline['risks']-optimized['risks']).abs().max())}
    result['block_values_baseline']=baseline['blocks']; result['block_values_optimized']=optimized['blocks']; result['endpoint_gradient_max_abs']={k:float((baseline['endpoint_grads'][k]-optimized['endpoint_grads'][k]).abs().max()) for k in baseline['endpoint_grads']}
    result['baseline_ref']=args.baseline_ref; result['baseline_state_sha256']=hashlib.sha256(old_source.encode()).hexdigest(); result['qualification_helper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
finally: dataset.close(); sr.group_projection_scores=new_group; qr.score=new_score; fb._assign_flat_gradient=assign
