#!/usr/bin/env python3
"""Real, bounded SOURCE TRAIN800 units and fixed-input finite-action replays.

Run inside the campaign's measured preparation lease. Reuse frozen panel IDs and
transforms; units and real actions both use fresh training observations.
Partial panels remain execution evidence and cannot qualify a model diagnostic.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import torch
import yaml

from phycoflow_reconstruction.coherence.compose import build_enabled_families, three_family_order
from phycoflow_reconstruction.coherence.query_sampling import ConditionQueryGroups
from phycoflow_reconstruction.data.physical_measure import batch_measure_context
from phycoflow_reconstruction.data.training_batches import build_training_batch_source
from phycoflow_reconstruction.evaluation.adaptive_campaign import atomic_json, digest
from phycoflow_reconstruction.models.flows.pointcloud.adapters.gl_rbf_cq_adapter import _generator_rng_context
from phycoflow_reconstruction.training.direct_supervision import direct_scalar, finite_field_action
from phycoflow_reconstruction.training.gradient_balance import _assign_flat_gradient, _flat_gradient
from phycoflow_reconstruction.training.post_training import _coherence_objective, _physical_endpoint_risk, _conditional_draw_seed
from phycoflow_reconstruction.training.rollout import differentiable_reconstruction
from phycoflow_reconstruction.training.scalable_fitting import _digest, validate_scalable_artifacts
from phycoflow_reconstruction.training.scalable_runtime import ScalableRuntime
from phycoflow_reconstruction.training.source import load_source_model, source_hashes
from scripts.qualification.car_round5_startup import fit_units

LABELS = ('state', 'scale', 'shape')


def load_replay_core(config_path, cfg, *, reference_config=None,
                     reference_config_sha256=None, replay_artifact_sha256=None):
    """Load a hash-bound completed runtime without changing its envelope."""
    config_path = Path(config_path)
    if reference_config is not None:
        reference_config = Path(reference_config)
        if (not reference_config_sha256 or not replay_artifact_sha256
                or digest(reference_config) != reference_config_sha256):
            raise ValueError('reference configuration hash mismatch or missing artifact hash')
        historical_cfg = yaml.safe_load(reference_config.read_text())
        current_cfg = deepcopy(cfg)
        current_cfg['coherence']['scalable_anchored_coherence'].pop('condition_query_groups', None)
        original_cfg = deepcopy(historical_cfg)
        original_cfg['coherence']['scalable_anchored_coherence'].pop('condition_query_groups', None)
        if current_cfg != original_cfg:
            raise ValueError('replay configuration differs from reference beyond query grouping')
        artifact_path = reference_config.parent / 'artifacts/scalable_coherence.pt'
        if digest(artifact_path) != replay_artifact_sha256:
            raise ValueError('completed replay artifact hash mismatch')
    else:
        if reference_config_sha256 is not None or replay_artifact_sha256 is not None:
            raise ValueError('artifact hashes require a reference configuration')
        artifact_path = config_path.parent / 'artifacts/scalable_coherence.pt'
        if not artifact_path.is_file():
            raise ValueError('relocated configuration requires a hash-bound reference configuration')
    core = torch.load(artifact_path, map_location='cpu', weights_only=True)
    validate_scalable_artifacts(core, core['provenance'])
    bank_ref = cfg['coherence']['scalable_anchored_coherence'].get('adaptive', {}).get('bank_artifact')
    if bank_ref:
        if digest(Path(bank_ref['path'])) != bank_ref['sha256']:
            raise ValueError('selected bank changed')
        bank = json.loads(Path(bank_ref['path']).read_text())
        bank = bank.get('bank', bank)
        if 'adaptive_bank' not in core or _digest(core['adaptive_bank']) != _digest(bank):
            raise ValueError('completed replay artifact differs from selected bank')
    return core


def load_context(config_path, device, *, reference_config=None,
                 reference_config_sha256=None, replay_artifact_sha256=None):
    cfg = yaml.safe_load(config_path.read_text())
    model, dataset, metadata = load_source_model(cfg, device)
    model.eval()
    model.rollout_checkpointing = True
    model.rollout_context_cache = cfg['optimization'].get('rollout_execution', {}).get('context_cache', 'none')
    settings = cfg['coherence']['scalable_anchored_coherence']
    declaration = settings['common_core']
    panel_path = Path(declaration['source_panel_path'])
    if digest(panel_path) != declaration['source_panel_sha256']:
        raise ValueError('cached SOURCE panel changed')
    panel = torch.load(panel_path, map_location='cpu', weights_only=True)
    if (panel['schema'] != 'car_source_panel_v1' or len(panel['sample_ids']) != 800
            or len(set(panel['sample_ids'])) != 800
            or panel['source_weight_view'] != 'ema_materialized'
            or panel['source_weight_view_sha256'] != metadata['materialized_view_sha256']
            or panel['parameter_buffer_sha256'] != _digest(model.state_dict())):
        raise ValueError('matched immutable SOURCE TRAIN800 required')
    core = load_replay_core(config_path, cfg, reference_config=reference_config,
        reference_config_sha256=reference_config_sha256, replay_artifact_sha256=replay_artifact_sha256)
    families = build_enabled_families(cfg['coherence'], dataset.data_spec, dataset.normalizer)
    for name, family in families.items():
        ref = cfg['coherence']['frozen_family_artifacts'][name]
        if digest(Path(ref['path'])) != ref['sha256']:
            raise ValueError('frozen family changed')
        family.load_state_artifact(torch.load(ref['path'], map_location='cpu', weights_only=True))
        families[name] = family.to(device)
    runtime = ScalableRuntime(core, families, steps_per_epoch=12,
        expected_provenance=core['provenance'], seed=47+800048, allocation_mode='uniform',
        condition_query_groups=settings.get('condition_query_groups', 1))
    prefit = settings['common_prefit']
    if digest(Path(prefit['path'])) != prefit['sha256']:
        raise ValueError('inner calibration changed')
    runtime.calibration.load_prefit(json.loads(Path(prefit['path']).read_text()))
    scales = {name: 1.0 for name in families}
    return cfg, model, dataset, panel, runtime, scales


def native_value(model, batch, seed):
    generator = torch.Generator(device=batch.query_coords.device).manual_seed(seed)
    with _generator_rng_context(generator, batch.query_coords.device):
        return model.training_loss(batch).total


def batches(dataset, panel, cfg, device):
    ids = panel['sample_indices']
    return build_training_batch_source(dataset, [ids[i:i+64] for i in range(0,800,64)],
        cfg, query_points=4096, device=device, start_step=0)


def fit_source_units(args, ctx):
    cfg, model, dataset, panel, runtime, scales = ctx
    if runtime.condition_query_groups != 1:
        raise ValueError('SOURCE units must be fitted with the original shared query law')
    source = batches(dataset, panel, cfg, args.device)
    output, rows = args.output, []
    output.mkdir(parents=True, exist_ok=False)
    panel_path = output / 'panel.json'
    atomic_json(panel_path, {'split':'train', 'sample_ids':panel['sample_ids'],
        'sample_indices':panel['sample_indices'], 'sampling':'existing reproducible TRAIN800 of TRAIN8000'})
    began, cpu = time.time(), time.process_time()
    try:
        with torch.no_grad(), (output / 'per_condition.jsonl').open('w') as stream:
            cursor = 0
            for index, batch in enumerate(source):
                count = len(batch.sample_ids)
                if list(batch.sample_ids) != panel['sample_ids'][cursor:cursor+count]:
                    raise ValueError('TRAIN800 order differs')
                native = float(native_value(model, batch, 2_000_050+index))
                runtime.begin_step(index)
                values = {}
                original_score = runtime.score_family
                def capture(name, family, prediction, reference, **kwargs):
                    result = original_score(name, family, prediction, reference, **kwargs)
                    label = LABELS[three_family_order(runtime.families).index(name)]
                    values[label] = (result.per_sample_cost * scales[name] *
                        float(getattr(family, 'family_weight', 1.0))).cpu().tolist()
                    return result
                runtime.score_family = capture
                try:
                    _coherence_objective(model,batch,runtime.families,{name:None for name in runtime.families},cfg,
                        step=index,generator=torch.Generator(device=args.device).manual_seed(47+1_000_003+index),
                        family_scales=scales,scalable_runtime=runtime,scalable_probe=False)
                finally:
                    runtime.score_family = original_score
                runtime.finish_step(True)
                for j, sample_id in enumerate(batch.sample_ids):
                    row = {'sample_id':sample_id,'native':native,
                        **{key:values[key][j] for key in LABELS}}
                    rows.append(row); stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
                cursor += count
                atomic_json(output/'progress.json',{'conditions_completed':cursor,'elapsed_seconds':time.time()-began})
                if args.execution_batches and index+1 >= args.execution_batches:
                    break
        full = len(rows) == 800
        units, floors = fit_units({'sample_ids':panel['sample_ids']}, rows) if full else ({}, {})
        receipt = {'schema':'car_round5_source_units_v1','status':'completed' if full else 'execution_only',
            'split':'train','sample_count':len(rows),'members':2,'units':units,'floor_active':floors,
            'panel_sha256':digest(panel_path),'source_checkpoint_sha256':source_hashes(cfg)['checkpoint'],
            'source_panel_sha256':cfg['coherence']['scalable_anchored_coherence']['common_core']['source_panel_sha256'],
            'native_definition':'sample-count-weighted batch mean native RF loss with fresh original observations; repeated within each batch only for exact averaging',
            'sss_definition':'mean absolute per-condition signed SOURCE scores; fresh original observations and J2/four-Euler endpoints; original shared query law and frozen inner coefficients',
            'cached_endpoints_reused':False,
            'cache_disposition':'fixed-256-T SOURCE endpoints do not match the fresh-observation contract; reused panel IDs and frozen transforms only',
            'replay_sha256':digest(output/'per_condition.jsonl'),
            'wall_seconds':time.time()-began,'cpu_process_seconds':time.process_time()-cpu}
        atomic_json(output/'result.json',receipt)
        if full and any(floors.values()): raise ValueError('outer unit floor active; inspect before learning')
    finally:
        source.close(); dataset.close()


def replay_members(model, endpoint, cfg, rollout_seed):
    members = []
    for member in range(2):
        generator = torch.Generator(device=endpoint['prediction'].device)
        if member == 0: generator.set_state(endpoint['rollout_generator_state'])
        else: generator.manual_seed(_conditional_draw_seed(rollout_seed, member))
        members.append(differentiable_reconstruction(model, endpoint['rollout_model_batch'],
            steps=4, solver='euler', generator=generator,
            observation_config=dict(cfg['observation_consistency'])))
    return tuple(members)


def independent_query(runtime, index):
    """Fresh private draws with each live group's unchanged marginal law."""
    draws = []
    for group, source in enumerate((runtime.sampler, *runtime.group_samplers)):
        sampler = deepcopy(source)
        sampler.pending = None
        if hasattr(sampler, '_transaction'):
            sampler._transaction = None
        sampler.rng.seed(93473 + index + group * 7919)
        sampler.inner_rng.seed(93474 + index + group * 7919)
        draws.append(sampler.draw())
    return ConditionQueryGroups(tuple(draws)) if runtime.group_samplers else draws[0]


def correction_settings(evidence_path, *, candidate_only, units_path):
    """Resolve only the plan's single evidence-directed LR/beta correction."""
    settings = {'learning_rates': (5e-6, 5e-5), 'beta1': .9, 'beta2': .999,
                'weight_decay': 1e-6, 'grad_clip': 1.0, 'correction_evidence': None}
    if evidence_path is None:
        return settings
    if not candidate_only:
        raise ValueError('correction replay requires --candidate-only')
    evidence_path = evidence_path.resolve(strict=True)
    evidence = json.loads(evidence_path.read_text())
    if (evidence.get('schema') != 'car_round5_startup_qualification_v1'
            or evidence.get('demonstrated_failure') is not True
            or not isinstance(evidence.get('correction'), str) or not evidence['correction'].strip()
            or evidence.get('units_sha256') != digest(units_path)):
        raise ValueError('correction requires demonstrated, unit-bound startup failure evidence')
    fraction = evidence.get('ascending_proposal_fraction', 0.)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('ascending proposal fraction must be finite and in [0, 1]')
    overshoot = evidence.get('finite_step_overshoot_demonstrated') is True
    beta_zero = fraction > .1
    if not (overshoot or beta_zero):
        raise ValueError('correction evidence must demonstrate overshoot or more than 10% ascending proposals')
    if overshoot:
        settings['learning_rates'] = (2.5e-6, 2.5e-5)
    if beta_zero:
        settings['beta1'] = 0.
    settings['correction_evidence'] = {'path': str(evidence_path), 'sha256': digest(evidence_path)}
    return settings


def replay_optimizer(parameters, method, lr, settings):
    if method == 'Small_Step_Scalar_SGD':
        return torch.optim.SGD(parameters, lr=lr)
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=settings['weight_decay'],
                             betas=(settings['beta1'], settings['beta2']))


def replay_sample_ids(candidate_rows, initial_lr):
    return [sample_id for row in candidate_rows if row['lr'] == initial_lr for sample_id in row['sample_ids']]


def action_replay(args, ctx):
    from phycoflow_reconstruction.training.multitask import OuterAggregator
    cfg, model, dataset, panel, runtime, scales = ctx
    units_record = json.loads(args.units.read_text())
    if units_record['status'] != 'completed' or units_record['sample_count'] != 800:
        raise ValueError('complete SOURCE TRAIN800 units required')
    units = units_record['units']
    settings = correction_settings(args.correction_evidence, candidate_only=args.candidate_only, units_path=args.units)
    reference = json.loads(Path(cfg['optimization']['multitask']['fidelity_budget']['reference_artifact']['path']).read_text())
    source_mse = reference['risk_reference_panel']['source_member_mse']
    parameters = [p for p in model.parameters() if p.requires_grad]
    anchor = deepcopy(model).eval().requires_grad_(False)
    source = batches(dataset, panel, cfg, args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    began, cpu, records = time.time(), time.process_time(), []
    try:
        with (args.output/'actions.jsonl').open('w') as stream:
            for index, batch in enumerate(source):
                seed = 47+1_000_003+index
                native_seed = 47+2_000_003+index
                runtime.begin_step(index)
                endpoint, family_losses = {}, {}
                result, _ = _coherence_objective(model,batch,runtime.families,{name:None for name in runtime.families},cfg,
                    step=index,generator=torch.Generator(device=args.device).manual_seed(seed),
                    family_scales=scales,family_losses=family_losses,endpoint_context=endpoint,
                    scalable_runtime=runtime,scalable_probe=False)
                native = native_value(model,batch,native_seed)
                scalar = direct_scalar(native,family_losses,units,scales,runtime.families)
                family_order = three_family_order(family_losses)
                gradients = torch.stack([_flat_gradient(family_losses[name]*scales[name]*float(getattr(runtime.families[name],'family_weight',1.0)),parameters).detach() for name in family_order])
                scalar_gradient = _flat_gradient(scalar,parameters).detach()
                original = [p.detach().clone() for p in parameters]
                with torch.no_grad():
                    source_risk = _physical_endpoint_risk(dict(endpoint,members=replay_members(anchor,endpoint,cfg,seed)),dataset,mean_risk=True).mean(0)
                    pre = _physical_endpoint_risk(endpoint,dataset,mean_risk=True).mean(0)-source_risk
                    replayed_members = replay_members(model,endpoint,cfg,seed)
                    replay_pre = _physical_endpoint_risk(dict(endpoint,members=replayed_members),dataset,mean_risk=True).mean(0)-source_risk
                    precision = (pre-replay_pre).abs()
                tolerance = torch.maximum(precision*5, torch.tensor(source_mse,device=args.device)*1e-6)
                if bool((tolerance > torch.tensor(source_mse,device=args.device)*1e-4).any()):
                    raise ValueError('deterministic replay error is material to field allowance')
                old_direction = None
                if not args.candidate_only:
                    # Historical SOURCE composition with fresh zero dual moments.
                    old_direction, old_diagnostics = OuterAggregator({'method':'torchjd_config'})._aggregate(gradients.double())
                    old_direction = old_direction / float(reference['reference']['objective_scale'])
                independent_batch = independent_query(runtime, index)
                def score_endpoints(members, query):
                    context={'sample_ids':batch.sample_ids,'reference_ids':batch.sample_ids,
                        **batch_measure_context(endpoint['coupling_selected_batch']),
                        'conditional_draws':(members[1],)}
                    values={}
                    for label,family_name in zip(LABELS,family_order):
                        value,_,_=runtime._score(family_name,runtime.families[family_name],members[0],endpoint['target'],
                            endpoint['coupling_selected_batch'].query_coords,context,query)
                        values[label]=float(value.scalar_loss)*scales[family_name]*float(getattr(runtime.families[family_name],'family_weight',1.0))
                    native_now=float(native_value(model,batch,native_seed))
                    return {'native':native_now,**values,
                        'direct_scalar':native_now/units['native']+sum(values[k]/units[k]/3 for k in LABELS)}
                with torch.no_grad():
                    independent_before=score_endpoints(endpoint['members'],independent_batch)
                    scalar_replay_before=score_endpoints(replayed_members,runtime.batch)['direct_scalar']
                scalar_precision=abs(float(scalar.detach())-scalar_replay_before)
                scalar_tolerance=max(5*scalar_precision,abs(scalar_replay_before)*1e-6)
                directions = [('Direct_Scalar_AdamW',scalar_gradient)]
                if not args.candidate_only:
                    directions = [('Historical_Fresh_SOURCE_ConFIG',old_direction.to(scalar_gradient.dtype)),
                        *directions, ('Small_Step_Scalar_SGD',scalar_gradient)]
                for name, direction in directions:
                    for lr in (settings['learning_rates'] if name != 'Small_Step_Scalar_SGD' else (1e-7,)):
                        with torch.no_grad():
                            for p, old in zip(parameters,original): p.copy_(old)
                        optimizer = replay_optimizer(parameters, name, lr, settings)
                        # A linear local surrogate reuses the exact full model
                        # gradients; it changes neither the optimizer proposal nor
                        # finite post-action measurement at these parameters.
                        offset, surrogate = 0, scalar.detach()
                        for p in parameters:
                            piece=direction[offset:offset+p.numel()].reshape_as(p); offset+=p.numel()
                            term=(p*piece).sum(); surrogate=surrogate+term-term.detach()
                        if name=='Direct_Scalar_AdamW':
                            def excess():
                                return _physical_endpoint_risk(dict(endpoint,members=replay_members(model,endpoint,cfg,seed)),dataset,mean_risk=True).mean(0)-source_risk
                            action=finite_field_action(model,optimizer,surrogate,pre,source_mse,tolerance,excess,grad_clip=1.0)
                        else:
                            optimizer.zero_grad(set_to_none=True); surrogate.backward()
                            before_clip=float(torch.nn.utils.clip_grad_norm_(parameters,1.0)); optimizer.step()
                            action={'update_accepted':True,'action_alpha':1.0,
                                'model_grad_norm':before_clip,'gradient_clip_coefficient':min(1.,1./(before_clip+1e-6))}
                        delta=torch.cat([(p.detach()-old).reshape(-1) for p,old in zip(parameters,original)])
                        dots=(gradients.double()@delta.double()).cpu().tolist()
                        # Replay scoring uses the same saved query without any
                        # sampler advancement; an independent query draw is also
                        # reported. SOURCE/post filter itself is field-only.
                        measured=[]
                        with torch.no_grad():
                            members=replay_members(model,endpoint,cfg,seed)
                            committed_excess=_physical_endpoint_risk(dict(endpoint,members=members),dataset,mean_risk=True).mean(0)-source_risk
                            for query in (runtime.batch,independent_batch):
                                measured.append(score_endpoints(members,query))
                        before={'native':float(native.detach()),**{k:float(family_losses[n].detach())*scales[n]*float(getattr(runtime.families[n],'family_weight',1.0)) for k,n in zip(LABELS,family_order)},'direct_scalar':float(scalar.detach())}
                        row={'sample_ids':list(batch.sample_ids),'sample_count':len(batch.sample_ids),'method':name,'lr':lr,
                            **action,'family_displacement_dots':dict(zip(LABELS,dots)),
                            'family_scalar_gradient_products':dict(zip(LABELS,(gradients.double()@scalar_gradient.double()).cpu().tolist())),
                            'family_config_direction_products':None if old_direction is None else dict(zip(LABELS,(gradients.double()@old_direction.double()).cpu().tolist())),
                            'family_gradient_gram':(gradients.double()@gradients.double().T).cpu().tolist(),
                            'scalar_displacement_dot':float((scalar_gradient.double()*delta.double()).sum()),
                            'committed_displacement_norm':float(torch.linalg.vector_norm(delta.double())),
                            'field_excess_pre':pre.cpu().tolist(),'committed_field_excess':committed_excess.cpu().tolist(),
                            'before':before,'independent_query_before':independent_before,
                            'same_query_after':measured[0],'independent_query_after':measured[1],
                            'scalar_replay_before':scalar_replay_before,'scalar_replay_precision':scalar_precision,
                            'scalar_descent_tolerance':scalar_tolerance,
                            'field_replay_precision':precision.cpu().tolist(),'field_tolerance':tolerance.cpu().tolist(),
                            'query_law':'condition_groups_8' if runtime.condition_query_groups == 8 else 'shared'}
                        records.append(row); stream.write(json.dumps(row,allow_nan=False)+'\n'); stream.flush()
                with torch.no_grad():
                    for p,old in zip(parameters,original):p.copy_(old)
                runtime.finish_step(True)
                del result, native, scalar, gradients, scalar_gradient, endpoint, family_losses
                if args.execution_batches and index+1>=args.execution_batches:break
        candidate=[r for r in records if r['method']=='Direct_Scalar_AdamW']
        ids=replay_sample_ids(candidate, settings['learning_rates'][0])
        complete=ids==panel['sample_ids']
        accepted=[r for r in candidate if r['update_accepted'] and r['action_norm']>0]
        decreases=[r['before']['direct_scalar']-r['same_query_after']['direct_scalar'] for r in accepted]
        median=float(torch.tensor(decreases).median()) if decreases else None
        independent_decreases=[r['independent_query_before']['direct_scalar']-r['independent_query_after']['direct_scalar'] for r in accepted]
        atomic_json(args.output/'result.json',{'schema':'car_round5_action_replay_v1','status':'completed' if complete else 'execution_only',
            'sample_count':len(ids),'units_sha256':digest(args.units),
            'learning_rates':list(settings['learning_rates']),
            'optimizer':{'method':'AdamW','beta1':settings['beta1'],'beta2':settings['beta2'],
                         'weight_decay':settings['weight_decay'],'grad_clip':settings['grad_clip']},
            'correction_evidence':settings['correction_evidence'],
            'query_law':'condition_groups_8' if runtime.condition_query_groups == 8 else 'shared',
            'candidate_only':bool(args.candidate_only),
            'accepted_fraction':sum(bool(r['update_accepted']) for r in candidate)/len(candidate),
            'accepted_descent_fraction':sum(v>0 for v in decreases)/len(decreases) if decreases else 0.,
            'median_scalar_decrease':median,'candidate_owned_moments':False,
            'scalar_descent_tolerance':max(r['scalar_descent_tolerance'] for r in candidate),
            'median_independent_scalar_decrease':float(torch.tensor(independent_decreases).median()) if independent_decreases else None,
            'independent_descent_fraction':sum(v>0 for v in independent_decreases)/len(independent_decreases) if independent_decreases else 0.,
            'field_tolerance_max':[max(r['field_tolerance'][i] for r in records) for i in range(len(source_mse))],
            'limitation':'fixed-SOURCE virtual actions use fresh moments; startup traces must separately qualify candidate-owned moments and resume',
            'wall_seconds':time.time()-began,'cpu_process_seconds':time.process_time()-cpu})
    finally:
        source.close();dataset.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('units','actions'),required=True)
    p.add_argument('--units',type=Path)
    p.add_argument('--device',default='cuda:0',type=torch.device)
    p.add_argument('--execution-batches',type=int,default=0)
    p.add_argument('--candidate-only',action='store_true',help='replay only Direct_Scalar_AdamW at the predeclared two learning rates')
    p.add_argument('--correction-evidence',type=Path,help='recorded startup qualification authorizing the single bounded correction')
    p.add_argument('--reference-config',type=Path,help='original completed run config for a relocated configuration')
    p.add_argument('--reference-config-sha256')
    p.add_argument('--replay-artifact-sha256')
    args=p.parse_args()
    if not os.environ.get('CAR_ROUND5_LEASE_TOKEN'):p.error('run under the measured Round 5 preparation lease')
    torch.set_num_threads(4)
    if args.mode=='actions' and args.units is None:p.error('actions require frozen SOURCE units')
    if args.mode != 'actions' and args.candidate_only:p.error('--candidate-only applies only to actions')
    if args.correction_evidence is not None and (args.mode != 'actions' or not args.candidate_only):
        p.error('--correction-evidence requires actions and --candidate-only')
    if args.correction_evidence is not None:
        correction_settings(args.correction_evidence, candidate_only=args.candidate_only, units_path=args.units)
    ctx=load_context(args.config,args.device,reference_config=args.reference_config,
        reference_config_sha256=args.reference_config_sha256,replay_artifact_sha256=args.replay_artifact_sha256)
    (fit_source_units if args.mode=='units' else action_replay)(args,ctx)

if __name__=='__main__':main()
