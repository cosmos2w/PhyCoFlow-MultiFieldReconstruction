#!/usr/bin/env python3
"""Paired same-state Scale actions on matched TRAIN800, using new-law feedback.

One composite and one joint shadow action are reset to the same starting model,
AdamW state and multipliers for every batch. Endpoint banks support later CPU
Scale decomposition; this is not a trajectory or VALIDATION evaluation.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time

CODE = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('CAR_REPO_ROOT', CODE)).resolve()
sys.path[:0] = [str(CODE/'src'), str(CODE)]
import torch
from phycoflow_reconstruction.coherence.compose import three_family_order
from phycoflow_reconstruction.data.physical_measure import batch_measure_context
from phycoflow_reconstruction.data.training_batches import build_training_batch_source
from phycoflow_reconstruction.evaluation.adaptive_campaign import atomic_json, digest
from phycoflow_reconstruction.training.fidelity_budget import MeanRiskDifferenceBudget
from phycoflow_reconstruction.training.fidelity_quadrature import rotating_field_risks
from phycoflow_reconstruction.training.gradient_balance import _assign_flat_gradient, _flat_gradient
from phycoflow_reconstruction.training.gradients import stable_clip_grad_norm_
from phycoflow_reconstruction.training.model_lifecycle import load_training_aux_state
from phycoflow_reconstruction.training.multitask import OuterAggregator
from phycoflow_reconstruction.training.post_training import _coherence_objective, _conditional_draw_seed, _scheduled_model_lr, _fidelity_reference_binding
from phycoflow_reconstruction.training.scalable_fitting import _digest
from phycoflow_reconstruction.training.source import source_hashes
from scripts.qualification.car_round5_panel_replay import independent_query, replay_members
from scripts.qualification.car_round6_mature_first_batch import flattened
from scripts.qualification.car_round7_training_reference import load_context


def summary(vector):
    return float(torch.linalg.vector_norm(vector.double()).detach().cpu())


def cosine(left, right):
    denominator = summary(left)*summary(right)
    return float(torch.dot(left.double(), right.double()).detach().cpu())/denominator if denominator else None


def score_result(runtime, members, endpoint, name, objective, kappa, *, query=None):
    selected = endpoint['coupling_selected_batch']
    context = {'sample_ids':selected.sample_ids, 'reference_ids':selected.sample_ids,
               **batch_measure_context(selected), 'conditional_draws':tuple(members[1:])}
    previous = runtime.scale_objective, runtime.scale_joint_kappa
    runtime.scale_objective, runtime.scale_joint_kappa = objective, kappa
    try:
        result, _, _ = runtime._score(name, runtime.families[name], members[0], endpoint['target'],
                                       selected.query_coords, context, runtime.batch if query is None else query)
        return result
    finally:
        runtime.scale_objective, runtime.scale_joint_kappa = previous


def score(runtime, members, endpoint, name, objective, kappa):
    return score_result(runtime, members, endpoint, name, objective, kappa).scalar_loss


def shape_summary(result):
    """Record only components returned by this query scorer without extra pairings."""
    diagnostics = result.diagnostics
    return {'scalar': float(result.scalar_loss.detach()),
            'components': {name: {'raw': float(value['raw_scalar_loss']),
                                  'weight': float(value['weight']),
                                  'weighted': float(value['weighted_scalar_contribution'])}
                           for name, value in diagnostics.get('components', {}).items()
                           if value.get('executed', True)},
            'score_definition': diagnostics.get('score_definition'),
            'shape_sampling': diagnostics.get('shape_sampling'),
            'attraction_separation_finite_essential': 'not exposed by sampled scorer without additional pairings'}


def detach_tree(value):
    if isinstance(value, torch.Tensor): return value.detach()
    if isinstance(value, dict): return {key:detach_tree(item) for key,item in value.items()}
    if isinstance(value, list): return [detach_tree(item) for item in value]
    if isinstance(value, tuple): return tuple(detach_tree(item) for item in value)
    return value


def run(args):
    if not os.environ.get('CAR_ROUND7_LEASE_TOKEN') or args.device.type != 'cuda':
        raise ValueError('a measured Round 7 CUDA preparation lease is required')
    if not math.isfinite(args.kappa) or args.kappa <= 0:
        raise ValueError('kappa must be positive and finite')
    args.output.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.time(), time.process_time()
    receipt = {'schema':'car_round7_paired_action_audit_v1', 'status':'failed',
        'config_sha256':digest(args.config), 'reference_sha256':digest(args.reference),
        'parent_checkpoint_sha256':digest(args.parent_checkpoint), 'kappa':args.kappa,
        'scope':'TRAIN800 selected-support J2 and selected-plus-complement field law; independent same-state shadow AdamW actions; no controller/EMA update or learning claim',
        'test_locked':True, 'batches_completed':0, 'sample_count':0}
    dataset = source = None
    try:
        torch.set_num_threads(4)
        torch.manual_seed(47)
        cfg, model, dataset, metadata, identity, panel, runtime = load_context(args.config, args.output, args.device)
        if cfg['rollout']['steps'] != 4 or cfg['rollout']['solver'] != 'euler':
            raise ValueError('member replay requires the frozen four-step Euler law')
        reference_report = json.loads(args.reference.read_text())
        if reference_report['risk_reference_panel'].get('reference_law') != 'training_selected_complement_v1':
            raise ValueError('new training-law reference required')
        binding = _fidelity_reference_binding(cfg,dataset,identity,metadata,runtime,
                                               {n:1.0 for n in runtime.families})
        if reference_report.get('binding') != binding:
            raise ValueError('training reference binding differs from active contract')
        calibration = json.loads(args.calibration.read_text())
        if (calibration.get('status') != 'completed' or calibration.get('reference',{}).get('sha256') != digest(args.reference)
            or not math.isclose(float(calibration.get('kappa',float('nan'))),args.kappa,rel_tol=1e-14)):
            raise ValueError('kappa calibration or referenced field artifact differs')
        receipt['calibration_sha256'] = digest(args.calibration)
        ref = MeanRiskDifferenceBudget.from_state_dict(reference_report['reference'])
        if tuple(ref.field_names) != tuple(dataset.field_names): raise ValueError('field order differs')
        ck = torch.load(args.parent_checkpoint, map_location='cpu', weights_only=False)
        if ck['global_step'] != 6000 or ck['external_policy']['completed_updates'] != 6000:
            raise ValueError('epoch-500 mature checkpoint required')
        parameters = [p for p in model.parameters() if p.requires_grad]
        names = [n for n,p in model.named_parameters() if p.requires_grad]
        parent_names = args.parent_checkpoint.parent.parent/'artifacts/trainable_parameters.json'
        if json.loads(parent_names.read_text())['names'] != names:
            raise ValueError('mature optimizer parameter ordering differs')
        source_model = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        source_aux = deepcopy(model.training_aux_state_dict()) if callable(getattr(model,'training_aux_state_dict',None)) else None
        mature_model = ck['model']
        multipliers = torch.as_tensor(ck['external_policy']['multipliers'], dtype=torch.float64, device=args.device)
        if multipliers.numel() != len(ref.field_names): raise ValueError('field multiplier shape differs')
        order = three_family_order(runtime.families)
        if len(order) != 3 or 'multiscale_spectrum' not in order: raise ValueError('three-family order required')
        opt_cfg = cfg['optimization']
        def make_optimizer(state_name):
            completed = 0 if state_name == 'source' else 6000
            lr = _scheduled_model_lr(opt_cfg, completed, 12)
            if lr is None: lr = float(opt_cfg['lr'])
            optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=float(opt_cfg.get('weight_decay',0)),
                betas=tuple(opt_cfg.get('adam_betas',(0.9,0.999))))
            if state_name == 'mature':
                optimizer.load_state_dict(deepcopy(ck['optimizer']))
                for group in optimizer.param_groups: group['lr'] = lr
            return optimizer
        indices = panel['sample_indices']
        source = build_training_batch_source(dataset, [indices[i:i+64] for i in range(0,800,64)],
                                            cfg, query_points=4096, device=args.device, start_step=0)
        receipt.update(source_identity_before=identity, parent_checkpoint=str(args.parent_checkpoint.resolve()),
            fields=list(ref.field_names), family_order=list(order), source_lr=make_optimizer('source').param_groups[0]['lr'],
            mature_lr=make_optimizer('mature').param_groups[0]['lr'], mature_multipliers=multipliers.cpu().tolist(),
            sampler_scope='same TRAIN calibration sampler and grouped query for both states and arms; not replay of mature trajectory query cursor',
            selected_output_scope='Q4096, J2, training observation law; selected-support bank for subsequent CPU Scale assessment',
            field_contract='selected plus 512 uniform complement; fixed new TRAIN reference and inclusion weighting')
        receipt['shape_query_scope'] = ('One private independently seeded grouped query per batch, reused across states and arms. '
            'No additional endpoint generation; finite/essential and attraction/separation require extra pairings and are not inferred.')
        aggregator = OuterAggregator({'method':'torchjd_config'})
        records = args.output/'batches.jsonl'
        cursor = 0
        with records.open('x') as stream:
            for index, batch in enumerate(source):
                if index >= args.max_batches: break
                if time.time() >= float(os.environ['CAR_ROUND7_LEASE_DEADLINE'])-180:
                    raise TimeoutError('stopping before inclusive lease deadline')
                batch_started = time.monotonic()
                count = len(batch.sample_ids)
                if list(batch.sample_ids) != list(panel['sample_ids'][cursor:cursor+count]):
                    raise ValueError('matched TRAIN800 sample order differs')
                runtime.begin_step(index)
                independent_shape_query = independent_query(runtime,index)
                independent_shape_manifest = asdict(independent_shape_query)
                independent_shape_digest = _digest(independent_shape_manifest)
                for state_name in ('source','mature'):
                    model.load_state_dict(source_model if state_name == 'source' else mature_model, strict=True)
                    if state_name == 'mature': load_training_aux_state(model, ck)
                    elif source_aux is not None: model.load_training_aux_state_dict(deepcopy(source_aux))
                    torch.manual_seed(47)
                    if args.device.type == 'cuda': torch.cuda.manual_seed_all(47)
                    seed = int(cfg['runtime']['seed'])+1_000_003+index
                    losses, endpoint = {}, {}
                    runtime.scale_objective, runtime.scale_joint_kappa = 'composite', 1.0
                    combined_result, _ = _coherence_objective(model,batch,runtime.families,{n:None for n in runtime.families},cfg,
                        step=index,generator=torch.Generator(device=args.device).manual_seed(seed),
                        family_scales={n:1.0 for n in runtime.families}, family_losses=losses,
                        endpoint_context=endpoint, scalable_runtime=runtime, scalable_probe=False)
                    selected = endpoint['coupling_selected_batch']
                    joint = score(runtime, endpoint['members'], endpoint, 'multiscale_spectrum','joint_anchor',args.kappa)
                    pre_risks, complement = rotating_field_risks(model, endpoint, dataset, cfg,
                        absolute_step=index, rollout_seed=seed, device=args.device)
                    risk = pre_risks.mean(0)
                    violation = ref.violations(risk)
                    weighted_losses = {n: losses[n]*float(getattr(runtime.families[n], 'family_weight', 1.0)) for n in order}
                    joint_weighted = joint*float(getattr(runtime.families['multiscale_spectrum'], 'family_weight', 1.0))
                    rows = {n:_flat_gradient(weighted_losses[n], parameters).detach().double() for n in order}
                    joint_row = _flat_gradient(joint_weighted, parameters).detach().double()
                    active = torch.zeros_like(multipliers) if state_name == 'source' else multipliers
                    field_row = (_flat_gradient(torch.dot(active.to(violation),violation),parameters).detach().double()
                                 if bool((active != 0).any()) else torch.zeros_like(joint_row))
                    before_values = {n:float(losses[n].detach()) for n in order}
                    before_values['joint_anchor_scaled'] = float(joint.detach())
                    # The combined result carries the original family's detached
                    # component metadata; its scalar is the unweighted family loss.
                    fixed_shape_diagnostics = combined_result.diagnostics['families']['topology']
                    fixed_shape_pre = {'scalar':float(losses['topology'].detach()),
                        'components': {name: {'raw':float(value['raw_scalar_loss']), 'weight':float(value['weight']),
                                              'weighted':float(value['weighted_scalar_contribution'])}
                                       for name,value in fixed_shape_diagnostics.get('components',{}).items()
                                       if value.get('executed',True)},
                        'score_definition':fixed_shape_diagnostics.get('score_definition'),
                        'shape_sampling':fixed_shape_diagnostics.get('shape_sampling'),
                        'attraction_separation_finite_essential':'not exposed by sampled scorer without additional pairings'}
                    with torch.no_grad():
                        independent_shape_pre = shape_summary(score_result(runtime,endpoint['members'],endpoint,
                            'topology','composite',1.0,query=independent_shape_query))
                    bank = {'schema':'car_round7_selected_action_bank_v1', 'sample_ids':list(batch.sample_ids),
                        'query_indices':selected.metadata['query_indices'].detach().cpu(),
                        'query_coords':selected.query_coords.detach().cpu(), 'query_valid_mask':selected.query_valid_mask.detach().cpu(),
                        'target':endpoint['target'].detach().cpu(),
                        'pre_members':torch.stack([v.detach().cpu() for v in endpoint['members']],dim=0),
                        'member_layout':'J,B,Q,C', 'rollout_seed':seed,
                        'shape_fixed_query':asdict(runtime.batch),
                        'shape_independent_query':independent_shape_manifest}
                    record = {'batch':index,'state':state_name,'sample_count':count,'sample_ids':list(batch.sample_ids),
                        'selected_sha256':_digest(selected.metadata['query_indices']), 'complement_sha256':_digest(complement),
                        'observation_sha256':_digest([selected.obs_indices, selected.obs_field_ids, selected.obs_values, selected.obs_valid_mask]),
                        'member_seeds':[seed,_conditional_draw_seed(seed,1)],
                        'first_member_generator_state_sha256':endpoint['rollout_generator_state_sha256'],
                        'shape_fixed_query_sha256':_digest(asdict(runtime.batch)),
                        'shape_independent_query_sha256':independent_shape_digest,
                        'shape_fixed_pre':fixed_shape_pre, 'shape_independent_pre':independent_shape_pre,
                        'field_risk_pre':risk.detach().cpu().tolist(), 'field_violation_pre':violation.detach().cpu().tolist(),
                        'sampled_family_pre':before_values, 'scale_row_cosine':cosine(rows['multiscale_spectrum'],joint_row),
                        'scale_row_norms':{'composite':summary(rows['multiscale_spectrum']), 'joint_anchor':summary(joint_row)},
                        'family_row_norms':{n:summary(rows[n]) for n in order}, 'weighted_field_row_norm':summary(field_row),
                        'multipliers':active.cpu().tolist(), 'arms':{}}
                    # All parameter rows and pre-action observations are now
                    # saved. Release descriptor and rollout graphs before the
                    # optimizer shadows; retain detached endpoint replay inputs.
                    endpoint['members'] = tuple(value.detach() for value in endpoint['members'])
                    endpoint['prediction'] = endpoint['members'][0]
                    runtime.blocks = detach_tree(runtime.blocks)
                    runtime.difficulties = detach_tree(runtime.difficulties)
                    del weighted_losses, joint_weighted, losses, joint, pre_risks, risk, violation, combined_result
                    for arm in ('composite','joint_anchor'):
                        model.load_state_dict(source_model if state_name == 'source' else mature_model, strict=True)
                        if state_name == 'mature': load_training_aux_state(model, ck)
                        elif source_aux is not None: model.load_training_aux_state_dict(deepcopy(source_aux))
                        torch.manual_seed(47)
                        torch.cuda.manual_seed_all(47)
                        optimizer = make_optimizer(state_name)
                        matrix = torch.stack([joint_row if n == 'multiscale_spectrum' and arm == 'joint_anchor' else rows[n] for n in order])
                        direction, _ = aggregator(matrix)
                        direction = direction/ref.objective_scale + field_row
                        before = flattened(parameters)
                        _assign_flat_gradient(parameters,direction)
                        clip = stable_clip_grad_norm_(parameters,float(opt_cfg['grad_clip']))
                        optimizer.step()
                        displacement = flattened(parameters)-before
                        with torch.no_grad():
                            after = replay_members(model,endpoint,cfg,seed)
                            replay = dict(endpoint,prediction=after[0],members=after)
                            post_risks, post_complement = rotating_field_risks(model,replay,dataset,cfg,
                                absolute_step=index,rollout_seed=seed,device=args.device)
                            if not torch.equal(complement,post_complement): raise ValueError('complement changed on action replay')
                            post = {n:float(score(runtime,after,replay,n,'composite',1.0).detach()) for n in order if n != 'topology'}
                            fixed_shape_post = shape_summary(score_result(runtime,after,replay,'topology','composite',1.0))
                            independent_shape_post = shape_summary(score_result(runtime,after,replay,'topology','composite',1.0,query=independent_shape_query))
                            post['topology'] = fixed_shape_post['scalar']
                            post['joint_anchor_scaled'] = float(score(runtime,after,replay,'multiscale_spectrum','joint_anchor',args.kappa).detach())
                        bank[arm+'_post_members'] = torch.stack([v.detach().cpu() for v in after],dim=0)
                        record['arms'][arm] = {'sampled_family_post':post,'field_risk_post':post_risks.mean(0).cpu().tolist(),
                            'shape_fixed_post':fixed_shape_post, 'shape_independent_post':independent_shape_post,
                            'field_violation_post':ref.violations(post_risks.mean(0)).cpu().tolist(),
                            'row_dot_displacement':{n:float(rows[n]@displacement) for n in order},
                            'joint_row_dot_displacement':float(joint_row@displacement),
                            'weighted_field_row_dot_displacement':float(field_row@displacement),
                            'displacement_norm':summary(displacement),'direction_norm':summary(direction),
                            'clip_input_norm':float(clip),'learning_rate':optimizer.param_groups[0]['lr']}
                        del optimizer,matrix,direction,displacement,after,replay,post_risks
                    bank_path = args.output/f'batch_{index:02d}_{state_name}.pt'
                    torch.save(bank,bank_path)
                    record['endpoint_bank'] = {'path':str(bank_path.resolve()),'sha256':digest(bank_path)}
                    stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
                    del rows,joint_row,field_row,endpoint,bank,selected
                runtime.finish_step(True)
                cursor += count
                receipt.update(batches_completed=index+1,sample_count=cursor)
                receipt['last_batch_wall_seconds'] = time.monotonic()-batch_started
                atomic_json(args.output/'receipt.json',receipt)
        expected = 800 if args.max_batches == 13 else min(64*args.max_batches,800)
        if cursor != expected: raise ValueError('requested matched panel was incomplete')
        receipt.update(status='completed' if cursor == 800 else 'completed_fixture_only',
            batches_sha256=digest(records), source_identity_after=source_hashes(cfg),
            parent_checkpoint_sha256_after=digest(args.parent_checkpoint),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device))
        if receipt['source_identity_after'] != identity or receipt['parent_checkpoint_sha256_after'] != receipt['parent_checkpoint_sha256']:
            raise ValueError('SOURCE or parent bytes changed')
    except BaseException as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        if source is not None: source.close()
        if dataset is not None: dataset.close()
        receipt.update(wall_seconds=time.time()-started,cpu_seconds=time.process_time()-cpu_started)
        atomic_json(args.output/'receipt.json',receipt)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--reference',required=True,type=Path)
    p.add_argument('--calibration',required=True,type=Path)
    p.add_argument('--parent-checkpoint',required=True,type=Path)
    p.add_argument('--kappa',required=True,type=float)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--device',required=True,type=torch.device)
    p.add_argument('--max-batches',type=int,choices=range(1,14),default=13)
    run(p.parse_args())


if __name__=='__main__': main()
