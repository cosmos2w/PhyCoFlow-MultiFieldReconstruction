#!/usr/bin/env python3
"""TRAIN800 mature-state paired AdamW calibration, with state reset each action."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import numpy as np
import torch
from phycoflow_reconstruction.coherence.compose import three_family_order
from phycoflow_reconstruction.data.physical_measure import batch_measure_context
from phycoflow_reconstruction.evaluation.adaptive_campaign import atomic_json
from phycoflow_reconstruction.training.fidelity_budget import MeanRiskDifferenceBudget
from phycoflow_reconstruction.training.gradient_balance import _assign_flat_gradient, _flat_gradient
from phycoflow_reconstruction.training.gradients import stable_clip_grad_norm_
from phycoflow_reconstruction.training.multitask import FamilyScalarizer, OuterAggregator
from phycoflow_reconstruction.training.post_training import _coherence_objective, _physical_endpoint_risk
from phycoflow_reconstruction.training.retention import post_training_mode
from phycoflow_reconstruction.training.source import set_trainable_scope
from scripts.qualification.car_round5_panel_replay import load_context, batches, replay_members, independent_query
from scripts.qualification.car_round6_mature_first_batch import flattened
from scripts.qualification.car_round6_joint_audit import coarse_profile, fit_gain, weighted_mean, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--device', required=True, type=torch.device)
    p.add_argument('--seconds-limit', required=True, type=float)
    args = p.parse_args()
    if not os.environ.get('CAR_ROUND6_LEASE_TOKEN') or args.device.type != 'cuda':
        p.error('requires a measured Round 6 CUDA preparation lease')
    if args.seconds_limit <= 0:
        p.error('positive time limit required')
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    ckpt_path = args.run / 'checkpoints/epoch_100.pt'
    config = args.run / 'resolved_config.yaml'
    cfg, model, dataset, panel, runtime, scales = load_context(config, args.device)
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if checkpoint['global_step'] != 1200:
        raise ValueError('checkpoint is not mature epoch 100')
    set_trainable_scope(model, cfg['trainable'])
    model.load_state_dict(checkpoint['model'], strict=True)
    post_training_mode(model, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    names = [n for n,p in model.named_parameters() if p.requires_grad]
    if names != json.loads((args.run/'artifacts/trainable_parameters.json').read_text())['names']:
        raise ValueError('trainable parameter order mismatch')
    optimizer = torch.optim.AdamW(params, lr=float(cfg['optimization']['lr']),
        weight_decay=float(cfg['optimization'].get('weight_decay', 0)),
        betas=tuple(cfg['optimization'].get('adam_betas', (0.9,0.999))))
    reference = MeanRiskDifferenceBudget.from_state_dict(checkpoint['external_policy']['reference'])
    history = [json.loads(line) for line in (args.run/'logs/history.jsonl').read_text().splitlines() if line.strip()]
    profile, z = coarse_profile(history, reference.field_names, reference.absolute_allowance, reference.risk_scales)
    profile_t = torch.tensor(profile, device=args.device, dtype=torch.float64)
    order = three_family_order(runtime.families)
    ratios, counts, records = [], [], []
    source = batches(dataset, panel, cfg, args.device)
    result = {'status':'running', 'checkpoint_sha256':sha256(ckpt_path), 'config_sha256':sha256(config),
        'fields':list(reference.field_names), 'profile_lambda':profile.tolist(), 'profile_z':z.tolist(),
        'panel_size':800, 'device':torch.cuda.get_device_name(args.device), 'attempts':[]}
    atomic_json(args.output/'progress.json', result)
    try:
        for index,batch in enumerate(source):
            if time.monotonic()-started > args.seconds_limit:
                raise TimeoutError('preparation time limit reached')
            count = len(batch.sample_ids)
            if list(batch.sample_ids) != panel['sample_ids'][sum(counts):sum(counts)+count]:
                raise ValueError('TRAIN800 panel mismatch')
            runtime.begin_step(index)
            seed = 47+1_000_003+index
            family_losses, endpoint = {}, {}
            _coherence_objective(model, batch, runtime.families, {name:None for name in runtime.families}, cfg,
                step=index, generator=torch.Generator(device=args.device).manual_seed(seed), family_scales=scales,
                family_losses=family_losses, endpoint_context=endpoint, scalable_runtime=runtime, scalable_probe=False)
            scalar, _ = FamilyScalarizer(cfg['optimization']['multitask'].get('family_scalarization', {}))(
                family_losses, scales=scales, weights={n:float(getattr(f,'family_weight',1.0)) for n,f in runtime.families.items()})
            risk = _physical_endpoint_risk(endpoint, dataset, mean_risk=True).mean(0)
            violation = reference.violations(risk)
            family_rows = torch.stack([_flat_gradient(scalar[n],params) for n in order]).double()
            field_rows = torch.stack([_flat_gradient(violation[i],params) for i in range(len(reference.field_names))]).double()
            direction,_ = OuterAggregator({'method':'torchjd_config'})(family_rows)
            direction /= reference.objective_scale
            h = profile_t @ field_rows
            norm_h = float(torch.linalg.vector_norm(h))
            if not np.isfinite(norm_h) or norm_h <= 0:
                raise ValueError('zero or nonfinite feedback direction')
            ratios.append(float(torch.linalg.vector_norm(direction))/norm_h)
            counts.append(count)
            offset = 0
            leaf_rms = {}
            for name, parameter in zip(names,params):
                width = parameter.numel() * (2 if torch.is_complex(parameter) else 1)
                leaf_rms[name] = float(family_rows[2,offset:offset+width].square().mean().sqrt())
                offset += width
            records.append({'sample_ids':list(batch.sample_ids),'pre':violation.detach().cpu().tolist(),
                            'shape_parameter_leaf_gradient_rms':leaf_rms})
            torch.save({'direction':direction.cpu(), 'field_rows':field_rows.cpu(), 'family_rows':family_rows.cpu(),
                        'sample_ids':list(batch.sample_ids), 'pre':records[-1]['pre']}, args.output/f'batch_{index:02d}.pt')
            runtime.finish_step(True)
            del family_rows, field_rows, direction, h, scalar, family_losses, endpoint, risk, violation
        if sum(counts) != 800 or len(counts) != 13:
            raise ValueError('incomplete matched TRAIN800 panel')
        gamma_probe = float(np.median(ratios))
        result.update(gamma_probe=gamma_probe, batch_ratios=ratios, batch_counts=counts,
                      gradient_pass_seconds=time.monotonic()-started)
        atomic_json(args.output/'progress.json', result)
    finally:
        source.close()
    # A fresh batch source and runtime are required for each action pass so that
    # sampled observations, support, priors and query draws are exactly paired.
    dataset.close()
    del model, optimizer, params, runtime, dataset
    torch.cuda.empty_cache()

    def action_pass(amplitudes):
        current_cfg, model, dataset, current_panel, runtime, scales = load_context(config, args.device)
        if current_panel['sample_ids'] != panel['sample_ids']:
            raise ValueError('panel changed between action passes')
        set_trainable_scope(model, current_cfg['trainable'])
        model.load_state_dict(checkpoint['model'], strict=True)
        post_training_mode(model, current_cfg)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=float(current_cfg['optimization']['lr']),
            weight_decay=float(current_cfg['optimization'].get('weight_decay',0)),
            betas=tuple(current_cfg['optimization'].get('adam_betas',(0.9,0.999))))
        source = batches(dataset, current_panel, current_cfg, args.device)
        actions = [[] for _ in amplitudes]
        try:
            for index,batch in enumerate(source):
                if time.monotonic()-started > args.seconds_limit:
                    raise TimeoutError('preparation time limit reached')
                saved = torch.load(args.output/f'batch_{index:02d}.pt', map_location='cpu', weights_only=True)
                if list(batch.sample_ids) != saved['sample_ids']:
                    raise ValueError('sample IDs changed between passes')
                model.load_state_dict(checkpoint['model'], strict=True)
                runtime.begin_step(index)
                seed = 47+1_000_003+index
                endpoint, losses = {}, {}
                _coherence_objective(model, batch, runtime.families, {name:None for name in runtime.families}, current_cfg,
                    step=index, generator=torch.Generator(device=args.device).manual_seed(seed), family_scales=scales,
                    family_losses=losses, endpoint_context=endpoint, scalable_runtime=runtime, scalable_probe=False)
                pre = reference.violations(_physical_endpoint_risk(endpoint,dataset,mean_risk=True).mean(0)).detach()
                if not torch.allclose(pre.cpu(), torch.tensor(saved['pre'],dtype=torch.float64), rtol=2e-5, atol=2e-6):
                    raise ValueError('pre-action risk changed between paired passes')
                selected = endpoint['coupling_selected_batch']
                query = independent_query(runtime, index)
                def shape_score(members, query_draw):
                    context = {'sample_ids':batch.sample_ids, 'reference_ids':batch.sample_ids,
                        **batch_measure_context(selected), 'conditional_draws':(members[1],)}
                    score, _, _ = runtime._score(order[2], runtime.families[order[2]], members[0],
                        endpoint['target'], selected.query_coords, context, query_draw)
                    return {'per_condition':score.per_sample_cost.detach().cpu().tolist(),
                            'components':score.diagnostics.get('components',{})}
                with torch.no_grad():
                    shape_pre_fixed = shape_score(endpoint['members'], runtime.batch)
                    shape_pre_independent = shape_score(endpoint['members'], query)
                direction = saved['direction'].to(args.device)
                field_rows = saved['field_rows'].to(args.device)
                family_rows = saved['family_rows'].to(args.device)
                for amplitude_index, amplitude in enumerate(amplitudes):
                    model.load_state_dict(checkpoint['model'], strict=True)
                    optimizer.load_state_dict(deepcopy(checkpoint['optimizer']))
                    before = flattened(parameters)
                    composed = direction + float(amplitude) * (profile_t @ field_rows)
                    _assign_flat_gradient(parameters, composed)
                    norm = stable_clip_grad_norm_(parameters, float(current_cfg['optimization']['grad_clip']))
                    clipped = flattened([p.grad for p in parameters])
                    optimizer.step()
                    displacement = flattened(parameters)-before
                    with torch.no_grad():
                        members = replay_members(model, endpoint, current_cfg, seed)
                        replay = dict(endpoint, prediction=members[0], members=members)
                        post = reference.violations(_physical_endpoint_risk(replay,dataset,mean_risk=True).mean(0))
                        shape_fixed = shape_score(members, runtime.batch)
                        shape_independent = shape_score(members, query)
                    actions[amplitude_index].append({'sample_ids':list(batch.sample_ids),
                        'post':post.cpu().tolist(), 'clip_pre_norm':float(norm),
                        'family_products_config':(family_rows @ direction).cpu().tolist(),
                        'family_products_composed':(family_rows @ composed).cpu().tolist(),
                        'family_products_clipped':(family_rows @ clipped).cpu().tolist(),
                        'family_products_actual_displacement':(family_rows @ displacement).cpu().tolist(),
                        'field_products_config':(field_rows @ direction).cpu().tolist(),
                        'field_products_composed':(field_rows @ composed).cpu().tolist(),
                        'field_products_clipped':(field_rows @ clipped).cpu().tolist(),
                        'field_products_actual_displacement':(field_rows @ displacement).cpu().tolist(),
                        'shape_pre_fixed':shape_pre_fixed, 'shape_pre_independent':shape_pre_independent,
                        'shape_post_fixed':shape_fixed, 'shape_post_independent':shape_independent})
                runtime.finish_step(True)
                del endpoint, losses, pre, direction, field_rows, family_rows
        finally:
            source.close()
            dataset.close()
        if any(len(rows) != 13 for rows in actions):
            raise ValueError('incomplete paired action pass')
        return actions

    try:
        zero, probe = action_pass([0, gamma_probe])
        pre = [row['pre'] for row in records]
        calibration = fit_gain(pre, [r['post'] for r in zero], [r['post'] for r in probe], counts,
            gamma_probe, reference.absolute_allowance, reference.risk_scales)
        star = calibration['gamma_star']
        zero_repeat, verified = action_pass([0, star])
        pre_mean = weighted_mean(pre,counts)
        zero_mean = weighted_mean([r['post'] for r in zero],counts)
        repeat_mean = weighted_mean([r['post'] for r in zero_repeat],counts)
        star_mean = weighted_mean([r['post'] for r in verified],counts)
        weights = np.asarray(reference.risk_scales)/np.asarray(reference.absolute_allowance)
        target = np.asarray(calibration['target_drift'])
        baseline_residual = float(np.sum((weights*(12*(zero_mean-pre_mean)-target))**2))
        actual_residual = float(np.sum((weights*(12*(star_mean-pre_mean)-target))**2))
        replay_discrepancy = float(np.linalg.norm(weights*12*(repeat_mean-zero_mean)))
        improvement = float(np.sqrt(baseline_residual)-np.sqrt(actual_residual))
        measured_drift = 12 * (star_mean - pre_mean)
        correction = measured_drift - 12 * (zero_mean - pre_mean)
        required = target - 12 * (zero_mean - pre_mean)
        per_field_recovery = [float(c/r) if r != 0 else None for c,r in zip(correction, required)]
        result.update(status='completed', calibration=calibration, paired_pre=pre, gradient_batch_records=records, zero_actions=zero,
            probe_actions=probe, zero_replay_actions=zero_repeat, star_actions=verified,
            star_mean=star_mean.tolist(), zero_repeat_mean=repeat_mean.tolist(),
            star_measured_drift=measured_drift.tolist(), per_field_target_correction_fraction=per_field_recovery,
            baseline_weighted_residual_squared=baseline_residual, star_weighted_residual_squared=actual_residual,
            residual_norm_improvement=improvement, replay_discrepancy_norm=replay_discrepancy,
            gain_response_qualified=bool(improvement>replay_discrepancy),
            limitation='TRAIN-only one-step calibration; independent Shape and native-support audits remain separate',
            elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output/'joint_audit.json',result)
        atomic_json(args.output/'progress.json',result)
    except BaseException as exc:
        result.update(status='failed', error_type=type(exc).__name__, error=str(exc), elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output/'progress.json',result)
        raise

if __name__ == '__main__':
    main()
