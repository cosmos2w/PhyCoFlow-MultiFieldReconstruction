#!/usr/bin/env python3
"""Timed one-batch mature-state replay; execution check only, not calibration."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import torch
from phycoflow_reconstruction.coherence.compose import three_family_order
from phycoflow_reconstruction.training.fidelity_budget import MeanRiskDifferenceBudget
from phycoflow_reconstruction.training.gradient_balance import _assign_flat_gradient, _flat_gradient
from phycoflow_reconstruction.training.gradients import stable_clip_grad_norm_
from phycoflow_reconstruction.training.multitask import FamilyScalarizer, OuterAggregator
from phycoflow_reconstruction.training.post_training import _coherence_objective, _physical_endpoint_risk
from phycoflow_reconstruction.training.retention import post_training_mode
from phycoflow_reconstruction.training.source import set_trainable_scope
from scripts.qualification.car_round5_panel_replay import load_context, batches, replay_members
from scripts.qualification.car_round6_joint_audit import sha256


def flattened(parameters):
    return torch.cat([(torch.view_as_real(p.detach()) if torch.is_complex(p) else p.detach()).reshape(-1) for p in parameters]).double()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--device', required=True, type=torch.device)
    args = p.parse_args()
    if not os.environ.get('CAR_ROUND6_LEASE_TOKEN'):
        p.error('run under a measured Round 6 preparation lease')
    if args.device.type != 'cuda':
        p.error('the timed first-batch check requires the selected CUDA device')
    started = time.monotonic()
    torch.set_num_threads(4)
    config = args.run / 'resolved_config.yaml'
    checkpoint_path = args.run / 'checkpoints/epoch_100.pt'
    cfg, model, dataset, panel, runtime, scales = load_context(config, args.device)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint['global_step'] != 1200:
        raise ValueError('expected completed epoch-100 mature checkpoint')
    set_trainable_scope(model, cfg['trainable'])
    model.load_state_dict(checkpoint['model'], strict=True)
    post_training_mode(model, cfg)
    parameters = [p for p in model.parameters() if p.requires_grad]
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    expected = json.loads((args.run / 'artifacts/trainable_parameters.json').read_text())['names']
    if names != expected:
        raise ValueError('trainable parameter order does not match mature run')
    optimizer = torch.optim.AdamW(parameters, lr=float(cfg['optimization']['lr']),
        weight_decay=float(cfg['optimization'].get('weight_decay', 0)),
        betas=tuple(cfg['optimization'].get('adam_betas', (0.9, 0.999))))
    optimizer.load_state_dict(checkpoint['optimizer'])
    reference = MeanRiskDifferenceBudget.from_state_dict(checkpoint['external_policy']['reference'])
    multiplier = torch.tensor(checkpoint['external_policy']['multipliers'], device=args.device, dtype=torch.float64)
    if multiplier.numel() != len(reference.field_names):
        raise ValueError('mature multiplier dimension mismatch')
    source = batches(dataset, panel, cfg, args.device)
    try:
        batch = next(iter(source))
        if list(batch.sample_ids) != panel['sample_ids'][:len(batch.sample_ids)]:
            raise ValueError('TRAIN panel order mismatch')
        runtime.begin_step(0)
        torch.cuda.synchronize(args.device)
        begin = time.monotonic()
        losses, endpoint = {}, {}
        seed = 47 + 1_000_003
        _coherence_objective(model, batch, runtime.families, {name: None for name in runtime.families}, cfg,
            step=0, generator=torch.Generator(device=args.device).manual_seed(seed),
            family_scales=scales, family_losses=losses, endpoint_context=endpoint,
            scalable_runtime=runtime, scalable_probe=False)
        torch.cuda.synchronize(args.device)
        rollout_seconds = time.monotonic() - begin
        family_losses, _ = FamilyScalarizer(cfg['optimization']['multitask'].get('family_scalarization', {}))(
            losses, scales=scales, weights={name: float(getattr(f, 'family_weight', 1.0)) for name, f in runtime.families.items()})
        order = three_family_order(runtime.families)
        risks = _physical_endpoint_risk(endpoint, dataset, mean_risk=True).mean(dim=0)
        violations = reference.violations(risks)
        begin = time.monotonic()
        family_rows = torch.stack([_flat_gradient(family_losses[name], parameters) for name in order]).double()
        field_rows = torch.stack([_flat_gradient(violations[i], parameters) for i in range(len(reference.field_names))]).double()
        torch.cuda.synchronize(args.device)
        backward_seconds = time.monotonic() - begin
        begin = time.monotonic()
        direction, _ = OuterAggregator({'method':'torchjd_config'})(family_rows)
        config_direction = direction / reference.objective_scale
        feedback = multiplier @ field_rows
        combined = config_direction + feedback
        torch.cuda.synchronize(args.device)
        config_seconds = time.monotonic() - begin
        begin = time.monotonic()
        before = flattened(parameters)
        _assign_flat_gradient(parameters, combined)
        norm = stable_clip_grad_norm_(parameters, float(cfg['optimization']['grad_clip']))
        optimizer.step()
        displacement = flattened(parameters) - before
        torch.cuda.synchronize(args.device)
        action_seconds = time.monotonic() - begin
        begin = time.monotonic()
        with torch.no_grad():
            members = replay_members(model, endpoint, cfg, seed)
            replay = dict(endpoint, prediction=members[0], members=members)
            after = _physical_endpoint_risk(replay, dataset, mean_risk=True).mean(dim=0)
        torch.cuda.synchronize(args.device)
        replay_seconds = time.monotonic() - begin
        runtime.finish_step(True)
        args.output.mkdir(parents=True, exist_ok=False)
        torch.save({'family_rows':family_rows.cpu(), 'field_rows':field_rows.cpu(), 'parameter_names':names,
            'optimizer_state_before':checkpoint['optimizer'], 'model_state_before':checkpoint['model'],
            'family_order':order, 'field_order':reference.field_names}, args.output / 'mature_batch_tensors.pt')
        result = {'status':'completed_first_batch_only', 'sample_ids':list(batch.sample_ids),
            'checkpoint_sha256':sha256(checkpoint_path), 'config_sha256':sha256(config),
            'actual_device':torch.cuda.get_device_name(args.device), 'family_order':list(order),
            'field_order':list(reference.field_names), 'multipliers':multiplier.cpu().tolist(),
            'pre_risk':risks.detach().cpu().tolist(), 'post_risk':after.cpu().tolist(),
            'pre_violation':violations.detach().cpu().tolist(), 'family_products_config':(family_rows @ config_direction).cpu().tolist(),
            'family_products_composed':(family_rows @ combined).cpu().tolist(),
            'family_products_actual_displacement':(family_rows @ displacement).cpu().tolist(),
            'field_products_actual_displacement':(field_rows @ displacement).cpu().tolist(),
            'config_norm':float(torch.linalg.vector_norm(config_direction)), 'feedback_norm':float(torch.linalg.vector_norm(feedback)),
            'combined_norm':float(norm), 'lr':optimizer.param_groups[0]['lr'], 'weight_decay':optimizer.param_groups[0]['weight_decay'],
            'rollout_and_scoring_seconds':rollout_seconds, 'family_and_field_backward_seconds':backward_seconds,
            'config_seconds':config_seconds, 'action_seconds':action_seconds, 'risk_replay_seconds':replay_seconds,
            'total_seconds':time.monotonic()-started,
            'limitation':'one batch with historical multiplier; not panel calibration or grouped-query qualification'}
        with (args.output / 'first_batch.json').open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    finally:
        source.close()
        dataset.close()

if __name__ == '__main__':
    main()
