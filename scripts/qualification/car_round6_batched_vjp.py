#!/usr/bin/env python3
"""One real mature J2 batch: qualify batched fresh family VJPs against separate VJPs."""
from __future__ import annotations

import argparse
import copy
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
from scripts.qualification.car_round5_panel_replay import load_context, batches
from scripts.qualification.car_round6_joint_audit import sha256
from scripts.qualification.car_round6_mature_first_batch import flattened


def batched_rows(losses, parameters):
    values = torch.stack(tuple(losses))
    gradients = torch.autograd.grad(values, parameters, grad_outputs=torch.eye(
        len(losses), device=values.device, dtype=values.dtype), is_grads_batched=True,
        retain_graph=True, allow_unused=True)
    pieces = []
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            shape = (len(losses), *parameter.shape)
            gradient = torch.zeros(shape, dtype=parameter.dtype, device=parameter.device)
        if torch.is_complex(gradient):
            gradient = torch.view_as_real(gradient)
        pieces.append(gradient.reshape(len(losses), -1))
    return torch.cat(pieces, dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=torch.device, required=True)
    args = parser.parse_args()
    if not os.environ.get('CAR_ROUND6_LEASE_TOKEN') or args.device.type != 'cuda':
        parser.error('a measured Round 6 CUDA preparation lease is required')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    receipt = {'schema': 'car_round6_batched_vjp_v1', 'status': 'failed',
        'device': str(args.device), 'device_name': torch.cuda.get_device_name(args.device),
        'helper_sha256': sha256(Path(__file__)),
        'scope': 'one real TRAIN B64 J2 mature batch; fresh parameter-space family gradients, no model efficacy or epoch speed claim'}
    started = time.monotonic()
    dataset = source = None
    try:
        torch.set_num_threads(4)
        config = args.run / 'resolved_config.yaml'
        ckpath = args.run / 'checkpoints/epoch_100.pt'
        cfg, model, dataset, panel, runtime, scales = load_context(config, args.device)
        ck = torch.load(ckpath, map_location='cpu', weights_only=False)
        if ck['global_step'] != 1200:
            raise ValueError('expected completed epoch-100 mature checkpoint')
        set_trainable_scope(model, cfg['trainable'])
        model.load_state_dict(ck['model'], strict=True)
        post_training_mode(model, cfg)
        parameters = [p for p in model.parameters() if p.requires_grad]
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        expected = json.loads((args.run / 'artifacts/trainable_parameters.json').read_text())['names']
        if names != expected:
            raise ValueError('trainable parameter ordering differs from mature checkpoint')
        source = batches(dataset, panel, cfg, args.device)
        batch = next(iter(source))
        if list(batch.sample_ids) != panel['sample_ids'][:len(batch.sample_ids)]:
            raise ValueError('TRAIN panel order mismatch')
        runtime.begin_step(0)
        losses, endpoint = {}, {}
        seed = 47 + 1_000_003
        _coherence_objective(model, batch, runtime.families, {name: None for name in runtime.families}, cfg,
            step=0, generator=torch.Generator(device=args.device).manual_seed(seed), family_scales=scales,
            family_losses=losses, endpoint_context=endpoint, scalable_runtime=runtime, scalable_probe=False)
        family_losses, _ = FamilyScalarizer(cfg['optimization']['multitask'].get('family_scalarization', {}))(
            losses, scales=scales, weights={n: float(getattr(f, 'family_weight', 1.0)) for n, f in runtime.families.items()})
        order = three_family_order(runtime.families)
        reference = MeanRiskDifferenceBudget.from_state_dict(ck['external_policy']['reference'])
        multipliers = torch.tensor(ck['external_policy']['multipliers'], device=args.device, dtype=torch.float64)
        risks = _physical_endpoint_risk(endpoint, dataset, mean_risk=True).mean(dim=0)
        violation = reference.violations(risks)
        field_gradient = _flat_gradient((multipliers.to(violation) * violation).sum(), parameters).double()

        def measured(fn):
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            initial = torch.cuda.memory_allocated(args.device)
            begin = time.monotonic()
            value = fn()
            torch.cuda.synchronize(args.device)
            return value, {'seconds': time.monotonic()-begin, 'initial_allocated_bytes': initial,
                           'peak_allocated_bytes': torch.cuda.max_memory_allocated(args.device)}

        separate, separate_cost = measured(lambda: torch.stack([_flat_gradient(family_losses[n], parameters) for n in order]))
        batched, batched_cost = measured(lambda: batched_rows([family_losses[n] for n in order], parameters))
        aggregator = OuterAggregator({'method': 'torchjd_config'})
        directions = [aggregator(rows.double())[0] / reference.objective_scale + field_gradient
                      for rows in (separate, batched)]

        def action(direction):
            model.load_state_dict(ck['model'], strict=True)
            opt = torch.optim.AdamW(parameters, lr=float(cfg['optimization']['lr']),
                weight_decay=float(cfg['optimization'].get('weight_decay', 0)),
                betas=tuple(cfg['optimization'].get('adam_betas', (0.9, 0.999))))
            opt.load_state_dict(copy.deepcopy(ck['optimizer']))
            before = flattened(parameters)
            _assign_flat_gradient(parameters, direction)
            norm = stable_clip_grad_norm_(parameters, float(cfg['optimization']['grad_clip']))
            opt.step()
            return flattened(parameters) - before, float(norm)

        action_separate, norm_separate = action(directions[0])
        action_batched, norm_batched = action(directions[1])
        grad_diff = batched.double() - separate.double()
        direction_diff = directions[1] - directions[0]
        action_diff = action_batched - action_separate
        grad_ok = torch.allclose(batched, separate, rtol=2e-4, atol=3e-6)
        action_ok = torch.allclose(action_batched, action_separate, rtol=2e-5, atol=2e-7)
        direction_relative = float(direction_diff.norm() / directions[0].norm())
        receipt.update(status='qualified' if grad_ok and action_ok and direction_relative <= 1e-5 else 'rejected',
            sample_ids=list(batch.sample_ids), family_order=list(order),
            config_sha256=sha256(config), checkpoint_sha256=sha256(ckpath),
            family_losses={name: float(family_losses[name].detach()) for name in order},
            gradient_allclose=bool(grad_ok), gradient_max_abs=float(grad_diff.abs().max()),
            gradient_relative_l2=float(grad_diff.norm() / separate.double().norm()),
            direction_relative_l2=direction_relative, action_allclose=bool(action_ok),
            action_max_abs=float(action_diff.abs().max()), action_relative_l2=float(action_diff.norm()/action_separate.norm()),
            preclip_norms=[norm_separate, norm_batched], separate=separate_cost, batched=batched_cost,
            tolerances={'gradient_rtol':2e-4, 'gradient_atol':3e-6, 'action_rtol':2e-5,
                        'action_atol':2e-7, 'direction_relative_l2':1e-5},
            timing_limit='fixed order on one retained graph; compatibility screening only')
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if source is not None:
            source.close()
        if dataset is not None:
            dataset.close()
        receipt['wall_seconds'] = time.monotonic() - started
        with (args.output / 'receipt.json').open('x') as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write('\n')


if __name__ == '__main__':
    main()
