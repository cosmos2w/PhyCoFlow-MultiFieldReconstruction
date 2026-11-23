#!/usr/bin/env python3
"""Paired four-seed TRAIN800 parameter-gradient comparison of CQ query laws.

This diagnostic leaves SOURCE parameters unchanged and reuses each batch's
native bridge and two rollout members for all query draws. A full panel is
required before the Gram matrices can support an estimator decision.
"""
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
import torch
from phycoflow_reconstruction.coherence.compose import three_family_order
from phycoflow_reconstruction.coherence.query_sampling import ConditionQueryGroups
from phycoflow_reconstruction.data.physical_measure import batch_measure_context
from phycoflow_reconstruction.evaluation.adaptive_campaign import atomic_json, digest
from phycoflow_reconstruction.training.direct_supervision import direct_scalar
from phycoflow_reconstruction.training.gradient_balance import _flat_gradient
from phycoflow_reconstruction.training.post_training import _coherence_objective
from phycoflow_reconstruction.training.source import source_hashes
from scripts.qualification.car_round5_panel_replay import load_context, batches, native_value
from scripts.qualification.car_round5_query_endpoint_replay import gram_summary


def plan(runtime, seed, index):
    """Couple the first query and retain the runtime's frozen marginal law."""
    batches = []
    for group in range(8):
        sampler = deepcopy(runtime.sampler)
        sampler.pending = None
        if hasattr(sampler, '_transaction'):
            sampler._transaction = None
        base = int(seed) * 1000003 + int(index) * 65537 + group * 7919
        sampler.rng.seed(base)
        sampler.inner_rng.seed(base ^ 0x524F554E4438)
        batches.append(sampler.draw())
    return batches[0], ConditionQueryGroups(tuple(batches))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--units', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu', type=torch.device)
    parser.add_argument('--seeds', default=4, type=int)
    parser.add_argument('--seconds-limit', required=True, type=float)
    parser.add_argument('--save-batch-gradients', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('CAR_ROUND5_LEASE_TOKEN'):
        parser.error('run under the measured Round 5 preparation lease')
    if not 1 <= args.seeds <= 4 or args.seconds_limit <= 0:
        parser.error('one to four seeds and positive seconds limit required')
    start = time.monotonic()
    torch.set_num_threads(4)
    units_record = json.loads(args.units.read_text())
    if (units_record.get('status') != 'completed' or units_record.get('sample_count') != 800
            or any(units_record.get('floor_active', {}).values())):
        raise ValueError('complete, floor-free frozen TRAIN800 SOURCE units required')
    units = units_record['units']
    if any(not 0 < float(units[key]) < float('inf') for key in ('native', 'state', 'scale', 'shape')):
        raise ValueError('invalid frozen direct-objective units')
    cfg, model, dataset, panel, runtime, scales = load_context(args.config, args.device)
    if units_record.get('source_panel_sha256') != cfg['coherence']['scalable_anchored_coherence']['common_core']['source_panel_sha256']:
        raise ValueError('units and SOURCE panel differ')
    if units_record.get('source_checkpoint_sha256') != source_hashes(cfg)['checkpoint']:
        raise ValueError('units and SOURCE checkpoint differ')
    parameters = [p for p in model.parameters() if p.requires_grad]
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    family_order = three_family_order(runtime.families)
    source = batches(dataset, panel, cfg, args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'schema':'car_round5_parameter_query_replay_v1', 'status':'running',
        'sample_count':800, 'completed_conditions':0, 'seeds':args.seeds,
        'units_sha256':digest(args.units), 'source_panel_sha256':units_record['source_panel_sha256'],
        'config_sha256':digest(args.config), 'source_checkpoint_sha256':units_record['source_checkpoint_sha256'],
        'parameter_names':names, 'parameter_elements':sum(p.numel() * (2 if torch.is_complex(p) else 1) for p in parameters),
        'query_laws':['shared', 'condition_groups_8'], 'shape_gradient':'physical family weight / (3 * frozen Shape unit)',
        'device':str(args.device), 'seconds_limit':args.seconds_limit,
        'gradient_accumulation':'sample-count-weighted panel mean; all active parameter tensors',
        'reuse':'same SOURCE, observations, native bridge and conditional members across query laws and seeds'}
    atomic_json(args.output/'result.json', result)
    sums = None
    try:
        for index, batch in enumerate(source):
            if time.monotonic() - start >= args.seconds_limit:
                result['status'] = 'budget_expired'
                break
            batch_started = time.monotonic()
            count = len(batch.sample_ids)
            offset = result['completed_conditions']
            if list(batch.sample_ids) != panel['sample_ids'][offset:offset+count]:
                raise ValueError('TRAIN800 sample order changed')
            runtime.begin_step(index)
            rollout_seed = 47+1_000_003+index
            native_seed = 47+2_000_003+index
            endpoint, initial_losses = {}, {}
            _coherence_objective(model, batch, runtime.families, {name:None for name in runtime.families}, cfg,
                step=index, generator=torch.Generator(device=args.device).manual_seed(rollout_seed),
                family_scales=scales, family_losses=initial_losses, endpoint_context=endpoint,
                scalable_runtime=runtime, scalable_probe=False)
            native = native_value(model, batch, native_seed)
            condition = endpoint['coupling_selected_batch']
            context = {'sample_ids':batch.sample_ids, 'reference_ids':batch.sample_ids,
                **batch_measure_context(condition), 'conditional_draws':(endpoint['members'][1],)}
            members, truth = endpoint['members'], endpoint['target']
            state_loss = initial_losses[family_order[0]]
            batch_gradients = []
            for seed in range(1, args.seeds+1):
                for query in plan(runtime, seed, index):
                    losses = {family_order[0]:state_loss}
                    for family_name in family_order[1:]:
                        scored, _, _ = runtime._score(family_name, runtime.families[family_name],
                            members[0], truth, condition.query_coords, context, query)
                        losses[family_name] = scored.scalar_loss
                    shape_name = family_order[2]
                    shape = (losses[shape_name] * scales[shape_name] *
                             float(getattr(runtime.families[shape_name], 'family_weight', 1.0)) /
                             float(units['shape']) / 3.0)
                    objective = direct_scalar(native, losses, units, scales, runtime.families)
                    if not torch.isfinite(shape) or not torch.isfinite(objective):
                        raise FloatingPointError('nonfinite replay loss')
                    grads = torch.stack([_flat_gradient(shape, parameters).detach().cpu(),
                                         _flat_gradient(objective, parameters).detach().cpu()])
                    if not torch.isfinite(grads).all():
                        raise FloatingPointError('nonfinite replay gradient')
                    batch_gradients.append(grads)
            batch_gradients = torch.stack(batch_gradients)
            if sums is None:
                sums = torch.zeros_like(batch_gradients, device='cpu')
            sums.add_(batch_gradients, alpha=count / 800)
            if args.save_batch_gradients:
                torch.save({'sample_ids':list(batch.sample_ids), 'gradients':batch_gradients,
                            'parameter_names':names}, args.output/f'batch_{index:02d}.pt')
            runtime.finish_step(True)
            del batch_gradients, grads, shape, objective, losses, native, initial_losses, endpoint, state_loss, members
            result['completed_conditions'] += count
            elapsed = time.monotonic()-start
            result['elapsed_seconds'] = elapsed
            if index == 0:
                result['first_batch_seconds'] = time.monotonic()-batch_started
                result['forecast_total_seconds'] = elapsed + 12 * result['first_batch_seconds']
                if result['forecast_total_seconds'] > args.seconds_limit:
                    result['status'] = 'forecast_exceeds_budget'
                    atomic_json(args.output/'result.json', result)
                    break
            atomic_json(args.output/'result.json', result)
        else:
            result['status'] = 'completed'
        if result['status'] == 'completed':
            if result['completed_conditions'] != 800:
                raise ValueError('incomplete panel cannot qualify')
            for pos, label in enumerate(('shape', 'direct_scalar')):
                values = sums[:, pos].double()
                gram = values @ values.T
                shared = list(range(0, 2*args.seeds, 2))
                grouped = list(range(1, 2*args.seeds, 2))
                result[label] = {'gradient_gram':gram.tolist(),
                    'shared':gram_summary(gram[shared][:, shared]),
                    'grouped':gram_summary(gram[grouped][:, grouped])}
            torch.save({'gradient_sums':sums, 'parameter_names':names}, args.output/'panel_gradients.pt')
        result['elapsed_seconds'] = time.monotonic()-start
        atomic_json(args.output/'result.json', result)
    finally:
        source.close()
        dataset.close()


if __name__ == '__main__':
    main()
