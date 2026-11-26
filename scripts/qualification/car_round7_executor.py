#!/usr/bin/env python3
"""Profile real mature J2 endpoint VJPs and paired clipped AdamW actions.

One TRAIN batch, diagnostic only. Full-epoch finalist timing and the real J1
adapter check are separate required qualifications, not implied by this screen.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
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
from phycoflow_reconstruction.utils.reproducibility import seed_everything


def configure_backend(config):
    """Apply the production trainer policy before importing its CUDA-aware stack."""
    seed = int(config['runtime'].get('seed',42))
    deterministic = bool(config['runtime'].get('deterministic',True))
    strict = bool(config.get('coherence',{}).get('scalable_anchored_coherence',{}).get('common_prefit'))
    seed_everything(seed,deterministic,strict_algorithms=strict)
    return {'seed':seed,'deterministic':deterministic,'strict_algorithms':strict,
        'deterministic_algorithms_enabled':torch.are_deterministic_algorithms_enabled(),
        'cudnn_deterministic':torch.backends.cudnn.deterministic,
        'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
        'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
        'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}


def differences(actual, reference):
    delta = (actual.double() - reference.double()).abs()
    denominator = reference.double().norm()
    return {'max_abs': float(delta.max()), 'l2_abs': float(delta.norm()),
            'relative_l2': float(delta.norm() / denominator) if denominator > 0 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path, help='Completed mature parent run')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--checkpoint', type=Path, help='Actual mature checkpoint with saved AdamW and multipliers')
    mode.add_argument('--fresh-source', action='store_true', help='Verified SOURCE materialization, fresh AdamW, zero multipliers and first-action warmup LR')
    parser.add_argument('--reference', required=True, type=Path, help='New-law reference JSON')
    parser.add_argument('--kappa', required=True, type=float)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', required=True, type=torch.device)
    parser.add_argument('--trace', action='store_true', help='Trace the first requested factored setting; its timing includes profiler overhead')
    parser.add_argument('--objectives', nargs='+', choices=('composite','joint_anchor'),
                        default=('composite','joint_anchor'))
    parser.add_argument('--settings', nargs='+', choices=('batched','factored_chunk_1','factored_chunk_2','factored_static_features','factored_no_checkpoint'),
                        default=('factored_chunk_2','factored_static_features'),
                        help='Candidate settings; two separate/checkpointed baseline repeats are always included')
    args = parser.parse_args()
    if not os.environ.get('CAR_ROUND7_LEASE_TOKEN') or args.device.type != 'cuda':
        parser.error('requires a measured Round 7 lease and runtime CUDA device')
    if args.kappa <= 0 or not math.isfinite(args.kappa):
        parser.error('kappa must be the positive finite fitted TRAIN value')
    config_path=args.run/'resolved_config.yaml'
    initial_config=yaml.safe_load(config_path.read_text())
    backend_policy=configure_backend(initial_config)
    # The historical replay loader uses this fixed sampler seed. Do not
    # silently claim a different configured seed was used.
    if backend_policy['seed'] != 47:
        raise ValueError('historical replay loader requires runtime seed 47')
    from phycoflow_reconstruction.coherence.compose import three_family_order
    from phycoflow_reconstruction.training.fidelity_budget import _budget_from_state
    from phycoflow_reconstruction.training.fidelity_quadrature import rotating_field_risks
    from phycoflow_reconstruction.training.gradient_balance import _assign_flat_gradient, _flat_gradient
    from phycoflow_reconstruction.training.gradients import stable_clip_grad_norm_
    from phycoflow_reconstruction.training.multitask import EndpointBoundary, FamilyScalarizer, OuterAggregator, batched_family_gradients
    from phycoflow_reconstruction.training.post_training import _coherence_objective, _scheduled_model_lr
    from phycoflow_reconstruction.training.retention import post_training_mode
    from phycoflow_reconstruction.training.source import set_trainable_scope, source_hashes
    from phycoflow_reconstruction.training.scalable_fitting import _digest
    from scripts.qualification.car_round5_panel_replay import batches, load_context
    from scripts.qualification.car_round6_joint_audit import sha256
    from scripts.qualification.car_round6_mature_first_batch import flattened
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = {'schema': 'car_round7_executor_screen_v1', 'status': 'running',
        'scope': ('one matched TRAIN batch, verified SOURCE and fresh AdamW with zero multipliers, rotating complement; no epoch cost or learning claim'
                  if args.fresh_source else 'one matched TRAIN batch, mature model/AdamW/multipliers, rotating complement; no epoch cost or learning claim'),
        'device': str(args.device), 'device_name': torch.cuda.get_device_name(args.device), 'backend_policy':backend_policy,
        'checkpoint_sha256': sha256(args.checkpoint) if args.checkpoint else None, 'reference_sha256': sha256(args.reference),
        'config_sha256': sha256(args.run / 'resolved_config.yaml'), 'attempts': []}
    source = dataset = active_profiler = None
    try:
        torch.set_num_threads(4)
        cfg, model, dataset, panel, runtime, scales = load_context(args.run / 'resolved_config.yaml', args.device)
        set_trainable_scope(model, cfg['trainable'])
        ck = None if args.fresh_source else torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if ck is not None:
            model.load_state_dict(ck['model'], strict=True)
        post_training_mode(model, cfg)
        params = [p for p in model.parameters() if p.requires_grad]
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        if names != json.loads((args.run / 'artifacts/trainable_parameters.json').read_text())['names']:
            raise ValueError('mature trainable parameter order mismatch')
        reference = _budget_from_state(json.loads(args.reference.read_text())['reference'])
        if ck is None:
            initial_model = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            multiplier = torch.zeros(len(reference.field_names), device=args.device, dtype=torch.float64)
            absolute_step = 0
            scheduled_lr = _scheduled_model_lr(cfg['optimization'], 0, 12)
            learning_rate = float(cfg['optimization']['lr']) if scheduled_lr is None else scheduled_lr
            receipt.update(state='fresh_source', source_identity=source_hashes(cfg),
                source_materialized_state_sha256=_digest(model.state_dict()),
                source_panel_materialized_view_sha256=panel['source_weight_view_sha256'],
                learning_rate=learning_rate, optimizer_state='fresh AdamW', checkpoint_sha256=None)
        else:
            initial_model = ck['model']
            multiplier = torch.as_tensor(ck['external_policy']['multipliers'], device=args.device, dtype=torch.float64)
            absolute_step = int(ck['global_step'])
            learning_rate = float(cfg['optimization']['lr'])
            receipt.update(state='mature_checkpoint', absolute_step=absolute_step, optimizer_state='checkpoint AdamW')
        if (multiplier.numel() != len(reference.field_names)
            or not bool(torch.isfinite(multiplier).all()) or bool((multiplier < 0).any())):
            raise ValueError('checkpoint field multipliers must be finite, nonnegative and match fields')
        field_row_active = bool((multiplier != 0).any())
        cfg['optimization']['multitask']['fidelity_budget']['rotating_complement'] = {'points':512, 'seed':20261125}
        source = batches(dataset, panel, cfg, args.device)
        batch = next(iter(source))
        if list(batch.sample_ids) != panel['sample_ids'][:len(batch.sample_ids)]:
            raise ValueError('TRAIN panel ordering changed')
        runtime_state = deepcopy(runtime.state_dict())
        order = three_family_order(runtime.families)
        seed = backend_policy['seed'] + 1_000_003
        candidates = {'batched':('batched',1,True,'none'),
                      'factored_chunk_1':('factored',1,True,'none'),
                      'factored_chunk_2':('factored',2,True,'none'),
                      'factored_static_features':('factored',2,True,'static_features'),
                      'factored_no_checkpoint':('factored',1,False,'none')}
        if len(set(args.objectives)) != len(args.objectives) or len(set(args.settings)) != len(args.settings):
            raise ValueError('objective and setting selections must be unique')
        settings = [('separate_repeat_1','separate',1,True,'none'),
                    ('separate_repeat_2','separate',1,True,'none')]
        settings.extend((label,*candidates[label]) for label in args.settings)
        receipt.update(sample_ids=list(batch.sample_ids), family_order=list(order),
            multipliers=multiplier.cpu().tolist(), field_row_active=field_row_active,
            sampler_scope='same restored initial panel sampler state, not claimed as parent sampler replay',
            limitations=['short single-batch screen; no full-epoch cost', 'deterministic J1 adapter check is separate from this screen'])
        trace_done = False
        for objective in args.objectives:
            baseline = None
            runtime.scale_objective = objective
            runtime.scale_joint_kappa = args.kappa if objective == 'joint_anchor' else 1.0
            objective_state = deepcopy(runtime_state)
            objective_state.update(scale_objective=runtime.scale_objective, scale_joint_kappa=runtime.scale_joint_kappa)
            for label, method, chunk, checkpointing, context_cache in settings:
                model.load_state_dict(initial_model, strict=True)
                model.rollout_checkpointing = checkpointing
                model.rollout_context_cache = context_cache
                runtime.load_state_dict(deepcopy(objective_state))
                runtime.begin_step(runtime.step)
                optimizer = torch.optim.AdamW(params, lr=learning_rate,
                    weight_decay=float(cfg['optimization'].get('weight_decay', 0)),
                    betas=tuple(cfg['optimization'].get('adam_betas',(0.9,0.999))))
                if ck is not None:
                    optimizer.load_state_dict(deepcopy(ck['optimizer']))
                boundary = EndpointBoundary() if method == 'factored' else None
                torch.cuda.synchronize(args.device)
                torch.cuda.reset_peak_memory_stats(args.device)
                traced = args.trace and method == 'factored' and not trace_done
                if traced:
                    active_profiler = torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=False, profile_memory=True)
                    active_profiler.start()
                start = time.monotonic()
                losses, endpoint = {}, {}
                _coherence_objective(model, batch, runtime.families, {n: None for n in runtime.families}, cfg,
                    step=runtime.step, generator=torch.Generator(device=args.device).manual_seed(seed), family_scales=scales,
                    family_losses=losses, endpoint_context=endpoint, scalable_runtime=runtime, scalable_probe=False,
                    endpoint_boundary=boundary)
                family, _ = FamilyScalarizer(cfg['optimization']['multitask'].get('family_scalarization',{}))(
                    losses, scales=scales, weights={n:float(getattr(f,'family_weight',1.0)) for n,f in runtime.families.items()})
                risks, complement = rotating_field_risks(model, endpoint, dataset, cfg,
                    absolute_step=absolute_step, rollout_seed=seed, device=args.device, endpoint_boundary=boundary)
                violation = reference.violations(risks.mean(0))
                torch.cuda.synchronize(args.device)
                forward_seconds = time.monotonic()-start
                start = time.monotonic()
                ordered = [family[n] for n in order]
                if method == 'factored':
                    rows = boundary.rows(ordered, params, chunk_size=chunk)
                    field = (boundary.rows([(multiplier.to(violation)*violation).sum()], params,
                                           chunk_size=1, include_complement=True)[0]
                             if field_row_active else torch.zeros_like(rows[0]))
                else:
                    rows = batched_family_gradients(ordered, params) if method == 'batched' else torch.stack([_flat_gradient(v, params) for v in ordered])
                    field = (_flat_gradient((multiplier.to(violation)*violation).sum(), params)
                             if field_row_active else torch.zeros_like(rows[0]))
                torch.cuda.synchronize(args.device)
                reverse_seconds = time.monotonic()-start
                start = time.monotonic()
                direction, _ = OuterAggregator({'method':'torchjd_config'})(rows.double())
                direction = direction / reference.objective_scale + field.double()
                before = flattened(params)
                _assign_flat_gradient(params, direction)
                clip_input = stable_clip_grad_norm_(params, float(cfg['optimization']['grad_clip']))
                optimizer.step()
                displacement = flattened(params) - before
                torch.cuda.synchronize(args.device)
                action_seconds = time.monotonic()-start
                if traced:
                    active_profiler.stop()
                    trace_path = args.output/f'{objective}_{label}_trace.json'
                    active_profiler.export_chrome_trace(str(trace_path))
                    active_profiler = None
                    trace_done = True
                    receipt['trace'] = {'path':str(trace_path.resolve()), 'sha256':sha256(trace_path),
                        'scope':f'one {objective} {label} screen; inclusive nested profiler ranges, not additive phase totals'}
                payload = {'rows':rows.detach().double().cpu(), 'field':field.detach().double().cpu(),
                    'direction':direction.detach().cpu(), 'displacement':displacement.detach().cpu(),
                    'values':torch.stack([family[n].detach() for n in order]).double().cpu(),
                    'risk':risks.detach().double().mean(0).cpu()}
                if baseline is None:
                    baseline = payload
                comparison = {key: differences(payload[key], baseline[key]) for key in payload}
                per_family = {}
                for index, name in enumerate(order):
                    reference_row, candidate_row = baseline['rows'][index], payload['rows'][index]
                    per_family[name] = {
                        **differences(candidate_row, reference_row),
                        'baseline_norm':float(reference_row.norm()),
                        'candidate_norm':float(candidate_row.norm()),
                    }
                field_norms = {'baseline':float(baseline['field'].norm()),
                               'candidate':float(payload['field'].norm())}
                receipt['attempts'].append({'objective':objective, 'setting':label, 'method':method, 'chunk':chunk,
                    'parameter_row_dtype':str(rows.dtype),
                    'checkpointing':checkpointing, 'context_cache':context_cache,
                    'forward_seconds':forward_seconds, 'reverse_seconds':reverse_seconds,
                    'profiler_enabled':traced,
                    'solver_optimizer_seconds':action_seconds, 'peak_allocated_bytes':torch.cuda.max_memory_allocated(args.device),
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved(args.device), 'clip_input_norm':float(clip_input),
                    'selected_query_sha256':sha256_tensor(endpoint['query_indices']), 'complement_sha256':sha256_tensor(complement),
                    'comparison_to_first_separate':comparison,
                    'per_family_row_comparison_to_first_separate':per_family,
                    'weighted_field_row_norms':field_norms})
                (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
                runtime.finish_step(True)
                del losses, endpoint, family, risks, violation, ordered, rows, field, direction, displacement, boundary, payload, optimizer, clip_input
            del baseline
        receipt['status'] = 'completed_screen'
    except BaseException as error:
        receipt.update(status='failed',error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if active_profiler is not None:
            active_profiler.stop()
        if source is not None: source.close()
        if dataset is not None: dataset.close()
        receipt['wall_seconds'] = time.monotonic() - started
        (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')


def sha256_tensor(value):
    import hashlib
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


if __name__ == '__main__':
    main()
