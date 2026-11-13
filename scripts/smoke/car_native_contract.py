#!/usr/bin/env python3
"""Pinned-source native numerical microcase; not a population evaluation."""
from __future__ import annotations

import argparse
import copy
import os
import platform
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('ffm', 'edm'), required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--check-recompute', action='store_true')
    parser.add_argument('--check-staged', action='store_true')
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    from phycoflow_reconstruction.evaluation.adaptive_campaign import enforce_execution_constraints
    enforce_execution_constraints(args.campaign.resolve(), args.gpu_uuid, args.source)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu_uuid:
        raise ValueError('bind the declared physical UUID before Python starts')
    if not os.environ.get('KEOPS_CACHE_FOLDER'):
        raise ValueError('device-specific KeOps cache required before imports')
    cache = Path(os.environ['KEOPS_CACHE_FOLDER'])
    build = '_'.join(platform.uname()[:3]) + f'_p{sys.version.split()[0]}'
    # KeOps 2.3 CUDAConfig uses this directory before creating it.
    (cache / f'{build}_CUDA_VISIBLE_DEVICES_{args.gpu_uuid}').mkdir(parents=True, exist_ok=True)
    import torch
    import yaml

    from phycoflow_reconstruction.coherence.physical_risk import physical_ensemble_risks
    from phycoflow_reconstruction.data.physical_measure import attach_batch_measure
    from phycoflow_reconstruction.data.sensor_protocols import (
        SensorProtocol,
        build_observation_batch,
    )
    from phycoflow_reconstruction.data.training_batches import fixed_query_indices
    from phycoflow_reconstruction.evaluation.adaptive_campaign import (
        atomic_json,
        begin,
        end,
        locked_ledger,
    )
    from phycoflow_reconstruction.training.rollout import differentiable_reconstruction
    from phycoflow_reconstruction.training.scalable_fitting import _digest
    from phycoflow_reconstruction.training.source import load_source_model

    torch.set_num_threads(4)
    campaign = args.campaign.resolve()
    suffix = '_staged_bf16_unit_roundoff' if args.check_staged else '_recompute' if args.check_recompute else ''
    output = campaign / 'qualification' / f'native_{args.source}{suffix}'
    output.mkdir(parents=True, exist_ok=False)
    ledger = campaign / 'gpu_ledger.json'
    token = str(uuid.uuid4())
    with locked_ledger(ledger) as state:
        begin(state, token, args.gpu_uuid, 'native_microcase')
    began = time.perf_counter()
    receipt = {'schema': 'car_native_microcase_v1', 'source': args.source,
               'status': 'running', 'population_claim': 'one local derivative microcase',
               'gpu_uuid': args.gpu_uuid, 'pid': os.getpid()}
    atomic_json(output / 'receipt.json', receipt)
    try:
        snapshot = campaign / 'sources' / args.source
        config = yaml.safe_load((snapshot / 'resolved_config.yaml').read_text())
        config.update(source_run=str(snapshot), source_checkpoint='last.pt')
        config['source'] = {'kind': 'native_run', 'admission': 'frozen_snapshot',
                            'snapshot_receipt': str(snapshot / 'snapshot_receipt.json'),
                            'weight_view': 'ema_materialized'}
        config['dataset']['physical_measure'] = 'native_rectilinear_area'
        model, dataset, identity = load_source_model(config, torch.device('cuda:0'))
        model.eval()
        frozen = copy.deepcopy(model).requires_grad_(False).eval()
        sample = dataset[0]
        query = None if args.source == 'edm' else fixed_query_indices(40300, 4096, seed=100045)
        batch = build_observation_batch(
            [sample], SensorProtocol(field_counts={'T': 256}, seed=2047), query_indices=query
        )
        batch = attach_batch_measure(dataset, batch).to('cuda:0')
        target = batch.target_fields
        batch.target_fields = None
        steps, solver = (32, 'heun') if args.source == 'edm' else (4, 'euler')
        consistency = ({'mode': 'native'} if args.source == 'edm' else {
            'mode': 'endpoint_smooth', 'strength': 1.0, 'sigma': 0.05,
            'schedule_power': 2.0, 'final_clamp': True, 'chunk_size': 4096,
        })
        seeds = (31001, 31002)
        def generate(candidate, seed):
            return differentiable_reconstruction(
                candidate, batch, steps=steps, solver=solver,
                generator=torch.Generator(device='cuda:0').manual_seed(seed),
                observation_config=consistency,
            )
        with torch.no_grad():
            anchor = generate(frozen, seeds[0])
            native = model.reconstruct(
                batch, steps=steps,
                generator=torch.Generator(device='cuda:0').manual_seed(seeds[0]),
            ).prediction
        # Native view and generic differentiable view must denote one function.
        tolerance = (3e-4, 3e-4) if args.source == 'edm' else (3e-5, 3e-5)
        torch.testing.assert_close(anchor, native, rtol=tolerance[0], atol=tolerance[1])
        del native
        if args.source == 'edm' or args.check_recompute:
            model.rollout_checkpointing = True
        members = torch.stack([generate(model, seed) for seed in seeds], dim=1)
        torch.testing.assert_close(members[:, 0], anchor, rtol=tolerance[0], atol=tolerance[1])
        mass = batch.metadata['sample_context']['auxiliary']['physical_weights']
        sampling = {} if args.source == 'edm' else {
            'sampling': 'without_replacement', 'probabilities': 4096 / 40300,
            'total_mass': dataset.physical_weights.sum().to('cuda:0'),
        }
        member_risk, mean_risk = physical_ensemble_risks(members, target, mass, **sampling)
        params = tuple(p for p in model.parameters() if p.requires_grad)
        gradients = {}
        reference_gradients = {}
        for label, value in (('member', member_risk), ('signed_mean', mean_risk)):
            grads = torch.autograd.grad(value.mean(), (*params, members),
                                        retain_graph=label == 'member', allow_unused=True)
            if any(not torch.isfinite(g).all() for g in grads if g is not None):
                raise FloatingPointError('nonfinite native parameter/member risk gradients')
            endpoint_grad = grads[-1]
            if args.check_recompute or args.check_staged:
                reference_gradients[label] = [None if g is None else g.detach().cpu() for g in grads]
            gradients[label] = {
                'parameter_norm': float(sum(g.double().square().sum() for g in grads[:-1]
                                           if g is not None).sqrt()),
                'per_member_norm': endpoint_grad.double().square().sum((0, 2, 3)).sqrt().tolist(),
                'parameters_with_gradient': sum(g is not None for g in grads[:-1]),
            }
            del grads
        if args.check_staged:
            checks = {}
            gradient_rtol = 5e-3 if getattr(model, 'precision', 'fp32') == 'bf16' else 3e-5
            gradient_atol = 1e-5 if gradient_rtol > 3e-5 else 3e-6
            for label in ('member', 'signed_mean'):
                expected = reference_gradients[label]
                contributions = [None] * len(params)
                for member_index, seed in enumerate(seeds):
                    endpoint = generate(model, seed)
                    part = torch.autograd.grad(endpoint, params,
                        grad_outputs=expected[-1][:, member_index].to(endpoint), allow_unused=True)
                    for i, value in enumerate(part):
                        if value is not None:
                            value = value.detach().cpu()
                            contributions[i] = value if contributions[i] is None else contributions[i] + value
                    del endpoint, part
                worst = 0.0
                error_squared = reference_squared = 0.0
                for a, b in zip(contributions, expected[:-1]):
                    if a is None or b is None:
                        if a is not None or b is not None:
                            raise ValueError('staged parameter participation changed')
                    else:
                        worst = max(worst, float((a - b).abs().max()))
                        error_squared += float((a.double() - b.double()).square().sum())
                        reference_squared += float(b.double().square().sum())
                relative_error = (error_squared / max(reference_squared, 1e-30)) ** .5
                norm_tolerance = 2 ** -8 if getattr(model, 'precision', 'fp32') == 'bf16' else 1e-5
                if relative_error > norm_tolerance:
                    raise AssertionError(f'staged full-parameter relative norm error {relative_error} > {norm_tolerance}')
                checks[label] = {'all_parameters_and_members': True,
                                 'max_abs_error': worst, 'relative_norm_error': relative_error,
                                 'rtol': gradient_rtol, 'atol': gradient_atol,
                                 'max_relative_norm_error': norm_tolerance,
                                 'acceptance_metric': 'complete_parameter_vector_relative_L2',
                                 'precision': getattr(model, 'precision', 'fp32')}
                del contributions
            receipt['staged_gradient_parity'] = checks
        if args.check_recompute:
            model.rollout_checkpointing = False
            baseline = torch.stack([generate(model, seed) for seed in seeds], dim=1)
            torch.testing.assert_close(baseline, members.detach(), rtol=tolerance[0], atol=tolerance[1])
            baseline_risks = physical_ensemble_risks(baseline, target, mass, **sampling)
            checks = {}
            for index, label in enumerate(('member', 'signed_mean')):
                grads = torch.autograd.grad(baseline_risks[index].mean(), (*params, baseline),
                                            retain_graph=index == 0, allow_unused=True)
                worst = 0.0
                for a, b in zip(grads, reference_gradients[label]):
                    if a is None or b is None:
                        if a is not None or b is not None:
                            raise ValueError('recomputation parameter participation changed')
                    else:
                        a = a.detach().cpu()
                        torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
                        worst = max(worst, float((a - b).abs().max()))
                checks[label] = {'all_parameters_and_members': True, 'max_abs_error': worst,
                                 'rtol': 3e-5, 'atol': 3e-6}
                del grads
            receipt['recomputation_gradient_parity'] = checks
            del baseline, baseline_risks, reference_gradients
        torch.cuda.synchronize()
        receipt.update(status='passed', source_identity=identity, sample_ids=list(batch.sample_ids),
                       support_sites=batch.query_coords.shape[1], steps=steps, solver=solver,
                       native_parity_tolerance={'rtol': tolerance[0], 'atol': tolerance[1]},
                       observation_sha256=_digest([batch.obs_indices, batch.obs_values]),
                       parameter_buffer_sha256=identity['materialized_view_sha256'],
                       random_seeds=seeds, model_units_risk=member_risk.detach().cpu().tolist(),
                       signed_mean_risk=mean_risk.detach().cpu().tolist(), gradients=gradients,
                       peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                       seconds=time.perf_counter() - began)
        if hasattr(model, 'prior'):
            receipt['prior_buffer_sha256'] = _digest({k: v for k, v in model.state_dict().items()
                                                     if k in ('prior.omega', 'prior.phase')})
        torch.save({'members': members.detach().cpu(), 'reference': target.cpu(),
                    'query_indices': batch.metadata['query_indices'].cpu(),
                    'masses': mass.cpu()}, output / 'endpoints.pt')
        dataset.close()
    except BaseException as exc:
        receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                       seconds=time.perf_counter() - began)
        raise
    finally:
        atomic_json(output / 'receipt.json', receipt)
        with locked_ledger(ledger) as state:
            end(state, token)
    print(output / 'receipt.json')


if __name__ == '__main__':
    main()
