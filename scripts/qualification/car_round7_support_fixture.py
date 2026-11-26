#!/usr/bin/env python3
"""Compare one TRAIN condition at selected, complement and native query supports.

All rollouts reuse the exact original padded observation tensors and J2 random
states. The repeated rollout supplies a measured numerical-repeatability floor.
No population or unbiased-native claim is inferred from this fixture.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import torch
import yaml
from phycoflow_reconstruction.coherence.physical_risk import physical_ensemble_risks, physical_field_mse
from phycoflow_reconstruction.data.sensor_protocols import build_observation_batch
from phycoflow_reconstruction.data.training_batches import build_training_batch_source
from phycoflow_reconstruction.evaluation.adaptive_campaign import atomic_json, digest
from phycoflow_reconstruction.training.common import sensor_protocol_from_config
from phycoflow_reconstruction.training.fidelity_quadrature import complement_indices
from phycoflow_reconstruction.training.post_training import _conditional_draw_seed, _slice_batch, _without_target
from phycoflow_reconstruction.training.retention import post_training_mode
from phycoflow_reconstruction.training.rollout import differentiable_reconstruction, subset_query_batch
from phycoflow_reconstruction.training.source import load_source_model, source_hashes
from phycoflow_reconstruction.training.scalable_fitting import _digest


def support_batch(selected, sample, indices, config, device):
    """Build requested query tensors, retaining all original observation bytes."""
    mask = selected.obs_valid_mask[0].cpu()
    obs = list(zip(selected.obs_indices[0].cpu()[mask].tolist(),
                   selected.obs_field_ids[0].cpu()[mask].tolist(), strict=True))
    sample_id = selected.sample_ids[0]
    rebuilt = build_observation_batch([sample], sensor_protocol_from_config(config),
        query_indices={sample_id: indices.tolist()}, manifest_indices={sample_id: obs}).to(device)
    if rebuilt.sample_ids != selected.sample_ids or not torch.equal(rebuilt.metadata['query_indices'].cpu()[0], indices):
        raise ValueError('query identity or sample identity changed')
    rebuilt_mask = rebuilt.obs_valid_mask[0]
    for key in ('obs_indices', 'obs_field_ids', 'obs_coords', 'obs_values'):
        if not torch.equal(getattr(selected, key)[0][mask.to(device)], getattr(rebuilt, key)[0][rebuilt_mask]):
            raise ValueError('valid observation changed: ' + key)
    metadata = dict(rebuilt.metadata)
    metadata['protocol'] = selected.metadata['protocol']
    batch = replace(selected, query_coords=rebuilt.query_coords,
                    query_valid_mask=rebuilt.query_valid_mask,
                    target_fields=rebuilt.target_fields, metadata=metadata)
    batch.validate()
    return batch


def endpoints(model, batch, config, state, seed, device):
    values = []
    for member in range(2):
        generator = torch.Generator(device=device)
        if member == 0:
            generator.set_state(state)
        else:
            generator.manual_seed(_conditional_draw_seed(seed, member))
        values.append(differentiable_reconstruction(model, _without_target(batch),
            steps=4, solver='euler', generator=generator,
            observation_config=dict(config['observation_consistency'])))
    return torch.stack(values, dim=1)


def risk(members, target, mass, total, probability=None):
    sampling = {'total_mass': total, 'sampling': 'full' if probability is None else 'without_replacement'}
    if probability is not None:
        sampling['probabilities'] = probability
    member, signed = physical_ensemble_risks(members, target, mass, **sampling)
    raw = [physical_field_mse(members[:, j], target, mass, **sampling)[0].cpu().tolist() for j in range(2)]
    return {'member_mse_each': raw, 'member_mse_mean': member[0].cpu().tolist(),
            'signed_mean_risk': signed[0].cpu().tolist()}


def compare(a, b, repeat_a, repeat_b):
    difference = (a-b).abs()
    floor = torch.maximum((a-repeat_a).abs(), (b-repeat_b).abs())
    tolerance = 5*floor + 1e-6 + 1e-5*torch.maximum(a.abs(), b.abs())
    return {'max_abs_difference': float(difference.max()),
            'field_max_abs_difference': difference.amax(dim=(0, 1, 2)).cpu().tolist(),
            'max_repeated_incumbent_difference': float(floor.max()),
            'max_tolerance': float(tolerance.max()),
            'all_within_repeatability_tolerance': bool((difference <= tolerance).all()),
            'tolerance_definition': '5*max(repeat absolute differences)+1e-6+1e-5*max(abs(endpoints))'}


def run(args):
    if not os.environ.get('CAR_ROUND7_LEASE_TOKEN'):
        raise ValueError('a Round 7 preparation lease is required')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    receipt = {'schema': 'car_round7_support_fixture_v1', 'status': 'failed',
               'config_sha256': digest(args.config), 'start_unix': started}
    source = dataset = None
    try:
        torch.set_num_threads(4)
        cfg = yaml.safe_load(args.config.read_text())
        if cfg['rollout']['steps'] != 4 or cfg['rollout']['solver'] != 'euler':
            raise ValueError('four Euler steps are required')
        model, dataset, metadata = load_source_model(cfg, args.device)
        post_training_mode(model, cfg)
        if model.capabilities.structured_grid_required:
            raise ValueError('fixture requires a query-support model')
        panel_ref = cfg['coherence']['scalable_anchored_coherence']['common_core']
        if digest(Path(panel_ref['source_panel_path'])) != panel_ref['source_panel_sha256']:
            raise ValueError('SOURCE panel hash differs')
        panel = torch.load(panel_ref['source_panel_path'], map_location='cpu', weights_only=True)
        index = int(panel['sample_indices'][0])
        source = build_training_batch_source(dataset, [panel['sample_indices'][:64]], cfg, query_points=4096,
                                              device=args.device, start_step=0)
        # Slice the actual first B64 producer batch, including its original
        # padded observation width, before any model endpoint is evaluated.
        batch = _slice_batch(next(iter(source)), 1)
        if list(batch.sample_ids) != [panel['sample_ids'][0]]:
            raise ValueError('TRAIN panel condition changed')
        seed = int(cfg['runtime']['seed']) + 1_000_003
        generator = torch.Generator(device=args.device).manual_seed(seed)
        selected = subset_query_batch(batch, 4096, generator=generator)
        initial_state = generator.get_state().clone()
        selected_sites = selected.metadata['query_indices'].cpu()
        n = dataset.physical_weights.numel()
        qsettings = cfg['optimization']['multitask']['fidelity_budget']['rotating_complement']
        complement = complement_indices(selected_sites, n, qsettings['points'], seed=qsettings['seed'], absolute_step=0)
        if qsettings['points'] != 512 or selected_sites.shape != (1, 4096):
            raise ValueError('expected Q4096 and C512')
        sample = dataset[index]
        comp_batch = support_batch(selected, sample, complement[0], cfg, args.device)
        native_batch = support_batch(selected, sample, torch.arange(n), cfg, args.device)
        with torch.no_grad():
            selected_members = endpoints(model, selected, cfg, initial_state, seed, args.device)
            selected_repeat = endpoints(model, selected, cfg, initial_state, seed, args.device)
            comp_members = endpoints(model, comp_batch, cfg, initial_state, seed, args.device)
            comp_repeat = endpoints(model, comp_batch, cfg, initial_state, seed, args.device)
            native_members = endpoints(model, native_batch, cfg, initial_state, seed, args.device)
            native_repeat = endpoints(model, native_batch, cfg, initial_state, seed, args.device)
            s = selected_sites.to(args.device)[0]
            c = complement.to(args.device)[0]
            if not torch.equal(selected.target_fields, native_batch.target_fields[:, s]) or not torch.equal(comp_batch.target_fields, native_batch.target_fields[:, c]):
                raise ValueError('query supports have different reference values')
            mass = torch.as_tensor(dataset.physical_weights, device=args.device, dtype=selected_members.dtype)
            combined_members = torch.cat((selected_members, comp_members), dim=2)
            combined_truth = torch.cat((selected.target_fields, comp_batch.target_fields), dim=1)
            sites = torch.cat((s, c))
            probability = torch.cat((torch.ones(len(s), device=args.device),
                torch.full((len(c),), len(c)/(n-len(s)), device=args.device)))
            native_combined = native_members[:, :, sites]
            results = {
                'selected_vs_native_at_selected': compare(selected_members, native_members[:, :, s], selected_repeat, native_repeat[:, :, s]),
                'complement_vs_native_at_complement': compare(comp_members, native_members[:, :, c], comp_repeat, native_repeat[:, :, c]),
                'selected_support_integral': risk(selected_members, selected.target_fields, mass[s], mass[s].sum()),
                'complement_support_integral': risk(comp_members, comp_batch.target_fields, mass[c], mass[c].sum()),
                'training_stratified': risk(combined_members, combined_truth, mass[sites], mass.sum(), probability),
                'native_predictions_training_stratified': risk(native_combined, combined_truth, mass[sites], mass.sum(), probability),
                'native_full_integral': risk(native_members, native_batch.target_fields, mass, mass.sum()),
            }
        artifact = {'selected_indices': selected_sites, 'complement_indices': complement,
            'selected_members': selected_members.cpu(), 'complement_members': comp_members.cpu(),
            'native_members': native_members.cpu(), 'native_target': native_batch.target_fields.cpu()}
        torch.save(artifact, args.output/'endpoints.pt')
        receipt.update(status='completed', sample_id=batch.sample_ids[0], sample_index=index,
            field_order=list(dataset.field_names), selected_points=len(s), complement_points=len(c), native_points=n,
            original_observation_sha256={key: _digest(getattr(selected, key)) for key in
                ('obs_indices', 'obs_field_ids', 'obs_coords', 'obs_values', 'obs_valid_mask')},
            rollout_seed=seed, member_second_seed=_conditional_draw_seed(seed, 1),
            first_member_generator_state_sha256=hashlib.sha256(initial_state.cpu().numpy().tobytes()).hexdigest(),
            source_identity=source_hashes(cfg), source_metadata=metadata, results=results,
            endpoints={'path': str((args.output/'endpoints.pt').resolve()), 'sha256': digest(args.output/'endpoints.pt')},
            interpretation='One TRAIN condition; support sensitivity is an operator diagnostic, not a population estimate.')
    except BaseException as exc:
        receipt.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if source is not None: source.close()
        if dataset is not None: dataset.close()
        receipt.update(end_unix=time.time(), wall_seconds=time.time()-started)
        atomic_json(args.output/'receipt.json', receipt)
    print(args.output/'receipt.json', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=torch.device, required=True)
    run(p.parse_args())

if __name__ == '__main__':
    main()
