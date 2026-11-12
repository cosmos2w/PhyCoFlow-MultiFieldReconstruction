"""Stream frozen Round 8 endpoint fields and rescore without model inference.

Every condition is an independent shard; identities bind the actual observed
points and actual initial fields, not merely their pseudo-random seeds.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .native_coherence import _digest_arrays, _physical_weights
from .round8_campaign import canonical_hash, file_hash, validate_t2000_receipt
from ..data.factory import open_field_dataset
from ..data.manifest import dataset_fingerprint
from ..data.sensor_protocols import build_observation_batch
from ..data.training_batches import fixed_query_indices
from ..training.common import sensor_protocol_from_config
from ..training.model_lifecycle import load_training_aux_state
from ..training.post_training import _without_target
from ..training.rollout import differentiable_reconstruction
from ..training.run_store import load_model_state_strict, load_project_checkpoint
from ..training.source import load_source_model


def _tensor_hash(value):
    return _digest_arrays(value.detach().cpu().numpy())


def prior_identity(model):
    """Bind all prior buffers and parameters as well as prior implementation."""
    prior = getattr(model, 'prior', None)
    if prior is None:
        raise ValueError('model must expose its source prior')
    values = {name: _tensor_hash(value) for name, value in prior.state_dict().items()}
    return canonical_hash({'type': type(prior).__module__ + '.' + type(prior).__qualname__, 'state': values})


@contextmanager
def _capture_initial(model):
    """Intercept the field actually consumed by the rollout, without a second draw."""
    if not callable(getattr(model, 'sample_source', None)) or not hasattr(model, 'velocity'):
        raise ValueError('paired inference requires an interceptable flow source')
    original = model.sample_source
    had_instance = 'sample_source' in model.__dict__
    previous = model.__dict__.get('sample_source')
    captured = []
    def sample(*args, **kwargs):
        value = original(*args, **kwargs)
        captured.append(_tensor_hash(value))
        return value
    model.sample_source = sample
    try:
        yield captured
    finally:
        if had_instance:
            model.sample_source = previous
        else:
            delattr(model, 'sample_source')


def _freeze_json(path, value):
    path = Path(path)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(encoded)


def _check_inputs(config, manifest):
    from ..training.source import source_checkpoint_path
    from ..data.manifest import dataset_fingerprint
    identity = manifest['dataset_identity']
    if dataset_fingerprint(config['dataset']['path']) != identity['structural_fingerprint_sha256']:
        raise ValueError('dataset structural fingerprint differs')
    for record in manifest['inputs'].values():
        if file_hash(record['path']) != record['sha256']:
            raise ValueError('frozen preparation input differs')
    if file_hash(source_checkpoint_path(config)) != manifest['inputs']['source_checkpoint']['sha256']:
        raise ValueError('configured SOURCE differs from manifest')
    if manifest['sampling']['members'] != 2:
        raise ValueError('Round 8 requires J=2')


def _load_model(config, manifest, device, endpoint, checkpoint, t_receipt):
    _check_inputs(config, manifest)
    model, train, _ = load_source_model(config, torch.device(device))
    source_prior = prior_identity(model)
    epoch = None
    if endpoint == 'SOURCE':
        if checkpoint is not None or t_receipt is not None:
            raise ValueError('SOURCE uses only the frozen source checkpoint')
        digest = manifest['inputs']['source_checkpoint']['sha256']
    else:
        if endpoint not in ('T2000', 'B8', 'C8') or checkpoint is None:
            raise ValueError('specify a declared endpoint and checkpoint')
        if endpoint == 'T2000':
            if t_receipt is None:
                raise ValueError('T2000 completion receipt required')
            verified = validate_t2000_receipt(t_receipt)
            if Path(checkpoint).resolve() != Path(verified['checkpoint']).resolve():
                raise ValueError('T2000 checkpoint does not match completed receipt')
        payload = load_project_checkpoint(checkpoint)
        if endpoint == 'T2000':
            epoch = 2000
        else:
            step = payload.get('global_step')
            if not isinstance(step, int) or step % 13 or step // 13 not in (250, 500, 750, 1000):
                raise ValueError('B8/C8 checkpoint must be at a declared assessment epoch')
            epoch = step // 13
        load_model_state_strict(model, payload['model'])
        load_training_aux_state(model, payload)
        digest = file_hash(checkpoint)
        if prior_identity(model) != source_prior:
            raise ValueError('endpoint prior buffers differ from SOURCE')
    model.eval()
    return model, train, digest, source_prior, epoch


def infer_panel(config, manifest_path, output_dir, *, population, endpoint='SOURCE', checkpoint=None,
                t_receipt=None, device='cuda:0', profiling_first_condition=False):
    """Run and save one declared 800- or 200-condition matched endpoint panel."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'round8_manifest_v1' or population not in ('TRAIN', 'ASSESS'):
        raise ValueError('invalid frozen Round 8 manifest or population')
    expected = 800 if population == 'TRAIN' else 200
    rows = manifest[population]
    if len(rows) != expected:
        raise ValueError('population cardinality differs from frozen contract')
    if profiling_first_condition:
        if population != 'ASSESS' or endpoint != 'SOURCE':
            raise ValueError('qualification profiling uses one frozen ASSESS SOURCE condition')
        rows = rows[:1]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    model, train, checkpoint_sha, prior_sha, epoch = _load_model(config, manifest, device, endpoint, checkpoint, t_receipt)
    if population == 'TRAIN' and endpoint in ('B8', 'C8') and epoch not in (500, 1000):
        train.close()
        raise ValueError('TRAIN comparison is scheduled only at epochs 500 and 1000')
    datasets = {'train': train}
    started = time.perf_counter()
    condition_records = []
    try:
        if population == 'ASSESS':
            for split in ('validation', 'test'):
                datasets[split] = open_field_dataset(config['dataset'], split=split, normalizer=train.normalizer)
        protocol = sensor_protocol_from_config(config)
        risk_rows = []
        with torch.no_grad():
            for ordinal, row in enumerate(rows):
                split = row.get('source', 'train')
                dataset = datasets[split]
                sample = dataset[int(row['dataset_index'])]
                sample_id = f'{sample.trajectory_id}:{sample.time_index if sample.time_index is not None else "all"}'
                if sample_id != row['id']:
                    raise ValueError('dataset position differs from frozen sample identity')
                n = sample.values.shape[0]
                if n != 40300:
                    raise ValueError('case expects 40300 native points')
                weights, mask, measure = _physical_weights(sample)
                if not mask.all():
                    raise ValueError('sampled design requires declared compact all-valid physical support')
                if population == 'TRAIN':
                    if row['spatial_seed'] != 100045:
                        raise ValueError('TRAIN must use the frozen R7 Q4096 geometry')
                    query = fixed_query_indices(n, 4096, seed=100045)
                    inclusion = 4096 / n
                else:
                    query = torch.arange(n)
                    inclusion = 1.0
                batch = build_observation_batch([sample], replace(protocol, seed=row['observation_seed']),
                                                query_indices=query).to(device)
                inference = _without_target(batch)
                values = []
                latent_hashes = []
                for seed in row['member_seeds']:
                    generator = torch.Generator(device=torch.device(device)).manual_seed(seed)
                    with _capture_initial(model) as initial:
                        predicted = differentiable_reconstruction(model, inference,
                            steps=int(config['rollout']['steps']), solver=config['rollout']['solver'],
                            generator=generator, observation_config=dict(config['observation_consistency']))
                    if len(initial) != 1:
                        raise ValueError('rollout did not consume exactly one initial random field')
                    latent_hashes.append(initial[0])
                    values.append(predicted[0].detach().cpu())
                if latent_hashes[0] == latent_hashes[1]:
                    raise ValueError('independent member inputs unexpectedly match')
                draws = torch.stack(values)
                target = batch.target_fields[0].detach().cpu()
                if draws.shape != (2, query.numel(), len(dataset.field_names)):
                    raise ValueError('endpoint output shape mismatch')
                selected = query.numpy()
                spatial_weights = weights[selected]
                physical_draws = dataset.normalizer.decode(draws).double().numpy()
                physical_target = dataset.normalizer.decode(target).double().numpy()
                mse = (((physical_draws - physical_target[None]) ** 2) *
                       (spatial_weights / inclusion)[None, :, None]).sum(axis=1)
                if not np.isfinite(mse).all():
                    raise ValueError('nonfinite physical risk')
                arrays = {'draws_model': draws.numpy(), 'target_model': target.numpy(),
                    'draws_physical': physical_draws, 'target_physical': physical_target,
                    'model_coordinates': batch.query_coords[0].cpu().numpy(),
                    'raw_coordinates': sample.coordinates_raw[query].cpu().numpy(),
                    'native_ids': selected, 'physical_mask': mask[selected], 'physical_weights': spatial_weights,
                    'inclusion_probability': np.asarray(inclusion),
                    'observation_coords': batch.obs_coords[0].cpu().numpy(),
                    'observation_values': batch.obs_values[0].cpu().numpy(),
                    'observation_field_ids': batch.obs_field_ids[0].cpu().numpy(),
                    'observation_indices': batch.obs_indices[0].cpu().numpy(),
                    'observation_mask': batch.obs_valid_mask[0].cpu().numpy(),
                    'draw_seeds': np.asarray(row['member_seeds'], dtype=np.int64),
                    'field_names': np.asarray(dataset.field_names), 'sample_id': np.asarray(sample_id),
                    'field_units': np.asarray(dataset.data_spec.field_units),
                    'logical_shape': np.asarray(sample.logical_shape, dtype=np.int64),
                    'normalizer_sha256': np.asarray(dataset.normalizer.digest())}
                contract = _digest_arrays(*(arrays[k] for k in ('native_ids', 'model_coordinates',
                    'raw_coordinates', 'physical_mask', 'physical_weights', 'target_model',
                    'observation_coords', 'observation_values', 'observation_field_ids',
                    'observation_indices', 'observation_mask', 'draw_seeds')))
                shard = output / f'condition_{ordinal:05d}.npz'
                with shard.open('xb') as stream:
                    np.savez_compressed(stream, **arrays)
                condition_records.append({'sample_id': sample_id, 'source': split,
                    'dataset_index': row['dataset_index'], 'path': shard.name, 'sha256': file_hash(shard),
                    'coupling_sha256': contract, 'initial_field_sha256': latent_hashes,
                    'physical_measure': measure, 'member_mse_physical': mse.tolist()})
                risk_rows.append(mse)
        panel = {'schema': 'round8_endpoint_panel_v1', 'status': 'completed', 'endpoint': endpoint,
                 'population': population, 'epoch': epoch, 'manifest_sha256': file_hash(manifest_path),
                 'source_checkpoint_sha256': manifest['inputs']['source_checkpoint']['sha256'],
                 'checkpoint_sha256': checkpoint_sha, 'prior_sha256': prior_sha,
                 'solver_sha256': canonical_hash(config['rollout']),
                 'clamp_sha256': canonical_hash(config['observation_consistency']),
                 'normalizer_sha256': train.normalizer.digest(),
                 'field_names': list(train.field_names), 'conditions': condition_records,
                 'member_risk_physical': np.asarray(risk_rows).mean(axis=(0, 1)).tolist(),
                 'risk_design': 'HT_uniform_without_replacement' if population == 'TRAIN' else 'native',
                 'inclusion_probability': 4096/40300 if population == 'TRAIN' else 1.0,
                 'elapsed_seconds': time.perf_counter() - started}
        if profiling_first_condition:
            panel.update(schema='round8_one_condition_profile_v1', status='profiled',
                         usage='qualification timing/memory only; not a completed panel')
            _freeze_json(output / 'profile.json', panel)
            return output / 'profile.json'
        _freeze_json(output / 'panel.json', panel)
        return output / 'panel.json'
    except BaseException as exc:
        _freeze_json(output / 'failure_receipt.json', {
            'schema': 'round8_endpoint_failure_v1', 'status': 'incomplete',
            'endpoint': endpoint, 'population': population, 'manifest_sha256': file_hash(manifest_path),
            'checkpoint_sha256': checkpoint_sha, 'completed_conditions': condition_records,
            'error_type': type(exc).__name__, 'message': str(exc),
            'elapsed_seconds': time.perf_counter() - started})
        raise
    finally:
        for dataset in datasets.values():
            dataset.close()


def validate_panel(path):
    """Verify a completed panel and every retained condition shard."""
    path = Path(path)
    panel = json.loads(path.read_text())
    if panel.get('schema') != 'round8_endpoint_panel_v1' or panel.get('status') != 'completed':
        raise ValueError('completed Round 8 endpoint panel required')
    expected = 800 if panel['population'] == 'TRAIN' else 200 if panel['population'] == 'ASSESS' else 0
    if len(panel['conditions']) != expected or len({x['sample_id'] for x in panel['conditions']}) != expected:
        raise ValueError('panel condition count/identity differs')
    for row in panel['conditions']:
        if file_hash(path.parent / row['path']) != row['sha256']:
            raise ValueError('condition payload hash differs')
    return panel


def validate_matched_panels(source_path, endpoint_path):
    source, endpoint = validate_panel(source_path), validate_panel(endpoint_path)
    if source['endpoint'] != 'SOURCE':
        raise ValueError('first panel must be SOURCE')
    for key in ('population', 'manifest_sha256', 'source_checkpoint_sha256', 'prior_sha256',
                'solver_sha256', 'clamp_sha256', 'normalizer_sha256', 'field_names',
                'risk_design', 'inclusion_probability'):
        if source[key] != endpoint[key]:
            raise ValueError(f'panel coupling differs: {key}')
    for left, right in zip(source['conditions'], endpoint['conditions']):
        for key in ('sample_id', 'source', 'dataset_index', 'coupling_sha256', 'initial_field_sha256'):
            if left[key] != right[key]:
                raise ValueError(f'condition coupling differs: {key}')
    return source, endpoint


def score_cached_panel(config, source_panel_path, endpoint_panel_path, *, fair_shape_artifact,
                       output_path, alpha, device='cpu', assessment_policy=None,
                       interim_diagnostic=False, native_bundle=None, reference_only=False):
    """Score a matched blend on saved geometry with the complete common families.

    The declared fair Shape artifact must match the geometry of the payload;
    the family loader and forwards reject incompatible calibration/support.
    No inference, observer fitting from ASSESS, or score interpolation occurs.
    """
    from copy import deepcopy
    from ..coherence.compose import build_enabled_families
    from ..training.coherence_sampling import _fit_training_references, _family_context
    from ..training.post_training import _configure_persistence_workers
    from .loss_aligned import _collect_family_result, _diagnostic_value
    started = time.perf_counter()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be between zero and one')
    _configure_persistence_workers(config)
    source, endpoint = validate_matched_panels(source_panel_path, endpoint_panel_path)
    if source['population'] == 'ASSESS' and endpoint['endpoint'] != 'SOURCE':
        if interim_diagnostic:
            if (endpoint['endpoint'] not in ('B8', 'C8') or endpoint.get('epoch') not in (250, 500, 750)
                    or alpha not in (0., .5, 1.)):
                raise ValueError('interim assessment permits only B8/C8 SOURCE, raw, or alpha=0.5')
        else:
            if endpoint['endpoint'] in ('B8', 'C8') and endpoint.get('epoch') != 1000:
                raise ValueError('final assessment requires epoch 1000')
            if assessment_policy is None:
                raise ValueError('ASSESS endpoint scoring requires a TRAIN-frozen policy')
            policy = json.loads(Path(assessment_policy).read_text())
            if (policy.get('schema') != 'round8_frozen_policy_v1' or policy.get('population') != 'TRAIN'
                    or policy.get('manifest_sha256') != source['manifest_sha256']
                    or policy.get('endpoint_checkpoint_sha256') != endpoint['checkpoint_sha256']
                    or policy.get('source_checkpoint_sha256') != source['source_checkpoint_sha256']
                    or not isinstance(policy.get('families'), list)):
                raise ValueError('TRAIN policy is not bound to this endpoint and manifest')
            permitted = {0., .25, .5, .75, 1.}
            if policy.get('alpha') is not None:
                permitted.add(float(policy['alpha']))
            if reference_only:
                if (endpoint['endpoint'] != 'B8' or policy.get('alpha') is not None
                        or policy.get('reference_alpha') is None
                        or alpha != float(policy['reference_alpha'])):
                    raise ValueError('reference-only blend requires B8 with no nomination and frozen reference_alpha')
                permitted.add(alpha)
            if alpha not in permitted:
                raise ValueError('alpha not selected by the frozen policy or declared frontier')
    if reference_only and (source['population'] != 'ASSESS' or interim_diagnostic):
        raise ValueError('reference-only scoring requires final ASSESS')
    fair_path = Path(fair_shape_artifact)
    if not fair_path.is_file():
        raise FileNotFoundError('frozen TRAIN fair Shape artifact required')
    train = open_field_dataset(config['dataset'], split='train')
    try:
        families = {name: family.to(device) for name, family in
                    build_enabled_families(config['coherence'], train.data_spec, train.normalizer).items()}
        _fit_training_references(families, train, config, device=torch.device(device),
                                 structured_grid_required=False)
        fair_settings = deepcopy(config['coherence'])
        fair_settings['families']['topology']['persistence']['definition'] = 'conditional_fixed_transform_fair_v1'
        fair = build_enabled_families(fair_settings, train.data_spec, train.normalizer)['topology'].to(device)
        fair.load_state_artifact(torch.load(fair_path, map_location=device, weights_only=True))
        families['topology'] = fair
        native_identity = None
        if source['population'] == 'ASSESS':
            if native_bundle is None:
                raise ValueError('ASSESS scoring requires the declared native TRAIN bundle')
            from .round8_native_observers import native_common_families
            families, native_identity = native_common_families(config, train, families, fair, native_bundle, device=device)
        if set(families) != {'global_distribution', 'multiscale_spectrum', 'topology'}:
            raise ValueError('complete common State/Scale/fair Shape families required')
        results = []
        risks = []
        with torch.no_grad():
            for srow, erow in zip(source['conditions'], endpoint['conditions']):
                with np.load(Path(source_panel_path).parent / srow['path'], allow_pickle=False) as a, \
                     np.load(Path(endpoint_panel_path).parent / erow['path'], allow_pickle=False) as b:
                    if not all(np.array_equal(a[k], b[k]) for k in ('target_model', 'native_ids',
                            'model_coordinates', 'raw_coordinates', 'physical_mask', 'physical_weights')):
                        raise ValueError('cached field support or target differs')
                    physical = a['draws_physical'] + alpha * (b['draws_physical'] - a['draws_physical'])
                    risk = (((physical - a['target_physical'][None]) ** 2) *
                            (a['physical_weights'] / float(a['inclusion_probability']))[None, :, None]).sum(axis=1)
                    if not np.isfinite(risk).all():
                        raise ValueError('nonfinite cached physical risk')
                    risks.append(risk)
                    predictions = torch.from_numpy(a['draws_model'] + alpha * (b['draws_model'] - a['draws_model'])).to(device)
                    target = torch.from_numpy(a['target_model']).unsqueeze(0).to(device)
                    coords = torch.from_numpy(a['model_coordinates']).unsqueeze(0).to(device)
                    weights = torch.from_numpy(a['physical_weights']).unsqueeze(0).to(device)
                    mask = torch.from_numpy(a['physical_mask']).unsqueeze(0).to(device)
                    context = {'sample_ids': (srow['sample_id'],), 'reference_ids': (srow['sample_id'],),
                               'generated_weights': weights, 'reference_weights': weights,
                               'generated_mask': mask, 'reference_mask': mask,
                               'measure_metadata': {'native_measure_sha256': train.physical_measure_receipt['sha256']}}
                    losses, components, leaves, diagnostic = {}, {}, {}, {}
                    for name, family in families.items():
                        local = _family_context(family, context, [predictions[1:2]], report_detail=True)
                        # No sampled query selection: score the declared complete common observer.
                        result = family(predictions[0:1], target, coordinates=coords, context=local)
                        _collect_family_result(name, result, losses, components, leaves)
                        diagnostic[name] = {}
                        for key, value in result.diagnostics.items():
                            try:
                                diagnostic[name][key] = _diagnostic_value(value)
                            except (TypeError, ValueError):
                                continue
                    results.append({'sample_id': srow['sample_id'], 'legacy_source': srow['source'],
                                    'losses': losses, 'components': components, 'query_leaf_diagnostics': leaves, 'diagnostics': diagnostic,
                                    'member_mse_physical': risk.tolist()})
        report = {'schema': 'round8_common_observers_v1', 'population': source['population'],
                  'source_panel_sha256': file_hash(source_panel_path),
                  'endpoint_panel_sha256': file_hash(endpoint_panel_path), 'alpha': alpha,
                  'config_sha256': canonical_hash({key: config[key] for key in ('dataset', 'coherence', 'rollout', 'observation_consistency')}),
                  'frozen_family_artifacts_sha256': {name: file_hash(spec['path'])
                      for name, spec in config['coherence'].get('frozen_family_artifacts', {}).items()},
                  'fair_shape_artifact_sha256': file_hash(fair_path),
                  'assessment_policy_sha256': file_hash(assessment_policy) if assessment_policy else None,
                  'interim_diagnostic': interim_diagnostic, 'reference_only': reference_only,
                  'native_observer_identity': native_identity, 'rows': results,
                  'member_risk_physical': np.asarray(risks).mean(axis=(0, 1)).tolist(),
                  'support': 'Q4096' if source['population'] == 'TRAIN' else 'full_native_40300',
                  'scoring_seconds': time.perf_counter() - started,
                  'scoring_device': str(device)}
        _freeze_json(output_path, report)
        return Path(output_path)
    finally:
        train.close()
