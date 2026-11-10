"""Evaluate SOURCE on the frozen TRAIN800/VALIDATION100 Round 7 observers."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import time


def evaluate_source(config, *, manifest_path, positions_path, fair_artifact, output_dir, device='cuda:0',
                    s100_receipt=None, native_source=False):
    import torch
    from ..coherence.compose import build_enabled_families
    from ..data.factory import open_field_dataset
    from ..training.coherence_sampling import _fit_training_references
    from ..training.run_store import file_sha256, load_project_checkpoint, load_model_state_strict
    from ..training.source import load_source_model, source_hashes
    from .loss_aligned import evaluate_split, legacy_metrics, make_batches, model_state_digest
    from .round7_calibrate import SOURCE_SHA

    if source_hashes(config)['checkpoint'] != SOURCE_SHA:
        raise ValueError('SOURCE checkpoint changed')
    manifest = json.loads(Path(manifest_path).read_text())
    positions = json.loads(Path(positions_path).read_text())
    output = Path(output_dir)
    model, train, _ = load_source_model(config, torch.device(device))
    validation = None
    try:
        families = {key: value.to(device) for key, value in
                    build_enabled_families(config['coherence'], train.data_spec, train.normalizer).items()}
        _fit_training_references(families, train, config, device=torch.device(device), structured_grid_required=False)
        legacy = families['topology']
        fair = None
        if fair_artifact is not None:
            settings = copy.deepcopy(config['coherence'])
            settings['families']['topology']['persistence']['definition'] = 'conditional_fixed_transform_fair_v1'
            fair = build_enabled_families(settings, train.data_spec, train.normalizer)['topology'].to(device)
            artifact = torch.load(fair_artifact, map_location=device, weights_only=True)
            fair.load_state_artifact(artifact)
        validation = open_field_dataset(config['dataset'], split='validation', normalizer=train.normalizer)
        bank = __import__('phycoflow_reconstruction.coherence.relation_queries', fromlist=['QueryBank']).QueryBank(
            train.field_names, seed=int(config['coherence']['query_coverage']['relation_seed']))
        observers = {'legacy': legacy}
        identities = {'legacy': config['coherence']['frozen_family_artifacts']['topology']['sha256']}
        if fair is not None:
            observers['fair'] = fair
            identities['fair'] = file_sha256(fair_artifact)
        started = time.perf_counter()
        result = {}
        checkpoint_sha = SOURCE_SHA
        panels = [('train', train), ('validation', validation)]
        label = 'SOURCE'
        if s100_receipt is not None:
            saved = json.loads(Path(s100_receipt).read_text())
            if saved.get('status') != 'completed' or file_sha256(saved['endpoint']) != saved['endpoint_sha256']:
                raise ValueError('S100 endpoint receipt or checkpoint mismatch')
            checkpoint = load_project_checkpoint(saved['endpoint'])
            load_model_state_strict(model, checkpoint['model'])
            if model_state_digest(model) != saved['model_state_sha256']:
                raise ValueError('S100 loaded model identity mismatch')
            checkpoint_sha, label, panels = saved['endpoint_sha256'], 'S100', [('validation', validation)]
        for split, dataset in panels:
            path = evaluate_split(model=model, batches=make_batches(dataset, positions[split], config, 812711, device),
                families=families, config=config, manifest=manifest, split=split,
                checkpoint_sha256=model_state_digest(model), weight_identity_kind='model_state_sha256',
                query_bank=bank, output_dir=output / split,
                round7_observers=observers, round7_observer_identities=identities)
            payload = json.loads(path.read_text())
            attractions = ([row['shape_observers']['fair']['diagnostics']['attraction_per_sample'][0] for row in payload['rows']]
                           if fair is not None else None)
            result[split] = {'reductions': str(path), 'sha256': file_sha256(path),
                             'conditions': len(payload['rows']), 'metrics': legacy_metrics(path, train.field_names),
                             'fair_shape_attraction_mean': sum(attractions) / len(attractions) if attractions else None}
        if fair is not None and label == 'SOURCE' and not result['train']['fair_shape_attraction_mean'] > 0:
            raise ValueError('SOURCE TRAIN fair Shape attraction scale must be positive')
        if native_source:
            if label != 'SOURCE':
                raise ValueError('native SOURCE denominator cannot use S100 weights')
            from types import SimpleNamespace
            from ..data.manifest import SensorManifest, manifest_from_batch
            from .loss_aligned import _batch_identity
            from .round7_native import evaluate_round7_native
            entries = []
            for batch in make_batches(validation, positions['validation'], config, 812711, 'cpu'):
                identity = _batch_identity(batch)
                frozen = next(row for row in manifest['splits']['validation'] if row['sample_id'] == identity['sample_id'])
                if any(frozen.get(k) != v for k, v in identity.items()):
                    raise ValueError('native sensor manifest differs from fixed Q4096 observations')
                entries.append(manifest_from_batch(batch, validation.path, 'validation'))
            first = entries[0]
            sensors = SensorManifest(dataset_path=first.dataset_path, dataset_fingerprint=first.dataset_fingerprint,
                split=first.split, protocol=first.protocol,
                indices={key: value for item in entries for key, value in item.indices.items()},
                query_indices={key: value for item in entries for key, value in item.query_indices.items()})
            sensor_path = output / 'validation_sensor_manifest.json'
            sensors.save(sensor_path)
            checkpoint = Path(config['source_run']) / 'checkpoints' / config['source_checkpoint']
            if file_sha256(checkpoint) != SOURCE_SHA:
                raise ValueError('native SOURCE checkpoint file differs')
            saved_model = load_project_checkpoint(checkpoint)['model']
            class SavedState:
                def state_dict(self):
                    return saved_model
            if model_state_digest(model) != model_state_digest(SavedState()):
                raise ValueError('native SOURCE live weights differ from saved checkpoint')
            runtime = SimpleNamespace(model=model, device=torch.device(device), config=config, checkpoint_path=checkpoint)
            endpoint = {'steps': config['rollout']['steps'], 'solver': config['rollout']['solver'],
                        'observation_config': config['observation_consistency'],
                        'weight_selection': 'live', 'sensor_manifest': sensors}
            native_receipt = evaluate_round7_native(runtime, validation, positions['validation'],
                output / 'native_validation', endpoint, panel_seed_base=812711)
            result['native_validation'] = {'receipt': str(native_receipt), 'sha256': file_sha256(native_receipt),
                                           'sensor_manifest': str(sensor_path), 'sensor_manifest_sha256': sensors.digest()}
        result.update({'schema': 'round7_reference_loss_support_v1', 'reference': label,
                       'checkpoint_sha256': checkpoint_sha, 'parent_source_checkpoint_sha256': SOURCE_SHA,
                       'manifest_sha256': manifest['sha256'], 'observer_identities': identities,
                       'wall_seconds': time.perf_counter() - started})
        return result
    finally:
        train.close()
        if validation is not None:
            validation.close()
