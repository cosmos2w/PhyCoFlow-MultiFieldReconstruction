"""One frozen Round 7 TEST100 comparison; requires a pre-TEST selection receipt."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from types import SimpleNamespace


def validate_completed_test(receipt_path, selection_path):
    """Validate a finished TEST receipt for report-only retries, without inference."""
    from ..training.run_store import file_sha256
    from .round7_native import _load_receipt, compare_round7_native
    receipt = json.loads(Path(receipt_path).read_text())
    selection = json.loads(Path(selection_path).read_text())
    if receipt.get('schema') != 'round7_fixed_test_v1' or receipt.get('selection_sha256') != file_sha256(selection_path):
        raise ValueError('completed TEST does not match the frozen selection')
    expected = {selection['source_checkpoint_sha256']}
    for item in (selection.get('B1000'), selection.get('endpoint')):
        if item:
            expected.add(item['checkpoint_sha256'])
    actual = {item['checkpoint_sha256'] for item in receipt['endpoints'].values()}
    if expected != actual:
        raise ValueError('completed TEST endpoint set differs from frozen selection')
    for item in receipt['endpoints'].values():
        for key in ('reductions', 'native_receipt'):
            if file_sha256(item[key]) != item[key + '_sha256']:
                raise ValueError('completed TEST artifact changed')
        q = json.loads(Path(item['reductions']).read_text())
        if (q['split'] != 'test' or q['manifest_sha256'] != receipt['manifest_sha256']
                or q['observer_identities'] != receipt['observer_identities'] or len(q['rows']) != 100):
            raise ValueError('completed TEST Q4096 contract changed')
        native, _ = _load_receipt(Path(item['native_receipt']).parent)
        if (native['checkpoint_sha256'] != item['checkpoint_sha256']
                or native['sensor_manifest_sha256'] != receipt['sensor_manifest_sha256']):
            raise ValueError('completed native TEST identity changed')
    for label, item in receipt['endpoints'].items():
        if label != 'SOURCE':
            actual = compare_round7_native(Path(receipt['endpoints']['SOURCE']['native_receipt']).parent,
                                           Path(item['native_receipt']).parent)
            if receipt.get('native_comparisons', {}).get(label) != actual:
                raise ValueError('completed native TEST comparison changed')
    return receipt


def evaluate_frozen_test(config, *, selection_path, manifest_path, positions_path, fair_artifact,
                         output_root, device='cuda:0'):
    import torch
    from ..coherence.compose import build_enabled_families
    from ..coherence.relation_queries import QueryBank
    from ..data.factory import open_field_dataset
    from ..data.manifest import SensorManifest, manifest_from_batch
    from ..training.coherence_sampling import _fit_training_references
    from ..training.run_store import file_sha256, load_model_state_strict, load_project_checkpoint
    from ..training.source import load_source_model, source_hashes
    from .loss_aligned import _batch_identity, evaluate_split, make_batches, model_state_digest
    from .round7_native import evaluate_round7_native, compare_round7_native, _load_receipt
    from .round7_calibrate import SOURCE_SHA
    selection = json.loads(Path(selection_path).read_text())
    if selection.get('status') != 'frozen_before_test' or selection.get('source_checkpoint_sha256') != SOURCE_SHA:
        raise ValueError('TEST requires a frozen selection and immutable SOURCE')
    if file_sha256(selection['decision_input']) != selection['decision_input_sha256']:
        raise ValueError('frozen selection evidence changed')
    if source_hashes(config)['checkpoint'] != SOURCE_SHA:
        raise ValueError('SOURCE checkpoint changed')
    manifest = json.loads(Path(manifest_path).read_text())
    positions = json.loads(Path(positions_path).read_text())['test']
    root = Path(output_root)
    model, train, _ = load_source_model(config, torch.device(device))
    test = None
    try:
        families = {k: f.to(device) for k, f in build_enabled_families(config['coherence'], train.data_spec, train.normalizer).items()}
        _fit_training_references(families, train, config, device=torch.device(device), structured_grid_required=False)
        test = open_field_dataset(config['dataset'], split='test', normalizer=train.normalizer)
        legacy = families['topology']
        observers = {'legacy': legacy}
        identities = {'legacy': config['coherence']['frozen_family_artifacts']['topology']['sha256']}
        if fair_artifact is not None:
            settings = copy.deepcopy(config['coherence'])
            settings['families']['topology']['persistence']['definition'] = 'conditional_fixed_transform_fair_v1'
            fair = build_enabled_families(settings, train.data_spec, train.normalizer)['topology'].to(device)
            fair.load_state_artifact(torch.load(fair_artifact, map_location=device, weights_only=True))
            observers['fair'] = fair
            identities['fair'] = file_sha256(fair_artifact)
        entries = []
        for batch in make_batches(test, positions, config, 812711, 'cpu'):
            identity = _batch_identity(batch)
            frozen = next(r for r in manifest['splits']['test'] if r['sample_id'] == identity['sample_id'])
            if any(frozen.get(k) != v for k, v in identity.items()):
                raise ValueError('TEST condition differs from frozen manifest')
            entries.append(manifest_from_batch(batch, test.path, 'test'))
        first = entries[0]
        sensor = SensorManifest(dataset_path=first.dataset_path, dataset_fingerprint=first.dataset_fingerprint,
            split='test', protocol=first.protocol,
            indices={k: v for entry in entries for k, v in entry.indices.items()},
            query_indices={k: v for entry in entries for k, v in entry.query_indices.items()})
        root.mkdir(parents=True, exist_ok=True)
        sensor_path = root / 'sensor_manifest.json'
        if sensor_path.exists():
            if SensorManifest.load(sensor_path).digest() != sensor.digest():
                raise ValueError('frozen TEST sensors changed')
        else:
            sensor.save(sensor_path)
        bank = QueryBank(train.field_names, seed=int(config['coherence']['query_coverage']['relation_seed']))
        source_checkpoint = Path(config['source_run']) / 'checkpoints' / config['source_checkpoint']
        endpoints = [('SOURCE', {'checkpoint': str(source_checkpoint), 'checkpoint_sha256': SOURCE_SHA})]
        for label, item in [('B1000', selection.get('B1000')), ('selected', selection.get('endpoint'))]:
            if item is not None:
                binding = json.loads(Path(item['binding']).read_text())
                if (file_sha256(item['binding']) != item['binding_sha256']
                        or binding['checkpoint_sha256'] != item['checkpoint_sha256']
                        or binding['checkpoint'] != item['checkpoint']):
                    raise ValueError('frozen TEST endpoint binding changed')
            if item is not None and all(item['checkpoint_sha256'] != existing['checkpoint_sha256'] for _, existing in endpoints):
                endpoints.append((label, item))
        result = {}
        for label, item in endpoints:
            checkpoint = Path(item['checkpoint'])
            if file_sha256(checkpoint) != item['checkpoint_sha256']:
                raise ValueError('frozen TEST checkpoint changed')
            state = load_project_checkpoint(checkpoint)['model']
            load_model_state_strict(model, state)
            class SavedState:
                def state_dict(self):
                    return state
            if model_state_digest(model) != model_state_digest(SavedState()):
                raise ValueError('TEST live weights differ from pinned checkpoint')
            path = evaluate_split(model=model, batches=make_batches(test, positions, config, 812711, device),
                families=families, config=config, manifest=manifest, split='test',
                checkpoint_sha256=model_state_digest(model), weight_identity_kind='model_state_sha256',
                query_bank=bank, output_dir=root / label / 'q4096',
                round7_observers=observers, round7_observer_identities=identities)
            runtime = SimpleNamespace(model=model, device=torch.device(device), config=config, checkpoint_path=checkpoint)
            endpoint = {'steps': config['rollout']['steps'], 'solver': config['rollout']['solver'],
                        'observation_config': config['observation_consistency'], 'weight_selection': 'live',
                        'sensor_manifest': sensor}
            native_dir = root / label / 'native'
            if native_dir.exists():
                saved, _ = _load_receipt(native_dir)
                if saved['checkpoint_sha256'] != item['checkpoint_sha256'] or saved['sensor_manifest_sha256'] != sensor.digest():
                    raise ValueError('cached TEST native endpoint identity mismatch')
                native = native_dir / 'round7_native_receipt.json'
            else:
                native = evaluate_round7_native(runtime, test, positions, native_dir, endpoint, panel_seed_base=812711)
            result[label] = {'checkpoint_sha256': item['checkpoint_sha256'], 'reductions': str(path),
                             'reductions_sha256': file_sha256(path), 'native_receipt': str(native),
                             'native_receipt_sha256': file_sha256(native)}
        native_comparisons = {label: compare_round7_native(Path(result['SOURCE']['native_receipt']).parent,
                                                          Path(item['native_receipt']).parent)
                              for label, item in result.items() if label != 'SOURCE'}
        return {'schema': 'round7_fixed_test_v1', 'selection_sha256': file_sha256(selection_path),
                'manifest_sha256': manifest['sha256'], 'observer_identities': identities,
                'sensor_manifest_sha256': sensor.digest(), 'endpoints': result,
                'native_comparisons': native_comparisons}
    finally:
        train.close()
        if test is not None:
            test.close()
