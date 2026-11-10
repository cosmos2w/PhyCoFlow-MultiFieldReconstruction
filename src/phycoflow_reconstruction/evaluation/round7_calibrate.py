"""Stream the frozen TRAIN800 panel to fit one Round 7 Shape observer.

Invocation requires a measured, explicitly declared descriptor floor in model
or physical units. This stage does not fit the original fidelity reference.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import time

SOURCE_SHA = '2d1da713538f6e9613560f3a7eb4a9df2a95d1ad1f56dd11a1a8a4f9ef65050c'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--positions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--floor', type=float, required=True, help='positive descriptor floor in declared units')
    parser.add_argument('--floor-units', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    if args.floor <= 0 or not args.floor_units.strip():
        parser.error('positive floor and declared units required')
    if not os.environ.get('KEOPS_CACHE_FOLDER'):
        parser.error('set writable device-specific KEOPS_CACHE_FOLDER before Python starts')
    import torch
    import yaml
    from ..coherence.compose import build_enabled_families
    from ..data.manifest import dataset_fingerprint
    from ..data.training_batches import fixed_query_indices
    from ..evaluation.loss_aligned import _batch_identity, _digest, make_batches
    from ..training.post_training import _without_target
    from ..training.rollout import differentiable_reconstruction
    from ..training.source import load_source_model, source_hashes

    config = yaml.safe_load(Path(args.config).read_text())
    if source_hashes(config)['checkpoint'] != SOURCE_SHA:
        raise ValueError('immutable SOURCE checkpoint does not match')
    manifest = json.loads(Path(args.manifest).read_text())
    if manifest.get('sha256') != _digest({k: v for k, v in manifest.items() if k != 'sha256'}):
        raise ValueError('frozen manifest hash mismatch')
    frozen = {r['sample_id']: r for r in manifest['splits']['train']}
    positions = json.loads(Path(args.positions).read_text())['train']
    if len(frozen) != 800 or len(positions) != 800 or len(set(positions)) != 800:
        raise ValueError('exact frozen TRAIN800 panel required')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('frozen observer already exists')
    device = torch.device(args.device)
    started = time.time()
    model, train, _ = load_source_model(config, device)
    try:
        setting = copy.deepcopy(config['coherence'])
        setting['families']['topology']['persistence']['definition'] = 'conditional_fixed_transform_fair_v1'
        family = build_enabled_families(setting, train.data_spec, train.normalizer)['topology'].to(device)
        query = fixed_query_indices(40300, 4096, seed=100045)
        family.fit_training_reference(train, query_indices=query, device=device)
        legacy_decl = config['coherence']['frozen_family_artifacts']['topology']
        from ..training.run_store import file_sha256
        if file_sha256(legacy_decl['path']) != legacy_decl['sha256']:
            raise ValueError('historical topology artifact changed')
        legacy = torch.load(legacy_decl['path'], map_location='cpu', weights_only=True)
        if family.geometry_sha256 != legacy['geometry_sha256']:
            raise ValueError('new Shape geometry differs from frozen legacy observer')
        for name in ('neighbor_indices', 'neighbor_weights', 'grid_coordinates'):
            if not torch.equal(getattr(family, name).detach().cpu(), legacy['state_dict'][name].cpu()):
                raise ValueError(f'new Shape raster differs from legacy observer: {name}')
        records = []
        for batch in make_batches(train, positions, config, 812711, device):
            identity = _batch_identity(batch)
            record = frozen.get(identity['sample_id'])
            if record is None or any(record.get(k) != v for k, v in identity.items()):
                raise ValueError('TRAIN condition does not match frozen manifest')
            family.fair_accumulate_transform(batch.target_fields, coordinates=batch.query_coords)
            records.append(identity['sample_id'])
        if len(set(records)) != 800 or set(records) != set(frozen):
            raise ValueError('TRAIN sample identities do not match')
        provenance = {'split': 'train', 'sample_ids': records, 'manifest_sha256': manifest['sha256'],
                      'floor_units': args.floor_units}
        transform = family.fair_freeze_transform(floors=[args.floor], provenance=provenance)
        model.eval()
        with torch.no_grad():
            for batch in make_batches(train, positions, config, 812711, device):
                sample_id = str(batch.sample_ids[0])
                generator = torch.Generator(device=device).manual_seed(frozen[sample_id]['draw_seed'])
                pred = differentiable_reconstruction(model, _without_target(batch),
                    steps=int(config['rollout']['steps']), solver=config['rollout']['solver'],
                    generator=generator, observation_config=dict(config['observation_consistency']))
                family.fair_accumulate_calibration(pred, batch.target_fields, coordinates=batch.query_coords)
        calibration = family.fair_freeze_calibration(provenance={**provenance, 'source_checkpoint_sha256': SOURCE_SHA})
        artifact = family.state_artifact()
        import hashlib
        artifact['frozen_training_context'] = {
            'dataset_fingerprint': dataset_fingerprint(train.path),
            'query_indices_sha256': hashlib.sha256(fixed_query_indices(40300, 4096, seed=100045).numpy().tobytes()).hexdigest(),
            'normalizer_digest': train.normalizer.digest(),
            'field_names': list(train.data_spec.field_names), 'split': 'train'}
        artifact['round7_calibration_receipt'] = {'transform': transform, 'calibration': calibration,
            'source_forwards': 800, 'reference_visits': 800, 'manifest_sha256': manifest['sha256'],
            'calibration_endpoint_filtrations': 2 * 800 * len(family.spatial_objective.units),
            'calibration_dimension_diagrams': 2 * 800 * len(family.spatial_objective.units) * len(family.spatial_objective.dimensions),
            'elapsed_seconds': time.time() - started, 'floor_units': args.floor_units}
        check = build_enabled_families(setting, train.data_spec, train.normalizer)['topology'].to(device)
        check.load_state_artifact(artifact)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name('.' + output.name + '.tmp')
        torch.save(artifact, temporary)
        os.replace(temporary, output)
    finally:
        train.close()


if __name__ == '__main__':
    main()
