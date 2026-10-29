"""Freeze checkpoint identities, then extract native predictions on the saved TEST cohort.

Run `python inference.py snapshot` before any extraction, then `python inference.py
extract --model NAME --device cuda:0`. Set CUDA_VISIBLE_DEVICES before launching.
Each model's checkpoint bytes are verified against the snapshot before TEST access.
Per-frame files provide restartability; completed endpoint arrays need only NumPy.
"""
from pathlib import Path
import argparse
from dataclasses import replace
import hashlib
import io
import json
import math
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
STUDY = OUT.parent
ROOT = HERE.parents[4]
RUNS = ROOT / 'cases/turbulent_combustion/runs'
REGISTRY = HERE / 'registry.json'
SNAPSHOT = OUT / 'metadata/checkpoints.json'
OLD_MANIFEST = STUDY / 'metadata/study_manifest.json'


def sha_bytes(content):
    return hashlib.sha256(content).hexdigest()


def stable_read(path):
    with path.open('rb') as f:
        before = os.fstat(f.fileno())
        content = f.read()
        after = os.fstat(f.fileno())
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'Checkpoint changed during read: {path}')
    if len(content) != before.st_size:
        raise RuntimeError(f'Short read: {path}')
    return content, before


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    tmp.replace(path)


def snapshot():
    if SNAPSHOT.exists():
        raise SystemExit(f'{SNAPSHOT} exists; preserve it and use a new study directory for a new snapshot')
    records = {}
    for name, relative in json.loads(REGISTRY.read_text()).items():
        run = RUNS / relative
        path = run / 'checkpoints/last.pt'
        content, stat = stable_read(path)
        records[name] = {'run': str(run), 'checkpoint': str(path), 'sha256': sha_bytes(content),
                         'file_size': stat.st_size, 'file_mtime_ns': stat.st_mtime_ns,
                         'resolved_config_sha256': sha_bytes((run / 'resolved_config.yaml').read_bytes()),
                         'frozen_utc': datetime.now(timezone.utc).isoformat()}
        print('FROZEN', name, records[name]['sha256'], flush=True)
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    write_json(SNAPSHOT, {'checkpoints': records, 'registry_sha256': sha_bytes(REGISTRY.read_bytes())})


def extend_snapshot():
    """Add registry entries without altering frozen identities of existing models."""
    state = json.loads(SNAPSHOT.read_text())
    registry = json.loads(REGISTRY.read_text())
    for name, record in state['checkpoints'].items():
        if name not in registry or (RUNS / registry[name]).resolve() != Path(record['run']).resolve():
            raise RuntimeError(f'Existing frozen registry entry changed: {name}')
    new = set(registry) - set(state['checkpoints'])
    if not new:
        raise RuntimeError('No new registry entries to freeze')
    for name in sorted(new):
        run = RUNS / registry[name]; path = run / 'checkpoints/last.pt'
        content, stat = stable_read(path)
        state['checkpoints'][name] = {'run':str(run), 'checkpoint':str(path), 'sha256':sha_bytes(content),
            'file_size':stat.st_size, 'file_mtime_ns':stat.st_mtime_ns,
            'resolved_config_sha256':sha_bytes((run/'resolved_config.yaml').read_bytes()),
            'frozen_utc':datetime.now(timezone.utc).isoformat()}
        print('FROZEN',name,state['checkpoints'][name]['sha256'],flush=True)
    state['registry_sha256'] = sha_bytes(REGISTRY.read_bytes())
    write_json(SNAPSHOT,state)


def extract(name, device):
    if not SNAPSHOT.exists():
        raise SystemExit('Run snapshot before accessing TEST data')
    snapshot_data = json.loads(SNAPSHOT.read_text())
    if snapshot_data['registry_sha256'] != sha_bytes(REGISTRY.read_bytes()):
        raise RuntimeError('Registry changed since snapshot')
    record = snapshot_data['checkpoints'][name]
    content, _ = stable_read(Path(record['checkpoint']))
    if sha_bytes(content) != record['sha256']:
        raise RuntimeError(f'{name}: last.pt changed since snapshot')
    run = Path(record['run'])
    if sha_bytes((run / 'resolved_config.yaml').read_bytes()) != record['resolved_config_sha256']:
        raise RuntimeError('Resolved config changed since snapshot')
    # KeOps inspects visibility on import. The launcher specifies the physical device.
    cache = Path(os.environ.setdefault('KEOPS_CACHE_FOLDER', '/tmp/premise_v4_keops'))
    visibility = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    (cache / (platform.system()+'_'+platform.node()+'_'+platform.release()+'_p'+platform.python_version()+'_CUDA_VISIBLE_DEVICES_'+visibility)).mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / 'src'))
    from phycoflow_reconstruction.config import load_config
    from phycoflow_reconstruction.data.factory import open_field_dataset
    from phycoflow_reconstruction.data.normalization import FieldNormalizer
    from phycoflow_reconstruction.data.manifest import dataset_fingerprint
    from phycoflow_reconstruction.data.sensor_protocols import SensorProtocol, build_observation_batch
    from phycoflow_reconstruction.models import build_model
    from phycoflow_reconstruction.training.model_lifecycle import load_training_aux_state, selected_evaluation_weight_context
    from phycoflow_reconstruction.training.run_store import load_project_checkpoint, load_model_state_strict
    torch.set_num_threads(4)
    payload = load_project_checkpoint(io.BytesIO(content))
    del content
    config = load_config(run / 'resolved_config.yaml')
    prior = json.loads(OLD_MANIFEST.read_text())
    old = prior['checkpoints'].get(name)
    expected = json.loads((STUDY / 'metadata/inference_checks.json').read_text())['models'].get(name)
    out = OUT / 'data' / f'{name}_test100.npz'
    meta = OUT / 'metadata' / f'{name}.json'
    OUT.joinpath('data').mkdir(exist_ok=True)
    if out.exists() or meta.exists():
        if not out.exists() or not meta.exists():
            raise RuntimeError('Incomplete final artifact pair; inspect before resuming')
        m = json.loads(meta.read_text())
        with np.load(out) as f: a = f['fields']
        if m['checkpoint']['sha256'] != record['sha256'] or sha_bytes(np.ascontiguousarray(a).tobytes()) != m['prediction_sha256']:
            raise RuntimeError('Completed output identity mismatch')
        print('COMPLETE', name, flush=True)
        return
    if old and expected and old['sha256'] == record['sha256'] and old['resolved_config_sha256'] == record['resolved_config_sha256']:
        source = STUDY / 'data' / f'{name}_test100.npz'
        with np.load(source) as f: a = f['fields']
        digest = sha_bytes(np.ascontiguousarray(a).tobytes())
        if digest != expected['prediction_sha256'] or a.shape != (100,100,403,5) or not np.isfinite(a).all():
            raise RuntimeError('Prior endpoint identity or shape mismatch')
        out.symlink_to(os.path.relpath(source, out.parent))
        write_json(meta, {'checkpoint':record, 'prediction_sha256':digest, 'source':'verified previous study',
                          'epoch':old['epoch'], 'global_step':old['global_step'],
                          'evaluation_weights':old['evaluation_weights'],
                          'native_model_config':config['model'], 'native_evaluation_config':config['evaluation'],
                          'prior_protocol_manifest':str(OLD_MANIFEST), 'checks':expected})
        print('REUSED', name, flush=True)
        return
    train = open_field_dataset(config['dataset'], split='train')
    train_count = len(train)
    train.close()
    dataset = open_field_dataset(config['dataset'], split='test')
    ids = np.asarray(prior['local_indices'])
    assert len(dataset) == 1000 and tuple(dataset.grid_shape) == (100,403)
    assert dataset_fingerprint(dataset.path) == prior['dataset_fingerprint']
    assert dataset.normalizer.digest() == prior['normalization_digest']
    assert FieldNormalizer(**payload['normalization']).digest() == dataset.normalizer.digest()
    model = build_model(config['model'], dataset.data_spec).to(device)
    load_model_state_strict(model, payload['model'])
    load_training_aux_state(model, payload)
    step = payload.get('global_step')
    steps_per_epoch = math.ceil(train_count / int(config['optimization']['batch_size']))
    epoch = step / steps_per_epoch
    run_manifest = json.loads((run / 'run_manifest.json').read_text())
    if run_manifest.get('last_checkpoint_step') == step and run_manifest.get('last_checkpoint_epoch') is not None:
        assert math.isclose(epoch, float(run_manifest['last_checkpoint_epoch']))
    del payload
    model.eval()
    partial = OUT / 'data' / ('.partial_' + name)
    partial.mkdir(exist_ok=True)
    identity = {'checkpoint_sha256':record['sha256'], 'config_sha256':record['resolved_config_sha256'],
                'protocol_manifest_sha256':sha_bytes(OLD_MANIFEST.read_bytes()),
                'device':device, 'cuda_visible_devices':visibility}
    identity_file = partial / 'identity.json'
    if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
        raise RuntimeError('Partial cache belongs to a different checkpoint or protocol')
    write_json(identity_file, identity)
    gt_path = STUDY / 'data/gt_test100.npz'
    with np.load(gt_path) as f:
        saved_indices = f['obs_indices'].copy()
        saved_values = f['obs_values'].copy()
        gt_hash = sha_bytes(np.ascontiguousarray(f['fields']).tobytes())
    assert gt_hash == prior['normalized_gt_sha256']
    started = time.monotonic()
    steps = config.get('evaluation', {}).get('generation_steps', 2)
    with selected_evaluation_weight_context(model, 'configured'), torch.no_grad():
        for j, index in enumerate(ids):
            file = partial / f'{j:03d}.npy'
            if file.exists():
                pred = np.load(file)
                if pred.shape != (100,403,5) or not np.isfinite(pred).all(): raise RuntimeError(f'Invalid partial {file}')
                continue
            sample = dataset[int(index)]
            assert sample.time_index == prior['global_indices'][j]
            batch = build_observation_batch([sample], SensorProtocol(name='random_uniform', field_counts={'T':256}, seed=42+int(index)), query_points=None)
            assert np.array_equal(batch.obs_indices[0].numpy(), saved_indices[j])
            assert np.array_equal(batch.obs_values[0,:,0].numpy(), saved_values[j])
            assert torch.equal(batch.metadata['query_indices'][0], torch.arange(40300))
            assert (batch.obs_field_ids == 2).all()
            batch = replace(batch, target_fields=None)
            result = model.reconstruct(batch.to(device), steps=steps, generator=torch.Generator(device=device).manual_seed(2027+int(index)))
            pred = result.prediction[0].cpu().numpy().reshape(100,403,5).astype(np.float32)
            if not np.isfinite(pred).all(): raise RuntimeError(f'Nonfinite output at {j}')
            tmp = file.with_suffix('.tmp')
            with tmp.open('wb') as f: np.save(f, pred)
            tmp.replace(file)
            print(name, j+1, '/100', round(time.monotonic()-started,1), 's', flush=True)
    a = np.stack([np.load(partial / f'{j:03d}.npy') for j in range(len(ids))])
    tmp = out.with_suffix('.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f, fields=a)
    tmp.replace(out)
    write_json(meta, {'checkpoint':record, 'epoch':epoch, 'global_step':step, 'prediction_sha256':sha_bytes(np.ascontiguousarray(a).tobytes()),
                      'epoch_basis':f'global_step / ceil({train_count} training frames / {config["optimization"]["batch_size"]} batch size)',
                      'native_model_config':config['model'], 'native_evaluation_config':config['evaluation'],
                      'evaluation_weights':'EMA' if config['model'].get('model_ema_eval') else 'live',
                      'source':'new extraction', 'prior_protocol_manifest':str(OLD_MANIFEST), 'device':device,
                      'cuda_visible_devices':visibility, 'inference_seconds_this_session':time.monotonic()-started,
                      'shape':list(a.shape), 'strict_state_loaded':True, 'normalizer_match':True,
                      'observations_match':True, 'target_fields_removed':True, 'finite':True})
    shutil.rmtree(partial)
    dataset.close()
    print('SAVED', name, flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('snapshot')
    sub.add_parser('extend')
    p = sub.add_parser('extract')
    p.add_argument('--model', required=True, choices=json.loads(REGISTRY.read_text()).keys())
    p.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.action == 'snapshot': snapshot()
    elif args.action == 'extend': extend_snapshot()
    else: extract(args.model, args.device)

if __name__ == '__main__': main()
