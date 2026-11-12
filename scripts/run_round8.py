#!/usr/bin/env python3
"""Round 8 CPU preparation and explicit campaign accounting commands."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
import signal
import threading
from pathlib import Path
import sys
import time
import uuid
from contextlib import contextmanager

ROOT = Path(os.environ.get('ROUND8_REPO_ROOT', Path(__file__).resolve().parents[1]))
CODE_ROOT = Path(os.environ.get('ROUND8_CODE_ROOT', ROOT))
sys.path.insert(0, str(CODE_ROOT / 'src'))
from phycoflow_reconstruction.evaluation.round8_campaign import (  # noqa: E402
    file_hash, forecast, freeze_manifest, occupied_seconds, validate_t2000_receipt,
)

CAMPAIGN = ROOT / 'cases/turbulent_combustion/runs/coherence_compute_round8'


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def prepare(dataset, historical, output):
    import h5py
    from phycoflow_reconstruction.data.splits import resolve_split
    from phycoflow_reconstruction.data.manifest import dataset_fingerprint
    with h5py.File(dataset, 'r') as h:
        selections = {name: resolve_split(h, name) for name in ('train', 'validation', 'test')}
        trajectory_ids = ([v.decode() if isinstance(v, bytes) else str(v) for v in h['trajectory_id'][:]]
                          if 'trajectory_id' in h else
                          [f'trajectory_{i:06d}' for i in range(h['fields'].shape[0])])
        records = {}
        for name, selection in selections.items():
            records[name] = [dict(id=f'{trajectory_ids[b]}:{t}', trajectory=trajectory_ids[b],
                                  frame=t, dataset_index=i, **({'source': name} if name != 'train' else {}))
                             for i, (b, t) in enumerate((b, t) for b in selection.trajectory_indices
                                                         for t in selection.frame_indices)]
    old = json.loads(Path(historical).read_text())
    exposed = {(split, row['sample_id']) for split in ('validation', 'test')
               for row in old['splits'][split]}
    manifest = freeze_manifest(records['train'], records['validation'] + records['test'], exposed=exposed)
    manifest['exposure_provenance'] = 'known historical inventory preferred away; other prior exposure unknown'
    manifest['known_exposure_manifest_sha256'] = file_hash(historical)
    dataset_path = Path(dataset).resolve()
    stat = dataset_path.stat()
    manifest['dataset_identity'] = {'path': str(dataset_path), 'size': stat.st_size,
                                    'mtime_ns': stat.st_mtime_ns,
                                    'structural_fingerprint_sha256': dataset_fingerprint(dataset_path)}
    # Seeds are condition/member specific and deterministic; their realised
    # observations, random fields and sampled indices must still be hashed by
    # the inference adapter before any endpoint cache is accepted.
    def derived(*parts):
        token = ':'.join(map(str, (47, *parts))).encode()
        return int.from_bytes(hashlib.sha256(token).digest()[:8], 'big') % (2**63)
    for population in ('TRAIN', 'ASSESS'):
        for row in manifest[population]:
            identity = (population, row.get('source', 'train'), row['id'])
            row['observation_seed'] = derived('observation', *identity)
            row['member_seeds'] = [derived('latent', *identity, j) for j in range(2)]
            row['spatial_seed'] = 100045 if population == 'TRAIN' else None
    manifest['sampling'] = {'TRAIN': {'sites': 4096, 'design': 'fixed_uniform_without_replacement', 'seed': 100045},
                            'ASSESS': {'sites': 40300, 'design': 'native_physical_mask'},
                            'members': 2}
    pinned = {
        'source_checkpoint': ROOT / 'cases/turbulent_combustion/runs/tc_gl_rbf_cq_coverage_5000ep/20261029T210910Z_cf9c2276/checkpoints/last.pt',
        'base_config': ROOT / 'cases/turbulent_combustion/runs/coherence_compute_round6/configs/U6.yaml',
        'normalizer': ROOT / 'cases/turbulent_combustion/configs/normalization/train_mean_std.json',
        'historical_positions': ROOT / 'cases/turbulent_combustion/runs/coherence_compute_round6/artifacts/positions.json',
    }
    manifest['inputs'] = {k: {'path': str(v), 'sha256': file_hash(v)} for k, v in pinned.items()}
    manifest['split_strategy'] = {k: v.strategy for k, v in selections.items()}
    write_new(output, manifest)
    return {'manifest': str(output), 'manifest_sha256': file_hash(output),
            'train_count': len(manifest['TRAIN']), 'assess_count': len(manifest['ASSESS'])}


@contextmanager
def ledger_transaction(directory=CAMPAIGN / 'artifacts'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = directory / 'budget_ledger.json'
        ledger = json.loads(path.read_text()) if path.exists() else {
            'schema': 'round8_budget_v1', 'qualification_attempts': 0, 'formal_attempts': 0, 'intervals': []}
        yield ledger
        tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
        tmp.write_text(json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False) + '\n')
        os.replace(tmp, path)


def charge_attempt(phase, directory=CAMPAIGN / 'artifacts', *, arm, step):
    with ledger_transaction(directory) as ledger:
        spent = occupied_seconds(ledger['intervals'], time.time())
        if spent >= (28 if phase == 'formal' else 32) * 3600:
            raise RuntimeError('GPU budget or formal closeout reserve exhausted')
        if phase == 'qualification':
            if ledger['qualification_attempts'] >= 100:
                raise RuntimeError('qualification-attempt cap reached')
            ledger['qualification_attempts'] += 1
        elif phase == 'formal':
            if arm not in ('warmup', 'control', 'candidate'):
                raise ValueError('formal attempts require a declared arm')
            expected_start = 0 if arm == 'warmup' else 130
            positions = ledger.setdefault('formal_last_position', {})
            last = positions.get(arm, expected_start - 1)
            if step > last + 1 or step < expected_start:
                raise ValueError('formal optimizer positions must follow the shared fork contiguously')
            replay = step <= last
            if replay:
                ledger['formal_replays'] = ledger.get('formal_replays', 0) + 1
            else:
                if ledger.get('formal_unique', 0) >= 25870:
                    raise RuntimeError('fixed 1000-epoch matched position count exhausted')
                ledger['formal_unique'] = ledger.get('formal_unique', 0) + 1
                positions[arm] = int(step)
            ledger['formal_attempts'] += 1
        else:
            raise ValueError('unknown optimizer attempt phase')
        with (Path(directory) / 'attempts.jsonl').open('a') as stream:
            stream.write(json.dumps({'phase':phase, 'arm':arm, 'step':int(step),
                'pid':os.getpid(), 'lease_token':os.environ.get('ROUND8_LEASE_TOKEN'),
                'time':time.time(), 'replay':replay if phase == 'formal' else False}) + '\n')


@contextmanager
def occupied(stage, device_uuid, *, formal=False, directory=CAMPAIGN / 'artifacts', arm=None):
    """Charge GPU-seconds additively and request a clean stop at the cap.

    The launcher must bind CUDA_VISIBLE_DEVICES to the verified physical UUID.
    The trainer must save on interruption; watchdog headroom is not extra budget.
    """
    if not device_uuid or os.environ.get('CUDA_VISIBLE_DEVICES') != device_uuid:
        raise ValueError('physical GPU UUID must match CUDA_VISIBLE_DEVICES')
    if not os.environ.get('KEOPS_CACHE_FOLDER'):
        raise ValueError('device-specific KeOps cache required')
    inherited_token = os.environ.get('ROUND8_LEASE_TOKEN')
    token = inherited_token or uuid.uuid4().hex
    limit = (28 if formal else 32) * 3600
    with ledger_transaction(directory) as ledger:
        now = time.time()
        if occupied_seconds(ledger['intervals'], now) >= limit - 60:
            raise RuntimeError('GPU budget or closeout reserve unavailable')
        if inherited_token:
            lease = next((row for row in ledger['intervals'] if row['id'] == inherited_token), None)
            if (lease is None or lease['device_uuid'] != device_uuid
                    or lease['pid'] != os.getpid() or 'end' in lease):
                raise RuntimeError('inherited GPU lease differs from this process')
        else:
            if any(row['device_uuid'] == device_uuid and 'end' not in row for row in ledger['intervals']):
                raise RuntimeError('GPU already reserved in campaign ledger')
            lease = {'id': token, 'stage': stage, 'device_uuid': device_uuid,
                     'start': now, 'pid': os.getpid()}
            ledger['intervals'].append(lease)
        if arm is not None:
            if lease.get('arm') not in (None, arm):
                raise ValueError('GPU lease arm differs from its declared training arm')
            lease['arm'] = arm
    stop = threading.Event()
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        raise RuntimeError('Round 8 GPU budget watchdog requested clean stop')
    signal.signal(signal.SIGTERM, terminate)
    def guard():
        while not stop.wait(5):
            try:
                with ledger_transaction(directory) as ledger:
                    now = time.time()
                    active = sum('end' not in row for row in ledger['intervals'])
                    left = limit - occupied_seconds(ledger['intervals'], now)
                if left <= 60 * max(active, 1):
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
            except Exception:
                os.kill(os.getpid(), signal.SIGTERM)
                return
    thread = threading.Thread(target=guard, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        signal.signal(signal.SIGTERM, previous)
        with ledger_transaction(directory) as ledger:
            next(row for row in ledger['intervals'] if row['id'] == token)['end'] = time.time()



def release_inherited_lease(directory=CAMPAIGN / 'artifacts'):
    """Close this process's launcher lease on errors before occupied() starts."""
    token = os.environ.get('ROUND8_LEASE_TOKEN')
    if not token or not (Path(directory) / 'budget_ledger.json').is_file():
        return
    with ledger_transaction(directory) as ledger:
        for row in ledger['intervals']:
            if row.get('id') == token and row.get('pid') == os.getpid() and 'end' not in row:
                row['end'] = time.time()
                break

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--dataset', required=True)
    p.add_argument('--historical-manifest', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('validate-t2000')
    p.add_argument('receipt')
    p = sub.add_parser('infer')
    p.add_argument('--config', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--population', choices=('TRAIN', 'ASSESS'), required=True)
    p.add_argument('--endpoint', choices=('SOURCE', 'T2000', 'B8', 'C8'), default='SOURCE')
    p.add_argument('--checkpoint')
    p.add_argument('--t-receipt')
    p.add_argument('--device', default='cuda:0')
    p = sub.add_parser('score')
    p.add_argument('--config', required=True)
    p.add_argument('--source-panel', required=True)
    p.add_argument('--endpoint-panel', required=True)
    p.add_argument('--fair-shape-artifact', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--alpha', type=float, required=True)
    p.add_argument('--assessment-policy')
    p.add_argument('--native-bundle')
    p.add_argument('--interim-diagnostic', action='store_true')
    p.add_argument('--reference-only', action='store_true')
    p.add_argument('--device', default='cpu')
    p = sub.add_parser('forecast')
    p.add_argument('measurements', help='JSON of measured remaining costs in seconds and cache bytes')
    p = sub.add_parser('fit')
    p.add_argument('--config', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--source-panel', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('qualify')
    p.add_argument('--config', required=True)
    p.add_argument('--artifacts', required=True)
    p.add_argument('--manifest', required=True)
    p = sub.add_parser('train')
    p.add_argument('--config', required=True)
    p.add_argument('--artifacts', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--native-bundle', required=True)
    p.add_argument('--fair-shape-artifact', required=True)
    p.add_argument('--arm', choices=('warmup','B8','C8'), required=True)
    p.add_argument('--resume')
    p = sub.add_parser('fork')
    p.add_argument('--config', required=True)
    p.add_argument('--warmup', required=True)
    p = sub.add_parser('stage0')
    p.add_argument('--config', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--t-receipt', required=True)
    p.add_argument('--fair-shape-artifact', required=True)
    p.add_argument('--native-bundle', required=True)
    p = sub.add_parser('profile-training')
    p.add_argument('--qualification-receipt', required=True)
    p.add_argument('--artifacts', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('profile-observers')
    p.add_argument('--config', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--train-panel', required=True)
    p.add_argument('--assess-panel', required=True)
    p.add_argument('--fair-shape-artifact', required=True)
    p.add_argument('--native-bundle', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p = sub.add_parser('benchmark')
    p.add_argument('--config', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--endpoints', required=True, help='JSON fixed endpoint checkpoint and alpha mapping')
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.stage == 'prepare':
        result = prepare(args.dataset, args.historical_manifest, args.output)
    elif args.stage == 'validate-t2000':
        result = validate_t2000_receipt(args.receipt)
    elif args.stage in ('infer', 'score'):
        import yaml
        from contextlib import nullcontext
        from phycoflow_reconstruction.evaluation.round8_inference import infer_panel, score_cached_panel
        config = yaml.safe_load(Path(args.config).read_text())
        lease = (occupied(args.stage, os.environ.get('ROUND8_PHYSICAL_GPU_UUID'))
                 if args.device.startswith('cuda') else nullcontext())
        with lease:
            if args.stage == 'infer':
                result = {'panel': str(infer_panel(config, args.manifest, args.output,
                    population=args.population, endpoint=args.endpoint, checkpoint=args.checkpoint,
                    t_receipt=args.t_receipt, device=args.device))}
            else:
                result = {'report': str(score_cached_panel(config, args.source_panel, args.endpoint_panel,
                    fair_shape_artifact=args.fair_shape_artifact, output_path=args.output,
                    alpha=args.alpha, device=args.device, assessment_policy=args.assessment_policy,
                    interim_diagnostic=args.interim_diagnostic, native_bundle=args.native_bundle,
                    reference_only=args.reference_only))}
    elif args.stage == 'forecast':
        result = forecast(**json.loads(Path(args.measurements).read_text()))
    elif args.stage == 'fit':
        import torch
        from phycoflow_reconstruction.training.round8_artifacts import prepare_from_panel
        result = prepare_from_panel(args.config, args.manifest, args.source_panel)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            torch.save(result, stream)
        result = {'artifact': str(path), 'sha256': file_hash(path)}
    elif args.stage == 'profile-training':
        from phycoflow_reconstruction.training.round8_compute_profile import profile_training
        with occupied(args.stage, os.environ.get('ROUND8_PHYSICAL_GPU_UUID')):
            with ledger_transaction() as ledger:
                if ledger['qualification_attempts'] > 96:
                    raise RuntimeError('four qualification attempts are not available for training profiling')
            result = {'profile':str(profile_training(
                qualification_receipt=args.qualification_receipt, artifacts_path=args.artifacts,
                output_path=args.output, campaign=CAMPAIGN,
                case_dir=ROOT / 'cases/turbulent_combustion', charge_attempt=charge_attempt))}
    elif args.stage == 'profile-observers':
        import yaml
        from contextlib import nullcontext
        from phycoflow_reconstruction.evaluation.round8_observer_profile import profile_observers
        config = yaml.safe_load(Path(args.config).read_text())
        lease = occupied(args.stage, os.environ.get('ROUND8_PHYSICAL_GPU_UUID')) if args.device.startswith('cuda') else nullcontext()
        with lease:
            result = {'profile':str(profile_observers(config,train_panel=args.train_panel,
                assess_panel=args.assess_panel,manifest_path=args.manifest,
                fair_shape_artifact=args.fair_shape_artifact,native_bundle=args.native_bundle,
                output_path=args.output,device=args.device))}
    elif args.stage in ('stage0', 'benchmark'):
        import yaml
        config = yaml.safe_load(Path(args.config).read_text())
        with occupied(args.stage, os.environ.get('ROUND8_PHYSICAL_GPU_UUID')):
            if args.stage == 'stage0':
                from phycoflow_reconstruction.evaluation.round8_stage0 import execute
                result = {'receipt':str(execute(
                    config, campaign=CAMPAIGN, manifest=args.manifest, t_receipt=args.t_receipt,
                    fair_shape_artifact=args.fair_shape_artifact, native_bundle=args.native_bundle))}
            else:
                from phycoflow_reconstruction.evaluation.round8_benchmark import benchmark
                result = {'benchmark':str(benchmark(
                    config, args.manifest, json.loads(Path(args.endpoints).read_text()),
                    output_path=args.output))}
    else:
        import yaml
        import torch
        from phycoflow_reconstruction.training.round8_campaign import fork_warmup, validate_formal_child
        from phycoflow_reconstruction.training.round8_execution import (
            execute_training, qualify, ScheduledObserver, require_formal_admission)
        config = yaml.safe_load(Path(args.config).read_text())
        if args.stage == 'fork':
            result = fork_warmup(args.warmup, case_dir=ROOT / 'cases/turbulent_combustion',
                                 retained_config=config)
            write_new(CAMPAIGN / 'artifacts/fork.json', result)
        else:
            artifacts = torch.load(args.artifacts, map_location='cpu', weights_only=True)
            if args.stage == 'train':
                require_formal_admission(CAMPAIGN, artifacts_path=args.artifacts, config=config, manifest_path=args.manifest)
            with occupied(args.stage, os.environ.get('ROUND8_PHYSICAL_GPU_UUID'),
                          formal=args.stage == 'train',
                          arm=args.arm if args.stage == 'train' else 'qualification'):
                if args.stage == 'qualify':
                    result = qualify(config, case_dir=ROOT / 'cases/turbulent_combustion',
                                     artifacts=artifacts, campaign=CAMPAIGN,
                                     charge_attempt=charge_attempt, manifest=args.manifest)
                else:
                    if args.arm != 'warmup' and args.resume is None:
                        raise ValueError('B8/C8 require the exact shared-warmup fork')
                    if args.arm in ('B8','C8'):
                        validate_formal_child(CAMPAIGN / 'artifacts/fork.json', args.arm, args.resume)
                    mode = {'warmup':'warmup', 'B8':'control', 'C8':'candidate'}[args.arm]
                    observer = (None if args.arm == 'warmup' else ScheduledObserver(
                        campaign=CAMPAIGN, manifest=args.manifest, arm=args.arm,
                        artifacts=artifacts, native_bundle=args.native_bundle,
                        fair_shape_artifact=args.fair_shape_artifact))
                    run = execute_training(
                        config, case_dir=ROOT / 'cases/turbulent_combustion',
                        artifacts=artifacts, phase='formal', charge_attempt=charge_attempt,
                        mode=mode, resume=args.resume, checkpoint_callback=observer,
                        max_steps=130 if args.arm == 'warmup' else None)
                    checkpoint = Path(run) / 'checkpoints/last.pt'
                    result = {'status':'completed', 'arm':args.arm, 'run':str(run),
                              'checkpoint_sha256':file_hash(checkpoint),
                              'epoch':10 if args.arm == 'warmup' else 1000}
                    write_new(CAMPAIGN / 'artifacts' / (args.arm + '_complete.json'), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    finally:
        release_inherited_lease()
