#!/usr/bin/env python3
"""Bounded Round 4 native students; requires an admitted teacher and campaign gate."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time
import uuid

CODE = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('CAR_REPO_ROOT', CODE)).resolve()
sys.path[:0] = [str(CODE / 'src'), str(CODE), str(CODE / 'scripts/posttraining')]
from prepare_car_round4 import device_identity, resolve_config, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--round3', type=Path, default=ROOT/'cases/turbulent_combustion/runs/coherence_adaptive_repair/round_3')
    p.add_argument('--path', choices=['FFM_SSS_Target_Amortization','FFM_Native_Supervised_Continuation'], required=True)
    p.add_argument('--attempt', required=True, help='Immutable descriptive attempt name; reuse it only for resume')
    p.add_argument('--epochs', type=int, choices=[100, 500], required=True)
    p.add_argument('--max-seconds', type=int, required=True)
    p.add_argument('--epoch-reserve-seconds', type=int, required=True,
                   help='Measured conservative wall-time reserve for an epoch and persistence')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{1,127}', args.attempt):
        p.error('attempt must be a descriptive path component containing letters, numbers, underscores or hyphens')
    if args.max_seconds <= args.epoch_reserve_seconds or args.epoch_reserve_seconds < 30:
        p.error('Provide a positive bounded lease larger than the conservative epoch reserve')
    started = time.time()
    cpu_started = time.process_time()
    device = device_identity()
    import torch
    from phycoflow_reconstruction.data.manifest import SensorManifest
    from phycoflow_reconstruction.evaluation.adaptive_campaign import atomic_json, locked_ledger, spent
    from phycoflow_reconstruction.training.native_amortization import run_native_amortization
    from phycoflow_reconstruction.training.scalable_fitting import _digest
    from phycoflow_reconstruction.training.source import load_source_model, source_checkpoint_path
    from phycoflow_reconstruction.utils.reproducibility import seed_everything

    campaign = args.campaign.resolve()
    run_dir = campaign/'runs'/'formal'/args.path/args.attempt
    if not args.resume and run_dir.exists():
        raise FileExistsError('A fresh attempt must use a new immutable attempt name')
    if args.resume and not (run_dir/'checkpoint.pt').is_file():
        raise FileNotFoundError('Resume requires the existing attempt checkpoint')
    manifest_path = campaign/'configs/train_manifest.json'
    sensor_path = campaign/'configs/train_sensor_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = sha(manifest_path)
    sensor = SensorManifest.load(sensor_path)
    field_index = manifest['field_order'].index('T')
    expected_observations = {item['sample_id']: [[point, field_index] for point in item['observation_indices']]
                             for item in manifest['items']}
    if sensor.indices != expected_observations or sensor.dataset_fingerprint != manifest['dataset_fingerprint']:
        raise ValueError('Student sensor manifest differs from SOURCE and teacher conditioning')
    cfg, binding = resolve_config(args.round3)
    seed = int(cfg['runtime'].get('seed', 42))
    deterministic = bool(cfg['runtime'].get('deterministic', True))
    strict_algorithms = bool(cfg.get('coherence', {}).get('scalable_anchored_coherence', {}).get('common_prefit'))
    seed_everything(seed, deterministic, strict_algorithms=strict_algorithms)
    if binding != manifest['config'] or sha(source_checkpoint_path(cfg)) != manifest['source_checkpoint_sha256']:
        raise ValueError('Inherited SOURCE or configuration identity changed')
    gate = json.loads((campaign/'operations/feasibility_gate.json').read_text())
    if (gate.get('admitted') is not True or gate.get('manifest_sha256') != manifest_hash
            or not 0 < gate.get('candidate_inclusive_gpu_seconds', 0) <= 35317
            or not 0 < gate.get('whole_round_gpu_seconds', 0) <= 86400):
        raise ValueError('Manifest-bound inclusive-cost feasibility gate required')
    teacher_gate = json.loads((campaign/'operations/teacher_qualification.json').read_text())
    teacher_file = (campaign / teacher_gate['checkpoint']).resolve()
    if not teacher_file.is_relative_to(campaign):
        raise ValueError('Teacher checkpoint must belong to this campaign')
    teacher_hash = sha(teacher_file)
    if (teacher_gate.get('admitted') is not True or teacher_gate.get('train_manifest_sha256') != manifest_hash
            or teacher_gate.get('teacher_checkpoint_sha256') != teacher_hash or teacher_gate.get('completed_epochs') != 100):
        raise ValueError('Qualified epoch-100 teacher checkpoint and manifest binding required for both students')
    if args.epochs == 500:
        continuation = json.loads((campaign/'operations/student_continuation.json').read_text())
        review_checkpoint = run_dir/'checkpoints/epoch_100.pt'
        expected_review = {'path': str(review_checkpoint.relative_to(campaign)), 'sha256': sha(review_checkpoint)}
        if (not args.resume or continuation.get('admitted') is not True
                or continuation.get('train_manifest_sha256') != manifest_hash
                or continuation.get('teacher_checkpoint_sha256') != teacher_hash
                or continuation.get('epoch_100_checkpoints', {}).get(args.path) != expected_review):
            raise ValueError('Epoch-500 continuation requires a bound epoch-100 review decision')
    token = uuid.uuid4().hex
    ledger = campaign/'operations/gpu_ledger.json'
    with locked_ledger(ledger) as state:
        if any('max_seconds' not in entry for entry in state['active'].values()):
            raise ValueError('An active GPU lease lacks a known reservation; cannot safely budget concurrent work')
        reserved = sum(max(0., float(x.get('max_seconds', 0)) - (started - x['start_unix'])) for x in state['active'].values())
        if (spent(state) + reserved + args.max_seconds > 86400 or
                any(x['gpu_uuid'] == device['uuid'] for x in state['active'].values())):
            raise ValueError('GPU ownership or reserved whole-round budget unavailable')
        state['active'][token] = {'gpu_uuid':device['uuid'], 'phase':args.path, 'start_unix':started,
                                  'pid':os.getpid(), 'max_seconds':args.max_seconds}
    receipt_path = campaign/'operations/native_jobs'/f'{token}.json'
    receipt = {'schema':'car_round4_native_job_v1', 'status':'running', 'path':args.path,
               'attempt':args.attempt, 'run_dir':str(run_dir),
               'start_unix':started, 'token':token, 'device':device, 'manifest_sha256':manifest_hash,
               'teacher_checkpoint_sha256':teacher_hash, 'max_seconds':args.max_seconds,
               'epoch_reserve_seconds':args.epoch_reserve_seconds, 'script_sha256':sha(__file__)}
    atomic_json(receipt_path, receipt)
    old_alarm = signal.getsignal(signal.SIGALRM)
    def alarm(_sig, _frame): raise TimeoutError('Native job reached its bounded GPU lease')
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(max(1, args.max_seconds - 5))
    data = None
    try:
        torch.set_num_threads(4)
        model, data, metadata = load_source_model(cfg, torch.device('cuda:0'))
        source_digest = _digest(model.state_dict())
        if len(data) != manifest['population'] or data.normalizer.digest() != manifest['normalizer_sha256']:
            raise ValueError('TRAIN membership or normalization changed')
        items = manifest['items']
        ids = [item['sample_id'] for item in items]
        id_index = {sid:i for i,sid in enumerate(ids)}
        membership = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
        teacher = torch.load(teacher_file, map_location='cpu', weights_only=True)
        if (teacher.get('schema') != 'car_round4_teacher_checkpoint_v1'
                or teacher.get('train_manifest_sha256') != manifest_hash or teacher.get('completed_epochs') != 100
                or teacher.get('source_parameter_buffer_sha256') != source_digest):
            raise ValueError('Teacher checkpoint identity or completed horizon mismatch')
        corrections = teacher['corrections']
        support = torch.as_tensor(manifest['query_indices'], dtype=torch.long)
        if (corrections.dtype != torch.float64 or teacher.get('query_indices') != manifest['query_indices']
                or corrections.shape != (len(ids), 4, len(support), len(manifest['field_order']))):
            raise ValueError('Teacher correction shape does not match frozen membership and support')
        if not torch.isfinite(corrections).all(): raise ValueError('Nonfinite teacher correction')
        support_lookup = torch.full((int(manifest['native_points']),), -1, dtype=torch.long)
        support_lookup[support] = torch.arange(len(support))
        bank = campaign/'artifacts/source_bank'
        def reader(sample_ids, atoms, queries):
            outputs = []
            for sid, atom, query in zip(sample_ids, atoms, queries):
                idx = id_index[sid]
                row_path = bank/f'{idx:04d}.pt'
                if row_path.with_suffix('.sha256').read_text().strip() != sha(row_path):
                    raise ValueError('SOURCE bank row checksum mismatch')
                row = torch.load(row_path, map_location='cpu', weights_only=True)
                if (row['manifest_sha256'] != manifest_hash or row['sample_id'] != sid
                        or row['source_parameter_buffer_sha256'] != source_digest
                        or not torch.equal(row['normalizer_offset'], data.normalizer.offset)
                        or not torch.equal(row['normalizer_scale'], data.normalizer.scale)):
                    raise ValueError('SOURCE row identity mismatch')
                values = row['members_model'][int(atom), 0, query].to(torch.float64)
                locations = support_lookup[query]
                included = locations >= 0
                values[included] += corrections[idx, int(atom), locations[included]]
                outputs.append(values * row['normalizer_scale'].double() + row['normalizer_offset'].double())
            return torch.stack(outputs)
        kind = 'teacher' if args.path == 'FFM_SSS_Target_Amortization' else 'truth'
        identity = {'target_kind':kind, 'source_sha256':manifest['source_checkpoint_sha256'],
                    'membership_sha256':membership, 'manifest_sha256':sensor.digest()}
        if kind == 'teacher': identity['bank_sha256'] = teacher_hash
        native_config = {'seed':seed, 'epochs':args.epochs, 'deterministic':deterministic,
                         'strict_algorithms':strict_algorithms,
                         'device':'cuda:0', 'query_points':int(cfg['model']['query_points']),
                         'temperature_field_index':manifest['field_order'].index('T'), 'identity':identity}
        import yaml
        resolved_path = run_dir/'inherited_inference_config.yaml'
        if args.resume:
            if yaml.safe_load(resolved_path.read_text()) != cfg:
                raise ValueError('Saved inherited inference configuration changed')
        else:
            run_dir.mkdir(parents=True, exist_ok=False)
            resolved_path.write_text(yaml.safe_dump(cfg, sort_keys=True))
        if time.time() - started + args.epoch_reserve_seconds >= args.max_seconds:
            raise TimeoutError('Insufficient reserved time to begin a complete reporting epoch')
        def stop(row): return time.time() - started + args.epoch_reserve_seconds >= args.max_seconds
        result = run_native_amortization(model=model, dataset=data, manifest=sensor, config=native_config,
                    run_dir=run_dir, target_reader=reader if kind == 'teacher' else None,
                    resume=args.resume, stop_after_epoch=stop, inference_config=cfg)
        saved = torch.load(result, map_location='cpu', weights_only=False)
        receipt.update(status='completed' if saved['epoch'] == args.epochs else 'paused',
                       completed_epochs=saved['epoch'], checkpoint=str(result), source_metadata=metadata,
                       source_parameter_buffer_sha256=source_digest)
    except BaseException as exc:
        receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_alarm)
        if torch.cuda.is_initialized():
            try: torch.cuda.synchronize()
            except Exception as exc: receipt.update(status='failed', teardown_error=str(exc))
        if data is not None: data.close()
        receipt['end_unix'] = time.time(); receipt['wall_seconds'] = receipt['end_unix'] - started
        receipt['process_cpu_seconds'] = time.process_time() - cpu_started
        with locked_ledger(ledger) as state:
            entry = state['active'].pop(token)
            state['closed'].append({**entry, 'token':token, 'end_unix':receipt['end_unix'], 'seconds':receipt['wall_seconds']})
        atomic_json(receipt_path, receipt)
        print(receipt_path, flush=True)

if __name__ == '__main__': main()
