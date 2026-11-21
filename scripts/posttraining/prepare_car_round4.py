#!/usr/bin/env python3
"""Round 4 fixed TRAIN conditioning and bounded SOURCE/RF timing.

Preparation is training-target construction, not model evaluation. Device selection
is supplied by CUDA_VISIBLE_DEVICES; historical artifacts are never modified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('CAR_REPO_ROOT', CODE)).resolve()
sys.path[:0] = [str(CODE / 'src'), str(CODE)]
ARM = 'FFM_SourceScaledRisk_Uniform_Repair'
NATIVE = 40300


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(doc, f, indent=2, allow_nan=False)
        f.write('\n')


def resolve_config(round3):
    import yaml
    p = json.loads((round3 / 'configs/prepared.json').read_text())['arms'][ARM]
    path = Path(p['path'])
    if sha(path) != p['sha256']:
        raise ValueError('Round 3 inherited configuration hash mismatch')
    return yaml.safe_load(path.read_text()), p


def prepare(args):
    started, cpu_started = time.time(), time.process_time()
    import torch
    torch.set_num_threads(4)
    from phycoflow_reconstruction.data.factory import open_field_dataset
    from phycoflow_reconstruction.data.manifest import SensorManifest, dataset_fingerprint
    from phycoflow_reconstruction.data.sensor_protocols import SensorProtocol
    from phycoflow_reconstruction.data.training_batches import fixed_query_indices
    from phycoflow_reconstruction.training.source import source_checkpoint_path
    cfg, binding = resolve_config(args.round3)
    data = open_field_dataset(cfg['dataset'], split='train')
    try:
        if len(data) != 8000 or int(data.data_spec.logical_shape[0]) * int(data.data_spec.logical_shape[1]) != NATIVE:
            raise ValueError('Expected frozen TRAIN8000 and native 40300 support')
        if cfg['source']['weight_view'] != 'ema_materialized':
            raise ValueError('Expected materialized SOURCE EMA')
        query = fixed_query_indices(NATIVE, 4096, seed=int(cfg['coherence']['compute_budget']['query_seed']))
        items = []
        for i in range(len(data)):
            sample_id = f'{data.trajectory_ids[data._items[i][0]]}:{data._items[i][1]}'
            obs = torch.randperm(NATIVE, generator=torch.Generator().manual_seed(42 + i))[:256].tolist()
            items.append({'sample_id': sample_id, 'sample_index': i, 'observation_indices': obs,
                          'member_seeds': [9100003 + 11*i + j for j in range(4)]})
        if len({x['sample_id'] for x in items}) != 8000:
            raise ValueError('TRAIN sample IDs are not unique')
        doc = {'schema':'car_round4_training_manifest_v1', 'split':'train', 'population':len(data),
               'native_points':NATIVE, 'sensor_field':'T', 'sensor_count':256,
               'observation_seed_rule':'42 + TRAIN sample index', 'member_seed_rule':'9100003 + 11*TRAIN sample index + member',
               'members':4, 'query_indices':query.tolist(), 'query_seed':cfg['coherence']['compute_budget']['query_seed'],
               'field_order':list(data.field_names), 'normalizer_sha256':data.normalizer.digest(),
               'dataset_fingerprint':dataset_fingerprint(data.path), 'config':binding,
               'source_checkpoint_sha256':sha(source_checkpoint_path(cfg)), 'source_weight_view':cfg['source']['weight_view'],
               'rollout':cfg['rollout'], 'bank_units':'model field units; decode with saved normalizer',
               'observation_values':'original dataset values, reconstructed using dataset fingerprint and persisted indices',
               'preparation_timer':{'start_unix':started, 'end_unix':time.time(), 'cpu_seconds':time.process_time()-cpu_started},
               'items':items}
        write_new(args.campaign / 'configs/train_manifest.json', doc)
        sensor = SensorManifest(dataset_path=Path(data.path).name, dataset_fingerprint=doc['dataset_fingerprint'],
                                split='train', protocol=SensorProtocol(field_counts={'T':256}, seed=42).to_dict(),
                                indices={x['sample_id']:[[p, list(data.field_names).index('T')] for p in x['observation_indices']] for x in items})
        sensor_path = args.campaign / 'configs/train_sensor_manifest.json'
        if sensor_path.exists():
            raise FileExistsError(sensor_path)
        sensor.save(sensor_path)
        print(args.campaign / 'configs/train_manifest.json', flush=True)
    finally:
        data.close()


def device_identity():
    selected = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not selected or ',' in selected:
        raise ValueError('Set CUDA_VISIBLE_DEVICES to one permitted device')
    listing = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'], text=True)
    rows = [[x.strip() for x in line.split(',')] for line in listing.splitlines()]
    matches = [r for r in rows if selected in r[:2]]
    if len(matches) != 1:
        raise ValueError('Cannot uniquely resolve selected GPU')
    running = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'], text=True)
    if any(line.split(',')[0].strip() == matches[0][1]
           and int(line.split(',')[1].strip()) != os.getpid() for line in running.splitlines()):
        raise ValueError('Selected GPU has an existing compute owner')
    return dict(zip(('index','uuid','name'), matches[0]))


def execute(args):
    started = time.time()
    cpu_started = time.process_time()
    device = device_identity()
    import torch
    from phycoflow_reconstruction.data.sensor_protocols import SensorProtocol, build_observation_batch
    from phycoflow_reconstruction.evaluation.adaptive_campaign import atomic_json, locked_ledger, spent
    from phycoflow_reconstruction.training.rollout import differentiable_reconstruction
    from phycoflow_reconstruction.training.model_lifecycle import backward_and_clip_model_loss
    from phycoflow_reconstruction.training.scalable_fitting import _digest
    from phycoflow_reconstruction.training.source import load_source_model, source_checkpoint_path
    campaign = args.campaign.resolve()
    manifest_path = campaign / 'configs/train_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = sha(manifest_path)
    cfg, binding = resolve_config(args.round3)
    if binding != manifest['config'] or sha(source_checkpoint_path(cfg)) != manifest['source_checkpoint_sha256']:
        raise ValueError('Inherited source or config changed')
    if args.command == 'generate':
        gate = json.loads((campaign / 'operations/feasibility_gate.json').read_text())
        if (gate.get('admitted') is not True or gate.get('manifest_sha256') != manifest_hash
                or not 0 < gate.get('candidate_inclusive_gpu_seconds', 0) <= 35317
                or not 0 < gate.get('whole_round_gpu_seconds', 0) <= 86400):
            raise ValueError('Formal generation requires an admitted manifest-bound feasibility gate')
    token = uuid.uuid4().hex
    ledger = campaign / 'operations/gpu_ledger.json'
    with locked_ledger(ledger) as state:
        if any('max_seconds' not in entry for entry in state['active'].values()):
            raise ValueError('active lease lacks a known reservation')
        outstanding = sum(max(0., x['max_seconds'] - (started-x['start_unix']))
                          for x in state['active'].values())
        if spent(state) + outstanding + args.max_seconds > 86400 or any(x['gpu_uuid'] == device['uuid'] for x in state['active'].values()):
            raise ValueError('GPU lease or aggregate 24-hour budget unavailable')
        state['active'][token] = {'gpu_uuid':device['uuid'], 'phase':args.command,
                                'start_unix':started, 'pid':os.getpid(), 'max_seconds':args.max_seconds}
    receipt = {'schema':'car_round4_preparation_job_v1','status':'running','token':token, 'device':device,
               'start_unix':started, 'manifest_sha256':manifest_hash, 'max_seconds':args.max_seconds,
               'command':args.command, 'script_sha256':sha(__file__)}
    receipt_path = campaign / 'operations' / 'preparation_jobs' / f'{token}.json'
    atomic_json(receipt_path, receipt)
    old = signal.getsignal(signal.SIGALRM)
    def alarm(_sig, _frame):
        raise TimeoutError('Job reached its bounded GPU lease')
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(max(1, int(args.max_seconds)-10))
    data = None
    try:
        torch.set_num_threads(4)
        model, data, meta = load_source_model(cfg, torch.device('cuda:0'))
        if len(data) != manifest['population'] or data.normalizer.digest() != manifest['normalizer_sha256']:
            raise ValueError('Live TRAIN differs from manifest')
        model.eval()
        protocol = SensorProtocol(field_counts={'T':256}, seed=42)
        identity = _digest(model.state_dict())
        times = []
        output = campaign / ('artifacts/source_bank' if args.command == 'generate' else f'runs/qualification/source_timing_{token}')
        output.mkdir(parents=True, exist_ok=True)
        count = len(manifest['items']) if args.command == 'generate' else args.conditions
        for item in manifest['items'][:count]:
            index = item['sample_index']
            row_path = output / f'{index:04d}.pt'
            if row_path.exists():
                if args.command != 'generate':
                    raise ValueError('qualification row already exists')
                # A hash sidecar is the durable commit marker; unfinished rows are never reused.
                side = row_path.with_suffix('.sha256')
                if not side.is_file() or side.read_text().strip() != sha(row_path):
                    raise ValueError(f'incomplete or damaged bank row {row_path}')
                saved = torch.load(row_path, map_location='cpu', weights_only=True)
                if (saved.get('manifest_sha256') != manifest_hash or saved.get('sample_id') != item['sample_id']
                        or saved.get('source_parameter_buffer_sha256') != identity):
                    raise ValueError(f'bank row identity differs: {row_path}')
                continue
            start_row = time.perf_counter()
            sample = data[index]
            sid = f'{sample.trajectory_id}:{sample.time_index if sample.time_index is not None else "all"}'
            if sid != item['sample_id'] or sample.values.shape[0] != NATIVE:
                raise ValueError('TRAIN sample identity or support changed')
            obs = {sid:[(i, sample.field_names.index('T')) for i in item['observation_indices']]}
            batch = build_observation_batch([sample], protocol, manifest_indices=obs).to('cuda:0')
            with torch.no_grad():
                outputs = [differentiable_reconstruction(model, replace(batch,target_fields=None),
                           steps=int(cfg['rollout']['steps']), solver=cfg['rollout']['solver'],
                           generator=torch.Generator(device='cuda:0').manual_seed(seed),
                           observation_config=cfg['observation_consistency']).detach().cpu() for seed in item['member_seeds']]
            torch.cuda.synchronize()
            row = {'sample_id':sid, 'sample_index':index, 'members_model':torch.stack(outputs),
                   'truth_model':batch.target_fields.cpu(), 'coordinates_model':batch.query_coords.cpu(),
                   'coordinates_raw':sample.coordinates_raw.cpu(), 'native_masses':data.physical_weights.cpu(),
                   'obs_indices':batch.obs_indices.cpu(), 'obs_values':batch.obs_values.cpu(),
                   'obs_fields':batch.obs_field_ids.cpu(), 'member_seeds':item['member_seeds'],
                   'normalizer_offset':data.normalizer.offset.cpu(), 'normalizer_scale':data.normalizer.scale.cpu(),
                   'manifest_sha256':manifest_hash, 'source_parameter_buffer_sha256':identity}
            tmp = row_path.with_suffix('.tmp')
            torch.save(row, tmp)
            os.replace(tmp, row_path)
            row_path.with_suffix('.sha256').write_text(sha(row_path)+'\n')
            times.append(time.perf_counter()-start_row)
        receipt.update(status='completed', rows_generated=len(times), row_end_to_end_seconds=times,
                       source_metadata=meta, source_parameter_buffer_sha256=identity, output=str(output))
        if args.command == 'generate':
            rows = [output / f'{i:04d}.pt' for i in range(len(manifest['items']))]
            bank_receipt = {'schema':'car_round4_source_bank_v1', 'status':'completed', 'split':'train',
                            'population':len(rows), 'members':4, 'native_points':NATIVE,
                            'units':'model field units', 'manifest_sha256':manifest_hash,
                            'source_checkpoint_sha256':manifest['source_checkpoint_sha256'],
                            'source_parameter_buffer_sha256':identity,
                            'rows_sha256':[p.with_suffix('.sha256').read_text().strip() for p in rows]}
            path = output / 'receipt.json'
            if path.exists():
                if json.loads(path.read_text()) != bank_receipt:
                    raise ValueError('existing completed SOURCE bank receipt differs')
            else:
                write_new(path, bank_receipt)
        if args.command == 'qualify':
            # Native batch training uses fixed manifest observations and the inherited Q4096.
            input_start = time.perf_counter()
            samples = [data[x['sample_index']] for x in manifest['items'][:args.native_batch]]
            obs = {x['sample_id']:[(i, samples[k].field_names.index('T')) for i in x['observation_indices']]
                   for k,x in enumerate(manifest['items'][:args.native_batch])}
            batch = build_observation_batch(samples, protocol, manifest_indices=obs,
                                             query_indices=manifest['query_indices']).to('cuda:0')
            torch.cuda.synchronize()
            input_seconds = time.perf_counter() - input_start
            io_start = time.perf_counter()
            for index in range(args.conditions):
                row = torch.load(output / f'{index:04d}.pt', map_location='cpu', weights_only=True)
                _ = row['members_model'][0, 0, manifest['query_indices']]
            target_io_seconds = time.perf_counter() - io_start
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-6)
            native_times = []
            for _ in range(args.native_updates):
                torch.cuda.synchronize(); t = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                backward_and_clip_model_loss(model, batch, model.parameters(), 1.0, device='cuda:0')
                optimizer.step()
                torch.cuda.synchronize(); native_times.append(time.perf_counter()-t)
            receipt.update(native_batch=args.native_batch, native_query_points=4096, native_update_seconds=native_times,
                           native_batch_input_seconds=input_seconds,
                           source_row_read_seconds=target_io_seconds, source_rows_read=args.conditions,
                           native_scope='actual microbatched backward; separate one-batch input and recently written SOURCE row reads; repaired-bank I/O may differ')
    except BaseException as e:
        receipt.update(status='failed', error_type=type(e).__name__, error=str(e))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old)
        if torch.cuda.is_initialized():
            try: torch.cuda.synchronize()
            except Exception as e: receipt.update(status='failed', teardown_error=str(e))
        if data is not None: data.close()
        receipt['end_unix'] = time.time(); receipt['wall_seconds'] = receipt['end_unix'] - started
        receipt['process_cpu_seconds'] = time.process_time() - cpu_started
        with locked_ledger(ledger) as state:
            entry = state['active'].pop(token)
            state['closed'].append({**entry, 'token':token, 'end_unix':receipt['end_unix'], 'seconds':receipt['wall_seconds']})
        atomic_json(receipt_path, receipt)
        print(receipt_path, flush=True)


def main():
    cache = os.environ.get('KEOPS_CACHE_FOLDER')
    if cache:
        build = '_'.join(platform.uname()[:3]) + f'_p{sys.version.split()[0]}'
        selected = os.environ.get('CUDA_VISIBLE_DEVICES')
        if selected:
            (Path(cache) / f'{build}_CUDA_VISIBLE_DEVICES_{selected}').mkdir(parents=True, exist_ok=True)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--round3', type=Path, default=ROOT/'cases/turbulent_combustion/runs/coherence_adaptive_repair/round_3')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    for name in ('qualify','generate'):
        s = sub.add_parser(name)
        s.add_argument('--max-seconds', type=int, required=True)
        if name == 'qualify':
            s.add_argument('--conditions', type=int, default=2)
            s.add_argument('--native-batch', type=int, default=64)
            s.add_argument('--native-updates', type=int, default=2)
    args = p.parse_args()
    if args.command != 'prepare' and args.max_seconds <= 10:
        p.error('max-seconds must exceed 10')
    if args.command == 'qualify' and (args.max_seconds > 7200 or min(args.conditions,args.native_batch,args.native_updates) < 1):
        p.error('qualification limit is 7200 seconds; counts must be positive')
    prepare(args) if args.command == 'prepare' else execute(args)

if __name__ == '__main__':
    main()
