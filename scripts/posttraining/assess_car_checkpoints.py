#!/usr/bin/env python3
"""Generate native J4 endpoint banks and replay frozen Round 1 checkpoints.

Two stages: `generate` streams one SOURCE or immutable epoch checkpoint on one
fixed panel; `score` streams a matched pair through the shared C8 observer and
native physical risk, fitting alpha on TRAIN or applying its frozen value on
VALIDATION. This tool never scores the reserved event at intermediate epochs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(os.environ.get('CAR_REPO_ROOT', Path(__file__).resolve().parents[2])).resolve()
CODE_ROOT = Path(os.environ.get('CAR_CODE_ROOT', Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(CODE_ROOT / 'src'))
sys.path.insert(0, str(CODE_ROOT))
CAMPAIGN = ROOT / 'cases/turbulent_combustion/runs/coherence_adaptive_repair/round_1'
EPOCHS = (0, 100, 250, 500, 750, 1000)
FRONTIER = (0., .125, .25, .375, .5, .625, .75, .875, 1.)


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def checked_json(path):
    value = json.loads(path.read_text())
    if value.get('sha256') != content_hash({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError(f'invalid receipt hash: {path}')
    return value


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _completed_status(path):
    status = json.loads(path.read_text())
    if status.get('status') != 'completed' or status.get('global_step') != 13000 or status.get('configured_steps') != 13000:
        raise ValueError('both arms must complete all 13000 updates')
    return status


def build_assessment_freeze(campaign, arms):
    """Bind every completed arm checkpoint and TRAIN alpha before held-out access."""
    formal_path = campaign / 'formal_freeze.json'
    formal = json.loads(formal_path.read_text())
    manifest = formal.get('manifest', {})
    if formal.get('schema') != 'car_freeze_receipt_v1' or manifest.get('status') != 'frozen' or not formal.get('manifest_sha256') or not manifest.get('train_qualification_sha256'):
        raise ValueError('valid formal freeze required')
    event = checked_json(campaign / 'assessment/native_event_redesigned.json')
    if manifest.get('event_task_sha256') != event['sha256']:
        raise ValueError('formal freeze must bind the redesigned event')
    if set(arms) != {'FFM_B1', 'FFM_A1'}:
        raise ValueError('both formal arms required')
    records = {}
    common_signature = None
    for arm, (run, alpha_dir) in arms.items():
        run, alpha_dir = Path(run).resolve(), Path(alpha_dir).resolve()
        status_path = run / 'status.json'
        _completed_status(status_path)
        checkpoints = {}
        for epoch in EPOCHS:
            checkpoint = run / 'checkpoints' / f'epoch_{epoch:03d}.pt'
            alpha_path = alpha_dir / f'epoch_{epoch:03d}.json'
            alpha = checked_json(alpha_path)
            checkpoint_sha = file_hash(checkpoint)
            if alpha.get('schema') != 'car_streamed_alpha_v1' or alpha.get('split') != 'train' or alpha.get('epoch') != epoch or alpha.get('child_checkpoint_sha256') != checkpoint_sha:
                raise ValueError(f'{arm} epoch {epoch} TRAIN alpha/checkpoint mismatch')
            signature = {k: alpha.get(k) for k in ('panel_sha256', 'source_checkpoint_sha256', 'coupling_sha256', 'core_identity', 'sample_ids')}
            if any(not signature[k] for k in signature):
                raise ValueError('TRAIN nomination lacks matched panel/source/core identities')
            if common_signature is None:
                common_signature = signature
            elif signature != common_signature:
                raise ValueError('TRAIN nominations use mismatched panels, source, streams or observer')
            checkpoints[str(epoch)] = {'checkpoint_path':str(checkpoint), 'checkpoint_sha256':checkpoint_sha,
                                       'alpha_path':str(alpha_path), 'alpha_file_sha256':file_hash(alpha_path),
                                       'alpha_receipt_sha256':alpha['sha256']}
        records[arm] = {'run':str(run), 'status_path':str(status_path), 'status_sha256':file_hash(status_path),
                        'checkpoints':checkpoints}
    if records['FFM_B1']['run'] == records['FFM_A1']['run']:
        raise ValueError('formal arms must be distinct runs')
    return {'schema':'car_assessment_freeze_v1', 'formal_freeze_sha256':file_hash(formal_path),
            'event_task_sha256':event['sha256'], 'arms':records}


def validation_gate(campaign, alpha_path):
    formal_path = campaign / 'formal_freeze.json'
    freeze_path = campaign / 'assessment/assessment_freeze.json'
    if not formal_path.is_file() or not freeze_path.is_file() or alpha_path is None or not alpha_path.is_file():
        raise ValueError('VALIDATION requires formal and complete two-arm assessment freezes and TRAIN alpha receipt')
    freeze = checked_json(freeze_path)
    if freeze.get('schema') != 'car_assessment_freeze_v1' or freeze.get('formal_freeze_sha256') != file_hash(formal_path):
        raise ValueError('formal freeze changed after assessment freeze')
    arms = freeze.get('arms', {})
    if set(arms) != {'FFM_B1', 'FFM_A1'}:
        raise ValueError('assessment freeze requires both completed arms')
    event = checked_json(campaign / 'assessment/native_event_redesigned.json')
    if event['sha256'] != freeze.get('event_task_sha256'):
        raise ValueError('frozen event changed')
    allowed = set()
    for arm, record in arms.items():
        status_path = Path(record['status_path'])
        if file_hash(status_path) != record['status_sha256']:
            raise ValueError(f'{arm} completion receipt changed')
        _completed_status(status_path)
        checkpoints = record['checkpoints']
        if set(checkpoints) != {str(epoch) for epoch in EPOCHS}:
            raise ValueError(f'{arm} missing a scheduled epoch')
        for epoch in EPOCHS:
            item = checkpoints[str(epoch)]
            checkpoint, nomination = Path(item['checkpoint_path']), Path(item['alpha_path'])
            if file_hash(checkpoint) != item['checkpoint_sha256'] or file_hash(nomination) != item['alpha_file_sha256']:
                raise ValueError(f'{arm} epoch {epoch} evidence changed')
            alpha = checked_json(nomination)
            if alpha.get('sha256') != item['alpha_receipt_sha256'] or alpha.get('epoch') != epoch or alpha.get('child_checkpoint_sha256') != item['checkpoint_sha256'] or alpha.get('split') != 'train' or alpha.get('schema') != 'car_streamed_alpha_v1':
                raise ValueError(f'{arm} epoch {epoch} TRAIN nomination invalid')
            allowed.add((str(nomination.resolve()), item['alpha_receipt_sha256']))
    alpha = checked_json(alpha_path)
    if (str(alpha_path.resolve()), alpha.get('sha256')) not in allowed:
        raise ValueError('TRAIN alpha receipt is not included in two-arm assessment freeze')
    return alpha


def panel_contract(panel, split):
    ids = panel['sample_ids']
    population = panel['population']
    if panel.get('split') != split or len(ids) != (population + 9) // 10 or len(set(ids)) != len(ids):
        raise ValueError('fixed 10% panel identity invalid')
    if panel.get('sensor_count') != 256 or panel.get('members') != 4:
        raise ValueError('panel must declare T256 and J4')
    return ids


def member_seed(index, member):
    return 9_100_003 + 11 * int(index) + int(member)


def publish_json(path, payload):
    from phycoflow_reconstruction.evaluation.adaptive_campaign import immutable_json
    immutable_json(path, {**payload, 'sha256': content_hash(payload)})


def checkpoint_contract(model, arm, epoch, config, global_step):
    """Validate the distinct FFM mature and EDM qualification epoch contracts."""
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise TypeError('integer reporting epoch required')
    optimization = config.get('optimization', {})
    rollout = config.get('rollout', {})
    if model == 'ffm':
        valid = (arm in {'FFM_B1', 'FFM_A1'} and config['model']['name'] == 'gl_rbf_cq'
                 and optimization.get('steps_per_epoch') == 13 and optimization.get('epochs') == 1000
                 and epoch in EPOCHS and rollout.get('steps') == 4 and rollout.get('solver') == 'euler')
        steps = 13
    elif model == 'edm':
        valid = (arm == 'EDM_qualification' and config['model']['name'] == 'diffusion_pde'
                 and optimization.get('steps_per_epoch') == 1 and optimization.get('epochs') == 12
                 and 0 <= epoch <= 12 and rollout.get('steps') == 32 and rollout.get('solver') == 'heun'
                 and config.get('observation_consistency', {}).get('mode') == 'native'
                 and config.get('coherence', {}).get('compute_budget', {}).get('point_count') == 40300)
        steps = 1
    else:
        raise ValueError('unknown model')
    if not valid or isinstance(global_step, bool) or global_step != steps * epoch:
        raise ValueError('checkpoint epoch or model-specific qualification contract mismatch')
    return {'steps_per_epoch': steps, 'epoch': epoch, 'global_step': global_step,
            'scope': 'EDM execution qualification; no efficacy claim' if model == 'edm' else 'FFM mature checkpoint'}


def observer_support(mode, model, config):
    count = int(config['coherence']['compute_budget']['point_count'])
    if mode == 'edm-qualification' and model == 'edm' and count == 40300:
        return 40300
    if mode == 'mature' and model == 'ffm' and count == 4096:
        return 4096
    raise ValueError('observer support is not qualified for the requested mode and model')


def generate(args):
    from phycoflow_reconstruction.evaluation.adaptive_campaign import enforce_execution_constraints
    enforce_execution_constraints(args.campaign, args.gpu_uuid, args.model)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu_uuid or not os.environ.get('KEOPS_CACHE_FOLDER'):
        raise ValueError('set exact CUDA_VISIBLE_DEVICES UUID and device-specific KEOPS_CACHE_FOLDER before Python starts')
    allowed = {'GPU-233fcd85-5c6a-6212-3f44-655252afec70', 'GPU-3ceda40c-fd5c-4b88-6c47-b3301711571e'}
    if args.gpu_uuid not in allowed:
        raise ValueError('assessment requires authorized physical GPU0 or GPU1 UUID')
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader,nounits'], text=True).splitlines()
    if args.gpu_uuid not in [v.strip() for v in devices]:
        raise ValueError('physical GPU UUID not found')
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    if any(row.split(',')[0].strip() == args.gpu_uuid for row in active.splitlines() if ',' in row):
        raise ValueError('physical GPU has an existing compute process')
    cache = Path(os.environ['KEOPS_CACHE_FOLDER'])
    name = '_'.join(platform.uname()[:3]) + f'_p{sys.version.split()[0]}_CUDA_VISIBLE_DEVICES_{args.gpu_uuid}'
    (cache / name).mkdir(parents=True, exist_ok=True)
    import torch
    import yaml

    from phycoflow_reconstruction.data.factory import open_field_dataset
    from phycoflow_reconstruction.data.manifest import dataset_fingerprint
    from phycoflow_reconstruction.data.sensor_protocols import (
        SensorProtocol,
        build_observation_batch,
    )
    from phycoflow_reconstruction.evaluation.adaptive_campaign import (
        CAP_SECONDS,
        RESERVE_SECONDS,
        SOURCES,
        begin,
        budget_stop,
        digest,
        end,
        locked_ledger,
        spent,
    )
    from phycoflow_reconstruction.training.run_store import (
        load_model_state_strict,
        load_project_checkpoint,
    )
    from phycoflow_reconstruction.training.source import load_source_model
    if args.model == 'edm' and (args.arm != 'EDM_qualification' or args.split != 'train'):
        raise ValueError('EDM is a TRAIN-only execution qualification')
    if args.model == 'ffm' and args.arm not in {'FFM_B1', 'FFM_A1'}:
        raise ValueError('FFM requires its mature arm identity')
    if args.split == 'validation':
        alpha = validation_gate(args.campaign, args.alpha_receipt)
        if args.epoch is not None and alpha.get('epoch') != args.epoch:
            raise ValueError('TRAIN alpha receipt epoch does not match requested checkpoint')
    panel_path = args.campaign / 'panels' / ('train.json' if args.split == 'train' else 'validation_sealed.json')
    panel = json.loads(panel_path.read_text())
    ids = panel_contract(panel, args.split)
    snapshot = args.campaign / 'sources' / args.model
    cfg = yaml.safe_load((snapshot / 'resolved_config.yaml').read_text())
    cfg.update(source_run=str(snapshot), source_checkpoint='last.pt')
    cfg['source'] = {'kind': 'native_run', 'admission': 'frozen_snapshot',
                     'snapshot_receipt': str(snapshot / 'snapshot_receipt.json'), 'weight_view': 'ema_materialized'}
    cfg['dataset']['physical_measure'] = 'native_rectilinear_area'
    source_hash = digest(snapshot / 'checkpoints/last.pt')
    if source_hash != SOURCES[args.model][1]:
        raise ValueError('snapshot differs from pinned source')
    if args.epoch is None:
        checkpoint = None
        checkpoint_hash = source_hash
        bank_name = 'SOURCE'
    else:
        if args.run is None:
            raise ValueError('child epoch needs --run')
        checkpoint = args.run / 'checkpoints' / f'epoch_{args.epoch:03d}.pt'
        checkpoint_hash = digest(checkpoint)
        bank_name = f'{args.arm}_epoch_{args.epoch:03d}'
        if args.split == 'validation' and alpha.get('child_checkpoint_sha256') != checkpoint_hash:
            raise ValueError('TRAIN alpha is bound to a different child checkpoint')
    output = args.campaign / 'assessment' / 'banks' / args.model / args.split / bank_name
    output.mkdir(parents=True, exist_ok=False)
    ledger = args.campaign / 'gpu_ledger.json'
    token = str(uuid.uuid4())
    with locked_ledger(ledger) as state:
        if spent(state) + args.estimated_gpu_seconds + args.stop_margin_seconds >= CAP_SECONDS - RESERVE_SECONDS:
            raise ValueError('assessment forecast exceeds remaining budget after closeout reserve')
        begin(state, token, args.gpu_uuid, 'assessment')
    started = time.perf_counter()
    try:
        model, train, source_meta = load_source_model(cfg, torch.device('cuda:0'))
        if checkpoint is not None:
            state = load_project_checkpoint(checkpoint)
            run_config = yaml.safe_load((args.run / 'resolved_config.yaml').read_text())
            checkpoint_contract(args.model, args.arm, args.epoch, run_config, state['global_step'])
            if state['source_hashes']['checkpoint'] != source_hash:
                raise ValueError('checkpoint pinned source identity mismatch')
            load_model_state_strict(model, state['model'])
        model.eval()
        dataset = train if args.split == 'train' else open_field_dataset(cfg['dataset'], split='validation', normalizer=train.normalizer)
        try:
            if len(dataset) != panel['population']:
                raise ValueError('actual split population differs from frozen manifest')
            lookup = {f'{dataset.trajectory_ids[b]}:{t}': i for i, (b,t) in enumerate(dataset._items)}
            if any(i not in lookup for i in ids):
                raise ValueError('manifest sample absent from split')
            protocol = SensorProtocol(field_counts={'T': 256}, seed=2047)
            timing, files, obs_digests = [], [], []
            with torch.no_grad():
                for ordinal, sample_id in enumerate(ids):
                    if budget_stop(ledger, safety_seconds=args.stop_margin_seconds):
                        raise RuntimeError('budget pause before next condition; partial bank preserved without receipt')
                    index = lookup[sample_id]
                    sample = dataset[index]
                    if sample.values.shape[0] != 40300:
                        raise ValueError('native 40300-point grid required')
                    points = torch.randperm(40300, generator=torch.Generator().manual_seed(2047 + index))[:256]
                    manifest = {sample_id: [(int(point), sample.field_names.index('T')) for point in points]}
                    batch = build_observation_batch([sample], protocol, query_indices=None, manifest_indices=manifest).to('cuda:0')
                    truth = batch.target_fields.detach().cpu()
                    batch.target_fields = None
                    predictions = []
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for j in range(4):
                        gen = torch.Generator(device='cuda:0').manual_seed(member_seed(index, j))
                        kwargs = {'steps': 4 if args.model == 'ffm' else 32, 'generator': gen}
                        if args.model == 'ffm':
                            kwargs['observation_config'] = {'ode_solver': 'euler', 'obs_consistency_mode': 'endpoint_smooth',
                                'obs_consistency_strength': 1., 'obs_consistency_sigma': .05,
                                'obs_consistency_schedule_power': 2., 'obs_consistency_final_clamp': True,
                                'obs_consistency_chunk_size': 4096}
                        predictions.append(model.reconstruct(batch, **kwargs).prediction.detach().cpu())
                    torch.cuda.synchronize()
                    timing.append(time.perf_counter() - t0)
                    row = {'sample_id': sample_id, 'sample_index': index, 'members_model': torch.stack(predictions),
                           'truth_model': truth, 'coordinates_model': batch.query_coords.detach().cpu(),
                           'coordinates_raw': sample.coordinates_raw.cpu(), 'native_masses': dataset.physical_weights.cpu(),
                           'obs_indices': batch.obs_indices.detach().cpu(), 'obs_values': batch.obs_values.detach().cpu(),
                           'obs_fields': batch.obs_field_ids.detach().cpu(),
                           'member_seeds': [member_seed(index,j) for j in range(4)],
                           'normalizer_offset': dataset.normalizer.offset.cpu(), 'normalizer_scale': dataset.normalizer.scale.cpu()}
                    path = output / f'{ordinal:04d}.pt'
                    torch.save(row, path)
                    files.append(digest(path))
                    obs_digests.append(content_hash({'sample_id': sample_id, 'points': points.tolist(), 'seeds': row['member_seeds']}))
            payload = {'schema':'car_native_bank_v1', 'split':args.split, 'model':args.model, 'arm':args.arm,
                       'epoch':args.epoch, 'source_sha256':source_hash, 'checkpoint_sha256':checkpoint_hash,
                       'source_weight_view':source_meta['materialized_view_sha256'], 'panel_sha256':digest(panel_path),
                       'dataset_fingerprint':dataset_fingerprint(dataset.path), 'normalizer_sha256':dataset.normalizer.digest(),
                       'field_order':list(dataset.field_names), 'sample_ids':ids,
                       'sensor_count':256, 'sensor_seed_rule':'2047+split_sample_index',
                       'member_seed_rule':'9100003+11*split_sample_index+member', 'members':4,
                       'support':40300, 'units':'model values; decode with recorded normalizer; physical units undeclared',
                       'coupling_sha256':content_hash(obs_digests), 'files_sha256':files,
                       'inference_seconds_total':sum(timing), 'first_condition_seconds':timing[0],
                       'warm_condition_seconds_mean':sum(timing[1:])/max(1,len(timing)-1),
                       'storage_bytes':sum(p.stat().st_size for p in output.glob('*.pt')),
                       'wall_seconds':time.perf_counter()-started, 'device_uuid':args.gpu_uuid}
            publish_json(output / 'receipt.json', payload)
        finally:
            if dataset is not train: dataset.close()
            train.close()
    finally:
        with locked_ledger(ledger) as state: end(state, token)
    print(output / 'receipt.json')


def score(args):
    import torch
    import yaml

    from phycoflow_reconstruction.coherence import build_enabled_families
    from phycoflow_reconstruction.coherence.physical_risk import blend_risk_quadratic
    from phycoflow_reconstruction.data.factory import open_field_dataset
    from phycoflow_reconstruction.data.training_batches import fixed_query_indices
    from phycoflow_reconstruction.evaluation.adaptive_campaign import digest
    from phycoflow_reconstruction.evaluation.frozen_coherence import FrozenCoherenceObserver
    from scripts.posttraining.freeze_car_event import FrozenRedesignedEvent
    if args.split == 'validation':
        alpha_receipt = validation_gate(args.campaign, args.alpha_receipt)
    source, child = checked_json(args.source_bank / 'receipt.json'), checked_json(args.child_bank / 'receipt.json')
    qualification = args.mode == 'edm-qualification'
    valid_child = (child['model'] == 'edm' and child['arm'] == 'EDM_qualification' and isinstance(child['epoch'], int) and 0 <= child['epoch'] <= 12) if qualification else (child['model'] == 'ffm' and child['arm'] in {'FFM_B1', 'FFM_A1'} and child['epoch'] in EPOCHS)
    if source['epoch'] is not None or not valid_child or (qualification and args.split != 'train'):
        raise ValueError('SOURCE and mode-specific saved child bank required')
    for key in ('split', 'model', 'source_sha256', 'source_weight_view', 'panel_sha256',
                'dataset_fingerprint', 'normalizer_sha256', 'field_order', 'sample_ids', 'coupling_sha256', 'support', 'members'):
        if source[key] != child[key]:
            raise ValueError(f'unmatched endpoint bank {key}')
    if source['split'] != args.split or source['support'] != 40300 or source['members'] != 4:
        raise ValueError('native J4 panel required')
    panel_path = args.campaign / 'panels' / ('train.json' if args.split == 'train' else 'validation_sealed.json')
    panel = json.loads(panel_path.read_text())
    if panel_contract(panel, args.split) != source['sample_ids'] or digest(panel_path) != source['panel_sha256']:
        raise ValueError('endpoint banks do not match fixed assessment panel')
    if args.split == 'validation':
        if alpha_receipt['child_checkpoint_sha256'] != child['checkpoint_sha256'] or alpha_receipt['source_checkpoint_sha256'] != source['checkpoint_sha256'] or alpha_receipt['epoch'] != child['epoch']:
            raise ValueError('TRAIN alpha receipt does not match checkpoint')
        fixed_feasible = next(row['feasible'] for row in alpha_receipt['rows'] if row['alpha'] == .5)
        alphas = tuple(sorted({0., 1., float(alpha_receipt['selected_alpha'])} | ({.5} if fixed_feasible else set())))
    else:
        alphas = (0., 1.) if qualification else FRONTIER
    if not (args.source_seconds > 0 and args.child_seconds > 0):
        raise ValueError('positive measured per-condition SOURCE and child inference costs required')
    cfg = yaml.safe_load((args.core_run / 'resolved_config.yaml').read_text())
    observer_points = observer_support(args.mode, source['model'], cfg)
    dataset = open_field_dataset(cfg['dataset'], split='train')
    try:
        families = build_enabled_families(cfg['coherence'], dataset.data_spec, dataset.normalizer)
        for name, family in families.items():
            family.load_state_artifact(torch.load(args.core_run / 'artifacts' / f'{name}_family.pt', map_location='cpu', weights_only=True))
        artifacts = torch.load(args.core_run / 'artifacts/scalable_coherence.pt', map_location='cpu', weights_only=True)
        core_identity = {'config_sha256': digest(args.core_run / 'resolved_config.yaml'),
                         'scalable_sha256': digest(args.core_run / 'artifacts/scalable_coherence.pt'),
                         'family_sha256': {name: digest(args.core_run / 'artifacts' / f'{name}_family.pt') for name in families}}
        if artifacts['provenance']['source_checkpoint_sha256'] != source['source_sha256']:
            raise ValueError('core observer source identity differs from endpoint bank')
        if args.split == 'validation' and alpha_receipt['core_identity'] != core_identity:
            raise ValueError('core observer differs from frozen TRAIN alpha')
        if (list(artifacts['provenance']['field_order']) != source['field_order']
                or list(dataset.field_names) != source['field_order']
                or artifacts['provenance']['normalizer_sha256'] != source['normalizer_sha256']
                or dataset.normalizer.digest() != source['normalizer_sha256']):
            raise ValueError('core field order or normalizer identity mismatch')
        qseed = int(cfg['coherence']['compute_budget']['query_seed'])
        query = fixed_query_indices(40300, observer_points, seed=qseed)
        event = None
        if args.split == 'validation' and child['epoch'] == 1000:
            event = FrozenRedesignedEvent(json.loads((args.campaign / 'assessment/native_event_redesigned.json').read_text()))
            formal = json.loads((args.campaign / 'formal_freeze.json').read_text())
            if formal['manifest'].get('event_task_sha256') != event.receipt()['sha256']:
                raise ValueError('formal freeze does not bind the redesigned event')
        sums = {a: torch.zeros(3, dtype=torch.float64) for a in alphas}
        risks = {a: None for a in alphas}
        sensor_max = {a: 0. for a in alphas}
        event_values = {a: [] for a in alphas}
        compile_time = score_time = 0.
        first_score_seconds = None
        started = time.perf_counter()
        count = len(source['sample_ids'])
        for i, sample_id in enumerate(source['sample_ids']):
            sp, cp = args.source_bank / f'{i:04d}.pt', args.child_bank / f'{i:04d}.pt'
            if digest(sp) != source['files_sha256'][i] or digest(cp) != child['files_sha256'][i]:
                raise ValueError('native endpoint bank corrupted')
            s, c = (torch.load(p, map_location='cpu', weights_only=True) for p in (sp, cp))
            if s['sample_id'] != sample_id or c['sample_id'] != sample_id:
                raise ValueError('endpoint sample identity mismatch')
            for key in ('truth_model', 'coordinates_model', 'coordinates_raw', 'native_masses', 'obs_indices', 'obs_values', 'obs_fields', 'normalizer_offset', 'normalizer_scale'):
                if not torch.equal(s[key], c[key]):
                    raise ValueError(f'paired endpoint mismatch: {key}')
            if s['member_seeds'] != c['member_seeds']:
                raise ValueError('paired member stream mismatch')
            x, y, z = s['members_model'], c['members_model'], s['truth_model']
            mass = s['native_masses'].double()
            off, scale = s['normalizer_offset'], s['normalizer_scale']
            xphys, yphys, zphys = x * scale + off, y * scale + off, z * scale + off
            r0, cross, square = blend_risk_quadratic(xphys, yphys, zphys, mass)
            r0, cross, square = (v.double().mean((0, 1)) for v in (r0, cross, square))
            begin = time.perf_counter()
            observer = FrozenCoherenceObserver(artifacts, families,
                expected_provenance=artifacts['provenance'], reference=z[:, query],
                coordinates=s['coordinates_model'][:, query], sample_ids=[sample_id],
                member_weights=mass[query].unsqueeze(0), reference_weights=mass[query].unsqueeze(0),
                mode='conditional', query_seed=2027, complete_scale=True)
            compile_time += time.perf_counter() - begin
            for a in alphas:
                blend = x + a * (y - x)
                begin = time.perf_counter()
                values = observer.score_tensors(blend[:, :, query])
                elapsed = time.perf_counter() - begin
                if first_score_seconds is None: first_score_seconds = elapsed
                score_time += elapsed
                sums[a] += torch.tensor([float(values[k].mean()) for k in ('state', 'scale', 'shape')])
                risk = r0 + 2*a*cross + a*a*square
                risks[a] = risk if risks[a] is None else risks[a] + risk
                obs, fields = s['obs_indices'].long()[0], s['obs_fields'].long()[0]
                observed = blend[:, 0, obs, fields]
                sensor_max[a] = max(sensor_max[a], float((observed - s['obs_values'][0, :, 0]).abs().max()))
            if event is not None:
                for a in alphas:
                    blend = xphys + a * (yphys - xphys)
                    event_values[a].append(event.score(blend, zphys, mass, s['coordinates_raw']))
        core = {a:(sums[a] / count).tolist() for a in alphas}
        risk = {a:(risks[a] / count).tolist() for a in alphas}
        if qualification:
            payload = {'schema':'car_edm_qualification_assessment_v1', 'split':'train', 'epoch':child['epoch'],
                       'scope':'native EDM execution qualification; no efficacy or alpha-adoption claim',
                       'source_bank_receipt_sha256':source['sha256'], 'child_bank_receipt_sha256':child['sha256'],
                       'sample_ids':source['sample_ids'], 'core_identity':core_identity,
                       'core_support':'full native 40300', 'risk_support':'native decoded 40300',
                       'policies':{str(a):{'core':core[a], 'member_mse':risk[a],
                                            'sensor_max_abs_model_units':sensor_max[a]} for a in alphas},
                       'observer_compile_seconds':compile_time, 'observer_score_seconds':score_time,
                       'observer_first_score_seconds':first_score_seconds,
                       'wall_seconds':time.perf_counter()-started, 'samples':count}
            publish_json(args.output, payload)
            print(args.output)
            return
        if args.split == 'train':
            denominators = torch.tensor(core[0.], dtype=torch.float64)
            if not torch.isfinite(denominators).all() or (denominators <= 0).any():
                raise ValueError('positive frozen SOURCE core denominators required')
            source_risk = torch.tensor(risk[0.], dtype=torch.float64)
            rows = []
            for a in alphas:
                eligible = bool(torch.all(torch.tensor(risk[a], dtype=torch.float64) <= 1.1 * source_risk)) and sensor_max[a] <= args.sensor_tolerance
                utility = float((torch.tensor(core[a], dtype=torch.float64) / denominators).sum())
                infer = args.source_seconds if a == 0 else args.child_seconds if a == 1 else args.source_seconds + args.child_seconds
                rows.append({'alpha':a, 'core':core[a], 'member_mse':risk[a], 'sensor_max_abs_model_units':sensor_max[a],
                             'feasible':eligible, 'utility':utility, 'inference_seconds_per_condition':infer})
            eligible_rows = [r for r in rows if r['feasible']]
            if not eligible_rows: raise ValueError('no TRAIN-feasible output including SOURCE')
            selected = min(eligible_rows, key=lambda r:(r['utility'],r['inference_seconds_per_condition'],r['alpha']))['alpha']
            payload = {'schema':'car_streamed_alpha_v1','split':'train','epoch':child['epoch'],
                       'source_checkpoint_sha256':source['checkpoint_sha256'], 'child_checkpoint_sha256':child['checkpoint_sha256'],
                       'source_bank_receipt_sha256':source['sha256'],'child_bank_receipt_sha256':child['sha256'],
                       'panel_sha256':source['panel_sha256'],'coupling_sha256':source['coupling_sha256'],
                       'core_artifact_sha256':core_identity['scalable_sha256'], 'core_identity':core_identity,
                       'core_support':'fixed Q4096', 'risk_support':'native decoded 40300',
                       'sample_ids':source['sample_ids'], 'source_core_denominators':denominators.tolist(),
                       'sensor_tolerance_model_units':args.sensor_tolerance, 'frontier':list(FRONTIER),
                       'rows':rows,'selected_alpha':selected,
                       'tie_break':'lower measured per-condition inference cost, then lower alpha'}
        else:
            a = float(alpha_receipt['selected_alpha'])
            payload = {'schema':'car_checkpoint_assessment_v1','split':'validation','epoch':child['epoch'],
                       'alpha':a,'alpha_receipt_sha256':alpha_receipt['sha256'],
                       'source_bank_receipt_sha256':source['sha256'],'child_bank_receipt_sha256':child['sha256'],
                       'sample_ids':source['sample_ids'],'core':core[a],'member_mse':risk[a],
                       'sensor_max_abs_model_units':sensor_max[a], 'core_support':'fixed Q4096',
                       'risk_support':'native decoded 40300', 'policies': {str(k): {'core':core[k], 'member_mse':risk[k], 'sensor_max_abs_model_units':sensor_max[k]} for k in alphas}}
            if event is not None:
                payload['event_task_sha256'] = event.receipt()['sha256']
                payload['event'] = {}
                for event_alpha, events in event_values.items():
                    def stack(key, events=events):
                        return torch.cat([v[key].reshape(-1) for v in events]).tolist()
                    result = {k:stack(k) for k in ('area_crps','truth_area_fraction','member_area_fraction',
                                  'truth_first_moment','member_first_moment','overlap_fraction','missing_fraction','excess_fraction')}
                    result['truth_unary_area_fractions'] = {k:torch.cat([v['truth_unary_area_fractions'][k].reshape(-1) for v in events]).tolist() for k in ('T','CO','CH4')}
                    result['member_unary_area_fractions'] = {k:torch.cat([v['member_unary_area_fractions'][k].reshape(-1) for v in events]).tolist() for k in ('T','CO','CH4')}
                    payload['event'][str(event_alpha)] = result
        selected_alpha = selected if args.split == 'train' else float(alpha_receipt['selected_alpha'])
        blend_dir = args.blend_bank
        blend_dir.mkdir(parents=True, exist_ok=False)
        blend_hashes = []
        blend_started = time.perf_counter()
        for i, sample_id in enumerate(source['sample_ids']):
            sp, cp = args.source_bank / f'{i:04d}.pt', args.child_bank / f'{i:04d}.pt'
            if digest(sp) != source['files_sha256'][i] or digest(cp) != child['files_sha256'][i]:
                raise ValueError('native bank changed during blend generation')
            s, c = (torch.load(p, map_location='cpu', weights_only=True) for p in (sp, cp))
            blended = {'sample_id': sample_id, 'members_model': s['members_model'] + selected_alpha * (c['members_model'] - s['members_model']),
                       'truth_model': s['truth_model'], 'member_seeds': s['member_seeds'],
                       'normalizer_offset': s['normalizer_offset'], 'normalizer_scale': s['normalizer_scale']}
            path = blend_dir / f'{i:04d}.pt'
            torch.save(blended, path)
            blend_hashes.append(digest(path))
        blend_receipt = {'schema': 'car_blended_bank_v1', 'split': args.split, 'epoch': child['epoch'],
                         'alpha': selected_alpha, 'source_bank_sha256': source['sha256'],
                         'child_bank_sha256': child['sha256'], 'coupling_sha256': source['coupling_sha256'],
                         'sample_ids': source['sample_ids'], 'files_sha256': blend_hashes,
                         'support': 40300, 'members': 4,
                         'storage_bytes': sum(p.stat().st_size for p in blend_dir.glob('*.pt')),
                         'write_seconds': time.perf_counter() - blend_started}
        publish_json(blend_dir / 'receipt.json', blend_receipt)
        payload['blended_bank_receipt_sha256'] = checked_json(blend_dir / 'receipt.json')['sha256']
        payload.update({'observer_compile_seconds':compile_time,'observer_score_seconds':score_time,
                        'observer_first_score_seconds':first_score_seconds,
                        'wall_seconds':time.perf_counter()-started, 'samples':count})
        publish_json(args.output, payload)
        print(args.output)
    finally:
        dataset.close()



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=CAMPAIGN)
    sub = parser.add_subparsers(dest='command', required=True)
    gen = sub.add_parser('generate')
    gen.add_argument('--split', choices=('train', 'validation'), required=True)
    gen.add_argument('--model', choices=('ffm', 'edm'), required=True)
    gen.add_argument('--gpu-uuid', required=True)
    gen.add_argument('--estimated-gpu-seconds', type=float, required=True)
    gen.add_argument('--stop-margin-seconds', type=float, required=True)
    gen.add_argument('--arm', choices=('FFM_B1', 'FFM_A1', 'EDM_qualification'), default='FFM_B1')
    gen.add_argument('--run', type=Path)
    gen.add_argument('--epoch', type=int)
    gen.add_argument('--alpha-receipt', type=Path)
    freeze = sub.add_parser('freeze-assessment')
    freeze.add_argument('--b1-run', type=Path, required=True)
    freeze.add_argument('--a1-run', type=Path, required=True)
    freeze.add_argument('--b1-alpha-dir', type=Path, required=True)
    freeze.add_argument('--a1-alpha-dir', type=Path, required=True)
    assess = sub.add_parser('score')
    assess.add_argument('--split', choices=('train', 'validation'), required=True)
    assess.add_argument('--mode', choices=('mature', 'edm-qualification'), default='mature')
    assess.add_argument('--source-bank', type=Path, required=True)
    assess.add_argument('--child-bank', type=Path, required=True)
    assess.add_argument('--core-run', type=Path, required=True)
    assess.add_argument('--alpha-receipt', type=Path)
    assess.add_argument('--source-seconds', type=float, required=True)
    assess.add_argument('--child-seconds', type=float, required=True)
    assess.add_argument('--sensor-tolerance', type=float, required=True)
    assess.add_argument('--output', type=Path, required=True)
    assess.add_argument('--blend-bank', type=Path)
    args = parser.parse_args()
    args.campaign = args.campaign.resolve()
    if args.command == 'freeze-assessment':
        payload = build_assessment_freeze(args.campaign, {'FFM_B1':(args.b1_run, args.b1_alpha_dir), 'FFM_A1':(args.a1_run, args.a1_alpha_dir)})
        path = args.campaign / 'assessment/assessment_freeze.json'
        publish_json(path, payload)
        print(path)
    elif args.command == 'generate':
        import math
        if not all(math.isfinite(x) and x > 0 for x in (args.estimated_gpu_seconds, args.stop_margin_seconds)):
            raise ValueError('positive finite forecast and stop margin required')
        generate(args)
    else:
        if not 0 <= args.sensor_tolerance < 1:
            raise ValueError('frozen sensor tolerance must be nonnegative and <1 model unit')
        if args.mode == 'mature' and args.blend_bank is None:
            raise ValueError('mature assessment requires a blended bank path')
        score(args)


if __name__ == '__main__':
    main()
