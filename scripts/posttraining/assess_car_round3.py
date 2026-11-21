#!/usr/bin/env python3
"""Generate one matched raw J4 bank per checkpoint and score both repairs together."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from scripts.posttraining.assess_car_round2_checkpoints import (hash_file, matched_rows,
    panel_contract, score_event, sensor_mse)

ARMS = ('FFM_RawRisk_Uniform_Repair', 'FFM_SourceScaledRisk_Uniform_Repair')


def load_json(path):
    return json.loads(Path(path).read_text())


def verify(ref):
    path = Path(ref['path']).resolve(strict=True)
    if hash_file(path) != ref['sha256']:
        raise ValueError(f'bound artifact changed: {path}')
    return path


def context(campaign):
    campaign = campaign.resolve(strict=True)
    round2 = campaign.parent / 'round_2'
    prepared_path = campaign / 'configs/prepared.json'
    prepared = load_json(prepared_path)
    freeze_path = round2 / 'assessment/early_closeout/freeze/freeze.json'
    freeze = load_json(freeze_path)
    panel_path = verify(freeze['validation_panel'])
    panel = load_json(panel_path)
    ids = panel_contract(panel, 'validation')
    if len(ids) != 100 or panel['members']['ffm'] != 4:
        raise ValueError('frozen VALIDATION100 J4 required')
    source_dir = round2 / 'assessment/early_closeout/validation/source_ffm'
    receipt_path = source_dir / 'receipt.json'
    source = load_json(receipt_path)
    baseline_path = round2 / 'assessment/early_closeout/validation/scores/FFM_Selected_Bank_Uniform_FullModel_epoch_750/assessment.json'
    baseline = load_json(baseline_path)
    artifact = freeze['artifacts']['ffm']
    for key in ('core', 'source_checkpoint', 'contract'):
        verify(artifact[key])
    if (source.get('status') != 'completed' or source.get('kind') != 'source'
            or source.get('split') != 'validation' or source.get('family') != 'ffm'
            or source.get('sample_ids') != ids or source.get('members') != 4
            or source.get('panel_sha256') != hash_file(panel_path)
            or source.get('source_sha256') != artifact['source_checkpoint']['sha256']
            or baseline.get('source_bank_sha256') != hash_file(receipt_path)
            or baseline.get('panel_sha256') != hash_file(panel_path)
            or baseline.get('contract_sha256') != artifact['contract']['sha256']
            or baseline.get('core_sha256') != artifact['core']['sha256']):
        raise ValueError('cached SOURCE, panel, score and observer identities do not match')
    if Path(prepared['source_snapshot']).resolve() != Path(artifact['source_checkpoint']['path']).parent.parent.resolve():
        raise ValueError('prepared source snapshot differs from cached SOURCE')
    for arm in ARMS:
        verify(prepared['arms'][arm])
    return dict(campaign=campaign, prepared=prepared, prepared_path=prepared_path,
        freeze=freeze, freeze_path=freeze_path, panel=panel, panel_path=panel_path,
        source_dir=source_dir, source=source, source_receipt=receipt_path,
        baseline=baseline, baseline_path=baseline_path, artifact=artifact)


def check_bank(bank, ctx, epoch, arm):
    receipt = load_json(bank / 'receipt.json')
    if (receipt.get('schema') != 'car_round3_native_bank_v1' or receipt.get('status') != 'completed'
            or receipt.get('epoch') != epoch or receipt.get('arm') != arm
            or receipt.get('source_bank_sha256') != hash_file(ctx['source_receipt'])
            or receipt.get('panel_sha256') != hash_file(ctx['panel_path'])
            or receipt.get('prepared_sha256') != hash_file(ctx['prepared_path'])
            or receipt.get('source_checkpoint_sha256') != ctx['source']['source_sha256']
            or receipt.get('sample_ids') != ctx['source']['sample_ids'] or receipt.get('members') != 4):
        raise ValueError(f'bank identity mismatch: {bank}')
    if hash_file(receipt['checkpoint']) != receipt['checkpoint_sha256']:
        raise ValueError('bank checkpoint changed')
    if hash_file(receipt['config_path']) != receipt['config_sha256']:
        raise ValueError('bank configuration changed')
    return receipt


def finalize_decision(result):
    """Apply the predeclared endpoint margins to the two raw outputs."""
    for row in result['candidates'].values():
        differences = row['normalized_core_differences']
        fidelity, task = row['fidelity'], row['reused_tasks']
        row['replacement_eligible'] = bool(
            fidelity['primary_pass'] and fidelity['legacy_member_pass']
            and all(value <= .02 for value in differences.values())
            and any(value <= -.05 for value in differences.values())
            and task['regional_normalized_improvement'] >= .05
            and task['event_normalized_difference'] <= .05)
    raw, scaled = (result['candidates'][arm] for arm in ARMS)
    result['paired_scaled_minus_raw'] = {
        'regional_score': scaled['reused_tasks']['regional_score'] - raw['reused_tasks']['regional_score'],
        'regional_normalized_improvement': scaled['reused_tasks']['regional_normalized_improvement'] - raw['reused_tasks']['regional_normalized_improvement'],
        'event_score': scaled['reused_tasks']['event_score'] - raw['reused_tasks']['event_score'],
        'event_per_condition': [b-a for a,b in zip(raw['reused_tasks']['event_per_condition'], scaled['reused_tasks']['event_per_condition'], strict=True)],
        'regional_per_condition': [b-a for a,b in zip(raw['reused_tasks']['regional_per_condition'], scaled['reused_tasks']['regional_per_condition'], strict=True)],
        'normalized_core_differences': {k: scaled['normalized_core_differences'][k] - raw['normalized_core_differences'][k] for k in ('state','scale','shape')},
        'signed_mean_risk': [b-a for a,b in zip(raw['risk']['signed_mean_risk'], scaled['risk']['signed_mean_risk'], strict=True)],
        'member_mse': [b-a for a,b in zip(raw['risk']['member_mse'], scaled['risk']['member_mse'], strict=True)],
    }
    eligible = [arm for arm in ARMS if result['candidates'][arm]['replacement_eligible']]
    if len(eligible) == 2:
        gain = result['paired_scaled_minus_raw']['regional_normalized_improvement']
        choice = ARMS[1] if gain >= .02 else ARMS[0]
    else:
        choice = eligible[0] if eligible else 'retain SOURCE/C8'
    result['predeclared_recommendation'] = choice


def generate(args, ctx, *, config_name='resolved_config.yaml'):
    import torch
    import yaml
    from phycoflow_reconstruction.data.factory import open_field_dataset
    from phycoflow_reconstruction.data.manifest import dataset_fingerprint
    from phycoflow_reconstruction.data.sensor_protocols import SensorProtocol, build_observation_batch
    from phycoflow_reconstruction.training.model_lifecycle import load_training_aux_state
    from phycoflow_reconstruction.training.rollout import differentiable_reconstruction
    from phycoflow_reconstruction.training.run_store import load_model_state_strict, load_project_checkpoint
    from phycoflow_reconstruction.training.source import load_source_model
    from phycoflow_reconstruction.training.scalable_fitting import _digest
    from scripts.posttraining.assess_car_round2_checkpoints import same_scientific_config

    run = args.run.resolve(strict=True)
    if (ctx['campaign'] / 'runs/formal' / args.arm).resolve() not in run.parents:
        raise ValueError('named formal run required')
    config_path = run / config_name
    cfg = yaml.safe_load(config_path.read_text())
    expected = yaml.safe_load(verify(ctx['prepared']['arms'][args.arm]).read_text())
    if not same_scientific_config(cfg, expected):
        raise ValueError('resolved run differs from prepared scientific config')
    checkpoint = run / 'checkpoints' / f'epoch_{args.epoch:03d}.pt'
    checkpoint_sha = hash_file(checkpoint)
    state = load_project_checkpoint(checkpoint)
    if (state.get('global_step') != args.epoch * 12 or
        state.get('source_hashes', {}).get('checkpoint') != ctx['source']['source_sha256']):
        raise ValueError('checkpoint age or source identity differs')
    output = args.output.resolve()
    if ctx['campaign'] / 'assessment' not in output.parents:
        raise ValueError('bank must be in Round 3 assessment storage')
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    receipt = dict(schema='car_round3_native_bank_v1', status='failed', arm=args.arm, epoch=args.epoch,
        checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        config_path=str(config_path), config_sha256=hash_file(config_path), prepared_sha256=hash_file(ctx['prepared_path']),
        source_bank_sha256=hash_file(ctx['source_receipt']), panel_sha256=hash_file(ctx['panel_path']),
        source_checkpoint_sha256=ctx['source']['source_sha256'], device=args.device, start_unix=started,
        code_sha256=hash_file(Path(__file__)), members=4, sample_ids=ctx['source']['sample_ids'])
    receipt.update(args.runtime_device)
    receipt.update(torch_version=torch.__version__, torch_cuda_version=torch.version.cuda)
    train = dataset = None
    try:
        device = torch.device(args.device)
        model, train, _ = load_source_model(cfg, device)
        if _digest(model.state_dict()) != ctx['source']['source_parameter_buffer_sha256']:
            raise ValueError('materialized SOURCE weight view differs')
        load_model_state_strict(model, state['model'])
        load_training_aux_state(model, state)
        model.eval()
        dataset = open_field_dataset(cfg['dataset'], split='validation', normalizer=train.normalizer)
        panel = ctx['panel']
        if (len(dataset) != panel['population'] or dataset.normalizer.digest() != panel['normalizer_sha256']
                or dataset_fingerprint(dataset.path) != panel['dataset_fingerprint']):
            raise ValueError('validation dataset/normalizer differs from frozen panel')
        files = []
        protocol = SensorProtocol(field_counts={'T': 256}, seed=2047)
        with torch.no_grad():
            for ordinal, (index, sample_id) in enumerate(zip(panel['sample_indices'], panel['sample_ids'], strict=True)):
                source_path = ctx['source_dir'] / f'{ordinal:04d}.pt'
                if hash_file(source_path) != ctx['source']['files_sha256'][ordinal]:
                    raise ValueError('SOURCE row changed')
                source = torch.load(source_path, map_location='cpu', weights_only=True)
                sample = dataset[index]
                actual_id = f'{dataset.trajectory_ids[dataset._items[index][0]]}:{dataset._items[index][1]}'
                if actual_id != sample_id or sample.values.shape[0] != 40300:
                    raise ValueError('native support or sample identity changed')
                points = torch.randperm(40300, generator=torch.Generator().manual_seed(2047 + index))[:256]
                observations = {sample_id: [(int(point), sample.field_names.index('T')) for point in points]}
                batch = build_observation_batch([sample], protocol, query_indices=None, manifest_indices=observations).to(device)
                truth, inputs = batch.target_fields.detach().cpu(), replace(batch, target_fields=None)
                members = [differentiable_reconstruction(model, inputs, steps=int(cfg['rollout']['steps']),
                    solver=cfg['rollout']['solver'], generator=torch.Generator(device=device).manual_seed(seed),
                    observation_config=cfg['observation_consistency']).detach().cpu() for seed in source['member_seeds']]
                row = dict(sample_id=sample_id, sample_index=index, members_model=torch.stack(members),
                    truth_model=truth, coordinates_model=batch.query_coords.detach().cpu(),
                    coordinates_raw=sample.coordinates_raw.cpu(), native_masses=dataset.physical_weights.cpu(),
                    obs_indices=batch.obs_indices.detach().cpu(), obs_values=batch.obs_values.detach().cpu(),
                    obs_fields=batch.obs_field_ids.detach().cpu(), member_seeds=source['member_seeds'],
                    normalizer_offset=dataset.normalizer.offset.cpu(), normalizer_scale=dataset.normalizer.scale.cpu())
                matched_rows(source, row)
                path = output / f'{ordinal:04d}.pt'
                torch.save(row, path)
                files.append(hash_file(path))
        receipt.update(status='completed', files_sha256=files,
            device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device))
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if dataset is not None: dataset.close()
        if train is not None: train.close()
        receipt.update(end_unix=time.time(), wall_seconds=time.time()-started)
        (output / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')


def generate_with_lease(args, ctx):
    """Bound and charge the entire occupied device interval, including failures."""
    from phycoflow_reconstruction.evaluation.adaptive_campaign import atomic_json, locked_ledger, spent

    selected = os.environ.get('CUDA_VISIBLE_DEVICES')
    if not selected or ',' in selected or not os.environ.get('KEOPS_CACHE_FOLDER') or args.device != 'cuda:0':
        raise ValueError('select one CUDA_VISIBLE_DEVICES entry, KeOps cache, and logical cuda:0')
    if not math.isfinite(args.max_seconds) or not math.isfinite(args.save_margin_seconds) or not 0 < args.save_margin_seconds < args.max_seconds:
        raise ValueError('positive bounded assessment and save margin required')
    devices = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'], text=True)
    rows = [[v.strip() for v in line.split(',', 2)] for line in devices.splitlines()]
    matches = [row for row in rows if len(row) == 3 and selected in row[:2]]
    if len(matches) != 1: raise ValueError('runtime device does not resolve uniquely')
    index, gpu_uuid, name = matches[0]
    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'], text=True)
    if any(line.split(',')[0].strip() == gpu_uuid for line in active.splitlines()):
        raise ValueError('runtime GPU already has a compute owner')
    token, started = uuid.uuid4().hex, time.time()
    ledger = ctx['campaign'] / 'operations/gpu_ledger.json'
    with locked_ledger(ledger) as state:
        if spent(state) + args.max_seconds > 45 * 3600:
            raise ValueError('assessment exceeds remaining 45 GPU-hour budget')
        if any(item['gpu_uuid'] == gpu_uuid for item in state['active'].values()):
            raise ValueError('runtime GPU already leased')
        state['active'][token] = dict(gpu_uuid=gpu_uuid, phase='round3_native_assessment', start_unix=started, pid=os.getpid())
    args.runtime_device = dict(gpu_uuid=gpu_uuid, gpu_index=index, gpu_name=name,
        cuda_visible_devices=selected, platform=platform.platform(), python=sys.version,
        keops_cache=os.environ['KEOPS_CACHE_FOLDER'], lease_token=token,
        max_seconds=args.max_seconds, save_margin_seconds=args.save_margin_seconds)
    launch = dict(schema='car_round3_assessment_launch_v1', token=token, arm=args.arm, epoch=args.epoch,
        start_unix=started, pid=os.getpid(), **args.runtime_device)
    outcome = {'status':'failed'}
    previous = signal.getsignal(signal.SIGALRM)
    def expire(_signum, _frame):
        raise TimeoutError('assessment reached its save margin')
    try:
        atomic_json(ctx['campaign']/ 'operations/launches' / f'{token}.json', launch)
        signal.signal(signal.SIGALRM, expire)
        signal.alarm(max(1, math.ceil(args.max_seconds-args.save_margin_seconds)))
        generate(args, ctx)
        outcome['status'] = 'completed'
    except BaseException as error:
        outcome.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
        ended = time.time()
        with locked_ledger(ledger) as state:
            entry = state['active'].pop(token)
            state['closed'].append({**entry, 'token':token, 'end_unix':ended, 'seconds':ended-started})
        atomic_json(ctx['campaign']/ 'operations/completions' / f'{token}.json',
            {**launch, **outcome, 'end_unix':ended, 'seconds':ended-started})


def score(args, ctx):
    import torch
    import yaml
    from phycoflow_reconstruction.coherence import build_enabled_families
    from phycoflow_reconstruction.data.factory import open_field_dataset
    from phycoflow_reconstruction.evaluation.frozen_coherence import FrozenCoherenceObserver
    from phycoflow_reconstruction.evaluation.round2_assessment import native_risks, field_sensor_eligibility, scalar_task_score
    from phycoflow_reconstruction.evaluation.round2_task import regional_covariance
    from scripts.posttraining.freeze_car_event import FrozenRedesignedEvent

    banks = {ARMS[0]: args.raw_bank.resolve(strict=True), ARMS[1]: args.scaled_bank.resolve(strict=True)}
    receipts = {arm: check_bank(bank, ctx, args.epoch, arm) for arm, bank in banks.items()}
    output = args.output.resolve()
    if ctx['campaign'] / 'assessment' not in output.parents:
        raise ValueError('score output must be in Round 3 assessment storage')
    output.mkdir(parents=True, exist_ok=False)
    started, cpu = time.time(), time.process_time()
    status = dict(schema='car_round3_cpu_score_v1', status='failed', epoch=args.epoch, start_unix=started)
    dataset = None
    try:
        baseline = next(row for row in ctx['baseline']['rows'] if row['alpha'] == 0.)
        contract = load_json(verify(ctx['artifact']['contract']))
        if contract['source_checkpoint_sha256'] != ctx['source']['source_sha256']:
            raise ValueError('contract source differs')
        core_run = Path(ctx['artifact']['core_run'])
        cfg = yaml.safe_load((core_run / 'resolved_config.yaml').read_text())
        dataset = open_field_dataset(cfg['dataset'], split='train')
        if dataset.normalizer.digest() != ctx['source']['normalizer_sha256'] or list(dataset.field_names) != ctx['source']['field_order']:
            raise ValueError('TRAIN normalizer or field order differs')
        reference = load_json(verify(ctx['prepared']['source_reference']))
        allowance = torch.as_tensor(reference['reference']['absolute_allowance'], dtype=torch.float64)
        native_allowance = allowance * dataset.normalizer.scale.double().square()
        if not torch.allclose(native_allowance, torch.as_tensor(contract['mean_allowance_physical'], dtype=torch.float64), rtol=1e-9, atol=0):
            raise ValueError('TRAIN allowance conversion differs from frozen native field allowance')
        families = build_enabled_families(cfg['coherence'], dataset.data_spec, dataset.normalizer)
        manifest = load_json(core_run / 'run_manifest.json')
        for name, family in families.items():
            path = core_run / 'artifacts' / f'{name}_family.pt'
            if hash_file(path) != manifest['coherence_family_state_sha256s'][name]:
                raise ValueError('observer family artifact changed')
            family.load_state_artifact(torch.load(path, map_location='cpu', weights_only=True))
        core = torch.load(verify(ctx['artifact']['core']), map_location='cpu', weights_only=True)
        if core.get('adaptive_bank') is not None or core['provenance']['source_checkpoint_sha256'] != ctx['source']['source_sha256']:
            raise ValueError('SOURCE-bound core observer required')
        query = torch.tensor(ctx['panel']['query_indices'], dtype=torch.long)
        if query.numel() != 4096 or query.unique().numel() != 4096 or (query < 0).any() or (query >= 40300).any():
            raise ValueError('invalid frozen Q4096 support')
        endpoint = args.epoch == 500
        event = FrozenRedesignedEvent(load_json(verify(ctx['freeze']['event']))) if endpoint else None
        task = load_json(verify(ctx['freeze']['reserved_task'])) if endpoint else None
        rows = {arm: {'scores': {k: 0. for k in ('state','scale','shape')}, 'risk': {},
                      'sensor': None, 'sensor_counts': None, 'per_condition_scores': [],
                      'per_condition_member_mse': [], 'event': [], 'regional': []} for arm in ARMS}
        for i, sample_id in enumerate(ctx['source']['sample_ids']):
            source_path = ctx['source_dir'] / f'{i:04d}.pt'
            if hash_file(source_path) != ctx['source']['files_sha256'][i]: raise ValueError('SOURCE row changed')
            source = torch.load(source_path, map_location='cpu', weights_only=True)
            if source['sample_id'] != sample_id: raise ValueError('SOURCE order differs')
            truth, mass = source['truth_model'], source['native_masses'].double()
            scale, offset = source['normalizer_scale'], source['normalizer_offset']
            observer = FrozenCoherenceObserver(core, families, expected_provenance=core['provenance'],
                reference=truth[:,query], coordinates=source['coordinates_model'][:,query], sample_ids=[sample_id],
                member_weights=mass[query].unsqueeze(0), reference_weights=mass[query].unsqueeze(0),
                mode='conditional', query_seed=int(contract['shape_query_seed']), complete_scale=True)
            truth_physical = truth*scale+offset
            for arm, bank in banks.items():
                path = bank / f'{i:04d}.pt'
                if hash_file(path) != receipts[arm]['files_sha256'][i]: raise ValueError('candidate row changed')
                child = torch.load(path, map_location='cpu', weights_only=True)
                matched_rows(source, child)
                x = child['members_model']; physical = x*scale+offset
                vals = observer.score_tensors(x[:,:,query])
                row = rows[arm]
                values = {key: float(vals[key].mean()) for key in ('state','scale','shape')}
                row['per_condition_scores'].append(values)
                for key in values: row['scores'][key] += values[key]
                risks = native_risks(physical, truth_physical, mass, model_family='ffm')
                for key in ('member_mse','ensemble_mean_mse','signed_mean_risk','unbiased_member_variance'):
                    row['risk'][key] = row['risk'].get(key, torch.zeros_like(risks[key], dtype=torch.float64)) + risks[key].double()
                row['per_condition_member_mse'].append(risks['per_condition_member_mse'].reshape(-1).tolist())
                sensor, counts = sensor_mse(x, source, len(ctx['source']['field_order']))
                sensor = sensor*scale.double().square()
                row['sensor'] = sensor if row['sensor'] is None else row['sensor'] + sensor
                row['sensor_counts'] = counts if row['sensor_counts'] is None else row['sensor_counts'] + counts
                if endpoint:
                    row['event'].append(float(score_event(event, physical, truth_physical, mass)[0].mean()))
                    members = torch.stack([regional_covariance(m, mass, task, tuple(ctx['source']['field_order'])) for m in physical])
                    target = regional_covariance(truth_physical, mass, task, tuple(ctx['source']['field_order']))
                    row['regional'].append(float(scalar_task_score(members, target, model_family='ffm').mean()))
        count = len(ctx['source']['sample_ids'])
        norm = contract['normalizers']
        base_risk = baseline['risk']
        source_sensor = [0. if v is None else v for v in baseline['sensor_mse']]
        result_rows = {}
        for arm, row in rows.items():
            scores = {k:v/count for k,v in row['scores'].items()}
            risk = {k:(v/count).tolist() for k,v in row['risk'].items()}
            sensor = (row['sensor']/count).tolist()
            fidelity = field_sensor_eligibility(base_risk, risk, model_family='ffm',
                absolute_mean_allowance=contract['mean_allowance_physical'], source_sensor_mse=source_sensor,
                child_sensor_mse=sensor, sensor_equality_tolerance=contract['sensor_equality_tolerance'])
            differences = {k:(scores[k]-baseline['scores'][k])/norm[k] for k in scores}
            result = dict(scores=scores, normalized_core_differences=differences, risk=risk,
                sensor_mse=[v if n else None for v,n in zip(sensor,row['sensor_counts'].tolist(),strict=True)],
                sensor_observation_counts=row['sensor_counts'].tolist(), fidelity=fidelity,
                per_condition_scores=row['per_condition_scores'], per_condition_member_mse=row['per_condition_member_mse'])
            if endpoint:
                event_score = sum(row['event'])/count
                regional_score = sum(row['regional'])/count
                regional_source = ctx['baseline']['reserved_task']['source_score']
                task_norm = ctx['freeze']['task_units']['ffm']['normalizer']
                result['reused_tasks'] = dict(event_score=event_score,
                    event_normalized_difference=(event_score-baseline['scores']['event'])/norm['event'],
                    regional_score=regional_score, regional_source_score=regional_source,
                    regional_normalized_improvement=(regional_source-regional_score)/task_norm,
                    event_per_condition=row['event'], regional_per_condition=row['regional'])
            result_rows[arm] = result
        source_view = dict(scores={k:baseline['scores'][k] for k in ('state','scale','shape')},
            risk=baseline['risk'], sensor_mse=baseline['sensor_mse'],
            per_condition_scores=baseline['per_condition_scores'], per_condition_member_mse=baseline['per_condition_member_mse'])
        if endpoint:
            source_view['reused_tasks'] = dict(event_score=baseline['scores']['event'],
                regional_score=ctx['baseline']['reserved_task']['source_score'])
        result = dict(schema='car_round3_matched_validation_v1', epoch=args.epoch, split='validation',
            population=ctx['panel']['population'],
            selection_use='previously accessed same-trajectory development validation',
            sample_ids=ctx['source']['sample_ids'], field_order=ctx['source']['field_order'], members=4,
            checkpoint_sha256={arm:receipts[arm]['checkpoint_sha256'] for arm in ARMS},
            source_checkpoint_sha256=ctx['source']['source_sha256'],
            bank_receipt_sha256={arm:hash_file(banks[arm]/'receipt.json') for arm in ARMS},
            source_bank_sha256=hash_file(ctx['source_receipt']), panel_sha256=hash_file(ctx['panel_path']),
            source_baseline_sha256=hash_file(ctx['baseline_path']), contract_sha256=hash_file(verify(ctx['artifact']['contract'])),
            assessment_code_sha256=hash_file(Path(__file__)), prepared_sha256=hash_file(ctx['prepared_path']),
            core_sha256=hash_file(verify(ctx['artifact']['core'])), freeze_sha256=hash_file(ctx['freeze_path']),
            source=source_view, candidates=result_rows, normalizers={k:norm[k] for k in ('state','scale','shape')},
            task_scope='reused development tasks at endpoint' if endpoint else 'both tasks excluded',
            wall_seconds=time.time()-started, cpu_seconds=time.process_time()-cpu)
        if endpoint:
            result['task_normalizers'] = {'event': norm['event'],
                'regional': ctx['freeze']['task_units']['ffm']['normalizer']}
            finalize_decision(result)
        (output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        status['status']='completed'
    except BaseException as error:
        status.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if dataset is not None: dataset.close()
        status.update(end_unix=time.time(), wall_seconds=time.time()-started, cpu_seconds=time.process_time()-cpu)
        (output/'receipt.json').write_text(json.dumps(status,indent=2,allow_nan=False)+'\n')


def score_early_endpoint_tasks(args, ctx):
    """Score task functions only after an explicit terminal epoch-100 decision."""
    import torch
    from phycoflow_reconstruction.evaluation.round2_assessment import scalar_task_score
    from phycoflow_reconstruction.evaluation.round2_task import regional_covariance
    from scripts.posttraining.freeze_car_event import FrozenRedesignedEvent

    decision_path = args.decision.resolve(strict=True)
    if ctx['campaign'] not in decision_path.parents:
        raise ValueError('terminal decision must be campaign-owned')
    decision = load_json(decision_path)
    review_path = args.review.resolve(strict=True)
    review = load_json(review_path)
    banks = {ARMS[0]: args.raw_bank.resolve(strict=True), ARMS[1]: args.scaled_bank.resolve(strict=True)}
    receipts = {arm: check_bank(bank, ctx, 100, arm) for arm, bank in banks.items()}
    hashes = {arm: hash_file(banks[arm] / 'receipt.json') for arm in ARMS}
    if (decision.get('schema') != 'car_round3_terminal_decision_v1'
            or decision.get('decision') != 'stop_at_epoch_100' or decision.get('epoch') != 100
            or decision.get('authorize_endpoint_tasks') is not True
            or decision.get('review_sha256') != hash_file(review_path)
            or decision.get('bank_receipt_sha256') != hashes):
        raise ValueError('explicit terminal epoch-100 task decision with matched review and banks required')
    if (review.get('schema') != 'car_round3_matched_validation_v1' or review.get('epoch') != 100
            or review.get('task_scope') != 'both tasks excluded' or review.get('bank_receipt_sha256') != hashes
            or review.get('source_bank_sha256') != hash_file(ctx['source_receipt'])
            or review.get('panel_sha256') != hash_file(ctx['panel_path'])):
        raise ValueError('review does not bind the same task-free matched banks')
    output = args.output.resolve()
    if ctx['campaign'] / 'assessment' not in output.parents:
        raise ValueError('score output must be in Round 3 assessment storage')
    output.mkdir(parents=True, exist_ok=False)
    started, cpu = time.time(), time.process_time()
    status = dict(schema='car_round3_early_endpoint_task_job_v1', status='failed', start_unix=started)
    try:
        event = FrozenRedesignedEvent(load_json(verify(ctx['freeze']['event'])))
        task = load_json(verify(ctx['freeze']['reserved_task']))
        per_arm = {arm: {'event': [], 'regional': []} for arm in ARMS}
        for i, sample_id in enumerate(ctx['source']['sample_ids']):
            source_path = ctx['source_dir'] / f'{i:04d}.pt'
            if hash_file(source_path) != ctx['source']['files_sha256'][i]: raise ValueError('SOURCE row changed')
            source = torch.load(source_path, map_location='cpu', weights_only=True)
            if source['sample_id'] != sample_id: raise ValueError('SOURCE order differs')
            mass, scale, offset = source['native_masses'].double(), source['normalizer_scale'], source['normalizer_offset']
            truth = source['truth_model']*scale+offset
            target = regional_covariance(truth, mass, task, tuple(ctx['source']['field_order']))
            for arm, bank in banks.items():
                path = bank / f'{i:04d}.pt'
                if hash_file(path) != receipts[arm]['files_sha256'][i]: raise ValueError('candidate row changed')
                child = torch.load(path, map_location='cpu', weights_only=True)
                matched_rows(source, child)
                physical = child['members_model']*scale+offset
                per_arm[arm]['event'].append(float(score_event(event, physical, truth, mass)[0].mean()))
                members = torch.stack([regional_covariance(m, mass, task, tuple(ctx['source']['field_order'])) for m in physical])
                per_arm[arm]['regional'].append(float(scalar_task_score(members, target, model_family='ffm').mean()))
        result = copy.deepcopy(review)
        baseline = next(row for row in ctx['baseline']['rows'] if row['alpha'] == 0.)
        regional_source = ctx['baseline']['reserved_task']['source_score']
        event_norm = ctx['baseline']['normalizers']['event']
        regional_norm = ctx['freeze']['task_units']['ffm']['normalizer']
        count = len(ctx['source']['sample_ids'])
        for arm, values in per_arm.items():
            event_score = sum(values['event']) / count
            regional_score = sum(values['regional']) / count
            result['candidates'][arm]['reused_tasks'] = {
                'event_score': event_score,
                'event_normalized_difference': (event_score-baseline['scores']['event'])/event_norm,
                'regional_score': regional_score, 'regional_source_score': regional_source,
                'regional_normalized_improvement': (regional_source-regional_score)/regional_norm,
                'event_per_condition': values['event'], 'regional_per_condition': values['regional']}
        result['source']['reused_tasks'] = {'event_score':baseline['scores']['event'], 'regional_score':regional_source}
        result['task_scope'] = 'reused development tasks at explicitly decided early endpoint'
        result['task_normalizers'] = {'event':event_norm, 'regional':regional_norm}
        result['terminal_decision_sha256'] = hash_file(decision_path)
        result['task_free_review_sha256'] = hash_file(review_path)
        result['assessment_code_sha256'] = hash_file(Path(__file__))
        result['task_wall_seconds'] = time.time()-started
        result['task_cpu_seconds'] = time.process_time()-cpu
        finalize_decision(result)
        (output/'assessment.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        status['status'] = 'completed'
    except BaseException as error:
        status.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        status.update(end_unix=time.time(), wall_seconds=time.time()-started, cpu_seconds=time.process_time()-cpu)
        (output/'receipt.json').write_text(json.dumps(status, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    subs = parser.add_subparsers(dest='command', required=True)
    gen = subs.add_parser('generate'); gen.add_argument('--arm', choices=ARMS, required=True)
    gen.add_argument('--epoch', type=int, choices=(100,500), required=True); gen.add_argument('--run', type=Path, required=True)
    gen.add_argument('--device', required=True); gen.add_argument('--output', type=Path, required=True)
    gen.add_argument('--max-seconds', type=float, default=900.); gen.add_argument('--save-margin-seconds', type=float, default=60.)
    scoring = subs.add_parser('score'); scoring.add_argument('--epoch', type=int, choices=(100,500), required=True)
    scoring.add_argument('--raw-bank', type=Path, required=True); scoring.add_argument('--scaled-bank', type=Path, required=True)
    scoring.add_argument('--output', type=Path, required=True)
    early = subs.add_parser('early-endpoint-tasks')
    early.add_argument('--decision', type=Path, required=True); early.add_argument('--review', type=Path, required=True)
    early.add_argument('--raw-bank', type=Path, required=True); early.add_argument('--scaled-bank', type=Path, required=True)
    early.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); ctx = context(args.campaign)
    {'generate':generate_with_lease, 'score':score, 'early-endpoint-tasks':score_early_endpoint_tasks}[args.command](args, ctx)


if __name__ == '__main__':
    main()
