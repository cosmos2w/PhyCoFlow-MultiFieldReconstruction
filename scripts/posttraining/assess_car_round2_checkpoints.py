#!/usr/bin/env python3
"""Matched native Round 2 checkpoint generation and TRAIN/VALIDATION assessment.

TEST and EDM are deliberately absent. Native endpoint rows are immutable and
share observation IDs and member streams. Formal and TRAIN-nomination freezes
are required before any VALIDATION data access.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('CAR_REPO_ROOT', CODE_ROOT)).resolve()
sys.path.insert(0, str(CODE_ROOT / 'src'))
sys.path.insert(0, str(CODE_ROOT))
if os.environ.get('KEOPS_CACHE_FOLDER') and os.environ.get('CUDA_VISIBLE_DEVICES'):
    build = '_'.join(platform.uname()[:3]) + f'_p{sys.version.split()[0]}'
    (Path(os.environ['KEOPS_CACHE_FOLDER']) / f"{build}_CUDA_VISIBLE_DEVICES_{os.environ['CUDA_VISIBLE_DEVICES']}").mkdir(parents=True, exist_ok=True)
EPOCHS = (0, 100, 250, 500, 750, 1000)
ARMS = ('FFM_Selected_Bank_Uniform_FullModel', 'FFM_Selected_Bank_Residual_Repair',
        'Senseiver_Selected_Bank_Uniform_FullModel', 'Senseiver_Selected_Bank_Residual_Repair')


def hash_file(path):
    from phycoflow_reconstruction.evaluation.adaptive_campaign import digest
    return digest(Path(path))


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def same_scientific_config(left, right):
    a, b = copy.deepcopy(left), copy.deepcopy(right)
    for config in (a, b):
        config.setdefault('runtime', {}).pop('device', None)
    return a == b


def completed_frozen_mature_runs(campaign):
    """Return terminal, receipt-bound frozen mature runs, without reading active runs."""
    import yaml

    formal = json.loads((campaign / 'configs/formal_freeze.json').read_text())
    manifest = formal.get('manifest', {})
    if (formal.get('schema') != 'car_round2_freeze_receipt_v1'
            or manifest.get('schema') != 'car_round2_freeze_v1'
            or manifest.get('status') != 'frozen' or set(manifest.get('arms', {})) != set(ARMS)):
        raise ValueError('final assessment requires the four-arm formal freeze')
    completed = {arm: set() for arm in ARMS}
    for completion_path in (campaign / 'operations/completions').glob('*.json'):
        completion = json.loads(completion_path.read_text())
        if completion.get('status') != 'completed' or completion.get('global_step') != 12000:
            continue
        token = completion.get('token')
        if not isinstance(token, str) or completion_path.stem != token:
            continue
        launch_path = campaign / 'operations/launches' / f'{token}.json'
        if not launch_path.is_file():
            continue
        launch = json.loads(launch_path.read_text())
        arm = launch.get('arm')
        if (launch.get('schema') != 'car_round2_launch_v1' or launch.get('token') != token
                or arm not in ARMS or launch.get('pause_at_epoch') is not None):
            continue
        binding = manifest['arms'][arm]['config']
        config_path = Path(binding['path'])
        if (not config_path.is_file() or hash_file(config_path) != binding['sha256']
                or launch.get('config_sha256') != binding['sha256']):
            continue
        run_value = completion.get('run_dir')
        if not isinstance(run_value, str):
            continue
        run = Path(run_value).resolve()
        formal_arm = (campaign / 'runs/formal' / arm).resolve()
        if formal_arm not in run.parents:
            continue
        status_path, resolved_path = run / 'status.json', run / 'resolved_config.yaml'
        if not status_path.is_file() or not resolved_path.is_file():
            continue
        status = json.loads(status_path.read_text())
        if (status.get('status') != 'completed' or status.get('global_step') != 12000
                or not same_scientific_config(yaml.safe_load(resolved_path.read_text()),
                                              yaml.safe_load(config_path.read_text()))):
            continue
        completed[arm].add(run)
    return completed


def assessment_budget_limit(campaign, final_assessment):
    """Release the closeout reserve only after all frozen mature arms finish."""
    if not final_assessment:
        return 124 * 3600
    completed = completed_frozen_mature_runs(campaign)
    missing = {arm for arm, runs in completed.items() if not runs}
    if missing:
        raise ValueError(f'final assessment requires all frozen mature arms completed at 12000 updates; missing {sorted(missing)}')
    return 130 * 3600


def admit_assessment_budget(state, *, gpu_uuid, max_seconds, cap, spent_seconds):
    if any(item['gpu_uuid'] == gpu_uuid for item in state['active'].values()):
        raise ValueError('device leased')
    if spent_seconds + max_seconds >= cap:
        raise ValueError('assessment job exceeds remaining budget')


def panel_contract(panel, split):
    if (panel.get('schema') != 'car_round2_panel_v1' or panel.get('split') != split
            or split not in ('train', 'validation') or panel.get('sensor_count') != 256
            or panel.get('members') != {'ffm': 4, 'senseiver': 1}
            or panel.get('native_points') != 40300 or panel.get('observer_points') != 4096):
        raise ValueError('invalid fixed T256, J4/J1, Q4096/native panel contract')
    ids, indices, n = panel['sample_ids'], panel['sample_indices'], panel['population']
    if (len(ids) != max(1, math.ceil(.1*n)) or len(ids) != len(indices)
            or len(set(ids)) != len(ids) or len(set(indices)) != len(indices)
            or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= n for i in indices)):
        raise ValueError('panel must have unique, reproducible 10% sample IDs and indices')
    return ids


def member_seed(index, member):
    return 9_100_003 + 11 * int(index) + int(member)


def score_event(event, members, truth, mass):
    """Reused event: J4 fair area CRPS or genuine J1 absolute area error."""
    import torch
    if members.ndim != 4 or len(members) not in (1, 4) or members.shape[1:] != truth.shape:
        raise ValueError('event requires genuine J1 or J4 matched members')
    m = torch.as_tensor(mass, device=members.device, dtype=members.dtype)
    m = m / m.sum()
    area = (event.masks(members)[-1] * m).sum(-1)
    target = (event.masks(truth)[-1] * m).sum(-1)
    score = (area - target).abs().mean(0)
    if len(members) == 4:
        score -= sum((area[i]-area[j]).abs() for i in range(4) for j in range(i+1, 4)) / 12
    return score, area, target


def sensor_mse(members, row, field_count):
    import torch
    ids, fields, observed = row['obs_indices'].long()[0], row['obs_fields'].long()[0], row['obs_values'][0, :, 0]
    errors = members[:, 0, ids, fields] - observed
    values = torch.zeros(field_count, dtype=torch.float64)
    for field in range(field_count):
        selected = fields == field
        if selected.any():
            values[field] = errors[:, selected].double().square().mean()
    return values


def matched_rows(source, child):
    import torch
    for name in ('sample_id', 'sample_index', 'member_seeds'):
        if source[name] != child[name]:
            raise ValueError(f'matched endpoint {name} differs')
    for name in ('truth_model', 'coordinates_model', 'coordinates_raw', 'native_masses',
                 'obs_indices', 'obs_values', 'obs_fields', 'normalizer_offset', 'normalizer_scale'):
        if not torch.equal(source[name], child[name]):
            raise ValueError(f'matched endpoint {name} differs')


def build_panel(args, campaign):
    import torch
    import yaml

    from phycoflow_reconstruction.data.factory import open_field_dataset
    from phycoflow_reconstruction.data.manifest import dataset_fingerprint
    from phycoflow_reconstruction.data.training_batches import fixed_query_indices
    from phycoflow_reconstruction.evaluation.adaptive_campaign import immutable_json
    started, cpu_started = time.time(), time.process_time()
    cfg = yaml.safe_load((args.core_run / 'resolved_config.yaml').read_text())
    train = open_field_dataset(cfg['dataset'], split='train')
    dataset = train if args.split == 'train' else open_field_dataset(cfg['dataset'], split='validation', normalizer=train.normalizer)
    try:
        if int(dataset.data_spec.logical_shape[0]) * int(dataset.data_spec.logical_shape[1]) != 40300:
            raise ValueError('native support must contain 40300 points')
        if args.split == 'train':
            panel = torch.load(args.core_run / 'artifacts/source_panel.pt', map_location='cpu', weights_only=True)
            indices = list(panel['sample_indices'])
            expected = list(panel['sample_ids'])
        else:
            indices = list(range(max(1, math.ceil(.1*len(dataset)))))
            expected = None
        ids = [f'{dataset.trajectory_ids[dataset._items[i][0]]}:{dataset._items[i][1]}' for i in indices]
        if expected is not None and ids != expected:
            raise ValueError('prepared SOURCE panel IDs differ from live TRAIN dataset')
        query = fixed_query_indices(40300, 4096, seed=int(cfg['coherence']['compute_budget']['query_seed']))
        doc = {'schema':'car_round2_panel_v1', 'split':args.split, 'population':len(dataset),
               'sample_indices':indices, 'sample_ids':ids, 'sensor_count':256,
               'observation_seed_rule':'2047+split_sample_index',
               'member_seed_rule':'9100003+11*split_sample_index+member',
               'members':{'ffm':4,'senseiver':1}, 'native_points':40300, 'observer_points':4096,
               'query_indices':query.tolist(), 'query_seed':int(cfg['coherence']['compute_budget']['query_seed']),
               'normalizer_sha256':dataset.normalizer.digest(), 'dataset_fingerprint':dataset_fingerprint(dataset.path),
               'field_order':list(dataset.field_names), 'core_preparation_sha256':hash_file(args.core_run / 'artifacts/scalable_coherence.pt'),
               'preparation_timer':{'start_unix':started, 'end_unix':time.time(), 'cpu_seconds':time.process_time()-cpu_started}}
        panel_contract(doc, args.split)
        output = campaign / 'configs/panels' / f'{args.split}.json'
        immutable_json(output, doc)
        print(output)
    finally:
        if dataset is not train:
            dataset.close()
        train.close()


def validation_freeze(campaign, family, arm=None, epoch=None):
    path = campaign / 'assessment/train' / f'method_priority_{family}.json'
    freeze = json.loads(path.read_text())
    formal = campaign / 'configs/formal_freeze.json'
    if (freeze.get('schema') != 'car_round2_assessment_freeze_v1' or freeze.get('formal_freeze_sha256') != hash_file(formal)
            or freeze.get('sha256') != content_hash({k:v for k,v in freeze.items() if k != 'sha256'})):
        raise ValueError('VALIDATION requires a matching complete TRAIN assessment freeze')
    expected = [a for a in ARMS if a.lower().startswith(family)]
    if set(freeze.get('arms', {})) != set(expected):
        raise ValueError('VALIDATION requires both complete TRAIN arm trajectories')
    for name, record in freeze['arms'].items():
        if set(record) != {str(x) for x in EPOCHS}:
            raise ValueError('all six TRAIN checkpoint nominations required')
        for e, item in record.items():
            if hash_file(item['nomination']) != item['sha256']:
                raise ValueError(f'frozen TRAIN nomination changed: {name} epoch {e}')
    if arm is not None:
        return freeze, json.loads(Path(freeze['arms'][arm][str(epoch)]['nomination']).read_text())
    return freeze, None


def global_train_choice_barrier(campaign):
    """Verify both families' frozen nominations and method priorities before held-out work."""
    for family in ('ffm', 'senseiver'):
        freeze, _ = validation_freeze(campaign, family)
        priority = freeze.get('method_priority')
        if (not isinstance(priority, list) or len(set(priority)) != len(priority)
                or any(arm not in freeze['arms'] for arm in priority)):
            raise ValueError(f'{family} requires a valid frozen TRAIN method priority')


def validate_reserved_freeze(campaign, family):
    """Recheck the post-nomination TRAIN SOURCE scale before any held-out access."""
    freeze, _ = validation_freeze(campaign, family)
    path = campaign / 'assessment/train' / f'reserved_normalizer_{family}.json'
    doc = json.loads(path.read_text())
    formal = json.loads((campaign / 'configs/formal_freeze.json').read_text())['manifest']
    contract_path = Path(formal['artifacts'][f'assessment_contract_{family}']['path'])
    contract_hash = hash_file(contract_path)
    contract = json.loads(contract_path.read_text())
    bank = Path(doc['source_bank_receipt'])
    task = Path(formal['artifacts']['reserved_task']['path'])
    score = doc.get('source_score')
    rule = contract.get('reserved_task_normalization_rule', {})
    fallback = contract.get('reserved_task_normalizer')
    threshold = rule.get('near_zero_threshold')
    if (doc.get('schema') != 'car_round2_reserved_source_normalizer_v1'
            or doc.get('family') != family or doc.get('split') != 'train'
            or doc.get('sha256') != content_hash({k:v for k,v in doc.items() if k != 'sha256'})
            or doc.get('train_freeze_sha256') != hash_file(campaign / 'assessment/train' / f'method_priority_{family}.json')
            or doc.get('contract_sha256') != contract_hash
            or contract_hash != formal['artifacts'][f'assessment_contract_{family}']['sha256']
            or doc.get('task_sha256') != hash_file(task) or doc.get('task_sha256') != contract['reserved_task_sha256']
            or doc.get('task_sha256') != formal['artifacts']['reserved_task']['sha256']
            or doc.get('source_bank_sha256') != hash_file(bank) or doc.get('source_bank_sha256') != contract['source_bank_sha256']
            or not isinstance(score, (int, float)) or not math.isfinite(score)
            or rule.get('primary') != 'final_train_source_task_score'
            or rule.get('fallback') != 'train_truth_absolute_covariance'
            or not isinstance(fallback, (int, float)) or not math.isfinite(fallback) or fallback <= 0
            or threshold != 1e-12*fallback):
        raise ValueError('reserved SOURCE normalizer evidence does not match frozen TRAIN selection and contract')
    expected = fallback if score <= threshold else score
    if doc.get('normalizer') != expected or doc.get('used_truth_fallback') != (score <= threshold):
        raise ValueError('reserved normalizer does not follow the predeclared fallback rule')
    for records in freeze['arms'].values():
        nomination = json.loads(Path(records['1000']['nomination']).read_text())
        if (nomination['source_bank_sha256'] != doc['source_bank_sha256']
                or nomination['contract_sha256'] != contract_hash):
            raise ValueError('reserved SOURCE bank does not match both final TRAIN nominations')
    return doc


def freeze_reserved_normalizer(args, campaign):
    """Score only the already saved TRAIN SOURCE bank after nominations are frozen."""
    import torch

    from phycoflow_reconstruction.evaluation.adaptive_campaign import immutable_json
    from phycoflow_reconstruction.evaluation.round2_assessment import scalar_task_score
    from phycoflow_reconstruction.evaluation.round2_task import regional_covariance

    global_train_choice_barrier(campaign)
    started, cpu = time.time(), time.process_time()
    family = args.family
    freeze, _ = validation_freeze(campaign, family)
    formal = json.loads((campaign / 'configs/formal_freeze.json').read_text())['manifest']
    binding = formal['artifacts'][f'assessment_contract_{family}']
    contract_path = Path(binding['path'])
    contract = json.loads(contract_path.read_text())
    bank_path = args.source_bank.resolve()
    receipt_path = bank_path / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    panel_path = campaign / 'configs/panels/train.json'
    panel = json.loads(panel_path.read_text())
    ids = panel_contract(panel, 'train')
    if (contract.get('schema') != 'car_round2_assessment_contract_v2'
            or hash_file(contract_path) != binding['sha256']
            or hash_file(args.reserved_task) != contract['reserved_task_sha256']
            or hash_file(receipt_path) != contract['source_bank_sha256']
            or receipt.get('schema') != 'car_round2_native_bank_v1'
            or receipt.get('status') != 'completed' or receipt.get('split') != 'train'
            or receipt.get('kind') != 'source' or receipt.get('family') != family
            or receipt.get('source_sha256') != contract['source_checkpoint_sha256']
            or receipt.get('panel_sha256') != hash_file(panel_path) or receipt.get('sample_ids') != ids):
        raise ValueError('frozen SOURCE bank, task, panel and contract must match')
    for records in freeze['arms'].values():
        nomination = json.loads(Path(records['1000']['nomination']).read_text())
        if nomination.get('source_bank_sha256') != hash_file(receipt_path) or nomination.get('contract_sha256') != binding['sha256']:
            raise ValueError('both final TRAIN nominations must bind this SOURCE and contract')
    manifest = json.loads(args.reserved_task.read_text())
    scores = []
    for i, sample_id in enumerate(ids):
        path = bank_path / f'{i:04d}.pt'
        if hash_file(path) != receipt['files_sha256'][i]:
            raise ValueError('SOURCE bank row hash differs')
        row = torch.load(path, map_location='cpu', weights_only=True)
        if row['sample_id'] != sample_id:
            raise ValueError('SOURCE bank row order differs')
        mass = row['native_masses'].double()
        offset, scale = row['normalizer_offset'], row['normalizer_scale']
        members = row['members_model']*scale+offset
        truth = row['truth_model']*scale+offset
        member_values = torch.stack([regional_covariance(member, mass, manifest, tuple(receipt['field_order'])) for member in members])
        target = regional_covariance(truth, mass, manifest, tuple(receipt['field_order']))
        scores.append(float(scalar_task_score(member_values, target, model_family=family).mean()))
    score = sum(scores)/len(scores)
    fallback = contract['reserved_task_normalizer']
    rule = contract['reserved_task_normalization_rule']
    threshold = rule['near_zero_threshold']
    if (not math.isfinite(score) or not math.isfinite(fallback) or fallback <= 0
            or rule.get('primary') != 'final_train_source_task_score'
            or rule.get('fallback') != 'train_truth_absolute_covariance' or threshold != 1e-12*fallback):
        raise ValueError('invalid predeclared SOURCE normalization rule')
    result = {'schema':'car_round2_reserved_source_normalizer_v1', 'split':'train', 'family':family,
              'source_score':score, 'normalizer':fallback if score <= threshold else score,
              'used_truth_fallback':score <= threshold, 'near_zero_threshold':threshold,
              'train_freeze_sha256':hash_file(campaign / 'assessment/train' / f'method_priority_{family}.json'),
              'contract_sha256':binding['sha256'], 'task_sha256':hash_file(args.reserved_task),
              'source_bank_receipt':str(receipt_path), 'source_bank_sha256':hash_file(receipt_path),
              'sample_ids':ids, 'start_unix':started, 'end_unix':time.time(), 'cpu_seconds':time.process_time()-cpu}
    result['sha256'] = content_hash(result)
    path = campaign / 'assessment/train' / f'reserved_normalizer_{family}.json'
    immutable_json(path, result)
    validate_reserved_freeze(campaign, family)
    print(path)


def prepare_contract(args, campaign):
    import torch
    import yaml

    from phycoflow_reconstruction.coherence import build_enabled_families
    from phycoflow_reconstruction.data.factory import open_field_dataset
    from phycoflow_reconstruction.evaluation.adaptive_campaign import immutable_json
    from phycoflow_reconstruction.evaluation.frozen_coherence import FrozenCoherenceObserver
    from scripts.posttraining.freeze_car_event import FrozenRedesignedEvent
    start, cpu_start = time.time(), time.process_time()
    if (campaign / 'configs/formal_freeze.json').exists():
        raise ValueError('assessment contract must be frozen before formal learning')
    receipt = json.loads((args.source_bank/'receipt.json').read_text())
    if receipt.get('schema') != 'car_round2_native_bank_v1' or receipt.get('status') != 'completed' or receipt.get('split') != 'train' or receipt.get('kind') != 'source':
        raise ValueError('completed TRAIN SOURCE native bank required')
    family = receipt['family']
    panel_path = campaign / 'configs/panels/train.json'
    panel = json.loads(panel_path.read_text()); ids = panel_contract(panel,'train')
    if receipt['panel_sha256'] != hash_file(panel_path) or receipt['sample_ids'] != ids:
        raise ValueError('SOURCE bank must bind frozen TRAIN panel')
    cfg = yaml.safe_load((args.core_run/'resolved_config.yaml').read_text())
    dataset = open_field_dataset(cfg['dataset'],split='train')
    try:
        families = build_enabled_families(cfg['coherence'], dataset.data_spec, dataset.normalizer)
        manifest = json.loads((args.core_run/'run_manifest.json').read_text())
        for name, obj in families.items():
            path = args.core_run/'artifacts'/f'{name}_family.pt'
            if hash_file(path) != manifest['coherence_family_state_sha256s'][name]: raise ValueError('observer family hash differs')
            obj.load_state_artifact(torch.load(path, map_location='cpu',weights_only=True))
        core_path = args.core_run/'artifacts/scalable_coherence.pt'
        core = torch.load(core_path,map_location='cpu',weights_only=True)
        if core.get('adaptive_bank') is not None or core['provenance']['source_checkpoint_sha256'] != receipt['source_sha256']:
            raise ValueError('SOURCE-bound core-only observer required')
        event = FrozenRedesignedEvent(json.loads(args.event.read_text()))
        frozen_task = json.loads(args.reserved_task.read_text())
        if frozen_task.get('schema') != 'car_round2_reserved_task_v1': raise ValueError('frozen reserved task required')
        if not math.isfinite(args.reserved_task_normalizer) or args.reserved_task_normalizer <= 0:
            raise ValueError('positive TRAIN-truth reserved task normalization required')
        norm_evidence = json.loads(args.reserved_normalizer_evidence.read_text())
        if (norm_evidence.get('schema') != 'car_round2_reserved_normalizer_v1'
                or norm_evidence.get('split') != 'train' or norm_evidence.get('reserved_model_scores_used') is not False
                or norm_evidence.get('task_sha256') != hash_file(args.reserved_task)
                or norm_evidence.get('panel_sha256') != hash_file(panel_path)
                or norm_evidence.get('value') != args.reserved_task_normalizer):
            raise ValueError('reserved normalization must match its frozen TRAIN-truth evidence')
        tolerance = [float(x) for x in args.sensor_equality_tolerance]
        if len(tolerance) != len(dataset.field_names) or any(not math.isfinite(x) or x < 0 for x in tolerance):
            raise ValueError('one predeclared nonnegative physical sensor tolerance per field required')
        query = torch.tensor(panel['query_indices'],dtype=torch.long)
        if query.numel() != 4096 or query.unique().numel() != 4096:
            raise ValueError('fixed Q4096 support required')
        totals = {k:0. for k in ('state','scale','shape','event')}
        for i, sample_id in enumerate(ids):
            path = args.source_bank/f'{i:04d}.pt'
            if hash_file(path) != receipt['files_sha256'][i]: raise ValueError('SOURCE bank row hash differs')
            row = torch.load(path,map_location='cpu',weights_only=True)
            if row['sample_id'] != sample_id: raise ValueError('SOURCE panel order differs')
            x,z,mass = row['members_model'],row['truth_model'],row['native_masses'].double()
            offset,scale = row['normalizer_offset'],row['normalizer_scale']
            observer = FrozenCoherenceObserver(core,families,expected_provenance=core['provenance'],reference=z[:,query],
                coordinates=row['coordinates_model'][:,query],sample_ids=[sample_id],member_weights=mass[query].unsqueeze(0),
                reference_weights=mass[query].unsqueeze(0),mode='conditional' if family=='ffm' else 'deterministic',
                query_seed=args.shape_query_seed,complete_scale=True)
            values = observer.score_tensors(x[:,:,query])
            for key in ('state','scale','shape'): totals[key] += float(values[key].mean())
            totals['event'] += float(score_event(event,x*scale+offset,z*scale+offset,mass)[0].mean())
        normalizers = {k:v/len(ids) for k,v in totals.items()}
        if any(not math.isfinite(v) or v <= 0 for v in normalizers.values()):
            raise ValueError('nonpositive SOURCE score requires a separately frozen absolute normalization')
        fidelity = json.loads((args.core_run/'artifacts/fidelity_budget_reference.json').read_text())
        if fidelity.get('binding',{}).get('source_checkpoint_sha256') != receipt['source_sha256']:
            raise ValueError('fidelity reference differs from SOURCE')
        if family == 'ffm':
            ref = fidelity['reference']; allowance = torch.tensor(ref['absolute_allowance'],dtype=torch.float64)
            member = torch.tensor(fidelity['risk_reference_panel']['source_member_mse'],dtype=torch.float64)
            if not torch.allclose(allowance,.1*member,rtol=1e-9,atol=0):
                raise ValueError('FFM allowance must equal 0.1 times source member risk')
            allowance = allowance*dataset.normalizer.scale.double().square()
        else:
            allowance = torch.zeros(len(dataset.field_names),dtype=torch.float64)
        result = {'schema':'car_round2_assessment_contract_v2','source_checkpoint_sha256':receipt['source_sha256'],
                  'family':family,'normalizers':normalizers,'mean_allowance_physical':allowance.tolist(),
                  'sensor_equality_tolerance':tolerance,'reserved_task_normalizer':args.reserved_task_normalizer,
                  'reserved_task_normalization_rule':{'primary':'final_train_source_task_score',
                      'fallback':'train_truth_absolute_covariance', 'near_zero_threshold':1e-12*args.reserved_task_normalizer},
                  'reserved_task_normalizer_evidence_sha256':hash_file(args.reserved_normalizer_evidence),
                  'shape_query_seed':args.shape_query_seed,'core_sha256':hash_file(core_path),
                  'event_sha256':hash_file(args.event),'reserved_task_sha256':hash_file(args.reserved_task),
                  'source_bank_sha256':hash_file(args.source_bank/'receipt.json'),
                  'fidelity_reference_sha256':hash_file(args.core_run/'artifacts/fidelity_budget_reference.json'),
                  'train_panel_sha256':hash_file(panel_path),'sample_ids':ids,
                  'reserved_model_scores_used':False,'start_unix':start,'end_unix':time.time(),
                  'cpu_seconds':time.process_time()-cpu_start}
        if campaign/'artifacts' not in args.output.resolve().parents: raise ValueError('contract must reside in campaign artifacts')
        immutable_json(args.output,result)
    finally:
        dataset.close()
    print(args.output)


def reserved_normalizer(args, campaign):
    import yaml

    from phycoflow_reconstruction.data.factory import open_field_dataset
    from phycoflow_reconstruction.evaluation.adaptive_campaign import immutable_json
    from phycoflow_reconstruction.evaluation.round2_task import regional_covariance
    if (campaign/'configs/formal_freeze.json').exists():
        raise ValueError('reserved task normalization must be frozen before formal training')
    started, cpu = time.time(), time.process_time()
    panel_path = campaign/'configs/panels/train.json'
    panel = json.loads(panel_path.read_text()); ids = panel_contract(panel,'train')
    manifest = json.loads(args.reserved_task.read_text())
    cfg = yaml.safe_load((args.core_run/'resolved_config.yaml').read_text())
    dataset = open_field_dataset(cfg['dataset'],split='train')
    try:
        if len(dataset) != panel['population'] or dataset.normalizer.digest() != panel['normalizer_sha256']:
            raise ValueError('TRAIN truth data differs from frozen panel')
        values = []
        for index, sample_id in zip(panel['sample_indices'], ids):
            if f'{dataset.trajectory_ids[dataset._items[index][0]]}:{dataset._items[index][1]}' != sample_id:
                raise ValueError('TRAIN truth identity differs')
            physical = dataset.normalizer.decode(dataset[index].values).unsqueeze(0)
            values.append(float(regional_covariance(physical,dataset.physical_weights,manifest,tuple(dataset.field_names))[0]))
        value = sum(abs(x) for x in values)/len(values)
        if not math.isfinite(value) or value <= 0:
            raise ValueError('degenerate TRAIN truth normalization requires explicit preformal disposition')
        if campaign/'artifacts' not in args.output.resolve().parents:
            raise ValueError('normalization evidence must reside in campaign artifacts')
        immutable_json(args.output,{'schema':'car_round2_reserved_normalizer_v1','split':'train','value':value,
            'definition':'mean absolute frozen regional covariance on TRAIN truths', 'task_sha256':hash_file(args.reserved_task),
            'panel_sha256':hash_file(panel_path), 'sample_ids':ids, 'reserved_model_scores_used':False,
            'start_unix':started, 'end_unix':time.time(), 'cpu_seconds':time.process_time()-cpu})
    finally:
        dataset.close()
    print(args.output)


def generate(args, campaign):
    import yaml

    from phycoflow_reconstruction.evaluation.adaptive_campaign import (
        atomic_json,
        locked_ledger,
        spent,
    )
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu_uuid or not os.environ.get('KEOPS_CACHE_FOLDER'):
        raise ValueError('bind CUDA_VISIBLE_DEVICES to the physical UUID and set KeOps cache before launch')
    if not math.isfinite(args.max_seconds) or not math.isfinite(args.save_margin_seconds) or args.save_margin_seconds <= 0 or args.max_seconds <= args.save_margin_seconds:
        raise ValueError('a bounded job and positive save margin are required')
    family = 'ffm' if args.arm.startswith('FFM_') else 'senseiver'
    if args.kind == 'child' and args.epoch not in EPOCHS:
        raise ValueError('child generation requires one of the six prescribed epochs')
    if args.kind == 'source' and args.epoch is not None:
        raise ValueError('SOURCE has no child epoch')
    run = args.run.resolve(strict=True)
    cfg = yaml.safe_load((run / 'resolved_config.yaml').read_text())
    if args.preformal_source:
        if args.final_assessment:
            raise ValueError('preformal SOURCE generation cannot spend the final assessment reserve')
        if args.kind != 'source' or args.split != 'train' or args.epoch is not None:
            raise ValueError('preformal SOURCE generation is TRAIN-only and has no checkpoint')
        if json.loads((run/'status.json').read_text()).get('status') != 'prepared':
            raise ValueError('preformal SOURCE requires a completed SOURCE preparation')
    else:
        formal_path = campaign / 'configs/formal_freeze.json'
        formal = json.loads(formal_path.read_text())
        if formal.get('schema') != 'car_round2_freeze_receipt_v1' or formal.get('manifest', {}).get('status') != 'frozen':
            raise ValueError('formal frozen campaign required before generation')
        declaration = formal['manifest']['arms'][args.arm]
        if hash_file(declaration['config']['path']) != declaration['config']['sha256']:
            raise ValueError('frozen arm configuration changed')
        frozen_cfg = yaml.safe_load(Path(declaration['config']['path']).read_text())
        if not same_scientific_config(cfg, frozen_cfg):
            raise ValueError('run configuration differs from frozen arm')
    if cfg['optimization'].get('epochs') != 1000 or cfg['optimization'].get('steps_per_epoch') != 12:
        raise ValueError('expected the 1000-epoch B64/12 mature protocol')
    if args.split == 'validation':
        global_train_choice_barrier(campaign)
        _freeze, nomination = validation_freeze(campaign, family, args.arm if args.kind == 'child' else None, args.epoch)
        validate_reserved_freeze(campaign, family)
    else:
        nomination = None
    panel_path = campaign / 'configs/panels' / f'{args.split}.json'
    panel = json.loads(panel_path.read_text())
    ids = panel_contract(panel, args.split)
    snapshot = (campaign / 'sources' / family).resolve()
    if Path(cfg['source_run']).resolve() != snapshot:
        raise ValueError('run source does not match source snapshot')
    checkpoint = run / 'checkpoints' / f'epoch_{args.epoch:03d}.pt' if args.kind == 'child' else None
    checkpoint_hash = hash_file(checkpoint) if checkpoint else hash_file(snapshot / 'checkpoints/last.pt')
    if nomination is not None and nomination.get('child_checkpoint_sha256') != checkpoint_hash:
        raise ValueError('frozen TRAIN nomination binds a different checkpoint')
    if args.kind == 'child' and args.source_bank is None:
        raise ValueError('child generation requires the matched SOURCE bank')
    source_receipt = None
    if args.source_bank is not None:
        source_receipt = json.loads((args.source_bank / 'receipt.json').read_text())
        if (source_receipt.get('schema') != 'car_round2_native_bank_v1' or source_receipt.get('status') != 'completed' or source_receipt.get('kind') != 'source'
                or source_receipt.get('split') != args.split or source_receipt.get('family') != family
                or source_receipt.get('panel_sha256') != hash_file(panel_path)):
            raise ValueError('child requires the corresponding matched SOURCE bank')
    output = args.output.resolve()
    if campaign / 'assessment' not in output.parents:
        raise ValueError('assessment output must reside inside the campaign')
    listed = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'], text=True)
    devices = [tuple(v.strip() for v in row.split(',')) for row in listed.splitlines()]
    matching = [int(index) for index, gpu in devices if gpu == args.gpu_uuid]
    runtime = json.loads((campaign / 'operations/runtime.json').read_text())
    if len(matching) != 1 or (runtime.get('user_authorized_gpu_indices') is not None and matching[0] not in runtime['user_authorized_gpu_indices']):
        raise ValueError('unavailable or excluded physical GPU')
    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'], text=True)
    if any(row.split(',')[0].strip() == args.gpu_uuid for row in active.splitlines() if ',' in row):
        raise ValueError('physical GPU has an existing compute process')
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location('car_round2_driver', CODE_ROOT / 'scripts/posttraining/run_coherence_adaptive_repair_round2.py')
    driver = module_from_spec(spec); spec.loader.exec_module(driver)
    capsule = driver.check_code_capsule(campaign)
    token, started = str(uuid.uuid4()), time.time()
    ledger = campaign / 'operations/gpu_ledger.json'
    with locked_ledger(ledger) as state:
        limit = assessment_budget_limit(campaign, args.final_assessment)
        admit_assessment_budget(state, gpu_uuid=args.gpu_uuid, max_seconds=args.max_seconds,
                                cap=limit, spent_seconds=spent(state))
        output.mkdir(parents=True, exist_ok=False)
        state['active'][token] = {'gpu_uuid':args.gpu_uuid, 'phase':'native_assessment', 'start_unix':started}
    receipt = {'schema':'car_round2_native_bank_v1', 'status':'running', 'token':token, 'kind':args.kind,
               'split':args.split, 'family':family, 'arm':args.arm, 'epoch':args.epoch, 'start_unix':started,
               'checkpoint_sha256':checkpoint_hash, 'panel_sha256':hash_file(panel_path), 'code_capsule_sha256':capsule,
               'final_assessment':args.final_assessment, 'budget_limit_seconds':limit}
    old_alarm = signal.getsignal(signal.SIGALRM)
    def expire(_signum, _frame):
        raise TimeoutError('native assessment reached its save margin')
    signal.signal(signal.SIGALRM, expire)
    signal.alarm(max(1, math.ceil(args.max_seconds-args.save_margin_seconds)))
    train = dataset = None
    try:
        import torch

        from phycoflow_reconstruction.data.factory import open_field_dataset
        from phycoflow_reconstruction.data.manifest import dataset_fingerprint
        from phycoflow_reconstruction.data.sensor_protocols import (
            SensorProtocol,
            build_observation_batch,
        )
        from phycoflow_reconstruction.models.source_residual import FrozenSourceResidual
        from phycoflow_reconstruction.training.model_lifecycle import load_training_aux_state
        from phycoflow_reconstruction.training.rollout import differentiable_reconstruction
        from phycoflow_reconstruction.training.run_store import (
            load_model_state_strict,
            load_project_checkpoint,
        )
        from phycoflow_reconstruction.training.scalable_fitting import _digest
        from phycoflow_reconstruction.training.source import load_source_model
        model, train, source_meta = load_source_model(cfg, torch.device('cuda:0'))
        source_identity = _digest(model.state_dict())
        if checkpoint is not None:
            state = load_project_checkpoint(checkpoint)
            if state.get('global_step') != args.epoch*12 or state.get('source_hashes', {}).get('checkpoint') != hash_file(snapshot / 'checkpoints/last.pt'):
                raise ValueError('checkpoint epoch or source identity differs')
            if cfg.get('residual', {}).get('enabled'):
                r = cfg['residual']
                model = FrozenSourceResidual(model, coordinate_dim=r['coordinate_dim'], field_scales=r['field_scales'], width=r['width'], max_fraction=r['max_fraction']).to('cuda:0')
                load_model_state_strict(model, state['model'])
                if _digest(model.source.state_dict()) != source_identity:
                    raise ValueError('residual checkpoint changed frozen SOURCE state')
            else:
                load_model_state_strict(model, state['model'])
                load_training_aux_state(model, state)
        model.eval()
        dataset = train if args.split == 'train' else open_field_dataset(cfg['dataset'], split='validation', normalizer=train.normalizer)
        if len(dataset) != panel['population'] or dataset.normalizer.digest() != panel['normalizer_sha256'] or dataset_fingerprint(dataset.path) != panel['dataset_fingerprint']:
            raise ValueError('live dataset does not match frozen panel')
        files, coupling, times = [], [], []
        protocol = SensorProtocol(field_counts={'T':256}, seed=2047)
        with torch.no_grad():
            for ordinal, (index, sample_id) in enumerate(zip(panel['sample_indices'], ids)):
                sample = dataset[index]
                actual_id = f'{dataset.trajectory_ids[dataset._items[index][0]]}:{dataset._items[index][1]}'
                if actual_id != sample_id or sample.values.shape[0] != 40300:
                    raise ValueError('sample identity or native support differs')
                points = torch.randperm(40300, generator=torch.Generator().manual_seed(2047+index))[:256]
                observations = {sample_id:[(int(point), sample.field_names.index('T')) for point in points]}
                batch = build_observation_batch([sample], protocol, query_indices=None, manifest_indices=observations).to('cuda:0')
                truth, input_batch = batch.target_fields.detach().cpu(), replace(batch, target_fields=None)
                seeds = [member_seed(index, j) for j in range(panel['members'][family])]
                reference_row = None
                if source_receipt is not None:
                    path = args.source_bank / f'{ordinal:04d}.pt'
                    if hash_file(path) != source_receipt['files_sha256'][ordinal]:
                        raise ValueError('SOURCE bank row hash differs')
                    reference_row = torch.load(path, map_location='cpu', weights_only=True)
                torch.cuda.synchronize(); started_condition = time.perf_counter()
                predictions = []
                for j, seed in enumerate(seeds):
                    if checkpoint is not None and cfg.get('residual', {}).get('enabled'):
                        source_member = reference_row['members_model'][j].to('cuda:0')
                        prediction = source_member + model.correction(input_batch, source_member)
                    else:
                        prediction = differentiable_reconstruction(model, input_batch,
                            steps=int(cfg['rollout']['steps']), solver=cfg['rollout']['solver'],
                            generator=torch.Generator(device='cuda:0').manual_seed(seed),
                            observation_config=cfg['observation_consistency'])
                    predictions.append(prediction.detach().cpu())
                torch.cuda.synchronize(); times.append(time.perf_counter()-started_condition)
                row = {'sample_id':sample_id, 'sample_index':index, 'members_model':torch.stack(predictions),
                       'truth_model':truth, 'coordinates_model':batch.query_coords.detach().cpu(),
                       'coordinates_raw':sample.coordinates_raw.cpu(), 'native_masses':dataset.physical_weights.cpu(),
                       'obs_indices':batch.obs_indices.detach().cpu(), 'obs_values':batch.obs_values.detach().cpu(),
                       'obs_fields':batch.obs_field_ids.detach().cpu(), 'member_seeds':seeds,
                       'normalizer_offset':dataset.normalizer.offset.cpu(), 'normalizer_scale':dataset.normalizer.scale.cpu()}
                if reference_row is not None:
                    matched_rows(reference_row, row)
                path = output / f'{ordinal:04d}.pt'; torch.save(row, path)
                files.append(hash_file(path))
                coupling.append({'sample_id':sample_id, 'points':points.tolist(), 'seeds':seeds})
        receipt.update(status='completed', sample_ids=ids, source_sha256=hash_file(snapshot / 'checkpoints/last.pt'),
                       source_weight_view=source_meta['weight_view'], source_parameter_buffer_sha256=source_identity,
                       source_bank_receipt_sha256=hash_file(args.source_bank / 'receipt.json') if args.source_bank else None,
                       field_order=list(dataset.field_names), normalizer_sha256=dataset.normalizer.digest(),
                       population=len(dataset), members=panel['members'][family], native_points=40300,
                       coupling_sha256=hashlib.sha256(json.dumps(coupling, sort_keys=True).encode()).hexdigest(),
                       files_sha256=files, per_condition_seconds=times, gpu_uuid=args.gpu_uuid,
                       inference_scope='generation timing only; deployment profiling is a separate measured graph')
    except BaseException as exc:
        receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_alarm)
        torch_module = sys.modules.get('torch')
        if torch_module is not None and torch_module.cuda.is_initialized():
            try: torch_module.cuda.synchronize()
            except (RuntimeError, OSError) as exc: receipt.update(status='failed', teardown_error=str(exc))
        if dataset is not None and dataset is not train: dataset.close()
        if train is not None: train.close()
        receipt['end_unix'] = time.time(); receipt['wall_seconds'] = receipt['end_unix'] - started
        with locked_ledger(ledger) as state:
            entry = state['active'].pop(token)
            state['closed'].append({**entry, 'token':token, 'end_unix':receipt['end_unix'], 'seconds':receipt['wall_seconds']})
        atomic_json(output / 'receipt.json', receipt)
    print(output / 'receipt.json')


def score(args, campaign):
    import torch
    import yaml

    from phycoflow_reconstruction.coherence import build_enabled_families
    from phycoflow_reconstruction.data.factory import open_field_dataset
    from phycoflow_reconstruction.evaluation.adaptive_campaign import immutable_json
    from phycoflow_reconstruction.evaluation.frozen_coherence import FrozenCoherenceObserver
    from phycoflow_reconstruction.evaluation.round2_assessment import (
        AMPLITUDES,
        core_eligibility,
        field_sensor_eligibility,
        native_risks,
        scalar_task_score,
    )
    from phycoflow_reconstruction.evaluation.round2_task import regional_covariance
    from scripts.posttraining.freeze_car_event import FrozenRedesignedEvent
    started, cpu_started = time.time(), time.process_time()
    child = json.loads((args.child_bank / 'receipt.json').read_text())
    if child.get('split') == 'validation':
        global_train_choice_barrier(campaign)
    source = json.loads((args.source_bank / 'receipt.json').read_text())
    for receipt in (source, child):
        if receipt.get('schema') != 'car_round2_native_bank_v1' or receipt.get('status') != 'completed':
            raise ValueError('completed native endpoint banks required')
    if source['kind'] != 'source' or child['kind'] != 'child':
        raise ValueError('SOURCE and child endpoint banks required')
    if child.get('source_bank_receipt_sha256') != hash_file(args.source_bank / 'receipt.json'):
        raise ValueError('child bank does not bind this SOURCE bank')
    for key in ('split','family','source_sha256','source_weight_view','source_parameter_buffer_sha256',
                'panel_sha256','field_order','sample_ids','coupling_sha256','native_points','members','normalizer_sha256'):
        if source[key] != child[key]:
            raise ValueError(f'unmatched endpoint identity: {key}')
    family, split, epoch, arm = child['family'], child['split'], child['epoch'], child['arm']
    if split not in ('train','validation') or epoch not in EPOCHS or arm not in ARMS:
        raise ValueError('invalid checkpoint identity')
    panel_path = campaign / 'configs/panels' / f'{split}.json'
    panel = json.loads(panel_path.read_text())
    if panel_contract(panel, split) != source['sample_ids'] or hash_file(panel_path) != source['panel_sha256']:
        raise ValueError('saved banks do not bind the declared matched panel')
    formal = json.loads((campaign / 'configs/formal_freeze.json').read_text())
    manifest = formal['manifest']
    contract_binding = manifest['artifacts'][f'assessment_contract_{family}']
    if hash_file(contract_binding['path']) != contract_binding['sha256'] or Path(contract_binding['path']).resolve() != args.contract.resolve():
        raise ValueError('assessment contract is not the frozen source-specific artifact')
    contract = json.loads(args.contract.read_text())
    if contract.get('schema') != 'car_round2_assessment_contract_v2' or contract.get('source_checkpoint_sha256') != source['source_sha256']:
        raise ValueError('assessment contract does not bind SOURCE')
    normalizers = contract['normalizers']
    if set(normalizers) != {'state','scale','shape','event'} or any(not math.isfinite(float(v)) or float(v) <= 0 for v in normalizers.values()):
        raise ValueError('four positive TRAIN-frozen core/event normalizers required')
    if hash_file(args.event) != contract['event_sha256'] or hash_file(args.reserved_task) != contract['reserved_task_sha256']:
        raise ValueError('task manifest differs from the frozen assessment contract')
    event = FrozenRedesignedEvent(json.loads(args.event.read_text()))
    timing = json.loads(args.deployment_timing.read_text())
    if (timing.get('schema') != 'car_round2_deployment_timing_v1' or timing.get('status') != 'completed' or timing.get('arm') != arm
            or timing.get('epoch') != epoch or timing.get('checkpoint_sha256') != child['checkpoint_sha256']
            or timing.get('source_checkpoint_sha256') != source['source_sha256']):
        raise ValueError('measured deployment timing must bind the same actual functions')
    latency = {float(k):float(v) for k,v in timing['alpha_seconds'].items()}
    if set(latency) != set(AMPLITUDES) or any(not math.isfinite(v) or v <= 0 for v in latency.values()):
        raise ValueError('measured deployment latency for each declared amplitude required')
    if split == 'validation':
        _freeze, nomination = validation_freeze(campaign, family, arm, epoch)
        validate_reserved_freeze(campaign, family)
        if nomination['child_checkpoint_sha256'] != child['checkpoint_sha256'] or nomination['source_checkpoint_sha256'] != source['source_sha256']:
            raise ValueError('VALIDATION checkpoint differs from frozen TRAIN nomination')
        selected = float(nomination['selected_alpha'])
        alphas = tuple(sorted({0., 1., selected}))
    else:
        nomination, alphas = None, AMPLITUDES
    cfg = yaml.safe_load((args.core_run / 'resolved_config.yaml').read_text())
    dataset = open_field_dataset(cfg['dataset'], split='train')
    output = args.output.resolve()
    if campaign / 'assessment' not in output.parents:
        raise ValueError('score output must remain in campaign assessment storage')
    output.mkdir(parents=True, exist_ok=False)
    status = {'schema':'car_round2_assessment_cpu_job_v1','status':'running','start_unix':started}
    try:
        families = build_enabled_families(cfg['coherence'], dataset.data_spec, dataset.normalizer)
        core_manifest = json.loads((args.core_run / 'run_manifest.json').read_text())
        for name, family_module in families.items():
            path = args.core_run / 'artifacts' / f'{name}_family.pt'
            if hash_file(path) != core_manifest['coherence_family_state_sha256s'][name]:
                raise ValueError('common observer family artifact changed')
            family_module.load_state_artifact(torch.load(path, map_location='cpu', weights_only=True))
        core_path = args.core_run / 'artifacts/scalable_coherence.pt'
        if hash_file(core_path) != contract['core_sha256']:
            raise ValueError('common observer core differs from frozen contract')
        core = torch.load(core_path, map_location='cpu', weights_only=True)
        if core.get('adaptive_bank') is not None or core['provenance']['source_checkpoint_sha256'] != source['source_sha256']:
            raise ValueError('common observer must be SOURCE-bound C8 core only')
        if dataset.normalizer.digest() != source['normalizer_sha256'] or list(dataset.field_names) != source['field_order']:
            raise ValueError('observer normalizer or field order differs')
        query = torch.tensor(panel['query_indices'], dtype=torch.long)
        if query.numel() != 4096 or query.unique().numel() != 4096 or (query < 0).any() or (query >= 40300).any():
            raise ValueError('frozen Q4096 support is invalid')
        aggregates = {a:{'scores':{key:0. for key in normalizers}, 'risk':{}, 'sensor':None, 'per_condition':[], 'per_condition_scores':[]} for a in alphas}
        reserved = {'source':[], 'selected':[], 'truth':[], 'source_members':[], 'selected_members':[]}
        reserved_manifest = json.loads(args.reserved_task.read_text()) if split == 'validation' and epoch == 1000 else None
        for i, sample_id in enumerate(source['sample_ids']):
            sp, cp = args.source_bank / f'{i:04d}.pt', args.child_bank / f'{i:04d}.pt'
            if hash_file(sp) != source['files_sha256'][i] or hash_file(cp) != child['files_sha256'][i]:
                raise ValueError('endpoint row hash differs')
            s, c = (torch.load(path, map_location='cpu', weights_only=True) for path in (sp, cp))
            matched_rows(s,c)
            if s['sample_id'] != sample_id:
                raise ValueError('endpoint sample order differs')
            x,y,z = s['members_model'], c['members_model'], s['truth_model']
            offset, scale, mass = s['normalizer_offset'], s['normalizer_scale'], s['native_masses'].double()
            zphys = z*scale+offset
            observer = FrozenCoherenceObserver(core, families, expected_provenance=core['provenance'],
                reference=z[:,query], coordinates=s['coordinates_model'][:,query], sample_ids=[sample_id],
                member_weights=mass[query].unsqueeze(0), reference_weights=mass[query].unsqueeze(0),
                mode='conditional' if family == 'ffm' else 'deterministic',
                query_seed=int(contract['shape_query_seed']), complete_scale=True)
            for alpha in alphas:
                blend = x + alpha*(y-x)
                values = observer.score_tensors(blend[:,:,query])
                area_score, _, _ = score_event(event, blend*scale+offset, zphys, mass)
                row = aggregates[alpha]
                row['per_condition_scores'].append({key:float(values[key].mean()) for key in ('state','scale','shape')})
                for key in ('state','scale','shape'):
                    row['scores'][key] += float(values[key].mean())
                row['scores']['event'] += float(area_score.mean())
                risks = native_risks(blend*scale+offset, zphys, mass, model_family=family)
                for key in ('member_mse','ensemble_mean_mse','signed_mean_risk','unbiased_member_variance'):
                    if risks[key] is not None:
                        value = risks[key].double()
                        row['risk'][key] = value if key not in row['risk'] else row['risk'][key] + value
                row['per_condition'].append(risks['per_condition_member_mse'].reshape(-1).tolist())
                sensor = sensor_mse(blend, s, len(source['field_order'])) * scale.double().square()
                row['sensor'] = sensor if row['sensor'] is None else row['sensor'] + sensor
                if reserved_manifest is not None and alpha in (0., selected):
                    task_members = torch.stack([regional_covariance(member, mass, reserved_manifest, tuple(source['field_order'])) for member in blend*scale+offset])
                    task_truth = regional_covariance(zphys, mass, reserved_manifest, tuple(source['field_order']))
                    task_score = scalar_task_score(task_members, task_truth, model_family=family)
                    reserved['source' if alpha == 0. else 'selected'].append(float(task_score.mean()))
                    reserved['source_members' if alpha == 0. else 'selected_members'].append(task_members[:,0].tolist())
                    if alpha == 0.: reserved['truth'].append(float(task_truth.mean()))
        count = len(source['sample_ids'])
        for row in aggregates.values():
            row['scores'] = {key:value/count for key,value in row['scores'].items()}
            row['risk'] = {key:(value/count) for key,value in row['risk'].items()}
            row['sensor'] /= count
        base = aggregates[0.]
        for alpha, row in aggregates.items():
            row['core'] = core_eligibility(base['scores'], row['scores'], normalizers)
            row['fidelity'] = field_sensor_eligibility(base['risk'], row['risk'], model_family=family,
                absolute_mean_allowance=contract['mean_allowance_physical'], source_sensor_mse=base['sensor'],
                child_sensor_mse=row['sensor'], sensor_equality_tolerance=contract['sensor_equality_tolerance'])
            row['utility'] = sum(row['scores'][key]/normalizers[key] for key in normalizers)
            row['eligible'] = row['core']['pass'] and row['fidelity']['primary_pass']
            row['latency_seconds'] = latency[alpha]
        if split == 'train':
            feasible = [a for a,row in aggregates.items() if row['eligible']]
            if feasible:
                best = min(aggregates[a]['utility'] for a in feasible)
                selected = min((a for a in feasible if aggregates[a]['utility'] <= best+1e-6), key=lambda a:(latency[a],a))
            else:
                selected = 0.
        rows = []
        for alpha,row in aggregates.items():
            rows.append({'alpha':alpha, 'scores':row['scores'], 'risk':{k:v.tolist() for k,v in row['risk'].items()},
                         'sensor_mse':row['sensor'].tolist(), 'core':row['core'], 'fidelity':row['fidelity'],
                         'utility':row['utility'], 'eligible':row['eligible'], 'latency_seconds':row['latency_seconds'],
                         'per_condition_member_mse':row['per_condition'], 'per_condition_scores':row['per_condition_scores']})
        # Save the actual selected nonlinear blend, retaining native model fields.
        selected_dir = output / 'selected'; selected_dir.mkdir()
        hashes, selected_tails, raw_tails = [], [], []
        represented_decomposition = {'selected':[], 'raw':[]}
        scale_components = {'selected':[], 'raw':[]}
        event_distributions = {'selected':[], 'raw':[]}
        for i in range(count):
            s,c = (torch.load(path / f'{i:04d}.pt', map_location='cpu', weights_only=True) for path in (args.source_bank,args.child_bank))
            target = selected_dir / f'{i:04d}.pt'
            selected_values = s['members_model'] + selected*(c['members_model']-s['members_model'])
            mass = s['native_masses'].double()
            scale_observer = FrozenCoherenceObserver(core, families, expected_provenance=core['provenance'],
                reference=s['truth_model'][:,query], coordinates=s['coordinates_model'][:,query],
                sample_ids=[s['sample_id']], member_weights=mass[query].unsqueeze(0),
                reference_weights=mass[query].unsqueeze(0), mode='conditional' if family == 'ffm' else 'deterministic',
                query_seed=int(contract['shape_query_seed']), complete_scale=True)
            for label, prediction in (('selected',selected_values), ('raw',c['members_model'])):
                sampled = prediction[:,:,query]
                context = {'scoring_mode':scale_observer.mode, 'field_order':scale_observer.fields,
                           'conditional_draws':tuple(sampled[1:]), 'score_all_conditional_draws':True}
                _, blocks, _ = scale_observer.adapter.score('multiscale_spectrum', sampled[0],
                    s['truth_model'][:,query], coordinates=s['coordinates_model'][:,query],
                    context=context, batch=scale_observer.scale_batches[0], coefficients=scale_observer.coefficients)
                scale_components[label].append({key:float(value) for key,value in blocks.items()})
                physical_prediction = prediction*s['normalizer_scale'] + s['normalizer_offset']
                physical_truth = s['truth_model']*s['normalizer_scale'] + s['normalizer_offset']
                member_masks, truth_masks = event.masks(physical_prediction), event.masks(physical_truth)
                mass_weights = mass/mass.sum()
                def areas(masks, mass_weights=mass_weights):
                    return {key:(mask*mass_weights).sum(-1).tolist()
                            for key,mask in zip(('T','CO','CH4','joint'),masks)}
                event_distributions[label].append({'member_areas':areas(member_masks),
                                                    'truth_areas':areas(truth_masks)})
            if family == 'ffm':
                point_observer = FrozenCoherenceObserver(core, families, expected_provenance=core['provenance'],
                    reference=s['truth_model'][:,query], coordinates=s['coordinates_model'][:,query],
                    sample_ids=[s['sample_id']], member_weights=mass[query].unsqueeze(0),
                    reference_weights=mass[query].unsqueeze(0), mode='deterministic',
                    query_seed=int(contract['shape_query_seed']), complete_scale=True)
                for label, prediction, alpha in (('selected',selected_values,selected), ('raw',c['members_model'],1.)):
                    attraction = {key:0. for key in ('state','scale','shape')}
                    for member in prediction:
                        values = point_observer.score_tensors(member.unsqueeze(0)[:,:,query])
                        for key in attraction: attraction[key] += float(values[key].mean())/len(prediction)
                    fair = aggregates[alpha]['per_condition_scores'][i]
                    represented_decomposition[label].append({'attraction':attraction,
                        'fair_separation_correction':{key:attraction[key]-fair[key] for key in attraction}})
            for prediction, collection in ((selected_values, selected_tails), (c['members_model'], raw_tails)):
                errors = ((prediction-s['truth_model'])*s['normalizer_scale']).abs()[...,mass > 0,:].reshape(-1,len(source['field_order'])).float()
                tails = torch.quantile(errors, torch.tensor([.95,.99,1.]), dim=0)
                collection.append(tails.tolist())
            torch.save({'sample_id':s['sample_id'], 'members_model':selected_values,
                        'truth_model':s['truth_model'], 'member_seeds':s['member_seeds']}, target)
            hashes.append(hash_file(target))
        result = {'schema':'car_round2_train_nomination_v1' if split == 'train' else 'car_round2_validation_assessment_v1',
                  'split':split, 'family':family, 'arm':arm, 'epoch':epoch, 'sample_ids':source['sample_ids'],
                  'population':panel['population'], 'source_checkpoint_sha256':source['source_sha256'],
                  'child_checkpoint_sha256':child['checkpoint_sha256'], 'source_bank_sha256':hash_file(args.source_bank/'receipt.json'),
                  'child_bank_sha256':hash_file(args.child_bank/'receipt.json'), 'panel_sha256':hash_file(panel_path),
                  'core_sha256':hash_file(core_path), 'contract_sha256':hash_file(args.contract), 'selected_alpha':selected,
                  'rows':rows, 'selected_files_sha256':hashes, 'reserved_task_used':False,
                  'native_abs_error_tail_quantiles':[.95,.99,1.],
                  'native_abs_error_tails_scope':'per-condition unweighted positive-mass native points and members, decoded field units',
                  'selected_native_abs_error_tails':selected_tails,'raw_native_abs_error_tails':raw_tails,
                  'represented_decomposition':represented_decomposition if family == 'ffm' else None,
                  'represented_decomposition_scope':'per-condition mean paired attraction and attraction minus fair score for each represented family' if family == 'ffm' else 'unavailable for J1 point reconstruction',
                  'selected_and_raw_scale_components':scale_components,
                  'scale_component_scope':'Q4096 complete signed, joint, log-power and rho blocks with unit internal coefficients',
                  'reused_event_area_distributions':event_distributions,
                  'normalizers':normalizers, 'source_latency_seconds':latency[0.], 'gpu_uuid':timing['gpu_uuid'],
                  'deployment_timing_sha256':hash_file(args.deployment_timing), 'start_unix':started, 'end_unix':time.time()}
        if reserved_manifest is not None:
            source_score = sum(reserved['source'])/count
            selected_score = source_score if selected == 0. else sum(reserved['selected'])/count
            normalizer_receipt = validate_reserved_freeze(campaign, family)
            denominator = float(normalizer_receipt['normalizer'])
            if not math.isfinite(denominator) or denominator <= 0:
                raise ValueError('positive TRAIN-frozen reserved-task normalizer required')
            result['reserved_task'] = {'source_score':source_score,'selected_score':selected_score,
                                       'normalized_improvement':(source_score-selected_score)/denominator,
                                       'source_per_condition':reserved['source'],'selected_per_condition':reserved['selected'] if selected else reserved['source'],
                                       'source_member_values':reserved['source_members'],
                                       'selected_member_values':reserved['selected_members'] if selected else reserved['source_members'],
                                       'truth':reserved['truth'],'task_sha256':hash_file(args.reserved_task),
                                       'train_source_normalizer':denominator,
                                       'normalizer_receipt_sha256':hash_file(campaign / 'assessment/train' / f'reserved_normalizer_{family}.json')}
        immutable_json(output / ('nomination.json' if split == 'train' else 'assessment.json'), result)
        status['status'] = 'completed'
    except BaseException as exc:
        status.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        dataset.close()
        status.update(end_unix=time.time(), cpu_seconds=time.process_time()-cpu_started)
        status['wall_seconds'] = status['end_unix']-started
        immutable_json(output / 'cpu_receipt.json', status)
    print(output)


def freeze_nominations(args, campaign):
    from phycoflow_reconstruction.evaluation.adaptive_campaign import immutable_json
    from phycoflow_reconstruction.evaluation.round2_assessment import freeze_method_priority
    formal_path = campaign / 'configs/formal_freeze.json'
    formal = json.loads(formal_path.read_text())
    if formal.get('schema') != 'car_round2_freeze_receipt_v1' or formal.get('manifest', {}).get('status') != 'frozen':
        raise ValueError('formal freeze required')
    prefix = 'FFM' if args.family == 'ffm' else 'Senseiver'
    runs = {f'{prefix}_Selected_Bank_Uniform_FullModel':args.full_run.resolve(),
            f'{prefix}_Selected_Bank_Residual_Repair':args.residual_run.resolve()}
    records, final_rows, common, final_devices = {}, [], None, []
    for arm, run in runs.items():
        status = json.loads((run / 'status.json').read_text())
        if status.get('status') != 'completed' or status.get('global_step') != 12000:
            raise ValueError('both mature arms must complete 12000 updates')
        entries = {}
        for epoch in EPOCHS:
            path = campaign / 'assessment/train' / arm / f'epoch_{epoch:03d}' / 'nomination.json'
            value = json.loads(path.read_text())
            checkpoint = run / 'checkpoints' / f'epoch_{epoch:03d}.pt'
            if (value.get('schema') != 'car_round2_train_nomination_v1' or value.get('split') != 'train'
                    or value.get('arm') != arm or value.get('epoch') != epoch
                    or value.get('reserved_task_used') is not False
                    or value.get('child_checkpoint_sha256') != hash_file(checkpoint)):
                raise ValueError('TRAIN nomination does not bind the correct checkpoint')
            signature = tuple(value[k] for k in ('source_checkpoint_sha256','panel_sha256','core_sha256','contract_sha256'))
            if common is None: common = signature
            if common != signature: raise ValueError('TRAIN nominations lack matched SOURCE/panel/core/contract')
            entries[str(epoch)] = {'nomination':str(path), 'sha256':hash_file(path)}
            if epoch == 1000:
                alpha = float(value['selected_alpha'])
                row = next(item for item in value['rows'] if item['alpha'] == alpha)
                final_devices.append(value['gpu_uuid'])
                final_rows.append({'method':arm, 'alpha':alpha, 'eligible':row['eligible'],
                                   'utility':row['utility'], 'latency_seconds':row['latency_seconds']})
        records[arm] = entries
    if len({str(p) for p in runs.values()}) != 2:
        raise ValueError('distinct completed arm runs required')
    if len(set(final_devices)) != 1:
        raise ValueError('TRAIN method priority requires same-device deployment measurements')
    result = {'schema':'car_round2_assessment_freeze_v1', 'family':args.family,
              'formal_freeze_sha256':hash_file(formal_path), 'arms':records,
              'method_priority':freeze_method_priority(final_rows), 'final_train_candidates':final_rows,
              'reserved_task_used':False, 'frozen_unix':time.time()}
    result['sha256'] = content_hash(result)
    path = campaign / 'assessment/train' / f'method_priority_{args.family}.json'
    immutable_json(path, result)
    print(path)


def profile_deployment(args, campaign):
    import yaml

    from phycoflow_reconstruction.evaluation.adaptive_campaign import (
        immutable_json,
        locked_ledger,
        spent,
    )
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu_uuid or not os.environ.get('KEOPS_CACHE_FOLDER'):
        raise ValueError('bind physical GPU UUID and device-specific KeOps cache before launch')
    if args.warmup < 1 or not math.isfinite(args.max_seconds) or not math.isfinite(args.save_margin_seconds) or args.save_margin_seconds <= 0 or args.max_seconds <= args.save_margin_seconds:
        raise ValueError('warmup and bounded GPU job required')
    panel_path = campaign / 'configs/panels/train.json'
    panel = json.loads(panel_path.read_text()); ids = panel_contract(panel, 'train')
    run = args.run.resolve(strict=True)
    cfg = yaml.safe_load((run / 'resolved_config.yaml').read_text())
    family = 'ffm' if args.arm.startswith('FFM_') else 'senseiver'
    formal = json.loads((campaign / 'configs/formal_freeze.json').read_text())
    if formal.get('schema') != 'car_round2_freeze_receipt_v1':
        raise ValueError('formal freeze required')
    binding = formal['manifest']['arms'][args.arm]['config']
    if hash_file(binding['path']) != binding['sha256'] or not same_scientific_config(cfg,yaml.safe_load(Path(binding['path']).read_text())):
        raise ValueError('deployment run differs from frozen configuration')
    if args.epoch not in EPOCHS or cfg['optimization'].get('steps_per_epoch') != 12:
        raise ValueError('prescribed mature checkpoint required')
    checkpoint = run / 'checkpoints' / f'epoch_{args.epoch:03d}.pt'
    source = campaign / 'sources' / family
    if Path(cfg['source_run']).resolve() != source.resolve():
        raise ValueError('source identity differs from frozen snapshot')
    listed = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'], text=True)
    devices = [tuple(x.strip() for x in row.split(',')) for row in listed.splitlines()]
    matching = [int(index) for index, gpu in devices if gpu == args.gpu_uuid]
    runtime = json.loads((campaign / 'operations/runtime.json').read_text())
    if len(matching) != 1 or (runtime.get('user_authorized_gpu_indices') is not None and matching[0] not in runtime['user_authorized_gpu_indices']):
        raise ValueError('unavailable or excluded device')
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'], text=True)
    if any(row.split(',')[0].strip() == args.gpu_uuid for row in apps.splitlines() if ',' in row):
        raise ValueError('device already has compute processes')
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location('car_round2_driver', CODE_ROOT / 'scripts/posttraining/run_coherence_adaptive_repair_round2.py')
    driver = module_from_spec(spec); spec.loader.exec_module(driver)
    capsule = driver.check_code_capsule(campaign)
    output = args.output.resolve()
    if campaign / 'assessment' not in output.parents or output.exists():
        raise ValueError('unique deployment timing output inside assessment required')
    token, started, ledger = str(uuid.uuid4()), time.time(), campaign / 'operations/gpu_ledger.json'
    with locked_ledger(ledger) as state:
        limit = assessment_budget_limit(campaign, args.final_assessment)
        admit_assessment_budget(state, gpu_uuid=args.gpu_uuid, max_seconds=args.max_seconds,
                                cap=limit, spent_seconds=spent(state))
        state['active'][token] = {'gpu_uuid':args.gpu_uuid,'phase':'deployment_timing','start_unix':started}
    receipt = {'schema':'car_round2_deployment_timing_v1','status':'running','token':token,
               'start_unix':started, 'arm':args.arm, 'epoch':args.epoch, 'gpu_uuid':args.gpu_uuid,
               'checkpoint_sha256':hash_file(checkpoint), 'source_checkpoint_sha256':hash_file(source/'checkpoints/last.pt'),
               'panel_sha256':hash_file(panel_path), 'code_capsule_sha256':capsule,
               'final_assessment':args.final_assessment, 'budget_limit_seconds':limit}
    old_alarm = signal.getsignal(signal.SIGALRM)
    def expire(_signum, _frame): raise TimeoutError('deployment profiling reached save margin')
    signal.signal(signal.SIGALRM, expire)
    signal.alarm(max(1, math.ceil(args.max_seconds-args.save_margin_seconds)))
    dataset = None
    try:
        import torch

        from phycoflow_reconstruction.data.sensor_protocols import (
            SensorProtocol,
            build_observation_batch,
        )
        from phycoflow_reconstruction.evaluation.round2_assessment import AMPLITUDES
        from phycoflow_reconstruction.models.source_residual import FrozenSourceResidual
        from phycoflow_reconstruction.training.model_lifecycle import load_training_aux_state
        from phycoflow_reconstruction.training.rollout import differentiable_reconstruction
        from phycoflow_reconstruction.training.run_store import (
            load_model_state_strict,
            load_project_checkpoint,
        )
        from phycoflow_reconstruction.training.scalable_fitting import _digest
        from phycoflow_reconstruction.training.source import load_source_model
        source_model, dataset, _ = load_source_model(cfg, torch.device('cuda:0'))
        source_hash = _digest(source_model.state_dict())
        state = load_project_checkpoint(checkpoint)
        if state.get('global_step') != 12*args.epoch or state.get('source_hashes', {}).get('checkpoint') != receipt['source_checkpoint_sha256']:
            raise ValueError('checkpoint epoch/source mismatch')
        residual = bool(cfg.get('residual',{}).get('enabled'))
        if residual:
            r = cfg['residual']
            child = FrozenSourceResidual(source_model, coordinate_dim=r['coordinate_dim'], field_scales=r['field_scales'], width=r['width'], max_fraction=r['max_fraction']).to('cuda:0')
            load_model_state_strict(child, state['model'])
            if _digest(source_model.state_dict()) != source_hash: raise ValueError('residual changed SOURCE state')
        else:
            child = copy.deepcopy(source_model)
            load_model_state_strict(child, state['model']); load_training_aux_state(child, state)
        source_model.eval(); child.eval()
        if len(dataset) != panel['population'] or dataset.normalizer.digest() != panel['normalizer_sha256']:
            raise ValueError('timing dataset differs from TRAIN panel')
        protocol = SensorProtocol(field_counts={'T':256}, seed=2047)
        def invoke(sample, index, sample_id, alpha):
            points = torch.randperm(40300, generator=torch.Generator().manual_seed(2047+index))[:256]
            obs = {sample_id:[(int(p), sample.field_names.index('T')) for p in points]}
            batch = build_observation_batch([sample], protocol, query_indices=None, manifest_indices=obs).to('cuda:0')
            batch = replace(batch, target_fields=None)
            outputs = []
            for j in range(panel['members'][family]):
                seed = member_seed(index,j)
                def reconstruct(model, seed=seed):
                    return differentiable_reconstruction(model, batch, steps=int(cfg['rollout']['steps']), solver=cfg['rollout']['solver'],
                        generator=torch.Generator(device='cuda:0').manual_seed(seed), observation_config=cfg['observation_consistency'])
                if alpha == 0.:
                    value = reconstruct(source_model)
                elif residual:
                    base = reconstruct(source_model)
                    value = base + alpha * child.correction(batch, base)
                elif alpha == 1.:
                    value = reconstruct(child)
                else:
                    base, repaired = reconstruct(source_model), reconstruct(child)
                    value = base + alpha * (repaired-base)
                outputs.append(value.detach().cpu())
            return outputs
        first_index, first_id = panel['sample_indices'][0], ids[0]
        first_sample = dataset[first_index]
        with torch.no_grad():
            for _ in range(args.warmup):
                for alpha in AMPLITUDES: invoke(first_sample, first_index, first_id, alpha)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            samples = {str(alpha):[] for alpha in AMPLITUDES}
            for index, sample_id in zip(panel['sample_indices'], ids):
                sample = dataset[index]
                if f'{dataset.trajectory_ids[dataset._items[index][0]]}:{dataset._items[index][1]}' != sample_id:
                    raise ValueError('timing panel sample identity changed')
                for alpha in AMPLITUDES:
                    torch.cuda.synchronize(); begin = time.perf_counter()
                    invoke(sample, index, sample_id, alpha)
                    torch.cuda.synchronize(); samples[str(alpha)].append(time.perf_counter()-begin)
        receipt.update(status='completed', alpha_seconds={key:sum(values)/len(values) for key,values in samples.items()},
                       alpha_per_condition_seconds=samples, sample_ids=ids, members=panel['members'][family],
                       native_points=40300, sensor_count=256, warmup_repeats=args.warmup,
                       device_name=torch.cuda.get_device_name(0), cuda_version=torch.version.cuda,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       scope='full input/sensor assembly, host-device transfer, native reconstruction, member draws, mixing and device-host output transfer')
    except BaseException as exc:
        receipt.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_alarm)
        torch_module = sys.modules.get('torch')
        if torch_module is not None and torch_module.cuda.is_initialized():
            try: torch_module.cuda.synchronize()
            except (RuntimeError, OSError) as exc: receipt.update(status='failed',teardown_error=str(exc))
        if dataset is not None: dataset.close()
        receipt['end_unix']=time.time(); receipt['wall_seconds']=receipt['end_unix']-started
        with locked_ledger(ledger) as state:
            entry=state['active'].pop(token); state['closed'].append({**entry,'token':token,'end_unix':receipt['end_unix'],'seconds':receipt['wall_seconds']})
        immutable_json(output, receipt)
    print(output)


def decide(args, campaign):
    from phycoflow_reconstruction.evaluation.adaptive_campaign import immutable_json
    from phycoflow_reconstruction.evaluation.round2_assessment import deployment_disposition
    freeze, _ = validation_freeze(campaign, args.family)
    priority = freeze['method_priority']
    paths = {arm: campaign / 'assessment/validation' / arm for arm in freeze['arms']}
    costs = json.loads(args.costs.read_text())
    if costs.get('schema') != 'car_round2_cost_reconciliation_v1' or set(costs.get('arms',{})) != set(paths):
        raise ValueError('complete per-arm cost reconciliation required')
    complete = costs.get('complete') is True
    within = costs.get('within_budget') is True
    if complete:
        receipts = costs.get('source_receipts', [])
        if not receipts or any(hash_file(item['path']) != item['sha256'] for item in receipts):
            raise ValueError('complete costs require hash-verified job and lease receipts')
        for item in costs['arms'].values():
            serial = item.get('serial_cost_seconds')
            if not isinstance(serial,(int,float)) or not math.isfinite(serial) or serial <= 0:
                raise ValueError('complete costs require positive measured serial cost for each arm')
        occupied = costs.get('total_occupied_gpu_seconds')
        if not isinstance(occupied,(int,float)) or not math.isfinite(occupied) or occupied < 0:
            raise ValueError('complete costs require a measured GPU occupancy total')
        within = within and occupied <= 130*3600
    evaluated, final = {}, {}
    for arm, directory in paths.items():
        evaluated[arm] = {}
        for epoch in EPOCHS:
            path = directory / f'epoch_{epoch:03d}' / 'assessment.json'
            doc = json.loads(path.read_text())
            if (doc.get('schema') != 'car_round2_validation_assessment_v1' or doc.get('split') != 'validation'
                    or doc.get('arm') != arm or doc.get('epoch') != epoch
                    or doc['child_checkpoint_sha256'] != json.loads(Path(freeze['arms'][arm][str(epoch)]['nomination']).read_text())['child_checkpoint_sha256']):
                raise ValueError('held-out assessment does not match the frozen trajectory')
            evaluated[arm][epoch] = doc
        final[arm] = evaluated[arm][1000]
        if 'reserved_task' not in final[arm]:
            raise ValueError('final fixed reserved task assessment is missing')
    if len({doc['gpu_uuid'] for doc in final.values()}) != 1:
        raise ValueError('deployment comparison requires measured same-device timing')
    rows = {}
    for arm, doc in final.items():
        row = next(row for row in doc['rows'] if row['alpha'] == doc['selected_alpha'])
        rows[arm] = {'fidelity':row['fidelity'], 'core':row['core'],
                     'reserved_normalized_improvement':doc['reserved_task']['normalized_improvement'],
                     'source_latency_ratio':row['latency_seconds']/doc['source_latency_seconds']}
    disposition = deployment_disposition(priority, rows, model_family=args.family,
                                         complete_costs=complete, within_budget=within)
    full = next(arm for arm in paths if arm.endswith('Uniform_FullModel'))
    residual = next(arm for arm in paths if arm.endswith('Residual_Repair'))
    frow = next(row for row in final[full]['rows'] if row['alpha'] == final[full]['selected_alpha'])
    rrow = next(row for row in final[residual]['rows'] if row['alpha'] == final[residual]['selected_alpha'])
    fidelity = rows[residual]['fidelity']['primary_pass'] and rows[residual]['core']['pass']
    residual_task = rows[residual]['reserved_normalized_improvement']
    full_task = rows[full]['reserved_normalized_improvement']
    task_difference = abs(residual_task-full_task)
    equivalent_task = task_difference <= .05 or math.isclose(task_difference,.05,rel_tol=1e-12,abs_tol=1e-12)
    residual_cost = costs['arms'][residual].get('serial_cost_seconds')
    full_cost = costs['arms'][full].get('serial_cost_seconds')
    valid_costs = all(isinstance(x,(int,float)) and math.isfinite(x) and x > 0 for x in (residual_cost,full_cost))
    residual_task_eligible = fidelity and residual_task >= .05
    quality = bool(residual_task_eligible and residual_task-full_task >= .05 and complete and within and valid_costs
                   and residual_cost <= 1.1*full_cost and rows[residual]['source_latency_ratio'] <= 1.2)
    full_eligible = rows[full]['fidelity']['primary_pass'] and rows[full]['core']['pass'] and full_task >= .05
    deployment = bool(residual_task_eligible and full_eligible and equivalent_task
                      and rrow['latency_seconds'] <= .8*frow['latency_seconds']
                      and rows[residual]['source_latency_ratio'] <= 1.2 and complete and within)
    crossings = {}
    for arm, trajectory in evaluated.items():
        qualifies = []
        for epoch in EPOCHS:
            doc = trajectory[epoch]
            base = next(row for row in doc['rows'] if row['alpha'] == 0.)
            row = next(row for row in doc['rows'] if row['alpha'] == doc['selected_alpha'])
            qualifies.append(row['fidelity']['primary_pass'] and row['scores']['scale'] <= .99*base['scores']['scale']
                             and row['scores']['shape'] <= .95*base['scores']['shape']
                             and (row['scores']['state']-base['scores']['state'])/doc['normalizers']['state'] <= .02)
        crossing = next((EPOCHS[i] for i in range(1,len(EPOCHS)) if qualifies[i-1] and qualifies[i]), None)
        crossing_cost = costs['arms'][arm].get('checkpoint_cost_seconds',{}).get(str(crossing)) if crossing is not None else None
        crossings[arm] = {'qualifies_by_epoch':dict(zip(map(str,EPOCHS),qualifies)), 'crossing_epoch':crossing, 'crossing_cost_seconds':crossing_cost}
    a,b = crossings[residual]['crossing_cost_seconds'], crossings[full]['crossing_cost_seconds']
    efficiency = bool(residual_task_eligible and full_eligible and equivalent_task and complete and within
                      and isinstance(a,(int,float)) and isinstance(b,(int,float)) and a >= 0 and b > 0 and a <= .8*b)
    result = {'schema':'car_round2_final_decision_v1','family':args.family,'frozen_priority':priority,
              'deployment_disposition':disposition,'quality_route':quality,'learning_efficiency_route':efficiency,
              'deployment_comparison_route':deployment,'crossings':crossings,
              'cost_reconciliation_sha256':hash_file(args.costs),'costs_complete':complete,'within_budget':within,
              'heldout_selection':'TRAIN-frozen amplitude and method order; no held-out search',
              'device_uuid':next(iter({doc['gpu_uuid'] for doc in final.values()}))}
    if campaign / 'assessment' not in args.output.resolve().parents:
        raise ValueError('decision output must reside in campaign assessment storage')
    immutable_json(args.output, result)
    print(args.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    sub = p.add_subparsers(dest='command', required=True)
    panel = sub.add_parser('panel')
    panel.add_argument('--split', choices=('train','validation'), required=True)
    panel.add_argument('--core-run', type=Path, required=True)
    prep = sub.add_parser('prepare-contract')
    prep.add_argument('--source-bank',type=Path,required=True)
    prep.add_argument('--core-run',type=Path,required=True)
    prep.add_argument('--event',type=Path,required=True)
    prep.add_argument('--reserved-task',type=Path,required=True)
    prep.add_argument('--reserved-task-normalizer',type=float,required=True)
    prep.add_argument('--reserved-normalizer-evidence',type=Path,required=True)
    prep.add_argument('--sensor-equality-tolerance',type=float,nargs='+',required=True)
    prep.add_argument('--shape-query-seed',type=int,default=2027)
    prep.add_argument('--output',type=Path,required=True)
    norm = sub.add_parser('reserved-normalizer')
    norm.add_argument('--core-run',type=Path,required=True)
    norm.add_argument('--reserved-task',type=Path,required=True)
    norm.add_argument('--output',type=Path,required=True)
    gen = sub.add_parser('generate')
    gen.add_argument('--kind', choices=('source','child'), required=True)
    gen.add_argument('--preformal-source', action='store_true')
    gen.add_argument('--final-assessment', action='store_true', help='use closeout reserve after all four frozen mature runs complete')
    gen.add_argument('--split', choices=('train','validation'), required=True)
    gen.add_argument('--arm', choices=ARMS, required=True)
    gen.add_argument('--run', type=Path, required=True)
    gen.add_argument('--epoch', type=int, choices=EPOCHS)
    gen.add_argument('--source-bank', type=Path)
    gen.add_argument('--output', type=Path, required=True)
    gen.add_argument('--gpu-uuid', required=True)
    gen.add_argument('--max-seconds', type=float, required=True)
    gen.add_argument('--save-margin-seconds', type=float, required=True)
    evaluate = sub.add_parser('score')
    evaluate.add_argument('--source-bank', type=Path, required=True)
    evaluate.add_argument('--child-bank', type=Path, required=True)
    evaluate.add_argument('--core-run', type=Path, required=True)
    evaluate.add_argument('--contract', type=Path, required=True)
    evaluate.add_argument('--event', type=Path, required=True)
    evaluate.add_argument('--reserved-task', type=Path, required=True)
    evaluate.add_argument('--deployment-timing', type=Path, required=True)
    evaluate.add_argument('--output', type=Path, required=True)
    reserved_freeze = sub.add_parser('freeze-reserved-normalizer')
    reserved_freeze.add_argument('--family', choices=('ffm','senseiver'), required=True)
    reserved_freeze.add_argument('--source-bank', type=Path, required=True)
    reserved_freeze.add_argument('--reserved-task', type=Path, required=True)
    freeze = sub.add_parser('freeze-nominations')
    freeze.add_argument('--family', choices=('ffm','senseiver'), required=True)
    freeze.add_argument('--full-run', type=Path, required=True)
    freeze.add_argument('--residual-run', type=Path, required=True)
    profile = sub.add_parser('profile-deployment')
    profile.add_argument('--arm', choices=ARMS, required=True)
    profile.add_argument('--final-assessment', action='store_true', help='use closeout reserve after all four frozen mature runs complete')
    profile.add_argument('--run', type=Path, required=True)
    profile.add_argument('--epoch', type=int, choices=EPOCHS, required=True)
    profile.add_argument('--gpu-uuid', required=True)
    profile.add_argument('--max-seconds', type=float, required=True)
    profile.add_argument('--save-margin-seconds', type=float, required=True)
    profile.add_argument('--warmup', type=int, default=3)
    profile.add_argument('--output', type=Path, required=True)
    decision = sub.add_parser('decide')
    decision.add_argument('--family', choices=('ffm','senseiver'), required=True)
    decision.add_argument('--costs', type=Path, required=True)
    decision.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    campaign = args.campaign.resolve(strict=True)
    if '/data2/wanglz/PhyCoFlow-MultiFieldReconstruction/' not in str(campaign) + '/':
        raise ValueError('assessment campaign must reside on DATA2 project storage')
    if args.command == 'panel':
        build_panel(args, campaign)
    elif args.command == 'prepare-contract':
        prepare_contract(args, campaign)
    elif args.command == 'reserved-normalizer':
        reserved_normalizer(args, campaign)
    elif args.command == 'generate':
        generate(args, campaign)
    elif args.command == 'score':
        score(args, campaign)
    elif args.command == 'freeze-reserved-normalizer':
        freeze_reserved_normalizer(args, campaign)
    elif args.command == 'freeze-nominations':
        freeze_nominations(args, campaign)
    elif args.command == 'profile-deployment':
        profile_deployment(args, campaign)
    elif args.command == 'decide':
        decide(args, campaign)

if __name__ == '__main__':
    main()
