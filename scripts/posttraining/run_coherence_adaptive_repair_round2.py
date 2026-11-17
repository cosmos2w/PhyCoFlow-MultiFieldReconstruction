#!/usr/bin/env python3
"""Round 2 campaign admission, immutable configuration, and timed execution.

All scientific artifacts must already have been produced and reviewed; admission
checks their hashes and qualification receipts. TEST and EDM are never opened.
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
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('CAR_REPO_ROOT', CODE_ROOT)).resolve()
sys.path.insert(0, str(CODE_ROOT / 'src'))
from phycoflow_reconstruction.evaluation.adaptive_campaign import (
    atomic_json,
    digest,
    freeze_execution_code,
    immutable_json,
    locked_ledger,
    spent,
)

ARMS = ('FFM_Selected_Bank_Uniform_FullModel', 'FFM_Selected_Bank_Residual_Repair',
        'Senseiver_Selected_Bank_Uniform_FullModel', 'Senseiver_Selected_Bank_Residual_Repair')
CAP = 130 * 3600
RESERVE = 6 * 3600
CHECKPOINTS = [100, 250, 500, 750, 1000]


def check_config(config, arm, epochs, *, preparation=False):
    if arm not in ARMS:
        raise ValueError('unknown Round 2 arm')
    ffm, residual = arm.startswith('FFM_'), arm.endswith('Residual_Repair')
    opt, coh = config['optimization'], config['coherence']
    for key, value in {'epochs': epochs, 'steps_per_epoch': 12, 'batch_size': 64,
                       'train_fraction': 1.0, 'sampling': 'full_pass'}.items():
        if opt.get(key) != value:
            raise ValueError(f'optimization.{key} must be {value!r}')
    if coh['compute_budget'].get('batch_size') != 64 or coh['compute_budget'].get('point_count') != 4096:
        raise ValueError('coherence support must be B64/Q4096')
    if config['runtime'].get('plot_every_steps') != 10:
        raise ValueError('loss curves must refresh every 10 epochs')
    if bool(config.get('residual', {}).get('enabled', False)) != residual:
        raise ValueError('residual configuration disagrees with arm')
    if config.get('trainable', {}).get('scope') != 'full_model':
        raise ValueError('Round 2 requires the full backbone or complete residual head trainable')
    if config['model']['name'] != ('gl_rbf_cq' if ffm else 'senseiver'):
        raise ValueError('arm source model mismatch')
    if config['evaluation'].get('split') != 'train' or config['evaluation'].get('members') != (4 if ffm else 1):
        raise ValueError('training callbacks require TRAIN and model-specific J4/J1')
    if config['evaluation'].get('sensor_counts') != {'T': 256}:
        raise ValueError('training callbacks require T256')
    fidelity = opt['multitask']['fidelity_budget']
    if fidelity.get('metric') != ('physical_mean_risk_model_units' if ffm else 'physical_member_mse_model_units'):
        raise ValueError('model-specific fidelity metric required')
    if not preparation and not fidelity.get('reference_artifact'):
        raise ValueError('frozen common fidelity reference required')
    if not preparation and not coh.get('scalable_anchored_coherence', {}).get('common_core'):
        raise ValueError('frozen common assessment core required')
    settings = coh['scalable_anchored_coherence']
    if not preparation:
        allocation = settings.get('allocation', {})
        if allocation.get('uniform_epochs') != 10 or allocation.get('mode') not in {'uniform', 'qualified_fixed'}:
            raise ValueError('Round 2 requires ten initial uniform epochs and an explicit allocation disposition')
        if not residual and allocation['mode'] != 'uniform':
            raise ValueError('full-model controls must retain uniform sampling')
    if not preparation and (not settings.get('common_prefit') or not settings.get('adaptive', {}).get('bank_artifact')):
        raise ValueError('common prefit and selected bank are required')
    if ffm and (config['rollout'].get('steps') != 4 or config['rollout'].get('solver') != 'euler'):
        raise ValueError('FFM requires native four-step Euler')
    if ffm and not preparation and opt.get('rollout_execution', {}).get('checkpointing') is not True:
        raise ValueError('FFM qualification and formal execution require rollout checkpointing')
    native_ffm = {'mode': 'endpoint_smooth', 'strength': 1.0, 'sigma': 0.05,
                  'schedule_power': 2.0, 'final_clamp': True, 'chunk_size': 4096}
    if ffm and config['observation_consistency'] != native_ffm:
        raise ValueError('FFM requires the approved native endpoint-smooth observation contract')
    if not ffm and (config['observation_consistency'].get('mode') != 'none' or config['observation_consistency'].get('final_clamp')):
        raise ValueError('Senseiver requires native unclamped sensor behavior')
    if epochs == 1000 and (config['checkpointing'].get('epochs') != CHECKPOINTS or not config['checkpointing'].get('save_epoch_zero')):
        raise ValueError('required checkpoint schedule missing')


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as out:
        out.write(data)


def check_binding(record, campaign):
    path = Path(record['path']).resolve()
    if campaign.resolve() not in path.parents or not path.is_file() or digest(path) != record['sha256']:
        raise ValueError(f'campaign artifact missing or hash differs: {path}')
    return path


def check_code_capsule(campaign):
    if (campaign / 'operations/code').resolve() not in CODE_ROOT.parents:
        raise ValueError('GPU execution requires the frozen campaign code capsule')
    receipt = json.loads((CODE_ROOT / 'execution_receipt.json').read_text())
    files = receipt.get('files', {})
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if not files or receipt.get('schema') != 'car_execution_code_v1' or identity != receipt.get('sha256') or CODE_ROOT.name != identity:
        raise ValueError('invalid frozen code inventory')
    for name, expected in files.items():
        path = (CODE_ROOT / name).resolve()
        if CODE_ROOT not in path.parents or not path.is_file() or digest(path) != expected:
            raise ValueError(f'frozen execution file changed: {name}')
    return identity


def check_formal_code_inventory(manifest, campaign, capsule_sha256):
    inventory = json.loads(check_binding(manifest['artifacts']['code_inventory'], campaign).read_text())
    if inventory.get('schema') != 'car_execution_code_v1' or inventory.get('sha256') != capsule_sha256:
        raise ValueError('formal execution capsule differs from frozen code inventory')


def scheduled_callback_bound(rows):
    """Charge measured worst cases at the formal union of callback epochs."""
    callback_epochs = sorted({1, *range(10, 1001, 10), *CHECKPOINTS, 1000})
    quiet_qualification_epochs = [2, 3, 4, 5, 6, 7, 8, 11]
    callback_max = max(row['callback_and_reporting_seconds'] for row in rows)
    quiet_max = max(rows[epoch - 1]['callback_and_reporting_seconds'] for epoch in quiet_qualification_epochs)
    quiet_count = 1000 - len(callback_epochs)
    total = len(callback_epochs) * callback_max + quiet_count * quiet_max
    return {'method': 'maximum combined qualification callback at each formal plot/checkpoint epoch; maximum quiet reporting elsewhere',
            'formal_epochs': 1000, 'callback_epochs': callback_epochs,
            'callback_epoch_count': len(callback_epochs), 'quiet_epoch_count': quiet_count,
            'qualification_quiet_epochs': quiet_qualification_epochs,
            'maximum_combined_callback_seconds': callback_max,
            'maximum_quiet_reporting_seconds': quiet_max, 'total_seconds': total}


def measured_forecast(campaign, arm_records, timing_paths, *, spent_seconds, remaining_cost):
    """Bind conservative per-arm forecasts to qualified complete epoch journals."""
    if len(timing_paths) != len(ARMS) or not math.isfinite(spent_seconds) or spent_seconds < 0:
        raise ValueError('four timing receipts and nonnegative spent time required')
    cost_path = check_binding(remaining_cost, campaign)
    cost = json.loads(cost_path.read_text())
    components = cost.get('components')
    if cost.get('schema') != 'car_round2_remaining_cost_v1' or not isinstance(components, list) or not components:
        raise ValueError('measured or evidenced remaining cost components required')
    names, remaining_seconds = set(), 0.
    for component in components:
        name, seconds, evidence = component.get('name'), component.get('seconds'), component.get('evidence')
        if (not isinstance(name, str) or not name or name in names or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or seconds < 0 or not isinstance(evidence, list) or not evidence):
            raise ValueError('each remaining cost component requires a unique name, seconds and bound evidence')
        for reference in evidence:
            check_binding(reference, campaign)
        names.add(name)
        remaining_seconds += seconds
    timings, updates, callbacks, outside, cold, callback_evidence = {}, [], [], [], [], {}
    for arm, path in zip(ARMS, timing_paths, strict=True):
        path = Path(path).resolve(strict=True)
        record = {'path': str(path), 'sha256': digest(path)}
        check_binding(record, campaign)
        timing = json.loads(path.read_text())
        config_sha = arm_records[arm]['config']['sha256']
        expected_kind = 'residual_head' if arm.endswith('Residual_Repair') else 'full_model'
        if (timing.get('schema') != 'car_round2_complete_update_timing_v1'
                or timing.get('config_sha256') != config_sha or timing.get('parameterization') != expected_kind
                or timing.get('batch_size') != 64 or timing.get('selected_epochs') != [2, 3, 4, 5, 6, 7, 8, 11]):
            raise ValueError(f'{arm}: timing does not bind its qualified formal config')
        for key in ('qualification_receipt', 'pause_launch', 'resume_launch', 'pause_completion', 'resume_completion', 'epoch_timing'):
            check_binding({'path': timing[f'{key}_path'], 'sha256': timing[f'{key}_sha256']}, campaign)
        qualification = json.loads(Path(timing['qualification_receipt_path']).read_text())
        if (qualification.get('arm') != arm or qualification.get('passed') is not True
                or qualification.get('formal_config_sha256') != config_sha
                or qualification.get('final_step') != 144 or qualification.get('pause_step') != 108
                or qualification.get('pause_launch_sha256') != timing['pause_launch_sha256']
                or qualification.get('resume_launch_sha256') != timing['resume_launch_sha256']
                or qualification.get('pause_completion_sha256') != timing['pause_completion_sha256']):
            raise ValueError(f'{arm}: matching 12-epoch qualification required')
        rows = [json.loads(line) for line in Path(timing['epoch_timing_path']).read_text().splitlines() if line.strip()]
        if len(rows) != 12 or rows != timing.get('per_epoch') or [r.get('epoch') for r in rows] != list(range(1, 13)):
            raise ValueError(f'{arm}: complete measured epoch journal required')
        for row in rows:
            values = [row.get(key) for key in ('train_seconds', 'wall_seconds', 'callback_and_reporting_seconds')]
            if (row.get('version') != 'complete_epoch_timing_v1' or row.get('accepted_updates') != 12
                    or row.get('step') != row['epoch'] * 12
                    or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in values)
                    or values[0] <= 0 or not math.isclose(values[1] - values[0], values[2], rel_tol=1e-9, abs_tol=1e-9)):
                raise ValueError(f'{arm}: invalid measured epoch timing')
        update = max(rows[i - 1]['train_seconds'] / 12 for i in timing['selected_epochs'])
        if not math.isclose(update, timing.get('complete_update_seconds', -1), rel_tol=1e-12):
            raise ValueError(f'{arm}: complete update time differs from its journal')
        callback_evidence[arm] = scheduled_callback_bound(rows)
        callbacks.append(callback_evidence[arm]['total_seconds'] / 1000)
        walls = []
        for phase, expected in (('pause', ('budget_paused', 108)), ('resume', ('completed', 144))):
            launch = json.loads(Path(timing[f'{phase}_launch_path']).read_text())
            completion = json.loads(Path(timing[f'{phase}_completion_path']).read_text())
            start, end = launch.get('started_unix'), completion.get('finished_unix')
            if (launch.get('arm') != arm or not timing.get('gpu_uuid') or launch.get('gpu_uuid') != timing['gpu_uuid']
                    or completion.get('token') != launch.get('token')
                    or (completion.get('status'), completion.get('global_step')) != expected
                    or Path(completion.get('run_dir', '')).resolve() != Path(qualification['resumed']).resolve()
                    or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                    or not math.isfinite(start) or not math.isfinite(end) or end <= start):
                raise ValueError(f'{arm}: measured qualification lease intervals differ')
            walls.append(end - start)
        overhead = max(0., sum(walls) - sum(row['wall_seconds'] for row in rows))
        excess = sum(max(0., rows[epoch - 1]['train_seconds'] - 12 * update) for epoch in (1, 10))
        if (timing.get('lease_wall_seconds') != walls
                or not math.isclose(timing.get('outside_epoch_seconds', -1), overhead, rel_tol=1e-12, abs_tol=1e-9)
                or not math.isclose(timing.get('cold_train_excess_seconds', -1), excess, rel_tol=1e-12, abs_tol=1e-9)):
            raise ValueError(f'{arm}: startup/closure measurement differs from lease and epoch records')
        outside.append(overhead)
        cold.append(excess)
        updates.append(update)
        timings[arm] = record
    total = spent_seconds + sum(updates) * 12000 + sum(callbacks) * 1000 + sum(outside) + sum(cold) + remaining_seconds
    return {'schema': 'car_round2_forecast_v2', 'spent_seconds': spent_seconds,
            'per_arm_update_seconds': updates, 'per_arm_callback_seconds_per_epoch': callbacks,
            'per_arm_outside_epoch_seconds': outside, 'per_arm_cold_train_excess_seconds': cold,
            'timing_receipts': timings, 'remaining_seconds': remaining_seconds, 'remaining_cost': remaining_cost,
            'callback_estimation': 'schedule-aware maximum combined callback and maximum quiet reporting wall per arm',
            'per_arm_callback_schedule_evidence': callback_evidence,
            'projected_seconds': total, 'cap_seconds': CAP, 'reserve_seconds': RESERVE,
            'fits': total + RESERVE <= CAP}


def common_pair_view(config):
    """Remove only the explicitly permitted within-source arm differences."""
    view = copy.deepcopy(config)
    view.pop('residual', None)
    view.pop('trainable', None)
    view['optimization'].pop('lr', None)
    view['optimization'].pop('scheduler', None)
    view['optimization'].pop('lr_warmup', None)
    view['runtime'].pop('device', None)
    view['output'].pop('experiment_name', None)
    view['coherence']['scalable_anchored_coherence']['allocation'].pop('mode', None)
    return view


def check_source_pair(configs, family):
    prefix = 'FFM' if family == 'ffm' else 'Senseiver'
    full = configs[f'{prefix}_Selected_Bank_Uniform_FullModel']
    residual = configs[f'{prefix}_Selected_Bank_Residual_Repair']
    if common_pair_view(full) != common_pair_view(residual):
        raise ValueError(f'{family}: matched full-model and residual scientific targets or clocks differ')


def check_common_core_binding(common_core, observer, campaign):
    if (Path(common_core['path']).resolve() != Path(observer['path']).resolve()
        or common_core['sha256'] != observer['sha256']):
        raise ValueError('common core differs from frozen assessment observer')
    check_binding({'path': common_core['source_panel_path'],
                   'sha256': common_core['source_panel_sha256']}, campaign)


def check_assessment_contracts(artifacts, campaign, configs):
    """Bind assessment observers and truth-only normalization before training."""
    task = artifacts['reserved_task']['sha256']
    panel = json.loads(check_binding(artifacts['train_panel'], campaign).read_text())
    evidence = json.loads(check_binding(artifacts['reserved_task_normalizer'], campaign).read_text())
    value = evidence.get('value')
    if (panel.get('schema') != 'car_round2_panel_v1' or panel.get('split') != 'train'
        or evidence.get('schema') != 'car_round2_reserved_normalizer_v1'
        or evidence.get('split') != 'train' or evidence.get('reserved_model_scores_used') is not False
        or evidence.get('task_sha256') != task or evidence.get('panel_sha256') != artifacts['train_panel']['sha256']
        or evidence.get('sample_ids') != panel.get('sample_ids')
        or isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0):
        raise ValueError('reserved task normalization must bind positive TRAIN-truth evidence')
    for family in ('ffm', 'senseiver'):
        doc = json.loads(check_binding(artifacts[f'assessment_contract_{family}'], campaign).read_text())
        expected = {'source_checkpoint_sha256': artifacts[f'source_{family}']['sha256'],
                    'core_sha256': artifacts[f'observer_{family}']['sha256'],
                    'fidelity_reference_sha256': artifacts[f'fidelity_{family}']['sha256'],
                    'reserved_task_sha256': task, 'event_sha256': artifacts['development_event']['sha256'],
                    'train_panel_sha256': artifacts['train_panel']['sha256'],
                    'reserved_task_normalizer_evidence_sha256': artifacts['reserved_task_normalizer']['sha256']}
        if (doc.get('schema') != 'car_round2_assessment_contract_v2' or doc.get('family') != family
            or any(doc.get(key) != digest_value for key, digest_value in expected.items())
            or doc.get('sample_ids') != panel.get('sample_ids') or doc.get('reserved_model_scores_used') is not False
            or doc.get('reserved_task_normalizer') != value
            or doc.get('reserved_task_normalization_rule') != {
                'primary': 'final_train_source_task_score', 'fallback': 'train_truth_absolute_covariance',
                'near_zero_threshold': 1e-12 * value}):
            raise ValueError(f'{family}: assessment contract does not bind the frozen source and tasks')
        normalizers = doc.get('normalizers', {})
        if (set(normalizers) != {'state', 'scale', 'shape', 'event'}
            or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0
                   for v in normalizers.values())):
            raise ValueError(f'{family}: assessment requires four positive SOURCE normalizers')
        prefix = 'FFM' if family == 'ffm' else 'Senseiver'
        field_count = len(configs[f'{prefix}_Selected_Bank_Uniform_FullModel']['dataset']['field_names'])
        for key in ('mean_allowance_physical', 'sensor_equality_tolerance'):
            values = doc.get(key)
            if (not isinstance(values, list) or len(values) != field_count or
                any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 for v in values)):
                raise ValueError(f'{family}: invalid assessment {key}')
        if (isinstance(doc.get('shape_query_seed'), bool) or not isinstance(doc.get('shape_query_seed'), int)
            or not isinstance(doc.get('source_bank_sha256'), str) or len(doc['source_bank_sha256']) != 64):
            raise ValueError(f'{family}: assessment observer and source bank identities required')


def check_allocation_disposition(artifacts, campaign, configs):
    """Reject policy claims without both exact matched parameter-gradient audits."""
    disposition = json.loads(check_binding(artifacts['allocation_disposition'], campaign).read_text())
    mode = disposition.get('mode')
    if disposition.get('schema') != 'car_round2_allocation_disposition_v1' or mode not in {'uniform', 'qualified_fixed'}:
        raise ValueError('invalid Round 2 allocation disposition')
    profiles = disposition.get('profiles', {})
    if not isinstance(profiles, dict) or set(profiles) - {'ffm', 'senseiver'}:
        raise ValueError('allocation profiles must be named by source family')
    if mode == 'qualified_fixed' and set(profiles) != {'ffm', 'senseiver'}:
        raise ValueError('qualified allocation requires measured profiles for both source families')
    for family in ('ffm', 'senseiver'):
        prefix = 'FFM' if family == 'ffm' else 'Senseiver'
        full = configs[f'{prefix}_Selected_Bank_Uniform_FullModel']
        residual = configs[f'{prefix}_Selected_Bank_Residual_Repair']
        alloc = residual['coherence']['scalable_anchored_coherence']['allocation']
        if full['coherence']['scalable_anchored_coherence']['allocation']['mode'] != 'uniform' or alloc['mode'] != mode:
            raise ValueError(f'{family}: arm allocation disagrees with frozen disposition')
        if family in profiles:
            check_binding(profiles[family], campaign)
        if mode == 'uniform':
            continue
        profile = json.loads(Path(profiles[family]['path']).read_text())
        bank = json.loads(check_binding(artifacts[f'bank_{family}'], campaign).read_text())
        base = {k: v for k, v in bank.items() if k != 'sha256' and not k.startswith('policy_')}
        base['policy_qualified'] = False
        base_sha = hashlib.sha256(json.dumps(base, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        core = residual['coherence']['scalable_anchored_coherence']['common_core']
        candidate = profile.get('policies', {}).get('lagged_gradient_cost', {})
        uniform = profile.get('policies', {}).get('uniform', {})
        probabilities = candidate.get('pair_probabilities')
        k = len(residual['dataset']['field_names']) * (len(residual['dataset']['field_names']) - 1) // 2
        valid_probabilities = (isinstance(probabilities, list) and len(probabilities) == k and k > 0 and
                               all(isinstance(p, (int, float)) and not isinstance(p, bool) and math.isfinite(p) and p >= .25 / k for p in probabilities) and
                               math.isclose(sum(probabilities), 1., rel_tol=1e-9, abs_tol=1e-9))
        ratio, overhead = profile.get('gradient_cost_uniform_ratio'), profile.get('overhead_fraction')
        base_cost, candidate_cost = uniform.get('mse_cost'), candidate.get('mse_cost')
        population, sample_ids = profile.get('train_population'), profile.get('sample_ids')
        if (profile.get('schema') != 'car_round2_allocator_job_v1' or profile.get('status') != 'completed'
            or profile.get('source_family') != family or profile.get('parameterization') != 'residual_head'
            or profile.get('qualified_one_path') is not True or profile.get('selected_bank_sha256') != base_sha
            or isinstance(population, bool) or not isinstance(population, int) or population < 1
            or not isinstance(sample_ids, list) or len(sample_ids) != max(1, math.ceil(.1 * population))
            or len(set(sample_ids)) != len(sample_ids)
            or isinstance(profile.get('actual_head_parameters'), bool) or not isinstance(profile.get('actual_head_parameters'), int)
            or profile['actual_head_parameters'] < 1 or isinstance(profile.get('pair_draws'), bool)
            or not isinstance(profile.get('pair_draws'), int) or profile['pair_draws'] < 1
            or profile.get('source_panel_sha256') != core['source_panel_sha256']
            or profile.get('core_sha256') != artifacts[f'observer_{family}']['sha256']
            or profile.get('prefit_sha256') != artifacts[f'coefficients_{family}']['sha256']
            or profile.get('sample_ids') != json.loads(Path(artifacts[f'coefficients_{family}']['path']).read_text())['binding']['sample_ids']
            or not valid_probabilities or not isinstance(ratio, (float, int)) or isinstance(ratio, bool) or not math.isfinite(ratio) or not 0 <= ratio <= .8
            or not isinstance(overhead, (float, int)) or isinstance(overhead, bool) or not math.isfinite(overhead) or not 0 <= overhead <= .05
            or not isinstance(base_cost, (float, int)) or not math.isfinite(base_cost) or base_cost <= 0
            or not isinstance(candidate_cost, (float, int)) or not math.isfinite(candidate_cost) or candidate_cost < 0
            or not math.isclose(ratio, candidate_cost / base_cost, rel_tol=1e-8, abs_tol=1e-12)
            or bank.get('policy_qualified') is not True or bank.get('policy_mode') != 'fixed_source_profile'
            or bank.get('policy_profile_sha256') != profiles[family]['sha256']
            or bank.get('policy_pair_probabilities') != probabilities):
            raise ValueError(f'{family}: allocation profile, target, probabilities or measured qualification differs')


def check_freeze(manifest, campaign):
    import yaml
    if manifest.get('schema') != 'car_round2_freeze_v1' or manifest.get('status') != 'frozen':
        raise ValueError('invalid freeze schema/status')
    if set(manifest.get('arms', {})) != set(ARMS):
        raise ValueError('all four arms are required')
    required = {'source_ffm', 'source_senseiver', 'train_panel', 'validation_panel', 'reserved_task',
                'observer_ffm', 'observer_senseiver', 'bank_ffm', 'bank_senseiver',
                'coefficients_ffm', 'coefficients_senseiver', 'fidelity_ffm', 'fidelity_senseiver',
                'assessment_contract_ffm', 'assessment_contract_senseiver',
                'reserved_task_normalizer', 'development_event',
                'allocation_disposition', 'forecast', 'code_inventory'}
    artifacts = manifest.get('artifacts', {})
    if required - artifacts.keys():
        raise ValueError(f'missing freeze artifacts: {sorted(required - artifacts.keys())}')
    for value in artifacts.values():
        check_binding(value, campaign)
    configs = {}
    for arm in ARMS:
        binding = manifest['arms'][arm]
        config_path = check_binding(binding['config'], campaign)
        config = yaml.safe_load(config_path.read_text())
        configs[arm] = config
        check_config(config, arm, 1000)
        family = 'ffm' if arm.startswith('FFM_') else 'senseiver'
        source = campaign / 'sources' / family
        if Path(config['source_run']).resolve() != source.resolve():
            raise ValueError(f'{arm}: source snapshot differs')
        if config.get('source', {}).get('weight_view') != ('ema_materialized' if family == 'ffm' else 'live'):
            raise ValueError(f'{arm}: source weight view differs')
        settings = config['coherence']['scalable_anchored_coherence']
        common_core = settings['common_core']
        observer = artifacts[f'observer_{family}']
        check_common_core_binding(common_core, observer, campaign)
        declared = {'bank': settings['adaptive']['bank_artifact'],
                    'coefficients': settings['common_prefit'],
                    'fidelity': config['optimization']['multitask']['fidelity_budget']['reference_artifact']}
        for name, value in declared.items():
            expected = artifacts[f'{name}_{family}']
            if Path(value['path']).resolve() != Path(expected['path']).resolve() or value['sha256'] != expected['sha256']:
                raise ValueError(f'{arm}: {name} differs from source-pair artifact')
        qualification = check_binding(binding['qualification'], campaign)
        q = json.loads(qualification.read_text())
        if q.get('schema') != 'car_round2_qualification_v1' or q.get('arm') != arm or q.get('passed') is not True:
            raise ValueError(f'{arm}: passing 12-epoch fresh-process qualification required')
        if q.get('pause_step') != 108 or q.get('final_step') != 144 or q.get('formal_config_sha256') != binding['config']['sha256']:
            raise ValueError(f'{arm}: qualification does not bind formal configuration and clock')
    for family in ('ffm', 'senseiver'):
        check_source_pair(configs, family)
        from phycoflow_reconstruction.coherence.adaptive_fitting import (
            validate_bank,
            validate_bank_target_binding,
        )

        bank = json.loads(Path(artifacts[f'bank_{family}']['path']).read_text())
        prefit = json.loads(Path(artifacts[f'coefficients_{family}']['path']).read_text())
        binding = prefit.get('binding', {})
        prefix = 'FFM' if family == 'ffm' else 'Senseiver'
        field_order = configs[f'{prefix}_Selected_Bank_Uniform_FullModel']['dataset']['field_names']
        if (prefit.get('version') != 'common_prefit_calibration_v1'
            or binding.get('source_checkpoint_sha256') != artifacts[f'source_{family}']['sha256']
            or binding.get('field_order') != field_order):
            raise ValueError(f'{family}: prefit source or field order differs')
        validate_bank(bank, field_order=field_order, support_identity=binding.get('support_sha256'),
                      source_identity=artifacts[f'source_{family}']['sha256'])
        validate_bank_target_binding(binding, bank)
    check_assessment_contracts(artifacts, campaign, configs)
    check_allocation_disposition(artifacts, campaign, configs)
    forecast = json.loads(check_binding(artifacts['forecast'], campaign).read_text())
    if (forecast.get('schema') != 'car_round2_forecast_v2' or forecast.get('fits') is not True
            or forecast.get('cap_seconds') != CAP or forecast.get('reserve_seconds', 0) < RESERVE):
        raise ValueError('complete measured forecast with protected closeout is required')
    timing_paths = [check_binding(forecast['timing_receipts'][arm], campaign) for arm in ARMS]
    for arm, path in zip(ARMS, timing_paths, strict=True):
        timing = json.loads(path.read_text())
        if timing['qualification_receipt_sha256'] != manifest['arms'][arm]['qualification']['sha256']:
            raise ValueError(f'{arm}: forecast timing qualification differs from formal qualification')
    measured = measured_forecast(campaign, manifest['arms'], timing_paths,
                                 spent_seconds=forecast['spent_seconds'], remaining_cost=forecast['remaining_cost'])
    if measured != forecast:
        raise ValueError('forecast differs from its measured qualification timings')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('freeze-code')
    cfg = sub.add_parser('prepare-config')
    cfg.add_argument('--template', type=Path, required=True)
    cfg.add_argument('--arm', choices=ARMS, required=True)
    cfg.add_argument('--qualification', action='store_true')
    cfg.add_argument('--preparation', action='store_true', help='initial source fit without updates')
    cfg.add_argument('--output', type=Path, required=True)
    fr = sub.add_parser('freeze'); fr.add_argument('--manifest', type=Path, required=True)
    forecast = sub.add_parser('forecast')
    forecast.add_argument('--arm-configs', type=Path, required=True)
    forecast.add_argument('--timing-receipts', type=Path, nargs=4, required=True, help='FFM full/residual, then Senseiver full/residual')
    forecast.add_argument('--remaining-cost', type=Path, required=True, help='evidenced remaining assessment and closeout costs')
    forecast.add_argument('--output', type=Path, required=True)
    run = sub.add_parser('run')
    run.add_argument('--arm', choices=ARMS, required=True)
    run.add_argument('--config', type=Path, required=True)
    run.add_argument('--device', required=True, help='runtime physical GPU UUID or index')
    run.add_argument('--qualification', action='store_true')
    run.add_argument('--prepare-only', action='store_true', help='fit source artifacts without updates')
    run.add_argument('--resume', type=Path)
    run.add_argument('--pause-at-epoch', type=int, choices=[9])
    run.add_argument('--save-margin-seconds', type=float, required=True)
    run.add_argument('--initial-stage-bound-seconds', type=float, required=True)
    run.add_argument('--max-seconds', type=float, help='stop at an update boundary before this child wall-time limit')
    run.add_argument('--launch-receipt-output', type=Path, help='unique campaign-owned copy of the launch receipt')
    args = p.parse_args()
    campaign = args.campaign.resolve(strict=True)
    if not campaign.is_dir() or '/data2/wanglz/PhyCoFlow-MultiFieldReconstruction/' not in str(campaign) + '/':
        raise ValueError('campaign must resolve to the established DATA2 project root')
    if args.command == 'freeze-code':
        capsule = freeze_execution_code(ROOT, campaign / 'operations/code',
                                        ['scripts/posttraining/run_coherence_adaptive_repair_round2.py',
                                         'scripts/posttraining/qualify_car_round2_resume.py',
                                         'scripts/posttraining/sequence_car_round2_qualification.py',
                                         'scripts/posttraining/sequence_car_round2_formal.py',
                                         'scripts/posttraining/sequence_car_round2_assessment.py',
                                         'scripts/posttraining/assess_car_round2_checkpoints.py',
                                         'scripts/posttraining/freeze_car_event.py',
                                         'scripts/analysis/fit_car_round2_prefit.py',
                                         'scripts/analysis/fit_car_round2_candidates.py',
                                         'scripts/analysis/assemble_car_round2.py',
                                         'scripts/analysis/finalize_car_round2_bank.py',
                                         'scripts/analysis/reconcile_car_round2_costs.py',
                                         'scripts/analysis/assemble_car_round2_cost_manifest.py',
                                         'scripts/analysis/summarize_car_round2_timing.py',
                                         'scripts/analysis/freeze_car_round2_task.py',
                                         'scripts/smoke/car_round2_source_residual.py',
                                         'scripts/smoke/car_round2_ffm_checkpointing.py',
                                         'scripts/smoke/car_round2_native_head_chunk.py',
                                         'scripts/benchmarks/profile_car_round2_allocator.py',
                                         'scripts/benchmarks/profile_car_allocator.py'])
        print(capsule)
    elif args.command == 'prepare-config':
        import yaml
        if args.preparation and args.qualification:
            raise ValueError('preparation and qualification are distinct attempts')
        c = yaml.safe_load(args.template.read_text())
        family = 'ffm' if args.arm.startswith('FFM_') else 'senseiver'
        source = campaign / 'sources' / family
        identity = json.loads((source / 'identity.json').read_text())
        for name, expected in identity['files'].items():
            if digest(source / name) != expected:
                raise ValueError(f'source snapshot changed: {name}')
        base = yaml.safe_load((source / 'resolved_config.yaml').read_text())
        measure = c['dataset'].get('physical_measure')
        for section in ('dataset', 'model', 'observations'):
            c[section] = base[section]
        c['dataset']['physical_measure'] = measure
        c['inherit_base_config'] = False
        c['source_run'] = str(source)
        c['source_checkpoint'] = 'last.pt'
        expected_view = 'ema' if family == 'ffm' else 'live'
        if identity['weight_view'] != expected_view:
            raise ValueError('source identity declares an unexpected weight view')
        c['source'] = {'kind': 'native_run', 'admission': 'frozen_snapshot',
                       'snapshot_receipt': str(source / 'snapshot_receipt.json'),
                       'weight_view': 'ema_materialized' if family == 'ffm' else 'live',
                       'allow_integration_source': False}
        if not (source / 'snapshot_receipt.json').is_file():
            raise ValueError('source snapshot admission receipt missing')
        if family == 'ffm':
            c['rollout'] = {'steps': 4, 'solver': 'euler'}
        else:
            c['observation_consistency'] = {'mode': 'none', 'final_clamp': False}
        c['optimization'].update(epochs=12 if args.qualification else 1000, steps_per_epoch=12,
                                 batch_size=64, train_fraction=1.0, sampling='full_pass')
        c['coherence']['compute_budget'].update(batch_size=64, point_count=4096)
        c['runtime']['plot_every_steps'] = 10
        c['coherence']['scalable_anchored_coherence']['allocation'] = {'mode': 'uniform', 'uniform_epochs': 10}
        if args.preparation:
            # Initial SOURCE fitting has no already-fitted shared artifacts.
            settings = c['coherence']['scalable_anchored_coherence']
            for key in ('common_core', 'common_prefit', 'adaptive'):
                settings.pop(key, None)
            settings['save_source_panel'] = True
            fidelity = c['optimization']['multitask']['fidelity_budget']
            fidelity.pop('reference_artifact', None)
            if family == 'ffm':
                fidelity.pop('relative_increase', None)
                fidelity['metric'] = 'physical_mean_risk_model_units'
                fidelity['member_allowance_fraction'] = 0.10
                # These are numerical optimization scales, not the scientific allowance.
                fidelity['mean_risk_scales'] = [1.0] * len(c['dataset']['field_names'])
                fidelity.pop('absolute_allowance', None)
            else:
                fidelity.pop('mean_risk_scales', None)
                fidelity.pop('member_allowance_fraction', None)
                fidelity['metric'] = 'physical_member_mse_model_units'
                fidelity['relative_increase'] = 0.10
        phase = 'pilot' if args.preparation else ('qualification' if args.qualification else 'formal')
        c['output']['experiment_name'] = f'coherence_adaptive_repair/round_2/runs/{phase}/{args.arm}'
        c['evaluation']['split'] = 'train'
        c['evaluation']['members'] = 4 if args.arm.startswith('FFM_') else 1
        c['evaluation']['sensor_counts'] = {'T': 256}
        c['checkpointing']['epochs'] = [9, 10, 12] if args.qualification else CHECKPOINTS
        c['checkpointing']['save_epoch_zero'] = True
        check_config(c, args.arm, 12 if args.qualification else 1000, preparation=args.preparation)
        output = args.output.resolve()
        if campaign not in output.parents:
            raise ValueError('config must be inside campaign')
        write_new(output, yaml.safe_dump(c, sort_keys=False))
        print(output)
    elif args.command == 'forecast':
        assembly = json.loads(args.arm_configs.read_text())
        if assembly.get('schema') != 'car_round2_assembled_arms_v1':
            raise ValueError('assembled arm configurations required')
        arms = {arm: {'config': assembly['arms'][arm]} for arm in ARMS}
        for record in arms.values():
            check_binding(record['config'], campaign)
        with locked_ledger(campaign / 'operations/gpu_ledger.json') as state:
            used = spent(state)
        cost_path = args.remaining_cost.resolve(strict=True)
        result = measured_forecast(campaign, arms, args.timing_receipts, spent_seconds=used,
                                   remaining_cost={'path': str(cost_path), 'sha256': digest(cost_path)})
        output = args.output.resolve()
        if campaign not in output.parents:
            raise ValueError('forecast output must belong to campaign')
        immutable_json(output, result)
        print(json.dumps(result, indent=2))
        return 0 if result['fits'] else 2
    elif args.command == 'freeze':
        doc = check_freeze(json.loads(args.manifest.read_text()), campaign)
        immutable_json(campaign / 'configs/formal_freeze.json', {'schema': 'car_round2_freeze_receipt_v1',
                       'manifest_sha256': digest(args.manifest), 'manifest': doc})
    elif args.command == 'run':
        import yaml
        capsule_sha256 = check_code_capsule(campaign)
        if args.prepare_only and (args.qualification or args.resume or args.pause_at_epoch):
            raise ValueError('preparation requires a fresh, non-qualification attempt')
        config = yaml.safe_load(args.config.read_text())
        check_config(config, args.arm, 12 if args.qualification else 1000, preparation=args.prepare_only)
        phase = 'pilot' if args.prepare_only else ('qualification' if args.qualification else 'formal')
        expected_output = f'coherence_adaptive_repair/round_2/runs/{phase}/{args.arm}'
        if config['output'].get('experiment_name') != expected_output:
            raise ValueError('attempt output must belong to the campaign arm and phase')
        if args.resume and (campaign / 'runs' / phase / args.arm).resolve() not in args.resume.resolve().parents:
            raise ValueError('resume must belong to the same campaign arm and phase')
        if any(not math.isfinite(v) or v <= 0 for v in (args.save_margin_seconds, args.initial_stage_bound_seconds)):
            raise ValueError('positive finite safe-save and initial-stage bounds required')
        if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= args.save_margin_seconds
                                             or args.initial_stage_bound_seconds > args.max_seconds - args.save_margin_seconds):
            raise ValueError('child wall bound must leave the admitted startup and safe-save margin')
        if args.launch_receipt_output is not None:
            launch_copy = args.launch_receipt_output.resolve()
            if campaign not in launch_copy.parents or launch_copy.exists():
                raise ValueError('launch receipt copy must be a new campaign-owned path')
        if not args.qualification and not args.prepare_only:
            if args.pause_at_epoch:
                raise ValueError('qualification pause only')
            doc = json.loads((campaign / 'configs/formal_freeze.json').read_text())['manifest']
            check_freeze(doc, campaign)
            check_formal_code_inventory(doc, campaign, capsule_sha256)
            if doc['arms'][args.arm]['config']['sha256'] != digest(args.config):
                raise ValueError('config differs from frozen arm')
        physical = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
        devices = [tuple(x.strip() for x in row.split(',')) for row in physical.splitlines()]
        matches = [u for i, u in devices if args.device in (i, u)]
        if len(matches) != 1:
            raise ValueError('device must identify one available physical GPU')
        device = matches[0]
        runtime = json.loads((campaign / 'operations/runtime.json').read_text())
        index = next(int(i) for i, u in devices if u == device)
        permitted = runtime.get('user_authorized_gpu_indices')
        if permitted is not None and index not in permitted:
            raise ValueError('device excluded by current runtime configuration')
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
        if any(row.split(',')[0].strip() == device for row in apps.splitlines() if ',' in row):
            raise ValueError('requested GPU already has compute processes')
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        cache = campaign / 'cache/keops' / device
        cache.mkdir(parents=True, exist_ok=True)
        # KeOps queries CUDA headers with NamedTemporaryFile during import,
        # before it creates its device-specific compiler directory itself.
        build_name = '_'.join(platform.uname()[:3]) + f'_p{sys.version.split()[0]}'
        (cache / f'{build_name}_CUDA_VISIBLE_DEVICES_{device}').mkdir(exist_ok=True)
        os.environ['KEOPS_CACHE_FOLDER'] = str(cache)
        config['runtime']['device'] = 'cuda:0'
        token, start = str(uuid.uuid4()), time.time()
        ledger = campaign / 'operations/gpu_ledger.json'
        with locked_ledger(ledger) as state:
            if spent(state) + args.initial_stage_bound_seconds + args.save_margin_seconds >= CAP - RESERVE:
                raise ValueError('insufficient remaining budget for safe save and closeout')
            if any(v['gpu_uuid'] == device for v in state['active'].values()):
                raise ValueError('device already leased by campaign')
            state['active'][token] = {'gpu_uuid': device, 'phase': 'preparation' if args.prepare_only else ('qualification' if args.qualification else 'formal'), 'start_unix': start}
        receipt = {'schema': 'car_round2_launch_v1', 'token': token, 'arm': args.arm, 'gpu_uuid': device,
                   'started_unix': start, 'pid': os.getpid(), 'config_sha256': digest(args.config),
                   'driver_sha256': digest(Path(__file__)), 'resume': str(args.resume) if args.resume else None,
                   'code_capsule_sha256': capsule_sha256,
                   'pause_at_epoch': args.pause_at_epoch,
                   'initial_stage_bound_seconds': args.initial_stage_bound_seconds,
                   'platform': platform.platform()}
        atomic_json(campaign / 'operations/launches' / f'{token}.json', receipt)
        if args.launch_receipt_output is not None:
            immutable_json(launch_copy, receipt)
        previous_alarm = signal.getsignal(signal.SIGALRM)
        def expired(_signum, _frame):
            raise TimeoutError('initial preparation/update exceeded the admitted time bound')
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(max(1, math.ceil(args.initial_stage_bound_seconds - (time.time() - start))))
        try:
            from phycoflow_reconstruction.training.post_training import run_post_training
            from phycoflow_reconstruction.training.run_store import load_project_checkpoint
            initial_step = load_project_checkpoint(args.resume / 'checkpoints/last.pt')['global_step'] if args.resume else 0
            def stop(step, run_dir):
                if step > initial_step:
                    signal.alarm(0)
                with locked_ledger(ledger) as state:
                    out_of_budget = spent(state) + args.save_margin_seconds >= CAP - RESERVE
                deadline = args.max_seconds is not None and time.time() >= start + args.max_seconds - args.save_margin_seconds
                return out_of_budget or deadline or bool(args.pause_at_epoch and step >= args.pause_at_epoch * 12)
            path = run_post_training(config, case_dir=ROOT / 'cases/turbulent_combustion', resume=args.resume,
                                     budget_stop_requested=stop, prepare_only=args.prepare_only)
            status = json.loads((path / 'status.json').read_text())
            outcome = {'status': status.get('status'), 'global_step': status.get('global_step'), 'run_dir': str(path)}
            print(path)
        except BaseException as error:
            outcome = {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)}
            raise
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_alarm)
            finished = time.time()
            atomic_json(campaign / 'operations/completions' / f'{token}.json', {'token': token, 'finished_unix': finished, **outcome})
            with locked_ledger(ledger) as state:
                entry = state['active'].pop(token)
                state['closed'].append({**entry, 'token': token, 'end_unix': finished, 'seconds': finished - entry['start_unix']})
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
