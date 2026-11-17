"""Admission and resume-comparison checks for the Round 2 campaign."""
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / 'scripts' / 'posttraining' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_state_comparison_detects_optimizer_and_order_changes():
    compare = load('qualify_car_round2_resume').differences
    source = {'optimizer': {'momentum': torch.tensor([1., 2.])}, 'columns': ['loss', 'risk']}
    same = {'optimizer': {'momentum': torch.tensor([1., 2.])}, 'columns': ['loss', 'risk']}
    assert not compare(source, same)
    same['optimizer']['momentum'][0] = 3
    same['columns'].reverse()
    failures = compare(source, same)
    assert any('momentum' in path for path in failures)
    assert any('columns' in path for path in failures)


def test_qualification_compares_scientific_state_without_demanding_identical_wall_times():
    module = load('qualify_car_round2_resume')
    first = {'optimizer': {'moment': torch.tensor([1.])}, 'scalable_runtime': {'policy_seconds': .1, 'occupied_seconds': 1., 'step': 144}}
    second = {'optimizer': {'moment': torch.tensor([1.])}, 'scalable_runtime': {'policy_seconds': .2, 'occupied_seconds': 2., 'step': 144}}
    assert module.differences(module.scientific_checkpoint_view(first), module.scientific_checkpoint_view(second)) == []
    second['optimizer']['moment'].add_(1)
    assert module.differences(module.scientific_checkpoint_view(first), module.scientific_checkpoint_view(second))
    definitions = {'columns': [{'name': 'update_seconds', 'units': 'seconds'}, {'name': 'loss', 'units': 'objective'}]}
    assert module.scientific_history_view([{'update_seconds': .2, 'loss': 3.}], definitions) == [{'loss': 3.}]


def test_freeze_does_not_accept_missing_evidence(tmp_path):
    driver = load('run_coherence_adaptive_repair_round2')
    with pytest.raises(ValueError, match='all four arms'):
        driver.check_freeze({'schema': 'car_round2_freeze_v1', 'status': 'frozen', 'arms': {}}, tmp_path)
    with pytest.raises(ValueError, match='missing freeze artifacts'):
        driver.check_freeze({'schema': 'car_round2_freeze_v1', 'status': 'frozen',
                             'arms': {arm: {} for arm in driver.ARMS}}, tmp_path)


def test_foreign_or_changed_artifact_is_rejected(tmp_path):
    driver = load('run_coherence_adaptive_repair_round2')
    inside = tmp_path / 'file'
    inside.write_text('actual')
    with pytest.raises(ValueError, match='hash differs'):
        driver.check_binding({'path': str(inside), 'sha256': '0' * 64}, tmp_path)
    with pytest.raises(ValueError, match='hash differs'):
        driver.check_binding({'path': str(tmp_path.parent / 'elsewhere'), 'sha256': '0' * 64}, tmp_path)


def test_formal_capsule_must_match_frozen_inventory(tmp_path):
    driver = load('run_coherence_adaptive_repair_round2')
    path = tmp_path / 'inventory.json'
    path.write_text(json.dumps({'schema': 'car_execution_code_v1', 'sha256': 'capsule'}))
    manifest = {'artifacts': {'code_inventory': {'path': str(path), 'sha256': driver.digest(path)}}}
    driver.check_formal_code_inventory(manifest, tmp_path, 'capsule')
    with pytest.raises(ValueError, match='differs from frozen'):
        driver.check_formal_code_inventory(manifest, tmp_path, 'other')


def basic_ffm_config():
    return {'model': {'name': 'gl_rbf_cq'}, 'dataset': {'field_names': ['a', 'b', 'c']},
            'trainable': {'scope': 'full_model'},
            'optimization': {'epochs': 12, 'steps_per_epoch': 12, 'batch_size': 64,
                             'train_fraction': 1., 'sampling': 'full_pass', 'lr': .1,
                             'rollout_execution': {'checkpointing': True},
                             'multitask': {'fidelity_budget': {'metric': 'physical_mean_risk_model_units', 'reference_artifact': {'path': 'ref'}}}},
            'coherence': {'compute_budget': {'batch_size': 64, 'point_count': 4096},
                          'scalable_anchored_coherence': {'common_core': {'path': 'core'},
                              'common_prefit': {'path': 'prefit'}, 'adaptive': {'bank_artifact': {'path': 'bank'}},
                              'allocation': {'uniform_epochs': 10, 'mode': 'uniform'}}},
            'runtime': {'plot_every_steps': 10, 'seed': 42, 'device': 'cuda:0'},
            'evaluation': {'split': 'train', 'members': 4, 'sensor_counts': {'T': 256}},
            'rollout': {'steps': 4, 'solver': 'euler'},
            'observation_consistency': {'mode': 'endpoint_smooth', 'strength': 1., 'sigma': .05,
                                        'schedule_power': 2., 'final_clamp': True, 'chunk_size': 4096},
            'output': {'experiment_name': 'full'}, 'checkpointing': {}}


def test_native_ffm_consistency_and_common_source_pair():
    driver = load('run_coherence_adaptive_repair_round2')
    full = basic_ffm_config()
    driver.check_config(full, driver.ARMS[0], 12)
    changed = deepcopy(full)
    changed['observation_consistency']['final_clamp'] = False
    with pytest.raises(ValueError, match='native endpoint-smooth'):
        driver.check_config(changed, driver.ARMS[0], 12)
    changed = deepcopy(full)
    changed['optimization']['rollout_execution']['checkpointing'] = False
    with pytest.raises(ValueError, match='require rollout checkpointing'):
        driver.check_config(changed, driver.ARMS[0], 12)
    residual = deepcopy(full)
    residual['residual'] = {'enabled': True}
    residual['optimization']['lr'] = .01
    residual['output']['experiment_name'] = 'residual'
    configs = {driver.ARMS[0]: full, driver.ARMS[1]: residual}
    driver.check_source_pair(configs, 'ffm')
    residual['runtime']['seed'] = 100
    with pytest.raises(ValueError, match='targets or clocks'):
        driver.check_source_pair(configs, 'ffm')


def test_allocation_disposition_rejects_unqualified_claim(tmp_path):
    driver = load('run_coherence_adaptive_repair_round2')
    disposition = tmp_path / 'disposition.json'
    disposition.write_text(json.dumps({'schema': 'car_round2_allocation_disposition_v1',
                                        'mode': 'qualified_fixed', 'profiles': {}}))
    artifacts = {'allocation_disposition': {'path': str(disposition), 'sha256': driver.digest(disposition)}}
    with pytest.raises(ValueError, match='both source families'):
        driver.check_allocation_disposition(artifacts, tmp_path, {})


def test_common_core_must_match_frozen_observer_and_panel(tmp_path):
    driver = load('run_coherence_adaptive_repair_round2')
    observer, panel = tmp_path / 'observer.pt', tmp_path / 'panel.pt'
    observer.write_bytes(b'observer')
    panel.write_bytes(b'panel')
    binding = {'path': str(observer), 'sha256': driver.digest(observer)}
    core = {**binding, 'source_panel_path': str(panel), 'source_panel_sha256': driver.digest(panel)}
    driver.check_common_core_binding(core, binding, tmp_path)
    with pytest.raises(ValueError, match='frozen assessment observer'):
        driver.check_common_core_binding({**core, 'sha256': 'a' * 64}, binding, tmp_path)
    panel.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash differs'):
        driver.check_common_core_binding(core, binding, tmp_path)


def test_common_prefit_bank_digest_ignores_policy_but_rejects_geometry():
    from phycoflow_reconstruction.coherence.adaptive_fitting import (
        bank_target_digest,
        validate_bank_target_binding,
    )
    bank = {'directions': [[1., 2.]], 'triples': [], 'target_weights': {'state.group_projection': [1.]},
            'field_scales': [1., 2.], 'region_masses': [[1.]], 'triple_scales': [[[1., 1.]]],
            'physical_masses': [1.], 'field_order': ['a', 'b'], 'source_identity': 'source',
            'support_identity': 'support', 'units': 'model', 'block_units': {'state.group_projection': 1.}}
    binding = {'bank_target_sha256': bank_target_digest(bank)}
    policy = {**bank, 'policy_qualified': True, 'policy_pair_probabilities': [1.], 'policy_profile_sha256': 'x'}
    validate_bank_target_binding(binding, policy)
    altered = deepcopy(policy)
    altered['directions'][0][0] += .1
    with pytest.raises(ValueError, match='bank target differs'):
        validate_bank_target_binding(binding, altered)


def test_allocation_qualified_requires_real_profiles_and_exact_policy(tmp_path):
    from phycoflow_reconstruction.coherence.adaptive_fitting import _digest

    driver = load('run_coherence_adaptive_repair_round2')
    artifacts, profiles, configs = {}, {}, {}
    for family, prefix in (('ffm', 'FFM'), ('senseiver', 'Senseiver')):
        core_sha, panel_sha = 'a' * 64, 'b' * 64
        coefficient = tmp_path / f'{family}_coeff.json'
        coefficient.write_text(json.dumps({'binding': {'sample_ids': ['one', 'two']}}))
        artifacts[f'coefficients_{family}'] = {'path': str(coefficient), 'sha256': driver.digest(coefficient)}
        artifacts[f'observer_{family}'] = {'sha256': core_sha}
        base = {'policy_qualified': False, 'directions': [[1., 2., 3.]], 'source_identity': family}
        profile = {'schema': 'car_round2_allocator_job_v1', 'status': 'completed', 'source_family': family,
                   'parameterization': 'residual_head', 'qualified_one_path': True,
                   'selected_bank_sha256': _digest(base), 'source_panel_sha256': panel_sha, 'core_sha256': core_sha,
                   'prefit_sha256': driver.digest(coefficient), 'sample_ids': ['one', 'two'],
                   'train_population': 20, 'actual_head_parameters': 8, 'pair_draws': 3,
                   'gradient_cost_uniform_ratio': .5, 'overhead_fraction': .01,
                   'policies': {'uniform': {'mse_cost': 2.}, 'lagged_gradient_cost': {'mse_cost': 1., 'pair_probabilities': [1/3] * 3}}}
        path = tmp_path / f'{family}_profile.json'
        path.write_text(json.dumps(profile))
        profiles[family] = {'path': str(path), 'sha256': driver.digest(path)}
        bank = {**base, 'policy_qualified': True, 'policy_mode': 'fixed_source_profile',
                'policy_profile_sha256': driver.digest(path), 'policy_pair_probabilities': [1/3] * 3}
        bank_path = tmp_path / f'{family}_bank.json'
        bank_path.write_text(json.dumps(bank))
        artifacts[f'bank_{family}'] = {'path': str(bank_path), 'sha256': driver.digest(bank_path)}
        full = basic_ffm_config()
        full['coherence']['scalable_anchored_coherence']['common_core']['source_panel_sha256'] = panel_sha
        residual = deepcopy(full)
        residual['coherence']['scalable_anchored_coherence']['allocation']['mode'] = 'qualified_fixed'
        configs[f'{prefix}_Selected_Bank_Uniform_FullModel'] = full
        configs[f'{prefix}_Selected_Bank_Residual_Repair'] = residual
    disposition = tmp_path / 'qualified.json'
    disposition.write_text(json.dumps({'schema': 'car_round2_allocation_disposition_v1', 'mode': 'qualified_fixed', 'profiles': profiles}))
    artifacts['allocation_disposition'] = {'path': str(disposition), 'sha256': driver.digest(disposition)}
    driver.check_allocation_disposition(artifacts, tmp_path, configs)
    changed = json.loads((tmp_path / 'ffm_bank.json').read_text())
    changed['policy_pair_probabilities'] = [.2, .3, .5]
    (tmp_path / 'ffm_bank.json').write_text(json.dumps(changed))
    artifacts['bank_ffm']['sha256'] = driver.digest(tmp_path / 'ffm_bank.json')
    with pytest.raises(ValueError, match='profile, target, probabilities'):
        driver.check_allocation_disposition(artifacts, tmp_path, configs)


def test_assessment_contract_binds_truth_normalizer_and_frozen_observer(tmp_path):
    driver = load('run_coherence_adaptive_repair_round2')
    artifacts = {}
    def save(name, payload):
        path = tmp_path / f'{name}.json'
        path.write_text(json.dumps(payload))
        artifacts[name] = {'path': str(path), 'sha256': driver.digest(path)}
    for name in ('reserved_task', 'development_event', 'source_ffm', 'source_senseiver',
                 'observer_ffm', 'observer_senseiver', 'fidelity_ffm', 'fidelity_senseiver'):
        save(name, {'id': name})
    save('train_panel', {'schema': 'car_round2_panel_v1', 'split': 'train', 'sample_ids': ['case']})
    save('reserved_task_normalizer', {'schema': 'car_round2_reserved_normalizer_v1', 'split': 'train',
                                      'reserved_model_scores_used': False, 'value': 2.,
                                      'sample_ids': ['case'],
                                      'task_sha256': artifacts['reserved_task']['sha256'],
                                      'panel_sha256': artifacts['train_panel']['sha256']})
    configs = {}
    for family, prefix in (('ffm', 'FFM'), ('senseiver', 'Senseiver')):
        configs[f'{prefix}_Selected_Bank_Uniform_FullModel'] = basic_ffm_config()
        save(f'assessment_contract_{family}', {'schema': 'car_round2_assessment_contract_v2', 'family': family,
            'source_checkpoint_sha256': artifacts[f'source_{family}']['sha256'], 'core_sha256': artifacts[f'observer_{family}']['sha256'],
            'fidelity_reference_sha256': artifacts[f'fidelity_{family}']['sha256'], 'reserved_task_sha256': artifacts['reserved_task']['sha256'],
            'event_sha256': artifacts['development_event']['sha256'], 'train_panel_sha256': artifacts['train_panel']['sha256'],
            'reserved_task_normalizer_evidence_sha256': artifacts['reserved_task_normalizer']['sha256'],
            'reserved_task_normalizer': 2., 'reserved_task_normalization_rule': {'primary': 'final_train_source_task_score', 'fallback': 'train_truth_absolute_covariance', 'near_zero_threshold': 2e-12}, 'reserved_model_scores_used': False, 'sample_ids': ['case'],
            'normalizers': {'state': 1., 'scale': 1., 'shape': 1., 'event': 1.},
            'mean_allowance_physical': [0., 0., 0.], 'sensor_equality_tolerance': [0., 0., 0.],
            'shape_query_seed': 10, 'source_bank_sha256': 'a' * 64})
    driver.check_assessment_contracts(artifacts, tmp_path, configs)
    path = Path(artifacts['assessment_contract_ffm']['path'])
    changed = json.loads(path.read_text())
    changed['core_sha256'] = '0' * 64
    path.write_text(json.dumps(changed))
    artifacts['assessment_contract_ffm']['sha256'] = driver.digest(path)
    with pytest.raises(ValueError, match='assessment contract'):
        driver.check_assessment_contracts(artifacts, tmp_path, configs)


def test_complete_consistent_uniform_freeze_is_admitted(tmp_path):
    import yaml

    from phycoflow_reconstruction.coherence.adaptive_fitting import _digest, bank_target_digest

    driver = load('run_coherence_adaptive_repair_round2')
    artifacts, arms = {}, {}
    def save(name, content):
        path = tmp_path / f'{name}.json'
        path.write_text(json.dumps(content))
        record = {'path': str(path), 'sha256': driver.digest(path)}
        artifacts[name] = record
        return record
    for name in ('reserved_task', 'development_event', 'validation_panel', 'code_inventory'):
        save(name, {'id': name})
    save('train_panel', {'schema': 'car_round2_panel_v1', 'split': 'train', 'sample_ids': ['sample']})
    save('reserved_task_normalizer', {'schema': 'car_round2_reserved_normalizer_v1', 'split': 'train',
         'reserved_model_scores_used': False, 'value': 1., 'sample_ids': ['sample'],
         'task_sha256': artifacts['reserved_task']['sha256'], 'panel_sha256': artifacts['train_panel']['sha256']})
    save('allocation_disposition', {'schema': 'car_round2_allocation_disposition_v1', 'mode': 'uniform'})
    for family, prefix in (('ffm', 'FFM'), ('senseiver', 'Senseiver')):
        source_dir = tmp_path / 'sources' / family
        source_dir.mkdir(parents=True)
        source_path = source_dir / 'checkpoint.pt'
        source_path.write_text(family)
        artifacts[f'source_{family}'] = {'path': str(source_path), 'sha256': driver.digest(source_path)}
        observer = save(f'observer_{family}', {'core': family})
        fidelity = save(f'fidelity_{family}', {'reference': family})
        source_panel = save(f'source_panel_{family}', {'panel': family})
        bank = {'version': 'adaptive_measured_train_bank_v1', 'split': 'TRAIN', 'field_order': ['a', 'b', 'c'],
                'source_identity': driver.digest(source_path), 'support_identity': 'support', 'directions': [[1., 0., 0.]],
                'triples': [], 'target_weights': {'state.group_projection': [1.]}, 'field_scales': [1., 1., 1.],
                'region_masses': [[1.]], 'triple_scales': [[[1., 1., 1.]]], 'physical_masses': [1.], 'units': 'model',
                'block_units': {'state.group_projection': 1.}, 'policy_qualified': False}
        bank['sha256'] = _digest(bank)
        bank_binding = save(f'bank_{family}', bank)
        prefit = save(f'coefficients_{family}', {'version': 'common_prefit_calibration_v1', 'binding': {
            'source_checkpoint_sha256': driver.digest(source_path), 'support_sha256': 'support',
            'field_order': ['a', 'b', 'c'], 'sample_ids': ['sample'], 'bank_target_sha256': bank_target_digest(bank)}})
        save(f'assessment_contract_{family}', {'schema': 'car_round2_assessment_contract_v2', 'family': family,
            'source_checkpoint_sha256': driver.digest(source_path), 'core_sha256': observer['sha256'],
            'fidelity_reference_sha256': fidelity['sha256'], 'reserved_task_sha256': artifacts['reserved_task']['sha256'],
            'event_sha256': artifacts['development_event']['sha256'], 'train_panel_sha256': artifacts['train_panel']['sha256'],
            'reserved_task_normalizer_evidence_sha256': artifacts['reserved_task_normalizer']['sha256'],
            'reserved_task_normalizer': 1., 'reserved_task_normalization_rule': {'primary': 'final_train_source_task_score', 'fallback': 'train_truth_absolute_covariance', 'near_zero_threshold': 1e-12}, 'reserved_model_scores_used': False, 'sample_ids': ['sample'],
            'normalizers': {'state': 1., 'scale': 1., 'shape': 1., 'event': 1.}, 'mean_allowance_physical': [0.] * 3,
            'sensor_equality_tolerance': [0.] * 3, 'shape_query_seed': 1, 'source_bank_sha256': 'a' * 64})
        for residual in (False, True):
            arm = f'{prefix}_Selected_Bank_' + ('Residual_Repair' if residual else 'Uniform_FullModel')
            cfg = basic_ffm_config()
            cfg['optimization']['epochs'] = 1000
            cfg['checkpointing'] = {'epochs': driver.CHECKPOINTS, 'save_epoch_zero': True}
            cfg['model']['name'] = 'gl_rbf_cq' if family == 'ffm' else 'senseiver'
            cfg['source_run'] = str(source_dir)
            cfg['source'] = {'weight_view': 'ema_materialized' if family == 'ffm' else 'live'}
            cfg['output']['experiment_name'] = arm
            cfg['residual'] = {'enabled': residual}
            cfg['optimization']['multitask']['fidelity_budget']['reference_artifact'] = fidelity
            settings = cfg['coherence']['scalable_anchored_coherence']
            settings['common_core'] = {**observer, 'source_panel_path': source_panel['path'], 'source_panel_sha256': source_panel['sha256']}
            settings['common_prefit'] = prefit
            settings['adaptive']['bank_artifact'] = bank_binding
            if family == 'senseiver':
                cfg['observation_consistency'] = {'mode': 'none', 'final_clamp': False}
                cfg['evaluation']['members'] = 1
                cfg['optimization']['multitask']['fidelity_budget']['metric'] = 'physical_member_mse_model_units'
            config_path = tmp_path / f'{arm}.yaml'
            config_path.write_text(yaml.safe_dump(cfg))
            config_binding = {'path': str(config_path), 'sha256': driver.digest(config_path)}
            q = save(f'qual_{arm}', {'schema': 'car_round2_qualification_v1', 'arm': arm, 'passed': True,
                     'pause_step': 108, 'final_step': 144, 'formal_config_sha256': config_binding['sha256']})
            arms[arm] = {'config': config_binding, 'qualification': q}
    receipts = []
    for arm in driver.ARMS:
        rows = [{'version': 'complete_epoch_timing_v1', 'epoch': epoch, 'step': epoch * 12,
                 'accepted_updates': 12, 'train_seconds': 12., 'wall_seconds': 13.,
                 'callback_and_reporting_seconds': 1.} for epoch in range(1, 13)]
        journal = tmp_path / f'{arm}_journal.jsonl'
        journal.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        run_dir = str(tmp_path / f'{arm}_resume_run')
        pause = save(f'{arm}_pause', {'arm': arm, 'token': 'pause', 'started_unix': 0., 'gpu_uuid': 'GPU-test'})
        resume = save(f'{arm}_resume', {'arm': arm, 'token': 'resume', 'started_unix': 200., 'gpu_uuid': 'GPU-test'})
        pause_completion = save(f'{arm}_pause_completion', {'token': 'pause', 'status': 'budget_paused',
                                                            'global_step': 108, 'run_dir': run_dir, 'finished_unix': 100.})
        resume_completion = save(f'{arm}_resume_completion', {'token': 'resume', 'status': 'completed',
                                                              'global_step': 144, 'run_dir': run_dir, 'finished_unix': 300.})
        q = arms[arm]['qualification']
        q_path = Path(q['path'])
        q_data = json.loads(q_path.read_text())
        q_data.update(pause_launch_sha256=pause['sha256'], resume_launch_sha256=resume['sha256'],
                      pause_completion_sha256=pause_completion['sha256'], resumed=run_dir)
        q_path.write_text(json.dumps(q_data))
        q['sha256'] = driver.digest(q_path)
        evidence = {'qualification_receipt': q, 'pause_launch': pause, 'resume_launch': resume,
                    'pause_completion': pause_completion, 'resume_completion': resume_completion,
                    'epoch_timing': {'path': str(journal), 'sha256': driver.digest(journal)}}
        timing = {'schema': 'car_round2_complete_update_timing_v1', 'config_sha256': arms[arm]['config']['sha256'],
                  'parameterization': 'residual_head' if arm.endswith('Residual_Repair') else 'full_model',
                  'batch_size': 64, 'gpu_uuid': 'GPU-test', 'selected_epochs': [2, 3, 4, 5, 6, 7, 8, 11],
                  'per_epoch': rows, 'complete_update_seconds': 1., 'lease_wall_seconds': [100., 100.],
                  'outside_epoch_seconds': 44., 'cold_train_excess_seconds': 0.}
        for key, value in evidence.items():
            timing[f'{key}_path'], timing[f'{key}_sha256'] = value['path'], value['sha256']
        receipts.append(Path(save(f'timing_{arm}', timing)['path']))
    remaining = save('remaining_cost', {'schema': 'car_round2_remaining_cost_v1', 'components': [
        {'name': 'assessment', 'seconds': 1., 'evidence': [artifacts['train_panel']]}]})
    forecast = driver.measured_forecast(tmp_path, arms, receipts, spent_seconds=0., remaining_cost=remaining)
    assert forecast['per_arm_outside_epoch_seconds'] == [44.] * 4
    assert forecast['per_arm_callback_schedule_evidence'][driver.ARMS[0]]['callback_epoch_count'] == 101
    assert forecast['projected_seconds'] == 4 * (12000. + 1000. + 44.) + 1.
    save('forecast', forecast)
    manifest = {'schema': 'car_round2_freeze_v1', 'status': 'frozen', 'arms': arms, 'artifacts': artifacts}
    assert driver.check_freeze(manifest, tmp_path) is manifest
    snapshot = save('family_snapshot', {'schema': 'car_round2_ledger_snapshot_v1', 'snapshot_unix': 1.,
                    'state': {'schema': 'car_gpu_ledger_v1', 'closed': [], 'active': {}}})
    family_costs = save('family_remaining', {'schema': 'car_round2_remaining_cost_v1',
        'ledger_snapshot': snapshot, 'components': [
            {'name': 'pending_other_training', 'seconds': 100., 'evidence': [snapshot],
             'future_reservation_rule': {'per_arm_predicted_seconds': {driver.ARMS[0]: 40., driver.ARMS[1]: 60.}}},
            {'name': 'pending_other_qualification', 'seconds': 50., 'evidence': [snapshot],
             'future_reservation_rule': {'gpu_uuid': 'GPU-test', 'dispatch_start_unix': 0.}},
            {'name': 'pending_other_allocator', 'seconds': 10., 'evidence': [snapshot]},
            {'name': 'assessment_closeout', 'seconds': 20., 'evidence': [snapshot]}]})
    family_forecast = driver.measured_forecast(tmp_path, arms, receipts[2:], spent_seconds=0.,
                                               remaining_cost=family_costs, arms=driver.ARMS[2:])
    assert family_forecast['schema'] == 'car_round2_family_forecast_v1'
    assert family_forecast['projected_seconds'] == 2 * (12000. + 1000. + 44.) + 180.
    # Build a real two-arm admission from the same bound qualification records.
    import hashlib
    files = {'src/unchanged.py': 'source'}
    capsule = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    inventory = {'schema': 'car_execution_code_v1', 'files': files, 'sha256': capsule}
    save('qualified_code_inventory', inventory)
    save('code_inventory', inventory)
    save('qualification_code_compatibility', {'schema': 'car_round2_family_code_compatibility_v1',
         'passed': True, 'qualified_capsule_sha256': capsule, 'execution_capsule_sha256': capsule,
         'changed_files': [], 'scientific_code_identical': True})
    for arm, timing_path in zip(driver.ARMS[2:], receipts[2:], strict=True):
        timing = json.loads(timing_path.read_text())
        for phase in ('pause', 'resume'):
            path = Path(timing[f'{phase}_launch_path'])
            launch = json.loads(path.read_text()); launch['code_capsule_sha256'] = capsule
            path.write_text(json.dumps(launch)); timing[f'{phase}_launch_sha256'] = driver.digest(path)
        qpath = Path(arms[arm]['qualification']['path'])
        qualification = json.loads(qpath.read_text())
        qualification['pause_launch_sha256'] = timing['pause_launch_sha256']
        qualification['resume_launch_sha256'] = timing['resume_launch_sha256']
        qpath.write_text(json.dumps(qualification))
        timing['qualification_receipt_sha256'] = driver.digest(qpath)
        arms[arm]['qualification']['sha256'] = timing['qualification_receipt_sha256']
        timing_path.write_text(json.dumps(timing))
    family_forecast = driver.measured_forecast(tmp_path, arms, receipts[2:], spent_seconds=0.,
                                               remaining_cost=family_costs, arms=driver.ARMS[2:])
    save('forecast', family_forecast)
    bank_data = json.loads(Path(artifacts['bank_senseiver']['path']).read_text())
    profile = save('senseiver_profile', {'schema': 'car_round2_allocator_job_v1', 'status': 'completed',
        'source_family': 'senseiver', 'parameterization': 'residual_head', 'qualified_one_path': False,
        'source_panel_sha256': artifacts['source_panel_senseiver']['sha256'],
        'core_sha256': artifacts['observer_senseiver']['sha256'], 'prefit_sha256': artifacts['coefficients_senseiver']['sha256'],
        'selected_bank_sha256': bank_data['sha256'], 'gradient_cost_uniform_ratio': .9, 'overhead_fraction': .01,
        'train_population': 10, 'sample_ids': ['sample'], 'actual_head_parameters': 5, 'pair_draws': 2,
        'policies': {'uniform': {'mse_cost': 1.}, 'lagged_gradient_cost': {'mse_cost': .9}}})
    save('allocation_disposition', {'schema': 'car_round2_allocation_disposition_v1', 'mode': 'uniform',
                                    'profiles': {'senseiver': profile}})
    needed = {'source_senseiver', 'train_panel', 'validation_panel', 'reserved_task',
              'observer_senseiver', 'bank_senseiver', 'coefficients_senseiver', 'fidelity_senseiver',
              'assessment_contract_senseiver', 'reserved_task_normalizer', 'development_event',
              'allocation_disposition', 'forecast', 'code_inventory', 'qualified_code_inventory',
              'qualification_code_compatibility'}
    family_manifest = {'schema': 'car_round2_family_freeze_v1', 'status': 'frozen', 'family': 'senseiver',
                       'arms': {arm: arms[arm] for arm in driver.ARMS[2:]},
                       'artifacts': {name: artifacts[name] for name in needed}}
    assert driver.check_freeze(family_manifest, tmp_path, 'senseiver') is family_manifest
    tampered = json.loads(receipts[0].read_text())
    tampered['outside_epoch_seconds'] = 0.
    receipts[0].write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match='startup/closure measurement differs'):
        driver.measured_forecast(tmp_path, arms, receipts, spent_seconds=0., remaining_cost=remaining)


def test_callback_forecast_uses_union_of_formal_schedule_and_quiet_maximum():
    driver = load('run_coherence_adaptive_repair_round2')
    rows = [{'callback_and_reporting_seconds': .2} for _ in range(12)]
    rows[0]['callback_and_reporting_seconds'] = 80.
    rows[8]['callback_and_reporting_seconds'] = 3.  # qualification pause
    rows[9]['callback_and_reporting_seconds'] = 82.
    rows[11]['callback_and_reporting_seconds'] = 90.  # qualification completion
    rows[4]['callback_and_reporting_seconds'] = .5
    bound = driver.scheduled_callback_bound(rows)
    assert bound['callback_epochs'] == [1, *range(10, 1001, 10)]
    assert (bound['callback_epoch_count'], bound['quiet_epoch_count']) == (101, 899)
    assert bound['maximum_combined_callback_seconds'] == 90.
    assert bound['maximum_quiet_reporting_seconds'] == .5
    assert bound['total_seconds'] == 101 * 90. + 899 * .5


def test_family_forecast_requires_pending_family_and_closeout_reservations(tmp_path):
    driver = load('run_coherence_adaptive_repair_round2')
    evidence = tmp_path / 'evidence.json'
    evidence.write_text('{}')
    binding = {'path': str(evidence), 'sha256': driver.digest(evidence)}
    costs = tmp_path / 'remaining.json'
    snapshot = tmp_path / 'snapshot.json'
    snapshot.write_text(json.dumps({'schema': 'car_round2_ledger_snapshot_v1', 'snapshot_unix': 1.,
                                    'state': {'closed': [], 'active': {}}}))
    snapshot_binding = {'path': str(snapshot), 'sha256': driver.digest(snapshot)}
    costs.write_text(json.dumps({'schema': 'car_round2_remaining_cost_v1', 'ledger_snapshot': snapshot_binding, 'components': [
        {'name': 'assessment_closeout', 'seconds': 1., 'evidence': [binding]}]}))
    remaining = {'path': str(costs), 'sha256': driver.digest(costs)}
    with pytest.raises(ValueError, match='reserve all pending-family'):
        driver.measured_forecast(tmp_path, {}, [evidence, evidence], spent_seconds=0.,
                                 remaining_cost=remaining, arms=driver.ARMS[2:])


def test_early_uniform_requires_real_matching_failed_allocator(tmp_path):
    driver = load('run_coherence_adaptive_repair_round2')
    def save(name, body):
        path = tmp_path / name
        path.write_text(json.dumps(body))
        return {'path': str(path), 'sha256': driver.digest(path)}
    bank = save('bank.json', {'sha256': 'bank-target'})
    prefit = save('prefit.json', {'binding': {'sample_ids': ['sample']}})
    artifacts = {'bank_senseiver': bank, 'observer_senseiver': {'sha256': 'core'},
                 'coefficients_senseiver': prefit}
    profile = {'schema': 'car_round2_allocator_job_v1', 'status': 'completed',
               'source_family': 'senseiver', 'parameterization': 'residual_head',
               'qualified_one_path': False, 'source_panel_sha256': 'panel', 'core_sha256': 'core',
               'prefit_sha256': prefit['sha256'], 'selected_bank_sha256': 'bank-target',
               'gradient_cost_uniform_ratio': .9, 'overhead_fraction': .01,
               'train_population': 10, 'sample_ids': ['sample'], 'actual_head_parameters': 5, 'pair_draws': 2,
               'policies': {'uniform': {'mse_cost': 1.}, 'lagged_gradient_cost': {'mse_cost': .9}}}
    profile_binding = save('profile.json', profile)
    artifacts['allocation_disposition'] = save('disposition.json', {
        'schema': 'car_round2_allocation_disposition_v1', 'mode': 'uniform',
        'profiles': {'senseiver': profile_binding}})
    config = {'coherence': {'scalable_anchored_coherence': {
        'allocation': {'mode': 'uniform'}, 'common_core': {'source_panel_sha256': 'panel'}}}}
    configs = {'Senseiver_Selected_Bank_Uniform_FullModel': config,
               'Senseiver_Selected_Bank_Residual_Repair': config}
    driver.check_allocation_disposition(artifacts, tmp_path, configs, ('senseiver',))
    profile['gradient_cost_uniform_ratio'] = .7
    profile_binding = save('profile.json', profile)
    artifacts['allocation_disposition'] = save('disposition.json', {
        'schema': 'car_round2_allocation_disposition_v1', 'mode': 'uniform',
        'profiles': {'senseiver': profile_binding}})
    with pytest.raises(ValueError, match='measured failure'):
        driver.check_allocation_disposition(artifacts, tmp_path, configs, ('senseiver',))


def test_family_code_compatibility_rejects_changed_training_source(tmp_path):
    import hashlib

    driver = load('run_coherence_adaptive_repair_round2')
    def inventory(files):
        return {'schema': 'car_execution_code_v1', 'files': files,
                'sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return {'path': str(path), 'sha256': driver.digest(path)}
    original = inventory({'src/train.py': 'same', 'scripts/posttraining/sequence_car_round2_formal.py': 'old'})
    current = inventory({'src/train.py': 'same', 'scripts/posttraining/sequence_car_round2_formal.py': 'new'})
    proof = {'schema': 'car_round2_family_code_compatibility_v1', 'passed': True,
             'qualified_capsule_sha256': original['sha256'], 'execution_capsule_sha256': current['sha256'],
             'changed_files': ['scripts/posttraining/sequence_car_round2_formal.py'], 'scientific_code_identical': True}
    launch = save('launch.json', {'code_capsule_sha256': original['sha256']})
    timing = save('timing.json', {'pause_launch_path': launch['path'], 'resume_launch_path': launch['path']})
    artifacts = {'qualified_code_inventory': save('old.json', original), 'code_inventory': save('new.json', current),
                 'qualification_code_compatibility': save('proof.json', proof)}
    driver.check_family_code_compatibility(artifacts, tmp_path, [Path(timing['path'])])
    current = inventory({'src/train.py': 'changed', 'scripts/posttraining/sequence_car_round2_formal.py': 'new'})
    artifacts['code_inventory'] = save('new.json', current)
    with pytest.raises(ValueError, match='scientific code differs'):
        driver.check_family_code_compatibility(artifacts, tmp_path, [Path(timing['path'])])


def test_live_family_budget_consumes_reservations_once_and_rejects_unreserved_overage(tmp_path, monkeypatch):
    driver = load('run_coherence_adaptive_repair_round2')
    monkeypatch.setattr(driver.time, 'time', lambda: 100.)
    monkeypatch.setattr(driver, 'CAP', 1000.)
    monkeypatch.setattr(driver, 'RESERVE', 100.)
    def save(name, body):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body))
        return {'path': str(path), 'sha256': driver.digest(path)}
    audit = {'gpu_uuid': 'GPU-audit', 'phase': 'residual_allocator_audit', 'start_unix': 0.}
    snapshot = save('snapshot.json', {'schema': 'car_round2_ledger_snapshot_v1', 'snapshot_unix': 10.,
                                     'state': {'schema': 'car_gpu_ledger_v1', 'closed': [], 'active': {'audit': audit}}})
    costs = save('costs.json', {'schema': 'car_round2_remaining_cost_v1', 'ledger_snapshot': snapshot,
        'components': [{'name': 'pending_other_training', 'seconds': 200.,
                        'future_reservation_rule': {'per_arm_predicted_seconds': {driver.ARMS[0]: 100., driver.ARMS[1]: 100.}}},
                       {'name': 'pending_other_qualification', 'seconds': 200.,
                        'future_reservation_rule': {'gpu_uuid': 'GPU-ffm', 'dispatch_start_unix': 1.}},
                       {'name': 'pending_other_allocator', 'seconds': 200.,
                        'active_reservations': [{'token': 'audit', 'remaining_seconds_at_snapshot': 200.}]},
                       {'name': 'assessment_closeout', 'seconds': 50.}]})
    forecast = save('forecast.json', {'projected_seconds': 880., 'remaining_cost': costs,
        'per_arm_update_seconds': [.001, .001], 'per_arm_callback_seconds_per_epoch': [0., 0.],
        'per_arm_outside_epoch_seconds': [0., 0.], 'per_arm_cold_train_excess_seconds': [0., 0.]})
    manifest = {'family': 'senseiver', 'arms': {arm: {} for arm in driver.ARMS[2:]},
                'artifacts': {'forecast': forecast}}
    ledger = {'schema': 'car_gpu_ledger_v1', 'closed': [], 'active': {'audit': audit,
        'formal': {'gpu_uuid': 'GPU-sense', 'phase': 'formal', 'arm': driver.ARMS[2], 'start_unix': 95.}}}
    save('operations/gpu_ledger.json', ledger)
    result = driver.check_family_live_budget(manifest, tmp_path)
    assert result['projected_seconds'] == 880.
    assert result['charged_seconds_by_reservation']['pending_other_allocator'] == 90.
    assert result['charged_seconds_by_reservation'][driver.ARMS[2]] == 5.
    # Full Senseiver arm completes while a reserved FFM training lease runs;
    # its elapsed usage replaces the pending FFM obligation rather than adding it.
    ledger['active'].pop('formal')
    ledger['closed'].append({'token': 'formal', 'gpu_uuid': 'GPU-sense', 'phase': 'formal',
                             'arm': driver.ARMS[2], 'start_unix': 95., 'seconds': 5.})
    ledger['active']['ffm_formal'] = {'gpu_uuid': 'GPU-ffm', 'phase': 'formal',
                                      'arm': driver.ARMS[0], 'start_unix': 80.}
    ledger['active']['assessment'] = {'gpu_uuid': 'GPU-assessment', 'phase': 'native_assessment',
                                       'start_unix': 95.}
    save('operations/gpu_ledger.json', ledger)
    result = driver.check_family_live_budget(manifest, tmp_path)
    assert result['projected_seconds'] == 880.
    assert result['charged_seconds_by_reservation'][driver.ARMS[0]] == 20.
    assert result['charged_seconds_by_reservation']['assessment_closeout'] == 5.
    assert result['charged_seconds_by_reservation'][driver.ARMS[2]] == 5.
    ledger['active']['ffm_formal']['start_unix'] = -1.
    save('operations/gpu_ledger.json', ledger)
    assert driver.check_family_live_budget(manifest, tmp_path)['projected_seconds'] == 881.
    ledger['active']['ffm_formal']['start_unix'] = -25.
    save('operations/gpu_ledger.json', ledger)
    with pytest.raises(ValueError, match='exceed GPU ceiling'):
        driver.check_family_live_budget(manifest, tmp_path)
    ledger['active']['ffm_formal']['start_unix'] = 80.
    ledger['active']['unexpected'] = {'gpu_uuid': 'GPU-extra', 'phase': 'unbudgeted', 'start_unix': 75.}
    save('operations/gpu_ledger.json', ledger)
    with pytest.raises(ValueError, match='exceed GPU ceiling'):
        driver.check_family_live_budget(manifest, tmp_path)


def test_formal_duplicate_guard_preserves_lineage_resume():
    driver = load('run_coherence_adaptive_repair_round2')
    arm = driver.ARMS[2]
    state = {'active': {}, 'closed': [{'phase': 'formal', 'arm': arm}]}
    with pytest.raises(ValueError, match='duplicate launch'):
        driver.check_formal_attempt_available(state, arm, resume=False)
    driver.check_formal_attempt_available(state, arm, resume=True)
    driver.check_formal_attempt_available(state, driver.ARMS[3], resume=False)
    state['active']['live'] = {'phase': 'formal', 'arm': arm}
    with pytest.raises(ValueError, match='already active'):
        driver.check_formal_attempt_available(state, arm, resume=True)


def test_final_freeze_rejects_any_changed_early_admitted_arm(tmp_path, monkeypatch):
    driver = load('run_coherence_adaptive_repair_round2')
    directory = tmp_path / 'configs/family_admissions'
    directory.mkdir(parents=True)
    arm = driver.ARMS[2]
    early = {'arms': {arm: {'config': {'sha256': 'original'}, 'qualification': {'sha256': 'qualified'}}},
             'artifacts': {}}
    (directory / 'senseiver.json').write_text(json.dumps(early))
    monkeypatch.setattr(driver, 'check_freeze', lambda doc, campaign, family: doc)
    final = {'arms': {arm: {'config': {'sha256': 'different'}, 'qualification': {'sha256': 'qualified'}}},
             'artifacts': {}}
    with pytest.raises(ValueError, match='changes an already admitted arm'):
        driver.check_family_freeze_consistency(final, tmp_path)


def test_uniform_other_failed_admission_allows_own_passing_profile(tmp_path, monkeypatch):
    driver = load('run_coherence_adaptive_repair_round2')
    def save(name, body):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body))
        return {'path': str(path), 'sha256': driver.digest(path)}
    failed = save('sense_audit.json', {'qualified_one_path': False})
    passed = save('ffm_audit.json', {'qualified_one_path': True})
    other_disposition = save('sense_disposition.json', {
        'schema': 'car_round2_allocation_disposition_v1', 'mode': 'uniform', 'profiles': {'senseiver': failed}})
    save('configs/family_admissions/senseiver.json', {'artifacts': {'allocation_disposition': other_disposition}})
    disposition = save('disposition.json', {'schema': 'car_round2_allocation_disposition_v1', 'mode': 'uniform',
                                           'profiles': {'senseiver': failed, 'ffm': passed}})
    config = {'coherence': {'scalable_anchored_coherence': {
        'allocation': {'mode': 'uniform'}, 'common_core': {'source_panel_sha256': 'panel'}}}}
    configs = {'FFM_Selected_Bank_Uniform_FullModel': config, 'FFM_Selected_Bank_Residual_Repair': config}
    verified = []
    monkeypatch.setattr(driver, 'check_freeze', lambda doc, campaign, family: verified.append(family))
    driver.check_allocation_disposition({'allocation_disposition': disposition}, tmp_path, configs, ('ffm',))
    assert verified == ['senseiver']
