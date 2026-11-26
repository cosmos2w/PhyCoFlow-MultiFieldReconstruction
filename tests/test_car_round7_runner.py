import copy
import json
from types import SimpleNamespace

import pytest
import yaml

from scripts.posttraining.run_car_round7 import prepare, validate_common, validate_reference
from phycoflow_reconstruction.evaluation.adaptive_campaign import digest


def base_config():
    path = 'cases/turbulent_combustion/runs/coherence_adaptive_repair/round_6/configs/FFM_SSS_EpochCalibrated_PI_Repair.yaml'
    cfg = yaml.safe_load(open(path))
    cfg['optimization']['multitask']['fidelity_budget'].update(
        reference_law='training_selected_complement_v1',
        rotating_complement={'points': 512, 'seed': 20261125})
    cfg['coherence']['scalable_anchored_coherence']['condition_query_groups'] = 8
    cfg['runtime']['plot_every_steps'] = 10
    return cfg


def inputs(tmp_path):
    mse = [0.0, 0.02]
    scales = [1e-8, 0.02]
    old = tmp_path / 'old_reference.json'
    old.write_text(json.dumps({'reference': {'objective_scale': 1.8591270517644938}}))
    ref = {'reference': {'kind': 'mean_risk_difference', 'risk_scales': scales,
                         'absolute_allowance': [0.0, 0.002], 'reference_risks': [-0.01, 0.01],
                         'objective_scale': 1.8591270517644938},
           'risk_reference_panel': {'sample_count': 800, 'source_member_mse': mse,
                                    'source_mean_risk': [-0.01, 0.01]},
           'binding': {'fidelity_settings': {'reference_law': 'training_selected_complement_v1',
                                              'mean_risk_scales': scales}},
           'c0_preserved_from': {'path': str(old), 'sha256': digest(old)}}
    path = tmp_path / 'reference.json'
    path.write_text(json.dumps(ref))
    cal = {'schema': 'car_round7_training_law_calibration_v1', 'status': 'completed',
           'sample_count': 800, 'kappa': 1.2, 'reference': {'sha256': digest(path)}}
    return ref, cal, path


def test_common_contract_rejects_changed_warmup_and_native_loss():
    cfg = base_config()
    validate_common(cfg)
    changed = copy.deepcopy(cfg)
    changed['optimization']['lr_warmup']['epochs'] = 5
    with pytest.raises(ValueError):
        validate_common(changed)
    changed = copy.deepcopy(cfg)
    changed['objectives']['data_retention']['enabled'] = True
    with pytest.raises(ValueError):
        validate_common(changed)


def test_reference_uses_member_mse_floor_without_enlarging_allowance(tmp_path):
    ref, cal, path = inputs(tmp_path)
    validate_reference(ref, cal, path)
    ref['reference']['absolute_allowance'][0] = 1e-9
    with pytest.raises(ValueError):
        validate_reference(ref, cal, path)


def test_prepare_records_executor_and_preserves_common_streams(tmp_path):
    ref, cal, reference = inputs(tmp_path)
    del ref
    calibration = tmp_path / 'calibration.json'
    calibration.write_text(json.dumps(cal))
    base = tmp_path / 'base.yaml'
    base.write_text(yaml.safe_dump(base_config()))
    prepare(SimpleNamespace(base_config=base, reference=reference, calibration=calibration,
        campaign=tmp_path, family_vjp='factored', vjp_chunk_size=2, checkpoint_policy='off',
        context_cache='static_features'))
    manifest = json.loads((tmp_path/'configs/prepared.json').read_text())
    assert manifest['executor'] == {'family_vjp':'factored', 'vjp_chunk_size':2,
                                    'rollout_checkpointing':False, 'context_cache':'static_features'}
    arms = [yaml.safe_load(open(item['path'])) for item in manifest['arms'].values()]
    for cfg in arms:
        assert cfg['optimization']['multitask']['coherence_aggregation']['vjp_chunk_size'] == 2
        assert cfg['optimization']['rollout_execution']['checkpointing'] is False
        assert cfg['optimization']['rollout_execution']['context_cache'] == 'static_features'
        validate_common(cfg)
    assert arms[0]['runtime']['seed'] == arms[1]['runtime']['seed']
    assert arms[0]['observations'] == arms[1]['observations']
