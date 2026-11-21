import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / 'scripts/posttraining/forecast_car_round4.py'
SPEC = importlib.util.spec_from_file_location('car_round4_forecast', PATH)
forecast = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(forecast)


def gate():
    return {'admitted': True, 'reference_gpu_seconds': 70634.0,
            'forecast_gpu_seconds': {'qualification_measured_plus_upper_bound': 99999,
                'full_train_target_preparation_remaining': 99999,
                'teacher_including_calibration_replay_audit_and_persistence': 10000,
                'native_amortization_500_epochs': 5000, 'native_supervision_500_epochs': 4000,
                'assessment': 900},
            'teacher_cpu_worker_seconds': 11000, 'assessment_cpu_worker_seconds': 800,
            'teacher_epoch_reserves': {'gpu_wall_seconds': 109.2, 'cpu_worker_seconds': 125.1},
            'teacher_closeout_reserves': {'wall_seconds': 1478.1, 'cpu_worker_seconds': 1577.2}}


def test_replaces_preparation_and_counts_allowances_once():
    result = forecast.refresh_calculation(gate(), 150, 2500, 842)
    assert result['admitted']
    assert result['candidate_inclusive_gpu_seconds'] == 150 + 2500 + 842 + 60 + 10120 + 5000 + 900
    assert result['whole_round_gpu_seconds'] == result['candidate_inclusive_gpu_seconds'] + 4000
    assert (result['teacher_lease_seconds'], result['teacher_cpu_worker_forecast_seconds']) == (10120, 11120)
    assert (result['teacher_epoch_reserve_seconds'], result['teacher_epoch_reserve_cpu_seconds']) == (110, 126)
    assert (result['teacher_closing_reserve_seconds'], result['teacher_closing_reserve_cpu_seconds']) == (1599, 1698)


def test_refuses_over_budget_and_nonfinite_inputs():
    assert not forecast.refresh_calculation(gate(), 150, 7201, 842)['admitted']
    assert not forecast.refresh_calculation(gate(), 150, 2500, 20000)['checks']['inclusive_candidate_twofold_target']
    with pytest.raises(ValueError):
        forecast.refresh_calculation(gate(), float('nan'), 1, 0)
    g = gate(); g['teacher_cpu_worker_seconds'] = 21600
    assert not forecast.refresh_calculation(g, 150, 2500, 842)['checks']['teacher_6_cpu_worker_hours']
    assert not forecast.refresh_calculation(gate(), 150, 7190, 0,
                                           source_qualification_seconds=20)['checks']['target_preparation_2_gpu_hours']


def test_active_lease_blocks_before_bank_inspection(tmp_path):
    operations = tmp_path / 'operations'
    operations.mkdir()
    (operations / 'gpu_ledger.json').write_text(json.dumps({'schema':'car_gpu_ledger_v1',
                                                            'active':{'owned':{}}, 'closed':[]}))
    with pytest.raises(ValueError, match='all GPU leases must be closed'):
        forecast.refresh_main(['--campaign', str(tmp_path)])
    assert not (operations / 'teacher_launch_forecast.json').exists()


def test_bank_must_bind_exact_manifest_and_source():
    manifest = {'population': 8000, 'source_checkpoint_sha256': 'checkpoint'}
    source = {'source_parameter_buffer_sha256': 'weights'}
    bank = {'schema':'car_round4_source_bank_v1', 'status':'completed', 'split':'train',
            'population':8000, 'members':4, 'native_points':40300, 'units':'model field units',
            'manifest_sha256':'manifest', 'source_checkpoint_sha256':'checkpoint',
            'source_parameter_buffer_sha256':'weights', 'rows_sha256':['a'*64]*8000}
    forecast.validate_bank(bank, manifest, source, 'manifest')
    for key, wrong in [('manifest_sha256','other'), ('source_parameter_buffer_sha256','other'),
                       ('population',7999), ('rows_sha256',['a'*64]*7999)]:
        changed = {**bank, key:wrong}
        with pytest.raises(ValueError):
            forecast.validate_bank(changed, manifest, source, 'manifest')
