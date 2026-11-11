import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'src'))
from phycoflow_reconstruction.evaluation.round7_cost_refresh import build_cost_receipt, verify_cost_receipt, sha


def write(path, obj):
    path.write_text(json.dumps(obj, sort_keys=True))


@pytest.fixture
def evidence(tmp_path):
    art = tmp_path / 'artifacts'
    art.mkdir()
    measurements = art / 'measurements.json'
    write(measurements, {'update_seconds': {'B': .001, 'P': .05}, 'validation_seconds': 1.,
                         'training_panel_seconds': 2., 'native_panel_seconds': 3.})
    write(art / 'resource_forecast.json', {'measurements_path': str(measurements),
          'measurement_sha256': sha(measurements), 'selected_arms': ['B', 'P']})
    intervals, attempts = [], []
    for arm, pid, start, elapsed in [('B', 101, 100., 100.), ('P', 102, 300., 200.)]:
        intervals.append({'stage': arm + '_500', 'device_uuid': 'GPU-fixture', 'pid': pid,
                          'start': start, 'end': start + elapsed + 1})
        attempts.extend({'arm': arm, 'pid': pid, 'phase': 'formal', 'step': step,
                         'time': start + .1 + elapsed * step / 6500} for step in range(6500))
        write(art / f'{arm}_500_complete.json', {'status': 'completed', 'epoch': 500,
              'elapsed_seconds': elapsed, 'timing': {'totals_seconds': {'callback': 20.}}})
    (art / 'attempts.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in attempts))
    write(art / 'budget_ledger.json', {'attempt_count': len(attempts), 'intervals': intervals})
    return art


def test_refresh_fails_closed_and_preserves_initial_forecast(evidence):
    frozen = (evidence / 'resource_forecast.json').read_bytes()
    measures = (evidence / 'measurements.json').read_bytes()
    cost = build_cost_receipt(evidence, 500, ['B', 'P'])
    assert cost['arms']['B']['whole_stage_seconds_per_attempt'] == 100 / 6500
    assert cost['arms']['B']['forecast_seconds_per_attempt'] == 80 / 6500
    assert cost['arms']['P']['forecast_seconds_per_attempt'] == .05
    path = evidence / 'cost.json'
    write(path, cost)
    assert verify_cost_receipt(path, 500, ['B', 'P']) == cost
    assert (evidence / 'resource_forecast.json').read_bytes() == frozen
    assert (evidence / 'measurements.json').read_bytes() == measures
    ledger = json.loads((evidence / 'budget_ledger.json').read_text())
    del ledger['intervals'][0]['end']
    write(evidence / 'budget_ledger.json', ledger)
    with pytest.raises(ValueError):
        build_cost_receipt(evidence, 500, ['B', 'P'])
    with pytest.raises(ValueError):
        verify_cost_receipt(path, 500, ['B', 'P'])


def test_incomplete_recovery_fails_closed(evidence):
    ledger = json.loads((evidence / 'budget_ledger.json').read_text())
    ledger['intervals'].append(dict(ledger['intervals'][0], pid=999, start=1., end=2.))
    write(evidence / 'budget_ledger.json', ledger)
    with pytest.raises(ValueError, match='multiply invoked'):
        build_cost_receipt(evidence, 500, ['B', 'P'])


@pytest.mark.parametrize('mixed', [False, True])
def test_controller_uses_whole_stage_tie_and_refreshed_forecast(evidence, tmp_path, monkeypatch, mixed):
    spec = importlib.util.spec_from_file_location('isolated_round7_runner', BASE / 'scripts/run_round7.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.ART = evidence
    runner.CAMPAIGN = tmp_path
    (tmp_path / 'comparison').mkdir()
    monkeypatch.setattr(runner, 'verify_contract', lambda *args: {})
    @contextmanager
    def ledger_transaction():
        yield json.loads((evidence / 'budget_ledger.json').read_text())
    monkeypatch.setattr(runner, 'ledger_transaction', ledger_transaction)
    fields = ['CH4', 'CO', 'T', 'U_1', 'p']
    cohort = 'a' * 64
    if mixed:
        ledger_path = evidence / 'budget_ledger.json'
        ledger = json.loads(ledger_path.read_text())
        ledger['intervals'][1]['device_uuid'] = 'GPU-other'
        ledger['intervals'].append({'stage': 'qualify', 'device_uuid': 'GPU-fixture',
                                    'pid': 99, 'start': 1., 'end': 2.})
        write(ledger_path, ledger)
        measurements = evidence / 'measurements.json'
        values = json.loads(measurements.read_text())
        values['update_seconds']['P'] = .0005
        write(measurements, values)
        forecast_path = evidence / 'resource_forecast.json'
        forecast = json.loads(forecast_path.read_text())
        forecast['measurement_sha256'] = sha(measurements)
        write(forecast_path, forecast)
    cost = build_cost_receipt(evidence, 500, ['B', 'P'])
    cost_path = evidence / 'cost.json'
    write(cost_path, cost)
    def arm(name, scale):
        observations = {str(e): {'epoch': e, 'scale': scale, 'state_delta': 0., 'shape_delta': 0.,
                                  'q_mse': {f: 1. for f in fields},
                                  'train_mse': {f: 1. for f in fields},
                                  'native_mse': {f: 1. for f in fields}} for e in (300, 400, 500)}
        return {'name': name, 'observations': observations, 'manifest_sha256': cohort,
                'observer_sha256': cohort, 'checkpoint_sha256': cohort,
                'update_seconds': cost['arms'][name]['decision_seconds_per_attempt'],
                'unchanged_terms': True}
    data = {'epoch': 500, 'source_validation_scale': 1., 'manifest_sha256': cohort,
            'observer_sha256': cohort, 'arms': [arm('B', .981), arm('P', .98)],
            'proposal_context': {k: {} for k in ('actual_step_findings', 'shape_attraction_separation',
                'native_field_checks', 'measured_cost', 'remaining_budget', 'continued_run_identities')}}
    input_path = tmp_path / 'input.json'
    write(input_path, data)
    result = runner.stage_decide(input_path, str(cost_path))
    assert result['selected'] == ('P' if mixed else 'B')
    receipt = json.loads((evidence / 'continuation_1000.json').read_text())
    p_rate = 180 / 6500 if mixed else .05
    expected = 6500 * (80 / 6500 + p_rate) + 15 + 6 + 15 + 650 * p_rate
    assert receipt['projected_seconds'] == pytest.approx(expected)


def test_mixed_device_uses_common_qualification_for_tie_but_formal_forecast(evidence):
    ledger_path = evidence / 'budget_ledger.json'
    ledger = json.loads(ledger_path.read_text())
    ledger['intervals'][1]['device_uuid'] = 'GPU-other'
    ledger['intervals'].append({'stage': 'qualify', 'device_uuid': 'GPU-fixture',
                                'pid': 99, 'start': 1., 'end': 2.})
    write(ledger_path, ledger)
    cost = build_cost_receipt(evidence, 500, ['B', 'P'])
    assert cost['tie_cost_basis'] == 'common-device cold-start qualification'
    assert cost['qualification_device_uuid'] == 'GPU-fixture'
    assert cost['arms']['B']['decision_seconds_per_attempt'] == .001
    assert cost['arms']['P']['decision_seconds_per_attempt'] == .05
    assert cost['arms']['B']['whole_stage_seconds_per_attempt'] == 100 / 6500
    assert cost['arms']['P']['whole_stage_seconds_per_attempt'] == 200 / 6500
    assert cost['arms']['B']['forecast_seconds_per_attempt'] == 80 / 6500
    assert cost['arms']['P']['forecast_seconds_per_attempt'] == .05
    ledger['intervals'].append({'stage': 'qualify-repair', 'device_uuid': 'GPU-other',
                                'pid': 98, 'start': 3., 'end': 4.})
    write(ledger_path, ledger)
    with pytest.raises(ValueError, match='common qualification GPU'):
        build_cost_receipt(evidence, 500, ['B', 'P'])
