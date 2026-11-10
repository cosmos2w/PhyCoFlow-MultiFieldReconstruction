import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('round7_runner', Path(__file__).resolve().parents[1] / 'scripts/run_round7.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_attempt_ledger_counts_replays_and_caps(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'ART', tmp_path)
    runner.charge_attempt('formal', 'B', 0)
    runner.charge_attempt('formal', 'B', 0)
    runner.charge_attempt('qualification', 'B', 1)
    ledger = runner.read(tmp_path / 'budget_ledger.json')
    assert (ledger['attempt_count'], ledger['formal_unique'], ledger['formal_replays']) == (3, 1, 1)
    assert len((tmp_path / 'attempts.jsonl').read_text().splitlines()) == 3
    ledger['qualification_attempts'] = 64
    runner.write(tmp_path / 'budget_ledger.json', ledger)
    with pytest.raises(RuntimeError, match='qualification attempt cap'):
        runner.charge_attempt('qualification', 'P', 1)


def test_occupied_intervals_count_device_unions():
    intervals = [{'device_uuid': 'x', 'start': 0, 'end': 4},
                 {'device_uuid': 'x', 'start': 2, 'end': 5},
                 {'device_uuid': 'y', 'start': 1, 'end': 3}]
    assert runner.total_seconds(intervals) == 7


def test_stage_decision_freezes_proposal_without_inference(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'CAMPAIGN', tmp_path)
    monkeypatch.setattr(runner, 'ART', tmp_path / 'artifacts')
    monkeypatch.setattr(runner, 'verify_contract', lambda *args: {})
    runner.ART.mkdir()
    measurements = tmp_path / 'measured.json'
    runner.write(measurements, {'update_seconds': {'B': 2}, 'validation_seconds': 1,
                               'training_panel_seconds': 1, 'native_panel_seconds': 1})
    runner.write(runner.ART / 'resource_forecast.json', {'measurements_path': str(measurements),
                                                         'measurement_sha256': runner.digest(measurements)})
    fields = {f: 1.0 for f in ('CH4', 'CO', 'T', 'U_1', 'p')}
    observations = {str(e): {'epoch': e, 'scale': .98, 'state_delta': 0, 'shape_delta': 0,
                            'q_mse': fields, 'train_mse': fields, 'native_mse': fields}
                    for e in (300, 400, 500)}
    required = ('actual_step_findings', 'shape_attraction_separation', 'native_field_checks',
                'measured_cost', 'remaining_budget', 'continued_run_identities')
    payload = {'epoch': 500, 'source_validation_scale': 1., 'manifest_sha256': 'a' * 64,
               'observer_sha256': 'b' * 64,
               'arms': [{'name': 'B', 'observations': observations, 'update_seconds': 2, 'unchanged_terms': True,
                         'manifest_sha256': 'a' * 64, 'observer_sha256': 'b' * 64, 'checkpoint_sha256': 'c' * 64}],
               'proposal_context': {key: 'recorded evidence' for key in required}}
    source = tmp_path / 'input.json'
    runner.write(source, payload)
    assert runner.stage_decide(source)['continue'] == ['B']
    assert 'Decision: eligible' in (tmp_path / 'FORMAL_PROPOSAL.md').read_text()
    with pytest.raises(FileExistsError):
        runner.stage_decide(source)


def test_projection_binding_uses_source_attraction_only_for_t(monkeypatch, tmp_path):
    import yaml
    monkeypatch.setattr(runner, 'CAMPAIGN', tmp_path)
    monkeypatch.setattr(runner, 'ART', tmp_path / 'artifacts')
    runner.ART.mkdir()
    configs = {}
    for arm in ('B', 'P', 'T'):
        path = tmp_path / (arm + '.yaml')
        path.write_text(yaml.safe_dump({'optimization': {'multitask': {'external_policy': {}}}}))
        configs[arm] = str(path)
    source = tmp_path / 'train.json'
    losses = {'global_distribution.total': .05, 'multiscale_spectrum.total': .15, 'topology.total': 1.8}
    runner.write(source, {'manifest_sha256': 'a' * 64, 'split': 'train', 'rows': [{'losses': losses}] * 800})
    runner.write(runner.ART / 'reference_SOURCE.json', {'reference': 'SOURCE',
        'parent_source_checkpoint_sha256': 's', 'manifest_sha256': 'a' * 64,
        'train': {'conditions': 800, 'reductions': str(source), 'sha256': runner.digest(source),
                  'fair_shape_attraction_mean': .7}})
    fixed = tmp_path / 'fixed_scales.json'
    runner.write(fixed, losses)
    monkeypatch.setattr(runner, 'verify_contract', lambda: {'source_sha256': 's', 'files': {}, 'configs': configs,
                                                             'decision_scales': str(fixed)})
    monkeypatch.setattr(runner, 'runtime_source_files', lambda: [])
    result = runner.bind_projection_scales()
    assert result['B']['topology'] == pytest.approx(1.8)
    assert result['T']['topology'] == .7
    assert result['T']['multiscale_spectrum'] == pytest.approx(.15)


def test_legacy_branch_requires_observer_diagnosis_before_learning(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'CAMPAIGN', tmp_path)
    monkeypatch.setattr(runner, 'ART', tmp_path / 'artifacts')
    runner.ART.mkdir()
    runner.write(runner.ART / 'formal_contract.json', {})
    diagnosis = runner.ART / 'diagnosis.json'
    runner.write(diagnosis, {'status': 'observer_unqualified', 'reason': 'domain mismatch', 'evidence': ['saved failure']})
    assert runner.freeze_legacy_branch(diagnosis)['omitted_arm'] == 'T'
    with pytest.raises(FileExistsError):
        runner.freeze_legacy_branch(diagnosis)


def test_completed_endpoint_reuse_is_read_only_and_identity_checked(tmp_path):
    identities = {'fair': 'a' * 64, 'legacy': 'b' * 64}
    paths = {}
    for label in ('checkpoint', 'train_reductions', 'validation_reductions', 'native_receipt'):
        path = tmp_path / label
        runner.write(path, {'manifest_sha256': 'c' * 64, 'observer_identities': identities})
        paths[label] = str(path)
        paths[label + '_sha256'] = runner.digest(path)
    paths['model_state_sha256'] = 'd' * 64
    binding = tmp_path / 'binding.json'
    runner.write(binding, paths)
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    for _ in range(2):
        runner.verify_endpoint_binding(binding, model_state_sha256='d' * 64,
                                       manifest_sha256='c' * 64, observer_identities=identities)
    assert before == {p: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError):
        runner.verify_endpoint_binding(binding, model_state_sha256='e' * 64,
                                       manifest_sha256='c' * 64, observer_identities=identities)


def test_1000_extension_only_when_measured_and_within_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'CAMPAIGN', tmp_path)
    monkeypatch.setattr(runner, 'ART', tmp_path / 'artifacts')
    monkeypatch.setattr(runner, 'verify_contract', lambda *args: {})
    runner.ART.mkdir()
    measured = tmp_path / 'measured.json'
    runner.write(measured, {'update_seconds': {'B': 2}, 'validation_seconds': 1,
                            'training_panel_seconds': 1, 'native_panel_seconds': 1})
    runner.write(runner.ART / 'resource_forecast.json', {'measurements_path': str(measured),
                                                        'measurement_sha256': runner.digest(measured)})
    fields = {f: 1.0 for f in ('CH4', 'CO', 'T', 'U_1', 'p')}
    observations = {str(e): {'epoch': e, 'scale': .99 if e == 800 else .98, 'state_delta': 0,
                            'shape_delta': 0, 'q_mse': fields, 'train_mse': fields, 'native_mse': fields}
                    for e in (800, 900, 1000)}
    source = tmp_path / 'input1000.json'
    runner.write(source, {'epoch': 1000, 'source_validation_scale': 1.,
        'manifest_sha256': 'a' * 64, 'observer_sha256': 'b' * 64,
        'arms': [{'name': 'B', 'observations': observations, 'update_seconds': 2,
                  'manifest_sha256': 'a' * 64, 'observer_sha256': 'b' * 64, 'checkpoint_sha256': 'c' * 64}]})
    decision = runner.stage_decide(source)
    assert decision['selected'] == 'B' and decision['extension'] == 'B'
    assert runner.read(runner.ART / 'continuation_1250.json')['pinned_1000_fallback'] == 'B'
    assert not (runner.ART / 'final_selection.json').exists()


def test_protectively_stopped_baseline_closes_without_a_candidate(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'CAMPAIGN', tmp_path)
    monkeypatch.setattr(runner, 'ART', tmp_path / 'artifacts')
    monkeypatch.setattr(runner, 'verify_contract', lambda *args: {})
    runner.ART.mkdir()
    checkpoint = tmp_path / 'baseline_checkpoint.pt'
    checkpoint.write_bytes(b'fixed checkpoint')
    runner.write(runner.ART / 'B_protective_stop.json', {'arm': 'B', 'epoch': 300,
        'checkpoint': str(checkpoint), 'checkpoint_sha256': runner.digest(checkpoint)})
    result = runner.close_incomplete()
    assert result['status'] == 'closed_incomplete'
    final = runner.read(runner.ART / 'final_selection.json')
    assert final['selected'] == 'SOURCE' and final['endpoint'] is None


def test_measurements_come_from_complete_saved_panels_and_updates(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'ART', tmp_path / 'artifacts')
    monkeypatch.setattr(runner, 'verify_contract', lambda *args: {})
    runner.ART.mkdir()
    panels = {}
    for split, seconds in (('train', 80), ('validation', 10), ('s100', 12)):
        path = tmp_path / (split + '.json')
        runner.write(path, {'timing': {'condition_seconds': seconds, 'invocation_seconds': seconds + 1}})
        panels[split] = {'reductions': str(path), 'sha256': runner.digest(path)}
    native = tmp_path / 'native.json'
    runner.write(native, {'wrapper_wall_seconds': 20})
    source = runner.ART / 'reference_SOURCE.json'
    runner.write(source, {'train': panels['train'], 'validation': panels['validation'],
                           'native_validation': {'receipt': str(native), 'sha256': runner.digest(native)}})
    s100 = runner.ART / 'reference_S100.json'
    runner.write(s100, {'validation': panels['s100']})
    segments = []
    for arm in ('B', 'P'):
        run = tmp_path / arm
        (run / 'logs').mkdir(parents=True)
        (run / 'logs/full_updates.jsonl').write_text(''.join(json.dumps({'update_seconds': 2}) + '\n' for _ in range(12)))
        segments.append({'arm': arm, 'run': str(run), 'seconds': 36})
    timing = runner.ART / 'qualification_timing.json'
    runner.write(timing, {'segments': segments})
    runner.write(runner.ART / 'qualified.json', {'status': 'completed', 'qualified_arms': ['B', 'P'],
        'observer_qualified': True, 'timing_sha256': runner.digest(timing), 'evaluation_reference_receipts': {
            'SOURCE': runner.digest(source), 'S100': runner.digest(s100)}})
    result = runner.build_measurements()
    assert result['update_seconds'] == {'B': 3, 'P': 3}
    assert (result['training_panel_seconds'], result['validation_seconds'], result['native_panel_seconds']) == (81, 13, 20)
