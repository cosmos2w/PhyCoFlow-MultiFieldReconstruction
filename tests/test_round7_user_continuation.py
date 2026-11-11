import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location('round7_user_runner', Path(__file__).resolve().parents[1] / 'scripts/run_round7.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def authority(tmp_path, monkeypatch):
    campaign = tmp_path / 'campaign'
    artifacts = campaign / 'artifacts'
    run = campaign / 'T' / 'owned'
    artifacts.mkdir(parents=True)
    (run / 'evaluation').mkdir(parents=True)
    (run / 'checkpoints').mkdir()
    monkeypatch.setattr(runner, 'CAMPAIGN', campaign)
    monkeypatch.setattr(runner, 'ART', artifacts)
    config = campaign / 'T.yaml'
    config.write_text('frozen: true\n')
    checkpoint = run / 'checkpoints' / 'fixed_endpoint_500.pt'
    checkpoint.write_bytes(b'frozen T500')
    checksum = runner.digest(checkpoint)
    runner.write(run / 'evaluation/checkpoint_binding_500.json', {'checkpoint': str(checkpoint), 'checkpoint_sha256': checksum})
    runner.write(artifacts / 'T_500_complete.json', {'status': 'completed', 'epoch': 500,
        'run': str(run), 'last_checkpoint_sha256': checksum})
    runner.write(artifacts / 'owned_run_T.json', {'run': str(run), 'config_sha256': runner.digest(config)})
    receipt = {'schema': 'round7_user_continuation_v1', 'status': 'user_authorized', 'arm': 'T',
        'start_epoch': 500, 'target_epoch': 2000, 'run': str(run), 'config_sha256': runner.digest(config),
        'checkpoint_500': str(checkpoint), 'checkpoint_500_sha256': checksum,
        'update_seconds': 3., 'fixed_overhead_seconds': 1000., 'replay_reserve_seconds': 1950.,
        'projected_gpu_seconds': 61450., 'gpu_cap_seconds': 129600}
    path = artifacts / 'user_continuation.json'
    runner.write(path, receipt)
    return path, receipt, run, {'configs': {'T': str(config)}}


def test_only_fixed_user_authorized_same_run_and_checkpoint(authority, tmp_path):
    path, receipt, run, contract = authority
    assert runner.verify_user_continuation(path, arm='T', target_epoch=2000, resume=run, contract=contract) == receipt
    for arm, target, resume in [('B', 2000, run), ('T', 1000, run), ('T', 2000, tmp_path / 'other')]:
        with pytest.raises(ValueError):
            runner.verify_user_continuation(path, arm=arm, target_epoch=target, resume=resume, contract=contract)
    altered = {**receipt, 'projected_gpu_seconds': 1.}
    runner.write(path, altered)
    with pytest.raises(ValueError, match='budget forecast'):
        runner.verify_user_continuation(path, arm='T', target_epoch=2000, resume=run, contract=contract)
    runner.write(path, receipt)
    Path(receipt['checkpoint_500']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='completed T500'):
        runner.verify_user_continuation(path, arm='T', target_epoch=2000, resume=run, contract=contract)


def test_resume_preserves_full_boundary_and_optional_state():
    keys = ('model', 'optimizer', 'external_policy', 'query_coverage', 'rng_state', 'training_aux_state',
            'family_states', 'family_configs', 'normalization', 'family_scales', 'source_hashes',
            'config_sha256', 'coherence_calibration_sha256', 'round7_child_binding_sha256')
    saved = {key: {'value': torch.tensor([1.])} for key in keys}
    saved['global_step'] = 6500
    runner.verify_continuation_state(copy.deepcopy(saved), saved, strict=True)
    for key in ('optimizer', 'rng_state', 'training_aux_state', 'family_states', 'round7_child_binding_sha256'):
        changed = copy.deepcopy(saved)
        changed[key]['value'][0] += 1
        with pytest.raises(ValueError, match=key):
            runner.verify_continuation_state(changed, saved, strict=True)


def test_budget_and_recovery_replay_caps(authority):
    _, receipt, _, _ = authority
    ledger = {'attempt_count': 19549, 'formal_unique': 19500, 'formal_replays': 0,
              'arm_last_step': {'T': 6499}, 'intervals': [{'device_uuid': 'gpu0', 'start': 1., 'end': 49813.}]}
    assert runner.verify_user_continuation_budget(receipt, start_step=6500, ledger=ledger) == 61450.
    with pytest.raises(ValueError, match='shared attempt or GPU'):
        runner.verify_user_continuation_budget(receipt, start_step=6500, ledger={**ledger, 'attempt_count': 30000})
    with pytest.raises(ValueError, match='shared attempt or GPU'):
        runner.verify_user_continuation_budget(receipt, start_step=6500, ledger={**ledger, 'formal_unique': 30000})
    with pytest.raises(ValueError, match='shared attempt or GPU'):
        runner.verify_user_continuation_budget(receipt, start_step=6500, ledger={**ledger, 'formal_replays': 50,
                                                'arm_last_step': {'T': 7100}})
    with pytest.raises(ValueError, match='outside'):
        runner.verify_user_continuation_budget(receipt, start_step=6499, ledger=ledger)
    with pytest.raises(ValueError, match='shared attempt or GPU'):
        runner.verify_user_continuation_budget(receipt, start_step=6500, ledger={**ledger,
            'intervals': [{'device_uuid': 'gpu0', 'start': 1., 'end': 75000.}]})


def test_user_fixed_horizon_keeps_only_protective_stop():
    fields = ['T']
    rows = [{'source_ratios': {'T': 1.20}}, {'source_ratios': {'T': 1.21}}]
    assert runner.extension_fidelity_breach(rows, fields, 2000, False)
    assert not runner.extension_fidelity_breach(rows, fields, 2000, True)
    assert not runner.consecutive_fidelity_breach(rows, fields, 1.25)
    rows[-1]['source_ratios']['T'] = 1.30
    assert not runner.consecutive_fidelity_breach(rows, fields, 1.25)
    rows[0]['source_ratios']['T'] = 1.26
    assert runner.consecutive_fidelity_breach(rows, fields, 1.25)
