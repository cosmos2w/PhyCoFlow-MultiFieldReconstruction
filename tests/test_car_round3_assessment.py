"""Bounded real-bank replay of the Round 3 matched assessor."""
from argparse import Namespace
import json
from pathlib import Path

import pytest

from scripts.posttraining import assess_car_round3 as assessor

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'cases/turbulent_combustion/runs/coherence_adaptive_repair/round_3'
requires_saved_campaign = pytest.mark.skipif(
    not (CAMPAIGN / 'configs/prepared.json').is_file(),
    reason='local Round 3 preparation and immutable Round 2 banks are required',
)


@pytest.mark.parametrize("epoch", [100, 500])
@requires_saved_campaign
def test_frozen_source_context_and_real_prediction_replay(tmp_path, epoch):
    pytest.importorskip('gudhi')
    ctx = assessor.context(CAMPAIGN)
    old_bank = CAMPAIGN.parent / 'round_2/assessment/early_closeout/validation/banks/FFM_Selected_Bank_Uniform_FullModel_epoch_750'
    old_receipt = assessor.load_json(old_bank / 'receipt.json')
    freeze_arm = ctx['freeze']['arms']['FFM_Selected_Bank_Uniform_FullModel']
    assert old_receipt['checkpoint_sha256'] == freeze_arm['checkpoint']['sha256']
    ctx['source'] = dict(ctx['source'], sample_ids=ctx['source']['sample_ids'][:1])
    ctx['campaign'] = tmp_path
    banks = []
    for arm in assessor.ARMS:
        bank = tmp_path / arm
        bank.mkdir()
        (bank / '0000.pt').symlink_to((old_bank / '0000.pt').resolve())
        receipt = dict(schema='car_round3_native_bank_v1', status='completed', epoch=epoch,
            arm=arm, source_bank_sha256=assessor.hash_file(ctx['source_receipt']),
            panel_sha256=assessor.hash_file(ctx['panel_path']), sample_ids=ctx['source']['sample_ids'], members=4,
            prepared_sha256=assessor.hash_file(ctx['prepared_path']), source_checkpoint_sha256=ctx['source']['source_sha256'],
            config_path=str(ctx['panel_path']), config_sha256=assessor.hash_file(ctx['panel_path']),
            checkpoint=freeze_arm['checkpoint']['path'], checkpoint_sha256=freeze_arm['checkpoint']['sha256'],
            files_sha256=old_receipt['files_sha256'][:1])
        (bank / 'receipt.json').write_text(json.dumps(receipt))
        banks.append(bank)
    output = tmp_path / 'assessment/review'
    assessor.score(Namespace(raw_bank=banks[0], scaled_bank=banks[1], epoch=epoch, output=output), ctx)
    report = json.loads((output / 'assessment.json').read_text())
    historical = next(row for row in ctx['baseline']['rows'] if row['alpha'] == 1.)
    for arm in assessor.ARMS:
        candidate = report['candidates'][arm]
        if epoch == 100:
            assert 'reused_tasks' not in candidate
        else:
            assert candidate['reused_tasks']['regional_per_condition'][0] == pytest.approx(ctx['baseline']['reserved_task']['selected_per_condition'][0], rel=1e-5, abs=1e-7)
        assert candidate['per_condition_scores'][0] == pytest.approx(historical['per_condition_scores'][0], rel=1e-5, abs=1e-7)
        assert candidate['per_condition_member_mse'][0] == pytest.approx(historical['per_condition_member_mse'][0], rel=1e-5, abs=1e-7)
    if epoch == 100:
        assert report['task_scope'] == 'both tasks excluded'
        assert 'reused_tasks' not in report['source']
        decision_path = tmp_path / 'terminal_decision.json'
        decision = {'schema':'car_round3_terminal_decision_v1', 'decision':'stop_at_epoch_100',
            'epoch':100, 'authorize_endpoint_tasks':False,
            'review_sha256':assessor.hash_file(output/'assessment.json'),
            'bank_receipt_sha256':{arm:assessor.hash_file(bank/'receipt.json') for arm,bank in zip(assessor.ARMS,banks,strict=True)}}
        decision_path.write_text(json.dumps(decision))
        task_args = Namespace(decision=decision_path, review=output/'assessment.json', raw_bank=banks[0],
            scaled_bank=banks[1], output=tmp_path/'assessment/early_endpoint')
        with pytest.raises(ValueError, match='explicit terminal'):
            assessor.score_early_endpoint_tasks(task_args, ctx)
        decision['authorize_endpoint_tasks'] = True
        decision_path.write_text(json.dumps(decision))
        assessor.score_early_endpoint_tasks(task_args, ctx)
        endpoint = json.loads((task_args.output/'assessment.json').read_text())
        assert endpoint['candidates'][assessor.ARMS[0]]['reused_tasks']['regional_per_condition'][0] == pytest.approx(
            ctx['baseline']['reserved_task']['selected_per_condition'][0], rel=1e-5, abs=1e-7)
        assert endpoint['task_free_review_sha256'] == assessor.hash_file(output/'assessment.json')
        assert endpoint['paired_scaled_minus_raw']['event_score'] == 0.
    else:
        assert 'reused_tasks' in report['source']
        assert 'predeclared_recommendation' in report
    assert report['candidates'][assessor.ARMS[0]] == report['candidates'][assessor.ARMS[1]]


@requires_saved_campaign
def test_bank_rejects_changed_checkpoint(tmp_path):
    ctx = assessor.context(CAMPAIGN)
    receipt = dict(schema='car_round3_native_bank_v1', status='completed', epoch=100,
        arm=assessor.ARMS[0], source_bank_sha256=assessor.hash_file(ctx['source_receipt']),
        panel_sha256=assessor.hash_file(ctx['panel_path']), sample_ids=ctx['source']['sample_ids'], members=4,
            prepared_sha256=assessor.hash_file(ctx['prepared_path']), source_checkpoint_sha256=ctx['source']['source_sha256'],
            config_path=str(ctx['panel_path']), config_sha256=assessor.hash_file(ctx['panel_path']),
        checkpoint=str(ctx['panel_path']), checkpoint_sha256='wrong')
    (tmp_path / 'receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='checkpoint changed'):
        assessor.check_bank(tmp_path, ctx, 100, assessor.ARMS[0])


def test_failed_generation_is_charged_to_gpu_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('KEOPS_CACHE_FOLDER', str(tmp_path/'keops'))
    def command(argv, **_):
        return '0, GPU-test, test device\n' if '--query-gpu=index,uuid,name' in argv else ''
    monkeypatch.setattr(assessor.subprocess, 'check_output', command)
    def fail(*_):
        raise RuntimeError('deliberate bounded qualification failure')
    monkeypatch.setattr(assessor, 'generate', fail)
    args = Namespace(device='cuda:0', max_seconds=900., save_margin_seconds=60., arm=assessor.ARMS[0], epoch=100)
    with pytest.raises(RuntimeError, match='deliberate'):
        assessor.generate_with_lease(args, {'campaign':tmp_path})
    ledger = json.loads((tmp_path/'operations/gpu_ledger.json').read_text())
    assert not ledger['active'] and len(ledger['closed']) == 1
    assert ledger['closed'][0]['gpu_uuid'] == 'GPU-test'
    receipt = json.loads(next((tmp_path/'operations/completions').glob('*.json')).read_text())
    assert receipt['status'] == 'failed' and receipt['seconds'] >= 0
