import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "round8_runner_ledger", Path(__file__).resolve().parents[1] / "scripts/run_round8.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_attempt_budget_and_fork_positions_have_persistent_audit(tmp_path):
    runner.charge_attempt("formal", tmp_path, arm="warmup", step=0)
    runner.charge_attempt("formal", tmp_path, arm="warmup", step=0)
    runner.charge_attempt("formal", tmp_path, arm="control", step=130)
    runner.charge_attempt("formal", tmp_path, arm="candidate", step=130)
    with pytest.raises(ValueError, match="contiguously"):
        runner.charge_attempt("formal", tmp_path, arm="control", step=132)
    ledger = json.loads((tmp_path / "budget_ledger.json").read_text())
    assert (ledger["formal_unique"], ledger["formal_replays"], ledger["formal_attempts"]) == (3, 1, 4)
    rows = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and rows[1]["replay"] and rows[2]["step"] == 130
    for step in range(100):
        runner.charge_attempt("qualification", tmp_path, arm="warmup", step=step)
    with pytest.raises(RuntimeError, match="qualification-attempt cap"):
        runner.charge_attempt("qualification", tmp_path, arm="warmup", step=100)


def test_training_cannot_spend_reserved_closeout_time(tmp_path):
    payload = {"qualification_attempts": 0, "formal_attempts": 0,
               "intervals": [{"device_uuid": "one", "start": 0, "end": 28 * 3600}]}
    (tmp_path / "budget_ledger.json").write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="closeout reserve"):
        runner.charge_attempt("formal", tmp_path, arm="warmup", step=0)


def test_occupied_lease_records_arm_without_rewriting_history(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-1')
    monkeypatch.setenv('KEOPS_CACHE_FOLDER',str(tmp_path/'keops'))
    monkeypatch.delenv('ROUND8_LEASE_TOKEN',raising=False)
    previous={'id':'old','stage':'train','device_uuid':'GPU-2','start':1.,'end':2.}
    (tmp_path/'budget_ledger.json').write_text(json.dumps({'qualification_attempts':0,'formal_attempts':0,'intervals':[previous]}))
    with runner.occupied('train','GPU-1',formal=True,directory=tmp_path,arm='B8'):
        live=json.loads((tmp_path/'budget_ledger.json').read_text())
        assert live['intervals'][1]['arm']=='B8'
    result=json.loads((tmp_path/'budget_ledger.json').read_text())
    assert result['intervals'][0]==previous and result['intervals'][1]['arm']=='B8'
    assert 'end' in result['intervals'][1]


def test_failed_preflight_closes_only_own_inherited_lease(tmp_path,monkeypatch):
    import os
    monkeypatch.setenv('ROUND8_LEASE_TOKEN','ours')
    ledger={'intervals':[{'id':'ours','pid':os.getpid(),'stage':'train','device_uuid':'a','start':1.},{'id':'other','pid':os.getpid()+1,'stage':'train','device_uuid':'b','start':1.}]}
    (tmp_path/'budget_ledger.json').write_text(json.dumps(ledger))
    runner.release_inherited_lease(tmp_path)
    saved=json.loads((tmp_path/'budget_ledger.json').read_text())
    assert 'end' in saved['intervals'][0] and 'end' not in saved['intervals'][1]
    end=saved['intervals'][0]['end']
    runner.release_inherited_lease(tmp_path)
    assert json.loads((tmp_path/'budget_ledger.json').read_text())['intervals'][0]['end']==end
