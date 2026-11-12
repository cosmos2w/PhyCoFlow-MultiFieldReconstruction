import json
from pathlib import Path
import pytest
from phycoflow_reconstruction.training.round8_calibration import InternalCalibration, BLOCKS
from phycoflow_reconstruction.coherence.round8_queries import SharedQuerySampler
from phycoflow_reconstruction.training.round8_log_recovery import restore_journals


def make_calibration():
    keys=[f'{family}.{block}' for family,blocks in BLOCKS.items() for block in blocks]
    return InternalCalibration(dict.fromkeys(keys,1.),dict.fromkeys(keys,2.),dict.fromkeys(keys,True))


def lines(path, rows):
    path.write_text(''.join(json.dumps(x)+'\n' for x in rows))


def exposure(sampler):
    recorded=[]
    sampler.draw(); sampler.finish(exposure_sink=recorded.append)
    return recorded


def test_recover_both_journals_and_preserve_exact_tail(tmp_path):
    logs=tmp_path/'logs';logs.mkdir()
    cal=make_calibration(); norms={k:1. for k in cal.active}
    cal.observe(1,norms); cal.observe(3,norms)
    stale_cal=[{'epoch':6,'norms':norms},{'epoch':8,'norms':norms}]
    cal_rows=cal.records+stale_cal
    sampler=SharedQuerySampler(['a','b','c'],seed=5)
    retained=exposure(sampler)+exposure(sampler)
    checkpoint=SharedQuerySampler.from_state_dict(sampler.state_dict())
    stale=exposure(sampler)+exposure(sampler)[:1]  # interruption during final exposure write
    cal_path=logs/'round8_calibration.jsonl'; exp_path=logs/'round8_query_exposure.jsonl'
    lines(cal_path,cal_rows); lines(exp_path,retained+stale)
    original_cal=cal_path.read_bytes(); original_exp=exp_path.read_bytes()
    out=restore_journals(tmp_path,cal,checkpoint,'candidate')
    expected_cal=''.join(json.dumps(x)+'\n' for x in cal.records).encode()
    expected_exp=''.join(json.dumps(x)+'\n' for x in retained).encode()
    assert cal_path.read_bytes()==expected_cal and exp_path.read_bytes()==expected_exp
    assert Path(out['calibration_archive']).read_bytes()==original_cal[len(expected_cal):]
    assert Path(out['exposure_archive']).read_bytes()==original_exp[len(expected_exp):]
    assert restore_journals(tmp_path,cal,checkpoint,'candidate') == {'calibration_archive':None,'exposure_archive':None}


def test_frozen_fork_keeps_calibration_and_missing_exposure(tmp_path):
    logs=tmp_path/'logs';logs.mkdir();cal=make_calibration(); norms={k:1. for k in cal.active}
    for epoch in (1,3,6,8,10): cal.observe(epoch,norms)
    cal.freeze(10)
    freeze={'event':'frozen','epoch':10,'coefficients':cal.coefficients,'priorities':cal.priorities,'units':cal.units}
    path=logs/'round8_calibration.jsonl';lines(path,cal.records+[freeze]);before=path.read_bytes()
    sampler=SharedQuerySampler(['a','b'])
    assert restore_journals(tmp_path,cal,sampler,'candidate') == {'calibration_archive':None,'exposure_archive':None}
    assert path.read_bytes()==before


def test_recovery_validates_both_logs_before_mutating(tmp_path):
    logs=tmp_path/'logs';logs.mkdir();cal=make_calibration();norms={k:1. for k in cal.active};cal.observe(1,norms)
    cal_path=logs/'round8_calibration.jsonl';lines(cal_path,cal.records+[{'epoch':3,'norms':norms}])
    sampler=SharedQuerySampler(['a','b']); retained=exposure(sampler)
    exp_path=logs/'round8_query_exposure.jsonl';lines(exp_path,retained)
    exp_path.write_bytes(exp_path.read_bytes()+b'{broken')
    before=cal_path.read_bytes()
    with pytest.raises(ValueError,match='incomplete'): restore_journals(tmp_path,cal,sampler,'candidate')
    assert cal_path.read_bytes()==before
    assert sorted(p.name for p in logs.iterdir()) == ['round8_calibration.jsonl','round8_query_exposure.jsonl']


def test_recovery_rejects_missing_or_tampered_inner_exposure_provenance(tmp_path):
    logs=tmp_path/'logs';logs.mkdir();cal=make_calibration()
    sampler=SharedQuerySampler(['a','b']); entries=exposure(sampler)
    path=logs/'round8_query_exposure.jsonl'
    modified=[dict(row) for row in entries]; modified[0].pop('inner_sign')
    lines(path,modified);before=path.read_bytes()
    with pytest.raises(ValueError,match='inner sampling'):
        restore_journals(tmp_path,cal,sampler,'candidate')
    assert path.read_bytes()==before
