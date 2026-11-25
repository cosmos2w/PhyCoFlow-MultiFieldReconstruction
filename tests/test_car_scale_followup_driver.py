"""Fork phase clocks, original risk binding and completed-record checks."""
from copy import deepcopy
import hashlib
import json
import pytest
from scripts.posttraining.run_car_scale_followup import (CONTROL,CANDIDATE,target_step,validate_config,verify_origin,completed_epoch_evidence)


def test_config_preserves_original_risk_and_limits_clock(tmp_path):
    path=tmp_path/'reference.json';path.write_text('{}');sha=hashlib.sha256(path.read_bytes()).hexdigest()
    budget={'metric':'physical_mean_risk_model_units','reference_artifact':{'path':str(path),'sha256':sha},'member_allowance_fraction':.1}
    parent={'optimization':{'epochs':500,'steps_per_epoch':12,'multitask':{'fidelity_budget':budget}},
            'runtime':{'plot_every_steps':10},'objectives':{'data_retention':{'enabled':False,'weight':0.}}}
    control=deepcopy(parent);candidate=deepcopy(parent);candidate['optimization']['multitask']['fidelity_budget']['rotating_complement']={'points':512,'seed':20261125}
    assert validate_config(control,parent,CONTROL)==budget['reference_artifact']
    assert validate_config(candidate,parent,CANDIDATE)==budget['reference_artifact']
    assert target_step(100)==7200 and target_step(500)==12000
    candidate['optimization']['multitask']['fidelity_budget']['member_allowance_fraction']=.2
    with pytest.raises(ValueError,match='reference or authorized'):validate_config(candidate,parent,CANDIDATE)
    with pytest.raises(ValueError,match='endpoint'):target_step(200)


def test_origin_and_100_epoch_evidence_preserve_phase_history(tmp_path):
    parent=tmp_path/'parent';parent.mkdir();origin={'parent_run':str(parent.resolve()),'parent_checkpoint_sha256':'hash','absolute_step':6000,
        'scientific_transition':'rotating_complement_field_quadrature_512_v1'}
    assert verify_origin({'fork_origin':origin},{'fork_origin':origin},parent,'hash',CANDIDATE)==origin
    with pytest.raises(ValueError,match='origin identity'):verify_origin({'fork_origin':origin},{'fork_origin':origin},parent,'hash',CONTROL)
    run=tmp_path/'fork';(run/'logs').mkdir(parents=True);(run/'checkpoints').mkdir()
    def write_rows(name,rows):(run/'logs'/name).write_text(''.join(json.dumps(x)+'\n' for x in rows))
    write_rows('history.jsonl',[{'epoch':i,'epoch_complete':True} for i in range(1,101)])
    write_rows('epoch_calibrated_pi_updates.jsonl',[{'step':i} for i in range(6001,7201)])
    write_rows('epoch_timing.jsonl',[{'epoch':i,'complete_epoch_duration':True,'wall_seconds':2.} for i in range(1,101)])
    (run/'logs/recording_definitions.json').write_text('{}')
    for ext in ('png','pdf'):(run/f'logs/loss_history.{ext}').write_text('curve')
    (run/'checkpoints/epoch_100.pt').write_text('checkpoint')
    out=completed_epoch_evidence(run,100,7200)
    assert out['history_rows']==100 and out['pi_action_rows']==1200 and out['complete_epoch_mean_wall_seconds']==2.
    write_rows('epoch_calibrated_pi_updates.jsonl',[{'step':i} for i in range(6001,7200)])
    with pytest.raises(ValueError,match='journal incomplete'):completed_epoch_evidence(run,100,7200)
