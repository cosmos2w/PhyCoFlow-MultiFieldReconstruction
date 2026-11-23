import json
import pytest
from scripts.posttraining.review_car_round5_100 import completed_run, ARMS


def test_review_requires_recorded_epoch100_and_milestone(tmp_path):
    arm=ARMS[0];campaign=tmp_path/'campaign';run=campaign/'runs/formal'/arm/'attempt'
    (run/'logs').mkdir(parents=True);(run/'checkpoints').mkdir()
    (run/'status.json').write_text(json.dumps({'global_step':1200,'steps_per_epoch':12}))
    (run/'checkpoints/epoch_100.pt').write_bytes(b'checkpoint')
    (run/'logs/direct_updates.jsonl').write_text(''.join(json.dumps({'step':i})+'\n' for i in range(1,1201)))
    (run/'logs/history.jsonl').write_text(''.join(json.dumps({'epoch':i,'epoch_complete':True})+'\n' for i in range(1,101)))
    log=tmp_path/'training.log';log.write_text(str(run)+'\n')
    assert completed_run(log,campaign,arm)[0]==run
    (run/'status.json').write_text(json.dumps({'global_step':1199,'steps_per_epoch':12}))
    with pytest.raises(ValueError,match='stopped before'):
        completed_run(log,campaign,arm)
