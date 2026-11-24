import pytest

from scripts.posttraining.run_car_round6 import acquire, CAP_SECONDS, PREPARATION_SECONDS


def test_round6_budget_sums_devices_and_failed_preparation():
    state = {'closed': [{'phase': 'preparation', 'seconds': 1200, 'status': 'failed'}], 'active': {}}
    acquire(state, token='first', device={'uuid': 'runtime-device-1'}, phase='preparation', duration=600, now=0)
    with pytest.raises(ValueError, match='preparation'):
        acquire(state, token='second', device={'uuid': 'runtime-device-2'}, phase='preparation', duration=PREPARATION_SECONDS, now=0)
    acquire(state, token='second', device={'uuid': 'runtime-device-2'}, phase='formal_training', duration=CAP_SECONDS-1800, now=0)
    with pytest.raises(ValueError, match='30 occupied'):
        acquire(state, token='third', device={'uuid': 'runtime-device-3'}, phase='assessment', duration=1, now=0)


def test_round6_rejects_two_leases_on_same_device():
    state = {'closed': [], 'active': {}}
    acquire(state, token='first', device={'uuid': 'runtime-device-1'}, phase='preparation', duration=60, now=0)
    with pytest.raises(ValueError, match='already leased'):
        acquire(state, token='second', device={'uuid': 'runtime-device-1'}, phase='assessment', duration=60, now=0)


def test_measured_response_admission_keeps_original_audit_and_binds_probe(tmp_path):
    import hashlib
    import json
    from scripts.posttraining.run_car_round6 import controller_admission
    def ref(name, value):
        path=tmp_path/name;path.write_text(json.dumps(value));return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    probe=ref('probe.json', {'status':'completed'})
    joint=ref('joint.json', {'status':'completed','controller_qualification':{'qualified':False,'gamma_star':0.},'evidence':{'parity':probe}})
    revision={'schema':'car_round6_measured_response_qualification_v1','status':'completed','qualified':True,'gamma':2.5,'rationale':'Measured response supports a bounded candidate.', 'evidence':{'joint_audit':joint,'production_probe':probe}}
    assert controller_admission(revision)==(2.5,'revised measured-response qualification')
    assert controller_admission({'status':'completed','controller_qualification':{'qualified':True,'gamma_star':1.}})==(1.,'original target-fit qualification')
    with pytest.raises(ValueError,match='measured controller'):
        controller_admission({'status':'completed','controller_qualification':{'qualified':False,'gamma_star':0.}})
    with pytest.raises(ValueError,match='positive finite'):
        controller_admission({**revision,'gamma':0.})
    (tmp_path/'probe.json').write_text('{}')
    with pytest.raises(ValueError,match='bound artifact changed'):
        controller_admission(revision)


def test_forecast_requires_complete_epoch_durations_and_accepts_legacy(tmp_path):
    import json
    from types import SimpleNamespace
    from scripts.posttraining.run_car_round6 import forecast, validate_pilot_epoch_timing
    timing=[{'version':'complete_epoch_timing_v1','epoch':i,'step':12*i,
             'attempted_updates':12,'accepted_updates':12,'wall_seconds':100.+i}
            for i in range(1,21)]
    assert validate_pilot_epoch_timing(timing)==[100.+i for i in range(1,21)]
    valid=[dict(row,complete_epoch_duration=True,timed_updates=12) for row in timing]
    assert validate_pilot_epoch_timing(valid)==[100.+i for i in range(1,21)]
    pilot=tmp_path/'pilot.json';pilot.write_text(json.dumps({'qualified':True,'epochs':20,'epoch_timing':timing}))
    (tmp_path/'configs').mkdir()
    forecast(SimpleNamespace(pilot=pilot),tmp_path)
    result=json.loads((tmp_path/'configs/measured_forecast.json').read_text())
    assert result['formal_training_gpu_seconds']==pytest.approx(500*max(120.,1.1*115.5))
    partial=[dict(row) for row in timing];partial[4]['timed_updates']=7
    pilot.write_text(json.dumps({'qualified':True,'epochs':20,'epoch_timing':partial}))
    (tmp_path/'configs/measured_forecast.json').unlink()
    with pytest.raises(ValueError,match='timed_updates must equal 12'):
        forecast(SimpleNamespace(pilot=pilot),tmp_path)
    assert not (tmp_path/'configs/measured_forecast.json').exists()
    for key,value in [('complete_epoch_duration',False),('step',59),('epoch',6),('accepted_updates',11),('attempted_updates',11),('wall_seconds',float('nan'))]:
        altered=[dict(row) for row in timing];altered[4][key]=value
        with pytest.raises(ValueError):validate_pilot_epoch_timing(altered)
