import json
import pytest


def test_rejected_attempt_is_timed_without_being_counted_as_accepted(tmp_path, monkeypatch):
    from phycoflow_reconstruction.training.monitoring import TrainingMonitor
    monitor = TrainingMonitor(tmp_path, start_step=0, final_step=2, configured_steps=2,
        steps_per_epoch=2, description='consumed rejection', enabled=False)
    monkeypatch.setattr(monitor, '_plot', lambda: None)
    for step, accepted in ((1, False), (2, True)):
        monitor.record({'step':step, 'total':1., 'update_accepted':accepted,
            'attempted_condition_visits':64, 'accepted_objective_condition_visits':64 if accepted else 0})
        monitor.finish_step()
    monitor.close()
    timing = json.loads((tmp_path/'logs/epoch_timing.jsonl').read_text())
    assert timing['attempted_updates'] == 2 and timing['accepted_updates'] == 1
    assert timing['wall_seconds'] >= timing['train_seconds'] > 0

from scripts.posttraining.run_car_round5 import acquire, forecast
from scripts.posttraining.assess_car_round5 import ARMS, finalize_decision


def state():
    return {'closed': [], 'active': {}}


def test_forecast_keeps_review_and_full_budget():
    out = forecast(already=7200, native_epoch_seconds=50, direct_epoch_seconds=200)
    assert out['final_epoch'] == 500
    assert out['forecast_gpu_seconds'] <= 45 * 3600
    out = forecast(already=7200, native_epoch_seconds=80, direct_epoch_seconds=300)
    assert 100 <= out['final_epoch'] < 500
    with pytest.raises(ValueError, match='100 matched'):
        forecast(already=150000, native_epoch_seconds=80, direct_epoch_seconds=300)


def test_overlapping_reservations_and_prep_cap():
    s = state()
    acquire(s, token='one', gpu_uuid='GPU-one', phase='startup_initial', now=0, duration=4000)
    with pytest.raises(ValueError, match='two GPU-hour'):
        acquire(s, token='two', gpu_uuid='GPU-two', phase='startup_initial', now=100, duration=4000)
    with pytest.raises(ValueError, match='already leased'):
        acquire(s, token='two', gpu_uuid='GPU-one', phase='assessment', now=100, duration=1)


def row(regional):
    return {'fidelity': {'primary_pass': True, 'legacy_member_pass': True},
            'normalized_core_differences': {'state': -.06, 'scale': 0, 'shape': 0},
            'reused_tasks': {'regional_normalized_improvement': regional,
                             'event_normalized_difference': 0}}


def test_single_qualified_and_cost_pending():
    result = {'candidates': {ARMS[0]: row(.1)}}
    finalize_decision(result)
    assert result['predeclared_recommendation'] == ARMS[0]
    result = {'candidates': {ARMS[0]: row(.1), ARMS[1]: row(.15)}}
    finalize_decision(result)
    assert result['predeclared_recommendation'].startswith('pending')
    result['candidates'][ARMS[1]]['reused_tasks']['regional_normalized_improvement'] = .11
    finalize_decision(result)
    assert result['predeclared_recommendation'] == ARMS[0]


def test_control_only_forecast_and_terminal_rejection():
    from scripts.posttraining.run_car_round5 import qualification
    out = forecast(already=7200, native_epoch_seconds=80)
    assert out['final_epoch'] == 500
    assert out['direct_epoch_seconds'] is None
    startup = {'schema': 'car_round5_startup_qualification_v1', 'units_sha256': 'hash',
               'qualified': False, 'native_qualified': True, 'candidate_rejection_terminal': True,
               'candidate_rejection_reason': 'bounded action replay failed'}
    assert qualification(startup, 'hash') == [ARMS[0]]
    startup['candidate_rejection_terminal'] = False
    with pytest.raises(ValueError, match='terminal rejection'):
        qualification(startup, 'hash')


def test_prepare_configurations_control_and_qualified_grouping(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import yaml
    from scripts.posttraining import run_car_round5 as driver
    import scripts.posttraining.prepare_car_round4 as inherited
    import phycoflow_reconstruction.config.validate as validation
    old = tmp_path / 'old.json'
    old.write_text(json.dumps({'risk_reference_panel': {'source_member_mse': [1]*5}}))
    cfg = {'optimization': {'multitask': {'fidelity_budget': {'reference_artifact': driver.reference(old)}}},
           'coherence': {'scalable_anchored_coherence': {'enabled': True}},
           'runtime': {}, 'evaluation': {}, 'output': {}, 'checkpointing': {}, 'source_run': '/source'}
    monkeypatch.setattr(inherited, 'resolve_config', lambda _: (cfg, {'path':'historical','sha256':'x'}))
    monkeypatch.setattr(validation, 'validate_config', lambda config: None)
    units = tmp_path/'units.json'
    units.write_text(json.dumps({'schema':'car_round5_source_units_v1','status':'completed','sample_count':800,
        'units':{'native':1,'state':1,'scale':1,'shape':1},'floor_active':{}}))
    timing = tmp_path/'timing.json'; timing.write_text('{}')
    startup = {'schema':'car_round5_startup_qualification_v1', 'units_sha256':driver.digest(units),
        'qualified':False, 'native_qualified':True, 'candidate_rejection_terminal':True,
        'candidate_rejection_reason':'qualified replay rejected candidate', 'lr_start':5e-6,'lr_target':5e-5,
        'beta1':.9,'beta2':.999,'native_epoch_seconds':100,
        'cpu_assessment_forecast':{'worker_hours':1,'wall_seconds':100,'tail_seconds':20,
                                   'basis':'measured prior scoring','evidence':driver.reference(timing)}}
    start = tmp_path/'start.json'; start.write_text(json.dumps(startup))
    disp = tmp_path/'disp.json'; disp.write_text(json.dumps({'schema':'car_round5_context_estimator_v1',
        'qualified':True, 'query_law':'shared','context_cache':'none'}))
    control = tmp_path/'control'; control.mkdir()
    driver.prepare(SimpleNamespace(units=units,startup=start,disposition=disp),control)
    prep = json.loads((control/'configs/prepared.json').read_text())
    assert prep['active_arms'] == [ARMS[0]] and prep['test_locked']
    native = yaml.safe_load((control/'configs'/f'{ARMS[0]}.yaml').read_text())
    assert native['optimization']['direct_supervision']['mode'] == 'native'
    assert native['coherence']['scalable_anchored_coherence']['enabled'] is False
    assert native['evaluation']['split'] == 'validation'
    startup.update(qualified=True,field_tolerance=[0]*5,dot_tolerance=0,direct_epoch_seconds=100)
    start.write_text(json.dumps(startup))
    disp.write_text(json.dumps({'schema':'car_round5_context_estimator_v1','qualified':True,
        'query_law':'condition_groups_8','context_cache':'static_features'}))
    both = tmp_path/'both'; both.mkdir()
    driver.prepare(SimpleNamespace(units=units,startup=start,disposition=disp),both)
    candidate = yaml.safe_load((both/'configs'/f'{ARMS[1]}.yaml').read_text())
    assert candidate['coherence']['scalable_anchored_coherence']['condition_query_groups'] == 8
    assert candidate['optimization']['direct_supervision']['mode'] == 'safeguarded'
    # A successful correction need not repeat the failure that justified it.
    prepared_trace = tmp_path/'trace_preparation.json'
    prepared_trace.write_text(json.dumps({'schema':'car_round5_startup_preparation_v1','kind':'correction',
        'units_sha256':driver.digest(units),'lr_start':2.5e-6,'lr_target':2.5e-5,'beta1':0.,
        'context_cache':'static_features','query_law':'condition_groups_8','correction_evidence_sha256':'evidence'}))
    startup.update(lr_start=2.5e-6,lr_target=2.5e-5,beta1=0.,
        finite_step_overshoot_demonstrated=False,ascending_proposal_fraction=0.,
        evidence={'startup_preparation':driver.reference(prepared_trace)})
    start.write_text(json.dumps(startup))
    corrected = tmp_path/'corrected';corrected.mkdir()
    driver.prepare(SimpleNamespace(units=units,startup=start,disposition=disp),corrected)
    candidate = yaml.safe_load((corrected/'configs'/f'{ARMS[1]}.yaml').read_text())
    assert candidate['optimization']['lr'] == 2.5e-5
    assert candidate['optimization']['adam_betas'][0] == 0.


def test_failed_lease_records_child_cpu_and_elapsed(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace
    from scripts.posttraining import run_car_round5 as driver
    monkeypatch.setattr(driver, 'device_identity', lambda: {'uuid':'GPU-example'})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setenv('KEOPS_CACHE_FOLDER','/tmp')
    args = SimpleNamespace(command=[sys.executable,'-c','sum(range(500000)); raise SystemExit(3)'],
        phase='startup_initial', max_seconds=30, correction_evidence=None)
    with pytest.raises(Exception):
        driver.lease(args,tmp_path)
    result = json.loads(next((tmp_path/'operations/completions').glob('*.json')).read_text())
    assert result['status'] == 'failed' and result['seconds'] > 0
    assert result['child_user_cpu_seconds'] >= 0 and result['child_system_cpu_seconds'] >= 0
    assert 'do not add' in result['cpu_accounting']


def test_prepared_configs_pass_real_historical_validator(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    import scripts.posttraining.prepare_car_round4 as historical
    from scripts.posttraining import run_car_round5 as driver
    path = Path(__file__).resolve().parents[1] / 'cases/turbulent_combustion/runs/coherence_adaptive_repair/round_3'
    if not (path/'configs/prepared.json').is_file():
        pytest.skip('historical campaign payload unavailable')
    real = historical.resolve_config
    monkeypatch.setattr(historical, 'resolve_config', lambda _: real(path))
    units = tmp_path/'units.json'
    units.write_text(json.dumps({'schema':'car_round5_source_units_v1','status':'completed','sample_count':800,
        'units':{'native':1,'state':1,'scale':1,'shape':1},'floor_active':{}}))
    evidence = tmp_path/'timing.json'; evidence.write_text('{}')
    startup = {'schema':'car_round5_startup_qualification_v1', 'qualified':True,
        'units_sha256':driver.digest(units), 'field_tolerance':[0]*5, 'dot_tolerance':0,
        'lr_start':5e-6,'lr_target':5e-5,'beta1':.9,'beta2':.999,
        'native_epoch_seconds':50,'direct_epoch_seconds':150,
        'cpu_assessment_forecast':{'worker_hours':1,'wall_seconds':100,'tail_seconds':20,
                                   'basis':'test only','evidence':driver.reference(evidence)}}
    start = tmp_path/'start.json'; start.write_text(json.dumps(startup))
    disp = tmp_path/'disp.json'; disp.write_text(json.dumps({'schema':'car_round5_context_estimator_v1',
        'qualified':True,'query_law':'shared','context_cache':'none'}))
    campaign = tmp_path/'campaign';campaign.mkdir()
    driver.prepare(SimpleNamespace(units=units,startup=start,disposition=disp), campaign)
    assert len(json.loads((campaign/'configs/prepared.json').read_text())['arms']) == 2
    # Exercise the actual cached SOURCE and VALIDATION binding and code snapshot
    # in temporary storage, without generating predictions or touching a GPU.
    (tmp_path/'round_2').symlink_to(path.parent/'round_2', target_is_directory=True)
    driver.freeze(campaign)
    frozen=json.loads((campaign/'configs/formal_freeze.json').read_text())
    assert frozen['final_epoch']>=100 and frozen['test_locked'] is True
    assert frozen['cpu_assessment_forecast']['wall_seconds']==100
    assert frozen['forecast']['evaluation_wall_forecast']['seconds'] is None


def test_terminal_task_intervals_include_source_and_matched_arm_pairs():
    from scripts.posttraining.assess_car_round5 import paired_condition_intervals, source_task_values
    source_events=[2.,4.,6.]
    baseline={'scores':{'event':4.}}
    ctx={'baseline':{'reserved_task':{'source_score':2.,'source_per_condition':[1.,2.,3.]}}}
    source_tasks=source_task_values(ctx,baseline,source_events,3)
    def scores(offset):
        return [{'state':v+offset,'scale':v+offset,'shape':v+offset} for v in (1.,2.,3.)]
    source={'per_condition_scores':scores(0.),'reused_tasks':source_tasks}
    native={'per_condition_scores':scores(-1.),'reused_tasks':{'event_per_condition':[1.,2.,3.],
             'regional_per_condition':[0.,1.,2.]}}
    direct={'per_condition_scores':scores(-2.),'reused_tasks':{'event_per_condition':[0.,1.,2.],
             'regional_per_condition':[-1.,0.,1.]}}
    data={'sample_ids':['one','two','three'],'source':source,'candidates':{ARMS[0]:native,ARMS[1]:direct},
          'normalizers':dict(state=1.,scale=1.,shape=1.),'task_normalizers':dict(event=2.,regional=2.)}
    pairs=paired_condition_intervals(data)
    assert pairs[ARMS[0]]['regional_score_difference_negative_better']['mean']==-.5
    assert pairs[ARMS[1]]['regional_score_difference_negative_better']['mean']==-1.
    assert pairs['direct_minus_native']['regional_score_difference_negative_better']['mean']==-.5
    assert pairs[ARMS[0]]['event_score_difference_negative_better']['interval_95'][1]<0
    review={'sample_ids':data['sample_ids'], 'source':{'per_condition_scores':scores(0)},
            'candidates':{ARMS[0]:{'per_condition_scores':scores(-1)}},'normalizers':data['normalizers']}
    assert not any('regional' in name or 'event' in name for name in paired_condition_intervals(review)[ARMS[0]])
    with pytest.raises(ValueError, match='mean differs'):
        source_task_values(ctx,baseline,[1.,1.,1.],3)
