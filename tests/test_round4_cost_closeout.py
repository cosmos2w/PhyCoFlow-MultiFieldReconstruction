import pytest
from scripts.posttraining.closeout_car_round4_cost import add_cpu_only_auxiliary, add_cpu_resume_auxiliary, add_standalone_review_auxiliary, aggregate, enclosing_elapsed, validate_budget_stop, TEACHER, NATIVE


def data():
    pairs = [('qualify', 27, {'command':'qualify'}), ('generate', 300, {}),
             ('teacher_numerical_qualification', 9, {}),
             ('endpoint_teacher', 100, {'status':'paused','cpu_worker_seconds':80}),
             ('endpoint_teacher', 200, {'status':'completed','completed_epochs':100,'cpu_worker_seconds':250}),
             (TEACHER, 500, {'status':'completed','attempt':'Attempt_One','completed_epochs':100}),
             (NATIVE, 510, {'status':'completed'}),
             ('round4_assessment', 40, {'status':'completed','arm':TEACHER,'epoch':100})]
    ledger={'active':{},'closed':[]}; jobs=[]
    for index, (phase,seconds,extra) in enumerate(pairs):
        token=f'job{index}'
        entry={'token':token,'phase':phase,'seconds':seconds}
        if phase=='endpoint_teacher': entry['cpu_seconds'] = 10 if seconds == 100 else 20
        ledger['closed'].append(entry)
        jobs.append(({'path':token,'sha256':token},{'token':token,'status':'completed',
                    'wall_seconds':seconds,'process_cpu_seconds':seconds/2,**extra}))
    failed=[({'path':'early','sha256':'early'},{'conservative_gpu_seconds_charge':842})]
    return ledger,jobs,failed


def test_disjoint_cost_and_cumulative_teacher_cpu():
    ledger,jobs,failed=data()
    result=aggregate(ledger,jobs,failed,attempt='Attempt_One',epoch=100)
    assert result['missing'] == []
    assert result['components'] == {'source_generation':327, 'teacher':300, 'student':500,
                                    'assessment':40, 'candidate_overhead':851}
    assert result['cpu_by_component']['teacher'] == 30
    assert result['cost_basis'] == 'conservative_upper_bound'
    assert result['bounded_unknown_components'][0]['gpu_seconds_upper_bound'] == 842
    assert result['excluded_jobs'][0]['phase'] == NATIVE


def test_missing_and_ambiguous_are_not_zero():
    ledger,jobs,failed=data()
    del jobs[1]
    result=aggregate(ledger,jobs,failed,attempt='Attempt_One',epoch=100)
    assert any('receipt missing' in item for item in result['missing'])
    ledger,jobs,failed=data()
    failed[0][1].pop('conservative_gpu_seconds_charge')
    result=aggregate(ledger,jobs,failed,attempt='Attempt_One',epoch=100)
    assert any('unbounded' in item for item in result['missing'])


def test_teacher_only_does_not_require_students():
    ledger,jobs,failed=data()
    ledger['closed']=ledger['closed'][:5]; jobs=jobs[:5]
    result=aggregate(ledger,jobs,failed,attempt=None,epoch=100,teacher_only=True)
    assert result['missing'] == []
    assert result['components']['teacher'] == 300


def test_real_native_receipt_attempt_and_enclosing_failed_bound():
    ledger,jobs,failed=data()
    native=next(job for _,job in jobs if job.get('attempt') == 'Attempt_One')
    native.update(schema='car_round4_native_job_v1', path=TEACHER,
                  run_dir='/campaign/runs/formal/FFM_SSS_Target_Amortization/Attempt_One')
    assert aggregate(ledger,jobs,failed,attempt='Attempt_One',epoch=100)['missing'] == []
    native.pop('attempt')
    assert 'named learner attempt endpoint job missing' in aggregate(ledger,jobs,failed,attempt='Attempt_One',epoch=100)['missing']
    clocks=[({'path':'measured'}, {'start_unix':100., 'end_unix':130.})]
    failed_clock=[({'path':'failed'}, {'launch_utc':'1970-01-01T00:01:30Z',
                  'failure_discovered_before_utc':'1970-01-01T00:01:40Z'})]
    seconds, basis=enclosing_elapsed(clocks,failed_clock)
    assert seconds == 40 and 'including gaps' in basis and 'upper bounds' in basis


def test_cpu_only_auxiliary_keeps_prior_unknowns_and_disjoint_cpu():
    entries, clocks, unknown = [], [], []
    ref = {'path':'copy.json','sha256':'copy'}
    add_cpu_only_auxiliary(entries, clocks, unknown, scope='SOURCE copy', ref=ref,
        doc={'status':'completed','gpu_used':False,'start_unix':1.,'end_unix':3.,
             'wall_seconds':2.,'process_cpu_seconds':5.}, cpu_key='process_cpu_seconds')
    validation = {'status':'completed','gpu_used':False,'wall_seconds':3.,'cpu_process_seconds':4.,
                  'previous_failed_attempts':[{'cpu_seconds':None}, {'cpu_seconds':0.5,'wall_seconds':0.1},
                                              {'cpu_seconds':5.,'wall_seconds':4.}]}
    add_cpu_only_auxiliary(entries, clocks, unknown, scope='fixed-case validation',
        ref={'path':'validation.json','sha256':'validation'},doc=validation,cpu_key='cpu_process_seconds')
    assert sum(x['process_cpu_seconds'] for x in entries if x['process_cpu_seconds'] is not None) == 14.5
    assert len(clocks) == 1
    assert {(x['scope'],x['timer']) for x in unknown} == {
        ('fixed-case validation: prior attempt 1','process CPU'),
        ('fixed-case validation: prior attempt 1','wall'),
        ('fixed-case validation','absolute start/end')}


def test_resume_cpu_counts_parent_and_waited_children_once_and_preserves_unknown():
    entries, clocks, unknown = [], [], []
    ref={'path':'resume_receipt.json','sha256':'bound'}
    doc={'schema':'car_round4_native_real_ffm_resume_cpu_v1','gpu_used':False,
         'status':'completed','start_unix':1.,'end_unix':5.,'wall_seconds':4.,
         'parent_process_cpu_seconds':2.,'waited_children_cpu_seconds':7.,
         'children':[{'process_cpu_seconds':7.}]}
    add_cpu_resume_auxiliary(entries,clocks,unknown,ref=ref,doc=doc)
    assert entries[0]['process_cpu_seconds'] == 9.
    assert len(entries) == 1 and len(clocks) == 1 and unknown == []
    entries, clocks, unknown = [], [], []
    doc.update(status='failed', waited_children_cpu_seconds=None)
    add_cpu_resume_auxiliary(entries,clocks,unknown,ref=ref,doc=doc)
    assert entries[0]['status'] == 'failed' and entries[0]['process_cpu_seconds'] is None
    assert any(item['timer'] == 'waited children CPU' for item in unknown)


def test_standalone_review_and_visual_exclude_nested_teacher_cost_and_keep_unknowns():
    entries, clocks, unknown = [], [], []
    ref={'path':'forecast.json','sha256':'bound'}
    forecast={'schema':'car_round4_teacher_warm_cache_forecast_v1','train_manifest_sha256':'manifest',
              'review_timer':{'start_unix':10.,'end_unix':12.,'process_cpu_seconds':3.},
              'teacher_cpu_worker_seconds':9000.}
    add_standalone_review_auxiliary(entries,clocks,unknown,ref=ref,doc=forecast,
        kind='warm_forecast',manifest_sha256='manifest')
    visual={'schema':'car_round4_epoch1_history_visual_v1','scope':'CPU-only static rendering',
            'wall_start_unix':13.,'wall_end_unix':15.,'elapsed_wall_seconds':2.,
            'process_cpu_seconds':4.,'earlier_setup_attempts':[{'process_cpu_seconds':None,
                                                                 'enclosing_wall_seconds_rounded':3.5}]}
    add_standalone_review_auxiliary(entries,clocks,unknown,ref={'path':'visual','sha256':'visual'},
        doc=visual,kind='visual_recovery',manifest_sha256='manifest')
    assert sum(row['process_cpu_seconds'] for row in entries if row['process_cpu_seconds'] is not None) == 7.
    assert len(clocks) == 2 and entries[-1]['wall_precision'] == 'rounded'
    assert {row['timer'] for row in unknown} == {'process CPU','absolute start/end'}


def test_partial_teacher_requires_paused_epoch_and_bound_exceeded_forecast():
    decision={'schema':'car_round4_teacher_budget_stop_v1', 'decision':'stop_cost_infeasible',
              'train_manifest_sha256':'manifest', 'completed_epochs':7,
              'checkpoint':{'sha256':'checkpoint'}, 'history':{'sha256':'history'},
              'projected_teacher_gpu_seconds':31000., 'projected_teacher_cpu_worker_seconds':20000.,
              'teacher_gpu_cap_seconds':30400, 'teacher_cpu_cap_seconds':21600}
    forecast={'schema':'car_round4_teacher_warm_cache_forecast_v1', 'train_manifest_sha256':'manifest',
              'completed_epochs':7, 'remaining_epochs':93, 'teacher_checkpoint_sha256':'checkpoint',
              'teacher_history_sha256':'history', 'teacher_gpu_cap_seconds':30400,
              'teacher_cpu_cap_seconds':21600, 'safety_factor':1.2, 'training_window_epochs':[3,4,5,6,7],
              'base_teacher_gpu_allowance_seconds':14400, 'variation_reserve_gpu_seconds':18000,
              'reserve_committed_elsewhere_gpu_seconds':2000, 'candidate_other_gpu_seconds':3000,
              'whole_other_gpu_seconds':20000, 'candidate_gpu_cap_seconds':35317,
              'whole_gpu_cap_seconds':86400,
              'projected_teacher_gpu_seconds':31000., 'projected_teacher_cpu_worker_seconds':20000.}
    assert validate_budget_stop(decision, forecast, 'manifest') == 7
    with pytest.raises(ValueError): validate_budget_stop(decision, {**forecast,'projected_teacher_gpu_seconds':10000}, 'manifest')
    with pytest.raises(ValueError): validate_budget_stop({**decision,'completed_epochs':100}, forecast, 'manifest')
    with pytest.raises(ValueError):
        validate_budget_stop({**decision,'projected_teacher_gpu_seconds':15000}, {**forecast,'projected_teacher_gpu_seconds':15000}, 'manifest')
    with pytest.raises(ValueError):
        validate_budget_stop({**decision,'teacher_gpu_cap_seconds':14400}, {**forecast,'teacher_gpu_cap_seconds':14400}, 'manifest')
    constrained={**forecast, 'candidate_other_gpu_seconds':20000, 'teacher_gpu_cap_seconds':15317}
    assert validate_budget_stop({**decision,'teacher_gpu_cap_seconds':15317}, constrained, 'manifest') == 7
    cpu_stop={**decision,'projected_teacher_gpu_seconds':15000, 'projected_teacher_cpu_worker_seconds':22000}
    cpu_forecast={**forecast,'projected_teacher_gpu_seconds':15000, 'projected_teacher_cpu_worker_seconds':22000}
    assert validate_budget_stop(cpu_stop, cpu_forecast, 'manifest') == 7
    ledger,jobs,failed=data()
    ledger['closed']=ledger['closed'][:5]; jobs=jobs[:5]
    jobs[4][1].update(status='paused',completed_epochs=7)
    partial=aggregate(ledger,jobs,failed,attempt=None,epoch=7,teacher_only=True,partial_teacher=True)
    assert partial['missing'] == [] and partial['components']['teacher'] == 300
    normal=aggregate(ledger,jobs,failed,attempt=None,epoch=100,teacher_only=True)
    assert 'completed 100-epoch teacher job missing' in normal['missing']
