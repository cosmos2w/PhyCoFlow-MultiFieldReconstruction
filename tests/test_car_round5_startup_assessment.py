import pytest

from scripts.qualification.assess_car_round5_startup import analyze_replay, candidate_actions, coupling, assessment_forecast


def replay_row(sample_id,lr,decrease,tol=.001):
    return {'sample_ids':[sample_id],'sample_count':1,'method':'Direct_Scalar_AdamW','lr':lr,
        'update_accepted':True,'committed_displacement_norm':1,'scalar_displacement_dot':-1,
        'before':{'direct_scalar':1},'same_query_after':{'direct_scalar':1-decrease},
        'independent_query_before':{'direct_scalar':1},'independent_query_after':{'direct_scalar':.98},
        'scalar_replay_before':1,'scalar_replay_precision':tol/5,'scalar_descent_tolerance':tol}


def test_replay_gates_are_aggregate_with_per_lr_reporting():
    ids=[str(i) for i in range(10)]
    rows=[replay_row(i,1e-5,.01) for i in ids]+[replay_row(i,1e-4,.01 if int(i)<6 else 0) for i in ids]
    per,aggregate,overshoot=analyze_replay(rows,ids,(1e-5,1e-4))
    assert per['0.0001']['descent_fraction']==.6
    assert aggregate['descent_fraction']==.8 and aggregate['passed'] and not overshoot
    rows[-1]['same_query_after']['direct_scalar']=1.01
    _,aggregate,overshoot=analyze_replay(rows,ids,(1e-5,1e-4))
    assert aggregate['passed'] and overshoot
    rows.pop()
    with pytest.raises(ValueError,match='matched TRAIN800'):
        analyze_replay(rows,ids,(1e-5,1e-4))


def test_matched_native_inputs_and_candidate_actions():
    identity={'sample_ids':['one'],'observation_sha256':'a','query_sha256':'b',
              'native_seed':13,'native_generator_state_after_sha256':'c'}
    assert coupling([{'native_coupling':identity}],[{'native_coupling':dict(identity)}])
    wrong=dict(identity,native_seed=14)
    with pytest.raises(ValueError,match='coupled inputs'):
        coupling([{'native_coupling':identity}],[{'native_coupling':wrong}])
    actions=[{'action_norm':1,'action_dot':dot,'update_accepted':ok,'action_alpha':alpha,
              'field_source_ensemble_checks':1,'field_post_ensemble_checks':1}
             for dot,ok,alpha in [(-1,True,1),(-1,True,.5),(1,False,0)]]
    result=candidate_actions(actions,0)
    assert result['acceptance_fraction']==2/3 and result['ascending_proposal_fraction']==1/3
    assert result['action_counts']=={'full':1,'half':1,'rejected':1}


def test_historical_cpu_forecast_uses_saved_timings():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/'cases/turbulent_combustion/runs/coherence_adaptive_repair/round_3'
    if not root.exists():pytest.skip('historical receipts unavailable')
    result=assessment_forecast(root)
    assert result['worker_hours']>0 and result['wall_seconds']>result['tail_seconds']>0
    assert len(result['cpu_timing_receipts'])==2 and len(result['generation_receipts'])==4


def test_assessment_writes_launcher_compatible_qualification(tmp_path):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    import yaml
    from scripts.qualification import assess_car_round5_startup as assess
    round3=Path(__file__).resolve().parents[1]/'cases/turbulent_combustion/runs/coherence_adaptive_repair/round_3'
    if not round3.exists():pytest.skip('historical receipts unavailable')
    def write(path,data):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data));return path
    def jsonlines(path,data):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(x)+'\n' for x in data))
    ids=[str(i) for i in range(800)]
    unitdir=tmp_path/'units';unitdir.mkdir();panel=write(unitdir/'panel.json',{'sample_ids':ids})
    units=write(unitdir/'result.json',{'schema':'car_round5_source_units_v1','status':'completed','sample_count':800,
        'split':'train','members':2,'floor_active':{},'panel_sha256':assess.digest(panel),
        'units':dict(native=1,state=1,scale=1,shape=1)})
    actiondir=tmp_path/'actions';actiondir.mkdir()
    replay=[replay_row(i,lr,.01) for lr in (5e-6,5e-5) for i in ids]
    jsonlines(actiondir/'actions.jsonl',replay)
    action=write(actiondir/'result.json',{'schema':'car_round5_action_replay_v1','status':'completed','sample_count':800,
        'units_sha256':assess.digest(units),'field_tolerance_max':[0]*5})
    identities=[{'sample_ids':[str(i)],'observation_sha256':str(i),'query_sha256':'q',
        'native_seed':i,'native_generator_state_after_sha256':'s'} for i in range(24)]
    paths={}
    for role,mode,epochs,key in [('candidate','safeguarded',2,'Native_GT_SSS_Safeguarded_Continuation'),
                                  ('native','native',2,'Native_GT_Continuation'),
                                  ('historical',None,1,'Historical_SourceScaledRisk_Timing')]:
        run=tmp_path/role/'run';(run/'logs').mkdir(parents=True)
        opt={'lr':5e-5,'lr_warmup':{'start_lr':5e-6},'adam_betas':[.9,.999],
             'batch_size':64,'steps_per_epoch':12,'sampling':'full_pass'}
        if mode:opt['direct_supervision']={'mode':mode,'units':dict(native=1,state=1,scale=1,shape=1),'dot_tolerance':0,'field_tolerance':[0]*5}
        cfg=tmp_path/role/'config.yaml';cfg.write_text(yaml.safe_dump({'optimization':opt}))
        jsonlines(run/'logs/epoch_timing.jsonl',[{'epoch':i,'step':12*i,'wall_seconds':10+i} for i in range(1,epochs+1)])
        if mode:
            updates=[]
            for i in range(24):
                row={'step':i+1,'native_coupling':identities[i],'native_data_loss':1,'optimizer_lr':5e-6}
                if mode=='safeguarded':row.update(direct_scalar=2,action_norm=1,action_dot=-1,update_accepted=True,
                    action_alpha=1,field_source_ensemble_checks=1,field_post_ensemble_checks=1)
                updates.append(row)
            jsonlines(run/'logs/direct_updates.jsonl',updates)
        receipt=write(tmp_path/role/'trace.json',{'schema':'car_round5_training_trace_v1','status':'completed',
            'config':str(cfg),'run_dir':str(run),'reporting_epochs':epochs,'start_step':0,'target_step':epochs*12,
            'historical_timing':role=='historical','lease_token':role,'started_unix':10})
        lease=write(tmp_path/role/'lease.json',{'schema':'car_round5_lease_v1','status':'completed','token':role,'device':{'uuid':'GPU-test'},'start_unix':9,'end_unix':11})
        paths[role]=(receipt,lease,cfg,key)
    prep=write(tmp_path/'prep.json',{'schema':'car_round5_startup_preparation_v1','kind':'initial',
        'units_sha256':assess.digest(units),'action_replay_sha256':assess.digest(action),
        'configs':{key:assess.ref(cfg) for _,_,cfg,key in paths.values()}})
    testfile=write(tmp_path/'test.py',{'test':'placeholder'})
    resume=write(tmp_path/'resume.json',{'schema':'car_round5_resume_qualification_v1','status':'completed','exit_code':0,
        'fresh_process':True,'saved_config_reload':True,'resume_and_append':True,'forced_rejection_then_acceptance':True,
        'state_rng_sampler_and_recording_match':True,'command':'pytest specific test','test_file':assess.ref(testfile)})
    args=SimpleNamespace(units=units,action_replay=action,preparation=prep,
        candidate_trace=paths['candidate'][0],candidate_lease=paths['candidate'][1],
        native_trace=paths['native'][0],native_lease=paths['native'][1],
        historical_trace=paths['historical'][0],historical_lease=paths['historical'][1],
        resume_receipt=resume,round3=round3,output=tmp_path/'result.json')
    result=assess.assess(args)
    assert result['qualified'] and result['native_input_identity_matched']
    assert result['cpu_assessment_forecast']['wall_seconds']>0
    assert result['direct_epoch_seconds']==11.5 and result['historical_epoch_seconds']==11

    # An epoch-one fresh-process prefix and its resumed epoch-two trace must
    # be bound to the same run, config, and measured lease.
    candidate_receipt=json.loads(paths['candidate'][0].read_text())
    saved_config=Path(candidate_receipt['run_dir'])/'resolved_config.yaml'; saved_config.write_text('saved: true\n')
    candidate_receipt.update(start_step=12, resume=candidate_receipt['run_dir'], saved_config_reloaded=True, saved_config_sha256=assess.digest(saved_config))
    paths['candidate'][0].write_text(json.dumps(candidate_receipt))
    args.output=tmp_path/'resumed_result.json'
    with pytest.raises(ValueError,match='epoch-one trace receipt'):
        assess.assess(args)
    prefix=write(tmp_path/'candidate'/'prefix.json',{
        **candidate_receipt, 'start_step':0, 'target_step':12, 'reporting_epochs':1, 'resume':None})
    args.candidate_prefix_trace=prefix
    assert assess.assess(args)['qualified']
    prefix_data=json.loads(prefix.read_text());prefix_data['lease_token']='wrong';prefix.write_text(json.dumps(prefix_data))
    args.output=tmp_path/'wrong_lease_result.json'
    with pytest.raises(ValueError,match='same run, configuration and lease'):
        assess.assess(args)


def test_corrected_replay_must_bind_actual_adamw_and_evidence():
    from scripts.qualification.assess_car_round5_startup import validate_replay_optimizer
    opt={'lr_warmup':{'start_lr':2.5e-6},'lr':2.5e-5,'adam_betas':[0.,.999], 'weight_decay':1e-6,'grad_clip':1.}
    prep={'kind':'correction','correction_evidence_sha256':'evidence'}
    action={'candidate_only':True,'learning_rates':[2.5e-6,2.5e-5],
            'optimizer':{'method':'AdamW','beta1':0.,'beta2':.999,'weight_decay':1e-6,'grad_clip':1.},
            'correction_evidence':{'sha256':'evidence'}}
    validate_replay_optimizer(action,opt,prep)
    with pytest.raises(ValueError,match='candidate-only action replay'):
        validate_replay_optimizer({**action,'optimizer':{**action['optimizer'],'beta1':.9}},opt,prep)
    with pytest.raises(ValueError,match='correction evidence'):
        validate_replay_optimizer({**action,'correction_evidence':{'sha256':'different'}},opt,prep)
    # Existing pre-correction replay artifacts predate optimizer metadata.
    validate_replay_optimizer({},opt,{'kind':'initial'})
