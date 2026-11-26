import pytest

from scripts.analysis.report_car_round7 import (ARMS, action_figure, engineering_figure, finite,
    calibration_intervals, localization_figures, readout_templates, resolve_visual_cases,
    shape_component_figures, task_interval_text, template, methods_lines, named_evidence,
    calibration_tables, native_field_tables, validate_matched100, matched100_tables,
    continuation_context_lines, validate_terminal_history)


def save_json(path, data):
    import hashlib
    import json
    path.write_text(json.dumps(data))
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def test_report_template_declares_locked_evidence_and_terminal_tasks():
    payload=template()
    assert len(payload['snapshots']) == 4
    assert {item['split'] for item in payload['snapshots']} == {'train','validation'}
    assert len(payload['terminal_snapshots']) == 2
    assert 'source_calibration_sha256' in readout_templates()['calibration_readout']
    assert 'formal_freeze' in payload
    assert 'matched100_readout' in payload
    with pytest.raises(ValueError, match='finite'):
        finite(float('nan'))


def test_methods_and_named_identities_use_hash_bound_metadata(tmp_path):
    import copy
    campaign=tmp_path/'campaign'
    config={'source':{'weight_view':'ema_materialized'},
        'optimization':{'sampling':'full_pass','batch_size':64,'steps_per_epoch':12,
            'lr_warmup':{'epochs':50},'multitask':{'fidelity_budget':{'rotating_complement':{'points':512}},
                'external_policy':{'lagrangian':{'epoch_calibrated_pi':{'gamma':11.6,'tau_epochs':1.,'horizon_epochs':10.}}}}},
        'coherence':{'scalable_anchored_coherence':{'condition_query_groups':8,'scale_objective':'composite'},
                     'compute_budget':{'point_count':4096}},
        'rollout':{'steps':4,'solver':'euler'},'runtime':{'plot_every_steps':10}}
    candidate=copy.deepcopy(config)
    candidate['coherence']['scalable_anchored_coherence']['scale_objective']='joint_anchor'
    control_ref=save_json(tmp_path/'control.json',config)
    joint_ref=save_json(tmp_path/'joint.json',candidate)
    prior_ref=save_json(tmp_path/'prior.json',{'risk_reference_panel':{'field_order':['CO'],'source_member_mse':[.1]}})
    reference=save_json(tmp_path/'reference.json',{'reference':{'objective_scale':1.86},
        'binding':{'weight_view':'ema_materialized','materialized_view_sha256':'ema-id'},'c0_preserved_from':prior_ref})
    calibration=save_json(tmp_path/'calibration.json',{'kappa':1.2})
    prepared={'arms':{ARMS[0]:control_ref,ARMS[1]:joint_ref},'reference':reference,'calibration':calibration}
    prepared_ref=save_json(tmp_path/'prepared.json',prepared)
    amendment=save_json(tmp_path/'amendment.json',{'qualified_code_root':'/capsules/qualified-id',
                                                   'qualified_execution_receipt':{'sha256':'qualified-receipt'}})
    freeze_ref=save_json(tmp_path/'freeze.json',{'prepared':prepared_ref,'code_root':'/capsules/formal-id',
                                                   'code_amendment':amendment})
    lines=' '.join(methods_lines({'formal_freeze':freeze_ref,'calibration_readout':calibration}))
    for expected in ('TRAIN8000','Q4096','512 rotating','four independent members','VALIDATION100','TEST remains locked','ConFIG','AdamW'):
        assert expected in lines
    panel=save_json(tmp_path/'panel.json',{'sample_ids':[f'validation:{i}' for i in range(100)]})
    source_spec={'panel':panel,'checkpoint':{'path':'/named/checkpoint','sha256':'checkpoint-id'},
                 'source_checkpoint':{'path':'/named/source','sha256':'source-id'}}
    spec=save_json(tmp_path/'spec.json',source_spec)
    native={'path':'/named/native','sha256':'native-id'}
    selected={'path':'/named/selected','sha256':'selected-id'}
    assessment={'spec':spec,'child_native':native,'child_selected':selected,
        'source_native':native,'source_selected':selected,'source_baseline':{'sha256':'baseline-id'}}
    entry={'arm':ARMS[0],'epoch':100,'split':'validation','assessment':{'path':'/raw/assessment','sha256':'assessment-id'},
           'scale_summary':{'path':'/raw/scale','sha256':'scale-id'}}
    inputs={'formal_freeze':freeze_ref,'calibration_readout':calibration,'snapshots':[entry],
        'calibration':prepared['calibration'],'continuation_decision':{'sha256':'continuation-id'},
        'terminal_decisions':{'sha256':'terminal-id'},'calibration_uncertainty':{},
        'preparation_readout':{},'action_audit':{},'action_readout':{},'engineering':{},'training':[],
        'matched100_readout':{'sha256':'matched-readout-id'}}
    mapping=named_evidence(inputs,{(ARMS[0],100,'validation'):(assessment,{})},campaign,'logical/path',tmp_path)
    assert mapping['formal_code_capsule_id']=='formal-id'
    assert mapping['qualified_code_capsule_id'] and mapping['source_materialized_view_sha256']
    assert mapping['source_checkpoint']==source_spec['source_checkpoint']
    assert mapping['snapshots'][0]['panel_sample_count']==100
    assert mapping['snapshots'][0]['checkpoint']['sha256']=='checkpoint-id'
    assert mapping['raw_data2_campaign_root']!=mapping['portable_publication_root']
    assert mapping['matched100_readout']['sha256']=='matched-readout-id'


def test_direct_matched100_tables_bind_panels_and_checkpoints(tmp_path):
    import copy
    effect={'mean':-.2,'interval_95':[-.3,-.1]}
    scale={'definitions':{'Complete_Expanded_Prefit':effect},'raw_joint':{'score':effect},
           'weighted_blocks':{key:{term:effect for term in ('attraction','separation','score')}
                              for key in ('signed','joint','log_power_self','rho','third_order')}}
    direct={'paired_core':{'state':effect,'shape':effect},'scale':scale}
    readout={'schema':'car_round7_review100_readout_v1','status':'completed','test_locked':True,
             'training':{arm:{'evidence':{'checkpoint':{'sha256':arm+'-checkpoint'}}} for arm in ARMS},
             'assessment':{}}
    snapshots=[];loaded={}
    for split in ('train','validation'):
        panel=save_json(tmp_path/f'{split}-panel.json',{'sample_ids':[split+':0',split+':1']})
        evidence={}
        for label,arm in (('control',ARMS[0]),('joint',ARMS[1])):
            checkpoint={'sha256':arm+'-checkpoint'}
            spec=save_json(tmp_path/f'{split}-{label}-spec.json',{'checkpoint':checkpoint,'panel':panel})
            assessment_ref={'sha256':split+'-'+label+'-assessment'}
            scale_ref={'sha256':split+'-'+label+'-scale'}
            entry={'arm':arm,'epoch':100,'split':split,'checkpoint_sha256':checkpoint['sha256'],
                   'assessment':assessment_ref,'scale_summary':scale_ref}
            snapshots.append(entry)
            loaded[(arm,100,split)]=({'spec':spec,'sample_ids':[split+':0',split+':1']},{})
            evidence[label]={'arm':arm,'epoch':100,'split':split,'checkpoint_sha256':checkpoint['sha256'],
                             'assessment':assessment_ref,'scale_summary':scale_ref}
        readout['assessment'][split]={'sample_count':2,'evidence':evidence,'joint_minus_control':direct}
    inputs={'snapshots':snapshots,'matched100_readout':save_json(tmp_path/'readout.json',readout)}
    assert validate_matched100(inputs,loaded)['status']=='completed'
    rendered='\n'.join(matched100_tables(readout))
    assert 'joint-anchor minus composite-control' in rendered
    assert '| train |' in rendered and '| validation |' in rendered
    assert '-0.2 [-0.3, -0.1]' in rendered
    assert all('| '+name+' |' in rendered for name in ('signed','joint','log_power_self','rho','third_order'))
    broken=copy.deepcopy(readout)
    broken['assessment']['train']['evidence']['joint']['checkpoint_sha256']='different'
    inputs['matched100_readout']=save_json(tmp_path/'bad-readout.json',broken)
    with pytest.raises(ValueError,match='checkpoint or panel identity differs'):
        validate_matched100(inputs,loaded)
    inputs['matched100_readout']=save_json(tmp_path/'good-readout.json',readout)
    changed_panel=save_json(tmp_path/'changed-panel.json',{'sample_ids':['train:0','train:1'],'other':'different panel'})
    changed_spec=save_json(tmp_path/'changed-spec.json',{'checkpoint':{'sha256':ARMS[1]+'-checkpoint'},'panel':changed_panel})
    different=copy.deepcopy(loaded)
    different[(ARMS[1],100,'train')][0]['spec']=changed_spec
    with pytest.raises(ValueError,match='same exact panel'):
        validate_matched100(inputs,different)


def test_continuation_context_distinguishes_uncertain_matched100_from_later_age():
    effect={'mean':-.04,'interval_95':[-.11,.03]}
    readout={'assessment':{'validation':{'joint_minus_control':{'scale':{
        'definitions':{'Complete_Expanded_Prefit':effect},'raw_joint':{'score':effect}}}}}}
    diagnosis={'arms':{ARMS[1]:{'windows':{
        '51-75':{'epoch_means':{'constraint_gradient_norm':2.9}},
        '76-100':{'epoch_means':{'constraint_gradient_norm':4.6},'field_violation':{'CO':.08,'T':.02}}}}}}
    decision={'comparison_claim_boundary':'Joint500 versus composite100 is not an isolated treatment effect.'}
    text=' '.join(continuation_context_lines(decision,readout,diagnosis))
    assert 'intervals include zero' in text and '2 of 2' in text
    assert '2.9' in text and '4.6' in text
    assert 'not an isolated treatment effect' in text


def test_terminal_history_requires_full_continued_lineage():
    first100=[{'epoch':i,'epoch_complete':True} for i in range(1,101)]
    validate_terminal_history(first100,ARMS[0],100)
    with pytest.raises(ValueError,match='entire declared terminal epoch 500'):
        validate_terminal_history(first100,ARMS[1],500)
    all500=[{'epoch':i,'epoch_complete':True} for i in range(1,501)]
    validate_terminal_history(all500,ARMS[1],500)


def test_measured_figures_use_supplied_values_without_model_evaluation(tmp_path):
    audit={'comparisons':[{'checkpoint':'SOURCE','metric':'Gradient norm','unit':'model units',
                           'composite':2.0,'joint':1.0}]}
    assert 'local actions' in action_figure(audit,tmp_path)
    engineering={'complete_epoch_measurements':[
        {'method':'incumbent','seconds':3.,'peak_allocated_gib':4.,'peak_reserved_gib':5.,'device_name':'same device'},
        {'method':'candidate','seconds':2.,'peak_allocated_gib':4.,'peak_reserved_gib':5.,'device_name':'same device'}],
        'measured_breakdown':[{'scope':'forward','seconds':1.}]}
    assert 'overlap' in engineering_figure(engineering,tmp_path)
    for name in ('same_state_action_audit','executor_time_memory_breakdown'):
        assert (tmp_path/f'{name}.pdf').is_file() and (tmp_path/f'{name}.png').is_file()
    engineering['complete_epoch_measurements'][1]['device_name']='other device'
    with pytest.raises(ValueError,match='same measured device'):
        engineering_figure(engineering,tmp_path)


def test_localized_and_shape_figures_reconcile_saved_arrays(tmp_path):
    import numpy as np
    fields=['CH4','CO','T','U_1','p']; bands=[1.5,.45,.15]
    config=tmp_path/'config.json'; config.write_text(__import__('json').dumps({'coherence':{'families':{'multiscale_spectrum':{'bands':{'wavelengths':bands}}}}}))
    import hashlib
    config_ref={'path':str(config),'sha256':hashlib.sha256(config.read_bytes()).hexdigest()}
    prefit=save_json(tmp_path/'prefit.json',{'coefficients':{'scale.log_power_self':.5,'scale.third_order':.5}})
    bank=save_json(tmp_path/'bank.json',{'field_order':fields,'triples':[[0,[[0,0],[1,1],[2,2]]]]*8,
                                         'target_weights':{'scale.third_order':[.125]*8}})
    spec=save_json(tmp_path/'spec.json',{'core_config':config_ref,'prefit':prefit,'expanded_bank':bank})
    def item(array):return {'mean':np.asarray(array).tolist()}
    def block(array):
        x=np.asarray(array,dtype=float)
        return {'attraction':item(x),'separation':item(x*0),'score':item(x)}
    joint=np.zeros((3,5));joint[0,1]=.2;piece=np.zeros((1,5));piece[0,1]=.3
    log=np.zeros((3,5));log[0,0]=.6;triple=np.full(8,.4)
    loc={'joint_spectral_band_field_attribution':block(joint),'joint_piece_field_attribution':block(piece),
         'signed_self_band_field':block(np.zeros((3,5))),'signed_pair_band':block(np.zeros((3,10))),
         'rho_pair_band':block(np.zeros((3,10))),'log_power_band_field_attribution':block(log),
         'third_order_leaf':block(triple)}
    summary={'localization_raw':loc,'raw_joint':block(.5),'weighted_blocks':{'log_power_self':block(.3),'third_order':block(.2)}}
    def confidence(value):return {'mean':value,'interval_95':[value-.1,value+.1]}
    comps={}
    for kind in ('self','mutual'):
        comps[f'shape.{kind}']=confidence(.3)
        for category in ('finite','essential'):
            comps[f'shape.{kind}.{category}.attraction']=confidence(.25)
            comps[f'shape.{kind}.{category}.separation']=confidence(.10)
    assessment={'spec':spec,'field_order':fields,'paired_shape_components':comps,'paired_core':{'shape':confidence(.6)}}
    snapshot={'arm':ARMS[0],'epoch':100,'split':'train'};loaded={(ARMS[0],100,'train'):(assessment,summary)}
    captions,co=localization_figures([snapshot],loaded,tmp_path)
    assert len(captions)==2 and co[0][3:]==pytest.approx((.5,0,.5))
    assert len(shape_component_figures([snapshot],loaded,tmp_path))==1
    summary['raw_joint']['score']['mean']=.7
    with pytest.raises(ValueError,match='does not sum'):
        localization_figures([snapshot],loaded,tmp_path)


def test_visual_policy_keeps_fixed_cases_when_worst_overlaps(tmp_path):
    import torch
    scale=torch.tensor([2.,3.]); row=tmp_path/'0000.pt';torch.save({'normalizer_scale':scale},row)
    import hashlib
    row_hash=hashlib.sha256(row.read_bytes()).hexdigest();ids=['fixed0','fixed1','third']
    panel=save_json(tmp_path/'panel.json',{'sample_ids':ids})
    source=save_json(tmp_path/'source.json',{'files_sha256':[row_hash],'sample_ids':ids})
    reference=save_json(tmp_path/'reference.json',{'reference':{'field_names':['CO','T'],'absolute_allowance':[.1,.2]}})
    child=save_json(tmp_path/'child.json',{'sample_ids':ids,'checkpoint_sha256':'checkpoint','start_unix':2.})
    decision=save_json(tmp_path/'decision.json',{'status':'sealed'})
    tasks=save_json(tmp_path/'tasks.json',{'status':'completed','child_native':child,'decision':decision})
    spec=save_json(tmp_path/'spec.json',{'panel':panel,'training_law_reference':reference})
    policy=save_json(tmp_path/'policy.json',{'schema':'car_round7_visual_example_policy_v1','status':'frozen_before_formal_learning',
        'split':'validation','test_locked':True,'panel':panel,'source_native':source,'training_reference':reference,
        'normalizer_source_row':{'path':str(row),'sha256':row_hash},'field_order':['CO','T'],'field_units':['unknown','unknown'],
        'physical_mask':'native mask','created_unix':1.,'fixed_cases':[{'ordinal':0,'sample_id':ids[0]},{'ordinal':1,'sample_id':ids[1]}],
        'representative_case_rule':{'field_denominators_physical':[4.,18.]}})
    assessment={'spec':spec,'source_native':source,'child_native':child,'field_order':['CO','T'],
        'field_units':['unknown','unknown'],'physical_mask':'native mask','sample_ids':ids,
        'per_condition_fields':{'candidate':[{'ensemble_mean_mse':[8.,36.]},{'ensemble_mean_mse':[8.,36.]},{'ensemble_mean_mse':[4.,18.]}]}}
    selection=resolve_visual_cases(policy,assessment,tasks)
    assert [x['ordinal'] for x in selection['cases']] == [0,1,0]
    assert selection['cases'][2]['overlap_fixed_cases'] == [1]
    assert selection['checkpoint_sha256']=='checkpoint'


def test_saved_calibration_and_task_intervals_are_reported_without_refitting():
    calibration={'fields':[{'field':'CO','risk0':.04,'member_mse0':.06}]}
    uncertainty={'schema':'car_round7_source_training_law_uncertainty_v1','sample_count':800,'replicates':2000,
        'reference':{'sha256':'source'},'field_order':['CO'],
        'source_signed_risk':{'mean':[.04],'interval_95':[[.03],[.05]]},
        'source_member_mse':{'mean':[.06],'interval_95':[[.05],[.07]]}}
    assert calibration_intervals(calibration,uncertainty,'source')==[((.03,.05),(.05,.07))]
    task={'paired':{'event':{'mean':-.02,'interval_95':[-.04,.01]}}}
    assert task_interval_text(task,'event')=='-0.02 [-0.04, +0.01]'
    uncertainty['source_member_mse']['mean'][0]=.07
    with pytest.raises(ValueError,match='disagrees'):
        calibration_intervals(calibration,uncertainty,'source')


def test_native_tables_show_both_splits_intervals_and_historical_allowance():
    def interval(means,lower,upper): return {'mean':means,'interval_95':[lower,upper]}
    values={'signed_risk_difference_over_allowance':interval([-.4,.2],[-.6,-.1],[-.2,.5]),
        'member_mse_ratio':interval([1.05,.95],[1.,.9],[1.1,1.]),
        'ensemble_mean_mse':interval([-.2,.1],[-.3,0.],[0.,.2]),
        'unbiased_member_variance':interval([.3,-.1],[.1,-.2],[.5,0.]),
        'sensor_mse':interval([0.,-.2],[0.,-.3],[0.,-.1])}
    assessment={'field_order':['CO','T'],'field_units':['unknown','K'],
        'physical_mask':'positive-mass native support','paired_fields':values,
        'per_condition_fields':{'SOURCE':[{'sensor_counts':[0,2]}]}}
    snapshots=[{'arm':ARMS[0],'epoch':100,'split':'train'},
               {'arm':ARMS[0],'epoch':100,'split':'validation'}]
    loaded={(item['arm'],item['epoch'],item['split']):(assessment,{}) for item in snapshots}
    calibration={'fields':[{'field':'CO','allowance':.2,'prior_allowance':.1},
                           {'field':'T','allowance':.1,'prior_allowance':.2}]}
    text='\n'.join(native_field_tables(snapshots,loaded,calibration))
    assert 'epoch 100 · train' in text and 'epoch 100 · validation' in text
    assert '-0.8 [-1.2, -0.4]' in text
    assert '+0.1 [-0.05, +0.25]' in text
    assert 'unavailable (no sensors)' in text
    assert 'Member variance describes physical ensemble spread' in text


def test_prior_member_mse_comes_from_saved_reference_or_is_unrecorded():
    prior={'risk_reference_panel':{'field_order':['CH4','CO'],
                                   'source_member_mse':[.010150245189142879,.06753054986940697]}}
    row={'field':'CO','risk0':.1,'prior_risk0':.2,'member_mse0':.3,
         'risk_scale':.3,'allowance':.03,'prior_allowance':.04}
    result='\n'.join(calibration_tables({'fields':[row]},[((.05,.15),(.2,.4))],prior))
    assert '0.0675305' in result
    prior['risk_reference_panel'].pop('source_member_mse')
    assert 'unrecorded' in '\n'.join(calibration_tables({'fields':[row]},[((.05,.15),(.2,.4))],prior))
