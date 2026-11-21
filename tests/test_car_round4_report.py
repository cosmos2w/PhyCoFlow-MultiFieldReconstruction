"""Terminal reporting uses bound evidence and never starts model evaluation."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('round4_report', ROOT/'scripts/analysis/report_car_round4.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_teacher_rejection_render_and_evidence_binding(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'DATA_ROOT', tmp_path)
    campaign=tmp_path/'case'/'round_4'; campaign.mkdir(parents=True)
    docs=campaign/'evidence';docs.mkdir()
    def save(name, data):
        path=docs/name;path.write_text(json.dumps(data));return {'path':str(path),'sha256':report.sha(path)}
    manifest=save('manifest.json',{'schema':'car_round4_training_manifest_v1','field_order':['T','p'],'population':8000,'sensor_field':'T','sensor_count':256})
    rows=[]
    for epoch in range(1,101):
        batches=[{'condition_indices':[i], 'batch_size':1, 'accepted_alpha':[.5]} for i in range(12)]
        rows.append({'epoch':epoch,'visits':12,'cumulative_visits':epoch*12,'dataset_pass_equivalents':epoch*12/8000,
                     'batch_stream':batches,'family_v':dict.fromkeys(report.FAMILIES,.9),'mean_alpha':.5,
                     'field_displacement':[.01,.02],'field_member_spread':[.02,.03],'min_field_slack':[.001,.002],
                     'elapsed_seconds':1.,'cpu_seconds':.5})
    history=docs/'history.jsonl';history.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    checkpoint=docs/'teacher.pt';torch.save({'schema':'car_round4_teacher_checkpoint_v1','completed_epochs':100,
                                           'train_manifest_sha256':manifest['sha256'],'history':rows,'normalizers':dict.fromkeys(report.FAMILIES,1.),'panel_ids':[f'id{i}' for i in range(800)], 'epsilon_model_squared':[1.,1.], 'epsilon_physical_squared':[4.,9.]},checkpoint)
    model_fields={'finite_law_mean_mse':[.5,3.], 'member_mse':[1.,4.], 'member_spread':[.5,1.], 'source_displacement':[0.,0.], 'observed_entry_mse':[1.,None]}
    physical_fields={key:[None if value is None else value*factor for value,factor in zip(values,[4.,9.])] for key,values in model_fields.items()}
    field_diagnostics={'schema':'car_round4_teacher_train800_fields_v1','split':'TRAIN','sample_count':800,'members':4,'field_order':['T','p'],'physical_field_units':['unknown','unknown'],'model_field_units_squared':model_fields,'decoded_field_units_squared':physical_fields,'observation_entry_counts':[819200,0]}
    gate=save('gate.json',{'schema':'car_round4_teacher_qualification_v1','completed_epochs':100,
              'train_manifest_sha256':manifest['sha256'],'teacher_checkpoint_sha256':report.sha(checkpoint),
              'admitted':False,'normalized_improvement':dict.fromkeys(report.FAMILIES,0.),'source_scores':dict.fromkeys(report.FAMILIES,1.),'teacher_scores':dict.fromkeys(report.FAMILIES,1.),'panel_ids':[f'id{i}' for i in range(800)],'feasibility_population':8000,'replay_seed':819103,'violations':0,'zero_correction_fraction':.1,'train800_field_diagnostics':{'source':field_diagnostics,'teacher':field_diagnostics}})
    invalid_diagnostics=json.loads(json.dumps(field_diagnostics))
    invalid_diagnostics['decoded_field_units_squared']['member_mse'][0]=999.
    with pytest.raises(ValueError,match='unit conversion'):
        report.validate_teacher_field_diagnostics({'train800_field_diagnostics':{'source':field_diagnostics,'teacher':invalid_diagnostics}}, {'epsilon_model_squared':[1.,1.], 'epsilon_physical_squared':[4.,9.]}, {'field_order':['T','p'],'sensor_field':'T','sensor_count':256})
    forecast=save('forecast.json',{'schema':'car_round4_feasibility_gate_v1','manifest_sha256':manifest['sha256'],
                   'candidate_inclusive_gpu_seconds':20000.,'teacher_cpu_worker_seconds':5000.,
                   'forecast_gpu_seconds':{'full_train_target_preparation_remaining':1000.,
                      'teacher_including_calibration_replay_audit_and_persistence':9000.,
                      'native_amortization_500_epochs':8000.,'assessment':1000.}})
    cost=save('cost.json',{'schema':'car_round4_teacher_closeout_cost_v1','complete':True,'cost_basis':'measured',
              'train_manifest_sha256':manifest['sha256'],'gpu_seconds':{'source_generation':1000.,'teacher':9000.},'candidate_overhead_gpu_seconds':0.,'teacher_cpu_worker_seconds':4000.,'elapsed_wall_seconds':9000.,'elapsed_wall_basis':'known-clock span excludes validation with unknown absolute timestamps','auxiliary_cpu_work':[{'scope':'CPU-only SOURCE copy','process_cpu_seconds':12.}], 'auxiliary_cpu_unknown':[{'scope':'prior validation','timer':'process CPU'}]})
    inputs={'schema':'car_round4_report_inputs_v1','mode':'teacher_rejected','train_manifest':manifest,'teacher_gate':gate,
            'teacher_checkpoint':{'path':str(checkpoint),'sha256':report.sha(checkpoint)},
            'teacher_history':{'path':str(history),'sha256':report.sha(history)},'forecast':forecast,'costs':cost}
    manifest_path=docs/'inputs.json';manifest_path.write_text(json.dumps(inputs))
    output=campaign/'reports'/'teacher_rejection'; result=report.produce(manifest_path,campaign,output)
    assert result['status']=='terminal_evidence_assembled'
    assert len(result['figures'])==3
    markdown=(output/'report.md').read_text()
    assert 'Close this local mechanism and retain SOURCE/C8' in markdown
    assert 'finite-law' in markdown.lower() and 'TRAIN800' in markdown
    assert 'known-clock span excludes validation' in markdown and 'CPU-only SOURCE copy' in markdown and 'prior validation' in markdown
    assert not any(x in markdown for x in ('Student fair', 'regional improvement'))
    assert (output/'report.html').is_file()
    inputs['assessment']={'path':'not-read','sha256':'not-read'}
    manifest_path.write_text(json.dumps(inputs))
    with pytest.raises(ValueError,match='cannot contain student'):
        report.produce(manifest_path,campaign,campaign/'reports'/'invalid')


def test_history_rejects_missing_epoch(tmp_path):
    path=tmp_path/'history.jsonl'; path.write_text(json.dumps({'epoch':2})+'\n')
    with pytest.raises(ValueError,match='every completed epoch'):
        report.history_rows(path,1,8000)


def test_terminal_student_report_with_bound_fixed_case_atlas(tmp_path, monkeypatch):
    test_teacher_rejection_render_and_evidence_binding(tmp_path, monkeypatch)
    campaign=tmp_path/'case'/'round_4'; docs=campaign/'evidence'
    input_path=docs/'inputs.json'; inputs=json.loads(input_path.read_text()); inputs.pop('assessment');inputs['mode']='student_terminal'
    def write(path, value):
        path.write_text(json.dumps(value)); return {'path':str(path),'sha256':report.sha(path)}
    gate_path=Path(inputs['teacher_gate']['path']); gate=json.loads(gate_path.read_text());gate['admitted']=True
    gate['teacher_scores']=dict.fromkeys(report.FAMILIES,.9);gate['normalized_improvement']=dict.fromkeys(report.FAMILIES,.1)
    inputs['teacher_gate']=write(gate_path,gate)
    source_dir=docs/'source';source_dir.mkdir()
    ids=[f'trajectory_000000:{8000+i}' for i in range(100)]; source_files=[]
    coords=torch.tensor([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    truth=torch.tensor([[[1.,2.],[2.,3.],[3.,4.],[4.,5.]]])
    source_rows=[]
    for ordinal,sid in enumerate(ids):
        row={'sample_id':sid,'sample_index':ordinal,'member_seeds':[1,2,3,4],'members_model':truth.unsqueeze(0).repeat(4,1,1,1)+.1,'truth_model':truth,
             'coordinates_model':coords.unsqueeze(0),'coordinates_raw':coords,'native_masses':torch.ones(4),
             'normalizer_offset':torch.zeros(2),'normalizer_scale':torch.ones(2),
             'obs_indices':torch.tensor([[0]]),'obs_values':truth[:,0,:1].unsqueeze(-1),'obs_fields':torch.tensor([[0]])}
        path=source_dir/f'{ordinal:04d}.pt';torch.save(row,path);source_files.append(report.sha(path));source_rows.append(row)
    source_path=source_dir/'receipt.json';source_ref=write(source_path,{'status':'completed','kind':'source','split':'validation','members':4,'field_order':['T','p'],'source_sha256':'source-hash','sample_ids':ids,'files_sha256':source_files+['unused']*98})
    inputs['source_bank_receipt']=source_ref
    baseline_ref=write(docs/'baseline.json',{'sample_ids':ids,'reserved_task':{'source_per_condition':[.3]*100}})
    selection={'baseline':baseline_ref,'schema':'car_round4_source_defined_visual_cases_v1','source_bank':source_ref,'candidates_used':False,'test_locked':True,
               'panel':{'sha256':'panel-hash'},'cases':{'representative':{'ordinal':0,'sample_id':ids[0]},'source_tail':{'ordinal':1,'sample_id':ids[1]}}}
    inputs['visual_selection']=write(docs/'selection.json',selection)
    receipts={}
    for num,arm in enumerate(report.ARMS):
        bank=docs/f'bank{num}';bank.mkdir();file_hashes=[]
        for ordinal,row in enumerate(source_rows):
            row=dict(row);row['members_model']=row['members_model']+(.05 if num==0 else .1)
            path=bank/f'{ordinal:04d}.pt';torch.save(row,path);file_hashes.append(report.sha(path))
        receipt=bank/'receipt.json';receipts[arm]=write(receipt,{'status':'completed','epoch':100,'arm':arm,'members':4,'source_checkpoint_sha256':'source-hash','source_bank_sha256':source_ref['sha256'],'sample_ids':ids,'files_sha256':file_hashes})
    inputs['student_bank_receipts']=receipts
    candidates={}
    for arm in report.ARMS:
        candidates[arm]={'normalized_core_differences':{'state':-.06,'scale':0.,'shape':0.},
                         'per_condition_scores':[{'state':.2,'scale':.2,'shape':.2} for _ in ids],
                         'risk':{'member_mse':[.04,.09]},'fidelity':{'primary_pass':True,'legacy_member_pass':True},
                         'reused_tasks':{'regional_normalized_improvement':.08,'event_normalized_difference':.01,'regional_per_condition':[.2]*100}}
    assessment={'schema':'car_round3_matched_validation_v1','epoch':100,'task_scope':'reused development tasks at endpoint',
                'field_order':['T','p'],'source_baseline_sha256':baseline_ref['sha256'],'normalizers':{'state':1.,'scale':1.,'shape':1.},'task_normalizers':{'regional':1.},'source_checkpoint_sha256':'source-hash','split':'validation','members':4,'sample_ids':ids,'panel_sha256':'panel-hash','source_bank_sha256':source_ref['sha256'],'visual_selection_sha256':inputs['visual_selection']['sha256'],
                'bank_receipt_sha256':{arm:receipts[arm]['sha256'] for arm in report.ARMS},
                'source':{'risk':{'member_mse':[.01,.04]},'per_condition_scores':[{'state':.3,'scale':.3,'shape':.3} for _ in ids]},'candidates':candidates,
                'native_learning_histories':{arm:{'native_rf_loss':[.5]*100} for arm in report.ARMS}}
    inputs['assessment']=write(docs/'assessment.json',assessment)
    historical_path=campaign.parent/'round_3/assessment/epoch_500/costs.json';historical_path.parent.mkdir(parents=True)
    historical_ref=write(historical_path,{'training':{'FFM_SourceScaledRisk_Uniform_Repair':{'segments':[{'through_epoch':100,'occupied_gpu_seconds':7600.}]}}})
    cost={'schema':'car_round4_inclusive_cost_v1','complete':True,'cost_basis':'measured','assessment_sha256':inputs['assessment']['sha256'],
          'reference_epoch':100,'reference_receipt':historical_ref,'reference_gpu_seconds':7600.,'candidate_overhead_gpu_seconds':0.,'candidate_gpu_seconds':{'source_generation':1000.,'teacher':2000.,'student':500.,'assessment':300.}}
    inputs['costs']=write(docs/'cost.json',cost)
    from scripts.posttraining.assess_car_round4 import decision as apply_decision
    decision=apply_decision(assessment,3800.,7600.)
    decision['round4_decision'].update(assessment_sha256=inputs['assessment']['sha256'],cost_receipt_sha256=inputs['costs']['sha256'])
    inputs['decision']=write(docs/'decision.json',decision);input_path.write_text(json.dumps(inputs))
    output=campaign/'reports'/'student_terminal';result=report.produce(input_path,campaign,output)
    assert result['status']=='pending'
    assert 'bound frozen observer' in result['pending'][0]
    assert {'atlas_representative','atlas_source_tail'} <= set(result['figures'])
    text=(output/'report.md').read_text()
    assert 'regional' in text and 'Held-out fields' in text and 'RMSE' in text
    assert (output/'figures'/'atlas_source_tail.pdf').is_file()


def test_case_relationship_uses_member_mixture_and_frozen_task(tmp_path):
    import hashlib
    from phycoflow_reconstruction.coherence.measures import measure_fingerprint
    from phycoflow_reconstruction.coherence.families.topology.geometry import coordinate_digest
    from phycoflow_reconstruction.evaluation.round2_task import regional_covariance
    torch.manual_seed(9)
    coords=torch.tensor([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    truth=torch.tensor([[[0.,1.],[1.,2.],[2.,1.],[3.,-1.]]])
    draws=torch.randn(4,1,4,2)
    mass=torch.tensor([1.,2.,3.,4.]); ids=torch.arange(4,dtype=torch.long)
    task={'schema':'car_round2_reserved_task_v1','field_names':['CO','U_1'],
          'region_point_indices':ids.tolist(),
          'region_indices_sha256':hashlib.sha256(ids.numpy().astype('<i8',copy=False).tobytes()).hexdigest(),
          'native_measure':measure_fingerprint(mass,metadata={}), 'truth_scales':[1.,1.]}
    # The native-measure receipt carries the original metadata alongside its hash.
    task['native_measure']['metadata']={}
    state={'mass':mass.double(),'basis':torch.eye(4,dtype=torch.double),
           'filters':torch.tensor([[1.,1.,1.,1.]],dtype=torch.double),'power_floor':torch.ones(1,2,dtype=torch.double),
           'piece_measures':mass.double().unsqueeze(0),'piece_index':torch.zeros(4,dtype=torch.long),
           'analysis_index':torch.arange(4)}
    shape={'neighbor_indices':torch.arange(4).reshape(4,1),'neighbor_weights':torch.ones(4,1)}
    context={'task':task,'query':torch.arange(4),'baseline':{'reserved_task':{'source_per_condition':[.2]}},
             'core':{'shape_artifact':{'field_mean':[0.,0.],'field_scale':[1.,1.],'domain':((2,2),None,False)}},
             'families':{'multiscale_spectrum':{'field_names':('CO','U_1'),'units':'model_units','state_dict':state},
                         'topology':{'field_names':('CO','U_1'),'units':'model_units','state_dict':shape,
                           'geometry_sha256':coordinate_digest(coords),
                           'config':{'geometry':{'grid_shape':[2,2],'periodic':False,'raster_mapping':'physical_cell_overlap_v1'},
                                     'filtration':{'smoothing_sigma':0.}}}}}
    source={'truth_model':truth,'members_model':draws,'normalizer_scale':torch.ones(2),
            'normalizer_offset':torch.zeros(2),'native_masses':mass,'coordinates_model':coords.unsqueeze(0)}
    candidates={arm:{'members_model':draws.clone()} for arm in report.ARMS}
    assessment={'field_order':['CO','U_1'],'candidates':{arm:{'reused_tasks':{'regional_per_condition':[.2]}} for arm in report.ARMS}}
    figdir=tmp_path/'figures';figdir.mkdir()
    got=report.plot_case_relationship('representative','case:1',0,source,candidates,assessment,context,figdir)
    expected=regional_covariance(draws[:,0].double(),mass.double(),task,('CO','U_1'))
    torch.testing.assert_close(torch.tensor(got['regional_covariance_members'][1]),expected.float())
    assert all(abs(value-1.) < 1e-12 for value in got['occupation_mass_sums'])
    assert (figdir/'relationship_representative.png').is_file()


def test_partial_teacher_cost_rejection_uses_bound_history_only(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'DATA_ROOT', tmp_path)
    campaign=tmp_path/'case'/'round_4'; docs=campaign/'evidence'; docs.mkdir(parents=True)
    def save(name, data):
        path=docs/name; path.write_text(json.dumps(data)); return {'path':str(path),'sha256':report.sha(path)}
    manifest=save('manifest.json',{'schema':'car_round4_training_manifest_v1','field_order':['T'],'population':8000})
    identity={'train_manifest_sha256':manifest['sha256'], 'recording_columns':[{'name':'epoch','type':'int','units':'epoch','meaning':'complete epochs'}], 'accumulation_order':list(report.FAMILIES)}
    config=save('config.json',{'schema':'car_round4_teacher_scientific_config_v1','identity':identity,'normalizers':dict.fromkeys(report.FAMILIES,1.)})
    rows=[]
    for epoch in (1,2):
        batch=[{'condition_indices':[i], 'batch_size':1, 'accepted_alpha':[.5]} for i in range(12)]
        rows.append({'epoch':epoch,'visits':12,'cumulative_visits':12*epoch,'dataset_pass_equivalents':12*epoch/8000,
                     'batch_stream':batch,'family_v':dict.fromkeys(report.FAMILIES,.9),'mean_alpha':.5,
                     'field_displacement':[.01],'field_member_spread':[.02],'min_field_slack':[.001],
                     'elapsed_seconds':1.,'cpu_seconds':.5})
    history_path=docs/'history.jsonl';history_path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    history={'path':str(history_path),'sha256':report.sha(history_path)}
    visits=torch.zeros(8000,dtype=torch.long);visits[:12]=2
    checkpoint_path=docs/'checkpoint.pt';torch.save({'schema':'car_round4_teacher_checkpoint_v1', **identity,
        'normalizers':dict.fromkeys(report.FAMILIES,1.),'completed_epochs':2,'history':rows,
        'runtime':{'step':24},'visits':visits,'stream':{'visits':visits}},checkpoint_path)
    checkpoint={'path':str(checkpoint_path),'sha256':report.sha(checkpoint_path)}
    forecast=save('initial_forecast.json',{'schema':'car_round4_feasibility_gate_v1','manifest_sha256':manifest['sha256'],
        'candidate_inclusive_gpu_seconds':20000.,'teacher_cpu_worker_seconds':5000.,
        'forecast_gpu_seconds':{'full_train_target_preparation_remaining':1000.,'teacher_including_calibration_replay_audit_and_persistence':9000.,
                                'native_amortization_500_epochs':8000.,'assessment':1000.}})
    warm=save('warm_forecast.json',{'schema':'car_round4_teacher_warm_cache_forecast_v1','base_teacher_gpu_allowance_seconds':14400,'variation_reserve_gpu_seconds':18000,'reserve_committed_elsewhere_gpu_seconds':1000,'candidate_other_gpu_seconds':10000,'whole_other_gpu_seconds':20000,'candidate_gpu_cap_seconds':35317,'whole_gpu_cap_seconds':86400,'train_manifest_sha256':manifest['sha256'],'completed_epochs':2,'teacher_checkpoint_sha256':checkpoint['sha256'],'teacher_history_sha256':history['sha256'],'remaining_epochs':98,'training_window_epochs':[1,2],'safety_factor':1.2,'projected_teacher_gpu_seconds':26000.,'projected_teacher_cpu_worker_seconds':20000.,'teacher_gpu_cap_seconds':25317,'teacher_cpu_cap_seconds':21600})
    decision={'schema':'car_round4_teacher_budget_stop_v1','decision':'stop_cost_infeasible',
        'train_manifest_sha256':manifest['sha256'],'completed_epochs':2,'checkpoint':checkpoint,'history':history,
        'scientific_config':config,'forecast':warm,'projected_teacher_gpu_seconds':26000.,
        'projected_teacher_cpu_worker_seconds':20000.,'teacher_gpu_cap_seconds':25317,'teacher_cpu_cap_seconds':21600}
    decision_ref=save('stop.json',decision)
    cost=save('cost.json',{'schema':'car_round4_teacher_cost_rejected_closeout_v1','complete':True,
        'cost_basis':'measured','train_manifest_sha256':manifest['sha256'],'partial_epoch':2,
        'checkpoint_sha256':checkpoint['sha256'],'history_sha256':history['sha256'],
        'scientific_config_sha256':config['sha256'],'forecast_sha256':warm['sha256'],
        'projected_teacher_gpu_seconds':26000.,'projected_teacher_cpu_worker_seconds':20000.,
        'teacher_gpu_cap_seconds':25317,'teacher_cpu_cap_seconds':21600,
        'budget_stop_decision_sha256':decision_ref['sha256'],'gpu_seconds':{'source_generation':1000.,'teacher':300.},
        'candidate_overhead_gpu_seconds':50.,'teacher_cpu_worker_seconds':400.,'elapsed_wall_basis':'known clock span'})
    inputs={'schema':'car_round4_report_inputs_v1','mode':'teacher_cost_rejected','train_manifest':manifest,
        'teacher_checkpoint':checkpoint,'teacher_history':history,'teacher_scientific_config':config,
        'budget_stop_decision':decision_ref,'forecast':forecast,'costs':cost}
    input_path=docs/'inputs.json';input_path.write_text(json.dumps(inputs))
    output=campaign/'reports/cost_stop';result=report.produce(input_path,campaign,output)
    assert result['status']=='terminal_evidence_assembled' and len(result['figures'])==2
    text=(output/'report.md').read_text()
    assert 'retain SOURCE/C8' in text and 'no efficacy failure' in text and 'final TRAIN800 replay' in text
    assert 'Sampled teacher training diagnostics' in text and 'Neither is a measured speedup' in text
    assert 'Teacher empirical V' not in text and 'Held-out fields' not in text
    inputs['teacher_gate']={'path':'unused','sha256':'unused'};input_path.write_text(json.dumps(inputs))
    with pytest.raises(ValueError,match='cannot contain teacher utility'):
        report.produce(input_path,campaign,campaign/'reports/invalid_gate')
    inputs.pop('teacher_gate');decision['projected_teacher_gpu_seconds']=25000.; warm_doc=json.loads(Path(warm['path']).read_text()); warm_doc['projected_teacher_gpu_seconds']=25000.; decision['forecast']=save('warm_forecast.json',warm_doc); inputs['budget_stop_decision']=save('stop.json',decision)
    input_path.write_text(json.dumps(inputs))
    with pytest.raises(ValueError,match='neither teacher budget cap'):
        report.produce(input_path,campaign,campaign/'reports/invalid_budget')
    decision['projected_teacher_gpu_seconds']=26000.
    warm_doc['projected_teacher_gpu_seconds']=26000.; warm_doc['completed_epochs']=3
    decision['forecast']=save('warm_forecast.json',warm_doc)
    inputs['budget_stop_decision']=save('stop.json',decision);input_path.write_text(json.dumps(inputs))
    with pytest.raises(ValueError,match='invalid manifest-bound'):
        report.produce(input_path,campaign,campaign/'reports/invalid_warm_forecast')
