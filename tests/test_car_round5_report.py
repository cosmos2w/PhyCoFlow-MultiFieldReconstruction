import json
from pathlib import Path

import pytest

from scripts.analysis import report_car_round5 as report


def ref(path, data):
    path.write_text(json.dumps(data))
    return {'path':str(path), 'sha256':report.sha(path)}


def test_engineering_closure_renders_and_missing_evidence_fails(tmp_path):
    campaign=tmp_path/'campaign'; campaign.mkdir()
    decision=ref(tmp_path/'decision.json',{'scope':'engineering','recommendation':'retain SOURCE',
        'reason':'qualification failed','model_assessment_performed':False})
    costs=ref(tmp_path/'costs.json',{'schema':'car_round5_costs_v1','training':{},
        'assessment':{'gpu_seconds':None,'cpu_seconds':None,'wall_seconds':None,'queue_tail_seconds':None},
        'campaign_gpu_seconds':12})
    engineering=ref(tmp_path/'engineering.json',{'summary':'The tested computation did not qualify.'})
    manifest=tmp_path/'inputs.json'
    manifest.write_text(json.dumps({'schema':'car_round5_report_inputs_v1','decision':decision,'costs':costs,'engineering':engineering}))
    out=campaign/'reports'/'engineering'
    receipt=report.produce(manifest,campaign,out)
    assert receipt['status']=='rendered_pending_visual_inspection'
    assert (out/'report.pdf').exists() and list((out/'page_previews').glob('page-*.png'))
    text=(out/'report.md').read_text()
    assert 'No learned reconstruction is adopted' in text and 'unavailable' in text
    (tmp_path/'engineering.json').write_text('{}')
    with pytest.raises(ValueError,match='evidence changed'):
        report.produce(manifest,campaign,campaign/'reports'/'bad')


def test_native_only_learning_has_no_invented_sss(tmp_path):
    rows={report.ARMS[0]:[{'epoch':i,'native_data_loss':1/i,'optimizer_lr':1e-5} for i in range(1,13)]}
    report.learning(rows,tmp_path)
    assert (tmp_path/'learning.pdf').exists() and (tmp_path/'learning.png').exists()


def test_figure_preview_master_and_pdf_render(tmp_path):
    figures=tmp_path/'figures';figures.mkdir()
    rows={report.ARMS[0]:[{'epoch':i,'native_data_loss':1/i,'optimizer_lr':1e-5} for i in range(1,13)]}
    report.learning(rows,figures)
    (tmp_path/'report.md').write_text('# Coherence Adaptive Repair: Round 5\n\n## Figure index\n\n- [Learning](#figure-learning)\n\n## Figure learning\n\n![Training history](figures/learning.png)\n\nTraining history. [PDF master](figures/learning.pdf)\n')
    assert report.render(tmp_path,{'learning':'Training history.'}) >= 1
    import subprocess
    text=subprocess.run(['pdftotext','-layout',str(tmp_path/'report.pdf'),'-'],check=True,capture_output=True,text=True).stdout
    pages=text.split('\f')
    assert 'Coherence Adaptive Repair: Round 5' in pages[0]
    assert any('Figure learning' in page and 'Training history.' in page for page in pages)


def test_recorded_action_summary_counts_and_rejects_incomplete():
    row={'batches':12,'accepted_actions':9,'action_alpha':7.5/12,
         'field_source_ensemble_checks':1,'field_post_ensemble_checks':13/12,
         'finite_counts':{'action_alpha':12,'field_source_ensemble_checks':12,'field_post_ensemble_checks':12}}
    result=' '.join(report.action_summary({report.ARMS[1]:[row]}))
    assert '9 accepted' in result and '3 half-displacement' in result and '3 rejected' in result
    assert '12 SOURCE' in result and '13 post-proposal' in result
    row['finite_counts']['action_alpha']=11
    with pytest.raises(ValueError,match='per-attempt'):
        report.action_summary({report.ARMS[1]:[row]})


def test_historical_confg_overlay_uses_only_common_columns(tmp_path):
    historical=[]
    for epoch in range(1,13):
        row={'epoch':epoch,'optimizer_lr':1e-5}
        row.update({f'coherence_family/{family}/raw':1. for family in report.FAMILY.values()})
        historical.append(row)
    path=tmp_path/'history.jsonl';path.write_text(''.join(json.dumps(row)+'\n' for row in historical))
    reference={'path':str(path),'sha256':report.sha(path)}
    loaded=report.historical_history(reference,12)
    current={report.ARMS[0]:[{'epoch':i,'native_data_loss':1/i,'optimizer_lr':1e-5} for i in range(1,13)]}
    report.learning(current,tmp_path,loaded)
    assert (tmp_path/'learning.pdf').exists()
    with pytest.raises(ValueError,match='covering'):
        report.historical_history(reference,13)


def test_distinct_terminal_ages_render_and_bind_each_bank_and_history(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    campaign=tmp_path/'campaign';campaign.mkdir()
    arms=report.ARMS;ids=[f'condition-{i}' for i in range(100)]
    checkpoint='source-checkpoint';panel='panel-hash'
    source_path=tmp_path/'source.json'
    source_ref=ref(source_path,{'sample_ids':ids,'field_order':['CO'],'members':4,'panel_sha256':panel,'source_sha256':checkpoint})
    bank_refs={}
    for arm,epoch in zip(arms,(2,3),strict=True):
        bank_refs[arm]=ref(tmp_path/(str(epoch)+'bank.json'),{'status':'completed','epoch':epoch,'arm':arm,
            'sample_ids':ids,'members':4,'source_bank_sha256':source_ref['sha256'],
            'source_checkpoint_sha256':checkpoint,'panel_sha256':panel})
    histories={}
    family_keys=[f'coherence_family/{family}/raw' for family in report.FAMILY.values()]
    for arm,epoch in zip(arms,(2,3),strict=True):
        rows=[]
        for i in range(1,epoch+1):
            row={'epoch':i,'native_data_loss':1.,'optimizer_lr':.001,'epoch_complete':True,
                 'attempted_condition_visits':64,'accepted_objective_condition_visits':64}
            if arm==arms[1]:
                row.update({key:1. for key in family_keys})
                row.update(batches=1,accepted_actions=1,action_alpha=1.,action_dot=-1.,
                    update_accepted_fraction=1.,field_source_ensemble_checks=1,field_post_ensemble_checks=1,
                    finite_counts={'action_alpha':1,'field_source_ensemble_checks':1,'field_post_ensemble_checks':1})
            rows.append(row)
        path=tmp_path/arm/'logs/history.jsonl';path.parent.mkdir(parents=True);(path.parent.parent/'run_manifest.json').write_text(json.dumps({'effective_child_ema_enabled':False}));path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        histories[arm]={'path':str(path),'sha256':report.sha(path)}
    historical=tmp_path/'historical.jsonl';historical.write_text(''.join(json.dumps(dict(epoch=i,optimizer_lr=.001,**{k:1. for k in family_keys}))+'\n' for i in range(1,4)))
    risk={'member_mse':[1.],'ensemble_mean_mse':[1.],'unbiased_member_variance':[1.]}
    source={'risk':risk,'sensor_mse':[1.]}
    def candidate():
        return {'replacement_eligible':False,'risk':risk,'sensor_mse':[1.],
            'fidelity':{'field_difference':[0.],'field_allowance':[.1],'primary_pass':True,'legacy_member_pass':True},
            'normalized_core_differences':dict(state=0.,scale=0.,shape=0.),
            'reused_tasks':dict(regional_normalized_improvement=0.,event_normalized_difference=0.)}
    assessment={'split':'validation','sample_ids':ids,'population':1000,'members':4,'epoch_by_arm':{arms[0]:2,arms[1]:3},
        'field_order':['CO'],'task_scope':'reused development tasks at endpoint','source':source,
        'candidates':{arm:candidate() for arm in arms},'paired_condition_differences':{},
        'source_bank_sha256':source_ref['sha256'],'source_checkpoint_sha256':checkpoint,'panel_sha256':panel,
        'bank_receipt_sha256':{a:bank_refs[a]['sha256'] for a in arms}}
    assessment_ref=ref(tmp_path/'assessment.json',assessment)
    decision=ref(tmp_path/'decision.json',{'scope':'model','recommendation':'retain SOURCE/C8','reason':'both paths fail',
        'assessment_sha256':assessment_ref['sha256']})
    cost_entry={'seconds_per_epoch':1.,'occupied_gpu_seconds':2.,'update_work_seconds':1.,
                'data_wait_and_loop_overhead_seconds':.5,'callback_and_reporting_seconds':.5}
    cost=ref(tmp_path/'cost.json',{'schema':'car_round5_costs_v1','training':{a:cost_entry for a in arms},
        'assessment':{'gpu_seconds':1.,'cpu_seconds':1.,'wall_seconds':2.,'queue_tail_seconds':1.},'campaign_gpu_seconds':10.})
    engineering=ref(tmp_path/'engineering.json',{'summary':'The optimizer was qualified.'})
    selection=ref(tmp_path/'selection.json',{});task=ref(tmp_path/'task.json',{})
    import yaml
    units=ref(tmp_path/'units.json',{'status':'completed','sample_count':800,'units':dict(native=1,state=1,scale=1,shape=1)})
    configs={}
    for arm in arms:
        config=tmp_path/(arm+'.yaml')
        config.write_text(yaml.safe_dump({'source':{'weight_view':'ema_materialized'},'optimization':{
            'batch_size':64,'steps_per_epoch':12,'direct_supervision':{'units':dict(native=1,state=1,scale=1,shape=1),
            'mode':'native' if arm==arms[0] else 'safeguarded'}}}))
        configs[arm]={'path':str(config),'sha256':report.sha(config)}
    freeze=ref(tmp_path/'freeze.json',{'schema':'car_round5_freeze_v1','units':units,'arms':configs})
    manifest=ref(tmp_path/'inputs.json',{'schema':'car_round5_report_inputs_v1','decision':decision,'costs':cost,
        'engineering':engineering,'assessment':assessment_ref,'formal_freeze':freeze,'source_bank':source_ref,'candidate_banks':bank_refs,
        'histories':histories,'historical_history':{'path':str(historical),'sha256':report.sha(historical)},
        'visual_selection':selection,'regional_task':task})
    def fake_figure(figdir,name):
        fig,ax=plt.subplots();ax.plot([0,1],[0,1]);report.save_figure(fig,figdir,name)
    def fake_spatial(assessment,selection,source_path,source,banks,figdir):
        fake_figure(figdir,'spatial_case_1');return [('spatial_case_1','Fixture spatial preview.')]
    def fake_regional(assessment,source_path,source,banks,task,figdir):fake_figure(figdir,'regional_relationship')
    monkeypatch.setattr(report,'spatial',fake_spatial);monkeypatch.setattr(report,'regional',fake_regional)
    out=campaign/'reports'/'different_ages';result=report.produce(Path(manifest['path']),campaign,out)
    text=(out/'report.md').read_text()
    assert result['pages']>0 and 'different learning ages' in text
    assert 'original native bridge-velocity objective' in text and '768 attempted visits' in text
    assert '100 of 1000 validation conditions (10.0%)' in text
    assert 'previously accessed same-trajectory' in text and 'J4 members' in text
    assert 'do not cover temporal dependence or training-seed uncertainty' in text
    assert 'Round 3 and Round 4 artifacts remain unchanged' in text
    with pytest.raises(ValueError,match='positive declared population'):
        report.panel_coverage(dict(assessment,population=0))
    assert 'Native continuation | 2 | 128 | 128' in text
    assert 'Native + SSS safeguard | 3 | 192 | 192' in text
    assert 'Learning ages differ.' in text and 'figures/spatial_case_1.png' in text
    bad=dict(assessment,epoch_by_arm={arms[0]:3,arms[1]:3})
    with pytest.raises(ValueError,match='history must end'):
        report.histories(histories,bad)
    bad=dict(assessment,epoch_by_arm={arms[0]:2,arms[1]:4})
    with pytest.raises(ValueError,match='candidate bank mismatch'):
        report.bank_context({'source_bank':source_ref,'candidate_banks':bank_refs},bad)


def test_query_engineering_uses_completed_bound_receipts(tmp_path):
    base=Path(__file__).resolve().parents[1]/'cases/turbulent_combustion/runs/coherence_adaptive_repair/round_5/assessment/engineering'
    gradient=base/'Query_Estimator_Gradient_Review.json';actions=base/'Query_Finite_Action_Comparison.json'
    if not gradient.exists() or not actions.exists():pytest.skip('completed query engineering receipts unavailable')
    inputs=tmp_path/'inputs.json'
    inputs.write_text(json.dumps({'query_gradient_review':{'path':str(gradient),'sha256':report.sha(gradient)},
                                  'query_actions':{'path':str(actions),'sha256':report.sha(actions)}}))
    campaign=tmp_path/'campaign';output=campaign/'assessment/engineering/figures'
    result=report.render_query_only(inputs,campaign.mkdir() or campaign,output)
    assert result['figure_png_sha256']==report.sha(output/'query_engineering.png')
    assert result['figure_pdf_sha256']==report.sha(output/'query_engineering.pdf')
    assert 'lower fixed-query median' in result['caption']
    assert 'unmeasured' not in result['caption'] or 'cost' in result['caption']
    with pytest.raises(FileExistsError):report.render_query_only(inputs,campaign,output)


def test_timing_distinguishes_wait_and_recorded_training_cpu():
    arm=report.ARMS[0]
    costs={'schema':'car_round5_costs_v1','training':{arm:{'seconds_per_epoch':10,
        'occupied_gpu_seconds':1000,'update_work_seconds':700,'data_wait_and_loop_overhead_seconds':200,
        'callback_and_reporting_seconds':100,'complete_epoch_wall_seconds':1000,
        'inside_lease_child_cpu_seconds':123}},'campaign_gpu_seconds':1100,
        'assessment':{'gpu_seconds':40,'cpu_seconds':50,'wall_seconds':202,'queue_tail_seconds':162}}
    timing={'schema':'car_round5_assessment_timing_note_v1','assessment_window_elapsed_seconds':202,
        'sequential_gpu_lease_and_cpu_job_wall_sum_seconds':16,
        'between_review_cpu_end_and_terminal_cpu_start_seconds':150,'assessment_cpu_queue_tail_seconds':162}
    lines=' '.join(report.costs_table(costs,[arm],timing))
    assert '202.00 seconds (20.2%' in lines and '16.00 seconds (1.6%' in lines
    assert '150.00 seconds between the review and terminal CPU jobs' in lines
    assert '123 seconds' in lines and 'not CPU computation' in lines
    with pytest.raises(ValueError,match='disagrees'):
        report.costs_table(costs,[arm],dict(timing,assessment_window_elapsed_seconds=201))
