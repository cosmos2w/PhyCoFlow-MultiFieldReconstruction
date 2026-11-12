import json
from pathlib import Path
import numpy as np
import pytest
from phycoflow_reconstruction.evaluation import round8_report as report


def test_report_render_and_field_error_annotation(tmp_path,monkeypatch):
    folders=[]; panels=[]
    for endpoint in ('SOURCE','C8'):
        d=tmp_path/endpoint;d.mkdir();folders.append(d)
        y=np.array([[1.],[2.],[3.]])
        pred=np.stack([y if endpoint=='SOURCE' else y+1,y+1])
        np.savez(d/'part.npz',target_physical=y,draws_physical=pred,raw_coordinates=np.array([[0,0],[1,0],[0,1.]]),physical_weights=np.array([.2,.3,.5]),physical_mask=np.ones(3,dtype=bool),native_ids=np.arange(3),field_units=np.array(['K']))
        p={'population':'ASSESS','field_names':['T'],'conditions':[{'sample_id':'x','path':'part.npz'}]};panels.append(p)
        (d/'panel.json').write_text(json.dumps(p))
    monkeypatch.setattr(report,'validate_matched_panels',lambda x,y:tuple(panels))
    history=tmp_path/'train.jsonl'
    history.write_text('\n'.join(json.dumps({'epoch':e,'round8':{'scores':{'state':{'raw':2/e,'calibrated':1/e},'shape':{'raw':3/e,'calibrated':2/e}}}}) for e in (1,10,20)))
    decision=tmp_path/'decision.json'
    decision.write_text(json.dumps({'schema':'round8_campaign_decision_v1','operational':{'choice':'SOURCE'},'computation_by_population':{}}))
    result=report.build_report(tmp_path/'result',decision_path=decision,histories={'C8':history},reconstructions=[{'source_panel':folders[0]/'panel.json','endpoint_panel':folders[1]/'panel.json','sample_id':'x','field':'T','alpha':.5,'member':0,'label':'C8 reference-only'}])
    assert result['reconstructions'][0]['rmse']==pytest.approx(.5)
    assert result['verification']['png_images_decoded']==2
    assert result['verification']['pdf_pages_decoded']==2
    content=Path(result['markdown']).read_text()
    assert 'weighted RMSE' in content and 'TRAIN' in content and 'SOURCE' in content
    assert result['reconstructions'][0]['label']=='C8 reference-only' and 'C8 reference-only' in content
    assert (tmp_path/'result/REPORT.html').is_file()
    with pytest.raises(FileExistsError): report.build_report(tmp_path/'result')


def test_validator_rejects_broken_link(tmp_path):
    (tmp_path/'REPORT.md').write_text('![missing](figures/missing.png)')
    with pytest.raises(ValueError,match='unresolved'): report.validate_report(tmp_path)


def test_cached_physical_diagnostics_and_reported_exposure(tmp_path,monkeypatch):
    dirs=[]; panels=[]
    for endpoint in ('SOURCE','C8'):
        directory=tmp_path/endpoint;directory.mkdir();dirs.append(directory)
        rows=[]
        for i in range(2):
            truth=np.array([[1.,2.],[3.,4.]])
            members=np.stack((truth,truth+2))
            if endpoint=='C8': members=members+1
            np.savez(directory/f'{i}.npz',target_physical=truth,draws_physical=members,physical_weights=np.array([.2,.3]),physical_mask=np.ones(2,dtype=bool),native_ids=np.arange(2),field_units=np.array(['K','m']),inclusion_probability=.5)
            rows.append({'sample_id':str(i),'source':'validation' if i==0 else 'test','path':f'{i}.npz'})
        panel={'population':'TRAIN','field_names':['T','u'],'conditions':rows};panels.append(panel)
        (directory/'panel.json').write_text(json.dumps(panel))
    monkeypatch.setattr(report,'validate_matched_panels',lambda *_:tuple(panels))
    spec={'endpoint':'C8','source_panel':dirs[0]/'panel.json','endpoint_panel':dirs[1]/'panel.json','alpha':.5,'regime_by_id':{'0':'hot','1':'cold'}}
    result=report.cached_field_diagnostics(spec)
    f=result['pooled']['fields']['T']
    assert f['source_rms_spread']==pytest.approx(2**.5)
    assert f['blend_rms_spread']==pytest.approx(2**.5)
    assert f['rms_displacement']==pytest.approx(.5)
    assert result['groups']['regime']['hot']['n']==1
    assert result['pooled']['relationships']['T / u']['paired_absolute_error_change']['mean']==pytest.approx(3.95)
    assert result['support']=='fixed Q spatial Horvitz-Thompson proxy'
    log=tmp_path/'queries.jsonl'
    log.write_text('\n'.join(json.dumps(x) for x in [{'batch':1,'kind':'self','index':0,'count':1,'probability':.5},{'batch':1,'kind':'pair','index':0,'count':3,'probability':1.}]))
    ledger=tmp_path/'ledger.json';ledger.write_text(json.dumps({'intervals':[{'stage':'train','device_uuid':'gpu1','start':0,'end':10},{'stage':'train','device_uuid':'gpu2','start':2,'end':9},{'stage':'infer','start':0}]}))
    result=report.build_report(tmp_path/'report',field_panels=[spec],exposure_logs={'C8':{'path':log,'field_names':['T','u']}},cost_ledger=ledger)
    text=Path(result['markdown']).read_text()
    assert 'J2 RMS spread' in text and 'Horvitz-Thompson' in text and '17 GPU seconds' in text
    assert 'cold' in text and 'Field pair' in text
    assert json.loads((tmp_path/'report/cached_field_diagnostics.json').read_text())[0]['population']=='TRAIN'


def test_report_includes_fixed_source_raw_and_selected_values(tmp_path):
    from phycoflow_reconstruction.evaluation.round8_campaign import file_hash
    from phycoflow_reconstruction.evaluation.round8_decision import KEYS
    def score_file(name,alpha,values):
        p=tmp_path/name;p.write_text(json.dumps({'schema':'round8_common_observers_v1','population':'ASSESS','alpha':alpha,'rows':[{'losses':dict(zip(KEYS.values(),values))}]}));return p
    selected=score_file('train_selected.json',.5,[.9,.8,.7]);raw=score_file('assess_raw.json',1.,[.8,.7,.6])
    policy={'schema':'round8_frozen_policy_v1','endpoint':'C8','alpha':.5,'train_source_scores':dict.fromkeys(KEYS,1.),'train_raw_scores':dict.fromkeys(KEYS,.8),'score_reports':{'0.5':{'path':str(selected),'sha256':file_hash(selected)}},'train_source_risk':[1.],'risk_quadratic':{'b':[-.1],'a':[.1]}}
    pp=tmp_path/'policy.json';pp.write_text(json.dumps(policy))
    assessment={'schema':'round8_assessment_v1','endpoint':'C8','policy_sha256':file_hash(pp),'eligible':True,'qualifying_families':['state'],'alpha':.5,'gain':dict.fromkeys(KEYS,.1),'risk_ratio':[.95],'field_names':['T'],'source_risk':[1.],'blend_risk':[.95],'rmse_change':[-.025],'source_scores':dict.fromkeys(KEYS,1.),'blend_scores':dict.fromkeys(KEYS,.9),'raw_report':str(raw)}
    ap=tmp_path/'assess.json';ap.write_text(json.dumps(assessment))
    decision=tmp_path/'decision.json';decision.write_text(json.dumps({'schema':'round8_campaign_decision_v1','operational':{'choice':'C8','benchmarks':{'C8':{'latency_seconds':2,'peak_resident_bytes':100}}},'computation_by_population':{}}))
    out=report.build_report(tmp_path/'report',decision_path=decision,assessment_paths={'C8':ap},policy_paths={'C8':pp})
    text=Path(out['markdown']).read_text()
    assert 'Raw endpoint' in text and 'SOURCE-normalized gains' in text and 'TRAIN member risks' in text
    assert Path(out['pdf']).is_file()


def test_non_nominated_endpoint_still_reports_source_scores_and_raw_native_risks(tmp_path,monkeypatch):
    from phycoflow_reconstruction.evaluation.round8_decision import KEYS
    def score(name, alpha, risk):
        path=tmp_path/name
        path.write_text(json.dumps({'schema':'round8_common_observers_v1','population':'ASSESS',
            'alpha':alpha,'member_risk_physical':risk,'rows':[{'losses':dict.fromkeys(KEYS.values(),.75)}]}))
        return path
    source=score('source.json',0.,[0.,1.]);raw=score('raw.json',1.,[.2,1.21])
    assessment=tmp_path/'assessment.json'
    assessment.write_text(json.dumps({'schema':'round8_assessment_v1','endpoint':'C8','alpha':None,
        'eligible':False,'reason':'no TRAIN-nominated blend','field_names':['a','T'],
        'source_report':str(source),'raw_report':str(raw)}))
    monkeypatch.setattr(report,'validate_report',lambda output:{})
    result=report.build_report(tmp_path/'result',assessment_paths={'C8':assessment})
    text=Path(result['markdown']).read_text()
    assert '| C8 | SOURCE | state | unavailable | 0.75 |' in text
    assert '| a | 0 | 0.2 | unavailable | unavailable |' in text
    assert '| T | 1 | 1.21 | 1.21 | 10.00% |' in text


def test_amplitude_frontier_uses_saved_points_and_renders(tmp_path):
    from phycoflow_reconstruction.evaluation.round8_campaign import file_hash
    from phycoflow_reconstruction.evaluation.round8_decision import KEYS
    identity={'config_sha256':'config','frozen_family_artifacts_sha256':'family',
              'fair_shape_artifact_sha256':'shape','native_observer_identity':{'mesh':'native'}}
    policy={'schema':'round8_frozen_policy_v1','endpoint':'C8','alpha':.37,
            'score_floors':{'state':.1,'scale':.1,'shape':.1}}
    pp=tmp_path/'policy.json'; pp.write_text(json.dumps(policy)); policy_hash=file_hash(pp)
    def write_score(alpha):
        values={'state':2-.2*alpha,'scale':1-.1*alpha,'shape':-2-.4*alpha}
        risk=[1+.2*alpha,.05*alpha]
        obj={'schema':'round8_common_observers_v1','population':'ASSESS','alpha':alpha,
             'source_panel_sha256':'source','endpoint_panel_sha256':'endpoint',
             'assessment_policy_sha256':policy_hash,'member_risk_physical':risk,
             'rows':[{'sample_id':'fixed','losses':{KEYS[k]:v for k,v in values.items()}}],**identity}
        path=tmp_path/f'score_{alpha}.json';path.write_text(json.dumps(obj))
        return path,values,risk
    frontier={}
    for alpha in (0.,.25,.5,.75,1.):
        path,values,risk=write_score(alpha)
        frontier[str(alpha)]={'path':str(path),'sha256':file_hash(path),'scores':values,'member_risk_physical':risk}
    selected,_,_=write_score(.37)
    assessment={'schema':'round8_assessment_v1','endpoint':'C8','alpha':.37,'policy_sha256':policy_hash,
                'source_report':frontier['0.0']['path'],'selected_report':str(selected),
                'frontier':frontier,'observer_identity':identity,'field_names':['T','CO']}
    ap=tmp_path/'assessment.json';ap.write_text(json.dumps(assessment))
    out=report.build_report(tmp_path/'report',assessment_paths={'C8':ap},policy_paths={'C8':pp})
    data=out['amplitude_frontier']['C8']; point=data['selected']
    assert point['alpha']==.37 and point['gains']['state']==pytest.approx(.037)
    assert point['gains']['shape']==pytest.approx(.074)
    assert point['max_defined_risk_ratio']==pytest.approx(1.074)
    assert point['zero_source_risk_fields'][0]['absolute_risk']==pytest.approx(.0185)
    assert point['zero_source_risk_fields'][0]['violates_zero_bound']
    assert data['score_denominators']=={'state':2,'scale':1,'shape':2}
    assert out['verification']['png_images_decoded']==1
    text=Path(out['markdown']).read_text()
    assert 'figures/amplitude_frontier.png' in text and 'figures/amplitude_frontier.pdf' in text
    assert 'SOURCE scores and positive denominators' in text
    frontier['0.5']['sha256']='invalid';ap.write_text(json.dumps(assessment))
    with pytest.raises(ValueError,match='hash differs'):
        report._frontier_data({'C8':assessment},{'C8':policy},{'C8':pp})


def test_amplitude_frontier_all_zero_source_risks_are_undefined():
    from phycoflow_reconstruction.evaluation.round8_decision import KEYS
    score={'alpha':.5,'member_risk_physical':[0.,.1],
           'rows':[{'losses':dict.fromkeys(KEYS.values(),1.)}]}
    point=report._frontier_point(score,dict.fromkeys(KEYS,1.),dict.fromkeys(KEYS,1.),[0.,0.],['A','B'])
    assert point['max_defined_risk_ratio'] is None
    assert [x['violates_zero_bound'] for x in point['zero_source_risk_fields']]==[False,True]
    assert point['zero_source_risk_fields'][1]['absolute_risk']==.1


def _decomposition_row():
    return {'sample_id':'sample','losses':{'global_distribution.total':1.5,'multiscale_spectrum.total':2.75,'topology.total':1.},
            'diagnostics':{'global_distribution':{'attraction':2.,'separation':.5},
                'topology':{'attraction_per_sample':[1.5],'separation_per_sample':[.5]},
                'multiscale_spectrum':{'components':{
                    'signed':{'weight':1.,'executed':True},
                    'joint':{'weight':.25,'executed':True},
                    'inactive':{'weight':0.,'executed':False}}}},
            'components':{'signed':{'weighted_attraction':3.,'weighted_separation':1.},
                          'joint':{'attraction':4.,'repulsion':1.}}}


def test_scale_complete_weighted_attraction_and_missing_refusal():
    row=_decomposition_row()
    assert report._family_attraction_separation(row,'scale')==pytest.approx((4.,1.25))
    assert report._family_attraction_separation(row,'state')==pytest.approx((2.,.5))
    assert report._family_attraction_separation(row,'shape')==pytest.approx((1.5,.5))
    del row['components']['joint']['repulsion']
    assert report._family_attraction_separation(row,'scale') is None
    row=_decomposition_row(); row['losses']['multiscale_spectrum.total']=10.
    assert report._family_attraction_separation(row,'scale') is None


def test_assessment_figures_render_two_supports(tmp_path):
    series=[]
    for population in ('TRAIN','ASSESS'):
        for epoch in (1,10):
            path=tmp_path/f'{population}_{epoch}.json'
            path.write_text(json.dumps({'schema':'round8_common_observers_v1','population':population,
                'alpha':.5,'rows':[_decomposition_row()]}))
            series.append({'endpoint':'C8','population':population,'epoch':epoch,'path':str(path)})
    out=report.build_report(tmp_path/'result',assessment_history=series)
    assert out['verification']['png_images_decoded']==2
    assert out['verification']['pdf_pages_decoded']==2
    assert 'Complete recorded family-total attraction and separation' in Path(out['markdown']).read_text()


def test_scale_fallback_uses_hash_verified_scientific_config(tmp_path):
    from phycoflow_reconstruction.evaluation.round8_campaign import canonical_hash
    import yaml
    config={'dataset':{},'rollout':{},'observation_consistency':{},'coherence':{'families':{'multiscale_spectrum':{
        'objective':{'kind':'signed_band_energy_score','score_definition_version':'signed_spectral_mean_augmented_v4',
                     'joint_energy_score_weight':.25},
        'relation_extension':{'definition':'self_rho_fair_v1','self_weight':.5,'rho_weight':.25}}}}}
    cfg=tmp_path/'config.yaml'; cfg.write_text(yaml.safe_dump(config))
    row=_decomposition_row();row['diagnostics']['multiscale_spectrum']={}
    prefix='multiscale_spectrum.'
    row['components']={prefix+'signed_band_energy_score.conditional':{'weighted_attraction':3.,'weighted_separation':1.},
        prefix+'joint_spectral_mean_energy_score.conditional':{'attraction':4.,'repulsion':1.},
        prefix+'relation_self_energy_score.conditional':{'attraction':2.,'repulsion':1.},
        prefix+'relation_rho_crps.conditional':{'attraction':2.,'repulsion':1.}}
    row['losses']['multiscale_spectrum.total']=3.5
    weights=report._scale_weights_from_config(config)
    assert report._family_attraction_separation(row,'scale',weights)==pytest.approx((5.5,2.))
    del row['components'][prefix+'relation_rho_crps.conditional']
    assert report._family_attraction_separation(row,'scale',weights) is None
    config['coherence']['families']['multiscale_spectrum']['objective']['coherence_crps_weight']=.1
    assert report._scale_weights_from_config(config) is None
    score=tmp_path/'score.json';score.write_text(json.dumps({'schema':'round8_common_observers_v1','population':'TRAIN','alpha':.5,
        'config_sha256':'bad','rows':[row]}))
    with pytest.raises(ValueError,match='scientific config hash differs'):
        report._assessment_figure([{'endpoint':'C8','population':'TRAIN','epoch':1,'path':str(score),'config_path':str(cfg)}],tmp_path/'figures')
    obj=json.loads(score.read_text());obj['config_sha256']=canonical_hash(yaml.safe_load(cfg.read_text()));score.write_text(json.dumps(obj))
    assert report._assessment_figure([{'endpoint':'C8','population':'TRAIN','epoch':1,'path':str(score),'config_path':str(cfg)}],tmp_path/'figures')


def test_parameter_diagnostics_preserve_measured_values_and_require_complete(tmp_path):
    from phycoflow_reconstruction.training.round8_calibration import BLOCKS, InternalCalibration
    blocks=[f'{family}.{block}' for family,names in BLOCKS.items() for block in names]
    calibration=InternalCalibration({k:1. for k in blocks},{k:2. for k in blocks},{k:True for k in blocks})
    for epoch in (1,3,6,8,10): calibration.observe(epoch,{k:float(i+epoch) for i,k in enumerate(blocks)})
    calibration.freeze(10)
    log=tmp_path/'calibration.jsonl'
    rows=calibration.records+[{'event':'frozen','epoch':10,'coefficients':dict(calibration.coefficients),'units':calibration.units,'priorities':calibration.priorities}]
    log.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    families=['global_distribution','multiscale_spectrum','topology']
    history=tmp_path/'full_updates.jsonl'
    def row(epoch):
        diag={'schema':'round8_actual_step_v1','identity':{'epoch':epoch}}
        for level,names in (('block',blocks),('family',families)):
            for suffix,base in (('gradient_norms',1),('gradient_dot_config_direction',-2),('negative_gradient_dot_displacement',3)):
                diag[f'{level}_{suffix}']={k:float(base+i) for i,k in enumerate(names)}
        return {'epoch':epoch,'step':epoch*13,'round8_actual_step':diag}
    history.write_text(''.join(json.dumps(row(e))+'\n' for e in (100,500,1000)))
    data=report._parameter_diagnostics({'C8':history},calibration.state_dict(),log,require_complete=True)
    assert data['calibration']['records'][2]['epoch']==6
    assert data['actual_steps']['C8']['500']['block_gradient_dot_config_direction'][blocks[0]]==-2
    text='\n'.join(report._parameter_diagnostic_lines(data))
    assert 'State: marginal' in text and 'chain-rule gradient contributions' in text and 'C8 epoch 1000, update 13000' in text
    history.write_text(json.dumps(row(100))+'\n')
    with pytest.raises(ValueError,match='missing scheduled'): report._parameter_diagnostics({'C8':history},calibration.state_dict(),log,require_complete=True)
    rows[-1]['coefficients'][blocks[0]]=100
    log.write_text(''.join(json.dumps(item)+'\n' for item in rows))
    with pytest.raises(ValueError,match='differs from checkpoint'): report._parameter_diagnostics({},calibration.state_dict(),log)


def test_copied_history_requires_complete_history_and_indexes_figures(tmp_path,monkeypatch):
    import matplotlib.pyplot as plt
    logs=tmp_path/'logs'; logs.mkdir()
    history=logs/'full_updates.jsonl'; history.write_text(json.dumps({'step':130})+'\n')
    (logs/'round8_history_state.json').write_text(json.dumps({'last_step':130,'history_rows':1}))
    fig,ax=plt.subplots(); ax.plot([1,10],[0,1]); fig.savefig(logs/'round8_history.png'); fig.savefig(logs/'round8_history.pdf'); plt.close(fig)
    monkeypatch.setattr(report,'validate_report',lambda *_:{})
    result=report.build_report(tmp_path/'report',histories={'C8':history})
    text=Path(result['markdown']).read_text()
    assert 'training_components_c8' in text and 'epoch-10 calibration freeze' in text
    assert (tmp_path/'report/figures/training_components_c8.pdf').read_bytes()==(logs/'round8_history.pdf').read_bytes()
    provenance=json.loads((tmp_path/'report/input_provenance.json').read_text())
    assert provenance['training_history_png:C8']['sha256']==report.file_hash(logs/'round8_history.png')
    (logs/'round8_history_state.json').write_text(json.dumps({'last_step':129,'history_rows':1}))
    with pytest.raises(ValueError,match='does not cover'): report._copy_training_history_figures({'C8':history},tmp_path/'other',require_complete=True)
