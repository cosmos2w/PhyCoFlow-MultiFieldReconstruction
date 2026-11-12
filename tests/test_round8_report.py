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
    result=report.build_report(tmp_path/'result',decision_path=decision,histories={'C8':history},reconstructions=[{'source_panel':folders[0]/'panel.json','endpoint_panel':folders[1]/'panel.json','sample_id':'x','field':'T','alpha':.5,'member':0}])
    assert result['reconstructions'][0]['rmse']==pytest.approx(.5)
    assert result['verification']['png_images_decoded']==2
    assert result['verification']['pdf_pages_decoded']==2
    content=Path(result['markdown']).read_text()
    assert 'weighted RMSE' in content and 'TRAIN' in content and 'SOURCE' in content
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
