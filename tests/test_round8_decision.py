import json
from pathlib import Path
import numpy as np
import pytest
from phycoflow_reconstruction.evaluation import round8_decision as r
from phycoflow_reconstruction.evaluation.round8_campaign import file_hash


def _report(source,endpoint,alpha, values, risks, population='ASSESS', policy_hash=None):
    rows=[]
    for i,(s,sh,st) in enumerate(values):
        rows.append({'sample_id':str(i),'legacy_source':'validation' if i==0 else 'test', 'losses':{r.KEYS['scale']:s,r.KEYS['shape']:sh,r.KEYS['state']:st,r.SIGNED:s,r.JOINT:s},'member_mse_physical':[[risks[i]],[risks[i]]]})
    return {'schema':'round8_common_observers_v1','population':population,'source_panel_sha256':source,'endpoint_panel_sha256':endpoint,'alpha':alpha,'config_sha256':'cfg','frozen_family_artifacts_sha256':{'a':'b'},'fair_shape_artifact_sha256':'fair','native_observer_identity':'native' if population=='ASSESS' else None,'assessment_policy_sha256':policy_hash,'rows':rows,'member_risk_physical':[sum(risks)/len(risks)]}


def test_full_train_assess_flow_with_actual_cached_quadratic(tmp_path, monkeypatch):
    paths={}
    for pop in ('TRAIN','ASSESS'):
        for name in ('SOURCE','C8'):
            directory=tmp_path/pop/name;directory.mkdir(parents=True)
            rows=[]
            for i in range(2):
                path=directory/f'{i}.npz'
                source=np.ones((2,3,1)); pred=source if name=='SOURCE' else source*.9
                np.savez(path,draws_physical=pred,target_physical=np.zeros((3,1)),physical_weights=np.full(3,1/3),native_ids=np.arange(3),physical_mask=np.ones(3,dtype=bool),inclusion_probability=np.asarray(1.))
                rows.append({'sample_id':str(i),'source':'validation' if i==0 else 'test','path':path.name})
            payload={'endpoint':name,'population':pop,'manifest_sha256':'manifest','source_checkpoint_sha256':'source_ckpt','checkpoint_sha256':'source_ckpt' if name=='SOURCE' else 'candidate_ckpt','field_names':['a'],'conditions':rows}
            p=directory/'panel.json';p.write_text(json.dumps(payload));paths[(pop,name)]=p
    def validate(left,right): return json.loads(Path(left).read_text()),json.loads(Path(right).read_text())
    monkeypatch.setattr(r,'validate_matched_panels',validate)
    def score(config,source,endpoint,*,output_path,alpha,assessment_policy=None,**kwargs):
        pop=validate(source,endpoint)[0]['population']; v=1.-.2*alpha
        report=_report(file_hash(source),file_hash(endpoint),alpha,[(v,v,v),(v,v,v)],[(1-.1*alpha)**2]*2,pop, file_hash(assessment_policy) if assessment_policy else None)
        Path(output_path).parent.mkdir(parents=True,exist_ok=True);Path(output_path).write_text(json.dumps(report))
    monkeypatch.setattr(r,'score_cached_panel',score)
    policy_path=r.freeze_train_policy({},paths['TRAIN','SOURCE'],paths['TRAIN','C8'],fair_shape_artifact='x',score_floors={'scale':.01,'shape':.01,'state':.01},output_dir=tmp_path/'out')
    policy=json.loads(policy_path.read_text())
    assert policy['alpha']==.5 and set(policy['families'])=={'state','scale','shape'}
    result_path=r.assess_endpoint({},paths['ASSESS','SOURCE'],paths['ASSESS','C8'],policy_path,fair_shape_artifact='x',native_bundle='x',output_dir=tmp_path/'out',regime_by_id={'0':'hot','1':'cold'})
    result=json.loads(result_path.read_text())
    assert result['eligible'] and result['alpha']==.5 and len(result['frontier'])==5
    assert result['groups']['regime']['hot']['n']==1
    assert result['per_condition'][0]['sample_id']=='0'


def test_frozen_family_retention_and_zero_risk():
    src=_report('s','e',0.,[(1,1,1)]*2,[0.,0.]);raw=_report('s','e',1.,[(.9,.8,.9)]*2,[.1,.1]);blend=_report('s','e',.5,[(.95,.9,.95)]*2,[.01,.01]); p={'alpha':.5,'families':['shape'],'score_floors':dict.fromkeys(r.KEYS,.01)}
    answer=r.assess_fixed_policy(p,src,raw,blend,field_names=['a'])
    assert not answer['eligible'] and answer['risk_ratio']==[None] and answer['rmse_change']==[None]
    bad=_report('s','e',1.,[(.9,1.1,.9)]*2,[0.,0.]);okay=_report('s','e',.5,[(.95,.9,.95)]*2,[0.,0.])
    assert not r.assess_fixed_policy(p,src,bad,okay,field_names=['a'])['eligible']


def test_strong_scale_computation_and_operational_choice():
    src=_report('s','source',0.,[(1,1,1)]*2,[1,1]);b=_report('s','b',1.,[(.99,1,1)]*2,[1,1]);c=_report('s','c',1.,[(.97,.94,.97)]*2,[1.01,1.01])
    assert r.strong_scale(src,c,b,score_floor=.01,assessment={'eligible':True})['pass']
    cost=lambda t:{'actual_updates':10,'amortized_seconds_per_update':t,'campaign_gpu_seconds':100,'comparison_device':'gpu-x'}
    assert r.computation_decision(src,b,c,floors=dict.fromkeys(r.KEYS,.01),b8_cost=cost(1.),c8_cost=cost(1.05))['retain_c8_default']
    base={'device_uuid':'gpu-x','batch_size':1,'member_count':2,'warmup_policy':'fixed','request_ids_sha256':'abc','synchronized':True,'latency_seconds':1.}
    test={**base,'latency_seconds':2.,'peak_resident_bytes':40,'both_models_resident':True}
    decision=r.operational_choice({'C8':{'eligible':True,'alpha':.5},'B8':{'eligible':True,'alpha':.5}}, {'C8':test,'B8':test},source_benchmark=base,device_vram_bytes=100)
    assert decision['choice']=='C8'
    assert r.operational_choice({'C8':{'eligible':True,'alpha':.5}}, {'C8':{**test,'latency_seconds':3.}},source_benchmark=base,device_vram_bytes=100)['choice']=='SOURCE'


def test_reused_source_baseline_is_bound_to_new_endpoint(tmp_path):
    original=_report('sourcehash','oldendpoint',0.,[(1,2,3)]*2,[1,1],population='TRAIN')
    src=tmp_path/'old.json';src.write_text(json.dumps(original))
    target=tmp_path/'rebound.json'
    r._reuse_source_report(src,target,source_sha='sourcehash',endpoint_sha='newendpoint',population='TRAIN',expected_ids=['0','1'])
    value=json.loads(target.read_text())
    assert value['endpoint_panel_sha256']=='newendpoint' and value['shared_source_baseline_sha256']==file_hash(src)
    assert value['rows']==original['rows']
    with pytest.raises(ValueError,match='baseline identity'):
        r._reuse_source_report(src,tmp_path/'bad.json',source_sha='other',endpoint_sha='newendpoint',population='TRAIN',expected_ids=['0','1'])


def test_frozen_json_write_is_idempotent_and_never_replaced(tmp_path):
    path=tmp_path/'policy.json'
    r._save(path,{'policy':'same','values':[1,2]})
    first=path.read_bytes()
    r._save(path,{'values':[1,2],'policy':'same'})
    assert path.read_bytes()==first
    with pytest.raises(ValueError,match='differs'):
        r._save(path,{'policy':'changed'})
    assert path.read_bytes()==first
