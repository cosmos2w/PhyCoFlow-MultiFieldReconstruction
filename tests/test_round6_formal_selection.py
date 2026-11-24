import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import yaml

BASE=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

@pytest.fixture
def fixtures(tmp_path,monkeypatch):
    driver=load('candidate_round6_driver',BASE/'scripts/posttraining/run_car_round6.py')
    campaign=tmp_path/'round_6';(campaign/'configs').mkdir(parents=True)
    arm=driver.ARM
    config={'optimization':{'multitask':{'coherence_aggregation':{'method':'torchjd_config'}}},'output':{'experiment_name':'original'},'coherence':{'scalable_anchored_coherence':{}}}
    def write_config(name,cfg):
        p=campaign/'configs'/name;p.write_text(yaml.safe_dump(cfg,sort_keys=False));return driver.reference(p)
    base={'schema':'car_round6_prepared_v1','arms':{arm:write_config('original.yaml',config)},'source_snapshot':'source','source_reference':{'path':'source-reference','sha256':'risk'},'native_loss':False,'query_law':'shared'}
    base_path=campaign/'configs/prepared.json';base_path.write_text(json.dumps(base))
    derived=copy.deepcopy(config);derived['optimization']['multitask']['coherence_aggregation']['family_vjp']='batched'
    selected=copy.deepcopy(base);selected['arms']={arm:write_config('batched.yaml',derived)};selected['base_prepared']=driver.reference(base_path)
    def receipt(name,obj):
        p=campaign/'configs'/name;p.write_text(json.dumps(obj));return driver.reference(p)
    selected['execution_qualification']={'production_resume':receipt('resume.json',{'qualified':True,'config_sha256':selected['arms'][arm]['sha256']}),'engineering':receipt('engineering.json',{'qualified':True,'config_sha256':base['arms'][arm]['sha256']})}
    selection=campaign/'configs/formal_prepared.json';selection.write_text(json.dumps(selected))
    return driver,campaign,base,selected,selection,config,derived


def test_freeze_training_resume_and_assessment_share_selected_manifest(fixtures,monkeypatch,tmp_path):
    driver,campaign,base,selected,selection,original,derived=fixtures
    pilot=campaign/'configs/pilot.json';pilot.write_text(json.dumps({'qualified':True}))
    forecast=campaign/'configs/forecast.json';forecast.write_text(json.dumps({'fits_gpu_budget':True,'forecast_gpu_seconds':100.,'pilot':driver.reference(pilot)}))
    code=tmp_path/'frozen_code';code.mkdir();(code/'scripts').mkdir()
    monkeypatch.setattr(driver,'ROOT',code);monkeypatch.setattr(driver,'CODE',code)
    monkeypatch.setattr(driver,'freeze_execution_code',lambda root,dest,drivers:code)
    driver.freeze(SimpleNamespace(prepared=selection,pilot=pilot,forecast=forecast),campaign)
    freeze=json.loads((campaign/'configs/formal_freeze.json').read_text())
    assert driver.frozen_prepared(campaign,freeze)==(selection,selected)
    monkeypatch.setenv('CAR_ROUND6_LEASE_TOKEN','lease')
    import phycoflow_reconstruction.training.post_training as trainer
    output=tmp_path/'run';output.mkdir();(output/'status.json').write_text('{}')
    received=[]
    monkeypatch.setattr(trainer,'run_post_training',lambda cfg,**kw:received.append(cfg) or output)
    driver.train(SimpleNamespace(pilot=False,stop_epoch=100,resume=None,candidate_config=None,candidate_sha256=None,resume_qualification=None),campaign)
    assert received[-1]==derived
    (output/'resolved_config.yaml').write_text(yaml.safe_dump(derived))
    driver.train(SimpleNamespace(pilot=False,stop_epoch=100,resume=output,candidate_config=None,candidate_sha256=None,resume_qualification=None),campaign)
    assert received[-1]==derived
    (output/'resolved_config.yaml').write_text(yaml.safe_dump(original))
    with pytest.raises(ValueError,match='unchanged scientific configuration'):
        driver.train(SimpleNamespace(pilot=False,stop_epoch=100,resume=output,candidate_config=None,candidate_sha256=None,resume_qualification=None),campaign)
    import sys
    monkeypatch.setitem(sys.modules,'scripts.posttraining.run_car_round6',driver)
    assess=load('candidate_round6_assess',BASE/'scripts/posttraining/assess_car_round6.py')
    monkeypatch.setattr(assess,'legacy_context',lambda _: {'campaign':campaign,'prepared':base,'prepared_path':campaign/'configs/prepared.json','source':'verified'})
    result=assess.context(campaign)
    assert result['prepared']==selected and result['prepared_path']==selection and result['source']=='verified'


@pytest.mark.parametrize('mutation',['gamma','source','resume','timing','family'])
def test_derived_selection_rejects_scientific_drift_and_unbound_qualification(fixtures,mutation):
    driver,campaign,base,selected,selection,original,derived=fixtures
    if mutation in ('gamma','family'):
        path=Path(selected['arms'][driver.ARM]['path']);config=yaml.safe_load(path.read_text())
        if mutation=='gamma':config['optimization']['gamma']=9
        else:config['optimization']['multitask']['coherence_aggregation']['family_vjp']='separate'
        path.write_text(yaml.safe_dump(config));selected['arms'][driver.ARM]=driver.reference(path)
    elif mutation=='source':selected['source_snapshot']='different'
    elif mutation=='resume':selected['execution_qualification']['production_resume']['sha256']='bad'
    else:selected['execution_qualification']['engineering']['sha256']='bad'
    selection.write_text(json.dumps(selected))
    with pytest.raises(ValueError):driver.selected_prepared(campaign,selection)


def test_original_default_and_legacy_freeze_remain_readable(fixtures):
    driver,campaign,base,selected,selection,original,derived=fixtures
    path=campaign/'configs/prepared.json'
    assert driver.selected_prepared(campaign,path)==base
    assert driver.frozen_prepared(campaign,dict(base, schema='car_round6_freeze_v1'))==(path,base)


def grouped_fixture(fixtures):
    driver,campaign,base,selected,selection,original,derived=fixtures
    path=Path(selected['arms'][driver.ARM]['path']);cfg=yaml.safe_load(path.read_text())
    cfg['coherence']['scalable_anchored_coherence']['condition_query_groups']=8
    path.write_text(yaml.safe_dump(cfg));selected['arms'][driver.ARM]=driver.reference(path);selected['query_law']='condition_groups_8'
    def ref(name,value):
        p=campaign/'configs'/name;p.write_text(json.dumps(value));return driver.reference(p)
    resume=ref('grouped_resume.json',{'qualified':True,'config_sha256':selected['arms'][driver.ARM]['sha256']})
    geom=ref('geometry.json',{'status':'completed','sample_count':800})
    cost=ref('query_cost.json',{'status':'completed'})
    decision=ref('query_decision.json',{'qualified':True,'geometry_sha256':geom['sha256'],'cost_sha256':cost['sha256']})
    pilot=ref('grouped_pilot.json',{'qualified':True,'epochs':20,'candidate_config_sha256':selected['arms'][driver.ARM]['sha256'],'resume_qualification':resume})
    selected['execution_qualification'].update(production_resume=resume,query_decision=decision,query_geometry=geom,query_cost=cost,candidate_pilot=pilot)
    selection.write_text(json.dumps(selected));return selected,cfg


def test_grouped_selection_requires_own_full_bound_evidence(fixtures):
    driver,campaign,base,selected,selection,original,derived=fixtures
    selected,cfg=grouped_fixture(fixtures)
    assert driver.selected_prepared(campaign,selection)==selected
    for name in ('query_geometry','query_cost','query_decision','candidate_pilot','production_resume'):
        changed=copy.deepcopy(selected);changed['execution_qualification'][name]['sha256']='incorrect'
        selection.write_text(json.dumps(changed))
        with pytest.raises(ValueError):driver.selected_prepared(campaign,selection)
    selection.write_text(json.dumps(selected))
    cfg['optimization']['multitask']['coherence_aggregation']['family_vjp']='separate'
    path=Path(selected['arms'][driver.ARM]['path']);path.write_text(yaml.safe_dump(cfg));selected['arms'][driver.ARM]=driver.reference(path);selection.write_text(json.dumps(selected))
    with pytest.raises(ValueError,match='derived config differs'):driver.selected_prepared(campaign,selection)


def test_grouped_pilot_uses_explicit_candidate_and_matched_resume(fixtures,monkeypatch,tmp_path):
    driver,campaign,base,selected,selection,original,derived=fixtures
    selected,cfg=grouped_fixture(fixtures);binding=selected['arms'][driver.ARM]
    monkeypatch.setenv('CAR_ROUND6_LEASE_TOKEN','lease')
    import phycoflow_reconstruction.training.post_training as trainer
    output=tmp_path/'run';output.mkdir();(output/'status.json').write_text('{}');received=[]
    monkeypatch.setattr(trainer,'run_post_training',lambda cfg,**kw:received.append(cfg) or output)
    args=SimpleNamespace(pilot=True,stop_epoch=20,resume=None,candidate_config=Path(binding['path']),candidate_sha256=binding['sha256'],resume_qualification=Path(selected['execution_qualification']['production_resume']['path']))
    (campaign/'operations').mkdir(exist_ok=True)
    original_pilot=campaign/'operations/pilot_run.json';original_pilot.write_bytes(b'original hash-bound pilot receipt')
    before=original_pilot.read_bytes()
    driver.train(args,campaign)
    assert original_pilot.read_bytes()==before
    assert (campaign/'operations'/('pilot_'+driver.ARM+'_GroupedQueries.json')).is_file()
    assert received[-1]['output']['experiment_name'].endswith('_GroupedQueries')
    assert received[-1]['coherence']['scalable_anchored_coherence']['condition_query_groups']==8
    args.candidate_sha256='bad'
    with pytest.raises(ValueError,match='bound artifact changed'):driver.train(args,campaign)
    assert yaml.safe_load(Path(base['arms'][driver.ARM]['path']).read_text())==original


def test_pilot_review_binds_candidate_config_and_resume(fixtures):
    driver,campaign,base,selected,selection,original,derived=fixtures
    selected,cfg=grouped_fixture(fixtures);binding=selected['arms'][driver.ARM]
    review=load('candidate_grouped_pilot_review',BASE/'scripts/qualification/car_round6_pilot_review.py')
    resume=json.loads(Path(selected['execution_qualification']['production_resume']['path']).read_text())
    expected,used=review.expected_pilot_config(campaign,Path(binding['path']),binding['sha256'],resume)
    assert expected['output']['experiment_name'].endswith('_GroupedQueries') and used==binding
    with pytest.raises(ValueError,match='same prepared configuration'):
        review.expected_pilot_config(campaign,Path(binding['path']),binding['sha256'],{'config_sha256':'other'})
    original_expected,_=review.expected_pilot_config(campaign,None,None,{'config_sha256':base['arms'][driver.ARM]['sha256']})
    assert not original_expected['output']['experiment_name'].endswith('_GroupedQueries')


def test_grouped_formal_freeze_cannot_use_old_pilot(fixtures,tmp_path,monkeypatch):
    driver,campaign,base,selected,selection,original,derived=fixtures
    selected,_=grouped_fixture(fixtures)
    old=campaign/'configs/old_pilot.json';old.write_text(json.dumps({'qualified':True}))
    forecast=campaign/'configs/forecast_old.json';forecast.write_text(json.dumps({'fits_gpu_budget':True,'forecast_gpu_seconds':100.,'pilot':driver.reference(old)}))
    with pytest.raises(ValueError,match='requires its own qualified pilot'):
        driver.freeze(SimpleNamespace(prepared=selection,pilot=old,forecast=forecast),campaign)
