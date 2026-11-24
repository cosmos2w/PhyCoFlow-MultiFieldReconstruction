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
    config={'optimization':{'multitask':{'coherence_aggregation':{'method':'torchjd_config'}}},'output':{'experiment_name':'original'}}
    def write_config(name,cfg):
        p=campaign/'configs'/name;p.write_text(yaml.safe_dump(cfg,sort_keys=False));return driver.reference(p)
    base={'schema':'car_round6_prepared_v1','arms':{arm:write_config('original.yaml',config)},'source_snapshot':'source','source_reference':{'path':'source-reference','sha256':'risk'},'native_loss':False}
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
    driver.train(SimpleNamespace(pilot=False,stop_epoch=100,resume=None),campaign)
    assert received[-1]==derived
    (output/'resolved_config.yaml').write_text(yaml.safe_dump(derived))
    driver.train(SimpleNamespace(pilot=False,stop_epoch=100,resume=output),campaign)
    assert received[-1]==derived
    (output/'resolved_config.yaml').write_text(yaml.safe_dump(original))
    with pytest.raises(ValueError,match='unchanged scientific configuration'):
        driver.train(SimpleNamespace(pilot=False,stop_epoch=100,resume=output),campaign)
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
