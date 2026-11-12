import json
from pathlib import Path

import pytest
import yaml

from phycoflow_reconstruction.evaluation.round8_campaign import canonical_hash, file_hash
from phycoflow_reconstruction.training import round8_execution as execution


def test_formal_admission_binds_resource_forecast_to_current_qualification(tmp_path, monkeypatch):
    import torch
    from types import SimpleNamespace
    directory = tmp_path/'artifacts'
    directory.mkdir()
    artifacts, manifest, profile = [directory/name for name in ('runtime.pt', 'manifest.json', 'profile.json')]
    for path in (artifacts, manifest, profile):
        path.write_text(path.name)
    qualification = directory/'qualified.json'
    qualification.write_text(json.dumps({'status':'completed', 'callback_memory_qualification':{
        'status':'completed', 'device_name':'same-class', 'device_total_bytes':1000,
        'training_and_optimizer_resident':True, 'second_source_model':True,
        'native_sites':40300, 'members':2, 'manifest_sha256':file_hash(manifest),
        'peak_allocated_bytes':100, 'peak_reserved_bytes':300,
        'profile_path':str(profile), 'profile_sha256':file_hash(profile)}}))
    (directory/'stage0_complete.json').write_text(json.dumps({'status':'completed'}))
    forecast = {'fits':True, 'closeout_reserved_seconds':14400,
                'artifacts_sha256':file_hash(artifacts), 'qualification_sha256':file_hash(qualification)}
    forecast_path = directory/'resource_forecast.json'
    forecast_path.write_text(json.dumps(forecast))
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda *args:SimpleNamespace(name='same-class', total_memory=1000))
    config = {'optimization':{'epochs':1000, 'steps_per_epoch':13}}
    execution.require_formal_admission(tmp_path, artifacts_path=artifacts, config=config, manifest_path=manifest)
    qualification.write_text(qualification.read_text()+'\n')
    with pytest.raises(ValueError, match='current timing qualification'):
        execution.require_formal_admission(tmp_path, artifacts_path=artifacts, config=config, manifest_path=manifest)


def test_scheduled_assessment_reuses_sources_and_pins_milestones(tmp_path, monkeypatch):
    from phycoflow_reconstruction.evaluation import round8_decision
    campaign = tmp_path / 'campaign'
    manifest, fair, native = tmp_path/'manifest', tmp_path/'fair', tmp_path/'native'
    for p in (manifest, fair, native): p.write_text(p.name)
    config = {'dataset': {}, 'coherence': {}, 'rollout': {}, 'observation_consistency': {}}
    run = tmp_path / 'run'
    (run/'checkpoints').mkdir(parents=True)
    (run/'logs').mkdir()
    (run/'resolved_config.yaml').write_text(yaml.safe_dump(config))
    for epoch in (250,500,750,1000):
        (run/'checkpoints'/f'epoch_{epoch:03d}.pt').write_text(str(epoch*13))
    (run/'checkpoints/last.pt').write_text('rolling alias')
    for population in ('TRAIN','ASSESS'):
        path = campaign/'comparison/SOURCE'/population/'panel.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'endpoint':'SOURCE','population':population,
                                    'manifest_sha256':file_hash(manifest),
                                    'conditions':[{'sample_id':'case'}]}))
    calls = {'infer': [], 'score': [], 'policy': [], 'assessment': []}
    def load_checkpoint(path): return {'global_step':int(Path(path).read_text())}
    def validate(path): return json.loads(Path(path).read_text())
    def matched(source, endpoint): return validate(source), validate(endpoint)
    def infer(config, manifest_path, out, *, population, endpoint, checkpoint, **kwargs):
        out=Path(out);out.mkdir(parents=True)
        epoch=int(Path(checkpoint).read_text())//13
        panel={'endpoint':endpoint,'epoch':epoch,'population':population,
               'manifest_sha256':file_hash(manifest_path),'checkpoint_sha256':file_hash(checkpoint),
               'conditions':[{'sample_id':'case'}]}
        (out/'panel.json').write_text(json.dumps(panel)); calls['infer'].append((epoch,population,Path(checkpoint).name))
        return out/'panel.json'
    def score(config, source, endpoint, *, output_path, alpha, fair_shape_artifact, native_bundle=None,
              interim_diagnostic=False, **kwargs):
        output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True)
        population=validate(source)['population']
        report={'schema':'round8_common_observers_v1','population':population,'alpha':alpha,
                'source_panel_sha256':file_hash(source),'endpoint_panel_sha256':file_hash(endpoint),
                'config_sha256':canonical_hash(config),'fair_shape_artifact_sha256':file_hash(fair_shape_artifact),
                'frozen_family_artifacts_sha256':{},
                'native_observer_identity':{'native_bundle_sha256':file_hash(native_bundle)} if native_bundle else None,
                'interim_diagnostic':interim_diagnostic,
                'rows':[{'sample_id':'case','losses':{'global_distribution.total':1.,'multiscale_spectrum.total':1.,'topology.total':1.}}]}
        output_path.write_text(json.dumps(report));calls['score'].append((population,alpha,Path(source)==Path(endpoint)))
    def policy(config,source,panel,*,output_dir,source_report_path,**kwargs):
        assert Path(source_report_path).exists()
        path=Path(output_dir)/'frozen_policy.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('{}');calls['policy'].append(path);return path
    def assess(config,source,panel,policy_path,*,output_dir,source_report_path,native_bundle,**kwargs):
        assert Path(policy_path).exists() and Path(source_report_path).exists() and native_bundle == native
        path=Path(output_dir)/'assessment.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('{}');calls['assessment'].append(path);return path
    monkeypatch.setattr(execution,'load_project_checkpoint',load_checkpoint)
    monkeypatch.setattr(execution,'validate_panel',validate)
    monkeypatch.setattr(execution,'validate_matched_panels',matched)
    monkeypatch.setattr(execution,'infer_panel',infer)
    monkeypatch.setattr(execution,'score_cached_panel',score)
    monkeypatch.setattr(round8_decision,'freeze_train_policy',policy)
    monkeypatch.setattr(round8_decision,'assess_endpoint',assess)
    observer=execution.ScheduledObserver(campaign=campaign,manifest=manifest,arm='B8',
        artifacts={'score_floors':{}},native_bundle=native,fair_shape_artifact=fair,device='cpu')
    for epoch in (250,500,750,1000): observer(run,run/'checkpoints/last.pt',epoch*13)
    assert len(calls['infer'])==6
    assert all(name == f'epoch_{epoch:03d}.pt' for epoch,_,name in calls['infer'])
    assert [(p,a) for p,a,is_source in calls['score'] if is_source] == [('ASSESS',0.),('TRAIN',0.)]
    assert len(calls['policy'])==1 and len(calls['assessment'])==1
    before={k:len(v) for k,v in calls.items()}
    for epoch in (250,500,750,1000): observer(run,run/'checkpoints/last.pt',epoch*13)
    assert {k:len(v) for k,v in calls.items()} == before
    (campaign/'comparison/SOURCE/ASSESS/common_source_score.json').write_text('{}')
    with pytest.raises(ValueError,match='artifact changed'):
        observer(run,run/'checkpoints/last.pt',250*13)


def test_resident_profile_uses_one_declared_native_condition(tmp_path, monkeypatch):
    import torch
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    called=[]
    def infer(config, actual_manifest, output, **kwargs):
        called.append(kwargs)
        path=Path(output)/'profile.json';path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'schema':'round8_one_condition_profile_v1','status':'profiled',
                                    'conditions':[{'initial_field_sha256':['a','b']}]}))
        return path
    monkeypatch.setattr(execution,'infer_panel',infer)
    monkeypatch.setenv('ROUND8_PHYSICAL_GPU_UUID','GPU-1')
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *x:None)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda *x:(900,1000))
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda *x:100)
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda *x:120)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda *x:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *x:250)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda *x:300)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda *x:type('P',(),{'name':'same-class','total_memory':1000})())
    result=execution.profile_resident_inference({},campaign=tmp_path,manifest=manifest)
    assert called == [{'population':'ASSESS','endpoint':'SOURCE','device':'cuda:0','profiling_first_condition':True}]
    assert result['conditions']==1 and result['native_sites']==40300 and result['members']==2
    assert result['peak_reserved_bytes']==300 and result['device_name']=='same-class'


def test_qualification_receipt_republication_archives_exact_bytes(tmp_path):
    path=tmp_path/'qualified.json'
    old=b'{"status":"completed", "run":"old", "original_spacing": true}\n'
    path.write_bytes(old)
    previous=file_hash(path)
    new={'status':'completed','run':'new','optimizer_attempts':14}
    published=execution.publish_qualification_receipt(path,new)
    archive=tmp_path/f'qualified.{previous}.json'
    assert archive.read_bytes()==old
    assert published['previous_qualification_sha256']==previous
    current=path.read_bytes()
    assert execution.publish_qualification_receipt(path,new)==published
    assert path.read_bytes()==current and archive.read_bytes()==old
    with pytest.raises(ValueError,match='completed'):
        execution.publish_qualification_receipt(path,{'status':'failed'})
    assert path.read_bytes()==current


def test_resident_profile_output_can_be_scoped_to_qualification_run(tmp_path,monkeypatch):
    import torch
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    out=tmp_path/'warmup/artifacts/profile'
    def infer(config, actual_manifest, output, **kwargs):
        assert Path(output)==out
        path=Path(output)/'profile.json';path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'schema':'round8_one_condition_profile_v1','status':'profiled','conditions':[{'initial_field_sha256':['a','b']}]}))
        return path
    monkeypatch.setattr(execution,'infer_panel',infer)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *x:None)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda *x:(900,1000))
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda *x:100)
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda *x:120)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda *x:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *x:250)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda *x:300)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda *x:type('P',(),{'name':'synthetic','total_memory':1000})())
    actual=execution.profile_resident_inference({},campaign=tmp_path,manifest=manifest,output_dir=out)
    assert Path(actual['profile_path'])==out/'profile.json'
