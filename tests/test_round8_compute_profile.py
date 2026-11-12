import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import yaml

from phycoflow_reconstruction.training import round8_compute_profile as profile
from phycoflow_reconstruction.training.run_store import config_digest


@pytest.fixture
def instrumentation_fakes(monkeypatch):
    def original_score(self,name,family,prediction,reference,*,coordinates,context):
        return name
    def original_flat(loss,parameters): return loss
    def original_policy(self,model,optimizer,family_losses,*args,**kwargs):
        for value in family_losses.values(): profile.fidelity_budget._flat_gradient(value,[])
        profile.fidelity_budget._flat_gradient('constraint',[])
        return {'family_gradient_evaluations':3,'constraint_gradient_evaluations':1}
    monkeypatch.setattr(profile.Round8Runtime,'score_family',original_score)
    monkeypatch.setattr(profile.ExternalPolicy,'update_with_coherence_aggregation',original_policy)
    monkeypatch.setattr(profile.fidelity_budget,'_flat_gradient',original_flat)
    return original_score,original_policy,original_flat


def one_update(step):
    runtime=SimpleNamespace(step=step)
    for name in ('global_distribution','topology','multiscale_spectrum'):
        profile.Round8Runtime.score_family(runtime,name,None,None,None,coordinates=None,context={})
    losses={k: k for k in ('global_distribution','multiscale_spectrum','topology')}
    profile.ExternalPolicy.update_with_coherence_aggregation(None,None,None,losses,None,None,None)


def test_timing_wrappers_restore_and_label_real_family_order(instrumentation_fakes):
    records=[];sync_calls=[]
    with profile.synchronized_training_timings(device='cpu',records=records,synchronize=lambda:sync_calls.append(1)):
        one_update(12); one_update(13)
    assert len(sync_calls)==32
    profile._validate_records(records,12)
    policies=[v for v in records if v['kind']=='policy_update']
    assert [v['label'] for v in policies[0]['gradient_calls']]==['global_distribution','multiscale_spectrum','topology','constraint']
    assert profile.Round8Runtime.score_family is instrumentation_fakes[0]
    assert profile.ExternalPolicy.update_with_coherence_aggregation is instrumentation_fakes[1]
    assert profile.fidelity_budget._flat_gradient is instrumentation_fakes[2]
    with pytest.raises(RuntimeError):
        with profile.synchronized_training_timings(device='cpu',records=[],synchronize=lambda:None):
            raise RuntimeError('stop')
    assert profile.Round8Runtime.score_family is instrumentation_fakes[0]
    assert profile.fidelity_budget._flat_gradient is instrumentation_fakes[2]


def test_profile_clones_completed_arms_and_charges_exactly_four(tmp_path,monkeypatch,instrumentation_fakes):
    campaign=tmp_path/'campaign';qualification=campaign/'qualification'
    segments={};fork={}; original_paths={}
    for arm, mode in (('B8','control'),('C8','candidate')):
        parent=qualification/arm/'original';(parent/'checkpoints').mkdir(parents=True);(parent/'logs').mkdir()
        config={'runtime':{'device':'cpu','plot_every_steps':10},'optimization':{'steps_per_epoch':1}}
        (parent/'resolved_config.yaml').write_text(yaml.safe_dump(config))
        digest=config_digest(config)
        (parent/'run_manifest.json').write_text(json.dumps({'config_sha256':digest}))
        (parent/'logs'/'history.jsonl').write_text('original\n')
        checkpoint={'global_step':12,'qualification_only':True,'config_sha256':digest,'round8_runtime':{'mode':mode,'step':12}}
        torch.save(checkpoint,parent/'checkpoints'/'last.pt')
        original_paths[arm]=(parent/'checkpoints'/'last.pt').read_bytes()
        segments[arm]={'run':str(parent),'update_seconds':[1.,2.]}
        fork[arm]={'run':str(parent)}
    receipt={'schema':'round8_disposable_qualification_v1','status':'completed','optimizer_attempts':14,
             'fork':{'qualification_only':True,'arms':fork},'segments':segments}
    receipt_path=campaign/'artifacts'/'qualified.json';receipt_path.parent.mkdir();receipt_path.write_text(json.dumps(receipt))
    artifact=campaign/'train.pt';torch.save({'synthetic':True},artifact)
    attempts=[]
    def fake_execute(config,*,case_dir,artifacts,phase,charge_attempt,mode,resume,max_steps):
        assert max_steps==2 and phase=='qualification' and artifacts['synthetic']
        for step in (12,13):
            charge_attempt(phase,arm=mode,step=step)
            one_update(step)
        ckpt=Path(resume)/'checkpoints'/'last.pt'
        data=torch.load(ckpt,weights_only=True);data['global_step']=14;data['round8_runtime']['step']=14
        torch.save(data,ckpt)
        (Path(resume)/'logs'/'history.jsonl').write_text('clone only\n')
        return resume
    monkeypatch.setattr(profile,'execute_training',fake_execute)
    output=campaign/'profile.json'
    profile.profile_training(qualification_receipt=receipt_path,artifacts_path=artifact,
        output_path=output,campaign=campaign,case_dir=tmp_path,
        charge_attempt=lambda phase,*,arm,step:attempts.append((phase,arm,step)),synchronize=lambda:None)
    report=json.loads(output.read_text())
    assert report['optimizer_attempts']==4 and attempts==[('qualification','control',12),('qualification','control',13),('qualification','candidate',12),('qualification','candidate',13)]
    for arm in ('B8','C8'):
        parent=Path(segments[arm]['run']); entry=report['arms'][arm]; clone=Path(entry['clone_run'])
        assert (parent/'checkpoints'/'last.pt').read_bytes()==original_paths[arm]
        assert (parent/'logs'/'history.jsonl').read_text()=='original\n'
        assert (clone/'logs'/'history.jsonl').read_text()=='clone only\n'
        assert entry['start_step']==12 and entry['end_step']==14 and len(entry['timings'])==8
