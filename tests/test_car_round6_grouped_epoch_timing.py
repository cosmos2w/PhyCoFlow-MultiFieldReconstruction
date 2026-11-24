"""Input coupling and initial-state boundaries for query cost comparisons."""
from copy import deepcopy
import json
import sys

import pytest
import torch
from scripts.qualification.car_round6_grouped_epoch_timing import common_initial_state_digest, completed_epoch_record, read_coupling
from scripts.qualification import car_round6_grouped_epoch_timing as driver


def test_common_state_fingerprint_excludes_only_query_group_metadata(tmp_path):
    state={'global_step':0,'config_sha256':'shared-config','model':{'weight':torch.tensor([1.])},
        'optimizer':{'step':0},'rng_state':{'torch_cpu':torch.tensor([1],dtype=torch.uint8)},
        'external_policy':{'multipliers':[0.]},'training_monitor':{'step':0},
        'scalable_runtime':{'condition_query_groups':1,'group_samplers':[],
                            'sampler':{'seed':1},'probe_sampler':{'seed':2}}}
    changed=deepcopy(state)
    changed['config_sha256']='grouped-config'
    changed['scalable_runtime']['condition_query_groups']=8
    changed['scalable_runtime']['group_samplers']=[{'seed':n} for n in range(7)]
    first=tmp_path/'first.pt';second=tmp_path/'second.pt'
    torch.save(state,first);torch.save(changed,second)
    assert common_initial_state_digest(first)==common_initial_state_digest(second)
    changed['scalable_runtime']['sampler']['seed']=3
    torch.save(changed,second)
    assert common_initial_state_digest(first)!=common_initial_state_digest(second)


def test_coupling_read_requires_both_members_and_accepted_actions(tmp_path):
    (tmp_path/'metrics').mkdir()
    path=tmp_path/'metrics'/'training_coupling.jsonl'
    row={'step':1,'sample_ids':['sample'], 'observation_sha256':'obs','query_sha256':'grid',
         'member_seeds':[5,6],'first_member_generator_state_sha256':'rng','accepted':True}
    path.write_text(json.dumps(row)+'\n')
    assert read_coupling(tmp_path,1)==[row]
    row['member_seeds']=[5]
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='two member'):
        read_coupling(tmp_path,1)


def test_cost_comparison_excludes_explicit_partial_epoch_duration():
    legacy={'attempted_updates':12,'accepted_updates':12,'wall_seconds':100.}
    assert completed_epoch_record([legacy],12)==legacy
    assert completed_epoch_record([{**legacy,'complete_epoch_duration':True}],12)['wall_seconds']==100.
    with pytest.raises(RuntimeError,match='complete epoch duration'):
        completed_epoch_record([{**legacy,'complete_epoch_duration':False}],12)


def test_complete_epoch_trials_preserve_fixed_vjp_and_verify_coupling(tmp_path,monkeypatch):
    import yaml
    config={'runtime':{'seed':47},'optimization':{'steps_per_epoch':12,
        'multitask':{'coherence_aggregation':{'method':'torchjd_config','family_vjp':'batched'}}},
        'coherence':{'scalable_anchored_coherence':{}},'output':{'experiment_name':'original'}}
    source=tmp_path/'config.yaml';source.write_text(yaml.safe_dump(config)); calls=[]
    def train(cfg,*,case_dir,max_steps):
        calls.append((deepcopy(cfg),max_steps))
        run=tmp_path/f'run_{len(calls)}'
        for part in ('metrics','logs','checkpoints','artifacts'):(run/part).mkdir(parents=True)
        group=cfg['coherence']['scalable_anchored_coherence']['condition_query_groups']
        torch.save({'global_step':0,'config_sha256':f'config{group}','model':{'weight':torch.tensor([1.])},
            'scalable_runtime':{'condition_query_groups':group,'group_samplers':list(range(group-1)),
                                'sampler':{'seed':1}}},run/'checkpoints'/'epoch_000.pt')
        row={'sample_ids':['one'],'observation_sha256':'obs','query_sha256':'grid','member_seeds':[5,6],
             'first_member_generator_state_sha256':'rng','accepted':True}
        (run/'metrics'/'training_coupling.jsonl').write_text(''.join(json.dumps(dict(row,step=i+1))+'\n' for i in range(max_steps)))
        (run/'artifacts'/'trainable_parameters.json').write_text(json.dumps({'names':['weight']}))
        if max_steps==12:
            (run/'logs'/'epoch_timing.jsonl').write_text(json.dumps({'attempted_updates':12,'accepted_updates':12,'wall_seconds':100.})+'\n')
            torch.save({'global_step':12,'model':{'weight':torch.tensor([2.])}},run/'checkpoints'/'last.pt')
        return run
    monkeypatch.setenv('CAR_ROUND6_LEASE_TOKEN','fixture')
    monkeypatch.setattr(driver,'run_post_training',train)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda _:'fixture')
    monkeypatch.setattr(torch.cuda,'synchronize',lambda _:None)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda _:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda _:123)
    monkeypatch.setattr(sys,'argv',['driver','--config',str(source),'--case-dir',str(tmp_path),
        '--output',str(tmp_path/'out'),'--experiment-prefix','timing','--device','cuda:0'])
    driver.main()
    assert [c['coherence']['scalable_anchored_coherence']['condition_query_groups'] for c,_ in calls]==[1,8,1,8,8,1]
    assert all(c['optimization']['multitask']['coherence_aggregation']['family_vjp']=='batched' for c,_ in calls)
    result=json.loads((tmp_path/'out'/'receipt.json').read_text())
    assert result['status']=='completed' and result['matching_training_coupling'] is True
    assert yaml.safe_load(source.read_text())==config
