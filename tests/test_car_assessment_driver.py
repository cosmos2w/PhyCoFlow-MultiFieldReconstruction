import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('car_assessment_driver', Path('scripts/posttraining/assess_car_checkpoints.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_panel_requires_fixed_t256_and_j4():
    panel = {'split': 'train', 'population': 20, 'sample_ids': ['x', 'y'], 'sensor_count': 256, 'members': 4}
    assert mod.panel_contract(panel, 'train') == ['x', 'y']
    panel['sensor_count'] = 255
    with pytest.raises(ValueError): mod.panel_contract(panel, 'train')
    assert len({mod.member_seed(17, j) for j in range(4)}) == 4
    assert mod.member_seed(17, 1) != mod.member_seed(18, 0)


def _freeze_fixture(tmp_path):
    campaign = tmp_path / 'campaign'
    (campaign / 'assessment').mkdir(parents=True)
    event = {'schema':'car_event_task_redesign_v1','split':'train'}
    event['sha256'] = mod.content_hash(event)
    (campaign / 'assessment/native_event_redesigned.json').write_text(json.dumps(event))
    formal = {'schema':'car_freeze_receipt_v1','manifest_sha256':'manifest',
              'manifest': {'status':'frozen','train_qualification_sha256':'q','event_task_sha256':event['sha256'],
                           'driver_sha256':'driver','assessment_driver_sha256':mod.file_hash(Path(mod.__file__)), 'code_root':'code',
                           'arms':{'FFM_B1':{'config_sha256':'b1'},'FFM_A1':{'config_sha256':'a1'}}}}
    (campaign / 'formal_freeze.json').write_text(json.dumps(formal))
    (campaign/'launches').mkdir()
    (campaign/'completions').mkdir()
    arms = {}
    for arm in ('FFM_B1','FFM_A1'):
        run, alphas = tmp_path/arm, tmp_path/(arm+'_alpha')
        (run / 'checkpoints').mkdir(parents=True)
        alphas.mkdir()
        (run/'status.json').write_text(json.dumps({'status':'completed','global_step':13000,'configured_steps':13000}))
        for epoch in mod.EPOCHS:
            checkpoint = run/'checkpoints'/f'epoch_{epoch:03d}.pt'
            checkpoint.write_bytes(f'{arm}-{epoch}'.encode())
            alpha = {'schema':'car_streamed_alpha_v1','split':'train','epoch':epoch,
                     'child_checkpoint_sha256':mod.file_hash(checkpoint), 'selected_alpha':.5, 'assessment_driver_sha256':mod.file_hash(Path(mod.__file__)),
                     'panel_sha256':'panel', 'source_checkpoint_sha256':'source', 'coupling_sha256':'stream',
                     'core_identity':{'hash':'core'}, 'sample_ids':['sample']}
            (alphas/f'epoch_{epoch:03d}.json').write_text(json.dumps({**alpha,'sha256':mod.content_hash(alpha)}))
        token = arm
        launch = {'schema':'car_formal_launch_v1','token':token,'arm':arm,
                  'config_sha256':'b1' if arm == 'FFM_B1' else 'a1',
                  'driver_sha256':'driver','code_root':'code'}
        completion = {'token':token,'status':'completed','global_step':13000,'run_dir':str(run)}
        (campaign/'launches'/f'{token}.json').write_text(json.dumps(launch))
        (campaign/'completions'/f'{token}.json').write_text(json.dumps(completion))
        arms[arm] = (run,alphas)
    return campaign, arms


def test_validation_requires_two_completed_arms_all_train_nominations(tmp_path):
    campaign, arms = _freeze_fixture(tmp_path)
    chosen = arms['FFM_B1'][1] / 'epoch_100.json'
    with pytest.raises(ValueError, match='complete two-arm'):
        mod.validation_gate(campaign, chosen)
    with pytest.raises(ValueError, match='both formal arms'):
        mod.build_assessment_freeze(campaign, {'FFM_B1':arms['FFM_B1']})
    missing = arms['FFM_A1'][1] / 'epoch_750.json'
    saved = missing.read_bytes()
    missing.unlink()
    with pytest.raises(FileNotFoundError):
        mod.build_assessment_freeze(campaign, arms)
    missing.write_bytes(saved)
    payload = mod.build_assessment_freeze(campaign, arms)
    freeze = {**payload,'sha256':mod.content_hash(payload)}
    (campaign/'assessment/assessment_freeze.json').write_text(json.dumps(freeze))
    assert mod.validation_gate(campaign, chosen)['selected_alpha'] == .5
    missing.write_text('{}')
    with pytest.raises(ValueError, match='evidence changed'):
        mod.validation_gate(campaign, chosen)


def test_validation_rejects_incomplete_other_arm(tmp_path):
    campaign, arms = _freeze_fixture(tmp_path)
    (arms['FFM_A1'][0]/'status.json').write_text(json.dumps({'status':'running','global_step':12000,'configured_steps':13000}))
    with pytest.raises(ValueError, match='both arms must complete'):
        mod.build_assessment_freeze(campaign, arms)


@pytest.mark.parametrize('mode', ['mature', 'edm-qualification'])
def test_train_score_streams_and_writes_bound_blend(tmp_path, monkeypatch, mode):
    import argparse

    import torch
    import yaml

    import phycoflow_reconstruction.evaluation.frozen_coherence as frozen
    from phycoflow_reconstruction import coherence
    from phycoflow_reconstruction.data import factory
    from phycoflow_reconstruction.evaluation.adaptive_campaign import digest
    class Family:
        def load_state_artifact(self, artifact): pass
    class Dataset:
        field_names = ('T',)
        data_spec = object()
        class normalizer:
            @staticmethod
            def digest(): return 'norm'
        def close(self): pass
    class Observer:
        def __init__(self, *args, **kwargs): pass
        def score_tensors(self, members):
            scalar = members.square().mean() + 1
            return {'state':scalar, 'scale':scalar, 'shape':scalar}
    monkeypatch.setattr(coherence, 'build_enabled_families', lambda *args: {'global_distribution': Family()})
    monkeypatch.setattr(factory, 'open_field_dataset', lambda *args, **kwargs: Dataset())
    monkeypatch.setattr(frozen, 'FrozenCoherenceObserver', Observer)
    campaign = tmp_path / 'campaign'
    (campaign / 'panels').mkdir(parents=True)
    panel = {'split':'train', 'population':10, 'sample_ids':['sample'], 'sensor_count':256, 'members':4}
    panel_path = campaign / 'panels/train.json'
    panel_path.write_text(json.dumps(panel))
    core = tmp_path / 'core'
    (core / 'artifacts').mkdir(parents=True)
    (core / 'resolved_config.yaml').write_text(yaml.safe_dump({'dataset':{}, 'coherence':{'compute_budget':{'point_count':4096 if mode == 'mature' else 40300, 'query_seed':1}}}))
    torch.save({}, core / 'artifacts/global_distribution_family.pt')
    torch.save({'provenance':{'source_checkpoint_sha256':'source', 'field_order':['T'], 'normalizer_sha256':'norm'}}, core / 'artifacts/scalable_coherence.pt')
    truth = torch.zeros((1,40300,1))
    row = {'sample_id':'sample', 'truth_model':truth, 'coordinates_model':torch.zeros((1,40300,2)),
           'coordinates_raw':torch.zeros((40300,2)), 'native_masses':torch.ones(40300),
           'obs_indices':torch.tensor([[0]]), 'obs_values':torch.zeros((1,1,1)), 'obs_fields':torch.tensor([[0]]),
           'normalizer_offset':torch.tensor([0.]), 'normalizer_scale':torch.tensor([1.]), 'member_seeds':[1,2,3,4]}
    for label in ('source', 'child'):
        bank = tmp_path / label
        bank.mkdir()
        values = torch.ones((4,1,40300,1)) if label == 'source' else torch.zeros((4,1,40300,1))
        values[:,:,0,:] = 0
        torch.save({**row, 'members_model':values}, bank / '0000.pt')
        payload = {'split':'train', 'model':'ffm' if mode == 'mature' else 'edm', 'arm':'FFM_B1' if mode == 'mature' else 'EDM_qualification', 'source_sha256':'source', 'source_weight_view':'view',
                   'panel_sha256':digest(panel_path), 'dataset_fingerprint':'data', 'normalizer_sha256':'norm',
                   'assessment_driver_sha256':mod.file_hash(Path(mod.__file__)),
                   'field_order':['T'], 'sample_ids':['sample'],
                   'coupling_sha256':'coupling', 'support':40300, 'members':4,
                   'epoch':None if label=='source' else (100 if mode == 'mature' else 9), 'checkpoint_sha256':label,
                   'files_sha256':[digest(bank / '0000.pt')]}
        (bank / 'receipt.json').write_text(json.dumps({**payload, 'sha256':mod.content_hash(payload)}))
    args = argparse.Namespace(split='train', mode=mode, campaign=campaign, alpha_receipt=None,
        source_bank=tmp_path/'source', child_bank=tmp_path/'child', core_run=core,
        source_seconds=1., child_seconds=1., sensor_tolerance=0.,
        output=tmp_path/'alpha.json', blend_bank=tmp_path/'blend')
    mod.score(args)
    receipt = mod.checked_json(args.output)
    if mode == 'mature':
        assert receipt['selected_alpha'] == 1.
        assert receipt['core_support'] == 'fixed Q4096'
        assert mod.checked_json(args.blend_bank / 'receipt.json')['alpha'] == 1.
    else:
        assert receipt['schema'] == 'car_edm_qualification_assessment_v1'
        assert receipt['core_support'] == 'full native 40300'
        assert 'selected_alpha' not in receipt and not args.blend_bank.exists()


def test_edm_qualification_epoch_and_native_support_contract():
    config = {'model':{'name':'diffusion_pde'}, 'optimization':{'epochs':12,'steps_per_epoch':1},
              'rollout':{'steps':32,'solver':'heun'}, 'observation_consistency':{'mode':'native'},
              'coherence':{'compute_budget':{'point_count':40300}}}
    out = mod.checkpoint_contract('edm','EDM_qualification',9,config,9)
    assert out['steps_per_epoch'] == 1 and 'no efficacy' in out['scope']
    assert mod.observer_support('edm-qualification','edm',config) == 40300
    with pytest.raises(ValueError, match='contract'):
        mod.checkpoint_contract('edm','EDM_qualification',9,config,117)
    with pytest.raises(ValueError, match='contract'):
        mod.checkpoint_contract('edm','FFM_B1',9,config,9)
    with pytest.raises(ValueError, match='support'):
        mod.observer_support('mature','edm',config)
    config['coherence']['compute_budget']['point_count'] = 4096
    with pytest.raises(ValueError, match='support'):
        mod.observer_support('edm-qualification','edm',config)
    with pytest.raises(ValueError, match='contract'):
        mod.checkpoint_contract('edm','EDM_qualification',9,config,9)


def test_frozen_code_and_case_roots_can_differ(tmp_path):
    import os
    import subprocess
    import sys
    code_root = Path('scripts/posttraining/assess_car_checkpoints.py').resolve()
    env = {**os.environ, 'CAR_REPO_ROOT':str(tmp_path / 'repo'), 'CAR_CODE_ROOT':str(tmp_path / 'capsule')}
    program = ('import importlib.util; s=importlib.util.spec_from_file_location("m", ' + repr(str(code_root)) + '); '
               'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(m.ROOT); print(m.CODE_ROOT); print(m.CAMPAIGN)')
    lines = subprocess.check_output([sys.executable, '-c', program], env=env, text=True).splitlines()
    assert lines == [str(tmp_path/'repo'), str(tmp_path/'capsule'),
                     str(tmp_path/'repo/cases/turbulent_combustion/runs/coherence_adaptive_repair/round_1')]


def test_assessment_freeze_rejects_unrelated_completed_run(tmp_path):
    campaign, arms = _freeze_fixture(tmp_path)
    completion = campaign/'completions/FFM_A1.json'
    value = json.loads(completion.read_text())
    value['run_dir'] = str(tmp_path/'unrelated_completed_run')
    completion.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='matching frozen formal launch'):
        mod.build_assessment_freeze(campaign, arms)
