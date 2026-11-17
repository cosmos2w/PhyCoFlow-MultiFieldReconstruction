"""Boundaries for genuine point scoring and held-out freeze enforcement."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

path = Path(__file__).resolve().parents[1] / 'scripts/posttraining/assess_car_round2_checkpoints.py'
spec = importlib.util.spec_from_file_location('round2_checkpoint_runner', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class Event:
    def masks(self, values):
        mask = values[..., 0] > 0
        return mask, mask, mask, mask


def test_event_point_score_is_genuine_single_prediction():
    member = torch.tensor([[[[1.], [-1.]]]])
    truth = torch.tensor([[[-1.], [-1.]]])
    score, areas, reference = mod.score_event(Event(), member, truth, torch.tensor([1., 3.]))
    torch.testing.assert_close(score, torch.tensor([.25]))
    assert areas.shape == (1, 1)
    torch.testing.assert_close(reference, torch.tensor([0.]))
    with pytest.raises(ValueError):
        mod.score_event(Event(), member.expand(2, -1, -1, -1), truth, torch.tensor([1., 3.]))


def test_sensor_mse_counts_only_observed_fields_and_preserves_actual_error():
    members = torch.tensor([[[[2., 8.], [5., 9.]]]])
    row = {'obs_indices':torch.tensor([[0, 1]]), 'obs_fields':torch.tensor([[0, 0]]),
           'obs_values':torch.tensor([[[1.], [3.]]])}
    mse, counts = mod.sensor_mse(members, row, 2)
    torch.testing.assert_close(mse, torch.tensor([2.5, 0.], dtype=torch.float64))
    assert counts.tolist() == [2, 0]


def test_closeout_budget_requires_all_frozen_mature_completions(tmp_path):
    import yaml
    configs = tmp_path / 'configs'; configs.mkdir()
    launches = tmp_path / 'operations/launches'; launches.mkdir(parents=True)
    completions = tmp_path / 'operations/completions'; completions.mkdir()
    arms = {}
    for index, arm in enumerate(mod.ARMS):
        cfg = {'optimization': {'epochs': 1000, 'steps_per_epoch': 12},
               'runtime': {'device': 'cuda:0'}, 'arm_identity': arm}
        config = configs / f'{arm}.yaml'; config.write_text(yaml.safe_dump(cfg))
        digest = mod.hash_file(config)
        arms[arm] = {'config': {'path': str(config), 'sha256': digest}}
        run = tmp_path / 'runs/formal' / arm / 'attempt'; run.mkdir(parents=True)
        cfg['runtime']['device'] = 'cuda:1'
        (run / 'resolved_config.yaml').write_text(yaml.safe_dump(cfg))
        (run / 'status.json').write_text(json.dumps({'status':'completed', 'global_step':12000}))
        token = f'token-{index}'
        (launches / f'{token}.json').write_text(json.dumps({'schema':'car_round2_launch_v1',
            'token':token, 'arm':arm, 'config_sha256':digest, 'pause_at_epoch':None}))
        (completions / f'{token}.json').write_text(json.dumps({'token':token, 'status':'completed',
            'global_step':12000, 'run_dir':str(run)}))
    frozen = {'schema':'car_round2_freeze_receipt_v1', 'manifest':{'schema':'car_round2_freeze_v1',
              'status':'frozen', 'arms':arms}}
    (configs / 'formal_freeze.json').write_text(json.dumps(frozen))
    assert mod.assessment_budget_limit(tmp_path, False) == 124 * 3600
    assert mod.assessment_budget_limit(tmp_path, True) == 130 * 3600
    last = completions / 'token-3.json'
    value = json.loads(last.read_text()); value['global_step'] = 11999; last.write_text(json.dumps(value))
    completed = mod.completed_frozen_mature_runs(tmp_path)
    assert all(completed[arm] for arm in mod.ARMS if arm.startswith('FFM_'))
    assert not completed[mod.ARMS[3]]
    assert mod.assessment_budget_limit(tmp_path, False) == 124 * 3600
    with pytest.raises(ValueError, match='all frozen mature arms'):
        mod.assessment_budget_limit(tmp_path, True)
    value['global_step'] = 12000; last.write_text(json.dumps(value))
    binding = Path(arms[mod.ARMS[0]]['config']['path']); binding.write_text('changed')
    with pytest.raises(ValueError, match='all frozen mature arms'):
        mod.assessment_budget_limit(tmp_path, True)


def test_assessment_budget_boundary_and_active_device():
    state = {'active': {}}
    for cap in (124 * 3600, 130 * 3600):
        mod.admit_assessment_budget(state, gpu_uuid='gpu', max_seconds=9, cap=cap, spent_seconds=cap-10)
        with pytest.raises(ValueError, match='remaining budget'):
            mod.admit_assessment_budget(state, gpu_uuid='gpu', max_seconds=10, cap=cap, spent_seconds=cap-10)
    state['active']['existing'] = {'gpu_uuid':'gpu'}
    with pytest.raises(ValueError, match='device leased'):
        mod.admit_assessment_budget(state, gpu_uuid='gpu', max_seconds=1, cap=130*3600, spent_seconds=0)


def test_panel_is_fixed_10_percent_and_never_test():
    panel = {'schema':'car_round2_panel_v1','split':'train','population':10,
             'sample_indices':[3], 'sample_ids':['trajectory:3'], 'sensor_count':256,
             'members':{'ffm':4,'senseiver':1}, 'native_points':40300,'observer_points':4096}
    assert mod.panel_contract(panel, 'train') == ['trajectory:3']
    with pytest.raises(ValueError): mod.panel_contract(panel, 'test')
    panel['sample_indices'] = [3, 3]
    with pytest.raises(ValueError): mod.panel_contract(panel, 'train')


def test_validation_freeze_detects_changed_nomination(tmp_path):
    formal = tmp_path / 'configs/formal_freeze.json'
    formal.parent.mkdir(); formal.write_text('{}')
    directory = tmp_path / 'assessment/train'; directory.mkdir(parents=True)
    arms = {}
    for arm in (name for name in mod.ARMS if name.startswith('FFM_')):
        nominations = {}
        for epoch in mod.EPOCHS:
            nomination = directory / f'{arm}_{epoch}.json'
            nomination.write_text(json.dumps({'value':epoch}))
            nominations[str(epoch)] = {'nomination':str(nomination),'sha256':mod.hash_file(nomination)}
        arms[arm] = nominations
    receipt = {'schema':'car_round2_assessment_freeze_v1','formal_freeze_sha256':mod.hash_file(formal),'arms':arms}
    receipt['sha256'] = mod.content_hash(receipt)
    (directory/'method_priority_ffm.json').write_text(json.dumps(receipt))
    mod.validation_freeze(tmp_path, 'ffm')
    first = next(iter(arms.values()))['0']['nomination']
    Path(first).write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        mod.validation_freeze(tmp_path, 'ffm')


def test_global_barrier_requires_both_families_and_unchanged_nominations(tmp_path):
    formal = tmp_path / 'configs/formal_freeze.json'; formal.parent.mkdir(); formal.write_text('{}')
    train = tmp_path / 'assessment/train'; train.mkdir(parents=True)
    def freeze(family):
        records = {}
        for arm in (a for a in mod.ARMS if a.lower().startswith(family)):
            records[arm] = {}
            for epoch in mod.EPOCHS:
                path = train / f'{arm}_{epoch}.json'; path.write_text('{}')
                records[arm][str(epoch)] = {'nomination':str(path), 'sha256':mod.hash_file(path)}
        payload = {'schema':'car_round2_assessment_freeze_v1', 'formal_freeze_sha256':mod.hash_file(formal),
                   'arms':records, 'method_priority':list(records)}
        payload['sha256'] = mod.content_hash(payload)
        (train / f'method_priority_{family}.json').write_text(json.dumps(payload))
        return records
    freeze('ffm')
    with pytest.raises(FileNotFoundError):
        mod.global_train_choice_barrier(tmp_path)
    senseiver = freeze('senseiver')
    mod.global_train_choice_barrier(tmp_path)
    Path(next(iter(senseiver.values()))['250']['nomination']).write_text('changed')
    with pytest.raises(ValueError, match='nomination changed'):
        mod.global_train_choice_barrier(tmp_path)


def test_decision_uses_frozen_priority_and_reports_crossing(tmp_path):
    import argparse

    formal = tmp_path/'configs/formal_freeze.json'
    formal.parent.mkdir()
    allocation = tmp_path/'allocation.json'
    allocation.write_text(json.dumps({'schema':'car_round2_allocation_disposition_v1', 'mode':'qualified_fixed'}))
    formal.write_text(json.dumps({'manifest':{'artifacts':{'allocation_disposition':{
        'path':str(allocation), 'sha256':mod.hash_file(allocation)}}}}))
    train = tmp_path/'assessment/train'; train.mkdir(parents=True)
    full, residual = [name for name in mod.ARMS if name.startswith('Senseiver_')]
    arms = {}
    for name in (full, residual):
        entries = {}
        for epoch in mod.EPOCHS:
            nomination = train/f'{name}_{epoch}.json'
            nomination.write_text(json.dumps({'child_checkpoint_sha256':f'{name}-{epoch}'}))
            entries[str(epoch)] = {'nomination':str(nomination), 'sha256':mod.hash_file(nomination)}
            directory = tmp_path/'assessment/validation'/name/f'epoch_{epoch:03d}'
            directory.mkdir(parents=True)
            base = {'alpha':0., 'scores':{'state':1.,'scale':1.,'shape':1.,'event':1.},
                    'fidelity':{'primary_pass':True, 'legacy_member_pass':True},'core':{'pass':True},'latency_seconds':1.}
            chosen = {'alpha':1., 'scores':{'state':1.,'scale':.98,'shape':.94,'event':1.},
                      'fidelity':{'primary_pass':True,'legacy_member_pass':True},'core':{'pass':True},
                      'latency_seconds':.9 if name == residual else 1.0}
            doc = {'schema':'car_round2_validation_assessment_v1','split':'validation','arm':name,'epoch':epoch,
                   'child_checkpoint_sha256':f'{name}-{epoch}', 'selected_alpha':1., 'rows':[base,chosen],
                   'normalizers':{'state':1.,'scale':1.,'shape':1.,'event':1.},'gpu_uuid':'same-device',
                   'source_latency_seconds':1., 'reserved_task':{'normalized_improvement':.2 if name == residual else .15}}
            (directory/'assessment.json').write_text(json.dumps(doc))
        arms[name] = entries
    freeze = {'schema':'car_round2_assessment_freeze_v1','formal_freeze_sha256':mod.hash_file(formal),
              'arms':arms, 'method_priority':[full,residual]}
    freeze['sha256'] = mod.content_hash(freeze)
    (train/'method_priority_senseiver.json').write_text(json.dumps(freeze))
    costs = {'schema':'car_round2_cost_reconciliation_v1','complete':True,'within_budget':True,
             'allocator_final':{'status':'measured', 'qualified':False},
             'arms':{full:{'serial_cost_seconds':100.,'checkpoint_cost_seconds':{'100':50.}},
                     residual:{'serial_cost_seconds':90.,'checkpoint_cost_seconds':{'100':30.}}}}
    receipt_path = tmp_path/'assessment/source-job.json'; receipt_path.write_text('{}')
    costs['source_receipts'] = [{'path':str(receipt_path),'sha256':mod.hash_file(receipt_path)}]
    costs['total_occupied_gpu_seconds'] = 190.
    cost_path = tmp_path/'assessment/costs.json'; cost_path.write_text(json.dumps(costs))
    output = tmp_path/'assessment/decision.json'
    mod.decide(argparse.Namespace(family='senseiver',costs=cost_path,output=output),tmp_path)
    result = json.loads(output.read_text())
    assert result['frozen_priority'] == [full,residual]
    assert result['deployment_disposition']['operational_function'] == full
    assert result['learning_efficiency_route'] is True
    assert result['crossings'][residual]['crossing_epoch'] == 100
    assert result['allocation_decision']['training_mode'] == 'qualified_fixed'
    assert result['allocation_decision']['status'] == 'reject'
    assert result['deployment_disposition']['operational_function'] == full
    costs['complete'] = False
    costs['allocator_final'] = {'status':'incomplete_campaign_costs', 'qualified':False}
    cost_path.write_text(json.dumps(costs))
    output = tmp_path/'assessment/decision_incomplete.json'
    mod.decide(argparse.Namespace(family='senseiver',costs=cost_path,output=output),tmp_path)
    result = json.loads(output.read_text())
    assert result['allocation_decision']['status'] == 'unavailable'
    assert result['quality_route'] is None and result['learning_efficiency_route'] is None
    assert result['deployment_comparison_route'] is None
    assert result['deployment_disposition']['decisions'][full]['task_useful'] is True


def test_final_allocation_tracks_frozen_mode_and_actual_recheck(tmp_path):
    allocation = tmp_path/'allocation.json'
    allocation.write_text(json.dumps({'schema':'car_round2_allocation_disposition_v1', 'mode':'qualified_fixed'}))
    formal = {'manifest':{'artifacts':{'allocation_disposition':{'path':str(allocation), 'sha256':mod.hash_file(allocation)}}}}
    for qualified, expected in ((True, 'adopt'), (False, 'reject')):
        result = mod.final_allocation_disposition(formal, {'complete':True, 'allocator_final':{'status':'measured', 'qualified':qualified}})
        assert result['training_mode'] == 'qualified_fixed' and result['status'] == expected
    assert mod.final_allocation_disposition(formal, {'allocator_final':{'status':'incomplete'}})['status'] == 'unavailable'
    with pytest.raises(ValueError, match='contradicts'):
        mod.final_allocation_disposition(formal, {'allocator_final':{'status':'uniform_fallback', 'qualified':False}})


def test_fair_family_decomposition_uses_each_member_without_duplication():
    from test_frozen_coherence_observer import make_observer

    from phycoflow_reconstruction.evaluation.frozen_coherence import FrozenCoherenceObserver

    conditional, draws, families, artifact = make_observer('conditional')
    members = torch.cat((draws, draws + .173), dim=0)
    paired = FrozenCoherenceObserver(artifact, families, expected_provenance=artifact['provenance'],
        reference=conditional.reference, coordinates=conditional.coordinates,
        sample_ids=list(conditional.receipt['sample_ids']), member_weights=conditional.member_weights,
        reference_weights=conditional.reference_weights, mode='deterministic', query_seed=3, query_batches=2)
    with torch.no_grad():
        fair = conditional.score_tensors(members)
        attractions = [paired.score_tensors(member.unsqueeze(0)) for member in members]
    for key in ('state','scale','shape'):
        attraction = torch.stack([item[key] for item in attractions]).mean(0)
        correction = attraction-fair[key]
        assert torch.isfinite(correction).all()
        assert (correction >= -1e-6).all()


def test_source_normalizer_is_frozen_after_nominations_and_applied_before_validation(tmp_path, monkeypatch):
    import argparse

    from phycoflow_reconstruction.evaluation import round2_task

    configs = tmp_path / 'configs/panels'
    configs.mkdir(parents=True)
    train = tmp_path / 'assessment/train'
    train.mkdir(parents=True)
    bank = tmp_path / 'bank'
    bank.mkdir()
    panel = {'schema':'car_round2_panel_v1', 'split':'train', 'population':10,
             'sample_indices':[0], 'sample_ids':['sample'], 'sensor_count':256,
             'members':{'ffm':4, 'senseiver':1}, 'native_points':40300, 'observer_points':4096}
    (configs/'train.json').write_text(json.dumps(panel))
    torch.save({'sample_id':'sample', 'native_masses':torch.ones(1),
                'normalizer_offset':torch.zeros(1), 'normalizer_scale':torch.ones(1),
                'members_model':torch.tensor([[[[3.]]]]), 'truth_model':torch.tensor([[[1.]]])}, bank/'0000.pt')
    task = tmp_path/'task.json'
    task.write_text('{}')
    receipt = {'schema':'car_round2_native_bank_v1', 'status':'completed', 'split':'train',
               'kind':'source', 'family':'senseiver', 'source_sha256':'source',
               'panel_sha256':mod.hash_file(configs/'train.json'), 'sample_ids':['sample'],
               'files_sha256':[mod.hash_file(bank/'0000.pt')], 'field_order':['field']}
    (bank/'receipt.json').write_text(json.dumps(receipt))
    contract = {'schema':'car_round2_assessment_contract_v2', 'source_checkpoint_sha256':'source',
                'source_bank_sha256':mod.hash_file(bank/'receipt.json'), 'reserved_task_sha256':mod.hash_file(task),
                'reserved_task_normalizer':10., 'reserved_task_normalization_rule':{
                    'primary':'final_train_source_task_score', 'fallback':'train_truth_absolute_covariance',
                    'near_zero_threshold':1e-11}}
    contract_path = tmp_path/'contract.json'
    contract_path.write_text(json.dumps(contract))
    formal = {'manifest':{'artifacts':{'reserved_task':{'path':str(task), 'sha256':mod.hash_file(task)},
              'assessment_contract_senseiver':{'path':str(contract_path),'sha256':mod.hash_file(contract_path)}}}}
    formal_path = tmp_path/'configs/formal_freeze.json'
    formal_path.write_text(json.dumps(formal))
    arms = {}
    for arm in (name for name in mod.ARMS if name.startswith('Senseiver_')):
        entries = {}
        for epoch in mod.EPOCHS:
            path = train/f'{arm}_{epoch}.json'
            path.write_text(json.dumps({'source_bank_sha256':mod.hash_file(bank/'receipt.json'),
                                        'contract_sha256':mod.hash_file(contract_path)}))
            entries[str(epoch)] = {'nomination':str(path), 'sha256':mod.hash_file(path)}
        arms[arm] = entries
    freeze = {'schema':'car_round2_assessment_freeze_v1','formal_freeze_sha256':mod.hash_file(formal_path),
              'arms':arms, 'method_priority':list(arms)}
    freeze['sha256'] = mod.content_hash(freeze)
    (train/'method_priority_senseiver.json').write_text(json.dumps(freeze))
    ffm_arms = {}
    for arm in (name for name in mod.ARMS if name.startswith('FFM_')):
        ffm_arms[arm] = {}
        for epoch in mod.EPOCHS:
            path = train / f'{arm}_{epoch}.json'; path.write_text('{}')
            ffm_arms[arm][str(epoch)] = {'nomination':str(path), 'sha256':mod.hash_file(path)}
    ffm_freeze = {'schema':'car_round2_assessment_freeze_v1', 'formal_freeze_sha256':mod.hash_file(formal_path),
                  'arms':ffm_arms, 'method_priority':list(ffm_arms)}
    ffm_freeze['sha256'] = mod.content_hash(ffm_freeze)
    (train/'method_priority_ffm.json').write_text(json.dumps(ffm_freeze))
    monkeypatch.setattr(round2_task, 'regional_covariance', lambda values, mass, manifest, fields: values[..., 0, 0])
    args = argparse.Namespace(family='senseiver', source_bank=bank, reserved_task=task)
    mod.freeze_reserved_normalizer(args, tmp_path)
    result = mod.validate_reserved_freeze(tmp_path, 'senseiver')
    assert result['source_score'] == 2.
    assert result['normalizer'] == 2.
    assert result['used_truth_fallback'] is False
    result['normalizer'] = 10.
    result['sha256'] = mod.content_hash({k:v for k,v in result.items() if k != 'sha256'})
    (train/'reserved_normalizer_senseiver.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='fallback rule'):
        mod.validate_reserved_freeze(tmp_path, 'senseiver')
