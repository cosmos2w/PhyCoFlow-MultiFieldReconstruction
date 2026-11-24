"""Contract checks for the opt-in fresh-family batched backward."""
from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from phycoflow_reconstruction.config.validate import _validate_coherence_aggregation
from phycoflow_reconstruction.training.gradient_balance import _flat_gradient
from phycoflow_reconstruction.training.multitask import OuterAggregator, batched_family_gradients
from test_fidelity_budget_config_direction import _TwoParameterModel, _family_losses, _reference, _risk_vector
from phycoflow_reconstruction.training.fidelity_budget import ExternalPolicy
from scripts.qualification.car_round6_grouped_query_geometry import paired_replay_discrepancy, pooled_feasibility
from scripts.qualification import car_round6_batched_epoch_timing as timing
from scripts.qualification.car_round5_query_parameter_replay import plan
from phycoflow_reconstruction.coherence.query_sampling import SharedQuerySampler


def test_batched_rows_match_checkpointed_separate_rows_with_unused_parameter():
    p = torch.nn.Parameter(torch.randn(8, dtype=torch.float64))
    unused = torch.nn.Parameter(torch.randn(2, dtype=torch.float64))
    y = checkpoint(lambda value: value.sin().square(), p, use_reentrant=False)
    losses = [y.sum(), y.square().sum(), y.cos().sum()]
    expected = torch.stack([_flat_gradient(loss, [p, unused]) for loss in losses])
    torch.testing.assert_close(batched_family_gradients(losses, [p, unused]), expected)


def test_opt_in_matches_adam_actions_and_dual_state_across_two_updates():
    pytest.importorskip('torchjd')
    pytest.importorskip('cooper')
    models = [_TwoParameterModel(), _TwoParameterModel()]
    models[1].load_state_dict(deepcopy(models[0].state_dict()))
    policies = [ExternalPolicy({'method':'lagrangian','lagrangian':{'dual_lr':0.01}}, _reference(), 'cpu') for _ in models]
    for policy in policies:
        state = policy.state_dict()
        state['multipliers'] = [0.2, 0.3, 0.4]
        policy.load_state_dict(state)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=0.001) for model in models]
    for _ in range(2):
        for model, policy, optimizer, method in zip(models, policies, optimizers, ('separate', 'batched')):
            report = policy.update_with_coherence_aggregation(
                model, optimizer, _family_losses(model), _risk_vector(model), 1.0,
                OuterAggregator({'method':'torchjd_config', 'family_vjp':method}))
            assert report['coherence_aggregation']['family_vjp'] == method
        for left, right in zip(models[0].parameters(), models[1].parameters()):
            torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-12)
        assert policies[0]._multipliers() == pytest.approx(policies[1]._multipliers())
        for left, right in zip(optimizers[0].state.values(), optimizers[1].state.values()):
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(left[key], right[key], rtol=1e-10, atol=1e-12)


def test_vjp_selector_defaults_to_separate_and_rejects_unknown():
    _validate_coherence_aggregation({'coherence_aggregation':{'method':'torchjd_config'}})
    _validate_coherence_aggregation({'coherence_aggregation':{'method':'torchjd_config','family_vjp':'batched'}})
    assert OuterAggregator({'method':'torchjd_config'}).family_vjp == 'separate'
    with pytest.raises(ValueError, match='family_vjp'):
        _validate_coherence_aggregation({'coherence_aggregation':{'method':'torchjd_config','family_vjp':'other'}})


def test_grouped_query_replay_noise_and_pooled_feasibility():
    first = [{'sample_ids':['one'], 'seed':1, 'independent_normalized_mean_decrease':0.0},
             {'sample_ids':['two'], 'seed':1, 'independent_normalized_mean_decrease':0.0}]
    replay = [dict(first[0], independent_normalized_mean_decrease=1.0),
              dict(first[1], independent_normalized_mean_decrease=-1.0)]
    difference = paired_replay_discrepancy(first, replay)
    assert difference == {'signed_weighted_mean':0.0, 'absolute_weighted_mean':1.0}
    with pytest.raises(ValueError, match='sample IDs'):
        paired_replay_discrepancy(first, list(reversed(replay)))
    pooled, excess, norm = pooled_feasibility([[1.,-1.],[-1.,3.]], [1,1], [1.,2.], [1.,1.])
    assert pooled.tolist() == [0., 1.]
    assert excess.tolist() == [0., 2.] and norm == 2.0


def test_grouped_query_seed_pairing_and_independent_draws():
    sampler = SharedQuerySampler(('alpha','beta','gamma','delta','epsilon'), seed=19)
    runtime = SimpleNamespace(sampler=sampler)
    original_state = sampler.rng.getstate()
    for seed in (1,2,3,4):
        shared, groups = plan(runtime, seed, 0)
        independent, _ = plan(runtime, seed+1000, 0)
        assert groups.batches[0] == shared
        assert plan(runtime, seed, 0) == (shared, groups)
        assert independent != shared
        assert [(part.start,part.stop) for part,_ in groups.slices(64)] == [
            (0,8),(8,16),(16,24),(24,32),(32,40),(40,48),(48,56),(56,64)]
    assert sampler.rng.getstate() == original_state


def test_epoch_driver_uses_fresh_matched_configs_and_alternating_order(tmp_path, monkeypatch):
    import yaml
    config = {'runtime':{'seed':47}, 'optimization':{'steps_per_epoch':12,
        'multitask':{'coherence_aggregation':{'method':'torchjd_config'}}},
        'output':{'experiment_name':'original'}, 'science':{'weight':2}}
    source = tmp_path/'input.yaml'
    source.write_text(yaml.safe_dump(config))
    observed = []
    def trainer(cfg, *, case_dir, max_steps):
        observed.append((deepcopy(cfg), max_steps))
        path = tmp_path/f'run_{len(observed)}'
        (path/'logs').mkdir(parents=True)
        (path/'checkpoints').mkdir()
        (path/'artifacts').mkdir()
        (path/'artifacts'/'trainable_parameters.json').write_text(json.dumps({'names':['weight']}))
        torch.save({'global_step':0, 'model':{'weight':torch.tensor([1.])},
            'optimizer':{},'rng_state':{},'external_policy':{},'family_scales':{},'source_hashes':{},
            'scalable_runtime':{'step':0,'sampler':{},'group_samplers':[],'probe_sampler':{}}},
            path/'checkpoints'/'epoch_000.pt')
        if max_steps == 12:
            (path/'logs'/'epoch_timing.jsonl').write_text(json.dumps(
                {'attempted_updates':12,'accepted_updates':12,'wall_seconds':100.})+'\n')
            torch.save({'global_step':12, 'model':{'weight':torch.tensor([2.])}},path/'checkpoints'/'last.pt')
        return path
    monkeypatch.setenv('CAR_ROUND6_LEASE_TOKEN','fixture')
    monkeypatch.setattr(timing, 'run_post_training', trainer)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda _: 'fixture')
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda _: None)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda _: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda _: 123)
    monkeypatch.setattr(sys, 'argv', ['timing', '--config',str(source),'--case-dir',str(tmp_path),
        '--output',str(tmp_path/'receipt'),'--experiment-prefix','timing','--device','cuda:0'])
    timing.main()
    assert [row[0]['optimization']['multitask']['coherence_aggregation']['family_vjp'] for row in observed] == [
        'separate','batched','separate','batched','batched','separate']
    assert [row[1] for row in observed] == [1,1,12,12,12,12]
    assert all(row[0]['science'] == config['science'] and row[0]['runtime']['seed'] == 47 for row in observed)
    assert yaml.safe_load(source.read_text()) == config
    result = json.loads((tmp_path/'receipt'/'receipt.json').read_text())
    assert result['status'] == 'completed' and result['mean_wall_seconds'] == {'separate':100.,'batched':100.}
