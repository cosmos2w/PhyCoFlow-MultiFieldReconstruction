from copy import deepcopy
from types import SimpleNamespace

import pytest

from phycoflow_reconstruction.coherence.query_sampling import ConditionQueryGroups, HierarchicalQuerySampler
from scripts.qualification.car_round5_panel_replay import fit_source_units, independent_query


def test_independent_query_preserves_live_streams_and_shared_law():
    samplers = [HierarchicalQuerySampler(['a', 'b', 'c'], seed=20 + i,
        hierarchy_weight=0, fixed_probabilities=(.2, .3, .5), profile_sha256='a'*64) for i in range(8)]
    for sampler in samplers:
        sampler.draw()
    before = [deepcopy(s.state_dict()) for s in samplers]
    grouped = SimpleNamespace(sampler=samplers[0], group_samplers=samplers[1:])
    batch = independent_query(grouped, 5)
    assert isinstance(batch, ConditionQueryGroups)
    assert len(batch.batches) == 8
    assert all(d.probability == pytest.approx((.2, .3, .5)[d.index])
               for q in batch.batches for d in q.pair_queries)
    assert [s.state_dict() for s in samplers] == before
    shared = SimpleNamespace(sampler=samplers[0], group_samplers=[])
    first = independent_query(shared, 5)
    assert first == batch.batches[0]
    assert independent_query(grouped, 5) == batch
    assert independent_query(grouped, 6) != batch


def test_grouped_configuration_cannot_fit_source_units():
    runtime = SimpleNamespace(condition_query_groups=8)
    with pytest.raises(ValueError, match='original shared query law'):
        fit_source_units(None, (None, None, None, None, runtime, None))


def test_relocated_replay_requires_bound_validated_original_and_same_bank(tmp_path, monkeypatch):
    import json
    import torch
    import yaml
    import scripts.qualification.car_round5_panel_replay as replay

    original_dir = tmp_path / 'original'
    (original_dir / 'artifacts').mkdir(parents=True)
    bank_path = tmp_path / 'bank.json'
    bank = {'sha256':'b'*64, 'queries':[1, 2]}
    bank_path.write_text(json.dumps(bank))
    original = {'coherence':{'scalable_anchored_coherence':{'adaptive':{'bank_artifact':
        {'path':str(bank_path), 'sha256':replay.digest(bank_path)}}}}, 'rollout':{'steps':4}}
    reference = original_dir / 'resolved_config.yaml'
    reference.write_text(yaml.safe_dump(original))
    artifact_path = original_dir / 'artifacts/scalable_coherence.pt'
    torch.save({'provenance':{'identity':'source'}, 'adaptive_bank':bank, 'sha256':'valid'}, artifact_path)
    grouped = deepcopy(original)
    grouped['coherence']['scalable_anchored_coherence']['condition_query_groups'] = 8
    relocated = tmp_path / 'grouped.yaml'
    relocated.write_text(yaml.safe_dump(grouped))
    calls = []
    monkeypatch.setattr(replay, 'validate_scalable_artifacts', lambda obj, identity: calls.append((obj, identity)))
    kwargs = {'reference_config':reference, 'reference_config_sha256':replay.digest(reference),
              'replay_artifact_sha256':replay.digest(artifact_path)}
    core = replay.load_replay_core(relocated, grouped, **kwargs)
    assert core['adaptive_bank'] == bank and len(calls) == 1
    with pytest.raises(ValueError, match='hash-bound'):
        replay.load_replay_core(relocated, grouped)
    with pytest.raises(ValueError, match='artifact hash'):
        replay.load_replay_core(relocated, grouped, **{**kwargs, 'replay_artifact_sha256':'f'*64})
    changed = deepcopy(grouped)
    changed['rollout']['steps'] = 2
    with pytest.raises(ValueError, match='beyond query grouping'):
        replay.load_replay_core(relocated, changed, **kwargs)
    torch.save({'provenance':{'identity':'source'}, 'adaptive_bank':{'other':1}, 'sha256':'valid'}, artifact_path)
    with pytest.raises(ValueError, match='differs from selected bank'):
        replay.load_replay_core(relocated, grouped, **{**kwargs, 'replay_artifact_sha256':replay.digest(artifact_path)})


def test_bounded_correction_selects_actual_optimizer_and_initial_lr(tmp_path):
    import json
    import torch
    from scripts.qualification.car_round5_panel_replay import correction_settings, replay_optimizer, replay_sample_ids, digest

    units=tmp_path/'units.json'; units.write_text('{"native": 1}')
    defaults=correction_settings(None, candidate_only=False, units_path=units)
    assert defaults['learning_rates'] == (5e-6, 5e-5)
    param=torch.nn.Parameter(torch.tensor([1.]))
    default_optimizer=replay_optimizer([param], 'Direct_Scalar_AdamW', defaults['learning_rates'][0], defaults)
    assert default_optimizer.param_groups[0]['betas'] == (.9, .999)
    evidence=tmp_path/'startup.json'
    receipt={'schema':'car_round5_startup_qualification_v1','demonstrated_failure':True,
             'correction':'halve LR and use beta1=0', 'units_sha256':digest(units),
             'finite_step_overshoot_demonstrated':True, 'ascending_proposal_fraction':.11}
    evidence.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='candidate-only'):
        correction_settings(evidence, candidate_only=False, units_path=units)
    corrected=correction_settings(evidence, candidate_only=True, units_path=units)
    assert corrected['learning_rates'] == (2.5e-6, 2.5e-5)
    optimizer=replay_optimizer([param], 'Direct_Scalar_AdamW', corrected['learning_rates'][1], corrected)
    assert optimizer.param_groups[0]['betas'] == (0., .999)
    assert optimizer.param_groups[0]['weight_decay'] == 1e-6
    assert corrected['correction_evidence']['sha256'] == digest(evidence)
    assert replay_sample_ids([{'lr':2.5e-6,'sample_ids':['first']}, {'lr':2.5e-5,'sample_ids':['second']}], corrected['learning_rates'][0]) == ['first']
    for overshoot, fraction, expected_rates, expected_beta in [
        (True, .1, (2.5e-6, 2.5e-5), .9),
        (False, .10001, (5e-6, 5e-5), 0.),
    ]:
        evidence.write_text(json.dumps({**receipt, 'finite_step_overshoot_demonstrated':overshoot,'ascending_proposal_fraction':fraction}))
        selected=correction_settings(evidence,candidate_only=True,units_path=units)
        assert selected['learning_rates']==expected_rates and selected['beta1']==expected_beta
    evidence.write_text(json.dumps({**receipt, 'finite_step_overshoot_demonstrated':False,'ascending_proposal_fraction':.1}))
    with pytest.raises(ValueError, match='overshoot'):
        correction_settings(evidence,candidate_only=True,units_path=units)
    evidence.write_text(json.dumps({**receipt, 'units_sha256':'wrong'}))
    with pytest.raises(ValueError, match='unit-bound'):
        correction_settings(evidence,candidate_only=True,units_path=units)
