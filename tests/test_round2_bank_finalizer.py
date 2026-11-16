import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location(
    "car_round2_bank_finalizer", Path(__file__).resolve().parents[1] / "scripts/analysis/finalize_car_round2_bank.py"
)
finalizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finalizer)


def test_offline_perturbation_is_bounded_and_null_at_observed_field_site():
    shard = {"source": torch.zeros(2, 1, 3, 2), "reference": torch.full((1, 3, 2), 100.),
             "query_indices": torch.tensor([[2, 3, 4]]), "obs_indices": torch.tensor([[3, 4]]),
             "obs_fields": torch.tensor([[0, 1]]), "obs_valid_mask": torch.tensor([[True, False]])}
    changed = finalizer.null_sensor_perturbation(shard, [2., 4.])
    assert torch.equal(changed[:, 0, 1, 0], torch.zeros(2))
    torch.testing.assert_close(changed[:, 0, 1, 1], torch.full((2,), .04))
    torch.testing.assert_close(changed[:, 0, 2, 1], torch.full((2,), .04))
    assert changed.abs().amax() <= .04


def test_useful_source_specific_starter_precedes_higher_ranked_replacement():
    from phycoflow_reconstruction.coherence.round2_qualification import select_task_conditioned_bank

    candidates = [{"id": f"group-{i}", "kind": "state.group_projection",
                   "fields": ["a", "b"], "signs": [1, 1 if i % 2 else -1]} for i in range(5)]
    starter = candidates[0]
    bank = {"selection_audit": [{"candidate": row} for row in candidates], "selected": [starter],
            "field_order": ["a", "b"], "field_scales": [1., 1.], "floor": .01,
            "target_weights": {"state.group_projection": [], "scale.third_order": []},
            "block_units": {"state.group_projection": 1., "scale.third_order": 1.}}
    evidence = {}
    for index, candidate in enumerate(candidates):
        evidence[candidate["id"]] = {"discrepancy": 1., "replay_error": 0., "repeatability": 1.,
                                     "development_response": 1., "full_model_response": 1. if index == 0 else 2.,
                                     "residual_head_response": 1. if index == 0 else 2.,
                                     "complete_seconds": 1., "saved_update_fraction": 0.}
    result = select_task_conditioned_bank(bank, evidence, repeatability_threshold=.1, response_threshold=.01)
    assert starter in result["selected"]
    assert len(result["selected"]) == 4
    assert "starter" in result["selection_rule"]["retention_preference"]


def test_prefit_summary_schema_requires_both_complete_matched_seed_panels():
    import copy

    import pytest

    rows = [{'parameterization':kind, 'seed':seed, 'ratio':2, 'sample_ids':['one', 'two'],
             'sample_count':2, 'field_and_sensor_feasible':True}
            for seed in (11, 12, 13) for kind in ('full_model', 'residual_head')]
    summary = {'version':'common_prefit_calibration_v1', 'scale_joint_ratio':2,
               'inputs':{'bank_sha256':'bank', 'source_panel_sha256':'panel', 'scalable_artifact_sha256':'core'},
               'scale_probe':{'schema':'car_round2_scale_probe_v1', 'reserved_task_used':False,
                              'source_scale':1., 'decision':{'chosen_ratio':2}, 'rows':rows}}
    kwargs = {'bank_sha256':'bank', 'panel_sha256':'panel', 'core_sha256':'core', 'sample_ids':['one','two']}
    probes, chosen, selected, seeds = finalizer.validate_probe_summary(summary, **kwargs)
    assert probes is summary['scale_probe'] and chosen == 2 and len(selected) == 6 and len(seeds) == 3
    changed = copy.deepcopy(summary)
    changed['scale_probe']['rows'].pop()
    with pytest.raises(ValueError, match='cover the frozen panel'):
        finalizer.validate_probe_summary(changed, **kwargs)
    changed = copy.deepcopy(summary)
    changed['inputs']['source_panel_sha256'] = 'other'
    with pytest.raises(ValueError, match='SOURCE panel'):
        finalizer.validate_probe_summary(changed, **kwargs)
