import pytest
import torch

from phycoflow_reconstruction.evaluation.round2_assessment import (
    deployment_disposition,
    field_sensor_eligibility,
    freeze_method_priority,
    native_risks,
    nominate_train_amplitude,
    scalar_task_score,
)


def test_signed_mean_fidelity_never_uses_signed_risk_as_denominator():
    source = {'member_mse': [4., 0.], 'signed_mean_risk': [-2., 0.]}
    child = {'member_mse': [8., .01], 'signed_mean_risk': [-1.7, .001]}
    result = field_sensor_eligibility(source, child, model_family='ffm',
        absolute_mean_allowance=[.4, 0.], source_sensor_mse=[0., 2.],
        child_sensor_mse=[0., 2.1], sensor_equality_tolerance=[1e-12, 0.])
    assert not result['primary_field_pass']  # zero allowance remains zero
    child['signed_mean_risk'][1] = 0.
    result = field_sensor_eligibility(source, child, model_family='ffm',
        absolute_mean_allowance=[.4, 0.], source_sensor_mse=[0., 2.],
        child_sensor_mse=[0., 2.1], sensor_equality_tolerance=[1e-12, 0.])
    assert result['primary_pass'] and not result['legacy_member_pass']
    assert result['field_allowance'] == [.4, 0.]


def test_native_ffm_spread_and_fair_scalar_score_are_distinct_from_point_case():
    x = torch.tensor([-1., 1., -1., 1.]).reshape(4, 1, 1, 1)
    truth = torch.zeros(1, 1, 1)
    risk = native_risks(x, truth, torch.ones(1), model_family='ffm')
    assert risk['member_mse'].item() == 1
    assert risk['ensemble_mean_mse'].item() == 0
    assert risk['unbiased_member_variance'].item() == pytest.approx(4 / 3)
    assert risk['signed_mean_risk'].item() == pytest.approx(-1 / 3)
    assert scalar_task_score(x[:, :, 0, 0], truth[:, 0, 0], model_family='ffm').item() == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match='J1'):
        scalar_task_score(x[:, :, 0, 0], truth[:, 0, 0], model_family='senseiver')


def test_train_amplitude_uses_actual_blend_observer_and_freezes_without_reserved_task():
    source = torch.ones(1, 1, 1, 1)
    child = source * .5
    baseline = {key: 1. for key in ('state', 'scale', 'shape', 'event')}
    def evaluate(fields):
        return {'scores': {key: float(fields.mean()) for key in baseline},
                'fidelity': {'primary_pass': True}}
    result = nominate_train_amplitude(source, child, sample_ids=['train:0'], population=10,
        source_scores=baseline, normalizers=baseline, evaluate=evaluate,
        latency=lambda alpha: 1. if alpha in (0., 1.) else 2.,
        source_identity='source-bytes', child_identity='child-bytes')
    assert result['selected_alpha'] == 1 and not result['reserved_task_used']
    assert len(result['rows']) == 9


def test_frozen_priority_and_mean_only_opt_in_preserve_operational_fallback():
    nominees = [{'method': 'FullModel', 'eligible': True, 'alpha': .5, 'latency_seconds': 1.005, 'utility': 1.},
                {'method': 'ResidualRepair', 'eligible': True, 'alpha': 1., 'latency_seconds': 1., 'utility': 2.}]
    priority = freeze_method_priority(nominees)
    assert priority == ['FullModel', 'ResidualRepair']
    assessments = {name: {'fidelity': {'primary_pass': True, 'legacy_member_pass': False},
                          'core': {'pass': True}, 'reserved_normalized_improvement': .08,
                          'source_latency_ratio': 1.1} for name in priority}
    result = deployment_disposition(priority, assessments, model_family='ffm', complete_costs=True, within_budget=True)
    assert result['opt_in_function'] == 'FullModel'
    assert result['operational_recommendation'] == 'retain SOURCE/C8'
    assert not result['held_out_amplitude_search']


def test_incomplete_costs_preserve_task_evidence_without_false_route_verdict():
    assessments = {'ResidualRepair': {'fidelity': {'primary_pass': True, 'legacy_member_pass': True},
                                     'core': {'pass': True}, 'reserved_normalized_improvement': .08,
                                     'source_latency_ratio': 1.1}}
    result = deployment_disposition(['ResidualRepair'], assessments, model_family='ffm',
                                    complete_costs=False, within_budget=True)
    row = result['decisions']['ResidualRepair']
    assert row['task_useful'] is True
    assert row['source_deployment_route'] is None and row['operational_eligible'] is None
    assert result['cost_qualification_status'] == 'unavailable_incomplete_costs'
    assert result['opt_in_function'] is None and result['operational_function'] is None
    assert result['operational_recommendation'] == 'retain SOURCE/C8'
    known = deployment_disposition(['ResidualRepair'], assessments, model_family='ffm',
                                   complete_costs=True, within_budget=False)
    assert known['decisions']['ResidualRepair']['source_deployment_route'] is False
