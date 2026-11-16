import pytest
import torch

from phycoflow_reconstruction.evaluation.adaptive_assessment import (
    FrozenEventTask,
    select_train_alpha,
)


def test_frozen_weighted_event_and_realization_metrics():
    # One TRAIN condition is 10% of its declared ten-sample population.
    truth = torch.tensor([[[0., 0., 0.], [1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]])
    mass = torch.tensor([1., 1., 1., 2.])
    task = FrozenEventTask.fit(truth, mass, field_names=('T', 'CO', 'CH4'),
        sample_ids=['sample'], population=10, units='unknown', data_identity='dataset', split='train')
    assert task.temperature_q75 == 3 and task.co_q90 == 3 and task.ch4_q25 == 1
    # Change thresholds for an interpretable nonempty event with independently
    # hand-computable area, overlap, missing, excess and moment.
    task = FrozenEventTask(('T', 'CO', 'CH4'), .5, .5, 2.5, ('sample',), 'unknown', 10, 'dataset')
    members = truth.unsqueeze(0).repeat(4, 1, 1, 1)
    result = task.score(members, truth, mass, torch.arange(4.).unsqueeze(-1))
    torch.testing.assert_close(result['truth_area_fraction'], torch.tensor([.4], dtype=torch.float64))
    torch.testing.assert_close(result['truth_first_moment'], torch.tensor([[3.]], dtype=torch.float64))
    assert torch.equal(result['area_crps'], torch.zeros(1, dtype=torch.float64))
    assert torch.equal(result['missing_fraction'], torch.zeros((4, 1), dtype=torch.float64))
    assert torch.equal(result['excess_fraction'], torch.zeros((4, 1), dtype=torch.float64))
    with pytest.raises(ValueError):
        FrozenEventTask.fit(truth, mass, field_names=('T', 'CO', 'CH4'),
          sample_ids=['sample'], population=10, units='unknown', data_identity='dataset', split='validation')


def test_alpha_frontier_uses_actual_blends_and_exact_risk_guard():
    truth = torch.zeros((1, 2, 1))
    source = torch.ones((4, 1, 2, 1))
    child = torch.zeros_like(source)
    seen = []
    def observer(members):
        seen.append(float(members.mean()))
        return [members.square().mean(), torch.tensor(1.), torch.tensor(1.)]
    result = select_train_alpha(source, child, truth, [1., 2.], sample_ids=['a'], population=10,
        source_hash='source', child_hash='child', coupling_identity='paired', source_core=[1., 1., 1.],
        core_observer=observer, source_inference_cost=1., child_inference_cost=1.5, blend_inference_cost=2., split='train')
    assert result['selected_alpha'] == 1.0
    assert result['source_endpoints_sha256'] != result['child_endpoints_sha256']
    assert seen == [1., .875, .75, .625, .5, .375, .25, .125, 0.]
    for row in result['rows']:
        assert row['risk'][0] == pytest.approx((1 - row['alpha']) ** 2)
    with pytest.raises(ValueError):
        select_train_alpha(source, child, truth, [1., 2.], sample_ids=['a'], population=10,
          source_hash='source', child_hash='child', coupling_identity='paired', source_core=[1., 1., 1.],
          core_observer=observer, source_inference_cost=1., child_inference_cost=1.5, blend_inference_cost=2., split='validation')
    with pytest.raises(ValueError, match='SOURCE denominators'):
        select_train_alpha(source, child, truth, [1., 2.], sample_ids=['a'], population=10,
          source_hash='source', child_hash='child', coupling_identity='paired', source_core=[2., 1., 1.],
          core_observer=observer, source_inference_cost=1., child_inference_cost=1.5, blend_inference_cost=2., split='train')


def test_alpha_ties_prefer_source_cost():
    source = torch.ones((4, 1, 2, 1))
    result = select_train_alpha(source, source, torch.zeros((1, 2, 1)), [1., 1.],
        sample_ids=['a'], population=10, source_hash='s', child_hash='c', coupling_identity='p',
        source_core=[1., 1., 1.], core_observer=lambda _: [1., 1., 1.],
        source_inference_cost=1., child_inference_cost=1.5, blend_inference_cost=2., split='train')
    assert result['selected_alpha'] == 0


def test_j4_area_crps_uses_all_six_member_pairs():
    task = FrozenEventTask(('T', 'CO', 'CH4'), .5, .5, .5, ('a',), 'unknown', 10, 'dataset')
    truth = torch.zeros((1, 1, 3))
    members = torch.zeros((4, 1, 1, 3))
    members[2:, :, :, 0:2] = 1
    result = task.score(members, truth, [1.], torch.tensor([[2.]]))
    assert float(result['area_crps']) == pytest.approx(1.0 / 6)
    assert result['excess_fraction'].flatten().tolist() == [0., 0., 1., 1.]
