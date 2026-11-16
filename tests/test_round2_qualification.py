import math

import pytest
import torch

from phycoflow_reconstruction.coherence.round2_qualification import (
    audit_parameter_panel,
    full_parameter_leaf_gram,
    importance_mse,
    second_moment_proposal,
)


def test_uncentered_mse_matches_exhaustive_draw_and_detects_old_centering_error():
    gradients = torch.tensor([[2., 3.], [2., -1.], [-3., 5.]], dtype=torch.float64)
    weights = torch.tensor([.2, .3, .5], dtype=torch.float64)
    probabilities = torch.tensor([.1, .2, .7], dtype=torch.float64)
    target = weights @ gradients
    direct = sum(float(p * ((w * g / p - target) ** 2).sum())
                 for p, w, g in zip(probabilities, weights, gradients))
    actual = importance_mse(gradients @ gradients.T, weights, probabilities, 1)
    assert actual == pytest.approx(direct)
    assert importance_mse(gradients @ gradients.T, weights, probabilities, 4) == pytest.approx(direct / 4)
    shifted = gradients - gradients[0]
    assert importance_mse(shifted @ shifted.T, weights, probabilities, 1) != pytest.approx(actual)


def test_actual_full_parameter_gradients_include_unused_and_each_member():
    weight = torch.nn.Parameter(torch.tensor([2., 3.], dtype=torch.float64))
    unused = torch.nn.Parameter(torch.tensor([9.], dtype=torch.float64))
    leaves = [weight[0] ** 2, weight[0] * weight[1]]
    gram, size = full_parameter_leaf_gram(leaves, [weight, unused])
    expected = torch.tensor([[4., 0., 0.], [3., 2., 0.]], dtype=torch.float64)
    torch.testing.assert_close(gram, expected @ expected.T)
    assert size == 3


def test_second_moment_is_detached_positive_and_has_uniform_exploration():
    proposal = second_moment_proposal([1., 1.], torch.tensor([0., 100.], requires_grad=True), [1., 4.])
    assert not proposal.requires_grad
    assert proposal.sum() == 1
    assert proposal.min() >= .25 / 2
    torch.testing.assert_close(second_moment_proposal([1., 1.], [0., 0.], [1., 2.]), torch.tensor([.5, .5], dtype=torch.float64))


def test_panel_audit_charges_model_fixed_work_and_rejects_reduced_population():
    ids = [f'train:{i}' for i in range(8)]
    gram = torch.diag(torch.tensor([1., 100.], dtype=torch.float64))
    calls = []
    def panel():
        calls.append(1)
        for start in (0, 4):
            yield {'ids': ids[start:start + 4], 'gram': gram, 'leaf_seconds': [.1, .2],
                   'fixed_seconds': 20., 'parameter_count': 2}
    result = audit_parameter_panel(panel, sample_ids=ids, population=80, weights=[.5, .5],
        c8_probabilities=[.4, .6], draws=3, source_family='senseiver',
        parameterization='residual_head', preparation_seconds=2., training_seconds=10000.)
    assert len(calls) == 2
    assert result['policies']['uniform']['full_seconds'] == pytest.approx(20.45)
    assert result['parameter_count'] == 2
    assert result['policies']['lagged_gradient_cost']['mse_cost'] < result['policies']['uniform']['mse_cost']
    assert math.isfinite(result['overhead_fraction'])
    with pytest.raises(ValueError, match='10%'):
        audit_parameter_panel(panel, sample_ids=ids[:2], population=80, weights=[.5, .5],
            c8_probabilities=[.4, .6], draws=3, source_family='senseiver',
            parameterization='residual_head', preparation_seconds=2., training_seconds=10000.)
