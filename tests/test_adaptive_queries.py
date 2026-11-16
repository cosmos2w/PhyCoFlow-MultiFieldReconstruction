import json

import pytest
import torch

from phycoflow_reconstruction.coherence.adaptive_discovery import candidates, freeze
from phycoflow_reconstruction.coherence.query_sampling import (
    HierarchicalQuerySampler,
    allocate_cost_counts,
    allocate_counts,
    pair_count,
)
from phycoflow_reconstruction.coherence.scale_compute import third_order_scores
from phycoflow_reconstruction.coherence.state_compute import group_projection_scores


def test_fixed_profile_draws_and_resume():
    probabilities = [.10, .15, .15, .20, .20, .20]
    sampler = HierarchicalQuerySampler(list('abcd'), seed=8, hierarchy_weight=0,
                                        fixed_probabilities=probabilities, profile_sha256='a' * 64)
    for _ in range(5):
        batch = sampler.draw()
        for draw in batch.pair_queries:
            assert draw.probability == probabilities[draw.index]
        sampler.finish()
    restored = HierarchicalQuerySampler.from_state_dict(json.loads(json.dumps(sampler.state_dict())))
    assert restored.draw() == sampler.draw()
    with pytest.raises(ValueError):
        HierarchicalQuerySampler(list('abcd'), fixed_probabilities=[0., .2, .2, .2, .2, .2],
                                 profile_sha256='a' * 64)


@pytest.mark.parametrize("n", [1, 2, 5, 20, 100, 1000])
def test_hierarchy_probability_and_rollback(n):
    sampler = HierarchicalQuerySampler([str(i) for i in range(n)], seed=91)
    if n > 1:
        # Full normalization on modest populations; bounded spot checks at 1000.
        if n <= 100:
            assert sum(sampler.pair_probability(i) for i in range(pair_count(n))) == pytest.approx(
                1
            )
        for i in (0, pair_count(n) // 2, pair_count(n) - 1):
            assert sampler.pair_probability(i) >= 0.25 / pair_count(n)
    batch = sampler.draw()
    restored = HierarchicalQuerySampler.from_state_dict(
        json.loads(json.dumps(sampler.state_dict()))
    )
    restored.reject()
    assert restored.draw() == batch
    restored.finish()
    assert restored.counter == 1


def test_allocation_and_discovery_contract():
    assert sum(allocate_counts([1, 1], [2, 1], [1, 4], 10)) == 10
    assert all(x >= 1 for x in allocate_counts([1, 1], [2, 1], [1, 4], 10))
    bank = candidates(list("abcde"), bands=2, regions=2, seed=3)
    assert len(bank["candidates"]) <= 40
    assert bank == candidates(list("abcde"), bands=2, regions=2, seed=3)
    metrics = {
        q["id"]: {"discrepancy": 1, "repeatability": 1, "sensitivity": 1, "cost": 1}
        for q in bank["candidates"]
    }
    with pytest.raises(ValueError):
        freeze(bank, metrics, split="TEST", population_ids=[1], source_identity="source")
    frozen = freeze(bank, metrics, split="TRAIN", population_ids=[1, 2], source_identity="source")
    assert sum(x["kind"] == "state.group_projection" for x in frozen["selected"]) <= 4
    assert sum(x["kind"] == "scale.third_order" for x in frozen["selected"]) <= 8


def test_cost_allocation_uses_charged_budget_and_active_minima():
    counts = allocate_cost_counts([1, 1, 0], [1, 1, 9], [1, 4, 1], 24)
    assert counts == [8, 4, 0]
    assert sum(n * c for n, c in zip(counts, [1, 4, 1])) <= 24
    assert allocate_cost_counts([1, 1], [0, 0], [2, 3], 20) == [1, 1]
    with pytest.raises(ValueError, match="cover"):
        allocate_cost_counts([1, 1], [2, 1], [2, 3], 4)


def test_fair_added_leaves_and_member_gradients():
    torch.manual_seed(3)
    x = torch.randn(4, 2, 5, 3, dtype=torch.double, requires_grad=True)
    y = torch.randn(2, 5, 3, dtype=torch.double)
    masses = torch.tensor([1.0, 2.0, 1.0, 3.0, 1.0], dtype=torch.double)
    score = group_projection_scores(x, y, masses, masses, [[1.0, -1.0, 0.0]])
    assert score.shape == (2, 1)
    score.sum().backward()
    assert all(x.grad[i].abs().sum() > 0 for i in range(4))
    x = torch.randn(4, 2, 2, 5, 3, dtype=torch.double, requires_grad=True)
    y = 20 * torch.randn(2, 2, 5, 3, dtype=torch.double)
    score = third_order_scores(
        x, y, [(0, ((0, 0), (1, 1), (0, 2)))], masses[None], torch.ones(1, 2, 3)
    )
    score.sum().backward()
    assert score.shape == (2, 1)
    assert all(x.grad[i].abs().sum() > 0 for i in range(4))


def test_compiled_state_reference_value_and_gradient_parity():
    from phycoflow_reconstruction.coherence.state_compute import (
        exact_common_embeddings,
        prepare_endpoint,
    )

    torch.manual_seed(8)
    x = torch.randn(2, 5, 3, dtype=torch.double, requires_grad=True)
    y = torch.randn(2, 5, 3, dtype=torch.double)
    mass = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.double)
    directions = torch.tensor([[1.0, -1.0, 0.0]], dtype=torch.double)
    cache = prepare_endpoint(y, mass, directions)

    def value(compiled):
        e = exact_common_embeddings([x, y], [mass, mass], directions, compiled_reference=compiled)
        return sum((a - b).square().sum() for a, b in zip(e[0], e[1]))

    direct, cached = value(None), value(cache)
    assert torch.allclose(direct, cached)
    assert torch.allclose(torch.autograd.grad(direct, x)[0], torch.autograd.grad(cached, x)[0])


def test_retained_scale_bound_and_margin_with_unequal_disconnected_masses():
    from phycoflow_reconstruction.coherence.representation_contracts import (
        retained_scale_constant,
        task_distance_bound,
        threshold_margin_bound,
    )

    # Two disconnected pieces, one fluctuation mode each and unequal masses.
    filters = torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=torch.double)
    scales = torch.tensor([[2.0, 1.0], [1.0, 3.0]], dtype=torch.double)
    piece_masses = torch.tensor([3.0, 7.0], dtype=torch.double)
    mean_scales = torch.tensor([[2.0, 3.0], [1.0, 4.0]], dtype=torch.double)
    c = retained_scale_constant(filters, scales, piece_masses, mean_scales)
    modal = torch.tensor([[1.0, -2.0], [0.7, 2.0]], dtype=torch.double)
    means = torch.tensor([[0.3, 2.0], [-0.5, 1.0]], dtype=torch.double)
    physical_sq = modal.square().sum() / 10 + (means.square() * piece_masses[:, None]).sum() / 10
    embedding_sq = (modal.square() * c["modal_weights"]).sum() / 10 + (
        means.square() * piece_masses[:, None] * c["mean_weights"]
    ).sum() / 10
    assert embedding_sq >= c["c_min"] * physical_sq
    assert task_distance_bound(embedding_sq.sqrt(), 0, c["c_min"]) >= physical_sq.sqrt()
    zero = retained_scale_constant(torch.zeros_like(filters), scales, piece_masses, mean_scales)
    assert not zero["inverse_available"]
    z = torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 1.0]], dtype=torch.double)
    x = z + torch.tensor([[0.2, 0.0], [-0.4, 0.2], [0.0, -0.5]], dtype=torch.double)
    out = threshold_margin_bound(x, z, torch.tensor([1.0, 2.0, 7.0]), [1.0, 1.2], [0.3, 0.4])
    assert out["disagreement"] <= out["bound"]


def test_flat_interval_and_member_mean_identity():
    # J2 CRPS is exactly zero on a bracket; member risk penalizes model spread.
    x = torch.tensor([-2.0, 3.0], dtype=torch.double, requires_grad=True)
    z = torch.tensor(0.5, dtype=torch.double)
    fair = (x - z).abs().mean() - (x[0] - x[1]).abs() / 2
    assert fair == 0
    assert torch.equal(torch.autograd.grad(fair, x)[0], torch.zeros_like(x))
    truth = torch.tensor([-1.0, 1.0], dtype=torch.double)
    law = torch.tensor([-1.0, 1.0], dtype=torch.double)
    member_risk = (law[:, None] - truth[None, :]).square().mean()
    truth_variance = truth.var(unbiased=False)
    model_variance = law.var(unbiased=False)
    mean_error = (law.mean() - truth.mean()).square()
    assert member_risk == truth_variance + mean_error + model_variance
    assert member_risk > truth_variance + mean_error


def test_margin_bound_observes_mixed_strict_event_including_cut_equality():
    from phycoflow_reconstruction.coherence.representation_contracts import threshold_margin_bound
    z = torch.tensor([[2., 2., 0.], [1., 2., 0.], [2., 2., 1.]], dtype=torch.double)
    x = torch.tensor([[2., 2., 1.], [2., 2., 0.], [2., 2., 0.]], dtype=torch.double)
    result = threshold_margin_bound(x, z, torch.tensor([1., 2., 3.]),
                                    [1., 1., 1.], [.2, .2, .2],
                                    operators=[">", ">", "<"])
    assert result["disagreement"] == 1
    assert result["disagreement"] <= result["bound"]
    with pytest.raises(ValueError, match="operator"):
        threshold_margin_bound(x, z, torch.ones(3), [1., 1., 1.], [.2]*3,
                               operators=[">", "<"])


def test_parity_pair_blindness_and_third_order_sensitivity():
    import itertools

    even = torch.tensor(
        [v for v in itertools.product((-1.0, 1.0), repeat=3) if v[0] * v[1] * v[2] > 0]
    )
    odd = torch.tensor(
        [v for v in itertools.product((-1.0, 1.0), repeat=3) if v[0] * v[1] * v[2] < 0]
    )
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert torch.equal(even[:, [i, j]].sort(dim=0).values, odd[:, [i, j]].sort(dim=0).values)
        assert torch.equal(
            torch.sort(even[:, i] * even[:, j]).values, torch.sort(odd[:, i] * odd[:, j]).values
        )
    assert even.prod(-1).mean() == 1 and odd.prod(-1).mean() == -1
