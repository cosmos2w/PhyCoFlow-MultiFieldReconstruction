from types import SimpleNamespace

import pytest
import torch

from phycoflow_reconstruction.coherence.query_sampling import SharedQuerySampler
from scripts.qualification.car_round5_query_endpoint_replay import gram_summary, query_groups
from scripts.qualification.car_round5_query_parameter_replay import plan


def test_query_replays_couple_first_query_and_cover_short_batch():
    runtime = SimpleNamespace(sampler=SharedQuerySampler(["a", "b", "c"], seed=17))
    shared, grouped = plan(runtime, 3, 12)
    assert shared == grouped.batches[0]
    assert len(list(grouped.slices(5))) == 5
    assert sum(sl.stop-sl.start for sl, _ in grouped.slices(5)) == 5
    assert all(d.weight == pytest.approx(1.0) for batch in grouped.batches
               for d in batch.self_queries)
    endpoint_shared, endpoint_grouped = query_groups(("a", "b", "c"), 3, 768, 32)
    assert endpoint_shared[0][1] == endpoint_grouped[0][1]
    assert sum(sl.stop-sl.start for sl, _ in endpoint_grouped) == 32


def test_gradient_gram_summary_uses_all_coordinates():
    gradients = torch.tensor([[1., 2., 5.], [3., 4., 7.]], dtype=torch.double)
    result = gram_summary(gradients @ gradients.T)
    mean = gradients.mean(0)
    assert result['mean_gradient_squared_norm'] == pytest.approx(float(mean @ mean))
    assert result['centered_gradient_squared_norm'] == pytest.approx(
        float((gradients - mean).square().sum() / len(gradients)))
