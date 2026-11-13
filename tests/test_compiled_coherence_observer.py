"""Small numerical cases qualify computation, not population performance."""

import pytest
import torch
from test_frozen_coherence_observer import make_observer


@pytest.mark.parametrize("members", [1, 2, 4])
def test_compiled_values_and_every_member_parameter_gradients(members):
    observer, draws, _, _ = make_observer("deterministic" if members == 1 else "conditional")
    if members == 4:
        draws = torch.cat((draws, draws + 0.173), 0)
    draws = draws[:members]
    parameter = torch.tensor(0.93, dtype=draws.dtype, requires_grad=True)
    endpoint = (draws * parameter).requires_grad_()
    fast = observer.score_tensors(endpoint)
    slow = observer.score_tensors(endpoint, compiled=False)
    for name in fast:
        torch.testing.assert_close(fast[name], slow[name], rtol=3e-6, atol=3e-7)
        fg = torch.autograd.grad(fast[name].sum(), (endpoint, parameter), retain_graph=True)
        sg = torch.autograd.grad(slow[name].sum(), (endpoint, parameter), retain_graph=True)
        for a, b in zip(fg, sg):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
        assert all(torch.isfinite(x).all() for x in fg)
    scored = observer.score(endpoint)
    assert scored["members"] == members
    if members > 1:
        reversed_score = observer.score(endpoint.flip(0))
        for name in fast:
            torch.testing.assert_close(
                scored["per_sample"][name],
                reversed_score["per_sample"][name],
                rtol=3e-6,
                atol=3e-7,
            )


def test_reference_cache_mutation_is_rejected():
    observer, draws, _, _ = make_observer("conditional")
    observer.scale_reference[0].add_(1)
    with pytest.raises(ValueError, match="content changed"):
        observer.score(draws)


def test_scale_observer_covers_every_leaf():
    observer, _, _, _ = make_observer("conditional")
    (batch,) = observer.scale_batches
    assert len(batch.self_queries) == 2
    assert len(batch.pair_queries) == 1
    assert sum(q.weight for q in batch.self_queries) == 1
    assert sum(q.weight for q in batch.pair_queries) == 1


def test_j4_equals_independent_j2_pair_average_and_uses_later_members():
    observer, draws, _, _ = make_observer("conditional")
    generator = torch.Generator().manual_seed(991)
    extra = torch.randn(draws.shape, generator=generator, dtype=draws.dtype)
    members = torch.cat((draws, extra), 0).requires_grad_()
    actual = observer.score_tensors(members)
    pairs = [
        observer.score_tensors(members[[a, b]], compiled=False)
        for a in range(4)
        for b in range(a + 1, 4)
    ]
    for name, value in actual.items():
        expected = torch.stack([row[name] for row in pairs]).mean(0)
        torch.testing.assert_close(value, expected, rtol=3e-6, atol=3e-7)
        fast = torch.autograd.grad(value.sum(), members, retain_graph=True)[0]
        reference = torch.autograd.grad(expected.sum(), members, retain_graph=True)[0]
        torch.testing.assert_close(fast, reference, rtol=3e-5, atol=3e-6)
        if name != "shape":  # Piecewise-flat scalar persistence scores may have zero derivatives.
            assert fast[2].norm() > 0 and fast[3].norm() > 0


def test_cached_state_uses_same_normalized_mass_as_direct():
    from phycoflow_reconstruction.coherence.state_compute import prepare_endpoint
    from phycoflow_reconstruction.evaluation.frozen_coherence import FrozenCoherenceObserver

    original, _, families, artifact = make_observer("conditional")
    masses = torch.linspace(0.0001, 0.0019, 16, dtype=torch.float32).repeat(2, 1)
    observer = FrozenCoherenceObserver(
        artifact,
        families,
        expected_provenance=artifact["provenance"],
        reference=original.reference.float(),
        coordinates=original.coordinates.float(),
        sample_ids=["a", "b"],
        member_weights=masses,
        reference_weights=masses,
        mode="conditional",
        query_seed=3,
        query_batches=2,
    )
    family = observer.families["global_distribution"]
    _, normalized = family._endpoint_measures(
        observer.reference, observer.reference, observer.context, required=True
    )
    reference = family._in_declared_units(observer.reference, observer.reference)[1]
    expected = prepare_endpoint(reference, normalized, artifact["state_bank"]["directions"])
    for actual, wanted in zip(observer.state_reference, expected):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


def test_compiled_observer_uses_reference_shape_path_for_gradients(monkeypatch):
    observer, draws, _, _ = make_observer("conditional")
    calls = []
    original = observer.adapter.score

    def traced(name, *args, **kwargs):
        if name == "topology":
            calls.append(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(observer.adapter, "score", traced)
    members = torch.cat((draws, draws + 0.2)).requires_grad_()
    observer.score_tensors(members, compiled=True)
    assert len(calls) == len(observer.batches)
    calls.clear()
    with torch.no_grad():
        observer.score_tensors(members, compiled=True)
    assert not calls
