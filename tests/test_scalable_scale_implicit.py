import pytest
import torch

from phycoflow_reconstruction.coherence.families.multiscale_relation import MultiscaleRelationFamily
from phycoflow_reconstruction.coherence.query_runtime import score
from phycoflow_reconstruction.coherence.query_sampling import Draw, QueryBatch, pair_count
from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer


def build(count, states, *, implicit):
    names = tuple(f"f{i}" for i in range(count))
    config = load_config("configs/coherence/scalable_scale.yaml")["coherence"]["families"][
        "multiscale_spectrum"
    ]
    if not implicit:
        del config["relation_extension"]["implicit_pairs"]
        del config["components"]
    spec = DataSpec(names, ("1",) * count, 1, (16,))
    family = MultiscaleRelationFamily(config, spec, FieldNormalizer.identity(count))
    coords = torch.linspace(0.0, 1.0, 16, dtype=torch.float64)[:, None] ** 1.3
    family.fit_reference_tensors(
        coords, states, sample_ids=[f"train:{i}" for i in range(len(states))]
    )
    return family, coords


def full_batch(count):
    self_draws = tuple(Draw(i, 1, 1 / count, count, count) for i in range(count))
    p = pair_count(count)
    pair_draws = tuple(Draw(i, 1, 1 / p, p, p) for i in range(p))
    return QueryBatch(self_draws, pair_draws, 0)


@pytest.mark.parametrize("count", [1, 2, 20, 100])
def test_implicit_pair_fit_avoids_dense_gram_and_pair_manifests(monkeypatch, count):
    from phycoflow_reconstruction.coherence.families.multiscale_spectrum import features

    def prohibited(*args, **kwargs):
        raise AssertionError("dense Gram allocation")

    monkeypatch.setattr(features, "band_gram", prohibited)
    states = torch.randn(
        3, 16, count, generator=torch.Generator().manual_seed(23), dtype=torch.float64
    )
    family, _ = build(count, states, implicit=True)
    assert family.coverage_pairs == ()
    assert family.settings.component("signed_band_cospectrum") is None
    stats = family._statistics(states[:1])
    assert "gram" not in stats and stats["power"].shape[-1] == count
    assert stats["pair_gram"].numel() == 0
    assert family.layout is None or sum(len(x) for x in family.layout.names) <= 2 * count
    assert family.relation_self_scale.numel() == 2 * count


def test_implicit_scale_fit_and_exhaustive_values_gradients_match_full_reference():
    states = torch.randn(8, 16, 2, generator=torch.Generator().manual_seed(29), dtype=torch.float64)
    sparse, coords = build(2, states, implicit=True)
    dense, _ = build(2, states, implicit=False)
    for name in ("band_scale", "power_floor", "piece_mean_scale", "relation_self_scale"):
        torch.testing.assert_close(
            getattr(sparse, name), getattr(dense, name), rtol=1e-10, atol=1e-12
        )
    first, second = (states[i : i + 2].clone().requires_grad_() for i in (0, 2))
    reference = states[4:6]
    coordinates = coords[None].expand(2, -1, -1)
    context = {"conditional_draws": (second,)}
    actual, _, _ = score("scale", sparse, first, reference, coordinates, context, full_batch(2), {})
    expected = dense(first, reference, coordinates=coordinates, context=context)
    torch.testing.assert_close(actual.scalar_loss, expected.scalar_loss, rtol=1e-8, atol=1e-9)
    a = torch.autograd.grad(actual.scalar_loss, (first, second), retain_graph=True)
    b = torch.autograd.grad(expected.scalar_loss, (first, second))
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, rtol=1e-7, atol=1e-9)
    with pytest.raises(ValueError, match="ScalableQueryAdapter"):
        sparse(first, reference, coordinates=coordinates, context=context)
