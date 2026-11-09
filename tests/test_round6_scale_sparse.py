import itertools

import pytest
import torch
from test_multiscale_gram_extension import _fitted, _score
from test_multiscale_relation import config


@pytest.mark.parametrize("count", [1, 2, 5])
def test_sparse_scale_preserves_values_and_draw_gradients(count):
    torch.set_num_threads(1)
    names = tuple(f"f{i}" for i in range(count))
    generator = torch.Generator().manual_seed(33)
    states = torch.randn(6, 16, count, generator=generator, dtype=torch.float64)
    family, coords, _ = _fitted(config(names), states)
    x = states[:2].clone().requires_grad_()
    y = states[2:4].clone().requires_grad_()
    target = states[4:6]
    dense = _score(family, coords, x, y, target)
    dense_grad = torch.autograd.grad(dense.scalar_loss, (x, y))
    family.coverage_pairs = tuple(itertools.combinations(range(count), 2))
    sparse = _score(family, coords, x, y, target)
    sparse_grad = torch.autograd.grad(sparse.scalar_loss, (x, y))
    torch.testing.assert_close(sparse.scalar_loss, dense.scalar_loss, rtol=1e-12, atol=1e-12)
    for a, b in zip(sparse_grad, dense_grad):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    assert "gram" not in family._statistics(x)
    assert len(sparse.diagnostics["coverage_priorities"]["self"]) == count
    assert len(sparse.diagnostics["coverage_priorities"]["pair"]) == count * (count - 1) // 2


@pytest.mark.parametrize("count", [20, 100])
def test_implicit_large_coverage_bank_is_bounded_and_sparse(count, monkeypatch):
    from phycoflow_reconstruction.coherence import build_enabled_families
    from phycoflow_reconstruction.coherence.families.multiscale_spectrum import features
    from phycoflow_reconstruction.coherence.relation_queries import (
        QueryBank,
        resolve_coverage_config,
    )
    from phycoflow_reconstruction.contracts import DataSpec
    from phycoflow_reconstruction.data.normalization import FieldNormalizer

    names = tuple(f"f{i:03}" for i in range(count))
    scale = config(names)
    coherence = {"query_coverage": {"enabled": True, "relation_seed": 11},
                 "families": {"multiscale_spectrum": scale}}
    resolved = resolve_coverage_config(coherence, names)
    bank = QueryBank(names, seed=11)
    assert len(resolved["families"]["multiscale_spectrum"]["components"]["signed_band_cospectrum"]["pairs"]) == 2 * count
    assert "components" not in scale
    family = build_enabled_families(coherence, DataSpec(names, ("1",) * count, 2, (16,)),
                                    FieldNormalizer.identity(count))["multiscale_spectrum"]
    assert len(family.coverage_pairs) == len(bank.pairs)
    assert len(family.settings.component("signed_band_cospectrum").pairs) == len(bank.pairs)
    def forbidden(*args, **kwargs):
        raise AssertionError("dense Gram computation")
    monkeypatch.setattr(features, "band_gram", forbidden)
    data = torch.randn(2, 16, count, dtype=torch.float64, requires_grad=True)
    stats = features.raw_statistics(data, mass=torch.ones(16, dtype=data.dtype) / 16,
        basis=torch.randn(16, 3, dtype=data.dtype), filters=torch.ones(2, 3, dtype=data.dtype),
        piece_measures=torch.ones(1, 16, dtype=data.dtype) / 16, pieces=torch.zeros(16, dtype=torch.long),
        manifest=family.manifest, relation_pairs=family.coverage_pairs)
    assert "gram" not in stats
    assert stats["power"].shape == (2, 2, count)
    assert stats["pair_gram"].shape == (2, 2, 2 * count)
    assert torch.isfinite(torch.autograd.grad(stats["pair_gram"].sum(), data)[0]).all()


def test_five_field_coverage_keeps_config_identity():
    from phycoflow_reconstruction.coherence.relation_queries import resolve_coverage_config

    names = tuple(f"f{i}" for i in range(5))
    original = {"query_coverage": {"enabled": True},
                "families": {"multiscale_spectrum": config(names)}}
    assert resolve_coverage_config(original, names) == original


@pytest.mark.parametrize("count", [1, 5, 20])
def test_full_family_default_resolves_shared_shape_bank(count):
    from test_topology_finite_primary import config as topology_config

    from phycoflow_reconstruction.coherence import build_enabled_families
    from phycoflow_reconstruction.coherence.relation_queries import (
        QueryBank,
        resolve_coverage_config,
    )
    from phycoflow_reconstruction.contracts import DataSpec
    from phycoflow_reconstruction.data.normalization import FieldNormalizer

    names = tuple(f"f{i:03}" for i in range(count))
    shape = topology_config()
    shape["fields"] = ["CO", "T"]
    shape["components"]["self"]["fields"] = ["CO", "T"]
    original = {"query_coverage": {"enabled": True, "relation_seed": 9}, "families": {
        "multiscale_spectrum": config(names), "topology": shape}}
    resolved = resolve_coverage_config(original, names)
    bank = QueryBank(names, seed=9)
    shape = resolved["families"]["topology"]
    assert shape["fields"] == list(names)
    assert shape["components"]["self"]["fields"] == list(names)
    assert shape["components"]["mutual"]["groups"] == [list(pair) for pair in bank.pairs]
    assert shape["components"]["mutual"]["enabled"] == (count > 1)
    assert resolve_coverage_config(resolved, names) == resolved
    families = build_enabled_families(original, DataSpec(names, ("1",) * count, 2, (16,)),
                                      FieldNormalizer.identity(count))
    objective = families["topology"].spatial_objective
    assert objective.self_fields == names
    assert tuple(tuple(objective.fields[i] for i in pair) for pair in objective.groups) == bank.pairs
    assert ("mutual.persistence" in objective.weights) == (count > 1)
    legacy = {"families": original["families"]}
    assert resolve_coverage_config(legacy, names) == legacy
