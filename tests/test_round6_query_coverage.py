"""Small synthetic checks for bounded query allocation and persistence selection."""

from dataclasses import replace

import pytest
import torch

from phycoflow_reconstruction.coherence.relation_queries import QueryBank
from phycoflow_reconstruction.training.adaptive_coverage import (
    AdaptiveCoverageSampler,
    QueryDraw,
    QuerySelection,
)


def test_banks_and_permutation():
    for n in (1, 2, 5, 1000):
        fields = [f"f{i:04}" for i in range(n)]
        bank = QueryBank(fields, seed=17)
        assert len(bank.pairs) == min(n * (n - 1) // 2, 2 * n)
        assert bank.pairs == QueryBank(fields[::-1], seed=17).pairs
        if n > 1:
            assert {field for pair in bank.pairs for field in pair} == set(fields)


def test_draw_accounting_and_resume():
    bank = QueryBank(["a", "b", "c", "d", "e"])
    sampler = AdaptiveCoverageSampler(bank, policy="adaptive", seed=7)
    for _ in range(10):
        sampler.observe([1, 2, 3, 4, 5], range(10))
        sampler.finalize_epoch()
    assert sampler.probabilities["self"][-1] > sampler.probabilities["self"][0]
    state = sampler.state_dict()
    restored = AdaptiveCoverageSampler(bank, policy="adaptive", seed=99)
    restored.load_state_dict(state)
    selection = sampler.select(4, 12)
    assert selection == restored.select(4, 12)
    assert sum(q.multiplicity for q in selection.queries if q.kind == "self") == 4
    assert sum(q.multiplicity for q in selection.queries if q.kind == "mutual") == 12
    for q in selection.queries:
        quota = 4 if q.kind == "self" else 12
        assert q.estimator_weight == pytest.approx(
            q.multiplicity * q.target_weight / q.probability / quota
        )
        assert q.target_weight / q.probability <= 2 + 1e-12
    solo = AdaptiveCoverageSampler(QueryBank(["a"]), seed=1).select()
    assert solo.mutual_draws == 0 and all(q.kind == "self" for q in solo.queries)


def test_dense_selected_value_gradient_and_skipped_calls(monkeypatch):
    pytest.importorskip("gudhi")
    from test_topology_finite_primary import family, provenance

    from phycoflow_reconstruction.coherence.families.topology import finite_primary

    objective = family().spatial_objective.double()
    g = torch.Generator().manual_seed(133)
    target = torch.randn(1, 2, 4, 4, generator=g, dtype=torch.float64)
    source = target + torch.randn(target.shape, generator=g, dtype=torch.float64) * 0.2
    objective.fit_calibration(source, target, provenance=provenance())
    prediction = source.detach().requires_grad_()
    dense = objective(prediction, target)
    grad = torch.autograd.grad(dense.scalar_loss, prediction)[0]
    queries = []
    for component, _, ids, signs, line in objective.units:
        kind = "self" if component.startswith("self") else "mutual"
        fields = tuple(objective.fields[i] for i in ids)
        weight = 1 / 4 if line is None else float(objective.line_weights[line]) / 4
        key = (kind, fields, signs, line)
        queries.append(QueryDraw(key, kind, fields, signs, line, 1.0, weight, 1, weight))
    complete = QuerySelection(tuple(queries), 4, 12)
    selected = objective(prediction, target, query_selection=complete)
    selected_grad = torch.autograd.grad(selected.scalar_loss, prediction)[0]
    torch.testing.assert_close(selected.scalar_loss, dense.scalar_loss, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(selected_grad, grad, rtol=1e-12, atol=1e-12)
    calls = []
    original = finite_primary.cubical_diagrams

    def counting(x, **kwargs):
        calls.append(x.shape[0])
        return original(x, **kwargs)

    monkeypatch.setattr(finite_primary, "cubical_diagrams", counting)
    subset = QuerySelection((replace(queries[0], multiplicity=2), queries[5]), 2, 1)
    result = objective(prediction, target, query_selection=subset)
    assert calls == [2, 2]
    assert result.diagnostics["scalar_filtrations"] == 2
    assert len([k for k in result.component_results if ".h0.finite" in k]) == 2


def test_five_field_selection_and_manifest_mismatch():
    pytest.importorskip("gudhi")
    from test_topology_finite_primary import config, provenance

    from phycoflow_reconstruction.coherence.families.topology.family import TopologyFamily
    from phycoflow_reconstruction.contracts import DataSpec
    from phycoflow_reconstruction.data.normalization import FieldNormalizer

    fields = ("a", "b", "c", "d", "e")
    bank = QueryBank(fields)
    settings = config()
    settings["fields"] = list(fields)
    settings["components"]["mutual"]["groups"] = [list(pair) for pair in bank.pairs]
    family = TopologyFamily(
        settings, DataSpec(fields, ("1",) * 5, 2, (16,)), FieldNormalizer.identity(5)
    )
    objective = family.spatial_objective
    generator = torch.Generator().manual_seed(22)
    target = torch.randn(1, 5, 4, 4, generator=generator)
    prediction = target + torch.randn(target.shape, generator=generator) * 0.1
    objective.fit_calibration(prediction, target, provenance=provenance())
    selection = AdaptiveCoverageSampler(bank, seed=23).select()
    result = objective(prediction, target, query_selection=selection)
    assert result.diagnostics["scalar_filtrations"] <= 4
    wrong = AdaptiveCoverageSampler(QueryBank(("a", "b", "c", "d", "x")), seed=23).select()
    with pytest.raises(ValueError, match="bank differs"):
        objective(prediction, target, query_selection=wrong)


def test_historical_topology_rebinding_is_explicit_and_strict(tmp_path):
    import hashlib
    from copy import deepcopy

    pytest.importorskip("gudhi")
    from test_topology_finite_primary import coords, family, provenance, witness

    from scripts.rebind_round6_topology_artifact import rebind_topology_artifact

    model = family()
    prediction, reference = witness()
    model.fit_calibration(prediction, reference, coordinates=coords(), provenance=provenance())
    artifact = deepcopy(model.state_artifact())
    artifact["scientific_source"]["source_sha256"]["finite_primary.py"] = "a" * 64
    source = tmp_path / "historical.pt"
    destination = tmp_path / "rebound.pt"
    torch.save(artifact, source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="source/backend"):
        family().load_state_artifact(artifact)
    with pytest.raises(ValueError, match="SHA-256"):
        rebind_topology_artifact(
            source,
            destination,
            expected_sha256="0" * 64,
            family=family(),
            expected_geometry_sha256=artifact["geometry_sha256"],
        )
    with pytest.raises(ValueError, match="geometry identity"):
        rebind_topology_artifact(
            source,
            destination,
            expected_sha256=digest,
            family=family(),
            expected_geometry_sha256="0" * 64,
        )
    rebind_topology_artifact(
        source,
        destination,
        expected_sha256=digest,
        family=family(),
        expected_geometry_sha256=artifact["geometry_sha256"],
    )
    rebound = torch.load(destination, weights_only=True)
    assert rebound["round6_rebinding"]["input_sha256"] == digest
    assert rebound["round6_rebinding"]["old_scientific_source"] == artifact["scientific_source"]
    assert rebound["calibration"] == artifact["calibration"]
    assert rebound["config"] == artifact["config"]
    for key, value in artifact["state_dict"].items():
        assert torch.equal(value, rebound["state_dict"][key])
    family().load_state_artifact(rebound)
