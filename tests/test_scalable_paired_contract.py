"""Deterministic scoring and frozen Shape artifact contracts."""

from copy import deepcopy

import pytest
import torch

from phycoflow_reconstruction.coherence.query_sampling import Draw, QueryBatch
from phycoflow_reconstruction.coherence.shape_compute import LazyPooledShape
from phycoflow_reconstruction.coherence.state_compute import (
    PAIRED_VERSION,
    score_family,
    select_directions,
)


def test_paired_state_one_field_exact_gradient_and_explicit_mode():
    bank = select_directions(torch.ones(1, 1), torch.ones(1), ["x"], 17)
    x = torch.tensor([[[0.0], [2.0], [4.0]]], dtype=torch.double, requires_grad=True)
    y = torch.tensor([[[0.0], [1.0], [3.0]]], dtype=torch.double)
    w = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.double)
    result, blocks = score_family(x[None], y, w, w, bank, mode="paired")
    assert result.diagnostics["score_definition"] == PAIRED_VERSION
    assert blocks["state.mixed"] is None
    assert torch.isfinite(torch.autograd.grad(result.scalar_loss, x)[0]).all()
    equal = x.detach().clone().requires_grad_()
    coincident, _ = score_family(equal[None], equal.detach(), w, w, bank, mode="paired")
    g = torch.autograd.grad(coincident.scalar_loss, equal)[0]
    assert coincident.scalar_loss == 0 and torch.equal(g, torch.zeros_like(g))
    with pytest.raises(ValueError, match="conditional State"):
        score_family(x[None], y, w, w, bank)
    with pytest.raises(ValueError, match="paired State"):
        score_family(torch.stack((x, x)), y, w, w, bank, mode="paired")


def test_paired_shape_and_frozen_artifact_validation():
    pytest.importorskip("gudhi")
    torch.manual_seed(42)
    x = torch.randn(1, 1, 4, 4, dtype=torch.double, requires_grad=True)
    y = torch.randn_like(x)
    batch = QueryBatch((Draw(0, 1, 1.0, 1, 1),), (), 1)
    shape = LazyPooledShape([0.0], [1.0], dimensions=(0, 1), projections=4)
    shape.accumulate(x.detach(), y, batch)
    artifact = shape.freeze(
        provenance={"split": "train", "sample_ids": ["a"], "source_checkpoint_sha256": "a" * 64}
    )
    paired = shape.score(x, None, y, batch, exhaustive=True, mode="paired")
    # The numerical identity follows from the formula; the paired definition is explicit.
    fair = shape.score(x, x, y, batch, exhaustive=True)
    torch.testing.assert_close(paired["self"], fair["self"])
    gp = torch.autograd.grad(paired["self"].sum(), x, retain_graph=True)[0]
    gf = torch.autograd.grad(fair["self"].sum(), x)[0]
    torch.testing.assert_close(gp, gf)
    same = y.clone().requires_grad_()
    identity = shape.score(same, None, y, batch, exhaustive=True, mode="paired")["self"]
    grad = torch.autograd.grad(identity.sum(), same)[0]
    assert torch.isfinite(grad).all() and torch.count_nonzero(grad) == 0
    with pytest.raises(ValueError, match="conditional Shape"):
        shape.score(x, None, y, batch, exhaustive=True)
    restored = LazyPooledShape.from_state_dict(deepcopy(artifact))
    torch.testing.assert_close(
        restored.score(x, None, y, batch, exhaustive=True, mode="paired")["self"], paired["self"]
    )
    for damage in ("scale", "missing", "split", "domain"):
        broken = deepcopy(artifact)
        if damage == "scale":
            broken["scales"][0][1][0][0] = float("nan")
        if damage == "missing":
            broken["scales"].pop()
        if damage == "split":
            broken["provenance"]["split"] = "test"
        if damage == "domain":
            broken["domain"] = ((1, 4), None, False)
        with pytest.raises(ValueError):
            LazyPooledShape.from_state_dict(broken)


def test_conditional_shape_cache_rejects_changed_frozen_transform_and_stale_source():
    from test_topology_conditional_fair import setup
    from test_topology_finite_primary import family

    model, x, y, points = setup()
    artifact = deepcopy(model.state_artifact())
    artifact["scientific_source"]["source_sha256"]["family.py"] = "0" * 64
    restored = family(artifact["config"])
    with pytest.raises(ValueError, match="source/backend mismatch"):
        restored.load_state_artifact(artifact)
    restored.load_state_artifact(model.state_artifact())
    cache = {"persistence_cache": {}, "second_generated": x}
    restored(x, y, coordinates=points, context=cache)
    with torch.no_grad():
        restored.spatial_objective.descriptor_mean.add_(1)
    with pytest.raises(ValueError, match="target/coordinates changed"):
        restored(x, y, coordinates=points, context=cache)
