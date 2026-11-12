"""CPU contracts for the versioned conditional Shape observer."""

from copy import deepcopy

import pytest
import torch

pytest.importorskip("gudhi")
from test_topology_finite_primary import config, coords, family, provenance


def setup():
    settings = config()
    settings["persistence"]["definition"] = "conditional_fixed_transform_fair_v1"
    model = family(settings)
    gen = torch.Generator().manual_seed(15)
    y = torch.randn(2, 16, 2, generator=gen)
    x = y + 0.2 * torch.randn(y.shape, generator=gen)
    points = coords(2)
    target = model.fair_prepare_grid(y, coordinates=points)
    source = model.fair_prepare_grid(x, coordinates=points)
    obj = model.spatial_objective
    obj.accumulate_transform(target[:1])
    obj.accumulate_transform(target[1:])
    obj.freeze_transform(
        floors=[1e-4, 1e-4], provenance={"split": "train", "sample_ids": ["train:0", "train:1"]}
    )
    obj.accumulate_calibration(source[:1], target[:1])
    obj.accumulate_calibration(source[1:], target[1:])
    obj.freeze_calibration(provenance=provenance(2))
    return model, x, y, points


def test_fair_identity_symmetry_and_member_gradients():
    model, x, y, points = setup()
    first = x.detach().requires_grad_()
    second = (x.detach() * 0.8 + 0.1).requires_grad_()

    def score(a, b):
        return model(a, y, coordinates=points, context={"second_generated": b})

    result = score(first, second)
    reverse = score(second, first)
    torch.testing.assert_close(result.scalar_loss, reverse.scalar_loss)
    assert score(y, y).scalar_loss == 0
    g1, g2 = torch.autograd.grad(result.scalar_loss, (first, second))
    assert torch.isfinite(g1).all() and torch.isfinite(g2).all()
    assert g1.abs().sum() > 0 and g2.abs().sum() > 0
    assert result.diagnostics["predicted_endpoint_filtrations_per_condition"] == 32
    torch.testing.assert_close(
        torch.tensor(result.diagnostics["attraction_per_sample"])
        - torch.tensor(result.diagnostics["separation_per_sample"]),
        result.per_sample_cost.detach(),
    )


def test_versioned_restore_and_fixed_domain():
    model, x, y, points = setup()
    artifact = deepcopy(model.state_artifact())
    restored = family(artifact["config"])
    restored.load_state_artifact(artifact)
    actual = restored(x, y, coordinates=points, context={"second_generated": x})
    expect = model(x, y, coordinates=points, context={"second_generated": x})
    torch.testing.assert_close(actual.scalar_loss, expect.scalar_loss, rtol=0, atol=0)
    bad = deepcopy(artifact)
    bad["transform"]["scale"][0] *= 2
    with pytest.raises(ValueError, match="transform artifact contents"):
        family(artifact["config"]).load_state_artifact(bad)
    with pytest.raises(ValueError, match="identity mismatch"):
        family().load_state_artifact(artifact)
    mask = torch.ones(4, 4, dtype=torch.bool)
    mask[:, 1] = False
    obj = model.spatial_objective
    grid = model.fair_prepare_grid(y, coordinates=points)
    with pytest.raises(ValueError, match="fixed-domain raster mask"):
        obj(grid, grid, second_generated=grid, valid_mask=mask)


def test_finite_and_essential_distance_domain_contract():
    from phycoflow_reconstruction.coherence.families.topology.persistence import Diagram

    model, _, _, _ = setup()
    obj = model.spatial_objective

    def diagram(bars, births):
        return Diagram(
            finite=torch.tensor(bars, dtype=torch.float32).reshape(-1, 2),
            essential=torch.tensor(births, dtype=torch.float32),
        )

    a = diagram([[0.0, 2.0]], [0.0])
    b = diagram([[0.0, 1.0]], [2.0])
    ab = obj._distance([a, a], [b, b], (1, 1, 4, 4), 16)
    ba = obj._distance([b, b], [a, a], (1, 1, 4, 4), 16)
    torch.testing.assert_close(ab, ba)
    assert ab[0, 0, 0, 0] > 0 and ab[0, 0, 0, 1] == 2
    with pytest.raises(ValueError, match="essential class counts differ"):
        obj._distance([a], [diagram([], [0.0, 1.0])], (1, 1, 4, 4), 16)
