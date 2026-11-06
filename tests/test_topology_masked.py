"""Masked coastal domains: independent Betti counts, gradients and composition."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from scipy.ndimage import label

pytest.importorskip("gudhi")

from phycoflow_reconstruction.coherence.families.topology.family import TopologyFamily
from phycoflow_reconstruction.coherence.families.topology.geometry import build_raster_map
from phycoflow_reconstruction.coherence.families.topology.persistence import cubical_diagrams
from phycoflow_reconstruction.coherence.families.topology.persistence_objective import (
    PersistenceTopologyObjective,
)
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer


def settings(fields=("u", "v")):
    return {
        "strategy": "cubical_persistence",
        "target_use": "paired_supervised",
        "units": "model_units",
        "fields": list(fields),
        "geometry": {"grid_shape": [5, 6], "native_masked": True},
        "filtration": {"smoothing_sigma": 0},
        "components": {
            "self": {"enabled": True},
            "mutual": {"enabled": True, "groups": [list(fields)], "lines": 2},
        },
    }


@pytest.mark.parametrize("seed", [11, 42])
def test_masked_persistence_matches_independent_cubical_euler_counts(seed):
    rng = np.random.default_rng(seed)
    field = rng.normal(size=(7, 9))
    mask = rng.uniform(size=field.shape) > 0.2
    mask[2:4, 3:5] = False
    x = torch.tensor(field[None], requires_grad=True)
    result = cubical_diagrams(x, periodic=False, valid_mask=torch.tensor(mask))[0]
    for threshold in [-2, -1, 0, 1, 2, np.inf]:
        included = mask & (field <= threshold)
        components = label(included)[1]  # four-neighbor connectivity
        vertices = included.sum()
        edges = (included[:-1] & included[1:]).sum() + (included[:, :-1] & included[:, 1:]).sum()
        faces = (
            included[:-1, :-1] & included[1:, :-1] & included[:-1, 1:] & included[1:, 1:]
        ).sum()
        expected = [components, components - vertices + edges - faces]
        for dim in (0, 1):
            d = result[dim]
            count = ((d.finite[:, 0] <= threshold) & (d.finite[:, 1] > threshold)).sum()
            count += (d.essential <= threshold).sum()
            assert int(count) == expected[dim]
            assert torch.isfinite(d.finite).all() and torch.isfinite(d.essential).all()
    assert torch.equal(x.detach(), torch.tensor(field[None]))


def test_masked_objective_ignores_land_values_and_backpropagates_only_through_water():
    config = settings()
    objective = PersistenceTopologyObjective(config, ("u", "v"))
    rng = torch.Generator().manual_seed(9)
    target = torch.randn(1, 2, 5, 6, generator=rng)
    pred = (target + 0.2 * torch.randn(target.shape, generator=rng)).requires_grad_()
    mask = torch.ones(5, 6, dtype=torch.bool)
    mask[2, 2:4] = False
    loss = objective(pred, target, valid_mask=mask).scalar_loss
    gradient = torch.autograd.grad(loss, pred)[0]
    changed_target, changed_pred = target.clone(), pred.detach().clone()
    changed_target[..., ~mask] = 1000
    changed_pred[..., ~mask] = -500
    torch.testing.assert_close(
        loss, objective(changed_pred, changed_target, valid_mask=mask).scalar_loss
    )
    assert torch.isfinite(gradient).all() and gradient[..., mask].abs().sum() > 0
    assert torch.count_nonzero(gradient[..., ~mask]) == 0
    assert objective(target, target, valid_mask=mask).scalar_loss == 0
    cache = {}
    objective(pred, target, valid_mask=mask, reference_cache=cache)
    mask[0, 0] = False
    with pytest.raises(ValueError, match="domain mask changed"):
        objective(pred, target, valid_mask=mask, reference_cache=cache)


def test_curvilinear_native_mapping_roundtrip_and_gradient():
    y, x = torch.meshgrid(torch.arange(5), torch.arange(6), indexing="ij")
    mask = torch.ones(5, 6, dtype=torch.bool)
    mask[2, 2:4] = False
    rc = torch.stack((y[mask], x[mask]), -1)
    coords = torch.stack((x[mask] + 0.03 * y[mask] ** 2, y[mask] + 0.02 * x[mask] ** 2), -1)
    mapping = build_raster_map(
        coords, grid_shape=(5, 6), native_masked=True, native_grid_indices=rc
    )
    assert torch.equal(mapping.valid_mask, mask)
    family = TopologyFamily(
        settings(),
        DataSpec(("u", "v"), ("1", "1"), 2, (int(mask.sum()),)),
        FieldNormalizer.identity(2),
    )
    family._raster_map(coords[None], native_grid_indices=rc)
    target = torch.randn(1, len(coords), 2, generator=torch.Generator().manual_seed(12))
    pred = (target + 0.3 * torch.randn_like(target)).requires_grad_()
    actual = family(pred, target, coordinates=coords[None])
    restored = TopologyFamily(
        settings(), DataSpec(("u", "v"), ("1", "1"), 2, (len(coords),)), FieldNormalizer.identity(2)
    )
    restored.load_state_artifact(deepcopy(family.state_artifact()))
    torch.testing.assert_close(
        actual.scalar_loss, restored(pred, target, coordinates=coords[None]).scalar_loss
    )
    assert torch.isfinite(torch.autograd.grad(actual.scalar_loss, pred)[0]).all()
    assert actual.diagnostics["point_normalization"] == int(mask.sum())
