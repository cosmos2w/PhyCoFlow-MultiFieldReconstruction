"""Physical raster geometry and fixed-domain exact topology contracts."""

import json
from copy import deepcopy

import numpy as np
import pytest
import torch
from scipy.spatial.distance import cdist

from phycoflow_reconstruction.coherence.families.topology.exact import (
    exact_betti_curves,
    reference_levels,
)
from phycoflow_reconstruction.coherence.families.topology.family import TopologyFamily
from phycoflow_reconstruction.coherence.families.topology.geometry import (
    build_physical_cell_overlap_map,
    build_raster_map,
    rasterize_fields,
)
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer


def _grid():
    y, x = torch.meshgrid(
        torch.tensor([0., 2., 5.]), torch.tensor([0., 1., 3., 6.]), indexing="ij"
    )
    return torch.stack((x, y), -1).reshape(-1, 2)


def _oracle(values):
    wx = np.array([[.5, 1.5, 1., 0.], [0., 0., 1.5, 1.5]]) / 3
    wy = np.array([[1., 1.5, 0.], [0., 1., 1.5]]) / 2.5
    return wy @ values.reshape(3, 4) @ wx.T


def test_nondivisible_nonuniform_overlap_physical_averages_and_gradients():
    coords = _grid()
    order = torch.tensor([5, 8, 1, 11, 0, 4, 3, 2, 6, 10, 7, 9])
    selected = torch.tensor([9, 2, 7, 5, 11, 0, 4, 3, 8, 10, 6, 1])
    native = coords[order]
    mapping = build_physical_cell_overlap_map(native, selected, grid_shape=(2, 2))
    original_values = torch.arange(12, dtype=torch.float32).square()
    values = original_values[order][selected].reshape(1, 12, 1).requires_grad_()
    got = rasterize_fields(values, mapping.neighbor_indices, mapping.neighbor_weights, (2, 2))
    np.testing.assert_allclose(got.detach().numpy()[0, 0], _oracle(original_values.numpy()), rtol=1e-6)
    assert mapping.diagnostics["mapping"] == "physical_cell_overlap_v1"
    assert mapping.diagnostics["antialias_applied"] == 1
    np.testing.assert_allclose(mapping.grid_coordinates.numpy()[0, 0], [1.5, 1.25])
    grad = torch.autograd.grad(got.sum(), values)[0]
    expected = np.empty(12)
    for i in range(12):
        indicator = np.zeros(12)
        indicator[i] = 1
        expected[i] = _oracle(indicator).sum()
    np.testing.assert_allclose(grad.numpy()[0, :, 0], expected[order.numpy()][selected.numpy()], rtol=1e-6)


def test_sparse_surrogate_matches_explicit_native_idw_then_area_average():
    coords = _grid()
    query = torch.tensor([0, 2, 5, 7, 8, 11])
    mapping = build_physical_cell_overlap_map(coords, query, grid_shape=(2, 2), neighbors=3)
    values = np.array([3., 1., 8., -2., 2., 7.])
    distances = cdist(coords.numpy(), coords[query].numpy())
    nearest = np.argsort(distances, axis=1)[:, :3]
    approx = np.empty(len(coords))
    for i, ids in enumerate(nearest):
        d = distances[i, ids]
        if np.any(d < 1e-12):
            approx[i] = values[ids[np.argmin(d)]]
        else:
            w = d ** -2
            approx[i] = np.dot(w, values[ids]) / w.sum()
    actual = rasterize_fields(
        torch.tensor(values)[None, :, None], mapping.neighbor_indices,
        mapping.neighbor_weights.double(), (2, 2),
    )
    np.testing.assert_allclose(actual.numpy()[0, 0], _oracle(approx), rtol=1e-6)
    assert mapping.diagnostics["mapping"] == "sparse_idw_then_physical_cell_overlap_v1"
    assert mapping.diagnostics["antialias_applied"] == 0
    assert mapping.diagnostics["sparse_neighbors"] == 3
    json.dumps(mapping.diagnostics)
    assert mapping.diagnostics["sparse_max_nearest_raw_distance"] > 0


def test_geometry_rejects_unsupported_support_and_binds_query_identity():
    coords = _grid()
    with pytest.raises(ValueError, match="complete rectilinear"):
        build_physical_cell_overlap_map(coords[:-1], torch.arange(11), grid_shape=(2, 2))
    with pytest.raises(ValueError, match="unique native indices"):
        build_physical_cell_overlap_map(coords, torch.tensor([0, 1, 1, 2]), grid_shape=(2, 2))
    config = {
        "strategy": "cubical_persistence", "target_use": "paired_supervised", "fields": ["u"],
        "geometry": {"grid_shape": [2, 2], "raster_mapping": "physical_cell_overlap_v1"},
        "filtration": {"smoothing_sigma": 0},
        "components": {"self": {"enabled": True}, "mutual": {"enabled": False}},
    }
    pytest.importorskip("gudhi")
    family = TopologyFamily(config, DataSpec(("u",), ("1",), 2, (12,)), FieldNormalizer.identity(1))
    query = torch.tensor([0, 2, 5, 7, 8, 11])
    model_coords = coords[query] / 10
    with pytest.raises(ValueError, match="fit physical topology raster"):
        family._raster_map(model_coords[None])
    family.fit_physical_raster(coords, query, query_coordinates=model_coords)
    family._raster_map(model_coords[None])
    with pytest.raises(ValueError, match="coordinates changed"):
        family._raster_map((model_coords + 1)[None])
    restored = TopologyFamily(config, DataSpec(("u",), ("1",), 2, (12,)), FieldNormalizer.identity(1))
    restored.load_state_artifact(deepcopy(family.state_artifact()))
    assert restored.geometry_diagnostics == family.geometry_diagnostics
    torch.testing.assert_close(restored.neighbor_weights, family.neighbor_weights)


def test_preceding_legacy_persistence_artifact_still_replays():
    pytest.importorskip("gudhi")
    config = {
        "strategy": "cubical_persistence", "target_use": "paired_supervised", "fields": ["u"],
        "geometry": {"grid_shape": [3, 4]}, "filtration": {"smoothing_sigma": 0},
        "components": {"self": {"enabled": True}, "mutual": {"enabled": False}},
    }
    spec = DataSpec(("u",), ("1",), 2, (12,))
    family = TopologyFamily(config, spec, FieldNormalizer.identity(1))
    coords = _grid()[None]
    target = torch.arange(12., dtype=torch.float32)[None, :, None]
    pred = target.roll(1, dims=1)
    expected = family(pred, target, coordinates=coords).scalar_loss
    artifact = deepcopy(family.state_artifact())
    artifact["scientific_source"]["source_sha256"].update({
        "family.py": "0c80a1a350939dc143b594990216fa454f28526f467adb354befd140543edfda",
        "geometry.py": "eed9bb67a0aac7114982d5d1af1055ea5780a1d365a0584611a317b0dc9370c7",
    })
    restored = TopologyFamily(config, spec, FieldNormalizer.identity(1))
    restored.load_state_artifact(artifact)
    torch.testing.assert_close(restored(pred, target, coordinates=coords).scalar_loss, expected, rtol=0, atol=0)
    artifact["scientific_source"]["source_sha256"]["persistence.py"] = "0" * 64
    with pytest.raises(ValueError, match="source/backend mismatch"):
        restored.load_state_artifact(artifact)


def test_native_masked_accepts_complete_nonuniform_rectilinear_grid():
    coords = _grid()
    mapping = build_raster_map(coords, grid_shape=(3, 4), native_masked=True)
    assert mapping.valid_mask.all()
    values = torch.arange(12.)[None, :, None]
    torch.testing.assert_close(
        rasterize_fields(values, mapping.neighbor_indices, mapping.neighbor_weights, (3, 4))[0, 0],
        values.reshape(3, 4),
    )


def test_masked_betti_matches_persistence_including_hole_and_ties():
    pytest.importorskip("gudhi")
    from phycoflow_reconstruction.coherence.families.topology.persistence import cubical_diagrams

    mask = torch.ones(5, 6, dtype=torch.bool)
    mask[2, 2:4] = False
    field = torch.tensor([[1, 1, 2, 2, 1, 0], [1, 2, 2, 2, 1, 0],
                          [1, 2, 99, -99, 1, 0], [1, 2, 2, 2, 1, 0],
                          [1, 1, 1, 1, 1, 0]], dtype=torch.float64)[None]
    levels = torch.tensor([0., 1., 2., 3.], dtype=torch.float64)
    curves = exact_betti_curves(field, levels, periodic=False, valid_mask=mask)[0]
    diagrams = cubical_diagrams(-field, periodic=False, valid_mask=mask)[0]
    for j, level in enumerate(levels):
        for dim in (0, 1):
            diagram = diagrams[dim]
            threshold = -level
            count = ((diagram.finite[:, 0] <= threshold) & (diagram.finite[:, 1] > threshold)).sum()
            count += (diagram.essential <= threshold).sum()
            assert int(curves[dim, j]) == int(count)
    assert curves[1, 0] == 1
    actual = reference_levels(field, (0.5,), valid_mask=mask)
    assert actual.item() == np.quantile(field[0][mask].numpy(), .5)
    with pytest.raises(ValueError, match="masked periodic"):
        exact_betti_curves(field, levels, periodic=True, valid_mask=mask)
