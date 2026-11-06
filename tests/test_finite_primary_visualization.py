"""Saved-data Shape views use the scored signed bank and fixed domain."""

import json
from copy import deepcopy

import numpy as np
import pytest
import torch

pytest.importorskip("gudhi")

from phycoflow_reconstruction.coherence.families.topology.family import TopologyFamily
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.evaluation.finite_primary_visualization import (
    _event_metrics,
    render_finite_primary_payload,
    save_finite_primary_payload,
)


def _fixture():
    config = {
        "strategy": "cubical_persistence", "target_use": "paired_supervised",
        "units": "model_units", "fields": ["a", "b"],
        "geometry": {"grid_shape": [4, 4], "native_masked": True},
        "filtration": {"smoothing_sigma": 0, "dimensions": [0, 1]},
        "persistence": {"definition": "finite_primary_v1", "projections": 8,
                        "calibration": {"finite_floor": 1e-3, "essential_floor": 2e-3}},
        "components": {"self": {"enabled": True},
                       "mutual": {"enabled": True, "groups": [["a", "b"]], "lines": 3}},
    }
    mask = torch.ones(4, 4, dtype=torch.bool)
    mask[1, 1] = False
    y, x = torch.meshgrid(torch.arange(4.), torch.arange(4.), indexing="ij")
    coordinates = torch.stack((x, y), -1)
    rng = torch.Generator().manual_seed(8)
    target_grid = torch.randn(4, 4, 2, generator=rng)
    pred_grid = target_grid + .35 * torch.randn(4, 4, 2, generator=rng)
    target = target_grid[mask][None]
    pred = pred_grid[mask][None]
    query = coordinates[mask][None]
    family = TopologyFamily(
        config, DataSpec(("a", "b"), ("1", "1"), 2, (15,)), FieldNormalizer.identity(2)
    )
    family.fit_calibration(
        pred, target, coordinates=query,
        provenance={"split": "train", "sample_ids": ["train:0"], "source_checkpoint_sha256": "a" * 64},
    )
    event = {
        "reference": target_grid.permute(2, 0, 1).numpy(),
        "prediction": pred_grid.permute(2, 0, 1).numpy(),
        "coordinates": coordinates.numpy(), "valid_mask": mask.numpy(),
        "weights": np.arange(1., 17.).reshape(4, 4), "field_names": ("a", "b"),
        "pair": ("a", "b"), "signs": (1, -1), "thresholds": (0., .2),
        "units": "unknown", "measure": "declared physical area",
    }
    return family, pred, target, query, event


def test_payload_contains_actual_full_bank_and_roundtrip_visuals(tmp_path):
    family, pred, target, coords, event = _fixture()
    result = family(pred, target, coordinates=coords)
    path = save_finite_primary_payload(
        family, pred, target, coordinates=coords, output_path=tmp_path / "payload.npz",
        event=event, sample_id="held-out:0", endpoint="source",
    )
    with np.load(path, allow_pickle=False) as data:
        assert len(data["unit_names"]) == 16
        for i, name in enumerate(data["unit_names"]):
            for j, dim in enumerate(data["dimensions"]):
                for k, kind in enumerate(("finite", "essential")):
                    score = result.component_results[f"topology.{name}.h{dim}.{kind}"].per_sample_cost
                    np.testing.assert_allclose(data["raw_leaves"][i, j, k], score[0].item(), rtol=1e-6, atol=1e-7)
        assert json.loads(str(data["metadata"].item()))["sample_id"] == "held-out:0"
        assert not data["valid_mask"][1, 1]
        assert data["prediction_betti"].shape == (16, 2, 5)
        metrics = _event_metrics(data["event_reference"], data["event_prediction"],
                                 data["event_valid_mask"], data["event_weights"])
        assert metrics["missing"] >= 0 and metrics["excess"] >= 0
        assert metrics["iou"] is None or 0 <= metrics["iou"] <= 1
        names = tuple(str(name) for name in data["unit_names"][[0, 4, 7, 10, 13]])
    outputs = render_finite_primary_payload(path, tmp_path / "figures", units=names, dpi=60)
    assert len(outputs) == 12
    from PIL import Image
    for output in outputs:
        assert output.stat().st_size > 100
        if output.suffix == ".png":
            with Image.open(output) as image:
                image.verify()
        else:
            assert output.read_bytes().startswith(b"%PDF")
    with pytest.raises(ValueError, match="filtration names"):
        render_finite_primary_payload(path, tmp_path / "figures", units=("wrong",))


def test_event_empty_union_and_missing_excess_are_distinct():
    mask = np.ones((2, 2), dtype=bool)
    weights = np.array([[1., 2.], [3., 4.]])
    empty = np.zeros((2, 2), dtype=bool)
    assert _event_metrics(empty, empty, mask, weights)["iou"] is None
    truth = np.array([[True, False], [False, False]])
    prediction = np.array([[False, True], [False, False]])
    result = _event_metrics(truth, prediction, mask, weights)
    assert result["missing"] == .1 and result["excess"] == .2
    assert result["iou"] == 0 and result["precision"] == 0 and result["recall"] == 0


def test_event_rejects_unsupported_geometry_and_metadata(tmp_path):
    family, pred, target, coords, event = _fixture()
    bad = deepcopy(event)
    bad["pair"] = ("a", "absent")
    with pytest.raises(ValueError, match="named pair"):
        save_finite_primary_payload(family, pred, target, coordinates=coords, output_path=tmp_path / "bad.npz", event=bad)
    bad = deepcopy(event)
    bad["coordinates"][1, 2, 0] += .1
    with pytest.raises(ValueError, match="rectilinear"):
        save_finite_primary_payload(family, pred, target, coordinates=coords, output_path=tmp_path / "bad.npz", event=bad)
