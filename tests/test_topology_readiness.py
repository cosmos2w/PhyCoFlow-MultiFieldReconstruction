"""Phase-1 correctness gates for Betti-curve topology v1."""

from __future__ import annotations

import pytest
import torch

from phycoflow_reconstruction.coherence.families.topology.family import TopologyFamily
from phycoflow_reconstruction.coherence.families.topology.geometry import build_raster_map
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer


def _lattice(size: int = 4, pitch: float = 1.0) -> torch.Tensor:
    axis = torch.arange(size, dtype=torch.float32) * pitch
    return torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1).reshape(-1, 2)


def _config(*, periodic: bool = False, sigma: float = 0.0) -> dict:
    return {
        "fields": ["u", "v"],
        "geometry": {"grid_shape": [4, 4], "periodic": periodic, "neighbors": 1},
        "filtration": {
            "quantiles": [0.25, 0.5, 0.75],
            "dimensions": [0],
            "directions": ["superlevel"],
            "smoothing_sigma": sigma,
        },
        "components": {
            "self": {"enabled": True, "weight": 1.0},
            "mutual": {
                "enabled": False,
                "weight": 0.0,
            },
        },
    }


def _family(config: dict) -> TopologyFamily:
    spec = DataSpec(("u", "v"), ("1", "1"), 2, (4, 4))
    return TopologyFamily(config, spec, FieldNormalizer.identity(2))


def test_period_inference_uses_span_plus_pitch_and_wraps_seam() -> None:
    coordinates = _lattice()
    mapping = build_raster_map(
        coordinates, grid_shape=(4, 4), neighbors=1, periodic=True
    )
    assert mapping.periods == pytest.approx((4.0, 4.0))
    # The last raster column is x=3, and its nearest source is itself rather than
    # an incorrectly shifted span-copy of x=0.
    last_column = mapping.neighbor_indices.reshape(4, 4, 1)[:, -1, 0]
    assert torch.all(coordinates[last_column, 0] == 3)


def test_explicit_periods_are_used_and_unreliable_inference_raises() -> None:
    coordinates = _lattice()
    coordinates[-1, 0] = 2.25
    with pytest.raises(ValueError, match="cannot infer topology period"):
        build_raster_map(coordinates, grid_shape=(4, 4), periodic=True)
    mapping = build_raster_map(
        coordinates, grid_shape=(4, 4), periodic=True, periods=(4.0, 4.0)
    )
    assert mapping.periods == (4.0, 4.0)


def test_period_inference_rejects_incomplete_cartesian_lattice() -> None:
    coordinates = _lattice()[:-1]
    with pytest.raises(ValueError, match="complete endpoint-free Cartesian lattice"):
        build_raster_map(coordinates, grid_shape=(4, 4), periodic=True)


def test_projection_collisions_are_rejected_by_default_and_diagnosed_when_allowed() -> None:
    coordinates = _lattice()
    coordinates[-1] = coordinates[0]
    with pytest.raises(ValueError, match="projection contains coordinate collisions"):
        build_raster_map(coordinates, grid_shape=(4, 4))
    mapping = build_raster_map(
        coordinates, grid_shape=(4, 4), allow_projected_collisions=True
    )
    assert mapping.diagnostics["point_count"] == 16
    assert mapping.diagnostics["unique_projected_count"] == 15
    assert mapping.diagnostics["collision_fraction"] == pytest.approx(1 / 16)
    assert mapping.diagnostics["nearest_spacing_min"] > 0


def test_nonperiodic_smoothing_radius_must_fit_grid() -> None:
    with pytest.raises(ValueError, match="smoothing radius"):
        _family(_config(sigma=1.1))
    _family(_config(periodic=True, sigma=1.1))


def test_near_constant_mutual_axis_raises_explicitly() -> None:
    config = _config()
    config["components"]["self"] = {"enabled": False, "weight": 0.0}
    config["components"]["mutual"] = {
        "enabled": True,
        "weight": 1.0,
        "pairs": [["u", "v"]],
        "lines": 1,
        "axis_tolerance": 1e-5,
    }
    family = _family(config)
    coordinates = _lattice().unsqueeze(0)
    reference = torch.randn(1, 16, 2)
    reference[..., 0] = 2.0 + torch.linspace(0, 1e-7, 16)
    with pytest.raises(ValueError, match="constant or near-constant"):
        family(torch.randn_like(reference), reference, coordinates=coordinates)


def test_collapsed_quantiles_are_deduplicated_and_reported() -> None:
    family = _family(_config())
    coordinates = _lattice().unsqueeze(0)
    reference = torch.zeros(1, 16, 2)
    result = family(reference.clone(), reference, coordinates=coordinates)
    diagnostics = result.component_results["topology.self.betti_curves"].diagnostics
    assert diagnostics["requested_threshold_count"] == 6
    assert diagnostics["unique_threshold_count"] == 2
    assert diagnostics["collapsed_threshold_fraction"] == pytest.approx(2 / 3)
    assert diagnostics["threshold_rule"] == "deduplicate_equal_then_equal_weight_unique"


def test_zero_weight_component_is_skipped_at_runtime() -> None:
    config = _config()
    config["components"]["mutual"] = {
        "enabled": True,
        "weight": 0.0,
        "pairs": [["u", "v"]],
        "lines": 1,
    }
    family = _family(config)
    coordinates = _lattice().unsqueeze(0)
    reference = torch.zeros(1, 16, 2)  # would fail mutual degeneracy if executed
    result = family(reference.clone(), reference, coordinates=coordinates)
    assert "topology.mutual.fibered_betti_curves" not in result.component_results
