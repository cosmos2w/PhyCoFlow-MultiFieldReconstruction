"""Exact metrics and compatibility with the recorded spatial objective."""

from copy import deepcopy

import pytest
import torch
from helpers.spatial import _config, _family, _fields

from phycoflow_reconstruction.coherence.families.topology.betti_curves import gaussian_blur
from phycoflow_reconstruction.coherence.families.topology.exact import (
    curve_statistics,
    exact_betti_curves,
    reference_levels,
)
from phycoflow_reconstruction.coherence.families.topology.geometry import (
    build_raster_map,
    rasterize_fields,
)
from phycoflow_reconstruction.coherence.families.topology.spatial import descriptor
from phycoflow_reconstruction.coherence.families.topology.spatial_objective import (
    SpatialTopologyObjective,
)


def test_recorded_spatial_objectives_and_full_bank_counts():
    # Values measured directly from the source snapshot in topology_source_manifest.json.
    generator = torch.Generator().manual_seed(731)
    reference = gaussian_blur(torch.randn(3, 3, 12, 12, generator=generator), 0.8, True, radius=2)
    prediction = reference.roll(2, -1) + 0.2 * torch.randn(reference.shape, generator=generator)
    config = _config()
    config["filtration"]["physical_levels"] = [-0.4, -0.2, 0, 0.2, 0.4]
    config["anchor"]["quantiles"] = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    config["components"]["mutual"]["lines"] = 16
    objective = SpatialTopologyObjective(config, ("phi", "vx", "vy"))
    result = objective(prediction.requires_grad_(), reference)
    terms = result.component_results
    expected_self = 0.46120938658714294
    expected_anchor = 0.605191171169281
    expected_mutual = 0.8144019246101379
    actual_self = (
        terms["topology.self.region"].scalar_loss
        + 0.25 * terms["topology.self.connectivity"].scalar_loss
    )
    assert actual_self.item() == pytest.approx(expected_self, abs=2e-7)
    assert terms["topology.anchor_self.spatial"].scalar_loss.item() == pytest.approx(
        expected_anchor, abs=2e-7
    )
    assert terms["topology.mutual.spatial"].scalar_loss.item() == pytest.approx(
        expected_mutual, abs=2e-7
    )
    assert result.scalar_loss.item() == pytest.approx(
        expected_self + 0.2 * expected_anchor + expected_mutual, abs=5e-7
    )
    with torch.no_grad():
        stats = objective._mutual_curves(
            descriptor(prediction, "gradient_magnitude", (0,), True),
            descriptor(prediction, "vorticity", (1, 2), True),
            descriptor(reference, "gradient_magnitude", (0,), True),
            descriptor(reference, "vorticity", (1, 2), True),
        )
    assert stats["error_mass"].sum(0).tolist() == [3752, 720]
    assert stats["reference_mass"].sum(0).tolist() == [6936, 1701]


def test_draft_artifact_without_working_tree_provenance_cannot_silently_resume():
    family = _family()
    coordinates, prediction, reference = _fields()
    family(prediction, reference, coordinates=coordinates)
    artifact = family.state_artifact()
    artifact["scientific_source"].pop("working_tree_snapshot_sha256")
    with pytest.raises(ValueError, match="scientific source mismatch"):
        _family().load_state_artifact(artifact)


@pytest.mark.parametrize("periodic", [False, True])
def test_exact_components_holes_seams_and_full_complex(periodic):
    fields = torch.zeros(5, 6, 6)
    fields[1] = 1
    fields[2, 1:5, 1:5] = 1
    fields[2, 2:4, 2:4] = 0  # Contractible ring on either domain.
    fields[3, 2, 0] = fields[3, 2, -1] = 1  # Seam components.
    fields[4, 2] = 1  # One noncontractible loop on the torus.
    result = exact_betti_curves(fields, torch.tensor([0.5]), periodic=periodic)[..., 0]
    assert result.tolist() == [
        [0, 0],
        [1, 2 if periodic else 0],
        [1, 1],
        [1 if periodic else 2, 0],
        [1, 1 if periodic else 0],
    ]
    assert result.dtype == torch.int64 and not result.requires_grad


def test_exact_quantile_oracle_preserves_float64_ties_and_repeated_levels():
    field = torch.zeros(1, 4, 4)
    field[0, 0, 0], field[0, 2, 2] = 1, 2
    quantile = 14 / 15 + 1e-9
    levels = reference_levels(field, (0.0, 0.0, quantile))
    assert levels.dtype == torch.float64
    assert levels[0, -1] > 1 and levels.float()[0, -1] == 1
    exact = exact_betti_curves(field, levels, periodic=True)
    rounded = exact_betti_curves(field, levels.float(), periodic=True)
    assert exact[0, 0].tolist() == [1, 1, 1]
    assert rounded[0, 0, -1] == 2
    assert exact[0, 1, :2].tolist() == [2, 2]


def test_pooled_errors_retain_additive_evidence_instead_of_averaging_ratios():
    ref = torch.zeros(2, 6, 6)
    ref[0, 1, 1] = 1
    ref[1, 1, 1] = ref[1, 4, 4] = 1
    pred = torch.zeros_like(ref)
    pred[1, 1, 1] = 1
    levels = torch.full((2, 1), 0.5)
    stats = curve_statistics([(pred, ref, levels)], periodic=True)
    assert stats["error_mass"][:, 0].tolist() == [1, 1]
    assert stats["reference_mass"][:, 0].tolist() == [1, 2]
    config = _config(size=6)
    config["fields"] = ["phi"]
    config["filtration"]["physical_levels"] = [0.5]
    config["filtration"]["directions"] = ["superlevel"]
    config["components"]["anchor_self"]["enabled"] = False
    config["components"]["mutual"]["enabled"] = False
    coordinates = _fields(6)[0].expand(2, -1, -1)
    prediction = pred.flatten(1)[..., None].expand(-1, -1, 3)
    reference = ref.flatten(1)[..., None].expand(-1, -1, 3)
    with torch.no_grad():
        result = _family(config)(prediction, reference, coordinates=coordinates)
    term = result.component_results["topology.self.h0_nmae"]
    assert term.scalar_loss.item() == pytest.approx(2 / 3)
    assert term.per_sample_cost.mean() == 0.75
    assert term.diagnostics["error_mass"] == 2
    assert term.diagnostics["reference_mass"] == 3


def test_all_sobol_lines_report_poolable_metrics_and_worst_line_guard():
    coordinates, prediction, reference = _fields()
    with torch.no_grad():
        family = _family()
        together = family(
            prediction.repeat(2, 1, 1),
            reference.repeat(2, 1, 1),
            coordinates=coordinates.expand(2, -1, -1),
        )
        single = family(prediction, reference, coordinates=coordinates)
    for dimension in (0, 1):
        path = f"topology.mutual.h{dimension}_nmae"
        a, b = together.component_results[path], single.component_results[path]
        torch.testing.assert_close(a.scalar_loss, b.scalar_loss)
        assert a.diagnostics["error_mass"] == 2 * b.diagnostics["error_mass"]
        assert a.diagnostics["reference_mass"] == 2 * b.diagnostics["reference_mass"]
        assert a.diagnostics["lines_evaluated"] == 4
        assert len(a.diagnostics["line_error_mass"]) == 4
        assert a.diagnostics["line_max_nmae"] >= a.scalar_loss.item() - 1e-6


def test_antialias_uses_all_native_points_and_survives_permutation_and_reload():
    coordinates, prediction, reference = _fields(12)
    config = _config(size=6)
    config["geometry"]["antialias_downsample"] = True
    family = _family(config)
    result = family(prediction, reference, coordinates=coordinates)
    expected = torch.nn.functional.avg_pool2d(
        prediction.reshape(1, 12, 12, 3).permute(0, 3, 1, 2), 2
    )
    actual = rasterize_fields(prediction, family.neighbor_indices, family.neighbor_weights, (6, 6))
    torch.testing.assert_close(actual, expected)
    assert family.geometry_diagnostics["antialias_applied"] == 1
    live = prediction.clone().requires_grad_()
    rasterize_fields(
        live, family.neighbor_indices, family.neighbor_weights, (6, 6)
    ).sum().backward()
    assert torch.all(live.grad == 0.25)
    order = torch.randperm(144, generator=torch.Generator().manual_seed(17))
    shuffled = _family(config)(
        prediction[:, order], reference[:, order], coordinates=coordinates[:, order]
    )
    torch.testing.assert_close(result.scalar_loss, shuffled.scalar_loss)
    restored = _family(config)
    restored.load_state_artifact(family.state_artifact())
    torch.testing.assert_close(
        result.scalar_loss, restored(prediction, reference, coordinates=coordinates).scalar_loss
    )


def test_sparse_antialias_reports_interpolation_fallback():
    coordinates = _fields(12)[0][0, :100]
    mapping = build_raster_map(
        coordinates, grid_shape=(6, 6), periodic=True, periods=(1.0, 1.0), antialias_downsample=True
    )
    assert mapping.diagnostics["antialias_applied"] == 0


def test_autocast_does_not_round_topology_smoothing_or_gradients():
    coordinates, prediction, reference = _fields()
    config = _config()
    config["filtration"]["smoothing_sigma"] = 0.8
    family = _family(config)
    prediction.requires_grad_()
    expected = family(prediction, reference, coordinates=coordinates)
    expected_gradient = torch.autograd.grad(expected.scalar_loss, prediction)[0]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = family(prediction, reference, coordinates=coordinates)
    actual_gradient = torch.autograd.grad(actual.scalar_loss, prediction)[0]
    assert torch.equal(expected.scalar_loss, actual.scalar_loss)
    assert torch.equal(expected_gradient, actual_gradient)


def test_mutual_backward_scale_changes_only_applied_gradient():
    coordinates, prediction, reference = _fields()
    config = _config()
    config["components"]["self"]["enabled"] = False
    config["components"]["anchor_self"]["enabled"] = False
    unscaled = deepcopy(config)
    unscaled["components"]["mutual"]["gradient_scale"] = 1
    prediction.requires_grad_()
    scaled = _family(config)(prediction, reference, coordinates=coordinates)
    full = _family(unscaled)(prediction, reference, coordinates=coordinates)
    assert torch.equal(scaled.scalar_loss, full.scalar_loss)
    scaled_gradient = torch.autograd.grad(scaled.scalar_loss, prediction)[0]
    full_gradient = torch.autograd.grad(full.scalar_loss, prediction)[0]
    torch.testing.assert_close(scaled_gradient, 0.2 * full_gradient)
    assert torch.count_nonzero(full_gradient[..., 0]) == 0
