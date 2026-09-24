"""Scientific and lifecycle checks for the updated topology strategy."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from helpers.spatial import _config, _family, _fields

from phycoflow_reconstruction.coherence.families.topology import exact
from phycoflow_reconstruction.coherence.families.topology.betti_curves import betti_curves
from phycoflow_reconstruction.coherence.families.topology.spatial import (
    anchor_spatial_cost,
    descriptor,
    mutual_spatial_cost,
    self_spatial_costs,
)
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.training.post_training import _component_reports


def test_three_terms_route_gradients_to_generated_fields_and_detach_truth():
    coordinates, prediction, reference = _fields()
    prediction.requires_grad_()
    reference.requires_grad_()
    result = _family()(prediction, reference, coordinates=coordinates)
    assert set(result.component_results) == {
        "topology.self.region",
        "topology.self.connectivity",
        "topology.anchor_self.spatial",
        "topology.mutual.spatial",
    }
    for name in ["anchor_self.spatial", "mutual.spatial"]:
        grad = torch.autograd.grad(
            result.component_results[f"topology.{name}"].scalar_loss, prediction, retain_graph=True
        )[0]
        assert torch.count_nonzero(grad[..., 0]) == 0
        assert grad[..., 1:].norm() > 0
        assert torch.isfinite(grad).all()
    result.scalar_loss.backward()
    assert prediction.grad[..., 0].norm() > 0
    assert prediction.grad[..., 1:].norm() > 0
    assert reference.grad is None


def test_exact_counts_are_evaluation_only_and_do_not_change_the_objective(monkeypatch):
    coordinates, prediction, reference = _fields()
    family = _family()
    original = exact.exact_betti_curves

    def forbidden(*args, **kwargs):
        raise AssertionError("training must not call CPU Betti pairing")

    monkeypatch.setattr(exact, "exact_betti_curves", forbidden)
    training = family(prediction.requires_grad_(), reference, coordinates=coordinates)
    monkeypatch.setattr(exact, "exact_betti_curves", original)
    with torch.no_grad():
        evaluation = family(prediction, reference, coordinates=coordinates)
    assert torch.equal(training.scalar_loss, evaluation.scalar_loss)
    metrics = [name for name in evaluation.component_results if name.endswith("nmae")]
    assert len(metrics) == 6
    report = _component_reports(family, evaluation)
    assert all(report[name]["weighted_contribution"] == 0 for name in metrics)
    assert sum(row["weighted_contribution"] for row in report.values()) == pytest.approx(
        evaluation.scalar_loss.item()
    )


def test_artifact_roundtrip_keeps_geometry_objective_and_version():
    coordinates, prediction, reference = _fields()
    original = _family()
    expected = original(prediction, reference, coordinates=coordinates)
    artifact = original.state_artifact()
    assert artifact["version"] == "2"
    restored = _family()
    restored.load_state_artifact(artifact)
    actual = restored(prediction, reference, coordinates=coordinates)
    assert torch.equal(expected.scalar_loss, actual.scalar_loss)
    for name in ("version", "config"):
        bad = deepcopy(artifact)
        bad[name] = "1" if name == "version" else {}
        with pytest.raises(ValueError, match="artifact.*mismatch"):
            restored.load_state_artifact(bad)


def test_physical_thresholds_use_decoded_fields():
    coordinates, prediction, reference = _fields()
    normalizer = FieldNormalizer(torch.tensor([2.0, -3.0, 4.0]), torch.tensor([3.0, 2.0, 0.5]))
    identity_result = _family()(prediction, reference, coordinates=coordinates)
    encoded_result = _family(normalizer=normalizer)(
        normalizer.encode(prediction), normalizer.encode(reference), coordinates=coordinates
    )
    torch.testing.assert_close(identity_result.scalar_loss, encoded_result.scalar_loss)


@pytest.mark.parametrize("structure", ["component", "hole"])
def test_spatial_descent_repairs_a_missing_structure(structure):
    # A target-only island and a filled-in ring require gradients at locations
    # absent from the prediction's persistence pairs.
    reference = torch.full((1, 1, 16, 16), -0.5)
    reference[:, :, 4:12, 4:12] = 0.5
    prediction = torch.full_like(reference, -0.5)
    if structure == "hole":
        prediction.copy_(reference)
        reference[:, :, 6:10, 6:10] = -0.5
    prediction.requires_grad_()

    def loss():
        return self_spatial_costs(
            prediction,
            reference,
            directions=("superlevel", "sublevel"),
            quantiles=(0.3, 0.5, 0.9),
            physical_levels=(0.0,),
            sharpness=8,
            skeleton_sharpness=16,
            skeleton_iterations=4,
            periodic=True,
        )[0].mean()

    initial_loss = loss().item()
    dimension = 0 if structure == "component" else 1
    levels = torch.tensor([0.0])

    def count(field):
        return betti_curves(field[0, 0], levels, (dimension,), sharpness=8, periodic=True)[
            dimension
        ].item()

    target_count = count(reference)
    assert count(prediction) != target_count
    for _ in range(20):
        gradient = torch.autograd.grad(loss(), prediction)[0]
        with torch.no_grad():
            prediction -= 0.1 * gradient / gradient.abs().max().clamp_min(1e-12)
    assert loss().item() < initial_loss
    assert count(prediction) == target_count


def test_anchor_identity_has_zero_value_and_gradient_and_excludes_constant_truth():
    _, _, fields = _fields()
    reference = fields[..., 0].reshape(1, 12, 12)
    prediction = reference.clone().requires_grad_()
    cost, valid = anchor_spatial_cost(
        prediction, reference, quantiles=(0.3, 0.5, 0.9), sharpness=12
    )
    assert valid.all() and cost.item() == 0
    assert torch.count_nonzero(torch.autograd.grad(cost.sum(), prediction)[0]) == 0
    _, valid = anchor_spatial_cost(
        prediction, torch.zeros_like(reference), quantiles=(0.3, 0.5, 0.9), sharpness=12
    )
    assert not valid.any()


def test_constant_carrier_does_not_disable_anchor_self_and_valid_mean_is_preserved():
    coordinates, prediction, reference = _fields()
    config = _config()
    config["components"]["self"]["enabled"] = False
    config["components"]["mutual"]["enabled"] = False
    reference[..., 0] = 0
    family = _family(config)
    original = family(prediction, reference, coordinates=coordinates)
    assert original.scalar_loss > 0
    batch = family(
        torch.cat((prediction, prediction)),
        torch.cat((reference, reference * 0)),
        coordinates=coordinates.expand(2, -1, -1),
    )
    torch.testing.assert_close(original.scalar_loss, batch.scalar_loss)
    assert batch.component_results["topology.anchor_self.spatial"].valid_mask.tolist() == [
        True,
        False,
    ]
    degenerate = family(prediction, reference * 0, coordinates=coordinates)
    assert not degenerate.component_results["topology.anchor_self.spatial"].available


def test_vorticity_is_offset_invariant_and_periodic_translation_equivariant():
    _, _, reference = _fields()
    grids = reference.reshape(1, 12, 12, 3).permute(0, 3, 1, 2)
    baseline = descriptor(grids, "vorticity", (1, 2), True)
    offset = grids.clone()
    offset[:, 1] += 4
    offset[:, 2] -= 3
    torch.testing.assert_close(descriptor(offset, "vorticity", (1, 2), True), baseline)
    shifted = descriptor(grids.roll((3, 5), (-2, -1)), "vorticity", (1, 2), True)
    torch.testing.assert_close(shifted, baseline.roll((3, 5), (-2, -1)))


def test_mutual_masks_detect_relative_shift_and_backpropagate_through_both_generated_axes():
    _, _, reference = _fields()
    carrier = reference[..., 0].reshape(1, 12, 12)
    anchor = reference[..., 2].reshape(1, 12, 12)
    pred_carrier = carrier.clone().requires_grad_()
    pred_anchor = anchor.roll(2, -1).requires_grad_()
    kwargs = {
        "quantiles": (0.3, 0.5, 0.9),
        "sharpness": 12,
        "directions": ("superlevel", "sublevel"),
        "detach_carrier": False,
    }
    identity, _ = mutual_spatial_cost(carrier, anchor, carrier, anchor, **kwargs)
    shifted, valid = mutual_spatial_cost(pred_carrier, pred_anchor, carrier, anchor, **kwargs)
    assert identity.item() == 0 and shifted.item() > 0 and valid.all()
    gradients = torch.autograd.grad(shifted.sum(), (pred_carrier, pred_anchor))
    assert all(torch.isfinite(gradient).all() and gradient.norm() > 0 for gradient in gradients)


@pytest.mark.parametrize(
    "update,message",
    [
        ({"target_use": "training_reference"}, "paired_supervised"),
        ({"units": "model_units"}, "physical_units"),
        ({"anchor": {"provider": "vorticity", "fields": ["vx"]}}, "arity"),
        ({"anchor": {"provider": "raw", "fields": ["missing"]}}, "outside"),
        (
            {
                "anchor": {
                    "provider": "vorticity",
                    "fields": ["vx", "vy"],
                    "sharpness": float("nan"),
                }
            },
            "finite",
        ),
        ({"filtration": {"level_mode": "physical", "physical_levels": []}}, "physical_levels"),
    ],
)
def test_invalid_spatial_contracts_are_rejected(update, message):
    config = _config()
    config.update(update)
    with pytest.raises(ValueError, match=message):
        _family(config)


def test_nonfinite_fields_fail_before_topology_work():
    coordinates, prediction, reference = _fields()
    prediction[0, 0, 0] = torch.nan
    with pytest.raises(FloatingPointError, match="finite"):
        _family()(prediction, reference, coordinates=coordinates)
