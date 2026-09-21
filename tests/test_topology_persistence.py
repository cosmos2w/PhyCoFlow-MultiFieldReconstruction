"""Independent mathematical, gradient, fidelity and lifecycle regressions."""

from copy import deepcopy

import numpy as np
import pytest
import torch

gudhi = pytest.importorskip("gudhi")

from helpers.persistence import config

from phycoflow_reconstruction.coherence.families.topology.persistence import (
    cubical_diagrams,
    sliced_diagram_distance,
)
from phycoflow_reconstruction.coherence.families.topology.persistence_objective import (
    PersistenceTopologyObjective,
)
from phycoflow_reconstruction.training.retention import endpoint_retention, post_training_mode
from phycoflow_reconstruction.training.topology_selection import fidelity_eligibility


@pytest.mark.parametrize("periodic", [True, False])
def test_critical_vertices_reproduce_independent_intervals(periodic):
    field = torch.randn(2, 7, 9, dtype=torch.float64, generator=torch.Generator().manual_seed(9))
    diagrams = cubical_diagrams(field, periodic=periodic)
    for i in range(2):
        cc = gudhi.PeriodicCubicalComplex(
            vertices=field[i].numpy(), periodic_dimensions=[periodic, periodic]
        )
        cc.compute_persistence()
        for d in (0, 1):
            intervals = cc.persistence_intervals_in_dimension(d)
            actual = diagrams[i][d]
            finite = intervals[np.isfinite(intervals[:, 1])]
            np.testing.assert_allclose(
                sorted(map(tuple, actual.finite.numpy())), sorted(map(tuple, finite))
            )
            np.testing.assert_allclose(
                sorted(actual.essential.numpy()),
                sorted(intervals[~np.isfinite(intervals[:, 1]), 0]),
            )
    assert diagrams[0][0].essential.numel() == 1
    assert diagrams[0][1].essential.numel() == (2 if periodic else 0)


def test_pairing_and_projection_autograd_agrees_with_finite_difference():
    g = torch.Generator().manual_seed(94)
    target = torch.randn(1, 5, 6, generator=g, dtype=torch.float64)
    pred = (
        target + 0.13 * torch.randn(target.shape, generator=g, dtype=torch.float64)
    ).requires_grad_()
    ref = cubical_diagrams(target, periodic=True)[0]

    def loss(x):
        diagrams = cubical_diagrams(x, periodic=True)[0]
        return sum(sliced_diagram_distance(diagrams[d], ref[d], projections=8) for d in (0, 1))

    assert torch.autograd.gradcheck(loss, (pred,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_identity_stationary_and_mutual_reaches_all_fields():
    objective = PersistenceTopologyObjective(config(), ("a", "b", "c"))
    target = torch.randn(2, 3, 9, 9, generator=torch.Generator().manual_seed(91))
    pred = target.clone().requires_grad_()
    result = objective(pred, target)
    assert result.scalar_loss.item() == 0
    assert torch.count_nonzero(torch.autograd.grad(result.scalar_loss, pred)[0]) == 0
    pred = (target + 0.2 * torch.randn_like(target)).requires_grad_()
    mutual = objective(pred, target).component_results["topology.mutual.persistence"].scalar_loss
    grad = torch.autograd.grad(mutual, pred)[0]
    assert (grad.abs().sum((0, 2, 3)) > 0).all()
    assert torch.isfinite(grad).all()


def test_relative_field_shift_preserves_self_but_changes_joint_structure():
    objective = PersistenceTopologyObjective(config(), ("a", "b", "c"))
    target = torch.randn(1, 3, 9, 9, generator=torch.Generator().manual_seed(92))
    pred = target.clone()
    pred[:, 1] = pred[:, 1].roll(3, -1)
    result = objective(pred, target)
    assert result.component_results["topology.self.persistence"].scalar_loss.item() == 0
    assert result.component_results["topology.mutual.persistence"].scalar_loss.item() > 0


def test_endpoint_can_create_a_missing_component_while_ph_alone_is_sparse():
    target = torch.zeros(1, 8, 8)
    target[:, 2:5, 2:5] = 1
    prediction = torch.zeros_like(target).requires_grad_()
    before = cubical_diagrams(-prediction, periodic=True)[0][0]
    optimizer = torch.optim.SGD([prediction], lr=1.0)
    for _ in range(50):
        optimizer.zero_grad()
        endpoint_retention(
            prediction.flatten(1)[..., None],
            target.flatten(1)[..., None],
            scale_reference=target.flatten(1)[..., None],
        ).backward()
        optimizer.step()
    assert prediction[:, 3, 3].item() > 0.9
    # Superlevel foreground was empty and now has the target component.
    after = cubical_diagrams(-prediction, periodic=True)[0][0]
    assert before.essential.item() == 0
    assert after.essential.item() < -0.9


def test_eval_mode_has_gradients_and_disables_dropout():
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(0.9))
    post_training_mode(model, {})
    x = torch.ones(2, 3)
    assert torch.equal(model(x), model(x))
    model(x).sum().backward()
    assert model[0].weight.grad.abs().sum() > 0
    post_training_mode(model, {"optimization": {"model_mode": "train"}})
    assert model.training


def test_fidelity_gate_rejects_hidden_field_regression_and_nonfinite():
    source = {"mse_normalized": 1.0, "per_field_mse_normalized": {"a": 2.0, "b": 0.01}}
    candidate = deepcopy(source)
    candidate["mse_normalized"] = 0.9
    candidate["per_field_mse_normalized"]["b"] = 0.02
    budget = {"max_relative_mse_increase": 0.05, "max_relative_field_mse_increase": 0.05}
    assert not fidelity_eligibility(source, candidate, budget)["eligible"]
    candidate["per_field_mse_normalized"]["b"] = float("nan")
    assert not fidelity_eligibility(source, candidate, budget)["eligible"]
    assert fidelity_eligibility(source, source, budget)["eligible"]


def test_derived_curl_uses_sign_and_physical_spacing_and_has_joint_gradients():
    cfg = config()
    cfg.update(units="physical_units", geometry={"periodic": True, "periods": [2.0, 3.0]})
    cfg["persistence"] = {
        "descriptors": {"curl": {"provider": "signed_vorticity", "fields": ["b", "c"]}}
    }
    cfg["components"]["mutual"]["groups"] = [["a", "curl"]]
    objective = PersistenceTopologyObjective(cfg, ("a", "b", "c"))
    axis = torch.arange(12) * (2 * torch.pi / 12)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    target = torch.stack((torch.cos(x + y), torch.sin(y), torch.sin(x)))[None]
    curl = objective._descriptor_fields(target)[:, -1]
    expected = (torch.sin(x + 2 * torch.pi / 12) - torch.sin(x - 2 * torch.pi / 12)) / (2 * 2 / 12)
    expected -= (torch.sin(y + 2 * torch.pi / 12) - torch.sin(y - 2 * torch.pi / 12)) / (2 * 3 / 12)
    torch.testing.assert_close(curl[0], expected, atol=2e-6, rtol=2e-6)
    assert curl.min() < 0 < curl.max()
    pred = (target + 0.15 * torch.randn_like(target)).requires_grad_()
    loss = objective(pred, target).component_results["topology.mutual.persistence"].scalar_loss
    gradient = torch.autograd.grad(loss, pred)[0]
    assert (gradient.abs().sum((0, 2, 3)) > 0).all()


def test_phase_vorticity_mutual_keeps_selected_self_fields_and_all_gradients():
    cfg = config()
    cfg.update(units="physical_units", geometry={"periodic": True, "periods": [2.0, 3.0]})
    cfg["persistence"] = {
        "descriptors": {"vorticity": {"provider": "signed_vorticity", "fields": ["b", "c"]}}
    }
    cfg["components"]["self"]["fields"] = ["a", "b", "c"]
    cfg["components"]["mutual"].update(groups=[["a", "vorticity"]], lines=4)
    objective = PersistenceTopologyObjective(cfg, ("a", "b", "c"))
    y = torch.randn(8, 3, 9, 11, generator=torch.Generator().manual_seed(71), dtype=torch.float64)
    x = (y + 0.1 * y.roll(2, -1)).requires_grad_()
    result = objective(x, y)
    expected = {
        f"topology.{group}.h{dim}"
        for group in ["self.a", "self.b", "self.c", "mutual.a+vorticity"]
        for dim in (0, 1)
    }
    from phycoflow_reconstruction.training.topology_constraints import leaf_components

    assert set(leaf_components(result)) == expected
    assert result.diagnostics["scalar_filtrations"] == 14
    gradient = torch.autograd.grad(
        result.component_results["topology.mutual.persistence"].scalar_loss, x
    )[0]
    assert torch.isfinite(gradient).all()
    assert (gradient.abs().sum((0, 2, 3)) > 0).all()
    # A constant velocity offset has zero periodic curl, including at seams.
    shifted = y.clone()
    shifted[:, 1] += 2.0
    shifted[:, 2] -= 3.0
    torch.testing.assert_close(
        objective._descriptor_fields(shifted)[:, -1],
        objective._descriptor_fields(y)[:, -1],
        rtol=1e-12,
        atol=1e-12,
    )
    for fields in [[], ["a", "a"], ["unknown"]]:
        cfg["components"]["self"]["fields"] = fields
        with pytest.raises(ValueError, match="self.fields"):
            PersistenceTopologyObjective(cfg, ("a", "b", "c"))


def test_family_training_evaluation_state_roundtrip_and_weight_reporting():
    from phycoflow_reconstruction.coherence.families.topology.family import TopologyFamily
    from phycoflow_reconstruction.contracts import DataSpec
    from phycoflow_reconstruction.data.normalization import FieldNormalizer
    from phycoflow_reconstruction.training.post_training import _component_reports

    cfg = config()
    cfg["geometry"].update(grid_shape=[6, 6], antialias_downsample=True)
    spec = DataSpec(("a", "b", "c"), ("1",) * 3, 2, (6, 6))
    family = TopologyFamily(cfg, spec, FieldNormalizer.identity(3))
    y, x = torch.meshgrid(torch.arange(6) / 6, torch.arange(6) / 6, indexing="ij")
    coords = torch.stack((x, y), -1).reshape(1, 36, 2)
    target = torch.randn(2, 36, 3)
    pred = (target + 0.1 * torch.randn_like(target)).requires_grad_()
    train = family(pred, target, coordinates=coords.expand(2, -1, -1))
    restored = TopologyFamily(cfg, spec, FieldNormalizer.identity(3))
    restored.load_state_artifact(family.state_artifact())
    with torch.no_grad():
        evaluation = restored(pred, target, coordinates=coords.expand(2, -1, -1))
    torch.testing.assert_close(train.scalar_loss, evaluation.scalar_loss, atol=0, rtol=0)
    report = _component_reports(family, train)
    assert sum(term["weighted_contribution"] for term in report.values()) == pytest.approx(
        train.scalar_loss.item()
    )
    assert all(
        term["weighted_contribution"] == 0
        for key, term in report.items()
        if key.endswith(("h0", "h1"))
    )
    from phycoflow_reconstruction.training.post_training import _component_history_report

    history = _component_history_report(train, {"topology": family})
    contributions = {
        key: value for key, value in history.items() if key.endswith("/weighted_contribution")
    }
    assert sum(contributions.values()) == pytest.approx(train.scalar_loss.item())
    assert all(value == 0 for key, value in contributions.items() if ".h0/" in key or ".h1/" in key)


def test_rollout_contract_detects_inference_operator_mismatch_and_preserves_rng():
    from phycoflow_reconstruction.contracts import (
        ModelCapabilities,
        ObservationBatch,
        ReconstructionBatch,
    )
    from phycoflow_reconstruction.training.rollout_contract import verify_rollout_contract

    class Flow(torch.nn.Module):
        capabilities = ModelCapabilities("point", True, True, False, True)
        offset = 0.0

        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(0.3))

        def sample_source(self, batch, generator=None):
            return torch.zeros(1, 4, 2)

        def velocity(self, batch, state, time):
            return torch.ones_like(state) * self.value

        def reconstruct(self, batch, **kwargs):
            return ReconstructionBatch(torch.ones(1, 4, 2) * self.value + self.offset)

    batch = ObservationBatch(
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 1),
        torch.zeros(1, 1, dtype=torch.long),
        torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, 4, 2),
        torch.ones(1, 4, dtype=torch.bool),
        None,
        ("sample",),
    )
    cfg = {
        "rollout": {"steps": 2, "solver": "euler"},
        "evaluation": {"generation_steps": 2},
        "observation_consistency": {"mode": "none", "final_clamp": False},
    }
    model = Flow()
    state = torch.random.get_rng_state()
    assert verify_rollout_contract(model, batch, cfg)["passed"]
    assert torch.equal(state, torch.random.get_rng_state())
    model.offset = 0.01
    with pytest.raises(ValueError, match="rollout mismatch"):
        verify_rollout_contract(model, batch, cfg)


def test_rollout_agreement_accepts_roundoff_at_signed_field_zero_crossings():
    from phycoflow_reconstruction.training.rollout_contract import _endpoint_agreement

    reference = torch.tensor([[[-1.0, -0.01], [0.0, 0.0], [1.0, 0.01]]])
    actual = reference.clone()
    actual[0, 1] = torch.tensor([4.3e-6, 1.3e-7])
    report = _endpoint_agreement(actual, reference)
    assert report["passed"]
    assert not report["pointwise_allclose_legacy"]
    # Tolerances must scale with each field's units, including small velocities.
    units = torch.tensor([1e3, 1e-3])
    assert _endpoint_agreement(actual * units, reference * units)["passed"]


@pytest.mark.parametrize(
    "kind", ["small_field", "localized", "distributed", "zero_field", "nan", "inf"]
)
def test_rollout_agreement_rejects_material_and_nonfinite_errors(kind):
    from phycoflow_reconstruction.training.rollout_contract import _endpoint_agreement

    reference = torch.ones(1, 100, 2) * torch.tensor([1.0, 1e-3])
    actual = reference.clone()
    if kind == "small_field":
        actual[..., 1] += 1e-6  # Hidden by the old shared 2e-6 absolute floor.
    elif kind == "localized":
        actual[0, 0, 0] += 2e-4
    elif kind == "distributed":
        actual[..., 0] += 2e-5  # Below peak tolerance but above RMS tolerance.
    elif kind == "zero_field":
        reference[..., 1] = 0
        actual[..., 1] = 1e-12
    else:
        reference[0, 0, 0] = actual[0, 0, 0] = float(kind)
    assert not _endpoint_agreement(actual, reference)["passed"]
    assert _endpoint_agreement(torch.zeros_like(reference), torch.zeros_like(reference))["passed"]
