"""Tests of actual Adam transactions, component calibration, and cached PH."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from phycoflow_reconstruction.training.topology_constraints import (
    CONSTRAINT_DEFAULTS,
    ComponentConstrainedAdam,
    calibrated_score,
    make_calibration,
    project_halfspaces,
)


class Point(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = nn.Parameter(torch.tensor([1.0, 1.0]))
        self.register_buffer("constant", torch.tensor(2.0))


def test_transaction_projects_actual_adam_and_commits_moments():
    model = Point()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)
    controller = ComponentConstrainedAdam(model, optimizer, CONSTRAINT_DEFAULTS)
    # Initial score wants x0 down and x1 up. Constraint protects x1, so Adam's
    # proposal must be corrected after its adaptive scaling and weight decay.
    score = model.x[0] - model.x[1]

    def evaluate(candidate):
        return {"x1": candidate.x[1]}, candidate.x[0] - candidate.x[1]

    report = controller.step(score, {"x1": model.x[1]}, {"x1": 1.0}, {"x1": 1e-6}, evaluate)
    assert report["update_accepted"]
    assert report["constraint_active_gradients"] == 1
    assert model.x[0] < 1 and model.x[1] <= 1 + 1e-6
    assert optimizer.state[model.x]["step"] == 1
    assert controller.state_dict() == {"attempted": 1, "accepted": 1, "rejected": 0}


@pytest.mark.parametrize("incompatible", [False, True])
def test_working_set_corrects_correlated_constraints_but_checks_every_sample(incompatible):
    model = Point()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.0)
    settings = {
        **CONSTRAINT_DEFAULTS,
        "max_active_gradients": 1,
        "correction_selection": "largest_violation",
    }
    controller = ComponentConstrainedAdam(model, optimizer, settings)
    constraints = {f"sample{i}": model.x[1] for i in range(8)}
    bounds = {name: 1.0 for name in constraints}
    tolerances = {name: 1e-6 for name in constraints}

    def evaluate(candidate):
        values = {name: candidate.x[1] for name in constraints}
        if incompatible:
            values["sample7"] = candidate.x[1] * 0 + 2.0
        return values, candidate.x[0] - candidate.x[1]

    report = controller.step(model.x[0] - model.x[1], constraints, bounds, tolerances, evaluate)
    assert report["update_accepted"] is not incompatible
    assert report["constraint_active_gradients"] <= 1
    assert len(report["constraint_candidate"]) == 8
    if incompatible:
        torch.testing.assert_close(model.x, torch.ones(2), rtol=0, atol=0)
        assert not optimizer.state
    else:
        assert all(
            value <= bounds[name] + tolerances[name]
            for name, value in report["constraint_candidate"].items()
        )


def test_rejected_adam_is_bitwise_unchanged_including_existing_moments_and_rng():
    model = Point()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    model.x.square().sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    controller = ComponentConstrainedAdam(
        model, optimizer, {**CONSTRAINT_DEFAULTS, "max_corrections": 0}
    )
    weights, state = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())
    rng = torch.get_rng_state()

    def evaluate(candidate):
        torch.rand(5)
        candidate.constant.add_(100)  # Must not touch the live model's buffer.
        return {"bad": candidate.x.sum() * 0 + 2}, candidate.x.square().sum()

    report = controller.step(
        model.x.square().sum(), {"bad": model.x.sum() * 0}, {"bad": 0}, {"bad": 0}, evaluate
    )
    assert not report["update_accepted"]
    for key, value in weights.items():
        assert torch.equal(value, model.state_dict()[key])
    for key, value in state["state"][0].items():
        assert torch.equal(value, optimizer.state_dict()["state"][0][key])
    assert torch.equal(rng, torch.get_rng_state())


def test_conflicting_halfspaces_and_zero_gradient_do_not_fallback():
    proposal = torch.tensor([1.0, 1.0])
    assert project_halfspaces(proposal, [torch.zeros(2)], [-1.0]) is None
    assert (
        project_halfspaces(
            proposal, [torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])], [-1.0, -1.0]
        )
        is None
    )
    result = project_halfspaces(proposal, [torch.tensor([1.0, 0.0])], [0.0])
    torch.testing.assert_close(result, torch.tensor([0.0, 1.0]))


def test_calibration_uses_fixed_category_balanced_source_scales_and_floor():
    records = [
        {
            "raw_total": 10.0,
            "components": {
                "topology.self.a.h0": 0.0,
                "topology.self.a.h1": 2.0,
                "topology.mutual.a+b.h0": 8.0,
            },
            "sample_ids": ["train"],
        }
    ]
    calibration = make_calibration(
        records,
        floor=1e-4,
        weights={"self.persistence": 1.0, "mutual.persistence": 1.0},
        source_hash="source",
    )
    assert calibration["scales"]["topology.self.a.h0"] == 1e-4
    score = calibrated_score(
        {name: value for name, value in calibration["scales"].items()}, calibration
    )
    assert score == pytest.approx(10.0)
    with pytest.raises(ValueError, match="components differ"):
        calibrated_score({}, calibration)


def test_cached_persistence_matches_value_and_gradient():
    pytest.importorskip("gudhi")
    from phycoflow_reconstruction.coherence.families.topology.persistence_objective import (
        PersistenceTopologyObjective,
    )

    config = {
        "target_use": "paired_supervised",
        "fields": ["u", "v"],
        "geometry": {"periodic": True},
        "components": {
            "self": {"enabled": True},
            "mutual": {"enabled": True, "groups": [["u", "v"]], "lines": 2},
        },
    }
    objective = PersistenceTopologyObjective(config, ("u", "v"))
    torch.manual_seed(72)
    prediction = torch.randn(1, 2, 6, 6, requires_grad=True)
    reference = torch.randn_like(prediction)
    uncached = objective(prediction, reference)
    cache = {}
    objective(prediction.detach(), reference, reference_cache=cache)
    cached = objective(prediction, reference, reference_cache=cache)
    torch.testing.assert_close(cached.scalar_loss, uncached.scalar_loss, rtol=0, atol=0)
    a = torch.autograd.grad(uncached.scalar_loss, prediction)[0]
    b = torch.autograd.grad(cached.scalar_loss, prediction)[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_candidate_acceptance_rejects_nonfinite_and_non_descent():
    for invalid in (float("nan"), 100.0):
        model = Point()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        controller = ComponentConstrainedAdam(model, optimizer, CONSTRAINT_DEFAULTS)
        original = model.x.detach().clone()
        report = controller.step(
            model.x.sum(),
            {"x": model.x.sum()},
            {"x": 100.0},
            {"x": 0.0},
            lambda candidate, value=invalid: ({"x": candidate.x.sum()}, value),
        )
        assert not report["update_accepted"]
        assert torch.equal(model.x, original)
        assert not optimizer.state


def test_reference_cache_rejects_changed_reference():
    pytest.importorskip("gudhi")
    from phycoflow_reconstruction.coherence.families.topology.persistence_objective import (
        PersistenceTopologyObjective,
    )

    objective = PersistenceTopologyObjective(
        {"target_use": "paired_supervised", "fields": ["u"]}, ("u",)
    )
    target = torch.randn(1, 1, 4, 4)
    prediction = torch.randn_like(target)
    cache = {}
    objective(prediction, target, reference_cache=cache)
    with pytest.raises(ValueError, match="target changed"):
        objective(prediction, target.clone(), reference_cache=cache)
    target.add_(1)
    with pytest.raises(ValueError, match="target changed"):
        objective(prediction, target, reference_cache=cache)


def test_constraints_preserve_sample_identity_instead_of_averaging():
    from phycoflow_reconstruction.contracts import FamilyResult, TermResult
    from phycoflow_reconstruction.training.topology_constraints import (
        fidelity_components,
        sample_leaf_components,
    )

    values = torch.tensor([1.0, 2.0], requires_grad=True)
    result = FamilyResult(
        {"topology.self.u.h0": TermResult(values, values.mean())}, values, values.mean()
    )
    leaves = sample_leaf_components(result)
    assert set(leaves) == {"topology.self.u.h0#0", "topology.self.u.h0#1"}
    assert float(leaves["topology.self.u.h0#1"].detach()) == 2.0
    reference = torch.tensor([[[0.0], [2.0]], [[0.0], [2.0]]])
    source = reference + 1
    prediction = source.clone()
    prediction[0] += 1
    prediction[1] -= 1
    losses, bounds = fidelity_components(
        prediction,
        reference,
        source,
        ["u"],
        {"max_relative_field_mse_increase": 0.05, "max_source_anchor_nmse": 0.01},
    )
    assert losses["fidelity.endpoint.u#0"] > bounds["fidelity.endpoint.u#0"]
    assert losses["fidelity.endpoint.u#1"] < bounds["fidelity.endpoint.u#1"]


@pytest.mark.parametrize(
    "reduction,bad_fidelity,bad_component,accepted",
    [
        ("per_sample", False, False, False),
        ("batch_mean", False, False, True),
        ("batch_mean", True, False, False),
        ("batch_mean", False, True, False),
    ],
)
def test_batch_mean_update_allows_one_worse_sample_but_protects_each_component_and_fidelity(
    reduction, bad_fidelity, bad_component, accepted
):
    from phycoflow_reconstruction.contracts import FamilyResult, TermResult
    from phycoflow_reconstruction.training.post_training import _constrained_topology_update

    model = Point()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    settings = {**CONSTRAINT_DEFAULTS, "topology_reduction": reduction, "max_corrections": 0}
    controller = ComponentConstrainedAdam(model, optimizer, settings)

    class Family:
        names = ["topology.self.u.h0"] + (["topology.self.u.h1"] if bad_component else [])

        def __init__(self):
            self.component_calibration = {
                "scales": dict.fromkeys(self.names, 1.0),
                "coefficients": dict.fromkeys(self.names, 1.0),
            }

        def __call__(self, prediction, reference, **kwargs):
            z = prediction[:, 0, 0]
            # Reducing z improves sample 0 four times as much as sample 1 worsens.
            h0 = torch.stack((4 * z[0], 2 - z[1]))
            terms = {"topology.self.u.h0": TermResult(h0, h0.mean())}
            total = h0
            if bad_component:
                h1 = 2 - z  # Another component worsens despite a lower total score.
                terms["topology.self.u.h1"] = TermResult(h1, h1.mean())
                total = total + h1
            return FamilyResult(terms, total, total.mean())

    def reconstruct(candidate):
        return candidate.x[0] + torch.tensor([[[0.0], [1.0]], [[0.0], [1.0]]])

    prediction = reconstruct(model)
    reference = torch.tensor([[[0.0], [1.0]], [[0.0], [1.0]]])
    if bad_fidelity:
        reference[1] += 1.0  # The second sample starts perfect: its fidelity cannot be sacrificed.
    context = {
        "prediction": prediction,
        "reference": reference,
        "coordinates": None,
        "source_prediction": prediction.detach(),
        "persistence_cache": {},
        "reconstruct": reconstruct,
    }
    family = Family()
    report = _constrained_topology_update(
        controller,
        family(prediction, reference),
        context,
        family,
        {
            "optimization": {"component_constraints": settings},
            "posttrain_fidelity": {"max_relative_field_mse_increase": 0.05},
        },
        ["u"],
    )
    assert report["update_accepted"] is accepted
    assert report["topology_reduction"] == reduction
    assert report["fidelity_constraint_count"] == 4
    assert report["constraint_batch_size"] == 2
    assert report["topology_constraint_count"] == len(family.names) * (
        2 if reduction == "per_sample" else 1
    )
    if accepted:
        after = family(reconstruct(model), reference).component_results["topology.self.u.h0"]
        assert after.per_sample_cost[1] > 1.0  # Individual topology may worsen.
        assert after.scalar_loss < 2.5  # Its component's batch mean improves.
        assert optimizer.state[model.x]["step"] == 1
    else:
        torch.testing.assert_close(model.x, torch.ones(2), rtol=0, atol=0)
        assert not optimizer.state


def test_batch_mean_gradient_uses_actual_short_batch_size():
    from phycoflow_reconstruction.contracts import FamilyResult, TermResult
    from phycoflow_reconstruction.training.topology_constraints import (
        topology_constraint_components,
    )

    values = torch.tensor([1.0, 2.0, 6.0], requires_grad=True)
    result = FamilyResult(
        {"topology.self.u.h0": TermResult(values, values.mean())}, values, values.mean()
    )
    averaged = topology_constraint_components(result, "batch_mean")["topology.self.u.h0"]
    assert averaged.item() == 3.0
    torch.testing.assert_close(
        torch.autograd.grad(averaged, values)[0], torch.full_like(values, 1 / 3)
    )
    with pytest.raises(ValueError, match="topology_reduction"):
        topology_constraint_components(result, "sum")


def test_audit_resume_discards_uncheckpointed_attempts_and_partial_write(tmp_path):
    from phycoflow_reconstruction.training.topology_constraints import restore_constraint_audit

    path = tmp_path / "constraint_updates.jsonl"
    path.write_text('{"step": 1}\n{"step": 2}\n{"step": 3}\n{"ste')
    restore_constraint_audit(path, 2)
    assert path.read_text() == '{"step": 1}\n{"step": 2}\n'
    path.write_text('{"step": 1}\n{"ste')
    restore_constraint_audit(path, 2)
    assert path.read_text() == '{"step": 1}\n'


def test_reconstruction_proposal_cannot_override_topology_descent():
    model = Point()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.0)
    controller = ComponentConstrainedAdam(model, optimizer, CONSTRAINT_DEFAULTS)
    score = model.x[0]
    report = controller.step(
        score,
        {"other": model.x[1]},
        {"other": 2.0},
        {"other": 0.0},
        lambda candidate: ({"other": candidate.x[1]}, candidate.x[0]),
        proposal_loss=-score,
    )
    assert not report["update_accepted"]
    assert report["constraint_reason"] == "no_topology_descent"
    torch.testing.assert_close(model.x, torch.ones(2), rtol=0, atol=0)


def test_correction_budget_and_unchanged_projections_stop_without_committing():
    for budget, reason in ((0, "correction_gradient_budget"), (12, "unchanged_linearization")):
        model = Point()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.0)
        settings = {**CONSTRAINT_DEFAULTS, "max_active_gradients": budget}
        controller = ComponentConstrainedAdam(model, optimizer, settings)
        report = controller.step(
            model.x[0],
            {"x1": model.x[1]},
            {"x1": 1.0},
            {"x1": 0.0},
            lambda candidate: ({"x1": 2.0}, candidate.x[0]),
        )
        assert not report["update_accepted"]
        assert report["constraint_reason"] == reason
        assert report["constraint_checks"] <= 2
        torch.testing.assert_close(model.x, torch.ones(2), rtol=0, atol=0)


def test_nonfinite_source_bound_cannot_authorize_an_update():
    model = Point()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    controller = ComponentConstrainedAdam(model, optimizer, CONSTRAINT_DEFAULTS)
    with pytest.raises(FloatingPointError, match="baseline"):
        controller.step(
            model.x.sum(),
            {"source": model.x.sum()},
            {"source": float("nan")},
            {"source": 0.0},
            lambda candidate: ({"source": 0.0}, 0.0),
        )
    assert not optimizer.state


def test_candidate_rollout_matches_baseline_grad_mode_even_inside_no_grad():
    from phycoflow_reconstruction.contracts import FamilyResult, TermResult
    from phycoflow_reconstruction.training.post_training import _constrained_topology_update

    model = Point()

    class Family:
        def __init__(self):
            self.component_calibration = {
                "scales": {"topology.self.u.h0": 1.0},
                "coefficients": {"topology.self.u.h0": 1.0},
            }

        def __call__(self, prediction, reference, **kwargs):
            value = prediction[:, 0, 0].square()
            return FamilyResult(
                {"topology.self.u.h0": TermResult(value, value.mean())}, value, value.mean()
            )

    def reconstruct(candidate):
        # Stand-in for a fused inference kernel with different roundoff.
        bias = 0.0 if torch.is_grad_enabled() else 1e-3
        return candidate.x.reshape(1, 2, 1) + bias

    class CheckIdenticalParameters:
        def step(self, score, constraints, bounds, tolerances, evaluate, **kwargs):
            with torch.no_grad():
                _values, candidate_score = evaluate(model)
            torch.testing.assert_close(candidate_score, score, atol=0, rtol=0)
            return {}

    prediction = reconstruct(model)
    reference = torch.tensor([[[-1.0], [1.0]]])
    family = Family()
    context = {
        "prediction": prediction,
        "reference": reference,
        "coordinates": None,
        "source_prediction": prediction.detach(),
        "persistence_cache": {},
        "reconstruct": reconstruct,
    }
    _constrained_topology_update(
        CheckIdenticalParameters(),
        family(prediction, reference),
        context,
        family,
        {"optimization": {}, "posttrain_fidelity": {"max_relative_field_mse_increase": 0.05}},
        ["u"],
    )
