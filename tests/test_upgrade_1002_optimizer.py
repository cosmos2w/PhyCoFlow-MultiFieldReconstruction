"""Independent numerical/controller and lifecycle gates for the opt-in upgrade."""

import json
from copy import deepcopy

import pytest
import torch

from phycoflow_reconstruction.training.fidelity_controller import (
    FIDELITY_DEFAULTS,
    FidelityController,
    coherence_selection_report,
    endpoint_risks,
)
from phycoflow_reconstruction.training.gradient_balance import (
    _assign_flat_gradient,
    _flat_gradient,
    coherence_primal_dual_update,
    combine_coherence_gradients,
)
from phycoflow_reconstruction.training.gradients import stable_clip_grad_norm_


def controller(*, state=None, scale=1):
    calibration = {"field_names": ["u", "v"], "source_risks": [1.5 * scale, scale, 2 * scale]}
    return FidelityController(("u", "v"), FIDELITY_DEFAULTS, calibration, state=state)


@pytest.mark.parametrize("method", ["config", "cagrad", "weighted_sum"])
@pytest.mark.parametrize("gradients", [
    [[1., 0.], [2., 0.], [3., 0.]],
    [[1., 0.], [0., 1.], [1., 1.]],
    [[1., 0.], [-1., 0.], [0., 0.]],
    [[0., 0.], [0., 0.], [0., 0.]],
])
def test_generic_coherence_combiner_degenerate_and_finite(method, gradients):
    bank = {name: torch.tensor(g, dtype=torch.float64) for name, g in zip("ABC", gradients)}
    direction, report = combine_coherence_gradients(bank, method=method)
    assert torch.isfinite(direction).all()
    assert report["families"] == ["A", "B", "C"]
    if gradients == [[1., 0.], [-1., 0.], [0., 0.]]:
        assert not report["strict_common_descent"]
        if method == "config":
            assert report["fallback_reason"] == "config_no_strict_common_descent_weighted_sum"
    if report["strict_common_descent"]:
        assert all(report["combined_dot"][n] > 0 for n in bank if torch.count_nonzero(bank[n]))


def test_cagrad_zero_alpha_equals_mean_and_orthogonal_solution():
    bank = {"A": torch.tensor([1., 0.], dtype=torch.float64),
            "B": torch.tensor([0., 1.], dtype=torch.float64)}
    zero, _ = combine_coherence_gradients(bank, method="cagrad", cagrad_alpha=0)
    assert torch.allclose(zero, torch.tensor([.5, .5], dtype=torch.float64), atol=2e-8)
    # Symmetry fixes the simplex solution to (1/2,1/2). Authors' rescale=1.
    direction, report = combine_coherence_gradients(bank, method="cagrad", cagrad_alpha=.5)
    expected = (.5 + .5 * (.5 * (0.5 + 1e-8)**.5 + 1e-8) / (0.5**.5 + 1e-8)) / 1.25
    assert torch.allclose(direction, torch.full((2,), expected, dtype=torch.float64), atol=1e-8)
    assert report["cagrad_weights"] == pytest.approx([.5, .5])


def test_nonfinite_gradient_is_rejected_before_update():
    with pytest.raises(FloatingPointError):
        combine_coherence_gradients({"A": torch.tensor([float("nan")])})


def test_fidelity_normalized_risk_unit_invariance_and_near_zero_source():
    prediction = torch.tensor([[[2., 1.], [1., 3.]]], requires_grad=True)
    source = torch.ones_like(prediction)
    target = torch.zeros_like(prediction)
    first = controller().violations(prediction, source, target)[0]
    second = controller(scale=9).violations(prediction * 3, source * 3, target)[0]
    assert torch.allclose(first, second)
    c = FidelityController(("u", "v"), FIDELITY_DEFAULTS,
                           {"field_names": ["u", "v"], "source_risks": [1e-8] * 3})
    violation, _, _ = c.violations(prediction * 1e-4, target, target)
    loss, _, _ = c.primal(violation)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(prediction.grad).all()


def test_dual_slack_saturation_order_and_exact_resume():
    c = controller()
    for _ in range(20):
        c.advance(torch.full((3,), .2))
    assert (c.multipliers > 0).all()
    restored = controller(state=c.state_dict())
    for values in ([-.5] * 3, [.1] * 3, [-.8] * 3):
        c.advance(torch.tensor(values))
        restored.advance(torch.tensor(values))
    assert torch.equal(c.multipliers, restored.multipliers)
    assert torch.equal(c.ema, restored.ema)
    assert c.updates == restored.updates
    for _ in range(100):
        c.advance(torch.full((3,), -.5))
    assert torch.equal(c.multipliers, torch.zeros(3, dtype=torch.float64))
    invalid = restored.state_dict()
    invalid["settings"] = {**invalid["settings"], "dual_lr": 3.}
    with pytest.raises(ValueError, match="calibration/settings"):
        controller(state=invalid)


def test_augmented_fidelity_gradient_and_detached_teacher():
    prediction = torch.tensor([[[2., 3.], [1., 2.]]], dtype=torch.float64, requires_grad=True)
    teacher = torch.ones_like(prediction, requires_grad=True)
    c = controller()
    violations, _, _ = c.violations(prediction, teacher, torch.zeros_like(prediction))
    loss, _, pressure = c.primal(violations)
    derivative = torch.autograd.grad(loss, violations, retain_graph=True)[0]
    assert torch.equal(derivative, pressure)
    loss.backward()
    assert teacher.grad is None
    assert torch.isfinite(prediction.grad).all()


def test_real_complex_unused_layout_and_one_adamw_displacement():
    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.real = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.float64))
            self.complex = torch.nn.Parameter(torch.tensor([1. + 2j], dtype=torch.complex128))
            self.unused = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))

    model = Fixture()
    parameters = list(model.parameters())
    a = model.real.square().sum() + model.complex.abs().square().sum()
    flat = _flat_gradient(a, parameters)
    assert flat.numel() == 6
    _assign_flat_gradient(parameters, flat)
    assert torch.equal(model.complex.grad, torch.tensor([2. + 4j], dtype=torch.complex128))
    assert not torch.count_nonzero(model.unused.grad)
    optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=.01)
    b = model.real.sum() + model.complex.real.sum()
    fidelity = a * 0
    result = coherence_primal_dual_update(model, optimizer, {"A": a, "B": b}, fidelity,
                                         method="weighted_sum", grad_clip=1, diagnostics=True)
    assert all(int(value["step"]) == 1 for value in optimizer.state.values())
    assert result["update/actual_displacement_norm"] > 0
    assert result["update/actual_dot/A"] < 0
    assert result["gradient/fidelity/fidelity/cosine"] is None
    assert result["gradient/fidelity/fidelity/cosine/undefined_reason"] == "zero_gradient"


def test_selector_refuses_hidden_field_regression_and_ranks_balanced_ratios():
    source = {"mse_normalized": 1., "per_field_mse_normalized": {"u": 1., "v": 1.},
              "coherence": {"families": {n: {"total": 2.} for n in "ABC"}}}
    candidate = deepcopy(source)
    candidate["coherence"]["families"]["A"]["total"] = 1.
    settings = {"max_relative_mse_increase": .05, "max_relative_field_mse_increase": .05}
    eligible = coherence_selection_report(source, candidate, settings)
    assert eligible["eligible"] and eligible["metric"] == pytest.approx(5 / 6)
    candidate["per_field_mse_normalized"]["v"] = 1.06
    rejected = coherence_selection_report(source, candidate, settings)
    assert not rejected["eligible"] and rejected["failed_fidelity_fields"] == ["v"]


def test_selector_replays_sorted_json_without_weakening_membership():
    source = {
        "mse_normalized": 1.,
        "per_field_mse_normalized": {"u": 1., "v": 1.},
        "coherence": {"families": {
            "global_distribution": {"total": 2.},
            "cross_spectrum": {"total": 4.},
            "topology": {"total": 8.},
        }},
    }
    candidate = deepcopy(source)
    candidate["coherence"]["families"]["topology"]["total"] = 4.
    settings = {"max_relative_mse_increase": .05, "max_relative_field_mse_increase": .05}
    replayed = json.loads(json.dumps(source, sort_keys=True))
    original = coherence_selection_report(source, candidate, settings)
    resumed = coherence_selection_report(replayed, candidate, settings)
    assert resumed["eligible"] and resumed["metric"] == original["metric"]
    assert resumed["family_source_normalized_scores"] == original["family_source_normalized_scores"]
    del candidate["coherence"]["families"]["cross_spectrum"]
    with pytest.raises(ValueError, match="membership changed"):
        coherence_selection_report(replayed, candidate, settings)
    candidate["coherence"]["families"]["unexpected"] = {"total": 4.}
    with pytest.raises(ValueError, match="membership changed"):
        coherence_selection_report(replayed, candidate, settings)


def test_endpoint_risk_masks_and_aggregate_not_sample_ratios():
    prediction = torch.tensor([[[1.], [100.]], [[3.], [100.]]])
    risks = endpoint_risks(prediction, torch.zeros_like(prediction), torch.tensor([[1, 0], [1, 0]], dtype=torch.bool))
    assert risks.item() == 5.
    with pytest.raises(ValueError, match="valid query"):
        endpoint_risks(prediction, prediction, torch.zeros(2, 2, dtype=torch.bool))


def test_complex_clipping_preserves_phase_and_joint_real_complex_norm():
    real = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    complex_value = torch.nn.Parameter(torch.zeros(1, dtype=torch.complex128))
    real.grad = torch.tensor([12.], dtype=torch.float64)
    complex_value.grad = torch.tensor([3. + 4j], dtype=torch.complex128)
    norm = stable_clip_grad_norm_([real, complex_value], 1.)
    assert norm.item() == pytest.approx(13.)
    factor = 1 / (13 + 1e-6)
    assert real.grad.item() == pytest.approx(12 * factor)
    assert complex_value.grad.item() == pytest.approx((3 + 4j) * factor)


def test_constant_and_disconnected_coherence_objectives_use_zero_gradients():
    model = torch.nn.Linear(1, 1, bias=False)
    unrelated = torch.tensor(2., requires_grad=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0.)
    before = model.weight.detach().clone()
    row = coherence_primal_dual_update(
        model, optimizer, {"constant": torch.tensor(0.), "disconnected": unrelated.square()},
        torch.tensor(0.), method="config", grad_clip=1., diagnostics=True,
    )
    assert row["coherence_combiner"]["active_count"] == 0
    assert not row["coherence_combiner"]["strict_common_descent"]
    assert row["update/actual_displacement_norm"] == 0.
    assert torch.equal(before, model.weight)


def test_controller_recovery_rejects_unknown_definition():
    state = controller().state_dict()
    state["version"] = "unknown"
    with pytest.raises(ValueError, match="version mismatch"):
        controller(state=state)
