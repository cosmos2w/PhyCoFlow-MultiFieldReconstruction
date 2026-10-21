from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import pytest
import torch
from torch import nn

from phycoflow_reconstruction.training.multitask import (
    FamilyScalarizer,
    OuterAggregator,
    modular_objective_update,
)


class _MixedParameters(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.real = nn.Parameter(torch.tensor([1.25]))
        self.complex = nn.Parameter(torch.tensor([0.4 + 0.7j], dtype=torch.complex64))
        self.unused = nn.Parameter(torch.tensor([2.5]))
        self.unused_complex = nn.Parameter(torch.tensor([0.6 + 0.2j], dtype=torch.complex64))


def _joint_losses(model: _MixedParameters) -> tuple[torch.Tensor, OrderedDict[str, torch.Tensor]]:
    real = model.real
    complex_value = model.complex
    data_loss = (real - 0.1).square().sum() + 0.15 * complex_value.abs().square().sum()
    family_losses = OrderedDict(
        A=(real + 0.3).square().sum() + 0.20 * (complex_value.real - 0.5).square().sum(),
        B=(real - 1.7).square().sum() + 0.10 * (complex_value.imag + 0.2).square().sum(),
        T=(0.4 * real + complex_value.real + complex_value.imag).square().sum(),
    )
    return data_loss, family_losses


def _make_adamw(model: nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=0.025, weight_decay=0.12)


def _unit_scaling(names: tuple[str, ...]) -> tuple[dict[str, float], dict[str, float]]:
    return ({name: 1.0 for name in names}, {name: 1.0 for name in names})


def test_four_objective_sum_matches_zero_padded_backward_and_adamw() -> None:
    modular_model = _MixedParameters()
    reference_model = _MixedParameters()
    modular_optimizer = _make_adamw(modular_model)
    reference_optimizer = _make_adamw(reference_model)
    data_weight = 0.7
    coherence_weight = 0.35

    data_loss, family_losses = _joint_losses(modular_model)
    reference_data, reference_families = _joint_losses(reference_model)
    (data_weight * reference_data + coherence_weight * sum(reference_families.values())).backward()
    # The modular updater assigns zero gradients to unused parameters; match that established
    # behavior before AdamW so its decoupled weight decay is included in the reference update.
    for parameter in reference_model.parameters():
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
    reference_optimizer.step()

    diagnostics = modular_objective_update(
        modular_model,
        modular_optimizer,
        data_loss,
        family_losses,
        aggregator=OuterAggregator({"method": "sum"}),
        data_weight=data_weight,
        coherence_weight=coherence_weight,
        grad_clip=None,
    )

    for actual, expected in zip(modular_model.parameters(), reference_model.parameters()):
        torch.testing.assert_close(actual, expected)
    assert torch.equal(modular_model.unused.grad, torch.zeros_like(modular_model.unused))
    assert torch.equal(
        modular_model.unused_complex.grad, torch.zeros_like(modular_model.unused_complex)
    )
    assert diagnostics["aggregation_scope"] == "joint"
    assert diagnostics["outer_aggregation"]["objective_names"] == [
        "data_retention",
        "A",
        "B",
        "T",
    ]
    assert len(diagnostics["outer_aggregation"]["directional_products"]) == 4


def test_four_objective_config_matches_torchjd_and_runs_when_aligned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torchjd")
    import torchjd.aggregation as torchjd_aggregation

    upstream_config = torchjd_aggregation.ConFIG
    conflicting = torch.tensor(
        [[1.0, 0.2, 0.0], [-0.6, 0.8, 0.3], [0.1, -0.4, 1.2], [0.5, 0.7, -0.2]],
        dtype=torch.float64,
    )
    direction, diagnostics = OuterAggregator({"method": "torchjd_config"})(conflicting)
    torch.testing.assert_close(direction, upstream_config()(conflicting))

    pairwise = torch.tensor(diagnostics["pairwise_gradient_cosines"], dtype=torch.float64)
    norms = torch.linalg.vector_norm(conflicting, dim=1)
    expected_pairwise = (conflicting @ conflicting.T) / (norms[:, None] * norms[None, :])
    torch.testing.assert_close(pairwise, expected_pairwise)
    remainder = conflicting[1:].sum(dim=0)
    expected_rf_cosine = torch.dot(conflicting[0], remainder) / (
        norms[0] * torch.linalg.vector_norm(remainder)
    )
    assert diagnostics["gradient_cosine"] == pytest.approx(float(expected_rf_cosine))
    assert len(diagnostics["directional_products"]) == 4

    calls: list[torch.Tensor] = []

    class RecordingConFIG:
        def __call__(self, gradients: torch.Tensor) -> torch.Tensor:
            calls.append(gradients.detach().clone())
            return gradients.new_tensor([3.0, -2.0])

    monkeypatch.setattr(torchjd_aggregation, "ConFIG", RecordingConFIG)
    aligned = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
    aligned_direction, aligned_diagnostics = OuterAggregator(
        {"method": "torchjd_config"}
    )(aligned)
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], aligned)
    torch.testing.assert_close(aligned_direction, torch.tensor([3.0, -2.0]))
    assert aligned_diagnostics["gradient_cosine"] == pytest.approx(1.0)


@pytest.mark.parametrize("method", ["sum", "torchjd_config", "upgrad", "cagrad"])
def test_outer_methods_accept_one_objective(method: str) -> None:
    if method != "sum":
        pytest.importorskip("torchjd")
    direction, diagnostics = OuterAggregator({"method": method})(
        torch.tensor([[1.0, -2.0]])
    )
    assert direction.shape == (2,)
    assert torch.isfinite(direction).all()
    assert len(diagnostics["directional_products"]) == 1


@pytest.mark.parametrize(
    "matrix",
    [
        torch.empty((0, 2)),
        torch.empty((2, 0)),
        torch.empty((2,)),
        torch.empty((1, 2, 1)),
    ],
    ids=("no-objectives", "no-parameters", "rank-one", "rank-three"),
)
def test_outer_aggregator_rejects_invalid_matrix_dimensions(matrix: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        OuterAggregator({"method": "sum"})(matrix)


def test_none_scalarizer_calibrates_ordered_families_and_global_weights_once() -> None:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    reference = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        reference.weight.copy_(model.weight)

    raw = OrderedDict(
        A=(model.weight - 0.0).square().sum(),
        B=(2.0 * model.weight + 1.0).square().sum(),
    )
    scales = {"A": 3.0, "B": 0.25}
    family_weights = {"A": 0.5, "B": 2.0}
    calibrated, scalarizer_diagnostics = FamilyScalarizer({"method": "none"})(
        raw, scales=scales, weights=family_weights
    )

    assert isinstance(calibrated, Mapping)
    assert list(calibrated) == ["A", "B"]
    assert scalarizer_diagnostics["method"] == "none"
    assert scalarizer_diagnostics["effective_weights"] == {"A": 1.0, "B": 1.0}
    assert "scalarized_loss" not in scalarizer_diagnostics
    assert scalarizer_diagnostics["raw_losses"] == pytest.approx({"A": 1.0, "B": 9.0})
    assert scalarizer_diagnostics["calibrated_losses"] == pytest.approx(
        {"A": 1.5, "B": 4.5}
    )
    torch.testing.assert_close(calibrated["A"], raw["A"] * 1.5)
    torch.testing.assert_close(calibrated["B"], raw["B"] * 0.5)

    reference_raw = OrderedDict(
        A=(reference.weight - 0.0).square().sum(),
        B=(2.0 * reference.weight + 1.0).square().sum(),
    )
    reference_data = (reference.weight + 3.0).square().sum()
    (0.2 * reference_data + 0.4 * (1.5 * reference_raw["A"] + 0.5 * reference_raw["B"])).backward()
    expected_weight = reference.weight.detach() - 0.1 * reference.weight.grad

    data_loss = (model.weight + 3.0).square().sum()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    diagnostics = modular_objective_update(
        model,
        optimizer,
        data_loss,
        calibrated,
        aggregator=OuterAggregator({"method": "sum"}),
        data_weight=0.2,
        coherence_weight=0.4,
        grad_clip=None,
    )

    torch.testing.assert_close(model.weight, expected_weight)
    assert diagnostics["aggregation_scope"] == "joint"
    assert diagnostics["outer_aggregation"]["objective_names"] == ["data_retention", "A", "B"]


def test_unit_fixed_sum_and_outer_sum_match_joint_sum() -> None:
    joint_model = nn.Linear(1, 1, bias=False)
    hierarchical_model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        joint_model.weight.fill_(0.8)
        hierarchical_model.weight.copy_(joint_model.weight)

    joint_data = (joint_model.weight - 0.3).square().sum()
    joint_families = OrderedDict(
        A=(joint_model.weight + 0.6).square().sum(),
        B=(2.0 * joint_model.weight - 1.0).square().sum(),
    )
    hierarchical_data = (hierarchical_model.weight - 0.3).square().sum()
    hierarchical_families = OrderedDict(
        A=(hierarchical_model.weight + 0.6).square().sum(),
        B=(2.0 * hierarchical_model.weight - 1.0).square().sum(),
    )
    family_scales, family_weights = _unit_scaling(("A", "B"))
    scalarized, _ = FamilyScalarizer({"method": "fixed_sum"})(
        hierarchical_families, scales=family_scales, weights=family_weights
    )
    joint_optimizer = torch.optim.SGD(joint_model.parameters(), lr=0.05)
    hierarchical_optimizer = torch.optim.SGD(hierarchical_model.parameters(), lr=0.05)

    joint_diagnostics = modular_objective_update(
        joint_model,
        joint_optimizer,
        joint_data,
        joint_families,
        aggregator=OuterAggregator({"method": "sum"}),
        data_weight=1.0,
        coherence_weight=1.0,
        grad_clip=None,
    )
    hierarchical_diagnostics = modular_objective_update(
        hierarchical_model,
        hierarchical_optimizer,
        hierarchical_data,
        scalarized,
        aggregator=OuterAggregator({"method": "sum"}),
        data_weight=1.0,
        coherence_weight=1.0,
        grad_clip=None,
    )

    torch.testing.assert_close(joint_model.weight, hierarchical_model.weight)
    assert joint_diagnostics["aggregation_scope"] == "joint"
    assert hierarchical_diagnostics["aggregation_scope"] == "hierarchical"


def test_zero_global_weight_bypasses_aggregation_and_all_zero_weights_reject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data_loss = (model.weight - 2.0).square().sum()
    family_losses = OrderedDict(A=(model.weight + 3.0).square().sum())

    def unexpected_aggregation(self: OuterAggregator, gradients: torch.Tensor):
        raise AssertionError("a sole active objective must bypass the outer aggregator")

    monkeypatch.setattr(OuterAggregator, "__call__", unexpected_aggregation)
    diagnostics = modular_objective_update(
        model,
        optimizer,
        data_loss,
        family_losses,
        aggregator=OuterAggregator({"method": "sum"}),
        data_weight=1.0,
        coherence_weight=0.0,
        grad_clip=None,
    )

    assert diagnostics["outer_aggregation"]["method"] == "single_objective"
    assert diagnostics["outer_aggregation"]["objective_names"] == ["data_retention"]

    before = model.weight.detach().clone()
    with pytest.raises(ValueError, match="weight|coefficient|active"):
        modular_objective_update(
            model,
            optimizer,
            data_loss,
            family_losses,
            aggregator=OuterAggregator({"method": "sum"}),
            data_weight=0.0,
            coherence_weight=0.0,
            grad_clip=None,
        )
    torch.testing.assert_close(model.weight, before)


def test_zero_gradient_active_family_is_kept_and_inactive_data_is_removed() -> None:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    data_loss = (model.weight - 5.0).square().sum()
    family_losses = OrderedDict(
        zero=model.weight.sum() * 0.0,
        active=(model.weight - 2.0).square().sum(),
    )

    diagnostics = modular_objective_update(
        model,
        optimizer,
        data_loss,
        family_losses,
        aggregator=OuterAggregator({"method": "sum"}),
        data_weight=0.0,
        coherence_weight=1.0,
        grad_clip=None,
    )

    outer = diagnostics["outer_aggregation"]
    assert outer["objective_names"] == ["zero", "active"]
    assert outer["gradient_norms"][0] == 0.0
    assert len(outer["directional_products"]) == 2


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


def test_nonfinite_family_loss_rejects_optimizer_step() -> None:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = _CountingSGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()

    with pytest.raises(FloatingPointError):
        modular_objective_update(
            model,
            optimizer,
            model.weight.square().sum(),
            OrderedDict(A=model.weight * torch.tensor(float("nan"))),
            aggregator=OuterAggregator({"method": "sum"}),
            data_weight=1.0,
            coherence_weight=1.0,
            grad_clip=None,
        )

    assert optimizer.step_calls == 0
    torch.testing.assert_close(model.weight, before)


def test_nonfinite_joint_direction_rejects_optimizer_step() -> None:
    class InvalidDirection:
        method = "invalid-test"

        def __call__(self, gradients: torch.Tensor):
            return torch.full_like(gradients[0], float("nan")), {"method": self.method}

    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = _CountingSGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()

    with pytest.raises(FloatingPointError, match="invalid|non-finite|direction"):
        modular_objective_update(
            model,
            optimizer,
            model.weight.square().sum(),
            OrderedDict(A=(model.weight - 2.0).square().sum()),
            aggregator=InvalidDirection(),  # type: ignore[arg-type]
            data_weight=1.0,
            coherence_weight=1.0,
            grad_clip=None,
        )

    assert optimizer.step_calls == 0
    torch.testing.assert_close(model.weight, before)
