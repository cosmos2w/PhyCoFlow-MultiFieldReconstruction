from __future__ import annotations

import json
from collections import OrderedDict

import pytest
import torch
from torch import nn

from phycoflow_reconstruction.training.multitask import (
    FamilyScalarizer,
    OuterAggregator,
    modular_objective_update,
)


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _family_inputs(values: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, values[index].detach().clone().requires_grad_())
        for index, name in enumerate(("A", "B", "T"))
    )


def _scale_kwargs(names: tuple[str, ...]) -> tuple[dict[str, float], dict[str, float]]:
    return ({name: 1.0 for name in names}, {name: 1.0 for name in names})


def test_fixed_sum_matches_calibrated_value_and_gradient(device: torch.device) -> None:
    losses = OrderedDict(
        A=torch.tensor(2.0, device=device, requires_grad=True),
        B=torch.tensor(3.0, device=device, requires_grad=True),
        T=torch.tensor(4.0, device=device, requires_grad=True),
    )
    scales = {"A": 0.5, "B": 2.0, "T": 1.5}
    weights = {"A": 2.0, "B": 0.25, "T": 1.0}
    scalar, diagnostics = FamilyScalarizer({})(losses, scales=scales, weights=weights)
    expected = sum(losses[name] * scales[name] * weights[name] for name in losses)
    actual_gradient = torch.autograd.grad(scalar, tuple(losses.values()))
    expected_gradient = torch.autograd.grad(expected, tuple(losses.values()))

    torch.testing.assert_close(scalar, expected)
    for actual, reference in zip(actual_gradient, expected_gradient):
        torch.testing.assert_close(actual, reference)
    assert diagnostics["method"] == "fixed_sum"
    assert diagnostics["effective_weights"] == {"A": 1.0, "B": 1.0, "T": 1.0}
    json.dumps(diagnostics)


def test_stch_is_differentiable_and_tracks_input_changes(device: torch.device) -> None:
    pytest.importorskip("torchjd")
    scalarizer = FamilyScalarizer({"method": "stch", "stch": {"mu": 0.1}})
    scales, weights = _scale_kwargs(("A", "B", "T"))
    first = _family_inputs(torch.tensor([1.0, 2.0, 3.0], device=device))
    second = _family_inputs(torch.tensor([1.0, 2.0, 6.0], device=device))

    first_scalar, first_diagnostics = scalarizer(first, scales=scales, weights=weights)
    second_scalar, _ = scalarizer(second, scales=scales, weights=weights)
    gradients = torch.autograd.grad(first_scalar, tuple(first.values()))
    reported_derivative = torch.tensor(
        list(first_diagnostics["effective_weights"].values()), device=device
    )

    assert torch.isfinite(first_scalar)
    assert all(torch.isfinite(gradient) for gradient in gradients)
    assert all(torch.linalg.vector_norm(gradient) > 0 for gradient in gradients)
    torch.testing.assert_close(torch.stack(gradients), reported_derivative)
    assert not torch.isclose(first_scalar, second_scalar)
    assert first_diagnostics["method"] == "stch"
    assert sum(first_diagnostics["effective_weights"].values()) == pytest.approx(1.0 / 3.0)


def test_dwa_matches_torchjd_and_restores_partial_epoch_state(device: torch.device) -> None:
    torchjd = pytest.importorskip("torchjd")
    from torchjd.scalarization import DWA

    assert getattr(torchjd, "__version__", None) is None or torchjd.__version__ == "0.17.1"
    adapter = FamilyScalarizer({"method": "dwa", "dwa": {"temperature": 2.0}})
    library = DWA(temperature=2.0)
    scales, weights = _scale_kwargs(("A", "B", "T"))
    adapter.step()
    assert adapter.state_dict()["epochs_completed"] == 0

    epochs = (
        ((9.0, 2.0, 4.0), (7.0, 3.0, 5.0)),
        ((3.0, 8.0, 6.0), (4.0, 7.0, 5.0)),
        ((6.0, 2.0, 3.0), (5.0, 4.0, 7.0)),
        ((2.0, 8.0, 3.0), (3.0, 6.0, 5.0)),
    )
    restored: FamilyScalarizer | None = None
    for epoch_index, batches in enumerate(epochs):
        for batch_index, batch in enumerate(batches):
            values = torch.tensor(batch, device=device)
            adapter_value, diagnostics = adapter(
                _family_inputs(values), scales=scales, weights=weights
            )
            library_value = library(values)
            torch.testing.assert_close(adapter_value, library_value)
            if epoch_index == 2 and batch_index == 0:
                assert diagnostics["effective_weights"] != {"A": 1.0, "B": 1.0, "T": 1.0}
                restored = FamilyScalarizer({"method": "dwa", "dwa": {"temperature": 2.0}})
                restored.load_state_dict(json.loads(json.dumps(adapter.state_dict())))
            elif restored is not None:
                restored_value, _ = restored(
                    _family_inputs(values), scales=scales, weights=weights
                )
                torch.testing.assert_close(restored_value, library_value)
        adapter.step()
        library.step()
        if restored is not None:
            restored.step()

    assert restored is not None
    assert adapter.state_dict()["epochs_completed"] == 4
    with pytest.raises(TypeError, match="requires saved scalarizer state"):
        FamilyScalarizer({"method": "dwa"}).load_state_dict(None)


def test_fixed_sum_rejects_nonfinite_scalarized_output(device: torch.device) -> None:
    losses = OrderedDict(
        A=torch.tensor(3.0e38, device=device, requires_grad=True),
        B=torch.tensor(3.0e38, device=device, requires_grad=True),
    )
    with pytest.raises(FloatingPointError, match="returned non-finite loss"):
        FamilyScalarizer({"method": "fixed_sum"})(
            losses,
            scales={"A": 1.0, "B": 1.0},
            weights={"A": 1.0, "B": 1.0},
        )


@pytest.mark.parametrize(
    "gradients",
    [
        [[1.0, 2.0, 0.0], [2.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [-0.5, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
    ],
    ids=("aligned", "conflicting", "orthogonal", "zero-objective", "all-zero", "opposed"),
)
@pytest.mark.parametrize("method", ["sum", "torchjd_config", "upgrad", "cagrad"])
def test_outer_aggregators_return_finite_directions(
    device: torch.device, method: str, gradients: list[list[float]]
) -> None:
    if method != "sum":
        pytest.importorskip("torchjd")
    matrix = torch.tensor(gradients, device=device)
    settings = {"method": method}
    if method == "cagrad":
        settings["cagrad"] = {"c": 0.5}
    aggregator = OuterAggregator(settings)

    direction, diagnostics = aggregator(matrix)

    assert direction.shape == (matrix.shape[1],)
    assert direction.device == device
    assert torch.isfinite(direction).all()
    assert len(diagnostics["directional_products"]) == 2
    assert diagnostics["method"] == method
    json.dumps(diagnostics)


@pytest.mark.parametrize("scale", [1.0, 1.0e-8])
@pytest.mark.parametrize("orientation", [1.0, -1.0, 0.0])
def test_outer_cosine_preserves_scale_and_handles_zero_gradients(
    device: torch.device, scale: float, orientation: float
) -> None:
    gradients = scale * torch.tensor([[1.0, 0.0], [orientation, 0.0]], device=device)
    _, diagnostics = OuterAggregator({"method": "sum"})(gradients)
    assert diagnostics["gradient_cosine"] == pytest.approx(orientation)


def test_torchjd_config_is_unconditional_on_positive_cosine(device: torch.device) -> None:
    pytest.importorskip("torchjd")
    gradients = torch.tensor([[1.0, 0.0], [0.1, 1.0]], device=device)
    direction, diagnostics = OuterAggregator({"method": "torchjd_config"})(gradients)
    legacy_aligned_sum = gradients.sum(dim=0)

    assert diagnostics["gradient_cosine"] > 0
    assert not torch.allclose(direction, legacy_aligned_sum)
    assert diagnostics["method"] == "torchjd_config"


def test_cagrad_accepts_zero_scale_boundary(device: torch.device) -> None:
    pytest.importorskip("torchjd")
    gradients = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)
    direction, diagnostics = OuterAggregator(
        {"method": "cagrad", "cagrad": {"c": 0.0}}
    )(gradients)

    assert torch.isfinite(direction).all()
    assert diagnostics["method"] == "cagrad"


def test_modular_update_preserves_unused_and_complex_parameters(device: torch.device) -> None:
    class MixedParameters(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.real = nn.Parameter(torch.tensor([1.0], device=device))
            self.unused = nn.Parameter(torch.tensor([2.0], device=device))
            self.complex = nn.Parameter(torch.tensor([1.0 + 2.0j], device=device))

    model = MixedParameters()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    data_loss = model.real.square().sum() + model.complex.real.square().sum()
    coherence_loss = (model.real - 1.5).square().sum() + model.complex.imag.square().sum()

    diagnostics = modular_objective_update(
        model,
        optimizer,
        data_loss,
        coherence_loss,
        aggregator=OuterAggregator({"method": "sum"}),
        data_weight=0.1,
        coherence_weight=1.0,
        grad_clip=1.0,
    )

    assert torch.is_complex(model.complex.grad)
    assert model.complex.grad.imag.abs().item() > 0.5
    assert torch.equal(model.unused.grad, torch.zeros_like(model.unused))
    assert torch.isfinite(model.real).all() and torch.isfinite(model.complex).all()
    assert diagnostics["update_mode"] == "multitask_modular"
    json.dumps(diagnostics)


@pytest.mark.parametrize("method", ["sum", "torchjd_config", "upgrad", "cagrad"])
@pytest.mark.parametrize("data_weight,coherence_weight", [(0.0, 1.0), (1.0, 0.0)])
def test_disabled_objective_uses_the_active_gradient_without_two_task_aggregation(
    device: torch.device, method: str, data_weight: float, coherence_weight: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_aggregation(self, gradients):
        raise AssertionError("disabled objectives must not be passed to a two-task solver")

    monkeypatch.setattr(OuterAggregator, "__call__", unexpected_aggregation)
    model = nn.Linear(1, 1, bias=False, device=device)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data_loss = model.weight.square().sum()
    coherence_loss = (model.weight - 2.0).square().sum()
    active_loss = coherence_loss if data_weight == 0 else data_loss
    expected_gradient = torch.autograd.grad(active_loss, model.weight, retain_graph=True)[0]
    expected_weight = model.weight.detach() - 0.1 * expected_gradient

    diagnostics = modular_objective_update(
        model, optimizer, data_loss, coherence_loss,
        aggregator=OuterAggregator({"method": method}),
        data_weight=data_weight, coherence_weight=coherence_weight, grad_clip=None,
    )

    torch.testing.assert_close(model.weight, expected_weight)
    assert diagnostics["update_mode"] == "modular_single_objective"
    assert diagnostics["outer_method"] == method
    assert diagnostics["outer_aggregation"]["method"] == "single_objective"
    assert diagnostics["config_fallback_used"] is False


def test_modular_update_rejects_nonfinite_direction_before_step(device: torch.device) -> None:
    class CountingSGD(torch.optim.SGD):
        steps = 0

        def step(self, closure=None):
            self.steps += 1
            return super().step(closure)

    class InvalidAggregator:
        method = "invalid-test"

        def __call__(self, gradients: torch.Tensor):
            return torch.full_like(gradients[0], float("nan")), {"method": self.method}

    model = nn.Linear(1, 1, bias=False, device=device)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = CountingSGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()
    data_loss = model.weight.square().sum()
    coherence_loss = (model.weight - 2.0).square().sum()

    with pytest.raises(FloatingPointError, match="invalid direction"):
        modular_objective_update(
            model,
            optimizer,
            data_loss,
            coherence_loss,
            aggregator=InvalidAggregator(),  # type: ignore[arg-type]
            data_weight=1.0,
            coherence_weight=1.0,
            grad_clip=None,
        )

    assert optimizer.steps == 0
    torch.testing.assert_close(model.weight, before)
