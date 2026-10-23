from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from phycoflow_reconstruction.contracts import ModelCapabilities, ObservationBatch
from phycoflow_reconstruction.training.fast_lagrangian import (
    training_execution,
    validate_fast_lagrangian,
)
from phycoflow_reconstruction.training.fidelity_budget import ExternalPolicy, FidelityBudget
from phycoflow_reconstruction.training.multitask import OuterAggregator
from phycoflow_reconstruction.training.periodic_config import PeriodicConfigState
from phycoflow_reconstruction.training.post_training import _coherence_objective

FAMILY_NAMES = ("global_distribution", "cross_spectrum", "topology")


class _ParameterModel(nn.Module):
    def __init__(self, value: torch.Tensor | None = None) -> None:
        super().__init__()
        if value is None:
            value = torch.linspace(-0.2, 0.2, 12, dtype=torch.float64)
        self.parameter = nn.Parameter(value.clone())


class _RecordingAdamW(torch.optim.AdamW):
    def step(self, closure=None):
        parameter = self.param_groups[0]["params"][0]
        self.last_gradient = parameter.grad.detach().clone()
        return super().step(closure)


def _family_gradients() -> torch.Tensor:
    return torch.randn((3, 12), generator=torch.Generator().manual_seed(31), dtype=torch.float64)


def _curvatures() -> torch.Tensor:
    return torch.stack(
        (
            torch.linspace(0.02, 0.08, 12, dtype=torch.float64),
            torch.linspace(0.03, 0.12, 12, dtype=torch.float64),
            torch.linspace(0.01, 0.06, 12, dtype=torch.float64),
        )
    )


def _family_losses(parameter: torch.Tensor) -> dict[str, torch.Tensor]:
    gradients = _family_gradients()
    curvature = _curvatures()
    return {
        name: (gradients[index] * parameter).sum()
        + 0.5 * (curvature[index] * parameter.square()).sum()
        for index, name in enumerate(FAMILY_NAMES)
    }


def _risks(parameter: torch.Tensor) -> torch.Tensor:
    return torch.stack(((parameter[0] - 0.4).square() + 0.5,))


def _reference() -> FidelityBudget:
    return FidelityBudget.from_reference(("field",), (0.5,), 2.0, relative_increase=0.0)


def _policy(reference: FidelityBudget) -> ExternalPolicy:
    policy = ExternalPolicy(
        {
            "method": "lagrangian",
            "lagrangian": {"dual_lr": 0.01, "penalty_interval_updates": 1},
        },
        reference,
        "cpu",
    )
    with torch.no_grad():
        policy._multiplier.weight.fill_(0.2)
    return policy


def _optimizer(model: _ParameterModel) -> _RecordingAdamW:
    return _RecordingAdamW(model.parameters(), lr=1.0e-3, weight_decay=0.01, betas=(0.8, 0.95))


def _update_refresh(
    policy: ExternalPolicy,
    model: _ParameterModel,
    optimizer: _RecordingAdamW,
):
    return policy.update_with_coherence_aggregation(
        model,
        optimizer,
        _family_losses(model.parameter),
        _risks(model.parameter),
        None,
        OuterAggregator({"method": "torchjd_config"}),
    )


def test_attached_periodic_interval_one_matches_dense_config_update() -> None:
    pytest.importorskip("torchjd")
    pytest.importorskip("cooper")
    reference = _reference()
    periodic_model = _ParameterModel()
    dense_model = _ParameterModel()
    periodic_optimizer = _optimizer(periodic_model)
    dense_optimizer = _optimizer(dense_model)
    periodic_policy = _policy(reference)
    dense_policy = _policy(reference)
    state = PeriodicConfigState(interval=1, compatibility_key="unit-config-objective")
    periodic_policy.periodic_config = state

    periodic_diagnostics = _update_refresh(periodic_policy, periodic_model, periodic_optimizer)
    _update_refresh(dense_policy, dense_model, dense_optimizer)

    torch.testing.assert_close(periodic_optimizer.last_gradient, dense_optimizer.last_gradient)
    torch.testing.assert_close(periodic_model.parameter, dense_model.parameter)
    torch.testing.assert_close(
        periodic_policy._multiplier().detach(), dense_policy._multiplier().detach()
    )
    assert periodic_diagnostics["family_gradient_evaluations"] == 3
    assert (
        periodic_diagnostics["coherence_aggregation"]["coefficient_reconstruction"][
            "relative_reconstruction_error"
        ]
        < 1.0e-8
    )
    assert periodic_policy.completed_updates == state.completed_steps == 1
    assert state.refresh_due


def test_interval_four_reuse_uses_current_gradients_and_advances_dense_dual_state() -> None:
    pytest.importorskip("torchjd")
    pytest.importorskip("cooper")
    reference = _reference()
    model = _ParameterModel()
    optimizer = _optimizer(model)
    policy = _policy(reference)
    settings_key = "A-B-T/fixed-family-weights/source-bound"
    state = PeriodicConfigState(interval=4, compatibility_key=settings_key)
    policy.periodic_config = state

    refresh_losses = _family_losses(model.parameter)
    refresh_rows = torch.stack(
        [
            torch.autograd.grad(loss, model.parameter, retain_graph=True)[0]
            for loss in refresh_losses.values()
        ]
    )
    refresh_diagnostics = _update_refresh(policy, model, optimizer)
    assert refresh_diagnostics["family_gradient_evaluations"] == 3
    assert any(
        sign < 0
        for sign in refresh_diagnostics["coherence_aggregation"]["coefficient_reconstruction"][
            "coefficient_signs"
        ]
    )
    refreshed_coefficients = torch.tensor(state.coefficients, dtype=torch.float64)
    refreshed_family_gradient = refresh_rows.T @ refreshed_coefficients
    multiplier_before_reuse = policy._multiplier().detach().clone()

    observed_fresh_gradients = []
    for _ in range(3):
        assert not state.refresh_due
        current_losses = _family_losses(model.parameter)
        risks = _risks(model.parameter)
        surrogate = state.reuse_objective(current_losses, scale=1.0)
        (expected_primary_gradient,) = torch.autograd.grad(
            surrogate / reference.objective_scale,
            model.parameter,
            retain_graph=True,
        )
        violations = reference.violations(risks)
        constraint_objective = policy._lagrangian_objective(surrogate.new_zeros(()), violations)
        (expected_constraint_gradient,) = torch.autograd.grad(
            constraint_objective,
            model.parameter,
            retain_graph=True,
        )
        expected_gradient = expected_primary_gradient + expected_constraint_gradient
        diagnostics = policy.update(model, optimizer, surrogate, risks, None)
        torch.testing.assert_close(optimizer.last_gradient, expected_gradient)
        assert diagnostics["update_accepted"]
        assert policy.completed_updates == state.completed_steps + 1
        observed_fresh_gradients.append(expected_primary_gradient.detach().clone())
        multiplier_after = policy._multiplier().detach().clone()
        assert bool((multiplier_after > multiplier_before_reuse).all())
        multiplier_before_reuse = multiplier_after
        state.advance()
        assert policy.completed_updates == state.completed_steps

    assert not torch.allclose(
        observed_fresh_gradients[0], refreshed_family_gradient / reference.objective_scale
    )
    assert state.age == 4
    assert state.refresh_due

    checkpoint = copy.deepcopy(policy.state_dict())
    resumed_policy = _policy(reference)
    resumed_policy.periodic_config = PeriodicConfigState(interval=4, compatibility_key=settings_key)
    resumed_policy.load_state_dict(checkpoint)
    assert resumed_policy.completed_updates == 4
    assert resumed_policy.periodic_config.completed_steps == 4
    assert resumed_policy.periodic_config.age == 4
    assert resumed_policy.periodic_config.refresh_due
    assert resumed_policy.periodic_config.coefficients == state.coefficients
    torch.testing.assert_close(resumed_policy._multiplier().detach(), policy._multiplier().detach())


class _DeterministicFlow(nn.Module):
    capabilities = ModelCapabilities("point", True, True, False, True)

    def __init__(self) -> None:
        super().__init__()
        self.speed = nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
        self.rollout_calls = 0

    def sample_source(self, batch, *, generator=None):
        return torch.zeros(1, 4, 1, dtype=self.speed.dtype)

    def velocity(self, batch, state, time):
        self.rollout_calls += 1
        return self.speed * torch.ones_like(state)


class _MustNotEvaluateTopology:
    family_name = "topology"
    target_use = "training_reference"
    family_weight = 1.0

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("excluded topology family must not be evaluated")


def test_training_t_skip_avoids_topology_family_compute() -> None:
    config = {"optimization": {"fast_lagrangian": {"family_probability": 0.5, "seed": 1}}}
    execution = training_execution(config, 0)
    assert execution["topology_included"] is False
    family = _MustNotEvaluateTopology()
    model = _DeterministicFlow()
    batch = ObservationBatch(
        obs_coords=torch.zeros(1, 1, 1, dtype=torch.float64),
        obs_values=torch.zeros(1, 1, 1, dtype=torch.float64),
        obs_field_ids=torch.zeros(1, 1, dtype=torch.long),
        obs_valid_mask=torch.ones(1, 1, dtype=torch.bool),
        query_coords=torch.arange(4, dtype=torch.float64).view(1, 4, 1),
        query_valid_mask=torch.ones(1, 4, dtype=torch.bool),
        target_fields=None,
        sample_ids=("train-case",),
        obs_indices=torch.zeros(1, 1, dtype=torch.long),
        logical_shapes=((4,),),
        metadata={"query_indices": torch.arange(4).view(1, 4)},
    )

    result, reference_ids = _coherence_objective(
        model,
        batch,
        {"topology": family},
        {"topology": None},
        {
            "coherence": {"compute_budget": {"batch_size": 1, "point_count": 4}},
            "rollout": {"steps": 1, "solver": "euler"},
            "observation_consistency": {"mode": "none", "final_clamp": False},
        },
        step=0,
        generator=torch.Generator().manual_seed(9),
        family_losses={},
        execution=execution,
    )

    assert family.calls == 0
    assert model.rollout_calls == 1
    assert reference_ids == ()
    assert result.diagnostics["families"] == {}
    assert result.scalar_loss.item() == 0.0
    assert result.scalar_loss.requires_grad
    (gradient,) = torch.autograd.grad(result.scalar_loss, model.speed)
    assert gradient.item() == 0.0


def _fast_config(
    *,
    probability: float = 1.0,
    interval: int = 1,
    internal_config: bool = False,
    penalty_interval: int = 1,
    scalarizer: str = "fixed_sum",
) -> dict:
    multitask = {
        "family_scalarization": {"method": scalarizer},
        "external_policy": {
            "method": "lagrangian",
            "lagrangian": {"penalty_interval_updates": penalty_interval},
        },
    }
    if internal_config:
        multitask["coherence_aggregation"] = {"method": "torchjd_config"}
    return {
        "optimization": {
            "fast_lagrangian": {
                "family_probability": probability,
                "coefficient_interval": interval,
            },
            "multitask": multitask,
        }
    }


def test_fast_settings_accept_periodic_or_scalar_routes_and_reject_mixtures() -> None:
    validate_fast_lagrangian(_fast_config())
    validate_fast_lagrangian(_fast_config(interval=4, internal_config=True, scalarizer="none"))

    with pytest.raises(ValueError, match="family sampling requires scalar sum"):
        validate_fast_lagrangian(_fast_config(probability=0.5, interval=4, internal_config=True))
    with pytest.raises(ValueError, match="periodic coefficients require internal ConFIG"):
        validate_fast_lagrangian(_fast_config(interval=4))
    with pytest.raises(ValueError, match="dense Lagrangian fidelity"):
        validate_fast_lagrangian(_fast_config(penalty_interval=2))
    with pytest.raises(ValueError, match="fixed family weights"):
        validate_fast_lagrangian(_fast_config(scalarizer="dwa"))


@pytest.mark.parametrize("probability", [1.0, 0.5])
def test_nonflow_shared_endpoint_and_importance_scalar_per_sample_contract(probability):
    from phycoflow_reconstruction.contracts import FamilyResult
    from phycoflow_reconstruction.training.fidelity_budget import field_mse

    class EndpointModel(nn.Module):
        capabilities = ModelCapabilities("point", False, True, False, False)

        def __init__(self):
            super().__init__()
            self.value = nn.Parameter(torch.tensor(0.25))
            self.calls = 0

        def differentiable_reconstruct(self, batch, *, steps, generator):
            assert batch.target_fields is None
            self.calls += 1
            return self.value.expand(1, 4, 1)

    class EndpointFamily:
        target_use = "paired_supervised"
        family_weight = 1.0

        def __init__(self, name):
            self.family_name = name

        def __call__(self, prediction, reference, **kwargs):
            values = prediction.square().mean((1, 2))
            return FamilyResult({}, values, values.mean())

    model = EndpointModel()
    batch = ObservationBatch(
        obs_coords=torch.zeros(1, 1, 1),
        obs_values=torch.zeros(1, 1, 1),
        obs_field_ids=torch.zeros(1, 1, dtype=torch.long),
        obs_valid_mask=torch.ones(1, 1, dtype=torch.bool),
        query_coords=torch.arange(4).float().view(1, 4, 1),
        query_valid_mask=torch.ones(1, 4, dtype=torch.bool),
        target_fields=torch.ones(1, 4, 1),
        sample_ids=("train-only",),
    )
    endpoint = {}
    result, _ = _coherence_objective(
        model,
        batch,
        {name: EndpointFamily(name) for name in FAMILY_NAMES},
        None,
        {
            "coherence": {"compute_budget": {"batch_size": 1, "point_count": 4}},
            "rollout": {"steps": 2, "solver": "euler"},
            "observation_consistency": {"mode": "none", "final_clamp": False},
        },
        step=0,
        generator=torch.Generator().manual_seed(3),
        endpoint_context=endpoint,
        execution={
            "estimator": "full",
            "topology_included": True,
            "family_probability": probability,
            "topology_seed": 7,
        },
    )
    torch.testing.assert_close(result.scalar_loss, result.per_sample_cost.mean())
    assert float(result.scalar_loss) == pytest.approx(0.25**2 * (2 + 1 / probability))
    assert float(
        result.diagnostics["families"]["topology"]["weighted_contribution"]
    ) == pytest.approx(0.25**2 / probability)
    policy = ExternalPolicy(
        {"method": "lagrangian"}, FidelityBudget.from_reference(("f",), (0.5,), 1.0), "cpu"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    report = policy.update(
        model,
        optimizer,
        result.scalar_loss,
        field_mse(endpoint["prediction"], endpoint["target"], endpoint["mask"]),
        None,
    )
    assert report["model_grad_norm"] > 0
    assert report["fidelity_penalty_active"] is True
    assert report["multipliers_after"][0] > 0
    assert model.calls == 1
