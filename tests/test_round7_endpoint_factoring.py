"""Exact endpoint cotangents reproduce parameter rows and the optimizer action."""
from copy import deepcopy

import pytest
import torch

from phycoflow_reconstruction.training.gradient_balance import _flat_gradient
from phycoflow_reconstruction.training.multitask import EndpointBoundary, OuterAggregator
from phycoflow_reconstruction.training.fidelity_budget import ExternalPolicy
from test_fidelity_budget_config_direction import _TwoParameterModel, _reference


@pytest.mark.parametrize('members', [1, 2])
@pytest.mark.parametrize('chunk', [1, 2, 3])
def test_rows_include_all_members_complement_complex_and_unused(members, chunk):
    complex_parameter = torch.nn.Parameter(torch.tensor([0.4+0.2j, -0.1+0.3j], dtype=torch.complex128))
    unused = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    parameters = [complex_parameter, unused]
    selected = [(complex_parameter * (index + 1)).abs().square() for index in range(members)]
    complement = [(complex_parameter.conj() * (index + 2)).abs().square() for index in range(members)]
    boundary = EndpointBoundary()
    leaf_selected = [boundary.leaf(value) for value in selected]
    leaf_complement = [boundary.leaf(value, complement=True) for value in complement]
    def families(values):
        y = torch.stack(values)
        return [y.sum(), y.square().sum() * 1e-8, y.sin().sum()]
    baseline = families(selected)
    factored = families(leaf_selected)
    torch.testing.assert_close(boundary.rows(factored, parameters, chunk_size=chunk),
        torch.stack([_flat_gradient(loss, parameters) for loss in baseline]), rtol=1e-12, atol=1e-14)
    field = sum(v.square().sum() for v in selected + complement)
    field_leaf = sum(v.square().sum() for v in leaf_selected + leaf_complement)
    torch.testing.assert_close(boundary.rows([field_leaf], parameters, chunk_size=1, include_complement=True)[0],
        _flat_gradient(field, parameters), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('chunk', [1, 2])
@pytest.mark.parametrize('active', [False, True])
def test_factored_config_adamw_and_dual_match(chunk, active):
    pytest.importorskip('torchjd')
    pytest.importorskip('cooper')
    models = [_TwoParameterModel(), _TwoParameterModel()]
    models[1].load_state_dict(deepcopy(models[0].state_dict()))
    policies = [ExternalPolicy({'method': 'lagrangian', 'lagrangian': {'dual_lr': 0.01}}, _reference(), 'cpu') for _ in models]
    for policy in policies:
        state = policy.state_dict()
        state['multipliers'] = [0.2, 0.3, 0.4] if active else [0., 0., 0.]
        policy.load_state_dict(state)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=0.001) for model in models]
    for _ in range(2):
        for index, (model, policy, optimizer) in enumerate(zip(models, policies, optimizers)):
            endpoint = torch.stack((model.x, model.y))
            complement = torch.stack((model.x + model.y, model.x - model.y))
            boundary = EndpointBoundary() if index else None
            if boundary is not None:
                endpoint = boundary.leaf(endpoint)
                complement = boundary.leaf(complement, complement=True)
            x, y = endpoint.unbind()
            families = {'state': x.square() + .1*y, 'scale': y.square() + .2*x,
                        'shape': (x-y).square() + .1*x*y}
            risks = torch.stack((x.square(), y.square(), complement[0].square()))
            aggregator = OuterAggregator({'method': 'torchjd_config', 'family_vjp': 'factored' if index else 'separate', 'vjp_chunk_size': chunk})
            policy.update_with_coherence_aggregation(model, optimizer, families, risks, .1, aggregator, endpoint_boundary=boundary)
        for left, right in zip(models[0].parameters(), models[1].parameters()):
            torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-12)
        assert policies[0]._multipliers() == pytest.approx(policies[1]._multipliers())
        for left, right in zip(optimizers[0].state.values(), optimizers[1].state.values()):
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(left[key], right[key], rtol=1e-11, atol=1e-12)


def test_real_deterministic_coordinate_model_j1_adapter():
    from phycoflow_reconstruction.contracts import ObservationBatch
    from phycoflow_reconstruction.models.deterministic.coordinate_mlp import CoordinateMLP
    from phycoflow_reconstruction.training.rollout import differentiable_reconstruction

    torch.manual_seed(72)
    model = CoordinateMLP(coordinate_dim=1, num_fields=2, hidden_dim=8, fourier_bands=2).double()
    batch = ObservationBatch(
        obs_coords=torch.zeros(1, 2, 1, dtype=torch.float64),
        obs_values=torch.tensor([[[.2], [.5]]], dtype=torch.float64),
        obs_field_ids=torch.tensor([[0, 1]]), obs_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        query_coords=torch.linspace(0, 1, 7, dtype=torch.float64).reshape(1, 7, 1),
        query_valid_mask=torch.ones(1, 7, dtype=torch.bool), target_fields=None,
        sample_ids=('deterministic_fixture',), obs_indices=torch.tensor([[0, 1]]),
        logical_shapes=((7,),), metadata={'query_indices': torch.arange(7).view(1, 7)})
    endpoint = differentiable_reconstruction(model, batch, steps=1, solver='euler',
        generator=torch.Generator().manual_seed(3), observation_config={'mode':'none'})
    boundary = EndpointBoundary()
    leaf = boundary.leaf(endpoint)
    losses = lambda value: [value.square().mean(), value.sin().mean(), value.exp().mean()]
    params = [p for p in model.parameters() if p.requires_grad]
    expected = torch.stack([_flat_gradient(loss, params) for loss in losses(endpoint)])
    actual = boundary.rows(losses(leaf), params, chunk_size=2)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)
