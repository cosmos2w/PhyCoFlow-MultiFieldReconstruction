from copy import deepcopy

import pytest
import torch
from torch import nn

from phycoflow_reconstruction.training.topology_constraints import (
    CONSTRAINT_DEFAULTS,
    RegularizedTopologyAdam,
    regularized_fidelity_loss,
)


def test_regularized_step_matches_single_adam_backward_without_candidate_evaluation():
    model = nn.Linear(2, 1, bias=False)
    reference = deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    expected_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
    controller = RegularizedTopologyAdam(model, optimizer, CONSTRAINT_DEFAULTS)
    calls = []
    model.weight.register_hook(lambda gradient: calls.append(True))
    loss = model.weight.square().sum()
    score = model.weight.sum()

    def evaluate(candidate):
        pytest.fail("single-backward method must not evaluate a candidate")

    report = controller.step(
        score,
        {"fidelity.anchor.u#0": loss},
        {"fidelity.anchor.u#0": 0.01},
        {"fidelity.anchor.u#0": 1e-7},
        evaluate,
        proposal_loss=loss,
    )
    reference.weight.square().sum().backward()
    expected_optimizer.step()
    torch.testing.assert_close(model.weight, reference.weight, atol=0, rtol=0)
    assert len(calls) == 1
    assert report["constraint_checks"] == 0
    assert report["balanced_topology_committed"] is None
    assert report["acceptance_definition"] == "finite_optimizer_step_not_constraint_certification"
    assert controller.state_dict() == {"attempted": 1, "accepted": 1, "rejected": 0}


def test_soft_penalty_pushes_endpoint_and_anchor_back_inside_budget():
    endpoint = torch.tensor([1.0, 1.2, 0.5], requires_grad=True)
    anchor = torch.tensor([0.0025, 0.005, 0.01], requires_grad=True)
    fidelity = {
        **{f"fidelity.endpoint.u#{i}": v for i, v in enumerate(endpoint)},
        **{f"fidelity.anchor.u#{i}": v for i, v in enumerate(anchor)},
    }
    bounds = {name: 1.0 if "endpoint" in name else 0.01 for name in fidelity}
    loss = regularized_fidelity_loss(fidelity, bounds, CONSTRAINT_DEFAULTS)
    ge, ga = torch.autograd.grad(loss, [endpoint, anchor])
    assert ge[0] > 0 and ge[1] > ge[0] and ge[2] == 0
    assert torch.all(ga > 0) and ga[2] > ga[1] > ga[0]
    # Duplicating a batch changes neither loss nor total gradient per source item.
    duplicated = {**fidelity, **{k + "copy": v for k, v in fidelity.items()}}
    torch.testing.assert_close(
        regularized_fidelity_loss(
            duplicated,
            {**bounds, **{k + "copy": v for k, v in bounds.items()}},
            CONSTRAINT_DEFAULTS,
        ),
        loss,
    )


def test_nonfinite_gradient_cannot_modify_model_or_adam_state():
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    controller = RegularizedTopologyAdam(model, optimizer, CONSTRAINT_DEFAULTS)
    before = model.weight.detach().clone()
    model.weight.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    loss = model.weight.square().sum()
    with pytest.raises(FloatingPointError):
        controller.step(
            loss,
            {"fidelity.anchor.u#0": loss},
            {"fidelity.anchor.u#0": 0.01},
            {"fidelity.anchor.u#0": 0.0},
            None,
            proposal_loss=loss,
        )
    torch.testing.assert_close(before, model.weight, atol=0, rtol=0)
    assert not optimizer.state and controller.counts["attempted"] == 0
