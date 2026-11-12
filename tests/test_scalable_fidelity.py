from types import SimpleNamespace

import torch

from phycoflow_reconstruction.training.post_training import _physical_endpoint_risk


def test_shared_member_physical_risk_uses_native_total_and_inclusion():
    first = torch.tensor([[[1.0], [3.0]]], requires_grad=True)
    second = torch.tensor([[[3.0], [1.0]]], requires_grad=True)
    dataset = SimpleNamespace(physical_weights=torch.tensor([1.0, 2.0, 3.0, 4.0]))
    endpoint = {
        "prediction": first,
        "members": (first, second),
        "target": torch.zeros_like(first),
        "mask": torch.ones(1, 2, dtype=torch.bool),
        "measure": {"generated_weights": torch.tensor([[1.0, 3.0]])},
    }
    risk = _physical_endpoint_risk(endpoint, dataset)
    # Half of the four nodes are sampled, with native total mass 10.
    assert torch.allclose(risk, torch.tensor([[4.0]]))
    risk.sum().backward()
    assert first.grad is not None and second.grad is not None
