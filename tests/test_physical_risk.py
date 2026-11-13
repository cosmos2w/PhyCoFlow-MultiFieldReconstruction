import itertools

import pytest
import torch

from phycoflow_reconstruction.coherence.physical_risk import (
    blend_risk_quadratic,
    physical_ensemble_risks,
    physical_field_mse,
)


def test_without_replacement_matches_native_in_expectation_and_gradient():
    x = torch.tensor(
        [[1.0, -2.0], [3.0, 1.0], [-1.0, 4.0]], dtype=torch.float64, requires_grad=True
    )
    w = x.new_tensor([1.0, 2.0, 7.0])
    full = physical_field_mse(x, torch.zeros_like(x), w)
    draws = [
        physical_field_mse(
            x[list(ids)],
            torch.zeros_like(x[list(ids)]),
            w[list(ids)],
            total_mass=w.sum(),
            sampling="without_replacement",
            probabilities=2 / 3,
        )
        for ids in itertools.combinations(range(3), 2)
    ]
    estimate = torch.stack(draws).mean(0)
    torch.testing.assert_close(estimate, full)
    torch.testing.assert_close(
        torch.autograd.grad(estimate.sum(), x, retain_graph=True)[0],
        torch.autograd.grad(full.sum(), x)[0],
    )


def test_with_replacement_probability_and_duplicates():
    x = torch.tensor([[1.0], [3.0]], dtype=torch.float64)
    w, p = x.new_tensor([1.0, 3.0]), x.new_tensor([0.8, 0.2])
    expectation = sum(
        p[a]
        * p[b]
        * physical_field_mse(
            x[[a, b]],
            torch.zeros_like(x[[a, b]]),
            w[[a, b]],
            total_mass=w.sum(),
            sampling="with_replacement",
            probabilities=p[[a, b]],
        )
        for a, b in itertools.product(range(2), repeat=2)
    )
    torch.testing.assert_close(expectation, physical_field_mse(x, torch.zeros_like(x), w))


def test_native_total_cannot_be_replaced_by_sampled_mass():
    x = torch.ones(2, 3, 4)
    with pytest.raises(ValueError, match="native total"):
        physical_field_mse(x, x, torch.ones(3), sampling="without_replacement", probabilities=0.5)


def test_blend_risk_quadratic_matches_all_alphas_and_gradients():
    source = torch.tensor([[1.0, 3.0], [2.0, -1.0]], dtype=torch.float64)
    child = torch.tensor([[2.0, -1.0], [1.0, 2.0]], dtype=torch.float64, requires_grad=True)
    target, mass = torch.zeros_like(source), source.new_tensor([1.0, 3.0])
    r0, b, a = blend_risk_quadratic(source, child, target, mass)
    for alpha in (0.0, 0.2, 0.5, 1.0):
        polynomial = r0 + 2 * alpha * b + alpha**2 * a
        direct = physical_field_mse(torch.lerp(source, child, alpha), target, mass)
        torch.testing.assert_close(polynomial, direct)
        torch.testing.assert_close(
            torch.autograd.grad(polynomial.sum(), child, retain_graph=True)[0],
            torch.autograd.grad(direct.sum(), child, retain_graph=True)[0],
        )


def test_signed_mean_risk_j2_identity_and_gradients():
    members = torch.tensor([[[[1.0], [3.0]], [[-2.0], [1.0]]]], dtype=torch.float64, requires_grad=True)
    target = torch.zeros(1, 2, 1, dtype=torch.float64)
    masses = torch.tensor([[1.0, 3.0]], dtype=torch.float64)
    member, mean = physical_ensemble_risks(members, target, masses)
    direct = ((members[:, 0] - target) * (members[:, 1] - target) * masses[..., None]).sum(-2) / masses.sum(-1, keepdim=True)
    torch.testing.assert_close(mean, direct)
    torch.testing.assert_close(torch.autograd.grad(mean.sum(), members, retain_graph=True)[0], torch.autograd.grad(direct.sum(), members)[0])
    assert mean.item() < 0 or member.item() > mean.item()


def test_mean_risk_difference_allows_negative_reference_and_checkpoint():
    from phycoflow_reconstruction.training.fidelity_budget import MeanRiskDifferenceBudget

    budget = MeanRiskDifferenceBudget.from_reference(["a"], [-0.01], [0.02], [0.5])
    risk = torch.tensor([-0.03], requires_grad=True)
    torch.testing.assert_close(budget.violations(risk), torch.tensor([-0.08], dtype=torch.float64))
    assert MeanRiskDifferenceBudget.from_state_dict(budget.state_dict()) == budget


def test_mean_risk_policy_update_and_resume():
    from phycoflow_reconstruction.training.fidelity_budget import (
        ExternalPolicy,
        MeanRiskDifferenceBudget,
    )

    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    budget = MeanRiskDifferenceBudget.from_reference(["a"], [-0.1], [0.01], [0.5])
    policy = ExternalPolicy({"method": "rectify"}, budget, "cpu")
    value = model(torch.ones(1, 1)).reshape(())
    result = policy.update(model, optimizer, value.square() + value, value.reshape(1), None)
    assert result["update_accepted"] and policy.completed_updates == 1
    restored = ExternalPolicy({"method": "rectify"}, budget, "cpu")
    restored.load_state_dict(policy.state_dict())
    assert restored.reference == budget and restored.completed_updates == 1


def test_endpoint_signed_risk_preserves_both_member_gradients():
    from types import SimpleNamespace

    from phycoflow_reconstruction.training.post_training import _physical_endpoint_risk

    first = torch.tensor([[[2.0], [1.0]]], requires_grad=True)
    second = torch.tensor([[[-1.0], [2.0]]], requires_grad=True)
    endpoint = {"prediction": first, "members": (first, second), "target": torch.zeros_like(first), "mask": torch.ones(1, 2, dtype=torch.bool), "measure": {"generated_weights": torch.tensor([[1.0, 2.0]])}}
    risk = _physical_endpoint_risk(endpoint, SimpleNamespace(physical_weights=torch.tensor([1.0, 2.0])), mean_risk=True)
    torch.testing.assert_close(risk, torch.tensor([[2.0 / 3.0]]))
    gradients = torch.autograd.grad(risk.sum(), (first, second))
    assert all(g.abs().sum() > 0 for g in gradients)


def test_snapshot_admission_checks_all_copied_metadata(tmp_path):
    import json

    from phycoflow_reconstruction.training.run_store import file_sha256
    from phycoflow_reconstruction.training.source import _validate_native_source_status

    (tmp_path / "checkpoints").mkdir()
    checkpoint = tmp_path / "checkpoints" / "last.pt"
    checkpoint.write_bytes(b"frozen checkpoint")
    names = ("resolved_config.yaml", "run_manifest.json", "status.json")
    for name in names:
        (tmp_path / name).write_text(name)
    receipt_path = tmp_path / "snapshot_receipt.json"
    receipt_path.write_text(json.dumps({"schema": "frozen_source_snapshot.v1", "checkpoint_sha256": file_sha256(checkpoint), "metadata_sha256": {name: file_sha256(tmp_path / name) for name in names}}))
    config = {"source_run": str(tmp_path), "source_checkpoint": "last.pt", "source": {"admission": "frozen_snapshot", "snapshot_receipt": str(receipt_path)}}
    _validate_native_source_status(config)
    (tmp_path / "resolved_config.yaml").write_text("changed")
    with pytest.raises(ValueError, match="metadata hash"):
        _validate_native_source_status(config)
