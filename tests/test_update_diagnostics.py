import pytest
import torch

from phycoflow_reconstruction.training.update_diagnostics import UpdateProbe


def test_actual_update_includes_optimizer_momentum_and_decay():
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    opt = torch.optim.AdamW([p], lr=0.1, weight_decay=0.2)
    for _ in range(3):
        opt.zero_grad()
        p.sum().backward()
        opt.step()
    # Objective gradient can oppose the momentum-driven accepted step.
    loss = -p.sum()
    before = p.detach().clone()
    probe = UpdateProbe.capture({"family": loss}, [p], units={"family": 2.0})
    opt.zero_grad()
    loss.backward()
    opt.step()
    delta = p.detach() - before
    row = probe.measure()
    assert row["family/first_order_benefit"] == pytest.approx(float(delta.sum()) / 2)
    assert row["family/first_order_benefit"] < 0
    assert row["parameter_displacement_norm"] == pytest.approx(float(delta.norm()))
    with pytest.raises(RuntimeError, match="consumed"):
        probe.measure()


def test_diagnostic_capture_reuses_graph_without_mutating_parameters_or_rng():
    p = torch.nn.Parameter(torch.tensor([2.0]))
    q = torch.nn.Parameter(torch.tensor([3.0]))
    loss = p.square().sum()
    rng = torch.get_rng_state()
    probe = UpdateProbe.capture({"self": loss}, [p, q])
    assert torch.equal(rng, torch.get_rng_state())
    assert p.grad is None and q.grad is None
    loss.backward()
    with torch.no_grad():
        p.add_(-0.1 * p.grad)
    assert probe.measure()["self/first_order_benefit"] > 0
