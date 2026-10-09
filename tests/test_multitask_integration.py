"""Check coefficient ownership at the legacy/modular trainer boundary."""

import pytest
import torch

from phycoflow_reconstruction.training.post_training import (
    _balanced_weights,
    _sparse_family_gradient_diagnostics,
)


@pytest.mark.parametrize("modular, expected", [(False, (0.2, 3.0)), (True, (0.1, 1.0))])
def test_modular_coefficients_ignore_legacy_config_scales(modular, expected):
    config = {
        "objectives": {"data_retention": {"enabled": True, "weight": 0.1}},
        "optimization": {
            "gradient_balance": "config",
            "config_data_grad_scale": 2.0,
            "config_coherence_grad_scale": 3.0,
        },
    }
    if modular:
        config["optimization"]["multitask"] = {}
    assert _balanced_weights(config, 1.0) == expected


def test_modular_family_diagnostics_name_the_surrogate_without_another_rollout():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]]))
    left, right = model.weight.unbind(dim=1)
    raw = {"A": left.square().sum(), "T": right.square().sum()}
    data_loss = model.weight.square().sum()
    surrogate = 0.2 * raw["A"] + 0.8 * raw["T"]
    report = _sparse_family_gradient_diagnostics(
        model, None, data_loss, surrogate, {name: None for name in raw}, {},
        {"coherence": {}}, step=0, rollout_seed=1, device=torch.device("cpu"),
        raw_family_losses=raw, scalarized=True,
    )
    assert "total_coherence_gradient_norm" not in report
    assert report["total_scalarized_coherence_gradient_norm"] == pytest.approx(
        (0.4**2 + 3.2**2)**0.5
    )
    # The diagnostic kept the original shared graph available for the optimizer.
    surrogate.backward()
    torch.testing.assert_close(model.weight.grad, torch.tensor([[0.4, 3.2]]))
