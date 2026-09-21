"""Endpoint fidelity in inference mode, sharing the topology rollout graph."""

from collections.abc import Mapping

import torch


def post_training_mode(model: torch.nn.Module, config: Mapping) -> None:
    # eval() disables dropout/BN updates, not autograd. An explicit train-mode
    # switch exists only for controlled legacy ablations.
    model.train(config.get("optimization", {}).get("model_mode", "eval") == "train")


def endpoint_retention(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    scale_reference: torch.Tensor,
    variance_floor: float = 1e-8,
) -> torch.Tensor:
    """Equal-field normalized MSE; statistics are detached and target-defined."""
    scale = scale_reference.detach().var(dim=1, unbiased=False).clamp_min(variance_floor)
    return ((prediction - target.detach()).square().mean(dim=1) / scale).mean()
