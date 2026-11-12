import pytest
import torch

from phycoflow_reconstruction.coherence.base import weighted_w2_columns
from phycoflow_reconstruction.coherence.state_compute import (
    exact_common_embeddings,
    select_directions,
)


@pytest.mark.parametrize("fields", [1, 2, 4])
def test_exact_embeddings_match_pairwise_weighted_values_and_gradients(fields):
    generator = torch.Generator().manual_seed(31 + fields)
    endpoints = [
        torch.randn(2, n, fields, generator=generator, dtype=torch.double, requires_grad=True)
        for n in (5, 7, 6)
    ]
    masses = [torch.rand(2, x.shape[1], generator=generator, dtype=torch.double) for x in endpoints]
    directions = (
        torch.randn(3, fields, generator=generator, dtype=torch.double)
        if fields > 1
        else torch.empty(0, 1, dtype=torch.double)
    )
    if len(directions):
        directions = directions / directions.norm(dim=1, keepdim=True)
    embedded = exact_common_embeddings(endpoints, masses, directions)
    cached = sum(
        (embedded[0][0] - embedded[i][0]).square().sum()
        + (0 if not len(directions) else (embedded[0][1] - embedded[i][1]).square().sum())
        for i in (1, 2)
    )
    direct = torch.zeros((), dtype=torch.double)
    for i in (1, 2):
        for b in range(2):
            left, right = endpoints[0][b], endpoints[i][b]
            value = weighted_w2_columns(left, right, masses[0][b], masses[i][b]).mean()
            if len(directions):
                value = (
                    value
                    + weighted_w2_columns(
                        left @ directions.T, right @ directions.T, masses[0][b], masses[i][b]
                    ).mean()
                )
            direct = direct + value
    torch.testing.assert_close(cached, direct, rtol=1e-10, atol=1e-11)
    cached_grad = torch.autograd.grad(cached, endpoints, retain_graph=True)
    direct_grad = torch.autograd.grad(direct, endpoints)
    for a, b in zip(cached_grad, direct_grad):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-10)


def test_direction_selection_is_frozen_deterministic_and_single_field_has_marginal_only():
    bank = select_directions(torch.eye(1), torch.ones(1), ["x"], 7)
    assert not bank["mixed_active"] and bank["directions"].shape == (0, 1)
    candidates = torch.randn(20, 4, generator=torch.Generator().manual_seed(1))
    hardness = torch.arange(20, dtype=torch.float64)
    first = select_directions(candidates, hardness, ["a", "b", "c", "d"], 7)
    second = select_directions(candidates, hardness, ["a", "b", "c", "d"], 7)
    assert len(first["selected_ids"]) == 8 and first["sha256"] == second["sha256"]
