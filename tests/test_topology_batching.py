"""Batched persistence must agree with the scalar objective and its derivatives."""

import pytest
import torch

pytest.importorskip("gudhi")
from helpers.persistence import config

from phycoflow_reconstruction.coherence.families.topology.persistence import (
    Diagram,
    cubical_diagrams,
    sliced_diagram_distance,
    sliced_diagram_distances,
)
from phycoflow_reconstruction.coherence.families.topology.persistence_objective import (
    PersistenceTopologyObjective,
)


@pytest.mark.parametrize("limit", ["1", "2000000"])
def test_ragged_batched_value_and_gradient_matches_scalar(monkeypatch, limit):
    monkeypatch.setenv("PHYCOFLOW_TOPOLOGY_DISTANCE_ELEMENTS", limit)
    rng = torch.Generator().manual_seed(18)
    left, right, inputs = [], [], []
    for n, m in [(0, 0), (0, 4), (3, 0), (2, 9), (13, 2)]:
        x = torch.randn(n, 2, generator=rng, dtype=torch.float64).requires_grad_()
        y = torch.randn(m, 2, generator=rng, dtype=torch.float64).requires_grad_()
        a = torch.randn(2, generator=rng, dtype=torch.float64).requires_grad_()
        b = torch.randn(2, generator=rng, dtype=torch.float64).requires_grad_()
        left.append(Diagram(x, a))
        right.append(Diagram(y, b))
        inputs.extend([x, y, a, b])
    actual = sliced_diagram_distances(left, right, normalization=27.0)
    expected = torch.stack(
        [sliced_diagram_distance(a, b, normalization=27.0) for a, b in zip(left, right)]
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    ag = torch.autograd.grad(actual.sum(), inputs, retain_graph=True)
    eg = torch.autograd.grad(expected.sum(), inputs)
    for a, b in zip(ag, eg):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("periodic", [True, False])
def test_packed_pairing_threads_and_cache_preserve_rows_and_gradients(monkeypatch, periodic):
    x = torch.randn(
        8, 7, 9, generator=torch.Generator().manual_seed(42), dtype=torch.float64
    ).requires_grad_()
    monkeypatch.setenv("PHYCOFLOW_TOPOLOGY_WORKERS", "1")
    separate = [cubical_diagrams(row[None], periodic=periodic, locations=True)[0] for row in x]
    monkeypatch.setenv("PHYCOFLOW_TOPOLOGY_WORKERS", "3")
    for _ in range(2):
        combined = cubical_diagrams(x, periodic=periodic, locations=True, cacheable=True)
        for a, b in zip(separate, combined):
            for d in (0, 1):
                torch.testing.assert_close(a[d].finite, b[d].finite, rtol=0, atol=0)
                torch.testing.assert_close(a[d].essential, b[d].essential, rtol=0, atol=0)
                torch.testing.assert_close(
                    a[d].finite_locations, b[d].finite_locations, rtol=0, atol=0
                )
        loss = sum(item[d].finite.square().sum() for item in combined for d in (0, 1))
        original = sum(item[d].finite.square().sum() for item in separate for d in (0, 1))
        torch.testing.assert_close(
            torch.autograd.grad(loss, x, retain_graph=True)[0],
            torch.autograd.grad(original, x, retain_graph=True)[0],
            rtol=0,
            atol=0,
        )


def test_batched_objective_preserves_all_components_and_field_gradients(monkeypatch):
    objective = PersistenceTopologyObjective(config(), ("a", "b", "c"))
    y = torch.randn(8, 3, 7, 9, generator=torch.Generator().manual_seed(143), dtype=torch.float64)
    x = (y + 0.12 * y.roll(2, -1)).requires_grad_()
    monkeypatch.setenv("PHYCOFLOW_TOPOLOGY_BATCHED", "0")
    expected = objective(x, y)
    eg = torch.autograd.grad(expected.scalar_loss, x)[0]
    monkeypatch.setenv("PHYCOFLOW_TOPOLOGY_BATCHED", "1")
    actual = objective(x, y)
    for name, term in expected.component_results.items():
        torch.testing.assert_close(
            actual.component_results[name].per_sample_cost,
            term.per_sample_cost,
            rtol=1e-12,
            atol=1e-12,
        )
    torch.testing.assert_close(
        torch.autograd.grad(actual.scalar_loss, x)[0], eg, rtol=1e-10, atol=1e-12
    )
