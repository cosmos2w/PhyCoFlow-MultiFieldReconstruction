"""Assignment oracles, periodic locations, gradients, and spatial failure modes."""

import math
from copy import deepcopy
from itertools import combinations, permutations

import numpy as np
import pytest
import torch

pytest.importorskip("gudhi")

from helpers.persistence import config

from phycoflow_reconstruction.coherence.families.topology.persistence import (
    Diagram,
    cubical_diagrams,
    sliced_diagram_distance,
)
from phycoflow_reconstruction.coherence.families.topology.persistence_objective import (
    PersistenceTopologyObjective,
    validate_persistence_config,
)
from phycoflow_reconstruction.coherence.families.topology.spatial_persistence import (
    spatial_diagram_distance,
)


def diagram(values, positions=None):
    points = torch.tensor(values, dtype=torch.float64).reshape(-1, 2)
    return Diagram(
        points,
        torch.empty(0, dtype=torch.float64),
        None if positions is None else torch.tensor(positions, dtype=torch.float64).reshape(-1, 2),
    )


def exhaustive_assignment(a, b):
    """Independent enumerate-all-partial-matchings oracle, including diagonals."""
    best = float("inf")
    for count in range(min(len(a), len(b)) + 1):
        for left in combinations(range(len(a)), count):
            for right in permutations(range(len(b)), count):
                cost = sum(abs(a[i] - b[j]).sum() for i, j in zip(left, right))
                cost += sum(abs(a[i, 1] - a[i, 0]) for i in range(len(a)) if i not in left)
                cost += sum(abs(b[j, 1] - b[j, 0]) for j in range(len(b)) if j not in right)
                best = min(best, cost)
    return best


@pytest.mark.parametrize("n,m", [(0, 0), (0, 3), (3, 0), (1, 2), (3, 3)])
def test_zero_spatial_weight_matches_exhaustive_assignment(n, m):
    rng = np.random.default_rng(391 + n + m)
    a, b = np.sort(rng.normal(size=(n, 2)), axis=-1), np.sort(rng.normal(size=(m, 2)), axis=-1)
    actual = spatial_diagram_distance(diagram(a), diagram(b), periodic=True, spatial_weight=0)
    assert actual.item() == pytest.approx(exhaustive_assignment(a, b), abs=1e-12)


def test_periodic_distance_crosses_seam_and_essential_births_keep_old_rule():
    a, b = diagram([[0.0, 4.0]], [[0.98, 0.5]]), diagram([[0.0, 4.0]], [[0.02, 0.5]])
    periodic = spatial_diagram_distance(a, b, periodic=True)
    planar = spatial_diagram_distance(a, b, periodic=False)
    assert periodic.item() == pytest.approx(0.04 / math.sqrt(2))
    assert planar.item() == pytest.approx(0.96 / math.sqrt(2))
    a.essential, b.essential = torch.tensor([2.0, -1.0]), torch.tensor([0.0, 3.0])
    assert spatial_diagram_distance(a, b, periodic=True).item() == pytest.approx(
        periodic.item() + 0.2
    )


@pytest.mark.parametrize("mode", ["additive", "multiplicative"])
@pytest.mark.parametrize("periodic", [True, False])
def test_rectangular_spatial_assignment_matches_all_partial_matchings(mode, periodic):
    rng = np.random.default_rng(342)
    for n, m in [(1, 3), (3, 1), (3, 4), (4, 3)]:
        a = np.sort(rng.normal(size=(n, 2)), axis=-1)
        b = np.sort(rng.normal(size=(m, 2)), axis=-1)
        u, v = rng.random((n, 2)), rng.random((m, 2))
        value = np.abs(a[:, None] - b[None]).sum(-1)
        delta = np.abs(u[:, None] - v[None])
        if periodic:
            delta = np.minimum(delta, 1 - delta)
        spatial = 5.0 * np.linalg.norm(delta, axis=-1) / math.sqrt(2)
        cost = value + spatial if mode == "additive" else value * (1 + spatial)
        optimum = math.inf
        for count in range(min(n, m) + 1):
            for left in combinations(range(n), count):
                for right in permutations(range(m), count):
                    candidate = sum(cost[i, j] for i, j in zip(left, right))
                    candidate += sum(a[i, 1] - a[i, 0] for i in range(n) if i not in left)
                    candidate += sum(b[j, 1] - b[j, 0] for j in range(m) if j not in right)
                    optimum = min(optimum, candidate)
        actual = spatial_diagram_distance(
            diagram(a, u), diagram(b, v), periodic=periodic, spatial_weight=5.0, spatial_mode=mode
        )
        assert actual.item() == pytest.approx(optimum, abs=1e-12)


@pytest.mark.parametrize("periodic", [True, False])
def test_fortran_creator_coordinates_match_birth_values(periodic):
    field = torch.randn(1, 7, 9, generator=torch.Generator().manual_seed(67), dtype=torch.float64)
    result = cubical_diagrams(field, periodic=periodic, locations=True)[0]
    extent = torch.tensor([9, 7] if periodic else [8, 6])
    for item in result.values():
        xy = (item.finite_locations * extent).round().long()
        torch.testing.assert_close(field[0, xy[:, 1], xy[:, 0]], item.finite[:, 0])


def test_translated_copy_exposes_value_gradient_distinction():
    rng = torch.Generator().manual_seed(730)
    target = torch.randn(1, 7, 9, generator=rng, dtype=torch.float64)
    pred = target.roll(2, -1).requires_grad_()
    ref = cubical_diagrams(target, periodic=True, locations=True)[0]
    gen = cubical_diagrams(pred, periodic=True, locations=True)[0]
    assert sum(sliced_diagram_distance(gen[d], ref[d]) for d in (0, 1)) == 0
    assert (
        sum(
            spatial_diagram_distance(gen[d], ref[d], periodic=True, spatial_weight=0)
            for d in (0, 1)
        )
        == 0
    )
    assert (
        sum(
            spatial_diagram_distance(gen[d], ref[d], periodic=True, spatial_mode="multiplicative")
            for d in (0, 1)
        )
        == 0
    )
    additive = sum(
        spatial_diagram_distance(gen[d], ref[d], periodic=True, spatial_weight=0.01) for d in (0, 1)
    )
    assert additive > 0
    # Small enough spatial weight leaves matching by identical persistence intact.
    # Locations are fixed grid indices: a nonzero score does not imply a movement gradient.
    assert torch.count_nonzero(torch.autograd.grad(additive, pred)[0]) == 0


@pytest.mark.parametrize("mode", ["additive", "multiplicative"])
def test_spatial_assignment_gradcheck_and_identity(mode):
    g = torch.Generator().manual_seed(94)
    target = torch.randn(1, 5, 6, generator=g, dtype=torch.float64)
    pred = (
        target + 0.13 * torch.randn(target.shape, generator=g, dtype=torch.float64)
    ).requires_grad_()
    ref = cubical_diagrams(target, periodic=True, locations=True)[0]

    def loss(x):
        diagrams = cubical_diagrams(x, periodic=True, locations=True)[0]
        return sum(
            spatial_diagram_distance(
                diagrams[d],
                ref[d],
                periodic=True,
                spatial_mode=mode,
                spatial_weight=0.3,
                normalization=30,
            )
            for d in (0, 1)
        )

    assert torch.autograd.gradcheck(loss, (pred,), eps=1e-6, atol=1e-5, rtol=1e-4)
    identity = target.clone().requires_grad_()
    value = loss(identity)
    assert value.item() == 0
    assert torch.count_nonzero(torch.autograd.grad(value, identity)[0]) == 0


def test_objective_cache_eval_and_all_field_gradients():
    cfg = config()
    cfg["persistence"] = {"distance": "spatial_wasserstein", "spatial_weight": 0.5}
    objective = PersistenceTopologyObjective(cfg, ("a", "b", "c"))
    target = torch.randn(2, 3, 7, 8, generator=torch.Generator().manual_seed(36))
    pred = (target + 0.12 * target.roll(2, -1)).requires_grad_()
    cache = {}
    first = objective(pred, target, reference_cache=cache)
    second = objective(pred, target, reference_cache=cache)
    torch.testing.assert_close(first.scalar_loss, second.scalar_loss, atol=0, rtol=0)
    mutual = first.component_results["topology.mutual.persistence"].scalar_loss
    assert (torch.autograd.grad(mutual, pred)[0].abs().sum((0, 2, 3)) > 0).all()
    with torch.no_grad():
        torch.testing.assert_close(
            objective(pred, target).scalar_loss, second.scalar_loss, atol=0, rtol=0
        )
    assert first.diagnostics["spatial_scope"] == "finite_creators_only"


def test_limits_and_invalid_settings_do_not_silently_change_metric():
    d = diagram([[0.0, 1.0], [0.0, 2.0]], [[0.0, 0.0], [0.5, 0.5]])
    with pytest.raises(ValueError, match="no bars are silently truncated"):
        spatial_diagram_distance(d, d, periodic=True, max_assignment_size=3)
    for settings in [
        {"spatial_weight": 1.0},
        {"distance": "oops"},
        {"distance": "spatial_wasserstein", "spatial_weight": -1.0},
        {"distance": "spatial_wasserstein", "spatial_mode": "oops"},
    ]:
        cfg = deepcopy(config())
        cfg["persistence"] = settings
        with pytest.raises(ValueError):
            validate_persistence_config(cfg, ("a", "b", "c"))
