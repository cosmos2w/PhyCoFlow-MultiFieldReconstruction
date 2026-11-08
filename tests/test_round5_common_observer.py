import pytest
import torch

from phycoflow_reconstruction.evaluation.conditional_state import (
    CommonStateObserver, physical_relation_observer, query_accounting,
    squared_distance_accounting,
)
from test_state_weighted_measure import family


def test_j4_state_pair_matrix_parity_and_heldout_and_permutation():
    torch.manual_seed(17)
    f = family(fields=("a", "b"))
    members = torch.randn(4, 2, 8, 2, dtype=torch.float64)
    target = torch.randn(2, 8, 2, dtype=torch.float64)
    w = torch.tensor([1., 4., 2., 7., 1., 2., 1., 8.], dtype=torch.float64)
    mask = torch.tensor([True, True, True, False, True, True, True, True])
    observer = CommonStateObserver(f)
    observed = observer.score(members, target, weights=w, mask=mask)
    matrix = observed["main"]["squared_distances"]
    legacy = f(members[0], target, context={"generated_weights": w, "reference_weights": w,
        "generated_mask": mask, "reference_mask": mask})
    torch.testing.assert_close(matrix[0, 1], legacy.per_sample_cost)
    assert len(observed["main"]["leaf_squared_distances"]) == 2
    assert observer.heldout_directions.shape == (32, 2)
    assert not torch.equal(observer.heldout_directions, f.components_by_key["cross_joint_topk_swd"].directions)
    a = observed["main"]
    torch.testing.assert_close(a["m_member"], a["m_bank"] + a["v_bank"])
    torch.testing.assert_close(a["fair_score"], a["attraction"] - a["separation"])
    order = torch.tensor([3, 1, 0, 2])
    permuted = observer.score(members[order], target, weights=w, mask=mask)["main"]
    for name in ("m_member", "m_bank", "v_bank", "u_mean", "fair_score"):
        torch.testing.assert_close(a[name], permuted[name])
    one = CommonStateObserver(family(fields=("a",)))
    assert one.heldout_directions is None
    assert len(one.score(members[..., :1], target[..., :1], weights=w)["main"]["leaf_squared_distances"]) == 1


def test_hilbert_accounting_matches_direct_vectors_and_j2_flat_case():
    target = torch.tensor([[0., 1.], [1., -1.]], dtype=torch.float64)
    draws = torch.tensor([[[1., 2.], [0., 1.]], [[-1., 0.], [2., 3.]],
                          [[2., 4.], [-1., 2.]], [[1., -1.], [4., 2.]]], dtype=torch.float64)
    endpoints = torch.cat([target[None], draws])
    matrix = (endpoints[:, None] - endpoints[None, :]).square().sum(-1)
    out = squared_distance_accounting(matrix)
    mean = draws.mean(0)
    torch.testing.assert_close(out["m_bank"], (mean-target).square().sum(-1))
    torch.testing.assert_close(out["v_bank"], (draws-mean).square().sum(-1).mean(0))
    e = draws-target
    cross = sum((e[i]*e[k]).sum(-1) for i in range(4) for k in range(4) if i != k) / 12
    torch.testing.assert_close(out["u_mean"], cross)
    points = torch.tensor([0., -2., 3.])[:, None, None]
    matrix = (points[:, None]-points[None, :]).square().sum(-1)
    assert squared_distance_accounting(matrix)["fair_score"].item() == 0


def test_relation_queries_and_unclamped_negative_u():
    target = torch.tensor([[[[1., .5], [.5, 4.]]]], dtype=torch.float64)
    draws = torch.stack([target.clone(), target.clone(), target.clone(), target.clone()])
    draws[:, 0, 0, 0, 1] = torch.tensor([-3., -1., 1., 3.])
    draws[:, 0, 0, 1, 0] = draws[:, 0, 0, 0, 1]
    out = physical_relation_observer(draws, target, torch.tensor([[1., 2.]]), torch.tensor([[.1, .2]]))
    acc = out["mutual"]["accounting"]
    torch.testing.assert_close(acc["m_member"], acc["m_bank"] + acc["v_bank"])
    assert acc["u_mean"].item() < 0
    torch.testing.assert_close(acc["fair_score"], acc["attraction"] - acc["separation"])
    q = query_accounting(torch.tensor([[[1.]], [[-1.]]]), torch.tensor([[0.]]))
    assert q["fair_score"].item() == 0
    with pytest.raises(ValueError):
        squared_distance_accounting(torch.ones(3, 3, 1))


def test_scale_companion_matches_frozen_r_family():
    from test_multiscale_gram_extension import _fitted, _states
    from test_multiscale_relation import config
    from phycoflow_reconstruction.evaluation.conditional_state import scale_companion
    from phycoflow_reconstruction.coherence.families.multiscale_relation import ANCHOR_PATH, SELF_PATH, RHO_PATH
    states = _states()
    f, coords, _ = _fitted(config(), states)
    target = states[:1]
    members = torch.stack([states[i:i+1] for i in range(1, 5)])
    coordinates = coords[None]
    baseline = f(members[0], target, coordinates=coordinates, context={
        "conditional_draws": tuple(members[1:]), "score_all_conditional_draws": True})
    result = scale_companion(f, members, target, coordinates=coordinates,
        gram_scales=torch.ones_like(f.power_floor), gram_power_floor=f.power_floor)
    torch.testing.assert_close(result["signed_score"], baseline.component_results[ANCHOR_PATH].per_sample_cost)
    torch.testing.assert_close(result["self_proper"]["score"], baseline.component_results[SELF_PATH].per_sample_cost)
    torch.testing.assert_close(result["rho_proper"]["score"], baseline.component_results[RHO_PATH].per_sample_cost)
    assert set(result["retained"]) == {"gram", "coefficients", "piece_means"}
