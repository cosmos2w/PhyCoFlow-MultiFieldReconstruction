import itertools
from types import SimpleNamespace

import pytest
import torch

from phycoflow_reconstruction.coherence.empirical_teacher import (
    FAMILIES, empirical_family_scores, normalized_teacher_loss, teacher_step,
)


class DistanceRuntime:
    def __init__(self):
        self.batch = object()
        self.scoring_mode = "conditional"
        self.families = {name: SimpleNamespace(conditional_draws=2) for name in FAMILIES}

    def _score(self, name, family, prediction, reference, coordinates, context, batch):
        distance = lambda a, b: (a - b).square().sum((-1, -2))
        atoms = (prediction, *context["conditional_draws"])
        value = sum(distance(a, reference) for a in atoms) / len(atoms)
        value = value - sum(distance(atoms[i], atoms[j]) for i, j in itertools.combinations(range(len(atoms)), 2)) / len(atoms) ** 2
        return SimpleNamespace(per_sample_cost=value), {}, None


def test_v_score_all_atoms_values_and_gradients():
    torch.manual_seed(2)
    atoms = torch.randn(4, 2, 3, 2, dtype=torch.double, requires_grad=True)
    reference = torch.randn(2, 3, 2, dtype=torch.double)
    actual = empirical_family_scores(DistanceRuntime(), atoms, reference, lambda x: x,
                                     coordinates=None, context={})
    distance = lambda a, b: (a - b).square().sum((-1, -2))
    expected = sum(distance(a, reference) for a in atoms) / 4 - sum(
        distance(atoms[i], atoms[j]) for i, j in itertools.combinations(range(4), 2)
    ) / 16
    for value in actual.values():
        torch.testing.assert_close(value, expected)
    grad = torch.autograd.grad(actual[FAMILIES[0]].sum(), atoms, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), atoms)[0]
    torch.testing.assert_close(grad, expected_grad)
    assert (grad.abs().sum((1, 2, 3)) > 0).all()
    torch.testing.assert_close(normalized_teacher_loss(actual, dict.fromkeys(FAMILIES, 2)), expected / 2)


def test_source_anchored_feasibility_and_sensors():
    source = torch.zeros(4, 2, 3, 1, dtype=torch.double)
    reference = torch.ones(2, 3, 1, dtype=torch.double)
    observed = torch.zeros(2, 3, 1, dtype=torch.bool)
    observed[:, 0] = True
    masses = torch.tensor([1., 3., 0.])
    allowance = torch.tensor([0.01])
    loss = lambda y: -100 * y.sum((0, 2, 3))
    updated, info = teacher_step(source, source, reference, masses, allowance, observed, loss)
    assert torch.equal(updated[:, :, 0], source[:, :, 0])
    assert torch.equal(updated[:, :, 2], source[:, :, 2])
    assert (info["displacement"] <= allowance.double()).all()
    assert (info["mean_error"] <= 1 + allowance.double()).all()
    assert (info["member_error"] <= 1 + allowance.double()).all()
    again, second = teacher_step(source, updated, reference, masses, allowance, observed, loss)
    assert (second["alpha"] < 1).all()
    assert torch.isfinite(again).all()
    stationary, diag = teacher_step(source, source, reference, masses, allowance, observed,
                                    lambda y: (y * 0).sum((0, 2, 3)))
    assert torch.equal(stationary, source)
    assert (diag["alpha"] == 1).all()


def test_reject_bad_units_or_sensor_state():
    x = torch.zeros(4, 1, 2, 1)
    z = torch.zeros(1, 2, 1)
    mask = torch.tensor([[[True], [False]]])
    y = x.clone()
    y[:, :, 0] = 1
    loss = lambda a: a.sum((0, 2, 3))
    with pytest.raises(ValueError, match="observed"):
        teacher_step(x, y, z, torch.ones(2), torch.ones(1), mask, loss)
    with pytest.raises(ValueError, match="allowances"):
        teacher_step(x, x, z, torch.ones(2), torch.zeros(1), mask, loss)


def test_real_state_and_shape_empirical_match_pairwise_value_and_gradient():
    pytest.importorskip("gudhi")
    from phycoflow_reconstruction.coherence.query_sampling import Draw, QueryBatch
    from phycoflow_reconstruction.coherence.shape_compute import LazyPooledShape
    from phycoflow_reconstruction.coherence.state_compute import select_directions, score_family
    torch.manual_seed(42)
    atoms = torch.randn(4, 1, 16, 1, dtype=torch.double, requires_grad=True)
    ref = torch.randn(1, 16, 1, dtype=torch.double)
    weights = torch.ones(16, dtype=torch.double)
    bank = select_directions(torch.ones(1, 1), torch.ones(1), ["x"], 17)
    batch = QueryBatch((Draw(0, 1, 1.0, 1, 1),), (), 1)
    shape = LazyPooledShape([0.], [1.], dimensions=(0,), projections=3)
    raster = lambda a: a.reshape(1, 1, 4, 4)
    shape.accumulate(raster(atoms[0].detach()), raster(ref), batch)
    shape.freeze(provenance={"split": "train", "sample_ids": ["a"], "source_checkpoint_sha256": "a" * 64})
    for family in ("state", "shape"):
        def score(indices, empirical=False):
            a = atoms[list(indices)]
            if family == "state":
                return score_family(a, ref, weights, weights, bank, mode="paired" if len(a) == 1 else "conditional", empirical=empirical)[0].per_sample_cost
            return shape.score(raster(a[0]), None if len(a) == 1 else raster(a[1]), raster(ref), batch,
                               additional_members=tuple(raster(v) for v in a[2:]), empirical=empirical, exhaustive=True,
                               mode="paired" if len(a) == 1 else "conditional")["self"]
        direct = score(range(4), empirical=True)
        attraction = sum(score((i,)) for i in range(4)) / 4
        fair = sum(score(pair) for pair in itertools.combinations(range(4), 2)) / 6
        expected = .75 * fair + .25 * attraction
        torch.testing.assert_close(direct, expected, rtol=1e-6, atol=1e-7)
        grad = torch.autograd.grad(direct.sum(), atoms, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected.sum(), atoms, retain_graph=True)[0]
        torch.testing.assert_close(grad, expected_grad, rtol=1e-5, atol=1e-7)


def test_scale_empirical_all_blocks_match_pairwise_value_and_gradient():
    from phycoflow_reconstruction.coherence.query_sampling import Draw, QueryBatch
    from phycoflow_reconstruction.coherence.scale_compute import sampled_scale
    torch.manual_seed(91)
    atoms = torch.randn(4, 1, 4, 2, dtype=torch.double, requires_grad=True)
    ref = torch.randn(1, 4, 2, dtype=torch.double)
    batch = QueryBatch((Draw(0, 1, .5, 2, 1),), (Draw(0, 1, 1., 1, 1),), 1)
    def score(indices, empirical=False):
        values = atoms[list(indices)]
        return sampled_scale(values, ref, batch, filters=torch.ones(1, 4, dtype=torch.double),
                             scales=torch.ones(1, 2, dtype=torch.double), mass=torch.ones(4, dtype=torch.double),
                             power_floor=torch.ones(1, 2, dtype=torch.double), band_weights=torch.ones(1, dtype=torch.double),
                             self_descriptors=values.flatten(2)[:, :, :2], target_self_descriptor=ref.flatten(1)[:, :2],
                             mean_kwargs={"draw_piece_means": values.mean(2).unsqueeze(2), "target_piece_means": ref.mean(1).unsqueeze(1),
                                          "piece_weights": torch.ones(1, dtype=torch.double), "mean_scales": torch.ones(1, 2, dtype=torch.double)},
                             mode="empirical" if empirical else ("paired" if len(indices) == 1 else "conditional"))
    direct = score(range(4), True)
    for key in ("signed", "joint", "log_power_self", "rho"):
        attraction = sum(score((i,))[key] for i in range(4)) / 4
        fair = sum(score(pair)[key] for pair in itertools.combinations(range(4), 2)) / 6
        expected = .75 * fair + .25 * attraction
        torch.testing.assert_close(direct[key], expected)
        got = torch.autograd.grad(direct[key].sum(), atoms, retain_graph=True)[0]
        want = torch.autograd.grad(expected.sum(), atoms, retain_graph=True)[0]
        torch.testing.assert_close(got, want)


def test_expanded_empirical_leaves_match_pairwise_reduction():
    from phycoflow_reconstruction.coherence.state_compute import group_projection_scores
    from phycoflow_reconstruction.coherence.scale_compute import third_order_scores
    torch.manual_seed(512)
    atoms = torch.randn(4, 1, 5, 3, dtype=torch.double, requires_grad=True)
    ref = torch.randn(1, 5, 3, dtype=torch.double)
    mass = torch.ones(5, dtype=torch.double) / 5
    directions = [[1., -.5, .3]]
    triples = [[0, [[0, 0], [0, 1], [0, 2]]]]
    def score(indices, empirical=False, family="state"):
        values = atoms[list(indices)]
        if family == "state":
            return group_projection_scores(values, ref, mass, mass, directions, empirical=empirical)
        return third_order_scores(values.unsqueeze(2), ref.unsqueeze(1), triples,
                                  mass.unsqueeze(0), torch.ones(1, 1, 3), empirical=empirical)
    for family in ("state", "scale"):
        actual = score(range(4), True, family)
        paired = sum(score((i,), family=family) for i in range(4)) / 4
        fair = sum(score(pair, family=family) for pair in itertools.combinations(range(4), 2)) / 6
        expected = .75 * fair + .25 * paired
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), atoms, retain_graph=True)[0],
                                   torch.autograd.grad(expected.sum(), atoms, retain_graph=True)[0])


def test_observer_compatibility_is_limited_to_pinned_changes():
    from phycoflow_reconstruction.coherence.observer_compatibility import _PAIRS, compatible_pooled_source, compatible_scale_source
    for name, key, check in (("pooled", "source_sha256", compatible_pooled_source), ("scale", "sources", compatible_scale_source)):
        saved = {key: {k: hashes[0] for k, hashes in _PAIRS[name].items()}, 'backend': 'fixed'}
        current = {key: {k: hashes[1] for k, hashes in _PAIRS[name].items()}, 'backend': 'fixed'}
        assert check(saved, current)
        assert not check({**saved, 'backend': 'changed'}, current)
        assert not check(saved, {**current, key: {**current[key], next(iter(current[key])): 'changed'}})


def test_reference_transform_cache_reuses_condition_identity(monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path
    from phycoflow_reconstruction.coherence import query_runtime
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'scripts/posttraining'))
    spec = importlib.util.spec_from_file_location('teacher_driver_test', root/'scripts/posttraining/run_car_round4_teacher.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    def transform(_family, value):
        calls.append(value.shape[0])
        return value * 2, value.mean(1), value.sum(1)
    monkeypatch.setattr(query_runtime, '_scale_statistics', transform)
    data = torch.arange(12, dtype=torch.double).reshape(3, 2, 2)
    cache = {}
    first = module.cached_scale_reference(cache, [3, 5], data[:2], object())
    second = module.cached_scale_reference(cache, [5, 8], data[1:], object())
    assert calls == [2, 1]
    for left, right in zip(first, second):
        torch.testing.assert_close(left[1], right[0])
    assert set(cache) == {3, 5, 8}


def test_endpoint_replay_uses_exact_finite_mean_and_physical_units():
    import importlib.util
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'scripts/posttraining'))
    spec=importlib.util.spec_from_file_location('teacher_field_diagnostics_test',root/'scripts/posttraining/run_car_round4_teacher.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=torch.tensor([0.,0.,2.,2.],dtype=torch.double).view(4,1,1,1).expand(-1,1,2,2).clone()
    reference=torch.zeros(1,2,2,dtype=torch.double)
    mask=torch.tensor([[[True,False],[False,False]]])
    masses=torch.tensor([[1.,3.]],dtype=torch.double)
    candidate=source+1
    parts=module.endpoint_field_sums(source,candidate,reference,masses,mask)
    result=module.finish_endpoint_field_diagnostics(parts,1,['T','p'],torch.tensor([2.,3.]),['unknown','unknown'])
    model=result['model_field_units_squared'];physical=result['decoded_field_units_squared']
    assert model['finite_law_mean_mse']==[4.,4.]
    assert model['member_mse']==[5.,5.]
    assert model['member_spread']==[1.,1.]
    assert model['source_displacement']==[1.,1.]
    assert model['observed_entry_mse']==[5.,None]
    assert physical['finite_law_mean_mse']==[16.,36.]
    assert physical['observed_entry_mse']==[20.,None]
    assert result['observation_entry_counts']==[4,0]
