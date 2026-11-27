"""Bounded geometry, scoring and frozen-observer contracts for Round 8."""

from copy import deepcopy
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

pytest.importorskip("gudhi")

from phycoflow_reconstruction.coherence.positional_shape import PositionalShapeFitter
from phycoflow_reconstruction.coherence.query_sampling import full_target_query_batch
from phycoflow_reconstruction.coherence.shape_compute import LazyPooledShape


# Keep the pre-Round-8 observer fixed after the new implementation is committed.
PRE_ROUND8_COMMIT = "f303d464aab2bd970a805de484a41c01a015f4da"


@pytest.mark.parametrize("member_count", [1, 2])
def test_disabled_ph_matches_saved_pre_round8_values_and_gradients(member_count):
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(["git", "show",
        f"{PRE_ROUND8_COMMIT}:src/phycoflow_reconstruction/coherence/shape_compute.py"], cwd=root)
    historical = types.ModuleType("phycoflow_reconstruction.coherence._saved_ph_shape")
    historical.__package__ = "phycoflow_reconstruction.coherence"
    exec(compile(source, "<saved PH Shape>", "exec"), historical.__dict__)
    current, _, batch, _, source, target = fitted_shape(fields=2, size=(4, 5))
    previous = historical.LazyPooledShape.from_state_dict(current.state_dict())
    members = torch.stack((source, source + .11)).detach().requires_grad_()
    second = None if member_count == 1 else members[1]
    mode = "paired" if member_count == 1 else "conditional"
    before = previous.score(members[0], second, target, batch, exhaustive=True, mode=mode)
    after = current.score(members[0], second, target, batch, exhaustive=True, mode=mode)
    for kind in ("self", "mutual"):
        torch.testing.assert_close(before[kind], after[kind], rtol=0, atol=0)
        before_grad = torch.autograd.grad(before[kind].sum(), members, retain_graph=True)[0]
        after_grad = torch.autograd.grad(after[kind].sum(), members, retain_graph=True)[0]
        torch.testing.assert_close(before_grad, after_grad, rtol=0, atol=0)


def fitted_shape(fields=1, *, mask=None, periodic=False, size=(6, 8), chunk=1):
    generator = torch.Generator().manual_seed(821)
    target = torch.randn(2, fields, *size, generator=generator, dtype=torch.double)
    source = target + torch.randn(target.shape, generator=generator, dtype=torch.double) * 0.3
    batch = full_target_query_batch(fields)
    shape = LazyPooledShape([0.] * fields, [1.] * fields, dimensions=(0,),
                            projections=4, periodic=periodic)
    shape.accumulate(source, target, batch, valid_mask=mask)
    shape.freeze(provenance={"split": "train", "sample_ids": ["a", "b"],
                             "source_checkpoint_sha256": "a" * 64})
    yy, xx = torch.meshgrid(torch.arange(size[0], dtype=torch.double),
                            torch.arange(size[1], dtype=torch.double), indexing="ij")
    coordinates = torch.stack((xx / size[1], yy / size[0]), -1)
    mass = torch.ones(size, dtype=torch.double)
    if mask is not None:
        mass[~mask] = 0
    queries = shape.compile_queries((batch,), exhaustive=True)
    fitter = PositionalShapeFitter(shape, coordinates, mass, valid_mask=mask,
        periods=(1., 1.) if periodic else (None, None), axis_extents=((0., 1.), (0., 1.)),
        threshold_chunk_size=chunk)
    fitter.observe_reference(target, queries)
    fitter.freeze_thresholds()
    fitter.observe_pair(source, target, queries)
    return shape, fitter.freeze(), batch, queries, source, target


def test_j1_j2_j3_fair_scores_components_and_no_extra_ph(monkeypatch):
    import phycoflow_reconstruction.coherence.shape_compute as module

    shape, position, batch, queries, source, target = fitted_shape(fields=2, size=(4, 5))
    third = (source * 0.7 + 0.17).requires_grad_()
    members = torch.stack((source, source + 0.13, third)).detach().requires_grad_()
    calls = []
    original = module.cubical_diagrams
    monkeypatch.setattr(module, "cubical_diagrams", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    old = shape.score(members[0], members[1], target, batch, additional_members=(members[2],),
                      exhaustive=True)
    assert len(calls) == 4
    shape.set_position(position)
    calls.clear()
    actual = shape.score(members[0], members[1], target, batch, additional_members=(members[2],),
                         exhaustive=True)
    assert len(calls) == 4  # one target and three members, just as PH only
    cache = shape.compile_reference(target, queries)
    pieces = shape.score_compiled(members, queries, cache, include_components=True)
    sampled, parts = shape.score(members[0], members[1], target, batch,
        additional_members=(members[2],), exhaustive=True, return_components=True)
    for kind in ("self", "mutual"):
        torch.testing.assert_close(actual[kind], pieces[kind])
        torch.testing.assert_close(sampled[kind], parts[f"{kind}.ph"] + parts[f"{kind}.position"])
        reconstructed = sum(pieces[f"{kind}.{component}.attraction"]
                            - pieces[f"{kind}.{component}.separation"]
                            for component in ("finite", "essential", "position"))
        torch.testing.assert_close(reconstructed, pieces[kind])
        assert torch.isfinite(old[kind]).all()
        pair_scores = []
        for left, right in ((0, 1), (0, 2), (1, 2)):
            pair_scores.append(shape.score(members[left], members[right], target, batch,
                                           exhaustive=True)[kind])
        expected = torch.stack(pair_scores).mean(0)
        torch.testing.assert_close(actual[kind], expected)
        gradient = torch.autograd.grad(actual[kind].sum(), members, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected.sum(), members, retain_graph=True)[0]
        torch.testing.assert_close(gradient, expected_grad, rtol=1e-6, atol=1e-7)
        assert torch.isfinite(gradient).all()
    for count in (1, 2, 3):
        compiled = shape.score_compiled(members[:count], queries, cache)
        direct = shape.score(members[0], None if count == 1 else members[1], target, batch,
                             additional_members=tuple(members[2:count]), exhaustive=True,
                             mode="paired" if count == 1 else "conditional")
        for kind in direct:
            torch.testing.assert_close(compiled[kind], direct[kind], rtol=1e-6, atol=1e-7)


def test_location_translation_chunking_mask_and_saturation():
    mask = torch.ones(16, 32, dtype=torch.bool)
    mask[4:8, 13:15] = False
    shape, position, _, queries, _, _ = fitted_shape(mask=mask, size=mask.shape)
    shape.set_position(position)
    yy, xx = torch.meshgrid(torch.arange(16, dtype=torch.double),
                            torch.arange(32, dtype=torch.double), indexing="ij")
    shift = torch.tensor(0.25, dtype=torch.double, requires_grad=True)
    def bowl(displacement):
        return (-torch.exp(-((xx - (8.0 + displacement)) ** 2 + (yy - 9.0) ** 2) / 8))[None, None]
    reference = bowl(0.)
    observed = bowl(shift)
    one = shape.position.features(shape._filtration_bank(observed, queries), queries)
    baseline = shape.position.features(shape._filtration_bank(reference, queries), queries)
    distance = shape.position.distance(one, baseline).mean()
    assert distance.item() > 0 and torch.autograd.grad(distance, shift)[0] > 0
    modified = reference.clone()
    modified[..., ~mask] = 1000
    torch.testing.assert_close(baseline, shape.position.features(shape._filtration_bank(modified, queries), queries))
    restored = deepcopy(position)
    restored["threshold_chunk_size"] = 3
    other = LazyPooledShape.from_state_dict(shape.state_dict())
    other.set_position(restored)
    shift2 = shift.detach().clone().requires_grad_()
    first = shape.position.features(shape._filtration_bank(bowl(shift2), queries), queries)
    second = other.position.features(other._filtration_bank(bowl(shift2), queries), queries)
    torch.testing.assert_close(first, second)
    for result in (first, second):
        assert torch.isfinite(torch.autograd.grad(result.sum(), shift2, retain_graph=True)[0])
    a = torch.autograd.grad(first.square().sum(), shift2, retain_graph=True)[0]
    b = torch.autograd.grad(second.square().sum(), shift2)[0]
    torch.testing.assert_close(a, b)
    identical = baseline.detach().clone().requires_grad_()
    zero = shape.position.distance(identical, baseline).sum()
    assert zero.item() == 0 and torch.count_nonzero(torch.autograd.grad(zero, identical)[0]) == 0
    saturated = torch.full_like(reference, 1000., requires_grad=True)
    saturation = shape.position.features(shape._filtration_bank(saturated, queries), queries)
    assert torch.isfinite(saturation).all() and torch.isfinite(torch.autograd.grad(saturation.sum(), saturated)[0]).all()


def test_periodic_window_seam_and_fixed_target_weights():
    shape, position, batch, queries, _, target = fitted_shape(periodic=True, size=(12, 32))
    shape.set_position(position)
    yy, xx = torch.meshgrid(torch.arange(12, dtype=torch.double) / 12,
                            torch.arange(32, dtype=torch.double) / 32, indexing="ij")
    def bump(center):
        dx = torch.remainder(xx - center + .5, 1.) - .5
        dy = torch.remainder(yy - .5 + .5, 1.) - .5
        return torch.exp(-(dx.square() + dy.square()) / .015)[None, None]
    vectors = [shape.position.features(shape._filtration_bank(bump(v), queries), queries)
               for v in (.015, .985, .5)]
    near = shape.position.distance(vectors[0], vectors[1]).mean()
    far = shape.position.distance(vectors[0], vectors[2]).mean()
    assert 0 < near < far
    fitter = PositionalShapeFitter(shape, position["coordinates"], position["raster_mass"],
                                   periods=(1., 1.), axis_extents=((0., 1.), (0., 1.)))
    altered = list(queries)
    altered[0] = (*altered[0][:-1], altered[0][-1] * 2)
    with pytest.raises(ValueError, match="target weights"):
        fitter.observe_reference(target, tuple(altered))
    assert not batch.pair_queries  # N=1 never fabricates pair observables


def test_position_artifact_fresh_process_reload(tmp_path):
    shape, position, batch, queries, source, target = fitted_shape(size=(4, 5))
    shape.set_position(position)
    expected = shape.score(source, None, target, batch, exhaustive=True, mode="paired")["self"]
    file = tmp_path / "artifact.pt"
    output = tmp_path / "result.pt"
    torch.save({"shape": shape.state_dict(), "source": source, "target": target}, file)
    code = """
import sys, torch
from phycoflow_reconstruction.coherence.query_sampling import full_target_query_batch
from phycoflow_reconstruction.coherence.shape_compute import LazyPooledShape
p=torch.load(sys.argv[1], map_location='cpu', weights_only=True)
s=LazyPooledShape.from_state_dict(p['shape'])
v=s.score(p['source'],None,p['target'],full_target_query_batch(1),exhaustive=True,mode='paired')['self']
torch.save(v,sys.argv[2])
"""
    subprocess.run([sys.executable, "-c", code, str(file), str(output)], check=True, env=os.environ.copy())
    torch.testing.assert_close(torch.load(output, weights_only=True), expected, rtol=0, atol=0)


def test_train_sensitivity_uses_fixed_mass_and_equal_members():
    from scripts.analysis.car_round8_position_sensitivity import METRICS, observe
    from phycoflow_reconstruction.coherence.positional_shape import _key

    shape, position, _, queries, source, target = fitted_shape(fields=2, size=(4, 5))
    mass = torch.arange(1, 21, dtype=torch.double).reshape(4, 5)
    fitter = PositionalShapeFitter(shape, position["coordinates"], mass)
    fitter.observe_reference(target, queries)
    fitter.freeze_thresholds()
    fitter.observe_pair(source, target, queries)
    shape.set_position(fitter.freeze())
    endpoints = torch.cat((target[:1], source[:1], source[:1] + .3), dim=0)
    observed = observe(shape, endpoints, queries)
    category = _key(("self", 1, None))
    indices = [index for index, query in enumerate(queries) if _key(query[0]) == category]
    filtration = shape._filtration_bank(endpoints, queries)
    tau = shape.position.thresholds[category][0]
    temp = shape.position.temperatures[category]
    normalizer = sum(queries[index][4] for index in indices)
    expected = sum(
        queries[index][4] * ((torch.sigmoid((tau - filtration[:, index]) / temp) * mass)
                             .sum((-1, -2)) / mass.sum())
        for index in indices
    ) / normalizer
    numerator, weight = observed[category]
    torch.testing.assert_close(numerator[:, 0, METRICS.index("occupancy")] / weight, expected)
    assert torch.isfinite(numerator).all()
    assert (numerator[:, :, METRICS.index("absolute_filtration_derivative")] >= 0).all()


def test_action_fixture_restores_saved_scale_contract():
    from types import SimpleNamespace
    from scripts.qualification.car_round8_j3_position_action import restore_scale_contract

    runtime = SimpleNamespace(bank={"fixed": True}, scale_objective="composite", scale_joint_kappa=1.)
    restore_scale_contract(runtime, {"scale_objective": "joint_anchor", "scale_joint_kappa": 1.25})
    assert runtime.scale_objective == "joint_anchor" and runtime.scale_joint_kappa == 1.25
    with pytest.raises(ValueError, match="Scale contract"):
        restore_scale_contract(SimpleNamespace(bank=None), {"scale_objective": "joint_anchor"})


def test_position_frozen_observer_export_reload_and_components():
    from test_frozen_coherence_observer import make_observer
    from phycoflow_reconstruction.coherence.query_runtime import _shape_grid
    from phycoflow_reconstruction.evaluation.frozen_coherence import FrozenCoherenceObserver
    from phycoflow_reconstruction.training.scalable_fitting import _digest

    original, members, families, artifact = make_observer("conditional")
    shape = LazyPooledShape.from_state_dict(artifact["shape_artifact"])
    family = families["topology"]
    reference = _shape_grid(family, original.reference, original.reference)
    source = _shape_grid(family, members[0], original.reference)
    queries = shape.compile_queries((full_target_query_batch(2),), exhaustive=True)
    fitter = PositionalShapeFitter(shape, family.grid_coordinates,
                                  torch.ones(reference.shape[-2:]))
    fitter.observe_reference(reference, queries)
    fitter.freeze_thresholds()
    fitter.observe_pair(source, reference, queries)
    shape.set_position(fitter.freeze())
    artifact = deepcopy(artifact)
    artifact["shape_artifact"] = shape.state_dict()
    artifact["sha256"] = _digest({key: value for key, value in artifact.items() if key != "sha256"})
    kwargs = dict(expected_provenance=artifact["provenance"], reference=original.reference,
                  coordinates=original.coordinates, sample_ids=original.receipt["sample_ids"],
                  member_weights=original.member_weights, reference_weights=original.reference_weights,
                  mode="conditional", query_seed=3, query_batches=2)
    observer = FrozenCoherenceObserver(artifact, families, **kwargs)
    assert observer.receipt["sha256"] != original.receipt["sha256"]
    cache = observer.export_compiled_target()
    restored = FrozenCoherenceObserver(artifact, families, compiled_target=cache, **kwargs)
    with torch.no_grad():
        values = observer.score_tensors(members, include_components=True)
        restored_values = restored.score_tensors(members, include_components=True)
    assert "shape.self.position.attraction" in values
    for key in values:
        torch.testing.assert_close(values[key], restored_values[key], rtol=0, atol=0)
    broken = deepcopy(cache)
    broken["position_features"].add_(1)
    with pytest.raises(ValueError, match="cache changed"):
        FrozenCoherenceObserver(artifact, families, compiled_target=broken, **kwargs)
    from phycoflow_reconstruction.coherence.query_sampling import ConditionQueryGroups, SharedQuerySampler
    from phycoflow_reconstruction.training.scalable_runtime import ScalableRuntime
    runtime = ScalableRuntime(artifact, families, steps_per_epoch=1,
                              expected_provenance=artifact["provenance"])
    runtime.capture_shape_components = True
    sampler = SharedQuerySampler(original.fields, seed=22, epsilon=1.)
    batches = []
    for _ in range(2):
        batches.append(sampler.draw())
        sampler.finish()
    context = {"generated_weights": original.member_weights,
               "reference_weights": original.reference_weights,
               "conditional_draws": (members[1],)}
    result, _, _ = runtime._score("topology", families["topology"], members[0], original.reference,
                                   original.coordinates, context, ConditionQueryGroups(tuple(batches)))
    parts = result.diagnostics["shape_component_tensors"]
    assert set(parts) == {"ph", "position"}
    torch.testing.assert_close(parts["ph"] + parts["position"], result.scalar_loss)


def test_grouped_j3_real_core_values_and_member_gradients():
    from test_frozen_coherence_observer import make_observer
    from phycoflow_reconstruction.coherence.query_sampling import ConditionQueryGroups, SharedQuerySampler
    from phycoflow_reconstruction.training.scalable_runtime import ScalableRuntime

    observer, draws, families, artifact = make_observer("conditional")
    runtime = ScalableRuntime(artifact, families, steps_per_epoch=1,
                              expected_provenance=artifact["provenance"])
    runtime.capture_shape_components = True
    sampler = SharedQuerySampler(observer.fields, seed=91, epsilon=1.)
    groups = []
    for _ in range(2):
        groups.append(sampler.draw())
        sampler.finish()
    groups = ConditionQueryGroups(tuple(groups))
    members = torch.cat((draws, draws[:1] * .75 + .2), dim=0).requires_grad_()
    context = {"generated_weights": observer.member_weights,
               "reference_weights": observer.reference_weights}
    for name in families:
        def evaluate(indices):
            chosen = members[list(indices)]
            result, _, _ = runtime._score(name, families[name], chosen[0], observer.reference,
                observer.coordinates, {**context, "conditional_draws": tuple(chosen[1:])}, groups)
            if name == "topology":
                torch.testing.assert_close(result.diagnostics["shape_component_tensors"]["ph"],
                                           result.scalar_loss)
            return result.per_sample_cost
        actual = evaluate((0, 1, 2))
        expected = torch.stack([evaluate(pair) for pair in ((0, 1), (0, 2), (1, 2))]).mean(0)
        torch.testing.assert_close(actual, expected, rtol=4e-6, atol=4e-7)
        gradient = torch.autograd.grad(actual.sum(), members, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected.sum(), members, retain_graph=True)[0]
        torch.testing.assert_close(gradient, expected_grad, rtol=3e-5, atol=3e-6)
        assert torch.isfinite(gradient).all()
        assert gradient[2].abs().sum() > 0


def test_j3_expanded_group_and_triple_scores_use_all_members(monkeypatch):
    from types import SimpleNamespace
    import phycoflow_reconstruction.training.scalable_runtime as module

    runtime = object.__new__(module.ScalableRuntime)
    runtime.scoring_mode = "conditional"
    runtime.bank = {"directions": [[1., -.4, .2]],
                    "triples": [[0, [[0, 0], [0, 1], [0, 2]]]],
                    "physical_masses": [0.25] * 4, "region_masses": [[0.25] * 4],
                    "triple_scales": [[[1., 1., 1.]]],
                    "target_weights": {"state.group_projection": [1.], "scale.third_order": [1.]}}
    class State:
        def _endpoint_measures(self, prediction, reference, context, required):
            return torch.ones(4), torch.ones(4)
    runtime.families = {"global_distribution": State(), "multiscale_spectrum": object()}
    monkeypatch.setattr(module, "spectral_filter", lambda family, values: values.unsqueeze(-3))
    generator = torch.Generator().manual_seed(99)
    members = torch.randn(3, 2, 4, 3, generator=generator, dtype=torch.double, requires_grad=True)
    reference = torch.randn(2, 4, 3, generator=generator, dtype=torch.double)
    for name in ("global_distribution", "multiscale_spectrum"):
        def evaluate(indices):
            chosen = members[list(indices)]
            return runtime._expanded(name, chosen[0], reference,
                                     {"conditional_draws": tuple(chosen[1:])})[1]
        actual = evaluate((0, 1, 2))
        expected = torch.stack([evaluate(pair) for pair in ((0, 1), (0, 2), (1, 2))]).mean(0)
        torch.testing.assert_close(actual, expected)
        gradient = torch.autograd.grad(actual.sum(), members, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected.sum(), members, retain_graph=True)[0]
        torch.testing.assert_close(gradient, expected_grad)
        assert gradient[2].abs().sum() > 0
        runtime.scoring_mode = "deterministic"
        paired = runtime._expanded(name, members[0], reference, {})[1]
        assert paired.shape == (2,) and torch.isfinite(paired).all()
        assert torch.autograd.grad(paired.sum(), members, retain_graph=True)[0][0].abs().sum() > 0
        runtime.scoring_mode = "conditional"


@pytest.mark.parametrize("field_count", [1, 3, 5, 7])
def test_field_permutation_preserves_frozen_shape_observation(field_count):
    from phycoflow_reconstruction.coherence.positional_shape import _key

    generator = torch.Generator().manual_seed(921 + field_count)
    target = torch.randn(1, field_count, 3, 4, dtype=torch.double, generator=generator)
    source = target + torch.randn(target.shape, dtype=torch.double, generator=generator) * .2
    mean = torch.linspace(-.3, .4, field_count).tolist()
    scales = torch.linspace(.8, 1.4, field_count).tolist()
    shape = LazyPooledShape(mean, scales, dimensions=(0,), projections=4)
    batch = full_target_query_batch(field_count)
    shape.accumulate(source, target, batch)
    names = [f"field_{index}" for index in range(field_count)]
    shape.freeze(provenance={"split": "train", "sample_ids": ["a"],
                 "field_order": names, "source_checkpoint_sha256": "a" * 64})
    yy, xx = torch.meshgrid(torch.arange(3), torch.arange(4), indexing="ij")
    coordinates = torch.stack((xx, yy), -1)
    queries = shape.compile_queries((batch,), exhaustive=True)
    fitter = PositionalShapeFitter(shape, coordinates, torch.ones(3, 4))
    fitter.observe_reference(target, queries)
    fitter.freeze_thresholds()
    fitter.observe_pair(source, target, queries)
    shape.set_position(fitter.freeze())
    state = shape.state_dict()
    changed = deepcopy(state)
    changed["field_mean"] = list(reversed(changed["field_mean"]))
    changed["field_scale"] = list(reversed(changed["field_scale"]))
    changed["provenance"]["field_order"] = list(reversed(names))

    def reversed_category(key):
        kind, sign, line = key
        if kind == "self":
            return key
        return (kind, tuple(reversed(sign)), 0 if line == 0 else 3 - line)

    changed["scales"] = [(reversed_category(key), value, count)
                         for key, value, count in state["scales"]]
    mapping = {_key(key): _key(reversed_category(key)) for key, _, _ in state["scales"]}
    for item in ("thresholds", "temperatures", "fit_counts"):
        changed["position"][item] = {mapping[key]: value
                                     for key, value in state["position"][item].items()}
    restored = LazyPooledShape.from_state_dict(changed)
    original = source.detach().clone().requires_grad_()
    permuted = source.flip(1).detach().clone().requires_grad_()
    observed = shape.score(original, None, target, batch, mode="paired", exhaustive=True)
    reordered = restored.score(permuted, None, target.flip(1), batch, mode="paired", exhaustive=True)
    for kind in ("self", "mutual"):
        torch.testing.assert_close(observed[kind], reordered[kind], rtol=1e-7, atol=1e-8)
    grad = torch.autograd.grad(sum(value.sum() for value in observed.values()), original)[0]
    permuted_grad = torch.autograd.grad(sum(value.sum() for value in reordered.values()), permuted)[0]
    torch.testing.assert_close(grad.flip(1), permuted_grad, rtol=1e-6, atol=1e-7)
    if field_count == 1:
        assert not batch.pair_queries and observed["mutual"].item() == 0
