import json

import pytest
import torch

from phycoflow_reconstruction.coherence.families.multiscale_spectrum.losses import (
    field_projections,
    signed_band_coefficient_energy_score,
    vector_energy_score,
)
from phycoflow_reconstruction.coherence.query_sampling import (
    Draw,
    QueryBatch,
    SharedQuerySampler,
    index_pair,
    pair_count,
    pair_index,
)
from phycoflow_reconstruction.coherence.scale_compute import sampled_scale
from phycoflow_reconstruction.coherence.shape_compute import LazyPooledShape


def full_batch(n):
    p = pair_count(n)
    return QueryBatch(
        tuple(Draw(i, 1, 1 / n, n, n) for i in range(n)),
        tuple(Draw(i, 1, 1 / p, p, p) for i in range(p)),
        1,
    )


def test_bijection_resume_and_positive_probabilities():
    for n in (1, 2, 5, 100, 100000):
        p = pair_count(n)
        for k in {0, p // 2, p - 1} if p else []:
            assert pair_index(n, *index_pair(n, k)) == k
    sampler = SharedQuerySampler(["a", "b", "c"], seed=2, capacity=1)
    b = sampler.draw()
    restored = SharedQuerySampler.from_state_dict(json.loads(json.dumps(sampler.state_dict())))
    assert restored.pending == b
    for obj in (sampler, restored):
        obj.finish(
            self_difficulties={d.index: (1, 2) for d in b.self_queries},
            pair_difficulties={d.index: (3, 4) for d in b.pair_queries},
        )
        assert obj.refresh(10)
    for _ in range(100):
        a, b = sampler.draw(), restored.draw()
        assert a == b
        assert all(d.weight <= 2 for d in (*a.self_queries, *a.pair_queries))
        sampler.finish()
        restored.finish()
    single = SharedQuerySampler(["a"]).draw()
    assert not single.pair_queries


def test_probability_weighted_query_mean_and_variance():
    sampler = SharedQuerySampler([str(i) for i in range(8)], seed=9, capacity=2)
    b = sampler.draw()
    sampler.finish(pair_difficulties={d.index: (d.index + 1, d.index + 1) for d in b.pair_queries})
    sampler.refresh(10)
    vals = torch.arange(28, dtype=torch.float64) - 10
    estimates = []
    for _ in range(12000):
        q = sampler.draw()
        estimates.append(sum(d.weight * float(vals[d.index]) for d in q.pair_queries))
        sampler.finish()
    x = torch.tensor(estimates, dtype=torch.float64)
    assert abs(x.mean().item() - vals.mean().item()) < 4 * x.std().item() / len(x) ** 0.5
    assert x.var().item() > 0


def test_sampled_scale_exhaustive_value_and_gradient():
    torch.manual_seed(4)
    c, k, s, batch_size = 4, 5, 3, 2
    draws = torch.randn(2, batch_size, k, c, dtype=torch.double, requires_grad=True)
    target = torch.randn(batch_size, k, c, dtype=torch.double)
    f = torch.rand(s, k, dtype=torch.double)
    scale = torch.rand(s, c, dtype=torch.double) + 0.5
    mass = torch.ones(6, dtype=torch.double)
    bw = torch.tensor([0.2, 0.3, 0.5], dtype=torch.double)
    desc = torch.randn(2, batch_size, s * c, dtype=torch.double)
    td = torch.randn(batch_size, s * c, dtype=torch.double)
    means = torch.randn(2, batch_size, 1, c, dtype=torch.double)
    tm = torch.randn(batch_size, 1, c, dtype=torch.double)
    kwargs = {
        "draw_piece_means": means,
        "target_piece_means": tm,
        "piece_weights": torch.ones(1, dtype=torch.double),
        "mean_scales": torch.ones(1, c, dtype=torch.double),
        "mean_weight": 1.0,
    }
    result = sampled_scale(
        draws,
        target,
        full_batch(c),
        filters=f,
        scales=scale,
        mass=mass,
        power_floor=torch.ones(s, c, dtype=torch.double),
        band_weights=bw,
        self_descriptors=desc,
        target_self_descriptor=td,
        mean_kwargs=kwargs,
    )
    projections = field_projections(c, "axes_and_pairs", dtype=draws.dtype, device=draws.device)
    old = (
        signed_band_coefficient_energy_score(
            draws, target, filters=f, scales=scale, mass=mass, projections=projections
        )[0]
        * bw
    ).sum(-1)
    assert torch.allclose(result["signed"], old, atol=1e-12)
    assert torch.allclose(
        torch.autograd.grad(result["signed"].sum(), draws, retain_graph=True)[0],
        torch.autograd.grad(old.sum(), draws)[0],
        atol=1e-12,
    )
    assert torch.allclose(result["log_power_self"], vector_energy_score(desc, td)[0])
    assert result["rho"].shape == result["joint"].shape == (batch_size,)


def test_lazy_shape_calibration_roundtrip_and_gradients():
    torch.manual_seed(2)
    x = torch.randn(2, 2, 4, 4, dtype=torch.double, requires_grad=True)
    y = torch.randn_like(x)
    z = torch.randn_like(x, requires_grad=True)
    shape = LazyPooledShape([0, 0], [1, 1], dimensions=(0,), projections=4)
    q = full_batch(2)
    shape.accumulate(x.detach(), y, q)
    state = shape.freeze(
        provenance={
            "split": "train",
            "sample_ids": ["a", "b"],
            "source_checkpoint_sha256": "a" * 64,
        }
    )
    clone = LazyPooledShape.from_state_dict(json.loads(json.dumps(state)))
    first, second = (
        shape.score(x, z, y, q, exhaustive=True),
        clone.score(x, z, y, q, exhaustive=True),
    )
    assert all(torch.allclose(first[k], second[k]) for k in first)
    grad = torch.autograd.grad(sum(v.sum() for v in first.values()), (x, z))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grad)
    with pytest.raises(ValueError):
        shape.score(x, z, y, q, valid_mask=torch.ones(4, 4, dtype=torch.bool), exhaustive=True)


def test_fitted_relation_adapter_complete_anchor_and_gradients():
    from scalable_test_helpers import _fitted, _score, _states, config

    from phycoflow_reconstruction.coherence.families.multiscale_relation import RHO_PATH, SELF_PATH
    from phycoflow_reconstruction.coherence.families.multiscale_spectrum.objectives import (
        MEAN_JOINT_ENERGY_PATH,
        SIGNED_BAND_PATH,
    )
    from phycoflow_reconstruction.coherence.query_runtime import score

    states = _states()
    family, coords, _ = _fitted(config(), states)
    x, y = (states[i : i + 2].clone().requires_grad_() for i in (0, 2))
    target = states[4:6]
    old = _score(family, coords, x, y, target)
    result, _blocks, _ = score(
        "scale",
        family,
        x,
        target,
        coords[None].expand(len(x), -1, -1),
        {"conditional_draws": (y,)},
        full_batch(2),
        {},
    )
    for key, path in [
        ("signed", f"multiscale_spectrum.{SIGNED_BAND_PATH}"),
        ("joint", f"multiscale_spectrum.{MEAN_JOINT_ENERGY_PATH}"),
        ("log_power_self", SELF_PATH),
        ("rho", RHO_PATH),
    ]:
        new = result.component_results["multiscale_spectrum." + key].per_sample_cost
        existing = old.component_results[path].per_sample_cost
        torch.testing.assert_close(new, existing, rtol=1e-8, atol=1e-9)
        ga = torch.autograd.grad(new.sum(), (x, y), retain_graph=True)
        gb = torch.autograd.grad(existing.sum(), (x, y), retain_graph=True)
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b, rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(result.scalar_loss, old.scalar_loss, rtol=1e-8, atol=1e-9)
    changed, _, priority = score(
        "scale",
        family,
        x,
        target,
        coords[None].expand(len(x), -1, -1),
        {"conditional_draws": (y,)},
        full_batch(2),
        {"scale.signed": 2.0, "_difficulty_units": {"scale.signed": 4.0, "scale.rho": 2.0}},
    )
    torch.testing.assert_close(
        changed.scalar_loss,
        result.scalar_loss + result.component_results["multiscale_spectrum.signed"].scalar_loss,
    )
    _, _, original = score(
        "scale",
        family,
        x,
        target,
        coords[None].expand(len(x), -1, -1),
        {"conditional_draws": (y,)},
        full_batch(2),
        {},
    )
    for i, value in original["self"].items():
        torch.testing.assert_close(priority["self"][i], value / 4)


def test_real_sampled_scale_and_shape_importance_value_gradient_and_variance():
    """Evaluate each real leaf once, then audit the exact finite sampling law."""
    import random
    import statistics

    torch.manual_seed(923)
    c = 3
    n_pairs = pair_count(c)
    batch_size = 1
    k = 4
    bands = 2
    fields = torch.randn(2, batch_size, c, 4, 4, dtype=torch.double, requires_grad=True)
    truth = torch.randn(batch_size, c, 4, 4, dtype=torch.double)
    shape = LazyPooledShape([0] * c, [1] * c, dimensions=(0,), projections=4)
    shape.accumulate(fields[0].detach(), truth, full_batch(c))
    shape.freeze(
        provenance={"split": "train", "sample_ids": ["case"], "source_checkpoint_sha256": "a" * 64}
    )
    spectral = torch.randn(2, batch_size, k, c, dtype=torch.double, requires_grad=True)
    target = torch.randn(batch_size, k, c, dtype=torch.double)
    f = torch.rand(bands, k, dtype=torch.double)
    scales = torch.ones(bands, c, dtype=torch.double)
    desc = torch.randn(2, batch_size, bands * c, dtype=torch.double)
    td = torch.randn(batch_size, bands * c, dtype=torch.double)
    kw = {
        "draw_piece_means": torch.zeros(2, batch_size, 1, c, dtype=torch.double),
        "target_piece_means": torch.zeros(batch_size, 1, c, dtype=torch.double),
        "piece_weights": torch.ones(1, dtype=torch.double),
        "mean_scales": torch.ones(1, c, dtype=torch.double),
    }

    def evaluate(q):
        sc = sampled_scale(
            spectral,
            target,
            q,
            filters=f,
            scales=scales,
            mass=torch.ones(k, dtype=torch.double),
            power_floor=scales,
            band_weights=torch.ones(bands, dtype=torch.double) / bands,
            self_descriptors=desc,
            target_self_descriptor=td,
            mean_kwargs=kw,
        )
        sh = shape.score(fields[0], fields[1], truth, q, exhaustive=True)
        return sc, sh

    exhaustive_scale, exhaustive_shape = evaluate(full_batch(c))
    # The Scale API requires a self draw and a pair draw together; isolate the
    # per-query leaves and Shape family components exposed by the same batch.
    leaves = {"scale_self": [], "scale_pair": [], "shape_self": [], "shape_pair": []}
    for i in range(c):
        q = QueryBatch((Draw(i, 1, 1 / c, c, 1),), (Draw(0, 1, 1 / n_pairs, n_pairs, 1),), i)
        sc, sh = evaluate(q)
        for key, value, tensor in [
            ("scale_self", sc["leaves"]["self"][i], spectral),
            ("shape_self", sh["self"], fields),
        ]:
            grad = torch.autograd.grad(value.sum(), tensor, retain_graph=True)[0]
            leaves[key].append((value.item(), grad.detach().clone()))
    for i in range(n_pairs):
        q = QueryBatch((Draw(0, 1, 1 / c, c, 1),), (Draw(i, 1, 1 / n_pairs, n_pairs, 1),), i)
        sc, sh = evaluate(q)
        for key, value, tensor in [
            ("scale_pair", sc["leaves"]["pair_signed"][i], spectral),
            ("shape_pair", sh["mutual"], fields),
        ]:
            grad = torch.autograd.grad(value.sum(), tensor, retain_graph=True)[0]
            leaves[key].append((value.item(), grad.detach().clone()))
    targets = {
        "scale_self": (
            exhaustive_scale["signed"] * 0 + sum(v for v, _ in leaves["scale_self"]) / c
        ),
        "scale_pair": (
            exhaustive_scale["signed"] * 0 + sum(v for v, _ in leaves["scale_pair"]) / n_pairs
        ),
        "shape_self": exhaustive_shape["self"],
        "shape_pair": exhaustive_shape["mutual"],
    }
    for key, rows in leaves.items():
        n = len(rows)
        probabilities = [0.5 / n + 0.5 * (i + 1) / sum(range(1, n + 1)) for i in range(n)]
        mean = sum(v for v, _ in rows) / n
        expect = sum(p * v / (n * p) for p, (v, _) in zip(probabilities, rows))
        torch.testing.assert_close(
            torch.tensor(expect, dtype=torch.double), targets[key].detach().reshape(())
        )
        exact_grad = sum(g for _, g in rows) / n
        weighted_grad = sum(p * g / (n * p) for p, (_, g) in zip(probabilities, rows))
        torch.testing.assert_close(weighted_grad, exact_grad, atol=1e-12, rtol=1e-12)
        if key.startswith("shape_"):
            actual = torch.autograd.grad(targets[key].sum(), fields, retain_graph=True)[0]
            torch.testing.assert_close(actual, exact_grad, atol=1e-10, rtol=1e-9)
        variance = sum(p * (v / (n * p) - mean) ** 2 for p, (v, _) in zip(probabilities, rows))
        rng = random.Random(18)
        samples = rng.choices(range(n), weights=probabilities, k=10000)
        observed_samples = [rows[i][0] / (n * probabilities[i]) for i in samples]
        observed = sum(observed_samples) / len(samples)
        assert abs(observed - mean) <= 5 * (variance / len(samples)) ** 0.5 + 1e-12
        assert statistics.variance(observed_samples) == pytest.approx(variance, rel=0.12, abs=1e-12)
        gradient_samples = torch.stack([rows[i][1] / (n * probabilities[i]) for i in samples])
        se = gradient_samples.std(dim=0, unbiased=True) / len(samples) ** 0.5
        assert torch.all((gradient_samples.mean(dim=0) - exact_grad).abs() <= 5 * se + 1e-10)
    expected_signed = (sum(v for v, _ in leaves["scale_self"]) / c) / c + (1 - 1 / c) * (
        sum(v for v, _ in leaves["scale_pair"]) / n_pairs
    )
    assert exhaustive_scale["signed"].item() == pytest.approx(expected_signed)
    exact_signed_grad = (sum(g for _, g in leaves["scale_self"]) / c) / c + (1 - 1 / c) * (
        sum(g for _, g in leaves["scale_pair"]) / n_pairs
    )
    torch.testing.assert_close(
        torch.autograd.grad(exhaustive_scale["signed"].sum(), spectral)[0],
        exact_signed_grad,
        atol=1e-12,
        rtol=1e-10,
    )


def test_batched_shape_matches_original_loop_values_gradients_and_duplicate_weights(monkeypatch):
    from phycoflow_reconstruction.coherence import shape_compute as module

    torch.manual_seed(125)
    c = 4
    p = pair_count(c)
    first = torch.randn(2, c, 4, 4, dtype=torch.double, requires_grad=True)
    second = torch.randn_like(first, requires_grad=True)
    target = torch.randn_like(first)
    mask = torch.ones(4, 4, dtype=torch.bool)
    mask[1, 1] = False
    calibration = QueryBatch(
        (Draw(1, 1, 1 / c, c, 1),), (Draw(0, 2, 1 / p, p, 3), Draw(4, 1, 1 / p, p, 3)), 1
    )
    obj = LazyPooledShape(
        [0.1, -0.2, 0.3, -0.4], [0.7, 1.1, 1.4, 0.8], dimensions=(0, 1), projections=5
    )
    # Independent reference calibration loops from the pre-batched implementation.
    norm = obj._domain(first, mask)
    legacy_sums = {}
    legacy_counts = {}
    for key, ids, signs, line, _ in obj._queries(calibration):
        a = obj._diagram(first.detach(), ids, signs, line, mask)
        t = obj._diagram(target, ids, signs, line, mask)
        raw = obj._distance(a, t, norm)
        idx = ids[0] if key[0] == "self" else pair_index(c, *ids)
        draws = calibration.self_queries if key[0] == "self" else calibration.pair_queries
        mult = next(d.count for d in draws if d.index == idx)
        legacy_sums[key] = legacy_sums.get(key, torch.zeros_like(raw[0])) + mult * raw.sum(0)
        legacy_counts[key] = legacy_counts.get(key, 0) + mult * len(first)
    calls = []
    original = module.cubical_diagrams

    def recording(*args, **kwargs):
        calls.append(args[0].shape[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "cubical_diagrams", recording)
    obj.accumulate(first.detach(), target, calibration, valid_mask=mask)
    assert len(calls) == 2
    for key, value in legacy_sums.items():
        torch.testing.assert_close(obj.sums[key], value, rtol=1e-12, atol=1e-12)
    assert obj.counts == legacy_counts
    obj.freeze(
        provenance={
            "split": "train",
            "sample_ids": ["a", "b"],
            "source_checkpoint_sha256": "a" * 64,
        }
    )
    # Category-specific TRAIN scales ensure batching does not accidentally share normalization.
    for i, key in enumerate(obj.scales):
        obj.scales[key] = obj.scales[key] * (1 + i / 7)
    selected = QueryBatch(
        (Draw(2, 1, 0.4, c, 1),), (Draw(0, 2, 0.2, p, 3), Draw(4, 1, 0.1, p, 3)), 2
    )
    legacy = {"self": first.new_zeros(len(first)), "mutual": first.new_zeros(len(first))}
    old_leaves = {"self": {}, "pair": {}}
    for key, ids, signs, line, weight in obj._queries(selected):
        a = obj._diagram(first, ids, signs, line, mask)
        b = obj._diagram(second, ids, signs, line, mask)
        t = obj._diagram(target, ids, signs, line, mask)
        raw = (
            obj._distance(a, t, norm) + obj._distance(b, t, norm) - obj._distance(a, b, norm)
        ) / 2
        scaled = raw / obj.scales[key].to(raw)
        value = (0.9 * scaled[..., 0] + 0.1 * scaled[..., 1]).mean(-1)
        legacy[key[0]] = legacy[key[0]] + weight * value
        kind = "self" if key[0] == "self" else "pair"
        idx = ids[0] if kind == "self" else pair_index(c, *ids)
        draw = next(
            d
            for d in (selected.self_queries if kind == "self" else selected.pair_queries)
            if d.index == idx
        )
        old_leaves[kind][idx] = old_leaves[kind].get(idx, 0) + (weight / draw.weight) * value
    calls.clear()
    actual, leaves = obj.score(
        first, second, target, selected, valid_mask=mask, return_leaves=True, exhaustive=True
    )
    assert len(calls) == 3
    assert all(n == len(tuple(obj._queries(selected))) * len(first) for n in calls)
    for key, value in legacy.items():
        torch.testing.assert_close(actual[key], value, rtol=1e-11, atol=1e-11)
    for kind in leaves:
        for idx in leaves[kind]:
            torch.testing.assert_close(
                leaves[kind][idx], old_leaves[kind][idx], rtol=1e-11, atol=1e-11
            )
    left = torch.autograd.grad(
        sum(v.sum() for v in legacy.values()), (first, second), retain_graph=True
    )
    right = torch.autograd.grad(sum(v.sum() for v in actual.values()), (first, second))
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


def test_inner_rng_resume_exposure_and_unchanged_outer_stream():
    from phycoflow_reconstruction.coherence.query_sampling import (
        LINE_WEIGHTS,
        SHAPE_SAMPLING_VERSION,
        VERSION,
    )

    sampler = SharedQuerySampler([str(i) for i in range(5)], seed=812)
    oracle = SharedQuerySampler([str(i) for i in range(5)], seed=812)
    for _ in range(50):
        expected_self = oracle._draw("self", 5, 1)
        expected_pair = oracle._draw("pair", 10, 3)
        batch = sampler.draw()
        for got, old in zip(
            (*batch.self_queries, *batch.pair_queries), (*expected_self, *expected_pair)
        ):
            assert (got.index, got.count, got.probability, got.population, got.draws) == (
                old.index,
                old.count,
                old.probability,
                old.population,
                old.draws,
            )
        saved = json.loads(json.dumps(sampler.state_dict()))
        resumed = SharedQuerySampler.from_state_dict(saved)
        assert resumed.pending == batch
        records = []
        sampler.finish(exposure_sink=records.append)
        resumed.finish()
        assert sampler.state_dict() == resumed.state_dict()
        for record in records:
            assert (
                record["query_protocol"] == VERSION
                and record["shape_sampling"] == SHAPE_SAMPLING_VERSION
            )
            if record["kind"] == "self":
                assert (
                    record["inner_sign"] in [[1], [-1]]
                    and record["inner_line"] is None
                    and record["inner_probability"] == 0.5
                )
            else:
                assert record["inner_probability"] == LINE_WEIGHTS[record["inner_line"]] / 4
    old = dict(sampler.state_dict())
    old["version"] = "round8_implicit_queries_v1"
    with pytest.raises(ValueError, match="incompatible sampler version"):
        SharedQuerySampler.from_state_dict(old)


def test_inner_sampling_exact_enumerated_value_gradient_and_duplicate_variance():
    import random
    import statistics

    from phycoflow_reconstruction.coherence.query_sampling import LINE_WEIGHTS, PAIR_SIGNS

    torch.manual_seed(731)
    c = 4
    p = pair_count(c)
    first = torch.randn(1, c, 4, 4, dtype=torch.double, requires_grad=True)
    second = torch.randn_like(first, requires_grad=True)
    target = torch.randn_like(first)
    shape = LazyPooledShape([0] * c, [1] * c, dimensions=(0,), projections=4)
    calibration = QueryBatch(
        (Draw(0, 1, 1 / c, c, 1),), (Draw(0, 2, 1 / p, p, 3), Draw(4, 1, 1 / p, p, 3)), 1
    )
    shape.accumulate(first.detach(), target, calibration)
    shape.freeze(
        provenance={"split": "train", "sample_ids": ["one"], "source_checkpoint_sha256": "a" * 64}
    )
    for i, key in enumerate(shape.scales):
        shape.scales[key] *= 1 + i / 11
    outer_self = Draw(2, 1, 0.4, c, 1)
    outer_pairs = (Draw(0, 2, 0.2, p, 3), Draw(4, 1, 0.1, p, 3))
    full = shape.score(
        first, second, target, QueryBatch((outer_self,), outer_pairs, 5), exhaustive=True
    )

    def inner(draw, sign, line=None):
        from dataclasses import replace

        return replace(
            draw,
            inner_sign=sign,
            inner_line=line,
            inner_probability=0.5 if line is None else LINE_WEIGHTS[line] / 4,
        )

    # The single chosen leaf already includes the outer multiplicity weight.
    self_terms = []
    for sign in ((1,), (-1,)):
        q = QueryBatch((inner(outer_self, sign),), (), 5)
        self_terms.append(shape.score(first, second, target, q)["self"])
    expected_self = sum(self_terms) / 2
    pair_terms = []
    weighted_pairs = []
    for outer in outer_pairs:
        samples = []
        for sign in PAIR_SIGNS:
            for line in range(3):
                q = QueryBatch((inner(outer_self, (1,)),), (inner(outer, sign, line),), 5)
                samples.append(
                    (LINE_WEIGHTS[line] / 4, shape.score(first, second, target, q)["mutual"])
                )
        pair_terms.append(samples)
        weighted_pairs.append(sum(prob * value for prob, value in samples))
    torch.testing.assert_close(expected_self, full["self"], atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(sum(weighted_pairs), full["mutual"], atol=1e-11, rtol=1e-11)
    expected = expected_self + sum(weighted_pairs)
    baseline = full["self"] + full["mutual"]
    g1 = torch.autograd.grad(expected.sum(), (first, second), retain_graph=True)
    g2 = torch.autograd.grad(baseline.sum(), (first, second))
    for a, b in zip(g1, g2):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)
    # With a duplicated pair ID, sharing one inner draw doubles the variance
    # relative to two independent draws of the same inner leaf.
    probs = [p for p, _ in pair_terms[0]]
    values = [v.item() / 2 for _, v in pair_terms[0]]
    mean = sum(p * v for p, v in zip(probs, values))
    base_var = sum(p * (v - mean) ** 2 for p, v in zip(probs, values))
    assert base_var > 0
    rng = random.Random(87)
    first_ids = rng.choices(range(len(probs)), weights=probs, k=20000)
    second_ids = rng.choices(range(len(probs)), weights=probs, k=20000)
    shared = statistics.variance([2 * values[i] for i in first_ids])
    independent = statistics.variance(
        [values[i] + values[j] for i, j in zip(first_ids, second_ids)]
    )
    assert shared == pytest.approx(4 * base_var, rel=0.08)
    assert independent == pytest.approx(2 * base_var, rel=0.08)


def test_inner_sampled_shape_keeps_signed_fair_contribution(monkeypatch):
    # Synthetic distances isolate the signed repulsion term; the score must not
    # clip a negative realized contribution before outer importance weighting.
    shape = LazyPooledShape([0], [1], dimensions=(0,), projections=3)
    x = torch.zeros(1, 1, 2, 2)
    sampler = SharedQuerySampler(["one"], seed=1)
    batch = sampler.draw()
    shape.accumulate(x, x, batch)
    shape.freeze(
        provenance={"split": "train", "sample_ids": ["a"], "source_checkpoint_sha256": "a" * 64}
    )
    monkeypatch.setattr(shape, "_diagram_bank", lambda *args: [None])
    distances = iter([torch.zeros(1, 1, 2), torch.zeros(1, 1, 2), torch.ones(1, 1, 2)])
    monkeypatch.setattr(shape, "_distance", lambda *args: next(distances))
    result = shape.score(x, x, x, batch)
    assert result["self"].item() < 0


def test_paired_scale_matches_conditional_degenerate_point_values_and_gradients():
    """One real endpoint gives the point limit for all four retained blocks."""
    torch.manual_seed(47)
    c, k, bands = 3, 5, 2
    prediction = torch.randn(1, 2, k, c, dtype=torch.double, requires_grad=True)
    target = torch.randn(2, k, c, dtype=torch.double)
    filters = torch.rand(bands, k, dtype=torch.double)
    means = torch.randn(1, 2, 2, c, dtype=torch.double, requires_grad=True)
    truth_means = torch.randn(2, 2, c, dtype=torch.double)
    descriptors = torch.randn(1, 2, bands * c, dtype=torch.double, requires_grad=True)
    target_desc = torch.randn(2, bands * c, dtype=torch.double)

    def evaluate(draws, pm, desc, mode):
        return sampled_scale(
            draws,
            target,
            full_batch(c),
            filters=filters,
            scales=torch.ones(bands, c, dtype=torch.double),
            mass=torch.ones(k, dtype=torch.double),
            power_floor=torch.ones(bands, c, dtype=torch.double),
            band_weights=torch.ones(bands, dtype=torch.double) / bands,
            self_descriptors=desc,
            target_self_descriptor=target_desc,
            mean_kwargs={
                "draw_piece_means": pm,
                "target_piece_means": truth_means,
                "piece_weights": torch.ones(2, dtype=torch.double),
                "mean_scales": torch.ones(2, c, dtype=torch.double),
            },
            mode=mode,
        )

    paired = evaluate(prediction, means, descriptors, "paired")
    fair = evaluate(
        prediction.expand(2, -1, -1, -1),
        means.expand(2, -1, -1, -1),
        descriptors.expand(2, -1, -1),
        "conditional",
    )
    assert paired["version"] == "paired_sampled_scale_v1"
    for name in ("signed", "rho", "joint", "log_power_self"):
        torch.testing.assert_close(paired[name], fair[name], rtol=1e-10, atol=1e-10)
        a = torch.autograd.grad(
            paired[name].sum(),
            (prediction, means, descriptors),
            retain_graph=True,
            allow_unused=True,
        )
        b = torch.autograd.grad(
            fair[name].sum(), (prediction, means, descriptors), retain_graph=True, allow_unused=True
        )
        for x, y in zip(a, b):
            if x is None:
                assert y is None
            else:
                torch.testing.assert_close(x, y, rtol=1e-9, atol=1e-10)
