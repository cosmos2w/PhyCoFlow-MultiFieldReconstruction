import copy

import pytest
import torch

from phycoflow_reconstruction.coherence.adaptive_benchmark import benchmark
from phycoflow_reconstruction.coherence.adaptive_fitting import (
    fit_train_bank,
    score_bank,
    validate_bank,
)


def test_disk_backed_parameter_gram_matches_full_gradients_and_cleans_up(tmp_path):
    from phycoflow_reconstruction.coherence.adaptive_profile import _parameter_gradient_gram

    matrix = torch.randn(4, 29, generator=torch.Generator().manual_seed(42), dtype=torch.float64)
    target = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    gram, count = _parameter_gradient_gram(
        lambda coeff: coeff @ matrix, target, spill_directory=tmp_path, chunk_size=7
    )
    torch.testing.assert_close(gram, matrix @ matrix.T, rtol=1e-12, atol=1e-12)
    assert count == 29 and list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="linear"):
        _parameter_gradient_gram(
            lambda coeff: (coeff @ matrix) + coeff.square().sum() - coeff.sum(),
            target,
            spill_directory=tmp_path,
            chunk_size=7,
        )
    assert list(tmp_path.iterdir()) == []


def test_streamed_train_fit_frozen_bank_and_runtime_gradients():
    generator = torch.Generator().manual_seed(32)
    truths = torch.randn(4, 8, 3, generator=generator, dtype=torch.float64)
    members = torch.randn(2, 4, 8, 3, generator=generator, dtype=torch.float64)

    def panel():
        for k in (0, 2):
            yield {
                "ids": [k, k + 1],
                "members": members[:, k : k + 2],
                "reference": truths[k : k + 2],
            }

    def filt(x):
        return x.unsqueeze(-3)

    def anchor(x, y, m):
        return ((x - y).square() * m[None, None, :, None]).sum()

    mass = torch.arange(1, 9, dtype=torch.float64)
    bank = fit_train_bank(
        panel,
        field_order=["a", "b", "c"],
        masses=mass,
        region_masses=torch.stack([mass, mass.flip(0)]),
        filter_fn=filt,
        anchor_fn=anchor,
        source_identity="source-hash",
        support_identity="support-hash",
        units="model",
        seed=4,
    )
    validate_bank(
        bank,
        field_order=["a", "b", "c"],
        source_identity="source-hash",
        support_identity="support-hash",
    )
    assert bank["population_ids"] == [0, 1, 2, 3]
    assert all(x > 0 for row in bank["triple_scales"] for band in row for x in band)
    assert bank["selected"]
    audit = bank["selection_audit"]
    assert len(audit) == bank["candidate_generation"]["candidate_count"]
    assert {row["id"] for row in audit if row["decision"] == "selected"} == {
        q["id"] for q in bank["selected"]
    }
    assert all(
        row["decision"]
        in {"selected", "redundant", "bank_limit", "nonpositive_priority", "zero_gradient_sketch"}
        for row in audit
    )
    x = members.clone().requires_grad_()
    leaves = score_bank(bank, x, truths, mass, filt)
    assert sum(v.shape[-1] for v in leaves.values()) == len(bank["selected"])
    sum(v.sum() for v in leaves.values()).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    altered = copy.deepcopy(bank)
    altered["directions"][0][0] += 1
    with pytest.raises(ValueError):
        validate_bank(
            altered,
            field_order=["a", "b", "c"],
            source_identity="source-hash",
            support_identity="support-hash",
        )
    with pytest.raises(ValueError):
        fit_train_bank(
            panel,
            field_order=["a", "b", "c"],
            masses=mass,
            region_masses=mass[None],
            filter_fn=filt,
            anchor_fn=anchor,
            source_identity="s",
            support_identity="p",
            units="model",
            split="VALIDATION",
        )


def test_estimator_audit_reports_actual_gradient_error():
    endpoint = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
    parameter = torch.tensor([[2.0, 1.0], [-1.0, 3.0]], dtype=torch.float64)
    result = benchmark(
        [0.5, 0.5],
        {"uniform": [0.5, 0.5], "biased": [0.8, 0.2]},
        [1.0, 2.0],
        draws=3,
        replicates=20,
        seed=10,
        endpoint_gradient_fn=lambda w: w @ endpoint,
        parameter_gradient_fn=lambda w: w @ parameter,
    )
    assert result["proposals"]["uniform"]["parameter_gradient_mse"] > 0
    assert result["proposals"]["uniform"]["mean_charged_cost"] > 0
    assert -1 <= result["proposals"]["uniform"]["proxy_error_correlation"] <= 1


def test_real_spectral_filter_follows_fitted_family():
    from scalable_test_helpers import _fitted, _states, config

    from phycoflow_reconstruction.coherence.adaptive_fitting import spectral_filter
    from phycoflow_reconstruction.coherence.families.multiscale_spectrum.statistics import (
        band_fields,
        piecewise_fluctuations,
        spectral_coefficients,
    )

    states = _states()
    family, _, _ = _fitted(config(), states)
    x = states[:2].clone().requires_grad_()
    actual = spectral_filter(family, x)
    g = family._geometry_tensors(family.compute_dtype)
    chosen = family._select(x)
    fluctuations = piecewise_fluctuations(chosen, g["piece_measures"], g["pieces"])
    expected = band_fields(
        spectral_coefficients(fluctuations, g["basis"] * g["mass"][:, None]),
        g["basis"],
        g["filters"],
    )
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_fixed_panel_outer_profile_enumerates_inner_and_charges_cost(monkeypatch):
    from types import SimpleNamespace

    import phycoflow_reconstruction.coherence.adaptive_profile as module

    calls = []

    class Adapter:
        def __init__(self, families, artifact):
            pass

        def score(self, name, prediction, reference, *, coordinates, context, batch, coefficients):
            q = batch.pair_queries[0]
            calls.append((name, q.index, q.inner_sign, q.inner_line))
            unary = batch.self_queries[0]
            val = (prediction[..., 0] * (q.index + 1) * q.weight
                   + context["conditional_draws"][0][..., 1] * unary.weight
                   + prediction[..., 2] * 10).mean()
            return SimpleNamespace(scalar_loss=val), {}, {}

    monkeypatch.setattr(module, "ScalableQueryAdapter", Adapter)
    generator = torch.Generator().manual_seed(5)
    members = torch.randn(2, 4, 4, 3, generator=generator)
    panel = {
        "members": members,
        "reference": members[:, 0],
        "coordinates": torch.zeros(2, 4, 2),
        "masses": torch.ones(2, 4),
        "sample_ids": ["train:1", "train:2"],
    }
    artifact = {
        "provenance": {
            "field_order": ["a", "b", "c"],
            "source_checkpoint_sha256": "source",
            "sample_ids": ["train:1", "train:2"],
        },
        "shape_artifact": {},
    }
    result = module.profile_source_panel(
        panel,
        families={},
        artifacts=artifact,
        filter_fn=lambda x: x,
        source_family="ffm",
        family_weights={"multiscale_spectrum": 1.0, "topology": 1.0},
        parameter_gradient_fn=lambda w: (torch.arange(42, dtype=w.dtype) * w).sum(),
        research_costs={
            "screening_seconds": 1.0,
            "fitting_seconds": 1.0,
            "audit_seconds": 1.0,
            "refresh_seconds": 1.0,
            "refresh_count": 2,
        },
        policy_overhead_fraction=0.01,
        bundle_cost_fn=lambda i: (0.1 + 0.01 * i, 2.0),
        draws=2,
        replicates=10,
        seed=3,
    )
    assert len(calls) == (3 * 12 + 2 * 3) * 2 * 2
    assert result["outer_population"] == 3 and result["inner_shape_enumerated"]
    assert set(result["proposals"]) == {"uniform", "c8_difficulty", "hierarchy", "gradient_cost"}
    assert all(row["total_charged_seconds"] > 2 for row in result["proposals"].values())


def test_policy_qualification_requires_both_sources_twenty_percent_and_overhead():
    from phycoflow_reconstruction.coherence.adaptive_profile import qualify_policy

    def profile(family, source, candidate, overhead=0.03):
        return {
            "source_family": family,
            "source_identity": source,
            "variance_scope": "joint pair and unary outer and inner draws; duplicate pair IDs share one inner draw",
            "research_amortization_updates": 13000,
            "policy_overhead_fraction": overhead,
            "operational_sampler_overhead_fraction": overhead,
            "complete_update_seconds": 10.0,
            "total_policy_research_seconds": 0.0,
            "policy_research_amortized_seconds_per_update": 0.0,
            "incremental_leaf_seconds": [1.0, 2.0, 3.0],
            "outer_population": 3,
            "proposal_probabilities": {"gradient_cost": [1/3, 1/3, 1/3]},
            "proposals": {
                "uniform": {"parameter_mse_times_total_cost": 10.0},
                "gradient_cost": {
                    "parameter_mse_times_total_cost": candidate,
                    "proxy_error_correlation": 0.5,
                    "maximum_importance_weight": 1.0,
                },
            },
        }

    bank = {"source_identity": "ffm-source", "sha256": "old"}
    good = [profile("ffm", "ffm-source", 7.9), profile("edm", "edm-source", 8.0)]
    qualified = qualify_policy(bank, good)
    assert qualified["policy_qualified"]
    assert qualified["policy_mode"] == "fixed_source_profile"
    assert qualified["policy_pair_probabilities"] == [1/3, 1/3, 1/3]
    assert len(qualified["policy_profile_sha256"]) == 64
    undercounted = copy.deepcopy(good)
    undercounted[0].update(total_policy_research_seconds=3900.0,
                           policy_research_amortized_seconds_per_update=0.3,
                           policy_overhead_fraction=0.03)
    assert not qualify_policy(bank, undercounted)["policy_qualified"]
    undercounted[0]["policy_overhead_fraction"] = 0.6 / 10.3
    assert not qualify_policy(bank, undercounted)["policy_qualified"]
    missing_vector = copy.deepcopy(good)
    del missing_vector[0]["proposal_probabilities"]
    assert not qualify_policy(bank, missing_vector)["policy_qualified"]
    good[1]["proposals"]["gradient_cost"]["parameter_mse_times_total_cost"] = 8.1
    assert not qualify_policy(bank, good)["policy_qualified"]
    good[1] = profile("edm", "edm-source", 7.0, overhead=0.06)
    assert not qualify_policy(bank, good)["policy_qualified"]
    with pytest.raises(ValueError):
        qualify_policy(bank, good, minimum_improvement=0.05)


def test_profile_bundle_cost_attributes_added_pair_and_removes_anchor():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / 'scripts/benchmarks/profile_car_allocator.py'
    spec = importlib.util.spec_from_file_location('car_allocator_driver', path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    costs = [1., 4., 9.]
    def timed(anchor, extra):
        return 100. + costs[anchor] + (costs[extra] if extra is not None else 0.)
    rows = driver.measure_bundle_costs(timed, 3, 3)
    assert [row['incremental_seconds'] for row in rows] == costs
    assert all(row['fixed_seconds'] == 100. for row in rows)


def test_discovery_anchor_all_four_members_change_complementary_gradient(monkeypatch):
    from types import SimpleNamespace

    import phycoflow_reconstruction.training.scalable_fitting as fitting

    class State:
        def __init__(self):
            self.component_weights = {"self_marginal_w2": 1.0, "cross_joint_topk_swd": 1.0}

        def _in_declared_units(self, generated, reference):
            return generated, reference

    class Adapter:
        def __init__(self):
            self.observed = []

        def score(self, name, first, reference, *, coordinates, context, batch, coefficients):
            members = (first, *context["conditional_draws"])
            self.observed.append((name, len(members), context["score_all_conditional_draws"]))
            factor = 2.0 if name == "multiscale_spectrum" else 3.0
            loss = sum((i + 1) * item.square().sum() for i, item in enumerate(members)) * factor
            return SimpleNamespace(scalar_loss=loss), {}, {}

    def state_score(members, reference, *args, **kwargs):
        return SimpleNamespace(scalar_loss=members.square().sum()), {}

    monkeypatch.setattr(fitting, "score_family", state_score)
    adapter = Adapter()
    x = torch.arange(1.0, 33.0).reshape(4, 1, 4, 2).requires_grad_()
    y = torch.zeros(1, 4, 2)

    def anchor(value):
        return fitting._discovery_anchor(
            value,
            y,
            torch.ones(4),
            state_family=State(),
            bank={},
            component=SimpleNamespace(channel_weights=torch.ones(2)),
            adapter=adapter,
            fields=("a", "b"),
            coordinates=torch.zeros(1, 4, 2),
            batch=None,
        )

    gradient = torch.autograd.grad(anchor(x), x)[0]
    assert all(gradient[i].abs().sum() > 0 for i in range(4))
    assert adapter.observed == [("multiscale_spectrum", 4, True), ("topology", 4, True)]
    changed = x.detach().clone()
    changed[2] *= 2
    changed[3] *= 3
    changed.requires_grad_()
    changed_gradient = torch.autograd.grad(anchor(changed), changed)[0]
    assert not torch.equal(gradient[2:], changed_gradient[2:])
    probe = torch.arange(1.0, 33.0).reshape_as(gradient).flip(0)

    def complement(g):
        return probe - (probe.flatten() @ g.flatten()) / g.square().sum() * g

    assert not torch.allclose(complement(gradient), complement(changed_gradient))


def test_policy_overhead_charges_profile_and_prior_research_against_complete_updates():
    from phycoflow_reconstruction.coherence.adaptive_profile import (
        _valid_overhead_accounting,
        finalize_policy_overhead,
    )

    profile = {"operational_sampler_overhead_fraction": 0.03,
               "research_amortization_updates": 13000,
               "research_costs": {"audit_seconds": 1300.0},
               "total_policy_research_seconds": 1300.0,
               "proposals": {"uniform": {"total_charged_seconds": 2.0},
                             "gradient_cost": {"research_amortized_seconds": 0.1,
                                               "total_charged_seconds": 2.1,
                                               "parameter_gradient_mse": 2.0}}}
    finalize_policy_overhead(profile, measured_profile_seconds=2600.0, complete_update_seconds=10.0)
    assert profile["policy_overhead_fraction"] == pytest.approx((0.3 + 0.3) / 10.3)
    assert profile["policy_overhead_fraction"] > 0.05
    assert profile["proposals"]["gradient_cost"]["parameter_mse_times_total_cost"] == pytest.approx(4.6)
    assert _valid_overhead_accounting(profile)
    profile["policy_overhead_fraction"] = 0.03  # old, operational-only accounting
    assert not _valid_overhead_accounting(profile)
    with pytest.raises(ValueError, match="unfinalized"):
        finalize_policy_overhead(profile, measured_profile_seconds=1, complete_update_seconds=10)


def test_uncentered_ht_leaf_gradients_match_runtime_with_fixed_anchor():
    from phycoflow_reconstruction.coherence.adaptive_profile import isolate_query_stratum
    from phycoflow_reconstruction.coherence.query_sampling import Draw, QueryBatch

    theta = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    probabilities = (0.25, 0.75)
    pair_gradients = (2.0, 6.0)
    extracted = []
    for index in range(2):
        query = QueryBatch((Draw(0, 1, 1.0, 1, 1),),
                           (Draw(index, 1, 0.5, 2, 1),), 1)
        def runtime(batch, index=index):
            return theta * (37.0 + 5.0 * batch.self_queries[0].weight
                            + pair_gradients[index] * batch.pair_queries[0].weight)
        leaf = isolate_query_stratum(runtime, query, "pair")
        extracted.append(torch.autograd.grad(leaf, theta, retain_graph=True)[0])
        unary = isolate_query_stratum(runtime, query, "self")
        assert torch.autograd.grad(unary, theta, retain_graph=True)[0].item() == pytest.approx(5.0)
    torch.testing.assert_close(torch.stack(extracted), torch.tensor(pair_gradients, dtype=torch.float64))
    target = sum(pair_gradients) / 2
    estimates = [extracted[i].item() / (2 * probabilities[i]) for i in range(2)]
    mse = sum(p * (estimate - target) ** 2 for p, estimate in zip(probabilities, estimates))
    assert mse == pytest.approx(0.0)
    centered = [(extracted[i] - extracted[0]).item() / (2 * probabilities[i]) for i in range(2)]
    centered_target = (extracted[1] - extracted[0]).item() / 2
    assert sum(p * (v - centered_target) ** 2 for p, v in zip(probabilities, centered)) > 0
    # The actual runtime estimator, with its complete anchor added once, has
    # the same zero error even though its HT weight sums differ by proposal.
    for i in range(2):
        batch = QueryBatch((Draw(0, 1, 1.0, 1, 1),),
                           (Draw(i, 1, probabilities[i], 2, 1),), 1)
        objective = theta * (37 + 5 * batch.self_queries[0].weight + pair_gradients[i] * batch.pair_queries[0].weight)
        assert torch.autograd.grad(objective, theta, retain_graph=True)[0].item() == pytest.approx(37 + 5 + target)
