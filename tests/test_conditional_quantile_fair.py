"""Numerical contracts for conditional fair weighted-occupation State."""

import copy
import itertools

import pytest
import torch

from phycoflow_reconstruction.coherence import build_coherence_family
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer


def config(n=2):
    names = tuple(f"f{i}" for i in range(n))
    return {
        "definition": "conditional_quantile_fair_v1",
        "base_definition": "marginal_mixed_weighted_v3",
        "conditional_draws": 2,
        "fields": names,
        "target_use": "paired_supervised",
        "units": "model_units",
        "components": {
            "self": {"enabled": True, "weight": 1.3},
            "mutual": {"enabled": False},
            "cross": {
                "enabled": n > 1,
                "weight": 0.7,
                "directions": 8,
                "top_fraction": 1.0,
                "include_axes": False,
                "qmc": True,
            },
        },
    }


def family(n=2, cfg=None):
    return build_coherence_family(
        "global_distribution",
        cfg or config(n),
        DataSpec(tuple(f"f{i}" for i in range(n)), ("1",) * n, 2, (4,)),
        FieldNormalizer.identity(n),
    )


def ctx(n=4, **kw):
    return {
        "generated_weights": torch.ones(1, n, dtype=torch.double),
        "reference_weights": torch.ones(1, n, dtype=torch.double),
        **kw,
    }


def scalar(x):
    return torch.full((1, 4, 1), float(x), dtype=torch.double)


def test_two_state_truthful_mixture_beats_collapse_but_squared_score_ties():
    f = family(1)
    fair_truth, fair_collapse, wrong_truth, wrong_collapse = [], [], [], []
    for x, y, z in itertools.product((0, 2), repeat=3):
        a, b, target = scalar(x), scalar(y), scalar(z)
        fair_truth.append(f(a, target, context=ctx(conditional_draws=(b,))).scalar_loss.item())
        fair_collapse.append(
            f(scalar(1), target, context=ctx(conditional_draws=(scalar(1),))).scalar_loss.item()
        )
        d = lambda p, q: f.paired_squared(p, q, context=ctx()).scalar_loss.item()
        wrong_truth.append((d(a, target) + d(b, target) - d(a, b)) / 2)
        wrong_collapse.append(d(scalar(1), target))
    mean = lambda xs: sum(xs) / len(xs)
    assert mean(fair_truth) < mean(fair_collapse)
    assert mean(wrong_truth) == pytest.approx(mean(wrong_collapse))


@pytest.mark.parametrize("j", [2, 4])
def test_coefficients_order_gradient_and_decomposition(j):
    torch.manual_seed(4)
    f = family()
    members = [torch.randn(1, 4, 2, dtype=torch.double, requires_grad=True) for _ in range(j)]
    target = torch.randn(1, 4, 2, dtype=torch.double)
    context = ctx(conditional_draws=tuple(members[1:]), score_all_conditional_draws=True)
    result = f(members[0], target, context=context)
    d = lambda a, b: f.paired_squared(a, b, context=ctx()).per_sample_cost.sqrt()
    expected_a = sum(d(a, target) for a in members) / j
    expected_r = sum(d(a, b) for a, b in itertools.combinations(members, 2)) / (j * (j - 1))
    assert torch.allclose(result.per_sample_cost, expected_a - expected_r)
    assert torch.allclose(result.diagnostics["attraction"], expected_a.detach())
    assert torch.allclose(result.diagnostics["separation"], expected_r.detach())
    reverse = f(
        members[-1],
        target,
        context=ctx(
            conditional_draws=tuple(reversed(members[:-1])), score_all_conditional_draws=True
        ),
    )
    assert torch.allclose(result.scalar_loss, reverse.scalar_loss)
    result.scalar_loss.backward()
    assert all(
        m.grad is not None and torch.isfinite(m.grad).all() and m.grad.abs().sum() > 0
        for m in members
    )
    assert all(
        v["weight"] == 0
        for k, v in result.diagnostics["components"].items()
        if not k.endswith("conditional_quantile_fair")
    )


def test_coincidence_flat_collinear_outlier_and_ties():
    f = family(1)
    zero = scalar(0).requires_grad_()
    second = scalar(0).requires_grad_()
    loss = f(zero, scalar(0), context=ctx(conditional_draws=(second,))).scalar_loss
    loss.backward()
    assert loss == 0 and torch.isfinite(zero.grad).all() and torch.isfinite(second.grad).all()
    for a, b, z in [(-2, 2, 0), (0, 100, 0), (-5, 20, 4)]:
        assert (
            abs(
                f(
                    scalar(a), scalar(z), context=ctx(conditional_draws=(scalar(b),))
                ).scalar_loss.item()
            )
            < 1e-12
        )


@pytest.mark.parametrize("nfields", [1, 2, 3])
def test_nonuniform_masks_support_permutations_and_base_parity(nfields):
    torch.manual_seed(nfields)
    f = family(nfields)
    a = torch.randn(1, 4, nfields, dtype=torch.double)
    b = torch.randn_like(a)
    z = torch.randn_like(a)
    w = torch.tensor([[1.0, 2.0, 5.0, 8.0]], dtype=torch.double)
    v = torch.tensor([[5.0, 1.0, 4.0, 2.0]], dtype=torch.double)
    mask = torch.tensor([[True, False, True, True]])
    c = {
        "generated_weights": w,
        "reference_weights": v,
        "generated_mask": mask,
        "reference_mask": mask,
        "conditional_draws": (b,),
    }
    res = f(a, z, context=c)
    perm = torch.tensor([2, 0, 3, 1])
    pc = {
        "generated_weights": w[:, perm],
        "reference_weights": v[:, perm],
        "generated_mask": mask[:, perm],
        "reference_mask": mask[:, perm],
        "conditional_draws": (b[:, perm],),
    }
    assert torch.allclose(
        res.scalar_loss, f(a[:, perm], z[:, perm], context=pc).scalar_loss, atol=1e-12
    )
    base_cfg = copy.deepcopy(config(nfields))
    base_cfg["definition"] = "marginal_mixed_weighted_v3"
    base_cfg.pop("base_definition")
    base_cfg.pop("conditional_draws")
    base = family(nfields, base_cfg)
    expected = base(a, z, context=c)
    assert torch.equal(f.paired_squared(a, z, context=c).per_sample_cost, expected.per_sample_cost)
    assert torch.equal(res.diagnostics["paired_squared"], expected.per_sample_cost)


def test_unsupported_definitions_and_missing_members_rejected():
    c = config()
    c["components"]["cross"]["top_fraction"] = 0.5
    with pytest.raises(ValueError):
        family(cfg=c)
    c = config()
    c["components"]["mutual"]["enabled"] = True
    with pytest.raises(ValueError):
        family(cfg=c)
    c = config()
    c["base_definition"] = "other"
    with pytest.raises(ValueError):
        family(cfg=c)
    with pytest.raises(ValueError, match="configured draws"):
        family()(torch.zeros(1, 4, 2), torch.zeros(1, 4, 2), context=ctx())


def test_base_artifact_conversion_and_strict_resume(monkeypatch):
    from phycoflow_reconstruction.coherence.families.global_distribution import conditional
    from phycoflow_reconstruction.coherence.measures import measure_fingerprint

    c = config()
    base_cfg = copy.deepcopy(c)
    base_cfg["definition"] = "marginal_mixed_weighted_v3"
    base_cfg.pop("base_definition")
    base_cfg.pop("conditional_draws")
    base = family(cfg=base_cfg)
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
    receipt = measure_fingerprint(weights)
    base.bind_measure_artifact(receipt, weights=weights)
    candidate = family(cfg=c)
    candidate.load_base_artifact(base.state_artifact())
    assert torch.equal(
        candidate.components_by_key["cross_joint_topk_swd"].directions,
        base.components_by_key["cross_joint_topk_swd"].directions,
    )
    restored = family(cfg=c)
    restored.load_state_artifact(candidate.state_artifact())
    altered = copy.deepcopy(candidate.state_artifact())
    altered["conditional_draws"] = 4
    with pytest.raises(ValueError, match="identity"):
        restored.load_state_artifact(altered)
    altered = copy.deepcopy(candidate.state_artifact())
    altered["implementation_identity"]["sources"]["v3_family"] = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        restored.load_state_artifact(altered)
    saved = candidate.state_artifact()
    current_identity = conditional._implementation_identity()
    current_identity["sources"]["draw_routing"] = "1" * 64
    monkeypatch.setattr(conditional, "_implementation_identity", lambda: current_identity)
    with pytest.raises(ValueError, match="identity"):
        restored.load_state_artifact(saved)
    altered_base = base.state_artifact()
    altered_base["config"] = copy.deepcopy(altered_base["config"])
    altered_base["config"]["units"] = "physical_units"
    with pytest.raises(ValueError, match="configuration"):
        candidate.load_base_artifact(altered_base)


def test_exact_nonuniform_quantile_integral():
    f = family(1)
    a = torch.tensor([[[0.0], [4.0], [30.0], [50.0]]], dtype=torch.double)
    b = torch.tensor([[[1.0], [2.0], [-20.0], [-40.0]]], dtype=torch.double)
    context = {
        "generated_weights": torch.tensor([[1.0, 3.0, 9.0, 9.0]], dtype=torch.double),
        "reference_weights": torch.tensor([[1.0, 1.0, 9.0, 9.0]], dtype=torch.double),
        "generated_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.bool),
        "reference_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.bool),
    }
    # Integral = .25*(0-1)^2 + .25*(4-1)^2 + .5*(4-2)^2.
    assert f.paired_squared(a, b, context=context).scalar_loss.item() == pytest.approx(1.3 * 4.5)
