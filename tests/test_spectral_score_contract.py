"""Regression checks for joint identification and finite score diagnostics."""

import itertools
import json

import pytest
import torch

from phycoflow_reconstruction.coherence.families.multiscale_spectrum import (
    MultiscaleSpectrumFamily,
    parse_multiscale_config,
)
from phycoflow_reconstruction.coherence.families.multiscale_spectrum.losses import (
    field_projections,
    joint_spectral_energy_score,
    signed_band_energy_score,
    vector_energy_score,
)
from phycoflow_reconstruction.coherence.families.multiscale_spectrum.reporting import (
    finite_values,
    positive_ratio,
)
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer


def test_joint_coefficients_match_spatial_distance_and_gradients():
    rng = torch.Generator().manual_seed(71)
    options = {"generator": rng, "dtype": torch.float64}
    mass = torch.rand(11, **options) + 0.2
    orthogonal, _ = torch.linalg.qr(torch.randn(11, 6, **options))
    basis = orthogonal / mass.sqrt()[:, None]
    filters = torch.rand(3, 6, **options)  # deliberately overlapping filters
    scales = torch.rand(3, 2, **options) + 0.1
    draws = torch.randn(4, 2, 6, 2, **options).requires_grad_()
    target = torch.randn(2, 6, 2, **options)
    actual = joint_spectral_energy_score(draws, target, filters=filters, scales=scales, mass=mass)
    fields = torch.einsum("nk,sk,jbkc->jbsnc", basis, filters, draws)
    truth = torch.einsum("nk,sk,bkc->bsnc", basis, filters, target)
    root = (mass / mass.sum()).sqrt()[None, :, None] / scales[:, None, :]
    expected = vector_energy_score((fields * root).flatten(2), (truth * root).flatten(1))
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right)
    left = torch.autograd.grad(actual[0].sum(), draws, retain_graph=True)[0]
    right = torch.autograd.grad(expected[0].sum(), draws)[0]
    torch.testing.assert_close(left, right)
    assert all(value.norm() > 0 for value in left)


def test_joint_score_distinguishes_parity_that_axes_and_pairs_cannot():
    # Every univariate and bivariate marginal agrees; the three-way laws differ.
    support = torch.tensor(list(itertools.product((-1.0, 1.0), repeat=3)), dtype=torch.float64)
    truth = support[support.prod(-1) == 1]
    wrong = support[support.prod(-1) == -1]
    triples = torch.tensor(list(itertools.product(range(4), repeat=3)))
    targets = truth[triples[:, 2]]
    projections = field_projections(3, "axes_and_pairs", dtype=truth.dtype, device=truth.device)

    def expected_scores(forecast):
        draws = forecast[triples[:, :2]].transpose(0, 1)
        marginal = signed_band_energy_score(
            draws[:, :, None, None, :],
            targets[:, None, None, :],
            mass=torch.ones(1),
            scales=torch.ones(1, 3),
            projections=projections,
        )[0].mean()
        joint = joint_spectral_energy_score(
            draws[:, :, None, :],
            targets[:, None, :],
            filters=torch.ones(1, 1),
            scales=torch.ones(1, 3),
            mass=torch.ones(1),
        )[0].mean()
        return marginal, joint

    correct_marginal, correct_joint = expected_scores(truth)
    wrong_marginal, wrong_joint = expected_scores(wrong)
    torch.testing.assert_close(correct_marginal, wrong_marginal)
    assert wrong_joint > correct_joint + 0.05


def test_chunked_estimator_matches_all_pairs_and_keeps_coincident_gradients_finite():
    draws = torch.randn(9, 2, 5, dtype=torch.float64, requires_grad=True)
    target = torch.randn(2, 5, dtype=torch.float64)
    small = vector_energy_score(draws, target, alpha=0.95, pair_chunk_size=3)
    large = vector_energy_score(draws, target, alpha=0.95, pair_chunk_size=100)
    for left, right in zip(small, large):
        torch.testing.assert_close(left, right)
    coincident = torch.zeros(3, 2, 5, requires_grad=True)
    vector_energy_score(coincident, torch.zeros(2, 5))[0].sum().backward()
    assert torch.equal(coincident.grad, torch.zeros_like(coincident))


@pytest.mark.parametrize("fields", [("u",), ("u", "v")])
def test_fair_joint_family_is_finite_and_round_trips(fields):
    channels = len(fields)
    config = {
        "fields": list(fields),
        "geometry": {"operator": "fem_p1", "coordinate_scaling": "none"},
        "bands": {"count": 2},
        "numerics": {"dtype": "float64"},
        "objective": {
            "kind": "signed_band_energy_score",
            "almost_fair_alpha": 1.0,
            "joint_energy_score_weight": 0.25,
        },
    }
    spec = DataSpec(fields, ("1",) * channels, 1, (12,))
    normalizer = FieldNormalizer.identity(channels)
    family = MultiscaleSpectrumFamily(config, spec, normalizer)
    coords = torch.linspace(0, 1, 12, dtype=torch.float64)[:, None]
    reference = torch.randn(
        10, 12, channels, generator=torch.Generator().manual_seed(21), dtype=torch.float64
    )
    family.fit_reference_tensors(coords, reference, sample_ids=[str(i) for i in range(10)])
    restored = MultiscaleSpectrumFamily(config, spec, normalizer)
    restored.load_state_artifact(family.state_artifact())
    draw = reference[:2].clone().requires_grad_()
    other = reference[2:4].clone().requires_grad_()
    kwargs = {
        "coordinates": coords[None].expand(2, -1, -1),
        "context": {"conditional_draws": (other,)},
    }
    result = family(draw, reference[4:6], **kwargs)
    torch.testing.assert_close(
        result.scalar_loss, restored(draw, reference[4:6], **kwargs).scalar_loss
    )
    result.scalar_loss.backward()
    assert torch.isfinite(draw.grad).all() and torch.isfinite(other.grad).all()
    components = result.component_results
    main = components["multiscale_spectrum.signed_band_energy_score.conditional"].scalar_loss
    joint = components["multiscale_spectrum.joint_spectral_energy_score.conditional"].scalar_loss
    torch.testing.assert_close(result.scalar_loss, main + 0.25 * joint)
    json.dumps(result.diagnostics, allow_nan=False)
    if channels == 1:
        auxiliary = components["multiscale_spectrum.band_coherence_crps.conditional"]
        assert auxiliary.scalar_loss == 0 and not auxiliary.diagnostics["available"]
        config["objective"]["coherence_crps_weight"] = 0.1
        with pytest.raises(ValueError, match="at least two fields"):
            parse_multiscale_config(config, fields)


@pytest.mark.parametrize("value", [-1.0, float("nan"), True])
def test_joint_weight_rejects_invalid_settings(value):
    config = {
        "bands": {"count": 2},
        "objective": {"kind": "signed_band_energy_score", "joint_energy_score_weight": value},
    }
    with pytest.raises(ValueError, match="joint_energy_score_weight"):
        parse_multiscale_config(config, ("u", "v"))


def test_joint_score_requires_all_terms_to_be_fair():
    with pytest.raises(ValueError, match="almost_fair_alpha=1"):
        parse_multiscale_config(
            {
                "bands": {"count": 2},
                "objective": {"kind": "signed_band_energy_score", "joint_energy_score_weight": 0.1},
            },
            ("u", "v"),
        )


def test_nonpositive_debiased_error_does_not_become_a_huge_ratio():
    ratios = positive_ratio(torch.ones(4), torch.tensor([-1.0, 0.0, 0.25, float("nan")]))
    assert finite_values(ratios) == [None, None, 4.0, None]
    assert finite_values(ratios.sqrt()) == [None, None, 2.0, None]
