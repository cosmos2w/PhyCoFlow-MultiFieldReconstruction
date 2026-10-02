"""Acceptance tests for the opt-in marginal/copula distribution definition."""

from __future__ import annotations

import math

import pytest
import torch

from phycoflow_reconstruction.coherence import build_coherence_family
from phycoflow_reconstruction.coherence.families.global_distribution.components.copula import (
    exact_midrank_uniform,
    quantile_landmark_smooth_cdf,
)
from phycoflow_reconstruction.coherence.families.global_distribution.components.tail_risk import (
    identity_calibrated_smooth_cvar,
)
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer


def _v2_config(
    fields: tuple[str, ...] = ("u", "v"),
    *,
    pairwise_diagnostics: bool = False,
) -> dict:
    return {
        "definition": "marginal_copula_v2",
        "target_use": "paired_supervised",
        "units": "model_units",
        "fields": fields,
        "components": {
            "self": {"enabled": True, "weight": 1.0},
            "mutual": {"enabled": False, "weight": 0.0},
            "cross": {
                "enabled": True,
                "weight": 1.0,
                "directions": 32,
                "seed": 19,
                "include_axes": False,
                "qmc": True,
                "pairwise_diagnostics": pairwise_diagnostics,
                "copula": {
                    "canonicalizer": "quantile_landmark_smooth_cdf",
                    "landmarks": 64,
                    "bandwidth": 0.1,
                    "scale_floor": 1e-6,
                    "chunk_size": 16,
                },
                "tail": {
                    "alpha": 0.25,
                    "rho": 0.10,
                    "temperature": 1e-3,
                    "eta_tolerance": 1e-7,
                    "eta_max_iterations": 80,
                },
            },
        },
    }


def _family(config: dict, field_names: tuple[str, ...] | None = None):
    names = field_names or config["fields"]
    return build_coherence_family(
        "global_distribution",
        config,
        DataSpec(names, ("1",) * len(names), 2, (8,)),
        FieldNormalizer.identity(len(names)),
    )


def _cross(result):
    return result.component_results["global_distribution.cross.joint_copula_cvar"]


def test_exact_midranks_use_deterministic_average_ranks_for_ties() -> None:
    values = torch.tensor([[0.0], [0.0], [2.0], [2.0], [2.0], [5.0]])
    expected = torch.tensor([[1.0 / 6.0], [1.0 / 6.0], [7.0 / 12.0], [7.0 / 12.0], [7.0 / 12.0], [11.0 / 12.0]])
    torch.testing.assert_close(exact_midrank_uniform(values), expected)
    assert not exact_midrank_uniform(values.requires_grad_()).requires_grad


def test_exact_midranks_exclude_masked_rows_and_are_row_permutation_invariant() -> None:
    values = torch.tensor([[0.0, 1.0], [2.0, 3.0], [2.0, -1.0], [7.0, 0.0]])
    mask = torch.tensor([True, False, True, True])
    actual = exact_midrank_uniform(values, mask)
    expected = exact_midrank_uniform(values[mask])
    torch.testing.assert_close(actual, expected)
    permutation = torch.tensor([2, 0, 1])
    torch.testing.assert_close(actual[permutation], exact_midrank_uniform(values[mask][permutation]))


def test_smooth_copula_is_scale_normalized_and_has_gradients_through_landmarks() -> None:
    torch.manual_seed(1002)
    values = torch.randn(19, 3, dtype=torch.float64)
    transformed = 7.25 * values + torch.tensor([3.0, -5.0, 1.5], dtype=torch.float64)
    first = quantile_landmark_smooth_cdf(values, landmarks=16, bandwidth=0.1)
    second = quantile_landmark_smooth_cdf(transformed, landmarks=16, bandwidth=0.1)
    torch.testing.assert_close(first, second, atol=2e-12, rtol=2e-12)

    weights = torch.randn_like(values)
    objective = lambda current: (
        quantile_landmark_smooth_cdf(
            current, landmarks=16, bandwidth=0.1, chunk_size=5
        )
        * weights
    ).sum()
    assert torch.autograd.gradcheck(objective, (values.clone().requires_grad_(),))


def test_exact_copula_is_monotone_invariant_and_soft_approximation_departure_is_visible() -> None:
    torch.manual_seed(6)
    values = torch.randn(96, 2, dtype=torch.float64)
    monotone = torch.exp(0.4 * values) + 0.1 * values.pow(3)
    exact = exact_midrank_uniform(values)
    exact_monotone = exact_midrank_uniform(monotone)
    torch.testing.assert_close(exact, exact_monotone, atol=0.0, rtol=0.0)

    soft = quantile_landmark_smooth_cdf(values, landmarks=64, bandwidth=0.1)
    soft_monotone = quantile_landmark_smooth_cdf(
        monotone, landmarks=64, bandwidth=0.1
    )
    departure = (soft - soft_monotone).abs().mean()
    assert 1e-5 < float(departure) < 0.1


def test_canonicalizers_handle_ties_constants_near_constants_and_small_n() -> None:
    tied = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 2.0], [0.0, 2.0]])
    exact = exact_midrank_uniform(tied)
    torch.testing.assert_close(exact[:, 0], torch.full((4,), 0.5))
    soft = quantile_landmark_smooth_cdf(tied, landmarks=8, chunk_size=2)
    torch.testing.assert_close(soft[:, 0], torch.full((4,), 0.5))
    assert torch.isfinite(soft).all()

    almost_constant = torch.tensor(
        [[1e-9, 0.0], [0.0, 1e-9], [-1e-9, -1e-9]],
        dtype=torch.float64,
        requires_grad=True,
    )
    output = quantile_landmark_smooth_cdf(
        almost_constant, landmarks=4, scale_floor=1e-6
    )
    gradient = torch.autograd.grad(output.square().sum(), almost_constant)[0]
    assert torch.isfinite(output).all()
    assert torch.isfinite(gradient).all()

    with pytest.raises(ValueError, match="at least 2 points"):
        quantile_landmark_smooth_cdf(torch.ones(1, 2))


def _hard_cvar(values: torch.Tensor, rho: float) -> torch.Tensor:
    tail_mass = rho * values.numel()
    whole_directions = math.floor(tail_mass)
    fractional_direction = tail_mass - whole_directions
    descending = values.sort(descending=True).values
    numerator = descending[:whole_directions].sum()
    if fractional_direction > 0.0:
        numerator = numerator + fractional_direction * descending[whole_directions]
    return numerator / tail_mass


def test_smooth_cvar_is_identity_calibrated_monotone_and_translation_equivariant() -> None:
    costs = torch.tensor([0.02, 0.04, 0.09, 0.12, 0.18], dtype=torch.float64)
    result = identity_calibrated_smooth_cvar(
        costs,
        rho=0.4,
        temperature=0.05,
        tolerance=1e-10,
        max_iterations=120,
    )
    shift = 3.25
    shifted = identity_calibrated_smooth_cvar(
        costs + shift,
        rho=0.4,
        temperature=0.05,
        tolerance=1e-10,
        max_iterations=120,
    )
    torch.testing.assert_close(shifted.value, result.value + shift)
    assert float(result.eta_residual) <= 1e-10

    constant = identity_calibrated_smooth_cvar(
        torch.full((7,), 0.37, dtype=torch.float64),
        rho=0.2,
        temperature=0.03,
        tolerance=1e-10,
        max_iterations=120,
    )
    torch.testing.assert_close(constant.value, torch.tensor(0.37, dtype=torch.float64))

    hotter_costs = costs.clone()
    hotter_costs[-1] += 0.05
    hotter = identity_calibrated_smooth_cvar(
        hotter_costs,
        rho=0.4,
        temperature=0.05,
        tolerance=1e-10,
        max_iterations=120,
    )
    assert hotter.value > result.value

    low_temperature = identity_calibrated_smooth_cvar(
        costs,
        rho=0.3,
        temperature=1e-4,
        tolerance=1e-10,
        max_iterations=120,
    )
    torch.testing.assert_close(
        low_temperature.value, _hard_cvar(costs, 0.3), atol=2e-4, rtol=0.0
    )

    mean_case = identity_calibrated_smooth_cvar(
        costs, rho=1.0, temperature=0.05
    )
    torch.testing.assert_close(mean_case.value, costs.mean())


def test_smooth_cvar_envelope_gradient_matches_finite_difference() -> None:
    values = torch.tensor(
        [0.03, 0.06, 0.08, 0.11, 0.19], dtype=torch.float64, requires_grad=True
    )
    kwargs = {
        "rho": 0.4,
        "temperature": 0.04,
        "tolerance": 1e-12,
        "max_iterations": 140,
    }
    result = identity_calibrated_smooth_cvar(values, **kwargs)
    gradient = torch.autograd.grad(result.value, values)[0]
    delta = 2e-6
    finite_difference = []
    for index in range(values.numel()):
        plus = values.detach().clone()
        minus = values.detach().clone()
        plus[index] += delta
        minus[index] -= delta
        plus_value = identity_calibrated_smooth_cvar(plus, **kwargs).value
        minus_value = identity_calibrated_smooth_cvar(minus, **kwargs).value
        finite_difference.append((plus_value - minus_value) / (2.0 * delta))
    torch.testing.assert_close(gradient, torch.stack(finite_difference), atol=2e-6, rtol=2e-5)
    assert torch.isfinite(gradient).all()


def test_v2_identity_is_zero_with_finite_gradients_and_a_serialized_mixed_bank() -> None:
    config = _v2_config()
    family = _family(config)
    second = _family(config)
    first_component = family.components_by_key["cross_joint_copula_cvar"]
    second_component = second.components_by_key["cross_joint_copula_cvar"]
    torch.testing.assert_close(first_component.directions, second_component.directions)
    assert not any(
        torch.allclose(direction.abs(), torch.eye(2)[axis])
        for direction in first_component.directions
        for axis in range(2)
    )
    second.load_state_artifact(family.state_artifact())

    generated = torch.randn(2, 24, 2, requires_grad=True)
    result = family(generated, generated.detach())
    assert abs(float(result.scalar_loss)) < 2e-7
    gradient = torch.autograd.grad(result.scalar_loss, generated)[0]
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().max()) < 1e-5
    cross = _cross(result)
    assert torch.isfinite(cross.diagnostics["eta_residual"]).all()
    assert cross.diagnostics["tail_temperature_policy"] == "fixed_configured_value"
    assert cross.diagnostics["exact_midranks_projected_w2_diagnostic"].max() == 0


def test_joint_row_permutation_preserves_marginal_and_dependence_descriptors() -> None:
    torch.manual_seed(21)
    generated = torch.randn(2, 32, 2)
    reference = torch.randn(2, 32, 2)
    permutation = torch.randperm(32)
    family = _family(_v2_config())
    direct = family(generated, reference)
    permuted = family(generated[:, permutation], reference[:, permutation])
    torch.testing.assert_close(direct.per_sample_cost, permuted.per_sample_cost)
    direct_cross = _cross(direct)
    permuted_cross = _cross(permuted)
    torch.testing.assert_close(
        direct_cross.diagnostics["exact_midranks_projected_w2_diagnostic"],
        permuted_cross.diagnostics["exact_midranks_projected_w2_diagnostic"],
    )


def test_independent_channel_permutations_preserve_marginals_but_change_dependence() -> None:
    count = 32
    base = torch.linspace(-1.0, 1.0, count)
    generated = torch.stack((base, base), dim=-1).unsqueeze(0)
    aligned_reference = generated.clone()
    permuted_reference = torch.stack((base, base.flip(0)), dim=-1).unsqueeze(0)
    family = _family(_v2_config())

    aligned = family(generated, aligned_reference)
    permuted = family(generated, permuted_reference)
    torch.testing.assert_close(
        aligned.component_results["global_distribution.self.marginal_w2"].scalar_loss,
        permuted.component_results["global_distribution.self.marginal_w2"].scalar_loss,
    )
    assert float(_cross(permuted).scalar_loss) > float(_cross(aligned).scalar_loss) + 1e-3


def test_selected_fields_and_point_masks_are_respected() -> None:
    fields = ("u", "v", "p")
    config = _v2_config(fields)
    family = _family(config)
    torch.manual_seed(31)
    generated = torch.randn(2, 20, 3, requires_grad=True)
    reference = torch.randn(2, 20, 3)
    changed_generated = generated.detach().clone()
    changed_reference = reference.clone()
    changed_generated[..., 1] += 100.0
    changed_reference[..., 1] -= 200.0
    selected_config = _v2_config(("u", "p"))
    selected_family = _family(selected_config, fields)
    subset = selected_family(generated, reference)
    subset_changed = selected_family(changed_generated, changed_reference)
    torch.testing.assert_close(subset.scalar_loss, subset_changed.scalar_loss)

    mask = torch.tensor(
        [[True] * 13 + [False] * 7, [True] * 12 + [False] * 8]
    )
    masked = family(generated, reference, point_mask=mask)
    perturbed = generated.detach().clone()
    perturbed[~mask] = 1e4
    masked_perturbed = family(perturbed, reference, point_mask=mask)
    torch.testing.assert_close(masked.scalar_loss, masked_perturbed.scalar_loss)
    masked_gradient = torch.autograd.grad(masked.scalar_loss, generated)[0]
    assert masked_gradient[~mask].count_nonzero() == 0


def test_generated_marginal_collapse_remains_in_exact_dependence_diagnostic() -> None:
    family = _family(_v2_config())
    generated = torch.zeros(1, 32, 2, requires_grad=True)
    reference = torch.randn(1, 32, 2)
    result = family(generated, reference)
    cross = _cross(result)
    assert cross.diagnostics["exact_midranks_projected_w2_diagnostic"].item() > 0.0
    assert cross.diagnostics["mean_directional_w2"].item() > 0.0
    assert torch.isfinite(result.scalar_loss)
    assert torch.isfinite(torch.autograd.grad(result.scalar_loss, generated)[0]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_v2_cuda_identity_ties_and_collapsed_generated_gradients() -> None:
    device = torch.device("cuda")
    family = _family(_v2_config()).to(device)

    identical = torch.randn(1, 24, 2, device=device, requires_grad=True)
    identity_result = family(identical, identical.detach())
    identity_gradient = torch.autograd.grad(identity_result.scalar_loss, identical)[0]
    assert abs(float(identity_result.scalar_loss)) < 2e-7
    assert torch.isfinite(identity_gradient).all()

    tied = torch.tensor(
        [[0.0, 1.0], [0.0, 1.0], [0.0, 2.0], [0.0, 2.0]],
        dtype=torch.float32,
        device=device,
    )
    exact_ties = exact_midrank_uniform(tied)
    soft_ties = quantile_landmark_smooth_cdf(tied, landmarks=8, chunk_size=2)
    torch.testing.assert_close(exact_ties[:, 0], torch.full((4,), 0.5, device=device))
    torch.testing.assert_close(soft_ties[:, 0], torch.full((4,), 0.5, device=device))

    collapsed = torch.zeros(1, 24, 2, device=device, requires_grad=True)
    reference = torch.randn(1, 24, 2, device=device)
    collapsed_result = family(collapsed, reference)
    collapsed_gradient = torch.autograd.grad(collapsed_result.scalar_loss, collapsed)[0]
    exact_diagnostic = _cross(collapsed_result).diagnostics[
        "exact_midranks_projected_w2_diagnostic"
    ]
    assert exact_diagnostic.item() > 0.0
    assert torch.isfinite(collapsed_gradient).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_smooth_cvar_cuda_float64_envelope_gradient_matches_finite_difference() -> None:
    values = torch.tensor(
        [0.03, 0.06, 0.08, 0.11, 0.19],
        dtype=torch.float64,
        device="cuda",
        requires_grad=True,
    )
    kwargs = {
        "rho": 0.4,
        "temperature": 0.04,
        "tolerance": 1e-12,
        "max_iterations": 140,
    }
    result = identity_calibrated_smooth_cvar(values, **kwargs)
    gradient = torch.autograd.grad(result.value, values)[0]
    delta = 2e-6
    finite_difference = []
    for index in range(values.numel()):
        plus = values.detach().clone()
        minus = values.detach().clone()
        plus[index] += delta
        minus[index] -= delta
        plus_value = identity_calibrated_smooth_cvar(plus, **kwargs).value
        minus_value = identity_calibrated_smooth_cvar(minus, **kwargs).value
        finite_difference.append((plus_value - minus_value) / (2.0 * delta))
    torch.testing.assert_close(
        gradient,
        torch.stack(finite_difference),
        atol=2e-6,
        rtol=2e-5,
    )
    assert torch.isfinite(gradient).all()


def test_optional_pairwise_diagnostics_report_raw_soft_and_exact_copula_swd() -> None:
    family = _family(_v2_config(pairwise_diagnostics=True))
    generated = torch.randn(1, 24, 2)
    reference = torch.randn(1, 24, 2)
    result = family(generated, reference)
    diagnostics = _cross(result).diagnostics
    assert diagnostics["pair_indices"].shape == (1, 2)
    for name in (
        "pairwise_raw_state_swd",
        "pairwise_soft_copula_swd",
        "pairwise_exact_copula_swd",
    ):
        assert diagnostics[name].shape == (1, 1)
        assert torch.isfinite(diagnostics[name]).all()


def test_legacy_default_and_explicit_legacy_definition_keep_values_and_gradients() -> None:
    fields = ("u", "v")
    legacy_config = {
        "target_use": "paired_supervised",
        "fields": fields,
        "components": {
            "self": {"weight": 1.0},
            "mutual": {"enabled": False, "weight": 0.0},
            "cross": {
                "weight": 0.3,
                "directions": 12,
                "top_fraction": 0.25,
                "seed": 29,
                "include_axes": True,
                "qmc": False,
            },
        },
    }
    explicit_config = {**legacy_config, "definition": "legacy_v1"}
    old = _family(legacy_config)
    explicit = _family(explicit_config)
    torch.testing.assert_close(
        old.components_by_key["cross_joint_topk_swd"].directions,
        explicit.components_by_key["cross_joint_topk_swd"].directions,
    )
    generated = torch.randn(2, 17, 2, requires_grad=True)
    reference = torch.randn(2, 17, 2)
    old_result = old(generated, reference)
    old_gradient = torch.autograd.grad(old_result.scalar_loss, generated, retain_graph=True)[0]
    explicit_result = explicit(generated, reference)
    explicit_gradient = torch.autograd.grad(explicit_result.scalar_loss, generated)[0]
    torch.testing.assert_close(old_result.scalar_loss, explicit_result.scalar_loss, atol=0.0, rtol=0.0)
    torch.testing.assert_close(old_gradient, explicit_gradient, atol=0.0, rtol=0.0)

    directions = old.components_by_key["cross_joint_topk_swd"].directions
    oracle_samples = []
    for generated_item, reference_item in zip(generated, reference):
        generated_marginals = generated_item.sort(dim=0).values
        reference_marginals = reference_item.sort(dim=0).values
        marginal_cost = (generated_marginals - reference_marginals).square().mean()
        generated_projection = generated_item @ directions.T
        reference_projection = reference_item @ directions.T
        projected_costs = (
            generated_projection.sort(dim=0).values
            - reference_projection.sort(dim=0).values
        ).square().mean(dim=0)
        legacy_top_count = math.ceil(0.25 * projected_costs.numel())
        oracle_samples.append(
            marginal_cost + 0.3 * projected_costs.topk(legacy_top_count).values.mean()
        )
    oracle_loss = torch.stack(oracle_samples).mean()
    oracle_gradient = torch.autograd.grad(oracle_loss, generated)[0]
    torch.testing.assert_close(old_result.scalar_loss, oracle_loss, atol=1e-7, rtol=2e-7)
    torch.testing.assert_close(old_gradient, oracle_gradient, atol=1e-7, rtol=2e-7)


def test_definition_versions_keep_legacy_artifacts_and_reject_cross_definition_identity() -> None:
    legacy = _family({"fields": ("u", "v")})
    v2 = _family(_v2_config())
    assert legacy.version == "1"
    assert v2.version == "2"

    legacy_artifact = legacy.state_artifact()
    legacy.load_state_artifact(legacy_artifact)
    wrong_version = v2.state_artifact()
    wrong_version["version"] = "1"
    with pytest.raises(ValueError, match="artifact identity mismatch"):
        v2.load_state_artifact(wrong_version)
