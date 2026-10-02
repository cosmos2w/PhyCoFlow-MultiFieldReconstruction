"""Acceptance tests for opt-in second-order graph-spectral covariance blocks."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from phycoflow_reconstruction.coherence.families.cross_spectrum.basis import (
    build_graph_basis,
    interval_band_ids,
)
from phycoflow_reconstruction.coherence.families.cross_spectrum.covariance_blocks import (
    LinearCoefficientCovarianceAccumulator,
    covariance_band_block,
    covariance_band_energies,
    linear_coefficient_covariance,
    second_order_covariance_block_losses,
)
from phycoflow_reconstruction.coherence.families.cross_spectrum.family import (
    CrossSpectrumFamily,
)
from phycoflow_reconstruction.coherence.families.cross_spectrum.statistics import (
    band_energies,
    graph_fourier,
    normalized_cross_band_coupling,
    spectral_coherence,
)
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer


def _irregular_coordinates(count: int = 24) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1002)
    coordinates = torch.rand(count, 2, generator=generator, dtype=torch.float64)
    coordinates[:, 1] = (coordinates[:, 1] + 0.037 * coordinates[:, 0].square()).clamp(0, 1)
    return coordinates.float()


def _intervals_for_coordinates(coordinates: torch.Tensor, num_modes: int = 6):
    basis = build_graph_basis(
        coordinates,
        k_neighbors=5,
        sigma=None,
        num_modes=num_modes,
        band_names=("low", "high"),
        exclude_zero=True,
    )
    eigenvalues = basis.eigenvalues.detach().cpu().numpy()
    split = int(np.argmax(np.diff(eigenvalues))) + 1
    boundary = float((eigenvalues[split - 1] + eigenvalues[split]) / 2.0)
    return [
        {"name": "low", "lower": float(eigenvalues[0] - 1.0e-4), "upper": boundary},
        {"name": "high", "lower": boundary, "upper": float(eigenvalues[-1] + 1.0e-4)},
    ]


def _config(coordinates: torch.Tensor) -> dict:
    intervals = _intervals_for_coordinates(coordinates)
    return {
        "definition": "second_order_blocks_v4",
        "fields": ["u", "v"],
        "pairs": [["u", "v"]],
        "graph": {
            "k_neighbors": 5,
            "num_modes": 6,
            "exclude_zero": True,
            "bands": [item["name"] for item in intervals],
            "band_intervals": intervals,
            "degeneracy_tolerance": 1.0e-6,
        },
        "minimum_ensemble_size": 32,
        "stabilization": {
            "relative_floor": 1.0e-6,
            "absolute_floor": 1.0e-12,
            "minimum_reference_band_fraction": 1.0e-8,
        },
        "components": {
            "self_spectrum": {"enabled": False, "weight": 0.0},
            "same_frequency": {"enabled": True, "weight": 1.0},
            "cross_frequency": {"enabled": True, "weight": 1.0},
            "band_energy": {"enabled": False, "weight": 0.0},
        },
    }


def _family(coordinates: torch.Tensor, config: dict | None = None) -> CrossSpectrumFamily:
    return CrossSpectrumFamily(
        _config(coordinates) if config is None else config,
        DataSpec(("u", "v"), ("1", "1"), 2, (4, 6)),
        FieldNormalizer.identity(2),
    )


def _coefficients_from_correlated_modes(
    batch: int, correlation: float, seed: int
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    coefficients = torch.randn(batch, 4, 2, generator=generator, dtype=torch.float64)
    z = torch.randn(batch, generator=generator, dtype=torch.float64)
    independent = torch.randn(batch, generator=generator, dtype=torch.float64)
    coefficients[:, 0, 0] = z
    coefficients[:, 2, 1] = correlation * z + (1.0 - correlation**2) ** 0.5 * independent
    return coefficients


def test_gaussian_cross_band_covariance_keeps_sign_and_stays_out_of_same_band() -> None:
    batch = 4096
    reference = _coefficients_from_correlated_modes(batch, 0.8, 12)
    generated = _coefficients_from_correlated_modes(batch, -0.8, 12)
    # Use the same remaining coefficient draws while reversing only the
    # designed cross-band correlation.
    generated[:, 0, 0] = reference[:, 0, 0]
    generated[:, 2, 1] = -0.8 * reference[:, 0, 0] + 0.6 * torch.randn(
        batch, generator=torch.Generator().manual_seed(13), dtype=torch.float64
    )
    band_ids = torch.tensor([0, 0, 1, 1])
    reference_covariance = linear_coefficient_covariance(reference)
    generated_covariance = linear_coefficient_covariance(generated)
    ref_cross = covariance_band_block(reference_covariance, band_ids, 0, 1, 0, 1)[0, 0]
    gen_cross = covariance_band_block(generated_covariance, band_ids, 0, 1, 0, 1)[0, 0]
    assert ref_cross > 0.7
    assert gen_cross < -0.7

    losses = second_order_covariance_block_losses(
        generated, reference, band_ids, ((0, 1),)
    )
    assert float(losses.cross_frequency) > 0.1
    assert float(losses.same_frequency) < float(losses.cross_frequency) / 20.0
    identity = second_order_covariance_block_losses(
        reference, reference, band_ids, ((0, 1),)
    )
    assert float(identity.same_frequency) == pytest.approx(0.0, abs=1.0e-24)
    assert float(identity.cross_frequency) == pytest.approx(0.0, abs=1.0e-24)


def test_stationary_independent_modes_have_small_cross_band_covariance() -> None:
    generator = torch.Generator().manual_seed(150)
    coefficients = torch.randn(20_000, 6, 2, generator=generator, dtype=torch.float64)
    covariance = linear_coefficient_covariance(coefficients)
    band_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    block = covariance_band_block(covariance, band_ids, 0, 1, 0, 1)
    assert float(torch.linalg.vector_norm(block)) < 0.04


def test_nonlinear_energy_dependence_is_outside_the_new_linear_covariance_target() -> None:
    generator = torch.Generator().manual_seed(207)
    z = torch.randn(20_000, generator=generator, dtype=torch.float64)
    coefficients = torch.zeros(20_000, 4, 2, dtype=torch.float64)
    coefficients[:, 0, 0] = z
    coefficients[:, 2, 1] = z.square() - 1.0
    covariance = linear_coefficient_covariance(coefficients)
    assert abs(float(covariance[0, 2, 0, 1])) < 0.04

    energies = band_energies(coefficients, torch.tensor([0, 0, 1, 1]))
    old_energy_coupling = normalized_cross_band_coupling(energies, eps=1.0e-12)
    assert float(old_energy_coupling[0, 1, 0, 1]) > 0.45


def test_common_orthogonal_rotations_within_bands_leave_loss_invariant() -> None:
    generator = torch.Generator().manual_seed(991)
    reference = torch.randn(256, 6, 2, generator=generator, dtype=torch.float64)
    generated = reference + 0.17 * torch.randn(
        256, 6, 2, generator=generator, dtype=torch.float64
    )
    band_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    before = second_order_covariance_block_losses(
        generated, reference, band_ids, ((0, 1),)
    )

    q0, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    q1, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))

    def rotate(coefficients: torch.Tensor) -> torch.Tensor:
        result = coefficients.clone()
        result[:, :3] = torch.einsum("bkc,kr->brc", coefficients[:, :3], q0)
        result[:, 3:] = torch.einsum("bkc,kr->brc", coefficients[:, 3:], q1)
        return result

    after = second_order_covariance_block_losses(
        rotate(generated), rotate(reference), band_ids, ((0, 1),)
    )
    torch.testing.assert_close(after.same_frequency, before.same_frequency, atol=1.0e-10, rtol=1.0e-9)
    torch.testing.assert_close(after.cross_frequency, before.cross_frequency, atol=1.0e-10, rtol=1.0e-9)


def test_complex_covariance_conjugacy_and_field_pair_order() -> None:
    generator = torch.Generator().manual_seed(431)
    coefficients = torch.randn(64, 4, 2, generator=generator, dtype=torch.float64).to(torch.complex128)
    coefficients = coefficients + 1j * torch.randn(
        64, 4, 2, generator=generator, dtype=torch.float64
    )
    covariance = linear_coefficient_covariance(coefficients)
    band_ids = torch.tensor([0, 0, 1, 1])
    forward = covariance_band_block(covariance, band_ids, 0, 1, 0, 1)
    conjugate = covariance_band_block(covariance, band_ids, 1, 0, 1, 0)
    torch.testing.assert_close(conjugate, forward.conj().T)

    generated = coefficients + 0.03 * torch.randn(
        64, 4, 2, generator=generator, dtype=torch.float64
    ).to(torch.complex128)
    ordered = second_order_covariance_block_losses(
        generated, coefficients, band_ids, ((0, 1),)
    )
    reversed_order = second_order_covariance_block_losses(
        generated, coefficients, band_ids, ((1, 0),)
    )
    torch.testing.assert_close(ordered.same_frequency, reversed_order.same_frequency)
    torch.testing.assert_close(ordered.cross_frequency, reversed_order.cross_frequency)


def test_reference_mask_is_fixed_and_generated_collapse_does_not_disable_blocks() -> None:
    generator = torch.Generator().manual_seed(821)
    reference = torch.randn(96, 4, 2, generator=generator, dtype=torch.float64)
    reference[:, 2:, 0] = 0.0  # Field u has no reference power in the high band.
    band_ids = torch.tensor([0, 0, 1, 1])
    collapsed = torch.zeros_like(reference)
    ordinary = torch.randn(96, 4, 2, generator=generator, dtype=torch.float64)
    kwargs = {"minimum_reference_band_fraction": 0.2}
    collapse_result = second_order_covariance_block_losses(
        collapsed, reference, band_ids, ((0, 1),), **kwargs
    )
    ordinary_result = second_order_covariance_block_losses(
        ordinary, reference, band_ids, ((0, 1),), **kwargs
    )
    assert collapse_result.diagnostics["same_frequency_eligible_blocks"] == ordinary_result.diagnostics[
        "same_frequency_eligible_blocks"
    ]
    assert collapse_result.diagnostics["cross_frequency_eligible_blocks"] == ordinary_result.diagnostics[
        "cross_frequency_eligible_blocks"
    ]
    assert collapse_result.diagnostics["same_frequency_skipped_blocks"] > 0
    assert collapse_result.diagnostics["cross_frequency_skipped_blocks"] > 0
    assert float(collapse_result.cross_frequency) > 0
    assert all(
        min(record["reference_energy_fractions"]) >= 0.2
        for record in collapse_result.diagnostics["cross_frequency_blocks"]
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_calibrated_energy_floor_policy_preserves_low_energy_identity(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    generator = torch.Generator().manual_seed(1002)
    calibration_coefficients = torch.randn(
        64, 4, 2, generator=generator, dtype=torch.float64
    ).to(device)
    band_ids = torch.tensor([0, 0, 1, 1], device=device)
    calibration = covariance_band_energies(
        linear_coefficient_covariance(calibration_coefficients), band_ids
    )

    for scale in (1.0, 1.0e-4, 1.0e-6):
        reference = (scale * calibration_coefficients).detach()
        generated = reference.clone().requires_grad_(True)
        symmetric = second_order_covariance_block_losses(
            generated,
            reference,
            band_ids,
            ((0, 1),),
            include_block_values=True,
            energy_floor_policy="symmetric_calibrated",
            calibration_reference_energies=calibration,
            calibration_ensemble_size=64,
        )
        symmetric_total = symmetric.same_frequency + symmetric.cross_frequency
        torch.testing.assert_close(
            symmetric_total, torch.zeros_like(symmetric_total), atol=1.0e-28, rtol=0.0
        )
        symmetric_gradient = torch.autograd.grad(symmetric_total, generated)[0]
        assert torch.isfinite(symmetric_gradient).all()
        torch.testing.assert_close(
            symmetric_gradient, torch.zeros_like(symmetric_gradient), atol=1.0e-24, rtol=0.0
        )
        assert symmetric.diagnostics["energy_floor_policy"] == "symmetric_calibrated"

    low_reference = (1.0e-4 * calibration_coefficients).detach()
    low_generated = low_reference.clone().requires_grad_(True)
    legacy = second_order_covariance_block_losses(
        low_generated,
        low_reference,
        band_ids,
        ((0, 1),),
        include_block_values=True,
        energy_floor_policy="generated_only_legacy",
        calibration_reference_energies=calibration,
        calibration_ensemble_size=64,
    )
    legacy_total = legacy.same_frequency + legacy.cross_frequency
    assert float(legacy_total) > 0.0
    assert legacy.diagnostics["energy_floor_policy"] == "generated_only_legacy"
    record = legacy.diagnostics["same_frequency_blocks"][0]
    generated_block = torch.tensor(record["generated_normalized_block"], dtype=torch.float64)
    reference_block = torch.tensor(record["reference_normalized_block"], dtype=torch.float64)
    ratio = torch.linalg.vector_norm(generated_block) / torch.linalg.vector_norm(reference_block)
    assert float(ratio) == pytest.approx(0.505, abs=0.01)


def test_v4_energy_floor_policy_is_versioned_and_artifact_bound() -> None:
    coordinates = _irregular_coordinates()
    legacy = _family(coordinates)
    assert legacy.version == "4"
    assert legacy.energy_floor_policy == "generated_only_legacy"
    legacy._basis(coordinates.unsqueeze(0), torch.float32)
    old_artifact = legacy.state_artifact()
    assert old_artifact["energy_floor_policy"] == "generated_only_legacy"
    old_artifact.pop("energy_floor_policy")  # saved before policy provenance was added
    _family(coordinates).load_state_artifact(old_artifact)

    symmetric_config = _config(coordinates)
    symmetric_config["stabilization"]["energy_floor_policy"] = "symmetric_calibrated"
    symmetric = _family(coordinates, symmetric_config)
    assert symmetric.version == "4.1"
    assert symmetric.energy_floor_policy == "symmetric_calibrated"
    symmetric._basis(coordinates.unsqueeze(0), torch.float32)
    symmetric_artifact = symmetric.state_artifact()
    assert symmetric_artifact["version"] == "4.1"
    assert symmetric_artifact["energy_floor_policy"] == "symmetric_calibrated"
    with pytest.raises(ValueError, match="identity mismatch"):
        symmetric.load_state_artifact(old_artifact)
    _family(coordinates, symmetric_config).load_state_artifact(symmetric_artifact)


def test_exact_generated_collapse_and_all_low_energy_masks_have_finite_gradients() -> None:
    generator = torch.Generator().manual_seed(901)
    reference = torch.randn(64, 4, 2, generator=generator, dtype=torch.float64)
    band_ids = torch.tensor([0, 0, 1, 1])
    collapsed = torch.zeros_like(reference, requires_grad=True)
    collapse_loss = second_order_covariance_block_losses(
        collapsed, reference, band_ids, ((0, 1),)
    )
    collapse_gradient = torch.autograd.grad(
        collapse_loss.same_frequency + collapse_loss.cross_frequency, collapsed
    )[0]
    assert torch.isfinite(collapse_gradient).all()

    all_low_reference = torch.zeros_like(reference)
    all_low_generated = torch.zeros_like(reference, requires_grad=True)
    low_loss = second_order_covariance_block_losses(
        all_low_generated, all_low_reference, band_ids, ((0, 1),)
    )
    assert low_loss.diagnostics["same_frequency_eligible_blocks"] == 0
    assert low_loss.diagnostics["cross_frequency_eligible_blocks"] == 0
    low_gradient = torch.autograd.grad(
        low_loss.same_frequency + low_loss.cross_frequency, all_low_generated
    )[0]
    assert torch.isfinite(low_gradient).all()


def test_pooled_sufficient_statistics_match_single_batch_covariance() -> None:
    generator = torch.Generator().manual_seed(320)
    coefficients = torch.randn(137, 7, 3, generator=generator, dtype=torch.float64)
    expected = linear_coefficient_covariance(coefficients)
    left = LinearCoefficientCovarianceAccumulator()
    right = LinearCoefficientCovarianceAccumulator()
    left.update(coefficients[:53])
    right.update(coefficients[53:])
    left.merge(right)
    assert left.sample_count == coefficients.shape[0]
    torch.testing.assert_close(left.covariance(), expected, atol=1.0e-12, rtol=1.0e-12)


def test_covariance_block_loss_passes_gradcheck_and_directional_difference() -> None:
    generator = torch.Generator().manual_seed(77)
    generated = torch.randn(8, 4, 2, generator=generator, dtype=torch.float64, requires_grad=True)
    reference = torch.randn(8, 4, 2, generator=generator, dtype=torch.float64)
    band_ids = torch.tensor([0, 0, 1, 1])

    def objective(value: torch.Tensor) -> torch.Tensor:
        losses = second_order_covariance_block_losses(
            value, reference, band_ids, ((0, 1),)
        )
        return losses.same_frequency + losses.cross_frequency

    assert torch.autograd.gradcheck(objective, (generated,), eps=1.0e-6, atol=1.0e-4, rtol=1.0e-3)
    value = objective(generated)
    gradient = torch.autograd.grad(value, generated)[0]
    direction = torch.randn(generated.shape, generator=generator, dtype=torch.float64)
    step = 1.0e-6
    with torch.no_grad():
        finite_difference = (
            objective(generated + step * direction)
            - objective(generated - step * direction)
        ) / (2 * step)
    directional = (gradient * direction).sum()
    torch.testing.assert_close(directional, finite_difference, atol=2.0e-5, rtol=2.0e-4)


def test_no_dense_spatial_square_is_created_in_second_order_forward() -> None:
    point_count = 257
    coordinates = _irregular_coordinates(point_count)
    family = _family(coordinates)
    generator = torch.Generator().manual_seed(77)
    generated = torch.randn(32, point_count, 2, generator=generator)
    reference = torch.randn(32, point_count, 2, generator=generator)
    batched_coordinates = coordinates.expand(32, -1, -1)

    class RejectSpatialSquare(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))

            def inspect(value):
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim == 2
                    and tuple(value.shape) == (point_count, point_count)
                ):
                    raise AssertionError("dense [N,N] tensor allocated by covariance path")
                if isinstance(value, (tuple, list)):
                    for nested in value:
                        inspect(nested)
                if isinstance(value, dict):
                    for nested in value.values():
                        inspect(nested)

            inspect(result)
            return result

    with RejectSpatialSquare():
        result = family(generated, reference, coordinates=batched_coordinates)
    assert torch.isfinite(result.scalar_loss)
    assert family.eigenvectors.shape == (point_count, 6)


def test_point_permutation_preserves_v4_loss_and_artifact_hash_is_checked() -> None:
    coordinates = _irregular_coordinates()
    config = _config(coordinates)
    generator = torch.Generator().manual_seed(470)
    generated = torch.randn(32, coordinates.shape[0], 2, generator=generator)
    reference = torch.randn(32, coordinates.shape[0], 2, generator=generator)
    family = _family(coordinates, config)
    base_result = family(generated, reference, coordinates=coordinates.expand(32, -1, -1))
    assert base_result.diagnostics["definition"] == "second_order_blocks_v4"
    assert base_result.diagnostics["basis_sha256"]
    assert base_result.diagnostics["band_intervals"] == tuple(
        (item["name"], item["lower"], item["upper"]) for item in config["graph"]["band_intervals"]
    )

    permutation = torch.randperm(coordinates.shape[0], generator=generator)
    permuted_family = _family(coordinates, copy.deepcopy(config))
    permuted_result = permuted_family(
        generated[:, permutation],
        reference[:, permutation],
        coordinates=coordinates[permutation].expand(32, -1, -1),
    )
    torch.testing.assert_close(permuted_result.scalar_loss, base_result.scalar_loss, atol=2e-5, rtol=2e-5)

    artifact = family.state_artifact()
    restored = _family(coordinates, copy.deepcopy(config))
    restored.load_state_artifact(artifact)
    restored_result = restored(generated, reference, coordinates=coordinates.expand(32, -1, -1))
    torch.testing.assert_close(restored_result.scalar_loss, base_result.scalar_loss)

    corrupt = copy.deepcopy(artifact)
    corrupt["state_dict"]["band_ids"][0] = 1 - corrupt["state_dict"]["band_ids"][0]
    with pytest.raises(ValueError, match="membership|basis hash"):
        _family(coordinates, copy.deepcopy(config)).load_state_artifact(corrupt)


def test_frozen_reference_calibration_is_idempotent_and_artifact_bound() -> None:
    coordinates = _irregular_coordinates()
    config = _config(coordinates)
    generator = torch.Generator().manual_seed(831)
    reference = torch.randn(32, coordinates.shape[0], 2, generator=generator)
    generated = torch.randn_like(reference)
    batched_coordinates = coordinates.expand(32, -1, -1)
    family = _family(coordinates, config)

    with pytest.raises(ValueError, match="requires frozen reference calibration"):
        family(
            generated,
            reference,
            coordinates=batched_coordinates,
            context={"phase": "train"},
        )
    family.freeze_reference_calibration(reference, batched_coordinates)
    frozen_energies = family.calibrated_band_energies.clone()
    frozen_sha = family.calibration_sha256
    family.freeze_reference_calibration(reference, batched_coordinates)
    torch.testing.assert_close(family.calibrated_band_energies, frozen_energies)
    assert family.calibration_sha256 == frozen_sha

    different_reference = torch.randn_like(reference)
    with pytest.raises(ValueError, match="already frozen"):
        family.freeze_reference_calibration(different_reference, batched_coordinates)
    result = family(
        generated,
        reference,
        coordinates=batched_coordinates,
        context={"phase": "train"},
    )
    assert result.diagnostics["covariance"]["calibration_source"] == "frozen_reference_panel"

    artifact = family.state_artifact()
    restored = _family(coordinates, copy.deepcopy(config))
    restored.load_state_artifact(artifact)
    torch.testing.assert_close(restored.calibrated_band_energies, frozen_energies)
    assert restored.calibration_sha256 == frozen_sha
    corrupted = copy.deepcopy(artifact)
    corrupted["calibrated_band_energies"][0, 0] += 0.5
    with pytest.raises(ValueError, match="calibration hash"):
        _family(coordinates, copy.deepcopy(config)).load_state_artifact(corrupted)


def test_near_degenerate_cluster_cannot_be_split_by_band_interval() -> None:
    eigenvalues = np.asarray([0.10, 0.1999998, 0.2000001, 0.8])
    intervals = (("low", 0.0, 0.2), ("high", 0.2, 1.0))
    with pytest.raises(ValueError, match="near-degenerate"):
        interval_band_ids(eigenvalues, intervals, degeneracy_tolerance=1.0e-6)


def test_v4_uses_affine_fields_and_does_not_optimize_auto_power() -> None:
    coordinates = _irregular_coordinates()
    config = _config(coordinates)
    generator = torch.Generator().manual_seed(702)
    reference = torch.randn(32, coordinates.shape[0], 2, generator=generator)
    generated = reference.clone()
    generated[..., 0] *= 3.0
    family = _family(coordinates, config)
    result = family(generated, reference, coordinates=coordinates.expand(32, -1, -1))
    assert "cross_spectrum.self_spectrum.auto_spectrum" not in result.component_results
    assert "cross_spectrum.band_energy.log_power" not in result.component_results
    assert "cross_spectrum.same_frequency.second_order_covariance_blocks" in result.component_results
    assert "cross_spectrum.cross_frequency.second_order_covariance_blocks" in result.component_results
    assert float(result.scalar_loss) < 1.0e-8

    for key in ("self_spectrum", "band_energy"):
        invalid = copy.deepcopy(config)
        invalid["components"][key] = {"enabled": True, "weight": 1.0}
        with pytest.raises(ValueError, match="does not optimize"):
            _family(coordinates, invalid)


def test_new_definition_is_opt_in_and_legacy_paths_keep_v3_values() -> None:
    coordinates = _irregular_coordinates(16)
    config = {
        "fields": ["u", "v"],
        "pairs": [["u", "v"]],
        "graph": {
            "k_neighbors": 4,
            "num_modes": 6,
            "exclude_zero": True,
            "bands": ["low", "high"],
        },
        "eps": 1.0e-8,
        "components": {
            "self_spectrum": {"enabled": False, "weight": 0.0},
            "same_frequency": {"enabled": True, "weight": 1.0},
            "cross_frequency": {"enabled": True, "weight": 1.0},
            "band_energy": {"enabled": False, "weight": 0.0},
        },
    }
    family = _family(coordinates, config)
    generator = torch.Generator().manual_seed(302)
    generated = torch.randn(4, 16, 2, generator=generator)
    reference = torch.randn(4, 16, 2, generator=generator)
    result = family(generated, reference, coordinates=coordinates.expand(4, -1, -1))
    assert family.version == "3"
    assert result.diagnostics["definition"] == "legacy_v3"
    assert "cross_spectrum.same_frequency.magnitude_squared" in result.component_results
    assert "cross_spectrum.cross_frequency.band_energy_coupling" in result.component_results
    assert "cross_spectrum.same_frequency.second_order_covariance_blocks" not in result.component_results

    coefficients_generated = graph_fourier(generated, family.eigenvectors)
    coefficients_reference = graph_fourier(reference, family.eigenvectors)
    generated_same = spectral_coherence(coefficients_generated, family.eps)
    reference_same = spectral_coherence(coefficients_reference, family.eps)
    expected_same = (generated_same[:, 0, 1] - reference_same[:, 0, 1]).square().mean()
    generated_energy = band_energies(coefficients_generated, family.band_ids)
    reference_energy = band_energies(coefficients_reference, family.band_ids)
    generated_cross = normalized_cross_band_coupling(generated_energy, family.eps)
    reference_cross = normalized_cross_band_coupling(reference_energy, family.eps)
    off_diagonal = ~torch.eye(generated_cross.shape[0], dtype=torch.bool)
    expected_cross = (generated_cross[:, :, 0, 1] - reference_cross[:, :, 0, 1])[
        off_diagonal
    ].square().mean()
    torch.testing.assert_close(
        result.component_results["cross_spectrum.same_frequency.magnitude_squared"].scalar_loss,
        expected_same,
    )
    torch.testing.assert_close(
        result.component_results["cross_spectrum.cross_frequency.band_energy_coupling"].scalar_loss,
        expected_cross,
    )


def test_evaluation_phase_can_export_signed_block_values_as_json_data() -> None:
    coordinates = _irregular_coordinates()
    family = _family(coordinates)
    generator = torch.Generator().manual_seed(640)
    generated = torch.randn(32, coordinates.shape[0], 2, generator=generator)
    reference = torch.randn(32, coordinates.shape[0], 2, generator=generator)
    result = family(
        generated,
        reference,
        coordinates=coordinates.expand(32, -1, -1),
        context={"phase": "audit"},
    )
    diagnostics = result.diagnostics["covariance"]
    assert diagnostics is not None
    json.dumps(diagnostics)
    assert "generated_normalized_block" in diagnostics["cross_frequency_blocks"][0]
    assert "reference_normalized_block" in diagnostics["cross_frequency_blocks"][0]
    assert isinstance(diagnostics["cross_frequency_blocks"][0]["generated_normalized_block"], list)
