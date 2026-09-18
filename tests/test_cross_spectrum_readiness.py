"""Readiness regression tests for canonical graph cross-spectrum v3."""

from __future__ import annotations

import json

import pytest
import torch

from phycoflow_reconstruction.coherence.families.cross_spectrum import family as family_module
from phycoflow_reconstruction.coherence.families.cross_spectrum.family import (
    CrossSpectrumFamily,
)
from phycoflow_reconstruction.coherence.families.cross_spectrum.statistics import (
    auto_spectrum,
    auto_spectrum_mean_square,
    auto_spectrum_mean_square_values,
    graph_fourier,
    self_spectrum_coherence,
    self_spectrum_coherence_scores,
)
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer


def _coordinates(size: int = 4) -> torch.Tensor:
    axis = torch.linspace(0.0, 1.0, size)
    return torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1).reshape(-1, 2)


def _config(*, same_weight: float = 1.0, band_weight: float = 0.0) -> dict:
    return {
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
            # Keep the historical readiness fixtures focused on the original
            # same-frequency/band-energy terms; self_spectrum has its own
            # omission/default coverage below.
            "self_spectrum": {"enabled": False, "weight": 0.0},
            "same_frequency": {"enabled": True, "weight": same_weight},
            "cross_frequency": {"enabled": False, "weight": 0.0},
            "band_energy": {"enabled": True, "weight": band_weight},
        },
    }


def _family(config: dict) -> CrossSpectrumFamily:
    return CrossSpectrumFamily(
        config,
        DataSpec(("u", "v"), ("1", "1"), 2, (4, 4)),
        FieldNormalizer.identity(2),
    )


def test_zero_weight_spectral_component_is_not_executed(monkeypatch) -> None:
    family = _family(_config(same_weight=0.0, band_weight=1.0))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("zero-weight same-frequency component executed")

    monkeypatch.setattr(family_module, "spectral_coherence", fail_if_called)
    coordinates = _coordinates().expand(2, -1, -1)
    generated = torch.randn(2, 16, 2)
    result = family(generated, torch.randn_like(generated), coordinates=coordinates)

    path = "cross_spectrum.same_frequency.magnitude_squared"
    assert path not in result.component_results
    assert result.diagnostics["components"][path]["executed"] is False


def test_basis_and_epsilon_provenance_are_json_safe_and_persisted() -> None:
    family = _family(_config())
    coordinates = _coordinates().expand(2, -1, -1)
    generated = torch.randn(2, 16, 2)
    result = family(generated, torch.randn_like(generated), coordinates=coordinates)
    diagnostics = result.diagnostics

    json.dumps(diagnostics)
    assert diagnostics["exclude_zero"] is True
    assert len(diagnostics["eigenvalues"]) == 6
    assert len(diagnostics["band_mode_ids"]) == 6
    assert diagnostics["zero_mode_eigenvalue"] == pytest.approx(0.0, abs=1.0e-5)
    assert diagnostics["first_retained_eigengap"] > 0
    same = result.component_results[
        "cross_spectrum.same_frequency.magnitude_squared"
    ].diagnostics
    assert 0.0 <= same["generated"]["epsilon_dominated_fraction"] <= 1.0
    artifact = family.state_artifact()
    assert artifact["zero_mode_eigenvalue"] == diagnostics["zero_mode_eigenvalue"]
    assert artifact["first_retained_eigengap"] == diagnostics["first_retained_eigengap"]


def test_self_spectrum_matches_modewise_per_field_auto_spectrum_mse() -> None:
    config = _config(same_weight=0.0, band_weight=0.0)
    config["components"]["self_spectrum"] = {"enabled": True, "weight": 2.5}
    config["components"]["cross_frequency"] = {"enabled": False, "weight": 0.0}
    family = _family(config)
    coordinates = _coordinates().expand(3, -1, -1)
    generated = torch.randn(3, 16, 2)
    reference = torch.randn_like(generated)

    result = family(generated, reference, coordinates=coordinates)
    basis = family.eigenvectors.to(generated)
    expected = (
        auto_spectrum(graph_fourier(generated, basis))
        - auto_spectrum(graph_fourier(reference, basis))
    ).square().mean()
    path = "cross_spectrum.self_spectrum.auto_spectrum"
    torch.testing.assert_close(result.component_results[path].scalar_loss, expected)
    torch.testing.assert_close(result.scalar_loss, 2.5 * expected)
    assert result.diagnostics["components"][path]["weight"] == 2.5
    assert result.component_results[path].diagnostics["minimum_batch_size"] == 1


@pytest.mark.parametrize("self_settings", [None, {"weight": 1.0}])
def test_self_spectrum_requires_explicit_enabled_true(
    self_settings: dict | None,
) -> None:
    config = _config(same_weight=0.0, band_weight=1.0)
    if self_settings is None:
        config["components"].pop("self_spectrum")
    else:
        config["components"]["self_spectrum"] = self_settings
    family = _family(config)
    assert "self_spectrum" not in family.component_weights

    coordinates = _coordinates().expand(2, -1, -1)
    result = family(
        torch.randn(2, 16, 2), torch.randn(2, 16, 2), coordinates=coordinates
    )
    path = "cross_spectrum.self_spectrum.auto_spectrum"
    assert path not in result.component_results
    assert result.diagnostics["components"][path]["executed"] is False

    disabled = _family(_config(same_weight=1.0, band_weight=0.0))
    disabled_result = disabled(
        torch.randn(2, 16, 2), torch.randn(2, 16, 2), coordinates=coordinates
    )
    assert path not in disabled_result.component_results
    assert disabled_result.diagnostics["components"][path]["executed"] is False


def test_self_spectrum_can_run_standalone_for_one_field_without_pairs() -> None:
    config = _config(same_weight=0.0, band_weight=0.0)
    config["fields"] = ["u"]
    config["pairs"] = []
    config["components"]["self_spectrum"] = {"enabled": True, "weight": 1.0}
    family = _family(config)
    coordinates = _coordinates().expand(1, -1, -1)

    result = family(
        torch.randn(1, 16, 2), torch.randn(1, 16, 2), coordinates=coordinates
    )

    assert family.required_batch_size == 1
    assert family.pairs == ()
    assert tuple(result.component_results) == (
        "cross_spectrum.self_spectrum.auto_spectrum",
    )


def _spectral_coefficients() -> torch.Tensor:
    """Deterministic [batch, mode, field] coefficients for self-spectrum tests."""
    return torch.tensor(
        [
            [
                [1.0, 2.0],
                [2.0, 1.0],
                [3.0, 4.0],
            ],
            [
                [2.0, 1.0],
                [1.0, 3.0],
                [4.0, 2.0],
            ],
        ],
        dtype=torch.float32,
    )


def test_auto_spectrum_mean_square_matches_per_field_mean() -> None:
    reference = _spectral_coefficients()
    generated = 1.5 * reference

    values = auto_spectrum_mean_square_values(generated, reference)
    aggregate = auto_spectrum_mean_square(generated, reference)

    torch.testing.assert_close(aggregate, values.mean())


def test_self_spectrum_coherence_exact_agreement_is_one() -> None:
    reference = _spectral_coefficients()

    scores = self_spectrum_coherence_scores(
        reference,
        reference,
        eps=1.0e-8,
    )
    aggregate = self_spectrum_coherence(
        reference,
        reference,
        eps=1.0e-8,
    )

    assert scores.shape == (reference.shape[-1],)
    assert aggregate.ndim == 0

    torch.testing.assert_close(scores, torch.ones_like(scores))
    torch.testing.assert_close(
        aggregate,
        torch.tensor(1.0, dtype=aggregate.dtype),
    )


def test_self_spectrum_coherence_has_expected_scaled_power_score() -> None:
    reference = _spectral_coefficients()
    generated = 2.0 * reference

    scores = self_spectrum_coherence_scores(
        generated,
        reference,
        eps=1.0e-8,
    )

    # Scaling coefficients by 2 scales auto-spectrum power by 4:
    #
    #   score = 1 - ||4P - P|| / (||4P|| + ||P||)
    #         = 1 - 3 / 5
    #         = 0.4
    expected = torch.full_like(scores, 0.4)

    torch.testing.assert_close(
        scores,
        expected,
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_self_spectrum_coherence_zero_power_is_finite() -> None:
    reference = torch.zeros((2, 3, 2), dtype=torch.float32)
    generated = torch.zeros_like(reference)

    scores = self_spectrum_coherence_scores(
        generated,
        reference,
        eps=1.0e-8,
    )
    aggregate = self_spectrum_coherence(
        generated,
        reference,
        eps=1.0e-8,
    )

    assert torch.isfinite(scores).all()
    assert torch.isfinite(aggregate)

    torch.testing.assert_close(scores, torch.ones_like(scores))
    torch.testing.assert_close(
        aggregate,
        torch.tensor(1.0, dtype=aggregate.dtype),
    )


def test_self_spectrum_coherence_is_symmetric_and_bounded() -> None:
    reference = _spectral_coefficients()
    generated = reference.clone()

    generated[:, 0, 0] *= 4.0
    generated[:, 1, 1] *= 0.25

    forward = self_spectrum_coherence_scores(
        generated,
        reference,
        eps=1.0e-8,
    )
    reverse = self_spectrum_coherence_scores(
        reference,
        generated,
        eps=1.0e-8,
    )

    torch.testing.assert_close(forward, reverse)

    assert torch.isfinite(forward).all()
    assert torch.all(forward >= 0.0)
    assert torch.all(forward <= 1.0)
    assert torch.any(forward < 1.0)


def test_self_spectrum_coherence_has_finite_gradients() -> None:
    reference = _spectral_coefficients()

    # Keep the statistic away from either clamp boundary so this tests
    # differentiation through the interior of the agreement score.
    generated = (1.5 * reference).clone().requires_grad_(True)

    loss = 1.0 - self_spectrum_coherence(
        generated,
        reference,
        eps=1.0e-8,
    )
    loss.backward()

    assert generated.grad is not None
    assert torch.isfinite(generated.grad).all()
    assert generated.grad.abs().sum() > 0.0


def test_self_spectrum_coherence_rejects_misaligned_coefficients() -> None:
    generated = torch.zeros((2, 3, 2))
    reference = torch.zeros((2, 3, 1))

    with pytest.raises(
        ValueError,
        match=r"self-spectrum coefficients must align as \[B,K,C\]",
    ):
        self_spectrum_coherence_scores(
            generated,
            reference,
            eps=1.0e-8,
        )