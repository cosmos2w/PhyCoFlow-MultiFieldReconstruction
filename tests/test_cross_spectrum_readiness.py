"""Readiness regression tests for canonical graph cross-spectrum v2."""

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
    graph_fourier,
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


def test_self_spectrum_omission_defaults_to_enabled_unit_weight() -> None:
    config = _config(same_weight=0.0, band_weight=1.0)
    config["components"].pop("self_spectrum")
    family = _family(config)
    assert family.component_weights["self_spectrum"] == 1.0

    coordinates = _coordinates().expand(2, -1, -1)
    result = family(
        torch.randn(2, 16, 2), torch.randn(2, 16, 2), coordinates=coordinates
    )
    assert "cross_spectrum.self_spectrum.auto_spectrum" in result.component_results

    disabled = _family(_config(same_weight=1.0, band_weight=0.0))
    disabled_result = disabled(
        torch.randn(2, 16, 2), torch.randn(2, 16, 2), coordinates=coordinates
    )
    path = "cross_spectrum.self_spectrum.auto_spectrum"
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
