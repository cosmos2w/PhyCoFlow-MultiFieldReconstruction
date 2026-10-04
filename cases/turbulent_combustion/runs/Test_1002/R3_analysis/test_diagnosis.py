from types import SimpleNamespace

import numpy as np
import pytest
import torch

from phycoflow_reconstruction.contracts import FamilyResult, TermResult
from phycoflow_reconstruction.training.coherence_diagnostics import (
    cached_B_regrouping,
    component_gradient_diagnostics,
    pressure_error_decomposition,
)


def test_live_components_retain_graph_and_parameter_grad():
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    parameter.grad = torch.tensor([7.0, 8.0])
    components = {
        "global_distribution.self.marginal_w2": TermResult(None, parameter[0] ** 2),
        "global_distribution.cross.copula": TermResult(None, parameter[1] ** 2),
    }
    result = FamilyResult(components, None, parameter.square().sum(), diagnostics={
        "components": {key: {"weight": 1} for key in components}
    })
    payload = component_gradient_diagnostics(
        {"global_distribution": result}, {"global_distribution": SimpleNamespace(family_weight=1)},
        [parameter], abc_direction=torch.tensor([2.0, 0.0]), native_loss=parameter.sum(),
        native_pressure=parameter.sum() * 0,
    )
    assert payload["components"]["native.pressure"]["raw_parameter_gradient_norm"] == 0
    assert payload["components"]["global_distribution.self.marginal_w2"]["raw_parameter_gradient_norm"] == 4
    assert torch.equal(parameter.grad, torch.tensor([7.0, 8.0]))
    result.scalar_loss.backward()
    assert torch.equal(parameter.grad, torch.tensor([11.0, 14.0]))
    with pytest.raises(ValueError, match="TRAIN"):
        component_gradient_diagnostics({}, {}, [parameter], split="test")


def test_pressure_keeps_uncorrected_error():
    error = np.array([[1., 3.], [2., 6.]])
    result = pressure_error_decomposition(error, np.zeros_like(error))
    assert result["uncorrected_mse"] == 12.5
    assert result["mean_error_squared"] == 10
    assert result["centered_error_mse"] == 2.5
    assert result["identity_absolute_error"] == 0


def test_C_category_weights_reconstruct_historical_scalar():
    parameter = torch.nn.Parameter(torch.tensor(2.))
    f, e = parameter ** 2, parameter ** 3
    result = FamilyResult({
        "topology.self.T.h0.finite": TermResult(None, f),
        "topology.self.T.h0.essential": TermResult(None, e),
    }, None, 2 * (f + .1 * e))
    objective = SimpleNamespace(
        source_calibration={"source_means": {"finite": {"self.T.h0": 4}, "essential": {"self.T.h0": 8}},
                            "scales": {"finite": {"self.T.h0": 4}, "essential": {"self.T.h0": 8}}},
        reporting_group_weights=lambda: {"self.T.h0": 1},
        weights={"self.persistence": 1, "mutual.persistence": 1}, essential_weight=.1)
    family = SimpleNamespace(family_weight=1, spatial_objective=objective)
    result = component_gradient_diagnostics({"topology": result}, {"topology": family}, [parameter],
                                           abc_direction=torch.tensor([1.]))["components"]
    assert result["topology.legacy_finite_contribution"]["weighted_value"] == 8
    assert result["topology.legacy_essential_contribution"]["weighted_value"] == pytest.approx(1.6)
    assert result["topology.finite_primary_finite"]["weighted_value"] == pytest.approx(.9)
    assert result["topology.finite_primary_essential"]["weighted_value"] == pytest.approx(.1)
    assert result["topology.self.T.h0.finite"]["raw_parameter_gradient_norm"] == 4


def test_cached_regrouping_is_deterministic_and_identity():
    generator = torch.Generator().manual_seed(2027)
    source = torch.randn(64, 3, 2, generator=generator)
    target = torch.randn(64, 3, 2, generator=generator)
    family = SimpleNamespace(
        band_ids=torch.tensor([0, 1, 1]), pairs=((0, 1),),
        component_weights={"same_frequency": 1, "cross_frequency": 1},
        relative_floor=1e-6, absolute_floor=1e-12, minimum_reference_band_fraction=1e-8,
        energy_floor_policy="symmetric_calibrated", calibrated_band_energies=torch.ones(2, 2),
        calibration_ensemble_size=64)
    first = cached_B_regrouping(source, source, target, family)
    second = cached_B_regrouping(source, source, target, family)
    assert first == second
    assert first["ratio_range"] == [1., 1.]
    assert first["pooled_ratio"] == 1
    with pytest.raises(ValueError, match="eight"):
        cached_B_regrouping(source[:32], source[:32], target[:32], family)
