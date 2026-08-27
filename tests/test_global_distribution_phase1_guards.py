"""Regression tests for global-distribution configuration and runtime guards."""

from __future__ import annotations

import pytest
import torch

from phycoflow_reconstruction.coherence import build_coherence_family
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer


def _family(*, fields=("u", "v"), pairs=(("u", "v"),), mutual_weight=1.0):
    names = ("u", "v", "p")
    return build_coherence_family(
        "global_distribution",
        {
            "target_use": "paired_supervised",
            "units": "model_units",
            "fields": fields,
            "components": {
                "self": {"enabled": True, "weight": 1.0},
                "mutual": {
                    "enabled": True,
                    "weight": mutual_weight,
                    "pairs": pairs,
                    "directions": 4,
                },
                "cross": {"enabled": False},
            },
        },
        DataSpec(names, ("1",) * len(names), 2, (8,)),
        FieldNormalizer.identity(len(names)),
    )


@pytest.mark.parametrize(
    ("pairs", "message"),
    [
        ((("u", "u"),), "distinct fields"),
        ((("u", "v"), ("u", "v")), "unique regardless of order"),
        ((("u", "v"), ("v", "u")), "unique regardless of order"),
    ],
)
def test_mutual_pairs_reject_self_duplicate_and_reversed_duplicate(pairs, message):
    with pytest.raises(ValueError, match=message):
        _family(pairs=pairs)


def test_mutual_pairs_must_stay_within_family_declared_fields():
    with pytest.raises(KeyError, match="outside global-distribution fields"):
        _family(fields=("u", "v"), pairs=(("u", "p"),))


def test_zero_weight_component_is_not_executed_and_is_reported():
    family = _family(mutual_weight=0.0)
    mutual = family.components_by_key["mutual_pairwise_swd"]

    def fail_if_called(*args, **kwargs):
        raise AssertionError("zero-weight mutual component executed")

    mutual.forward = fail_if_called
    generated = torch.randn(2, 8, 3, requires_grad=True)
    result = family(generated, torch.randn_like(generated))

    path = "global_distribution.mutual.pairwise_swd"
    assert path not in result.component_results
    assert result.diagnostics["components"][path] == {
        "weight": 0.0,
        "executed": False,
        "raw_scalar_loss": None,
        "weighted_scalar_contribution": None,
    }


def test_component_diagnostics_expose_raw_and_inner_weighted_scalars():
    family = _family(mutual_weight=2.5)
    generated = torch.randn(2, 8, 3)
    result = family(generated, torch.randn_like(generated))
    path = "global_distribution.mutual.pairwise_swd"
    raw = result.component_results[path].scalar_loss
    diagnostics = result.diagnostics["components"][path]

    assert diagnostics["executed"] is True
    assert diagnostics["raw_scalar_loss"] == pytest.approx(float(raw))
    assert diagnostics["weighted_scalar_contribution"] == pytest.approx(float(2.5 * raw))
