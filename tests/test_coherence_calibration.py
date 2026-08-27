"""Unit tests for immutable source-checkpoint family calibration."""

from __future__ import annotations

import pytest
import torch

from phycoflow_reconstruction.training.coherence_calibration import (
    gradient_cosine,
    gradient_diagnostics,
    loss_gradients,
    resolve_family_scales,
)


def test_initial_grad_norm_scales_use_median_aggregate_and_reference():
    norms, scales = resolve_family_scales(
        [
            {"a": 1.0, "b": 4.0, "c": 16.0},
            {"a": 3.0, "b": 8.0, "c": 32.0},
        ],
        {
            "reference": "median",
            "epsilon": 1.0e-12,
            "scale_min": 0.01,
            "scale_max": 100.0,
        },
    )
    assert norms == {"a": 2.0, "b": 6.0, "c": 24.0}
    assert scales == pytest.approx({"a": 3.0, "b": 1.0, "c": 0.25})


@pytest.mark.parametrize(
    "records",
    [
        [{"a": 0.0}],
        [{"a": float("nan")}],
        [{"a": 1.0}, {"a": 101.0}],
    ],
)
def test_calibration_fails_zero_nonfinite_or_unstable_gradients(records):
    with pytest.raises(FloatingPointError):
        resolve_family_scales(records, {"max_batch_ratio": 100.0})


def test_gradient_diagnostics_preserve_sparse_parameter_alignment():
    left = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    right = torch.nn.Parameter(torch.tensor([3.0]))
    parameters = (left, right)
    data = loss_gradients((left.square()).sum(), parameters)
    family_a = loss_gradients(2.0 * left.sum() + right.sum(), parameters)
    family_b = loss_gradients(-left.sum() + 3.0 * right.sum(), parameters)

    report = gradient_diagnostics(
        data,
        {"a": family_a, "b": family_b},
        epsilon=1.0e-12,
    )

    assert report["native_data_gradient_norm"] == pytest.approx((20.0) ** 0.5)
    assert report["family_family_cosines"]["a"]["a"] == 1.0
    expected_cosine = -1.0 / (3.0 * (11.0**0.5))
    assert report["family_family_cosines"]["a"]["b"] == pytest.approx(expected_cosine)
    assert report["data_family_cosines"]["a"] > 0
    assert gradient_cosine(family_a, family_b, epsilon=1.0e-12) == pytest.approx(
        expected_cosine
    )
