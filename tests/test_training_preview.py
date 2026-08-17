"""Unit checks for quantitative training-preview annotations."""

import numpy as np

from phycoflow_reconstruction.training.preview import (
    _absolute_error_title,
    _relative_l2_error,
)


def test_relative_l2_error_uses_field_reference_norm():
    truth = np.asarray([3.0, 4.0])
    estimate = np.asarray([0.0, 0.0])

    value = _relative_l2_error(estimate, truth)

    assert value == 1.0
    assert _absolute_error_title(value) == "Absolute error\nRelative $L_2$ = 1.000e+00"


def test_relative_l2_error_marks_zero_reference_as_unavailable():
    value = _relative_l2_error(np.ones(4), np.zeros(4))

    assert value is None
    assert _absolute_error_title(value) == "Absolute error\nRelative $L_2$ = N/A"
