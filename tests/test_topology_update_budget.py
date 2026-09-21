"""Reporting boundaries preserve full-pass sampling and exact recovery."""

import math

import pytest

from phycoflow_reconstruction.data.topology_subset import iter_full_pass_indices
from phycoflow_reconstruction.training.update_budget import (
    post_training_steps_per_epoch,
    sample_exposure,
)


@pytest.mark.parametrize("size,batch_size,report_steps", [(11, 3, 2), (19, 4, 3), (12, 4, 1)])
def test_reporting_epochs_preserve_coverage_and_recovery(size, batch_size, report_steps):
    per_pass = math.ceil(size / batch_size)
    config = {
        "optimization": {
            "sampling": "full_pass",
            "batch_size": batch_size,
            "steps_per_epoch": report_steps,
        }
    }
    assert post_training_steps_per_epoch(config, size) == report_steps
    stream = list(iter_full_pass_indices(size, 2 * per_pass, batch_size, seed=59))
    for offset in (0, per_pass):
        assert sorted(i for batch in stream[offset : offset + per_pass] for i in batch) == list(
            range(size)
        )
        assert len(stream[offset + per_pass - 1]) == size - batch_size * (per_pass - 1)
    assert stream[report_steps:] == list(
        iter_full_pass_indices(
            size, 2 * per_pass - report_steps, batch_size, seed=59, start_step=report_steps
        )
    )
    assert sample_exposure(per_pass, size, batch_size) == {
        "samples_seen": size,
        "dataset_passes": 1.0,
        "batches_per_dataset_pass": per_pass,
    }
    assert sample_exposure(per_pass + 1, size, batch_size)["samples_seen"] == size + batch_size
    del config["optimization"]["steps_per_epoch"]
    assert post_training_steps_per_epoch(config, size) == per_pass


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "9"])
def test_invalid_budgets_are_rejected(value):
    with pytest.raises(ValueError, match="positive integer"):
        post_training_steps_per_epoch({"optimization": {"steps_per_epoch": value}}, 11)
