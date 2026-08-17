"""Training samplers preserve batch identity and deterministic replay."""

import pytest
import torch

from phycoflow_reconstruction.training.common import sample_unique_batch_indices


def test_random_batches_have_unique_indices_and_are_reproducible():
    first = sample_unique_batch_indices(
        7,
        100,
        5,
        generator=torch.Generator().manual_seed(23),
    )
    second = sample_unique_batch_indices(
        7,
        100,
        5,
        generator=torch.Generator().manual_seed(23),
    )

    assert first == second
    assert all(len(batch) == len(set(batch)) for batch in first)
    assert all(0 <= index < 7 for batch in first for index in batch)


def test_batch_cannot_exceed_dataset_size():
    with pytest.raises(ValueError, match="batch_size"):
        sample_unique_batch_indices(
            2,
            1,
            3,
            generator=torch.Generator().manual_seed(23),
        )
