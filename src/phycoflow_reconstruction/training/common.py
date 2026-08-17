"""Small, stage-independent helpers shared by training and evaluation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import torch

from ..contracts import FieldSample
from ..data.sensor_protocols import SensorProtocol


def collate_field_samples(samples: list[FieldSample]) -> list[FieldSample]:
    """Keep typed samples intact instead of applying PyTorch's tensor collation."""
    return samples


def iter_unique_batch_indices(
    dataset_size: int,
    num_batches: int,
    batch_size: int,
    *,
    generator: torch.Generator,
) -> Iterator[list[int]]:
    """Yield deterministic random batches without materializing the full run.

    Sampling remains independent across optimizer steps. Within one step,
    however, every dataset item is distinct so the observation contract has
    an unambiguous sample ID for each batch row. Keeping this lazy is important:
    a long run can otherwise retain tens of millions of Python integers before
    the first optimizer step.
    """
    if dataset_size < 1:
        raise ValueError("dataset_size must be positive")
    if num_batches < 0:
        raise ValueError("num_batches must be non-negative")
    if not 1 <= batch_size <= dataset_size:
        raise ValueError(
            f"batch_size must lie in [1, dataset_size], got {batch_size} for "
            f"dataset_size={dataset_size}"
        )

    for _ in range(num_batches):
        yield torch.randperm(dataset_size, generator=generator)[:batch_size].tolist()


def sample_unique_batch_indices(
    dataset_size: int,
    num_batches: int,
    batch_size: int,
    *,
    generator: torch.Generator,
) -> list[list[int]]:
    """Materialize the lazy sampler for small callers and compatibility tests."""
    return list(
        iter_unique_batch_indices(
            dataset_size,
            num_batches,
            batch_size,
            generator=generator,
        )
    )


def sensor_protocol_from_config(
    config: Mapping[str, Any], *, seed_offset: int = 0
) -> SensorProtocol:
    """Build the one canonical sensor protocol used by every model stage."""
    observations = config["observations"]
    counts: dict[str, int] = {}
    ranges: dict[str, tuple[int, int]] = {}
    for name, settings in observations.get("fields", {}).items():
        if "count" in settings:
            counts[name] = int(settings["count"])
        else:
            low = int(settings.get("count_min", 1))
            ranges[name] = (low, int(settings.get("count_max", low)))
    return SensorProtocol(
        name=observations.get("protocol", "random_uniform"),
        field_counts=counts or None,
        field_count_ranges=ranges or None,
        spatial_downsample_ratio=int(observations.get("spatial_downsample_ratio", 1)),
        temporal_downsample_ratio=int(observations.get("temporal_downsample_ratio", 1)),
        phase=int(observations.get("phase", 0)),
        seed=int(observations.get("seed", config.get("runtime", {}).get("seed", 42)))
        + int(seed_offset),
        shared_locations=bool(observations.get("shared_locations", False)),
    )
