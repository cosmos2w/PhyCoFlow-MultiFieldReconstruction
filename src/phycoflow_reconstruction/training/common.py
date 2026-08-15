"""Small, stage-independent helpers shared by training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..contracts import FieldSample
from ..data.sensor_protocols import SensorProtocol


def collate_field_samples(samples: list[FieldSample]) -> list[FieldSample]:
    """Keep typed samples intact instead of applying PyTorch's tensor collation."""
    return samples


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
