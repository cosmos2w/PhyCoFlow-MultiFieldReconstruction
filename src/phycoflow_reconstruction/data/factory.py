"""Construct a canonical field dataset from a resolved dataset config.

Keeping format dispatch here prevents trainers and evaluators from silently
drifting apart. HDF5 remains preferred; trusted tensor-only PT mappings use the
same sample, split, normalization, and lifecycle surface.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

from .h5_dataset import H5FieldDataset
from .manifest import dataset_fingerprint
from .normalization import FieldNormalizer
from .pt_dataset import PTFieldDataset

FieldDataset: TypeAlias = H5FieldDataset | PTFieldDataset


def open_field_dataset(
    config: Mapping[str, Any],
    *,
    split: str | None = None,
    normalizer: FieldNormalizer | None = None,
) -> FieldDataset:
    """Open one supported payload using the common dataset config vocabulary."""
    path = config["path"]
    statistics_path = config.get("normalization_stats_path")
    if normalizer is None and statistics_path:
        configured_fields = config.get("field_names")
        if not configured_fields:
            raise ValueError("dataset.field_names is required with normalization_stats_path")
        normalizer = FieldNormalizer.from_artifact(
            statistics_path,
            field_names=tuple(str(name) for name in configured_fields),
            dataset_fingerprint=dataset_fingerprint(path),
        )
        if normalizer.method != config.get("normalization"):
            raise ValueError("normalization statistics method disagrees with dataset config")
    common = {
        "split": split or config.get("split", "train"),
        "reconstruction_unit": config.get("reconstruction_unit", "snapshot"),
        "field_names": config.get("field_names"),
        "field_units": config.get("field_units"),
        "time_stride": int(config.get("time_stride", 1)),
        "normalization": config.get("normalization", "auto"),
        "normalizer": normalizer,
    }
    suffix = str(path).lower()
    if suffix.endswith((".h5", ".hdf5")):
        return H5FieldDataset(
            path,
            **common,
            coordinate_dim=config.get("coordinate_dim"),
            grid_shape=config.get("grid_shape"),
            coordinate_reorder=config.get("coordinate_reorder", "stored"),
            include_temporal_derivative=bool(config.get("include_temporal_derivative", False)),
        )
    if suffix.endswith((".pt", ".pth")):
        if config.get("coordinate_reorder", "stored") != "stored":
            raise ValueError("PT datasets do not support coordinate_reorder")
        if bool(config.get("include_temporal_derivative", False)):
            raise ValueError(
                "PT datasets do not yet expose paired temporal derivatives for physics training"
            )
        dataset = PTFieldDataset(path, **common)
        expected_shape = config.get("grid_shape")
        if (
            expected_shape is not None
            and tuple(int(v) for v in expected_shape) != dataset.grid_shape
        ):
            raise ValueError("configured grid_shape disagrees with the trusted PT payload")
        coordinate_dim = config.get("coordinate_dim")
        if coordinate_dim is not None and int(coordinate_dim) != dataset.data_spec.coordinate_dim:
            raise ValueError("configured coordinate_dim disagrees with the trusted PT payload")
        return dataset
    raise ValueError(f"unsupported dataset extension for {path!r}")
