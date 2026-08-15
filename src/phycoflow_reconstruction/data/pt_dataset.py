"""Trusted plain-mapping `.pt` loader for the canonical dense data contract.

PT support intentionally accepts tensors and basic containers only. Its field
layout, split rules, reconstruction units, and returned samples mirror the HDF5
loader; custom pickled Dataset objects are never instantiated.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch.utils.data import Dataset

from ..contracts import DataSpec, FieldSample
from .normalization import FieldNormalizer
from .splits import SplitSelection, chronological_frame_indices, normalize_split


def _trusted_load(path: str | Path) -> dict[str, Any]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("trusted PT dataset must contain a plain mapping")
    return payload


def _minmax(values: torch.Tensor) -> torch.Tensor:
    return (values - values.amin(0)) / (values.amax(0) - values.amin(0)).clamp_min(1e-8)


class PTFieldDataset(Dataset[FieldSample]):
    REQUIRED: ClassVar[frozenset[str]] = frozenset(
        {"fields", "coordinates", "time", "conditions", "field_names"}
    )

    def __init__(
        self,
        path: str | Path,
        *,
        split: str = "train",
        reconstruction_unit: str = "snapshot",
        field_names: Sequence[str] | None = None,
        field_units: Sequence[str] | None = None,
        time_stride: int = 1,
        normalization: str = "auto",
        normalizer: FieldNormalizer | None = None,
    ) -> None:
        self.path = Path(path).resolve()
        payload = _trusted_load(self.path)
        missing = sorted(self.REQUIRED - payload.keys())
        if missing:
            raise ValueError(f"trusted PT dataset is missing keys {missing}")
        tensor_keys = ("fields", "coordinates", "time", "conditions")
        if not all(isinstance(payload[key], torch.Tensor) for key in tensor_keys):
            raise TypeError("PT fields, coordinates, time, and conditions must be tensors")

        self.payload = payload
        self.fields = payload["fields"].float()
        if self.fields.ndim != 6:
            raise ValueError("PT fields must use canonical [B,T,Nx,Ny,Nz,C] layout")
        coordinates = payload["coordinates"]
        if coordinates.ndim < 2 or coordinates.shape[-1] < 1:
            raise ValueError("PT coordinates must have shape [...,D] with D>=1")
        self.coordinates_raw = coordinates.float().reshape(-1, coordinates.shape[-1])
        self.times = payload["time"].float().flatten()
        self.conditions = payload["conditions"].float()
        self.batch_count, self.time_count, _, _, _, channels = self.fields.shape
        if self.times.shape != (self.time_count,):
            raise ValueError("PT time does not align with fields")
        if self.conditions.ndim != 2 or self.conditions.shape[0] != self.batch_count:
            raise ValueError("PT conditions must use [B,K] layout")
        if self.times.numel() > 1 and torch.any(torch.diff(self.times) <= 0):
            raise ValueError("PT time must be strictly increasing")

        names = tuple(str(value) for value in (field_names or payload["field_names"]))
        if len(names) != channels:
            raise ValueError("PT field_names do not align with channels")
        self.field_names = names
        unit_values = field_units or payload.get("field_units", ["unknown"] * channels)
        self.field_units = tuple(str(value) for value in unit_values)
        if len(self.field_units) != channels:
            raise ValueError("PT field_units do not align with channels")
        default_shape = self.fields.shape[2:5]
        self.grid_shape = tuple(
            int(value) for value in payload.get("grid_shape", default_shape) if int(value) > 1
        ) or (int(self.fields.shape[2]),)
        if math.prod(self.grid_shape) != math.prod(self.fields.shape[2:5]):
            raise ValueError("PT grid_shape does not match the stored point count")
        ids = payload.get("trajectory_id", range(self.batch_count))
        self.trajectory_ids = tuple(str(value) for value in ids)
        if len(self.trajectory_ids) != self.batch_count:
            raise ValueError("PT trajectory_id must contain one ID per trajectory")
        if len(set(self.trajectory_ids)) != len(self.trajectory_ids):
            raise ValueError("PT trajectory_id values must be unique")

        self.split_name = normalize_split(split)
        self.selection = self._resolve_split(time_stride)
        trajectories = self.selection.trajectory_indices
        frames = self.selection.frame_indices
        self.reconstruction_unit = reconstruction_unit
        if reconstruction_unit == "snapshot":
            self._items = [(trajectory, frame) for trajectory in trajectories for frame in frames]
            logical_shape = self.grid_shape
            active = int((_minmax(self.coordinates_raw).amax(0) > 0).sum())
            coordinate_dim = max(1, active)
        elif reconstruction_unit == "space_time_trajectory":
            if len(self.grid_shape) != 1:
                raise ValueError("PT space_time_trajectory currently requires one spatial axis")
            self._items = [(trajectory, None) for trajectory in trajectories]
            logical_shape = (self.time_count, *self.grid_shape)
            coordinate_dim = 2
        else:
            raise ValueError(f"unsupported reconstruction_unit={reconstruction_unit!r}")

        self.normalizer = normalizer or self._normalizer(normalization, channels)
        self.data_spec = DataSpec(
            self.field_names,
            self.field_units,
            coordinate_dim,
            logical_shape,
            reconstruction_unit,
            str(payload.get("mesh_type", "structured")),
        )
        self.data_spec.validate()

    def _resolve_split(self, time_stride: int) -> SplitSelection:
        if time_stride < 1:
            raise ValueError("time_stride must be positive")
        splits = self.payload.get("splits")
        if self.batch_count > 1:
            if not isinstance(splits, Mapping) or self.split_name not in splits:
                raise ValueError("multi-trajectory PT data requires a splits mapping")
            required = {"train", "validation", "test"}
            if not required <= set(splits):
                raise ValueError(
                    "multi-trajectory PT splits must define train, validation, and test"
                )
            all_values = {
                name: tuple(
                    int(value) for value in torch.as_tensor(splits[name]).flatten().tolist()
                )
                for name in sorted(required)
            }
            flattened = [value for values in all_values.values() for value in values]
            if any(value < 0 or value >= self.batch_count for value in flattened):
                raise ValueError("PT trajectory splits contain out-of-range indices")
            if len(flattened) != len(set(flattened)):
                raise ValueError("PT trajectory splits contain duplicates or overlap")
            if sorted(flattened) != list(range(self.batch_count)):
                raise ValueError("PT trajectory splits must be exhaustive")
            values = torch.as_tensor(splits[self.split_name]).tolist()
            trajectories = tuple(int(value) for value in values)
            return SplitSelection(
                self.split_name,
                trajectories,
                tuple(range(0, self.time_count, time_stride)),
                "stored_trajectory",
            )
        frames = tuple(
            int(value)
            for value in chronological_frame_indices(self.time_count, self.split_name, time_stride)
        )
        return SplitSelection(
            self.split_name,
            (0,),
            frames,
            "chronological_frames_80_10_10",
        )

    def _normalizer(self, method: str, channels: int) -> FieldNormalizer:
        statistics = self.payload.get("statistics", {})
        if (
            method in {"auto", "robust_99"}
            and {"channel_offset", "channel_scale_99"} <= statistics.keys()
        ):
            return FieldNormalizer(
                statistics["channel_offset"], statistics["channel_scale_99"], "robust_99"
            )
        if method in {"auto", "mean_std"} and {"train_mean", "train_std"} <= statistics.keys():
            return FieldNormalizer(statistics["train_mean"], statistics["train_std"], "mean_std")
        if method == "none":
            return FieldNormalizer.identity(channels)
        raise ValueError("PT payload has no compatible training-only normalization statistics")

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> FieldSample:
        trajectory, frame = self._items[index]
        if frame is None:
            return self._space_time_sample(trajectory)
        values = self.fields[trajectory, frame].reshape(-1, self.fields.shape[-1])
        raw = self.coordinates_raw[:, : self.data_spec.coordinate_dim]
        sample = FieldSample(
            values=self.normalizer.encode(values),
            coordinates=_minmax(raw),
            coordinates_raw=raw,
            time=self.times[frame],
            trajectory_id=self.trajectory_ids[trajectory],
            time_index=frame,
            conditions=self.conditions[trajectory],
            field_names=self.field_names,
            logical_shape=self.grid_shape,
            metadata={
                "trajectory_index": trajectory,
                "query_indices": torch.arange(values.shape[0]),
            },
        )
        sample.validate()
        return sample

    def _space_time_sample(self, trajectory: int) -> FieldSample:
        values = self.fields[trajectory].reshape(-1, self.fields.shape[-1])
        spatial = self.coordinates_raw[:, 0]
        time_grid, space_grid = torch.meshgrid(self.times, spatial, indexing="ij")
        raw = torch.stack((time_grid, space_grid), dim=-1).reshape(-1, 2)
        sample = FieldSample(
            values=self.normalizer.encode(values),
            coordinates=_minmax(raw),
            coordinates_raw=raw,
            time=self.times,
            trajectory_id=self.trajectory_ids[trajectory],
            time_index=None,
            conditions=self.conditions[trajectory],
            field_names=self.field_names,
            logical_shape=(self.time_count, self.coordinates_raw.shape[0]),
            reconstruction_unit="space_time_trajectory",
            metadata={
                "trajectory_index": trajectory,
                "query_indices": torch.arange(values.shape[0]),
            },
        )
        sample.validate()
        return sample

    def close(self) -> None:
        """Match the HDF5 dataset lifecycle API; PT payloads hold no open handle."""
