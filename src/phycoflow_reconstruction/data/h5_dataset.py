"""Lazy canonical HDF5 loader for snapshots and complete space-time states.

The loader respects stored trajectory splits and uses chronological 80/10/10
frames only for a single long trajectory without canonical split datasets.
It never writes statistics beside a shared HDF5 payload.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ..contracts import DataSpec, FieldSample
from .normalization import FieldNormalizer
from .splits import SplitSelection, resolve_split


def _json_attr(handle: h5py.File, name: str, default: Any) -> Any:
    value = handle.attrs.get(name)
    if value is None:
        return default
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _field_names(handle: h5py.File, override: Sequence[str] | None) -> tuple[str, ...]:
    channels = int(handle["fields"].shape[-1])
    if override is not None:
        names = tuple(str(v) for v in override)
    else:
        names = tuple(_json_attr(handle, "field_names", []))
        if not names:
            selected = handle["fields"].attrs.get("selected_fields", "")
            if isinstance(selected, bytes):
                selected = selected.decode("utf-8")
            names = tuple(part.strip() for part in str(selected).split(",") if part.strip())
    if len(names) != channels:
        raise ValueError(
            f"field names {names} do not match C={channels}; provide an explicit mapping"
        )
    return names


def _normalize_coordinates(raw: torch.Tensor) -> torch.Tensor:
    low = raw.amin(dim=0)
    high = raw.amax(dim=0)
    return (raw - low) / (high - low).clamp_min(1e-8)


class H5FieldDataset(Dataset[FieldSample]):
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
        coordinate_dim: int | None = None,
        grid_shape: Sequence[int] | None = None,
        coordinate_reorder: str = "stored",
        include_temporal_derivative: bool = False,
    ) -> None:
        self.path = Path(path).resolve()
        self.split_name = split
        self.reconstruction_unit = reconstruction_unit
        self.time_stride = int(time_stride)
        self.include_temporal_derivative = bool(include_temporal_derivative)
        self.coordinate_reorder = str(coordinate_reorder)
        self._h5: h5py.File | None = None

        with h5py.File(self.path, "r") as handle:
            shape = tuple(int(v) for v in handle["fields"].shape)
            if len(shape) != 6 or any(size < 1 for size in shape):
                raise ValueError("HDF5 fields must use non-empty canonical [B,T,Nx,Ny,Nz,C] layout")
            self.batch_count, self.time_count, self.channel_count = shape[0], shape[1], shape[-1]
            if handle["time"].shape != (self.time_count,):
                raise ValueError("HDF5 time does not align with the fields time axis")
            if handle["conditions"].ndim != 2 or handle["conditions"].shape[0] != self.batch_count:
                raise ValueError("HDF5 conditions must use [B,K] layout")
            coordinate_shape = handle["coordinates"].shape
            if (
                len(coordinate_shape) < 2
                or coordinate_shape[-1] != 3
                or int(np.prod(coordinate_shape[:-1])) != int(np.prod(shape[2:5]))
            ):
                raise ValueError("HDF5 coordinates must align with stored spatial points and x,y,z")
            self.field_names = _field_names(handle, field_names)
            raw_units = tuple(_json_attr(handle, "field_units", []))
            self.field_units = tuple(field_units or raw_units or ("unknown",) * self.channel_count)
            self.selection: SplitSelection = resolve_split(handle, split, self.time_stride)
            stored_grid_shape = tuple(int(v) for v in _json_attr(handle, "grid_shape", []))
            if not stored_grid_shape:
                stored_grid_shape = tuple(v for v in shape[2:5] if v > 1) or (shape[2],)
            self.grid_shape = tuple(int(v) for v in grid_shape) if grid_shape else stored_grid_shape
            if int(np.prod(self.grid_shape)) != shape[2] * shape[3] * shape[4]:
                raise ValueError("configured grid_shape does not match the stored point count")
            self.case_name = str(_json_attr(handle, "case_name", self.path.stem))
            stored_coords = handle["coordinates"][:].reshape(-1, 3).astype(np.float32)
            self.point_permutation = self._coordinate_permutation(stored_coords)
            self.raw_spatial_coords = torch.from_numpy(stored_coords)[self.point_permutation]
            self.spatial_coords = _normalize_coordinates(self.raw_spatial_coords)
            self.times = torch.from_numpy(handle["time"][:].astype(np.float32))
            self.conditions = torch.from_numpy(handle["conditions"][:].astype(np.float32))
            self.trajectory_ids = self._read_trajectory_ids(handle)
            if len(self.trajectory_ids) != self.batch_count:
                raise ValueError("trajectory_id must contain one ID per trajectory")
            if len(set(self.trajectory_ids)) != len(self.trajectory_ids):
                raise ValueError("trajectory_id values must be unique")
            self.normalizer = normalizer or FieldNormalizer.from_h5(handle, normalization)
            self.dataset_metadata = self._read_metadata(handle)

        if self.normalizer is None:
            if normalization == "none":
                self.normalizer = FieldNormalizer.identity(self.channel_count)
            else:
                raise ValueError(
                    f"{self.path.name} has no embedded training statistics; pass a verified normalizer "
                    "or use normalization='none' for structural checks"
                )

        if reconstruction_unit == "snapshot":
            self._items = [
                (b, t)
                for b in self.selection.trajectory_indices
                for t in self.selection.frame_indices
            ]
            logical_shape = self.grid_shape
            inferred_coordinate_dim = max(
                1,
                int((self.spatial_coords.amax(0) > 0).sum()),
            )
            if coordinate_dim is not None and not inferred_coordinate_dim <= coordinate_dim <= 3:
                raise ValueError(
                    "coordinate_dim override cannot discard an active axis or exceed stored x,y,z"
                )
            coordinate_dim = (
                inferred_coordinate_dim if coordinate_dim is None else int(coordinate_dim)
            )
        elif reconstruction_unit == "space_time_trajectory":
            self._items = [(b, None) for b in self.selection.trajectory_indices]
            logical_shape = (self.time_count, *self.grid_shape)
            coordinate_dim = 2
            spatial_x = self.raw_spatial_coords[:, 0]
            time_grid, space_grid = torch.meshgrid(self.times, spatial_x, indexing="ij")
            self.space_time_coords_raw = torch.stack(
                (time_grid, space_grid), dim=-1
            ).reshape(-1, 2)
            self.space_time_coords = _normalize_coordinates(self.space_time_coords_raw)
        else:
            raise ValueError(f"unsupported reconstruction_unit={reconstruction_unit!r}")

        self.data_spec = DataSpec(
            field_names=self.field_names,
            field_units=self.field_units,
            coordinate_dim=coordinate_dim,
            logical_shape=logical_shape,
            reconstruction_unit=reconstruction_unit,
            mesh_type="structured"
            if len(self.grid_shape) > 1 or self.case_name == "ks"
            else "point",
        )
        self.data_spec.validate()

    def _coordinate_permutation(self, coordinates: np.ndarray) -> torch.Tensor:
        if self.coordinate_reorder == "stored":
            return torch.arange(coordinates.shape[0])
        if self.coordinate_reorder != "lexicographic_yx":
            raise ValueError("coordinate_reorder must be stored or lexicographic_yx")
        if len(self.grid_shape) != 2:
            raise ValueError("lexicographic_yx requires a two-dimensional logical grid")
        x_values = np.unique(coordinates[:, 0])
        y_values = np.unique(coordinates[:, 1])
        if (y_values.size, x_values.size) != self.grid_shape:
            raise ValueError("coordinate Cartesian topology does not match grid_shape=(Ny,Nx)")
        pairs = np.unique(coordinates[:, :2], axis=0)
        if pairs.shape[0] != coordinates.shape[0]:
            raise ValueError("coordinate Cartesian topology contains duplicate points")
        permutation = np.lexsort((coordinates[:, 0], coordinates[:, 1]))
        return torch.from_numpy(permutation.astype(np.int64))

    def _read_trajectory_ids(self, handle: h5py.File) -> tuple[str, ...]:
        if "trajectory_id" not in handle:
            return tuple(f"trajectory_{index:06d}" for index in range(self.batch_count))
        values = handle["trajectory_id"][:]
        return tuple(v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values)

    def _handle(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    @staticmethod
    def _read_metadata(handle: h5py.File) -> dict[str, Any]:
        if "metadata/json" not in handle:
            return {}
        raw = handle["metadata/json"][()]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            value = json.loads(str(raw))
        except json.JSONDecodeError:
            return {"raw": str(raw)}
        return value if isinstance(value, dict) else {"value": value}

    def _sample_context(self, trajectory_index: int, time_index: int | None) -> dict[str, Any]:
        """Lazily expose optional auxiliary and diagnostic arrays for one item."""
        handle = self._handle()
        context: dict[str, Any] = {"dataset_metadata": self.dataset_metadata}
        for group_name in ("auxiliary", "diagnostics"):
            if group_name not in handle:
                continue
            group_values: dict[str, torch.Tensor] = {}
            for name, dataset in handle[group_name].items():
                if not isinstance(dataset, h5py.Dataset) or not np.issubdtype(
                    dataset.dtype, np.number
                ):
                    continue
                selection: Any = (...,)
                if dataset.ndim and dataset.shape[0] == self.batch_count:
                    if (
                        time_index is not None
                        and dataset.ndim > 1
                        and dataset.shape[1] == self.time_count
                    ):
                        selection = (trajectory_index, time_index)
                    else:
                        selection = trajectory_index
                value = torch.from_numpy(np.asarray(dataset[selection])).float()
                if value.ndim and value.shape[0] == self.point_permutation.numel():
                    value = value[self.point_permutation]
                group_values[name] = value
            context[group_name] = group_values
        return context

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> FieldSample:
        trajectory_index, time_index = self._items[index]
        if time_index is None:
            sample = self._space_time_sample(trajectory_index)
        else:
            sample = self._snapshot_sample(trajectory_index, time_index)
        sample.validate()
        return sample

    def _snapshot_sample(self, trajectory_index: int, time_index: int) -> FieldSample:
        handle = self._handle()
        values = torch.from_numpy(
            handle["fields"][trajectory_index, time_index]
            .reshape(-1, self.channel_count)
            .astype(np.float32)
        )[self.point_permutation]
        coords_raw = self.raw_spatial_coords
        active = max(1, self.data_spec.coordinate_dim)
        coords_raw = coords_raw[:, :active]
        metadata: dict[str, Any] = {
            "trajectory_index": trajectory_index,
            **self._sample_context(trajectory_index, time_index),
        }
        if self.include_temporal_derivative:
            if self.selection.strategy == "chronological_frames_80_10_10":
                first_allowed = self.selection.frame_indices[0]
                last_allowed = self.selection.frame_indices[-1]
            else:
                first_allowed = 0
                last_allowed = self.time_count - 1
            left = max(first_allowed, time_index - 1)
            right = min(last_allowed, time_index + 1)
            if left == right:
                raise ValueError("temporal derivative requires at least two stored frames")
            left_values = torch.from_numpy(
                handle["fields"][trajectory_index, left]
                .reshape(-1, self.channel_count)
                .astype(np.float32)
            )[self.point_permutation]
            right_values = torch.from_numpy(
                handle["fields"][trajectory_index, right]
                .reshape(-1, self.channel_count)
                .astype(np.float32)
            )[self.point_permutation]
            delta_time = float(self.times[right] - self.times[left])
            if not np.isfinite(delta_time) or delta_time <= 0:
                raise ValueError(
                    "temporal derivative requires strictly increasing time within the selected split"
                )
            metadata["physics"] = {
                "temporal_derivative": (right_values - left_values) / delta_time,
                "temporal_derivative_source": "paired_finite_difference",
            }
        return FieldSample(
            values=self.normalizer.encode(values),
            coordinates=self.spatial_coords[:, :active],
            coordinates_raw=coords_raw,
            time=self.times[time_index].clone(),
            trajectory_id=self.trajectory_ids[trajectory_index],
            time_index=time_index,
            conditions=self.conditions[trajectory_index].clone(),
            field_names=self.field_names,
            logical_shape=self.grid_shape,
            reconstruction_unit="snapshot",
            metadata=metadata,
        )

    def _space_time_sample(self, trajectory_index: int) -> FieldSample:
        handle = self._handle()
        values = torch.from_numpy(
            handle["fields"][trajectory_index]
            .reshape(self.time_count, -1, self.channel_count)
            .astype(np.float32)
        )[:, self.point_permutation].reshape(-1, self.channel_count)
        return FieldSample(
            values=self.normalizer.encode(values),
            coordinates=self.space_time_coords,
            coordinates_raw=self.space_time_coords_raw,
            time=self.times.clone(),
            trajectory_id=self.trajectory_ids[trajectory_index],
            time_index=None,
            conditions=self.conditions[trajectory_index].clone(),
            field_names=self.field_names,
            logical_shape=(self.time_count, self.raw_spatial_coords.shape[0]),
            reconstruction_unit="space_time_trajectory",
            metadata={
                "trajectory_index": trajectory_index,
                **self._sample_context(trajectory_index, None),
            },
        )

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __del__(self) -> None:
        self.close()
