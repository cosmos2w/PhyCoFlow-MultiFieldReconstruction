"""Fast structural HDF5 validation with split and metadata diagnostics."""

from __future__ import annotations

import json
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

CORE_DATASETS = ("fields", "coordinates", "time", "conditions")


def validate_h5_dataset(
    path: str | Path, field_names: Sequence[str] | None = None
) -> dict[str, Any]:
    path = Path(path).resolve()
    report: dict[str, Any] = {"path": str(path), "errors": [], "warnings": []}
    if not path.exists():
        report["errors"].append("payload does not exist")
        report["valid"] = False
        return report

    with h5py.File(path, "r") as handle:
        missing = [name for name in CORE_DATASETS if name not in handle]
        if missing:
            report["errors"].append(f"missing datasets: {missing}")
            report["valid"] = False
            return report
        shape = tuple(int(v) for v in handle["fields"].shape)
        report["shape"] = shape
        if len(shape) != 6:
            report["errors"].append(f"fields must be rank 6, got {shape}")
            report["valid"] = False
            return report
        report.update(
            trajectories=shape[0],
            times=shape[1],
            points=int(np.prod(shape[2:5])),
            channels=shape[-1],
        )
        if any(size < 1 for size in shape):
            report["errors"].append("fields axes must all be non-empty")
        if handle["time"].shape != (shape[1],):
            report["errors"].append("time length does not match fields")
        if handle["conditions"].ndim != 2 or handle["conditions"].shape[0] != shape[0]:
            report["errors"].append("conditions do not match trajectory count")
        if handle["coordinates"].ndim < 2 or handle["coordinates"].shape[-1] != 3:
            report["errors"].append("coordinates must end in x,y,z")
        elif int(np.prod(handle["coordinates"].shape[:-1])) != int(np.prod(shape[2:5])):
            report["errors"].append("coordinate point count does not match fields")
        times = handle["time"][:]
        if times.size > 1 and not np.all(np.diff(times) > 0):
            reset_indices = (np.flatnonzero(np.diff(times) <= 0) + 1).astype(int).tolist()
            if not handle.attrs.get("schema_version") and shape[0] == 1:
                report["warnings"].append(
                    f"legacy time coordinate resets at frame indices {reset_indices}; split and load by frame order"
                )
                report["time_reset_indices"] = reset_indices
            else:
                report["errors"].append("time must be strictly increasing")

        if "splits" in handle:
            split_values: dict[str, list[int]] = {}
            for name in ("train", "validation", "test"):
                if f"splits/{name}" not in handle:
                    report["errors"].append(f"missing splits/{name}")
                    continue
                split_values[name] = [int(v) for v in handle[f"splits/{name}"][:]]
                if len(split_values[name]) != len(set(split_values[name])):
                    report["errors"].append(f"splits/{name} contains duplicate trajectory IDs")
                if any(value < 0 or value >= shape[0] for value in split_values[name]):
                    report["errors"].append(f"splits/{name} contains out-of-range trajectory IDs")
            flattened = [v for values in split_values.values() for v in values]
            if len(flattened) != len(set(flattened)):
                report["errors"].append("trajectory splits overlap")
            if shape[0] > 1 and sorted(flattened) != list(range(shape[0])):
                report["errors"].append("trajectory splits are not exhaustive")
            report["splits"] = {key: len(values) for key, values in split_values.items()}
        elif shape[0] == 1:
            train_end, validation_end = int(shape[1] * 0.8), int(shape[1] * 0.9)
            report["splits"] = {
                "train_frames": [0, train_end],
                "validation_frames": [train_end, validation_end],
                "test_frames": [validation_end, shape[1]],
            }
        else:
            report["errors"].append("multi-trajectory data requires stored splits")

        if "statistics" not in handle:
            report["warnings"].append("no embedded training statistics")
        else:
            for name in ("train_mean", "train_std"):
                if f"statistics/{name}" in handle and handle[f"statistics/{name}"].shape != (
                    shape[-1],
                ):
                    report["errors"].append(f"statistics/{name} must contain one value per field")
        if not handle.attrs.get("schema_version"):
            report["warnings"].append("no schema_version attribute")
        if field_names is not None:
            if len(field_names) != shape[-1]:
                report["errors"].append("explicit field_names do not match channel count")
            stored_names = handle.attrs.get("field_names")
            if isinstance(stored_names, bytes):
                stored_names = stored_names.decode()
            if isinstance(stored_names, str):
                try:
                    stored_names = json.loads(stored_names)
                except json.JSONDecodeError:
                    stored_names = [part.strip() for part in stored_names.split(",")]
            names_missing = stored_names is None or (
                isinstance(stored_names, (str, list, tuple)) and len(stored_names) == 0
            )
            if names_missing:
                stored_names = handle["fields"].attrs.get("selected_fields")
                if isinstance(stored_names, bytes):
                    stored_names = stored_names.decode()
                if isinstance(stored_names, str):
                    stored_names = [part.strip() for part in stored_names.split(",")]
            if stored_names is not None and tuple(str(value) for value in stored_names) != tuple(
                field_names
            ):
                report["errors"].append("explicit field_names disagree with stored channel order")

        # Read one scalar from each end without scanning the full payload.
        if all(size > 0 for size in shape):
            first = float(handle["fields"][0, 0, 0, 0, 0, 0])
            last = float(handle["fields"][-1, -1, -1, -1, -1, -1])
            if not np.isfinite([first, last]).all():
                report["errors"].append("sampled endpoint values are non-finite")
            report["sampled_endpoints"] = [first, last]

    report["valid"] = not report["errors"]
    return report


def report_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


def validate_pt_dataset(
    path: str | Path, field_names: Sequence[str] | None = None
) -> dict[str, Any]:
    """Validate the trusted tensor-only PT form without constructing samples."""
    path = Path(path).resolve()
    report: dict[str, Any] = {"path": str(path), "errors": [], "warnings": []}
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as error:
        report.update(valid=False, errors=[f"cannot safely load PT mapping: {error}"])
        return report
    if not isinstance(payload, dict):
        report["errors"].append("PT payload must be a plain mapping")
        report["valid"] = False
        return report
    required = {"fields", "coordinates", "time", "conditions", "field_names"}
    missing = sorted(required - payload.keys())
    if missing:
        report["errors"].append(f"missing keys: {missing}")
        report["valid"] = False
        return report
    tensors = ("fields", "coordinates", "time", "conditions")
    if not all(isinstance(payload[key], torch.Tensor) for key in tensors):
        report["errors"].append("fields, coordinates, time, and conditions must be tensors")
        report["valid"] = False
        return report
    shape = tuple(int(value) for value in payload["fields"].shape)
    report["shape"] = shape
    if len(shape) != 6:
        report["errors"].append(f"fields must be rank 6, got {shape}")
    else:
        if any(size < 1 for size in shape):
            report["errors"].append("fields axes must all be non-empty")
        if payload["time"].ndim != 1 or payload["time"].numel() != shape[1]:
            report["errors"].append("time does not align with fields")
        elif payload["time"].numel() > 1 and torch.any(torch.diff(payload["time"].float()) <= 0):
            report["errors"].append("time must be strictly increasing")
        if payload["conditions"].ndim != 2 or payload["conditions"].shape[0] != shape[0]:
            report["errors"].append("conditions do not align with fields")
        if payload["coordinates"].ndim < 2 or payload["coordinates"].shape[-1] < 1:
            report["errors"].append("coordinates must have shape [...,D] with D>=1")
        elif payload["coordinates"].numel() // payload["coordinates"].shape[-1] != int(
            np.prod(shape[2:5])
        ):
            report["errors"].append("coordinates do not align with spatial points")
        names = tuple(str(value) for value in (field_names or payload["field_names"]))
        if len(names) != shape[-1]:
            report["errors"].append("field_names do not align with channels")
        trajectory_ids = tuple(
            str(value) for value in payload.get("trajectory_id", range(shape[0]))
        )
        if len(trajectory_ids) != shape[0] or len(set(trajectory_ids)) != len(trajectory_ids):
            report["errors"].append("trajectory_id values must be unique and align with fields")
        if shape[0] > 1:
            splits = payload.get("splits")
            if not isinstance(splits, dict):
                report["errors"].append("multi-trajectory PT data requires a splits mapping")
            else:
                required_splits = {"train", "validation", "test"}
                missing_splits = sorted(required_splits - splits.keys())
                if missing_splits:
                    report["errors"].append(f"missing PT splits: {missing_splits}")
                else:
                    split_values = {
                        name: [
                            int(value) for value in torch.as_tensor(splits[name]).flatten().tolist()
                        ]
                        for name in sorted(required_splits)
                    }
                    flattened = [value for values in split_values.values() for value in values]
                    if any(value < 0 or value >= shape[0] for value in flattened):
                        report["errors"].append("PT splits contain out-of-range trajectory IDs")
                    if len(flattened) != len(set(flattened)):
                        report["errors"].append(
                            "PT trajectory splits overlap or contain duplicates"
                        )
                    if sorted(flattened) != list(range(shape[0])):
                        report["errors"].append("PT trajectory splits are not exhaustive")
                    report["splits"] = {name: len(values) for name, values in split_values.items()}
        else:
            train_end, validation_end = int(shape[1] * 0.8), int(shape[1] * 0.9)
            report["splits"] = {
                "train_frames": [0, train_end],
                "validation_frames": [train_end, validation_end],
                "test_frames": [validation_end, shape[1]],
            }
        statistics = payload.get("statistics", {})
        if isinstance(statistics, dict):
            for name in ("train_mean", "train_std"):
                if name in statistics and torch.as_tensor(statistics[name]).shape != (shape[-1],):
                    report["errors"].append(f"statistics.{name} must contain one value per field")
    if "statistics" not in payload:
        report["warnings"].append("no training-only normalization statistics")
    report["valid"] = not report["errors"]
    return report


def validate_dataset(path: str | Path, field_names: Sequence[str] | None = None) -> dict[str, Any]:
    """Dispatch structural validation by the supported file extension."""
    suffix = Path(path).suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return validate_h5_dataset(path, field_names)
    if suffix in {".pt", ".pth"}:
        return validate_pt_dataset(path, field_names)
    raise ValueError(f"unsupported dataset extension {suffix!r}")
