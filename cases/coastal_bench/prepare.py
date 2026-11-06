#!/usr/bin/env python
"""Audit the stored split and compute training-only statistics without editing HDF5."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from phycoflow_reconstruction.data.manifest import dataset_fingerprint
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.data.validation import validate_h5_dataset

FIELDS = ("u", "v", "temp", "salt")
DEFAULT_SOURCE = PROJECT / "datasets/coastal_bench/coastal_bench.h5"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "artifacts"


def training_statistics(handle: h5py.File) -> tuple[np.ndarray, np.ndarray, int]:
    """Merge float64 per-sequence moments using only stored training trajectories."""
    channels = handle["fields"].shape[-1]
    count = 0
    mean = np.zeros(channels, dtype=np.float64)
    m2 = np.zeros(channels, dtype=np.float64)
    trajectories = handle["splits/train"][:]
    if not len(trajectories):
        raise ValueError("training split is empty")
    for position, trajectory in enumerate(trajectories):
        values = np.asarray(handle["fields"][int(trajectory)], dtype=np.float64).reshape(
            -1, channels
        )
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite training values in trajectory {trajectory}")
        local_count = len(values)
        local_mean = values.mean(axis=0)
        values -= local_mean
        local_m2 = np.einsum("ij,ij->j", values, values)
        delta = local_mean - mean
        total = count + local_count
        m2 += local_m2 + delta**2 * (count * local_count / total)
        mean += delta * (local_count / total)
        count = total
        if (position + 1) % 50 == 0 or position + 1 == len(trajectories):
            print(f"normalization: {position + 1}/{len(trajectories)} train sequences", flush=True)
    scale = np.sqrt(m2 / count)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("training standard deviations must be finite and positive")
    return mean, scale, count


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare(source: Path, output: Path, *, recompute: bool = False) -> dict:
    source, output = source.resolve(), output.resolve()
    report = validate_h5_dataset(source, FIELDS)
    if not report["valid"]:
        raise ValueError(f"invalid CoastalBench payload: {report['errors']}")
    fingerprint = dataset_fingerprint(source)
    output.mkdir(parents=True, exist_ok=True)
    statistics_path = output / "train_mean_std.json"
    with h5py.File(source, "r") as handle:
        splits = {
            name: handle[f"splits/{name}"][:].astype(int).tolist()
            for name in ("train", "validation", "test")
        }
        if any(not indices for indices in splits.values()):
            raise ValueError("all three stored trajectory splits must be non-empty")
        ids = handle["trajectory_id"].asstr()[:].tolist()
        if len(ids) != handle["fields"].shape[0] or len(set(ids)) != len(ids):
            raise ValueError("trajectory IDs must be unique and align with fields")
        times = handle["fields"].shape[1]
        audit = {
            **report,
            "dataset_fingerprint": fingerprint,
            "split_strategy": "stored_trajectory",
            "trajectory_indices": splits,
            "trajectory_ids": {
                name: [ids[index] for index in indices] for name, indices in splits.items()
            },
            "snapshot_counts": {name: len(indices) * times for name, indices in splits.items()},
            "time_stride": 1,
            "field_names": list(FIELDS),
            "source_modified": False,
        }
        if statistics_path.exists() and not recompute:
            FieldNormalizer.from_artifact(
                statistics_path, field_names=FIELDS, dataset_fingerprint=fingerprint
            )
            existing = json.loads(statistics_path.read_text())
            if existing.get("trajectory_indices") != splits["train"] or existing.get(
                "frame_indices"
            ) != list(range(times)):
                raise ValueError(
                    "existing normalization selection disagrees with stored training split"
                )
            print(f"Reusing verified statistics: {statistics_path}", flush=True)
        else:
            mean, std, count = training_statistics(handle)
            payload = {
                "version": "1",
                "method": "mean_std",
                "statistics_split": "train",
                "dataset_fingerprint": fingerprint,
                "field_names": list(FIELDS),
                "offset": mean.tolist(),
                "scale": std.tolist(),
                "trajectory_indices": splits["train"],
                "frame_indices": list(range(times)),
                "point_count_per_field": count,
            }
            payload["artifact_sha256"] = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            _write_json(statistics_path, payload)
            print(f"Training mean: {mean.tolist()}\nTraining std: {std.tolist()}", flush=True)
    _write_json(output / "split_audit.json", audit)
    print(
        json.dumps(
            {"snapshots": audit["snapshot_counts"], "normalization": str(statistics_path)}, indent=2
        ),
        flush=True,
    )
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    prepare(args.source, args.output, recompute=args.recompute)
