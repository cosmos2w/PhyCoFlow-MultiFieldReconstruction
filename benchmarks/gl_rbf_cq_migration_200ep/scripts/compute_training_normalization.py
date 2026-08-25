"""Compute a checksummed mean/std artifact from a contiguous HDF5 train split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from phycoflow_reconstruction.data.manifest import dataset_fingerprint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--field-names", nargs="+", required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--chunk-frames", type=int, default=32)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    dataset_path = args.dataset.resolve()
    if args.frame_start < 0 or args.frame_stop <= args.frame_start:
        raise ValueError("frame range must be non-empty and increasing")
    if args.frame_stride < 1 or args.chunk_frames < 1:
        raise ValueError("frame stride and chunk size must be positive")

    total = np.zeros(len(args.field_names), dtype=np.float64)
    total_squared = np.zeros_like(total)
    value_count = 0
    frame_indices = np.arange(
        args.frame_start, args.frame_stop, args.frame_stride, dtype=np.int64
    )
    with h5py.File(dataset_path, "r") as handle:
        fields = handle["fields"]
        if fields.shape[0] != 1 or fields.shape[-1] != len(args.field_names):
            raise ValueError("dataset layout or field count disagrees with the request")
        if args.frame_stop > fields.shape[1]:
            raise ValueError("frame stop exceeds the dataset time axis")
        for start in range(0, len(frame_indices), args.chunk_frames):
            selected = frame_indices[start : start + args.chunk_frames]
            values = fields[0, selected, :, 0, 0, :]
            total += values.sum(axis=(0, 1), dtype=np.float64)
            total_squared += np.square(values, dtype=np.float64).sum(
                axis=(0, 1), dtype=np.float64
            )
            value_count += int(values.shape[0] * values.shape[1])

    offset = total / value_count
    variance = np.maximum(total_squared / value_count - np.square(offset), 1.0e-12)
    payload = {
        "version": "1",
        "method": "mean_std",
        "field_names": list(args.field_names),
        "offset": offset.tolist(),
        "scale": np.sqrt(variance).tolist(),
        "dataset_fingerprint": dataset_fingerprint(dataset_path),
        "statistics_split": "train",
        "split_strategy": "chronological_frames_80_10_10",
        "trajectory_indices": [0],
        "frame_start": args.frame_start,
        "frame_stop_exclusive": args.frame_stop,
        "frame_stride": args.frame_stride,
        "sample_value_count_per_field": value_count,
        "variance_estimator": "population",
    }
    payload["artifact_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    encoded = json.dumps(payload, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
