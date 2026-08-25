"""Build and verify the fixed validation sensor manifest for a benchmark suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phycoflow_reconstruction.data.manifest import SensorManifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="turbulent_combustion")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--query-points", type=int, default=4096)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--field-name", default="T")
    parser.add_argument("--min-count", type=int, default=192)
    parser.add_argument("--max-count", type=int, default=384)
    args = parser.parse_args()
    if args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if args.min_count < 1 or args.max_count < args.min_count:
        raise ValueError("sensor count bounds are invalid")

    case_dir = PROJECT_ROOT / "Cases" / args.case
    config = args.config.resolve()
    output = args.output.resolve()
    command = [
        sys.executable,
        "run.py",
        "build-manifest",
        "--config",
        str(config),
        "--output",
        str(output),
        "--max-samples",
        str(args.max_samples),
        "--query-points",
        str(args.query_points),
        "--split",
        args.split,
    ]
    subprocess.run(command, cwd=case_dir, check=True)

    manifest = SensorManifest.load(output)
    protocol = manifest.protocol
    ranges = protocol.get("field_count_ranges") or {}
    counts = protocol.get("field_counts") or {}
    configured = ranges.get(args.field_name)
    if counts or configured is None:
        raise ValueError(
            "fixed benchmark manifest must use a count range for the requested field only"
        )
    if list(configured) != [args.min_count, args.max_count] or set(ranges) != {args.field_name}:
        raise ValueError(
            f"manifest protocol range does not match {args.field_name}={args.min_count}..{args.max_count}"
        )
    field_id = {"CH4": 0, "CO": 1, "T": 2, "U_1": 3, "p": 4}.get(args.field_name)
    if field_id is None:
        raise ValueError("manifest verification needs the downstream field-order mapping")
    counts_by_sample = {}
    for sample_id, pairs in manifest.indices.items():
        if not pairs or any(int(pair[1]) != field_id for pair in pairs):
            raise ValueError(f"manifest sample {sample_id!r} is not T-only")
        count = len(pairs)
        if not args.min_count <= count <= args.max_count:
            raise ValueError(f"manifest sample {sample_id!r} has invalid count {count}")
        counts_by_sample[sample_id] = count
    if len(counts_by_sample) != args.max_samples:
        raise ValueError(
            f"manifest contains {len(counts_by_sample)} samples, expected {args.max_samples}"
        )
    if not manifest.query_indices or any(
        len(points) != args.query_points for points in manifest.query_indices.values()
    ):
        raise ValueError("fixed manifest query indices do not match --query-points")
    print(
        json.dumps(
            {
                "path": str(output),
                "sha256": manifest.digest(),
                "split": manifest.split,
                "samples": len(counts_by_sample),
                "sensor_count_min": min(counts_by_sample.values()),
                "sensor_count_max": max(counts_by_sample.values()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
