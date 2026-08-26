#!/usr/bin/env python
"""Build a deterministic sensor manifest through a case launcher configuration."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    cases = tuple(
        path.parent.parent.name
        for path in sorted((root / "cases").glob("*/configs/dataset.yaml"))
    )
    parser.add_argument("case", choices=cases)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="validation"
    )
    args = parser.parse_args()
    case_dir = root / "cases" / args.case
    command = [
        "python",
        "run.py",
        "build-manifest",
        "--config",
        args.config,
        "--output",
        args.output,
        "--max-samples",
        str(args.max_samples),
        "--split",
        args.split,
    ]
    return subprocess.run(command, cwd=case_dir, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
