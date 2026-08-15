#!/usr/bin/env python
"""Build a deterministic sensor manifest through a case launcher configuration."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("turbulent_combustion", "brusselator", "ks", "mass_transport_fluid"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    case_dir = root / "Cases" / args.case
    command = [
        "python", "run.py", "build-manifest", "--config", args.config,
        "--output", args.output, "--max-samples", str(args.max_samples), "--split", args.split,
    ]
    return subprocess.run(command, cwd=case_dir, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
