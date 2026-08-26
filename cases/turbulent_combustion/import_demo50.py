#!/usr/bin/env python
"""Validate and record the version-locked DemoN50 checkpoint import.

This case utility performs no conversion and never modifies the historical run.
It strictly loads the checkpoint, verifies its explicit channel mapping against
the source HDF5 file, and writes a small provenance manifest for later Phase-5
post-training runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

CASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CASE_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from phycoflow_reconstruction.models.compatibility import load_legacy_demo50


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (CASE_DIR / candidate).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/compatibility/demo50.yaml"))
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else CASE_DIR / args.config
    config = yaml.safe_load(config_path.read_text())
    legacy = config["legacy"]
    _, manifest = load_legacy_demo50(
        _resolve(legacy["run_directory"]),
        _resolve(legacy["dataset_path"]),
        legacy["channel_mapping"],
        checkpoint=legacy.get("checkpoint", "best.pt"),
    )
    output = _resolve(config["output"]["manifest"])
    manifest.save(output)
    print(f"DemoN50 strict import passed; manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
