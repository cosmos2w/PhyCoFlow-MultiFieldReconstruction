#!/usr/bin/env python
"""Create or verify one case-local dataset symlink without copying payloads."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from phycoflow_reconstruction.config import load_config

ROOT = Path(__file__).resolve().parents[2]


def dataset_target(case: str) -> Path:
    """Read the canonical payload location from the case dataset config."""

    case_dir = ROOT / "cases" / case
    config_path = case_dir / "configs" / "dataset.yaml"
    config = load_config(config_path)
    dataset = config.get("dataset")
    if not isinstance(dataset, dict) or not dataset.get("path"):
        raise ValueError(f"{config_path} must define dataset.path")
    configured_path = Path(str(dataset["path"]))
    # Keep the catalog path itself when it is already a symlink.  Resolving
    # here would follow an existing link into an external payload and make the
    # command replace the link target instead of verifying the catalog entry.
    return Path(os.path.abspath(configured_path if configured_path.is_absolute() else case_dir / configured_path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    case_paths = sorted((ROOT / "cases").glob("*/configs/dataset.yaml"))
    cases = tuple(path.parents[1].name for path in case_paths)
    parser.add_argument("--case", choices=cases, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    target = dataset_target(args.case)
    target.parent.mkdir(parents=True, exist_ok=True)
    relative_source = Path(os.path.relpath(source, target.parent))
    if target.is_symlink():
        if target.resolve() != source:
            raise FileExistsError(f"{target} already links to {target.resolve()}")
        print(f"verified {target} -> {relative_source}")
        return 0
    if target.exists():
        raise FileExistsError(f"refusing to replace existing payload {target}")
    target.symlink_to(relative_source)
    print(f"linked {target} -> {relative_source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
