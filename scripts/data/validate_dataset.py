#!/usr/bin/env python
"""Validate one payload or every canonical case dataset contract.

The case ``configs/dataset.yaml`` files are the single catalog of dataset
paths and field order.  Keeping this helper discovery-based prevents the
command-line tool from drifting from the configs used by the launchers.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.data.validation import validate_dataset

ROOT = Path(__file__).resolve().parents[2]


def dataset_catalog() -> dict[str, tuple[Path, tuple[str, ...] | None]]:
    """Return dataset paths/field order from every canonical case config."""

    catalog: dict[str, tuple[Path, tuple[str, ...] | None]] = {}
    for config_path in sorted((ROOT / "cases").glob("*/configs/dataset.yaml")):
        case_name = config_path.parents[1].name
        config = load_config(config_path)
        dataset = config.get("dataset")
        if not isinstance(dataset, dict) or not dataset.get("path"):
            raise ValueError(f"{config_path} must define dataset.path")
        payload = Path(dataset["path"])
        # Launchers resolve dataset paths against the case root (not the
        # config directory), so catalog validation follows the same contract.
        case_dir = config_path.parents[1]
        # Preserve the canonical catalog path when it is a symlink to a local
        # or external payload.  The validator follows it when opening HDF5/PT,
        # while callers can still see that the payload belongs to this case's
        # dataset catalog.
        payload = Path(os.path.abspath(payload if payload.is_absolute() else case_dir / payload))
        names = dataset.get("field_names")
        fields = tuple(str(name) for name in names) if names is not None else None
        catalog[case_name] = (payload, fields)
    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.all and args.path is None:
        parser.error("provide a path or --all")

    items = (
        dataset_catalog().items()
        if args.all
        else [(args.path.stem, (args.path, None))]
    )
    failed = False
    for name, (path, fields) in items:
        report = validate_dataset(path, fields)
        report["name"] = name
        print(json.dumps(report, indent=2, sort_keys=True))
        failed |= not report["valid"]
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
