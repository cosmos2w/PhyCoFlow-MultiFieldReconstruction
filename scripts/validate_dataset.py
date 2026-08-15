#!/usr/bin/env python
"""Validate one payload or all five registered datasets without loading them fully."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phycoflow_reconstruction.data.validation import validate_dataset

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "turbulent_combustion": (
        ROOT / "Dataset/turbulent_combustion/Merged_CH4COTU1P.h5",
        ("CH4", "CO", "T", "U_1", "p"),
    ),
    "brusselator": (ROOT / "Dataset/brusselator/brusselator.h5", None),
    "kolmogorov": (ROOT / "Dataset/kolmogorov/kolmogorov.h5", None),
    "ks": (ROOT / "Dataset/ks/ks.h5", None),
    "mass_transport_fluid": (
        ROOT / "Dataset/mass_transport_fluid/mass_transport_fluid_demo.h5",
        None,
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.all and args.path is None:
        parser.error("provide a path or --all")

    items = DATASETS.items() if args.all else [(args.path.stem, (args.path, None))]
    failed = False
    for name, (path, fields) in items:
        report = validate_dataset(path, fields)
        report["name"] = name
        print(json.dumps(report, indent=2, sort_keys=True))
        failed |= not report["valid"]
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
