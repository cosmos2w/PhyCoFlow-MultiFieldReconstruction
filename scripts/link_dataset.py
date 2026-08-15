#!/usr/bin/env python
"""Create or verify one case-local dataset symlink without copying payloads."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

TARGET_NAMES = {
    "brusselator": "brusselator.h5",
    "kolmogorov": "kolmogorov.h5",
    "ks": "ks.h5",
    "mass_transport_fluid": "mass_transport_fluid_demo.h5",
    "turbulent_combustion": "Merged_CH4COTU1P.h5",
}
ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(TARGET_NAMES), required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    target = ROOT / "Dataset" / args.case / TARGET_NAMES[args.case]
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
