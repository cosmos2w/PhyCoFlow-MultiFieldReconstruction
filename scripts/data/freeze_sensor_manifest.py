#!/usr/bin/env python
"""Copy a verified portable SensorManifest into a release directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from phycoflow_reconstruction.data.manifest import SensorManifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = SensorManifest.load(args.source)
    if manifest.version != "3" or Path(manifest.dataset_path).name != manifest.dataset_path:
        raise ValueError("release manifests must use portable version-3 catalog identity")
    manifest.save(args.output)
    print(f"{args.output}: {manifest.digest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
