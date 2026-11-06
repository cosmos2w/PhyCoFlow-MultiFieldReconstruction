#!/usr/bin/env python
"""Evaluate matched CoastalBench checkpoints with frozen spectral scores."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phycoflow_reconstruction.evaluation.spectral import main

if __name__ == "__main__":
    main()
