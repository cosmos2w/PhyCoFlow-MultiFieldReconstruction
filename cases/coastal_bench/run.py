#!/usr/bin/env python
"""Launch CoastalBench through the shared reconstruction CLI."""

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from case import CASE_SPEC  # noqa: F401

from phycoflow_reconstruction.cli import run_case_cli

if __name__ == "__main__":
    raise SystemExit(run_case_cli("coastal_bench", Path(__file__).resolve().parent))
