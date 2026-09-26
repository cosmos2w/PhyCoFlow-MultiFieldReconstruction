#!/usr/bin/env python
"""Launch active-emulsion work through the common reconstruction CLI."""

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from case import CASE_SPEC  # noqa: F401

from phycoflow_reconstruction.cli import run_case_cli

if __name__ == "__main__":
    raise SystemExit(run_case_cli("active_emulsion", Path(__file__).resolve().parent))
