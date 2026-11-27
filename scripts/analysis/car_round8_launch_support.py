#!/usr/bin/env python3
"""Launch one bounded cached support readout and write its terminal receipt."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from scripts.analysis.car_round8_cached_summary import ref


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--joint100-assessment", type=Path, required=True)
    p.add_argument("--joint500-assessment", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    args = p.parse_args()
    directory = args.output.resolve()
    if not args.worker:
        directory.mkdir(parents=True, exist_ok=False)
        argv = [sys.executable, str(Path(__file__).resolve()), "--worker",
                "--joint100-assessment", str(args.joint100_assessment.resolve(strict=True)),
                "--joint500-assessment", str(args.joint500_assessment.resolve(strict=True)),
                "--summary", str(args.summary.resolve(strict=True)), "--output", str(directory)]
        log = (directory / "run.log").open("x")
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        log.close()
        receipt = {"schema": "car_round8_support_launch_v1", "pid": process.pid,
                   "started_unix": time.time(), "expected_wall_seconds": "60-120",
                   "completion_signal": str(directory / "completion.json"), "log": str(directory / "run.log"),
                   "scorer": ref(ROOT / "scripts/analysis/car_round8_support_readout.py"),
                   "inputs": {"joint100": ref(args.joint100_assessment), "joint500": ref(args.joint500_assessment),
                              "summary": ref(args.summary)}, "gpu": "none", "threads": 2,
                   "scope": "matched cached VALIDATION100 SOURCE, JointAnchor100 and JointAnchor500; no inference or TEST"}
        with (directory / "launch.json").open("x") as f:
            json.dump(receipt, f, indent=2); f.write("\n")
        print(json.dumps(receipt))
        return
    started = time.time()
    argv = [sys.executable, str(ROOT / "scripts/analysis/car_round8_support_readout.py"),
            "--joint100-assessment", str(args.joint100_assessment), "--joint500-assessment", str(args.joint500_assessment),
            "--summary", str(args.summary), "--output", str(directory / "result"), "--threads", "2"]
    code = subprocess.run(argv, cwd=ROOT, check=False).returncode
    output = directory / "result/readout.json"
    status = "failed"
    if code == 0 and output.exists():
        readout = json.loads(output.read_text())
        if readout.get("status") == "completed" and len(readout.get("sample_ids", [])) == 100:
            status = "completed"
    receipt = {"schema": "car_round8_support_completion_v1", "status": status, "exit_code": code,
               "started_unix": started, "ended_unix": time.time(), "wall_seconds": time.time()-started,
               "readout": ref(output) if output.exists() else None}
    temporary = directory / "completion.json.tmp"
    with temporary.open("x") as f:
        json.dump(receipt, f, indent=2); f.write("\n")
    temporary.replace(directory / "completion.json")


if __name__ == "__main__":
    main()
