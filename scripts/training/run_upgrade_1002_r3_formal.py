#!/usr/bin/env python
"""User-only case launch with an optional sustained-catastrophe watchdog.

Default is validation only. The campaign never invokes --execute.
This wrapper delegates all training/storage/recovery to the existing case CLI.
"""
import argparse
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
spec = importlib.util.spec_from_file_location("r3_pilot_monitor", ROOT / "scripts/training/run_upgrade_1002_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stop-on-sustained-emergency", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    config = pilot._resolved_config(args.config.resolve())
    source = pilot.validate_source_identity(config, pilot._read_json(pilot.SOURCE_CONTRACT_PATH))
    pilot._validate_visible_device(config, os.environ.get("CUDA_VISIBLE_DEVICES"))
    if config["optimization"]["epochs"] != 5000:
        raise ValueError("this user-only formal wrapper requires exactly 5000 epochs")
    experiment = Path(config["output"]["experiment_name"])
    if experiment.is_absolute() or ".." in experiment.parts or experiment.parts[0] == "Test_1002":
        raise ValueError("formal output must be a distinct standard case run")
    output_root = (pilot.RUNS_DIR / experiment).resolve()
    if output_root.parent != pilot.RUNS_DIR.resolve() or output_root.name in {
        Path(config["source_run"]).parent.name,
        "coherence_fix_ABC_sliced_persistence_formal_5000ep_gpu1"}:
        raise ValueError("formal output must be a distinct standard case directory")
    if pilot._git_identity()["branch"] != pilot.EXPECTED_BRANCH:
        raise ValueError("formal candidate must run on its tested working branch")
    if config["fidelity_controller"].get("native_budget") != .02 or config["fidelity_controller"].get("version") != "endpoint_native_primal_dual_v2":
        raise ValueError("formal candidate must retain tested R3 native corridor and controller")
    command = [sys.executable, "-u", str(pilot.CASE_DIR / "run.py"), "post-train", "--config", str(args.config.resolve())]
    print(json.dumps({"mode":"user_only_formal", "epochs":5000,
        "SOURCE_sha256":source["source_checkpoint_sha256"], "output":str(output_root),
        "emergency_policy":"sustained_catastrophe_stop" if args.stop_on_sustained_emergency else "report_only",
        "execute":args.execute, "command":command}, indent=2), flush=True)
    if not args.execute:
        return 0
    pilot._verify_gpu0()
    before = set(output_root.iterdir()) if output_root.is_dir() else set()
    child = subprocess.Popen(command, cwd=ROOT)
    seen_emergency = False
    monitor = None
    if args.stop_on_sustained_emergency:
        monitor = pilot.R3SafetyMonitor({"output_root":str(output_root),
            "config_sha256":pilot._run_config_digest(config), "started_utc":pilot._utc_now()},
            remaining_seconds=math.inf)
        monitor.start()
    try:
        while child.poll() is None:
            time.sleep(10)
            if monitor is not None:
                continue
            candidates = [path for path in output_root.iterdir() if path not in before and path.is_dir()] if output_root.is_dir() else []
            if not candidates:
                continue
            run = max(candidates, key=lambda path:path.stat().st_mtime)
            evidence = pilot.r3_emergency_evidence(run)
            if evidence is not None and not seen_emergency:
                seen_emergency = True
                print(json.dumps({"formal_emergency":evidence, "run":str(run)}),flush=True)
                if args.stop_on_sustained_emergency:
                    child.send_signal(signal.SIGINT)
        return child.wait()
    except KeyboardInterrupt:
        child.send_signal(signal.SIGINT)
        return child.wait()
    finally:
        if monitor is not None:
            monitor.close()

if __name__ == "__main__":
    raise SystemExit(main())
