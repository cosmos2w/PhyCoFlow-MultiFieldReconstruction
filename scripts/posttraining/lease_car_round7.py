#!/usr/bin/env python3
"""Charge all Round 7 GPU occupation, including preparation and failed work.

The five-hour preparation allocation is a planning target, not an admission
gate. Device selection belongs to the runtime JSON, outside scientific hashes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time
import uuid

CODE = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("CAR_REPO_ROOT", CODE)).resolve()
sys.path[:0] = [str(CODE / "src"), str(CODE)]
from phycoflow_reconstruction.evaluation.adaptive_campaign import (
    atomic_json, digest, locked_ledger,
)

CAP_SECONDS = 40 * 3600


def committed_seconds(state):
    return sum(float(row["seconds"]) for row in state["closed"]) + sum(
        float(row["max_seconds"]) for row in state["active"].values()
    )


def acquire(state, *, token, device, phase, duration, now):
    if not math.isfinite(duration) or not 60 <= duration <= CAP_SECONDS:
        raise ValueError("lease duration must be finite and between 60 seconds and 40 hours")
    if token in state["active"] or any(row.get("token") == token for row in state["closed"]):
        raise ValueError("lease identity already used")
    if committed_seconds(state) + duration > CAP_SECONDS:
        raise ValueError("lease exceeds the inclusive 40 GPU-hour ceiling")
    if any(row["gpu_uuid"] == device["uuid"] for row in state["active"].values()):
        raise ValueError("runtime device already leased")
    state["active"][token] = {
        "gpu_uuid": device["uuid"], "phase": phase, "start_unix": now,
        "max_seconds": duration, "pid": os.getpid(),
    }


def run(args):
    from scripts.posttraining.prepare_car_round4 import device_identity

    campaign = args.campaign.resolve(strict=True)
    runtime = json.loads(args.runtime.read_text())
    selected = str(runtime["cuda_visible_devices"])
    if not selected or "," in selected:
        raise ValueError("one runtime GPU required")
    command = args.child_command[1:] if args.child_command[:1] == ["--"] else args.child_command
    if not command:
        raise ValueError("child command required")
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    device = device_identity()
    cache = campaign / "cache/keops"
    build = "_".join(platform.uname()[:3]) + "_p" + sys.version.split(" ")[0]
    (cache / (build + "_CUDA_VISIBLE_DEVICES_" + selected)).mkdir(parents=True, exist_ok=True)
    started = time.time()
    token = uuid.uuid4().hex
    ledger = campaign / "operations/gpu_ledger.json"
    with locked_ledger(ledger) as state:
        acquire(state, token=token, device=device, phase=args.phase,
                duration=args.max_seconds, now=started)
    receipt = {
        "schema": "car_round7_gpu_lease_v1", "token": token, "phase": args.phase,
        "label": args.label, "status": "running", "start_unix": started,
        "max_seconds": args.max_seconds, "device": device,
        "runtime": str(args.runtime.resolve()), "command": command,
        "python": sys.executable, "script_sha256": digest(Path(__file__)),
        "inclusive_gpu_ceiling_seconds": CAP_SECONDS,
    }
    launch = campaign / "operations/launches" / f"{token}.json"
    atomic_json(launch, receipt)
    print(str(launch), flush=True)
    child = None
    requested = False

    def stop(_signal, _frame):
        nonlocal requested
        requested = True
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=selected,
                   KEOPS_CACHE_FOLDER=str(cache),
                   CAR_ROUND7_LEASE_TOKEN=token, CAR_ROUND7_CAMPAIGN=str(campaign),
                   CAR_ROUND7_LEASE_DEADLINE=str(started + args.max_seconds),
                   CAR_REPO_ROOT=str(ROOT))
        child = subprocess.Popen(command, env=env, start_new_session=True)
        try:
            code = child.wait(timeout=max(1., args.max_seconds - (time.time()-started) - 15.))
        except subprocess.TimeoutExpired:
            raise TimeoutError("Round 7 inclusive GPU lease expired")
        receipt.update(status="completed" if code == 0 and not requested else "failed", exit_code=code)
        if code or requested:
            raise subprocess.CalledProcessError(code or -signal.SIGTERM, command)
    except BaseException as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        ended = time.time()
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        receipt.update(end_unix=ended, seconds=ended-started,
                       child_user_cpu_seconds=after.ru_utime-usage.ru_utime,
                       child_system_cpu_seconds=after.ru_stime-usage.ru_stime,
                       cpu_accounting="inside enclosing GPU lease; not charged again")
        with locked_ledger(ledger) as state:
            row = state["active"].pop(token)
            state["closed"].append({**row, "token": token, "label": args.label,
                                    "end_unix": ended, "seconds": ended-started,
                                    "status": receipt["status"]})
        atomic_json(campaign / "operations/completions" / f"{token}.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--phase", choices=("preparation", "learning", "assessment"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("child_command", nargs=argparse.REMAINDER)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
