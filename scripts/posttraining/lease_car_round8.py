#!/usr/bin/env python3
"""Run one owned GPU job inside Round 8's inclusive preparation/learning budget.

Device choice lives in runtime JSON. The default ceiling is six GPU-hours; a
sealed, affirmative learning decision can enable the authorized 24-hour total.
Every attempt, including a failed attempt, is charged to the same ledger.
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
from phycoflow_reconstruction.evaluation.adaptive_campaign import atomic_json, digest, locked_ledger

PREPARATION_SECONDS = 6 * 3600
TOTAL_SECONDS = 24 * 3600


def admitted_ceiling(decision):
    if decision is None:
        return PREPARATION_SECONDS
    if (decision.get("status") != "sealed" or decision.get("admitted") is not True
            or decision.get("test_locked") is not True
            or decision.get("total_gpu_ceiling_seconds") != TOTAL_SECONDS
            or not decision.get("question") or not decision.get("forecast")):
        raise ValueError("a sealed, budgeted learning question is required for the 24-hour ceiling")
    return TOTAL_SECONDS


def acquire(state, *, token, device, phase, duration, now, ceiling):
    if not math.isfinite(duration) or not 60 <= duration <= ceiling:
        raise ValueError("lease duration must be finite and fit the authorized ceiling")
    rows = list(state["closed"]) + list(state["active"].values())
    committed = lambda row: float(row.get("seconds", row.get("max_seconds", 0.)))
    if token in state["active"] or any(row.get("token") == token for row in state["closed"]):
        raise ValueError("lease identity already used")
    if sum(committed(row) for row in rows) + duration > ceiling:
        raise ValueError("lease exceeds Round 8's inclusive authorized GPU budget")
    if phase == "preparation" and sum(committed(row) for row in rows
            if row["phase"] == "preparation") + duration > PREPARATION_SECONDS:
        raise ValueError("lease exceeds the six GPU-hour preparation allowance")
    if phase == "learning" and ceiling != TOTAL_SECONDS:
        raise ValueError("formal learning has not been admitted")
    if any(row["gpu_uuid"] == device["uuid"] for row in state["active"].values()):
        raise ValueError("runtime device already has an owned lease")
    state["active"][token] = {"token": token, "gpu_uuid": device["uuid"],
        "phase": phase, "start_unix": now, "max_seconds": duration, "pid": os.getpid()}


def run(args):
    from scripts.posttraining.prepare_car_round4 import device_identity
    campaign = args.campaign.resolve(strict=True)
    runtime = json.loads(args.runtime.read_text())
    selected = str(runtime["cuda_visible_devices"])
    if not selected or "," in selected:
        raise ValueError("one runtime-selected GPU is required")
    decision_path = campaign / "operations/learning_decision.json"
    decision = json.loads(decision_path.read_text()) if decision_path.exists() else None
    ceiling = admitted_ceiling(decision)
    command = args.child_command[1:] if args.child_command[:1] == ["--"] else args.child_command
    if not command:
        raise ValueError("child command required")
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    device = device_identity()
    cache = campaign / "cache/keops"
    build = "_".join(platform.uname()[:3]) + "_p" + sys.version.split(" ")[0]
    (cache / (build + "_CUDA_VISIBLE_DEVICES_" + selected)).mkdir(parents=True, exist_ok=True)
    started, token = time.time(), uuid.uuid4().hex
    ledger = campaign / "operations/gpu_ledger.json"
    with locked_ledger(ledger) as state:
        acquire(state, token=token, device=device, phase=args.phase,
                duration=args.max_seconds, now=started, ceiling=ceiling)
    receipt = {"schema": "car_round8_gpu_lease_v1", "token": token,
        "label": args.label, "phase": args.phase, "status": "running",
        "start_unix": started, "max_seconds": args.max_seconds,
        "device": device, "runtime": {"path": str(args.runtime.resolve()),
        "sha256": digest(args.runtime)}, "command": command, "python": sys.executable,
        "inclusive_gpu_ceiling_seconds": ceiling, "script_sha256": digest(Path(__file__)),
        "learning_decision": {"path": str(decision_path), "sha256": digest(decision_path)}
            if decision else None}
    launch = campaign / "operations/launches" / f"{token}.json"
    atomic_json(launch, receipt)
    print(launch, flush=True)
    child, requested = None, False

    def stop(_signal, _frame):
        nonlocal requested
        requested = True
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=selected, KEOPS_CACHE_FOLDER=str(cache),
            CAR_ROUND8_LEASE_TOKEN=token, CAR_ROUND8_CAMPAIGN=str(campaign),
            CAR_ROUND8_LEASE_DEADLINE=str(started + args.max_seconds), CAR_REPO_ROOT=str(ROOT))
        child = subprocess.Popen(command, env=env, start_new_session=True)
        code = child.wait(timeout=max(1., args.max_seconds - (time.time() - started) - 15.))
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
            state["closed"].append({**row, "label": args.label, "end_unix": ended,
                                  "seconds": ended-started, "status": receipt["status"]})
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
