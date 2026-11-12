#!/usr/bin/env python3
"""Launch one owned Round 8 stage on an idle physical GPU 1 or 2."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import uuid as uuid_module


def snapshot_sources(root, campaign):
    files = sorted((root / "src/phycoflow_reconstruction").rglob("*.py"))
    files += [root / "scripts/run_round8.py"]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    target = campaign / "artifacts/execution" / digest
    if not target.exists():
        staging = target.with_name("." + digest + f".{os.getpid()}")
        for path in files:
            destination = staging / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        (staging / "sources.json").write_text(json.dumps(hashes, indent=2) + "\n")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(target)
    for relative, expected in hashes.items():
        if hashlib.sha256((target / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("execution snapshot changed: " + relative)
    return target, digest


def launch(root, *, gpu, stage, arguments, expected_seconds, monitor_receipt):
    root = Path(root).resolve()
    campaign = root / "cases/turbulent_combustion/runs/coherence_compute_round8"
    monitor = json.loads(Path(monitor_receipt).read_text())
    if (monitor.get("name") != "round8_luna_monitor" or monitor.get("model") != "gpt-6-luna"
            or monitor.get("reasoning_effort") != "low" or not monitor.get("agent_id")):
        raise ValueError("the named Luna-low monitor must be available before launch")
    if gpu not in (1, 2) or expected_seconds <= 0:
        raise ValueError("physical GPU 1/2 and a positive measured forecast are required")
    devices = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    applications = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits"], text=True)
    row = next((line.split(",") for line in devices.splitlines()
                if int(line.split(",")[0].strip()) == gpu), None)
    if row is None:
        raise RuntimeError(f"physical GPU {gpu} is unavailable")
    uuid = row[1].strip()
    occupied = [line for line in applications.splitlines() if line.split(",")[0].strip() == uuid]
    if occupied or float(row[3]) >= 512 or float(row[5]) > 0:
        raise RuntimeError(f"physical GPU {gpu} is occupied; preserve existing jobs")
    snapshot, snapshot_digest = snapshot_sources(root, campaign)
    cache = campaign / f"caches/gpu{gpu}"
    build = f"{platform.system()}_{platform.node()}_{platform.release()}_p{platform.python_version()}"
    for name in ("keops", "matplotlib", "torch_extensions", f"keops/{build}",
                 f"keops/{build}_CUDA_VISIBLE_DEVICES_{uuid}"):
        (cache / name).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=uuid, ROUND8_PHYSICAL_GPU_UUID=uuid,
        ROUND8_PHYSICAL_GPU_INDEX=str(gpu), ROUND8_REPO_ROOT=str(root),
        ROUND8_CODE_ROOT=str(snapshot), KEOPS_CACHE_FOLDER=str(cache / "keops"),
        MPLCONFIGDIR=str(cache / "matplotlib"), TORCH_EXTENSIONS_DIR=str(cache / "torch_extensions"),
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        PHYCOFLOW_TOPOLOGY_WORKERS="4",
        CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1",
        PYTHONPATH=str(snapshot / "src"),
    )
    lease_token = uuid_module.uuid4().hex
    env['ROUND8_LEASE_TOKEN'] = lease_token
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log = campaign / "logs" / f"{stage}_{stamp}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    # Reserve before Python imports initialize CUDA/KeOps so occupied time
    # includes startup and compilation. The child adopts this exact lease.
    ledger_path = campaign / 'artifacts/budget_ledger.json'
    with (campaign / 'artifacts/budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
            'schema':'round8_budget_v1', 'qualification_attempts':0,
            'formal_attempts':0, 'intervals':[]}
        if any(x['device_uuid'] == uuid and 'end' not in x for x in ledger['intervals']):
            raise RuntimeError('device already reserved; reconcile its owned lease before relaunch')
        lease_start = time.time()
        spent = sum(x.get('end', lease_start) - x['start'] for x in ledger['intervals'])
        limit = 28 * 3600 if stage == 'train' else 32 * 3600
        if spent >= limit - 120:
            raise RuntimeError('campaign budget or closeout reserve unavailable')
        lease = {'id':lease_token, 'stage':stage, 'device_uuid':uuid, 'start':lease_start}
        ledger['intervals'].append(lease)
        with log.open("x") as stream:
            process = subprocess.Popen(
                [sys.executable, str(snapshot / "scripts/run_round8.py"), stage, *arguments],
                cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        lease['pid'] = process.pid
        temporary = ledger_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(ledger, indent=2) + '\n')
        temporary.replace(ledger_path)
    receipt = {
        "schema": "round8_launch_v1", "stage": stage, "arguments": arguments,
        "pid": process.pid, "launched_utc": stamp, "launched_unix": datetime.now(timezone.utc).timestamp(),
        "gpu_index": gpu, "gpu_uuid": uuid, "device_name": row[2].strip(),
        "vram_mib": float(row[4]), "expected_seconds": float(expected_seconds),
        "persistence_workers": 4,
        "monitor_receipt": str(Path(monitor_receipt).resolve()), "log": str(log),
        "source_snapshot": str(snapshot), "source_snapshot_sha256": snapshot_digest,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "devices_preflight": devices, "applications_preflight": applications,
        "initial_status": process.poll(),
        "lease_token": lease_token,
    }
    path = campaign / "artifacts" / f"{stage}_launch_{stamp}.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    return {"receipt": str(path), **receipt}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(1, 2), required=True)
    parser.add_argument("--expected-seconds", type=float, required=True)
    parser.add_argument("--monitor-receipt", required=True)
    parser.add_argument("stage")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    print(json.dumps(launch(Path(__file__).resolve().parents[1], gpu=args.gpu,
                           stage=args.stage, arguments=args.arguments,
                           expected_seconds=args.expected_seconds,
                           monitor_receipt=args.monitor_receipt), indent=2))


if __name__ == "__main__":
    main()
