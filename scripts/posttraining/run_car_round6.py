#!/usr/bin/env python3
"""Bound Round 6 GPU work and run the existing SSS-only post-trainer.

Runtime device choices remain outside the scientific configuration. The ledger
charges enclosing GPU leases once, including preparation and failed attempts.
"""
from __future__ import annotations

import argparse
import copy
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
    atomic_json, digest, freeze_execution_code, immutable_json, locked_ledger, spent,
)

CAP_SECONDS = 30 * 3600
PREPARATION_SECONDS = 4 * 3600
ARM = "FFM_SSS_EpochCalibrated_PI_Repair"


def reference(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": digest(path)}


def checked(ref):
    path = Path(ref["path"]).resolve(strict=True)
    if digest(path) != ref["sha256"]:
        raise ValueError(f"bound artifact changed: {path}")
    return path


def committed_seconds(state, phase=None):
    closed = sum(float(row["seconds"]) for row in state["closed"]
                 if phase is None or row["phase"] == phase)
    active = sum(float(row["max_seconds"]) for row in state["active"].values()
                 if phase is None or row["phase"] == phase)
    return closed + active


def acquire(state, *, token, device, phase, duration, now):
    if not math.isfinite(duration) or not 0 < duration <= CAP_SECONDS:
        raise ValueError("positive bounded GPU duration required")
    if committed_seconds(state) + duration > CAP_SECONDS:
        raise ValueError("lease exceeds the additive 30 occupied GPU-hour ceiling")
    if phase == "preparation" and committed_seconds(state, phase) + duration > PREPARATION_SECONDS:
        raise ValueError("lease exceeds the four occupied GPU-hour preparation allowance")
    if any(row["gpu_uuid"] == device["uuid"] for row in state["active"].values()):
        raise ValueError("selected runtime device already leased")
    state["active"][token] = {"gpu_uuid": device["uuid"], "phase": phase,
        "start_unix": now, "max_seconds": duration, "pid": os.getpid()}


def lease(args, campaign):
    from scripts.posttraining.prepare_car_round4 import device_identity
    if not args.child_command:
        raise ValueError("command required after --")
    selected = os.environ.get("CUDA_VISIBLE_DEVICES")
    cache_path = os.environ.get("KEOPS_CACHE_FOLDER")
    if not selected or "," in selected or not cache_path:
        raise ValueError("one runtime CUDA device and a campaign KeOps cache required")
    cache = Path(cache_path)
    build = "_".join(platform.uname()[:3]) + "_p" + sys.version.split(" ")[0]
    (cache / (build + "_CUDA_VISIBLE_DEVICES_" + selected)).mkdir(parents=True, exist_ok=True)
    device = device_identity()
    token, started = uuid.uuid4().hex, time.time()
    ledger = campaign / "operations/gpu_ledger.json"
    with locked_ledger(ledger) as state:
        acquire(state, token=token, device=device, phase=args.phase,
                duration=args.max_seconds, now=started)
    receipt = {"schema": "car_round6_gpu_lease_v1", "token": token,
        "phase": args.phase, "status": "running", "start_unix": started,
        "max_seconds": args.max_seconds, "device": device,
        "cuda_visible_devices": selected, "command": args.child_command,
        "python": sys.executable, "script_sha256": digest(Path(__file__))}
    atomic_json(campaign / "operations/launches" / f"{token}.json", receipt)
    child = None
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        env = dict(os.environ, CAR_ROUND6_LEASE_TOKEN=token,
                   CAR_ROUND6_CAMPAIGN=str(campaign),
                   CAR_ROUND6_LEASE_DEADLINE=str(started + args.max_seconds))
        child = subprocess.Popen(args.child_command, env=env, start_new_session=True)
        try:
            code = child.wait(timeout=max(1., args.max_seconds - 15.))
        except subprocess.TimeoutExpired:
            raise TimeoutError("bounded Round 6 GPU lease expired")
        receipt.update(status="completed" if code == 0 else "failed", exit_code=code)
        if code:
            raise subprocess.CalledProcessError(code, args.child_command)
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
            cpu_accounting="inside enclosing GPU lease; do not add again")
        with locked_ledger(ledger) as state:
            row = state["active"].pop(token)
            state["closed"].append({**row, "token": token,
                                   "end_unix": ended, "seconds": ended-started})
        atomic_json(campaign / "operations/completions" / f"{token}.json", receipt)


def prepare(args, campaign):
    import yaml
    from scripts.posttraining.prepare_car_round4 import resolve_config
    audit = json.loads(args.audit.read_text())
    qualification = audit["controller_qualification"]
    gamma = float(qualification["gamma_star"])
    if audit.get("status") != "completed" or qualification.get("qualified") is not True:
        raise ValueError("a completed, measured controller qualification is required")
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("PI requires a measured positive finite gain")
    cfg, prior = resolve_config(campaign.parent / "round_3")
    cfg = copy.deepcopy(cfg)
    policy = cfg["optimization"]["multitask"]["external_policy"]
    if policy["method"] != "lagrangian":
        raise ValueError("Round 6 preserves the Lagrangian SSS-only objective")
    policy["lagrangian"]["epoch_calibrated_pi"] = {
        "gamma": gamma, "tau_epochs": 1.0, "horizon_epochs": 10.0,
        "steps_per_epoch": 12}
    cfg["optimization"]["epochs"] = 500
    cfg["runtime"]["plot_every_steps"] = 10
    cfg["evaluation"]["external_assessment"] = True
    cfg["evaluation"]["max_samples"] = 100
    cfg["checkpointing"].update(every_epochs=25, epochs=[20, 100, 500],
                                  panel_milestones_only=True)
    cfg["output"]["experiment_name"] = "coherence_adaptive_repair/round_6/runs/formal/" + ARM
    if args.group_queries:
        if audit.get("query_qualification", {}).get("qualified") is not True:
            raise ValueError("grouped queries require the measured B64 action and cost qualification")
        cfg["coherence"]["scalable_anchored_coherence"]["condition_query_groups"] = 8
    config_path = campaign / "configs" / f"{ARM}.yaml"
    content = yaml.safe_dump(cfg, sort_keys=False)
    if config_path.exists() and config_path.read_text() != content:
        raise ValueError("candidate config already exists with different content")
    config_path.write_text(content)
    immutable_json(campaign / "configs/prepared.json", {
        "schema": "car_round6_prepared_v1", "arms": {ARM: reference(config_path)},
        "audit": reference(args.audit), "historical_config": prior,
        "source_snapshot": cfg["source_run"], "maximum_epoch": 500,
        "source_reference": cfg["optimization"]["multitask"]["fidelity_budget"]["reference_artifact"],
        "review_epoch": 100, "pilot_epoch": 20, "gpu_hour_ceiling": 30,
        "native_loss": False, "test_locked": True,
        "query_law": "condition_groups_8" if args.group_queries else "shared",
        "config_solver": "torchjd_wide",
    })
    print(config_path, flush=True)


def freeze(args, campaign):
    prepared = json.loads((campaign / "configs/prepared.json").read_text())
    for ref in prepared["arms"].values():
        checked(ref)
    pilot = json.loads(args.pilot.read_text())
    if pilot.get("qualified") is not True:
        raise ValueError("20-epoch pilot and affected fresh-process resume must qualify")
    forecast = json.loads(args.forecast.read_text())
    if not forecast.get("fits_gpu_budget") or forecast["forecast_gpu_seconds"] > CAP_SECONDS:
        raise ValueError("whole measured forecast must fit the authorized GPU ceiling")
    # All repository Python drivers are modest; include their real dependencies
    # rather than importing mutable helper code from the working checkout.
    drivers = [str(p.relative_to(ROOT)) for p in (ROOT / "scripts").rglob("*.py")]
    code = freeze_execution_code(ROOT, campaign / "operations/code", drivers)
    immutable_json(campaign / "configs/formal_freeze.json", {
        **prepared, "schema": "car_round6_freeze_v1", "code_root": str(code),
        "code_sha256": code.name, "created_unix": time.time(),
        "pilot": reference(args.pilot), "forecast": reference(args.forecast),
        "selection_use": "reused development VALIDATION100; tasks withheld until endpoint",
    })
    print(code, flush=True)


def forecast(args, campaign):
    """Resolve the whole bounded forecast from completed pilot epoch records."""
    import statistics
    pilot = json.loads(args.pilot.read_text())
    if pilot.get("qualified") is not True or pilot.get("epochs") != 20:
        raise ValueError("completed qualified 20-epoch pilot required")
    timings = pilot["epoch_timing"]
    values = [float(row["wall_seconds"]) for row in timings]
    if len(values) != 20 or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("twenty complete measured epoch times required")
    ledger_path = campaign / "operations/gpu_ledger.json"
    with locked_ledger(ledger_path) as state:
        if state["active"]:
            raise ValueError("finalize preparation leases before the substantive forecast")
        already = spent(state)
    recent = statistics.mean(values[-10:])
    upper = max(max(values), 1.1 * recent)
    assessment, reserve = 3600., 3600.
    forecast500 = already + 500 * upper + assessment + reserve
    immutable_json(campaign / "configs/measured_forecast.json", {
        "schema": "car_round6_measured_forecast_v1",
        "pilot": reference(args.pilot), "spent_gpu_seconds": already,
        "recent_10_epoch_mean_seconds": recent, "upper_epoch_seconds": upper,
        "upper_basis": "maximum observed complete epoch or 110% of recent ten-epoch mean, whichever is larger",
        "review_epoch": 100, "maximum_epoch": 500,
        "review_training_gpu_seconds": 100 * upper,
        "formal_training_gpu_seconds": 500 * upper,
        "assessment_gpu_seconds": assessment, "remaining_gpu_reserve_seconds": reserve,
        "forecast_gpu_seconds": forecast500,
        "fits_gpu_budget": forecast500 <= CAP_SECONDS,
        "cpu_assessment_and_reporting_wall_allowance_seconds": 7200.,
        "assessment_basis": "Reuse SOURCE and one cached J4 VALIDATION100 bank per checkpoint; historical Round 5 two-candidate CPU score took 68.6 seconds and one child bank lease 33.3 seconds. One GPU-hour reserve covers both scheduled child banks and contingency. CPU/report work budgeted separately.",
        "optional_integral_baseline": False,
        "baseline_disposition": "Source, clocks, panel, objectives and observer contracts remain comparable to the retained historical failure; no additional baseline question changes adoption.",
        "test_locked": True,
    })
    print(campaign / "configs/measured_forecast.json", flush=True)


def train(args, campaign):
    import yaml
    from phycoflow_reconstruction.training.post_training import run_post_training
    if not os.environ.get("CAR_ROUND6_LEASE_TOKEN"):
        raise ValueError("training must run under the Round 6 enclosing GPU lease")
    prepared = json.loads((campaign / "configs/prepared.json").read_text())
    cfg = yaml.safe_load(checked(prepared["arms"][ARM]).read_text())
    if args.pilot:
        if args.stop_epoch != 20 or args.resume is not None:
            raise ValueError("pilot is a fresh SOURCE 20-epoch run")
        cfg["output"]["experiment_name"] = "coherence_adaptive_repair/round_6/runs/pilot/" + ARM
    else:
        manifest = json.loads((campaign / "configs/formal_freeze.json").read_text())
        if CODE.resolve() != Path(manifest["code_root"]).resolve():
            raise ValueError("formal training must use the frozen execution copy")
        if args.stop_epoch not in (100, 500):
            raise ValueError("formal stop is the 100 review or conditional 500 endpoint")
        if args.stop_epoch == 500:
            review = json.loads((campaign / "operations/learning_review_epoch_100.json").read_text())
            if args.resume is None or review.get("continue_to_500") is not True:
                raise ValueError("continuation requires the explicit measured epoch-100 decision")
            checked(review["assessment"])
    if args.resume is not None:
        cfg = yaml.safe_load((args.resume / "resolved_config.yaml").read_text())
        expected = yaml.safe_load(checked(prepared["arms"][ARM]).read_text())
        if cfg != expected:
            raise ValueError("resume uses saved unchanged scientific configuration")
    requested = False
    def stop_signal(_sig, _frame):
        nonlocal requested
        requested = True
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    def stop(step, _run):
        deadline = float(os.environ.get("CAR_ROUND6_LEASE_DEADLINE", "inf"))
        return requested or step >= 12 * args.stop_epoch or time.time() >= deadline - 180
    run = run_post_training(cfg, case_dir=ROOT / "cases/turbulent_combustion",
                            resume=args.resume, budget_stop_requested=stop)
    status = json.loads((run / "status.json").read_text())
    atomic_json(campaign / "operations" / ("pilot_run.json" if args.pilot else f"formal_epoch_{args.stop_epoch}.json"),
                {"schema": "car_round6_training_result_v1", "run_dir": str(run),
                 "target_epoch": args.stop_epoch, "status": status})
    print(run, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("lease")
    p.add_argument("--phase", choices=("preparation", "formal_training", "assessment"), required=True)
    p.add_argument("--max-seconds", type=float, required=True)
    p.add_argument("child_command", nargs=argparse.REMAINDER)
    p = sub.add_parser("prepare")
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--group-queries", action="store_true")
    p = sub.add_parser("freeze")
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--forecast", type=Path, required=True)
    p = sub.add_parser("forecast")
    p.add_argument("--pilot", type=Path, required=True)
    p = sub.add_parser("train")
    p.add_argument("--stop-epoch", type=int, choices=(20, 100, 500), required=True)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--resume", type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    if args.command == "lease" and args.child_command[:1] == ["--"]:
        args.child_command.pop(0)
    {"lease": lease, "prepare": prepare, "freeze": freeze, "forecast": forecast, "train": train}[args.command](args, campaign)


if __name__ == "__main__":
    main()
