#!/usr/bin/env python3
"""Prepare and run the two bounded Round 3 FFM feedback paths.

Uses the existing campaign ledger and code snapshot tools. Runtime device
selection is operational and is never part of the scientific freeze.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("CAR_REPO_ROOT", CODE)).resolve()
sys.path[:0] = [str(CODE / "src"), str(CODE)]
ARMS = ("FFM_RawRisk_Uniform_Repair", "FFM_SourceScaledRisk_Uniform_Repair")
CAP_SECONDS = 45 * 3600
ASSESSMENT_RESERVE_SECONDS = 3600


def prepare(campaign):
    import yaml
    from phycoflow_reconstruction.evaluation.adaptive_campaign import digest, immutable_json
    from phycoflow_reconstruction.training.scalable_fitting import _digest

    historical = campaign.parent / "round_2"
    run = historical / "runs/formal/FFM_Selected_Bank_Uniform_FullModel/20261118T001357Z_f8b517be"
    config = yaml.safe_load((run / "resolved_config.yaml").read_text())
    old_ref_path = Path(config["optimization"]["multitask"]["fidelity_budget"]["reference_artifact"]["path"])
    old = json.loads(old_ref_path.read_text())
    if digest(old_ref_path) != config["optimization"]["multitask"]["fidelity_budget"]["reference_artifact"]["sha256"]:
        raise ValueError("historical SOURCE risk receipt changed")
    bindings = {}
    for arm in ARMS:
        cfg = copy.deepcopy(config)
        cfg["optimization"]["epochs"] = 500
        cfg["optimization"]["lr_warmup"] = {"enabled": True, "start_lr": 5e-6, "epochs": 50}
        cfg["runtime"]["device"] = "cuda:0"
        cfg["runtime"]["plot_every_steps"] = 10
        cfg["evaluation"]["external_assessment"] = True
        cfg["evaluation"]["split"] = "validation"
        cfg["evaluation"]["max_samples"] = 100
        cfg["checkpointing"].update(every_epochs=25, epochs=[100, 500], panel_milestones_only=True)
        cfg["output"]["experiment_name"] = f"coherence_adaptive_repair/round_3/runs/formal/{arm}"
        candidate = arm == ARMS[1]
        cfg["optimization"]["multitask"]["external_policy"]["lagrangian"]["dual_lr"] = 1e-4 if candidate else .01
        reference = copy.deepcopy(old)
        if candidate:
            scales = [max(float(x), 1e-8) for x in old["risk_reference_panel"]["source_member_mse"]]
            reference["reference"]["risk_scales"] = scales
            reference["settings"]["mean_risk_scales"] = scales
            reference["binding"]["fidelity_settings"]["mean_risk_scales"] = scales
            reference["binding"].pop("sha256")
            reference["binding"]["sha256"] = _digest(reference["binding"])
            cfg["optimization"]["multitask"]["fidelity_budget"]["mean_risk_scales"] = scales
        # Keep the complete original SOURCE panel and allowance; only bind the
        # declared positive scale change. No SOURCE inference or refit.
        ref = campaign / "artifacts" / arm / "fidelity_budget_reference.json"
        immutable_json(ref, reference)
        cfg["optimization"]["multitask"]["fidelity_budget"]["reference_artifact"] = {"path": str(ref), "sha256": digest(ref)}
        path = campaign / "configs" / f"{arm}.yaml"
        text = yaml.safe_dump(cfg, sort_keys=False)
        if path.exists() and path.read_text() != text:
            raise ValueError(f"prepared config already differs: {path}")
        path.write_text(text)
        bindings[arm] = {"path": str(path), "sha256": digest(path)}
    immutable_json(campaign / "configs/prepared.json", {
        "schema": "car_round3_prepared_v1", "arms": bindings,
        "source_reference": {"path": str(old_ref_path), "sha256": digest(old_ref_path)},
        "source_snapshot": config["source_run"], "maximum_epoch": 500,
        "warmup_epochs": 50, "review_epoch": 100, "gpu_hour_ceiling": 45,
        "validation_panel": str(historical / "configs/panels/validation.json"),
        "test_locked": True, "deployment_profiling": False,
    })
    print(campaign / "configs/prepared.json")


def freeze(campaign):
    from phycoflow_reconstruction.evaluation.adaptive_campaign import digest, freeze_execution_code, immutable_json
    prepared = json.loads((campaign / "configs/prepared.json").read_text())
    for ref in prepared["arms"].values():
        if digest(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("prepared arm changed")
    code = freeze_execution_code(ROOT, campaign / "operations/code", [
        "scripts/posttraining/run_car_round3.py",
        "scripts/posttraining/assess_car_round3.py",
        "scripts/posttraining/assess_car_round2_checkpoints.py",
        "scripts/posttraining/freeze_car_event.py",
    ])
    from scripts.posttraining.assess_car_round3 import context
    assessment = context(campaign)
    source_refs = {
        name: {"path": str(assessment[key]), "sha256": digest(assessment[key])}
        for name, key in (("validation_panel", "panel_path"),
                          ("source_receipt", "source_receipt"),
                          ("source_baseline", "baseline_path"),
                          ("historical_assessment_freeze", "freeze_path"))
    }
    immutable_json(campaign / "configs/formal_freeze.json", {
        **prepared, "schema": "car_round3_freeze_v1", "code_root": str(code),
        "code_sha256": code.name, "created_unix": time.time(),
        "selection_use": "previously accessed same-trajectory development VALIDATION100",
        "tasks": "reused event and regional CO-U1 task only at the decided endpoint",
        "assessment_sources": source_refs,
    })
    print(code)


def train(args, campaign):
    import yaml
    from phycoflow_reconstruction.evaluation.adaptive_campaign import atomic_json, digest, locked_ledger, spent

    manifest = json.loads((campaign / "configs/formal_freeze.json").read_text())
    if manifest["schema"] != "car_round3_freeze_v1" or args.stop_epoch not in (100, 500):
        raise ValueError("frozen two-path study and declared epoch stop required")
    ref = manifest["arms"][args.arm]
    config_path = Path(ref["path"])
    if digest(config_path) != ref["sha256"]:
        raise ValueError("frozen configuration changed")
    if CODE.resolve() != Path(manifest["code_root"]).resolve():
        raise ValueError("launch from the frozen execution copy")
    cfg = yaml.safe_load(config_path.read_text())
    selected = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not selected or "," in selected or not os.environ.get("KEOPS_CACHE_FOLDER"):
        raise ValueError("one runtime CUDA device and KeOps cache required")
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], text=True)
    matches = [row.split(",") for row in devices.splitlines() if row.split(",")[0].strip() == selected or row.split(",")[1].strip() == selected]
    if len(matches) != 1:
        raise ValueError("runtime device does not resolve")
    device_index, device_uuid, device_name = [x.strip() for x in matches[0]]
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    if any(row.split(",")[0].strip() == device_uuid for row in active.splitlines()):
        raise ValueError("runtime GPU already has a compute owner")
    if args.resume is not None:
        saved = yaml.safe_load((args.resume / "resolved_config.yaml").read_text())
        if saved != cfg:
            raise ValueError("resume must use the run's unchanged saved configuration")
    if args.stop_epoch == 500:
        # Protect the substantive review decision; a recovery checkpoint alone
        # does not authorize bypassing the epoch-100 assessment.
        decision = json.loads((campaign / "operations/learning_review_epoch_100.json").read_text())
        review = decision["assessment"]
        if (args.resume is None or args.arm not in decision["continued_arms"]
                or digest(Path(review["path"])) != review["sha256"]):
            raise ValueError("epoch-500 continuation requires the recorded epoch-100 review")
    ledger = campaign / "operations/gpu_ledger.json"
    start, token = time.time(), uuid.uuid4().hex
    with locked_ledger(ledger) as state:
        if spent(state) + args.max_seconds > CAP_SECONDS:
            raise ValueError("job exceeds remaining additive GPU budget")
        if any(v["gpu_uuid"] == device_uuid for v in state["active"].values()):
            raise ValueError("runtime GPU already leased")
        state["active"][token] = {"gpu_uuid": device_uuid, "phase": args.arm,
                                  "start_unix": start, "pid": os.getpid()}
    launch = {"token": token, "arm": args.arm, "start_unix": start, "pid": os.getpid(),
              "device_index": device_index, "gpu_uuid": device_uuid, "device_name": device_name,
              "max_seconds": args.max_seconds, "stop_epoch": args.stop_epoch,
              "config": ref, "code_root": str(CODE), "platform": platform.platform(),
              "resume": str(args.resume) if args.resume else None}
    atomic_json(campaign / "operations/launches" / f"{token}.json", launch)
    outcome = {"status": "failed"}
    requested_stop = False
    def request_stop(_signal, _frame):
        nonlocal requested_stop
        requested_stop = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        from phycoflow_reconstruction.training.post_training import run_post_training
        def stop(step, _run):
            with locked_ledger(ledger) as state:
                limit = spent(state) >= CAP_SECONDS - ASSESSMENT_RESERVE_SECONDS
            return (step >= 12 * args.stop_epoch or requested_stop or limit
                    or time.time() - start >= args.max_seconds - 300)
        run = run_post_training(cfg, case_dir=ROOT / "cases/turbulent_combustion",
                                resume=args.resume, budget_stop_requested=stop)
        outcome = {**json.loads((run / "status.json").read_text()), "run_dir": str(run)}
        print(run, flush=True)
    except BaseException as exc:
        outcome.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        ended = time.time()
        atomic_json(campaign / "operations/completions" / f"{token}.json",
                    {**launch, **outcome, "end_unix": ended, "seconds": ended-start})
        with locked_ledger(ledger) as state:
            entry = state["active"].pop(token)
            state["closed"].append({**entry, "token": token, "end_unix": ended, "seconds": ended-start})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("freeze")
    run = commands.add_parser("train")
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--stop-epoch", type=int, choices=(100, 500), required=True)
    run.add_argument("--max-seconds", type=float, required=True)
    run.add_argument("--resume", type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    if args.command == "prepare":
        prepare(campaign)
    elif args.command == "freeze":
        freeze(campaign)
    else:
        if not 300 < args.max_seconds <= CAP_SECONDS:
            parser.error("max-seconds must exceed the 300-second save margin and fit the study")
        train(args, campaign)


if __name__ == "__main__":
    main()
