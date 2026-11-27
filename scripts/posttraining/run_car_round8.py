#!/usr/bin/env python3
"""Prepare matched member-MSE Shape arms and run a declared epoch endpoint.

This is a small Round 8 adapter over the existing post-training runner and
Round 8 lease. Preparation makes reviewable configurations only; learning
requires the separately sealed campaign decision and a leased invocation.
Assessment is a separate, once-per-endpoint operation on the matched panel.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import sys
import time

CODE = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("CAR_REPO_ROOT", CODE)).resolve()
sys.path[:0] = [str(CODE / "src"), str(CODE)]
from phycoflow_reconstruction.evaluation.adaptive_campaign import digest
from scripts.posttraining.run_car_round7 import validate_common

BASE = "MemberMSE_BaseShape"
SPATIAL = "MemberMSE_SpatialShape"
ARMS = (BASE, SPATIAL)


def ref(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": digest(path)}


def matched_configs(base, position):
    """Keep the source, field risk, PI, Scale, streams and exposure identical."""
    from phycoflow_reconstruction.config.validate import validate_config
    validate_common(base)
    budget = base["optimization"]["multitask"]["fidelity_budget"]
    if (budget.get("metric") != "physical_member_mse_model_units"
            or budget.get("relative_increase") != .1 or budget.get("absolute_allowance", 0.) != 0.
            or budget.get("reference_member_mse_from_panel") is not True):
        raise ValueError("the matched arms require the declared 10% member-MSE SOURCE reference")
    scalable = base["coherence"]["scalable_anchored_coherence"]
    if scalable.get("scale_objective", "composite") != "composite" or "position_artifact" in scalable:
        raise ValueError("the common base requires the unchanged composite Scale and PH-only Shape")
    configs = {}
    for name in (BASE,) if position is None else ARMS:
        cfg = deepcopy(base)
        cfg["optimization"]["epochs"] = 500
        cfg["runtime"]["plot_every_steps"] = 10
        cfg["checkpointing"].update(every_epochs=25, epochs=[100, 500], panel_milestones_only=True)
        cfg["output"]["experiment_name"] = "coherence_adaptive_repair/round_8/runs/formal/" + name
        if name == SPATIAL:
            cfg["coherence"]["scalable_anchored_coherence"]["position_artifact"] = dict(position)
        validate_config(cfg)
        configs[name] = cfg
    return configs


def prepare(args):
    import torch
    import yaml
    from phycoflow_reconstruction.training.scalable_prefit import attach_position_artifact

    base = yaml.safe_load(args.base_config.read_text())
    settings = base["coherence"]["scalable_anchored_coherence"]
    core_ref = settings["common_core"]
    if digest(Path(core_ref["path"])) != core_ref["sha256"]:
        raise ValueError("common PH core hash changed")
    prefit = settings["common_prefit"]
    if digest(Path(prefit["path"])) != prefit["sha256"]:
        raise ValueError("common prefit hash changed")
    reference = base["optimization"]["multitask"]["fidelity_budget"]["reference_artifact"]
    if digest(Path(reference["path"])) != reference["sha256"]:
        raise ValueError("common TRAIN SOURCE reference hash changed")
    position, fit_ref = None, None
    if args.position_fit_receipt is not None:
        fit = json.loads(args.position_fit_receipt.read_text())
        if (fit.get("status") != "completed" or len(fit.get("sample_ids", ())) != 800
                or fit.get("members") != 4 or fit.get("test_locked") is not True):
            raise ValueError("a completed, TEST-locked TRAIN800 positional fit is required")
        position = fit["artifact"]
        if ref(position["path"]) != position:
            raise ValueError("position observer hash changed")
        if (fit["inputs"]["core"]["sha256"] != settings["common_core"]["sha256"]
                or fit["inputs"]["prefit"]["sha256"] != settings["common_prefit"]["sha256"]):
            raise ValueError("position fit used a different common core or prefit")
        core = torch.load(settings["common_core"]["path"], map_location="cpu", weights_only=True)
        attach_position_artifact(core, position)
        fit_ref = ref(args.position_fit_receipt)
    configs = matched_configs(base, position)
    folder = args.campaign / "configs"
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "conditional_learning_prepared.json"
    paths = {name: folder / (name + ".yaml") for name in configs}
    if manifest_path.exists() or any(path.exists() for path in paths.values()):
        raise FileExistsError("prepared conditional learning configuration already exists")
    arms = {}
    for name, cfg in configs.items():
        with paths[name].open("x", encoding="utf-8") as stream:
            yaml.safe_dump(cfg, stream, sort_keys=False)
        arms[name] = ref(paths[name])
    manifest = {"schema": "car_round8_conditional_learning_prepared_v1", "arms": arms,
        "base_config": ref(args.base_config), "position_fit": fit_ref,
        "reference": base["optimization"]["multitask"]["fidelity_budget"]["reference_artifact"],
        "common_prefit": base["coherence"]["scalable_anchored_coherence"]["common_prefit"],
        "scale_objective": "composite", "initial_review_epoch": 100, "maximum_selected_epoch": 500,
        "maximum_total_substantive_epochs": 600, "steps_per_epoch": 12, "batch_size": 64,
        "assessment": {"split": "validation", "panel": "matched_validation_100",
                       "evaluations_per_endpoint": 1, "test_locked": True},
        "state": "prepared_not_admitted", "test_locked": True}
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(manifest_path)


def train(args):
    import torch
    import yaml
    from phycoflow_reconstruction.training.post_training import (
        prepare_post_training_cuda_workspace, run_post_training, strict_post_training_algorithms,
    )
    from phycoflow_reconstruction.training.run_store import load_project_checkpoint
    from scripts.posttraining.lease_car_round8 import admitted_ceiling, TOTAL_SECONDS

    if not os.environ.get("CAR_ROUND8_LEASE_TOKEN"):
        raise ValueError("training requires an enclosing Round 8 GPU lease")
    decision_path = args.campaign / "operations/learning_decision.json"
    decision = json.loads(decision_path.read_text())
    if admitted_ceiling(decision) != TOTAL_SECONDS:
        raise ValueError("formal learning requires the sealed and forecast campaign decision")
    prepared_path = args.campaign / "configs/conditional_learning_prepared.json"
    if decision.get("prepared_sha256") != digest(prepared_path):
        raise ValueError("sealed decision names a different prepared comparison")
    prepared = json.loads(prepared_path.read_text())
    admitted = decision.get("admitted_arms")
    if (not isinstance(admitted, list) or not admitted
            or any(not isinstance(name, str) for name in admitted)
            or len(set(admitted)) != len(admitted)
            or not set(admitted).issubset(prepared["arms"])
            or args.arm not in admitted):
        raise ValueError("the requested arm must be explicitly admitted from the prepared comparison")
    declaration = prepared["arms"][args.arm]
    path = Path(declaration["path"])
    if ref(path) != declaration:
        raise ValueError("prepared scientific configuration changed")
    cfg = yaml.safe_load(path.read_text())
    cfg["runtime"]["device"] = args.device
    if args.stop_epoch < 1 or args.stop_epoch > 500:
        raise ValueError("the selected trajectory is bounded to 500 reporting epochs")
    if args.stop_epoch > 100:
        review = json.loads((args.campaign / "operations/learning_review_epoch_100.json").read_text())
        if (review.get("status") != "sealed" or review.get("selected_arm") != args.arm
                or review.get("continue_to_epoch", 0) < args.stop_epoch or args.resume is None):
            raise ValueError("continuation requires the selected epoch-100 lineage and sealed review")
    elif args.stop_epoch != 100:
        raise ValueError("the initial learning review occurs at epoch 100")
    if args.resume is not None:
        saved = yaml.safe_load((args.resume / "resolved_config.yaml").read_text())
        saved["runtime"].pop("device", None)
        expected = deepcopy(cfg)
        expected["runtime"].pop("device", None)
        if saved != expected:
            raise ValueError("saved scientific configuration differs from the prepared arm")
        start_step = int(load_project_checkpoint(args.resume / "checkpoints/last.pt")["global_step"])
        if args.stop_epoch > 100 and start_step < 1200:
            raise ValueError("a selected continuation must have completed the epoch-100 review")
        if start_step >= args.stop_epoch * 12:
            raise ValueError("the declared endpoint is already complete")
    if args.result.exists():
        raise FileExistsError(args.result)
    requested = False
    def stop_signal(_sig, _frame):
        nonlocal requested
        requested = True
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    deadline = float(os.environ.get("CAR_ROUND8_LEASE_DEADLINE", "inf"))
    def stop(step, _run):
        return requested or step >= 12 * args.stop_epoch or time.time() >= deadline - 180
    device = torch.device(args.device)
    requested_strict = strict_post_training_algorithms(cfg)
    if device.type == "cuda":
        prepare_post_training_cuda_workspace(cfg)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    run = run_post_training(cfg, case_dir=ROOT / "cases/turbulent_combustion", resume=args.resume,
                            budget_stop_requested=stop)
    memory = ({"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
              if device.type == "cuda" else {})
    status = json.loads((run / "status.json").read_text())
    checkpoint = run / "checkpoints/last.pt"
    step = int(load_project_checkpoint(checkpoint)["global_step"])
    if status.get("global_step") is not None and status["global_step"] != step:
        raise ValueError("terminal status and saved checkpoint differ")
    reached = step >= 12 * args.stop_epoch
    disposition = ("target_reached" if reached else "interrupted" if requested else
                   "budget_paused" if time.time() >= deadline - 180 else "unexpected_stop")
    result = {"schema": "car_round8_training_result_v1", "arm": args.arm, "run_dir": str(run),
        "target_epoch": args.stop_epoch, "global_step": step, "reporting_epoch": step / 12,
        "target_reached": reached, "disposition": disposition, "status": status,
        "prepared_config": declaration, "saved_config": ref(run / "resolved_config.yaml"),
        "checkpoint": ref(checkpoint), "lease_token": os.environ["CAR_ROUND8_LEASE_TOKEN"],
        "assessment": prepared["assessment"] if reached else None, "test_locked": True,
        "numerical_runtime": {"strict_requested": requested_strict,
            "strict_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_deterministic": torch.backends.cudnn.deterministic},
        "torch_memory": {**memory, "scope": "current worker including model/artifact setup and training"}}
    args.result.parent.mkdir(parents=True, exist_ok=True)
    with args.result.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(run, flush=True)
    if disposition == "unexpected_stop":
        raise RuntimeError("training stopped before its endpoint without a signal or lease deadline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--base-config", type=Path, required=True)
    prep.add_argument("--position-fit-receipt", type=Path)
    run = sub.add_parser("train")
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--stop-epoch", type=int, required=True)
    run.add_argument("--device", required=True)
    run.add_argument("--resume", type=Path)
    run.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else train)(args)


if __name__ == "__main__":
    main()
