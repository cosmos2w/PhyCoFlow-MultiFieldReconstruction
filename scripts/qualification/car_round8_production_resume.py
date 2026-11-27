#!/usr/bin/env python3
"""Qualify a declared real-adapter member-MSE action and fresh-process resume.

The supplied resolved configuration owns its SOURCE, field law, observer and
reporting-epoch clock. Two independent runs cover two epochs each, with one
saved and resumed at epoch one. No validation/TEST assessment is requested.
Run inside the Round 8 preparation lease with a case-owned output directory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("CAR_REPO_ROOT", CODE)).resolve()
sys.path[:0] = [str(CODE / "src"), str(CODE)]
from phycoflow_reconstruction.evaluation.adaptive_campaign import atomic_json, digest
from scripts.posttraining.qualify_car_round2_resume import (
    differences, rows, scientific_checkpoint_view, scientific_history_view,
)


def validate_qualification_horizon(cfg):
    """Reject a non-truncated scalable horizon before expensive SOURCE fitting."""
    epochs = int(cfg["optimization"]["epochs"])
    if epochs < 2:
        raise ValueError("qualification requires at least two declared epochs")
    if cfg["coherence"].get("scalable_anchored_coherence", {}).get("enabled") and epochs < 10:
        raise ValueError("scalable qualification requires a declared horizon of at least 10 epochs; its worker stops at epochs 1 and 2")


def j1_action_evidence(action_rows, checkpoint):
    if any(row.get("update_mode") != "coherence_config_lagrangian_improve"
           or row.get("family_gradient_evaluations") != 3 or not row.get("update_accepted")
           or row.get("field_signed_mean_risk") is not None for row in action_rows):
        raise ValueError("J1 qualification requires accepted fresh three-family parameter-space ConFIG actions")
    norms = [row.get("constraint_gradient_norm") for row in action_rows]
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in norms):
        raise ValueError("J1 weighted-field gradient norms must be recorded and finite")
    calibration = checkpoint["scalable_runtime"]["calibration"]
    frozen = calibration.get("coefficients") is not None or calibration.get("version") == "internal_prefit_state_v1"
    return {"fresh_config_actions": len(action_rows), "weighted_field_gradient_norms": norms,
        "active_field_actions": sum(value > 0 for value in norms), "calibration_frozen": frozen,
        "scope": "two real optimizer actions during online unit-coefficient calibration" if not frozen
                 else "two real optimizer actions with a frozen declared calibration"}


def worker(args):
    import torch
    import yaml
    from phycoflow_reconstruction.training.post_training import (
        prepare_post_training_cuda_workspace, run_post_training, strict_post_training_algorithms,
    )

    torch.set_num_threads(4)
    path = args.resume / "resolved_config.yaml" if args.resume else args.config
    cfg = yaml.safe_load(path.read_text())
    validate_qualification_horizon(cfg)
    settings = cfg["optimization"]["multitask"]["fidelity_budget"]
    pi = cfg["optimization"]["multitask"]["external_policy"]["lagrangian"].get("epoch_calibrated_pi")
    if (settings.get("metric") != "physical_member_mse_model_units" or
            settings.get("rotating_complement") is None or pi is None):
        raise ValueError("qualification requires member-MSE rotating feedback and the PI policy")
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    if args.resume is None:
        cfg["output"]["experiment_name"] = f"coherence_adaptive_repair/round_8/runs/qualification/{args.name}"
    device = torch.device(cfg["runtime"]["device"])
    requested_strict = strict_post_training_algorithms(cfg)
    if device.type == "cuda":
        prepare_post_training_cuda_workspace(cfg)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    run = run_post_training(cfg, case_dir=ROOT / "cases/turbulent_combustion", resume=args.resume,
        budget_stop_requested=lambda step, _run: step >= args.stop_action)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    atomic_json(args.output, {"run_dir": str(run), "pid": os.getpid(), "config": str(path),
        "resume": str(args.resume) if args.resume else None, "stop_action": args.stop_action,
        "start_unix": started, "end_unix": time.time(), "peak_allocated_bytes": peak,
        "peak_scope": "entire fresh worker, including model/artifact setup and training",
        "numerical_runtime": {"strict_requested": requested_strict,
            "strict_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_deterministic": torch.backends.cudnn.deterministic}})


def compare_epoch_one(uninterrupted, resumed, *, steps_per_epoch):
    """Retain optional immutable-boundary evidence even if the final check fails."""
    import torch
    paths = [Path(run) / "checkpoints/epoch_001.pt" for run in (uninterrupted, resumed)]
    present = [path.is_file() for path in paths]
    if not all(present):
        return {"status": "unavailable", "present": present, "paths": [str(path) for path in paths],
                "reason": "explicit epoch-one milestones were not saved in both runs"}
    states = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    steps = [state.get("global_step") for state in states]
    # Regular checkpoint-manager saves carry selection metadata absent from a
    # budget-pause save. It does not affect the optimizer/model/RNG trajectory.
    metadata = ("checkpoint_metric", "best_metric_name", "best_metric_value")
    views = [scientific_checkpoint_view(state) for state in states]
    for view in views:
        for key in metadata:
            view.pop(key, None)
    diff = differences(*views, path="epoch_one")
    if steps != [steps_per_epoch, steps_per_epoch]:
        diff.append(f"epoch_one/global_step: expected {steps_per_epoch}, got {steps}")
    return {"status": "different" if diff else "matched", "global_steps": steps,
            "checkpoints": [{"path": str(path), "sha256": digest(path)} for path in paths],
            "comparison_atol": 1e-7, "comparison_rtol": 1e-6,
            "excluded_checkpoint_manager_metadata": list(metadata), "differences": diff}


def compare(uninterrupted, resumed, *, steps_per_epoch, members):
    import torch
    checkpoints = [torch.load(path / "checkpoints/last.pt", map_location="cpu", weights_only=False)
                   for path in (uninterrupted, resumed)]
    expected_steps = 2 * steps_per_epoch
    if any(ck["global_step"] != expected_steps for ck in checkpoints):
        raise ValueError("both trajectories must end after two reporting epochs")
    diff = differences(*(scientific_checkpoint_view(ck) for ck in checkpoints))
    definitions = [json.loads((path / "logs/recording_definitions.json").read_text())
                   for path in (uninterrupted, resumed)]
    diff += differences(*definitions, path="recording_definitions")
    histories = [scientific_history_view(rows(path / "logs/history.jsonl"), definition)
                 for path, definition in zip((uninterrupted, resumed), definitions, strict=True)]
    references = [json.loads((path / "artifacts/fidelity_budget_reference.json").read_text())
                  for path in (uninterrupted, resumed)]
    for reference in references:
        reference.pop("calibration_seconds", None)
    diff += differences(*references, path="scientific_reference")
    # Independently fitted calibration files have different elapsed-time bytes.
    # Their full scientific content was compared above; retain both file hashes
    # in the receipt while excluding only that operational hash from the rows.
    for history in histories:
        for row in history:
            if isinstance(row.get("round7_executor"), dict):
                row["round7_executor"].pop("field_reference_sha256", None)
    diff += differences(*histories, path="history")
    for history in histories:
        if len(history) != 2 or any(row.get("fidelity_estimator", {}).get("members") != members or
                "field_member_mse" not in row or "field_signed_mean_risk" not in row for row in history):
            raise ValueError("history must identify the genuine member law and both risk diagnostics")
    for relative in ("logs/epoch_calibrated_pi_updates.jsonl", "metrics/training_coupling.jsonl"):
        values = [rows(path / relative) for path in (uninterrupted, resumed)]
        if any([row["step"] for row in value] != list(range(1, expected_steps + 1)) for value in values):
            raise ValueError(f"contiguous per-action append history required: {relative}")
        if relative.startswith("metrics/") and any(len(row["member_seeds"]) != members for value in values for row in value):
            raise ValueError("endpoint member count differs from the declared qualification")
        diff += differences(*values, path=relative)
    for path in (uninterrupted, resumed):
        for name in ("loss_history.png", "loss_history.pdf"):
            if not (path / "logs" / name).is_file():
                raise ValueError(f"missing pause/end curve: {path / 'logs' / name}")
    if diff:
        raise ValueError("production fresh-process resume differs: " + "; ".join(diff[:20]))
    result = {"uninterrupted": str(uninterrupted), "resumed": str(resumed),
        "checkpoint_sha256": [digest(path / "checkpoints/last.pt") for path in (uninterrupted, resumed)],
        "reference_sha256": [digest(path / "artifacts/fidelity_budget_reference.json") for path in (uninterrupted, resumed)],
        "recording_definitions_sha256": [digest(path / "logs/recording_definitions.json") for path in (uninterrupted, resumed)]}
    if members == 1 and steps_per_epoch == 1:
        result["j1_action_evidence"] = j1_action_evidence(histories[0], checkpoints[0])
    return result


def orchestrate(args):
    import yaml
    if not os.environ.get("CAR_ROUND8_LEASE_TOKEN"):
        raise ValueError("run under the Round 8 preparation lease")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]+", args.name):
        raise ValueError("use a descriptive qualification name")
    cfg = yaml.safe_load(args.config.read_text())
    validate_qualification_horizon(cfg)
    steps = cfg["optimization"]["multitask"]["external_policy"]["lagrangian"]["epoch_calibrated_pi"]["steps_per_epoch"]
    if type(steps) is not int or steps < 1 or int(cfg["optimization"]["epochs"]) < 2:
        raise ValueError("qualification requires at least two epochs and a positive resolved epoch clock")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    receipt = {"schema": "car_round8_production_resume_v1", "status": "failed",
        "config_sha256": digest(args.config), "members": args.members, "start_unix": started}
    paths = []
    try:
        for name, stop, resume in (("continuous", 2 * steps, False), ("pause", steps, False),
                                   ("resume", 2 * steps, True)):
            output = args.output / f"{name}.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--name", args.name,
                "--config", str(args.config), "--output", str(output), "--members", str(args.members),
                "--stop-action", str(stop)]
            if args.device is not None:
                command += ["--device", args.device]
            if resume:
                command += ["--resume", paths[-1]["run_dir"]]
            with (args.output / f"{name}.log").open("x") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            paths.append(json.loads(output.read_text()))
        if len({item["pid"] for item in paths}) != 3 or paths[2]["run_dir"] != paths[1]["run_dir"]:
            raise ValueError("three fresh processes and actual saved-run resume required")
        receipt["epoch_one_boundary"] = compare_epoch_one(
            Path(paths[0]["run_dir"]), Path(paths[2]["run_dir"]), steps_per_epoch=steps)
        receipt.update(compare(Path(paths[0]["run_dir"]), Path(paths[2]["run_dir"]),
                               steps_per_epoch=steps, members=args.members))
        evidence = receipt.get("j1_action_evidence")
        qualified = ((evidence is None or evidence["active_field_actions"] > 0)
                     and receipt["epoch_one_boundary"]["status"] != "different")
        receipt.update(status="completed", qualified=qualified, pause_reporting_epoch=1,
            final_reporting_epoch=2, steps_per_epoch=steps, saved_config_reloaded=True,
            device=args.device, test_locked=True, process_records=paths,
            comparison_atol=1e-7, comparison_rtol=1e-6)
        if not qualified:
            receipt["qualification_limit"] = (
                "explicit epoch-one scientific checkpoint states differed"
                if receipt["epoch_one_boundary"]["status"] == "different" else
                "save/resume and fresh ConFIG matched, but neither action had a nonzero weighted-field gradient")
    except BaseException as exc:
        receipt.update(error_type=type(exc).__name__, error=str(exc), qualified=False)
        raise
    finally:
        receipt.update(end_unix=time.time(), wall_seconds=time.time() - started)
        atomic_json(args.output / "receipt.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--members", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--device", help="runtime device override")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--stop-action", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    (worker if args.worker else orchestrate)(args)


if __name__ == "__main__":
    main()
