#!/usr/bin/env python3
"""Fit common full-target SOURCE-gradient coefficients from a prepared run."""

import argparse
import json
import math
import os
import platform
import signal
import sys
import time
import uuid
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / "src"))

# KeOps initializes its CUDA build folder during imports, before its own
# directory creation. The caller chooses both cache root and visible device.
if os.environ.get("KEOPS_CACHE_FOLDER") and os.environ.get("CUDA_VISIBLE_DEVICES"):
    build = "_".join(platform.uname()[:3]) + f"_p{sys.version.split()[0]}"
    (Path(os.environ["KEOPS_CACHE_FOLDER"]) /
     f"{build}_CUDA_VISIBLE_DEVICES_{os.environ['CUDA_VISIBLE_DEVICES']}").mkdir(parents=True, exist_ok=True)

def fit(args, start, deadline):
    from copy import deepcopy

    import torch
    import yaml

    from phycoflow_reconstruction.coherence import build_enabled_families
    from phycoflow_reconstruction.models.source_residual import FrozenSourceResidual
    from phycoflow_reconstruction.training.fidelity_budget import ExternalPolicy, _budget_from_state
    from phycoflow_reconstruction.training.multitask import OuterAggregator
    from phycoflow_reconstruction.training.post_training import _scheduled_model_lr
    from phycoflow_reconstruction.training.round2_prefit import fit_round2_common_prefit
    from phycoflow_reconstruction.training.round2_virtual_probe import actual_scale_panel
    from phycoflow_reconstruction.training.run_store import file_sha256
    from phycoflow_reconstruction.training.scalable_fitting import (
        _adaptive_input_binding,
        _digest,
        _identity,
    )
    from phycoflow_reconstruction.training.scalable_runtime import ScalableRuntime
    from phycoflow_reconstruction.training.source import load_source_model
    from phycoflow_reconstruction.utils.reproducibility import seed_everything

    run = args.prepared_run.resolve()
    config = yaml.safe_load((run / "resolved_config.yaml").read_text())
    device = torch.device(args.device)
    seed_everything(int(config["runtime"].get("seed", 42)), bool(config["runtime"].get("deterministic", True)))
    model, dataset, identity = load_source_model(config, device)
    if model.capabilities.generative:
        config["optimization"].setdefault("rollout_execution", {})["checkpointing"] = True
        model.rollout_checkpointing = True
    residual_config = yaml.safe_load(args.residual_config.read_text())
    residual_settings = residual_config["residual"]
    if not residual_settings.get("enabled"):
        raise ValueError("residual probe configuration must enable the residual")
    if model.capabilities.generative and residual_config["optimization"].get("rollout_execution", {}).get("checkpointing") is not True:
        raise ValueError("FFM residual probe config must declare rollout checkpointing")
    wrapper_rng = torch.get_rng_state()
    wrapper_cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    try:
        families = build_enabled_families(config["coherence"], dataset.data_spec, dataset.normalizer)
        manifest = json.loads((run / "run_manifest.json").read_text())
        expected = manifest["coherence_family_state_sha256s"]
        for name, family in families.items():
            path = run / "artifacts" / f"{name}_family.pt"
            if file_sha256(path) != expected[name]:
                raise ValueError(f"prepared family artifact hash mismatch: {name}")
            family.to(device).load_state_artifact(torch.load(path, map_location=device, weights_only=True))
        path = run / "artifacts" / "scalable_coherence.pt"
        panel_path = run / "artifacts" / "source_panel.pt"
        artifacts = torch.load(path, map_location="cpu", weights_only=True)
        panel = torch.load(panel_path, map_location="cpu", weights_only=True)
        bank = json.loads(args.bank.read_text())
        weights = panel["masses"].to(panel["reference"])
        binding = _adaptive_input_binding(source=panel["members"], reference=panel["reference"],
                                          coordinates=panel["coordinates"], source_weights=weights,
                                          reference_weights=weights, identity=_identity(panel["provenance"]),
                                          core=artifacts)
        if bank.get("source_fit_binding_sha256") != binding:
            raise ValueError("candidate bank does not bind the prepared SOURCE panel and core")
        artifacts["adaptive_bank"] = bank
        artifacts["sha256"] = _digest({key: value for key, value in artifacts.items() if key != "sha256"})
        if panel["parameter_buffer_sha256"] != _digest(model.state_dict()):
            raise ValueError("prepared panel does not match loaded SOURCE parameters and buffers")
        runtime = ScalableRuntime(artifacts, families, steps_per_epoch=12,
                                  expected_provenance=artifacts["provenance"],
                                  scoring_mode="conditional" if model.capabilities.generative else "deterministic")
        result = fit_round2_common_prefit(runtime, model, dataset, config, panel,
                                          device=device, seed=args.seed, deadline_unix=deadline)
        result["inputs"] = {"source_identity": identity, "source_panel_sha256": file_sha256(panel_path),
                            "scalable_artifact_sha256": file_sha256(path),
                            "bank_sha256": file_sha256(args.bank),
                            "script_sha256": file_sha256(Path(__file__)),
                            "family_sha256s": expected,
                            "rollout_execution": config["optimization"].get("rollout_execution", {})}
        interim = args.output.with_suffix(args.output.suffix + ".unprobed.json")
        with interim.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        reference_report = json.loads((run / "artifacts" / "fidelity_budget_reference.json").read_text())
        reference = _budget_from_state(reference_report["reference"])
        family_scales = json.loads((run / "artifacts" / "coherence_calibration.json").read_text())["resolved_scales"]
        source = deepcopy(model).requires_grad_(False).eval()
        torch.set_rng_state(wrapper_rng)
        if wrapper_cuda_rng is not None:
            torch.cuda.set_rng_state(wrapper_cuda_rng, device)
        residual = FrozenSourceResidual(deepcopy(model), coordinate_dim=int(residual_settings["coordinate_dim"]),
                                        field_scales=residual_settings["field_scales"], width=int(residual_settings["width"]),
                                        max_fraction=float(residual_settings["max_fraction"])).to(device)
        external = config["optimization"]["multitask"]["external_policy"]
        aggregation = config["optimization"]["multitask"]["coherence_aggregation"]
        def optimizer_factory(kind, candidate):
            settings = config["optimization"] if kind == "full_model" else residual_config["optimization"]
            scheduled_lr = _scheduled_model_lr(settings, 0)
            return torch.optim.AdamW([p for p in candidate.parameters() if p.requires_grad],
                                     lr=float(settings["lr"]) if scheduled_lr is None else scheduled_lr,
                                     weight_decay=float(settings.get("weight_decay", 0.0)))
        probes = actual_scale_panel(runtime=runtime, source=source, full_model=model, residual_model=residual,
                                    dataset=dataset, config=config, panel=panel, reference=reference,
                                    controller_factory=lambda: ExternalPolicy(external, reference, device),
                                    optimizer_factory=optimizer_factory,
                                    aggregator_factory=lambda: OuterAggregator(aggregation),
                                    output_dir=args.probe_output_dir, seeds=(204701, 204702, 204703),
                                    zero_tolerance=args.sensor_zero_tolerance, device=device,
                                    family_scales=family_scales, deadline_unix=deadline)
        ratio = probes["decision"]["chosen_ratio"]
        result["scale_joint_ratio"] = ratio
        result["coefficients"]["scale.joint"] *= ratio
        result["scale_probe"] = probes
        result["inputs"]["residual_config_sha256"] = file_sha256(args.residual_config)
        result["inputs"]["fidelity_reference_sha256"] = file_sha256(run / "artifacts" / "fidelity_budget_reference.json")
        result["inputs"]["sensor_zero_tolerance"] = args.sensor_zero_tolerance
        result["timer"] = {"start_unix": start, "end_unix": time.time(),
                           "device": str(device)}
        result["timer"]["wall_seconds"] = result["timer"]["end_unix"] - start
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    finally:
        dataset.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--residual-config", type=Path, required=True)
    parser.add_argument("--probe-output-dir", type=Path, required=True)
    parser.add_argument("--sensor-zero-tolerance", type=float, nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1129001)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 0:
        raise ValueError("a finite positive job time bound is required")
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".receipt.json").exists():
        raise FileExistsError("prefit output or job receipt already exists")
    if args.device not in {"cpu", "cuda:0"}:
        raise ValueError("device must be cpu or cuda:0")
    is_cuda = args.device == "cuda:0"
    from phycoflow_reconstruction.evaluation.adaptive_campaign import locked_ledger, spent

    start, token = time.time(), str(uuid.uuid4())
    if is_cuda:
        if args.device != "cuda:0" or not args.gpu_uuid or not args.ledger:
            raise ValueError("GPU job requires cuda:0, physical GPU UUID, and campaign ledger")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
            raise ValueError("visible device must be restricted to the declared physical GPU UUID")
        with locked_ledger(args.ledger) as state:
            if any(x["gpu_uuid"] == args.gpu_uuid for x in state["active"].values()):
                raise ValueError("GPU is already leased by the campaign")
            if spent(state) + args.max_wall_seconds >= (130 - 6) * 3600:
                raise ValueError("prefit exceeds available GPU budget with closeout reserve")
            state["active"][token] = {"gpu_uuid": args.gpu_uuid, "phase": "common_prefit",
                                      "start_unix": start}
    receipt = {"schema": "car_round2_prefit_job_v1", "token": token, "pid": os.getpid(),
               "device": args.device, "gpu_uuid": args.gpu_uuid, "start_unix": start,
               "max_wall_seconds": args.max_wall_seconds, "status": "running"}
    previous_alarm = signal.getsignal(signal.SIGALRM)
    def timeout(_signal, _frame):
        raise TimeoutError("common prefit exceeded its admitted wall-time bound")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(max(1, math.ceil(args.max_wall_seconds)))
    try:
        fit(args, start, start + args.max_wall_seconds)
        receipt["status"] = "completed"
    except BaseException as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
        torch = sys.modules.get("torch")
        if is_cuda and torch is not None and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        receipt["end_unix"] = time.time()
        receipt["wall_seconds"] = receipt["end_unix"] - start
        if is_cuda:
            with locked_ledger(args.ledger) as state:
                entry = state["active"].pop(token)
                state["closed"].append({**entry, "token": token, "end_unix": receipt["end_unix"],
                                        "seconds": receipt["wall_seconds"]})
        with args.output.with_suffix(args.output.suffix + ".receipt.json").open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
