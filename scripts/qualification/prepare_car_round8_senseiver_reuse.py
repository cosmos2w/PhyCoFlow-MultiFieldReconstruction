#!/usr/bin/env python3
"""Bind a new J1 qualification to its own completed failed-start SOURCE fit.

Only the declared horizon and hash-bound reuse paths change. The reference law,
SOURCE model, support, observations, family definitions, and PI remain intact.
This CPU-only preparation performs no model inference or optimizer action.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

CODE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(CODE / "src"), str(CODE)]


def ref(path):
    from phycoflow_reconstruction.training.run_store import file_sha256
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": file_sha256(path)}


def prepare(args):
    import torch
    import yaml
    from phycoflow_reconstruction.coherence import build_enabled_families
    from phycoflow_reconstruction.config.validate import validate_config
    from phycoflow_reconstruction.training.coherence_sampling import _fit_source_calibrations, _fit_training_references
    from phycoflow_reconstruction.training.post_training import _fidelity_reference_binding, _load_shared_fidelity_reference
    from phycoflow_reconstruction.training.scalable_fitting import _digest, validate_scalable_artifacts
    from phycoflow_reconstruction.training.source import load_source_model, source_hashes

    run = args.failed_run.resolve(strict=True)
    original = yaml.safe_load(args.source_config.read_text())
    saved = yaml.safe_load((run / "resolved_config.yaml").read_text())
    failure = json.loads(args.failure_receipt.read_text())
    log = args.failure_receipt.parent / "continuous.log"
    if (original != saved or original["model"]["name"] != "senseiver"
            or original["optimization"]["epochs"] != 2
            or failure.get("qualified") is not False or failure.get("status") != "failed"
            or failure.get("config_sha256") != ref(args.source_config)["sha256"]
            or "scalable coherence requires at least 10 epochs" not in log.read_text()):
        raise ValueError("matching failed Senseiver calibration-horizon attempt required")
    manifest = json.loads((run / "run_manifest.json").read_text())
    if source_hashes(saved) != manifest["source_hashes"]:
        raise ValueError("original Senseiver SOURCE changed after the failed start")
    core_path, panel_path = (run / "artifacts" / name for name in ("scalable_coherence.pt", "source_panel.pt"))
    reference_path = run / "artifacts/fidelity_budget_reference.json"
    calibration_path = run / "artifacts/coherence_calibration.json"
    if ref(calibration_path)["sha256"] != manifest["coherence_calibration_sha256"]:
        raise ValueError("failed-start family calibration differs from its manifest")
    core = torch.load(core_path, map_location="cpu", weights_only=True)
    panel = torch.load(panel_path, map_location="cpu", weights_only=True)
    validate_scalable_artifacts(core, core["provenance"])
    if (panel.get("schema") != "car_source_panel_v1" or panel.get("provenance") != core["provenance"]
            or panel.get("members_count") != 1 or panel.get("sensor_count") != 256
            or len(panel.get("sample_ids", ())) != 800 or panel["sample_ids"] != core["provenance"]["sample_ids"]
            or panel.get("source_weight_view") != manifest["source_metadata"]["weight_view"]
            or panel.get("source_weight_view_sha256") != manifest["source_metadata"]["materialized_view_sha256"]
            or core["provenance"]["source_checkpoint_sha256"] != manifest["source_hashes"]["checkpoint"]):
        raise ValueError("Senseiver SOURCE panel, J1 law or TRAIN support differs")
    candidate = deepcopy(saved)
    candidate["optimization"]["epochs"] = 10
    candidate["output"]["experiment_name"] = "coherence_adaptive_repair/round_8/runs/qualification/Senseiver_MemberMSE_J1_Qualification_v2"
    families = {}
    for name in candidate["coherence"]["families"]:
        path = run / "artifacts" / f"{name}_family.pt"
        families[name] = ref(path)
        if manifest["coherence_family_state_sha256s"].get(name) != families[name]["sha256"]:
            raise ValueError(f"frozen Senseiver family hash differs: {name}")
    candidate["coherence"]["frozen_family_artifacts"] = families
    candidate["coherence"]["scalable_anchored_coherence"]["common_core"] = {
        "path": str(core_path), "sha256": ref(core_path)["sha256"],
        "source_panel_path": str(panel_path), "source_panel_sha256": ref(panel_path)["sha256"]}
    declaration = ref(reference_path)
    candidate["optimization"]["multitask"]["fidelity_budget"]["reference_artifact"] = declaration
    validate_config(candidate)
    report = json.loads(reference_path.read_text())
    risk_panel = report.get("risk_reference_panel", {})
    if (risk_panel.get("members") != 1 or risk_panel.get("sample_count") != 800
            or risk_panel.get("reference_law") != "training_selected_complement_v1"
            or risk_panel.get("source_member_mse") != report["reference"]["reference_risks"]):
        raise ValueError("saved member-MSE calibration is not the matching Senseiver J1 TRAIN law")
    model, dataset, metadata = load_source_model(candidate, torch.device("cpu"))
    try:
        if (metadata != manifest["source_metadata"] or _digest(model.state_dict()) != panel["parameter_buffer_sha256"]
                or dataset.normalizer.digest() != core["provenance"]["normalizer_sha256"]):
            raise ValueError("loaded SOURCE parameter view or normalizer differs from the saved panel")
        live_families = build_enabled_families(candidate["coherence"], dataset.data_spec, dataset.normalizer)
        _fit_training_references(live_families, dataset, candidate, device=torch.device("cpu"),
                                 structured_grid_required=model.capabilities.structured_grid_required)
        _fit_source_calibrations(live_families, model, dataset, candidate, device=torch.device("cpu"),
                                 source_checkpoint_sha256=manifest["source_hashes"]["checkpoint"])
        binding = _fidelity_reference_binding(candidate, dataset, manifest["source_hashes"], metadata,
            SimpleNamespace(artifacts=core), manifest["coherence_family_scales"])
        if binding != report["binding"]:
            raise ValueError("reference scientific binding changed")
        _load_shared_fidelity_reference(declaration, binding)
    finally:
        dataset.close()
    for path in (args.output_config, args.output_receipt):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_config.open("x") as stream:
        yaml.safe_dump(candidate, stream, sort_keys=False)
    receipt = {"schema": "car_round8_senseiver_j1_reuse_v2", "status": "prepared_not_run",
        "source_config": ref(args.source_config), "failed_run": str(run), "failed_receipt": ref(args.failure_receipt),
        "failed_log": ref(log), "source_checkpoint_sha256": manifest["source_hashes"]["checkpoint"],
        "failed_run_manifest": ref(run / "run_manifest.json"), "core": ref(core_path), "source_panel": ref(panel_path),
        "reference": declaration, "family_artifacts": families, "family_calibration": ref(calibration_path),
        "output_config": ref(args.output_config), "reference_binding_sha256": binding["sha256"],
        "source_sample_count": 800, "source_members": 1, "field_risk": "member_mse",
        "reference_law": "training_selected_complement_v1", "support": "Q4096 plus rotating complement Q512",
        "declared_horizon_epochs": 10, "qualification_pause_epoch": 1, "qualification_endpoint_epoch": 2,
        "scalable_coefficients": "online calibration with unit coefficients; no completed common prefit exists",
        "changes": "ten-epoch declared horizon and hash-bound reuse of this exact Senseiver SOURCE core, panel, family artifacts, and member-MSE/C0 reference",
        "validation": "actual CPU SOURCE parameter/normalizer, frozen family loaders and exact shared reference binding; no SOURCE inference or optimizer update",
        "test_locked": True}
    with args.output_receipt.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.output_config, flush=True)
    print(args.output_receipt, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-config", type=Path, required=True)
    p.add_argument("--failed-run", type=Path, required=True)
    p.add_argument("--failure-receipt", type=Path, required=True)
    p.add_argument("--output-config", type=Path, required=True)
    p.add_argument("--output-receipt", type=Path, required=True)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
