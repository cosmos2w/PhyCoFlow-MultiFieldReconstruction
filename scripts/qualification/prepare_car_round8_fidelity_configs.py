#!/usr/bin/env python3
"""Stage named real-adapter Round 8 fidelity qualifications without inference.

The J3 observer is a separately identified copy of the J2 observer: only the
State and Scale scorers' declared independent draw counts change, and saved tensor
state, geometry, normalization and fitted scales are verified identical.
The FFM SOURCE panel remains the original immutable artifact. Senseiver fits
its own SOURCE-dependent scalable and member-risk references when launched.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / "src"))
import torch
import yaml
from phycoflow_reconstruction.config.validate import validate_config
from phycoflow_reconstruction.training.run_store import file_sha256
from phycoflow_reconstruction.training.scalable_fitting import _digest


def write_yaml(path, value):
    with path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--senseiver-source", type=Path, required=True)
    parser.add_argument("--position-fit-receipt", type=Path,
                        help="completed TRAIN800 position fit for spatial J2 recovery")
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True,
                        help="absolute final Round 8 campaign path")
    args = parser.parse_args()
    if not args.campaign.is_absolute():
        raise ValueError("campaign must be absolute")
    args.staging.mkdir(parents=True, exist_ok=False)
    configs = args.staging / "configs"
    configs.mkdir()
    artifact_dir = args.staging / "artifacts/FFM_J3_B32_Observer_Qualification_v1/artifacts"
    artifact_dir.mkdir(parents=True)
    final_dir = args.campaign / "artifacts/FFM_J3_B32_Observer_Qualification_v1"
    original = yaml.safe_load(args.source_config.read_text())
    j2 = deepcopy(original)
    budget = j2["optimization"]["multitask"]["fidelity_budget"]
    budget.pop("mean_risk_scales", None)
    budget.pop("member_allowance_fraction", None)
    budget.update(metric="physical_member_mse_model_units", relative_increase=0.10,
                  absolute_allowance=0.0, reference_member_mse_from_panel=True)
    j2["output"]["experiment_name"] = "coherence_adaptive_repair/round_8/runs/qualification/FFM_MemberMSE_J2_Qualification"
    j2["runtime"]["plot_every_steps"] = 10
    j3 = deepcopy(j2)
    j3["coherence"]["families"]["global_distribution"]["conditional_draws"] = 3
    j3["coherence"]["families"]["multiscale_spectrum"]["objective"]["conditional_draws"] = 3
    j3["optimization"].update(batch_size=32)
    j3["coherence"]["compute_budget"]["batch_size"] = 32
    j3["optimization"]["multitask"]["coherence_aggregation"]["vjp_chunk_size"] = 1
    j3["optimization"]["multitask"]["fidelity_budget"].update(
        reference_member_count_reuse=True)
    j3["output"]["experiment_name"] = "coherence_adaptive_repair/round_8/runs/qualification/FFM_MemberMSE_J3_B32_Qualification"

    core = original["coherence"]["scalable_anchored_coherence"]["common_core"]
    old_core = Path(core["path"])
    if file_sha256(old_core) != core["sha256"] or file_sha256(Path(core["source_panel_path"])) != core["source_panel_sha256"]:
        raise ValueError("source core or SOURCE panel hash mismatch")
    shutil.copyfile(old_core, artifact_dir / "scalable_coherence.pt")
    family_hashes, source_families, state_digests = {}, {}, {}
    for name, declaration in original["coherence"]["frozen_family_artifacts"].items():
        source = Path(declaration["path"])
        if file_sha256(source) != declaration["sha256"]:
            raise ValueError(f"source family hash mismatch: {name}")
        target = artifact_dir / f"{name}_family.pt"
        source_families[name] = declaration["sha256"]
        if name in {"global_distribution", "multiscale_spectrum"}:
            artifact = torch.load(source, map_location="cpu", weights_only=True)
            before = _digest(artifact["state_dict"])
            count_settings = artifact["config"] if name == "global_distribution" else artifact["config"]["objective"]
            if count_settings["conditional_draws"] != 2:
                raise ValueError(f"expected original J2 {name} family")
            count_settings["conditional_draws"] = 3
            if name == "global_distribution":
                if artifact.get("conditional_draws") != 2:
                    raise ValueError("expected original J2 State artifact draw metadata")
                artifact["conditional_draws"] = 3
            torch.save(artifact, target)
            after = torch.load(target, map_location="cpu", weights_only=True)
            if _digest(after["state_dict"]) != before:
                raise ValueError(f"derived {name} observer tensor state changed")
            state_digests[name] = before
            j3["coherence"]["frozen_family_artifacts"][name] = {
                "path": str(final_dir / "artifacts" / target.name), "sha256": file_sha256(target)}
        else:
            shutil.copyfile(source, target)
        family_hashes[name] = file_sha256(target)
    old_parent = yaml.safe_load((old_core.parent.parent / "resolved_config.yaml").read_text())
    if (old_parent["coherence"]["families"] != original["coherence"]["families"] or
            old_parent["coherence"]["compute_budget"] != original["coherence"]["compute_budget"]):
        raise ValueError("source observer family or support differs")
    old_parent["coherence"]["families"] = deepcopy(j3["coherence"]["families"])
    old_parent["coherence"]["compute_budget"] = deepcopy(j3["coherence"]["compute_budget"])
    write_yaml(artifact_dir.parent / "resolved_config.yaml", old_parent)
    (artifact_dir.parent / "run_manifest.json").write_text(json.dumps({
        "coherence_family_state_sha256s": family_hashes}, indent=2) + "\n")
    j3_core = j3["coherence"]["scalable_anchored_coherence"]["common_core"]
    j3_core["path"] = str(final_dir / "artifacts/scalable_coherence.pt")
    j3_core["sha256"] = file_sha256(artifact_dir / "scalable_coherence.pt")
    receipt = {"schema": "car_round8_j3_observer_derivation_v1",
        "source_config_sha256": file_sha256(args.source_config),
        "source_core_sha256": core["sha256"], "source_panel_sha256": core["source_panel_sha256"],
        "source_family_sha256": source_families, "derived_family_sha256": family_hashes,
        "source_member_count": 2, "declared_training_member_count": 3,
        "state_tensor_digests": state_digests, "source_batch_size": 64, "training_batch_size": 32,
        "change": "State and Scale scorer conditional_draws 2 to 3; execution batch 64 to 32; frozen observer tensors, support and scales unchanged"}
    (artifact_dir.parent / "derivation.json").write_text(json.dumps(receipt, indent=2) + "\n")

    senseiver = deepcopy(j2)
    source_config = yaml.safe_load((args.senseiver_source / "resolved_config.yaml").read_text())
    if source_config["model"]["name"] != "senseiver":
        raise ValueError("expected an existing Senseiver source")
    senseiver["source_run"] = str(args.senseiver_source.resolve())
    senseiver["source"] = {"kind": "native_run", "weight_view": "live", "allow_integration_source": False}
    senseiver["model"] = source_config["model"]
    # The qualification driver truncates at epochs 1/2. The declared training
    # horizon must still accommodate scalable coherence's ten-epoch calibration.
    senseiver["optimization"].update(epochs=10, batch_size=8, steps_per_epoch=1)
    senseiver["optimization"].pop("rollout_execution", None)
    senseiver["optimization"]["multitask"]["coherence_aggregation"] = {"method": "torchjd_config"}
    senseiver["optimization"]["multitask"]["external_policy"]["lagrangian"]["epoch_calibrated_pi"]["steps_per_epoch"] = 1
    senseiver_budget = senseiver["optimization"]["multitask"]["fidelity_budget"]
    senseiver_budget.pop("reference_artifact", None)
    senseiver_budget.pop("reference_member_mse_from_panel", None)
    senseiver["coherence"]["compute_budget"]["batch_size"] = 8
    scalable = senseiver["coherence"]["scalable_anchored_coherence"]
    for key in ("common_core", "common_prefit", "allocation"):
        scalable.pop(key, None)
    scalable.update(source_members=1, source_sensor_field="T", adaptive={"enabled": False},
                    condition_query_groups=1, save_source_panel=True)
    senseiver["checkpointing"] = {"enabled": True, "every_epochs": 1, "save_epoch_one": True,
        "save_epoch_zero": True, "selection_metric": "reconstruction_mse"}
    senseiver["output"]["experiment_name"] = "coherence_adaptive_repair/round_8/runs/qualification/Senseiver_MemberMSE_J1_Qualification"
    configs_to_write = [("FFM_MemberMSE_J2_Qualification", j2),
                        ("FFM_MemberMSE_J3_B32_Qualification", j3),
                        ("Senseiver_MemberMSE_J1_Qualification", senseiver)]
    if args.position_fit_receipt is not None:
        from phycoflow_reconstruction.training.scalable_prefit import attach_position_artifact
        fit = json.loads(args.position_fit_receipt.read_text())
        if (fit.get("status") != "completed" or len(fit.get("sample_ids", ())) != 800
                or fit.get("members") != 4 or fit.get("test_locked") is not True):
            raise ValueError("completed TRAIN800 position fitting required")
        if (fit["inputs"]["core"]["sha256"] != core["sha256"] or
                fit["inputs"]["prefit"]["sha256"] != j2["coherence"]["scalable_anchored_coherence"]["common_prefit"]["sha256"]):
            raise ValueError("position fit uses different core or prefit")
        attach_position_artifact(torch.load(core["path"], map_location="cpu", weights_only=True), fit["artifact"])
        spatial = deepcopy(j2)
        spatial["coherence"]["scalable_anchored_coherence"]["position_artifact"] = fit["artifact"]
        spatial["output"]["experiment_name"] = "coherence_adaptive_repair/round_8/runs/qualification/FFM_MemberMSE_SpatialShape_J2_Qualification"
        configs_to_write.append(("FFM_MemberMSE_SpatialShape_J2_Qualification", spatial))
    for name, config in configs_to_write:
        validate_config(config)
        write_yaml(configs / f"{name}.yaml", config)
    print(json.dumps({"status": "staged", "path": str(args.staging),
        "configs": [path.name for path in configs.iterdir()], "derivation": receipt}, indent=2))


if __name__ == "__main__":
    main()
