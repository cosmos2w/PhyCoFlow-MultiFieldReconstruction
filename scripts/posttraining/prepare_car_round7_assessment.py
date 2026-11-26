#!/usr/bin/env python3
"""Bind one Round 7 checkpoint to the existing matched TRAIN/VALIDATION panels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(CODE / "src"), str(CODE)]
import yaml

from scripts.posttraining.assess_car_round7 import checked, ref, validate_spec


def prepare(args):
    if args.checkpoint and (args.reporting_epoch is None or args.reporting_epoch < 1):
        raise ValueError("a child checkpoint requires its explicit reporting epoch")
    if not args.checkpoint and args.reporting_epoch not in (None, 0):
        raise ValueError("SOURCE has reporting epoch zero")
    campaign = args.campaign.resolve(strict=True)
    historical = campaign.parent / "round_2"
    freeze = json.loads((historical / "assessment/early_closeout/freeze/freeze.json").read_text())
    artifact = freeze["artifacts"]["ffm"]
    contract = json.loads(checked(artifact["contract"]).read_text())
    config = yaml.safe_load(args.config.read_text())
    reference = json.loads(args.reference.read_text())
    if reference["binding"]["fidelity_settings"].get("reference_law") != "training_selected_complement_v1":
        raise ValueError("the new common training-law reference is required")
    core_run = Path(artifact["core_run"])
    panel = (historical / "configs/panels/train.json" if args.split == "train"
             else checked(freeze["validation_panel"]))
    source = (historical / "assessment/source_train_ffm_v1/receipt.json" if args.split == "train"
              else historical / "assessment/early_closeout/validation/source_ffm/receipt.json")
    scalable = config["coherence"]["scalable_anchored_coherence"]
    spec = {
        "schema": "car_round7_assessment_spec_v1", "split": args.split,
        "campaign_root": str(campaign), "panel": ref(panel),
        "source_native_receipt": ref(source), "run_config": ref(args.config),
        "checkpoint": ref(args.checkpoint) if args.checkpoint else None,
        "reporting_epoch": args.reporting_epoch if args.checkpoint else 0,
        "core": artifact["core"], "core_config": ref(core_run / "resolved_config.yaml"),
        "core_manifest": ref(core_run / "run_manifest.json"),
        "expanded_bank": scalable["adaptive"]["bank_artifact"],
        "prefit": scalable["common_prefit"], "assessment_contract": artifact["contract"],
        "source_checkpoint": artifact["source_checkpoint"],
        "native_allowance_model": reference["reference"]["absolute_allowance"],
        "training_law_reference": ref(args.reference),
        "field_units": config["dataset"]["field_units"],
        "physical_mask": "All valid 40,300 native rectilinear sites, native physical area weights; original declared field units.",
        "event": freeze["event"], "regional_task": freeze["reserved_task"],
        "task_normalizers": {"event": contract["normalizers"]["event"],
                             "regional": contract["reserved_task_normalizer"]},
        "test_locked": True,
    }
    validate_spec(spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(spec, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reporting-epoch", type=int)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
