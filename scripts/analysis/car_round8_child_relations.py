#!/usr/bin/env python3
"""Add State A/R/S and signed Scale context for one cached child endpoint.

The matched SOURCE quantities are reused from the sealed Round 8 readout.
Only the child's separately queried selected-support rows are read and scored;
there is no inference, persistence computation, native-support analysis or TEST.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np
import torch

from scripts.analysis.car_round8_cached_summary import checked_ref, paired_interval, ref
from scripts.analysis.car_round8_support_readout import (
    build_state_scale, direct_readout_metadata, scale_context, state_triplet,
)
from scripts.posttraining.assess_car_round2_checkpoints import matched_rows
from scripts.posttraining.assess_car_round7 import read_row, validate_spec, verify_bank

SCALE_KEYS = ("rho_signed_error_mean_member", "rho_absolute_error_mean_member",
              "log1p_power_over_floor_signed_error_mean_member",
              "log1p_power_over_floor_absolute_error_mean_member",
              "member_power_over_floor_mean", "member_low_energy_fraction")
DIRECT_KEYS = ("signed_error_mean_member", "absolute_error_mean_member")
FROZEN_KEYS = ("panel", "core", "core_config", "core_manifest", "expanded_bank", "prefit",
               "assessment_contract", "source_checkpoint")


def same_ref(left, right):
    return isinstance(left, dict) and isinstance(right, dict) and left.get("sha256") == right.get("sha256") and bool(left.get("sha256"))


def validate_identities(assessment, cached, historical, spec, historical_spec, decomposition, panel):
    """Bind the reused SOURCE and child to the same scores, support and law."""
    ids = assessment.get("sample_ids", [])
    if (assessment.get("schema") != "car_round7_matched_assessment_v1" or assessment.get("status") != "completed"
            or assessment.get("split") != "validation" or assessment.get("members") != 4
            or assessment.get("test_locked") is not True or len(ids) != 100 or len(set(ids)) != 100
            or cached.get("schema") != "car_round8_support_readout_v1" or cached.get("status") != "completed"
            or cached.get("split") != "validation" or cached.get("members") != 4
            or cached.get("test_locked") is not True or cached.get("sample_ids") != ids
            or historical.get("sample_ids") != ids or panel.get("sample_ids") != ids
            or cached.get("field_order") != assessment.get("field_order")
            or historical.get("field_order") != assessment.get("field_order")
            or historical.get("field_units") != assessment.get("field_units")
            or historical.get("physical_mask") != assessment.get("physical_mask")):
        raise ValueError("completed matched VALIDATION100 J4 support and field identity required")
    for name in ("source_native", "source_selected", "source_baseline"):
        if not same_ref(assessment.get(name), historical.get(name)):
            raise ValueError(f"SOURCE {name} differs from cached readout")
    for name in ("native", "selected"):
        if not same_ref(assessment["source_" + name], cached["inputs"]["receipts"]["SOURCE"][name]):
            raise ValueError(f"cached SOURCE {name} receipt differs")
    for name in ("panel", "core", "core_config", "core_manifest", "expanded_bank", "prefit",
                 "assessment_contract", "source_checkpoint"):
        if not same_ref(spec.get(name), historical_spec.get(name)):
            raise ValueError(f"frozen observer or support differs: {name}")
    if (decomposition.get("status") != "completed" or decomposition.get("split") != "validation"
            or decomposition.get("members") != 4 or decomposition.get("sample_ids") != ids
            or decomposition.get("field_order") != assessment["field_order"]
            or not same_ref(decomposition.get("spec"), assessment.get("spec"))
            or not same_ref(decomposition.get("source_receipt"), assessment.get("source_selected"))
            or not same_ref(decomposition.get("child_receipt"), assessment.get("child_selected"))
            or not same_ref(decomposition.get("source_baseline"), assessment.get("source_baseline"))
            or [row["sample_id"] for row in decomposition.get("records", [])] != ids
            or [row["sample_id"] for row in cached.get("records", [])] != ids):
        raise ValueError("saved decomposition, row order or selected-bank binding differs")
    for side in ("SOURCE", "candidate"):
        if len(assessment.get("per_condition_scores", {}).get(side, [])) != len(ids):
            raise ValueError("saved condition scores are incomplete")


def validate_direct_companion(companion, assessment, cached, historical, spec, historical_spec, source_path):
    """Use a SOURCE transform only for the exact same frozen panel and observer."""
    ids = assessment["sample_ids"]
    if (companion.get("schema") != "car_round8_state_scale_direct_companion_v1"
            or companion.get("status") != "completed" or companion.get("split") != "validation"
            or companion.get("members") != 4 or companion.get("test_locked") is not True
            or companion.get("sample_ids") != ids or companion.get("field_order") != assessment["field_order"]
            or companion.get("field_units") != assessment["field_units"]
            or companion.get("physical_mask") != assessment["physical_mask"]
            or [row.get("sample_id") for row in companion.get("records", [])] != ids):
        raise ValueError("direct SOURCE companion has different panel, fields or support")
    inputs = companion.get("inputs", {})
    for name, expected in (("source_readout", ref(source_path)),
                           ("historical_assessment", cached["inputs"]["joint500_assessment"]),
                           ("source_selected", assessment["source_selected"]),
                           ("spec", historical["spec"])):
        if not same_ref(inputs.get(name), expected):
            raise ValueError(f"direct SOURCE companion identity differs: {name}")
    for name in FROZEN_KEYS:
        if not same_ref(inputs.get(name), historical_spec.get(name)) or not same_ref(inputs.get(name), spec.get(name)):
            raise ValueError(f"direct SOURCE companion frozen binding differs: {name}")


def summarize_direct(records, normalizer, indices):
    """Paired-frame summaries of direct subspace distances and piece errors."""
    output = {"state_groups": {}, "piece_mean": {}}
    for group in ("marginal", "mixed"):
        values = {}
        for side in ("SOURCE", "candidate"):
            array = np.asarray([row["models"][side]["state_groups"][group] for row in records], dtype=np.float64)
            if (array.shape != (len(records), 3) or not np.isfinite(array).all()
                    or not np.allclose(array[:, 0] - array[:, 1], array[:, 2], rtol=2e-5, atol=1e-7)):
                raise ValueError(f"invalid direct State {group} A/R/S")
            values[side] = array
        delta = values["candidate"] - values["SOURCE"]
        output["state_groups"][group] = {
            "SOURCE_mean": values["SOURCE"].mean(0).tolist(),
            "candidate_mean": values["candidate"].mean(0).tolist(),
            "paired_delta_raw": paired_interval(delta, indices),
            "paired_delta_frozen_train_units": paired_interval(delta / normalizer, indices),
        }
    for key in DIRECT_KEYS:
        values = {side: np.asarray([row["models"][side]["piece_mean"][key] for row in records], dtype=np.float64)
                  for side in ("SOURCE", "candidate")}
        if (values["SOURCE"].shape != values["candidate"].shape or values["SOURCE"].ndim != 3
                or not all(np.isfinite(value).all() for value in values.values())
                or (key.startswith("absolute") and any((value < 0).any() for value in values.values()))):
            raise ValueError(f"invalid direct piece-mean errors: {key}")
        output["piece_mean"][key] = {
            "SOURCE_mean": values["SOURCE"].mean(0).tolist(),
            "candidate_mean": values["candidate"].mean(0).tolist(),
            "paired_delta": paired_interval(values["candidate"] - values["SOURCE"], indices),
        }
    return output


def summarize(records, normalizer, *, seed=6042026, replicates=2000):
    if not math.isfinite(float(normalizer)) or normalizer <= 0 or not records or replicates < 1:
        raise ValueError("positive fixed State normalizer and paired records required")
    indices = np.random.default_rng(seed).integers(0, len(records), size=(replicates, len(records)))
    output = {"state": {}, "scale": {}}
    for name in ("SOURCE", "candidate"):
        values = np.asarray([row["models"][name]["state"] for row in records], dtype=np.float64)
        if values.shape != (len(records), 3) or not np.isfinite(values).all() or not np.allclose(values[:, 0] - values[:, 1], values[:, 2], rtol=2e-5, atol=1e-7):
            raise ValueError("State must contain finite paired A/R/S values")
        output["state"][name + "_mean"] = values.mean(0).tolist()
    state_delta = np.asarray([np.asarray(row["models"]["candidate"]["state"]) - np.asarray(row["models"]["SOURCE"]["state"]) for row in records])
    output["state"]["paired_delta_raw"] = paired_interval(state_delta, indices)
    output["state"]["paired_delta_frozen_train_units"] = paired_interval(state_delta / normalizer, indices)
    for key in SCALE_KEYS:
        source = np.asarray([row["models"]["SOURCE"]["scale"][key] for row in records], dtype=np.float64)
        child = np.asarray([row["models"]["candidate"]["scale"][key] for row in records], dtype=np.float64)
        if source.shape != child.shape or not np.isfinite(source).all() or not np.isfinite(child).all():
            raise ValueError(f"nonfinite or unmatched Scale context: {key}")
        output["scale"][key] = {"source_mean": source.mean(0).tolist(), "candidate_mean": child.mean(0).tolist(),
                                "paired_delta": paired_interval(child - source, indices)}
    output["scale"]["reference_power_over_floor_mean"] = np.asarray(
        [row["models"]["SOURCE"]["scale"]["reference_power_over_floor"] for row in records]).mean(0).tolist()
    output["scale"]["reference_low_energy_fraction"] = np.asarray(
        [row["models"]["SOURCE"]["scale"]["reference_low_energy"] for row in records]).mean(0).tolist()
    if "state_groups" in records[0]["models"]["SOURCE"]:
        output.update(summarize_direct(records, normalizer, indices))
    return output


def build_direct_companion(assessment_path, source_readout_path, *, threads=2):
    """Cache SOURCE and Joint500 direct readouts once from their saved J4 banks."""
    if threads < 1:
        raise ValueError("positive CPU thread count required")
    started, cpu = time.perf_counter(), time.process_time()
    assessment_path = Path(assessment_path).resolve(strict=True)
    source_readout_path = Path(source_readout_path).resolve(strict=True)
    assessment = json.loads(assessment_path.read_text())
    cached = json.loads(source_readout_path.read_text())
    if not same_ref(cached["inputs"]["joint500_assessment"], ref(assessment_path)):
        raise ValueError("companion must use the historical assessment bound to the SOURCE readout")
    spec = json.loads(checked_ref(assessment["spec"]).read_text())
    panel, native_source = validate_spec(spec)
    decomposition_path = assessment_path.parent / "decomposition.json"
    decomposition = json.loads(decomposition_path.read_text())
    validate_identities(assessment, cached, assessment, spec, spec, decomposition, panel)
    banks = {}
    for name, key in (("SOURCE", "source_selected"), ("JointAnchor500", "child_selected")):
        path = checked_ref(assessment[key])
        _, receipt = verify_bank(path, spec, panel, native_source, "selected")
        expected = spec["source_checkpoint"]["sha256"] if name == "SOURCE" else spec["checkpoint"]["sha256"]
        if receipt.get("checkpoint_sha256") != expected:
            raise ValueError(f"{name} selected bank checkpoint differs")
        banks[name] = (path, receipt)
    contract = json.loads(checked_ref(spec["assessment_contract"]).read_text())
    normalizer = float(contract["normalizers"]["state"])
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("positive frozen State unit required")
    torch.set_num_threads(threads)
    dataset, families, core = build_state_scale(spec, native_source)
    records, paired = [], []
    try:
        weights, geometry = direct_readout_metadata(families["global_distribution"],
                                                    families["multiscale_spectrum"], assessment["field_order"])
        with torch.no_grad():
            for index, sid in enumerate(assessment["sample_ids"]):
                rows = {name: read_row(path, receipt, index) for name, (path, receipt) in banks.items()}
                matched_rows(rows["SOURCE"], rows["JointAnchor500"])
                if rows["SOURCE"]["member_seeds"] != decomposition["records"][index]["member_seeds"]:
                    raise ValueError(f"member seeds differ for {sid}")
                direct = {}
                for name, row in rows.items():
                    families["multiscale_spectrum"]._require_ready(row["truth_model"], row["coordinates_model"])
                    state, groups = state_triplet(families["global_distribution"], core["state_bank"],
                        row["members_model"], row["truth_model"], row["native_masses"], return_groups=True)
                    scale, piece = scale_context(families["multiscale_spectrum"], row["members_model"],
                                                 row["truth_model"], return_piece_means=True)
                    previous = cached["records"][index]["models"][name]
                    if state != previous["state"] or scale != previous["scale"]:
                        raise ValueError(f"previous State or Scale context changed: {sid} {name}")
                    side = "SOURCE" if name == "SOURCE" else "candidate"
                    if not math.isclose(state[2], assessment["per_condition_scores"][side][index]["state"],
                                        rel_tol=2e-5, abs_tol=1e-7):
                        raise ValueError(f"saved State score differs: {sid} {name}")
                    direct[name] = {"state_groups": groups, "piece_mean": piece}
                records.append({"sample_id": sid, "models": direct})
                paired.append({"models": {"SOURCE": direct["SOURCE"], "candidate": direct["JointAnchor500"]}})
    finally:
        dataset.close()
    indices = np.random.default_rng(6042026).integers(0, len(records), size=(2000, len(records)))
    return {"schema": "car_round8_state_scale_direct_companion_v1", "status": "completed",
            "split": "validation", "members": 4, "sample_ids": assessment["sample_ids"],
            "field_order": assessment["field_order"], "field_units": assessment["field_units"],
            "physical_mask": assessment["physical_mask"], "state_frozen_train_unit": normalizer,
            "state_group_weights": weights, "piece_geometry": geometry, "records": records,
            "summary": summarize_direct(paired, normalizer, indices),
            "bootstrap": {"seed": 6042026, "replicates": 2000, "unit": "paired frame; one trajectory"},
            "definitions": {
                "state_groups": "weighted marginal and mixed root norms on one common exact quantile embedding; each is A, positive R, S=A-R; group scores do not sum to total root-norm State",
                "piece_mean": "per frozen selected-support connected piece and field: mass-weighted member minus truth means, averaged across J4; absolute errors take absolute value before member averaging; units are the frozen Scale unit_space",
                "support": "separately queried Q4096 selected support with frozen physical piece measures; no native-support recomputation"},
            "inputs": {"source_readout": ref(source_readout_path), "historical_assessment": ref(assessment_path),
                       "source_selected": assessment["source_selected"], "child_selected": assessment["child_selected"],
                       "decomposition": ref(decomposition_path), "spec": assessment["spec"],
                       **{name: spec[name] for name in FROZEN_KEYS}, "script": ref(Path(__file__)),
                       "support_code": ref(ROOT / "scripts/analysis/car_round8_support_readout.py")},
            "timing": {"wall_seconds": time.perf_counter()-started, "cpu_seconds": time.process_time()-cpu,
                       "threads": threads, "device": "CPU"},
            "additional_inference": False, "additional_persistence_calls": 0, "test_locked": True}


def build_readout(assessment_path, source_readout_path, *, label, threads=2, source_companion_path=None):
    if not isinstance(label, str) or len(label.strip()) < 2 or threads < 1:
        raise ValueError("a descriptive endpoint label and positive CPU thread count are required")
    started, cpu = time.perf_counter(), time.process_time()
    assessment_path = Path(assessment_path).resolve(strict=True)
    source_readout_path = Path(source_readout_path).resolve(strict=True)
    assessment = json.loads(assessment_path.read_text())
    cached = json.loads(source_readout_path.read_text())
    historical = json.loads(checked_ref(cached["inputs"]["joint500_assessment"]).read_text())
    spec = json.loads(checked_ref(assessment["spec"]).read_text())
    historical_spec = json.loads(checked_ref(historical["spec"]).read_text())
    panel, source_receipt = validate_spec(spec)
    decomposition_path = assessment_path.parent / "decomposition.json"
    decomposition = json.loads(decomposition_path.read_text())
    validate_identities(assessment, cached, historical, spec, historical_spec, decomposition, panel)
    companion = None
    if source_companion_path is not None:
        source_companion_path = Path(source_companion_path).resolve(strict=True)
        companion = json.loads(source_companion_path.read_text())
        validate_direct_companion(companion, assessment, cached, historical, spec, historical_spec,
                                  source_readout_path)
    selected_path = checked_ref(assessment["child_selected"])
    _, selected = verify_bank(selected_path, spec, panel, source_receipt, "selected")
    if selected.get("checkpoint_sha256") != spec.get("checkpoint", {}).get("sha256"):
        raise ValueError("selected bank and endpoint checkpoint differ")
    contract = json.loads(checked_ref(spec["assessment_contract"]).read_text())
    normalizer = float(contract["normalizers"]["state"])
    expected_pairs = [list(pair) for pair in itertools.combinations(range(len(assessment["field_order"])), 2)]
    torch.set_num_threads(threads)
    dataset, families, core = build_state_scale(spec, source_receipt)
    records = []
    try:
        if companion is not None:
            weights, geometry = direct_readout_metadata(families["global_distribution"],
                families["multiscale_spectrum"], assessment["field_order"])
            if (weights != companion.get("state_group_weights") or geometry != companion.get("piece_geometry")
                    or normalizer != companion.get("state_frozen_train_unit")):
                raise ValueError("direct SOURCE companion State weights, Scale geometry or TRAIN unit differs")
        with torch.no_grad():
            for index, sid in enumerate(assessment["sample_ids"]):
                row = read_row(selected_path, selected, index)
                source_row = cached["records"][index]["models"]["SOURCE"]
                saved = decomposition["records"][index]
                if row["member_seeds"] != saved["member_seeds"]:
                    raise ValueError(f"member seeds differ for {sid}")
                families["multiscale_spectrum"]._require_ready(row["truth_model"], row["coordinates_model"])
                state_result = state_triplet(families["global_distribution"], core["state_bank"],
                    row["members_model"], row["truth_model"], row["native_masses"], return_groups=companion is not None)
                scale_result = scale_context(families["multiscale_spectrum"], row["members_model"],
                                             row["truth_model"], return_piece_means=companion is not None)
                if companion is None:
                    state, scale = state_result, scale_result
                else:
                    state, state_groups = state_result
                    scale, piece_mean = scale_result
                for name, actual in (("SOURCE", source_row["state"][2]), ("candidate", state[2])):
                    expected = assessment["per_condition_scores"][name][index]["state"]
                    if not math.isclose(actual, expected, rel_tol=2e-5, abs_tol=1e-7):
                        raise ValueError(f"cached State fair score differs: {sid} {name}")
                for name in ("reference_power_over_floor", "reference_low_energy"):
                    if not np.allclose(scale[name], source_row["scale"][name], rtol=1e-8, atol=1e-8):
                        raise ValueError(f"reference Scale support or truth differs: {sid} {name}")
                if (scale["field_pair_indices"] != expected_pairs or source_row["scale"]["field_pair_indices"] != expected_pairs
                        or not np.allclose(scale["rho_absolute_error_mean_member"], saved["models"]["candidate"]["localization"]["rho_pair_band"]["attraction"], rtol=2e-8, atol=1e-10)):
                    raise ValueError(f"field pairs or saved Scale rho attraction differ: {sid}")
                models = {"SOURCE": {"state": source_row["state"], "scale": source_row["scale"]},
                          "candidate": {"state": state, "scale": scale}}
                if companion is not None:
                    source_direct = companion["records"][index]["models"]["SOURCE"]
                    models["SOURCE"].update(state_groups=source_direct["state_groups"], piece_mean=source_direct["piece_mean"])
                    models["candidate"].update(state_groups=state_groups, piece_mean=piece_mean)
                records.append({"sample_id": sid, "models": models})
    finally:
        dataset.close()
    result = {"schema": "car_round8_endpoint_relation_context_v1", "status": "completed", "endpoint_label": label,
            "reporting_epoch": spec.get("reporting_epoch"), "split": "validation", "sample_ids": assessment["sample_ids"],
            "members": 4, "field_order": assessment["field_order"], "field_units": assessment["field_units"],
            "physical_mask": assessment["physical_mask"], "field_pair_indices": expected_pairs, "state_frozen_train_unit": normalizer,
            "records": records, "summary": summarize(records, normalizer),
            "bootstrap": {"seed": 6042026, "replicates": 2000, "unit": "paired frame; one trajectory"},
            "definitions": {"state": "exact weighted-CDF A, positive R, fair S=A-R; normalized deltas use frozen TRAIN State unit",
                            "scale": "signed rho and log1p(power/floor) member-reference errors on fixed Q4096; original pair/band order and frozen floor",
                            "support": "separately queried fixed selected support; no native-support recomputation"},
            "inputs": {"assessment": ref(assessment_path), "decomposition": ref(decomposition_path),
                       "source_readout": ref(source_readout_path), "spec": assessment["spec"],
                       "selected_bank": assessment["child_selected"], "contract": spec["assessment_contract"],
                       "script": ref(Path(__file__)), "support_code": ref(ROOT / "scripts/analysis/car_round8_support_readout.py")},
            "timing": {"wall_seconds": time.perf_counter()-started, "cpu_seconds": time.process_time()-cpu, "threads": threads, "device": "CPU"},
            "additional_inference": False, "additional_persistence_calls": 0, "test_locked": True}
    if companion is not None:
        result["state_group_weights"] = weights
        result["piece_geometry"] = geometry
        result["definitions"].update(state_groups=companion["definitions"]["state_groups"],
                                     piece_mean=companion["definitions"]["piece_mean"])
        result["inputs"]["source_companion"] = ref(source_companion_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--source-readout", type=Path, required=True)
    parser.add_argument("--label", help="descriptive child endpoint label")
    parser.add_argument("--build-source-companion", action="store_true",
                        help="cache SOURCE and historical Joint500 direct readouts once")
    parser.add_argument("--source-companion", type=Path,
                        help="explicit hash-bound SOURCE direct-readout companion")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.build_source_companion:
        if args.source_companion is not None:
            parser.error("a SOURCE companion cannot depend on itself")
        result = build_direct_companion(args.assessment, args.source_readout, threads=args.threads)
    else:
        result = build_readout(args.assessment, args.source_readout, label=args.label,
                               threads=args.threads, source_companion_path=args.source_companion)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output), "timing": result["timing"]}))


if __name__ == "__main__":
    main()
