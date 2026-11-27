#!/usr/bin/env python3
"""Summarize completed matched Round 8 learning endpoints from cached evidence.

The two required arms are compared at epoch 100. An optional nominated final
endpoint and post-nomination task results remain separately identified. The
program reads saved scores, histories and two preselected native-bank rows;
it never runs inference, an observer, a controller or checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.analysis.car_round8_cached_summary import checked_ref, paired_interval, ref
from scripts.analysis.car_round8_child_relations import SCALE_KEYS
from scripts.analysis.report_car_round5 import save_figure
from scripts.analysis.report_car_round8 import native_grid, physical_native, physical_rmse
from scripts.posttraining.assess_car_round2_checkpoints import matched_rows
from scripts.posttraining.assess_car_round7 import read_row
from scripts.posttraining.run_car_round8 import BASE, SPATIAL

SEED, REPLICATES = 6042026, 2000
FAMILIES = ("State", "Complete expanded Scale", "PH Shape", "Location / TRAIN unit",
            "Half(PH + location / TRAIN unit)", "Native-to-raster location / TRAIN unit")
SPEC_KEYS = ("panel", "core", "core_config", "core_manifest", "expanded_bank", "prefit",
             "assessment_contract", "source_checkpoint", "source_native_receipt")
BLOCKS = ("signed", "joint", "log_power_self", "rho", "third_order")
TERMS = ("attraction", "separation", "score")
RISK_DIAGNOSTICS = ("member_mse", "ensemble_mean_mse", "signed_mean_risk", "unbiased_member_variance")


def load(path):
    return json.loads(Path(path).read_text())


def same_ref(a, b):
    return isinstance(a, dict) and isinstance(b, dict) and bool(a.get("sha256")) and a.get("sha256") == b.get("sha256")


def rows_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def require_panel(obj, ids):
    if (obj.get("status") != "completed" or obj.get("split") != "validation" or obj.get("members") != 4
            or obj.get("sample_ids") != ids or obj.get("test_locked") is not True):
        raise ValueError("completed matched VALIDATION100 J4 evidence required")


def load_endpoint(training_path, assessment_path, *, expected_epoch=None):
    """Read immutable receipts without reopening the mutable last checkpoint."""
    training_path, assessment_path = Path(training_path).resolve(strict=True), Path(assessment_path).resolve(strict=True)
    training, result = load(training_path), load(assessment_path)
    arm, epoch = training.get("arm"), training.get("target_epoch")
    if (training.get("schema") != "car_round8_training_result_v1" or arm not in (BASE, SPATIAL)
            or not isinstance(epoch, int) or not 100 <= epoch <= 500
            or expected_epoch is not None and epoch != expected_epoch
            or training.get("reporting_epoch") != epoch or training.get("global_step") != 12 * epoch
            or training.get("target_reached") is not True or training.get("disposition") != "target_reached"
            or training.get("test_locked") is not True):
        raise ValueError("completed exact Round 8 training endpoint required")
    ids = result.get("sample_ids", [])
    if (result.get("schema") != "car_round8_validation_assessment_v1" or result.get("arm") != arm
            or result.get("epoch") != epoch or len(ids) != 100 or len(set(ids)) != 100):
        raise ValueError("assessment arm, reporting age or unique panel differs")
    require_panel(result, ids)
    plan_path, generation_path = checked_ref(result["plan"]), checked_ref(result["generation"])
    plan, generation = load(plan_path), load(generation_path)
    if (plan.get("schema") != "car_round8_validation_endpoint_plan_v1" or plan.get("arm") != arm
            or plan.get("epoch") != epoch or plan.get("sample_ids") != ids or plan.get("test_locked") is not True
            or not same_ref(plan.get("training_result"), ref(training_path))
            or not same_ref(plan.get("checkpoint"), training.get("checkpoint"))
            or generation.get("status") != "completed" or not same_ref(generation.get("plan"), ref(plan_path))
            or set(generation.get("banks", {})) != {"native", "selected"}):
        raise ValueError("training, endpoint plan, checkpoint or generated banks differ")
    checked_ref(plan["checkpoint"])
    if not same_ref(plan.get("original_training_checkpoint"), training["checkpoint"]):
        raise ValueError("immutable endpoint checkpoint is not bound to the training receipt")
    prepared = load(checked_ref(plan["prepared"]))
    checked_ref(training["prepared_config"]); checked_ref(training["saved_config"])
    if (not same_ref(prepared.get("arms", {}).get(arm), training["prepared_config"])
            or not same_ref(prepared.get("position_fit"), plan.get("position_fit"))):
        raise ValueError("training configuration is not from the common prepared pair")
    spec = load(checked_ref(plan["spec"]))
    if (spec.get("reporting_epoch") != epoch or not same_ref(spec.get("checkpoint"), training["checkpoint"])):
        raise ValueError("frozen specification checkpoint or reporting age differs")
    for key in SPEC_KEYS:
        checked_ref(spec[key])
    panel = load(checked_ref(spec["panel"]))
    if panel.get("sample_ids") != ids or panel.get("split") != "validation":
        raise ValueError("frozen validation panel differs")
    assessment = load(checked_ref(result["assessment"]))
    require_panel(assessment, ids)
    if (assessment.get("schema") != "car_round7_matched_assessment_v1"
            or not same_ref(assessment.get("spec"), plan["spec"])
            or not same_ref(assessment.get("source_native"), spec["source_native_receipt"])
            or not same_ref(assessment.get("source_selected"), plan["source_selected"])
            or not same_ref(assessment.get("source_baseline"), plan["source_baseline"])):
        raise ValueError("saved assessment and source observer binding differ")
    banks = {}
    for support in ("native", "selected"):
        identity = generation["banks"][support]
        path = checked_ref(identity)
        bank = load(path)
        if (not same_ref(assessment["child_" + support], identity) or bank.get("status") != "completed"
                or bank.get("split") != "validation" or bank.get("support") != support
                or bank.get("sample_ids") != ids or bank.get("members") != 4
                or bank.get("checkpoint_sha256") != training["checkpoint"]["sha256"]
                or bank.get("spec_sha256") != plan["spec"]["sha256"]
                or len(bank.get("files_sha256", [])) != len(ids)):
            raise ValueError(f"{support} bank differs from the completed endpoint")
        banks[support] = {"path": path, "receipt": bank, "ref": identity}
    source_path = checked_ref(assessment["source_native"])
    source_bank = load(source_path)
    if (source_bank.get("status") != "completed" or source_bank.get("sample_ids") != ids
            or source_bank.get("members") != 4 or source_bank.get("native_points") != panel.get("native_points")
            or source_bank.get("source_sha256") != spec["source_checkpoint"]["sha256"]):
        raise ValueError("cached SOURCE native bank differs")
    checked_ref(assessment["source_selected"]); checked_ref(assessment["source_baseline"])
    member = load(checked_ref(result["member_mse_summary"]))
    scale = load(checked_ref(result["scale_summary"]))
    position = load(checked_ref(result["position_readout"]))
    relation = load(checked_ref(result["relation_context"]))
    for value in (position, relation):
        require_panel(value, ids)
        if [row.get("sample_id") for row in value.get("records", [])] != ids:
            raise ValueError("condition readout order differs")
    if (member.get("schema") != "car_round8_cached_summary_v1" or member.get("split") != "validation"
            or member.get("sample_ids") != ids or member.get("members") != 4 or member.get("test_locked") is not True
            or member.get("bootstrap", {}).get("seed") != SEED or member.get("bootstrap", {}).get("replicates") != REPLICATES
            or not same_ref(member.get("inputs", {}).get("assessment"), result["assessment"])
            or relation.get("schema") != "car_round8_endpoint_relation_context_v1"
            or relation.get("endpoint_label") != arm or relation.get("reporting_epoch") != epoch
            or relation.get("bootstrap", {}).get("seed") != SEED or relation.get("bootstrap", {}).get("replicates") != REPLICATES
            or not same_ref(relation["inputs"]["assessment"], result["assessment"])
            or not same_ref(relation["inputs"]["selected_bank"], banks["selected"]["ref"])
            or not same_ref(relation["inputs"]["spec"], plan["spec"])
            or position.get("schema") != "car_round8_endpoint_position_readout_v1"
            or position.get("bootstrap", {}).get("seed") != SEED or position.get("bootstrap", {}).get("replicates") != REPLICATES
            or not same_ref(position.get("fit"), plan.get("position_fit"))):
        raise ValueError("saved member, State/Scale or common-position readout differs")
    companion = relation.get("inputs", {}).get("source_companion")
    if not isinstance(companion, dict):
        raise ValueError("saved State-group and piece-mean SOURCE companion is required")
    checked_ref(companion)
    decomposition_path = checked_ref(relation["inputs"]["decomposition"])
    decomposition = load(decomposition_path)
    if (not same_ref(member["inputs"]["decomposition"], ref(decomposition_path))
            or decomposition.get("status") != "completed" or decomposition.get("sample_ids") != ids
            or decomposition.get("members") != 4 or decomposition.get("field_order") != assessment["field_order"]
            or not same_ref(decomposition.get("spec"), plan["spec"])
            or not same_ref(decomposition.get("source_receipt"), assessment["source_selected"])
            or not same_ref(decomposition.get("child_receipt"), banks["selected"]["ref"])
            or [row.get("sample_id") for row in decomposition.get("records", [])] != ids):
        raise ValueError("Scale decomposition does not bind the scored endpoint")
    return {"arm": arm, "epoch": epoch, "training_path": training_path, "training": training,
            "result_path": assessment_path, "result": result, "plan": plan, "generation": generation, "spec": spec, "panel": panel,
            "assessment": assessment, "member": member, "scale": scale, "position": position,
            "relation": relation, "decomposition": decomposition, "banks": banks,
            "source_bank": {"path": source_path, "receipt": source_bank}}


def check_common(base, spatial):
    a, b = base["assessment"], spatial["assessment"]
    for key in ("sample_ids", "members", "field_order", "field_units", "physical_mask"):
        if a[key] != b[key]: raise ValueError(f"matched endpoint {key} differs")
    for key in ("source_native", "source_selected", "source_baseline"):
        if not same_ref(a[key], b[key]): raise ValueError(f"matched SOURCE {key} differs")
    for key in SPEC_KEYS:
        if not same_ref(base["spec"][key], spatial["spec"][key]):
            raise ValueError(f"matched observer, support or normalizer differs: {key}")
    if (base["decomposition"]["prefit_coefficients"] != spatial["decomposition"]["prefit_coefficients"]
            or not same_ref(base["position"]["fit"], spatial["position"]["fit"])
            or not same_ref(base["position"]["cached_source"], spatial["position"]["cached_source"])
            or not same_ref(base["relation"]["inputs"]["source_readout"], spatial["relation"]["inputs"]["source_readout"])
            or not same_ref(base["relation"]["inputs"]["source_companion"], spatial["relation"]["inputs"]["source_companion"])
            or base["position"]["kind_weights"] != spatial["position"]["kind_weights"]
            or base["position"]["location_unit"] != spatial["position"]["location_unit"]
            or base["relation"]["field_pair_indices"] != spatial["relation"]["field_pair_indices"]
            or base["relation"]["state_frozen_train_unit"] != spatial["relation"]["state_frozen_train_unit"]
            or base["member"]["example_selection"] != spatial["member"]["example_selection"]):
        raise ValueError("common positional observer, SOURCE readout or prefit differs")
    if (a["per_condition_fields"]["SOURCE"] != b["per_condition_fields"]["SOURCE"]
            or a["per_condition_scores"]["SOURCE"] != b["per_condition_scores"]["SOURCE"]
            or [r["models"]["SOURCE"] for r in base["relation"]["records"]] != [r["models"]["SOURCE"] for r in spatial["relation"]["records"]]
            or [r["models"]["SOURCE"] for r in base["position"]["records"]] != [r["models"]["SOURCE"] for r in spatial["position"]["records"]]
            or [r["models"]["SOURCE"] for r in base["decomposition"]["records"]] != [r["models"]["SOURCE"] for r in spatial["decomposition"]["records"]]):
        raise ValueError("cached SOURCE per-condition values differ")
    member_seeds = [r.get("member_seeds") for r in base["decomposition"]["records"]]
    if (any(not isinstance(seeds, list) or len(seeds) != 4 or len(set(seeds)) != 4 for seeds in member_seeds)
            or member_seeds != [r.get("member_seeds") for r in spatial["decomposition"]["records"]]):
        raise ValueError("matched condition member identities differ")


def check_matched(base, spatial):
    if (base["arm"] != BASE or spatial["arm"] != SPATIAL or base["epoch"] != 100 or spatial["epoch"] != 100):
        raise ValueError("package attribution requires both descriptive arms at epoch 100")
    check_common(base, spatial)


def pair_interval(a, b, indices):
    return paired_interval(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), indices)


def member_diagnostics(row):
    """Retain the saved pooled J4 quantities and check their finite-J identities."""
    out = {}
    for side in ("source", "candidate"):
        values = {key: float(row[side][key]) for key in RISK_DIAGNOSTICS}
        if not np.isfinite(list(values.values())).all():
            raise ValueError("nonfinite saved member diagnostic")
        mean, signed, variance = (values[k] for k in RISK_DIAGNOSTICS[1:])
        if (not np.isclose(values["member_mse"], mean + .75 * variance, rtol=2e-5, atol=1e-9)
                or not np.isclose(signed, mean - .25 * variance, rtol=2e-5, atol=1e-9)):
            raise ValueError("saved J4 member/mean/spread identity differs")
        out[side] = values
    return out


def family_arrays(endpoint):
    assessment, decomposition, relation, position = (endpoint[k] for k in ("assessment", "decomposition", "relation", "position"))
    prefit = decomposition["prefit_coefficients"]
    weights = {key: float(prefit["scale." + key]) * (.25 if key == "joint" else 1.) for key in BLOCKS}
    out = {}
    for side in ("SOURCE", "candidate"):
        state = np.asarray([row["models"][side]["state"] for row in relation["records"]], float)
        scale = np.asarray([[sum(weights[key] * row["models"][side]["blocks"][key][term] for key in BLOCKS)
                             for term in TERMS] for row in decomposition["records"]], float)
        ph = np.asarray([[sum(float(row[f"shape.{kind}.{category}.{term}"]) for kind in ("self", "mutual")
                               for category in ("finite", "essential")) for term in ("attraction", "separation")]
                         for row in assessment["per_condition_scores"][side]], float)
        ph = np.column_stack((ph, ph[:, 0] - ph[:, 1]))
        arrays = {"State": state, "Complete expanded Scale": scale, "PH Shape": ph}
        for label, key in (("Location / TRAIN unit", "location_unit_matched_ARS"),
                           ("Half(PH + location / TRAIN unit)", "combined_ARS"),
                           ("Native-to-raster location / TRAIN unit", "native_location_unit_matched_ARS")):
            arrays[label] = np.asarray([sum(np.asarray(row["models"][side][kind][key], float) for kind in ("self", "mutual"))
                                        for row in position["records"]])
        for family, values in arrays.items():
            if (values.shape != (100, 3) or not np.isfinite(values).all()
                    or not np.allclose(values[:, 0] - values[:, 1], values[:, 2], rtol=2e-5, atol=1e-7)):
                raise ValueError(f"invalid {family} direct A/R/S")
        saved = assessment["per_condition_scores"][side]
        if not np.allclose(state[:, 2], [r["state"] for r in saved], rtol=2e-5, atol=1e-7):
            raise ValueError("direct State does not reproduce the saved fair score")
        if not np.allclose(scale[:, 2], [r["models"][side]["definitions"]["Complete_Expanded_Prefit"] for r in decomposition["records"]], rtol=1e-8, atol=1e-10):
            raise ValueError("complete expanded Scale does not reproduce saved decomposition")
        position_ph = np.asarray([sum(np.asarray(row["models"][side][kind]["ph_ARS"], float) for kind in ("self", "mutual"))
                                  for row in position["records"]])
        if not np.allclose(ph, position_ph, rtol=2e-5, atol=1e-6):
            raise ValueError("common PH observer differs from saved assessment")
        out[side] = arrays
    return out, weights


def component_arrays(endpoint):
    """Read raw Scale blocks and PH leaves from already-scored condition rows."""
    out = {}
    for side in ("SOURCE", "candidate"):
        scale = {key: np.asarray([[row["models"][side]["blocks"][key][term] for term in TERMS]
                                  for row in endpoint["decomposition"]["records"]], float) for key in BLOCKS}
        shape = {}
        for kind in ("self", "mutual"):
            for category in ("finite", "essential"):
                key = f"{kind}.{category}"
                rows = endpoint["assessment"]["per_condition_scores"][side]
                a = np.asarray([row[f"shape.{key}.attraction"] for row in rows], float)
                r = np.asarray([row[f"shape.{key}.separation"] for row in rows], float)
                shape[key] = np.column_stack((a, r, a - r))
        for values in (*scale.values(), *shape.values()):
            if (values.shape != (100, 3) or not np.isfinite(values).all()
                    or not np.allclose(values[:, 0] - values[:, 1], values[:, 2], rtol=2e-5, atol=1e-7)):
                raise ValueError("invalid saved component A/R/S")
        out[side] = {"Scale blocks": scale, "PH Shape components": shape}
    return out


def component_comparison(endpoints, index, weights):
    """Summarize saved raw components with one paired contrast, without rescoring."""
    arrays = {name: component_arrays(endpoint) for name, endpoint in endpoints.items()}
    names = list(endpoints)
    if len(names) not in (1, 2):
        raise ValueError("one selected endpoint or two matched endpoints required")
    source = arrays[names[0]]["SOURCE"]
    result = {}
    for group in ("Scale blocks", "PH Shape components"):
        result[group] = {}
        for key, values in source[group].items():
            if any(not np.allclose(values, arrays[name]["SOURCE"][group][key], rtol=1e-10, atol=1e-10) for name in names):
                raise ValueError(f"SOURCE component differs across endpoints: {group} {key}")
            candidates = {name: arrays[name]["candidate"][group][key] for name in names}
            left = candidates[names[-1]]
            right = values if len(names) == 1 else candidates[names[0]]
            item = {"SOURCE_mean_ARS": values.mean(0).tolist(),
                    **{name + "_mean_ARS": value.mean(0).tolist() for name, value in candidates.items()},
                    "paired_contrast": pair_interval(left, right, index)}
            if group == "Scale blocks":
                factor = weights[key]
                item["fixed_contribution_factor"] = factor
                item["weighted_paired_contrast"] = pair_interval(factor * left, factor * right, index)
            result[group][key] = item
    return result


def detail_comparison(endpoints, index):
    """Summarize completed direct State groups and piece means on paired frames."""
    names = list(endpoints)
    if len(names) not in (1, 2):
        raise ValueError("one selected endpoint or two matched endpoints required")
    first = endpoints[names[0]]
    geometry, weights = first["relation"]["piece_geometry"], first["relation"]["state_group_weights"]
    fields, ids = first["assessment"]["field_order"], first["assessment"]["sample_ids"]
    pieces, masses = geometry["piece_indices"], geometry["mass_fractions"]
    if (geometry["unit_space"] != "model_units" or geometry["field_order"] != fields
            or not pieces or len(pieces) != len(masses) or len(set(pieces)) != len(pieces)
            or not np.isfinite(masses).all() or any(x <= 0 for x in masses)
            or not np.isclose(sum(masses), 1.) or set(weights) != {"marginal", "mixed"}
            or not np.isfinite(list(weights.values())).all() or weights["marginal"] <= 0 or weights["mixed"] < 0):
        raise ValueError("invalid frozen State group weights or piece geometry")
    for endpoint in endpoints.values():
        if (endpoint["relation"]["piece_geometry"] != geometry or endpoint["relation"]["state_group_weights"] != weights
                or endpoint["assessment"]["sample_ids"] != ids):
            raise ValueError("State group or piece geometry differs across endpoints")
    result = {"piece_geometry": geometry, "state_group_weights": weights, "state_groups": {}, "piece_mean": {}}
    def summary(kind, key, expected):
        def read(endpoint, side):
            rows = endpoint["relation"]["records"]
            values = np.asarray([row["models"][side][kind][key] for row in rows], float)
            if [row["sample_id"] for row in rows] != ids or values.shape != expected or not np.isfinite(values).all():
                raise ValueError(f"invalid saved {kind} {key} condition values")
            return values
        source = read(first, "SOURCE")
        candidates = {name: read(endpoint, "candidate") for name, endpoint in endpoints.items()}
        if any(not np.allclose(source, read(endpoint, "SOURCE"), rtol=1e-10, atol=1e-10) for endpoint in endpoints.values()):
            raise ValueError(f"SOURCE {kind} {key} differs across endpoints")
        if kind == "state_groups":
            for values in (source, *candidates.values()):
                if not np.allclose(values[:, 0] - values[:, 1], values[:, 2], rtol=2e-5, atol=1e-7):
                    raise ValueError(f"invalid State {key} A/R/S")
        left = candidates[names[-1]]
        right = source if len(names) == 1 else candidates[names[0]]
        return {"SOURCE_mean": source.mean(0).tolist(),
                **{name + "_mean": value.mean(0).tolist() for name, value in candidates.items()},
                **{name + "_minus_SOURCE": pair_interval(value, source, index) for name, value in candidates.items()},
                "paired_contrast": pair_interval(left, right, index)}
    for key in ("marginal", "mixed"):
        result["state_groups"][key] = summary("state_groups", key, (len(ids), 3))
    for key in ("signed_error_mean_member", "absolute_error_mean_member"):
        result["piece_mean"][key] = summary("piece_mean", key, (len(ids), len(pieces), len(fields)))
    for endpoint in endpoints.values():
        for row in endpoint["relation"]["records"]:
            for side in ("SOURCE", "candidate"):
                values = row["models"][side]["piece_mean"]
                signed = np.asarray(values["signed_error_mean_member"], float)
                absolute = np.asarray(values["absolute_error_mean_member"], float)
                if np.any(absolute + 1e-7 + 1e-6 * absolute < np.abs(signed)):
                    raise ValueError("absolute piece-mean error must average member absolute errors")
    return result


def sensor_evidence(endpoint):
    """Retain saved observation errors and their dimensional allowance."""
    assessment = endpoint["assessment"]
    fields, ids = assessment["field_order"], assessment["sample_ids"]
    rows = assessment["per_condition_fields"]
    if not all("sensor_mse" in row and "sensor_counts" in row
               for side in ("SOURCE", "candidate") for row in rows[side]):
        return {"available": False}
    counts = {side: np.asarray([row["sensor_counts"] for row in rows[side]])
              for side in ("SOURCE", "candidate")}
    if (counts["SOURCE"].shape != (len(ids), len(fields))
            or not np.array_equal(counts["SOURCE"], counts["candidate"])):
        raise ValueError("SOURCE and child observation counts differ")
    errors = {side: np.asarray([row["sensor_mse"] for row in rows[side]], float)
              for side in ("SOURCE", "candidate")}
    if any(value.shape != (len(ids), len(fields)) or not np.isfinite(value).all()
           for value in errors.values()):
        raise ValueError("invalid saved sensor errors")
    safeguard = assessment.get("native_safeguards", {})
    allowance = safeguard.get("sensor_allowance")
    if allowance is not None:
        values = np.asarray(allowance, float)
        if values.shape != (len(fields),) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("invalid saved sensor allowance")
    return {"available": True,
            "observations_per_condition_min": counts["SOURCE"].min(0).tolist(),
            "observations_per_condition_max": counts["SOURCE"].max(0).tolist(),
            "SOURCE_mse": errors["SOURCE"].mean(0).tolist(),
            "candidate_mse": errors["candidate"].mean(0).tolist(),
            "saved_sensor_allowance": allowance,
            "saved_sensor_pass": safeguard.get("sensor_pass")}


def assemble(base, spatial):
    check_matched(base, spatial)
    ids = base["assessment"]["sample_ids"]
    index = np.random.default_rng(SEED).integers(0, len(ids), size=(REPLICATES, len(ids)))
    bv, weights = family_arrays(base); sv, sw = family_arrays(spatial)
    if weights != sw: raise ValueError("Scale prefit weights differ")
    result = {"families": {}, "scale_context": {}, "fields": [],
              "bootstrap": {"seed": SEED, "replicates": REPLICATES, "unit": "paired frame; one trajectory"},
              "prefit_weights": weights,
              "scale_prefit_coefficients": {key: float(base["decomposition"]["prefit_coefficients"]["scale." + key]) for key in BLOCKS}}
    for family in FAMILIES:
        source, b, s = bv["SOURCE"][family], bv["candidate"][family], sv["candidate"][family]
        if not np.allclose(source, sv["SOURCE"][family], rtol=1e-10, atol=1e-10):
            raise ValueError(f"SOURCE {family} differs across arms")
        result["families"][family] = {"SOURCE_mean_ARS": source.mean(0).tolist(),
            BASE + "_mean_ARS": b.mean(0).tolist(), SPATIAL + "_mean_ARS": s.mean(0).tolist(),
            BASE + "_minus_SOURCE": pair_interval(b, source, index),
            SPATIAL + "_minus_SOURCE": pair_interval(s, source, index),
            "SpatialShape_minus_BaseShape": pair_interval(s, b, index)}
    result["components"] = component_comparison({BASE: base, SPATIAL: spatial}, index, weights)
    result["relation_details"] = detail_comparison({BASE: base, SPATIAL: spatial}, index)
    for side, family in (("SOURCE", bv["SOURCE"]), (BASE, bv["candidate"]), (SPATIAL, sv["candidate"])):
        key = "SOURCE_mean_ARS" if side == "SOURCE" else side + "_mean_ARS"
        weighted = sum(np.asarray(item[key]) * item["fixed_contribution_factor"] for item in result["components"]["Scale blocks"].values())
        ph = sum(np.asarray(item[key]) for item in result["components"]["PH Shape components"].values())
        if not np.allclose(weighted, family["Complete expanded Scale"].mean(0), rtol=1e-8, atol=1e-10) or not np.allclose(ph, family["PH Shape"].mean(0), rtol=2e-5, atol=1e-7):
            raise ValueError("Scale blocks or PH components do not reconstruct the reported family")
    fields = base["assessment"]["field_order"]
    risks = {name: np.asarray([row["member_mse"] for row in endpoint["assessment"]["per_condition_fields"][side]], float)
             for name, endpoint, side in (("SOURCE", base, "SOURCE"), (BASE, base, "candidate"), (SPATIAL, spatial, "candidate"))}
    if any(v.shape != (len(ids), len(fields)) or not np.isfinite(v).all() for v in risks.values()) or (risks["SOURCE"].mean(0) <= 0).any():
        raise ValueError("positive matched native member-MSE arrays required")
    source = risks["SOURCE"]
    for j, field in enumerate(fields):
        row = {"field": field, "unit": base["assessment"]["field_units"][j], "SOURCE_pooled_member_mse": float(source[:, j].mean())}
        for arm, endpoint in ((BASE, base), (SPATIAL, spatial)):
            saved = endpoint["member"]["fields"][j]
            diagnostics = member_diagnostics(saved)
            if saved["field"] != field or not np.isclose(saved["source"]["member_mse"], row["SOURCE_pooled_member_mse"], rtol=1e-10):
                raise ValueError("saved member summary field order or SOURCE mean differs")
            if not np.isclose(diagnostics["candidate"]["member_mse"], risks[arm][:, j].mean(), rtol=1e-10):
                raise ValueError("saved candidate member mean differs")
            if "SOURCE_diagnostics" in row and any(not np.isclose(row["SOURCE_diagnostics"][key], value, rtol=1e-10, atol=1e-12)
                                                    for key, value in diagnostics["source"].items()):
                raise ValueError("saved SOURCE member diagnostics differ across arms")
            row["SOURCE_diagnostics"] = diagnostics["source"]
            if not np.isclose(saved["member_ratio"], risks[arm][:, j].mean() / source[:, j].mean(), rtol=1e-10):
                raise ValueError("member ratio must be a ratio of pooled MSE")
            if not np.isclose(saved["member_margin"], (risks[arm][:, j] - 1.1*source[:, j]).mean(), rtol=1e-8, atol=1e-10):
                raise ValueError("saved member margin differs from the prospective bound")
            row[arm] = {k: saved[k] for k in ("member_ratio", "member_ratio_interval_95", "member_margin", "member_margin_interval_95", "member_point_pass", "old_signed_point_pass")}
            row[arm]["diagnostics"] = diagnostics["candidate"]
        diff = risks[SPATIAL][:, j] - risks[BASE][:, j]
        boot = diff[index].mean(1) / source[:, j][index].mean(1)
        row["SpatialShape_minus_BaseShape_member_mse"] = pair_interval(risks[SPATIAL][:, j], risks[BASE][:, j], index)
        row["SpatialShape_minus_BaseShape_ratio_to_SOURCE"] = {"mean": float(diff.mean() / source[:, j].mean()),
                                                                  "interval_95": np.quantile(boot, [.025, .975]).tolist()}
        result["fields"].append(row)
    for key in SCALE_KEYS:
        def values(endpoint, side):
            return np.asarray([row["models"][side]["scale"][key] for row in endpoint["relation"]["records"]], float)
        source_values, b, s = values(base, "SOURCE"), values(base, "candidate"), values(spatial, "candidate")
        if source_values.shape != b.shape or b.shape != s.shape or not np.isfinite([source_values, b, s]).all():
            raise ValueError(f"invalid Scale context {key}")
        result["scale_context"][key] = {"SOURCE_mean": source_values.mean(0).tolist(), BASE + "_mean": b.mean(0).tolist(),
            SPATIAL + "_mean": s.mean(0).tolist(), "SpatialShape_minus_BaseShape": pair_interval(s, b, index)}
    for key in ("reference_power_over_floor", "reference_low_energy"):
        x = np.asarray([row["models"]["SOURCE"]["scale"][key] for row in base["relation"]["records"]], float)
        result["scale_context"][key + "_mean"] = x.mean(0).tolist()
    result["scale_context"]["field_pair_indices"] = base["relation"]["field_pair_indices"]
    result["sensor_compatibility"] = {endpoint["arm"]: sensor_evidence(endpoint)
                                      for endpoint in (base, spatial)}
    result["examples"] = base["member"]["example_selection"]
    result["sample_ids"] = ids
    return result


def nomination_check(path, base, spatial):
    path = Path(path).resolve(strict=True); decision = load(path)
    if (decision.get("schema") != "car_round8_learning_review_epoch_100_v1" or decision.get("status") != "sealed"
            or decision.get("comparison_epoch") != 100 or not isinstance(decision.get("sealed_unix"), (int, float))
            or not math.isfinite(decision["sealed_unix"]) or decision.get("tasks_excluded_from_nomination") is not True
            or not isinstance(decision.get("reason"), str) or not decision["reason"].strip()
            or decision.get("selected_arm") not in (None, BASE, SPATIAL)
            or not isinstance(decision.get("continue_to_epoch"), int)
            or not 100 <= decision["continue_to_epoch"] <= 500
            or decision.get("selected_arm") is None and decision["continue_to_epoch"] != 100):
        raise ValueError("sealed post-comparison epoch-100 nomination required")
    checked_ref(decision["prepared"])
    learning_decision = load(checked_ref(decision["learning_decision"]))
    if (learning_decision.get("status") != "sealed" or learning_decision.get("admitted") is not True
            or learning_decision.get("test_locked") is not True):
        raise ValueError("nomination must bind the sealed learning admission")
    for endpoint in (base, spatial):
        arm = endpoint["arm"]
        if (not same_ref(decision["assessments"].get(arm), ref(endpoint["result_path"]))
                or not same_ref(decision["checkpoints"].get(arm), endpoint["training"]["checkpoint"])
                or not same_ref(decision["prepared"], endpoint["plan"]["prepared"])):
            raise ValueError("nomination does not bind both epoch-100 assessments and checkpoints")
    return path, decision


def released_tasks(path, endpoint, nomination_path, nomination):
    result_path = Path(path).resolve(strict=True); result = load(result_path)
    ids = endpoint["assessment"]["sample_ids"]; require_panel(result, ids)
    if (result.get("schema") != "car_round8_validation_task_assessment_v1"
            or result.get("arm") != endpoint["arm"] or result.get("epoch") != endpoint["epoch"]
            or not same_ref(result.get("plan"), endpoint["result"]["plan"])
            or not same_ref(result.get("checkpoint"), endpoint["training"]["checkpoint"])
            or not same_ref(result.get("native_bank"), endpoint["banks"]["native"]["ref"])):
        raise ValueError("task result differs from the scored endpoint")
    release = load(checked_ref(result["release"]))
    if (release.get("schema") != "car_round7_task_release_v1" or release.get("status") != "sealed"
            or release.get("split") != "validation" or release.get("test_locked") is not True
            or release.get("selection_use") is not False or not same_ref(release.get("nomination"), ref(nomination_path))
            or not isinstance(release.get("released_unix"), (int, float))
            or not math.isfinite(release["released_unix"]) or release["released_unix"] < nomination["sealed_unix"]
            or release.get("child_checkpoint_sha256") != endpoint["training"]["checkpoint"]["sha256"]
            or release.get("child_native_receipt_sha256") != endpoint["banks"]["native"]["ref"]["sha256"]
            or not isinstance(release.get("reason"), str) or not release["reason"].strip()):
        raise ValueError("task release must be sealed after and bound to this nomination, checkpoint and bank")
    task = load(checked_ref(result["tasks"])); receipt = load(checked_ref(result["task_receipt"]))
    require_panel(task, ids)
    if (receipt.get("status") != "completed" or not same_ref(task.get("decision"), result["release"])
            or not same_ref(task.get("child_native"), endpoint["banks"]["native"]["ref"])
            or not same_ref(task.get("source_native"), endpoint["assessment"]["source_native"])
            or not same_ref(task.get("spec"), endpoint["plan"]["spec"])):
        raise ValueError("released task evidence differs from its assessment")
    return {"arm": endpoint["arm"], "epoch": endpoint["epoch"], "paired_frozen_train_units": task["paired"],
            "result": ref(result_path), "release": result["release"], "selection_use": False}


def training_evidence(endpoint, campaign, population):
    """Bound saved histories to the endpoint and report measured exposure/cost."""
    epoch = endpoint["epoch"]
    run = Path(endpoint["training"]["run_dir"]).resolve(strict=True)
    paths = {"history": run / "logs/history.jsonl", "timing": run / "logs/epoch_timing.jsonl",
             "recording_definitions": run / "logs/recording_definitions.json",
             "coupling": run / "metrics/training_coupling.jsonl"}
    rows = [r for r in rows_jsonl(paths["history"]) if r.get("epoch", 0) <= epoch]
    if ([r.get("epoch") for r in rows] != list(range(1, epoch + 1))
            or any(r.get("step") != 12 * r["epoch"] or r.get("epoch_complete") is not True for r in rows)):
        raise ValueError("training history is not complete and ordered through the endpoint")
    timings = [r for r in rows_jsonl(paths["timing"]) if r.get("epoch", 0) <= epoch] if paths["timing"].is_file() else []
    complete = [r for r in timings if r.get("complete_epoch_duration") is True and r.get("step") == 12*r.get("epoch", 0)]
    if len({r["epoch"] for r in complete}) != len(complete):
        raise ValueError("duplicate completed-epoch timing records")
    timing_complete = sorted(r["epoch"] for r in complete) == list(range(1, epoch + 1))
    coupling = [r for r in rows_jsonl(paths["coupling"]) if r.get("step", 0) <= 12*epoch]
    identities = ("observation_sha256", "query_sha256", "first_member_generator_state_sha256", "field_complement_sha256")
    if ([r.get("step") for r in coupling] != list(range(1, 12*epoch + 1))
            or any(not isinstance(r.get("sample_ids"), list) or not r["sample_ids"] or not isinstance(r.get("accepted"), bool)
                   or not isinstance(r.get("member_seeds"), list) or len(set(r["member_seeds"])) != len(r["member_seeds"])
                   or not r["member_seeds"] or any(not isinstance(r.get(k), str) or not r[k] for k in identities)
                   or not isinstance(r.get("field_complement_seed"), int) for r in coupling)):
        raise ValueError("complete ordered training coupling records required")
    definitions = load(paths["recording_definitions"])
    columns = definitions.get("columns", [])
    if (definitions.get("version") != "ordered_epoch_recording_v1" or not columns
            or any(not all(k in col for k in ("name", "type", "units", "meaning")) for col in columns)
            or not isinstance(definitions.get("accumulation_order"), list)
            or not isinstance(definitions.get("loss_component_accumulation_order"), list)):
        raise ValueError("ordered training recording definitions required")
    token = endpoint["training"]["lease_token"]
    lease_path = Path(campaign) / "operations/completions" / (token + ".json")
    lease = load(lease_path) if lease_path.is_file() else None
    if lease is not None and (lease.get("schema") != "car_round8_gpu_lease_v1" or lease.get("status") != "completed"
            or lease.get("phase") != "learning" or lease.get("token") != token
            or not isinstance(lease.get("seconds"), (int, float)) or lease["seconds"] <= 0):
        raise ValueError("learning lease receipt differs from the completed endpoint")
    accepted = sum(len(row["sample_ids"]) for row in coupling if row["accepted"])
    attempted = sum(len(row["sample_ids"]) for row in coupling)
    epoch_exposure = []
    for i in range(epoch):
        group = coupling[12*i:12*(i+1)]
        epoch_exposure.append({"epoch": i+1, "accepted_actions": sum(row["accepted"] for row in group),
            "objective_condition_visits": sum(len(row["sample_ids"]) for row in group if row["accepted"]),
            "member_counts": sorted({len(row["member_seeds"]) for row in group})})
    deviations = [x for x in epoch_exposure if x["accepted_actions"] != 12 or x["objective_condition_visits"] != 768 or x["member_counts"] != [2]]
    if not isinstance(population, int) or population <= 0: raise ValueError("TRAIN population required")
    def weighted_time(key):
        values = [row.get(key) for row in rows]
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in values): return None
        return float(sum(row[key] * row["batches"] for row in rows))
    wall = float(sum(row["wall_seconds"] for row in complete)) if complete else None
    train = float(sum(row["train_seconds"] for row in complete)) if complete else None
    if (wall is not None and (not math.isfinite(wall) or wall <= 0)
            or train is not None and (not math.isfinite(train) or train <= 0)):
        raise ValueError("invalid measured training timing")
    mature = [row for row in complete if 51 <= row["epoch"] <= min(100, epoch)]
    def mature_mean(key):
        return float(np.mean([row[key] for row in mature])) if mature else None
    mature_summary = {"epochs": "51–100", "complete_epoch_records": len(mature),
        "complete_coverage": len(mature) == 50,
        "mean_wall_seconds": mature_mean("wall_seconds"), "mean_train_span_seconds": mature_mean("train_seconds"),
        "mean_callback_and_reporting_seconds": float(np.mean([row.get("callback_and_reporting_seconds", max(0., row["wall_seconds"] - row["train_seconds"])) for row in mature])) if mature else None}
    curves = {}
    for png in sorted((run / "logs").glob("*.png")):
        pdf = png.with_suffix(".pdf")
        if pdf.is_file(): curves[png.stem] = {"png": png, "pdf": pdf}
    if "loss_history" not in curves:
        raise ValueError("completed training loss_history PNG and PDF are required")
    all_history = rows_jsonl(paths["history"])
    latest = max(int(row["epoch"]) for row in all_history)
    return {"run": run, "paths": paths, "curves": curves, "coupling_rows": coupling,
            "summary": {"arm": endpoint["arm"], "reporting_epoch": epoch, "history_saved_through_epoch": latest,
                "accepted_objective_condition_visits": accepted, "attempted_condition_visits": attempted,
                "epoch_exposure": epoch_exposure, "expected_exposure_deviations": deviations,
                "train_population": population, "dataset_pass_equivalents": accepted / population,
                "completed_epoch_wall_seconds": wall, "completed_epoch_train_span_seconds": train,
                "mean_complete_epoch_wall_seconds": wall / len(complete) if complete else None,
                "timing_covered_epochs": len(complete), "timing_complete_through_endpoint": timing_complete,
                "mature_epochs_51_100": mature_summary,
                "summed_update_seconds": weighted_time("update_seconds"),
                "summed_data_wait_seconds": weighted_time("data_wait_seconds"),
                "lease_segment_occupied_gpu_seconds": float(lease["seconds"]) if lease else None,
                "actual_device": lease.get("device") if lease else None,
                "whole_worker_torch_memory": endpoint["training"].get("torch_memory", {"scope": "unavailable"}),
                "lease_scope": "one recorded learning lease segment, including startup and shutdown; not necessarily all earlier resume segments",
                "timing_scope": "epoch timing includes data waiting; update and wait totals are sums of recorded epoch means times batch counts when present",
                "recording_definitions": definitions, "lease_completion": ref(lease_path) if lease else None}}


def coupling_comparison(left, right):
    a, b = left["coupling_rows"], right["coupling_rows"]
    if len(a) != len(b): raise ValueError("training coupling ages differ")
    keys = ("sample_ids", "observation_sha256", "query_sha256", "member_seeds", "first_member_generator_state_sha256",
            "field_complement_sha256", "field_complement_seed", "accepted")
    return {key: sum(x.get(key) != y.get(key) for x, y in zip(a, b, strict=True)) for key in keys}


def selected_training_evidence(initial, previous, final, campaign, population):
    """Read the selected continuation in its original run and count leases once."""
    if (initial["arm"] != final["arm"]
            or Path(initial["training"]["run_dir"]).resolve() != Path(final["training"]["run_dir"]).resolve()):
        raise ValueError("selected continuation must use the epoch-100 training run")
    evidence = training_evidence(final, campaign, population)
    leases = {initial["training"]["lease_token"]: previous["summary"]["lease_segment_occupied_gpu_seconds"],
              final["training"]["lease_token"]: evidence["summary"]["lease_segment_occupied_gpu_seconds"]}
    evidence["summary"]["lineage_occupied_gpu_seconds"] = (
        sum(leases.values()) if all(value is not None for value in leases.values()) else None)
    return evidence


def assessment_evidence(endpoint, campaign):
    token = endpoint["generation"]["lease_token"]
    path = Path(campaign) / "operations/completions" / (token + ".json")
    lease = load(path) if path.is_file() else None
    if lease is not None and (lease.get("schema") != "car_round8_gpu_lease_v1" or lease.get("status") != "completed"
            or lease.get("phase") != "assessment" or lease.get("token") != token
            or not isinstance(lease.get("seconds"), (int, float)) or lease["seconds"] <= 0):
        raise ValueError("generation lease receipt differs from the completed endpoint")
    result = endpoint["result"]
    return {"generation_lease_occupied_gpu_seconds": float(lease["seconds"]) if lease else None,
            "generation_lease": ref(path) if lease else None,
            "generation_device": lease.get("device") if lease else None, "cpu_score_wall_seconds": float(result["wall_seconds"]),
            "cpu_score_process_seconds": float(result["cpu_seconds"]),
            "cpu_score_reused_completed_scores": bool(result["reused_completed_cpu_scores"]),
            "scope": "generation lease includes startup and shutdown; CPU score timing is the final scoring invocation, which may reuse completed work"}


def observer_identity(endpoint, fit):
    position = endpoint["position"]
    common = load(checked_ref(position["cached_source"]))
    require_panel(common, endpoint["assessment"]["sample_ids"])
    if (common.get("schema") != "car_round8_common_position_readout_v1"
            or not same_ref(common["inputs"]["fit"], position["fit"])
            or not same_ref(common["inputs"]["contract"], endpoint["spec"]["assessment_contract"])
            or common["kind_weights"] != position["kind_weights"]
            or common["location_unit"] != position["location_unit"]):
        raise ValueError("saved common positional query law differs")
    contract = load(checked_ref(endpoint["spec"]["assessment_contract"]))
    return {"sample_ids": endpoint["assessment"]["sample_ids"], "members": 4,
        "field_order": endpoint["assessment"]["field_order"], "field_units": endpoint["assessment"]["field_units"],
        "physical_mask": endpoint["assessment"]["physical_mask"],
        "source_native": endpoint["assessment"]["source_native"], "source_selected": endpoint["assessment"]["source_selected"],
        "source_baseline": endpoint["assessment"]["source_baseline"],
        "frozen_components": {key: endpoint["spec"][key] for key in SPEC_KEYS},
        "prefit_coefficients": endpoint["decomposition"]["prefit_coefficients"],
        "frozen_train_normalizers": contract["normalizers"], "position_fit": position["fit"],
        "position_location_unit": position["location_unit"], "observation_kind_weights": position["kind_weights"],
        "observation_query_batches": common["query_batches"], "observation_weighted_queries": common["queries"],
        "fit_query_rule": fit["query_rule"], "fit_query_count": fit["query_count"],
        "fit_kind_weights": fit["kind_weights"], "common_position_source": position["cached_source"],
        "common_state_scale_source": endpoint["relation"]["inputs"]["source_readout"],
        "common_state_scale_detail_source": endpoint["relation"]["inputs"].get("source_companion")}


def final_summary(endpoint, source_endpoint):
    """Describe a selected later endpoint without asserting equal exposure."""
    check_common(source_endpoint, endpoint)
    for saved, source_saved in zip(endpoint["member"]["fields"], source_endpoint["member"]["fields"], strict=True):
        source = member_diagnostics(saved)["source"]
        original = member_diagnostics(source_saved)["source"]
        if saved["field"] != source_saved["field"] or any(not np.isclose(source[k], original[k], rtol=1e-10, atol=1e-12) for k in RISK_DIAGNOSTICS):
            raise ValueError("selected endpoint SOURCE member diagnostics differ")
    arrays, weights = family_arrays(endpoint)
    index = np.random.default_rng(SEED).integers(0, 100, size=(REPLICATES, 100))
    families = {name: {"SOURCE_mean_ARS": arrays["SOURCE"][name].mean(0).tolist(),
                       "candidate_mean_ARS": arrays["candidate"][name].mean(0).tolist(),
                       "candidate_minus_SOURCE": pair_interval(arrays["candidate"][name], arrays["SOURCE"][name], index)}
                for name in FAMILIES}
    return {"arm": endpoint["arm"], "epoch": endpoint["epoch"], "exposure": "selected later endpoint; unequal training age versus epoch-100 arms",
            "fields": endpoint["member"]["fields"], "families": families, "prefit_weights": weights,
            "scale_prefit_coefficients": {key: float(endpoint["decomposition"]["prefit_coefficients"]["scale." + key]) for key in BLOCKS},
            "components": component_comparison({endpoint["arm"]: endpoint}, index, weights),
            "relation_details": detail_comparison({endpoint["arm"]: endpoint}, index),
            "scale_context": endpoint["relation"]["summary"]["scale"],
            "sensor_compatibility": sensor_evidence(endpoint)}


def make_figures(data, base, spatial, final, folder):
    captions = {}; fields = [r["field"] for r in data["fields"]]; x = np.arange(len(fields))
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    for shift, arm, color in ((-.12, BASE, "#397aa6"), (.12, SPATIAL, "#b14e40")):
        mean = np.array([r[arm]["member_ratio"] for r in data["fields"]]); ci = np.array([r[arm]["member_ratio_interval_95"] for r in data["fields"]])
        ax.vlines(x + shift, ci[:, 0], ci[:, 1], color=color)
        ax.scatter(x + shift, mean, label=arm, color=color)
    ax.axhline(1., color="0.5", lw=1); ax.axhline(1.1, color="#a04446", ls="--", label="Prospective 1.10 limit")
    ax.set_xticks(x, fields); ax.set_ylabel("Pooled member MSE / matched SOURCE"); ax.set_title("Matched VALIDATION100, reporting epoch 100"); ax.legend(frameon=False)
    save_figure(fig, folder, "learning_field_ratios")
    captions["learning_field_ratios"] = "Native physical-mass member-MSE ratios at matched reporting epoch 100, each a ratio of pooled risks; whiskers are paired-frame 95% intervals and the dashed line is the prospective 1.10 point limit."
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for ax, family in zip(axes.flat, FAMILIES, strict=True):
        item = data["families"][family]["SpatialShape_minus_BaseShape"]
        mean, ci = np.array(item["mean"]), np.array(item["interval_95"])
        ax.vlines(np.arange(3), ci[0], ci[1], color="#9a4943"); ax.scatter(np.arange(3), mean, color="#9a4943")
        ax.axhline(0, color="0.5", lw=1); ax.set_xticks(range(3), ("A", "R", "S")); ax.set_title(family); ax.set_ylabel("Paired raw-observer difference")
    save_figure(fig, folder, "learning_paired_direct_ars")
    captions["learning_paired_direct_ars"] = "SpatialShape minus BaseShape at matched reporting epoch 100 on the same conditions: direct member discrepancy A, positive member separation R and fair score S = A − R, with paired-frame 95% intervals. Each panel has its own observer units."
    rho = np.asarray(data["scale_context"]["rho_signed_error_mean_member"]["SpatialShape_minus_BaseShape"]["mean"])
    power = np.asarray(data["scale_context"]["log1p_power_over_floor_signed_error_mean_member"]["SpatialShape_minus_BaseShape"]["mean"])
    pairs = [f"{fields[i]} / {fields[j]}" for i, j in data["scale_context"]["field_pair_indices"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    for ax, values, labels, title in ((axes[0], rho.T, pairs, "Signed rho error"), (axes[1], power.T, fields, "Signed log1p(power / floor) error")):
        limit = max(float(np.abs(values).max()), 1e-12)
        im = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        ax.set_yticks(range(len(labels)), labels); ax.set_xticks(range(values.shape[1]), [f"band {i+1}" for i in range(values.shape[1])]); ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=.8)
    save_figure(fig, folder, "learning_signed_rho_power")
    captions["learning_signed_rho_power"] = "Matched-epoch SpatialShape minus BaseShape mean signed member-minus-reference rho and log1p(power/floor) errors on the frozen selected support. Fixed basis bands are not physical wavelengths; power and low-energy floor context is retained in the summary."
    captions.update(example_figures(data, base, spatial, final, folder))
    return captions


def example_figures(data, base, spatial, final, folder):
    """Use only the two already SOURCE-selected IDs and saved native fields."""
    ids, fields = data["sample_ids"], base["assessment"]["field_order"]
    if "CO" not in fields or "T" not in fields: raise ValueError("CO and T field examples required")
    captions = {}
    endpoints = [(f"{BASE} epoch 100", base), (f"{SPATIAL} epoch 100", spatial)] + (
        [(f"{final['arm']} epoch {final['epoch']}", final)] if final else [])
    for choice, title in (("representative_median", "Representative"), ("difficult_maximum", "Difficult")):
        sid = data["examples"][choice]["sample_id"]
        if sid not in ids: raise ValueError("saved SOURCE-selected example is outside the matched panel")
        ordinal = ids.index(sid); src = base["source_bank"]
        source_row = read_row(src["path"], src["receipt"], ordinal)
        truth, source_members = physical_native(source_row); values = [("SOURCE", source_members.mean(0))]
        for label, endpoint in endpoints:
            bank = endpoint["banks"]["native"]
            row = read_row(bank["path"], bank["receipt"], ordinal); matched_rows(source_row, row)
            _, members = physical_native(row); values.append((label, members.mean(0)))
        mass = source_row["native_masses"].double().numpy(); mask = mass > 0
        order, shape, extent = native_grid(source_row)
        fig, axes = plt.subplots(2, 1 + len(values), figsize=(3.25*(1+len(values)), 7.6), squeeze=False, constrained_layout=True)
        for line, field in enumerate(("CO", "T")):
            i = fields.index(field); reference = truth[:, i]; all_fields = [reference] + [value[:, i] for _, value in values]
            lo = min(float(v[mask].min()) for v in all_fields); hi = max(float(v[mask].max()) for v in all_fields)
            for ax, arr, label in zip(axes[line], all_fields, ("Reference", *[x[0] for x in values]), strict=True):
                grid = np.ma.array(arr[order].reshape(shape), mask=(~mask[order]).reshape(shape))
                im = ax.imshow(grid, origin="lower", extent=extent, aspect="auto", cmap="viridis", vmin=lo, vmax=hi)
                heading = label.replace("_", "_\n", 1).replace(" epoch ", "\nepoch ")
                suffix = "" if label == "Reference" else f"\nphysical RMSE {physical_rmse(arr, reference, mass):.4g}"
                ax.set_title(heading + suffix, fontsize=12); ax.set_xlabel("coordinate 1 (dataset units)", fontsize=12)
                if ax is axes[line, 0]: ax.set_ylabel(field + "\ncoordinate 2 (dataset units)", fontsize=13)
                ax.tick_params(labelsize=10); fig.colorbar(im, ax=ax, fraction=.045, pad=.02).ax.tick_params(labelsize=10)
        fig.suptitle(f"{title} SOURCE-selected frame {sid}: native fields", fontsize=16)
        name = "learning_" + choice + "_CO_T"; save_figure(fig, folder, name)
        captions[name] = (f"{title} previously SOURCE-selected frame {sid}. Each native CO and T row shares physical field color limits across reference and model means; every reconstruction is annotated with physical-mass-weighted RMSE in dataset units. Axes are stretched independently for readability.")
    return captions


def capture_training(evidence_by_arm, output, captions):
    """Snapshot histories and include each saved paired training curve."""
    snapshots = {}
    for arm, evidence in evidence_by_arm.items():
        recording = output / "recordings" / arm; recording.mkdir(parents=True)
        files = {}
        for key, source in evidence["paths"].items():
            if not source.is_file(): continue
            target = recording / source.name; shutil.copyfile(source, target)
            if ref(source)["sha256"] != ref(target)["sha256"]: raise ValueError("training recording copy differs")
            files[key] = ref(target)
        for stem, pair in evidence["curves"].items():
            name = "training_" + arm + "_" + stem
            for ext, source in pair.items():
                target = output / "figures" / f"{name}.{ext}"; shutil.copyfile(source, target)
                if ref(source)["sha256"] != ref(target)["sha256"]: raise ValueError("training curve copy differs")
            captions[name] = (f"Saved {stem.replace('_', ' ')} for {arm}, through reporting epoch {evidence['summary']['history_saved_through_epoch']}. Sampled training components, validation metrics and gradient diagnostics retain their recorded meanings; ConFIG has no invented scalar objective.")
        snapshots[arm] = files
    return snapshots


def format_value(value):
    return "unavailable" if value is None else f"{float(value):.5g}"


def interval_text(value):
    lo, hi = value["interval_95"]
    return f"{format_value(value['mean'])} [{format_value(lo)}, {format_value(hi)}]"


def diagnostic_table(rows, models):
    """Compact cells list mean-output MSE, signed mean risk, then member variance."""
    lines = ["| Field | " + " | ".join(models) + " |", "| --- | " + " | ".join(["---:"] * len(models)) + " |"]
    for row in rows:
        cells = [" / ".join(format_value(row[model][key]) for key in RISK_DIAGNOSTICS[1:]) for model in models]
        lines.append("| " + row["field"] + " | " + " | ".join(cells) + " |")
    return lines


def sensor_table(evidence, fields, labels):
    """Show dimensional saved tolerances, including a true zero SOURCE MSE."""
    if not all(item.get("available") for item in evidence.values()):
        return ["", "The saved observation-safeguard readout is unavailable for at least one endpoint.", ""]
    first = next(iter(evidence.values()))
    for item in evidence.values():
        for key in ("SOURCE_mse", "observations_per_condition_min", "observations_per_condition_max",
                    "saved_sensor_allowance"):
            if item[key] != first[key]:
                raise ValueError("common SOURCE observation errors, counts or allowance differ")
    lines = ["", "The saved observation safeguard concerns only fields with actual observations. Sensor MSE and its allowed increase are in each field's squared dataset units; errors are averaged over the matched conditions. Unobserved fields are omitted.", "",
        "| Observed field | Count range | SOURCE sensor MSE | "
        + " | ".join(labels[arm] + " sensor MSE" for arm in evidence) + " | Allowed increase |",
        "| --- | ---: | ---: | " + " | ".join(["---:"] * len(evidence)) + " | ---: |"]
    for index, field in enumerate(fields):
        low, high = (first[key][index] for key in
                     ("observations_per_condition_min", "observations_per_condition_max"))
        if high:
            allowance = (None if first["saved_sensor_allowance"] is None else
                         first["saved_sensor_allowance"][index])
            cells = [format_value(item["candidate_mse"][index]) for item in evidence.values()]
            lines.append(f"| {field} | {low}–{high} | {format_value(first['SOURCE_mse'][index])} | "
                         + " | ".join(cells) + f" | {format_value(allowance)} |")
    statuses = "; ".join(f"{labels[arm]}: {item['saved_sensor_pass']}" for arm, item in evidence.items())
    lines += ["", "The saved rule is child sensor MSE minus SOURCE sensor MSE <= allowed increase. When SOURCE sensor MSE is zero, the saved allowance is its declared absolute numerical tolerance; no zero-baseline ratio is formed.",
              "", f"The saved overall sensor-safeguard result is {statuses}.", ""]
    return lines


def relation_detail_tables(details, arms):
    contrast = "SpatialShape − BaseShape" if len(arms) == 2 else "Selected − SOURCE"
    weights, geometry = details["state_group_weights"], details["piece_geometry"]
    lines = ["", f"State marginal and mixed groups below use their separate weighted distances (frozen weights {format_value(weights['marginal'])} and {format_value(weights['mixed'])}). Each reports direct A / R / S; their scores are not additive components of the total root-norm State score.", "",
        "| State group | SOURCE A / R / S | " + " | ".join(arm + " A / R / S" for arm in arms) + f" | {contrast} S [95% CI] |",
        "| --- | --- | " + " | ".join(["---"] * len(arms)) + " | ---: |"]
    def ars(item, key):
        return " / ".join(format_value(x) for x in item[key])
    def scalar(item, position):
        mean = np.asarray(item["mean"])[position]
        lower, upper = (np.asarray(bound)[position] for bound in item["interval_95"])
        return interval_text({"mean": mean, "interval_95": [lower, upper]})
    for group, item in details["state_groups"].items():
        models = " | ".join(ars(item, arm + "_mean") for arm in arms)
        lines.append(f"| {group} | {ars(item, 'SOURCE_mean')} | {models} | {scalar(item['paired_contrast'], 2)} |")
    masses = ", ".join(f"{piece}: {format_value(mass)}" for piece, mass in zip(geometry["piece_indices"], geometry["mass_fractions"], strict=True))
    lines += ["", f"Selected-support piece-mean entries are signed / absolute member-minus-reference errors, averaged over members and frames in each field's model-normalized units. Absolute values are taken before member averaging. Frozen physical-mass fractions by piece are {masses}; these are not squared field errors or joint-norm attributions.", "",
        "| Piece | Field | SOURCE signed / absolute | " + " | ".join(arm + " signed / absolute" for arm in arms) + " |",
        "| ---: | --- | ---: | " + " | ".join(["---:"] * len(arms)) + " |"]
    signed, absolute = (details["piece_mean"][key] for key in ("signed_error_mean_member", "absolute_error_mean_member"))
    for i, piece in enumerate(geometry["piece_indices"]):
        for j, field in enumerate(geometry["field_order"]):
            def values(key):
                return format_value(signed[key][i][j]) + " / " + format_value(absolute[key][i][j])
            models = " | ".join(values(arm + "_mean") for arm in arms)
            lines.append(f"| {piece} | {field} | {values('SOURCE_mean')} | {models} |")
    lines += ["", f"| Piece | Field | {contrast} signed [95% CI] | {contrast} absolute [95% CI] |",
              "| ---: | --- | ---: | ---: |"]
    for i, piece in enumerate(geometry["piece_indices"]):
        for j, field in enumerate(geometry["field_order"]):
            lines.append(f"| {piece} | {field} | {scalar(signed['paired_contrast'], (i, j))} | {scalar(absolute['paired_contrast'], (i, j))} |")
    return lines


def component_tables(components, coefficients, arms):
    contrast = "SpatialShape − BaseShape" if len(arms) == 2 else "Selected − SOURCE"
    coefficient_text = ", ".join(f"{key}={format_value(coefficients[key])}" for key in BLOCKS)
    lines = ["", f"Scale block entries below are raw A / R / S before prefit weighting, in their frozen observer units. Their fixed contribution factor multiplies each raw block in Complete Expanded Prefit. Frozen prefit coefficients are {coefficient_text}; the joint factor additionally includes the fixed 0.25 multiplier. Weighted differences are contributions to the complete score, not separately refitted scores.", "",
        "| Raw Scale block | SOURCE A / R / S | " + " | ".join(arm + " A / R / S" for arm in arms) + " |",
        "| --- | --- | " + " | ".join(["---"] * len(arms)) + " |"]
    def score_interval(item):
        return {"mean": item["mean"][2], "interval_95": [bound[2] for bound in item["interval_95"]]}
    def ars(item, key):
        return " / ".join(format_value(x) for x in item[key])
    for name, item in components["Scale blocks"].items():
        model_values = " | ".join(ars(item, arm + "_mean_ARS") for arm in arms)
        lines.append(f"| {name} | {ars(item, 'SOURCE_mean_ARS')} | {model_values} |")
    lines += ["", f"| Scale block | Fixed factor | {contrast} raw S [95% CI] | {contrast} weighted S [95% CI] |",
              "| --- | ---: | ---: | ---: |"]
    for name, item in components["Scale blocks"].items():
        lines.append(f"| {name} | {format_value(item['fixed_contribution_factor'])} | {interval_text(score_interval(item['paired_contrast']))} | {interval_text(score_interval(item['weighted_paired_contrast']))} |")
    lines += ["", "PH components use the same frozen self/mutual and finite/essential definitions as the common PH observer; their sum is the reported PH Shape score.", "",
        "| PH component | SOURCE A / R / S | " + " | ".join(arm + " A / R / S" for arm in arms) + f" | {contrast} S [95% CI] |",
        "| --- | --- | " + " | ".join(["---"] * len(arms)) + " | ---: |"]
    for name, item in components["PH Shape components"].items():
        model_values = " | ".join(ars(item, arm + "_mean_ARS") for arm in arms)
        lines.append(f"| {name} | {ars(item, 'SOURCE_mean_ARS')} | {model_values} | {interval_text(score_interval(item['paired_contrast']))} |")
    return lines


def make_section(data, training, tasks, final):
    lines = ["## Round 8 matched learning endpoints", "",
        "The two descriptive packages are compared at the same 100 reporting epochs on the same cached VALIDATION100 conditions, four independent members, SOURCE banks, native physical mask and fixed Q4096 common observer. VALIDATION is a development panel; its paired frame-bootstrap intervals use one trajectory and do not establish independent trajectory transfer.", "",
        "The matched package contrast uses within-condition SpatialShape minus BaseShape differences; its uncertainty does not use overlapping separate SOURCE intervals. Field ratios are ratios of pooled member MSE, and margins are child minus 1.10 times same-panel SOURCE in squared dataset units.", "",
        "| Field | BaseShape ratio [95% CI] | BaseShape margin [95% CI] | SpatialShape ratio [95% CI] | SpatialShape margin [95% CI] | SpatialShape − BaseShape / SOURCE [95% CI] |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in data["fields"]:
        def field_val(arm, kind):
            item = row[arm]; return f"{format_value(item[kind])} [{format_value(item[kind + '_interval_95'][0])}, {format_value(item[kind + '_interval_95'][1])}]"
        lines.append(f"| {row['field']} | {field_val(BASE, 'member_ratio')} | {field_val(BASE, 'member_margin')} | {field_val(SPATIAL, 'member_ratio')} | {field_val(SPATIAL, 'member_margin')} | {interval_text(row['SpatialShape_minus_BaseShape_ratio_to_SOURCE'])} |")
    lines += ["", "Each cell below gives finite-J mean-output MSE / signed mean-risk estimate / unbiased member variance, in the field's squared dataset units on the native physical mask. For J = 4, member MSE = mean-output MSE + 3/4 variance and signed mean risk = mean-output MSE − variance/4. The signed estimate may be negative; these separate diagnostics do not establish posterior calibration.", ""]
    diagnostics = [{"field": row["field"], "SOURCE": row["SOURCE_diagnostics"],
                    BASE: row[BASE]["diagnostics"], SPATIAL: row[SPATIAL]["diagnostics"]} for row in data["fields"]]
    lines += diagnostic_table(diagnostics, ("SOURCE", BASE, SPATIAL))
    lines += sensor_table(data["sensor_compatibility"], [row["field"] for row in data["fields"]],
                          {BASE: "BaseShape", SPATIAL: "SpatialShape"})
    lines += ["", "A is direct member-to-reference discrepancy, R is positive member separation and S = A − R. State uses the exact common State observer; Scale is the complete expanded prefit definition; PH and position use the same frozen observation and target weights for both packages. Raw observer units differ by family.", "",
        "| Common observer | SOURCE A / R / S | BaseShape A / R / S | SpatialShape A / R / S | SpatialShape − BaseShape S [95% CI] |",
        "| --- | --- | --- | --- | --- |"]
    for family, item in data["families"].items():
        fmt = lambda key: " / ".join(format_value(x) for x in item[key])
        delta = item["SpatialShape_minus_BaseShape"]
        score = {"mean": delta["mean"][2], "interval_95": [delta["interval_95"][0][2], delta["interval_95"][1][2]]}
        lines.append(f"| {family} | {fmt('SOURCE_mean_ARS')} | {fmt(BASE + '_mean_ARS')} | {fmt(SPATIAL + '_mean_ARS')} | {interval_text(score)} |")
    lines += component_tables(data["components"], data["scale_prefit_coefficients"], (BASE, SPATIAL))
    lines += relation_detail_tables(data["relation_details"], (BASE, SPATIAL))
    context = data["scale_context"]
    floor = lambda key: float(np.asarray(context[key]).mean())
    member_floor = context["member_low_energy_fraction"]
    avg = lambda name: float(np.asarray(member_floor[name]).mean())
    lines += ["", f"The signed rho and signed log1p(power/floor) panels show member-minus-reference errors on the fixed selected spectral basis. Mean reference low-energy fraction is {format_value(floor('reference_low_energy_mean'))}; SOURCE, BaseShape and SpatialShape member low-energy fractions are {format_value(avg('SOURCE_mean'))}, {format_value(avg(BASE + '_mean'))} and {format_value(avg(SPATIAL + '_mean'))}. Band and floor definitions are frozen, and basis bands are not physical wavelengths.", "",
        "| Package | Completed epochs | Objective-bearing visits | TRAIN dataset-pass equivalents | Mean recorded epoch wall seconds | Occupied GPU seconds in recorded lease segment |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm in (BASE, SPATIAL):
        item = training[arm]
        lines.append(f"| {arm} | {item['reporting_epoch']} | {item['accepted_objective_condition_visits']} | {format_value(item['dataset_pass_equivalents'])} | {format_value(item['mean_complete_epoch_wall_seconds'])} | {format_value(item['lease_segment_occupied_gpu_seconds'])} |")
    lines += ["", "| Package | Mature epoch coverage | Mean wall s | Mean training-span s | Mean callback/reporting s | Worker peak allocated / reserved GiB |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm in (BASE, SPATIAL):
        item = training[arm]; mature = item["mature_epochs_51_100"]; memory = item["whole_worker_torch_memory"]
        gib = lambda key: format_value(memory[key] / 2**30) if isinstance(memory.get(key), (float, int)) else "unavailable"
        lines.append(f"| {arm} | {mature['complete_epoch_records']}/50 | {format_value(mature['mean_wall_seconds'])} | {format_value(mature['mean_train_span_seconds'])} | {format_value(mature['mean_callback_and_reporting_seconds'])} | {gib('peak_allocated_bytes')} / {gib('peak_reserved_bytes')} |")
    lines += ["", "Mature timing uses completed reporting epochs 51–100. Peak allocated and reserved values are each current-worker PyTorch measurements including model/artifact setup and training, with the original scope recorded in the evidence. These measurements do not by themselves transfer a previous same-objective speedup to this changed objective.", ""]
    mismatches = data["training_coupling_mismatch_counts"]
    deviations = {arm: training[arm]["expected_exposure_deviations"] for arm in (BASE, SPATIAL)}
    comparison = ("Recorded training coupling and the declared epoch exposures matched."
        if not any(mismatches.values()) and not any(deviations.values()) else
        "Recorded training coupling or epoch exposure differed, so the package contrast also includes the disclosed training exposure differences.")
    coverage = {arm: f"{training[arm]['timing_covered_epochs']}/{training[arm]['reporting_epoch']}" for arm in (BASE, SPATIAL)}
    lines += ["", f"{comparison} Coupling mismatch counts by identity are {json.dumps(mismatches, sort_keys=True)}; deviations from 12 accepted actions, 768 objective-bearing visits and two training members per epoch are {json.dumps(deviations, sort_keys=True)}. Complete-epoch timing coverage is {json.dumps(coverage, sort_keys=True)}. Epoch timing includes data waiting; saved histories separately retain measured update and data-wait timing where recorded. A lease segment includes startup and shutdown and is not a complete resumed-run cost unless all segments are included.", "",
        "Saved loss and component histories are training diagnostics, while the endpoint scores above are cached held-out measurements. The update uses a ConFIG direction, so no scalar training objective is inferred from its components.", ""]
    for arm in (BASE, SPATIAL):
        saved_epoch = training[arm].get("history_saved_through_epoch", training[arm]["reporting_epoch"])
        if saved_epoch > training[arm]["reporting_epoch"]:
            lines += [f"The {arm} recording snapshots and saved curves include the full run through epoch {saved_epoch}; its matched epoch-100 exposure, timing and coupling above use only records through epoch 100.", ""]
    for arm in (BASE, SPATIAL):
        cost = data["assessment_costs"][arm]
        lines += [f"For {arm} at epoch 100, cached-bank generation occupied its assessment GPU lease for {format_value(cost['generation_lease_occupied_gpu_seconds'])} seconds; the final CPU scoring invocation used {format_value(cost['cpu_score_wall_seconds'])} wall seconds and {format_value(cost['cpu_score_process_seconds'])} CPU seconds. The invocation reused prior completed CPU scores: {str(cost['cpu_score_reused_completed_scores']).lower()}.", ""]
    if final:
        final_memory = data["selected_final_training"]["whole_worker_torch_memory"]
        final_gib = lambda key: format_value(final_memory[key] / 2**30) if isinstance(final_memory.get(key), (float, int)) else "unavailable"
        lines += [f"### Nominated later endpoint: {final['arm']} at epoch {final['epoch']}", "",
            "This selected continuation has more training exposure than either epoch-100 package. Its SOURCE-relative scores are descriptive; the different ages do not identify the positional package effect.", "",
            f"Its saved history records {data['selected_final_training']['accepted_objective_condition_visits']} objective-bearing visits and {format_value(data['selected_final_training']['dataset_pass_equivalents'])} TRAIN dataset-pass equivalents through this endpoint; mean completed-epoch wall time is {format_value(data['selected_final_training']['mean_complete_epoch_wall_seconds'])} seconds. The epoch-100 and continuation training leases together occupied the GPU for {format_value(data['selected_final_training']['lineage_occupied_gpu_seconds'])} seconds, counting each distinct lease once.", "",
            f"The current continuation worker's peak allocated and reserved PyTorch memory was {final_gib('peak_allocated_bytes')} and {final_gib('peak_reserved_bytes')} GiB; scope: {final_memory.get('scope', 'unavailable')}. This is a worker-segment measurement, not an equal-age memory comparison.", "",
            "| Common observer | SOURCE A / R / S | Selected endpoint A / R / S | Endpoint − SOURCE S [95% CI] |",
            "| --- | --- | --- | --- |"]
        for family, item in final["families"].items():
            score = item["candidate_minus_SOURCE"]
            delta = {"mean": score["mean"][2], "interval_95": [score["interval_95"][0][2], score["interval_95"][1][2]]}
            fmt = lambda key: " / ".join(format_value(x) for x in item[key])
            lines.append(f"| {family} | {fmt('SOURCE_mean_ARS')} | {fmt('candidate_mean_ARS')} | {interval_text(delta)} |")
        lines += component_tables(final["components"], final["scale_prefit_coefficients"], (final["arm"],))
        lines += relation_detail_tables(final["relation_details"], (final["arm"],))
        lines.append("")
        lines += ["| Field | Selected member-MSE ratio [95% CI] | Member margin [95% CI] |", "| --- | ---: | ---: |"]
        for row in final["fields"]:
            lines.append(f"| {row['field']} | {format_value(row['member_ratio'])} [{format_value(row['member_ratio_interval_95'][0])}, {format_value(row['member_ratio_interval_95'][1])}] | {format_value(row['member_margin'])} [{format_value(row['member_margin_interval_95'][0])}, {format_value(row['member_margin_interval_95'][1])}] |")
        lines += ["", "The selected endpoint's saved J4 mean-output MSE / signed mean-risk estimate / unbiased member variance are shown in the same squared dataset units and with the same identities as above.", ""]
        diagnostics = [{"field": row["field"], "SOURCE": row["source"], final["arm"]: row["candidate"]} for row in final["fields"]]
        lines += diagnostic_table(diagnostics, ("SOURCE", final["arm"]))
        lines += sensor_table({final["arm"]: final["sensor_compatibility"]},
                              [row["field"] for row in final["fields"]], {final["arm"]: "Selected"})
        lines.append("")
        cost = data["selected_final_assessment_cost"]
        lines += [f"The later endpoint's generation lease occupied the GPU for {format_value(cost['generation_lease_occupied_gpu_seconds'])} seconds; its final CPU scoring invocation took {format_value(cost['cpu_score_wall_seconds'])} wall seconds and {format_value(cost['cpu_score_process_seconds'])} CPU seconds. The invocation reused prior completed CPU scores: {str(cost['cpu_score_reused_completed_scores']).lower()}.", ""]
    if tasks:
        lines += ["### Released task readouts", "",
            "These cached event and regional task scores were released only after the sealed epoch-100 nomination and were not used in that decision. Paired differences use the frozen TRAIN task normalizers; they are not percentages of VALIDATION SOURCE scores.", "",
            "| Endpoint | Epoch | Event score change [95% CI] | Regional score change [95% CI] |",
            "| --- | ---: | ---: | ---: |"]
        for entry in tasks:
            paired = entry["paired_frozen_train_units"]
            lines.append(f"| {entry['arm']} | {entry['epoch']} | {interval_text(paired['event'])} | {interval_text(paired['regional'])} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--base-training", type=Path, required=True)
    parser.add_argument("--base-assessment", type=Path, required=True)
    parser.add_argument("--spatial-training", type=Path, required=True)
    parser.add_argument("--spatial-assessment", type=Path, required=True)
    parser.add_argument("--nomination", type=Path)
    parser.add_argument("--final-training", type=Path)
    parser.add_argument("--final-assessment", type=Path)
    parser.add_argument("--task-result", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); started, cpu = time.perf_counter(), time.process_time()
    if bool(args.final_training) != bool(args.final_assessment) or (args.final_training or args.task_result) and not args.nomination:
        raise ValueError("a selected final pair or released tasks require the sealed nomination")
    campaign = args.campaign.resolve(strict=True); output = args.output.resolve()
    if (campaign / "assessment").resolve() not in output.parents:
        raise ValueError("output must be a new directory under this campaign's assessment tree")
    base = load_endpoint(args.base_training, args.base_assessment, expected_epoch=100)
    spatial = load_endpoint(args.spatial_training, args.spatial_assessment, expected_epoch=100)
    data = assemble(base, spatial)
    nom_path, nomination = nomination_check(args.nomination, base, spatial) if args.nomination else (None, None)
    final = None
    if args.final_training:
        final = load_endpoint(args.final_training, args.final_assessment)
        if (final["arm"] != nomination["selected_arm"] or not 100 < final["epoch"] <= nomination["continue_to_epoch"]):
            raise ValueError("later endpoint is not the nominated continuation")
        data["selected_final"] = final_summary(final, base)
    endpoints = {(x["arm"], x["epoch"]): x for x in (base, spatial) + ((final,) if final else ())}
    tasks = []
    for path in args.task_result:
        task_header = load(path); key = (task_header.get("arm"), task_header.get("epoch"))
        if key not in endpoints or any((t["arm"], t["epoch"]) == key for t in tasks):
            raise ValueError("task result must name one unique included endpoint")
        tasks.append(released_tasks(path, endpoints[key], nom_path, nomination))
    fit = load(checked_ref(base["position"]["fit"]))
    data["observer_identity"] = observer_identity(base, fit)
    train_panel = load(checked_ref(fit["inputs"]["panel"]))
    if (train_panel.get("split") != "train" or fit.get("sample_ids") != train_panel.get("sample_ids")
            or len(fit.get("sample_ids", [])) != 800):
        raise ValueError("position fit must identify the matched TRAIN800 panel and its population")
    population = train_panel.get("population")
    train_base = training_evidence(base, campaign, population)
    train_spatial = training_evidence(spatial, campaign, population)
    data["training_coupling_mismatch_counts"] = coupling_comparison(train_base, train_spatial)
    evidence = {BASE: train_base, SPATIAL: train_spatial}
    if final:
        original = base if final["arm"] == BASE else spatial
        previous = train_base if final["arm"] == BASE else train_spatial
        evidence[final["arm"]] = selected_training_evidence(original, previous, final, campaign, population)
    data["training"] = {BASE: train_base["summary"], SPATIAL: train_spatial["summary"]}
    data["assessment_costs"] = {BASE: assessment_evidence(base, campaign), SPATIAL: assessment_evidence(spatial, campaign)}
    if final:
        data["selected_final_training"] = evidence[final["arm"]]["summary"]
        data["selected_final_assessment_cost"] = assessment_evidence(final, campaign)
    data["released_tasks"] = tasks
    data["scope"] = {"inference_added": False, "persistence_or_observer_calls_added": False, "test_locked": True,
        "selection": "no feature, controller, checkpoint, continuation or task selection performed",
        "unit": "matched development frames from one trajectory; not independent confirmation"}
    data["inputs"] = {"base_training": ref(args.base_training), "base_assessment": ref(args.base_assessment),
        "spatial_training": ref(args.spatial_training), "spatial_assessment": ref(args.spatial_assessment),
        "nomination": ref(nom_path) if nom_path else None,
        "final_training": ref(args.final_training) if args.final_training else None,
        "final_assessment": ref(args.final_assessment) if args.final_assessment else None,
        "script": ref(Path(__file__))}
    output.mkdir(parents=True, exist_ok=False); figures = output / "figures"; figures.mkdir()
    source_files = ("analysis/car_round8_learning_summary.py", "analysis/car_round8_cached_summary.py",
        "analysis/car_round8_child_relations.py", "analysis/report_car_round8.py", "analysis/report_car_round5.py",
        "posttraining/assess_car_round2_checkpoints.py", "posttraining/assess_car_round7.py", "posttraining/run_car_round8.py")
    data["source_snapshots"] = {}
    for name in source_files:
        source = ROOT / "scripts" / name; target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
        if ref(source)["sha256"] != ref(target)["sha256"]: raise ValueError("source snapshot copy differs")
        data["source_snapshots"][name] = ref(target)
    captions = make_figures(data, base, spatial, final, figures)
    data["training_recording_snapshots"] = capture_training(evidence, output, captions)
    (output / "section.md").write_text(make_section(data, data["training"], tasks, data.get("selected_final")))
    (output / "summary.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    (output / "figure_captions.json").write_text(json.dumps(captions, indent=2, allow_nan=False) + "\n")
    artifacts = [p for p in sorted(output.rglob('*')) if p.is_file()]
    index = {"schema": "car_round8_learning_summary_v1", "status": "completed", "inputs": data["inputs"],
        "files_sha256": {str(p.relative_to(output)): ref(p)["sha256"] for p in artifacts},
        "bootstrap": data["bootstrap"], "test_locked": True, "additional_inference": False,
        "figure_inspection": "generation only; inspect rendered figures before publication",
        "timing": {"wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu, "device": "CPU"}}
    (output / "evidence_index.json").write_text(json.dumps(index, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "completed", "output": str(output), "figures": len(captions), "timing": index["timing"]}))


if __name__ == "__main__":
    main()
