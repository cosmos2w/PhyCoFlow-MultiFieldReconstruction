#!/usr/bin/env python3
"""Re-express sealed Round 7 records under the prospective Round 8 readout.

This program reads already-scored, matched conditions. It performs no model
inference, feature fitting, or access to a dataset outside the saved panel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ref(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": digest(path)}


def checked_ref(value: dict) -> Path:
    path = Path(value["path"]).resolve(strict=True)
    if digest(path) != value["sha256"]:
        raise ValueError(f"input hash mismatch: {path}")
    return path


def paired_interval(values, indices):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite condition values")
    estimates = np.concatenate([values[chunk].mean(axis=1) for chunk in np.array_split(indices, 32) if len(chunk)])
    return {"mean": values.mean(axis=0).tolist(),
            "interval_95": np.quantile(estimates, [0.025, 0.975], axis=0).tolist()}


def score_triplet(records, name, indices):
    arrays = np.asarray([[r[name][term] for term in ("attraction", "separation", "score")] for r in records], dtype=float)
    if not np.allclose(arrays[:, 0] - arrays[:, 1], arrays[:, 2], rtol=1e-8, atol=1e-10):
        raise ValueError(f"{name} does not reconstruct A - R")
    return arrays


def summarize(assessment, decomposition, tasks=None, *, bootstrap=2000, seed=6042026):
    if assessment.get("status") != "completed" or decomposition.get("status") != "completed":
        raise ValueError("completed cached records required")
    ids = assessment["sample_ids"]
    n, j = len(ids), assessment["members"]
    if (not ids or len(set(ids)) != n or j != 4 or decomposition["members"] != j
            or decomposition["sample_ids"] != ids or decomposition["split"] != assessment["split"]
            or [r["sample_id"] for r in decomposition["records"]] != ids
            or decomposition["field_order"] != assessment["field_order"]):
        raise ValueError("matched ordered J4 panel, fields and decomposition required")
    for side in ("SOURCE", "candidate"):
        if len(assessment["per_condition_fields"][side]) != n or len(assessment["per_condition_scores"][side]) != n:
            raise ValueError("missing condition rows")
    if bootstrap < 1:
        raise ValueError("positive bootstrap replicate count required")
    indices = np.random.default_rng(seed).integers(0, n, size=(bootstrap, n))
    fields = assessment["field_order"]
    by_side = {}
    for side in ("SOURCE", "candidate"):
        rows = assessment["per_condition_fields"][side]
        arrays = {key: np.asarray([r[key] for r in rows], dtype=np.float64) for key in
                  ("member_mse", "ensemble_mean_mse", "signed_mean_risk", "unbiased_member_variance")}
        if any(value.shape != (n, len(fields)) or not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("invalid ordered native field records")
        if not np.allclose(arrays["member_mse"], arrays["ensemble_mean_mse"] + (j-1)/j*arrays["unbiased_member_variance"], rtol=2e-5, atol=1e-9):
            raise ValueError("member/mean/spread identity failed")
        if not np.allclose(arrays["signed_mean_risk"], arrays["ensemble_mean_mse"] - arrays["unbiased_member_variance"]/j, rtol=2e-5, atol=1e-9):
            raise ValueError("signed/mean/spread identity failed")
        by_side[side] = arrays
    source, child = by_side["SOURCE"], by_side["candidate"]
    denominator = source["member_mse"].mean(0)
    if (denominator <= 0).any():
        raise ValueError("positive SOURCE pooled risk required for relative bound")
    ratio = child["member_mse"].mean(0) / denominator
    boot_ratio = np.concatenate([child["member_mse"][chunk].mean(1) / source["member_mse"][chunk].mean(1)
                                 for chunk in np.array_split(indices, 32) if len(chunk)])
    margin = child["member_mse"] - 1.1 * source["member_mse"]
    allowance = np.asarray(assessment["native_allowance_physical"], dtype=float)
    signed_delta = child["signed_mean_risk"] - source["signed_mean_risk"]
    if not np.allclose(signed_delta.mean(0), assessment["native_safeguards"]["field_difference"], rtol=1e-8, atol=1e-9):
        raise ValueError("historical signed field difference changed")
    field_table = []
    ratio_ci = np.quantile(boot_ratio, [0.025, 0.975], axis=0)
    margin_summary = paired_interval(margin, indices)
    signed_summary = paired_interval(signed_delta, indices)
    for i, field in enumerate(fields):
        row = {"field": field, "unit": assessment["field_units"][i],
               "source": {k: float(v[:, i].mean()) for k, v in source.items()},
               "candidate": {k: float(v[:, i].mean()) for k, v in child.items()},
               "member_ratio": float(ratio[i]), "member_ratio_interval_95": ratio_ci[:, i].tolist(),
               "member_margin": margin_summary["mean"][i],
               "member_margin_interval_95": [x[i] for x in margin_summary["interval_95"]],
               "member_point_pass": bool(ratio[i] <= 1.1),
               "old_signed_difference": signed_summary["mean"][i],
               "old_signed_difference_interval_95": [x[i] for x in signed_summary["interval_95"]],
               "old_signed_allowance": float(allowance[i]),
               "old_signed_point_pass": bool(signed_summary["mean"][i] <= allowance[i]),
               "source_condition_member_mse_quantiles_10_50_90": np.quantile(source["member_mse"][:, i], [.1, .5, .9]).tolist()}
        field_table.append(row)
    normalized = (source["member_mse"] / denominator).mean(axis=1)
    ordering = sorted(range(n), key=lambda i: (float(normalized[i]), ids[i]))
    chosen = {name: {"sample_id": ids[index], "source_mean_normalized_member_mse": float(normalized[index])}
              for name, index in (("representative_median", ordering[(n-1)//2]), ("difficult_maximum", ordering[-1]))}
    blocks = ("signed", "joint", "log_power_self", "rho", "third_order")
    prefit = decomposition["prefit_coefficients"]
    factors = {k: float(prefit["scale." + k]) * (.25 if k == "joint" else 1.0) for k in blocks}
    scale = {}
    for side in ("SOURCE", "candidate"):
        model = [r["models"][side] for r in decomposition["records"]]
        raw = {k: score_triplet([m["blocks"] for m in model], k, indices) for k in blocks}
        scale[side] = {k: raw[k].mean(axis=0).tolist() for k in blocks}
        expanded = sum(raw[k] * factors[k] for k in blocks)
        published = raw["signed"] + .25*raw["joint"] + raw["log_power_self"] + raw["rho"]
        for key, data in (("Complete_Expanded_Prefit", expanded), ("Published_Core_Default", published)):
            if not np.allclose(data[:, 2], [m["definitions"][key] for m in model], rtol=1e-8, atol=1e-10):
                raise ValueError(f"Scale {key} reconstruction failed")
            scale[side][key] = data.mean(axis=0).tolist()
    shape = {}
    for side in ("SOURCE", "candidate"):
        rows = assessment["per_condition_scores"][side]
        shape[side] = {}
        for kind in ("self", "mutual"):
            for category in ("finite", "essential"):
                a = np.asarray([r[f"shape.{kind}.{category}.attraction"] for r in rows])
                r = np.asarray([r[f"shape.{kind}.{category}.separation"] for r in rows])
                shape[side][f"{kind}.{category}"] = [float(a.mean()), float(r.mean()), float((a-r).mean())]
        total = sum(np.asarray(v) for v in shape[side].values())
        if not np.isclose(total[2], np.mean([r["shape"] for r in rows]), rtol=1e-6, atol=1e-8):
            raise ValueError("Shape component reconstruction failed")
        shape[side]["total"] = total.tolist()
    task_result = None
    if tasks is not None:
        if tasks.get("status") != "completed" or tasks.get("split") != assessment["split"] or tasks.get("sample_ids") != ids:
            raise ValueError("task panel differs")
        task_result = {"paired": tasks["paired"], "per_model": {}}
        for side in ("SOURCE", "candidate"):
            rows = tasks["rows"][side]
            if [r["sample_id"] for r in rows] != ids:
                raise ValueError("task rows differ from condition panel")
            task_result["per_model"][side] = {metric: float(np.mean([r[metric] for r in rows]))
                                                for metric in ("event_score", "regional_score")}
    return {"schema": "car_round8_cached_summary_v1", "split": assessment["split"], "sample_ids": ids,
            "members": j, "selection_use": assessment["selection_use"], "fields": field_table,
            "bootstrap": {"seed": seed, "replicates": bootstrap, "unit": "paired condition; frames share one trajectory"},
            "example_selection": {"rule": "SOURCE native member MSE per condition divided by its pooled field risk, averaged equally over fields; median and maximum; ties by sample ID", **chosen},
            "scale_terms_order": ["attraction", "positive_separation", "fair_score"], "scale": scale,
            "scale_weights": factors, "shape_terms_order": ["attraction", "positive_separation", "fair_score"], "shape": shape,
            "tasks": task_result,
            "interpretation": "Member and signed contracts are distinct; point passes do not certify a population bound. A is direct discrepancy, R is half mean pair distance, S=A-R. SOURCE/child use separately reconstructed fixed Q4096 for SSS; fields use native physical masses. Unknown units are dataset units.",
            "test_locked": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--decomposition", type=Path, required=True)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assessment = json.loads(args.assessment.read_text())
    decomposition = json.loads(args.decomposition.read_text())
    tasks = json.loads(args.tasks.read_text()) if args.tasks else None
    for value in (assessment["spec"], assessment["source_native"], assessment["child_native"],
                  assessment["source_selected"], assessment["child_selected"], decomposition["spec"],
                  decomposition["source_receipt"], decomposition["child_receipt"]):
        checked_ref(value)
    result = summarize(assessment, decomposition, tasks)
    result["inputs"] = {"assessment": ref(args.assessment), "decomposition": ref(args.decomposition),
                        "tasks": ref(args.tasks) if args.tasks else None, "script": ref(Path(__file__))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
