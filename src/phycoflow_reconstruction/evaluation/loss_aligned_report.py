"""Offline Round 6 comparison: JSON reductions in, CSV/JSON/Markdown out.

Run ``python -m phycoflow_reconstruction.evaluation.loss_aligned_report SPEC.json``.
No model, training, tensor or inference modules are imported.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

_COUNTS = {"train": 800, "validation": 100, "test": 100}


def _mean(values):
    return sum(values) / len(values)


_REQUIRED = {"scale_rho", "shape_mutual", "scale_self", "shape_self", "state", "scale", "shape"}
_MODELS = ("SOURCE", "U6", "A6")


def _assess(source, candidate, metrics, scales):
    changes = {
        name: (candidate["losses"][key] - source["losses"][key]) / scales[key]
        for name, key in metrics.items()
    }
    ratios = [c / s for c, s in zip(candidate["field_risk"], source["field_risk"])]
    return {
        "normalized_changes": changes,
        "field_ratios": ratios,
        "qualified": min(changes["scale_rho"], changes["shape_mutual"]) <= -0.02
        and all(x <= 0.02 for x in changes.values())
        and all(r <= 1.10 for r in ratios),
    }


def _choose(assessments, costs):
    for arm in ("U6", "A6"):
        cost = float(costs[arm]["update_seconds_per_step"])
        if not math.isfinite(cost) or cost <= 0:
            raise ValueError("positive measured update times are required")
    if not assessments["U6"]["qualified"]:
        return "A6" if assessments["A6"]["qualified"] else "SOURCE"
    if assessments["A6"]["qualified"]:
        a, u = (assessments[arm]["normalized_changes"] for arm in ("A6", "U6"))
        a_eff = max(-a["scale_rho"], -a["shape_mutual"]) / costs["A6"]["update_seconds_per_step"]
        u_eff = max(-u["scale_rho"], -u["shape_mutual"]) / costs["U6"]["update_seconds_per_step"]
        if (
            min(a[k] - u[k] for k in ("scale_rho", "shape_mutual")) <= -0.02
            and all(a[k] - u[k] <= 0.02 for k in _REQUIRED)
            and a_eff > u_eff
        ):
            return "A6"
    return "U6"


def freeze_validation_choice(
    validation_inputs, decision_metrics, scales, costs, *, output_path=None
):
    """Freeze the validation-only decision before opening TEST reductions.

    validation_inputs maps SOURCE/U6/A6 to saved reduction paths; scales map actual
    decision loss keys to frozen positive TRAIN/SOURCE calibration scales. The
    optional output_path records reproducible input hashes, assessed values and rule.
    """
    if set(validation_inputs) != set(_MODELS) or set(decision_metrics) != _REQUIRED:
        raise ValueError("all validation models and decision metrics are required")
    scales = {key: float(value) for key, value in scales.items()}
    if not set(decision_metrics.values()) <= set(scales) or any(
        not math.isfinite(x) or x <= 0 for x in scales.values()
    ):
        raise ValueError("positive frozen TRAIN/SOURCE calibration scales required")
    inputs, rows_by_model, hashes = {}, {}, {}
    for model in _MODELS:
        payload_bytes = Path(validation_inputs[model]).read_bytes()
        payload = json.loads(payload_bytes)
        rows = {row["sample_id"]: row for row in payload["rows"]}
        if (
            payload["split"] != "validation"
            or len(rows) != _COUNTS["validation"]
            or len(rows) != len(payload["rows"])
        ):
            raise ValueError("invalid validation coverage")
        inputs[model], rows_by_model[model] = payload, rows
        hashes[model] = hashlib.sha256(payload_bytes).hexdigest()
    ref, ref_rows = inputs["SOURCE"], rows_by_model["SOURCE"]
    for model in _MODELS[1:]:
        if (
            inputs[model]["manifest_sha256"] != ref["manifest_sha256"]
            or rows_by_model[model].keys() != ref_rows.keys()
        ):
            raise ValueError("unmatched validation conditions")
        if any(
            rows_by_model[model][i]["query_selection"] != ref_rows[i]["query_selection"]
            for i in ref_rows
        ):
            raise ValueError("unmatched validation query selections")
    summary = {}
    for model in _MODELS:
        rows = rows_by_model[model]
        keys = set(decision_metrics.values())
        if any(not keys <= set(row["losses"]) for row in rows.values()):
            raise ValueError("missing validation decision loss")
        count = len(next(iter(rows.values()))["field_error_sums"])
        risks = [
            sum(row["field_error_sums"][j] for row in rows.values())
            / sum(row["field_error_counts"][j] for row in rows.values())
            for j in range(count)
        ]
        if not risks or any(not math.isfinite(x) or x < 0 for x in risks):
            raise ValueError("invalid validation field risks")
        summary[model] = {
            "losses": {key: _mean([row["losses"][key] for row in rows.values()]) for key in keys},
            "field_risk": risks,
        }
    if any(x <= 0 for x in summary["SOURCE"]["field_risk"]) or any(
        len(summary[model]["field_risk"]) != len(summary["SOURCE"]["field_risk"])
        for model in _MODELS
    ):
        raise ValueError("SOURCE risks must be positive and field counts matched")
    assessments = {
        model: _assess(summary["SOURCE"], summary[model], decision_metrics, scales)
        for model in ("U6", "A6")
    }
    receipt = {
        "version": "round6_validation_decision_v1",
        "validation_choice": _choose(assessments, costs),
        "validation_manifest_sha256": ref["manifest_sha256"],
        "input_sha256": hashes,
        "weight_identities": {
            model: {
                "kind": inputs[model]["weight_identity_kind"],
                "sha256": inputs[model]["weight_sha256"],
            }
            for model in _MODELS
        },
        "decision_metrics": dict(decision_metrics),
        "scales": scales,
        "update_seconds_per_step": {
            model: float(costs[model]["update_seconds_per_step"]) for model in ("U6", "A6")
        },
        "assessments": assessments,
    }
    receipt["sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if output_path is not None:
        Path(output_path).write_text(
            json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
    return receipt


def build_report(spec_path):
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text())
    base = spec_path.parent
    splits, models = ("train", "validation", "test"), ("SOURCE", "U6", "A6")
    metrics = spec["decision_metrics"]
    required = _REQUIRED
    if set(metrics) != required or not set(metrics.values()) <= set(spec["scales"]):
        raise ValueError("decision metrics and frozen positive TRAIN/SOURCE scales are required")
    scales = {k: float(v) for k, v in spec["scales"].items()}
    if any(not math.isfinite(v) or v <= 0 for v in scales.values()):
        raise ValueError("calibration scales must be positive and finite")
    data = {}
    for split in splits:
        for model in models:
            path = base / spec["inputs"][split][model]
            payload = json.loads(path.read_text())
            if payload["split"] != split or len(payload["rows"]) != _COUNTS[split]:
                raise ValueError("split coverage does not match manifest")
            rows = {r["sample_id"]: r for r in payload["rows"]}
            if len(rows) != len(payload["rows"]):
                raise ValueError("duplicate condition")
            data[split, model] = payload, rows
        reference, ref_rows = data[split, "SOURCE"]
        for model in models[1:]:
            payload, rows = data[split, model]
            if (
                payload["manifest_sha256"] != reference["manifest_sha256"]
                or rows.keys() != ref_rows.keys()
            ):
                raise ValueError("unmatched comparison identities")
            for sample_id in rows:
                if rows[sample_id]["query_selection"] != ref_rows[sample_id]["query_selection"]:
                    raise ValueError("unmatched query identities")
    for model in models:
        identities = {
            (data[split, model][0]["weight_identity_kind"], data[split, model][0]["weight_sha256"])
            for split in splits
        }
        if len(identities) != 1:
            raise ValueError("model weight identity differs between splits")
    first_row = next(iter(data["train", "SOURCE"][1].values()))
    field_count = len(first_row["field_error_sums"])
    bank = first_row.get("query_selection", {}).get("bank_manifest") or {}
    field_names = (
        spec.get("field_names")
        or bank.get("field_ids")
        or [f"field_{i}" for i in range(field_count)]
    )
    if len(field_names) != field_count or len(set(field_names)) != field_count:
        raise ValueError("field names must match the recorded field order")
    table, summary = [], {}
    for split in splits:
        summary[split] = {}
        ref_rows = data[split, "SOURCE"][1]
        common = set.intersection(
            *(set(row["losses"]) for model in models for row in data[split, model][1].values())
        )
        if not set(metrics.values()) <= common:
            raise ValueError("required decision loss is missing")
        for model in models:
            rows = data[split, model][1]
            results = {key: _mean([rows[i]["losses"][key] for i in ref_rows]) for key in common}
            nfields = len(next(iter(rows.values()))["field_error_sums"])
            risks = [
                sum(r["field_error_sums"][j] for r in rows.values())
                / sum(r["field_error_counts"][j] for r in rows.values())
                for j in range(nfields)
            ]
            summary[split][model] = {"losses": results, "field_risk": risks}
        src, uniform = summary[split]["SOURCE"], summary[split]["U6"]
        for model in models:
            current = summary[split][model]
            for key in sorted(common):
                value = current["losses"][key]
                delta = value - src["losses"][key]
                table.append(
                    {
                        "split": split,
                        "model": model,
                        "metric": key,
                        "value": value,
                        "source_delta": delta,
                        "source_delta_scaled": delta / scales[key] if key in scales else "",
                        "a_vs_u": value - uniform["losses"][key] if model == "A6" else "",
                    }
                )
            for j, risk in enumerate(current["field_risk"]):
                source = src["field_risk"][j]
                if source <= 0:
                    raise ValueError("SOURCE fidelity risk must be positive")
                table.append(
                    {
                        "split": split,
                        "model": model,
                        "metric": f"field_mse.{field_names[j]}",
                        "value": risk,
                        "source_delta": risk - source,
                        "source_delta_scaled": "",
                        "a_vs_u": risk - uniform["field_risk"][j] if model == "A6" else "",
                    }
                )

    assessments = {
        split: {
            model: _assess(summary[split]["SOURCE"], summary[split][model], metrics, scales)
            for model in ("U6", "A6")
        }
        for split in splits
    }
    costs = spec["costs"]
    for arm in ("U6", "A6"):
        if (
            not math.isfinite(float(costs[arm]["total_wall_seconds"]))
            or float(costs[arm]["total_wall_seconds"]) <= 0
        ):
            raise ValueError("positive measured total wall times are required")
    pick = _choose(assessments["validation"], costs)
    if spec.get("validation_receipt"):
        frozen = json.loads((base / spec["validation_receipt"]).read_text())
        actual = freeze_validation_choice(
            {m: base / spec["inputs"]["validation"][m] for m in models}, metrics, scales, costs
        )
        if frozen != actual:
            raise ValueError("frozen validation decision receipt mismatch")
        pick = frozen["validation_choice"]
    gaps = {
        model: {
            key: summary["test"][model]["losses"][key] - summary["train"][model]["losses"][key]
            for key in metrics.values()
        }
        for model in models
    }
    coverage = {}
    for split in splits:
        source_rows = data[split, "SOURCE"][1]
        queries = [q for row in source_rows.values() for q in row["query_selection"]["queries"]]
        coverage[split] = {
            "conditions": len(source_rows),
            "spatial_points_per_condition": 4096,
            "conditional_draws": 2,
            "self_draws": sum(q["multiplicity"] for q in queries if q["kind"] == "self"),
            "mutual_draws": sum(q["multiplicity"] for q in queries if q["kind"] == "mutual"),
            "unique_relation_queries": len(
                {json.dumps(q["query_id"], sort_keys=True) for q in queries}
            ),
            "exposed_self_fields": sorted(
                {q["fields"][0] for q in queries if q["kind"] == "self" and q.get("fields")}
            ),
            "exposed_mutual_pairs": sorted(
                {tuple(q["fields"]) for q in queries if q["kind"] == "mutual" and q.get("fields")}
            ),
            "unrepresented_pairs": bank.get("unrepresented_pair_count"),
        }
    result = {
        "train_test_gaps": gaps,
        "coverage": coverage,
        "summary": summary,
        "assessments": assessments,
        "validation_choice": pick,
        "test_confirmation": None if pick == "SOURCE" else assessments["test"][pick]["qualified"],
        "operational_choice": pick
        if pick != "SOURCE" and assessments["test"][pick]["qualified"]
        else "SOURCE",
        "costs": spec["costs"],
        "spending": spec.get("spending"),
        "field_names": field_names,
        "input_provenance": {
            split: {
                model: {
                    "path": spec["inputs"][split][model],
                    "manifest_sha256": data[split, model][0]["manifest_sha256"],
                    "weight_identity_kind": data[split, model][0]["weight_identity_kind"],
                    "weight_sha256": data[split, model][0]["weight_sha256"],
                    "provenance": data[split, model][0].get("provenance"),
                }
                for model in models
            }
            for split in splits
        },
        "metadata": spec.get("metadata"),
        "decision_metrics": metrics,
    }
    out = base / spec.get("output_dir", "postprocess")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    with (out / "loss_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "model",
                "metric",
                "value",
                "source_delta",
                "source_delta_scaled",
                "a_vs_u",
            ],
        )
        writer.writeheader()
        writer.writerows(table)
    lines = [
        "# Round 6 loss-aligned comparison",
        "",
        "Sampled-population results: TRAIN 800/8,000, VALIDATION 100/1,000 and TEST 100/1,000 matched conditions. Shape also samples the declared relation population.",
        f"Validation choice: **{pick}**; fixed TEST confirmation: {result['test_confirmation']}; operational model: **{result['operational_choice']}**.",
        "Loss changes are paired means divided by frozen positive TRAIN/SOURCE scales; negative values are gains and positive values are losses.",
        "",
        "| Split | Loss quantity | U6 − SOURCE | A6 − SOURCE | A6 − U6 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for split in splits:
        for name in sorted(metrics):
            u = assessments[split]["U6"]["normalized_changes"][name]
            a = assessments[split]["A6"]["normalized_changes"][name]
            lines.append(f"| {split} | {name} | {u:.4g} | {a:.4g} | {a - u:.4g} |")
    lines += [
        "",
        "TRAIN-to-TEST gaps below are TEST minus TRAIN in the same normalized loss units.",
        "",
        "| Loss quantity | SOURCE gap | U6 gap | A6 gap |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, key in sorted(metrics.items()):
        lines.append(
            f"| {name} | {gaps['SOURCE'][key] / scales[key]:.4g} | {gaps['U6'][key] / scales[key]:.4g} | {gaps['A6'][key] / scales[key]:.4g} |"
        )
    lines += [
        "",
        "Field fidelity: pooled model-unit MSE divided by matched SOURCE, with an allowance of 1.10.",
        "",
        "| Split | Field | U6 / SOURCE | A6 / SOURCE |",
        "| --- | --- | ---: | ---: |",
    ]
    for split in splits:
        for index, name in enumerate(field_names):
            lines.append(
                f"| {split} | {name} | {assessments[split]['U6']['field_ratios'][index]:.4g} | {assessments[split]['A6']['field_ratios'][index]:.4g} |"
            )
    lines += [
        "",
        "Field TRAIN-to-TEST MSE gaps (TEST minus TRAIN, model units squared).",
        "",
        "| Field | SOURCE gap | U6 gap | A6 gap |",
        "| --- | ---: | ---: | ---: |",
    ]
    for index, name in enumerate(field_names):
        vals = [
            summary["test"][model]["field_risk"][index]
            - summary["train"][model]["field_risk"][index]
            for model in models
        ]
        lines.append(f"| {name} | {vals[0]:.4g} | {vals[1]:.4g} | {vals[2]:.4g} |")
    lines += [
        "",
        "Coverage: "
        + "; ".join(
            f"{split} {v['conditions']} conditions, {v['spatial_points_per_condition']} points/condition, {v['conditional_draws']} draws, {v['self_draws']} self and {v['mutual_draws']} mutual Shape draws, {v['unique_relation_queries']} distinct Shape queries, {len(v['exposed_self_fields'])} fields and {len(v['exposed_mutual_pairs'])} pairs exposed"
            for split, v in coverage.items()
        )
        + ".",
        "",
        "Measured costs: "
        + "; ".join(
            f"{arm} {costs[arm]['update_seconds_per_step']:.4g} s/update and {costs[arm]['total_wall_seconds']:.4g} s total wall"
            for arm in ("U6", "A6")
        )
        + ".",
        "Actual spending (qualification and formal attempts): "
        + (
            json.dumps(spec["spending"], sort_keys=True)
            if spec.get("spending") is not None
            else "not supplied"
        )
        + ".",
        "",
        "A6 must improve a mutual score over U6 by 0.02 scale without guard regression and give better best-mutual gain per update second. A failed fixed TEST confirmation retains SOURCE without reranking.",
        "State/Scale use two conditional draws; Shape and fidelity use the primary endpoint. Q4096 measurements do not replace earlier full-native results. Complete component values and provenance are in the CSV and JSON files.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python -m phycoflow_reconstruction.evaluation.loss_aligned_report SPEC.json"
        )
    build_report(sys.argv[1])


if __name__ == "__main__":
    main()
