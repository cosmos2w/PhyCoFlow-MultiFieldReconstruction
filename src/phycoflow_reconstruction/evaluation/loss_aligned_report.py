"""Offline Round 6 comparison: JSON reductions in, CSV/JSON/Markdown out.

Run ``python -m phycoflow_reconstruction.evaluation.loss_aligned_report SPEC.json``.
No model, training, tensor or inference modules are imported.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

_COUNTS = {"train": 800, "validation": 100, "test": 100}


def _mean(values):
    return sum(values) / len(values)


def build_report(spec_path):
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text())
    base = spec_path.parent
    splits, models = ("train", "validation", "test"), ("SOURCE", "U6", "A6")
    metrics = spec["decision_metrics"]
    required = {"scale_rho", "shape_mutual", "scale_self", "shape_self", "state", "scale", "shape"}
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
                        "metric": f"field_mse.{j}",
                        "value": risk,
                        "source_delta": risk - source,
                        "source_delta_scaled": "",
                        "a_vs_u": risk - uniform["field_risk"][j] if model == "A6" else "",
                    }
                )

    def assess(split, model):
        source, candidate = summary[split]["SOURCE"], summary[split][model]
        changes = {
            name: (candidate["losses"][key] - source["losses"][key]) / scales[key]
            for name, key in metrics.items()
        }
        mutual = (changes["scale_rho"], changes["shape_mutual"])
        guards = all(x <= 0.02 for x in changes.values())
        fields = all(c <= 1.10 * s for c, s in zip(candidate["field_risk"], source["field_risk"]))
        return {
            "normalized_changes": changes,
            "field_ratios": [c / s for c, s in zip(candidate["field_risk"], source["field_risk"])],
            "qualified": min(mutual) <= -0.02 and guards and fields,
        }

    assessments = {
        split: {model: assess(split, model) for model in ("U6", "A6")} for split in splits
    }
    val = assessments["validation"]
    costs = spec["costs"]
    for arm in ("U6", "A6"):
        if (
            float(costs[arm]["update_seconds_per_step"]) <= 0
            or float(costs[arm]["total_wall_seconds"]) <= 0
        ):
            raise ValueError("positive measured update and total wall times are required")
    if val["U6"]["qualified"]:
        pick = "U6"
        if val["A6"]["qualified"]:
            a, u = val["A6"]["normalized_changes"], val["U6"]["normalized_changes"]
            # Same engineering margin defines a material held-out difference;
            # actual cost is disclosed rather than folded into a fictitious score.
            a_eff = (
                max(-a["scale_rho"], -a["shape_mutual"]) / costs["A6"]["update_seconds_per_step"]
            )
            u_eff = (
                max(-u["scale_rho"], -u["shape_mutual"]) / costs["U6"]["update_seconds_per_step"]
            )
            if (
                min(a[k] - u[k] for k in ("scale_rho", "shape_mutual")) <= -0.02
                and all(a[k] - u[k] <= 0.02 for k in required)
                and a_eff > u_eff
            ):
                pick = "A6"
    else:
        pick = "A6" if val["A6"]["qualified"] else "SOURCE"
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
        f"Validation choice: **{pick}**.",
        f"Fixed TEST confirmation: {result['test_confirmation']}.",
        f"Operational model after fixed TEST confirmation: **{result['operational_choice']}**.",
        "",
        "Changes below are paired condition-mean changes divided by frozen positive TRAIN/SOURCE calibration scales. Negative values are improvements. Fidelity uses pooled model-unit squared errors and the 1.10 SOURCE bound.",
        "",
        "| Split | Arm | Qualified | Scale rho | Shape mutual | Worst field ratio |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for split in splits:
        for model in ("U6", "A6"):
            item = assessments[split][model]
            lines.append(
                f"| {split} | {model} | {item['qualified']} | {item['normalized_changes']['scale_rho']:.4g} | {item['normalized_changes']['shape_mutual']:.4g} | {max(item['field_ratios']):.4g} |"
            )
    lines += [
        "",
        "The CSV contains all common loss components, SOURCE-relative changes and A6 versus U6 differences for each split; summary.json contains the complete guard results, train-to-TEST gaps, query coverage, split values and supplied costs.",
        "A6 must improve a mutual score over U6 by 0.02 scale without guard regression, and yield a larger best-mutual improvement per update second, to justify its overhead. If only A6 qualifies against SOURCE, it is selected.",
        "",
        f"Costs: `{json.dumps(spec['costs'], sort_keys=True)}`.",
        "",
        "Shape uses 4 self and 12 mutual sampled complete queries per condition. State/Scale use two conditional draws; Shape and fidelity use the primary endpoint. These Q4096 results do not replace earlier full-native measurements.",
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
