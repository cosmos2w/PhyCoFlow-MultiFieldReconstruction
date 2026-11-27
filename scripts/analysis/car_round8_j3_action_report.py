#!/usr/bin/env python3
"""Render one completed matched real-SOURCE J3 action receipt.

This reads existing measurements only. Main action phases, extra component
reverse passes, and post-action finite-difference replay are shown separately.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.analysis.car_round8_cached_summary import digest, ref
from scripts.analysis.report_car_round5 import save_figure


PACKAGES = ("PH_Only", "PH_Plus_Position")
LABELS = ("PH only", "PH + position")
COLORS = ("#3b78a0", "#c15e45")


def _number(value, name, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return float(value)


def _field_order(receipt, reference_path):
    fields = receipt.get("field_order")
    if reference_path is not None:
        reference_path = Path(reference_path)
        if digest(reference_path) != receipt.get("reference_sha256"):
            raise ValueError("field reference hash differs from the completed action")
        reference = json.loads(reference_path.read_text())
        from_reference = reference["reference"]["field_names"]
        if fields is not None and fields != from_reference:
            raise ValueError("field order differs from the action's reference")
        fields = from_reference
    if (not isinstance(fields, list) or not fields or len(set(fields)) != len(fields)
            or any(not isinstance(value, str) or not value for value in fields)):
        raise ValueError("field order is required in the receipt or a hash-matched reference")
    return fields


def launch_bindings(launch_path, receipt_path, receipt):
    """Bind the completed action to its frozen source and recorded inputs."""
    launch_path = Path(launch_path)
    launch = json.loads(launch_path.read_text())
    if Path(launch["result"]).resolve() != Path(receipt_path).resolve():
        raise ValueError("launch does not identify the completed receipt")
    root = Path(launch["code_root"])
    manifest = root / "source_manifest.json"
    if digest(manifest) != launch["source_manifest_sha256"]:
        raise ValueError("frozen source manifest differs from launch")
    script_name = "scripts/qualification/car_round8_j3_position_action.py"
    source = root / script_name
    if digest(source) != json.loads(manifest.read_text()).get(script_name):
        raise ValueError("frozen action source differs from its manifest")
    command = launch["command"]
    paths = {name: Path(command[command.index(flag) + 1]) for name, flag in
             (("config", "--config"), ("reference", "--reference"),
              ("position_artifact", "--position-artifact"))}
    for name, key in (("config", "config_sha256"), ("reference", "reference_sha256"),
                      ("position_artifact", "position_sha256")):
        if digest(paths[name]) != receipt[key]:
            raise ValueError(f"action {name} differs from the completed receipt")
    reference = json.loads(paths["reference"].read_text())
    panel, settings = reference["risk_reference_panel"], reference["settings"]
    if (settings["metric"] != receipt["metric"] or panel["members"] != settings["calibration_draws"]
            or panel["reference_law"] != "training_selected_complement_v1"
            or panel["sample_count"] != len(panel["sample_ids"])
            or not set(receipt["actions"][0]["sample_ids"]).issubset(panel["sample_ids"])):
        raise ValueError("member-MSE reference does not match the declared action law and TRAIN panel")
    inputs = {name: ref(path) for name, path in paths.items()}
    inputs.update({"launch": ref(launch_path), "source_manifest": ref(manifest),
                   "frozen_action_script": ref(source)})
    calibration = {"members": panel["members"], "train_conditions": panel["sample_count"],
                   "reference_law": panel["reference_law"]}
    return paths["reference"], inputs, calibration


def summarize(receipt, *, reference_path=None):
    if receipt.get("schema") != "car_round8_j3_position_action_v1" or receipt.get("status") != "completed":
        raise ValueError("completed Round 8 J3 action receipt required")
    if receipt.get("metric") != "physical_member_mse_model_units":
        raise ValueError("this report requires the declared physical member-MSE metric")
    fields = _field_order(receipt, reference_path)
    raw = receipt.get("actions", ())
    if len(raw) != 2 or {row.get("package") for row in raw} != set(PACKAGES):
        raise ValueError("one PH-only and one PH-plus-position action required")
    actions = {row["package"]: row for row in raw}
    baseline, spatial = (actions[name] for name in PACKAGES)
    for key in ("sample_ids", "query_sha256", "selected_sha256", "complement_sha256", "first_generator_sha256"):
        if not baseline.get(key) or baseline[key] != spatial.get(key):
            raise ValueError(f"matched SOURCE action identity differs: {key}")
    result = {}
    for name in PACKAGES:
        row = actions[name]
        if row.get("members") != 3:
            raise ValueError("three genuine members are required")
        risk = {}
        for key in ("field_risk_before", "field_risk_after", "field_risk_change"):
            value = row.get(key)
            if not isinstance(value, list) or len(value) != len(fields):
                raise ValueError(f"{key} must match the field order")
            risk[key] = np.asarray([_number(x, key) for x in value], dtype=np.float64)
        if (risk["field_risk_before"] < 0).any() or (risk["field_risk_after"] < 0).any():
            raise ValueError("member-MSE risks cannot be negative")
        if not np.allclose(risk["field_risk_after"] - risk["field_risk_before"],
                           risk["field_risk_change"], rtol=1e-6, atol=1e-12):
            raise ValueError("saved field-risk change does not match before and after")
        multiplier = row.get("multiplier")
        if not isinstance(multiplier, list) or len(multiplier) != len(fields):
            raise ValueError("the field multiplier must match the field order")
        multiplier = [_number(value, "field multiplier", nonnegative=True) for value in multiplier]
        if not any(multiplier):
            raise ValueError("a nonzero field multiplier is required")
        parts = row["shape_component_row_norms"]
        if "ph" not in parts or (name == "PH_Plus_Position" and "position" not in parts):
            raise ValueError("the applicable PH and location component rows are required")
        norms = {"shape": _number(row["family_row_norms"]["topology"], "shape row", nonnegative=True),
                 "ph": _number(parts["ph"], "PH row", nonnegative=True),
                 "position": None if "position" not in parts else _number(parts["position"], "position row", nonnegative=True),
                 "field": _number(row["weighted_field_row_norm"], "field row", nonnegative=True)}
        if norms["field"] <= 0:
            raise ValueError("an active weighted field row is required")
        clip_input = _number(row["clip_input_norm"], "clip input norm", nonnegative=True)
        clip_limit = _number(row["clip_limit"], "clip limit", nonnegative=True)
        timing_keys = ("forward_seconds", "parameter_rows_seconds", "optimizer_seconds",
                       "component_diagnostic_seconds", "finite_difference_seconds")
        times = {key: _number(row[key], key, nonnegative=True) for key in timing_keys}
        action_seconds = sum(times[key] for key in timing_keys[:3])
        peak = row.get("peak_allocated_bytes")
        increment = row.get("peak_increment_bytes")
        diagnostic_peak = row.get("component_diagnostic_peak_allocated_bytes")
        if (peak is None) != (increment is None):
            raise ValueError("both GPU peak and increment must be present or unavailable")
        if peak is not None:
            peak = _number(peak, "peak allocated bytes", nonnegative=True)
            increment = _number(increment, "peak increment bytes", nonnegative=True)
            if increment > peak:
                raise ValueError("peak increment exceeds allocated peak")
        if diagnostic_peak is not None:
            diagnostic_peak = _number(diagnostic_peak, "component diagnostic peak bytes", nonnegative=True)
        percentages = [100 * float(change) / float(before) if before > 0 else None
                       for before, change in zip(risk["field_risk_before"], risk["field_risk_change"])]
        result[name] = {"row_norms": norms, "clip_input_norm": clip_input, "clip_limit": clip_limit,
                        "clip_factor": min(1., clip_limit / (clip_input + 1e-6)),
                        "adam_displacement_norm": _number(row["adam_displacement_norm"], "Adam displacement", nonnegative=True),
                        "field_risk_before": risk["field_risk_before"].tolist(),
                        "field_risk_after": risk["field_risk_after"].tolist(),
                        "field_risk_change": risk["field_risk_change"].tolist(),
                        "field_risk_change_percent": percentages, "field_multiplier": multiplier, "times": times,
                        "main_action_seconds": action_seconds,
                        "peak_allocated_bytes": peak, "peak_increment_bytes": increment,
                        "component_diagnostic_peak_allocated_bytes": diagnostic_peak}
    if not np.allclose(result[PACKAGES[0]]["field_risk_before"], result[PACKAGES[1]]["field_risk_before"],
                       rtol=1e-6, atol=1e-8):
        raise ValueError("matched initial field risks differ")
    if result[PACKAGES[0]]["field_multiplier"] != result[PACKAGES[1]]["field_multiplier"]:
        raise ValueError("matched field multipliers differ")
    return {"field_order": fields, "actions": result, "sample_ids": baseline["sample_ids"],
            "metric": receipt.get("metric"), "device": receipt.get("device"),
            "device_name": receipt.get("device_name"), "scope": receipt.get("scope"),
            "learning_rate": _number(receipt["learning_rate"], "learning rate", nonnegative=True),
            "scale_objective": receipt.get("scale_objective"), "scale_joint_kappa": receipt.get("scale_joint_kappa"),
            "pi_state": receipt.get("pi_state"), "adam_state": receipt.get("adam_state")}


def figures(summary, directory):
    actions, fields = summary["actions"], summary["field_order"]
    captions = {}
    fig, ax = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)
    labels = ("Complete Shape family", "PH contribution", "Position contribution", "Weighted field row")
    keys = ("shape", "ph", "position", "field")
    all_positive = [value for package in PACKAGES for value in actions[package]["row_norms"].values()
                    if value is not None and value > 0]
    if all_positive:
        ax.set_xscale("log")
        ax.set_xlim(min(all_positive) / 2, max(all_positive) * 3)
    for index, (package, color) in enumerate(zip(PACKAGES, COLORS, strict=True)):
        values = actions[package]["row_norms"]
        for y, key in enumerate(keys):
            value = values[key]
            if value is not None and value > 0:
                ordinate = y + (-.15 if index == 0 else .15)
                ax.scatter(value, ordinate, color=color, s=55,
                           label=LABELS[index] if y == 0 else None, zorder=3)
                ax.annotate(f"{value:.3g}", (value, ordinate), xytext=(5, 0),
                            textcoords="offset points", va="center", color=color, fontsize=9)
            elif value == 0 and all_positive:
                ordinate = y + (-.15 if index == 0 else .15)
                edge = min(all_positive) / 2
                ax.scatter(edge, ordinate, color=color, marker="x", s=55, zorder=3)
                ax.annotate("0", (edge, ordinate), xytext=(5, 0),
                            textcoords="offset points", va="center", color=color, fontsize=9)
    ax.set_yticks(range(len(keys)), labels)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.25)
    ax.set_xlabel("Parameter-gradient row L2 norm (log scale)")
    ax.legend(frameon=False, loc="best")
    name = "j3_action_rows"
    save_figure(fig, directory, name)
    captions[name] = ("Parameter-gradient norms from one matched J3 SOURCE/TRAIN shadow action. "
        "PH and position are weighted contributions to the Shape row, while the field row uses the declared nonzero multiplier; the PH-only action has no position contribution.")

    fig, ax = plt.subplots(figsize=(max(8.5, len(fields) * 1.4), 4.6), constrained_layout=True)
    indices = np.arange(len(fields))
    for index, (package, color) in enumerate(zip(PACKAGES, COLORS, strict=True)):
        values = actions[package]["field_risk_change_percent"]
        shown = [np.nan if value is None else value for value in values]
        ax.bar(indices + (-.18 if index == 0 else .18), shown, width=.36,
               color=color, label=LABELS[index])
    ax.axhline(0, color="#333333", lw=.8)
    ax.set_xticks(indices, fields)
    ax.set_ylabel("Change from each field's starting member MSE (%)")
    ax.legend(frameon=False)
    name = "j3_field_risk_changes"
    save_figure(fig, directory, name)
    captions[name] = ("Finite selected-plus-complement field member-MSE changes after the two same-state actions, relative to each field's own starting value. "
        "A zero starting value has no percentage; absolute changes are reported in the table. One action does not establish learning or field retention.")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    labels = ("Main action\nphases", "Extra component\nreverse", "Post-action\nfield replay")
    positions = np.arange(3)
    for index, (package, color) in enumerate(zip(PACKAGES, COLORS, strict=True)):
        action = actions[package]
        times = action["times"]
        values = (action["main_action_seconds"], times["component_diagnostic_seconds"],
                  times["finite_difference_seconds"])
        axes[0].bar(positions + (-.18 if index == 0 else .18), values, width=.36,
                    color=color, label=LABELS[index])
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("Wall seconds")
    if summary.get("first_action_cold_start"):
        axes[0].set_title("Sequential actions; PH only includes cold start", fontsize=10)
    axes[0].legend(frameon=False)
    memory = ("Action peak\nallocated", "Action increment\nfrom start", "Component reverse\npeak allocated")
    positions = np.arange(3)
    have_memory = all(actions[package]["peak_allocated_bytes"] is not None for package in PACKAGES)
    if have_memory:
        for index, (package, color) in enumerate(zip(PACKAGES, COLORS, strict=True)):
            action = actions[package]
            diagnostic = action["component_diagnostic_peak_allocated_bytes"]
            values = [action["peak_allocated_bytes"] / 2**30, action["peak_increment_bytes"] / 2**30,
                      np.nan if diagnostic is None else diagnostic / 2**30]
            axes[1].bar(positions + (-.18 if index == 0 else .18), values, width=.36, color=color)
        axes[1].set_xticks(positions, memory)
        axes[1].set_ylabel("GPU allocated memory (GiB)")
    else:
        axes[1].text(.5, .5, "GPU allocation unavailable", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_axis_off()
    name = "j3_action_cost"
    save_figure(fig, directory, name)
    captions[name] = ("Main action phases sum the recorded forward, parameter-row reverse, and optimizer phases; extra component-gradient reverse and finite-difference replay are separate. "
        "Memory is PyTorch allocated memory for this small sequential diagnostic, with the extra component-reverse peak shown separately; it is not a full-epoch benchmark or evidence of a speedup.")
    if summary.get("first_action_cold_start"):
        captions[name] += " The first PH-only action included cold-start overhead."
    return captions


def section(summary):
    actions, fields = summary["actions"], summary["field_order"]
    lines = ["## Matched real-SOURCE J3 actions", "",
        f"One matched batch of {len(summary['sample_ids'])} TRAIN conditions used three independent SOURCE members per condition, factored parameter gradients with chunk size one, and the saved {summary['scale_objective']} Scale objective. Both shadow actions began from identical SOURCE, sampler, RNG and fresh AdamW states at learning rate {summary['learning_rate']:.6g}, with the same declared nonzero field multipliers; the PI state was not advanced. Field risk is average member MSE in each field's model units squared.", "",
        "| Package | Shape row norm | PH row norm | Position row norm | Weighted field row norm | Clip input / limit / factor | Adam displacement norm |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    calibration = summary.get("reference_calibration")
    if calibration:
        lines[4:4] = [f"The saved SOURCE member-MSE reference was calibrated on {calibration['train_conditions']} TRAIN conditions, including these action conditions, with J{calibration['members']} under the selected-plus-complement field law. The J3 action reuses the same population member-MSE estimand; its different member count changes Monte Carlo precision, and this action does not fit a new SOURCE reference.", ""]
    for package, label in zip(PACKAGES, LABELS, strict=True):
        item = actions[package]
        norms = item["row_norms"]
        position = "not enabled" if norms["position"] is None else f"{norms['position']:.6g}"
        lines.append(f"| {label} | {norms['shape']:.6g} | {norms['ph']:.6g} | {position} | {norms['field']:.6g} | {item['clip_input_norm']:.6g} / {item['clip_limit']:.6g} / {item['clip_factor']:.6g} | {item['adam_displacement_norm']:.6g} |")
    lines += ["", "The clipping factor is reconstructed from the recorded pre-clip norm and limit using the implemented global-norm clipping rule. With position enabled, the Shape vector row is the sum of the PH and position contribution vectors. Their individual weighted norms need not sum to the complete Shape row norm because the gradient directions can differ.", "",
        "| Field | Multiplier | Starting member MSE | PH-only change | PH + position change |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for index, field in enumerate(fields):
        name = field.replace("|", "\\|")
        baseline, spatial = (actions[package] for package in PACKAGES)
        lines.append(f"| {name} | {baseline['field_multiplier'][index]:.6g} | {baseline['field_risk_before'][index]:.6g} | {baseline['field_risk_change'][index]:+.6g} | {spatial['field_risk_change'][index]:+.6g} |")
    decreases = [sum(value < 0 for value in actions[package]["field_risk_change"]) for package in PACKAGES]
    lines += ["", f"Field changes are finite post-action differences on the identical selected-plus-complement query law, in each field's model units squared. Member MSE decreased in {decreases[0]}/{len(fields)} fields for PH only and {decreases[1]}/{len(fields)} for PH + position in this diagnostic action; these changes do not establish training retention.", "",
        "| Package | Main action phases (s) | Extra component reverse (s) | Field replay (s) | Action peak / increment (GiB) | Component reverse peak (GiB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for package, label in zip(PACKAGES, LABELS, strict=True):
        item = actions[package]
        memory = ("unavailable" if item["peak_allocated_bytes"] is None else
                  f"{item['peak_allocated_bytes'] / 2**30:.4g} / {item['peak_increment_bytes'] / 2**30:.4g}")
        diagnostic = item["component_diagnostic_peak_allocated_bytes"]
        diagnostic_memory = "unavailable" if diagnostic is None else f"{diagnostic / 2**30:.4g}"
        lines.append(f"| {label} | {item['main_action_seconds']:.4g} | {item['times']['component_diagnostic_seconds']:.4g} | {item['times']['finite_difference_seconds']:.4g} | {memory} | {diagnostic_memory} |")
    cold_start = (" The first PH-only action included cold-start overhead."
                  if summary.get("first_action_cold_start") else "")
    lines += ["", "The main phase total is the sum of recorded forward, parameter-row reverse and optimizer timings. Additional component reverse and post-action field replay are reported separately." + cold_start + " These sequential, instrumented small-batch actions establish execution behavior; they do not measure an epoch, model learning or a training speedup.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--reference", type=Path,
                        help="hash-matched field reference if an older receipt lacks field_order")
    parser.add_argument("--launch", type=Path,
                        help="completed launch record binding frozen source, config and reference")
    parser.add_argument("--first-action-cold-start", action="store_true",
                        help="record observed cold-start overhead on the first PH-only action")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    inputs = {"receipt": ref(args.receipt), "script": ref(Path(__file__))}
    calibration = None
    reference_path = args.reference
    if args.launch is not None:
        bound_reference, bound_inputs, calibration = launch_bindings(args.launch, args.receipt, receipt)
        if reference_path is not None and digest(reference_path) != digest(bound_reference):
            raise ValueError("provided reference differs from the completed launch")
        reference_path = bound_reference
        inputs.update(bound_inputs)
    elif reference_path is not None:
        inputs["reference"] = ref(reference_path)
    result = summarize(receipt, reference_path=reference_path)
    result["reference_calibration"] = calibration
    result["first_action_cold_start"] = args.first_action_cold_start
    args.output.mkdir(parents=True, exist_ok=False)
    directory = args.output / "figures"
    directory.mkdir()
    captions = figures(result, directory)
    payload = {"schema": "car_round8_j3_action_summary_v1", "status": "completed",
               "summary": result, "inputs": inputs}
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    (args.output / "figure_captions.json").write_text(json.dumps(captions, indent=2, allow_nan=False) + "\n")
    (args.output / "section.md").write_text(section(result))


if __name__ == "__main__":
    main()
