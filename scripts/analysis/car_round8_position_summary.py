#!/usr/bin/env python3
"""Summarize completed, cached positional evidence without inference or PH."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.analysis.car_round8_cached_summary import checked_ref, ref
from scripts.analysis.report_car_round5 import save_figure

MODELS = ("SOURCE", "JointAnchor100", "JointAnchor500")
OBSERVERS = ("ph_ARS", "location_unit_matched_ARS", "combined_ARS",
             "native_location_unit_matched_ARS")
BOOTSTRAP_SEED = 6042026


def summarize(readout):
    if (readout.get("status") != "completed" or readout.get("split") != "validation"
            or len(readout.get("sample_ids", ())) != 100 or readout.get("members") != 4
            or not readout.get("test_locked")):
        raise ValueError("the completed matched VALIDATION100 J4 readout is required")
    ids = readout["sample_ids"]
    if len(set(ids)) != len(ids) or [row.get("sample_id") for row in readout.get("records", ())] != ids:
        raise ValueError("position records must match the declared unique sample IDs in order")
    bootstrap = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 100, (2000, 100))
    output, arrays = {}, {}
    for observer in OBSERVERS:
        arrays[observer] = {}
        for model in MODELS:
            values = np.asarray([sum((np.asarray(row["models"][model][kind][observer])
                for kind in ("self", "mutual"))) for row in readout["records"]])
            if values.shape != (100, 3) or not np.isfinite(values).all():
                raise ValueError("finite matched A/R/S rows are required")
            np.testing.assert_allclose(values[:, 0] - values[:, 1], values[:, 2], rtol=2e-5, atol=2e-5)
            arrays[observer][model] = values
        output[observer] = {"means": {name: rows.mean(0).tolist()
            for name, rows in arrays[observer].items()}, "paired_source_deltas": {}}
        for model in MODELS[1:]:
            delta = arrays[observer][model] - arrays[observer]["SOURCE"]
            output[observer]["paired_source_deltas"][model] = {
                "mean": delta.mean(0).tolist(),
                "interval_95": np.quantile(delta[bootstrap].mean(1), [.025, .975], axis=0).T.tolist()}
    for model in MODELS:
        np.testing.assert_allclose(arrays["combined_ARS"][model], .5 * (
            arrays["ph_ARS"][model] + arrays["location_unit_matched_ARS"][model]), rtol=2e-5, atol=2e-5)
    return output


def position_conclusion(summary):
    """Describe the measured fair-score direction and each paired interval."""
    def interval_sentence(observer, label):
        row = summary[observer]["paired_source_deltas"]["JointAnchor500"]
        delta, (low, high) = row["mean"][2], row["interval_95"][2]
        if low > 0:
            finding = "is above zero"
        elif high < 0:
            finding = "is below zero"
        else:
            finding = "includes zero"
        return f"For {label}, the fair-score change is {delta:+.5g} and its paired 95% interval {finding} [{low:.5g}, {high:.5g}]."

    ph = summary["ph_ARS"]["paired_source_deltas"]["JointAnchor500"]["mean"][2]
    direction = "decreased" if ph < 0 else ("increased" if ph > 0 else "was unchanged")
    return (f"On the matched VALIDATION panel, the JointAnchor500 PH fair score {direction} by {abs(ph):.5g} relative to SOURCE. "
            + interval_sentence("location_unit_matched_ARS", "selected-support location") + " "
            + interval_sentence("native_location_unit_matched_ARS", "native-to-raster location") + " "
            + "The location feature also responds to excursion amplitude and extent; it does not isolate geometric misregistration. The half-sum changes PH weighting as well as adding position, so a learning comparison would attribute its complete package. Frame uncertainty is development evidence from one trajectory.")


def figures(summary, sensitivity, output):
    plt.rcParams.update({"font.size": 16})
    captions = {}
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10), constrained_layout=True)
    for ax, observer, title in zip(axes, OBSERVERS[:3],
            ("Existing PH", "Location / frozen TRAIN unit", "Half(PH + location / unit)"), strict=True):
        for offset, model, color in zip((-.24, 0, .24), MODELS,
                ("#397aa6", "#d19838", "#a04446"), strict=True):
            values = summary[observer]["means"][model]
            ax.bar(np.arange(3) + offset, values, .24, label=model, color=color)
        ax.set_xticks(range(3), ("Attraction A", "Separation R", "Fair score S"))
        ax.set_ylabel("Observer units"); ax.set_title(title)
    axes[0].legend(frameon=False, ncol=3, fontsize=12)
    name = "position_common_scores"
    save_figure(fig, output, name)
    captions[name] = "Matched cached VALIDATION100 J4 scores under the unchanged PH observer, the TRAIN-fitted regional-excursion observation, and their declared half-sum. A/R/S are shown separately. Absolute scores from different observers are not comparable; each endpoint is compared to SOURCE under the same observer."
    categories = sensitivity["categories"]
    keys = list(categories)
    reference = np.asarray([categories[k]["reference"]["absolute_filtration_derivative"] for k in keys])
    source = np.asarray([categories[k]["source_members"]["absolute_filtration_derivative"] for k in keys])
    saturation = np.asarray([np.asarray(categories[k]["source_members"]["window_near_empty_fraction"]) +
        np.asarray(categories[k]["source_members"]["window_near_full_fraction"]) for k in keys])
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.8), constrained_layout=True)
    limit = max(reference.max(), source.max())
    for ax, values, title, vmax in zip(axes, (source, saturation),
            ("SOURCE member derivative", "SOURCE saturated windows"),
            (limit, max(float(saturation.max()), .01)), strict=True):
        image = ax.imshow(values, vmin=0, vmax=vmax, cmap="viridis", aspect="auto")
        ax.set_xticks(range(3), ("q20", "q50", "q80")); ax.set_yticks(range(len(keys)), keys, fontsize=13)
        ax.set_title(title, fontsize=15); fig.colorbar(image, ax=ax, shrink=.7)
    name = "position_train_sensitivity"
    save_figure(fig, output, name)
    captions[name] = "Frozen TRAIN800 sensitivity over complete fixed target queries and all J4 SOURCE members. Absolute sigmoid derivatives have inverse standardized-filtration units; saturated windows have occupancy at or below 0.01 or at or above 0.99. Category and threshold pooling retains the declared physical raster mass and fixed query weights."
    return captions


def main():
    started, cpu = time.perf_counter(), time.process_time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fit, sensitivity, readout = [json.loads(p.read_text()) for p in (args.fit, args.sensitivity, args.readout)]
    if (fit.get("status") != "completed" or len(fit.get("sample_ids", ())) != 800
            or sensitivity.get("status") != "completed" or sensitivity.get("sample_ids") != fit["sample_ids"]
            or sensitivity.get("members") != 4 or sensitivity["inputs"]["fit_receipt"] != ref(args.fit)
            or readout["inputs"]["fit"] != ref(args.fit) or not sensitivity.get("test_locked")):
        raise ValueError("matching completed TRAIN fit/sensitivity and common observer readout required")
    checked_ref(fit["artifact"])
    summary = summarize(readout)
    args.output.mkdir(parents=True, exist_ok=False)
    folder = args.output / "figures"; folder.mkdir()
    captions = figures(summary, sensitivity, folder)
    result = {"schema": "car_round8_position_summary_v1", "status": "completed",
        "summary": summary, "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": 2000,
        "unit": "paired correlated frames; not independent trajectories"},
        "inputs": {name: ref(path) for name, path in (("fit", args.fit),
            ("sensitivity", args.sensitivity), ("readout", args.readout), ("script", Path(__file__)))},
        "test_locked": True, "timing": {"wall_seconds": time.perf_counter()-started,
            "cpu_seconds": time.process_time()-cpu, "gpu_seconds": 0}}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output / "figure_captions.json").write_text(json.dumps(captions, indent=2) + "\n")
    categories = sensitivity["categories"]
    derivative = np.asarray([row[side]["absolute_filtration_derivative"]
        for row in categories.values() for side in ("reference", "source_members")])
    empty = max(max(row["source_members"]["window_near_empty_fraction"]) for row in categories.values())
    full = max(max(row["source_members"]["window_near_full_fraction"]) for row in categories.values())
    duplicates = sum(row["duplicate_thresholds"] for row in categories.values())
    lines = ["## Positional Shape evidence", "",
        f"The position observation was fitted on cached TRAIN800 reference fields and all four SOURCE members, with the complete {fit['query_count']} fixed target queries. Its frozen location unit is {fit['location_unit']:.8g}; no VALIDATION values fitted thresholds, windows, temperatures or units. It reuses the existing filtrations and adds no persistence calculations.", "",
        f"The TRAIN reference and member absolute derivatives range from {derivative.min():.4g} to {derivative.max():.4g} over {len(categories)} filtration categories; {duplicates} categories have coincident thresholds. At most {100*empty:.3g}% of SOURCE windows within a category/threshold are nearly empty and at most {100*full:.3g}% nearly full. These are properties of this fitted raster observation, not a guarantee for every future input.", "",
        "| Observer | SOURCE A / R / S | Joint500 A / R / S | Joint500 − SOURCE S [paired 95% interval] |",
        "| --- | --- | --- | --- |"]
    for observer, label in zip(OBSERVERS, ("PH", "Location / TRAIN unit", "Declared half-sum", "Native-to-raster location / TRAIN unit"), strict=True):
        value = summary[observer]; delta = value["paired_source_deltas"]["JointAnchor500"]
        fmt = lambda row: " / ".join(f"{x:.6g}" for x in row)
        lines.append(f"| {label} | {fmt(value['means']['SOURCE'])} | {fmt(value['means']['JointAnchor500'])} | {delta['mean'][2]:.6g} [{delta['interval_95'][2][0]:.6g}, {delta['interval_95'][2][1]:.6g}] |")
    lines += ["", position_conclusion(summary), ""]
    (args.output / "section.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
