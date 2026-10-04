"""Compact R3 epoch report: four PDF masters plus one numerical JSON.

Saved development evidence only; no inference, fitting, TEST, or checkpoint
tensor loading. All windows include every recorded observation in their range.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from phycoflow_reconstruction.training.coherence_diagnostics import pressure_error_decomposition

SPEC = importlib.util.spec_from_file_location(
    "r2_saved_reader", Path(__file__).with_name("upgrade_1002_r2_campaign_report.py")
)
READER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(READER)
COLORS = ("#27698e", "#c77b29", "#537b40")
FAMILIES = ("global_distribution", "cross_spectrum", "topology")


def windows(rows, value):
    latest = max((row["epoch"] for row in rows), default=0)
    payload = {}
    for name, low, high in (
        ("rolling20", latest - 20, latest),
        ("rolling50", latest - 50, latest),
        ("previous25", latest - 50, latest - 25),
        ("final25", latest - 25, latest),
    ):
        observations = [
            value(row)
            for row in rows
            if low < row["epoch"] <= high and value(row) is not None and np.isfinite(value(row))
        ]
        payload[name] = {
            "mean": float(np.mean(observations)) if observations else None,
            "median": float(np.median(observations)) if observations else None,
            "observations": len(observations),
            "epoch_range": [low, high],
        }
    payload["final25_minus_previous25"] = (
        payload["final25"]["mean"] - payload["previous25"]["mean"]
        if payload["final25"]["mean"] is not None and payload["previous25"]["mean"] is not None
        else None
    )
    return payload


def draw(axis, rows, value, label, color, *, rolling=True, rolling_value=None):
    xy = [
        (row["epoch"], value(row))
        for row in rows
        if value(row) is not None and np.isfinite(value(row))
    ]
    if not xy:
        return
    x, y = np.asarray(xy).T
    axis.plot(x, y, color=color, alpha=0.35 if rolling else 1, linewidth=0.7, label=label)
    if rolling:
        for width, style in ((20, "-"), (50, "--")):
            means = [
                rolling_value([row for row in rows if epoch - width < row["epoch"] <= epoch])
                if rolling_value
                else np.mean(y[(x > epoch - width) & (x <= epoch)])
                for epoch in x
            ]
            axis.plot(x, means, color=color, linestyle=style, linewidth=1.1)


def finish(fig, axes, pdf, title):
    for axis in np.asarray(axes).reshape(-1):
        axis.grid(alpha=0.2)
        if not axis.get_xlabel():
            axis.set_xlabel("Epoch")
        handles, _labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=6, frameon=False, loc="best")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def native_ratio(row):
    source = row.get("fidelity/native/source_risk")
    child = row.get("fidelity/native/live_risk", row.get("loss/native"))
    return child / source if source is not None and source > 0 and child is not None else None


def pooled_native_ratio(rows, *, return_values=False):
    """Ratio of exposure-weighted mean losses, never mean stochastic ratios."""
    source_total, child_total, observations = 0.0, 0.0, 0
    for row in rows:
        source = row.get("fidelity/native/source_risk")
        child = row.get("fidelity/native/live_risk")
        if source is None or child is None or not np.isfinite([source, child]).all():
            continue
        counts = row.get("finite_counts", {})
        source_count = counts.get("fidelity/native/source_risk", row.get("batches", 1))
        child_count = counts.get("fidelity/native/live_risk", row.get("batches", 1))
        if source_count != child_count:
            raise ValueError("matched native epoch losses have unequal exposure counts")
        source_total += source * source_count
        child_total += child * child_count
        observations += source_count
    ratio = child_total / source_total if source_total > 0 else None
    return (
        {
            "ratio_of_mean_losses": ratio,
            "source_mean_loss": source_total / observations if observations else None,
            "candidate_mean_loss": child_total / observations if observations else None,
            "matched_batch_exposures": observations,
        }
        if return_values
        else ratio
    )


def native_windows(rows):
    latest = max((row["epoch"] for row in rows), default=0)
    payload = {}
    for name, low, high in (
        ("rolling20", latest - 20, latest),
        ("rolling50", latest - 50, latest),
        ("previous25", latest - 50, latest - 25),
        ("final25", latest - 25, latest),
    ):
        selected = [row for row in rows if low < row["epoch"] <= high]
        payload[name] = {
            **pooled_native_ratio(selected, return_values=True),
            "observed_epochs": len(selected),
            "epoch_range": [low, high],
        }
    left, right = (payload[name]["ratio_of_mean_losses"] for name in ("previous25", "final25"))
    payload["final25_minus_previous25"] = (
        right - left if left is not None and right is not None else None
    )
    payload["aggregation"] = (
        "sum(epoch mean LIVE risk * matched recorded batch count) / sum(epoch mean SOURCE risk * same count)"
    )
    return payload


def ratio_value(row, name):
    source = row.get(f"fidelity/{name}/source_risk")
    child = row.get(f"fidelity/{name}/live_risk")
    return child / source if source is not None and source > 0 and child is not None else None


def topology_components(run):
    rows = []
    divisor = run["manifest"]["steps_per_epoch"]
    path = Path(run["run_dir"]) / "metrics/coherence_validation.jsonl"
    for row in READER._read_jsonl(path):
        components = (
            row.get("metrics", {})
            .get("coherence", {})
            .get("families", {})
            .get("topology", {})
            .get("component_scalars", {})
        )
        values = {
            key: components.get(
                f"topology.normalized_{key.split('_')[0]}"
                if key.endswith("_normalized")
                else f"topology.{key}"
            )
            for key in (
                "finite_raw",
                "essential_raw",
                "finite_normalized",
                "essential_normalized",
                "legacy",
                "finite_primary",
            )
        }
        rows.append({"epoch": row["step"] / divisor, **values})
    return rows


def summarize(run):
    epochs = [row for row in run["epochs"] if row.get("epoch_complete")]
    source_mse = run["before"].get("mse_normalized")
    selection = run["selection"]
    final = max((row["epoch"] for row in epochs), default=0)
    family_windows = {
        name: windows(selection, lambda row, name=name: row["family_ratios"].get(name))
        for name in FAMILIES
    }
    medians = [family_windows[name]["rolling50"]["median"] for name in FAMILIES]
    useful = (
        all(value is not None for value in medians)
        and max(medians) <= 1.10
        and (sum(value <= 0.97 for value in medians) >= 2 or np.mean(medians) <= 0.95)
    )
    scale = next(
        (
            row["fidelity/native/frozen_scale"]
            for row in epochs
            if "fidelity/native/frozen_scale" in row
        ),
        None,
    )
    budget = run["config"].get("fidelity_controller", {}).get("native_budget")
    return {
        "label": run["label"],
        "completed_epochs": final,
        "native_budget": budget,
        "frozen_TRAIN_native_scale": scale,
        "absolute_native_allowance": scale * budget
        if scale is not None and budget is not None
        else None,
        "native_ratio_windows": native_windows(epochs),
        "native_raw_windows": windows(epochs, lambda row: row.get("loss/native")),
        "family_ratio_windows": family_windows,
        "useful_coherence_screen": bool(useful),
        "endpoint_total_windows": windows(
            selection,
            lambda row: row["mse"] / source_mse if row["mse"] is not None and source_mse else None,
        ),
        "field_ratio_windows": {
            field: windows(
                selection,
                lambda row, field=field: (
                    1 + row["per_field_relative_change"][field]
                    if field in row["per_field_relative_change"]
                    else None
                ),
            )
            for field in run["field_names"]
        },
        "source_weight_selection": "live",
        "test_accessed": False,
    }


def load_calibration(path):
    """Mean fixed TRAIN batch diagnostics, keeping norms explicitly averaged."""
    payload = json.loads(Path(path).read_text())
    if "components" in payload:
        if payload.get("split") not in (None, "train"):
            raise ValueError("calibration figures require TRAIN diagnostics")
        return payload
    records = [
        family
        for batch in payload.get("calibration_batches", [])
        for family in batch.get("R3_component_diagnostics", {}).values()
    ]
    if not records or any(row.get("split") != "train" for row in records):
        raise ValueError("missing fixed TRAIN R3 component diagnostics")
    components = {}
    for name in sorted({name for row in records for name in row["components"]}):
        rows = [row["components"][name] for row in records if name in row["components"]]
        components[name] = {}
        for key in rows[0]:
            values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
            components[name][key] = float(np.mean(values)) if values else None
        components[name]["observed_batches"] = len(rows)
    return {
        "components": components,
        "split": "train",
        "representation": "training_raster",
        "aggregation": "arithmetic mean of fixed batch values and parameter-gradient norms",
        "calibration_batches": len(payload["calibration_batches"]),
    }


def load_audit(label, path):
    payload = json.loads(Path(path).read_text())
    if payload.get("split") != "validation" or payload.get("test_accessed") is not False:
        raise ValueError("R3 figures accept declared development audits only")
    source, child = (payload["results"][role] for role in ("source_live", "candidate_live"))
    selection = payload["R3_selection_descriptive"]
    names = tuple(source["per_field_mse_normalized"])
    return {
        "label": label,
        "epoch": payload["epoch"],
        "panel": payload["panel"],
        "source_mse": source["mse_normalized"],
        "candidate_mse": child["mse_normalized"],
        "total_MSE_ratio": child["mse_normalized"] / source["mse_normalized"],
        "field_MSE_ratios": {
            name: child["per_field_mse_normalized"][name] / source["per_field_mse_normalized"][name]
            for name in names
        },
        "family_source_ratios": selection["family_source_normalized_scores"],
        "ABC_mean_ratio": selection["metric"],
        "topology_selection_accounting": selection["topology_selection_accounting"],
        "native_monitor": payload["matched_native_monitor"],
        "B_regrouping": payload["B_regrouping"],
        "native_fields": payload.get("native_fields"),
        "evidence_path": str(path),
        "test_accessed": False,
    }


def render_fields(specifications, pdf):
    if not specifications:
        fig, axis = plt.subplots(figsize=(8, 3))
        axis.text(
            0.5,
            0.5,
            "Native development field arrays have not been exported.\nField-quality evidence remains unavailable.",
            ha="center",
            va="center",
        )
        axis.axis("off")
        pdf.savefig(fig)
        plt.close(fig)
        return []
    evidence = []
    for label, path in specifications:
        with np.load(path, allow_pickle=False) as data:
            if str(data.get("split", "validation")) != "validation":
                raise ValueError("R3 fields require development validation arrays, never TEST")
            if not bool(data["full_native_grid"]):
                raise ValueError("R3 contours require complete native lattice")
            names = data["field_names"].tolist()
            count = min(2, len(data["ground_truth"]))
            for index in range(count):
                coords = data["coordinates_raw"][index]
                nx, ny = len(np.unique(coords[:, 0])), len(np.unique(coords[:, 1]))
                order = np.lexsort((coords[:, 0], coords[:, 1]))
                if nx * ny != len(coords):
                    raise ValueError("native coordinates do not form rectangular lattice")
                fig, axes = plt.subplots(len(names), 5, figsize=(12, 7.8), squeeze=False)
                for field_id, field in enumerate(names):
                    values = [
                        data[role][index, :, field_id]
                        for role in ("ground_truth", "source_live", "candidate_live")
                    ]
                    errors = [values[1] - values[0], values[2] - values[0]]
                    lo, hi = (
                        min(value.min() for value in values),
                        max(value.max() for value in values),
                    )
                    lim = max(max(np.abs(error).max() for error in errors), 1e-12)
                    for column, value in enumerate(values + errors):
                        axis = axes[field_id, column]
                        image = axis.imshow(
                            value[order].reshape(ny, nx),
                            origin="lower",
                            aspect="auto",
                            extent=(
                                coords[:, 0].min(),
                                coords[:, 0].max(),
                                coords[:, 1].min(),
                                coords[:, 1].max(),
                            ),
                            cmap="viridis" if column < 3 else "RdBu_r",
                            vmin=lo if column < 3 else -lim,
                            vmax=hi if column < 3 else lim,
                        )
                        axis.set_xticks([])
                        axis.set_yticks([])
                        if column == 0:
                            axis.set_ylabel(field, rotation=0, labelpad=14)
                        if field_id == 0:
                            axis.set_title(
                                ("GT", "SOURCE LIVE", "Child LIVE", "SOURCE − GT", "Child − GT")[
                                    column
                                ],
                                fontsize=9,
                            )
                        if column in (2, 4):
                            fig.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
                        if column < 3 and "observation_coordinates_raw" in data:
                            mask = data["observation_valid_mask"][index] & (
                                data["observation_field_ids"][index] == field_id
                            )
                            sensor = data["observation_coordinates_raw"][index][mask]
                            axis.scatter(sensor[:, 0], sensor[:, 1], s=2, c="white", linewidths=0)
                fig.suptitle(
                    f"{label} • epoch {int(data['epoch'])} • {data['sample_ids'][index]} • native {len(coords)} points; units unspecified",
                    fontsize=10,
                )
                fig.tight_layout(rect=(0, 0, 1, 0.95))
                pdf.savefig(fig, bbox_inches="tight", pad_inches=0.12)
                plt.close(fig)
            p = names.index("p")
            evidence.append(
                {
                    "label": label,
                    "path": str(path),
                    "displayed_snapshots": count,
                    "pressure": {
                        role: pressure_error_decomposition(
                            data[role][..., p], data["ground_truth"][..., p]
                        )
                        for role in ("source_live", "candidate_live")
                    },
                }
            )
    return evidence


def render(runs, output, fields=(), calibration=(), audits=()):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    with PdfPages(output / "R3_training_and_coherence.pdf") as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(11, 6.6))
        for i, run in enumerate(runs):
            color = COLORS[i % len(COLORS)]
            draw(
                axes[0, 0],
                run["epochs"],
                native_ratio,
                run["label"],
                color,
                rolling_value=pooled_native_ratio,
            )
            draw(
                axes[0, 1],
                run["epochs"],
                lambda row: ratio_value(row, "total"),
                run["label"],
                color,
            )
            draw(
                axes[0, 2],
                run["native"],
                lambda row: row.get("validation", {}).get("ratio"),
                run["label"],
                color,
                rolling=False,
            )
            for j, family in enumerate(FAMILIES):
                draw(
                    axes[1, j],
                    run["selection"],
                    lambda row, family=family: row["family_ratios"].get(family),
                    run["label"],
                    color,
                )
        for axis, title in zip(
            axes.flat,
            (
                "TRAIN matched native mean-risk ratio",
                "TRAIN matched endpoint ratio",
                "Frozen validation native draws",
                "A own-CDF/smooth-tail",
                "B grouped32 signed blocks",
                "C same-definition source ratio",
            ),
        ):
            axis.set_title(title)
            axis.axhline(1, c="grey", ls=":", lw=0.7)
        axes[0, 1].axhline(1.05, c="#a7473d", ls=":", lw=0.7)
        axes[0, 2].axhline(1.05, c="#a7473d", ls=":", lw=0.7)
        finish(
            fig,
            axes,
            pdf,
            "R3 histories: raw translucent, rolling20 solid, rolling50 dashed; no TEST",
        )
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
        for i, run in enumerate(runs):
            color = COLORS[i % len(COLORS)]
            for axis, key in zip(
                axes,
                (
                    "fidelity/native/multiplier",
                    "fidelity/native/violation",
                    "gradient/clipping_applied_fraction",
                ),
            ):
                draw(axis, run["epochs"], lambda row, key=key: row.get(key), run["label"], color)
        for axis, title in zip(
            axes, ("Native multiplier", "Native normalized violation", "Clipping fraction")
        ):
            axis.set_title(title)
        finish(fig, axes, pdf, "Controller evidence; native allowance is 0.02 × frozen TRAIN scale")
    with PdfPages(output / "R3_finite_essential_topology.pdf") as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(11, 6))
        for i, run in enumerate(runs):
            rows = topology_components(run)
            for axis, key in zip(
                axes.flat,
                (
                    "finite_raw",
                    "essential_raw",
                    "legacy",
                    "finite_normalized",
                    "essential_normalized",
                    "finite_primary",
                ),
            ):
                draw(
                    axis,
                    rows,
                    lambda row, key=key: row.get(key),
                    run["label"],
                    COLORS[i % len(COLORS)],
                )
        for axis, title in zip(
            axes.flat,
            (
                "Finite old/HW",
                "Essential raw unscaled/HW",
                "Historical C, separate definition",
                "Finite TRAIN normalized",
                "Essential TRAIN normalized",
                "Finite-primary 0.9F + 0.1E",
            ),
        ):
            axis.set_title(title)
        finish(
            fig,
            axes,
            pdf,
            "C at training raster; full16 validation bank; raw and normalized remain separate",
        )
        if calibration:
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
            for i, (label, payload) in enumerate(calibration):
                records = payload.get("components", {})
                wanted = {
                    key: value
                    for key, value in records.items()
                    if any(
                        term in key
                        for term in (
                            "marginal",
                            "copula",
                            "second_order",
                            "finite_primary_finite",
                            "finite_primary_essential",
                            "native.raw",
                            "endpoint.raw",
                        )
                    )
                }
                names = list(wanted)
                axes[0].bar(
                    np.arange(len(names)) + i * 0.2,
                    [wanted[key]["weighted_value"] for key in names],
                    width=0.2,
                    label=label,
                )
                axes[1].bar(
                    np.arange(len(names)) + i * 0.2,
                    [wanted[key]["weighted_parameter_gradient_norm"] for key in names],
                    width=0.2,
                    label=label,
                )
                for axis in axes:
                    axis.set_xticks(
                        np.arange(len(names)),
                        [name.rsplit(".", 1)[-1] for name in names],
                        rotation=35,
                        ha="right",
                        fontsize=6,
                    )
            axes[0].set_title("Fixed TRAIN component contributions")
            axes[1].set_title("Model-parameter gradient norms")
            finish(
                fig,
                axes,
                pdf,
                "Fixed SOURCE LIVE TRAIN calibration; zero pressure at feasible source is allowed",
            )
        native_audits = [audit for audit in audits if audit.get("native_fields")]
        if native_audits:
            fig, axes = plt.subplots(2, 2, figsize=(10, 5))
            for index, audit in enumerate(native_audits):
                components = audit["native_fields"]["components"]
                for axis, key in zip(
                    axes.flat,
                    ("finite_raw", "essential_raw", "normalized_finite", "normalized_essential"),
                ):
                    source = components["source_live"][f"topology.{key}"]
                    child = components["candidate_live"][f"topology.{key}"]
                    axis.bar(
                        index - 0.16,
                        source,
                        width=0.3,
                        color="grey",
                        alpha=0.6,
                        label="SOURCE LIVE" if index == 0 else None,
                    )
                    axis.bar(
                        index + 0.16,
                        child,
                        width=0.3,
                        color=COLORS[index % len(COLORS)],
                        label="Child LIVE" if index == 0 else None,
                    )
                    axis.set_xticks(
                        range(len(native_audits)),
                        [row["label"] for row in native_audits],
                        fontsize=7,
                    )
                    axis.set_xlabel("Saved mature development panel")
                    axis.set_title(f"Native {key.replace('_', ' ')}")
            finish(
                fig,
                axes,
                pdf,
                "Native100×403 PH: separate ≤4 TRAIN SOURCE calibration; ≤4 DEV snapshots; full16 GUDHI",
            )
    with PdfPages(output / "R3_fidelity_and_pareto.pdf") as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(11, 6.3))
        names = runs[0]["field_names"] if runs else []
        for i, run in enumerate(runs):
            color = COLORS[i % len(COLORS)]
            for axis, field in zip(axes.flat, names):
                draw(
                    axis,
                    run["selection"],
                    lambda row, field=field: (
                        1 + row["per_field_relative_change"][field]
                        if field in row["per_field_relative_change"]
                        else None
                    ),
                    run["label"],
                    color,
                )
            source_mse = run["before"].get("mse_normalized")
            points = [
                row
                for row in run["selection"]
                if row["epoch"] >= 100
                and row["mse"] is not None
                and row["family_mean_diagnostic"] is not None
            ]
            axes.flat[-1].plot(
                [row["mse"] / source_mse for row in points],
                [row["family_mean_diagnostic"] for row in points],
                "o-",
                ms=3,
                lw=0.7,
                color=color,
                label=run["label"],
            )
        for axis, field in zip(axes.flat, names):
            axis.set_title(f"{field}: original endpoint MSE ratio")
            for level, style in ((1, ":"), (1.05, "--"), (1.10, ":")):
                axis.axhline(level, c="grey" if level == 1 else "#a7473d", ls=style, lw=0.6)
        for audit in audits:
            for axis, field in zip(axes.flat, names):
                axis.plot(
                    audit["epoch"], audit["field_MSE_ratios"][field], "*", color="black", ms=7
                )
            axes.flat[-1].plot(
                audit["total_MSE_ratio"], audit["ABC_mean_ratio"], "*", color="black", ms=7
            )
        axes.flat[-1].set_title("Mature selection observations ≥100 epochs")
        axes.flat[-1].set_xlabel("Total endpoint MSE / SOURCE")
        axes.flat[-1].set_ylabel("Mean A/B/C source ratio")
        finish(
            fig,
            axes,
            pdf,
            "Strict fields ≤1.05; exploratory fields ≤1.10; total/native ≤1.05; every saved observation shown",
        )
    with PdfPages(output / "R3_fields.pdf") as pdf:
        field_evidence = render_fields(fields, pdf)
    summary = {
        "schema": "phycoflow.upgrade_1002_r3_campaign_report.v1",
        "runs": [summarize(run) for run in runs],
        "fields": field_evidence,
        "development_audits": list(audits),
        "test_accessed": False,
        "formal_run_launched": False,
        "window_policy": "all saved observations in (latest-width,latest]; mean/median; sparse count explicit",
        "figures": [
            f"R3_{name}.pdf"
            for name in (
                "training_and_coherence",
                "finite_essential_topology",
                "fidelity_and_pareto",
                "fields",
            )
        ],
    }
    (output / "campaign_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="LABEL=RUN_DIRECTORY")
    parser.add_argument("--fields", action="append", default=[], help="LABEL=NATIVE_NPZ")
    parser.add_argument(
        "--calibration", action="append", default=[], help="LABEL=GRADIENT_DIAGNOSTIC_JSON"
    )
    parser.add_argument(
        "--audit", action="append", default=[], help="LABEL=R3_AUDIT_JSON; development only"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2] / "cases/turbulent_combustion/runs/Test_1002"
    args.output.resolve().relative_to(root.resolve())
    runs = []
    for value in args.run:
        label, path = value.split("=", 1)
        if not Path(path).resolve().relative_to(root.resolve()).parts[0].startswith("R3_"):
            raise ValueError("R3 campaign report accepts only R3 runs")
        runs.append(READER.read_run(label, Path(path)))
    fields = [(label, Path(path)) for label, path in (value.split("=", 1) for value in args.fields)]
    calibration = [
        (label, load_calibration(path))
        for label, path in (value.split("=", 1) for value in args.calibration)
    ]
    audits = [
        load_audit(label, path) for label, path in (value.split("=", 1) for value in args.audit)
    ]
    result = render(runs, args.output, fields, calibration, audits)
    print(
        json.dumps(
            {
                "completed_epochs": {
                    run["label"]: run["completed_epochs"] for run in result["runs"]
                },
                "output": str(args.output),
                "test_accessed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
