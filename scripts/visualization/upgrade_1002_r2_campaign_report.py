"""Read-only epoch/PDF campaign reporting from saved R2 numerical evidence.

No model, dataset, checkpoint tensor, or test split is opened. Missing series
and unavailable mature windows remain missing. Run with --run LABEL=RUN_DIR.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.evaluation.upgrade_1002_r2_protocol import (
    epoch_window_summary,
    mature_checkpoint_summary,
)
from phycoflow_reconstruction.training.coherence_history import (
    _finite_number,
    _read_jsonl,
    build_adaptive_coherence_figures,
    extract_adaptive_coherence_history,
)
from phycoflow_reconstruction.training.run_store import file_sha256

FAMILIES = ("global_distribution", "cross_spectrum", "topology")
LABELS = {"global_distribution": "A", "cross_spectrum": "B", "topology": "C"}
COLORS = ("#2476A5", "#C7772A", "#577D46", "#A05B86", "#7B688F", "#3B8B87")
AUDIT_MARKERS = {
    "selection": ("D", "saved selection64"),
    "extended": ("s", "extended128"),
    "audit": ("d", "fresh audit64"),
}


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def epoch_means(rows, steps_per_epoch):
    """Merge resumed partial epochs using each metric's recorded finite count."""
    totals, counts, batch_counts = defaultdict(dict), defaultdict(dict), defaultdict(int)
    explicit_counts = defaultdict(dict)
    for row in rows:
        epoch = math.ceil(int(row["step"]) / steps_per_epoch)
        batches = int(row.get("batches", 1))
        batch_counts[epoch] += batches
        for key, value in row.items():
            numeric = _finite_number(value)
            if numeric is None or key in {"epoch", "step", "batches"}:
                continue
            weight = int(row.get("finite_counts", {}).get(key, batches))
            explicit_counts[epoch][key] = explicit_counts[epoch].get(key, True) and key in row.get(
                "finite_counts", {}
            )
            totals[epoch][key] = totals[epoch].get(key, 0.0) + numeric * weight
            counts[epoch][key] = counts[epoch].get(key, 0) + weight
    result = []
    for epoch in sorted(totals):
        result.append(
            {
                "epoch": epoch,
                "batches": batch_counts[epoch],
                "epoch_complete": batch_counts[epoch] == steps_per_epoch,
                "finite_counts": counts[epoch],
                "finite_count_scope": {
                    key: "recorded" if value else "batch_count_fallback"
                    for key, value in explicit_counts[epoch].items()
                },
                **{
                    key: value / counts[epoch][key]
                    for key, value in totals[epoch].items()
                    if counts[epoch][key]
                },
            }
        )
        update_seconds = result[-1].get("update_seconds", result[-1].get("runtime/step_seconds"))
        if update_seconds is not None:
            result[-1]["recorded_update_seconds_in_epoch"] = update_seconds * batch_counts[epoch]
        monitor_seconds = result[-1].get("runtime/native_monitor_seconds")
        if monitor_seconds is not None:
            result[-1]["recorded_native_monitor_seconds_in_epoch"] = (
                monitor_seconds * counts[epoch]["runtime/native_monitor_seconds"]
            )
    return result


CONTROLLER_DIAGNOSTICS = {
    "native_active_fraction": "gradient/native/constraint_active_fraction",
    "clipping_fraction": "gradient/clipping_applied_fraction",
    "native_norm": "gradient/native/norm",
    "fidelity_norm": "fidelity_grad_norm",
    "coherence_raw_norm": "gradient/coherence_raw_norm",
    "coherence_calibrated_norm": "gradient/coherence_calibrated_norm",
    "native_actual_dot": "update/actual_dot/native",
    "native_proposal_dot": "gradient/final_direction_dot/native",
}


def controller_diagnostics(epochs):
    """Completed epoch means only; ratios never average per-batch ratios."""
    records = []
    for row in epochs:
        if not row.get("epoch_complete", False):
            continue
        item = {"epoch": row["epoch"], "batches": row["batches"], "diagnostics": {}}
        for name, key in CONTROLLER_DIAGNOSTICS.items():
            value = _finite_number(row.get(key))
            fraction = name.endswith("fraction")
            count = row.get("finite_counts", {}).get(key)
            scope = (
                "processed_batch_fraction" if fraction else "mean_over_recorded_finite_diagnostics"
            )
            reason = "not_recorded" if value is None else None
            if value is not None and fraction:
                if not 0 <= value <= 1:
                    value, reason = None, "fraction_outside_zero_one"
                count = row["batches"]
            elif value is not None:
                if (
                    count is None
                    or row.get("finite_count_scope", {}).get(key) == "batch_count_fallback"
                ):
                    value, count, reason = None, None, "diagnostic_count_not_recorded"
                elif count <= 0:
                    value, reason = None, "zero_recorded_diagnostics"
            item["diagnostics"][name] = {
                "value": value,
                "finite_count": count,
                "scope": scope,
                "undefined_reason": reason,
            }
        numerator = item["diagnostics"]["fidelity_norm"]
        denominator = item["diagnostics"]["coherence_calibrated_norm"]
        ratio, reason = None, None
        if numerator["value"] is None or denominator["value"] is None:
            reason = "numerator_or_denominator_diagnostics_missing"
        elif denominator["value"] <= 0:
            reason = (
                "zero_denominator" if denominator["value"] == 0 else "invalid_negative_denominator"
            )
        else:
            ratio = numerator["value"] / denominator["value"]
        item["fidelity_over_calibrated_coherence"] = {
            "value": ratio,
            "definition": "ratio_of_epoch_mean_norms_not_mean_of_ratios",
            "numerator_finite_count": numerator["finite_count"],
            "denominator_finite_count": denominator["finite_count"],
            "undefined_reason": reason,
        }
        records.append(item)
    return records


def controller_gradient_page(run):
    """Second controller page: saved local diagnostics, without recomputation."""
    records = controller_diagnostics(run["epochs"])
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")

    def plot_values(axis, name, label, color, *, count=False):
        values = [
            (row["epoch"], row["diagnostics"][name]["finite_count" if count else "value"])
            for row in records
        ]
        if any(_finite_number(value) is not None for _, value in values):
            line(
                axis,
                [(epoch, value if value is not None else np.nan) for epoch, value in values],
                label,
                color,
            )

    for index, (name, label) in enumerate(
        (
            ("native_active_fraction", "native constraint active"),
            ("clipping_fraction", "clipping applied"),
        )
    ):
        plot_values(axes[0, 0], name, label, COLORS[index])
    axes[0, 0].set_title("True processed-batch fractions")
    axes[0, 0].set_ylabel("Fraction; never any-active boolean")
    axes[0, 0].set_ylim(0, 1.05)
    plot_values(axes[0, 1], "native_norm", "bare native", COLORS[0])
    axes[0, 1].set_title("Bare native gradient norm")
    axes[0, 1].set_ylabel("Mean over recorded finite diagnostics")
    for index, (name, label) in enumerate(
        (
            ("fidelity_norm", "fidelity"),
            ("coherence_raw_norm", "coherence raw"),
            ("coherence_calibrated_norm", "coherence calibrated"),
        )
    ):
        plot_values(axes[0, 2], name, label, COLORS[index])
    axes[0, 2].set_title("Fidelity and coherence gradient norms")
    axes[0, 2].set_ylabel("Mean over recorded finite diagnostics")
    ratios = [(row["epoch"], row["fidelity_over_calibrated_coherence"]["value"]) for row in records]
    if any(value is not None for _, value in ratios):
        line(
            axes[1, 0],
            [(epoch, value if value is not None else np.nan) for epoch, value in ratios],
            "fidelity / calibrated coherence",
            COLORS[3],
        )
    axes[1, 0].set_title("Ratio of epoch-mean norms")
    axes[1, 0].set_ylabel("Missing if denominator is zero/unrecorded")
    for index, (name, label) in enumerate(
        (
            ("native_actual_dot", "native gradient · actual displacement"),
            ("native_proposal_dot", "native gradient · proposal direction"),
        )
    ):
        plot_values(axes[1, 1], name, label, COLORS[index])
    axes[1, 1].axhline(0, color=".3", linestyle="--", linewidth=0.7)
    axes[1, 1].set_yscale("symlog", linthresh=0.001)
    axes[1, 1].set_title("Local directional diagnostics · symlog; no guarantees")
    axes[1, 1].set_ylabel("Mean dot; actual negative / proposal positive favors decrease")
    for index, name in enumerate(tuple(CONTROLLER_DIAGNOSTICS)[2:]):
        plot_values(
            axes[1, 2], name, name.replace("_", " "), COLORS[index % len(COLORS)], count=True
        )
    axes[1, 2].set_title("Recorded finite diagnostic counts")
    axes[1, 2].set_ylabel("Count per completed epoch, per metric")
    axes[1, 2].yaxis.set_major_locator(MaxNLocator(integer=True))
    latest = max((row["epoch"] for row in records), default=0)
    for axis in axes.flat:
        epoch_axis(axis, latest)
        axis.grid(alpha=0.2)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(
                fontsize=7,
                frameon=False,
                ncol=2 if len(axis.get_legend_handles_labels()[0]) > 3 else 1,
            )
        else:
            axis.text(0.5, 0.5, "Diagnostics unavailable", transform=axis.transAxes, ha="center")
    figure.suptitle(
        run["label"]
        + " · completed merged epochs only\nSparse diagnostic means use their recorded counts; unknown legacy counts and zero ratios remain missing",
        fontsize=10,
    )
    return figure


def family_reporting_ratios(source, candidate):
    """Own-definition saved totals only; never replace a legacy selector."""
    before = source.get("coherence", {}).get("families", {})
    after = candidate.get("coherence", {}).get("families", {})
    return {
        name: float(values["total"]) / max(float(before[name]["total"]), 1e-12)
        for name, values in after.items()
        if name in before
        and _finite_number(values.get("total")) is not None
        and _finite_number(before[name].get("total")) is not None
    }


def family_mean_diagnostic(ratios):
    values = [_finite_number(ratios.get(name)) for name in FAMILIES]
    return sum(values) / len(values) if all(value is not None for value in values) else None


def topology_bank_label(config, *, training=False):
    mutual = (
        config.get("coherence", {})
        .get("families", {})
        .get("topology", {})
        .get("components", {})
        .get("mutual", {})
    )
    if "line_bank_size" in mutual:
        size = int(mutual["line_bank_size"])
        subset = int(mutual.get("training_subset_size", min(4, size)))
        return f"sampled {subset}/{size}-line bank" if training else f"full {size}-line bank"
    if "lines" in mutual:
        return f"fixed {int(mutual['lines'])}-line bank"
    return "bank not recorded"


def read_run(label, run_dir):
    manifest = read_json(run_dir / "run_manifest.json")
    divisor = int(manifest["steps_per_epoch"])
    if divisor < 1:
        raise ValueError("saved historical epoch mapping must be positive")
    config = load_config(run_dir / "resolved_config.yaml")
    history = _read_jsonl(run_dir / "metrics/history.jsonl")
    epochs = epoch_means(history, divisor)
    before = read_json(run_dir / "evaluation/before.json")
    source_fields = before.get("per_field_mse_normalized", {})
    selection = []
    legacy_selector = (
        config.get("checkpointing", {}).get("selection_metric") == "topology_with_fidelity"
    )
    history_name = "topology_validation" if legacy_selector else "coherence_validation"
    for row in _read_jsonl(run_dir / f"metrics/{history_name}.jsonl"):
        metrics = row.get("metrics", {})
        ratios = row.get("family_source_normalized_scores", {})
        if legacy_selector and not ratios:
            ratios = family_reporting_ratios(before, metrics)
        item = {
            "epoch": float(row["step"]) / divisor,
            "eligible": bool(row.get("eligible", False)),
            "is_source_baseline": bool(row.get("is_source_baseline", False)) or row["step"] == 0,
            "metric": _finite_number(row.get("metric")),
            "mse": _finite_number(row.get("mse")),
            "failed_fidelity_fields": row.get("failed_fidelity_fields", []),
            "family_ratios": ratios,
            "family_ratio_role": "within_definition_reporting_only"
            if legacy_selector
            else "selector_components",
            "family_mean_diagnostic": family_mean_diagnostic(ratios),
            "selector_definition": "legacy_raw_topology_with_fidelity"
            if legacy_selector
            else row.get("definition", "equal_active_family_mean_source_ratio_v1"),
            "pareto_metric": family_mean_diagnostic(ratios)
            if legacy_selector
            else _finite_number(row.get("metric")),
            "per_field_mse": metrics.get("per_field_mse_normalized", {}),
            "decoded_relative_l2": metrics.get("per_field_relative_l2_physical", {}),
            "error_decomposition": metrics.get("error_decomposition", {}),
        }
        item["per_field_relative_change"] = {
            field: value / source_fields[field] - 1
            for field, value in item["per_field_mse"].items()
            if source_fields.get(field, 0) > 0 and _finite_number(value) is not None
        }
        item["worst_field_relative_change"] = max(
            item["per_field_relative_change"].values(), default=None
        )
        selection.append(item)
    by_epoch = {row["epoch"]: row for row in selection}
    selection = [by_epoch[epoch] for epoch in sorted(by_epoch)]
    native = []
    initial = read_json(run_dir / "evaluation/native_before.json")
    if initial:
        native.append({"epoch": 0, **initial})
    for path in sorted((run_dir / "evaluation").glob("native_epoch_*.json")):
        epoch = int(path.stem.rsplit("_", 1)[-1])
        native.append({"epoch": epoch, **read_json(path)})
    # Row telemetry is a fallback for historical files that lack native sidecars.
    if len(native) <= 1:
        for row in epochs:
            roles = {
                role: {
                    key: row.get(f"native_monitor/{role}/{key}")
                    for key in ("candidate", "source", "ratio")
                }
                for role in ("train", "validation")
            }
            if any(values["ratio"] is not None for values in roles.values()):
                native.append({"epoch": row["epoch"], **roles})
    return {
        "label": label,
        "run_dir": str(run_dir),
        "manifest": manifest,
        "config": config,
        "epochs": epochs,
        "history": history,
        "selection": selection,
        "native": native,
        "before": before,
        "selection_history": history_name,
        "effective_topology_bank": topology_bank_label(config),
        "latest_epoch": max((row["epoch"] for row in epochs), default=0),
        "field_names": list(source_fields),
        "audits": [],
    }


def compact_endpoint(result):
    """Keep saved scalars, excluding descriptor matrices and query arrays."""
    families = result.get("coherence", {}).get("families", {})
    return {
        "mse": _finite_number(result.get("mse_normalized")),
        "per_field_mse": result.get("per_field_mse_normalized", {}),
        "decoded_relative_l2": result.get("per_field_relative_l2_physical", {}),
        "error_decomposition": result.get("error_decomposition", {}),
        "family_totals": {
            name: _finite_number(values.get("total")) for name, values in families.items()
        },
        "family_components": result.get("coherence", {}).get("components", {}),
        "pooled_B": result.get("pooled_B"),
    }


def compact_native(values):
    return {
        role: {
            **{
                key: item.get(key)
                for key in (
                    "candidate",
                    "source",
                    "ratio",
                    "draws",
                    "group_size",
                    "aggregation",
                    "rng_policy",
                    "role",
                )
                if key in item
            },
            "sample_count": len(item.get("sample_ids", [])),
            "groups": [
                {
                    "group_index": group.get("group_index", index),
                    "sample_count": len(group.get("sample_ids", [])),
                    **{
                        key: group.get(key)
                        for key in ("candidate", "source", "ratio", "draws")
                        if key in group
                    },
                }
                for index, group in enumerate(item.get("groups", []))
            ],
        }
        for role, item in values.items()
    }


def read_audit(run, path):
    """Associate a saved development panel with one run and its immutable source."""
    audit = read_json(path)
    if audit.get("schema") != "phycoflow.upgrade_1002_r2_audit.v1":
        raise ValueError("unsupported or missing R2 audit schema")
    if Path(audit.get("run", "")).resolve() != Path(run["run_dir"]).resolve():
        raise ValueError("audit run does not match its report label")
    source_hash = run["manifest"].get("source_hashes", {}).get("checkpoint")
    if not source_hash or audit.get("source_checkpoint_sha256") != source_hash:
        raise ValueError("audit immutable source hash differs from the labeled run")
    panel = audit.get("panel")
    expected_split = "train" if panel == "training" else "validation"
    if (
        panel not in {"training", "selection", "extended", "audit"}
        or audit.get("split") != expected_split
        or not audit.get("test_locked")
    ):
        raise ValueError("only declared development panels with locked test are supported")
    expected_count = {"training": 32, "selection": 64, "extended": 128, "audit": 64}[panel]
    indices = audit.get("dataset_indices", [])
    if len(indices) != expected_count or len(set(indices)) != expected_count:
        raise ValueError("audit sample count differs from its declared panel")
    epoch = int(audit["epoch"])
    if panel in {"extended", "audit"} and epoch < 100:
        raise ValueError("broad development audit must be mature")
    results = audit.get("results", {})
    source, candidate = [
        compact_endpoint(results.get(role, {})) for role in ("source_live", "candidate_live")
    ]
    relative = {
        field: value / source["per_field_mse"][field] - 1
        for field, value in candidate["per_field_mse"].items()
        if _finite_number(value) is not None and source["per_field_mse"].get(field, 0) > 0
    }
    fidelity = audit.get("fidelity_coherence_audit") or {}
    groups = []
    for role, result in results.items():
        for index, group in enumerate(result.get("group_reports", [])):
            groups.append(
                {
                    "role": role,
                    "group_index": index,
                    "sample_count": len(group.get("sample_ids", [])),
                    **compact_endpoint(group),
                }
            )
    record = {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "panel": panel,
        "split": expected_split,
        "epoch": epoch,
        "sample_count": expected_count,
        "dataset_indices": indices,
        "purpose": audit.get("purpose"),
        "scientific_maturity": bool(audit.get("scientific_maturity", epoch >= 100)),
        "checkpoint_sha256": audit.get("checkpoint_sha256"),
        "source_checkpoint_sha256": source_hash,
        "sensor_manifest_sha256": audit.get("sensor_manifest_sha256"),
        "declaration_sha256": audit.get("declaration_sha256"),
        "group_role": "equal_group32_selection_scalars_with_separate_pooled_B"
        if candidate["pooled_B"] is not None
        else "equal_group32_selection_scalars_no_pooled_B",
        "pooled_B_present": candidate["pooled_B"] is not None,
        "effective_topology_bank": topology_bank_label(run.get("config", {})),
        "group_count": len(results.get("candidate_live", {}).get("group_reports", [])),
        "generation_seed_policy": results.get("candidate_live", {}).get("generation_seed_policy"),
        "source_baseline": {
            "is_source_baseline": True,
            "epoch": 0,
            "weight_selection": audit.get("source_weight_selection"),
            **source,
        },
        "candidate": candidate,
        "is_source_baseline": False,
        "per_field_relative_change": relative,
        "total_relative_change": candidate["mse"] / source["mse"] - 1
        if candidate["mse"] is not None and source["mse"] is not None and source["mse"] > 0
        else None,
        "worst_field_relative_change": max(relative.values(), default=None),
        "family_ratios": fidelity.get("family_source_normalized_scores", {}),
        "family_ratio_role": "within_definition_reporting_only"
        if run.get("config", {}).get("checkpointing", {}).get("selection_metric")
        == "topology_with_fidelity"
        else "selector_components",
        "family_mean_diagnostic": family_mean_diagnostic(
            fidelity.get("family_source_normalized_scores", {})
        ),
        "pareto_metric": family_mean_diagnostic(fidelity.get("family_source_normalized_scores", {}))
        if run.get("config", {}).get("checkpointing", {}).get("selection_metric")
        == "topology_with_fidelity"
        else _finite_number(fidelity.get("metric")),
        "metric": _finite_number(fidelity.get("metric")),
        "eligible": fidelity.get("eligible"),
        "selector_definition": fidelity.get("definition"),
        "groups": groups,
        "matched_native_monitor": compact_native(audit.get("matched_native_monitor", {})),
        "native_noise_check": None,
        "test_locked": True,
    }
    noise_reference = audit.get("native_noise_check", {})
    noise_path = (path.parent / noise_reference.get("path", "native_noise_check.json")).resolve()
    noise_path.relative_to(path.parent.resolve())
    if noise_path.is_file():
        noise = read_json(noise_path)
        if (
            noise.get("schema") != "phycoflow.upgrade_1002_r2_native_noise_check.v1"
            or noise.get("role")
            != "development_noise_check_only_no_selection_controller_or_fitting"
        ):
            raise ValueError(
                "native noise sidecar must remain development evidence, never controller calibration"
            )
        if noise_reference.get("sha256") and file_sha256(noise_path) != noise_reference["sha256"]:
            raise ValueError("native noise sidecar hash differs from its audit reference")
        for key, expected in (
            ("epoch", epoch),
            ("panel", panel),
            ("source_checkpoint_sha256", source_hash),
            ("checkpoint_sha256", record["checkpoint_sha256"]),
            ("sensor_manifest_sha256", record["sensor_manifest_sha256"]),
        ):
            if noise.get(key) != expected:
                raise ValueError("native noise sidecar identity differs from its audit: " + key)
        draws = compact_native(noise.get("matched_native_draws", {}))
        statistics = {}
        for role, item in draws.items():
            unique = {draw["seed"]: draw for draw in item.get("draws", [])}
            ratios = [
                draw["candidate"] / draw["source"]
                for draw in unique.values()
                if draw.get("source", 0) > 0
            ]
            statistics[role] = {
                "unique_seed_count": len(unique),
                "ratio_mean_over_unique_draws": float(np.mean(ratios)) if ratios else None,
                "ratio_sample_std_over_unique_draws": float(np.std(ratios, ddof=1))
                if len(ratios) > 1
                else None,
                "controller_calibration": False,
                "interpretation": "descriptive_seed_variation_only_no_significance_claim",
            }
        record["native_noise_check"] = {
            "path": str(noise_path),
            "sha256": file_sha256(noise_path),
            "role": "development_noise_check_only_no_selection_controller_or_fitting",
            "requested_seeds": noise.get("requested_seeds", []),
            "baseline_replay_precision": noise.get("baseline_replay_precision", []),
            "matched_native_draws": draws,
            "summary": statistics,
        }
    return record


def audit_markers(axis, audits, value, color, *, labels=True):
    for panel, (marker, label) in AUDIT_MARKERS.items():
        records = [
            (row["epoch"], value(row))
            for row in audits
            if row["panel"] == panel
            and row["split"] == "validation"
            and _finite_number(value(row)) is not None
        ]
        if records:
            x, y = zip(*records)
            axis.scatter(
                x,
                y,
                marker=marker,
                s=38,
                facecolors="none",
                edgecolors=color,
                linewidths=1.2,
                label=label if labels else None,
                zorder=5,
            )


def points(records, key):
    return [
        (record["epoch"], _finite_number(record.get(key)))
        for record in records
        if _finite_number(record.get(key)) is not None
    ]


def line(axis, values, label, color, *, rolling=False, linestyle="-"):
    if not values:
        return False
    x, y = map(np.asarray, zip(*values))
    axis.plot(
        x,
        y,
        color=color,
        linewidth=1,
        alpha=0.55 if rolling else 0.9,
        marker="o" if len(x) < 8 else None,
        markersize=3,
        label=label,
        linestyle=linestyle,
    )
    if rolling:
        medians = [np.median(y[(x > epoch - 10) & (x <= epoch)]) for epoch in x]
        axis.plot(x, medians, color=color, linewidth=1.8)
    return True


def epoch_axis(axis, latest_epoch):
    axis.set_xlabel("Epoch")
    axis.set_xlim(0, max(1, latest_epoch) * 1.025)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))


def finish(figure, axes, path, title, latest_epoch):
    for axis in np.asarray(axes).reshape(-1):
        epoch_axis(axis, latest_epoch)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=7, frameon=False)
        elif not axis.collections:
            axis.text(
                0.5,
                0.5,
                "Evidence not recorded",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color=".4",
            )
    figure.suptitle(title, fontsize=11)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def write_csv(path, rows):
    def flattened(row, prefix=""):
        result = {}
        for key, value in row.items():
            name = prefix + key
            if isinstance(value, dict):
                result.update(flattened(value, name + "/"))
            elif not isinstance(value, list):
                result[name] = value
        return result

    rows = [flattened(row) for row in rows]
    columns = sorted({key for row in rows for key in row if not isinstance(row[key], (dict, list))})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(run):
    epoch_series = {}
    keys = sorted(
        {
            key
            for row in run["epochs"]
            for key in row
            if key.startswith(
                (
                    "loss/",
                    "fidelity/",
                    "native_monitor/",
                    "coherence_family/",
                    "coherence_component/",
                )
            )
            or key in {"native_data_loss", "data_loss"}
        }
    )
    complete_epochs = [row for row in run["epochs"] if row["epoch_complete"]]
    for key in keys:
        epoch_series[key] = epoch_window_summary(complete_epochs, key)
        latest = max((row["epoch"] for row in complete_epochs), default=0)
        recent = [
            float(row[key])
            for row in complete_epochs
            if latest - 25 < row["epoch"] <= latest and _finite_number(row.get(key)) is not None
        ]
        epoch_series[key]["latest_25_epoch_median"] = float(np.median(recent)) if recent else None
        epoch_series[key]["latest_25_observed_epochs"] = len(recent)
    mature_records = [row for row in run["selection"] if row["metric"] is not None]
    mature = mature_checkpoint_summary(mature_records)
    mature_selection = [row for row in mature_records if row["epoch"] >= 100]
    family_flags = {
        family: any(
            _finite_number(row["family_ratios"].get(family)) is not None
            and row["family_ratios"][family] > 1.05
            for row in mature_selection
        )
        if mature_selection
        else None
        for family in FAMILIES
    }
    latest_complete = max((row["epoch"] for row in complete_epochs), default=0)
    return {
        "latest_epoch": run["latest_epoch"],
        "latest_completed_epoch": latest_complete,
        "minimum_horizon_observed": latest_complete >= 100,
        "primary_horizon_observed": latest_complete >= 150,
        "epoch_series": epoch_series,
        "controller_gradient_diagnostics": controller_diagnostics(run["epochs"]),
        "mature_checkpoint_evidence": mature,
        "effective_topology_bank": topology_bank_label(run["config"]),
        "family_mature_regression_above_5pct": family_flags,
        "family_regression_reporting_threshold": 0.05,
        "family_regression_reporting_threshold_is_a_training_budget": False,
        "development_audits": run.get("audits", []),
        "development_panel_evidence": {
            panel: {
                "observed": any(row["panel"] == panel for row in run.get("audits", [])),
                "mature_observed": any(
                    row["panel"] == panel and row["scientific_maturity"]
                    for row in run.get("audits", [])
                ),
                "recorded_evaluations": sum(row["panel"] == panel for row in run.get("audits", [])),
            }
            for panel in ("extended", "audit")
        },
        "family_definitions": {
            name: run["config"]
            .get("coherence", {})
            .get("families", {})
            .get(name, {})
            .get(
                "definition",
                run["config"]
                .get("coherence", {})
                .get("families", {})
                .get(name, {})
                .get("strategy", "legacy"),
            )
            for name in FAMILIES
        },
        "timing_scope": "recorded update durations plus separately recorded native monitoring; checkpoint/selection overhead is not inferred",
        "readiness_status": "requires_scientific_review",
        "final_test_evidence": "not_read_by_this_utility",
        "formal_run_launched": False,
    }


def render_campaign(runs, output):
    output.mkdir(parents=True, exist_ok=True)
    latest_epoch = max(
        max(run["latest_epoch"], max((row["epoch"] for row in run.get("audits", [])), default=0))
        for run in runs
    )
    # Ordinary native loss, frozen TRAIN, frozen validation have separate axes.
    figure, axes = plt.subplots(1, 4, figsize=(16, 4), layout="constrained")
    for index, run in enumerate(runs):
        color = COLORS[index % len(COLORS)]
        native_key = (
            "native_data_loss"
            if any("native_data_loss" in row for row in run["epochs"])
            else "loss/native"
        )
        line(axes[0], points(run["epochs"], native_key), run["label"], color, rolling=True)
        line(
            axes[0],
            points(run["epochs"], "fidelity/native/source_risk"),
            run["label"] + " matched source",
            color,
            linestyle="--",
        )
        for axis, role in zip(axes[1:3], ("train", "validation")):
            values = [
                (row["epoch"], row.get(role, {}).get("ratio"))
                for row in run["native"]
                if _finite_number(row.get(role, {}).get("ratio")) is not None
            ]
            line(axis, values, run["label"], color, rolling=True)
            axis.axhline(1, color=".3", linestyle="--", linewidth=0.8)
        line(
            axes[3],
            points(run["epochs"], "loss/fidelity_augmented"),
            run["label"],
            color,
            rolling=True,
        )
    axes[0].set_title("Stochastic TRAIN native objective")
    axes[0].set_ylabel("Unchanged native loss")
    for axis, role in zip(axes[1:3], ("TRAIN", "validation")):
        axis.set_title(f"Frozen {role}, common native draws")
        axis.set_ylabel("Candidate/source native ratio")
    axes[3].set_title("Augmented constraint penalty")
    axes[3].set_ylabel("Optimization quantity, distinct from data loss")
    finish(
        figure,
        axes,
        output / "native_data_history.pdf",
        "Native data loss · epoch means and trailing 10-epoch medians",
        latest_epoch,
    )

    with PdfPages(output / "coherence_history.pdf") as pdf:
        for run in runs:
            figure, axes = plt.subplots(4, 3, figsize=(13, 11), layout="constrained")
            for index, family in enumerate(FAMILIES):
                color = COLORS[index]
                raw = f"loss/coherence/{family}/raw"
                if not any(raw in row for row in run["epochs"]):
                    raw = f"coherence_family/{family}/raw"
                line(
                    axes[0, index],
                    points(run["epochs"], raw),
                    "TRAIN raw, " + topology_bank_label(run["config"], training=True)
                    if family == "topology"
                    else "TRAIN raw, sampled estimator",
                    color,
                    rolling=True,
                )
                line(
                    axes[1, index],
                    [
                        (row["epoch"], row["family_ratios"][family])
                        for row in run["selection"]
                        if _finite_number(row["family_ratios"].get(family)) is not None
                    ],
                    "Selection/source, " + topology_bank_label(run["config"])
                    if family == "topology"
                    else "Selection/source, groups of 32",
                    COLORS[3],
                )
                audit_markers(
                    axes[1, index],
                    run.get("audits", []),
                    lambda row, family=family: row["family_ratios"].get(family),
                    COLORS[3],
                )
                axes[0, index].set_title(f"{LABELS[family]} · {family.replace('_', ' ')}")
                axes[0, index].set_ylabel("Raw family TRAIN loss")
                axes[1, index].set_ylabel("Matched panel/source family ratio")
                axes[1, index].axhline(1, color=".4", linewidth=0.7, linestyle="--")
                calibrated = f"loss/coherence/{family}/calibrated"
                if not any(calibrated in row for row in run["epochs"]):
                    calibrated = f"coherence_family/{family}/weighted_contribution"
                line(
                    axes[2, index],
                    points(run["epochs"], calibrated),
                    "Calibrated weighted contribution",
                    color,
                    rolling=True,
                )
                axes[2, index].set_ylabel("Calibrated contribution (own TRAIN scale)")
                components = sorted(
                    {
                        key
                        for row in run["epochs"]
                        for key in row
                        if (
                            key.startswith(
                                (f"loss/coherence/{family}.", f"coherence_component/{family}/")
                            )
                        )
                        and not key.endswith("weighted_contribution")
                    }
                )
                canonical = [
                    key
                    for key in components
                    if key.startswith("coherence_component/") and key.endswith("/raw")
                ]
                if canonical:
                    components = canonical
                for component_index, key in enumerate(components):
                    line(
                        axes[3, index],
                        points(run["epochs"], key),
                        key.split(family, 1)[-1].strip("/.").removesuffix("/raw") or key,
                        COLORS[component_index % len(COLORS)],
                        rolling=True,
                        linestyle="--" if component_index >= len(COLORS) else "-",
                    )
                axes[3, index].set_ylabel("Raw component loss")
            for axis in axes.flat:
                epoch_axis(axis, latest_epoch)
                axis.grid(alpha=0.2)
                if axis.get_legend_handles_labels()[0]:
                    axis.legend(
                        fontsize=7,
                        frameon=False,
                        ncol=2 if len(axis.get_legend_handles_labels()[0]) > 6 else 1,
                    )
                else:
                    axis.text(0.5, 0.5, "Not recorded", transform=axis.transAxes, ha="center")
            figure.suptitle(
                f"{run['label']} · raw families/components; training C uses sampled lines",
                fontsize=10,
            )
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)

    with (
        PdfPages(output / "fidelity_history.pdf") as pdf,
        PdfPages(output / "controller_history.pdf") as controller_pdf,
    ):
        for run in runs:
            figure, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
            total_key = (
                "loss/endpoint_total"
                if any("loss/endpoint_total" in row for row in run["epochs"])
                else "fidelity/total/live_risk"
            )
            line(axes[0], points(run["epochs"], total_key), "total", "#202733", rolling=True)
            source_total = run["before"].get("mse_normalized", 0)
            if source_total > 0:
                line(
                    axes[1],
                    [
                        (row["epoch"], row["mse"] / source_total - 1)
                        for row in run["selection"]
                        if row["mse"] is not None
                    ],
                    "total",
                    "#202733",
                )
            audit_markers(
                axes[1], run.get("audits", []), lambda row: row["total_relative_change"], "#202733"
            )
            for index, field in enumerate(run["field_names"]):
                color = COLORS[index]
                line(
                    axes[0],
                    points(run["epochs"], f"loss/endpoint/{field}"),
                    field,
                    color,
                    rolling=True,
                )
                line(
                    axes[1],
                    [
                        (row["epoch"], row["per_field_relative_change"][field])
                        for row in run["selection"]
                        if field in row["per_field_relative_change"]
                    ],
                    field,
                    color,
                )
                audit_markers(
                    axes[1],
                    run.get("audits", []),
                    lambda row, field=field: row["per_field_relative_change"].get(field),
                    color,
                    labels=False,
                )
                audit_markers(
                    axes[2],
                    run.get("audits", []),
                    lambda row, field=field: row["candidate"]["decoded_relative_l2"].get(field),
                    color,
                    labels=index == 0,
                )
                line(
                    axes[2],
                    [
                        (row["epoch"], row["decoded_relative_l2"][field])
                        for row in run["selection"]
                        if _finite_number(row["decoded_relative_l2"].get(field)) is not None
                    ],
                    field,
                    color,
                )
            axes[0].set_title("TRAIN endpoint MSE, model units")
            axes[1].set_title("Matched-panel MSE relative to source")
            axes[1].axhline(0.05, color=".3", linestyle="--")
            axes[2].set_title("Matched-panel decoded relative L2")
            axes[0].set_ylabel("MSE")
            axes[1].set_ylabel("Fractional MSE change")
            axes[2].set_ylabel("Decoded relative L2")
            for axis in axes:
                epoch_axis(axis, latest_epoch)
                axis.grid(alpha=0.2)
                if axis.lines:
                    axis.legend(fontsize=7, frameon=False)
                else:
                    axis.text(0.5, 0.5, "Not recorded", transform=axis.transAxes, ha="center")
            figure.suptitle(run["label"], fontsize=11)
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            config = deepcopy(run["config"])
            config.setdefault("runtime", {})["plot_format"] = "pdf"
            adaptive = extract_adaptive_coherence_history(
                run["history"],
                [],
                before=run["before"],
                config=config,
                steps_per_epoch=run["manifest"]["steps_per_epoch"],
            )
            figures = build_adaptive_coherence_figures(adaptive, plt)
            figures["controller_state"].suptitle(
                run["label"] + " · constraint multipliers and violations", fontsize=11
            )
            for axis in figures["controller_state"].axes:
                if axis.get_xlabel() == "Epoch":
                    epoch_axis(axis, run["latest_epoch"])
            controller_pdf.savefig(figures["controller_state"], bbox_inches="tight")
            for value in figures.values():
                plt.close(value)
            gradient_page = controller_gradient_page(run)
            controller_pdf.savefig(gradient_page, bbox_inches="tight")
            plt.close(gradient_page)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for index, run in enumerate(runs):
        color = COLORS[index % len(COLORS)]
        native = [
            (row["epoch"], row.get("validation", {}).get("ratio"))
            for row in run["native"]
            if _finite_number(row.get("validation", {}).get("ratio")) is not None
        ]
        line(axes[0], native, run["label"], color)
        line(
            axes[1],
            [
                (row["epoch"], row["worst_field_relative_change"])
                for row in run["selection"]
                if row["worst_field_relative_change"] is not None
            ],
            run["label"],
            color,
        )
        audit_markers(
            axes[1],
            run.get("audits", []),
            lambda row: row["worst_field_relative_change"],
            color,
            labels=index == 0,
        )
    axes[0].set_ylabel("Frozen validation native/source")
    axes[0].axhline(1, color=".3", linestyle="--")
    axes[1].set_ylabel("Worst field MSE fractional increase")
    axes[1].axhline(0.05, color=".3", linestyle="--")
    finish(
        figure,
        axes,
        output / "optimizer_comparison.pdf",
        "Matched native and per-field fidelity · optimizer/control trajectories",
        latest_epoch,
    )

    figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    for index, run in enumerate(runs):
        for mature in (False, True):
            records = [
                row
                for row in run["selection"]
                if (row["epoch"] >= 100) == mature
                and row["worst_field_relative_change"] is not None
                and row["pareto_metric"] is not None
                and not row["is_source_baseline"]
            ]
            if records:
                axis.scatter(
                    [row["worst_field_relative_change"] for row in records],
                    [row["pareto_metric"] for row in records],
                    color=COLORS[index % len(COLORS)],
                    marker="o" if mature else "+",
                    s=24,
                    label=run["label"]
                    + (" mature" if mature else " early")
                    + (
                        " legacy diagnostic"
                        if run.get("selection_history") == "topology_validation"
                        else " selector"
                    ),
                )
        for panel, (marker, label) in AUDIT_MARKERS.items():
            records = [
                row
                for row in run.get("audits", [])
                if row["panel"] == panel
                and row["split"] == "validation"
                and row["pareto_metric"] is not None
                and row["worst_field_relative_change"] is not None
            ]
            if records:
                axis.scatter(
                    [row["worst_field_relative_change"] for row in records],
                    [row["pareto_metric"] for row in records],
                    edgecolors=COLORS[index % len(COLORS)],
                    facecolors="none",
                    marker=marker,
                    s=46,
                    label=run["label"] + " " + label,
                )
    axis.axvline(0.05, color=".3", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Worst per-field MSE fractional increase versus matched source")
    axis.set_ylabel("Within-arm mean A/B/C source ratio")
    axis.set_title(
        "Modern selector / legacy reporting diagnostic · definitions differ; no cross-definition ranking"
    )
    axis.grid(alpha=0.2)
    if axis.collections:
        axis.legend(fontsize=7, frameon=False)
    else:
        axis.text(
            0.5, 0.5, "No checkpoint evidence recorded", transform=axis.transAxes, ha="center"
        )
    figure.savefig(output / "fidelity_coherence_pareto.pdf", bbox_inches="tight")
    plt.close(figure)

    with PdfPages(output / "pressure_error_decomposition.pdf") as pdf:
        for run in runs:
            figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
            p_index = run["field_names"].index("p") if "p" in run["field_names"] else None
            if p_index is not None:
                for index, key in enumerate(("mse", "mean_offset_squared", "centered_mse")):
                    values = [
                        (row["epoch"], row["error_decomposition"][key][p_index])
                        for row in run["selection"]
                        if key in row["error_decomposition"]
                    ]
                    line(axes[0], values, key.replace("_", " "), COLORS[index])
                    source_value = run["before"].get("error_decomposition", {}).get(key, [])
                    if len(source_value) > p_index and source_value[p_index] > 0:
                        line(
                            axes[1],
                            [(epoch, value / source_value[p_index]) for epoch, value in values],
                            key.replace("_", " "),
                            COLORS[index],
                        )
            axes[0].set_ylabel("Pressure squared error, original model units")
            axes[1].set_ylabel("Candidate/source contribution ratio")
            for axis in axes:
                epoch_axis(axis, run["latest_epoch"])
                axis.grid(alpha=0.2)
                if axis.lines:
                    axis.legend(fontsize=8, frameon=False)
                else:
                    axis.text(
                        0.5,
                        0.5,
                        "Pressure decomposition not recorded",
                        transform=axis.transAxes,
                        ha="center",
                    )
            figure.suptitle(
                run["label"] + " · MSE = offset squared + centered MSE; no gauge removal",
                fontsize=10,
            )
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)

    summaries = {run["label"]: summarize(run) for run in runs}
    for run in runs:
        safe = run["label"].replace("/", "_")
        write_csv(output / f"{safe}_epoch_metrics.csv", run["epochs"])
        write_csv(output / f"{safe}_checkpoint_metrics.csv", run["selection"])
        write_csv(
            output / f"{safe}_controller_gradient_diagnostics.csv",
            controller_diagnostics(run["epochs"]),
        )
        audits = run.get("audits", [])
        write_csv(
            output / f"{safe}_audit_metrics.csv",
            [
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"groups", "native_noise_check"}
                }
                for row in audits
            ],
        )
        write_csv(
            output / f"{safe}_audit_groups.csv",
            [
                {"panel": row["panel"], "epoch": row["epoch"], **group}
                for row in audits
                for group in row["groups"]
            ],
        )
        noise_draws = []
        for row in audits:
            noise = row["native_noise_check"]
            if noise is not None:
                for role, item in noise["matched_native_draws"].items():
                    for draw in item.get("draws", []):
                        noise_draws.append(
                            {
                                "panel": row["panel"],
                                "epoch": row["epoch"],
                                "role": role,
                                "controller_calibration": False,
                                **draw,
                            }
                        )
        write_csv(output / f"{safe}_native_noise_draws.csv", noise_draws)
        (output / f"{safe}_audit_evidence.json").write_text(
            json.dumps(audits, indent=2, allow_nan=False) + "\n"
        )
    (output / "campaign_evidence.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=False) + "\n"
    )
    return summaries


def render_saved_fields(specifications, output):
    """Export native/query field arrays without rerunning an evaluator."""
    grouped = defaultdict(list)
    for label, path in specifications:
        with np.load(path, allow_pickle=False) as payload:
            data = {key: np.asarray(payload[key]) for key in payload.files}
        required = {
            "coordinates_raw",
            "ground_truth",
            "source_live",
            "candidate_live",
            "field_names",
            "epoch",
            "sample_ids",
        }
        if not required <= data.keys():
            raise ValueError("field arrays missing required source-grounded metadata: " + str(path))
        epoch = int(data["epoch"])
        names = [str(name) for name in data["field_names"]]
        gt = data["ground_truth"]
        if (
            gt.ndim != 3
            or gt.shape != data["source_live"].shape
            or gt.shape != data["candidate_live"].shape
        ):
            raise ValueError("saved fields must be aligned [samples,points,fields]")
        if len(names) != gt.shape[-1] or len(data["sample_ids"]) != gt.shape[0]:
            raise ValueError("saved field names/sample IDs disagree with numerical arrays")
        if data["coordinates_raw"].ndim == 2:
            data["coordinates_raw"] = np.broadcast_to(data["coordinates_raw"], (*gt.shape[:2], 2))
        if data["coordinates_raw"].shape != (*gt.shape[:2], 2):
            raise ValueError("raw coordinates must align with the saved field points")
        grouped[epoch].append((label, names, data))
    written = []
    for epoch, arrays in sorted(grouped.items()):
        path = output / f"fields_epoch_{epoch:03d}.pdf"
        with PdfPages(path) as pdf:
            for label, names, data in arrays:
                limits = [
                    (
                        float(
                            min(
                                data[key][..., i].min()
                                for key in ("ground_truth", "source_live", "candidate_live")
                            )
                        ),
                        float(
                            max(
                                data[key][..., i].max()
                                for key in ("ground_truth", "source_live", "candidate_live")
                            )
                        ),
                    )
                    for i in range(len(names))
                ]
                error_limits = [
                    max(
                        float(np.abs(data[key][..., i] - data["ground_truth"][..., i]).max())
                        for key in ("source_live", "candidate_live")
                    )
                    for i in range(len(names))
                ]
                for sample_index, sample_id in enumerate(data["sample_ids"]):
                    figure, axes = plt.subplots(
                        len(names),
                        5,
                        figsize=(14, max(3, len(names) * 2.5)),
                        squeeze=False,
                        layout="constrained",
                    )
                    coords = data["coordinates_raw"][sample_index]
                    for i, name in enumerate(names):
                        gt, base, child = [
                            data[key][sample_index, :, i]
                            for key in ("ground_truth", "source_live", "candidate_live")
                        ]
                        low, high = limits[i]
                        high = max(high, low + 1e-10)
                        error_limit = max(error_limits[i], 1e-12)
                        for j, (value, title) in enumerate(
                            zip(
                                (gt, base, child, base - gt, child - gt),
                                (
                                    "GT",
                                    "Source LIVE",
                                    "Candidate LIVE",
                                    "Source − GT",
                                    "Candidate − GT",
                                ),
                            )
                        ):
                            axis = axes[i, j]
                            levels = (
                                np.linspace(-error_limit, error_limit, 25)
                                if j >= 3
                                else np.linspace(low, high, 25)
                            )
                            contour = axis.tricontourf(
                                coords[:, 0],
                                coords[:, 1],
                                value,
                                levels=levels,
                                cmap="RdBu_r" if j >= 3 else "viridis",
                            )
                            if name == "T" and j < 3 and "observation_coordinates_raw" in data:
                                mask = data["observation_valid_mask"][sample_index] & (
                                    data["observation_field_ids"][sample_index] == i
                                )
                                sensors = data["observation_coordinates_raw"][sample_index][mask]
                                axis.scatter(
                                    sensors[:, 0],
                                    sensors[:, 1],
                                    s=4,
                                    color="black",
                                    edgecolors="white",
                                    linewidths=0.15,
                                )
                            axis.set_aspect("equal")
                            axis.set_title(name + " · " + title, fontsize=8)
                            axis.tick_params(labelsize=7)
                            figure.colorbar(contour, ax=axis, shrink=0.7)
                    native = bool(data.get("full_native_grid", False))
                    figure.suptitle(
                        f"{label} · epoch {epoch} · {sample_id} · {'native' if native else 'saved query'} fields\nRaw coordinates; decoded dataset values; units unspecified; common field/error scales across saved snapshots",
                        fontsize=10,
                    )
                    if "observation_coordinates_raw" in data:
                        figure.text(
                            0.5,
                            0.002,
                            "Black markers: actual observed T sensors; other fields have no sensor overlay",
                            ha="center",
                            fontsize=8,
                        )
                    pdf.savefig(figure, bbox_inches="tight")
                    plt.close(figure)
        written.append(path.name)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="LABEL=RUN_DIRECTORY")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--audit",
        action="append",
        default=[],
        help="RUN_LABEL=SAVED_AUDIT_JSON (development panels only)",
    )
    parser.add_argument(
        "--fields-array", action="append", default=[], help="LABEL=SAVED_NPZ (no inference)"
    )
    args = parser.parse_args(argv)
    repository = Path(__file__).resolve().parents[2]
    test_root = (repository / "cases/turbulent_combustion/runs/Test_1002").resolve()
    args.output.resolve().relative_to(test_root)
    runs, input_hashes = [], {}
    for argument in args.run:
        label, value = argument.split("=", 1)
        path = Path(value).resolve()
        path.relative_to(test_root)
        if not label or any(run["label"] == label for run in runs):
            raise ValueError("run labels must be nonempty and unique")
        files = [
            path / "run_manifest.json",
            path / "resolved_config.yaml",
            *sorted((path / "metrics").glob("*.jsonl")),
            *sorted((path / "evaluation").glob("*.json")),
        ]
        input_hashes.update({str(file): file_sha256(file) for file in files if file.is_file()})
        runs.append(read_run(label, path))
    for argument in args.audit:
        label, value = argument.split("=", 1)
        run = next((item for item in runs if item["label"] == label), None)
        if run is None:
            raise ValueError("audit label must identify a declared --run label")
        path = Path(value).resolve()
        path.relative_to(test_root)
        input_hashes[str(path)] = file_sha256(path)
        item = read_audit(run, path)
        if any(row["path"] == item["path"] for row in run["audits"]):
            raise ValueError("duplicate audit input")
        run["audits"].append(item)
        if item["native_noise_check"] is not None:
            noise = item["native_noise_check"]
            input_hashes[noise["path"]] = noise["sha256"]
        run["field_names"] = list(
            dict.fromkeys([*run["field_names"], *item["source_baseline"]["per_field_mse"]])
        )
    render_campaign(runs, args.output)
    field_inputs = []
    for argument in args.fields_array:
        label, value = argument.split("=", 1)
        path = Path(value).resolve()
        path.relative_to(test_root)
        input_hashes[str(path)] = file_sha256(path)
        field_inputs.append((label, path))
    fields = render_saved_fields(field_inputs, args.output)
    for path, digest in input_hashes.items():
        if file_sha256(path) != digest:
            raise RuntimeError(
                "input changed during report; rerun on a frozen numerical snapshot: " + path
            )
    manifest = {
        "schema": "phycoflow.upgrade_1002_r2_campaign_report.v1",
        "inputs": input_hashes,
        "uses_test_data": False,
        "runs_inference": False,
        "scientific_readiness_automatically_certified": False,
        "outputs": {
            path.name: file_sha256(path)
            for path in args.output.iterdir()
            if path.is_file() and path.name != "report_manifest.json"
        },
        "field_figures": fields,
        "missing_field_figures": None
        if fields
        else "Supply saved native field arrays; this utility never generates predictions",
    }
    (args.output / "report_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"report": str(args.output.resolve()), "runs": len(runs)}))


if __name__ == "__main__":
    main()
