"""Render the Round 6 fidelity record from saved histories, without model evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path


def _rows(path):
    if not path.is_file():
        return []
    result = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A writer may be completing the last line at a checkpoint.
                    continue
                if isinstance(row, dict):
                    result.append(row)
    return result


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def render_coverage_history(run_dir, source_validation_metrics=None):
    """Write checkpoint_fidelity PDF/PNG from run-owned numeric records.

    Training ratios use each frozen TRAIN bound; validation ratios use the
    matched validation SOURCE risk. Missing validation points remain absent.
    """
    root = Path(run_dir)
    ref_path = root / "artifacts" / "fidelity_budget_reference.json"
    if not ref_path.is_file():
        return None
    reference = json.loads(ref_path.read_text(encoding="utf-8"))["reference"]
    names, bounds = reference["field_names"], reference["upper_bounds"]
    if not names or len(names) != len(bounds) or len(set(names)) != len(names):
        raise ValueError("invalid frozen field bounds")
    if any(not _finite(bound) or bound <= 0 for bound in bounds):
        raise ValueError("frozen TRAIN field bounds must be finite and positive")
    training = _rows(root / "logs" / "full_updates.jsonl")
    validation = _rows(root / "logs" / "loss_aligned_validation.jsonl")
    if not training and not validation:
        return None
    source_risks = (source_validation_metrics or {}).get("endpoint_mse_model_units", {})

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.5), sharex=True, constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for index, (name, bound) in enumerate(zip(names, bounds)):
        color = colors(index % 10)
        train_points = [
            (row["epoch"], row["field_risk"][name] / bound)
            for row in training
            if _finite(row.get("epoch"))
            and isinstance(row.get("field_risk"), dict)
            and _finite(row["field_risk"].get(name))
        ]
        if train_points:
            axes[0].plot(*zip(*train_points), color=color, linewidth=0.9, label=name)
        val_points = []
        for row in validation:
            if not _finite(row.get("epoch")):
                continue
            ratios = row.get("source_ratios") or {}
            value = ratios.get(name)
            if value is None:
                metrics = row.get("metrics") or {}
                risks = metrics.get("endpoint_mse_model_units") or {}
                source = source_risks.get(name)
                if _finite(risks.get(name)) and _finite(source) and source > 0:
                    value = risks[name] / source
            if _finite(value):
                val_points.append((row["epoch"], value))
        if val_points:
            axes[1].plot(*zip(*val_points), marker="o", markersize=3.5, color=color, linewidth=1)
        dual_points = []
        for row in training:
            values = row.get("multipliers_after")
            value = (
                values.get(name)
                if isinstance(values, dict)
                else (
                    values[index]
                    if isinstance(values, list) and len(values) == len(names)
                    else None
                )
            )
            if _finite(row.get("epoch")) and _finite(value):
                dual_points.append((row["epoch"], value))
        if dual_points:
            axes[2].plot(*zip(*dual_points), color=color, linewidth=0.9)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].axhline(1.10, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("TRAIN risk / TRAIN bound")
    axes[1].set_ylabel("VALIDATION risk /\nmatched SOURCE risk")
    axes[2].set_ylabel("Dual multiplier λ")
    axes[2].set_xlabel("Reporting epoch")
    axes[0].set_title("Sampled TRAIN endpoint risk (model-unit MSE)", loc="left", fontsize=10)
    axes[1].set_title(
        "Fixed-panel VALIDATION endpoint risk (model-unit MSE); guard 1.10", loc="left", fontsize=10
    )
    axes[2].set_title("Fidelity dual activity", loc="left", fontsize=10)
    if not axes[1].lines or len(axes[1].lines) == 1:
        axes[1].text(
            0.5,
            0.5,
            "No matched validation yet",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
            color="0.4",
        )
    for axis in axes:
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="outside upper center", ncols=min(len(names), 5), frameon=False
        )
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    paths = [logs / "checkpoint_fidelity.pdf", logs / "checkpoint_fidelity.png"]
    try:
        for path in paths:
            fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)
    return tuple(paths)
