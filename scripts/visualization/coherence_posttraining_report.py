"""Create CPU-only diagnostic figures for a coherence post-training run.

The report only plots values present in the run artifacts. Missing or malformed
optional inputs are recorded in ``coherence_report.json`` rather than replaced
with guessed values.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILIES = ("global_distribution", "cross_spectrum", "topology")
FAMILY_LABELS = {
    "global_distribution": "A: global",
    "cross_spectrum": "B: cross",
    "topology": "C: topology",
}


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "missing"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable: {exc}"
    if not isinstance(value, dict):
        return None, "root is not an object"
    return value, None


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not path.is_file():
        return [], "missing"
    rows: list[dict[str, Any]] = []
    try:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                return rows, f"line {number} is not an object"
            rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        return rows, f"unreadable: {exc}"
    return rows, None


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_numeric(item, name))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        numbers = [_finite(item) for item in value]
        clean = [item for item in numbers if item is not None]
        if clean and len(clean) == len(value):
            result[f"{prefix}.mean"] = sum(clean) / len(clean)
            result[f"{prefix}.min"] = min(clean)
            result[f"{prefix}.max"] = max(clean)
    else:
        number = _finite(value)
        if number is not None and prefix:
            result[prefix] = number
    return result


def _family_payload(payload: Mapping[str, Any] | None, family: str) -> Mapping[str, Any]:
    if not payload:
        return {}
    coherence = payload.get("coherence", {})
    if not isinstance(coherence, Mapping):
        return {}
    families = coherence.get("families", {})
    return families.get(family, {}) if isinstance(families, Mapping) else {}


def parse_run_payload(run_dir: str | Path) -> dict[str, Any]:
    """Read standard run payloads and return a plot-ready inventory."""
    root = Path(run_dir)
    before, before_error = _read_json(root / "evaluation" / "before.json")
    after, after_error = _read_json(root / "evaluation" / "after.json")
    history, history_error = _read_jsonl(root / "metrics" / "history.jsonl")
    calibration = None
    calibration_path = None
    calibration_error = "missing"
    for candidate in (
        root / "artifacts" / "coherence_calibration.json",
        root / "metrics" / "coherence_calibration.json",
        root / "coherence_calibration.json",
    ):
        value, error = _read_json(candidate)
        if error != "missing":
            calibration, calibration_path, calibration_error = value, candidate, error
            break

    families: dict[str, Any] = {}
    for family in FAMILIES:
        family_before = _family_payload(before, family)
        family_after = _family_payload(after, family)
        before_total = _finite(family_before.get("total"))
        after_total = _finite(family_after.get("total"))
        families[family] = {
            "present": bool(family_before or family_after),
            "before_total": before_total,
            "after_total": after_total,
            "source_normalized": (
                after_total / before_total
                if before_total is not None and before_total != 0 and after_total is not None
                else None
            ),
            "before_weighted": _finite(family_before.get("weighted_total")),
            "after_weighted": _finite(family_after.get("weighted_total")),
            "before_components": _flatten_numeric(family_before.get("components", {})),
            "after_components": _flatten_numeric(family_after.get("components", {})),
            "before_diagnostics": _flatten_numeric(family_before.get("diagnostics", {})),
            "after_diagnostics": _flatten_numeric(family_after.get("diagnostics", {})),
        }

    return {
        "run_dir": str(root.resolve()),
        "inputs": {
            "evaluation_before": {"path": "evaluation/before.json", "error": before_error},
            "evaluation_after": {"path": "evaluation/after.json", "error": after_error},
            "history": {"path": "metrics/history.jsonl", "error": history_error, "rows": len(history)},
            "calibration": {
                "path": None if calibration_path is None else str(calibration_path.relative_to(root)),
                "error": calibration_error,
            },
        },
        "evaluation": {"before": before, "after": after},
        "history": history,
        "calibration": calibration,
        "families": families,
    }


def _empty(axis: Any, text: str) -> None:
    axis.text(0.5, 0.5, text, ha="center", va="center", transform=axis.transAxes)
    axis.set_xticks([])
    axis.set_yticks([])


def _bar_before_after(axis: Any, labels: list[str], before: list[float], after: list[float]) -> None:
    positions = list(range(len(labels)))
    axis.bar([position - 0.18 for position in positions], before, 0.36, label="source")
    axis.bar([position + 0.18 for position in positions], after, 0.36, label="post")
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.legend(fontsize=8)


def _summary_figure(parsed: Mapping[str, Any], output: Path) -> None:
    figure, axes = plt.subplots(4, 1, figsize=(10, 15), constrained_layout=True)
    evaluation = parsed["evaluation"]
    before = evaluation["before"] or {}
    after = evaluation["after"] or {}

    fidelity_keys = ("mse_normalized", "mean_relative_l2", "worst_field_relative_l2")
    fidelity = [
        (key, _finite(before.get(key)), _finite(after.get(key))) for key in fidelity_keys
    ]
    fidelity = [row for row in fidelity if row[1] is not None and row[2] is not None]
    axes[0].set_title("Fidelity vs source")
    if fidelity:
        _bar_before_after(
            axes[0], [row[0] for row in fidelity], [row[1] for row in fidelity], [row[2] for row in fidelity]
        )
        axes[0].set_yscale("log")
    else:
        _empty(axes[0], "No comparable fidelity metrics")

    axes[1].set_title("Source-normalized coherence family metrics (post / source)")
    ratios = [(name, data["source_normalized"]) for name, data in parsed["families"].items()]
    ratios = [(name, value) for name, value in ratios if value is not None]
    if ratios:
        axes[1].bar([FAMILY_LABELS[name] for name, _ in ratios], [value for _, value in ratios])
        axes[1].axhline(1.0, color="black", linewidth=1, linestyle="--")
    else:
        _empty(axes[1], "No source-normalizable family totals")

    axes[2].set_title("Weighted A/B/C family contributions")
    contributions = [
        (name, data["before_weighted"], data["after_weighted"])
        for name, data in parsed["families"].items()
        if data["before_weighted"] is not None and data["after_weighted"] is not None
    ]
    if contributions:
        _bar_before_after(
            axes[2],
            [FAMILY_LABELS[row[0]] for row in contributions],
            [row[1] for row in contributions],
            [row[2] for row in contributions],
        )
    else:
        _empty(axes[2], "No weighted family contributions")

    axes[3].set_title("Gradient / conflict diagnostics")
    history = parsed["history"]
    keys = (
        "data_grad_norm",
        "coherence_grad_norm",
        "gradient_cosine",
        "gradient_conflict",
        "conflict_cosine",
    )
    plotted = False
    for key in keys:
        points = [(index + 1, _finite(row.get(key))) for index, row in enumerate(history)]
        points = [(x, y) for x, y in points if y is not None]
        if points:
            axes[3].plot([x for x, _ in points], [y for _, y in points], label=key)
            plotted = True
    calibration = _flatten_numeric(parsed.get("calibration") or {})
    calibration = {
        key: value
        for key, value in calibration.items()
        if any(token in key.lower() for token in ("grad", "norm", "cos", "conflict", "scale"))
    }
    if calibration and not plotted:
        axes[3].bar(list(calibration), list(calibration.values()), color="tab:purple")
        axes[3].tick_params(axis="x", labelrotation=20)
        plotted = True
    if plotted:
        axes[3].axhline(0.0, color="black", linewidth=0.8)
        axes[3].set_xlabel("history row")
        axes[3].legend(fontsize=8, ncol=2)
    else:
        _empty(axes[3], "No gradient/conflict diagnostics")
    figure.suptitle(f"Coherence post-training report: {Path(parsed['run_dir']).name}")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _family_figure(family: str, data: Mapping[str, Any], output: Path) -> bool:
    before = {**data["before_components"], **data["before_diagnostics"]}
    after = {**data["after_components"], **data["after_diagnostics"]}
    keys = sorted(set(before) & set(after))
    if not keys:
        return False
    # Keep reports readable while retaining the complete numeric inventory in JSON.
    keys = keys[:30]
    figure, axis = plt.subplots(figsize=(max(9, len(keys) * 0.42), 5), constrained_layout=True)
    _bar_before_after(axis, keys, [before[key] for key in keys], [after[key] for key in keys])
    axis.set_title(f"{family.replace('_', ' ').title()} component / diagnostic values")
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return True


def generate_report(run_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Generate summary/family figures and a machine-readable inventory."""
    root = Path(run_dir)
    destination = Path(output_dir) if output_dir is not None else root / "visualization"
    destination.mkdir(parents=True, exist_ok=True)
    parsed = parse_run_payload(root)
    generated = ["coherence_summary.png"]
    _summary_figure(parsed, destination / generated[0])
    for family, data in parsed["families"].items():
        filename = f"{family}_diagnostics.png"
        if _family_figure(family, data, destination / filename):
            generated.append(filename)
    missing = [name for name, data in parsed["families"].items() if not data["present"]]
    report = {
        **parsed,
        "output_dir": str(destination.resolve()),
        "generated": generated,
        "missing_families": missing,
        "notes": [
            "Figures contain only finite numeric values found in run artifacts.",
            "Sequence diagnostics are summarized by mean/min/max in the JSON inventory.",
        ],
    }
    (destination / "coherence_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = generate_report(args.run_dir, args.output_dir)
    print(Path(report["output_dir"]) / "coherence_report.json")


if __name__ == "__main__":
    main()
