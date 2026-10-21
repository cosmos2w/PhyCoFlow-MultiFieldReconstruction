"""Summarize committed post-training histories without loading models or datasets.

Ported from the local comparison's metric-specific finite-count aggregation.
All results describe saved training telemetry, not held-out checkpoint quality.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

METRICS = (
    "update_seconds", "data_loss", "coherence_loss", "scientific_coherence",
    "coherence_family/global_distribution/raw", "coherence_family/cross_spectrum/raw",
    "coherence_family/topology/raw",
)


def _read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_run(label: str, run: Path) -> dict[str, Any]:
    run = run.resolve()
    paths = [run / name for name in (
        "run_manifest.json", "status.json", "resolved_config.yaml", "metrics/history.jsonl",
    )]
    before = {str(path): _sha256(path) for path in paths}
    manifest, status = (_read_json(path) for path in paths[:2])
    committed = manifest.get("last_checkpoint_step", status.get("global_step"))
    if isinstance(committed, bool) or not isinstance(committed, int) or committed < 0:
        raise ValueError(f"missing valid committed update cursor: {run}")
    # Snapshot bytes before parsing and verify again after reading. Reject a
    # changing input rather than silently publishing mixed live-run telemetry.
    totals: dict[str, list[float | int]] = {name: [0.0, 0] for name in METRICS}
    epochs: dict[int, dict[str, list[float | int]]] = defaultdict(
        lambda: {name: [0.0, 0] for name in METRICS}
    )
    last_metric_step = {name: 0 for name in METRICS}
    ignored = 0
    with paths[-1].open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            step, epoch = row.get("step"), row.get("epoch")
            if isinstance(step, bool) or not isinstance(step, int) or step < 1:
                raise ValueError(f"invalid history step at {run}:{number}")
            if step > committed:
                ignored += 1
                continue
            if (
                isinstance(epoch, bool) or not isinstance(epoch, (int, float))
                or not math.isfinite(epoch) or epoch < 1 or int(epoch) != epoch
            ):
                raise ValueError(f"invalid reporting epoch at {run}:{number}")
            counts = row.get("finite_counts", {})
            batches = row.get("batches")
            if batches is not None and (
                isinstance(batches, bool) or not isinstance(batches, int) or batches < 1
            ):
                raise ValueError(f"invalid observed batch count at {run}:{number}")
            for metric in METRICS:
                count = counts.get(metric, 0)
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError(f"invalid finite count for {metric} at {run}:{number}")
                if not count:
                    continue
                if batches is not None and count > batches:
                    raise ValueError(f"finite count exceeds batches for {metric} at {run}:{number}")
                value = row.get(metric)
                if (
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(f"nonfinite counted metric {metric} at {run}:{number}")
                # Epoch summaries may include 1+37 rows after a resume. Counts
                # are per metric, not the epoch size or the number of JSON rows.
                if count > step or step - count < last_metric_step[metric]:
                    raise ValueError(f"overlapping metric coverage for {metric} at {run}:{number}")
                last_metric_step[metric] = step
                totals[metric][0] += value * count
                totals[metric][1] += count
                epochs[int(epoch)][metric][0] += value * count
                epochs[int(epoch)][metric][1] += count
    if before != {str(path): _sha256(path) for path in paths}:
        raise ValueError(f"run artifacts changed during summary; retry from a saved snapshot: {run}")
    metrics = {
        name: {"count": count, "mean": total / count if count else None}
        for name, (total, count) in totals.items()
    }
    warnings = []
    timing_count = metrics["update_seconds"]["count"]
    if timing_count != committed:
        warnings.append(f"update timing covers {timing_count} of {committed} committed updates")
    if ignored:
        warnings.append(f"ignored {ignored} history rows beyond the committed cursor")
    return {
        "label": label, "run": str(run), "status": status.get("status"),
        "committed_updates": committed,
        "source_hashes": manifest.get("source_hashes"),
        "config_sha256": manifest.get("config_sha256"),
        "input_sha256": before, "metrics": metrics,
        "timed_training_seconds": totals["update_seconds"][0] if timing_count else None,
        "warnings": warnings,
        "epochs": [
            {"epoch": epoch, "metric": metric, "count": count, "mean": total / count}
            for epoch, values in sorted(epochs.items())
            for metric, (total, count) in values.items() if count
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True, help="new local output directory")
    args = parser.parse_args()
    runs: dict[str, Path] = {}
    for item in args.run:
        label, separator, value = item.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label) or not value:
            parser.error("--run must be NAME=PATH with a simple alphanumeric name")
        path = Path(value).resolve()
        if label in runs or path in runs.values():
            parser.error("run names and directories must be distinct")
        runs[label] = path
    output = args.output.resolve()
    if any(output == run or output.is_relative_to(run) for run in runs.values()):
        parser.error("output must be outside every input run")
    if output.exists():
        parser.error("output already exists; select a new directory to preserve earlier results")
    summaries = [summarize_run(label, run) for label, run in runs.items()]
    output.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "scope": "committed training telemetry; no checkpoint or validation assessment",
        "timing_scope": "logged update work; excludes validation, checkpoints, startup and pauses",
        "runs": summaries,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    columns = ["run", "status", "committed_updates", "timed_training_seconds"]
    columns += [f"{metric}_{key}" for metric in METRICS for key in ("mean", "count")]
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in summaries:
            writer.writerow({
                "run": result["label"], "status": result["status"],
                "committed_updates": result["committed_updates"],
                "timed_training_seconds": result["timed_training_seconds"],
                **{f"{name}_{key}": values[key]
                   for name, values in result["metrics"].items() for key in ("mean", "count")},
            })
    with (output / "epoch_history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run", "epoch", "metric", "count", "mean"])
        writer.writeheader()
        for result in summaries:
            writer.writerows({"run": result["label"], **row} for row in result["epochs"])
    lines = [
        "# Saved post-training telemetry", "",
        "Means use each metric's finite counts through the saved recovery cursor. Missing counts are unmeasured. Timing excludes evaluation, checkpointing, startup, pauses and unlogged replay. These summaries do not assess selected checkpoints, validation fidelity, inference speed or method superiority.",
        "", "| Run | Saved status | Committed updates | Timed updates | Mean s/update |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in summaries:
        timing = result["metrics"]["update_seconds"]
        mean = "unmeasured" if timing["mean"] is None else f"{timing['mean']:.6f}"
        lines.append(
            f"| {result['label']} | {result['status']} | {result['committed_updates']} "
            f"| {timing['count']} | {mean} |"
        )
    for result in summaries:
        for warning in result["warnings"]:
            lines.extend(["", f"{result['label']}: {warning}."])
    lines.extend([
        "", "Before comparing runs, verify matching SOURCE hashes, exposure, data/sensors, family definitions, calibration and execution windows. Identical SOURCE hashes alone do not establish a matched ablation. Exact input hashes and metric coverage are in summary.json.",
    ])
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(output / "summary.md")


if __name__ == "__main__":
    main()
