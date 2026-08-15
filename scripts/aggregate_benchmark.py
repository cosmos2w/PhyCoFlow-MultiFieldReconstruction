#!/usr/bin/env python
"""Freeze traceable integration/benchmark rows from common evaluation reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _statistics(values: list[float], trajectories: list[str]) -> dict[str, Any]:
    count = len(values)
    spread = stdev(values) if count > 1 else None
    return {
        "trajectory_count": len(set(trajectories)),
        "sample_count": count,
        "mean": mean(values),
        "standard_deviation": spread,
        "standard_error": None if spread is None else spread / math.sqrt(count),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = yaml.safe_load(args.suite.read_text())
    root = args.suite.resolve().parents[2]
    frozen_manifest = json.loads((root / suite["frozen_sensor_manifest"]).read_text())
    frozen_digest = frozen_manifest.pop("manifest_sha256")
    computed_manifest_digest = hashlib.sha256(
        json.dumps(frozen_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if frozen_digest != computed_manifest_digest:
        raise ValueError("frozen sensor manifest checksum is invalid")
    rows = []
    shared_sensor_hash = None
    for entry in suite["entries"]:
        report_path = root / entry["report"]
        report = json.loads(report_path.read_text())
        trace = report["trace"]
        resolved_config_path = root / entry["resolved_config"]
        if _sha256(resolved_config_path) != trace["resolved_config_sha256"]:
            raise ValueError(f"frozen resolved config hash mismatch for {entry['id']}")
        sensor_hash = trace["sensor_manifest_sha256"]
        if entry.get("matched_sensor_group"):
            if shared_sensor_hash is None:
                shared_sensor_hash = sensor_hash
            elif sensor_hash != shared_sensor_hash:
                raise ValueError("matched benchmark entries use different sensor manifests")
        trajectories = [sample.rsplit(":", 1)[0] for sample in trace["sample_ids"]]
        coherence = None
        if entry.get("coherence_report"):
            coherence_payload = json.loads((root / entry["coherence_report"]).read_text())
            coherence = coherence_payload.get("coherence")
        rows.append(
            {
                "id": entry["id"],
                "stage": entry["stage"],
                "formal_benchmark": bool(entry.get("formal_benchmark", True)),
                "budget": entry["budget"],
                "reconstruction": {
                    "mse_normalized": _statistics(
                        [float(report["mse_normalized"])], trajectories
                    ),
                    "per_field_mse_normalized": report["per_field_mse_normalized"],
                },
                "coherence": coherence
                or {"available": False, "reason": "not evaluated for this ablation"},
                "physics": report.get("case_diagnostics", {}),
                "uncertainty": report["uncertainty"],
                "compute": report["compute"],
                "trace": {
                    **trace,
                    "report": entry["report"],
                    "report_sha256": _sha256(report_path),
                    "frozen_resolved_config": entry["resolved_config"],
                },
            }
        )
    output = {
        "release": suite["release"],
        "scope": suite["scope"],
        "claims": suite["claims"],
        "matched_sensor_manifest_sha256": shared_sensor_hash,
        "frozen_sensor_manifest": suite["frozen_sensor_manifest"],
        "dataset_full_sha256": suite["dataset_full_sha256"],
        "code_snapshot": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "working_tree_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=root, text=True
                ).strip()
            ),
            "file_sha256": {
                path: _sha256(root / path) for path in suite["code_files"]
            },
        },
        "entries": rows,
    }
    if shared_sensor_hash != frozen_digest:
        raise ValueError("frozen sensor manifest does not match evaluated rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(output, sort_keys=False), encoding="utf-8")
    markdown = [
        f"# {suite['release']}",
        "",
        suite["scope"],
        "",
        "| Ablation | Stage | Updates | MSE | PDE-u | PDE-v | Time (ms) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        physics = row["physics"]
        markdown.append(
            "| {id} | {stage} | {updates} | {mse:.6g} | {pde_u:.6g} | "
            "{pde_v:.6g} | {milliseconds:.3f} |".format(
                id=row["id"],
                stage=row["stage"],
                updates=row["budget"]["optimizer_updates"],
                mse=row["reconstruction"]["mse_normalized"]["mean"],
                pde_u=float(physics.get("relative_pde_residual_u", float("nan"))),
                pde_v=float(physics.get("relative_pde_residual_v", float("nan"))),
                milliseconds=1000.0 * float(row["compute"]["seconds_per_sample"]),
            )
        )
    markdown.extend(
        [
            "",
            (
                "All rows use one validation trajectory and one optimizer update; standard errors "
                "are therefore intentionally unavailable. This table proves pipeline comparability "
                "only."
            ),
        ]
    )
    args.output.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
