"""Render publication-style training figures from an immutable JSONL snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.training.coherence_history import render_coherence_history
from phycoflow_reconstruction.training.fidelity_history import render_fidelity_history
from phycoflow_reconstruction.training.monitoring import TrainingMonitor

LOSS_KEYS = {"total", "data_loss", "coherence_loss", "physics_loss", "validation_loss"}
OPTIMIZATION_KEYS = {
    "data_grad_norm",
    "coherence_grad_norm",
    "combined_grad_norm",
    "gradient_cosine",
    "gradient_conflict_fraction",
}


def _complete_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    complete_payload = payload[: payload.rfind(b"\n") + 1]
    rows = [json.loads(line) for line in complete_payload.splitlines() if line.strip()]
    return rows, digest


def _compact_training_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = {"step", "epoch", "epoch_complete"}
    keys.update(LOSS_KEYS)
    keys.update(OPTIMIZATION_KEYS)
    keys.update(
        key
        for key in row
        if key.startswith("coherence_family/")
        and key.endswith("/weighted_contribution")
    )
    keys.update(
        key
        for key in row
        if key.startswith("coherence_component/")
        and key.endswith(("/raw", "/weighted_contribution"))
    )
    return {key: row[key] for key in sorted(keys) if key in row}


def _snapshot(source: Path, snapshot: Path, through_epoch: int | None) -> dict[str, Any]:
    metrics_source = source / "metrics"
    metrics_snapshot = snapshot / "metrics"
    metrics_snapshot.mkdir(parents=True, exist_ok=True)
    history, history_hash = _complete_jsonl(metrics_source / "history.jsonl")
    validation, validation_hash = _complete_jsonl(metrics_source / "validation_history.jsonl")
    topology, topology_hash = _complete_jsonl(metrics_source / "topology_validation.jsonl")

    if through_epoch is not None:
        history = [row for row in history if float(row.get("epoch", 0)) <= through_epoch]
    steps_per_epoch = int(json.loads((source / "run_manifest.json").read_text()).get("steps_per_epoch", 1))
    max_step = max((int(row.get("step", 0)) for row in history), default=0)
    epoch_by_step = {int(row.get("step", 0)): float(row.get("epoch", 0)) for row in history}
    validation = [row for row in validation if int(row.get("step", 0)) <= max_step]
    topology = [row for row in topology if int(row.get("step", 0)) <= max_step]

    compact_history = [_compact_training_row(row) for row in history]
    compact_validation = [
        {key: row[key] for key in ("step", "epoch", "validation_loss") if key in row}
        for row in validation
    ]
    compact_topology = [
        {
            key: row[key]
            for key in (
                "step",
                "metric",
                "eligible",
                "failed_fidelity_fields",
                "relative_mse_increase",
            )
            if key in row
        }
        for row in topology
    ]
    for filename, rows in (
        ("history.jsonl", compact_history),
        ("validation_history.jsonl", compact_validation),
        ("topology_validation.jsonl", compact_topology),
    ):
        (metrics_snapshot / filename).write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    config_bytes = (source / "resolved_config.yaml").read_bytes()
    (snapshot / "resolved_config.yaml").write_bytes(config_bytes)
    manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    compact_manifest = {
        "run_id": manifest.get("run_id", source.name),
        "steps_per_epoch": steps_per_epoch,
        "source_history_sha256": history_hash,
        "source_validation_history_sha256": validation_hash,
        "source_topology_validation_sha256": topology_hash,
        "resolved_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "through_epoch": through_epoch,
        "last_included_epoch": max((float(row.get("epoch", 0)) for row in history), default=0.0),
        "last_included_step": max_step,
        "source_rows": {
            "history": len(history),
            "validation_history": len(validation),
            "topology_validation": len(topology),
        },
        "snapshot_rows": {
            "history": len(compact_history),
            "validation_history": len(compact_validation),
            "topology_validation": len(compact_topology),
        },
        "snapshot_created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (snapshot / "run_manifest.json").write_text(
        json.dumps(compact_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    eligible = [row for row in compact_topology if row.get("eligible") and row.get("metric") is not None]
    if eligible:
        selected = min(eligible, key=lambda row: float(row["metric"]))
        selected_payload = {
            "global_step": int(selected["step"]),
            "metric": float(selected["metric"]),
            "eligible": True,
        }
        (snapshot / "evaluation").mkdir(exist_ok=True)
        (snapshot / "evaluation" / "selected.json").write_text(
            json.dumps(selected_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        compact_manifest["selected_candidate_step"] = selected_payload["global_step"]
        compact_manifest["selected_candidate_epoch"] = epoch_by_step.get(
            selected_payload["global_step"],
            0.0 if selected_payload["global_step"] == 0 else None,
        )
        (snapshot / "run_manifest.json").write_text(
            json.dumps(compact_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return compact_manifest


def render_examples(source: Path, output: Path, through_epoch: int | None) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if output == source or source in output.parents:
        raise ValueError("example output must be outside the source training run")
    snapshot = output / "snapshot"
    manifest = _snapshot(source, snapshot, through_epoch)

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    config = load_config(snapshot / "resolved_config.yaml")
    stage = str(config.get("stage", "training"))
    model_name = str(config.get("model", {}).get("name", "model"))
    description = f"{stage}:{model_name}"
    last_step = int(manifest["last_included_step"])
    steps_per_epoch = max(1, int(manifest["steps_per_epoch"]))
    monitor = TrainingMonitor(
        snapshot,
        start_step=0,
        final_step=max(1, last_step),
        configured_steps=max(1, last_step),
        steps_per_epoch=steps_per_epoch,
        description=description,
        enabled=False,
    )
    output.mkdir(parents=True, exist_ok=True)
    figures = {
        "loss_history.png": monitor._build_loss_figure(plt),
        "optimization_diagnostics.png": monitor._build_optimization_figure(plt),
    }
    for name, figure in figures.items():
        if figure is not None:
            figure.savefig(output / name, dpi=180, format="png")
            plt.close(figure)
    render_coherence_history(
        snapshot,
        description=description,
        output_path=output / "coherence_history.png",
        pyplot=plt,
    )
    render_fidelity_history(
        snapshot,
        output_path=output / "checkpoint_fidelity.png",
        pyplot=plt,
    )
    monitor.progress.close()
    manifest.update(
        source_run_dir=str(source),
        description=description,
        figures=[name for name in figures if (output / name).is_file()]
        + [
            name
            for name in ("coherence_history.png", "checkpoint_fidelity.png")
            if (output / name).is_file()
        ],
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Source training run; read-only")
    parser.add_argument("--through-epoch", type=int, default=1440, help="Freeze data at this complete epoch")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cases/turbulent_combustion/diagnostics/ExampleVisual/history"),
    )
    args = parser.parse_args()
    manifest = render_examples(args.run_dir, args.output_dir, args.through_epoch)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), **manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
