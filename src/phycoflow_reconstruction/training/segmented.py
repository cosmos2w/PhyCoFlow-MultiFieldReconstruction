"""Resume the common CLI in bounded recovery segments on a preemptable GPU.

This only selects run directories and invokes the existing post-train command;
optimization, evaluation, state validation and checkpoints remain trainer-owned.
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from phycoflow_reconstruction.cli import _load_case_config
from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.data.factory import open_field_dataset
from phycoflow_reconstruction.data.topology_subset import apply_training_subset
from phycoflow_reconstruction.training.run_store import config_digest, load_project_checkpoint
from phycoflow_reconstruction.training.update_budget import post_training_steps_per_epoch


def select_run(root: Path, digest: str) -> tuple[Path | None, int, bool]:
    """Select one matching checkpoint, ignoring setup interrupted before a save."""
    matching = []
    for path in sorted(root.glob("*/run_manifest.json")):
        manifest = json.loads(path.read_text())
        if manifest["config_sha256"] != digest:
            raise ValueError(f"experiment directory contains a different config: {path.parent}")
        status_path = path.parent / "status.json"
        # Preemption can occur during setup, before the first status/checkpoint.
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        checkpoint = path.parent / "checkpoints/last.pt"
        if checkpoint.exists():
            state = load_project_checkpoint(checkpoint)
            matching.append(
                (path.parent, int(state["global_step"]), status.get("status") == "completed")
            )
        elif status.get("status") == "completed":
            raise FileNotFoundError(
                f"completed run is missing its recovery checkpoint: {checkpoint}"
            )
    if len(matching) > 1:
        raise RuntimeError("multiple resumable children; resolve the run selection explicitly")
    return matching[0] if matching else (None, 0, False)


def segment_budget(step: int, total_steps: int, per_epoch: int, segment_epochs: int) -> int:
    """Finish the first epoch, then whole segments; zero means reporting recovery."""
    if not 0 <= step <= total_steps:
        raise ValueError("checkpoint step is outside the configured training budget")
    if per_epoch < 1 or segment_epochs < 1:
        raise ValueError("epoch and segment budgets must be positive")
    segment_steps = segment_epochs * per_epoch
    next_boundary = per_epoch if step < per_epoch else (step // segment_steps + 1) * segment_steps
    return min(next_boundary, total_steps) - step


def configured_epoch_steps(config: dict) -> int:
    """Count the same selected training snapshots as the trainer."""
    dataset = open_field_dataset(config["dataset"], split="train")
    try:
        subset = config["optimization"].get("training_subset")
        if subset:
            apply_training_subset(dataset, subset)
        return post_training_steps_per_epoch(config, len(dataset))
    finally:
        dataset.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit experiment config; no implicit historical recipe",
    )
    parser.add_argument("--allocation-hours", type=float, default=24.0)
    parser.add_argument("--segment-epochs", type=int, default=100)
    parser.add_argument(
        "--case-dir",
        type=Path,
        required=True,
        help="Case directory containing run.py and the runs directory",
    )
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument(
        "--requeue", action="store_true", help="Requeue a Slurm job at the deadline"
    )
    args = parser.parse_args(argv)
    if args.segment_epochs < 1 or args.allocation_hours <= 0:
        parser.error("segment-epochs and allocation-hours must be positive")
    case_dir = args.case_dir.resolve()
    if not (case_dir / "run.py").is_file():
        parser.error("case-dir must contain a run.py entrypoint")
    case_name = load_config(args.config, args.override).get("case", case_dir.name)
    config = _load_case_config(args.config.resolve(), case_dir, case_name, args.override)
    if config.get("stage") != "post_training":
        parser.error("segmented training requires a post_training config")
    experiment = config["output"]["experiment_name"]
    job_root = case_dir / "runs/long_jobs"
    job_root.mkdir(parents=True, exist_ok=True)
    with (job_root / f"{experiment}.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        per_epoch = configured_epoch_steps(config)
        total_steps = per_epoch * config["optimization"]["epochs"]
        deadline = time.monotonic() + args.allocation_hours * 3600
        estimate = 10.0
        while True:
            run, step, completed = select_run(case_dir / "runs" / experiment, config_digest(config))
            if completed:
                print(f"Completed {run}", flush=True)
                return
            budget = segment_budget(step, total_steps, per_epoch, args.segment_epochs)
            if time.monotonic() + budget * estimate * 1.25 + 600 > deadline:
                job_id = os.environ.get("SLURM_JOB_ID")
                if not args.requeue or not job_id:
                    raise RuntimeError(
                        "allocation deadline reached; resume this launcher in a new allocation"
                    )
                print(f"Requeueing job {job_id} after checkpoint step {step}", flush=True)
                subprocess.run(["scontrol", "requeue", job_id], check=True)
                return
            command = [
                sys.executable,
                "-u",
                str(case_dir / "run.py"),
                "post-train",
                "--config",
                str(args.config.resolve()),
                "--max-steps",
                str(budget),
            ]
            for override in args.override:
                command += ["--override", override]
            if run:
                command += ["--resume", str(run)]
            print(
                f"Training segment: steps {step}..{step + budget} of {total_steps}; epochs={config['optimization']['epochs']}, steps_per_epoch={per_epoch}",
                flush=True,
            )
            segment_started = time.monotonic()
            subprocess.run(command, check=True)
            run, new_step, completed = select_run(
                case_dir / "runs" / experiment, config_digest(config)
            )
            if new_step <= step and not (budget == 0 and completed):
                raise RuntimeError("training segment made no progress")
            status = json.loads((run / "status.json").read_text())
            elapsed = time.monotonic() - segment_started
            # Include startup, validation and reporting in the deadline estimate.
            if new_step > step:
                estimate = elapsed / (new_step - step)
            record = {
                "start_step": step,
                "end_step": new_step,
                "wall_seconds": elapsed,
                "training_loop_seconds": status["post_training_seconds"],
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            }
            with (run / "metrics/segments.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            summary = {
                "run": str(run),
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "global_step": new_step,
                "steps_per_epoch": per_epoch,
                "configured_steps": total_steps,
                "seconds_per_step": estimate,
                "estimated_remaining_hours": (total_steps - new_step) * estimate / 3600,
                "peak_cuda_memory_bytes": status["peak_cuda_memory_bytes"],
                "completed": completed,
            }
            temporary = job_root / f"{experiment}.json.tmp"
            temporary.write_text(json.dumps(summary, indent=2) + "\n")
            temporary.replace(job_root / f"{experiment}.json")
            print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
