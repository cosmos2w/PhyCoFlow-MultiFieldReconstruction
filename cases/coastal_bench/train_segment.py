#!/usr/bin/env python
"""Continue one immutable training run in bounded, sequential Slurm segments."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path

import yaml

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parents[1] / "src"))

from phycoflow_reconstruction.cli import _load_case_config
from phycoflow_reconstruction.config import load_config, validate_config
from phycoflow_reconstruction.data.factory import open_field_dataset
from phycoflow_reconstruction.training.base_training import run_base_training
from phycoflow_reconstruction.training.run_store import load_project_checkpoint
from phycoflow_reconstruction.training.update_budget import base_training_steps_per_epoch


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def train_segment(
    state_dir: Path, *, segment_epochs: int = 2500, config_path: Path | None = None
) -> Path:
    if segment_epochs < 1:
        raise ValueError("segment_epochs must be positive")
    state_dir = state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "lock").open("a") as lock:
        # Never allow two jobs to update the same optimizer/checkpoint stream.
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen_config = state_dir / "resolved_config.yaml"
        if not frozen_config.exists():
            config = _load_case_config(
                config_path or CASE_DIR / "configs/base/gl_rbf_cq.yaml",
                CASE_DIR,
                "coastal_bench",
                [],
            )
            _atomic_text(frozen_config, yaml.safe_dump(config, sort_keys=False))
        config = load_config(frozen_config)
        validate_config(config)
        dataset = open_field_dataset(config["dataset"])
        try:
            steps_per_epoch = base_training_steps_per_epoch(config, len(dataset))
        finally:
            dataset.close()
        total_steps = int(config["optimization"]["epochs"]) * steps_per_epoch
        pointer = state_dir / "run_dir.txt"
        started = state_dir / "started.json"
        resume = Path(pointer.read_text().strip()) if pointer.exists() else None
        if resume is None and started.exists():
            raise RuntimeError(
                "The first segment was interrupted. Set run_dir.txt to its absolute run "
                "directory with a valid last.pt checkpoint before resubmitting; refusing "
                "to silently start another model."
            )
        completed = 0
        if resume is not None:
            checkpoint = load_project_checkpoint(resume / "checkpoints/last.pt")
            completed = int(checkpoint["global_step"])
            del checkpoint
            if completed >= total_steps:
                print(f"Already complete: {resume}", flush=True)
                return resume
        segment_steps = min(segment_epochs * steps_per_epoch, total_steps - completed)
        _atomic_text(
            started,
            json.dumps(
                {
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "resume": str(resume) if resume is not None else None,
                    "start_step": completed,
                    "segment_steps": segment_steps,
                    "configured_steps": total_steps,
                },
                indent=2,
            )
            + "\n",
        )
        print(
            f"Training steps {completed + 1}–{completed + segment_steps} / {total_steps}",
            flush=True,
        )
        run_dir = run_base_training(
            config, case_dir=CASE_DIR, max_steps=segment_steps, resume=resume
        )
        _atomic_text(pointer, str(run_dir.resolve()) + "\n")
        status = json.loads((run_dir / "status.json").read_text())
        _atomic_text(
            state_dir / "progress.json",
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "global_step": status["global_step"],
                    "configured_steps": total_steps,
                    "completed_epochs": status["global_step"] / steps_per_epoch,
                    "complete": status["global_step"] >= total_steps,
                },
                indent=2,
            )
            + "\n",
        )
        print(run_dir, flush=True)
        return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--segment-epochs", type=int, default=2500)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="case config frozen by the first segment (default: base/gl_rbf_cq.yaml)",
    )
    args = parser.parse_args()
    train_segment(args.state_dir, segment_epochs=args.segment_epochs, config_path=args.config)
