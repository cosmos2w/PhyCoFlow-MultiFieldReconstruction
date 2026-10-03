#!/usr/bin/env python
"""Budget-guarded launcher for Test_1002 post-training pilot runs.

R2 exposes cumulative/additional epoch segments and separately accounts for
the sustained campaign. Historical R1 counters remain readable. Formal runs
and outputs outside the reviewed Test_1002 tree are rejected.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = PROJECT_ROOT / "cases" / "turbulent_combustion"
RUNS_DIR = CASE_DIR / "runs"
TEST_ROOT = RUNS_DIR / "Test_1002"
AUDIT_DIR = TEST_ROOT / "_audit"
SOURCE_CONTRACT_PATH = AUDIT_DIR / "source_contract.json"
PLANNED_RUNS_PATH = AUDIT_DIR / "planned_runs.json"
LEDGER_PATH = AUDIT_DIR / "stage_ledger.json"
LOCK_PATH = AUDIT_DIR / "pilot_launcher.lock"

EXPECTED_BRANCH = "codex/phycoflow-upgrade-1002"
PINNED_SOURCE_SHA256 = "03f2f45b966a00850cc45c011b4cf1cd644b9408de355324fec8dc434a84a810"
PINNED_STEPS_PER_EPOCH = 38
PILOT_BATCH_SIZE = 32
PILOT_TRAIN_FRACTION = 0.15
MAX_TARGETED_FIXES_PER_STAGE = 2

sys.path.insert(0, str(PROJECT_ROOT / "src"))


class PilotContractError(ValueError):
    """Raised when a proposed pilot violates the reviewed run contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PilotContractError(f"cannot read JSON contract {path}: {error}") from error
    if not isinstance(value, dict):
        raise PilotContractError(f"JSON contract must contain an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True, stderr=subprocess.PIPE
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise PilotContractError(f"cannot verify Git identity: {error}") from error


def _git_identity(repo: Path = PROJECT_ROOT) -> dict[str, str]:
    return {
        "branch": _git_value(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _git_value(repo, "rev-parse", "HEAD"),
    }


def _experiment_parts(value: Any) -> tuple[Path, str]:
    if not isinstance(value, str) or not value.strip():
        raise PilotContractError("output.experiment_name must be a non-empty relative path")
    if "\\" in value:
        raise PilotContractError("output.experiment_name cannot contain backslashes")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise PilotContractError(
            "output.experiment_name must be relative and cannot contain '..'"
        )
    if len(relative.parts) < 2 or relative.parts[0] != "Test_1002":
        raise PilotContractError(
            "pilot output.experiment_name must be nested below Test_1002/"
        )
    stage_match = re.match(r"^(R2_\d{2}|T\d{2})(?:_|$)", relative.parts[1])
    if stage_match is None:
        raise PilotContractError(
            "the first Test_1002 output directory must begin with a planned stage ID"
        )
    for part in relative.parts:
        token = part.casefold()
        if "formal" in token or "5000ep" in token:
            raise PilotContractError("formal-run output names are forbidden by this launcher")
    resolved = (RUNS_DIR / relative).resolve()
    try:
        resolved.relative_to(TEST_ROOT.resolve())
    except ValueError as error:
        raise PilotContractError("resolved output path escaped Test_1002") from error
    return relative, stage_match.group(1)


def validate_test_envelope(
    config: Mapping[str, Any],
    *,
    planned_runs: Mapping[str, Any],
    train_count: int,
) -> dict[str, Any]:
    """Validate test-mode settings and return a normalized budget contract."""
    if config.get("stage") != "post_training":
        raise PilotContractError("the pilot launcher accepts only stage=post_training")
    if config.get("case") != "turbulent_combustion":
        raise PilotContractError("the pilot launcher accepts only turbulent_combustion")

    relative_output, stage = _experiment_parts(config.get("output", {}).get("experiment_name"))
    caps = planned_runs.get("maximum_initial_epochs")
    if not isinstance(caps, Mapping) or stage not in caps:
        raise PilotContractError(f"stage {stage!r} is not present in planned_runs.json")
    stage_epoch_cap = int(caps[stage])
    lineage_epoch_cap = int(planned_runs.get("every_lineage_epoch_cap", 0))
    if stage_epoch_cap < 1 or lineage_epoch_cap < 1:
        raise PilotContractError("planned run caps must be positive")
    if stage_epoch_cap >= 250 or lineage_epoch_cap >= 250:
        raise PilotContractError("planned run caps must remain below 250 epochs")

    optimization = config.get("optimization", {})
    epochs = optimization.get("epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise PilotContractError("optimization.epochs must be a positive integer")
    # This guard is unconditional: --max-steps never makes a >=250 epoch config safe.
    if epochs >= 250:
        raise PilotContractError("test-mode configs with optimization.epochs >= 250 are refused")
    if (stage.startswith("R2_")
            and stage not in planned_runs.get("non_scientific_stages", []) and epochs < 100):
        raise PilotContractError("R2 scientific comparison horizons must be at least 100 epochs")
    if epochs > min(stage_epoch_cap, lineage_epoch_cap):
        raise PilotContractError(
            f"{stage} allows at most {min(stage_epoch_cap, lineage_epoch_cap)} configured epochs"
        )

    batch_size = optimization.get("batch_size")
    fraction = optimization.get("train_fraction")
    if isinstance(batch_size, bool) or batch_size != PILOT_BATCH_SIZE:
        raise PilotContractError(f"optimization.batch_size must be {PILOT_BATCH_SIZE}")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isclose(
        float(fraction), PILOT_TRAIN_FRACTION, rel_tol=0.0, abs_tol=1e-12
    ):
        raise PilotContractError(
            f"optimization.train_fraction must be {PILOT_TRAIN_FRACTION}"
        )
    if "training_subset" in optimization:
        raise PilotContractError(
            "training subsets change the pinned 38-updates-per-epoch pilot contract"
        )
    if optimization.get("sampling") == "full_pass":
        raise PilotContractError("full_pass sampling changes the pinned pilot update semantics")
    if optimization.get("steps_per_epoch") is not None:
        raise PilotContractError("explicit steps_per_epoch is not allowed for pilot configs")
    calculated_steps = math.ceil(int(train_count) * float(fraction) / int(batch_size))
    if calculated_steps != PINNED_STEPS_PER_EPOCH:
        raise PilotContractError(
            f"resolved training contract gives {calculated_steps} updates per epoch; "
            f"expected {PINNED_STEPS_PER_EPOCH}"
        )

    runtime = config.get("runtime", {})
    if runtime.get("device") != "cuda:0":
        raise PilotContractError("runtime.device must be cuda:0 for physical GPU 0")
    evaluation = config.get("evaluation", {})
    if evaluation.get("split", "validation") != "validation":
        raise PilotContractError("pilot selection/evaluation must use the validation split")
    preview = evaluation.get("preview", {})
    if preview.get("enabled", False) and preview.get("split", "validation") != "validation":
        raise PilotContractError("training preview selection must use the validation split")
    checkpointing = config.get("checkpointing", {})
    selection_metric = str(checkpointing.get("selection_metric", "")).casefold()
    if "test" in selection_metric:
        raise PilotContractError("test-set checkpoint selection is forbidden")
    if checkpointing.get("selection_split", "validation") != "validation":
        raise PilotContractError("checkpoint selection must use the validation split")

    return {
        "stage": stage,
        "experiment_name": relative_output.as_posix(),
        "output_root": str(RUNS_DIR / relative_output),
        "configured_epochs": epochs,
        "configured_steps": epochs * PINNED_STEPS_PER_EPOCH,
        "steps_per_epoch": PINNED_STEPS_PER_EPOCH,
        "stage_epoch_cap": stage_epoch_cap,
        "stage_attempted_update_cap": stage_epoch_cap * PINNED_STEPS_PER_EPOCH,
        "lineage_epoch_cap": lineage_epoch_cap,
        # R2's predeclared stage headroom covers lost/replayed work without
        # changing the frozen accepted-epoch horizon. R1 keeps its old cap.
        "lineage_attempted_update_cap": min(
            stage_epoch_cap if stage.startswith("R2_") else epochs, lineage_epoch_cap
        ) * PINNED_STEPS_PER_EPOCH,
        "batch_size": int(batch_size),
        "train_fraction": float(fraction),
        "evaluation_split": "validation",
    }


def validate_source_identity(
    config: Mapping[str, Any],
    source_contract: Mapping[str, Any],
    *,
    sha256_fn=_sha256,
) -> dict[str, Any]:
    """Verify the immutable live last.pt source pinned by the audit contract."""
    if source_contract.get("source_weight_selection") != "live":
        raise PilotContractError("source contract does not pin live source weights")
    expected_hash = source_contract.get("source_checkpoint_sha256")
    if expected_hash != PINNED_SOURCE_SHA256:
        raise PilotContractError("source contract differs from the pinned live checkpoint SHA-256")
    try:
        source_run = Path(source_contract["runs"]["source"]["path"]).resolve()
    except (KeyError, TypeError) as error:
        raise PilotContractError("source contract has no resolved base source run") from error
    configured_run = Path(str(config.get("source_run", ""))).resolve()
    if configured_run != source_run:
        raise PilotContractError("config source_run differs from the pinned base source run")
    configured_checkpoint = str(config.get("source_checkpoint", ""))
    if configured_checkpoint != "last.pt":
        raise PilotContractError("pilot source_checkpoint must be last.pt")
    checkpoint_path = source_run / "checkpoints" / "last.pt"
    if not checkpoint_path.is_file():
        raise PilotContractError(f"pinned source checkpoint is missing: {checkpoint_path}")
    actual_hash = sha256_fn(checkpoint_path)
    if actual_hash != expected_hash:
        raise PilotContractError(
            f"pinned source checkpoint hash changed: expected {expected_hash}, got {actual_hash}"
        )
    model = config.get("model", {})
    if model.get("model_ema_eval") is not False:
        raise PilotContractError(
            "model.model_ema_eval must be false so the pilot loads the pinned live weights"
        )
    contract_dataset = Path(str(source_contract.get("dataset_path", ""))).resolve()
    configured_dataset = Path(str(config.get("dataset", {}).get("path", ""))).resolve()
    if not contract_dataset or configured_dataset != contract_dataset:
        raise PilotContractError("config dataset path differs from source_contract.json")
    return {
        "source_run": str(source_run),
        "source_checkpoint": str(checkpoint_path),
        "source_weight_selection": "live",
        "source_checkpoint_sha256": actual_hash,
        "source_global_step": int(source_contract.get("source_global_step", -1)),
        "dataset": str(contract_dataset),
        "dataset_fingerprint": source_contract.get("dataset_fingerprint"),
    }


def _validate_visible_device(config: Mapping[str, Any], visible_devices: str | None) -> None:
    if visible_devices != "0":
        raise PilotContractError(
            "set CUDA_VISIBLE_DEVICES=0; the pilot launcher will not select another GPU"
        )
    if config.get("runtime", {}).get("device") != "cuda:0":
        raise PilotContractError("runtime.device must be cuda:0 after exposing physical GPU 0")


def _empty_ledger(global_cap: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "global_accepted_update_cap": int(global_cap),
        "accepted_updates": 0,
        "attempted_updates": 0,
        "branch": None,
        "stages": {},
        "lineages": {},
        "runs": [],
        "active_attempt": None,
    }


def _load_ledger(path: Path, global_cap: int) -> dict[str, Any]:
    if not path.exists():
        return _empty_ledger(global_cap)
    ledger = _read_json(path)
    if int(ledger.get("global_accepted_update_cap", -1)) != int(global_cap):
        raise PilotContractError("stage ledger global cap differs from planned_runs.json")
    if ledger.get("runs") and not ledger.get("lineages"):
        raise PilotContractError(
            "legacy stage ledger has run entries but no lineage accounting; refusing reset"
        )
    ledger.setdefault("schema_version", 1)
    ledger.setdefault("accepted_updates", 0)
    ledger.setdefault("attempted_updates", 0)
    ledger.setdefault("branch", None)
    ledger.setdefault("stages", {})
    ledger.setdefault("lineages", {})
    ledger.setdefault("runs", [])
    ledger.setdefault("active_attempt", None)
    return ledger


def _save_ledger(path: Path, ledger: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(ledger), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            if index == len(lines) - 1:
                break
            raise PilotContractError(f"corrupt non-terminal history row in {path}") from error
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    checkpoint_path = path / "checkpoints" / "last.pt"
    if not checkpoint_path.is_file():
        return None
    from phycoflow_reconstruction.training.run_store import load_project_checkpoint

    payload = load_project_checkpoint(checkpoint_path)
    return payload if isinstance(payload, dict) else None


def _checkpoint_counters(checkpoint: Mapping[str, Any] | None) -> dict[str, int]:
    if checkpoint is None:
        return {"step": 0, "attempted": 0, "accepted": 0}
    step = int(checkpoint.get("global_step", 0))
    if isinstance(checkpoint.get("fidelity_controller"), Mapping):
        controller = checkpoint["fidelity_controller"]
        updates = int(controller.get("updates", step))
        return {"step": step, "attempted": updates, "accepted": updates}
    if isinstance(checkpoint.get("component_controller"), Mapping):
        controller = checkpoint["component_controller"]
        attempted = int(controller.get("attempted", step))
        accepted = int(controller.get("accepted", attempted))
        return {"step": step, "attempted": attempted, "accepted": accepted}
    return {"step": step, "attempted": step, "accepted": step}


def _run_snapshot(run_dir: Path) -> dict[str, int]:
    checkpoint = _load_checkpoint(run_dir)
    counters = _checkpoint_counters(checkpoint)
    status_path = run_dir / "status.json"
    if status_path.is_file():
        status = _read_json(status_path)
        status_step = int(status.get("global_step", 0))
        if status_step > counters["step"]:
            counters["step"] = status_step
            if checkpoint is None or not checkpoint.get("component_controller"):
                counters["attempted"] = status_step
                counters["accepted"] = status_step
    progress_path = run_dir / "progress.json"
    if progress_path.is_file():
        progress = _read_json(progress_path)
        progress_step = int(progress.get("completed_updates", 0))
        if progress_step > counters["step"]:
            counters["step"] = progress_step
            if checkpoint is None or not checkpoint.get("component_controller"):
                counters["attempted"] = progress_step
                counters["accepted"] = progress_step
    return counters


def _pending_history_counters(
    run_dir: Path, after_step: int, *, include_update_stream: bool = False
) -> dict[str, int]:
    attempted = accepted = 0
    summary_cursor = int(after_step)
    for row in _read_history(run_dir / "metrics" / "history.jsonl"):
        if int(row.get("step", 0)) <= int(after_step):
            continue
        batches = int(row.get("batches", 1))
        summary_cursor = max(summary_cursor, int(row.get("step", 0)))
        attempted += batches
        fraction = row.get("update_accepted_fraction")
        if isinstance(fraction, (int, float)):
            accepted += round(batches * float(fraction))
        elif isinstance(row.get("update_accepted"), bool):
            accepted += batches if row["update_accepted"] else 0
        else:
            # The default update path accepts every optimizer update.
            accepted += batches
    if include_update_stream:
        # A killed process can leave a partial epoch beyond the last flushed
        # summary. Charge this durable tail before recovery truncates it.
        tail = {int(row.get("step", 0)): row
                for row in _read_history(run_dir / "metrics" / "coherence_updates.jsonl")
                if int(row.get("step", 0)) > summary_cursor}
        attempted += len(tail)
        accepted += sum(row.get("update_accepted", True) is not False for row in tail.values())
    return {"attempted": attempted, "accepted": accepted}


def _attempt_delta(run_dir: Path, active: Mapping[str, Any]) -> dict[str, int]:
    snapshot = _run_snapshot(run_dir)
    start_step = int(active.get("start_step", 0))
    start_attempted = int(active.get("start_attempted", start_step))
    start_accepted = int(active.get("start_accepted", start_step))
    committed_attempted = max(0, snapshot["attempted"] - start_attempted)
    committed_accepted = max(0, snapshot["accepted"] - start_accepted)
    pending = _pending_history_counters(
        run_dir, snapshot["step"],
        include_update_stream=str(active.get("stage", "")).startswith("R2_"),
    )
    return {
        "attempted": committed_attempted + pending["attempted"],
        "accepted": committed_accepted + pending["accepted"],
        "end_step": snapshot["step"],
        "committed_accepted_total": snapshot["accepted"],
        "committed_attempted_total": snapshot["attempted"],
    }


def _locate_active_run(active: Mapping[str, Any]) -> Path | None:
    value = active.get("run_dir")
    if value:
        path = Path(str(value)).resolve()
        return path if path.is_dir() else None
    output_root = Path(str(active.get("output_root", ""))).resolve()
    if not output_root.is_dir():
        return None
    digest = str(active.get("config_sha256", ""))
    started = str(active.get("started_utc", ""))
    try:
        started_floor = datetime.fromisoformat(started.replace("Z", "+00:00")).strftime(
            "%Y%m%dT%H%M%S"
        )
    except ValueError:
        started_floor = ""
    candidates = []
    for child in output_root.iterdir():
        manifest_path = child / "run_manifest.json"
        if not child.is_dir() or not manifest_path.is_file():
            continue
        try:
            manifest = _read_json(manifest_path)
        except PilotContractError:
            continue
        created = str(manifest.get("created_utc", ""))
        if manifest.get("config_sha256") == digest and (not started_floor or created >= started_floor):
            candidates.append(child)
    return max(candidates, key=lambda item: item.stat().st_mtime) if candidates else None


def _stage_entry(ledger: dict[str, Any], stage: str) -> dict[str, Any]:
    stages = ledger.setdefault("stages", {})
    entry = stages.setdefault(
        stage,
        {
            "accepted_updates": 0,
            "attempted_updates": 0,
            "configurations": {},
            "targeted_fix_count": 0,
            "revision_semantics": (
                "first resolved config is initial; each distinct later resolved-config "
                "digest counts as one targeted fix, including failed attempts"
            ),
        },
    )
    return entry


def _register_stage_config(
    stage_entry: dict[str, Any], config_sha256: str, experiment_name: str
) -> str:
    configurations = stage_entry.setdefault("configurations", {})
    if config_sha256 in configurations:
        return str(configurations[config_sha256]["revision_type"])
    fix_count = int(stage_entry.get("targeted_fix_count", 0))
    if configurations:
        fix_count += 1
        if fix_count > MAX_TARGETED_FIXES_PER_STAGE:
            raise PilotContractError(
                f"{MAX_TARGETED_FIXES_PER_STAGE} targeted fixes are already recorded for this stage"
            )
        revision_type = f"targeted_fix_{fix_count}"
    else:
        revision_type = "initial"
    configurations[config_sha256] = {
        "experiment_name": experiment_name,
        "first_seen_utc": _utc_now(),
        "revision_type": revision_type,
    }
    stage_entry["targeted_fix_count"] = fix_count
    return revision_type


def _apply_attempt(
    ledger: dict[str, Any],
    active: Mapping[str, Any],
    *,
    run_dir: Path | None,
    status: str,
    error: str | None = None,
) -> dict[str, int]:
    delta = (
        _attempt_delta(run_dir, active)
        if run_dir is not None and run_dir.is_dir()
        else {"attempted": 0, "accepted": 0, "end_step": int(active.get("start_step", 0))}
    )
    stage = str(active["stage"])
    stage_entry = _stage_entry(ledger, stage)
    accepted, attempted = int(delta["accepted"]), int(delta["attempted"])
    ledger["accepted_updates"] = int(ledger.get("accepted_updates", 0)) + accepted
    ledger["attempted_updates"] = int(ledger.get("attempted_updates", 0)) + attempted
    stage_entry["accepted_updates"] = int(stage_entry.get("accepted_updates", 0)) + accepted
    stage_entry["attempted_updates"] = int(stage_entry.get("attempted_updates", 0)) + attempted

    if run_dir is not None:
        lineage_key = str(run_dir.resolve())
    else:
        lineage_key = str(active.get("lineage_key", f"unmaterialized:{uuid.uuid4()}"))
    lineages = ledger.setdefault("lineages", {})
    lineage = lineages.setdefault(
        lineage_key,
        {
            "run_dir": str(run_dir.resolve()) if run_dir is not None else None,
            "stage": stage,
            "config_sha256": str(active["config_sha256"]),
            "experiment_name": str(active["experiment_name"]),
            "configured_epochs": int(active["configured_epochs"]),
            "steps_per_epoch": int(active["steps_per_epoch"]),
            "accepted_updates": 0,
            "attempted_updates": 0,
            "last_committed_global_step": int(active.get("start_step", 0)),
        },
    )
    lineage["accepted_updates"] = int(lineage.get("accepted_updates", 0)) + accepted
    lineage["attempted_updates"] = int(lineage.get("attempted_updates", 0)) + attempted
    lineage["last_committed_global_step"] = int(
        delta.get("end_step", lineage.get("last_committed_global_step", 0))
    )
    lineage["last_status"] = status
    lineage["last_implementation_commit"] = active.get("implementation_commit")
    run = {
        "stage": stage,
        "run_dir": str(run_dir.resolve()) if run_dir is not None else None,
        "lineage_key": lineage_key,
        "config_sha256": str(active["config_sha256"]),
        "implementation_commit": active.get("implementation_commit"),
        "revision_type": str(active.get("revision_type", "initial")),
        "kind": str(active.get("kind", "initial")),
        "status": status,
        "start_step": int(active.get("start_step", 0)),
        "end_step": int(delta.get("end_step", active.get("start_step", 0))),
        "attempted_updates": attempted,
        "accepted_updates": accepted,
        "started_utc": active.get("started_utc"),
        "finished_utc": _utc_now(),
    }
    if stage.startswith("R2_"):
        epoch_size = int(active["steps_per_epoch"])
        run.update(
            start_epoch=int(active.get("start_step", 0)) / epoch_size,
            end_epoch=int(delta.get("end_step", 0)) / epoch_size,
            consumed_epochs=attempted / epoch_size,
            minimum_scientific_horizon_reached=int(delta.get("end_step", 0)) >= 100 * epoch_size,
        )
    if error:
        run["error"] = error[:1000]
    ledger.setdefault("runs", []).append(run)
    ledger["active_attempt"] = None
    return delta


def _reconcile_active(ledger: dict[str, Any]) -> bool:
    active = ledger.get("active_attempt")
    if not isinstance(active, Mapping):
        return False
    run_dir = _locate_active_run(active)
    status = "recovered_interrupted" if run_dir is not None else "interrupted_before_run_store"
    if run_dir is not None:
        status_path = run_dir / "status.json"
        if status_path.is_file() and _read_json(status_path).get("status") in {
            "completed",
            "integration_truncated",
            "fidelity_guard_failed",
        }:
            status = "recovered_" + str(_read_json(status_path).get("status"))
    _apply_attempt(ledger, active, run_dir=run_dir, status=status)
    return True


def _lineage_key_for_resume(resume: Path, ledger: Mapping[str, Any]) -> str:
    key = str(resume.resolve())
    if key not in ledger.get("lineages", {}):
        active = ledger.get("active_attempt")
        if not isinstance(active, Mapping) or Path(str(active.get("run_dir", ""))).resolve() != resume.resolve():
            raise PilotContractError(
                "resume lineage is absent from the pilot ledger; refusing an untracked resume"
            )
    return key


def _effective_step_cap(
    *,
    requested_max_steps: int | None,
    configured_steps: int,
    start_step: int,
    stage_remaining: int,
    global_remaining: int,
    lineage_remaining: int,
    is_resume: bool,
) -> int:
    if requested_max_steps is not None and requested_max_steps < 0:
        raise PilotContractError("--max-steps must be non-negative")
    configured_remaining = max(0, int(configured_steps) - int(start_step))
    requested = configured_remaining if requested_max_steps is None else int(requested_max_steps)
    if requested == 0 and not (is_resume and configured_remaining == 0):
        raise PilotContractError("--max-steps 0 is allowed only for a completed resume")
    return max(
        0,
        min(
            requested,
            configured_remaining,
            int(stage_remaining),
            int(global_remaining),
            int(lineage_remaining),
        ),
    )


def epoch_segment_limit(
    *, until_epoch: int | None, additional_epochs: int | None, start_step: int,
    configured_epochs: int, steps_per_epoch: int = PINNED_STEPS_PER_EPOCH,
) -> int | None:
    """Translate scientist-facing epochs without resetting resumed lineage age."""
    if until_epoch is not None and additional_epochs is not None:
        raise PilotContractError("choose --until-epoch or --additional-epochs")
    if until_epoch is None and additional_epochs is None:
        return None
    value = until_epoch if until_epoch is not None else additional_epochs
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PilotContractError("epoch segment arguments must be positive integers")
    target = (until_epoch * steps_per_epoch if until_epoch is not None
              else start_step + additional_epochs * steps_per_epoch)
    if target >= 250 * steps_per_epoch:
        raise PilotContractError("cumulative pilot age must remain strictly below 250 epochs")
    if target > configured_epochs * steps_per_epoch:
        raise PilotContractError("requested epoch exceeds the frozen configured horizon")
    if target <= start_step:
        raise PilotContractError("requested epoch has already been reached")
    return target - start_step


def _verify_gpu0() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise PilotContractError(
            "physical GPU 0 must be the only visible CUDA device for a pilot launch"
        )
    torch.cuda.set_device(0)
    visible_name = torch.cuda.get_device_name(0)
    visible_uuid = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    try:
        physical = subprocess.check_output(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=uuid,name",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise PilotContractError(f"cannot verify physical GPU 0 with nvidia-smi: {error}") from error
    try:
        uuid, physical_name = [part.strip() for part in physical.split(",", 1)]
    except ValueError as error:
        raise PilotContractError(f"unexpected nvidia-smi GPU identity: {physical!r}") from error
    if (physical_name != visible_name
            or visible_uuid.removeprefix("GPU-").lower() != uuid.removeprefix("GPU-").lower()):
        raise PilotContractError(
            "logical cuda:0 name/UUID does not match physical GPU 0 selected by nvidia-smi"
        )
    return {
        "physical_index": 0,
        "uuid": uuid,
        "name": physical_name,
        "logical_device": "cuda:0",
        "cuda_visible_devices": "0",
    }


def _resolved_config(config_path: Path) -> dict[str, Any]:
    from phycoflow_reconstruction.cli import _load_case_config

    return _load_case_config(
        config_path,
        CASE_DIR,
        "turbulent_combustion",
        [],
        validate_stage=True,
    )


def _resolve_resume_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    try:
        path.relative_to(TEST_ROOT.resolve())
    except ValueError as error:
        raise PilotContractError("--resume must point to a run under Test_1002") from error
    if not (path / "run_manifest.json").is_file():
        raise PilotContractError(f"--resume has no run_manifest.json: {path}")
    return path


def _run_config_digest(config: Mapping[str, Any]) -> str:
    from phycoflow_reconstruction.training.run_store import config_digest

    return config_digest(config)


def _prepare_contract(
    config: dict[str, Any],
    *,
    source_contract: dict[str, Any],
    planned_runs: dict[str, Any],
    branch: str,
    visible_devices: str | None,
) -> dict[str, Any]:
    envelope = validate_test_envelope(
        config,
        planned_runs=planned_runs,
        train_count=int(source_contract.get("train_count", 0)),
    )
    source = validate_source_identity(config, source_contract)
    _validate_visible_device(config, visible_devices)
    if branch != EXPECTED_BRANCH:
        raise PilotContractError(
            f"pilot launches are pinned to working branch {EXPECTED_BRANCH!r}; got {branch!r}"
        )
    return {
        "mode": "bounded_test_pilot",
        "branch": branch,
        "config_sha256": _run_config_digest(config),
        "resolved_config": config,
        "source": source,
        "stage": envelope["stage"],
        "experiment_name": envelope["experiment_name"],
        "output_root": envelope["output_root"],
        "configured_epochs": envelope["configured_epochs"],
        "configured_steps": envelope["configured_steps"],
        "steps_per_epoch": envelope["steps_per_epoch"],
        "stage_epoch_cap": envelope["stage_epoch_cap"],
        "stage_attempted_update_cap": envelope["stage_attempted_update_cap"],
        "lineage_epoch_cap": envelope["lineage_epoch_cap"],
        "lineage_attempted_update_cap": envelope["lineage_attempted_update_cap"],
        "batch_size": envelope["batch_size"],
        "train_fraction": envelope["train_fraction"],
        "evaluation_split": envelope["evaluation_split"],
        "physical_cuda_visible_devices": visible_devices,
        "runtime_device": config["runtime"]["device"],
        "formal_launch_authorized": False,
        "test_set_selection_enabled": False,
    }


def _verify_resume_identity(
    resume: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    ledger: Mapping[str, Any],
    stage: str,
) -> tuple[str, dict[str, int]]:
    lineage_key = _lineage_key_for_resume(resume, ledger)
    lineage = ledger.get("lineages", {}).get(lineage_key)
    if not isinstance(lineage, Mapping):
        active = ledger.get("active_attempt")
        if isinstance(active, Mapping) and Path(str(active.get("run_dir", ""))).resolve() == resume:
            lineage = active
        else:
            raise PilotContractError("resume lineage is not durably recorded")
    manifest = _read_json(resume / "run_manifest.json")
    if manifest.get("config_sha256") != config_sha256:
        raise PilotContractError("resume resolved config hash differs from the saved lineage")
    if str(lineage.get("config_sha256", config_sha256)) != config_sha256:
        raise PilotContractError("resume config differs from the ledgered lineage")
    if str(lineage.get("stage", stage)) != stage:
        raise PilotContractError("resume stage differs from the ledgered lineage")
    if manifest.get("experiment_name") != config.get("output", {}).get("experiment_name"):
        raise PilotContractError("resume experiment path differs from the resolved config")
    snapshot = _run_snapshot(resume)
    return lineage_key, snapshot


def _make_active_attempt(
    contract: Mapping[str, Any],
    *,
    resume: Path | None,
    start_counters: Mapping[str, int],
    max_steps: int,
    revision_type: str,
) -> dict[str, Any]:
    config_sha = str(contract["config_sha256"])
    return {
        "attempt_id": str(uuid.uuid4()),
        "stage": str(contract["stage"]),
        "kind": "resume" if resume is not None else "initial",
        "run_dir": str(resume.resolve()) if resume is not None else None,
        "output_root": str(contract["output_root"]),
        "lineage_key": (
            str(resume.resolve())
            if resume is not None
            else f"pending:{contract['stage']}:{config_sha}:{contract['experiment_name']}"
        ),
        "experiment_name": str(contract["experiment_name"]),
        "config_sha256": config_sha,
        "implementation_commit": contract.get("implementation_commit"),
        "revision_type": revision_type,
        "configured_epochs": int(contract["configured_epochs"]),
        "steps_per_epoch": int(contract["steps_per_epoch"]),
        "start_step": int(start_counters["step"]),
        "start_attempted": int(start_counters["attempted"]),
        "start_accepted": int(start_counters["accepted"]),
        "planned_max_steps": int(max_steps),
        "started_utc": _utc_now(),
    }


def _print_contract(contract: Mapping[str, Any], *, output_path: Path, max_steps: int) -> None:
    payload = dict(contract)
    payload["effective_max_steps"] = int(max_steps)
    payload["exact_output_path"] = str(output_path.resolve())
    if contract.get("study") == "r2":
        keys = (
            "study", "stage", "branch", "implementation_commit", "config_sha256",
            "configured_epochs", "start_epoch", "until_epoch", "campaign_consumed_epochs",
            "minimum_scientific_horizon_epochs", "source", "gpu", "exact_output_path",
            "formal_launch_authorized", "test_set_selection_enabled",
        )
        payload = {key: payload.get(key) for key in keys}
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    print(f"Output path: {output_path.resolve()}", flush=True)


def run_pilot(
    *,
    config_path: Path,
    resume_arg: str | Path | None = None,
    max_steps: int | None = None,
    until_epoch: int | None = None,
    additional_epochs: int | None = None,
    study: str = "r1",
    dry_run: bool = False,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Validate and optionally execute one bounded post-training invocation."""
    environment = os.environ if env is None else env
    config_path = Path(config_path).expanduser().resolve()
    config = _resolved_config(config_path)
    source_contract = _read_json(SOURCE_CONTRACT_PATH)
    if study not in {"r1", "r2"}:
        raise PilotContractError("study must be r1 or r2")
    r2_root = AUDIT_DIR / "R2_campaign"
    planned_path = r2_root / "planned_runs.json" if study == "r2" else PLANNED_RUNS_PATH
    ledger_path = r2_root / "stage_ledger.json" if study == "r2" else LEDGER_PATH
    planned_runs = _read_json(planned_path)
    if max_steps is not None and (until_epoch is not None or additional_epochs is not None):
        raise PilotContractError("epoch arguments cannot be combined with --max-steps")
    branch_identity = _git_identity()
    contract = _prepare_contract(
        config,
        source_contract=source_contract,
        planned_runs=planned_runs,
        branch=branch_identity["branch"],
        visible_devices=environment.get("CUDA_VISIBLE_DEVICES"),
    )
    contract["implementation_commit"] = branch_identity["commit"]
    resume = _resolve_resume_path(resume_arg)
    config_sha = str(contract["config_sha256"])
    stage = str(contract["stage"])
    if (study == "r2") != stage.startswith("R2_"):
        raise PilotContractError("output stage must match the declared study")
    output_root = Path(str(contract["output_root"])).resolve()
    test_root = TEST_ROOT.resolve()
    try:
        output_root.relative_to(test_root)
    except ValueError as error:
        raise PilotContractError("resolved output path escaped Test_1002") from error
    global_cap = int(planned_runs.get("global_accepted_update_cap", 0))
    if study == "r2" and global_cap != 1100 * PINNED_STEPS_PER_EPOCH:
        raise PilotContractError("R2 campaign cap must be exactly 1100 historical epochs")
    if global_cap < 1:
        raise PilotContractError("planned_runs.json has no global accepted-update cap")

    lock_path = LOCK_PATH
    with _exclusive_lock(lock_path):
        ledger = _load_ledger(ledger_path, global_cap)
        recorded_branch = ledger.get("branch")
        if recorded_branch not in (None, branch_identity["branch"]):
            raise PilotContractError(
                f"ledger branch {recorded_branch!r} differs from current branch "
                f"{branch_identity['branch']!r}"
            )
        if ledger.get("active_attempt") is not None:
            _reconcile_active(ledger)
            if not dry_run:
                _save_ledger(ledger_path, ledger)

        stage_entry = _stage_entry(ledger, stage)
        configurations = stage_entry.setdefault("configurations", {})
        existing_revision = configurations.get(config_sha)
        if resume is not None:
            lineage_key, start_counters = _verify_resume_identity(
                resume, config, config_sha, ledger, stage
            )
            lineage = ledger["lineages"].get(lineage_key, {})
            lineage_attempted = int(lineage.get("attempted_updates", 0))
            revision_type = str(
                lineage.get(
                    "revision_type",
                    existing_revision.get("revision_type", "initial")
                    if existing_revision
                    else "initial",
                )
            )
            run_dir = resume
        else:
            lineage_key = "new:" + config_sha + ":" + contract["experiment_name"]
            start_counters = {"step": 0, "attempted": 0, "accepted": 0}
            lineage_attempted = 0
            run_dir = None
            if output_root.exists() and not output_root.is_dir():
                raise PilotContractError(f"experiment output root is not a directory: {output_root}")
            revision_type = str(existing_revision["revision_type"]) if existing_revision else ""
            if not dry_run:
                revision_type = _register_stage_config(
                    stage_entry, config_sha, contract["experiment_name"]
                )
            elif not existing_revision and configurations:
                if int(stage_entry.get("targeted_fix_count", 0)) >= MAX_TARGETED_FIXES_PER_STAGE:
                    raise PilotContractError(
                        f"{MAX_TARGETED_FIXES_PER_STAGE} targeted fixes are already recorded for this stage"
                    )
                revision_type = f"targeted_fix_{int(stage_entry.get('targeted_fix_count', 0)) + 1}"
            elif not existing_revision:
                revision_type = "initial"

        steps_per_epoch = int(contract["steps_per_epoch"])
        epoch_limit = epoch_segment_limit(
            until_epoch=until_epoch, additional_epochs=additional_epochs,
            start_step=int(start_counters["step"]),
            configured_epochs=int(contract["configured_epochs"]),
            steps_per_epoch=steps_per_epoch,
        )
        if epoch_limit is not None:
            max_steps = epoch_limit
        stage_remaining = int(contract["stage_attempted_update_cap"]) - int(
            stage_entry.get("attempted_updates", 0)
        )
        # R2 charges lost/replayed work as well as accepted work. R1 keeps its
        # original ledger mathematics and remains readable without migration.
        consumption_key = "attempted_updates" if study == "r2" else "accepted_updates"
        global_remaining = global_cap - int(ledger.get(consumption_key, 0))
        lineage_cap = (
            min(int(contract["lineage_attempted_update_cap"]), 249 * steps_per_epoch)
        )
        lineage_remaining = lineage_cap - lineage_attempted
        if study == "r2":
            requested_work = (max_steps if max_steps is not None else
                              int(contract["configured_steps"]) - int(start_counters["step"]))
            if requested_work > min(stage_remaining, global_remaining, lineage_remaining):
                raise PilotContractError("requested epoch segment exceeds the remaining R2 budget")
        effective_max_steps = _effective_step_cap(
            requested_max_steps=max_steps,
            configured_steps=int(contract["configured_steps"]),
            start_step=int(start_counters["step"]),
            stage_remaining=stage_remaining,
            global_remaining=global_remaining,
            lineage_remaining=lineage_remaining,
            is_resume=resume is not None,
        )
        if effective_max_steps < 1 and not (
            resume is not None
            and int(start_counters["step"]) == int(contract["configured_steps"])
            and max_steps == 0
        ):
            raise PilotContractError(
                "no pilot update budget remains for this stage, lineage, or global ledger"
            )

        contract["ledger_before"] = {
            "global_accepted_updates": int(ledger.get("accepted_updates", 0)),
            "global_accepted_update_cap": global_cap,
            "stage_attempted_updates": int(stage_entry.get("attempted_updates", 0)),
            "stage_attempted_update_cap": int(contract["stage_attempted_update_cap"]),
            "lineage_attempted_updates": lineage_attempted,
            "lineage_attempted_update_cap": lineage_cap,
            "revision_type": revision_type,
        }
        contract["resume"] = str(resume) if resume is not None else None
        contract["start_counters"] = dict(start_counters)
        contract["formal_launch_authorized"] = False
        contract["study"] = study
        contract["start_epoch"] = int(start_counters["step"]) / steps_per_epoch
        contract["until_epoch"] = (
            int(start_counters["step"]) + effective_max_steps
        ) / steps_per_epoch
        contract["minimum_scientific_horizon_epochs"] = 100 if study == "r2" else None
        contract["campaign_consumed_epochs"] = int(ledger.get(consumption_key, 0)) / steps_per_epoch
        gpu_identity = None if dry_run else _verify_gpu0()
        contract["gpu"] = (
            gpu_identity
            if gpu_identity is not None
            else {
                "physical_index": 0,
                "uuid": None,
                "name": "not probed in dry-run",
                "logical_device": "cuda:0",
                "cuda_visible_devices": "0",
            }
        )
        display_path = resume if resume is not None else output_root
        _print_contract(contract, output_path=display_path, max_steps=effective_max_steps)
        if dry_run:
            return None

        _validate_visible_device(config, environment.get("CUDA_VISIBLE_DEVICES"))
        current_source = validate_source_identity(config, source_contract)
        if current_source != contract["source"]:
            raise PilotContractError("source identity changed between preflight and launch")
        before_branch = _git_identity()
        if before_branch != branch_identity:
            raise PilotContractError("Git branch or commit changed during pilot preflight")
        if ledger.get("branch") is None:
            ledger["branch"] = branch_identity["branch"]
        active = _make_active_attempt(
            contract,
            resume=resume,
            start_counters=start_counters,
            max_steps=effective_max_steps,
            revision_type=revision_type,
        )
        if resume is None:
            active["lineage_key"] = lineage_key
        ledger["active_attempt"] = active
        _save_ledger(ledger_path, ledger)

        try:
            from phycoflow_reconstruction.training.post_training import run_post_training

            result = run_post_training(
                config,
                case_dir=CASE_DIR,
                max_steps=effective_max_steps,
                resume=resume,
            )
            run_dir = Path(result).resolve()
            try:
                run_dir.relative_to(test_root)
            except ValueError as error:
                raise PilotContractError("RunStore returned an output outside Test_1002") from error
            after_source = validate_source_identity(config, source_contract)
            after_branch = _git_identity()
            if after_source != contract["source"]:
                raise PilotContractError("pinned source changed during post-training")
            if after_branch != branch_identity:
                raise PilotContractError("Git branch or commit changed during post-training")
            final_status = (
                _read_json(run_dir / "status.json").get("status", "returned")
                if (run_dir / "status.json").is_file()
                else "returned_without_status"
            )
            _apply_attempt(ledger, active, run_dir=run_dir, status=str(final_status))
            ledger["branch"] = branch_identity["branch"]
            _save_ledger(ledger_path, ledger)
            print(f"Run directory: {run_dir}", flush=True)
            return run_dir
        except BaseException as error:
            current_run = _locate_active_run(active)
            _apply_attempt(
                ledger,
                active,
                run_dir=current_run,
                status="failed" if current_run is not None else "failed_before_run_store",
                error=f"{type(error).__name__}: {error}",
            )
            ledger["branch"] = branch_identity["branch"]
            _save_ledger(ledger_path, ledger)
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--study", choices=("r1", "r2"), default="r1")
    segment = parser.add_mutually_exclusive_group()
    segment.add_argument("--max-steps", type=int)
    segment.add_argument("--until-epoch", type=int)
    segment.add_argument("--additional-epochs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    run_pilot(
        config_path=args.config,
        resume_arg=args.resume,
        max_steps=args.max_steps,
        until_epoch=args.until_epoch,
        additional_epochs=args.additional_epochs,
        study=args.study,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
