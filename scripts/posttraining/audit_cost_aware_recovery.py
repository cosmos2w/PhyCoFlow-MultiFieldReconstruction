"""Capture and compare the predeclared CPU-only cost-aware recovery probe.

This helper reads only a run manifest, resolved configuration bytes, JSONL
training diagnostics, and a PyTorch checkpoint loaded with ``weights_only`` on
CPU. It never opens datasets, constructs a model, invokes validators, or touches
the training process. ``capture`` must be run before recovery trims speculative
rows 77 and 78; ``compare`` is run after the same run lineage has replayed them.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(
    "/data2/wanglz/CostAwareLPC/CostAwareLPC_20261023T185427Z_01a16a15"
)
DEFAULT_PROTOCOL = DEFAULT_ROOT / "analysis/recovery_probe_protocol.json"
DEFAULT_DECISIONS = DEFAULT_ROOT / "analysis/recovery_probe_expected_counter_decisions.json"
UPDATE_LOG = Path("metrics/execution_strategy_updates.jsonl")
EXPECTED_TASK = "D"
EXPECTED_ARM = "strategy"
EXPECTED_STEP = 76
EXPECTED_PLAN = "nested_topology"
EXPECTED_EXECUTION_SEED = 700211
EXPECTED_CASE = "turbulent_combustion"
EXPECTED_STOP_EPOCH = 2.0
NUMERIC_REL_TOL = 1e-6
NUMERIC_ABS_TOL = 1e-10
STATE_KEYS = (
    "model",
    "optimizer",
    "family_states",
    "external_policy",
    "execution_strategy",
    "rng_state",
    "component_controller",
    "family_scalarizer",
    "normalization",
)


class RecoveryAuditError(RuntimeError):
    """Raised when an artifact cannot support the requested recovery audit."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RecoveryAuditError("artifact metadata is not finite JSON data") from error


def _feed(hasher: Any, tag: bytes, payload: bytes = b"") -> None:
    hasher.update(len(tag).to_bytes(4, "big"))
    hasher.update(tag)
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def tensor_tree_sha256(value: Any) -> str:
    """Hash a checkpoint's nested tensor/scalar state deterministically."""

    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if item is None:
            _feed(digest, b"none")
        elif isinstance(item, bool):
            _feed(digest, b"bool", b"1" if item else b"0")
        elif isinstance(item, int):
            _feed(digest, b"int", str(item).encode("ascii"))
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise RecoveryAuditError("checkpoint contains a non-finite scalar")
            _feed(digest, b"float", item.hex().encode("ascii"))
        elif isinstance(item, str):
            _feed(digest, b"str", item.encode("utf-8"))
        elif isinstance(item, bytes):
            _feed(digest, b"bytes", item)
        elif isinstance(item, Mapping):
            _feed(digest, b"mapping-start", str(len(item)).encode("ascii"))
            ordered = sorted(item.items(), key=lambda pair: (type(pair[0]).__qualname__, repr(pair[0])))
            for key, child in ordered:
                visit(key)
                visit(child)
            _feed(digest, b"mapping-end")
        elif isinstance(item, tuple):
            _feed(digest, b"tuple-start", str(len(item)).encode("ascii"))
            for child in item:
                visit(child)
            _feed(digest, b"tuple-end")
        elif isinstance(item, list):
            _feed(digest, b"list-start", str(len(item)).encode("ascii"))
            for child in item:
                visit(child)
            _feed(digest, b"list-end")
        else:
            # Import lazily so simple report/test operations need not initialize
            # Torch. The only supported object beyond JSON primitives is Tensor.
            try:
                import torch
            except ImportError as error:  # pragma: no cover - environment failure
                raise RecoveryAuditError("Torch is required to digest checkpoint tensors") from error
            if not isinstance(item, torch.Tensor):
                raise RecoveryAuditError(
                    f"unsupported checkpoint state value: {type(item).__qualname__}"
                )
            _feed(digest, b"tensor-meta", _canonical_json({
                "dtype": str(item.dtype),
                "shape": list(item.shape),
                "layout": str(item.layout),
                "quantized": bool(item.is_quantized),
            }))
            tensor = item.detach().cpu()
            if tensor.is_quantized:
                visit(tensor.int_repr())
                _feed(digest, b"quantization", _canonical_json({
                    "qscheme": str(tensor.qscheme()),
                    "scale": float(tensor.q_scale()) if tensor.qscheme() in (
                        torch.per_tensor_affine, torch.per_tensor_symmetric,
                    ) else None,
                    "zero_point": int(tensor.q_zero_point()) if tensor.qscheme() in (
                        torch.per_tensor_affine, torch.per_tensor_symmetric,
                    ) else None,
                }))
            elif tensor.layout == torch.sparse_coo:
                sparse = tensor.coalesce()
                visit(sparse.indices())
                visit(sparse.values())
            elif tensor.layout == torch.strided:
                raw = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                _feed(digest, b"tensor-data", raw)
            else:
                raise RecoveryAuditError(f"unsupported checkpoint tensor layout: {tensor.layout}")

    visit(value)
    return digest.hexdigest()


def _load_checkpoint_bytes_cpu(payload_bytes: bytes, label: str) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment failure
        raise RecoveryAuditError("Torch is required to inspect recovery checkpoints") from error
    try:
        payload = torch.load(io.BytesIO(payload_bytes), map_location="cpu", weights_only=True)
    except Exception as error:
        raise RecoveryAuditError(
            f"restricted CPU checkpoint load failed for {label}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise RecoveryAuditError("checkpoint payload must be a mapping")
    return payload


def _read_checkpoint_bytes_stably(path: Path) -> tuple[bytes, dict[str, int]]:
    """Read one atomic checkpoint inode and reject replacement during inspection."""

    try:
        with path.open("rb") as handle:
            before = handle.fileno()
            stat_before = os.fstat(before)
            data = handle.read()
            stat_after = os.fstat(before)
        current = path.stat()
    except OSError as error:
        raise RecoveryAuditError(f"cannot read checkpoint bytes at {path}: {error}") from error
    identity_before = (stat_before.st_dev, stat_before.st_ino)
    identity_after = (stat_after.st_dev, stat_after.st_ino)
    current_identity = (current.st_dev, current.st_ino)
    if (
        identity_before != identity_after
        or identity_before != current_identity
        or stat_before.st_size != stat_after.st_size
        or len(data) != stat_after.st_size
    ):
        raise RecoveryAuditError("rolling last.pt changed during byte capture; retry at a stable cursor")
    return data, {
        "device": int(stat_after.st_dev),
        "inode": int(stat_after.st_ino),
        "size_bytes": int(stat_after.st_size),
        "mtime_ns": int(stat_after.st_mtime_ns),
    }


def _assert_checkpoint_path_unchanged(path: Path, binding: Mapping[str, Any]) -> None:
    try:
        current = path.stat()
    except OSError as error:
        raise RecoveryAuditError(f"checkpoint path disappeared during capture: {path}") from error
    if (
        int(current.st_dev) != binding.get("device")
        or int(current.st_ino) != binding.get("inode")
        or int(current.st_size) != binding.get("size_bytes")
        or int(current.st_mtime_ns) != binding.get("mtime_ns")
    ):
        raise RecoveryAuditError("rolling last.pt changed while its bytes were being inspected")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RecoveryAuditError(f"cannot read {label} JSON at {path}: {error}") from error
    if not isinstance(value, dict):
        raise RecoveryAuditError(f"{label} must be a JSON object")
    return value


def _read_ledger(path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise RecoveryAuditError(f"cannot read campaign runner ledger {path}: {error}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RecoveryAuditError(
                f"malformed campaign ledger line {line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise RecoveryAuditError(f"campaign ledger line {line_number} is not an object")
        row = dict(row)
        row["_line_number"] = line_number
        rows.append(row)
    return rows, _sha256_bytes(text.encode("utf-8"))


def _ledger_attempts(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    attempts: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        attempt_id = row.get("attempt_id")
        if isinstance(attempt_id, str) and attempt_id:
            attempts.setdefault(attempt_id, []).append(row)
    return attempts


def _terminal_ledger_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in reversed(rows):
        if row.get("status") not in (None, "running"):
            return row
    return None


def _row_sha256(row: Mapping[str, Any]) -> str:
    clean = {key: value for key, value in row.items() if key != "_line_number"}
    return _sha256_bytes(_canonical_json(clean))


def _utc_seconds(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _campaign_root(run_dir: Path) -> Path:
    if len(run_dir.parents) < 4:
        raise RecoveryAuditError("run path is too shallow to identify its campaign root")
    root = run_dir.parents[3]
    if run_dir.parent.name != EXPECTED_ARM or run_dir.parent.parent.name != EXPECTED_TASK:
        raise RecoveryAuditError(
            f"probe must use runs/{EXPECTED_TASK}/{EXPECTED_ARM}/<run>; got {run_dir}"
        )
    return root


def _campaign_source_bindings(
    run_dir: Path,
    manifest: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    decisions: Mapping[str, Any],
) -> dict[str, Any]:
    root = _campaign_root(run_dir)
    source_binding_path = root / "manifests/science_source_binding_v1.json"
    task_contract_path = root / "tasks/D_task_contract.json"
    strategy_config_path = root / "configs/D_strategy.yaml"
    source_binding = _read_json(source_binding_path, "frozen source snapshot manifest")
    task_contract = _read_json(task_contract_path, "D task contract")
    if not strategy_config_path.is_file():
        raise RecoveryAuditError(f"missing frozen D strategy config: {strategy_config_path}")
    expected_experiment = f"_CostAwareLPC/{root.name}/D/strategy"
    source_bindings = checkpoint.get("source_config_bindings")
    if not isinstance(source_bindings, Mapping):
        raise RecoveryAuditError("checkpoint snapshot has no SOURCE/config bindings")
    source_hashes = source_bindings.get("source_hashes")
    runtime_identity = checkpoint.get("runtime_identity")
    if not isinstance(source_hashes, Mapping):
        raise RecoveryAuditError("checkpoint has no SOURCE hash mapping")
    expected_config_hash = source_binding.get("configs", {}).get("configs/D_strategy.yaml")
    source_snapshot = source_binding.get("source_snapshot")
    checks = {
        "manifest_task_case": manifest.get("case") == EXPECTED_CASE,
        "experiment_name": manifest.get("experiment_name") == expected_experiment,
        "task_contract_id": task_contract.get("task_id") == EXPECTED_TASK,
        "task_contract_case": task_contract.get("case_id") == EXPECTED_CASE,
        "task_hash": task_contract.get("task_hash") == decisions["76"].get("task_hash"),
        "runtime_task_hash": isinstance(runtime_identity, Mapping)
        and runtime_identity.get("task_hash") == task_contract.get("task_hash"),
        "source_checkpoint_identity": (
            task_contract.get("source_identity") == source_hashes.get("checkpoint")
        ),
        "source_hashes_identity": (
            task_contract.get("metadata", {}).get("source_hashes") == dict(source_hashes)
        ),
        "source_parent_identity": (
            Path(str(manifest.get("parent_run", ""))).resolve()
            == Path(str(source_bindings.get("source_run", ""))).resolve()
        ),
        "frozen_strategy_config": expected_config_hash == _sha256_file(strategy_config_path),
        "source_snapshot_declared": isinstance(source_snapshot, str) and bool(source_snapshot),
    }
    if not all(checks.values()):
        failed = [key for key, matched in checks.items() if not matched]
        raise RecoveryAuditError(
            "campaign/source snapshot bindings do not match the D strategy run: "
            + ", ".join(failed)
        )
    return {
        "campaign_root": str(root.resolve()),
        "task_id": EXPECTED_TASK,
        "case_id": EXPECTED_CASE,
        "experiment_name": expected_experiment,
        "source_identity": task_contract["source_identity"],
        "task_hash": task_contract["task_hash"],
        "source_snapshot": source_snapshot,
        "task_contract_sha256": _sha256_file(task_contract_path),
        "source_binding_manifest_sha256": _sha256_file(source_binding_path),
        "strategy_config_sha256": _sha256_file(strategy_config_path),
        "checks": checks,
    }


def _read_jsonl_records(path: Path) -> tuple[list[tuple[str, dict[str, Any]]], str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise RecoveryAuditError(f"cannot read update log {path}: {error}") from error
    records: list[tuple[str, dict[str, Any]]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RecoveryAuditError(
                f"malformed update log line {line_number} in {path}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise RecoveryAuditError(f"update log line {line_number} is not an object")
        records.append((line, record))
    return records, _sha256_bytes(text.encode("utf-8"))


def _manifest_identity(manifest: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    identity = {
        key: manifest.get(key)
        for key in ("stage", "case", "experiment_name", "run_id", "parent_run", "config_sha256")
    }
    identity["run_dir"] = str(run_dir.resolve())
    return identity


def _runtime_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    runtime = payload.get("execution_strategy")
    if not isinstance(runtime, Mapping):
        raise RecoveryAuditError("checkpoint is missing execution_strategy state")
    execution_state = runtime.get("execution_state")
    plan = runtime.get("current_plan")
    if not isinstance(execution_state, Mapping) or not isinstance(plan, Mapping):
        raise RecoveryAuditError("checkpoint execution strategy lacks plan or cursor state")
    return {
        "task_hash": runtime.get("task_hash"),
        "profile_identity": runtime.get("profile_identity"),
        "profiled_epoch": runtime.get("profiled_epoch"),
        "rng_seed": execution_state.get("rng_seed"),
        "plan_id": plan.get("plan_id"),
        "state_plan_id": execution_state.get("plan_id"),
        "accepted_cursor": execution_state.get("accepted_cursor"),
        "execution_rng_identity": execution_state.get("rng_identity"),
    }


def _checkpoint_snapshot(
    run_dir: Path,
    *,
    boundary: int | None = None,
    checkpoint_copy_path: Path | None = None,
) -> dict[str, Any]:
    checkpoint_path = run_dir / "checkpoints/last.pt"
    manifest_path = run_dir / "run_manifest.json"
    config_path = run_dir / "resolved_config.yaml"
    for path, label in (
        (checkpoint_path, "last checkpoint"),
        (manifest_path, "run manifest"),
        (config_path, "resolved config"),
    ):
        if not path.is_file():
            raise RecoveryAuditError(f"missing {label}: {path}")
    manifest = _read_json(manifest_path, "run manifest")
    checkpoint_bytes, stat_binding = _read_checkpoint_bytes_stably(checkpoint_path)
    checkpoint_sha = _sha256_bytes(checkpoint_bytes)
    payload = _load_checkpoint_bytes_cpu(checkpoint_bytes, str(checkpoint_path))
    _assert_checkpoint_path_unchanged(checkpoint_path, stat_binding)
    if not isinstance(payload.get("source_hashes"), Mapping) or not payload["source_hashes"]:
        raise RecoveryAuditError("checkpoint is missing immutable SOURCE hashes")
    if not isinstance(payload.get("config_sha256"), str) or not payload["config_sha256"]:
        raise RecoveryAuditError("checkpoint is missing its resolved configuration digest")
    if manifest.get("config_sha256") != payload.get("config_sha256"):
        raise RecoveryAuditError("run manifest and checkpoint configuration digests differ")
    if not isinstance(payload.get("execution_strategy_implementation"), Mapping):
        raise RecoveryAuditError("checkpoint is missing execution strategy implementation hashes")
    global_step = payload.get("global_step")
    if isinstance(global_step, bool) or not isinstance(global_step, int):
        raise RecoveryAuditError("checkpoint global_step must be an integer")
    runtime = payload.get("execution_strategy")
    if not isinstance(runtime, Mapping):
        raise RecoveryAuditError("checkpoint is missing execution_strategy state")
    execution_state = runtime.get("execution_state")
    if not isinstance(execution_state, Mapping):
        raise RecoveryAuditError("checkpoint is missing execution_state")
    if boundary is not None:
        cursor = execution_state.get("accepted_cursor")
        if global_step != boundary or cursor != boundary:
            raise RecoveryAuditError(
                f"planned capture requires global_step and accepted_cursor {boundary}; "
                f"observed global_step={global_step}, accepted_cursor={cursor}"
            )
    checkpoint_copy = None
    if checkpoint_copy_path is not None:
        checkpoint_copy_path = checkpoint_copy_path.resolve()
        checkpoint_copy_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with checkpoint_copy_path.open("xb") as handle:
                handle.write(checkpoint_bytes)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as error:
            raise RecoveryAuditError(
                f"refusing to overwrite existing checkpoint copy: {checkpoint_copy_path}"
            ) from error
        copied_sha = _sha256_file(checkpoint_copy_path)
        if copied_sha != checkpoint_sha:
            raise RecoveryAuditError("preserved checkpoint copy differs from the inspected bytes")
        checkpoint_copy = {
            "path": str(checkpoint_copy_path),
            "sha256": copied_sha,
            "size_bytes": len(checkpoint_bytes),
        }
    state_digests = {
        key: tensor_tree_sha256(payload[key])
        for key in STATE_KEYS
        if key in payload
    }
    source_config_bindings = {
        key: payload.get(key)
        for key in (
            "source_run", "source_checkpoint", "source_hashes", "config_sha256",
            "coherence_calibration_sha256", "family_scales", "training_subset_sha256",
        )
        if key in payload
    }
    return {
        "path": str(checkpoint_path.resolve()),
        "sha256": checkpoint_sha,
        "stat_binding": stat_binding,
        "byte_copy": checkpoint_copy,
        "global_step": global_step,
        "payload_sha256": tensor_tree_sha256(payload),
        "state_digests": state_digests,
        "source_config_bindings": source_config_bindings,
        "runtime_identity": _runtime_identity(payload),
        "runtime_implementation": payload.get("execution_strategy_implementation"),
        "manifest_identity": _manifest_identity(manifest, run_dir),
        "manifest_sha256": _sha256_file(manifest_path),
        "resolved_config_sha256": _sha256_file(config_path),
    }


def _load_protocol(protocol_path: Path, decisions_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _read_json(protocol_path, "recovery protocol")
    decisions = _read_json(decisions_path, "counter decisions")
    if protocol.get("schema_version") != 1:
        raise RecoveryAuditError("unsupported recovery protocol schema")
    if (
        protocol.get("task") != EXPECTED_TASK
        or protocol.get("arm") != EXPECTED_ARM
        or protocol.get("checkpoint_step") != EXPECTED_STEP
    ):
        raise RecoveryAuditError("protocol does not describe the predeclared D/strategy step-76 probe")
    for cursor in ("76", "77", "78"):
        if cursor not in decisions:
            raise RecoveryAuditError(f"counter decision mapping is missing cursor {cursor}")
    return protocol, decisions


def _validate_run_identity(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    resolved = run_dir.resolve()
    if resolved.parent.name != EXPECTED_ARM or resolved.parent.parent.name != EXPECTED_TASK:
        raise RecoveryAuditError(
            f"probe must use runs/{EXPECTED_TASK}/{EXPECTED_ARM}/<run>; got {resolved}"
        )
    if manifest.get("case") != EXPECTED_CASE or manifest.get("stage") != "post_training":
        raise RecoveryAuditError("run manifest is not a D post-training lineage")
    if manifest.get("run_id") != resolved.name:
        raise RecoveryAuditError("run manifest run_id differs from the run directory")
    root = _campaign_root(resolved)
    expected_experiment = f"_CostAwareLPC/{root.name}/D/strategy"
    if manifest.get("experiment_name") != expected_experiment:
        raise RecoveryAuditError("run manifest experiment_name does not bind D/strategy")


def _unique_update_rows(
    run_dir: Path, steps: tuple[int, ...]
) -> tuple[dict[int, tuple[str, dict[str, Any]]], str]:
    path = run_dir / UPDATE_LOG
    records, log_sha = _read_jsonl_records(path)
    found: dict[int, tuple[str, dict[str, Any]]] = {}
    for raw_line, record in records:
        step = record.get("step")
        if isinstance(step, bool) or not isinstance(step, int):
            raise RecoveryAuditError("execution strategy update row has no integer step")
        if step not in steps:
            continue
        if step in found:
            raise RecoveryAuditError(f"execution update log contains duplicate step {step}")
        if not isinstance(record.get("execution"), Mapping):
            raise RecoveryAuditError(f"execution update row {step} lacks its decision mapping")
        found[step] = (raw_line, record)
    missing = sorted(set(steps) - set(found))
    if missing:
        raise RecoveryAuditError(f"execution update log is missing rows {missing}")
    return found, log_sha


def _matches_expected_execution(record: Mapping[str, Any], decision: Mapping[str, Any]) -> bool:
    return record.get("execution") == decision


def _active_launch_attempt(ledger_path: Path) -> dict[str, Any]:
    rows, ledger_sha = _read_ledger(ledger_path)
    attempts = _ledger_attempts(rows)
    active: list[tuple[str, list[dict[str, Any]]]] = []
    for attempt_id, attempt_rows in attempts.items():
        first = attempt_rows[0]
        last = attempt_rows[-1]
        if (
            first.get("task_id") == EXPECTED_TASK
            and first.get("arm") == EXPECTED_ARM
            and first.get("action") == "launch"
            and first.get("status") == "running"
            and last.get("status") == "running"
        ):
            active.append((attempt_id, attempt_rows))
    if len(active) != 1:
        raise RecoveryAuditError(
            f"expected exactly one active D/strategy launch attempt in {ledger_path}; found {len(active)}"
        )
    attempt_id, attempt_rows = active[0]
    start_row = attempt_rows[0]
    return {
        "attempt_id": attempt_id,
        "action": "launch",
        "status_at_capture": "running",
        "started_utc": start_row.get("started_utc"),
        "command_sha256": start_row.get("command_sha256"),
        "ledger_path": str(ledger_path.resolve()),
        "ledger_sha256_at_capture": ledger_sha,
    }


def _validate_stopped_launch(
    capture: Mapping[str, Any], run_dir: Path, ledger_path: Path
) -> tuple[dict[str, Any], str]:
    rows, ledger_sha = _read_ledger(ledger_path)
    launch = capture.get("runner_launch")
    if not isinstance(launch, Mapping) or not isinstance(launch.get("attempt_id"), str):
        raise RecoveryAuditError("capture does not bind an initial runner launch attempt")
    if Path(str(launch.get("ledger_path", ""))).resolve() != ledger_path.resolve():
        raise RecoveryAuditError("capture and stop check do not use the same campaign ledger")
    attempt_rows = _ledger_attempts(rows).get(str(launch["attempt_id"]), [])
    terminal = _terminal_ledger_row(attempt_rows)
    if (
        not attempt_rows
        or attempt_rows[0].get("action") != "launch"
        or attempt_rows[0].get("task_id") != EXPECTED_TASK
        or attempt_rows[0].get("arm") != EXPECTED_ARM
        or attempt_rows[0].get("command_sha256") != launch.get("command_sha256")
        or attempt_rows[0].get("started_utc") != launch.get("started_utc")
        or terminal is None
        or terminal.get("status") != "failed_or_stopped"
    ):
        raise RecoveryAuditError("initial D/strategy launch lacks a stopped terminal ledger receipt")
    observation = terminal.get("actual_science_epoch_observation")
    if not isinstance(observation, Mapping):
        raise RecoveryAuditError("stopped launch receipt lacks accepted-checkpoint observation")
    if (
        Path(str(observation.get("run_dir", ""))).resolve() != run_dir.resolve()
        or not math.isclose(
            float(observation.get("accepted_end_epoch", -1.0)),
            EXPECTED_STOP_EPOCH,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RecoveryAuditError("stopped launch receipt does not bind this run at accepted epoch 2")
    if not isinstance(terminal.get("finished_utc"), str) or terminal.get("return_code") in (None, 0):
        raise RecoveryAuditError("stopped launch receipt lacks a terminal nonzero process result")
    return terminal, ledger_sha


def _resume_attempt_evidence(
    rows: list[dict[str, Any]],
    *,
    launch_terminal: Mapping[str, Any],
    run_dir: Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    issues: list[str] = []
    stop_finished = _utc_seconds(launch_terminal.get("finished_utc"))
    stop_line = int(launch_terminal.get("_line_number", -1))
    attempts = _ledger_attempts(rows)
    candidates: list[list[dict[str, Any]]] = []
    for attempt_rows in attempts.values():
        start = attempt_rows[0]
        if (
            start.get("task_id") == EXPECTED_TASK
            and start.get("arm") == EXPECTED_ARM
            and start.get("action") == "resume"
            and int(start.get("_line_number", -1)) > stop_line
        ):
            candidates.append(attempt_rows)
    candidates.sort(key=lambda items: int(items[0].get("_line_number", -1)))
    for attempt_rows in candidates:
        start = attempt_rows[0]
        terminal = _terminal_ledger_row(attempt_rows)
        if start.get("status") != "running" or terminal is None:
            continue
        if terminal.get("status") not in {"completed", "failed_or_stopped"}:
            continue
        command = start.get("command")
        if not isinstance(command, list):
            issues.append("resume command is not preserved as an argument list")
            continue
        resume_index = next(
            (index for index, value in enumerate(command) if value == "--resume"), None
        )
        if resume_index is None or resume_index + 1 >= len(command):
            issues.append("resume command does not include --resume <run_dir>")
            continue
        if Path(str(command[resume_index + 1])).resolve() != run_dir.resolve():
            issues.append("resume command points to a different run directory")
            continue
        observation = terminal.get("actual_science_epoch_observation")
        if not isinstance(observation, Mapping):
            issues.append("resume terminal receipt has no run observation")
            continue
        if Path(str(observation.get("run_dir", ""))).resolve() != run_dir.resolve():
            issues.append("resume terminal observation points to a different run")
            continue
        try:
            start_epoch = float(observation.get("start_epoch"))
        except (TypeError, ValueError):
            start_epoch = math.nan
        if not math.isclose(start_epoch, EXPECTED_STOP_EPOCH, rel_tol=0.0, abs_tol=1e-12):
            issues.append("resume did not start from accepted epoch 2")
            continue
        resume_started = _utc_seconds(start.get("started_utc"))
        if stop_finished is None or resume_started is None or resume_started <= stop_finished:
            issues.append("resume launch did not occur after the stopped launch terminal receipt")
            continue
        log_path = run_dir / UPDATE_LOG
        try:
            log_mtime = log_path.stat().st_mtime
        except OSError:
            issues.append("resumed execution update log is missing")
            continue
        if log_mtime < resume_started:
            issues.append("execution update log was not rewritten after the resume launch")
            continue
        return {
            "attempt_id": start.get("attempt_id"),
            "started_utc": start.get("started_utc"),
            "terminal_status": terminal.get("status"),
            "finished_utc": terminal.get("finished_utc"),
            "start_epoch": start_epoch,
            "run_dir": str(run_dir.resolve()),
            "start_line_number": start.get("_line_number"),
            "terminal_line_number": terminal.get("_line_number"),
            "terminal_row_sha256": _row_sha256(terminal),
            "update_log_mtime_utc_seconds": log_mtime,
            "terminal_observation": observation,
        }, issues
    if not candidates:
        issues.append("no D/strategy resume attempt follows the stopped launch")
    return None, issues


def capture_probe(
    run_dir: str | Path,
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL,
    decisions_path: str | Path = DEFAULT_DECISIONS,
    checkpoint_copy_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    """Preserve the exact step-76 checkpoint and speculative 77/78 log lines."""

    run_path = Path(run_dir).resolve()
    protocol_file = Path(protocol_path).resolve()
    decisions_file = Path(decisions_path).resolve()
    protocol, decisions = _load_protocol(protocol_file, decisions_file)
    manifest = _read_json(run_path / "run_manifest.json", "run manifest")
    _validate_run_identity(run_path, manifest)
    campaign_root = _campaign_root(run_path)
    ledger_file = (
        Path(ledger_path).resolve()
        if ledger_path is not None
        else campaign_root / "manifests/campaign_runner_ledger.jsonl"
    )
    copy_path = (
        Path(checkpoint_copy_path)
        if checkpoint_copy_path is not None
        else run_path / "checkpoints/last_step76_recovery_probe.pt"
    )
    checkpoint = _checkpoint_snapshot(
        run_path, boundary=EXPECTED_STEP, checkpoint_copy_path=copy_path
    )
    checkpoint["campaign_source_bindings"] = _campaign_source_bindings(
        run_path, manifest, checkpoint, decisions
    )
    runtime = checkpoint["runtime_identity"]
    if (
        runtime.get("plan_id") != EXPECTED_PLAN
        or runtime.get("state_plan_id") != EXPECTED_PLAN
        or runtime.get("task_hash") != decisions["76"].get("task_hash")
        or runtime.get("rng_seed") != EXPECTED_EXECUTION_SEED
        or runtime.get("accepted_cursor") != EXPECTED_STEP
    ):
        raise RecoveryAuditError("step-76 checkpoint does not match the predeclared runtime binding")
    rows, log_sha = _unique_update_rows(run_path, (77, 78))
    runner_launch = _active_launch_attempt(ledger_file)
    speculative: dict[str, Any] = {}
    for step, cursor in ((77, "76"), (78, "77")):
        raw_line, record = rows[step]
        if not _matches_expected_execution(record, decisions[cursor]):
            raise RecoveryAuditError(
                f"speculative update {step} does not carry predeclared counter decision {cursor}"
            )
        speculative[str(step)] = {
            "decision_cursor": int(cursor),
            "line_sha256": _sha256_bytes(raw_line.encode("utf-8")),
            "raw_jsonl_line": raw_line,
        }
    return {
        "schema_version": 1,
        "status": "pending_replay",
        "task_id": EXPECTED_TASK,
        "arm": EXPECTED_ARM,
        "checkpoint_step": EXPECTED_STEP,
        "protocol_path": str(protocol_file),
        "protocol_sha256": _sha256_file(protocol_file),
        "expected_decisions_path": str(decisions_file),
        "expected_decisions_sha256": _sha256_file(decisions_file),
        "tolerance": protocol.get("tolerance"),
        "run_identity": checkpoint["manifest_identity"],
        "checkpoint": checkpoint,
        "runner_launch": runner_launch,
        "speculative_update_log": {
            "path": str((run_path / UPDATE_LOG).resolve()),
            "sha256_at_capture": log_sha,
            "rows": speculative,
        },
    }


def confirm_stopped_probe(
    capture: Mapping[str, Any],
    run_dir: str | Path,
    *,
    ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    """Confirm stop occurred at the captured step-76 checkpoint, before resume."""

    if capture.get("schema_version") != 1 or capture.get("status") != "pending_replay":
        raise RecoveryAuditError("input is not a pending recovery capture")
    run_path = Path(run_dir).resolve()
    campaign_root = _campaign_root(run_path)
    ledger_file = (
        Path(ledger_path).resolve()
        if ledger_path is not None
        else campaign_root / "manifests/campaign_runner_ledger.jsonl"
    )
    before = capture.get("checkpoint")
    if not isinstance(before, Mapping):
        raise RecoveryAuditError("capture has no checkpoint state bindings")
    copy_binding = before.get("byte_copy")
    if not isinstance(copy_binding, Mapping):
        raise RecoveryAuditError("capture did not preserve a checkpoint byte copy")
    copy_path = Path(str(copy_binding.get("path", "")))
    if not copy_path.is_file() or _sha256_file(copy_path) != before.get("sha256"):
        raise RecoveryAuditError("preserved checkpoint byte copy is missing or has changed")
    manifest = _read_json(run_path / "run_manifest.json", "run manifest")
    _validate_run_identity(run_path, manifest)
    stopped_checkpoint = _checkpoint_snapshot(run_path, boundary=EXPECTED_STEP)
    stopped_checkpoint["campaign_source_bindings"] = _campaign_source_bindings(
        run_path, manifest, stopped_checkpoint, {
            "76": {"task_hash": before.get("runtime_identity", {}).get("task_hash")}
        }
    )
    checkpoint_matches = (
        stopped_checkpoint.get("sha256") == before.get("sha256")
        and stopped_checkpoint.get("payload_sha256") == before.get("payload_sha256")
        and stopped_checkpoint.get("state_digests") == before.get("state_digests")
        and stopped_checkpoint.get("campaign_source_bindings")
        == before.get("campaign_source_bindings")
    )
    if not checkpoint_matches:
        raise RecoveryAuditError("post-stop last.pt is not the byte-identical captured step-76 checkpoint")
    terminal, ledger_sha = _validate_stopped_launch(capture, run_path, ledger_file)
    return {
        "schema_version": 1,
        "status": "stopped_checkpoint_verified",
        "task_id": EXPECTED_TASK,
        "arm": EXPECTED_ARM,
        "checkpoint_step": EXPECTED_STEP,
        "capture_checkpoint_sha256": before.get("sha256"),
        "preserved_checkpoint_copy_sha256": copy_binding.get("sha256"),
        "stopped_last_checkpoint_sha256": stopped_checkpoint.get("sha256"),
        "stopped_last_checkpoint_payload_sha256": stopped_checkpoint.get("payload_sha256"),
        "run_identity": capture.get("run_identity"),
        "runner_ledger_path": str(ledger_file.resolve()),
        "runner_ledger_sha256_at_stop": ledger_sha,
        "launch_attempt_id": capture.get("runner_launch", {}).get("attempt_id"),
        "launch_terminal_row_sha256": _row_sha256(terminal),
        "launch_terminal_status": terminal.get("status"),
        "launch_return_code": terminal.get("return_code"),
        "launch_finished_utc": terminal.get("finished_utc"),
        "accepted_end_epoch_at_stop": terminal.get("actual_science_epoch_observation", {}).get(
            "accepted_end_epoch"
        ),
        "checkpoint_sha256_still_step76": True,
        "state_digests_match_capture": True,
        "campaign_source_bindings": stopped_checkpoint.get("campaign_source_bindings"),
    }


def _is_clock_key(key: str) -> bool:
    name = key.lower()
    return any(
        marker in name
        for marker in (
            "second", "elapsed", "duration", "timestamp", "_utc", "started", "finished",
            "perf_counter", "wall_time", "update_time",
        )
    )


def _compare_values(
    before: Any,
    after: Any,
    *,
    path: str,
    errors: list[str],
    numeric_errors: list[dict[str, Any]],
    rel_tol: float = NUMERIC_REL_TOL,
    abs_tol: float = NUMERIC_ABS_TOL,
) -> None:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        before_keys = {key for key in before if not (isinstance(key, str) and _is_clock_key(key))}
        after_keys = {key for key in after if not (isinstance(key, str) and _is_clock_key(key))}
        if before_keys != after_keys:
            errors.append(f"{path}: field sets differ")
            return
        for key in sorted(before_keys, key=lambda item: str(item)):
            _compare_values(
                before[key], after[key], path=f"{path}.{key}", errors=errors,
                numeric_errors=numeric_errors, rel_tol=rel_tol, abs_tol=abs_tol,
            )
        return
    if isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            errors.append(f"{path}: list lengths differ")
            return
        for index, (left, right) in enumerate(zip(before, after, strict=True)):
            _compare_values(
                left, right, path=f"{path}[{index}]", errors=errors,
                numeric_errors=numeric_errors, rel_tol=rel_tol, abs_tol=abs_tol,
            )
        return
    if isinstance(before, bool) or isinstance(after, bool):
        if type(before) is not type(after) or before != after:
            errors.append(f"{path}: exact identity/value differs")
        return
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        if isinstance(before, int) and isinstance(after, int):
            if before != after:
                errors.append(f"{path}: integer identity/count differs")
            return
        left = float(before)
        right = float(after)
        if not math.isfinite(left) or not math.isfinite(right):
            errors.append(f"{path}: non-finite diagnostic")
            return
        if not math.isclose(left, right, rel_tol=rel_tol, abs_tol=abs_tol):
            numeric_errors.append({
                "path": path,
                "before": left,
                "after": right,
                "absolute_error": abs(left - right),
                "relative_error": abs(left - right) / max(abs(left), abs(right), abs_tol),
            })
        return
    if type(before) is not type(after) or before != after:
        errors.append(f"{path}: exact identity/value differs")


def _record_from_capture(capture: Mapping[str, Any], step: int) -> dict[str, Any]:
    rows = capture.get("speculative_update_log", {}).get("rows", {})
    entry = rows.get(str(step))
    if not isinstance(entry, Mapping) or not isinstance(entry.get("raw_jsonl_line"), str):
        raise RecoveryAuditError(f"capture does not preserve speculative update {step}")
    raw = entry["raw_jsonl_line"]
    if _sha256_bytes(raw.encode("utf-8")) != entry.get("line_sha256"):
        raise RecoveryAuditError(f"captured update {step} raw-line SHA-256 does not match")
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RecoveryAuditError(f"captured update {step} is malformed: {error}") from error
    if not isinstance(record, dict) or record.get("step") != step:
        raise RecoveryAuditError(f"captured update {step} has a mismatched step identity")
    return record


def _source_config_bindings_unchanged(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    return (
        before.get("source_config_bindings") == after.get("source_config_bindings")
        and before.get("campaign_source_bindings") == after.get("campaign_source_bindings")
        and before.get("resolved_config_sha256") == after.get("resolved_config_sha256")
        and before.get("manifest_identity", {}).get("config_sha256")
        == after.get("manifest_identity", {}).get("config_sha256")
    )


def _runtime_binding_unchanged(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    left = before.get("runtime_identity", {})
    right = after.get("runtime_identity", {})
    stable_keys = ("task_hash", "profile_identity", "profiled_epoch", "rng_seed")
    return (
        before.get("runtime_implementation") == after.get("runtime_implementation")
        and all(left.get(key) == right.get(key) for key in stable_keys)
    )


def compare_probe(
    capture: Mapping[str, Any],
    run_dir: str | Path,
    *,
    stop_receipt: Mapping[str, Any],
    protocol_path: str | Path = DEFAULT_PROTOCOL,
    decisions_path: str | Path = DEFAULT_DECISIONS,
    ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare replayed rows and bindings; only this function can verify."""

    if capture.get("schema_version") != 1 or capture.get("status") != "pending_replay":
        raise RecoveryAuditError("input is not a pending step-76 recovery capture")
    run_path = Path(run_dir).resolve()
    protocol_file = Path(protocol_path).resolve()
    decisions_file = Path(decisions_path).resolve()
    protocol, decisions = _load_protocol(protocol_file, decisions_file)
    if capture.get("protocol_sha256") != _sha256_file(protocol_file):
        raise RecoveryAuditError("recovery protocol changed after capture")
    if capture.get("expected_decisions_sha256") != _sha256_file(decisions_file):
        raise RecoveryAuditError("expected counter decisions changed after capture")
    manifest = _read_json(run_path / "run_manifest.json", "run manifest")
    _validate_run_identity(run_path, manifest)
    after_checkpoint = _checkpoint_snapshot(run_path)
    campaign_binding_error = None
    try:
        after_checkpoint["campaign_source_bindings"] = _campaign_source_bindings(
            run_path, manifest, after_checkpoint, decisions
        )
    except RecoveryAuditError as error:
        after_checkpoint["campaign_source_bindings"] = None
        campaign_binding_error = str(error)
    before_checkpoint = capture.get("checkpoint")
    if not isinstance(before_checkpoint, Mapping):
        raise RecoveryAuditError("capture has no checkpoint bindings")

    before_identity = capture.get("run_identity")
    after_identity = after_checkpoint.get("manifest_identity")
    same_manifest_lineage = isinstance(before_identity, Mapping) and all(
        before_identity.get(key) == after_identity.get(key)
        for key in ("stage", "case", "experiment_name", "run_id", "parent_run", "config_sha256", "run_dir")
    )
    campaign_root = _campaign_root(run_path)
    ledger_file = (
        Path(ledger_path).resolve()
        if ledger_path is not None
        else campaign_root / "manifests/campaign_runner_ledger.jsonl"
    )
    ledger_rows, ledger_sha = _read_ledger(ledger_file)
    stopped_launch, _stop_ledger_sha = _validate_stopped_launch(capture, run_path, ledger_file)
    stop_receipt_matches = (
        stop_receipt.get("schema_version") == 1
        and stop_receipt.get("status") == "stopped_checkpoint_verified"
        and stop_receipt.get("task_id") == EXPECTED_TASK
        and stop_receipt.get("arm") == EXPECTED_ARM
        and stop_receipt.get("checkpoint_step") == EXPECTED_STEP
        and stop_receipt.get("run_identity") == before_identity
        and Path(str(stop_receipt.get("runner_ledger_path", ""))).resolve()
        == ledger_file.resolve()
        and stop_receipt.get("capture_checkpoint_sha256") == before_checkpoint.get("sha256")
        and stop_receipt.get("stopped_last_checkpoint_sha256") == before_checkpoint.get("sha256")
        and stop_receipt.get("stopped_last_checkpoint_payload_sha256")
        == before_checkpoint.get("payload_sha256")
        and stop_receipt.get("campaign_source_bindings")
        == before_checkpoint.get("campaign_source_bindings")
        and stop_receipt.get("launch_attempt_id")
        == capture.get("runner_launch", {}).get("attempt_id")
        and stop_receipt.get("launch_terminal_row_sha256") == _row_sha256(stopped_launch)
        and stop_receipt.get("launch_finished_utc") == stopped_launch.get("finished_utc")
        and stop_receipt.get("accepted_end_epoch_at_stop") == EXPECTED_STOP_EPOCH
        and stop_receipt.get("checkpoint_sha256_still_step76") is True
        and stop_receipt.get("state_digests_match_capture") is True
    )
    resume_evidence, resume_issues = _resume_attempt_evidence(
        ledger_rows, launch_terminal=stopped_launch, run_dir=run_path
    )
    same_manifest_lineage = same_manifest_lineage and stop_receipt.get("run_identity") == before_identity
    resumed_same_lineage = same_manifest_lineage and stop_receipt_matches and resume_evidence is not None
    execution_log_sha = None
    rows: dict[int, tuple[str, dict[str, Any]]] = {}
    row_errors: list[str] = []
    row_numeric_errors: list[dict[str, Any]] = []
    try:
        rows, execution_log_sha = _unique_update_rows(run_path, (77, 78))
    except RecoveryAuditError as error:
        row_errors.append(str(error))

    mapping_checks: dict[str, bool] = {}
    replay_checks: dict[str, bool] = {}
    compared: dict[str, Any] = {}
    if rows:
        for step, cursor in ((77, "76"), (78, "77")):
            raw_record = rows[step][1]
            mapping_checks[str(step)] = _matches_expected_execution(raw_record, decisions[cursor])
            original = _record_from_capture(capture, step)
            value_errors: list[str] = []
            numeric_mismatches: list[dict[str, Any]] = []
            _compare_values(
                original, raw_record, path=f"step_{step}", errors=value_errors,
                numeric_errors=numeric_mismatches,
                rel_tol=float(protocol["tolerance"]["numeric_relative"]),
                abs_tol=float(protocol["tolerance"]["numeric_absolute"]),
            )
            # The sampled execution map is an exact identity, not a tolerant
            # scalar diagnostic. Confirm it separately before accepting the
            # recursive float comparison.
            replay_checks[str(step)] = not value_errors and not numeric_mismatches
            row_errors.extend(value_errors)
            row_numeric_errors.extend(numeric_mismatches)
            compared[str(step)] = {
                "expected_decision_cursor": int(cursor),
                "execution_mapping_exact": mapping_checks[str(step)],
                "diagnostics_within_tolerance": replay_checks[str(step)],
                "numeric_mismatch_count": len(numeric_mismatches),
            }
    else:
        mapping_checks = {"77": False, "78": False}
        replay_checks = {"77": False, "78": False}

    source_config_same = _source_config_bindings_unchanged(before_checkpoint, after_checkpoint)
    campaign_bindings_before = before_checkpoint.get("campaign_source_bindings")
    campaign_bindings_after = after_checkpoint.get("campaign_source_bindings")
    campaign_bindings_same = (
        campaign_bindings_before is not None
        and campaign_bindings_after is not None
        and campaign_bindings_before == campaign_bindings_after
    )
    runtime_same = _runtime_binding_unchanged(before_checkpoint, after_checkpoint)
    replay_advanced = after_checkpoint.get("global_step", 0) >= 78
    check_map = {
        "task_id_is_D": capture.get("task_id") == EXPECTED_TASK,
        "arm_is_strategy": capture.get("arm") == EXPECTED_ARM,
        "checkpoint_step_is_76": (
            capture.get("checkpoint_step") == EXPECTED_STEP
            and before_checkpoint.get("global_step") == EXPECTED_STEP
        ),
        "next_execution_mapping_match": mapping_checks.get("77", False),
        "step_78_counter_mapping_match": mapping_checks.get("78", False),
        "replay_update_diagnostics_match": all(replay_checks.values()) and len(replay_checks) == 2,
        "post_resume_checkpoint_advanced": replay_advanced,
        "stopped_checkpoint_hash_matches_capture": stop_receipt_matches,
        "resumed_same_lineage": resumed_same_lineage,
        "campaign_source_bindings_unchanged": campaign_bindings_same,
        "source_and_config_bindings_unchanged": source_config_same,
        "runtime_mapping_unchanged": runtime_same,
    }
    if resume_evidence is None:
        status = "not_established"
        reason = "; ".join(resume_issues) or "no completed same-lineage resume attempt is recorded"
    elif all(check_map.values()):
        status = "verified"
        reason = None
    else:
        status = "failed"
        reason = "required recovery checks failed: " + ", ".join(
            key for key, matched in check_map.items() if not matched
        )
    return {
        "schema_version": 1,
        "status": status,
        "reason": reason,
        "task_id": EXPECTED_TASK,
        "arm": EXPECTED_ARM,
        "checkpoint_step": EXPECTED_STEP,
        "next_execution_mapping_match": check_map["next_execution_mapping_match"],
        "replay_update_diagnostics_match": check_map["replay_update_diagnostics_match"],
        "source_and_config_bindings_unchanged": check_map[
            "source_and_config_bindings_unchanged"
        ],
        "campaign_source_bindings_unchanged": check_map[
            "campaign_source_bindings_unchanged"
        ],
        "resumed_same_lineage": check_map["resumed_same_lineage"],
        "stopped_checkpoint_hash_matches_capture": check_map[
            "stopped_checkpoint_hash_matches_capture"
        ],
        "post_resume_checkpoint_step": after_checkpoint.get("global_step"),
        "post_resume_checkpoint_advanced": replay_advanced,
        "capture_checkpoint_sha256": before_checkpoint.get("sha256"),
        "post_resume_checkpoint_sha256": after_checkpoint.get("sha256"),
        "tolerance": protocol.get("tolerance"),
        "checks": check_map,
        "replay_rows": compared,
        "row_errors": row_errors,
        "numeric_mismatches": row_numeric_errors,
        "runner_ledger_path": str(ledger_file.resolve()),
        "runner_ledger_sha256_at_compare": ledger_sha,
        "runner_launch_attempt": capture.get("runner_launch"),
        "runner_stop_receipt": dict(stop_receipt),
        "runner_resume_evidence": resume_evidence,
        "runner_resume_issues": resume_issues,
        "execution_update_log_sha256_after_resume": execution_log_sha,
        "lineage": {
            "before": before_identity,
            "after": after_identity,
            "same_manifest_identity": same_manifest_lineage,
            "checkpoint_advanced_beyond_replay": replay_advanced,
        },
        "source_and_config": {
            "unchanged": source_config_same,
            "before": before_checkpoint.get("source_config_bindings"),
            "after": after_checkpoint.get("source_config_bindings"),
            "campaign_bindings_before": before_checkpoint.get("campaign_source_bindings"),
            "campaign_bindings_after": after_checkpoint.get("campaign_source_bindings"),
            "campaign_binding_error_after": campaign_binding_error,
            "resolved_config_sha256_before": before_checkpoint.get("resolved_config_sha256"),
            "resolved_config_sha256_after": after_checkpoint.get("resolved_config_sha256"),
        },
        "runtime_binding": {
            "unchanged": runtime_same,
            "implementation_unchanged": (
                before_checkpoint.get("runtime_implementation")
                == after_checkpoint.get("runtime_implementation")
            ),
            "stable_identity_before": before_checkpoint.get("runtime_identity"),
            "stable_identity_after": after_checkpoint.get("runtime_identity"),
        },
        "evidence_scope": (
            "CPU-only checkpoint/log comparison; measured agreement on steps 77 and 78 does not prove bitwise CUDA reproducibility or cover future global Python/NumPy RNG consumers."
        ),
    }


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise RecoveryAuditError(f"refusing to overwrite existing audit artifact: {path}") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("capture", "confirm-stopped", "compare"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--run-dir", type=Path, required=True)
        subparser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
        subparser.add_argument("--expected-decisions", type=Path, default=DEFAULT_DECISIONS)
        subparser.add_argument("--output", type=Path, required=True)
        subparser.add_argument("--ledger", type=Path)
        if command == "capture":
            subparser.add_argument("--checkpoint-copy", type=Path)
        if command in {"confirm-stopped", "compare"}:
            subparser.add_argument("--capture", type=Path, required=True)
        if command == "compare":
            subparser.add_argument("--stop-receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "capture":
            checkpoint_copy = args.checkpoint_copy or args.output.with_name(
                f"{args.output.stem}_step76_checkpoint.pt"
            )
            if args.output.exists():
                raise RecoveryAuditError(f"refusing to overwrite existing audit artifact: {args.output}")
            if checkpoint_copy.exists():
                raise RecoveryAuditError(
                    f"refusing to overwrite existing checkpoint copy: {checkpoint_copy}"
                )
            result = capture_probe(
                args.run_dir,
                protocol_path=args.protocol,
                decisions_path=args.expected_decisions,
                checkpoint_copy_path=checkpoint_copy,
                ledger_path=args.ledger,
            )
        elif args.command == "confirm-stopped":
            capture = _read_json(args.capture, "recovery capture")
            result = confirm_stopped_probe(
                capture, args.run_dir, ledger_path=args.ledger
            )
        else:
            capture = _read_json(args.capture, "recovery capture")
            stop_receipt = _read_json(args.stop_receipt, "stopped-checkpoint receipt")
            result = compare_probe(
                capture,
                args.run_dir,
                stop_receipt=stop_receipt,
                protocol_path=args.protocol, decisions_path=args.expected_decisions,
                ledger_path=args.ledger,
            )
        _write_json_new(args.output, result)
    except RecoveryAuditError as error:
        print(f"recovery audit: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))
    return 0 if result["status"] in {
        "pending_replay", "stopped_checkpoint_verified", "verified", "not_established"
    } else 1


if __name__ == "__main__":  # pragma: no cover - exercised as a CLI manually
    raise SystemExit(main())
