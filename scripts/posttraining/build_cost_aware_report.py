#!/usr/bin/env python3
"""Build a compact, artifact-driven cost-aware Lagrangian campaign report.

This generator reads the campaign's frozen profiles, plan decisions, resource
ledger, task inventory, and read-only quality-history analysis. It never opens a
dataset or run checkpoint and writes one Markdown report plus at most three
Matplotlib PDF masters below the owned campaign root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

DEFAULT_TASKS = ("G", "D", "X")
MANDATORY_FALLBACK = {"G": ("dense", "heuristic", "strategy"), "D": ("dense", "heuristic", "strategy")}
FIELD_PALETTE = ("#3366a6", "#dd8452", "#55a868", "#c44e52", "#8172b3")
METRIC_COLORS = {
    "coherence_sum": "#222222",
    "family_A": "#3366a6",
    "family_B": "#dd8452",
    "family_T": "#55a868",
    "endpoint_aggregate": "#c44e52",
}


class ReportBuildError(RuntimeError):
    """Raised when a present report input is malformed or contradicts its schema."""


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReportBuildError(f"cannot read JSON evidence {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReportBuildError(f"expected a JSON object at {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file() or path.is_symlink():
        return None
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ReportBuildError(f"non-object JSONL row {line_no} in {path}")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise ReportBuildError(f"cannot read JSONL evidence {path}: {error}") from error
    return rows


def _sha256(path: Path) -> str | None:
    if not path.is_file() or path.is_symlink():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _csv_number(value: Any) -> float | None:
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    return _number(value)


def _fmt(value: Any, digits: int = 3) -> str:
    number = _number(value)
    return f"{number:.{digits}f}" if number is not None else "not recorded"


def _pct(value: Any, digits: int = 1) -> str:
    number = _number(value)
    return f"{number * 100:.{digits}f}%" if number is not None else "not recorded"


def _hours(value: Any, digits: int = 2) -> str:
    number = _number(value)
    return f"{number / 3600:.{digits}f} h" if number is not None else "not recorded"


def _ratio_text(value: Any, limit: float) -> str:
    number = _number(value)
    if number is None:
        return "not recorded"
    return f"{number:.4f}{' !' if number > limit else ''}"


def _safe_text(value: Any) -> str:
    if value is None:
        return "not recorded"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()
    except ValueError:
        return path.as_posix()


def _load_inputs(root: Path, history_path: Path | None = None) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise ReportBuildError(f"campaign root must be an owned directory: {root}")
    root = root.resolve(strict=True)
    inventory_path = root / "tasks" / "task_inventory.json"
    inventory = _read_json(inventory_path)
    history_path = history_path or root / "analysis" / "quality_history.json"
    history = _read_json(history_path)
    profiles = {}
    raw_profiles = {}
    plans = {}
    calibrations = {}
    contracts = {}
    for task_id in DEFAULT_TASKS:
        profiles[task_id] = _read_json(root / "profiles" / f"{task_id}_profile.json")
        raw_profiles[task_id] = _read_json(root / "profiles" / f"{task_id}_raw_profile.json")
        plans[task_id] = _read_json(root / "plans" / f"{task_id}_plan_decision.json")
        calibrations[task_id] = _read_json(
            root / "profiles" / f"{task_id}_source_calibration.json"
        )
        contracts[task_id] = _read_json(root / "tasks" / f"{task_id}_task_contract.json")
    ledger_path = root / "manifests" / "campaign_runner_ledger.jsonl"
    ledger_rows = _read_jsonl(ledger_path) or []
    source_binding = _read_json(root / "manifests" / "science_source_binding_v1.json")
    predeclaration = _read_json(root / "manifests" / "science_predeclaration.json")
    contract_check = _read_json(root / "analysis" / "first_science_contract_check.json")
    recovery_probe = _read_json(root / "analysis" / "recovery_probe_result.json")
    fixed_query_attestation = _read_json(root / "analysis" / "fixed_query_count_attestation.json")
    bottleneck_scope_limitations = _read_json(root / "analysis" / "bottleneck_scope_limitations.json")
    profile_proxy_label_erratum = _read_json(root / "analysis" / "profile_proxy_label_erratum.json")
    runtime_policy_summary_path = root / "analysis" / "runtime_policy_summary.json"
    runtime_policy_summary_error = None
    try:
        runtime_policy_summary = _read_json(runtime_policy_summary_path)
    except ReportBuildError:
        runtime_policy_summary = None
        runtime_policy_summary_error = "runtime policy summary is invalid JSON"
    components_path = root / "analysis" / "quality_components.csv"
    fast_report_path = root / "manifests" / "FAST_LAGRANGIAN_LPC_REPORT_at_6021c62.md"
    revisions = {
        "warm_reference": _read_json(
            root / "analysis" / "profiles_warm_reference_v1" / "revision.json"
        ),
        "complete_profile_cost": _read_json(
            root / "analysis" / "profile_v2_complete_cost_revision" / "revision.json"
        ),
        "noise_cache_scope": _read_json(
            root / "analysis" / "first_science_contract_check.json"
        ),
    }
    return {
        "root": root,
        "inventory": inventory,
        "history": history,
        "history_path": history_path,
        "profiles": profiles,
        "raw_profiles": raw_profiles,
        "plans": plans,
        "calibrations": calibrations,
        "contracts": contracts,
        "ledger_rows": ledger_rows,
        "ledger_path": ledger_path,
        "source_binding": source_binding,
        "predeclaration": predeclaration,
        "contract_check": contract_check,
        "recovery_probe": recovery_probe,
        "recovery_probe_path": root / "analysis" / "recovery_probe_result.json",
        "fixed_query_attestation": fixed_query_attestation,
        "fixed_query_attestation_path": root / "analysis" / "fixed_query_count_attestation.json",
        "bottleneck_scope_limitations": bottleneck_scope_limitations,
        "bottleneck_scope_limitations_path": root / "analysis" / "bottleneck_scope_limitations.json",
        "profile_proxy_label_erratum": profile_proxy_label_erratum,
        "profile_proxy_label_erratum_path": root / "analysis" / "profile_proxy_label_erratum.json",
        "runtime_policy_summary": runtime_policy_summary,
        "runtime_policy_summary_path": runtime_policy_summary_path,
        "runtime_policy_summary_error": runtime_policy_summary_error,
        "components_path": components_path,
        "fast_report_path": fast_report_path,
        "revisions": revisions,
    }


def _attempts(ledger_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(ledger_rows):
        attempt_id = row.get("attempt_id") or f"row-{index}"
        grouped[str(attempt_id)].append(row)
    attempts = []
    for attempt_id, rows in grouped.items():
        last = rows[-1]
        action = next((row.get("action") for row in reversed(rows) if row.get("action")), None)
        task = next((row.get("task_id") for row in reversed(rows) if row.get("task_id")), None)
        arm = next((row.get("arm") for row in reversed(rows) if row.get("arm")), None)
        charge = next(
            (
                _number(row.get("charged_process_seconds"))
                for row in reversed(rows)
                if _number(row.get("charged_process_seconds")) is not None
            ),
            None,
        )
        reserved = next(
            (
                _number(row.get("reserved_process_seconds"))
                for row in reversed(rows)
                if _number(row.get("reserved_process_seconds")) is not None
            ),
            None,
        )
        attempts.append(
            {
                "attempt_id": attempt_id,
                "action": action,
                "task_id": str(task) if task is not None else None,
                "arm": str(arm) if arm is not None else None,
                "status": last.get("status", "unknown"),
                "charged_process_seconds": charge,
                "reserved_process_seconds": reserved,
                "return_code": last.get("return_code"),
            }
        )
    return attempts


def _charged(attempts: list[dict[str, Any]], *, task_id: str | None = None, arm: str | None = None, actions: set[str] | None = None) -> float | None:
    selected = [
        item
        for item in attempts
        if (task_id is None or item["task_id"] == task_id)
        and (arm is None or item["arm"] == arm)
        and (actions is None or item["action"] in actions)
        and item["charged_process_seconds"] is not None
    ]
    if not selected:
        return None
    return sum(float(item["charged_process_seconds"]) for item in selected)


def _terminal_charge_summary(
    attempts: list[dict[str, Any]],
    *,
    task_id: str,
    arm: str,
    actions: set[str],
    zero_if_absent: bool = False,
) -> dict[str, Any]:
    selected = [
        item
        for item in attempts
        if item["task_id"] == task_id and item["arm"] == arm and item["action"] in actions
    ]
    terminal = [item for item in selected if item["status"] in {"completed", "failed_or_stopped"}]
    charged = [item for item in terminal if item["charged_process_seconds"] is not None]
    pending_count = len(selected) - len(terminal)
    unmeasured_count = len(terminal) - len(charged)
    complete = (
        (not selected and zero_if_absent)
        or (bool(selected) and pending_count == 0 and unmeasured_count == 0)
    )
    if charged:
        seconds = sum(float(item["charged_process_seconds"]) for item in charged)
    elif not selected and zero_if_absent:
        seconds = 0.0
    else:
        seconds = None
    return {
        "seconds": seconds,
        "complete": complete,
        "attempt_count": len(selected),
        "terminal_attempt_count": len(terminal),
        "pending_attempt_count": pending_count,
        "unmeasured_terminal_attempt_count": unmeasured_count,
    }


def _arm_runs(inputs: dict[str, Any], task_id: str, arm: str) -> list[dict[str, Any]]:
    history = inputs.get("history") or {}
    task = history.get("tasks", {}).get(task_id, {})
    arm_record = task.get("arms", {}).get(arm, {}) if isinstance(task, dict) else {}
    runs = arm_record.get("runs", []) if isinstance(arm_record, dict) else []
    return [run for run in runs if isinstance(run, dict)] if isinstance(runs, list) else []


def _mandatory_arms(inputs: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    inventory = inputs.get("inventory") or {}
    task_values = inventory.get("tasks", {}) if isinstance(inventory, dict) else {}
    required = {}
    if isinstance(task_values, dict):
        for task_id, task in task_values.items():
            if isinstance(task, dict) and task.get("role") == "mandatory":
                arms = task.get("arms")
                if isinstance(arms, list) and arms:
                    required[str(task_id)] = tuple(str(arm) for arm in arms)
    return required or dict(MANDATORY_FALLBACK)


def _arm_status(inputs: dict[str, Any], task_id: str, arm: str, horizon: int = 100) -> dict[str, Any]:
    runs = _arm_runs(inputs, task_id, arm)
    if not runs:
        history_task = (inputs.get("history") or {}).get("tasks", {}).get(task_id, {})
        arm_record = history_task.get("arms", {}).get(arm, {}) if isinstance(history_task, dict) else {}
        missing_state = arm_record.get("status", "pending_run") if isinstance(arm_record, dict) else "pending_run"
        return {"status": "pending_run", "detail": missing_state, "run": None, "runs": runs}
    if len(runs) != 1:
        return {
            "status": "ambiguous_multiple_runs",
            "detail": f"{len(runs)} run lineages require reviewer resolution",
            "run": None,
            "runs": runs,
        }
    run = runs[0]
    accepted = run.get("accepted_end_epoch")
    end_epoch = None
    if _number(accepted) is not None and math.isclose(float(accepted), round(float(accepted)), rel_tol=0.0, abs_tol=1e-8):
        end_epoch = round(float(accepted))
    quality_status = run.get("quality_assessment_status", "pending_quality_checks")
    assessment = run.get("quality_assessment")
    if end_epoch is None or end_epoch < horizon:
        status = (
            "running" if run.get("status") == "running"
            else "censored" if end_epoch is not None else "pending_end_epoch"
        )
        detail = f"accepted end epoch {end_epoch if end_epoch is not None else 'unknown'} of {horizon}"
    elif end_epoch != horizon:
        status = "quality_ambiguous"
        detail = f"accepted end epoch {end_epoch} does not match frozen horizon {horizon}"
    elif run.get("status") != "completed":
        status = "censored"
        detail = f"horizon reached, run status {run.get('status', 'unknown')}"
    elif quality_status != "assessed" or not isinstance(assessment, dict):
        status = "quality_pending"
        detail = str(quality_status)
    else:
        assessed_end = assessment.get("actual_end_epoch")
        assessed_horizon = (assessment.get("contract") or {}).get("maximum_epochs")
        if (
            isinstance(assessed_end, bool)
            or not isinstance(assessed_end, (int, float))
            or not math.isclose(float(assessed_end), float(end_epoch), rel_tol=0.0, abs_tol=1e-8)
            or isinstance(assessed_horizon, bool)
            or not isinstance(assessed_horizon, (int, float))
            or not math.isclose(float(assessed_horizon), float(horizon), rel_tol=0.0, abs_tol=1e-8)
        ):
            status = "quality_ambiguous"
            detail = "assessor end epoch or frozen horizon does not match the accepted run cursor"
        else:
            native_history = run.get("native_history")
            integrity = native_history.get("integrity_status") if isinstance(native_history, dict) else None
            if not isinstance(native_history, dict):
                status = "quality_pending"
                detail = "native history provenance is absent"
            elif integrity != "ok":
                status = "quality_ambiguous"
                detail = f"native history integrity is {integrity}"
            else:
                schedule = (inputs.get("inventory") or {}).get("quality_contract", {}).get("monitor_epochs")
                expected_epochs = list(range(10, horizon + 1, 10))
                expected_frozen_epochs = list(range(0, horizon + 1, 10))
                frozen_epochs = (
                    [int(value) for value in schedule if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isclose(float(value), round(float(value)), rel_tol=0.0, abs_tol=1e-8)]
                    if isinstance(schedule, list)
                    else []
                )
                if not isinstance(schedule, list) or frozen_epochs != expected_frozen_epochs or len(frozen_epochs) != len(schedule):
                    status = "quality_ambiguous"
                    detail = "frozen monitor schedule does not match the complete 10-epoch native history through the accepted horizon"
                else:
                    records = _uniform_rows(run)
                    observed = []
                    invalid_epoch_records = []
                    for item in records:
                        value = item.get("epoch")
                        if (
                            isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or not math.isclose(float(value), round(float(value)), rel_tol=0.0, abs_tol=1e-8)
                        ):
                            invalid_epoch_records.append(item)
                        else:
                            observed.append(round(float(value)))
                    missing = [epoch for epoch in expected_epochs if observed.count(epoch) != 1]
                    unexpected = [epoch for epoch in observed if epoch not in expected_epochs]
                    ambiguous = [
                        int(item["epoch"])
                        for item in records
                        if item.get("status") != "native_primary"
                        or item.get("panel_kind") != "uniform"
                        or item.get("measurement_source") != "native_topology_history"
                    ]
                    if missing or ambiguous or unexpected or invalid_epoch_records:
                        status = "quality_pending" if missing and not ambiguous and not unexpected and not invalid_epoch_records else "quality_ambiguous"
                        detail_parts = []
                        if missing:
                            detail_parts.append("missing/non-unique scheduled native checks at " + ", ".join(map(str, missing)))
                        if ambiguous:
                            detail_parts.append("time-ambiguous native checks at " + ", ".join(map(str, sorted(set(ambiguous)))))
                        if unexpected:
                            detail_parts.append("unexpected native-check epochs at " + ", ".join(map(str, sorted(set(unexpected)))))
                        if invalid_epoch_records:
                            detail_parts.append("noninteger or invalid native-check epoch values are present")
                        detail = "; ".join(detail_parts)
                    else:
                        status = "complete"
                        detail = "100-epoch run with every scheduled native quality check and matching assessment"
    return {"status": status, "detail": detail, "run": run, "runs": runs}


def _arm_metric_summary(run: dict[str, Any] | None) -> dict[str, Any]:
    if not run:
        return {"last_epoch": None, "ratios": {}, "cold_seconds": None, "wall_seconds": None}
    records = run.get("uniform_evaluations", [])
    records = [item for item in records if isinstance(item, dict) and isinstance(item.get("epoch"), int)]
    records.sort(key=lambda item: item["epoch"])
    latest = records[-1] if records else {}
    cold_seconds = _number(latest.get("cold_process_seconds_including_profile_acquisition"))
    if cold_seconds is None:
        cold_seconds = _number(latest.get("completion_seconds"))
    return {
        "last_epoch": latest.get("epoch"),
        "ratios": latest.get("ratios") or {},
        "cold_seconds": cold_seconds,
        "wall_seconds": _number(latest.get("wall_seconds_from_first_training_command")),
    }


def _quality_status_text(status: dict[str, Any]) -> tuple[str, str]:
    run = status.get("run")
    if not run:
        return "not assessed", "not assessed"
    assessment = run.get("quality_assessment") if isinstance(run.get("quality_assessment"), dict) else {}
    confirmation = assessment.get("confirmation") or {}
    retention = assessment.get("retention") or {}
    confirm_text = confirmation.get("status", "not assessed")
    if confirmation.get("confirmation_epoch") is not None:
        confirm_text += f" E{confirmation['confirmation_epoch']}"
    retention_text = retention.get("status", "not assessed")
    if retention.get("through_observed_end_epoch") is not None:
        retention_text += f" through E{retention['through_observed_end_epoch']}"
    return confirm_text, retention_text


def _all_mandatory_complete(inputs: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    records = []
    required = _mandatory_arms(inputs)
    complete = bool(required)
    inventory_tasks = (inputs.get("inventory") or {}).get("tasks", {})
    for task_id, arms in required.items():
        task = inventory_tasks.get(task_id, {}) if isinstance(inventory_tasks, dict) else {}
        horizon = int(task.get("horizon", 100)) if isinstance(task, dict) else 100
        for arm in arms:
            status = _arm_status(inputs, task_id, arm, horizon)
            records.append({"task_id": task_id, "arm": arm, **status})
            complete = complete and status["status"] == "complete"
    return complete, records


def _final_interpretation_state(
    inputs: dict[str, Any], all_mandatory_complete: bool
) -> dict[str, Any]:
    """Return reviewed prose only when its exact evidence bindings are current."""
    root = inputs["root"]
    path = root / "analysis" / "final_interpretation.json"
    try:
        review = _read_json(path)
    except ReportBuildError:
        return {"status": "stale", "detail": "review input is not valid JSON", "paragraphs": []}
    if review is None:
        return {"status": "pending", "detail": "no reviewed interpretation input", "paragraphs": []}
    if review.get("schema") != 1 or review.get("status") != "reviewed":
        return {"status": "stale", "detail": "review schema or status is invalid", "paragraphs": []}
    paragraphs = review.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs or any(
        not isinstance(item, str) or not item.strip() for item in paragraphs
    ):
        return {"status": "stale", "detail": "review paragraphs are missing or invalid", "paragraphs": []}

    evidence_paths = {
        "quality_history.json": root / "analysis" / "quality_history.json",
        "campaign_runner_ledger.jsonl": root / "manifests" / "campaign_runner_ledger.jsonl",
        "science_source_binding_v1.json": root / "manifests" / "science_source_binding_v1.json",
    }
    hashes = review.get("evidence_sha256")
    required_names = {"quality_history.json", "campaign_runner_ledger.jsonl"}
    if not isinstance(hashes, dict) or not required_names.issubset(hashes) or not set(hashes).issubset(evidence_paths):
        return {"status": "stale", "detail": "review evidence hash set is invalid", "paragraphs": []}
    for name, expected in hashes.items():
        supplied = expected
        if (
            not isinstance(supplied, str)
            or len(supplied) != 64
            or any(character not in "0123456789abcdef" for character in supplied.lower())
        ):
            return {"status": "stale", "detail": f"review hash for {name} is invalid", "paragraphs": []}
        current = _sha256(evidence_paths[name])
        if current is None or supplied.lower() != current:
            return {"status": "stale", "detail": f"review hash for {name} does not match current evidence", "paragraphs": []}

    canonical_history = (root / "analysis" / "quality_history.json").resolve(strict=False)
    selected_history = Path(inputs.get("history_path", canonical_history)).resolve(strict=False)
    if selected_history != canonical_history:
        return {"status": "stale", "detail": "report uses a noncanonical quality-history input", "paragraphs": []}
    if not all_mandatory_complete:
        return {"status": "pending", "detail": "mandatory 100-epoch evidence is incomplete", "paragraphs": []}
    return {"status": "reviewed", "detail": "evidence hashes match", "paragraphs": paragraphs}


def _final_interpretation_lines(state: dict[str, Any]) -> list[str]:
    if state["status"] == "reviewed":
        return [
            "",
            "## Reviewed findings",
            "",
            "\n\n".join(state["paragraphs"]),
            "",
        ]
    return [
        "",
        f"> **Main-reviewed interpretation:** {state['status'].upper()} — {state['detail']}. No prior interpretation is carried into this report.",
        "",
    ]


def _runtime_policy_table_lines(inputs: dict[str, Any]) -> list[str]:
    root = inputs["root"]
    summary_path = inputs.get("runtime_policy_summary_path", root / "analysis" / "runtime_policy_summary.json")
    summary = inputs.get("runtime_policy_summary")
    inventory_tasks = (inputs.get("inventory") or {}).get("tasks", {})
    expected_tasks = [
        str(task_id)
        for task_id, task in inventory_tasks.items()
        if isinstance(task, dict)
        and isinstance(task.get("arms"), list)
        and "strategy" in task["arms"]
    ] if isinstance(inventory_tasks, dict) else []
    expected_tasks = expected_tasks or ["G", "D", "X"]
    artifact_digest = _sha256(summary_path)
    if artifact_digest:
        summary_ref = _rel(summary_path, root)
        summary_line = f"[Runtime-policy summary](../{summary_ref}), SHA-256 `{artifact_digest}`."
    elif inputs.get("runtime_policy_summary_error"):
        summary_line = f"Pending: {inputs['runtime_policy_summary_error']}."
    else:
        summary_line = "Pending: `analysis/runtime_policy_summary.json` is absent."

    records_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid_count = 0
    schema_detail = None
    if not isinstance(summary, dict):
        schema_detail = "schema 1 summary is unavailable"
    elif type(summary.get("schema")) is not int or summary.get("schema") != 1 or not isinstance(summary.get("records"), list):
        schema_detail = "summary does not match schema 1 with a records list"
    else:
        for record in summary["records"]:
            if not isinstance(record, dict) or not isinstance(record.get("task_id"), str) or record.get("arm") != "strategy":
                invalid_count += 1
                continue
            records_by_task[record["task_id"]].append(record)

    task_ids = list(expected_tasks)
    if schema_detail is None:
        task_ids.extend(sorted(set(records_by_task) - set(task_ids)))
    required_fields = (
        "checkpoint_step", "checkpoint_sha256", "initial_plan_id", "final_plan_id",
        "accepted_cursor", "plan_step_counts", "topology_included_updates",
        "max_importance_multiplier_observed", "monitor_overhead_seconds",
        "accepted_training_seconds", "fallback_reason", "full_recovery_intervals",
        "train_audit_epochs", "noise_audit_epochs", "full_train_audit_artifact",
    )
    table_rows = []
    for task_id in task_ids:
        row: dict[str, Any] = {"task_id": task_id, "status": "pending", "record": {}}
        if schema_detail is not None:
            row["status"] = "pending: " + schema_detail
        else:
            candidates = records_by_task.get(task_id, [])
            if not candidates:
                row["status"] = "pending: no strategy record"
            elif len(candidates) != 1:
                row["status"] = f"ambiguous: {len(candidates)} strategy records"
            else:
                row["record"] = candidates[0]
                missing = [name for name in required_fields if name not in candidates[0]]
                row["status"] = "recorded" if not missing else "partial: missing " + ", ".join(missing)
        table_rows.append(row)

    def display(value: Any, *, missing: bool = False) -> str:
        if missing:
            return "not recorded"
        if isinstance(value, (dict, list)):
            return "`" + _safe_text(json.dumps(value, sort_keys=True, separators=(",", ":"))) + "`"
        if value is None:
            return "null"
        return "`" + _safe_text(value) + "`"

    def audit_artifact(value: Any, *, missing: bool = False) -> str:
        if missing:
            return "not recorded"
        if not isinstance(value, str) or not value:
            return display(value)
        candidate = Path(value)
        candidate = candidate if candidate.is_absolute() else root / candidate
        try:
            relative = candidate.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()
        except ValueError:
            return "`" + _safe_text(value) + "` (outside campaign root; not linked)"
        if candidate.is_file() and not candidate.is_symlink():
            return f"[`{relative}`](<../{relative}>)"
        return "`" + _safe_text(value) + "` (not present)"

    lines = [summary_line]
    if invalid_count:
        lines.append(f"The schema-1 input contains {invalid_count} malformed or non-strategy record(s), which are not attributed to an arm.")
    if not task_ids:
        lines.extend(["", "| Task / arm | Status | Checkpoint | Plan initial → final | Accepted cursor | Observed coverage | Monitor / training | Fallback | Recovery intervals | Fixed-TRAIN audit epochs | Full-TRAIN audit artifact |", "|---|---|---|---|---:|---|---|---|---|---|---|"])
        lines.append("| strategy arms | pending | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded |")
        return lines
    lines.extend(
        [
            "",
            "| Task / arm | Status | Checkpoint step / SHA-256 | Plan initial → final | Accepted cursor | Observed coverage | Monitor / training | Fallback | Recovery intervals | Fixed-TRAIN audit epochs | Full-TRAIN audit artifact |",
            "|---|---|---|---|---:|---|---|---|---|---|---|",
        ]
    )
    for row in table_rows:
        record = row["record"]
        step = display(record.get("checkpoint_step"), missing="checkpoint_step" not in record)
        raw_digest = record.get("checkpoint_sha256")
        digest = (
            f"`{raw_digest[:12]}…`" if isinstance(raw_digest, str) and len(raw_digest) == 64
            else display(raw_digest, missing="checkpoint_sha256" not in record)
        )
        checkpoint = f"{step} / {digest}"
        plan = f"{display(record.get('initial_plan_id'), missing='initial_plan_id' not in record)} → {display(record.get('final_plan_id'), missing='final_plan_id' not in record)}"
        counts = display(record.get("plan_step_counts"), missing="plan_step_counts" not in record)
        topology = display(record.get("topology_included_updates"), missing="topology_included_updates" not in record)
        importance = display(record.get("max_importance_multiplier_observed"), missing="max_importance_multiplier_observed" not in record)
        coverage = f"plans {counts}; topology updates {topology}; max importance {importance}"
        monitor = _number(record.get("monitor_overhead_seconds"))
        training = _number(record.get("accepted_training_seconds"))
        monitor_cell = (
            f"{display(record.get('monitor_overhead_seconds'), missing='monitor_overhead_seconds' not in record)} s / "
            f"{display(record.get('accepted_training_seconds'), missing='accepted_training_seconds' not in record)} s"
        )
        if monitor is not None and training is not None and monitor >= 0 and training > 0:
            monitor_cell = f"{monitor:.2f} s / {training:.2f} s ({_pct(monitor / training, 1)})"
        fallback = display(record.get("fallback_reason"), missing="fallback_reason" not in record)
        intervals = display(record.get("full_recovery_intervals"), missing="full_recovery_intervals" not in record)
        train_epochs = display(record.get("train_audit_epochs"), missing="train_audit_epochs" not in record)
        noise_epochs = display(record.get("noise_audit_epochs"), missing="noise_audit_epochs" not in record)
        audit_epochs = f"train {train_epochs}; noise {noise_epochs}"
        audit_link = audit_artifact(record.get("full_train_audit_artifact"), missing="full_train_audit_artifact" not in record)
        lines.append(
            f"| {row['task_id']} / strategy | {row['status']} | {checkpoint} | {plan} | {display(record.get('accepted_cursor'), missing='accepted_cursor' not in record)} | {coverage} | {monitor_cell} | {fallback} | {intervals} | {audit_epochs} | {audit_link} |"
        )
    return lines


def _profile_cost_rows(inputs: dict[str, Any], attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for task_id in DEFAULT_TASKS:
        profile = inputs["profiles"].get(task_id) or {}
        raw = inputs["raw_profiles"].get(task_id) or {}
        decision = inputs["plans"].get(task_id) or {}
        if not any((profile, raw, decision)):
            continue
        profile_charge = _charged(attempts, task_id=task_id, actions={"profile"})
        rows.append(
            {
                "task_id": task_id,
                "profile": profile,
                "raw": raw,
                "decision": decision,
                "profile_charge": profile_charge,
                "profile_attempt_count": sum(
                    item["task_id"] == task_id and item["action"] == "profile" for item in attempts
                ),
                "profile_failed_attempts": sum(
                    item["task_id"] == task_id
                    and item["action"] == "profile"
                    and item["status"] not in {"completed", "running"}
                    for item in attempts
                ),
            }
        )
    return rows


def _prediction_records(inputs: dict[str, Any], attempts: list[dict[str, Any]], completion_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_arm = {(item["task_id"], item["arm"]): item for item in completion_records}
    records = []
    for row in _profile_cost_rows(inputs, attempts):
        task_id = row["task_id"]
        decision = row["decision"]
        full_forecast = _number(decision.get("predicted_full_total_s"))
        selected_forecast = _number(decision.get("predicted_total_s"))
        profile_seconds = row["profile_charge"]
        strategy_attempts = [item for item in attempts if item["task_id"] == task_id and item["arm"] == "strategy" and item["action"] in {"launch", "resume"}]
        dense_attempts = [item for item in attempts if item["task_id"] == task_id and item["arm"] == "dense" and item["action"] in {"launch", "resume"}]
        strategy_train = _charged(attempts, task_id=task_id, arm="strategy", actions={"launch", "resume"})
        dense_train = _charged(attempts, task_id=task_id, arm="dense", actions={"launch", "resume"})
        if strategy_train is None and not strategy_attempts:
            strategy_train = 0.0
        if dense_train is None and not dense_attempts:
            dense_train = 0.0
        strategy_state = by_arm.get((task_id, "strategy"), {"status": "pending_run"})
        dense_state = by_arm.get((task_id, "dense"), {"status": "pending_run"})
        selected_actual = (
            profile_seconds + strategy_train
            if profile_seconds is not None and strategy_train is not None
            else None
        )
        strategy_complete = strategy_state["status"] == "complete"
        dense_complete = dense_state["status"] == "complete"
        selected_error = (
            (selected_actual - selected_forecast) / selected_forecast
            if strategy_complete and selected_actual is not None and selected_forecast and selected_forecast > 0
            else None
        )
        full_error = (
            (dense_train - full_forecast) / full_forecast
            if dense_complete and dense_train is not None and full_forecast and full_forecast > 0
            else None
        )
        records.append(
            {
                **row,
                "strategy_train_seconds": strategy_train,
                "dense_train_seconds": dense_train,
                "selected_actual_seconds": selected_actual,
                "selected_forecast_seconds": selected_forecast,
                "full_forecast_seconds": full_forecast,
                "selected_error": selected_error,
                "full_error": full_error,
                "strategy_state": strategy_state,
                "dense_state": dense_state,
            }
        )
    return records


def _matplotlib(root: Path):
    cache_dir = root / "caches" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    return plt, PdfPages, Rectangle


def _write_cost_figure(inputs: dict[str, Any], attempts: list[dict[str, Any]], completion_records: list[dict[str, Any]], path: Path) -> None:
    plt, PdfPages, _Rectangle = _matplotlib(inputs["root"])
    rows = _prediction_records(inputs, attempts, completion_records)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), constrained_layout=True)
    fig.suptitle("Cost decision, estimator-noise eligibility, and charged execution", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    if rows:
        x = list(range(len(rows)))
        width = 0.34
        selected = [_number(row["selected_forecast_seconds"]) for row in rows]
        full = [_number(row["full_forecast_seconds"]) for row in rows]
        ax.bar([value - width / 2 for value in x], [value / 3600 if value is not None else 0 for value in selected], width, color="#4c78a8", label="selected-plan forecast")
        ax.bar([value + width / 2 for value in x], [value / 3600 if value is not None else 0 for value in full], width, color="#b7c6d7", label="full-reference forecast")
        for position, row in zip(x, rows):
            plan_name = row["decision"].get("plan_id", "pending")
            ax.text(position, 0.02, f"{plan_name}\n{'no-action' if row['decision'].get('no_action') else 'selected'}", ha="center", va="bottom", fontsize=6, rotation=90, transform=ax.get_xaxis_transform())
        ax.set_xticks(x, [row["task_id"] for row in rows])
        ax.set_ylabel("conservative forecast (hours / 100 epochs)")
        ax.legend(loc="upper right", frameon=False)
        ax.grid(axis="y", alpha=0.2)
    else:
        ax.text(0.5, 0.5, "Profile and plan artifacts pending", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("A. Frozen whole-plan forecast")

    ax = axes[0, 1]
    variance_rows = []
    for task_id in DEFAULT_TASKS:
        profile = inputs["profiles"].get(task_id) or {}
        intrinsic = _number(profile.get("intrinsic_noise_variance"))
        for item in profile.get("measurements", []) if isinstance(profile.get("measurements"), list) else []:
            extra = _number(item.get("extra_variance"))
            if intrinsic is not None and intrinsic > 0 and extra is not None:
                selected_plan = (inputs["plans"].get(task_id) or {}).get("plan_id")
                variance_rows.append((task_id, str(item.get("plan_id", "unknown")), extra / intrinsic, str(item.get("plan_id")) == selected_plan))
    if variance_rows:
        y = list(range(len(variance_rows)))
        ax.axvline(1.0, color="#c44e52", linestyle="--", linewidth=1, label="TRAIN-noise ceiling")
        for yy, (task_id, plan_id, ratio, selected) in zip(y, variance_rows):
            color = {"G": "#4c78a8", "D": "#f58518", "X": "#54a24b"}.get(task_id, "#777777")
            ax.scatter(ratio, yy, s=34 if selected else 18, color=color if selected else "#a7a7a7", edgecolor=color, linewidth=0.8 if selected else 0, zorder=3)
            ax.text(max(ratio, 0) + 0.018, yy, f"{task_id}/{plan_id}{' *' if selected else ''}", va="center", fontsize=6)
        ax.set_yticks([])
        ax.set_xlabel("added estimator variance / intrinsic TRAIN estimate")
        ax.set_xlim(left=0, right=max(1.15, max(item[2] for item in variance_rows) * 1.10))
        ax.grid(axis="x", alpha=0.2)
        ax.legend(loc="upper right", frameon=False)
    else:
        ax.text(0.5, 0.5, "Noise profiles pending", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("B. Noise-budget screening (selected plan marked *)")

    ax = axes[1, 0]
    inventory_tasks = (inputs.get("inventory") or {}).get("tasks", {})
    task_ids = ["G", "D"]
    if _x_task_present(inputs) or any(
        item["task_id"] == "X" and item["action"] in {"launch", "resume"}
        for item in attempts
    ):
        task_ids.append("X")
    actual_status = {(item["task_id"], item["arm"]): item for item in completion_records}
    forecast_by_task = {item["task_id"]: item for item in rows}
    category_positions = {}
    category_labels = []
    category_arms = []
    for task_id in task_ids:
        task_record = inventory_tasks.get(task_id, {}) if isinstance(inventory_tasks, dict) else {}
        default_arms = ("dense", "heuristic", "strategy") if task_id in {"G", "D"} else ("dense", "strategy")
        task_arms = task_record.get("arms", list(default_arms)) if isinstance(task_record, dict) else list(default_arms)
        for arm in task_arms:
            position = len(category_labels)
            arm = str(arm)
            category_positions[(task_id, arm)] = position
            category_arms.append((task_id, arm))
            arm_label = {"dense": "Dense", "heuristic": "Fixed\nheuristic", "strategy": "Strategy"}.get(arm, arm)
            category_labels.append(f"{task_id}\n{arm_label}")

    if category_positions:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        arm_colors = {"dense": "#aeb8c2", "heuristic": "#d9a441", "strategy": "#4c78a8"}
        max_y = 0.0
        for task_id, arm in category_arms:
            position = category_positions[(task_id, arm)]
            run_state = actual_status.get((task_id, arm))
            if run_state is None:
                task_record = inventory_tasks.get(task_id, {}) if isinstance(inventory_tasks, dict) else {}
                run_state = _arm_status(
                    inputs,
                    task_id,
                    arm,
                    int(task_record.get("horizon", 100)) if isinstance(task_record, dict) else 100,
                )
            training_attempts = [
                item for item in attempts
                if item["task_id"] == task_id and item["arm"] == arm and item["action"] in {"launch", "resume"}
            ]
            training_seconds = _charged(attempts, task_id=task_id, arm=arm, actions={"launch", "resume"})
            profile_attempts = [
                item for item in attempts
                if item["task_id"] == task_id and item["action"] == "profile"
            ]
            profile_seconds = _charged(attempts, task_id=task_id, actions={"profile"})
            strategy_profile_incomplete = arm == "strategy" and bool(profile_attempts) and profile_seconds is None
            actual_seconds = None
            if not strategy_profile_incomplete and (training_seconds is not None or (arm == "strategy" and profile_seconds is not None)):
                profile_component = (profile_seconds or 0.0) if arm == "strategy" else 0.0
                actual_seconds = (training_seconds or 0.0) + profile_component
            if actual_seconds is not None:
                actual_hours = actual_seconds / 3600.0
                max_y = max(max_y, actual_hours)
                ax.bar(
                    position,
                    actual_hours,
                    width=0.66,
                    color=arm_colors.get(arm, "#aeb8c2"),
                    edgecolor="#333333",
                    linewidth=0.55,
                    hatch="" if run_state["status"] == "complete" else "//",
                    zorder=2,
                )
                forecast_row = forecast_by_task.get(task_id, {})
                error = (
                    forecast_row.get("selected_error") if arm == "strategy"
                    else forecast_row.get("full_error") if arm == "dense"
                    else None
                )
                label = f"{actual_hours:.2f} h"
                if error is not None:
                    label += f"\n{error:+.1%}"
                ax.text(position, actual_hours, label, ha="center", va="bottom", fontsize=5.8)
            else:
                active = any(item["status"] == "running" for item in training_attempts)
                state_label = "running; charge pending" if active else "not launched / uncharged"
                if strategy_profile_incomplete:
                    state_label = "profile charge pending"
                ax.text(position, 0.035, state_label, ha="center", va="bottom", fontsize=5.4, rotation=90, transform=ax.get_xaxis_transform())

            forecast_row = forecast_by_task.get(task_id, {})
            if arm == "dense":
                forecast = _number(forecast_row.get("full_forecast_seconds"))
                marker = "D"
            elif arm == "strategy":
                forecast = _number(forecast_row.get("selected_forecast_seconds"))
                marker = "^"
            else:
                forecast = None
                marker = "o"
            if forecast is not None:
                forecast_hours = forecast / 3600.0
                max_y = max(max_y, forecast_hours)
                ax.scatter(position, forecast_hours, marker=marker, s=30, facecolor="white", edgecolor="#222222", linewidth=0.9, zorder=5)

        ax.set_xticks(range(len(category_labels)), category_labels)
        ax.tick_params(axis="x", labelsize=6.5, pad=3)
        ax.set_ylabel("whole-command process hours")
        ax.set_ylim(bottom=0, top=max(0.25, max_y * 1.22))
        ax.grid(axis="y", alpha=0.2, zorder=0)
        ax.legend(
            handles=[
                Patch(facecolor=arm_colors["dense"], edgecolor="#333", label="dense command"),
                Patch(facecolor=arm_colors["heuristic"], edgecolor="#333", label="fixed-heuristic command"),
                Patch(facecolor=arm_colors["strategy"], edgecolor="#333", label="strategy command"),
                Line2D([], [], marker="D", linestyle="none", markerfacecolor="white", markeredgecolor="#222", label="full-reference forecast"),
                Line2D([], [], marker="^", linestyle="none", markerfacecolor="white", markeredgecolor="#222", label="selected-plan forecast"),
            ],
            loc="upper left",
            frameon=False,
            fontsize=5.4,
            ncol=2,
        )
    else:
        ax.text(0.5, 0.5, "Arm histories and ledger charges pending", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("C. Actual arm command charges and planner markers")

    ax = axes[1, 1]
    fresh_x = []
    fresh_counts = []
    for task_id in DEFAULT_TASKS:
        raw = inputs["raw_profiles"].get(task_id) or {}
        fresh = raw.get("fresh_TRAIN_stream_cache") or []
        if not fresh:
            continue
        hits = sum(_number(item.get("reference_hits")) or 0 for item in fresh)
        misses = sum(_number(item.get("reference_misses")) or 0 for item in fresh)
        fresh_x.append(task_id)
        fresh_counts.append((hits, misses))
    if fresh_counts:
        x = list(range(len(fresh_counts)))
        hits = [item[0] for item in fresh_counts]
        misses = [item[1] for item in fresh_counts]
        totals = [hit + miss for hit, miss in fresh_counts]
        ax.bar(x, hits, width=0.55, color="#55a868", label="reference hits")
        ax.bar(x, misses, width=0.55, bottom=hits, color="#dd8452", label="reference misses")
        for pos, hit, miss, total in zip(x, hits, misses, totals):
            percent = 100.0 * hit / total if total > 0 else float("nan")
            ax.text(pos, total + max(totals) * 0.025, f"{hit:.0f} hit / {miss:.0f} miss\n{percent:.1f}% hits", ha="center", va="bottom", fontsize=6.0)
        ax.set_xticks(x, fresh_x)
        max_total = max(totals, default=0)
        ax.set_ylim(0, max(1.0, max_total * 1.20))
        ax.set_ylabel("fresh-stream instrumented calls")
        ax.grid(axis="y", alpha=0.2, zorder=0)
        ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=6.2)
    else:
        ax.text(0.5, 0.5, "Fresh TRAIN stream counters pending", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("D. Fresh TRAIN reference calls only")

    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _figure1_cache_caption(inputs: dict[str, Any]) -> str:
    """Summarize exact fresh-stream counts and finite four-batch full timings."""
    stream_parts = []
    timing_parts = []
    for task_id in DEFAULT_TASKS:
        raw = inputs["raw_profiles"].get(task_id) or {}
        fresh = raw.get("fresh_TRAIN_stream_cache") or []
        if fresh:
            hits = sum(_number(item.get("reference_hits")) or 0 for item in fresh)
            misses = sum(_number(item.get("reference_misses")) or 0 for item in fresh)
            total = hits + misses
            fraction = 100.0 * hits / total if total > 0 else None
            stream_parts.append(
                f"{task_id}: {hits:.0f} hits / {misses:.0f} misses of {total:.0f} calls "
                f"({f'{fraction:.1f}%' if fraction is not None else 'hit fraction unavailable'})"
            )
        timings = raw.get("timings_seconds") if isinstance(raw.get("timings_seconds"), dict) else {}
        matched = timings.get("full", []) if isinstance(timings, dict) else []
        warm = raw.get("warm_full_timings_seconds", [])
        matched_values = [_number(value) for value in matched] if isinstance(matched, list) else []
        warm_values = [_number(value) for value in warm] if isinstance(warm, list) else []
        if len(matched_values) == 4 and len(warm_values) == 4 and all(value is not None for value in (*matched_values, *warm_values)):
            matched_mean = sum(value for value in matched_values if value is not None) / 4.0
            warm_mean = sum(value for value in warm_values if value is not None) / 4.0
            timing_parts.append(f"{task_id} {matched_mean:.2f}/{warm_mean:.2f} s")
    stream_text = "; ".join(stream_parts) if stream_parts else "fresh TRAIN counters not recorded"
    timing_text = "; ".join(timing_parts) if timing_parts else "four-batch matched/warm means not recorded"
    return (
        "Fresh TRAIN reference calls across the finite batches (instrumented oracle/input-gradient and support-enumeration scope): "
        f"{stream_text}. Full-update four-batch means, matched pre-oracle / repeated warm: {timing_text}. "
        "The hit fractions do not describe the timed production-cache path; neither these call counts nor the finite timing means establish stationary-cache or virgin-cache performance."
    )


def _elapsed_axis(rows: list[dict[str, Any]]) -> tuple[list[float] | None, str]:
    """Choose native check-completion time, including allocated profile work."""
    candidates = (
        ("cold_process_seconds_including_profile_acquisition", "cold check-completion hours including profile acquisition"),
        ("completion_seconds", "cold check-completion hours including profile acquisition"),
        ("completion_seconds_from_first_training_command", "active process hours excluding profile acquisition"),
        ("wall_seconds_from_first_training_command", "calendar wall hours from first training command"),
    )
    for key, label in candidates:
        values = [_number(row.get(key)) for row in rows]
        if rows and all(value is not None for value in values):
            return [float(value) / 3600.0 for value in values if value is not None], label
    return None, "elapsed time unavailable"


def _uniform_rows(run: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not run:
        return []
    rows = run.get("uniform_evaluations", [])
    if not isinstance(rows, list):
        return []
    return sorted(
        [row for row in rows if isinstance(row, dict) and isinstance(row.get("epoch"), (int, float))],
        key=lambda row: float(row["epoch"]),
    )


def _quality_history_bundle(inputs: dict[str, Any], task_id: str, arms: list[str]) -> dict[str, Any]:
    """Collect only observed scheduled native checks; keep absent arms empty."""
    task_record = ((inputs.get("inventory") or {}).get("tasks") or {}).get(task_id, {})
    horizon_value = _number(task_record.get("horizon")) if isinstance(task_record, dict) else None
    horizon = int(horizon_value) if horizon_value is not None else 100
    bundle: dict[str, Any] = {"arms": {}, "fields": [], "time_max_hours": 0.0}
    fields = task_record.get("frozen_task_config", {}).get("dataset", {}).get("field_names", []) if isinstance(task_record, dict) else []
    bundle["fields"] = [str(field) for field in fields] if isinstance(fields, list) else []
    for arm in arms:
        status = _arm_status(inputs, task_id, arm, horizon)
        runs = status.get("runs", [])
        run = runs[0] if len(runs) == 1 else None
        evaluations = [
            row for row in _uniform_rows(run)
            if 10 <= float(row["epoch"]) <= min(horizon, 100)
        ]
        time_values = []
        for row in evaluations:
            seconds = _number(row.get("cold_process_seconds_including_profile_acquisition"))
            if seconds is None:
                seconds = _number(row.get("completion_seconds"))
            if seconds is not None and seconds >= 0:
                time_values.append(seconds / 3600.0)
        if time_values:
            bundle["time_max_hours"] = max(bundle["time_max_hours"], max(time_values))
        end_epoch = run.get("accepted_end_epoch") if isinstance(run, dict) else None
        bundle["arms"][arm] = {
            "evaluations": evaluations,
            "status": status.get("status", "pending_run"),
            "detail": status.get("detail", "pending run"),
            "end_epoch": end_epoch,
        }
    return bundle


def _recovery_probe_status(inputs: dict[str, Any]) -> tuple[str, str]:
    receipt = inputs.get("recovery_probe")
    if not isinstance(receipt, dict):
        return "pending_receipt", "No recovery-probe result receipt is present."
    status = str(receipt.get("status", "pending"))
    if status != "verified":
        return status, str(receipt.get("reason", "Recovery-probe receipt does not record verified status."))
    required = {
        "task_id": "D",
        "arm": "strategy",
        "checkpoint_step": 76,
        "next_execution_mapping_match": True,
        "replay_update_diagnostics_match": True,
        "source_and_config_bindings_unchanged": True,
        "resumed_same_lineage": True,
    }
    mismatches = [key for key, expected in required.items() if receipt.get(key) != expected]
    if mismatches:
        return "invalid_receipt", "Verified label lacks required matching fields: " + ", ".join(mismatches)
    return "verified", "D/strategy recovery probe matches the predeclared cursor, replay, binding, and lineage checks."


def _quality_limits(inputs: dict[str, Any], task_id: str) -> dict[str, float]:
    contract = (inputs.get("inventory") or {}).get("quality_contract", {})
    task = ((inputs.get("inventory") or {}).get("tasks", {}) or {}).get(task_id, {})
    dataset = task.get("frozen_task_config", {}).get("dataset", {}) if isinstance(task, dict) else {}
    fields = dataset.get("field_names", []) if isinstance(dataset, dict) else []
    limits = {
        "coherence_sum": _number(contract.get("additive_ratio_max")) or 0.90,
        "family_A": _number(contract.get("family_ratio_max")) or 1.05,
        "family_B": _number(contract.get("family_ratio_max")) or 1.05,
        "family_T": _number(contract.get("family_ratio_max")) or 1.05,
        "endpoint_aggregate": _number(contract.get("aggregate_endpoint_ratio_max")) or 1.05,
    }
    field_limit = _number(contract.get("each_field_ratio_max")) or 1.05
    if isinstance(fields, list):
        limits.update({f"endpoint_field:{field}": field_limit for field in fields})
    return limits


def _ratios_for_quality_check(row: dict[str, Any], fields: list[str]) -> dict[str, float | None]:
    ratios = row.get("ratios") if isinstance(row.get("ratios"), dict) else {}
    values: dict[str, float | None] = {
        name: _number(ratios.get(name))
        for name in ("coherence_sum", "family_A", "family_B", "family_T", "endpoint_aggregate")
    }
    endpoint_fields = ratios.get("endpoint_fields") if isinstance(ratios.get("endpoint_fields"), dict) else {}
    values.update({f"endpoint_field:{field}": _number(endpoint_fields.get(field)) for field in fields})
    return values


def _raw_history_diagnostics(
    evaluations: list[dict[str, Any]], limits: dict[str, float], fields: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Summarize raw per-check target occupancy and worst excursions without pooling checks."""
    counts = {name: 0 for name in limits}
    maxima: dict[str, float | None] = {name: None for name in limits}
    observed_checks = 0
    whole_target_checks = 0
    for row in evaluations:
        values = _ratios_for_quality_check(row, fields)
        if any(values.get(name) is None for name in limits):
            continue
        observed_checks += 1
        passes = True
        for name, limit in limits.items():
            value = float(values[name])
            maxima[name] = value if maxima[name] is None else max(float(maxima[name]), value)
            if value > limit:
                counts[name] += 1
                passes = False
        whole_target_checks += bool(passes)
    excursions = {}
    for name, max_value in maxima.items():
        if max_value is not None and max_value > limits[name]:
            excursions[name] = {
                "max_ratio": max_value,
                "limit": limits[name],
                "max_excess": max_value - limits[name],
                "raw_violation_count": counts[name],
                "raw_check_count": observed_checks,
            }
    occupancy = {
        "target_checks": whole_target_checks,
        "check_count": observed_checks,
        "fraction": whole_target_checks / observed_checks if observed_checks else None,
    }
    return occupancy, excursions


def _mature_window_metric_rows(
    inputs: dict[str, Any], task_id: str, arm: str, run: dict[str, Any] | None
) -> list[dict[str, Any]]:
    if not isinstance(run, dict):
        return []
    assessment = run.get("quality_assessment")
    if not isinstance(assessment, dict):
        return []
    records = []
    for window in assessment.get("mature_windows", []):
        if not isinstance(window, dict):
            continue
        for metric_name, metric in (window.get("metrics") or {}).items():
            if not isinstance(metric, dict):
                continue
            pooled = metric.get("pooled_raw") if isinstance(metric.get("pooled_raw"), dict) else {}
            value_range = metric.get("range") if isinstance(metric.get("range"), list) else []
            records.append(
                {
                    "task": task_id,
                    "arm": arm,
                    "run_id": run.get("run_id"),
                    "window_end_epoch": window.get("epoch"),
                    "window_open_epoch": window.get("window_open_epoch"),
                    "window_interval": window.get("window_interval"),
                    "window_complete": bool(window.get("complete", False)),
                    "window_meets_target": bool(window.get("meets_target", False)),
                    "raw_target_checks": window.get("raw_target_checks"),
                    "raw_check_count": window.get("raw_check_count"),
                    "metric": metric_name,
                    "pooled_numerator": _number(pooled.get("numerator")),
                    "pooled_count": _number(pooled.get("count")),
                    "pooled_source_numerator": _number(pooled.get("source_numerator")),
                    "pooled_source_count": _number(pooled.get("source_count")),
                    "pooled_ratio": _number(metric.get("pooled_ratio")),
                    "mean_check_ratio": _number(metric.get("mean_check_ratio")),
                    "median_check_ratio": _number(metric.get("median_check_ratio")),
                    "slope_per_epoch": _number(metric.get("slope_per_epoch")),
                    "range_min": _number(value_range[0]) if len(value_range) == 2 else None,
                    "range_max": _number(value_range[1]) if len(value_range) == 2 else None,
                    "raw_check_violations": metric.get("raw_check_violations"),
                    "limit": _number(metric.get("limit")),
                }
            )
    return records


def _decision_cost_quality_records(
    inputs: dict[str, Any], attempts: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inventory_tasks = (inputs.get("inventory") or {}).get("tasks", {})
    profile_rows = {item["task_id"]: item for item in _profile_cost_rows(inputs, attempts)}
    recovery_status, recovery_detail = _recovery_probe_status(inputs)
    records: list[dict[str, Any]] = []
    window_records: list[dict[str, Any]] = []
    if not isinstance(inventory_tasks, dict):
        return records, window_records
    for task_id, task in inventory_tasks.items():
        if not isinstance(task, dict):
            continue
        arms = task.get("arms", [])
        if not isinstance(arms, list):
            continue
        horizon = int(_number(task.get("horizon")) or 100)
        for arm_value in arms:
            arm = str(arm_value)
            status = _arm_status(inputs, str(task_id), arm, horizon)
            run = status.get("run")
            if not isinstance(run, dict):
                runs = status.get("runs", [])
                run = runs[0] if len(runs) == 1 else None
            native_history = run.get("native_history") if isinstance(run, dict) else None
            native_integrity = native_history.get("integrity_status") if isinstance(native_history, dict) else None
            evaluations = _uniform_rows(run)
            latest = evaluations[-1] if evaluations else {}
            fields = task.get("frozen_task_config", {}).get("dataset", {}).get("field_names", [])
            fields = [str(field) for field in fields] if isinstance(fields, list) else []
            limits = _quality_limits(inputs, str(task_id))
            occupancy, excursions = _raw_history_diagnostics(evaluations, limits, fields)
            assessment = run.get("quality_assessment") if isinstance(run, dict) and isinstance(run.get("quality_assessment"), dict) else {}
            confirmation = assessment.get("confirmation") if isinstance(assessment.get("confirmation"), dict) else {}
            retention = assessment.get("retention") if isinstance(assessment.get("retention"), dict) else {}
            retention_final_window = retention.get("final_window") if isinstance(retention.get("final_window"), dict) else None
            retention_final_metrics = (
                retention_final_window.get("metrics")
                if isinstance(retention_final_window, dict) and isinstance(retention_final_window.get("metrics"), dict)
                else None
            )
            retention_final_pooled_ratios = (
                {
                    str(metric): values.get("pooled_ratio")
                    for metric, values in retention_final_metrics.items()
                    if isinstance(values, dict) and "pooled_ratio" in values
                }
                if isinstance(retention_final_metrics, dict)
                else None
            )
            confirmation_status = str(confirmation.get("status", "pending"))
            confirm_epoch = confirmation.get("confirmation_epoch")
            confirm_seconds = _number(confirmation.get("completion_seconds")) if confirmation_status == "confirmed" else None
            censor_epoch = confirmation.get("censor_epoch") if confirmation_status == "right_censored" else None
            censor_seconds = _number(confirmation.get("completion_seconds")) if confirmation_status == "right_censored" else None
            if not assessment:
                confirmation_status = "pending"
                if status.get("status") in {"censored", "running", "pending_end_epoch"}:
                    censor_epoch = run.get("accepted_end_epoch") if isinstance(run, dict) else None
                    censor_seconds = _number(latest.get("cold_process_seconds_including_profile_acquisition"))
            if confirmation_status == "right_censored":
                retention_status = "not_established_no_confirmed_attainment"
            elif confirmation_status == "confirmed":
                retention_status = str(retention.get("status", "not_assessed"))
            elif not assessment:
                retention_status = "pending_run_or_assessment"
            else:
                retention_status = str(retention.get("status", "not_established_no_confirmed_attainment"))
            after_confirmation = []
            if confirm_epoch is not None:
                after_confirmation = [row for row in evaluations if _number(row.get("epoch")) is not None and float(row["epoch"]) > float(confirm_epoch)]
            post_confirm_occupancy, post_confirm_excursions = _raw_history_diagnostics(after_confirmation, limits, fields)

            if not run:
                execution_status = "not_started"
            elif native_integrity and native_integrity != "ok":
                execution_status = f"history_{native_integrity}"
            elif status.get("status") == "complete":
                execution_status = "completed_native_history_valid"
            elif run.get("status") == "running" and native_integrity == "ok":
                execution_status = "running_native_history_valid"
            else:
                execution_status = str(status.get("status", "not_verified"))

            profile_row = profile_rows.get(str(task_id), {})
            profile_actual_ledger = profile_row.get("profile_charge")
            profile_measured = _number(run.get("task_profile_acquisition_seconds_measured")) if isinstance(run, dict) else None
            profile_estimated = _number(run.get("task_profile_acquisition_seconds_estimated")) if isinstance(run, dict) else None
            profile_total = _number(run.get("task_profile_acquisition_seconds")) if isinstance(run, dict) else None
            if profile_measured is None:
                profile_measured = _number(latest.get("task_profile_acquisition_seconds_measured"))
            if profile_estimated is None:
                profile_estimated = _number(latest.get("task_profile_acquisition_seconds_estimated"))
            if profile_total is None:
                profile_total = _number(latest.get("task_profile_acquisition_seconds"))
            allocated_measured = _number(run.get("allocated_profile_acquisition_seconds_measured")) if isinstance(run, dict) else None
            allocated_estimated = _number(run.get("allocated_profile_acquisition_seconds_estimated")) if isinstance(run, dict) else None
            allocated_total = _number(run.get("allocated_profile_acquisition_seconds")) if isinstance(run, dict) else None
            if allocated_measured is None:
                allocated_measured = _number(latest.get("allocated_profile_acquisition_seconds_measured"))
            if allocated_estimated is None:
                allocated_estimated = _number(latest.get("allocated_profile_acquisition_seconds_estimated"))
            if allocated_total is None:
                allocated_total = _number(latest.get("allocated_profile_acquisition_seconds"))
            if arm != "strategy":
                allocated_measured = allocated_estimated = allocated_total = 0.0

            decision = inputs.get("plans", {}).get(str(task_id)) or {}
            task_role = task.get("role", "optional")
            charged_train = _charged(attempts, task_id=str(task_id), arm=arm, actions={"launch", "resume"})
            training_charge = _terminal_charge_summary(
                attempts,
                task_id=str(task_id),
                arm=arm,
                actions={"launch", "resume"},
            )
            evaluation_charge = _terminal_charge_summary(
                attempts,
                task_id=str(task_id),
                arm=arm,
                actions={"evaluate"},
                zero_if_absent=True,
            )
            if not evaluation_charge["attempt_count"]:
                evaluation_status = "no_recorded_evaluate_attempts"
            elif evaluation_charge["complete"]:
                evaluation_status = "terminal_charges_complete_for_recorded_evaluations"
            else:
                evaluation_status = "recorded_evaluations_incomplete_or_unmeasured"
            observed_cold_total = None
            if (
                training_charge["complete"]
                and training_charge["seconds"] is not None
                and allocated_total is not None
                and evaluation_charge["complete"]
                and evaluation_charge["seconds"] is not None
            ):
                observed_cold_total = (
                    training_charge["seconds"]
                    + allocated_total
                    + evaluation_charge["seconds"]
                )
            latest_ratios = latest.get("ratios") if isinstance(latest.get("ratios"), dict) else {}
            records.append(
                {
                    "task_id": str(task_id),
                    "arm": arm,
                    "role": str(task_role),
                    "run_id": run.get("run_id") if isinstance(run, dict) else None,
                    "run_status": run.get("status") if isinstance(run, dict) else status.get("status"),
                    "arm_evidence_status": status.get("status"),
                    "arm_detail": status.get("detail"),
                    "accepted_end_epoch": run.get("accepted_end_epoch") if isinstance(run, dict) else None,
                    "native_history_integrity": native_integrity,
                    "execution_correctness_status": execution_status,
                    "recovery_probe_status": recovery_status,
                    "recovery_probe_detail": recovery_detail,
                    "task_profile_acquisition_measured_seconds": profile_measured,
                    "task_profile_acquisition_estimated_seconds": profile_estimated,
                    "task_profile_acquisition_total_seconds": profile_total,
                    "task_profile_acquisition_ledger_charged_seconds": profile_actual_ledger,
                    "allocated_profile_acquisition_measured_seconds": allocated_measured,
                    "allocated_profile_acquisition_estimated_seconds": allocated_estimated,
                    "allocated_profile_acquisition_total_seconds": allocated_total,
                    "charged_training_process_seconds": charged_train,
                    "terminal_training_process_seconds": training_charge["seconds"],
                    "training_process_charge_status": (
                        "terminal_charges_complete"
                        if training_charge["complete"]
                        else "incomplete_or_unmeasured"
                        if training_charge["attempt_count"]
                        else "no_launch_or_resume_attempts"
                    ),
                    "charged_final_evaluation_process_seconds": evaluation_charge["seconds"],
                    "final_evaluation_attempt_count": evaluation_charge["attempt_count"],
                    "final_evaluation_terminal_attempt_count": evaluation_charge["terminal_attempt_count"],
                    "final_evaluation_pending_attempt_count": evaluation_charge["pending_attempt_count"],
                    "final_evaluation_unmeasured_terminal_attempt_count": evaluation_charge["unmeasured_terminal_attempt_count"],
                    "final_evaluation_charge_status": evaluation_status,
                    "final_evaluation_scope": "all recorded evaluate commands; final-audit completeness is separate",
                    "observed_cold_task_total_seconds": observed_cold_total,
                    "observed_cold_task_total_scope": (
                        "terminal launch/resume charges + allocated profile total (measured and recorded estimate) + all recorded terminal evaluate charges; no planned audit inferred"
                    ),
                    "selected_plan": decision.get("plan_id"),
                    "selected_forecast_seconds": _number(decision.get("predicted_total_s")),
                    "full_reference_forecast_seconds": _number(decision.get("predicted_full_total_s")),
                    "projected_savings_ratio": _number(decision.get("cost_savings_ratio")),
                    "latest_native_epoch": latest.get("epoch"),
                    "latest_accounted_cold_process_seconds": (
                        _number(latest.get("cold_process_seconds_including_profile_acquisition"))
                        if _number(latest.get("cold_process_seconds_including_profile_acquisition")) is not None
                        else _number(latest.get("completion_seconds"))
                    ),
                    "latest_wall_seconds": _number(latest.get("wall_seconds_from_first_training_command")),
                    "latest_native_ratios": latest_ratios,
                    "confirmation_status": confirmation_status,
                    "confirmation_epoch": confirm_epoch,
                    "confirmation_accounted_cold_seconds": confirm_seconds,
                    "censor_epoch": censor_epoch,
                    "censor_accounted_cold_seconds": censor_seconds,
                    "retention_status": retention_status,
                    "retention_window_epochs": retention.get("window_epochs"),
                    "retention_window_target_count": retention.get("window_target_count"),
                    "retention_window_count": retention.get("window_count"),
                    "retention_window_target_fraction": retention.get("window_target_fraction"),
                    "retention_raw_check_target_count": retention.get("raw_check_target_count"),
                    "retention_raw_check_count": retention.get("raw_check_count"),
                    "retention_raw_check_target_fraction": retention.get("raw_check_target_fraction"),
                    "retention_final_window": retention_final_window,
                    "retention_final_window_pooled_ratios": retention_final_pooled_ratios,
                    "retention_excursions": retention.get("excursions"),
                    "retention_censored_spans": retention.get("censored_spans"),
                    "raw_whole_target_occupancy": occupancy,
                    "post_confirmation_raw_whole_target_occupancy": post_confirm_occupancy if confirm_epoch is not None else None,
                    "worst_raw_excursions": excursions,
                    "post_confirmation_worst_raw_excursions": post_confirm_excursions if confirm_epoch is not None else None,
                    "mature_window_statuses": {
                        str(endpoint): _window_status(
                            next((item for item in assessment.get("mature_windows", []) if isinstance(item, dict) and item.get("epoch") == endpoint), None)
                        )
                        for endpoint in (60, 70, 80, 90, 100)
                    },
                    "mature_window_occupancy": {
                        str(item.get("epoch")): {
                            "target_checks": item.get("raw_target_checks"),
                            "check_count": item.get("raw_check_count"),
                            "fraction": (
                                item.get("raw_target_checks") / item.get("raw_check_count")
                                if isinstance(item.get("raw_target_checks"), (int, float))
                                and isinstance(item.get("raw_check_count"), (int, float))
                                and item.get("raw_check_count") > 0
                                else None
                            ),
                        }
                        for item in assessment.get("mature_windows", [])
                        if isinstance(item, dict) and item.get("epoch") in (60, 70, 80, 90, 100)
                    },
                }
            )
            window_records.extend(_mature_window_metric_rows(inputs, str(task_id), arm, run))
    return records, window_records


def _window_status(window: dict[str, Any] | None) -> str:
    if not isinstance(window, dict):
        return "pending"
    if not window.get("complete", False):
        return "censored"
    return "pass" if window.get("meets_target", False) else "fail"


def _source_calibration_rows(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    root = inputs["root"]
    for task_id in DEFAULT_TASKS:
        calibration = inputs.get("calibrations", {}).get(task_id)
        contract = inputs.get("contracts", {}).get(task_id)
        reference = calibration.get("reference", {}) if isinstance(calibration, dict) else {}
        c0 = _number(reference.get("c0")) if isinstance(reference, dict) else None
        contract_c0 = _number(contract.get("c0")) if isinstance(contract, dict) else None
        calibration_path = root / "profiles" / f"{task_id}_source_calibration.json"
        identity = contract.get("source_calibration_identity") if isinstance(contract, dict) else None
        matches = c0 is not None and contract_c0 is not None and c0 == contract_c0
        rows.append(
            {
                "task_id": task_id,
                "train_c0": c0,
                "contract_c0": contract_c0,
                "c0_exact_match": matches,
                "source_calibration_identity": identity,
                "source_calibration_file": _rel(calibration_path, root),
                "source_calibration_sha256": _sha256(calibration_path),
                "calibration_status": contract.get("calibration_status") if isinstance(contract, dict) else None,
            }
        )
    return rows


def _has_fixed_query_attestation(inputs: dict[str, Any]) -> bool:
    attestation = inputs.get("fixed_query_attestation")
    if not isinstance(attestation, dict):
        return False
    return bool(
        attestation.get("status") == "static_invariant_under_frozen_fixed_shape_H5_contract"
        and attestation.get("authorized_split") == "validation"
        and attestation.get("queries_per_snapshot") == 4096
        and isinstance(attestation.get("tasks"), list)
        and all(isinstance(task_id, str) for task_id in attestation["tasks"])
        and {"G", "D", "X"}.issubset(set(attestation["tasks"]))
    )


def _fixed_query_count_text(inputs: dict[str, Any]) -> str:
    relative = _rel(inputs.get("fixed_query_attestation_path", Path("analysis/fixed_query_count_attestation.json")), inputs["root"])
    if not _has_fixed_query_attestation(inputs):
        return "The native history does not serialize per-check valid-query counts; the 4,096-query count is not independently verified for this input set."
    return (
        f"The native JSONL omits valid-mask counts, so 4,096 valid queries per validation snapshot are reconstructed from the frozen equal-shape HDF5 loader and all-valid query-mask invariant, not read from serialized per-check counts ([attestation](../{relative})). "
        "This code-derived count applies only to the current fixed-shape loader; masked or variable-length future loaders require serialized valid counts. The attestation opened no dataset bytes or validators."
    )


def _bottleneck_scope_text(inputs: dict[str, Any]) -> str:
    evidence = inputs.get("bottleneck_scope_limitations")
    relative = _rel(inputs.get("bottleneck_scope_limitations_path", Path("analysis/bottleneck_scope_limitations.json")), inputs["root"])
    task_rows = evidence.get("tasks", {}) if isinstance(evidence, dict) else {}
    if (
        not isinstance(evidence, dict)
        or evidence.get("status") != "diagnostic_trace_with_incomplete_phase_attribution"
        or not {"G", "D", "X"}.issubset(set(task_rows))
    ):
        return "The optional one-update profiler trace is absent or incomplete, so no exclusive phase-share or Amdahl fraction is reported."
    return (
        f"The one-update trace is diagnostic only ([scope limitations](../{relative})): family CPU scope values are zero but do not imply zero host work, public reconstruction bypasses the module-forward hook, and data preparation is outside candidate update ticks. "
        "Those traces identify no exclusive reconstruction, data-preparation, I/O, CPU, or GPU phase share and no numeric Amdahl affected fraction. Production whole-update timings separately cover four batches but exclude prebuilt-batch I/O; actual command charges include setup, data I/O, validation, and checkpoint work."
    )


def _preserved_negative_evidence_text(inputs: dict[str, Any]) -> str:
    path = inputs.get("fast_report_path")
    if not isinstance(path, Path) or not path.is_file() or path.is_symlink():
        return "The retained FAST report was unavailable to this report build, so its historical periodic-reuse findings are not restated."
    relative = _rel(path, inputs["root"])
    digest = _sha256(path)
    return (
        f"Preserved negative evidence: the signed ConFIG coefficient oracle recovered its declared FP64 reference, but both periodic reuse cadences failed the fixed-SOURCE direction-cosine threshold 0.9: the four-update sequence reached 0.891 and the two-update reuse pair reached 0.800 ([retained FAST report](../{relative}#L11), SHA-256 `{digest}`). "
        "The periodic arm stayed quarantined. The earlier Goal 1 and ordered 100-epoch histories did not establish sustained routine-history retention; the ordered study remains historical and was not continued."
    )


def _profile_proxy_erratum_text(inputs: dict[str, Any]) -> str:
    erratum = inputs.get("profile_proxy_label_erratum")
    path = inputs.get("profile_proxy_label_erratum_path", Path("analysis/profile_proxy_label_erratum.json"))
    relative = _rel(path, inputs["root"])
    if not isinstance(erratum, dict) or erratum.get("schema_version") != 1:
        return "No profile proxy-label erratum was present; output-space and parameter-gradient claims are limited to the profile fields as saved."
    digest = _sha256(path)
    return (
        f"Profile metadata correction ([erratum](../{relative}), SHA-256 `{digest}`): `batches[].output_space_proxy_norms` are unnormalized gradients with respect to reconstructed endpoint predictions, not parameter-gradient norms; the retained `proxy_label` text was wrong. "
        "Selection uses the separate weighted/C0-normalized model-parameter gradient arrays (`parameter_execution_variances` and `intrinsic_parameter_variance`); numerical arrays and frozen source were unchanged."
    )


def _parameter_expectation_summary(inputs: dict[str, Any]) -> str:
    parts = []
    limit = 5e-4
    for task_id in DEFAULT_TASKS:
        raw = (inputs.get("raw_profiles") or {}).get(task_id)
        batches = raw.get("batches", []) if isinstance(raw, dict) else []
        errors = [_number(batch.get("parameter_expectation_relative_error")) for batch in batches if isinstance(batch, dict)]
        if len(errors) != 4 or any(value is None for value in errors):
            parts.append(f"{task_id}: not recorded for four fixed TRAIN batches")
            continue
        maximum = max(float(value) for value in errors if value is not None)
        status = "within" if maximum <= limit else "above"
        parts.append(f"{task_id}: max relative error {maximum:.3e} ({status} 5e-4 across four fixed TRAIN batches)")
    summary = "; ".join(parts)
    return summary + ". This is a finite estimator-expectation software correctness check, not a confidence bound, convergence result, or trajectory-identity claim; X is software-only and does not establish learning transfer."


DECISION_CSV_COLUMNS = (
    "task_id", "arm", "role", "run_id", "run_status", "arm_evidence_status", "arm_detail",
    "accepted_end_epoch", "native_history_integrity", "execution_correctness_status", "recovery_probe_status",
    "task_profile_acquisition_measured_seconds", "task_profile_acquisition_estimated_seconds",
    "task_profile_acquisition_total_seconds", "task_profile_acquisition_ledger_charged_seconds",
    "allocated_profile_acquisition_measured_seconds", "allocated_profile_acquisition_estimated_seconds",
    "allocated_profile_acquisition_total_seconds", "charged_training_process_seconds",
    "terminal_training_process_seconds", "training_process_charge_status",
    "charged_final_evaluation_process_seconds", "final_evaluation_attempt_count",
    "final_evaluation_terminal_attempt_count", "final_evaluation_pending_attempt_count",
    "final_evaluation_unmeasured_terminal_attempt_count", "final_evaluation_charge_status",
    "final_evaluation_scope", "observed_cold_task_total_seconds", "observed_cold_task_total_scope",
    "selected_plan",
    "selected_forecast_seconds", "full_reference_forecast_seconds", "projected_savings_ratio",
    "latest_native_epoch", "latest_accounted_cold_process_seconds", "latest_wall_seconds",
    "latest_native_ratios_json", "confirmation_status", "confirmation_epoch",
    "confirmation_accounted_cold_seconds", "censor_epoch", "censor_accounted_cold_seconds",
    "retention_status", "retention_window_epochs_json", "retention_window_target_count", "retention_window_count",
    "retention_window_target_fraction", "retention_raw_check_target_count", "retention_raw_check_count",
    "retention_raw_check_target_fraction", "retention_final_window_json", "retention_final_window_pooled_ratios_json",
    "retention_excursions_json", "retention_censored_spans_json",
    "raw_whole_target_occupancy_json", "post_confirmation_raw_whole_target_occupancy_json",
    "worst_raw_excursions_json", "post_confirmation_worst_raw_excursions_json",
    "mature_window_statuses_json", "mature_window_occupancy_json",
)

WINDOW_CSV_COLUMNS = (
    "task", "arm", "run_id", "window_end_epoch", "window_open_epoch", "window_interval",
    "window_complete", "window_meets_target", "raw_target_checks", "raw_check_count", "metric",
    "pooled_numerator", "pooled_count", "pooled_source_numerator", "pooled_source_count",
    "pooled_ratio", "mean_check_ratio", "median_check_ratio", "slope_per_epoch", "range_min",
    "range_max", "raw_check_violations", "limit",
)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...], *, json_fields: set[str] | None = None) -> None:
    json_fields = json_fields or set()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            for key in json_fields:
                value = encoded.get(key)
                encoded[key] = json.dumps(value, sort_keys=True, allow_nan=False) if value is not None else ""
            writer.writerow(encoded)
    os.replace(temporary, path)


def _write_analysis_tables(inputs: dict[str, Any], attempts: list[dict[str, Any]]) -> None:
    records, windows = _decision_cost_quality_records(inputs, attempts)
    root = inputs["root"]
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    recovery_status, recovery_detail = _recovery_probe_status(inputs)
    payload = {
        "schema": "cost_aware_decision_cost_quality.v1",
        "created_at_utc": created,
        "root": str(root),
        "source_history": _rel(inputs["history_path"], root),
        "source_history_sha256": _sha256(inputs["history_path"]),
        "recovery_probe": {"status": recovery_status, "detail": recovery_detail},
        "records": records,
    }
    json_path = analysis / "decision_cost_quality.json"
    temporary = json_path.with_name(f".{json_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, json_path)
    flattened = []
    json_fields = {
        "latest_native_ratios_json": "latest_native_ratios",
        "retention_window_epochs_json": "retention_window_epochs",
        "retention_final_window_json": "retention_final_window",
        "retention_final_window_pooled_ratios_json": "retention_final_window_pooled_ratios",
        "retention_excursions_json": "retention_excursions",
        "retention_censored_spans_json": "retention_censored_spans",
        "raw_whole_target_occupancy_json": "raw_whole_target_occupancy",
        "post_confirmation_raw_whole_target_occupancy_json": "post_confirmation_raw_whole_target_occupancy",
        "worst_raw_excursions_json": "worst_raw_excursions",
        "post_confirmation_worst_raw_excursions_json": "post_confirmation_worst_raw_excursions",
        "mature_window_statuses_json": "mature_window_statuses",
        "mature_window_occupancy_json": "mature_window_occupancy",
    }
    for record in records:
        row = {key: value for key, value in record.items() if key not in {
            "latest_native_ratios", "retention_window_epochs", "retention_final_window",
            "retention_final_window_pooled_ratios", "retention_excursions", "retention_censored_spans",
            "raw_whole_target_occupancy", "post_confirmation_raw_whole_target_occupancy",
            "worst_raw_excursions", "post_confirmation_worst_raw_excursions",
            "mature_window_statuses", "mature_window_occupancy",
        }}
        row.update({csv_key: record.get(json_key) for csv_key, json_key in json_fields.items()})
        flattened.append(row)
    _write_csv(analysis / "decision_cost_quality.csv", flattened, DECISION_CSV_COLUMNS, json_fields=set(json_fields))
    _write_csv(analysis / "mature_window_metrics.csv", windows, WINDOW_CSV_COLUMNS)
    inputs["decision_cost_records"] = records
    inputs["mature_window_metric_records"] = windows
    inputs["calibration_rows"] = _source_calibration_rows(inputs)


def _quality_common_scales(bundles: dict[str, dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """Choose shared y ranges across G/D pages for each comparable metric group."""
    values: dict[str, list[float]] = {"sum": [], "families": [], "fields": []}
    for bundle in bundles.values():
        for arm_data in bundle["arms"].values():
            for row in arm_data["evaluations"]:
                ratios = row.get("ratios") if isinstance(row.get("ratios"), dict) else {}
                value = _number(ratios.get("coherence_sum"))
                if value is not None:
                    values["sum"].append(value)
                for family in ("family_A", "family_B", "family_T"):
                    value = _number(ratios.get(family))
                    if value is not None:
                        values["families"].append(value)
                endpoints = ratios.get("endpoint_fields") if isinstance(ratios.get("endpoint_fields"), dict) else {}
                values["fields"].extend(value for value in (_number(item) for item in endpoints.values()) if value is not None)
                endpoint_aggregate = _number(ratios.get("endpoint_aggregate"))
                if endpoint_aggregate is not None:
                    values["fields"].append(endpoint_aggregate)

    def shared_ylim(group: str, threshold: float) -> tuple[float, float]:
        maximum = max([1.0, threshold, *values[group]])
        return (0.0, max(threshold + 0.08, maximum * 1.06))

    return {
        "sum": shared_ylim("sum", 0.90),
        "families": shared_ylim("families", 1.05),
        "fields": shared_ylim("fields", 1.05),
    }


def _quality_status_summary(bundle: dict[str, Any], arms: list[str]) -> str:
    labels = {"dense": "Dense", "heuristic": "Fixed heuristic", "strategy": "Strategy"}
    pieces = []
    for arm in arms:
        item = bundle["arms"][arm]
        evaluations = item["evaluations"]
        label = f"{labels.get(arm, arm)}: {str(item['status']).replace('_', ' ')}"
        if item["end_epoch"] is not None:
            label += f" through E{item['end_epoch']}"
        if evaluations:
            label += f" ({len(evaluations)} native checks)"
        pieces.append(label)
    return "  ·  ".join(pieces)


def _plot_quality_series(
    ax: Any,
    bundle: dict[str, Any],
    *,
    metric_group: str,
    x_mode: str,
    arms: list[str],
    arm_colors: dict[str, str],
    arm_styles: dict[str, str],
    arm_markers: dict[str, str],
    fields: list[str],
) -> int:
    """Plot observed rows only; NaNs retain gaps where values/times are missing."""
    plotted = 0
    for arm in arms:
        rows = bundle["arms"][arm]["evaluations"]
        if not rows:
            continue
        if metric_group == "sum":
            entries = [("coherence_sum", arm_colors.get(arm, "#777777"))]
        elif metric_group == "families":
            entries = [(name, METRIC_COLORS[name]) for name in ("family_A", "family_B", "family_T")]
        else:
            entries = [("endpoint_aggregate", "#111111")]
            entries.extend((field, FIELD_PALETTE[index % len(FIELD_PALETTE)]) for index, field in enumerate(fields))
        for key, color in entries:
            xs: list[float] = []
            ys: list[float] = []
            for row in rows:
                epoch = _number(row.get("epoch"))
                if x_mode == "hours":
                    seconds = _number(row.get("cold_process_seconds_including_profile_acquisition"))
                    if seconds is None:
                        seconds = _number(row.get("completion_seconds"))
                    x_value = seconds / 3600.0 if seconds is not None and seconds >= 0 else None
                else:
                    x_value = epoch
                ratios = row.get("ratios") if isinstance(row.get("ratios"), dict) else {}
                if metric_group == "fields":
                    if key == "endpoint_aggregate":
                        y_value = _number(ratios.get("endpoint_aggregate"))
                    else:
                        endpoint_values = ratios.get("endpoint_fields") if isinstance(ratios.get("endpoint_fields"), dict) else {}
                        y_value = _number(endpoint_values.get(key))
                else:
                    y_value = _number(ratios.get(key))
                if x_value is None or y_value is None:
                    xs.append(float("nan"))
                    ys.append(float("nan"))
                else:
                    xs.append(float(x_value))
                    ys.append(float(y_value))
            if any(math.isfinite(value) for value in ys):
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    linestyle=arm_styles.get(arm, "-"),
                    marker=arm_markers.get(arm, "o"),
                    markersize=3.0,
                    linewidth=1.05,
                    alpha=0.95,
                    zorder=3,
                )
                plotted += 1
    return plotted


def _write_quality_figure(inputs: dict[str, Any], path: Path) -> None:
    plt, PdfPages, _Rectangle = _matplotlib(inputs["root"])
    from matplotlib.lines import Line2D

    inventory_tasks = (inputs.get("inventory") or {}).get("tasks", {})
    task_arms = {}
    bundles = {}
    for task_id in ("G", "D"):
        task_value = inventory_tasks.get(task_id, {}) if isinstance(inventory_tasks, dict) else {}
        arms = task_value.get("arms", list(MANDATORY_FALLBACK.get(task_id, ()))) if isinstance(task_value, dict) else list(MANDATORY_FALLBACK.get(task_id, ()))
        task_arms[task_id] = [str(arm) for arm in arms]
        bundles[task_id] = _quality_history_bundle(inputs, task_id, task_arms[task_id])
    ylimits = _quality_common_scales(bundles)
    max_hours = max((bundle["time_max_hours"] for bundle in bundles.values()), default=0.0)
    common_hours_xlim = (0.0, max(0.1, max_hours * 1.04))

    arm_colors = {"dense": "#59636e", "heuristic": "#e49b2f", "strategy": "#4c78a8"}
    arm_styles = {"dense": "-", "heuristic": "--", "strategy": "-."}
    arm_markers = {"dense": "o", "heuristic": "s", "strategy": "^"}
    family_names = {"family_A": "A", "family_B": "B", "family_T": "T"}

    with PdfPages(path) as pdf:
        for task_id in ("G", "D"):
            bundle = bundles[task_id]
            arms = task_arms[task_id]
            fields = bundle["fields"]
            fig, axes = plt.subplots(2, 2, figsize=(15.8, 10.6), squeeze=False)
            fig.suptitle(
                f"{task_id}: scheduled native validation against paired SOURCE",
                fontsize=14,
                fontweight="bold",
                y=0.985,
            )
            fig.subplots_adjust(left=0.075, right=0.975, bottom=0.245, top=0.805, wspace=0.22, hspace=0.34)

            # Upper row: method colors make the three task-level trajectories easy to compare.
            ax = axes[0, 0]
            n_plotted = _plot_quality_series(
                ax, bundle, metric_group="sum", x_mode="epoch", arms=arms,
                arm_colors=arm_colors, arm_styles={arm: "-" for arm in arms},
                arm_markers=arm_markers, fields=fields,
            )
            ax.set_title("A. Additive coherence-sum ratio vs epoch", loc="left")
            ax.set_ylabel("child / paired SOURCE additive sum")
            ax.set_xlim(0, 100)
            ax.set_xticks(range(0, 101, 20))
            ax.set_xlabel("training epoch")
            ax.set_ylim(*ylimits["sum"])
            ax.axhline(1.0, color="#222222", linestyle="-", linewidth=0.85, zorder=2)
            ax.axhline(0.90, color="#c44e52", linestyle="--", linewidth=0.9, zorder=2)
            ax.grid(alpha=0.18)
            if not n_plotted:
                ax.text(0.5, 0.5, "No scheduled native validation checks yet", ha="center", va="center", transform=ax.transAxes)

            ax = axes[0, 1]
            n_plotted = _plot_quality_series(
                ax, bundle, metric_group="sum", x_mode="hours", arms=arms,
                arm_colors=arm_colors, arm_styles={arm: "-" for arm in arms},
                arm_markers=arm_markers, fields=fields,
            )
            ax.set_title("B. Same additive ratio vs cold check-completion time", loc="left")
            ax.set_ylabel("child / paired SOURCE additive sum")
            ax.set_xlim(*common_hours_xlim)
            ax.set_xlabel("cold check-completion hours\n(including strategy profile allocation)")
            ax.set_ylim(*ylimits["sum"])
            ax.axhline(1.0, color="#222222", linestyle="-", linewidth=0.85, zorder=2)
            ax.axhline(0.90, color="#c44e52", linestyle="--", linewidth=0.9, zorder=2)
            ax.grid(alpha=0.18)
            if not n_plotted:
                ax.text(0.5, 0.5, "No checks with numeric cold-process time yet", ha="center", va="center", transform=ax.transAxes)

            # Lower row: color identifies family/field and line style identifies the method.
            ax = axes[1, 0]
            n_plotted = _plot_quality_series(
                ax, bundle, metric_group="families", x_mode="epoch", arms=arms,
                arm_colors=arm_colors, arm_styles=arm_styles,
                arm_markers=arm_markers, fields=fields,
            )
            ax.set_title("C. A/B/T family ratios vs epoch", loc="left")
            ax.set_ylabel("child / paired SOURCE family risk")
            ax.set_xlim(0, 100)
            ax.set_xticks(range(0, 101, 20))
            ax.set_xlabel("training epoch")
            ax.set_ylim(*ylimits["families"])
            ax.axhline(1.0, color="#222222", linestyle="-", linewidth=0.85, zorder=2)
            ax.axhline(1.05, color="#c44e52", linestyle="--", linewidth=0.9, zorder=2)
            ax.grid(alpha=0.18)
            if not n_plotted:
                ax.text(0.5, 0.5, "No scheduled native validation checks yet", ha="center", va="center", transform=ax.transAxes)

            ax = axes[1, 1]
            n_plotted = _plot_quality_series(
                ax, bundle, metric_group="fields", x_mode="epoch", arms=arms,
                arm_colors=arm_colors, arm_styles=arm_styles,
                arm_markers=arm_markers, fields=fields,
            )
            ax.set_title("D. Endpoint aggregate and field MSE ratios vs epoch", loc="left")
            ax.set_ylabel("child / paired SOURCE endpoint MSE")
            ax.set_xlim(0, 100)
            ax.set_xticks(range(0, 101, 20))
            ax.set_xlabel("training epoch")
            ax.set_ylim(*ylimits["fields"])
            ax.axhline(1.0, color="#222222", linestyle="-", linewidth=0.85, zorder=2)
            ax.axhline(1.05, color="#c44e52", linestyle="--", linewidth=0.9, zorder=2)
            ax.grid(alpha=0.18)
            if not fields:
                ax.text(0.5, 0.5, "Frozen endpoint field names unavailable", ha="center", va="center", transform=ax.transAxes)
            elif not n_plotted:
                ax.text(0.5, 0.5, "No scheduled native endpoint checks yet", ha="center", va="center", transform=ax.transAxes)

            # Upper method-color key; bottom keys decode family/field colors and method line styles.
            color_handles = [
                Line2D([], [], color=arm_colors.get(arm, "#777777"), linewidth=2.0, label={"dense": "Dense", "heuristic": "Fixed heuristic", "strategy": "Strategy"}.get(arm, arm))
                for arm in arms
            ]
            fig.legend(
                handles=color_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.93),
                ncol=max(1, len(color_handles)),
                frameon=False,
                fontsize=8,
                title="Upper-row color = method",
                title_fontsize=7,
            )
            status_text = _quality_status_summary(bundle, arms)
            fig.text(0.5, 0.845, status_text, ha="center", va="center", fontsize=7.2)

            color_handles = [
                Line2D([], [], color=METRIC_COLORS[name], linewidth=2.0, label=f"Family {family_names[name]}")
                for name in ("family_A", "family_B", "family_T")
            ]
            color_handles.extend(
                Line2D([], [], color=FIELD_PALETTE[index % len(FIELD_PALETTE)], linewidth=2.0, label=f"Field {field}")
                for index, field in enumerate(fields)
            )
            color_handles.insert(0, Line2D([], [], color="#111111", linewidth=2.0, label="Endpoint aggregate"))
            fig.legend(
                handles=color_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.128),
                ncol=min(8, max(1, len(color_handles))),
                frameon=False,
                fontsize=7,
                title="Lower-row color = family or frozen endpoint field",
                title_fontsize=7,
            )
            style_handles = [
                Line2D([], [], color="#333333", linestyle=arm_styles.get(arm, "-"), marker=arm_markers.get(arm, "o"), markersize=4, label={"dense": "Dense", "heuristic": "Fixed heuristic", "strategy": "Strategy"}.get(arm, arm))
                for arm in arms
            ]
            fig.legend(
                handles=style_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.073),
                ncol=max(1, len(style_handles)),
                frameon=False,
                fontsize=7,
                title="Lower-row line style = method",
                title_fontsize=7,
            )
            fig.text(
                0.5,
                0.012,
                (
                    "Source baseline = 1.00; targets: additive sum 0.90, A/B/T and endpoint fields 1.05. Frozen validation panel: 128 snapshots in 4 ordered groups of 32; "
                    + ("4,096 valid queries per snapshot by fixed-shape loader attestation" if _has_fixed_query_attestation(inputs) else "valid-query count not independently attested")
                    + "; no TEST data. Epoch 0 is an axis origin only; no E0 value/time is imputed. Checks and overlapping windows are descriptive, not independent significance tests; elapsed positions use numeric accounted cold process charges."
                ),
                ha="center",
                va="center",
                fontsize=6.6,
                wrap=True,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        component_series = _self_mutual_component_series(inputs)
        for task_id in ("G", "D"):
            task_series = component_series.get(task_id, {})
            if not task_series:
                continue
            metric_paths = sorted({metric for arm_data in task_series.values() for metric in arm_data})
            page_size = 12
            arm_style = {"dense": "-", "heuristic": "--", "strategy": "-."}
            arm_colors = {"dense": "#59636e", "heuristic": "#e49b2f", "strategy": "#4c78a8"}
            for page_start in range(0, len(metric_paths), page_size):
                page_metrics = metric_paths[page_start : page_start + page_size]
                nrows = math.ceil(len(page_metrics) / 4)
                fig, axes = plt.subplots(nrows, 4, figsize=(16, max(8.5, 3.0 * nrows)), squeeze=False)
                fig.suptitle(
                    f"{task_id}: paired self/mutual component histories, page {page_start // page_size + 1}",
                    fontsize=14,
                    fontweight="bold",
                    y=0.975,
                )
                fig.subplots_adjust(left=0.07, right=0.985, bottom=0.13, top=0.83, wspace=0.28, hspace=0.45)
                for metric_index, ax in enumerate(axes.flat):
                    if metric_index >= len(page_metrics):
                        ax.set_visible(False)
                        continue
                    metric_path = page_metrics[metric_index]
                    plotted = 0
                    for arm, metric_data in task_series.items():
                        points = sorted(metric_data.get(metric_path, []), key=lambda item: item[0])
                        valid_points = [point for point in points if point[1] is not None]
                        if valid_points:
                            ax.plot(
                                [point[0] for point in valid_points],
                                [point[1] for point in valid_points],
                                color=arm_colors.get(arm, "#777777"),
                                linestyle=arm_style.get(arm, "-"),
                                marker={"dense": "o", "heuristic": "s", "strategy": "^"}.get(arm, "o"),
                                markersize=2.5,
                                linewidth=0.9,
                            )
                            plotted += 1
                        undefined_points = [point for point in points if point[1] is None]
                        if undefined_points:
                            undefined_text = "; ".join(
                                f"E{point[0]}: SOURCE sum=0, child sum={point[3]:.4g}; ratio undefined"
                                for point in undefined_points
                            )
                            ax.text(0.02, 0.02, undefined_text, transform=ax.transAxes, fontsize=4.8, va="bottom")
                    ax.set_title(metric_path.removeprefix("coherence.components."), loc="left", fontsize=7.3)
                    ax.set_xlim(0, 100)
                    ax.set_xticks(range(0, 101, 20))
                    ax.set_xlabel("training epoch", fontsize=7)
                    ax.set_ylabel("4-group pooled ratio", fontsize=7)
                    ax.axhline(1.0, color="#222222", linestyle=":", linewidth=0.8)
                    ax.grid(alpha=0.18)
                    if not plotted:
                        ax.text(0.5, 0.5, "No paired group history", ha="center", va="center", transform=ax.transAxes, fontsize=7)
                arm_handles = [
                    Line2D([], [], color=arm_colors.get(arm, "#777777"), linestyle=arm_style.get(arm, "-"), marker={"dense": "o", "heuristic": "s", "strategy": "^"}.get(arm, "o"), markersize=4, label=arm)
                    for arm in task_series
                ]
                fig.legend(
                    handles=arm_handles,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.067),
                    ncol=max(1, len(arm_handles)),
                    frameon=False,
                    fontsize=7,
                    title="Method",
                    title_fontsize=7,
                )
                fig.text(
                    0.5,
                    0.008,
                    "Diagnostic only: each point pools the same canonical component across four fixed 32-snapshot groups, with B independently computed in each, as sum(child)/sum(SOURCE); no 128-snapshot covariance is claimed, parent scalars/aliases are excluded, and self/mutual paths are not summed.",
                    ha="center",
                    va="center",
                    fontsize=6.8,
                    wrap=True,
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)


def _self_mutual_component_series(inputs: dict[str, Any]) -> dict[str, dict[str, dict[str, list[tuple[int, float | None, float, float]]]]]:
    """Pool each canonical self/mutual component over four paired B(32) groups."""
    path = inputs.get("components_path")
    if not isinstance(path, Path) or not path.is_file() or path.is_symlink():
        return {}
    rows: list[dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise ReportBuildError(f"cannot read paired component evidence {path}: {error}") from error
    candidate_paths = {
        row.get("metric_path", "")
        for row in rows
        if row.get("panel") == "uniform"
        and row.get("metric_path", "").startswith("coherence.components.")
        and (".self." in row.get("metric_path", "") or ".mutual." in row.get("metric_path", ""))
        and _positive_component_scope(inputs, row.get("task", ""), row.get("metric_path", ""))
    }
    # Keep the canonical leaf diagnostics; parent totals remain in the raw source CSV only.
    leaf_paths = {
        metric_path
        for metric_path in candidate_paths
        if not any(other.startswith(metric_path + ".") for other in candidate_paths)
    }
    grouped: dict[tuple[str, str, int, str], dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        metric_path = row.get("metric_path", "")
        if row.get("panel") != "uniform" or metric_path not in leaf_paths:
            continue
        if not _positive_component_scope(inputs, row.get("task", ""), metric_path):
            continue
        epoch = _csv_number(row.get("epoch"))
        group = row.get("group")
        source = _csv_number(row.get("source_value"))
        child = _csv_number(row.get("child_value"))
        if epoch is None or not math.isclose(epoch, round(epoch), rel_tol=0.0, abs_tol=1e-8) or group is None:
            continue
        if source is None or child is None:
            continue
        grouped[(row.get("task", ""), row.get("arm", ""), round(epoch), metric_path)][str(group)]["source"] = source
        grouped[(row.get("task", ""), row.get("arm", ""), round(epoch), metric_path)][str(group)]["child"] = child
    result: dict[str, dict[str, dict[str, list[tuple[int, float | None, float, float]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (task, arm, epoch, metric_path), group_values in grouped.items():
        if set(group_values) != {"0", "1", "2", "3"} or any(
            set(values) != {"source", "child"} or values["source"] < 0 or values["child"] < 0
            for values in group_values.values()
        ):
            continue
        source_sum = math.fsum(values["source"] for values in group_values.values())
        child_sum = math.fsum(values["child"] for values in group_values.values())
        ratio = child_sum / source_sum if source_sum > 0 else None
        result[task][arm][metric_path].append((epoch, ratio, source_sum, child_sum))
    return {task: {arm: dict(metrics) for arm, metrics in arms.items()} for task, arms in result.items()}


def _positive_component_scope(inputs: dict[str, Any], task_id: str, metric_path: str) -> bool:
    parts = metric_path.split(".")
    if len(parts) < 5 or parts[:2] != ["coherence", "components"] or parts[3] not in {"self", "mutual"}:
        return False
    task = ((inputs.get("inventory") or {}).get("tasks") or {}).get(task_id, {})
    config = task.get("frozen_task_config", {}).get("coherence", {}).get("families", {}) if isinstance(task, dict) else {}
    family = config.get(parts[2], {}) if isinstance(config, dict) else {}
    components = family.get("components", {}) if isinstance(family, dict) else {}
    component = components.get(parts[3], {}) if isinstance(components, dict) else {}
    return (
        isinstance(component, dict)
        and component.get("enabled") is True
        and (_number(component.get("weight")) or 0.0) > 0.0
    )


def _x_task_present(inputs: dict[str, Any]) -> bool:
    history = inputs.get("history") or {}
    task = history.get("tasks", {}).get("X", {}) if isinstance(history, dict) else {}
    arms = task.get("arms", {}) if isinstance(task, dict) else {}
    return isinstance(arms, dict) and any(
        isinstance(value, dict) and bool(value.get("runs")) for value in arms.values()
    )


def _source_validation_coherence_sum(inputs: dict[str, Any], task_id: str) -> float | None:
    for arm in ("dense", "heuristic", "strategy"):
        for run in _arm_runs(inputs, task_id, arm):
            native_history = run.get("native_history")
            baseline = native_history.get("baseline") if isinstance(native_history, dict) else None
            baseline_value = _number(baseline.get("source_baseline_coherence_total")) if isinstance(baseline, dict) else None
            if baseline_value is not None and baseline_value > 0:
                return baseline_value
            for evaluation in _uniform_rows(run):
                if int(evaluation.get("epoch", -1)) != 0:
                    continue
                raw = evaluation.get("raw")
                item = raw.get("coherence_sum") if isinstance(raw, dict) else None
                if isinstance(item, dict):
                    numerator = _number(item.get("source_numerator"))
                    count = _number(item.get("source_count"))
                    if numerator is not None and count is not None and count > 0:
                        return numerator / count
    return None


def _write_maturity_figure(inputs: dict[str, Any], path: Path) -> None:
    plt, PdfPages, Rectangle = _matplotlib(inputs["root"])
    inventory_tasks = (inputs.get("inventory") or {}).get("tasks", {})
    x_present = _x_task_present(inputs)
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(14.2, 7.2 if x_present else 5.4), constrained_layout=True)
        if x_present:
            grid = fig.add_gridspec(1, 2, width_ratios=(1.12, 1.0))
            heat_ax = fig.add_subplot(grid[0, 0])
            x_grid = grid[0, 1].subgridspec(2, 1, hspace=0.28)
            x_family_ax = fig.add_subplot(x_grid[0, 0])
            x_field_ax = fig.add_subplot(x_grid[1, 0])
        else:
            grid = fig.add_gridspec(1, 1)
            heat_ax = fig.add_subplot(grid[0, 0])
            x_family_ax = x_field_ax = None
        fig.suptitle("Mature-window confirmation, retention, and optional transfer", fontsize=14, fontweight="bold")
        endpoints = [60, 70, 80, 90, 100]
        rows = []
        for task_id in ("G", "D"):
            task = inventory_tasks.get(task_id, {}) if isinstance(inventory_tasks, dict) else {}
            arms = task.get("arms", list(MANDATORY_FALLBACK.get(task_id, ()))) if isinstance(task, dict) else list(MANDATORY_FALLBACK.get(task_id, ()))
            for arm in arms:
                status = _arm_status(inputs, task_id, str(arm), int(task.get("horizon", 100)) if isinstance(task, dict) else 100)
                run = status.get("run")
                assessment = run.get("quality_assessment") if isinstance(run, dict) and isinstance(run.get("quality_assessment"), dict) else {}
                windows = {int(item["epoch"]): item for item in assessment.get("mature_windows", []) if isinstance(item, dict) and isinstance(item.get("epoch"), int)}
                confirmation = assessment.get("confirmation") or {}
                retention = assessment.get("retention") or {}
                rows.append((task_id, str(arm), status, windows, confirmation, retention))
        heat_ax.set_xlim(0, len(endpoints) + 1.95)
        heat_ax.set_ylim(-0.5, len(rows) - 0.5 if rows else 0.5)
        heat_ax.invert_yaxis()
        heat_ax.set_xticks([index + 0.5 for index in range(len(endpoints))], [f"E{epoch}" for epoch in endpoints])
        heat_ax.set_yticks(range(len(rows)), [f"{task}/{arm}" for task, arm, *_ in rows])
        heat_ax.set_title("40-epoch mature-window pooled checks", loc="left")
        heat_ax.set_xlabel("window end epoch; (e-40,e]")
        for y, (task_id, arm, status, windows, confirmation, retention) in enumerate(rows):
            for x_index, endpoint in enumerate(endpoints):
                window = windows.get(endpoint)
                if window is None:
                    color, label = "#e4e4e4", "pending"
                elif not window.get("complete", False):
                    color, label = "#b9c4cf", "censored"
                elif window.get("meets_target", False):
                    color, label = "#67a873", f"pass\n{window.get('raw_target_checks', 0)}/{window.get('raw_check_count', 0)}"
                else:
                    color, label = "#d47a70", f"fail\n{window.get('raw_target_checks', 0)}/{window.get('raw_check_count', 0)}"
                heat_ax.add_patch(Rectangle((x_index, y - 0.38), 0.98, 0.76, facecolor=color, edgecolor="white", linewidth=0.8))
                heat_ax.text(x_index + 0.49, y, label, ha="center", va="center", fontsize=5.7)
            confirm = confirmation.get("status", "pending")
            if confirmation.get("confirmation_epoch") is not None:
                confirm += f" E{confirmation['confirmation_epoch']}"
            retained = retention.get("status", "pending")
            if retention.get("through_observed_end_epoch") is not None:
                retained += f" to E{retention['through_observed_end_epoch']}"
            heat_ax.text(len(endpoints) + 0.06, y, f"{confirm}; {retained}", va="center", fontsize=6.2)
        heat_ax.grid(axis="x", alpha=0.1)
        heat_ax.spines["left"].set_visible(False)
        heat_ax.spines["bottom"].set_visible(False)

        if x_present and x_family_ax is not None and x_field_ax is not None:
            x_task = inventory_tasks.get("X", {}) if isinstance(inventory_tasks, dict) else {}
            x_arms = x_task.get("arms", ["dense", "strategy"]) if isinstance(x_task, dict) else ["dense", "strategy"]
            family_names = ["coherence_sum", "family_A", "family_B", "family_T", "endpoint_aggregate"]
            for arm in x_arms:
                runs = _arm_runs(inputs, "X", str(arm))
                if len(runs) != 1:
                    continue
                evaluations = _uniform_rows(runs[0])
                if not evaluations:
                    x_family_ax.text(0.5, 0.5, f"{arm}: scheduled native validation pending", ha="center", va="center", transform=x_family_ax.transAxes)
                    x_field_ax.text(0.5, 0.5, "endpoint field histories pending", ha="center", va="center", transform=x_field_ax.transAxes)
                    continue
                x = [float(item["epoch"]) for item in evaluations]
                linestyle = "-" if str(arm) == "strategy" else "--"
                for name in family_names:
                    y = [_number((item.get("ratios") or {}).get(name)) for item in evaluations]
                    if any(value is not None for value in y):
                        x_family_ax.plot(x, y, marker="o", markersize=2.5, linewidth=1.0, linestyle=linestyle, color=METRIC_COLORS[name], label=f"{arm}: {name}")
                field_names = x_task.get("frozen_task_config", {}).get("dataset", {}).get("field_names", []) if isinstance(x_task, dict) else []
                for index, field in enumerate(field_names if isinstance(field_names, list) else []):
                    y = [_number((item.get("ratios") or {}).get("endpoint_fields", {}).get(field)) for item in evaluations]
                    if any(value is not None for value in y):
                        x_field_ax.plot(x, y, marker="o", markersize=2.5, linewidth=1.0, linestyle=linestyle, color=FIELD_PALETTE[index % len(FIELD_PALETTE)], label=f"{arm}: {field}")
            for ax in (x_family_ax, x_field_ax):
                ax.set_xlim(0, 100)
                ax.set_xticks(range(0, 101, 20))
                ax.axhline(1.05, color="#999999", linestyle="--", linewidth=0.75)
                ax.grid(alpha=0.16)
                ax.set_ylabel("child / paired SOURCE")
                ax.legend(frameon=False, fontsize=5.5, ncol=2, loc="best")
            x_family_ax.axhline(0.90, color="#222222", linestyle=":", linewidth=0.8)
            x_family_ax.set_title("X: A/B/T and additive ratios (validation)", loc="left")
            x_field_ax.set_title("X: endpoint-field model-unit MSE ratios", loc="left")
            x_family_ax.set_xlabel("training epoch")
            x_field_ax.set_xlabel("training epoch")
            present_arms = {arm for arm in x_arms if _arm_runs(inputs, "X", str(arm))}
            if present_arms != {"dense", "strategy"}:
                x_family_ax.text(0.02, 0.02, "Optional X pair incomplete; transfer pending", transform=x_family_ax.transAxes, fontsize=6.5)
        else:
            heat_ax.text(0.01, -0.20, "Optional X transfer is omitted because the analyzer has no X run evidence.", transform=heat_ax.transAxes, fontsize=7)
        fig.text(0.01, 0.005, "A window is green only when all scheduled checks are complete and every pooled additive, family, and endpoint risk is within its frozen limit; gray cells are pending/right-censored.", fontsize=7)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _latest_quality_status(run: dict[str, Any] | None) -> tuple[str, str]:
    if not run:
        return "pending", "pending"
    assessment = run.get("quality_assessment") if isinstance(run.get("quality_assessment"), dict) else {}
    confirmation = assessment.get("confirmation") or {}
    retention = assessment.get("retention") or {}
    confirmation_text = str(confirmation.get("status", "pending"))
    if confirmation.get("confirmation_epoch") is not None:
        confirmation_text += f" E{confirmation['confirmation_epoch']}"
    retention_text = str(retention.get("status", "pending"))
    if retention.get("through_observed_end_epoch") is not None:
        retention_text += f" to E{retention['through_observed_end_epoch']}"
    return confirmation_text, retention_text


def _ratio_occupancy_text(value: dict[str, Any] | None) -> str:
    if not isinstance(value, dict) or _number(value.get("check_count")) in {None, 0.0}:
        return "not observed"
    count = _number(value.get("check_count"))
    hits = _number(value.get("target_checks"))
    fraction = _number(value.get("fraction"))
    return f"{hits:g}/{count:g} ({fraction:.1%})" if hits is not None and count is not None and fraction is not None else "not recorded"


def _worst_excursion_text(excursions: dict[str, Any] | None) -> str:
    if not isinstance(excursions, dict):
        return "not assessed"
    if not excursions:
        return "none observed"
    return "; ".join(
        f"{name}: max {_fmt(item.get('max_ratio'), 4)} vs {_fmt(item.get('limit'), 3)} "
        f"(+{_fmt(item.get('max_excess'), 4)}; {item.get('raw_violation_count', 0)}/{item.get('raw_check_count', 0)} checks)"
        for name, item in sorted(excursions.items())
        if isinstance(item, dict)
    ) or "not assessed"


def _accounted_attainment_text(record: dict[str, Any]) -> str:
    status = record.get("confirmation_status")
    if status == "confirmed":
        seconds = _number(record.get("confirmation_accounted_cold_seconds"))
        suffix = f"; {_hours(seconds)} accounted cold" if seconds is not None else ""
        return f"confirmed E{record.get('confirmation_epoch')}{suffix}"
    if status == "right_censored":
        seconds = _number(record.get("censor_accounted_cold_seconds"))
        suffix = f"; {_hours(seconds)} accounted cold" if seconds is not None else ""
        return f"right-censored at E{record.get('censor_epoch')}{suffix}"
    epoch = record.get("latest_native_epoch")
    seconds = _number(record.get("latest_accounted_cold_process_seconds"))
    suffix = f"; {_hours(seconds)} accounted cold" if seconds is not None else ""
    if epoch is not None:
        return f"not assessed through E{epoch}{suffix}"
    return "pending run/assessment"


def _attainment_table_lines(inputs: dict[str, Any]) -> list[str]:
    records = inputs.get("decision_cost_records", [])
    included = [
        row for row in records
        if row.get("role") == "mandatory"
        or (row.get("task_id") == "X" and row.get("run_id"))
    ]
    lines = [
        "| Task / arm | Execution evidence | Accepted end | Confirmed time or censor | Retention | Raw whole-target occupancy | Worst raw per-check excursions |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in included:
        retention = str(row.get("retention_status", "pending"))
        if retention == "not_established_no_confirmed_attainment":
            retention = "not established (no confirmed attainment)"
        elif retention == "pending_run_or_assessment":
            retention = "pending run/assessment"
        accepted = row.get("accepted_end_epoch")
        lines.append(
            f"| {row['task_id']} / {row['arm']} | {row.get('execution_correctness_status', 'not verified')}; recovery {row.get('recovery_probe_status', 'pending')} | {accepted if accepted is not None else 'not recorded'} | {_accounted_attainment_text(row)} | {retention} | {_ratio_occupancy_text(row.get('raw_whole_target_occupancy'))} | {_worst_excursion_text(row.get('worst_raw_excursions'))} |"
        )
    if not included:
        lines.append("| G/D arms | pending | not recorded | pending run/assessment | pending | not observed | not assessed |")
    lines.extend(
        [
            "",
            "| Task / arm | E60 | E70 | E80 | E90 | E100 |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in included:
        statuses = row.get("mature_window_statuses", {})
        occupancy = row.get("mature_window_occupancy", {})
        cells = []
        for endpoint in (60, 70, 80, 90, 100):
            status = statuses.get(str(endpoint), "pending") if isinstance(statuses, dict) else "pending"
            point = occupancy.get(str(endpoint), {}) if isinstance(occupancy, dict) else {}
            if status in {"pass", "fail", "censored"}:
                target = _ratio_occupancy_text(point)
                cell = f"{status} ({target})" if target != "not observed" else status
            else:
                cell = str(status)
            cells.append(cell)
        lines.append(f"| {row['task_id']} / {row['arm']} | " + " | ".join(cells) + " |")
    return lines


def _retention_excursions_text(value: Any) -> str:
    if not isinstance(value, list):
        return "not assessed"
    if not value:
        return "none observed"
    parts = []
    for item in value:
        if not isinstance(item, dict):
            continue
        start = item.get("start_window_epoch")
        end = item.get("end_window_epoch")
        duration = item.get("duration_epochs")
        maxima = item.get("max_excess_by_metric")
        if isinstance(maxima, dict) and maxima:
            maximum_text = ", ".join(f"{name} +{_fmt(excess, 4)}" for name, excess in sorted(maxima.items()))
        else:
            maximum_text = "no metric excess recorded"
        parts.append(f"E{start}–E{end}, {duration} epochs; {maximum_text}")
    return "; ".join(parts) if parts else "not assessed"


def _censored_spans_text(value: Any) -> str:
    if not isinstance(value, list):
        return "not assessed"
    if not value:
        return "none observed"
    parts = []
    for item in value:
        if not isinstance(item, dict):
            continue
        start = item.get("start_window_epoch")
        end = item.get("end_window_epoch")
        missing = item.get("missing_check_epochs")
        missing_text = ",".join(str(epoch) for epoch in missing) if isinstance(missing, list) and missing else "not recorded"
        parts.append(f"E{start}–E{end}; missing checks {missing_text}")
    return "; ".join(parts) if parts else "not assessed"


def _retention_excursion_table_lines(inputs: dict[str, Any]) -> list[str]:
    records = inputs.get("decision_cost_records", [])
    included = [
        row for row in records
        if row.get("role") == "mandatory"
        or (row.get("task_id") == "X" and row.get("run_id"))
    ]
    lines = [
        "| Task / arm | Assessor-native retention excursions (start–end; duration; maximum excess by metric) | Censored spans |",
        "|---|---|---|",
    ]
    for row in included:
        lines.append(
            f"| {row['task_id']} / {row['arm']} | {_retention_excursions_text(row.get('retention_excursions'))} | {_censored_spans_text(row.get('retention_censored_spans'))} |"
        )
    if not included:
        lines.append("| G/D arms | not assessed | not assessed |")
    return lines


def _portable_audit_table_lines(inputs: dict[str, Any]) -> list[str]:
    inventory_tasks = (inputs.get("inventory") or {}).get("tasks", {})
    history_tasks = (inputs.get("history") or {}).get("tasks", {})
    root = inputs.get("root")
    lines = [
            "| Task / arm | Audit status | Checkpoint / epoch | Panel | Full validation scope | Additive ratio | A ratio | B ratio | T ratio | Endpoint aggregate MSE ratio | Endpoint-field model-unit MSE ratios | Artifact |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    if not isinstance(inventory_tasks, dict):
        inventory_tasks = {}
    for task_id, task in inventory_tasks.items():
        if not isinstance(task, dict) or not isinstance(task.get("arms"), list):
            continue
        fields = task.get("frozen_task_config", {}).get("dataset", {}).get("field_names", [])
        fields = [str(field) for field in fields] if isinstance(fields, list) else []
        for arm_value in task["arms"]:
            arm = str(arm_value)
            task_history = history_tasks.get(str(task_id)) if isinstance(history_tasks, dict) else None
            arm_histories = task_history.get("arms") if isinstance(task_history, dict) else None
            arm_history = arm_histories.get(arm, {}) if isinstance(arm_histories, dict) else {}
            runs = arm_history.get("runs", []) if isinstance(arm_history, dict) else []
            audits = []
            for run in runs if isinstance(runs, list) else []:
                if not isinstance(run, dict):
                    continue
                for collection, label in (
                    ("portable_evaluations", "portable"),
                    ("supplementary_evaluations", "supplementary"),
                ):
                    evaluations = run.get(collection, [])
                    if not isinstance(evaluations, list):
                        continue
                    audits.extend((run, item, label) for item in evaluations if isinstance(item, dict))
            if not audits:
                lines.append(
                    f"| {task_id} / {arm} | pending | not recorded | pending | pending full A+B+T validation audit (128 snapshots; 4 groups of 32; 4096 points; six topology filtrations) | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded |"
                )
                continue
            for run, evaluation, collection_label in audits:
                panel_kind = str(evaluation.get("panel_kind", "not recorded"))
                panel_text = f"{panel_kind} ({collection_label})"
                epoch = evaluation.get("epoch")
                checkpoint = _safe_text(evaluation.get("checkpoint"))
                epoch_text = f"{checkpoint} / E{epoch}" if epoch is not None else f"{checkpoint} / epoch not recorded"
                status = _safe_text(evaluation.get("status"))
                ratios = evaluation.get("ratios") if isinstance(evaluation.get("ratios"), dict) else {}
                endpoint_fields = ratios.get("endpoint_fields", {})
                endpoint_fields = endpoint_fields if isinstance(endpoint_fields, dict) else {}
                field_names = fields or sorted(str(name) for name in endpoint_fields)
                field_text = ", ".join(
                    f"{_safe_text(name)} {_ratio_text(endpoint_fields.get(name), 1.05)}"
                    for name in field_names
                ) or "not recorded"
                artifact = "not recorded"
                artifact_path = evaluation.get("path")
                if root is not None and isinstance(artifact_path, str) and artifact_path:
                    try:
                        relative_path = Path(artifact_path).resolve(strict=False).relative_to(
                            Path(root).resolve(strict=False)
                        )
                    except (OSError, ValueError):
                        artifact = "outside owned root (not linked)"
                    else:
                        href = quote(f"../{relative_path.as_posix()}", safe="/._-~")
                        artifact = f"[artifact]({href})"
                scope = "validation; full A+B+T; 128 snapshots; 4 groups of 32; 4096 points; six topology filtrations"
                lines.append(
                    f"| {task_id} / {arm} | {status} | {epoch_text} | {panel_text} | {scope} | "
                    f"{_ratio_text(ratios.get('coherence_sum'), 0.90)} | {_ratio_text(ratios.get('family_A'), 1.05)} | "
                    f"{_ratio_text(ratios.get('family_B'), 1.05)} | {_ratio_text(ratios.get('family_T'), 1.05)} | "
                    f"{_ratio_text(ratios.get('endpoint_aggregate'), 1.05)} | {field_text} | {artifact} |"
                )
    if len(lines) == 2:
        lines.append("| all task/arm audits | pending | not recorded | pending | pending full A+B+T audit | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded |")
    return lines


def _latest_ratio_table_lines(arm_records: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Task / arm | Run status | Accepted end | Latest native epoch | Accounted cold | Wall elapsed | Additive | A | B | T | Endpoint aggregate MSE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in arm_records:
        run = item.get("run")
        summary = _arm_metric_summary(run)
        ratios = summary["ratios"] if isinstance(summary["ratios"], dict) else {}
        accepted = run.get("accepted_end_epoch") if run else None
        lines.append(
            f"| {item['task_id']} / {item['arm']} | {item['status']} | {accepted if accepted is not None else 'not recorded'} | {summary['last_epoch'] if summary['last_epoch'] is not None else 'not recorded'} | {_hours(summary['cold_seconds'])} | {_hours(summary['wall_seconds'])} | {_ratio_text(ratios.get('coherence_sum'), 0.90)} | {_ratio_text(ratios.get('family_A'), 1.05)} | {_ratio_text(ratios.get('family_B'), 1.05)} | {_ratio_text(ratios.get('family_T'), 1.05)} | {_ratio_text(ratios.get('endpoint_aggregate'), 1.05)} |"
        )
    if not arm_records:
        lines.append("| G/D arms | pending | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded |")
    lines.extend(
        [
            "",
            "| Task / arm | Latest endpoint-field model-unit MSE ratios |",
            "|---|---|",
        ]
    )
    for item in arm_records:
        run = item.get("run")
        summary = _arm_metric_summary(run)
        ratios = summary["ratios"] if isinstance(summary["ratios"], dict) else {}
        endpoint_fields = ratios.get("endpoint_fields", {}) if isinstance(ratios.get("endpoint_fields"), dict) else {}
        field_text = ", ".join(f"{name} {_ratio_text(value, 1.05)}" for name, value in endpoint_fields.items()) or "not recorded"
        lines.append(f"| {item['task_id']} / {item['arm']} | {field_text} |")
    return lines


def _study_dimension_lines(inputs: dict[str, Any], arm_records: list[dict[str, Any]]) -> list[str]:
    records = inputs.get("decision_cost_records", [])
    mandatory = [row for row in records if row.get("role") == "mandatory"]
    integrity_ok = sum(row.get("native_history_integrity") == "ok" for row in mandatory if row.get("run_id"))
    recovery_status, recovery_detail = _recovery_probe_status(inputs)
    completed_pairs = all(
        next((row.get("arm_evidence_status") == "complete" for row in mandatory if row.get("task_id") == task and row.get("arm") == arm), False)
        for task in ("G", "D") for arm in ("dense", "strategy")
    )
    if completed_pairs:
        latency_status = "paired runs complete; quality-matched cold time still depends on confirmed mature attainment"
    else:
        latency_status = "pending paired 100-epoch dense/strategy histories; only profile forecasts and partial charges are available"
    assessed = [row for row in mandatory if row.get("confirmation_status") in {"confirmed", "right_censored"}]
    if assessed:
        attainment_status = "; ".join(f"{row['task_id']}/{row['arm']} {row.get('confirmation_status')}" for row in assessed)
    else:
        attainment_status = "pending mature-window assessment"
    confirmed = [row for row in mandatory if row.get("confirmation_status") == "confirmed"]
    retention_status = (
        "; ".join(f"{row['task_id']}/{row['arm']} {row.get('retention_status')}" for row in confirmed)
        if confirmed
        else "not established until an arm confirms mature attainment"
    )
    d_rows = {row.get("arm"): row for row in mandatory if row.get("task_id") == "D"}
    d_science_complete = all(
        d_rows.get(arm, {}).get("arm_evidence_status") == "complete"
        for arm in ("dense", "heuristic", "strategy")
    )
    d_scope = (
        "all three mandatory D 100-epoch model histories complete; cross-model learning comparison is available, with quality efficacy assessed separately"
        if d_science_complete
        else "pending all three mandatory D 100-epoch model histories"
    )
    x_rows = {row.get("arm"): row for row in records if row.get("task_id") == "X"}
    x_complete = all(
        x_rows.get(arm, {}).get("arm_evidence_status") == "complete"
        for arm in ("dense", "strategy")
    )
    if x_complete:
        x_scope = "optional X dense/strategy physical-transfer pair complete; paired quality review remains separate"
    elif x_rows:
        x_scope = "optional X physical-transfer pair is incomplete; transfer is not established"
    else:
        x_scope = "optional X physical-transfer pair not run; gated on all mandatory arms and remaining budget"
    transfer_status = f"D: {d_scope}; X: {x_scope}"
    return [
        "| Evidence axis | Current status |",
        "|---|---|",
        f"| Correct execution | {integrity_ok}/{sum(bool(row.get('run_id')) for row in mandatory)} launched mandatory histories report native-history integrity `ok`; recovery probe is **{recovery_status.replace('_', ' ')}**. {recovery_detail} |",
        f"| Parameter-estimator expectation | {_parameter_expectation_summary(inputs)} |",
        f"| Latency benefit | {latency_status} |",
        f"| Target attainment | {attainment_status} |",
        f"| Target retention | {retention_status} |",
        f"| Transfer / scope | {transfer_status} |",
    ]


def _arm_cost_table_lines(inputs: dict[str, Any]) -> list[str]:
    records = inputs.get("decision_cost_records", [])
    lines = [
        "| Task / arm | Terminal training (s) | Profile measured (s) | Profile estimate (s) | Terminal evaluate (s) | Observed cold total (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        training_value = _fmt(record.get("terminal_training_process_seconds"), 1)
        if record.get("training_process_charge_status") == "incomplete_or_unmeasured" and training_value != "not recorded":
            training_value += " (partial)"
        evaluation_value = _fmt(record.get("charged_final_evaluation_process_seconds"), 1)
        evaluation_status = record.get("final_evaluation_charge_status")
        if evaluation_status == "no_recorded_evaluate_attempts":
            evaluation_value += " (none recorded)"
        elif evaluation_status == "recorded_evaluations_incomplete_or_unmeasured" and evaluation_value != "not recorded":
            evaluation_value += " (partial)"
        lines.append(
            f"| {record.get('task_id', 'unknown')} / {record.get('arm', 'unknown')} | {training_value} | {_fmt(record.get('allocated_profile_acquisition_measured_seconds'), 1)} | {_fmt(record.get('allocated_profile_acquisition_estimated_seconds'), 1)} | {evaluation_value} | {_fmt(record.get('observed_cold_task_total_seconds'), 1)} |"
        )
    if not records:
        lines.append("| all | not recorded | not recorded | not recorded | 0.0 (none recorded) | not recorded |")
    return lines


def _make_report(inputs: dict[str, Any], attempts: list[dict[str, Any]], completion_records: list[dict[str, Any]], figure_names: tuple[str, str, str]) -> str:
    root = inputs["root"]
    all_complete, arm_records = _all_mandatory_complete(inputs)
    interpretation = _final_interpretation_state(inputs, all_complete)
    history = inputs.get("history")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    source_binding = inputs.get("source_binding") or {}
    contract_check = inputs.get("contract_check") or {}
    status_label = "All mandatory 100-epoch horizons complete; quality assessments available." if all_complete else "INCOMPLETE: one or more mandatory G/D arms remain pending, censored, ambiguous, or below 100 accepted epochs; no campaign completion or verdict is available."
    lines = [
        "# Cost-Aware Lagrangian Execution Strategy: Evidence Report",
        "",
        f"> **Study status:** {status_label}",
        "",
        f"Prepared {now} from the owned campaign root `{root}`. The report reads campaign artifacts and analyzer output only; it does not open datasets, checkpoints, or TEST. The science source binding records commit `{source_binding.get('git_head', 'not recorded')}`.",
        *_final_interpretation_lines(interpretation),
        "",
        "## Decision and frozen scope",
        "",
        "The execution planner compares applicable measured whole-plan costs after profile acquisition, setup, scheduled refresh, and full-gradient audit overhead, and it admits only TRAIN-calibrated estimator variance under the predeclared noise ceiling and nested importance limit. A no-action full execution is a valid choice when accelerated candidates fail support, noise, amortization, or break-even gates.",
        "",
        "| Task | Selected plan | Projected cost | Full-reference forecast | Projected saving | Added variance / intrinsic TRAIN estimate | Maximum joint weight |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    profile_rows = _profile_cost_rows(inputs, attempts)
    for row in profile_rows:
        decision = row["decision"]
        profile = row["profile"]
        plan_id = decision.get("plan_id", "pending")
        intrinsic = _number(profile.get("intrinsic_noise_variance"))
        extra = _number(decision.get("extra_variance"))
        variance_text = f"{_fmt(extra)} / {_fmt(intrinsic)}" if extra is not None and intrinsic is not None else "not recorded"
        lines.append(
            f"| {row['task_id']} | {plan_id}{' (no action)' if decision.get('no_action') else ''} | {_hours(decision.get('predicted_total_s'))} | {_hours(decision.get('predicted_full_total_s'))} | {_pct(decision.get('cost_savings_ratio'))} | {variance_text} | {_fmt(decision.get('max_joint_importance_multiplier'), 1)} |"
        )
    if not profile_rows:
        lines.append("| G/D/X | pending | pending | pending | pending | pending | pending |")
    selection_reasons = [
        f"{row['task_id']}: {_safe_text(row['decision'].get('selected_reason'))}"
        for row in profile_rows
        if row["decision"].get("selected_reason")
    ]
    if selection_reasons:
        lines.extend(["", "Selection rationale: " + " ".join(selection_reasons) + "."])
    if (inputs.get("revisions") or {}).get("warm_reference") and (inputs.get("revisions") or {}).get("complete_profile_cost"):
        lines.extend(
            [
                "",
                "Profile lineage: the warm-reference v1 profiles, plans, and failed/superseded attempts remain preserved under `analysis/profiles_warm_reference_v1`; the current forecasts use the complete-retained-profile-lifetime v2 revision. Ledger totals retain every admitted profile lifetime, including failures and earlier warm-reference attempts.",
            ]
        )
    lines.extend(
        [
            "",
            "Forecasts are model-based projections from a finite TRAIN profile and are not realized campaign savings or statistical upper bounds. Prediction error is computed only after the matching 100-epoch strategy arm is complete; partial charges remain censored. The full-reference forecast excludes profile acquisition while the selected-plan forecast includes it. For full/no action, the recorded zero saving means no acceleration was adopted; its inclusive forecast still includes profiling and monitoring and can exceed unprofiled dense execution.",
            "",
            "### Separate evidence axes",
            "",
            *_study_dimension_lines(inputs, arm_records),
            "",
            "### Mature attainment and retention",
            "",
            *_attainment_table_lines(inputs),
            "",
            "Window cells show assessor-native pass/fail/censored status and raw whole-target check occupancy; the linked `analysis/mature_window_metrics.csv` retains each metric’s pooled numerator/counts, pooled ratio, per-check mean/median, slope, range, raw violations, and limit. Raw occupancy and per-metric worst excursions above are computed from each observed native check separately and remain visible even when a pooled window passes. If no qualifying consecutive window pair is observed by E100, confirmation is right-censored at the horizon and retention is not established. If confirmation first occurs at E100, subsequent retention is right-censored because no later window exists.",
            "",
            "The table below copies assessor-native post-confirmation excursion durations and maximum metric excesses, and lists censored spans without recomputation. The compact JSON/CSV also preserve post-confirmation window/check counts and fractions, final-window pooled ratios, and complete native excursion objects; raw whole-history maxima and counts remain in the attainment table.",
            "",
            *_retention_excursion_table_lines(inputs),
            "",
            "### Latest native quality checks",
            "",
            *_latest_ratio_table_lines(arm_records),
            "",
            "A ! marker denotes a full-precision raw ratio above its limit before display rounding: 0.90 for the additive sum and 1.05 for families and endpoint risks. The compact `analysis/decision_cost_quality.json` and `.csv` preserve one row per task/arm with status, cost attribution, ratios, occupancy, excursions, and window states.",
            "",
            "### Portable full A+B+T endpoint audits",
            "",
            "Every analyzer-validated portable uniform audit and disjoint supplementary audit is listed; arms without an artifact remain explicitly pending. These endpoint audits do not enter native scheduled histories, mature-window assessment, or time-to-quality. Checkpoint labels follow the existing anchor-bound `topology_with_fidelity` selector, separate from the additive primary optimization and assessment objective; displayed validation ratios do not select or revise a checkpoint.",
            "For D/X, the portable evaluator's legacy two-Euler inference metadata records unused common configuration fields: the deterministic adapter executes its native direct reconstruction and unchanged endpoint observation consistency, with no flow solver or generative loss. G executes the actual two-Euler flow path. The [adapter-scope clarification](../analysis/deterministic_inference_metadata_scope.json) binds this distinction to the frozen source.",
            "",
            *_portable_audit_table_lines(inputs),
            "",
            "### Strategy runtime policy and audit coverage",
            "",
            *_runtime_policy_table_lines(inputs),
            "",
            "The monitor percentage is computed from that record as `monitor_overhead_seconds / accepted_training_seconds`; the other plan counts, included-update counts, importance maximum, fallback text, recovery intervals, and audit epochs are copied from the record. Missing strategy records remain pending.",
            "",
            "The strategy runtime summary is checkpoint-bound actual run metadata. Sampled optimized Lagrangian training logs describe updates executed under the sampled plan; fixed full-TRAIN audit logs describe the separate full-objective training audits recorded in the summary; scheduled validation histories are a third evidence stream. Their epochs, coverage counts, or outcomes are not substituted for one another.",
        ]
    )
    lines.extend(
        [
            "",
            "### TRAIN calibration bindings",
            "",
            "| Task | SOURCE TRAIN C0 | Contract C0 | Exact match | Calibration identity | Source-calibration artifact | SHA-256 prefix |",
            "|---|---:|---:|---|---|---|---|",
        ]
    )
    for row in inputs.get("calibration_rows", _source_calibration_rows(inputs)):
        lines.append(
            f"| {row['task_id']} | {_fmt(row['train_c0'], 8)} | {_fmt(row['contract_c0'], 8)} | {'yes' if row['c0_exact_match'] else 'not verified'} | `{_safe_text(row['source_calibration_identity'])}` | [artifact](../{row['source_calibration_file']}) | {(row['source_calibration_sha256'] or '')[:16] or 'missing'} |"
        )
    lines.extend(
        [
            "",
            "These C0 values are bound from SOURCE TRAIN calibration for training-side eligibility; validation ratios use the paired validation SOURCE baseline from the frozen panel and are a separate population.",
            "",
            _fixed_query_count_text(inputs),
            "",
            _preserved_negative_evidence_text(inputs),
            "",
            _profile_proxy_erratum_text(inputs),
        ]
    )
    lines.extend(
        [
            "",
            "The frozen mandatory set is G and D, each with dense, heuristic, and strategy arms at a 100-epoch horizon. X’s cost forecast is plan-only: the optional dense+strategy transfer pair remains gated until all six mandatory arms complete and the remaining budget permits it. A partial or single-arm X history does not establish transfer. A missing, failed, interrupted, under-horizon, multiple-lineage, or time-ambiguous run is not promoted to a completed result.",
            "",
            "## Cost and quality evidence",
            "",
            f"![Figure 1. Plan forecasts, noise screening, ledger-charged execution, and profile cache scope.]({figure_names[0]})",
            (
                "Figure 1. Measured whole-update alternatives and TRAIN-noise screening are shown beside the planner's 100-epoch forecast. "
                "Panel C bars are terminal training command charges plus allocated profile acquisition, including failed and resumed process lifetimes; final evaluation commands are excluded from these bars and shown separately in the per-arm cost table. The strategy bar includes all task profile attempts once, while dense and fixed-heuristic bars have zero profile allocation. "
                "Planner markers cover measured profile cost, whole-optimizer-update extrapolation, and scheduled refresh/audit monitors, but omit native validation, checkpoint and I/O, and fresh-training-calibration startup that remain in actual cold charges. "
                "The forecast-to-command gap therefore combines unmodeled common cost and prediction error; it is not pure prediction error. Full-reference and selected-plan markers appear only where measured; no fixed-heuristic forecast is imputed. "
                f"{_figure1_cache_caption(inputs)}"
            ),
            "",
            f"![Figure 2. Complete native scheduled quality histories for G and D, including endpoint aggregate and fields, A/B/T, and paired self/mutual diagnostics.]({figure_names[1]})",
            "Figure 2. Each task’s first page compares the same additive coherence ratio against epoch and numeric accounted cold check-completion hours; these clocks stop at each native check and include profile allocation split into measured and recorded estimated amounts in the companion table. Terminal command charges include later closeout and are reported separately in the cost table. Lower panels show A/B/T and endpoint aggregate plus all five endpoint-field model-unit MSE ratios against epoch. Additional pages show positively weighted direct self/mutual component leaves separately: their values are pooled only on the same canonical path as sum(child)/sum(SOURCE) across four fixed 32-snapshot groups, with B independently computed in each; this is a source-weighted mean, not a 128-snapshot covariance claim. Zero-weight diagnostics remain in the raw component CSV, no distinct metric paths or aliases are combined, and self/mutual paths are not summed. Every history point is a primary native scheduled validation check paired to validation SOURCE; no epoch-0 completion time is imputed. Calendar wall completion times are separate in the machine-readable table, and duplicate rows and disagreements remain in analyzer provenance. Final portable evaluations are endpoint audits only and cannot replace scheduled histories or mature-window completion times.",
            "",
            f"![Figure 3. Mature-window confirmation and retention with optional X transfer when run evidence exists.]({figure_names[2]})",
            "Figure 3. Mature windows use the frozen 40-epoch interval `(e-40,e]` and are marked complete only when every scheduled check is present; retention and confirmation are copied from the assessor-native output. X histories are exploratory until both predeclared arms are present; the cost plan alone is not transfer evidence.",
            "",
            "### Resource accounting",
            "",
            "| Task | Action | Attempts | Completed/failed state counts | Charged process time |",
            "|---|---|---:|---|---:|",
        ]
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        grouped[(attempt.get("task_id") or "unknown", attempt.get("action") or "unknown")].append(attempt)
    for (task_id, action), values in sorted(grouped.items()):
        charged = sum(item["charged_process_seconds"] for item in values if item["charged_process_seconds"] is not None)
        has_charge = any(item["charged_process_seconds"] is not None for item in values)
        states: dict[str, int] = defaultdict(int)
        for item in values:
            states[str(item["status"])] += 1
        state_text = ", ".join(f"{name}: {count}" for name, count in sorted(states.items()))
        lines.append(f"| {task_id} | {action} | {len(values)} | {state_text} | {_hours(charged) if has_charge else 'not charged yet'} |")
    if not attempts:
        lines.append("| all | pending | 0 | no ledger rows | pending |")
    total_charged = sum(item["charged_process_seconds"] for item in attempts if item["charged_process_seconds"] is not None)
    active = [item for item in attempts if item["status"] == "running"]
    lines.extend(
        [
            "",
            f"The ledger reports {_hours(total_charged)} in terminal process charges across all recorded attempts. {len(active)} attempts are still marked running and their reservations are not counted as realized charges. Profile attempts, including failed and retained warm-reference attempts, remain charged once under their task. Validation evaluation commands are reported as separate ledger actions and are not silently folded into the strategy forecast.",
            "",
            "### Per-arm observed cold cost",
            "",
            *_arm_cost_table_lines(inputs),
            "",
            "The evaluation column sums terminal `evaluate` ledger charges for all recorded commands. A zero with no recorded evaluation means only that no evaluation command is currently in the ledger; final-audit completeness is separate and is never inferred from absence. An observed total is emitted only when every training and recorded evaluation charge is terminal and measured and the allocated profile total is available; it includes any explicit profile-time estimate. Native TTQ completion times remain unchanged and do not receive future evaluation charges retroactively.",
            "",
            "## Methods and limits",
            "",
            "For a finite-support unbiased estimator, let `I_i` denote inclusion of scalar contribution `a_i`, with marginal probability `p_i` and joint inclusion probability `π_ij`; the Horvitz–Thompson scalar is `L̂ = Σ_i I_i a_i / p_i`, and its finite-support variance is `Σ_i Σ_j (π_ij − p_i p_j) a_i a_j / (p_i p_j)`. The implemented audits enumerate accepted joint support or use the explicitly declared independent family streams; the report does not substitute marginal-only or assumed-independent covariance for supported correlated draws. Nested family and within-family probabilities multiply: for topology with `p_family=0.5` and conditional polarity inclusion `0.5`, each unit has inclusion `0.25` and inverse-probability multiplier `4`, the predeclared ceiling. Original additive coefficients remain in `a_i`.",
            "",
            "Targeted unary reorder and subset-oracle checks verify the fixed descriptor-index/`self_ids` mapping in the opt-in generalized path. Measured CO and T orders already coincide, so these checks make no historical measured-error claim. Legacy identity-ordered v1 estimator state remains accepted; version 2 binds real descriptor IDs and rejects incompatible older reordered state. Existing P3/S2 scientific weights are retained, and correlated/nested variance uses joint inclusion support. The generalized path requires the user's explicit descriptor-mapping checkpoint and makes no numerical-data or configuration edits.",
            "",
            "For parameter gradients `g_i = ∇θ a_i`, the allocation statistic is `V_exec = E||ĝ − g||² = Σ_ij (π_ij/(p_i p_j) − 1) g_iᵀg_j`. For one whole family with gradient `G`, outer inclusion probability `p`, and an unbiased conditional inner estimator of variance `V_inner`, the exact nested variance is `(1/p − 1)||G||² + V_inner/p`; the inner and outer variances cannot simply be added. Enumerating the polarity support gives same-polarity joint inclusion 1/2 and cross-polarity joint inclusion zero; nesting an outer probability 1/2 changes these to 1/4 and zero. Tests also enumerate correlated outer-family support and unequal scientific weights.",
            "",
            "The ideal independent-work cost–variance allocation has interior form `p_i ∝ sqrt(v_i/c_i)` with clipping to the allowed inclusion range. This motivates the rule; the implemented planner chooses among six measured whole plans because shared reconstruction, caches, batching, backward and dense fidelity make costs nonadditive. It does not introduce another optimizer. A smooth fixed-dual SGD descent inequality motivates controlling added noise, but is not a convergence proof for nonsmooth persistence, clipping, projected duals or AdamW.",
            "",
            "The sampled scalar and gradient are importance-weighted before global gradient clipping and AdamW. Unbiasedness at that pre-clipping estimator does not establish trajectory equivalence: clipping, adaptive optimizer state, dual updates, checkpoint selection, and later validation are nonlinear and path-dependent. The finite four-batch parameter-gradient variance is coherence-only with SOURCE duals zero; the two-fixed-TRAIN-batch refresh audits are also coherence-gradient estimates, not the full projected dual/AdamW stochastic variance or formal upper bounds. The maximum observed batch duration is an engineering envelope, not a confidence interval.",
            "",
            "The cost profile uses four fixed TRAIN batches and matched pre-oracle reference-cache state; the full candidate also has a repeated warm measurement. Instrumented cache counters cover oracle/input-gradient and support-enumeration calls, while whole-update timing uses the production cache without accounting scans. Compiled KeOps cache construction and all admitted profile attempt lifetimes are charged; dense science commands begin after profiling, so the report labels this baseline cache condition rather than claiming a virgin-cache dense comparison. Profiler self-time and inclusive scope durations are not summed because scopes overlap.",
            "",
            _bottleneck_scope_text(inputs),
            "",
            "C0 and the per-field fidelity denominators are bound from SOURCE TRAIN calibration and are used for training-side eligibility. Validation quality ratios use the paired validation SOURCE baseline from the frozen four-group panel; these are separate populations and are not interchanged.",
            "",
            "Quality monitoring uses the fixed 128 uniformly spread validation snapshots in four ordered groups of 32 and the full six-filtration A/B/T path. The disjoint panel, if present, is supplementary only and never enters the primary quality assessment. No TEST path is authorized or opened by this reporting flow. Dataset identity uses the retained fingerprint and metadata/stat attestation; this report does not claim a new full-file byte rehash or prove absence of every operating-system-level read beyond the analyzer's declared access scope.",
            "",
            "Resume state binds the accepted cursor and estimator-specific counter RNG, but Python and NumPy global RNG streams are not a byte-for-byte restart contract. Process time is the sum of charged command lifetimes; wall gaps between commands are reported separately by analyzer-native fields when available and are not counted as GPU-active time. One seed per arm and shared-host timings do not establish significance, universal convergence, or causal speedup.",
            "",
            "### Artifact scope and rerun commands",
            "",
            "This report is generated from the owned profile, raw-profile, plan, attempt-ledger, predeclaration, source-binding, task-inventory, contract-check, and analyzer JSON artifacts under the campaign root; companion analyzer CSVs and preserved profile-revision manifests are fingerprinted for drill-down. The analyzer and report generator are read-only with respect to datasets and checkpoints; they write only analysis/report outputs.",
            "",
            "```bash",
            f"ROOT='{root}'",
            "CUDA_VISIBLE_DEVICES='' python scripts/posttraining/analyze_cost_aware_campaign.py --root \"$ROOT\"",
            "CUDA_VISIBLE_DEVICES='' python scripts/posttraining/build_cost_aware_report.py --root \"$ROOT\"",
            "python scripts/posttraining/cost_aware_campaign.py --root \"$ROOT\" status",
            "```",
            "",
            "The campaign controller supports `inspect TASK`, `profile TASK --execute`, `plan TASK`, `launch TASK ARM --execute`, `resume TASK ARM --run RUN_DIR --execute`, `evaluate TASK ARM --run RUN_DIR --checkpoint best --max-samples 128 --panel uniform --execute`, and `stop`; execution commands accept explicit physical GPU and bounded `--max-seconds` options. Inspection, analysis and reporting may write owned metadata or derived outputs and do not train models; status reads records. GPU work requires `--execute`. Stop signals only the active owned process group.",
            "",
        ]
    )
    c0 = None
    if isinstance(contract_check, dict) and contract_check.get("reference_exactly_matches_profile"):
        c0 = _number(contract_check.get("C0"))
    validation_source = _source_validation_coherence_sum(inputs, "G")
    if c0 is not None or validation_source is not None:
        lines.extend(
            [
                "",
                f"For the G contract check, TRAIN C0 is `{_fmt(c0, 8) if c0 is not None else 'not recorded'}` and the paired epoch-0 validation SOURCE coherence sum is `{_fmt(validation_source, 8) if validation_source is not None else 'not yet present in analyzer output'}`. These values serve different populations and are never used interchangeably.",
            ]
        )
    lines.extend(["", "| Evidence artifact | SHA-256 (prefix) |", "|---|---|"])
    evidence_paths = [
        inputs["root"] / "manifests" / "science_source_binding_v1.json",
        inputs["root"] / "manifests" / "science_predeclaration.json",
        inputs["root"] / "tasks" / "task_inventory.json",
        inputs["root"] / "manifests" / "campaign_runner_ledger.jsonl",
        inputs["root"] / "analysis" / "first_science_contract_check.json",
        inputs["history_path"],
        inputs["root"] / "analysis" / "final_interpretation.json",
        inputs["runtime_policy_summary_path"],
        inputs["root"] / "analysis" / "quality_history.csv",
        inputs["root"] / "analysis" / "quality_components.csv",
        inputs["root"] / "analysis" / "decision_cost_quality.json",
        inputs["root"] / "analysis" / "decision_cost_quality.csv",
        inputs["root"] / "analysis" / "mature_window_metrics.csv",
        inputs["fixed_query_attestation_path"],
        inputs["bottleneck_scope_limitations_path"],
        inputs["profile_proxy_label_erratum_path"],
        inputs["root"] / "analysis" / "deterministic_inference_metadata_scope.json",
        inputs["fast_report_path"],
        inputs["root"] / "analysis" / "profiles_warm_reference_v1" / "revision.json",
        inputs["root"] / "analysis" / "profile_v2_complete_cost_revision" / "revision.json",
    ]
    evidence_paths.extend(inputs["root"] / "profiles" / f"{task}_{suffix}.json" for task in DEFAULT_TASKS for suffix in ("profile", "raw_profile"))
    evidence_paths.extend(inputs["root"] / "plans" / f"{task}_plan_decision.json" for task in DEFAULT_TASKS)
    evidence_paths.extend(inputs["root"] / "profiles" / f"{task}_source_calibration.json" for task in DEFAULT_TASKS)
    evidence_paths.extend(inputs["root"] / "tasks" / f"{task}_task_contract.json" for task in DEFAULT_TASKS)
    seen = set()
    for evidence_path in evidence_paths:
        if evidence_path in seen:
            continue
        seen.add(evidence_path)
        digest = _sha256(evidence_path)
        relative = _rel(evidence_path, root)
        lines.append(f"| [{relative}](../{relative}) | {digest[:16] if digest else 'missing'} |")
    if not history:
        lines.extend(["", "Analyzer output is absent; all primary quality histories and mature-window classifications remain pending. Rerun the analyzer after its inputs are finalized, then regenerate this report."])
    elif not all_complete:
        pending = [f"{item['task_id']}/{item['arm']}={item['status']}" for item in arm_records if item["status"] != "complete"]
        lines.extend(["", "Mandatory horizon gate remains open for: " + ", ".join(pending) + ". No overall quality verdict or campaign-completion claim is emitted."])
    elif interpretation["status"] == "reviewed":
        lines.extend(["", "The main-reviewed interpretation above is bound to the current quality history and campaign ledger; this report does not auto-select a winner."])
    else:
        lines.extend(["", "All mandatory horizons and native assessor inputs are present. This report preserves the predeclared per-arm results and does not auto-select a winner; comparative interpretation remains subject to main-review of the complete evidence."])
    lines.append("")
    return "\n".join(lines)


def build_report(root: str | Path, *, history_path: str | Path | None = None) -> Path:
    """Generate the owned Markdown report and three PDF figures from evidence."""
    root = Path(root).resolve(strict=True)
    requested_history = Path(history_path).resolve(strict=False) if history_path else None
    inputs = _load_inputs(root, requested_history)
    attempts = _attempts(inputs["ledger_rows"])
    _, completion_records = _all_mandatory_complete(inputs)
    _write_analysis_tables(inputs, attempts)
    reports_dir = root / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_names = (
        "figures/figure_1_cost_decision.pdf",
        "figures/figure_2_quality_history.pdf",
        "figures/figure_3_mature_windows_transfer.pdf",
    )
    figure_paths = tuple(reports_dir / name for name in figure_names)
    _write_cost_figure(inputs, attempts, completion_records, figure_paths[0])
    _write_quality_figure(inputs, figure_paths[1])
    _write_maturity_figure(inputs, figure_paths[2])
    report = _make_report(inputs, attempts, completion_records, figure_names)
    output_path = reports_dir / "COST_AWARE_LAGRANGIAN_REPORT.md"
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.write_text(report, encoding="utf-8")
    os.replace(temporary, output_path)
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="owned campaign root below /data2")
    parser.add_argument("--history", type=Path, help="quality-history JSON (default ROOT/analysis/quality_history.json)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = build_report(args.root, history_path=args.history)
    except (ReportBuildError, OSError, ValueError, TypeError) as error:
        print(f"build_cost_aware_report: {error}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
