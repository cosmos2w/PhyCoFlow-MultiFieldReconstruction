#!/usr/bin/env python3
"""Finalize compact campaign metadata and optionally export its Markdown report.

This utility reads only campaign metadata, the runner ledger, readiness records,
and the already-built Markdown report. It does not open datasets, checkpoints,
validators, or raw profile vectors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit, urlunsplit

TASK_IDS = ("G", "D", "X")
OWNER_SCHEMAS = {
    "task_contracts.json": "cost_aware_task_contracts.v1",
    "action_registry.json": "cost_aware_action_registry.v1",
    "profiles.json": "cost_aware_compact_profiles.v1",
    "resource_summary.json": "cost_aware_resource_summary.v1",
}
PLAN_ROW_SCHEMA = "cost_aware_plan_decision_record.v1"
REPORT_NAME = "COST_AWARE_LAGRANGIAN_REPORT.md"
REPORT_MARKER_PREFIX = "<!-- CostAwareLPC owned export v1; root="
_INLINE_LINK_RE = re.compile(
    r"(?P<open>!?\[[^\]]*\]\()"
    r"(?P<destination><[^>]*>|(?:\\.|[^\s)])+)"
    r"(?P<tail>(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\))"
)


class FinalizeError(RuntimeError):
    """Raised for missing, malformed, or conflicting campaign artifacts."""


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise FinalizeError(f"expected a regular, non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    _sha256(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FinalizeError(f"cannot read JSON artifact {path}: {error}") from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    _sha256(path)
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise FinalizeError(f"expected an object on line {line_number} of {path}")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise FinalizeError(f"cannot read JSONL artifact {path}: {error}") from error
    return rows


def _source_record(root: Path, relative_path: str) -> tuple[Any, dict[str, str]]:
    path = root / relative_path
    value = _read_json(path)
    return value, {"path": relative_path, "sha256": _sha256(path)}


def _owned_output_preflight(path: Path, expected_schema: str, expected_root: str) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_file():
        raise FinalizeError(f"refusing to replace non-file or symlink output: {path}")
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FinalizeError(f"refusing to replace unowned or malformed output {path}: {error}") from error
    if (
        not isinstance(existing, dict)
        or existing.get("schema") != expected_schema
        or existing.get("root") != expected_root
    ):
        raise FinalizeError(f"refusing to replace output without matching ownership schema/root: {path}")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_owned(path: Path, value: dict[str, Any], schema: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _owned_output_preflight(path, schema, str(value["root"]))
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _compact_measurement(measurement: dict[str, Any]) -> dict[str, Any]:
    # Explicit allowlist excludes support-point arrays and any raw vectors.
    fields = (
        "plan_id",
        "mean_epoch_cost_s",
        "upper_epoch_cost_s",
        "actions",
        "acquisition_overhead_s",
        "refresh_overhead_s",
        "audit_overhead_s",
        "extra_variance",
        "variance_method",
        "estimator_kind",
        "family_probabilities",
        "within_family",
        "support_validated",
        "support_nonempty",
        "measured_split",
        "evidence_status",
        "reason",
        "max_joint_importance_multiplier",
    )
    return {field: measurement[field] for field in fields if field in measurement}


def compact_profile(task_id: str, profile: dict[str, Any], source: dict[str, str]) -> dict[str, Any]:
    if not isinstance(profile.get("measurements"), list):
        raise FinalizeError(f"profile for {task_id} has no measurements list")
    fields = (
        "schema_version",
        "task_hash",
        "hardware_fingerprint",
        "profiled_epoch",
        "max_epochs",
        "intrinsic_noise_variance",
        "noise_split",
        "profile_split",
        "profile_overhead_s",
    )
    result = {field: profile[field] for field in fields if field in profile}
    result["task_id"] = task_id
    result["measurements"] = [
        _compact_measurement(item)
        for item in profile["measurements"]
        if isinstance(item, dict)
    ]
    result["source"] = source
    return result


def _latest_value(rows: list[dict[str, Any]], field: str) -> Any:
    for row in reversed(rows):
        if field in row and row[field] is not None:
            return row[field]
    return None


def deduplicate_ledger_attempts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse append-only runner events to one latest accounting row per attempt."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = str(row.get("attempt_id") or f"row-{index}")
        grouped[key].append(row)

    charge_fields = (
        "charged_process_seconds",
        "charged_profile_seconds",
        "charged_calibration_seconds",
        "charged_closeout_seconds",
        "charged_science_epochs",
        "charged_aux_epochs",
        "charged_stop_loss_epochs",
        "reserved_process_seconds",
        "reserved_profile_seconds",
        "reserved_calibration_seconds",
        "reserved_closeout_seconds",
        "reserved_science_epochs",
        "reserved_aux_epochs",
        "stop_loss_reserve_epochs",
    )
    result = []
    for attempt_id, events in grouped.items():
        latest = events[-1]
        gpu = latest.get("gpu")
        result.append(
            {
                "attempt_id": attempt_id,
                "task_id": latest.get("task_id"),
                "arm": latest.get("arm"),
                "action": latest.get("action"),
                "status": latest.get("status", "unknown"),
                "started_utc": _latest_value(events, "started_utc"),
                "finished_utc": _latest_value(events, "finished_utc"),
                "return_code": _latest_value(events, "return_code"),
                "gpu_physical_index": gpu.get("physical_index") if isinstance(gpu, dict) else None,
                "command_sha256": _latest_value(events, "command_sha256"),
                "accounting": {field: _latest_value(events, field) for field in charge_fields},
                "event_count": len(events),
            }
        )
    result.sort(key=lambda item: (str(item.get("started_utc") or ""), item["attempt_id"]))
    return result


def _resource_totals(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    charge_fields = (
        "charged_process_seconds",
        "charged_profile_seconds",
        "charged_calibration_seconds",
        "charged_closeout_seconds",
        "charged_science_epochs",
        "charged_aux_epochs",
        "charged_stop_loss_epochs",
    )
    totals: dict[str, Any] = {}
    for field in charge_fields:
        values = [item["accounting"].get(field) for item in attempts]
        numeric = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
        totals[field] = {
            "recorded_sum": sum(numeric),
            "attempts_with_recorded_value": len(numeric),
            "attempts_missing_value": len(values) - len(numeric),
            "complete_for_latest_attempts": len(numeric) == len(values),
        }
    by_task_action: dict[tuple[str, str], dict[str, Any]] = {}
    for item in attempts:
        key = (str(item.get("task_id") or "unassigned"), str(item.get("action") or "unassigned"))
        aggregate = by_task_action.setdefault(
            key,
            {"attempt_count": 0, "recorded_process_seconds": 0.0, "process_charge_count": 0, "statuses": Counter()},
        )
        aggregate["attempt_count"] += 1
        aggregate["statuses"][str(item.get("status") or "unknown")] += 1
        value = item["accounting"].get("charged_process_seconds")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            aggregate["recorded_process_seconds"] += float(value)
            aggregate["process_charge_count"] += 1
    grouped = []
    for (task_id, action), aggregate in sorted(by_task_action.items()):
        grouped.append(
            {
                "task_id": task_id,
                "action": action,
                "attempt_count": aggregate["attempt_count"],
                "recorded_process_seconds": aggregate["recorded_process_seconds"],
                "process_charge_count": aggregate["process_charge_count"],
                "status_counts": dict(aggregate["statuses"]),
            }
        )
    return {"recorded_charges": totals, "by_task_action": grouped}


def compact_readiness_estimates(rows: list[dict[str, Any]], source: dict[str, str]) -> dict[str, Any]:
    estimates = []
    for row in rows:
        seconds = row.get("conservative_command_seconds")
        estimates.append(
            {
                "phase": row.get("phase"),
                "outcome": row.get("outcome"),
                "reason": row.get("reason"),
                "conservative_command_seconds": seconds,
                "estimated_command_seconds": seconds,
                "recorded_gpu_process_seconds": row.get("gpu_process_seconds"),
                "optimizer_exposures": row.get("optimizer_exposures"),
                "estimate_scope": row.get(
                    "estimate_scope",
                    "Recorded conservative pre-admission command-time estimate; not a measured duration or proven upper bound.",
                ),
                "counted_in_runner_actuals": False,
                "allocated_to_strategy_profile_estimate": row.get("phase") == "D profile preflight",
                "counted_as_measured_profile_acquisition": False,
                "counted_as_gpu_process_time": False,
            }
        )
    recorded = [
        float(item["estimated_command_seconds"])
        for item in estimates
        if isinstance(item["estimated_command_seconds"], (int, float))
        and not isinstance(item["estimated_command_seconds"], bool)
    ]
    return {
        "source": source,
        "records": estimates,
        "recorded_estimated_command_seconds": sum(recorded),
        "allocated_to_strategy_profile_estimate_seconds": sum(
            float(item["estimated_command_seconds"])
            for item in estimates
            if item["allocated_to_strategy_profile_estimate"]
            and isinstance(item["estimated_command_seconds"], (int, float))
            and not isinstance(item["estimated_command_seconds"], bool)
        ),
        "counted_in_runner_actuals": False,
        "counted_as_measured_profile_acquisition": False,
        "counted_as_gpu_process_time": False,
        "scope_note": (
            "CPU readiness/pre-admission command estimates are not runner charges or measured GPU/process time. "
            "The recorded D profile preflight estimate is allocated separately to the strategy profile-acquisition "
            "estimate; it is not part of measured profile acquisition."
        ),
    }


def _charge_value(attempt: dict[str, Any], field: str) -> float:
    value = attempt["accounting"].get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise FinalizeError(f"attempt {attempt['attempt_id']} has a missing or invalid {field} charge")
    return float(value)


def _optional_zero_charge(attempt: dict[str, Any], field: str) -> float:
    value = attempt["accounting"].get(field)
    if value is None:
        return 0.0
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise FinalizeError(f"attempt {attempt['attempt_id']} has an invalid {field} charge")
    return float(value)


def _assert_final_resource_accounting(root: Path, attempts: list[dict[str, Any]]) -> None:
    active_path = root / "manifests/active_process.json"
    if active_path.exists() or active_path.is_symlink():
        raise FinalizeError(f"refusing final resource summary while active-process receipt exists: {active_path}")
    running = [item["attempt_id"] for item in attempts if item.get("status") == "running"]
    if running:
        raise FinalizeError(f"refusing final resource summary while ledger attempts are active: {running}")

    required_fields = (
        "charged_process_seconds",
        "charged_profile_seconds",
        "charged_science_epochs",
        "charged_aux_epochs",
    )
    for attempt in attempts:
        for field in required_fields:
            _charge_value(attempt, field)
        # The runner records closeout only for evaluation commands. Its absence
        # on profile/training commands means zero closeout exposure by design.
        if attempt.get("action") == "evaluate":
            _charge_value(attempt, "charged_closeout_seconds")


def resource_budget_summary(
    attempts: list[dict[str, Any]], readiness: dict[str, Any]
) -> dict[str, Any]:
    science = sum(_charge_value(item, "charged_science_epochs") for item in attempts)
    auxiliary = sum(_charge_value(item, "charged_aux_epochs") for item in attempts)
    process = sum(_charge_value(item, "charged_process_seconds") for item in attempts)
    profile_calibration = sum(_charge_value(item, "charged_profile_seconds") for item in attempts)
    closeout = sum(
        _charge_value(item, "charged_closeout_seconds")
        if item.get("action") == "evaluate"
        else _optional_zero_charge(item, "charged_closeout_seconds")
        for item in attempts
    )

    measured_totals = {
        "science_epochs": science,
        "auxiliary_epoch_exposures": auxiliary,
        "combined_epoch_exposures": science + auxiliary,
        "process_seconds": process,
        "profile_calibration_seconds": profile_calibration,
        "closeout_seconds": closeout,
    }
    caps = {
        "science_epochs": 800.0,
        "auxiliary_epoch_exposures": 50.0,
        "combined_epoch_exposures": 850.0,
        "process_seconds": 57_600.0,
        "profile_calibration_seconds": 5_400.0,
        "closeout_seconds": 3_600.0,
    }
    within_caps = {name: measured_totals[name] <= cap for name, cap in caps.items()}
    within_caps["all"] = all(within_caps.values())
    readiness_seconds = float(readiness["recorded_estimated_command_seconds"])
    return {
        "measured_totals": measured_totals,
        "caps": caps,
        "within_caps": within_caps,
        "recorded_cpu_readiness_estimates_seconds": readiness_seconds,
        "accounted_process_plus_cpu_readiness_estimates_seconds": process + readiness_seconds,
        "resource_accounting_complete": True,
        "scope_note": (
            "Profile/calibration and closeout are reported as named subscopes of process_seconds and are not added "
            "again to process totals. charged_profile_seconds already carries measured profile time or the runner's "
            "calibration charge; charged_calibration_seconds is a detail label and is not added again. Auxiliary "
            "epoch exposures use charged_aux_epochs, which already includes any charged stop-loss exposure. CPU "
            "readiness estimates are separate from measured runner totals."
        ),
    }


def aggregate_campaign(root: Path) -> dict[str, dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise FinalizeError(f"campaign root must be a real owned directory: {root}")
    root = root.resolve(strict=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    task_contracts: dict[str, Any] = {"schema": OWNER_SCHEMAS["task_contracts.json"], "root": root.name, "generated_at_utc": generated_at, "tasks": {}}
    action_registry: dict[str, Any] = {"schema": OWNER_SCHEMAS["action_registry.json"], "root": root.name, "generated_at_utc": generated_at, "tasks": {}}
    profiles: dict[str, Any] = {"schema": OWNER_SCHEMAS["profiles.json"], "root": root.name, "generated_at_utc": generated_at, "tasks": {}}
    plan_rows = []
    for task_id in TASK_IDS:
        contract_path = f"tasks/{task_id}_task_contract.json"
        contract, contract_source = _source_record(root, contract_path)
        if not isinstance(contract, dict):
            raise FinalizeError(f"task contract for {task_id} must be an object")
        task_contracts["tasks"][task_id] = {"source": contract_source, "contract": contract}

        registry_path = f"tasks/{task_id}_action_registry.json"
        registry, registry_source = _source_record(root, registry_path)
        if not isinstance(registry, list) or not all(isinstance(row, dict) for row in registry):
            raise FinalizeError(f"action registry for {task_id} must be a list of objects")
        action_registry["tasks"][task_id] = {"source": registry_source, "actions": registry}

        profile_path = f"profiles/{task_id}_profile.json"
        profile, profile_source = _source_record(root, profile_path)
        if not isinstance(profile, dict):
            raise FinalizeError(f"profile for {task_id} must be an object")
        profiles["tasks"][task_id] = compact_profile(task_id, profile, profile_source)

        decision_path = f"plans/{task_id}_plan_decision.json"
        decision, decision_source = _source_record(root, decision_path)
        if not isinstance(decision, dict):
            raise FinalizeError(f"plan decision for {task_id} must be an object")
        plan_rows.append(
            {
                "schema": PLAN_ROW_SCHEMA,
                "root": root.name,
                "task_id": task_id,
                "source": decision_source,
                "decision": decision,
            }
        )

    ledger_path = root / "manifests/campaign_runner_ledger.jsonl"
    ledger_rows = _read_jsonl(ledger_path)
    attempts = deduplicate_ledger_attempts(ledger_rows)
    _assert_final_resource_accounting(root, attempts)
    readiness_path = root / "analysis/readiness_attempts.jsonl"
    if readiness_path.is_symlink():
        raise FinalizeError(f"readiness attempts must not be a symlink: {readiness_path}")
    readiness_rows = _read_jsonl(readiness_path) if readiness_path.is_file() else []
    readiness = compact_readiness_estimates(
        readiness_rows,
        {
            "path": "analysis/readiness_attempts.jsonl",
            "sha256": _sha256(readiness_path) if readiness_path.is_file() else None,
        },
    )
    budget = resource_budget_summary(attempts, readiness)
    resource_summary = {
        "schema": OWNER_SCHEMAS["resource_summary.json"],
        "root": root.name,
        "generated_at_utc": generated_at,
        "actuals_source": "latest append-only campaign runner ledger event per attempt_id",
        "accounting_scope_note": "Process lifetimes are measured admitted scientific commands. Epoch exposures are ledger charges, including one conservative lost-interval recovery epoch; they are not exact optimizer-action counts. CPU readiness estimates are separate. Software development and report authoring are outside admitted GPU-process lifetimes.",
        "ledger": {
            "source": {"path": "manifests/campaign_runner_ledger.jsonl", "sha256": _sha256(ledger_path)},
            "event_row_count": len(ledger_rows),
            "unique_attempt_count": len(attempts),
            "accounting_note": (
                "charged_process_seconds is the primary process total. Profile, calibration, and closeout charges are "
                "reported as separate subscopes and must not be added to it without a disjoint accounting rule."
            ),
            "latest_status_counts": dict(Counter(str(item.get("status") or "unknown") for item in attempts)),
            **_resource_totals(attempts),
            "latest_attempts": attempts,
        },
        "cpu_readiness_estimates": readiness,
        **budget,
    }
    return {
        "task_contracts.json": task_contracts,
        "action_registry.json": action_registry,
        "profiles.json": profiles,
        "plan_decisions.jsonl": {"rows": plan_rows, "root": root},
        "resource_summary.json": resource_summary,
    }


def _preflight_plan_jsonl(path: Path, expected_root: str) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_file():
        raise FinalizeError(f"refusing to replace non-file or symlink output: {path}")
    existing = _read_jsonl(path)
    if not existing or any(
        row.get("schema") != PLAN_ROW_SCHEMA or row.get("root") != expected_root for row in existing
    ):
        raise FinalizeError(f"refusing to replace plan JSONL without owned schema {PLAN_ROW_SCHEMA}: {path}")


def write_aggregates(root: Path, records: dict[str, Any]) -> list[Path]:
    root = root.resolve(strict=True)
    analysis = root / "analysis"
    if analysis.is_symlink() or not analysis.is_dir():
        raise FinalizeError(f"analysis must be a real directory: {analysis}")
    for name, schema in OWNER_SCHEMAS.items():
        _owned_output_preflight(analysis / name, schema, root.name)
    plan_path = analysis / "plan_decisions.jsonl"
    _preflight_plan_jsonl(plan_path, root.name)

    plan_record = records["plan_decisions.jsonl"]
    if len(plan_record["rows"]) != 3 or {row["task_id"] for row in plan_record["rows"]} != set(TASK_IDS):
        raise FinalizeError("plan output must contain exactly one record for G, D, and X")

    written = []
    for filename, schema in OWNER_SCHEMAS.items():
        path = analysis / filename
        _write_json_owned(path, records[filename], schema)
        written.append(path)
    _atomic_write_text(
        plan_path,
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in plan_record["rows"]),
    )
    written.append(plan_path)
    return written


def _rewrite_destination(destination: str, source_report: Path, root: Path, alias_relative: str) -> str:
    angle = destination.startswith("<") and destination.endswith(">")
    raw = destination[1:-1] if angle else destination
    parsed = urlsplit(raw)
    if parsed.scheme or parsed.netloc or not parsed.path or parsed.path.startswith("/"):
        return destination
    decoded_path = unquote(parsed.path)
    if decoded_path.startswith("#"):
        return destination
    resolved = (source_report.parent / decoded_path).resolve(strict=False)
    try:
        relative_to_campaign = resolved.relative_to(root.resolve(strict=True)).as_posix()
    except ValueError as error:
        raise FinalizeError(f"relative report link escapes the owned campaign root: {raw}") from error
    new_path = f"{alias_relative}/{quote(relative_to_campaign, safe='/-._~')}"
    rewritten = urlunsplit(("", "", new_path, parsed.query, parsed.fragment))
    return f"<{rewritten}>" if angle else rewritten


def rebase_markdown_links(text: str, source_report: Path, root: Path, alias_relative: str) -> str:
    """Rebase every inline relative Markdown link/image to the owned evidence alias."""

    def replace(match: re.Match[str]) -> str:
        destination = match.group("destination")
        return (
            match.group("open")
            + _rewrite_destination(destination, source_report, root, alias_relative)
            + match.group("tail")
        )

    return _INLINE_LINK_RE.sub(replace, text)


def ensure_evidence_alias(repo_root: Path, campaign_root: Path) -> Path:
    repo_root = Path(repo_root)
    campaign_root = Path(campaign_root)
    if repo_root.is_symlink():
        raise FinalizeError(f"repository root must not be a symlink: {repo_root}")
    if campaign_root.is_symlink():
        raise FinalizeError(f"campaign root must not be a symlink: {campaign_root}")
    repo_root = repo_root.resolve(strict=True)
    if not repo_root.is_dir():
        raise FinalizeError(f"repository root must be a real directory: {repo_root}")
    campaign_root = campaign_root.resolve(strict=True)
    if not campaign_root.is_dir():
        raise FinalizeError(f"campaign root must be a real directory: {campaign_root}")
    report_directory = repo_root / "reports"
    if report_directory.is_symlink():
        raise FinalizeError(f"refusing to use symlink report directory: {report_directory}")
    if report_directory.exists() and not report_directory.is_dir():
        raise FinalizeError(f"repository reports path is not a directory: {report_directory}")
    report_directory.mkdir(parents=True, exist_ok=True)
    alias_name = f"{campaign_root.name}_evidence"
    if not re.fullmatch(r"[A-Za-z0-9._-]+", alias_name):
        raise FinalizeError(f"campaign root name is not safe for an evidence alias: {campaign_root.name}")
    alias = report_directory / alias_name
    if alias.is_symlink():
        try:
            alias_target = alias.resolve(strict=True)
        except OSError as error:
            raise FinalizeError(f"evidence alias is broken or unreadable: {alias}") from error
        if alias_target != campaign_root:
            raise FinalizeError(f"evidence alias already points elsewhere: {alias}")
        return alias
    if alias.exists():
        raise FinalizeError(f"evidence alias path is occupied by a non-symlink: {alias}")
    os.symlink(campaign_root, alias, target_is_directory=True)
    return alias


def export_repository_report(campaign_root: Path, repo_root: Path) -> Path:
    campaign_root = campaign_root.resolve(strict=True)
    source_report = campaign_root / "reports" / REPORT_NAME
    if source_report.is_symlink() or not source_report.is_file():
        raise FinalizeError(f"owned campaign Markdown report is missing or unsafe: {source_report}")
    if Path(repo_root).is_symlink():
        raise FinalizeError(f"repository root must not be a symlink: {repo_root}")
    repo_root = Path(repo_root).resolve(strict=True)
    destination = repo_root / REPORT_NAME
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise FinalizeError(f"refusing to overwrite non-file or symlink repository report: {destination}")

    marker = f"{REPORT_MARKER_PREFIX}{campaign_root} -->"
    source_sha = _sha256(source_report)
    if destination.exists():
        first_line = destination.open("r", encoding="utf-8").readline().rstrip("\n")
        if first_line != marker:
            raise FinalizeError(f"refusing to overwrite repository Markdown without this campaign ownership marker: {destination}")
    alias_name = f"{campaign_root.name}_evidence"
    alias_relative = f"reports/{alias_name}"
    source_text = source_report.read_text(encoding="utf-8")
    rebased = rebase_markdown_links(source_text, source_report, campaign_root, alias_relative)
    alias = ensure_evidence_alias(repo_root, campaign_root)
    if alias.name != alias_name:
        raise FinalizeError("evidence alias name changed during export")
    exported = f"{marker}\n<!-- Source SHA-256: {source_sha} -->\n\n{rebased}"
    _atomic_write_text(destination, exported)
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="owned campaign root on /data2")
    parser.add_argument(
        "--repository-report",
        action="store_true",
        help="export the owned Markdown to the repository root and create a safe evidence symlink",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root
    if root.is_symlink() or not root.is_dir():
        raise FinalizeError(f"campaign root must be a real directory: {root}")
    records = aggregate_campaign(root)
    written = write_aggregates(root, records)
    if args.repository_report:
        repo_root = Path(__file__).resolve().parents[2]
        written.append(export_repository_report(root, repo_root))
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FinalizeError as error:
        print(f"finalize_cost_aware_records: {error}", file=sys.stderr)
        raise SystemExit(2) from error
