#!/usr/bin/env python3
"""Build a read-only quality-history index from campaign run artifacts.

The analyzer reads campaign manifests, ledgers, training metric JSONL, and
paired validation JSON. It never imports a trainer/evaluator, opens a dataset,
loads a checkpoint, or runs a validator. The current evaluator stores paired
group means rather than raw accumulators, so reconstructed numerators are
explicitly labeled and use the frozen estimator counts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src"
DEFAULT_ROOT = Path(
    "/data2/wanglz/CostAwareLPC/CostAwareLPC_20261023T185427Z_01a16a15"
)
EVALUATION_SCHEMA = "phycoflow_full_abt_validation.v1"
QUALITY_SCHEMA = "cost_aware_campaign_quality_history.v1"
EXPECTED_PANEL_ALGORITHM = "chronological_linspace_v1"
FIXED_SAMPLES = 128
GROUP_SIZE = 32
QUERY_COUNT = 4096
TOPOLOGY_FILTRATIONS = 6


class CampaignAnalysisError(ValueError):
    """Raised when campaign metadata cannot be read safely."""


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignAnalysisError(f"cannot read {label} at {path}: {error}") from error
    if not isinstance(value, dict):
        raise CampaignAnalysisError(f"expected an object for {label}: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignAnalysisError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise CampaignAnalysisError(f"{label} must be finite and positive")
    return number


def _utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    if len(value) in {15, 16} and value[:8].isdigit() and value[8] == "T":
        try:
            parsed = datetime.strptime(
                value.rstrip("Z") + "+0000", "%Y%m%dT%H%M%S%z"
            )
        except ValueError:
            return None
        return parsed.astimezone(timezone.utc)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _relative_seconds(later: datetime | None, earlier: datetime | None) -> float | None:
    if later is None or earlier is None or later < earlier:
        return None
    return (later - earlier).total_seconds()


def _quality_history_module():
    """Import only the pure assessor module, without importing the evaluator."""
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    return importlib.import_module("phycoflow_reconstruction.evaluation.quality_history")


def _option(command: Any, name: str) -> str | None:
    if not isinstance(command, list):
        return None
    try:
        return str(command[command.index(name) + 1])
    except (ValueError, IndexError):
        return None


def _ledger_attempts(root: Path) -> dict[str, list[dict[str, Any]]]:
    path = root / "manifests" / "campaign_runner_ledger.jsonl"
    attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not path.is_file():
        return attempts
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("attempt_id"):
                raise CampaignAnalysisError(f"invalid campaign ledger row {line_number}")
            attempts[str(row["attempt_id"])].append(row)
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignAnalysisError(f"cannot read campaign runner ledger: {error}") from error
    return attempts


def _attempt_run_path(rows: Sequence[Mapping[str, Any]]) -> Path | None:
    for row in reversed(rows):
        observation = row.get("actual_science_epoch_observation")
        if isinstance(observation, Mapping) and observation.get("run_dir"):
            return Path(str(observation["run_dir"]))
        if row.get("run_dir"):
            return Path(str(row["run_dir"]))
    return None


def _attempt_summary(attempt_id: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    terminal = dict(rows[-1])
    started = next((row.get("started_utc") for row in rows if row.get("started_utc")), None)

    def latest(name: str) -> Any:
        return next((row.get(name) for row in reversed(rows) if row.get(name) is not None), None)

    return {
        "attempt_id": attempt_id,
        "action": terminal.get("action"),
        "status": terminal.get("status", "unknown"),
        "started_utc": started,
        "finished_utc": terminal.get("finished_utc"),
        "charged_process_seconds": terminal.get("charged_process_seconds"),
        "charged_science_epochs": terminal.get("charged_science_epochs"),
        "charged_aux_epochs": terminal.get("charged_aux_epochs"),
        "charged_stop_loss_epochs": terminal.get("charged_stop_loss_epochs"),
        "science_epoch_charge_basis": terminal.get("science_epoch_charge_basis"),
        "return_code": terminal.get("return_code"),
        "command": latest("command"),
        "command_sha256": latest("command_sha256"),
        "gpu": latest("gpu"),
        "output_artifact": latest("output_artifact"),
        "epoch_observation": latest("actual_science_epoch_observation"),
        "ledger_states": [
            {
                "status": row.get("status"),
                "recorded_utc": row.get("started_utc") or row.get("finished_utc"),
                "reserved_process_seconds": row.get("reserved_process_seconds"),
                "reserved_science_epochs": row.get("reserved_science_epochs"),
                "reserved_aux_epochs": row.get("reserved_aux_epochs"),
                "charged_process_seconds": row.get("charged_process_seconds"),
                "charged_science_epochs": row.get("charged_science_epochs"),
                "charged_aux_epochs": row.get("charged_aux_epochs"),
            }
            for row in rows
        ],
    }


def _run_attempts(
    attempts: Mapping[str, Sequence[Mapping[str, Any]]],
    run_dir: Path,
    *,
    task_id: str | None = None,
    arm: str | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    selected = []
    expected = run_dir.resolve()
    created = _utc(manifest.get("created_utc")) if isinstance(manifest, Mapping) else None
    experiment_name = manifest.get("experiment_name") if isinstance(manifest, Mapping) else None
    experiment_matches = (
        isinstance(experiment_name, str)
        and task_id is not None
        and arm is not None
        and experiment_name.endswith(f"/{task_id}/{arm}")
    )
    campaign_root = expected.parents[3] if len(expected.parents) > 3 else None
    active_unbound: list[tuple[str, Sequence[Mapping[str, Any]]]] = []
    for attempt_id, rows in attempts.items():
        actions = {row.get("action") for row in rows}
        if not actions.intersection({"launch", "resume"}):
            continue
        path = _attempt_run_path(rows)
        if path is not None and path.resolve() == expected:
            selected.append(_attempt_summary(attempt_id, rows))
            continue
        # The foreground controller records the run path at terminal closeout.
        # During a live launch, bind its pathless ledger row only when the
        # manifest identifies this campaign task/arm, the exact frozen config
        # path is in the command, and the manifest was created immediately
        # after this one active launch. Never guess across a resume or an old
        # failed launch.
        terminal = rows[-1] if rows else {}
        if (
            path is None
            and created is not None
            and experiment_matches
            and terminal.get("status") == "running"
            and terminal.get("action") == "launch"
            and terminal.get("task_id") == task_id
            and terminal.get("arm") == arm
            and campaign_root is not None
        ):
            started = _utc(next((row.get("started_utc") for row in rows if row.get("started_utc")), None))
            command_config = _option(terminal.get("command"), "--config")
            if (
                started is not None
                and 0 <= (created - started).total_seconds() <= 60
                and command_config is not None
                and Path(command_config).resolve()
                == (campaign_root / "configs" / f"{task_id}_{arm}.yaml").resolve()
            ):
                active_unbound.append((attempt_id, rows))
    if len(active_unbound) == 1:
        attempt_id, rows = active_unbound[0]
        selected.append(_attempt_summary(attempt_id, rows))
    selected.sort(key=lambda item: item.get("started_utc") or "")
    return selected


def _evaluation_attempts(
    attempts: Mapping[str, Sequence[Mapping[str, Any]]], output_dir: Path
) -> list[dict[str, Any]]:
    expected = output_dir.resolve()
    selected = []
    for attempt_id, rows in attempts.items():
        if not any(row.get("action") == "evaluate" for row in rows):
            continue
        output = None
        for row in reversed(rows):
            if row.get("output_artifact"):
                output = Path(str(row["output_artifact"]))
                break
            output_text = _option(row.get("command"), "--output-dir")
            if output_text:
                output = Path(output_text)
                break
        if output is None or output.resolve() != expected:
            continue
        selected.append(_attempt_summary(attempt_id, rows))
    selected.sort(key=lambda item: item.get("started_utc") or "")
    return selected


def _metric_history_summary(run_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name in (
        "history",
        "validation_history",
        "topology_validation",
        "execution_strategy_updates",
        "execution_strategy_audits",
    ):
        path = run_dir / "metrics" / f"{name}.jsonl"
        if not path.is_file():
            continue
        row_count = 0
        max_step: int | None = None
        max_epoch: float | None = None
        malformed_lines: list[int] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed_lines.append(line_number)
                    continue
                if not isinstance(row, Mapping):
                    malformed_lines.append(line_number)
                    continue
                row_count += 1
                step = row.get("step")
                epoch = row.get("epoch")
                if isinstance(step, int) and not isinstance(step, bool):
                    max_step = step if max_step is None else max(max_step, step)
                if isinstance(epoch, (int, float)) and not isinstance(epoch, bool):
                    max_epoch = float(epoch) if max_epoch is None else max(max_epoch, float(epoch))
        summary[name] = {
            "path": str(path),
            "rows": row_count,
            "max_recorded_step": max_step,
            "max_recorded_epoch_label": max_epoch,
            "malformed_lines": malformed_lines,
        }
    return summary


def _measurement_mapping(child: Any, source: Any, count: float, label: str) -> dict[str, float]:
    child_value = _number(child, f"{label} child mean")
    source_value = _number(source, f"{label} SOURCE mean", positive=True)
    return {
        "numerator": child_value * count,
        "count": count,
        "source_numerator": source_value * count,
        "source_count": count,
    }


def _ratios(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return descriptive ratios from already-validated reconstructed counts."""
    def ratio(item: Mapping[str, Any]) -> float:
        return (float(item["numerator"]) / float(item["count"])) / (
            float(item["source_numerator"]) / float(item["source_count"])
        )

    b_groups = raw["b_groups"]
    b_pooled = {
        key: sum(float(group["measurement"][key]) for group in b_groups)
        for key in ("numerator", "count", "source_numerator", "source_count")
    }
    return {
        "coherence_sum": ratio(raw["coherence_sum"]),
        "family_A": ratio(raw["families"]["A"]),
        "family_B": ratio(b_pooled),
        "family_T": ratio(raw["families"]["T"]),
        "endpoint_aggregate": ratio(raw["endpoint_aggregate"]),
        "endpoint_fields": {
            name: ratio(value) for name, value in raw["endpoint_fields"].items()
        },
    }


def _group_component_rows(
    *, task_id: str, arm: str, run_id: str, epoch: int, panel: str, payload: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Flatten paired raw diagnostics into tidy rows without combining aliases."""
    rows: list[dict[str, Any]] = []

    def leaves(value: Any, prefix: str):
        if isinstance(value, Mapping):
            for key, item in value.items():
                yield from leaves(item, f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from leaves(item, f"{prefix}[{index}]")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                yield prefix, number

    groups = payload.get("groups", [])
    for group in groups:
        if not isinstance(group, Mapping):
            continue
        group_index = group.get("group")
        paired: dict[str, dict[str, float]] = defaultdict(dict)
        for side in ("source", "child"):
            metrics = group.get(side)
            if not isinstance(metrics, Mapping):
                continue
            paths: list[tuple[str, Any]] = []
            coherence = metrics.get("coherence")
            if isinstance(coherence, Mapping):
                paths.extend((f"coherence.{path}", value) for path, value in leaves(coherence, ""))
            endpoint = metrics.get("endpoint_mse_model_units")
            if isinstance(endpoint, Mapping):
                paths.extend((f"endpoint_mse_model_units.{path}", value) for path, value in leaves(endpoint, ""))
            for metric_path, number in paths:
                paired[metric_path][side] = number
        for metric_path, values in paired.items():
            rows.append(
                {
                    "task": task_id,
                    "arm": arm,
                    "run_id": run_id,
                    "epoch": epoch,
                    "panel": panel,
                    "group": group_index,
                    "metric_path": metric_path,
                    "source_value": values.get("source"),
                    "child_value": values.get("child"),
                }
            )
    return rows


def _paired_group_components(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    components = []
    for group in payload.get("groups", []):
        if not isinstance(group, Mapping):
            continue
        source = group.get("source")
        child = group.get("child")
        if not isinstance(source, Mapping) or not isinstance(child, Mapping):
            continue
        components.append(
            {
                "group": group.get("group"),
                "snapshot_count": len(group.get("sample_ids", [])),
                "source": {
                    "coherence": source.get("coherence"),
                    "endpoint_mse_model_units": source.get("endpoint_mse_model_units"),
                },
                "child": {
                    "coherence": child.get("coherence"),
                    "endpoint_mse_model_units": child.get("endpoint_mse_model_units"),
                },
            }
        )
    return components


def _group_measurement(group: Mapping[str, Any], side: str, metric: str, *path: str) -> float:
    current: Any = group[side]
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise CampaignAnalysisError(f"evaluation group is missing {side}.{'.'.join(path)}")
        current = current[key]
    return _number(current, f"group {group.get('group')} {side} {metric}")


def _topology_metrics(group_metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    coherence = group_metrics.get("coherence")
    if not isinstance(coherence, Mapping):
        return {}
    families = coherence.get("families")
    if not isinstance(families, Mapping):
        return {}
    topology = families.get("topology")
    if not isinstance(topology, Mapping):
        return {}
    # Native run history and run/evaluation/before.json store the full
    # topology contract under diagnostics.scalar_filtrations. Portable
    # evaluations flatten the same value as topology_scalar_filtrations.
    # Normalize both serializations for the shared six-filtration check.
    result = dict(topology)
    diagnostics = topology.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        result.update(diagnostics)
    if "topology_scalar_filtrations" not in result:
        result["topology_scalar_filtrations"] = result.get("scalar_filtrations")
    return result


def _evaluation_epoch(payload: Mapping[str, Any]) -> int:
    epoch_value = _number(payload.get("checkpoint_training_epoch"), "checkpoint_training_epoch")
    epoch = round(epoch_value)
    if not math.isclose(epoch_value, epoch, rel_tol=0.0, abs_tol=1e-8):
        raise CampaignAnalysisError("quality evaluation checkpoint is not on an integer reporting epoch")
    if epoch < 0 or epoch > 100 or epoch % 10:
        raise CampaignAnalysisError("quality evaluation is outside the frozen ten-epoch reporting cadence")
    return int(epoch)


def _validate_evaluation(
    payload: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    run_dir: Path,
    manifest: Mapping[str, Any],
) -> tuple[str, int, dict[str, Any] | None]:
    if payload.get("schema") != EVALUATION_SCHEMA:
        raise CampaignAnalysisError("evaluation schema is not the frozen full A+B+T format")
    if payload.get("run") is None or Path(str(payload["run"])).resolve() != run_dir.resolve():
        raise CampaignAnalysisError("evaluation names a different campaign run")
    if payload.get("config_sha256") != manifest.get("config_sha256"):
        raise CampaignAnalysisError("evaluation config hash differs from the run manifest")
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise CampaignAnalysisError("evaluation is missing the paired SOURCE identity")
    if Path(str(source.get("run", ""))).resolve() != Path(str(task["source_run"])).resolve():
        raise CampaignAnalysisError("evaluation SOURCE run differs from the frozen task")
    source_hashes = source.get("hashes")
    if not isinstance(source_hashes, Mapping) or source_hashes.get("checkpoint") != task.get(
        "source_checkpoint_sha256"
    ):
        raise CampaignAnalysisError("evaluation SOURCE checkpoint differs from the frozen task")

    dataset = payload.get("dataset")
    if not isinstance(dataset, Mapping) or dataset.get("split") != "validation":
        raise CampaignAnalysisError("only explicit validation-split evaluation is admitted")
    identity = task.get("dataset_identity", {})
    if (
        dataset.get("fingerprint") != identity.get("fingerprint")
        or Path(str(dataset.get("path", ""))).resolve() != Path(str(identity.get("path", ""))).resolve()
    ):
        raise CampaignAnalysisError("evaluation dataset identity differs from the frozen task")
    field_names = task.get("frozen_task_config", {}).get("dataset", {}).get("field_names")
    if not isinstance(field_names, list) or dataset.get("field_names") != field_names:
        raise CampaignAnalysisError("evaluation field order differs from the frozen task")

    checkpoint = payload.get("checkpoint")
    checkpoint_path = Path(str(payload.get("checkpoint_path", ""))).resolve()
    expected_checkpoint_root = (run_dir / "checkpoints").resolve()
    if not isinstance(checkpoint, str) or checkpoint_path.parent != expected_checkpoint_root:
        raise CampaignAnalysisError("evaluation checkpoint path escapes its child run")
    checkpoint_sha = payload.get("checkpoint_sha256")
    checkpoint_hashes = manifest.get("checkpoint_hashes", {})
    if not isinstance(checkpoint_hashes, Mapping) or checkpoint_hashes.get(checkpoint) != checkpoint_sha:
        raise CampaignAnalysisError("evaluation checkpoint hash is not attested by the run manifest")
    if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64:
        raise CampaignAnalysisError("evaluation checkpoint SHA-256 is invalid")

    inference = payload.get("inference_contract")
    if not isinstance(inference, Mapping) or inference.get("weight_identity") != "raw_live_model_weights":
        raise CampaignAnalysisError("quality evidence must use raw live model weights")
    if inference.get("model_ema_eval") is not False or inference.get("rollout_steps") != 2 or inference.get(
        "rollout_solver"
    ) != "euler":
        raise CampaignAnalysisError("evaluation inference differs from the frozen two-step raw-weight contract")
    selection = payload.get("selection")
    evaluation = payload.get("evaluation")
    if not isinstance(selection, Mapping) or not isinstance(evaluation, Mapping):
        raise CampaignAnalysisError("evaluation selection or metric record is missing")
    if selection.get("split") != "validation":
        raise CampaignAnalysisError("evaluation selection must explicitly be validation")
    panel = selection.get("panel")
    if not isinstance(panel, Mapping) or panel.get("kind") not in {"uniform", "disjoint"}:
        raise CampaignAnalysisError("evaluation panel identity is missing")
    panel_kind = str(panel["kind"])
    if panel.get("reference_count") != FIXED_SAMPLES:
        raise CampaignAnalysisError("evaluation panel must contain exactly 128 snapshots")
    if panel_kind == "uniform" and (
        panel.get("algorithm") != EXPECTED_PANEL_ALGORITHM
        or panel.get("excluded_validation_indices") != []
    ):
        raise CampaignAnalysisError("uniform validation panel metadata differs from its frozen selector")

    if (
        evaluation.get("samples") != FIXED_SAMPLES
        or evaluation.get("panel_kind") != panel_kind
        or evaluation.get("topology_scalar_filtrations") != TOPOLOGY_FILTRATIONS
        or selection.get("group_size") != GROUP_SIZE
        or selection.get("group_count") != FIXED_SAMPLES // GROUP_SIZE
        or selection.get("comparison_query_count") != QUERY_COUNT
    ):
        raise CampaignAnalysisError("evaluation does not satisfy the fixed 128/4x32/4096 full descriptor contract")
    query_shape = selection.get("query_indices_shape")
    if not isinstance(query_shape, list) or not query_shape or query_shape[-1] != QUERY_COUNT:
        raise CampaignAnalysisError("evaluation query tensor is not the fixed 4096-point geometry")
    indices = selection.get("validation_indices")
    sample_ids = selection.get("sample_ids")
    groups = payload.get("groups")
    if (
        not isinstance(indices, list)
        or len(indices) != FIXED_SAMPLES
        or len(set(indices)) != FIXED_SAMPLES
        or not isinstance(sample_ids, list)
        or len(sample_ids) != FIXED_SAMPLES
        or len(set(sample_ids)) != FIXED_SAMPLES
        or not isinstance(groups, list)
        or len(groups) != 4
    ):
        raise CampaignAnalysisError("evaluation does not contain exactly four unique 32-snapshot groups")
    listed_groups = selection.get("groups")
    if not isinstance(listed_groups, list) or len(listed_groups) != 4:
        raise CampaignAnalysisError("evaluation selection is missing the four fixed validation groups")
    concatenated_ids: list[str] = []
    for index, (group, listed_indices) in enumerate(zip(groups, listed_groups)):
        if not isinstance(group, Mapping) or group.get("group") != index:
            raise CampaignAnalysisError("evaluation groups must be ordered 0 through 3")
        group_ids = group.get("sample_ids")
        group_indices = group.get("validation_indices")
        if (
            not isinstance(group_ids, list)
            or len(group_ids) != GROUP_SIZE
            or len(set(group_ids)) != GROUP_SIZE
            or not isinstance(group_indices, list)
            or len(group_indices) != GROUP_SIZE
        ):
            raise CampaignAnalysisError("every evaluation group must contain exactly 32 sample IDs")
        if group_indices != listed_indices:
            raise CampaignAnalysisError("paired group indices differ from the frozen selection manifest")
        if group_indices != indices[index * GROUP_SIZE : (index + 1) * GROUP_SIZE]:
            raise CampaignAnalysisError("four groups do not reproduce the selected 128-index panel")
        source_group = group.get("source")
        child_group = group.get("child")
        if not isinstance(source_group, Mapping) or not isinstance(child_group, Mapping):
            raise CampaignAnalysisError("each validation group must have SOURCE and child metrics")
        source_topology = _topology_metrics(source_group)
        child_topology = _topology_metrics(child_group)
        if source_topology.get("topology_scalar_filtrations") != TOPOLOGY_FILTRATIONS or child_topology.get(
            "topology_scalar_filtrations"
        ) != TOPOLOGY_FILTRATIONS:
            raise CampaignAnalysisError("each paired group must use six complete topology filtrations")
        concatenated_ids.extend(str(value) for value in group_ids)
    if concatenated_ids != [str(value) for value in sample_ids]:
        raise CampaignAnalysisError("group sample IDs do not reproduce the fixed panel order")
    epoch = _evaluation_epoch(payload)
    if panel_kind == "disjoint":
        if panel.get("algorithm") != "chronological_linspace_after_uniform_exclusion_v1":
            raise CampaignAnalysisError("disjoint validation panel metadata differs from its frozen selector")
        excluded = panel.get("excluded_validation_indices")
        if (
            not isinstance(excluded, list)
            or len(excluded) != FIXED_SAMPLES
            or len(set(excluded)) != FIXED_SAMPLES
            or set(excluded).intersection(indices)
        ):
            raise CampaignAnalysisError("disjoint validation panel is not separate from the uniform panel")

    raw: dict[str, Any] = {
        "families": {},
        "b_groups": [],
        "endpoint_fields": {},
    }
    coherence_child: list[float] = []
    coherence_source: list[float] = []
    family_child: dict[str, list[float]] = {"A": [], "T": []}
    family_source: dict[str, list[float]] = {"A": [], "T": []}
    endpoint_child: dict[str, list[float]] = {name: [] for name in field_names}
    endpoint_source: dict[str, list[float]] = {name: [] for name in field_names}
    for group_index, group in enumerate(groups):
        source_metrics = group["source"]
        child_metrics = group["child"]
        source_coherence = _group_measurement(group, "source", "coherence total", "coherence", "total")
        child_coherence = _group_measurement(group, "child", "coherence total", "coherence", "total")
        coherence_source.append(source_coherence)
        coherence_child.append(child_coherence)
        for family, source_name in (("A", "global_distribution"), ("T", "topology")):
            source_value = _group_measurement(
                group, "source", f"{family} weighted total", "coherence", "families", source_name, "weighted_total"
            )
            child_value = _group_measurement(
                group, "child", f"{family} weighted total", "coherence", "families", source_name, "weighted_total"
            )
            family_source[family].append(source_value)
            family_child[family].append(child_value)
        source_b = _group_measurement(
            group, "source", "B(32) weighted total", "coherence", "families", "cross_spectrum", "weighted_total"
        )
        child_b = _group_measurement(
            group, "child", "B(32) weighted total", "coherence", "families", "cross_spectrum", "weighted_total"
        )
        raw["b_groups"].append(
            {
                "group_id": f"group_{group_index}",
                "snapshot_count": GROUP_SIZE,
                "measurement": _measurement_mapping(child_b, source_b, 1.0, "B(32)"),
            }
        )
        source_endpoint = source_metrics.get("endpoint_mse_model_units")
        child_endpoint = child_metrics.get("endpoint_mse_model_units")
        if (
            not isinstance(source_endpoint, Mapping)
            or not isinstance(child_endpoint, Mapping)
            or set(source_endpoint) != set(field_names)
            or set(child_endpoint) != set(field_names)
        ):
            raise CampaignAnalysisError("endpoint MSE fields differ from the frozen task field order")
        for name in field_names:
            endpoint_source[name].append(_number(source_endpoint[name], f"SOURCE endpoint MSE {name}", positive=True))
            endpoint_child[name].append(_number(child_endpoint[name], f"child endpoint MSE {name}"))

    raw["coherence_sum"] = _measurement_mapping(
        sum(coherence_child) / 4,
        sum(coherence_source) / 4,
        FIXED_SAMPLES,
        "coherence sum",
    )
    for family in ("A", "T"):
        raw["families"][family] = _measurement_mapping(
            sum(family_child[family]) / 4,
            sum(family_source[family]) / 4,
            FIXED_SAMPLES,
            f"family {family}",
        )
    for name in field_names:
        count = float(GROUP_SIZE * QUERY_COUNT * 4)
        raw["endpoint_fields"][name] = _measurement_mapping(
            sum(endpoint_child[name]) / 4,
            sum(endpoint_source[name]) / 4,
            count,
            f"endpoint field {name}",
        )
    raw["endpoint_aggregate"] = {
        key: sum(raw["endpoint_fields"][name][key] for name in field_names)
        for key in ("numerator", "count", "source_numerator", "source_count")
    }
    raw["numerator_origin"] = "reconstructed_from_serialized_group_means_and_frozen_counts"
    raw["fixed_counts"] = {
        "coherence_family_and_sum_snapshots": FIXED_SAMPLES,
        "B_groups": 4,
        "B_snapshots_per_group": GROUP_SIZE,
        "endpoint_queries_per_snapshot": QUERY_COUNT,
    }
    raw["paired_group_components"] = _paired_group_components(payload)
    return panel_kind, epoch, raw


def _contract(task: Mapping[str, Any], quality: Mapping[str, Any]):
    if str(quality.get("window")) != "e>=60 and (e-40,e]":
        raise CampaignAnalysisError("campaign quality window differs from the assessor contract")
    if quality.get("TEST") != "forbidden":
        raise CampaignAnalysisError("frozen quality contract does not forbid TEST access")
    fields = task.get("frozen_task_config", {}).get("dataset", {}).get("field_names")
    if not isinstance(fields, list) or not fields:
        raise CampaignAnalysisError("frozen task has no field-name contract")
    QualityContract = _quality_history_module().QualityContract

    monitors = quality.get("monitor_epochs", [])
    cadence = (
        monitors[1] - monitors[0]
        if isinstance(monitors, list) and len(monitors) >= 2
        else 10
    )
    return QualityContract(
        field_names=tuple(str(name) for name in fields),
        coherence_sum_limit=float(quality["additive_ratio_max"]),
        family_limits={
            "A": float(quality["family_ratio_max"]),
            "B": float(quality["family_ratio_max"]),
            "T": float(quality["family_ratio_max"]),
        },
        endpoint_aggregate_limit=float(quality["aggregate_endpoint_ratio_max"]),
        endpoint_field_limit=float(quality["each_field_ratio_max"]),
        mature_start_epoch=60,
        window_epochs=40,
        check_interval_epochs=int(cadence),
        maximum_epochs=int(task.get("horizon", 100)),
    )


def _read_history_artifacts(
    root: Path,
    task_id: str,
    arm: str,
    run_dir: Path,
    *,
    task: Mapping[str, Any],
    manifest: Mapping[str, Any],
    train_start: datetime | None,
    attempts: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Any]]:
    artifact_root = root / "analysis" / task_id / arm / run_dir.name
    if not artifact_root.is_dir() or artifact_root.is_symlink():
        return [], [], []
    uniform_records: list[dict[str, Any]] = []
    supplementary: list[dict[str, Any]] = []
    quality_checks = []
    for path in sorted(artifact_root.rglob("*.json")):
        if path.is_symlink() or not path.resolve().is_relative_to(artifact_root.resolve()):
            continue
        try:
            payload = _json_object(path, "evaluation artifact")
            panel_kind, epoch, raw = _validate_evaluation(
                payload, task=task, run_dir=run_dir, manifest=manifest
            )
        except (CampaignAnalysisError, KeyError, TypeError, ValueError) as error:
            uniform_records.append(
                {"path": str(path), "status": "rejected", "reason": str(error)}
            )
            continue
        ledger_segments = _evaluation_attempts(attempts, path.parent)
        finished = next(
            (
                _utc(segment.get("finished_utc"))
                for segment in reversed(ledger_segments)
                if _utc(segment.get("finished_utc")) is not None
            ),
            None,
        )
        artifact_time = _utc(payload.get("evaluated_at_utc"))
        completion = finished or artifact_time
        completion_basis = "controller_child_exit" if finished is not None else "artifact_timestamp_only"
        completion_seconds = _relative_seconds(completion, train_start)
        group_components = _paired_group_components(payload)
        record = {
            "path": str(path),
            "sha256": _sha256(path),
            "status": "supplementary_disjoint_panel" if panel_kind == "disjoint" else "valid_metrics",
            "panel_kind": panel_kind,
            "epoch": epoch,
            "checkpoint": payload.get("checkpoint"),
            "checkpoint_sha256": payload.get("checkpoint_sha256"),
            "sensor_query_manifest_sha256": payload.get("selection", {}).get(
                "sensor_query_manifest_sha256"
            ),
            "evaluated_at_utc": payload.get("evaluated_at_utc"),
            "completion_utc": completion.isoformat().replace("+00:00", "Z") if completion else None,
            "completion_time_basis": completion_basis,
            "completion_seconds_from_first_training_command": completion_seconds,
            "wall_seconds_from_first_training_command": completion_seconds,
            "active_process_seconds_from_first_training_command": None,
            "completion_seconds": None,
            "evaluation_ledger_segments": ledger_segments,
            "raw": raw,
            "ratios": _ratios(raw) if raw is not None else None,
            "paired_group_components": group_components,
        }
        if panel_kind == "disjoint":
            supplementary.append(record)
            continue
        if completion_seconds is None:
            record["status"] = "time_ambiguous"
        uniform_records.append(record)
    return uniform_records, supplementary, quality_checks


def _profile_acquisition(
    task_id: str,
    attempts: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], float | None, str]:
    segments = []
    seconds = 0.0
    complete = True
    for attempt_id, rows in attempts.items():
        if not rows or rows[0].get("task_id") != task_id or not any(
            row.get("action") == "profile" for row in rows
        ):
            continue
        summary = _attempt_summary(attempt_id, rows)
        segments.append(summary)
        charge = summary.get("charged_process_seconds")
        if summary.get("status") == "running" or not isinstance(charge, (int, float)):
            complete = False
        elif not isinstance(charge, bool) and math.isfinite(float(charge)) and charge >= 0:
            seconds += float(charge)
        else:
            complete = False
    if not segments:
        return [], None, "not_recorded"
    if not complete:
        return segments, None, "incomplete_profile_ledger"
    return segments, seconds, "observed"


def _profile_readiness_estimate(
    campaign_root: Path, task_id: str
) -> tuple[float, list[dict[str, Any]]]:
    """Read recorded conservative pre-admission estimates separately from measured time."""
    path = campaign_root / "analysis" / "readiness_attempts.jsonl"
    if not path.is_file():
        return 0.0, []
    estimates: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise CampaignAnalysisError(f"cannot read readiness attempts at {path}: {error}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise CampaignAnalysisError(
                f"invalid readiness attempt at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(row, Mapping):
            raise CampaignAnalysisError(f"readiness attempt at {path}:{line_number} is not an object")
        if row.get("phase") != f"{task_id} profile preflight":
            continue
        if row.get("outcome") != "rejected before GPU admission":
            continue
        gpu_seconds = row.get("gpu_process_seconds")
        optimizer_exposures = row.get("optimizer_exposures")
        command_seconds = row.get("conservative_command_seconds")
        if (
            isinstance(gpu_seconds, bool)
            or not isinstance(gpu_seconds, (int, float))
            or gpu_seconds != 0
            or isinstance(optimizer_exposures, bool)
            or not isinstance(optimizer_exposures, (int, float))
            or optimizer_exposures != 0
            or isinstance(command_seconds, bool)
            or not isinstance(command_seconds, (int, float))
            or not math.isfinite(float(command_seconds))
            or command_seconds < 0
        ):
            raise CampaignAnalysisError(
                f"unbounded or exposed pre-admission profile attempt at {path}:{line_number}"
            )
        estimates.append(
            {
                "path": str(path),
                "line": line_number,
                "phase": row["phase"],
                "outcome": row["outcome"],
                "conservative_command_seconds": float(command_seconds),
                "basis": "recorded conservative pre-admission command estimate; zero GPU time and optimizer exposures; not a proven timing bound",
            }
        )
    return sum(item["conservative_command_seconds"] for item in estimates), estimates


def _allocated_profile_seconds(arm: str, task_profile_seconds: float | None) -> float | None:
    """Charge strategy-plan acquisition only to the arm that requires it."""
    if arm != "strategy":
        return 0.0
    return task_profile_seconds


def _native_metric_identity(
    metrics: Mapping[str, Any],
    *,
    fields: Sequence[str],
    expected: Mapping[str, Any] | None = None,
) -> None:
    scope = metrics.get("fixed_estimator_scope")
    if not isinstance(scope, Mapping) or (
        scope.get("sample_count") != FIXED_SAMPLES
        or scope.get("group_size") != GROUP_SIZE
        or scope.get("groups") != 4
    ):
        raise CampaignAnalysisError("native history does not use the fixed 128/4x32 estimator")
    if metrics.get("generation_steps") != 2.0:
        raise CampaignAnalysisError("native history differs from the two-step inference contract")
    if metrics.get("evaluation_weight_source") != "configured_evaluation_weights":
        raise CampaignAnalysisError("native history does not identify configured raw evaluation weights")
    sensor_digest = metrics.get("sensor_manifest_sha256")
    if not isinstance(sensor_digest, str) or len(sensor_digest) != 64:
        raise CampaignAnalysisError("native history lacks its fixed sensor-manifest digest")
    sample_ids = metrics.get("sample_ids")
    query_ids = metrics.get("query_ids")
    groups = metrics.get("fixed_estimator_groups")
    if (
        not isinstance(sample_ids, list)
        or len(sample_ids) != FIXED_SAMPLES
        or len(set(sample_ids)) != FIXED_SAMPLES
        or not isinstance(query_ids, list)
        or len(query_ids) != FIXED_SAMPLES
        or any(not isinstance(row, list) or len(row) != QUERY_COUNT for row in query_ids)
        or not isinstance(groups, list)
        or len(groups) != 4
    ):
        raise CampaignAnalysisError("native history lacks the exact selected sample/query panel")
    if expected is not None:
        for key in ("sample_ids", "query_ids", "sensor_manifest_sha256", "generation_steps", "evaluation_weight_source"):
            if metrics.get(key) != expected.get(key):
                raise CampaignAnalysisError(f"native history SOURCE pairing differs at {key}")
    flattened_samples: list[str] = []
    flattened_queries: list[Any] = []
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise CampaignAnalysisError("native fixed-estimator group is malformed")
        group_samples = group.get("sample_ids")
        group_queries = group.get("query_ids")
        if (
            not isinstance(group_samples, list)
            or len(group_samples) != GROUP_SIZE
            or not isinstance(group_queries, list)
            or len(group_queries) != GROUP_SIZE
            or any(not isinstance(row, list) or len(row) != QUERY_COUNT for row in group_queries)
        ):
            raise CampaignAnalysisError("native history group does not contain 32x4096 comparisons")
        if group.get("generation_seed") is None or group.get("generation_steps") != 2.0:
            raise CampaignAnalysisError("native history group lacks its fixed seed/two-step identity")
        if group.get("evaluation_weight_source") != "configured_evaluation_weights":
            raise CampaignAnalysisError("native history group uses an unpinned evaluation weight source")
        if expected is not None:
            reference_group = expected["fixed_estimator_groups"][index]
            for key in ("sample_ids", "query_ids", "generation_seed", "generation_steps", "evaluation_weight_source"):
                if group.get(key) != reference_group.get(key):
                    raise CampaignAnalysisError(
                        f"native group {index} SOURCE pairing differs at {key}"
                    )
        flattened_samples.extend(str(value) for value in group_samples)
        flattened_queries.extend(group_queries)
        if not isinstance(group.get("coherence"), Mapping):
            raise CampaignAnalysisError("native group has no coherence metrics")
        if _topology_metrics(group).get("topology_scalar_filtrations") != TOPOLOGY_FILTRATIONS:
            raise CampaignAnalysisError("native group did not use all six topology filtrations")
        endpoints = group.get("endpoint_mse_model_units")
        if not isinstance(endpoints, Mapping) or set(endpoints) != set(fields):
            raise CampaignAnalysisError("native group endpoint fields differ from the task contract")
    if flattened_samples != [str(value) for value in sample_ids] or flattened_queries != query_ids:
        raise CampaignAnalysisError("native four groups do not reproduce their top-level fixed panel")


def _native_raw_metrics(
    child_metrics: Mapping[str, Any],
    source_metrics: Mapping[str, Any],
    *,
    fields: Sequence[str],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    _native_metric_identity(source_metrics, fields=fields)
    _native_metric_identity(child_metrics, fields=fields, expected=baseline)
    raw: dict[str, Any] = {"families": {}, "b_groups": [], "endpoint_fields": {}}
    aggregate_child: dict[str, list[float]] = {"coherence": [], "A": [], "T": []}
    aggregate_source: dict[str, list[float]] = {"coherence": [], "A": [], "T": []}
    endpoints_child: dict[str, list[float]] = {name: [] for name in fields}
    endpoints_source: dict[str, list[float]] = {name: [] for name in fields}
    source_groups = source_metrics["fixed_estimator_groups"]
    child_groups = child_metrics["fixed_estimator_groups"]
    if len(source_groups) != 4 or len(child_groups) != 4:
        raise CampaignAnalysisError("native source/child history must contain four exact groups")
    for index, (source_group, child_group) in enumerate(zip(source_groups, child_groups)):
        for family, family_key in (("A", "global_distribution"), ("B", "cross_spectrum"), ("T", "topology")):
            source_family = source_group["coherence"]["families"][family_key]
            child_family = child_group["coherence"]["families"][family_key]
            source_value = _number(source_family.get("weighted_total"), f"SOURCE {family} group {index}", positive=True)
            child_value = _number(child_family.get("weighted_total"), f"child {family} group {index}")
            if family == "B":
                raw["b_groups"].append(
                    {
                        "group_id": f"group_{index}",
                        "snapshot_count": GROUP_SIZE,
                        "measurement": _measurement_mapping(child_value, source_value, 1.0, "B(32)"),
                    }
                )
            else:
                aggregate_child[family].append(child_value)
                aggregate_source[family].append(source_value)
        aggregate_source["coherence"].append(
            _number(source_group["coherence"].get("total"), f"SOURCE coherence group {index}", positive=True)
        )
        aggregate_child["coherence"].append(
            _number(child_group["coherence"].get("total"), f"child coherence group {index}")
        )
        for field_name in fields:
            endpoints_source[field_name].append(
                _number(source_group["endpoint_mse_model_units"][field_name], f"SOURCE endpoint {field_name}", positive=True)
            )
            endpoints_child[field_name].append(
                _number(child_group["endpoint_mse_model_units"][field_name], f"child endpoint {field_name}")
            )
    raw["coherence_sum"] = _measurement_mapping(
        sum(aggregate_child["coherence"]) / 4,
        sum(aggregate_source["coherence"]) / 4,
        FIXED_SAMPLES,
        "native coherence sum",
    )
    for family in ("A", "T"):
        raw["families"][family] = _measurement_mapping(
            sum(aggregate_child[family]) / 4,
            sum(aggregate_source[family]) / 4,
            FIXED_SAMPLES,
            f"native family {family}",
        )
    for field_name in fields:
        count = float(GROUP_SIZE * QUERY_COUNT * 4)
        raw["endpoint_fields"][field_name] = _measurement_mapping(
            sum(endpoints_child[field_name]) / 4,
            sum(endpoints_source[field_name]) / 4,
            count,
            f"native endpoint {field_name}",
        )
    raw["endpoint_aggregate"] = {
        key: sum(raw["endpoint_fields"][field_name][key] for field_name in fields)
        for key in ("numerator", "count", "source_numerator", "source_count")
    }
    raw["numerator_origin"] = "reconstructed_from_native_group_means_and_fixed_counts"
    raw["fixed_counts"] = {
        "coherence_family_and_sum_snapshots": FIXED_SAMPLES,
        "B_groups": 4,
        "B_snapshots_per_group": GROUP_SIZE,
        "endpoint_queries_per_snapshot": QUERY_COUNT,
    }
    raw["paired_group_components"] = [
        {
            "group": index,
            "snapshot_count": GROUP_SIZE,
            "source": {
                "coherence": source_groups[index]["coherence"],
                "endpoint_mse_model_units": source_groups[index]["endpoint_mse_model_units"],
            },
            "child": {
                "coherence": child_groups[index]["coherence"],
                "endpoint_mse_model_units": child_groups[index]["endpoint_mse_model_units"],
            },
        }
        for index in range(4)
    ]
    raw["source_baseline"] = {
        "path_role": "run/evaluation/before.json",
        "sensor_manifest_sha256": source_metrics.get("sensor_manifest_sha256"),
    }
    return raw


def _native_row_projection(metrics: Mapping[str, Any]) -> dict[str, Any]:
    groups = metrics.get("fixed_estimator_groups", [])
    return {
        "sample_ids": metrics.get("sample_ids"),
        "query_ids": metrics.get("query_ids"),
        "sensor_manifest_sha256": metrics.get("sensor_manifest_sha256"),
        "groups": [
            {
                "sample_ids": group.get("sample_ids"),
                "query_ids": group.get("query_ids"),
                "generation_seed": group.get("generation_seed"),
                "generation_steps": group.get("generation_steps"),
                "evaluation_weight_source": group.get("evaluation_weight_source"),
                "coherence_total": group.get("coherence", {}).get("total"),
                "family_weighted_totals": {
                    name: group.get("coherence", {}).get("families", {}).get(name, {}).get("weighted_total")
                    for name in ("global_distribution", "cross_spectrum", "topology")
                },
                "endpoint_mse_model_units": group.get("endpoint_mse_model_units"),
            }
            for group in groups
        ],
    }


def _native_metric_difference(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    left = _native_row_projection(first)
    right = _native_row_projection(second)
    identity_match = all(left[key] == right[key] for key in ("sample_ids", "query_ids", "sensor_manifest_sha256"))
    identity_match = identity_match and all(
        all(left_group[key] == right_group[key] for key in (
            "sample_ids", "query_ids", "generation_seed", "generation_steps", "evaluation_weight_source"
        ))
        for left_group, right_group in zip(left["groups"], right["groups"])
    ) and len(left["groups"]) == len(right["groups"])
    metric_match = identity_match
    for left_group, right_group in zip(left["groups"], right["groups"]):
        pairs = [(left_group["coherence_total"], right_group["coherence_total"])]
        pairs.extend(
            (left_group["family_weighted_totals"].get(name), right_group["family_weighted_totals"].get(name))
            for name in ("global_distribution", "cross_spectrum", "topology")
        )
        pairs.extend(
            (left_group["endpoint_mse_model_units"].get(name), right_group["endpoint_mse_model_units"].get(name))
            for name in left_group["endpoint_mse_model_units"]
        )
        if any(
            not isinstance(a, (int, float))
            or not isinstance(b, (int, float))
            or not math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-12)
            for a, b in pairs
        ):
            metric_match = False
            break
    return {
        "identity_match": bool(identity_match),
        "metric_match_within_1e-6_relative_1e-12_absolute": bool(metric_match),
    }


def _attempt_for_completion(
    completion: datetime | None, attempts: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    if completion is None:
        return None
    candidates = []
    for attempt in attempts:
        started = _utc(attempt.get("started_utc"))
        finished = _utc(attempt.get("finished_utc"))
        if started is not None and started <= completion and (finished is None or completion <= finished):
            candidates.append((started, attempt))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _native_history(
    run_dir: Path,
    *,
    task: Mapping[str, Any],
    task_id: str,
    arm: str,
    attempts: Mapping[str, Sequence[Mapping[str, Any]]],
    training_attempts: Sequence[Mapping[str, Any]],
    train_start: datetime | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[Any]]:
    """Pair scheduled child-history rows to the run's fixed SOURCE baseline."""
    before_path = run_dir / "evaluation" / "before.json"
    history_path = run_dir / "metrics" / "topology_validation.jsonl"
    if not before_path.is_file() or before_path.is_symlink():
        return None, [], []
    baseline = _json_object(before_path, "paired SOURCE baseline")
    fields = task.get("frozen_task_config", {}).get("dataset", {}).get("field_names")
    eval_config = task.get("frozen_task_config", {}).get("evaluation", {})
    if (
        not isinstance(fields, list)
        or not isinstance(eval_config, Mapping)
        or eval_config.get("split") != "validation"
        or eval_config.get("max_samples") != FIXED_SAMPLES
        or eval_config.get("group_size") != GROUP_SIZE
        or eval_config.get("query_points") != QUERY_COUNT
        or eval_config.get("generation_steps") != 2
        or eval_config.get("sample_selection") != "uniform"
    ):
        raise CampaignAnalysisError("frozen task does not declare the native validation panel contract")
    _native_metric_identity(baseline, fields=fields)
    baseline_digest = _sha256(before_path)
    steps_per_epoch = None
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file() and not manifest_path.is_symlink():
        steps_per_epoch = _json_object(manifest_path, "campaign run manifest").get("steps_per_epoch")
    if isinstance(steps_per_epoch, bool) or not isinstance(steps_per_epoch, int) or steps_per_epoch <= 0:
        raise CampaignAnalysisError("run manifest lacks a valid steps_per_epoch for native monitor rows")

    rows: list[tuple[int, Mapping[str, Any], str, int]] = []
    invalid_rows: list[dict[str, Any]] = []
    if history_path.is_file() and not history_path.is_symlink():
        with history_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, Mapping):
                        raise CampaignAnalysisError("native history row must be an object")
                    step = row.get("step")
                    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                        raise CampaignAnalysisError("native history step must be a nonnegative integer")
                    if step % steps_per_epoch:
                        raise CampaignAnalysisError("native history step is not an integer epoch")
                    epoch = step // steps_per_epoch
                    if epoch > int(task.get("horizon", 100)) or epoch % 10:
                        raise CampaignAnalysisError("native history row is outside the ten-epoch schedule")
                    metrics = row.get("metrics")
                    if not isinstance(metrics, Mapping):
                        raise CampaignAnalysisError("native history row lacks fixed estimator metrics")
                    if epoch > 0:
                        _native_metric_identity(metrics, fields=fields, expected=baseline)
                        raw = _native_raw_metrics(metrics, baseline, fields=fields, baseline=baseline)
                    rows.append((epoch, row, hashlib.sha256(line.encode("utf-8")).hexdigest(), line_number))
                except (CampaignAnalysisError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    invalid_rows.append({"line": line_number, "status": "rejected", "reason": str(error)})

    by_epoch: dict[int, list[tuple[Mapping[str, Any], str, int]]] = defaultdict(list)
    for epoch, row, digest, line_number in rows:
        by_epoch[epoch].append((row, digest, line_number))
    if 0 in by_epoch:
        first_zero = by_epoch[0][0][0].get("metrics", {})
        baseline_record = {
            "epoch": 0,
            "status": "baseline_only",
            "path": str(before_path),
            "sha256": baseline_digest,
            "native_step_zero_row_sha256": by_epoch[0][0][1],
            "sensor_manifest_sha256": baseline.get("sensor_manifest_sha256"),
            "sample_count": FIXED_SAMPLES,
            "group_size": GROUP_SIZE,
            "group_count": 4,
            "source_baseline_coherence_total": baseline.get("coherence", {}).get("total"),
            "step_zero_child_coherence_total": first_zero.get("coherence", {}).get("total")
            if isinstance(first_zero, Mapping)
            else None,
        }
    else:
        baseline_record = {
            "epoch": 0,
            "status": "source_baseline_only",
            "path": str(before_path),
            "sha256": baseline_digest,
            "sensor_manifest_sha256": baseline.get("sensor_manifest_sha256"),
            "sample_count": FIXED_SAMPLES,
            "group_size": GROUP_SIZE,
            "group_count": 4,
        }

    native_records: list[dict[str, Any]] = []
    checks = []
    profile_segments, task_profile_seconds_measured, profile_status_observed = _profile_acquisition(
        task_id, attempts
    )
    campaign_root = run_dir.resolve().parents[3]
    task_profile_seconds_estimated, profile_estimates = _profile_readiness_estimate(
        campaign_root, task_id
    )
    task_profile_seconds = (
        task_profile_seconds_measured + task_profile_seconds_estimated
        if task_profile_seconds_measured is not None
        else None
    )
    allocated_profile_seconds = _allocated_profile_seconds(arm, task_profile_seconds)
    allocated_profile_seconds_measured = _allocated_profile_seconds(
        arm, task_profile_seconds_measured
    )
    allocated_profile_seconds_estimated = _allocated_profile_seconds(
        arm, task_profile_seconds_estimated
    )
    for epoch in sorted(e for e in by_epoch if e > 0):
        epoch_rows = by_epoch[epoch]
        primary_row, primary_digest, primary_line = epoch_rows[0]
        metrics = primary_row["metrics"]
        raw = _native_raw_metrics(metrics, baseline, fields=fields, baseline=baseline)
        evaluated_at = _utc(metrics.get("evaluation_completed_utc"))
        command_elapsed = metrics.get("command_elapsed_seconds")
        attempt = _attempt_for_completion(evaluated_at, training_attempts)
        ambiguous_reason = None
        active_seconds = None
        if evaluated_at is None:
            ambiguous_reason = "native evaluation completion UTC is missing"
        elif isinstance(command_elapsed, bool) or not isinstance(command_elapsed, (int, float)) or not math.isfinite(float(command_elapsed)) or command_elapsed < 0:
            ambiguous_reason = "native evaluation command elapsed seconds are missing or invalid"
        elif attempt is None:
            ambiguous_reason = "no owned launch/resume ledger segment contains the native evaluation"
        else:
            attempt_start = _utc(attempt.get("started_utc"))
            prior_seconds = 0.0
            for previous in training_attempts:
                previous_start = _utc(previous.get("started_utc"))
                if previous_start is None or attempt_start is None or previous_start >= attempt_start:
                    continue
                charge = previous.get("charged_process_seconds")
                if isinstance(charge, bool) or not isinstance(charge, (int, float)) or not math.isfinite(float(charge)):
                    ambiguous_reason = "a preceding launch/resume has no terminal process charge"
                    break
                prior_seconds += float(charge)
            if ambiguous_reason is None:
                active_seconds = prior_seconds + float(command_elapsed)
                if allocated_profile_seconds is None:
                    ambiguous_reason = f"profile acquisition timing is {profile_status_observed}"
        cold_seconds = (
            allocated_profile_seconds + active_seconds
            if active_seconds is not None and allocated_profile_seconds is not None
            else None
        )
        wall_seconds = _relative_seconds(evaluated_at, train_start)
        record = {
            "epoch": epoch,
            "step": primary_row["step"],
            "panel_kind": "uniform",
            "measurement_source": "native_topology_history",
            "status": "native_primary" if cold_seconds is not None else "time_ambiguous",
            "primary_history_line": primary_line,
            "native_metric_line_sha256": primary_digest,
            "baseline_path": str(before_path),
            "baseline_sha256": baseline_digest,
            "sensor_manifest_sha256": metrics.get("sensor_manifest_sha256"),
            "evaluation_completed_utc": metrics.get("evaluation_completed_utc"),
            "completion_utc": metrics.get("evaluation_completed_utc"),
            "command_elapsed_seconds": float(command_elapsed) if isinstance(command_elapsed, (int, float)) and not isinstance(command_elapsed, bool) else None,
            "training_attempt_id": attempt.get("attempt_id") if attempt else None,
            "completion_time_basis": "allocated_profile_process_charges_plus_prior_training_process_charges_plus_native_command_elapsed",
            "active_process_seconds_from_first_training_command": active_seconds,
            "wall_seconds_from_first_training_command": wall_seconds,
            "completion_seconds_from_first_training_command": active_seconds,
            "task_profile_acquisition_seconds": task_profile_seconds,
            "task_profile_acquisition_seconds_measured": task_profile_seconds_measured,
            "task_profile_acquisition_seconds_estimated": task_profile_seconds_estimated,
            "allocated_profile_acquisition_seconds": allocated_profile_seconds,
            "allocated_profile_acquisition_seconds_measured": allocated_profile_seconds_measured,
            "allocated_profile_acquisition_seconds_estimated": allocated_profile_seconds_estimated,
            "task_profile_acquisition_process_seconds": task_profile_seconds,
            "allocated_profile_acquisition_process_seconds": allocated_profile_seconds,
            "cold_process_seconds_including_profile_acquisition": cold_seconds,
            "completion_seconds": cold_seconds,
            "time_ambiguity_reason": ambiguous_reason,
            "raw": raw,
            "ratios": _ratios(raw),
            "paired_group_components": raw["paired_group_components"],
            "duplicate_native_rows": [],
            "duplicate_disagreement": False,
        }
        for duplicate_row, duplicate_digest, duplicate_line in epoch_rows[1:]:
            duplicate_metrics = duplicate_row["metrics"]
            comparison = _native_metric_difference(metrics, duplicate_metrics)
            duplicate = {
                "line": duplicate_line,
                "sha256": duplicate_digest,
                **comparison,
                "metrics_differ": not comparison["metric_match_within_1e-6_relative_1e-12_absolute"],
            }
            record["duplicate_native_rows"].append(duplicate)
            record["duplicate_disagreement"] = record["duplicate_disagreement"] or duplicate["metrics_differ"]
        if cold_seconds is not None:
            checks.append(
                _quality_history_module().quality_check_from_mapping(
                    {
                        "epoch": epoch,
                        "completion_seconds": cold_seconds,
                        "split": "validation",
                        "raw": raw,
                    }
                )
            )
        native_records.append(record)
    integrity = "ok"
    if invalid_rows:
        integrity = "invalid_history_rows"
    if any(record["duplicate_disagreement"] for record in native_records):
        integrity = "duplicate_metric_disagreement_first_row_retained"
    native_history = {
        "baseline": baseline_record,
        "path": str(history_path),
        "status": "available" if rows else ("rejected_rows" if invalid_rows else "pending_native_history"),
        "integrity_status": integrity,
        "invalid_rows": invalid_rows,
        "profile_acquisition_status": profile_status_observed,
        "task_profile_acquisition_seconds": task_profile_seconds,
        "task_profile_acquisition_seconds_measured": task_profile_seconds_measured,
        "task_profile_acquisition_seconds_estimated": task_profile_seconds_estimated,
        "allocated_profile_acquisition_seconds": allocated_profile_seconds,
        "allocated_profile_acquisition_seconds_measured": allocated_profile_seconds_measured,
        "allocated_profile_acquisition_seconds_estimated": allocated_profile_seconds_estimated,
        "task_profile_acquisition_process_seconds": task_profile_seconds,
        "allocated_profile_acquisition_process_seconds": allocated_profile_seconds,
        "profile_attempt_segments": profile_segments,
        "profile_estimate_records": profile_estimates,
    }
    return native_history, native_records, checks


def _accepted_end_epoch(
    manifest: Mapping[str, Any],
    status: Mapping[str, Any],
    training_attempts: Sequence[Mapping[str, Any]],
) -> tuple[int | None, str]:
    observed = []
    for attempt in training_attempts:
        observation = attempt.get("epoch_observation")
        if isinstance(observation, Mapping):
            value = observation.get("accepted_end_epoch")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                observed.append((float(value), "checkpoint_cursor_from_runner_ledger"))
    if observed:
        value, basis = max(observed, key=lambda item: item[0])
    else:
        value = manifest.get("last_checkpoint_epoch")
        basis = "run_manifest_last_checkpoint_epoch"
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None, "unknown"
    if not math.isclose(float(value), round(float(value)), rel_tol=0.0, abs_tol=1e-8):
        return None, "noninteger_checkpoint_epoch"
    epoch = round(float(value))
    if not 0 <= epoch <= 100:
        return None, "checkpoint_epoch_outside_horizon"
    return epoch, basis


def _run_record(
    root: Path,
    task_id: str,
    arm: str,
    run_dir: Path,
    *,
    task: Mapping[str, Any],
    quality: Mapping[str, Any],
    attempts: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return {"run_id": run_dir.name, "status": "pending_manifest"}, []
    manifest = _json_object(manifest_path, "campaign run manifest")
    if Path(str(manifest.get("parent_run", ""))).resolve() != Path(
        str(task.get("source_run", ""))
    ).resolve():
        return {
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "status": "rejected_source_run_mismatch",
            "quality_assessment_status": "not_assessed",
        }, []
    source_hashes = manifest.get("source_hashes")
    if not isinstance(source_hashes, Mapping) or source_hashes.get("checkpoint") != task.get(
        "source_checkpoint_sha256"
    ):
        return {
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "status": "rejected_source_checkpoint_mismatch",
            "quality_assessment_status": "not_assessed",
        }, []
    identity = task.get("dataset_identity")
    if not isinstance(identity, Mapping) or (
        manifest.get("dataset_fingerprint") != identity.get("fingerprint")
        or Path(str(manifest.get("dataset_path", ""))).resolve()
        != Path(str(identity.get("path", ""))).resolve()
    ):
        return {
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "status": "rejected_dataset_identity_mismatch",
            "quality_assessment_status": "not_assessed",
        }, []
    status_path = run_dir / "status.json"
    status = _json_object(status_path, "run status") if status_path.is_file() else {}
    training_attempts = _run_attempts(
        attempts,
        run_dir,
        task_id=task_id,
        arm=arm,
        manifest=manifest,
    )
    starts = [_utc(item.get("started_utc")) for item in training_attempts]
    starts = [item for item in starts if item is not None]
    train_start = min(starts) if starts else None
    finished = [_utc(item.get("finished_utc")) for item in training_attempts]
    finished = [item for item in finished if item is not None]
    if status.get("status") == "completed":
        status_finish = _utc(status.get("updated_utc"))
        if status_finish is not None:
            finished.append(status_finish)
    train_finish = max(finished) if finished else None
    actual_end, end_basis = _accepted_end_epoch(manifest, status, training_attempts)
    portable_uniform, supplementary, _unused_portable_checks = _read_history_artifacts(
        root,
        task_id,
        arm,
        run_dir,
        task=task,
        manifest=manifest,
        train_start=train_start,
        attempts=attempts,
    )
    native_history, native_records, checks = _native_history(
        run_dir,
        task=task,
        task_id=task_id,
        arm=arm,
        attempts=attempts,
        training_attempts=training_attempts,
        train_start=train_start,
    )
    quality_assessment: dict[str, Any] | None = None
    assessment_status = "pending_quality_checks"
    task_profile_seconds = (
        native_history.get("task_profile_acquisition_process_seconds")
        if native_history is not None
        else None
    )
    task_profile_seconds_measured = (
        native_history.get("task_profile_acquisition_seconds_measured")
        if native_history is not None
        else None
    )
    task_profile_seconds_estimated = (
        native_history.get("task_profile_acquisition_seconds_estimated", 0.0)
        if native_history is not None
        else None
    )
    allocated_profile_seconds = _allocated_profile_seconds(arm, task_profile_seconds)
    allocated_profile_seconds_measured = _allocated_profile_seconds(
        arm, task_profile_seconds_measured
    )
    allocated_profile_seconds_estimated = _allocated_profile_seconds(
        arm, task_profile_seconds_estimated
    )
    training_process_seconds = 0.0
    process_charge_complete = bool(training_attempts)
    for attempt in training_attempts:
        charge = attempt.get("charged_process_seconds")
        if isinstance(charge, bool) or not isinstance(charge, (int, float)) or not math.isfinite(float(charge)):
            process_charge_complete = False
            break
        training_process_seconds += float(charge)
    end_seconds = (
        allocated_profile_seconds + training_process_seconds
        if allocated_profile_seconds is not None and process_charge_complete
        else None
    )
    if checks and actual_end is not None and train_finish is not None and end_seconds is not None:
        integrity_status = (
            native_history.get("integrity_status")
            if isinstance(native_history, Mapping)
            else None
        )
        if integrity_status != "ok":
            assessment_status = (
                "ambiguous_native_history_integrity:"
                f"{integrity_status or 'unavailable'}"
            )
        else:
            try:
                if len({item.epoch for item in checks}) != len(checks):
                    raise CampaignAnalysisError("multiple uniform evaluations exist at one reporting epoch")
                quality_assessment = _quality_history_module().assess_quality_history(
                    checks,
                    actual_end_epoch=actual_end,
                    end_completion_seconds=end_seconds,
                    contract=_contract(task, quality),
                )
                assessment_status = "assessed"
            except (CampaignAnalysisError, ValueError, TypeError) as error:
                assessment_status = f"ambiguous: {error}"
    elif native_records or portable_uniform or supplementary:
        if actual_end is None:
            assessment_status = f"pending_actual_end:{end_basis}"
        elif train_finish is None or end_seconds is None:
            assessment_status = "pending_training_completion_or_profile_charge"
        elif not checks:
            assessment_status = "pending_controller_timed_native_checks"

    environment_path = run_dir / "environment.json"
    environment = (
        _json_object(environment_path, "run environment") if environment_path.is_file() else None
    )
    run_record = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "run_manifest": str(manifest_path),
        "status": status.get("status", "unknown"),
        "run_created_utc": manifest.get("created_utc"),
        "first_training_command_started_utc": (
            train_start.isoformat().replace("+00:00", "Z") if train_start else None
        ),
        "training_command_finished_utc": (
            train_finish.isoformat().replace("+00:00", "Z") if train_finish else None
        ),
        "training_elapsed_seconds_from_commands": _relative_seconds(train_finish, train_start),
        "training_active_process_seconds": training_process_seconds if process_charge_complete else None,
        "task_profile_acquisition_seconds": task_profile_seconds,
        "task_profile_acquisition_seconds_measured": task_profile_seconds_measured,
        "task_profile_acquisition_seconds_estimated": task_profile_seconds_estimated,
        "allocated_profile_acquisition_seconds": allocated_profile_seconds,
        "allocated_profile_acquisition_seconds_measured": allocated_profile_seconds_measured,
        "allocated_profile_acquisition_seconds_estimated": allocated_profile_seconds_estimated,
        "profile_estimate_records": (
            native_history.get("profile_estimate_records", [])
            if native_history is not None
            else []
        ),
        "task_profile_acquisition_process_seconds": task_profile_seconds,
        "allocated_profile_acquisition_process_seconds": allocated_profile_seconds,
        "cold_process_seconds_including_profile_acquisition": end_seconds,
        "accepted_end_epoch": actual_end,
        "accepted_end_epoch_basis": end_basis,
        "training_attempt_segments": training_attempts,
        "environment": environment,
        "metric_jsonl": _metric_history_summary(run_dir),
        "native_history": native_history,
        "uniform_evaluations": native_records,
        "portable_evaluations": portable_uniform,
        "supplementary_evaluations": supplementary,
        "quality_assessment_status": assessment_status,
        "quality_assessment": quality_assessment,
    }
    csv_rows = []
    all_evaluations = [*native_records, *portable_uniform, *supplementary]
    if not all_evaluations:
        csv_rows.append(
            {
                "task": task_id,
                "arm": arm,
                "run_id": run_dir.name,
                "epoch": "",
                "panel": "",
                "status": "pending_evaluation",
                "evaluation_path": "",
                "completion_utc": "",
                "completion_seconds": "",
                "task_profile_acquisition_seconds_measured": "",
                "task_profile_acquisition_seconds_estimated": "",
                "allocated_profile_acquisition_seconds_measured": "",
                "allocated_profile_acquisition_seconds_estimated": "",
                "checkpoint_sha256": "",
                "coherence_sum_ratio": "",
                "family_A_ratio": "",
                "family_B_ratio": "",
                "family_T_ratio": "",
                "endpoint_aggregate_ratio": "",
                "endpoint_field_ratios_json": "",
                "training_segments_json": json.dumps(training_attempts, sort_keys=True),
            }
        )
    for record in all_evaluations:
        ratio_summary = record.get("ratios") or {}
        ratios = {
            "coherence_sum_ratio": ratio_summary.get("coherence_sum", ""),
            "family_A_ratio": ratio_summary.get("family_A", ""),
            "family_B_ratio": ratio_summary.get("family_B", ""),
            "family_T_ratio": ratio_summary.get("family_T", ""),
            "endpoint_aggregate_ratio": ratio_summary.get("endpoint_aggregate", ""),
            "endpoint_field_ratios_json": json.dumps(
                ratio_summary.get("endpoint_fields", {}), sort_keys=True
            ),
        }
        csv_rows.append(
            {
                "task": task_id,
                "arm": arm,
                "run_id": run_dir.name,
                "epoch": record.get("epoch", ""),
                "panel": record.get("panel_kind", "native_uniform" if record.get("measurement_source") == "native_topology_history" else ""),
                "status": record.get("status", "unknown"),
                "evaluation_path": record.get("path", ""),
                "completion_utc": record.get("completion_utc", record.get("evaluation_completed_utc", "")),
                "completion_seconds": record.get("completion_seconds", record.get("completion_seconds_from_first_training_command", "")),
                "completion_time_basis": record.get("completion_time_basis", record.get("completion_time_basis", "")),
                "measurement_source": record.get("measurement_source", "portable_full_abt"),
                "wall_seconds_from_first_training_command": record.get("wall_seconds_from_first_training_command", ""),
                "active_process_seconds_from_first_training_command": record.get("active_process_seconds_from_first_training_command", ""),
                "task_profile_acquisition_seconds": record.get("task_profile_acquisition_seconds", ""),
                "task_profile_acquisition_seconds_measured": record.get("task_profile_acquisition_seconds_measured", ""),
                "task_profile_acquisition_seconds_estimated": record.get("task_profile_acquisition_seconds_estimated", ""),
                "allocated_profile_acquisition_seconds": record.get("allocated_profile_acquisition_seconds", ""),
                "allocated_profile_acquisition_seconds_measured": record.get("allocated_profile_acquisition_seconds_measured", ""),
                "allocated_profile_acquisition_seconds_estimated": record.get("allocated_profile_acquisition_seconds_estimated", ""),
                "checkpoint_sha256": record.get("checkpoint_sha256", ""),
                **{key: ratios.get(key, "") for key in (
                    "coherence_sum_ratio",
                    "family_A_ratio",
                    "family_B_ratio",
                    "family_T_ratio",
                    "endpoint_aggregate_ratio",
                    "endpoint_field_ratios_json",
                )},
                "training_segments_json": json.dumps(training_attempts, sort_keys=True),
            }
        )
    return run_record, csv_rows


def build_campaign_quality_history(root: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse existing campaign evidence without accessing model or dataset files."""
    campaign_root = Path(root).resolve(strict=True)
    inventory = _json_object(campaign_root / "tasks" / "task_inventory.json", "task inventory")
    tasks = inventory.get("tasks")
    quality = inventory.get("quality_contract")
    if not isinstance(tasks, Mapping) or not isinstance(quality, Mapping):
        raise CampaignAnalysisError("frozen inventory lacks task or quality contracts")
    attempts = _ledger_attempts(campaign_root)
    report: dict[str, Any] = {
        "schema": QUALITY_SCHEMA,
        "root": str(campaign_root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_scope": [
            "campaign task inventory and SOURCE identity metadata",
            "campaign runner ledger",
            "owned run manifests, status, environment, and native topology-validation JSONL",
            "run evaluation/before.json SOURCE baseline and paired validation evaluation JSON",
        ],
        "forbidden_access": ["TEST datasets", "TRAIN or validation dataset arrays", "configuration validators"],
        "numerator_reconstruction": (
            "Native and portable evaluators serialize paired group means, not raw per-entry accumulator sums. "
            "Numerators below are reconstructed from those means and the frozen snapshot/query counts; "
            "they are not exact stored raw sums."
        ),
        "completion_time_method": (
            "Native check time is prior owned launch/resume process charges plus that metric row's "
            "command_elapsed_seconds, plus task profile acquisition charges allocated once to the strategy arm. "
            "Profile acquisition separates measured admitted-ledger process charges from explicitly labeled "
            "conservative pre-admission command-time estimates; estimates are not measured GPU time. "
            "Dense and fixed-heuristic arms receive zero strategy-profile allocation, while the full task "
            "profile cost remains visible for campaign-global accounting. Per-arm task profile fields repeat "
            "the task-level value for traceability and must not be summed across arms. Wall seconds from the "
            "first training command are retained separately; waiting gaps do not count as process cost."
        ),
        "tasks": {},
    }
    csv_rows: list[dict[str, Any]] = []
    for task_id, task_value in tasks.items():
        if not isinstance(task_value, Mapping):
            raise CampaignAnalysisError(f"inventory task {task_id} is malformed")
        arms = task_value.get("arms", [])
        if not isinstance(arms, list):
            raise CampaignAnalysisError(f"inventory task {task_id} has no arm list")
        task_record: dict[str, Any] = {"case": task_value.get("case"), "arms": {}}
        for arm in arms:
            arm_root = campaign_root / "runs" / str(task_id) / str(arm)
            run_dirs = []
            if arm_root.is_dir() and not arm_root.is_symlink():
                run_dirs = [
                    path
                    for path in sorted(arm_root.iterdir())
                    if path.is_dir() and not path.is_symlink() and path.resolve().is_relative_to(campaign_root)
                ]
            arm_record = {"status": "pending_run", "runs": []}
            if run_dirs:
                arm_record["status"] = "runs_present"
                for run_dir in run_dirs:
                    run_record, rows = _run_record(
                        campaign_root,
                        str(task_id),
                        str(arm),
                        run_dir,
                        task=task_value,
                        quality=quality,
                        attempts=attempts,
                    )
                    arm_record["runs"].append(run_record)
                    csv_rows.extend(rows)
            else:
                csv_rows.append(
                    {
                        "task": task_id,
                        "arm": arm,
                        "run_id": "",
                        "epoch": "",
                        "panel": "",
                        "status": "pending_run",
                        "evaluation_path": "",
                        "completion_utc": "",
                        "completion_seconds": "",
                        "checkpoint_sha256": "",
                        "coherence_sum_ratio": "",
                        "family_A_ratio": "",
                        "family_B_ratio": "",
                        "family_T_ratio": "",
                        "endpoint_aggregate_ratio": "",
                        "endpoint_field_ratios_json": "",
                        "training_segments_json": "[]",
                    }
                )
            task_record["arms"][str(arm)] = arm_record
        report["tasks"][str(task_id)] = task_record
    return report, csv_rows


CSV_COLUMNS = (
    "task",
    "arm",
    "run_id",
    "epoch",
    "panel",
    "status",
    "evaluation_path",
    "completion_utc",
    "completion_seconds",
    "completion_time_basis",
    "measurement_source",
    "wall_seconds_from_first_training_command",
    "active_process_seconds_from_first_training_command",
    "task_profile_acquisition_seconds",
    "allocated_profile_acquisition_seconds",
    "checkpoint_sha256",
    "coherence_sum_ratio",
    "family_A_ratio",
    "family_B_ratio",
    "family_T_ratio",
    "endpoint_aggregate_ratio",
    "endpoint_field_ratios_json",
    "training_segments_json",
)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


COMPONENT_CSV_COLUMNS = (
    "task",
    "arm",
    "run_id",
    "epoch",
    "panel",
    "group",
    "metric_path",
    "source_value",
    "child_value",
)


def _component_rows_from_report(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id, task in report.get("tasks", {}).items():
        for arm, arm_record in task.get("arms", {}).items():
            for run in arm_record.get("runs", []):
                for evaluation in [
                    *run.get("uniform_evaluations", []),
                    *run.get("portable_evaluations", []),
                    *run.get("supplementary_evaluations", []),
                ]:
                    rows.extend(
                        _group_component_rows(
                            task_id=str(task_id),
                            arm=str(arm),
                            run_id=str(run.get("run_id", "")),
                            epoch=int(evaluation["epoch"]),
                            panel=str(evaluation.get("panel_kind", "unknown")),
                            payload={"groups": evaluation.get("paired_group_components", [])},
                        )
                    )
    return rows


def _atomic_components_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPONENT_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _validate_output_root(root: Path) -> Path:
    if root.is_symlink():
        raise CampaignAnalysisError("campaign root may not be a symlink")
    resolved = root.resolve(strict=True)
    if not resolved.is_dir() or resolved.stat().st_uid != os.getuid():
        raise CampaignAnalysisError("campaign root must be an existing directory owned by this user")
    if not resolved.is_relative_to(Path("/data2/wanglz/CostAwareLPC").resolve(strict=True)):
        raise CampaignAnalysisError("campaign root must remain below /data2/wanglz/CostAwareLPC")
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    try:
        root = _validate_output_root(args.root)
        report, csv_rows = build_campaign_quality_history(root)
        _atomic_json(root / "analysis" / "quality_history.json", report)
        _atomic_csv(root / "analysis" / "quality_history.csv", csv_rows)
        _atomic_components_csv(
            root / "analysis" / "quality_components.csv",
            _component_rows_from_report(report),
        )
    except CampaignAnalysisError as error:
        print(f"campaign analysis blocked: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"root": str(root), "runs": sum(
        len(arm["runs"])
        for task in report["tasks"].values()
        for arm in task["arms"].values()
    ), "quality_json": str(root / "analysis" / "quality_history.json"),
        "quality_csv": str(root / "analysis" / "quality_history.csv"),
        "components_csv": str(root / "analysis" / "quality_components.csv")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
