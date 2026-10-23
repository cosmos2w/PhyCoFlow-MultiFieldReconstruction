#!/usr/bin/env python3
"""Foreground runner for the bounded cost-aware reconstruction campaign.

Every action is explicit. Inspect/plan only read configs and receipts; profile,
launch, resume, and evaluate require ``--execute`` and stay in the foreground.
The runner pins sources, uses the campaign's owned /data2 root, and bootstraps
the retained dataset identity before importing any child trainer/evaluator.
"""

from __future__ import annotations

import time

# Capture startup before importing the controller's remaining dependencies.
CLI_START_MONOTONIC = time.monotonic()

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import runpy
import shlex
import signal
import stat
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src"
DEFAULT_ROOT = Path(
    "/data2/wanglz/CostAwareLPC/CostAwareLPC_20261023T185427Z_01a16a15"
)
PROFILE_SCRIPT = Path(__file__).resolve().with_name("cost_aware_profile.py")
EVALUATOR_SCRIPT = Path(__file__).resolve().with_name("evaluate_full_abt.py")
DEFAULT_GPU = 2
MAX_PHYSICAL_GPU = 2
MAX_TASK_EPOCHS = 100
SCIENCE_EPOCHS_CAP = 800
AUX_EPOCHS_CAP = 50
TOTAL_EPOCHS_CAP = 850
PROCESS_SECONDS_CAP = 16 * 60 * 60
PROFILE_SECONDS_CAP = 90 * 60
CLOSEOUT_RESERVE_SECONDS = 60 * 60
EVALUATION_SECONDS_CAP = CLOSEOUT_RESERVE_SECONDS
PROFILE_AUX_RESERVE_EPOCHS = 6
STOP_LOSS_RESERVE_EPOCHS = 1.0
FRESH_LAUNCH_CALIBRATION_RESERVE_SECONDS = 120.0
STOP_WAIT_SECONDS = 30
ROOT_SUBDIRS = (
    "tasks",
    "profiles",
    "plans",
    "runs",
    "analysis",
    "reports",
    "caches",
    "tmp",
    "manifests",
    "configs",
)
LEDGER_PATH = Path("manifests/campaign_runner_ledger.jsonl")
ACTIVE_PATH = Path("manifests/active_process.json")
LOCK_PATH = Path("manifests/campaign_runner.lock")
IDENTITY_RECEIPT_PATH = Path("manifests/dataset_identities.json")
LEGACY_EVALUATION_GUARD_ENV = (
    "PHYCOFLOW_CAMPAIGN_SOURCE_STAGE",
    "PHYCOFLOW_CAMPAIGN_ADMITTED",
    "PHYCOFLOW_CAMPAIGN_GPU",
)


class CampaignError(RuntimeError):
    """Raised when a frozen task, resource, or process-ownership check fails."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignError(f"{label} must be a mapping")
    return dict(value)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read {label} at {path}: {error}") from error
    return _mapping(payload, label)


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise CampaignError(f"cannot read {label} at {path}: {error}") from error
    return _mapping(payload, label)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> str:
    expected = _canonical_json(dict(payload))
    if path.exists():
        existing = _read_json(path, "immutable campaign artifact")
        if _canonical_json(existing) != expected:
            raise CampaignError(f"immutable campaign artifact changed: {path}")
        return _sha256(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise CampaignError(f"campaign artifact appeared concurrently: {path}") from error
    return _sha256(path)


def _safe_name(value: str, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value):
        raise CampaignError(f"{label} must be a short filesystem-safe identifier")
    return value


def _root_path(value: Path) -> Path:
    if value.is_symlink():
        raise CampaignError(f"campaign root may not be a symlink: {value}")
    try:
        root = value.resolve(strict=True)
    except OSError as error:
        raise CampaignError(f"campaign root does not exist: {value}") from error
    storage_parent = Path("/data2/wanglz/CostAwareLPC").resolve(strict=True)
    if Path("/data2/wanglz/CostAwareLPC").is_symlink():
        raise CampaignError("campaign storage parent may not be a symlink")
    if root == storage_parent or not root.is_relative_to(storage_parent):
        raise CampaignError(f"campaign root must be owned below {storage_parent}: {root}")
    return root


def _mount_identity() -> str:
    if not Path("/data2").is_mount():
        raise CampaignError("/data2 is not mounted")
    try:
        result = subprocess.run(
            ["findmnt", "-n", "-o", "SOURCE,FSTYPE,TARGET", "--target", "/data2"],
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise CampaignError(f"cannot verify /data2 mount identity: {error}") from error
    return " ".join(result.stdout.split())


def _verify_root(root_arg: Path, *, create_payload_dirs: bool) -> tuple[Path, dict[str, Any]]:
    root = _root_path(root_arg)
    if not root.is_dir() or root.stat().st_uid != os.getuid():
        raise CampaignError(f"campaign root must be a directory owned by uid {os.getuid()}: {root}")
    if os.stat(root).st_dev != os.stat("/data2").st_dev:
        raise CampaignError("campaign root is not on the current /data2 filesystem")
    preflight = _read_json(root / "manifests/preflight.json", "campaign preflight receipt")
    if Path(str(preflight.get("root", ""))).resolve() != root:
        raise CampaignError("preflight receipt names a different campaign root")
    if preflight.get("uid") != os.getuid():
        raise CampaignError("preflight receipt was created for a different user")
    actual_mount = _mount_identity()
    if " ".join(str(preflight.get("mount", "")).split()) != actual_mount:
        raise CampaignError("/data2 mount identity differs from the preflight receipt")
    expected_limits = {
        "science_epochs_cap": SCIENCE_EPOCHS_CAP,
        "aux_epochs_cap": AUX_EPOCHS_CAP,
        "total_epochs_cap": TOTAL_EPOCHS_CAP,
        "combined_process_seconds_cap": PROCESS_SECONDS_CAP,
        "profile_calibration_seconds_cap": PROFILE_SECONDS_CAP,
        "closeout_reserve_seconds": CLOSEOUT_RESERVE_SECONDS,
    }
    for name, value in expected_limits.items():
        if preflight.get(name) != value:
            raise CampaignError(f"preflight {name} differs from the campaign contract")
    if preflight.get("test_access") != "forbidden":
        raise CampaignError("preflight does not forbid TEST access")
    if preflight.get("long_formal_launch_or_queue") != "forbidden":
        raise CampaignError("preflight does not forbid long formal launches and queueing")
    if create_payload_dirs:
        for name in ROOT_SUBDIRS:
            path = root / name
            if os.path.lexists(path):
                if path.is_symlink() or not path.is_dir() or not path.resolve().is_relative_to(root):
                    raise CampaignError(f"campaign payload path is unsafe: {path}")
            else:
                path.mkdir(mode=0o750)
            if path.stat().st_uid != os.getuid() or path.stat().st_dev != root.stat().st_dev:
                raise CampaignError(f"campaign payload ownership/device mismatch: {path}")
    return root, preflight


def _ensure_link(link: Path, target: Path) -> None:
    target = target.resolve(strict=True)
    if link.is_symlink():
        try:
            current = link.resolve(strict=True)
        except FileNotFoundError as error:
            raise CampaignError(f"refusing to replace a dangling symlink: {link}") from error
        if current != target:
            raise CampaignError(f"refusing to replace unrelated symlink {link} -> {current}")
        return
    if os.path.lexists(link):
        raise CampaignError(f"refusing to replace an existing path: {link}")
    link.symlink_to(target, target_is_directory=True)


def _ensure_repository_links(root: Path, case_dir: Path | None = None) -> dict[str, str]:
    report_link = REPO_ROOT / "reports" / root.name
    _ensure_link(report_link, root / "reports")
    result = {"reports": str(report_link), "reports_target": str(report_link.resolve())}
    if case_dir is not None:
        runs = case_dir / "runs"
        if not runs.is_dir() or runs.is_symlink():
            raise CampaignError(f"case runs directory must be an existing local directory: {runs}")
        parent = runs / "_CostAwareLPC"
        if os.path.lexists(parent):
            if parent.is_symlink() or not parent.is_dir():
                raise CampaignError(f"campaign run-link directory is unsafe: {parent}")
        else:
            parent.mkdir(mode=0o750)
        run_link = parent / root.name
        _ensure_link(run_link, root / "runs")
        result.update({"runs": str(run_link), "runs_target": str(run_link.resolve())})
    return result


def _inventory(root: Path) -> dict[str, Any]:
    payload = _read_json(root / "tasks/task_inventory.json", "frozen task inventory")
    tasks = _mapping(payload.get("tasks"), "task inventory tasks")
    if not tasks:
        raise CampaignError("task inventory is empty")
    return payload


def _task(root: Path, task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    task_id = _safe_name(task_id, "task id")
    inventory = _inventory(root)
    tasks = _mapping(inventory["tasks"], "task inventory tasks")
    if task_id not in tasks:
        raise CampaignError(f"task {task_id!r} is not frozen in the campaign inventory")
    task = _mapping(tasks[task_id], f"task {task_id}")
    if task.get("role") == "optional" and not inventory.get("optional_pair_predeclared_before_new_scientific_outcomes"):
        raise CampaignError("optional task was not predeclared before scientific outcomes")
    return task, inventory


def _config_path(root: Path, task_id: str, arm: str) -> Path:
    task_id = _safe_name(task_id, "task id")
    arm = _safe_name(arm, "arm")
    path = root / "configs" / f"{task_id}_{arm}.yaml"
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
        raise CampaignError(f"frozen config is missing or unsafe: {path}")
    return path


def _config_digest(config: Mapping[str, Any]) -> str:
    serialized = yaml.safe_dump(dict(config), sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _resolve_source_checkpoint(config: Mapping[str, Any], source_run: Path) -> Path:
    setting = Path(str(config.get("source_checkpoint", ""))).expanduser()
    if not str(setting):
        raise CampaignError("frozen config has no source_checkpoint")
    if setting.is_absolute():
        checkpoint = setting
    elif setting.suffix == ".pt":
        checkpoint = source_run / "checkpoints" / setting.name
    else:
        checkpoint = source_run / "checkpoints" / f"{setting.name}.pt"
    try:
        resolved = checkpoint.resolve(strict=True)
    except OSError as error:
        raise CampaignError(f"pinned SOURCE checkpoint is missing: {checkpoint}") from error
    if not resolved.is_file():
        raise CampaignError(f"pinned SOURCE checkpoint is not a file: {resolved}")
    if not resolved.is_relative_to(source_run.resolve(strict=True)):
        raise CampaignError(f"pinned SOURCE checkpoint escapes its SOURCE run: {resolved}")
    return resolved


def _source_binding(root: Path, task: Mapping[str, Any], config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    source_run_text = str(task.get("source_run", config.get("source_run", "")))
    source_run = Path(source_run_text).expanduser().resolve(strict=True)
    configured_source = Path(str(config.get("source_run", ""))).expanduser().resolve(strict=True)
    if configured_source != source_run:
        raise CampaignError("prepared config source_run differs from the frozen task inventory")
    source_manifest_path = source_run / "run_manifest.json"
    source_config_path = source_run / "resolved_config.yaml"
    source_status_path = source_run / "status.json"
    source_manifest = _read_json(source_manifest_path, "SOURCE run manifest")
    source_status = _read_json(source_status_path, "SOURCE status")
    if source_status.get("status") != "completed" or source_manifest.get("stage") not in {
        "base_training",
        "post_training",
    }:
        raise CampaignError("SOURCE is not a completed native training run")
    checkpoint = _resolve_source_checkpoint(config, source_run)
    checkpoint_sha = _sha256(checkpoint)
    pinned_sha = str(task.get("source_checkpoint_sha256", ""))
    if checkpoint_sha != pinned_sha:
        raise CampaignError("SOURCE checkpoint hash differs from the frozen task inventory")
    manifest_hash = source_manifest.get("checkpoint_hashes", {}).get(checkpoint.stem)
    if manifest_hash != checkpoint_sha:
        raise CampaignError("SOURCE run manifest does not attest the configured checkpoint")
    resolved_config_sha = _sha256(source_config_path)
    inventory_identity = _mapping(task.get("dataset_identity"), "task dataset identity")
    if (
        source_manifest.get("dataset_fingerprint") != inventory_identity.get("fingerprint")
        or Path(str(source_manifest.get("dataset_path", ""))).resolve()
        != Path(str(inventory_identity.get("path", ""))).resolve()
    ):
        raise CampaignError("SOURCE run manifest dataset identity differs from the frozen inventory")
    if Path(str(config.get("dataset", {}).get("path", ""))).resolve() != Path(
        str(inventory_identity.get("path", ""))
    ).resolve():
        raise CampaignError("prepared config dataset differs from the retained SOURCE identity")
    if str(config.get("case", "")) != str(task.get("case", "")):
        raise CampaignError("prepared config case differs from the frozen task inventory")
    return {
        "source_run": str(source_run),
        "source_run_manifest": str(source_manifest_path),
        "source_run_manifest_sha256": _sha256(source_manifest_path),
        "source_resolved_config": str(source_config_path),
        "source_resolved_config_sha256": resolved_config_sha,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": checkpoint_sha,
        "dataset_path": str(inventory_identity["path"]),
        "dataset_fingerprint": str(inventory_identity["fingerprint"]),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
    }


def _verify_identity_receipt(root: Path, task: Mapping[str, Any] | None = None) -> dict[str, Any]:
    receipt_path = root / IDENTITY_RECEIPT_PATH
    receipt = _read_json(receipt_path, "dataset identity receipt")
    if receipt.get("version") != 1 or not isinstance(receipt.get("identities"), list):
        raise CampaignError("dataset identity receipt schema is invalid")
    identities: dict[str, dict[str, Any]] = {}
    for raw_entry in receipt["identities"]:
        entry = _mapping(raw_entry, "dataset identity entry")
        path = Path(str(entry.get("path", ""))).resolve(strict=True)
        fingerprint = str(entry.get("fingerprint", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise CampaignError(f"invalid retained dataset fingerprint for {path}")
        expected_stat = entry.get("expected_stat")
        if not isinstance(expected_stat, list) or len(expected_stat) != 4:
            raise CampaignError(f"invalid frozen stat tuple for {path}")
        attestation_path = Path(str(entry.get("attestation_source", ""))).resolve(strict=True)
        attestation_sha = _sha256(attestation_path)
        if attestation_sha != entry.get("attestation_sha256"):
            raise CampaignError(f"SOURCE dataset identity attestation changed: {attestation_path}")
        source_record = _read_json(attestation_path, "dataset identity SOURCE manifest")
        if (
            source_record.get("dataset_fingerprint") != fingerprint
            or Path(str(source_record.get("dataset_path", ""))).resolve() != path
        ):
            raise CampaignError(f"SOURCE manifest does not bind retained dataset identity: {attestation_path}")
        identities[str(path)] = {
            "path": str(path),
            "fingerprint": fingerprint,
            "expected_stat": [int(value) for value in expected_stat],
            "attestation_source": str(attestation_path),
            "attestation_sha256": attestation_sha,
        }
        info = path.stat()
        live_stat = [int(info.st_dev), int(info.st_ino), int(info.st_size), int(info.st_mtime_ns)]
        if live_stat != [int(value) for value in expected_stat]:
            raise CampaignError(f"dataset metadata differs from the retained identity receipt: {path}")
    if task is not None:
        expected = _mapping(task.get("dataset_identity"), "task dataset identity")
        data_path = str(Path(str(expected.get("path", ""))).resolve(strict=True))
        entry = identities.get(data_path)
        if entry is None or entry["fingerprint"] != expected.get("fingerprint"):
            raise CampaignError("task dataset is not bound by the mounted identity receipt")
        if tuple(entry["expected_stat"]) != tuple(int(x) for x in expected.get("expected_stat", ())):
            raise CampaignError("task dataset stat binding differs from the mounted identity receipt")
        if (
            entry["attestation_sha256"] != expected.get("attestation_sha256")
            or entry["attestation_source"]
            != str(Path(str(expected.get("attestation_source", ""))).resolve(strict=True))
        ):
            raise CampaignError("task dataset attestation differs from the frozen inventory")
    return {"path": str(receipt_path), "sha256": _sha256(receipt_path), "identities": identities}


def _register_child_identities(root: Path) -> None:
    receipt = _verify_identity_receipt(root)
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    from phycoflow_reconstruction.data.manifest import register_dataset_identity

    for entry in receipt["identities"].values():
        register_dataset_identity(
            entry["path"], entry["fingerprint"], entry["expected_stat"]
        )


def _read_task_contract_seed(
    task_id: str,
    task: Mapping[str, Any],
    inventory: Mapping[str, Any],
    config: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    explicit = task.get("task_contract")
    if isinstance(explicit, Mapping):
        contract = dict(explicit)
        contract.setdefault("task_id", task_id)
        contract.setdefault("source_identity", binding["source_checkpoint_sha256"])
        contract.setdefault("case_id", task.get("case", config.get("case")))
        identity = _mapping(task.get("dataset_identity"), "task dataset identity")
        contract.setdefault(
            "dataset_id", f"{identity.get('path')}#sha256:{identity.get('fingerprint')}"
        )
        contract.setdefault("quality_contract_identity", hashlib.sha256(
            _canonical_json(inventory.get("quality_contract", {})).encode()
        ).hexdigest())
        contract.setdefault("max_epochs", int(task.get("horizon", MAX_TASK_EPOCHS)))
        return contract

    source_manifest = _read_json(Path(str(binding["source_run_manifest"])), "SOURCE run manifest")
    model_registry = source_manifest.get("model_registry_metadata", {})
    family_name = str(model_registry.get("family", "")) if isinstance(model_registry, Mapping) else ""
    adapter_kind = task.get("adapter_kind")
    if adapter_kind not in {"deterministic", "generative"}:
        adapter_kind = "generative" if any(
            token in family_name.lower() for token in ("flow", "diffusion", "generative")
        ) else "deterministic"
    dataset = _mapping(config.get("dataset"), "prepared config dataset")
    optimization = _mapping(config.get("optimization"), "prepared config optimization")
    coherence = _mapping(config.get("coherence", {}), "prepared coherence config")
    families = _mapping(coherence.get("families", {}), "coherence families")
    component_weights = task.get("component_weights")
    if not component_weights:
        component_weights = [
            [str(name), float(spec.get("weight", 1.0))]
            for name, spec in sorted(families.items())
            if isinstance(spec, Mapping) and spec.get("enabled", True)
        ] or [["objective", 1.0]]
    task_quality = task.get("quality_contract", inventory.get("quality_contract", {}))
    identity = _mapping(task.get("dataset_identity"), "task dataset identity")
    multitask = optimization.get("multitask", {})
    fixed_scalar = (
        isinstance(multitask, Mapping)
        and isinstance(multitask.get("family_scalarization"), Mapping)
        and multitask["family_scalarization"].get("method") == "fixed_sum"
        and multitask.get("outer_aggregation") in (None, "sum")
    )
    raw_objective_rule = task.get("objective_rule") or (
        "scalar_additive" if fixed_scalar else str(optimization.get("gradient_balance", "unspecified"))
    )

    def identity_hash(value: Any) -> str:
        return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()

    return {
        "task_id": task_id,
        "source_identity": (
            f"run_manifest_sha256:{binding['source_run_manifest_sha256']}"
            f"|checkpoint_sha256:{binding['source_checkpoint_sha256']}"
        ),
        "case_id": str(task.get("case", config.get("case", ""))),
        "dataset_id": f"{identity.get('path')}#sha256:{identity.get('fingerprint')}",
        "split_identity": str(
            task.get("split_identity")
            or f"{source_manifest.get('split_strategy', 'unknown')}"
            f"|split:{dataset.get('split', 'train')}"
            f"|train_fraction:{optimization.get('train_fraction', 'unspecified')}"
        ),
        "adapter_kind": str(adapter_kind),
        "reconstruction_identity": str(task.get("reconstruction_identity") or identity_hash({
            "model": config.get("model", {}),
            "rollout": config.get("rollout", {}),
            "observations": config.get("observations", {}),
            "geometry": {
                key: dataset.get(key)
                for key in ("coordinate_dim", "grid_shape", "field_names", "coordinate_reorder")
            },
        })),
        "normalization_identity": str(task.get("normalization_identity") or identity_hash({
            "method": dataset.get("normalization"),
            "stats_path": dataset.get("normalization_stats_path"),
            "source_normalization_digest": source_manifest.get("normalization_digest"),
        })),
        "objective_identity": str(task.get("objective_identity") or identity_hash({
            "objectives": config.get("objectives", {}),
            "coherence": coherence,
            "rollout": config.get("rollout", {}),
            "fidelity": config.get("posttrain_fidelity", {}),
        })),
        "objective_rule": str(raw_objective_rule),
        "component_weights": component_weights,
        "aggregation_rule": str(task.get("aggregation_rule") or _canonical_json({
            "multitask": multitask,
            "gradient_balance": optimization.get("gradient_balance"),
        })),
        "query_geometry_identity": str(task.get("query_geometry_identity") or identity_hash({
            "compute_budget": coherence.get("compute_budget", {}),
            "query_points": config.get("model", {}).get("query_points"),
            "grid_shape": dataset.get("grid_shape"),
            "coordinate_dim": dataset.get("coordinate_dim"),
        })),
        "sampling_identity": str(task.get("sampling_identity") or identity_hash({
            "observations": config.get("observations", {}),
            "batch_size": optimization.get("batch_size"),
            "train_fraction": optimization.get("train_fraction"),
            "seed": config.get("runtime", {}).get("seed"),
        })),
        "numerical_precision": str(task.get("numerical_precision") or config.get("runtime", {}).get("precision", "float32")),
        "optimizer_identity": str(task.get("optimizer_identity") or identity_hash({
            "multitask": multitask,
            "gradient_balance": optimization.get("gradient_balance"),
            "lr": optimization.get("lr"),
            "weight_decay": optimization.get("weight_decay"),
        })),
        "epoch_definition": str(task.get("epoch_definition") or identity_hash({
            "batch_size": optimization.get("batch_size"),
            "train_fraction": optimization.get("train_fraction"),
            "epoch_definition": source_manifest.get("epoch_definition"),
        })),
        "quality_contract_identity": str(task.get("quality_contract_identity") or identity_hash(task_quality)),
        "max_epochs": int(task.get("horizon", optimization.get("epochs", MAX_TASK_EPOCHS))),
        # Applicability is deliberately sourced only from explicit inventory declarations.
        "provider_ids": list(task.get("provider_ids", ())),
        "features": list(task.get("features", ())),
        "metadata": {
            "calibration_status": "pending",
            "source_run": binding["source_run"],
            "source_checkpoint_sha256": binding["source_checkpoint_sha256"],
            "dataset_fingerprint": identity["fingerprint"],
        },
    }


def _inspect_task(root: Path, task_id: str) -> dict[str, Any]:
    task, inventory = _task(root, task_id)
    _verify_identity_receipt(root, task)
    case_dir = REPO_ROOT / "cases" / str(task.get("case", ""))
    if not case_dir.is_dir():
        raise CampaignError(f"frozen case directory is unavailable: {case_dir}")
    links = _ensure_repository_links(root, case_dir)
    bindings: dict[str, Any] = {}
    base_config: dict[str, Any] | None = None
    base_path: Path | None = None
    # Strategy is produced by the measured profile. Keep inspection bound to
    # the predeclared source arms so the request remains stable afterward.
    for arm in ("dense", "heuristic"):
        path = root / "configs" / f"{task_id}_{arm}.yaml"
        if not path.exists():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise CampaignError(f"prepared config escapes the campaign root: {path}")
        config = _read_yaml(path, f"{task_id} {arm} config")
        binding = _source_binding(root, task, path, config)
        horizon = int(config.get("optimization", {}).get("epochs", 0))
        if not 1 <= horizon <= MAX_TASK_EPOCHS:
            raise CampaignError(f"{task_id} {arm} horizon must be within 1..{MAX_TASK_EPOCHS}")
        bindings[arm] = {
            "path": str(path),
            "sha256": binding["config_sha256"],
            "horizon": horizon,
            "source_binding": binding,
        }
        if arm == "dense":
            base_config, base_path = config, path
    if not bindings:
        raise CampaignError(f"no frozen configs exist for task {task_id}")
    if base_config is None or base_path is None:
        # A deterministic second-seed task can use its declared strategy as the base.
        chosen = next(iter(bindings))
        base_path = Path(bindings[chosen]["path"])
        base_config = _read_yaml(base_path, f"{task_id} {chosen} config")
    source_binding = bindings.get("dense", next(iter(bindings.values())))["source_binding"]
    task_contract = _read_task_contract_seed(task_id, task, inventory, base_config, source_binding)
    if int(task_contract.get("max_epochs", 0)) > MAX_TASK_EPOCHS:
        raise CampaignError("task contract exceeds the 100-epoch per-child cap")
    if "c0" not in task_contract or "field_bounds" not in task_contract:
        task_contract.setdefault("calibration_status", "pending")
        task_contract.setdefault("source_calibration_identity", "pending")

    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    from phycoflow_reconstruction.training.execution_strategy import manifest_for_task

    try:
        strategy_manifest = manifest_for_task(task_contract)
    except (TypeError, ValueError, KeyError) as error:
        raise CampaignError(f"cannot build config-only task capability manifest: {error}") from error
    task_dir = root / "tasks"
    contract_request = {
        "schema_version": 1,
        "status": "inspection_only",
        "task_id": task_id,
        "task_contract": strategy_manifest["task_contract"],
        "calibration_status": strategy_manifest["task_contract"].get("calibration_status", "pending"),
        "profile_not_authorized": True,
        "source_binding": source_binding,
        "dataset_identity_receipt_sha256": _verify_identity_receipt(root, task)["sha256"],
    }
    inspection_basis = {
        "task_id": task_id,
        "task_hash": strategy_manifest.get("task_hash"),
        "source_binding": source_binding,
        "config_sha256": {
            name: value["sha256"] for name, value in sorted(bindings.items())
        },
        "dataset_identity_receipt_sha256": contract_request[
            "dataset_identity_receipt_sha256"
        ],
    }
    inspection_id = hashlib.sha256(_canonical_json(inspection_basis).encode()).hexdigest()[:16]
    stem = f"{task_id}_{inspection_id}"
    contract_path = task_dir / f"{stem}_task_contract_request.json"
    capabilities_path = task_dir / f"{stem}_capabilities.json"
    request_path = task_dir / f"{stem}_cost_profile_request.json"
    binding_path = task_dir / f"{stem}_config_bindings.json"
    hashes = {
        "task_contract_request": _write_immutable_json(contract_path, contract_request),
        "capabilities": _write_immutable_json(
            capabilities_path,
            {"schema_version": 1, "task_hash": strategy_manifest.get("task_hash"), "capabilities": strategy_manifest["capabilities"]},
        ),
        "cost_profile_request": _write_immutable_json(
            request_path,
            {**strategy_manifest["cost_profile_request"], "profile_not_authorized": True},
        ),
        "config_bindings": _write_immutable_json(
            binding_path,
            {
                "schema_version": 1,
                "task_id": task_id,
                "source_identity": source_binding,
                "configs": bindings,
                "case_dir": str(case_dir),
                "repository_links": links,
            },
        ),
    }
    return {
        "task_id": task_id,
        "inspection_id": inspection_id,
        "root": str(root),
        "status": "inspection_only",
        "calibration_status": contract_request["calibration_status"],
        "task_hash": strategy_manifest.get("task_hash"),
        "artifacts": {
            "task_contract_request": str(contract_path),
            "capabilities": str(capabilities_path),
            "cost_profile_request": str(request_path),
            "config_bindings": str(binding_path),
            "sha256": hashes,
        },
    }


def _verify_frozen_config_binding(
    root: Path, task_id: str, arm: str, binding: Mapping[str, Any]
) -> None:
    """Require the config bytes to match a prior immutable inspection/profile pin."""
    if arm == "strategy":
        candidate = root / "tasks" / f"{task_id}_strategy_config_binding.json"
        if not candidate.is_file() or candidate.is_symlink():
            raise CampaignError("strategy config has no immutable profile binding")
        payload = _read_json(candidate, "immutable strategy config binding")
        if (
            payload.get("path") == str(Path(str(binding["config_path"])).resolve(strict=True))
            and payload.get("sha256") == str(binding["config_sha256"])
            and _sha256(Path(str(binding["config_path"])).resolve(strict=True))
            == str(binding["config_sha256"])
        ):
            return
        raise CampaignError("strategy config differs from its immutable profile binding")
    candidates = sorted((root / "tasks").glob(f"{task_id}_*_config_bindings.json"))
    expected_path = str(Path(str(binding["config_path"])).resolve(strict=True))
    expected_sha = str(binding["config_sha256"])
    if _sha256(Path(expected_path)) != expected_sha:
        raise CampaignError(f"{task_id} {arm} config changed after its immutable inspection binding")
    for candidate in candidates:
        if not candidate.is_file() or candidate.is_symlink():
            continue
        payload = _read_json(candidate, "immutable config binding")
        configs = payload.get("configs")
        if not isinstance(configs, Mapping):
            continue
        record = configs.get(arm)
        if not isinstance(record, Mapping):
            continue
        if record.get("path") == expected_path and record.get("sha256") == expected_sha:
            return
    raise CampaignError(
        f"{task_id} {arm} config does not match an immutable campaign binding; inspect it before execution"
    )


def _env_name() -> str:
    conda_name = os.environ.get("CONDA_DEFAULT_ENV", "")
    prefix_name = Path(sys.prefix).name
    return conda_name or prefix_name


def _require_environment() -> None:
    if _env_name() != "phycoflow_env":
        raise CampaignError(
            f"campaign execution requires phycoflow_env; active environment is {_env_name()!r}"
        )


def _gpu_record(gpu: int, preflight: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(gpu, bool) or not isinstance(gpu, int) or not 0 <= gpu <= MAX_PHYSICAL_GPU:
        raise CampaignError("physical GPU must be 0, 1, or 2")
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise CampaignError(f"cannot inspect physical GPU {gpu}: {error}") from error
    columns = [part.strip() for part in query.split(",")]
    if len(columns) != 6 or int(columns[0]) != gpu:
        raise CampaignError(f"nvidia-smi returned an invalid record for physical GPU {gpu}")
    try:
        pids = subprocess.run(
            ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise CampaignError(f"cannot verify compute processes on physical GPU {gpu}: {error}") from error
    if pids:
        raise CampaignError(f"physical GPU {gpu} has compute processes; the runner does not queue")
    total_mib, used_mib, utilization = int(columns[3]), int(columns[4]), int(columns[5])
    if used_mib > max(1024, int(total_mib * 0.05)) or utilization > 5:
        raise CampaignError(
            f"physical GPU {gpu} is not idle (memory {used_mib}/{total_mib} MiB, utilization {utilization}%)"
        )
    if gpu == int(preflight.get("gpu", DEFAULT_GPU)) and columns[1] != preflight.get("gpu_uuid"):
        raise CampaignError("physical GPU UUID differs from the preflight receipt")
    return {
        "physical_index": gpu,
        "uuid": columns[1],
        "name": columns[2],
        "memory_total_mib": total_mib,
        "memory_used_mib": used_mib,
        "utilization_percent": utilization,
    }


def _acquire_physical_gpu_lock(gpu: int) -> Any:
    """Hold a per-physical-device lock across each admitted child process."""
    path = Path(f"/tmp/phycoflow_cost_aware_gpu{gpu}.lock")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise CampaignError(f"cannot open physical GPU {gpu} admission lock: {error}") from error
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        os.close(descriptor)
        raise CampaignError(f"physical GPU lock is not a regular file owned by uid {os.getuid()}: {path}")
    handle = os.fdopen(descriptor, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise CampaignError(f"physical GPU {gpu} already has an admitted cost-aware process") from error
    return handle


def _campaign_environment(
    root: Path, gpu: int, *, started_monotonic: float = CLI_START_MONOTONIC
) -> dict[str, str]:
    cache_root = root / "caches"
    directories = {
        "keops": cache_root / "keops",
        "matplotlib": cache_root / "matplotlib",
        "pycache": cache_root / "pycache",
        "tmp": root / "tmp",
    }
    for path in directories.values():
        if os.path.lexists(path):
            if path.is_symlink() or not path.is_dir() or not path.resolve().is_relative_to(root):
                raise CampaignError(f"campaign runtime directory is unsafe: {path}")
        else:
            path.mkdir(mode=0o750, parents=True)
        if path.stat().st_uid != os.getuid() or path.stat().st_dev != root.stat().st_dev:
            raise CampaignError(f"campaign runtime directory ownership/device mismatch: {path}")
    env = os.environ.copy()
    for name in LEGACY_EVALUATION_GUARD_ENV:
        env.pop(name, None)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONPYCACHEPREFIX": str(directories["pycache"]),
            "TMPDIR": str(directories["tmp"]),
            "KEOPS_CACHE_FOLDER": str(directories["keops"]),
            "MPLCONFIGDIR": str(directories["matplotlib"]),
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "PHYCOFLOW_TOPOLOGY_WORKERS": "4",
            "COST_AWARE_CAMPAIGN_ADMITTED": str(root),
            "COST_AWARE_CAMPAIGN_GPU": str(gpu),
            "COST_AWARE_CAMPAIGN_ROOT": str(root),
            "COST_AWARE_DATASET_IDENTITIES": str(root / IDENTITY_RECEIPT_PATH),
            "PHYCOFLOW_COMMAND_STARTED_MONOTONIC": str(started_monotonic),
        }
    )
    build_name = (
        f"{'_'.join(platform.uname()[:3])}"
        f"_p{sys.version.split()[0]}_CUDA_VISIBLE_DEVICES_{gpu}"
    )
    build_dir = directories["keops"] / build_name
    build_dir.mkdir(mode=0o750, exist_ok=True)
    if build_dir.is_symlink() or not build_dir.resolve().is_relative_to(root):
        raise CampaignError(f"KeOps build directory is unsafe: {build_dir}")
    if build_dir.stat().st_uid != os.getuid() or build_dir.stat().st_dev != root.stat().st_dev:
        raise CampaignError(f"KeOps build directory ownership/device mismatch: {build_dir}")
    return env


def _process_group_members(process_group: int) -> list[tuple[int, str]]:
    members: list[tuple[int, str]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat_text = (entry / "stat").read_text(encoding="utf-8")
            _comm, remainder = stat_text.rsplit(")", 1)
            fields = remainder.strip().split()
            state, pgrp = fields[0], int(fields[2])
        except (OSError, ValueError, IndexError):
            continue
        if pgrp == process_group and state != "Z":
            members.append((int(entry.name), state))
    return members


def _terminate_owned_group(child: subprocess.Popen[bytes]) -> None:
    """Escalate only within the verified child session, then verify it is gone."""
    process_group = child.pid
    for signum, wait_seconds in (
        (signal.SIGINT, STOP_WAIT_SECONDS),
        (signal.SIGTERM, 5),
        (signal.SIGKILL, 5),
    ):
        try:
            os.killpg(process_group, signum)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=wait_seconds)
        except subprocess.TimeoutExpired:
            pass
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline and _process_group_members(process_group):
            time.sleep(0.1)
        if not _process_group_members(process_group):
            break
    members = _process_group_members(process_group)
    if child.poll() is None or members:
        raise CampaignError(
            f"owned process group {process_group} remains after SIGINT/SIGTERM/SIGKILL: {members}"
        )


def _latest_ledger(root: Path) -> dict[str, dict[str, Any]]:
    path = root / LEDGER_PATH
    if not path.exists():
        return {}
    latest: dict[str, dict[str, Any]] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            attempt_id = str(row["attempt_id"])
            latest[attempt_id] = _mapping(row, "campaign ledger row")
    except (OSError, json.JSONDecodeError, KeyError) as error:
        raise CampaignError(f"campaign resource ledger is invalid: {error}") from error
    return latest


def _resource_totals(root: Path) -> dict[str, float]:
    totals = {
        "process_seconds": 0.0,
        "profile_seconds": 0.0,
        "closeout_seconds": 0.0,
        "science_epochs": 0.0,
        "aux_epochs": 0.0,
    }
    for row in _latest_ledger(root).values():
        for key, final_name, reserve_name in (
            ("process_seconds", "charged_process_seconds", "reserved_process_seconds"),
            ("profile_seconds", "charged_profile_seconds", "reserved_profile_seconds"),
            ("closeout_seconds", "charged_closeout_seconds", "reserved_closeout_seconds"),
            ("science_epochs", "charged_science_epochs", "reserved_science_epochs"),
            ("aux_epochs", "charged_aux_epochs", "reserved_aux_epochs"),
        ):
            if final_name in row:
                totals[key] += float(row[final_name])
            elif row.get("status") == "running":
                totals[key] += float(row.get(reserve_name, 0.0))
    return totals


def _append_ledger(root: Path, row: Mapping[str, Any]) -> None:
    path = root / LEDGER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _acquire_admission(root: Path) -> Any:
    lock_path = root / LOCK_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise CampaignError("another campaign action owns admission; no process was queued") from error
    return handle


def _check_no_active_process(root: Path) -> None:
    active_path = root / ACTIVE_PATH
    if not active_path.exists():
        return
    active = _read_json(active_path, "active process receipt")
    pid = active.get("pid")
    if isinstance(pid, int) and Path(f"/proc/{pid}").exists():
        raise CampaignError(f"owned process {pid} is still active; use status or stop first")
    raise CampaignError(
        "a stale active-process receipt remains; inspect its task/run/log before removing it"
    )


def _parse_duration(value: float | None, *, maximum: float, label: str) -> float:
    duration = maximum if value is None else float(value)
    if not math.isfinite(duration) or duration <= 0 or duration > maximum:
        raise CampaignError(f"{label} must be within (0, {maximum:g}] seconds")
    return duration


def _child_script_command(
    root: Path,
    script: Path,
    child_args: Sequence[str],
    *,
    case_dir: Path | None = None,
) -> list[str]:
    script = script.resolve(strict=True)
    if not script.is_relative_to(REPO_ROOT):
        raise CampaignError(f"child script must be inside this repository: {script}")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--root",
        str(root),
        "_child",
        "--identities",
        str(root / IDENTITY_RECEIPT_PATH),
        "--script",
        str(script),
    ]
    if case_dir is not None:
        command.extend(["--case-dir", str(case_dir.resolve(strict=True))])
    return [*command, "--", *map(str, child_args)]


def _discover_run_directory(root: Path, task_id: str, arm: str, case_dir: Path) -> Path:
    link = case_dir / "runs" / "_CostAwareLPC" / root.name / task_id / arm
    try:
        target = link.resolve(strict=True)
    except OSError as error:
        raise CampaignError(f"expected campaign output directory is missing: {link}") from error
    if not target.is_relative_to((root / "runs").resolve(strict=True)):
        raise CampaignError(f"run directory escaped the campaign root: {target}")
    candidates = sorted(
        (
            item
            for item in target.iterdir()
            if item.is_dir() and not item.is_symlink()
        ),
        key=lambda item: item.stat().st_mtime_ns,
    )
    if not candidates:
        raise CampaignError(f"no run directory was created below {target}")
    if any(item.stat().st_uid != os.getuid() for item in candidates):
        raise CampaignError("campaign run directory ownership differs from the current user")
    return candidates[-1]


def _run_epoch(
    run_dir: Path, *, checkpoint_payload: Mapping[str, Any] | None = None
) -> float:
    manifest_path = run_dir / "run_manifest.json"
    status_path = run_dir / "status.json"
    manifest = _read_json(manifest_path, "run manifest") if manifest_path.is_file() else {}
    status = _read_json(status_path, "run status") if status_path.is_file() else {}
    progress_path = run_dir / "progress.json"
    progress = _read_json(progress_path, "run progress") if progress_path.is_file() else {}
    checkpoint_path = run_dir / "checkpoints" / "last.pt"
    if checkpoint_path.is_symlink():
        raise CampaignError("campaign recovery checkpoint may not be a symlink")
    if checkpoint_path.is_file():
        checkpoint = (
            _mapping(checkpoint_payload, "campaign recovery checkpoint")
            if checkpoint_payload is not None
            else _load_campaign_checkpoint(run_dir, manifest=manifest)
        )
        updates = checkpoint["global_step"]
        step_values = {
            int(value)
            for value in (
                status.get("steps_per_epoch"),
                progress.get("steps_per_epoch"),
            )
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
        }
        if len(step_values) > 1:
            raise CampaignError("run status and progress disagree on steps_per_epoch")
        steps_per_epoch = next(iter(step_values), 0)
        if steps_per_epoch <= 0 and updates > 0:
            checkpoint_epoch = manifest.get("last_checkpoint_epoch")
            if manifest.get("last_checkpoint_step") == updates and isinstance(
                checkpoint_epoch, (int, float)
            ) and not isinstance(checkpoint_epoch, bool) and checkpoint_epoch > 0:
                derived = updates / float(checkpoint_epoch)
                if math.isclose(derived, round(derived), rel_tol=0.0, abs_tol=1e-9):
                    steps_per_epoch = round(derived)
            if steps_per_epoch <= 0:
                steps_per_epoch = _steps_per_epoch_from_history(run_dir, updates)
        if updates == 0:
            epochs = 0.0
        elif steps_per_epoch > 0:
            if updates % steps_per_epoch:
                raise CampaignError(
                    "campaign checkpoint cursor is not on the frozen epoch boundary"
                )
            epochs = updates / steps_per_epoch
        else:
            raise CampaignError(
                "cannot convert the authoritative recovery checkpoint cursor to epochs"
            )
        if manifest.get("last_checkpoint_step") == updates:
            manifest_epoch = manifest.get("last_checkpoint_epoch")
            if (
                isinstance(manifest_epoch, (int, float))
                and not isinstance(manifest_epoch, bool)
                and not math.isclose(float(manifest_epoch), float(epochs), rel_tol=0.0, abs_tol=1e-9)
            ):
                raise CampaignError("run manifest checkpoint epoch differs from last.pt")
    else:
        # Progress/history rows are written before their matching atomic
        # checkpoint. They are replayable work, never accepted science.
        reported_updates = max(
            int(status.get("global_step", 0) or 0),
            int(progress.get("completed_updates", 0) or 0),
            int(manifest.get("last_checkpoint_step", 0) or 0),
        )
        if reported_updates > 0:
            raise CampaignError("run reports committed updates but has no last.pt checkpoint")
        epochs = 0.0
    if (
        isinstance(epochs, bool)
        or not isinstance(epochs, (int, float))
        or not math.isfinite(float(epochs))
        or epochs < 0
    ):
        raise CampaignError("run progress does not expose a trustworthy completed-update epoch count")
    if epochs > MAX_TASK_EPOCHS + 1e-9:
        raise CampaignError("run lineage exceeded the per-child 100-epoch cap")
    return float(epochs)


def _load_campaign_checkpoint(
    run_dir: Path, *, manifest: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Read the owned recovery cursor using PyTorch's restricted CPU loader."""
    checkpoint_path = run_dir / "checkpoints" / "last.pt"
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise CampaignError(f"campaign recovery checkpoint is missing or unsafe: {checkpoint_path}")
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    try:
        import torch

        safe_symbols: list[Any] = [torch._C._nn.gelu]
        try:
            from neuralop.layers.spectral_convolution import SpectralConv
        except ImportError:
            pass
        else:
            safe_symbols.append(SpectralConv)
        with torch.serialization.safe_globals(safe_symbols):
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise CampaignError(f"cannot restricted-load campaign recovery checkpoint: {error}") from error
    checkpoint = _mapping(payload, "campaign recovery checkpoint")
    global_step = checkpoint.get("global_step")
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
        raise CampaignError("campaign recovery checkpoint has an invalid global_step")
    manifest_record = (
        dict(manifest)
        if manifest is not None
        else _read_json(run_dir / "run_manifest.json", "run manifest")
    )
    config_sha = manifest_record.get("config_sha256")
    if not isinstance(config_sha, str) or checkpoint.get("config_sha256") != config_sha:
        raise CampaignError("recovery checkpoint config binding differs from its run manifest")
    parent_run = manifest_record.get("parent_run")
    checkpoint_source = checkpoint.get("source_run")
    if not isinstance(parent_run, str) or not isinstance(checkpoint_source, str):
        raise CampaignError("recovery checkpoint is missing its SOURCE run binding")
    if Path(parent_run).resolve() != Path(checkpoint_source).resolve():
        raise CampaignError("recovery checkpoint SOURCE binding differs from its run manifest")
    return checkpoint


def _steps_per_epoch_from_history(run_dir: Path, checkpoint_step: int) -> int:
    """Infer the fixed update count from the checkpoint-matching history row."""
    history_path = run_dir / "metrics" / "history.jsonl"
    if not history_path.is_file():
        return 0
    rows: list[tuple[int, int]] = []
    try:
        with history_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                step = row.get("step")
                epoch = row.get("epoch")
                if (
                    isinstance(step, int)
                    and not isinstance(step, bool)
                    and step >= 0
                    and isinstance(epoch, int)
                    and not isinstance(epoch, bool)
                    and epoch > 0
                ):
                    rows.append((step, epoch))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise CampaignError(f"cannot infer steps_per_epoch from training history: {error}") from error
    matches = [epoch for step, epoch in rows if step == checkpoint_step]
    if not matches or len(set(matches)) != 1:
        return 0
    epoch = matches[0]
    if checkpoint_step % epoch:
        return 0
    return checkpoint_step // epoch


def _run_history_epoch(run_dir: Path) -> float:
    """Return the latest optimizer epoch recorded before checkpoint commit."""
    history_path = run_dir / "metrics" / "history.jsonl"
    if not history_path.is_file():
        return 0.0
    last_step = 0
    last_epoch: float | None = None
    try:
        with history_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                step = row.get("step")
                if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                    raise CampaignError("training history contains an invalid optimizer step")
                last_step = max(last_step, step)
                epoch = row.get("epoch")
                if (
                    isinstance(epoch, (int, float))
                    and not isinstance(epoch, bool)
                    and math.isfinite(float(epoch))
                    and epoch >= 0
                ):
                    last_epoch = max(last_epoch or 0.0, float(epoch))
        status_path = run_dir / "status.json"
        progress_path = run_dir / "progress.json"
        status = _read_json(status_path, "run status") if status_path.is_file() else {}
        progress = _read_json(progress_path, "run progress") if progress_path.is_file() else {}
        step_values = {
            int(value)
            for value in (status.get("steps_per_epoch"), progress.get("steps_per_epoch"))
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
        }
        if len(step_values) > 1:
            raise CampaignError("run status and progress disagree on steps_per_epoch")
        steps = next(iter(step_values), 0)
        if steps <= 0 and last_step > 0:
            checkpoint_path = run_dir / "checkpoints" / "last.pt"
            if checkpoint_path.is_file() and not checkpoint_path.is_symlink():
                checkpoint = _load_campaign_checkpoint(run_dir)
                steps = _steps_per_epoch_from_history(run_dir, checkpoint["global_step"])
        if steps <= 0 and last_epoch is not None and last_epoch > 0:
            return float(last_epoch)
        if steps <= 0:
            raise CampaignError("run history has no trustworthy steps_per_epoch")
        return last_step / steps
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise CampaignError(f"cannot read training history cursor: {error}") from error


def _training_epoch_observation(
    run_dir: Path, *, start_epoch: float, horizon: int
) -> dict[str, Any]:
    """Separate accepted checkpoint exposure from work that may be replayed."""
    accepted_end = _run_epoch(run_dir)
    history_end = _run_history_epoch(run_dir)
    return {
        "run_dir": str(run_dir.resolve()),
        "start_epoch": start_epoch,
        "accepted_end_epoch": accepted_end,
        "observed_history_end_epoch": history_end,
        "epoch_exposure": max(0.0, accepted_end - start_epoch),
        "uncheckpointed_epoch_exposure_lower_bound": max(0.0, history_end - accepted_end),
        "training_complete": accepted_end >= horizon,
        "checkpoint_interval_epochs": 1,
    }


def _calibration_observation(run_dir: Path, *, reserve_seconds: float) -> dict[str, Any]:
    """Read only per-run calibration timing receipts; missing receipts use reserve."""
    fidelity_path = run_dir / "artifacts" / "fidelity_budget_reference.json"
    coherence_path = run_dir / "artifacts" / "coherence_calibration.json"
    try:
        if not fidelity_path.is_file() or not coherence_path.is_file():
            raise FileNotFoundError("one or more calibration timing artifacts are missing")
        fidelity = _read_json(fidelity_path, "fidelity calibration artifact")
        coherence = _read_json(coherence_path, "coherence calibration artifact")
        fidelity_seconds = fidelity.get("calibration_seconds")
        if (
            isinstance(fidelity_seconds, bool)
            or not isinstance(fidelity_seconds, (int, float))
            or not math.isfinite(float(fidelity_seconds))
            or fidelity_seconds < 0
        ):
            raise CampaignError("fidelity calibration artifact has no finite duration")
        coherence_mode = coherence.get("mode", "none")
        coherence_seconds = coherence.get("calibration_seconds", 0.0)
        if (
            isinstance(coherence_seconds, bool)
            or not isinstance(coherence_seconds, (int, float))
            or not math.isfinite(float(coherence_seconds))
            or coherence_seconds < 0
        ):
            raise CampaignError("coherence calibration artifact has no finite duration")
        if coherence_mode != "none" and "calibration_seconds" not in coherence:
            raise FileNotFoundError("coherence calibration duration was not recorded")
        return {
            "status": "measured_artifacts",
            "fidelity_seconds": float(fidelity_seconds),
            "coherence_seconds": float(coherence_seconds),
            "seconds": float(fidelity_seconds + coherence_seconds),
            "reserve_seconds": reserve_seconds,
        }
    except (OSError, CampaignError) as error:
        return {
            "status": "reserve_fallback",
            "reason": str(error),
            "fidelity_seconds": None,
            "coherence_seconds": None,
            "seconds": reserve_seconds,
            "reserve_seconds": reserve_seconds,
        }


def _verify_resume_run(root: Path, task_id: str, arm: str, case_dir: Path, config_path: Path, config: Mapping[str, Any], source_binding: Mapping[str, Any], run_arg: Path) -> tuple[Path, int]:
    if run_arg.is_symlink():
        raise CampaignError("resume run path may not be a symlink")
    run_dir = run_arg.resolve(strict=True)
    campaign_runs = (root / "runs").resolve(strict=True)
    if not run_dir.is_relative_to(campaign_runs):
        raise CampaignError("resume is limited to a child run owned by this campaign root")
    run_manifest = _read_json(run_dir / "run_manifest.json", "resume run manifest")
    resolved_config = _read_yaml(run_dir / "resolved_config.yaml", "resume resolved config")
    if run_manifest.get("case") != config.get("case") or run_manifest.get("stage") != "post_training":
        raise CampaignError("resume run has a different case or training stage")
    if _canonical_json(resolved_config) != _canonical_json(dict(config)):
        raise CampaignError("resume config differs from the pinned child config")
    if run_manifest.get("config_sha256") != _config_digest(config):
        raise CampaignError("resume run manifest does not bind the pinned child config")
    if Path(str(run_manifest.get("parent_run", ""))).resolve() != Path(
        str(source_binding["source_run"])
    ).resolve():
        raise CampaignError("resume run parent SOURCE differs from the frozen task")
    expected_experiment = f"_CostAwareLPC/{root.name}/{task_id}/{arm}"
    if config.get("output", {}).get("experiment_name") != expected_experiment:
        raise CampaignError("resume config experiment name differs from its task/arm")
    expected_parent = campaign_runs / task_id / arm
    if run_dir.parent != expected_parent:
        raise CampaignError("resume run is not in its exact frozen task/arm lineage")
    checkpoint = run_dir / "checkpoints/last.pt"
    if not checkpoint.is_file():
        raise CampaignError("resume run has no recovery checkpoint")
    checkpoint_payload = _load_campaign_checkpoint(run_dir, manifest=run_manifest)
    if Path(checkpoint_payload["source_run"]).resolve() != Path(
        str(source_binding["source_run"])
    ).resolve():
        raise CampaignError("resume checkpoint parent SOURCE differs from the frozen task")
    source_hashes = checkpoint_payload.get("source_hashes")
    if (
        not isinstance(source_hashes, Mapping)
        or source_hashes.get("checkpoint") != source_binding.get("source_checkpoint_sha256")
    ):
        raise CampaignError("resume checkpoint SOURCE hash differs from the frozen task")
    completed_epochs = _run_epoch(run_dir, checkpoint_payload=checkpoint_payload)
    horizon = int(config["optimization"]["epochs"])
    if completed_epochs >= horizon:
        raise CampaignError("resume run has already reached its frozen epoch horizon")
    return run_dir, horizon - completed_epochs


def _existing_output(config: Mapping[str, Any], case_dir: Path, root: Path) -> bool:
    experiment = str(config.get("output", {}).get("experiment_name", ""))
    if not experiment:
        raise CampaignError("prepared config has no output experiment name")
    campaign_runs = (root / "runs").resolve(strict=True)
    external_link = case_dir / "runs" / "_CostAwareLPC" / root.name
    if external_link.resolve(strict=True) != campaign_runs:
        raise CampaignError("case run link does not resolve to this campaign's owned run root")
    prefix = Path("_CostAwareLPC") / root.name
    relative = Path(experiment).relative_to(prefix)
    target = campaign_runs / relative
    if not target.resolve().is_relative_to(campaign_runs):
        raise CampaignError(f"prepared output path escapes the campaign root: {target}")
    cursor = campaign_runs
    for part in target.relative_to(campaign_runs).parts:
        cursor = cursor / part
        if os.path.lexists(cursor):
            if cursor.is_symlink() or not cursor.is_dir():
                raise CampaignError(f"campaign output path has an unsafe component: {cursor}")
            if cursor.stat().st_uid != os.getuid() or cursor.stat().st_dev != campaign_runs.stat().st_dev:
                raise CampaignError(f"campaign output path ownership/device mismatch: {cursor}")
    return target.exists() and any(target.iterdir())


def _run_foreground(
    root: Path,
    *,
    command: Sequence[str],
    task_id: str,
    arm: str,
    action: str,
    gpu: int,
    preflight: Mapping[str, Any],
    max_seconds: float,
    science_epochs: float = 0,
    aux_epochs: float = 0,
    stop_loss_reserve_epochs: float = 0,
    is_profile: bool = False,
    output_artifact: Path | None = None,
    epoch_observer: Callable[[], Mapping[str, Any] | None] | None = None,
) -> dict[str, Any]:
    _require_environment()
    admission = _acquire_admission(root)
    gpu_admission = None

    def release_admission() -> None:
        if gpu_admission is not None and not gpu_admission.closed:
            gpu_admission.close()
        if not admission.closed:
            admission.close()

    attempt_id = uuid.uuid4().hex
    started = CLI_START_MONOTONIC
    created = _utc_now()
    try:
        totals = _resource_totals(root)
        _check_no_active_process(root)
        gpu_admission = _acquire_physical_gpu_lock(gpu)
        gpu_receipt = _gpu_record(gpu, preflight)
    except BaseException:
        release_admission()
        raise
    process_limit = PROCESS_SECONDS_CAP if action == "evaluate" else PROCESS_SECONDS_CAP - CLOSEOUT_RESERVE_SECONDS
    if totals["process_seconds"] + max_seconds > process_limit:
        release_admission()
        raise CampaignError("requested process could consume the reserved closeout or exceed the 16-hour cap")
    if action == "evaluate" and totals["closeout_seconds"] + max_seconds > CLOSEOUT_RESERVE_SECONDS:
        release_admission()
        raise CampaignError("evaluation and closeout exceed their one-hour reserve")
    if totals["science_epochs"] + science_epochs > SCIENCE_EPOCHS_CAP:
        release_admission()
        raise CampaignError("science exposure budget of 800 epochs would be exceeded")
    if totals["aux_epochs"] + aux_epochs + stop_loss_reserve_epochs > AUX_EPOCHS_CAP:
        release_admission()
        raise CampaignError("auxiliary exposure budget of 50 epochs would be exceeded")
    if (
        totals["science_epochs"] + totals["aux_epochs"] + science_epochs
        + aux_epochs + stop_loss_reserve_epochs > TOTAL_EPOCHS_CAP
    ):
        release_admission()
        raise CampaignError("combined exposure budget of 850 epochs would be exceeded")
    profile_remaining = PROFILE_SECONDS_CAP - totals["profile_seconds"]
    profile_reservation = max_seconds if is_profile else 0.0
    calibration_reservation = (
        FRESH_LAUNCH_CALIBRATION_RESERVE_SECONDS if action == "launch" else 0.0
    )
    if profile_reservation + calibration_reservation > profile_remaining:
        release_admission()
        raise CampaignError("profiling/calibration budget of 90 minutes would be exceeded")

    try:
        log_dir = root / "tasks" / f"{task_id}_logs"
        if not log_dir.exists():
            log_dir.mkdir(mode=0o750)
        if not log_dir.resolve().is_relative_to(root):
            raise CampaignError("campaign task log directory escaped the root")
        env = _campaign_environment(root, gpu, started_monotonic=started)
    except BaseException:
        release_admission()
        raise
    log_path = log_dir / f"{attempt_id}_{action}_{arm}.log"
    active_payload = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "task_id": task_id,
        "arm": arm,
        "action": action,
        "pid": None,
        "process_group": None,
        "command": list(command),
        "command_sha256": hashlib.sha256(_canonical_json(list(command)).encode()).hexdigest(),
        "log_path": str(log_path),
        "gpu": gpu_receipt,
        "started_utc": created,
        "reserved_seconds": max_seconds,
        "reserved_science_epochs": science_epochs,
        "reserved_aux_epochs": aux_epochs + stop_loss_reserve_epochs,
        "stop_loss_reserve_epochs": stop_loss_reserve_epochs,
        "reserved_profile_seconds": profile_reservation + calibration_reservation,
        "reserved_calibration_seconds": calibration_reservation,
        "reserved_closeout_seconds": max_seconds if action == "evaluate" else 0.0,
    }
    start_ledger = {
        **active_payload,
        "status": "running",
        "reserved_process_seconds": max_seconds,
        "reserved_profile_seconds": profile_reservation + calibration_reservation,
        "reserved_calibration_seconds": calibration_reservation,
        "reserved_closeout_seconds": max_seconds if action == "evaluate" else 0.0,
    }
    try:
        _append_ledger(root, start_ledger)
    except BaseException:
        release_admission()
        raise
    previous_handler = signal.getsignal(signal.SIGINT)
    child: subprocess.Popen[bytes] | None = None
    handler_installed = False

    def forward_sigint(_signum: int, _frame: Any) -> None:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass

    try:
        signal.signal(signal.SIGINT, forward_sigint)
        handler_installed = True
        with log_path.open("xb") as log:
            child = subprocess.Popen(
                list(command),
                cwd=REPO_ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            active_payload["pid"] = child.pid
            active_payload["process_group"] = child.pid
            _atomic_json(root / ACTIVE_PATH, active_payload)
            deadline = started + max_seconds
            while child.poll() is None:
                if time.monotonic() >= deadline:
                    _terminate_owned_group(child)
                    break
                time.sleep(0.5)
            return_code = int(child.returncode or 0)
    except BaseException:
        if child is not None and child.poll() is None:
            try:
                _terminate_owned_group(child)
            except CampaignError:
                pass
        raise
    finally:
        if handler_installed:
            signal.signal(signal.SIGINT, previous_handler)
        elapsed = time.monotonic() - started
        if child is not None and _process_group_members(child.pid):
            try:
                _terminate_owned_group(child)
            except CampaignError:
                pass
        group_members = _process_group_members(child.pid) if child is not None else []
        if child is not None and child.poll() is not None and not group_members:
            observation: Mapping[str, Any] | None = None
            charged_science_epochs = float(science_epochs)
            charge_basis = "conservative_reserved_upper_bound"
            calibration = None
            charged_stop_loss_epochs = 0.0
            if epoch_observer is not None:
                try:
                    observation = epoch_observer()
                    if observation is not None:
                        observed_exposure = float(observation["epoch_exposure"])
                        if not math.isfinite(observed_exposure) or observed_exposure < 0:
                            raise ValueError("observed epoch exposure must be finite and nonnegative")
                        charged_science_epochs = observed_exposure
                        charge_basis = "observed_completed_update_delta"
                except (CampaignError, KeyError, TypeError, ValueError, OSError) as error:
                    observation = {"status": "unavailable", "reason": str(error)}
            if action == "launch":
                run_value = observation.get("run_dir") if observation is not None else None
                if run_value is not None:
                    calibration = _calibration_observation(
                        Path(str(run_value)),
                        reserve_seconds=calibration_reservation,
                    )
                else:
                    calibration = {
                        "status": "reserve_fallback",
                        "reason": "campaign run directory or calibration artifacts are unavailable",
                        "fidelity_seconds": None,
                        "coherence_seconds": None,
                        "seconds": calibration_reservation,
                        "reserve_seconds": calibration_reservation,
                    }
            incomplete_training = action in {"launch", "resume"} and (
                observation is None
                or observation.get("training_complete") is not True
            )
            if incomplete_training and stop_loss_reserve_epochs > 0:
                observed_loss = 0.0
                if observation is not None:
                    try:
                        observed_loss = float(
                            observation.get("uncheckpointed_epoch_exposure_lower_bound", 0.0)
                        )
                    except (TypeError, ValueError):
                        observed_loss = 0.0
                if not math.isfinite(observed_loss) or observed_loss < 0:
                    observed_loss = stop_loss_reserve_epochs
                charged_stop_loss_epochs = max(stop_loss_reserve_epochs, observed_loss)
            charged_aux_epochs = aux_epochs + charged_stop_loss_epochs
            charged_calibration_seconds = (
                float(calibration["seconds"]) if calibration is not None else 0.0
            )
            _append_ledger(
                root,
                {
                    **active_payload,
                    "status": (
                        "completed"
                        if child.returncode == 0 and not incomplete_training
                        else "failed_or_stopped"
                    ),
                    "return_code": child.returncode,
                    "finished_utc": _utc_now(),
                    "charged_process_seconds": elapsed,
                    "charged_profile_seconds": (
                        elapsed if is_profile else charged_calibration_seconds
                    ),
                    "charged_calibration_seconds": charged_calibration_seconds,
                    "calibration_observation": calibration,
                    "charged_closeout_seconds": elapsed if action == "evaluate" else 0.0,
                    "charged_science_epochs": charged_science_epochs,
                    "actual_science_epoch_observation": observation,
                    "science_epoch_charge_basis": charge_basis,
                    "charged_stop_loss_epochs": charged_stop_loss_epochs,
                    "charged_aux_epochs": charged_aux_epochs,
                },
            )
            (root / ACTIVE_PATH).unlink(missing_ok=True)
        elif child is None:
            _append_ledger(
                root,
                {
                    **active_payload,
                    "status": "failed_to_start",
                    "finished_utc": _utc_now(),
                    "charged_process_seconds": elapsed,
                    "charged_profile_seconds": elapsed if is_profile else 0.0,
                    "charged_calibration_seconds": 0.0,
                    "calibration_observation": None,
                    "charged_closeout_seconds": elapsed if action == "evaluate" else 0.0,
                    "charged_science_epochs": 0.0,
                    "actual_science_epoch_observation": None,
                    "science_epoch_charge_basis": "no_child_started",
                    "charged_aux_epochs": 0.0,
                },
            )
        release_admission()
    tail = ""
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-30:])
    if return_code != 0:
        raise CampaignError(
            f"{action} child exited with code {return_code}; log={log_path}\n{tail}"
        )
    result: dict[str, Any] = {
        "attempt_id": attempt_id,
        "task_id": task_id,
        "arm": arm,
        "action": action,
        "return_code": return_code,
        "elapsed_seconds": elapsed,
        "charged_science_epochs": charged_science_epochs,
        "actual_science_epoch_observation": observation,
        "science_epoch_charge_basis": charge_basis,
        "charged_stop_loss_epochs": charged_stop_loss_epochs,
        "charged_calibration_seconds": charged_calibration_seconds,
        "calibration_observation": calibration,
        "charged_aux_epochs": charged_aux_epochs,
        "gpu": gpu_receipt,
        "log_path": str(log_path),
        "output_artifact": str(output_artifact) if output_artifact else None,
    }
    if action in {"launch", "resume"} and (
        observation is None or observation.get("training_complete") is not True
    ):
        raise CampaignError(
            f"training child exited before its frozen epoch horizon; resume the same run; log={log_path}\n{tail}"
        )
    return result


def _profile(root: Path, preflight: Mapping[str, Any], task_id: str, *, config_path: Path | None, gpu: int, max_seconds: float | None, execute: bool) -> dict[str, Any]:
    task, _inventory_payload = _task(root, task_id)
    _verify_identity_receipt(root, task)
    _inspect_task(root, task_id)
    config_path = config_path or _config_path(root, task_id, "dense")
    try:
        config_path = config_path.resolve(strict=True)
    except OSError as error:
        raise CampaignError(f"profiling config is missing: {config_path}") from error
    if not config_path.is_relative_to((root / "configs").resolve(strict=True)):
        raise CampaignError("profiling config must be an immutable campaign-owned config")
    selected_arm = None
    for candidate_arm in ("dense", "heuristic"):
        try:
            candidate_path = _config_path(root, task_id, candidate_arm).resolve(strict=True)
        except CampaignError:
            continue
        if config_path == candidate_path:
            selected_arm = candidate_arm
            break
    if selected_arm is None:
        raise CampaignError("profiling may use only a predeclared dense or heuristic task config")
    config = _read_yaml(config_path, "profiling config")
    binding = _source_binding(root, task, config_path, config)
    _verify_frozen_config_binding(root, task_id, selected_arm, binding)
    requested = _parse_duration(
        max_seconds,
        maximum=min(PROFILE_SECONDS_CAP, PROFILE_SECONDS_CAP - _resource_totals(root)["profile_seconds"]),
        label="profile max-seconds",
    )
    task_contract_path = root / "tasks" / f"{task_id}_task_contract.json"
    profile_path = root / "profiles" / f"{task_id}_profile.json"
    if profile_path.exists():
        raise CampaignError(f"an immutable profile already exists: {profile_path}")
    command = _child_script_command(
        root,
        PROFILE_SCRIPT,
        [
            "--root", str(root),
            "--task", task_id,
            "--config", str(config_path),
            "--gpu", str(gpu),
            "--max-seconds", str(requested),
        ],
    )
    if not execute:
        return {
            "action": "profile",
            "execute_required": True,
            "task_id": task_id,
            "config": str(config_path),
            "source_binding": binding,
            "command": shlex.join(command),
            "reserved_aux_epochs": PROFILE_AUX_RESERVE_EPOCHS,
            "max_seconds": requested,
        }
    _require_environment()
    result = _run_foreground(
        root,
        command=command,
        task_id=task_id,
        arm="profile",
        action="profile",
        gpu=gpu,
        preflight=preflight,
        max_seconds=requested,
        aux_epochs=PROFILE_AUX_RESERVE_EPOCHS,
        is_profile=True,
        output_artifact=profile_path,
    )
    if not profile_path.is_file() or not task_contract_path.is_file():
        raise CampaignError(
            "profile child returned success without its frozen task-contract/profile artifacts"
        )
    profile = _read_json(profile_path, "measured cost profile")
    if profile.get("profile_split", "TRAIN").upper() != "TRAIN" or profile.get("noise_split", "TRAIN").upper() != "TRAIN":
        raise CampaignError("profile artifact includes a non-TRAIN measurement")
    raw_profile_path = root / "profiles" / f"{task_id}_raw_profile.json"
    if not raw_profile_path.is_file():
        raise CampaignError("profile child returned success without raw exposure accounting")
    raw_profile = _read_json(raw_profile_path, "raw measured profile")
    raw_exposure = raw_profile.get("auxiliary_epoch_exposures")
    if (
        isinstance(raw_exposure, bool)
        or not isinstance(raw_exposure, (int, float))
        or not math.isfinite(float(raw_exposure))
        or raw_exposure < 0
    ):
        raise CampaignError("raw profile auxiliary exposure count is invalid")
    result["charged_aux_epochs"] = float(raw_exposure)
    _append_ledger(
        root,
        {
            **result,
            "status": "completed",
            "charged_process_seconds": result["elapsed_seconds"],
            "charged_profile_seconds": result["elapsed_seconds"],
            "charged_science_epochs": 0,
            "charged_aux_epochs": float(raw_exposure),
            "aux_epoch_charge_basis": "observed_restored_optimizer_actions",
            "finished_utc": _utc_now(),
        },
    )
    strategy_config_path = _config_path(root, task_id, "strategy")
    strategy_config = _read_yaml(strategy_config_path, "profile-generated strategy config")
    strategy_binding = _source_binding(root, task, strategy_config_path, strategy_config)
    _write_immutable_json(
        root / "tasks" / f"{task_id}_strategy_config_binding.json",
        {
            "schema_version": 1,
            "task_id": task_id,
            "arm": "strategy",
            "path": str(strategy_config_path),
            "sha256": strategy_binding["config_sha256"],
            "source_binding": strategy_binding,
        },
    )
    result["profile_path"] = str(profile_path)
    result["task_contract_path"] = str(task_contract_path)
    result["source_identity"] = binding["source_checkpoint_sha256"]
    return result


def _plan(root: Path, task_id: str, *, profile_path: Path | None) -> dict[str, Any]:
    task_contract_path = root / "tasks" / f"{task_id}_task_contract.json"
    profile_path = profile_path or root / "profiles" / f"{task_id}_profile.json"
    if not task_contract_path.is_file() or not profile_path.is_file():
        raise CampaignError("plan requires the measured task contract and profile artifacts")
    contract_record = _read_json(task_contract_path, "calibrated task contract")
    contract_mapping = contract_record.get("task_contract", contract_record)
    if not isinstance(contract_mapping, Mapping):
        raise CampaignError("calibrated task contract has no task_contract mapping")
    profile_record = _read_json(profile_path, "measured cost profile")
    profile_mapping = profile_record.get("cost_profile", profile_record)
    if not isinstance(profile_mapping, Mapping):
        raise CampaignError("profile artifact has no cost profile mapping")
    if str(contract_mapping.get("calibration_status", "bound")) != "bound":
        raise CampaignError("planning is blocked until SOURCE TRAIN calibration is bound")
    if profile_mapping.get("profile_split", "TRAIN").upper() != "TRAIN" or profile_mapping.get("noise_split", "TRAIN").upper() != "TRAIN":
        raise CampaignError("planning accepts TRAIN-only cost and noise profiles")
    if str(contract_mapping.get("task_hash")) != str(profile_mapping.get("task_hash")):
        raise CampaignError("profile task hash does not match the calibrated task contract")
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    from phycoflow_reconstruction.training.execution_strategy import (
        CostProfile,
        TaskContract,
        select_execution_plan,
    )

    try:
        contract = TaskContract.from_mapping(contract_mapping)
        profile = CostProfile.from_mapping(profile_mapping)
        plan = select_execution_plan(contract, profile)
    except (TypeError, ValueError, KeyError) as error:
        raise CampaignError(f"cannot select a qualified execution plan: {error}") from error
    plan_path = root / "plans" / f"{task_id}_plan_decision.json"
    payload = plan.to_dict()
    if plan_path.exists():
        recorded = _read_json(plan_path, "profile-selected immutable plan")
        if _canonical_json(recorded) != _canonical_json(payload):
            raise CampaignError("recomputed plan differs from the profile-selected immutable plan")
        digest = _sha256(plan_path)
    else:
        digest = _write_immutable_json(plan_path, payload)
    return {
        "task_id": task_id,
        "plan_id": plan.plan_id,
        "no_action": plan.no_action,
        "reason": plan.selected_reason,
        "plan_path": str(plan_path),
        "plan_sha256": digest,
        "rejection_reasons": dict(plan.rejection_reasons),
    }


def _run_or_resume(
    root: Path,
    preflight: Mapping[str, Any],
    task_id: str,
    arm: str,
    *,
    gpu: int,
    max_seconds: float | None,
    run_arg: Path | None,
    execute: bool,
) -> dict[str, Any]:
    task, _inventory_payload = _task(root, task_id)
    _verify_identity_receipt(root, task)
    case_dir = REPO_ROOT / "cases" / str(task.get("case", ""))
    links = _ensure_repository_links(root, case_dir)
    task_config_path = _config_path(root, task_id, arm)
    config = _read_yaml(task_config_path, f"{task_id} {arm} config")
    binding = _source_binding(root, task, task_config_path, config)
    _verify_frozen_config_binding(root, task_id, arm, binding)
    horizon = int(config.get("optimization", {}).get("epochs", 0))
    if not 1 <= horizon <= MAX_TASK_EPOCHS:
        raise CampaignError("training child exceeds the 100-epoch maximum")
    checkpointing = _mapping(config.get("checkpointing", {}), "training checkpoint config")
    checkpoint_interval = int(checkpointing.get("every_epochs", 0) or 0)
    if checkpoint_interval != 1:
        raise CampaignError("campaign recovery accounting requires a checkpoint every epoch")
    expected_experiment = f"_CostAwareLPC/{root.name}/{task_id}/{arm}"
    if config.get("output", {}).get("experiment_name") != expected_experiment:
        raise CampaignError("prepared config output path differs from its frozen task/arm identity")
    if arm == "strategy":
        contract = _read_json(root / "tasks" / f"{task_id}_task_contract.json", "strategy task contract")
        profile = _read_json(root / "profiles" / f"{task_id}_profile.json", "strategy cost profile")
        decision = _read_json(root / "plans" / f"{task_id}_plan_decision.json", "strategy plan decision")
        selected = decision.get("plan", decision)
        if not isinstance(selected, Mapping) or selected.get("task_hash") != contract.get("task_hash"):
            raise CampaignError("strategy config plan is not bound to its calibrated task contract")
        if not task_config_path.is_file():
            raise CampaignError("profile script did not write the immutable strategy config")
        strategy_cfg = config.get("optimization", {}).get("execution_strategy")
        if not isinstance(strategy_cfg, Mapping) or strategy_cfg.get("plan", {}).get("plan_id") != selected.get("plan_id"):
            raise CampaignError("strategy config does not bind the current measured plan")
        if profile.get("task_hash") != contract.get("task_hash"):
            raise CampaignError("strategy profile task hash differs from the task contract")
    if run_arg is None:
        if _existing_output(config, case_dir, root):
            raise CampaignError("output path already has a run; use resume with its exact run directory")
        remaining_epochs = horizon
        resume_dir = None
        start_epoch = 0.0
    else:
        resume_dir, remaining_epochs = _verify_resume_run(
            root, task_id, arm, case_dir, task_config_path, config, binding, run_arg
        )
        start_epoch = _run_epoch(resume_dir)
    command_args = [
        "_train",
        "--config",
        str(task_config_path),
        "--case-dir",
        str(case_dir),
    ]
    if resume_dir is not None:
        command_args.extend(["--resume", str(resume_dir)])
    command = _child_script_command(
        root, Path(__file__), command_args, case_dir=case_dir
    )
    allowed_seconds = PROCESS_SECONDS_CAP - CLOSEOUT_RESERVE_SECONDS - _resource_totals(root)["process_seconds"]
    requested_seconds = _parse_duration(max_seconds, maximum=allowed_seconds, label="max-seconds")
    if not execute:
        return {
            "action": "resume" if run_arg is not None else "launch",
            "execute_required": True,
            "task_id": task_id,
            "arm": arm,
            "config": str(task_config_path),
            "config_sha256": _sha256(task_config_path),
            "source_binding": binding,
            "repository_links": links,
            "remaining_epochs": remaining_epochs,
            "command": shlex.join(command),
            "max_seconds": requested_seconds,
        }
    def observe_epoch_exposure() -> Mapping[str, Any] | None:
        observed_run = resume_dir
        if observed_run is None:
            try:
                observed_run = _discover_run_directory(root, task_id, arm, case_dir)
            except CampaignError:
                return None
        return _training_epoch_observation(
            observed_run,
            start_epoch=start_epoch,
            horizon=horizon,
        )

    return _run_foreground(
        root,
        command=command,
        task_id=task_id,
        arm=arm,
        action="resume" if run_arg is not None else "launch",
        gpu=gpu,
        preflight=preflight,
        max_seconds=requested_seconds,
        science_epochs=remaining_epochs,
        stop_loss_reserve_epochs=STOP_LOSS_RESERVE_EPOCHS,
        epoch_observer=observe_epoch_exposure,
    )


def _evaluate(
    root: Path,
    preflight: Mapping[str, Any],
    task_id: str,
    arm: str,
    run_arg: Path,
    *,
    gpu: int,
    max_seconds: float | None,
    max_samples: int,
    panel_kind: str,
    checkpoint: str,
    execute: bool,
) -> dict[str, Any]:
    task, _inventory_payload = _task(root, task_id)
    _verify_identity_receipt(root, task)
    config_path = _config_path(root, task_id, arm)
    config = _read_yaml(config_path, f"{task_id} {arm} config")
    binding = _source_binding(root, task, config_path, config)
    _verify_frozen_config_binding(root, task_id, arm, binding)
    run_dir = run_arg.resolve(strict=True)
    if not run_dir.is_relative_to((root / "runs").resolve(strict=True)):
        raise CampaignError("evaluation is limited to a run owned by this campaign")
    run_manifest = _read_json(run_dir / "run_manifest.json", "campaign run manifest")
    if run_manifest.get("case") != task.get("case") or run_manifest.get("stage") != "post_training":
        raise CampaignError("evaluation run has a different case or stage")
    if Path(str(run_manifest.get("parent_run", ""))).resolve() != Path(
        str(binding["source_run"])
    ).resolve():
        raise CampaignError("evaluation run parent SOURCE differs from the frozen task")
    expected_parent = (root / "runs" / task_id / arm).resolve(strict=True)
    if run_dir.parent != expected_parent:
        raise CampaignError("evaluation run is not in its exact frozen task/arm lineage")
    resolved = _read_yaml(run_dir / "resolved_config.yaml", "campaign run config")
    if resolved.get("evaluation", {}).get("split") != "validation":
        raise CampaignError("only validation-split evaluation is permitted")
    if max_samples not in (32, 64, 96, 128):
        raise CampaignError("full A+B+T evaluation supports only exact sample groups of 32")
    if panel_kind not in {"uniform", "disjoint"}:
        raise CampaignError("evaluation panel must be uniform or disjoint")
    if checkpoint not in {"best", "last", "epoch_50", "epoch_60", "epoch_70", "epoch_80", "epoch_90", "epoch_100"}:
        raise CampaignError("checkpoint must be best, last, or an explicit monitored epoch checkpoint")
    panel_suffix = "_disjoint" if panel_kind == "disjoint" else ""
    output_dir = root / "analysis" / task_id / arm / run_dir.name / f"full_abt_{checkpoint}{panel_suffix}"
    if output_dir.exists():
        raise CampaignError(f"evaluation output already exists: {output_dir}")
    command = _child_script_command(
        root,
        EVALUATOR_SCRIPT,
        [
            "--run", str(run_dir),
            "--checkpoint", checkpoint,
            "--max-samples", str(max_samples),
            "--panel", panel_kind,
            "--device", "cuda:0",
            "--output-dir", str(output_dir),
        ],
    )
    requested_seconds = _parse_duration(max_seconds, maximum=EVALUATION_SECONDS_CAP, label="evaluation max-seconds")
    if not execute:
        return {
            "action": "evaluate",
            "execute_required": True,
            "task_id": task_id,
            "arm": arm,
            "run": str(run_dir),
            "split": "validation",
            "panel": panel_kind,
            "groups": max_samples // 32,
            "output_dir": str(output_dir),
            "source_binding": binding,
            "command": shlex.join(command),
            "max_seconds": requested_seconds,
        }
    result = _run_foreground(
        root,
        command=command,
        task_id=task_id,
        arm=arm,
        action="evaluate",
        gpu=gpu,
        preflight=preflight,
        max_seconds=requested_seconds,
        output_artifact=output_dir,
    )
    if not output_dir.exists():
        raise CampaignError("validation evaluator returned success without an output directory")
    result["output_dir"] = str(output_dir)
    return result


def _stop(root: Path, *, expected_pid: int | None = None) -> dict[str, Any]:
    active_path = root / ACTIVE_PATH
    if not active_path.is_file():
        raise CampaignError("no owned campaign process is active")
    active = _read_json(active_path, "active process receipt")
    pid = active.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        raise CampaignError("active process receipt has no valid PID")
    if expected_pid is not None and expected_pid != pid:
        raise CampaignError("requested PID assertion does not match the owned child")
    proc = Path(f"/proc/{pid}")
    if not proc.is_dir():
        raise CampaignError("active process is already absent; inspect its log and stale receipt")
    try:
        cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        pgid = os.getpgid(pid)
    except OSError as error:
        raise CampaignError(f"cannot verify active process ownership: {error}") from error
    if pgid != pid or str(Path(__file__).resolve()) not in cmdline or str(root) not in cmdline:
        raise CampaignError("active PID no longer matches the runner-owned child identity")
    os.killpg(pid, signal.SIGINT)
    return {
        "status": "SIGINT sent",
        "pid": pid,
        "task_id": active.get("task_id"),
        "arm": active.get("arm"),
        "log_path": active.get("log_path"),
    }


def _status(root: Path) -> dict[str, Any]:
    inventory = _inventory(root)
    totals = _resource_totals(root)
    active_path = root / ACTIVE_PATH
    active = _read_json(active_path, "active process receipt") if active_path.is_file() else None
    return {
        "root": str(root),
        "task_ids": list(_mapping(inventory["tasks"], "tasks")),
        "active_process": active,
        "resource_usage": totals,
        "resource_caps": {
            "science_epochs": SCIENCE_EPOCHS_CAP,
            "aux_epochs": AUX_EPOCHS_CAP,
            "total_epochs": TOTAL_EPOCHS_CAP,
            "process_seconds": PROCESS_SECONDS_CAP,
            "profile_seconds": PROFILE_SECONDS_CAP,
            "closeout_seconds": CLOSEOUT_RESERVE_SECONDS,
            "closeout_reserve_seconds": CLOSEOUT_RESERVE_SECONDS,
        },
    }


def _child(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--identities", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path)
    parsed, child_args = parser.parse_known_args(list(argv))
    if child_args and child_args[0] == "--":
        child_args = child_args[1:]
    root, _preflight = _verify_root(parsed.root, create_payload_dirs=False)
    receipt = parsed.identities.resolve(strict=True)
    if receipt != (root / IDENTITY_RECEIPT_PATH).resolve(strict=True):
        raise CampaignError("child identity receipt is not the campaign-owned receipt")
    if parsed.case_dir is not None and not parsed.case_dir.resolve(strict=True).is_relative_to(REPO_ROOT):
        raise CampaignError("child case directory escaped the repository")
    script = parsed.script.resolve(strict=True)
    if not script.is_relative_to(REPO_ROOT):
        raise CampaignError("child script escaped the repository")
    _register_child_identities(root)
    sys.argv = [str(script), *child_args]
    runpy.run_path(str(script), run_name="__main__")
    return 0


def _train_direct(config_path: Path, case_dir: Path, resume: Path | None) -> int:
    """Run the existing post-trainer from its exact resolved YAML mapping."""
    config = _read_yaml(config_path.resolve(strict=True), "frozen post-training config")
    case_dir = case_dir.resolve(strict=True)
    if not case_dir.is_relative_to(REPO_ROOT) or not (case_dir / "run.py").is_file():
        raise CampaignError("training case directory is outside the repository")
    if str(config.get("case", "")) != case_dir.name:
        raise CampaignError("frozen config case does not match the selected case directory")
    if config.get("stage") != "post_training":
        raise CampaignError("campaign training requires a post_training config")
    if config.get("source_run") is None or not Path(str(config["source_run"])).is_absolute():
        raise CampaignError("frozen config must pin an absolute SOURCE run path")
    if not Path(str(config.get("dataset", {}).get("path", ""))).is_absolute():
        raise CampaignError("frozen config must pin an absolute dataset path")
    from phycoflow_reconstruction.training.post_training import run_post_training

    run_dir = run_post_training(config, case_dir=case_dir, resume=resume)
    print(str(run_dir), flush=True)
    return 0


def _parse_gpu(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("physical GPU must be 0, 1, or 2") from error
    if not 0 <= parsed <= MAX_PHYSICAL_GPU:
        raise argparse.ArgumentTypeError("physical GPU must be 0, 1, or 2")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="write config-only task/capability/profile requests")
    inspect.add_argument("task")

    profile = subparsers.add_parser("profile", help="run bounded SOURCE calibration and profiling")
    profile.add_argument("task")
    profile.add_argument("--config", type=Path)
    profile.add_argument("--gpu", type=_parse_gpu, default=DEFAULT_GPU)
    profile.add_argument("--max-seconds", type=float)
    profile.add_argument("--execute", action="store_true")

    plan = subparsers.add_parser("plan", help="select a plan from a measured TRAIN-only profile")
    plan.add_argument("task")
    plan.add_argument("--profile", type=Path)

    for name in ("launch", "train"):
        launch = subparsers.add_parser(name, help="launch one bounded foreground task child")
        launch.add_argument("task")
        launch.add_argument("arm", choices=("dense", "heuristic", "strategy"))
        launch.add_argument("--gpu", type=_parse_gpu, default=DEFAULT_GPU)
        launch.add_argument("--max-seconds", type=float)
        launch.add_argument("--execute", action="store_true")

    resume = subparsers.add_parser("resume", help="resume the same frozen child run")
    resume.add_argument("task")
    resume.add_argument("arm", choices=("dense", "heuristic", "strategy"))
    resume.add_argument("--run", type=Path, required=True)
    resume.add_argument("--gpu", type=_parse_gpu, default=DEFAULT_GPU)
    resume.add_argument("--max-seconds", type=float)
    resume.add_argument("--execute", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="run fixed validation-only full A+B+T evaluation")
    evaluate.add_argument("task")
    evaluate.add_argument("arm", choices=("dense", "heuristic", "strategy"))
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--checkpoint", default="best")
    evaluate.add_argument("--max-samples", type=int, default=128, choices=(32, 64, 96, 128))
    evaluate.add_argument("--panel", choices=("uniform", "disjoint"), default="uniform")
    evaluate.add_argument("--gpu", type=_parse_gpu, default=DEFAULT_GPU)
    evaluate.add_argument("--max-seconds", type=float)
    evaluate.add_argument("--execute", action="store_true")

    stop = subparsers.add_parser("stop", help="send SIGINT only to the runner-owned foreground child")
    stop.add_argument("--pid", type=int, help="optional PID assertion for the active child")

    subparsers.add_parser("status", help="show resource ledger totals and owned process state")

    child = subparsers.add_parser("_child", help=argparse.SUPPRESS)
    child.add_argument("--identities", type=Path, required=True)
    child.add_argument("--script", type=Path, required=True)
    child.add_argument("--case-dir", type=Path)
    child.add_argument("child_args", nargs=argparse.REMAINDER)

    train_child = subparsers.add_parser("_train", help=argparse.SUPPRESS)
    train_child.add_argument("--config", type=Path, required=True)
    train_child.add_argument("--case-dir", type=Path, required=True)
    train_child.add_argument("--resume", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "_child":
            # Parent launcher injects --root before the subcommand; argparse still
            # stores it at the top-level namespace.
            return _child(
                [
                    "--root", str(args.root),
                    "--identities", str(args.identities),
                    "--script", str(args.script),
                    *(["--case-dir", str(args.case_dir)] if args.case_dir else []),
                    *args.child_args,
                ]
            )
        root, preflight = _verify_root(args.root, create_payload_dirs=True)
        if args.command == "inspect":
            result = _inspect_task(root, args.task)
        elif args.command == "profile":
            result = _profile(
                root,
                preflight,
                args.task,
                config_path=args.config,
                gpu=args.gpu,
                max_seconds=args.max_seconds,
                execute=args.execute,
            )
        elif args.command == "plan":
            result = _plan(root, args.task, profile_path=args.profile)
        elif args.command in {"launch", "train"}:
            result = _run_or_resume(
                root,
                preflight,
                args.task,
                args.arm,
                gpu=args.gpu,
                max_seconds=args.max_seconds,
                run_arg=None,
                execute=args.execute,
            )
        elif args.command == "resume":
            result = _run_or_resume(
                root,
                preflight,
                args.task,
                args.arm,
                gpu=args.gpu,
                max_seconds=args.max_seconds,
                run_arg=args.run,
                execute=args.execute,
            )
        elif args.command == "evaluate":
            result = _evaluate(
                root,
                preflight,
                args.task,
                args.arm,
                args.run,
                gpu=args.gpu,
                max_seconds=args.max_seconds,
                max_samples=args.max_samples,
                panel_kind=args.panel,
                checkpoint=args.checkpoint,
                execute=args.execute,
            )
        elif args.command == "stop":
            root, _preflight = _verify_root(args.root, create_payload_dirs=False)
            result = _stop(root, expected_pid=args.pid)
        elif args.command == "status":
            result = _status(root)
        elif args.command == "_train":
            return _train_direct(args.config, args.case_dir, args.resume)
        else:  # pragma: no cover - argparse restricts commands
            raise CampaignError(f"unsupported command {args.command}")
    except CampaignError as error:
        print(f"cost-aware campaign blocked: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
