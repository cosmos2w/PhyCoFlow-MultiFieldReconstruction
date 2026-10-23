#!/usr/bin/env python3
"""Prepare and operate the bounded fast-Lagrangian LPC campaign.

Preparation is read-only with respect to historical runs and writes campaign
payloads only below the verified /data2 root. Training commands are printed by
default; ``--execute`` runs the existing case CLI in the foreground.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = REPO_ROOT / "cases" / "turbulent_combustion"
SOURCE_CONFIG = (
    CASE_DIR / "runs/_MultiTask/11f_lagrangian_lr_warmup_dense_fast_100ep/"
    "20261017T170131Z_615167aa/resolved_config.yaml"
)
DEFAULT_CAMPAIGN_ROOT = Path(
    "/data2/wanglz/PhyCoFlow-MultiFieldReconstruction/experiments/LPCAccel_20261022_01a16674"
)
# The campaign is invoked from the user's selected environment.  The optional
# override is useful for wrappers, while sys.executable keeps the default
# portable across machines and conda installation paths.
PYTHON = Path(os.environ.get("PHYCOFLOW_PYTHON", sys.executable))
MAX_TOTAL_EPOCHS = 556
MAX_GPU_SECONDS = 16 * 60 * 60
MAX_TIMING_PROFILES = 5
MAX_TIMING_EPOCHS = 10
MAX_G1_EPOCHS = 100
OFFLINE_GPU_DIAGNOSTIC_SECONDS = 180.0
MAX_OFFLINE_DIAGNOSTIC_SECONDS = 2700.0
GOAL2_RESIDUAL_MAX_SECONDS = 600.0
GOAL2_RECOVERY_SMOKE_EPOCHS = 2
GOAL2_RECOVERY_SMOKE_EPOCH_BUDGET = 3
GPU_CAP_GRACE_SECONDS = 150.0
STOP_WAIT_SECONDS = 30.0
LOCK_POLL_INTERVAL_SECONDS = 0.1
MIN_EVALUATION_RESERVE_HOURS = 1.0
GOAL2_GPU_HOURS_RESERVE = 2.0
GOAL1_RUNTIME_QUOTA_FACTOR = 2.0
MAX_CONCURRENT_GPU_RUNS = 3
ADMISSION_LOCK_TIMEOUT_SECONDS = 5.0
NATIVE_MAP_MAX_SECONDS = 180.0
NATIVE_MAP_APPROVAL_FILES = {
    "extractor": (
        "analysis/extract_native_grid_field_maps.py",
        "b9f7ca1c6cbce8bee9c164dad9c31aa2eb215b2c3f13e86730af32edd70bfbf7",
    ),
    "plan": (
        "analysis/native_grid_field_map_predeclaration.json",
        "b189c295b2cda6b630df1a86ad93263f9a726455f2a37d4d046db914a891f4b6",
    ),
    "clarification": (
        "analysis/native_grid_field_map_schema_clarification.json",
        "29736e04ba0357bbaaa04f5b20a1aa0dd0c6b693ce66c459055ff53330fd776a",
    ),
}
GOAL2_RESIDUAL_APPROVAL_FILES = {
    "script": (
        "analysis/goal2_residual_probe.py",
        "ab6c667da30fbf2f791503872ce8c1a9abad994497ff4b723247ff5642150174",
    ),
    "plan": (
        "analysis/goal2_residual_probe.plan.json",
        "b38a96c2ca834e96410e57aa4d79c3f800a0a5442c32fe618235d3e57ab0c872",
    ),
    "binding": (
        "analysis/goal2_residual_probe.binding.json",
        "27b2eb1c666dbc85110940076b3dc7ad1f04c6cded7df097d93dab9796d2182e",
    ),
}
GOAL2_RESIDUAL_RESULT = "analysis/goal2_residual_probe.result.json"
GOAL2_RESIDUAL_RESULT_SCHEMA = "goal2_residual_probe.result.v1"
GOAL2_RESIDUAL_ACTION = "evaluate-goal2-residual-probe"
GOAL2_RESIDUAL_OFFLINE_RESERVE_SECONDS = 180.0
GOAL2_RUNTIME_QUOTA_SECONDS = GOAL2_GPU_HOURS_RESERVE * 3600.0
GOAL2_SMOKE_MAX_SECONDS = 900.0
GOAL2_ORDERED_SETTINGS = {"dual_lr": 0.01, "denominator_floor": 1.0e-8}
GOAL2_FORMAL_ARM = "G2-O"
GOAL2_SMOKE_ARM = "recovery_smoke2ep"
GOAL2_SMOKE_EPOCH1_RECEIPT = "manifests/goal2_smoke_epoch1_checkpoint.json"
GOAL2_SMOKE_RECOVERY_VERIFICATION = "manifests/goal2_smoke_recovery_verification.json"
GOAL2_SMOKE_INTERRUPTION_INVENTORY = "analysis/goal2_smoke_pre_resume/interruption_inventory.json"
GOAL2_FORMAL_READINESS_STATUS = "goal2_formal_readiness_assessed"
GOAL2_MIN_RESIDUAL_FRACTION = 0.1
GOAL2_MAX_ORTHOGONALITY_ERROR = 1.0e-6
GOAL2_ORDERED_PROFILE = {
    "topology_estimator": "p3",
    "family_probability": 1.0,
    "coefficient_interval": 1,
    "seed": 700211,
}
GOAL1_SNAPSHOT_RELATIVE = Path("source_snapshots/goal1_c073eba")
GOAL1_SNAPSHOT_MANIFEST_SHA256 = (
    "3e05b001318b16876c33c759a8014db2559afd4f5f949704bd9c21b4e1d66fd9"
)
GOAL1_SNAPSHOT_GIT_HEAD = "c073ebad9a48192c7e6eb3a8533199c323281db4"
GOAL1_COMPLETION_ASSESSMENT = "manifests/goal1_completion_assessment.json"
GOAL1_COMPLETION_ASSESSMENT_SHA256 = (
    "2a6f4ee7666e5a782f7d8f0c03a6d33bfb35cb55393766481f2ad36648390c8f"
)
GOAL2_CONSTRAINT_CLARIFICATION = "manifests/goal2_constraint_clarification_20261023.json"
GOAL2_CONSTRAINT_CLARIFICATION_SHA256 = (
    "30d52a085bc6c53c835bc10783448a5ae33a509e916ffe421b757d7e6804c196"
)
EXCLUDED_ARMS = {
    "T4": "periodic coefficients failed the TRAIN-only age probe: cosine fell to 0.79993 at refresh pair 2->3",
    "G1-P": "periodic coefficients failed the TRAIN-only age probe: cosine fell to 0.79993 at refresh pair 2->3",
}
G1_TIMING_PROFILE = {"G1-C": "T1", "G1-S": "T2", "G1-R": "T3", "G1-P": "T4"}
G1_GPU_ASSIGNMENTS = {"G1-C": 1, "G1-S": 2, "G1-R": 0}


class CampaignError(RuntimeError):
    """Raised when a campaign safety or readiness gate fails."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise CampaignError(f"cannot read YAML config {path}: {error}") from error
    if not isinstance(value, dict):
        raise CampaignError(f"expected a YAML mapping in {path}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise CampaignError(f"expected a JSON object in {path}")
    return value


def _campaign_id(root: Path) -> str:
    return root.name


def ensure_missing_symlink(link: Path, target: Path) -> None:
    """Create a repository symlink only when absent; preserve every existing path."""
    target = target.resolve(strict=True)
    if not link.parent.is_dir():
        raise CampaignError(f"refusing to create missing repository directories: {link.parent}")
    if link.is_symlink():
        try:
            current = link.resolve(strict=True)
        except FileNotFoundError as error:
            raise CampaignError(f"refusing to replace a dangling symlink: {link}") from error
        if current == target:
            return
        raise CampaignError(f"refusing to replace an unrelated symlink: {link} -> {current}")
    if os.path.lexists(link):
        raise CampaignError(f"refusing to replace an existing non-symlink path: {link}")
    link.symlink_to(target, target_is_directory=True)


def _verify_campaign_root(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    expected_parent = Path("/data2/wanglz/PhyCoFlow-MultiFieldReconstruction/experiments")
    if not root.is_relative_to(expected_parent) or not root.name.startswith("LPCAccel_"):
        raise CampaignError(f"campaign root is outside the owned /data2 experiment area: {root}")
    if not root.is_dir():
        raise CampaignError(f"campaign root is not a directory: {root}")
    if root.stat().st_uid != os.getuid():
        raise CampaignError(f"campaign root is not owned by the current user: {root}")
    preflight_path = root / "manifests/preflight.json"
    preflight = _read_json(preflight_path)
    if Path(preflight.get("root", "")).resolve() != root:
        raise CampaignError("preflight receipt names a different campaign root")
    for package, expected in preflight.get("versions", {}).items():
        if importlib.metadata.version(package) != expected:
            raise CampaignError(f"dependency version changed since preflight: {package}")
    if not Path("/data2").is_mount():
        raise CampaignError("/data2 is not currently a mounted filesystem")
    mounted_source = subprocess.check_output(
        ["findmnt", "-n", "-o", "SOURCE", "--target", "/data2"], text=True,
    ).strip()
    if mounted_source != preflight.get("mount"):
        raise CampaignError("/data2 mount identity differs from the verified preflight")
    if os.stat(root).st_dev != os.stat("/data2").st_dev:
        raise CampaignError("campaign root is not on the currently mounted /data2 filesystem")
    for name in ("runs", "timing", "analysis", "reports", "configs", "caches", "tmp", "manifests"):
        path = root / name
        if (not path.is_dir() or not path.resolve().is_relative_to(root)
                or path.stat().st_dev != root.stat().st_dev):
            raise CampaignError(f"campaign payload directory is missing or escapes /data2: {path}")

    protected = preflight.get("protected", {})
    expected_protected = {
        str(SOURCE_CONFIG): _sha256(SOURCE_CONFIG),
    }
    source_config = _read_yaml(SOURCE_CONFIG)
    source_run = Path(str(source_config.get("source_run", ""))).expanduser().resolve()
    checkpoint_setting = Path(str(source_config.get("source_checkpoint", "")))
    source_checkpoint = (
        checkpoint_setting.resolve()
        if checkpoint_setting.is_absolute()
        else (
            source_run / checkpoint_setting.name
            if source_config.get("source", {}).get("kind", "native_run") == "legacy_demo50"
            else source_run / "checkpoints" / f"{checkpoint_setting.stem}.pt"
        ).resolve()
    )
    if not source_run.is_dir() or not source_checkpoint.is_file():
        raise CampaignError("the Run 11f resolved config does not resolve to its original SOURCE")
    expected_protected[str(source_checkpoint)] = _sha256(source_checkpoint)
    for path_text, digest in expected_protected.items():
        record = protected.get(path_text)
        if not isinstance(record, dict) or record.get("sha256") != digest:
            raise CampaignError(f"/data2 preflight does not protect the current input: {path_text}")

    run_link = CASE_DIR / "runs/_LPCAccel" / root.name
    ensure_missing_symlink(run_link, root / "runs")
    return {
        "root": str(root),
        "mount": preflight.get("mount"),
        "source_config": str(SOURCE_CONFIG.resolve()),
        "source_config_sha256": expected_protected[str(SOURCE_CONFIG)],
        "source_run": str(source_run),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": expected_protected[str(source_checkpoint)],
        "run_link": str(run_link),
        "run_link_target": str(run_link.resolve()),
    }


def _arm_specs(selected_estimator: str | None) -> dict[str, dict[str, Any]]:
    selected = selected_estimator or "full"
    if selected not in {"full", "p3", "s2"}:
        raise CampaignError("selected topology estimator must be full, p3, or s2")
    specs = {
        "T0": {
            "stage": "timing",
            "epochs": 10,
            "topology_estimator": "full",
            "family_probability": 1.0,
            "coefficient_interval": 1,
            "route": "config",
        },
    }
    if selected_estimator is None:
        return specs
    specs.update(
        {
            "T1": {
                "stage": "timing",
                "epochs": 10,
                "topology_estimator": selected,
                "family_probability": 1.0,
                "coefficient_interval": 1,
                "route": "config",
            },
            "T2": {
                "stage": "timing",
                "epochs": 10,
                "topology_estimator": selected,
                "family_probability": 1.0,
                "coefficient_interval": 1,
                "route": "scalar",
            },
            "T3": {
                "stage": "timing",
                "epochs": 10,
                "topology_estimator": selected,
                "family_probability": 0.5,
                "coefficient_interval": 1,
                "route": "scalar",
            },
            "T4": {
                "stage": "timing",
                "epochs": 10,
                "topology_estimator": selected,
                "family_probability": 1.0,
                "coefficient_interval": 2,
                "route": "config",
            },
            "G1-C": {
                "stage": "goal1",
                "epochs": 100,
                "topology_estimator": selected,
                "family_probability": 1.0,
                "coefficient_interval": 1,
                "route": "config",
            },
            "G1-S": {
                "stage": "goal1",
                "epochs": 100,
                "topology_estimator": selected,
                "family_probability": 1.0,
                "coefficient_interval": 1,
                "route": "scalar",
            },
            "G1-R": {
                "stage": "goal1",
                "epochs": 100,
                "topology_estimator": selected,
                "family_probability": 0.5,
                "coefficient_interval": 1,
                "route": "scalar",
            },
            "G1-P": {
                "stage": "goal1",
                "epochs": 100,
                "topology_estimator": selected,
                "family_probability": 1.0,
                "coefficient_interval": 2,
                "route": "config",
            },
        }
    )
    return specs


def _goal2_arm_specs() -> dict[str, dict[str, Any]]:
    profile = {
        "topology_estimator": GOAL2_ORDERED_PROFILE["topology_estimator"],
        "family_probability": GOAL2_ORDERED_PROFILE["family_probability"],
        "coefficient_interval": GOAL2_ORDERED_PROFILE["coefficient_interval"],
        "route": "scalar",
    }
    return {
        GOAL2_FORMAL_ARM: {"stage": "goal2", "epochs": 100, **profile},
        GOAL2_SMOKE_ARM: {"stage": "goal2_smoke", "epochs": GOAL2_RECOVERY_SMOKE_EPOCHS, **profile},
    }


def _all_config_entries(manifest: Mapping[str, Any]) -> dict[str, Any]:
    configs = dict(manifest.get("configs", {}))
    goal2_configs = manifest.get("goal2_configs", {})
    if isinstance(goal2_configs, Mapping):
        configs.update(goal2_configs)
    return configs


def _arm_spec(manifest: Mapping[str, Any], arm: str) -> dict[str, Any]:
    if arm in _goal2_arm_specs():
        if arm not in manifest.get("goal2_configs", {}):
            raise CampaignError(f"Goal-2 arm {arm} has not been prepared")
        return _goal2_arm_specs()[arm]
    try:
        return _arm_specs(manifest.get("selected_topology_estimator"))[arm]
    except KeyError as error:
        raise CampaignError(f"unknown campaign arm {arm!r}") from error


def _goal2_config_from_g1s(
    g1s_config: Mapping[str, Any], campaign_id: str, arm: str
) -> dict[str, Any]:
    if arm not in _goal2_arm_specs():
        raise CampaignError(f"unsupported Goal-2 config {arm!r}")
    config = deepcopy(dict(g1s_config))
    optimization = config.get("optimization")
    if not isinstance(optimization, dict):
        raise CampaignError("G1-S config has no optimization map")
    if "lpc_ordered" in optimization:
        raise CampaignError("G1-S unexpectedly contains LPC-ordered settings")
    spec = _goal2_arm_specs()[arm]
    optimization["epochs"] = spec["epochs"]
    optimization["lpc_ordered"] = dict(GOAL2_ORDERED_SETTINGS)
    config["output"]["experiment_name"] = f"_LPCAccel/{campaign_id}/{arm}"
    if arm == GOAL2_SMOKE_ARM:
        checkpointing = config.setdefault("checkpointing", {})
        checkpointing["enabled"] = True
        checkpointing["every_epochs"] = 1
        checkpointing["save_epoch_one"] = True
        checkpointing["epochs"] = [1, 2]
    return config


def prepare_goal2(root: Path) -> dict[str, Any]:
    """Materialize only the gated ordered arm and its two-epoch recovery smoke."""
    manifest = _campaign_manifest(root)
    campaign_root = Path(manifest["root"]).resolve(strict=True)
    if not manifest.get("gates", {}).get("goal1_authorized"):
        raise CampaignError("Goal-2 configs require the completed authorized Goal-1 campaign")
    _verify_goal2_source_binding(manifest)
    _goal1_residual_runs_complete(campaign_root)
    g1s = _read_yaml(_config_path(manifest, "G1-S"))
    configs: dict[str, dict[str, str]] = {}
    for arm in (GOAL2_FORMAL_ARM, GOAL2_SMOKE_ARM):
        config = _goal2_config_from_g1s(g1s, manifest["campaign_id"], arm)
        _validate_config_only(config)
        path = campaign_root / "configs" / f"{arm}.yaml"
        text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        configs[arm] = {"path": str(path), "sha256": _write_new_or_same(path, text)}
    admission = _acquire_admission_lock(campaign_root)
    try:
        current = _campaign_manifest(campaign_root)
        old = current.get("goal2_configs")
        if old is not None and old != configs:
            raise CampaignError("prepared Goal-2 config binding changed; refusing to replace it")
        if old is None:
            current["goal2_configs"] = configs
            _atomic_json(campaign_root / "manifests/campaign.json", current)
        return current
    finally:
        _release_campaign_lock(admission)


def _make_config(
    base: Mapping[str, Any], campaign_id: str, arm: str, spec: Mapping[str, Any]
) -> dict[str, Any]:
    config = deepcopy(dict(base))
    optimization = config.get("optimization")
    if not isinstance(optimization, dict):
        raise CampaignError("Run 11f resolved config has no optimization map")
    multitask = optimization.get("multitask")
    if not isinstance(multitask, dict) or "external_policy" not in multitask:
        raise CampaignError("Run 11f resolved config lacks the external Lagrangian policy")
    if spec["route"] == "scalar":
        scalar = deepcopy(multitask)
        scalar.pop("coherence_aggregation", None)
        scalar["family_scalarization"] = {"method": "fixed_sum"}
        scalar["outer_aggregation"] = None
        optimization["multitask"] = scalar
    optimization["epochs"] = int(spec["epochs"])
    if spec["stage"] == "goal1":
        checkpointing = config.setdefault("checkpointing", {})
        checkpointing["epochs"] = [50, 75, 100]
        checkpointing["validation_every_epochs"] = 10
    if arm == "T0":
        # T0 is the established Run 11f route, so it must not exercise the new
        # opt-in path being compared in T1–T4.
        optimization.pop("fast_lagrangian", None)
    else:
        optimization["fast_lagrangian"] = {
            "topology_estimator": str(spec["topology_estimator"]),
            "family_probability": float(spec["family_probability"]),
            "coefficient_interval": int(spec["coefficient_interval"]),
            "seed": 700211,
        }
    config["output"]["experiment_name"] = f"_LPCAccel/{campaign_id}/{arm}"
    # CUDA_VISIBLE_DEVICES=1 maps the selected physical GPU to logical cuda:0.
    config["runtime"]["device"] = "cuda:0"
    return config


def _validate_config_only(config: Mapping[str, Any]) -> None:
    """Run the schema validator without opening or sampling the HDF5 dataset."""
    source_path = str(REPO_ROOT / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from phycoflow_reconstruction.config.validate import validate_config

    try:
        validate_config(config)
    except (TypeError, ValueError, KeyError) as error:
        raise CampaignError(f"campaign config failed schema validation: {error}") from error


def _write_new_or_same(path: Path, text: str) -> str:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise CampaignError(f"refusing to overwrite a different campaign-owned file: {path}")
        return _sha256(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return _sha256(path)


def _numerical_source_binding() -> dict[str, str]:
    package = REPO_ROOT / "src/phycoflow_reconstruction"
    return {
        str(path.relative_to(REPO_ROOT)): _sha256(path)
        for path in sorted(package.rglob("*.py"))
    }


def _verify_goal1_source_snapshot(root: Path, manifest: Mapping[str, Any]) -> Path:
    snapshot_root = (root / GOAL1_SNAPSHOT_RELATIVE).resolve(strict=True)
    if not snapshot_root.is_relative_to(root.resolve(strict=True)):
        raise CampaignError("Goal-1 source snapshot escapes the owned campaign root")
    manifest_path = snapshot_root / "snapshot_manifest.json"
    if _sha256(manifest_path) != GOAL1_SNAPSHOT_MANIFEST_SHA256:
        raise CampaignError("Goal-1 source snapshot manifest changed")
    snapshot = _read_json(manifest_path)
    if (
        snapshot.get("stage") != "goal1"
        or snapshot.get("git_head") != GOAL1_SNAPSHOT_GIT_HEAD
        or Path(snapshot.get("snapshot", "")).resolve(strict=True) != snapshot_root
    ):
        raise CampaignError("Goal-1 source snapshot identity changed")
    snapshot_files = snapshot.get("files_sha256")
    if not isinstance(snapshot_files, dict) or not snapshot_files:
        raise CampaignError("Goal-1 source snapshot has no file digest map")
    for relative_text, expected_digest in snapshot_files.items():
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise CampaignError("Goal-1 source snapshot contains an unsafe relative path")
        path = (snapshot_root / relative).resolve(strict=True)
        if not path.is_relative_to(snapshot_root) or not path.is_file():
            raise CampaignError(f"Goal-1 source snapshot file escapes or is missing: {relative_text}")
        if _sha256(path) != expected_digest:
            raise CampaignError(f"Goal-1 source snapshot file changed: {relative_text}")
    snapshot_numerical = {
        name: digest
        for name, digest in snapshot_files.items()
        if name.startswith("src/phycoflow_reconstruction/") and name.endswith(".py")
    }
    if manifest.get("numerical_source_sha256") != snapshot_numerical:
        raise CampaignError("frozen Goal-1 numerical source pin differs from its immutable snapshot")
    return snapshot_root / "src"


def _stage_source_bindings(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    value = manifest.get("stage_source_bindings", {})
    if not isinstance(value, Mapping):
        raise CampaignError("campaign stage source bindings are malformed")
    return value


def _verify_goal2_source_binding(manifest: Mapping[str, Any]) -> dict[str, Any]:
    stage_bindings = _stage_source_bindings(manifest)
    binding = stage_bindings.get("goal2") if isinstance(stage_bindings, Mapping) else None
    if not isinstance(binding, dict):
        raise CampaignError("Goal-2 current-source pin is missing; bind it after passing integration tests")
    expected = binding.get("files_sha256")
    if not isinstance(expected, dict) or not expected:
        raise CampaignError("Goal-2 current-source pin has no numerical file digests")
    if expected != _numerical_source_binding():
        raise CampaignError("Goal-2 current numerical source changed after its explicit binding")
    root = Path(manifest["root"]).resolve(strict=True)
    try:
        evidence_path = (root / str(binding.get("evidence_path", ""))).resolve(strict=True)
    except OSError as error:
        raise CampaignError("Goal-2 source-binding test evidence is missing") from error
    if (
        not evidence_path.is_file()
        or not evidence_path.is_relative_to(root)
        or _sha256(evidence_path) != binding.get("evidence_sha256")
    ):
        raise CampaignError("Goal-2 source-binding test evidence changed")
    evidence = _read_json(evidence_path)
    if (
        evidence.get("status") != "goal2_source_binding_ready"
        or evidence.get("helper_unit_tests_passed") is not True
        or evidence.get("config_schema_tests_passed") is not True
        or evidence.get("integration_tests_passed") is not True
        or evidence.get("source_sha256") != expected
    ):
        raise CampaignError("Goal-2 source-binding evidence no longer passes its recorded test gate")
    return binding


def _stage_source_root(root: Path, manifest: Mapping[str, Any], stage: str) -> Path:
    root = root.resolve(strict=True)
    if stage == "goal1":
        return _verify_goal1_source_snapshot(root, manifest)
    if stage == "goal2":
        _verify_goal2_source_binding(manifest)
        return REPO_ROOT / "src"
    raise CampaignError(f"unsupported campaign numerical source stage {stage!r}")


def _stage_source_digest(root: Path, manifest: Mapping[str, Any], stage: str) -> str:
    _stage_source_root(root, manifest, stage)
    if stage == "goal1":
        return GOAL1_SNAPSHOT_MANIFEST_SHA256
    binding = _stage_source_bindings(manifest)["goal2"]
    return str(binding["source_binding_sha256"])


def bind_goal2_source(root: Path, evidence_path: Path, *, apply: bool = False) -> dict[str, Any]:
    manifest = _campaign_manifest(root)
    root = Path(manifest["root"]).resolve(strict=True)
    evidence_path = evidence_path.resolve(strict=True)
    if not evidence_path.is_relative_to(root):
        raise CampaignError("Goal-2 source-binding evidence must be stored under the campaign root")
    evidence = _read_json(evidence_path)
    current_source = _numerical_source_binding()
    if (
        evidence.get("status") != "goal2_source_binding_ready"
        or evidence.get("helper_unit_tests_passed") is not True
        or evidence.get("config_schema_tests_passed") is not True
        or evidence.get("integration_tests_passed") is not True
        or evidence.get("source_sha256") != current_source
    ):
        raise CampaignError(
            "Goal-2 source binding requires passing helper, schema, and integration tests for the exact current source"
        )
    binding = {
        "evidence_path": str(evidence_path.relative_to(root)),
        "evidence_sha256": _sha256(evidence_path),
        "git_head": str(evidence.get("git_head", "")),
        "files_sha256": current_source,
        "source_binding_sha256": hashlib.sha256(
            json.dumps(current_source, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    if not binding["git_head"]:
        raise CampaignError("Goal-2 source-binding evidence must name the tested Git commit")
    old = _stage_source_bindings(manifest).get("goal2")
    if old is not None and old != binding:
        raise CampaignError("Goal-2 current-source pin is immutable once applied")
    if not apply:
        return binding
    admission = _acquire_admission_lock(root)
    try:
        current = _campaign_manifest(root)
        stage_bindings = current.setdefault("stage_source_bindings", {})
        if not isinstance(stage_bindings, dict):
            raise CampaignError("campaign stage source bindings are malformed")
        previous = stage_bindings.get("goal2")
        if previous is not None and previous != binding:
            raise CampaignError("Goal-2 current-source pin changed before it could be applied")
        if previous is None:
            stage_bindings["goal2"] = binding
            _atomic_json(root / "manifests/campaign.json", current)
        return binding
    finally:
        _release_campaign_lock(admission)


def _verify_numerical_source_binding(expected: Any) -> None:
    if not isinstance(expected, dict) or not expected:
        raise CampaignError("campaign lacks its reviewed numerical source binding")
    if expected != _numerical_source_binding():
        raise CampaignError("numerical source changed after campaign preparation; use the reviewed version")


def prepare(root: Path, selected_estimator: str | None = None) -> dict[str, Any]:
    verified = _verify_campaign_root(root)
    root = Path(verified["root"])
    campaign_id = _campaign_id(root)
    manifest_path = root / "manifests/campaign.json"
    base = _read_yaml(Path(verified["source_config"]))
    specs = _arm_specs(selected_estimator)
    old_manifest: dict[str, Any] | None = None
    t0_migration_sha: str | None = None
    if manifest_path.exists():
        old = _read_json(manifest_path)
        _verify_goal1_source_snapshot(root, old)
        if _stage_source_bindings(old).get("goal2") is not None:
            _verify_goal2_source_binding(old)
        old_manifest = old
        for key in ("campaign_id", "root", "source_config_sha256", "source_checkpoint_sha256"):
            if old.get(key) != verified.get(key, campaign_id if key == "campaign_id" else None):
                # campaign_id and root are also present in the verified receipt mapping.
                raise CampaignError(f"existing campaign manifest differs at {key}")
        previous_selected = old.get("selected_topology_estimator")
        if previous_selected is not None and previous_selected != selected_estimator:
            raise CampaignError(
                "selected estimator is immutable after campaign configs are materialized"
            )
        if previous_selected is None:
            if selected_estimator is not None and any(
                _run_dirs(root, arm) for arm in old.get("configs", {})
            ):
                raise CampaignError("cannot materialize the selected estimator after a run exists")
            old_t0 = old.get("configs", {}).get("T0", {})
            t0_path = root / "configs/T0.yaml"
            if (
                t0_path.is_file()
                and _sha256(t0_path) == old_t0.get("sha256")
                and not _run_dirs(root, "T0")
            ):
                previous_t0 = _read_yaml(t0_path)
                previous_optimization = previous_t0.get("optimization", {})
                desired_t0 = _make_config(base, campaign_id, "T0", _arm_specs(None)["T0"])
                if "fast_lagrangian" in previous_optimization:
                    previous_optimization.pop("fast_lagrangian")
                    if previous_t0 == desired_t0:
                        t0_migration_sha = old_t0["sha256"]
            if selected_estimator is not None and t0_migration_sha is None:
                t0_path = root / "configs/T0.yaml"
                if not t0_path.is_file() or _sha256(t0_path) != old_t0.get("sha256"):
                    raise CampaignError(
                        "T0 changed; refusing to materialize the selected estimator"
                    )
    configs: dict[str, dict[str, str]] = {}
    for arm, spec in specs.items():
        config = _make_config(base, campaign_id, arm, spec)
        _validate_config_only(config)
        path = root / "configs" / f"{arm}.yaml"
        text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        if (
            arm == "T0"
            and t0_migration_sha is not None
            and path.is_file()
            and _sha256(path) == t0_migration_sha
        ):
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, path)
            digest = _sha256(path)
        else:
            digest = _write_new_or_same(path, text)
        configs[arm] = {"path": str(path), "sha256": digest}

    candidate = {
        "schema_version": 1,
        "numerical_source_sha256": _numerical_source_binding(),
        "campaign_id": campaign_id,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **verified,
        "selected_topology_estimator": selected_estimator,
        "limits": {
            "timing_profiles": MAX_TIMING_PROFILES,
            "timing_epochs_per_profile": MAX_TIMING_EPOCHS,
            "goal1_arms": 4,
            "goal1_epochs_per_arm": MAX_G1_EPOCHS,
            "optional_goal2_arms": 1,
            "optional_goal2_epochs": 100,
            "public_smokes_epochs": 6,
            "total_epochs": MAX_TOTAL_EPOCHS,
            "campaign_training_epochs": MAX_TOTAL_EPOCHS - 6,
            "gpu_hours": MAX_GPU_SECONDS / 3600,
            "physical_gpus": [0, 1, 2],
            "concurrent_gpu_runs": MAX_CONCURRENT_GPU_RUNS,
            "goal1_gpu_assignment": G1_GPU_ASSIGNMENTS,
            "gpu1_only": False,
            "test_split_locked": True,
        },
        "configs": configs,
        "eligible_arms": {
            "timing": ["T0", "T1", "T2", "T3"],
            "goal1": ["G1-C", "G1-S", "G1-R"],
        },
        "excluded_arms": EXCLUDED_ARMS,
        "gates": {
            "timing_passed": False,
            "goal1_authorized": False,
            "goal2_implemented": False,
            "goal2_authorized": False,
        },
        "notes": [
            "Configs are derived from the local Run 11f resolved map and start from its original SOURCE.",
            "T0 is the unchanged Run 11f dense ConFIG/full-topology baseline; T1-T4 exercise opt-in fast-Lagrangian profiles.",
            "T4/G1-P retain interval 2 as default-off configs but are excluded: the TRAIN-only refresh-pair probe measured cosine 0.96099 at age 0->1 and 0.79993 at age 2->3.",
            "Goal 2 remains unavailable until G1 completes and an LPC-ordered runner is implemented and verified.",
            "All training and evaluation commands require explicit --execute; no work is queued.",
        ],
    }
    if manifest_path.exists():
        old = old_manifest or _read_json(manifest_path)
        if old.get("configs") == configs:
            if old.get("limits") != candidate["limits"]:
                old["limits"] = candidate["limits"]
                old["eligible_arms"] = candidate["eligible_arms"]
                old["excluded_arms"] = candidate["excluded_arms"]
                old.setdefault("notes", []).extend(
                    note for note in candidate["notes"] if note not in old.get("notes", [])
                )
                _atomic_json(manifest_path, old)
            return old
        if old.get("selected_topology_estimator") is None and selected_estimator is not None:
            candidate["created_utc"] = old.get("created_utc", candidate["created_utc"])
            candidate["gates"] = old.get("gates", candidate["gates"])
            _atomic_json(manifest_path, candidate)
            return candidate
        raise CampaignError("existing campaign config inventory differs; refusing to rewrite it")
    _atomic_json(manifest_path, candidate)
    return candidate


def _campaign_manifest(root: Path) -> dict[str, Any]:
    verified = _verify_campaign_root(root)
    manifest = _read_json(Path(verified["root"]) / "manifests/campaign.json")
    if manifest.get("root") != verified["root"]:
        raise CampaignError("campaign manifest root does not match the verified data root")
    _verify_goal1_source_snapshot(Path(verified["root"]), manifest)
    if _stage_source_bindings(manifest).get("goal2") is not None:
        _verify_goal2_source_binding(manifest)
    if (
        manifest.get("source_config_sha256") != verified["source_config_sha256"]
        or manifest.get("source_checkpoint_sha256") != verified["source_checkpoint_sha256"]
    ):
        raise CampaignError("Run 11f source binding changed after campaign preparation")
    return manifest


def _run_dirs(root: Path, arm: str) -> list[Path]:
    base = CASE_DIR / "runs" / "_LPCAccel" / root.name / arm
    if not base.exists():
        return []
    return sorted(
        path for path in base.iterdir() if path.is_dir() and (path / "run_manifest.json").is_file()
    )


def _run_record(run_dir: Path) -> dict[str, Any]:
    status_path = run_dir / "status.json"
    progress_path = run_dir / "progress.json"
    status = _read_json(status_path) if status_path.is_file() else {}
    progress = _read_json(progress_path) if progress_path.is_file() else {}
    global_step = max(int(status.get("global_step", 0)), int(progress.get("completed_updates", 0)))
    steps_per_epoch = int(status.get("steps_per_epoch", progress.get("steps_per_epoch", 0)) or 0)
    completed_epochs = global_step / steps_per_epoch if steps_per_epoch else None
    return {
        "run_dir": str(run_dir.resolve()),
        "status": status.get("status", "unknown"),
        "completed_updates": global_step,
        "steps_per_epoch": steps_per_epoch or None,
        "completed_epochs": completed_epochs,
        "post_training_seconds": status.get("post_training_seconds"),
        "configured_steps": status.get("configured_steps"),
    }


def _ledger_seconds(root: Path) -> dict[str, float]:
    path = root / "manifests/launch_ledger.jsonl"
    totals: dict[str, float] = {}
    if not path.is_file():
        return totals
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
            arm = str(row["arm"])
            elapsed = float(row["elapsed_seconds"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise CampaignError(
                f"invalid campaign launch ledger row {line_number}: {error}"
            ) from error
        if elapsed < 0:
            raise CampaignError(
                f"negative elapsed time in campaign launch ledger row {line_number}"
            )
        totals[arm] = totals.get(arm, 0.0) + elapsed
    return totals


def _ledger_training_seconds(root: Path) -> dict[str, float]:
    path = root / "manifests/launch_ledger.jsonl"
    totals: dict[str, float] = {}
    if not path.is_file():
        return totals
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
            arm = str(row["arm"])
            elapsed = float(row["elapsed_seconds"])
            action = str(row.get("action", "launch"))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise CampaignError(
                f"invalid campaign launch ledger row {line_number}: {error}"
            ) from error
        if elapsed < 0:
            raise CampaignError(
                f"negative elapsed time in campaign launch ledger row {line_number}"
            )
        if action in {"launch", "resume"}:
            totals[arm] = totals.get(arm, 0.0) + elapsed
    return totals


def _ledger_action_seconds(root: Path, action: str, *, prefix: bool = False) -> float:
    path = root / "manifests/launch_ledger.jsonl"
    if not path.is_file():
        return 0.0
    total = 0.0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
            elapsed = float(row["elapsed_seconds"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise CampaignError(
                f"invalid campaign launch ledger row {line_number}: {error}"
            ) from error
        if elapsed < 0:
            raise CampaignError(
                f"negative elapsed time in campaign launch ledger row {line_number}"
            )
        row_action = str(row.get("action", ""))
        if (row_action.startswith(action) if prefix else row_action == action):
            total += elapsed
    return total


def segment_epoch_charge(return_code, horizon, start_epoch, end_epoch, has_run):
    """Charge interrupted segments to their remaining horizon as a safe upper bound."""
    if not has_run:
        return 0.0
    if return_code != 0:
        return max(0.0, float(horizon) - float(start_epoch))
    return max(0.0, float(end_epoch or 0.0) - float(start_epoch))


def _ledger_epochs(root):
    path = root / "manifests/launch_ledger.jsonl"
    totals = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            totals[row["arm"]] = totals.get(row["arm"], 0.0) + float(row.get("epochs_charged", 0.0))
    return totals


def _gpu_index(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1, 2):
        raise CampaignError("physical GPU must be 0, 1, or 2")
    return value


def _parse_gpu(value: str) -> int:
    text = str(value).strip().lower().removeprefix("physical")
    try:
        return _gpu_index(int(text))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use 0, 1, 2 or physical0, physical1, physical2") from error
    except CampaignError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _gpu_active_path(root: Path, gpu: int) -> Path:
    gpu = _gpu_index(gpu)
    name = "active_process.json" if gpu == 1 else f"active_process_gpu{gpu}.json"
    return root / "manifests" / name


def _gpu_lock_path(root: Path, gpu: int) -> Path:
    gpu = _gpu_index(gpu)
    name = "active_process.lock" if gpu == 1 else f"active_process_gpu{gpu}.lock"
    return root / "manifests" / name


def _active_process_records(root: Path) -> list[dict[str, Any]]:
    records = []
    seen_paths = set()
    for gpu in range(3):
        path = _gpu_active_path(root, gpu)
        if not path.is_file() or path in seen_paths:
            continue
        seen_paths.add(path)
        record = _read_json(path)
        record.setdefault("gpu_index", gpu)
        record["metadata_path"] = str(path)
        record["elapsed_seconds"] = _active_elapsed_seconds(record)
        records.append(record)
    # A pre-multi-GPU receipt used the original path and implicitly meant GPU 1.
    legacy_path = root / "manifests/active_process.json"
    if legacy_path.is_file() and legacy_path not in seen_paths:
        record = _read_json(legacy_path)
        record.setdefault("gpu_index", 1)
        record["metadata_path"] = str(legacy_path)
        record["elapsed_seconds"] = _active_elapsed_seconds(record)
        records.append(record)
    return records


def _active_elapsed_seconds(record: Mapping[str, Any]) -> float:
    start_ns = record.get("started_monotonic_ns")
    if isinstance(start_ns, int) and not isinstance(start_ns, bool):
        return max(0.0, (time.monotonic_ns() - start_ns) / 1e9)
    started = record.get("started_utc")
    if isinstance(started, str):
        try:
            value = datetime.fromisoformat(started.replace("Z", "+00:00"))
            return max(0.0, (datetime.now(timezone.utc) - value).total_seconds())
        except ValueError:
            pass
    return 0.0


def _status_data(manifest: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(manifest["root"])
    runs = {
        arm: [_run_record(path) for path in _run_dirs(root, arm)]
        for arm in _all_config_entries(manifest)
    }
    counted_epochs = sum(
        (float(record["completed_epochs"]) if record["completed_epochs"] is not None else 0.0)
        for records in runs.values()
        for record in records
    )
    ledger_seconds = _ledger_seconds(root)
    training_ledger_seconds = _ledger_training_seconds(root)
    trainer_seconds = {
        arm: sum(float(record["post_training_seconds"] or 0.0) for record in records)
        for arm, records in runs.items()
    }
    active_processes = _active_process_records(root)
    active_seconds: dict[str, float] = {}
    active_training_seconds: dict[str, float] = {}
    for record in active_processes:
        arm = str(record.get("arm", "unknown"))
        elapsed = float(record.get("elapsed_seconds", 0.0))
        active_seconds[arm] = active_seconds.get(arm, 0.0) + elapsed
        if str(record.get("action", "launch")) in {"launch", "resume"}:
            active_training_seconds[arm] = active_training_seconds.get(arm, 0.0) + elapsed
    training_process_seconds_by_arm = {}
    for arm in set(training_ledger_seconds) | set(trainer_seconds) | set(active_training_seconds):
        live_training = active_training_seconds.get(arm, 0.0)
        historical_training = max(
            training_ledger_seconds.get(arm, 0.0),
            max(0.0, trainer_seconds.get(arm, 0.0) - live_training),
        )
        training_process_seconds_by_arm[arm] = historical_training + live_training
    process_seconds_by_arm = {}
    for arm in set(ledger_seconds) | set(training_process_seconds_by_arm) | set(active_seconds):
        nontraining_ledger = max(
            0.0, ledger_seconds.get(arm, 0.0) - training_ledger_seconds.get(arm, 0.0)
        )
        nontraining_live = max(
            0.0, active_seconds.get(arm, 0.0) - active_training_seconds.get(arm, 0.0)
        )
        process_seconds_by_arm[arm] = (
            training_process_seconds_by_arm.get(arm, 0.0)
            + nontraining_ledger
            + nontraining_live
        )
    process_seconds = sum(process_seconds_by_arm.values())
    counted_seconds = process_seconds + OFFLINE_GPU_DIAGNOSTIC_SECONDS
    ledger_epochs = _ledger_epochs(root)
    charged_by_arm = {
        arm: max(
            ledger_epochs.get(arm, 0.0), sum(float(r["completed_epochs"] or 0.0) for r in records)
        )
        for arm, records in runs.items()
    }
    return {
        "campaign_id": manifest["campaign_id"],
        "root": str(root),
        "selected_topology_estimator": manifest.get("selected_topology_estimator"),
        "gates": manifest.get("gates", {}),
        "training_epochs_observed": counted_epochs,
        "training_epochs_charged": sum(charged_by_arm.values()),
        "charged_epochs_by_arm": charged_by_arm,
        "timing_epochs_charged": sum(
            value for arm, value in charged_by_arm.items() if arm.startswith("T")
        ),
        "training_epoch_cap": MAX_TOTAL_EPOCHS - 6,
        "public_smoke_epochs_reserved": 6,
        "total_epochs_including_reserved_smokes": sum(charged_by_arm.values()) + 6,
        "reported_gpu_hours": counted_seconds / 3600,
        "gpu_seconds_charged": counted_seconds,
        "reported_campaign_process_gpu_hours": process_seconds / 3600,
        "offline_gpu_diagnostic_seconds_reserved": OFFLINE_GPU_DIAGNOSTIC_SECONDS,
        "campaign_process_seconds_by_arm": ledger_seconds,
        "campaign_training_process_seconds_by_arm": training_ledger_seconds,
        "trainer_reported_seconds_by_arm": trainer_seconds,
        "active_process_seconds_by_arm": active_seconds,
        "active_training_process_seconds_by_arm": active_training_seconds,
        "process_seconds_by_arm": process_seconds_by_arm,
        "training_process_seconds_by_arm": training_process_seconds_by_arm,
        "active_processes": active_processes,
        "gpu_hour_cap": MAX_GPU_SECONDS / 3600,
        "timing_profile_count": sum(
            len(runs.get(arm, [])) for arm in ("T0", "T1", "T2", "T3", "T4")
        ),
        "runs": runs,
        "goal2_gpu_hours_reserved": GOAL2_GPU_HOURS_RESERVE,
        "minimum_evaluation_gpu_hours_reserved": MIN_EVALUATION_RESERVE_HOURS,
        "goal1_runtime_quota_factor": GOAL1_RUNTIME_QUOTA_FACTOR,
        "maximum_concurrent_gpu_runs": MAX_CONCURRENT_GPU_RUNS,
        "watchdog_shutdown_reserve_seconds": GPU_CAP_GRACE_SECONDS * MAX_CONCURRENT_GPU_RUNS,
        "accounting_note": "Completed ledger time, trainer time, and live coordinator clocks are combined without adding overlapping reports; 180 seconds of offline diagnostics and six public-smoke epochs remain reserved/charged.",
    }


def status(root: Path) -> dict[str, Any]:
    manifest = _campaign_manifest(root)
    admission = _acquire_admission_lock(Path(manifest["root"]))
    try:
        _reject_active_or_orphaned_process(Path(manifest["root"]))
        return _status_data(manifest)
    finally:
        _release_campaign_lock(admission)


def _config_path(manifest: Mapping[str, Any], arm: str) -> Path:
    config = _all_config_entries(manifest).get(arm)
    if not isinstance(config, dict):
        raise CampaignError(f"arm {arm!r} has no prepared config")
    path = Path(config["path"]).resolve(strict=True)
    if not path.is_relative_to(Path(manifest["root"]).resolve() / "configs"):
        raise CampaignError("arm config escapes the owned campaign config directory")
    if _sha256(path) != config.get("sha256"):
        raise CampaignError(f"arm config was edited after preparation: {path}")
    return path


def _validate_arm(manifest: Mapping[str, Any], arm: str) -> dict[str, Any]:
    if arm not in _all_config_entries(manifest):
        raise CampaignError(f"unknown or unprepared arm {arm!r}")
    excluded = manifest.get("excluded_arms", {}).get(arm)
    if excluded:
        raise CampaignError(f"{arm} is excluded by its TRAIN-only pretest: {excluded}")
    if (
        arm != "T0"
        and arm not in _goal2_arm_specs()
        and manifest.get("selected_topology_estimator") is None
    ):
        raise CampaignError(
            "prepare --selected-topology after TRAIN-only estimator selection before launching this arm"
        )
    path = _config_path(manifest, arm)
    config = _read_yaml(path)
    spec = _arm_spec(manifest, arm)
    fast = config.get("optimization", {}).get("fast_lagrangian", {})
    expected = {
        "topology_estimator": spec["topology_estimator"],
        "family_probability": spec["family_probability"],
        "coefficient_interval": spec["coefficient_interval"],
        "seed": 700211,
    }
    if arm == "T0" and "fast_lagrangian" in config.get("optimization", {}):
        raise CampaignError("T0 must retain the original Run 11f execution path")
    if arm != "T0" and fast != expected:
        raise CampaignError(f"{arm} fast_lagrangian settings differ from the campaign protocol")
    if config.get("dataset", {}).get("split") != "train":
        raise CampaignError(f"{arm} must train only on the TRAIN split")
    evaluation = config.get("evaluation", {})
    if evaluation.get("split", "validation") != "validation":
        raise CampaignError(f"{arm} must select checkpoints only on the validation split")
    preview = evaluation.get("preview", {})
    if preview.get("enabled", False) and preview.get("split", "validation") != "validation":
        raise CampaignError(f"{arm} preview must use the validation split")
    if int(config.get("optimization", {}).get("epochs", 0)) != spec["epochs"]:
        raise CampaignError(f"{arm} epoch horizon differs from the campaign cap")
    if arm.startswith("G1-") and spec["epochs"] > MAX_G1_EPOCHS:
        raise CampaignError("G1 lineage cannot exceed 100 epochs")
    if spec["stage"] == "timing" and spec["epochs"] > MAX_TIMING_EPOCHS:
        raise CampaignError("timing lineages cannot exceed 10 epochs")
    if spec["route"] == "scalar":
        multitask = config["optimization"]["multitask"]
        if "coherence_aggregation" in multitask or multitask.get("family_scalarization") != {
            "method": "fixed_sum"
        }:
            raise CampaignError("scalar config retains a recursive ConFIG selector")
    if arm in _goal2_arm_specs():
        expected_config = _goal2_config_from_g1s(
            _read_yaml(_config_path(manifest, "G1-S")), manifest["campaign_id"], arm
        )
        if config != expected_config:
            raise CampaignError(f"{arm} differs from the immutable ordered G1-S profile")
        if config.get("optimization", {}).get("lpc_ordered") != GOAL2_ORDERED_SETTINGS:
            raise CampaignError(f"{arm} LPC-ordered settings differ from the declared profile")
        fidelity = config.get("optimization", {}).get("multitask", {}).get("fidelity_budget", {})
        if (
            fidelity.get("relative_increase") != 0.05
            or fidelity.get("absolute_allowance") != 0.0
            or fidelity.get("metric") != "endpoint_mse_model_units"
        ):
            raise CampaignError(f"{arm} endpoint-fidelity contract differs from the +5% plan")
    return {"config": path, "spec": spec}


def _validate_gpu_for_arm(arm: str, gpu: int, *, enforce_assignment: bool = True) -> int:
    gpu = _gpu_index(gpu)
    expected = G1_GPU_ASSIGNMENTS.get(arm)
    if enforce_assignment and expected is not None and gpu != expected:
        raise CampaignError(f"{arm} is assigned to physical GPU {expected}; refusing GPU {gpu}")
    if (arm.startswith("G2") or arm == GOAL2_SMOKE_ARM) and gpu != 0:
        raise CampaignError("Goal 2 is reserved for physical GPU 0")
    return gpu


def _command(arm: str, config: Path, resume: Path | None = None) -> list[str]:
    command = [str(PYTHON), str(CASE_DIR / "run.py"), "post-train", "--config", str(config)]
    if resume is not None:
        command.extend(["--resume", str(resume)])
    return _with_short_multiprocessing_tmp(command)


def _with_short_multiprocessing_tmp(command: list[str]) -> list[str]:
    """Keep multiprocessing AF_UNIX socket names short despite /data2 TMPDIR."""
    bootstrap = (
        "import multiprocessing, os, runpy, shutil, sys, tempfile\n"
        "short_tmp = tempfile.mkdtemp(prefix='lpc-', dir='/tmp')\n"
        "multiprocessing.current_process()._config['tempdir'] = short_tmp\n"
        "script = sys.argv[1]\n"
        "sys.path.insert(0, os.path.dirname(os.path.abspath(script)))\n"
        "sys.argv = sys.argv[1:]\n"
        "try:\n"
        "    runpy.run_path(script, run_name='__main__')\n"
        "finally:\n"
        "    shutil.rmtree(short_tmp, ignore_errors=True)\n"
    )
    return [command[0], "-c", bootstrap, *command[1:]]


def _environment(
    root: Path, gpu: int = 1, *, source_stage: str | None = None
) -> dict[str, str]:
    gpu = _gpu_index(gpu)
    cache_root = root / "caches"
    (cache_root / "keops").mkdir(parents=True, exist_ok=True)
    (cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache_root / "pycache").mkdir(parents=True, exist_ok=True)
    (root / "tmp").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONPYCACHEPREFIX": str(cache_root / "pycache"),
            "TMPDIR": str(root / "tmp"),
            "KEOPS_CACHE_FOLDER": str(cache_root / "keops"),
            "MPLCONFIGDIR": str(cache_root / "matplotlib"),
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "PHYCOFLOW_TOPOLOGY_WORKERS": "4",
            "PHYCOFLOW_CAMPAIGN_ADMITTED": str(root.resolve()),
            "PHYCOFLOW_CAMPAIGN_GPU": str(gpu),
        }
    )
    if source_stage is not None:
        env["PHYCOFLOW_CAMPAIGN_SOURCE_STAGE"] = source_stage
    return env


def _prepare_keops_build_folder(root: Path, env: Mapping[str, str], gpu: int = 1) -> Path:
    gpu = _gpu_index(gpu)
    build_name = subprocess.check_output(
        [
            str(PYTHON),
            "-c",
            f"import platform, sys; print('_'.join(platform.uname()[:3]) + f'_p{{sys.version.split()[0]}}_CUDA_VISIBLE_DEVICES_{gpu}')",
        ],
        env=env,
        text=True,
    ).strip()
    build_folder = root / "caches/keops" / build_name
    build_folder.mkdir(parents=True, exist_ok=True)
    return build_folder


def _rendered_command(
    root: Path,
    command: list[str],
    gpu: int = 1,
    *,
    source_stage: str | None = None,
) -> str:
    env = _environment(root, gpu, source_stage=source_stage)
    names = (
        "CUDA_VISIBLE_DEVICES",
        "PYTHONPYCACHEPREFIX",
        "TMPDIR",
        "KEOPS_CACHE_FOLDER",
        "MPLCONFIGDIR",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "PHYCOFLOW_TOPOLOGY_WORKERS",
        "PHYCOFLOW_CAMPAIGN_ADMITTED",
        "PHYCOFLOW_CAMPAIGN_GPU",
    )
    if source_stage is not None:
        names += ("PHYCOFLOW_CAMPAIGN_SOURCE_STAGE",)
    prefix = " ".join(f"{name}={shlex.quote(env[name])}" for name in names)
    return f"{prefix} {shlex.join(command)}"


def _goal1_remaining_hours(
    manifest: Mapping[str, Any],
    *,
    skip_arm: str | None = None,
    current_status: Mapping[str, Any] | None = None,
) -> float:
    gate = manifest.get("gate_evidence", {}).get("goal1", {})
    current = current_status or status(Path(manifest["root"]))
    process_seconds = current.get(
        "training_process_seconds_by_arm", current.get("process_seconds_by_arm", {})
    )
    remaining = 0.0
    for arm in gate.get("approved_arms", []):
        if arm == skip_arm:
            continue
        completed = max(
            (
                float(record.get("completed_epochs") or 0.0)
                for record in current.get("runs", {}).get(arm, [])
            ),
            default=0.0,
        )
        if completed >= MAX_G1_EPOCHS:
            continue
        quota_hours = _goal1_quota_hours(manifest, arm)
        spent_hours = float(process_seconds.get(arm, 0.0)) / 3600
        remaining += max(0.0, quota_hours - spent_hours)
    return remaining


def _goal1_quota_hours(manifest: Mapping[str, Any], arm: str) -> float:
    gate = manifest.get("gate_evidence", {}).get("goal1", {})
    quotas = gate.get("runtime_quotas_hours_by_arm", {})
    if arm in quotas:
        return float(quotas[arm])
    estimates = gate.get("projected_hours_by_arm", {})
    return float(estimates.get(arm, 0.0)) * GOAL1_RUNTIME_QUOTA_FACTOR


def _evaluation_reserve_hours(manifest: Mapping[str, Any]) -> float:
    gate = manifest.get("gate_evidence", {}).get("goal1", {})
    return max(MIN_EVALUATION_RESERVE_HOURS, float(gate.get("evaluation_reserve_hours", 0.0)))


def _goal2_formal_remaining_seconds(
    current: Mapping[str, Any],
) -> float:
    completed = max(
        (
            float(record.get("completed_epochs") or 0.0)
            for record in current.get("runs", {}).get(GOAL2_FORMAL_ARM, [])
        ),
        default=0.0,
    )
    if completed >= MAX_G1_EPOCHS:
        return 0.0
    process_seconds = current.get(
        "training_process_seconds_by_arm", current.get("process_seconds_by_arm", {})
    )
    return max(
        0.0,
        GOAL2_RUNTIME_QUOTA_SECONDS - float(process_seconds.get(GOAL2_FORMAL_ARM, 0.0)),
    )


def _native_maps_approval_binding(root: Path) -> dict[str, Any]:
    binding: dict[str, Any] = {"schema_version": 1}
    for key, (relative_path, expected_sha256) in NATIVE_MAP_APPROVAL_FILES.items():
        path = root / relative_path
        if not path.is_file():
            raise CampaignError(f"native-map approval file is missing: {path}")
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise CampaignError(f"native-map approval file changed: {path}")
        binding[key] = {"path": relative_path, "sha256": actual_sha256}
    return binding


def _store_native_maps_approval_binding(
    root: Path, binding: Mapping[str, Any]
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    admission = _acquire_admission_lock(root)
    try:
        manifest = _campaign_manifest(root)
        current = manifest.get("native_map_approval_binding")
        if current is not None and current != binding:
            raise CampaignError("native-map approval binding changed after it was recorded")
        if current is None:
            manifest["native_map_approval_binding"] = dict(binding)
            _atomic_json(root / "manifests/campaign.json", manifest)
        return manifest
    finally:
        _release_campaign_lock(admission)


def _goal1_residual_runs_complete(root: Path) -> dict[str, Path]:
    completed: dict[str, Path] = {}
    for arm in ("G1-C", "G1-S", "G1-R"):
        runs = _run_dirs(root, arm)
        if len(runs) != 1:
            raise CampaignError(
                f"Goal-2 residual probe requires exactly one completed {arm} lineage; found {len(runs)}"
            )
        record = _run_record(runs[0])
        if (
            record.get("status") != "completed"
            or float(record.get("completed_epochs") or 0.0) != MAX_G1_EPOCHS
            or int(record.get("completed_updates") or 0) != 3800
            or int(record.get("steps_per_epoch") or 0) != 38
        ):
            raise CampaignError(
                f"Goal-2 residual probe requires exactly 100 completed epochs for {arm}"
            )
        completed[arm] = runs[0].resolve(strict=True)
    return completed


def _goal2_residual_approval_binding(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    snapshot_src = _verify_goal1_source_snapshot(root, _read_json(root / "manifests/campaign.json"))
    snapshot_root = snapshot_src.parent
    snapshot_manifest_path = snapshot_root / "snapshot_manifest.json"
    snapshot_manifest = _read_json(snapshot_manifest_path)
    files: dict[str, dict[str, str]] = {}
    for key, (relative_path, expected_sha256) in GOAL2_RESIDUAL_APPROVAL_FILES.items():
        path = root / relative_path
        if not path.is_file() or _sha256(path) != expected_sha256:
            raise CampaignError(f"Goal-2 residual approval file changed or is missing: {path}")
        files[key] = {"path": relative_path, "sha256": expected_sha256}

    binding = _read_json(root / GOAL2_RESIDUAL_APPROVAL_FILES["binding"][0])
    plan = _read_json(root / GOAL2_RESIDUAL_APPROVAL_FILES["plan"][0])
    script_path = root / GOAL2_RESIDUAL_APPROVAL_FILES["script"][0]
    plan_path = root / GOAL2_RESIDUAL_APPROVAL_FILES["plan"][0]
    if (
        binding.get("schema_version") != 1
        or binding.get("status") != "frozen_hash_binding"
        or binding.get("script_path") != str(script_path)
        or binding.get("script_sha256") != GOAL2_RESIDUAL_APPROVAL_FILES["script"][1]
        or binding.get("plan_path") != str(plan_path)
        or binding.get("plan_sha256") != GOAL2_RESIDUAL_APPROVAL_FILES["plan"][1]
        or binding.get("result_path") != str(root / GOAL2_RESIDUAL_RESULT)
        or binding.get("result_schema") != GOAL2_RESIDUAL_RESULT_SCHEMA
        or binding.get("source_snapshot_manifest_sha256") != GOAL1_SNAPSHOT_MANIFEST_SHA256
        or binding.get("code_dependencies_sha256") != snapshot_manifest.get("files_sha256")
    ):
        raise CampaignError("Goal-2 residual probe binding does not match its frozen code/plan contract")
    if (
        plan.get("schema_version") != 1
        or plan.get("status") != "predeclared_protocol_no_results"
        or plan.get("campaign_id") != root.name
        or plan.get("campaign_root") != str(root)
    ):
        raise CampaignError("Goal-2 residual probe plan identity or status changed")

    closeout = root / GOAL1_COMPLETION_ASSESSMENT
    clarification = root / GOAL2_CONSTRAINT_CLARIFICATION
    if (
        _sha256(closeout) != GOAL1_COMPLETION_ASSESSMENT_SHA256
        or _sha256(clarification) != GOAL2_CONSTRAINT_CLARIFICATION_SHA256
    ):
        raise CampaignError("Goal-2 residual closeout or endpoint-constraint receipt changed")
    closeout_data = _read_json(closeout)
    clarification_data = _read_json(clarification)
    if (
        closeout_data.get("status") != "goal1_complete_assessed"
        or set(closeout_data.get("goal1_arms_completed_100", ())) != {"G1-C", "G1-S", "G1-R"}
        or closeout_data.get("source_142_files_unchanged") is not True
        or closeout_data.get("test_evaluation_or_selection") is not False
        or clarification_data.get("endpoint_relative_increase") != 0.05
        or clarification_data.get("endpoint_absolute_allowance") != 0.0
        or clarification_data.get("A_bound") != "fixed full SOURCE TRAIN mean without slack"
        or clarification_data.get("B_bound") != "fixed full SOURCE TRAIN mean without slack"
    ):
        raise CampaignError("Goal-2 residual closeout or +5% endpoint contract is not verified")
    dependencies = binding.get("input_dependencies", {})
    if (
        dependencies.get("goal1_closeout_sha256") != GOAL1_COMPLETION_ASSESSMENT_SHA256
        or dependencies.get("goal2_constraint_clarification_sha256")
        != GOAL2_CONSTRAINT_CLARIFICATION_SHA256
    ):
        raise CampaignError("Goal-2 residual binding does not include the frozen Goal-1 and endpoint receipts")
    return {
        "schema_version": 1,
        "files": files,
        "source_snapshot_manifest_sha256": GOAL1_SNAPSHOT_MANIFEST_SHA256,
        "goal1_closeout_sha256": GOAL1_COMPLETION_ASSESSMENT_SHA256,
        "goal2_constraint_clarification_sha256": GOAL2_CONSTRAINT_CLARIFICATION_SHA256,
    }


def _store_goal2_residual_approval_binding(
    root: Path, binding: Mapping[str, Any]
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    admission = _acquire_admission_lock(root)
    try:
        manifest = _campaign_manifest(root)
        current = manifest.get("goal2_residual_probe_approval_binding")
        if current is not None and current != binding:
            raise CampaignError("Goal-2 residual probe approval binding changed after recording")
        if current is None:
            manifest["goal2_residual_probe_approval_binding"] = dict(binding)
            _atomic_json(root / "manifests/campaign.json", manifest)
        return manifest
    finally:
        _release_campaign_lock(admission)


def _goal2_offline_diagnostic_seconds(
    root: Path, current: Mapping[str, Any]
) -> float:
    completed = _ledger_action_seconds(root, GOAL2_RESIDUAL_ACTION)
    active = sum(
        float(record.get("elapsed_seconds", 0.0))
        for record in current.get("active_processes", [])
        if record.get("action") == GOAL2_RESIDUAL_ACTION
    )
    return OFFLINE_GPU_DIAGNOSTIC_SECONDS + completed + active


def _campaign_ledger_rows(root: Path, filename: str = "launch_ledger.jsonl") -> list[dict[str, Any]]:
    path = root / "manifests" / filename
    if not path.is_file():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise CampaignError(f"invalid {filename} row {number}: {error}") from error
        if not isinstance(row, dict):
            raise CampaignError(f"invalid {filename} row {number}: expected an object")
        rows.append(row)
    return rows


def _goal2_residual_is_meaningful(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    expected_binding = _goal2_residual_approval_binding(root)
    if manifest.get("goal2_residual_probe_approval_binding") != expected_binding:
        raise CampaignError("Goal-2 residual probe was not run under the stored immutable binding")
    result_path = root / GOAL2_RESIDUAL_RESULT
    if not result_path.is_file():
        raise CampaignError("the predeclared Goal-2 residual diagnostic has no result")
    result = _read_json(result_path)
    binding_path = root / GOAL2_RESIDUAL_APPROVAL_FILES["binding"][0]
    if (
        result.get("status") != "complete"
        or result.get("schema_version") != 1
        or result.get("result_schema") != GOAL2_RESIDUAL_RESULT_SCHEMA
        or result.get("campaign_id") != manifest.get("campaign_id")
        or result.get("plan_sha256") != GOAL2_RESIDUAL_APPROVAL_FILES["plan"][1]
        or result.get("script_sha256") != GOAL2_RESIDUAL_APPROVAL_FILES["script"][1]
        or result.get("binding_sha256") != _sha256(binding_path)
        or result.get("source_snapshot_manifest_sha256") != GOAL1_SNAPSHOT_MANIFEST_SHA256
        or result.get("engineering_gate_passed_both_states") is not True
        or result.get("no_parameter_vectors_written") is not True
    ):
        raise CampaignError("Goal-2 residual result does not pass its frozen schema and source bindings")
    execution = result.get("execution", {})
    if (
        execution.get("source_immutable_verified") is not True
        or execution.get("optimizer_constructed") is not False
        or execution.get("optimizer_steps") != 0
        or execution.get("parameter_updates") is not False
        or execution.get("test_access") is not False
    ):
        raise CampaignError("Goal-2 residual result includes optimizer or TEST activity")
    g1_runs = _goal1_residual_runs_complete(root)
    g1s_run = result.get("inputs", {}).get("target_g1s_run", {})
    if (
        g1s_run.get("run_path") != str(g1_runs["G1-S"])
        or g1s_run.get("global_step") != 3800
        or g1s_run.get("epoch") != MAX_G1_EPOCHS
    ):
        raise CampaignError("Goal-2 residual target is not the completed G1-S epoch-100 state")
    summaries = result.get("state_summaries", {})
    state_names = {"SOURCE", "G1-S-final-100"}
    if set(summaries) != state_names:
        raise CampaignError("Goal-2 residual result does not contain both predeclared states")
    state_rows = {item.get("state_name"): item for item in result.get("states", [])}
    if set(state_rows) != state_names:
        raise CampaignError("Goal-2 residual batch diagnostics do not match the two predeclared states")
    for name in sorted(state_names):
        summary = summaries[name]
        batches = state_rows[name].get("batches", [])
        residual = summary.get("residual_fraction_median")
        orthogonality = summary.get("max_relative_tangent_orthogonality_error")
        if (
            summary.get("engineering_gate_passed") is not True
            or summary.get("batch_count") != 4
            or not isinstance(residual, (int, float))
            or not math.isfinite(float(residual))
            or float(residual) < GOAL2_MIN_RESIDUAL_FRACTION
            or not isinstance(orthogonality, (int, float))
            or not math.isfinite(float(orthogonality))
            or float(orthogonality) > GOAL2_MAX_ORTHOGONALITY_ERROR
            or len(batches) != 4
            or any(
                batch.get("family_gradient_diagnostic", {}).get(
                    "meaningful_finite_nonzero_target"
                )
                is not True
                for batch in batches
            )
        ):
            raise CampaignError(f"Goal-2 residual is not meaningful and finite for {name}")
    return {"path": str(result_path), "sha256": _sha256(result_path), "status": "complete"}


def _goal2_smoke_epoch1_checkpoint_receipt(
    root: Path, manifest: Mapping[str, Any], run: Path
) -> dict[str, Any]:
    receipt_path = root / GOAL2_SMOKE_EPOCH1_RECEIPT
    receipt = _read_json(receipt_path)
    run = run.resolve(strict=True)
    try:
        checkpoint_path = Path(str(receipt.get("checkpoint_path", ""))).resolve(strict=True)
    except OSError as error:
        raise CampaignError("Goal-2 smoke epoch-one checkpoint is missing") from error
    _verify_goal2_source_binding(manifest)
    source_hashes = _stage_source_bindings(manifest)["goal2"]["files_sha256"]
    g1s_run = _goal1_residual_runs_complete(root)["G1-S"]
    g1s_manifest = _read_json(g1s_run / "run_manifest.json")
    expected_source_hashes = g1s_manifest.get("source_hashes")
    if not isinstance(expected_source_hashes, Mapping) or not expected_source_hashes:
        raise CampaignError("completed G1-S run manifest has no immutable SOURCE artifact hashes")
    if expected_source_hashes.get("checkpoint") != manifest.get("source_checkpoint_sha256"):
        raise CampaignError("G1-S source checkpoint differs from the independently pinned Run 11f SOURCE")
    config_path = run / "resolved_config.yaml"
    ordered_reference_path = run / "artifacts/lpc_ordered_reference.json"
    run_manifest = _read_json(run / "run_manifest.json")
    prepared_config_path = _config_path(manifest, GOAL2_SMOKE_ARM)
    prepared_config = _read_yaml(prepared_config_path)
    state = receipt.get("lpc_ordered_state")
    if not isinstance(state, Mapping):
        raise CampaignError("Goal-2 smoke epoch-one receipt has no ordered state")
    try:
        json.dumps(state, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise CampaignError("Goal-2 smoke epoch-one ordered state is not finite JSON") from error
    reference = state.get("reference")
    if not isinstance(reference, Mapping):
        raise CampaignError("Goal-2 smoke epoch-one receipt has no ordered SOURCE reference")
    try:
        ordered_reference_artifact = _read_json(ordered_reference_path)
    except CampaignError as error:
        raise CampaignError("Goal-2 smoke epoch-one ordered reference artifact is missing") from error
    reference_digest = _sha256(ordered_reference_path)
    multipliers = state.get("multipliers")
    if not isinstance(multipliers, Mapping) or set(multipliers) != {
        "global_distribution",
        "cross_spectrum",
    }:
        raise CampaignError("Goal-2 smoke epoch-one ordered multipliers are malformed")
    multiplier_values = [multipliers[name] for name in ("global_distribution", "cross_spectrum")]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in multiplier_values
    ):
        raise CampaignError("Goal-2 smoke epoch-one ordered multipliers are not finite and non-negative")
    implementation_files = {
        name: source_hashes[name]
        for name in (
            "src/phycoflow_reconstruction/training/lpc_ordered.py",
            "src/phycoflow_reconstruction/training/post_training.py",
            "src/phycoflow_reconstruction/training/run_store.py",
            "src/phycoflow_reconstruction/config/validate.py",
        )
    }
    rng_hashes = receipt.get("rng_state_sha256")
    optimizer_steps = receipt.get("optimizer_step_counts")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("status") != "atomic_epoch1_checkpoint_verified"
        or receipt.get("run") != str(run)
        or not checkpoint_path.is_file()
        or not checkpoint_path.is_relative_to(run)
        or checkpoint_path != run / "checkpoints/epoch_001.pt"
        or receipt.get("checkpoint_sha256") != _sha256(checkpoint_path)
        or receipt.get("global_step") != 38
        or receipt.get("reporting_epoch") != 1.0
        or receipt.get("config_sha256") != run_manifest.get("config_sha256")
        or run_manifest.get("config_sha256") != receipt.get("config_sha256")
        or _read_yaml(config_path) != prepared_config
        or _sha256(prepared_config_path)
        != manifest["goal2_configs"][GOAL2_SMOKE_ARM]["sha256"]
        or receipt.get("lpc_ordered_reference_sha256") != reference_digest
        or ordered_reference_artifact.get("reference") != reference
        or ordered_reference_artifact.get("settings") != GOAL2_ORDERED_SETTINGS
        or ordered_reference_artifact.get("source_checkpoint_sha256")
        != expected_source_hashes.get("checkpoint")
        or state.get("version") != 1
        or state.get("completed_updates") != 38
        or state.get("dual_lr") != GOAL2_ORDERED_SETTINGS["dual_lr"]
        or state.get("reference", {}).get("denominator_floor")
        != GOAL2_ORDERED_SETTINGS["denominator_floor"]
        or state.get("reference", {}).get("family_names")
        != ["global_distribution", "cross_spectrum", "topology"]
        or state.get("reference", {}).get("source_sha256")
        != expected_source_hashes.get("checkpoint")
        or state.get("reference", {}).get("no_regression_bounds")
        != {
            "global_distribution": state.get("reference", {}).get("means", {}).get("global_distribution"),
            "cross_spectrum": state.get("reference", {}).get("means", {}).get("cross_spectrum"),
        }
        or not isinstance(receipt.get("lpc_ordered_implementation"), Mapping)
        or dict(receipt["lpc_ordered_implementation"]) != implementation_files
        or receipt.get("external_policy_completed_updates") != 38
        or not isinstance(receipt.get("external_policy_multipliers"), list)
        or len(receipt["external_policy_multipliers"]) != 5
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in receipt.get("external_policy_multipliers", [])
        )
        or not isinstance(rng_hashes, Mapping)
        or not all(
            isinstance(rng_hashes.get(key), str)
            and len(rng_hashes[key]) == 64
            and _is_sha256(rng_hashes[key])
            for key in ("torch_cpu", "index_generator")
        )
        or not isinstance(rng_hashes.get("torch_cuda"), list)
        or not rng_hashes["torch_cuda"]
        or any(not isinstance(item, str) or not _is_sha256(item) for item in rng_hashes["torch_cuda"])
        or not isinstance(optimizer_steps, list)
        or optimizer_steps != [38]
        or receipt.get("source_hashes") != expected_source_hashes
        or not isinstance(receipt.get("created_utc"), str)
    ):
        raise CampaignError("Goal-2 smoke epoch-one checkpoint receipt failed its frozen binding checks")
    try:
        created_utc = datetime.fromisoformat(receipt["created_utc"].replace("Z", "+00:00"))
    except (TypeError, ValueError) as error:
        raise CampaignError("Goal-2 smoke epoch-one receipt has an invalid creation time") from error
    if created_utc.tzinfo is None:
        raise CampaignError("Goal-2 smoke epoch-one receipt time must include a timezone")
    return {
        "path": str(receipt_path.resolve(strict=True)),
        "sha256": _sha256(receipt_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "config_sha256": receipt["config_sha256"],
        "reference_sha256": reference_digest,
    }


def _goal2_smoke_recovery_verification(
    root: Path, manifest: Mapping[str, Any], run: Path, checkpoint_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    verification_path = root / GOAL2_SMOKE_RECOVERY_VERIFICATION
    verification = _read_json(verification_path)
    required_checks = {
        "checkpoint_and_source_binding",
        "lpc_ordered_reference_and_implementation",
        "per_field_mu_and_external_policy_state",
        "optimizer_and_rng_state",
        "first_resumed_step_is_39",
        "sampler_and_replayed_batch_values_reconciled",
        "log_tail_has_no_duplicate_update",
    }
    details = verification.get("details")
    checks = verification.get("checks")
    if (
        verification.get("schema_version") != 1
        or verification.get("status") != "verified"
        or verification.get("run") != str(run.resolve(strict=True))
        or verification.get("checkpoint_receipt_path") != checkpoint_receipt["path"]
        or verification.get("epoch1_checkpoint_receipt_sha256") != checkpoint_receipt["sha256"]
        or verification.get("checkpoint_sha256") != checkpoint_receipt["checkpoint_sha256"]
        or verification.get("source_binding_sha256") != _stage_source_digest(root, manifest, "goal2")
        or verification.get("config_sha256") != checkpoint_receipt["config_sha256"]
        or verification.get("lpc_ordered_reference_sha256")
        != checkpoint_receipt["reference_sha256"]
        or verification.get("first_resumed_step") != 39
        or verification.get("final_step") != 76
        or not isinstance(checks, Mapping)
        or set(checks) != required_checks
        or any(value is not True for value in checks.values())
        or not isinstance(details, Mapping)
    ):
        raise CampaignError("Goal-2 smoke recovery verification is incomplete or not bound to epoch one")
    return {
        "path": str(verification_path.resolve(strict=True)),
        "sha256": _sha256(verification_path),
        "status": "verified",
        "first_resumed_step": 39,
        "details": dict(details),
    }


def _goal2_smoke_interruption_inventory(
    root: Path,
    run: Path,
    checkpoint_receipt: Mapping[str, Any],
    recovery_details: Mapping[str, Any],
    ledger_reported_epoch: float,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    inventory_path = root / GOAL2_SMOKE_INTERRUPTION_INVENTORY
    try:
        inventory_path = inventory_path.resolve(strict=True)
    except OSError as error:
        raise CampaignError("Goal-2 smoke interruption inventory is missing") from error
    observed_step = recovery_details.get("observed_pre_interrupt_accepted_step")
    if isinstance(observed_step, bool) or not isinstance(observed_step, int):
        raise CampaignError("Goal-2 smoke interruption inventory has no accepted-update cursor")
    progress_epoch = recovery_details.get("progress_metadata_reported_epoch")
    if (
        isinstance(progress_epoch, bool)
        or not isinstance(progress_epoch, (int, float))
        or not math.isfinite(float(progress_epoch))
    ):
        raise CampaignError("Goal-2 smoke interruption inventory has no finite progress cursor")
    inventory = _read_json(inventory_path)
    if (
        not inventory_path.is_relative_to(root)
        or recovery_details.get("interruption_inventory_path") != str(inventory_path)
        or recovery_details.get("interruption_inventory_sha256") != _sha256(inventory_path)
        or recovery_details.get("atomic_recovery_checkpoint_epoch") != 1.0
        or progress_epoch not in {0.0, ledger_reported_epoch}
        or recovery_details.get("discarded_durable_accepted_updates") != observed_step - 38
        or recovery_details.get("possible_additional_inflight_update_upper_bound") != 1
        or recovery_details.get("first_ordered_row_bitwise_identical") is not True
        or recovery_details.get("all_preserved_tail_sampler_decisions_identical") is not True
        or recovery_details.get("index_rng_matches_exact_38_update_advance") is not True
        or inventory.get("status") != "stopped_before_resume"
        or inventory.get("run") != str(run.resolve(strict=True))
        or inventory.get("pre_stop_epoch1_receipt") != checkpoint_receipt["path"]
        or inventory.get("discarded_accepted_steps_after_checkpoint") != observed_step - 38
        or inventory.get("possible_additional_inflight_update_upper_bound") != 1
        or inventory.get("conservative_first_segment_epochs_charged")
        != GOAL2_RECOVERY_SMOKE_EPOCHS
        or not 39 <= observed_step <= GOAL2_RECOVERY_SMOKE_EPOCHS * 38
    ):
        raise CampaignError("Goal-2 smoke interruption inventory does not prove a stale-progress epoch-one stop")

    metrics = inventory.get("metrics")
    if not isinstance(metrics, Mapping):
        raise CampaignError("Goal-2 smoke interruption inventory has no metric-log inventory")
    expected_logs = {
        "lpc_ordered_updates": ("lpc_ordered_updates.jsonl", observed_step),
        "fast_lagrangian_updates": ("fast_lagrangian_updates.jsonl", observed_step),
        "lr_warmup_updates": ("lr_warmup_updates.jsonl", observed_step),
        "history": ("history.jsonl", 38),
    }
    for name, (filename, expected_last_step) in expected_logs.items():
        record = metrics.get(name)
        expected_path = root / "analysis/goal2_smoke_pre_resume" / filename
        if (
            not isinstance(record, Mapping)
            or record.get("path") != str(expected_path)
            or not expected_path.is_file()
            or record.get("sha256") != _sha256(expected_path)
            or record.get("last_complete_step") != expected_last_step
        ):
            raise CampaignError(f"Goal-2 smoke interruption inventory changed its {name} log binding")
        try:
            rows = [
                json.loads(line)
                for line in expected_path.read_text(encoding="utf-8").splitlines()
            ]
        except (OSError, json.JSONDecodeError) as error:
            raise CampaignError(f"Goal-2 smoke interruption {name} log is incomplete") from error
        expected_count = 1 if name == "history" else expected_last_step
        if len(rows) != expected_count or record.get("complete_rows") != expected_count:
            raise CampaignError(f"Goal-2 smoke interruption {name} row count does not match its receipt")
        if name == "history":
            if rows[0].get("epoch") != 1.0 or rows[0].get("epoch_complete") is not True:
                raise CampaignError("Goal-2 smoke history log does not end at the atomic epoch-one checkpoint")
        else:
            steps = [row.get("step") for row in rows]
            if steps != list(range(1, expected_last_step + 1)):
                raise CampaignError(f"Goal-2 smoke {name} tail is not a contiguous accepted-update sequence")
            if name == "lpc_ordered_updates" and any(
                row.get("ordered", {}).get("accepted_update") != step
                for row, step in zip(rows, steps, strict=True)
            ):
                raise CampaignError("Goal-2 ordered update tail contains a missing or repeated accepted cursor")
    return {
        "path": str(inventory_path),
        "sha256": _sha256(inventory_path),
        "observed_pre_interrupt_accepted_step": observed_step,
        "atomic_recovery_checkpoint_epoch": 1.0,
        "progress_metadata_reported_epoch": progress_epoch,
    }


def _goal2_smoke_recovery_receipt(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    runs = _run_dirs(root, GOAL2_SMOKE_ARM)
    if len(runs) != 1:
        raise CampaignError("Goal-2 formal gate requires exactly one recovery-smoke lineage")
    run = runs[0].resolve(strict=True)
    record = _run_record(run)
    run_manifest = _read_json(run / "run_manifest.json")
    config = _read_yaml(run / "resolved_config.yaml")
    prepared = _read_yaml(_config_path(manifest, GOAL2_SMOKE_ARM))
    if (
        record.get("status") != "completed"
        or record.get("completed_epochs") != 2.0
        or record.get("completed_updates") != 76
        or record.get("steps_per_epoch") != 38
        or run_manifest.get("experiment_name")
        != f"_LPCAccel/{manifest['campaign_id']}/{GOAL2_SMOKE_ARM}"
        or config != prepared
    ):
        raise CampaignError("Goal-2 recovery smoke did not complete two epochs on its frozen config")

    rows = [
        row
        for row in _campaign_ledger_rows(root)
        if row.get("arm") == GOAL2_SMOKE_ARM and row.get("action") in {"launch", "resume"}
    ]
    if len(rows) != 2 or [row.get("action") for row in rows] != ["launch", "resume"]:
        raise CampaignError("Goal-2 recovery smoke requires one interrupted launch and one same-lineage resume")
    first, resumed = rows
    source_digest = _stage_source_digest(root, manifest, "goal2")
    if (
        first.get("gpu_index") != 0
        or resumed.get("gpu_index") != 0
        or first.get("source_stage") != "goal2"
        or resumed.get("source_stage") != "goal2"
        or first.get("source_binding_sha256") != source_digest
        or resumed.get("source_binding_sha256") != source_digest
        or Path(first.get("run", "")).resolve() != run
        or Path(resumed.get("run", "")).resolve() != run
        or float(first.get("segment_start_epoch", -1)) != 0.0
        or float(first.get("epochs_charged", -1)) != GOAL2_RECOVERY_SMOKE_EPOCHS
        or first.get("return_code") == 0
        or float(resumed.get("segment_start_epoch", -1)) != 1.0
        or float(resumed.get("segment_end_epoch", -1)) != 2.0
        or float(resumed.get("epochs_charged", -1)) != 1.0
        or resumed.get("return_code") != 0
    ):
        raise CampaignError("Goal-2 recovery ledger does not prove a conservative SIGINT/resume sequence")
    first_end = float(first.get("segment_end_epoch", -1))
    if not math.isfinite(first_end):
        raise CampaignError("Goal-2 interrupted segment has no finite reported progress cursor")
    checkpoint_receipt = _goal2_smoke_epoch1_checkpoint_receipt(root, manifest, run)
    stop_rows = [
        row
        for row in _campaign_ledger_rows(root, "stop_requests.jsonl")
        if row.get("arm") == GOAL2_SMOKE_ARM
        and row.get("pid") == first.get("pid")
        and Path(row.get("run", "")).resolve() == run
        and row.get("signal") == "SIGINT"
        and row.get("checkpoint_receipt_path") == checkpoint_receipt["path"]
        and row.get("checkpoint_receipt_sha256") == checkpoint_receipt["sha256"]
    ]
    if len(stop_rows) != 1:
        raise CampaignError("Goal-2 smoke SIGINT is not bound to the preserved atomic epoch-one checkpoint")
    recovery_verification = _goal2_smoke_recovery_verification(
        root, manifest, run, checkpoint_receipt
    )
    inventory_receipt = _goal2_smoke_interruption_inventory(
        root,
        run,
        checkpoint_receipt,
        recovery_verification["details"],
        first_end,
    )
    progress_epoch = inventory_receipt["progress_metadata_reported_epoch"]
    nominal_tail = 1.0 <= first_end < 2.0 and float(progress_epoch) == first_end
    stale_progress_tail = first_end == 0.0 and float(progress_epoch) == 0.0
    if not nominal_tail and not stale_progress_tail:
        raise CampaignError(
            "Goal-2 interrupted segment progress is outside epoch one and has no bound stale-progress reconciliation"
        )
    charged = sum(float(row.get("epochs_charged", 0.0)) for row in rows)
    if charged != GOAL2_RECOVERY_SMOKE_EPOCH_BUDGET:
        raise CampaignError("Goal-2 recovery smoke exceeded its three-epoch conservative exposure cap")
    return {
        "run": str(run),
        "config_sha256": checkpoint_receipt["config_sha256"],
        "prepared_config_file_sha256": manifest["goal2_configs"][GOAL2_SMOKE_ARM]["sha256"],
        "launch_pid": first["pid"],
        "resume_pid": resumed["pid"],
        "stop_request": stop_rows[0],
        "checkpoint_receipt": checkpoint_receipt,
        "recovery_verification": recovery_verification,
        "interruption_inventory": inventory_receipt,
        "observed_first_segment_end_epoch": first_end,
        "epochs_charged": charged,
        "completed_epochs": 2,
    }


def _goal2_budget_readiness(
    manifest: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, float]:
    if float(current.get("training_epochs_charged", 0.0)) + MAX_G1_EPOCHS + 6 > MAX_TOTAL_EPOCHS:
        raise CampaignError("the single 100-epoch Goal-2 arm exceeds the remaining 556-epoch cap")
    if _goal1_remaining_hours(manifest, current_status=current) > 0:
        raise CampaignError("Goal-2 cannot reserve time while any approved Goal-1 lineage remains open")
    diagnostic_seconds = _goal2_offline_diagnostic_seconds(Path(manifest["root"]), current)
    if diagnostic_seconds > MAX_OFFLINE_DIAGNOSTIC_SECONDS:
        raise CampaignError("Goal-2 offline diagnostics exceed the 45-minute cap")
    evaluation_remaining = max(
        0.0,
        _evaluation_reserve_hours(manifest)
        - _evaluation_spent_seconds(Path(manifest["root"]), current) / 3600,
    )
    future_seconds = GOAL2_RUNTIME_QUOTA_SECONDS + evaluation_remaining * 3600
    future_seconds += GPU_CAP_GRACE_SECONDS * MAX_CONCURRENT_GPU_RUNS
    available_seconds = MAX_GPU_SECONDS - float(current.get("gpu_seconds_charged", 0.0))
    if future_seconds > available_seconds:
        raise CampaignError("the 16-hour cap cannot cover Goal-2, remaining evaluation, and watchdog reserves")
    return {
        "reported_gpu_seconds": float(current.get("gpu_seconds_charged", 0.0)),
        "goal2_gpu_seconds_reserved": GOAL2_RUNTIME_QUOTA_SECONDS,
        "evaluation_gpu_seconds_reserved": evaluation_remaining * 3600,
        "watchdog_shutdown_gpu_seconds_reserved": GPU_CAP_GRACE_SECONDS * MAX_CONCURRENT_GPU_RUNS,
        "available_gpu_seconds": available_seconds,
    }


def _goal2_smoke_gate(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not manifest.get("gates", {}).get("goal2_smoke_authorized"):
        raise CampaignError("the bounded Goal-2 smoke requires its explicit schema/unit-test gate")
    record = manifest.get("gate_evidence", {}).get("goal2-smoke", {})
    path = Path(manifest["root"]) / str(record.get("path", ""))
    if (
        not path.is_file()
        or _sha256(path) != record.get("sha256")
        or record.get("source_binding_sha256")
        != _stage_source_bindings(manifest).get("goal2", {}).get("source_binding_sha256")
    ):
        raise CampaignError("Goal-2 smoke implementation/schema evidence or source binding changed")
    source_binding = _verify_goal2_source_binding(manifest)
    if (
        record.get("goal2_config_sha256")
        != manifest.get("goal2_configs", {}).get(GOAL2_FORMAL_ARM, {}).get("sha256")
        or record.get("smoke_config_sha256")
        != manifest.get("goal2_configs", {}).get(GOAL2_SMOKE_ARM, {}).get("sha256")
        or record.get("source_binding_sha256") != source_binding["source_binding_sha256"]
    ):
        raise CampaignError("Goal-2 smoke gate no longer binds the prepared profiles and source")
    return record


def _goal2_formal_gate(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not manifest.get("gates", {}).get("goal2_authorized"):
        raise CampaignError("G2-O is gated until the residual, recovery, and budget evidence passes")
    record = manifest.get("gate_evidence", {}).get("goal2", {})
    path = Path(manifest["root"]) / str(record.get("path", ""))
    if not path.is_file() or _sha256(path) != record.get("sha256"):
        raise CampaignError("Goal-2 formal readiness evidence changed after authorization")
    evidence = _read_json(path)
    source_binding = _verify_goal2_source_binding(manifest)
    if (
        evidence.get("schema_version") != 1
        or evidence.get("status") != GOAL2_FORMAL_READINESS_STATUS
        or evidence.get("goal2_ready") is not True
        or evidence.get("source_binding_sha256") != source_binding["source_binding_sha256"]
        or record.get("source_binding_sha256") != source_binding["source_binding_sha256"]
    ):
        raise CampaignError("Goal-2 formal readiness receipt no longer matches the applied source binding")
    if record.get("goal2_config_sha256") != manifest.get("goal2_configs", {}).get(
        GOAL2_FORMAL_ARM, {}
    ).get("sha256"):
        raise CampaignError("Goal-2 formal readiness receipt names a changed G2-O config")
    _goal2_residual_is_meaningful(Path(manifest["root"]), manifest)
    _goal2_smoke_recovery_receipt(Path(manifest["root"]), manifest)
    return record


def goal2_residual_probe(
    root: Path,
    run_dir: Path,
    *,
    max_seconds: float = GOAL2_RESIDUAL_MAX_SECONDS,
    execute: bool = False,
) -> str:
    manifest = _campaign_manifest(root)
    campaign_root = Path(manifest["root"]).resolve(strict=True)
    if not manifest.get("gates", {}).get("goal1_authorized"):
        raise CampaignError("Goal-2 residual diagnostics require the authorized Goal-1 campaign")
    if isinstance(max_seconds, bool) or not 0 < max_seconds <= GOAL2_RESIDUAL_MAX_SECONDS:
        raise CampaignError("Goal-2 residual max-seconds must be positive and at most 600")
    g1_runs = _goal1_residual_runs_complete(campaign_root)
    run_dir = run_dir.resolve(strict=True)
    if run_dir != g1_runs["G1-C"]:
        raise CampaignError("Goal-2 residual outer binding must be the unique completed G1-C run")
    plan = _read_json(campaign_root / GOAL2_RESIDUAL_APPROVAL_FILES["plan"][0])
    anchor = plan.get("admission_anchor", {})
    if anchor.get("run_path") != str(run_dir):
        raise CampaignError("Goal-2 residual plan is frozen to a different G1-C admission anchor")
    binding = _goal2_residual_approval_binding(campaign_root)
    result_path = campaign_root / GOAL2_RESIDUAL_RESULT
    if result_path.exists():
        raise CampaignError("the one predeclared Goal-2 residual probe already has a result")

    command = [
        str(PYTHON),
        str(REPO_ROOT / "scripts/posttraining/fast_lagrangian_evaluate.py"),
        "--campaign-root",
        str(campaign_root),
        "--run",
        str(run_dir),
        "--goal2-residual-probe",
        "--device",
        "cuda:0",
    ]
    command = _with_short_multiprocessing_tmp(command)
    rendered = _rendered_command(campaign_root, command, 0, source_stage="goal2")
    if not execute:
        return rendered

    current = status(campaign_root)
    if current.get("active_processes"):
        raise CampaignError("Goal-2 residual probe requires all campaign GPU processes to be idle")
    diagnostic_spent = _goal2_offline_diagnostic_seconds(campaign_root, current)
    diagnostic_remaining = MAX_OFFLINE_DIAGNOSTIC_SECONDS - diagnostic_spent
    if diagnostic_remaining <= 0:
        raise CampaignError("the 45-minute offline GPU diagnostic cap is exhausted")
    evaluation_remaining = (
        _evaluation_reserve_hours(manifest) * 3600
        - _evaluation_spent_seconds(campaign_root, current)
    )
    if evaluation_remaining <= 0:
        raise CampaignError("the protected evaluation reserve is exhausted")
    binding_path = campaign_root / GOAL2_RESIDUAL_APPROVAL_FILES["binding"][0]
    _store_goal2_residual_approval_binding(campaign_root, binding)
    max_seconds = min(float(max_seconds), diagnostic_remaining)
    _execute_runner(
        manifest,
        "G1-C",
        GOAL2_RESIDUAL_ACTION,
        command,
        reserve_gpu_seconds=_goal1_remaining_hours(manifest, current_status=current) * 3600,
        max_seconds=max_seconds,
        evaluation_reserve_seconds=_evaluation_reserve_hours(manifest) * 3600,
        run=run_dir,
        gpu=0,
        source_stage="goal2",
    )
    if not result_path.is_file():
        raise CampaignError("Goal-2 residual runner exited without its declared JSON artifact")
    result = _read_json(result_path)
    if (
        result.get("status") != "complete"
        or result.get("result_schema") != GOAL2_RESIDUAL_RESULT_SCHEMA
        or result.get("plan_sha256") != GOAL2_RESIDUAL_APPROVAL_FILES["plan"][1]
        or result.get("script_sha256") != GOAL2_RESIDUAL_APPROVAL_FILES["script"][1]
    ):
        raise CampaignError("Goal-2 residual JSON does not match the approved result schema and hashes")
    # Retain the plan binding path explicitly for later gate evidence audits.
    if not binding_path.is_file():
        raise CampaignError("Goal-2 residual plan binding disappeared during execution")
    return f"completed the bounded Goal-2 residual diagnostic; inspect {result_path}"


def _evaluation_spent_seconds(root: Path, current: Mapping[str, Any]) -> float:
    completed = _ledger_action_seconds(root, "evaluate-", prefix=True)
    active = sum(
        float(record.get("elapsed_seconds", 0.0))
        for record in current.get("active_processes", [])
        if str(record.get("action", "")).startswith("evaluate-")
    )
    return completed + active


def _reserved_gpu_seconds(manifest: Mapping[str, Any], arm: str) -> float:
    gate = manifest.get("gate_evidence", {}).get("goal1", {})
    current = status(Path(manifest["root"]))
    evaluation_reserve = max(
        0.0,
        _evaluation_reserve_hours(manifest)
        - _evaluation_spent_seconds(Path(manifest["root"]), current) / 3600,
    )
    goal1_remaining = _goal1_remaining_hours(manifest, current_status=current)
    if arm.startswith("G1-"):
        if arm not in gate.get("approved_arms", []):
            raise CampaignError(f"{arm} has no approved GPU budget reservation")
        return (
            GOAL2_GPU_HOURS_RESERVE
            + evaluation_reserve
            + _goal1_remaining_hours(manifest, skip_arm=arm, current_status=current)
        ) * 3600
    if arm == GOAL2_FORMAL_ARM:
        return (evaluation_reserve + goal1_remaining) * 3600
    return (GOAL2_GPU_HOURS_RESERVE + evaluation_reserve + goal1_remaining) * 3600


def _check_gpu_reserve(manifest: Mapping[str, Any], arm: str) -> None:
    root = Path(manifest["root"])
    current = status(root)
    gate = manifest.get("gate_evidence", {}).get("goal1", {})
    goal1_required = (
        _goal1_remaining_hours(manifest, current_status=current)
        if gate.get("approved_arms")
        else 0.0
    )
    evaluation_remaining = max(
        0.0,
        _evaluation_reserve_hours(manifest) - _evaluation_spent_seconds(root, current) / 3600,
    )
    required = GOAL2_GPU_HOURS_RESERVE + evaluation_remaining + goal1_required
    required += GPU_CAP_GRACE_SECONDS * MAX_CONCURRENT_GPU_RUNS / 3600
    if arm.startswith("G1-") and not gate.get("approved_arms"):
        raise CampaignError(f"{arm} has no approved Goal-1 GPU and evaluation reserve")
    if current["reported_gpu_hours"] + required > MAX_GPU_SECONDS / 3600:
        raise CampaignError(
            f"remaining 16-hour cap cannot cover {arm}, the 2-hour Goal-2 reserve, and the protected Goal-1/evaluation reserves"
        )


def _watchdog_seconds(
    manifest: Mapping[str, Any],
    arm: str,
    action: str,
    current: Mapping[str, Any],
    *,
    reserve_gpu_seconds: float,
    max_seconds: float | None,
    evaluation_reserve_seconds: float | None,
) -> float:
    """Allocate a bounded process watchdog inside the combined campaign budget."""
    active = list(current.get("active_processes", []))
    process_seconds = current.get(
        "training_process_seconds_by_arm", current.get("process_seconds_by_arm", {})
    )
    approved = manifest.get("gate_evidence", {}).get("goal1", {}).get("approved_arms", [])
    goal1_remaining = _goal1_remaining_hours(manifest, current_status=current) * 3600
    formal_goal2_training = arm == GOAL2_FORMAL_ARM and action in {"launch", "resume"}
    goal2_remaining = _goal2_formal_remaining_seconds(current)
    future_goal2 = 0.0 if formal_goal2_training else goal2_remaining
    if arm.startswith("G1-"):
        if arm not in approved:
            raise CampaignError(f"{arm} has no individually approved runtime quota")
        own_goal1 = max(
            0.0,
            _goal1_quota_hours(manifest, arm) * 3600 - float(process_seconds.get(arm, 0.0)),
        )
        future_goal1 = max(0.0, goal1_remaining - own_goal1)
    else:
        own_goal1 = 0.0
        future_goal1 = goal1_remaining

    evaluation_hours = max(
        MIN_EVALUATION_RESERVE_HOURS,
        float(
            manifest.get("gate_evidence", {})
            .get("goal1", {})
            .get("evaluation_reserve_hours", MIN_EVALUATION_RESERVE_HOURS)
        ),
    )
    evaluation_total = (
        float(evaluation_reserve_seconds)
        if evaluation_reserve_seconds is not None
        else evaluation_hours * 3600
    )
    eval_ledger = _ledger_action_seconds(
        Path(manifest["root"]), "evaluate-", prefix=True
    )
    active_eval_elapsed = sum(
        float(record.get("elapsed_seconds", 0.0))
        for record in active
        if str(record.get("action", "")).startswith("evaluate-")
    )
    active_eval_remaining = sum(
        max(0.0, float(record.get("watchdog_seconds", 0.0)) - float(record.get("elapsed_seconds", 0.0)))
        for record in active
        if str(record.get("action", "")).startswith("evaluate-")
    )
    future_evaluation = max(
        0.0, evaluation_total - eval_ledger - active_eval_elapsed - active_eval_remaining
    )
    own_evaluation = 0.0
    if action.startswith("evaluate-"):
        own_evaluation = min(
            future_evaluation,
            float(max_seconds) if max_seconds is not None else future_evaluation,
        )
        future_evaluation -= own_evaluation

    active_evaluation_remaining = sum(
        max(0.0, float(record.get("watchdog_seconds", 0.0)) - float(record.get("elapsed_seconds", 0.0)))
        for record in active
        if str(record.get("action", "")).startswith("evaluate-")
    )
    active_non_goal1_remaining = sum(
        max(0.0, float(record.get("watchdog_seconds", 0.0)) - float(record.get("elapsed_seconds", 0.0)))
        for record in active
        if not str(record.get("arm", "")).startswith("G1-")
        and not str(record.get("action", "")).startswith("evaluate-")
    )
    global_room = (
        MAX_GPU_SECONDS
        - float(current["gpu_seconds_charged"])
        - future_goal1
        - future_evaluation
        - active_evaluation_remaining
        - active_non_goal1_remaining
        - future_goal2
        - GPU_CAP_GRACE_SECONDS * MAX_CONCURRENT_GPU_RUNS
    )
    if arm.startswith("G1-"):
        allocation = min(own_goal1, global_room)
    elif action.startswith("evaluate-"):
        allocation = min(own_evaluation, global_room)
    elif formal_goal2_training:
        allocation = min(goal2_remaining, global_room)
    else:
        available_slots = max(1, MAX_CONCURRENT_GPU_RUNS - len(active))
        allocation = global_room / available_slots
    if max_seconds is not None:
        allocation = min(allocation, float(max_seconds))
    if allocation <= 0:
        raise CampaignError(
            "the combined 16-hour budget has no safe watchdog allocation after Goal-1, Goal-2, evaluation, and diagnostic reserves"
        )
    explicit_ceiling = (
        MAX_GPU_SECONDS
        - float(current["gpu_seconds_charged"])
        - float(reserve_gpu_seconds)
        - GPU_CAP_GRACE_SECONDS * MAX_CONCURRENT_GPU_RUNS
    )
    allocation = min(allocation, explicit_ceiling)
    if allocation <= 0:
        raise CampaignError("explicit future GPU reservations leave no execution time for this process")
    return allocation


def _try_campaign_lock(lock_path: Path):
    """Open the persistent lock inode and take a nonblocking exclusive flock."""
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        lock = os.fdopen(descriptor, "r+", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        lock.close()
        if error.errno in (errno.EACCES, errno.EAGAIN):
            return None
        raise
    return lock


def _acquire_campaign_lock(lock_path: Path):
    lock = _try_campaign_lock(lock_path)
    if lock is None:
        raise CampaignError("another campaign launch, resume, or evaluation holds the campaign lock")
    try:
        lock.seek(0)
        lock.truncate()
        lock.write(f"pid={os.getpid()} acquired_utc={datetime.now(timezone.utc).isoformat()}\n")
        lock.flush()
    except BaseException:
        _release_campaign_lock(lock)
        raise
    return lock


def _release_campaign_lock(lock) -> None:
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    finally:
        lock.close()


def _campaign_lock_available(lock_path: Path) -> bool:
    if not lock_path.exists():
        return True
    lock = _try_campaign_lock(lock_path)
    if lock is None:
        return False
    _release_campaign_lock(lock)
    return True


def _acquire_gpu_lock(root: Path, gpu: int):
    gpu = _gpu_index(gpu)
    lock_path = _gpu_lock_path(root, gpu)
    lock = _try_campaign_lock(lock_path)
    if lock is None:
        raise CampaignError(f"physical GPU {gpu} already has an active campaign process")
    try:
        lock.seek(0)
        lock.truncate()
        lock.write(
            f"pid={os.getpid()} gpu_index={gpu} acquired_utc={datetime.now(timezone.utc).isoformat()}\n"
        )
        lock.flush()
    except BaseException:
        _release_campaign_lock(lock)
        raise
    return lock


def _admission_lock_path(root: Path) -> Path:
    return root / "manifests/campaign_admission.lock"


def _acquire_admission_lock(root: Path, *, timeout_seconds: float = ADMISSION_LOCK_TIMEOUT_SECONDS):
    """Take the short campaign-wide lock without leaving unbounded waiters."""
    deadline = time.monotonic() + timeout_seconds
    lock_path = _admission_lock_path(root)
    while True:
        lock = _try_campaign_lock(lock_path)
        if lock is not None:
            try:
                lock.seek(0)
                lock.truncate()
                lock.write(
                    f"pid={os.getpid()} acquired_monotonic_ns={time.monotonic_ns()}\n"
                )
                lock.flush()
            except BaseException:
                _release_campaign_lock(lock)
                raise
            return lock
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CampaignError(
                f"campaign admission lock stayed busy for {timeout_seconds:g}s; retry explicitly"
            )
        time.sleep(min(LOCK_POLL_INTERVAL_SECONDS, remaining))


def _pid_is_alive(pid_value: Any) -> bool:
    if isinstance(pid_value, bool) or not isinstance(pid_value, int) or pid_value <= 0:
        return False
    return Path(f"/proc/{pid_value}").exists()


def _proc_parent_pid(pid_value: int) -> int | None:
    """Read a live process's parent PID from Linux procfs."""
    try:
        stat = Path(f"/proc/{pid_value}/stat").read_text(encoding="ascii")
        close = stat.rfind(")")
        if close < 0:
            return None
        fields = stat[close + 1 :].split()
        # The suffix starts at proc field 3 (state); field 4 is its parent PID.
        if len(fields) < 2 or fields[0] in {"Z", "X"}:
            return None
        return int(fields[1])
    except (OSError, ValueError):
        return None


def _is_descendant_process(pid_value: int, ancestor_pid: int, *, max_depth: int = 64) -> bool:
    """Return true only while a live PPID chain reaches the verified owner."""
    if pid_value == ancestor_pid or pid_value <= 1 or ancestor_pid <= 1:
        return False
    current = pid_value
    visited: set[int] = set()
    for _ in range(max_depth):
        if current in visited:
            return False
        visited.add(current)
        parent_pid = _proc_parent_pid(current)
        if parent_pid is None or parent_pid <= 1:
            return False
        if parent_pid == ancestor_pid:
            return True
        current = parent_pid
    return False


def _campaign_process_environment(pid_value: int) -> dict[str, str]:
    try:
        entries = Path(f"/proc/{pid_value}/environ").read_bytes().split(b"\0")
    except OSError:
        return {}
    result: dict[str, str] = {}
    for entry in entries:
        key, separator, value = entry.partition(b"=")
        if separator:
            result[key.decode(errors="replace")] = value.decode(errors="replace")
    return result


def _campaign_process_matches_gpu(
    pid_value: int,
    root: Path,
    gpu: int,
    *,
    allow_legacy_gpu_token: bool = False,
) -> bool:
    """Check the inherited admission root and physical-GPU environment."""
    environment = _campaign_process_environment(pid_value)
    if environment.get("PHYCOFLOW_CAMPAIGN_ADMITTED") != str(root.resolve()):
        return False
    expected_gpu = str(gpu)
    if environment.get("CUDA_VISIBLE_DEVICES") != expected_gpu:
        return False
    campaign_gpu = environment.get("PHYCOFLOW_CAMPAIGN_GPU")
    if campaign_gpu == expected_gpu:
        return True
    # Older serial receipts predate the explicit physical-GPU token and were
    # restricted to physical GPU 1; keep that one compatibility case narrow.
    return allow_legacy_gpu_token and gpu == 1 and campaign_gpu is None


def _live_campaign_children(root: Path) -> list[tuple[int, str]]:
    """Find admitted training/evaluation children even if their JSON receipt vanished."""
    admission = f"PHYCOFLOW_CAMPAIGN_ADMITTED={root.resolve()}".encode()
    known_entries = (
        str(CASE_DIR / "run.py").encode(),
        str(REPO_ROOT / "scripts/posttraining/fast_lagrangian_evaluate.py").encode(),
    )
    live: list[tuple[int, str]] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / "cmdline").read_bytes()
            environ = (proc / "environ").read_bytes().split(b"\0")
            cwd = (proc / "cwd").resolve()
        except (OSError, PermissionError):
            continue
        if admission not in environ or not any(entry in command for entry in known_entries):
            continue
        if cwd == REPO_ROOT:
            live.append((int(proc.name), command.replace(b"\0", b" ").decode(errors="replace")))
    return live


def _reject_active_or_orphaned_process(
    root: Path, *, owned_gpu: int | None = None
) -> list[dict[str, Any]]:
    """Verify all admitted processes while allowing one owner per other GPU."""
    active = _active_process_records(root)
    verified = []
    represented_pids = set()
    for record in active:
        pid_value = record.get("pid")
        gpu = _gpu_index(record.get("gpu_index", 1))
        lock_path = _gpu_lock_path(root, gpu)
        child_live = _pid_is_alive(pid_value)
        child_matches = _pid_matches(pid_value, record.get("config"), record.get("run"))
        coordinator_pid = record.get("coordinator_pid")
        coordinator_live = _pid_is_alive(coordinator_pid)
        lock_held = not _campaign_lock_available(lock_path)
        legacy_coordinator = coordinator_pid is None

        if child_live and coordinator_pid is not None and not coordinator_live:
            raise CampaignError(
                f"physical GPU {gpu} has a live campaign child without its coordinator; inspect or stop that orphan before retrying"
            )

        if (
            owned_gpu == gpu
            and lock_held
            and not child_live
            and not coordinator_live
        ):
            # This executor acquired the otherwise-free GPU flock itself.  The
            # metadata is a dead receipt from a prior invocation, not an owner.
            Path(record["metadata_path"]).unlink(missing_ok=True)
            continue

        if child_live and not child_matches:
            raise CampaignError(
                f"active metadata for physical GPU {gpu} names live PID {pid_value} that cannot be verified"
            )
        if child_live and not _campaign_process_matches_gpu(
            pid_value,
            root,
            gpu,
            allow_legacy_gpu_token=legacy_coordinator,
        ):
            raise CampaignError(
                f"active metadata for physical GPU {gpu} names a live PID without matching campaign admission and GPU environment"
            )
        # Older serial receipts have no coordinator PID.  Their held lifetime
        # flock remains the authority while the child is present or winding down.
        if (
            (child_matches or coordinator_live or legacy_coordinator)
            and lock_held
        ):
            if child_matches:
                represented_pids.add(pid_value)
            verified.append(record)
            continue
        if child_live or coordinator_live or lock_held:
            raise CampaignError(
                f"active metadata for physical GPU {gpu} has no verified lifetime lock; inspect it before retrying"
            )
        Path(record["metadata_path"]).unlink(missing_ok=True)

    orphans = _live_campaign_children(root)
    verified_owners = [
        record
        for record in verified
        if record.get("pid") in represented_pids and _pid_is_alive(record.get("pid"))
    ]
    unrepresented = []
    for pid, _command in orphans:
        if pid in represented_pids:
            continue
        descendant_owner = next(
            (
                record
                for record in verified_owners
                if _is_descendant_process(pid, int(record["pid"]))
                and _pid_matches(pid, record.get("config"), record.get("run"))
                and _campaign_process_matches_gpu(
                    pid,
                    root,
                    _gpu_index(record.get("gpu_index", 1)),
                    allow_legacy_gpu_token=record.get("coordinator_pid") is None,
                )
            ),
            None,
        )
        if descendant_owner is None:
            unrepresented.append(pid)
        else:
            represented_pids.add(pid)
    if unrepresented:
        details = ", ".join(str(pid) for pid in unrepresented)
        raise CampaignError(
            f"live admitted campaign child process(es) remain without a matching per-GPU owner: {details}"
        )
    return verified


def _interrupt_process_group(process: subprocess.Popen[Any]) -> int:
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            return process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                return process.wait(timeout=10)
            except subprocess.TimeoutExpired as error:
                raise CampaignError(
                    f"campaign child {process.pid} did not exit after SIGKILL; its active receipt is retained for explicit stop/recovery"
                ) from error


def _execute_runner(
    manifest: Mapping[str, Any],
    arm: str,
    action: str,
    command: list[str],
    *,
    reserve_gpu_seconds: float,
    max_seconds: float | None = None,
    evaluation_reserve_seconds: float | None = None,
    run: Path | None = None,
    gpu: int = 1,
    source_stage: str | None = None,
) -> None:
    root = Path(manifest["root"]).resolve()
    gpu = _gpu_index(gpu)
    source_binding_digest = (
        _stage_source_digest(root, manifest, source_stage)
        if source_stage is not None
        else None
    )
    active_path = _gpu_active_path(root, gpu)
    if max_seconds is not None and (max_seconds <= 0 or not float(max_seconds) < float("inf")):
        raise CampaignError("runner max_seconds must be a finite positive number")
    gpu_lock = _acquire_gpu_lock(root, gpu)
    admission = None
    started = 0.0
    started_monotonic_ns = 0
    started_utc = ""
    process: subprocess.Popen[Any] | None = None
    active_written = False
    budget_interrupt = False
    runtime_interrupt = False
    shutdown_attempted = False
    return_code: int | None = None
    is_evaluation = action.startswith("evaluate-")
    selected_run = run.resolve() if run is not None else None
    start_epoch = 0.0
    try:
        admission = _acquire_admission_lock(root)
        _reject_active_or_orphaned_process(root, owned_gpu=gpu)
        _ensure_gpu_idle(gpu)
        current = _status_data(manifest)
        if any(record.get("arm") == arm for record in current["active_processes"]):
            raise CampaignError(f"{arm} already has an active process on a physical GPU")

        previous_runs = _run_dirs(root, arm)
        if not is_evaluation:
            if selected_run is None and action == "resume" and previous_runs:
                if len(previous_runs) != 1:
                    raise CampaignError(f"{arm} has multiple child runs; resume must name its lineage")
                selected_run = previous_runs[0].resolve()
            if action == "launch" and previous_runs:
                raise CampaignError(f"{arm} already has a child run; use resume for that lineage")
            if selected_run is not None:
                saved_manifest = _read_json(selected_run / "run_manifest.json")
                start_epoch = float(saved_manifest.get("last_checkpoint_epoch", 0.0))

        # Re-read the combined campaign budget under the admission lock.  Other
        # GPUs may run concurrently, but all three share the same ledger/cap.
        watchdog_seconds = _watchdog_seconds(
            manifest,
            arm,
            action,
            current,
            reserve_gpu_seconds=reserve_gpu_seconds,
            max_seconds=max_seconds,
            evaluation_reserve_seconds=evaluation_reserve_seconds,
        )
        if watchdog_seconds <= 0:
            raise CampaignError("the allocated watchdog has no remaining execution time")

        env = _environment(root, gpu, source_stage=source_stage)
        _prepare_keops_build_folder(root, env, gpu)
        gpu_snapshot = _ensure_gpu_idle(gpu)
        started = time.monotonic()
        started_monotonic_ns = time.monotonic_ns()
        started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, start_new_session=True)
        try:
            _atomic_json(
                active_path,
                {
                    "arm": arm,
                    "action": action,
                    "pid": process.pid,
                    "coordinator_pid": os.getpid(),
                    "config": str(_config_path(manifest, arm)),
                    "run": str(selected_run) if selected_run is not None else None,
                    "gpu_index": gpu,
                    "gpu_uuid": gpu_snapshot["uuid"],
                    "source_stage": source_stage,
                    "source_binding_sha256": source_binding_digest,
                    "started_utc": started_utc,
                    "started_monotonic_ns": started_monotonic_ns,
                    "watchdog_seconds": watchdog_seconds,
                    "command": command,
                },
            )
            active_written = True
        except BaseException:
            _interrupt_process_group(process)
            raise
        _release_campaign_lock(admission)
        admission = None

        deadline = started + watchdog_seconds
        next_run_check = 0.0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                runtime_interrupt = is_evaluation
                budget_interrupt = not runtime_interrupt
                shutdown_attempted = True
                return_code = _interrupt_process_group(process)
                break
            try:
                return_code = process.wait(timeout=min(0.5, remaining))
                break
            except subprocess.TimeoutExpired:
                now = time.monotonic()
                if now >= next_run_check:
                    next_run_check = now + 0.5
                    if selected_run is None and not is_evaluation:
                        candidates = _run_dirs(root, arm)
                        if candidates:
                            selected_run = candidates[0].resolve()
                    if selected_run is not None:
                        admission = _acquire_admission_lock(root)
                        try:
                            if active_path.is_file():
                                active = _read_json(active_path)
                                if (
                                    active.get("pid") == process.pid
                                    and active.get("run") != str(selected_run)
                                ):
                                    active["run"] = str(selected_run)
                                    _atomic_json(active_path, active)
                        finally:
                            _release_campaign_lock(admission)
                            admission = None
            except KeyboardInterrupt:
                shutdown_attempted = True
                return_code = _interrupt_process_group(process)
                break
    except BaseException:
        if process is not None and process.poll() is None and not shutdown_attempted:
            shutdown_attempted = True
            _interrupt_process_group(process)
        raise
    finally:
        try:
            if process is not None and process.poll() is not None:
                final_run = selected_run
                if final_run is None:
                    final_runs = _run_dirs(root, arm)
                    if len(final_runs) == 1:
                        final_run = final_runs[0].resolve()
                has_run = bool(final_run and (final_run / "run_manifest.json").is_file())
                end_epoch = _run_record(final_run)["completed_epochs"] if has_run else 0.0
                if is_evaluation:
                    epoch_charge = 0.0
                else:
                    epoch_charge = segment_epoch_charge(
                        process.poll(),
                        _arm_spec(manifest, arm)["epochs"],
                        start_epoch,
                        end_epoch,
                        has_run,
                    )
                    if (
                        arm == GOAL2_SMOKE_ARM
                        and action == "launch"
                        and float(end_epoch or 0.0) < GOAL2_RECOVERY_SMOKE_EPOCHS
                        and has_run
                    ):
                        # A planned SIGINT after checkpoint 1 conservatively
                        # charges the full two-epoch first segment; the one
                        # remaining epoch is allowed only for same-run resume.
                        epoch_charge = max(
                            epoch_charge,
                            GOAL2_RECOVERY_SMOKE_EPOCHS - start_epoch,
                        )
                elapsed_seconds = max(0.0, time.monotonic() - started)
                admission = admission or _acquire_admission_lock(root, timeout_seconds=STOP_WAIT_SECONDS)
                try:
                    ledger = root / "manifests/launch_ledger.jsonl"
                    row = {
                        "arm": arm,
                        "action": action,
                        "pid": process.pid,
                        "coordinator_pid": os.getpid(),
                        "gpu_index": gpu,
                        "gpu_uuid": gpu_snapshot["uuid"],
                        "source_stage": source_stage,
                        "source_binding_sha256": source_binding_digest,
                        "return_code": process.poll(),
                        "budget_interrupt": budget_interrupt,
                        "runtime_interrupt": runtime_interrupt,
                        "epochs_charged": epoch_charge,
                        "segment_start_epoch": start_epoch,
                        "segment_end_epoch": float(end_epoch or 0.0),
                        "epoch_charge_scope": (
                            "success charges observed completed-update delta; interruption charges the remaining configured horizon as a safe upper bound; evaluation charges zero"
                        ),
                        "run": str(final_run) if final_run is not None else None,
                        "started_utc": started_utc,
                        "started_monotonic_ns": started_monotonic_ns,
                        "elapsed_seconds": elapsed_seconds,
                        "watchdog_seconds": watchdog_seconds,
                        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    }
                    descriptor = os.open(ledger, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
                    try:
                        os.write(descriptor, (json.dumps(row, sort_keys=True) + "\n").encode())
                    finally:
                        os.close(descriptor)
                    if active_written and active_path.is_file():
                        active = _read_json(active_path)
                        if active.get("coordinator_pid") == os.getpid() and active.get("pid") == process.pid:
                            active_path.unlink(missing_ok=True)
                finally:
                    _release_campaign_lock(admission)
                    admission = None
        finally:
            if admission is not None:
                _release_campaign_lock(admission)
            _release_campaign_lock(gpu_lock)

    if runtime_interrupt:
        raise CampaignError("evaluation reached its protected max_seconds or reserved GPU-time limit")
    if budget_interrupt:
        raise CampaignError(
            "the 16-hour watchdog interrupted this lineage; resume only if budget remains"
        )
    if return_code is None:
        raise CampaignError("campaign runner ended without a process exit status")
    if return_code != 0:
        raise CampaignError(
            f"campaign runner exited with status {return_code}; resume its own checkpoint"
        )


def _timing_gate(manifest: Mapping[str, Any]) -> None:
    if not manifest.get("gates", {}).get("timing_passed"):
        raise CampaignError("G1 is gated until the measured timing/correctness receipt passes")


def _g1_gate(manifest: Mapping[str, Any], arm: str) -> None:
    if not manifest.get("gates", {}).get("goal1_authorized"):
        raise CampaignError(
            "100-epoch G1 launches are gated until the Goal-1 receipt is authorized"
        )
    gate = manifest.get("gate_evidence", {}).get("goal1", {})
    if arm not in gate.get("approved_arms", []):
        raise CampaignError(f"{arm} was not individually approved by the Goal-1 readiness evidence")


def _source_stage_for_arm(arm: str) -> str:
    return "goal2" if arm in _goal2_arm_specs() else "goal1"


def _check_budget(manifest: Mapping[str, Any], arm: str) -> None:
    current = status(Path(manifest["root"]))
    spec = _arm_spec(manifest, arm)
    next_horizon = int(spec["epochs"])
    if current["training_epochs_charged"] + 6 >= MAX_TOTAL_EPOCHS:
        raise CampaignError(
            "the 556-epoch cap, including its six reserved public-smoke epochs, is exhausted"
        )
    if current["training_epochs_charged"] + next_horizon + 6 > MAX_TOTAL_EPOCHS:
        raise CampaignError(
            "this arm would exceed the 556-epoch cap after reserving six public-smoke epochs"
        )
    if arm.startswith("T") and current["timing_epochs_charged"] + next_horizon > 50:
        raise CampaignError("this attempt would exceed the 50-epoch timing exposure cap")
    if arm == GOAL2_SMOKE_ARM and (
        float(current.get("charged_epochs_by_arm", {}).get(arm, 0.0)) + next_horizon
        > GOAL2_RECOVERY_SMOKE_EPOCH_BUDGET
    ):
        raise CampaignError("the one Goal-2 recovery smoke would exceed its three-epoch exposure cap")
    if arm == GOAL2_FORMAL_ARM and next_horizon != MAX_G1_EPOCHS:
        raise CampaignError("G2-O must remain exactly one 100-epoch lineage")
    if current["reported_gpu_hours"] >= MAX_GPU_SECONDS / 3600:
        raise CampaignError("the 16-hour GPU-use cap is exhausted")
    existing = _run_dirs(Path(manifest["root"]), arm)
    if existing:
        raise CampaignError(f"{arm} already has a child run; use resume for that same lineage")
    count = current["timing_profile_count"]
    if arm.startswith("T") and count >= MAX_TIMING_PROFILES:
        raise CampaignError("the five-profile timing cap is exhausted")
    _check_gpu_reserve(manifest, arm)


def launch(root: Path, arm: str, *, gpu: int = 1, execute: bool = False) -> str:
    manifest = _campaign_manifest(root)
    gpu = _validate_gpu_for_arm(arm, gpu)
    info = _validate_arm(manifest, arm)
    _validate_config_only(_read_yaml(info["config"]))
    spec = info["spec"]
    if spec["stage"] == "goal1":
        _timing_gate(manifest)
        _g1_gate(manifest, arm)
    elif arm == GOAL2_SMOKE_ARM:
        _goal2_smoke_gate(manifest)
    elif arm == GOAL2_FORMAL_ARM:
        _goal2_formal_gate(manifest)
    _check_budget(manifest, arm)
    source_stage = _source_stage_for_arm(arm)
    command = _command(arm, info["config"])
    rendered = _rendered_command(
        Path(manifest["root"]), command, gpu, source_stage=source_stage
    )
    if not execute:
        return rendered
    _execute_runner(
        manifest,
        arm,
        "launch",
        command,
        reserve_gpu_seconds=_reserved_gpu_seconds(manifest, arm),
        gpu=gpu,
        source_stage=source_stage,
    )
    return f"completed {arm}; inspect {status(root)['runs'].get(arm)}"


def _ensure_gpu_idle(gpu: int = 1) -> dict[str, Any]:
    gpu = _gpu_index(gpu)
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-gpu=uuid,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise CampaignError(f"cannot verify physical GPU {gpu} before launch: {error}") from error
    if not output:
        raise CampaignError(f"physical GPU {gpu} was not reported by nvidia-smi")
    fields = [part.strip() for part in output.split(",")]
    if len(fields) != 3:
        raise CampaignError(f"unexpected nvidia-smi physical GPU {gpu} response: {output}")
    try:
        memory_used = int(fields[1])
        utilization = int(fields[2])
    except ValueError as error:
        raise CampaignError(
            f"nvidia-smi did not return numeric usage for physical GPU {gpu}: {output}"
        ) from error
    if memory_used > 256 or utilization > 5:
        raise CampaignError(f"physical GPU {gpu} is occupied; refusing to launch: {output}")
    applications = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )
    if any(line.split(",")[0].strip() == fields[0] for line in applications.splitlines()):
        raise CampaignError(f"physical GPU {gpu} has a compute process; refusing to share it")
    return {
        "index": gpu,
        "uuid": fields[0],
        "memory_used_mib": memory_used,
        "utilization_percent": utilization,
    }


def _ensure_gpu1_idle() -> dict[str, Any]:
    """Compatibility wrapper for callers and tests from the serial launcher."""
    return _ensure_gpu_idle(1)


def _pid_matches(pid_value: Any, config_value: Any, run_value: Any = None) -> bool:
    if isinstance(pid_value, bool) or not isinstance(pid_value, int) or pid_value <= 0:
        return False
    try:
        command = Path(f"/proc/{pid_value}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        cwd = Path(f"/proc/{pid_value}/cwd").resolve()
    except OSError:
        return False
    matches_target = (config_value is not None and str(config_value) in command) or (
        run_value is not None and str(run_value) in command
    )
    known_entry = (
        str(CASE_DIR / "run.py") in command
        or str(REPO_ROOT / "scripts/posttraining/fast_lagrangian_evaluate.py") in command
    )
    return known_entry and matches_target and cwd == REPO_ROOT


def stop(
    root: Path,
    pid: int | None = None,
    *,
    gpu: int = 1,
    timeout_seconds: float = STOP_WAIT_SECONDS,
) -> str:
    manifest = _campaign_manifest(root)
    campaign_root = Path(manifest["root"])
    gpu = _gpu_index(gpu)
    if isinstance(timeout_seconds, bool) or timeout_seconds <= 0 or not timeout_seconds < float("inf"):
        raise CampaignError("stop timeout must be a finite positive number")
    active_path = _gpu_active_path(campaign_root, gpu)
    lock_path = _gpu_lock_path(campaign_root, gpu)
    admission = _acquire_admission_lock(campaign_root)
    try:
        active = _read_json(active_path) if active_path.is_file() else {}
    finally:
        _release_campaign_lock(admission)
    if not active_path.is_file():
        if not _campaign_lock_available(lock_path):
            raise CampaignError(
                "campaign lock is held but active-process metadata is missing; refusing to signal an unverified PID"
            )
        raise CampaignError("there is no active campaign process to stop")
    if _gpu_index(active.get("gpu_index", gpu)) != gpu:
        raise CampaignError(
            f"active metadata is stored under the physical GPU {gpu} path but identifies another device"
        )
    if pid is not None and pid != active.get("pid"):
        raise CampaignError("the requested PID is not the active process recorded by this campaign")
    selected_pid = active.get("pid")
    config = active.get("config")
    run = active.get("run")
    action = str(active.get("action", "training"))
    live = _pid_matches(selected_pid, config, run)
    if _pid_is_alive(selected_pid) and not live:
        raise CampaignError("active PID exists but does not match the recorded campaign command; refusing to signal it")
    if live:
        admission = _acquire_admission_lock(campaign_root)
        try:
            current = _read_json(active_path) if active_path.is_file() else {}
            if current.get("pid") != selected_pid:
                raise CampaignError("active campaign process changed before stop could signal it")
            if _pid_matches(selected_pid, config, run):
                try:
                    os.killpg(int(selected_pid), signal.SIGINT)
                except ProcessLookupError:
                    live = False
                except OSError as error:
                    raise CampaignError(
                        f"could not send SIGINT to verified campaign process {selected_pid}: {error}"
                    ) from error
                else:
                    stop_path = campaign_root / "manifests/stop_requests.jsonl"
                    request = {
                        "arm": active.get("arm"),
                        "action": action,
                        "pid": selected_pid,
                        "coordinator_pid": active.get("coordinator_pid"),
                        "gpu_index": gpu,
                        "run": active.get("run"),
                        "config": active.get("config"),
                        "signal": "SIGINT",
                        "requested_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    }
                    if active.get("arm") == GOAL2_SMOKE_ARM:
                        checkpoint_receipt_path = campaign_root / GOAL2_SMOKE_EPOCH1_RECEIPT
                        if checkpoint_receipt_path.is_file():
                            request["checkpoint_receipt_path"] = str(
                                checkpoint_receipt_path.resolve(strict=True)
                            )
                            request["checkpoint_receipt_sha256"] = _sha256(
                                checkpoint_receipt_path
                            )
                    descriptor = os.open(
                        stop_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600
                    )
                    try:
                        os.write(descriptor, (json.dumps(request, sort_keys=True) + "\n").encode())
                    finally:
                        os.close(descriptor)
            else:
                live = False
        finally:
            _release_campaign_lock(admission)

    deadline = time.monotonic() + timeout_seconds
    while True:
        still_live = _pid_matches(selected_pid, config, run)
        admission = _try_campaign_lock(_admission_lock_path(campaign_root))
        gpu_lock = _try_campaign_lock(lock_path) if admission is not None else None
        if admission is not None and gpu_lock is None:
            _release_campaign_lock(admission)
            admission = None
        if admission is not None and gpu_lock is not None:
            try:
                current = _read_json(active_path) if active_path.is_file() else None
                if not still_live and current is None:
                    message = (
                        f"campaign process {selected_pid} exited and its coordinator released the lock"
                    )
                    if action.startswith("evaluate-"):
                        return f"{message}; inspect the validation artifacts before another evaluation"
                    return f"{message}; resume from its last atomic checkpoint if needed"
                if current is not None and current.get("pid") != selected_pid:
                    raise CampaignError(
                        "active campaign metadata changed while stop was waiting; inspect status before resuming"
                    )
                if not still_live and current is not None and not _pid_is_alive(
                    current.get("coordinator_pid")
                ):
                    active_path.unlink(missing_ok=True)
                    message = f"orphaned campaign process {selected_pid} exited; its stale receipt was cleared"
                    if action.startswith("evaluate-"):
                        return f"{message}; inspect the validation artifacts before another evaluation"
                    return f"{message}; resume from its last atomic checkpoint if needed"
            finally:
                _release_campaign_lock(gpu_lock)
                _release_campaign_lock(admission)
        if time.monotonic() >= deadline:
            state = "process is still live" if still_live else "coordinator still holds the campaign lock"
            raise CampaignError(
                f"stop timed out after {timeout_seconds:g}s: {state}; do not resume until status confirms the lock is released"
            )
        time.sleep(min(LOCK_POLL_INTERVAL_SECONDS, max(0.0, deadline - time.monotonic())))


def resume(
    root: Path, arm: str, run_dir: Path, *, gpu: int = 1, execute: bool = False
) -> str:
    manifest = _campaign_manifest(root)
    gpu = _validate_gpu_for_arm(arm, gpu)
    info = _validate_arm(manifest, arm)
    if info["spec"]["stage"] == "goal1":
        _timing_gate(manifest)
        _g1_gate(manifest, arm)
    elif arm == GOAL2_SMOKE_ARM:
        _goal2_smoke_gate(manifest)
    elif arm == GOAL2_FORMAL_ARM:
        _goal2_formal_gate(manifest)
    run_dir = run_dir.resolve(strict=True)
    campaign_runs = (Path(manifest["root"]) / "runs").resolve()
    if not run_dir.is_relative_to(campaign_runs):
        raise CampaignError("resume path is outside this campaign's owned /data2 runs")
    run_manifest = _read_json(run_dir / "run_manifest.json")
    if run_manifest.get("experiment_name") != f"_LPCAccel/{manifest['campaign_id']}/{arm}":
        raise CampaignError("resume run belongs to another arm or campaign")
    run_config = _read_yaml(run_dir / "resolved_config.yaml")
    if int(run_config.get("optimization", {}).get("epochs", 999999)) > 100:
        raise CampaignError("refusing to resume any lineage whose horizon exceeds 100 epochs")
    if run_config != _read_yaml(info["config"]):
        raise CampaignError("resume config differs from the immutable campaign arm config")
    _validate_config_only(run_config)
    current = status(Path(manifest["root"]))
    remaining = max(
        0.0,
        float(run_config["optimization"]["epochs"])
        - float(run_manifest.get("last_checkpoint_epoch", 0.0)),
    )
    if remaining <= 0:
        raise CampaignError("the named campaign lineage has no remaining configured epochs")
    if arm == GOAL2_SMOKE_ARM:
        charged = float(current.get("charged_epochs_by_arm", {}).get(arm, 0.0))
        if charged + remaining > GOAL2_RECOVERY_SMOKE_EPOCH_BUDGET:
            raise CampaignError(
                "same-lineage recovery would exceed the three-epoch conservative smoke budget"
            )
    if current["training_epochs_charged"] + remaining + 6 > MAX_TOTAL_EPOCHS:
        raise CampaignError("resuming with discarded exposure charged would exceed 556 epochs")
    if arm.startswith("T") and current["timing_epochs_charged"] + remaining > 50:
        raise CampaignError(
            "resuming with discarded exposure charged would exceed 50 timing epochs"
        )
    _check_gpu_reserve(manifest, arm)
    command = _command(arm, info["config"], run_dir)
    source_stage = _source_stage_for_arm(arm)
    rendered = _rendered_command(
        Path(manifest["root"]), command, gpu, source_stage=source_stage
    )
    if not execute:
        return rendered
    _execute_runner(
        manifest,
        arm,
        "resume",
        command,
        reserve_gpu_seconds=_reserved_gpu_seconds(manifest, arm),
        run=run_dir,
        gpu=gpu,
        source_stage=source_stage,
    )
    return f"resumed {arm} in {run_dir}"


def evaluate(
    root: Path,
    arm: str,
    run_dir: Path,
    *,
    checkpoint: str = "best",
    max_samples: int = 64,
    max_seconds: float | None = None,
    gpu: int = 1,
    execute: bool = False,
) -> str:
    manifest = _campaign_manifest(root)
    if arm.startswith("G2") and arm != GOAL2_FORMAL_ARM:
        raise CampaignError("only the authorized G2-O lineage supports full A/B/T evaluation")
    gpu = _validate_gpu_for_arm(arm, gpu, enforce_assignment=False)
    if arm not in manifest.get("configs", {}) and arm not in manifest.get("goal2_configs", {}):
        raise CampaignError(f"unknown or unprepared arm {arm!r}")
    if arm in manifest.get("excluded_arms", {}):
        raise CampaignError(
            f"{arm} is excluded by its TRAIN-only pretest: {manifest['excluded_arms'][arm]}"
        )
    if arm == GOAL2_FORMAL_ARM:
        if checkpoint not in {"best", "last"}:
            raise CampaignError("G2-O selected/final validation supports only best and last checkpoints")
        if max_samples != 128:
            raise CampaignError("G2-O selected/final full validation requires the uniform 128-sample panel")
        _goal2_formal_gate(manifest)
        info = _validate_arm(manifest, arm)
        campaign_root = Path(manifest["root"]).resolve(strict=True)
        run_dir = run_dir.resolve(strict=True)
        runs = _run_dirs(campaign_root, arm)
        if len(runs) != 1 or runs[0].resolve(strict=True) != run_dir:
            raise CampaignError("G2-O evaluation requires its unique campaign-owned lineage")
        run_record = _run_record(run_dir)
        child_manifest = _read_json(run_dir / "run_manifest.json")
        if (
            not run_dir.is_relative_to((campaign_root / "runs").resolve(strict=True))
            or run_record.get("status") != "completed"
            or run_record.get("completed_epochs") != float(MAX_G1_EPOCHS)
            or run_record.get("completed_updates") != 3800
            or run_record.get("steps_per_epoch") != 38
            or run_record.get("configured_steps") != 3800
            or child_manifest.get("run_id") != run_dir.name
            or child_manifest.get("experiment_name")
            != f"_LPCAccel/{manifest['campaign_id']}/{GOAL2_FORMAL_ARM}"
            or _read_yaml(run_dir / "resolved_config.yaml") != _read_yaml(info["config"])
            or child_manifest.get("source_immutable_verified") is not True
            or child_manifest.get("source_hashes") != child_manifest.get("source_hashes_after")
        ):
            raise CampaignError("G2-O evaluation requires the complete immutable prepared 100-epoch run")
    if isinstance(max_samples, bool) or max_samples not in (32, 64, 96, 128):
        raise CampaignError("validation sample count must be 32, 64, 96, or 128")
    if max_seconds is not None and (
        isinstance(max_seconds, bool)
        or max_seconds <= 0
        or not float(max_seconds) < float("inf")
    ):
        raise CampaignError("evaluation max_seconds must be a finite positive number")
    run_dir = run_dir.resolve(strict=True)
    if not run_dir.is_relative_to((Path(manifest["root"]) / "runs").resolve()):
        raise CampaignError("evaluation run is outside this campaign's /data2 runs")
    run_manifest = _read_json(run_dir / "run_manifest.json")
    if run_manifest.get("experiment_name") != f"_LPCAccel/{manifest['campaign_id']}/{arm}":
        raise CampaignError("evaluation run belongs to another campaign arm")
    # Reuse the full evaluator without generating presentation duplicates.
    source_stage = _source_stage_for_arm(arm)
    command = [
        str(PYTHON),
        str(REPO_ROOT / "scripts/posttraining/fast_lagrangian_evaluate.py"),
        "--campaign-root",
        str(Path(manifest["root"])),
        "--run",
        str(run_dir),
        "--checkpoint",
        checkpoint,
        "--max-samples",
        str(max_samples),
        "--device",
        "cuda:0",
    ]
    command = _with_short_multiprocessing_tmp(command)
    rendered = _rendered_command(
        Path(manifest["root"]), command, gpu, source_stage=source_stage
    )
    if not execute:
        return rendered
    current = status(Path(manifest["root"]))
    goal1_gate = manifest.get("gate_evidence", {}).get("goal1", {})
    reserve_hours = float(goal1_gate.get("evaluation_reserve_hours", MIN_EVALUATION_RESERVE_HOURS))
    remaining_evaluation = reserve_hours * 3600 - _evaluation_spent_seconds(
        Path(manifest["root"]), current
    )
    remaining_training = (
        _goal1_remaining_hours(manifest) * 3600
        if manifest["gates"].get("goal1_authorized")
        else 0.0
    )
    overall_remaining = MAX_GPU_SECONDS - current["gpu_seconds_charged"] - remaining_training
    timeout_seconds = min(remaining_evaluation, overall_remaining) - GPU_CAP_GRACE_SECONDS
    if max_seconds is not None:
        timeout_seconds = min(timeout_seconds, float(max_seconds))
    if timeout_seconds <= 0:
        raise CampaignError(
            "no protected full A/B/T validation-evaluation time remains under the 16-hour GPU cap"
        )
    _execute_runner(
        manifest,
        arm,
        "evaluate-full-abt-validation",
        command,
        reserve_gpu_seconds=remaining_training,
        max_seconds=max_seconds,
        evaluation_reserve_seconds=reserve_hours * 3600,
        run=run_dir,
        gpu=gpu,
        source_stage=source_stage,
    )
    return f"evaluated {arm} checkpoint {checkpoint} with full A/B/T on the evenly spaced validation panel"


def maps(
    root: Path,
    run_dir: Path,
    *,
    max_seconds: float = NATIVE_MAP_MAX_SECONDS,
    execute: bool = False,
) -> str:
    manifest = _campaign_manifest(root)
    if not manifest.get("gates", {}).get("goal1_authorized"):
        raise CampaignError("native maps require the authorized Goal-1 campaign")
    if isinstance(max_seconds, bool) or not 0 < max_seconds <= NATIVE_MAP_MAX_SECONDS:
        raise CampaignError("native-map max-seconds must be positive and at most 180")

    campaign_root = Path(manifest["root"]).resolve(strict=True)
    run_dir = run_dir.resolve(strict=True)
    g1c_root = (campaign_root / "runs/G1-C").resolve(strict=True)
    if not run_dir.is_relative_to(g1c_root):
        raise CampaignError("native maps use the campaign-owned G1-C run as their outer binding")
    run_manifest = _read_json(run_dir / "run_manifest.json")
    if run_manifest.get("experiment_name") != f"_LPCAccel/{manifest['campaign_id']}/G1-C":
        raise CampaignError("native-map run belongs to another campaign arm")

    binding = _native_maps_approval_binding(campaign_root)
    existing_binding = manifest.get("native_map_approval_binding")
    if existing_binding is not None and existing_binding != binding:
        raise CampaignError("native-map approval binding changed after it was recorded")

    command = [
        str(PYTHON),
        str(REPO_ROOT / "scripts/posttraining/fast_lagrangian_evaluate.py"),
        "--campaign-root",
        str(campaign_root),
        "--run",
        str(run_dir),
        "--native-maps",
        "--device",
        "cuda:0",
    ]
    command = _with_short_multiprocessing_tmp(command)
    rendered = _rendered_command(campaign_root, command, 0, source_stage="goal1")
    if not execute:
        return rendered

    current = status(campaign_root)
    if current.get("active_processes"):
        raise CampaignError("native maps require all campaign GPU processes to be idle")
    goal1_gate = manifest.get("gate_evidence", {}).get("goal1", {})
    if not goal1_gate.get("approved_arms"):
        raise CampaignError("native maps require the approved Goal-1 arm set")
    approved_arms = tuple(goal1_gate.get("approved_arms", ()))
    if set(approved_arms) != {"G1-C", "G1-R", "G1-S"}:
        raise CampaignError("native maps require the complete approved G1-C/G1-R/G1-S campaign")
    for arm in ("G1-C", "G1-R", "G1-S"):
        runs = current.get("runs", {}).get(arm, [])
        if (
            len(runs) != 1
            or runs[0].get("status") != "completed"
            or float(runs[0].get("completed_epochs") or 0.0) != MAX_G1_EPOCHS
            or int(runs[0].get("completed_updates") or 0) != 3800
            or int(runs[0].get("steps_per_epoch") or 0) != 38
        ):
            raise CampaignError(f"native maps require exactly one completed 100-epoch {arm} run")

    goal1_remaining = _goal1_remaining_hours(manifest, current_status=current) * 3600
    reserve_hours = _evaluation_reserve_hours(manifest)
    evaluation_remaining = reserve_hours * 3600 - _evaluation_spent_seconds(
        campaign_root, current
    )
    if evaluation_remaining <= 0:
        raise CampaignError("the protected one-hour evaluation reserve is exhausted")
    manifest = _store_native_maps_approval_binding(campaign_root, binding)
    _execute_runner(
        manifest,
        "G1-C",
        "evaluate-native-maps",
        command,
        reserve_gpu_seconds=goal1_remaining,
        max_seconds=max_seconds,
        evaluation_reserve_seconds=reserve_hours * 3600,
        run=run_dir,
        gpu=0,
        source_stage="goal1",
    )
    return "extracted predeclared native-grid validation maps for the completed Goal-1 arms"


def authorize(root: Path, stage: str, evidence_path: Path) -> dict[str, Any]:
    manifest = _campaign_manifest(root)
    stage_evidence_extra: dict[str, Any] = {}
    evidence_path = evidence_path.resolve(strict=True)
    if not evidence_path.is_relative_to(Path(manifest["root"]).resolve()):
        raise CampaignError("gate evidence must be stored under the owned campaign root")
    evidence = _read_json(evidence_path)
    if stage == "timing":
        if manifest.get("selected_topology_estimator") not in {"full", "p3", "s2"}:
            raise CampaignError(
                "prepare must freeze the measured selected estimator before the timing gate"
            )
        if (
            evidence.get("implementation_verified") is not True
            or evidence.get("correctness_verified") is not True
        ):
            raise CampaignError(
                "timing gate requires passing implementation and correctness evidence"
            )
        if evidence.get("selected_topology_estimator") != manifest["selected_topology_estimator"]:
            raise CampaignError(
                "timing evidence estimator differs from the prepared campaign configs"
            )
        if evidence.get("latency_effect") != "positive":
            raise CampaignError("long Goal-1 arms require a measured positive timing effect")
        profiles = evidence.get("profiles")
        eligible_profiles = set(
            manifest.get("eligible_arms", {}).get("timing", ("T0", "T1", "T2", "T3"))
        )
        if (
            not isinstance(profiles, list)
            or not profiles
            or len(set(profiles)) != len(profiles)
            or len(profiles) > MAX_TIMING_PROFILES
            or any(profile not in eligible_profiles for profile in profiles)
        ):
            raise CampaignError("timing evidence must name only eligible timing profiles")
        if "T0" not in profiles or len(profiles) < 2:
            raise CampaignError(
                "timing evidence must compare the original T0 baseline with at least one candidate"
            )
        for profile in profiles:
            records = [_run_record(path) for path in _run_dirs(Path(manifest["root"]), profile)]
            completed = max(
                (float(record["completed_epochs"] or 0.0) for record in records), default=0.0
            )
            if completed < MAX_TIMING_EPOCHS:
                raise CampaignError(
                    f"timing evidence names {profile} without a complete ten-epoch campaign run"
                )
        manifest["gates"]["timing_passed"] = True
    elif stage == "goal1":
        if not manifest.get("gates", {}).get("timing_passed"):
            raise CampaignError("timing gate must pass before authorizing Goal 1")
        reserve_hours = evidence.get("evaluation_reserve_hours")
        if (
            evidence.get("goal1_ready") is not True
            or isinstance(reserve_hours, bool)
            or not isinstance(reserve_hours, (float, int))
            or reserve_hours <= 0
            or not float(reserve_hours) < float("inf")
        ):
            raise CampaignError(
                "Goal-1 evidence must explicitly declare readiness and evaluation reserve"
            )
        if evidence.get("selected_topology_estimator") != manifest["selected_topology_estimator"]:
            raise CampaignError("Goal-1 evidence estimator differs from prepared configs")
        approved_arms = evidence.get("approved_arms")
        eligible_goal1 = set(
            manifest.get("eligible_arms", {}).get("goal1", ("G1-C", "G1-S", "G1-R"))
        )
        if (
            not isinstance(approved_arms, list)
            or not approved_arms
            or len(set(approved_arms)) != len(approved_arms)
            or any(arm not in eligible_goal1 for arm in approved_arms)
        ):
            raise CampaignError("Goal-1 evidence must explicitly approve eligible, measured arms")
        timing_record = manifest.get("gate_evidence", {}).get("timing")
        if not isinstance(timing_record, dict):
            raise CampaignError("timing gate evidence is missing from the campaign manifest")
        timing_path = Path(timing_record["path"]).resolve(strict=True)
        if _sha256(timing_path) != timing_record.get("sha256"):
            raise CampaignError("timing evidence changed after authorization")
        timing_profiles = _read_json(timing_path).get("profiles", [])
        current_status = status(Path(manifest["root"]))
        projection_by_arm: dict[str, float] = {}
        for arm in approved_arms:
            profile = G1_TIMING_PROFILE[arm]
            if profile not in timing_profiles:
                raise CampaignError(f"{arm} requires a completed measured timing profile {profile}")
            completed = max(
                (
                    float(record["completed_epochs"] or 0.0)
                    for record in current_status["runs"].get(profile, [])
                ),
                default=0.0,
            )
            if completed < MAX_TIMING_EPOCHS:
                raise CampaignError(
                    f"{arm} requires ten completed epochs in {profile} for its cost estimate"
                )
            training_seconds = current_status.get(
                "campaign_training_process_seconds_by_arm",
                current_status.get("campaign_process_seconds_by_arm", {}),
            )
            seconds = max(
                training_seconds.get(profile, 0.0),
                current_status["trainer_reported_seconds_by_arm"].get(profile, 0.0),
            )
            projection_by_arm[arm] = seconds * MAX_G1_EPOCHS / MAX_TIMING_EPOCHS / 3600
        projected_hours = sum(projection_by_arm.values())
        runtime_quotas = {
            arm: hours * GOAL1_RUNTIME_QUOTA_FACTOR
            for arm, hours in projection_by_arm.items()
        }
        evaluation_reserve = max(MIN_EVALUATION_RESERVE_HOURS, float(reserve_hours))
        protected_hours = (
            sum(runtime_quotas.values())
            + GOAL2_GPU_HOURS_RESERVE
            + evaluation_reserve
            + GPU_CAP_GRACE_SECONDS * min(MAX_CONCURRENT_GPU_RUNS, len(approved_arms)) / 3600
        )
        available_hours = MAX_GPU_SECONDS / 3600 - current_status["reported_gpu_hours"]
        if protected_hours > available_hours:
            raise CampaignError(
                "measured timing projections with the 2x runtime allowance, the 2-hour Goal-2 reserve, and at least 1 hour of evaluation reserve exceed the remaining 16-hour budget"
            )
        manifest["gates"]["goal1_authorized"] = True
    elif stage == "goal2-smoke":
        root = Path(manifest["root"]).resolve(strict=True)
        if not manifest.get("gates", {}).get("goal1_authorized"):
            raise CampaignError("Goal-2 smoke requires the completed authorized Goal-1 campaign")
        if not manifest.get("goal2_configs"):
            raise CampaignError("run prepare-goal2 before authorizing the recovery smoke")
        source_binding = _verify_goal2_source_binding(manifest)
        bound_evidence_path = (root / source_binding["evidence_path"]).resolve(strict=True)
        if evidence_path != bound_evidence_path:
            raise CampaignError(
                "Goal-2 smoke gate must use the exact helper/schema/integration test receipt bound to the source pin"
            )
        _goal1_residual_runs_complete(root)
        for arm in (GOAL2_FORMAL_ARM, GOAL2_SMOKE_ARM):
            _validate_config_only(_read_yaml(_validate_arm(manifest, arm)["config"]))
            if _run_dirs(root, arm):
                raise CampaignError("Goal-2 smoke authorization must precede either Goal-2 lineage")
        manifest["gates"]["goal2_implemented"] = True
        manifest["gates"]["goal2_smoke_authorized"] = True
        stage_evidence_extra = {
            "source_binding_sha256": source_binding["source_binding_sha256"],
            "goal2_config_sha256": manifest["goal2_configs"][GOAL2_FORMAL_ARM]["sha256"],
            "smoke_config_sha256": manifest["goal2_configs"][GOAL2_SMOKE_ARM]["sha256"],
            "smoke_epoch_budget": GOAL2_RECOVERY_SMOKE_EPOCH_BUDGET,
        }
    elif stage == "goal2":
        root = Path(manifest["root"]).resolve(strict=True)
        if not manifest.get("gates", {}).get("goal2_smoke_authorized"):
            raise CampaignError("Goal-2 smoke schema/unit-test gate must pass before formal readiness")
        if (
            evidence.get("schema_version") != 1
            or evidence.get("status") != GOAL2_FORMAL_READINESS_STATUS
            or evidence.get("goal2_ready") is not True
        ):
            raise CampaignError("Goal-2 formal gate requires an explicit reviewed readiness receipt")
        source_binding = _verify_goal2_source_binding(manifest)
        if evidence.get("source_binding_sha256") != source_binding["source_binding_sha256"]:
            raise CampaignError("Goal-2 readiness evidence uses a different current numerical source pin")
        closeout = _read_json(root / GOAL1_COMPLETION_ASSESSMENT)
        if closeout.get("scalar_bounded_learning_gate", {}).get("passed") is not True:
            raise CampaignError("G1-S did not pass the bounded-learning prerequisite for LPC ordering")
        residual_receipt = _goal2_residual_is_meaningful(root, manifest)
        recovery_receipt = _goal2_smoke_recovery_receipt(root, manifest)
        current_status = status(root)
        budget_receipt = _goal2_budget_readiness(manifest, current_status)
        if _run_dirs(root, GOAL2_FORMAL_ARM):
            raise CampaignError("the single optional G2-O lineage already exists")
        manifest["gates"]["goal2_authorized"] = True
        stage_evidence_extra = {
            "source_binding_sha256": source_binding["source_binding_sha256"],
            "goal2_config_sha256": manifest["goal2_configs"][GOAL2_FORMAL_ARM]["sha256"],
            "residual_result": residual_receipt,
            "recovery_smoke": recovery_receipt,
            "budget": budget_receipt,
        }
    else:
        raise CampaignError("unsupported campaign authorization stage")
    manifest.setdefault("gate_evidence", {})[stage] = {
        "path": str(evidence_path),
        "sha256": _sha256(evidence_path),
    }
    manifest["gate_evidence"][stage].update(stage_evidence_extra)
    if stage == "goal1":
        manifest["gate_evidence"][stage].update(
            {
                "approved_arms": approved_arms,
                "evaluation_reserve_hours": float(reserve_hours),
                "protected_evaluation_reserve_hours": evaluation_reserve,
                "projected_goal1_hours": projected_hours,
                "projected_hours_by_arm": projection_by_arm,
                "runtime_quota_factor": GOAL1_RUNTIME_QUOTA_FACTOR,
                "runtime_quotas_hours_by_arm": runtime_quotas,
                "runtime_quotas_total_hours": sum(runtime_quotas.values()),
                "goal2_gpu_hours_reserved": GOAL2_GPU_HOURS_RESERVE,
                "watchdog_shutdown_reserve_hours": (
                    GPU_CAP_GRACE_SECONDS * min(MAX_CONCURRENT_GPU_RUNS, len(approved_arms)) / 3600
                ),
                "budget_check": {
                    "reported_gpu_hours_at_authorization": current_status["reported_gpu_hours"],
                    "protected_future_gpu_hours": protected_hours,
                    "cap_gpu_hours": MAX_GPU_SECONDS / 3600,
                },
            }
        )
    _atomic_json(Path(manifest["root"]) / "manifests/campaign.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("--selected-topology", choices=("full", "p3", "s2"))
    subparsers.add_parser("prepare-goal2")
    subparsers.add_parser("status")
    gate = subparsers.add_parser("authorize")
    gate.add_argument("stage", choices=("timing", "goal1", "goal2-smoke", "goal2"))
    gate.add_argument("--evidence", type=Path, required=True)
    source = subparsers.add_parser("bind-source")
    source.add_argument("stage", choices=("goal2",))
    source.add_argument("--evidence", type=Path, required=True)
    source.add_argument("--execute", action="store_true", help="apply this immutable source pin")
    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("arm")
    launch_parser.add_argument(
        "--execute", action="store_true", help="run the existing CLI in the foreground"
    )
    launch_parser.add_argument("--gpu", type=_parse_gpu, default=1, metavar="PHYSICAL_GPU")
    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("--pid", type=int)
    stop_parser.add_argument("--gpu", type=_parse_gpu, default=1, metavar="PHYSICAL_GPU")
    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("arm")
    resume_parser.add_argument("--run", type=Path, required=True)
    resume_parser.add_argument("--execute", action="store_true")
    resume_parser.add_argument("--gpu", type=_parse_gpu, default=1, metavar="PHYSICAL_GPU")
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("arm")
    eval_parser.add_argument("--run", type=Path, required=True)
    eval_parser.add_argument("--checkpoint", default="best")
    eval_parser.add_argument("--max-samples", type=int, default=64)
    eval_parser.add_argument("--max-seconds", type=float)
    eval_parser.add_argument("--execute", action="store_true")
    eval_parser.add_argument("--gpu", type=_parse_gpu, default=1, metavar="PHYSICAL_GPU")
    maps_parser = subparsers.add_parser("maps")
    maps_parser.add_argument("--run", type=Path, required=True, help="the campaign-owned G1-C run")
    maps_parser.add_argument("--max-seconds", type=float, default=NATIVE_MAP_MAX_SECONDS)
    maps_parser.add_argument("--execute", action="store_true")
    residual_parser = subparsers.add_parser("residual")
    residual_parser.add_argument("--run", type=Path, required=True, help="the completed campaign-owned G1-C anchor")
    residual_parser.add_argument("--max-seconds", type=float, default=GOAL2_RESIDUAL_MAX_SECONDS)
    residual_parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.root, args.selected_topology)
        elif args.command == "prepare-goal2":
            result = prepare_goal2(args.root)
        elif args.command == "status":
            result = status(args.root)
        elif args.command == "authorize":
            result = authorize(args.root, args.stage, args.evidence)
        elif args.command == "bind-source":
            result = bind_goal2_source(args.root, args.evidence, apply=args.execute)
        elif args.command == "launch":
            result = launch(args.root, args.arm, gpu=args.gpu, execute=args.execute)
        elif args.command == "stop":
            result = stop(args.root, args.pid, gpu=args.gpu)
        elif args.command == "resume":
            result = resume(args.root, args.arm, args.run, gpu=args.gpu, execute=args.execute)
        elif args.command == "evaluate":
            result = evaluate(
                args.root,
                args.arm,
                args.run,
                checkpoint=args.checkpoint,
                max_samples=args.max_samples,
                max_seconds=args.max_seconds,
                gpu=args.gpu,
                execute=args.execute,
            )
        elif args.command == "maps":
            result = maps(
                args.root,
                args.run,
                max_seconds=args.max_seconds,
                execute=args.execute,
            )
        elif args.command == "residual":
            result = goal2_residual_probe(
                args.root,
                args.run,
                max_seconds=args.max_seconds,
                execute=args.execute,
            )
        else:  # pragma: no cover - argparse constrains this
            raise CampaignError(f"unsupported command {args.command}")
    except CampaignError as error:
        print(f"campaign blocked: {error}", file=sys.stderr)
        return 2
    if isinstance(result, dict):
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
