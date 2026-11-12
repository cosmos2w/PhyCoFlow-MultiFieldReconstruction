"""Resolved matched Round 8 training configuration and the common epoch-10 fork."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import torch

from .run_store import RunStore, config_digest, file_sha256, load_project_checkpoint


def resolved_config(retained_config, *, arm):
    if arm not in {"warmup", "B8", "C8", "qualification"}:
        raise ValueError("unknown Round 8 arm")
    config = deepcopy(retained_config)
    config["optimization"]["epochs"] = 1000
    config["optimization"]["steps_per_epoch"] = 13
    config["runtime"]["plot_every_steps"] = 10
    config["runtime"]["device"] = "cuda:0"
    config["output"]["experiment_name"] = "coherence_compute_round8/" + arm
    config["checkpointing"].update(
        every_epochs=250, validation_every_epochs=250,
        epochs=[1, 10, 100, 250, 500, 750, 1000],
        validation_epochs=[250, 500, 750, 1000], selection_metric="fixed_endpoint")
    fidelity = config["optimization"]["multitask"]["fidelity_budget"]
    fidelity.pop("reference_artifact", None)
    fidelity.pop("reference_sha256", None)
    fidelity.update(relative_increase=0.10, absolute_allowance=0.0)
    external = config["optimization"]["multitask"]["external_policy"]
    external.pop("direction_preference", None)
    # R8 installs its sparse fixed-update diagnostics, avoiding R7's whole-epoch
    # component backward windows and their undeclared additional cost.
    external["round7_diagnostics"] = False
    if config["coherence"]["compute_budget"]["point_count"] != 4096:
        raise ValueError("retained Q4096 geometry is required")
    if config["optimization"]["lr_warmup"]["updates"] != 76:
        raise ValueError("retained 76-update LR warmup is required")
    return config


def equal_state(left, right):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal_state(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(equal_state(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def fork_warmup(warmup_dir, *, case_dir, retained_config, qualification=False):
    warmup = Path(warmup_dir)
    checkpoint_path = warmup / "checkpoints/last.pt"
    checkpoint = load_project_checkpoint(checkpoint_path)
    expected_step = 10 if qualification else 130
    if (checkpoint["global_step"] != expected_step
            or bool(checkpoint.get("qualification_only", False)) != qualification):
        raise ValueError("shared warmup must contain exactly ten completed epochs in its declared qualification/formal mode")
    runtime = checkpoint["round8_runtime"]
    calibration = runtime["calibration"]
    if (runtime["mode"] != "warmup" or runtime["step"] != expected_step
            or calibration["coefficients"] is None
            or [r["epoch"] for r in calibration["records"]] != [1, 3, 6, 8, 10]):
        raise ValueError("warmup fork requires all five probes and frozen calibration")
    # Auxiliary model state is stored as top-level EMA/etc. keys, and optional
    # query/family state must survive too. Only the child config hash changes.
    preserved = tuple(key for key in checkpoint if key != "config_sha256")
    result = {"schema": "round8_common_warmup_fork_v1", "parent_run": str(warmup.resolve()),
              "parent_checkpoint_sha256": file_sha256(checkpoint_path), "shared_updates": expected_step,
              "qualification_only": qualification,
              "fork_epoch": 10, "preserved_state_keys": list(preserved), "arms": {}}
    for arm in ("B8", "C8"):
        config = resolved_config(retained_config, arm=arm)
        if qualification:
            config["optimization"]["steps_per_epoch"] = 1
            config["output"]["experiment_name"] = "coherence_compute_round8/qualification/" + arm
        store = RunStore.create(case_dir, config["output"]["experiment_name"], config,
                                parent_run=str(warmup.resolve()))
        # Every child gets independent files. Parent rolling files remain intact.
        for directory in ("artifacts", "logs", "evaluation"):
            shutil.copytree(warmup / directory, store.run_dir / directory, dirs_exist_ok=True)
        payload = deepcopy(checkpoint)
        payload["config_sha256"] = config_digest(config)
        store.save_checkpoint("last", payload)
        restored = store.load_checkpoint("last")
        if any(not equal_state(checkpoint[key], restored[key]) for key in preserved):
            raise ValueError("fork changed shared optimizer/model/dual/RNG state")
        parent_manifest = json.loads((warmup / "run_manifest.json").read_text())
        own_manifest = json.loads((store.run_dir / "run_manifest.json").read_text())
        inherited = {key: value for key, value in parent_manifest.items()
                     if key not in {"experiment_name", "run_id", "config_sha256", "parent_run", "created_utc"}}
        inherited.update(own_manifest)
        (store.run_dir / "run_manifest.json").write_text(json.dumps(inherited, indent=2) + "\n")
        store.update_manifest(round8_fork_parent_sha256=result["parent_checkpoint_sha256"],
                              round8_mode="control" if arm == "B8" else "candidate")
        result["arms"][arm] = {"run": str(store.run_dir),
            "config_sha256": store.config_hash,
            "checkpoint_sha256": file_sha256(store.run_dir / "checkpoints/last.pt"),
            "exact_shared_state": True}
    return result


def validate_formal_child(fork_receipt_path, arm, run_path):
    """Bind each formal child or its resume to the recorded epoch-10 fork."""
    if arm not in {'B8','C8'}:
        raise ValueError('formal child arm must be B8 or C8')
    receipt = json.loads(Path(fork_receipt_path).read_text())
    if (receipt.get('schema') != 'round8_common_warmup_fork_v1'
            or receipt.get('qualification_only') is not False
            or receipt.get('fork_epoch') != 10 or receipt.get('shared_updates') != 130):
        raise ValueError('formal arm requires the verified 130-update common warmup fork')
    entry = receipt.get('arms', {}).get(arm, {})
    run = Path(run_path).resolve()
    if not entry.get('run') or Path(entry['run']).resolve() != run or entry.get('exact_shared_state') is not True:
        raise ValueError('formal arm run does not match its common fork receipt')
    manifest = json.loads((run / 'run_manifest.json').read_text())
    if (manifest.get('round8_fork_parent_sha256') != receipt.get('parent_checkpoint_sha256')
            or manifest.get('round8_mode') != ('control' if arm == 'B8' else 'candidate')
            or manifest.get('config_sha256') != entry.get('config_sha256')):
        raise ValueError('formal arm run lineage or mode differs from the fork receipt')
    checkpoint_path = run / 'checkpoints/last.pt'
    checkpoint = load_project_checkpoint(checkpoint_path)
    step = checkpoint.get('global_step')
    if isinstance(step,bool) or not isinstance(step,int) or not 130 <= step <= 13000:
        raise ValueError('formal child checkpoint lies outside the 10-to-1000 epoch horizon')
    if step == 130 and file_sha256(checkpoint_path) != entry.get('checkpoint_sha256'):
        raise ValueError('initial child checkpoint differs from the recorded exact fork')
    if bool(checkpoint.get('qualification_only',False)):
        raise ValueError('qualification checkpoint cannot enter a formal arm')
    return {'arm':arm,'run':str(run),'global_step':step,'fork_receipt_sha256':file_sha256(fork_receipt_path)}
