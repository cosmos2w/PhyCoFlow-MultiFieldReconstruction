"""Immutable run lineage, atomic checkpoints, histories, and resume checks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


def load_project_checkpoint(path: str | Path) -> dict[str, Any]:
    """Restricted-load project checkpoints with audited dependency symbols."""
    safe_symbols: list[Any] = [torch._C._nn.gelu]
    try:
        from neuralop.layers.spectral_convolution import SpectralConv
    except ImportError:
        pass
    else:
        safe_symbols.append(SpectralConv)
    try:
        with torch.serialization.safe_globals(safe_symbols):
            return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - PyTorch before weights_only support
        return torch.load(path, map_location="cpu")


def checkpoint_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return learned tensor state without NeuralOperator's config pseudo-key."""
    state = model.state_dict()
    metadata = state.pop("_metadata", None)
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("model state _metadata pseudo-key must be a mapping")
    return state


def load_model_state_strict(model: torch.nn.Module, state: Mapping[str, Any]) -> None:
    """Strictly load learned tensors after removing one known non-tensor key."""
    learned_state = dict(state)
    metadata = learned_state.pop("_metadata", None)
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("checkpoint _metadata pseudo-key must be a mapping")
    model.load_state_dict(learned_state, strict=True)


def config_digest(config: Mapping[str, Any]) -> str:
    payload = yaml.safe_dump(dict(config), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_state(directory: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=directory, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=directory, text=True
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


class RunStore:
    def __init__(self, run_dir: str | Path, config: Mapping[str, Any]) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.config = dict(config)
        self.config_hash = config_digest(config)

    @classmethod
    def create(
        cls,
        case_dir: str | Path,
        experiment_name: str,
        config: Mapping[str, Any],
        *,
        parent_run: str | None = None,
    ) -> RunStore:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        digest = config_digest(config)
        run_dir = Path(case_dir) / "runs" / experiment_name / f"{timestamp}_{digest[:8]}"
        store = cls(run_dir, config)
        if run_dir.exists():
            raise FileExistsError(f"run directory already exists: {run_dir}")
        for relative in ("checkpoints", "metrics", "artifacts", "evaluation", "logs"):
            (run_dir / relative).mkdir(parents=True, exist_ok=False)
        (run_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
        )
        (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
        environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "git": _git_state(Path(case_dir).resolve()),
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "conflictfree",
                    "gudhi",
                    "h5py",
                    "neuraloperator",
                    "numpy",
                    "pykeops",
                    "PyYAML",
                    "scipy",
                    "torch",
                )
                if _package_exists(name)
            },
            "cuda_devices": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        }
        (run_dir / "environment.json").write_text(
            json.dumps(environment, indent=2), encoding="utf-8"
        )
        manifest = {
            "version": "1",
            "stage": config.get("stage"),
            "case": config.get("case"),
            "experiment_name": experiment_name,
            "run_id": run_dir.name,
            "config_sha256": digest,
            "parent_run": parent_run,
            "created_utc": timestamp,
        }
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        store.set_status("created")
        return store

    @classmethod
    def resume(cls, run_dir: str | Path, config: Mapping[str, Any]) -> RunStore:
        store = cls(run_dir, config)
        manifest = json.loads((store.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest["config_sha256"] != store.config_hash:
            raise ValueError("resume config hash does not match the existing run")
        return store

    @classmethod
    def extend_epochs(cls, run_dir: str | Path, epochs: int) -> RunStore:
        """Explicitly extend a saved run's epoch horizon without changing its identity.

        This metadata-only operation is intended for an operator-approved resume
        of the same child run. It archives the exact pre-extension metadata and
        hard-links the current rolling/selected checkpoints before updating the
        resolved config and its manifest digest.
        """
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer")

        run_path = Path(run_dir).resolve()
        if not run_path.is_dir():
            raise FileNotFoundError(f"run directory does not exist: {run_path}")
        config_path = run_path / "resolved_config.yaml"
        manifest_path = run_path / "run_manifest.json"
        if not config_path.is_file() or not manifest_path.is_file():
            raise ValueError("run is missing its resolved config or manifest")

        original_config_text = config_path.read_text(encoding="utf-8")
        original_manifest_text = manifest_path.read_text(encoding="utf-8")
        config = yaml.safe_load(original_config_text)
        manifest = json.loads(original_manifest_text)
        if not isinstance(config, Mapping) or not isinstance(manifest, dict):
            raise TypeError("run config and manifest must be mappings")

        original_digest = manifest.get("config_sha256")
        if not isinstance(original_digest, str) or config_digest(config) != original_digest:
            raise ValueError("resolved config digest does not match the existing manifest")
        if manifest.get("run_id") != run_path.name:
            raise ValueError("manifest run_id does not match the run directory")
        for identity_key in ("stage", "case"):
            if identity_key in manifest and manifest[identity_key] != config.get(identity_key):
                raise ValueError(f"manifest {identity_key} does not match the resolved config")

        optimization = config.get("optimization")
        if not isinstance(optimization, Mapping):
            raise TypeError("resolved config optimization must be a mapping")
        old_epochs = optimization.get("epochs")
        if isinstance(old_epochs, bool) or not isinstance(old_epochs, int) or old_epochs <= 0:
            raise ValueError("resolved config optimization.epochs must be a positive integer")
        if epochs <= old_epochs:
            raise ValueError("new epoch horizon must exceed optimization.epochs")

        extensions = manifest.get("epoch_extensions", [])
        if not isinstance(extensions, list) or not all(
            isinstance(path, str) for path in extensions
        ):
            raise ValueError("manifest epoch_extensions must be a list of relative paths")
        extension_dir_relative = Path("artifacts") / "epoch_extensions" / f"{old_epochs}_to_{epochs}"
        archive_dir = run_path / extension_dir_relative
        receipt_relative = extension_dir_relative / "receipt.json"
        receipt_relative_text = receipt_relative.as_posix()
        if receipt_relative_text in extensions:
            raise ValueError("manifest already records this epoch extension")
        if archive_dir.exists():
            raise FileExistsError(f"epoch-extension archive already exists: {archive_dir}")

        checkpoint_dir = run_path / "checkpoints"
        last_path = checkpoint_dir / "last.pt"
        checkpoint: dict[str, Any] | None = None
        checkpoint_cursor: dict[str, Any] | None = None
        previous_checkpoint_binding: str | None = None
        if last_path.is_file():
            checkpoint = load_project_checkpoint(last_path)
            global_step = checkpoint.get("global_step")
            if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
                raise ValueError("last checkpoint has an invalid global_step cursor")
            checkpoint_cursor = {
                "global_step": global_step,
                "epoch": manifest.get("last_checkpoint_epoch"),
            }
            binding = checkpoint.get("config_sha256")
            if binding is not None:
                if not isinstance(binding, str):
                    raise ValueError("last checkpoint config_sha256 must be a string")
                if binding != original_digest:
                    raise ValueError("last checkpoint config digest does not match the run manifest")
                previous_checkpoint_binding = binding
            manifest_step = manifest.get("last_checkpoint_step")
            if manifest_step is not None and manifest_step != global_step:
                raise ValueError("last checkpoint cursor does not match the run manifest")

        updated_config = deepcopy(dict(config))
        updated_config["optimization"]["epochs"] = epochs
        new_digest = config_digest(updated_config)
        updated_manifest = dict(manifest)
        updated_manifest["config_sha256"] = new_digest
        updated_manifest["epoch_extensions"] = [*extensions, receipt_relative_text]

        archive_dir.parent.mkdir(parents=True, exist_ok=True)
        archive_dir.mkdir(exist_ok=False)
        metadata_update_started = False
        try:
            for name in (
                "resolved_config.yaml",
                "run_manifest.json",
                "status.json",
                "progress.json",
                "command.txt",
                "environment.json",
            ):
                source = run_path / name
                if source.is_file():
                    shutil.copy2(source, archive_dir / name)

            archived_checkpoints = archive_dir / "checkpoints"
            archived_checkpoint_names = ("last.pt", "best.pt", "best_fidelity.pt")
            existing_checkpoints = [
                name for name in archived_checkpoint_names if (checkpoint_dir / name).is_file()
            ]
            if existing_checkpoints:
                archived_checkpoints.mkdir()
                for name in existing_checkpoints:
                    os.link(checkpoint_dir / name, archived_checkpoints / name)

            receipt: dict[str, Any] = {
                "old_epochs": old_epochs,
                "new_epochs": epochs,
                "checkpoint_cursor": checkpoint_cursor,
                "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "before_metadata_archive": extension_dir_relative.as_posix(),
                "purpose": "explicit epoch-only continuation",
            }
            if previous_checkpoint_binding is not None:
                receipt["previous_checkpoint_config_sha256"] = previous_checkpoint_binding
            _atomic_text(
                archive_dir / "receipt.json",
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            )

            _atomic_text(
                config_path,
                yaml.safe_dump(updated_config, sort_keys=False),
            )
            metadata_update_started = True
            _atomic_text(
                manifest_path,
                json.dumps(updated_manifest, indent=2, sort_keys=True) + "\n",
            )
        except Exception:
            if metadata_update_started:
                try:
                    _atomic_text(config_path, original_config_text)
                    _atomic_text(manifest_path, original_manifest_text)
                except Exception as rollback_error:
                    raise RuntimeError(
                        "epoch extension metadata write failed and rollback was unsuccessful; "
                        f"pre-extension archive remains at {archive_dir}"
                    ) from rollback_error
            shutil.rmtree(archive_dir, ignore_errors=True)
            raise

        return cls(run_path, updated_config)

    def set_status(self, status: str, **details: Any) -> None:
        payload = {
            "status": status,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            **details,
        }
        _atomic_text(self.run_dir / "status.json", json.dumps(payload, indent=2) + "\n")

    def update_manifest(self, **details: Any) -> None:
        path = self.run_dir / "run_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(details)
        _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def write_json(self, relative_path: str | Path, payload: Mapping[str, Any]) -> Path:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_text(path, json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
        return path

    def save_artifact(self, name: str, payload: Mapping[str, Any]) -> Path:
        target = self.run_dir / "artifacts" / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(dict(payload), temporary)
        os.replace(temporary, target)
        return target

    def save_checkpoint(self, name: str, payload: Mapping[str, Any]) -> Path:
        target = self.run_dir / "checkpoints" / f"{name}.pt"
        temporary = target.with_suffix(".pt.tmp")
        torch.save(dict(payload), temporary)
        os.replace(temporary, target)
        return target

    def load_checkpoint(self, name: str = "last") -> dict[str, Any]:
        path = self.run_dir / "checkpoints" / f"{name}.pt"
        return load_project_checkpoint(path)

    def append_history(self, row: Mapping[str, Any]) -> None:
        path = self.run_dir / "metrics" / "history.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")

    def recover_metric_histories(self, completed_step: int) -> None:
        """Trim writes beyond the checkpoint, including a torn trailing JSON row.

        Training logs are written before the atomic checkpoint. Preemption may
        therefore leave metrics for updates that recovery will replay. Repeated
        validation of the same checkpoint is reduced to its last complete row.
        """
        for name in (
            "history", "validation_history", "topology_validation", "lr_warmup_updates",
            "fast_lagrangian_updates",
        ):
            path = self.run_dir / "metrics" / f"{name}.jsonl"
            if not path.exists():
                continue
            rows = {}
            lines = path.read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    if index == len(lines) - 1:
                        break
                    raise
                step = int(row["step"])
                if step <= completed_step:
                    rows[step] = line
            _atomic_text(path, "".join(rows[step] + "\n" for step in sorted(rows)))


def _package_exists(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
