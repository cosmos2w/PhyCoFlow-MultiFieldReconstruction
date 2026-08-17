"""Immutable run lineage, atomic checkpoints, histories, and resume checks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
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
        if name == "last":
            # `last.pt` remains the canonical resume name. `latest.pt` is a
            # relative symlink so tools and users accustomed to either name
            # see the same atomically replaced payload without duplicating a
            # potentially multi-gigabyte checkpoint.
            latest = target.with_name("latest.pt")
            temporary_link = target.with_name(".latest.pt.tmp")
            if os.path.lexists(temporary_link):
                temporary_link.unlink()
            temporary_link.symlink_to(target.name)
            os.replace(temporary_link, latest)
        return target

    def load_checkpoint(self, name: str = "last") -> dict[str, Any]:
        path = self.run_dir / "checkpoints" / f"{name}.pt"
        return load_project_checkpoint(path)

    def append_history(self, row: Mapping[str, Any]) -> None:
        path = self.run_dir / "metrics" / "history.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


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
