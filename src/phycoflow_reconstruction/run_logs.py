"""Lightweight paths for run-local histories and diagnostics."""

from __future__ import annotations

from pathlib import Path


def history_paths(run_dir: str | Path, name: str) -> list[Path]:
    """Existing legacy and current history files, in chronological location order."""
    root = Path(run_dir)
    paths: list[Path] = []
    for path in (root / "metrics" / name, root / "logs" / name):
        if path.is_file() and not any(path.samefile(previous) for previous in paths):
            paths.append(path)
    return paths


def log_path(run_dir: str | Path, name: str) -> Path:
    """Canonical destination for new run histories, telemetry and figures."""
    directory = Path(run_dir) / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name
