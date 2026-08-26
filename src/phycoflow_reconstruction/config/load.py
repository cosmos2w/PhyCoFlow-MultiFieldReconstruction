"""Load YAML configs with deterministic recursive defaults and dotted overrides."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    cursor = config
    parts = key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
        if not isinstance(cursor, dict):
            raise TypeError(f"override path {key!r} crosses a non-mapping value")
    cursor[parts[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config and recursively merge its ``defaults``.

    Defaults are resolved relative to the file that declares them.  The
    private stack makes malformed composition fail immediately with a useful
    cycle instead of recursing until Python's limit; it also keeps the public
    API unchanged for launchers and callers that pass dotted overrides.
    """

    def _load(current: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
        current = current.resolve()
        if current in stack:
            cycle = " -> ".join(str(item) for item in (*stack, current))
            raise ValueError(f"configuration defaults cycle: {cycle}")
        if not current.is_file():
            parent = f" referenced by {stack[-1]}" if stack else ""
            raise FileNotFoundError(f"configuration file not found{parent}: {current}")
        with current.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise TypeError(f"configuration root must be a mapping: {current}")

        defaults = raw.pop("defaults", [])
        if defaults is None:
            defaults = []
        if isinstance(defaults, (str, Path)):
            defaults = [defaults]
        if not isinstance(defaults, (list, tuple)):
            raise TypeError(f"configuration defaults must be a list: {current}")
        merged: dict[str, Any] = {}
        next_stack = (*stack, current)
        for item in defaults:
            if not isinstance(item, (str, Path)) or not str(item).strip():
                raise TypeError(f"configuration default paths must be non-empty strings: {current}")
            default_path = (current.parent / str(item)).resolve()
            merged = deep_merge(merged, _load(default_path, next_stack))
        return deep_merge(merged, raw)

    config = _load(Path(path), ())
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be KEY=VALUE, got {item!r}")
        key, raw_value = item.split("=", 1)
        _set_dotted(config, key, yaml.safe_load(raw_value))
    return config
