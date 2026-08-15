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
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"configuration root must be a mapping: {path}")

    defaults = raw.pop("defaults", [])
    merged: dict[str, Any] = {}
    for item in defaults:
        default_path = (path.parent / str(item)).resolve()
        merged = deep_merge(merged, load_config(default_path))
    merged = deep_merge(merged, raw)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be KEY=VALUE, got {item!r}")
        key, raw_value = item.split("=", 1)
        _set_dotted(merged, key, yaml.safe_load(raw_value))
    return merged
