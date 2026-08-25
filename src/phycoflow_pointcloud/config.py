"""Portable configuration loading and public/internal model-name validation."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PUBLIC_TO_INTERNAL = {
    "GL_rbf_CQ": "GL_rbf_ENH_CQ",
    "GL_rbf_CQ-fast": "GL_rbf_ENH_CQ",
    "GL_rbf_ENH": "GL_rbf_ENH",
    "GL_rbf": "GL_rbf",
}
INTERNAL_NAMES = frozenset({"GL_rbf", "GL_rbf_ENH", "GL_rbf_ENH_CQ"})
PATH_KEYS = ("data", "dataset_stats_path", "save_dir")


@dataclass(frozen=True)
class PublicModelIdentity:
    public_name: str
    internal_backbone: str


def resolve_model_identity(config: Mapping[str, Any]) -> PublicModelIdentity:
    """Resolve and cross-check public and historical model identifiers."""
    public = config.get("model_name")
    internal = config.get("backbone")
    if public is None and internal is None:
        public, internal = "GL_rbf_ENH", "GL_rbf_ENH"
    elif public is None:
        internal = str(internal)
        if internal not in INTERNAL_NAMES:
            raise ValueError(f"Unknown historical backbone {internal!r}.")
        public = "GL_rbf_CQ" if internal == "GL_rbf_ENH_CQ" else internal
    else:
        public = str(public)
        if public not in PUBLIC_TO_INTERNAL:
            raise ValueError(
                f"Unknown public model name {public!r}; expected one of "
                f"{sorted(PUBLIC_TO_INTERNAL)}."
            )
        expected = PUBLIC_TO_INTERNAL[public]
        if internal is not None and str(internal) != expected:
            raise ValueError(
                f"Conflicting model identifiers: model_name={public!r} maps to "
                f"backbone={expected!r}, but backbone={internal!r} was supplied."
            )
        internal = expected
    return PublicModelIdentity(str(public), str(internal))


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_public_config(
    path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
    resolve_paths: bool = True,
) -> dict[str, Any]:
    """Load a YAML config, apply explicit overrides, and validate identity.

    Relative runtime paths are rooted at the project directory, independent of
    the caller's current working directory. Exact research YAML files remain
    untouched and can be loaded through the same boundary.
    """
    config_path = Path(path).expanduser().resolve()
    loaded = yaml.safe_load(config_path.read_text())
    if not isinstance(loaded, MutableMapping):
        raise TypeError(f"Expected a YAML mapping in {config_path}.")
    config = dict(loaded)
    if overrides:
        config.update(
            {key: value for key, value in overrides.items() if value is not None}
        )
    identity = resolve_model_identity(config)
    config["model_name"] = identity.public_name
    config["backbone"] = identity.internal_backbone
    config.setdefault("coord_dim", 3)
    if int(config["coord_dim"]) <= 0:
        raise ValueError("coord_dim must be positive.")
    if resolve_paths:
        base = Path(root).expanduser().resolve() if root is not None else project_root()
        for key in PATH_KEYS:
            value = config.get(key)
            if value:
                candidate = Path(str(value)).expanduser()
                config[key] = str(
                    candidate if candidate.is_absolute() else base / candidate
                )
    config["config_source"] = str(config_path)
    return config
