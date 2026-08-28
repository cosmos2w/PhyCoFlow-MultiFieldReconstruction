"""Permanent checks for the standalone repository's architectural boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

from phycoflow_reconstruction.config import load_config, validate_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_PACKAGE = "phycoflow_" + "pointcloud"


def test_only_the_canonical_project_namespace_remains() -> None:
    assert not (PROJECT_ROOT / "src" / LEGACY_PACKAGE).exists()
    assert not (PROJECT_ROOT / "Cases").exists()
    assert not (PROJECT_ROOT / "Dataset").exists()

    source_roots = ("src", "tests", "scripts", "cases", "benchmarks")
    paths = sorted(
        path
        for root in source_roots
        for path in (PROJECT_ROOT / root).rglob("*.py")
        if "__pycache__" not in path.parts
    )
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_modules = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        ]
        imported_modules.extend(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert all(
            module != LEGACY_PACKAGE and not module.startswith(f"{LEGACY_PACKAGE}.")
            for module in imported_modules
        ), f"legacy package import remains in {path.relative_to(PROJECT_ROOT)}"


def test_all_canonical_model_and_case_launch_configs_resolve() -> None:
    model_configs = sorted((PROJECT_ROOT / "configs" / "models").glob("*.yaml"))
    launch_configs = sorted((PROJECT_ROOT / "cases").glob("*/configs/base/*.yaml"))
    assert model_configs
    assert launch_configs
    for path in (*model_configs, *launch_configs):
        resolved = load_config(path)
        assert resolved["model"]["name"]


def test_every_canonical_yaml_resolves_and_stage_configs_validate() -> None:
    roots = (PROJECT_ROOT / "configs", PROJECT_ROOT / "cases", PROJECT_ROOT / "benchmarks")
    paths = sorted(
        path
        for root in roots
        for path in root.rglob("*.yaml")
        if not {"_experiments", "readiness", "runs", "manifests", "generated"}.intersection(
            path.parts
        )
    )
    assert paths
    for path in paths:
        config = load_config(path)
        if "stage" not in config or path.is_relative_to(PROJECT_ROOT / "configs" / "defaults"):
            continue
        try:
            validate_config(config)
        except ValueError as error:
            message = str(error)
            incomplete_stage2 = (
                config.get("model", {}).get("name") == "latent_fm"
                and config.get("model", {}).get("stage") == 2
                and not config.get("model", {}).get("stage1_checkpoint")
                and "stage1_checkpoint" in message
            )
            incomplete_posttrain = (
                config.get("stage") == "post_training"
                and not config.get("source_run")
                and "source_run" in message
            )
            assert incomplete_stage2 or incomplete_posttrain, (
                f"unexpected validation failure in {path.relative_to(PROJECT_ROOT)}: {error}"
            )
