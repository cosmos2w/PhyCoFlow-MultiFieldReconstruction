"""Shared command-line routing used by the thin launchers in each case.

The CLI resolves paths relative to the active case, validates configs before
work begins, and lazily imports training code so dataset validation stays fast.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config, validate_config
from .data.factory import open_field_dataset
from .data.manifest import manifest_from_batch
from .data.sensor_protocols import build_observation_batch
from .data.validation import validate_dataset
from .training.common import sensor_protocol_from_config


def _resolve_dataset_path(config: dict[str, Any], case_dir: Path) -> Path:
    path = Path(config["dataset"]["path"])
    return path.resolve() if path.is_absolute() else (case_dir / path).resolve()


def _load_case_config(
    path: Path,
    case_dir: Path,
    case_name: str,
    overrides: list[str],
    *,
    validate_stage: bool = True,
) -> dict[str, Any]:
    config = load_config(path, overrides)
    config.setdefault("case", case_name)
    if config["case"] != case_name:
        raise ValueError(
            f"launcher case {case_name!r} does not match config case {config['case']!r}"
        )
    if config.get("stage") == "post_training":
        source_run = config.get("source_run")
        if source_run:
            source_path = Path(source_run)
            source_path = (
                source_path.resolve()
                if source_path.is_absolute()
                else (case_dir / source_path).resolve()
            )
            config["source_run"] = str(source_path)
            if bool(config.get("inherit_base_config", True)):
                source_config_path = source_path / "resolved_config.yaml"
                if source_config_path.is_file():
                    source_config = load_config(source_config_path)
                    if source_config.get("case") != case_name:
                        raise ValueError("source run belongs to a different case")
                    for key in ("dataset", "model", "observations"):
                        if key not in source_config:
                            raise ValueError(f"source run is missing inheritable section {key!r}")
                        config[key] = source_config[key]
                    source = config.setdefault("source", {})
                    source["inherited_base_keys"] = ["dataset", "model", "observations"]
                    source["config_origins"] = {
                        "dataset": "source_run.resolved_config.yaml",
                        "model": "source_run.resolved_config.yaml",
                        "observations": "source_run.resolved_config.yaml",
                        "post_training": "child_config",
                    }
                elif config.get("source", {}).get("kind") == "legacy_demo50":
                    source = config.setdefault("source", {})
                    source["inherited_base_keys"] = [
                        "model_architecture",
                        "normalization",
                        "field_order",
                    ]
                    source["config_origins"] = {
                        "model_architecture": "legacy_args_and_run_config",
                        "normalization": "legacy_checkpoint_and_dataset_stats",
                        "field_order": "explicit_compatibility_mapping",
                        "post_training": "child_config",
                    }
                else:
                    raise FileNotFoundError(
                        f"source run has no resolved config: {source_config_path}"
                    )
    if "dataset" not in config:
        raise ValueError("config must define a dataset section")
    if config.get("stage") == "direct_physics" or (
        config.get("stage") == "post_training" and "physics" in config
    ):
        config["dataset"]["include_temporal_derivative"] = True
    if validate_stage:
        validate_config(config)
    config["dataset"]["path"] = str(_resolve_dataset_path(config, case_dir))
    statistics_path = config["dataset"].get("normalization_stats_path")
    if statistics_path:
        statistics_path = Path(statistics_path)
        config["dataset"]["normalization_stats_path"] = str(
            statistics_path.resolve()
            if statistics_path.is_absolute()
            else (case_dir / statistics_path).resolve()
        )
    for family in config.get("coherence", {}).get("families", {}).values():
        reference = family.get("reference_bank", {})
        reference_path = reference.get("path")
        if not reference_path:
            continue
        reference_path = Path(reference_path)
        reference["path"] = str(
            reference_path.resolve()
            if reference_path.is_absolute()
            else (case_dir / reference_path).resolve()
        )
    stage1_checkpoint = config.get("model", {}).get("stage1_checkpoint")
    if stage1_checkpoint:
        path = Path(stage1_checkpoint)
        config["model"]["stage1_checkpoint"] = str(
            path.resolve() if path.is_absolute() else (case_dir / path).resolve()
        )
    return config


def run_case_cli(case_name: str, case_dir: str | Path) -> int:
    case_dir = Path(case_dir).resolve()
    parser = argparse.ArgumentParser(prog=f"{case_name}/run.py")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("validate", "build-manifest", "train-base", "post-train", "train-direct"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument("--override", action="append", default=[])
        if command == "build-manifest":
            subparser.add_argument("--output", type=Path, required=True)
            subparser.add_argument("--max-samples", type=int, default=8)
            subparser.add_argument("--query-points", type=int)
            subparser.add_argument(
                "--split", choices=("train", "validation", "test"), default="validation"
            )
        if command in {"train-base", "post-train", "train-direct"}:
            subparser.add_argument("--max-steps", type=int)
            subparser.add_argument("--resume", type=Path)

    evaluator = subparsers.add_parser("evaluate-run")
    evaluator.add_argument("--run", type=Path, required=True)
    evaluator.add_argument("--checkpoint", default="best")
    evaluator.add_argument("--sensor-config", type=Path)
    evaluator.add_argument("--sensor-manifest", type=Path)
    evaluator.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    evaluator.add_argument("--max-samples", type=int, default=1)
    evaluator.add_argument("--query-points", type=int)
    evaluator.add_argument("--generation-steps", type=int)
    evaluator.add_argument("--device")
    evaluator.add_argument("--report-name", default="benchmark")
    evaluator.add_argument(
        "--weight-selection",
        choices=("configured", "live"),
        default="configured",
    )

    args = parser.parse_args()
    if args.command == "evaluate-run":
        from .evaluation.checkpoint import evaluate_run

        run_dir = args.run.resolve() if args.run.is_absolute() else (case_dir / args.run).resolve()
        sensor_config = args.sensor_config
        if sensor_config is not None and not sensor_config.is_absolute():
            sensor_config = (case_dir / sensor_config).resolve()
        sensor_manifest = args.sensor_manifest
        if sensor_manifest is not None and not sensor_manifest.is_absolute():
            sensor_manifest = (case_dir / sensor_manifest).resolve()
        report = evaluate_run(
            run_dir,
            case_dir=case_dir,
            split=args.split,
            max_samples=args.max_samples,
            checkpoint=args.checkpoint,
            sensor_config=sensor_config,
            sensor_manifest=sensor_manifest,
            query_points=args.query_points,
            generation_steps=args.generation_steps,
            device_name=args.device,
            report_name=args.report_name,
            weight_selection=args.weight_selection,
        )
        print(report)
        return 0

    config_path = (
        args.config.resolve() if args.config.is_absolute() else (case_dir / args.config).resolve()
    )
    config = _load_case_config(
        config_path,
        case_dir,
        case_name,
        args.override,
        validate_stage=args.command != "validate",
    )

    if args.command == "validate":
        # Dataset-only catalog configs remain valid inputs. Complete training
        # configs additionally receive the same stage checks as an actual run.
        if "stage" in config:
            validate_config(config)
        report = validate_dataset(config["dataset"]["path"], config["dataset"].get("field_names"))
        if config["dataset"].get("normalization_stats_path"):
            dataset = open_field_dataset(config["dataset"], split="train")
            report["normalization"] = {
                "method": dataset.normalizer.method,
                "digest": dataset.normalizer.digest(),
                "statistics_path": config["dataset"]["normalization_stats_path"],
            }
            dataset.close()
        report["benchmark_eligible"] = bool(config["dataset"].get("benchmark_eligible", True))
        report["formal_benchmark_valid"] = bool(report["valid"] and report["benchmark_eligible"])
        if not report["benchmark_eligible"]:
            report["warnings"].append(
                "dataset is registered for integration only, not formal comparison"
            )
        print(json.dumps(report, indent=2, sort_keys=True))
        return int(not report["valid"])

    if args.command == "build-manifest":
        dataset = open_field_dataset(config["dataset"], split=args.split)
        sample_count = min(len(dataset), args.max_samples)
        if sample_count < 1:
            raise ValueError(f"split {args.split!r} contains no samples")
        samples = [dataset[index] for index in range(sample_count)]
        batch = build_observation_batch(
            samples,
            sensor_protocol_from_config(config),
            query_points=args.query_points,
        )
        manifest = manifest_from_batch(batch, config["dataset"]["path"], dataset.split_name)
        output = (
            args.output.resolve()
            if args.output.is_absolute()
            else (case_dir / args.output).resolve()
        )
        manifest.save(output)
        print(
            json.dumps(
                {"path": str(output), "sha256": manifest.digest(), "samples": sample_count},
                indent=2,
            )
        )
        dataset.close()
        return 0

    if args.command == "post-train":
        if "physics" in config and "coherence" not in config:
            from .training.physics_post_training import run_physics_post_training

            run_dir = run_physics_post_training(
                config,
                case_dir=case_dir,
                max_steps=args.max_steps,
                resume=args.resume,
            )
        else:
            from .training.post_training import run_post_training

            run_dir = run_post_training(
                config,
                case_dir=case_dir,
                max_steps=args.max_steps,
                resume=args.resume,
            )
        print(run_dir)
        return 0

    if args.command == "train-direct":
        from .training.direct_physics import run_direct_physics_training

        run_dir = run_direct_physics_training(
            config, case_dir=case_dir, max_steps=args.max_steps, resume=args.resume
        )
        print(run_dir)
        return 0

    from .training.base_training import run_base_training

    run_dir = run_base_training(
        config, case_dir=case_dir, max_steps=args.max_steps, resume=args.resume
    )
    print(run_dir)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Use a case-local run.py entrypoint.")
    parser.parse_args()
    parser.error("launch from Cases/<case>/run.py so case context is explicit")
    return 2
