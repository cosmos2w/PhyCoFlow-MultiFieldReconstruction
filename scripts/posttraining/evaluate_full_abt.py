#!/usr/bin/env python3
"""Evaluate a portable post-training run against its SOURCE on fixed validation groups.

The command runs the existing full A/B/T family path, including all six
cubical-persistence filtrations. It is validation-only, uses raw live model
weights with the effective post-training inference config, and writes one
immutable JSON result. It does not enter the bounded campaign wrapper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/data2")
GROUP_SIZE = 32
CAMPAIGN_ENVIRONMENT = (
    "PHYCOFLOW_CAMPAIGN_SOURCE_STAGE",
    "PHYCOFLOW_CAMPAIGN_ADMITTED",
    "PHYCOFLOW_CAMPAIGN_GPU",
)
CHECKPOINT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$")
EXPECTED_FAMILIES = ("global_distribution", "cross_spectrum", "topology")


class PortableEvaluationError(RuntimeError):
    """Raised when a portable run does not satisfy the fixed validation contract."""


def _reject_campaign_environment() -> None:
    present = [name for name in CAMPAIGN_ENVIRONMENT if name in os.environ]
    if present:
        raise PortableEvaluationError(
            "portable evaluation refuses campaign staging/admission environment: "
            + ", ".join(present)
        )


def _helpers():
    """Load shared numerical helpers only after refusing campaign source staging."""
    _reject_campaign_environment()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    from scripts.posttraining import fast_lagrangian_evaluate as shared

    expected_path = REPO_ROOT / "scripts" / "posttraining" / "fast_lagrangian_evaluate.py"
    if Path(shared.__file__).resolve() != expected_path.resolve():
        raise PortableEvaluationError("portable helper import resolved to a staged/nonlocal evaluator")
    return shared


def _is_campaign_path(path: Path) -> bool:
    text = str(path)
    normalized = text.replace("\\", "/")
    parts = path.parts
    campaign_layout = any(
        part.startswith("LPCAccel_")
        and index > 0
        and parts[index - 1] == "experiments"
        and index + 1 < len(parts)
        and parts[index + 1] == "runs"
        for index, part in enumerate(parts)
    )
    return campaign_layout or "_LPCAccel/" in normalized or "_LPCAccel" in parts


def _validate_run_location(run_path: Path, data_root: Path) -> Path:
    if _is_campaign_path(run_path):
        raise PortableEvaluationError("historical _LPCAccel runs require the study evaluator")
    try:
        run_dir = run_path.resolve(strict=True)
        root = data_root.resolve(strict=True)
    except OSError as error:
        raise PortableEvaluationError(f"run or data root is unavailable: {error}") from error
    if _is_campaign_path(run_dir):
        raise PortableEvaluationError("historical _LPCAccel runs require the study evaluator")
    if not run_dir.is_dir() or not run_dir.is_relative_to(root):
        raise PortableEvaluationError("run payload must be a directory below the mounted data root")
    if os.stat(run_dir).st_dev != os.stat(root).st_dev:
        raise PortableEvaluationError("run payload is not on the data root filesystem")
    return run_dir


def _output_directory(
    run_dir: Path,
    requested: Path | None,
    *,
    data_root: Path,
    require_mount: bool,
) -> Path:
    root = data_root.resolve(strict=True)
    if require_mount and not data_root.is_mount():
        raise PortableEvaluationError(f"data root is not a mounted filesystem: {data_root}")
    if os.stat(run_dir).st_dev != os.stat(root).st_dev:
        raise PortableEvaluationError("run payload and data root are on different filesystems")
    output = (requested if requested is not None else run_dir / "evaluation").resolve(
        strict=False
    )
    if not output.is_relative_to(root):
        raise PortableEvaluationError("output directory must be below the mounted data root")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve(strict=True)
    if not output.is_dir() or os.stat(output).st_dev != os.stat(root).st_dev:
        raise PortableEvaluationError("output directory is not on the mounted data root")
    if requested is None and not output.is_relative_to(run_dir):
        raise PortableEvaluationError("run evaluation directory escapes the run payload")
    return output


def _validate_supported_config(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if config.get("stage") != "post_training" or manifest.get("stage") != "post_training":
        raise PortableEvaluationError("full A/B/T evaluation requires a post_training run")
    if config.get("source", {}).get("kind", "native_run") != "native_run":
        raise PortableEvaluationError("portable full evaluation requires a native SOURCE run")
    evaluation = config.get("evaluation", {})
    if evaluation.get("split", "validation") != "validation":
        raise PortableEvaluationError("only the validation split is supported; TEST is never opened")
    if config.get("dataset", {}).get("include_temporal_derivative", False) is not False:
        raise PortableEvaluationError("include_temporal_derivative must be false")
    if config.get("model", {}).get("model_ema_eval") is not False:
        raise PortableEvaluationError("resolved post-training inference must use raw live weights")
    if config.get("rollout") != {"steps": 2, "solver": "euler"}:
        raise PortableEvaluationError("supported inference requires exactly two Euler rollout steps")
    generation_steps = evaluation.get("generation_steps")
    if (
        isinstance(generation_steps, bool)
        or not isinstance(generation_steps, int)
        or generation_steps != 2
    ):
        raise PortableEvaluationError("supported evaluation requires exactly two generation steps")
    multitask = config.get("optimization", {}).get("multitask", {})
    external = multitask.get("external_policy", {})
    if external.get("method") != "lagrangian":
        raise PortableEvaluationError("supported full evaluation requires the external Lagrangian path")
    if config.get("output", {}).get("experiment_name", "").startswith("_LPCAccel/"):
        raise PortableEvaluationError("historical _LPCAccel runs require the study evaluator")
    family_names = tuple(manifest.get("coherence_families", ()))
    configured_families = tuple(config.get("coherence", {}).get("families", ()))
    if family_names != EXPECTED_FAMILIES or configured_families != EXPECTED_FAMILIES:
        raise PortableEvaluationError(
            "resolved run must contain the exact ordered global-distribution, cross-spectrum, topology families"
        )
    compute_budget = config.get("coherence", {}).get("compute_budget", {})
    if compute_budget.get("query_policy") != "fixed_shared":
        raise PortableEvaluationError("full A/B/T validation requires the fixed_shared query policy")
    point_count = compute_budget.get("point_count")
    if not isinstance(point_count, int) or isinstance(point_count, bool) or point_count < 1:
        raise PortableEvaluationError("coherence.compute_budget.point_count must be a positive integer")


def _hash_tensor(tensor: torch.Tensor) -> str:
    payload = tensor.detach().to(device="cpu").contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _validate_child_checkpoint(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    source_hashes: Mapping[str, str],
    train_dataset,
    families: Mapping[str, Any],
    family_scales: Mapping[str, float],
    run_dir: Path,
    shared,
) -> None:
    from phycoflow_reconstruction.training.run_store import file_sha256
    from phycoflow_reconstruction.training.source import source_checkpoint_path

    if checkpoint.get("config_sha256") != manifest.get("config_sha256"):
        raise PortableEvaluationError("selected checkpoint is not bound to this resolved config")
    if checkpoint.get("source_hashes") != dict(source_hashes):
        raise PortableEvaluationError("selected checkpoint SOURCE hashes differ from current SOURCE")
    if Path(str(checkpoint.get("source_run", ""))).resolve() != Path(
        str(config["source_run"])
    ).resolve():
        raise PortableEvaluationError("selected checkpoint points to a different SOURCE run")
    expected_source_checkpoint = source_checkpoint_path(config).resolve()
    if Path(str(checkpoint.get("source_checkpoint", ""))).resolve() != expected_source_checkpoint:
        raise PortableEvaluationError("selected checkpoint points to a different SOURCE checkpoint")
    if tuple(checkpoint.get("data_spec", {}).get("field_names", ())) != tuple(
        train_dataset.field_names
    ):
        raise PortableEvaluationError("selected checkpoint field order differs from SOURCE data")
    normalization = checkpoint.get("normalization", {})
    if (
        not torch.equal(normalization.get("offset"), train_dataset.normalizer.offset)
        or not torch.equal(normalization.get("scale"), train_dataset.normalizer.scale)
        or normalization.get("method") != train_dataset.normalizer.method
    ):
        raise PortableEvaluationError("selected checkpoint normalization differs from SOURCE")
    calibration_path = run_dir / "artifacts" / "coherence_calibration.json"
    calibration_sha256 = file_sha256(calibration_path)
    if (
        manifest.get("coherence_calibration_sha256") != calibration_sha256
        or checkpoint.get("coherence_calibration_sha256") != calibration_sha256
    ):
        raise PortableEvaluationError("checkpoint calibration identity differs from run artifact")
    if checkpoint.get("family_scales") != dict(family_scales):
        raise PortableEvaluationError("checkpoint family scales differ from fixed calibration")
    states = checkpoint.get("family_states")
    if not isinstance(states, Mapping) or tuple(states) != tuple(families):
        raise PortableEvaluationError("checkpoint family states differ from full A/B/T composition")

    optimization = config.get("optimization", {})
    fast_settings = optimization.get("fast_lagrangian")
    if fast_settings is not None:
        from phycoflow_reconstruction.training.fast_lagrangian import implementation_binding

        if checkpoint.get("fast_lagrangian_implementation") != implementation_binding():
            raise PortableEvaluationError("fast-Lagrangian implementation differs from checkpoint")
    elif "fast_lagrangian_implementation" in checkpoint:
        raise PortableEvaluationError("fast-Lagrangian checkpoint does not match resolved config")
    ordered_settings = optimization.get("lpc_ordered")
    if ordered_settings is not None:
        from phycoflow_reconstruction.training.lpc_ordered import implementation_binding

        if checkpoint.get("lpc_ordered_implementation") != implementation_binding():
            raise PortableEvaluationError("LPC-ordered implementation differs from checkpoint")
        reference_path = run_dir / "artifacts" / "lpc_ordered_reference.json"
        if file_sha256(reference_path) != checkpoint.get("lpc_ordered_reference_sha256"):
            raise PortableEvaluationError("LPC-ordered reference artifact differs from checkpoint")
        reference_artifact = shared._read_json(reference_path)
        if (
            reference_artifact.get("settings") != dict(ordered_settings)
            or reference_artifact.get("source_checkpoint_sha256") != source_hashes.get("checkpoint")
            or "lpc_ordered_state" not in checkpoint
        ):
            raise PortableEvaluationError("LPC-ordered reference/source binding is invalid")
    elif any(
        key in checkpoint
        for key in (
            "lpc_ordered_state",
            "lpc_ordered_implementation",
            "lpc_ordered_reference_sha256",
        )
    ):
        raise PortableEvaluationError("ordered checkpoint cannot be evaluated under a different config")
    if file_sha256(checkpoint_path) != checkpoint_sha256:
        raise PortableEvaluationError("selected child checkpoint changed during validation")


def _artifact_snapshot(run_dir: Path, family_names: Sequence[str]) -> dict[Path, str]:
    from phycoflow_reconstruction.training.run_store import file_sha256

    names = [f"{name}_family.pt" for name in family_names]
    names.extend(("coherence_calibration.json", "coherence_reference.pt"))
    names.extend(f"coherence_reference_{name}.pt" for name in family_names)
    paths = [run_dir / "artifacts" / name for name in names]
    return {path: file_sha256(path) for path in paths if path.is_file()}


def _selection_digest(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _write_immutable_json(shared, path: Path, payload: Mapping[str, Any]) -> None:
    """Reuse the established canonical JSON/no-replacement behavior."""
    shared._atomic_json(path, payload)


def evaluate_run(
    run_path: str | Path,
    *,
    checkpoint_name: str = "best",
    max_samples: int = 64,
    device_name: str = "cpu",
    output_dir: str | Path | None = None,
    _data_root: Path | None = None,
    _require_mount: bool = True,
) -> Path:
    """Evaluate a native post-training run on paired, uniform validation groups."""
    shared = _helpers()
    shared._validate_max_samples(max_samples)
    if not isinstance(checkpoint_name, str) or not CHECKPOINT_PATTERN.fullmatch(checkpoint_name):
        raise PortableEvaluationError("checkpoint must be a safe checkpoint basename")
    if device_name not in {"cpu", "cuda:0"}:
        raise PortableEvaluationError("device must be cpu or an explicit cuda:0")

    data_root = DATA_ROOT if _data_root is None else Path(_data_root)
    if _require_mount and (not data_root.is_mount() or data_root != Path("/data2")):
        raise PortableEvaluationError("portable output requires the mounted /data2 filesystem")
    run_dir = _validate_run_location(Path(run_path), data_root)
    config, manifest, checkpoint_path, checkpoint_sha256 = shared._load_run_inputs(
        run_dir, checkpoint_name
    )
    if manifest.get("experiment_name") != config.get("output", {}).get("experiment_name"):
        raise PortableEvaluationError("run config and manifest experiment identities differ")
    if _is_campaign_path(run_dir) or "_LPCAccel/" in str(manifest.get("experiment_name", "")):
        raise PortableEvaluationError("historical _LPCAccel runs require the study evaluator")
    _validate_supported_config(config, manifest)
    if device_name == "cuda:0" and not torch.cuda.is_available():
        raise PortableEvaluationError("cuda:0 was explicitly requested but is unavailable")
    device = torch.device(device_name)

    from phycoflow_reconstruction.data.factory import open_field_dataset
    from phycoflow_reconstruction.data.manifest import dataset_fingerprint
    from phycoflow_reconstruction.data.sensor_protocols import build_observation_batch
    from phycoflow_reconstruction.training.common import sensor_protocol_from_config
    from phycoflow_reconstruction.training.model_lifecycle import load_training_aux_state
    from phycoflow_reconstruction.training.post_training import _build_comparison_batch
    from phycoflow_reconstruction.training.run_store import (
        file_sha256,
        load_model_state_strict,
        load_project_checkpoint,
    )
    from phycoflow_reconstruction.training.source import load_source_model, source_hashes
    from phycoflow_reconstruction.training.update_budget import post_training_steps_per_epoch

    dataset_path = Path(config["dataset"]["path"]).resolve(strict=True)
    resolved_config_path = run_dir / "resolved_config.yaml"
    run_manifest_path = run_dir / "run_manifest.json"
    initial_config_sha256 = file_sha256(resolved_config_path)
    initial_manifest_sha256 = file_sha256(run_manifest_path)
    if shared._read_yaml(resolved_config_path) != config or shared._read_json(
        run_manifest_path
    ) != manifest:
        raise PortableEvaluationError("resolved run inputs changed while being opened")
    manifest_dataset_path = manifest.get("dataset_path")
    if manifest_dataset_path is None or Path(manifest_dataset_path).resolve() != dataset_path:
        raise PortableEvaluationError("resolved dataset path differs from run manifest")
    initial_dataset_fingerprint = dataset_fingerprint(dataset_path)
    if initial_dataset_fingerprint != manifest.get("dataset_fingerprint"):
        raise PortableEvaluationError("current dataset fingerprint differs from run manifest")
    actual_source_hashes = source_hashes(config)
    if manifest.get("source_hashes") != actual_source_hashes:
        raise PortableEvaluationError("current SOURCE hashes differ from run manifest")

    output_path_directory = _output_directory(
        run_dir,
        None if output_dir is None else Path(output_dir),
        data_root=data_root,
        require_mount=_require_mount,
    )

    from phycoflow_reconstruction.training.post_training import _configure_persistence_workers

    _configure_persistence_workers(config)
    model, train_dataset, source_metadata = load_source_model(config, device)
    validation_dataset = None
    try:
        if getattr(model, "_ema_eval", False) is not False:
            raise PortableEvaluationError("loaded SOURCE model did not honor raw post-training weights")
        if tuple(train_dataset.field_names) != tuple(config["dataset"].get("field_names", ())):
            raise PortableEvaluationError("dataset field order differs from resolved config")
        if manifest.get("normalization_digest") is not None and train_dataset.normalizer.digest() != manifest.get(
            "normalization_digest"
        ):
            raise PortableEvaluationError("current SOURCE normalizer digest differs from manifest")
        validation_dataset_config = deepcopy(dict(config["dataset"]))
        validation_dataset_config["include_temporal_derivative"] = False
        validation_dataset = open_field_dataset(
            validation_dataset_config, split="validation", normalizer=train_dataset.normalizer
        )
        if tuple(validation_dataset.field_names) != tuple(train_dataset.field_names):
            raise PortableEvaluationError("validation field order differs from SOURCE field order")
        validation_indices = shared._uniform_validation_indices(len(validation_dataset), max_samples)
        samples = [validation_dataset[index] for index in validation_indices]
        protocol = sensor_protocol_from_config(config)
        evaluation_config = deepcopy(dict(config))
        evaluation_config.setdefault("evaluation", {})["seed"] = shared.VALIDATION_SEED
        evaluation_config["dataset"]["include_temporal_derivative"] = False
        complete_batch = build_observation_batch(samples, protocol, query_points=None).to(device)
        expected_query_count = 1
        for dimension in validation_dataset.data_spec.logical_shape:
            expected_query_count *= int(dimension)
        compute_point_count = int(config["coherence"]["compute_budget"]["point_count"])
        expected_comparison_count = min(compute_point_count, expected_query_count)
        configured_evaluation_count = int(
            evaluation_config["evaluation"].get("query_points", compute_point_count)
        )
        if configured_evaluation_count != compute_point_count:
            raise PortableEvaluationError(
                "evaluation.query_points must match the configured coherence point count"
            )
        comparison_batch = _build_comparison_batch(complete_batch, evaluation_config)
        if complete_batch.query_coords.shape[1] != expected_query_count:
            raise PortableEvaluationError("validation batch does not contain the complete native query grid")
        if not bool(complete_batch.query_valid_mask.all()):
            raise PortableEvaluationError("native validation query grid contains invalid positions")
        if comparison_batch.target_fields is None:
            raise PortableEvaluationError("validation comparison batch did not retain dense targets")
        if not bool(comparison_batch.query_valid_mask.all()):
            raise PortableEvaluationError("configured validation comparison points contain invalid positions")
        if comparison_batch.query_coords.shape[1] != expected_comparison_count:
            raise PortableEvaluationError(
                "validation comparison batch does not match the configured fixed query count"
            )
        if len(complete_batch.sample_ids) != max_samples:
            raise PortableEvaluationError("validation batch size differs from requested panel")

        family_artifact_hashes = _artifact_snapshot(run_dir, EXPECTED_FAMILIES)
        families, banks, family_scales = shared._load_families_and_banks(
            config, manifest, actual_source_hashes, run_dir, train_dataset, device
        )
        if tuple(families) != EXPECTED_FAMILIES:
            raise PortableEvaluationError("loaded family composition differs from full A/B/T")
        field_names = tuple(train_dataset.field_names)
        slices = shared._group_slices(max_samples)
        query_indices = comparison_batch.metadata.get("query_indices")
        observation_indices = comparison_batch.obs_indices
        if not isinstance(query_indices, torch.Tensor) or observation_indices is None:
            raise PortableEvaluationError("validation batches must retain observation/query indices")
        query_sha256 = _hash_tensor(query_indices)
        observation_digest = hashlib.sha256()
        for value in (
            observation_indices,
            comparison_batch.obs_field_ids,
            comparison_batch.obs_valid_mask,
        ):
            observation_digest.update(
                value.detach().to(device="cpu").contiguous().numpy().tobytes()
            )
        selection_payload = {
            "split": "validation",
            "dataset_path": str(dataset_path),
            "dataset_fingerprint": initial_dataset_fingerprint,
            "validation_indices": validation_indices,
            "sample_ids": list(comparison_batch.sample_ids),
            "group_size": GROUP_SIZE,
            "generation_seed": shared.VALIDATION_SEED,
            "generation_seed_stride": 100_003,
            "query_policy": "fixed_shared",
            "comparison_query_count": expected_comparison_count,
            "protocol": comparison_batch.metadata.get("protocol", protocol.to_dict()),
            "observation_indices_sha256": observation_digest.hexdigest(),
            "query_indices_sha256": query_sha256,
        }
        sensor_query_manifest_sha256 = _selection_digest(selection_payload)

        # Evaluate the SOURCE with the same post-training config before loading child weights.
        source_groups = shared._evaluate_groups(
            model,
            complete_batch,
            comparison_batch,
            families,
            banks,
            field_names,
            evaluation_config,
            normalizer=train_dataset.normalizer,
            family_scales=family_scales,
        )
        source_summary = shared._summarize_groups(source_groups)
        if len(source_groups) != len(slices):
            raise PortableEvaluationError("SOURCE evaluator returned an incomplete 32-sample grouping")

        checkpoint = load_project_checkpoint(checkpoint_path)
        _validate_child_checkpoint(
            config,
            manifest,
            checkpoint,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            source_hashes=actual_source_hashes,
            train_dataset=train_dataset,
            families=families,
            family_scales=family_scales,
            run_dir=run_dir,
            shared=shared,
        )
        load_model_state_strict(model, checkpoint["model"])
        load_training_aux_state(model, checkpoint)
        for name, family in families.items():
            family.load_state_dict(checkpoint["family_states"][name], strict=True)
        child_groups = shared._evaluate_groups(
            model,
            complete_batch,
            comparison_batch,
            families,
            banks,
            field_names,
            evaluation_config,
            normalizer=train_dataset.normalizer,
            family_scales=family_scales,
        )
        child_summary = shared._summarize_groups(child_groups)
        if len(child_groups) != len(slices) or [
            group["sample_ids"] for group in child_groups
        ] != [group["sample_ids"] for group in source_groups]:
            raise PortableEvaluationError("SOURCE and child were not evaluated on identical groups")
        if [group["generation_seed"] for group in child_groups] != [
            group["generation_seed"] for group in source_groups
        ]:
            raise PortableEvaluationError("SOURCE and child evaluation seeds differ")

        # Recheck the complete lineage envelope after the paired evaluation.
        if source_hashes(config) != actual_source_hashes:
            raise PortableEvaluationError("SOURCE artifacts changed during validation")
        if file_sha256(checkpoint_path) != checkpoint_sha256:
            raise PortableEvaluationError("child checkpoint changed during validation")
        if (
            file_sha256(resolved_config_path) != initial_config_sha256
            or dataset_fingerprint(dataset_path) != initial_dataset_fingerprint
        ):
            raise PortableEvaluationError("resolved run inputs changed during validation")
        if _artifact_snapshot(run_dir, tuple(families)) != family_artifact_hashes:
            raise PortableEvaluationError("family/calibration/reference artifacts changed during validation")
        if (
            file_sha256(run_manifest_path) != initial_manifest_sha256
            or shared._read_json(run_manifest_path) != manifest
        ):
            raise PortableEvaluationError("run manifest changed during validation")

        source_metrics = shared._ratioable_metrics(source_summary)
        child_metrics = shared._ratioable_metrics(child_summary)
        group_rows = []
        for (start, stop), source_group, child_group in zip(slices, source_groups, child_groups):
            group_rows.append(
                {
                    "group": source_group["group"],
                    "generation_seed": source_group["generation_seed"],
                    "validation_indices": validation_indices[start:stop],
                    "sample_ids": source_group["sample_ids"],
                    "source": source_group["metrics"],
                    "child": child_group["metrics"],
                }
            )
        if any(
            row["source"]["coherence"]["families"]["topology"][
                "topology_scalar_filtrations"
            ]
            != 6
            or row["child"]["coherence"]["families"]["topology"][
                "topology_scalar_filtrations"
            ]
            != 6
            for row in group_rows
        ):
            raise PortableEvaluationError("every paired group must use all six topology filtrations")

        evaluated_at = datetime.now(timezone.utc)
        steps_per_epoch = post_training_steps_per_epoch(config, len(train_dataset))
        payload = {
            "schema": "phycoflow_full_abt_validation.v1",
            "schema_version": 1,
            "evaluated_at_utc": evaluated_at.isoformat(),
            "device": device_name,
            "run": str(run_dir),
            "checkpoint": checkpoint_name,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
            "checkpoint_training_epoch": float(checkpoint.get("global_step", 0)) / max(
                steps_per_epoch, 1
            ),
            "source": {
                "run": str(Path(config["source_run"]).resolve()),
                "hashes": actual_source_hashes,
                "metadata": source_metadata,
            },
            "config_sha256": manifest["config_sha256"],
            "dataset": {
                "path": str(dataset_path),
                "split": "validation",
                "fingerprint": initial_dataset_fingerprint,
                "field_names": list(field_names),
                "normalizer_digest": train_dataset.normalizer.digest(),
                "logical_shape": list(validation_dataset.data_spec.logical_shape),
                "include_temporal_derivative": False,
            },
            "inference_contract": {
                "model_ema_eval": False,
                "weight_identity": "raw_live_model_weights",
                "rollout_steps": 2,
                "rollout_solver": "euler",
                "include_temporal_derivative": False,
            },
            "selection": {
                **selection_payload,
                "sensor_query_manifest_sha256": sensor_query_manifest_sha256,
                "query_indices_shape": list(query_indices.shape),
                "groups": [validation_indices[start:stop] for start, stop in slices],
                "group_size": GROUP_SIZE,
                "group_count": len(slices),
                "generation_seed": shared.VALIDATION_SEED,
                "generation_seed_stride": 100_003,
            },
            "evaluation": {
                "mode": "full_existing_A_B_T_family_forward",
                "samples": max_samples,
                "aggregation": (
                    "sample-weighted mean of full 32-snapshot group scores; "
                    "B is the mean of B(32) ensemble statistics; worst field is "
                    "the maximum pooled per-field relative L2; entrywise observed/unobserved "
                    "MSE remains per group"
                ),
                "topology_is_stochastic": False,
                "topology_scalar_filtrations": 6,
                "include_temporal_derivative": False,
                "source_group_means": source_summary,
                "child_group_means": child_summary,
                "full_metric_ratios_child_over_source": shared._ratio_tree(
                    child_metrics, source_metrics
                ),
                "endpoint_mse_model_units": {
                    "field_names": list(field_names),
                    "source": [source_summary["endpoint_mse_model_units"][name] for name in field_names],
                    "child": [child_summary["endpoint_mse_model_units"][name] for name in field_names],
                    "child_over_source": [
                        shared._ratio_tree(
                            child_summary["endpoint_mse_model_units"][name],
                            source_summary["endpoint_mse_model_units"][name],
                        )
                        for name in field_names
                    ],
                },
            },
            "components": {
                "source": source_summary["coherence"]["families"],
                "child": child_summary["coherence"]["families"],
            },
            "groups": group_rows,
        }
        output_path = output_path_directory / (
            f"full_abt_{checkpoint_name}_n{max_samples}_"
            f"{checkpoint_sha256[:12]}_{evaluated_at.strftime('%Y%m%dT%H%M%S%fZ')}_validation.json"
        )
        _write_immutable_json(shared, output_path, payload)
        return output_path
    finally:
        close = getattr(validation_dataset, "close", None)
        if callable(close):
            close()
        close = getattr(train_dataset, "close", None)
        if callable(close):
            close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="native post_training run directory")
    parser.add_argument("--checkpoint", default="best", help="checkpoint basename without .pt")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=64,
        choices=(32, 64, 96, 128),
        help="uniform validation panel, evaluated in exact groups of 32",
    )
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="optional destination below mounted /data2; default is RUN/evaluation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _reject_campaign_environment()
        output = evaluate_run(
            args.run,
            checkpoint_name=args.checkpoint,
            max_samples=args.max_samples,
            device_name=args.device,
            output_dir=args.output_dir,
        )
    except (
        PortableEvaluationError,
        ImportError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
    ) as error:
        print(f"evaluate_full_abt: {error}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
