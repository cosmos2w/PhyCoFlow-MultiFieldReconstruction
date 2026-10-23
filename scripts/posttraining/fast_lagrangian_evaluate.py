#!/usr/bin/env python3
"""Evaluate one LPC child against its SOURCE on a fixed validation manifest.

This command deliberately evaluates the existing full A/B/T family path. It
does not run training estimators, touch TEST, or export visualization assets.
Its only output is one compact campaign-owned JSON file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "src"
if str(SOURCE_PATH) not in sys.path:
    sys.path.insert(0, str(SOURCE_PATH))
GROUP_SIZE = 32
MAX_SAMPLES = 128
VALIDATION_SEED = 2027
CHECKPOINT_NAMES = ("best", "last", "epoch_050", "epoch_075", "epoch_100")


class EvaluationError(RuntimeError):
    """Raised when a campaign input does not satisfy the evaluation contract."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, help="one child run below campaign/runs")
    parser.add_argument("--checkpoint", choices=CHECKPOINT_NAMES, default="best")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=64,
        help="uniform validation sample count; a multiple of 32 from 32 through 128",
    )
    return parser


def _validate_max_samples(count: int) -> None:
    if isinstance(count, bool) or not isinstance(count, int):
        raise EvaluationError("max-samples must be an integer")
    if count < GROUP_SIZE or count > MAX_SAMPLES or count % GROUP_SIZE:
        raise EvaluationError("max-samples must be 32, 64, 96, or 128")


def _uniform_validation_indices(dataset_size: int, count: int) -> list[int]:
    _validate_max_samples(count)
    if isinstance(dataset_size, bool) or not isinstance(dataset_size, int) or dataset_size < count:
        raise EvaluationError(f"validation split has {dataset_size} samples; {count} are required")
    indices = np.linspace(0, dataset_size - 1, count).round().astype(np.int64).tolist()
    if len(indices) != len(set(indices)):
        raise EvaluationError("uniform validation selection produced duplicate indices")
    return [int(value) for value in indices]


def _group_slices(sample_count: int, group_size: int = GROUP_SIZE) -> tuple[tuple[int, int], ...]:
    if group_size < 1 or sample_count < 1 or sample_count % group_size:
        raise ValueError("sample_count must be a positive multiple of group_size")
    return tuple((start, start + group_size) for start in range(0, sample_count, group_size))


def _slice_context(value: Any, start: int, stop: int, batch_size: int) -> Any:
    if isinstance(value, torch.Tensor):
        return value[start:stop] if value.ndim and value.shape[0] == batch_size else value
    if isinstance(value, Mapping):
        return {key: _slice_context(item, start, stop, batch_size) for key, item in value.items()}
    if isinstance(value, tuple) and len(value) == batch_size:
        return value[start:stop]
    if isinstance(value, list) and len(value) == batch_size:
        return value[start:stop]
    return value


def _slice_batch(batch, start: int, stop: int):
    """Slice one already-built observation/query manifest without resampling it."""
    from phycoflow_reconstruction.contracts import ObservationBatch

    batch_size = len(batch.sample_ids)
    if not (0 <= start < stop <= batch_size):
        raise ValueError("invalid observation batch slice")
    metadata = dict(batch.metadata)
    query_indices = metadata.get("query_indices")
    if isinstance(query_indices, torch.Tensor) and query_indices.ndim:
        metadata["query_indices"] = query_indices[start:stop]
    context = metadata.get("sample_context")
    if context is not None:
        metadata["sample_context"] = _slice_context(context, start, stop, batch_size)
    result = ObservationBatch(
        obs_coords=batch.obs_coords[start:stop],
        obs_values=batch.obs_values[start:stop],
        obs_field_ids=batch.obs_field_ids[start:stop],
        obs_valid_mask=batch.obs_valid_mask[start:stop],
        query_coords=batch.query_coords[start:stop],
        query_valid_mask=batch.query_valid_mask[start:stop],
        target_fields=None if batch.target_fields is None else batch.target_fields[start:stop],
        sample_ids=tuple(batch.sample_ids[start:stop]),
        obs_indices=None if batch.obs_indices is None else batch.obs_indices[start:stop],
        logical_shapes=tuple(batch.logical_shapes[start:stop]) if batch.logical_shapes else (),
        metadata=metadata,
    )
    result.validate()
    return result


def _tensor_sha256(value: torch.Tensor) -> str:
    encoded = value.detach().to(device="cpu").contiguous().numpy().tobytes()
    return hashlib.sha256(encoded).hexdigest()


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _weighted_tree(values: Sequence[Any], weights: Sequence[int]) -> Any:
    if not values or len(values) != len(weights):
        raise ValueError("metric values and group weights must align")
    first = values[0]
    if isinstance(first, Mapping):
        keys = tuple(first)
        if any(not isinstance(item, Mapping) or tuple(item) != keys for item in values):
            raise ValueError("group metric mappings differ in keys or order")
        return {key: _weighted_tree([item[key] for item in values], weights) for key in keys}
    if first is None:
        if any(value is not None for value in values):
            raise ValueError("group metric availability changed between groups")
        return None
    if _number(first):
        if any(not _number(value) for value in values):
            raise ValueError("group metric types changed between groups")
        total_weight = sum(weights)
        return sum(float(value) * weight for value, weight in zip(values, weights)) / total_weight
    if any(value != first for value in values):
        raise ValueError("non-numeric group metadata changed between groups")
    return first


def _ratio_tree(numerator: Any, denominator: Any) -> Any:
    if isinstance(numerator, Mapping):
        if not isinstance(denominator, Mapping) or tuple(numerator) != tuple(denominator):
            raise ValueError("metric mappings differ while computing ratios")
        return {key: _ratio_tree(numerator[key], denominator[key]) for key in numerator}
    if numerator is None or denominator is None:
        return None
    if not _number(numerator) or not _number(denominator):
        raise ValueError("full metric ratios require numeric metrics")
    baseline = float(denominator)
    return None if baseline == 0.0 else float(numerator) / baseline


def _compact_evaluation(result: Mapping[str, Any], field_names: Sequence[str]) -> dict[str, Any]:
    reconstruction_names = (
        "mse_normalized",
        "per_field_mse_normalized",
        "mean_relative_l2",
        "per_field_relative_l2",
        "worst_field_relative_l2",
        "observed_entry_mse_normalized",
        "unobserved_entry_mse_normalized",
        "mse_physical",
        "mean_relative_l2_physical",
        "per_field_mse_physical",
        "per_field_relative_l2_physical",
    )
    reconstruction = {name: result[name] for name in reconstruction_names if name in result}
    endpoint = result.get("endpoint_mse_model_units")
    if not isinstance(endpoint, Mapping) or tuple(endpoint) != tuple(field_names):
        raise EvaluationError(
            "full external-policy evaluation did not return the model-unit endpoint vector"
        )
    coherence = result.get("coherence")
    if not isinstance(coherence, Mapping):
        raise EvaluationError("full family evaluation did not return coherence metrics")
    families = coherence.get("families")
    expected_families = {"global_distribution", "cross_spectrum", "topology"}
    if not isinstance(families, Mapping) or not expected_families.issubset(families):
        raise EvaluationError("full validation must include A/B/T family evaluations")
    compact_families = {}
    for name in ("global_distribution", "cross_spectrum", "topology"):
        family = families[name]
        diagnostics = family.get("diagnostics", {})
        if not isinstance(diagnostics, Mapping):
            diagnostics = {}
        if diagnostics.get("estimator_value") == "T_estimate":
            raise EvaluationError("sampled topology output is forbidden in full validation")
        if name == "topology" and diagnostics.get("scalar_filtrations") != 6:
            raise EvaluationError("full validation requires all six complete topology filtrations")
        compact_families[name] = {
            "total": float(family["total"]),
            "weighted_total": float(family["weighted_total"]),
            "components": {
                key: float(value) for key, value in family.get("component_scalars", {}).items()
            },
            "topology_scalar_filtrations": (
                int(diagnostics["scalar_filtrations"])
                if name == "topology" and "scalar_filtrations" in diagnostics
                else None
            ),
        }
    return {
        "reconstruction": reconstruction,
        "endpoint_mse_model_units": {name: float(endpoint[name]) for name in field_names},
        "coherence": {
            "total": float(coherence["total"]),
            "components": {key: float(value) for key, value in coherence["components"].items()},
            "families": compact_families,
        },
    }


def _evaluate_groups(
    model,
    complete_batch,
    comparison_batch,
    families,
    banks,
    field_names: tuple[str, ...],
    config: Mapping[str, Any],
    *,
    normalizer,
    family_scales: Mapping[str, float],
    group_size: int = GROUP_SIZE,
) -> list[dict[str, Any]]:
    """Use the existing full evaluator on fixed groups of the shared batches."""
    from phycoflow_reconstruction.training.post_training import _evaluate

    groups = []
    base_seed = VALIDATION_SEED
    for group_index, (start, stop) in enumerate(
        _group_slices(len(complete_batch.sample_ids), group_size)
    ):
        full_group = _slice_batch(complete_batch, start, stop)
        compare_group = _slice_batch(comparison_batch, start, stop)
        group_config = deepcopy(dict(config))
        group_config.setdefault("evaluation", {})["seed"] = base_seed + group_index * 100_003
        result = _evaluate(
            model,
            full_group,
            compare_group,
            families,
            banks,
            field_names,
            group_config,
            normalizer=normalizer,
            family_scales=family_scales,
        )
        groups.append(
            {
                "group": group_index,
                "batch_slice": [start, stop],
                "generation_seed": base_seed + group_index * 100_003,
                "sample_ids": list(full_group.sample_ids),
                "metrics": _compact_evaluation(result, field_names),
            }
        )
    return groups


def _summarize_groups(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not groups:
        raise ValueError("no validation groups were evaluated")
    metrics = [group["metrics"] for group in groups]
    weights = [len(group["sample_ids"]) for group in groups]
    summary = _weighted_tree(metrics, weights)
    reconstruction = summary.get("reconstruction", {})
    if "per_field_relative_l2" in reconstruction:
        reconstruction["worst_field_relative_l2"] = max(
            reconstruction["per_field_relative_l2"].values()
        )
    # Sensor counts vary per snapshot; sample weights do not pool these entry
    # denominators. Preserve the original scores only in individual groups.
    reconstruction.pop("observed_entry_mse_normalized", None)
    reconstruction.pop("unobserved_entry_mse_normalized", None)
    return summary


def _ratioable_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Select all reported numeric validation metrics for SOURCE ratios."""
    return {
        "reconstruction": dict(summary["reconstruction"]),
        "endpoint_mse_model_units": dict(summary["endpoint_mse_model_units"]),
        "coherence": {
            "total": summary["coherence"]["total"],
            "components": dict(summary["coherence"]["components"]),
            "families": {
                name: {
                    "total": family["total"],
                    "weighted_total": family["weighted_total"],
                    "components": dict(family["components"]),
                }
                for name, family in summary["coherence"]["families"].items()
            },
        },
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            return
        raise EvaluationError(f"refusing to overwrite a different validation result: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationError(f"expected a JSON object in {path}")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationError(f"expected a YAML mapping in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_campaign_run(campaign_root: Path, run_dir: Path) -> tuple[Path, Path, str]:
    expected_parent = Path("/data2/wanglz/PhyCoFlow-MultiFieldReconstruction/experiments")
    root = campaign_root.resolve(strict=True)
    if not root.is_relative_to(expected_parent) or not root.name.startswith("LPCAccel_"):
        raise EvaluationError(
            f"campaign root must be inside the owned /data2 experiments area: {root}"
        )
    if not Path("/data2").is_mount() or os.stat(root).st_dev != os.stat("/data2").st_dev:
        raise EvaluationError("campaign root is not on the mounted /data2 filesystem")
    if root.stat().st_uid != os.getuid():
        raise EvaluationError("campaign root is not owned by the current user")
    analysis_dir = (root / "analysis").resolve(strict=True)
    runs_dir = (root / "runs").resolve(strict=True)
    if not analysis_dir.is_relative_to(root) or not runs_dir.is_relative_to(root):
        raise EvaluationError("campaign analysis/runs directory escapes the campaign root")
    run = run_dir.resolve(strict=True)
    if not run.is_relative_to(runs_dir) or run.stat().st_uid != os.getuid():
        raise EvaluationError("run must be owned by the current user and below campaign/runs")
    relative = run.relative_to(runs_dir)
    if len(relative.parts) != 2:
        raise EvaluationError("run must use the campaign-owned runs/<arm>/<run> layout")
    arm = relative.parts[0]
    if not arm or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in arm
    ):
        raise EvaluationError("run arm name is not safe for a campaign JSON filename")
    return root, run, arm


def _load_run_inputs(run_dir: Path, checkpoint_name: str):
    from phycoflow_reconstruction.training.run_store import (
        config_digest,
        file_sha256,
    )

    config_path = run_dir / "resolved_config.yaml"
    manifest_path = run_dir / "run_manifest.json"
    checkpoint_path = run_dir / "checkpoints" / f"{checkpoint_name}.pt"
    if not config_path.is_file() or not manifest_path.is_file() or not checkpoint_path.is_file():
        raise EvaluationError(
            "run is missing its resolved config, manifest, or requested checkpoint"
        )
    config = _read_yaml(config_path)
    manifest = _read_json(manifest_path)
    if config.get("stage") != "post_training":
        raise EvaluationError("requested run is not a post_training child")
    run_id = manifest.get("run_id")
    if run_id != run_dir.name:
        raise EvaluationError("run manifest identity differs from its directory")
    expected_config_hash = config_digest(config)
    if manifest.get("config_sha256") != expected_config_hash:
        raise EvaluationError("resolved config hash does not match the run manifest")
    expected_checkpoint_hash = manifest.get("checkpoint_hashes", {}).get(checkpoint_name)
    checkpoint_sha256 = file_sha256(checkpoint_path)
    if expected_checkpoint_hash != checkpoint_sha256:
        raise EvaluationError("requested checkpoint hash does not match the run manifest")
    return config, manifest, checkpoint_path, checkpoint_sha256


def _validate_experiment_binding(config, manifest, root: Path, arm: str) -> None:
    expected = f"_LPCAccel/{root.name}/{arm}"
    if (
        manifest.get("experiment_name") != expected
        or config.get("output", {}).get("experiment_name") != expected
    ):
        raise EvaluationError("run config/manifest belongs to another campaign arm")


def _evaluation_filename(
    arm, checkpoint_name, epoch, sample_count, checkpoint_hash, evaluation_id=None,
) -> str:
    epoch_token = format(float(epoch), ".9g").replace(".", "p")
    suffix = f"_{evaluation_id}" if evaluation_id is not None else ""
    return (
        f"{arm}_{checkpoint_name}_epoch{epoch_token}_n{sample_count}_"
        f"{checkpoint_hash[:12]}{suffix}_validation.json"
    )


def _load_families_and_banks(config, manifest, source_hashes, run_dir, train_dataset, device):
    from phycoflow_reconstruction.coherence.compose import build_enabled_families
    from phycoflow_reconstruction.coherence.reference_bank import ReferenceBank
    from phycoflow_reconstruction.training.run_store import file_sha256

    family_names = tuple(manifest.get("coherence_families", ()))
    if not {"global_distribution", "cross_spectrum", "topology"}.issubset(family_names):
        raise EvaluationError(
            "campaign validation requires the saved full A/B/T family composition"
        )
    families = build_enabled_families(
        config["coherence"], train_dataset.data_spec, train_dataset.normalizer
    )
    if tuple(families) != family_names:
        raise EvaluationError("resolved family order differs from the saved campaign run")
    expected_hashes = manifest.get("coherence_family_state_sha256s", {})
    for name, family in families.items():
        path = run_dir / "artifacts" / f"{name}_family.pt"
        if not path.is_file() or expected_hashes.get(name) != file_sha256(path):
            raise EvaluationError(f"saved family artifact is missing or changed: {name}")
        artifact = torch.load(path, map_location=device, weights_only=True)
        family.load_state_artifact(artifact)

    calibration_path = run_dir / "artifacts" / "coherence_calibration.json"
    if not calibration_path.is_file():
        raise EvaluationError("run is missing its fixed SOURCE family calibration")
    if manifest.get("coherence_calibration_sha256") != file_sha256(calibration_path):
        raise EvaluationError("family calibration hash differs from the run manifest")
    calibration = _read_json(calibration_path)
    if calibration.get("source_checkpoint_sha256") != source_hashes.get("checkpoint"):
        raise EvaluationError("family calibration is not bound to the current SOURCE")
    family_scales = calibration.get("resolved_scales")
    if not isinstance(family_scales, Mapping) or set(family_scales) != set(families):
        raise EvaluationError("fixed family scales do not align with the saved A/B/T family set")

    banks = {}
    expected_bank_hashes = manifest.get("reference_bank_sha256s", {})
    for name, family in families.items():
        bank_path = (
            run_dir
            / "artifacts"
            / (
                "coherence_reference.pt"
                if name == "global_distribution"
                else f"coherence_reference_{name}.pt"
            )
        )
        if family.target_use == "paired_supervised":
            banks[name] = None
            continue
        if not bank_path.is_file():
            raise EvaluationError(f"training-reference artifact is missing for {name}")
        bank = ReferenceBank.load(bank_path)
        expected_digest = expected_bank_hashes.get(name)
        if expected_digest is None or bank.digest() != expected_digest:
            raise EvaluationError(
                f"training-reference digest differs from the run manifest: {name}"
            )
        if bank.metadata.get("dataset_fingerprint") != manifest.get("dataset_fingerprint"):
            raise EvaluationError(f"training-reference data identity differs from the run: {name}")
        if tuple(bank.metadata.get("field_names", ())) != tuple(train_dataset.field_names):
            raise EvaluationError(
                f"training-reference field order differs from the dataset: {name}"
            )
        banks[name] = bank

    for family in families.values():
        family.to(device)
    return families, banks, dict(family_scales)


def evaluate_campaign_run(
    campaign_root: Path,
    run_path: Path,
    *,
    checkpoint_name: str = "best",
    max_samples: int = 64,
    device_name: str = "cpu",
) -> Path:
    """Evaluate SOURCE and one child checkpoint on the same validation groups."""
    _validate_max_samples(max_samples)
    if checkpoint_name not in CHECKPOINT_NAMES:
        raise EvaluationError(f"unsupported checkpoint selection {checkpoint_name!r}")
    root, run_dir, arm = _validate_campaign_run(campaign_root, run_path)
    config, manifest, checkpoint_path, checkpoint_sha256 = _load_run_inputs(
        run_dir, checkpoint_name
    )
    _validate_experiment_binding(config, manifest, root, arm)
    # This is a hard-coded validation-only evaluator; there is no TEST override.
    if config.get("evaluation", {}).get("split", "validation") != "validation":
        raise EvaluationError("campaign evaluation requires evaluation.split='validation'")

    from phycoflow_reconstruction.data.factory import open_field_dataset
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

    if device_name not in {"cpu", "cuda:0"}:
        raise EvaluationError("use cpu or the explicitly reserved logical cuda:0")
    device = torch.device(device_name)
    if device.type == "cuda" and (
        os.environ.get("PHYCOFLOW_CAMPAIGN_ADMITTED") != str(root)
        or os.environ.get("CUDA_VISIBLE_DEVICES") != "1"
    ):
        raise EvaluationError("CUDA evaluation requires the campaign admission/lock wrapper")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise EvaluationError("the reserved CUDA device is unavailable")
    from phycoflow_reconstruction.training.post_training import _configure_persistence_workers

    _configure_persistence_workers(config)
    actual_source_hashes = source_hashes(config)
    if manifest.get("source_hashes") != actual_source_hashes:
        raise EvaluationError("current SOURCE hashes differ from the campaign run manifest")

    model, train_dataset, source_metadata = load_source_model(config, device)
    validation_dataset = None
    try:
        manifest_dataset_path = manifest.get("dataset_path")
        configured_dataset_path = Path(config["dataset"]["path"]).resolve()
        if (
            manifest_dataset_path is not None
            and Path(manifest_dataset_path).resolve() != configured_dataset_path
        ):
            raise EvaluationError("resolved dataset path differs from the campaign run manifest")
        validation_dataset = open_field_dataset(
            config["dataset"], split="validation", normalizer=train_dataset.normalizer
        )
        validation_indices = _uniform_validation_indices(len(validation_dataset), max_samples)
        samples = [validation_dataset[index] for index in validation_indices]
        protocol = sensor_protocol_from_config(config)
        evaluation_config = deepcopy(config)
        evaluation_config.setdefault("evaluation", {})["seed"] = VALIDATION_SEED
        complete_batch = build_observation_batch(samples, protocol, query_points=None).to(device)
        comparison_batch = _build_comparison_batch(complete_batch, evaluation_config)
        if comparison_batch.target_fields is None:
            raise EvaluationError("validation comparison batch did not retain dense targets")
        if not bool(comparison_batch.query_valid_mask.all()):
            raise EvaluationError(
                "equal-count pooling requires the complete valid Run 11 query mask"
            )
        if len(complete_batch.sample_ids) != max_samples:
            raise EvaluationError("validation batch size differs from the requested sample count")

        families, banks, family_scales = _load_families_and_banks(
            config, manifest, actual_source_hashes, run_dir, train_dataset, device
        )
        field_names = tuple(train_dataset.field_names)
        slices = _group_slices(max_samples)
        query_indices = comparison_batch.metadata.get("query_indices")
        observation_indices = comparison_batch.obs_indices
        if not isinstance(query_indices, torch.Tensor) or observation_indices is None:
            raise EvaluationError("validation batches must retain observation and query indices")
        query_sha256 = _tensor_sha256(query_indices)
        observation_sha256 = hashlib.sha256()
        for value in (
            observation_indices,
            comparison_batch.obs_field_ids,
            comparison_batch.obs_valid_mask,
        ):
            observation_sha256.update(value.detach().cpu().contiguous().numpy().tobytes())
        protocol_payload = comparison_batch.metadata.get("protocol", protocol.to_dict())
        manifest_payload = {
            "split": "validation",
            "dataset_path": str(Path(config["dataset"]["path"]).resolve()),
            "dataset_fingerprint": manifest.get("dataset_fingerprint"),
            "validation_indices": validation_indices,
            "sample_ids": list(comparison_batch.sample_ids),
            "protocol": protocol_payload,
            "observation_indices_sha256": observation_sha256.hexdigest(),
            "query_indices_sha256": query_sha256,
        }
        manifest_sha256 = hashlib.sha256(
            json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

        # Evaluate the immutable SOURCE before loading the child checkpoint.
        source_groups = _evaluate_groups(
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
        source_summary = _summarize_groups(source_groups)

        # Load and validate selected child state only after the paired SOURCE evaluation.
        checkpoint = load_project_checkpoint(checkpoint_path)
        if checkpoint.get("config_sha256") != manifest["config_sha256"]:
            raise EvaluationError("requested checkpoint is not bound to this resolved config")
        if checkpoint.get("source_hashes") != actual_source_hashes:
            raise EvaluationError(
                "selected checkpoint SOURCE hashes differ from current SOURCE artifacts"
            )
        if (
            Path(str(checkpoint.get("source_run", ""))).resolve()
            != Path(str(config["source_run"])).resolve()
        ):
            raise EvaluationError("selected checkpoint points to a different SOURCE run")
        if config["optimization"].get("fast_lagrangian"):
            from phycoflow_reconstruction.training.fast_lagrangian import implementation_binding

            if checkpoint.get("fast_lagrangian_implementation") != implementation_binding():
                raise EvaluationError(
                    "current fast-Lagrangian implementation differs from the run checkpoint"
                )
        if tuple(checkpoint.get("data_spec", {}).get("field_names", ())) != tuple(
            train_dataset.field_names
        ):
            raise EvaluationError(
                "selected checkpoint field order differs from the TRAIN/SOURCE data spec"
            )
        normalization = checkpoint.get("normalization", {})
        if (
            not torch.equal(normalization.get("offset"), train_dataset.normalizer.offset)
            or not torch.equal(normalization.get("scale"), train_dataset.normalizer.scale)
            or normalization.get("method") != train_dataset.normalizer.method
        ):
            raise EvaluationError("selected child checkpoint normalization differs from SOURCE")
        calibration_path = run_dir / "artifacts" / "coherence_calibration.json"
        if checkpoint.get("coherence_calibration_sha256") != file_sha256(calibration_path):
            raise EvaluationError(
                "checkpoint calibration hash differs from the current run artifact"
            )
        if checkpoint.get("family_scales") != family_scales:
            raise EvaluationError(
                "checkpoint family scales differ from the fixed calibration artifact"
            )
        checkpoint_family_states = checkpoint.get("family_states")
        if not isinstance(checkpoint_family_states, Mapping) or set(
            checkpoint_family_states
        ) != set(families):
            raise EvaluationError(
                "checkpoint family states do not match the full family composition"
            )

        load_model_state_strict(model, checkpoint["model"])
        load_training_aux_state(model, checkpoint)
        for name, family in families.items():
            family.load_state_dict(checkpoint_family_states[name], strict=True)
        child_groups = _evaluate_groups(
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
        child_summary = _summarize_groups(child_groups)

        # Detect an input replacement during a long CPU/GUDHI evaluation.
        if source_hashes(config) != actual_source_hashes:
            raise EvaluationError("SOURCE artifacts changed during paired validation")
        if file_sha256(checkpoint_path) != checkpoint_sha256:
            raise EvaluationError("selected child checkpoint changed during paired validation")

        source_metrics = _ratioable_metrics(source_summary)
        child_metrics = _ratioable_metrics(child_summary)
        field_vector = {
            "field_names": list(field_names),
            "source": [source_summary["endpoint_mse_model_units"][name] for name in field_names],
            "child": [child_summary["endpoint_mse_model_units"][name] for name in field_names],
            "child_over_source": [
                _ratio_tree(
                    child_summary["endpoint_mse_model_units"][name],
                    source_summary["endpoint_mse_model_units"][name],
                )
                for name in field_names
            ],
        }
        group_rows = []
        for interval, source_group, child_group in zip(slices, source_groups, child_groups):
            start, stop = interval
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

        evaluated_at = datetime.now(timezone.utc)
        payload = {
            "schema_version": 1,
            "evaluated_at_utc": evaluated_at.isoformat(),
            "campaign_root": str(root),
            "arm": arm,
            "run": str(run_dir),
            "checkpoint": checkpoint_name,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
            "checkpoint_training_epoch": (
                float(checkpoint.get("global_step", 0))
                / post_training_steps_per_epoch(config, len(train_dataset))
            ),
            "source": {
                "run": str(Path(config["source_run"]).resolve()),
                "hashes": actual_source_hashes,
                "metadata": source_metadata,
            },
            "config_sha256": manifest["config_sha256"],
            "dataset": {
                "path": str(Path(config["dataset"]["path"]).resolve()),
                "split": "validation",
                "run_manifest_fingerprint": manifest.get("dataset_fingerprint"),
                "normalizer": train_dataset.normalizer.digest(),
            },
            "selection": {
                **manifest_payload,
                "sensor_query_manifest_sha256": manifest_sha256,
                "query_indices_shape": list(query_indices.shape),
                "groups": [validation_indices[start:stop] for start, stop in slices],
                "group_size": GROUP_SIZE,
                "group_count": len(slices),
                "generation_seed": VALIDATION_SEED,
                "generation_seed_stride": 100_003,
            },
            "evaluation": {
                "mode": "full_existing_A_B_T_family_forward",
                "aggregation": (
                    "sample-weighted mean of full 32-snapshot group scores; "
                    "B is a mean of B(32) ensemble statistics, not B(whole panel); "
                    "worst field is the maximum pooled per-field relative L2; "
                    "observed/unobserved entry MSE is retained per group only"
                ),
                "topology_is_stochastic": False,
                "samples": max_samples,
                "source_group_means": source_summary,
                "child_group_means": child_summary,
                "full_metric_ratios_child_over_source": _ratio_tree(child_metrics, source_metrics),
                "endpoint_mse_model_units": field_vector,
            },
            "components": {
                "source": source_summary["coherence"]["families"],
                "child": child_summary["coherence"]["families"],
            },
            "groups": group_rows,
        }
        output_path = root / "analysis" / _evaluation_filename(
            arm, checkpoint_name, payload["checkpoint_training_epoch"],
            max_samples, checkpoint_sha256, evaluated_at.strftime("%Y%m%dT%H%M%S%fZ"),
        )
        _atomic_json(output_path, payload)
        return output_path
    finally:
        if validation_dataset is not None:
            close = getattr(validation_dataset, "close", None)
            if callable(close):
                close()
        close = getattr(train_dataset, "close", None)
        if callable(close):
            close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_max_samples(args.max_samples)
        output = evaluate_campaign_run(
            args.campaign_root,
            args.run,
            checkpoint_name=args.checkpoint,
            max_samples=args.max_samples,
            device_name=args.device,
        )
    except (EvaluationError, OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"fast_lagrangian_evaluate: {error}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
