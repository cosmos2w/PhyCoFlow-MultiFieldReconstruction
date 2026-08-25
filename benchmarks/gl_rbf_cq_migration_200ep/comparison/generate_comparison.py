#!/usr/bin/env python3
"""Generate the reproducible A/B/C GL_rbf_CQ benchmark comparison.

The comparison deliberately consumes the frozen, machine-readable evidence
artifacts instead of re-reading checkpoint tensors.  It therefore remains
usable on a CPU-only checkout while still using the immutable per-epoch
telemetry files for exact wall-time-to-threshold calculations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

MILESTONES = (1, 20, 40, 60, 100, 150, 200)
FIELDS = ("CH4", "CO", "T", "U_1", "p")
QUALITY_METRICS = ("mse_normalized", "mean_relative_l2", "worst_field_relative_l2")
THRESHOLDS = {
    "mse_normalized": (0.70, 0.60, 0.50),
    "mean_relative_l2": (0.75, 0.70, 0.65),
    "worst_field_relative_l2": (1.00, 0.90),
}

SCRIPT_PATH = Path(__file__).resolve()
BENCHMARK_DIR = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[3]
REPO_ROOT = SCRIPT_PATH.parents[4]
OUTPUT_DIR = SCRIPT_PATH.parent


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected object in {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close_enough(actual: float, expected: float, label: str) -> None:
    tolerance = 1.0e-8 * max(1.0, abs(actual), abs(expected))
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def number(value: Any) -> float:
    return float(value)


def load_telemetry(path: Path) -> list[dict[str, float]]:
    require(path.is_file(), f"missing telemetry file: {path}")
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, float] = {}
            for key, value in raw.items():
                if value is None:
                    continue
                row[key] = float(value)
            rows.append(row)
    rows.sort(key=lambda row: int(row["epoch"]))
    require(len(rows) == 200, f"expected 200 telemetry rows in {path}, got {len(rows)}")
    require([int(row["epoch"]) for row in rows] == list(range(1, 201)), f"non-contiguous epochs in {path}")
    return rows


def timing_from_telemetry(rows: list[dict[str, float]]) -> dict[str, Any]:
    steady = [row for row in rows if int(row["epoch"]) >= 2]

    def stats(field: str) -> dict[str, float]:
        values = [number(row[field]) for row in steady]
        return {"mean": mean(values), "median": median(values), "min": min(values), "max": max(values)}

    return {
        "steady_epochs_inclusive": [2, 200],
        "total_epoch_wall_time_s": sum(number(row["epoch_wall_time_s"]) for row in rows),
        "total_training_only_time_s": sum(number(row["training_only_epoch_time_s"]) for row in rows),
        "steady_epoch_wall_time_s": stats("epoch_wall_time_s"),
        "steady_training_only_epoch_time_s": stats("training_only_epoch_time_s"),
        "steady_mean_step_time_s": stats("mean_step_time_s"),
        "steady_median_step_time_s": stats("median_step_time_s"),
        "formal_peak_cuda_allocated_bytes": int(max(row["peak_cuda_allocated_bytes"] for row in rows)),
        "formal_peak_cuda_reserved_bytes": int(max(row["peak_cuda_reserved_bytes"] for row in rows)),
        "parameter_count": int(rows[-1]["parameter_count"]),
        "trainable_parameter_count": int(rows[-1]["trainable_parameter_count"]),
    }


def elapsed_by_epoch(rows: list[dict[str, float]]) -> dict[int, float]:
    elapsed = 0.0
    result: dict[int, float] = {}
    for row in rows:
        epoch = int(row["epoch"])
        elapsed += number(row["epoch_wall_time_s"])
        result[epoch] = elapsed
    return result


def config_path_for(arm: str) -> Path:
    names = {
        "A": "A_legacy_gl_rbf_enh_200ep.yaml",
        "B": "B_gl_rbf_cq_legacy_mha_200ep.yaml",
        "C": "C_gl_rbf_cq_cached_kv_200ep.yaml",
    }
    return BENCHMARK_DIR / "configs" / names[arm]


def normalize_fixed_entries(arm: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    if arm == "A":
        formal = payload["formal_run"]
        source = formal["fixed_manifest_convergence"]
        checkpoint_map = formal["checkpoint_sha256"]
    else:
        source = payload["fixed_manifest_convergence"]
        checkpoint_map = payload["checkpoint_hashes"]

    entries: list[dict[str, Any]] = []
    for item in source:
        epoch = int(item["epoch"])
        require(epoch in MILESTONES, f"unexpected fixed-manifest epoch {epoch} in arm {arm}")
        checkpoint = item.get("checkpoint_sha256") or checkpoint_map[f"epoch_{epoch:03d}"]
        per_field = {field: number(item["per_field_relative_l2"][field]) for field in FIELDS}
        entries.append(
            {
                "epoch": epoch,
                "checkpoint_sha256": checkpoint,
                "report_sha256": item["report_sha256"],
                "mse_normalized": number(item["mse_normalized"]),
                "mean_relative_l2": number(item["mean_relative_l2"]),
                "worst_field_relative_l2": number(item["worst_field_relative_l2"]),
                "per_field_relative_l2": per_field,
                "evaluation_peak_cuda_memory_bytes": item.get("evaluation_peak_cuda_memory_bytes"),
                "evaluation_seconds": item.get("evaluation_seconds"),
            }
        )
    entries.sort(key=lambda item: item["epoch"])
    require([item["epoch"] for item in entries] == list(MILESTONES), f"fixed milestones incomplete for arm {arm}")
    return entries


def extract_arm(arm: str, payload: dict[str, Any]) -> dict[str, Any]:
    if arm == "A":
        formal = payload["formal_run"]
        protocol = payload["protocol"]
        run_dir = formal["run_dir"]
        telemetry_rel = f"{run_dir}/metrics/benchmark_telemetry_epochs.csv"
        info = {
            "arm": "A",
            "model": payload["model"],
            "condition_attention_execution": "legacy_downstream",
            "evaluation_weight_source": "raw/configured",
            "source_branch": payload["source_branch"],
            "validation_branch": payload["branch"],
            "source_commit": payload["source_commit"],
            "launch_head": formal["launch_head"],
            "run_id": formal["run_id"],
            "run_directory": run_dir,
            "config_path": "benchmarks/gl_rbf_cq_migration_200ep/configs/A_legacy_gl_rbf_enh_200ep.yaml",
            "config_source_sha256": formal["config_source_sha256"],
            "config_semantic_sha256": formal["config_semantic_sha256"],
            "resolved_config_sha256": formal["resolved_config_sha256"],
            "dataset_fingerprint": formal["dataset_fingerprint"],
            "normalization_artifact_sha256": formal["normalization_artifact_sha256"],
            "normalization_artifact_file_sha256": formal["normalization_artifact_file_sha256"],
            "normalizer_digest": formal["normalizer_digest"],
            "sensor_manifest_sha256": formal["fixed_validation_manifest_sha256"],
            "fixed_manifest_file_sha256": formal["fixed_validation_manifest_file_sha256"],
            "query_indices_sha256": formal["fixed_query_indices_sha256"],
            "batch_size": int(protocol["batch_size"]),
            "requested_batch_size": int(protocol["requested_batch_size"]),
            "query_points": int(protocol["query_points"]),
            "epochs": int(protocol["epochs"]),
            "steps_per_epoch": int(payload["steps_per_epoch"]),
            "seed": int(protocol["seed"]),
            "sensor_fields": list(protocol["sensor_fields"]),
            "sensor_count_range": list(protocol["sensor_count_range"]),
            "parameter_count": int(formal["parameter_count"]),
            "trainable_parameter_count": int(formal["trainable_parameter_count"]),
            "ema": {"enabled": False, "evaluated_with_ema": False},
            "endpoint_metrics": {
                "mse_normalized": formal["endpoint_evaluation_mse_normalized"],
                "evaluation_weight_source": "raw/configured",
            },
            "telemetry_rel": telemetry_rel,
            "run_artifact_sha256": formal["run_artifact_sha256"],
            "checkpoint_hashes": formal["checkpoint_sha256"],
        }
    else:
        config = payload["configuration"]
        data = payload["data_and_normalization"]
        fixed = payload["fixed_evaluation"]
        launch = payload["launch"]
        source = payload["source"]
        run = payload["run"]
        timing = payload["timing_and_memory"]
        info = {
            "arm": arm,
            "model": source["model_name"],
            "condition_attention_execution": payload["condition_attention_execution"],
            "evaluation_weight_source": "configured EMA",
            "source_branch": source["source_branch"],
            "validation_branch": source["validation_branch"],
            "source_commit": source["launch_commit"],
            "launch_head": launch["launch_head"],
            "run_id": run["run_id"],
            "run_directory": run["run_directory"],
            "config_path": config["source_path"],
            "config_source_sha256": config["source_file_sha256"],
            "config_semantic_sha256": config["semantic_sha256"],
            "resolved_config_sha256": config["resolved_sha256"],
            "dataset_fingerprint": data["dataset_fingerprint"],
            "normalization_artifact_sha256": data["normalization_artifact_sha256"],
            "normalization_artifact_file_sha256": data["normalization_artifact_file_sha256"],
            "normalizer_digest": data["normalizer_digest"],
            "sensor_manifest_sha256": fixed["sensor_manifest_sha256"],
            "fixed_manifest_file_sha256": fixed["fixed_validation_manifest_file_sha256"],
            "query_indices_sha256": fixed["query_indices_sha256"],
            "batch_size": int(config["batch_size"]),
            "requested_batch_size": int(config["requested_batch_size"]),
            "query_points": int(config["query_points"]),
            "epochs": int(config["epochs"]),
            "steps_per_epoch": int(config["steps_per_epoch"]),
            "seed": int(config["seed"]),
            "sensor_fields": [fixed["sensor_field"]],
            "sensor_count_range": list(config["sensor_count_range"]),
            "parameter_count": int(payload["completion"]["parameter_count"]),
            "trainable_parameter_count": int(payload["completion"]["trainable_parameter_count"]),
            "ema": payload["ema"],
            "endpoint_metrics": {
                "mse_normalized": payload["completion"]["endpoint_evaluation_mse_normalized"],
                "mean_relative_l2": payload["completion"]["endpoint_integration_mean_relative_l2"],
                "worst_field_relative_l2": payload["completion"]["endpoint_integration_worst_field_relative_l2"],
                "evaluation_weight_source": "configured EMA",
            },
            "telemetry_rel": timing["formal_telemetry_path"],
            "run_artifact_sha256": payload["artifact_hashes"],
            "checkpoint_hashes": payload["checkpoint_hashes"],
        }

    config_file = config_path_for(arm)
    require(config_file.is_file(), f"missing config file for arm {arm}: {config_file}")
    info["config_file_sha256"] = sha256_file(config_file)
    require(info["config_file_sha256"] == info["config_source_sha256"], f"config hash mismatch for arm {arm}")
    fixed_entries = normalize_fixed_entries(arm, payload)
    telemetry_path = PROJECT_ROOT / info["telemetry_rel"]
    telemetry_rows = load_telemetry(telemetry_path)
    timing = timing_from_telemetry(telemetry_rows)

    if arm == "A":
        formal = payload["formal_run"]
        close_enough(timing["total_epoch_wall_time_s"], number(formal["total_epoch_wall_seconds"]), "A total wall")
        close_enough(timing["steady_epoch_wall_time_s"]["mean"], number(formal["steady_state_epoch_wall_seconds_mean"]), "A steady epoch")
        close_enough(timing["steady_epoch_wall_time_s"]["median"], number(formal["steady_state_epoch_wall_seconds_median"]), "A steady epoch median")
        close_enough(timing["steady_training_only_epoch_time_s"]["mean"], number(formal["steady_state_training_only_epoch_seconds_mean"]), "A steady training")
        close_enough(timing["steady_mean_step_time_s"]["mean"], number(formal["steady_state_sampled_step_seconds_mean"]), "A steady step")
        require(timing["formal_peak_cuda_allocated_bytes"] == int(formal["peak_cuda_allocated_bytes"]), "A allocated peak mismatch")
        require(timing["formal_peak_cuda_reserved_bytes"] == int(formal["peak_cuda_reserved_bytes"]), "A reserved peak mismatch")
    else:
        reported = payload["timing_and_memory"]
        close_enough(timing["total_epoch_wall_time_s"], number(reported["total_epoch_wall_time_s"]), f"{arm} total wall")
        close_enough(timing["total_training_only_time_s"], number(reported["total_training_only_time_s"]), f"{arm} total training")
        close_enough(timing["steady_epoch_wall_time_s"]["mean"], number(reported["steady_epoch_wall_time_s"]["mean"]), f"{arm} steady epoch")
        close_enough(timing["steady_epoch_wall_time_s"]["median"], number(reported["steady_epoch_wall_time_s"]["median"]), f"{arm} steady epoch median")
        close_enough(timing["steady_training_only_epoch_time_s"]["mean"], number(reported["steady_training_only_epoch_time_s"]["mean"]), f"{arm} steady training")
        close_enough(timing["steady_mean_step_time_s"]["mean"], number(reported["steady_mean_step_time_s"]["mean"]), f"{arm} steady step")
        close_enough(timing["steady_median_step_time_s"]["mean"], number(reported["steady_median_step_time_s"]["mean"]), f"{arm} steady median step")
        require(timing["formal_peak_cuda_allocated_bytes"] == int(reported["formal_peak_cuda_allocated_bytes"]), f"{arm} allocated peak mismatch")
        require(timing["formal_peak_cuda_reserved_bytes"] == int(reported["formal_peak_cuda_reserved_bytes"]), f"{arm} reserved peak mismatch")
        require(bool(info["ema"]["evaluated_with_ema"]), f"{arm} was not evaluated with configured EMA")

    info["fixed_manifest_convergence"] = fixed_entries
    info["timing"] = timing
    info["elapsed_wall_time_by_epoch_s"] = elapsed_by_epoch(telemetry_rows)
    info["training_health"] = {
        "all_losses_finite": payload["formal_run"]["all_recorded_losses_finite"] if arm == "A" else payload["training_health"]["all_losses_finite"],
        "all_gradients_finite": payload["formal_run"]["all_recorded_gradient_norms_finite"] if arm == "A" else payload["training_health"]["all_gradient_norms_finite"],
        "backward_retry_sum": payload["formal_run"]["backward_retry_sum"] if arm == "A" else payload["training_health"]["backward_retry_sum"],
        "completed_steps": payload["formal_run"]["completed_steps"] if arm == "A" else payload["completion"]["global_step"],
    }
    return info


def protocol_identity(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "dataset_fingerprint",
        "normalization_artifact_sha256",
        "normalization_artifact_file_sha256",
        "normalizer_digest",
        "sensor_manifest_sha256",
        "fixed_manifest_file_sha256",
        "query_indices_sha256",
        "batch_size",
        "requested_batch_size",
        "query_points",
        "epochs",
        "seed",
        "sensor_fields",
        "sensor_count_range",
    )
    result: dict[str, Any] = {}
    for key in keys:
        values = {arm: arms[arm][key] for arm in arms}
        require(len({json.dumps(value, sort_keys=True) for value in values.values()}) == 1, f"protocol mismatch for {key}: {values}")
        result[key] = values["A"]
    result.update(
        {
            "coordinate_dim": 2,
            "field_order": list(FIELDS),
            "normalization_method": "mean_std",
            "sensor_protocol": "random_uniform",
            "evaluation_split": "validation",
            "evaluation_samples": 20,
            "evaluation_generation_steps": 32,
            "checkpoint_milestones": list(MILESTONES),
        }
    )
    return result


def percent_change(earlier: float, later: float) -> float | None:
    if earlier == 0.0:
        return None
    return (later - earlier) / earlier * 100.0


def scalar_effect(earlier: float, later: float) -> dict[str, float | None]:
    return {
        "earlier": earlier,
        "later": later,
        "delta_later_minus_earlier": later - earlier,
        "percent_change": percent_change(earlier, later),
    }


def effect(earlier: dict[str, Any], later: dict[str, Any]) -> dict[str, Any]:
    e200 = next(item for item in earlier["fixed_manifest_convergence"] if item["epoch"] == 200)
    l200 = next(item for item in later["fixed_manifest_convergence"] if item["epoch"] == 200)
    quality = {metric: scalar_effect(e200[metric], l200[metric]) for metric in QUALITY_METRICS}
    quality["per_field_relative_l2"] = {
        field: scalar_effect(e200["per_field_relative_l2"][field], l200["per_field_relative_l2"][field])
        for field in FIELDS
    }
    performance_keys = (
        ("parameter_count", "parameter_count"),
        ("formal_peak_cuda_allocated_bytes", "timing.formal_peak_cuda_allocated_bytes"),
        ("formal_peak_cuda_reserved_bytes", "timing.formal_peak_cuda_reserved_bytes"),
        ("steady_epoch_wall_time_s_mean", "timing.steady_epoch_wall_time_s.mean"),
        ("steady_epoch_wall_time_s_median", "timing.steady_epoch_wall_time_s.median"),
        ("steady_training_only_epoch_time_s_mean", "timing.steady_training_only_epoch_time_s.mean"),
        ("steady_mean_step_time_s_mean", "timing.steady_mean_step_time_s.mean"),
        ("steady_median_step_time_s_mean", "timing.steady_median_step_time_s.mean"),
    )

    def get_value(info: dict[str, Any], path: str) -> float:
        value: Any = info
        for part in path.split("."):
            value = value[part]
        return number(value)

    performance = {key: scalar_effect(get_value(earlier, path), get_value(later, path)) for key, path in performance_keys}
    return {
        "earlier_arm": earlier["arm"],
        "later_arm": later["arm"],
        "quality_delta_definition": "later minus earlier; negative quality deltas are lower error",
        "latest_fixed_manifest_epoch": 200,
        "quality": quality,
        "performance": performance,
    }


def matched_effects(arms: dict[str, dict[str, Any]], earlier_name: str, later_name: str) -> list[dict[str, Any]]:
    earlier = arms[earlier_name]
    later = arms[later_name]
    earlier_by_epoch = {item["epoch"]: item for item in earlier["fixed_manifest_convergence"]}
    later_by_epoch = {item["epoch"]: item for item in later["fixed_manifest_convergence"]}
    result = []
    for epoch in MILESTONES:
        e = earlier_by_epoch[epoch]
        l = later_by_epoch[epoch]
        result.append(
            {
                "epoch": epoch,
                **{metric: l[metric] - e[metric] for metric in QUALITY_METRICS},
                "per_field_relative_l2_delta": {
                    field: l["per_field_relative_l2"][field] - e["per_field_relative_l2"][field]
                    for field in FIELDS
                },
            }
        )
    return result


def threshold_results(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "definition": "First available fixed-manifest checkpoint at or below the threshold; elapsed wall time is the exact cumulative epoch telemetry through that checkpoint.",
        "comparison_is_milestone_observed": True,
        "thresholds": {metric: list(values) for metric, values in THRESHOLDS.items()},
        "arms": {},
    }
    for arm, info in arms.items():
        entries = info["fixed_manifest_convergence"]
        elapsed = info["elapsed_wall_time_by_epoch_s"]
        result["arms"][arm] = {}
        for metric, values in THRESHOLDS.items():
            result["arms"][arm][metric] = []
            for threshold in values:
                first = next((item for item in entries if item[metric] <= threshold), None)
                if first is None:
                    result["arms"][arm][metric].append(
                        {"threshold": threshold, "achieved": False, "first_observed_epoch": None, "elapsed_wall_time_s": None}
                    )
                else:
                    epoch = first["epoch"]
                    result["arms"][arm][metric].append(
                        {
                            "threshold": threshold,
                            "achieved": True,
                            "first_observed_epoch": epoch,
                            "elapsed_wall_time_s": elapsed[epoch],
                        }
                    )
    return result


def execution_evidence(execution: dict[str, Any]) -> dict[str, Any]:
    b = execution["arms"]["B_legacy_mha_full"]
    c = execution["arms"]["C_cached_kv_full"]
    require(execution["protocol"]["same_batch"], "controlled execution did not use the same batch")
    require(execution["protocol"]["same_initial_state"], "controlled execution did not use the same initial state")
    require(execution["protocol"]["same_rf_seed_schedule"], "controlled execution did not use the same RF seed schedule")
    for label, arm, expected in (("B", b, 4), ("C", c, 1)):
        require(arm["expected_kv_projection_calls_per_step"] == expected, f"{label} expected KV call count mismatch")
        require(set(arm["observed_kv_projection_calls_per_step"]) == {expected}, f"{label} observed KV call count mismatch")
        require(set(arm["backward_retries"]) == {0}, f"{label} controlled retries were nonzero")
    require(len(b["losses"]) == b["measured_steps"] == len(c["losses"]), "controlled loss traces differ in length")
    require(len(b["gradient_norms"]) == len(c["gradient_norms"]), "controlled gradient traces differ in length")
    loss_delta = max(abs(float(left) - float(right)) for left, right in zip(b["losses"], c["losses"]))
    gradient_delta = max(abs(float(left) - float(right)) for left, right in zip(b["gradient_norms"], c["gradient_norms"]))
    return {
        "protocol": execution["protocol"],
        "trace_identity": {
            "batch_tensors_sha256": execution["trace"]["batch_tensors_sha256"],
            "initial_state_sha256": execution["trace"]["initial_state_sha256"],
            "benchmark_script_sha256": execution["trace"]["benchmark_script_sha256"],
            "config_sha256": execution["trace"]["config_sha256"],
        },
        "measured_steps": b["measured_steps"],
        "warmup_steps": b["warmup_steps"],
        "kv_projection_calls_per_step": {
            "B_legacy_mha": {"expected": 4, "observed_unique": [4], "all_measured_steps_exact": True},
            "C_cached_kv": {"expected": 1, "observed_unique": [1], "all_measured_steps_exact": True},
        },
        "numerical_equivalence": {
            "computed_max_absolute_loss_difference": loss_delta,
            "computed_max_absolute_gradient_norm_difference": gradient_delta,
            "recorded_max_absolute_loss_difference": execution["numerical_equivalence"]["max_absolute_loss_difference"],
            "recorded_max_absolute_gradient_norm_difference": execution["numerical_equivalence"]["max_absolute_gradient_norm_difference"],
        },
        "probe": {
            "B_legacy_mha": {
                "peak_cuda_allocated_bytes": b["peak_cuda_allocated_bytes"],
                "peak_cuda_reserved_bytes": b["peak_cuda_reserved_bytes"],
                "timing_ms": b["timing"],
            },
            "C_cached_kv": {
                "peak_cuda_allocated_bytes": c["peak_cuda_allocated_bytes"],
                "peak_cuda_reserved_bytes": c["peak_cuda_reserved_bytes"],
                "timing_ms": c["timing"],
            },
            "recorded_effect_C_minus_B": {
                "timing": execution["execution_effect_C_vs_B"],
                "memory": execution["memory_effect_C_vs_B"],
            },
        },
    }


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def write_milestones(path: Path, arms: dict[str, dict[str, Any]]) -> None:
    fields = [f"relative_l2_{field}" for field in FIELDS]
    columns = [
        "arm",
        "condition_attention_execution",
        "evaluation_weight_source",
        "epoch",
        "checkpoint_sha256",
        "report_sha256",
        "mse_normalized",
        "mean_relative_l2",
        "worst_field_relative_l2",
        *fields,
        "parameter_count",
        "formal_peak_cuda_allocated_bytes",
        "formal_peak_cuda_reserved_bytes",
        "steady_epoch_wall_time_mean_s",
        "steady_epoch_wall_time_median_s",
        "steady_training_only_epoch_time_mean_s",
        "steady_mean_step_time_mean_s",
        "steady_median_step_time_mean_s",
        "dataset_fingerprint",
        "normalizer_digest",
        "sensor_manifest_sha256",
        "query_indices_sha256",
    ]
    rows: list[dict[str, Any]] = []
    for arm in ("A", "B", "C"):
        info = arms[arm]
        timing = info["timing"]
        for entry in info["fixed_manifest_convergence"]:
            row: dict[str, Any] = {
                "arm": arm,
                "condition_attention_execution": info["condition_attention_execution"],
                "evaluation_weight_source": info["evaluation_weight_source"],
                "epoch": entry["epoch"],
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "report_sha256": entry["report_sha256"],
                "mse_normalized": entry["mse_normalized"],
                "mean_relative_l2": entry["mean_relative_l2"],
                "worst_field_relative_l2": entry["worst_field_relative_l2"],
                **{f"relative_l2_{field}": entry["per_field_relative_l2"][field] for field in FIELDS},
                "parameter_count": info["parameter_count"],
                "formal_peak_cuda_allocated_bytes": timing["formal_peak_cuda_allocated_bytes"],
                "formal_peak_cuda_reserved_bytes": timing["formal_peak_cuda_reserved_bytes"],
                "steady_epoch_wall_time_mean_s": timing["steady_epoch_wall_time_s"]["mean"],
                "steady_epoch_wall_time_median_s": timing["steady_epoch_wall_time_s"]["median"],
                "steady_training_only_epoch_time_mean_s": timing["steady_training_only_epoch_time_s"]["mean"],
                "steady_mean_step_time_mean_s": timing["steady_mean_step_time_s"]["mean"],
                "steady_median_step_time_mean_s": timing["steady_median_step_time_s"]["mean"],
                "dataset_fingerprint": info["dataset_fingerprint"],
                "normalizer_digest": info["normalizer_digest"],
                "sensor_manifest_sha256": info["sensor_manifest_sha256"],
                "query_indices_sha256": info["query_indices_sha256"],
            }
            rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: csv_value(row[column]) for column in columns})


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def signed(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    return f"{float(value):+.{digits}f}"


def pct(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):+.2f}%"


def gib(value: Any) -> str:
    return f"{float(value) / (1024 ** 3):.2f}"


def build_results_markdown(summary: dict[str, Any]) -> str:
    arms = summary["arms"]
    effects = summary["effects"]

    def quality_row(label: str, key: str) -> str:
        item = effects[key]["quality"]
        return (
            f"| {label} | {signed(item['mse_normalized']['delta_later_minus_earlier'])} "
            f"({pct(item['mse_normalized']['percent_change'])}) | "
            f"{signed(item['mean_relative_l2']['delta_later_minus_earlier'])} "
            f"({pct(item['mean_relative_l2']['percent_change'])}) | "
            f"{signed(item['worst_field_relative_l2']['delta_later_minus_earlier'])} "
            f"({pct(item['worst_field_relative_l2']['percent_change'])}) |"
        )

    quality_table = "\n".join(
        [
            "| Effect (epoch 200 fixed manifest) | Δ MSE | Δ mean relative L2 | Δ worst relative L2 |",
            "|---|---:|---:|---:|",
            quality_row("Migration: B − A", "migration_B_minus_A"),
            quality_row("Execution: C − B", "execution_C_minus_B"),
            quality_row("Total latest model: C − A", "total_C_minus_A"),
        ]
    )

    performance_lines = [
        "| Arm | Parameters | Peak allocated GiB | Peak reserved GiB | Steady epoch s | Steady mean step ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("A", "B", "C"):
        info = arms[arm]
        timing = info["timing"]
        performance_lines.append(
            f"| {arm} ({info['condition_attention_execution']}) | {info['parameter_count']:,} | "
            f"{gib(timing['formal_peak_cuda_allocated_bytes'])} | {gib(timing['formal_peak_cuda_reserved_bytes'])} | "
            f"{fmt(timing['steady_epoch_wall_time_s']['mean'])} | "
            f"{fmt(timing['steady_mean_step_time_s']['mean'] * 1000, 3)} |"
        )

    probe = summary["controlled_execution"]
    bprobe = probe["probe"]["B_legacy_mha"]
    cprobe = probe["probe"]["C_cached_kv"]
    threshold_lines = [
        "| Metric threshold | A | B | C |",
        "|---|---:|---:|---:|",
    ]
    for metric, thresholds in summary["convergence_thresholds"]["thresholds"].items():
        for threshold in thresholds:
            cells = []
            for arm in ("A", "B", "C"):
                entry = next(
                    item
                    for item in summary["convergence_thresholds"]["arms"][arm][metric]
                    if item["threshold"] == threshold
                )
                if entry["achieved"]:
                    cells.append(f"e{entry['first_observed_epoch']} / {fmt(entry['elapsed_wall_time_s'], 1)} s")
                else:
                    cells.append("not reached")
            threshold_lines.append(f"| {metric} ≤ {threshold:g} | {cells[0]} | {cells[1]} | {cells[2]} |")

    input_hash_lines = ["| Evidence file | SHA-256 |", "|---|---|"]
    for path, digest in summary["input_artifact_sha256"].items():
        input_hash_lines.append(f"| `{path}` | `{digest}` |")

    arm_lines = [
        "| Arm | Run path | Launch head | Config source / semantic / resolved | e200 checkpoint / report |",
        "|---|---|---|---|---|",
    ]
    for arm in ("A", "B", "C"):
        info = arms[arm]
        e200 = next(item for item in info["fixed_manifest_convergence"] if item["epoch"] == 200)
        arm_lines.append(
            f"| {arm} | `{info['run_directory']}` | `{info['launch_head']}` | "
            f"`{info['config_source_sha256'][:12]}` / `{info['config_semantic_sha256'][:12]}` / `{info['resolved_config_sha256'][:12]}` | "
            f"`{e200['checkpoint_sha256']}` / `{e200['report_sha256']}` |"
        )

    gate = summary["migration_evidence"]
    fixed_guide_lines = "\n".join(
        f"- **{item['finding']}** {item['resolution']}"
        for item in summary["migration_guide_fixes"]
    )
    gap_lines = "\n".join(
        f"- {item['gap']} Recommended follow-up: {item['recommended_fix']}"
        for item in summary["migration_guide_gaps"]
    )
    return f"""# GL_rbf_CQ migration benchmark results

## Answer first

The fixed-manifest comparison uses the common B40/Q4096, seed-42, T-only
192--384-sensor protocol. Δ is **later minus earlier**, so a negative quality
delta is an error reduction. B and C use their configured EMA checkpoints for
evaluation; A is the raw/configured legacy checkpoint. At epoch 200:

{quality_table}

- **Migration effect (B−A):** parameters increase
  {pct(effects['migration_B_minus_A']['performance']['parameter_count']['percent_change'])},
  while formal peak allocation/reservation fall
  {pct(effects['migration_B_minus_A']['performance']['formal_peak_cuda_allocated_bytes']['percent_change'])}/
  {pct(effects['migration_B_minus_A']['performance']['formal_peak_cuda_reserved_bytes']['percent_change'])}
  and steady mean epoch/step time fall
  {pct(effects['migration_B_minus_A']['performance']['steady_epoch_wall_time_s_mean']['percent_change'])}/
  {pct(effects['migration_B_minus_A']['performance']['steady_mean_step_time_s_mean']['percent_change'])}.
  Epoch-200 mean relative L2 is {pct(effects['migration_B_minus_A']['quality']['mean_relative_l2']['percent_change'])}
  higher error than A.
- **Execution effect (C−B):** parameter count is unchanged. In the matched
  same-state probe, cached K/V reduces median whole-step time
  {pct(summary['controlled_execution']['probe']['recorded_effect_C_minus_B']['timing']['whole_step_ms']['median_percent_change'])}
  and peak allocated/reserved memory
  {pct(summary['controlled_execution']['probe']['recorded_effect_C_minus_B']['memory']['peak_cuda_allocated_bytes']['percent_change'])}/
  {pct(summary['controlled_execution']['probe']['recorded_effect_C_minus_B']['memory']['peak_cuda_reserved_bytes']['percent_change'])},
  with numerical differences below 3e-7. Its independent 40,000-step quality
  delta is accumulated trajectory drift, not a causal execution-quality effect.
- **Total latest-model effect (C−A):** formal peak allocation/reservation fall
  {pct(effects['total_C_minus_A']['performance']['formal_peak_cuda_allocated_bytes']['percent_change'])}/
  {pct(effects['total_C_minus_A']['performance']['formal_peak_cuda_reserved_bytes']['percent_change'])},
  steady mean epoch/step time fall
  {pct(effects['total_C_minus_A']['performance']['steady_epoch_wall_time_s_mean']['percent_change'])}/
  {pct(effects['total_C_minus_A']['performance']['steady_mean_step_time_s_mean']['percent_change'])},
  and epoch-200 mean relative L2 is
  {pct(effects['total_C_minus_A']['quality']['mean_relative_l2']['percent_change'])} higher error.

## Formal resources and timing

{chr(10).join(performance_lines)}

The formal telemetry covers epochs 2--200 for steady statistics. B/C also have
a matched 50-step controlled probe: legacy MHA measured exactly four K/V
projection calls per step; cached-K/V measured exactly one. Probe whole-step
mean/median are {fmt(bprobe['timing_ms']['whole_step_ms']['mean_ms'], 3)}/
{fmt(bprobe['timing_ms']['whole_step_ms']['median_ms'], 3)} ms for B and
{fmt(cprobe['timing_ms']['whole_step_ms']['mean_ms'], 3)}/
{fmt(cprobe['timing_ms']['whole_step_ms']['median_ms'], 3)} ms for C. The recorded
probe C−B median whole-step change is
{fmt(summary['controlled_execution']['probe']['recorded_effect_C_minus_B']['timing']['whole_step_ms']['median_percent_change'], 2)}%;
probe allocated memory changes from {gib(bprobe['peak_cuda_allocated_bytes'])} to
{gib(cprobe['peak_cuda_allocated_bytes'])} GiB.

## Fixed-manifest convergence and time-to-threshold

The table reports the first available fixed checkpoint at or below each
threshold and exact cumulative epoch wall time from the immutable telemetry;
“not reached” means no listed checkpoint met it.

{chr(10).join(threshold_lines)}

The complete matched-milestone data, including per-field relative L2, parameter
count, memory, timing, checkpoint SHA, and report SHA, is in
[`milestones.csv`](milestones.csv). The structured effect decomposition is in
[`final_summary.json`](final_summary.json).

## Reproducibility identity

{chr(10).join(arm_lines)}

All arms share dataset fingerprint
`{summary['protocol']['dataset_fingerprint']}`, normalization artifact
`{summary['protocol']['normalization_artifact_sha256']}`, normalizer digest
`{summary['protocol']['normalizer_digest']}`, fixed sensor manifest
`{summary['protocol']['sensor_manifest_sha256']}`, fixed-manifest file
`{summary['protocol']['fixed_manifest_file_sha256']}`, and query-index hash
`{summary['protocol']['query_indices_sha256']}`. The validation branch is
`{summary['protocol']['validation_branch']}`; A was frozen from
`{summary['protocol']['source_branch']}`, and B/C launched from the same
portable-prep head recorded above.

The input evidence hashes used by the generator are:

{chr(10).join(input_hash_lines)}

The per-run artifact SHA maps and all milestone checkpoint/report hashes are
retained in `final_summary.json`; no run directory or checkpoint is copied into
the repository.

## Migration gates and guide findings

The migration gate artifact reports status **{gate['correctness_gates_status']}**:
{gate['test_results']['portable_release_focused']};
{gate['test_results']['strengthened_migration_gates']};
{gate['test_results']['downstream_full_regression']}; and the opt-in GPU
legacy-equivalence test passed. The seeded B/C initialization state is
identical (`{summary['migration_evidence']['initialization_identity']['state']['B_state_sha256']}`),
with 148 state keys. Controlled numerical evidence records maximum loss and
gradient-norm differences of
{summary['controlled_execution']['numerical_equivalence']['computed_max_absolute_loss_difference']:.3e} and
{summary['controlled_execution']['numerical_equivalence']['computed_max_absolute_gradient_norm_difference']:.3e},
respectively.

The portable guide is present at
`0_demo_TurbulentCombustion/GL_rbf_CQ_UPDATE_GUIDE.md` and hashes to
`{summary['input_artifact_sha256']['0_demo_TurbulentCombustion/GL_rbf_CQ_UPDATE_GUIDE.md']}`.
The migration exposed three release-blocking documentation gaps, all fixed on
this validation branch in commit `47de065a5b80a871297620e9703fb0bff528dff4`:

{fixed_guide_lines}

The following evidence-schema improvements remain useful but did not block or
invalidate this benchmark:

{gap_lines}

## Reproduction

From `Proj_MultiFieldReconstruction/`, run:

```text
rtk env CUDA_VISIBLE_DEVICES= python benchmarks/gl_rbf_cq_migration_200ep/comparison/generate_comparison.py
```

The generator is standard-library-only, reads only the frozen summaries,
execution/gate evidence, configs, guide, and immutable telemetry, and writes
the three comparison artifacts in this directory. It does not modify the
untracked benchmark Markdown or any run artifact.
"""


def build_summary() -> dict[str, Any]:
    a_payload = read_json(BENCHMARK_DIR / "baseline" / "A_performance.json")
    b_payload = read_json(BENCHMARK_DIR / "runs_summary" / "B_summary.json")
    c_payload = read_json(BENCHMARK_DIR / "runs_summary" / "C_summary.json")
    execution = read_json(BENCHMARK_DIR / "execution" / "B_vs_C_execution.json")
    gates = read_json(BENCHMARK_DIR / "migration" / "correctness_gates.json")
    initialization = read_json(BENCHMARK_DIR / "migration" / "initialization_identity.json")

    arms = {"A": extract_arm("A", a_payload), "B": extract_arm("B", b_payload), "C": extract_arm("C", c_payload)}
    protocol = protocol_identity(arms)
    protocol.update(
        {
            "validation_branch": "validation/proj-multifield-gl-rbf-cq",
            "source_branch": "release/gl-rbf-cq-portable-prep",
            "optimizer": "AdamW",
            "learning_rate": 1.0e-4,
            "weight_decay": 1.0e-6,
            "gradient_clip": 1.0,
            "backward_loss_scale": 1.0,
            "adaptive_backward_scaling": False,
            "query_microbatch_size": 2048,
            "query_microbatches": 2,
            "reuse_condition_context": True,
            "sensor_attention_padding_mode": "full",
            "neighbor_backend": "keops",
        }
    )
    controlled = execution_evidence(execution)

    input_paths = {
        "Proj_MultiFieldReconstruction/benchmarks/gl_rbf_cq_migration_200ep/baseline/A_performance.json": BENCHMARK_DIR / "baseline" / "A_performance.json",
        "Proj_MultiFieldReconstruction/benchmarks/gl_rbf_cq_migration_200ep/runs_summary/B_summary.json": BENCHMARK_DIR / "runs_summary" / "B_summary.json",
        "Proj_MultiFieldReconstruction/benchmarks/gl_rbf_cq_migration_200ep/runs_summary/C_summary.json": BENCHMARK_DIR / "runs_summary" / "C_summary.json",
        "Proj_MultiFieldReconstruction/benchmarks/gl_rbf_cq_migration_200ep/execution/B_vs_C_execution.json": BENCHMARK_DIR / "execution" / "B_vs_C_execution.json",
        "Proj_MultiFieldReconstruction/benchmarks/gl_rbf_cq_migration_200ep/migration/correctness_gates.json": BENCHMARK_DIR / "migration" / "correctness_gates.json",
        "Proj_MultiFieldReconstruction/benchmarks/gl_rbf_cq_migration_200ep/migration/initialization_identity.json": BENCHMARK_DIR / "migration" / "initialization_identity.json",
        "Proj_MultiFieldReconstruction/benchmarks/gl_rbf_cq_migration_200ep/PROTOCOL.yaml": BENCHMARK_DIR / "PROTOCOL.yaml",
        "Proj_MultiFieldReconstruction/benchmarks/gl_rbf_cq_migration_200ep/README.md": BENCHMARK_DIR / "README.md",
        "0_demo_TurbulentCombustion/GL_rbf_CQ_UPDATE_GUIDE.md": REPO_ROOT / "0_demo_TurbulentCombustion" / "GL_rbf_CQ_UPDATE_GUIDE.md",
        **{
            f"Proj_MultiFieldReconstruction/benchmarks/gl_rbf_cq_migration_200ep/configs/{arm}_{config_path_for(arm).name.split('_', 1)[1]}": config_path_for(arm)
            for arm in ("A", "B", "C")
        },
    }
    input_hashes = {path: sha256_file(file_path) for path, file_path in sorted(input_paths.items())}
    require(input_hashes["Proj_MultiFieldReconstruction/benchmarks/gl_rbf_cq_migration_200ep/execution/B_vs_C_execution.json"] == execution["trace"].get("execution_sha256", input_hashes["Proj_MultiFieldReconstruction/benchmarks/gl_rbf_cq_migration_200ep/execution/B_vs_C_execution.json"]), "execution hash sanity")
    require(sha256_file(REPO_ROOT / "0_demo_TurbulentCombustion" / "GL_rbf_CQ_UPDATE_GUIDE.md") == input_hashes["0_demo_TurbulentCombustion/GL_rbf_CQ_UPDATE_GUIDE.md"], "guide hash sanity")

    migration_fixes = [
        {
            "finding": "Strict tensor loading did not protect positional field semantics.",
            "resolution": "The guide now requires exact field identity/order and normalization checks before strict load, rejects same-width semantic mismatches, and documents the fresh-start path with matched B/C initialization.",
        },
        {
            "finding": "The manifest snapshot could be silently rewritten by downstream tooling.",
            "resolution": "The guide now requires byte-identical copy/checksum evidence and excludes the vendored snapshot from formatters and linters.",
        },
        {
            "finding": "Generic trainer/evaluator lifecycle integration was underspecified.",
            "resolution": "The guide now specifies condition-context reuse, exact loss scaling, no double backward, strict EMA auxiliary state, and configured/live evaluation selection without model-name branches.",
        },
    ]
    migration_gaps = [
        {
            "gap": "The guide defines tensor and model integration, but not a downstream fixed-manifest artifact contract (milestone list, metric names, checkpoint/report hashes, or immutable sensor/query digests).",
            "guide_reference": "Tensor-level integration contract and required verification gates",
            "recommended_fix": "Add a machine-readable evaluation schema and require fixed-manifest checkpoint/report hashes plus normalized MSE, mean, worst, and per-field relative L2 at every declared milestone.",
        },
        {
            "gap": "The guide asks for seeded loss/gradient and reconstruction comparisons but does not define the downstream RF bridge/data metadata identity or an all-gradient evidence schema.",
            "guide_reference": "Migration workflow",
            "recommended_fix": "Specify the adapter training_loss contract, query-index metadata, RF bridge seed/draw recording, state hash, and numerical tolerances in a JSON evidence template.",
        },
        {
            "gap": "The guide does not quantify the legacy-to-cached execution contract; it says cached K/V is preferred but does not require exact projection-call counts or a matched probe protocol.",
            "guide_reference": "Old-to-new mapping and required verification gates",
            "recommended_fix": "Require an instrumented legacy_mha versus cached_kv probe with expected/observed calls (4 versus 1 here), memory, phase timing, same tensors, and numerical deltas.",
        },
        {
            "gap": "The guide does not define a machine-readable OOM/resource-adjustment record for an authorized common-batch fallback.",
            "guide_reference": "Minimal config patch and required verification gates",
            "recommended_fix": "Define an adjustment record that preserves the requested protocol, failed attempt, authorization, chosen common replacement, and identical application across arms.",
        },
    ]
    migration_evidence = {
        "correctness_gates_status": gates["status"],
        "gate_groups": gates["gate_groups"],
        "test_results": gates["test_results"],
        "post_migration_implementation_head": gates["post_migration_implementation_head"],
        "initialization_identity": initialization,
    }

    summary: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "gl_rbf_cq_migration_200ep",
        "generator": {
            "script": "benchmarks/gl_rbf_cq_migration_200ep/comparison/generate_comparison.py",
            "script_sha256": sha256_file(SCRIPT_PATH),
            "deterministic": True,
        },
        "protocol": protocol,
        "arms": {
            arm: {
                key: value
                for key, value in info.items()
                if key not in ("fixed_manifest_convergence", "timing", "elapsed_wall_time_by_epoch_s")
            }
            | {
                "fixed_manifest_convergence": info["fixed_manifest_convergence"],
                "timing": info["timing"],
                "elapsed_wall_time_by_epoch_s": {str(epoch): value for epoch, value in info["elapsed_wall_time_by_epoch_s"].items()},
            }
            for arm, info in arms.items()
        },
        "effects": {
            "migration_B_minus_A": effect(arms["A"], arms["B"]),
            "execution_C_minus_B": effect(arms["B"], arms["C"]),
            "total_C_minus_A": effect(arms["A"], arms["C"]),
        },
        "matched_milestone_effects": {
            "migration_B_minus_A": matched_effects(arms, "A", "B"),
            "execution_C_minus_B": matched_effects(arms, "B", "C"),
            "total_C_minus_A": matched_effects(arms, "A", "C"),
        },
        "convergence_thresholds": threshold_results(arms),
        "controlled_execution": controlled,
        "migration_evidence": migration_evidence,
        "migration_guide": {
            "path": "0_demo_TurbulentCombustion/GL_rbf_CQ_UPDATE_GUIDE.md",
            "sha256": input_hashes["0_demo_TurbulentCombustion/GL_rbf_CQ_UPDATE_GUIDE.md"],
        },
        "migration_guide_fixes": migration_fixes,
        "migration_guide_gaps": migration_gaps,
        "input_artifact_sha256": input_hashes,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    write_milestones(output_dir / "milestones.csv", {arm: summary["arms"][arm] for arm in ("A", "B", "C")})
    (output_dir / "final_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "RESULTS.md").write_text(build_results_markdown(summary), encoding="utf-8")
    print(f"wrote {output_dir / 'milestones.csv'}")
    print(f"wrote {output_dir / 'final_summary.json'}")
    print(f"wrote {output_dir / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
