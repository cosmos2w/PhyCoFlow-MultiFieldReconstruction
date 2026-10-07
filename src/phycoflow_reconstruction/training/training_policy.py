"""Resolved, source-bound receipts for the matched scalar training path."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _descriptor_versions(families: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    versions: dict[str, dict[str, str]] = {}
    for name, family in families.items():
        item: dict[str, str] = {"version": str(getattr(family, "version", "unknown"))}
        definition = getattr(family, "definition", None)
        if definition is not None:
            item["definition"] = str(definition)
        strategy = getattr(family, "strategy", None)
        if strategy is not None:
            item["strategy"] = str(strategy)
        spatial = getattr(family, "spatial_objective", None)
        aggregation = getattr(spatial, "aggregation", None)
        if aggregation is not None:
            item["aggregation"] = str(aggregation)
        versions[name] = item
    return versions


def _native_objective_label(model) -> str:
    if hasattr(model, "sample_source") and hasattr(model, "velocity"):
        return "native_RF_loss"
    return "model_native_training_loss"


def resolve_training_policy(
    config: Mapping[str, Any],
    *,
    model,
    families: Mapping[str, Any],
    family_scales: Mapping[str, float],
    calibration: Mapping[str, Any],
    calibration_sha256: str,
    source_hashes: Mapping[str, str],
    source_metadata: Mapping[str, Any],
    normalizer_digest: str,
    parameter_retention=None,
) -> dict[str, Any]:
    """Describe the matched scalar branch and the exact fixed objective it uses."""
    optimization = config["optimization"]
    runtime = config["runtime"]
    objectives = config["objectives"]
    coherence_settings = objectives["coherence"]
    data_settings = objectives["data_retention"]
    if (
        runtime.get("random_stream_policy") != "matched_native_v1"
        or runtime.get("execution_mode") != "r4_exact"
        or optimization.get("gradient_balance", "weighted_sum") != "weighted_sum"
        or optimization.get("update_policy", "legacy") != "legacy"
    ):
        raise ValueError("resolved scalar receipt requires matched_native_v1 r4_exact weighted_sum/legacy dispatch")

    native_weight = float(data_settings.get("weight", 0.0)) if data_settings.get("enabled", True) else 0.0
    configured_beta = float(coherence_settings.get("weight", 0.0))
    coherence_enabled = bool(coherence_settings.get("enabled", True))
    if not math.isfinite(native_weight) or native_weight != 0.1:
        raise ValueError("matched scalar receipt requires the fixed 0.1 native coefficient")
    if not math.isfinite(configured_beta) or configured_beta < 0:
        raise ValueError("matched scalar receipt requires a finite nonnegative coherence coefficient")
    beta = configured_beta if coherence_enabled else 0.0

    if (parameter_retention is not None) != bool(
        objectives.get("parameter_retention", {}).get("enabled", False)
    ):
        raise ValueError("resolved scalar receipt disagrees with parameter-retention dispatch")
    scales = {name: float(family_scales[name]) for name in families}
    if any(not math.isfinite(value) or value <= 0 for value in scales.values()):
        raise ValueError("resolved scalar receipt requires positive finite calibrated family scales")
    family_weights = {
        name: float(getattr(family, "family_weight", 1.0)) for name, family in families.items()
    }
    retention = {
        "active": parameter_retention is not None,
        "coefficient": None if parameter_retention is None else float(parameter_retention.coefficient),
        "definition": None if parameter_retention is None else "source_parameter_l2_v1",
    }
    native_label = _native_objective_label(model)
    source_identity = source_metadata.get("initial_trainable_parameter_identity")
    if not isinstance(source_identity, Mapping) or not source_identity.get("tensor_sha256"):
        raise ValueError("matched scalar receipt is missing the initial SOURCE trainable-weight identity")
    versions = _descriptor_versions(families)
    calibrated_name = (
        "calibrated_ABC"
        if set(families) == {"global_distribution", "cross_spectrum", "topology"}
        else "calibrated_coherence"
    )
    if parameter_retention is None and calibrated_name == "calibrated_ABC" and native_label == "native_RF_loss":
        summary = (
            "Fixed weighted sum: 0.1 * native_RF_loss + beta * calibrated_ABC; "
            "no ConFIG; no fidelity controller; no parameter-retention penalty."
        )
    elif parameter_retention is None:
        summary = (
            f"Fixed weighted sum: {native_weight:g} * {native_label} + {beta:g} * {calibrated_name}; "
            "no ConFIG; no fidelity controller; no parameter-retention penalty."
        )
    else:
        summary = (
            f"Fixed weighted sum: {native_weight:g} * {native_label} + {beta:g} * {calibrated_name}; "
            f"no ConFIG; no fidelity controller; source-parameter retention active "
            f"(lambda_SP={parameter_retention.coefficient:g})."
        )
    resolved_summary = (
        f"Resolved coefficients: native={native_weight:g}, beta={beta:g}; "
        f"calibrated family scales={json.dumps(scales, sort_keys=True)}."
    )
    return {
        "schema_version": "matched_scalar_training_policy_v1",
        "summary": summary,
        "resolved_summary": resolved_summary,
        "update": {
            "implementation": "training.parameter_retention.scalar_update",
            "random_stream_policy": runtime["random_stream_policy"],
            "execution_mode": runtime["execution_mode"],
            "gradient_balance": optimization.get("gradient_balance", "weighted_sum"),
            "update_policy": optimization.get("update_policy", "legacy"),
            "conFIG_active": False,
            "fidelity_controller_active": False,
        },
        "objective": {
            "equation": f"{native_weight:g} * {native_label} + {beta:g} * {calibrated_name}"
            + (" + lambda_SP * Omega" if parameter_retention is not None else ""),
            "native_objective": native_label,
            "native_data_coefficient": native_weight,
            "configured_coherence_coefficient": configured_beta,
            "effective_coherence_coefficient": beta,
            "coherence_enabled": coherence_enabled,
            "family_scales": scales,
            "family_weights": family_weights,
            "family_balance_mode": calibration.get("mode", "none"),
            "family_calibration_sha256": calibration_sha256,
            "parameter_retention": retention,
        },
        "sampler": {
            "endpoint_rollout": dict(config.get("rollout", {})),
            "observation_consistency": dict(config.get("observation_consistency", {})),
            "coherence_query_policy": config.get("coherence", {}).get("compute_budget", {}).get(
                "query_policy", "random_per_sample"
            ),
            "coherence_batch_size": int(
                config.get("coherence", {}).get("compute_budget", {}).get("batch_size", 0)
            ),
            "coherence_point_count": int(
                config.get("coherence", {}).get("compute_budget", {}).get("point_count", 0)
            ),
            "coherence_every_n_steps": int(
                config.get("coherence", {}).get("schedule", {}).get("every_n_steps", 1)
            ),
        },
        "descriptor_versions": versions,
        "weights": {
            "source_parameter_identity": dict(source_identity),
            "source_checkpoint_sha256": source_hashes.get("checkpoint"),
            "selection": "live",
            "training_selection": "live",
            "evaluation_selection": "ema" if config.get("model", {}).get("model_ema_eval", False) else "live",
            "model_ema_eval": bool(config.get("model", {}).get("model_ema_eval", False)),
        },
        "source_identity": {
            "source_run": str(config["source_run"]),
            "source_checkpoint": str(config["source_checkpoint"]),
            "source_hashes": dict(source_hashes),
            "initial_trainable_parameters": dict(source_identity),
            "normalizer_digest": str(normalizer_digest),
            "source_kind": source_metadata.get("kind"),
        },
        "initialization": {
            "kind": "independent_initialization_from_declared_SOURCE",
            "source_checkpoint_sha256": source_hashes.get("checkpoint"),
            "source_trainable_tensor_sha256": source_identity["tensor_sha256"],
        },
    }


def persist_training_policy(
    store,
    policy: Mapping[str, Any],
    *,
    is_resume: bool,
    start_step: int,
    resume_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Write the resolved receipt and verify its immutable contract on recovery."""
    path = store.run_dir / "training_policy.json"
    resolved = dict(policy)
    prior = None
    if path.is_file():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior.get("policy_sha256") != _canonical_sha256(
            {key: value for key, value in prior.items() if key not in {"execution", "policy_sha256"}}
        ):
            raise ValueError("saved training policy receipt hash is invalid")
        resolved["initialization"] = prior["initialization"]
        prior_contract = {key: value for key, value in prior.items() if key not in {"execution", "policy_sha256"}}
        new_contract = {key: value for key, value in resolved.items() if key not in {"execution", "policy_sha256"}}
        if prior_contract != new_contract:
            raise ValueError("resume training policy differs from its original resolved receipt")
    elif is_resume:
        # Older matched-stream runs predate this receipt. Their resolved config,
        # SOURCE hashes and immutable calibration still provide the policy identity.
        resolved["initialization"] = {
            **dict(resolved["initialization"]),
            "receipt_origin": "reconstructed_on_own_resume",
        }

    resolved["execution"] = {
        "kind": "own_run_resume" if is_resume else "independent_source_initialization",
        "start_global_step": int(start_step),
        "resume_checkpoint_sha256": resume_checkpoint_sha256,
    }
    contract = {key: value for key, value in resolved.items() if key not in {"execution", "policy_sha256"}}
    resolved["policy_sha256"] = _canonical_sha256(contract)
    store.write_json("training_policy.json", resolved)
    store.update_manifest(training_policy_sha256=resolved["policy_sha256"])
    print(resolved["summary"], flush=True)
    print(resolved["resolved_summary"], flush=True)
    return resolved
