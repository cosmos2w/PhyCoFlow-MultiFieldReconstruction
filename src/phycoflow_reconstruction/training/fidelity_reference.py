"""Strict, opt-in SOURCE-reference sharing for matched post-training arms.

The reference is an immutable calibration artifact, not a new fidelity
estimator. Historical checkpoints and the default calibration path retain
their original semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from .fidelity_budget import FidelityBudget

SCHEMA = "shared_fidelity_reference_v1"
ESTIMATOR = "primary_draw_pooled_valid_equal_node_model_mse_v1"


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: Any) -> Any:
    """Normalize tuples and other JSON-compatible values to their persisted form."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _budget_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "metric": settings.get("metric", "endpoint_mse_model_units"),
        "relative_increase": float(settings.get("relative_increase", 0.05)),
        "absolute_allowance": float(settings.get("absolute_allowance", 0.0)),
        "denominator_floor": float(settings.get("denominator_floor", 1e-8)),
        "calibration_batches": int(settings.get("calibration_batches", 4)),
        "calibration_draws": int(settings.get("calibration_draws", 2)),
        "calibration_seed": int(settings.get("calibration_seed", 700103)),
    }


def _baseline_config_anchor(config: Mapping[str, Any]) -> dict[str, Any]:
    """Bind only settings that determine SOURCE calibration and frozen families.

    Output, checkpointing, evaluation, plotting, optimizer settings and horizon
    are not calibration inputs. The separately persisted run configuration still
    binds those training choices on resume. Candidate extensions are excluded
    from the family configs used for this shared baseline.
    """
    coherence = config["coherence"]
    families = deepcopy(coherence.get("families", {}))
    state = families.get("global_distribution")
    if isinstance(state, dict) and state.get("definition") == "conditional_quantile_fair_v1":
        # The estimator and frozen representation are those of the v3 control;
        # only this exact declared J2 score substitution shares its SOURCE C0.
        # All other fields remain in the binding and therefore reject tampering.
        if state.get("base_definition") != "marginal_mixed_weighted_v3" or state.get("conditional_draws") != 2:
            raise ValueError("shared fidelity reference requires the exact v3 State base and J2 policy")
        state["definition"] = state.pop("base_definition")
        state.pop("conditional_draws")
    scale = families.get("multiscale_spectrum")
    if isinstance(scale, dict):
        scale.pop("gram_extension", None)
        scale.pop("relation_extension", None)
    optimization = config["optimization"]
    runtime = config.get("runtime", {})
    observations = deepcopy(config["observations"])
    observations["seed"] = int(observations.get("seed", runtime.get("seed", 42)))
    compute = deepcopy(coherence["compute_budget"])
    compute["query_seed"] = int(compute.get("query_seed", observations["seed"] + 100_003))
    balance = deepcopy(coherence.get("family_balance", {}))
    balance.pop("gradient_diagnostics_every_epochs", None)
    if balance.get("mode", "none") == "none":
        balance = {"mode": "none"}
    elif balance.get("mode") == "initial_grad_norm":
        balance["seed"] = int(balance.get("seed", runtime.get("seed", 42) + 700_001))
    model_mode = optimization.get("model_mode", "eval")
    runtime_anchor = {
        "device": runtime.get("device", "cpu"),
        "deterministic": runtime.get("deterministic", True),
        "data_strategy": runtime.get("data_strategy", "auto"),
        "num_workers": runtime.get("num_workers", 0),
    }
    if runtime_anchor["data_strategy"] == "auto":
        runtime_anchor["vram_dataset_threshold_gb"] = runtime.get("vram_dataset_threshold_gb", 20.0)
    # Train-mode dropout and batch-normalization behavior can consume global RNG.
    if model_mode == "train":
        runtime_anchor["train_mode_seed"] = int(runtime.get("seed", 42))
    return _canonical({
        "dataset": config["dataset"], "model": config["model"],
        "observations": observations,
        "observation_consistency": config["observation_consistency"],
        "rollout": config["rollout"],
        "coherence": {"families": families, "compute_budget": compute, "family_balance": balance},
        "calibration_batch_size": int(optimization.get("batch_size", 1)),
        "model_mode": model_mode,
        "rollout_execution": optimization.get("rollout_execution", {}),
        "runtime_calibration": runtime_anchor,
    })


def shared_reference_binding(
    config: Mapping[str, Any],
    *,
    field_names: Sequence[str],
    normalizer_sha256: str,
    source_hashes: Mapping[str, str],
    dataset_fingerprint: str,
    train_selection: Mapping[str, Any],
    family_scales: Mapping[str, float],
) -> dict[str, Any]:
    """Construct the exact SOURCE, estimator, support, endpoint and recipe binding."""
    names = list(field_names)
    if not names or len(set(names)) != len(names) or any(not isinstance(x, str) or not x for x in names):
        raise ValueError("shared fidelity reference requires distinct ordered field names")
    _sha256(normalizer_sha256, "normalizer_sha256")
    _sha256(dataset_fingerprint, "dataset_fingerprint")
    if "checkpoint" not in source_hashes:
        raise ValueError("shared fidelity reference requires a SOURCE checkpoint hash")
    source_hashes = {name: _sha256(value, f"source_hashes.{name}") for name, value in source_hashes.items()}
    scales = {name: float(value) for name, value in family_scales.items()}
    if not scales or any(not math.isfinite(value) or value <= 0 for value in scales.values()):
        raise ValueError("shared fidelity reference requires positive finite family scales")
    selection = _canonical(train_selection)
    selection_hash = _digest_bytes(json.dumps(selection, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    budget = _budget_settings(config["optimization"].get("multitask", {}).get("fidelity_budget", {}))
    if budget["metric"] != "endpoint_mse_model_units":
        raise ValueError("shared fidelity reference requires the legacy model-unit endpoint MSE")
    return _canonical({
        "estimator": ESTIMATOR,
        "field_names": names,
        "normalizer_sha256": normalizer_sha256,
        "source_hashes": source_hashes,
        "dataset_fingerprint": dataset_fingerprint,
        "train_selection_sha256": selection_hash,
        "family_scales": scales,
        "fidelity_budget": budget,
        "baseline_config_anchor": _baseline_config_anchor(config),
    })


def _validate_reference(reference: Mapping[str, Any], binding: Mapping[str, Any]) -> FidelityBudget:
    budget = FidelityBudget.from_state_dict(reference)
    settings = binding["fidelity_budget"]
    if list(budget.field_names) != binding["field_names"]:
        raise ValueError("shared fidelity reference field order mismatch")
    if (budget.relative_increase != settings["relative_increase"]
            or budget.absolute_allowance != settings["absolute_allowance"]
            or budget.denominator_floor != settings["denominator_floor"]):
        raise ValueError("shared fidelity reference bounds differ from active budget settings")
    return budget


def create_shared_reference(
    report_path: str | Path,
    report_sha256: str,
    baseline_config: Mapping[str, Any],
    **binding_arguments: Any,
) -> dict[str, Any]:
    """Bind an existing SOURCE calibration report to a verified baseline recipe.

    The caller must independently verify that the original calibration report
    was actually produced by this SOURCE and baseline recipe. Its byte hash is
    checked here and recorded; binding metadata cannot establish provenance by
    itself. Do not call with an augmented Scale config.
    """
    scale = baseline_config["coherence"].get("families", {}).get("multiscale_spectrum", {})
    if "gram_extension" in scale or "relation_extension" in scale:
        raise ValueError("shared SOURCE reference must be created from the unaugmented baseline")
    expected = _sha256(report_sha256, "report_sha256")
    raw = Path(report_path).read_bytes()
    if _digest_bytes(raw) != expected:
        raise ValueError("SOURCE fidelity calibration report hash mismatch")
    report = json.loads(raw)
    binding = shared_reference_binding(baseline_config, **binding_arguments)
    if _budget_settings(report["settings"]) != binding["fidelity_budget"]:
        raise ValueError("SOURCE calibration settings differ from baseline config")
    reference = _validate_reference(report["reference"], binding).state_dict()
    return {"schema": SCHEMA, "binding": binding, "reference": reference,
            "origin": {"calibration_report_sha256": expected}}


def serialize_shared_reference(artifact: Mapping[str, Any]) -> bytes:
    """Return deterministic artifact bytes whose SHA-256 is pinned by the run."""
    return (json.dumps(artifact, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def load_shared_reference(
    path: str | Path, expected_sha256: str, binding: Mapping[str, Any]
) -> tuple[FidelityBudget, bytes]:
    """Load exact bytes and reject any mismatch in their complete bound contract."""
    expected = _sha256(expected_sha256, "reference_sha256")
    raw = Path(path).read_bytes()
    if _digest_bytes(raw) != expected:
        raise ValueError("shared fidelity reference artifact hash mismatch")
    artifact = json.loads(raw)
    if artifact.get("schema") != SCHEMA:
        raise ValueError("unsupported shared fidelity reference artifact schema")
    expected_binding = _canonical(binding)
    actual_binding = artifact.get("binding")
    if not isinstance(actual_binding, dict):
        raise TypeError("shared fidelity reference binding is not a mapping")
    if actual_binding != expected_binding:
        changed = sorted(k for k in set(actual_binding) | set(expected_binding)
                         if actual_binding.get(k) != expected_binding.get(k))
        raise ValueError(f"shared fidelity reference binding mismatch: {', '.join(changed)}")
    _sha256(artifact.get("origin", {}).get("calibration_report_sha256"), "origin.calibration_report_sha256")
    reference = _validate_reference(artifact["reference"], expected_binding)
    return reference, raw


def validate_shared_reference_recovery(
    reference: FidelityBudget,
    expected_sha256: str,
    checkpoint: Mapping[str, Any],
    manifest: Mapping[str, Any],
    local_path: str | Path,
) -> None:
    """Check the run-local artifact and checkpoint before loading policy state."""
    expected = _sha256(expected_sha256, "reference_sha256")
    local = Path(local_path)
    saved_policy = checkpoint.get("external_policy", {})
    if (
        checkpoint.get("shared_fidelity_reference_sha256") != expected
        or manifest.get("shared_fidelity_reference_sha256") != expected
        or not local.is_file()
        or _digest_bytes(local.read_bytes()) != expected
        or not isinstance(saved_policy, Mapping)
        or saved_policy.get("reference") != reference.state_dict()
    ):
        raise ValueError("resume shared fidelity reference artifact or checkpoint mismatch")
