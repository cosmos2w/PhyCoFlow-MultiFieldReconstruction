"""Resume-safe runtime guard for measured execution plans.

This module binds a selected plan to its frozen task and TRAIN-only profile,
provides deterministic decisions for accepted-update cursors, and switches to
full execution when monitoring detects sustained risk or excess estimator
noise. It does not read model or dataset state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from .contracts import (
    CostProfile,
    ExecutionPlan,
    ExecutionState,
    TaskContract,
    full_audit_due,
    policy_refresh_due,
)

_SETTINGS_KEYS = {"enabled", "task", "profile", "plan", "seed"}
_SUSTAINED_VIOLATION_EPOCHS = 2
_MAX_MONITOR_OVERHEAD_RATIO = 0.05
_IMPLEMENTATION_PATHS = (
    "training/execution_strategy/__init__.py",
    "training/execution_strategy/audit.py",
    "training/execution_strategy/contracts.py",
    "training/execution_strategy/planner.py",
    "training/execution_strategy/profiling.py",
    "training/execution_strategy/providers.py",
    "training/execution_strategy/runtime.py",
    "training/fast_lagrangian.py",
    "training/periodic_config.py",
    "training/fidelity_budget.py",
    "training/post_training.py",
    "training/run_store.py",
    "coherence/families/topology/filtration_sampling.py",
    "evaluation/fixed_groups.py",
    "evaluation/metrics.py",
    "contracts.py",
    "data/manifest.py",
    "data/training_batches.py",
    "training/common.py",
    "training/update_budget.py",
    "config/validate.py",
)


def scientific_config_hash(config: Mapping[str, Any]) -> str:
    """Hash scientific configuration while excluding execution/run labels.

    The execution plan is an implementation choice. The experiment name and
    resolved source-origin bookkeeping also do not define the scientific
    objective, while model, horizon, rollout, evaluation and checkpoint
    settings remain bound.
    """

    if not isinstance(config, Mapping):
        raise TypeError("scientific config must be a mapping")
    normalized = copy.deepcopy(dict(config))
    optimization = normalized.get("optimization")
    if isinstance(optimization, Mapping):
        optimization = dict(optimization)
        optimization.pop("execution_strategy", None)
        optimization.pop("fast_lagrangian", None)
        normalized["optimization"] = optimization
    output = normalized.get("output")
    if isinstance(output, Mapping):
        output = dict(output)
        output.pop("experiment_name", None)
        normalized["output"] = output
    source = normalized.get("source")
    if isinstance(source, Mapping):
        source = dict(source)
        source.pop("config_origins", None)
        normalized["source"] = source
    # Some resolved config versions expose origins at the root. They are the
    # same provenance bookkeeping and have no numerical effect.
    normalized.pop("config_origins", None)
    try:
        payload = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("scientific config must contain finite JSON-compatible values") from error
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _as_contract(value: Any, cls: type, name: str) -> Any:
    if isinstance(value, cls):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"execution_strategy.{name} must be a mapping or {cls.__name__}")
    return cls.from_mapping(value)


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_nonnegative(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite non-negative scalar") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative scalar")
    return result


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _profile_identity(profile: CostProfile) -> str:
    return hashlib.sha256(_canonical(profile.to_dict()).encode("utf-8")).hexdigest()


def _validate_dense_scalar_config(config: Mapping[str, Any], task: TaskContract) -> None:
    optimization = config.get("optimization", {})
    if not isinstance(optimization, Mapping):
        raise TypeError("optimization must be a mapping")
    epochs = optimization.get("epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs != task.max_epochs:
        raise ValueError("optimization.epochs must match TaskContract.max_epochs")
    for key in ("fast_lagrangian", "lpc_ordered"):
        if key in optimization and optimization[key] is not None:
            raise ValueError(f"execution_strategy cannot coexist with optimization.{key}")

    multitask = optimization.get("multitask", {})
    if not isinstance(multitask, Mapping):
        raise TypeError("optimization.multitask must be a mapping")
    if "coherence_aggregation" in multitask:
        raise ValueError("execution_strategy requires scalar additive family aggregation")
    scalarization = multitask.get("family_scalarization", {})
    if not isinstance(scalarization, Mapping) or scalarization.get("method") != "fixed_sum":
        raise ValueError("execution_strategy requires family_scalarization.method=fixed_sum")
    if multitask.get("outer_aggregation") not in (None, {}):
        raise ValueError("execution_strategy requires no nonlinear outer aggregation")

    external = multitask.get("external_policy")
    if not isinstance(external, Mapping) or external.get("method") != "lagrangian":
        raise ValueError("execution_strategy requires dense external Lagrangian fidelity")
    if external.get("vector_reduction", "sum") != "sum":
        raise ValueError("execution_strategy requires scalar-sum Lagrangian fidelity")
    if external.get("objective_scale", "source_mean") != "source_mean":
        raise ValueError("execution_strategy requires source-mean Lagrangian scaling")
    lagrangian = external.get("lagrangian", {})
    if not isinstance(lagrangian, Mapping):
        raise TypeError("external_policy.lagrangian must be a mapping")
    penalty_interval = lagrangian.get("penalty_interval_updates", 1)
    if (
        isinstance(penalty_interval, bool)
        or not isinstance(penalty_interval, int)
        or penalty_interval != 1
    ):
        raise ValueError("execution_strategy requires dense per-update field fidelity")
    fidelity = multitask.get("fidelity_budget")
    if not isinstance(fidelity, Mapping):
        raise TypeError("execution_strategy requires a fixed fidelity_budget mapping")

    coherence = config.get("coherence", {})
    if not isinstance(coherence, Mapping):
        raise TypeError("coherence must be a mapping")
    family_balance = coherence.get("family_balance", {})
    if not isinstance(family_balance, Mapping):
        raise TypeError("coherence.family_balance must be a mapping")
    if family_balance.get("mode", "none") != "none":
        raise ValueError(
            "execution_strategy v1 requires coherence.family_balance.mode=none; "
            "profiled and runtime family scales must remain identical"
        )
    schedule = coherence.get("schedule", {})
    if not isinstance(schedule, Mapping):
        raise TypeError("coherence.schedule must be a mapping")
    start_epoch = schedule.get("start_epoch", 1)
    every_n_steps = schedule.get("every_n_steps", 1)
    warmup_epochs = schedule.get("weight_warmup_epochs", 0)
    if (
        isinstance(start_epoch, bool)
        or not isinstance(start_epoch, int)
        or start_epoch != 1
        or isinstance(every_n_steps, bool)
        or not isinstance(every_n_steps, int)
        or every_n_steps != 1
        or isinstance(warmup_epochs, bool)
        or not isinstance(warmup_epochs, int)
        or warmup_epochs != 0
        or schedule.get("interval_rescale", False) is not False
    ):
        raise ValueError("execution_strategy requires the unchanged every-step scalar objective")
    if task.objective_rule != "scalar_additive":
        raise ValueError("execution_strategy requires TaskContract.objective_rule=scalar_additive")
    if task.calibration_status != "bound":
        raise ValueError("execution_strategy requires bound SOURCE calibration")


def _validate_plan_profile(
    task: TaskContract,
    profile: CostProfile,
    plan: ExecutionPlan,
    config: Mapping[str, Any],
) -> None:
    if task.objective_rule != "scalar_additive":
        raise ValueError("execution_strategy accepts only scalar-additive task contracts")
    if task.calibration_status != "bound":
        raise ValueError("execution_strategy cannot use pending SOURCE calibration")
    if profile.task_hash != task.task_hash or plan.task_hash != task.task_hash:
        raise ValueError("task, profile, and selected plan hashes must match")
    if profile.max_epochs != task.max_epochs:
        raise ValueError("profile and task maximum horizons differ")
    _integer(profile.profiled_epoch, name="profile.profiled_epoch")
    _integer(profile.max_epochs, name="profile.max_epochs", minimum=1)
    if profile.profile_split.upper() != "TRAIN" or profile.noise_split.upper() != "TRAIN":
        raise ValueError("execution profile and noise evidence must be TRAIN-only")
    _integer(plan.refresh_interval_epochs, name="plan.refresh_interval_epochs", minimum=10)
    _integer(plan.audit_interval_epochs, name="plan.audit_interval_epochs", minimum=20)
    if plan.refresh_interval_epochs != 10:
        raise ValueError(
            "execution probes must refresh every 10 epochs and audit no sooner than 20"
        )
    measurements = {measurement.plan_id: measurement for measurement in profile.measurements}
    if plan.plan_id not in measurements:
        raise ValueError("selected plan_id is absent from the cost profile")
    plan.validate_for_task(task)
    selected = measurements[plan.plan_id]
    if selected.measured_split.upper() != "TRAIN":
        raise ValueError("selected plan measurement must use TRAIN")
    if plan.no_action:
        if (
            plan.plan_id != "full"
            or selected.actions
            or selected.family_probabilities
            or selected.within_family
        ):
            raise ValueError("full recovery plan must be the measured no-action candidate")
    else:
        if selected.estimator_kind in {"approximate_direction", "unsupported"}:
            raise ValueError("selected plan is not an exact or unbiased scalar estimator")
        if not selected.support_validated or not selected.support_nonempty:
            raise ValueError("selected plan lacks validated non-empty estimator support")
        if (
            selected.family_probabilities or selected.within_family
        ) and selected.estimator_kind != "unbiased_scalar_estimator":
            raise ValueError("sampling actions require an unbiased scalar estimator contract")
        if tuple(plan.selected_actions) != tuple(selected.actions):
            raise ValueError("selected plan actions disagree with its measured candidate")
        if dict(plan.family_probabilities) != dict(selected.family_probabilities):
            raise ValueError("selected plan family probabilities disagree with its measurement")
        if _canonical(plan.within_family) != _canonical(selected.within_family):
            raise ValueError("selected plan within-family estimators disagree with measurement")
        normalized_plan_support = [
            (float(mass), list(names)) for mass, names in plan.family_support
        ]
        normalized_profile_support = [
            (float(mass), list(names)) for mass, names in selected.family_support
        ]
        if normalized_plan_support != normalized_profile_support:
            raise ValueError("selected plan joint support disagrees with its measurement")
        if not math.isclose(
            float(plan.extra_variance),
            float(selected.extra_variance),
            rel_tol=1e-8,
            abs_tol=1e-12,
        ):
            raise ValueError("selected plan variance disagrees with its TRAIN measurement")
        if plan.extra_variance > plan.variance_ceiling + 1e-12:
            raise ValueError("selected plan variance exceeds its declared ceiling")
        if plan.variance_ceiling > profile.intrinsic_noise_variance + 1e-12:
            raise ValueError(
                "selected plan variance ceiling exceeds the TRAIN intrinsic-noise budget"
            )
        if selected.max_joint_importance_multiplier > 4.0 + 1e-12:
            raise ValueError("selected nested importance multiplier exceeds four")
        if plan.max_joint_importance_multiplier > 4.0 + 1e-12:
            raise ValueError("selected nested importance multiplier exceeds four")
        if not math.isclose(
            float(plan.max_joint_importance_multiplier),
            float(selected.max_joint_importance_multiplier),
            rel_tol=1e-8,
            abs_tol=1e-10,
        ):
            raise ValueError("selected plan importance multiplier disagrees with its measurement")

        active_contract_families = {
            str(name) for name, weight in task.component_weights if float(weight) > 0
        }
        if not active_contract_families:
            raise ValueError("execution task must declare at least one active component family")
        sampling_families = set(plan.family_probabilities) | set(plan.within_family)
        declared_aliases: set[str] = set()
        alias_mapping = task.metadata.get("family_name_map", {})
        if isinstance(alias_mapping, Mapping):
            declared_aliases = {str(value) for value in alias_mapping.values()}
        config_families = config.get("coherence", {}).get("families", {})
        config_active = (
            {
                str(name)
                for name, settings in config_families.items()
                if isinstance(settings, Mapping)
                and settings.get("enabled", True) is True
                and float(settings.get("weight", 1.0)) > 0
            }
            if isinstance(config_families, Mapping)
            else set()
        )
        known_families = active_contract_families | declared_aliases | config_active
        if not sampling_families.issubset(known_families):
            unknown = sorted(sampling_families - known_families)
            raise ValueError(f"execution plan samples unknown/inactive families: {unknown}")
        if plan.family_support:
            for _, included in plan.family_support:
                if not included or not set(included).issubset(set(plan.family_probabilities)):
                    raise ValueError("joint family support has an empty or unknown outcome")
        providers = set(task.provider_ids)
        for family, specification in plan.within_family.items():
            if family not in known_families:
                raise ValueError(f"within-family estimator targets an inactive family {family!r}")
            if str(specification.get("provider_id")) not in providers:
                raise ValueError(f"within-family provider for {family!r} is not task-bound")

    metadata_hash = task.metadata.get("scientific_config_hash")
    if metadata_hash is not None and str(metadata_hash) != scientific_config_hash(config):
        raise ValueError("TaskContract scientific_config_hash does not match normalized config")
    _validate_dense_scalar_config(config, task)


def _validate_saved_current_plan(
    current_plan: ExecutionPlan,
    configured_plan: ExecutionPlan,
    profile: CostProfile,
    state: ExecutionState,
    envelope: Mapping[str, Any],
) -> None:
    """Accept the configured candidate or the one legal runtime transition: full fallback."""

    if _canonical(current_plan.to_dict()) == _canonical(configured_plan.to_dict()):
        return
    if current_plan.plan_id != "full" or configured_plan.plan_id == "full":
        raise ValueError("serialized current plan differs from the configured measured candidate")

    reason = envelope.get("fallback_reason")
    recovery_until = envelope.get("full_recovery_until_epoch")
    intervals = envelope.get("full_recovery_intervals", ())
    if not isinstance(reason, str) or not reason or recovery_until is None:
        raise ValueError("serialized full plan is missing its runtime fallback binding")
    matching_intervals = [
        item
        for item in intervals
        if isinstance(item, Mapping) and item.get("end_epoch") == recovery_until
    ]
    if not matching_intervals:
        raise ValueError("serialized full plan is missing its fallback recovery interval")
    epoch = matching_intervals[-1].get("start_epoch")
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or state.last_refresh_epoch is None
        or state.last_refresh_epoch < epoch
        or recovery_until != epoch + configured_plan.refresh_interval_epochs
    ):
        raise ValueError("serialized full plan has an inconsistent recovery interval")

    full = next(item for item in profile.measurements if item.plan_id == "full")
    projected = float(full.upper_epoch_cost_s) * max(0, profile.max_epochs - epoch)
    expected_rejections = dict(configured_plan.rejection_reasons)
    expected_rejections["runtime_fallback"] = reason
    valid_fallback = (
        current_plan.task_hash == configured_plan.task_hash
        and current_plan.no_action
        and not current_plan.selected_actions
        and not current_plan.family_probabilities
        and not current_plan.within_family
        and not current_plan.family_support
        and current_plan.accepted_cursor <= state.accepted_cursor
        and current_plan.refresh_interval_epochs == configured_plan.refresh_interval_epochs
        and current_plan.audit_interval_epochs == configured_plan.audit_interval_epochs
        and math.isclose(current_plan.predicted_total_s, projected, rel_tol=1e-12, abs_tol=1e-12)
        and math.isclose(
            current_plan.predicted_full_total_s, projected, rel_tol=1e-12, abs_tol=1e-12
        )
        and current_plan.cost_savings_ratio == 0.0
        and current_plan.extra_variance == 0.0
        and current_plan.variance_ceiling == configured_plan.variance_ceiling
        and current_plan.max_joint_importance_multiplier == 1.0
        and current_plan.selected_reason
        == f"Fell back to measured full execution at epoch {epoch}: {reason}."
        and dict(current_plan.rejection_reasons) == expected_rejections
    )
    if not valid_fallback:
        raise ValueError("serialized full plan does not match a clean measured runtime fallback")


def validate_execution_strategy(config: Mapping[str, Any]) -> bool:
    """Validate an enabled strategy and return whether it is active.

    Missing or explicitly disabled settings preserve the ordinary dense
    execution path. Enabled settings are strict and fail closed.
    """

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    optimization = config.get("optimization", {})
    if not isinstance(optimization, Mapping):
        raise TypeError("optimization must be a mapping")
    settings = optimization.get("execution_strategy")
    if settings is None:
        return False
    if not isinstance(settings, Mapping):
        raise TypeError("optimization.execution_strategy must be a mapping")
    unknown = set(settings) - _SETTINGS_KEYS
    if unknown:
        raise ValueError(f"unknown execution_strategy settings: {sorted(unknown)}")
    enabled = settings.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("execution_strategy.enabled must be a boolean")
    if not enabled:
        return False
    missing = {"task", "profile", "plan", "seed"} - set(settings)
    if missing:
        raise ValueError(f"enabled execution_strategy is missing {sorted(missing)}")
    seed = settings["seed"]
    _integer(seed, name="execution_strategy.seed")
    task = _as_contract(settings["task"], TaskContract, "task")
    profile = _as_contract(settings["profile"], CostProfile, "profile")
    plan = _as_contract(settings["plan"], ExecutionPlan, "plan")
    _validate_plan_profile(task, profile, plan, config)
    return True


def implementation_binding() -> dict[str, str]:
    """Return hashes for every runtime/scientific execution implementation."""

    package_root = Path(__file__).resolve().parents[2]
    return {
        relative: hashlib.sha256((package_root / relative).read_bytes()).hexdigest()
        for relative in _IMPLEMENTATION_PATHS
    }


class ExecutionRuntime:
    """Bound runtime decisions and TRAIN-only recovery monitoring."""

    def __init__(
        self,
        config: Mapping[str, Any],
        state: Mapping[str, Any] | ExecutionState | None = None,
        *,
        start_step: int | None = None,
        steps_per_epoch: int = 1,
    ) -> None:
        if not validate_execution_strategy(config):
            raise ValueError("ExecutionRuntime requires enabled execution_strategy settings")
        _integer(steps_per_epoch, name="steps_per_epoch", minimum=1)
        self.config = config
        self.settings = config["optimization"]["execution_strategy"]
        self.task = _as_contract(self.settings["task"], TaskContract, "task")
        self.profile = _as_contract(self.settings["profile"], CostProfile, "profile")
        configured_plan = _as_contract(self.settings["plan"], ExecutionPlan, "plan")
        self.seed = int(self.settings["seed"])
        self.steps_per_epoch = steps_per_epoch
        self.profile_identity = _profile_identity(self.profile)
        self._configured_plan = configured_plan
        self._risk_streaks: dict[str, int] = {}
        self._risk_last_epoch: dict[str, int] = {}
        self._risk_last_value: dict[str, bool] = {}
        self._noise_budget_failed = False
        self._fallback_reason: str | None = None
        self._full_recovery_intervals: list[dict[str, int]] = []
        self._full_recovery_until_epoch: int | None = None
        self._monitor_overhead_s = 0.0
        self._accepted_training_cost_s = 0.0
        self._risk_epoch: int | None = None
        self._risk_epoch_sums: dict[str, float] = {}
        self._risk_epoch_counts: dict[str, int] = {}
        self._reference_binding: str | None = None

        envelope: Mapping[str, Any] | None = None
        if isinstance(state, ExecutionState):
            saved_state = state
            current_plan = configured_plan
        elif state is None:
            saved_state = None
            current_plan = configured_plan
        elif isinstance(state, Mapping):
            if "execution_state" in state:
                envelope = state
                saved_state = ExecutionState.from_mapping(state["execution_state"])
                saved_plan = state.get("current_plan")
                current_plan = (
                    configured_plan
                    if saved_plan is None
                    else _as_contract(saved_plan, ExecutionPlan, "saved current_plan")
                )
            else:
                saved_state = ExecutionState.from_mapping(state)
                current_plan = configured_plan
        else:
            raise TypeError(
                "state must be an execution-state mapping, state envelope, or ExecutionState"
            )

        fresh_profile_rebase = False
        if envelope is not None:
            if envelope.get("task_hash") != self.task.task_hash:
                raise ValueError("serialized runtime task binding mismatch")
            if envelope.get("profile_identity") != self.profile_identity:
                # A newer, explicitly measured TRAIN profile may replace a
                # recovered full plan only after its full recovery interval.
                previous_profile_epoch = int(envelope.get("profiled_epoch", -1))
                recovery_until = envelope.get("full_recovery_until_epoch")
                is_fresh = (
                    self.profile.profiled_epoch > previous_profile_epoch
                    and recovery_until is not None
                    and self.profile.profiled_epoch >= int(recovery_until)
                )
                if not is_fresh:
                    raise ValueError("serialized runtime profile binding mismatch")
                current_plan = configured_plan
                fresh_profile_rebase = True
                self._full_recovery_intervals = [
                    {"start_epoch": int(item["start_epoch"]), "end_epoch": int(item["end_epoch"])}
                    for item in envelope.get("full_recovery_intervals", ())
                ]
                self._monitor_overhead_s = _finite_nonnegative(
                    envelope.get("monitor_overhead_s", 0), name="monitor_overhead_s"
                )
                self._accepted_training_cost_s = _finite_nonnegative(
                    envelope.get("accepted_training_cost_s", 0),
                    name="accepted_training_cost_s",
                )
                self._reference_binding = envelope.get("reference_binding")
            else:
                self._risk_streaks = {
                    str(key): int(value) for key, value in envelope.get("risk_streaks", {}).items()
                }
                self._risk_last_epoch = {
                    str(key): int(value)
                    for key, value in envelope.get("risk_last_epoch", {}).items()
                }
                self._risk_last_value = {
                    str(key): bool(value)
                    for key, value in envelope.get("risk_last_value", {}).items()
                }
                self._noise_budget_failed = bool(envelope.get("noise_budget_failed", False))
                self._fallback_reason = envelope.get("fallback_reason")
                self._full_recovery_intervals = [
                    {"start_epoch": int(item["start_epoch"]), "end_epoch": int(item["end_epoch"])}
                    for item in envelope.get("full_recovery_intervals", ())
                ]
                recovery_value = envelope.get("full_recovery_until_epoch")
                self._full_recovery_until_epoch = (
                    None if recovery_value is None else int(recovery_value)
                )
                self._monitor_overhead_s = _finite_nonnegative(
                    envelope.get("monitor_overhead_s", 0), name="monitor_overhead_s"
                )
                self._accepted_training_cost_s = _finite_nonnegative(
                    envelope.get("accepted_training_cost_s", 0),
                    name="accepted_training_cost_s",
                )
                risk_epoch = envelope.get("risk_epoch")
                self._risk_epoch = None if risk_epoch is None else int(risk_epoch)
                self._risk_epoch_sums = {
                    str(key): float(value)
                    for key, value in envelope.get("risk_epoch_sums", {}).items()
                }
                self._risk_epoch_counts = {
                    str(key): int(value)
                    for key, value in envelope.get("risk_epoch_counts", {}).items()
                }
                self._reference_binding = envelope.get("reference_binding")

        if envelope is not None and not fresh_profile_rebase:
            if saved_state is None:
                raise ValueError("serialized runtime envelope is missing its execution state")
            _validate_saved_current_plan(
                current_plan, configured_plan, self.profile, saved_state, envelope
            )
        if current_plan.task_hash != self.task.task_hash:
            raise ValueError("serialized runtime plan binding mismatch")
        if saved_state is None:
            cursor = 0 if start_step is None else _integer(start_step, name="start_step")
            accepted_epoch = cursor // steps_per_epoch
            self.state = ExecutionState.for_plan(
                self.task,
                current_plan,
                rng_seed=self.seed,
                accepted_cursor=cursor,
                accepted_epoch=accepted_epoch,
            )
        else:
            cursor = saved_state.accepted_cursor
            if start_step is not None and _integer(start_step, name="start_step") != cursor:
                raise ValueError("start_step disagrees with saved accepted cursor")
            if saved_state.rng_seed != self.seed:
                raise ValueError("execution RNG seed disagrees with saved state")
            if fresh_profile_rebase:
                # A newer TRAIN profile plus a completed full-recovery interval
                # is the only path that can re-enter a reduced plan.
                if self.profile.profiled_epoch > saved_state.accepted_epoch:
                    raise ValueError("fresh TRAIN profile is newer than accepted runtime state")
                saved_state.plan_id = current_plan.plan_id
                saved_state.last_refresh_epoch = self.profile.profiled_epoch
                self._risk_streaks.clear()
                self._risk_last_epoch.clear()
                self._risk_last_value.clear()
                self._noise_budget_failed = False
                self._fallback_reason = None
                self._full_recovery_until_epoch = None
            elif saved_state.plan_id != current_plan.plan_id:
                raise ValueError("saved execution state and current plan IDs differ")
            saved_state.validate_resume(
                task_hash=self.task.task_hash,
                plan_id=current_plan.plan_id,
                accepted_cursor=cursor,
            )
            self.state = saved_state
        self.plan = current_plan
        self.plan.validate_for_task(self.task)
        current_candidate_ids = {item.plan_id for item in self.profile.measurements}
        if self.plan.plan_id not in current_candidate_ids:
            raise ValueError("current runtime plan is absent from its TRAIN profile")
        known_fields = {name for name, _, _ in self.task.field_bounds}
        summary_fields = (
            set(self._risk_streaks)
            | set(self._risk_last_epoch)
            | set(self._risk_last_value)
            | set(self._risk_epoch_sums)
            | set(self._risk_epoch_counts)
        )
        if not summary_fields.issubset(known_fields):
            raise ValueError("saved runtime risk summary contains fields outside TaskContract")
        if set(self._risk_epoch_sums) != set(self._risk_epoch_counts):
            raise ValueError("saved runtime pooled risk sums/counts are inconsistent")
        for field, value in self._risk_streaks.items():
            _integer(value, name=f"risk_streaks[{field!r}]")
        for field, value in self._risk_epoch_sums.items():
            if not math.isfinite(value):
                raise ValueError(f"saved runtime pooled risk for {field!r} is non-finite")
            count = _integer(self._risk_epoch_counts[field], name=f"risk_epoch_counts[{field!r}]")
            if count < 1 or count > self.steps_per_epoch:
                raise ValueError(f"saved runtime pooled risk count for {field!r} is invalid")
        if self._risk_epoch is not None:
            _integer(self._risk_epoch, name="risk_epoch")
            if self._risk_epoch > self.state.accepted_epoch:
                raise ValueError("saved pooled risk epoch is newer than accepted TRAIN state")

    def decision(self, step: int) -> dict[str, Any]:
        """Return a deterministic plan decision for the current accepted step."""

        cursor = _integer(step, name="step")
        if cursor != self.state.accepted_cursor:
            raise ValueError("decision step must equal the saved accepted cursor")
        return self.plan.to_execution_mapping(accepted_cursor=cursor, rng_seed=self.seed)

    def accept(
        self,
        step: int,
        epoch: int,
        cost_s: float,
        field_violations: Mapping[str, Any] | None = None,
    ) -> None:
        """Commit one accepted update and record TRAIN-only risk diagnostics.

        ``field_violations`` maps field names to normalized ``h_f`` values (or
        booleans); positive values indicate a bound violation. Checks count at
        most once per epoch so batches do not masquerade as sustained epochs.
        """

        cursor = _integer(step, name="step")
        epoch_value = _integer(epoch, name="epoch")
        cost = _finite_nonnegative(cost_s, name="cost_s")
        if cursor != self.state.accepted_cursor:
            raise ValueError("accepted update step does not match runtime cursor")
        if epoch_value < self.state.accepted_epoch:
            raise ValueError("accepted update epoch moved backwards")
        parsed_violations: dict[str, float] = {}
        if field_violations is not None:
            if not isinstance(field_violations, Mapping):
                raise TypeError("field_violations must map field names to normalized violations")
            allowed = {name for name, _, _ in self.task.field_bounds}
            if set(field_violations) - allowed:
                raise ValueError("field_violations contains fields outside the task contract")
            parsed_violations = {
                str(name): _violation_value(value, name=str(name))
                for name, value in field_violations.items()
            }
        self.state.commit_accepted_update(cursor_before_update=cursor)
        self.state.observe_train(epoch=epoch_value, cost_s=cost, split="TRAIN")
        self._accepted_training_cost_s += cost
        if parsed_violations:
            if self._risk_epoch is not None and self._risk_epoch != epoch_value:
                # A partial epoch cannot establish a sustained-risk signal.
                self._risk_epoch_sums.clear()
                self._risk_epoch_counts.clear()
            self._risk_epoch = epoch_value
            for field, value in parsed_violations.items():
                self._risk_epoch_sums[field] = self._risk_epoch_sums.get(field, 0.0) + value
                self._risk_epoch_counts[field] = self._risk_epoch_counts.get(field, 0) + 1
        if (cursor + 1) % self.steps_per_epoch == 0 and self._risk_epoch == epoch_value:
            pooled = {
                field: total / self._risk_epoch_counts[field]
                for field, total in self._risk_epoch_sums.items()
                if self._risk_epoch_counts.get(field, 0) == self.steps_per_epoch
            }
            if pooled:
                self._record_risk_values(pooled, epoch_value)
            self._risk_epoch = None
            self._risk_epoch_sums.clear()
            self._risk_epoch_counts.clear()

    def observe_audit(
        self,
        epoch: int,
        cost_s: float,
        noise_variance: float,
        intrinsic_noise_variance: float,
        full_metrics: Mapping[str, Any] | None,
        split: str = "TRAIN",
    ) -> None:
        """Record a scheduled TRAIN probe and, when due, dense full-risk audit.

        ``full_metrics`` is a mapping from task field names to raw measured
        risks. It is used only at the plan's full-audit cadence (20+ epochs).
        """

        if str(split).upper() != "TRAIN":
            raise ValueError(
                "held-out observations cannot drive execution adaptation; TRAIN required"
            )
        epoch_value = _integer(epoch, name="epoch")
        cost = _finite_nonnegative(cost_s, name="cost_s")
        observed_noise = _finite_nonnegative(noise_variance, name="noise_variance")
        intrinsic = _finite_nonnegative(intrinsic_noise_variance, name="intrinsic_noise_variance")
        if epoch_value < self.plan.refresh_interval_epochs:
            return
        if epoch_value % self.plan.refresh_interval_epochs != 0:
            return
        risk_values: dict[str, float] | None = None
        if full_metrics is not None:
            if (
                epoch_value < self.plan.audit_interval_epochs
                or (epoch_value % self.plan.audit_interval_epochs != 0)
                or not full_audit_due(
                    self.state, epoch_value, interval_epochs=self.plan.audit_interval_epochs
                )
            ):
                raise ValueError("full_metrics may be supplied only at a scheduled 20+ epoch audit")
            if not isinstance(full_metrics, Mapping):
                raise TypeError("full_metrics must map task field names to raw risks")
            task_fields = {name for name, _, _ in self.task.field_bounds}
            if set(full_metrics) != task_fields:
                raise ValueError("full_metrics must provide exactly the task's dense field risks")
            risk_values = {
                name: _risk_violation(full_metrics[name], bound, scale, field=name)
                for name, bound, scale in self.task.field_bounds
            }

        self._monitor_overhead_s += cost
        self.state.charged_overhead_s += cost
        extra_noise = max(0.0, observed_noise - intrinsic)
        self.state.observe_train(
            epoch=epoch_value,
            cost_s=cost,
            noise_variance=observed_noise,
            split="TRAIN",
        )
        allowed_noise = min(float(self.plan.variance_ceiling), intrinsic)
        if extra_noise > allowed_noise + 1e-12:
            self._noise_budget_failed = True

        if risk_values is not None:
            self._record_risk_values(risk_values, epoch_value)
            self.state.last_full_audit_epoch = epoch_value

    def refresh(self, epoch: int) -> ExecutionPlan:
        """Refresh at the minimum interval and fall back to full when unsafe."""

        epoch_value = _integer(epoch, name="epoch")
        if epoch_value < self.plan.refresh_interval_epochs:
            return self.plan
        if epoch_value > self.state.accepted_epoch:
            raise ValueError("refresh epoch is newer than accepted TRAIN state")
        if not policy_refresh_due(
            self.state, epoch_value, interval_epochs=self.plan.refresh_interval_epochs
        ):
            return self.plan
        sustained = sorted(
            name
            for name, count in self._risk_streaks.items()
            if count >= _SUSTAINED_VIOLATION_EPOCHS
        )
        if self._noise_budget_failed:
            self._fallback_reason = (
                "observed TRAIN estimator noise exceeded the selected plan ceiling"
            )
        elif (
            self._monitor_overhead_s > _MAX_MONITOR_OVERHEAD_RATIO * self._accepted_training_cost_s
        ):
            self._fallback_reason = (
                "charged TRAIN monitoring overhead exceeded 5% of cumulative accepted-update cost"
            )
        elif sustained:
            self._fallback_reason = "sustained TRAIN fidelity violations for fields: " + ", ".join(
                sustained
            )
        if self._fallback_reason and not self.plan.no_action:
            self.plan = self._full_plan(epoch_value, self._fallback_reason)
            self.state.plan_id = self.plan.plan_id
            interval_end = epoch_value + self.plan.refresh_interval_epochs
            self._full_recovery_until_epoch = interval_end
            self._full_recovery_intervals.append(
                {"start_epoch": epoch_value, "end_epoch": interval_end}
            )
            if len(self._full_recovery_intervals) > 16:
                del self._full_recovery_intervals[:-16]
        self.state.last_refresh_epoch = epoch_value
        return self.plan

    def bind_reference(self, reference: Any) -> None:
        """Bind the task's C0, ordered field bounds and scales to initialized reference."""

        if isinstance(reference, Mapping) and "reference" in reference:
            reference = reference["reference"]
        if isinstance(reference, Mapping):
            field_names = tuple(reference.get("field_names", ()))
            bounds = tuple(reference.get("upper_bounds", ()))
            scales = tuple(reference.get("denominators", ()))
            c0 = reference.get("c0")
        else:
            field_names = tuple(getattr(reference, "field_names", ()))
            bounds = tuple(getattr(reference, "upper_bounds", ()))
            scales = tuple(getattr(reference, "denominators", ()))
            c0 = getattr(reference, "c0", None)
        if c0 is None:
            raise TypeError("reference must expose c0, field_names, upper_bounds, and denominators")
        expected_fields = tuple(name for name, _, _ in self.task.field_bounds)
        expected_bounds = tuple(float(bound) for _, bound, _ in self.task.field_bounds)
        expected_scales = tuple(float(scale) for _, _, scale in self.task.field_bounds)
        if field_names != expected_fields:
            raise ValueError("initialized fidelity field order differs from TaskContract")
        if len(bounds) != len(expected_bounds) or any(
            not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-14)
            for actual, expected in zip(bounds, expected_bounds)
        ):
            raise ValueError("initialized fidelity bounds differ from TaskContract")
        if len(scales) != len(expected_scales) or any(
            not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-14)
            for actual, expected in zip(scales, expected_scales)
        ):
            raise ValueError("initialized fidelity scales differ from TaskContract")
        if not math.isclose(float(c0), float(self.task.c0), rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError("initialized C0 differs from TaskContract")
        binding_payload = {
            "field_names": field_names,
            "upper_bounds": [float(value) for value in bounds],
            "denominators": [float(value) for value in scales],
            "c0": float(c0),
        }
        binding = hashlib.sha256(_canonical(binding_payload).encode("utf-8")).hexdigest()
        if self._reference_binding is not None and self._reference_binding != binding:
            raise ValueError("initialized fidelity reference differs from saved runtime binding")
        self._reference_binding = binding

    def state_dict(self) -> dict[str, Any]:
        """Return bounded recovery state, current plan, and charged monitoring cost."""

        return {
            "schema_version": 1,
            "task_hash": self.task.task_hash,
            "profile_identity": self.profile_identity,
            "profiled_epoch": self.profile.profiled_epoch,
            "current_plan": self.plan.to_dict(),
            "execution_state": self.state.to_dict(),
            "risk_streaks": dict(self._risk_streaks),
            "risk_last_epoch": dict(self._risk_last_epoch),
            "risk_last_value": dict(self._risk_last_value),
            "noise_budget_failed": self._noise_budget_failed,
            "fallback_reason": self._fallback_reason,
            "full_recovery_until_epoch": self._full_recovery_until_epoch,
            "full_recovery_intervals": copy.deepcopy(self._full_recovery_intervals),
            "monitor_overhead_s": float(self._monitor_overhead_s),
            "accepted_training_cost_s": float(self._accepted_training_cost_s),
            "risk_epoch": self._risk_epoch,
            "risk_epoch_sums": dict(self._risk_epoch_sums),
            "risk_epoch_counts": dict(self._risk_epoch_counts),
            "reference_binding": self._reference_binding,
        }

    def _record_risk_values(self, values: Mapping[str, float], epoch: int) -> None:
        for field, violation in values.items():
            previous_epoch = self._risk_last_epoch.get(field)
            previous_violated = self._risk_last_value.get(field, False)
            if previous_epoch == epoch:
                # A full audit may add a positive signal after the pooled
                # per-epoch training check, without counting the epoch twice.
                if violation > 0:
                    if not previous_violated:
                        self._risk_streaks[field] = max(1, self._risk_streaks.get(field, 0))
                    self._risk_last_value[field] = True
                continue
            violated = violation > 0
            if violated:
                if previous_epoch is not None and epoch > previous_epoch and previous_violated:
                    self._risk_streaks[field] = self._risk_streaks.get(field, 0) + 1
                else:
                    self._risk_streaks[field] = 1
            else:
                self._risk_streaks[field] = 0
            self._risk_last_epoch[field] = epoch
            self._risk_last_value[field] = violated

    def _full_plan(self, epoch: int, reason: str) -> ExecutionPlan:
        full = next(item for item in self.profile.measurements if item.plan_id == "full")
        remaining = max(0, self.task.max_epochs - epoch)
        projected = float(full.upper_epoch_cost_s) * remaining
        rejection = dict(self.plan.rejection_reasons)
        rejection["runtime_fallback"] = reason
        return replace(
            self.plan,
            plan_id="full",
            selected_actions=(),
            family_probabilities={},
            within_family={},
            family_support=(),
            predicted_total_s=projected,
            predicted_full_total_s=projected,
            cost_savings_ratio=0.0,
            extra_variance=0.0,
            max_joint_importance_multiplier=1.0,
            no_action=True,
            selected_reason=f"Fell back to measured full execution at epoch {epoch}: {reason}.",
            rejection_reasons=rejection,
            accepted_cursor=self.state.accepted_cursor,
        )


def _violation_value(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        return 1.0 if value else -1.0
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"field violation for {name!r} must be boolean or finite scalar") from error
    if not math.isfinite(result):
        raise ValueError(f"field violation for {name!r} must be finite")
    return result


def _risk_violation(value: Any, bound: float, scale: float, *, field: str) -> float:
    risk = _finite_nonnegative(value, name=f"full_metrics[{field!r}]")
    return (risk - float(bound)) / float(scale)


__all__ = [
    "ExecutionRuntime",
    "implementation_binding",
    "scientific_config_hash",
    "validate_execution_strategy",
]
