"""Measured, conservative selection among a few eligible execution plans."""

from __future__ import annotations

import math
from collections.abc import Mapping

from .contracts import (
    CostProfile,
    ExecutionPlan,
    ExecutionState,
    PlanMeasurement,
    TaskContract,
)
from .providers import action_registry

MAX_NESTED_IMPORTANCE_MULTIPLIER = 4.0
MIN_NET_SAVINGS_RATIO = 0.05
MAX_EXTRA_VARIANCE_RATIO = 1.0
POLICY_REFRESH_INTERVAL_EPOCHS = 10
FULL_AUDIT_INTERVAL_EPOCHS = 20


def _candidate_rejection(
    task: TaskContract,
    profile: CostProfile,
    candidate: PlanMeasurement,
    registry: Mapping[str, object],
) -> str | None:
    if candidate.plan_id == "full":
        return None
    if candidate.evidence_status not in {"measured", "validated"}:
        return f"candidate evidence status {candidate.evidence_status!r} is not measured"
    for action_id in candidate.actions:
        capability = registry.get(action_id)
        if capability is None:
            return f"action {action_id!r} is not registered"
        if not capability.applicable:
            return f"action {action_id!r} is ineligible: {capability.reason}"
    if candidate.estimator_kind in {"approximate_direction", "unsupported"}:
        return f"estimator kind {candidate.estimator_kind!r} is not eligible for this campaign"
    if (
        candidate.estimator_kind == "unbiased_scalar_estimator"
        and task.objective_rule != "scalar_additive"
    ):
        return "unbiased scalar sampling requires the frozen scalar-additive objective"
    if not candidate.support_validated:
        return "estimator support and joint inclusion probabilities are not validated"
    if not candidate.support_nonempty:
        return "candidate support can produce an empty objective mapping"
    if candidate.family_probabilities:
        if (
            not any(probability >= 1.0 for probability in candidate.family_probabilities.values())
            and not candidate.family_support
        ):
            return (
                "family sampler has no always-on contribution or explicit non-empty joint support"
            )
        if not any(action in candidate.actions for action in ("whole_family_sampling",)):
            return "family probabilities were supplied without the registered whole-family action"
    if candidate.within_family and "generalized_filtration_sampling" not in candidate.actions:
        return "within-family probabilities were supplied without their registered provider action"
    if candidate.within_family:
        from .providers import TOPOLOGY_FILTRATION_PROVIDER_ID

        for family, specification in candidate.within_family.items():
            if specification.get("provider_id") != TOPOLOGY_FILTRATION_PROVIDER_ID:
                return f"within-family entry {family!r} names an unregistered provider"
            if TOPOLOGY_FILTRATION_PROVIDER_ID not in task.provider_ids:
                return "task contract does not bind the complete-filtration provider"
            if specification.get("estimator") != "polarity":
                return f"within-family entry {family!r} uses an unsupported estimator"
            if not math.isclose(
                float(specification.get("conditional_inclusion_probability", 0.0)),
                0.5,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                return "the registered polarity estimator requires conditional p=0.5"
    multiplier = candidate.max_joint_importance_multiplier
    if not math.isfinite(multiplier) or multiplier > MAX_NESTED_IMPORTANCE_MULTIPLIER + 1e-12:
        return f"nested maximum importance multiplier {multiplier:.6g} exceeds 4"
    if candidate.extra_variance > 0 and candidate.variance_method not in {
        "exact_support",
        "exact_joint_support",
        "paired_train_probe",
        "fixed_train_probe",
    }:
        return "extra estimator variance has no accepted finite-support or TRAIN-probe evidence"
    variance_ceiling = profile.intrinsic_noise_variance * MAX_EXTRA_VARIANCE_RATIO
    if candidate.extra_variance > variance_ceiling + 1e-12:
        return (
            f"extra TRAIN estimator variance {candidate.extra_variance:.6g} exceeds "
            f"the allowed intrinsic-noise ceiling {variance_ceiling:.6g}"
        )
    return None


def _projected_cost(
    candidate: PlanMeasurement,
    *,
    profile: CostProfile,
    remaining_epochs: int,
    refresh_interval_epochs: int,
    audit_interval_epochs: int,
) -> float:
    if remaining_epochs <= 0:
        return 0.0
    refreshes = math.ceil(remaining_epochs / refresh_interval_epochs)
    audits = math.ceil(remaining_epochs / audit_interval_epochs)
    return (
        profile.profile_overhead_s
        + candidate.acquisition_overhead_s
        + candidate.upper_epoch_cost_s * remaining_epochs
        + candidate.refresh_overhead_s * refreshes
        + candidate.audit_overhead_s * audits
    )


def select_execution_plan(
    task: TaskContract,
    profile: CostProfile,
    state: ExecutionState | None = None,
    *,
    minimum_savings_ratio: float = MIN_NET_SAVINGS_RATIO,
    maximum_extra_variance_ratio: float = MAX_EXTRA_VARIANCE_RATIO,
    max_importance_multiplier: float = MAX_NESTED_IMPORTANCE_MULTIPLIER,
    refresh_interval_epochs: int = POLICY_REFRESH_INTERVAL_EPOCHS,
    audit_interval_epochs: int = FULL_AUDIT_INTERVAL_EPOCHS,
) -> ExecutionPlan:
    """Choose the lowest conservative total cost among qualified plans.

    Cost comparisons include task-specific profile, setup, refresh, and audit
    overhead. The reference is the measured full plan without strategy
    acquisition overhead, so an unamortized profile cannot appear beneficial.
    Estimator qualification uses only the profile's TRAIN noise summary.
    """

    if profile.task_hash != task.task_hash:
        raise ValueError("cost profile is bound to a different task contract")
    if task.calibration_status != "bound":
        raise ValueError("execution-plan selection requires completed SOURCE TRAIN calibration")
    if profile.max_epochs != task.max_epochs:
        raise ValueError("profile and task maximum horizons differ")
    if not 0 <= profile.profiled_epoch <= task.max_epochs:
        raise ValueError("profile epoch is outside the task horizon")
    if profile.profile_split.upper() != "TRAIN" or profile.noise_split.upper() != "TRAIN":
        raise ValueError("held-out measurements cannot drive plan selection")
    if not MIN_NET_SAVINGS_RATIO <= minimum_savings_ratio < 1:
        raise ValueError("minimum_savings_ratio cannot weaken the frozen 5% savings floor")
    if not 0 <= maximum_extra_variance_ratio <= MAX_EXTRA_VARIANCE_RATIO:
        raise ValueError("noise ratio cannot exceed the frozen intrinsic-noise ceiling")
    if not 1 <= max_importance_multiplier <= MAX_NESTED_IMPORTANCE_MULTIPLIER:
        raise ValueError("importance ceiling cannot exceed the frozen nested multiplier of four")
    if refresh_interval_epochs < 10 or audit_interval_epochs < 20:
        raise ValueError("refresh/audit intervals must be at least 10/20 epochs")
    if state is not None:
        state.validate_resume(task_hash=task.task_hash, plan_id=state.plan_id)
        if profile.profiled_epoch < state.accepted_epoch:
            raise ValueError("profile is older than the accepted TRAIN execution state")
        if (
            state.last_refresh_epoch is not None
            and profile.profiled_epoch - state.last_refresh_epoch < refresh_interval_epochs
        ):
            raise ValueError("execution policy refresh is not due; wait at least 10 epochs")

    measurements = {item.plan_id: item for item in profile.measurements}
    full = measurements["full"]
    if full.evidence_status not in {"measured", "validated"}:
        raise ValueError("the full/no-action reference must have measured TRAIN cost evidence")
    if full.actions or full.family_probabilities or full.within_family:
        raise ValueError(
            "the measured 'full' candidate must have no sampling actions or probabilities"
        )
    remaining_epochs = max(0, min(task.max_epochs, profile.max_epochs) - profile.profiled_epoch)
    full_reference_total = full.upper_epoch_cost_s * remaining_epochs
    actual_full_total = _projected_cost(
        full,
        profile=profile,
        remaining_epochs=remaining_epochs,
        refresh_interval_epochs=refresh_interval_epochs,
        audit_interval_epochs=audit_interval_epochs,
    )
    rejection_reasons: dict[str, str] = {}
    registry = {item.action_id: item for item in action_registry(task)}
    qualified: list[tuple[float, float, PlanMeasurement]] = []

    for candidate in profile.measurements:
        if candidate.plan_id == "full":
            continue
        reason = _candidate_rejection(task, profile, candidate, registry)
        if reason is not None:
            rejection_reasons[candidate.plan_id] = reason
            continue
        if candidate.max_joint_importance_multiplier > max_importance_multiplier + 1e-12:
            rejection_reasons[candidate.plan_id] = (
                f"nested maximum importance multiplier {candidate.max_joint_importance_multiplier:.6g} "
                f"exceeds configured ceiling {max_importance_multiplier:.6g}"
            )
            continue
        if (
            candidate.extra_variance
            > profile.intrinsic_noise_variance * maximum_extra_variance_ratio + 1e-12
        ):
            allowed = profile.intrinsic_noise_variance * maximum_extra_variance_ratio
            rejection_reasons[candidate.plan_id] = (
                f"extra TRAIN estimator variance {candidate.extra_variance:.6g} exceeds "
                f"configured ceiling {allowed:.6g}"
            )
            continue
        predicted_total = _projected_cost(
            candidate,
            profile=profile,
            remaining_epochs=remaining_epochs,
            refresh_interval_epochs=refresh_interval_epochs,
            audit_interval_epochs=audit_interval_epochs,
        )
        if remaining_epochs <= 0 or full_reference_total <= 0:
            rejection_reasons[candidate.plan_id] = (
                "no remaining horizon is available to repay adoption overhead"
            )
            continue
        net_ratio = (full_reference_total - predicted_total) / full_reference_total
        if predicted_total >= full_reference_total:
            rejection_reasons[candidate.plan_id] = (
                f"profile, adoption, refresh, and audit overhead do not break even within "
                f"the remaining {remaining_epochs} epochs"
            )
            continue
        if net_ratio + 1e-12 < minimum_savings_ratio:
            rejection_reasons[candidate.plan_id] = (
                f"net conservative savings {net_ratio:.1%} are below the "
                f"{minimum_savings_ratio:.1%} decision threshold"
            )
            continue
        qualified.append((predicted_total, candidate.extra_variance, candidate))

    if not qualified:
        return ExecutionPlan(
            task_hash=task.task_hash,
            plan_id="full",
            selected_actions=(),
            family_probabilities={},
            within_family={},
            family_support=(),
            predicted_total_s=actual_full_total,
            predicted_full_total_s=full_reference_total,
            cost_savings_ratio=0.0,
            extra_variance=0.0,
            variance_ceiling=profile.intrinsic_noise_variance * maximum_extra_variance_ratio,
            max_joint_importance_multiplier=1.0,
            no_action=True,
            selected_reason=(
                "Selected exact full execution; no accelerated candidate passed applicability, "
                "support, TRAIN-noise, multiplier, and cold-start break-even gates."
            ),
            rejection_reasons=rejection_reasons,
            refresh_interval_epochs=refresh_interval_epochs,
            audit_interval_epochs=audit_interval_epochs,
            accepted_cursor=0 if state is None else state.accepted_cursor,
        )

    predicted_total, extra_variance, winner = min(
        qualified, key=lambda item: (item[0], item[1], item[2].plan_id)
    )
    savings = (full_reference_total - predicted_total) / full_reference_total
    return ExecutionPlan(
        task_hash=task.task_hash,
        plan_id=winner.plan_id,
        selected_actions=winner.actions,
        family_probabilities=dict(winner.family_probabilities),
        within_family={key: dict(value) for key, value in winner.within_family.items()},
        family_support=winner.family_support,
        predicted_total_s=predicted_total,
        predicted_full_total_s=full_reference_total,
        cost_savings_ratio=savings,
        extra_variance=extra_variance,
        variance_ceiling=profile.intrinsic_noise_variance * maximum_extra_variance_ratio,
        max_joint_importance_multiplier=winner.max_joint_importance_multiplier,
        no_action=False,
        selected_reason=(
            f"Selected measured plan {winner.plan_id!r} at {savings:.1%} conservative net savings; "
            f"added TRAIN execution variance {extra_variance:.6g} is within the "
            f"{profile.intrinsic_noise_variance * maximum_extra_variance_ratio:.6g} ceiling."
        ),
        rejection_reasons=rejection_reasons,
        refresh_interval_epochs=refresh_interval_epochs,
        audit_interval_epochs=audit_interval_epochs,
        accepted_cursor=0 if state is None else state.accepted_cursor,
    )
