"""Default-off measured execution planning for post-training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .bottleneck import profile_one_update
from .contracts import (
    CostProfile,
    ExecutionCapability,
    ExecutionPlan,
    ExecutionState,
    PlanMeasurement,
    TaskContract,
    full_audit_due,
    policy_refresh_due,
)
from .planner import (
    FULL_AUDIT_INTERVAL_EPOCHS,
    MAX_EXTRA_VARIANCE_RATIO,
    MAX_NESTED_IMPORTANCE_MULTIPLIER,
    MIN_NET_SAVINGS_RATIO,
    POLICY_REFRESH_INTERVAL_EPOCHS,
    select_execution_plan,
)
from .profiling import ProfileBudget, ProfileMeasurement, preserve_profile_state, state_fingerprint
from .providers import (
    TOPOLOGY_FILTRATION_PROVIDER_ID,
    action_registry,
    compose_nested_filtration_support,
)


def manifest_for_task(task_or_mapping: TaskContract | Mapping[str, Any]) -> dict[str, Any]:
    """Build a config-only task/capability/profile-request manifest.

    This function is pure: it does not open a dataset, checkpoint, or config
    file. A caller must pass already-resolved, source-hash-pinned identities.
    """

    task = (
        task_or_mapping
        if isinstance(task_or_mapping, TaskContract)
        else TaskContract.from_mapping(task_or_mapping)
    )
    capabilities = action_registry(task)
    by_id = {capability.action_id: capability for capability in capabilities}
    candidate_requests: list[dict[str, Any]] = [{"plan_id": "full", "actions": []}]
    if by_id["lazy_target_cache"].applicable:
        candidate_requests.append(
            {
                "plan_id": "target_cache",
                "actions": ["lazy_target_cache"],
                "measurement": "whole_plan",
            }
        )
    if by_id["whole_family_sampling"].applicable:
        candidate_requests.append(
            {
                "plan_id": "whole_family_p0.5",
                "actions": ["whole_family_sampling"],
                "measurement": "whole_plan_plus_TRAIN_noise",
            }
        )
    if by_id["generalized_filtration_sampling"].applicable:
        candidate_requests.append(
            {
                "plan_id": "filtration_polarity",
                "actions": ["generalized_filtration_sampling"],
                "measurement": "whole_plan_plus_exact_joint_support",
            }
        )
    if (
        by_id["whole_family_sampling"].applicable
        and by_id["generalized_filtration_sampling"].applicable
    ):
        candidate_requests.append(
            {
                "plan_id": "family_p0.5_x_filtration_polarity",
                "actions": ["whole_family_sampling", "generalized_filtration_sampling"],
                "measurement": "whole_plan_plus_composed_joint_support_and_TRAIN_noise",
            }
        )
    if by_id["scalar_backward_fusion"].applicable and len(candidate_requests) < 6:
        candidate_requests.append(
            {
                "plan_id": "scalar_backward_fusion",
                "actions": ["scalar_backward_fusion"],
                "measurement": "whole_plan",
            }
        )
    candidate_requests = candidate_requests[:6]
    return {
        "task_contract": task.to_dict(),
        "task_hash": task.task_hash,
        "capabilities": [capability.to_dict() for capability in capabilities],
        "cost_profile_request": {
            "schema_version": 1,
            "status": "unexecuted",
            "task_hash": task.task_hash,
            "calibration_status": task.calibration_status,
            "calibration_required_before_profile": task.calibration_status != "bound",
            "profile_split": "TRAIN",
            "noise_split": "TRAIN",
            "max_candidates": 6,
            "maximum_horizon_epochs": task.max_epochs,
            "candidate_requests": candidate_requests,
            "required_measurements": [
                "whole_plan_mean_and_conservative_upper_epoch_cost_s",
                "candidate_acquisition_refresh_and_audit_overhead_s",
                "measured_intrinsic_TRAIN_noise_variance",
                "extra_estimator_variance_and_evidence_method",
                "exact_family_and_within_family_joint_inclusion_support",
                "memory_storage_and_cache_hit_miss_eviction_statistics_when_applicable",
            ],
            "policy_constants": {
                "minimum_net_savings_ratio": MIN_NET_SAVINGS_RATIO,
                "maximum_extra_variance_to_intrinsic_ratio": MAX_EXTRA_VARIANCE_RATIO,
                "maximum_nested_importance_multiplier": MAX_NESTED_IMPORTANCE_MULTIPLIER,
                "minimum_policy_refresh_interval_epochs": POLICY_REFRESH_INTERVAL_EPOCHS,
                "minimum_full_audit_interval_epochs": FULL_AUDIT_INTERVAL_EPOCHS,
            },
        },
    }


__all__ = [
    "FULL_AUDIT_INTERVAL_EPOCHS",
    "MAX_EXTRA_VARIANCE_RATIO",
    "MAX_NESTED_IMPORTANCE_MULTIPLIER",
    "MIN_NET_SAVINGS_RATIO",
    "POLICY_REFRESH_INTERVAL_EPOCHS",
    "TOPOLOGY_FILTRATION_PROVIDER_ID",
    "CostProfile",
    "ExecutionCapability",
    "ExecutionPlan",
    "ExecutionState",
    "PlanMeasurement",
    "ProfileBudget",
    "ProfileMeasurement",
    "TaskContract",
    "action_registry",
    "compose_nested_filtration_support",
    "full_audit_due",
    "manifest_for_task",
    "policy_refresh_due",
    "preserve_profile_state",
    "profile_one_update",
    "select_execution_plan",
    "state_fingerprint",
]
