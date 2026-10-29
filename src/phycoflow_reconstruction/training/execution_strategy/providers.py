"""Small capability registry and estimator-support helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .contracts import ExecutionCapability, TaskContract

TOPOLOGY_FILTRATION_PROVIDER_ID = "topology.complete_filtration.v1"


def action_registry(task: TaskContract) -> tuple[ExecutionCapability, ...]:
    """Return a bounded registry with applicability evaluated from declarations.

    ``features`` and ``provider_ids`` are resolved task-contract claims. This
    function does not infer estimator semantics from a coherence family name.
    """

    features = set(task.features)
    providers = set(task.provider_ids)
    scalar_additive = task.objective_rule == "scalar_additive"

    def cap(
        action_id: str,
        provider_id: str,
        applicable: bool,
        preservation_level: str,
        *,
        reason: str,
        assumptions: tuple[str, ...] = (),
        failure_conditions: tuple[str, ...] = (),
        prerequisites: tuple[str, ...] = (),
        shared_work_keys: tuple[str, ...] = (),
        output_scope: str = "",
        differentiable: bool = True,
        estimator_support: str = "exact",
        numerical_oracle: str = "",
        cache_identity: str = "",
        evidence_status: str = "unmeasured",
        validated_workload_scope: str = "",
    ) -> ExecutionCapability:
        return ExecutionCapability(
            action_id=action_id,
            provider_id=provider_id,
            applicable=applicable,
            preservation_level=preservation_level,
            assumptions=assumptions,
            failure_conditions=failure_conditions,
            prerequisites=prerequisites,
            shared_work_keys=shared_work_keys,
            output_scope=output_scope,
            differentiable=differentiable,
            estimator_support=estimator_support,
            numerical_oracle=numerical_oracle,
            cache_identity=cache_identity,
            evidence_status=evidence_status,
            validated_workload_scope=validated_workload_scope,
            reason=reason,
        )

    result = [
        cap(
            "full",
            "core.full_execution.v1",
            True,
            "equivalent",
            reason="Exact/no-action execution is always available.",
            output_scope="all active objective terms and dense fidelity terms",
            evidence_status="baseline",
        ),
        cap(
            "shared_reconstruction",
            "core.shared_reconstruction.v1",
            "shared_forward" in features,
            "equivalent",
            reason=(
                "The task declares one shared reconstruction for descriptor and endpoint consumers."
                if "shared_forward" in features
                else "The task contract does not declare a shared reconstruction boundary."
            ),
            assumptions=(
                "consumers use the same model state, batch, query set, and stochastic draw",
            ),
            shared_work_keys=("reconstruction", "query_geometry", "model_draw"),
            output_scope="one reconstructed endpoint reused by declared consumers",
            evidence_status="existing_behavior"
            if "shared_forward" in features
            else "not_applicable",
        ),
        cap(
            "lazy_target_cache",
            "core.target_reference_cache.v1",
            "complete_target_cache_identity" in features,
            "equivalent",
            reason=(
                "The task declares immutable target inputs and a complete cache identity."
                if "complete_target_cache_identity" in features
                else "A complete immutable target-cache identity is not declared."
            ),
            assumptions=(
                "target tensors, geometry, normalization, banks, and definition are immutable",
            ),
            failure_conditions=(
                "any key component changes",
                "cache lookup/I/O exceeds measured reuse savings",
            ),
            prerequisites=("bounded cache", "measured TRAIN hit rate"),
            shared_work_keys=(
                "reference_values",
                "geometry",
                "normalization",
                "projection_or_line_bank",
            ),
            output_scope="reference-side prepared values only",
            numerical_oracle="full target-side descriptor values",
            cache_identity="task hash + target identity + geometry + normalization + bank + backend/version",
            evidence_status="requires_profile"
            if "complete_target_cache_identity" in features
            else "not_applicable",
        ),
        cap(
            "whole_family_sampling",
            "core.additive_family_ht.v1",
            scalar_additive and "additive_family_provider" in features,
            "unbiased_scalar_estimator",
            reason=(
                "The task declares an additive scalar objective and a bypassable family provider."
                if scalar_additive and "additive_family_provider" in features
                else "Whole-family sampling requires a declared additive scalar family provider."
            ),
            assumptions=(
                "every enabled family has positive inclusion probability",
                "HT weights are applied once",
            ),
            failure_conditions=(
                "non-additive aggregation",
                "empty objective support",
                "noise exceeds TRAIN ceiling",
            ),
            prerequisites=(
                "measured whole-plan cost",
                "positive joint support",
                "dense fidelity path",
            ),
            output_scope="selected scalar objective families",
            estimator_support="Horvitz-Thompson whole-family inclusion with explicit joint support",
            numerical_oracle="full scalar objective and input/parameter gradient",
            evidence_status="requires_profile",
        ),
        cap(
            "generalized_filtration_sampling",
            TOPOLOGY_FILTRATION_PROVIDER_ID,
            scalar_additive and TOPOLOGY_FILTRATION_PROVIDER_ID in providers,
            "unbiased_scalar_estimator",
            reason=(
                "A complete-filtration provider is explicitly bound to this scalar task."
                if scalar_additive and TOPOLOGY_FILTRATION_PROVIDER_ID in providers
                else "The task does not declare the certified complete-filtration provider under a scalar objective."
            ),
            assumptions=(
                "all configured dimensions stay inside each unit",
                "positive-line coefficients are preserved",
            ),
            failure_conditions=(
                "missing polarity support",
                "invalid field-to-descriptor mapping",
                "noise exceeds TRAIN ceiling",
            ),
            prerequisites=(
                "provider support summary",
                "positive joint inclusion probabilities",
                "dense fidelity path",
            ),
            output_scope="complete scalar filtrations under the configured polarity law",
            estimator_support="exact finite support with marginal and pairwise inclusion probabilities",
            numerical_oracle="legacy complete-filtration scalar and input/parameter gradient",
            evidence_status="requires_profile",
        ),
        cap(
            "scalar_backward_fusion",
            "core.scalar_backward_fusion.v1",
            scalar_additive and "scalar_optimizer" in features,
            "equivalent",
            reason=(
                "A scalar-additive optimizer rule is declared."
                if scalar_additive and "scalar_optimizer" in features
                else "Scalar backward fusion is not applicable to the declared objective/optimizer."
            ),
            assumptions=("one fixed scalar objective", "no nonlinear family-gradient aggregator"),
            failure_conditions=("ConFIG or other nonlinear gradient combination",),
            output_scope="one scalar objective backward traversal",
            numerical_oracle="dense scalar gradient under the same objective",
            evidence_status="existing_behavior"
            if "scalar_optimizer" in features
            else "not_applicable",
        ),
        cap(
            "conditional_context_reuse",
            "core.conditional_context_cache.v1",
            "exact_within_reconstruction_context" in features,
            "equivalent",
            reason=(
                "The adapter declares an exact within-reconstruction context cache."
                if "exact_within_reconstruction_context" in features
                else "The adapter does not declare exact within-reconstruction context reuse."
            ),
            assumptions=(
                "same input tensors and same live model weights within one reconstruction",
            ),
            failure_conditions=("reuse across optimizer updates or changed input tensors",),
            output_scope="one unchanged reconstruction context",
            evidence_status="adapter_declared"
            if "exact_within_reconstruction_context" in features
            else "not_applicable",
        ),
        cap(
            "batched_parameter_vjp",
            "core.batched_parameter_vjp.v1",
            "batched_parameter_vjp" in features and "fixed_gradient_rule" in features,
            "equivalent",
            reason=(
                "The backend and fixed gradient rule declare batched parameter VJPs."
                if "batched_parameter_vjp" in features and "fixed_gradient_rule" in features
                else "A measured backend capability and fixed gradient rule are not both declared."
            ),
            assumptions=(
                "same full gradient matrix and aggregation rule",
                "memory fits the measured envelope",
            ),
            failure_conditions=(
                "gradient matrix or direction differs",
                "memory/latency regression",
            ),
            output_scope="declared parameter-gradient matrix",
            evidence_status="requires_profile",
        ),
        cap(
            "periodic_config_reuse",
            "core.periodic_config_reuse.v1",
            False,
            "approximate_direction",
            reason="Quarantined: the current signed-coefficient staleness audit failed its declared cosine threshold.",
            assumptions=(
                "freshness, sign/rank, and direction-staleness conditions would need separate validation",
            ),
            failure_conditions=("stale nonlinear direction",),
            prerequisites=("explicit separate study authorization",),
            output_scope="approximate ConFIG direction",
            estimator_support="not an unbiased scalar estimator",
            evidence_status="quarantined",
        ),
    ]

    for action_id, reason in (
        ("reduced_resolution", "Changes the evaluated raster/query resolution."),
        ("reduced_solver_stages", "Changes the reconstruction operator."),
        ("reduced_covariance_ensemble", "Changes the ensemble covariance estimator."),
        ("permanent_homology_drop", "Changes the active descriptor definition."),
        ("sparse_fidelity_feedback", "Changes the dense endpoint-constraint policy."),
        ("unqualified_mixed_precision", "No task/backend-specific numerical contract is declared."),
    ):
        result.append(
            cap(
                action_id,
                f"semantic_change.{action_id}.v1",
                False,
                "unsupported",
                reason=reason,
                assumptions=(),
                failure_conditions=("does not preserve the frozen task contract",),
                output_scope="unsupported for this unchanged-task strategy",
                differentiable=action_id != "reduced_solver_stages",
                estimator_support="semantic change or unqualified backend path",
                evidence_status="ineligible",
            )
        )
    return tuple(result)


def compose_nested_filtration_support(
    family_probability: float,
    conditional_support: Any,
) -> dict[str, Any]:
    """Compose outer family sampling with an exact filtration support summary.

    This propagates the sampler's correlated support directly. Pairwise unit
    probabilities are multiplied by the outer inclusion event; they are never
    reconstructed from products of marginal probabilities.
    """

    p_family = float(family_probability)
    if not math.isfinite(p_family) or not 0 < p_family <= 1:
        raise ValueError("family_probability must be in (0, 1]")

    def member(name: str) -> Any:
        if isinstance(conditional_support, Mapping):
            return conditional_support[name]
        return getattr(conditional_support, name)

    choices = tuple(member("choices"))
    masses = tuple(float(value) for value in member("choice_probabilities"))
    marginals = tuple(float(value) for value in member("inclusion_probabilities"))
    joints = tuple(
        tuple(float(value) for value in row) for row in member("joint_inclusion_probabilities")
    )
    if not choices or len(choices) != len(masses) or not marginals:
        raise ValueError("conditional support needs choices, masses, and marginal probabilities")
    if len(joints) != len(marginals) or any(len(row) != len(marginals) for row in joints):
        raise ValueError("conditional pairwise inclusion matrix has the wrong shape")
    if any(mass <= 0 or not math.isfinite(mass) for mass in masses) or not math.isclose(
        sum(masses), 1.0, abs_tol=1e-9
    ):
        raise ValueError("conditional support masses must be positive and sum to one")
    if any(probability <= 0 or probability > 1 for probability in marginals):
        raise ValueError("conditional unit inclusion probabilities must be positive")
    for choice in choices:
        indices = tuple(choice.indices if hasattr(choice, "indices") else choice[0])
        if (
            not indices
            or len(indices) != len(set(indices))
            or any(index < 0 or index >= len(marginals) for index in indices)
        ):
            raise ValueError("conditional support choices need unique valid unit indices")
        choice_probabilities = getattr(choice, "inclusion_probabilities", None)
        if choice_probabilities is not None:
            if len(choice_probabilities) != len(indices):
                raise ValueError("conditional choice inclusion probabilities have the wrong length")
            for index, probability in zip(indices, choice_probabilities):
                if not math.isclose(
                    float(probability), marginals[index], rel_tol=1e-8, abs_tol=1e-10
                ):
                    raise ValueError(
                        "conditional choice probability disagrees with support marginal"
                    )
    for left, row in enumerate(joints):
        if len(row) != len(marginals):
            raise ValueError("conditional pairwise inclusion matrix has the wrong shape")
        if not math.isclose(row[left], marginals[left], rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError("conditional joint inclusion diagonal must equal the marginal")
        for right, value in enumerate(row):
            if not 0 <= value <= 1 or not math.isclose(
                value, joints[right][left], rel_tol=1e-8, abs_tol=1e-10
            ):
                raise ValueError(
                    "conditional pairwise inclusion matrix must be symmetric in [0, 1]"
                )

    support: list[dict[str, Any]] = []
    if p_family < 1.0:
        support.append(
            {"probability": 1.0 - p_family, "family_included": False, "unit_indices": ()}
        )
    for choice, mass in zip(choices, masses):
        indices = tuple(choice.indices if hasattr(choice, "indices") else choice[0])
        support.append(
            {"probability": p_family * mass, "family_included": True, "unit_indices": indices}
        )
    unit_marginals = tuple(p_family * value for value in marginals)
    unit_joints = tuple(tuple(p_family * value for value in row) for row in joints)
    maximum_weight = 1.0 / min(unit_marginals)
    return {
        "support": support,
        "unit_inclusion_probabilities": unit_marginals,
        "joint_unit_inclusion_probabilities": unit_joints,
        "max_joint_importance_multiplier": maximum_weight,
        "support_probability_sum": sum(point["probability"] for point in support),
    }
