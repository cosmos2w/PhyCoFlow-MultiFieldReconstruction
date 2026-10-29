from __future__ import annotations

import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from phycoflow_reconstruction.training.execution_strategy import (
    CostProfile,
    ExecutionPlan,
    ExecutionState,
    PlanMeasurement,
    ProfileBudget,
    TaskContract,
    action_registry,
    compose_nested_filtration_support,
    full_audit_due,
    manifest_for_task,
    policy_refresh_due,
    select_execution_plan,
    state_fingerprint,
)


def make_task(**updates) -> TaskContract:
    values = {
        "task_id": "task-g",
        "source_identity": "sha256:source",
        "case_id": "combustion",
        "dataset_id": "dataset-v1",
        "split_identity": "train:ids-v1",
        "adapter_kind": "generative",
        "reconstruction_identity": "adapter-v3",
        "normalization_identity": "norm-v1",
        "objective_identity": "A+B+T-v2",
        "objective_rule": "scalar_additive",
        "component_weights": (("A", 1.0), ("B", 1.0), ("T", 1.0)),
        "aggregation_rule": "weighted-sum-over-C0",
        "source_calibration_identity": "source-calibration-v1",
        "c0": 0.25,
        "field_bounds": (("u", 0.05, 1.0), ("v", 0.05, 1.0)),
        "query_geometry_identity": "queries-v1",
        "sampling_identity": "batch32-noise7",
        "numerical_precision": "float32",
        "optimizer_identity": "scalar-adamw-v1",
        "epoch_definition": "38-batches",
        "quality_contract_identity": "quality-v1",
        "max_epochs": 100,
        "provider_ids": ("topology.complete_filtration.v1",),
        "features": ("shared_forward", "additive_family_provider", "scalar_optimizer"),
        "metadata": {"sensor_panel": "fixed-v1"},
    }
    values.update(updates)
    return TaskContract(**values)


def make_profile(
    task: TaskContract, *, candidate_updates=None, profile_overhead_s=20.0
) -> CostProfile:
    full = PlanMeasurement(
        plan_id="full",
        mean_epoch_cost_s=10.0,
        upper_epoch_cost_s=10.2,
        measured_split="TRAIN",
    )
    candidate = PlanMeasurement(
        plan_id="nested",
        mean_epoch_cost_s=7.8,
        upper_epoch_cost_s=8.0,
        actions=("whole_family_sampling", "generalized_filtration_sampling"),
        acquisition_overhead_s=10.0,
        refresh_overhead_s=1.0,
        audit_overhead_s=2.0,
        extra_variance=0.5,
        variance_method="exact_joint_support",
        estimator_kind="unbiased_scalar_estimator",
        family_probabilities={"A": 1.0, "B": 1.0, "T": 0.5},
        within_family={
            "T": {
                "provider_id": "topology.complete_filtration.v1",
                "estimator": "polarity",
                "conditional_inclusion_probability": 0.5,
            }
        },
        measured_split="TRAIN",
    )
    if candidate_updates:
        candidate = PlanMeasurement.from_mapping(candidate.to_dict() | candidate_updates)
    return CostProfile(
        task_hash=task.task_hash,
        hardware_fingerprint="cpu-fixture-v1",
        profiled_epoch=0,
        max_epochs=task.max_epochs,
        measurements=(full, candidate),
        intrinsic_noise_variance=1.0,
        profile_overhead_s=profile_overhead_s,
    )


def test_task_contract_hash_and_manifest_are_stable_and_config_only():
    task = make_task()
    restored = TaskContract.from_mapping(task.to_dict())
    assert restored.task_hash == task.task_hash
    assert make_task(c0=0.5).task_hash != task.task_hash

    manifest = manifest_for_task(task.to_dict())
    assert manifest["task_hash"] == task.task_hash
    assert manifest["cost_profile_request"]["status"] == "unexecuted"
    assert manifest["cost_profile_request"]["max_candidates"] == 6
    assert {item["action_id"] for item in manifest["capabilities"]} >= {
        "full",
        "whole_family_sampling",
        "generalized_filtration_sampling",
    }

    config_only = task.to_dict()
    config_only.pop("task_hash")
    config_only.pop("source_calibration_identity")
    config_only.pop("c0")
    config_only.pop("field_bounds")
    config_only.pop("calibration_status")
    pending = TaskContract.from_mapping(config_only)
    assert pending.calibration_status == "pending"
    assert pending.c0 is None
    assert pending.field_bounds == ()
    pending_manifest = manifest_for_task(config_only)
    assert pending_manifest["cost_profile_request"]["calibration_required_before_profile"] is True
    with pytest.raises(ValueError, match="completed SOURCE TRAIN calibration"):
        select_execution_plan(pending, make_profile(pending))


def test_registry_quarantines_config_reuse_and_rejects_semantic_changes():
    config_task = make_task(objective_rule="config", provider_ids=(), features=())
    capabilities = {item.action_id: item for item in action_registry(config_task)}
    assert capabilities["full"].applicable
    assert not capabilities["whole_family_sampling"].applicable
    assert not capabilities["generalized_filtration_sampling"].applicable
    assert not capabilities["periodic_config_reuse"].applicable
    assert capabilities["periodic_config_reuse"].evidence_status == "quarantined"
    assert not capabilities["reduced_resolution"].applicable
    assert capabilities["reduced_resolution"].preservation_level == "unsupported"


def test_planner_selects_measured_nested_plan_only_within_noise_and_overhead_limits():
    task = make_task()
    profile = make_profile(task)
    plan = select_execution_plan(task, profile)
    assert plan.plan_id == "nested"
    assert plan.max_joint_importance_multiplier == 4.0
    assert plan.cost_savings_ratio >= 0.05
    assert plan.extra_variance <= plan.variance_ceiling
    assert plan.predicted_total_s < plan.predicted_full_total_s

    execution = plan.to_execution_mapping(accepted_cursor=17, rng_seed=91)
    repeated = plan.to_execution_mapping(accepted_cursor=17, rng_seed=91)
    assert execution == repeated
    assert execution["accepted_cursor"] == 17
    assert execution["family_probabilities"] == {"A": 1.0, "B": 1.0, "T": 0.5}
    assert execution["within_family"]["T"]["provider_id"] == "topology.complete_filtration.v1"
    assert execution["within_family"]["T"]["estimator"] == "polarity"
    assert execution["dense_fidelity_required"] is True


@pytest.mark.parametrize(
    ("updates", "expected_reason"),
    [
        ({"extra_variance": 1.1}, "exceeds"),
        ({"upper_epoch_cost_s": 10.1, "mean_epoch_cost_s": 9.9}, "break even"),
        (
            {"upper_epoch_cost_s": 9.9, "mean_epoch_cost_s": 9.8, "acquisition_overhead_s": 300.0},
            "break even",
        ),
    ],
)
def test_planner_falls_back_to_full_for_noise_or_unamortized_candidates(updates, expected_reason):
    task = make_task()
    plan = select_execution_plan(task, make_profile(task, candidate_updates=updates))
    assert plan.no_action
    assert plan.plan_id == "full"
    assert expected_reason in plan.rejection_reasons["nested"]


def test_planner_rejects_nested_multiplier_above_four_and_empty_support():
    task = make_task()
    too_sparse = make_profile(
        task,
        candidate_updates={
            "family_probabilities": {"A": 1.0, "B": 1.0, "T": 0.25},
            "within_family": {
                "T": {
                    "provider_id": "topology.complete_filtration.v1",
                    "estimator": "polarity",
                    "conditional_inclusion_probability": 0.5,
                }
            },
        },
    )
    plan = select_execution_plan(task, too_sparse)
    assert plan.no_action
    assert "exceeds" in plan.rejection_reasons["nested"]

    with pytest.raises(ValueError, match="always-included|joint support"):
        PlanMeasurement(
            plan_id="empty-risk",
            mean_epoch_cost_s=1.0,
            upper_epoch_cost_s=1.0,
            actions=("whole_family_sampling",),
            estimator_kind="unbiased_scalar_estimator",
            family_probabilities={"A": 0.5, "B": 0.5},
            variance_method="exact_joint_support",
        )


def test_planner_uses_only_train_cost_and_noise_profiles():
    task = make_task()
    with pytest.raises(ValueError, match="TRAIN only"):
        CostProfile(
            task_hash=task.task_hash,
            hardware_fingerprint="fixture",
            profiled_epoch=0,
            max_epochs=100,
            measurements=(PlanMeasurement("full", 1.0, 1.0),),
            intrinsic_noise_variance=0.1,
            noise_split="validation",
        )


def test_plan_deserialization_binds_nested_probabilities_and_joint_support():
    task = make_task()
    plan = select_execution_plan(task, make_profile(task))
    serialized = plan.to_dict()
    restored = type(plan).from_mapping(serialized)
    assert restored.to_dict() == serialized

    tampered = copy.deepcopy(serialized)
    tampered["max_joint_importance_multiplier"] = 2.0
    with pytest.raises(ValueError, match="does not match"):
        ExecutionPlan.from_mapping(tampered)

    tampered = copy.deepcopy(serialized)
    tampered["within_family"]["T"]["provider_id"] = "unknown.provider.v9"
    forged = ExecutionPlan.from_mapping(tampered)
    with pytest.raises(ValueError, match="unregistered filtration provider"):
        forged.validate_for_task(task)

    tampered = copy.deepcopy(serialized)
    tampered["within_family"]["T"]["estimator"] = "legacy-shortcut"
    forged = ExecutionPlan.from_mapping(tampered)
    with pytest.raises(ValueError, match="unsupported filtration estimator"):
        forged.validate_for_task(task)


def test_planner_rejects_unmeasured_or_unbound_nested_provider_candidate():
    task = make_task()
    unmeasured = select_execution_plan(
        task,
        make_profile(task, candidate_updates={"evidence_status": "unmeasured"}),
    )
    assert unmeasured.no_action
    assert "not measured" in unmeasured.rejection_reasons["nested"]

    unbound_provider = select_execution_plan(
        task,
        make_profile(
            task,
            candidate_updates={
                "within_family": {
                    "T": {
                        "provider_id": "unknown.provider.v9",
                        "estimator": "polarity",
                        "conditional_inclusion_probability": 0.5,
                    }
                }
            },
        ),
    )
    assert unbound_provider.no_action
    assert "unregistered provider" in unbound_provider.rejection_reasons["nested"]


def test_stochastic_candidates_require_variance_evidence():
    with pytest.raises(ValueError, match="variance evidence"):
        PlanMeasurement(
            plan_id="missing-variance-evidence",
            mean_epoch_cost_s=1.0,
            upper_epoch_cost_s=1.0,
            actions=("whole_family_sampling",),
            estimator_kind="unbiased_scalar_estimator",
            family_probabilities={"A": 1.0},
            variance_method="none",
        )


def test_nested_filtration_support_preserves_correlations_and_outer_probability():
    choices = (
        SimpleNamespace(indices=(0, 1), inclusion_probabilities=(0.5, 0.5)),
        SimpleNamespace(indices=(2, 3), inclusion_probabilities=(0.5, 0.5)),
    )
    support = SimpleNamespace(
        choices=choices,
        choice_probabilities=(0.5, 0.5),
        inclusion_probabilities=(0.5, 0.5, 0.5, 0.5),
        joint_inclusion_probabilities=(
            (0.5, 0.5, 0.0, 0.0),
            (0.5, 0.5, 0.0, 0.0),
            (0.0, 0.0, 0.5, 0.5),
            (0.0, 0.0, 0.5, 0.5),
        ),
    )
    composed = compose_nested_filtration_support(0.5, support)
    assert composed["unit_inclusion_probabilities"] == (0.25, 0.25, 0.25, 0.25)
    assert composed["joint_unit_inclusion_probabilities"][0][1] == 0.25
    assert composed["joint_unit_inclusion_probabilities"][0][2] == 0.0
    assert composed["max_joint_importance_multiplier"] == 4.0
    assert composed["support_probability_sum"] == pytest.approx(1.0)


def test_execution_state_binds_task_plan_rng_and_only_advances_accepted_cursor():
    task = make_task()
    plan = select_execution_plan(task, make_profile(task))
    state = ExecutionState.for_plan(task, plan, rng_seed=104, accepted_cursor=3)
    expected_seed = state.counter_seed("family", accepted_cursor=3)
    assert state.counter_seed("family", accepted_cursor=3) == expected_seed
    with pytest.raises(ValueError, match="TRAIN"):
        state.observe_train(epoch=1, cost_s=2.0, split="validation")
    state.observe_train(epoch=1, cost_s=2.0, noise_variance=0.1)
    state.commit_accepted_update(cursor_before_update=3)
    assert state.accepted_cursor == 4
    restored = ExecutionState.from_mapping(state.to_dict())
    restored.validate_resume(
        task_hash=task.task_hash,
        plan_id=plan.plan_id,
        rng_identity=state.rng_identity,
        accepted_cursor=4,
    )
    with pytest.raises(ValueError, match="binding mismatch"):
        restored.validate_resume(task_hash="other", plan_id=plan.plan_id)


def test_policy_refresh_and_full_audit_minimum_intervals():
    task = make_task()
    plan = select_execution_plan(task, make_profile(task))
    state = ExecutionState.for_plan(task, plan, rng_seed=2)
    state.last_refresh_epoch = 10
    state.last_full_audit_epoch = 20
    assert not policy_refresh_due(state, 19)
    assert policy_refresh_due(state, 20)
    assert not full_audit_due(state, 39)
    assert full_audit_due(state, 40)
    with pytest.raises(ValueError, match="at least 10"):
        policy_refresh_due(state, 20, interval_epochs=5)


def test_profile_budget_restores_model_all_rng_streams_and_charges_candidate():
    model = torch.nn.BatchNorm1d(2)
    model.train()
    source_fingerprint = state_fingerprint(model)
    generator = torch.Generator().manual_seed(31)
    random.seed(30)
    np.random.seed(29)
    torch.manual_seed(28)
    python_state = random.getstate()
    numpy_state = copy.deepcopy(np.random.get_state())
    torch_state = torch.get_rng_state().clone()
    generator_state = generator.get_state().clone()
    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(1)
    expected_generator = torch.rand(1, generator=generator)
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    generator.set_state(generator_state)

    budget = ProfileBudget(max_candidates=1)

    def mutating_profile():
        with torch.no_grad():
            model.weight.add_(4)
        model(torch.ones((3, 2)))
        random.random()
        np.random.random()
        torch.rand(1)
        torch.rand(1, generator=generator)
        model.eval()
        return "profiled"

    measurement = budget.measure(
        "whole-plan",
        model,
        mutating_profile,
        source_identity="sha256:source",
        task_hash="task-hash",
        generators=(generator,),
    )
    assert measurement.succeeded
    assert measurement.value == "profiled"
    assert measurement.source_identity == "sha256:source"
    assert measurement.work_seconds >= 0
    assert measurement.charged_seconds > 0
    assert budget.charged_overhead_s == pytest.approx(measurement.charged_seconds)
    assert state_fingerprint(model) == source_fingerprint
    assert model.training is True
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    assert torch.equal(torch.rand(1), expected_torch)
    assert torch.equal(torch.rand(1, generator=generator), expected_generator)
    with pytest.raises(RuntimeError, match="limit"):
        budget.measure(
            "second",
            model,
            lambda: None,
            source_identity="sha256:source",
            task_hash="task-hash",
        )
