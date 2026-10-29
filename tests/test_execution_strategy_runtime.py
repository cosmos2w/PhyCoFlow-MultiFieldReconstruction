"""Focused tests for resume-safe execution strategy runtime behavior."""

from __future__ import annotations

import copy

import pytest

from phycoflow_reconstruction.training.execution_strategy.contracts import (
    CostProfile,
    PlanMeasurement,
    TaskContract,
)
from phycoflow_reconstruction.training.execution_strategy.planner import select_execution_plan
from phycoflow_reconstruction.training.execution_strategy.runtime import (
    ExecutionRuntime,
    implementation_binding,
    scientific_config_hash,
    validate_execution_strategy,
)


def _base_config() -> dict:
    return {
        "stage": "post_training",
        "source": {
            "checkpoint": "source.pt",
            "config_origins": {"model": "source_run", "post_training": "child"},
        },
        "model": {"name": "model-a", "width": 64},
        "rollout": {"steps": 4, "solver": "euler"},
        "objectives": {"coherence": {"enabled": True, "weight": 1.0}},
        "optimization": {
            "epochs": 100,
            "batch_size": 32,
            "multitask": {
                "family_scalarization": {"method": "fixed_sum"},
                "outer_aggregation": None,
                "external_policy": {
                    "method": "lagrangian",
                    "vector_reduction": "sum",
                    "objective_scale": "source_mean",
                    "lagrangian": {"penalty_interval_updates": 1},
                },
                "fidelity_budget": {
                    "metric": "endpoint_mse_model_units",
                    "relative_increase": 0.05,
                },
            },
        },
        "coherence": {
            "schedule": {
                "start_epoch": 1,
                "every_n_steps": 1,
                "weight_warmup_epochs": 0,
                "interval_rescale": False,
            },
            "families": {
                "A": {"enabled": True, "weight": 1.0},
                "B": {"enabled": True, "weight": 1.0},
                "T": {"enabled": True, "weight": 1.0},
            },
        },
        "evaluation": {"split": "validation", "max_samples": 8},
        "checkpointing": {"selection_metric": "reconstruction_mse", "every_epochs": 5},
        "output": {"experiment_name": "arm-a"},
    }


def _task(config: dict) -> TaskContract:
    return TaskContract(
        task_id="runtime-task",
        source_identity="sha256:source-identity",
        case_id="case-a",
        dataset_id="dataset-a",
        split_identity="train:split-a",
        adapter_kind="generative",
        reconstruction_identity="adapter-v1",
        normalization_identity="normalization-v1",
        objective_identity="A+B+T-v1",
        objective_rule="scalar_additive",
        component_weights=(("A", 1.0), ("B", 1.0), ("T", 1.0)),
        aggregation_rule="fixed-sum-over-c0",
        source_calibration_identity="source-calibration-v1",
        c0=1.25,
        field_bounds=(("u", 0.5, 0.1), ("v", 0.8, 0.2)),
        query_geometry_identity="geometry-v1",
        sampling_identity="batch32-noise7",
        numerical_precision="float32",
        optimizer_identity="adamw-v1",
        epoch_definition="4-updates",
        quality_contract_identity="quality-v1",
        max_epochs=100,
        provider_ids=("topology.complete_filtration.v1",),
        features=("additive_family_provider", "scalar_optimizer", "shared_forward"),
        metadata={"scientific_config_hash": scientific_config_hash(config)},
    )


def _profile(task: TaskContract, *, profiled_epoch: int = 0) -> CostProfile:
    full = PlanMeasurement("full", 10.0, 10.2)
    accelerated = PlanMeasurement(
        plan_id="nested",
        mean_epoch_cost_s=7.8,
        upper_epoch_cost_s=8.0,
        actions=("whole_family_sampling", "generalized_filtration_sampling"),
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
    )
    return CostProfile(
        task_hash=task.task_hash,
        hardware_fingerprint="cpu-runtime-fixture",
        profiled_epoch=profiled_epoch,
        max_epochs=task.max_epochs,
        measurements=(full, accelerated),
        intrinsic_noise_variance=1.0,
    )


def _bind_strategy(config: dict, *, profiled_epoch: int = 0) -> dict:
    task = _task(config)
    profile = _profile(task, profiled_epoch=profiled_epoch)
    plan = select_execution_plan(task, profile)
    assert plan.plan_id == "nested"
    config["optimization"]["execution_strategy"] = {
        "enabled": True,
        "task": task.to_dict(),
        "profile": profile.to_dict(),
        "plan": plan.to_dict(),
        "seed": 417,
    }
    return config


def _enabled_config(*, profiled_epoch: int = 0) -> dict:
    return _bind_strategy(_base_config(), profiled_epoch=profiled_epoch)


def test_scientific_hash_ignores_strategy_arm_labels_but_keeps_science():
    config = _base_config()
    initial = scientific_config_hash(config)
    changed_labels = copy.deepcopy(config)
    changed_labels["source"]["config_origins"] = {"model": "different bookkeeping"}
    changed_labels["output"]["experiment_name"] = "arm-b"
    changed_labels["optimization"]["fast_lagrangian"] = {"family_probability": 0.5}
    changed_labels["optimization"]["execution_strategy"] = {"enabled": True}
    assert scientific_config_hash(changed_labels) == initial

    for section, key, value in (
        ("model", "width", 128),
        ("rollout", "steps", 5),
        ("evaluation", "max_samples", 16),
        ("checkpointing", "every_epochs", 10),
        ("optimization", "epochs", 101),
    ):
        changed = copy.deepcopy(config)
        changed[section][key] = value
        assert scientific_config_hash(changed) != initial


def test_validator_is_default_off_and_enabled_contract_is_strict():
    config = _base_config()
    assert validate_execution_strategy(config) is False
    config["optimization"]["execution_strategy"] = {"enabled": False}
    assert validate_execution_strategy(config) is False

    enabled = _enabled_config()
    assert validate_execution_strategy(enabled) is True

    calibrated_family_balance = _base_config()
    calibrated_family_balance["coherence"]["family_balance"] = {"mode": "initial_grad_norm"}
    calibrated_family_balance = _bind_strategy(calibrated_family_balance)
    with pytest.raises(ValueError, match="family_balance.mode=none"):
        validate_execution_strategy(calibrated_family_balance)

    bad = copy.deepcopy(enabled)
    bad["optimization"]["fast_lagrangian"] = {"family_probability": 0.5}
    with pytest.raises(ValueError, match="cannot coexist"):
        validate_execution_strategy(bad)

    bad = copy.deepcopy(enabled)
    bad = _base_config()
    bad["optimization"]["multitask"]["external_policy"]["lagrangian"][
        "penalty_interval_updates"
    ] = 2
    bad = _bind_strategy(bad)
    with pytest.raises(ValueError, match="dense per-update"):
        validate_execution_strategy(bad)

    bad = copy.deepcopy(enabled)
    bad["model"]["width"] = 128
    with pytest.raises(ValueError, match="scientific_config_hash"):
        validate_execution_strategy(bad)

    # The scientific hash provides an independent contract-binding failure;
    # rebuild the task with the altered horizon to exercise the direct check.
    bad = _base_config()
    bad["optimization"]["epochs"] = 99
    bad = _bind_strategy(bad)
    with pytest.raises(ValueError, match="max_epochs"):
        validate_execution_strategy(bad)


def test_runtime_decisions_follow_only_accepted_cursors_and_round_trip():
    runtime = ExecutionRuntime(_enabled_config(), start_step=0, steps_per_epoch=4)
    decision = runtime.decision(0)
    assert decision == runtime.decision(0)
    assert decision["accepted_cursor"] == 0
    assert decision["family_probabilities"] == {"A": 1.0, "B": 1.0, "T": 0.5}
    with pytest.raises(ValueError, match="accepted cursor"):
        runtime.decision(1)

    with pytest.raises(TypeError, match="field_violations"):
        runtime.accept(0, 1, 0.1, field_violations=0.5)
    assert runtime.state.accepted_cursor == 0
    runtime.accept(0, 1, 0.1, field_violations={"u": -0.1, "v": -0.2})
    assert runtime.state.accepted_cursor == 1
    assert runtime.decision(1)["accepted_cursor"] == 1

    saved = runtime.state_dict()
    restored = ExecutionRuntime(_enabled_config(), state=saved, start_step=1, steps_per_epoch=4)
    assert restored.state_dict() == saved


def test_resume_rejects_modified_current_plan_under_valid_task_and_profile_bindings():
    config = _enabled_config()
    runtime = ExecutionRuntime(config, start_step=0, steps_per_epoch=4)
    saved = runtime.state_dict()
    saved["current_plan"]["family_probabilities"]["T"] = 0.75
    saved["current_plan"]["max_joint_importance_multiplier"] = 1.0 / (0.75 * 0.5)

    with pytest.raises(ValueError, match="differs from the configured measured candidate"):
        ExecutionRuntime(config, state=saved, start_step=0, steps_per_epoch=4)


def test_sustained_train_field_violations_switch_to_full_and_resume_sticky():
    config = _enabled_config()
    runtime = ExecutionRuntime(config, steps_per_epoch=4)
    for step in range(36):
        epoch = step // 4 + 1
        epoch9_values = (0.2, 0.2, -0.1, -0.1)
        violation = epoch9_values[step % 4] if epoch == 9 else -0.1
        runtime.accept(step, epoch, 0.2, field_violations={"u": violation})
    assert runtime.state_dict()["risk_streaks"]["u"] == 1
    assert runtime.refresh(9).plan_id == "nested"
    for step in range(36, 40):
        runtime.accept(step, 10, 0.2, field_violations={"u": 0.2})
    recovered = runtime.refresh(10)
    assert recovered.plan_id == "full"
    assert recovered.no_action
    assert recovered.family_probabilities == {}
    assert runtime.state.plan_id == "full"
    assert runtime.state_dict()["full_recovery_intervals"] == [{"start_epoch": 10, "end_epoch": 20}]

    restored = ExecutionRuntime(
        config,
        state=runtime.state_dict(),
        start_step=40,
        steps_per_epoch=4,
    )
    assert restored.plan.plan_id == "full"
    assert restored.decision(40)["family_probabilities"] == {}
    for step in range(40, 80):
        restored.accept(step, step // 4 + 1, 0.2)
    assert restored.refresh(20).plan_id == "full"
    restored_again = ExecutionRuntime(
        config,
        state=restored.state_dict(),
        start_step=80,
        steps_per_epoch=4,
    )
    assert restored_again.plan.plan_id == "full"


def test_only_newer_train_profile_after_full_recovery_can_reenable_sampling():
    old_config = _enabled_config()
    runtime = ExecutionRuntime(old_config, steps_per_epoch=4)
    for step in range(40):
        epoch = step // 4 + 1
        runtime.accept(step, epoch, 0.2, field_violations={"u": epoch >= 9})
    runtime.refresh(10)
    for step in range(40, 80):
        runtime.accept(step, step // 4 + 1, 0.2)

    task = TaskContract.from_mapping(old_config["optimization"]["execution_strategy"]["task"])
    profile = _profile(task, profiled_epoch=20)
    plan = select_execution_plan(task, profile)
    new_config = copy.deepcopy(old_config)
    new_config["optimization"]["execution_strategy"]["profile"] = profile.to_dict()
    new_config["optimization"]["execution_strategy"]["plan"] = plan.to_dict()
    resumed = ExecutionRuntime(
        new_config,
        state=runtime.state_dict(),
        start_step=80,
        steps_per_epoch=4,
    )
    assert resumed.plan.plan_id == "nested"
    assert resumed.state.accepted_cursor == 80


def test_train_noise_probe_charges_overhead_and_falls_back_when_over_budget():
    runtime = ExecutionRuntime(_enabled_config(), steps_per_epoch=4)
    with pytest.raises(ValueError, match="TRAIN"):
        runtime.observe_audit(10, 1.5, 3.0, 1.0, None, split="validation")
    runtime.observe_audit(9, 1.5, 3.0, 1.0, None)
    assert runtime.state.charged_overhead_s == 0

    runtime.observe_audit(10, 1.5, 3.0, 1.0, None)
    assert runtime.state.charged_overhead_s == pytest.approx(1.5)
    assert runtime.state_dict()["monitor_overhead_s"] == pytest.approx(1.5)
    assert runtime.refresh(10).plan_id == "full"


def test_refresh_falls_back_when_actual_monitor_overhead_exceeds_five_percent():
    runtime = ExecutionRuntime(_enabled_config(), steps_per_epoch=1)
    for step in range(10):
        runtime.accept(step, step + 1, 1.0)
    runtime.observe_audit(10, 0.6, 1.0, 1.0, None)
    assert runtime.state_dict()["monitor_overhead_s"] == pytest.approx(0.6)
    assert runtime.state_dict()["accepted_training_cost_s"] == pytest.approx(10.0)
    assert runtime.refresh(10).plan_id == "full"
    assert "overhead exceeded 5%" in runtime.plan.selected_reason


def test_full_metrics_require_scheduled_dense_train_audit_and_detect_risk():
    runtime = ExecutionRuntime(_enabled_config(), steps_per_epoch=4)
    for step in range(80):
        runtime.accept(step, step // 4 + 1, 0.5, field_violations={"u": -0.1, "v": -0.1})
    with pytest.raises(ValueError, match=r"scheduled 20\+"):
        runtime.observe_audit(10, 1.0, 1.0, 1.0, {"u": 0.6, "v": 0.8})
    with pytest.raises(ValueError, match="exactly the task"):
        runtime.observe_audit(20, 1.0, 1.0, 1.0, {"u": 0.6})

    runtime.observe_audit(20, 1.0, 1.0, 1.0, {"u": 0.6, "v": 0.8})
    assert runtime.state.last_full_audit_epoch == 20
    assert runtime.refresh(20).plan_id == "nested"
    runtime.observe_audit(40, 1.0, 1.0, 1.0, {"u": 0.6, "v": 0.8})
    assert runtime.refresh(40).plan_id == "full"


def test_bind_reference_matches_ordered_fields_bounds_scales_and_c0():
    runtime = ExecutionRuntime(_enabled_config())
    reference = {
        "field_names": ["u", "v"],
        "upper_bounds": [0.5, 0.8],
        "denominators": [0.1, 0.2],
        "c0": 1.25,
    }
    runtime.bind_reference(reference)
    saved = runtime.state_dict()
    restored = ExecutionRuntime(_enabled_config(), state=saved)
    restored.bind_reference(reference)
    with pytest.raises(ValueError, match="field order"):
        restored.bind_reference(reference | {"field_names": ["v", "u"]})
    with pytest.raises(ValueError, match="C0"):
        restored.bind_reference(reference | {"c0": 2.0})


def test_implementation_binding_hashes_runtime_and_scientific_dispatch_files():
    binding = implementation_binding()
    assert "training/execution_strategy/runtime.py" in binding
    assert "training/execution_strategy/contracts.py" in binding
    assert "training/fast_lagrangian.py" in binding
    assert "training/post_training.py" in binding
    assert "training/execution_strategy/audit.py" in binding
    assert "coherence/families/topology/filtration_sampling.py" in binding
    # Installed-package recovery must not depend on research checkout scripts.
    assert all(not path.startswith("scripts/") for path in binding)
    assert all(len(value) == 64 for value in binding.values())


def test_all_full_plan_remains_a_valid_default_scalar_execution():
    config = _base_config()
    task = _task(config)
    profile = CostProfile(
        task_hash=task.task_hash,
        hardware_fingerprint="cpu-full-fixture",
        profiled_epoch=0,
        max_epochs=task.max_epochs,
        measurements=(PlanMeasurement("full", 10.0, 10.2),),
        intrinsic_noise_variance=1.0,
    )
    plan = select_execution_plan(task, profile)
    assert plan.no_action and plan.plan_id == "full"
    config["optimization"]["execution_strategy"] = {
        "enabled": True,
        "task": task.to_dict(),
        "profile": profile.to_dict(),
        "plan": plan.to_dict(),
        "seed": 417,
    }
    runtime = ExecutionRuntime(config)
    assert runtime.decision(0)["family_probabilities"] == {}
    assert runtime.decision(0)["family_inclusions"] == {}
