"""Validate stage separation plus high-value model and observation invariants."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .schema import STAGE_SCHEMAS

COMMON_KEYS = {
    "stage",
    "case",
    "dataset",
    "model",
    "observations",
    "optimization",
    "runtime",
    "output",
    "evaluation",
    "source_run",
    "source_checkpoint",
    "coherence",
    "physics",
    "notes",
    "source",
    "inherit_base_config",
    "objectives",
    "rollout",
    "observation_consistency",
    "trainable",
}


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown {path} keys: {unknown}")


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return value


def _validate_common_sections(config: Mapping[str, Any]) -> None:
    dataset = _require_mapping(config, "dataset")
    _reject_unknown(
        dataset,
        {
            "path",
            "split",
            "reconstruction_unit",
            "field_names",
            "field_units",
            "normalization",
            "time_stride",
            "benchmark_eligible",
            "allow_nonbenchmark",
            "coordinate_dim",
            "grid_shape",
            "grid_mapping_verified",
            "coordinate_reorder",
            "include_temporal_derivative",
        },
        "dataset",
    )
    if not dataset.get("path"):
        raise ValueError("dataset.path is required")
    if dataset.get("reconstruction_unit", "snapshot") not in {
        "snapshot",
        "space_time_trajectory",
    }:
        raise ValueError("dataset.reconstruction_unit is invalid")
    if int(dataset.get("time_stride", 1)) < 1:
        raise ValueError("dataset.time_stride must be positive")
    names = dataset.get("field_names")
    units = dataset.get("field_units")
    if names is not None and (not names or len(set(names)) != len(names)):
        raise ValueError("dataset.field_names must be non-empty and unique")
    if names is not None and units is not None and len(names) != len(units):
        raise ValueError("dataset.field_units must align with field_names")

    observations = _require_mapping(config, "observations")
    _reject_unknown(
        observations,
        {
            "protocol",
            "seed",
            "fields",
            "shared_locations",
            "spatial_downsample_ratio",
            "temporal_downsample_ratio",
            "phase",
            "requires_validated_grid_mapping",
        },
        "observations",
    )
    if observations.get("protocol", "random_uniform") not in {
        "random_uniform",
        "structured_stride",
        "uniform_spacetime_stride",
    }:
        raise ValueError("observations.protocol is invalid")
    fields = observations.get("fields", {})
    if not isinstance(fields, Mapping) or not fields:
        raise ValueError("observations.fields must be a non-empty mapping")
    for name, settings in fields.items():
        if not isinstance(settings, Mapping):
            raise TypeError(f"observations.fields.{name} must be a mapping")
        _reject_unknown(
            settings,
            {"count", "count_min", "count_max"},
            f"observations.fields.{name}",
        )
        if "count" in settings:
            if set(settings) != {"count"} or int(settings["count"]) < 1:
                raise ValueError(
                    f"observations.fields.{name}.count must be the sole positive count setting"
                )
        else:
            low = int(settings.get("count_min", 0))
            high = int(settings.get("count_max", 0))
            if low < 1 or high < low:
                raise ValueError(f"observations.fields.{name} has an invalid count range")

    runtime = _require_mapping(config, "runtime")
    _reject_unknown(runtime, {"seed", "device", "deterministic", "num_workers"}, "runtime")
    if int(runtime.get("num_workers", 0)) < 0:
        raise ValueError("runtime.num_workers must be non-negative")
    output = _require_mapping(config, "output")
    _reject_unknown(output, {"experiment_name"}, "output")
    if not output.get("experiment_name"):
        raise ValueError("output.experiment_name is required")
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, Mapping):
        raise TypeError("evaluation must be a mapping")
    _reject_unknown(
        evaluation,
        {"split", "max_samples", "query_points", "generation_steps", "seed"},
        "evaluation",
    )
    for key in ("max_samples", "query_points", "generation_steps"):
        if key in evaluation and int(evaluation[key]) < 1:
            raise ValueError(f"evaluation.{key} must be positive")


def _validate_optimization_values(settings: Mapping[str, Any]) -> None:
    for key in ("epochs", "batch_size"):
        if key in settings and int(settings[key]) < 1:
            raise ValueError(f"optimization.{key} must be positive")
    if "lr" in settings and float(settings["lr"]) <= 0:
        raise ValueError("optimization.lr must be positive")
    if float(settings.get("weight_decay", 0.0)) < 0:
        raise ValueError("optimization.weight_decay must be non-negative")
    if settings.get("grad_clip") is not None and float(settings["grad_clip"]) <= 0:
        raise ValueError("optimization.grad_clip must be positive when provided")


def _validate_base_training(config: Mapping[str, Any]) -> None:
    optimization = _require_mapping(config, "optimization")
    _reject_unknown(
        optimization,
        {"epochs", "batch_size", "lr", "weight_decay", "grad_clip"},
        "optimization",
    )
    _validate_optimization_values(optimization)


def _validate_post_training(config: Mapping[str, Any]) -> None:
    if "coherence" in config and "physics" in config:
        raise ValueError("post_training must select exactly one of coherence or physics")
    if not config.get("source_run") or not config.get("source_checkpoint"):
        raise ValueError("post_training requires non-empty source_run and source_checkpoint")
    source = config.get("source", {})
    _reject_unknown(
        source,
        {
            "kind",
            "channel_mapping",
            "allow_integration_source",
            "inherited_base_keys",
            "config_origins",
        },
        "source",
    )
    source_kind = source.get("kind", "native_run")
    if source_kind not in {"native_run", "legacy_demo50"}:
        raise ValueError("source.kind must be native_run or legacy_demo50")
    if source_kind == "legacy_demo50" and config["model"].get("name") != "legacy_demo50":
        raise ValueError("legacy_demo50 source requires model.name=legacy_demo50")
    if "coherence" not in config:
        _validate_physics_settings(config["physics"])
        required = {"objectives", "rollout", "observation_consistency", "trainable"}
        missing = sorted(required - config.keys())
        if missing:
            raise ValueError(f"physics post_training is missing keys: {missing}")
        _reject_unknown(config["objectives"], {"data_retention", "physics"}, "objectives")
        for name in ("data_retention", "physics"):
            settings = config["objectives"].get(name)
            if not isinstance(settings, Mapping):
                raise TypeError(f"objectives.{name} must be a mapping")
            _reject_unknown(settings, {"enabled", "weight"}, f"objectives.{name}")
            if float(settings.get("weight", 0.0)) < 0:
                raise ValueError(f"objectives.{name}.weight must be non-negative")
        if not any(
            bool(config["objectives"][name].get("enabled", True))
            and float(config["objectives"][name].get("weight", 0.0)) > 0
            for name in ("data_retention", "physics")
        ):
            raise ValueError("physics post_training requires a positive enabled objective")
        _reject_unknown(config["rollout"], {"steps", "solver"}, "rollout")
        if int(config["rollout"].get("steps", 0)) < 1 or config["rollout"].get("solver") not in {
            "euler",
            "heun",
        }:
            raise ValueError("rollout requires steps>=1 and solver=euler or heun")
        _reject_unknown(
            config["observation_consistency"],
            {"mode", "strength", "sigma", "schedule_power", "final_clamp", "chunk_size"},
            "observation_consistency",
        )
        if config["observation_consistency"].get("mode") not in {
            "none",
            "hard",
            "endpoint",
            "endpoint_smooth",
        }:
            raise ValueError("invalid observation_consistency.mode")
        _reject_unknown(config["trainable"], {"scope", "modules"}, "trainable")
        if config["trainable"].get("scope", "full_model") not in {"full_model", "named_modules"}:
            raise ValueError("trainable.scope must be full_model or named_modules")
        _reject_unknown(
            config["optimization"],
            {
                "epochs",
                "batch_size",
                "lr",
                "weight_decay",
                "grad_clip",
                "gradient_balance",
                "config_missing_behavior",
            },
            "optimization",
        )
        _validate_optimization_values(config["optimization"])
        if config["optimization"].get("gradient_balance", "weighted_sum") not in {
            "weighted_sum",
            "config",
        }:
            raise ValueError("optimization.gradient_balance must be weighted_sum or config")
        _reject_unknown(
            config["runtime"], {"seed", "device", "deterministic", "num_workers"}, "runtime"
        )
        _reject_unknown(config["output"], {"experiment_name"}, "output")
        return
    required_phase5 = {"objectives", "rollout", "observation_consistency", "trainable"}
    missing_phase5 = sorted(required_phase5 - config.keys())
    if missing_phase5:
        raise ValueError(f"data-driven post_training is missing keys: {missing_phase5}")

    objectives = config["objectives"]
    _reject_unknown(objectives, {"data_retention", "coherence"}, "objectives")
    for name in ("data_retention", "coherence"):
        settings = objectives.get(name)
        if not isinstance(settings, Mapping):
            raise TypeError(f"objectives.{name} must be a mapping")
        _reject_unknown(settings, {"enabled", "weight"}, f"objectives.{name}")
        if float(settings.get("weight", 0.0)) < 0:
            raise ValueError(f"objectives.{name}.weight must be non-negative")
    if not any(
        bool(objectives[name].get("enabled", True))
        and float(objectives[name].get("weight", 0.0)) > 0
        for name in ("data_retention", "coherence")
    ):
        raise ValueError("post_training must enable at least one positive-weight objective")

    coherence = config["coherence"]
    _reject_unknown(coherence, {"schedule", "compute_budget", "families"}, "coherence")
    schedule = coherence.get("schedule", {})
    _reject_unknown(
        schedule,
        {"start_epoch", "every_n_steps", "weight_warmup_epochs", "interval_rescale"},
        "coherence.schedule",
    )
    if int(schedule.get("start_epoch", 1)) < 1 or int(schedule.get("every_n_steps", 1)) < 1:
        raise ValueError("coherence schedule start_epoch/every_n_steps must be positive")
    compute = coherence.get("compute_budget", {})
    _reject_unknown(compute, {"batch_size", "point_count"}, "coherence.compute_budget")
    if int(compute.get("batch_size", 1)) < 1 or int(compute.get("point_count", 2)) < 2:
        raise ValueError("coherence compute budget requires batch_size>=1 and point_count>=2")

    families = coherence.get("families", {})
    if set(families) != {"global_distribution"}:
        raise ValueError("Phase-5 post-training must configure exactly global_distribution")
    family = families["global_distribution"]
    _reject_unknown(
        family,
        {"enabled", "target_use", "units", "fields", "reference_bank", "components"},
        "coherence.families.global_distribution",
    )
    target_use = family.get("target_use", "training_reference")
    if target_use not in {"training_reference", "paired_supervised"}:
        raise ValueError(
            "global_distribution.target_use must be training_reference or paired_supervised"
        )
    reference = family.get("reference_bank", {})
    _reject_unknown(
        reference, {"enabled", "path", "max_samples", "points_per_sample", "seed"}, "reference_bank"
    )
    if target_use == "training_reference" and not bool(reference.get("enabled", True)):
        raise ValueError("training_reference coherence requires an enabled reference bank")
    if target_use == "training_reference":
        if (
            int(reference.get("max_samples", 0)) < 1
            or int(reference.get("points_per_sample", 0)) < 2
        ):
            raise ValueError("reference bank requires max_samples>=1 and points_per_sample>=2")
        if int(reference["points_per_sample"]) != int(compute["point_count"]):
            raise ValueError("reference-bank and coherence compute point counts must match")
    components = family.get("components", {})
    if not components:
        raise ValueError("global_distribution.components cannot be empty")
    component_keys = {
        "self": {"enabled", "weight", "channel_weights"},
        "mutual": {"enabled", "weight", "pairs", "directions", "seed"},
        "cross": {"enabled", "weight", "directions", "top_fraction", "seed", "include_axes", "qmc"},
    }
    _reject_unknown(components, set(component_keys), "global_distribution.components")
    for name, settings in components.items():
        _reject_unknown(settings, component_keys[name], f"global_distribution.components.{name}")

    rollout = config["rollout"]
    _reject_unknown(rollout, {"steps", "solver"}, "rollout")
    if int(rollout.get("steps", 0)) < 1 or rollout.get("solver") not in {"euler", "heun"}:
        raise ValueError("rollout requires steps>=1 and solver=euler or heun")
    observation = config["observation_consistency"]
    _reject_unknown(
        observation,
        {"mode", "strength", "sigma", "schedule_power", "final_clamp", "chunk_size"},
        "observation_consistency",
    )
    if observation.get("mode") not in {"none", "hard", "endpoint", "endpoint_smooth"}:
        raise ValueError("invalid observation_consistency.mode")
    trainable = config["trainable"]
    _reject_unknown(trainable, {"scope", "modules"}, "trainable")
    if trainable.get("scope", "full_model") not in {"full_model", "named_modules"}:
        raise ValueError("trainable.scope must be full_model or named_modules")
    if trainable.get("scope") == "named_modules" and not trainable.get("modules"):
        raise ValueError("named_modules scope requires trainable.modules")
    optimization = config["optimization"]
    _reject_unknown(
        optimization,
        {
            "epochs",
            "batch_size",
            "train_fraction",
            "lr",
            "weight_decay",
            "grad_clip",
            "gradient_balance",
            "config_missing_behavior",
            "config_data_grad_scale",
            "config_coherence_grad_scale",
        },
        "optimization",
    )
    _validate_optimization_values(optimization)
    gradient_mode = optimization.get("gradient_balance", "weighted_sum")
    if gradient_mode not in {"weighted_sum", "config"}:
        raise ValueError("optimization.gradient_balance must be weighted_sum or config")
    if optimization.get("config_missing_behavior", "error") not in {"error", "weighted_sum"}:
        raise ValueError("optimization.config_missing_behavior must be error or weighted_sum")
    _reject_unknown(
        config["runtime"], {"seed", "device", "deterministic", "num_workers"}, "runtime"
    )
    _reject_unknown(
        config.get("evaluation", {}),
        {"split", "max_samples", "query_points", "generation_steps", "seed"},
        "evaluation",
    )
    _reject_unknown(config["output"], {"experiment_name"}, "output")


def _validate_physics_settings(settings: Mapping[str, Any]) -> None:
    _reject_unknown(
        settings,
        {"provider", "domain_length", "temporal_derivative_source", "weights"},
        "physics",
    )
    if settings.get("temporal_derivative_source", "paired_finite_difference") != (
        "paired_finite_difference"
    ):
        raise ValueError("physics temporal derivative must be paired_finite_difference")
    weights = settings.get("weights", {})
    if not isinstance(weights, Mapping) or any(float(value) < 0 for value in weights.values()):
        raise ValueError("physics.weights must be a non-negative mapping")


def _validate_direct_physics(config: Mapping[str, Any]) -> None:
    if config["model"].get("name") != "pinn":
        raise ValueError("first-release direct_physics requires model.name=pinn")
    _validate_physics_settings(config["physics"])
    _reject_unknown(
        config["optimization"],
        {"epochs", "batch_size", "lr", "weight_decay", "grad_clip"},
        "optimization",
    )
    _validate_optimization_values(config["optimization"])
    _reject_unknown(
        config["runtime"], {"seed", "device", "deterministic", "num_workers"}, "runtime"
    )
    _reject_unknown(config["output"], {"experiment_name"}, "output")


def validate_config(config: Mapping[str, Any]) -> None:
    stage = config.get("stage")
    if stage not in STAGE_SCHEMAS:
        raise ValueError(f"stage must be one of {sorted(STAGE_SCHEMAS)}, got {stage!r}")
    schema = STAGE_SCHEMAS[stage]
    missing = sorted(schema.required - config.keys())
    forbidden = sorted(schema.forbidden & config.keys())
    unknown = sorted(set(config) - COMMON_KEYS)
    if missing:
        raise ValueError(f"missing required {stage} keys: {missing}")
    if forbidden:
        raise ValueError(f"forbidden {stage} keys: {forbidden}")
    if unknown:
        raise ValueError(f"unknown top-level config keys: {unknown}")
    if schema.requires_one_of and not (schema.requires_one_of & config.keys()):
        raise ValueError(f"{stage} requires one of {sorted(schema.requires_one_of)}")

    _validate_common_sections(config)

    model = config["model"]
    if not isinstance(model, Mapping) or not model.get("name"):
        raise ValueError("model.name is required")
    if model.get("name") == "pointcloud_ffm":
        backbone = model.get("backbone", "gl_rbf_enh")
        if backbone not in {"gl_rbf_enh", "fno"}:
            raise ValueError("new PointCloudFFM supports only gl_rbf_enh or fno")
        if backbone == "gl_rbf_enh" and model.get("gather_mode", "topk_rbf") != "topk_rbf":
            raise ValueError("new GL_rbf_ENH supports only gather_mode=topk_rbf")
    if stage == "base_training" and model.get("name") == "pinn":
        raise ValueError(
            "pinn is available only through direct_physics with a case PhysicsProvider"
        )
    if (
        model.get("name") == "latent_fm"
        and int(model.get("stage", 1)) == 2
        and not model.get("stage1_checkpoint")
    ):
        raise ValueError("latent_fm stage 2 requires model.stage1_checkpoint")

    observations = config["observations"]
    if observations.get("requires_validated_grid_mapping") and not config["dataset"].get(
        "grid_mapping_verified", False
    ):
        raise ValueError("structured sensor protocol requires dataset.grid_mapping_verified=true")
    for key in ("spatial_downsample_ratio", "temporal_downsample_ratio"):
        if key in observations and int(observations[key]) < 1:
            raise ValueError(f"observations.{key} must be a positive integer")
    if stage == "base_training":
        _validate_base_training(config)
    elif stage == "post_training":
        _validate_post_training(config)
    elif stage == "direct_physics":
        _validate_direct_physics(config)
