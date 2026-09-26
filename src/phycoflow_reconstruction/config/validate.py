"""Validate stage separation plus high-value model and observation invariants."""

from __future__ import annotations

import math
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
    "checkpointing",
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
    "benchmark_telemetry",
    "trainable",
    "posttrain_fidelity",
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


def _coherence_component_enabled(
    family_name: str, component_name: str, settings: Mapping[str, Any]
) -> bool:
    """Resolve component activation without implicitly enabling unstable terms."""
    default_enabled = not (family_name == "cross_spectrum" and component_name == "self_spectrum")
    return bool(settings.get("enabled", default_enabled))


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
            "normalization_stats_path",
            "time_stride",
            "benchmark_eligible",
            "allow_nonbenchmark",
            "coordinate_dim",
            "grid_shape",
            "grid_mapping_verified",
            "coordinate_reorder",
            "include_temporal_derivative",
            "augmentation",
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
    if "normalization_stats_path" in dataset:
        if not str(dataset["normalization_stats_path"]).strip():
            raise ValueError("dataset.normalization_stats_path must be non-empty")
        if dataset.get("normalization") not in {"mean_std", "robust_99"}:
            raise ValueError(
                "dataset.normalization_stats_path requires normalization=mean_std or robust_99"
            )
    names = dataset.get("field_names")
    units = dataset.get("field_units")
    if names is not None and (not names or len(set(names)) != len(names)):
        raise ValueError("dataset.field_names must be non-empty and unique")
    if names is not None and units is not None and len(names) != len(units):
        raise ValueError("dataset.field_units must align with field_names")
    if "normalization_stats_path" in dataset and not names:
        raise ValueError("dataset.field_names is required with normalization_stats_path")

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
        "structured_block_mean",
        "uniform_spacetime_stride",
    }:
        raise ValueError("observations.protocol is invalid")
    if dataset.get("augmentation"):
        augmentation = dataset["augmentation"]
        _reject_unknown(augmentation, {"kind", "vector_fields"}, "dataset.augmentation")
        if (
            augmentation.get("kind") != "periodic_translate_rot90"
            or observations.get("protocol") != "structured_block_mean"
        ):
            raise ValueError(
                "periodic_translate_rot90 currently requires structured_block_mean observations"
            )
        if dataset.get("reconstruction_unit", "snapshot") != "snapshot" or config.get("physics"):
            raise ValueError("augmentation requires snapshot data without paired physics context")
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
    runtime_keys = {
        "seed",
        "device",
        "deterministic",
        "num_workers",
        "progress",
        "plot_every_steps",
        "data_strategy",
        "vram_dataset_threshold_gb",
    }
    _reject_unknown(runtime, runtime_keys, "runtime")
    if int(runtime.get("num_workers", 0)) < 0:
        raise ValueError("runtime.num_workers must be non-negative")
    if "progress" in runtime and not isinstance(runtime["progress"], bool):
        raise TypeError("runtime.progress must be boolean")
    if int(runtime.get("plot_every_steps", 10)) < 1:
        raise ValueError("runtime.plot_every_steps must be a positive epoch interval")
    if runtime.get("data_strategy", "auto") not in {"auto", "vram", "async_cpu"}:
        raise ValueError("runtime.data_strategy must be auto, vram, or async_cpu")
    if float(runtime.get("vram_dataset_threshold_gb", 20.0)) <= 0:
        raise ValueError("runtime.vram_dataset_threshold_gb must be positive")
    output = _require_mapping(config, "output")
    _reject_unknown(output, {"experiment_name"}, "output")
    if not output.get("experiment_name"):
        raise ValueError("output.experiment_name is required")
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, Mapping):
        raise TypeError("evaluation must be a mapping")
    _reject_unknown(
        evaluation,
        {
            "split",
            "max_samples",
            "query_points",
            "generation_steps",
            "seed",
            "preview",
            "sample_selection",
            "native_topology",
        },
        "evaluation",
    )
    if config.get("evaluation", {}).get("sample_selection", "first") not in {"first", "uniform"}:
        raise ValueError("evaluation.sample_selection must be first or uniform")
    for key in ("max_samples", "query_points", "generation_steps"):
        if key in evaluation and evaluation[key] is not None and int(evaluation[key]) < 1:
            raise ValueError(f"evaluation.{key} must be positive")
    preview = evaluation.get("preview", {})
    if not isinstance(preview, Mapping):
        raise TypeError("evaluation.preview must be a mapping")
    _reject_unknown(
        preview,
        {
            "enabled",
            "every_epochs",
            "loss_every_epochs",
            "reconstruct_every_epochs",
            "split",
            "sample_index",
            "query_points",
            "generation_steps",
            "seed",
            "keep_history",
        },
        "evaluation.preview",
    )
    for key in ("enabled", "keep_history"):
        if key in preview and not isinstance(preview[key], bool):
            raise TypeError(f"evaluation.preview.{key} must be boolean")
    for key, default in (
        ("every_epochs", 10),
        ("loss_every_epochs", 10),
        ("reconstruct_every_epochs", 500),
    ):
        if int(preview.get(key, default)) < 1:
            raise ValueError(f"evaluation.preview.{key} must be positive")
    if {"loss_every_epochs", "reconstruct_every_epochs"} & set(preview) and not bool(
        preview.get("enabled", True)
    ):
        raise ValueError("modern evaluation.preview validation monitoring must be enabled")
    if int(preview.get("sample_index", 0)) < 0:
        raise ValueError("evaluation.preview.sample_index must be non-negative")
    for key in ("query_points", "generation_steps"):
        if preview.get(key) is not None and int(preview[key]) < 1:
            raise ValueError(f"evaluation.preview.{key} must be positive when provided")
    if preview.get("split", "validation") not in {"train", "validation", "test"}:
        raise ValueError("evaluation.preview.split is invalid")

    checkpointing = config.get("checkpointing", {})
    if not isinstance(checkpointing, Mapping):
        raise TypeError("checkpointing must be a mapping")
    _reject_unknown(
        checkpointing,
        {
            "enabled",
            "every_epochs",
            "every_steps",
            "epochs",
            "save_epoch_one",
            "selection_metric",
            "validation_every_epochs",
        },
        "checkpointing",
    )
    for key in ("enabled", "save_epoch_one"):
        if key in checkpointing and not isinstance(checkpointing[key], bool):
            raise TypeError(f"checkpointing.{key} must be boolean")
    if int(checkpointing.get("every_epochs", 10)) < 1:
        raise ValueError("checkpointing.every_epochs must be positive")
    if "every_steps" in checkpointing:
        interval = checkpointing["every_steps"]
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError("checkpointing.every_steps must be a positive integer")
    if "validation_every_epochs" in checkpointing:
        interval = checkpointing["validation_every_epochs"]
        if isinstance(interval, bool) or int(interval) != interval or int(interval) < 1:
            raise ValueError("checkpointing.validation_every_epochs must be a positive integer")
        if checkpointing.get("selection_metric") != "topology_with_fidelity":
            raise ValueError("validation_every_epochs currently requires topology_with_fidelity")
    if "epochs" in checkpointing:
        epochs = checkpointing["epochs"]
        if (
            not isinstance(epochs, (list, tuple))
            or not epochs
            or any(int(epoch) < 1 for epoch in epochs)
        ):
            raise ValueError("checkpointing.epochs must be a non-empty list of positive epochs")
        if len({int(epoch) for epoch in epochs}) != len(epochs):
            raise ValueError("checkpointing.epochs must not contain duplicates")
    if checkpointing.get("selection_metric", "native_validation_loss") not in {
        "native_validation_loss",
        "reconstruction_mse",
        "topology_with_fidelity",
    }:
        raise ValueError(
            "checkpointing.selection_metric must be native_validation_loss or reconstruction_mse"
        )
    if checkpointing.get("selection_metric") == "reconstruction_mse" and not bool(
        preview.get("enabled", True)
    ):
        raise ValueError("reconstruction_mse checkpoint selection requires preview.enabled=true")

    telemetry = config.get("benchmark_telemetry", {})
    if not isinstance(telemetry, Mapping):
        raise TypeError("benchmark_telemetry must be a mapping")
    _reject_unknown(telemetry, {"enabled", "sample_steps"}, "benchmark_telemetry")
    if "enabled" in telemetry and not isinstance(telemetry["enabled"], bool):
        raise TypeError("benchmark_telemetry.enabled must be boolean")
    if int(telemetry.get("sample_steps", 0)) < 0:
        raise ValueError("benchmark_telemetry.sample_steps must be non-negative")


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
    backward_loss_scale = float(settings.get("backward_loss_scale", 1.0))
    if not math.isfinite(backward_loss_scale) or not 0 < backward_loss_scale <= 1:
        raise ValueError("optimization.backward_loss_scale must be finite and in (0, 1]")
    if "adaptive_backward_scaling" in settings and not isinstance(
        settings["adaptive_backward_scaling"], bool
    ):
        raise TypeError("optimization.adaptive_backward_scaling must be boolean")


def _validate_base_training(config: Mapping[str, Any]) -> None:
    optimization = _require_mapping(config, "optimization")
    _reject_unknown(
        optimization,
        {
            "epochs",
            "batch_size",
            "lr",
            "weight_decay",
            "grad_clip",
            "backward_loss_scale",
            "adaptive_backward_scaling",
        },
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
            config["runtime"],
            {
                "seed",
                "device",
                "deterministic",
                "num_workers",
                "progress",
                "plot_every_steps",
                "data_strategy",
                "vram_dataset_threshold_gb",
            },
            "runtime",
        )
        _reject_unknown(config["output"], {"experiment_name"}, "output")
        return
    required_phase5 = {"objectives", "rollout", "observation_consistency", "trainable"}
    missing_phase5 = sorted(required_phase5 - config.keys())
    if missing_phase5:
        raise ValueError(f"data-driven post_training is missing keys: {missing_phase5}")

    objectives = config["objectives"]
    _reject_unknown(
        objectives, {"data_retention", "coherence", "endpoint", "source_anchor"}, "objectives"
    )
    for name in ("endpoint", "source_anchor"):
        if name not in objectives:
            continue
        term = objectives[name]
        _reject_unknown(term, {"enabled", "weight", "variance_floor"}, f"objectives.{name}")
        if not math.isfinite(float(term.get("weight", 0))) or float(term.get("weight", 0)) <= 0:
            raise ValueError(f"objectives.{name}.weight must be finite and positive")
        if (
            not math.isfinite(float(term.get("variance_floor", 1e-8)))
            or float(term.get("variance_floor", 1e-8)) <= 0
        ):
            raise ValueError(f"objectives.{name}.variance_floor must be finite and positive")
        if term.get("enabled", False) and not objectives.get("data_retention", {}).get(
            "enabled", True
        ):
            raise ValueError("endpoint/source retention requires data_retention")
    for name in ("data_retention", "coherence"):
        settings = objectives.get(name)
        if not isinstance(settings, Mapping):
            raise TypeError(f"objectives.{name} must be a mapping")
        _reject_unknown(settings, {"enabled", "weight"}, f"objectives.{name}")
        weight = float(settings.get("weight", 0.0))
        if weight < 0 or (bool(settings.get("enabled", True)) and weight <= 0):
            raise ValueError(f"enabled objectives.{name}.weight must be positive")
    if not any(
        bool(objectives[name].get("enabled", True))
        and float(objectives[name].get("weight", 0.0)) > 0
        for name in ("data_retention", "coherence")
    ):
        raise ValueError("post_training must enable at least one positive-weight objective")

    coherence = config["coherence"]
    _reject_unknown(
        coherence,
        {"schedule", "compute_budget", "family_balance", "families"},
        "coherence",
    )
    family_balance = coherence.get("family_balance", {})
    if not isinstance(family_balance, Mapping):
        raise TypeError("coherence.family_balance must be a mapping")
    _reject_unknown(
        family_balance,
        {
            "mode",
            "calibration_batches",
            "reference",
            "epsilon",
            "scale_min",
            "scale_max",
            "max_batch_ratio",
            "gradient_diagnostics_every_epochs",
            "seed",
        },
        "coherence.family_balance",
    )
    if family_balance.get("mode", "none") not in {"none", "initial_grad_norm"}:
        raise ValueError("coherence.family_balance.mode is invalid")
    if family_balance.get("mode", "none") != "none" and any(
        objectives.get(name, {}).get("enabled", False) for name in ("endpoint", "source_anchor")
    ):
        raise ValueError(
            "endpoint/source retention requires family_balance.mode=none; native-only calibration is not the retained objective"
        )
    if int(family_balance.get("calibration_batches", 2)) < 1:
        raise ValueError("coherence.family_balance.calibration_batches must be positive")
    if family_balance.get("reference", "median") != "median":
        raise ValueError("coherence.family_balance.reference currently supports only median")
    epsilon = float(family_balance.get("epsilon", 1.0e-12))
    scale_min = float(family_balance.get("scale_min", 1.0e-2))
    scale_max = float(family_balance.get("scale_max", 1.0e2))
    if epsilon <= 0 or scale_min <= 0 or scale_max < scale_min:
        raise ValueError("coherence.family_balance calibration bounds are invalid")
    if float(family_balance.get("max_batch_ratio", 1.0e2)) < 1:
        raise ValueError("coherence.family_balance.max_batch_ratio must be >=1")
    if int(family_balance.get("gradient_diagnostics_every_epochs", 0)) < 0:
        raise ValueError("gradient diagnostics interval must be non-negative")
    schedule = coherence.get("schedule", {})
    _reject_unknown(
        schedule,
        {"start_epoch", "every_n_steps", "weight_warmup_epochs", "interval_rescale"},
        "coherence.schedule",
    )
    if int(schedule.get("start_epoch", 1)) < 1 or int(schedule.get("every_n_steps", 1)) < 1:
        raise ValueError("coherence schedule start_epoch/every_n_steps must be positive")
    compute = coherence.get("compute_budget", {})
    _reject_unknown(
        compute,
        {"batch_size", "point_count", "query_policy", "query_seed"},
        "coherence.compute_budget",
    )
    if int(compute.get("batch_size", 1)) < 1 or int(compute.get("point_count", 2)) < 2:
        raise ValueError("coherence compute budget requires batch_size>=1 and point_count>=2")
    query_policy = compute.get("query_policy", "random_per_sample")
    if query_policy not in {"random_per_sample", "fixed_shared"}:
        raise ValueError("coherence.compute_budget.query_policy is invalid")

    families = coherence.get("families", {})
    supported_families = {"global_distribution", "cross_spectrum", "topology"}
    if not isinstance(families, Mapping) or not families:
        raise ValueError("post-training coherence must configure at least one family")
    unknown_families = sorted(set(families) - supported_families)
    if unknown_families:
        raise ValueError(f"unsupported coherence families: {unknown_families}")
    enabled_families = [
        name for name, settings in families.items() if bool(settings.get("enabled", True))
    ]
    if not enabled_families:
        raise ValueError("post-training coherence must enable at least one family")
    if any(name in {"cross_spectrum", "topology"} for name in enabled_families) and (
        query_policy != "fixed_shared"
    ):
        raise ValueError("cross_spectrum/topology coherence requires query_policy=fixed_shared")

    common_family_keys = {
        "enabled",
        "weight",
        "target_use",
        "units",
        "fields",
        "reference_bank",
        "components",
    }
    reference_keys = {"enabled", "path", "max_samples", "points_per_sample", "seed"}
    for family_name, family in families.items():
        if not isinstance(family, Mapping):
            raise TypeError(f"coherence.families.{family_name} must be a mapping")
        extra_keys = {
            "global_distribution": set(),
            "cross_spectrum": {"pairs", "graph", "eps"},
            "topology": {
                "geometry",
                "filtration",
                "strategy",
                "anchor",
                "evaluation",
                "persistence",
            },
        }[family_name]
        _reject_unknown(
            family,
            common_family_keys | extra_keys,
            f"coherence.families.{family_name}",
        )
        family_weight = float(family.get("weight", 1.0))
        family_enabled = bool(family.get("enabled", True))
        if family_weight < 0 or (family_enabled and family_weight <= 0):
            raise ValueError(f"enabled coherence family {family_name} weight must be positive")
        target_use = family.get("target_use", "training_reference")
        if target_use not in {"training_reference", "paired_supervised"}:
            raise ValueError(f"{family_name}.target_use is invalid")
        if family.get("units", "model_units") not in {"model_units", "physical_units"}:
            raise ValueError(f"{family_name}.units is invalid")
        reference = family.get("reference_bank", {})
        _reject_unknown(reference, reference_keys, f"{family_name}.reference_bank")
        if target_use == "training_reference" and bool(family.get("enabled", True)):
            if not bool(reference.get("enabled", True)):
                raise ValueError("training_reference coherence requires an enabled reference bank")
            if (
                int(reference.get("max_samples", 0)) < 1
                or int(reference.get("points_per_sample", 0)) < 2
            ):
                raise ValueError("reference bank requires max_samples>=1 and points_per_sample>=2")
            if int(reference["points_per_sample"]) != int(compute["point_count"]):
                raise ValueError("reference-bank and coherence compute point counts must match")

        components = family.get("components", {})
        topology_strategy = family.get("strategy", "betti_curves")
        if family_name == "topology":
            if topology_strategy not in {
                "betti_curves",
                "spatial_self_mutual",
                "cubical_persistence",
            }:
                raise ValueError("topology.strategy is invalid")
            if topology_strategy == "betti_curves" and (
                "anchor" in family or "evaluation" in family
            ):
                raise ValueError("topology anchor/evaluation requires strategy=spatial_self_mutual")
        if not isinstance(components, Mapping) or not components:
            raise ValueError(f"{family_name}.components cannot be empty")
        component_keys = {
            "global_distribution": {
                "self": {"enabled", "weight", "channel_weights"},
                "mutual": {"enabled", "weight", "pairs", "directions", "seed"},
                "cross": {
                    "enabled",
                    "weight",
                    "directions",
                    "top_fraction",
                    "seed",
                    "include_axes",
                    "qmc",
                },
            },
            "cross_spectrum": {
                "self_spectrum": {"enabled", "weight"},
                "same_frequency": {"enabled", "weight"},
                "cross_frequency": {"enabled", "weight"},
                "band_energy": {"enabled", "weight"},
            },
            "topology": {
                "self": {"enabled", "weight"},
                "mutual": {
                    "enabled",
                    "weight",
                    "pairs",
                    "lines",
                    "theta_min_degrees",
                    "axis_tolerance",
                },
            },
        }[family_name]
        if family_name == "topology" and topology_strategy == "spatial_self_mutual":
            from ..coherence.families.topology.spatial_objective import (
                SPATIAL_COMPONENT_KEYS,
                validate_spatial_config,
            )

            component_keys = SPATIAL_COMPONENT_KEYS
            validate_spatial_config(family, tuple(config["dataset"]["field_names"]))
        if family_name == "topology" and topology_strategy == "cubical_persistence":
            from ..coherence.families.topology.persistence_objective import (
                PERSISTENCE_COMPONENT_KEYS,
                validate_persistence_config,
            )

            component_keys = PERSISTENCE_COMPONENT_KEYS
            validate_persistence_config(family, tuple(config["dataset"]["field_names"]))
        _reject_unknown(components, set(component_keys), f"{family_name}.components")
        for component_name, settings in components.items():
            _reject_unknown(
                settings,
                component_keys[component_name],
                f"{family_name}.components.{component_name}",
            )
            component_weight = float(settings.get("weight", 1.0))
            component_enabled = _coherence_component_enabled(family_name, component_name, settings)
            if component_weight < 0 or (component_enabled and component_weight <= 0):
                raise ValueError("enabled coherence component weights must be positive")
        if family_enabled and not any(
            _coherence_component_enabled(family_name, component_name, settings)
            and float(settings.get("weight", 1.0)) > 0
            for component_name, settings in components.items()
        ):
            raise ValueError(
                f"enabled coherence family {family_name} requires a positive enabled component"
            )

        if family_name == "cross_spectrum":
            graph = family.get("graph", {})
            _reject_unknown(
                graph,
                {"k_neighbors", "sigma", "num_modes", "exclude_zero", "bands"},
                "cross_spectrum.graph",
            )
            if int(graph.get("k_neighbors", 16)) < 1 or int(graph.get("num_modes", 64)) < 1:
                raise ValueError("cross_spectrum graph sizes must be positive")
            cross_frequency = components.get("cross_frequency", {})
            cross_active = (
                family_enabled
                and family_weight > 0
                and bool(cross_frequency.get("enabled", True))
                and float(cross_frequency.get("weight", 1.0)) > 0
            )
            if cross_active and int(compute["batch_size"]) < 3:
                raise ValueError("cross-frequency coherence requires compute batch_size>=3")
            same_frequency = components.get("same_frequency", {})
            same_active = (
                family_enabled
                and family_weight > 0
                and bool(same_frequency.get("enabled", True))
                and float(same_frequency.get("weight", 1.0)) > 0
            )
            if same_active and int(compute["batch_size"]) < 2:
                raise ValueError("same-frequency coherence requires compute batch_size>=2")
            evaluation_minimum = 3 if cross_active else 2 if same_active else 1
            if int(config.get("evaluation", {}).get("max_samples", 1)) < evaluation_minimum:
                raise ValueError(
                    f"evaluation.max_samples must be >= {evaluation_minimum} for active spectral components"
                )
        elif family_name == "topology":
            geometry = family.get("geometry", {})
            _reject_unknown(
                geometry,
                {
                    "grid_shape",
                    "axes",
                    "neighbors",
                    "power",
                    "periodic",
                    "periods",
                    "allow_projected_collisions",
                    "antialias_downsample",
                },
                "topology.geometry",
            )
            if not isinstance(geometry.get("antialias_downsample", False), bool):
                raise ValueError("topology geometry.antialias_downsample must be boolean")
            if geometry.get("antialias_downsample", False) and topology_strategy == "betti_curves":
                raise ValueError("topology antialias_downsample requires v2 or v3")
            filtration = family.get("filtration", {})
            filtration_keys = {
                "quantiles",
                "dimensions",
                "directions",
                "sharpness",
                "smoothing_sigma",
            }
            if topology_strategy == "spatial_self_mutual":
                filtration_keys |= {"level_mode", "physical_levels"}
            _reject_unknown(
                filtration,
                filtration_keys,
                "topology.filtration",
            )

    if int(config["optimization"].get("batch_size", 1)) < int(compute["batch_size"]):
        raise ValueError("optimization.batch_size must be >= coherence.compute_budget.batch_size")

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
    if config["observations"].get("protocol") == "structured_block_mean" and (
        observation.get("mode") != "none" or observation.get("final_clamp", False)
    ):
        raise ValueError("block means cannot be used as pointwise observation clamps")
    if config["model"].get("physical_field_transform") and rollout["solver"] != config["model"].get(
        "ode_solver", "euler"
    ):
        raise ValueError("physical CQ rollout.solver must match model.ode_solver")
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
            "model_mode",
            "rollout_execution",
            "component_constraints",
            "training_subset",
            "sampling",
            "steps_per_epoch",
        },
        "optimization",
    )
    _validate_optimization_values(optimization)
    if "steps_per_epoch" in optimization:
        from ..training.update_budget import post_training_steps_per_epoch

        post_training_steps_per_epoch(config, 1)
    if optimization.get("sampling", "independent_batches") not in {
        "independent_batches",
        "full_pass",
    }:
        raise ValueError("optimization.sampling must be independent_batches or full_pass")
    if (
        optimization.get("sampling") == "full_pass"
        and float(optimization.get("train_fraction", 1.0)) != 1.0
    ):
        raise ValueError(
            "full_pass requires train_fraction=1; training_subset controls the dataset size"
        )
    if "training_subset" in optimization:
        subset = optimization["training_subset"]
        if not isinstance(subset, Mapping):
            raise TypeError("optimization.training_subset must be a mapping")
        _reject_unknown(
            subset,
            {
                "fraction",
                "seed",
                "min_frames_per_trajectory",
                "strata_keys",
                "trajectory_key",
                "time_key",
                "frame_key",
                "split_key",
            },
            "optimization.training_subset",
        )
        strata_keys = subset.get("strata_keys", [])
        if not isinstance(strata_keys, list) or any(
            not isinstance(key, str) or not key for key in strata_keys
        ):
            raise ValueError("training_subset.strata_keys must be a list of metadata column names")
        if len(strata_keys) != len(set(strata_keys)):
            raise ValueError("training_subset.strata_keys must be unique")
        for key in ("trajectory_key", "time_key", "frame_key", "split_key"):
            if key in subset and (not isinstance(subset[key], str) or not subset[key]):
                raise ValueError(f"training_subset.{key} must be a metadata column name")
        if not 0 < float(subset.get("fraction", 0.3)) <= 1:
            raise ValueError("training_subset.fraction must lie in (0,1]")
        minimum = subset.get("min_frames_per_trajectory", 4)
        if isinstance(minimum, bool) or int(minimum) != minimum or minimum < 1:
            raise ValueError("training_subset.min_frames_per_trajectory must be a positive integer")
        seed = subset.get("seed", 42)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("training_subset.seed must be a nonnegative integer")
        if optimization.get("sampling") != "full_pass":
            raise ValueError("training_subset requires full_pass sampling")
    if optimization.get("model_mode", "eval") not in {"eval", "train"}:
        raise ValueError("optimization.model_mode must be eval or train")
    execution = optimization.get("rollout_execution", {})
    if not isinstance(execution, Mapping):
        raise TypeError("optimization.rollout_execution must be a mapping")
    _reject_unknown(execution, {"context_cache", "checkpointing"}, "optimization.rollout_execution")
    if execution:
        model = config["model"]
        # The CLI resolves name-only source templates before validating again.
        inherited_model_pending = config.get("inherit_base_config", True) and model == {
            "name": "gl_rbf_cq"
        }
        if model.get("name") != "gl_rbf_cq" or (
            not model.get("physical_field_transform") and not inherited_model_pending
        ):
            raise ValueError("rollout_execution requires a physical CQ model")
        if execution.get("context_cache", "none") not in {
            "none",
            "condition",
            "geometry",
            "static_features",
        }:
            raise ValueError("invalid rollout_execution.context_cache")
        if (
            execution.get("context_cache", "none") != "none"
            and optimization.get("model_mode", "eval") != "eval"
        ):
            raise ValueError("shared rollout context requires model_mode=eval")
        if "checkpointing" in execution and not isinstance(execution["checkpointing"], bool):
            raise ValueError("rollout_execution.checkpointing must be boolean")
    if any(family.get("strategy") == "cubical_persistence" for family in families.values()):
        if optimization.get("model_mode", "eval") != "eval":
            raise ValueError(
                "cubical_persistence requires inference-mode gradients (model_mode=eval)"
            )
        if int(config.get("evaluation", {}).get("generation_steps", 0)) != int(rollout["steps"]):
            raise ValueError("cubical_persistence requires matching training/evaluation steps")
        if int(config.get("runtime", {}).get("num_workers", 0)) != 0 and config["dataset"].get(
            "augmentation"
        ):
            raise ValueError("resumable persistence augmentation requires num_workers=0")
    gradient_mode = optimization.get("gradient_balance", "weighted_sum")
    if gradient_mode not in {
        "weighted_sum",
        "config",
        "component_constrained",
        "topology_regularized",
    }:
        raise ValueError(
            "optimization.gradient_balance must be weighted_sum, config, "
            "component_constrained, or topology_regularized"
        )
    if "component_constraints" in optimization and gradient_mode not in {
        "component_constrained",
        "topology_regularized",
    }:
        raise ValueError(
            "component_constraints requires component_constrained or topology_regularized mode"
        )
    if gradient_mode in {"component_constrained", "topology_regularized"}:
        from ..training.topology_constraints import CONSTRAINT_DEFAULTS

        settings = optimization.get("component_constraints", {})
        _reject_unknown(settings, set(CONSTRAINT_DEFAULTS), "component_constraints")
        if settings.get("proposal_objective", "topology") not in {"topology", "endpoint"}:
            raise ValueError(
                "component_constraints.proposal_objective must be topology or endpoint"
            )
        if settings.get("correction_selection", "all_violations") not in {
            "all_violations",
            "largest_violation",
        }:
            raise ValueError(
                "component_constraints.correction_selection must be all_violations or largest_violation"
            )
        if settings.get("topology_reduction", "per_sample") not in {"per_sample", "batch_mean"}:
            raise ValueError(
                "component_constraints.topology_reduction must be per_sample or batch_mean"
            )
        if gradient_mode == "topology_regularized":
            if settings.get("topology_reduction") != "batch_mean":
                raise ValueError("topology_regularized requires batch_mean")
            for name in ["anchor_soft_fraction", "endpoint_soft_fraction"]:
                if not 0 < float(settings.get(name, CONSTRAINT_DEFAULTS[name])) < 1:
                    raise ValueError(f"{name} must lie in (0,1)")
            if float(settings.get("fidelity_penalty_weight", 1.0)) <= 0:
                raise ValueError("topology_regularized requires positive fidelity_penalty_weight")
        for key, default in CONSTRAINT_DEFAULTS.items():
            if key in {"proposal_objective", "correction_selection", "topology_reduction"}:
                continue
            value = settings.get(key, default)
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"invalid component_constraints.{key}")
            if (
                key in {"calibration_batches", "max_corrections", "max_active_gradients"}
                and int(value) != value
            ):
                raise ValueError(f"component_constraints.{key} must be integer")
            if (
                key
                in {
                    "calibration_batches",
                    "scale_floor",
                    "max_source_anchor_nmse",
                    "max_active_gradients",
                }
                and value <= 0
            ):
                raise ValueError(f"component_constraints.{key} must be positive")
        if int(settings.get("max_active_gradients", 12)) > 32:
            raise ValueError("component_constraints permits at most 32 correction gradients")
        if int(settings.get("max_corrections", 2)) > 4:
            raise ValueError("component_constraints permits at most four correction rounds")
        if (
            set(families) != {"topology"}
            or families["topology"].get("strategy") != "cubical_persistence"
            or families["topology"].get("target_use") != "paired_supervised"
        ):
            raise ValueError("component constraints require only paired cubical_persistence")
        if any(
            objectives.get(name, {}).get("enabled", name == "data_retention")
            for name in ("data_retention", "endpoint", "source_anchor")
        ):
            raise ValueError(
                "component constraints own fidelity terms; disable separate retention objectives"
            )
        if not math.isfinite(float(objectives["coherence"]["weight"])):
            raise ValueError("component constraints require finite objective weight")
        if not objectives["coherence"].get("enabled", True):
            raise ValueError("component constraints require the topology objective")
        if (
            int(schedule.get("every_n_steps", 1)) != 1
            or int(schedule.get("start_epoch", 1)) != 1
            or int(schedule.get("weight_warmup_epochs", 0)) != 0
        ):
            raise ValueError("component constraints must check every update from step one")
        topology = families["topology"]
        if (
            list(topology["geometry"]["grid_shape"])
            != list(config["dataset"].get("grid_shape", []))
            or not topology["geometry"].get("antialias_downsample", False)
            or float(topology["filtration"].get("smoothing_sigma", 0)) != 0
            or int(compute["point_count"]) != math.prod(topology["geometry"]["grid_shape"])
        ):
            raise ValueError(
                "component constraints require complete unsmoothed native grids per update"
            )
        if config.get("posttrain_fidelity", {}).get("max_relative_field_mse_increase") is None:
            raise ValueError("component constraints require per-field source fidelity bounds")
        if family_balance.get("mode", "none") != "none":
            raise ValueError("component constraints use their own fixed component calibration")
        if config.get("checkpointing", {}).get("selection_metric") != "topology_with_fidelity":
            raise ValueError("component constraints require topology_with_fidelity selection")
    if optimization.get("config_missing_behavior", "error") not in {"error", "weighted_sum"}:
        raise ValueError("optimization.config_missing_behavior must be error or weighted_sum")
    _reject_unknown(
        config["runtime"],
        {
            "seed",
            "device",
            "deterministic",
            "num_workers",
            "progress",
            "plot_every_steps",
            "data_strategy",
            "vram_dataset_threshold_gb",
        },
        "runtime",
    )
    _reject_unknown(
        config.get("evaluation", {}),
        {
            "split",
            "max_samples",
            "query_points",
            "generation_steps",
            "seed",
            "preview",
            "sample_selection",
            "native_topology",
        },
        "evaluation",
    )
    _reject_unknown(config["output"], {"experiment_name"}, "output")
    fidelity = config.get("posttrain_fidelity", {})
    if not isinstance(fidelity, Mapping):
        raise TypeError("posttrain_fidelity must be a mapping")
    _reject_unknown(
        fidelity,
        {
            "max_relative_mse_increase",
            "max_relative_field_mse_increase",
            "max_relative_native_topology_increase",
            "behavior",
        },
        "posttrain_fidelity",
    )
    if (
        fidelity.get("max_relative_mse_increase") is not None
        and float(fidelity["max_relative_mse_increase"]) < 0
    ):
        raise ValueError("posttrain_fidelity.max_relative_mse_increase must be non-negative")
    if fidelity.get("max_relative_mse_increase") is not None and not math.isfinite(
        float(fidelity["max_relative_mse_increase"])
    ):
        raise ValueError("posttrain_fidelity.max_relative_mse_increase must be finite")
    if fidelity.get("behavior", "report") not in {"report", "warn", "error"}:
        raise ValueError("posttrain_fidelity.behavior must be report, warn, or error")
    field_budget = fidelity.get("max_relative_field_mse_increase")
    if field_budget is not None and (
        not math.isfinite(float(field_budget)) or float(field_budget) < 0
    ):
        raise ValueError("per-field fidelity budget must be finite and nonnegative")
    if config.get("checkpointing", {}).get("selection_metric") == "topology_with_fidelity":
        if (
            config.get("evaluation", {}).get("split", "validation") != "validation"
            or int(config.get("evaluation", {}).get("max_samples", 1)) < 2
        ):
            raise ValueError("topology selection requires a multi-sample validation panel")
        if fidelity.get("max_relative_mse_increase") is None or field_budget is None:
            raise ValueError("topology selection requires total and per-field fidelity budgets")
        if "topology" not in families or not families["topology"].get("enabled", True):
            raise ValueError("topology selection requires an enabled topology family")
    if config.get("evaluation", {}).get("sample_selection", "first") not in {"first", "uniform"}:
        raise ValueError("evaluation.sample_selection must be first or uniform")
    if config.get("evaluation", {}).get("native_topology", False):
        if (
            "topology" not in families
            or families["topology"].get("target_use") != "paired_supervised"
            or families["topology"].get("strategy")
            not in {"spatial_self_mutual", "cubical_persistence"}
            or len(config["dataset"].get("grid_shape", [])) != 2
        ):
            raise ValueError("native_topology requires paired topology and a 2D dataset grid")
        if int(config.get("evaluation", {}).get("query_points", 0)) != math.prod(
            config["dataset"]["grid_shape"]
        ):
            raise ValueError("native_topology requires all native grid query points")
    native_budget = fidelity.get("max_relative_native_topology_increase")
    if native_budget is not None:
        if not config.get("evaluation", {}).get("native_topology", False):
            raise ValueError("native topology guard requires evaluation.native_topology")
        if not math.isfinite(float(native_budget)) or float(native_budget) < 0:
            raise ValueError("native topology budget must be finite and nonnegative")


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
        {"epochs", "batch_size", "lr", "weight_decay", "grad_clip", "backward_loss_scale"},
        "optimization",
    )
    _validate_optimization_values(config["optimization"])
    _reject_unknown(
        config["runtime"],
        {
            "seed",
            "device",
            "deterministic",
            "num_workers",
            "progress",
            "plot_every_steps",
            "data_strategy",
            "vram_dataset_threshold_gb",
        },
        "runtime",
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
    model_name = str(model.get("name")).lower()
    if model_name == "mimonet":
        conditioned = model.get("conditioning_fields")
        capacities = model.get("sensor_capacities")
        if (
            not isinstance(conditioned, (list, tuple))
            or not isinstance(capacities, (list, tuple))
            or not conditioned
            or len(conditioned) != len(capacities)
            or len(set(conditioned)) != len(conditioned)
        ):
            raise ValueError("mimonet requires one capacity per unique conditioning field")
        observed = config["observations"]["fields"]
        if set(observed) != set(conditioned):
            raise ValueError("mimonet observations must match conditioning_fields exactly")
        if set(conditioned) - set(config["dataset"]["field_names"]):
            raise ValueError("mimonet conditioning_fields must belong to the dataset")
        for name, capacity in zip(conditioned, capacities):
            maximum = observed[name].get("count", observed[name].get("count_max"))
            if int(capacity) < int(maximum):
                raise ValueError(f"mimonet sensor capacity for {name} is below observation count")
        for key in ("basis_dim", "branch_hidden_dim", "trunk_hidden_dim"):
            if key in model and int(model[key]) < 1:
                raise ValueError(f"mimonet {key} must be positive")
        if model.get("merge_type", "mul") not in {"mul", "sum"}:
            raise ValueError("mimonet merge_type must be mul or sum")
    elif model_name == "pointcloud_ffm":
        backbone = model.get("backbone", "gl_rbf_enh")
        if backbone not in {"gl_rbf_enh", "fno"}:
            raise ValueError("new PointCloudFFM supports only gl_rbf_enh or fno")
        if backbone == "gl_rbf_enh" and model.get("gather_mode", "topk_rbf") != "topk_rbf":
            raise ValueError("new GL_rbf_ENH supports only gather_mode=topk_rbf")
    elif model_name == "diffusion_pde":
        backbone = str(model.get("backbone", "plain_cnn")).lower()
        if backbone not in {"plain_cnn", "conditional_unet"}:
            raise ValueError("diffusion_pde backbone must be plain_cnn or conditional_unet")
        if int(model.get("training_timesteps", 1000)) < 2:
            raise ValueError("model.training_timesteps must be at least two")
        if backbone == "plain_cnn":
            hidden_channels = int(model.get("hidden_channels", 32))
            if hidden_channels < 4 or hidden_channels % 4:
                raise ValueError("model.hidden_channels must be a positive multiple of four")
        else:
            base_channels = int(model.get("base_channels", 64))
            multipliers = tuple(
                int(value) for value in model.get("channel_multipliers", (1, 2, 4, 8))
            )
            levels = tuple(int(value) for value in model.get("attention_levels", (2, 3)))
            heads = int(model.get("attention_heads", 4))
            if base_channels < 4:
                raise ValueError("model.base_channels must be at least four")
            if not multipliers or any(value < 1 for value in multipliers):
                raise ValueError("model.channel_multipliers must contain positive integers")
            if int(model.get("num_res_blocks", 2)) < 1:
                raise ValueError("model.num_res_blocks must be positive")
            if int(model.get("time_embed_dim", 256)) < 4:
                raise ValueError("model.time_embed_dim must be at least four")
            if heads < 1:
                raise ValueError("model.attention_heads must be positive")
            dropout = float(model.get("dropout", 0.0))
            if not 0.0 <= dropout < 1.0:
                raise ValueError("model.dropout must lie in [0, 1)")
            if any(level < 0 or level >= len(multipliers) for level in levels):
                raise ValueError("model.attention_levels contains an invalid U-Net level")
            if any(base_channels * multipliers[level] % heads for level in levels):
                raise ValueError(
                    "model attention level channels must be divisible by attention_heads"
                )
    elif model_name == "gl_rbf_cq":
        backbone = str(model.get("backbone", "GL_rbf_ENH_CQ")).lower()
        if backbone != "gl_rbf_enh_cq":
            raise ValueError("gl_rbf_cq requires backbone=GL_rbf_ENH_CQ")
        execution = str(model.get("condition_attention_execution", "cached_kv"))
        if execution not in {"legacy_mha", "cached_kv"}:
            raise ValueError(
                "gl_rbf_cq condition_attention_execution must be legacy_mha or cached_kv"
            )
        if model.get("sensor_attention_padding_mode", "full") != "full":
            raise ValueError("gl_rbf_cq requires sensor_attention_padding_mode=full")
        if str(model.get("gather_mode", "topk_rbf_glres")) != "topk_rbf_glres":
            raise ValueError("gl_rbf_cq requires gather_mode=topk_rbf_glres")
        for key in (
            "gather_topk",
            "gather_query_chunk_size",
            "train_query_microbatch_size",
            "reconstruction_query_chunk_size",
            "cq_query_dim",
            "cq_readout_rank",
            "cq_readout_heads",
        ):
            if key in model and int(model[key]) < 1:
                raise ValueError(f"model.{key} must be positive")
        query_dim = int(model.get("cq_query_dim", 128))
        query_heads = int(model.get("cq_readout_heads", 4))
        readout_rank = int(model.get("cq_readout_rank", 64))
        if query_dim % query_heads != 0:
            raise ValueError("model.cq_query_dim must be divisible by cq_readout_heads")
        if readout_rank % query_heads != 0:
            raise ValueError("model.cq_readout_rank must be divisible by cq_readout_heads")
        backend = str(model.get("neighbor_backend", "torch"))
        if backend not in {"torch", "keops"}:
            raise ValueError("model.neighbor_backend must be torch or keops")
        if "model_ema_decay" in model and not 0.0 <= float(model["model_ema_decay"]) < 1.0:
            raise ValueError("model.model_ema_decay must be in [0, 1)")
    if stage == "base_training" and model_name == "pinn":
        raise ValueError(
            "pinn is available only through direct_physics with a case PhysicsProvider"
        )
    if (
        model_name == "latent_fm"
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
