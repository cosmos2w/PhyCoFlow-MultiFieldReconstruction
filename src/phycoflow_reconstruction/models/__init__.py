"""Model registry and config-driven construction for all Phase-4 adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..contracts import DataSpec
from ..registry import MODEL_REGISTRY
from .deterministic import (
    CoordinateMLP,
    MLPRBFRegressor,
    PINNRegressor,
    SenseiverRegressor,
    SparseDeepONet,
)
from .flows import GLRbfCQ, PointCloudFFM
from .generative import DiffusionPDEModel, LatentFlowModel
from .operators import GeoFNORegressor


def _register_defaults() -> None:
    if MODEL_REGISTRY.names():
        return
    MODEL_REGISTRY.register(
        "coordinate_mlp",
        CoordinateMLP,
        metadata={
            "family": "deterministic_point",
            "license": "project-local",
            "stages": ("base_training",),
        },
    )
    MODEL_REGISTRY.register(
        "mlp_rbf",
        MLPRBFRegressor,
        metadata={
            "family": "deterministic_point",
            "license": "project-local",
            "stages": ("base_training",),
        },
    )
    MODEL_REGISTRY.register(
        "pinn",
        PINNRegressor,
        metadata={
            "family": "physics_informed",
            "requires_physics": True,
            "license": "project-local",
            "upstream_reference": "https://github.com/lululxvi/deepxde",
            "stages": ("direct_physics",),
        },
    )
    MODEL_REGISTRY.register(
        "deeponet",
        SparseDeepONet,
        metadata={
            "family": "deterministic_point",
            "license": "project-local",
            "upstream_reference": "https://github.com/lululxvi/deepxde",
            "stages": ("base_training",),
        },
    )
    MODEL_REGISTRY.register(
        "senseiver",
        SenseiverRegressor,
        metadata={
            "family": "deterministic_point",
            "license": "project-local clean-room",
            "upstream_reference": "https://github.com/OrchardLANL/Senseiver",
            "stages": ("base_training",),
        },
    )
    MODEL_REGISTRY.register(
        "geofno",
        GeoFNORegressor,
        metadata={
            "family": "deterministic_operator",
            "optional_dependency": "neuraloperator==2.0.0",
            "dependency_license": "MIT",
            "stages": ("base_training",),
        },
    )
    MODEL_REGISTRY.register(
        "diffusion_pde",
        DiffusionPDEModel,
        metadata={
            "family": "grid_generative",
            "license": "project-local clean-room",
            "upstream_reference": "https://github.com/jhhuangchloe/DiffusionPDE",
            "stages": ("base_training",),
        },
    )
    MODEL_REGISTRY.register(
        "latent_fm",
        LatentFlowModel,
        metadata={
            "family": "latent_generative",
            "license": "project-local",
            "stages": ("base_training",),
        },
    )
    MODEL_REGISTRY.register(
        "pointcloud_ffm",
        PointCloudFFM,
        metadata={
            "family": "point_rectified_flow",
            "license": "project-local",
            "optional_dependency": "neuraloperator==2.0.0 for backbone=fno",
            "stages": ("base_training",),
        },
    )
    MODEL_REGISTRY.register(
        "gl_rbf_cq",
        GLRbfCQ,
        metadata={
            "family": "point_rectified_flow",
            "license": "project-local portable-core adapter",
            "stages": ("base_training", "post_training"),
            "public_model_name": "GL_rbf_CQ",
            "historical_backbone_name": "GL_rbf_ENH_CQ",
        },
    )


def build_model(config: Mapping[str, Any], data_spec: DataSpec, physics_provider: Any = None):
    _register_defaults()
    name = str(config["name"]).lower()
    common = {"num_fields": data_spec.num_fields}
    if name in {
        "coordinate_mlp",
        "mlp_rbf",
        "deeponet",
        "senseiver",
        "pinn",
        "pointcloud_ffm",
        "gl_rbf_cq",
    }:
        common["coordinate_dim"] = data_spec.coordinate_dim
    if name in {
        "geofno",
        "diffusion_pde",
        "latent_fm",
        "pointcloud_ffm",
        "gl_rbf_cq",
    }:
        common["logical_shape"] = data_spec.logical_shape

    allowed_by_model = {
        "coordinate_mlp": {"hidden_dim", "fourier_bands"},
        "mlp_rbf": {"hidden_dim", "rbf_sigma", "fourier_bands"},
        "pinn": {"hidden_dim", "fourier_bands", "physics_weight"},
        "deeponet": {"width", "basis_dim"},
        "senseiver": {"width", "num_latents", "heads", "depth"},
        "geofno": {"hidden_channels", "modes", "layers"},
        "diffusion_pde": {"hidden_channels", "training_timesteps"},
        "latent_fm": {"latent_channels", "stage", "stage1_checkpoint"},
        "pointcloud_ffm": {
            "backbone",
            "prior",
            "hidden_dim",
            "latent_dim",
            "num_latents",
            "heads",
            "latent_blocks",
            "gather_topk",
            "rbf_sigma",
            "fno_hidden_channels",
            "query_chunk_size",
            "field_embedding_dim",
            "fourier_bands",
            "fourier_max_frequency",
            "rff_features",
            "rff_lengthscale",
        },
        "gl_rbf_cq": {
            "backbone",
            "prior",
            "sigma_min",
            "rff_features",
            "rff_lengthscale",
            "hidden_dim",
            "cond_dim",
            "field_embed_dim",
            "field_embedding_dim",
            "rbf_sigma",
            "USE_FOURIER_PE",
            "fourier_pe_num_bands",
            "fourier_pe_max_freq",
            "fourier_bands",
            "fourier_max_frequency",
            "latent_dim",
            "num_latents",
            "num_heads",
            "heads",
            "num_latent_blocks",
            "latent_blocks",
            "ff_mult",
            "attn_dropout",
            "mlp_dropout",
            "summary_type",
            "sensor_coord_encoding",
            "latent_sensor_reinject",
            "latent_reinject_every",
            "condition_attention_execution",
            "sensor_attention_padding_mode",
            "sensor_attention_buckets",
            "gather_mode",
            "gather_topk",
            "gather_topk_rbf_glres",
            "gather_query_chunk_size",
            "learnable_rbf_sigma",
            "neighbor_backend",
            "sensor_local_topk",
            "sensor_local_dropout",
            "query_latent_readout",
            "query_readout_type",
            "query_readout_scale_init",
            "enhanced_head_norm",
            "glres_scale_init",
            "cq_query_dim",
            "cq_readout_mode",
            "cq_readout_rank",
            "cq_readout_heads",
            "cq_fusion_mode",
            "cq_global_scale_init",
            "cq_local_scale_init",
            "cq_readout_scale_init",
            "cq_time_conditioning",
            "cq_time_embed_dim",
            "cq_time_max_period",
            "cq_time_film_zero_init",
            "cq_measurement_support_mode",
            "cq_measurement_support_normalize",
            "model_ema_enabled",
            "model_ema_decay",
            "model_ema_eval",
            "train_query_microbatch_size",
            "query_microbatch_size",
            "reuse_condition_context_across_query_microbatches",
            "reuse_condition_context",
            "reconstruction_execution_mode",
            "reconstruction_query_chunk_size",
            "reconstruction_cache_level",
            "ode_solver",
            "obs_consistency_mode",
            "obs_consistency_strength",
            "obs_consistency_sigma",
            "obs_consistency_schedule_power",
            "obs_consistency_final_clamp",
            "obs_consistency_chunk_size",
        },
    }
    if name not in allowed_by_model:
        MODEL_REGISTRY.get(name)  # raise the registry's actionable available-name error
    control_keys = {"name", "query_points", "gather_mode"}
    unknown = sorted(set(config) - allowed_by_model[name] - control_keys)
    if unknown:
        raise ValueError(f"unsupported {name} model keys: {unknown}")
    kwargs = {key: value for key, value in config.items() if key in allowed_by_model[name]}
    if "modes" in kwargs:
        kwargs["modes"] = tuple(int(v) for v in kwargs["modes"])
    if name == "pinn":
        kwargs["physics_provider"] = physics_provider
    return MODEL_REGISTRY.build(name, **common, **kwargs)


_register_defaults()

__all__ = ["GLRbfCQ", "build_model"]
