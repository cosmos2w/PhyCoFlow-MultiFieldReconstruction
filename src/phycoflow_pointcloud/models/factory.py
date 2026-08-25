"""Single read-only builder for public and historical GL-RBF configurations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .portable_core import (
    ConditionalPointHybridLocalGlobalRBF,
    ConditionalPointHybridLocalGlobalRBFCQ,
    PointCloudFFM,
)
from ..config import resolve_model_identity
from ..priors import IIDGaussianPrior, RFFGaussianPrior


def _get(config: Mapping[str, Any], key: str, default: Any) -> Any:
    value = config.get(key, default)
    return default if value is None else value


def build_pointcloud_model(
    model_name_or_config: str | Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
    *,
    n_fields: int = 5,
    device: torch.device | str = "cpu",
    prior_override: IIDGaussianPrior | RFFGaussianPrior | None = None,
) -> PointCloudFFM:
    """Build GL_rbf_CQ, GL_rbf_CQ-fast, GL_rbf_ENH, or historical aliases.

    Construction order and defaults intentionally match the RC1 fixed-manifest
    builder so seeded parameters, buffers, and checkpoint keys remain identical.
    """
    if isinstance(model_name_or_config, Mapping):
        if config is not None:
            raise TypeError(
                "Pass either a config mapping or (model_name, config), not both."
            )
        resolved = dict(model_name_or_config)
    else:
        resolved = dict(config or {})
        resolved["model_name"] = str(model_name_or_config)
    identity = resolve_model_identity(resolved)
    backbone_name = identity.internal_backbone
    coord_dim = int(_get(resolved, "coord_dim", 3))
    is_cq = backbone_name == "GL_rbf_ENH_CQ"
    enhanced = backbone_name in {"GL_rbf_ENH", "GL_rbf_ENH_CQ"}
    sensor_coord_encoding = _get(
        resolved, "sensor_coord_encoding", "fourier" if enhanced else "raw"
    )
    latent_sensor_reinject = bool(_get(resolved, "latent_sensor_reinject", enhanced))
    glres_scale_init = float(
        _get(resolved, "glres_scale_init", 1.0e-2 if enhanced else 0.0)
    )

    prior_name = str(_get(resolved, "prior", "rff"))
    if prior_name not in {"iid", "rff"}:
        raise ValueError(f"Unknown prior {prior_name!r}; expected 'iid' or 'rff'.")
    prior = prior_override
    if prior is None:
        prior = (
            IIDGaussianPrior()
            if prior_name == "iid"
            else RFFGaussianPrior(
                coord_dim=coord_dim,
                n_features=int(_get(resolved, "rff_features", 256)),
                lengthscale=float(_get(resolved, "rff_lengthscale", 0.15)),
            )
        )
    common = {
        "n_fields": n_fields,
        "coord_dim": coord_dim,
        "hidden_dim": int(_get(resolved, "hidden_dim", 256)),
        "cond_dim": int(_get(resolved, "cond_dim", 128)),
        "field_embed_dim": int(_get(resolved, "field_embed_dim", 64)),
        "latent_dim": int(_get(resolved, "latent_dim", 256)),
        "num_latents": int(_get(resolved, "num_latents", 128)),
        "num_heads": int(_get(resolved, "num_heads", 8)),
        "num_latent_blocks": int(_get(resolved, "num_latent_blocks", 4)),
        "ff_mult": int(_get(resolved, "ff_mult", 4)),
        "attn_dropout": float(_get(resolved, "attn_dropout", 0.0)),
        "mlp_dropout": float(_get(resolved, "mlp_dropout", 0.0)),
        "rbf_sigma": float(_get(resolved, "rbf_sigma", 0.05)),
        "summary_type": str(_get(resolved, "summary_type", "cls")),
        "gather_mode": str(_get(resolved, "gather_mode", "rbf")),
        "gather_topk": int(_get(resolved, "gather_topk", 32)),
        "gather_query_chunk_size": _get(resolved, "gather_query_chunk_size", None),
        "learnable_rbf_sigma": bool(_get(resolved, "learnable_rbf_sigma", False)),
        "neighbor_backend": str(_get(resolved, "neighbor_backend", "torch")),
        "sensor_local_topk": int(_get(resolved, "sensor_local_topk", 8)),
        "sensor_local_dropout": float(_get(resolved, "sensor_local_dropout", 0.0)),
        "use_fourier_pe": bool(_get(resolved, "USE_FOURIER_PE", False)),
        "fourier_pe_num_bands": int(_get(resolved, "fourier_pe_num_bands", 32)),
        "fourier_pe_max_freq": float(_get(resolved, "fourier_pe_max_freq", 64.0)),
        "sensor_coord_encoding": str(sensor_coord_encoding),
        "latent_sensor_reinject": latent_sensor_reinject,
        "latent_reinject_every": int(_get(resolved, "latent_reinject_every", 1)),
        "condition_attention_execution": str(
            _get(resolved, "condition_attention_execution", "legacy_mha")
        ),
        "sensor_attention_padding_mode": str(
            _get(resolved, "sensor_attention_padding_mode", "full")
        ),
        "sensor_attention_buckets": tuple(
            int(value)
            for value in _get(
                resolved, "sensor_attention_buckets", [256, 320, 384]
            )
        ),
        "glres_scale_init": glres_scale_init,
    }
    if is_cq:
        backbone = ConditionalPointHybridLocalGlobalRBFCQ(
            **common,
            cq_query_dim=int(_get(resolved, "cq_query_dim", 128)),
            cq_readout_mode=str(_get(resolved, "cq_readout_mode", "lowrank")),
            cq_fusion_mode=str(_get(resolved, "cq_fusion_mode", "additive")),
            cq_readout_rank=int(_get(resolved, "cq_readout_rank", 64)),
            cq_readout_heads=int(_get(resolved, "cq_readout_heads", 4)),
            cq_global_scale_init=float(_get(resolved, "cq_global_scale_init", 1.0)),
            cq_local_scale_init=float(_get(resolved, "cq_local_scale_init", 1.0)),
            cq_readout_scale_init=float(
                _get(resolved, "cq_readout_scale_init", 1.0e-2)
            ),
            cq_time_conditioning=str(
                _get(resolved, "cq_time_conditioning", "scalar_concat")
            ),
            cq_time_embed_dim=int(_get(resolved, "cq_time_embed_dim", 128)),
            cq_time_max_period=float(_get(resolved, "cq_time_max_period", 10000.0)),
            cq_time_film_zero_init=bool(_get(resolved, "cq_time_film_zero_init", True)),
            cq_measurement_support_mode=str(
                _get(resolved, "cq_measurement_support_mode", "none")
            ),
            cq_measurement_support_normalize=bool(
                _get(resolved, "cq_measurement_support_normalize", True)
            ),
        )
    else:
        query_latent_readout = bool(_get(resolved, "query_latent_readout", enhanced))
        backbone = ConditionalPointHybridLocalGlobalRBF(
            **common,
            enhanced_backbone=enhanced,
            query_latent_readout=query_latent_readout,
            query_readout_type=str(
                _get(
                    resolved,
                    "query_readout_type",
                    "coord" if enhanced or query_latent_readout else "point",
                )
            ),
            query_readout_scale_init=float(
                _get(resolved, "query_readout_scale_init", 1.0e-2 if enhanced else 0.0)
            ),
            enhanced_head_norm=bool(_get(resolved, "enhanced_head_norm", enhanced)),
        )
    return PointCloudFFM(
        backbone, prior, sigma_min=float(_get(resolved, "sigma_min", 1.0e-4))
    ).to(torch.device(device))
