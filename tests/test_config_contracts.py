"""Focused Phase-1 checks for config separation and shared dataclass shapes."""

from pathlib import Path

import pytest
import torch

from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.config.validate import validate_config
from phycoflow_reconstruction.contracts import (
    CoherenceComponentSpec,
    CoherenceFamilySpec,
    ObservationBatch,
)
from phycoflow_reconstruction.registry import Registry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIGS = tuple(
    path
    for path in sorted((PROJECT_ROOT / "Cases").glob("*/configs/base/*.yaml"))
    if path.name not in {"plain_defaults.yaml", "latent_fm_stage2.yaml"}
)
LATENT_STAGE2_TEMPLATES = tuple(
    sorted((PROJECT_ROOT / "Cases").glob("*/configs/base/latent_fm_stage2.yaml"))
)


def _base_config():
    return {
        "stage": "base_training",
        "case": "brusselator",
        "dataset": {"path": "unused.h5"},
        "model": {"name": "pointcloud_ffm", "backbone": "gl_rbf_enh", "gather_mode": "topk_rbf"},
        "observations": {"protocol": "random_uniform", "fields": {"u": {"count": 8}}},
        "optimization": {"lr": 1e-3},
        "runtime": {"seed": 1},
        "output": {"experiment_name": "test"},
    }


def test_stage_rejects_posttraining_keys():
    config = _base_config()
    config["coherence"] = {}
    with pytest.raises(ValueError, match="forbidden"):
        validate_config(config)


@pytest.mark.parametrize(
    "path", BASE_CONFIGS, ids=lambda path: f"{path.parents[2].name}-{path.stem}"
)
def test_every_case_base_config_satisfies_the_strict_schema(path):
    validate_config(load_config(path))


@pytest.mark.parametrize("path", LATENT_STAGE2_TEMPLATES)
def test_latent_stage2_templates_require_an_explicit_stage1_checkpoint(path):
    with pytest.raises(ValueError, match="stage1_checkpoint"):
        validate_config(load_config(path))


def test_pointcloud_scope_is_strict():
    config = _base_config()
    config["model"]["gather_mode"] = "topk_rbf_glres"
    with pytest.raises(ValueError, match="topk_rbf"):
        validate_config(config)


def test_base_sections_reject_misspelled_or_invalid_nested_values():
    config = _base_config()
    config["optimization"]["learning_rate"] = 1e-3
    with pytest.raises(ValueError, match="unknown optimization keys"):
        validate_config(config)

    config = _base_config()
    config["observations"]["fields"]["u"] = {"count_min": 4, "count_max": 2}
    with pytest.raises(ValueError, match="invalid count range"):
        validate_config(config)


def test_observation_batch_contract():
    batch = ObservationBatch(
        obs_coords=torch.zeros(2, 3, 2),
        obs_values=torch.zeros(2, 3, 1),
        obs_field_ids=torch.zeros(2, 3, dtype=torch.long),
        obs_valid_mask=torch.ones(2, 3, dtype=torch.bool),
        query_coords=torch.zeros(2, 5, 2),
        query_valid_mask=torch.ones(2, 5, dtype=torch.bool),
        target_fields=torch.zeros(2, 5, 1),
        sample_ids=("a", "b"),
    )
    batch.validate()


def test_registry_rejects_duplicates():
    registry = Registry("test")
    registry.register("x", lambda: 1)
    with pytest.raises(KeyError, match="duplicate"):
        registry.register("x", lambda: 2)


def test_coherence_family_keeps_component_taxonomy_nested():
    family = CoherenceFamilySpec(
        name="global_distribution",
        version="1",
        components=(
            CoherenceComponentSpec("self.marginal_w2", "training_reference", "model_units", True),
            CoherenceComponentSpec(
                "mutual.pairwise_swd", "training_reference", "model_units", True
            ),
            CoherenceComponentSpec(
                "cross.joint_topk_swd", "training_reference", "model_units", True
            ),
        ),
    )
    family.validate()
    assert family.name == "global_distribution"


def test_target_free_posttraining_requires_reference_bank():
    config = {
        **_base_config(),
        "stage": "post_training",
        "source_run": "/tmp/source",
        "source_checkpoint": "last.pt",
        "source": {"kind": "native_run"},
        "objectives": {
            "data_retention": {"enabled": True, "weight": 0.1},
            "coherence": {"enabled": True, "weight": 1.0},
        },
        "coherence": {
            "schedule": {"start_epoch": 1, "every_n_steps": 1},
            "compute_budget": {"batch_size": 1, "point_count": 8},
            "families": {
                "global_distribution": {
                    "target_use": "training_reference",
                    "reference_bank": {"enabled": False},
                    "components": {"self": {"enabled": True, "weight": 1.0}},
                }
            },
        },
        "rollout": {"steps": 1, "solver": "euler"},
        "observation_consistency": {"mode": "none"},
        "trainable": {"scope": "full_model"},
    }
    with pytest.raises(ValueError, match="enabled reference bank"):
        validate_config(config)


def test_direct_and_post_physics_stages_are_separate_and_valid():
    physics = {
        "provider": "brusselator_periodic",
        "temporal_derivative_source": "paired_finite_difference",
        "weights": {"pde_u": 1.0, "pde_v": 1.0},
    }
    direct = {
        **_base_config(),
        "stage": "direct_physics",
        "model": {"name": "pinn"},
        "physics": physics,
    }
    validate_config(direct)

    post = {
        **_base_config(),
        "stage": "post_training",
        "model": {"name": "coordinate_mlp"},
        "source_run": "/tmp/immutable-source",
        "source_checkpoint": "last.pt",
        "source": {"kind": "native_run"},
        "physics": physics,
        "objectives": {
            "data_retention": {"enabled": True, "weight": 0.1},
            "physics": {"enabled": True, "weight": 1.0},
        },
        "rollout": {"steps": 1, "solver": "euler"},
        "observation_consistency": {"mode": "none"},
        "trainable": {"scope": "full_model"},
    }
    validate_config(post)
    post["coherence"] = {}
    with pytest.raises(ValueError):
        validate_config(post)
