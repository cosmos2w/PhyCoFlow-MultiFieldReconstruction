"""Permanent regression coverage for shared coherence orchestration readiness."""

from __future__ import annotations

import copy

import pytest
import torch
from test_config_contracts import _base_config

from phycoflow_reconstruction.coherence import ReferenceBank, build_coherence_family
from phycoflow_reconstruction.config.validate import validate_config
from phycoflow_reconstruction.contracts import DataSpec, ObservationBatch
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.data.training_batches import fixed_query_indices
from phycoflow_reconstruction.training.post_training import _coherence_objective


def _post_config() -> dict:
    config = _base_config()
    config.update({
        "stage": "post_training", "source_run": "/tmp/source", "source_checkpoint": "last.pt",
        "source": {"kind": "native_run"},
        "objectives": {"data_retention": {"enabled": True, "weight": 0.1}, "coherence": {"enabled": True, "weight": 1.0}},
        "coherence": {"schedule": {}, "compute_budget": {"batch_size": 3, "point_count": 8, "query_policy": "fixed_shared"}, "families": {
            "cross_spectrum": {"enabled": True, "weight": 1.0, "target_use": "paired_supervised", "components": {
                "same_frequency": {"enabled": True, "weight": 1.0}, "cross_frequency": {"enabled": True, "weight": 1.0}, "band_energy": {"enabled": False, "weight": 0.0}
            }}
        }},
        "rollout": {"steps": 1, "solver": "euler"}, "observation_consistency": {"mode": "none"},
        "trainable": {"scope": "full_model"},
    })
    config["optimization"]["batch_size"] = 3
    config["evaluation"] = {"max_samples": 3}
    return config


def _topology_family() -> dict:
    """The default topology term, in the shape `validate_config` must accept."""
    return {
        "enabled": True,
        "weight": 1.0,
        "target_use": "paired_supervised",
        "fields": ["u", "v"],
        "geometry": {
            "grid_shape": [4, 4],
            "axes": [0, 1],
            "neighbors": 4,
            "power": 2.0,
            "periodic": False,
        },
        "filtration": {
            "dimensions": [0, 1],
            "directions": ["sublevel", "superlevel"],
            "smoothing_sigma": 0.0,
        },
        "matching": {
            "order": 1.0,
            "lambda_spatial": 1.0,
            "spatial_mode": "multiplicative",
            "min_persistence": 0.01,
        },
        "components": {
            "self": {"enabled": True, "weight": 1.0},
            "mutual": {
                "enabled": True,
                "weight": 1.0,
                "pairs": [["u", "v"]],
                "lines": 8,
                "line_sampling": "stratified",
                "angle_margin": 0.12,
                "seed": 0,
            },
        },
    }


def _persistence_post_config() -> dict:
    config = _post_config()
    config["coherence"]["families"] = {
        "topology": _topology_family()
    }
    return config


def test_topology_is_an_accepted_post_training_family():
    validate_config(_persistence_post_config())


def test_topology_requires_fixed_shared_queries():
    config = _persistence_post_config()
    config["coherence"]["compute_budget"]["query_policy"] = "random_per_sample"
    with pytest.raises(ValueError, match="query_policy=fixed_shared"):
        validate_config(config)


@pytest.mark.parametrize("key", ["quantiles", "sharpness"])
def test_topology_rejects_betti_only_filtration_keys(key: str) -> None:
    """The two topology terms are distinct definitions, not interchangeable spellings."""
    config = _persistence_post_config()
    config["coherence"]["families"]["topology"]["filtration"][key] = 1.0
    with pytest.raises(ValueError, match="topology.filtration"):
        validate_config(config)


@pytest.mark.parametrize("key", ["solver", "solver_tolerance"])
def test_topology_names_retired_matching_keys(key: str) -> None:
    config = _persistence_post_config()
    config["coherence"]["families"]["topology"]["matching"][key] = "exact"
    with pytest.raises(ValueError, match="were removed"):
        validate_config(config)


def test_topology_matching_bounds_are_enforced():
    config = _persistence_post_config()
    config["coherence"]["families"]["topology"]["matching"]["order"] = 0.0
    with pytest.raises(ValueError, match="order>0"):
        validate_config(config)
    config = _persistence_post_config()
    config["coherence"]["families"]["topology"]["matching"][
        "min_persistence"
    ] = 1.0
    with pytest.raises(ValueError, match=r"min_persistence must lie in \[0, 1\)"):
        validate_config(config)


def test_topology_rejects_periodic_geometry():
    config = _persistence_post_config()
    config["coherence"]["families"]["topology"]["geometry"]["periodic"] = True
    with pytest.raises(ValueError, match="only nonperiodic geometry"):
        validate_config(config)


def test_registry_schemas_cover_exactly_the_registered_families():
    from phycoflow_reconstruction.coherence.registry import family_schemas
    from phycoflow_reconstruction.registry import COHERENCE_FAMILY_REGISTRY

    schemas = family_schemas()
    assert set(schemas) == set(COHERENCE_FAMILY_REGISTRY.names())
    assert {"global_distribution", "cross_spectrum", "topology"} <= set(schemas)


def test_full_domain_fixed_queries_are_explicit_and_ordered():
    assert torch.equal(fixed_query_indices(5, None, seed=9), torch.arange(5))
    assert torch.equal(fixed_query_indices(5, 99, seed=9), torch.arange(5))


def test_strict_weights_allow_only_disabled_zero_and_require_effective_family():
    config = _post_config()
    validate_config(config)
    broken = copy.deepcopy(config)
    broken["coherence"]["families"]["cross_spectrum"]["weight"] = 0.0
    with pytest.raises(ValueError, match="family.*positive"):
        validate_config(broken)
    broken = copy.deepcopy(config)
    broken["coherence"]["families"]["cross_spectrum"]["components"]["same_frequency"]["weight"] = 0.0
    with pytest.raises(ValueError, match="component.*positive"):
        validate_config(broken)
    broken = copy.deepcopy(config)
    for component in broken["coherence"]["families"]["cross_spectrum"]["components"].values():
        component.update(enabled=False, weight=0.0)
    with pytest.raises(ValueError, match="positive enabled component"):
        validate_config(broken)


def test_self_spectrum_is_opt_in_during_config_validation():
    config = _post_config()
    components = config["coherence"]["families"]["cross_spectrum"]["components"]
    components["same_frequency"] = {"enabled": False, "weight": 0.0}
    components["cross_frequency"] = {"enabled": False, "weight": 0.0}
    components["band_energy"] = {"enabled": False, "weight": 0.0}
    components["self_spectrum"] = {"weight": 1.0}

    with pytest.raises(ValueError, match="positive enabled component"):
        validate_config(config)

    components["self_spectrum"]["enabled"] = True
    validate_config(config)


@pytest.mark.parametrize("family_name", ["global_distribution", "cross_spectrum"])
@pytest.mark.parametrize("weight", [0.0, -1.0])
def test_direct_family_constructors_require_positive_outer_weight(
    family_name: str, weight: float
) -> None:
    configs = {
        "global_distribution": {
            "weight": weight,
            "fields": ["u", "v"],
            "components": {
                "self": {"enabled": True, "weight": 1.0},
                "mutual": {"enabled": False, "weight": 0.0},
                "cross": {"enabled": False, "weight": 0.0},
            },
        },
        "cross_spectrum": {
            "weight": weight,
            "fields": ["u", "v"],
            "pairs": [["u", "v"]],
            "graph": {"num_modes": 2, "bands": ["low", "high"]},
            "components": {
                "same_frequency": {"enabled": True, "weight": 1.0},
                "cross_frequency": {"enabled": False, "weight": 0.0},
                "band_energy": {"enabled": False, "weight": 0.0},
            },
        },
    }
    data_spec = DataSpec(("u", "v"), ("1", "1"), 2, (2, 2))
    with pytest.raises(ValueError, match=rf"{family_name}\.weight must be positive"):
        build_coherence_family(
            family_name,
            configs[family_name],
            data_spec,
            FieldNormalizer.identity(2),
        )


def test_active_spectral_components_set_evaluation_ensemble_minimum():
    config = _post_config()
    config["evaluation"]["max_samples"] = 2
    with pytest.raises(ValueError, match="evaluation.max_samples must be >= 3"):
        validate_config(config)
    config["coherence"]["families"]["cross_spectrum"]["components"]["cross_frequency"] = {"enabled": False, "weight": 0.0}
    validate_config(config)


def test_reference_selection_excludes_current_ids_deterministically_and_fails_strictly():
    bank = ReferenceBank(torch.arange(3.0).view(3, 1, 1), ("a", "b", "c"), torch.zeros(3, 1, dtype=torch.long), {"split": "train"})
    kwargs = {"step": 1, "device": torch.device("cpu"), "dtype": torch.float32, "current_sample_ids": ("a", "b"), "strict_distinct": True}
    first = bank.select(2, **kwargs)
    second = bank.select(2, **kwargs)
    assert first[1] == second[1] == ("c", "c")
    with pytest.raises(ValueError, match="distinct"):
        bank.select(3, step=0, device=torch.device("cpu"), dtype=torch.float32, current_sample_ids=("a", "b", "c"), strict_distinct=True)


def test_coherence_rejects_any_invalid_selected_query_before_family_execution():
    batch = ObservationBatch(
        obs_coords=torch.zeros(1, 1, 1), obs_values=torch.zeros(1, 1, 1),
        obs_field_ids=torch.zeros(1, 1, dtype=torch.long), obs_valid_mask=torch.ones(1, 1, dtype=torch.bool),
        query_coords=torch.zeros(1, 2, 1), query_valid_mask=torch.tensor([[True, False]]),
        target_fields=torch.zeros(1, 2, 1), sample_ids=("sample",),
        metadata={"query_indices": torch.arange(2).view(1, 2)},
    )
    with pytest.raises(ValueError, match="every selected query_valid_mask"):
        _coherence_objective(
            object(), batch, object(), None,
            {"coherence": {"compute_budget": {"batch_size": 1, "point_count": 2}}},
            step=0, generator=torch.Generator().manual_seed(1),
        )
