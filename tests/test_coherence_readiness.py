"""Permanent regression coverage for shared coherence orchestration readiness."""

from __future__ import annotations

import copy

import pytest
import torch
from test_config_contracts import _base_config

from phycoflow_reconstruction.coherence import ReferenceBank
from phycoflow_reconstruction.config.validate import validate_config
from phycoflow_reconstruction.contracts import ObservationBatch
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
