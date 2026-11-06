"""Exercise CoastalBench's configured rollout against its actual model adapter."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from helpers.pointcloud import _small_core_config

from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.contracts import ObservationBatch
from phycoflow_reconstruction.models.flows.pointcloud.adapters.gl_rbf_cq_adapter import GLRbfCQ
from phycoflow_reconstruction.training.rollout_contract import verify_rollout_contract


@pytest.mark.parametrize("recipe", ["unified", "unified_fidelity"])
@pytest.mark.parametrize("clustered", [False, True])
def test_coastal_three_family_recipe_matches_production_guidance(recipe, clustered):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / f"cases/coastal_bench/configs/posttrain/{recipe}.yaml")
    torch.manual_seed(923)
    model_config = _small_core_config(2)
    model_config.update(config["model"])
    model_config.update(prior="rff", rff_features=16, reconstruction_query_chunk_size=5)
    count = 33 if clustered else 25
    model = GLRbfCQ(2, 4, (count,), **model_config).eval()
    coords = torch.rand(1, count, 2)
    if clustered:
        # Full-query cdist takes its matrix-multiply path above 25 queries;
        # production chunks use direct distances. Close sensors expose changed
        # top-k choices instead of hiding the discrepancy in roundoff tolerance.
        coords = 0.5 + coords * 0.0005
    sensor_indices = torch.tensor([[0, 6, 13, 22]])
    batch = ObservationBatch(
        obs_coords=coords[:, sensor_indices[0]],
        obs_values=torch.tensor([[[0.2], [-0.3], [1.2], [0.7]]]),
        obs_field_ids=torch.tensor([[1, 1, 2, 2]]),
        obs_valid_mask=torch.ones(1, 4, dtype=torch.bool),
        query_coords=coords,
        query_valid_mask=torch.ones(1, count, dtype=torch.bool),
        target_fields=None,
        sample_ids=("coastal_fixture",),
        obs_indices=sensor_indices,
        metadata={"query_indices": torch.arange(count)[None]},
    )
    for key, value in config["observation_consistency"].items():
        assert value == getattr(model, f"obs_consistency_{key}")
    # Both guidance parameters and production query chunk boundaries must agree.
    # The external-policy runtime checks its declared endpoint operator against
    # itself. Independently check this case recipe against production inference.
    config = deepcopy(config)
    config["optimization"]["multitask"].pop("external_policy", None)
    mismatched = deepcopy(config)
    mismatched["observation_consistency"] = {"mode": "none", "final_clamp": False}
    with pytest.raises(ValueError, match="training/inference rollout mismatch"):
        verify_rollout_contract(model, batch, mismatched)
    state = torch.random.get_rng_state()
    report = verify_rollout_contract(model, batch, config)
    assert report["passed"] and report["training_path_has_autograd"]
    assert report["steps"] == 16
    assert torch.equal(state, torch.random.get_rng_state())
