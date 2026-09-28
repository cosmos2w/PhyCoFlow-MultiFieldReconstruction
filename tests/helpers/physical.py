"""Physical-field fixtures with pooled sensors and nonlinear coordinates."""

import torch

from helpers.pointcloud import _small_core_config
from phycoflow_reconstruction.contracts import FieldSample


def sample():
    y, x = torch.meshgrid(torch.linspace(0, 1, 8), torch.linspace(0, 1, 8), indexing="ij")
    coords = torch.stack((x, y, x * 0), -1).reshape(-1, 3)
    fields = torch.stack((x + y, (x + 2 * y).square(), torch.sin(9 * x) * y), -1).reshape(-1, 3)
    return FieldSample(
        fields,
        coords,
        coords,
        torch.tensor(0.0),
        "run1",
        0,
        torch.empty(0),
        ("phi", "vx", "vy"),
        (8, 8),
    )


def model_config():
    config = _small_core_config(3)
    config.pop("model_name")
    config.pop("coord_dim")
    config.update(
        name="gl_rbf_cq",
        model_ema_enabled=False,
        obs_consistency_mode="none",
        obs_consistency_final_clamp=False,
        ode_solver="euler",
        physical_field_transform={
            "offset": [0.1, -0.2, 0.3],
            "scale": [0.5, 2.0, 0.4],
            "asinh_scale": [1.0, 0.3, 0.2],
            "linear_mask": [True, False, False],
        },
    )
    return config


def post_config(dataset_path, source_run, *, context_cache="static_features"):
    """Two-update native persistence run for value, gradient, and recovery tests."""
    from helpers.coherence import _post_config

    config = _post_config(dataset_path, source_run)
    config["dataset"].update(
        field_names=["phi", "vx", "vy"],
        field_units=["1"] * 3,
        normalization="none",
        coordinate_dim=3,
        grid_shape=[8, 8],
        augmentation={"kind": "periodic_translate_rot90", "vector_fields": ["vx", "vy"]},
    )
    config["model"] = {**model_config(), "query_points": 64, "data_query_points": 16}
    config["observations"] = {
        "protocol": "structured_block_mean",
        "fields": {"phi": {"count": 16}, "vx": {"count": 4}, "vy": {"count": 4}},
    }
    config["objectives"]["data_retention"]["enabled"] = False
    config["optimization"].update(
        epochs=2,
        batch_size=1,
        steps_per_epoch=1,
        sampling="full_pass",
        model_mode="eval",
        lr=1e-6,
        gradient_balance="topology_regularized",
        component_constraints={"calibration_batches": 1, "topology_reduction": "batch_mean"},
        rollout_execution={"context_cache": context_cache, "checkpointing": True},
    )
    config["coherence"]["compute_budget"] = {
        "batch_size": 1,
        "point_count": 64,
        "query_policy": "fixed_shared",
    }
    config["coherence"]["families"] = {
        "topology": {
            "strategy": "cubical_persistence",
            "target_use": "paired_supervised",
            "units": "physical_units",
            "fields": ["phi", "vx", "vy"],
            "geometry": {
                "grid_shape": [8, 8],
                "axes": [0, 1],
                "periodic": True,
                "periods": [8 / 7, 8 / 7],
                "antialias_downsample": True,
            },
            "filtration": {"smoothing_sigma": 0.0},
            "persistence": {
                "distance": "sliced_wasserstein",
                "projections": 4,
                "descriptors": {
                    "vorticity": {"provider": "signed_vorticity", "fields": ["vx", "vy"]}
                },
            },
            "components": {
                "self": {"enabled": True},
                "mutual": {"enabled": True, "groups": [["phi", "vorticity"]], "lines": 2},
            },
        }
    }
    config["rollout"]["steps"] = 2
    config["observation_consistency"] = {"mode": "none", "final_clamp": False}
    config["evaluation"].update(
        max_samples=2,
        query_points=64,
        generation_steps=2,
        sample_selection="uniform",
        native_topology=True,
        preview={"enabled": False},
    )
    config["checkpointing"] = {
        "every_epochs": 1,
        "every_steps": 1,
        "validation_every_epochs": 2,
        "selection_metric": "topology_with_fidelity",
    }
    config["posttrain_fidelity"] = {
        "max_relative_mse_increase": 0.05,
        "max_relative_field_mse_increase": 0.05,
        "max_relative_native_topology_increase": 0.0,
    }
    return config
