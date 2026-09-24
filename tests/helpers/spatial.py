"""Analytic fields for spatial topology compatibility tests."""

import torch

from phycoflow_reconstruction.coherence.families.topology.family import TopologyFamily
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer


def _config(size=12):
    return {
        "strategy": "spatial_self_mutual",
        "target_use": "paired_supervised",
        "units": "physical_units",
        "fields": ["phi"],
        "geometry": {"grid_shape": [size, size], "periodic": True, "neighbors": 1},
        "filtration": {
            "level_mode": "physical",
            "physical_levels": [-0.4, 0, 0.4],
            "quantiles": [0.3, 0.5, 0.7, 0.9],
            "directions": ["superlevel", "sublevel"],
            "dimensions": [0, 1],
            "smoothing_sigma": 0,
            "sharpness": 8,
        },
        "anchor": {"provider": "vorticity", "fields": ["vx", "vy"], "sharpness": 12},
        "components": {
            "self": {"enabled": True, "weight": 1, "cldice_weight": 0.25},
            "anchor_self": {"enabled": True, "weight": 0.2},
            "mutual": {
                "enabled": True,
                "weight": 1,
                "carrier_field": "phi",
                "carrier_gauge": "interface",
                "gradient_scale": 0.2,
                "detach_carrier": True,
                "lines": 4,
            },
        },
    }


def _family(config=None, normalizer=None):
    config = _config() if config is None else config
    shape = tuple(config["geometry"]["grid_shape"])
    return TopologyFamily(
        config,
        DataSpec(("phi", "vx", "vy"), ("1", "1", "1"), 2, shape),
        FieldNormalizer.identity(3) if normalizer is None else normalizer,
    )


def _fields(size=12):
    axis = torch.arange(size, dtype=torch.float32) / size
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    coords = torch.stack((x, y), -1).flatten(0, 1)[None]
    reference = torch.stack(
        (
            torch.cos(2 * torch.pi * x) * torch.cos(2 * torch.pi * y),
            torch.sin(2 * torch.pi * y),
            torch.cos(4 * torch.pi * x),
        ),
        -1,
    )[None]
    prediction = reference.roll(2, 2) + 0.1 * reference.roll(1, 1)
    return coords, prediction.flatten(1, 2), reference.flatten(1, 2)
