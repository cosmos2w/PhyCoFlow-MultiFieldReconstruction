"""Small distribution and spectral coherence configurations."""


def _global_config() -> dict:
    return {
        "target_use": "paired_supervised",
        "units": "model_units",
        "fields": ["u", "v"],
        "components": {
            "self": {"enabled": True, "weight": 1.0},
            "mutual": {
                "enabled": True,
                "weight": 1.0,
                "pairs": [["u", "v"]],
                "directions": 4,
                "seed": 3,
            },
            "cross": {
                "enabled": True,
                "weight": 1.0,
                "directions": 4,
                "top_fraction": 0.5,
                "seed": 5,
            },
        },
    }


def _cross_config() -> dict:
    return {
        "target_use": "paired_supervised",
        "units": "model_units",
        "fields": ["u", "v"],
        "pairs": [["u", "v"]],
        "graph": {
            "k_neighbors": 4,
            "num_modes": 9,
            "exclude_zero": True,
            "bands": ["low", "mid", "high"],
        },
        "components": {
            "self_spectrum": {"enabled": False, "weight": 0.0},
            "same_frequency": {"enabled": True, "weight": 1.0},
            "cross_frequency": {"enabled": True, "weight": 1.0},
            "band_energy": {"enabled": True, "weight": 0.25},
        },
    }
