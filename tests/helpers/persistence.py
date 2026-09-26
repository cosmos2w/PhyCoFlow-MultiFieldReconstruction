"""Compact persistence-family configuration."""


def config():
    return {
        "strategy": "cubical_persistence",
        "target_use": "paired_supervised",
        "fields": ["a", "b", "c"],
        "geometry": {"periodic": True},
        "filtration": {"smoothing_sigma": 0},
        "components": {
            "self": {"enabled": True},
            "mutual": {"enabled": True, "groups": [["a", "b", "c"]], "lines": 4},
        },
    }
