from copy import deepcopy

import pytest
import torch

from phycoflow_reconstruction.coherence.families.topology.family import TopologyFamily
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer


def config(fields):
    return {
        "strategy": "cubical_persistence",
        "target_use": "paired_supervised",
        "fields": fields,
        "geometry": {"grid_shape": [4, 4]},
        "filtration": {
            "smoothing_sigma": 0,
            "dimensions": [0, 1],
            "directions": ["sublevel", "superlevel"],
        },
        "persistence": {"definition": "scalable_pooled_v1", "projections": 4},
        "components": {"self": {"enabled": True}, "mutual": {"enabled": True}},
    }


@pytest.mark.parametrize("count", [1, 2, 1000])
def test_lazy_shape_geometry_no_pair_population(count):
    fields = [f"f{i}" for i in range(count)]
    spec = DataSpec(tuple(fields), ("1",) * count, 2, (16,))
    family = TopologyFamily(config(fields), spec, FieldNormalizer.identity(count))
    assert family.version == "6" and not hasattr(family.spatial_objective, "units")
    assert ("mutual.persistence" in family.spatial_objective.weights) == (count > 1)
    assert sum(t.numel() for t in family.state_dict().values()) == 2 * count
    x = torch.zeros(1, 16, count)
    points = torch.zeros(1, 16, 2)
    with pytest.raises(ValueError, match="ScalableQueryAdapter"):
        family(x, x, coordinates=points)
    a = family.state_artifact()
    restored = TopologyFamily(config(fields), spec, FieldNormalizer.identity(count))
    restored.load_state_artifact(a)
    bad = deepcopy(a)
    bad["scientific_source"]["source_sha256"]["geometry.py"] = "0" * 64
    with pytest.raises(ValueError, match="source mismatch"):
        restored.load_state_artifact(bad)
