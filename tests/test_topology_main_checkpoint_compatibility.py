"""Original main v3 artifact and scoring regression, captured before extraction."""

import hashlib
from copy import deepcopy

import pytest
import torch

pytest.importorskip("gudhi")
from phycoflow_reconstruction.coherence.families.topology.family import (
    _MAIN_V3_SOURCE,
    TopologyFamily,
)
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer


def test_main_v3_source_restore_and_numerics():
    spec = DataSpec(("a", "b"), ("1", "1"), 2, (16,))
    config = {
        "strategy": "cubical_persistence",
        "target_use": "paired_supervised",
        "fields": ["a", "b"],
        "geometry": {"periodic": False, "grid_shape": [4, 4]},
        "filtration": {"smoothing_sigma": 0},
        "components": {
            "self": {"enabled": True},
            "mutual": {"enabled": True, "groups": [["a", "b"]], "lines": 3},
        },
    }
    g = torch.Generator().manual_seed(91)
    y = torch.randn(2, 16, 2, generator=g)
    x = torch.randn(2, 16, 2, generator=g, requires_grad=True)
    points = (
        torch.stack(
            torch.meshgrid(torch.linspace(0, 1, 4), torch.linspace(0, 1, 4), indexing="ij"), dim=-1
        )
        .reshape(1, 16, 2)
        .expand(2, -1, -1)
    )
    model = TopologyFamily(config, spec, FieldNormalizer.identity(2))
    model(x, y, coordinates=points)
    artifact = deepcopy(model.state_artifact())
    artifact["scientific_source"]["source_sha256"] = dict(_MAIN_V3_SOURCE)
    restored = TopologyFamily(config, spec, FieldNormalizer.identity(2))
    restored.load_state_artifact(artifact)
    result = restored(x, y, coordinates=points)
    grad = torch.autograd.grad(result.scalar_loss, x)[0]
    assert result.scalar_loss.item() == pytest.approx(0.13684441149234772, abs=1e-8)
    assert (
        hashlib.sha256(grad.detach().numpy().tobytes()).hexdigest()
        == "2599f832f4165f05f7478190dc2dc461e0510a8232dd8b2d7a8614db7959a466"
    )
    artifact["scientific_source"]["source_sha256"]["geometry.py"] = "0" * 64
    with pytest.raises(ValueError, match="source/backend mismatch"):
        TopologyFamily(config, spec, FieldNormalizer.identity(2)).load_state_artifact(artifact)
