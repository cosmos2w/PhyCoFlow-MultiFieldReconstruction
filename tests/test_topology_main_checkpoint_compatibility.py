"""Original main v3 artifact and scoring regression, captured before extraction."""

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
    # Captured from original main 5bbcb35 with PyTorch 2.5.1. Compare the full
    # float32 gradient numerically across supported PyTorch versions.
    expected_grad = torch.tensor(
        [
            [
                [-0.0049362923, 0.00300571183],
                [-0.00321809552, -0.000636058161],
                [0.00323456549, 0.00246437313],
                [-0.00632257387, 0.00154305308],
                [0.00632257387, 0.000560235057],
                [-8.47461561e-05, -0.00307994243],
                [0.000159050847, -0.00235698698],
                [-5.15194125e-17, -0.00169098144],
                [-0.0058148196, -0.000290241995],
                [0.00255887606, -0.000196836379],
                [-0.00561087672, -0.00162783207],
                [-0.00379434787, 0.00145212084],
                [0.00150546932, 0.000586311333],
                [-0.00411777152, -0.011310284],
                [-0.00136872567, -0.0064872317],
                [-0.00493728975, 0.00165632355],
            ],
            [
                [-0.0017208918, -0.004977006],
                [0.00154053606, 0.00312796491],
                [0.0012691794, 3.48615686e-05],
                [-0.00735755591, -0.00909354631],
                [-0.00130447256, -0.00235125585],
                [0.011273074, -0.00157088402],
                [-0.00142799388, 0.00010064213],
                [0.00107120851, -4.34661597e-05],
                [0.0118856402, -0.00589487329],
                [-0.00935321115, -0.0067950231],
                [-0.000709716347, -0.000796761422],
                [-0.00898103695, -0.000614851888],
                [0.000200503389, -0.000356860721],
                [0.00478915637, -2.42352736e-17],
                [-0.00774862105, 0.00551430648],
                [0.00219884538, 6.96254399e-17],
            ],
        ],
        dtype=grad.dtype,
    )
    torch.testing.assert_close(grad, expected_grad, rtol=1e-6, atol=1e-9)
    artifact["scientific_source"]["source_sha256"]["geometry.py"] = "0" * 64
    with pytest.raises(ValueError, match="source/backend mismatch"):
        TopologyFamily(config, spec, FieldNormalizer.identity(2)).load_state_artifact(artifact)
