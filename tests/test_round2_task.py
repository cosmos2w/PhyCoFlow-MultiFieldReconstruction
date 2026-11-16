import hashlib

import pytest
import torch

from phycoflow_reconstruction.coherence.measures import measure_fingerprint
from phycoflow_reconstruction.evaluation.round2_task import regional_covariance


def test_regional_covariance_uses_frozen_mass_and_region():
    mass = torch.tensor([1., 2., 1.], dtype=torch.float64)
    ids = torch.tensor([1, 2], dtype=torch.int64)
    manifest = {"schema": "car_round2_reserved_task_v1", "field_names": ["CO", "U_1"],
                "truth_scales": [2., 4.], "region_point_indices": ids.tolist(),
                "region_indices_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                "native_measure": measure_fingerprint(mass, metadata={"provider": "test"})}
    fields = torch.tensor([[[100., 100.], [1., 5.], [4., -1.]]], dtype=torch.float64)
    # Weighted covariance of (1,4) and (5,-1) with weights 2/3 and 1/3 is -4.
    torch.testing.assert_close(regional_covariance(fields, mass, manifest, ("CO", "U_1")),
                               torch.tensor([-0.5], dtype=torch.float64))
    with pytest.raises(ValueError, match="physical masses"):
        regional_covariance(fields, mass + 1, manifest, ("CO", "U_1"))
    altered = {**manifest, "region_point_indices": [0, 2]}
    with pytest.raises(ValueError, match="index hash"):
        regional_covariance(fields, mass, altered, ("CO", "U_1"))
