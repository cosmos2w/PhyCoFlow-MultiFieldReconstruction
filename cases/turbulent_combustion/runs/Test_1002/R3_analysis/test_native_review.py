"""Exercise R3 native review API with genuine CPU GUDHI on a small lattice."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from phycoflow_reconstruction.coherence.families.topology.family import TopologyFamily
from phycoflow_reconstruction.contracts import DataSpec, ObservationBatch
from phycoflow_reconstruction.data.normalization import FieldNormalizer

ROOT = Path(__file__).resolve().parents[5]


def test_native_train_calibration_development_and_saved_fields(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/evaluation"))
    spec = importlib.util.spec_from_file_location("r3_audit", ROOT / "scripts/evaluation/audit_upgrade_1002_r3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    axis = torch.linspace(0, 1, 5)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    coords = torch.stack((x, y), -1).reshape(-1, 2)
    normalizer = FieldNormalizer.identity(2)
    data_spec = DataSpec(field_names=("a", "b"), field_units=("unknown", "unknown"),
                         coordinate_dim=2, logical_shape=(5, 5), mesh_type="structured")

    class Dataset:
        field_names = data_spec.field_names
        def __init__(self, split):
            self.normalizer, self.data_spec, self.split = normalizer, data_spec, split
        def __getitem__(self, index):
            return SimpleNamespace(coordinates_raw=coords)
        def close(self):
            pass

    train, development = Dataset("train"), Dataset("validation")

    def panel(dataset, protocol, indices, device):
        count = len(indices)
        target = torch.stack((coords[:, 0] * coords[:, 1], (coords[:, 0]-.5)**2 + (coords[:, 1]-.5)**2), -1).expand(count, -1, -1)
        return ObservationBatch(obs_coords=coords[:1].expand(count, -1, -1),
            obs_values=torch.zeros(count, 1, 1), obs_field_ids=torch.zeros(count, 1, dtype=torch.long),
            obs_valid_mask=torch.ones(count, 1, dtype=torch.bool), obs_indices=torch.zeros(count, 1, dtype=torch.long),
            query_coords=coords.expand(count, -1, -1), query_valid_mask=torch.ones(count, 25, dtype=torch.bool),
            target_fields=target, sample_ids=tuple(f"{dataset.split}:{index}" for index in indices),
            metadata={"query_indices": torch.arange(25).expand(count, -1)})

    config = {"dataset": {"grid_shape": [5, 5]}, "evaluation": {"seed": 2027}}
    family_config = {"strategy": "cubical_persistence", "target_use": "paired_supervised", "fields": ["a", "b"],
        "units": "model_units", "geometry": {"grid_shape": [5, 5], "periodic": False},
        "filtration": {"smoothing_sigma": 0},
        "persistence": {"aggregation": "finite_primary_v1", "projections": 8},
        "components": {"self": {"enabled": True}, "mutual": {"enabled": True,
            "groups": [["a", "b"]], "line_bank_size": 16, "training_subset_size": 4}}}
    family = TopologyFamily(family_config, data_spec, normalizer)
    monkeypatch.setattr(module, "build_panel_batch", panel)
    monkeypatch.setattr(module, "open_field_dataset", lambda *args, **kwargs: train)
    monkeypatch.setattr(module, "build_native_topology_family", lambda *args: (family, {}))
    # The reused exporter resolves its own batch builder.
    monkeypatch.setattr(sys.modules["audit_upgrade_1002_r2"], "build_panel_batch", panel)

    class Model(torch.nn.Module):
        def __init__(self, shift):
            super().__init__()
            self.shift = torch.nn.Parameter(torch.tensor(shift))
            self._ema_eval = True
        def reconstruct(self, batch, **kwargs):
            assert self._ema_eval is False
            xy = batch.query_coords
            prediction = torch.stack((xy[..., 0]*xy[..., 1], (xy[...,0]-.5)**2 + (xy[...,1]-.5)**2), -1)
            return SimpleNamespace(prediction=prediction + self.shift)

    source, child = Model(.1), Model(.11)
    runtime = SimpleNamespace(dataset=development, device=torch.device("cpu"), run_dir=tmp_path,
                              model=child, generation_steps=2)
    result = module.native_review(runtime, source, config, None, tmp_path, epoch=150,
                                  development_indices=[1, 3], train_indices=[0, 2])
    assert result["source_TRAIN_calibration"]["representation"] == "native_grid"
    assert result["source_TRAIN_calibration"]["provenance"]["split"] == "train"
    assert "topology.normalized_essential" in result["components"]["candidate_live"]
    with np.load(tmp_path/result["field_arrays"]["path"], allow_pickle=False) as arrays:
        assert str(arrays["split"]) == "validation"
        assert len(arrays["sample_ids"]) == 2
    assert source._ema_eval and child._ema_eval
    assert source.shift.grad is None and child.shift.grad is None
