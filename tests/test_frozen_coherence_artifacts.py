"""Frozen family loading uses the calibrated data and query contract."""
from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import pytest
import torch
from test_multiscale_relation import config as relation_config

from phycoflow_reconstruction.coherence.registry import build_coherence_family
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.manifest import dataset_fingerprint
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.data.training_batches import fixed_query_indices
from phycoflow_reconstruction.training.coherence_sampling import (
    _fit_source_calibrations,
    _fit_training_references,
)
from phycoflow_reconstruction.training.run_store import file_sha256


class Dataset:
    def __init__(self, path):
        self.path = path
        self.split_name = "train"
        self.selection = SimpleNamespace(split="train")
        self.data_spec = DataSpec(("u", "v"), ("1", "1"), 1, (16,))
        self.normalizer = FieldNormalizer.identity(2)
        self.coordinates = torch.linspace(0, 1, 16, dtype=torch.float64)[:, None] ** 1.3
        self.values = torch.randn(8, 16, 2, generator=torch.Generator().manual_seed(23), dtype=torch.float64)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return SimpleNamespace(coordinates=self.coordinates, coordinates_raw=self.coordinates,
            values=self.values[index], valid_points=None, metadata={}, trajectory_id=str(index), time_index=None)


def _family(dataset):
    return build_coherence_family("multiscale_spectrum", relation_config(), dataset.data_spec, dataset.normalizer)


def _fixture(tmp_path):
    path = tmp_path / "dataset.dat"
    path.write_bytes(b"training data")
    dataset = Dataset(path)
    cfg = {"coherence": {"compute_budget": {"point_count": 12, "query_seed": 7}},
           "observations": {}, "runtime": {}}
    fitted = _family(dataset)
    fitted.fit_training_reference(dataset, query_indices=fixed_query_indices(16, 12, seed=7), device=torch.device("cpu"))
    artifact = tmp_path / "family.pt"
    state = fitted.state_artifact()
    query = fixed_query_indices(16, 12, seed=7)
    state["frozen_training_context"] = {"dataset_fingerprint": dataset_fingerprint(dataset.path),
        "query_indices_sha256": hashlib.sha256(query.numpy().tobytes()).hexdigest(),
        "normalizer_digest": dataset.normalizer.digest(),
        "field_names": list(dataset.data_spec.field_names), "split": "train"}
    torch.save(state, artifact)
    cfg["coherence"]["frozen_family_artifacts"] = {"multiscale_spectrum": {"path": str(artifact), "sha256": file_sha256(artifact)}}
    return dataset, cfg, fitted


def _load(dataset, cfg, family):
    _fit_training_references({"multiscale_spectrum": family}, dataset, cfg,
                             device=torch.device("cpu"), structured_grid_required=False)


def test_frozen_relation_loads_without_fitting_and_rejects_changed_hash_or_query(tmp_path):
    dataset, cfg, fitted = _fixture(tmp_path)
    restored = _family(dataset)
    restored.fit_training_reference = lambda *a, **kw: pytest.fail("frozen artifact was refit")
    _load(dataset, cfg, restored)
    assert restored.fitted and restored.calibration_record == fitted.calibration_record
    for name, value in fitted.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
    bad = copy.deepcopy(cfg)
    bad["coherence"]["frozen_family_artifacts"]["multiscale_spectrum"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        _load(dataset, bad, _family(dataset))
    bad = copy.deepcopy(cfg)
    bad["coherence"]["compute_budget"]["query_seed"] = 8
    with pytest.raises(ValueError, match="data/query/normalizer binding"):
        _load(dataset, bad, _family(dataset))


def test_frozen_relation_rejects_dataset_and_normalization_change(tmp_path):
    dataset, cfg, _ = _fixture(tmp_path)
    dataset.path.write_bytes(b"changed training data")
    with pytest.raises(ValueError, match="data/query/normalizer binding"):
        _load(dataset, cfg, _family(dataset))
    dataset.path.write_bytes(b"training data")
    dataset.normalizer = FieldNormalizer(torch.tensor([1., 0.]), torch.ones(2), "none")
    with pytest.raises(ValueError, match="normaliz"):
        _load(dataset, cfg, _family(dataset))


def test_frozen_source_calibration_checks_source_and_skips_inference():
    record = {"provenance": {"source_checkpoint_sha256": "source-a"}}
    family = SimpleNamespace(needs_source_calibration=True, spatial_objective=SimpleNamespace(calibration_record=record))
    cfg = {"coherence": {"frozen_family_artifacts": {"topology": {}}}}
    _fit_source_calibrations({"topology": family}, None, None, cfg, device=torch.device("cpu"), source_checkpoint_sha256="source-a")
    with pytest.raises(ValueError, match="SOURCE calibration identity"):
        _fit_source_calibrations({"topology": family}, None, None, cfg, device=torch.device("cpu"), source_checkpoint_sha256="source-b")


def test_frozen_relation_checks_live_coordinates_and_constructor_normalization(tmp_path):
    dataset, cfg, _ = _fixture(tmp_path)
    dataset.coordinates = dataset.coordinates + 0.01
    with pytest.raises(ValueError, match="coordinate"):
        _load(dataset, cfg, _family(dataset))
    dataset.coordinates = dataset.coordinates - 0.01
    other = FieldNormalizer(torch.tensor([1., 0.]), torch.ones(2), "none")
    family = build_coherence_family("multiscale_spectrum", relation_config(), dataset.data_spec, other)
    with pytest.raises(ValueError, match="normalization mismatch"):
        _load(dataset, cfg, family)


def test_frozen_relation_rejects_nontraining_split(tmp_path):
    dataset, cfg, _ = _fixture(tmp_path)
    dataset.split_name = "validation"
    dataset.selection.split = "validation"
    with pytest.raises(ValueError, match="TRAIN|training|train"):
        _load(dataset, cfg, _family(dataset))
