import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from test_multiscale_gram_extension import _fitted, _score, _states
from test_multiscale_relation import config

from phycoflow_reconstruction.coherence.families.native_quadrature import NativeQuadrature
from phycoflow_reconstruction.coherence.families.native_relation import (
    NativeRelationFamily,
    parse_native_relation_config,
)
from phycoflow_reconstruction.coherence.registry import build_coherence_family
from phycoflow_reconstruction.data.normalization import FieldNormalizer


def fixture(tmp_path, query, monkeypatch):
    monkeypatch.setattr("phycoflow_reconstruction.coherence.families.native_relation.dataset_fingerprint", lambda _: "dataset-fingerprint")
    states = _states()
    original, coords, spec = _fitted(config(), states)
    refpath = tmp_path / "native.pt"
    native_artifact = original.state_artifact()
    native_artifact["calibration"].update(split="train", dataset_fingerprint="dataset-fingerprint")
    torch.save(native_artifact, refpath)
    refsha = hashlib.sha256(refpath.read_bytes()).hexdigest()
    adapter = NativeQuadrature(
        coordinates=coords.numpy(), model_coordinates=coords.numpy(),
        node_ids=original.analysis_index.numpy(), mass=original.mass.numpy(),
        pieces=original.piece_index.numpy(), basis=original.basis.numpy(),
        filters=original.filters.numpy(), query_ids=query.numpy(), metric_identity="test-x",
        calibration_identity=refsha)
    path = tmp_path / "adapter.json"
    path.write_text(json.dumps(adapter.state_artifact(), sort_keys=True))
    pconfig = config()
    pconfig["relation_extension"].update(definition="self_rho_native_quadrature_v1",
        adapter_path=str(path), adapter_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        native_reference_path=str(refpath), native_reference_sha256=refsha)
    dataset = SimpleNamespace(split_name="train", selection=SimpleNamespace(split="train"))
    class Dataset:
        path = "training-dataset"
        split_name = dataset.split_name
        selection = dataset.selection
        def __getitem__(self, index):
            return SimpleNamespace(coordinates=coords, valid_points=None)
    family = build_coherence_family("multiscale_spectrum", pconfig, spec, FieldNormalizer.identity(2))
    assert isinstance(family, NativeRelationFamily)
    family.fit_training_reference(Dataset(), query_indices=query)
    return family, original, coords, states, spec, pconfig, Dataset()


def test_full_support_scoring_parity_and_resume(tmp_path, monkeypatch):
    query = torch.arange(16)
    family, original, coords, states, spec, pconfig, _dataset = fixture(tmp_path, query, monkeypatch)
    a, b = states[:2].clone().requires_grad_(), states[2:4].clone().requires_grad_()
    p = _score(family, coords, a, b, states[4:6])
    r = _score(original, coords, a, b, states[4:6])
    torch.testing.assert_close(p.per_sample_cost, r.per_sample_cost)
    ga = torch.autograd.grad(p.scalar_loss, (a, b), retain_graph=True)
    gb = torch.autograd.grad(r.scalar_loss, (a, b))
    for left, right in zip(ga, gb):
        torch.testing.assert_close(left, right)
    restored = build_coherence_family("multiscale_spectrum", pconfig, spec, FieldNormalizer.identity(2))
    restored.load_state_artifact(family.state_artifact())
    torch.testing.assert_close(_score(restored, coords, a, b, states[4:6]).scalar_loss, p.scalar_loss)
    artifact = copy.deepcopy(family.state_artifact())
    artifact["native_reference"]["state_dict"]["band_scale"][0, 0] *= 2
    with pytest.raises(ValueError, match="embedded reference"):
        restored.load_state_artifact(artifact)
    with pytest.raises(RuntimeError, match="pinned native"):
        restored.fit_reference_tensors(coords, states, sample_ids=[])


def test_sparse_scores_and_query_identity(tmp_path, monkeypatch):
    query = torch.arange(0, 16, 2)
    family, _, coords, states, _, cfg, _ = fixture(tmp_path, query, monkeypatch)
    parse_native_relation_config(cfg, ("u", "v"))
    x = states[:1, query].clone().requires_grad_()
    y = states[1:2, query].clone().requires_grad_()
    result = _score(family, coords[query], x, y, states[2:3, query])
    assert result.diagnostics["relation_extension"]["definition"] == "self_rho_native_quadrature_v1"
    assert all(g.norm() > 0 for g in torch.autograd.grad(result.scalar_loss, (x, y)))
    with pytest.raises(ValueError, match="coordinates"):
        _score(family, coords[query] + .001, x, y, states[2:3, query])
    bad = copy.deepcopy(cfg)
    bad["relation_extension"]["adapter_sha256"] = "x" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        parse_native_relation_config(bad, ("u", "v"))
