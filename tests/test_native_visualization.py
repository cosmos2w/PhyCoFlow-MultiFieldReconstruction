"""Saved native atlases preserve physical masks, comparisons, and State meaning."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from phycoflow_reconstruction.coherence import build_coherence_family
from phycoflow_reconstruction.coherence.measures import (
    measure_fingerprint,
    rectilinear_nodal_weights,
)
from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.evaluation.canonical_coherence import content_sha256
from phycoflow_reconstruction.evaluation.native_coherence import _digest_arrays
from phycoflow_reconstruction.evaluation.native_visualization import (
    _CONTRACT_ARRAYS,
    _metric_label,
    render_native_atlas,
    render_saved_native_atlas,
)


def _fixture(tmp_path):
    names = ("a", "b", "c", "d", "e")
    units = ("m", "K", "unknown", "kg", "Pa")
    shape = (3, 4)
    spec = DataSpec(names, units, 2, shape)
    y, x = torch.meshgrid(torch.tensor([0., 1., 4.]), torch.tensor([0., .3, 1.7, 3.]), indexing="ij")
    raw = torch.stack((x, y), dim=-1).reshape(-1, 2)
    coordinates = raw / 4
    mask = torch.ones(12, dtype=torch.bool)
    mask[5] = False
    native_weights = rectilinear_nodal_weights(raw.double(), shape)
    receipt = measure_fingerprint(native_weights, mask=mask, metadata={"provider": "test_area", "coordinate_units": "m"})
    physical_weights = np.where(mask.numpy(), native_weights.numpy(), 0.)
    physical_weights = physical_weights / physical_weights.sum()
    normalizer = FieldNormalizer(torch.tensor([2., -1., 0., .5, 3.]), torch.tensor([2., 3., 1., .5, 4.]))
    config = load_config("configs/coherence/state_mixed_weighted.yaml")["coherence"]["families"]["global_distribution"]
    config["fields"] = list(names)
    config["components"]["mutual"]["pairs"] = [["b", "c"]]
    state = build_coherence_family("global_distribution", config, spec, normalizer)
    state.bind_measure_artifact(receipt, weights=native_weights, mask=mask)
    checkpoint_sha = "a" * 64
    bundle = {
        "version": "native_sss_bundle_v1", "query_policy": "full_native",
        "data_spec": asdict(spec), "normalization": normalizer.state_dict(),
        "normalizer_digest": normalizer.digest(), "configs": {"global_distribution": config},
        "states": {"global_distribution": state.state_artifact()}, "measure": receipt,
        "native_weights": native_weights, "native_mask": mask, "native_ids": torch.arange(12),
        "raw_coordinates": raw, "coordinates": coordinates[None],
        "source_checkpoint_sha256": checkpoint_sha,
    }
    bundle["sha256"] = content_sha256(bundle)
    bundle_path = tmp_path / "bundle.pt"
    torch.save(bundle, bundle_path)

    target = np.arange(60, dtype=np.float32).reshape(12, 5) / 10
    target[:, 2] = 0
    target[5, 2] = 500  # excluded from every displayed-support metric
    seeds = np.array([101, 102, 103, 104], dtype=np.int64)
    generator = np.random.default_rng(112)
    baseline = generator.normal(0, .1, size=(4, 12, 5)).astype(np.float32)
    payload_paths = {}
    for label, shift in (("SOURCE", .25), ("selected", .10), ("final", -.15)):
        directory = tmp_path / label
        directory.mkdir()
        draws = target[None] + baseline + shift
        draws[:, 5, :] = 900  # invalid point must not affect scores or plots
        payload = {
            "draws_model": draws, "target_model": target,
            "draws_physical": normalizer.decode(torch.from_numpy(draws)).double().numpy(),
            "target_physical": normalizer.decode(torch.from_numpy(target)).double().numpy(),
            "model_coordinates": coordinates.numpy(), "raw_coordinates": raw.numpy(),
            "native_ids": np.arange(12), "physical_mask": mask.numpy(),
            "physical_weights": physical_weights,
            "observation_coords": coordinates[[0, 8]].numpy(),
            "observation_values": target[[0, 8], 0, None],
            "observation_field_ids": np.array([0, 0]), "observation_indices": np.array([0, 8]),
            "observation_mask": np.array([True, True]), "draw_seeds": seeds,
            "field_names": np.asarray(names), "field_units": np.asarray(units),
            "logical_shape": np.asarray(shape), "sample_id": np.asarray("sample:17"),
            "normalizer_sha256": np.asarray(normalizer.digest()),
        }
        path = directory / "condition_00000.npz"
        np.savez_compressed(path, **payload)
        row = {
            "sample_id": "sample:17", "dataset_index": 17, "payload": path.name,
            "contract_sha256": _digest_arrays(*(payload[key] for key in _CONTRACT_ARRAYS)),
            "point_count": 12, "valid_count": 11, "physical_measure": "test_area", "draw_seeds": seeds.tolist(),
        }
        manifest = {
            "schema": "native_coherence_inference_v1", "query_policy": "full_native",
            "sampler": {"steps": 2, "solver": "euler", "draws": 4},
            "sensor_protocol": {"name": "fixture"}, "sensor_manifest_sha256": "sensor_hash",
            "field_names": list(names), "field_units": list(units), "normalizer_sha256": normalizer.digest(),
            "dataset_split": "validation", "conditions": [row],
            "checkpoint_sha256": checkpoint_sha if label == "SOURCE" else ("b" if label == "selected" else "c") * 64,
        }
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        payload_paths[label] = path
    return bundle_path, payload_paths, state


def test_native_atlas_renders_all_fields_both_views_and_frozen_state(tmp_path):
    bundle, inputs, state = _fixture(tmp_path)
    out = render_native_atlas(bundle, inputs, tmp_path / "atlas", sample_label="Synthetic panel")
    report = json.loads(Path(out["report"]).read_text())
    assert report["support_count"] == 11 and report["native_count"] == 12
    assert report["representations"] == ["member_0", "ensemble_mean_4_draws"]
    assert report["state"]["score_scope"] == "paired_member_0_only"
    assert report["state"]["ensemble_mean_scored"] is False
    assert report["state"]["score_units"] == "squared_model_units"
    assert report["state"]["pair_bank"] == [["b", "c"]]
    score = report["state"]["scores"]["SOURCE"]
    weights = score["component_weights"]
    assert score["family_total"] == pytest.approx(
        weights["self_marginal_w2"] * score["marginal_mean"]
        + weights["cross_joint_topk_swd"] * score["mixed_projection_mean"]
    )
    assert len(out["figures"]) == 12  # 5 fields × 2 views, marginals, one named pair
    with np.load(inputs["SOURCE"], allow_pickle=False) as source:
        mask = source["physical_mask"]
        weights = source["physical_weights"][mask]
        error = source["draws_physical"][0, mask, 2] - source["target_physical"][mask, 2]
        expected = np.sqrt(np.sum(weights * error**2) / weights.sum())
        assert report["fidelity"]["SOURCE"]["member_0"]["c"]["relative_l2"] is None
        assert report["fidelity"]["SOURCE"]["member_0"]["c"]["rmse"] == pytest.approx(expected)
        assert "RMSE" in _metric_label(source["draws_physical"][0, :, 2], source["target_physical"][:, 2],
                                       source["physical_weights"], mask, "unknown")
        with np.load(out["payload"], allow_pickle=False) as saved:
            np.testing.assert_allclose(saved["predictions_physical"][0, 1], source["draws_physical"].mean(axis=0))
            np.testing.assert_allclose(saved["marginal_occupation"].sum(axis=-1), 1., atol=1e-6)
            np.testing.assert_allclose(saved["pair_own_rank_occupation"].sum(axis=(-2, -1)), 1., atol=1e-6)
            np.testing.assert_array_equal(saved["state_pairs"], [["b", "c"]])
    assert state.score_definition == report["state"]["score_definition"]
    index = Path(out["index"]).read_text()
    assert index.count("![") == index.count("[PDF master]") == len(out["figures"])
    assert "| Endpoint | Family value | Marginal W2-squared mean | Mixed-projection W2-squared mean | Directions |" in index
    assert "| SOURCE |" in index and "| selected |" in index and "| final |" in index
    for relative in re.findall(r"\]\(([^)]+)\)", index):
        assert (Path(out["index"]).parent / relative).is_file()
    for figure in out["figures"]:
        with Image.open(Path(out["index"]).parent / figure["preview"]) as image:
            image.verify()
        assert (Path(out["index"]).parent / figure["pdf"]).read_bytes().startswith(b"%PDF")
    rerendered = render_saved_native_atlas(out["payload"], tmp_path / "rerendered", dpi=80)
    assert len(rerendered["figures"]) == len(out["figures"])
    assert Path(rerendered["payload"]).read_bytes() == Path(out["payload"]).read_bytes()


def test_native_atlas_accepts_source_only_qualification(tmp_path):
    bundle, inputs, _ = _fixture(tmp_path)
    out = render_native_atlas(bundle, {"SOURCE": inputs["SOURCE"]}, tmp_path / "atlas", include_ensemble_mean=False)
    report = json.loads(Path(out["report"]).read_text())
    assert report["endpoint_labels"] == ["SOURCE"]
    assert report["representations"] == ["member_0"]
    assert len(out["figures"]) == 7  # five fields, marginals and a named pair
    assert "ensemble mean are separate" not in Path(out["index"]).read_text()


@pytest.mark.parametrize("key", ["sample_id", "draw_seeds", "physical_mask", "target_model", "normalizer_sha256"])
def test_native_atlas_refuses_tampered_or_unmatched_contracts(tmp_path, key):
    bundle, inputs, _ = _fixture(tmp_path)
    path = inputs["selected"]
    with np.load(path, allow_pickle=False) as source:
        payload = {name: source[name].copy() for name in source.files}
    if key in {"sample_id", "normalizer_sha256"}:
        payload[key] = np.asarray("different")
    elif key == "physical_mask":
        payload[key][0] = False
    elif key == "draw_seeds":
        payload[key][0] += 1
    else:
        payload[key][0, 0] += 1
    np.savez_compressed(path, **payload)
    # Refresh the payload hash to exercise the actual matching/bundle checks,
    # rather than relying solely on a stale manifest checksum.
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["conditions"][0]["contract_sha256"] = _digest_arrays(*(payload[name] for name in _CONTRACT_ARRAYS))
    manifest["conditions"][0]["draw_seeds"] = payload["draw_seeds"].tolist()
    manifest["conditions"][0]["sample_id"] = str(payload["sample_id"].item())
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        render_native_atlas(bundle, inputs, tmp_path / "rejected")
