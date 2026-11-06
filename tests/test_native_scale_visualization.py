"""Saved native Scale comparisons retain support, score and figure contracts."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("gudhi")

from test_native_scoring import TrainData, _config

from phycoflow_reconstruction.evaluation.native_coherence import _digest_arrays
from phycoflow_reconstruction.evaluation.native_scale_visualization import (
    _band_error,
    _matched_manifests,
    _read_native_member,
    render_native_scale,
    render_native_scale_payload,
)
from phycoflow_reconstruction.evaluation.native_scoring import fit_native_metric_bundle


def test_band_error_uses_exact_display_mask_and_physical_weights():
    reference = np.array([1.0, 4.0, 200.0])
    prediction = np.array([2.0, 6.0, -500.0])
    weights = np.array([1.0, 3.0, 100.0])
    mask = np.array([True, True, False])
    kind, value = _band_error(prediction, reference, weights, mask)
    assert kind == "relative L2"
    assert value == pytest.approx(np.sqrt((1 + 3 * 4) / (1 + 3 * 16)))
    kind, value = _band_error(prediction, np.zeros(3), weights, mask)
    assert kind == "RMSE" and value == pytest.approx(np.sqrt((2**2 + 3 * 6**2) / 4))
    with pytest.raises(ValueError, match="positive finite"):
        _band_error(prediction, reference, np.zeros(3), mask)


def test_manifest_matching_detects_sampler_or_saved_contract_change(tmp_path):
    keys = (
        "native_ids",
        "model_coordinates",
        "raw_coordinates",
        "physical_mask",
        "physical_weights",
        "target_model",
        "observation_coords",
        "observation_values",
        "observation_indices",
        "observation_field_ids",
        "observation_mask",
        "draw_seeds",
    )
    item = {key: np.asarray([i], dtype=np.int64) for i, key in enumerate(keys)}
    item["sample_id"] = np.asarray("one")
    digest = _digest_arrays(*(item[key] for key in keys))
    paths = []
    for name in ("source", "child"):
        folder = tmp_path / name
        folder.mkdir()
        path = folder / "condition.npz"
        path.write_bytes(b"fixture")
        manifest = {
            "schema": "native_coherence_inference_v1",
            "query_policy": "full_native",
            "sampler": {"steps": 2},
            "conditions": [{"payload": path.name, "sample_id": "one", "contract_sha256": digest}],
        }
        (folder / "manifest.json").write_text(json.dumps(manifest))
        paths.append(path)
    assert _matched_manifests(paths, [item, item])["sampler"] == {"steps": 2}
    changed = dict(item, draw_seeds=np.asarray([999]))
    with pytest.raises(ValueError, match="does not match"):
        _matched_manifests(paths, [item, changed])
    manifest_path = paths[1].parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sampler"]["steps"] = 3
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="sampler contract"):
        _matched_manifests(paths, [item, item])


def _write_payload(
    path, dataset, target, draws, *, sample_id="synthetic:0", seeds=(2, 5, 8, 11), mask=None
):
    mask = np.ones(16, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    weights = np.where(mask, dataset.physical_weights.numpy(), 0.0)
    np.savez_compressed(
        path,
        draws_model=draws.numpy(),
        target_model=target.numpy(),
        model_coordinates=dataset.coordinates.numpy(),
        raw_coordinates=dataset.coordinates.numpy(),
        native_ids=np.arange(16),
        physical_mask=mask,
        physical_weights=weights / weights.sum(),
        field_names=np.asarray(dataset.field_names),
        field_units=np.asarray(dataset.data_spec.field_units),
        normalizer_sha256=np.asarray(dataset.normalizer.digest()),
        sample_id=np.asarray(sample_id),
        logical_shape=np.asarray([4, 4]),
        observation_coords=dataset.coordinates[:2].numpy(),
        observation_values=target[:2, :1].numpy(),
        observation_field_ids=np.zeros(2, dtype=np.int64),
        observation_indices=np.arange(2),
        observation_mask=np.ones(2, dtype=bool),
        draw_seeds=np.asarray(seeds),
    )


def _write_manifest(paths, checkpoint_sha256):
    keys = (
        "native_ids",
        "model_coordinates",
        "raw_coordinates",
        "physical_mask",
        "physical_weights",
        "target_model",
        "observation_coords",
        "observation_values",
        "observation_indices",
        "observation_field_ids",
        "observation_mask",
        "draw_seeds",
    )
    rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            rows.append(
                {
                    "payload": path.name,
                    "sample_id": str(data["sample_id"].item()),
                    "contract_sha256": _digest_arrays(*(data[key] for key in keys)),
                }
            )
    (paths[0].parent / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "native_coherence_inference_v1",
                "query_policy": "full_native",
                "checkpoint_sha256": checkpoint_sha256,
                "conditions": rows,
            }
        )
    )


def test_masked_payload_uses_only_frozen_valid_physical_mass(tmp_path):
    dataset = TrainData(tmp_path / "train.dat")
    mask = np.ones(16, dtype=bool)
    mask[[5, 14]] = False
    target = dataset[0].values
    draws = target[None].expand(4, -1, -1).clone()
    path = tmp_path / "member.npz"
    _write_payload(path, dataset, target, draws, mask=mask)
    bundle = {
        "coordinates": dataset.coordinates[None],
        "raw_coordinates": dataset.coordinates,
        "native_ids": torch.arange(16),
        "native_mask": torch.from_numpy(mask),
        "native_weights": dataset.physical_weights,
        "data_spec": {
            "field_names": dataset.field_names,
            "field_units": dataset.data_spec.field_units,
        },
        "normalizer_digest": dataset.normalizer.digest(),
    }
    item = _read_native_member(path, bundle, member=0)
    assert np.all(item["physical_weights"][~mask] == 0)
    assert item["physical_weights"][mask].sum() == pytest.approx(1)
    assert np.ptp(item["physical_weights"][mask]) > 0  # physical measure is nonuniform

    with np.load(path, allow_pickle=False) as data:
        unmasked = {key: data[key].copy() for key in data.files}
    weights = dataset.physical_weights.numpy()
    unmasked["physical_weights"] = weights / weights.sum()
    np.savez_compressed(path, **unmasked)
    with pytest.raises(ValueError, match="physical weights differ"):
        _read_native_member(path, bundle, member=0)


def test_saved_native_scale_atlas_scores_matches_and_rerenders(tmp_path):
    dataset = TrainData(tmp_path / "train.dat")
    targets = torch.stack([dataset[i].values for i in (0, 1)])
    generator = torch.Generator().manual_seed(432)
    source_train = targets[None] + 0.15 * torch.randn((2, 2, 16, 2), generator=generator)
    bundle_path = tmp_path / "bundle.pt"
    fit_native_metric_bundle(
        _config(),
        dataset,
        source_members=source_train,
        targets=targets,
        coordinates=dataset.coordinates[None],
        sample_ids=["synthetic:0", "synthetic:1"],
        sample_indices=[0, 1],
        source_checkpoint_sha256="a" * 64,
        output_path=bundle_path,
    )
    first = targets[0][None] + 0.2 * torch.randn((4, 16, 2), generator=generator)
    second = targets[0][None] + 0.12 * torch.randn((4, 16, 2), generator=generator)
    source, child = tmp_path / "source.npz", tmp_path / "child.npz"
    _write_payload(source, dataset, targets[0], first)
    _write_payload(child, dataset, targets[0], second)
    with pytest.raises(ValueError, match="SOURCE requires.*manifest"):
        render_native_scale(bundle_path, {"SOURCE": source}, tmp_path / "bad_source")
    _write_manifest((source, child), "b" * 64)
    with pytest.raises(ValueError, match="SOURCE checkpoint differs"):
        render_native_scale(bundle_path, {"SOURCE": source}, tmp_path / "bad_source")
    _write_manifest((source, child), "a" * 64)
    output = render_native_scale(
        bundle_path, {"SOURCE": source, "Selected": child}, tmp_path / "atlas", member=1
    )
    report = json.loads(Path(output["report"]).read_text())
    assert report["member"] == 1 and report["draw_count"] == 4
    assert report["labels"] == ["SOURCE", "Selected"]
    assert report["pairs"] == [["a", "b"]]
    assert len(report["scores"]) == 2 and len(report["scores"][0]["per_band_attraction"]) == 2
    assert np.isfinite(report["scores"][0]["joint_score"])
    for score in report["scores"]:
        np.testing.assert_allclose(
            score["per_band_mean_pair_separation"], 2 * np.asarray(score["per_band_repulsion"])
        )
        assert score["joint_mean_pair_separation"] == pytest.approx(2 * score["joint_repulsion"])
    with np.load(output["payload"], allow_pickle=False) as payload:
        ref = payload["reference_bands"][0, :, 0]
        pred = payload["member_bands"][0, 0, :, 0]
        weights = payload["physical_weights"]
        mask = payload["display_mask"]
        kind, expected = _band_error(pred, ref, weights, mask)
        first_error = report["band_errors"][f"SOURCE|{report['bands'][0]}|a"]
        assert first_error["metric"] == kind and first_error["value"] == pytest.approx(expected)
        assert payload["signed_cospectrum"].shape == (3, 2, 1)
        assert payload["piece_means"].shape == (3, 1, 2)
    for paths in output["figures"].values():
        with Image.open(paths["png"]) as image:
            image.verify()
        assert Path(paths["pdf"]).read_bytes().startswith(b"%PDF")
    index = Path(output["index"]).read_text()
    for paths in output["figures"].values():
        assert f"]({Path(paths['png']).name})" in index and f"]({Path(paths['pdf']).name})" in index
    rerendered = render_native_scale_payload(output["payload"], tmp_path / "rerendered")
    assert Path(rerendered["payload"]).is_file()
    assert (
        json.loads(Path(rerendered["report"]).read_text())["band_errors"] == report["band_errors"]
    )

    _write_payload(child, dataset, targets[0] + 0.01, second)
    _write_manifest((source, child), "a" * 64)
    with pytest.raises(ValueError, match="target_model"):
        render_native_scale(bundle_path, {"SOURCE": source, "Selected": child}, tmp_path / "bad")
    _write_payload(child, dataset, targets[0], second, seeds=(2, 5, 8, 12))
    _write_manifest((source, child), "a" * 64)
    with pytest.raises(ValueError, match="draw_seeds"):
        render_native_scale(bundle_path, {"SOURCE": source, "Selected": child}, tmp_path / "bad")
