"""Synthetic saved-data coverage for native Gram atlas integration."""

import json

import numpy as np
import pytest
import torch
from PIL import Image
from test_native_scoring import TrainData, _config, _payload

from phycoflow_reconstruction.evaluation.native_coherence import _digest_arrays
from phycoflow_reconstruction.evaluation.native_gram import fit_native_gram_bundle
from phycoflow_reconstruction.evaluation.native_gram_visualization import render_native_gram
from phycoflow_reconstruction.evaluation.native_scoring import fit_native_metric_bundle
from phycoflow_reconstruction.training.run_store import file_sha256


def fixture(tmp_path, *, single_field=False):
    dataset = TrainData(tmp_path / "train.dat")
    targets = torch.stack([dataset[i].values for i in (0, 1)])
    source = targets[None] + .2 * torch.randn((4, 2, 16, 2), generator=torch.Generator().manual_seed(17))
    anchor_path = tmp_path / "anchor.pt"
    config = _config()
    if single_field:
        config["coherence"]["families"]["multiscale_spectrum"]["fields"] = ["a"]
    fit_native_metric_bundle(config, dataset, output_path=anchor_path, source_members=source,
        targets=targets, coordinates=dataset.coordinates[None], sample_ids=["synthetic:0", "synthetic:1"],
        sample_indices=[0, 1], source_checkpoint_sha256="a" * 64)
    config["coherence"]["families"]["multiscale_spectrum"]["gram_extension"] = {
        "definition": "self_mutual_gram_v1", "weight": .06287622600055182,
        "std_floor": .001, "relative_std_floor": .05,
    }
    gram = tmp_path / "gram.pt"
    fit_native_gram_bundle(config, dataset, anchor_bundle_path=anchor_path, output_path=gram)
    payloads = {}
    pattern = torch.sin(torch.arange(16) / 3)[:, None] * torch.tensor([.08, -.05])
    for label, prediction in (("SOURCE", source[:, 0]), ("Candidate", source[:, 0] + pattern)):
        directory = tmp_path / label
        directory.mkdir(exist_ok=True)
        path = directory / "payload.npz"
        _payload(path, dataset, prediction, targets[0])
        with np.load(path) as original:
            values = {key: original[key] for key in original.files}
        indices = np.asarray([0, 5], dtype=np.int64)
        values.update(sample_id=np.asarray("synthetic:0"), logical_shape=np.asarray([4, 4]),
                      observation_coords=dataset.coordinates.numpy()[indices],
                      observation_values=targets[0, indices, :1].numpy(),
                      observation_field_ids=np.asarray([0, 0], dtype=np.int64),
                      observation_indices=indices, observation_mask=np.asarray([True, True]),
                      draw_seeds=np.asarray([1, 2, 3, 4], dtype=np.int64))
        np.savez_compressed(path, **values)
        digest_keys = ("native_ids", "model_coordinates", "raw_coordinates", "physical_mask", "physical_weights",
                       "target_model", "observation_coords", "observation_values", "observation_indices",
                       "observation_field_ids", "observation_mask", "draw_seeds")
        manifest = {"schema": "native_coherence_inference_v1", "query_policy": "full_native",
                    "checkpoint_sha256": "a" * 64 if label == "SOURCE" else "b" * 64,
                    "conditions": [{"payload": path.name, "sample_id": "synthetic:0",
                                    "contract_sha256": _digest_arrays(*(values[k] for k in digest_keys))}]}
        (directory / "manifest.json").write_text(json.dumps(manifest))
        payloads[label] = path
    return anchor_path, gram, payloads


def test_matched_saved_rendering_and_contract_rejection(tmp_path):
    _, gram, payloads = fixture(tmp_path)
    output = tmp_path / "render"
    result = render_native_gram(gram, payloads, output, field_pair=["a", "b"], dpi=80)
    receipt = json.loads((output / "native_gram_rendering.json").read_text())
    assert receipt["inference_performed"] is False and receipt["fitting_performed"] is False
    assert receipt["sample_id"] == "synthetic:0"
    assert receipt["inputs"]["SOURCE"]["sha256"] == file_sha256(payloads["SOURCE"])
    source_scores = json.loads((output / "scores_00.json").read_text())
    assert receipt["scores"]["SOURCE"] == source_scores["means"]
    components = source_scores["rows"][0]["physical"]
    assert components["gram_identity_max_abs"] < 1e-12
    assert components["skill_identity_max_abs"] < 1e-12
    assert result["index"] == str(output / "INDEX.md")
    for stem in ("native_gram_physical", "native_gram_coupling_scores"):
        assert (output / f"{stem}.pdf").read_bytes().startswith(b"%PDF")
        with Image.open(output / f"{stem}.png") as image:
            image.verify()
    candidate = payloads["Candidate"]
    with np.load(candidate) as original:
        changed = {key: original[key] for key in original.files}
    changed["draw_seeds"] = np.asarray([4, 3, 2, 1])
    np.savez_compressed(candidate, **changed)
    with pytest.raises(ValueError, match="manifest|matched"):
        render_native_gram(gram, payloads, tmp_path / "rejected", dpi=80)


def test_main_atlas_includes_gram_and_rejects_wrong_yardstick(tmp_path, monkeypatch):
    import phycoflow_reconstruction.evaluation.native_rendering as atlas

    anchor, gram, payloads = fixture(tmp_path)

    def fake_index(*args, **kwargs):
        destination = args[2]
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "INDEX.md").write_text("# Existing atlas fixture\n")
        return {"index": str(destination / "INDEX.md")}

    monkeypatch.setattr(atlas, "render_native_atlas", fake_index)
    monkeypatch.setattr(atlas, "render_native_scale", fake_index)
    monkeypatch.setattr(atlas, "save_finite_primary_payload", lambda *args, **kwargs: kwargs["output_path"])
    monkeypatch.setattr(atlas, "render_finite_primary_payload", lambda *args, **kwargs: [])
    out = tmp_path / "atlas"
    report = atlas.render_native_coherence(anchor, payloads, out, gram_bundle_path=gram,
                                            gram_field_pair=["a", "b"])
    assert "[Native Gram and skill](gram/INDEX.md)" in (out / "INDEX.md").read_text()
    assert report["gram"]["index"] == str(out / "gram/INDEX.md")
    assert json.loads((out / "native_rendering.json").read_text())["gram"] == report["gram"]
    with pytest.raises(ValueError, match="frozen v4 yardstick"):
        atlas.render_native_coherence(anchor, payloads, tmp_path / "wrong", gram_bundle_path=anchor)
    assert not (tmp_path / "wrong").exists()


def test_cli_passes_auxiliary_bundle_and_pair(tmp_path, monkeypatch):
    import sys

    import phycoflow_reconstruction.evaluation.native_rendering as atlas
    from phycoflow_reconstruction.cli import run_case_cli

    observed = {}

    def fake_renderer(*args, **kwargs):
        observed.update(kwargs)

    monkeypatch.setattr(atlas, "render_native_coherence", fake_renderer)
    monkeypatch.setattr(sys, "argv", ["run.py", "render-native-coherence", "--metric-bundle", "anchor.pt",
                                  "--source", "source.npz", "--gram-bundle", "gram.pt", "--gram-pair", "a", "b",
                                  "--output", str(tmp_path)])
    assert run_case_cli("fixture", tmp_path) == 0
    assert observed["gram_bundle_path"].name == "gram.pt"
    assert observed["gram_field_pair"] == ["a", "b"]


def test_one_field_keeps_self_and_marks_mutual_unavailable(tmp_path):
    _, gram, payloads = fixture(tmp_path, single_field=True)
    output = tmp_path / "one_field"
    render_native_gram(gram, payloads, output, dpi=80)
    receipt = json.loads((output / "native_gram_rendering.json").read_text())
    assert receipt["field_pair"] is None
    assert "gram_self_score" in receipt["scores"]["SOURCE"]
    assert "gram_mutual_score" not in receipt["scores"]["SOURCE"]
    with Image.open(output / "native_gram_coupling_scores.png") as image:
        image.verify()


def test_power_axis_includes_tiny_stacked_variation(tmp_path, monkeypatch):
    import phycoflow_reconstruction.evaluation.native_gram_visualization as visualization

    physical = {"target_power": [[1.0]], "mean_pattern_power": [[1_000_000.0]],
                "member_variation_power": [[0.1]], "mean_member_power": [[1_000_000.1]],
                "represented_band_member_mse": [[.2]], "represented_band_ensemble_mean_mse": [[.1]]}
    checked = []

    def check_axes(fig, *_):
        axis = fig.axes[0]
        assert axis.get_ylim()[1] > 1_000_000.1
        assert max(bar.get_y() + bar.get_height() for bar in axis.patches) < axis.get_ylim()[1]
        checked.append(True)

    monkeypatch.setattr(visualization, "_save_figure", check_axes)
    visualization._physical_panel([{"rows": [{"physical": physical}]}], ["SOURCE"],
                                  ["T"], ["b0"], ["K"], tmp_path / "unused", 80)
    assert checked == [True]


def test_signed_cross_gram_displays_negative_variation(tmp_path, monkeypatch):
    import phycoflow_reconstruction.evaluation.native_gram_visualization as visualization

    def matrix(cross):
        return [[[1., cross], [cross, 1.]]]
    physical = {"target_gram": matrix(-.1), "rho_regularized_target": matrix(-.1),
                "mean_pattern_gram": matrix(.5), "member_variation_gram": matrix(-.2),
                "rho_regularized_mean_member_gram": matrix(.3)}
    report = {"rows": [{"physical": physical}], "means": {"anchor_total": .2, "candidate_total": .3,
               "gram_self_score": .4, "gram_mutual_score": .5}}
    checked = []

    def check_axes(fig, *_):
        bars = fig.axes[0].patches
        assert bars[2].get_hatch() and bars[2].get_y() == pytest.approx(.5)
        assert bars[2].get_height() == pytest.approx(-.2)
        assert fig.axes[0].get_ylim()[0] < -.1 and fig.axes[0].get_ylim()[1] > .5
        checked.append(True)

    monkeypatch.setattr(visualization, "_save_figure", check_axes)
    visualization._coupling_panel([report], ["SOURCE"], ["a", "b"], ["b0"], ["a", "b"],
                                   ["a", "b"], tmp_path / "unused", 80)
    assert checked == [True]
