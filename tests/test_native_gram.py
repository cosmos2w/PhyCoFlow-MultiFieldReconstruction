"""Native auxiliary Gram scores retain the unchanged v4 yardstick."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from test_native_scoring import TrainData, _config, _payload

from phycoflow_reconstruction.coherence.families.multiscale_gram import (
    ANCHOR_PATH,
    MUTUAL_PATH,
    SELF_PATH,
)
from phycoflow_reconstruction.coherence.families.multiscale_spectrum.losses import band_coherence
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.evaluation.native_gram import (
    fit_native_gram_bundle,
    score_native_gram_payloads,
)
from phycoflow_reconstruction.evaluation.native_scoring import (
    fit_native_metric_bundle,
    load_native_metric_bundle,
    score_native_payloads,
)


def test_native_auxiliary_reuses_anchor_and_matches_direct_candidate_fit(tmp_path):
    dataset = TrainData(tmp_path / "train.dat")
    dataset.normalizer = FieldNormalizer(torch.tensor([10., -2.]), torch.tensor([2., 3.]))
    targets = torch.stack([dataset[i].values for i in (0, 1)])
    generator = torch.Generator().manual_seed(100)
    source = targets[None] + .2 * torch.randn((4, 2, 16, 2), generator=generator)
    anchor_path = tmp_path / "anchor.pt"
    args = {"source_members": source, "targets": targets, "coordinates": dataset.coordinates[None],
            "sample_ids": ["synthetic:0", "synthetic:1"], "sample_indices": [0, 1],
            "source_checkpoint_sha256": "a" * 64}
    anchor = fit_native_metric_bundle(_config(), dataset, output_path=anchor_path, **args)
    original_bytes = anchor_path.read_bytes()
    config = _config()
    config["coherence"]["families"]["multiscale_spectrum"]["gram_extension"] = {
        "definition": "self_mutual_gram_v1", "weight": .06287622600055182,
        "std_floor": .001, "relative_std_floor": .05,
    }
    aux_path = tmp_path / "aux.pt"
    aux = fit_native_gram_bundle(config, dataset, anchor_bundle_path=anchor_path, output_path=aux_path)
    assert anchor_path.read_bytes() == original_bytes and aux["anchor_bundle_sha256"] == anchor["sha256"]
    full_path = tmp_path / "full.pt"
    fit_native_metric_bundle(config, dataset, output_path=full_path, **args)
    _, auxiliary = load_native_metric_bundle(aux_path)
    _, full = load_native_metric_bundle(full_path)
    for name, value in full["multiscale_spectrum"].state_dict().items():
        torch.testing.assert_close(auxiliary["multiscale_spectrum"].state_dict()[name], value, rtol=0, atol=0)
    assert auxiliary["multiscale_spectrum"].calibration_record == full["multiscale_spectrum"].calibration_record

    payload = tmp_path / "payload.npz"
    _payload(payload, dataset, source[:, 0], targets[0])
    original = score_native_payloads(anchor_path, [payload], tmp_path / "anchor.json")
    result = score_native_gram_payloads(aux_path, [payload], tmp_path / "aux.json")
    full_result = score_native_gram_payloads(full_path, [payload], tmp_path / "full.json")
    assert result["means"] == full_result["means"]
    components = result["rows"][0]["components"]["multiscale_spectrum"]
    assert components[ANCHOR_PATH] == pytest.approx(original["means"]["multiscale_spectrum"])
    assert result["means"]["candidate_total"] == pytest.approx(
        components[ANCHOR_PATH] + .06287622600055182 * (components[SELF_PATH] + components[MUTUAL_PATH]) / 2)
    physical = result["rows"][0]["physical"]
    assert physical["gram_identity_max_abs"] < 1e-12
    assert physical["skill_identity_max_abs"] < 1e-12
    assert np.asarray(physical["member_grams"]).shape == (4, 2, 2, 2)
    assert np.asarray(physical["target_power"]).shape == (2, 2)
    assert np.asarray(physical["rho_target"]).shape == (2, 2, 2)
    modeled_gram = auxiliary["multiscale_spectrum"]._statistics(targets[0:1])["gram"][0]
    factors = torch.tensor([[4., 6.], [6., 9.]], dtype=modeled_gram.dtype)
    torch.testing.assert_close(torch.as_tensor(physical["target_gram"], dtype=modeled_gram.dtype), modeled_gram * factors)
    assert physical["raw_statistic_units"] == "physical_units"
    scale_family = auxiliary["multiscale_spectrum"]
    expected_floor = scale_family.power_floor * torch.tensor([4., 9.], dtype=modeled_gram.dtype)
    torch.testing.assert_close(torch.as_tensor(
        physical["rho_regularization"]["band_field_power_floor"], dtype=modeled_gram.dtype), expected_floor)
    expected_rho = band_coherence(modeled_gram[None], scale_family.power_floor)[0, :, 0]
    regularized = torch.as_tensor(physical["rho_regularized_target"], dtype=modeled_gram.dtype)
    torch.testing.assert_close(regularized[:, 0, 1], expected_rho)
    assert result["scoring_seconds"] >= 0
    with pytest.raises(ValueError, match="requires a candidate"):
        score_native_gram_payloads(anchor_path, [payload], tmp_path / "bad.json")
    with np.load(payload) as saved:
        altered = {key: saved[key] for key in saved.files}
    altered["physical_weights"] = altered["physical_weights"].copy()
    altered["physical_weights"][0] += .1
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, **altered)
    with pytest.raises(ValueError, match="weights disagree"):
        score_native_gram_payloads(aux_path, [bad], tmp_path / "bad.json")
    wrong = deepcopy(config)
    wrong["coherence"]["families"]["multiscale_spectrum"]["bands"]["count"] = 3
    with pytest.raises(ValueError, match="differs from the frozen"):
        fit_native_gram_bundle(wrong, dataset, anchor_bundle_path=anchor_path, output_path=tmp_path / "bad.pt")
