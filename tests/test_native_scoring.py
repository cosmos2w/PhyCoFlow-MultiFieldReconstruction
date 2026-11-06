"""Frozen native three-family bundle, exact support and saved-data rescoring."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("gudhi")

from phycoflow_reconstruction.coherence.measures import (
    measure_fingerprint,
    rectilinear_nodal_weights,
)
from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.contracts import DataSpec, FieldSample
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.evaluation.native_scoring import (
    fit_native_metric_bundle,
    score_native_payloads,
)


class TrainData:
    split_name = "train"
    selection = SimpleNamespace(split="train")
    data_spec = DataSpec(("a", "b"), ("unknown", "unknown"), 2, (4, 4))
    field_names = ("a", "b")
    normalizer = FieldNormalizer.identity(2)

    def __init__(self, path):
        self.path = path
        path.write_bytes(b"synthetic dataset")
        y, x = torch.meshgrid(torch.tensor([0., .2, .7, 1.]),
                              torch.tensor([0., .3, .6, 1.]), indexing="ij")
        self.coordinates = torch.stack((x, y), -1).reshape(-1, 2)
        generator = torch.Generator().manual_seed(10)
        self.fields = torch.randn(4, 16, 2, generator=generator)
        self.physical_weights = rectilinear_nodal_weights(self.coordinates.double(), (4, 4))
        self.physical_measure_receipt = measure_fingerprint(
            self.physical_weights, metadata={"provider": "test_native_area"})

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return FieldSample(self.fields[index], self.coordinates, self.coordinates,
                           torch.tensor(float(index)), "synthetic", index, torch.empty(0),
                           self.field_names, (4, 4))


def _config():
    state = load_config("configs/coherence/state_mixed_weighted.yaml")["coherence"]["families"]["global_distribution"]
    scale = {
        "fields": ["a", "b"], "target_use": "paired_supervised", "units": "model_units",
        "geometry": {"operator": "fem_p1", "coordinate_scaling": "none", "max_modes": 12},
        "bands": {"count": 2}, "calibration": {"field_unit_floor": .01, "samples": 4},
        "numerics": {"dtype": "float64"},
        "objective": {"kind": "signed_band_energy_score", "conditional_draws": 2,
                      "almost_fair_alpha": 1., "joint_energy_score_weight": .25,
                      "score_definition_version": "signed_spectral_mean_augmented_v4", "mean_weight": 1.},
    }
    shape = {
        "strategy": "cubical_persistence", "target_use": "paired_supervised", "units": "model_units",
        "fields": ["a", "b"], "geometry": {"grid_shape": [2, 2], "raster_mapping": "physical_cell_overlap_v1"},
        "filtration": {"smoothing_sigma": 0., "dimensions": [0, 1]},
        "persistence": {"definition": "finite_primary_v1", "projections": 8,
                        "calibration": {"finite_floor": .001, "essential_floor": .002}},
        "components": {"self": {"enabled": True},
                       "mutual": {"enabled": True, "groups": [["a", "b"]], "lines": 3}},
    }
    return {"coherence": {"families": {"global_distribution": state,
                                         "multiscale_spectrum": scale, "topology": shape}}}


def _payload(path, dataset, draws, target):
    weights = dataset.physical_weights.numpy()
    np.savez_compressed(path, draws_model=draws.numpy(), target_model=target.numpy(),
                        model_coordinates=dataset.coordinates.numpy(), raw_coordinates=dataset.coordinates.numpy(),
                        native_ids=np.arange(16), physical_mask=np.ones(16, dtype=bool),
                        physical_weights=weights / weights.sum(), field_names=np.asarray(dataset.field_names),
                        field_units=np.asarray(dataset.data_spec.field_units),
                        normalizer_sha256=np.asarray(dataset.normalizer.digest()))


def test_freeze_and_saved_native_rescore_preserve_bank_and_reject_changed_support(tmp_path):
    dataset = TrainData(tmp_path / "train.dat")
    targets = torch.stack([dataset[i].values for i in (0, 1)])
    generator = torch.Generator().manual_seed(100)
    source = targets[None] + .2 * torch.randn((2, 2, 16, 2), generator=generator)
    bundle_path = tmp_path / "bundle.pt"
    bundle = fit_native_metric_bundle(
        _config(), dataset, source_members=source, targets=targets,
        coordinates=dataset.coordinates[None], sample_ids=["synthetic:0", "synthetic:1"],
        sample_indices=[0, 1], source_checkpoint_sha256="a" * 64, output_path=bundle_path,
    )
    assert bundle["states"]["topology"]["calibration"]["point_normalization"] == 16
    assert len(bundle["calibration_rows"][0]) == 4
    payload = tmp_path / "member.npz"
    _payload(payload, dataset, source[:, 0], targets[0])
    result = score_native_payloads(bundle_path, [payload], tmp_path / "scores.json")
    for name, value in bundle["calibration_rows"][0][0].items():
        assert result["rows"][0]["values"][name] == pytest.approx(value)
    details = result["rows"][0]["details"]
    assert len(details["topology"]["per_result_family"]) == 2
    assert details["multiscale_spectrum"]["aggregation"] == "conditional_all_members"
    scale_components = details["multiscale_spectrum"]["per_result_components"][0]
    assert any("attraction_per_band" in value["diagnostics"] for value in scale_components.values())
    state_components = details["global_distribution"]["per_result_components"][0]
    assert any("per_field_w2" in value["diagnostics"] for value in state_components.values())
    with np.load(payload) as original:
        changed = {k: original[k] for k in original.files}
    for key in ("physical_weights", "raw_coordinates", "native_ids", "physical_mask", "normalizer_sha256"):
        altered = deepcopy(changed)
        if key == "normalizer_sha256":
            altered[key] = np.asarray("0" * 64)
        elif key == "physical_mask":
            altered[key][0] = False
        elif key == "native_ids":
            altered[key] = altered[key][::-1]
        else:
            altered[key].flat[0] += .1
        bad = tmp_path / f"bad_{key}.npz"
        np.savez_compressed(bad, **altered)
        with pytest.raises(ValueError, match="disagree"):
            score_native_payloads(bundle_path, [bad], tmp_path / "bad.json")


def test_freeze_rejects_wrong_training_ids_or_targets(tmp_path):
    dataset = TrainData(tmp_path / "train.dat")
    targets = torch.stack([dataset[i].values for i in (0, 1)])
    source = targets[None].expand(2, -1, -1, -1).clone()
    with pytest.raises(ValueError, match="TRAIN ID, target"):
        fit_native_metric_bundle(
            _config(), dataset, source_members=source, targets=targets,
            coordinates=dataset.coordinates[None], sample_ids=["synthetic:1", "synthetic:0"],
            sample_indices=[0, 1], source_checkpoint_sha256="a" * 64, output_path=tmp_path / "bad.pt",
        )
