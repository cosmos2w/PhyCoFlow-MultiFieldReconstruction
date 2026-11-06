"""Native streaming, matching, and weighted field-risk contracts."""

import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from phycoflow_reconstruction.contracts import DataSpec, FieldSample
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.evaluation.native_coherence import (
    _physical_weights,
    infer_native_panel,
    native_field_risk,
    require_matching_native_panels,
)


class NativeModel(torch.nn.Module):
    def __init__(self, shift=0.):
        super().__init__()
        self.shift = shift
        self.saw_full_target_free = []

    def differentiable_reconstruct(self, batch, *, steps, generator):
        self.saw_full_target_free.append(batch.target_fields is None and batch.query_coords.shape[1] == 6)
        signal = torch.randn((1, 6, 2), generator=generator, device=batch.query_coords.device)
        return signal + batch.obs_values.mean() + self.shift


class Dataset:
    split_name = "validation"
    field_names = ("a", "b")
    data_spec = DataSpec(field_names, ("m", "K"), 2, (2, 3))
    normalizer = FieldNormalizer(torch.tensor([1., 3.]), torch.tensor([2., 4.]))

    def __init__(self, path):
        self.path = path

    def __len__(self):
        return 2

    def __getitem__(self, index):
        y, x = torch.meshgrid(torch.tensor([0., 4.]), torch.tensor([0., 1., 3.]), indexing="ij")
        raw = torch.stack((x, y), -1).reshape(-1, 2)
        mask = torch.tensor([True, True, False, True, True, True])
        values = (torch.arange(12, dtype=torch.float32).reshape(6, 2) / 10) + index
        return FieldSample(
            values, raw / 4, raw, torch.tensor(float(index)), "experiment", index,
            torch.empty(0), self.field_names, (2, 3), valid_points=mask,
        )


def _runtime(tmp_path, model=None):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test checkpoint")
    config = {"observations": {"protocol": "structured_stride", "fields": {"a": {"count": 2}},
                               "seed": 11}}
    return SimpleNamespace(model=model or NativeModel(), device=torch.device("cpu"),
                           config=config, checkpoint_path=checkpoint)


def _endpoint():
    return {"steps": 2, "solver": "euler", "observation_config": {"mode": "none"},
            "weight_selection": "live"}


def test_streamed_native_members_match_across_models_and_seed_prefix(tmp_path):
    dataset = Dataset(tmp_path / "dataset.dat")
    source = _runtime(tmp_path)
    first = infer_native_panel(source, dataset, [0, 1], tmp_path / "source", _endpoint(),
                               panel_seed_base=29, draws=4)
    second_runtime = _runtime(tmp_path, NativeModel(.1))
    second = infer_native_panel(second_runtime, dataset, [0, 1], tmp_path / "child", _endpoint(),
                                panel_seed_base=29, draws=4)
    require_matching_native_panels(first, second)
    assert all(source.model.saw_full_target_free)
    assert len(source.model.saw_full_target_free) == 8
    with np.load(tmp_path / "source/condition_00001.npz", allow_pickle=False) as left, \
         np.load(tmp_path / "child/condition_00001.npz", allow_pickle=False) as right:
        assert left["draws_model"].shape == (4, 6, 2)
        np.testing.assert_allclose(right["draws_model"] - left["draws_model"], .1, atol=2e-7)
        assert left["physical_mask"].tolist() == [True, True, False, True, True, True]
        expected = native_field_risk(left["draws_physical"], left["target_physical"],
                                     left["physical_weights"], left["physical_mask"])
        with np.load(tmp_path / "source/field_risk.npz", allow_pickle=False) as risk:
            np.testing.assert_allclose(risk["member_rmse_physical"][1], expected["member_rmse_physical"])
        saved_prefix = left["draws_model"][:2].copy()
    infer_native_panel(_runtime(tmp_path), dataset, [1], tmp_path / "two", _endpoint(),
                       panel_seed_base=29, draws=2)
    with np.load(tmp_path / "two/condition_00000.npz", allow_pickle=False) as prefix:
        np.testing.assert_array_equal(prefix["draws_model"], saved_prefix)
    no_payload = infer_native_panel(_runtime(tmp_path), dataset, [1], tmp_path / "risk_only", _endpoint(),
                                    panel_seed_base=29, draws=2, retain_payloads=False)
    assert json.loads(no_payload.read_text())["conditions"][0]["payload"] is None
    assert not list((tmp_path / "risk_only").glob("condition_*.npz"))


def test_physical_area_risk_and_zero_reference():
    target = np.array([[1., 0.], [2., 0.], [10., 0.]])
    draws = np.stack([target + [1., 2.], target + [-1., 0.]])
    risk = native_field_risk(draws, target, np.array([1., 3., 2.]), np.array([True, True, False]))
    np.testing.assert_allclose(risk["member_mse_physical"], [[1., 4.], [1., 0.]])
    np.testing.assert_allclose(risk["ensemble_mean_mse_physical"], [0., 1.])
    np.testing.assert_allclose(risk["member_relative_l2_physical"][:, 0], np.sqrt(1 / 3.25))
    assert np.isnan(risk["member_relative_l2_physical"][:, 1]).all()


def test_comparison_rejects_contract_changes_and_existing_output(tmp_path):
    dataset = Dataset(tmp_path / "dataset.dat")
    manifest = infer_native_panel(_runtime(tmp_path), dataset, [0], tmp_path / "run", _endpoint(),
                                  panel_seed_base=1, draws=2)
    original = json.loads(manifest.read_text())
    for key in ("contract_sha256", "physical_measure", "sample_id"):
        modified = deepcopy(original)
        modified["conditions"][0][key] = "changed"
        with pytest.raises(ValueError, match="condition contract differs"):
            require_matching_native_panels(original, modified)
    modified = deepcopy(original)
    modified["sampler"]["solver"] = "heun"
    with pytest.raises(ValueError, match="sampler"):
        require_matching_native_panels(original, modified)
    with pytest.raises(FileExistsError):
        infer_native_panel(_runtime(tmp_path), dataset, [0], tmp_path / "run", _endpoint(),
                           panel_seed_base=1, draws=2)


def test_irregular_unweighted_native_support_requires_provider(tmp_path):
    sample = Dataset(tmp_path / "dataset.dat")[0]
    sample.coordinates_raw[2, 0] += .2
    with pytest.raises(ValueError, match="physical weights are required"):
        _physical_weights(sample)


def test_explicit_case_measure_and_native_fallback_share_comparison_identity(tmp_path):
    from phycoflow_reconstruction.data.physical_measure import (
        attach_sample_measure,
        configure_physical_measure,
    )

    class CompleteDataset(Dataset):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            sample.valid_points = None
            return attach_sample_measure(self, sample)

    source = CompleteDataset(tmp_path / "dataset.dat")
    child = configure_physical_measure(CompleteDataset(source.path), "native_rectilinear_area")
    fallback_weights, fallback_mask, fallback_label = _physical_weights(source[0])
    explicit_weights, explicit_mask, explicit_label = _physical_weights(child[0])
    np.testing.assert_array_equal(fallback_weights, explicit_weights)
    np.testing.assert_array_equal(fallback_mask, explicit_mask)
    assert fallback_label == explicit_label == "native_rectilinear_trapezoidal_area"
    left = infer_native_panel(_runtime(tmp_path), source, [0, 1], tmp_path / "fallback",
                              _endpoint(), panel_seed_base=29, draws=4)
    right = infer_native_panel(_runtime(tmp_path, NativeModel(.1)), child, [0, 1],
                               tmp_path / "explicit", _endpoint(), panel_seed_base=29, draws=4)
    require_matching_native_panels(left, right)
