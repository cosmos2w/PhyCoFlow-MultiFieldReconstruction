import pytest
import torch
from helpers.coherence import _base_config, _write_fixture

from phycoflow_reconstruction.data.factory import open_field_dataset
from phycoflow_reconstruction.data.sensor_protocols import SensorProtocol
from phycoflow_reconstruction.training.post_training import _build_evaluation_batch


def test_explicit_panel_and_fixed_sensors_preserve_training_protocol(tmp_path):
    path = tmp_path / "fixture.h5"
    _write_fixture(path)
    dataset = open_field_dataset(_base_config(path)["dataset"], split="train")
    protocol = SensorProtocol(field_count_ranges={"u": (3, 8)}, seed=3)
    config = {"evaluation": {"max_samples": 1, "sample_indices": [1],
                             "sensor_counts": {"u": 4}, "sensor_seed": 91}}
    first = _build_evaluation_batch(dataset, config, protocol, torch.device("cpu"))
    second = _build_evaluation_batch(dataset, config, protocol, torch.device("cpu"))
    assert first.sample_ids == ("train:1",)
    assert int(first.obs_valid_mask.sum()) == 4
    assert torch.equal(first.obs_indices, second.obs_indices)
    assert first.obs_indices[0].tolist() == torch.randperm(16, generator=torch.Generator().manual_seed(92))[:4].tolist()
    assert protocol.field_count_ranges == {"u": (3, 8)}
    config["evaluation"]["sample_indices"] = [2]
    with pytest.raises(ValueError, match="in-range"):
        _build_evaluation_batch(dataset, config, protocol, torch.device("cpu"))
    dataset.close()
