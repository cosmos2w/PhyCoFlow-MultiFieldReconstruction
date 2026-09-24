"""Portable configuration composition without local datasets or checkpoints."""

from pathlib import Path

import pytest

from phycoflow_reconstruction.config import load_config, validate_config

REPOSITORY = Path(__file__).resolve().parents[1]


def test_case_composes_shared_method_and_requires_explicit_source(tmp_path):
    path = REPOSITORY / "cases/active_emulsion/configs/posttrain/topology.yaml"
    config = load_config(path)
    assert config["source_run"] is None
    with pytest.raises(ValueError, match="non-empty source_run"):
        validate_config(config)
    config["source_run"] = str(tmp_path / "source")
    validate_config(config)
    topology = config["coherence"]["families"]["topology"]
    assert topology["strategy"] == "cubical_persistence"
    assert topology["persistence"]["distance"] == "sliced_wasserstein"
    assert config["optimization"]["gradient_balance"] == "topology_regularized"
    assert config["optimization"]["training_subset"]["strata_keys"] == ["regime", "m"]


@pytest.mark.parametrize(
    "mapping",
    [
        {"strata_keys": "category"},
        {"strata_keys": ["a", "a"]},
        {"trajectory_key": ""},
        {"split_key": 1},
    ],
)
def test_invalid_subset_column_mapping_is_rejected(tmp_path, mapping):
    path = REPOSITORY / "cases/active_emulsion/configs/posttrain/topology.yaml"
    config = load_config(path)
    config["source_run"] = str(tmp_path / "source")
    config["optimization"]["training_subset"].update(mapping)
    with pytest.raises(ValueError, match="training_subset"):
        validate_config(config)
