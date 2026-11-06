"""A native event receipt cannot silently apply model thresholds to decoded fields."""

import pytest

from phycoflow_reconstruction.evaluation.native_rendering import _validate_event_config


def test_native_event_thresholds_require_explicit_decoded_field_space():
    bundle = {
        "source_checkpoint_sha256": "source",
        "sha256": "bundle",
        "normalizer_digest": "normalizer",
        "data_spec": {"field_names": ["a", "b"]},
    }
    config = {
        "source_checkpoint_sha256": "source",
        "metric_bundle_sha256": "bundle",
        "fit_split": "train",
        "thresholds": [0.5, 1.5],
        "units": "model units",
    }
    for field_space in (None, "model"):
        with pytest.raises(ValueError, match="field_space: decoded_dataset"):
            _validate_event_config(dict(config, field_space=field_space), bundle)
    decoded = dict(config, field_space="decoded_dataset", units="unknown dataset units")
    _validate_event_config(decoded, bundle)
    with pytest.raises(ValueError, match="normalizer"):
        _validate_event_config(dict(decoded, normalizer_digest="other"), bundle)
    with pytest.raises(ValueError, match="field order"):
        _validate_event_config(dict(decoded, field_order=["b", "a"]), bundle)
