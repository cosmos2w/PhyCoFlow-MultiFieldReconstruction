"""The experimental recipe changes only the declared Scale augmentation."""

from pathlib import Path

import pytest

from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.config.validate import validate_config

ROOT = Path(__file__).resolve().parents[1]


def test_gram_case_inherits_exact_current_anchor_and_requires_shared_reference():
    folder = ROOT / "cases/turbulent_combustion/configs/posttrain"
    current = load_config(folder / "coherence_current.yaml")
    candidate = load_config(folder / "coherence_gram.yaml")
    anchor = candidate["coherence"]["families"]["multiscale_spectrum"]
    extension = anchor.pop("gram_extension")
    assert extension == {"definition": "self_mutual_gram_v1", "weight": 0.06287622600055182,
                         "std_floor": 0.001, "relative_std_floor": 0.05}
    candidate["output"] = current["output"]
    assert candidate == current
    anchor["gram_extension"] = extension
    with pytest.raises(ValueError, match="reference_artifact"):
        validate_config(candidate)
