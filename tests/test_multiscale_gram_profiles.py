"""The experimental recipe changes only the declared Scale augmentation."""

import hashlib
from pathlib import Path

import pytest
import yaml

from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.config.validate import validate_config

ROOT = Path(__file__).resolve().parents[1]


def test_gram_case_inherits_exact_current_anchor_and_requires_shared_reference(tmp_path):
    folder = ROOT / "cases/turbulent_combustion/configs/posttrain"
    current = load_config(folder / "coherence_current.yaml")
    fragment = folder / "fragments/coherence_gram.yaml"
    assert "stage" not in load_config(fragment)
    composed = tmp_path / "gram_composition.yaml"
    composed.write_text(yaml.safe_dump({"defaults": [str(folder / "coherence_current.yaml"), str(fragment)]}))
    candidate = load_config(composed)
    anchor = candidate["coherence"]["families"]["multiscale_spectrum"]
    extension = anchor.pop("gram_extension")
    assert extension == {"definition": "self_mutual_gram_v1", "weight": 0.06287622600055182,
                         "std_floor": 0.001, "relative_std_floor": 0.05}
    candidate["output"] = current["output"]
    assert candidate == current
    anchor["gram_extension"] = extension
    with pytest.raises(ValueError, match="reference_artifact"):
        validate_config(candidate)

    # Schema acceptance does not claim training-time artifact/binding validation.
    reference = tmp_path / "shared_source_reference.json"
    reference.write_text('{"qualification": "schema_only"}\n')
    candidate["optimization"]["multitask"]["fidelity_budget"].update(
        reference_artifact=str(reference.resolve()),
        reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
    )
    validate_config(candidate)
