"""Portable spectral config composition and strict project-level contracts."""

import copy
from pathlib import Path

import pytest
import yaml
from helpers.coherence import _post_config

from phycoflow_reconstruction.config.validate import validate_config


def _config():
    root = Path(__file__).resolve().parents[1]
    fragment = yaml.safe_load((root / "configs/coherence/multiscale_spectrum.yaml").read_text())
    config = _post_config(Path("fixture.h5"), Path("source"))
    config["coherence"]["families"] = fragment["coherence"]["families"]
    config["coherence"]["compute_budget"].update(fragment["coherence"]["compute_budget"])
    return config


def test_portable_fragment_composes_with_the_case_contract():
    validate_config(_config())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda c: c["coherence"]["compute_budget"].update(query_policy="random_per_sample"),
            "fixed_shared",
        ),
        (lambda c: c["evaluation"].update(query_points=4), "evaluation.query_points"),
        (
            lambda c: c["coherence"]["families"]["multiscale_spectrum"]["objective"].update(
                almost_fair_alpha=0.95
            ),
            "almost_fair_alpha",
        ),
        (lambda c: c["optimization"].update(lpc_ordered={}), "lpc_ordered"),
    ],
)
def test_project_validator_rejects_incompatible_spectral_contracts(mutation, message):
    config = copy.deepcopy(_config())
    mutation(config)
    with pytest.raises(ValueError, match=message):
        validate_config(config)
