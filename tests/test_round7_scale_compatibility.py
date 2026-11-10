"""Exact prior Scale identity bridge and telemetry-off scoring regression."""
import copy

import pytest
import torch
from test_multiscale_mean_augmented import _line_family, _new_config, _score

from phycoflow_reconstruction.coherence.families.multiscale_spectrum import MultiscaleSpectrumFamily
from phycoflow_reconstruction.coherence.families.multiscale_spectrum.round7_source_compatibility import PRIOR
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer


def test_prior_identity_replays_score_and_rejects_unrelated_changes():
    train = torch.randn(8, 16, 1, generator=torch.Generator().manual_seed(93), dtype=torch.float64)
    config = _new_config()
    family, coords = _line_family(config, train)
    artifact = copy.deepcopy(family.state_artifact())
    artifact['implementation_identity']['sources'].update(PRIOR)

    def fresh():
        return MultiscaleSpectrumFamily(config, DataSpec(('u',), ('1',), 1, (16,)), FieldNormalizer.identity(1))

    restored = fresh()
    restored.load_state_artifact(artifact)
    result = _score(restored, coords, train[:2], train[2:4], train[4:6])
    # Generated from exact prior HEAD sources, seed 93, with telemetry absent.
    assert float(result.scalar_loss) == pytest.approx(0.31523658171672275, abs=1e-12)
    assert float(result.component_results['multiscale_spectrum.signed_band_energy_score.conditional'].scalar_loss) == pytest.approx(0.21542090814133377, abs=1e-12)
    assert float(result.component_results['multiscale_spectrum.joint_spectral_mean_energy_score.conditional'].scalar_loss) == pytest.approx(0.3992626943015558, abs=1e-12)
    for category, name in [('sources', 'family.py'), ('sources', 'objectives.py'),
                           ('sources', 'losses.py'), ('backends', 'scipy')]:
        changed = copy.deepcopy(artifact)
        changed['implementation_identity'][category][name] = 'different'
        with pytest.raises(ValueError, match='implementation/backend identity mismatch'):
            fresh().load_state_artifact(changed)
