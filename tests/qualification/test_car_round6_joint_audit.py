import numpy as np
import pytest
from scripts.qualification.car_round6_joint_audit import coarse_profile, fit_gain, weighted_mean


def test_coarse_profile_integrates_all_actions_and_projects_once():
    history = [{'epoch': i + 1, 'epoch_complete': True, 'field_violation': {'field': .2}} for i in range(2)]
    multiplier, smooth = coarse_profile(history, ['field'], [.1], [1.], epochs=2)
    nu = np.exp(-2)
    assert smooth[0] == pytest.approx(.2 - .3 * nu)
    assert multiplier[0] == pytest.approx(.0012 * .4 + .012 * .3 * (1 - nu))


def test_weighted_response_and_clipped_interpolation():
    assert weighted_mean([[0.], [1.]], [64, 32])[0] == pytest.approx(1 / 3)
    result = fit_gain([[.2], [.2]], [[.2], [.2]], [[.19], [.19]], [64, 32], 1, [.1], [1])
    assert result['gamma_star'] == pytest.approx(1 / 6)
    with pytest.raises(ValueError, match='zero or nonfinite'):
        fit_gain([[.2]], [[.2]], [[.2]], [1], 1, [.1], [1])
    with pytest.raises(ValueError, match='positive physical allowance'):
        fit_gain([[.2]], [[.2]], [[.1]], [1], 1, [0], [1])
