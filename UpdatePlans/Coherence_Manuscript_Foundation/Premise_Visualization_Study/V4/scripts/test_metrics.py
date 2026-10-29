"""Core scientific invariants of the V4 descriptive metrics."""
import numpy as np
import metrics


def test_state_identity_and_independent_permutation():
    rng = np.random.default_rng(23)
    gt = rng.normal(size=(30, 40, 5))
    gt[..., 1] = gt[..., 0] + .1 * gt[..., 1]
    for bins in metrics.BINS:
        s = metrics.state_stats(gt, gt, 0, 1, bins)
        assert np.allclose(s[4:], 0)
        a, b, coverage, _ = metrics.conditional(s, s)
        assert a == b == 0 and coverage > 0
    prediction = gt.copy()
    prediction[..., 1] = rng.permutation(gt[..., 1].ravel()).reshape(gt.shape[:2])
    s = metrics.state_stats(prediction, gt, 0, 1, 12)
    assert np.allclose(s[5:7], 0)
    assert s[4] > .05 and s[7] > .05
    payload = metrics.example_payload(prediction, gt, 0, 1)
    assert np.isclose(.5*np.abs(payload['model_joint']-payload['gt_joint']).sum(), s[4])


def test_nonuniform_area_and_component_identity():
    x, y = np.linspace(0, 2, 40)**2, np.linspace(0, 1, 30)**2
    xx, yy = np.meshgrid(x, y)
    weights = metrics.area_weights(np.stack((xx, yy), axis=-1))
    assert np.isclose(weights.sum(), 1)
    gt = np.random.default_rng(7).normal(size=(30, 40, 5))
    assert np.all(metrics.shape_stats(gt, gt, 0, 1, weights)[:, :4] == 0)
    mask = np.zeros((30, 40), dtype=bool)
    mask[:5, :5] = True
    mask[15:25, 20:30] = True
    assert metrics.components(mask, weights, 0) == 2
    assert metrics.jaccard(mask, mask, weights) == 0


def test_ch4_floor_ties_do_not_collapse_threshold():
    gt = np.zeros((20, 20, 5))
    gt[..., 0] = np.tile(np.linspace(0, 1, 20), (20, 1))
    gt[:, :8, 0] = 0  # 40% of CH4 values share the floor.
    gt[..., 1] = np.tile(np.linspace(1, 0, 20), (20, 1))
    weights = np.full((20, 20), 1 / 400)
    details = metrics.shape_stats(gt, gt, 0, 1, weights)
    assert details[3, 8] == .5
    assert details[3, 6] > 0
