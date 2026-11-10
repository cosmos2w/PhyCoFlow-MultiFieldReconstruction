import pytest
import torch
from phycoflow_reconstruction.evaluation.round7_reductions import identity_hash, validate_cache, j2_field_reductions


def test_j2_fieldwise_mask_and_signed_proxy():
    x1 = torch.tensor([[[1., 2.], [999., 999.]]])
    x2 = torch.tensor([[[-1., 0.], [999., 999.]]])
    z = torch.zeros_like(x1)
    r = j2_field_reductions(x1, x2, z, torch.tensor([[True, False]]))
    assert r['primary_mse'] == [1, 4]
    assert r['variance_proxy'] == [2, 2]
    assert r['bias_plus_reference_variance_proxy'] == [-1, 0]
    with pytest.raises(ValueError):
        j2_field_reductions(x1, x2, z, torch.zeros(1, 2, dtype=torch.bool))


def test_identity_requires_exact_observer_and_model():
    identity = {k: 'a' * 64 for k in ('checkpoint_sha256', 'manifest_sha256', 'support_sha256', 'observer_sha256')}
    identity.update(draw_seed=47, query_seed=47, observer_version='fair_v1')
    cache = {'identity_sha256': identity_hash(identity)}
    validate_cache(cache, identity)
    with pytest.raises(ValueError):
        validate_cache(cache, {**identity, 'observer_version': 'paired_v1'})


def test_signed_band_weighted_attraction_and_separation():
    from phycoflow_reconstruction.contracts import FamilyResult, TermResult
    from phycoflow_reconstruction.evaluation.loss_aligned import _collect_family_result
    value = torch.tensor(2.)
    term = TermResult(value[None], value, diagnostics={'band_weights': [.25, .75],
        'attraction_per_band': [2., 4.], 'repulsion_per_band': [1., 2.]})
    result = FamilyResult({'multiscale_spectrum.signed': term}, value[None], value, {})
    losses, details, leaves = {}, {}, {}
    _collect_family_result('multiscale_spectrum', result, losses, details, leaves)
    assert details['multiscale_spectrum.signed']['weighted_attraction'] == 3.5
    assert details['multiscale_spectrum.signed']['weighted_separation'] == 1.75
