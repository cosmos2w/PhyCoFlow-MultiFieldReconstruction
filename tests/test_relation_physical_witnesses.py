"""Small physical witnesses for signed relation and native quadrature semantics."""
import numpy as np
import pytest
import torch

from phycoflow_reconstruction.coherence.families.multiscale_spectrum.losses import band_coherence
from phycoflow_reconstruction.coherence.families.multiscale_spectrum.statistics import band_gram
from phycoflow_reconstruction.coherence.families.native_quadrature import NativeQuadrature


def test_equal_self_power_opposite_sign_and_spatial_phase_blind_spot():
    filters = torch.ones(1, 2, dtype=torch.float64)
    # The two columns are fields and the rows are signed spatial modes.
    z = torch.tensor([[[1., 1.], [0., 0.]]], dtype=torch.float64)
    sign_flip = torch.tensor([[[1., -1.], [0., 0.]]], dtype=torch.float64)
    phase_shift = torch.tensor([[[0., 0.], [1., 1.]]], dtype=torch.float64)
    g, flipped, shifted = (band_gram(x, filters) for x in (z, sign_flip, phase_shift))
    assert torch.equal(torch.diagonal(g, dim1=-2, dim2=-1),
                       torch.diagonal(flipped, dim1=-2, dim2=-1))
    assert g[0, 0, 0, 1] == -flipped[0, 0, 0, 1]
    assert torch.equal(g, shifted)
    assert torch.linalg.vector_norm(z - phase_shift) > 0  # signed coefficients retain phase
    floor = torch.full((1, 2), 0.01, dtype=torch.float64)
    assert band_coherence(g, floor).item() == pytest.approx(-band_coherence(flipped, floor).item())
    assert torch.equal(band_coherence(g, floor), band_coherence(shifted, floor))


def test_orthogonal_and_correlated_fields_and_floor_damping_under_gain():
    filters = torch.ones(1, 2, dtype=torch.float64)
    orthogonal = torch.tensor([[[2., 0.], [0., 2.]]], dtype=torch.float64)
    correlated = torch.tensor([[[2., 2.], [0., 0.]]], dtype=torch.float64)
    floor = torch.ones((1, 2), dtype=torch.float64)
    orth_rho = band_coherence(band_gram(orthogonal, filters), floor).item()
    corr_rho = band_coherence(band_gram(correlated, filters), floor).item()
    assert orth_rho == pytest.approx(0.0, abs=1e-15)
    assert corr_rho == pytest.approx(0.8)
    high = band_coherence(band_gram(correlated * 100, filters), floor).item()
    higher = band_coherence(band_gram(correlated * 200, filters), floor).item()
    assert abs(high - higher) < 2e-5
    weak = band_coherence(band_gram(correlated * 0.01, filters), floor).item()
    assert 0 < weak < corr_rho < high < 1
    # Under a units change, power floors scale quadratically with fields.
    converted = band_coherence(band_gram(correlated * 7, filters), floor * 49).item()
    assert converted == pytest.approx(corr_rho)


def test_disconnected_nonuniform_geometry_preserves_piece_constants_and_means():
    # Two pieces, nonuniform physical masses, and a native coordinate mask with a gap.
    x = np.array([[0.], [1.], [2.], [5.], [7.]])
    mass = np.array([.05, .15, .20, .25, .35])
    pieces = np.array([0, 0, 0, 1, 1])
    # One mean-free, native mass-normalized mode on each connected piece.
    phi = np.zeros((5, 2))
    phi[:3, 0] = np.array([-3., -1., 1.5])
    phi[3:, 1] = np.array([-1., .25 / .35])
    for k in range(2):
        phi[:, k] /= np.sqrt(np.sum(mass * phi[:, k]**2))
    query = np.array([0, 2, 3, 4])
    quad = NativeQuadrature(coordinates=x, model_coordinates=x, node_ids=np.arange(5),
                            mass=mass, pieces=pieces, basis=phi, filters=np.ones((1, 2)),
                            query_ids=query, metric_identity='m', calibration_identity='c')
    fields = torch.tensor([[[2., -1.], [2., -1.], [7., 3.], [7., 3.]]],
                          dtype=torch.float64, requires_grad=True)
    stats = quad.statistics(fields)
    assert torch.allclose(stats['piece_means'], torch.tensor([[[2., -1.], [7., 3.]]], dtype=torch.float64))
    assert torch.allclose(stats['coefficients'], torch.zeros_like(stats['coefficients']), atol=1e-14)
    assert np.allclose(quad.diagnostics()['piece_mass_error'], 0, atol=1e-15)
    stats['piece_means'].sum().backward()
    assert torch.isfinite(fields.grad).all() and (fields.grad != 0).all()
    with pytest.raises(ValueError, match='every native piece'):
        NativeQuadrature(coordinates=x, model_coordinates=x, node_ids=np.arange(5), mass=mass,
                         pieces=pieces, basis=phi, filters=np.ones((1, 2)),
                         query_ids=np.array([0, 2]), metric_identity='m', calibration_identity='c')
