"""Spectral operators, physical scales, masked geometry and filter-bank contracts."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from helpers.spectral import (
    _config,
    _family,
    _fitted,
    _geometry,
    _grid,
    _smooth_states,
    _u_shaped_grid,
)

from phycoflow_reconstruction.coherence.families.multiscale_spectrum.operators import (
    fem_p1_operator,
    region_measures,
    spectral_basis,
    tight_filter_bank,
)


def test_fem_neumann_eigenvalues_converge_to_the_rectangle_spectrum():
    axis_x, axis_y = np.linspace(0, 2.0, 41), np.linspace(0, 1.0, 21)
    points = np.stack(np.meshgrid(axis_x, axis_y, indexing="ij"), axis=-1).reshape(-1, 2)
    operator = fem_p1_operator(
        points, periodic=(False, False), periods=None, max_edge_factor=None, spacing_neighbors=6
    )
    basis = spectral_basis(operator, max_modes=12, cutoff_eigenvalue=None)
    exact = sorted(
        (math.pi * i / 2.0) ** 2 + (math.pi * j) ** 2 for i in range(9) for j in range(9)
    )
    np.testing.assert_allclose(basis.eigenvalues[:6], exact[1:7], rtol=0.02)
    assert operator.diagnostics["total_measure"] == pytest.approx(2.0)
    assert operator.diagnostics["euler_characteristic"] == 1
    gram = basis.eigenvectors.T @ (operator.mass[:, None] * basis.eigenvectors)
    np.testing.assert_allclose(gram, np.eye(gram.shape[0]), atol=1e-10)
    np.testing.assert_allclose(operator.mass @ basis.eigenvectors, 0.0, atol=1e-12)


def test_periodic_fem_lattice_is_a_torus_with_fourier_degeneracy():
    axis = np.arange(16) / 16
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    operator = fem_p1_operator(
        points,
        periodic=(True, True),
        periods=np.array([1.0, 1.0]),
        max_edge_factor=None,
        spacing_neighbors=6,
    )
    basis = spectral_basis(operator, max_modes=8, cutoff_eigenvalue=None)
    assert operator.diagnostics["euler_characteristic"] == 0
    assert operator.diagnostics["boundary_edges"] == 0
    np.testing.assert_allclose(operator.mass, 1.0 / 256)
    # Four (+-1, 0)/(0, +-1) Fourier modes share (2 pi)^2 up to the P1 dispersion error.
    np.testing.assert_allclose(basis.eigenvalues[:4], basis.eigenvalues[0], rtol=1e-10)
    assert basis.eigenvalues[0] == pytest.approx((2 * math.pi) ** 2, rel=0.02)


@pytest.mark.parametrize("jitter", [0.0, 0.3])
def test_single_periodic_axis_gives_a_cylinder_with_mixed_spectrum(jitter):
    nx, ny, height = 32, 8, 0.3
    x, y = np.meshgrid(np.arange(nx) / nx, np.linspace(0.0, height, ny), indexing="ij")
    points = np.stack((x, y), axis=-1).reshape(-1, 2)
    if jitter:
        noise = np.random.default_rng(1).uniform(-jitter, jitter, points.shape) / np.array([nx, ny])
        noise[:, 1] *= ~np.isin(points[:, 1], (0.0, height))  # keep the walls straight
        points = points + noise
    operator = fem_p1_operator(
        points,
        periodic=(True, False),
        periods=np.array([1.0, 1.0]),
        max_edge_factor=None,
        spacing_neighbors=6,
    )
    assert operator.diagnostics["euler_characteristic"] == 0
    assert operator.diagnostics["total_measure"] == pytest.approx(height, rel=1e-9)
    if not jitter:
        assert operator.diagnostics["boundary_edges"] == 2 * nx
        basis = spectral_basis(operator, max_modes=4, cutoff_eigenvalue=None)
        # Periodic (2 pi k)^2 in x lies below the first Neumann wall mode (pi / 0.3)^2.
        assert basis.eigenvalues[0] == pytest.approx((2 * math.pi) ** 2, rel=0.02)
        assert basis.eigenvalues[1] == pytest.approx(basis.eigenvalues[0], rel=1e-9)


def test_count_truncation_never_splits_a_degenerate_eigenspace():
    axis = np.arange(12) / 12
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    operator = fem_p1_operator(
        points,
        periodic=(True, True),
        periods=np.array([1.0, 1.0]),
        max_edge_factor=None,
        spacing_neighbors=6,
    )
    assert spectral_basis(operator, max_modes=5, cutoff_eigenvalue=None).eigenvalues.size == 4
    with pytest.raises(ValueError, match="degenerate eigenspace"):
        spectral_basis(operator, max_modes=2, cutoff_eigenvalue=None)


def test_iterative_eigensolver_matches_the_dense_solver():
    operator = fem_p1_operator(
        _grid(14, jitter=0.2).numpy(),
        periodic=(False, False),
        periods=None,
        max_edge_factor=None,
        spacing_neighbors=6,
    )
    dense = spectral_basis(operator, max_modes=200, cutoff_eigenvalue=400.0)
    iterative = spectral_basis(operator, max_modes=200, cutoff_eigenvalue=400.0, dense_limit=10)
    assert iterative.diagnostics["eigen_solver"] == "shift_invert_lanczos"
    np.testing.assert_allclose(iterative.eigenvalues, dense.eigenvalues, rtol=1e-8)

    def projector(basis):
        return basis.eigenvectors @ (operator.mass[:, None] * basis.eigenvectors).T

    np.testing.assert_allclose(projector(iterative), projector(dense), atol=1e-7)


def test_long_edge_filter_removes_bridges_across_a_notch_and_rejects_disconnection():
    _index, points = _u_shaped_grid()
    hull = fem_p1_operator(
        points, periodic=(False, False), periods=None, max_edge_factor=None, spacing_neighbors=6
    )
    filtered = fem_p1_operator(
        points, periodic=(False, False), periods=None, max_edge_factor=1.8, spacing_neighbors=4
    )
    assert hull.diagnostics["total_measure"] == pytest.approx(1.0)
    assert filtered.diagnostics["removed_long_edge_triangles"] > 0
    # Only a corner-cutting triangle or two at the reentrant corners may remain.
    assert filtered.diagnostics["total_measure"] == pytest.approx(1.0 - 0.4 * 0.7, rel=0.01)
    full = np.stack(np.meshgrid(np.arange(21), np.arange(21), indexing="ij"), axis=-1).reshape(
        -1, 2
    )
    two_blocks = full[(full[:, 0] < 6) | (full[:, 0] > 14)] / 20.0
    with pytest.raises(ValueError, match="disconnected"):
        fem_p1_operator(
            two_blocks,
            periodic=(False, False),
            periods=None,
            max_edge_factor=1.8,
            spacing_neighbors=4,
        )


def test_disconnected_pieces_each_keep_their_own_null_mode():
    full = np.stack(np.meshgrid(np.arange(21), np.arange(21), indexing="ij"), axis=-1).reshape(
        -1, 2
    )
    left = full[full[:, 0] < 6] / 20.0
    right = full[full[:, 0] > 14] / 20.0
    pieces = np.concatenate([left, right, [[0.5, 0.5]]])  # plus one isolated point
    kwargs = {
        "periodic": (False, False),
        "periods": None,
        "max_edge_factor": 1.8,
        "spacing_neighbors": 4,
    }
    operator = fem_p1_operator(pieces, connectivity="components", **kwargs)
    assert operator.diagnostics["analysis_components"] == 2
    assert operator.diagnostics["excluded_points"] == 1
    basis = spectral_basis(operator, max_modes=400, cutoff_eigenvalue=2000.0)
    alone = spectral_basis(
        fem_p1_operator(left, **kwargs), max_modes=400, cutoff_eigenvalue=2000.0
    ).eigenvalues
    union = np.sort(np.concatenate([alone, alone]))  # the two blocks are congruent
    np.testing.assert_allclose(basis.eigenvalues, union, rtol=1e-8)
    indicator = np.stack([operator.components == piece for piece in (0, 1)], axis=1)
    np.testing.assert_allclose(
        (operator.mass[:, None] * indicator).T @ basis.eigenvectors, 0, atol=1e-12
    )


def test_structured_indices_mesh_follows_the_masked_grid_exactly():
    index, points = _u_shaped_grid()
    flat = fem_p1_operator(
        points,
        periodic=(False, False),
        periods=None,
        max_edge_factor=None,
        spacing_neighbors=6,
        grid_indices=index,
    )
    # Full cells plus one three-corner half cell at each reentrant notch corner.
    assert flat.diagnostics["total_measure"] == pytest.approx(1.0 - 0.4 * 0.7 + 2 * 0.5 * 0.05**2)
    assert flat.diagnostics["euler_characteristic"] == 1
    assert flat.diagnostics["triangulation"] == "structured_indices"
    shuffled = np.random.default_rng(2).permutation(index.shape[0])
    permuted = fem_p1_operator(
        points[shuffled],
        periodic=(False, False),
        periods=None,
        max_edge_factor=None,
        spacing_neighbors=6,
        grid_indices=index[shuffled],
    )
    np.testing.assert_allclose(permuted.mass, flat.mass[shuffled])


def test_support_filter_drops_land_bridges_and_reports_exclusions():
    _, support = _u_shaped_grid()
    subset = np.sort(np.random.default_rng(4).permutation(support.shape[0])[:150])
    hull = fem_p1_operator(
        support[subset],
        periodic=(False, False),
        periods=None,
        max_edge_factor=None,
        spacing_neighbors=6,
    )
    filtered = fem_p1_operator(
        support[subset],
        periodic=(False, False),
        periods=None,
        max_edge_factor=None,
        spacing_neighbors=6,
        support_points=support,
        support_factor=1.0,
        support_neighbors=8,
        connectivity="components",
    )
    assert hull.diagnostics["total_measure"] > 0.95
    assert filtered.diagnostics["removed_unsupported_triangles"] > 0
    assert abs(filtered.diagnostics["total_measure"] - (1.0 - 0.4 * 0.7)) < 0.1
    assert filtered.diagnostics["analysis_points"] + filtered.diagnostics["excluded_points"] == 150


def test_filter_bank_is_tight_with_a_smooth_unresolved_band():
    wavenumbers = np.array([0.5, 1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 8.0, 9.0, 10.0])
    bank = tight_filter_bank(wavenumbers, np.array([1.0, 3.0, 8.0]), 10.0)
    coverage = np.square(bank).sum(axis=0)
    np.testing.assert_allclose(coverage[wavenumbers <= 8.0], 1.0, atol=1e-14)
    assert 0.0 < coverage[-2] < 1.0 and coverage[-1] == pytest.approx(0.0, abs=1e-30)
    np.testing.assert_allclose(bank[:, 1], bank[:, 2])
    np.testing.assert_allclose(bank[:, 4], bank[:, 5])
    assert bank[0, 0] == 1.0 and bank[1, 4] == 1.0 and bank[2, 7] == 1.0
    with pytest.raises(ValueError, match="below the resolution cutoff"):
        tight_filter_bank(wavenumbers, np.array([1.0, 10.0]), 10.0)


def test_region_windows_form_normalized_overlapping_measures():
    coordinates = _grid(12).numpy()
    mass = np.full(coordinates.shape[0], 1.0 / coordinates.shape[0])
    measures, names, record = region_measures(
        coordinates, mass, grid=(2, 3), periodic=(False, False), periods=None, min_mass_fraction=0.0
    )
    assert len(names) == 6 and record["retained_regions"] == 6
    np.testing.assert_allclose(measures.sum(axis=1), 1.0)
    weights = measures * np.asarray(record["region_mass"])[:, None] / mass[None, :]
    np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-12)


def test_one_axis_domains_reject_two_dimensional_mesh_options():
    coordinates = torch.linspace(0.0, 1.0, 40, dtype=torch.float64)[:, None]
    states = torch.randn(4, 40, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    family = _family(_config(geometry=_geometry(max_edge_factor=1.5)), 40, dimension=1)
    with pytest.raises(ValueError, match="two-dimensional"):
        family.fit_reference_tensors(coordinates, states, sample_ids=list("abcd"))


def test_raw_extent_scaling_restores_the_physical_aspect_ratio():
    coordinates = _grid(9)
    states = _smooth_states(coordinates, 4)
    family = _family(_config(geometry={"operator": "fem_p1"}), 81)
    with pytest.raises(ValueError, match="raw coordinate extent"):
        family.fit_reference_tensors(coordinates, states, sample_ids=list("abcd"))
    family = _family(_config(geometry={"operator": "fem_p1"}), 81)
    family.fit_reference_tensors(
        coordinates,
        states,
        sample_ids=list("abcd"),
        raw_minimum=np.array([0.0, 0.0]),
        raw_maximum=np.array([4.0, 1.0]),
    )
    assert family.geometry_record["active_axis_lengths"] == [4.0, 1.0]
    assert family.geometry_record["operator_diagnostics"]["total_measure"] == pytest.approx(4.0)


def test_spatial_refinement_keeps_resolved_band_powers_stable():
    def band_power(size: int) -> torch.Tensor:
        coordinates = _grid(size)
        x, y = coordinates[:, 0], coordinates[:, 1]
        field = torch.stack(
            (
                torch.cos(math.pi * x)
                + 0.5 * torch.cos(3 * math.pi * x) * torch.cos(2 * math.pi * y),
                torch.sin(math.pi * x) * torch.cos(math.pi * y),
            ),
            dim=-1,
        )
        config = {
            "fields": ["u", "v"],
            "geometry": _geometry(),
            "bands": {"wavelengths": [2.0, 1.0, 0.55], "min_wavelength": 0.3},
            "numerics": {"dtype": "float64"},
            "components": {"log_band_power": {"enabled": True, "weight": 1.0}},
        }
        family = _fitted(config, coordinates, torch.stack((field, 1.1 * field)), fields=("u", "v"))
        return torch.diagonal(family._statistics(field[None])["gram"], dim1=-2, dim2=-1)[0]

    coarse, fine = band_power(17), band_power(33)
    torch.testing.assert_close(coarse, fine, atol=0.01, rtol=0.03)


def test_lonlat_scaling_uses_equirectangular_kilometres():
    coordinates = _grid(8)
    states = _smooth_states(coordinates, 4)
    family = _family(
        _config(geometry={"operator": "fem_p1", "coordinate_scaling": "lonlat_degrees"}), 64
    )
    family.fit_reference_tensors(
        coordinates,
        states,
        sample_ids=list("abcd"),
        raw_minimum=np.array([-82.15, 26.47]),
        raw_maximum=np.array([-81.83, 26.69]),
        raw_mean_latitude=26.58,
    )
    lengths = family.geometry_record["active_axis_lengths"]
    scale = math.pi / 180 * 6371.0088
    assert lengths[0] == pytest.approx(0.32 * scale * math.cos(math.radians(26.58)))
    assert lengths[1] == pytest.approx(0.22 * scale)
    assert family.geometry_record["length_unit"] == "km"
