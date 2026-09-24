"""Focused contracts for set-level cubical-persistence evaluation figures."""

import numpy as np
import torch

from phycoflow_reconstruction.coherence.families.topology.exact import (
    exact_betti_curves,
    reference_levels,
)
from phycoflow_reconstruction.data.training_batches import fixed_query_indices
from phycoflow_reconstruction.evaluation.topology_set import (
    TopologySetAccumulator,
    exact_betti_from_filtration_banks,
    render_topology_betti_curves,
    render_topology_distance_distributions,
)


def test_topology_accumulator_replays_training_fixed_shared_selector():
    coordinates = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1).repeat(1, 1, 2)
    values = torch.arange(32, dtype=torch.float32).reshape(1, 16, 2)
    accumulator = TopologySetAccumulator(
        family=None,
        config={},
        dataset_field_names=("u", "v"),
        normalization_method="mean_std",
        source_grid_shape=(4, 4),
        query_point_count=7,
        query_seed=100045,
    )

    prediction, target, selected_coordinates = accumulator._select_queries(
        values, values + 1, coordinates
    )
    expected = fixed_query_indices(16, 7, seed=100045)
    assert torch.equal(accumulator.query_indices, expected)
    assert torch.equal(prediction, values[:, expected])
    assert torch.equal(target, (values + 1)[:, expected])
    assert torch.equal(selected_coordinates, coordinates[:, expected])


def test_saved_preview_queries_are_explicitly_marked_auxiliary():
    coordinates = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1).repeat(1, 1, 2)
    values = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
    accumulator = TopologySetAccumulator(
        family=None,
        config={},
        dataset_field_names=("u", "v"),
        normalization_method="mean_std",
        source_grid_shape=(4, 4),
        query_point_count=8,
        query_seed=100045,
    )
    accumulator.use_preselected_query_points(seed=2027, description="pinned preview query_coords")

    prediction, _, selected_coordinates = accumulator._select_queries(
        values, values, coordinates
    )
    assert torch.equal(accumulator.query_indices, torch.arange(8))
    assert torch.equal(prediction, values)
    assert torch.equal(selected_coordinates, coordinates)
    assert accumulator.query_policy == "preselected_saved_queries"
    assert accumulator.matches_training_fixed_shared_selector is False
    assert accumulator.query_seed == 2027


def test_exact_betti_thresholds_use_reference_helper_shape_and_ties():
    reference = torch.tensor(
        [
            [[[0.0, 0.0, 1.0], [0.0, 2.0, 2.0]]],
            [[[3.0, 1.0, 1.0], [0.0, 0.0, 2.0]]],
        ],
        dtype=torch.float32,
    )
    generated = torch.flip(reference, dims=(-1,))
    quantiles = (0.1, 0.5, 0.9)

    levels = reference_levels(reference[:, 0], quantiles)
    generated_counts, reference_counts = exact_betti_from_filtration_banks(
        reference, generated, quantiles, periodic=False
    )

    assert levels.shape == (2, 3)
    assert levels.dtype == torch.float64
    assert generated_counts.shape == reference_counts.shape == (2, 2, 3)
    expected = exact_betti_curves(
        torch.cat((-generated[:, 0], -reference[:, 0])),
        torch.cat((-levels, -levels)),
        periodic=False,
    )
    torch.testing.assert_close(generated_counts, expected[:2])
    torch.testing.assert_close(reference_counts, expected[2:])


def test_topology_distance_figure_honors_shared_limits_and_writes_vectors(tmp_path):
    import matplotlib.pyplot as plt

    output = tmp_path / "persistence_term_distributions.png"
    before = set(plt.get_fignums())
    render_topology_distance_distributions(
        np.asarray([[0.2, 0.5], [0.4, 0.8], [0.6, 1.0]]),
        ("self.persistence", "component-weighted total"),
        ("component", "component_weighted_total"),
        output,
        title="Persistence distances",
        subtitle="matched reference and prediction",
        y_limits=(0.0, 1.2),
    )

    assert output.is_file()
    assert output.with_suffix(".pdf").is_file()
    assert output.with_suffix(".svg").is_file()
    assert set(plt.get_fignums()) == before


def test_betti_curves_support_single_direction_and_close_figure(tmp_path):
    import matplotlib.pyplot as plt

    reference = np.asarray([[[[2, 2, 1], [1, 0, 0]]]], dtype=np.int64)
    reconstruction = np.asarray([[[[2, 1, 1], [1, 0, 0]]]], dtype=np.int64)
    metadata = ({"metric": "self.u", "direction": "sublevel"},)
    output = tmp_path / "betti_curves.png"
    before = set(plt.get_fignums())

    render_topology_betti_curves(
        reference,
        reconstruction,
        metadata,
        ("self.u",),
        ("sublevel",),
        (0.1, 0.5, 0.9),
        output,
        title="Exact Betti curves",
        subtitle="configured raster",
    )

    assert output.is_file()
    assert output.with_suffix(".pdf").is_file()
    assert output.with_suffix(".svg").is_file()
    assert set(plt.get_fignums()) == before


def test_betti_curves_accept_dimension_specific_limits(tmp_path, monkeypatch):
    from phycoflow_reconstruction.evaluation import topology_set

    captured_limits = []
    original_save = topology_set._save_publication_figure

    def capture(figure, output_path, *, dpi=300):
        captured_limits.extend(axis.get_ylim() for axis in figure.axes[:2])
        return original_save(figure, output_path, dpi=dpi)

    monkeypatch.setattr(topology_set, "_save_publication_figure", capture)
    curves = np.asarray([[[[1, 2, 3], [20, 80, 125]]]], dtype=np.int64)
    output = tmp_path / "betti_dimension_limits.png"

    topology_set.render_topology_betti_curves(
        curves,
        curves,
        ({"metric": "self.u", "direction": "sublevel"},),
        ("self.u",),
        ("sublevel",),
        (0.1, 0.5, 0.9),
        output,
        title="Exact Betti curves",
        subtitle="separate homology scales",
        dimension_y_limits=((0, 4), (0, 140)),
    )

    assert captured_limits == [(0.0, 4.0), (0.0, 140.0)]
    assert output.is_file()
