"""Saved-payload contracts for supplementary coherence figures."""

import numpy as np
import pytest
import torch

from phycoflow_reconstruction.coherence.families.topology.persistence_objective import (
    PersistenceTopologyObjective,
)
from phycoflow_reconstruction.evaluation.explanatory_coherence import (
    render_cross_band_error,
    render_cross_pair_scores,
    render_topology_diagrams,
)


def _spectral_payloads():
    common = {
        "sample_ids": np.asarray(["s0", "s1"]),
        "selected_sample_ids": np.asarray(["s0", "s1"]),
        "ensemble_sample_ids": np.asarray([["s0", "s1"]]),
        "query_indices": np.asarray([0, 2, 4]),
        "pair_labels": np.asarray(["u–v"]),
        "graph_band_names": np.asarray(["low", "high"]),
        "field_names": np.asarray(["u", "v"]),
        "ensemble_count": np.asarray(1),
        "same_frequency_coherence_score_by_ensemble": np.asarray([[0.7]]),
        "cross_frequency_coherence_score_by_ensemble": np.asarray([[0.4]]),
        "graph_band_energy_fraction_reference_mean": np.asarray([[0.7, 0.6], [0.3, 0.4]]),
    }
    base = {
        **common,
        "graph_band_energy_fraction_reconstruction_mean": np.asarray([[0.6, 0.5], [0.4, 0.5]]),
    }
    post = {
        **common,
        "same_frequency_coherence_score_by_ensemble": np.asarray([[0.8]]),
        "cross_frequency_coherence_score_by_ensemble": np.asarray([[0.5]]),
        "graph_band_energy_fraction_reconstruction_mean": np.asarray([[0.68, 0.59], [0.32, 0.41]]),
    }
    return base, post


def test_cross_explanations_measure_matched_changes(tmp_path):
    base, post = _spectral_payloads()
    pairs = render_cross_pair_scores(base, post, tmp_path / "cross_pair_scores.png")
    bands = render_cross_band_error(base, post, tmp_path / "cross_band_error.png")

    assert pairs["score_change_percentage_points"]["same_frequency"]["u–v"] == pytest.approx(10.0)
    assert pairs["score_change_percentage_points"]["cross_frequency"]["u–v"] == pytest.approx(10.0)
    assert bands["mean_absolute_error_percentage_points"]["post_training"] < bands["mean_absolute_error_percentage_points"]["base"]
    for stem in ("cross_pair_scores", "cross_band_error"):
        assert all((tmp_path / f"{stem}.{suffix}").is_file() for suffix in ("png", "pdf", "svg"))

    invalid = {**post, "ensemble_sample_ids": np.asarray([["s1", "s0"]])}
    with pytest.raises(ValueError, match="ensemble_sample_ids"):
        render_cross_pair_scores(base, invalid, tmp_path / "invalid.png")


def test_topology_diagram_distance_matches_saved_self_component(tmp_path):
    config = {
        "target_use": "paired_supervised",
        "strategy": "cubical_persistence",
        "fields": ["u"],
        "geometry": {"grid_shape": [4, 4], "periodic": False},
        "filtration": {"dimensions": [0, 1], "directions": ["sublevel", "superlevel"]},
        "components": {"self": {"enabled": True, "fields": ["u"], "weight": 1.0}, "mutual": {"enabled": False}},
        "persistence": {"distance": "sliced_wasserstein", "projections": 8, "essential_weight": 0.1},
    }
    objective = PersistenceTopologyObjective(config, ("u",))
    reference = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) / 16
    prediction = reference.clone()
    prediction[0, 0, 1, 1] += 0.2
    result = objective(prediction, reference)
    _, _, _, metrics = objective._filtrations(objective._descriptor_fields(reference))
    scored = float(result.component_results["topology.self.persistence"].per_sample_cost[0])
    payload = {
        "field_names": np.asarray(["u"]),
        "sample_ids": np.asarray(["s0"]),
        "representative_index": np.asarray(0),
        "representative_reference_fields": reference[0].numpy(),
        "representative_reconstruction_fields": prediction[0].numpy(),
        "filtration_metric": np.asarray(metrics),
        "filtration_direction": np.asarray(["sublevel", "superlevel"]),
        "objective_component_names": np.asarray(["topology.self.persistence"]),
        "objective_component_distances": np.asarray([[scored]]),
    }
    report = {"objective": {"settings": config}}
    explanation = render_topology_diagrams(payload, report, tmp_path, role="Post-training")
    assert explanation["recomputed_self_persistence_distance"] == pytest.approx(scored, rel=2e-3, abs=2e-6)
    assert all((tmp_path / f"u.{suffix}").is_file() for suffix in ("png", "pdf", "svg"))

    invalid = {**payload, "objective_component_distances": np.asarray([[scored + 0.1]])}
    with pytest.raises(ValueError, match="parity failed"):
        render_topology_diagrams(invalid, report, tmp_path / "invalid", role="Post-training")
