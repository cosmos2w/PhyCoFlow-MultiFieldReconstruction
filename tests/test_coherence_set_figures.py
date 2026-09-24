"""Contracts for publication-ready cross-spectrum set-level figures."""

import numpy as np

from phycoflow_reconstruction.evaluation.coherence_set import (
    render_cross_spectrum_band_profiles,
    render_cross_spectrum_paired_score_bars,
)


def test_graph_band_profile_uses_validated_per_field_fractions_and_vectors(tmp_path):
    import matplotlib.pyplot as plt

    reference = np.asarray([[0.2, 0.5], [0.3, 0.25], [0.5, 0.25]])
    reconstruction = np.asarray([[0.3, 0.4], [0.25, 0.35], [0.45, 0.25]])
    output = tmp_path / "spectral_band_profiles.png"
    before = set(plt.get_fignums())

    render_cross_spectrum_band_profiles(
        reference,
        reconstruction,
        ("low", "mid", "high"),
        ("CO", "T"),
        output,
        title="Graph spectral energy across bands",
        subtitle="Validation · n=8 · per-snapshot fractions",
    )

    assert output.is_file()
    assert output.with_suffix(".pdf").is_file()
    assert output.with_suffix(".svg").is_file()
    assert set(plt.get_fignums()) == before


def test_paired_cross_spectrum_plot_requires_matched_ids_and_writes_vectors(tmp_path):
    import matplotlib.pyplot as plt

    source = {
        "component_names": np.asarray(["self_spectrum", "same_frequency", "cross_frequency"]),
        "component_coherence_scores": np.asarray([0.96, 0.90, 0.82]),
        "component_coherence_score_std": np.asarray([0.02, 0.03, 0.05]),
        "family_coherence_score": np.asarray(0.89),
        "family_coherence_score_std": np.asarray(0.04),
        "sample_ids": np.asarray(["s0", "s1", "s2"]),
        "ensemble_sample_ids": np.asarray([["s0", "s1"], ["s2", "s3"]]),
        "query_indices": np.asarray([1, 4, 8]),
    }
    post = {
        **source,
        "component_coherence_scores": np.asarray([0.98, 0.94, 0.88]),
        "family_coherence_score": np.asarray(0.94),
    }
    output = tmp_path / "cross_spectrum_source_post.png"
    before = set(plt.get_fignums())

    render_cross_spectrum_paired_score_bars(
        source,
        post,
        output,
        title="Matched configured cross-spectrum scores",
        subtitle="Source and post-training · mean ±1 SD across ensembles",
        component_filter=("same_frequency", "cross_frequency"),
        include_family_aggregate=False,
    )

    assert output.is_file()
    assert output.with_suffix(".pdf").is_file()
    assert output.with_suffix(".svg").is_file()
    assert set(plt.get_fignums()) == before

    post["query_indices"] = np.asarray([1, 4, 9])
    try:
        render_cross_spectrum_paired_score_bars(
            source,
            post,
            tmp_path / "must_not_render.png",
            title="Mismatched query selector",
            subtitle="invalid comparison",
        )
    except ValueError as error:
        assert "query_indices" in str(error)
    else:
        raise AssertionError("paired cross-spectrum plot accepted different query indices")
