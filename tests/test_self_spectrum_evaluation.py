"""Focused evaluation coverage for the cross-spectrum self-spectrum component."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from phycoflow_reconstruction.evaluation.coherence_set import (
    CrossSpectrumAccumulator,
    _self_spectrum_values,
)


def test_self_spectrum_values_are_modewise_per_field_and_bounded() -> None:
    generated = torch.tensor(
        [
            [[1.0, 2.0], [1.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ],
        dtype=torch.complex64,
    )
    reference = torch.ones_like(generated)

    discrepancy, score = _self_spectrum_values(generated, reference, 1.0e-8)

    # Generated field powers differ from reference at one field/mode only.
    torch.testing.assert_close(discrepancy, torch.tensor([0.0, 1.125]))
    assert score.shape == (2,)
    assert torch.all((score >= 0.0) & (score <= 1.0))
    assert score[0].item() == 1.0
    assert score[1].item() < 1.0


def _fake_family() -> SimpleNamespace:
    return SimpleNamespace(
        field_names=("u", "v"),
        pairs=((0, 1),),
        band_names=("low", "high"),
        band_ids=torch.tensor([0, 0, 1]),
        eigenvalues=torch.tensor([1.0, 2.0, 3.0]),
        eps=1.0e-8,
        units="model_units",
        family_weight=1.0,
        component_weights={"self_spectrum": 1.0},
        required_batch_size=1,
        geometry_sha256="geometry",
        k_neighbors=2,
        resolved_sigma=0.5,
    )


def test_self_spectrum_finalize_writes_field_metrics_report_and_figure(tmp_path: Path) -> None:
    accumulator = CrossSpectrumAccumulator(
        family=_fake_family(),
        config={"components": {"self_spectrum": {"enabled": True, "weight": 1.0}}},
        query_point_count=3,
        query_seed=11,
        aggregation="pooled",
        ensemble_size=2,
        ensemble_seed=7,
        generated_coefficients=[
            torch.tensor([[1.0, 2.0], [2.0, 1.0], [1.0, 1.0]]),
            torch.tensor([[1.0, 2.0], [2.0, 1.0], [1.0, 1.0]]),
        ],
        reference_coefficients=[
            torch.ones(3, 2),
            torch.ones(3, 2),
        ],
        sample_ids=["s0", "s1"],
        query_indices=torch.tensor([0, 1, 2]),
    )

    result = accumulator.finalize(
        tmp_path,
        split="test",
        checkpoint_label="best",
        run_label="fixture",
        scale="linear",
    )
    destination = Path(result["directory"])
    assert (destination / "self_spectrum_coherence.png").stat().st_size > 0
    assert (destination / "metrics.csv").is_file()
    report = json.loads((destination / "report.json").read_text(encoding="utf-8"))
    assert report["components"]["self_spectrum"]["definition"] == (
        "modewise per-field auto-spectrum matching"
    )
    assert report["statistics"]["self_spectrum"]["sub_terms"].keys() == {"u", "v"}
    with np.load(destination / "metrics.npz", allow_pickle=False) as payload:
        assert payload["self_spectrum_absolute_discrepancy"].shape == (2,)
        assert payload["self_spectrum_coherence_score"].shape == (2,)
        assert payload["self_spectrum_coherence_score_by_ensemble"].shape == (1, 2)
        assert payload["field_names"].tolist() == ["u", "v"]
        np.testing.assert_allclose(
            payload["self_spectrum_coherence_score"],
            payload["self_spectrum_coherence_score_by_ensemble"].mean(axis=0),
        )
        assert np.all((payload["self_spectrum_coherence_score"] >= 0.0) & (payload["self_spectrum_coherence_score"] <= 1.0))
