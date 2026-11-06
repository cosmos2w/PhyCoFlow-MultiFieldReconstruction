"""The offline yardstick uses declared, comparable metrics and frozen geometry."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from phycoflow_reconstruction.coherence.families.multiscale_spectrum import MultiscaleSpectrumFamily
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.evaluation import spectral as evaluation


@pytest.fixture
def panel():
    fields = ("u", "v")
    coords = torch.linspace(0, 1, 12, dtype=torch.float64)[:, None]
    states = torch.randn(
        10, 12, 2, generator=torch.Generator().manual_seed(34), dtype=torch.float64
    )
    config = {
        "fields": list(fields),
        "geometry": {"operator": "fem_p1", "coordinate_scaling": "none"},
        "bands": {"count": 2},
        "numerics": {"dtype": "float64"},
        "objective": {"kind": "signed_band_energy_score", "almost_fair_alpha": 1.0},
    }
    family = MultiscaleSpectrumFamily(
        config, DataSpec(fields, ("1", "1"), 1, (12,)), FieldNormalizer.identity(2)
    )
    family.fit_reference_tensors(coords, states, sample_ids=[str(i) for i in range(10)])
    batch = SimpleNamespace(
        query_coords=coords[None].expand(3, -1, -1),
        query_valid_mask=torch.ones(3, 12, dtype=torch.bool),
        target_fields=states[4:7],
        obs_coords=coords[None, :2].expand(3, -1, -1),
        obs_values=states[4:7, :2, :1].clone(),
        obs_field_ids=torch.zeros(3, 2, dtype=torch.long),
        obs_valid_mask=torch.ones(3, 2, dtype=torch.bool),
        sample_ids=["a", "b", "c"],
        metadata={},
    )
    return SimpleNamespace(
        families={"multiscale_spectrum": family},
        comparison_batch=batch,
        evaluation_batch=batch,
        config={
            "rollout": {"steps": 16, "solver": "euler"},
            "evaluation": {"generation_steps": 16},
        },
        model=SimpleNamespace(ode_solver="euler"),
        members=torch.stack([states[:3], states[1:4], states[2:5]]),
    )


def test_evaluator_defaults_to_fair_all_draws_and_labels_legacy_options(panel):
    arrays, summary = evaluation.analyse_ensemble(panel, panel.members, label="fair")
    assert summary["metric_settings"]["signed_band_es"]["alpha"] == 1.0
    assert summary["score_draws"] == 3 and summary["schema_version"] == 2
    legacy, old_summary = evaluation.analyse_ensemble(
        panel, panel.members, label="legacy", score_alpha=0.95, score_draws=2
    )
    assert old_summary["metric_settings"]["signed_band_es"]["draws"] == 2
    assert old_summary["metric_settings"]["coherence_crps"]["draws"] == 3
    torch.testing.assert_close(
        torch.from_numpy(arrays["joint_spectral_es"]), torch.from_numpy(legacy["joint_spectral_es"])
    )
    assert (legacy["coherence_crps"] >= arrays["coherence_crps"] - 1e-12).all()
    json.dumps(summary, allow_nan=False)
    text = evaluation.render_comparison(
        {"fair": {"summary": summary}}, {}, control="fair", reference=None, draws=3, steps=16
    )
    assert "alpha=1.0, J=3" in text and "variance / debiased error" in text
    assert "joint_attraction" in arrays and "coherence_repulsion" in arrays


def test_fingerprint_binds_sensor_values_and_calibration(panel):
    before = evaluation.comparison_fingerprint(panel)
    panel.comparison_batch.obs_values[0, 0, 0] += 0.1
    changed = evaluation.comparison_fingerprint(panel)
    assert changed != before
    panel.families["multiscale_spectrum"].band_scale.mul_(1.1)
    assert evaluation.comparison_fingerprint(panel) != changed


def test_incompatible_metric_metadata_cannot_be_compared(panel):
    arrays, summary = evaluation.analyse_ensemble(panel, panel.members, label="fair")
    changed = copy.deepcopy(summary)
    changed["metric_settings"]["signed_band_es"]["alpha"] = 0.95
    results = {
        "fair": {"arrays": arrays, "summary": summary},
        "other": {"arrays": arrays, "summary": changed},
    }
    with pytest.raises(ValueError, match="metric_settings"):
        evaluation.compare_arms(results, [0, 1, 2], control="fair", reference=None)


def test_undefined_bootstrap_ratios_are_not_marked_as_improvements():
    ratio = evaluation.ratio_bootstrap([1.0, 2.0], [-1.0, -2.0], [0, 1], reps=10)
    assert ratio["ratio"] is None and evaluation.fmt_ratio(ratio) == "undefined"



@pytest.mark.parametrize("external", [False, True])
def test_sampling_uses_the_declared_endpoint_and_needs_no_private_prior(monkeypatch, external):
    from phycoflow_reconstruction.contracts import ObservationBatch

    class Model(torch.nn.Module):
        capabilities = SimpleNamespace(structured_grid_required=False)

        def reconstruct(self, batch, **kwargs):
            assert batch.target_fields is None
            return SimpleNamespace(prediction=torch.full((1, 3, 1), 2.0))

    batch = ObservationBatch(
        obs_coords=torch.zeros(1, 1, 2), obs_values=torch.zeros(1, 1, 1),
        obs_field_ids=torch.zeros(1, 1, dtype=torch.long),
        obs_valid_mask=torch.ones(1, 1, dtype=torch.bool),
        query_coords=torch.zeros(1, 3, 2), query_valid_mask=torch.ones(1, 3, dtype=torch.bool),
        target_fields=torch.ones(1, 3, 1), sample_ids=("state",),
    )
    panel = SimpleNamespace(model=Model(), comparison_batch=batch, evaluation_batch=batch,
        config={"evaluation": {"generation_steps": 2}, "rollout": {"solver": "euler"},
                "observation_consistency": {"mode": "none", "final_clamp": False},
                "optimization": {"multitask": {"external_policy": {"method": "lagrangian"}}}
                    if external else {}})

    def endpoint(model, target_free, **options):
        assert target_free.target_fields is None
        return torch.full((1, 3, 1), 3.0)

    monkeypatch.setattr(evaluation, "differentiable_reconstruction", endpoint)
    members = evaluation.draw_members(panel, 2)
    assert members.shape == (2, 1, 3, 1)
    assert torch.all(members == (3.0 if external else 2.0))
