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
    assert summary["score_draws"] == 3 and summary["schema_version"] == 3
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


@pytest.mark.parametrize("mutation", ["solver", "clamp", "guidance", "chunk", "weight", "steps", "seed"])
def test_effective_sampler_fingerprint_rejects_incompatible_calls(panel, mutation):
    panel.config["optimization"] = {"multitask": {"external_policy": {"method": "lagrangian"}}}
    panel.config["observation_consistency"] = {"mode": "endpoint_smooth", "final_clamp": False,
                                               "strength": 1.0, "chunk_size": 4096}
    panel.model.reconstruction_query_chunk_size = 256
    before = evaluation.comparison_fingerprint(panel, steps=2, draws=3)
    kwargs = {"steps": 2, "draws": 3}
    if mutation == "solver":
        panel.config["rollout"]["solver"] = "heun"
    elif mutation == "clamp":
        panel.config["observation_consistency"]["final_clamp"] = True
    elif mutation == "guidance":
        panel.config["observation_consistency"]["strength"] = 0.5
    elif mutation == "chunk":
        panel.model.reconstruction_query_chunk_size = 128
    elif mutation == "weight":
        kwargs["weight_selection"] = "live"
    elif mutation == "steps":
        kwargs["steps"] = 4
    elif mutation == "seed":
        kwargs["seed"] = 99
    assert evaluation.comparison_fingerprint(panel, **kwargs) != before
    contract = evaluation.effective_sampler(panel, **kwargs)
    assert contract["requested_rollout_solver"] == panel.config["rollout"]["solver"]
    assert contract["solver"] == panel.model.ode_solver
    assert contract["observation_config"] == panel.config["observation_consistency"]


def test_single_cluster_has_point_estimate_without_fake_confidence():
    result = evaluation.ratio_bootstrap([0., 1., 5.], [1., 1., 1.], ["trajectory"] * 3)
    assert result["ratio"] == 2.0 and result["group_count"] == 1
    assert result["low"] is None and result["high"] is None
    assert not result["interval_available"]
    assert "interval unavailable" in evaluation.fmt_ratio(result)


def test_time_blocks_are_paired_descriptions_without_inferential_interval(tmp_path):
    path = tmp_path / "groups.json"
    path.write_text(json.dumps({"unit": "time_block", "labels": {"t:0": "early", "t:1": "late", "t:2": "late"}}))
    clusters, mode = evaluation.load_grouping(path, ["t:0", "t:1", "t:2"])
    result = evaluation.ratio_bootstrap([2., 1., 5.], [1., 1., 1.], clusters, grouping=mode)
    assert result["block_ratios"] == {"early": 2.0, "late": 3.0}
    assert result["group_sizes"] == {"early": 1, "late": 2}
    assert result["low"] is None and not result["interval_available"]
    with pytest.raises(ValueError, match="exactly match"):
        evaluation.load_grouping(path, ["t:0", "t:1"])


def test_comparison_rejects_sampler_or_identity_mismatch(panel):
    arrays, baseline = evaluation.analyse_ensemble(panel, panel.members, label="base")
    changed = copy.deepcopy(baseline)
    changed["sampler"]["steps"] += 1
    with pytest.raises(ValueError, match="sampler"):
        evaluation.compare_arms({"base": {"arrays": arrays, "summary": baseline},
                                 "other": {"arrays": arrays, "summary": changed}},
                                ["a", "b", "c"], control="base", reference=None)


def test_mean_augmented_score_keeps_legacy_score_and_detects_offset(panel):
    legacy = panel.families["multiscale_spectrum"]
    config = copy.deepcopy(legacy.config)
    config["objective"].update(score_definition_version="signed_spectral_mean_augmented_v4",
                               joint_energy_score_weight=0.25, mean_weight=1.0)
    config["calibration"] = {"field_unit_floor": 0.01}
    family = MultiscaleSpectrumFamily(config, DataSpec(("u", "v"), ("1", "1"), 1, (12,)),
                                      FieldNormalizer.identity(2))
    coords = panel.comparison_batch.query_coords[0]
    references = torch.randn(10, 12, 2, generator=torch.Generator().manual_seed(34), dtype=torch.float64)
    family.fit_reference_tensors(coords, references, sample_ids=[str(i) for i in range(10)])
    panel.families["multiscale_spectrum"] = family
    target = panel.comparison_batch.target_fields
    exact_draws = target.unsqueeze(0).repeat(3, 1, 1, 1)
    shifted = exact_draws + 2.0
    exact, exact_summary = evaluation.analyse_ensemble(panel, exact_draws, label="exact")
    offset, summary = evaluation.analyse_ensemble(panel, shifted, label="offset")
    assert summary["score_definition_version"] == "signed_spectral_v3"
    assert summary["metric_settings"]["joint_spectral_mean_es"]["definition"] == "signed_spectral_mean_augmented_v4"
    assert offset["joint_spectral_mean_es"].mean() > exact["joint_spectral_mean_es"].mean() + 1.0
    assert offset["joint_spectral_es"].mean() == pytest.approx(exact["joint_spectral_es"].mean(), abs=1e-6)
    assert exact_summary["metric_settings"] == summary["metric_settings"]


def test_live_and_configured_ema_contract_matches_actual_draw(monkeypatch):
    from contextlib import contextmanager

    from phycoflow_reconstruction.contracts import ObservationBatch

    class Model(torch.nn.Module):
        capabilities = SimpleNamespace(structured_grid_required=False)
        ode_solver = "euler"
        _ema_eval = True
        _ema = object()

        def __init__(self):
            super().__init__()
            self.swapped = False

        @contextmanager
        def evaluation_weight_context(self):
            self.swapped = True
            try:
                yield
            finally:
                self.swapped = False

        def reconstruct(self, batch, **kwargs):
            return SimpleNamespace(prediction=torch.full((1, 3, 1), 2. if self.swapped else 1.))

    batch = ObservationBatch(obs_coords=torch.zeros(1, 1, 2), obs_values=torch.zeros(1, 1, 1),
        obs_field_ids=torch.zeros(1, 1, dtype=torch.long), obs_valid_mask=torch.ones(1, 1, dtype=torch.bool),
        query_coords=torch.zeros(1, 3, 2), query_valid_mask=torch.ones(1, 3, dtype=torch.bool),
        target_fields=torch.ones(1, 3, 1), sample_ids=("state",))
    panel = SimpleNamespace(model=Model(), comparison_batch=batch, evaluation_batch=batch,
        config={"evaluation": {"generation_steps": 2}, "rollout": {"solver": "euler"}})
    live = evaluation.draw_members(panel, 2, weight_selection="live")
    ema = evaluation.draw_members(panel, 2, weight_selection="configured")
    assert torch.all(live == 1.) and torch.all(ema == 2.)
    assert evaluation.effective_sampler(panel, weight_selection="live")["effective_weights"] == "live"
    assert evaluation.effective_sampler(panel, weight_selection="configured")["effective_weights"] == "ema"


def test_external_dispatch_reports_actual_flow_or_native_solver(panel):
    panel.config["optimization"] = {"multitask": {"external_policy": {"method": "lagrangian"}}}
    panel.config["rollout"]["solver"] = "heun"
    panel.config["observation_consistency"] = {"mode": "none", "final_clamp": True}
    native = evaluation.effective_sampler(panel)
    assert native["dispatch"] == "native_differentiable_hook"
    assert native["requested_rollout_solver"] == "heun" and native["solver"] == "euler"
    assert native["observation_config"]["final_clamp"] is True
    assert native["observation_effective"]["final_clamp"] is False
    panel.model.sample_source = lambda *args, **kwargs: None
    panel.model.velocity = lambda *args, **kwargs: None
    flow = evaluation.effective_sampler(panel)
    assert flow["dispatch"] == "flow_hooks" and flow["solver"] == "heun"
    assert flow != native
