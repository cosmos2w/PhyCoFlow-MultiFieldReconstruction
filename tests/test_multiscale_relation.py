import copy

import pytest
import torch
from test_multiscale_gram_extension import _config, _fitted, _score, _states

from phycoflow_reconstruction.coherence.families.multiscale_relation import (
    ANCHOR_PATH,
    RELATION_DEFINITION,
    RHO_PATH,
    SELF_PATH,
    parse_relation_extension_config,
)
from phycoflow_reconstruction.coherence.families.multiscale_spectrum.losses import (
    almost_fair_crps,
    band_coherence,
    vector_energy_score,
)
from phycoflow_reconstruction.coherence.registry import build_coherence_family
from phycoflow_reconstruction.data.normalization import FieldNormalizer


def config(fields=("u", "v")):
    result = _config(fields)
    result.pop("gram_extension")
    result["relation_extension"] = {"definition": RELATION_DEFINITION, "self_weight": 0.2,
                                    "rho_weight": 0.3, "std_floor": 0.001,
                                    "relative_std_floor": 0.05}
    return result


def test_scores_match_direct_fair_arithmetic_and_draw_gradients():
    states = _states()
    family, coords, _ = _fitted(config(), states)
    x, y = (states[i:i+2].clone().requires_grad_() for i in (0, 2))
    target = states[4:6]
    result = _score(family, coords, x, y, target)
    stats = [family._statistics(v) for v in (x, y, target)]
    blocks = [family._descriptors(s)["log_band_power"] for s in stats]
    denominator = family.relation_self_scale * family.relation_self_scale.numel() ** 0.5
    expected_self = vector_energy_score(torch.stack([v / denominator for v in blocks[:2]]), blocks[2] / denominator)[0]
    rho = [band_coherence(s["gram"], family.power_floor) for s in stats]
    expected_rho = ((rho[0] - rho[2]).abs() + (rho[1] - rho[2]).abs() - (rho[0] - rho[1]).abs()).mean((1, 2)) / 2
    torch.testing.assert_close(result.component_results[SELF_PATH].per_sample_cost, expected_self)
    torch.testing.assert_close(result.component_results[RHO_PATH].per_sample_cost, expected_rho)
    torch.testing.assert_close(result.per_sample_cost, result.component_results[ANCHOR_PATH].per_sample_cost + 0.2 * expected_self + 0.3 * expected_rho)
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in torch.autograd.grad(result.scalar_loss, (x, y)))


def test_fair_j2_bracketing_coincidence_and_outside():
    draws = torch.tensor([[0., 1., 1.], [2., 1., 2.]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([1., 1., 4.], dtype=torch.float64)
    score = almost_fair_crps(draws, target)
    torch.testing.assert_close(score, torch.tensor([0., 0., 2.], dtype=torch.float64))
    gradient = torch.autograd.grad(score.sum(), draws)[0]
    torch.testing.assert_close(gradient[:, :2], torch.zeros_like(gradient[:, :2]))
    assert torch.isfinite(gradient).all()


def test_artifact_roundtrip_and_strict_calibration():
    cfg = config()
    family, coords, spec = _fitted(cfg, _states())
    artifact = family.state_artifact()
    restored = build_coherence_family("multiscale_spectrum", cfg, spec, FieldNormalizer.identity(2))
    restored.load_state_artifact(artifact)
    states = _states()
    torch.testing.assert_close(_score(restored, coords, *states[:3, None]).scalar_loss,
                               _score(family, coords, *states[:3, None]).scalar_loss)
    bad = copy.deepcopy(artifact)
    bad["state_dict"]["relation_self_scale"][0] *= 2
    with pytest.raises(ValueError, match="scales differ"):
        restored.load_state_artifact(bad)
    bad = copy.deepcopy(cfg)
    bad["gram_extension"] = _config()["gram_extension"]
    with pytest.raises(ValueError, match="cannot be combined"):
        parse_relation_extension_config(bad, spec.field_names)
    bad = copy.deepcopy(cfg)
    bad["objective"]["coherence_crps_weight"] = 0.1
    with pytest.raises(ValueError, match="coherence_crps_weight=0"):
        parse_relation_extension_config(bad, spec.field_names)


def test_one_field_omits_rho_and_zero_states_are_finite():
    states = torch.zeros(6, 16, 1, dtype=torch.float64)
    family, coords, _ = _fitted(config(("u",)), states)
    x, y = (states[:1].clone().requires_grad_() for _ in range(2))
    result = _score(family, coords, x, y, states[:1])
    assert RHO_PATH not in result.component_results
    assert torch.isfinite(result.scalar_loss)
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(result.scalar_loss, (x, y)))


def test_equal_self_power_opposite_signed_relation_and_unit_change():
    states = _states()
    family, coords, _ = _fitted(config(), states)
    target = states[:1]
    opposite = target.clone()
    opposite[..., 1] *= -1
    target_stats, opposite_stats = (family._statistics(value) for value in (target, opposite))
    torch.testing.assert_close(family._descriptors(target_stats)["log_band_power"],
                               family._descriptors(opposite_stats)["log_band_power"])
    rho = band_coherence(target_stats["gram"], family.power_floor)
    torch.testing.assert_close(band_coherence(opposite_stats["gram"], family.power_floor), -rho)
    assert rho.abs().max() > 0
    factors = torch.tensor([2., 3.], dtype=torch.float64)
    shifts = torch.tensor([4., -2.], dtype=torch.float64)
    converted = states * factors + shifts
    old = config()
    old["units"] = "physical_units"
    old["calibration"]["field_unit_floor"] = {"u": 0.01, "v": 0.02}
    new = copy.deepcopy(old)
    new["calibration"]["field_unit_floor"] = {"u": 0.02, "v": 0.06}
    a, coords, _ = _fitted(old, states)
    b, _, _ = _fitted(new, converted)
    original = _score(a, coords, states[:1], states[1:2], states[2:3])
    transformed = _score(b, coords, converted[:1], converted[1:2], converted[2:3])
    for path in (ANCHOR_PATH, SELF_PATH, RHO_PATH):
        torch.testing.assert_close(original.component_results[path].scalar_loss,
                                   transformed.component_results[path].scalar_loss, rtol=1e-8, atol=1e-8)


def test_set_evaluator_reports_relation_identity(tmp_path):
    import json

    from phycoflow_reconstruction.evaluation.multiscale_set import MultiscaleSetAccumulator

    family, coords, _ = _fitted(config(), _states())
    accumulator = MultiscaleSetAccumulator(family=family, artifact_path=tmp_path / "relation.pt",
        artifact_sha256="0" * 64, query_point_count=len(coords), query_seed=7, draw_count=4)
    states = _states()
    accumulator.update(states[:1], states[1:2], coords[None], "sample",
                       conditional_draws=tuple(states[i:i+1] for i in (2, 3, 4)))
    output = accumulator.finalize(tmp_path, split="validation", checkpoint_label="test", run_label="test", scale="linear")
    report = json.loads(__import__("pathlib").Path(output["report"]).read_text())
    assert report["score_definition_version"] == RELATION_DEFINITION
    assert report["anchor_score_definition_version"] == "signed_spectral_mean_augmented_v4"
    assert SELF_PATH in report["component_means"] and RHO_PATH in report["component_means"]
