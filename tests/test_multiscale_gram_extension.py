"""Versioned fair self/mutual Scale augmentation and frozen v4 compatibility."""

import copy
import json
import math
from pathlib import Path

import pytest
import torch

from phycoflow_reconstruction.coherence.families.multiscale_gram import (
    ANCHOR_PATH,
    GRAM_DEFINITION,
    MUTUAL_PATH,
    SELF_PATH,
    MultiscaleGramFamily,
    parse_gram_extension_config,
)
from phycoflow_reconstruction.coherence.families.multiscale_spectrum import MultiscaleSpectrumFamily
from phycoflow_reconstruction.coherence.families.multiscale_spectrum.losses import (
    vector_energy_score,
)
from phycoflow_reconstruction.coherence.families.multiscale_spectrum.statistics import band_fields
from phycoflow_reconstruction.coherence.registry import build_coherence_family
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer


def _config(fields=("u", "v"), *, units="model_units", floors=0.01):
    return {
        "fields": list(fields),
        "units": units,
        "geometry": {"operator": "fem_p1", "coordinate_scaling": "none"},
        "bands": {"count": 2},
        "calibration": {"field_unit_floor": floors},
        "numerics": {"dtype": "float64"},
        "objective": {
            "kind": "signed_band_energy_score",
            "conditional_draws": 2,
            "almost_fair_alpha": 1.0,
            "joint_energy_score_weight": 0.25,
            "score_definition_version": "signed_spectral_mean_augmented_v4",
        },
        "gram_extension": {
            "definition": GRAM_DEFINITION,
            "weight": 0.2,
            "std_floor": 0.001,
            "relative_std_floor": 0.05,
        },
    }


def _fitted(config, states):
    fields = tuple(config["fields"])
    n = states.shape[1]
    coords = torch.linspace(0.0, 1.0, n, dtype=torch.float64)[:, None] ** 1.3
    spec = DataSpec(fields, ("1",) * len(fields), 1, (n,))
    family = build_coherence_family("multiscale_spectrum", config, spec, FieldNormalizer.identity(len(fields)))
    family.fit_reference_tensors(coords, states, sample_ids=[str(i) for i in range(len(states))])
    return family, coords, spec


def _states(fields=2):
    return torch.randn(8, 16, fields, generator=torch.Generator().manual_seed(23), dtype=torch.float64)


def _score(family, coords, first, second, target):
    return family(first, target, coordinates=coords[None].expand(first.shape[0], -1, -1), context={"conditional_draws": (second,)})


def test_two_separate_fair_blocks_preserve_anchor_and_both_draw_gradients():
    config = _config()
    states = _states()
    family, coords, _ = _fitted(config, states)
    assert isinstance(family, MultiscaleGramFamily)
    components = {component.name: component for component in family.spec.components}
    assert components["signed_anchor_total.conditional"].metadata == {
        "trained": False, "derived_reporting_view": True,
    }
    assert {"gram_self_energy_score.conditional", "gram_mutual_energy_score.conditional"} <= components.keys()
    first = states[:2].clone().requires_grad_()
    second = states[2:4].clone().requires_grad_()
    target = states[4:6]
    actual = _score(family, coords, first, second, target)
    anchor = MultiscaleSpectrumFamily.forward(
        family, first, target, coordinates=coords[None].expand(2, -1, -1),
        context={"conditional_draws": (second,)},
    )
    assert actual.diagnostics["gram_extension"]["anchor_definition"] == "signed_spectral_mean_augmented_v4"
    torch.testing.assert_close(actual.component_results[ANCHOR_PATH].scalar_loss, anchor.scalar_loss)
    blocks = [family._descriptors(family._statistics(item)) for item in (first, second, target)]
    expected_blocks = []
    for name, scale in (("log_band_power", family.gram_self_scale), ("signed_band_cospectrum", family.gram_mutual_scale)):
        denominator = scale * math.sqrt(scale.numel())
        expected_blocks.append(vector_energy_score(
            torch.stack([item[name] / denominator for item in blocks[:2]]),
            blocks[2][name] / denominator,
        )[0])
    torch.testing.assert_close(actual.component_results[SELF_PATH].per_sample_cost, expected_blocks[0])
    torch.testing.assert_close(actual.component_results[MUTUAL_PATH].per_sample_cost, expected_blocks[1])
    torch.testing.assert_close(actual.per_sample_cost, anchor.per_sample_cost + 0.1 * sum(expected_blocks))
    for draw, gradient in zip((first, second), torch.autograd.grad(actual.scalar_loss, (first, second))):
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
        assert gradient.shape == draw.shape
    inputs = tuple(states[i:i+1].clone().requires_grad_() for i in range(4))
    extended = family(inputs[0], states[4:5], coordinates=coords[None],
                      context={"conditional_draws": inputs[1:], "score_all_conditional_draws": True})
    assert extended.component_results[SELF_PATH].diagnostics["conditional_draws"] == 4
    assert all(torch.isfinite(value).all() and value.norm() > 0
               for value in torch.autograd.grad(extended.scalar_loss, inputs))


def test_two_draw_gradient_matches_finite_differences():
    states = _states()
    family, coords, _ = _fitted(_config(), states)
    first, second = (states[i:i+1].clone().requires_grad_() for i in (0, 1))
    assert torch.autograd.gradcheck(
        lambda x, y: _score(family, coords, x, y, states[2:3]).scalar_loss,
        (first, second), eps=1e-6, atol=1e-5, rtol=1e-4, fast_mode=True,
    )


def test_one_field_constant_training_and_coincident_draws_are_finite():
    config = _config(("u",))
    states = torch.zeros(6, 16, 1, dtype=torch.float64)
    family, coords, _ = _fitted(config, states)
    assert family.gram_mutual_scale.numel() == 0
    torch.testing.assert_close(family.gram_self_scale, torch.full_like(family.gram_self_scale, 0.001))
    first = states[:1].clone().requires_grad_()
    second = states[:1].clone().requires_grad_()
    result = _score(family, coords, first, second, states[:1])
    assert result.scalar_loss == 0 and MUTUAL_PATH not in result.component_results
    assert result.diagnostics["components"][SELF_PATH]["weight"] == 0.2
    gradients = torch.autograd.grad(result.scalar_loss, (first, second))
    assert all(torch.isfinite(value).all() for value in gradients)


def test_raw_gram_matches_nonuniform_spatial_measure_and_unit_conversion():
    states = _states()
    config = _config(units="physical_units", floors={"u": 0.01, "v": 0.02})
    family, coords, _ = _fitted(config, states)
    stats = family._statistics(states[:3])
    fields = band_fields(stats["coefficients"], family.basis, family.filters)
    spatial = torch.einsum("n,bsni,bsnj->bsij", family.mass, fields, fields)
    torch.testing.assert_close(stats["gram"], spatial, rtol=1e-10, atol=1e-10)
    assert float(family.mass.max() / family.mass.min()) > 1.0
    factor = torch.tensor([2.0, 3.0], dtype=torch.float64)
    shift = torch.tensor([4.0, -2.0], dtype=torch.float64)
    converted = states * factor + shift
    converted_config = _config(units="physical_units", floors={"u": 0.02, "v": 0.06})
    transformed, _, _ = _fitted(converted_config, converted)
    original = _score(family, coords, states[:1], states[1:2], states[2:3])
    changed = _score(transformed, coords, converted[:1], converted[1:2], converted[2:3])
    torch.testing.assert_close(original.scalar_loss, changed.scalar_loss, rtol=1e-8, atol=1e-8)
    for path in (SELF_PATH, MUTUAL_PATH, ANCHOR_PATH):
        torch.testing.assert_close(original.component_results[path].scalar_loss, changed.component_results[path].scalar_loss, rtol=1e-8, atol=1e-8)


def test_mutual_block_breaks_equal_anchor_signed_gram_witness():
    family, coords, _ = _fitted(_config(), _states())
    # A common one-band-equivalent fixture: each mode has the same filter in
    # both bands, with equal field scales and zero piece means.
    family.filters[:] = 1.0
    family.band_scale[:] = 1.0
    family.power_floor[:] = 1.0
    coeff_target = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    coeff_a = torch.tensor([[0.5, 0.5], [math.sqrt(3) / 2, math.sqrt(3) / 6], [0.0, math.sqrt(6) / 3]], dtype=torch.float64)
    coeff_b = torch.tensor([[0.5, 0.5], [math.sqrt(3) / 2, -math.sqrt(3) / 2], [0.0, 0.0]], dtype=torch.float64)
    def fields(coeff):
        return (family.basis[:, :3] @ coeff)[None]
    target, left, right = fields(coeff_target), fields(coeff_a), fields(coeff_b)
    a = _score(family, coords, left, left, target)
    b = _score(family, coords, right, right, target)
    torch.testing.assert_close(a.component_results[ANCHOR_PATH].scalar_loss, b.component_results[ANCHOR_PATH].scalar_loss, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(a.component_results[SELF_PATH].scalar_loss, b.component_results[SELF_PATH].scalar_loss, atol=1e-10, rtol=1e-10)
    assert b.component_results[MUTUAL_PATH].scalar_loss > a.component_results[MUTUAL_PATH].scalar_loss


def test_artifact_restore_and_candidate_validation_are_strict(tmp_path):
    config = _config()
    family, coords, spec = _fitted(config, _states())
    artifact = family.state_artifact()
    destination = tmp_path / "candidate.pt"
    torch.save(artifact, destination)
    restored = build_coherence_family("multiscale_spectrum", config, spec, FieldNormalizer.identity(2))
    restored.load_state_artifact(torch.load(destination, weights_only=True))
    assert restored.fitted and restored.calibration_record == family.calibration_record
    for name, value in family.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name]), name
    states = _states()
    torch.testing.assert_close(_score(restored, coords, states[:1], states[1:2], states[2:3]).scalar_loss,
                               _score(family, coords, states[:1], states[1:2], states[2:3]).scalar_loss)
    for key in ("gram_extension_definition", "gram_extension_implementation_identity"):
        bad = copy.deepcopy(artifact)
        bad[key] = "different"
        with pytest.raises(ValueError, match="gram_extension"):
            restored.load_state_artifact(bad)
    bad = copy.deepcopy(artifact)
    bad["state_dict"]["gram_self_scale"][0] *= 2
    with pytest.raises(ValueError, match="scales differ"):
        restored.load_state_artifact(bad)
    bad_config = copy.deepcopy(config)
    bad_config["gram_extension"]["unknown"] = 1
    with pytest.raises(ValueError, match="unknown gram_extension"):
        parse_gram_extension_config(bad_config, spec.field_names)
    bad_config = copy.deepcopy(config)
    bad_config["gram_extension"]["weight"] = 0
    with pytest.raises(ValueError, match="positive"):
        parse_gram_extension_config(bad_config, spec.field_names)
    anchor_config = copy.deepcopy(config)
    anchor_config.pop("gram_extension")
    assert type(build_coherence_family("multiscale_spectrum", anchor_config, spec, FieldNormalizer.identity(2))) is MultiscaleSpectrumFamily


def test_set_evaluator_declares_candidate_and_anchor_identity(tmp_path):
    from phycoflow_reconstruction.evaluation.multiscale_set import MultiscaleSetAccumulator

    family, coords, _ = _fitted(_config(), _states())
    accumulator = MultiscaleSetAccumulator(
        family=family, artifact_path=tmp_path / "candidate.pt", artifact_sha256="0" * 64,
        query_point_count=len(coords), query_seed=7, draw_count=4,
    )
    states = _states()
    accumulator.update(states[:1], states[1:2], coords[None], "sample",
                       conditional_draws=tuple(states[i:i+1] for i in (2, 3, 4)))
    output = accumulator.finalize(tmp_path, split="validation", checkpoint_label="test",
                                  run_label="test", scale="linear")
    report = json.loads(Path(output["report"]).read_text())
    assert report["score_definition_version"] == GRAM_DEFINITION
    assert report["anchor_score_definition_version"] == "signed_spectral_mean_augmented_v4"
    assert SELF_PATH in report["component_means"] and MUTUAL_PATH in report["component_means"]
