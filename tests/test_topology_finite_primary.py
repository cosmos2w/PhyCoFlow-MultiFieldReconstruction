"""Scientific contract tests for the new paired finite-primary Shape score."""

import math
from copy import deepcopy

import pytest
import torch

pytest.importorskip("gudhi")

from phycoflow_reconstruction.coherence.families.topology.family import TopologyFamily
from phycoflow_reconstruction.coherence.families.topology.filtration_sampling import (
    TopologyFiltrationSampler,
)
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer


def config():
    return {
        "strategy": "cubical_persistence",
        "target_use": "paired_supervised",
        "units": "model_units",
        "fields": ["CO", "T"],
        "geometry": {"grid_shape": [4, 4]},
        "filtration": {
            "smoothing_sigma": 0,
            "directions": ["sublevel", "superlevel"],
            "dimensions": [0, 1],
        },
        "persistence": {
            "definition": "finite_primary_v1",
            "projections": 32,
            "scale_floor": 1e-4,
            "calibration": {"finite_floor": 1e-3, "essential_floor": 2e-3},
        },
        "components": {
            "self": {"enabled": True},
            "mutual": {"enabled": True, "groups": [["CO", "T"]], "lines": 3},
        },
    }


def family(settings=None):
    return TopologyFamily(
        settings or config(),
        DataSpec(("CO", "T"), ("1", "1"), 2, (16,)),
        FieldNormalizer.identity(2),
    )


def coords(batch=1):
    y, x = torch.meshgrid(torch.arange(4.0), torch.arange(4.0), indexing="ij")
    return torch.stack([x.flatten(), y.flatten()], dim=-1)[None].expand(batch, -1, -1)


def provenance(batch=1):
    return {
        "split": "train",
        "sample_ids": [f"train:{i}" for i in range(batch)],
        "source_checkpoint_sha256": "a" * 64,
    }


def witness():
    a = torch.tensor([[1, 1, -1, -1]] * 4, dtype=torch.float32)
    ref_b = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1], [-1, -1, -1, -1], [-1, -1, -1, -1]])
    pred_b = torch.tensor([[1, 1, 1, 1], [1, 1, -1, -1], [-1, 1, 1, -1], [-1, -1, -1, -1]])
    return torch.stack((a, pred_b), -1).reshape(1, 16, 2), torch.stack((a, ref_b), -1).reshape(
        1, 16, 2
    )


def test_opposing_sign_witness_raw_leaves_and_exact_bank():
    pred, target = witness()
    model = family()
    with pytest.raises(ValueError, match="calibrated"):
        model(pred, target, coordinates=coords())
    model.fit_calibration(pred, target, coordinates=coords(), provenance=provenance())
    output = model(pred, target, coordinates=coords())
    assert output.diagnostics["scalar_filtrations"] == 16
    assert model.version == model.spec.version == "4"
    weights = model.spatial_objective.line_weights
    expected = torch.tensor(
        [1 / math.sqrt(2), 1 / math.sqrt(5), 1 / math.sqrt(5)], dtype=weights.dtype
    )
    torch.testing.assert_close(weights, expected / expected.sum())
    for sign in ("positive_positive", "negative_negative", "negative_positive"):
        name = f"topology.mutual.CO+T.{sign}.line0.h0.finite"
        assert output.component_results[name].scalar_loss == 0
    leaf = output.component_results["topology.mutual.CO+T.positive_negative.line0.h0.finite"]
    assert leaf.scalar_loss.item() == pytest.approx(0.0796094385, abs=2e-7)
    for sign in ("positive", "negative"):
        for field in ("CO", "T"):
            for dimension in (0, 1):
                assert (
                    output.component_results[
                        f"topology.self.{field}.{sign}.h{dimension}.finite"
                    ].scalar_loss
                    == 0
                )
    identical = model(target, target, coordinates=coords())
    assert identical.scalar_loss == 0
    assert all(term.scalar_loss == 0 for term in identical.component_results.values())


def test_calibration_floors_structural_absence_and_resume():
    pred, target = witness()
    model = family()
    record = model.fit_calibration(target, target, coordinates=coords(), provenance=provenance())
    assert not model.needs_source_calibration
    scales = model.spatial_objective.leaf_scales
    torch.testing.assert_close(scales[..., 0], torch.full_like(scales[..., 0], 1e-3))
    torch.testing.assert_close(scales[..., 1], torch.full_like(scales[..., 1], 2e-3))
    assert not any(row[1] for row in record["essential_present"])  # H1 absent for a rectangle.
    output = model(pred, target, coordinates=coords())
    assert (
        output.component_results["topology.self.CO.positive.h1.essential"].diagnostics[
            "structurally_present"
        ]
        is False
    )
    with pytest.raises(ValueError, match="already frozen"):
        model.fit_calibration(pred, target, coordinates=coords(), provenance=provenance())
    restored = family()
    restored.load_state_artifact(deepcopy(model.state_artifact()))
    assert not restored.needs_source_calibration
    assert restored.spatial_objective.calibration_record == record
    torch.testing.assert_close(
        restored(pred, target, coordinates=coords()).scalar_loss, output.scalar_loss, rtol=0, atol=0
    )
    artifact = deepcopy(model.state_artifact())
    artifact["config"]["persistence"]["calibration"]["finite_floor"] *= 2
    with pytest.raises(ValueError, match="config mismatch"):
        family().load_state_artifact(artifact)


def test_gradients_and_reference_cache():
    generator = torch.Generator().manual_seed(715)
    target = torch.randn(2, 16, 2, generator=generator)
    source = target + torch.randn(2, 16, 2, generator=generator) * 0.3
    model = family()
    model.fit_calibration(source, target, coordinates=coords(2), provenance=provenance(2))
    pred = (source + torch.randn(2, 16, 2, generator=generator) * 0.05).requires_grad_()
    context = {"persistence_cache": {}}
    result = model(pred, target, coordinates=coords(2), context=context)
    gradient = torch.autograd.grad(result.scalar_loss, pred)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    # Keep the same object identity as required by the cache contract.
    points = coords(2)
    context = {"persistence_cache": {}}
    first = model(pred, target, coordinates=points, context=context)
    second = model(pred, target, coordinates=points, context=context)
    torch.testing.assert_close(first.scalar_loss, second.scalar_loss, rtol=0, atol=0)
    with pytest.raises(ValueError, match="target/coordinates changed"):
        model(pred, target.clone(), coordinates=points, context=context)


def test_calibration_validation_and_sampler_rejection():
    pred, target = witness()
    with pytest.raises(ValueError, match="TRAIN provenance"):
        family().fit_calibration(pred, target, coordinates=coords(), provenance={"split": "test"})
    with pytest.raises(ValueError, match="sample_ids"):
        family().fit_calibration(
            pred,
            target,
            coordinates=coords(),
            provenance={"split": "train", "sample_ids": [], "source_checkpoint_sha256": "a" * 64},
        )
    with pytest.raises(ValueError, match="sha256"):
        family().fit_calibration(
            pred,
            target,
            coordinates=coords(),
            provenance={"split": "train", "sample_ids": ["x"], "source_checkpoint_sha256": "bad"},
        )
    for strategy in ("full", "p3", "s2", "polarity"):
        with pytest.raises(ValueError, match="direct full-bank"):
            TopologyFiltrationSampler(family().spatial_objective, strategy=strategy)
    for key in ("finite_floor", "essential_floor"):
        bad = config()
        bad["persistence"]["calibration"][key] = 0
        with pytest.raises(ValueError, match="floors"):
            family(bad)
    bad = config()
    bad["components"]["mutual"]["lines"] = 1
    with pytest.raises(ValueError, match="three lines"):
        family(bad)


def test_finite_and_essential_contribute_as_separate_calibrated_leaves():
    model = family()
    target = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1).expand(-1, -1, 2).clone()
    pred = target.clone()
    pred[:, :, 0] += 1.0
    model.fit_calibration(pred, target, coordinates=coords(), provenance=provenance())
    output = model(pred, target, coordinates=coords())
    finite = output.component_results["topology.self.CO.positive.h0.finite"]
    essential = output.component_results["topology.self.CO.positive.h0.essential"]
    assert finite.scalar_loss == 0 and essential.scalar_loss > 0
    assert float(model.spatial_objective.leaf_scales[0, 0, 0]) == pytest.approx(1e-3)
    assert float(model.spatial_objective.leaf_scales[0, 0, 1]) == pytest.approx(
        essential.scalar_loss.item()
    )


def test_reflection_invariance_and_directional_gradient_away_from_ties():
    generator = torch.Generator().manual_seed(4141)
    target = torch.randn(1, 2, 4, 4, generator=generator, dtype=torch.float64)
    prediction = target + 0.2 * torch.randn(target.shape, generator=generator, dtype=torch.float64)
    objective = family().spatial_objective.double()
    objective.fit_calibration(prediction, target, provenance=provenance())
    reflected = objective(target.flip(-1), target)
    assert reflected.scalar_loss.item() == pytest.approx(0.0, abs=1e-12)
    prediction.requires_grad_()
    loss = objective(prediction, target).scalar_loss
    direction = torch.randn(prediction.shape, generator=generator, dtype=torch.float64)
    gradient = torch.autograd.grad(loss, prediction)[0]
    epsilon = 1e-6
    numeric = (
        objective(prediction + epsilon * direction, target).scalar_loss
        - objective(prediction - epsilon * direction, target).scalar_loss
    ) / (2 * epsilon)
    torch.testing.assert_close((gradient * direction).sum(), numeric, atol=2e-7, rtol=2e-5)


def test_masked_hole_and_periodic_essential_classes():
    from phycoflow_reconstruction.coherence.families.topology.persistence import cubical_diagrams

    mask = torch.ones(4, 4, dtype=torch.bool)
    mask[1:3, 1:3] = False
    target = torch.randn(1, 2, 4, 4, generator=torch.Generator().manual_seed(71))
    prediction = (
        target + 0.2 * torch.randn(target.shape, generator=torch.Generator().manual_seed(72))
    ).requires_grad_()
    objective = family().spatial_objective
    objective.fit_calibration(prediction, target, valid_mask=mask, provenance=provenance())
    output = objective(prediction, target, valid_mask=mask)
    assert output.diagnostics["point_normalization"] == 12
    assert output.component_results["topology.self.CO.positive.h1.essential"].diagnostics[
        "structurally_present"
    ]
    grad = torch.autograd.grad(output.scalar_loss, prediction)[0]
    assert torch.isfinite(grad).all() and grad[..., mask].abs().sum() > 0
    assert torch.count_nonzero(grad[..., ~mask]) == 0
    diagrams = cubical_diagrams(target[:, 0], periodic=False, valid_mask=mask)[0]
    assert len(diagrams[0].essential) == 1 and len(diagrams[1].essential) == 1
    disconnected = torch.ones(4, 4, dtype=torch.bool)
    disconnected[:, 1] = False
    diagrams = cubical_diagrams(target[:, 0], periodic=False, valid_mask=disconnected)[0]
    assert len(diagrams[0].essential) == 2 and len(diagrams[1].essential) == 0
    with pytest.raises(ValueError, match="masked periodic"):
        cubical_diagrams(target[:, 0], periodic=True, valid_mask=mask)
    settings = config()
    settings["geometry"]["periodic"] = True
    periodic = family(settings)
    reference = target.permute(0, 2, 3, 1).reshape(1, 16, 2)
    predicted = prediction.detach().permute(0, 2, 3, 1).reshape(1, 16, 2)
    periodic.fit_calibration(predicted, reference, coordinates=coords(), provenance=provenance())
    result = periodic(predicted, reference, coordinates=coords())
    assert result.component_results["topology.self.CO.positive.h1.essential"].diagnostics[
        "structurally_present"
    ]
    periodic_diagrams = cubical_diagrams(target[:, 0], periodic=True)[0]
    assert len(periodic_diagrams[1].essential) == 2


def test_calibration_record_must_match_frozen_tensor_state():
    pred, target = witness()
    model = family()
    model.fit_calibration(pred, target, coordinates=coords(), provenance=provenance())
    artifact = deepcopy(model.state_artifact())
    artifact["calibration"]["scales"][0][0][0] *= 2
    with pytest.raises(ValueError, match="contents mismatch"):
        family().load_state_artifact(artifact)


def test_component_aggregation_matches_declared_leaf_formula():
    gen = torch.Generator().manual_seed(414)
    target = torch.randn(2, 16, 2, generator=gen)
    pred = target + 0.3 * torch.randn(target.shape, generator=gen)
    model = family()
    model.fit_calibration(pred, target, coordinates=coords(2), provenance=provenance(2))
    result = model(pred, target, coordinates=coords(2))
    terms = result.component_results

    def normalized(prefix):
        finite = terms[prefix + ".finite"]
        essential = terms[prefix + ".essential"]
        value = 0.9 * finite.per_sample_cost / finite.diagnostics["scale"]
        if essential.diagnostics["structurally_present"]:
            value = value + 0.1 * essential.per_sample_cost / essential.diagnostics["scale"]
        return value

    self_values = [
        normalized(f"topology.self.{field}.{sign}.h{dimension}")
        for field in ("CO", "T")
        for sign in ("positive", "negative")
        for dimension in (0, 1)
    ]
    expected_self = torch.stack(self_values).mean(0)
    weights = result.diagnostics["line_weights"]
    pair_values = [
        sum(
            weights[line] * normalized(f"topology.mutual.CO+T.{sign}.line{line}.h{dimension}")
            for line in range(3)
        )
        for sign in (
            "positive_positive",
            "negative_negative",
            "positive_negative",
            "negative_positive",
        )
        for dimension in (0, 1)
    ]
    expected_pair = torch.stack(pair_values).mean(0)
    torch.testing.assert_close(terms["topology.self.persistence"].per_sample_cost, expected_self)
    torch.testing.assert_close(terms["topology.mutual.persistence"].per_sample_cost, expected_pair)
    torch.testing.assert_close(result.per_sample_cost, expected_self + expected_pair)


def test_constant_target_uses_floors_without_discarding_new_discrepancies():
    target = torch.zeros(1, 16, 2)
    model = family()
    model.fit_calibration(target, target, coordinates=coords(), provenance=provenance())
    prediction = torch.full_like(target, 0.1, requires_grad=True)
    result = model(prediction, target, coordinates=coords())
    essential = result.component_results["topology.self.CO.positive.h0.essential"]
    assert essential.scalar_loss > 0
    assert essential.diagnostics["scale"] == pytest.approx(2e-3)
    grad = torch.autograd.grad(result.scalar_loss, prediction)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_project_validator_rejects_invalid_new_definition_before_building(tmp_path):
    from helpers.coherence import _post_config

    from phycoflow_reconstruction.config.validate import validate_config

    project = _post_config(tmp_path / "fixture.h5", tmp_path / "source")
    project["dataset"]["field_names"] = ["CO", "T"]
    project["observations"]["fields"] = {"CO": {"count": 4}}
    project["coherence"]["families"] = {"topology": config()}
    project["coherence"]["compute_budget"]["query_policy"] = "fixed_shared"
    validate_config(project)
    mutations = [
        (lambda t: t["persistence"]["calibration"].update(finite_floor=0), "floors"),
        (lambda t: t["persistence"]["calibration"].update(essential_floor=-1), "floors"),
        (lambda t: t["components"]["mutual"].update(lines=1), "three lines"),
        (lambda t: t["components"]["mutual"].update(seed=5), "three lines"),
        (lambda t: t["filtration"].update(dimensions=[0]), "H0/H1"),
        (lambda t: t["filtration"].update(directions=["sublevel"]), "both unary directions"),
        (lambda t: t["persistence"].update(essential_weight=0.1), "split"),
    ]
    for mutation, message in mutations:
        invalid = deepcopy(project)
        mutation(invalid["coherence"]["families"]["topology"])
        with pytest.raises(ValueError, match=message):
            validate_config(invalid)
    for key in ("execution_strategy", "lpc_ordered"):
        invalid = deepcopy(project)
        invalid["optimization"][key] = {}
        with pytest.raises(ValueError, match="full direct"):
            validate_config(invalid)


def test_fixed_domain_requires_exact_essential_counts_not_only_presence():
    generator = torch.Generator().manual_seed(902)
    target = torch.randn(1, 2, 4, 4, generator=generator)
    prediction = target + 0.1 * torch.randn(target.shape, generator=generator)
    connected = torch.ones(4, 4, dtype=torch.bool)
    disconnected = connected.clone()
    disconnected[:, 1] = False  # H0 changes from one to two; H1 remains absent.
    objective = family().spatial_objective
    record = objective.fit_calibration(
        prediction, target, valid_mask=connected, provenance=provenance()
    )
    assert all(counts == [1, 0] for counts in record["essential_counts"])
    _, presence, _ = objective.raw_leaves(prediction, target, valid_mask=disconnected)
    assert torch.equal(presence, objective.essential_present)  # Presence alone cannot detect it.
    with pytest.raises(ValueError, match="essential class counts differ"):
        objective(prediction, target, valid_mask=disconnected)


def test_calibration_restore_rejects_essential_count_mismatch():
    pred, target = witness()
    model = family()
    model.fit_calibration(pred, target, coordinates=coords(), provenance=provenance())
    artifact = deepcopy(model.state_artifact())
    artifact["calibration"]["essential_counts"][0][0] = 2
    with pytest.raises(ValueError, match="contents mismatch"):
        family().load_state_artifact(artifact)
