"""A common frozen yardstick scores saved draws without refitting artifacts."""


import numpy as np
import pytest
import torch

from phycoflow_reconstruction.coherence.families.multiscale_spectrum import MultiscaleSpectrumFamily
from phycoflow_reconstruction.contracts import DataSpec, FamilyResult, ObservationBatch, TermResult
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.evaluation.canonical_coherence import (
    FrozenCanonicalEvaluator,
    assert_comparable,
    save_canonical_score,
)


class FrozenFixture(torch.nn.Module):
    family_name = "fixture"
    version = "1"
    target_use = "paired_supervised"

    def __init__(self, *, grouped=False):
        super().__init__()
        self.register_buffer("frozen_scale", torch.tensor(2.0))
        self.grouped = grouped
        self.calls = []

    def state_artifact(self):
        return {"family": self.family_name, "version": self.version,
                "config": {"grouped": self.grouped}, "state_dict": self.state_dict()}

    def forward(self, prediction, target, *, coordinates, context):
        self.calls.append((prediction.shape[0], context.get("score_all_conditional_draws", False)))
        error = ((prediction - target) ** 2).mean(dim=(1, 2)) / self.frozen_scale
        if self.grouped:
            score = error.mean()
            return FamilyResult({"fixture.leaf": TermResult(None, score, diagnostics={"batch": len(error)})}, None, score, {})
        return FamilyResult({"fixture.leaf": TermResult(error, error.mean(), diagnostics={})}, error, error.mean(), {})


def _batch(states=4, points=6):
    coordinates = torch.linspace(0, 1, points, dtype=torch.float64)[None, :, None].expand(states, -1, -1).clone()
    target = torch.arange(states * points * 2, dtype=torch.float64).reshape(states, points, 2) / 10
    return ObservationBatch(obs_coords=coordinates[:, :1].clone(), obs_values=target[:, :1, :1].clone(),
        obs_field_ids=torch.zeros(states, 1, dtype=torch.long), obs_valid_mask=torch.ones(states, 1, dtype=torch.bool),
        query_coords=coordinates, query_valid_mask=torch.ones(states, points, dtype=torch.bool),
        target_fields=target, sample_ids=tuple(f"s{i}" for i in range(states)), metadata={"query_seed": 100047})


def _sampler():
    return {"path": "external_differentiable_reconstruction", "steps": 2, "solver": "euler",
            "observation_config": {"mode": "endpoint_smooth", "final_clamp": True},
            "execution": {"query_chunk": 32}, "effective_weights": "live", "draw_seeds": [1, 2, 3, 4]}


def test_frozen_paired_and_group_scoring_roundtrips_without_refitting(tmp_path):
    batch = _batch()
    members = torch.stack([batch.target_fields + value for value in (1., 2., 3., 4.)])
    paired, grouped = FrozenFixture(), FrozenFixture(grouped=True)
    evaluator = FrozenCanonicalEvaluator({"legacy_state": paired, "legacy_scale": grouped},
        {"legacy_state": "paired", "legacy_scale": "ensemble_group"}, field_names=("u", "v"),
        query_points=6, group_size=2)
    arrays, report = evaluator.evaluate(members, batch, normalizer=FieldNormalizer.identity(2),
        sampler=_sampler(), model_identity={"checkpoint_sha256": "source"})
    assert paired.calls == [(4, False)] * 4
    assert grouped.calls == [(2, False)] * 8
    np.testing.assert_allclose(arrays["field_mse_per_member"][:, 0, 0], [1., 4., 9., 16.])
    np.testing.assert_allclose(arrays["field_mse_ensemble_mean"][:, 0], 6.25)
    assert arrays["family/legacy_scale/total_by_draw_group"].shape == (4, 2)
    np.testing.assert_allclose(arrays["family/legacy_state/leaf/fixture.leaf"][:, 0], [.5, 2., 4.5, 8.])
    assert report["condition_groups"] == [["s0", "s1"], ["s2", "s3"]]
    payload, manifest = save_canonical_score(tmp_path, arrays, report)
    with np.load(payload, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["family/legacy_scale/total_by_draw_group"], arrays["family/legacy_scale/total_by_draw_group"])
    assert manifest.is_file()
    _, candidate = evaluator.evaluate(members + .1, batch, normalizer=FieldNormalizer.identity(2),
        sampler=_sampler(), model_identity={"checkpoint_sha256": "candidate"})
    assert_comparable(report, candidate)
    with pytest.raises(ValueError, match="panel, sampler"):
        evaluator.evaluate(members, batch, normalizer=FieldNormalizer.identity(2),
            sampler={**_sampler(), "solver": "heun"}, model_identity={"checkpoint_sha256": "bad"})
    paired.frozen_scale.add_(1.)
    with pytest.raises(ValueError, match="artifact changed"):
        evaluator.evaluate(members, batch, normalizer=FieldNormalizer.identity(2),
            sampler=_sampler(), model_identity={"checkpoint_sha256": "bad"})


def test_panel_and_sampler_require_explicit_identity():
    batch = _batch()
    family = FrozenFixture()
    evaluator = FrozenCanonicalEvaluator({"state": family}, {"state": "paired"},
        field_names=("u", "v"), query_points=6, group_size=2)
    members = torch.stack([batch.target_fields] * 4)
    with pytest.raises(ValueError, match="live weights"):
        evaluator.evaluate(members, batch, normalizer=FieldNormalizer.identity(2),
            sampler={**_sampler(), "effective_weights": "ema"}, model_identity={"checkpoint": "x"})
    with pytest.raises(ValueError, match="reference IDs"):
        evaluator.evaluate(members, batch, normalizer=FieldNormalizer.identity(2), sampler=_sampler(),
            model_identity={"checkpoint": "x"}, reference_ids=tuple(reversed(batch.sample_ids)))
    batch.query_valid_mask[0, 0] = False
    with pytest.raises(ValueError, match="only valid query points"):
        evaluator.evaluate(members, batch, normalizer=FieldNormalizer.identity(2), sampler=_sampler(),
            model_identity={"checkpoint": "x"})


def test_real_v4_family_scores_all_four_draws_with_training_artifact():
    batch = _batch(points=12)
    normalizer = FieldNormalizer.identity(2)
    settings = {"fields": ["u", "v"], "geometry": {"operator": "fem_p1", "coordinate_scaling": "none"},
        "bands": {"count": 2}, "numerics": {"dtype": "float64"},
        "calibration": {"field_unit_floor": .01},
        "objective": {"kind": "signed_band_energy_score", "conditional_draws": 2,
                      "almost_fair_alpha": 1.0, "joint_energy_score_weight": .25,
                      "score_definition_version": "signed_spectral_mean_augmented_v4"}}
    family = MultiscaleSpectrumFamily(settings, DataSpec(("u", "v"), ("1", "1"), 1, (12,)), normalizer)
    train = torch.randn(8, 12, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(99))
    family.fit_reference_tensors(batch.query_coords[0], train, sample_ids=[str(i) for i in range(8)],
                                 provenance={"split": "train"})
    members = torch.stack([batch.target_fields + v for v in (.1, .4, .9, 1.5)])
    evaluator = FrozenCanonicalEvaluator({"next_scale": family}, {"next_scale": "conditional"},
        field_names=("u", "v"), query_points=12, group_size=2)
    arrays, report = evaluator.evaluate(members, batch, normalizer=normalizer, sampler=_sampler(),
                                         model_identity={"checkpoint_sha256": "example"})
    direct = family(members[0], batch.target_fields, coordinates=batch.query_coords,
        context={"conditional_draws": tuple(members[1:]), "score_all_conditional_draws": True})
    key = "multiscale_spectrum.joint_spectral_mean_energy_score.conditional"
    np.testing.assert_allclose(arrays[f"family/next_scale/leaf/{key}"], direct.component_results[key].per_sample_cost.detach().numpy())
    assert report["families"]["next_scale"]["result_diagnostics"][0]["conditional_draws"] == 4
    assert family.conditional_draws == 2


def test_real_legacy_scale_uses_exact_complete_condition_groups():
    from phycoflow_reconstruction.coherence.families.cross_spectrum.family import (
        CrossSpectrumFamily,
    )

    batch = _batch(states=6, points=8)
    normalizer = FieldNormalizer.identity(2)
    config = {"target_use": "paired_supervised", "fields": ["u", "v"],
              "graph": {"k_neighbors": 2, "num_modes": 4, "bands": ["low", "high"]}}
    family = CrossSpectrumFamily(config, DataSpec(("u", "v"), ("1", "1"), 1, (8,)), normalizer)
    family._basis(batch.query_coords, batch.target_fields.dtype)
    generator = torch.Generator().manual_seed(51)
    members = torch.stack([batch.target_fields + torch.randn(batch.target_fields.shape, generator=generator,
                            dtype=torch.float64) * .1 for _ in range(4)])
    evaluator = FrozenCanonicalEvaluator({"legacy_scale": family}, {"legacy_scale": "ensemble_group"},
        field_names=("u", "v"), query_points=8, group_size=3)
    arrays, report = evaluator.evaluate(members, batch, normalizer=normalizer, sampler=_sampler(),
                                       model_identity={"checkpoint_sha256": "example"})
    direct = family(members[0, :3], batch.target_fields[:3], coordinates=batch.query_coords[:3])
    assert arrays["family/legacy_scale/total_by_draw_group"][0, 0] == pytest.approx(float(direct.scalar_loss))
    assert report["condition_groups"] == [["s0", "s1", "s2"], ["s3", "s4", "s5"]]


def test_real_finite_primary_shape_retains_raw_leaves():
    pytest.importorskip("gudhi")
    from phycoflow_reconstruction.coherence.families.topology.family import TopologyFamily

    y, x = torch.meshgrid(torch.arange(4.), torch.arange(4.), indexing="ij")
    coordinates = torch.stack((x.flatten(), y.flatten()), dim=-1)[None]
    target = torch.randn(1, 16, 2, generator=torch.Generator().manual_seed(7))
    batch = ObservationBatch(obs_coords=coordinates[:, :1], obs_values=target[:, :1, :1],
        obs_field_ids=torch.zeros(1, 1, dtype=torch.long), obs_valid_mask=torch.ones(1, 1, dtype=torch.bool),
        query_coords=coordinates, query_valid_mask=torch.ones(1, 16, dtype=torch.bool), target_fields=target,
        sample_ids=("validation:1",), metadata={"query_seed": 100047})
    config = {"strategy": "cubical_persistence", "target_use": "paired_supervised", "units": "model_units",
        "fields": ["u", "v"], "geometry": {"grid_shape": [4, 4]},
        "filtration": {"smoothing_sigma": 0, "directions": ["sublevel", "superlevel"], "dimensions": [0, 1]},
        "persistence": {"definition": "finite_primary_v1", "projections": 8, "scale_floor": 1e-4,
                        "calibration": {"finite_floor": 1e-3, "essential_floor": 2e-3}},
        "components": {"self": {"enabled": True}, "mutual": {"enabled": True, "groups": [["u", "v"]], "lines": 3}}}
    normalizer = FieldNormalizer.identity(2)
    family = TopologyFamily(config, DataSpec(("u", "v"), ("1", "1"), 2, (16,)), normalizer)
    family.fit_calibration(target + .2, target, coordinates=coordinates,
        provenance={"split": "train", "sample_ids": ["train:0"], "source_checkpoint_sha256": "a" * 64})
    evaluator = FrozenCanonicalEvaluator({"next_shape": family}, {"next_shape": "paired"},
        field_names=("u", "v"), query_points=16, group_size=1)
    members = torch.stack([target + value for value in (.0, .1, .2, .3)])
    arrays, report = evaluator.evaluate(members, batch, normalizer=normalizer, sampler=_sampler(),
                                       model_identity={"checkpoint_sha256": "example"})
    key = "family/next_shape/leaf/topology.mutual.u+v.positive_negative.line0.h0.finite"
    assert arrays[key].shape == (4, 1)
    assert report["families"]["next_shape"]["mode"] == "paired"
