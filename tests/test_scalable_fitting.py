import copy
import random
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

from phycoflow_reconstruction.training.scalable_fitting import (
    fit_scalable_artifacts,
    validate_scalable_artifacts,
)


def test_common_core_rebase_preserves_fitted_body_and_rejects_stale_support():
    args = fixture(3)
    parent = fit_scalable_artifacts(**args)
    candidate = copy.deepcopy(args)
    candidate["provenance"]["config_sha256"] = "e" * 64
    candidate["reuse_core"] = parent
    rebased = fit_scalable_artifacts(**candidate)
    from phycoflow_reconstruction.training.scalable_prefit import _core_body_digest

    assert _core_body_digest(rebased) == _core_body_digest(parent)
    assert rebased["provenance"] == candidate["provenance"]
    assert parent["provenance"]["config_sha256"] == "d" * 64
    candidate["provenance"]["support_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="support"):
        fit_scalable_artifacts(**candidate)


def test_frozen_adaptive_bank_reuses_exact_source_and_common_core(tmp_path, monkeypatch):
    import json

    from phycoflow_reconstruction.coherence import adaptive_fitting
    from phycoflow_reconstruction.training.run_store import file_sha256
    from phycoflow_reconstruction.training.scalable_fitting import _adaptive_input_binding

    args = fixture(3)
    core = fit_scalable_artifacts(**args)
    source, reference, coords = args['source'], args['reference'], args['coordinates']
    weights = torch.ones(reference.shape[:2], dtype=reference.dtype)
    def panel():
        yield {'ids': ['a', 'b'], 'members': source.transpose(0, 1), 'reference': reference}
    bank = adaptive_fitting.fit_train_bank(
        panel, field_order=args['provenance']['field_order'], masses=weights[0],
        region_masses=weights[:1], filter_fn=lambda x: x.unsqueeze(-3),
        anchor_fn=lambda x, y, m: ((x-y).square() * m[None, None, :, None]).sum(),
        source_identity=args['provenance']['source_checkpoint_sha256'],
        support_identity=args['provenance']['support_sha256'], units='model',
    )
    bank['source_fit_binding_sha256'] = _adaptive_input_binding(
        source=source, reference=reference, coordinates=coords, source_weights=weights,
        reference_weights=weights, identity=args['provenance'], core=core)
    bank.pop('sha256')
    bank['sha256'] = adaptive_fitting._digest(bank)
    path = tmp_path / 'adaptive_bank.json'
    path.write_text(json.dumps(bank, sort_keys=True))
    settings = {'bank_artifact': {'path': str(path), 'sha256': file_sha256(path)}}
    candidate = {**args, 'reuse_core': core, 'adaptive_settings': settings,
                 'provenance': {**args['provenance'], 'config_sha256': 'e' * 64}}
    monkeypatch.setattr(adaptive_fitting, 'fit_train_bank', lambda *a, **k: pytest.fail('rediscovery'))
    restored = fit_scalable_artifacts(**candidate)
    assert restored['adaptive_bank'] == bank
    assert restored['units'] == core['units']
    changed = source.clone()
    changed[0, 0, 0, 0] += .01
    with pytest.raises(ValueError, match='SOURCE tensors'):
        fit_scalable_artifacts(**{**candidate, 'source': changed})
    for key in ('support_sha256', 'source_checkpoint_sha256', 'normalizer_sha256'):
        with pytest.raises(ValueError, match='common core'):
            fit_scalable_artifacts(**{**candidate, 'provenance': {**candidate['provenance'], key: 'f' * 64}})


def test_common_source_panel_requires_exact_stream_and_frozen_families(tmp_path):
    import json

    from phycoflow_reconstruction.contracts import FieldSample
    from phycoflow_reconstruction.data.sensor_protocols import (
        SensorProtocol,
        build_observation_batch,
    )
    from phycoflow_reconstruction.training.coherence_sampling import _conditional_draw_seed
    from phycoflow_reconstruction.training.run_store import file_sha256
    from phycoflow_reconstruction.training.scalable_fitting import _digest
    from phycoflow_reconstruction.training.scalable_prefit import _load_common_core

    args = fixture(3)
    names = ("f0", "f1", "f2")
    samples = [FieldSample(values=args["reference"][i], coordinates=args["coordinates"][i], coordinates_raw=args["coordinates"][i], time=torch.tensor(0.), trajectory_id=f"s{i}", time_index=0, conditions=torch.empty(0), field_names=names, logical_shape=(3, 3)) for i in range(2)]
    protocol = SensorProtocol(field_counts={"f0": 2}, seed=7)
    query = torch.arange(9)
    batch = build_observation_batch(samples, protocol, query_indices=query)
    batch.metadata.setdefault("sample_context", {}).setdefault("auxiliary", {})["physical_weights"] = torch.ones(2, 9)
    args["provenance"]["sample_ids"] = list(batch.sample_ids)
    args["provenance"]["field_order"] = list(names)
    parent = fit_scalable_artifacts(**args)
    model = torch.nn.Linear(1, 1)
    model.capabilities = SimpleNamespace(structured_grid_required=False)
    model.source_weight_view_sha256 = "ema-view"
    coupling = [{"sample_ids": list(batch.sample_ids), "observation_sha256": _digest([batch.obs_indices, batch.obs_values, batch.obs_field_ids, batch.obs_valid_mask]), "member_seeds": [_conditional_draw_seed(7, j) for j in range(4)]}]
    panel = {"schema": "car_source_panel_v1", "provenance": parent["provenance"], "members": torch.cat((args["source"], args["source"] + 0.3), dim=1), "reference": batch.target_fields, "coordinates": batch.query_coords, "masses": torch.ones(2, 9), "query_indices": batch.metadata["query_indices"], "sample_indices": [0, 1], "sample_ids": list(batch.sample_ids), "observation_manifest": None, "coupling": coupling, "source_weight_view": "ema_materialized", "source_weight_view_sha256": "ema-view", "parameter_buffer_sha256": _digest(model.state_dict()), "members_count": 4, "sensor_count": None}
    artifact_dir = tmp_path / "parent" / "artifacts"
    artifact_dir.mkdir(parents=True)
    core_path, panel_path = artifact_dir / "scalable_coherence.pt", artifact_dir / "source_panel.pt"
    torch.save(parent, core_path)
    torch.save(panel, panel_path)
    hashes, frozen = {}, {}
    for name in ("global_distribution", "multiscale_spectrum", "topology"):
        path = artifact_dir / f"{name}_family.pt"
        path.write_text(name)
        hashes[name] = file_sha256(path)
        frozen[name] = {"path": str(path), "sha256": hashes[name]}
    (artifact_dir.parent / "run_manifest.json").write_text(json.dumps({"coherence_family_state_sha256s": hashes}))
    declaration = {"path": str(core_path), "sha256": file_sha256(core_path), "source_panel_path": str(panel_path), "source_panel_sha256": file_sha256(panel_path)}
    config = {"source": {"weight_view": "ema_materialized"}, "rollout": {"steps": 4, "solver": "euler"}, "observations": {}, "observation_consistency": {"mode": "hard"}, "runtime": {"seed": 42}, "coherence": {"families": {name: {} for name in hashes}, "compute_budget": {"point_count": 9}, "frozen_family_artifacts": frozen}}
    import yaml
    (artifact_dir.parent / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    new_identity = {**parent["provenance"], "config_sha256": "e" * 64}
    kwargs = {"provenance": new_identity, "model": model, "config": config,
              "batch": batch, "indexes": [0, 1], "samples": samples, "query": query,
              "protocol": protocol, "manifest_indices": None, "seed": 7,
              "members": 4, "sensor_count": None, "microbatch": 2}
    _, reused, _ = _load_common_core(declaration, **kwargs)
    assert torch.equal(reused["members"], panel["members"])
    with pytest.raises(ValueError, match="stream mismatch"):
        _load_common_core(declaration, **{**kwargs, "seed": 8})
    changed = copy.deepcopy(config)
    changed["rollout"]["steps"] = 2
    with pytest.raises(ValueError, match="rollout contract"):
        _load_common_core(declaration, **{**kwargs, "config": changed})


def fixture(fields):
    names = [f"f{i}" for i in range(fields)]

    class Topology:
        field_names = names
        neighbor_indices = torch.arange(9)[:, None]
        neighbor_weights = torch.ones(9, 1)
        grid_shape = (3, 3)
        smoothing_sigma = 0.0
        smoothing_radius = 0
        periodic = False
        native_masked = False

        def _raster_map(self, coordinates):
            pass

        def _units_and_fields(self, generated, reference):
            return generated, reference

    class State:
        field_names = names
        config: ClassVar[dict] = {"components": {"cross": {"seed": 1234}}}
        components_by_key: ClassVar[dict] = {
            "self_marginal_w2": SimpleNamespace(channel_weights=torch.ones(fields)),
            "cross_joint_topk_swd": SimpleNamespace(
                directions=torch.nn.functional.normalize(
                    torch.randn(16, fields, generator=torch.Generator().manual_seed(41)), dim=1
                )
            ),
        }
        component_weights: ClassVar[dict] = {"self_marginal_w2": 1.0, "cross_joint_topk_swd": 1.0}

        def _in_declared_units(self, generated, target):
            return generated, target

    g = torch.Generator().manual_seed(17)
    reference = torch.randn(2, 9, fields, generator=g)
    source = reference[:, None] + 0.2 * torch.randn(2, 2, 9, fields, generator=g)
    coords = (
        torch.stack(torch.meshgrid(torch.arange(3), torch.arange(3), indexing="ij"), -1)
        .reshape(1, 9, 2)
        .float()
        .expand(2, -1, -1)
    )
    identity = {
        "split": "train",
        "sample_ids": ["a", "b"],
        "field_order": names,
        "source_checkpoint_sha256": "a" * 64,
        "normalizer_sha256": "b" * 64,
        "support_sha256": "c" * 64,
        "config_sha256": "d" * 64,
        "definition_version": "v1",
    }
    return {
        "families": {"topology": Topology(), "global_distribution": State()},
        "source": source,
        "reference": reference,
        "coordinates": coords,
        "provenance": identity,
        "shape_batches": 1,
        "candidate_count": 16,
        "units": {
            f"{family}.{block}": 1.0
            for family, blocks in {
                "scale": ("signed", "joint", "log_power_self", "rho"),
                "state": ("marginal", "mixed"),
                "shape": ("self", "mutual"),
            }.items()
            for block in blocks
        },
    }


@pytest.mark.parametrize("fields", [1, 2, 5])
def test_bounded_fit_is_reusable_deterministic_and_does_not_consume_rng(fields):
    args = fixture(fields)
    torch.manual_seed(987)
    random.seed(123)
    before_torch, before_python = torch.random.get_rng_state(), random.getstate()
    first = fit_scalable_artifacts(**args)
    assert (
        torch.equal(before_torch, torch.random.get_rng_state())
        and before_python == random.getstate()
    )
    second = fit_scalable_artifacts(**args)
    assert first["sha256"] == second["sha256"]
    validate_scalable_artifacts(first, args["provenance"])
    assert first["state_bank"]["mixed_active"] == (fields > 1)
    assert first["active"]["shape.mutual"] == (fields > 1)
    assert len(first["state_bank"]["selected_ids"]) == (
        0 if fields == 1 else min(8, len(first["state_bank"]["selected_ids"]))
    )


def test_stale_identity_and_tampered_artifact_rejected():
    args = fixture(2)
    artifact = fit_scalable_artifacts(**args)
    for key in (
        "source_checkpoint_sha256",
        "normalizer_sha256",
        "support_sha256",
        "config_sha256",
        "definition_version",
        "sample_ids",
        "field_order",
    ):
        changed = copy.deepcopy(args["provenance"])
        changed[key] = (
            list(reversed(changed[key]))
            if isinstance(changed[key], list)
            else ("e" * 64 if key.endswith("sha256") else "v2")
        )
        with pytest.raises(ValueError):
            validate_scalable_artifacts(artifact, changed)
    damaged = copy.deepcopy(artifact)
    damaged["state_bank"]["directions"][0, 0] += 1
    with pytest.raises(ValueError):
        validate_scalable_artifacts(damaged, args["provenance"])
    with pytest.raises(ValueError):
        fit_scalable_artifacts(**{**args, "max_samples": 1})


def test_fits_state_directions_using_declared_units_and_frozen_semantic_units(monkeypatch):
    import phycoflow_reconstruction.training.scalable_fitting as module

    args = fixture(2)
    args.pop("units")
    family = args["families"]["global_distribution"]
    family._in_declared_units = lambda x, y: (
        x * torch.tensor([2.0, 5.0]),
        y * torch.tensor([2.0, 5.0]),
    )
    original = module.fit_source_hardness

    def verify(batches, candidates, **kwargs):
        source, reference, _, _ = batches[0]
        torch.testing.assert_close(source, args["source"] * torch.tensor([2.0, 5.0]))
        torch.testing.assert_close(reference, args["reference"] * torch.tensor([2.0, 5.0]))
        return original(batches, candidates, **kwargs)

    monkeypatch.setattr(module, "fit_source_hardness", verify)

    class Adapter:
        def __init__(self, families, shape):
            pass

        def score(self, family, prediction, reference, **kwargs):
            prefix = "scale" if family == "multiscale_spectrum" else "shape"
            names = (
                ("signed", "joint", "log_power_self", "rho")
                if prefix == "scale"
                else ("self", "mutual")
            )
            return (
                None,
                {prefix + "." + name: torch.tensor(float(i + 2)) for i, name in enumerate(names)},
                None,
            )

    monkeypatch.setattr(module, "ScalableQueryAdapter", Adapter)
    result = fit_scalable_artifacts(**args)
    assert result["units"]["scale.joint"] == 3.0
    assert result["units"]["shape.self"] == 2.0
    assert result["units"]["state.marginal"] > 0


def test_derived_units_use_real_frozen_scale_and_shape_families():
    from scalable_test_helpers import _fitted, _states, config

    states = _states()
    scale, coord, _ = _fitted(config(), states)
    names = ["u", "v"]

    class Topology:
        field_names = names
        neighbor_indices = torch.arange(16)[:, None]
        neighbor_weights = torch.ones(16, 1)
        grid_shape = (4, 4)
        smoothing_sigma = 0.0
        smoothing_radius = 0
        periodic = False
        native_masked = False
        spatial_objective = SimpleNamespace(
            weights={"self.persistence": 1.0, "mutual.persistence": 1.0}, projections=4
        )
        dimensions = (0,)

        def _raster_map(self, coordinates):
            pass

        def _units_and_fields(self, generated, reference):
            return generated, reference

    class State:
        field_names = names
        config: ClassVar[dict] = {"components": {"cross": {"seed": 1234}}}
        components_by_key: ClassVar[dict] = {
            "self_marginal_w2": SimpleNamespace(channel_weights=torch.ones(2)),
            "cross_joint_topk_swd": SimpleNamespace(
                directions=torch.nn.functional.normalize(
                    torch.randn(8, 2, generator=torch.Generator().manual_seed(42)), dim=1
                )
            ),
        }
        component_weights: ClassVar[dict] = {"self_marginal_w2": 1.0, "cross_joint_topk_swd": 1.0}

        def _in_declared_units(self, generated, reference):
            return generated, reference

    identity = {
        "split": "train",
        "sample_ids": ["0", "1"],
        "field_order": names,
        "source_checkpoint_sha256": "a" * 64,
        "normalizer_sha256": "b" * 64,
        "support_sha256": "c" * 64,
        "config_sha256": "d" * 64,
        "definition_version": "v1",
    }
    artifact = fit_scalable_artifacts(
        families={
            "global_distribution": State(),
            "topology": Topology(),
            "multiscale_spectrum": scale,
        },
        source=states[:2, None],
        reference=states[2:4],
        coordinates=coord[None].expand(2, -1, -1),
        provenance=identity,
        shape_batches=1,
        candidate_count=8,
    )
    assert all(artifact["units"][key] > 0 for key in artifact["active"])
    validate_scalable_artifacts(artifact, identity)


def test_candidate_bank_exact_family_identity_and_explicit_overrides():
    args = fixture(2)
    args.pop("candidate_count")
    family = args["families"]["global_distribution"]
    directions = family.components_by_key["cross_joint_topk_swd"].directions
    first = fit_scalable_artifacts(**args)
    bank = first["state_bank"]
    torch.testing.assert_close(bank["candidate_directions"], directions.double(), rtol=0, atol=0)
    torch.testing.assert_close(
        bank["directions"], directions[list(bank["selected_ids"])].double(), rtol=0, atol=0
    )
    assert bank["seed"] == 1234
    custom = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32), dim=1
    )
    override = fit_scalable_artifacts(**args, candidate_matrix=custom)
    torch.testing.assert_close(
        override["state_bank"]["candidate_directions"], custom.double(), rtol=0, atol=0
    )
    assert override["sha256"] != first["sha256"]
    different_count = fit_scalable_artifacts(**args, candidate_count=7)
    assert different_count["state_bank"]["candidate_directions"].shape == (7, 2)
    assert different_count["sha256"] != first["sha256"]
    with pytest.raises(ValueError):
        fit_scalable_artifacts(**args, candidate_count=7, candidate_matrix=custom)
