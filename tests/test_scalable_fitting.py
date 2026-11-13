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
