"""Small CPU parity gates against the archived coherence equation kernels."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from phycoflow_reconstruction.coherence import build_coherence_family
from phycoflow_reconstruction.coherence.families.cross_spectrum.basis import (
    coordinate_digest,
)
from phycoflow_reconstruction.coherence.families.topology.betti_curves import (
    betti_curves,
)
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer

WORKTREE = Path(__file__).resolve().parents[1]
LEGACY_ROOT = WORKTREE.parent / "Proj_MultiFieldReconstruction" / "LegacyCoherence"
GLOBAL_SOURCE = WORKTREE.parent / "0_demo_TurbulentCombustion" / "src" / "direct_coherence_loss.py"
SPECTRAL_SOURCE = (
    LEGACY_ROOT
    / "PhyCoFlowModel-Cross-Spectral-Coherence"
    / "src"
    / "graph_spectral_coherence"
    / "cross_spectral.py"
)
TOPOLOGY_SOURCE = LEGACY_ROOT / "PhyCoFlow_dev" / "src"
GLOBAL_DEPENDENCIES = GLOBAL_SOURCE.parent


def _load_file(name: str, path: Path):
    if not path.is_file():
        pytest.skip(f"optional archived coherence source is absent: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_global_distribution_matches_archived_canonical_estimators() -> None:
    if not (GLOBAL_DEPENDENCIES / "coherence_dist.py").is_file():
        pytest.skip("optional archived global-distribution dependency is absent")
    sys.path.insert(0, str(GLOBAL_DEPENDENCIES))
    try:
        legacy = _load_file("_legacy_global_distribution", GLOBAL_SOURCE)
    finally:
        sys.path.remove(str(GLOBAL_DEPENDENCIES))
    generated = torch.tensor(
        [[[0.0, 1.0, 2.0], [2.0, -1.0, 1.0], [1.0, 3.0, -2.0], [4.0, 0.5, 0.0]]]
    )
    reference = torch.tensor(
        [[[1.0, 0.0, 2.0], [3.0, -2.0, 0.0], [0.0, 2.0, -1.0], [2.0, 1.5, 1.0]]]
    )
    settings = {
        "enabled": True,
        "self_weight": 1.0,
        "mutual_weight": 0.5,
        "cross_weight": 0.25,
        "mutual_num_directions": 4,
        "mutual_seed": 19,
        "cross_num_directions": 6,
        "cross_top_frac": 0.5,
        "cross_seed": 23,
        "cross_include_axes": True,
        "cross_qmc": True,
    }
    expected_total, expected = legacy.DirectGlobalCoherenceLoss(
        legacy.DirectCoherenceConfig(**settings)
    )(generated, reference)
    names = ("a", "b", "c")
    family = build_coherence_family(
        "global_distribution",
        {
            "fields": names,
            "components": {
                "self": {"weight": 1.0},
                "mutual": {"weight": 0.5, "directions": 4, "seed": 19},
                "cross": {
                    "weight": 0.25,
                    "directions": 6,
                    "top_fraction": 0.5,
                    "seed": 23,
                    "include_axes": True,
                    "qmc": True,
                },
            },
        },
        DataSpec(names, ("1",) * 3, 1, (4,)),
        FieldNormalizer.identity(3),
    )
    actual = family(generated, reference)
    keys = {
        "global_distribution.self.marginal_w2": "self_loss",
        "global_distribution.mutual.pairwise_swd": "mutual_loss",
        "global_distribution.cross.joint_topk_swd": "cross_loss",
    }
    for current, archived in keys.items():
        torch.testing.assert_close(actual.component_results[current].scalar_loss, expected[archived])
    torch.testing.assert_close(actual.scalar_loss, expected_total)


def test_cross_spectrum_restored_basis_matches_archived_zero_mode_and_bands() -> None:
    legacy = _load_file("_legacy_cross_spectral", SPECTRAL_SOURCE)
    config = {
        "fields": ["u", "v"],
        "pairs": [["u", "v"]],
        "eps": 1e-8,
        "graph": {
            "num_modes": 4,
            "exclude_zero": False,
            "bands": ["low", "high"],
        },
        "components": {
            "same_frequency": {"weight": 1.0},
            "cross_frequency": {"weight": 1.0},
            "band_energy": {"enabled": True, "weight": 1.0},
        },
    }
    data_spec = DataSpec(("u", "v"), ("1", "1"), 1, (4,))
    family = build_coherence_family(
        "cross_spectrum", config, data_spec, FieldNormalizer.identity(2)
    )
    coordinates = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1).expand(4, -1, -1)
    artifact = family.state_artifact()
    artifact["geometry_sha256"] = coordinate_digest(coordinates[0])
    artifact["resolved_sigma"] = 1.0
    artifact["state_dict"]["eigenvalues"] = torch.arange(4, dtype=torch.float32)
    artifact["state_dict"]["eigenvectors"] = torch.eye(4)
    artifact["state_dict"]["band_ids"] = torch.tensor([0, 0, 1, 1])
    family.load_state_artifact(artifact)

    generated = torch.arange(32, dtype=torch.float32).reshape(4, 4, 2) / 7.0
    reference = generated.flip(0).clone()
    reference[:, 0, 1] += torch.tensor([0.0, 1.0, -0.5, 2.0])
    actual = family(generated, reference, coordinates=coordinates)

    bands = {"low": [0, 1], "high": [2, 3]}
    cfg = legacy.CrossSpectralConfig(field_pairs=[(0, 1)], eps=1e-8)
    expected = legacy.compute_physical_coherence_loss(
        generated, reference, torch.eye(4), bands, cfg
    )
    torch.testing.assert_close(
        actual.component_results["cross_spectrum.same_frequency.magnitude_squared"].scalar_loss,
        expected["L_same"],
    )
    torch.testing.assert_close(
        actual.component_results[
            "cross_spectrum.cross_frequency.band_energy_coupling"
        ].scalar_loss,
        expected["L_crossfreq"],
    )
    generated_energy = torch.stack(
        [legacy.compute_band_energy(generated, indices) for indices in bands.values()], dim=1
    )
    reference_energy = torch.stack(
        [legacy.compute_band_energy(reference, indices) for indices in bands.values()], dim=1
    )
    expected_energy = (
        (generated_energy.mean(0) + 1e-8).log()
        - (reference_energy.mean(0) + 1e-8).log()
    ).square().mean()
    torch.testing.assert_close(
        actual.component_results["cross_spectrum.band_energy.log_power"].scalar_loss,
        expected_energy,
    )
    torch.testing.assert_close(
        actual.scalar_loss, expected["L_same"] + expected["L_crossfreq"] + expected_energy
    )


@pytest.mark.parametrize("periodic", [False, True])
def test_topology_hard_forward_betti_v1_matches_archived_kernel(periodic: bool) -> None:
    marginal = TOPOLOGY_SOURCE / "topo_coherence_training" / "marginal_betti.py"
    if not marginal.is_file():
        pytest.skip(f"optional archived topology source is absent: {marginal}")
    sys.path.insert(0, str(TOPOLOGY_SOURCE))
    try:
        legacy = _load_file(f"_legacy_marginal_betti_{periodic}", marginal)
    finally:
        sys.path.remove(str(TOPOLOGY_SOURCE))
    field = torch.tensor(
        [[2.0, 2.0, -1.0, -1.0], [2.0, 0.0, 0.0, -1.0], [-1.0, 0.0, 0.0, 2.0], [-1.0, -1.0, 2.0, 2.0]]
    )
    levels = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    actual = betti_curves(field, levels, (0, 1), sharpness=12.0, periodic=periodic)
    for dimension in (0, 1):
        expected = legacy.soft_betti_curve_dim(
            field,
            levels,
            dimension,
            beta=12.0,
            kappa=12.0,
            periodic=periodic,
            connectivity=1,
        )
        torch.testing.assert_close(actual[dimension], expected)
