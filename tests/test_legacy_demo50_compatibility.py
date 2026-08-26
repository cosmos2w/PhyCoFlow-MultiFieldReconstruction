"""Integration checks for the isolated DemoN50 checkpoint adapter.

The numerical equivalence check is opt-in because it loads a large historical
checkpoint and compiles/executes the GPU neighbor search. CI retains the strict
mapping unit check; release validation runs the marked test on physical GPU 1.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest
import torch

from phycoflow_reconstruction.coherence import build_coherence_family
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.models.compatibility.legacy_tc_demo50 import (
    DEMO50_DATASET_FIELDS,
    DEMO50_STALE_CHECKPOINT_FIELDS,
    load_legacy_demo50,
)

REPOSITORY = Path(__file__).resolve().parents[2]
RUN_DIR = (
    REPOSITORY
    / "0_demo_TurbulentCombustion/Save_TrainedModel/ffm_tc_pointcloud_DemoN50_20260706_084857"
)
DATASET = REPOSITORY / "0_demo_TurbulentCombustion/Dataset/Merged_COTU0U1P.h5"
LEGACY_SOURCE = REPOSITORY / "0_demo_TurbulentCombustion/src"
MAPPING = [
    {"channel": index, "checkpoint_label": stale, "dataset_field": actual}
    for index, (stale, actual) in enumerate(
        zip(DEMO50_STALE_CHECKPOINT_FIELDS, DEMO50_DATASET_FIELDS)
    )
]


def _require_optional_legacy_assets(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        pytest.skip(
            "optional local DemoN50 historical validation assets are absent: "
            + ", ".join(missing)
        )


def test_demo50_requires_the_verified_field_mapping() -> None:
    wrong = [dict(item) for item in MAPPING]
    wrong[0]["dataset_field"] = "CH4"
    with pytest.raises(ValueError, match="stale checkpoint field labels"):
        load_legacy_demo50(RUN_DIR, DATASET, wrong)


def test_global_distribution_components_match_historical_demo_math() -> None:
    """The refactor changes taxonomy and state storage, not the three estimators."""
    _require_optional_legacy_assets(LEGACY_SOURCE / "direct_coherence_loss.py")
    sys.path.insert(0, str(LEGACY_SOURCE))
    try:
        from direct_coherence_loss import (  # type: ignore
            DirectCoherenceConfig,
            DirectGlobalCoherenceLoss,
        )
    finally:
        sys.path.remove(str(LEGACY_SOURCE))

    generator = torch.Generator().manual_seed(17)
    generated = torch.randn(2, 64, 5, generator=generator)
    reference = torch.randn(2, 64, 5, generator=generator)
    old_config = DirectCoherenceConfig(
        enabled=True,
        self_weight=1.0,
        mutual_weight=1.0,
        cross_weight=1.0,
        mutual_num_directions=8,
        mutual_seed=1234,
        cross_num_directions=12,
        cross_top_frac=0.25,
        cross_seed=1234,
        cross_include_axes=True,
        cross_qmc=True,
    )
    old_total, old_components = DirectGlobalCoherenceLoss(old_config)(generated, reference)
    names = tuple(f"f{index}" for index in range(5))
    family = build_coherence_family(
        "global_distribution",
        {
            "target_use": "paired_supervised",
            "units": "model_units",
            "fields": names,
            "components": {
                "self": {"enabled": True, "weight": 1.0},
                "mutual": {"enabled": True, "weight": 1.0, "directions": 8, "seed": 1234},
                "cross": {
                    "enabled": True,
                    "weight": 1.0,
                    "directions": 12,
                    "top_fraction": 0.25,
                    "seed": 1234,
                    "include_axes": True,
                    "qmc": True,
                },
            },
        },
        DataSpec(names, ("1",) * 5, 3, (64,)),
        FieldNormalizer.identity(5),
    )
    new_result = family(generated, reference)
    assert torch.allclose(
        new_result.component_results["global_distribution.self.marginal_w2"].scalar_loss,
        old_components["self_loss"],
    )
    assert torch.allclose(
        new_result.component_results["global_distribution.mutual.pairwise_swd"].scalar_loss,
        old_components["mutual_loss"],
    )
    assert torch.allclose(
        new_result.component_results["global_distribution.cross.joint_topk_swd"].scalar_loss,
        old_components["cross_loss"],
    )
    assert torch.allclose(new_result.scalar_loss, old_total)


@pytest.mark.skipif(
    os.environ.get("PHYCOFLOW_RUN_LEGACY_GPU_TEST") != "1",
    reason="run explicitly on physical GPU 1 for Phase-4 release validation",
)
def test_demo50_fixed_seed_reconstruction_matches_legacy_source() -> None:
    _require_optional_legacy_assets(
        RUN_DIR / "args.json",
        RUN_DIR / "run_config.yaml",
        RUN_DIR / "best.pt",
        RUN_DIR / "dataset_stats.pt",
        DATASET,
        LEGACY_SOURCE / "Model.py",
    )
    if not torch.cuda.is_available():
        pytest.fail("legacy equivalence was requested but CUDA is unavailable")
    device = torch.device("cuda:0")
    model, manifest = load_legacy_demo50(RUN_DIR, DATASET, MAPPING, map_location=device)
    model = model.to(device).eval()

    # The reference class is imported only by this verification test. The new
    # compatibility module itself has no runtime dependency on the demo tree.
    sys.path.insert(0, str(LEGACY_SOURCE))
    try:
        from Model import ConditionalPointHybridLocalGlobalRBF  # type: ignore
    finally:
        sys.path.remove(str(LEGACY_SOURCE))
    args = json.loads((RUN_DIR / "args.json").read_text())
    reference = (
        ConditionalPointHybridLocalGlobalRBF(
            n_fields=5,
            coord_dim=3,
            hidden_dim=args["hidden_dim"],
            cond_dim=args["cond_dim"],
            field_embed_dim=args["field_embed_dim"],
            latent_dim=args["latent_dim"],
            num_latents=args["num_latents"],
            num_heads=args["num_heads"],
            num_latent_blocks=args["num_latent_blocks"],
            ff_mult=args["ff_mult"],
            attn_dropout=args["attn_dropout"],
            mlp_dropout=args["mlp_dropout"],
            rbf_sigma=args["rbf_sigma"],
            summary_type=args["summary_type"],
            gather_mode=args["gather_mode"],
            gather_topk=args["gather_topk"],
            gather_query_chunk_size=args["gather_query_chunk_size"],
            learnable_rbf_sigma=args["learnable_rbf_sigma"],
            neighbor_backend=args["neighbor_backend"],
            sensor_local_topk=args["sensor_local_topk"],
            sensor_local_dropout=args["sensor_local_dropout"],
            use_fourier_pe=args["USE_FOURIER_PE"],
            fourier_pe_num_bands=args["fourier_pe_num_bands"],
            fourier_pe_max_freq=args["fourier_pe_max_freq"],
            enhanced_backbone=True,
            sensor_coord_encoding=args["sensor_coord_encoding"],
            latent_sensor_reinject=args["latent_sensor_reinject"],
            latent_reinject_every=args["latent_reinject_every"],
            query_latent_readout=args["query_latent_readout"],
            query_readout_type=args["query_readout_type"],
            query_readout_scale_init=args["query_readout_scale_init"],
            enhanced_head_norm=args["enhanced_head_norm"],
            glres_scale_init=args["glres_scale_init"],
        )
        .to(device)
        .eval()
    )
    payload = torch.load(RUN_DIR / "best.pt", map_location=device, weights_only=False)
    reference.load_state_dict(
        {
            key.removeprefix("model."): value
            for key, value in payload["model"].items()
            if key.startswith("model.")
        },
        strict=True,
    )

    generator = torch.Generator(device=device).manual_seed(20260814)
    coordinates = torch.rand(1, 12, 3, generator=generator, device=device)
    obs_coordinates = torch.rand(1, 36, 3, generator=generator, device=device)
    obs_values = torch.randn(1, 36, 1, generator=generator, device=device)
    obs_mask = torch.ones(1, 36, dtype=torch.bool, device=device)
    obs_field_ids = torch.randint(0, 5, (1, 36), generator=generator, device=device)
    initial_state = torch.randn(1, 12, 5, generator=generator, device=device)

    with torch.no_grad():
        time = torch.full((1,), 0.375, device=device)
        expected_velocity = reference(
            time, initial_state, coordinates, obs_coordinates, obs_values, obs_mask, obs_field_ids
        )
        actual_velocity = model.model(
            time, initial_state, coordinates, obs_coordinates, obs_values, obs_mask, obs_field_ids
        )
        assert torch.allclose(actual_velocity, expected_velocity, rtol=2e-5, atol=2e-6)

        expected = initial_state.clone()
        times = torch.linspace(0.0, 1.0, 3, device=device)
        for index in range(2):
            velocity = reference(
                times[index].expand(1),
                expected,
                coordinates,
                obs_coordinates,
                obs_values,
                obs_mask,
                obs_field_ids,
            )
            expected = expected + (times[index + 1] - times[index]) * velocity
        actual = model.integrate(
            coordinates,
            obs_coordinates,
            obs_values,
            obs_mask,
            obs_field_ids,
            steps=2,
            initial_state=initial_state,
        )
    assert torch.allclose(actual, expected, rtol=2e-5, atol=2e-6)
    assert manifest.rectified_flow["prior"] == "rff"

    # Confirm the checkpoint buffers reproduce the historical RFF formula.
    torch.manual_seed(29)
    expected_weights = torch.randn(1, 5, args["rff_features"], device=device)
    phi = math.sqrt(2.0 / args["rff_features"]) * torch.cos(
        coordinates @ model.prior.omega + model.prior.phase
    )
    expected_prior = torch.einsum("bnf,bcf->bnc", phi, expected_weights)
    torch.manual_seed(29)
    assert torch.equal(model.sample_source(coordinates), expected_prior)
