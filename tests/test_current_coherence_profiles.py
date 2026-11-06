"""The complete recommended profiles survive recursive and SOURCE inheritance."""

from pathlib import Path

import pytest
import yaml

from phycoflow_reconstruction.cli import _load_case_config
from phycoflow_reconstruction.coherence.compose import build_enabled_families
from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.training.coherence_sampling import _conditional_draw_count


@pytest.mark.parametrize(
    ("profile", "objective", "version", "draws"),
    [
        ("coherence_current", "signed_band_energy_score", "signed_spectral_mean_augmented_v4", 2),
        ("coherence_deterministic", "paired_signed_band_mean_distance", "paired_signed_spectral_mean_v1", 1),
    ],
)
def test_complete_profiles_resolve_exactly_state_scale_shape(
    tmp_path, profile, objective, version, draws,
):
    root = Path(__file__).resolve().parents[1]
    case_dir = root / "cases/turbulent_combustion"
    profile_path = case_dir / "configs/posttrain" / f"{profile}.yaml"
    leaf = load_config(profile_path)
    # A legacy Scale block is buried under another default. The selected
    # complete profile must explicitly switch it off after recursive merging.
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(yaml.safe_dump({
        "coherence": {"families": {"cross_spectrum": {"enabled": True, "weight": 3.0}}},
    }), encoding="utf-8")
    middle = tmp_path / "middle.yaml"
    middle.write_text(yaml.safe_dump({"defaults": [str(legacy)]}), encoding="utf-8")
    composed = tmp_path / "composed.yaml"
    composed.write_text(yaml.safe_dump({"defaults": [str(middle), str(profile_path)]}), encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    # Mimic the CLI's SOURCE inheritance without relying on any local run.
    source_config = {"case": "turbulent_combustion"}
    source_config.update({key: leaf[key] for key in ("dataset", "model", "observations")})
    source_config["dataset"].pop("physical_measure", None)
    (source / "resolved_config.yaml").write_text(yaml.safe_dump(source_config), encoding="utf-8")
    resolved = _load_case_config(
        composed, case_dir, "turbulent_combustion", [f"source_run={source}"],
    )
    families = resolved["coherence"]["families"]
    assert {name for name, value in families.items() if value.get("enabled", True)} == {
        "global_distribution", "multiscale_spectrum", "topology",
    }
    assert families["cross_spectrum"]["enabled"] is False
    assert families["global_distribution"]["definition"] == "marginal_mixed_weighted_v3"
    scale = families["multiscale_spectrum"]["objective"]
    assert (scale["kind"], scale["score_definition_version"], scale["conditional_draws"]) == (
        objective, version, draws,
    )
    if draws == 2:
        assert scale["almost_fair_alpha"] == 1
        assert scale["joint_energy_score_weight"] == .25
        assert scale["coherence_crps_weight"] == 0
    else:
        assert "almost_fair_alpha" not in scale
        assert scale["joint_distance_weight"] == .25
    assert resolved["dataset"]["physical_measure"] == "native_rectilinear_area"
    assert resolved["coherence"]["compute_budget"]["point_count"] == 4096
    assert resolved["coherence"]["compute_budget"]["query_policy"] == "fixed_shared"
    assert families["topology"]["geometry"]["grid_shape"] == [32, 128]
    assert families["topology"]["geometry"]["raster_mapping"] == "physical_cell_overlap_v1"
    assert resolved["objectives"]["data_retention"] == {"enabled": False, "weight": 0.0}
    policy = resolved["optimization"]["multitask"]
    assert policy["family_scalarization"]["method"] == "none"
    assert policy["coherence_aggregation"]["method"] == "torchjd_config"
    assert policy["outer_aggregation"] is None
    assert policy["external_policy"]["method"] == "lagrangian"
    assert policy["fidelity_budget"]["relative_increase"] == .10
    names = tuple(resolved["dataset"]["field_names"])
    spec = DataSpec(names, ("unknown",) * len(names), 2, (100, 403))
    built = build_enabled_families(resolved["coherence"], spec, FieldNormalizer.identity(len(names)))
    assert set(built) == {"global_distribution", "multiscale_spectrum", "topology"}
    assert _conditional_draw_count(built) == draws
