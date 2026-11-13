from pathlib import Path

import pytest

from phycoflow_reconstruction.coherence.registry import build_coherence_family
from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("count", [1, 2, 20, 100])
def test_generic_fragments_build_for_arbitrary_field_counts(count):
    names = tuple(f"field_{i}" for i in range(count))
    spec = DataSpec(names, ("unit",) * count, 2, (32, 32))
    normalizer = FieldNormalizer.identity(count)
    fragments = {
        "global_distribution": "scalable_state.yaml",
        "multiscale_spectrum": "scalable_scale.yaml",
        "topology": "scalable_shape.yaml",
    }
    built = {}
    for name, filename in fragments.items():
        config = load_config(ROOT / "configs/coherence" / filename)["coherence"]["families"][name]
        built[name] = build_coherence_family(name, config, spec, normalizer)
        assert tuple(built[name].field_names) == names
    state = built["global_distribution"]
    assert "self_marginal_w2" in state.components_by_key
    assert "mutual_pairwise_swd" not in state.components_by_key
    if count > 1:
        assert state.components_by_key["cross_joint_topk_swd"].directions.shape == (32, count)
    shape = built["topology"]
    assert shape.spatial_objective.weights["self.persistence"] > 0
    if count > 1:
        assert shape.spatial_objective.weights["mutual.persistence"] > 0
    else:
        assert "mutual.persistence" not in shape.spatial_objective.weights
    assert not hasattr(shape.spatial_objective, "units")


@pytest.mark.parametrize("scale", ["cross_spectrum", "multiscale_spectrum"])
def test_three_family_order_preserves_legacy_and_new_scale(scale):
    from phycoflow_reconstruction.coherence.compose import three_family_order

    assert three_family_order(["topology", scale, "global_distribution"]) == (
        "global_distribution",
        scale,
        "topology",
    )
    with pytest.raises(ValueError, match="coherence_aggregation"):
        three_family_order(
            ["topology", "cross_spectrum", "multiscale_spectrum", "global_distribution"]
        )


def test_project_config_preserves_physical_cell_overlap(tmp_path):
    from copy import deepcopy

    from helpers.coherence import _post_config

    from phycoflow_reconstruction.config.load import deep_merge
    from phycoflow_reconstruction.config.validate import validate_config

    config = _post_config(tmp_path / "fixture.h5", tmp_path / "source")
    config = deep_merge(
        config, load_config(ROOT / "configs/defaults/scalable_anchored_coherence.yaml")
    )
    config["coherence"]["families"].pop("cross_spectrum", None)
    config["coherence"]["families"]["topology"]["geometry"]["raster_mapping"] = (
        "physical_cell_overlap_v1"
    )
    config["checkpointing"] = {"selection_metric": "reconstruction_mse"}
    config["evaluation"]["preview"] = {"enabled": False}
    config["evaluation"]["query_points"] = config["coherence"]["compute_budget"]["point_count"]
    config["evaluation"]["batch_size"] = 4
    validate_config(config)
    for size in (True, 0):
        invalid = deepcopy(config)
        invalid["evaluation"]["batch_size"] = size
        with pytest.raises(ValueError, match="evaluation.batch_size must be a positive integer"):
            validate_config(invalid)
    for key in ("periodic", "antialias_downsample"):
        invalid = deepcopy(config)
        invalid["coherence"]["families"]["topology"]["geometry"][key] = True
        with pytest.raises(ValueError, match="physical_cell_overlap"):
            validate_config(invalid)
    invalid = deepcopy(config)
    invalid["coherence"]["families"]["topology"]["geometry"]["raster_mapping"] = "unknown"
    with pytest.raises(ValueError, match="raster_mapping"):
        validate_config(invalid)
