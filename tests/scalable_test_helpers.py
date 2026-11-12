import torch

from phycoflow_reconstruction.coherence.families.multiscale_relation import MultiscaleRelationFamily
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer


def config(fields=("u", "v")):
    return {
        "fields": list(fields),
        "units": "model_units",
        "geometry": {"operator": "fem_p1", "coordinate_scaling": "none"},
        "bands": {"count": 2},
        "calibration": {"field_unit_floor": 0.01},
        "numerics": {"dtype": "float64"},
        "objective": {
            "kind": "signed_band_energy_score",
            "conditional_draws": 2,
            "almost_fair_alpha": 1.0,
            "joint_energy_score_weight": 0.25,
            "score_definition_version": "signed_spectral_mean_augmented_v4",
        },
        "relation_extension": {
            "definition": "self_rho_fair_v1",
            "self_weight": 0.2,
            "rho_weight": 0.3,
            "std_floor": 0.001,
            "relative_std_floor": 0.05,
        },
    }


def _states(fields=2):
    return torch.randn(
        8, 16, fields, generator=torch.Generator().manual_seed(23), dtype=torch.float64
    )


def _fitted(config, states):
    fields = tuple(config["fields"])
    n = states.shape[1]
    coords = torch.linspace(0.0, 1.0, n, dtype=torch.float64)[:, None] ** 1.3
    spec = DataSpec(fields, ("1",) * len(fields), 1, (n,))
    family = MultiscaleRelationFamily(config, spec, FieldNormalizer.identity(len(fields)))
    family.fit_reference_tensors(coords, states, sample_ids=[str(i) for i in range(len(states))])
    return family, coords, spec


def _score(family, coords, first, second, target):
    return family(
        first,
        target,
        coordinates=coords[None].expand(first.shape[0], -1, -1),
        context={"conditional_draws": (second,)},
    )
