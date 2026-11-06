"""Configuration contract for multiscale spectral consistency, version 1.

The same parser serves the family constructor and strict config validation, so
the accepted keys and their meaning have one definition. Field and band
references in pair and tuple manifests are resolved here into indices into the
selected fields and the coarse-to-fine band list. Options that have no effect
for the chosen operator are rejected rather than ignored.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, pairwise
from typing import Any

FAMILY_KEYS = frozenset({"geometry", "bands", "regions", "calibration", "numerics", "objective"})
COMPONENT_KEYS: dict[str, frozenset[str]] = {
    "log_band_power": frozenset({"enabled", "weight"}),
    "signed_band_cospectrum": frozenset({"enabled", "weight", "pairs"}),
    "third_order_product_moments": frozenset({"enabled", "weight", "tuples", "regional"}),
}
# Cross-scale moments need an explicit, predeclared manifest, so they never run
# because a key was omitted.
OPT_IN_COMPONENTS = frozenset({"third_order_product_moments"})
COMPONENT_ORDER = tuple(COMPONENT_KEYS)
FEM_GEOMETRY_KEYS = frozenset(
    {
        "triangulation",
        "structured_indices",
        "max_edge_factor",
        "spacing_neighbors",
        "support_factor",
        "support_neighbors",
        "connectivity",
        "min_component_fraction",
    }
)
GRAPH_GEOMETRY_KEYS = frozenset({"k_neighbors", "sigma"})
GEOMETRY_KEYS = (
    frozenset({"operator", "coordinate_scaling", "length_unit", "periodic", "periods", "max_modes"})
    | FEM_GEOMETRY_KEYS
    | GRAPH_GEOMETRY_KEYS
)
FEM_BAND_KEYS = frozenset({"wavelengths", "min_wavelength", "resolution_factor"})
BAND_KEYS = frozenset({"count", "names"}) | FEM_BAND_KEYS
REGION_KEYS = frozenset({"grid", "min_mass_fraction"})
CALIBRATION_KEYS = frozenset(
    {"samples", "seed", "band_relative_floor", "absolute_floor", "field_unit_floor"}
)
NUMERICS_KEYS = frozenset({"dtype"})
OBJECTIVE_KEYS = frozenset(
    {
        "kind",
        "conditional_draws",
        "standardize_descriptors",
        # Signed-field objectives only; report version is independent of artifact version.
        "almost_fair_alpha",
        "projections",
        "projection_pairs",
        "band_weights",
        "coherence_crps_weight",
        "joint_energy_score_weight",
        "score_definition_version",
        "mean_weight",
    }
)
OBJECTIVES = ("paired_descriptor_mse", "conditional_energy_score", "signed_band_energy_score")
SIGNED_BAND_KEYS = OBJECTIVE_KEYS - {"kind", "conditional_draws", "standardize_descriptors"}
PROJECTIONS = ("axes", "axes_and_pairs")
OPERATORS = ("fem_p1", "knn_graph")
COORDINATE_SCALINGS = ("none", "raw_extent", "lonlat_degrees")
TRIANGULATIONS = ("delaunay", "structured_indices")
CONNECTIVITY = ("require_connected", "components")
MAX_CONDITIONAL_DRAWS = 8
LEGACY_SCORE_VERSION = "signed_spectral_v3"
MEAN_AUGMENTED_SCORE_VERSION = "signed_spectral_mean_augmented_v4"


@dataclass(frozen=True)
class GeometrySettings:
    operator: str
    coordinate_scaling: str | tuple[float, ...]
    length_unit: str
    periodic: bool | tuple[bool, ...]
    periods: tuple[float, ...] | None
    triangulation: str
    structured_indices: str | None
    max_edge_factor: float | None
    spacing_neighbors: int
    support_factor: float | None
    support_neighbors: int
    connectivity: str
    min_component_fraction: float
    k_neighbors: int
    sigma: float | None
    max_modes: int


@dataclass(frozen=True)
class BandSettings:
    names: tuple[str, ...]
    wavelengths: tuple[float, ...] | None
    count: int
    min_wavelength: float | None
    resolution_factor: float


@dataclass(frozen=True)
class RegionSettings:
    grid: tuple[int, ...]
    min_mass_fraction: float


@dataclass(frozen=True)
class CalibrationSettings:
    samples: int
    seed: int
    band_relative_floor: float
    absolute_floor: float
    # Explicit tolerance in each selected field's declared units. None preserves
    # the historical zero-fluctuation rejection and numerical formula.
    field_unit_floor: tuple[float, ...] | None = None


@dataclass(frozen=True)
class ComponentSettings:
    key: str
    weight: float
    pairs: tuple[tuple[int, int], ...] = ()
    tuples: tuple[tuple[tuple[int, int], ...], ...] = ()
    regional: bool = False


@dataclass(frozen=True)
class MultiscaleSettings:
    fields: tuple[str, ...]
    target_use: str
    units: str
    family_weight: float
    geometry: GeometrySettings
    bands: BandSettings
    regions: RegionSettings | None
    calibration: CalibrationSettings
    dtype: str
    objective: str
    conditional_draws: int
    components: tuple[ComponentSettings, ...]
    # Divide each descriptor's weight by its variance across the training calibration states.
    standardize_descriptors: bool = False
    # signed_band_energy_score: almost-fair repulsion weight (1 = fair U-statistic), fixed
    # field-space projections, per-band weights (None = equal) and the auxiliary band-
    # coherence CRPS weight (0 = report only).
    almost_fair_alpha: float = 1.0
    projections: str = "axes_and_pairs"
    projection_pairs: tuple[tuple[int, int], ...] | None = None
    band_weights: tuple[float, ...] | None = None
    coherence_crps_weight: float = 0.0
    joint_energy_score_weight: float = 0.0
    score_definition_version: str = LEGACY_SCORE_VERSION
    mean_weight: float = 1.0

    def component(self, key: str) -> ComponentSettings | None:
        return next((item for item in self.components if item.key == key), None)


def _reject_unknown(mapping: Mapping[str, Any], allowed: frozenset[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown {path} keys: {unknown}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return value


def _positive_float(value: Any, path: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{path} must be positive and finite")
    return number


def _integer(value: Any, path: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer")
    if value < minimum:
        raise ValueError(f"{path} must be >= {minimum}")
    return int(value)


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be boolean")
    return value


def component_enabled(
    key: str, settings: Mapping[str, Any], field_count: int | None = None
) -> bool:
    """Resolve activation: cross-scale moments are opt-in; co-spectra need two fields."""
    default = key not in OPT_IN_COMPONENTS
    if key == "signed_band_cospectrum" and field_count is not None and field_count < 2:
        default = False
    return bool(settings.get("enabled", default))


def _geometry(settings: Mapping[str, Any]) -> GeometrySettings:
    _reject_unknown(settings, GEOMETRY_KEYS, "multiscale_spectrum.geometry")
    operator = str(settings.get("operator", "fem_p1"))
    if operator not in OPERATORS:
        raise ValueError(f"multiscale_spectrum.geometry.operator must be one of {OPERATORS}")
    inapplicable = sorted(
        set(settings) & (GRAPH_GEOMETRY_KEYS if operator == "fem_p1" else FEM_GEOMETRY_KEYS)
    )
    if inapplicable:
        raise ValueError(
            f"multiscale_spectrum.geometry keys {inapplicable} do not apply to {operator}"
        )
    scaling_value = settings.get("coordinate_scaling", "raw_extent")
    if isinstance(scaling_value, str):
        if scaling_value not in COORDINATE_SCALINGS:
            raise ValueError(
                "multiscale_spectrum.geometry.coordinate_scaling must be one of "
                f"{COORDINATE_SCALINGS} or a list of positive axis lengths"
            )
        scaling: str | tuple[float, ...] = scaling_value
    elif isinstance(scaling_value, Sequence) and scaling_value:
        scaling = tuple(
            _positive_float(value, "multiscale_spectrum.geometry.coordinate_scaling[]")
            for value in scaling_value
        )
    else:
        raise TypeError("multiscale_spectrum.geometry.coordinate_scaling is invalid")
    default_unit = {
        "none": "normalized",
        "raw_extent": "raw_coordinate_unit",
        "lonlat_degrees": "km",
    }.get(scaling if isinstance(scaling, str) else "", "declared")
    length_unit = str(settings.get("length_unit", default_unit))
    if scaling == "lonlat_degrees" and length_unit != "km":
        raise ValueError("lonlat_degrees coordinate scaling reports lengths in km")
    periodic_value = settings.get("periodic", False)
    if isinstance(periodic_value, bool):
        periodic: bool | tuple[bool, ...] = periodic_value
    elif isinstance(periodic_value, Sequence) and periodic_value:
        periodic = tuple(
            _bool(value, "multiscale_spectrum.geometry.periodic[]") for value in periodic_value
        )
    else:
        raise TypeError("multiscale_spectrum.geometry.periodic must be a boolean or boolean list")
    any_periodic = periodic if isinstance(periodic, bool) else any(periodic)
    periods_value = settings.get("periods")
    periods = None
    if periods_value is not None:
        if not isinstance(periods_value, Sequence) or isinstance(periods_value, str):
            raise TypeError("multiscale_spectrum.geometry.periods must be a list")
        periods = tuple(
            _positive_float(value, "multiscale_spectrum.geometry.periods[]")
            for value in periods_value
        )
    if any_periodic and periods is None:
        raise ValueError(
            "periodic multiscale geometry requires geometry.periods in normalized coordinate units"
        )
    if not any_periodic and periods is not None:
        raise ValueError("multiscale_spectrum.geometry.periods requires periodic axes")
    if any_periodic and scaling == "lonlat_degrees":
        raise ValueError("lonlat_degrees scaling does not support periodic axes")
    max_edge_factor = _positive_float(
        settings.get("max_edge_factor"),
        "multiscale_spectrum.geometry.max_edge_factor",
        allow_none=True,
    )
    if max_edge_factor is not None and max_edge_factor <= 1.0:
        raise ValueError("multiscale_spectrum.geometry.max_edge_factor must exceed one")
    triangulation = str(settings.get("triangulation", "delaunay"))
    if triangulation not in TRIANGULATIONS:
        raise ValueError(
            f"multiscale_spectrum.geometry.triangulation must be one of {TRIANGULATIONS}"
        )
    structured_indices = settings.get("structured_indices")
    if triangulation == "structured_indices":
        if any_periodic:
            raise ValueError("structured_indices triangulation does not support periodic axes")
        if not isinstance(structured_indices, str) or not structured_indices:
            raise ValueError(
                "triangulation=structured_indices requires geometry.structured_indices: the name "
                "of a per-point auxiliary (i, j) grid-index array"
            )
    elif structured_indices is not None:
        raise ValueError("geometry.structured_indices requires triangulation=structured_indices")
    support_factor = _positive_float(
        settings.get("support_factor"),
        "multiscale_spectrum.geometry.support_factor",
        allow_none=True,
    )
    if support_factor is not None and (triangulation != "delaunay" or any_periodic):
        raise ValueError("geometry.support_factor filters a non-periodic Delaunay mesh only")
    connectivity = str(settings.get("connectivity", "require_connected"))
    if connectivity not in CONNECTIVITY:
        raise ValueError(f"multiscale_spectrum.geometry.connectivity must be one of {CONNECTIVITY}")
    fraction_value = settings.get("min_component_fraction")
    if fraction_value is not None and connectivity != "components":
        raise ValueError("geometry.min_component_fraction requires connectivity=components")
    min_component_fraction = 0.01 if fraction_value is None else fraction_value
    if (
        isinstance(min_component_fraction, bool)
        or not isinstance(min_component_fraction, (int, float))
        or not 0.0 <= float(min_component_fraction) < 1.0
    ):
        raise ValueError("geometry.min_component_fraction must lie in [0, 1)")
    sigma = _positive_float(
        settings.get("sigma"), "multiscale_spectrum.geometry.sigma", allow_none=True
    )
    return GeometrySettings(
        operator=operator,
        coordinate_scaling=scaling,
        length_unit=length_unit,
        periodic=periodic,
        periods=periods,
        triangulation=triangulation,
        structured_indices=structured_indices,
        max_edge_factor=max_edge_factor,
        spacing_neighbors=_integer(
            settings.get("spacing_neighbors", 6),
            "multiscale_spectrum.geometry.spacing_neighbors",
            minimum=1,
        ),
        support_factor=support_factor,
        support_neighbors=_integer(
            settings.get("support_neighbors", 8),
            "multiscale_spectrum.geometry.support_neighbors",
            minimum=1,
        ),
        connectivity=connectivity,
        min_component_fraction=float(min_component_fraction),
        k_neighbors=_integer(
            settings.get("k_neighbors", 16), "multiscale_spectrum.geometry.k_neighbors", minimum=1
        ),
        sigma=sigma,
        max_modes=_integer(
            settings.get("max_modes", 2048), "multiscale_spectrum.geometry.max_modes", minimum=1
        ),
    )


def _bands(settings: Mapping[str, Any], operator: str) -> BandSettings:
    _reject_unknown(settings, BAND_KEYS, "multiscale_spectrum.bands")
    if operator != "fem_p1" and set(settings) & FEM_BAND_KEYS:
        raise ValueError(
            "knn_graph bands are graph scales, not wavenumber intervals: use bands.count; "
            "wavelengths, min_wavelength and resolution_factor need the physical fem_p1 operator"
        )
    if ("wavelengths" in settings) == ("count" in settings):
        raise ValueError("multiscale_spectrum.bands requires exactly one of wavelengths or count")
    wavelengths = None
    if "wavelengths" in settings:
        values = settings["wavelengths"]
        if not isinstance(values, Sequence) or isinstance(values, str) or not values:
            raise TypeError("multiscale_spectrum.bands.wavelengths must be a non-empty list")
        wavelengths = tuple(
            _positive_float(value, "multiscale_spectrum.bands.wavelengths[]") for value in values
        )
        if any(left <= right for left, right in pairwise(wavelengths)):
            raise ValueError("band wavelengths must be strictly decreasing (coarse to fine)")
        count = len(wavelengths)
    else:
        count = _integer(settings["count"], "multiscale_spectrum.bands.count", minimum=1)
    names_value = settings.get("names")
    if names_value is None:
        names = tuple(f"b{index}" for index in range(count))
    else:
        if not isinstance(names_value, Sequence) or isinstance(names_value, str):
            raise TypeError("multiscale_spectrum.bands.names must be a list")
        names = tuple(str(name) for name in names_value)
    if len(names) != count or len(set(names)) != count or any(not name for name in names):
        raise ValueError("band names must be unique, non-empty, and match the band count")
    min_wavelength = _positive_float(
        settings.get("min_wavelength"), "multiscale_spectrum.bands.min_wavelength", allow_none=True
    )
    if min_wavelength is not None and wavelengths is not None and min_wavelength >= wavelengths[-1]:
        raise ValueError("bands.min_wavelength must lie below the finest band wavelength")
    resolution_factor = _positive_float(
        settings.get("resolution_factor", 6.0), "multiscale_spectrum.bands.resolution_factor"
    )
    if resolution_factor < 2.0:
        raise ValueError("bands.resolution_factor below two exceeds the sampling Nyquist limit")
    return BandSettings(
        names=names,
        wavelengths=wavelengths,
        count=count,
        min_wavelength=min_wavelength,
        resolution_factor=resolution_factor,
    )


def _regions(settings: Mapping[str, Any] | None) -> RegionSettings | None:
    if settings is None:
        return None
    settings = _mapping(settings, "multiscale_spectrum.regions")
    _reject_unknown(settings, REGION_KEYS, "multiscale_spectrum.regions")
    grid = settings.get("grid")
    if not isinstance(grid, Sequence) or isinstance(grid, str) or not grid:
        raise TypeError("multiscale_spectrum.regions.grid must be a non-empty integer list")
    resolved = tuple(
        _integer(value, "multiscale_spectrum.regions.grid[]", minimum=1) for value in grid
    )
    if math.prod(resolved) < 2:
        raise ValueError("multiscale_spectrum.regions.grid must define at least two windows")
    fraction = _positive_float(
        settings.get("min_mass_fraction", 0.02), "multiscale_spectrum.regions.min_mass_fraction"
    )
    if fraction >= 1.0:
        raise ValueError("multiscale_spectrum.regions.min_mass_fraction must be below one")
    return RegionSettings(grid=resolved, min_mass_fraction=fraction)


def _calibration(settings: Mapping[str, Any], fields: tuple[str, ...]) -> CalibrationSettings:
    _reject_unknown(settings, CALIBRATION_KEYS, "multiscale_spectrum.calibration")
    field_floor = settings.get("field_unit_floor")
    if field_floor is not None:
        path = "multiscale_spectrum.calibration.field_unit_floor"
        if isinstance(field_floor, Mapping):
            if set(field_floor) != set(fields):
                raise ValueError(f"{path} must specify exactly the selected fields")
            field_floor = tuple(
                float(_positive_float(field_floor[field], f"{path}.{field}")) for field in fields
            )
        else:
            floor = float(_positive_float(field_floor, path))
            field_floor = (floor,) * len(fields)
    return CalibrationSettings(
        samples=_integer(
            settings.get("samples", 256), "multiscale_spectrum.calibration.samples", minimum=2
        ),
        seed=_integer(
            settings.get("seed", 1234), "multiscale_spectrum.calibration.seed", minimum=0
        ),
        band_relative_floor=float(
            _positive_float(
                settings.get("band_relative_floor", 1.0e-3),
                "multiscale_spectrum.calibration.band_relative_floor",
            )
        ),
        absolute_floor=float(
            _positive_float(
                settings.get("absolute_floor", 1.0e-6),
                "multiscale_spectrum.calibration.absolute_floor",
            )
        ),
        field_unit_floor=field_floor,
    )


def _field_index(name: Any, fields: tuple[str, ...], path: str) -> int:
    if str(name) not in fields:
        raise KeyError(f"{path} references a field outside multiscale_spectrum.fields: {name!r}")
    return fields.index(str(name))


def _band_index(value: Any, names: tuple[str, ...], path: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{path} band references must be names or integer indices")
    if isinstance(value, int):
        if not 0 <= value < len(names):
            raise ValueError(f"{path} band index {value} is outside 0..{len(names) - 1}")
        return int(value)
    if str(value) not in names:
        raise KeyError(f"{path} references an unknown band {value!r}; bands are {list(names)}")
    return names.index(str(value))


def _factor(
    value: Any, fields: tuple[str, ...], bands: tuple[str, ...], path: str
) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
        raise ValueError(f"{path} factors must be [field, band]")
    return _field_index(value[0], fields, path), _band_index(value[1], bands, path)


def _triples(
    values: Any, fields: tuple[str, ...], bands: tuple[str, ...], path: str
) -> tuple[tuple[tuple[int, int], ...], ...]:
    if not isinstance(values, Sequence) or isinstance(values, str) or not values:
        raise ValueError(f"{path} must be a non-empty list of 3-factor groups")
    groups = []
    for group in values:
        if not isinstance(group, Sequence) or isinstance(group, str) or len(group) != 3:
            raise ValueError(f"{path} entries must contain exactly 3 [field, band] factors")
        # The moment is symmetric in its factors; one canonical order makes
        # duplicate entries detectable.
        groups.append(tuple(sorted(_factor(item, fields, bands, path) for item in group)))
    if len(set(groups)) != len(groups):
        raise ValueError(f"{path} contains duplicate entries (factor order is irrelevant)")
    return tuple(groups)


def _cospectrum_pairs(
    value: Any, fields: tuple[str, ...], path: str
) -> tuple[tuple[int, int], ...]:
    if value is None:
        return tuple(combinations(range(len(fields)), 2))
    if not isinstance(value, Sequence) or isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty list of field pairs")
    pairs = []
    for pair in value:
        if not isinstance(pair, Sequence) or isinstance(pair, str) or len(pair) != 2:
            raise ValueError(f"{path} entries must be [field, field]")
        left, right = (_field_index(name, fields, path) for name in pair)
        if left == right:
            raise ValueError(f"{path} entries must contain distinct fields")
        pairs.append((min(left, right), max(left, right)))
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"{path} entries must be unique regardless of order")
    return tuple(pairs)


def parse_multiscale_config(
    config: Mapping[str, Any], dataset_field_names: Sequence[str]
) -> MultiscaleSettings:
    """Validate one ``multiscale_spectrum`` family block and resolve its manifests."""
    common = {"enabled", "weight", "target_use", "units", "fields", "reference_bank", "components"}
    _reject_unknown(
        config, frozenset(common) | FAMILY_KEYS, "coherence.families.multiscale_spectrum"
    )
    target_use = str(config.get("target_use", "paired_supervised"))
    if target_use != "paired_supervised":
        raise ValueError(
            "multiscale_spectrum v1 supports target_use=paired_supervised only; an unpaired "
            "reference bank does not represent the conditional target distribution"
        )
    reference_bank = _mapping(config.get("reference_bank"), "multiscale_spectrum.reference_bank")
    if bool(reference_bank.get("enabled", False)):
        raise ValueError("paired multiscale_spectrum does not use a reference bank")
    units = str(config.get("units", "model_units"))
    if units not in {"model_units", "physical_units"}:
        raise ValueError("multiscale_spectrum.units must be model_units or physical_units")
    family_weight = _positive_float(config.get("weight", 1.0), "multiscale_spectrum.weight")
    available = tuple(str(name) for name in dataset_field_names)
    fields = tuple(str(name) for name in (config.get("fields") or available))
    if not fields:
        raise ValueError("multiscale_spectrum needs at least one field")
    if len(set(fields)) != len(fields):
        raise ValueError("multiscale_spectrum.fields must be unique")
    unknown = sorted(set(fields) - set(available))
    if unknown:
        raise KeyError(f"unknown multiscale_spectrum fields: {unknown}")
    geometry = _geometry(_mapping(config.get("geometry"), "multiscale_spectrum.geometry"))
    bands = _bands(_mapping(config.get("bands"), "multiscale_spectrum.bands"), geometry.operator)
    regions = _regions(config.get("regions"))
    calibration = _calibration(
        _mapping(config.get("calibration"), "multiscale_spectrum.calibration"), fields
    )
    numerics = _mapping(config.get("numerics"), "multiscale_spectrum.numerics")
    _reject_unknown(numerics, NUMERICS_KEYS, "multiscale_spectrum.numerics")
    dtype = str(numerics.get("dtype", "float32"))
    if dtype not in {"float32", "float64"}:
        raise ValueError("multiscale_spectrum.numerics.dtype must be float32 or float64")
    objective_settings = _mapping(config.get("objective"), "multiscale_spectrum.objective")
    _reject_unknown(objective_settings, OBJECTIVE_KEYS, "multiscale_spectrum.objective")
    objective = str(objective_settings.get("kind", "paired_descriptor_mse"))
    if objective not in OBJECTIVES:
        raise ValueError(f"multiscale_spectrum.objective.kind must be one of {OBJECTIVES}")
    if objective in {"conditional_energy_score", "signed_band_energy_score"}:
        draws = _integer(
            objective_settings.get("conditional_draws", 2),
            "multiscale_spectrum.objective.conditional_draws",
            minimum=2,
        )
        if draws > MAX_CONDITIONAL_DRAWS:
            raise ValueError(
                f"multiscale_spectrum.objective.conditional_draws must be <= {MAX_CONDITIONAL_DRAWS}"
            )
    else:
        draws = _integer(
            objective_settings.get("conditional_draws", 1),
            "multiscale_spectrum.objective.conditional_draws",
            minimum=1,
        )
        if draws != 1:
            raise ValueError("paired_descriptor_mse uses one draw; conditional_draws must be 1")
    standardize = objective_settings.get("standardize_descriptors", False)
    if not isinstance(standardize, bool):
        raise TypeError("multiscale_spectrum.objective.standardize_descriptors must be boolean")
    signed_band = _signed_band_settings(objective_settings, objective, bands, fields, standardize)
    if signed_band.get("score_definition_version") == MEAN_AUGMENTED_SCORE_VERSION:
        if calibration.field_unit_floor is None:
            raise ValueError("mean-augmented score requires calibration.field_unit_floor")
        if units == "physical_units" and dtype != "float64":
            raise ValueError("mean-augmented physical_units require numerics.dtype=float64")
    if len(fields) < 2 and signed_band.get("coherence_crps_weight", 0.0) > 0:
        raise ValueError("coherence_crps_weight > 0 requires at least two fields")

    components_config = _mapping(config.get("components"), "multiscale_spectrum.components")
    _reject_unknown(components_config, frozenset(COMPONENT_KEYS), "multiscale_spectrum.components")
    components = []
    for key in COMPONENT_ORDER:
        settings = _mapping(components_config.get(key), f"multiscale_spectrum.components.{key}")
        path = f"multiscale_spectrum.components.{key}"
        _reject_unknown(settings, COMPONENT_KEYS[key], path)
        if "enabled" in settings:
            _bool(settings["enabled"], f"{path}.enabled")
        weight_value = settings.get("weight", 1.0)
        if isinstance(weight_value, bool) or not isinstance(weight_value, (int, float)):
            raise TypeError(f"{path}.weight must be a number")
        weight = float(weight_value)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{path}.weight must be non-negative")
        if not component_enabled(key, settings, len(fields)):
            continue
        if weight <= 0:
            raise ValueError(f"enabled multiscale_spectrum component {key} needs a positive weight")
        pairs: tuple[tuple[int, int], ...] = ()
        tuples: tuple[tuple[tuple[int, int], ...], ...] = ()
        regional = False
        if key == "signed_band_cospectrum":
            if len(fields) < 2:
                raise ValueError("signed_band_cospectrum requires at least two fields")
            pairs = _cospectrum_pairs(settings.get("pairs"), fields, f"{path}.pairs")
        elif key == "third_order_product_moments":
            if "tuples" not in settings:
                raise ValueError(f"{path} requires an explicit tuples manifest")
            tuples = _triples(settings["tuples"], fields, bands.names, f"{path}.tuples")
            regional = _bool(settings.get("regional", regions is not None), f"{path}.regional")
            if regional and regions is None:
                raise ValueError(f"{path}.regional requires multiscale_spectrum.regions")
        components.append(
            ComponentSettings(key=key, weight=weight, pairs=pairs, tuples=tuples, regional=regional)
        )
    if not components:
        raise ValueError("multiscale_spectrum must enable at least one positive-weight component")
    if regions is not None and not any(item.regional for item in components):
        raise ValueError("multiscale_spectrum.regions is configured but no component is regional")
    return MultiscaleSettings(
        fields=fields,
        target_use=target_use,
        units=units,
        family_weight=float(family_weight),
        geometry=geometry,
        bands=bands,
        regions=regions,
        calibration=calibration,
        dtype=dtype,
        objective=objective,
        conditional_draws=draws,
        components=tuple(components),
        standardize_descriptors=standardize,
        **signed_band,
    )


def _signed_band_settings(
    objective_settings: Mapping[str, Any],
    objective: str,
    bands: BandSettings,
    fields: tuple[str, ...],
    standardize: bool,
) -> dict[str, Any]:
    """Resolve the signed_band_energy_score settings; other objectives must not set them."""
    present = sorted(SIGNED_BAND_KEYS & set(objective_settings))
    if objective != "signed_band_energy_score":
        if present:
            raise ValueError(
                f"multiscale_spectrum.objective keys {present} apply to signed_band_energy_score only"
            )
        return {}
    if standardize:
        raise ValueError(
            "signed_band_energy_score scores signed band fields with frozen band scales; "
            "standardize_descriptors must be false"
        )
    alpha = objective_settings.get("almost_fair_alpha", 0.95)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < float(alpha) <= 1:
        raise ValueError("multiscale_spectrum.objective.almost_fair_alpha must lie in (0, 1]")
    projections = str(objective_settings.get("projections", "axes_and_pairs"))
    if projections not in PROJECTIONS:
        raise ValueError(f"multiscale_spectrum.objective.projections must be one of {PROJECTIONS}")
    projection_pairs = None
    if "projection_pairs" in objective_settings:
        if projections != "axes_and_pairs":
            raise ValueError("objective.projection_pairs requires projections=axes_and_pairs")
        projection_pairs = _cospectrum_pairs(
            objective_settings["projection_pairs"],
            fields,
            "multiscale_spectrum.objective.projection_pairs",
        )
    weights = objective_settings.get("band_weights")
    if weights is not None:
        if (
            not isinstance(weights, (list, tuple))
            or len(weights) != len(bands.names)
            or any(
                isinstance(w, bool)
                or not isinstance(w, (int, float))
                or not math.isfinite(w)
                or w <= 0
                for w in weights
            )
        ):
            raise ValueError(
                "multiscale_spectrum.objective.band_weights needs one positive finite weight per band"
            )
        weights = tuple(float(w) for w in weights)
    auxiliary = {}
    for key in ("coherence_crps_weight", "joint_energy_score_weight"):
        value = objective_settings.get(key, 0.0)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"multiscale_spectrum.objective.{key} must be finite and >= 0")
        auxiliary[key] = float(value)
    if auxiliary["joint_energy_score_weight"] > 0 and float(alpha) != 1.0:
        raise ValueError(
            "joint_energy_score_weight > 0 requires almost_fair_alpha=1 for a proper total score"
        )
    score_version = str(objective_settings.get("score_definition_version", LEGACY_SCORE_VERSION))
    if score_version not in {LEGACY_SCORE_VERSION, MEAN_AUGMENTED_SCORE_VERSION}:
        raise ValueError("unknown multiscale_spectrum.objective.score_definition_version")
    if score_version == LEGACY_SCORE_VERSION and "mean_weight" in objective_settings:
        raise ValueError("objective.mean_weight applies to the mean-augmented score only")
    mean_weight = 1.0
    if score_version == MEAN_AUGMENTED_SCORE_VERSION:
        mean_weight = float(
            _positive_float(
                objective_settings.get("mean_weight", 1.0),
                "multiscale_spectrum.objective.mean_weight",
            )
        )
        if auxiliary["joint_energy_score_weight"] <= 0:
            raise ValueError("mean-augmented score requires joint_energy_score_weight > 0")
    return {
        "almost_fair_alpha": float(alpha),
        "projections": projections,
        "projection_pairs": projection_pairs,
        "band_weights": weights,
        "score_definition_version": score_version,
        "mean_weight": mean_weight,
        **auxiliary,
    }
