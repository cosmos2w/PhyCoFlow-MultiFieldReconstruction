"""Spectral coherence on frozen geometry and TRAIN-only scales (artifact version 1).

Descriptor objectives score band powers, signed co-spectra and optional nonlinear
cross-scale moments. Signed-field objectives retain spatial structure, with an
optional fair joint score for cross-field and cross-band distributional identity.
The family orchestrates geometry, calibration, score kernels and diagnostics;
their definitions live in the corresponding focused modules.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import torch
from torch import nn

from ....contracts import (
    CoherenceComponentSpec,
    CoherenceFamilySpec,
    DataSpec,
    FamilyResult,
    TermResult,
)
from ....data.manifest import dataset_fingerprint
from ....data.normalization import FieldNormalizer
from ...base import require_field_tensor
from .config import (
    MEAN_AUGMENTED_SCORE_VERSION,
    PAIRED_MEAN_DISTANCE_VERSION,
    parse_multiscale_config,
)
from .features import (
    ReferenceAccumulator,
    descriptor_blocks,
    descriptor_layout,
    factor_manifest,
    raw_statistics,
)
from .losses import (
    conditional_energy_score,
    field_projections,
    paired_descriptor_loss,
)
from .objectives import (
    BAND_COHERENCE_PATH,
    ENERGY_SCORE_PATH,
    JOINT_ENERGY_PATH,
    MEAN_JOINT_ENERGY_PATH,
    PAIRED_BAND_PATH,
    PAIRED_MEAN_JOINT_PATH,
    SIGNED_BAND_PATH,
    paired_signed_band_terms,
    signed_band_terms,
)
from .operators import (
    active_axes,
    coordinate_digest,
    fem_p1_operator,
    knn_graph_operator,
    lonlat_axis_lengths,
    region_measures,
    spectral_basis,
    tight_filter_bank,
)
from .reporting import band_ensemble_report, spectral_report
from .statistics import band_fields

GEOMETRY_BUFFERS = {
    "mass": 1,
    "eigenvalues": 1,
    "basis": 2,
    "filters": 2,
    "measures": 2,
    "piece_measures": 2,
}
INDEX_BUFFERS = ("analysis_index", "piece_index")
CALIBRATION_BUFFERS = {
    "fluctuation_reference": 1,
    "power_reference": 2,
    "power_floor": 2,
    "band_scale": 2,
    "factor_power_reference": 2,
    "factor_scale": 2,
}
CALIBRATION_BATCH = 32
# Standardized descriptor variances are floored at this fraction of their block's mean,
# so a nearly constant descriptor cannot take an unbounded weight.
DESCRIPTOR_VARIANCE_FLOOR = 1.0e-3
SCIENTIFIC_SOURCE_FILES = (
    "config.py",
    "family.py",
    "features.py",
    "losses.py",
    "objectives.py",
    "operators.py",
    "reporting.py",
    "statistics.py",
)


def _implementation_identity() -> dict[str, Any]:
    """Bind v4's frozen artifacts to the scoring implementation and numeric backends."""
    directory = Path(__file__).resolve().parent
    return {
        "format": "scientific_sources_sha256_and_backends_v1",
        "sources": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in SCIENTIFIC_SOURCE_FILES
        },
        "backends": {
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "scipy": str(scipy.__version__),
        },
    }


def _plain(value: Any) -> Any:
    """Convert numpy containers and scalars to types accepted by weights_only loading."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


class MultiscaleSpectrumFamily(nn.Module):
    family_name = "multiscale_spectrum"
    version = "1"
    requires_full_grid_calibration = True
    supports_report_detail = True
    # The report accepts every available independent draw (``context["ensemble_draws"]``)
    # for per-band member/mean/spread power diagnostics.
    supports_ensemble_report = True

    def __init__(
        self,
        config: Mapping[str, Any],
        data_spec: DataSpec,
        normalizer: FieldNormalizer,
    ) -> None:
        super().__init__()
        if data_spec.reconstruction_unit != "snapshot":
            raise ValueError("multiscale_spectrum v1 analyzes snapshot states only")
        self.config = dict(config)
        self.settings = parse_multiscale_config(config, data_spec.field_names)
        settings = self.settings
        self.target_use = settings.target_use
        self.units = settings.units
        self.family_weight = settings.family_weight
        self.objective = settings.objective
        self.conditional_draws = settings.conditional_draws
        lookup = {name: index for index, name in enumerate(data_spec.field_names)}
        self.field_names = settings.fields
        self.field_ids = tuple(lookup[name] for name in settings.fields)
        self.input_data_contract = {
            "field_names": list(data_spec.field_names),
            "field_units": list(data_spec.field_units),
            "reconstruction_unit": data_spec.reconstruction_unit,
        }
        self.compute_dtype = torch.float64 if settings.dtype == "float64" else torch.float32
        self.mean_augmented = settings.score_definition_version in {
            MEAN_AUGMENTED_SCORE_VERSION,
            PAIRED_MEAN_DISTANCE_VERSION,
        }
        self.manifest = factor_manifest(settings)
        self.component_weights = {
            component.key: component.weight for component in settings.components
        }
        self.register_buffer("normalization_offset", normalizer.offset.clone())
        self.register_buffer("normalization_scale", normalizer.scale.clone())
        for name, ndim in {**GEOMETRY_BUFFERS, **CALIBRATION_BUFFERS}.items():
            self.register_buffer(name, torch.empty((0,) * ndim, dtype=self.compute_dtype))
        for name in INDEX_BUFFERS:
            self.register_buffer(name, torch.empty(0, dtype=torch.long))
        if self.mean_augmented:
            # Legacy artifacts have no extra key; this buffer belongs only to v4.
            self.register_buffer("piece_mean_scale", torch.empty((0, 0), dtype=self.compute_dtype))
        self.standardize_descriptors = settings.standardize_descriptors
        if self.standardize_descriptors:
            self.register_buffer("descriptor_variance", torch.empty(0, dtype=self.compute_dtype))
        self.geometry_record: dict[str, Any] | None = None
        self.calibration_record: dict[str, Any] | None = None
        self.layout = None
        trained_blocks = self.objective == "paired_descriptor_mse"
        self.signed_band = self.objective == "signed_band_energy_score"
        self.paired_band = self.objective == "paired_signed_band_mean_distance"
        self._projection_cache: dict[tuple[torch.dtype, torch.device], torch.Tensor] = {}
        specs = [
            CoherenceComponentSpec(
                f"{component.key}.squared_error",
                self.target_use,
                self.units,
                True,
                required_geometry="fixed_point_operator",
                aggregation="per_sample",
                metadata={"minimum_batch_size": 1, "trained": trained_blocks},
            )
            for component in settings.components
        ]
        if not trained_blocks and not self.paired_band:
            specs.append(
                CoherenceComponentSpec(
                    ENERGY_SCORE_PATH,
                    self.target_use,
                    self.units,
                    True,
                    required_geometry="fixed_point_operator",
                    aggregation="per_sample",
                    metadata={
                        "minimum_batch_size": 1,
                        "trained": not self.signed_band,
                        "conditional_draws": self.conditional_draws,
                    },
                )
            )
        if self.signed_band:
            signed_specs = [
                (SIGNED_BAND_PATH, True),
                (BAND_COHERENCE_PATH, settings.coherence_crps_weight > 0),
            ]
            if settings.joint_energy_score_weight > 0:
                signed_specs.append(
                    (MEAN_JOINT_ENERGY_PATH if self.mean_augmented else JOINT_ENERGY_PATH, True)
                )
            for path, trained in signed_specs:
                specs.append(
                    CoherenceComponentSpec(
                        path,
                        self.target_use,
                        self.units,
                        True,
                        required_geometry="fixed_point_operator",
                        aggregation="per_sample",
                        metadata={
                            "minimum_batch_size": 1,
                            "trained": trained,
                            "conditional_draws": self.conditional_draws,
                        },
                    )
                )
        if self.paired_band:
            for path in (PAIRED_BAND_PATH, PAIRED_MEAN_JOINT_PATH):
                specs.append(
                    CoherenceComponentSpec(
                        path,
                        self.target_use,
                        self.units,
                        True,
                        required_geometry="fixed_point_operator",
                        aggregation="per_sample",
                        metadata={"minimum_batch_size": 1, "trained": True, "paired_point": True},
                    )
                )
        self.spec = CoherenceFamilySpec(
            self.family_name,
            self.version,
            tuple(specs),
            metadata={"aggregation": "per_sample", "objective": self.objective},
        )
        self.spec.validate()

    @property
    def fitted(self) -> bool:
        return self.geometry_record is not None and self.calibration_record is not None

    @property
    def region_names(self) -> tuple[str, ...]:
        return tuple((self.geometry_record or {}).get("region_names", ()))

    # ------------------------------------------------------------------ geometry
    @staticmethod
    def _per_active_axis(values: tuple, dimension: int, axes: tuple[int, ...], name: str) -> tuple:
        """Accept one entry per coordinate axis or per active (nonzero-span) axis."""
        if len(values) == dimension:
            return tuple(values[axis] for axis in axes)
        if len(values) == len(axes):
            return tuple(values)
        raise ValueError(
            f"{name} needs one entry per coordinate axis ({dimension}) or per active axis "
            f"({len(axes)}; active axes {list(axes)})"
        )

    def _axis_lengths(
        self,
        dimension: int,
        axes: tuple[int, ...],
        raw_minimum: np.ndarray | None,
        raw_maximum: np.ndarray | None,
        raw_mean_latitude: float | None,
    ) -> np.ndarray:
        """Physical lengths of the loader's per-axis normalized unit interval, per active axis."""
        scaling = self.settings.geometry.coordinate_scaling
        if isinstance(scaling, tuple):
            lengths = np.asarray(
                self._per_active_axis(scaling, dimension, axes, "geometry.coordinate_scaling"),
                dtype=np.float64,
            )
        elif scaling == "none":
            lengths = np.ones(len(axes))
        elif raw_minimum is None or raw_maximum is None:
            raise ValueError(
                f"coordinate_scaling={scaling} needs the raw coordinate extent; pass it or set "
                "geometry.coordinate_scaling explicitly"
            )
        elif scaling == "raw_extent":
            extent = np.asarray(raw_maximum, dtype=np.float64) - np.asarray(
                raw_minimum, dtype=np.float64
            )
            lengths = extent[list(axes)]
        else:
            if axes != (0, 1):
                raise ValueError(
                    "lonlat_degrees requires active axes (longitude, latitude) = (0, 1)"
                )
            if raw_mean_latitude is None or not -90.0 < raw_mean_latitude < 90.0:
                raise ValueError(
                    "lonlat_degrees requires a mean latitude strictly within (-90, 90)"
                )
            lengths = np.asarray(
                lonlat_axis_lengths(
                    np.asarray(raw_minimum), np.asarray(raw_maximum), float(raw_mean_latitude)
                )
            )
        if np.any(lengths <= 0) or not np.all(np.isfinite(lengths)):
            raise ValueError("coordinate scaling produced a non-positive active axis length")
        return lengths

    def _build_geometry(
        self,
        coordinates: np.ndarray,
        *,
        raw_minimum: np.ndarray | None,
        raw_maximum: np.ndarray | None,
        raw_mean_latitude: float | None,
        device: torch.device,
        support_coordinates: np.ndarray | None = None,
        grid_indices: np.ndarray | None = None,
    ) -> None:
        geometry, bands = self.settings.geometry, self.settings.bands
        dimension = coordinates.shape[1]
        axes = active_axes(coordinates)
        if geometry.support_factor is not None and support_coordinates is None:
            raise ValueError("geometry.support_factor needs the dataset's full point set")
        if geometry.triangulation == "structured_indices" and grid_indices is None:
            raise ValueError("triangulation=structured_indices needs per-point grid indices")
        if len(axes) == 1 and (
            geometry.max_edge_factor is not None
            or geometry.support_factor is not None
            or geometry.triangulation != "delaunay"
        ):
            raise ValueError(
                "max_edge_factor, support_factor, and structured_indices build two-dimensional "
                "meshes; a one-axis domain uses the sorted-interval P1 mesh"
            )
        lengths = self._axis_lengths(dimension, axes, raw_minimum, raw_maximum, raw_mean_latitude)
        active_periodic = (
            (geometry.periodic,) * len(axes)
            if isinstance(geometry.periodic, bool)
            else tuple(
                bool(flag)
                for flag in self._per_active_axis(
                    geometry.periodic, dimension, axes, "geometry.periodic"
                )
            )
        )
        normalized_periods = None
        if any(active_periodic):
            normalized_periods = np.asarray(
                self._per_active_axis(geometry.periods, dimension, axes, "geometry.periods"),
                dtype=np.float64,
            )
        points = coordinates[:, list(axes)] * lengths
        periods = None if normalized_periods is None else normalized_periods * lengths
        seed = self.settings.calibration.seed
        physical = geometry.operator == "fem_p1"
        if physical:
            operator = fem_p1_operator(
                points,
                periodic=active_periodic,
                periods=periods,
                max_edge_factor=geometry.max_edge_factor,
                spacing_neighbors=geometry.spacing_neighbors,
                seed=seed,
                grid_indices=grid_indices,
                support_points=(
                    None
                    if geometry.support_factor is None
                    else support_coordinates[:, list(axes)] * lengths
                ),
                support_factor=geometry.support_factor,
                support_neighbors=geometry.support_neighbors,
                connectivity=geometry.connectivity,
                min_component_fraction=geometry.min_component_fraction,
            )
            spacing = float(operator.diagnostics["nearest_spacing_p90"])
            cutoff_wavelength = bands.min_wavelength or bands.resolution_factor * spacing
            if bands.wavelengths is not None and bands.wavelengths[-1] <= cutoff_wavelength:
                raise ValueError(
                    f"finest band wavelength {bands.wavelengths[-1]:.6g} does not exceed the "
                    f"resolution cutoff {cutoff_wavelength:.6g} {geometry.length_unit} "
                    "(resolution_factor x p90 nearest spacing); coarsen the bands or set "
                    "bands.min_wavelength"
                )
            cutoff_eigenvalue = (2.0 * math.pi / cutoff_wavelength) ** 2
        else:
            operator = knn_graph_operator(
                points,
                k_neighbors=geometry.k_neighbors,
                sigma=geometry.sigma,
                periodic=active_periodic,
                periods=periods,
            )
            cutoff_wavelength = None
            cutoff_eigenvalue = None
        basis = spectral_basis(
            operator,
            max_modes=geometry.max_modes,
            cutoff_eigenvalue=cutoff_eigenvalue,
            seed=seed,
        )
        wavenumbers = np.sqrt(basis.eigenvalues)
        # Graph scales end at the finest retained mode; physical scales at the resolution cutoff.
        cutoff = math.sqrt(cutoff_eigenvalue) if physical else float(wavenumbers[-1])
        if bands.wavelengths is not None:
            centers = 2.0 * math.pi / np.asarray(bands.wavelengths, dtype=np.float64)
            if centers[0] < wavenumbers[0] * (1 - 1e-9):
                raise ValueError(
                    f"coarsest band wavelength {bands.wavelengths[0]:.6g} exceeds the first "
                    f"nonconstant mode wavelength {2 * math.pi / wavenumbers[0]:.6g}"
                )
        else:
            if not wavenumbers[0] < cutoff:
                raise ValueError("automatic bands need at least two distinct retained scales")
            centers = np.geomspace(wavenumbers[0], cutoff, bands.count + 1)[:-1]
        filters = tight_filter_bank(wavenumbers, centers, cutoff)
        effective = np.square(filters).sum(axis=1)
        empty = [name for name, value in zip(bands.names, effective) if value <= 1e-12]
        if empty:
            raise ValueError(f"bands {empty} contain no retained spectral mode")
        measures = operator.mass[None, :]
        region_names: tuple[str, ...] = ()
        region_record = None
        if any(component.regional for component in self.settings.components):
            grid = self._per_active_axis(
                self.settings.regions.grid, dimension, axes, "regions.grid"
            )
            regional_measures, region_names, region_record = region_measures(
                coordinates[operator.nodes][:, list(axes)],
                operator.mass,
                grid=tuple(int(count) for count in grid),
                periodic=active_periodic,
                periods=normalized_periods,
                min_mass_fraction=self.settings.regions.min_mass_fraction,
            )
            measures = np.concatenate([measures, regional_measures])
        pieces = operator.components
        piece_measures = np.zeros((int(pieces.max()) + 1, pieces.size))
        piece_measures[pieces, np.arange(pieces.size)] = operator.mass
        options = {"dtype": self.compute_dtype, "device": device}
        self.analysis_index = torch.as_tensor(operator.nodes, dtype=torch.long, device=device)
        self.piece_index = torch.as_tensor(pieces, dtype=torch.long, device=device)
        self.piece_measures = torch.as_tensor(piece_measures, **options)
        self.mass = torch.as_tensor(operator.mass, **options)
        self.eigenvalues = torch.as_tensor(basis.eigenvalues, **options)
        self.basis = torch.as_tensor(basis.eigenvectors, **options)
        self.filters = torch.as_tensor(filters, **options)
        self.measures = torch.as_tensor(measures, **options)
        resolved = np.square(filters).sum(axis=0)
        self.geometry_record = _plain(
            {
                "coordinate_sha256": coordinate_digest(coordinates),
                "points": int(coordinates.shape[0]),
                "analysis_points": int(operator.nodes.size),
                "excluded_points": int(coordinates.shape[0] - operator.nodes.size),
                "analysis_components": int(piece_measures.shape[0]),
                "triangulation": geometry.triangulation if physical else None,
                "coordinate_dim": int(dimension),
                "active_axes": list(axes),
                "coordinate_scaling": (
                    list(geometry.coordinate_scaling)
                    if isinstance(geometry.coordinate_scaling, tuple)
                    else geometry.coordinate_scaling
                ),
                "active_axis_lengths": lengths.tolist(),
                "length_unit": geometry.length_unit,
                "active_periodic": [bool(flag) for flag in active_periodic],
                "active_periods": None if periods is None else periods.tolist(),
                "operator": operator.kind,
                "operator_diagnostics": operator.diagnostics,
                "basis": basis.diagnostics,
                "scale_interpretation": "physical_wavenumber" if physical else "graph_scale",
                "cutoff_wavelength": cutoff_wavelength,
                "cutoff_wavenumber": cutoff,
                "first_mode_wavelength": float(2 * math.pi / wavenumbers[0]) if physical else None,
                "band_names": list(bands.names),
                "band_center_wavenumbers": centers.tolist(),
                "band_center_wavelengths": (2 * math.pi / centers).tolist() if physical else None,
                "band_effective_modes": effective.tolist(),
                "band_mode_counts": (filters > 1e-12).sum(axis=1).tolist(),
                "tapered_modes": int((resolved < 1 - 1e-12).sum()),
                "region_names": list(region_names),
                "regions": region_record,
            }
        )
        self.layout = descriptor_layout(self.settings, region_names)

    def _geometry_tensors(self, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        return {
            "mass": self.mass.to(dtype),
            "basis": self.basis.to(dtype),
            "filters": self.filters.to(dtype),
            "piece_measures": self.piece_measures.to(dtype),
            "pieces": self.piece_index,
        }

    # --------------------------------------------------------------- calibration
    def fit_reference_tensors(
        self,
        coordinates: torch.Tensor,
        states: torch.Tensor | Iterable[torch.Tensor],
        *,
        sample_ids: Iterable[str],
        raw_minimum: np.ndarray | None = None,
        raw_maximum: np.ndarray | None = None,
        raw_mean_latitude: float | None = None,
        support_coordinates: torch.Tensor | None = None,
        grid_indices: torch.Tensor | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        """Fix the geometry and frozen reference scales from training states [S,N,C].

        ``raw_minimum``/``raw_maximum`` are the raw coordinate extents behind the
        loader's per-axis normalization (``coordinate_scaling``);
        ``support_coordinates`` is the dataset's full normalized point set (for
        ``geometry.support_factor``); ``grid_indices`` holds the (i, j) grid
        index of each reference point (for ``triangulation=structured_indices``).
        """
        if self.fitted:
            raise RuntimeError("multiscale_spectrum reference statistics are already frozen")
        if coordinates.ndim != 2:
            raise ValueError("reference coordinates must have shape [N,D]")
        device = self.normalization_offset.device

        def as_numpy(value: torch.Tensor | None) -> np.ndarray | None:
            return None if value is None else value.detach().to("cpu", torch.float64).numpy()

        self._build_geometry(
            as_numpy(coordinates),
            raw_minimum=raw_minimum,
            raw_maximum=raw_maximum,
            raw_mean_latitude=raw_mean_latitude,
            device=device,
            support_coordinates=as_numpy(support_coordinates),
            grid_indices=as_numpy(grid_indices),
        )
        accumulator = ReferenceAccumulator(
            len(self.field_names),
            len(self.settings.bands.names),
            len(self.manifest.factors),
            self.measures.shape[0],
            device,
            pieces=self.piece_measures.shape[0] if self.mean_augmented else None,
        )
        geometry = self._geometry_tensors(torch.float64)
        measures = self.measures.to(torch.float64)
        batches = [states] if isinstance(states, torch.Tensor) else states
        if self.standardize_descriptors:
            # Kept for a second pass that measures descriptors under the frozen scales.
            batches = list(batches)
        with torch.no_grad():
            for batch in batches:
                require_field_tensor("reference states", batch)
                if batch.shape[1] != coordinates.shape[0]:
                    raise ValueError("reference states do not align with the reference coordinates")
                fields = self._select(batch.to(device)).to(torch.float64)
                accumulator.update(
                    raw_statistics(
                        fields,
                        manifest=self.manifest,
                        with_piece_means=self.mean_augmented,
                        relation_pairs=getattr(self, "coverage_pairs", None),
                        **geometry,
                    ),
                    measures,
                )
        calibration = accumulator.finalize(
            self.manifest,
            band_relative_floor=self.settings.calibration.band_relative_floor,
            absolute_floor=self.settings.calibration.absolute_floor,
            field_names=self.field_names,
            field_unit_floor=self.settings.calibration.field_unit_floor,
        )
        for name, value in calibration.as_dict().items():
            setattr(self, name, value.to(dtype=self.compute_dtype, device=device))
        # A nominally positive floor can underflow in the declared precision.
        if self.settings.calibration.field_unit_floor is not None and (
            not torch.isfinite(self.power_floor).all() or not (self.power_floor > 0).all()
        ):
            raise ValueError(
                "field_unit_floor is not representable in the configured compute precision"
            )
        if self.mean_augmented:
            self.piece_mean_scale = accumulator.mean_scales(
                self.settings.calibration.field_unit_floor
            ).to(dtype=self.compute_dtype, device=device)
            if (
                not torch.isfinite(self.piece_mean_scale).all()
                or not (self.piece_mean_scale > 0).all()
            ):
                raise ValueError(
                    "piece-mean scales must be positive in the configured compute precision"
                )
        variance = (
            self._fit_descriptor_variance(batches, device) if self.standardize_descriptors else None
        )
        if variance is not None:
            self.descriptor_variance = variance.to(dtype=self.compute_dtype, device=device)
        identifiers = [str(item) for item in sample_ids]
        if len(identifiers) != accumulator.count:
            raise ValueError("reference sample IDs must align with the calibration states")
        self.calibration_record = _plain(
            {
                "source": "training_only",
                "states": accumulator.count,
                "sample_ids": identifiers,
                "sample_ids_sha256": hashlib.sha256("\n".join(identifiers).encode()).hexdigest(),
                "band_relative_floor": self.settings.calibration.band_relative_floor,
                "absolute_floor": self.settings.calibration.absolute_floor,
                "units": self.units,
                "fields": list(self.field_names),
                "reference_fluctuation_power": calibration.fluctuation_reference.tolist(),
                "reference_band_power": calibration.power_reference.tolist(),
                "reference_resolved_fraction": accumulator.resolved_fraction.tolist(),
                "prediction_dependent_denominators": False,
                **(
                    {"field_unit_floor": list(self.settings.calibration.field_unit_floor)}
                    if self.settings.calibration.field_unit_floor is not None
                    else {}
                ),
                **(
                    {
                        "score_definition_version": self.settings.score_definition_version,
                        "piece_mean_scale": self.piece_mean_scale.tolist(),
                        "piece_mean_scale_estimator": "sqrt(train_population_variance + field_unit_floor^2)",
                        "piece_mass_fractions": (
                            self.piece_measures.sum(1) / self.piece_measures.sum()
                        ).tolist(),
                        "mean_weight": self.settings.mean_weight,
                    }
                    if self.mean_augmented
                    else {}
                ),
                **({"descriptor_variance": variance.tolist()} if variance is not None else {}),
                **dict(provenance or {}),
            }
        )

    def _fit_descriptor_variance(
        self, batches: Iterable[torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        """Across-state variance of each descriptor under the frozen scales, floored per block."""
        values = []
        with torch.no_grad():
            for batch in batches:
                blocks = self._descriptors(self._statistics(batch.to(device)))
                values.append(torch.cat([blocks[key] for key in self.layout.keys], dim=1))
        variance = torch.cat(values).to(torch.float64).var(dim=0, unbiased=True)
        if not bool((variance > 0).all()):
            raise ValueError(
                "standardize_descriptors needs every descriptor to vary across training states"
            )
        floors = torch.cat(
            [
                torch.full_like(block, float(block.mean()) * DESCRIPTOR_VARIANCE_FLOOR)
                for block in variance.split(self.layout.counts)
            ]
        )
        return torch.maximum(variance, floors)

    def fit_training_reference(
        self, dataset, *, query_indices: torch.Tensor, device: torch.device | None = None
    ) -> None:
        """Fit geometry and scales from deterministic training-split samples only."""
        if (
            getattr(dataset, "split_name", None) != "train"
            or getattr(getattr(dataset, "selection", None), "split", None) != "train"
        ):
            raise ValueError(
                "multiscale reference calibration is permitted only on the training split"
            )
        if device is not None:
            self.to(device)
        query = torch.as_tensor(query_indices, dtype=torch.long).cpu()
        settings = self.settings.calibration
        generator = torch.Generator(device="cpu").manual_seed(settings.seed)
        indices = torch.randperm(len(dataset), generator=generator)[
            : min(settings.samples, len(dataset))
        ].tolist()
        first = dataset[indices[0]]
        coordinates = first.coordinates[query]
        raw = first.coordinates_raw.to(torch.float64)
        grid_indices = None
        name = self.settings.geometry.structured_indices
        if name is not None:
            auxiliary = first.metadata.get("auxiliary", {})
            if name not in auxiliary:
                raise KeyError(
                    f"structured_indices array {name!r} is not a per-point auxiliary array; "
                    f"available: {sorted(auxiliary)}"
                )
            grid_indices = auxiliary[name][query]
        sample_ids: list[str] = []

        def batches() -> Iterable[torch.Tensor]:
            buffer = []
            for position, index in enumerate(indices):
                sample = first if position == 0 else dataset[index]
                if not torch.equal(sample.coordinates[query], coordinates):
                    raise ValueError("training samples do not share one query geometry")
                if sample.valid_points is not None and not bool(sample.valid_points[query].all()):
                    raise ValueError("multiscale calibration queries contain invalid points")
                suffix = sample.time_index if sample.time_index is not None else "all"
                sample_ids.append(f"{sample.trajectory_id}:{suffix}")
                buffer.append(sample.values[query])
                if len(buffer) == CALIBRATION_BATCH:
                    yield torch.stack(buffer)
                    buffer = []
            if buffer:
                yield torch.stack(buffer)

        provenance = {
            "split": "train",
            "seed": settings.seed,
            "requested_samples": settings.samples,
            "query_points": int(query.numel()),
            "query_indices_sha256": hashlib.sha256(query.numpy().tobytes()).hexdigest(),
            "dataset_path": str(getattr(dataset, "path", "")),
            "normalization": {
                "method": getattr(dataset.normalizer, "method", None),
                "offset": dataset.normalizer.offset.tolist(),
                "scale": dataset.normalizer.scale.tolist(),
            },
        }
        if getattr(dataset, "path", None) is not None:
            provenance["dataset_fingerprint"] = dataset_fingerprint(dataset.path)
        self.fit_reference_tensors(
            coordinates,
            batches(),
            sample_ids=sample_ids,
            raw_minimum=raw.amin(dim=0).numpy(),
            raw_maximum=raw.amax(dim=0).numpy(),
            raw_mean_latitude=float(raw[:, 1].mean()) if raw.shape[1] > 1 else None,
            support_coordinates=first.coordinates,
            grid_indices=grid_indices,
            provenance=provenance,
        )

    # ------------------------------------------------------------- descriptors
    def _select(self, fields: torch.Tensor) -> torch.Tensor:
        # Cast first: a physical offset can dwarf small band amplitudes in float32.
        fields = fields.to(self.compute_dtype)
        if self.units == "physical_units":
            offset = self.normalization_offset.to(fields)
            scale = self.normalization_scale.to(fields)
            fields = fields * scale + offset
        # Points outside the analysis domain (see geometry.connectivity) never enter a statistic.
        return fields[..., self.field_ids].index_select(1, self.analysis_index)

    def _statistics(self, fields: torch.Tensor) -> dict[str, torch.Tensor]:
        return raw_statistics(
            self._select(fields),
            manifest=self.manifest,
            with_piece_means=self.mean_augmented,
            relation_pairs=getattr(self, "coverage_pairs", None),
            **self._geometry_tensors(self.compute_dtype),
        )

    def _projections(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        key = (dtype, device)
        if key not in self._projection_cache:
            self._projection_cache[key] = field_projections(
                len(self.field_names),
                self.settings.projections,
                pairs=(
                    list(self.coverage_pairs) if hasattr(self, "coverage_pairs") else (
                        None if self.settings.projection_pairs is None
                        else list(self.settings.projection_pairs))
                ),
                dtype=dtype,
                device=device,
            )
        return self._projection_cache[key]

    def _band_weights(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        weights = self.settings.band_weights
        count = len(self.settings.bands.names)
        values = (
            torch.ones(count, dtype=dtype, device=device)
            if weights is None
            else torch.tensor(weights, dtype=dtype, device=device)
        )
        return values / values.sum()

    def _descriptors(self, statistics: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        blocks = descriptor_blocks(
            statistics,
            {name: getattr(self, name) for name in CALIBRATION_BUFFERS},
            settings=self.settings,
            manifest=self.manifest,
            measures=self.measures,
        )
        for key, value in blocks.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"multiscale_spectrum {key} descriptors are non-finite")
        return blocks

    def descriptors(self, fields: torch.Tensor, *, coordinates: torch.Tensor) -> torch.Tensor:
        """Concatenated per-state descriptor vectors [B,D] in layout order."""
        self._require_ready(fields, coordinates)
        with torch.autocast(device_type=fields.device.type, enabled=False):
            blocks = self._descriptors(self._statistics(fields))
        return torch.cat([blocks[key] for key in self.layout.keys], dim=1)

    def _require_ready(self, fields: torch.Tensor, coordinates: torch.Tensor | None) -> None:
        if not self.fitted:
            raise RuntimeError(
                "multiscale_spectrum needs frozen training-reference statistics; call "
                "fit_training_reference before evaluation"
            )
        if (
            coordinates is None
            or coordinates.ndim != 3
            or coordinates.shape[:2] != fields.shape[:2]
        ):
            raise ValueError("multiscale_spectrum requires coordinates aligned with [B,N]")
        if not torch.equal(coordinates, coordinates[:1].expand_as(coordinates)):
            raise ValueError("multiscale_spectrum requires identical coordinates for every sample")
        digest = coordinate_digest(coordinates[0].detach().to("cpu", torch.float64).numpy())
        if digest != self.geometry_record["coordinate_sha256"]:
            raise ValueError("multiscale_spectrum coordinates differ from the fitted geometry")

    def forward(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor,
        *,
        coordinates: torch.Tensor | None = None,
        context: Any | None = None,
    ) -> FamilyResult:
        require_field_tensor("generated", generated)
        require_field_tensor("reference", reference)
        if generated.shape != reference.shape:
            raise ValueError("multiscale_spectrum generated/reference shapes differ")
        self._require_ready(generated, coordinates)
        context = context if isinstance(context, Mapping) else {}
        score_all_draws = context.get("score_all_conditional_draws", False)
        if not isinstance(score_all_draws, bool):
            raise TypeError("score_all_conditional_draws must be boolean")
        if score_all_draws and self.objective in {
            "paired_descriptor_mse",
            "paired_signed_band_mean_distance",
        }:
            raise ValueError("score_all_conditional_draws requires a conditional energy score")
        draws: tuple[torch.Tensor, ...] = ()
        if self.objective in {"conditional_energy_score", "signed_band_energy_score"}:
            supplied = tuple(context.get("conditional_draws", ()))
            if len(supplied) < self.conditional_draws - 1:
                raise ValueError(
                    f"{self.objective} needs {self.conditional_draws - 1} additional "
                    "independent draws for the same observations in context['conditional_draws']"
                )
            draws = supplied if score_all_draws else supplied[: self.conditional_draws - 1]
            for draw in draws:
                require_field_tensor("conditional draw", draw)
                if draw.shape != generated.shape:
                    raise ValueError("conditional draws must match the first draw shape")
        with torch.autocast(device_type=generated.device.type, enabled=False):
            with torch.no_grad():
                target_statistics = self._statistics(reference)
                target_blocks = self._descriptors(target_statistics)
            prediction_statistics = self._statistics(generated)
            prediction_blocks = self._descriptors(prediction_statistics)
            draw_statistics = [self._statistics(draw) for draw in draws]
            draw_blocks = [self._descriptors(statistics) for statistics in draw_statistics]
        keys = self.layout.keys
        weights = self.layout.weight_vector(dtype=self.compute_dtype, device=generated.device)
        if self.standardize_descriptors:
            weights = weights / self.descriptor_variance.to(weights)
        target = torch.cat([target_blocks[key] for key in keys], dim=1)
        first = torch.cat([prediction_blocks[key] for key in keys], dim=1)
        trained_blocks = self.objective == "paired_descriptor_mse"
        results: dict[str, TermResult] = {}
        component_diagnostics: dict[str, dict[str, Any]] = {}
        for key, count in zip(keys, self.layout.counts):
            error = (prediction_blocks[key] - target_blocks[key]).square().mean(dim=1)
            path = f"{self.family_name}.{key}.squared_error"
            weight = self.component_weights[key]
            diagnostics = {
                "descriptor_count": count,
                "descriptor_weight": weight,
                "statistic": "mean squared descriptor error of the first draw",
            }
            if not trained_blocks:
                diagnostics["evaluation_only"] = True
            results[path] = TermResult(error, error.mean(), diagnostics=diagnostics)
            raw = float(error.mean().detach())
            component_diagnostics[path] = {
                "weight": weight if trained_blocks else 0.0,
                "executed": True,
                "raw_scalar_loss": raw,
                "weighted_scalar_contribution": weight * raw if trained_blocks else 0.0,
            }
        if trained_blocks:
            per_sample = paired_descriptor_loss(first, target, weights)
        elif not self.paired_band:
            stacked = torch.stack(
                [
                    first,
                    *[torch.cat([blocks[key] for key in keys], dim=1) for blocks in draw_blocks],
                ],
                dim=1,
            )
            per_sample, attraction, repulsion = conditional_energy_score(stacked, target, weights)
            path = f"{self.family_name}.{ENERGY_SCORE_PATH}"
            energy_diagnostics = {
                "conditional_draws": len(draws) + 1,
                "descriptor_dimension": self.layout.dimension,
                "attraction": float(attraction.mean().detach()),
                "repulsion": float(repulsion.mean().detach()),
            }
            if self.signed_band:
                # Reported for comparability with objective v1; it carries no gradient here.
                per_sample = per_sample.detach()
                energy_diagnostics["evaluation_only"] = True
            results[path] = TermResult(
                per_sample, per_sample.mean(), diagnostics=energy_diagnostics
            )
            raw = float(per_sample.mean().detach())
            component_diagnostics[path] = {
                "weight": 0.0 if self.signed_band else 1.0,
                "executed": True,
                "raw_scalar_loss": raw,
                "weighted_contribution": 0.0 if self.signed_band else raw,
                "weighted_scalar_contribution": 0.0 if self.signed_band else raw,
            }
        signed_report = None
        if self.signed_band:
            with torch.autocast(device_type=generated.device.type, enabled=False):
                signed = signed_band_terms(
                    prediction_statistics,
                    draw_statistics,
                    target_statistics,
                    settings=self.settings,
                    mass=self.mass,
                    filters=self.filters,
                    band_scale=self.band_scale,
                    power_floor=self.power_floor,
                    projections=self._projections(self.compute_dtype, generated.device),
                    band_weights=self._band_weights(self.compute_dtype, generated.device),
                    piece_measures=self.piece_measures if self.mean_augmented else None,
                    mean_scale=self.piece_mean_scale if self.mean_augmented else None,
                    diagnostic_context=context.get("round7_scale_components"),
                )
            per_sample, signed_report = signed.per_sample_cost, signed.report
            results.update(signed.component_results)
            component_diagnostics.update(signed.component_diagnostics)
        elif self.paired_band:
            with torch.autocast(device_type=generated.device.type, enabled=False):
                paired = paired_signed_band_terms(
                    prediction_statistics,
                    target_statistics,
                    settings=self.settings,
                    mass=self.mass,
                    filters=self.filters,
                    band_scale=self.band_scale,
                    projections=self._projections(self.compute_dtype, generated.device),
                    band_weights=self._band_weights(self.compute_dtype, generated.device),
                    piece_measures=self.piece_measures,
                    mean_scale=self.piece_mean_scale,
                )
            per_sample = paired.per_sample_cost
            results.update(paired.component_results)
            component_diagnostics.update(paired.component_diagnostics)
        total = per_sample.mean()
        if not torch.isfinite(total):
            raise FloatingPointError("multiscale_spectrum produced a non-finite cost")
        diagnostics = {
            "family": self.family_name,
            "version": self.version,
            "aggregation": "per_sample",
            "target_use": self.target_use,
            "units": self.units,
            "fields": self.field_names,
            "objective": self.objective,
            "conditional_draws": len(draws) + 1,
            "bands": self.settings.bands.names,
            "band_center_wavelengths": self.geometry_record["band_center_wavelengths"],
            "length_unit": self.geometry_record["length_unit"],
            "operator": self.geometry_record["operator"],
            "scale_interpretation": self.geometry_record["scale_interpretation"],
            "geometry_sha256": self.geometry_record["coordinate_sha256"],
            "descriptor_dimension": self.layout.dimension,
            "component_weights": dict(self.component_weights),
            "components": component_diagnostics,
        }
        if signed_report is not None:
            diagnostics["signed_band"] = signed_report
        if self.paired_band:
            diagnostics["paired_signed_band"] = paired.report
        if bool(context.get("report_detail", False)):
            diagnostics["report"] = spectral_report(
                prediction_statistics,
                target_statistics,
                prediction_blocks,
                target_blocks,
                power_floor=self.power_floor,
                field_names=self.field_names,
                band_names=self.settings.bands.names,
                layout=self.layout,
            )
            if self.signed_band:
                diagnostics["report"]["signed_band"] = signed_report
                ensemble = tuple(context.get("ensemble_draws", ()))
                if ensemble:

                    def spatial_bands(statistics):
                        return band_fields(
                            statistics["coefficients"],
                            self.basis.to(self.compute_dtype),
                            self.filters.to(self.compute_dtype),
                        )

                    diagnostics["report"]["band_ensemble"] = band_ensemble_report(
                        (
                            spatial_bands(statistics)
                            for statistics in (
                                prediction_statistics,
                                *(self._statistics(draw) for draw in ensemble),
                            )
                        ),
                        spatial_bands(target_statistics),
                        mass=self.mass.to(self.compute_dtype),
                        count=1 + len(ensemble),
                        band_names=self.settings.bands.names,
                        field_names=self.field_names,
                    )
        return FamilyResult(
            component_results=results,
            per_sample_cost=per_sample,
            scalar_loss=total,
            diagnostics=diagnostics,
        )

    # --------------------------------------------------------------- artifacts
    def state_artifact(self) -> dict[str, Any]:
        artifact = {
            "family": self.family_name,
            "version": self.version,
            "config": self.config,
            "field_names": self.field_names,
            "target_use": self.target_use,
            "units": self.units,
            "objective": self.objective,
            "geometry": self.geometry_record,
            "calibration": self.calibration_record,
            "descriptor_names": None
            if self.layout is None
            else [list(names) for names in self.layout.names],
            "state_dict": self.state_dict(),
        }
        if self.mean_augmented:
            artifact["implementation_identity"] = _implementation_identity()
            artifact["input_data_contract"] = self.input_data_contract
        return artifact

    def load_state_artifact(self, artifact: Mapping[str, Any]) -> None:
        if artifact.get("family") != self.family_name or artifact.get("version") != self.version:
            raise ValueError("multiscale_spectrum family artifact identity mismatch")
        if dict(artifact.get("config", {})) != self.config:
            raise ValueError("multiscale_spectrum family artifact config mismatch")
        if self.mean_augmented and artifact.get("input_data_contract") != self.input_data_contract:
            raise ValueError("multiscale_spectrum mean-augmented input field order/units mismatch")
        if (
            self.mean_augmented
            and artifact.get("implementation_identity") != _implementation_identity()
        ):
            from .round7_source_compatibility import accepts_prior_scale_identity
            if not accepts_prior_scale_identity(artifact.get("implementation_identity"), _implementation_identity()):
                raise ValueError(
                    "multiscale_spectrum mean-augmented implementation/backend identity mismatch"
                )
        state = artifact["state_dict"]
        device = self.normalization_offset.device
        for name in {**GEOMETRY_BUFFERS, **CALIBRATION_BUFFERS}:
            setattr(self, name, state[name].to(device=device, dtype=self.compute_dtype))
        for name in INDEX_BUFFERS:
            setattr(self, name, state[name].to(device=device, dtype=torch.long))
        if self.mean_augmented:
            self.piece_mean_scale = state["piece_mean_scale"].to(
                device=device, dtype=self.compute_dtype
            )
        if self.standardize_descriptors:
            self.descriptor_variance = state["descriptor_variance"].to(
                device=device, dtype=self.compute_dtype
            )
        self.load_state_dict(state, strict=True)
        self.geometry_record = (
            None if artifact.get("geometry") is None else dict(artifact["geometry"])
        )
        self.calibration_record = (
            None if artifact.get("calibration") is None else dict(artifact["calibration"])
        )
        self.layout = (
            None
            if self.geometry_record is None
            else descriptor_layout(self.settings, tuple(self.geometry_record["region_names"]))
        )
