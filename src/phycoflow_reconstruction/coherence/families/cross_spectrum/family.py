"""Graph cross-spectrum coherence family."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from itertools import combinations
from math import isfinite
from typing import Any

import torch
from torch import nn

from ....contracts import (
    CoherenceComponentSpec,
    CoherenceFamilySpec,
    DataSpec,
    FamilyResult,
    TermResult,
)
from ....data.normalization import FieldNormalizer
from ...base import require_field_tensor
from .basis import basis_digest, build_graph_basis, coordinate_digest, interval_band_ids
from .covariance_blocks import (
    covariance_band_energies,
    linear_coefficient_covariance,
    second_order_covariance_block_losses,
)
from .statistics import (
    auto_spectrum,
    auto_spectrum_mean_square_values,
    band_energies,
    graph_fourier,
    normalized_cross_band_coupling,
    off_diagonal_pair_mean_square_values,
    off_diagonal_pair_symmetric_coherence_scores,
    pair_mean_square_values,
    pair_symmetric_coherence_scores,
    spectral_coherence,
)


def _energy_calibration_digest(energies: torch.Tensor) -> str | None:
    if not energies.numel():
        return None
    array = energies.detach().to(device="cpu").contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


class CrossSpectrumFamily(nn.Module):
    family_name = "cross_spectrum"
    version = "3"
    second_order_version = "4"

    def __init__(
        self,
        config: Mapping[str, Any],
        data_spec: DataSpec,
        normalizer: FieldNormalizer,
    ) -> None:
        super().__init__()
        self.config = dict(config)
        self.definition = str(config.get("definition", "legacy_v3"))
        if self.definition not in {"legacy_v3", "second_order_blocks_v4"}:
            raise ValueError(
                "cross_spectrum.definition must be legacy_v3 or second_order_blocks_v4"
            )
        self.version = (
            self.second_order_version
            if self.definition == "second_order_blocks_v4"
            else type(self).version
        )
        self.target_use = str(config.get("target_use", "paired_supervised"))
        self.units = str(config.get("units", "model_units"))
        self.family_weight = float(config.get("weight", 1.0))
        if self.family_weight <= 0:
            raise ValueError("cross_spectrum.weight must be positive")
        if self.target_use not in {"training_reference", "paired_supervised"}:
            raise ValueError("cross_spectrum.target_use is invalid")
        if self.units not in {"model_units", "physical_units"}:
            raise ValueError("cross_spectrum.units is invalid")
        lookup = {name: index for index, name in enumerate(data_spec.field_names)}
        self.field_names = tuple(config.get("fields") or data_spec.field_names)
        if len(set(self.field_names)) != len(self.field_names):
            raise ValueError("cross-spectrum fields must be unique")
        unknown = sorted(set(self.field_names) - set(lookup))
        if unknown:
            raise KeyError(f"unknown cross-spectrum fields: {unknown}")
        self.field_ids = tuple(lookup[name] for name in self.field_names)
        configured_pairs = config.get("pairs")
        pair_names = (
            list(combinations(self.field_names, 2))
            if configured_pairs is None
            else list(configured_pairs)
        )
        try:
            self.pairs = tuple(
                (self.field_names.index(left), self.field_names.index(right))
                for left, right in pair_names
            )
        except ValueError as error:
            raise KeyError("cross-spectrum pair contains a field outside fields") from error
        if (
            any(left == right for left, right in self.pairs)
            or len({tuple(sorted(pair)) for pair in self.pairs}) != len(self.pairs)
        ):
            raise ValueError("cross_spectrum pairs must be unique and contain distinct fields")
        self.register_buffer("normalization_offset", normalizer.offset.clone())
        self.register_buffer("normalization_scale", normalizer.scale.clone())
        self.register_buffer("eigenvalues", torch.empty(0), persistent=True)
        self.register_buffer("eigenvectors", torch.empty(0, 0), persistent=True)
        self.register_buffer("band_ids", torch.empty(0, dtype=torch.long), persistent=True)
        self.register_buffer("calibrated_band_energies", torch.empty(0), persistent=False)
        self.calibration_ensemble_size: int | None = None
        self.calibration_sha256: str | None = None

        graph = config.get("graph", {})
        self.k_neighbors = int(graph.get("k_neighbors", 16))
        self.sigma = None if graph.get("sigma") is None else float(graph["sigma"])
        self.num_modes = int(graph.get("num_modes", 64))
        self.exclude_zero = bool(graph.get("exclude_zero", True))
        raw_intervals = graph.get("band_intervals")
        self.band_intervals: tuple[tuple[str, float, float], ...] | None = None
        self.degeneracy_tolerance = float(graph.get("degeneracy_tolerance", 1.0e-6))
        if raw_intervals is not None:
            if not isinstance(raw_intervals, (list, tuple)):
                raise TypeError("cross_spectrum.graph.band_intervals must be a sequence")
            parsed_intervals = []
            for index, item in enumerate(raw_intervals):
                if not isinstance(item, Mapping):
                    raise TypeError(
                        "cross_spectrum.graph.band_intervals entries must be mappings"
                    )
                if set(item) != {"name", "lower", "upper"}:
                    raise ValueError(
                        "each graph.band_intervals entry requires exactly name, lower, and upper"
                    )
                parsed_intervals.append(
                    (str(item["name"]), float(item["lower"]), float(item["upper"]))
                )
            self.band_intervals = tuple(parsed_intervals)
        configured_band_names = tuple(graph.get("bands", ("low", "mid", "high")))
        if self.definition == "second_order_blocks_v4":
            if self.band_intervals is None:
                raise ValueError(
                    "second_order_blocks_v4 requires explicit graph.band_intervals"
                )
            interval_names = tuple(interval[0] for interval in self.band_intervals)
            if "bands" in graph and configured_band_names != interval_names:
                raise ValueError(
                    "cross_spectrum.graph.bands must match graph.band_intervals names"
                )
            self.band_names = interval_names
            if not isfinite(self.degeneracy_tolerance) or self.degeneracy_tolerance < 0:
                raise ValueError("cross_spectrum.graph.degeneracy_tolerance must be non-negative")
        else:
            self.band_names = configured_band_names
        if not self.band_names or len(set(self.band_names)) != len(self.band_names):
            raise ValueError("cross-spectrum graph bands must be non-empty and unique")
        self.geometry_sha256: str | None = None
        self.resolved_sigma: float | None = None
        self.zero_mode_eigenvalue: float | None = None
        self.first_retained_eigengap: float | None = None
        self.eps = float(config.get("eps", 1e-8))
        if self.eps <= 0:
            raise ValueError("cross_spectrum.eps must be positive")

        minimum_ensemble_size = config.get("minimum_ensemble_size", 32)
        if isinstance(minimum_ensemble_size, bool) or int(minimum_ensemble_size) != minimum_ensemble_size:
            raise ValueError("cross_spectrum.minimum_ensemble_size must be an integer")
        self.minimum_ensemble_size = int(minimum_ensemble_size)
        if self.minimum_ensemble_size < 2:
            raise ValueError("cross_spectrum.minimum_ensemble_size must be at least 2")
        stabilization = config.get("stabilization", {})
        if not isinstance(stabilization, Mapping):
            raise TypeError("cross_spectrum.stabilization must be a mapping")
        unknown_stabilization = set(stabilization) - {
            "relative_floor",
            "absolute_floor",
            "minimum_reference_band_fraction",
        }
        if unknown_stabilization:
            raise ValueError(
                f"unknown cross_spectrum.stabilization keys: {sorted(unknown_stabilization)}"
            )
        self.relative_floor = float(stabilization.get("relative_floor", 1.0e-6))
        self.absolute_floor = float(stabilization.get("absolute_floor", 1.0e-12))
        self.minimum_reference_band_fraction = float(
            stabilization.get("minimum_reference_band_fraction", 1.0e-8)
        )
        if (
            not isfinite(self.relative_floor)
            or not isfinite(self.absolute_floor)
            or self.relative_floor < 0
            or self.absolute_floor < 0
        ):
            raise ValueError("cross_spectrum stabilization floors must be finite and non-negative")
        if self.relative_floor == 0 and self.absolute_floor == 0:
            raise ValueError("cross_spectrum requires a positive stabilization floor")
        if not 0 <= self.minimum_reference_band_fraction < 1:
            raise ValueError(
                "cross_spectrum.stabilization.minimum_reference_band_fraction must be in [0,1)"
            )

        self.basis_sha256: str | None = None

        components = config.get("components", {})
        if self.definition == "second_order_blocks_v4":
            for forbidden in ("self_spectrum", "band_energy"):
                settings = components.get(forbidden, {})
                if bool(settings.get("enabled", False)) and float(
                    settings.get("weight", 1.0)
                ) > 0:
                    raise ValueError(
                        f"second_order_blocks_v4 does not optimize {forbidden}"
                    )
            definitions = (
                ("self_spectrum", "self_spectrum.auto_spectrum", 1),
                (
                    "same_frequency",
                    "same_frequency.second_order_covariance_blocks",
                    self.minimum_ensemble_size,
                ),
                (
                    "cross_frequency",
                    "cross_frequency.second_order_covariance_blocks",
                    self.minimum_ensemble_size,
                ),
                ("band_energy", "band_energy.log_power", 1),
            )
        else:
            definitions = (
                ("self_spectrum", "self_spectrum.auto_spectrum", 1),
                ("same_frequency", "same_frequency.magnitude_squared", 2),
                ("cross_frequency", "cross_frequency.band_energy_coupling", 3),
                ("band_energy", "band_energy.log_power", 1),
            )
        self.component_weights: dict[str, float] = {}
        self.minimum_batch: dict[str, int] = {}
        specs = []
        for key, path, minimum in definitions:
            settings = components.get(key, {})
            default_enabled = key not in {"self_spectrum", "band_energy"}
            if bool(settings.get("enabled", default_enabled)):
                weight = float(settings.get("weight", 1.0))
                if weight < 0:
                    raise ValueError("cross-spectrum component weights must be non-negative")
                self.component_weights[key] = weight
                self.minimum_batch[key] = minimum
                specs.append(
                    CoherenceComponentSpec(
                        path,
                        self.target_use,
                        self.units,
                        True,
                        required_geometry="fixed_point_graph",
                        aggregation="ensemble",
                        metadata={"minimum_batch_size": minimum},
                    )
                )
        if not self.component_weights or not any(self.component_weights.values()):
            raise ValueError("cross_spectrum must enable a positive-weight component")
        if (
            self.component_weights.get("same_frequency", 0.0) > 0
            or self.component_weights.get("cross_frequency", 0.0) > 0
        ) and not self.pairs:
            raise ValueError(
                "same-frequency and cross-frequency coherence require at least one field pair"
            )
        if "cross_frequency" in self.component_weights and len(self.band_names) < 2:
            raise ValueError("cross-frequency coherence requires at least two bands")
        self.spec = CoherenceFamilySpec(
            self.family_name,
            self.version,
            tuple(specs),
            metadata={"aggregation": "ensemble", "target_use": self.target_use},
        )
        self.spec.validate()

    @property
    def required_batch_size(self) -> int:
        return max(
            minimum
            for key, minimum in self.minimum_batch.items()
            if self.component_weights[key] > 0
        )

    def _basis(self, coordinates: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if coordinates.ndim != 3:
            raise ValueError("cross-spectrum coordinates must have shape [B,N,D]")
        if not torch.equal(coordinates, coordinates[:1].expand_as(coordinates)):
            raise ValueError("cross-spectrum requires identical coordinates for every sample")
        digest = coordinate_digest(coordinates[0])
        if not self.eigenvectors.numel():
            basis = build_graph_basis(
                coordinates[0],
                k_neighbors=self.k_neighbors,
                sigma=self.sigma,
                num_modes=self.num_modes,
                band_names=self.band_names,
                exclude_zero=self.exclude_zero,
                band_intervals=(
                    self.band_intervals
                    if self.definition == "second_order_blocks_v4"
                    else None
                ),
                degeneracy_tolerance=self.degeneracy_tolerance,
            )
            self.eigenvalues = basis.eigenvalues.to(coordinates.device)
            self.eigenvectors = basis.eigenvectors.to(coordinates.device)
            self.band_ids = basis.band_ids.to(coordinates.device)
            self.geometry_sha256 = basis.coordinate_sha256
            self.resolved_sigma = basis.sigma
            self.zero_mode_eigenvalue = basis.zero_mode_eigenvalue
            self.first_retained_eigengap = basis.first_retained_eigengap
            self.basis_sha256 = basis_digest(
                self.eigenvalues,
                self.eigenvectors,
                self.band_ids,
                band_names=self.band_names,
                band_intervals=(
                    self.band_intervals
                    if self.definition == "second_order_blocks_v4"
                    else None
                ),
                geometry_sha256=self.geometry_sha256,
                exclude_zero=self.exclude_zero,
            )
        elif digest != self.geometry_sha256:
            raise ValueError("cross-spectrum coordinates changed after graph-basis construction")
        elif self.basis_sha256 is not None:
            current_digest = basis_digest(
                self.eigenvalues,
                self.eigenvectors,
                self.band_ids,
                band_names=self.band_names,
                band_intervals=(
                    self.band_intervals
                    if self.definition == "second_order_blocks_v4"
                    else None
                ),
                geometry_sha256=str(self.geometry_sha256),
                exclude_zero=self.exclude_zero,
            )
            if current_digest != self.basis_sha256:
                raise ValueError("cross-spectrum graph basis changed after artifact validation")
        return self.eigenvectors.to(device=coordinates.device, dtype=dtype)

    def _units_and_fields(
        self, generated: torch.Tensor, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.units == "physical_units":
            offset = self.normalization_offset.to(generated)
            scale = self.normalization_scale.to(generated)
            generated = generated * scale + offset
            reference = reference * scale + offset
        return generated[..., self.field_ids], reference[..., self.field_ids]

    @torch.no_grad()
    def freeze_reference_calibration(
        self,
        reference: torch.Tensor,
        coordinates: torch.Tensor,
        *,
        replace: bool = False,
    ) -> None:
        """Freeze reference-only band energies for v4 masks and stability floors.

        The input must be a fixed reference panel with the same aligned point
        geometry used by forward calls. Replacing an existing calibration is
        explicit so training batches cannot silently change the floor policy.
        """
        if self.definition != "second_order_blocks_v4":
            raise ValueError("reference energy calibration is available only for v4")
        require_field_tensor("reference calibration", reference)
        if coordinates.ndim != 3 or coordinates.shape[:2] != reference.shape[:2]:
            raise ValueError("reference calibration coordinates must align with [B,N]")
        if reference.shape[0] < self.minimum_ensemble_size:
            raise ValueError(
                "reference calibration panel is smaller than minimum_ensemble_size"
            )
        _, selected_reference = self._units_and_fields(reference, reference)
        basis = self._basis(coordinates, reference.dtype)
        coefficients = graph_fourier(selected_reference, basis)
        covariance = linear_coefficient_covariance(coefficients, ddof=1)
        energies = covariance_band_energies(covariance, self.band_ids).detach()
        if self.calibrated_band_energies.numel() and not replace:
            if torch.allclose(
                self.calibrated_band_energies.to(energies),
                energies,
                atol=1.0e-12,
                rtol=1.0e-7,
            ):
                return
            raise ValueError("reference covariance calibration is already frozen")
        self.calibrated_band_energies = energies
        self.calibration_ensemble_size = int(reference.shape[0])
        self.calibration_sha256 = _energy_calibration_digest(energies)

    def _same_frequency_epsilon_diagnostics(self, coefficients: torch.Tensor) -> dict[str, float]:
        auto = coefficients.abs().square().mean(dim=0)
        denominator = torch.einsum("ki,kj->kij", auto, auto)
        return {
            "denominator_min": float(denominator.min().detach()),
            "epsilon_dominated_fraction": float((denominator <= self.eps).float().mean()),
        }

    def _cross_frequency_epsilon_diagnostics(self, energies: torch.Tensor) -> dict[str, float]:
        centered = energies - energies.mean(dim=0, keepdim=True)
        covariance = torch.einsum("bmi,bnj->mnij", centered, centered) / energies.shape[0]
        variances = torch.stack(
            [torch.diagonal(covariance[index, index]) for index in range(covariance.shape[0])]
        )
        denominator = torch.einsum("mi,nj->mnij", variances, variances)
        return {
            "denominator_min": float(denominator.min().detach()),
            "epsilon_dominated_fraction": float((denominator <= self.eps).float().mean()),
        }

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
            raise ValueError("cross-spectrum generated/reference shapes differ")
        if coordinates is None or coordinates.shape[:2] != generated.shape[:2]:
            raise ValueError("cross-spectrum requires coordinates aligned with [B,N]")
        if generated.shape[0] < self.required_batch_size:
            raise ValueError(
                f"cross-spectrum enabled components require batch_size>={self.required_batch_size}"
            )
        generated, reference = self._units_and_fields(generated, reference)
        basis = self._basis(coordinates, generated.dtype)
        coefficients_generated = graph_fourier(generated, basis)
        coefficients_reference = graph_fourier(reference, basis)
        results: dict[str, TermResult] = {}
        component_diagnostics: dict[str, dict[str, Any]] = {}
        total = generated.sum() * 0.0
        block_diagnostics: dict[str, Any] | None = None
        if self.definition == "second_order_blocks_v4" and (
            self.component_weights.get("same_frequency", 0.0) > 0
            or self.component_weights.get("cross_frequency", 0.0) > 0
        ):
            phase = context.get("phase") if isinstance(context, Mapping) else None
            if phase == "train" and not self.calibrated_band_energies.numel():
                raise ValueError(
                    "second_order_blocks_v4 training requires frozen reference calibration"
                )
            block_values_requested = phase in {"evaluation", "calibration", "audit"}
            block_losses = second_order_covariance_block_losses(
                coefficients_generated,
                coefficients_reference,
                self.band_ids,
                self.pairs,
                relative_floor=self.relative_floor,
                absolute_floor=self.absolute_floor,
                minimum_reference_band_fraction=self.minimum_reference_band_fraction,
                include_block_values=block_values_requested,
                calibration_reference_energies=(
                    self.calibrated_band_energies
                    if self.calibrated_band_energies.numel()
                    else None
                ),
                calibration_ensemble_size=self.calibration_ensemble_size,
                ddof=1,
            )
            block_diagnostics = block_losses.diagnostics
            for key, scalar_loss, path in (
                (
                    "same_frequency",
                    block_losses.same_frequency,
                    f"{self.family_name}.same_frequency.second_order_covariance_blocks",
                ),
                (
                    "cross_frequency",
                    block_losses.cross_frequency,
                    f"{self.family_name}.cross_frequency.second_order_covariance_blocks",
                ),
            ):
                weight = self.component_weights.get(key, 0.0)
                if weight <= 0:
                    continue
                eligible_name = f"{key}_eligible_blocks"
                skipped_name = f"{key}_skipped_blocks"
                results[path] = TermResult(
                    None,
                    scalar_loss,
                    diagnostics={
                        "definition": self.definition,
                        "covariance_ddof": 1,
                        "ensemble_size": int(generated.shape[0]),
                        "normalization": "signed_complex_frobenius_squared_sum_per_block",
                        "eligible_blocks": block_diagnostics[eligible_name],
                        "skipped_blocks": block_diagnostics[skipped_name],
                        "blocks": block_diagnostics[
                            "same_frequency_blocks"
                            if key == "same_frequency"
                            else "cross_frequency_blocks"
                        ],
                    },
                )
                component_diagnostics[path] = {
                    "weight": weight,
                    "executed": True,
                    "raw_scalar_loss": float(scalar_loss.detach()),
                    "weighted_scalar_contribution": float((weight * scalar_loss).detach()),
                    "eligible_blocks": block_diagnostics[eligible_name],
                    "skipped_blocks": block_diagnostics[skipped_name],
                }
                total = total + weight * scalar_loss
        if self.component_weights.get("self_spectrum", 0.0) > 0:
            generated_auto = auto_spectrum(coefficients_generated)
            reference_auto = auto_spectrum(coefficients_reference)
            per_field = auto_spectrum_mean_square_values(
                coefficients_generated, coefficients_reference
            )
            loss = per_field.mean()
            path = f"{self.family_name}.self_spectrum.auto_spectrum"
            weight = self.component_weights["self_spectrum"]
            results[path] = TermResult(
                None,
                loss,
                diagnostics={
                    "minimum_batch_size": 1,
                    "per_field_mean_square": per_field.detach().cpu().tolist(),
                    "generated_auto_spectrum": generated_auto.detach().cpu().tolist(),
                    "reference_auto_spectrum": reference_auto.detach().cpu().tolist(),
                },
            )
            component_diagnostics[path] = {
                "weight": weight,
                "executed": True,
                "raw_scalar_loss": float(loss.detach()),
                "weighted_scalar_contribution": float((weight * loss).detach()),
            }
            total = total + weight * loss
        if (
            self.definition == "legacy_v3"
            and self.component_weights.get("same_frequency", 0.0) > 0
        ):
            generated_coherence = spectral_coherence(coefficients_generated, self.eps)
            reference_coherence = spectral_coherence(coefficients_reference, self.eps)
            per_pair = pair_mean_square_values(generated_coherence, reference_coherence, self.pairs)
            per_pair_score = pair_symmetric_coherence_scores(
                generated_coherence, reference_coherence, self.pairs, self.eps
            )
            loss = per_pair.mean()
            path = f"{self.family_name}.same_frequency.magnitude_squared"
            weight = self.component_weights["same_frequency"]
            results[path] = TermResult(
                None,
                loss,
                diagnostics={
                    "minimum_batch_size": 2,
                    "epsilon": self.eps,
                    "generated": self._same_frequency_epsilon_diagnostics(coefficients_generated),
                    "reference": self._same_frequency_epsilon_diagnostics(coefficients_reference),
                    "per_pair_mean_square": per_pair.detach().cpu().tolist(),
                    "per_pair_coherence_score": per_pair_score.detach().cpu().tolist(),
                },
            )
            component_diagnostics[path] = {
                "weight": weight,
                "executed": True,
                "raw_scalar_loss": float(loss.detach()),
                "weighted_scalar_contribution": float((weight * loss).detach()),
            }
            total = total + weight * loss
        energies_generated = energies_reference = None
        if self.definition == "legacy_v3" and (
            self.component_weights.get("cross_frequency", 0.0) > 0
            or self.component_weights.get("band_energy", 0.0) > 0
        ):
            energies_generated = band_energies(coefficients_generated, self.band_ids)
            energies_reference = band_energies(coefficients_reference, self.band_ids)
        if (
            self.definition == "legacy_v3"
            and self.component_weights.get("cross_frequency", 0.0) > 0
        ):
            assert energies_generated is not None and energies_reference is not None
            generated_coupling = normalized_cross_band_coupling(energies_generated, self.eps)
            reference_coupling = normalized_cross_band_coupling(energies_reference, self.eps)
            per_pair = off_diagonal_pair_mean_square_values(
                generated_coupling, reference_coupling, self.pairs
            )
            per_pair_score = off_diagonal_pair_symmetric_coherence_scores(
                generated_coupling, reference_coupling, self.pairs, self.eps
            )
            loss = per_pair.mean()
            path = f"{self.family_name}.cross_frequency.band_energy_coupling"
            weight = self.component_weights["cross_frequency"]
            results[path] = TermResult(
                None,
                loss,
                diagnostics={
                    "minimum_batch_size": 3,
                    "epsilon": self.eps,
                    "generated": self._cross_frequency_epsilon_diagnostics(energies_generated),
                    "reference": self._cross_frequency_epsilon_diagnostics(energies_reference),
                    "per_pair_mean_square": per_pair.detach().cpu().tolist(),
                    "per_pair_coherence_score": per_pair_score.detach().cpu().tolist(),
                },
            )
            component_diagnostics[path] = {
                "weight": weight,
                "executed": True,
                "raw_scalar_loss": float(loss.detach()),
                "weighted_scalar_contribution": float((weight * loss).detach()),
            }
            total = total + weight * loss
        if (
            self.definition == "legacy_v3"
            and self.component_weights.get("band_energy", 0.0) > 0
        ):
            assert energies_generated is not None and energies_reference is not None
            per_band_field = (
                (energies_generated.mean(dim=0) + self.eps).log()
                - (energies_reference.mean(dim=0) + self.eps).log()
            ).square()
            loss = per_band_field.mean()
            path = f"{self.family_name}.band_energy.log_power"
            weight = self.component_weights["band_energy"]
            results[path] = TermResult(
                None,
                loss,
                diagnostics={"per_band_field_mean_square": per_band_field.detach().cpu().tolist()},
            )
            component_diagnostics[path] = {
                "weight": weight,
                "executed": True,
                "raw_scalar_loss": float(loss.detach()),
                "weighted_scalar_contribution": float((weight * loss).detach()),
            }
            total = total + weight * loss
        component_paths = {
            "self_spectrum": f"{self.family_name}.self_spectrum.auto_spectrum",
            "same_frequency": (
                f"{self.family_name}.same_frequency.second_order_covariance_blocks"
                if self.definition == "second_order_blocks_v4"
                else f"{self.family_name}.same_frequency.magnitude_squared"
            ),
            "cross_frequency": (
                f"{self.family_name}.cross_frequency.second_order_covariance_blocks"
                if self.definition == "second_order_blocks_v4"
                else f"{self.family_name}.cross_frequency.band_energy_coupling"
            ),
            "band_energy": f"{self.family_name}.band_energy.log_power",
        }
        for key, path in component_paths.items():
            if self.component_weights.get(key, 0.0) == 0.0:
                component_diagnostics[path] = {
                    "weight": 0.0,
                    "executed": False,
                    "raw_scalar_loss": None,
                    "weighted_scalar_contribution": None,
                }
        if not torch.isfinite(total):
            raise FloatingPointError("cross-spectrum family produced a non-finite cost")
        return FamilyResult(
            component_results=results,
            per_sample_cost=None,
            scalar_loss=total,
            diagnostics={
                "family": self.family_name,
                "version": self.version,
                "definition": self.definition,
                "aggregation": "ensemble",
                "target_use": self.target_use,
                "units": self.units,
                "fields": self.field_names,
                "pairs": self.pairs,
                "bands": self.band_names,
                "band_mode_ids": self.band_ids.detach().cpu().tolist(),
                "band_intervals": self.band_intervals,
                "eigenvalues": self.eigenvalues.detach().cpu().tolist(),
                "exclude_zero": self.exclude_zero,
                "zero_mode_eigenvalue": self.zero_mode_eigenvalue,
                "first_retained_eigengap": self.first_retained_eigengap,
                "minimum_retained_eigengap": (
                    None
                    if self.eigenvalues.numel() < 2
                    else float(torch.diff(self.eigenvalues).min().detach())
                ),
                "resolved_sigma": self.resolved_sigma,
                "epsilon": self.eps,
                "geometry_sha256": self.geometry_sha256,
                "basis_sha256": self.basis_sha256,
                "calibration_sha256": self.calibration_sha256,
                "calibration_ensemble_size": self.calibration_ensemble_size,
                "minimum_ensemble_size": self.minimum_ensemble_size,
                "stabilization": {
                    "relative_floor": self.relative_floor,
                    "absolute_floor": self.absolute_floor,
                    "minimum_reference_band_fraction": self.minimum_reference_band_fraction,
                },
                "covariance": block_diagnostics,
                "component_weights": dict(self.component_weights),
                "components": component_diagnostics,
            },
        )

    def state_artifact(self) -> dict[str, Any]:
        return {
            "family": self.family_name,
            "version": self.version,
            "definition": self.definition,
            "upstream": {
                "repository": "https://github.com/ctrl-is/PhyCoFlowModel-Cross-Spectral-Coherence",
                "revision": "add1b1a6422c",
                "license": "MIT",
            },
            "config": self.config,
            "field_names": self.field_names,
            "target_use": self.target_use,
            "units": self.units,
            "geometry_sha256": self.geometry_sha256,
            "resolved_sigma": self.resolved_sigma,
            "zero_mode_eigenvalue": self.zero_mode_eigenvalue,
            "first_retained_eigengap": self.first_retained_eigengap,
            "exclude_zero": self.exclude_zero,
            "band_names": self.band_names,
            "band_intervals": (
                None
                if self.band_intervals is None
                else [
                    {"name": name, "lower": lower, "upper": upper}
                    for name, lower, upper in self.band_intervals
                ]
            ),
            "basis_sha256": self.basis_sha256,
            "calibrated_band_energies": (
                self.calibrated_band_energies.detach().to(device="cpu")
                if self.calibrated_band_energies.numel()
                else None
            ),
            "calibration_ensemble_size": self.calibration_ensemble_size,
            "calibration_sha256": self.calibration_sha256,
            "state_dict": self.state_dict(),
        }

    def load_state_artifact(self, artifact: Mapping[str, Any]) -> None:
        if not isinstance(artifact, Mapping):
            raise TypeError("cross-spectrum family artifact must be a mapping")
        if artifact.get("family") != self.family_name or artifact.get("version") != self.version:
            raise ValueError("cross-spectrum family artifact identity mismatch")
        artifact_definition = artifact.get("definition", "legacy_v3")
        if artifact_definition != self.definition:
            raise ValueError("cross-spectrum family artifact definition mismatch")
        if dict(artifact.get("config", {})) != self.config:
            raise ValueError("cross-spectrum family artifact config mismatch")
        state = artifact.get("state_dict")
        if not isinstance(state, Mapping):
            raise TypeError("cross-spectrum family artifact state_dict must be a mapping")
        required_state = {"eigenvalues", "eigenvectors", "band_ids"}
        if not required_state <= set(state):
            raise ValueError("cross-spectrum family artifact is missing graph-basis tensors")
        eigenvalues = state["eigenvalues"]
        eigenvectors = state["eigenvectors"]
        band_ids = state["band_ids"]
        if not all(isinstance(value, torch.Tensor) for value in (eigenvalues, eigenvectors, band_ids)):
            raise TypeError("cross-spectrum graph-basis states must be tensors")
        has_basis = eigenvectors.numel() > 0
        if has_basis:
            if eigenvalues.ndim != 1 or eigenvectors.ndim != 2 or band_ids.ndim != 1:
                raise ValueError("cross-spectrum graph-basis tensor ranks are invalid")
            if eigenvalues.shape[0] != eigenvectors.shape[1] or band_ids.shape != eigenvalues.shape:
                raise ValueError("cross-spectrum graph-basis tensor shapes do not align")
            if eigenvalues.shape[0] != self.num_modes:
                raise ValueError("cross-spectrum artifact mode count does not match config")
            if not torch.isfinite(eigenvalues).all() or not torch.isfinite(eigenvectors).all():
                raise FloatingPointError("cross-spectrum graph-basis artifact contains non-finite values")
            if torch.any(torch.diff(eigenvalues) < -1.0e-7):
                raise ValueError("cross-spectrum artifact eigenvalues are not sorted")
            if not band_ids.dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
                raise TypeError("cross-spectrum artifact band_ids must be integer tensors")
            if int(band_ids.min()) < 0 or int(band_ids.max()) >= len(self.band_names):
                raise ValueError("cross-spectrum artifact band_ids are outside configured bands")
            if any(not torch.any(band_ids == index) for index in range(len(self.band_names))):
                raise ValueError("cross-spectrum artifact contains an empty spectral band")
            orthogonality = eigenvectors.T @ eigenvectors
            identity = torch.eye(
                orthogonality.shape[0],
                dtype=orthogonality.dtype,
                device=orthogonality.device,
            )
            if float((orthogonality - identity).abs().max()) > 1.0e-3:
                raise ValueError("cross-spectrum artifact eigenvectors are not orthonormal")
        elif self.definition == "second_order_blocks_v4":
            raise ValueError("second_order_blocks_v4 artifact must contain a realized graph basis")

        stored_intervals = artifact.get("band_intervals")
        if self.definition == "second_order_blocks_v4":
            try:
                normalized_stored_intervals = tuple(
                    (str(item["name"]), float(item["lower"]), float(item["upper"]))
                    for item in stored_intervals
                )
            except (TypeError, KeyError, ValueError) as error:
                raise ValueError("second_order_blocks_v4 artifact is missing valid band intervals") from error
            if normalized_stored_intervals != self.band_intervals:
                raise ValueError("cross-spectrum family artifact band intervals mismatch")
            expected_ids, expected_names = interval_band_ids(
                eigenvalues.detach().cpu().numpy(),
                self.band_intervals or (),
                self.degeneracy_tolerance,
            )
            if tuple(expected_names) != self.band_names or not torch.equal(
                band_ids.cpu(), torch.from_numpy(expected_ids).to(dtype=band_ids.dtype)
            ):
                raise ValueError("cross-spectrum artifact band membership disagrees with intervals")

        stored_basis_sha256 = artifact.get("basis_sha256")
        if has_basis:
            geometry_sha256 = artifact.get("geometry_sha256")
            if not isinstance(geometry_sha256, str) or len(geometry_sha256) != 64:
                raise ValueError("cross-spectrum artifact has no valid coordinate fingerprint")
            calculated_basis_sha256 = basis_digest(
                eigenvalues,
                eigenvectors,
                band_ids,
                band_names=self.band_names,
                band_intervals=(
                    self.band_intervals
                    if self.definition == "second_order_blocks_v4"
                    else None
                ),
                geometry_sha256=geometry_sha256,
                exclude_zero=self.exclude_zero,
            )
            if self.definition == "second_order_blocks_v4" and not stored_basis_sha256:
                raise ValueError("second_order_blocks_v4 artifact must persist basis_sha256")
            if stored_basis_sha256 is not None and stored_basis_sha256 != calculated_basis_sha256:
                raise ValueError("cross-spectrum family artifact basis hash mismatch")

        stored_calibration = artifact.get("calibrated_band_energies")
        stored_calibration_size = artifact.get("calibration_ensemble_size")
        stored_calibration_sha256 = artifact.get("calibration_sha256")
        if stored_calibration is None:
            if stored_calibration_size is not None or stored_calibration_sha256 is not None:
                raise ValueError("cross-spectrum artifact has incomplete calibration provenance")
            self.calibrated_band_energies = self.normalization_offset.new_empty(0)
            self.calibration_ensemble_size = None
            self.calibration_sha256 = None
        else:
            if self.definition != "second_order_blocks_v4":
                raise ValueError("legacy cross-spectrum artifacts cannot carry v4 calibration")
            if not isinstance(stored_calibration, torch.Tensor):
                raise TypeError("cross-spectrum calibration energies must be a tensor")
            if stored_calibration.shape != (len(self.band_names), len(self.field_names)):
                raise ValueError("cross-spectrum calibration energy shape does not match bands/fields")
            if not torch.isfinite(stored_calibration).all() or torch.any(stored_calibration < 0):
                raise ValueError("cross-spectrum calibration energies must be finite and non-negative")
            if (
                isinstance(stored_calibration_size, bool)
                or not isinstance(stored_calibration_size, int)
                or stored_calibration_size < self.minimum_ensemble_size
            ):
                raise ValueError("cross-spectrum calibration ensemble size is invalid")
            calculated_calibration_sha256 = _energy_calibration_digest(stored_calibration)
            if not stored_calibration_sha256 or (
                stored_calibration_sha256 != calculated_calibration_sha256
            ):
                raise ValueError("cross-spectrum calibration hash mismatch")
            self.calibrated_band_energies = stored_calibration.to(
                device=self.normalization_offset.device,
                dtype=self.normalization_offset.dtype,
            )
            self.calibration_ensemble_size = stored_calibration_size
            self.calibration_sha256 = stored_calibration_sha256

        device = self.normalization_offset.device
        self.eigenvalues = eigenvalues.to(device)
        self.eigenvectors = eigenvectors.to(device)
        self.band_ids = band_ids.to(device)
        self.geometry_sha256 = artifact.get("geometry_sha256")
        self.resolved_sigma = artifact.get("resolved_sigma")
        self.zero_mode_eigenvalue = artifact.get("zero_mode_eigenvalue")
        self.first_retained_eigengap = artifact.get("first_retained_eigengap")
        if tuple(artifact.get("band_names", ())) != self.band_names:
            raise ValueError("cross-spectrum family artifact bands mismatch")
        self.basis_sha256 = stored_basis_sha256
        self.load_state_dict(state, strict=True)
