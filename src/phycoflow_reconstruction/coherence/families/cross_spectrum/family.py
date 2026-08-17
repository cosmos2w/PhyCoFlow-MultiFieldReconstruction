"""Graph cross-spectrum coherence family."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
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
from .basis import build_graph_basis, coordinate_digest
from .statistics import (
    band_energies,
    graph_fourier,
    normalized_cross_band_coupling,
    off_diagonal_pair_mean_square,
    pair_mean_square,
    spectral_coherence,
)


class CrossSpectrumFamily(nn.Module):
    family_name = "cross_spectrum"
    version = "1"

    def __init__(
        self,
        config: Mapping[str, Any],
        data_spec: DataSpec,
        normalizer: FieldNormalizer,
    ) -> None:
        super().__init__()
        self.config = dict(config)
        self.target_use = str(config.get("target_use", "paired_supervised"))
        self.units = str(config.get("units", "model_units"))
        self.family_weight = float(config.get("weight", 1.0))
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
        pair_names = config.get("pairs") or list(combinations(self.field_names, 2))
        try:
            self.pairs = tuple(
                (self.field_names.index(left), self.field_names.index(right))
                for left, right in pair_names
            )
        except ValueError as error:
            raise KeyError("cross-spectrum pair contains a field outside fields") from error
        if (
            not self.pairs
            or any(left == right for left, right in self.pairs)
            or len({tuple(sorted(pair)) for pair in self.pairs}) != len(self.pairs)
        ):
            raise ValueError("cross_spectrum requires unique pairs of distinct fields")
        self.register_buffer("normalization_offset", normalizer.offset.clone())
        self.register_buffer("normalization_scale", normalizer.scale.clone())
        self.register_buffer("eigenvalues", torch.empty(0), persistent=True)
        self.register_buffer("eigenvectors", torch.empty(0, 0), persistent=True)
        self.register_buffer("band_ids", torch.empty(0, dtype=torch.long), persistent=True)

        graph = config.get("graph", {})
        self.k_neighbors = int(graph.get("k_neighbors", 16))
        self.sigma = None if graph.get("sigma") is None else float(graph["sigma"])
        self.num_modes = int(graph.get("num_modes", 64))
        self.exclude_zero = bool(graph.get("exclude_zero", True))
        self.band_names = tuple(graph.get("bands", ("low", "mid", "high")))
        if not self.band_names or len(set(self.band_names)) != len(self.band_names):
            raise ValueError("cross-spectrum graph bands must be non-empty and unique")
        self.geometry_sha256: str | None = None
        self.resolved_sigma: float | None = None
        self.eps = float(config.get("eps", 1e-8))
        if self.eps <= 0:
            raise ValueError("cross_spectrum.eps must be positive")

        components = config.get("components", {})
        definitions = (
            ("same_frequency", "same_frequency.magnitude_squared", 2),
            ("cross_frequency", "cross_frequency.band_energy_coupling", 3),
            ("band_energy", "band_energy.log_power", 1),
        )
        self.component_weights: dict[str, float] = {}
        self.minimum_batch: dict[str, int] = {}
        specs = []
        for key, path, minimum in definitions:
            settings = components.get(key, {})
            default_enabled = key != "band_energy"
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
            )
            self.eigenvalues = basis.eigenvalues.to(coordinates.device)
            self.eigenvectors = basis.eigenvectors.to(coordinates.device)
            self.band_ids = basis.band_ids.to(coordinates.device)
            self.geometry_sha256 = basis.coordinate_sha256
            self.resolved_sigma = basis.sigma
        elif digest != self.geometry_sha256:
            raise ValueError("cross-spectrum coordinates changed after graph-basis construction")
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
        total = generated.sum() * 0.0
        if "same_frequency" in self.component_weights:
            loss = pair_mean_square(
                spectral_coherence(coefficients_generated, self.eps),
                spectral_coherence(coefficients_reference, self.eps),
                self.pairs,
            )
            results[f"{self.family_name}.same_frequency.magnitude_squared"] = TermResult(
                None, loss, diagnostics={"minimum_batch_size": 2}
            )
            total = total + self.component_weights["same_frequency"] * loss
        energies_generated = energies_reference = None
        if "cross_frequency" in self.component_weights or "band_energy" in self.component_weights:
            energies_generated = band_energies(coefficients_generated, self.band_ids)
            energies_reference = band_energies(coefficients_reference, self.band_ids)
        if "cross_frequency" in self.component_weights:
            assert energies_generated is not None and energies_reference is not None
            loss = off_diagonal_pair_mean_square(
                normalized_cross_band_coupling(energies_generated, self.eps),
                normalized_cross_band_coupling(energies_reference, self.eps),
                self.pairs,
            )
            results[f"{self.family_name}.cross_frequency.band_energy_coupling"] = TermResult(
                None, loss, diagnostics={"minimum_batch_size": 3}
            )
            total = total + self.component_weights["cross_frequency"] * loss
        if "band_energy" in self.component_weights:
            assert energies_generated is not None and energies_reference is not None
            loss = (
                (energies_generated.mean(dim=0) + self.eps).log()
                - (energies_reference.mean(dim=0) + self.eps).log()
            ).square().mean()
            results[f"{self.family_name}.band_energy.log_power"] = TermResult(None, loss)
            total = total + self.component_weights["band_energy"] * loss
        if not torch.isfinite(total):
            raise FloatingPointError("cross-spectrum family produced a non-finite cost")
        return FamilyResult(
            component_results=results,
            per_sample_cost=None,
            scalar_loss=total,
            diagnostics={
                "family": self.family_name,
                "version": self.version,
                "aggregation": "ensemble",
                "target_use": self.target_use,
                "units": self.units,
                "fields": self.field_names,
                "pairs": self.pairs,
                "bands": self.band_names,
                "geometry_sha256": self.geometry_sha256,
                "component_weights": dict(self.component_weights),
            },
        )

    def state_artifact(self) -> dict[str, Any]:
        return {
            "family": self.family_name,
            "version": self.version,
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
            "band_names": self.band_names,
            "state_dict": self.state_dict(),
        }

    def load_state_artifact(self, artifact: Mapping[str, Any]) -> None:
        if artifact.get("family") != self.family_name or artifact.get("version") != self.version:
            raise ValueError("cross-spectrum family artifact identity mismatch")
        if dict(artifact.get("config", {})) != self.config:
            raise ValueError("cross-spectrum family artifact config mismatch")
        state = artifact["state_dict"]
        device = self.normalization_offset.device
        self.eigenvalues = state["eigenvalues"].to(device)
        self.eigenvectors = state["eigenvectors"].to(device)
        self.band_ids = state["band_ids"].to(device)
        self.geometry_sha256 = artifact.get("geometry_sha256")
        self.resolved_sigma = artifact.get("resolved_sigma")
        if tuple(artifact.get("band_names", ())) != self.band_names:
            raise ValueError("cross-spectrum family artifact bands mismatch")
        self.load_state_dict(state, strict=True)
