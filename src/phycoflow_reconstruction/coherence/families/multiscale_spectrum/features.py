"""Descriptor layout, frozen training-reference scales, and descriptor assembly.

The second-order part of the descriptor is one object: the band Gram matrix
``G^m_ij = <F_m q_i, F_m q_j>_M``. Its diagonal (auto spectrum) and selected
off-diagonal entries (signed co-spectra) get different fixed transforms:

- ``log_band_power``: ``log(1 + P_mi / delta_mi)`` with ``P_mi = G^m_ii``;
- ``signed_band_cospectrum``: ``asinh(G^m_ij / (s_mi s_mj))``.

The smooth band filters overlap, so cross-band linear statistics need not
vanish. The optional third-order block measures nonlinear cross-scale structure:

- ``third_order_product_moments``: ``asinh`` of third central moments of
  band fields, each divided by its frozen (regional) scale.

Floors are ``delta_mi = rho_band Pbar_mi + rho_abs Vbar_i`` and scales are
``s_mi = sqrt(Pbar_mi + delta_mi)``, where ``Pbar`` and ``Vbar`` (total
fluctuation power) are training means; regional scales use regional powers
with the same absolute floor. All are frozen; no denominator depends on a
prediction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from .config import MultiscaleSettings
from .statistics import (
    band_fields,
    band_gram,
    mean_square,
    piecewise_fluctuations,
    piecewise_means,
    regional_centered,
    regional_variance,
    spectral_coefficients,
    third_central_moments,
)


@dataclass(frozen=True)
class FactorManifest:
    """Unique (field, band) series used by the third-order block and their index triples."""

    factors: tuple[tuple[int, int], ...]
    triples: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class DescriptorLayout:
    keys: tuple[str, ...]
    names: tuple[tuple[str, ...], ...]
    weights: tuple[float, ...]

    @property
    def counts(self) -> tuple[int, ...]:
        return tuple(len(names) for names in self.names)

    @property
    def dimension(self) -> int:
        return sum(self.counts)

    def weight_vector(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Block-size-normalized diagonal ``W``: ``lambda_c / d_c`` on every block entry."""
        return torch.cat(
            [
                torch.full((count,), weight / count, dtype=dtype, device=device)
                for weight, count in zip(self.weights, self.counts)
            ]
        )


def factor_manifest(settings: MultiscaleSettings) -> FactorManifest:
    component = settings.component("third_order_product_moments")
    groups = component.tuples if component is not None else ()
    factors = tuple(sorted({factor for group in groups for factor in group}))
    lookup = {factor: index for index, factor in enumerate(factors)}
    return FactorManifest(
        factors=factors,
        triples=tuple(tuple(lookup[factor] for factor in group) for group in groups),
    )


def descriptor_layout(
    settings: MultiscaleSettings, region_names: tuple[str, ...]
) -> DescriptorLayout:
    fields, bands = settings.fields, settings.bands.names
    keys, names, weights = [], [], []
    for component in settings.components:
        if component.key == "log_band_power":
            entries = [f"power:{field}@{band}" for band in bands for field in fields]
        elif component.key == "signed_band_cospectrum":
            entries = [
                f"cospectrum:{fields[left]}*{fields[right]}@{band}"
                for band in bands
                for left, right in component.pairs
            ]
        else:
            regions = ("global", *region_names) if component.regional else ("global",)
            entries = [
                "third:" + "*".join(f"{fields[f]}@{bands[b]}" for f, b in group) + f"|{region}"
                for group in component.tuples
                for region in regions
            ]
        keys.append(component.key)
        names.append(tuple(entries))
        weights.append(component.weight)
    return DescriptorLayout(tuple(keys), tuple(names), tuple(weights))


def raw_statistics(
    fields: torch.Tensor,
    *,
    mass: torch.Tensor,
    basis: torch.Tensor,
    filters: torch.Tensor,
    piece_measures: torch.Tensor,
    pieces: torch.Tensor,
    manifest: FactorManifest,
    with_band_fields: bool = False,
    with_piece_means: bool = False,
    relation_pairs: tuple[tuple[int, int], ...] | None = None,
) -> dict[str, torch.Tensor]:
    """Untransformed per-state statistics shared by calibration and descriptors.

    ``with_band_fields`` adds the synthesized signed band fields [B,S,N,C] that the
    signed_band_energy_score objective scores directly.
    """
    fluctuations = piecewise_fluctuations(fields, piece_measures, pieces)
    coefficients = spectral_coefficients(fluctuations, basis * mass[:, None])
    statistics = {
        "coefficients": coefficients,
        "second_moment": mean_square(fields, mass),
        "fluctuation_power": mean_square(fluctuations, mass),
    }
    if relation_pairs is None:
        statistics["gram"] = band_gram(coefficients, filters.square())
    else:
        pairs = torch.as_tensor(relation_pairs, dtype=torch.long, device=fields.device).reshape(-1, 2)
        statistics["power"] = torch.einsum("sk,bkc->bsc", filters.square(), coefficients.square())
        statistics["pair_gram"] = torch.einsum("sk,bkp,bkp->bsp", filters.square(), coefficients[:, :, pairs[:, 0]], coefficients[:, :, pairs[:, 1]])
        statistics["pair_indices"] = pairs
    if with_piece_means:
        statistics["piece_means"] = piecewise_means(fields, piece_measures)
    if with_band_fields:
        statistics["band_fields"] = band_fields(coefficients, basis, filters)
    if manifest.factors:
        field_index = torch.tensor([item[0] for item in manifest.factors], device=fields.device)
        band_index = torch.tensor([item[1] for item in manifest.factors], device=fields.device)
        # Only the band fields that enter nonlinear products are synthesized.
        statistics["factor_series"] = torch.einsum(
            "nk,fk,bkf->bfn", basis, filters[band_index], coefficients[:, :, field_index]
        )
    return statistics


@dataclass(frozen=True)
class CalibrationTensors:
    fluctuation_reference: torch.Tensor
    power_reference: torch.Tensor
    power_floor: torch.Tensor
    band_scale: torch.Tensor
    factor_power_reference: torch.Tensor
    factor_scale: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class ReferenceAccumulator:
    """Float64 running sums of training-reference statistics."""

    def __init__(
        self,
        fields: int,
        bands: int,
        factors: int,
        regions: int,
        device: torch.device,
        *,
        pieces: int | None = None,
    ):
        options = {"dtype": torch.float64, "device": device}
        self.count = 0
        self.second_moment_sum = torch.zeros(fields, **options)
        self.fluctuation_sum = torch.zeros(fields, **options)
        self.power_sum = torch.zeros(bands, fields, **options)
        self.resolved_fraction_sum = torch.zeros(fields, **options)
        self.factor_variance_sum = torch.zeros(factors, regions, **options)
        self.piece_mean = torch.zeros(pieces, fields, **options) if pieces is not None else None
        self.piece_mean_m2 = torch.zeros_like(self.piece_mean) if pieces is not None else None

    def update(self, statistics: Mapping[str, torch.Tensor], measures: torch.Tensor) -> None:
        power = statistics["power"] if "power" in statistics else torch.diagonal(statistics["gram"], dim1=-2, dim2=-1)
        total = statistics["fluctuation_power"]
        previous_count = self.count
        batch_count = int(total.shape[0])
        self.count += batch_count
        if self.piece_mean is not None:
            # Parallel Welford combination avoids E[mu^2]-E[mu]^2 cancellation
            # when physical-unit means have a large offset.
            means = statistics["piece_means"]
            batch_mean = means.mean(dim=0)
            batch_m2 = (means - batch_mean).square().sum(dim=0)
            delta = batch_mean - self.piece_mean
            self.piece_mean_m2 += batch_m2 + delta.square() * (
                previous_count * batch_count / self.count
            )
            self.piece_mean += delta * (batch_count / self.count)
        self.second_moment_sum += statistics["second_moment"].sum(dim=0)
        self.fluctuation_sum += total.sum(dim=0)
        self.power_sum += power.sum(dim=0)
        self.resolved_fraction_sum += torch.where(
            total > 0, power.sum(dim=1) / total.clamp_min(1e-300), 1.0
        ).sum(dim=0)
        if "factor_series" in statistics:
            self.factor_variance_sum += regional_variance(
                statistics["factor_series"], measures
            ).sum(dim=0)

    @property
    def resolved_fraction(self) -> torch.Tensor:
        return self.resolved_fraction_sum / max(self.count, 1)

    def mean_scales(self, field_unit_floor: tuple[float, ...]) -> torch.Tensor:
        """TRAIN population spread per piece and field, with a task-unit tolerance."""
        if self.piece_mean_m2 is None or self.count < 2:
            raise ValueError("piece-mean calibration requires at least two training states")
        floor = self.piece_mean_m2.new_tensor(field_unit_floor)
        return (self.piece_mean_m2 / self.count + floor.square()[None, :]).sqrt()

    def finalize(
        self,
        manifest: FactorManifest,
        *,
        band_relative_floor: float,
        absolute_floor: float,
        field_names: tuple[str, ...],
        field_unit_floor: tuple[float, ...] | None = None,
    ) -> CalibrationTensors:
        if self.count < 2:
            raise ValueError("multiscale reference calibration needs at least two training states")
        fluctuation = self.fluctuation_sum / self.count
        # A spatially constant field leaves only round-off (~1e-32 of its square) here.
        featureless = fluctuation <= 1.0e-24 * (self.second_moment_sum / self.count)
        empty = [
            name
            for name, value, flag in zip(field_names, fluctuation.tolist(), featureless.tolist())
            if flag or not value > 0
        ]
        if empty and field_unit_floor is None:
            raise ValueError(
                f"fields {empty} have no spatial fluctuation in the training calibration sample; "
                "remove them from multiscale_spectrum.fields"
            )
        power = self.power_sum / self.count
        floor = band_relative_floor * power + absolute_floor * fluctuation[None, :]
        task_floor = (
            None if field_unit_floor is None else power.new_tensor(field_unit_floor).square()
        )
        if task_floor is not None:
            floor = floor + task_floor[None, :]
        factor_power = self.factor_variance_sum / self.count
        if manifest.factors:
            factor_fields = torch.tensor(
                [item[0] for item in manifest.factors], device=power.device
            )
            factor_floor = (
                band_relative_floor * factor_power
                + absolute_floor * fluctuation[factor_fields][:, None]
            )
            if task_floor is not None:
                factor_floor = factor_floor + task_floor[factor_fields][:, None]
        else:
            factor_floor = factor_power
        return CalibrationTensors(
            fluctuation_reference=fluctuation,
            power_reference=power,
            power_floor=floor,
            band_scale=(power + floor).sqrt(),
            factor_power_reference=factor_power,
            factor_scale=(factor_power + factor_floor).sqrt(),
        )


def descriptor_blocks(
    statistics: Mapping[str, torch.Tensor],
    calibration: Mapping[str, torch.Tensor],
    *,
    settings: MultiscaleSettings,
    manifest: FactorManifest,
    measures: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Transform raw statistics into the configured descriptor blocks, each [B,d_c]."""
    blocks: dict[str, torch.Tensor] = {}
    gram = statistics.get("gram")
    power = statistics["power"] if gram is None else torch.diagonal(gram, dim1=-2, dim2=-1)
    batch = power.shape[0]
    for component in settings.components:
        key = component.key
        if key == "log_band_power":
            blocks[key] = torch.log1p(power / calibration["power_floor"]).reshape(batch, -1)
        elif key == "signed_band_cospectrum":
            left = torch.tensor([pair[0] for pair in component.pairs], device=power.device)
            right = torch.tensor([pair[1] for pair in component.pairs], device=power.device)
            scale = calibration["band_scale"][:, left] * calibration["band_scale"][:, right]
            if gram is None:
                lookup = {tuple(sorted(pair)): i for i, pair in enumerate(statistics["pair_indices"].tolist())}
                indices = [lookup[tuple(sorted(pair))] for pair in component.pairs]
                cross = statistics["pair_gram"][:, :, indices]
            else:
                cross = gram[:, :, left, right]
            blocks[key] = torch.asinh(cross / scale).reshape(batch, -1)
        else:
            rows = measures.shape[0] if component.regional else 1
            centered = regional_centered(statistics["factor_series"], measures[:rows])
            normalized = centered / calibration["factor_scale"][None, :, :rows, None]
            index = torch.tensor(manifest.triples, device=power.device, dtype=torch.long)
            moment = third_central_moments(normalized, measures[:rows], index)
            blocks[key] = torch.asinh(moment).reshape(batch, -1)
    return blocks
