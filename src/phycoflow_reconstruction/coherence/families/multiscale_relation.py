"""Fair self-power and per-pair signed-relation scoring on the frozen v4 anchor."""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import torch

from ...contracts import CoherenceComponentSpec, CoherenceFamilySpec, FamilyResult, TermResult
from .multiscale_spectrum import MultiscaleSpectrumFamily, parse_multiscale_config
from .multiscale_spectrum.config import MEAN_AUGMENTED_SCORE_VERSION
from .multiscale_spectrum.losses import selected_band_coherence, vector_energy_score

RELATION_DEFINITION = "self_rho_fair_v1"
SELF_PATH = "multiscale_spectrum.relation_self_energy_score.conditional"
RHO_PATH = "multiscale_spectrum.relation_rho_crps.conditional"
ANCHOR_PATH = "multiscale_spectrum.signed_anchor_total.conditional"
_CAPTURE: ContextVar[tuple[int, list[tuple[dict[str, torch.Tensor], torch.Tensor]]] | None] = (
    ContextVar("multiscale_relation_descriptor_capture", default=None)
)


def parse_relation_extension_config(config: Mapping[str, Any], dataset_field_names: Sequence[str]):
    if not isinstance(config, Mapping):
        raise TypeError("multiscale_spectrum config must be a mapping")
    anchor = copy.deepcopy(dict(config))
    if "gram_extension" in anchor:
        raise ValueError("relation_extension cannot be combined with gram_extension")
    extension = anchor.pop("relation_extension", None)
    if not isinstance(extension, Mapping):
        raise TypeError("relation_extension must be a mapping")
    unknown = set(extension) - {
        "definition",
        "self_weight",
        "rho_weight",
        "std_floor",
        "relative_std_floor",
        "implicit_pairs",
    }
    if unknown:
        raise ValueError(f"unknown relation_extension keys: {sorted(unknown)}")
    if extension.get("definition") != RELATION_DEFINITION:
        raise ValueError(f"relation_extension.definition must be {RELATION_DEFINITION!r}")

    def positive(name):
        v = extension.get(name)
        if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"relation_extension.{name} must be positive and finite")
        return float(v)

    implicit_pairs = extension.get("implicit_pairs", False)
    if not isinstance(implicit_pairs, bool):
        raise TypeError("relation_extension.implicit_pairs must be boolean")
    if (
        implicit_pairs
        and anchor.get("components", {}).get("signed_band_cospectrum", {}).get("enabled")
        is not False
    ):
        raise ValueError("implicit pairs require disabling the optional cospectrum descriptor")
    settings = {
        "definition": RELATION_DEFINITION,
        "implicit_pairs": implicit_pairs,
        **{
            name: positive(name)
            for name in ("self_weight", "rho_weight", "std_floor", "relative_std_floor")
        },
    }
    parsed = parse_multiscale_config(anchor, dataset_field_names)
    if (
        parsed.objective != "signed_band_energy_score"
        or parsed.score_definition_version != MEAN_AUGMENTED_SCORE_VERSION
        or parsed.almost_fair_alpha != 1.0
        or parsed.joint_energy_score_weight <= 0
        or parsed.component("log_band_power") is None
        or (
            not implicit_pairs
            and len(parsed.fields) > 1
            and parsed.component("signed_band_cospectrum") is None
        )
    ):
        raise ValueError("relation_extension requires the fair mean-augmented signed anchor")
    if anchor.get("objective", {}).get("coherence_crps_weight", 0) != 0:
        raise ValueError("relation_extension requires coherence_crps_weight=0")
    return anchor, settings


def _candidate_implementation_identity():
    return {
        "format": "candidate_source_sha256_v1",
        "definition": RELATION_DEFINITION,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


class MultiscaleRelationFamily(MultiscaleSpectrumFamily):
    version = "2"
    score_definition_version = RELATION_DEFINITION

    def __init__(self, config, data_spec, normalizer):
        anchor, self.relation_extension = parse_relation_extension_config(
            config, data_spec.field_names
        )
        super().__init__(anchor, data_spec, normalizer)
        if self.relation_extension["implicit_pairs"]:
            # No pair descriptors are needed for pooled self calibration. The
            # sampled runtime evaluates every possible pair on demand.
            self.coverage_pairs = ()
        self.config = copy.deepcopy(dict(config))
        self.register_buffer("relation_self_scale", torch.empty(0, dtype=self.compute_dtype))
        names = [
            ("signed_anchor_total.conditional", False),
            ("relation_self_energy_score.conditional", True),
        ]
        if len(self.field_names) > 1:
            names.append(("relation_rho_crps.conditional", True))
        components = tuple(
            CoherenceComponentSpec(
                name,
                self.target_use,
                self.units,
                True,
                required_geometry="fixed_point_operator",
                aggregation="per_sample",
                metadata=(
                    {"trained": True, "conditional_draws": self.conditional_draws}
                    if trained
                    else {"trained": False, "derived_reporting_view": True}
                ),
            )
            for name, trained in names
        )
        self.spec = CoherenceFamilySpec(
            self.family_name,
            self.version,
            self.spec.components + components,
            metadata={**self.spec.metadata, "score_definition_version": RELATION_DEFINITION},
        )
        self.spec.validate()

    @property
    def fitted(self):
        return bool(
            super().fitted
            and self.calibration_record is not None
            and "relation_extension" in self.calibration_record
            and self.relation_self_scale.numel()
        )

    def _descriptors(self, statistics):
        blocks = super()._descriptors(statistics)
        capture = _CAPTURE.get()
        if capture is not None and capture[0] == id(self):
            capture[1].append((blocks, statistics))
        return blocks

    def fit_reference_tensors(
        self, coordinates, states: torch.Tensor | Iterable[torch.Tensor], **kwargs: Any
    ):
        batches = [states] if isinstance(states, torch.Tensor) else list(states)
        super().fit_reference_tensors(coordinates, batches, **kwargs)
        try:
            with torch.no_grad():
                values = torch.cat(
                    [
                        self._descriptors(
                            self._statistics(batch.to(self.normalization_offset.device))
                        )["log_band_power"].to(torch.float64)
                        for batch in batches
                    ]
                )
            if values.shape[0] < 2 or values.shape[1] == 0:
                raise ValueError("relation_extension needs at least two TRAIN states")
            variance = values.var(dim=0, unbiased=True)
            if not bool(torch.isfinite(variance).all()) or bool((variance < 0).any()):
                raise FloatingPointError("invalid relation self variance")
            floor = torch.maximum(
                variance.new_tensor(self.relation_extension["std_floor"] ** 2),
                variance.mean() * self.relation_extension["relative_std_floor"] ** 2,
            )
            scale = torch.maximum(variance, floor).sqrt().to(self.compute_dtype)
            if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
                raise FloatingPointError("invalid relation self scale")
            self.relation_self_scale = scale
            self.calibration_record["relation_extension"] = {
                "definition": RELATION_DEFINITION,
                "source": "training_only",
                "sample_ids_sha256": self.calibration_record["sample_ids_sha256"],
                "std_floor": self.relation_extension["std_floor"],
                "relative_std_floor": self.relation_extension["relative_std_floor"],
                "variance_estimator": "unbiased_across_training_states",
                "variance": variance.cpu().tolist(),
                "floor_variance": float(floor.cpu()),
                "floor_count": int((variance < floor).sum()),
                "scale": scale.cpu().tolist(),
            }
        except BaseException:
            self.calibration_record = None
            raise

    def forward(self, generated, reference, *, coordinates=None, context=None) -> FamilyResult:
        if self.relation_extension["implicit_pairs"]:
            raise ValueError("implicit-pair Scale scoring requires ScalableQueryAdapter")
        captured = []
        token = _CAPTURE.set((id(self), captured))
        try:
            result = super().forward(generated, reference, coordinates=coordinates, context=context)
        finally:
            _CAPTURE.reset(token)
        supplied = (
            tuple(context.get("conditional_draws", ())) if isinstance(context, Mapping) else ()
        )
        all_draws = (
            bool(context.get("score_all_conditional_draws", False))
            if isinstance(context, Mapping)
            else False
        )
        expected = 1 + (len(supplied) if all_draws else self.conditional_draws - 1)
        if len(captured) != 1 + expected:
            raise RuntimeError(
                "v4 descriptor calls do not match the declared target and draw count"
            )
        target, draws = captured[0], captured[1:]
        total = result.per_sample_cost
        if total is None:
            raise RuntimeError("signed Scale anchor did not return per-sample costs")
        result.component_results[ANCHOR_PATH] = TermResult(
            total,
            total.mean(),
            diagnostics={"evaluation_only": True, "definition": MEAN_AUGMENTED_SCORE_VERSION},
        )
        result.diagnostics["components"][ANCHOR_PATH] = {
            "weight": 0.0,
            "executed": True,
            "raw_scalar_loss": float(total.mean().detach()),
            "weighted_scalar_contribution": 0.0,
        }
        denominator = self.relation_self_scale.to(generated.device) * math.sqrt(
            self.relation_self_scale.numel()
        )
        self_score, attraction, repulsion = vector_energy_score(
            torch.stack([blocks["log_band_power"] / denominator for blocks, _ in draws]),
            target[0]["log_band_power"] / denominator,
            alpha=1.0,
        )
        # These are detached scheduling diagnostics, not a decomposition of the
        # complete joint self norm and not an additional optimized objective.
        bands, fields = len(self.settings.bands.names), len(self.field_names)
        field_draws = torch.stack(
            [
                blocks["log_band_power"].reshape(-1, bands, fields).transpose(1, 2)
                / denominator.reshape(bands, fields).T
                for blocks, _ in draws
            ]
        )
        field_target = (
            target[0]["log_band_power"].reshape(-1, bands, fields).transpose(1, 2)
            / denominator.reshape(bands, fields).T
        )
        field_scores, _, _ = vector_energy_score(field_draws, field_target, alpha=1.0)
        result.diagnostics["coverage_priorities"] = {
            "self": field_scores.mean(dim=0).detach().cpu().tolist(),
            "pair": [],
        }
        entries = [
            (
                SELF_PATH,
                self_score,
                self.relation_extension["self_weight"],
                {
                    "conditional_draws": expected,
                    "dimension": self.relation_self_scale.numel(),
                    "attraction": float(attraction.mean().detach()),
                    "repulsion": float(repulsion.mean().detach()),
                },
            )
        ]
        if len(self.field_names) > 1:
            rho_draws = torch.stack(
                [selected_band_coherence(statistics, self.power_floor) for _, statistics in draws]
            )
            rho_target = selected_band_coherence(target[1], self.power_floor)
            crps, rho_attraction, rho_repulsion = vector_energy_score(
                rho_draws[..., None], rho_target[..., None], alpha=1.0
            )
            score = crps.mean(dim=(1, 2))
            result.diagnostics["coverage_priorities"]["pair"] = (
                crps.mean(dim=(0, 1)).detach().cpu().tolist()
            )
            entries.append(
                (
                    RHO_PATH,
                    score,
                    self.relation_extension["rho_weight"],
                    {
                        "conditional_draws": expected,
                        "attraction": float(rho_attraction.mean().detach()),
                        "repulsion": float(rho_repulsion.mean().detach()),
                        "per_band": crps.mean(dim=(0, 2)).detach().cpu().tolist(),
                        "per_pair": crps.mean(dim=(0, 1)).detach().cpu().tolist(),
                    },
                )
            )
        for path, score, weight, diagnostics in entries:
            if not bool(torch.isfinite(score.detach()).all()):
                raise FloatingPointError(f"non-finite relation score: {path}")
            result.component_results[path] = TermResult(
                score, score.mean(), diagnostics=diagnostics
            )
            raw = float(score.mean().detach())
            result.diagnostics["components"][path] = {
                "weight": weight,
                "executed": True,
                "raw_scalar_loss": raw,
                "weighted_scalar_contribution": weight * raw,
            }
            total = total + weight * score
        result.per_sample_cost = total
        result.scalar_loss = total.mean()
        result.diagnostics["relation_extension"] = {
            "definition": RELATION_DEFINITION,
            "anchor_definition": MEAN_AUGMENTED_SCORE_VERSION,
            "self_weight": self.relation_extension["self_weight"],
            "rho_weight": self.relation_extension["rho_weight"],
        }
        return result

    def state_artifact(self):
        artifact = super().state_artifact()
        artifact["relation_extension_definition"] = RELATION_DEFINITION
        artifact["relation_extension_implementation_identity"] = (
            _candidate_implementation_identity()
        )
        return artifact

    def load_state_artifact(self, artifact):
        if artifact.get("relation_extension_definition") != RELATION_DEFINITION:
            raise ValueError("relation_extension artifact definition mismatch")
        if (
            artifact.get("relation_extension_implementation_identity")
            != _candidate_implementation_identity()
        ):
            raise ValueError("relation_extension implementation identity mismatch")
        state, record = artifact.get("state_dict"), artifact.get("calibration")
        if not isinstance(state, Mapping) or not isinstance(record, Mapping):
            raise TypeError("relation_extension artifact is missing frozen state")
        item = record.get("relation_extension")
        if not isinstance(item, Mapping) or item.get("definition") != RELATION_DEFINITION:
            raise ValueError("relation_extension artifact missing TRAIN calibration")
        if item.get("sample_ids_sha256") != record.get("sample_ids_sha256"):
            raise ValueError("relation_extension calibration IDs differ from anchor")
        for key in ("std_floor", "relative_std_floor"):
            if item.get(key) != self.relation_extension[key]:
                raise ValueError("relation_extension calibration floor differs from config")
        value = state.get("relation_self_scale")
        size = len(self.settings.bands.names) * len(self.field_names)
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (size,)
            or value.dtype != self.compute_dtype
        ):
            raise ValueError("relation_extension self scale shape or dtype mismatch")
        variance = torch.as_tensor(item.get("variance"), dtype=torch.float64, device=value.device)
        if (
            variance.shape != value.shape
            or not bool(torch.isfinite(variance).all())
            or bool((variance < 0).any())
        ):
            raise ValueError("relation_extension variance invalid")
        floor = torch.maximum(
            variance.new_tensor(self.relation_extension["std_floor"] ** 2),
            variance.mean() * self.relation_extension["relative_std_floor"] ** 2,
        )
        recorded = torch.as_tensor(item.get("scale"), dtype=value.dtype, device=value.device)
        tol = torch.finfo(value.dtype).eps * 8
        if (
            not math.isclose(
                float(floor), float(item.get("floor_variance")), rel_tol=1e-12, abs_tol=1e-18
            )
            or int((variance < floor).sum()) != item.get("floor_count")
            or not torch.equal(value, recorded)
            or not torch.allclose(
                value, torch.maximum(variance, floor).sqrt().to(value.dtype), rtol=tol, atol=tol
            )
        ):
            raise ValueError("relation_extension self scales differ from calibration")
        self.relation_self_scale = value.to(device=self.normalization_offset.device)
        super().load_state_artifact(artifact)
        if not self.fitted:
            raise ValueError("relation_extension restored an incomplete reference")
