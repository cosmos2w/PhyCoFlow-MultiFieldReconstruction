"""Fair self/mutual Gram augmentation of the frozen signed Scale anchor.

The v4 Scale modules remain byte-identical so historical scientific artifacts
retain their strict source identity. This wrapper owns its candidate identity,
TRAIN-only standardization, and two independently normalized energy scores.
"""

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
from .multiscale_spectrum.losses import vector_energy_score

GRAM_DEFINITION = "self_mutual_gram_v1"
SELF_PATH = "multiscale_spectrum.gram_self_energy_score.conditional"
MUTUAL_PATH = "multiscale_spectrum.gram_mutual_energy_score.conditional"
ANCHOR_PATH = "multiscale_spectrum.signed_anchor_total.conditional"
_CAPTURE: ContextVar[tuple[int, list[dict[str, torch.Tensor]]] | None] = ContextVar(
    "multiscale_gram_descriptor_capture", default=None
)


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"gram_extension.{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"gram_extension.{name} must be positive and finite")
    return result


def parse_gram_extension_config(
    config: Mapping[str, Any], dataset_field_names: Sequence[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the candidate and return its exact sanitized v4 anchor config.

    Only the candidate root key is removed. The original strict v4 parser owns
    every anchor option, while the new key owns all candidate-specific options.
    """
    if not isinstance(config, Mapping):
        raise TypeError("multiscale_spectrum config must be a mapping")
    anchor = copy.deepcopy(dict(config))
    extension = anchor.pop("gram_extension", None)
    if not isinstance(extension, Mapping):
        raise TypeError("gram_extension must be a mapping")
    unknown = set(extension) - {"definition", "weight", "std_floor", "relative_std_floor"}
    if unknown:
        raise ValueError(f"unknown gram_extension keys: {sorted(unknown)}")
    definition = extension.get("definition")
    if definition != GRAM_DEFINITION:
        raise ValueError(f"gram_extension.definition must be {GRAM_DEFINITION!r}")
    settings = {
        "definition": GRAM_DEFINITION,
        "weight": _positive_number(extension.get("weight"), "weight"),
        "std_floor": _positive_number(extension.get("std_floor"), "std_floor"),
        "relative_std_floor": _positive_number(
            extension.get("relative_std_floor"), "relative_std_floor"
        ),
    }
    parsed = parse_multiscale_config(anchor, dataset_field_names)
    if (
        parsed.objective != "signed_band_energy_score"
        or parsed.score_definition_version != MEAN_AUGMENTED_SCORE_VERSION
        or parsed.almost_fair_alpha != 1.0
        or parsed.joint_energy_score_weight <= 0
    ):
        raise ValueError("gram_extension requires the fair mean-augmented v4 signed anchor")
    if parsed.component("log_band_power") is None:
        raise ValueError("gram_extension requires the log_band_power descriptor")
    if len(parsed.fields) > 1 and parsed.component("signed_band_cospectrum") is None:
        raise ValueError("gram_extension requires signed_band_cospectrum for multiple fields")
    return anchor, settings


def _candidate_implementation_identity() -> dict[str, str]:
    return {
        "format": "candidate_source_sha256_v1",
        "definition": GRAM_DEFINITION,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


class MultiscaleGramFamily(MultiscaleSpectrumFamily):
    """The v4 anchor plus independently normalized self and signed-mutual ES."""

    version = "2"
    score_definition_version = GRAM_DEFINITION

    def __init__(self, config: Mapping[str, Any], data_spec, normalizer) -> None:
        anchor, extension = parse_gram_extension_config(config, data_spec.field_names)
        super().__init__(anchor, data_spec, normalizer)
        # Checkpoints and artifacts bind the complete candidate, not only v4.
        self.config = copy.deepcopy(dict(config))
        self.gram_extension = extension
        self.register_buffer("gram_self_scale", torch.empty(0, dtype=self.compute_dtype))
        self.register_buffer("gram_mutual_scale", torch.empty(0, dtype=self.compute_dtype))
        paths = [(SELF_PATH, "gram_self_energy_score.conditional")]
        if len(self.field_names) > 1:
            paths.append((MUTUAL_PATH, "gram_mutual_energy_score.conditional"))
        extra = tuple(
            CoherenceComponentSpec(
                component,
                self.target_use,
                self.units,
                True,
                required_geometry="fixed_point_operator",
                aggregation="per_sample",
                metadata={"trained": True, "conditional_draws": self.conditional_draws},
            )
            for _, component in paths
        )
        anchor_view = CoherenceComponentSpec(
            "signed_anchor_total.conditional",
            self.target_use,
            self.units,
            True,
            required_geometry="fixed_point_operator",
            aggregation="per_sample",
            metadata={"trained": False, "derived_reporting_view": True},
        )
        self.spec = CoherenceFamilySpec(
            self.family_name,
            self.version,
            self.spec.components + (anchor_view,) + extra,
            metadata={**self.spec.metadata, "score_definition_version": GRAM_DEFINITION},
        )
        self.spec.validate()

    @property
    def fitted(self) -> bool:
        return bool(
            super().fitted
            and self.calibration_record is not None
            and "gram_extension" in self.calibration_record
            and self.gram_self_scale.numel() > 0
            and (len(self.field_names) == 1 or self.gram_mutual_scale.numel() > 0)
        )

    def _descriptors(self, statistics: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        blocks = super()._descriptors(statistics)
        capture = _CAPTURE.get()
        if capture is not None and capture[0] == id(self):
            capture[1].append(blocks)
        return blocks

    def fit_reference_tensors(
        self,
        coordinates: torch.Tensor,
        states: torch.Tensor | Iterable[torch.Tensor],
        **kwargs: Any,
    ) -> None:
        # The v4 floors must be frozen before transformed descriptor variance
        # can be fitted. Materialize the bounded TRAIN reference for two passes.
        batches = [states] if isinstance(states, torch.Tensor) else list(states)
        super().fit_reference_tensors(coordinates, batches, **kwargs)
        names = ["log_band_power"]
        if len(self.field_names) > 1:
            names.append("signed_band_cospectrum")
        values: dict[str, list[torch.Tensor]] = {name: [] for name in names}
        try:
            with torch.no_grad():
                for batch in batches:
                    blocks = self._descriptors(self._statistics(batch.to(self.normalization_offset.device)))
                    for name in names:
                        values[name].append(blocks[name].to(torch.float64))
            record: dict[str, Any] = {
                "definition": GRAM_DEFINITION,
                "source": "training_only",
                "sample_ids_sha256": self.calibration_record["sample_ids_sha256"],
                "std_floor": self.gram_extension["std_floor"],
                "relative_std_floor": self.gram_extension["relative_std_floor"],
                "variance_estimator": "unbiased_across_training_states",
                "blocks": {},
            }
            for name in names:
                descriptors = torch.cat(values[name], dim=0)
                if descriptors.shape[0] < 2 or descriptors.shape[1] == 0:
                    raise ValueError("gram_extension needs at least two TRAIN states and a nonempty block")
                variance = descriptors.var(dim=0, unbiased=True)
                if not bool(torch.isfinite(variance).all()) or bool((variance < 0).any()):
                    raise FloatingPointError("gram_extension TRAIN descriptor variance is non-finite")
                floor = torch.maximum(
                    variance.new_tensor(self.gram_extension["std_floor"] ** 2),
                    variance.mean() * self.gram_extension["relative_std_floor"] ** 2,
                )
                scale = torch.maximum(variance, floor).sqrt().to(self.compute_dtype)
                if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
                    raise FloatingPointError("gram_extension scale is not positive in compute precision")
                buffer_name = "gram_self_scale" if name == "log_band_power" else "gram_mutual_scale"
                setattr(self, buffer_name, scale)
                record["blocks"][name] = {
                    "dimension": int(scale.numel()),
                    "variance": variance.detach().cpu().tolist(),
                    "floor_variance": float(floor.detach().cpu()),
                    "floor_count": int((variance < floor).sum().detach().cpu()),
                    "scale": scale.detach().cpu().tolist(),
                }
            self.calibration_record["gram_extension"] = record
        except BaseException:
            # A partially fitted candidate must never appear ready for scoring.
            self.calibration_record = None
            raise

    def forward(self, generated, reference, *, coordinates=None, context=None) -> FamilyResult:
        captured: list[dict[str, torch.Tensor]] = []
        token = _CAPTURE.set((id(self), captured))
        try:
            result = super().forward(generated, reference, coordinates=coordinates, context=context)
        finally:
            _CAPTURE.reset(token)
        supplied = tuple(context.get("conditional_draws", ())) if isinstance(context, Mapping) else ()
        all_draws = bool(context.get("score_all_conditional_draws", False)) if isinstance(context, Mapping) else False
        expected_draws = 1 + (len(supplied) if all_draws else self.conditional_draws - 1)
        if len(captured) != 1 + expected_draws:
            raise RuntimeError("v4 descriptor calls do not match the declared target and draw count")
        target, draws = captured[0], captured[1:]
        blocks = [("log_band_power", SELF_PATH, self.gram_self_scale)]
        if len(self.field_names) > 1:
            blocks.append(("signed_band_cospectrum", MUTUAL_PATH, self.gram_mutual_scale))
        total = result.per_sample_cost
        if total is None:
            raise RuntimeError("signed Scale anchor did not return per-sample costs")
        # This is a derived reporting view of an already trained anchor. Its
        # zero reporting weight prevents double counting the existing components.
        result.component_results[ANCHOR_PATH] = TermResult(
            total, total.mean(), diagnostics={"evaluation_only": True, "definition": MEAN_AUGMENTED_SCORE_VERSION}
        )
        result.diagnostics["components"][ANCHOR_PATH] = {
            "weight": 0.0, "executed": True, "raw_scalar_loss": float(total.mean().detach()),
            "weighted_scalar_contribution": 0.0,
        }
        block_weight = self.gram_extension["weight"] / len(blocks)
        block_report = {}
        for name, path, scales in blocks:
            dimension = scales.numel()
            denominator = scales.to(generated.device) * math.sqrt(dimension)
            samples = torch.stack([item[name] / denominator for item in draws])
            truth = target[name] / denominator
            score, attraction, repulsion = vector_energy_score(samples, truth, alpha=1.0)
            if not bool(torch.isfinite(score.detach()).all()):
                raise FloatingPointError(f"gram_extension {name} score is non-finite")
            diagnostics = {
                "conditional_draws": expected_draws,
                "dimension": dimension,
                "attraction": float(attraction.mean().detach()),
                "repulsion": float(repulsion.mean().detach()),
            }
            result.component_results[path] = TermResult(score, score.mean(), diagnostics=diagnostics)
            raw = float(score.mean().detach())
            result.diagnostics["components"][path] = {
                "weight": block_weight, "executed": True, "raw_scalar_loss": raw,
                "weighted_scalar_contribution": block_weight * raw,
            }
            block_report[name] = {**diagnostics, "weight": block_weight}
            total = total + block_weight * score
        result.per_sample_cost = total
        result.scalar_loss = total.mean()
        result.diagnostics["gram_extension"] = {
            "definition": GRAM_DEFINITION,
            "anchor_definition": MEAN_AUGMENTED_SCORE_VERSION,
            "weight": self.gram_extension["weight"],
            "blocks": block_report,
        }
        return result

    def state_artifact(self) -> dict[str, Any]:
        artifact = super().state_artifact()
        artifact["gram_extension_definition"] = GRAM_DEFINITION
        artifact["gram_extension_implementation_identity"] = _candidate_implementation_identity()
        return artifact

    def load_state_artifact(self, artifact: Mapping[str, Any]) -> None:
        if artifact.get("gram_extension_definition") != GRAM_DEFINITION:
            raise ValueError("gram_extension artifact definition mismatch")
        if artifact.get("gram_extension_implementation_identity") != _candidate_implementation_identity():
            raise ValueError("gram_extension implementation identity mismatch")
        state = artifact.get("state_dict")
        record = artifact.get("calibration")
        if not isinstance(state, Mapping) or not isinstance(record, Mapping):
            raise TypeError("gram_extension artifact is missing frozen state")
        gram_record = record.get("gram_extension")
        if not isinstance(gram_record, Mapping) or gram_record.get("definition") != GRAM_DEFINITION:
            raise ValueError("gram_extension artifact is missing TRAIN calibration")
        if gram_record.get("sample_ids_sha256") != record.get("sample_ids_sha256"):
            raise ValueError("gram_extension TRAIN calibration IDs differ from the anchor")
        for key in ("std_floor", "relative_std_floor"):
            if gram_record.get(key) != self.gram_extension[key]:
                raise ValueError("gram_extension calibration floor differs from config")
        blocks = gram_record.get("blocks")
        if not isinstance(blocks, Mapping):
            raise TypeError("gram_extension artifact is missing block calibration")
        self_dimension = len(self.settings.bands.names) * len(self.field_names)
        mutual_component = self.settings.component("signed_band_cospectrum")
        mutual_dimension = (
            0 if mutual_component is None
            else len(self.settings.bands.names) * len(mutual_component.pairs)
        )
        expected_blocks = {"log_band_power"}
        if mutual_dimension:
            expected_blocks.add("signed_band_cospectrum")
        if set(blocks) != expected_blocks:
            raise ValueError("gram_extension artifact block manifest differs from config")
        for name, buffer_name in (
            ("log_band_power", "gram_self_scale"),
            ("signed_band_cospectrum", "gram_mutual_scale"),
        ):
            value = state.get(buffer_name)
            expected_dimension = self_dimension if name == "log_band_power" else mutual_dimension
            if (not isinstance(value, torch.Tensor) or value.ndim != 1
                    or value.dtype != self.compute_dtype or value.numel() != expected_dimension):
                raise ValueError(f"gram_extension artifact has invalid {buffer_name}")
            present = name in blocks
            if present:
                metadata = blocks[name]
                if not isinstance(metadata, Mapping) or metadata.get("dimension") != value.numel():
                    raise ValueError("gram_extension artifact block dimensions differ")
                recorded = torch.as_tensor(metadata.get("scale"), dtype=value.dtype, device=value.device)
                if not torch.equal(value, recorded) or not bool(torch.isfinite(value).all()) or not bool((value > 0).all()):
                    raise ValueError("gram_extension artifact scales differ or are non-finite")
                variance = torch.as_tensor(metadata.get("variance"), dtype=torch.float64, device=value.device)
                if variance.shape != value.shape or not bool(torch.isfinite(variance).all()) or bool((variance < 0).any()):
                    raise ValueError("gram_extension artifact variance is invalid")
                floor = torch.maximum(
                    variance.new_tensor(self.gram_extension["std_floor"] ** 2),
                    variance.mean() * self.gram_extension["relative_std_floor"] ** 2,
                )
                recorded_floor = metadata.get("floor_variance")
                if not isinstance(recorded_floor, (int, float)) or not math.isclose(
                    float(floor), float(recorded_floor), rel_tol=1e-12, abs_tol=1e-18
                ):
                    raise ValueError("gram_extension artifact variance floor differs from calibration")
                fixed_floor = variance.new_tensor(float(recorded_floor))
                expected_scale = torch.maximum(variance, fixed_floor).sqrt().to(value.dtype)
                tolerance = torch.finfo(value.dtype).eps * 8
                if (int((variance < fixed_floor).sum()) != metadata.get("floor_count")
                        or not torch.allclose(value, expected_scale, rtol=tolerance, atol=tolerance)):
                    raise ValueError("gram_extension artifact TRAIN variance and floor do not reproduce scales")
            elif value.numel() != 0 or name == "log_band_power" or len(self.field_names) > 1:
                raise ValueError("gram_extension artifact is missing a required block")
            setattr(self, buffer_name, value.to(device=self.normalization_offset.device, dtype=self.compute_dtype))
        super().load_state_artifact(artifact)
        if not self.fitted:
            raise ValueError("gram_extension restored an incomplete reference")
