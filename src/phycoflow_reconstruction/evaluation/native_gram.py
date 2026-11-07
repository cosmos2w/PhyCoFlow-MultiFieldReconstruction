"""Native Gram extension, fitted from the frozen TRAIN v4 Scale geometry.

This auxiliary bundle is separate from the original three-family yardstick.
Its scorer consumes saved native endpoints and never invokes a model.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ..coherence.families.multiscale_gram import (
    ANCHOR_PATH,
    GRAM_DEFINITION,
    MUTUAL_PATH,
    SELF_PATH,
    _candidate_implementation_identity,
    parse_gram_extension_config,
)
from ..coherence.families.multiscale_spectrum.statistics import band_gram
from ..coherence.registry import build_coherence_family
from ..contracts import DataSpec
from ..data.manifest import dataset_fingerprint
from ..training.run_store import file_sha256
from .canonical_coherence import _json_value, content_sha256
from .native_scoring import _score, load_native_metric_bundle


@torch.no_grad()
def fit_native_gram_bundle(config, train_dataset, *, anchor_bundle_path, output_path):
    """Add TRAIN-only Gram scales to an intact, strictly restored native v4 anchor.

    ``config`` is either the full resolved run config or its Scale family config.
    The v4 geometry and calibration are reused; no geometry eigensolve, model
    evaluation, or optimizer update occurs here.
    """
    full_bundle, families = load_native_metric_bundle(anchor_bundle_path)
    anchor = families["multiscale_spectrum"]
    if hasattr(anchor, "gram_extension"):
        raise ValueError("native Gram fit requires a legacy v4 anchor bundle")
    if "coherence" in config:
        config = config["coherence"]["families"]["multiscale_spectrum"]
    config = deepcopy(config)
    anchor_config, extension = parse_gram_extension_config(config, train_dataset.data_spec.field_names)
    if anchor_config != full_bundle["configs"]["multiscale_spectrum"]:
        raise ValueError("native Gram config differs from the frozen v4 anchor")
    if (getattr(train_dataset, "split_name", None) != "train"
            or getattr(getattr(train_dataset, "selection", None), "split", None) != "train"):
        raise ValueError("native Gram scales may be fitted on TRAIN only")
    if train_dataset.data_spec != DataSpec(**full_bundle["data_spec"]):
        raise ValueError("native Gram TRAIN data specification differs from the anchor")
    if train_dataset.normalizer.digest() != full_bundle["normalizer_digest"]:
        raise ValueError("native Gram TRAIN normalizer differs from the anchor")
    if (train_dataset.physical_measure_receipt != full_bundle["measure"]
            or not torch.equal(train_dataset.physical_weights.cpu(), full_bundle["native_weights"])):
        raise ValueError("native Gram TRAIN physical measure differs from the anchor")
    frozen = anchor.calibration_record
    if frozen.get("split") != "train" or frozen.get("dataset_fingerprint") != dataset_fingerprint(train_dataset.path):
        raise ValueError("native Gram TRAIN dataset differs from the anchor")
    query = full_bundle["native_ids"].cpu()
    if (not torch.equal(query, torch.arange(query.numel()))
            or not bool(full_bundle["native_mask"].all())
            or hashlib.sha256(query.numpy().tobytes()).hexdigest() != frozen.get("query_indices_sha256")):
        raise ValueError("native Gram requires the anchor's full ordered native support")
    settings = anchor.settings.calibration
    generator = torch.Generator(device="cpu").manual_seed(settings.seed)
    indices = torch.randperm(len(train_dataset), generator=generator)[:min(settings.samples, len(train_dataset))]
    sample_ids = []
    names = ["log_band_power"]
    if len(anchor.field_names) > 1:
        names.append("signed_band_cospectrum")
    blocks = {name: [] for name in names}
    batch = []

    def consume():
        if batch:
            values = anchor._descriptors(anchor._statistics(torch.stack(batch)))
            for name in names:
                blocks[name].append(values[name].to(torch.float64))
            batch.clear()

    for index in indices.tolist():
        sample = train_dataset[index]
        suffix = sample.time_index if sample.time_index is not None else "all"
        sample_ids.append(f"{sample.trajectory_id}:{suffix}")
        if (not torch.equal(sample.coordinates, full_bundle["coordinates"][0])
                or not torch.equal(sample.coordinates_raw, full_bundle["raw_coordinates"])
                or tuple(sample.field_names) != tuple(full_bundle["data_spec"]["field_names"])
                or (sample.valid_points is not None and not bool(sample.valid_points.all()))
                or not torch.isfinite(sample.values).all()):
            raise ValueError("native Gram TRAIN support, field order or values differ")
        batch.append(sample.values)
        if len(batch) == 32:
            consume()
    consume()
    digest = hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()
    if (sample_ids != frozen.get("sample_ids") or digest != frozen.get("sample_ids_sha256")
            or len(sample_ids) != frozen.get("states")):
        raise ValueError("native Gram TRAIN samples differ from the frozen anchor")
    record = {
        "definition": GRAM_DEFINITION, "source": "training_only", "sample_ids_sha256": digest,
        "std_floor": extension["std_floor"], "relative_std_floor": extension["relative_std_floor"],
        "variance_estimator": "unbiased_across_training_states", "blocks": {},
    }
    state = deepcopy(anchor.state_artifact())
    state["version"] = "2"
    state["config"] = config
    state["state_dict"]["gram_self_scale"] = torch.empty(0, dtype=anchor.compute_dtype)
    state["state_dict"]["gram_mutual_scale"] = torch.empty(0, dtype=anchor.compute_dtype)
    for name in names:
        descriptors = torch.cat(blocks[name], dim=0)
        if descriptors.shape[0] < 2 or descriptors.shape[1] == 0:
            raise ValueError("native Gram requires at least two TRAIN states and nonempty blocks")
        variance = descriptors.var(dim=0, unbiased=True)
        floor = torch.maximum(variance.new_tensor(extension["std_floor"] ** 2),
                              variance.mean() * extension["relative_std_floor"] ** 2)
        scale = torch.maximum(variance, floor).sqrt().to(anchor.compute_dtype)
        if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
            raise ValueError("native Gram TRAIN scales are not finite and positive")
        buffer_name = "gram_self_scale" if name == "log_band_power" else "gram_mutual_scale"
        state["state_dict"][buffer_name] = scale
        record["blocks"][name] = {
            "dimension": int(scale.numel()), "variance": variance.tolist(),
            "floor_variance": float(floor), "floor_count": int((variance < floor).sum()),
            "scale": scale.tolist(),
        }
    state["calibration"]["gram_extension"] = record
    state["gram_extension_definition"] = GRAM_DEFINITION
    state["gram_extension_implementation_identity"] = _candidate_implementation_identity()
    candidate = build_coherence_family("multiscale_spectrum", config, train_dataset.data_spec,
                                       train_dataset.normalizer).cpu()
    candidate.load_state_artifact(state)
    keys = ("data_spec", "normalization", "normalizer_digest", "summary_families", "summary_scales",
            "measure", "native_weights", "coordinates", "raw_coordinates", "native_ids", "native_mask",
            "source_checkpoint_sha256")
    result = {key: deepcopy(full_bundle[key]) for key in keys}
    result.update(version="native_gram_aux_bundle_v1", query_policy="full_native",
                  anchor_bundle_sha256=full_bundle["sha256"],
                  anchor_bundle_file_sha256=file_sha256(anchor_bundle_path),
                  configs={"multiscale_spectrum": config},
                  states={"multiscale_spectrum": candidate.state_artifact()},
                  calibration_sample_ids=sample_ids)
    result["sha256"] = content_sha256(result)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    return result


def _rho(gram):
    power = torch.diagonal(gram, dim1=-2, dim2=-1).clamp_min(0)
    denom = (power[..., :, None] * power[..., None, :]).sqrt()
    result = torch.where(denom > 0, gram / denom.clamp_min(torch.finfo(gram.dtype).tiny),
                         torch.full_like(gram, float("nan"))).cpu().numpy()
    # Undefined correlation is JSON null, not a fabricated zero.
    return np.where(np.isfinite(result), result, None).tolist()


def _rho_regularized(gram, floor):
    power = torch.diagonal(gram, dim1=-2, dim2=-1)
    regularized_power = power + floor
    denominator = (regularized_power[..., :, None] * regularized_power[..., None, :]).sqrt()
    return (gram / denominator).tolist()


def _account_statistics(family, captured):
    target, *members = captured
    coefficients = torch.cat([value["coefficients"] for value in members], dim=0)
    member_grams = torch.cat([value["gram"] for value in members], dim=0)
    target_gram = target["gram"][0]
    target_coefficients = target["coefficients"]
    physical_floor = family.power_floor.to(coefficients)
    # The frozen scorer can operate in model units; piece-centered spectral
    # coefficients convert to physical units by scale alone (offsets cancel).
    if family.units == "model_units":
        scale = family.normalization_scale[list(family.field_ids)].to(coefficients)
        coefficients = coefficients * scale
        target_coefficients = target_coefficients * scale
        scale_gram = scale[:, None] * scale[None, :]
        member_grams = member_grams * scale_gram
        target_gram = target_gram * scale_gram
        physical_floor = physical_floor * scale.square()
    mean_coeff = coefficients.mean(dim=0, keepdim=True)
    mean_gram = band_gram(mean_coeff, family.filters.square())[0]
    variation_gram = band_gram(coefficients - mean_coeff, family.filters.square()).mean(dim=0)
    mean_member_gram = member_grams.mean(dim=0)
    difference = coefficients - target_coefficients
    member_error = band_gram(difference, family.filters.square()).diagonal(dim1=-2, dim2=-1).mean(dim=0)
    mean_error = band_gram(mean_coeff - target_coefficients, family.filters.square())[0].diagonal(dim1=-2, dim2=-1)
    variation_power = variation_gram.diagonal(dim1=-2, dim2=-1)
    return {
        "field_names": list(family.field_names), "band_names": list(family.settings.bands.names),
        "field_units": [family.input_data_contract["field_units"][i] for i in family.field_ids],
        "raw_statistic_units": "physical_units", "score_input_units": family.units,
        "gram_convention": "serialized v4 represented signed band Gram; diagonal is band power",
        "mean_member_gram": mean_member_gram.tolist(), "mean_pattern_gram": mean_gram.tolist(),
        "member_variation_gram": variation_gram.tolist(), "target_gram": target_gram.tolist(),
        "member_grams": member_grams.tolist(),
        "mean_member_power": mean_member_gram.diagonal(dim1=-2, dim2=-1).tolist(),
        "mean_pattern_power": mean_gram.diagonal(dim1=-2, dim2=-1).tolist(),
        "member_variation_power": variation_power.tolist(),
        "target_power": target_gram.diagonal(dim1=-2, dim2=-1).tolist(),
        "rho_mean_member_gram": _rho(mean_member_gram), "rho_mean_pattern": _rho(mean_gram),
        "rho_target": _rho(target_gram),
        "rho_regularization": {
            "definition": "G_ij / sqrt((G_ii + floor_i) * (G_jj + floor_j))",
            "floor_source": "frozen TRAIN v4 band power_floor converted to physical units",
            "band_field_power_floor": physical_floor.tolist(),
        },
        "rho_regularized_mean_member_gram": _rho_regularized(mean_member_gram, physical_floor),
        "rho_regularized_mean_pattern": _rho_regularized(mean_gram, physical_floor),
        "rho_regularized_target": _rho_regularized(target_gram, physical_floor),
        "rho_regularized_members": _rho_regularized(member_grams, physical_floor),
        "represented_band_member_mse": member_error.tolist(),
        "represented_band_ensemble_mean_mse": mean_error.tolist(),
        "gram_identity_max_abs": float((mean_member_gram - mean_gram - variation_gram).abs().max()),
        "skill_identity_max_abs": float((member_error - mean_error - variation_power).abs().max()),
    }


@torch.no_grad()
def score_native_gram_payloads(bundle_path, payload_paths, output_path):
    """Score saved native draws with either an auxiliary or full candidate bundle."""
    bundle, families = load_native_metric_bundle(bundle_path)
    family = families["multiscale_spectrum"]
    if not hasattr(family, "gram_extension"):
        raise ValueError("native Gram scoring requires a candidate Gram bundle")
    initial = content_sha256(family.state_artifact())
    context = {"generated_weights": bundle["native_weights"], "reference_weights": bundle["native_weights"],
               "measure_metadata": {"native_measure_sha256": bundle["measure"]["sha256"]}}
    rows = []
    started = perf_counter()
    for path in payload_paths:
        with np.load(path, allow_pickle=False) as payload:
            members = torch.from_numpy(payload["draws_model"]).unsqueeze(1)
            target = torch.from_numpy(payload["target_model"]).unsqueeze(0)
            coordinates = torch.from_numpy(payload["model_coordinates"]).unsqueeze(0)
            if (not torch.equal(coordinates, bundle["coordinates"])
                    or not torch.equal(torch.from_numpy(payload["raw_coordinates"]), bundle["raw_coordinates"])
                    or not torch.equal(torch.from_numpy(payload["native_ids"]), bundle["native_ids"])
                    or not torch.equal(torch.from_numpy(payload["physical_mask"]), bundle["native_mask"])):
                raise ValueError("saved native support disagrees with the metric bundle")
            weights = bundle["native_weights"].double()
            if not torch.allclose(torch.from_numpy(payload["physical_weights"]).double(), weights / weights.sum(), rtol=1e-12, atol=1e-15):
                raise ValueError("saved native physical weights disagree with the metric bundle")
            if (tuple(map(str, payload["field_names"])) != tuple(bundle["data_spec"]["field_names"])
                    or tuple(map(str, payload["field_units"])) != tuple(bundle["data_spec"]["field_units"])
                    or str(payload["normalizer_sha256"].item()) != bundle["normalizer_digest"]):
                raise ValueError("saved native field order, units or normalizer disagree with the metric bundle")
            if (members.ndim != 4 or members.shape[2:] != target.shape[1:] or members.shape[0] < 2
                    or not torch.isfinite(members).all() or not torch.isfinite(target).all()):
                raise ValueError("saved native members and target do not satisfy the conditional score")
            captured = []
            original = family._statistics

            def capture(fields, *, collect=captured, original_statistics=original):
                statistics = original_statistics(fields)
                collect.append(statistics)
                return statistics

            family._statistics = capture
            try:
                values, components, _, details = _score({"multiscale_spectrum": family}, members, target, coordinates, context)
            finally:
                del family._statistics
            if len(captured) != 1 + members.shape[0]:
                raise RuntimeError("native Gram statistics do not match the declared members")
            rows.append({"payload": str(Path(path).resolve()), "values": values, "components": components,
                         "details": details, "physical": _account_statistics(family, captured)})
    if not rows:
        raise ValueError("native Gram scoring requires saved conditions")
    if content_sha256(family.state_artifact()) != initial:
        raise RuntimeError("native Gram scoring mutated the frozen metric bundle")
    paths = [ANCHOR_PATH, SELF_PATH]
    if len(family.field_names) > 1:
        paths.append(MUTUAL_PATH)
    means = {"candidate_total": float(np.mean([row["values"]["multiscale_spectrum"] for row in rows]))}
    means.update({path: float(np.mean([row["components"]["multiscale_spectrum"][path] for row in rows]))
                  for path in paths})
    means["anchor_total"] = means[ANCHOR_PATH]
    means["gram_self_score"] = means[SELF_PATH]
    if MUTUAL_PATH in paths:
        means["gram_mutual_score"] = means[MUTUAL_PATH]
    report = {"metric_bundle_sha256": bundle["sha256"],
              "anchor_bundle_sha256": bundle.get("anchor_bundle_sha256"),
              "definition": GRAM_DEFINITION, "anchor_definition": family.settings.score_definition_version,
              "query_policy": "full_native", "condition_count": len(rows), "rows": rows,
              "means": means, "scoring_seconds": perf_counter() - started}
    Path(output_path).write_text(json.dumps(_json_value(report), indent=2, allow_nan=False) + "\n")
    return report
