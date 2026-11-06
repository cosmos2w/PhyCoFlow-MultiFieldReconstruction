"""TRAIN-frozen native SSS yardsticks and saved-member rescoring.

Inference is separate in native_coherence: this module never draws samples,
updates model parameters, or fits a denominator from validation predictions.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ..coherence.registry import build_coherence_family
from ..contracts import DataSpec
from ..data.normalization import FieldNormalizer
from .canonical_coherence import _json_value, content_sha256


def _contexts(dataset):
    weights = dataset.physical_weights
    return {"generated_weights": weights, "reference_weights": weights,
            "measure_metadata": {"native_measure_sha256": dataset.physical_measure_receipt["sha256"]}}


def _score(families, members, target, coordinates, context):
    values, components, scales, details = {}, {}, {}, {}
    for name, family in families.items():
        results = []
        if int(getattr(family, "conditional_draws", 1)) > 1:
            results.append(family(members[0], target, coordinates=coordinates, context={
                **context, "conditional_draws": tuple(members[1:]),
                "score_all_conditional_draws": True, "report_detail": True}))
        else:
            for draw in members:
                results.append(family(draw, target, coordinates=coordinates, context=context))
        values[name] = float(torch.stack([r.scalar_loss for r in results]).mean())
        components[name] = {
            key: float(torch.stack([r.component_results[key].scalar_loss for r in results]).mean())
            for key in results[0].component_results}
        details[name] = {
            "aggregation": "conditional_all_members" if len(results) == 1 and members.shape[0] > 1 else "paired_per_member",
            "per_result_family": [float(r.scalar_loss) for r in results],
            "per_result_components": [
                {key: {"value": float(term.scalar_loss), "diagnostics": _json_value(term.diagnostics)}
                 for key, term in r.component_results.items()}
                for r in results
            ],
            "family_diagnostics": [_json_value(r.diagnostics) for r in results],
        }
        if name == "topology":
            details[name]["calibration_essential_counts"] = family.spatial_objective.essential_counts.cpu().tolist()
        if name == "multiscale_spectrum":
            result = results[0]
            primary = result.component_results[
                "multiscale_spectrum.signed_band_energy_score.conditional"
            ]
            diag = primary.diagnostics
            attraction = np.asarray(_json_value(diag["attraction_per_band"]))
            band_weights = np.asarray(_json_value(diag["band_weights"]))
            joint = result.diagnostics["signed_band"]["joint_energy_score"]["attraction"]
            scales[name] = float((attraction * band_weights).sum()) + family.settings.joint_energy_score_weight * float(joint)
        else:
            scales[name] = values[name]
    return values, components, scales, details


@torch.no_grad()
def fit_native_metric_bundle(config, train_dataset, *, source_members, targets,
                             coordinates, sample_ids, sample_indices, source_checkpoint_sha256,
                             output_path):
    """Fit one native bundle from TRAIN references and TRAIN/SOURCE endpoints."""
    if train_dataset.split_name != "train":
        raise ValueError("native metrics must be fitted on TRAIN")
    if (not isinstance(source_members, torch.Tensor) or not isinstance(targets, torch.Tensor)
            or source_members.ndim != 4 or targets.shape != source_members.shape[1:]
            or source_members.shape[0] < 2):
        raise ValueError("source_members must be [J,B,N,C] aligned with TRAIN targets")
    if (len(sample_ids) != targets.shape[0] or len(set(sample_ids)) != len(sample_ids)
            or len(sample_indices) != targets.shape[0] or len(set(sample_indices)) != len(sample_indices)):
        raise ValueError("native calibration requires unique aligned TRAIN sample IDs and positions")
    if (not isinstance(source_checkpoint_sha256, str) or len(source_checkpoint_sha256) != 64
            or any(c not in "0123456789abcdef" for c in source_checkpoint_sha256)):
        raise ValueError("native calibration requires a source checkpoint SHA-256")
    source_members, targets = source_members.detach().cpu(), targets.detach().cpu()
    coordinates = torch.as_tensor(coordinates).detach().cpu()
    first = train_dataset[0]
    if targets.shape[1:] != first.values.shape or targets.shape[-1] != len(train_dataset.data_spec.field_names):
        raise ValueError("native calibration requires the full ordered TRAIN support")
    if coordinates.ndim != 3 or coordinates.shape[1:] != first.coordinates.shape or coordinates.shape[0] not in (1, len(sample_ids)):
        raise ValueError("native calibration coordinates must be [1,N,D] or [B,N,D]")
    if not torch.equal(coordinates, first.coordinates[None].expand_as(coordinates)):
        raise ValueError("native calibration coordinates do not match the TRAIN support")
    if not torch.isfinite(source_members).all() or not torch.isfinite(targets).all():
        raise ValueError("native calibration members and targets must be finite")
    if not hasattr(train_dataset, "physical_weights") or not hasattr(train_dataset, "physical_measure_receipt"):
        raise ValueError("native calibration requires a fitted physical measure provider")
    if first.valid_points is not None and not bool(first.valid_points.all()):
        raise ValueError("native SSS currently requires a compact all-valid domain provider")
    for i, position in enumerate(sample_indices):
        if int(position) < 0 or int(position) >= len(train_dataset):
            raise ValueError("native calibration sample position is outside TRAIN")
        sample = train_dataset[int(position)]
        actual_id = f"{sample.trajectory_id}:{sample.time_index if sample.time_index is not None else 'all'}"
        if (actual_id != sample_ids[i] or not torch.equal(sample.values, targets[i])
                or not torch.equal(sample.coordinates, first.coordinates)
                or not torch.equal(sample.coordinates_raw, first.coordinates_raw)
                or tuple(sample.field_names) != tuple(first.field_names)
                or (sample.valid_points is not None and not bool(sample.valid_points.all()))):
            raise ValueError("native calibration TRAIN ID, target, field order or support differs")
    configs = {name: deepcopy(value) for name, value in config["coherence"]["families"].items()
               if value.get("enabled", True)}
    if set(configs) != {"global_distribution", "multiscale_spectrum", "topology"}:
        raise ValueError("current native bundle requires exactly State, Scale and Shape")
    configs["state_equal_node_v2"] = deepcopy(configs["global_distribution"])
    configs["state_equal_node_v2"]["definition"] = "marginal_mixed_mean_v2"
    geometry = configs["topology"]["geometry"]
    geometry.update(grid_shape=list(train_dataset.data_spec.logical_shape),
                    antialias_downsample=False, native_masked=True)
    geometry.pop("raster_mapping", None)
    query = torch.arange(targets.shape[1])
    families = {}
    started = perf_counter()
    for name, settings in configs.items():
        family = build_coherence_family("global_distribution" if name == "state_equal_node_v2" else name,
                                        settings, train_dataset.data_spec, train_dataset.normalizer).cpu()
        if name == "global_distribution" and not getattr(family, "weighted_measure", False):
            raise ValueError("native bundle requires the physical-weighted State definition")
        if name == "topology" and family.version != "4":
            raise ValueError("native bundle requires finite_primary_v1 Shape")
        if name == "multiscale_spectrum" and (
            family.settings.objective != "signed_band_energy_score"
            or family.settings.joint_energy_score_weight <= 0
        ):
            raise ValueError("native bundle requires the conditional signed Scale score with joint weight")
        if getattr(family, "weighted_measure", False):
            family.bind_measure_artifact(train_dataset.physical_measure_receipt, weights=train_dataset.physical_weights)
        fit = getattr(family, "fit_training_reference", None)
        if callable(fit):
            fit(train_dataset, query_indices=query, device=torch.device("cpu"))
        if getattr(family, "needs_source_calibration", False):
            family.fit_calibration(source_members[0], targets,
                coordinates=coordinates[:1].expand(targets.shape[0], -1, -1),
                provenance={"split": "train", "sample_ids": list(sample_ids),
                            "source_checkpoint_sha256": source_checkpoint_sha256})
        families[name] = family
    rows = []
    for i in range(targets.shape[0]):
        coords = coordinates if coordinates.shape[0] == 1 else coordinates[i:i+1]
        rows.append(_score(families, source_members[:, i:i+1], targets[i:i+1], coords, _contexts(train_dataset)))
    scales = {name: max(float(np.mean([row[2][name] for row in rows])), 1e-6) for name in families}
    states = {name: family.state_artifact() for name, family in families.items()}
    bundle = {"version": "native_sss_bundle_v1", "query_policy": "full_native",
              "source_checkpoint_sha256": source_checkpoint_sha256,
              "data_spec": asdict(train_dataset.data_spec),
              "normalization": train_dataset.normalizer.state_dict(),
              "normalizer_digest": train_dataset.normalizer.digest(),
              "configs": configs, "states": states,
              "summary_families": ["global_distribution", "multiscale_spectrum", "topology"],
              "summary_scales": scales, "summary_floor": 1e-6,
              "scale_denominator": "TRAIN_SOURCE_mean_attraction_primary_plus_weighted_joint",
              "other_denominators": "TRAIN_SOURCE_mean_score_positive_floor",
              "measure": train_dataset.physical_measure_receipt,
              "native_weights": train_dataset.physical_weights.cpu(),
              "coordinates": coordinates[:1].cpu(),
              "raw_coordinates": first.coordinates_raw.cpu(),
              "native_ids": query, "native_mask": torch.ones_like(query, dtype=torch.bool),
              "train_sample_ids": list(sample_ids), "train_sample_indices": list(sample_indices),
              "calibration_rows": rows,
              "fit_seconds": perf_counter()-started}
    bundle["sha256"] = content_sha256(bundle)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output_path)
    return bundle


def load_native_metric_bundle(path):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    expected = bundle.pop("sha256")
    if content_sha256(bundle) != expected:
        raise ValueError("native metric bundle fingerprint mismatch")
    bundle["sha256"] = expected
    spec = DataSpec(**bundle["data_spec"])
    normalizer = FieldNormalizer(**bundle["normalization"])
    families = {}
    for name, config in bundle["configs"].items():
        family = build_coherence_family("global_distribution" if name == "state_equal_node_v2" else name,
                                        config, spec, normalizer).cpu()
        family.load_state_artifact(bundle["states"][name])
        families[name] = family
    return bundle, families


@torch.no_grad()
def score_native_payloads(bundle_path, payload_paths, output_path):
    """Rescore complete saved native members using one immutable TRAIN bundle."""
    bundle, families = load_native_metric_bundle(bundle_path)
    initial = {name: content_sha256(family.state_artifact()) for name, family in families.items()}
    context = {"generated_weights": bundle["native_weights"], "reference_weights": bundle["native_weights"],
               "measure_metadata": {"native_measure_sha256": bundle["measure"]["sha256"]}}
    rows = []
    started = perf_counter()
    for path in payload_paths:
        with np.load(path, allow_pickle=False) as payload:
            members = torch.from_numpy(payload["draws_model"]).unsqueeze(1)
            target = torch.from_numpy(payload["target_model"]).unsqueeze(0)
            coordinates = torch.from_numpy(payload["model_coordinates"]).unsqueeze(0)
            if not torch.equal(coordinates, bundle["coordinates"]):
                raise ValueError("saved native coordinate order disagrees with the metric bundle")
            if (not torch.equal(torch.from_numpy(payload["raw_coordinates"]), bundle["raw_coordinates"])
                    or not torch.equal(torch.from_numpy(payload["native_ids"]), bundle["native_ids"])
                    or not torch.equal(torch.from_numpy(payload["physical_mask"]), bundle["native_mask"])):
                raise ValueError("saved native physical coordinates, IDs or mask disagree with the metric bundle")
            expected_weights = bundle["native_weights"].double()
            expected_weights = expected_weights / expected_weights.sum()
            if not torch.allclose(torch.from_numpy(payload["physical_weights"]).double(), expected_weights, rtol=1e-12, atol=1e-15):
                raise ValueError("saved native physical weights disagree with the metric bundle")
            if (tuple(map(str, payload["field_names"])) != tuple(bundle["data_spec"]["field_names"])
                    or tuple(map(str, payload["field_units"])) != tuple(bundle["data_spec"]["field_units"])
                    or str(payload["normalizer_sha256"].item()) != bundle["normalizer_digest"]):
                raise ValueError("saved native field order, units or normalizer disagree with the metric bundle")
            if members.shape[2:] != target.shape[1:] or members.shape[0] < 2:
                raise ValueError("saved native member and target shapes do not satisfy the conditional score")
            values, components, _, details = _score(families, members, target, coordinates, context)
            summary = float(np.mean([values[name]/bundle["summary_scales"][name] for name in bundle["summary_families"]]))
            rows.append({"payload": str(Path(path).resolve()), "values": values,
                         "components": components, "details": details, "summary": summary})
    if not rows:
        raise ValueError("native scoring requires saved conditions")
    if any(content_sha256(f.state_artifact()) != initial[name] for name, f in families.items()):
        raise RuntimeError("native scoring mutated the frozen metric bundle")
    report = {"metric_bundle_sha256": bundle["sha256"], "query_policy": "full_native",
              "condition_count": len(rows), "rows": rows,
              "means": {name: float(np.mean([row["values"][name] for row in rows])) for name in families},
              "summary": float(np.mean([row["summary"] for row in rows])),
              "scoring_seconds": perf_counter()-started}
    Path(output_path).write_text(json.dumps(_json_value(report), indent=2)+"\n")
    return report
