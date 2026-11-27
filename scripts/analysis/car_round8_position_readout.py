#!/usr/bin/env python3
"""Compare frozen PH/location Shape observers from matched cached fields.

The location map is TRAIN-fitted. Old PH components are reused from sealed
assessments; no persistence call or inference is needed for this readout.
Native and sparse raster location observations remain separately identified.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np
import torch

from phycoflow_reconstruction.coherence.families.topology.geometry import build_physical_cell_overlap_map, rasterize_fields
from phycoflow_reconstruction.coherence.query_sampling import SharedQuerySampler
from phycoflow_reconstruction.coherence.shape_compute import LazyPooledShape
from scripts.analysis.car_round8_cached_summary import checked_ref, ref
from scripts.posttraining.assess_car_round7 import read_row, verify_selected_reference


def feature_triplet(shape, members, reference, queries, kind_weights, *, target_features=None):
    """A/R/S for the same fixed query distance, with every member retained."""
    target = target_features
    if target is None:
        target = shape.position.features(shape._filtration_bank(reference, queries), queries)
    features = [shape.position.features(shape._filtration_bank(x, queries), queries) for x in members]
    distance = shape.position.distance
    attraction = sum(distance(x, target) for x in features) / len(features)
    separation = torch.zeros_like(attraction)
    if len(features) > 1:
        separation = sum(distance(features[a], features[b])
            for a in range(len(features)) for b in range(a+1, len(features))) / (len(features) * (len(features)-1))
    output = {}
    for kind in ("self", "mutual"):
        weights = attraction.new_tensor([float(q[4]) * kind_weights[kind]
            if q[0][0] == kind else 0. for q in queries])
        a = (attraction * weights[:, None]).sum(0)
        r = (separation * weights[:, None]).sum(0)
        output[kind] = np.asarray([float(a.mean()), float(r.mean()), float((a-r).mean())])
    return output


def historical_ph(row):
    output = {}
    for kind in ("self", "mutual"):
        a = sum(float(row[f"shape.{kind}.{leaf}.attraction"]) for leaf in ("finite", "essential"))
        r = sum(float(row[f"shape.{kind}.{leaf}.separation"]) for leaf in ("finite", "essential"))
        output[kind] = np.asarray([a, r, a-r])
    if not np.isclose(sum(v[2] for v in output.values()), row["shape"], rtol=1e-6, atol=1e-7):
        raise ValueError("sealed PH components do not reconstruct the PH-only score")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-receipt", type=Path, required=True)
    parser.add_argument("--joint100-assessment", type=Path, required=True)
    parser.add_argument("--joint500-assessment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    started, cpu = time.perf_counter(), time.process_time()
    torch.set_num_threads(args.threads)
    fit = json.loads(args.fit_receipt.read_text())
    if fit.get("status") != "completed" or len(fit.get("sample_ids", [])) != 800 or not fit.get("test_locked"):
        raise ValueError("completed TRAIN800 fit required")
    core = torch.load(checked_ref(fit["artifact"]), map_location="cpu", weights_only=True)
    shape = LazyPooledShape.from_state_dict(core["shape_artifact"])
    if shape.position is None:
        raise ValueError("frozen positional observation required")
    assessments = {"JointAnchor100": json.loads(args.joint100_assessment.read_text()),
                   "JointAnchor500": json.loads(args.joint500_assessment.read_text())}
    assessment = assessments["JointAnchor500"]
    ids = assessment["sample_ids"]
    if len(ids) != 100 or assessment["split"] != "validation" or assessment["members"] != 4:
        raise ValueError("admitted matched VALIDATION100 J4 bank required")
    for value in assessments.values():
        if (value["status"] != "completed" or value["sample_ids"] != ids
                or value["source_native"] != assessment["source_native"]
                or value["source_selected"] != assessment["source_selected"]
                or not value["test_locked"]):
            raise ValueError("matched sealed SOURCE and child assessments required")
    spec = json.loads(checked_ref(assessment["spec"]).read_text())
    contract = json.loads(checked_ref(spec["assessment_contract"]).read_text())
    panel = json.loads(checked_ref(spec["panel"]).read_text())
    query = torch.as_tensor(panel["query_indices"], dtype=torch.long)
    sampler = SharedQuerySampler(assessment["field_order"], seed=int(contract["shape_query_seed"]), epsilon=1.)
    batches = []
    for _ in range(8):
        batches.append(sampler.draw())
        sampler.finish()
    queries = shape.compile_queries(tuple(batches))
    topology = torch.load(checked_ref(fit["inputs"]["topology_artifact"]), map_location="cpu", weights_only=True)
    if (topology["units"] != "model_units"
            or float(topology["config"]["filtration"]["smoothing_sigma"]) != 0.):
        raise ValueError("this readout requires the fitted model-unit raster with zero smoothing")
    raster_state = topology["state_dict"]
    if not torch.allclose(raster_state["grid_coordinates"].double(), shape.position.state["coordinates"].double()):
        raise ValueError("saved positional and topology raster coordinates differ")
    # Common frozen observers use unit internal coefficients, independently of
    # each child's training prefit coefficients. Match the sealed PH observer.
    kind_weights = {kind: float(topology["config"]["components"][kind]["weight"])
                    for kind in ("self", "mutual")}
    paths = {"SOURCE": {k: checked_ref(assessment["source_" + k]) for k in ("native", "selected")}}
    for name, value in assessments.items():
        paths[name] = {k: checked_ref(value["child_" + k]) for k in ("native", "selected")}
    receipts = {name: {k: json.loads(path.read_text()) for k, path in group.items()}
                for name, group in paths.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    records, native_map = [], None
    with torch.no_grad():
        for index, sid in enumerate(ids):
            native = {name: read_row(paths[name]["native"], receipts[name]["native"], index) for name in paths}
            selected = {name: read_row(paths[name]["selected"], receipts[name]["selected"], index) for name in paths}
            for name in paths:
                verify_selected_reference(native[name], selected[name], query)
            if native_map is None:
                coordinates = native["SOURCE"]["coordinates_raw"]
                native_map = build_physical_cell_overlap_map(coordinates, torch.arange(len(coordinates)), grid_shape=shape.position.grid_shape)
                if not torch.allclose(native_map.grid_coordinates.double(), shape.position.state["coordinates"].double()):
                    raise ValueError("native target raster differs from fitted positional raster")
            def raster(mapping, fields):
                return rasterize_fields(fields, mapping.neighbor_indices,
                    mapping.neighbor_weights.to(fields), mapping.grid_shape)
            def saved_sparse_raster(fields):
                return rasterize_fields(fields, raster_state["neighbor_indices"],
                    raster_state["neighbor_weights"].to(fields), shape.position.grid_shape)
            target_sparse = saved_sparse_raster(selected["SOURCE"]["truth_model"])
            target_native = raster(native_map, native["SOURCE"]["truth_model"])
            sparse_reference_features = shape.position.features(shape._filtration_bank(target_sparse, queries), queries)
            native_reference_features = shape.position.features(shape._filtration_bank(target_native, queries), queries)
            models = {}
            for name in paths:
                row = (assessment if name == "SOURCE" else assessments[name])["per_condition_scores"]["SOURCE" if name == "SOURCE" else "candidate"][index]
                ph = historical_ph(row)
                sparse = feature_triplet(shape, [saved_sparse_raster(x) for x in selected[name]["members_model"]],
                    target_sparse, queries, kind_weights, target_features=sparse_reference_features)
                native_location = feature_triplet(shape, [raster(native_map, x) for x in native[name]["members_model"]],
                    target_native, queries, kind_weights, target_features=native_reference_features)
                output = {}
                for kind in ("self", "mutual"):
                    normalized = sparse[kind] / shape.position.unit
                    output[kind] = {"ph_ARS": ph[kind].tolist(), "location_ARS": sparse[kind].tolist(),
                        "location_unit_matched_ARS": normalized.tolist(),
                        "combined_ARS": (0.5 * (ph[kind] + normalized)).tolist(),
                        "native_location_unit_matched_ARS": (native_location[kind] / shape.position.unit).tolist()}
                models[name] = output
            records.append({"sample_id": sid, "models": models})
    output = {"schema": "car_round8_common_position_readout_v1", "status": "completed",
        "sample_ids": ids, "split": "validation", "members": 4, "records": records,
        "location_unit": shape.position.unit, "queries": [{"category": q[0], "fields": q[1], "signs": q[2], "line": q[3], "weight": q[4]} for q in queries],
        "query_batches": [asdict(batch) for batch in batches], "kind_weights": kind_weights,
        "definition": "Common PH-only and half(PH+location/location_unit) observers; A/R/S retained. Native location uses direct native-to-raster values; selected location uses separately queried sparse values. PH is reused only on its original selected geometry.",
        "no_new_inference": True, "additional_persistence_calls": 0, "test_locked": True,
        "inputs": {"fit": ref(args.fit_receipt), "joint100": ref(args.joint100_assessment),
                   "joint500": ref(args.joint500_assessment), "contract": spec["assessment_contract"], "script": ref(Path(__file__))},
        "timing": {"wall_seconds": time.perf_counter()-started, "cpu_seconds": time.process_time()-cpu, "threads": args.threads}}
    (args.output / "readout.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
