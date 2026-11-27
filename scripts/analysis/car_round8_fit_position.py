#!/usr/bin/env python3
"""Fit the opt-in positional Shape observation from cached TRAIN SOURCE fields.

Two passes use the same admitted selected-support conditions and all four
independent cached members. No model inference or VALIDATION selection occurs.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import torch

from phycoflow_reconstruction.coherence.families.topology.geometry import coordinate_digest, rasterize_fields
from phycoflow_reconstruction.coherence.positional_shape import PositionalShapeFitter
from phycoflow_reconstruction.coherence.query_sampling import full_target_query_batch
from phycoflow_reconstruction.coherence.shape_compute import LazyPooledShape
from phycoflow_reconstruction.training.scalable_fitting import _digest
from scripts.analysis.car_round8_cached_summary import checked_ref, ref
from scripts.posttraining.assess_car_round7 import read_row, verify_selected_reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--source-selected", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--limit", type=int, help="smoke only; a partial result is never a fitted production artifact")
    args = p.parse_args()
    started, cpu = time.perf_counter(), time.process_time()
    torch.set_num_threads(args.threads)
    spec = json.loads(args.spec.read_text())
    if spec.get("split") != "train" or spec.get("test_locked") is not True:
        raise ValueError("TRAIN-only, TEST-locked specification required")
    panel_path = checked_ref(spec["panel"]); panel = json.loads(panel_path.read_text())
    source_path = checked_ref(spec["source_native_receipt"]); source = json.loads(source_path.read_text())
    selected_path = args.source_selected.resolve(strict=True); selected = json.loads(selected_path.read_text())
    ids = panel["sample_ids"]
    if (len(ids) != 800 or selected.get("status") != "completed" or selected.get("split") != "train"
            or selected.get("support") != "selected" or selected.get("members") != 4
            or selected.get("sample_ids") != ids or source.get("sample_ids") != ids
            or selected.get("panel_sha256") != spec["panel"]["sha256"]
            or selected.get("source_sha256") != spec["source_checkpoint"]["sha256"]
            or selected.get("checkpoint_sha256") != spec["source_checkpoint"]["sha256"]):
        raise ValueError("matched admitted SOURCE TRAIN800 J4 selected bank required")
    config_path = checked_ref(spec["core_config"])
    manifest = json.loads(checked_ref(spec["core_manifest"]).read_text())
    topology_path = config_path.parent / "artifacts/topology_family.pt"
    checked_ref({"path": str(topology_path), "sha256": manifest["coherence_family_state_sha256s"]["topology"]})
    topology = torch.load(topology_path, map_location="cpu", weights_only=True)
    if topology["units"] != "model_units" or topology["config"]["geometry"]["periodic"]:
        raise ValueError("this fitter's declared geometry is model-unit, nonperiodic physical raster")
    core = torch.load(checked_ref(spec["core"]), map_location="cpu", weights_only=True)
    prefit = json.loads(checked_ref(spec["prefit"]).read_text())
    shape = LazyPooledShape.from_state_dict(core["shape_artifact"])
    query = torch.as_tensor(panel["query_indices"], dtype=torch.long)
    source_row = read_row(source_path, source, 0); selected_row = read_row(selected_path, selected, 0)
    verify_selected_reference(source_row, selected_row, query)
    if coordinate_digest(selected_row["coordinates_model"][0]) != topology["geometry_sha256"]:
        raise ValueError("selected model coordinates differ from the saved topology raster")
    state = topology["state_dict"]
    indices, weights, grid = state["neighbor_indices"], state["neighbor_weights"], state["grid_coordinates"]
    raw_coordinates = source_row["coordinates_raw"]
    extents = tuple((float(raw_coordinates[:, axis].min()), float(raw_coordinates[:, axis].max())) for axis in range(2))
    extent_area = (extents[0][1] - extents[0][0]) * (extents[1][1] - extents[1][0])
    mass = torch.full(grid.shape[:2], extent_area / (grid.shape[0] * grid.shape[1]), dtype=torch.float64)
    kind_weights = {kind: float(topology["config"]["components"][kind]["weight"]) *
                    float(prefit["coefficients"]["shape." + kind]) for kind in ("self", "mutual")}
    fitter = PositionalShapeFitter(shape, grid, mass, periods=(None, None), axis_extents=extents,
                                   kind_weights=kind_weights, threshold_chunk_size=1)
    queries = shape.compile_queries((full_target_query_batch(len(shape.mean)),), exhaustive=True)
    count = len(ids) if args.limit is None else min(len(ids), args.limit)
    def raster(values):
        return rasterize_fields(values, indices, weights.to(values), tuple(grid.shape[:2])).float()
    for index in range(count):
        row = read_row(selected_path, selected, index)
        fitter.observe_reference(raster(row["truth_model"]), queries)
    fitter.freeze_thresholds()
    for index in range(count):
        row = read_row(selected_path, selected, index)
        reference = raster(row["truth_model"])
        for member in row["members_model"]:
            fitter.observe_pair(raster(member), reference, queries)
    position = fitter.freeze()
    shape.set_position(position)
    new_core = copy.deepcopy(core); new_core["shape_artifact"] = shape.state_dict()
    new_core["sha256"] = _digest({k: v for k, v in new_core.items() if k != "sha256"})
    args.output.mkdir(parents=True, exist_ok=False)
    artifact = args.output / "spatial_shape_observer.pt"
    torch.save(new_core, artifact)
    receipt = {"schema": "car_round8_train_position_fit_v1", "status": "completed" if count == len(ids) else "smoke_only",
               "fit_scope": "cached SOURCE selected-support TRAIN800, all J4 independent members; no inference, validation fitting or TEST",
               "sample_ids": ids[:count], "members": 4, "query_count": len(queries),
               "kind_weights": kind_weights, "query_rule": "complete fixed target field/pair queries with exhaustive sign/line categories",
               "raster": "saved Q4096 sparse-IDW physical-cell overlap geometry, 32x128; geometry mass is uniform cell area",
               "axis_extents_dataset_units": extents, "location_unit": position["location_unit"],
               "fit_counts": position["fit_counts"], "thresholds": position["thresholds"],
               "temperatures": position["temperatures"], "source_core_sha256": core["sha256"],
               "new_core_sha256": new_core["sha256"], "artifact": ref(artifact),
               "inputs": {"spec": ref(args.spec), "panel": ref(panel_path), "source_native_receipt": ref(source_path),
                          "source_selected_receipt": ref(selected_path), "topology_artifact": ref(topology_path),
                          "core": spec["core"], "prefit": spec["prefit"], "script": ref(Path(__file__))},
               "timing": {"wall_seconds": time.perf_counter()-started, "cpu_seconds": time.process_time()-cpu, "threads": args.threads},
               "test_locked": True}
    with (args.output / "fit_receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write("\n")


if __name__ == "__main__":
    main()
