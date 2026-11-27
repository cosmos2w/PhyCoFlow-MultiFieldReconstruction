#!/usr/bin/env python3
"""Measure frozen soft-excursion sensitivity on the cached TRAIN800 panel.

Uses the exact fitted raster, physical mass, category thresholds, temperature,
target query weights and all four cached SOURCE members. No inference or PH.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import torch

from phycoflow_reconstruction.coherence.families.topology.geometry import rasterize_fields
from phycoflow_reconstruction.coherence.positional_shape import _key
from phycoflow_reconstruction.coherence.query_sampling import full_target_query_batch
from phycoflow_reconstruction.coherence.shape_compute import LazyPooledShape
from scripts.analysis.car_round8_cached_summary import checked_ref, ref
from scripts.posttraining.assess_car_round7 import read_row


METRICS = ("occupancy", "absolute_filtration_derivative", "pixel_near_empty_mass_fraction",
           "pixel_near_full_mass_fraction", "window_near_empty_fraction", "window_near_full_fraction")


def observe(shape, raster, queries):
    """Return category numerators and weights for one reference plus J members.

    `raster` has [1+J,C,H,W]. Each category pools its fixed target query
    weights, while its endpoints are kept distinct for condition/member pooling.
    """
    position = shape.position
    if position is None or raster.ndim != 4 or len(raster) < 2:
        raise ValueError("frozen positional Shape and reference/member rasters required")
    filtration = shape._filtration_bank(raster, queries).permute(1, 0, 2, 3).flatten(-2)
    masses = position.mass.flatten().to(filtration)
    active = masses > 0
    if not torch.isfinite(filtration[..., active]).all():
        raise ValueError("nonfinite valid filtration")
    filtration = torch.where(active, filtration, torch.zeros_like(filtration))
    mass = masses / masses.sum()
    windows = position.windows.to(filtration)
    categories = [_key(q[0]) for q in queries]
    thresholds = filtration.new_tensor([position.thresholds[key] for key in categories])
    temperatures = filtration.new_tensor([position.temperatures[key] for key in categories])
    per_threshold = []
    for index in range(3):
        soft = torch.sigmoid((thresholds[:, None, index, None] - filtration)
                             / temperatures[:, None, None])
        regions = torch.einsum("qbp,kp->qbk", soft, windows)
        per_threshold.append(torch.stack((
            torch.einsum("qbp,p->qb", soft, mass),
            torch.einsum("qbp,p->qb", soft * (1 - soft), mass) / temperatures[:, None],
            torch.einsum("qbp,p->qb", (soft <= .01).to(soft), mass),
            torch.einsum("qbp,p->qb", (soft >= .99).to(soft), mass),
            (regions <= .01).to(soft).mean(-1),
            (regions >= .99).to(soft).mean(-1),
        ), dim=-1))
    values = torch.stack(per_threshold, dim=2).double()  # [Q, 1+J, threshold, metric]
    result = {}
    for category in sorted(set(categories)):
        ids = [index for index, value in enumerate(categories) if value == category]
        weight = torch.as_tensor([float(queries[index][4]) * position.state["kind_weights"][queries[index][0][0]]
                                  for index in ids], dtype=torch.double)
        if weight.sum() <= 0:
            raise ValueError("category requires positive fixed target weight")
        result[category] = ((values[ids] * weight[:, None, None, None]).sum(0), float(weight.sum()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--limit", type=int, help="smoke only; partial output is not TRAIN800 sensitivity")
    args = parser.parse_args()
    if args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error("threads and optional limit must be positive")
    started, cpu = time.perf_counter(), time.process_time()
    torch.set_num_threads(args.threads)
    fit = json.loads(args.fit_receipt.read_text())
    if fit.get("status") != "completed" or len(fit.get("sample_ids", ())) != 800 or not fit.get("test_locked"):
        raise ValueError("the completed TRAIN800 positional fit is required")
    core = torch.load(checked_ref(fit["artifact"]), map_location="cpu", weights_only=True)
    shape = LazyPooledShape.from_state_dict(core["shape_artifact"])
    if shape.position is None:
        raise ValueError("missing frozen position feature")
    topology_path = checked_ref(fit["inputs"]["topology_artifact"])
    topology = torch.load(topology_path, map_location="cpu", weights_only=True)
    geometry = topology["config"]["geometry"]
    if (topology["units"] != "model_units" or geometry["periodic"]
            or topology["config"]["filtration"].get("smoothing_sigma", 0.0) != 0.0):
        raise ValueError("this sensitivity readout requires the fitted unsmoothed physical raster")
    state = topology["state_dict"]
    indices, weights, grid = state["neighbor_indices"], state["neighbor_weights"], state["grid_coordinates"]
    if not torch.equal(grid.double(), shape.position.state["coordinates"].double()):
        raise ValueError("frozen position and topology grid differ")
    selected_path = checked_ref(fit["inputs"]["source_selected_receipt"])
    selected = json.loads(selected_path.read_text())
    if (selected.get("status") != "completed" or selected.get("split") != "train"
            or selected.get("support") != "selected" or selected.get("members") != 4
            or selected.get("sample_ids") != fit["sample_ids"]):
        raise ValueError("matched TRAIN800 J4 SOURCE cache required")
    queries = shape.compile_queries((full_target_query_batch(len(shape.mean)),), exhaustive=True)
    if len(queries) != fit["query_count"]:
        raise ValueError("frozen target query count differs")
    total = len(fit["sample_ids"]) if args.limit is None else min(args.limit, len(fit["sample_ids"]))
    totals, norms = {}, {}
    with torch.no_grad():
        for index in range(total):
            row = read_row(selected_path, selected, index)
            if row["sample_id"] != fit["sample_ids"][index] or len(row["members_model"]) != 4:
                raise ValueError("TRAIN condition or member identity changed")
            endpoints = torch.cat((row["truth_model"], *row["members_model"]), dim=0)
            raster = rasterize_fields(endpoints, indices, weights.to(endpoints), tuple(grid.shape[:2])).float()
            for category, (values, weight) in observe(shape, raster, queries).items():
                row_total = torch.stack((values[0], values[1:].mean(0)))
                totals[category] = totals.get(category, torch.zeros_like(row_total)) + row_total
                norms[category] = norms.get(category, 0.0) + weight
    categories = {}
    for category, sums in sorted(totals.items()):
        average = sums / norms[category]
        levels = shape.position.thresholds[category]
        categories[category] = {"thresholds": levels,
            "temperature": shape.position.temperatures[category],
            "duplicate_thresholds": len(set(levels)) != len(levels),
            "target_query_weight": norms[category] / total,
            "reference": {metric: average[0, :, i].tolist() for i, metric in enumerate(METRICS)},
            "source_members": {metric: average[1, :, i].tolist() for i, metric in enumerate(METRICS)}}
    output = {"schema": "car_round8_train_position_sensitivity_v1",
        "status": "completed" if total == 800 else "smoke_only", "split": "train",
        "sample_ids": fit["sample_ids"][:total], "sample_count": total, "members": 4,
        "categories": categories, "saturation_cutoffs": {"near_empty": .01, "near_full": .99},
        "scope": "frozen selected-support raster; equal condition and member pooling; physical raster mass and fixed target query weights",
        "derivative_units": "inverse standardized filtration units",
        "no_model_inference": True, "additional_persistence_calls": 0, "test_locked": True,
        "inputs": {"fit_receipt": ref(args.fit_receipt), "artifact": fit["artifact"],
                   "topology_artifact": fit["inputs"]["topology_artifact"],
                   "source_selected_receipt": fit["inputs"]["source_selected_receipt"], "script": ref(Path(__file__))},
        "timing": {"wall_seconds": time.perf_counter() - started,
                   "cpu_seconds": time.process_time() - cpu, "threads": args.threads}}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "sensitivity.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
