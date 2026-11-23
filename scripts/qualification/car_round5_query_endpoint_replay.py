#!/usr/bin/env python3
"""Paired shared/eight-group Shape endpoint-gradient replay on cached TRAIN800.

The squared centered-gradient statistic covers every endpoint tensor entry. This
is an endpoint diagnostic only; parameter-gradient and finite-action replay are
separate requirements for estimator adoption.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from phycoflow_reconstruction.coherence import build_enabled_families
from phycoflow_reconstruction.coherence.query_runtime import _shape_grid
from phycoflow_reconstruction.coherence.query_sampling import ConditionQueryGroups, SharedQuerySampler
from phycoflow_reconstruction.coherence.shape_compute import LazyPooledShape
from phycoflow_reconstruction.data.factory import open_field_dataset
from phycoflow_reconstruction.training.run_store import file_sha256
from phycoflow_reconstruction.training.scalable_fitting import validate_scalable_artifacts
from phycoflow_reconstruction.training.internal_calibration import InternalCalibration


def query_groups(fields, seed, start, count):
    base = seed * 1000003 + start
    batches = tuple(SharedQuerySampler(fields, seed=base + 7919 * i, epsilon=1).draw()
                    for i in range(8))
    grouped = ConditionQueryGroups(batches)
    return [(slice(0, count), batches[0])], list(grouped.slices(count))


def gram_summary(gram):
    gram = torch.as_tensor(gram, dtype=torch.float64)
    count = gram.shape[0]
    centered = torch.trace(gram) / count - gram.mean()
    return {"mean_gradient_squared_norm": float(gram.mean()),
            "centered_gradient_squared_norm": float(centered.clamp_min(0)),
            "mean_squared_gradient_norm": float(torch.trace(gram) / count)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--seconds-limit", type=float, required=True)
    args = parser.parse_args()
    if args.batch_size != 64 or not 1 <= args.seeds <= 8 or args.threads < 1 or args.seconds_limit <= 0:
        parser.error("B64, one to eight seeds, positive threads and seconds limit required")
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    source = json.loads(args.setup.read_text())["sources"]["ffm"]["source_fit"]
    run, panel_path, core_path = map(Path, (source["run"], source["source_panel"], source["core"]))
    for path, key in ((panel_path, "source_panel_sha256"), (core_path, "core_sha256")):
        if file_sha256(path) != source[key]:
            raise ValueError(f"input hash mismatch: {path}")
    panel = torch.load(panel_path, map_location="cpu", weights_only=True)
    core = torch.load(core_path, map_location="cpu", weights_only=True)
    validate_scalable_artifacts(core, panel["provenance"])
    ids = panel["sample_ids"]
    if panel["schema"] != "car_source_panel_v1" or len(ids) != 800 or ids != panel["provenance"]["sample_ids"] or panel["members"].shape[1] < 2:
        raise ValueError("matched TRAIN800 SOURCE J2 panel required")
    candidate_config = yaml.safe_load(args.config.read_text())
    settings = candidate_config["coherence"]["scalable_anchored_coherence"]
    if settings["common_core"]["source_panel_sha256"] != source["source_panel_sha256"]:
        raise ValueError("candidate and cached SOURCE panels differ")
    units_record = json.loads(args.units.read_text())
    if (units_record.get("status") != "completed" or units_record.get("sample_count") != 800
            or any(units_record.get("floor_active", {}).values())
            or units_record.get("source_panel_sha256") != source["source_panel_sha256"]
            or units_record.get("source_checkpoint_sha256") != panel["provenance"]["source_checkpoint_sha256"]):
        raise ValueError("matching completed, floor-free SOURCE TRAIN800 units required")
    shape_unit = float(units_record["units"]["shape"])
    if not 0 < shape_unit < float("inf"):
        raise ValueError("invalid frozen Shape unit")
    prefit_ref = settings["common_prefit"]
    prefit_path = Path(prefit_ref["path"])
    if file_sha256(prefit_path) != prefit_ref["sha256"]:
        raise ValueError("common prefit hash mismatch")
    prefit = json.loads(prefit_path.read_text())
    binding = prefit.get("binding", {})
    for key in ("source_checkpoint_sha256", "normalizer_sha256", "support_sha256", "field_order", "sample_ids"):
        if binding.get(key) != panel["provenance"].get(key):
            raise ValueError(f"common prefit binding mismatch: {key}")
    calibration = InternalCalibration(prefit["priorities"], prefit["units"], prefit["active"])
    calibration.load_prefit(prefit)
    config = candidate_config
    dataset = open_field_dataset(config["dataset"], split="train")
    family = build_enabled_families(config["coherence"], dataset.data_spec, dataset.normalizer)["topology"]
    family_path = run / "artifacts/topology_family.pt"
    manifest = json.loads((run / "run_manifest.json").read_text())
    if file_sha256(family_path) != manifest["coherence_family_state_sha256s"]["topology"]:
        raise ValueError("topology family hash mismatch")
    if file_sha256(family_path) != candidate_config["coherence"]["frozen_family_artifacts"]["topology"]["sha256"]:
        raise ValueError("candidate and cached replay topology artifacts differ")
    family.load_state_artifact(torch.load(family_path, map_location="cpu", weights_only=True))
    family = family.to(device)
    coordinates = panel["coordinates"].to(device)
    family._raster_map(coordinates)
    mask = family.domain_mask if family.native_masked else None
    shape = LazyPooledShape.from_state_dict(core["shape_artifact"])
    weights = {k: (float(family.spatial_objective.weights[f"{k}.persistence"])
                   * calibration.coefficients[f"shape.{k}"]
                   * float(getattr(family, "family_weight", 1.0)) / (3 * shape_unit))
               for k in ("self", "mutual")}
    total_variants = 2 * args.seeds
    gram = torch.zeros(total_variants, total_variants, dtype=torch.float64)
    scores = torch.zeros(total_variants, dtype=torch.float64)
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"schema": "car_round5_endpoint_query_replay_v1", "status": "running",
              "sample_count": 800, "panel_sha256": source["source_panel_sha256"],
              "sample_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
              "core_sha256": source["core_sha256"], "family_sha256": file_sha256(family_path),
              "seeds": args.seeds, "device": str(device), "batch_size": 64,
              "endpoint_weighting": "physical Shape weights, frozen inner coefficients, family weight and 1/(3*outer Shape unit); panel mean",
              "shape_weights": weights, "common_prefit_sha256": prefit_ref["sha256"],
              "units_sha256": file_sha256(args.units), "config_sha256": file_sha256(args.config),
              "conditioning": "cached fixed 256-T observation endpoints; parameter replay uses fresh observations",
              "completed_conditions": 0}
    def save():
        (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    save()
    for start in range(0, 800, 64):
        if time.monotonic() - started >= args.seconds_limit:
            result.update(status="budget_expired", elapsed_seconds=time.monotonic()-started)
            save()
            return
        batch_started = time.monotonic()
        sl = slice(start, min(start + 64, 800))
        reference = panel["reference"][sl].to(device)
        count = len(reference)
        gradients = []
        for seed in range(1, args.seeds + 1):
            shared, grouped = query_groups(tuple(family.field_names), seed, start, count)
            for arm, plan in enumerate((shared, grouped)):
                first = panel["members"][sl, 0].to(device).detach().requires_grad_()
                second = panel["members"][sl, 1].to(device).detach().requires_grad_()
                grids = [_shape_grid(family, first, reference), _shape_grid(family, second, reference)]
                with torch.no_grad():
                    target = _shape_grid(family, reference, reference)
                contributions = []
                for group_sl, query in plan:
                    value = shape.score(grids[0][group_sl], grids[1][group_sl], target[group_sl], query,
                                        valid_mask=mask, mode="conditional")
                    contributions.append(sum(weights[k] * value[k].sum() for k in weights))
                loss = sum(contributions) / 800
                scores[2*(seed-1)+arm] += float(loss.detach().cpu())
                grad = torch.autograd.grad(loss, (first, second))
                gradients.append(torch.cat([g.detach().reshape(-1) for g in grad]).cpu().double())
        rows = torch.stack(gradients)
        gram += rows @ rows.T
        result["completed_conditions"] += count
        result["elapsed_seconds"] = time.monotonic() - started
        if start == 0:
            result["first_batch_seconds"] = time.monotonic() - batch_started
            result["forecast_total_seconds"] = result["elapsed_seconds"] + 12 * result["first_batch_seconds"]
            if result["forecast_total_seconds"] > args.seconds_limit:
                result["status"] = "forecast_exceeds_budget"
                save()
                return
        save()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    shared_indices = list(range(0, total_variants, 2))
    grouped_indices = list(range(1, total_variants, 2))
    result.update(status="completed", elapsed_seconds=time.monotonic()-started,
                  scores=scores.tolist(), gradient_gram=gram.tolist(),
                  shared=gram_summary(gram[shared_indices][:, shared_indices]),
                  grouped=gram_summary(gram[grouped_indices][:, grouped_indices]))
    save()


if __name__ == "__main__":
    main()
