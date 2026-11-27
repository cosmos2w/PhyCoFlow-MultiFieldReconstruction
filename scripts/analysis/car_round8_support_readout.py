#!/usr/bin/env python3
"""Matched cached native/selected support and direct State/Scale diagnostics.

The declared bank is read once per row. No model predictions are generated.
The output compares a native slice and separately queried selected endpoint,
and never treats those two laws as automatically identical.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np
import torch

from phycoflow_reconstruction.coherence.families.topology.geometry import build_physical_cell_overlap_map, rasterize_fields
from phycoflow_reconstruction.coherence.query_runtime import _scale_statistics
from phycoflow_reconstruction.coherence.state_compute import common_distance, exact_common_embeddings
from scripts.analysis.car_round8_cached_summary import checked_ref, ref
from scripts.posttraining.assess_car_round2_checkpoints import matched_rows
from scripts.posttraining.assess_car_round7 import read_row, verify_selected_reference


def build_state_scale(spec, source):
    """Load only unchanged families used here; the historical PH source is not reloaded."""
    import yaml
    from phycoflow_reconstruction.coherence import build_enabled_families
    from phycoflow_reconstruction.data.factory import open_field_dataset
    config_path = checked_ref(spec["core_config"])
    cfg = yaml.safe_load(config_path.read_text())
    dataset = open_field_dataset(cfg["dataset"], split="train")
    try:
        if dataset.normalizer.digest() != source["normalizer_sha256"] or list(dataset.field_names) != source["field_order"]:
            raise ValueError("frozen normalizer or field order differs")
        families = build_enabled_families(cfg["coherence"], dataset.data_spec, dataset.normalizer)
        manifest = json.loads(checked_ref(spec["core_manifest"]).read_text())
        for name in ("global_distribution", "multiscale_spectrum"):
            path = config_path.parent / "artifacts" / f"{name}_family.pt"
            checked_ref({"path": str(path), "sha256": manifest["coherence_family_state_sha256s"][name]})
            families[name].load_state_artifact(torch.load(path, map_location="cpu", weights_only=True))
        core = torch.load(checked_ref(spec["core"]), map_location="cpu", weights_only=True)
        if core["provenance"]["source_checkpoint_sha256"] != spec["source_checkpoint"]["sha256"]:
            raise ValueError("core observer SOURCE identity differs")
        return dataset, families, core
    except BaseException:
        dataset.close()
        raise


def state_triplet(family, bank, members, truth, masses, *, return_groups=False):
    """Exact weighted-CDF State A, R, S using the historical unit coefficients."""
    converted = [family._in_declared_units(m, truth)[0] for m in members]
    target = family._in_declared_units(members[0], truth)[1]
    context = {"generated_weights": masses.unsqueeze(0), "reference_weights": masses.unsqueeze(0)}
    left, right = family._endpoint_measures(members[0], truth, context, required=True)
    endpoints = exact_common_embeddings([*converted, target], [left] * len(members) + [right],
        bank["directions"], family.components_by_key["self_marginal_w2"].channel_weights)
    reference = endpoints.pop()
    kwargs = {"marginal_weight": family.component_weights["self_marginal_w2"],
              "mixed_weight": family.component_weights.get("cross_joint_topk_swd", 0.0)}
    def dist(a, b):
        return common_distance(a[0], a[1], b[0], b[1], **kwargs)
    j = len(endpoints)
    attraction = torch.stack([dist(x, reference) for x in endpoints]).mean(0)
    separation = sum((dist(endpoints[i], endpoints[k]) for i in range(j) for k in range(i+1, j)),
                     torch.zeros_like(attraction)) / (j * (j-1))
    total = [float(attraction.mean()), float(separation.mean()), float((attraction-separation).mean())]
    if not return_groups:
        return total

    # These are direct distances on each weighted subspace of the *same*
    # embedding, not the detached gradient probes returned by score_family.
    # Their A/R/S values do not add to the score of the joint root norm.
    def grouped_dist(a, b):
        marginal = ((a[0] - b[0]).square().sum(-1) * kwargs["marginal_weight"]).sqrt()
        mixed = (torch.zeros_like(marginal) if a[1] is None else
                 ((a[1] - b[1]).square().sum(-1) * kwargs["mixed_weight"]).sqrt())
        return torch.stack((marginal, mixed))

    group_attraction = torch.stack([grouped_dist(x, reference) for x in endpoints]).mean(0)
    group_separation = sum(
        (grouped_dist(endpoints[i], endpoints[k]) for i in range(j) for k in range(i + 1, j)),
        torch.zeros_like(group_attraction),
    ) / (j * (j - 1))
    groups = {
        name: [float(group_attraction[index].mean()), float(group_separation[index].mean()),
               float((group_attraction[index] - group_separation[index]).mean())]
        for index, name in enumerate(("marginal", "mixed"))
    }
    return total, groups


def scale_context(family, members, truth, *, return_piece_means=False):
    """Signed rho errors and floor-aware amplitude on the frozen Q4096 basis."""
    coefficient, means, _ = _scale_statistics(family, torch.cat([*members, truth], dim=0))
    filters = family.filters.to(coefficient)
    floor = family.power_floor.to(coefficient)
    power = torch.einsum("sk,bkc->bsc", filters.square(), coefficient.square())
    pairs = list(itertools.combinations(range(coefficient.shape[-1]), 2))
    rho = []
    for left, right in pairs:
        cross = torch.einsum("sk,bk->bs", filters.square(), coefficient[..., left] * coefficient[..., right])
        rho.append(cross / ((power[..., left] + floor[:, left]) * (power[..., right] + floor[:, right])).sqrt())
    rho = torch.stack(rho, dim=-1)
    signed = rho[:-1] - rho[-1:]
    logpower = torch.log1p(power / floor)
    result = {"rho_signed_error_mean_member": signed.mean(0).tolist(),
            "rho_absolute_error_mean_member": signed.abs().mean(0).tolist(),
            "log1p_power_over_floor_signed_error_mean_member": (logpower[:-1] - logpower[-1:]).mean(0).tolist(),
            "log1p_power_over_floor_absolute_error_mean_member": (logpower[:-1] - logpower[-1:]).abs().mean(0).tolist(),
            "reference_power_over_floor": (power[-1] / floor).tolist(),
            "member_power_over_floor_mean": (power[:-1] / floor).mean(0).tolist(),
            "reference_low_energy": (power[-1] <= floor).double().tolist(),
            "member_low_energy_fraction": (power[:-1] <= floor).double().mean(0).tolist(),
            "field_pair_indices": [list(pair) for pair in pairs]}
    if not return_piece_means:
        return result
    difference = means[:-1] - means[-1:]
    return result, {"signed_error_mean_member": difference.mean(0).tolist(),
                    "absolute_error_mean_member": difference.abs().mean(0).tolist()}


def direct_readout_metadata(state_family, scale_family, field_order):
    """Bind direct readouts to the actual weighted State and Scale geometry."""
    masses = scale_family.piece_measures.detach().double().sum(dim=1)
    if masses.ndim != 1 or not len(masses) or not torch.isfinite(masses).all() or (masses <= 0).any():
        raise ValueError("positive finite frozen piece masses are required")
    if list(scale_family.field_names) != list(field_order):
        raise ValueError("frozen Scale field ordering differs")
    weights = {"marginal": float(state_family.component_weights["self_marginal_w2"]),
               "mixed": float(state_family.component_weights.get("cross_joint_topk_swd", 0.0))}
    return weights, {"piece_indices": list(range(len(masses))),
                     "mass_fractions": (masses / masses.sum()).tolist(),
                     "unit_space": scale_family.units, "field_order": list(field_order)}


def native_support_risks(row, query, *, seed):
    """Physical MSE and fixed/exact-complement/sample decomposition on one native draw."""
    members = row["members_model"].double()[:, 0]
    truth = row["truth_model"].double()[0]
    scale = row["normalizer_scale"].double()
    weights = row["native_masses"].double()
    e2 = ((members-truth)*scale).square().mean(0)
    n = len(weights)
    if (weights < 0).any() or weights.sum() <= 0 or len(query.unique()) != len(query):
        raise ValueError("invalid native mass or selected support")
    keep = torch.ones(n, dtype=torch.bool); keep[query] = False
    complement = torch.arange(n)[keep]
    count = min(512, len(complement))
    rng = torch.Generator().manual_seed(seed)
    sampled = complement[torch.randperm(len(complement), generator=rng)[:count]]
    total = weights.sum()
    selected = (e2[query] * weights[query, None]).sum(0) / total
    remaining = (e2[complement] * weights[complement, None]).sum(0) / total
    sampled_remaining = (e2[sampled] * weights[sampled, None]).sum(0) * (len(complement)/count) / total
    return {"native": (selected+remaining).tolist(), "fixed_contribution_to_native": selected.tolist(),
            "complement_contribution_to_native": remaining.tolist(),
            "fixed_self_normalized": (selected * total / weights[query].sum()).tolist(),
            "fixed_plus_uniform_complement512_estimate": (selected+sampled_remaining).tolist(),
            "fixed_mass_fraction": float(weights[query].sum()/total), "complement_sample_count": count}


def raster_fields(mapping, values):
    return rasterize_fields(values, mapping.neighbor_indices, mapping.neighbor_weights.to(values), mapping.grid_shape)


def raster_comparison(native, selected, query, native_map, sparse_map):
    scale, offset = native["normalizer_scale"], native["normalizer_offset"]
    truth = native["truth_model"] * scale + offset
    members = native["members_model"][:, 0] * scale + offset
    selected_members = selected["members_model"][:, 0] * scale + offset
    reference_native = raster_fields(native_map, truth)
    reference_sparse = raster_fields(sparse_map, truth[:, query])
    prediction_native = raster_fields(native_map, members)
    prediction_slice = raster_fields(sparse_map, members[:, query])
    prediction_selected = raster_fields(sparse_map, selected_members)
    def mse(x): return x.square().mean(dim=(-1, -2)).mean(dim=0).double().tolist()
    def paired(prediction, reference):
        return {"member_mse": mse(prediction-reference),
                "mean_output_mse": mse(prediction.mean(0, keepdim=True)-reference)}
    return {"reference_sparse_minus_native_mse": mse(reference_sparse-reference_native),
            "prediction_slice_minus_native_mse": mse(prediction_slice-prediction_native),
            "separately_queried_minus_native_slice_mse": mse(prediction_selected-prediction_slice),
            "native_reconstruction": paired(prediction_native, reference_native),
            "sparse_slice_reconstruction": paired(prediction_slice, reference_sparse),
            "separately_queried_reconstruction": paired(prediction_selected, reference_sparse)}, (reference_native, prediction_native)


def plot_examples(examples, output, field_order, grid_coordinates):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, y = grid_coordinates[..., 0].numpy(), grid_coordinates[..., 1].numpy()
    extent = [x.min(), x.max(), y.min(), y.max()]
    files = []
    for label, item in examples.items():
        truth, source, child = [item[k].detach().cpu().numpy() for k in ("reference", "SOURCE", "JointAnchor500")]
        fig, axes = plt.subplots(len(field_order), 5, figsize=(17, 2.8*len(field_order)), constrained_layout=True)
        for f, name in enumerate(field_order):
            t, s, c = truth[f], source[f], child[f]
            limit = max(float(np.abs(s-t).max()), float(np.abs(c-t).max()), 1e-20)
            lo, hi = min(float(v.min()) for v in (t,s,c)), max(float(v.max()) for v in (t,s,c))
            norm = max(float(np.sqrt(np.mean(t*t))), 1e-30)
            errors = [float(np.sqrt(np.mean((v-t)**2))/norm) for v in (s,c)]
            panels = [(t, "Reference", "viridis", lo, hi), (s, f"SOURCE mean\nrelative L2 {errors[0]:.3f}", "viridis", lo, hi),
                      (c, f"Joint500 mean\nrelative L2 {errors[1]:.3f}", "viridis", lo, hi),
                      (s-t, f"SOURCE error\nrelative L2 {errors[0]:.3f}", "coolwarm", -limit, limit),
                      (c-t, f"Joint500 error\nrelative L2 {errors[1]:.3f}", "coolwarm", -limit, limit)]
            for col, (array, title, cmap, vmin, vmax) in enumerate(panels):
                ax = axes[f, col]
                image = ax.imshow(array, origin="lower", extent=extent, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(f"{name}: {title}", fontsize=9)
                ax.set_xlabel("raw coordinate 1 (dataset units)")
                if col == 0: ax.set_ylabel("raw coordinate 2\n(dataset units)")
                fig.colorbar(image, ax=ax, fraction=.045, pad=.02)
        fig.suptitle(f"{label}: {item['sample_id']} | native-to-raster, physical area; field units unspecified")
        for suffix in ("png", "pdf"):
            path = output / f"{label}.{suffix}"
            fig.savefig(path, dpi=160)
            files.append(ref(path))
        plt.close(fig)
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--joint100-assessment", type=Path, required=True)
    p.add_argument("--joint500-assessment", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--limit", type=int, help="smoke only; full receipt requires the whole saved panel")
    args = p.parse_args()
    start, cpu = time.perf_counter(), time.process_time()
    torch.set_num_threads(args.threads)
    assessments = {"JointAnchor100": json.loads(args.joint100_assessment.read_text()),
                   "JointAnchor500": json.loads(args.joint500_assessment.read_text())}
    reference = assessments["JointAnchor500"]
    summary = json.loads(args.summary.read_text())
    ids = reference["sample_ids"]
    if reference["split"] != "validation" or reference["members"] != 4 or summary["sample_ids"] != ids:
        raise ValueError("matched VALIDATION J4 summary and bank required")
    for name, value in assessments.items():
        if value["sample_ids"] != ids or value["source_native"] != reference["source_native"] or value["source_selected"] != reference["source_selected"]:
            raise ValueError(f"{name}: SOURCE or ordered panel mismatch")
    paths = {"SOURCE": {k: checked_ref(reference["source_" + k]) for k in ("native", "selected")}}
    for name, value in assessments.items():
        paths[name] = {k: checked_ref(value["child_" + k]) for k in ("native", "selected")}
    receipts = {name: {k: json.loads(path.read_text()) for k, path in group.items()} for name, group in paths.items()}
    spec_path = checked_ref(reference["spec"]); spec = json.loads(spec_path.read_text())
    panel_path = checked_ref(spec["panel"]); panel = json.loads(panel_path.read_text())
    if panel["sample_ids"] != ids:
        raise ValueError("panel differs")
    query = torch.as_tensor(panel["query_indices"], dtype=torch.long)
    dataset, families, core = build_state_scale(spec, receipts["SOURCE"]["native"])
    args.output.mkdir(parents=True, exist_ok=False)
    records, examples = [], {}
    chosen = {v["sample_id"]: k for k, v in summary["example_selection"].items() if isinstance(v, dict)}
    total = len(ids) if args.limit is None else min(args.limit, len(ids))
    native_map = sparse_map = None
    try:
        for index, sid in enumerate(ids[:total]):
            native_rows = {name: read_row(paths[name]["native"], receipts[name]["native"], index) for name in paths}
            selected_rows = {name: read_row(paths[name]["selected"], receipts[name]["selected"], index) for name in paths}
            for name in paths:
                if name != "SOURCE":
                    matched_rows(native_rows["SOURCE"], native_rows[name]); matched_rows(selected_rows["SOURCE"], selected_rows[name])
                verify_selected_reference(native_rows[name], selected_rows[name], query)
            if native_map is None:
                row = native_rows["SOURCE"]
                native_map = build_physical_cell_overlap_map(row["coordinates_raw"], torch.arange(len(row["native_masses"])), grid_shape=(32,128))
                sparse_map = build_physical_cell_overlap_map(row["coordinates_raw"], query, grid_shape=(32,128))
                if not torch.allclose(native_map.grid_coordinates, sparse_map.grid_coordinates):
                    raise ValueError("native and sparse target rasters differ")
            model = {}
            rasters = {}
            for name in paths:
                native, selected = native_rows[name], selected_rows[name]
                state = state_triplet(families["global_distribution"], core["state_bank"], selected["members_model"], selected["truth_model"], selected["native_masses"])
                saved_name = "SOURCE" if name == "SOURCE" else "candidate"
                saved_assessment = reference if name == "SOURCE" else assessments[name]
                expected = saved_assessment["per_condition_scores"][saved_name][index]["state"]
                if not math.isclose(state[2], expected, rel_tol=2e-5, abs_tol=1e-7):
                    raise ValueError(f"State A-R does not reproduce saved score: {sid} {name}")
                support = native_support_risks(native, query, seed=80112026+index)
                expected_native = saved_assessment["per_condition_fields"][saved_name][index]["member_mse"]
                if not np.allclose(support["native"], expected_native, rtol=3e-5, atol=1e-8):
                    raise ValueError(f"native member risk does not reproduce saved value: {sid} {name}")
                raster, image = raster_comparison(native, selected, query, native_map, sparse_map)
                model[name] = {"state": state, "scale": scale_context(families["multiscale_spectrum"], selected["members_model"], selected["truth_model"]),
                               "support_risks": support, "raster": raster}
                rasters[name] = image
            if sid in chosen:
                examples[chosen[sid]] = {"sample_id": sid, "reference": rasters["SOURCE"][0][0],
                                         "SOURCE": rasters["SOURCE"][1].mean(0),
                                         "JointAnchor500": rasters["JointAnchor500"][1].mean(0)}
            records.append({"sample_id": sid, "models": model})
        figures = plot_examples(examples, args.output, reference["field_order"], native_map.grid_coordinates) if len(examples) == len(chosen) else []
        status = "completed" if total == len(ids) else "smoke_only"
        output = {"schema": "car_round8_support_readout_v1", "status": status, "split": "validation", "members": 4,
                  "sample_ids": ids[:total], "field_order": reference["field_order"], "records": records,
                  "native_geometry": native_map.diagnostics, "selected_geometry": sparse_map.diagnostics,
                  "definitions": {"state": "A, R, S=A-R; exact historical weighted-CDF norm; R is half mean pair distance",
                    "scale": "rho signed and absolute member errors on fixed selected support; floor and log1p(power/floor) context; uniform pair and band weights",
                    "support": "native physical weights; selected and exact complement contributions use native total mass; sampled complement is uniform without replacement",
                    "raster": "same physical 32x128 output cells; native direct overlap versus sparse IDW followed by overlap; selected query law displayed separately",
                    "units": "dataset units and their squares; field/coordinate units are unspecified", "example_rule": summary["example_selection"]["rule"]},
                  "inputs": {"joint100_assessment": ref(args.joint100_assessment), "joint500_assessment": ref(args.joint500_assessment),
                             "summary": ref(args.summary), "spec": ref(spec_path), "panel": ref(panel_path),
                             "receipts": {name: {k: ref(v) for k,v in group.items()} for name,group in paths.items()}, "script": ref(Path(__file__))},
                  "figures": figures, "test_locked": True,
                  "timing": {"wall_seconds": time.perf_counter()-start, "cpu_seconds": time.process_time()-cpu, "threads": args.threads}}
        with (args.output / "readout.json").open("x") as stream:
            json.dump(output, stream, indent=2, allow_nan=False); stream.write("\n")
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
