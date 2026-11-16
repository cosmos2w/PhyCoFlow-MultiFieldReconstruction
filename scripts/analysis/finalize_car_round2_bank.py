#!/usr/bin/env python3
"""Finalize the TRAIN-only bank from charged real optimizer displacement caches."""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

CODE_ROOT = Path(os.environ.get("CAR_CODE_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(CODE_ROOT / "src"))


def all_candidate_bank(bank):
    from phycoflow_reconstruction.coherence.adaptive_fitting import _geometry
    rows = [item["candidate"] for item in bank["selection_audit"]]
    groups = [r for r in rows if r["kind"] == "state.group_projection"]
    triples = [r for r in rows if r["kind"] == "scale.third_order"]
    result = dict(bank)
    result["directions"] = [_geometry(r, bank["field_order"], bank["field_scales"]) for r in groups]
    result["triples"] = [_geometry(r, bank["field_order"], bank["field_scales"]) for r in triples]
    return result, groups + triples


def validate_probe_summary(summary, *, bank_sha256, panel_sha256, core_sha256, sample_ids):
    """Bind the actual completed prefit producer's Scale trial schema."""
    inputs = summary.get("inputs", {})
    if (summary.get("version") != "common_prefit_calibration_v1"
            or inputs.get("bank_sha256") != bank_sha256
            or inputs.get("source_panel_sha256") != panel_sha256
            or inputs.get("scalable_artifact_sha256") != core_sha256):
        raise ValueError("probe summary does not bind provisional bank and SOURCE panel/core")
    probes = summary.get("scale_probe", {})
    decision = probes.get("decision", {})
    chosen = decision.get("chosen_ratio")
    if (probes.get("schema") != "car_round2_scale_probe_v1"
            or probes.get("reserved_task_used") is not False or chosen not in (1, 2, 4)
            or summary.get("scale_joint_ratio") != chosen
            or not isinstance(probes.get("source_scale"), (int, float))
            or not math.isfinite(probes["source_scale"]) or probes["source_scale"] < 0):
        raise ValueError("completed reserved-task-free Scale-ratio probe required")
    rows = [row for row in probes.get("rows", []) if row.get("ratio") == chosen]
    seeds = {row.get("seed") for row in rows}
    if len(seeds) != 3 or None in seeds:
        raise ValueError("exactly three matched Scale probe seeds required")
    for kind in ("full_model", "residual_head"):
        for seed in seeds:
            selected = [row for row in rows if row.get("parameterization") == kind and row.get("seed") == seed]
            if [sample for row in selected for sample in row.get("sample_ids", [])] != sample_ids:
                raise ValueError("each actual parameterization/seed must cover the frozen panel in order")
            if any(row.get("sample_count") != len(row["sample_ids"])
                   or not isinstance(row.get("field_and_sensor_feasible"), bool) for row in selected):
                raise ValueError("Scale trial rows require sample counts and measured feasibility")
    if {row.get("parameterization") for row in rows} != {"full_model", "residual_head"}:
        raise ValueError("unexpected Scale probe parameterization")
    return probes, chosen, rows, seeds


def null_sensor_perturbation(shard, scales, *, fraction=.01):
    """Offline bounded truth-directed response probe with exact sensor nulls."""
    import torch
    source, truth = shard["source"], shard["reference"]
    bounds = torch.as_tensor(scales, dtype=source.dtype)[None, None, None, :] * fraction
    delta = (truth[None] - source).clamp(-bounds, bounds)
    for batch in range(source.shape[1]):
        for obs in torch.where(shard["obs_valid_mask"][batch])[0].tolist():
            point = shard["obs_indices"][batch, obs]
            field = int(shard["obs_fields"][batch, obs])
            mask = shard["query_indices"][batch] == point
            delta[:, batch, mask, field] = 0
    return source + delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-run", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--probe-summary", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=int, default=1800)
    args = parser.parse_args()
    start = time.time()
    if args.max_wall_seconds <= 0:
        raise ValueError("positive CPU time bound required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "car_round2_bank_finalize_job_v1", "start_unix": start,
               "gpu_lease": None, "status": "running"}
    def timeout(_signal, _frame):
        raise TimeoutError("bank finalization CPU time bound exceeded")
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.max_wall_seconds)
    try:
        import torch
        import yaml

        from phycoflow_reconstruction.coherence import build_enabled_families
        from phycoflow_reconstruction.coherence.adaptive_fitting import (
            score_bank,
            spectral_filter,
            validate_bank,
        )
        from phycoflow_reconstruction.coherence.round2_qualification import (
            select_task_conditioned_bank,
        )
        from phycoflow_reconstruction.data.factory import open_field_dataset
        from phycoflow_reconstruction.training.run_store import file_sha256
        from phycoflow_reconstruction.training.scalable_fitting import (
            _adaptive_input_binding,
            _identity,
        )

        run = args.prepared_run.resolve()
        config = yaml.safe_load((run / "resolved_config.yaml").read_text())
        dataset = open_field_dataset(config["dataset"], split="train")
        try:
            core = torch.load(run / "artifacts/scalable_coherence.pt", map_location="cpu", weights_only=True)
            bank = json.loads(args.bank.read_text())
            validate_bank(bank, field_order=core["provenance"]["field_order"],
                          support_identity=core["provenance"]["support_sha256"],
                          source_identity=core["provenance"]["source_checkpoint_sha256"])
            if bank["population_ids"] != core["provenance"]["sample_ids"]:
                raise ValueError("bank and frozen TRAIN population differ")
            panel = torch.load(run / "artifacts/source_panel.pt", map_location="cpu", weights_only=True)
            if panel["provenance"] != core["provenance"]:
                raise ValueError("SOURCE panel and core provenance differ")
            weights = panel["masses"].to(panel["reference"])
            binding = _adaptive_input_binding(source=panel["members"], reference=panel["reference"],
                                              coordinates=panel["coordinates"], source_weights=weights,
                                              reference_weights=weights, identity=_identity(panel["provenance"]), core=core)
            if bank.get("source_fit_binding_sha256") != binding:
                raise ValueError("bank does not bind the frozen SOURCE panel and core")
            summary = json.loads(args.probe_summary.read_text())
            job_receipt_path = args.probe_summary.with_suffix(args.probe_summary.suffix + ".receipt.json")
            job_receipt = json.loads(job_receipt_path.read_text())
            if job_receipt.get("schema") != "car_round2_prefit_job_v1" or job_receipt.get("status") != "completed":
                raise ValueError("terminal completed prefit job receipt required")
            probes, chosen, rows, seed_ids = validate_probe_summary(
                summary, bank_sha256=file_sha256(args.bank),
                panel_sha256=file_sha256(run / "artifacts/source_panel.pt"),
                core_sha256=file_sha256(run / "artifacts/scalable_coherence.pt"),
                sample_ids=bank["population_ids"])
            manifest = json.loads((run / "run_manifest.json").read_text())
            families = build_enabled_families(config["coherence"], dataset.data_spec, dataset.normalizer)
            scale = families["multiscale_spectrum"]
            scale_path = run / "artifacts/multiscale_spectrum_family.pt"
            if file_sha256(scale_path) != manifest["coherence_family_state_sha256s"]["multiscale_spectrum"]:
                raise ValueError("Scale family hash changed")
            scale.load_state_artifact(torch.load(scale_path, map_location="cpu", weights_only=True))
            all_bank, candidates = all_candidate_bank(bank)
            records = {r["id"]: {"source": 0., "perturbed_response": 0., "full_model_response": 0.,
                                "residual_head_response": 0., "count": 0., "replay_error": 0.}
                       for r in candidates}
            development = {"full_model": 0., "residual_head": 0.}
            count_by_kind = {"full_model": 0, "residual_head": 0}
            seen = {(kind, seed): [] for kind in ("full_model", "residual_head") for seed in seed_ids}
            panel_rows = {sample_id: i for i, sample_id in enumerate(panel["sample_ids"])}
            evaluation_seconds = 0.
            update_seconds = 0.
            sensor_matches, query_field_entries = 0, 0
            for row in rows:
                path = args.cache_dir / row["cache"]
                if file_sha256(path) != row["cache_sha256"]:
                    raise ValueError("probe endpoint cache checksum mismatch")
                shard = torch.load(path, map_location="cpu", weights_only=True)
                if (shard.get("schema") != "car_round2_virtual_endpoints_v1" or
                    shard["sample_ids"] != row["sample_ids"] or shard["method"] != row["parameterization"] or
                    shard["ratio"] != chosen or shard["seed"] != row["seed"]):
                    raise ValueError("probe endpoint cache identity mismatch")
                index = torch.tensor([panel_rows[sample_id] for sample_id in shard["sample_ids"]])
                for key in ("reference", "masses", "query_indices"):
                    if not torch.equal(shard[key], panel[key].index_select(0, index)):
                        raise ValueError(f"probe cache differs from frozen panel {key}")
                seen[(row["parameterization"], row["seed"])].extend(shard["sample_ids"])
                n = len(shard["sample_ids"])
                query_field_entries += int(shard["query_indices"].numel() * len(bank["field_order"]))
                for batch_index in range(n):
                    for obs in torch.where(shard["obs_valid_mask"][batch_index])[0].tolist():
                        sensor_matches += int((shard["query_indices"][batch_index] == shard["obs_indices"][batch_index, obs]).sum())
                kind = row["parameterization"]
                count_by_kind[kind] += n
                development[kind] += n * (row["scale_before"] - row["scale_after"])
                began = time.perf_counter()
                def scores(values, shard=shard):
                    result = score_bank(all_bank, values, shard["reference"], shard["masses"],
                                        lambda x: spectral_filter(scale, x))
                    return torch.cat((result["state.group_projection"], result["scale.third_order"]), dim=1)
                with torch.no_grad():
                    baseline = scores(shard["source"])
                    replay = scores(shard["source"])
                    child = scores(shard["child"])
                    perturb = scores(null_sensor_perturbation(shard, bank["field_scales"]))
                evaluation_seconds += time.perf_counter() - began
                update_seconds += float(row["wall_seconds"])
                for index, candidate in enumerate(candidates):
                    item = records[candidate["id"]]
                    item["source"] += n * float(baseline[:, index].abs().mean())
                    item["perturbed_response"] += n * float((baseline[:, index] - perturb[:, index]).mean())
                    if row["field_and_sensor_feasible"]:
                        item[kind + "_response"] += n * float((baseline[:, index] - child[:, index]).mean())
                    item["count"] += n
                    item["replay_error"] = max(item["replay_error"], float((baseline[:, index] - replay[:, index]).abs().max()))
            for key, ids in seen.items():
                if ids != bank["population_ids"]:
                    raise ValueError(f"probe cache panel incomplete or reordered: {key}")
            if any(count_by_kind[k] <= 0 for k in count_by_kind):
                raise ValueError("both actual learner responses required")
            total_conditions = sum(count_by_kind.values())
            repeatability_threshold = 1 / math.sqrt(len(bank["population_ids"]))
            candidate_by_id = {candidate["id"]: candidate for candidate in candidates}
            replay_threshold = max(item["replay_error"] / float(bank["block_units"][candidate_by_id[ident]["kind"]])
                                   for ident, item in records.items())
            response_threshold = max(1e-12, 5 * replay_threshold)
            source_scale = float(probes["source_scale"])
            development_response = (min(development[k] / count_by_kind[k] for k in development) / source_scale
                                    if source_scale > 0 else 0.)
            complete_seconds = max(1e-12, (update_seconds + evaluation_seconds) / len(rows))
            measurements = {}
            for candidate in candidates:
                ident = candidate["id"]
                entry = records[ident]
                unit = float(bank["block_units"][candidate["kind"]])
                perturb_response = entry["perturbed_response"] / total_conditions / unit
                measurements[ident] = {
                    "discrepancy": max(0., entry["source"] / total_conditions / unit),
                    "replay_error": entry["replay_error"] / unit,
                    "repeatability": float(bank["measurements"][ident]["repeatability"]),
                    "development_response": min(max(0., development_response), max(0., perturb_response)),
                    "full_model_response": max(0., entry["full_model_response"] / count_by_kind["full_model"] / unit),
                    "residual_head_response": max(0., entry["residual_head_response"] / count_by_kind["residual_head"] / unit),
                    "complete_seconds": complete_seconds, "saved_update_fraction": 0.,
                    "sensor_null_output_perturbation_response": perturb_response,
                }
            final = select_task_conditioned_bank(bank, measurements, repeatability_threshold=repeatability_threshold,
                                                 response_threshold=response_threshold)
            # Selection uses dimensionless scores; runtime units remain in the
            # raw candidate-score units expected by internal calibration.
            for kind in ("state.group_projection", "scale.third_order"):
                selected = [r for r in final["selected"] if r["kind"] == kind]
                raw = [max(0., records[r["id"]]["source"] / total_conditions) for r in selected]
                final["block_units"][kind] = max(final["floor"], sum(raw) / len(raw)) if raw else 1.
            from phycoflow_reconstruction.coherence.adaptive_fitting import _digest
            final["sha256"] = _digest({k: v for k, v in final.items() if k != "sha256"})
            validate_bank(final, field_order=core["provenance"]["field_order"],
                          support_identity=core["provenance"]["support_sha256"],
                          source_identity=core["provenance"]["source_checkpoint_sha256"])
            changed = any(final[key] != bank[key] for key in ("directions", "triples", "target_weights", "block_units"))
            (args.output_dir / "final_bank.json").write_text(json.dumps(final, indent=2, allow_nan=False) + "\n")
            (args.output_dir / "decision.json").write_text(json.dumps({
                "schema": "car_round2_bank_decision_v1", "bank_changed": changed,
                "common_prefit_refit_required": changed, "final_vector_probe_required": changed,
                "selected_count": len(final["selected"]), "candidate_count": len(candidates),
                "input_bank_sha256": file_sha256(args.bank), "probe_summary_sha256": file_sha256(args.probe_summary),
                "prefit_job_receipt_sha256": file_sha256(job_receipt_path),
                "repeatability_threshold": repeatability_threshold, "response_threshold": response_threshold,
                "bounded_perturbation": "1 percent TRAIN field scales toward truth, zeroed at observed entries; offline only",
                "starter_bank": ("newly refit Senseiver TRAIN starter" if config["model"]["name"] == "senseiver"
                                 else "source-specific FFM TRAIN starter"),
                "response_units": "candidate scores divided by their frozen source-specific block units; common Scale response divided by SOURCE common Scale",
                "perturbation_scope": "offline output-space sensitivity only; learnability uses actual optimizer displacement",
                "sampled_support_sensor_field_entries": sensor_matches,
                "sampled_support_total_field_entries": query_field_entries,
                "sampled_support_sensor_fraction": sensor_matches / query_field_entries,
                "cost_scope": "complete measured virtual-update wall time plus charged candidate scoring",
                "fidelity_scope": "selected-support proxy; native-support qualification remains separate",
                "reserved_task_used": False}, indent=2, allow_nan=False) + "\n")
            receipt.update(status="completed", candidate_count=len(candidates), bank_changed=changed)
        finally:
            dataset.close()
    except BaseException as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        ended = time.time()
        receipt.update(end_unix=ended, wall_seconds=ended - start)
        (args.output_dir / "job_receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
