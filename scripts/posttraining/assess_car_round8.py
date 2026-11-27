#!/usr/bin/env python3
"""Assess one completed Round 8 endpoint on the cached VALIDATION100 panel.

Prepare is CPU-only. Generate runs two matched J4 banks inside the existing
Round 8 GPU lease. Score runs on CPU, reusing the Round 7 SOURCE target cache.
Tasks separately scores cached native banks only after the existing sealed
checkpoint-and-bank-bound task release. TEST is intentionally unavailable.

Generation is not resumable in the same attempt after a failed or completed
generation receipt; an abrupt interruption can also leave bank directories
that prevent rerun. Preserve and diagnose that attempt. CPU scoring can reuse
fully completed scores after a later readout failure; partial scores are not
resumed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import time

CODE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(CODE / "src"), str(CODE)]
from scripts.posttraining.assess_car_round7 import (
    baseline_binding, checked, generate as generate_round7, load, ref,
    score as score_round7, tasks as tasks_round7, validate_spec, verify_bank, write,
)
from scripts.posttraining.run_car_round8 import ARMS


def prepare(args):
    """Bind a completed, exact reporting-epoch checkpoint without inference."""
    import yaml
    from phycoflow_reconstruction.training.run_store import load_project_checkpoint
    from scripts.posttraining.prepare_car_round7_assessment import prepare as prepare_round7

    campaign = args.campaign.resolve(strict=True)
    result = load(args.training_result)
    epoch, arm = result.get("target_epoch"), result.get("arm")
    if (result.get("schema") != "car_round8_training_result_v1" or arm not in ARMS
            or not isinstance(epoch, int) or not 100 <= epoch <= 500
            or result.get("global_step") != 12 * epoch
            or result.get("reporting_epoch") != epoch or result.get("target_reached") is not True
            or result.get("disposition") != "target_reached" or result.get("test_locked") is not True):
        raise ValueError("a completed exact Round 8 learning endpoint is required")
    prepared_path = campaign / "configs/conditional_learning_prepared.json"
    prepared = load(prepared_path)
    declaration = prepared["arms"][arm]
    if result.get("prepared_config") != declaration:
        raise ValueError("endpoint is not from the prepared arm")
    for item in (declaration, result["saved_config"], result["checkpoint"], prepared["reference"]):
        checked(item)
    run = Path(result["run_dir"]).resolve(strict=True)
    if (checked(result["checkpoint"]) != (run / "checkpoints/last.pt").resolve(strict=True)
            or checked(result["saved_config"]) != (run / "resolved_config.yaml").resolve(strict=True)
            or load_project_checkpoint(checked(result["checkpoint"])).get("global_step") != 12 * epoch):
        raise ValueError("terminal checkpoint, saved configuration, or endpoint clock differs")
    saved = yaml.safe_load(checked(result["saved_config"]).read_text())
    expected = yaml.safe_load(checked(declaration).read_text())
    saved["runtime"].pop("device", None)
    expected["runtime"].pop("device", None)
    if saved != expected:
        raise ValueError("saved scientific configuration differs from the prepared arm")
    attempt = args.attempt.resolve()
    if campaign / "assessment" not in attempt.parents:
        raise ValueError("attempt must be a new directory inside the campaign assessment tree")
    attempt.mkdir(parents=True, exist_ok=False)
    # A selected continuation overwrites the run's rolling last.pt. Preserve
    # this endpoint's exact bytes before binding any prediction or task bank.
    checkpoint = attempt / "endpoint_checkpoint.pt"
    with checked(result["checkpoint"]).open("rb") as source, checkpoint.open("xb") as target:
        shutil.copyfileobj(source, target)
    frozen_checkpoint = ref(checkpoint)
    if frozen_checkpoint["sha256"] != result["checkpoint"]["sha256"]:
        raise ValueError("terminal checkpoint changed while preserving the assessment endpoint")
    spec_path = attempt / "validation_spec.json"
    prepare_round7(argparse.Namespace(campaign=campaign, config=checked(result["saved_config"]),
        reference=checked(prepared["reference"]), split="validation",
        checkpoint=checkpoint, reporting_epoch=epoch, output=spec_path))
    spec = load(spec_path)
    panel, source = validate_spec(spec)
    if len(panel["sample_ids"]) != 100 or spec["split"] != "validation":
        raise ValueError("the declared matched VALIDATION100 panel is required")
    baseline_path = campaign.parent / "round_7/assessment/SOURCE_Baseline_VALIDATION100_v1/baseline.json"
    baseline = load(baseline_path)
    selected_path = checked(baseline["source_selected"])
    verify_bank(selected_path, spec, panel, source, "selected")
    if (baseline.get("status") != "completed" or baseline.get("binding") != baseline_binding(spec)
            or baseline.get("sample_ids") != panel["sample_ids"]
            or baseline.get("source_native") != spec["source_native_receipt"]
            or len(baseline.get("files_sha256", [])) != 100):
        raise ValueError("cached VALIDATION SOURCE transforms do not match this endpoint")
    plan = {"schema": "car_round8_validation_endpoint_plan_v1", "arm": arm,
        "epoch": epoch, "training_result": ref(args.training_result), "prepared": ref(prepared_path),
        "checkpoint": frozen_checkpoint, "original_training_checkpoint": result["checkpoint"],
        "spec": ref(spec_path), "source_baseline": ref(baseline_path),
        "source_selected": baseline["source_selected"], "members": 4, "split": "validation",
        "supports": ["native", "selected"], "sample_ids": panel["sample_ids"],
        "position_fit": prepared.get("position_fit"), "test_locked": True,
        "policy": "One matched VALIDATION100 assessment per declared completed endpoint; no TEST or output sweep.",
        "recovery": {"generation": "A failed or completed generation receipt blocks rerun in this attempt; abrupt interruption can also leave bank directories that block rerun. Preserve and diagnose the attempt.",
                     "cpu_scoring": "Fully completed CPU scores can be reused after a later readout failure. Partial CPU scoring is not resumed."}}
    write(attempt / "plan.json", plan)
    print(attempt / "plan.json", flush=True)


def context(attempt):
    attempt = attempt.resolve(strict=True)
    plan_path = attempt / "plan.json"
    plan = load(plan_path)
    if (plan.get("schema") != "car_round8_validation_endpoint_plan_v1"
            or plan.get("split") != "validation" or plan.get("members") != 4
            or plan.get("supports") != ["native", "selected"] or plan.get("test_locked") is not True):
        raise ValueError("VALIDATION100 endpoint plan required")
    for name in ("training_result", "prepared", "checkpoint", "spec", "source_baseline", "source_selected"):
        checked(plan[name])
    result = load(checked(plan["training_result"]))
    original = plan.get("original_training_checkpoint")
    if (original != result.get("checkpoint") or not isinstance(original, dict)
            or original.get("sha256") != plan["checkpoint"]["sha256"]):
        raise ValueError("preserved assessment checkpoint differs from its completed training endpoint")
    spec = load(checked(plan["spec"]))
    panel, source = validate_spec(spec)
    if (len(panel["sample_ids"]) != 100 or plan["sample_ids"] != panel["sample_ids"]
            or spec.get("checkpoint") != plan["checkpoint"] or spec.get("reporting_epoch") != plan["epoch"]):
        raise ValueError("endpoint checkpoint or matched panel differs from the plan")
    return plan, spec, panel, source


def generate(args):
    """Reuse the historical generator with the Round 8 lease identity."""
    token = os.environ.get("CAR_ROUND8_LEASE_TOKEN")
    if not token:
        raise ValueError("bank generation requires an enclosing Round 8 GPU lease")
    if os.environ.get("CAR_ROUND7_LEASE_TOKEN"):
        raise ValueError("nested Round 7 and Round 8 leases are not supported")
    plan, spec, panel, source = context(args.attempt)
    path = args.attempt / "generation_result.json"
    if path.exists():
        raise FileExistsError(path)
    start = time.time()
    receipt = {"schema": "car_round8_validation_generation_v1", "status": "failed", "plan": ref(args.attempt / "plan.json"),
               "lease_token": token, "start_unix": start, "banks": {}, "test_locked": True}
    # The imported legacy generator requires its old token name; the value is
    # the enclosing Round 8 lease, not a second lease or an untracked GPU job.
    os.environ["CAR_ROUND7_LEASE_TOKEN"] = token
    try:
        for support in plan["supports"]:
            output = args.attempt / (support + "_bank")
            generate_round7(argparse.Namespace(spec=checked(plan["spec"]), output=output,
                support=support, device=args.device), spec, panel, source)
            bank_path = output / "receipt.json"
            _, bank = verify_bank(bank_path, spec, panel, source, support)
            if (bank.get("checkpoint_sha256") != plan["checkpoint"]["sha256"]
                    or bank.get("spec_sha256") != plan["spec"]["sha256"]
                    or bank.get("run_config_sha256") != spec["run_config"]["sha256"]):
                raise ValueError("generated bank is not bound to this endpoint")
            receipt["banks"][support] = ref(bank_path)
        receipt["status"] = "completed"
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        os.environ.pop("CAR_ROUND7_LEASE_TOKEN", None)
        receipt.update(end_unix=time.time(), wall_seconds=time.time()-start)
        write(path, receipt)


def position_readout(assessment, plan, source_readout, *, threads):
    """Reuse cached SOURCE location features and score only this child bank."""
    import numpy as np
    import torch
    from phycoflow_reconstruction.coherence.families.topology.geometry import build_physical_cell_overlap_map, rasterize_fields
    from phycoflow_reconstruction.coherence.query_sampling import SharedQuerySampler
    from phycoflow_reconstruction.coherence.shape_compute import LazyPooledShape
    from scripts.analysis.car_round8_cached_summary import paired_interval
    from scripts.analysis.car_round8_position_readout import feature_triplet, historical_ph
    from scripts.posttraining.assess_car_round7 import read_row, verify_selected_reference

    fit_ref = plan.get("position_fit")
    if fit_ref is None:
        raise ValueError("position readout requires the common frozen TRAIN fit")
    fit = load(checked(fit_ref))
    cached = load(source_readout)
    ids = assessment["sample_ids"]
    if (cached.get("schema") != "car_round8_common_position_readout_v1"
            or cached.get("status") != "completed" or cached.get("test_locked") is not True
            or cached.get("split") != "validation" or cached.get("members") != 4
            or cached.get("sample_ids") != ids or cached.get("inputs", {}).get("fit") != fit_ref):
        raise ValueError("cached SOURCE position readout is not the matched TRAIN-fitted observer")
    original_assessment = load(checked(cached["inputs"]["joint500"]))
    if (original_assessment["source_native"] != assessment["source_native"]
            or original_assessment["source_selected"] != assessment["source_selected"]):
        raise ValueError("cached position SOURCE banks differ from the endpoint")
    spec = load(checked(assessment["spec"]))
    if cached["inputs"]["contract"] != spec["assessment_contract"]:
        raise ValueError("cached position query contract differs")
    contract = load(checked(spec["assessment_contract"]))
    panel = load(checked(spec["panel"]))
    core = torch.load(checked(fit["artifact"]), map_location="cpu", weights_only=True)
    shape = LazyPooledShape.from_state_dict(core["shape_artifact"])
    if shape.position is None or shape.position.unit != cached["location_unit"]:
        raise ValueError("cached position unit differs from the frozen fit")
    sampler = SharedQuerySampler(assessment["field_order"], seed=int(contract["shape_query_seed"]), epsilon=1.)
    batches = []
    for _ in range(8):
        batches.append(sampler.draw())
        sampler.finish()
    if json.loads(json.dumps([asdict(batch) for batch in batches])) != cached["query_batches"]:
        raise ValueError("cached position queries differ from the current assessment")
    queries = shape.compile_queries(tuple(batches))
    topology = torch.load(checked(fit["inputs"]["topology_artifact"]), map_location="cpu", weights_only=True)
    if topology["units"] != "model_units" or float(topology["config"]["filtration"]["smoothing_sigma"]) != 0.:
        raise ValueError("position readout requires the TRAIN-fitted model-unit raster")
    raster_state = topology["state_dict"]
    if not torch.allclose(raster_state["grid_coordinates"].double(), shape.position.state["coordinates"].double()):
        raise ValueError("positional and topology raster coordinates differ")
    kind_weights = {kind: float(topology["config"]["components"][kind]["weight"]) for kind in ("self", "mutual")}
    if kind_weights != cached["kind_weights"]:
        raise ValueError("cached position component weights differ")
    native_path, selected_path = (checked(assessment["child_" + support]) for support in ("native", "selected"))
    native_receipt, selected_receipt = load(native_path), load(selected_path)
    query = torch.as_tensor(panel["query_indices"], dtype=torch.long)
    torch.set_num_threads(threads)
    records, native_map = [], None
    with torch.no_grad():
        for index, sid in enumerate(ids):
            native = read_row(native_path, native_receipt, index)
            selected = read_row(selected_path, selected_receipt, index)
            verify_selected_reference(native, selected, query)
            if native_map is None:
                coordinates = native["coordinates_raw"]
                native_map = build_physical_cell_overlap_map(coordinates, torch.arange(len(coordinates)), grid_shape=shape.position.grid_shape)
                if not torch.allclose(native_map.grid_coordinates.double(), shape.position.state["coordinates"].double()):
                    raise ValueError("native target raster differs from the frozen positional raster")
            def raster(mapping, fields):
                return rasterize_fields(fields, mapping.neighbor_indices, mapping.neighbor_weights.to(fields), mapping.grid_shape)
            def sparse(fields):
                return rasterize_fields(fields, raster_state["neighbor_indices"], raster_state["neighbor_weights"].to(fields), shape.position.grid_shape)
            sparse_target, native_target = sparse(selected["truth_model"]), raster(native_map, native["truth_model"])
            sparse_features = shape.position.features(shape._filtration_bank(sparse_target, queries), queries)
            native_features = shape.position.features(shape._filtration_bank(native_target, queries), queries)
            sparse_values = feature_triplet(shape, [sparse(x) for x in selected["members_model"]], sparse_target,
                                            queries, kind_weights, target_features=sparse_features)
            native_values = feature_triplet(shape, [raster(native_map, x) for x in native["members_model"]], native_target,
                                            queries, kind_weights, target_features=native_features)
            ph = historical_ph(assessment["per_condition_scores"]["candidate"][index])
            source_ph = historical_ph(assessment["per_condition_scores"]["SOURCE"][index])
            cache_row = cached["records"][index]
            if cache_row["sample_id"] != sid or any(not np.allclose(source_ph[kind], cache_row["models"]["SOURCE"][kind]["ph_ARS"], rtol=1e-6, atol=1e-8) for kind in ("self", "mutual")):
                raise ValueError("cached SOURCE PH values differ")
            child = {}
            for kind in ("self", "mutual"):
                normalized = sparse_values[kind] / shape.position.unit
                child[kind] = {"ph_ARS": ph[kind].tolist(), "location_ARS": sparse_values[kind].tolist(),
                    "location_unit_matched_ARS": normalized.tolist(), "combined_ARS": (0.5 * (ph[kind] + normalized)).tolist(),
                    "native_location_unit_matched_ARS": (native_values[kind] / shape.position.unit).tolist()}
            records.append({"sample_id": sid, "models": {"SOURCE": cache_row["models"]["SOURCE"], "candidate": child}})
    indices = np.random.default_rng(6042026).integers(0, len(ids), size=(2000, len(ids)))
    paired = {kind: {component: paired_interval([np.asarray(row["models"]["candidate"][kind][component]) - np.asarray(row["models"]["SOURCE"][kind][component]) for row in records], indices)
        for component in ("ph_ARS", "location_unit_matched_ARS", "combined_ARS", "native_location_unit_matched_ARS")}
        for kind in ("self", "mutual")}
    return {"schema": "car_round8_endpoint_position_readout_v1", "status": "completed", "split": "validation",
        "members": 4, "sample_ids": ids, "location_unit": shape.position.unit, "kind_weights": kind_weights,
        "records": records, "paired": paired, "cached_source": ref(source_readout), "fit": fit_ref,
        "bootstrap": {"seed": 6042026, "replicates": 2000, "unit": "paired condition"},
        "definition": "Common frozen PH and half(PH + location/unit); selected and native location keep their separate raster supports.",
        "additional_inference": False, "test_locked": True}


def score(args):
    if os.environ.get("CAR_ROUND8_LEASE_TOKEN") or os.environ.get("CAR_ROUND7_LEASE_TOKEN"):
        raise ValueError("cached scoring runs outside a GPU lease")
    from scripts.analysis.car_round8_cached_summary import summarize
    plan, spec, panel, source = context(args.attempt)
    generation_path = args.attempt / "generation_result.json"
    generation = load(generation_path)
    if (generation.get("status") != "completed" or generation.get("plan") != ref(args.attempt / "plan.json")
            or set(generation.get("banks", {})) != {"native", "selected"}):
        raise ValueError("both completed matched endpoint banks are required")
    banks = {}
    for support in ("native", "selected"):
        banks[support] = checked(generation["banks"][support])
        _, bank = verify_bank(banks[support], spec, panel, source, support)
        if (bank.get("checkpoint_sha256") != plan["checkpoint"]["sha256"]
                or bank.get("spec_sha256") != plan["spec"]["sha256"]):
            raise ValueError("bank differs from the planned endpoint")
    result_path = args.attempt / "assessment_result.json"
    if result_path.exists():
        raise FileExistsError("this endpoint assessment is already complete")
    start, cpu = time.time(), time.process_time()
    output = args.attempt / "score"
    reused_score = output.exists()
    if not reused_score:
        score_round7(argparse.Namespace(spec=checked(plan["spec"]), child_native=banks["native"],
            source_selected=checked(plan["source_selected"]), child_selected=banks["selected"],
            baseline=checked(plan["source_baseline"]), threads=args.threads, output=output), spec, panel, source)
    assessment = load(output / "assessment.json")
    decomposition = load(output / "decomposition.json")
    scoring_receipt = load(output / "receipt.json")
    if (scoring_receipt.get("status") != "completed" or assessment.get("status") != "completed"
            or assessment.get("spec") != plan["spec"] or assessment.get("sample_ids") != panel["sample_ids"]
            or assessment.get("child_native") != generation["banks"]["native"]
            or assessment.get("child_selected") != generation["banks"]["selected"]
            or assessment.get("source_selected") != plan["source_selected"]
            or assessment.get("source_baseline") != plan["source_baseline"]):
        raise ValueError("saved CPU scores are incomplete or differ from this endpoint")
    summary_path = output / "member_mse_summary.json"
    summary_inputs = {"assessment": ref(output / "assessment.json"), "decomposition": ref(output / "decomposition.json")}
    if summary_path.exists():
        if load(summary_path).get("inputs") != summary_inputs:
            raise ValueError("saved member-MSE summary differs from these CPU scores")
    else:
        summary = summarize(assessment, decomposition)
        summary["inputs"] = summary_inputs
        write(summary_path, summary)
    position = None
    if args.position_source_readout:
        position_path = output / "position_readout.json"
        if position_path.exists():
            position = load(position_path)
            if (position.get("status") != "completed" or position.get("fit") != plan.get("position_fit")
                    or position.get("cached_source") != ref(args.position_source_readout)):
                raise ValueError("saved position readout differs from the declared observer")
        else:
            position = position_readout(assessment, plan, args.position_source_readout.resolve(strict=True), threads=args.threads)
            write(position_path, position)
    relations = None
    relation_source = getattr(args, "relation_source_readout", None)
    direct_companion = getattr(args, "direct_source_companion", None)
    if direct_companion and not relation_source:
        raise ValueError("direct SOURCE companion requires the cached relation SOURCE readout")
    if relation_source:
        relation_path = output / "relation_context.json"
        if relation_path.exists():
            relations = load(relation_path)
            expected_inputs = {"assessment": ref(output / "assessment.json"),
                               "decomposition": ref(output / "decomposition.json"),
                               "source_readout": ref(relation_source)}
            if direct_companion:
                expected_inputs["source_companion"] = ref(direct_companion)
            elif relations.get("inputs", {}).get("source_companion"):
                raise ValueError("saved direct SOURCE companion must be supplied explicitly on reuse")
            if (relations.get("status") != "completed" or relations.get("endpoint_label") != plan["arm"]
                    or any(relations.get("inputs", {}).get(k) != v for k, v in expected_inputs.items())):
                raise ValueError("saved relation readout differs from this endpoint or cached SOURCE")
        else:
            from scripts.analysis.car_round8_child_relations import build_readout
            direct_options = {"source_companion_path": direct_companion} if direct_companion else {}
            relations = build_readout(output / "assessment.json", relation_source, label=plan["arm"],
                                      threads=args.threads, **direct_options)
            write(relation_path, relations)
    result = {"schema": "car_round8_validation_assessment_v1", "status": "completed", "arm": plan["arm"],
        "epoch": plan["epoch"], "plan": ref(args.attempt / "plan.json"), "generation": ref(generation_path),
        "assessment": ref(output / "assessment.json"), "member_mse_summary": ref(output / "member_mse_summary.json"),
        "scale_summary": ref(output / "scale_summary.json"), "position_readout": ref(output / "position_readout.json") if position else None,
        "relation_context": ref(output / "relation_context.json") if relations else None,
        "split": "validation", "sample_ids": panel["sample_ids"], "members": 4, "test_locked": True,
        "wall_seconds": time.time()-start, "cpu_seconds": time.process_time()-cpu, "threads": args.threads, "reused_completed_cpu_scores": reused_score,
        "interpretation": "Split-specific 10% member-MSE margins are in member_mse_summary; historical signed-risk fields keep their original allowance."}
    write(result_path, result)
    print(result_path, flush=True)


def tasks(args):
    """Score released endpoint tasks from cached native banks outside PI."""
    if os.environ.get("CAR_ROUND8_LEASE_TOKEN") or os.environ.get("CAR_ROUND7_LEASE_TOKEN"):
        raise ValueError("cached task scoring runs outside a GPU lease")
    plan, spec, panel, source = context(args.attempt)
    generation_path = args.attempt / "generation_result.json"
    generation = load(generation_path)
    if (generation.get("status") != "completed" or generation.get("plan") != ref(args.attempt / "plan.json")
            or set(generation.get("banks", {})) != {"native", "selected"}):
        raise ValueError("completed matched endpoint banks are required")
    native = checked(generation["banks"]["native"])
    _, bank = verify_bank(native, spec, panel, source, "native")
    if (bank.get("checkpoint_sha256") != plan["checkpoint"]["sha256"]
            or bank.get("spec_sha256") != plan["spec"]["sha256"]):
        raise ValueError("native bank differs from the planned endpoint")
    result_path = args.attempt / "task_result.json"
    if result_path.exists():
        raise FileExistsError("this endpoint task assessment is already complete")
    start, cpu = time.time(), time.process_time()
    output = args.attempt / "tasks"
    tasks_round7(argparse.Namespace(spec=checked(plan["spec"]), child_native=native,
        decision=args.decision, threads=args.threads, output=output), spec, panel, source)
    task = load(output / "tasks.json")
    receipt = load(output / "receipt.json")
    if (task.get("status") != "completed" or receipt.get("status") != "completed"
            or task.get("split") != "validation" or task.get("members") != 4
            or task.get("sample_ids") != panel["sample_ids"] or task.get("test_locked") is not True
            or task.get("spec") != plan["spec"] or task.get("decision") != ref(args.decision)
            or task.get("child_native") != generation["banks"]["native"]
            or task.get("source_native") != spec["source_native_receipt"]):
        raise ValueError("cached task results differ from the released endpoint")
    result = {"schema": "car_round8_validation_task_assessment_v1", "status": "completed",
        "arm": plan["arm"], "epoch": plan["epoch"], "plan": ref(args.attempt / "plan.json"),
        "generation": ref(generation_path), "checkpoint": plan["checkpoint"],
        "release": ref(args.decision), "native_bank": generation["banks"]["native"],
        "tasks": ref(output / "tasks.json"), "task_receipt": ref(output / "receipt.json"),
        "split": "validation", "sample_ids": panel["sample_ids"], "members": 4,
        "test_locked": True, "additional_inference": False,
        "scope": "Released cached endpoint tasks only; outside training PI and continuation nomination.",
        "wall_seconds": time.time()-start, "cpu_seconds": time.process_time()-cpu, "threads": args.threads}
    write(result_path, result)
    print(result_path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--campaign", type=Path, required=True)
    prep.add_argument("--training-result", type=Path, required=True)
    prep.add_argument("--attempt", type=Path, required=True)
    bank = sub.add_parser("generate")
    bank.add_argument("--attempt", type=Path, required=True)
    bank.add_argument("--device", required=True)
    scoring = sub.add_parser("score")
    scoring.add_argument("--attempt", type=Path, required=True)
    scoring.add_argument("--threads", type=int, default=2)
    scoring.add_argument("--position-source-readout", type=Path)
    scoring.add_argument("--relation-source-readout", type=Path, help="completed cached SOURCE State/rho/power readout")
    scoring.add_argument("--direct-source-companion", type=Path,
                         help="hash-bound cached SOURCE direct State groups and Scale piece means")
    task = sub.add_parser("tasks")
    task.add_argument("--attempt", type=Path, required=True)
    task.add_argument("--decision", type=Path, required=True, help="sealed checkpoint- and native-bank-bound task release")
    task.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    {"prepare": prepare, "generate": generate, "score": score, "tasks": tasks}[args.command](args)


if __name__ == "__main__":
    main()
