from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pytest

from scripts.analysis.car_round8_learning_summary import (
    BASE, SPATIAL, SPEC_KEYS, assemble, check_matched, component_comparison, detail_comparison, diagnostic_table, make_section, member_diagnostics, nomination_check,
    pair_interval, released_tasks, sensor_evidence, sensor_table, training_evidence,
)
from scripts.analysis.car_round8_cached_summary import ref


def identity(name):
    return {"path": "/unused/" + name, "sha256": name}


def synthetic_endpoint(arm):
    ids = [f"frame-{i:03}" for i in range(100)]
    fields = ["CO", "T"]
    chosen = {"rule": "saved SOURCE choice", "representative_median": {"sample_id": ids[49]},
              "difficult_maximum": {"sample_id": ids[99]}}
    assessment = {"sample_ids": ids, "members": 4, "field_order": fields, "field_units": ["unknown", "unknown"],
        "physical_mask": "physical weights", "source_native": identity("source_native"),
        "source_selected": identity("source_selected"), "source_baseline": identity("baseline"),
        "per_condition_fields": {"SOURCE": [], "candidate": []},
        "per_condition_scores": {"SOURCE": [], "candidate": []}}
    position = {"fit": identity("position_fit"), "cached_source": identity("position_source"),
                "kind_weights": {"self": 1., "mutual": 1.}, "location_unit": .5, "records": []}
    relation = {"inputs": {"source_readout": identity("relations_source"), "source_companion": identity("detail_source")}, "field_pair_indices": [[0, 1]],
                "state_frozen_train_unit": .5, "state_group_weights": {"marginal": .4, "mixed": .6},
                "piece_geometry": {"piece_indices": [0], "mass_fractions": [1.], "unit_space": "model_units", "field_order": fields}, "records": []}
    decomposition = {"prefit_coefficients": {"scale." + k: 1. for k in ("signed", "joint", "log_power_self", "rho", "third_order")}, "records": []}
    for i, sid in enumerate(ids):
        source_risk = np.array([1.+i/100., 2.+i/50.])
        factor = 1.+(.1*i/99. if arm == BASE else .2-.05*i/99.)
        risk = {"SOURCE": source_risk, "candidate": source_risk*factor}
        states = {"SOURCE": (1.+i/100., .2),
                  "candidate": (1.+i/100.+(.1 if arm == BASE else .2), .25)}
        pos_row, relation_row, decomp_row = {}, {}, {}
        for side in ("SOURCE", "candidate"):
            a, r = states[side]; triplet = np.array([a, r, a-r])
            assessment["per_condition_fields"][side].append({"member_mse": risk[side].tolist()})
            score = {"state": float(triplet[2]), "shape": float(triplet[2])}
            for kind in ("self", "mutual"):
                for leaf in ("finite", "essential"):
                    weight = .4 if leaf == "finite" else .1
                    score[f"shape.{kind}.{leaf}.attraction"] = a*weight
                    score[f"shape.{kind}.{leaf}.separation"] = r*weight
            assessment["per_condition_scores"][side].append(score)
            item = {}
            for kind in ("self", "mutual"):
                ph = triplet/2; loc = (triplet*2)/2
                item[kind] = {"ph_ARS": ph.tolist(), "location_unit_matched_ARS": loc.tolist(),
                              "combined_ARS": (.5*(ph+loc)).tolist(),
                              "native_location_unit_matched_ARS": loc.tolist()}
            pos_row[side] = item
            scale = {"rho_signed_error_mean_member": [[a]]*3, "rho_absolute_error_mean_member": [[abs(a)]]*3,
                     "log1p_power_over_floor_signed_error_mean_member": [[a, r]]*3,
                     "log1p_power_over_floor_absolute_error_mean_member": [[abs(a), abs(r)]]*3,
                     "member_power_over_floor_mean": [[1., 2.]]*3, "member_low_energy_fraction": [[0., 0.]]*3,
                     "reference_power_over_floor": [[1., 2.]]*3, "reference_low_energy": [[0., 0.]]*3}
            signed = [0.1 + a*.01, -0.1 + a*.01]
            relation_row[side] = {"state": triplet.tolist(), "scale": scale,
                "state_groups": {"marginal": (triplet*.5).tolist(), "mixed": (triplet*.3).tolist()},
                "piece_mean": {"signed_error_mean_member": [signed],
                               "absolute_error_mean_member": [[abs(value)+.1 for value in signed]]}}
            block = lambda weight: {"attraction": a*weight, "separation": r*weight, "score": (a-r)*weight}
            decomp_row[side] = {"blocks": {"signed": block(.4), "joint": block(.8), "log_power_self": block(.1),
                                           "rho": block(.2), "third_order": block(.1)},
                                "definitions": {"Complete_Expanded_Prefit": a-r}}
        position["records"].append({"sample_id": sid, "models": pos_row})
        relation["records"].append({"sample_id": sid, "models": relation_row})
        decomposition["records"].append({"sample_id": sid, "member_seeds": [1, 2, 3, 4], "models": decomp_row})
    source = np.array([r["member_mse"] for r in assessment["per_condition_fields"]["SOURCE"]])
    candidate = np.array([r["member_mse"] for r in assessment["per_condition_fields"]["candidate"]])
    rows = []
    for j, field in enumerate(fields):
        def risk_parts(values):
            member = float(values[:, j].mean()); variance = member * .2
            mean = member - .75 * variance
            return {"member_mse": member, "ensemble_mean_mse": mean,
                    "signed_mean_risk": mean - .25 * variance, "unbiased_member_variance": variance}
        rows.append({"field": field, "source": risk_parts(source), "candidate": risk_parts(candidate),
                     "member_ratio": float(candidate[:, j].mean()/source[:, j].mean()),
                     "member_ratio_interval_95": [.9, 1.2], "member_margin": float((candidate[:, j]-1.1*source[:, j]).mean()),
                     "member_margin_interval_95": [-1., 1.], "member_point_pass": True, "old_signed_point_pass": False})
    return {"arm": arm, "epoch": 100, "assessment": assessment,
            "spec": {key: identity(key) for key in SPEC_KEYS}, "member": {"fields": rows, "example_selection": chosen},
            "position": position, "relation": relation, "decomposition": decomposition}


def test_matched_frame_contrast_uses_pooled_ratio_and_direct_state():
    base, spatial = synthetic_endpoint(BASE), synthetic_endpoint(SPATIAL)
    result = assemble(base, spatial)
    assert result["bootstrap"]["seed"] == 6042026
    source = np.array([x["member_mse"][0] for x in base["assessment"]["per_condition_fields"]["SOURCE"]])
    b = np.array([x["member_mse"][0] for x in base["assessment"]["per_condition_fields"]["candidate"]])
    s = np.array([x["member_mse"][0] for x in spatial["assessment"]["per_condition_fields"]["candidate"]])
    assert result["fields"][0][BASE]["member_ratio"] == pytest.approx(b.mean()/source.mean())
    assert result["fields"][0][BASE]["member_ratio"] != pytest.approx(np.mean(b/source), abs=1e-4)
    assert result["fields"][0]["SpatialShape_minus_BaseShape_ratio_to_SOURCE"]["mean"] == pytest.approx((s.mean()-b.mean())/source.mean())
    assert result["families"]["State"]["SpatialShape_minus_BaseShape"]["mean"] == pytest.approx([.1, 0., .1])
    assert result["families"]["Complete expanded Scale"]["SOURCE_mean_ARS"] == pytest.approx(result["families"]["State"]["SOURCE_mean_ARS"])
    assert result["families"]["PH Shape"]["SpatialShape_minus_BaseShape"]["mean"] == pytest.approx([.1, 0., .1])
    joint = result["components"]["Scale blocks"]["joint"]
    assert joint["fixed_contribution_factor"] == pytest.approx(.25)
    assert joint["paired_contrast"]["mean"] == pytest.approx([.08, 0., .08])
    assert joint["weighted_paired_contrast"]["mean"] == pytest.approx([.02, 0., .02])
    assert result["components"]["PH Shape components"]["self.essential"]["paired_contrast"]["mean"] == pytest.approx([.01, 0., .01])
    details = result["relation_details"]
    assert details["state_groups"]["marginal"]["paired_contrast"]["mean"] == pytest.approx([.05, 0., .05])
    assert details["state_groups"]["mixed"]["paired_contrast"]["mean"] == pytest.approx([.03, 0., .03])
    np.testing.assert_allclose(details["piece_mean"]["signed_error_mean_member"]["paired_contrast"]["mean"], [[.001, .001]])
    np.testing.assert_allclose(details["piece_mean"]["signed_error_mean_member"]["paired_contrast"]["interval_95"], [[[.001, .001]], [[.001, .001]]])
    np.testing.assert_allclose(result["scale_context"]["rho_signed_error_mean_member"]["SpatialShape_minus_BaseShape"]["mean"], np.full((3, 1), .1))


def test_mismatch_rejected_before_attribution():
    base, spatial = synthetic_endpoint(BASE), synthetic_endpoint(SPATIAL)
    spatial["epoch"] = 500
    with pytest.raises(ValueError, match="epoch 100"):
        check_matched(base, spatial)
    spatial = synthetic_endpoint(SPATIAL); spatial["spec"]["core"] = identity("different_core")
    with pytest.raises(ValueError, match="observer"):
        check_matched(base, spatial)
    spatial = synthetic_endpoint(SPATIAL); spatial["position"]["records"][3]["models"]["SOURCE"]["self"]["ph_ARS"][0] += 1
    with pytest.raises(ValueError, match="SOURCE per-condition"):
        check_matched(base, spatial)
    spatial = synthetic_endpoint(SPATIAL); spatial["relation"]["inputs"]["source_companion"] = identity("different_source_companion")
    with pytest.raises(ValueError, match="observer"):
        check_matched(base, spatial)


def test_piece_member_absolute_error_is_not_absolute_of_ensemble_mean():
    base, spatial = synthetic_endpoint(BASE), synthetic_endpoint(SPATIAL)
    row = spatial["relation"]["records"][0]["models"]["candidate"]["piece_mean"]
    row["absolute_error_mean_member"][0][0] = 0.
    with pytest.raises(ValueError, match="absolute piece-mean error"):
        assemble(base, spatial)


def test_paired_interval_uses_condition_differences():
    source = np.array([0., 100., 200.]); candidate = source + 3.
    indices = np.random.default_rng(6042026).integers(0, 3, size=(100, 3))
    result = pair_interval(candidate, source, indices)
    assert result["mean"] == 3. and result["interval_95"] == [3., 3.]


def test_saved_j4_diagnostics_are_retained_and_rendered_for_matched_and_selected():
    base, spatial = synthetic_endpoint(BASE), synthetic_endpoint(SPATIAL)
    result = assemble(base, spatial)
    field = result["fields"][0]
    assert field["SOURCE_diagnostics"] == base["member"]["fields"][0]["source"]
    assert field[BASE]["diagnostics"] == base["member"]["fields"][0]["candidate"]
    assert field[SPATIAL]["diagnostics"] == spatial["member"]["fields"][0]["candidate"]
    for values in (field["SOURCE_diagnostics"], field[BASE]["diagnostics"], field[SPATIAL]["diagnostics"]):
        assert values["member_mse"] == pytest.approx(values["ensemble_mean_mse"] + .75 * values["unbiased_member_variance"])
        assert values["signed_mean_risk"] == pytest.approx(values["ensemble_mean_mse"] - .25 * values["unbiased_member_variance"])
    table = diagnostic_table([{"field": "CO", "SOURCE": field["SOURCE_diagnostics"],
                               BASE: field[BASE]["diagnostics"], SPATIAL: field[SPATIAL]["diagnostics"]}],
                             ("SOURCE", BASE, SPATIAL))
    assert all(label in table[0] for label in ("SOURCE", BASE, SPATIAL))
    assert "1.2708 / 1.196 / 0.299" in table[2]
    selected = diagnostic_table([{"field": "CO", "SOURCE": field["SOURCE_diagnostics"],
                                  BASE: field[BASE]["diagnostics"]}], ("SOURCE", BASE))
    assert BASE in selected[0] and SPATIAL not in selected[0]
    broken = deepcopy(base["member"]["fields"][0]); broken["candidate"]["signed_mean_risk"] += .1
    with pytest.raises(ValueError, match="J4 member/mean/spread identity"):
        member_diagnostics(broken)


def test_section_contains_matched_and_selected_j4_diagnostic_tables():
    base, spatial = synthetic_endpoint(BASE), synthetic_endpoint(SPATIAL)
    data = assemble(base, spatial)
    cost = {"generation_lease_occupied_gpu_seconds": 1., "cpu_score_wall_seconds": 2.,
            "cpu_score_process_seconds": 3., "cpu_score_reused_completed_scores": False}
    data["assessment_costs"] = {BASE: cost, SPATIAL: cost}
    data["selected_final_assessment_cost"] = cost
    data["training_coupling_mismatch_counts"] = {}
    training_row = {"reporting_epoch": 100, "accepted_objective_condition_visits": 76800,
        "dataset_pass_equivalents": 9.6, "mean_complete_epoch_wall_seconds": 1.,
        "lease_segment_occupied_gpu_seconds": 100., "timing_covered_epochs": 100,
        "expected_exposure_deviations": [], "whole_worker_torch_memory": {"scope": "fixture"},
        "mature_epochs_51_100": {"complete_epoch_records": 50, "mean_wall_seconds": 1.,
                                  "mean_train_span_seconds": .9, "mean_callback_and_reporting_seconds": .1}}
    training = {BASE: training_row, SPATIAL: training_row}
    data["selected_final_training"] = {**training_row, "lineage_occupied_gpu_seconds": 200.}
    families = {name: {"SOURCE_mean_ARS": value["SOURCE_mean_ARS"],
                       "candidate_mean_ARS": value[BASE + "_mean_ARS"],
                       "candidate_minus_SOURCE": value[BASE + "_minus_SOURCE"]}
                for name, value in data["families"].items()}
    indices = np.random.default_rng(6042026).integers(0, 100, size=(2000, 100))
    final = {"arm": BASE, "epoch": 200, "fields": base["member"]["fields"], "families": families,
             "scale_prefit_coefficients": data["scale_prefit_coefficients"],
             "components": component_comparison({BASE: base}, indices, data["prefit_weights"]),
             "relation_details": detail_comparison({BASE: base}, indices),
             "sensor_compatibility": {"available": False}}
    section = make_section(data, training, [], final)
    assert "| Field | SOURCE | " + BASE + " | " + SPATIAL + " |" in section
    assert "| Field | SOURCE | " + BASE + " |" in section
    assert section.count("1.2708 / 1.196 / 0.299") == 2
    assert "mean-output MSE + 3/4 variance" in section
    assert "signed mean risk = mean-output MSE − variance/4" in section
    assert section.count("| joint | 0.25 |") == 2
    assert "joint factor additionally includes the fixed 0.25 multiplier" in section
    assert section.count("| self.essential |") == 2
    assert "SpatialShape − BaseShape weighted S [95% CI]" in section
    assert "Selected − SOURCE weighted S [95% CI]" in section
    assert section.count("State marginal and mixed groups") == 2
    assert section.count("not additive components") == 2
    assert section.count("model-normalized units") == 2
    assert section.count("| 0 | CO |") == 4
    assert "SpatialShape − BaseShape absolute [95% CI]" in section
    assert "Selected − SOURCE absolute [95% CI]" in section


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value)); return ref(path)


def test_tasks_require_post_nomination_release_and_bank_binding(tmp_path):
    ids = [f"frame-{i:03}" for i in range(100)]
    training, bank = {"checkpoint": identity("checkpoint")}, {"ref": identity("bank")}
    endpoint = {"arm": BASE, "epoch": 100, "assessment": {"sample_ids": ids, "source_native": identity("source")},
        "training": training, "banks": {"native": bank}, "plan": {"spec": identity("spec")},
        "result": {"plan": identity("plan")}}
    nominal = {"sealed_unix": 100.}; nomination_path = tmp_path/"nomination.json"; nomination_path.write_text(json.dumps(nominal))
    release = {"schema": "car_round7_task_release_v1", "status": "sealed", "split": "validation", "test_locked": True,
        "selection_use": False, "nomination": ref(nomination_path), "released_unix": 101.,
        "child_checkpoint_sha256": "checkpoint", "child_native_receipt_sha256": "bank", "reason": "post nomination"}
    release_path = tmp_path/"release.json"; release_ref = write_json(release_path, release)
    task = {"status": "completed", "split": "validation", "members": 4, "sample_ids": ids, "test_locked": True,
        "decision": release_ref, "child_native": bank["ref"], "source_native": identity("source"),
        "spec": identity("spec"), "paired": {"event": {"mean": -.1}}}
    task_ref = write_json(tmp_path/"tasks.json", task)
    receipt_ref = write_json(tmp_path/"task_receipt.json", {"status": "completed"})
    result = {"schema": "car_round8_validation_task_assessment_v1", "status": "completed", "split": "validation", "members": 4,
        "sample_ids": ids, "test_locked": True, "arm": BASE, "epoch": 100, "plan": identity("plan"),
        "checkpoint": identity("checkpoint"), "native_bank": bank["ref"], "release": release_ref, "tasks": task_ref, "task_receipt": receipt_ref}
    result_path = tmp_path/"result.json"; write_json(result_path, result)
    assert released_tasks(result_path, endpoint, nomination_path, nominal)["selection_use"] is False
    release["released_unix"] = 99.; result["release"] = write_json(release_path, release); write_json(result_path, result)
    with pytest.raises(ValueError, match="after and bound"):
        released_tasks(result_path, endpoint, nomination_path, nominal)


def test_training_exposure_is_bounded_to_endpoint_even_after_resume(tmp_path):
    run = tmp_path/"run"; logs = run/"logs"; metrics = run/"metrics"; logs.mkdir(parents=True); metrics.mkdir()
    history = [{"epoch": i, "step": 12*i, "epoch_complete": True, "batches": 12,
                "data_wait_seconds": .25, "update_seconds": 2.} for i in range(1, 4)]
    timing = [{"epoch": i, "step": 12*i, "complete_epoch_duration": True,
               "wall_seconds": 30., "train_seconds": 27.} for i in range(1, 4)]
    coupling = [{"step": i, "accepted": True, "sample_ids": ["condition-1", "condition-2"],
        "observation_sha256": "obs", "query_sha256": "query", "member_seeds": [1, 2],
        "first_member_generator_state_sha256": "rng", "field_complement_sha256": "complement",
        "field_complement_seed": i} for i in range(1, 37)]
    for path, rows in ((logs/"history.jsonl", history), (logs/"epoch_timing.jsonl", timing), (metrics/"training_coupling.jsonl", coupling)):
        path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    write_json(logs/"recording_definitions.json", {"version": "ordered_epoch_recording_v1",
        "columns": [{"name": "epoch", "type": "number", "units": "reporting_epochs", "meaning": "completed epoch"}],
        "accumulation_order": ["data_loss"], "loss_component_accumulation_order": ["state", "scale", "shape"]})
    (logs/"loss_history.png").write_bytes(b"curve png")
    (logs/"loss_history.pdf").write_bytes(b"curve pdf")
    campaign = tmp_path/"campaign"; lease = campaign/"operations/completions/lease-token.json"
    write_json(lease, {"schema": "car_round8_gpu_lease_v1", "status": "completed", "phase": "learning", "token": "lease-token", "seconds": 100.})
    endpoint = {"epoch": 2, "arm": BASE, "training": {"run_dir": str(run), "lease_token": "lease-token"}}
    result = training_evidence(endpoint, campaign, population=8)["summary"]
    assert result["history_saved_through_epoch"] == 3
    assert result["accepted_objective_condition_visits"] == 48
    assert result["dataset_pass_equivalents"] == 6.
    assert len(result["expected_exposure_deviations"]) == 2
    assert result["completed_epoch_wall_seconds"] == 60.
    assert result["summed_data_wait_seconds"] == 6.
    assert result["summed_update_seconds"] == 48.
    (logs/"epoch_timing.jsonl").unlink(); lease.unlink()
    missing = training_evidence(endpoint, campaign, population=8)["summary"]
    assert missing["accepted_objective_condition_visits"] == 48
    assert missing["mean_complete_epoch_wall_seconds"] is None
    assert missing["lease_segment_occupied_gpu_seconds"] is None


def test_selected_continuation_preserves_100_prefix_full_history_and_distinct_leases(tmp_path):
    from scripts.analysis.car_round8_learning_summary import (
        capture_training, coupling_comparison, selected_training_evidence,
    )

    run = tmp_path/"run"; logs = run/"logs"; metrics = run/"metrics"
    logs.mkdir(parents=True); metrics.mkdir()
    history = [{"epoch": i, "step": 12*i, "epoch_complete": True, "batches": 12,
                "data_wait_seconds": .1 if i <= 100 else 5., "update_seconds": 1. if i <= 100 else 4.}
               for i in range(1, 501)]
    timing = [{"epoch": i, "step": 12*i, "complete_epoch_duration": True,
               "wall_seconds": 2. if i <= 100 else 200., "train_seconds": 1.5 if i <= 100 else 199.5}
              for i in range(1, 501)]
    coupling = [{"step": i, "accepted": True, "sample_ids": list(range(64)),
        "observation_sha256": "obs", "query_sha256": "query", "member_seeds": [1, 2],
        "first_member_generator_state_sha256": "rng", "field_complement_sha256": "complement",
        "field_complement_seed": i} for i in range(1, 6001)]
    paths = ((logs/"history.jsonl", history, 100), (logs/"epoch_timing.jsonl", timing, 100),
             (metrics/"training_coupling.jsonl", coupling, 1200))
    for path, records, cut in paths:
        path.write_text("".join(json.dumps(row)+"\n" for row in records[:cut]))
    write_json(logs/"recording_definitions.json", {"version": "ordered_epoch_recording_v1",
        "columns": [{"name": "epoch", "type": "number", "units": "reporting_epochs", "meaning": "completed epoch"}],
        "accumulation_order": ["data_loss"], "loss_component_accumulation_order": ["state", "scale", "shape"]})
    (logs/"loss_history.png").write_bytes(b"saved full-run curve png")
    (logs/"loss_history.pdf").write_bytes(b"saved full-run curve pdf")
    campaign = tmp_path/"campaign"; completions = campaign/"operations/completions"
    for token, seconds in (("initial-lease", 100.), ("continuation-lease", 400.)):
        write_json(completions/(token+".json"), {"schema": "car_round8_gpu_lease_v1", "status": "completed",
            "phase": "learning", "token": token, "seconds": seconds})
    initial = {"arm": SPATIAL, "epoch": 100, "training": {"run_dir": str(run), "lease_token": "initial-lease"}}
    final = {"arm": SPATIAL, "epoch": 500, "training": {"run_dir": str(run), "lease_token": "continuation-lease"}}
    before = training_evidence(initial, campaign, population=8000)
    for path, records, cut in paths:
        with path.open("a") as stream:
            stream.write("".join(json.dumps(row)+"\n" for row in records[cut:]))
    after = training_evidence(initial, campaign, population=8000)
    assert after["summary"]["history_saved_through_epoch"] == 500
    assert after["summary"]["accepted_objective_condition_visits"] == 76800
    assert after["summary"]["completed_epoch_wall_seconds"] == 200.
    assert after["summary"]["summed_update_seconds"] == 1200.
    assert len(after["coupling_rows"]) == 1200 and not any(coupling_comparison(before, after).values())
    selected = selected_training_evidence(initial, after, final, campaign, 8000)
    assert selected["summary"]["accepted_objective_condition_visits"] == 384000
    assert selected["summary"]["completed_epoch_wall_seconds"] == 80200.
    assert selected["summary"]["lineage_occupied_gpu_seconds"] == 500.
    same_lease = deepcopy(final); same_lease["training"]["lease_token"] = "initial-lease"
    assert selected_training_evidence(initial, after, same_lease, campaign, 8000)["summary"]["lineage_occupied_gpu_seconds"] == 100.
    missing_lease = deepcopy(final); missing_lease["training"]["lease_token"] = "missing"
    assert selected_training_evidence(initial, after, missing_lease, campaign, 8000)["summary"]["lineage_occupied_gpu_seconds"] is None
    wrong_run = deepcopy(final); wrong_run["training"]["run_dir"] = str(tmp_path/"different_run")
    with pytest.raises(ValueError, match="epoch-100 training run"):
        selected_training_evidence(initial, after, wrong_run, campaign, 8000)
    output = tmp_path/"bundle"; (output/"figures").mkdir(parents=True)
    captions = {}; snapshots = capture_training({SPATIAL: selected}, output, captions)
    saved = Path(snapshots[SPATIAL]["history"]["path"])
    assert len(saved.read_text().splitlines()) == 500
    assert "through reporting epoch 500" in captions[f"training_{SPATIAL}_loss_history"]


def test_closed_historical_joint500_direct_observer_parity():
    """A closed historical fixture checks formulas; it is not Round 8 learning."""
    from pathlib import Path
    from scripts.analysis.car_round8_learning_summary import family_arrays, observer_identity

    base = Path("cases/turbulent_combustion/runs/coherence_adaptive_repair")
    assessment_path = base/"round_7/assessment/JointAnchor_Selected500_CPUScore_v2/validation/assessment.json"
    relation_path = base/"round_8/assessment/JointAnchor500_RelationParity_v1/relation_context.json"
    position_path = base/"round_8/assessment/SpatialShape_Common_VALIDATION100_v1/readout.json"
    if not all(p.is_file() for p in (assessment_path, relation_path, position_path)):
        pytest.skip("optional closed campaign fixtures are unavailable")
    assessment, relation, position = [json.loads(p.read_text()) for p in (assessment_path, relation_path, position_path)]
    decomposition = json.loads((assessment_path.parent/"decomposition.json").read_text())
    for row in position["records"]:
        row["models"]["candidate"] = row["models"].pop("JointAnchor500")
    endpoint_data = {"assessment": assessment, "decomposition": decomposition,
                     "relation": relation, "position": position}
    arrays, weights = family_arrays(endpoint_data)
    assert arrays["SOURCE"]["State"][:, 2].mean() == pytest.approx(.3604483467, rel=1e-8)
    assert arrays["candidate"]["State"][:, 2].mean() == pytest.approx(.3651106149, rel=1e-8)
    assert arrays["candidate"]["PH Shape"][:, 2].mean()-arrays["SOURCE"]["PH Shape"][:, 2].mean() == pytest.approx(-.43117588, abs=1e-7)
    historical = json.loads((base/"round_8/assessment/Cached_Contract_Reassessment_v1/JointAnchor500_VALIDATION100.json").read_text())
    np.testing.assert_allclose(arrays["candidate"]["Complete expanded Scale"].mean(0),
                               historical["scale"]["candidate"]["Complete_Expanded_Prefit"], rtol=1e-10)
    indices = np.random.default_rng(6042026).integers(0, 100, size=(2000, 100))
    components = component_comparison({"Joint500": endpoint_data}, indices, weights)
    for side, label in (("SOURCE", "SOURCE"), ("candidate", "Joint500")):
        key = "SOURCE_mean_ARS" if side == "SOURCE" else label + "_mean_ARS"
        for block in ("signed", "joint", "log_power_self", "rho", "third_order"):
            np.testing.assert_allclose(components["Scale blocks"][block][key], historical["scale"][side][block], rtol=1e-10, atol=1e-12)
        for leaf in ("self.finite", "self.essential", "mutual.finite", "mutual.essential"):
            np.testing.assert_allclose(components["PH Shape components"][leaf][key], historical["shape"][side][leaf], rtol=1e-10, atol=1e-12)
    assert components["Scale blocks"]["joint"]["fixed_contribution_factor"] == pytest.approx(.25 * decomposition["prefit_coefficients"]["scale.joint"])
    common = json.loads(position_path.read_text())
    spec = json.loads(Path(assessment["spec"]["path"]).read_text())
    fit = json.loads(Path(common["inputs"]["fit"]["path"]).read_text())
    endpoint = {"assessment": assessment, "decomposition": decomposition, "relation": relation, "spec": spec,
        "position": {"fit": common["inputs"]["fit"], "cached_source": ref(position_path),
                     "kind_weights": common["kind_weights"], "location_unit": common["location_unit"]}}
    observer = observer_identity(endpoint, fit)
    assert observer["fit_query_count"] == 130 and len(observer["observation_weighted_queries"]) == 25
    assert observer["fit_kind_weights"] != observer["observation_kind_weights"]


def test_nomination_binds_both_results_and_prepared_manifest(tmp_path):
    prepared = write_json(tmp_path/"prepared.json", {"fixture": True})
    admission = write_json(tmp_path/"admission.json", {"status": "sealed", "admitted": True, "test_locked": True})
    base_path, spatial_path = tmp_path/"base.json", tmp_path/"spatial.json"
    base_result, spatial_result = write_json(base_path, {"which": BASE}), write_json(spatial_path, {"which": SPATIAL})
    base = {"arm": BASE, "result_path": base_path, "plan": {"prepared": prepared}, "training": {"checkpoint": identity("base-checkpoint")}}
    spatial = {"arm": SPATIAL, "result_path": spatial_path, "plan": {"prepared": prepared}, "training": {"checkpoint": identity("spatial-checkpoint")}}
    decision = {"schema": "car_round8_learning_review_epoch_100_v1", "status": "sealed", "comparison_epoch": 100,
        "sealed_unix": 100., "tasks_excluded_from_nomination": True, "selected_arm": BASE, "continue_to_epoch": 500,
        "reason": "continue the selected package", "prepared": prepared, "learning_decision": admission,
        "assessments": {BASE: base_result, SPATIAL: spatial_result},
        "checkpoints": {BASE: identity("base-checkpoint"), SPATIAL: identity("spatial-checkpoint")}}
    path = tmp_path/"nomination.json"; write_json(path, decision)
    assert nomination_check(path, base, spatial)[1]["selected_arm"] == BASE
    decision["tasks_excluded_from_nomination"] = False; write_json(path, decision)
    with pytest.raises(ValueError, match="nomination"):
        nomination_check(path, base, spatial)


def test_endpoint_receipts_bind_snapshot_after_mutable_checkpoint_changes(tmp_path):
    from scripts.analysis.car_round8_learning_summary import load_endpoint

    data = synthetic_endpoint(BASE); ids = data["assessment"]["sample_ids"]
    config = write_json(tmp_path/"prepared.yaml", {"fixture": "configuration"})
    fit = write_json(tmp_path/"position_fit.json", {"fixture": "position"})
    prepared = write_json(tmp_path/"prepared.json", {"arms": {BASE: config}, "position_fit": fit})
    original = tmp_path/"last.pt"; original.write_bytes(b"recorded 100 checkpoint")
    original_ref = ref(original)
    checkpoint = tmp_path/"endpoint_checkpoint.pt"; checkpoint.write_bytes(original.read_bytes())
    checkpoint_ref = ref(checkpoint)
    training = {"schema": "car_round8_training_result_v1", "arm": BASE, "target_epoch": 100,
        "reporting_epoch": 100, "global_step": 1200, "target_reached": True, "disposition": "target_reached",
        "test_locked": True, "checkpoint": original_ref, "prepared_config": config, "saved_config": config}
    train_path = tmp_path/"training_result.json"; train_ref = write_json(train_path, training)
    panel = write_json(tmp_path/"panel.json", {"sample_ids": ids, "split": "validation", "native_points": 4})
    source_checkpoint = write_json(tmp_path/"source_checkpoint.json", {"fixture": True})
    source_native = write_json(tmp_path/"source_native.json", {"status": "completed", "sample_ids": ids,
        "members": 4, "native_points": 4, "source_sha256": source_checkpoint["sha256"]})
    source_selected = write_json(tmp_path/"source_selected.json", {"fixture": True})
    source_baseline = write_json(tmp_path/"source_baseline.json", {"fixture": True})
    spec = {key: write_json(tmp_path/(key+".json"), {"name": key}) for key in SPEC_KEYS
            if key not in ("panel", "source_checkpoint", "source_native_receipt")}
    spec.update(panel=panel, source_checkpoint=source_checkpoint, source_native_receipt=source_native,
                checkpoint=checkpoint_ref, reporting_epoch=100)
    spec_ref = write_json(tmp_path/"spec.json", spec)
    plan = {"schema": "car_round8_validation_endpoint_plan_v1", "arm": BASE, "epoch": 100,
        "sample_ids": ids, "test_locked": True, "training_result": train_ref, "checkpoint": checkpoint_ref,
        "original_training_checkpoint": original_ref, "spec": spec_ref, "prepared": prepared,
        "source_baseline": source_baseline, "source_selected": source_selected, "position_fit": fit}
    plan_ref = write_json(tmp_path/"plan.json", plan)
    banks = {}
    for support in ("native", "selected"):
        banks[support] = write_json(tmp_path/(support+".json"), {"status": "completed", "split": "validation", "support": support,
            "sample_ids": ids, "members": 4, "checkpoint_sha256": checkpoint_ref["sha256"],
            "spec_sha256": spec_ref["sha256"], "files_sha256": ["row"]*100})
    generation_ref = write_json(tmp_path/"generation.json", {"status": "completed", "plan": plan_ref, "banks": banks})
    assessment = data["assessment"]
    assessment.update(schema="car_round7_matched_assessment_v1", status="completed", split="validation", test_locked=True,
                      spec=spec_ref, source_native=source_native, source_selected=source_selected, source_baseline=source_baseline,
                      child_native=banks["native"], child_selected=banks["selected"])
    assessment_ref = write_json(tmp_path/"assessment.json", assessment)
    decomposition = data["decomposition"]
    decomposition.update(status="completed", sample_ids=ids, members=4, field_order=assessment["field_order"], spec=spec_ref,
                         source_receipt=source_selected, child_receipt=banks["selected"])
    decomp_ref = write_json(tmp_path/"decomposition.json", decomposition)
    member = data["member"]
    member.update(schema="car_round8_cached_summary_v1", split="validation", sample_ids=ids, members=4, test_locked=True,
                  bootstrap={"seed": 6042026, "replicates": 2000}, inputs={"assessment": assessment_ref, "decomposition": decomp_ref})
    member_ref = write_json(tmp_path/"member.json", member)
    scale_ref = write_json(tmp_path/"scale.json", {"fixture": True})
    position = data["position"]
    position.update(schema="car_round8_endpoint_position_readout_v1", status="completed", split="validation", members=4,
                    sample_ids=ids, test_locked=True, fit=fit, bootstrap={"seed": 6042026, "replicates": 2000})
    position_ref = write_json(tmp_path/"position.json", position)
    source_companion = write_json(tmp_path/"source_companion.json", {"fixture": "completed source details"})
    relation = data["relation"]
    relation.update(schema="car_round8_endpoint_relation_context_v1", status="completed", split="validation", members=4,
        sample_ids=ids, test_locked=True, endpoint_label=BASE, reporting_epoch=100,
        bootstrap={"seed": 6042026, "replicates": 2000},
        inputs={"assessment": assessment_ref, "decomposition": decomp_ref, "selected_bank": banks["selected"],
                "spec": spec_ref, "source_companion": source_companion})
    relation_ref = write_json(tmp_path/"relation.json", relation)
    result = {"schema": "car_round8_validation_assessment_v1", "status": "completed", "split": "validation", "members": 4,
        "sample_ids": ids, "test_locked": True, "arm": BASE, "epoch": 100, "plan": plan_ref, "generation": generation_ref,
        "assessment": assessment_ref, "member_mse_summary": member_ref, "scale_summary": scale_ref,
        "position_readout": position_ref, "relation_context": relation_ref}
    result_path = tmp_path/"assessment_result.json"; write_json(result_path, result)
    original.write_bytes(b"overwritten by selected continuation")
    assert load_endpoint(train_path, result_path, expected_epoch=100)["epoch"] == 100
    relation["inputs"].pop("source_companion")
    result["relation_context"] = write_json(tmp_path/"relation.json", relation); write_json(result_path, result)
    with pytest.raises(ValueError, match="SOURCE companion is required"):
        load_endpoint(train_path, result_path, expected_epoch=100)
    relation["inputs"]["source_companion"] = source_companion
    result["relation_context"] = write_json(tmp_path/"relation.json", relation); write_json(result_path, result)
    checkpoint.write_bytes(b"corrupt immutable endpoint")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_endpoint(train_path, result_path, expected_epoch=100)


def test_only_saved_source_examples_are_rendered_with_masked_physical_rmse(tmp_path, monkeypatch):
    from pathlib import Path
    import torch
    import scripts.analysis.car_round8_learning_summary as learning

    ids = [f"frame-{i:03}" for i in range(100)]
    data = {"sample_ids": ids, "examples": {"representative_median": {"sample_id": ids[49]},
             "difficult_maximum": {"sample_id": ids[99]}}}
    coordinate = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    truth = torch.tensor([[[1., 10.], [2., 11.], [3., 12.], [4., 13.]]], dtype=torch.float64)
    calls, errors = [], []
    def row(offset, ordinal):
        return {"sample_id": ids[ordinal], "sample_index": ordinal, "member_seeds": [1, 2, 3, 4],
            "truth_model": truth, "members_model": torch.stack([truth + offset]*4),
            "coordinates_model": coordinate, "coordinates_raw": coordinate,
            "native_masses": torch.tensor([1., 2., 3., 0.]),
            "obs_indices": torch.tensor([0]), "obs_values": torch.tensor([0.]), "obs_fields": torch.tensor([0]),
            "normalizer_offset": torch.tensor([0., 0.]), "normalizer_scale": torch.tensor([1., 1.])}
    def fake_read(path, receipt, ordinal):
        calls.append((str(path), ordinal)); return row({"source": .5003e-6, "base": 1.0003e-6, "spatial": 2.0003e-6, "final": 3.0003e-6}[str(path)], ordinal)
    real_rmse = learning.physical_rmse
    def measured(*args):
        value = real_rmse(*args); errors.append(value); return value
    real_save = learning.save_figure
    def checked_save(fig, folder, name):
        fig.canvas.draw(); renderer = fig.canvas.get_renderer()
        headings = [ax for ax in fig.axes if ax.get_title()]
        assert len(headings) == 10
        for ax in headings:
            label, panel = ax.title.get_window_extent(renderer), ax.get_window_extent(renderer)
            assert label.x0 >= panel.x0 - .1 and label.x1 <= panel.x1 + .1
        assert sum("SpatialShape\nepoch 500" in ax.get_title() for ax in headings) == 2
        assert sum("epoch 100" in ax.get_title() for ax in headings) == 4
        real_save(fig, folder, name)
    monkeypatch.setattr(learning, "read_row", fake_read); monkeypatch.setattr(learning, "physical_rmse", measured)
    monkeypatch.setattr(learning, "save_figure", checked_save)
    base = {"assessment": {"field_order": ["CO", "T"]}, "source_bank": {"path": Path("source"), "receipt": {}},
            "banks": {"native": {"path": Path("base"), "receipt": {}}}}
    spatial = {"banks": {"native": {"path": Path("spatial"), "receipt": {}}}}
    final = {"arm": SPATIAL, "epoch": 500, "banks": {"native": {"path": Path("final"), "receipt": {}}}}
    captions = learning.example_figures(data, base, spatial, final, tmp_path)
    assert len(captions) == 2 and len(errors) == 16
    assert set(round(value * 1e6, 4) for value in errors) == {.5003, 1.0003, 2.0003, 3.0003}
    assert [ordinal for _, ordinal in calls] == [49, 49, 49, 49, 99, 99, 99, 99]
    assert {p.suffix for p in tmp_path.iterdir()} == {".png", ".pdf"}


def test_paired_scientific_figure_pairs_render_with_fixed_arm_labels(tmp_path, monkeypatch):
    import scripts.analysis.car_round8_learning_summary as learning

    base, spatial = synthetic_endpoint(BASE), synthetic_endpoint(SPATIAL)
    summary = assemble(base, spatial)
    monkeypatch.setattr(learning, "example_figures", lambda *_: {})
    captions = learning.make_figures(summary, base, spatial, None, tmp_path)
    assert set(captions) == {"learning_field_ratios", "learning_paired_direct_ars", "learning_signed_rho_power"}
    for name in captions:
        assert (tmp_path/(name+".png")).is_file() and (tmp_path/(name+".pdf")).is_file()
        assert "\n" not in captions[name]


def test_training_coupling_compares_acceptance_and_actual_streams():
    from scripts.analysis.car_round8_learning_summary import coupling_comparison

    row = {"sample_ids": ["sample"], "observation_sha256": "obs", "query_sha256": "query",
           "member_seeds": [1, 2], "first_member_generator_state_sha256": "rng",
           "field_complement_sha256": "complement", "field_complement_seed": 3, "accepted": True}
    altered = deepcopy(row); altered["accepted"] = False; altered["query_sha256"] = "different"
    result = coupling_comparison({"coupling_rows": [row]}, {"coupling_rows": [altered]})
    assert result["accepted"] == result["query_sha256"] == 1
    assert result["sample_ids"] == result["field_complement_sha256"] == 0


def test_section_labels_paired_contrast_and_separates_released_tasks():
    from scripts.analysis.car_round8_learning_summary import make_section

    base, spatial = synthetic_endpoint(BASE), synthetic_endpoint(SPATIAL)
    data = assemble(base, spatial)
    data["training_coupling_mismatch_counts"] = {"sample_ids": 0, "accepted": 0}
    training = {arm: {"reporting_epoch": 100, "accepted_objective_condition_visits": 76800,
        "dataset_pass_equivalents": 9.6, "mean_complete_epoch_wall_seconds": 20.,
        "lease_segment_occupied_gpu_seconds": None, "expected_exposure_deviations": [], "timing_covered_epochs": 100,
        "mature_epochs_51_100": {"complete_epoch_records": 50, "mean_wall_seconds": 20.,
            "mean_train_span_seconds": 19., "mean_callback_and_reporting_seconds": 1.},
        "whole_worker_torch_memory": {"peak_allocated_bytes": 2**30, "peak_reserved_bytes": 2*2**30}}
        for arm in (BASE, SPATIAL)}
    data["assessment_costs"] = {arm: {"generation_lease_occupied_gpu_seconds": None,
        "cpu_score_wall_seconds": 5., "cpu_score_process_seconds": 8., "cpu_score_reused_completed_scores": False}
        for arm in (BASE, SPATIAL)}
    tasks = [{"arm": BASE, "epoch": 100, "paired_frozen_train_units":
              {"event": {"mean": -.2, "interval_95": [-.3, -.1]}, "regional": {"mean": .1, "interval_95": [-.1, .2]}}}]
    text = make_section(data, training, tasks, None)
    assert "SpatialShape minus BaseShape" in text
    assert "ratios of pooled member MSE" in text
    assert "not percentages of VALIDATION SOURCE scores" in text
    assert "unavailable" in text and "no scalar training objective" in text
    assert "| MemberMSE_BaseShape | 100 | -0.2 [-0.3, -0.1] | 0.1 [-0.1, 0.2] |" in text
    assert "http://" not in text and "https://" not in text


def test_selected_final_is_not_presented_as_matched_package_attribution():
    from scripts.analysis.car_round8_learning_summary import final_summary, make_section

    base, spatial = synthetic_endpoint(BASE), synthetic_endpoint(SPATIAL)
    comparison = assemble(base, spatial)
    later = deepcopy(spatial); later["epoch"] = 500; later["relation"]["summary"] = {"scale": {}}
    final = final_summary(later, base)
    assert "unequal training age" in final["exposure"]
    comparison["training_coupling_mismatch_counts"] = {"accepted": 0}
    training = {arm: {"reporting_epoch": 100, "accepted_objective_condition_visits": 76800,
        "dataset_pass_equivalents": 9.6, "mean_complete_epoch_wall_seconds": 20.,
        "lease_segment_occupied_gpu_seconds": 2100., "expected_exposure_deviations": [], "timing_covered_epochs": 100,
        "mature_epochs_51_100": {"complete_epoch_records": 50, "mean_wall_seconds": 20.,
            "mean_train_span_seconds": 19., "mean_callback_and_reporting_seconds": 1.},
        "whole_worker_torch_memory": {"peak_allocated_bytes": 2**30, "peak_reserved_bytes": 2*2**30}}
        for arm in (BASE, SPATIAL)}
    costs = {"generation_lease_occupied_gpu_seconds": 400., "cpu_score_wall_seconds": 10.,
             "cpu_score_process_seconds": 20., "cpu_score_reused_completed_scores": False}
    training[SPATIAL]["history_saved_through_epoch"] = 500
    comparison["assessment_costs"] = {BASE: costs, SPATIAL: costs}
    comparison["selected_final_assessment_cost"] = costs
    comparison["selected_final_training"] = {"accepted_objective_condition_visits": 384000,
        "dataset_pass_equivalents": 48., "mean_complete_epoch_wall_seconds": 20., "lineage_occupied_gpu_seconds": 10200.,
        "whole_worker_torch_memory": {"peak_allocated_bytes": 2**30, "peak_reserved_bytes": 2*2**30, "scope": "continuation worker"}}
    section = make_section(comparison, training, [], final)
    assert "Nominated later endpoint: MemberMSE_SpatialShape at epoch 500" in section
    assert "different ages do not identify the positional package effect" in section
    assert "counting each distinct lease once" in section
    assert "recording snapshots and saved curves include the full run through epoch 500" in section
    assert "matched epoch-100 exposure, timing and coupling above use only records through epoch 100" in section


def test_sensor_readout_only_claims_fields_with_observations():
    base, spatial = synthetic_endpoint(BASE), synthetic_endpoint(SPATIAL)
    for endpoint in (base, spatial):
        endpoint["assessment"]["native_safeguards"] = {"sensor_pass": True}
        for side in ("SOURCE", "candidate"):
            for row in endpoint["assessment"]["per_condition_fields"][side]:
                row["sensor_counts"] = [0, 256]; row["sensor_mse"] = [0., .01 if side == "SOURCE" else .02]
    result = assemble(base, spatial)["sensor_compatibility"]
    assert result[BASE]["saved_sensor_pass"] is True
    assert result[SPATIAL]["observations_per_condition_min"] == [0, 256]
    assert result[BASE]["candidate_mse"] == pytest.approx([0., .02])


def test_zero_source_sensor_error_retains_absolute_allowance_for_selected_endpoint():
    from scripts.analysis.car_round8_learning_summary import final_summary

    base, spatial = synthetic_endpoint(BASE), synthetic_endpoint(SPATIAL)
    for endpoint in (base, spatial):
        endpoint["assessment"]["native_safeguards"] = {
            "sensor_pass": True, "sensor_allowance": [1e-9, 1e-11]}
        for side in ("SOURCE", "candidate"):
            for row in endpoint["assessment"]["per_condition_fields"][side]:
                row["sensor_counts"] = [0, 64]
                row["sensor_mse"] = [0., 0. if side == "SOURCE" else 5e-12]
    later = deepcopy(spatial)
    later["epoch"] = 300
    later["relation"]["summary"] = {"scale": {}}
    selected = final_summary(later, base)["sensor_compatibility"]
    assert selected["SOURCE_mse"] == [0., 0.]
    assert selected["candidate_mse"] == pytest.approx([0., 5e-12])
    assert selected["saved_sensor_allowance"] == [1e-9, 1e-11]
    table = "\n".join(sensor_table({SPATIAL: selected}, ["CO", "T"], {SPATIAL: "Selected"}))
    assert "| T | 64–64 | 0 | 5e-12 | 1e-11 |" in table
    assert "| CO |" not in table
    assert "absolute numerical tolerance" in table and "no zero-baseline ratio" in table


def test_sensor_allowance_requires_field_order_and_matches_the_common_source():
    endpoint = synthetic_endpoint(BASE)
    for side in ("SOURCE", "candidate"):
        for row in endpoint["assessment"]["per_condition_fields"][side]:
            row.update(sensor_counts=[0, 64], sensor_mse=[0., .01])
    endpoint["assessment"]["native_safeguards"] = {"sensor_pass": True, "sensor_allowance": [1e-9]}
    with pytest.raises(ValueError, match="sensor allowance"):
        sensor_evidence(endpoint)
    endpoint["assessment"]["native_safeguards"]["sensor_allowance"] = [1e-9, .001]
    source = sensor_evidence(endpoint)
    changed = deepcopy(source)
    changed["saved_sensor_allowance"][1] = .002
    with pytest.raises(ValueError, match="SOURCE observation"):
        sensor_table({BASE: source, SPATIAL: changed}, ["CO", "T"], {BASE: "Base", SPATIAL: "Spatial"})


def test_mature_timing_and_worker_memory_use_completed_saved_history(tmp_path):
    run = tmp_path/"run"; logs = run/"logs"; metrics = run/"metrics"; logs.mkdir(parents=True); metrics.mkdir()
    history = [{"epoch": i, "step": 12*i, "epoch_complete": True, "batches": 12} for i in range(1, 101)]
    timing = [{"epoch": i, "step": 12*i, "complete_epoch_duration": True, "wall_seconds": i+10.,
               "train_seconds": float(i), "callback_and_reporting_seconds": 10.} for i in range(1, 101)]
    coupling = [{"step": i, "accepted": True, "sample_ids": list(range(64)), "observation_sha256": "obs",
        "query_sha256": "query", "member_seeds": [1, 2], "first_member_generator_state_sha256": "rng",
        "field_complement_sha256": "complement", "field_complement_seed": i} for i in range(1, 1201)]
    for path, rows in ((logs/"history.jsonl", history), (logs/"epoch_timing.jsonl", timing), (metrics/"training_coupling.jsonl", coupling)):
        path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    write_json(logs/"recording_definitions.json", {"version": "ordered_epoch_recording_v1",
        "columns": [{"name": "epoch", "type": "number", "units": "reporting_epochs", "meaning": "completed epoch"}],
        "accumulation_order": ["data_loss"], "loss_component_accumulation_order": ["state", "scale", "shape"]})
    (logs/"loss_history.png").write_bytes(b"curve png"); (logs/"loss_history.pdf").write_bytes(b"curve pdf")
    campaign = tmp_path/"campaign"; (campaign/"operations/completions").mkdir(parents=True)
    memory = {"peak_allocated_bytes": 1234, "peak_reserved_bytes": 2345, "scope": "current worker including setup"}
    endpoint = {"epoch": 100, "arm": BASE, "training": {"run_dir": str(run), "lease_token": "missing", "torch_memory": memory}}
    result = training_evidence(endpoint, campaign, population=8000)["summary"]
    assert result["whole_worker_torch_memory"] == memory
    assert result["accepted_objective_condition_visits"] == 76800 and result["dataset_pass_equivalents"] == 9.6
    assert result["expected_exposure_deviations"] == []
    mature = result["mature_epochs_51_100"]
    assert mature["complete_coverage"] and mature["complete_epoch_records"] == 50
    assert mature["mean_train_span_seconds"] == 75.5 and mature["mean_wall_seconds"] == 85.5
    assert mature["mean_callback_and_reporting_seconds"] == 10.
