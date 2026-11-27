import json
from pathlib import Path

import pytest

from scripts.analysis.assemble_car_round8_report import (
    LEARNING_ARMS, copy_publication, dated_section, lead_report, localize_links,
    require_final_readouts, verify_bundle,
)
from scripts.analysis.car_round8_cached_summary import ref
from scripts.analysis.report_car_round8 import add_extra_figures
from tools.docs.check_markdown_links import check_document


def learning_bundle(path):
    path.mkdir(); (path / "figures").mkdir()
    (path / "section.md").write_text("## Completed learning\n\nBounded fixture.\n")
    (path / "summary.json").write_text("{}\n")
    (path / "figure_captions.json").write_text('{"learning_curve": "Recorded training curve."}\n')
    (path / "figures/learning_curve.png").write_bytes(b"png preview")
    (path / "figures/learning_curve.pdf").write_bytes(b"pdf master")
    names = ["section.md", "summary.json", "figure_captions.json", "figures/learning_curve.png", "figures/learning_curve.pdf"]
    for arm in LEARNING_ARMS:
        recording = path / "recordings" / arm; recording.mkdir(parents=True)
        for name in ("history.jsonl", "recording_definitions.json"):
            (recording / name).write_text("{}\n"); names.append(f"recordings/{arm}/{name}")
    index = {"schema": "car_round8_learning_summary_v1", "status": "completed", "test_locked": True,
             "additional_inference": False, "files_sha256": {name: ref(path / name)["sha256"] for name in names}}
    (path / "evidence_index.json").write_text(json.dumps(index))
    return index


def test_learning_inputs_are_completed_and_hash_bound(tmp_path):
    folder = tmp_path / "learning"; learning_bundle(folder)
    assert verify_bundle(folder, "section.md", learning=True)["manifest"] == folder / "figure_captions.json"
    (folder / "section.md").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_bundle(folder, "section.md", learning=True)
    learning_bundle_other = tmp_path / "other"; index = learning_bundle(learning_bundle_other)
    index["status"] = "running"; (learning_bundle_other / "evidence_index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="completed locked"):
        verify_bundle(learning_bundle_other, "section.md", learning=True)
    index["status"] = "completed"; index["files_sha256"].pop(f"recordings/{LEARNING_ARMS[0]}/history.jsonl")
    (learning_bundle_other / "evidence_index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="missing required outputs"):
        verify_bundle(learning_bundle_other, "section.md", learning=True)


def test_lead_and_historical_provenance_do_not_leave_stale_status(tmp_path):
    campaign = tmp_path / "campaign"; source = campaign / "assessment/qualification/section.md"
    source.parent.mkdir(parents=True); source.write_text("source")
    target = campaign / "operations/receipt.json"; target.parent.mkdir(); target.write_text("{}")
    text = "Evidence [receipt](../../operations/receipt.json)."
    assert localize_links(text, source, campaign) == "Evidence receipt (`operations/receipt.json`)."
    original = "# Coherence Adaptive Repair Round 8\n\nRound 8 engineering and any new learning are identified separately below.\n\n## Scope and remaining evidence\n"
    report = lead_report(original, "## Decision\n\nCoordinator decision.")
    assert report.index("Coordinator decision.") < report.index("## Historical cached reassessment")
    assert "any new learning" not in report and "## Scope and diagnostic cost" in report
    qualification = "Qualified. The SOURCE/C8 model decision awaits the admitted learning comparison."
    assert dated_section("Engineering_Qualification_Readout_v1", qualification) == "Qualified."
    executor = ("# Audit\n\nStale recommendation.\n\n## Matched historical measurement and changed-objective timings\n\n"
                "Actual measured results.\n\n## Cost and remaining opportunity\n\nHistorical cost.\n\n"
                "The completed Round 8 cached VALIDATION100 position-only readout old future forecast.")
    cleaned = dated_section("Executor_Retention_Audit_v1", executor)
    assert "Actual measured results." in cleaned and "Historical cost." in cleaned
    assert "Stale recommendation" not in cleaned and "old future forecast" not in cleaned


def test_final_assembly_requires_released_matched_and_selected_task_readouts(tmp_path):
    folder = tmp_path / "learning"
    learning_bundle(folder)
    bundle = {"folder": folder}
    with pytest.raises(ValueError, match="sealed epoch-100 nomination"):
        require_final_readouts(bundle)
    nomination_path = tmp_path / "nomination.json"
    nomination_path.write_text(json.dumps({"schema": "car_round8_learning_review_epoch_100_v1",
        "status": "sealed", "tasks_excluded_from_nomination": True, "sealed_unix": 100.}))
    nomination = ref(nomination_path)
    summary = {"inputs": {"nomination": nomination}, "released_tasks": []}
    def task(arm, epoch):
        release_path = tmp_path / f"{arm}_{epoch}_release.json"
        release_path.write_text(json.dumps({"nomination": nomination, "selection_use": False,
                                            "released_unix": 101.}))
        release = ref(release_path)
        result_path = tmp_path / f"{arm}_{epoch}_result.json"
        result_path.write_text(json.dumps({"status": "completed", "arm": arm, "epoch": epoch,
                                          "test_locked": True, "release": release}))
        return {"arm": arm, "epoch": epoch, "selection_use": False,
                "paired_frozen_train_units": {"event": {}, "regional": {}},
                "result": ref(result_path), "release": release}
    def save():
        (folder / "summary.json").write_text(json.dumps(summary))
    summary["released_tasks"] = [task(arm, 100) for arm in LEARNING_ARMS]
    save()
    assert require_final_readouts(bundle) == nomination
    summary["selected_final"] = {"arm": LEARNING_ARMS[1], "epoch": 300}
    save()
    with pytest.raises(ValueError, match="any selected later endpoint"):
        require_final_readouts(bundle)
    summary["released_tasks"].append(task(LEARNING_ARMS[1], 300))
    save()
    assert require_final_readouts(bundle) == nomination
    del summary["released_tasks"][-1]["paired_frozen_train_units"]["event"]
    save()
    with pytest.raises(ValueError, match="event and regional"):
        require_final_readouts(bundle)


def test_figure_collisions_rejected_and_relocated_report_links_work(tmp_path):
    folder = tmp_path / "learning"; learning_bundle(folder)
    staged = tmp_path / "staged"; staged.mkdir(); captions = {}
    add_extra_figures(folder / "figure_captions.json", folder / "figures", staged, captions)
    with pytest.raises(ValueError, match="unique"):
        add_extra_figures(folder / "figure_captions.json", folder / "figures", staged, captions)
    output = tmp_path / "output"; output.mkdir(); (output / "figures").mkdir()
    for ext in ("png", "pdf"):
        (output / f"figures/learning_curve.{ext}").write_bytes((folder / f"figures/learning_curve.{ext}").read_bytes())
    (output / "report.md").write_text("# Coherence Adaptive Repair\n\n## Figure learning curve\n\n![Curve](figures/learning_curve.png)\n\n[PDF master](figures/learning_curve.pdf)\n\n## Figure index\n\n- [Figure learning curve](#figure-learning-curve)\n")
    (output / "report.pdf").write_bytes(b"report")
    compact = {"figures": {"learning_curve": {ext: ref(output / f"figures/learning_curve.{ext}")["sha256"] for ext in ("png", "pdf")}},
               "report_sha256": {name: ref(output / name)["sha256"] for name in ("report.md", "report.pdf")}}
    destination = tmp_path / "relocated" / "publication"
    copy_publication(output, destination, compact)
    assert not check_document(destination / "report.md", tmp_path, portable=True)["issues"]
    assert (destination / "evidence/evidence_index.json").is_file()
    with pytest.raises(FileExistsError): copy_publication(output, destination, compact)
    incomplete = {"figures": {"learning_curve": {"png": compact["figures"]["learning_curve"]["png"]}},
                  "report_sha256": compact["report_sha256"]}
    with pytest.raises(ValueError, match="PNG and a PDF"):
        copy_publication(output, tmp_path / "incomplete-publication", incomplete)
    (output / "figures/learning_curve.pdf").write_bytes(b"changed")
    with pytest.raises(ValueError, match="figure pairs"):
        copy_publication(output, tmp_path / "other-publication", compact)
