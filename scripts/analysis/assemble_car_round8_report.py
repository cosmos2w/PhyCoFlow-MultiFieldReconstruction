#!/usr/bin/env python3
"""Assemble the final Round 8 report from completed campaign evidence.

The coordinator supplies the final decision, costs and completed learning
bundle. This wrapper preserves the existing cached-report renderer and never
runs a model, an observer or a learning/continuation decision.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from scripts.analysis import report_car_round8 as cached_report
from scripts.analysis.car_round8_cached_summary import checked_ref, ref

SECTIONS = (
    ("SpatialShape_Summary_v3", "section.md"),
    ("J3_Position_Action_Readout_v1", "section.md"),
    ("Engineering_Qualification_Readout_v1", "section.md"),
    ("Engineering_Verification_Readout_v1", "section.md"),
    ("Executor_Retention_Audit_v1", "note.md"),
    ("State_Scale_Direct_VALIDATION100_v1", "section.md"),
)
LEARNING_ARMS = ("MemberMSE_BaseShape", "MemberMSE_SpatialShape")


def locator(path, campaign):
    return Path(os.path.relpath(Path(path).resolve(strict=True), campaign)).as_posix()


def provenance(path, campaign):
    return {"campaign_relative_path": locator(path, campaign), "sha256": ref(path)["sha256"]}


def ordinary(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"ordinary input file required: {path}")
    return path


def verify_bundle(folder, section_name, *, learning=False):
    """Verify the completed index and indexed section/figure files."""
    folder = Path(folder).resolve(strict=True)
    index_path, section = ordinary(folder / "evidence_index.json"), ordinary(folder / section_name)
    index = json.loads(index_path.read_text())
    if learning and (index.get("schema") != "car_round8_learning_summary_v1"
                     or index.get("status") != "completed" or index.get("test_locked") is not True
                     or index.get("additional_inference") is not False):
        raise ValueError("completed locked Round 8 learning bundle required")
    entries = index.get("files_sha256", {})
    required_histories = {f"recordings/{arm}/{name}" for arm in LEARNING_ARMS
                          for name in ("history.jsonl", "recording_definitions.json")}
    if learning and ({section_name, "figure_captions.json", "summary.json"} | required_histories) - entries.keys():
        raise ValueError("learning bundle index is missing required outputs")
    for name, expected in entries.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("bundle index path must remain inside its directory")
        path = ordinary(folder / relative)
        if folder not in path.resolve().parents or ref(path)["sha256"] != expected:
            raise ValueError(f"bundle input hash mismatch: {name}")
    manifest = folder / "figure_captions.json"
    if manifest.exists(): ordinary(manifest)
    return {"folder": folder, "index": index, "index_path": index_path, "section": section,
            "manifest": manifest if manifest.exists() else None}


def localize_links(text, source, campaign):
    """Keep raw DATA2 paths visible as campaign-relative provenance, not URLs."""
    def replace(match):
        label, target = match.groups()
        if target.startswith(("http://", "https://", "#")):
            return f"{label} (`{target}`)"
        path, _, fragment = target.partition("#")
        resolved = (source.parent / path).resolve(strict=True)
        relative = os.path.relpath(resolved, campaign).replace(os.sep, "/")
        return f"{label} (`{relative}{'#' + fragment if fragment else ''}`)"
    return re.sub(r"(?<!!)\[([^\]]+)\]\(([^\)]+)\)", replace, text)


def require_final_readouts(bundle):
    """A review bundle becomes final only after its reserved tasks are released."""
    summary = json.loads((bundle["folder"] / "summary.json").read_text())
    nomination_ref = summary.get("inputs", {}).get("nomination")
    if not isinstance(nomination_ref, dict):
        raise ValueError("final learning report requires the sealed epoch-100 nomination")
    nomination = json.loads(checked_ref(nomination_ref).read_text())
    if (nomination.get("schema") != "car_round8_learning_review_epoch_100_v1"
            or nomination.get("status") != "sealed"
            or nomination.get("tasks_excluded_from_nomination") is not True):
        raise ValueError("final learning report requires the sealed epoch-100 nomination")
    required = {(arm, 100) for arm in LEARNING_ARMS}
    final = summary.get("selected_final")
    if final is not None:
        required.add((final["arm"], final["epoch"]))
    tasks = {(entry["arm"], entry["epoch"]): entry for entry in summary.get("released_tasks", [])}
    if not required.issubset(tasks):
        raise ValueError("final learning report requires released tasks for both matched children and any selected later endpoint")
    for key in required:
        entry = tasks[key]
        if (entry.get("selection_use") is not False
                or not {"event", "regional"}.issubset(entry.get("paired_frozen_train_units", {}))):
            raise ValueError("final event and regional readouts must remain outside nomination")
        result = json.loads(checked_ref(entry["result"]).read_text())
        release = json.loads(checked_ref(entry["release"]).read_text())
        if (result.get("status") != "completed" or (result.get("arm"), result.get("epoch")) != key
                or result.get("test_locked") is not True or result.get("release") != entry["release"]
                or release.get("nomination") != nomination_ref or release.get("selection_use") is not False
                or release.get("released_unix", 0) < nomination.get("sealed_unix", float("inf"))):
            raise ValueError("final task result must bind its completed endpoint and post-nomination release")
    return nomination_ref


def dated_section(name, text):
    """Remove only dated provisional language; retain the measured evidence."""
    if name == "completed_learning":
        return text.replace(
            "The matched package contrast uses within-condition SpatialShape minus BaseShape differences; its uncertainty does not use overlapping separate SOURCE intervals. Field ratios are ratios of pooled member MSE, and margins are child minus 1.10 times same-panel SOURCE in squared dataset units.",
            "Package intervals pair SpatialShape and BaseShape within frames. Field ratios use pooled member MSE; margins equal child − 1.10 × matched SOURCE in squared dataset units.")
    if name == "SpatialShape_Summary_v3":
        return text.replace("so a learning comparison would attribute its complete package.",
                            "so the matched learning comparison attributes its complete package.")
    if name == "Engineering_Qualification_Readout_v1":
        return text.replace(" The SOURCE/C8 model decision awaits the admitted learning comparison.", "")
    if name == "Engineering_Verification_Readout_v1":
        return text.replace(
            "The actual final report still needs its own full-page scientific/visual review, portable-link validation and clean-export check.",
            "Verification of this report's pages, figures, portable links and clean export is recorded separately in its compact evidence index.")
    if name == "Executor_Retention_Audit_v1":
        start = text.index("## Matched historical measurement and changed-objective timings")
        end = text.index("\nThe completed Round 8 cached VALIDATION100 position-only readout", start)
        excerpt = text[start:end].strip()
        excerpt = excerpt.removeprefix("## Matched historical measurement and changed-objective timings\n\n")
        excerpt = re.sub(r"(?m)^## ", "### ", excerpt)
        return ("## Executor retention and historical qualification\n\n"
                "The following completed audit records the earlier measurements and qualified mechanism. "
                "The final decision and any later matched benchmark are reported separately.\n\n" + excerpt)
    if name == "State_Scale_Direct_VALIDATION100_v1":
        return re.sub(r" Supply `--relation-source-readout`.*?rather than recomputing them\.", "", text)
    return text


def lead_report(markdown, decision):
    title, separator, body = markdown.partition("\n\n")
    if not separator or not title.startswith("# Coherence Adaptive Repair"):
        raise ValueError("cached report title differs")
    body = body.replace("Round 8 engineering and any new learning are identified separately below.",
                        "Round 8 engineering and the matched learning evidence are reported separately below.")
    body = body.replace("## Scope and remaining evidence", "## Scope and diagnostic cost")
    return title + "\n\n" + decision.strip() + "\n\n## Historical cached reassessment\n\n" + body


def copy_publication(output, destination, compact_index):
    """Copy only selected ordinary report files; all paths in Markdown stay relative."""
    output = Path(output)
    destination = Path(os.path.abspath(destination))
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"publication already exists: {destination}")
    if destination.resolve() != destination:
        raise ValueError("publication must not use a symlinked parent")
    reports = {name: ordinary(output / name) for name in ("report.md", "report.pdf")}
    if not compact_index["figures"] or any(set(pair) != {"png", "pdf"} for pair in compact_index["figures"].values()):
        raise ValueError("each selected figure requires a PNG and a PDF master")
    expected = {f"{name}.{ext}": digest for name, pair in compact_index["figures"].items() for ext, digest in pair.items()}
    actual = {path.name: ordinary(path) for path in (output / "figures").iterdir()}
    if set(expected) != set(actual) or any(ref(actual[name])["sha256"] != digest for name, digest in expected.items()):
        raise ValueError("selected figure pairs differ from the evidence index")
    if any(ref(path)["sha256"] != compact_index["report_sha256"][name] for name, path in reports.items()):
        raise ValueError("selected report differs from the evidence index")
    destination.mkdir(parents=True, exist_ok=False)
    figures = destination / "figures"; figures.mkdir()
    for name, path in reports.items(): shutil.copyfile(path, destination / name)
    for name, path in sorted(actual.items()): shutil.copyfile(path, figures / name)
    evidence = destination / "evidence"; evidence.mkdir()
    (evidence / "evidence_index.json").write_text(json.dumps(compact_index, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--learning-bundle", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True, help="coordinator-authored final decision and lead")
    parser.add_argument("--cost-section", type=Path, required=True, help="coordinator-authored final cost and provenance")
    parser.add_argument("--executor-result", type=Path, help="optional completed later benchmark section")
    parser.add_argument("--output", type=Path, required=True, help="new campaign reports directory")
    parser.add_argument("--publication", type=Path, help="new ordinary-file selected publication directory")
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    output = args.output.resolve()
    if (campaign / "reports").resolve() not in output.parents or output.exists():
        raise ValueError("output must be a new directory under campaign reports")
    if args.publication:
        destination = Path(os.path.abspath(args.publication))
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("selected publication directory already exists")
        if destination.resolve() != destination or destination == output or output in destination.parents:
            raise ValueError("publication must be a separate ordinary-file directory")
    decision_path, cost_path = ordinary(args.decision), ordinary(args.cost_section)
    decision = localize_links(decision_path.read_text(), decision_path, campaign).strip()
    cost = localize_links(cost_path.read_text(), cost_path, campaign).strip()
    if not decision or not cost or re.search(r"(?m)^# ", decision) or "![" in decision or "![" in cost:
        raise ValueError("nonempty decision and cost prose without a top-level title or embedded figures required")
    learning = verify_bundle(args.learning_bundle, "section.md", learning=True)
    require_final_readouts(learning)
    if (campaign / "assessment").resolve() not in learning["folder"].parents:
        raise ValueError("learning bundle must be under the campaign assessment tree")
    bundles = [(name, verify_bundle(campaign / "assessment" / name, section)) for name, section in SECTIONS]
    bundles.append(("completed_learning", learning))
    if args.executor_result:
        path = ordinary(args.executor_result)
        optional_index = path.parent / "evidence_index.json"
        bundles.append(("completed_executor_result", verify_bundle(path.parent, path.name) if optional_index.is_file()
                        else {"folder": path.parent.resolve(), "index": {}, "index_path": None,
                              "section": path, "manifest": None}))
    inputs = {"decision": provenance(decision_path, campaign), "cost_section": provenance(cost_path, campaign)}
    stage = output.parent / (output.name + "_Build_Inputs")
    stage.mkdir(parents=True, exist_ok=False); figure_source = stage / "figures"; figure_source.mkdir()
    captions = {}; figure_sources = {}; sections = []
    for name, bundle in bundles:
        source = bundle["section"]
        text = dated_section(name, source.read_text())
        text = re.sub(r"(?m)^# (.+)$", r"## \1", text)
        sections.append(localize_links(text, source, campaign).strip())
        inputs[name] = {"section": provenance(source, campaign),
                        "index": provenance(bundle["index_path"], campaign) if bundle["index_path"] else None}
        if bundle["manifest"]:
            additions = cached_report.add_extra_figures(bundle["manifest"], bundle["folder"] / "figures", figure_source, captions)
            figure_sources.update(additions)
    recording_location = locator(learning["folder"] / "recordings", campaign)
    sections += [f"The complete saved learning histories, coupling journals and ordered recording definitions remain in `{recording_location}`. Individual recording files and hashes are identified in the compact evidence index.",
                 cost, "Raw artifact locators in this report are relative to the DATA2-owned Round 8 campaign. "
                 "The compact evidence index identifies the immutable input hashes; native histories remain in that campaign."]
    section_path, manifest_path = stage / "extra_sections.md", stage / "figure_captions.json"
    section_path.write_text("\n\n".join(sections) + "\n")
    manifest_path.write_text(json.dumps(captions, indent=2, allow_nan=False) + "\n")
    cached = campaign / "assessment" / "Cached_Contract_Reassessment_v1"
    support = campaign / "assessment" / "Support_State_Rho_VALIDATION100_v1/result/readout.json"
    core = {"summary": cached / "JointAnchor500_VALIDATION100.json", "train_summary": cached / "JointAnchor500_TRAIN800.json",
            "earlier_summary": cached / "JointAnchor100_VALIDATION100.json", "support": support,
            "strata": cached / "SOURCE_reference_strata.json"}
    for name, path in core.items(): inputs[name] = provenance(ordinary(path), campaign)
    for name, relative in (("production_source_parity", "operations/Production_Source_Parity_v1.json"),
                           ("observation_tolerance_test", "operations/tests/Observation_Tolerance_Readout_v1/receipt.json")):
        path = campaign / relative
        if path.is_file(): inputs[name] = provenance(ordinary(path), campaign)
    command = ["report_car_round8.py", "--summary", str(core["summary"]), "--train-summary", str(core["train_summary"]),
        "--earlier-summary", str(core["earlier_summary"]), "--support", str(core["support"]), "--strata", str(core["strata"]),
        "--extra-sections", str(section_path), "--extra-figure-manifest", str(manifest_path),
        "--extra-figure-source", str(figure_source), "--output", str(output)]
    old_render, old_argv = cached_report.render, sys.argv
    def render_with_decision(render_output, figure_captions):
        path = render_output / "report.md"
        path.write_text(lead_report(path.read_text(), decision))
        return old_render(render_output, figure_captions)
    try:
        cached_report.render = render_with_decision; sys.argv = command
        cached_report.main()
    finally:
        cached_report.render, sys.argv = old_render, old_argv
    generated = json.loads((output / "evidence_index.json").read_text())
    recordings = {name: {"campaign_relative_path": locator(learning["folder"] / name, campaign), "sha256": digest}
                  for name, digest in learning["index"]["files_sha256"].items() if name.startswith("recordings/")}
    compact = {"schema": "car_round8_final_report_assembly_v1", "status": "rendered_pending_visual_inspection",
        "test_locked": True, "inputs": inputs, "complete_training_recordings": recordings,
        "extra_figure_sources": {name: {ext: {"campaign_relative_path": locator(item["path"], campaign), "sha256": item["sha256"]}
                                        for ext, item in pair.items()} for name, pair in figure_sources.items()},
        "generator_sha256": ref(Path(__file__))["sha256"],
        "cached_generator_sha256": ref(ROOT / "scripts/analysis/report_car_round8.py")["sha256"],
        "figures": {name: {ext: ref(output / "figures" / f"{name}.{ext}")["sha256"] for ext in ("png", "pdf")}
                    for name in generated["figures"]},
        "report_sha256": {name: ref(output / name)["sha256"] for name in ("report.md", "report.pdf")},
        "pages": generated["pages"], "additional_inference": False, "additional_observer_scoring": False,
        "presentation_changes": ["Coordinator decision placed at the lead", "Historical links converted to local provenance locators",
            "Dated prospective executor and pre-learning decision statements omitted; measured historical audit retained",
            "Matched-learning introduction condensed to avoid an orphan line; paired definitions and raw bundle preserved"]}
    evidence = output / "evidence"; evidence.mkdir()
    (evidence / "evidence_index.json").write_text(json.dumps(compact, indent=2, allow_nan=False) + "\n")
    if args.publication:
        copy_publication(output, args.publication, compact)
    print(json.dumps({"status": compact["status"], "output": str(output),
                      "publication": str(args.publication) if args.publication else None,
                      "pages": compact["pages"], "figures": len(compact["figures"])}))


if __name__ == "__main__":
    main()
