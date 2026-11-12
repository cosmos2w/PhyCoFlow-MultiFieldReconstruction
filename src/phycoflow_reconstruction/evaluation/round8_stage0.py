"""The fixed T2000 anchoring intervention, with no optimizer updates."""
from __future__ import annotations

import json
from pathlib import Path
import time

from .round8_campaign import file_hash, validate_t2000_receipt
from .round8_decision import freeze_train_policy, assess_endpoint
from .round8_inference import validate_matched_panels


def execute(config, *, campaign, manifest, t_receipt, fair_shape_artifact,
            native_bundle, device="cuda:0", score_floor=1e-8):
    from ..training.round8_execution import ScheduledObserver, write_new

    campaign = Path(campaign)
    receipt = validate_t2000_receipt(t_receipt)
    output = campaign / "artifacts/stage0_complete.json"
    if output.exists():
        old = json.loads(output.read_text())
        if (old["T2000_receipt_sha256"] != receipt["receipt_sha256"]
                or old["manifest_sha256"] != file_hash(manifest)):
            raise ValueError("fixed T2000 intervention identity changed")
        for path, digest in old["files"].items():
            if file_hash(path) != digest:
                raise ValueError("fixed T2000 intervention artifact changed: " + path)
        return output
    started = time.perf_counter()
    pairs = {}
    for population in ("TRAIN", "ASSESS"):
        source = campaign / "comparison/SOURCE" / population / "panel.json"
        endpoint = campaign / "comparison/T2000" / population / "panel.json"
        left, right = validate_matched_panels(source, endpoint)
        if (right["checkpoint_sha256"] != receipt["checkpoint_sha256"]
                or right["endpoint"] != "T2000" or right["epoch"] != 2000
                or left["manifest_sha256"] != file_hash(manifest)):
            raise ValueError("anchoring panels differ from the completed fixed endpoint")
        pairs[population] = (source, endpoint)
    observer = ScheduledObserver(
        campaign=campaign, manifest=manifest, arm="B8", artifacts={},
        native_bundle=native_bundle, fair_shape_artifact=fair_shape_artifact, device=device)
    source_train, endpoint_train = pairs["TRAIN"]
    baseline = observer._source_baseline(config, source_train, "TRAIN")
    policy = freeze_train_policy(
        config, source_train, endpoint_train, fair_shape_artifact=fair_shape_artifact,
        score_floors=dict.fromkeys(("state", "scale", "shape"), score_floor),
        output_dir=campaign / "comparison/T2000/policy", device=device,
        source_report_path=baseline)
    # SOURCE ASSESS scoring and the endpoint's ASSESS evaluation both follow
    # the frozen TRAIN nomination. No ASSESS result selects the coefficient.
    source_assess, endpoint_assess = pairs["ASSESS"]
    native_baseline = observer._source_baseline(config, source_assess, "ASSESS")
    assessment = assess_endpoint(
        config, source_assess, endpoint_assess, policy,
        fair_shape_artifact=fair_shape_artifact, native_bundle=native_bundle,
        output_dir=campaign / "comparison/T2000/assessment", device=device,
        source_report_path=native_baseline)
    result = json.loads(assessment.read_text())
    frozen = json.loads(policy.read_text())
    files = [Path(t_receipt), Path(manifest), policy, assessment,
             baseline, native_baseline, *[p for pair in pairs.values() for p in pair]]
    record = {
        "schema":"round8_stage0_complete_v1", "status":"completed", "optimizer_updates":0,
        "T2000_receipt_sha256":receipt["receipt_sha256"],
        "checkpoint_sha256":receipt["checkpoint_sha256"],
        "manifest_sha256":file_hash(manifest), "policy":str(policy), "assessment":str(assessment),
        "alpha":frozen["alpha"], "eligible_families":frozen["families"],
        "assessment_eligible":result["eligible"], "seconds":time.perf_counter() - started,
        "files":{str(p):file_hash(p) for p in files},
        "disposition":"anchored T2000 passes scientific guards; benchmark still required" if result["eligible"] else "T2000 not deployable under frozen procedure; continue requested framework comparison",
    }
    write_new(output, record)
    return output
