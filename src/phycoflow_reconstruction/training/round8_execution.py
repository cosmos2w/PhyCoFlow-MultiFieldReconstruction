"""Saved-checkpoint callbacks for the predeclared Round 8 experiment."""
from __future__ import annotations

import json
import os
import fcntl
import hashlib
import uuid
from pathlib import Path
import time

import torch

from ..evaluation.round8_campaign import file_hash, canonical_hash
from ..evaluation.round8_inference import infer_panel, score_cached_panel, validate_panel, validate_matched_panels
from .post_training import run_post_training
from .round8_runtime import Round8Runtime, VERSION as RUNTIME_VERSION
from .round8_campaign import fork_warmup, resolved_config, equal_state
from .run_store import load_project_checkpoint


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")



def publish_qualification_receipt(path, result):
    """Archive the previous receipt by exact bytes, then atomically publish.

    Duplicate publication of the same result is a no-op, including its previous
    hash. The lock covers both archival and replacement; conflicting archive
    bytes are rejected and no existing archive is replaced.
    """
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if result.get('status')!='completed': raise ValueError('only completed qualification can be published')
    lock_path=path.parent / '.qualification_receipts.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        previous=None
        if path.exists():
            original=path.read_bytes()
            existing=json.loads(original)
            comparable={k:v for k,v in existing.items() if k!='previous_qualification_sha256'}
            candidate={k:v for k,v in result.items() if k!='previous_qualification_sha256'}
            if comparable==candidate:
                return existing
            previous=hashlib.sha256(original).hexdigest()
            archive=path.with_name(f'{path.stem}.{previous}{path.suffix}')
            try:
                with archive.open('xb') as stream:
                    stream.write(original);stream.flush();os.fsync(stream.fileno())
            except FileExistsError:
                if archive.read_bytes()!=original: raise ValueError('existing qualification archive differs')
        payload={**result,'previous_qualification_sha256':previous}
        encoded=(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
        temp=path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
        try:
            with temp.open('xb') as stream:
                stream.write(encoded);stream.flush();os.fsync(stream.fileno())
            os.replace(temp,path)
        finally:
            temp.unlink(missing_ok=True)
        return payload

def require_formal_admission(campaign, *, artifacts_path, config, manifest_path):
    directory = Path(campaign) / "artifacts"
    records = {name: json.loads((directory / f"{name}.json").read_text())
               for name in ("qualified", "resource_forecast", "stage0_complete")}
    if records["qualified"].get("status") != "completed":
        raise ValueError("formal launch requires completed numerical/real-model qualification")
    if records["qualified"].get("runtime_version") != RUNTIME_VERSION:
        raise ValueError("formal launch requires qualification of the current runtime protocol")
    fitted = torch.load(artifacts_path, map_location="cpu", weights_only=True)
    if records["qualified"].get("common_stream") != Round8Runtime.stream_protocol(fitted):
        raise ValueError("formal launch requires qualification of the frozen common warmup stream")
    if records["stage0_complete"].get("status") != "completed":
        raise ValueError("formal launch requires the fixed T2000 anchoring intervention")
    forecast = records["resource_forecast"]
    if not forecast.get("fits") or forecast.get("closeout_reserved_seconds", 0) < 4 * 3600:
        raise ValueError("the measured 1000-epoch comparison does not fit with closeout reserved")
    if forecast.get("artifacts_sha256") != file_hash(artifacts_path):
        raise ValueError("resource admission is not bound to the fitted runtime artifact")
    if forecast.get("qualification_sha256") != file_hash(directory / "qualified.json"):
        raise ValueError("resource admission is not bound to the current timing qualification")
    memory = records["qualified"].get("callback_memory_qualification", {})
    properties = torch.cuda.get_device_properties(torch.device("cuda:0"))
    if (memory.get("status") != "completed" or memory.get("device_name") != properties.name
            or memory.get("device_total_bytes") != properties.total_memory
            or not memory.get("training_and_optimizer_resident") or not memory.get("second_source_model")
            or memory.get("native_sites") != 40300 or memory.get("members") != 2
            or memory.get("manifest_sha256") != file_hash(manifest_path)
            or not 0 < memory.get("peak_allocated_bytes", 0) <= memory.get("peak_reserved_bytes", 0) < memory.get("device_total_bytes", 0)):
        raise ValueError("formal launch requires qualified same-class concurrent live-training and native-J2 inference memory")
    if file_hash(memory["profile_path"]) != memory.get("profile_sha256"):
        raise ValueError("resident inference profile receipt changed")
    if config["optimization"]["epochs"] != 1000 or config["optimization"]["steps_per_epoch"] != 13:
        raise ValueError("formal horizon must remain 1000 epochs with 13 updates per epoch")
    return records


class ScheduledObserver:
    """Infer each checkpoint/population once and rescore the cached fields."""
    def __init__(self, *, campaign, manifest, arm, artifacts, native_bundle, fair_shape_artifact, device="cuda:0"):
        if arm not in {"B8", "C8"}:
            raise ValueError("only the two fixed formal arms are scheduled")
        self.campaign = Path(campaign)
        self.manifest = Path(manifest)
        self.arm = arm
        self.artifacts = artifacts
        self.native_bundle = Path(native_bundle)
        self.fair_shape_artifact = Path(fair_shape_artifact)
        self.fair_sha = file_hash(self.fair_shape_artifact)
        self.native_sha = file_hash(self.native_bundle)
        self.device = device

    def _source_baseline(self, config, source, population):
        from ..evaluation.round8_decision import _check_report
        source = Path(source)
        panel = validate_panel(source)
        manifest_sha = file_hash(self.manifest)
        if (panel['endpoint'] != 'SOURCE' or panel['population'] != population
                or panel['manifest_sha256'] != manifest_sha):
            raise ValueError('SOURCE baseline panel identity differs')
        directory = self.campaign / 'comparison' / 'SOURCE' / population
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / 'common_source_score.json'
        with (directory / '.common_score.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not output.exists():
                score_cached_panel(config, source, source, fair_shape_artifact=self.fair_shape_artifact,
                    output_path=output, alpha=0., device=self.device,
                    native_bundle=self.native_bundle if population == 'ASSESS' else None)
            report = json.loads(output.read_text())
            _check_report(report, file_hash(source), file_hash(source), population, 0.)
            scientific = canonical_hash({key: config[key] for key in ('dataset', 'coherence', 'rollout', 'observation_consistency')})
            family_hashes = {name: file_hash(spec['path']) for name, spec in config['coherence'].get('frozen_family_artifacts', {}).items()}
            if (report.get('frozen_family_artifacts_sha256') != family_hashes
                    or report.get('config_sha256') != scientific or report.get('fair_shape_artifact_sha256') != self.fair_sha
                    or [row['sample_id'] for row in report['rows']] != [row['sample_id'] for row in panel['conditions']]):
                raise ValueError('SOURCE baseline common observer differs')
            if population == 'ASSESS' and report.get('native_observer_identity', {}).get('native_bundle_sha256') != self.native_sha:
                raise ValueError('SOURCE baseline native observer differs')
        return output

    def __call__(self, run_dir, checkpoint, global_step):
        epoch = int(global_step) // 13
        if epoch not in (250, 500, 750, 1000) or global_step != epoch * 13:
            return
        import yaml
        from ..evaluation.round8_decision import freeze_train_policy, assess_endpoint, _check_report

        started = time.perf_counter()
        config = yaml.safe_load((Path(run_dir) / "resolved_config.yaml").read_text())
        base = self.campaign / "comparison" / self.arm / f"epoch_{epoch}"
        complete = base / "callback_complete.json"
        # CheckpointManager may return a rolling last.pt even when it also
        # saved an immutable milestone. Always pin the epoch-specific file.
        checkpoint = Path(run_dir) / "checkpoints" / f"epoch_{epoch:03d}.pt"
        saved_checkpoint = load_project_checkpoint(checkpoint)
        if saved_checkpoint.get("global_step") != global_step:
            raise ValueError("saved milestone step differs from callback")
        del saved_checkpoint
        if complete.exists():
            saved = json.loads(complete.read_text())
            if (saved["checkpoint_sha256"] != file_hash(checkpoint)
                    or saved["manifest_sha256"] != file_hash(self.manifest)
                    or saved["fair_sha256"] != self.fair_sha or saved["native_sha256"] != self.native_sha):
                raise ValueError("scheduled callback input identity changed")
            for name, digest in saved["files"].items():
                if file_hash(name) != digest:
                    raise ValueError("scheduled callback artifact changed: " + name)
            for path in saved["panels"].values():
                validate_panel(path)
            return
        fair = self.fair_shape_artifact
        if file_hash(fair) != self.fair_sha or file_hash(self.native_bundle) != self.native_sha:
            raise ValueError("frozen common observer artifacts changed")
        panels = {}
        scores = {}
        policy = None
        populations = ("TRAIN", "ASSESS") if epoch in (500, 1000) else ("ASSESS",)
        for population in populations:
            source = self.campaign / "comparison/SOURCE" / population / "panel.json"
            panel = base / population / "panel.json"
            if panel.exists():
                saved = validate_panel(panel)
                if saved["checkpoint_sha256"] != file_hash(checkpoint):
                    raise ValueError("cached scheduled checkpoint differs")
            else:
                panel = infer_panel(
                    config, self.manifest, base / population, population=population,
                    endpoint=self.arm, checkpoint=checkpoint, device=self.device)
            source_record, endpoint_record = validate_matched_panels(source, panel)
            if (source_record["manifest_sha256"] != file_hash(self.manifest)
                    or endpoint_record["endpoint"] != self.arm or endpoint_record["epoch"] != epoch
                    or endpoint_record["checkpoint_sha256"] != file_hash(checkpoint)):
                raise ValueError("scheduled cached panel identity differs")
            panels[population] = str(panel)
            baseline = self._source_baseline(config, source, population)
            scores[population + "_SOURCE"] = str(baseline)
            if epoch == 1000:
                if population == "TRAIN":
                    policy = freeze_train_policy(
                        config, source, panel, fair_shape_artifact=fair,
                        score_floors=self.artifacts["score_floors"],
                        output_dir=base / "policy", device=self.device, source_report_path=baseline)
                    scores["TRAIN_policy"] = str(policy)
                else:
                    scores["ASSESS"] = str(assess_endpoint(
                        config, source, panel, policy, fair_shape_artifact=fair,
                        native_bundle=self.native_bundle,
                        output_dir=base / "assessment", device=self.device, source_report_path=baseline))
            else:
                for alpha in (1.0, 0.5):
                    path = base / "scores" / f"{population}_alpha_{alpha}.json"
                    if not path.exists():
                        score_cached_panel(
                            config, source, panel, fair_shape_artifact=fair,
                            output_path=path, alpha=alpha, device=self.device,
                            interim_diagnostic=True,
                            native_bundle=self.native_bundle if population == "ASSESS" else None)
                    report = json.loads(path.read_text())
                    baseline_report = json.loads(baseline.read_text())
                    identity = _check_report(baseline_report, file_hash(source), file_hash(source), population, 0.)
                    _check_report(report, file_hash(source), file_hash(panel), population, alpha, observer=identity)
                    if population == "ASSESS" and not report.get("interim_diagnostic"):
                        raise ValueError("interim ASSESS report lacks fixed diagnostic declaration")
                    scores[f"{population}_{alpha}"] = str(path)
        receipt = {
            "schema": "round8_scheduled_callback_v1", "arm": self.arm,
            "epoch": epoch, "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_hash(checkpoint),
            "manifest_sha256": file_hash(self.manifest), "panels": panels,
            "scores": scores, "fair_sha256": self.fair_sha, "native_sha256": self.native_sha,
            "files": {**{str(path): file_hash(path) for path in base.rglob("*.json") if path != complete},
                      **{path: file_hash(path) for key, path in scores.items() if key.endswith("_SOURCE")}},
            "seconds": time.perf_counter() - started,
            "policy_selection": "TRAIN only at endpoint 1000" if epoch == 1000 else "none; fixed alpha0.5 diagnostic",
        }
        write_new(complete, receipt)
        with (Path(run_dir) / "logs/round8_assessments.jsonl").open("a") as stream:
            stream.write(json.dumps(receipt, allow_nan=False) + "\n")


def execute_training(config, *, case_dir, artifacts, phase, charge_attempt,
                     evaluation_callback=None, checkpoint_callback=None,
                     resume=None, max_steps=None, mode="warmup", resume_state_callback=None):
    if phase not in {"qualification", "formal"}:
        raise ValueError("unknown optimizer-attempt phase")
    if config["runtime"].get("plot_every_steps") != 10:
        raise ValueError("loss curves must refresh every ten completed epochs")
    runtime = Round8Runtime(artifacts, mode=mode)
    if resume is not None and phase == "formal" and mode in {"control", "candidate"} and checkpoint_callback is not None:
        # The save precedes the scheduled callback. A process failure in that
        # callback leaves last.pt at the boundary; replay the idempotent
        # observer before advancing or completing the resumed run.
        last = Path(resume) / "checkpoints/last.pt"
        saved = load_project_checkpoint(last)
        step = int(saved["global_step"])
        if step in {250 * 13, 500 * 13, 750 * 13, 1000 * 13}:
            checkpoint_callback(Path(resume), last, step)
    if evaluation_callback is None:
        def evaluation_callback(**payload):
            # Scientific assessment is performed from immutable saved endpoints
            # by ScheduledObserver. Return no legacy validation metric; those
            # scalar MSE/TEST guards are different from Round 8's member risk.
            return None
    return run_post_training(
        config, case_dir=case_dir, max_steps=max_steps, resume=resume,
        qualification_only=phase == "qualification", round8_runtime=runtime,
        update_attempt_callback=lambda step: charge_attempt(phase, arm=mode, step=step),
        evaluation_callback=evaluation_callback, checkpoint_callback=checkpoint_callback,
        resume_state_callback=resume_state_callback,
    )


def profile_resident_inference(config, *, campaign, manifest, device="cuda:0", output_dir=None):
    """Profile one existing ASSESS condition while training state stays resident."""
    dev = torch.device(device)
    if dev.type != 'cuda':
        raise ValueError('resident inference qualification requires the actual CUDA device')
    torch.cuda.synchronize(dev)
    free_before, total = torch.cuda.mem_get_info(dev)
    properties = torch.cuda.get_device_properties(dev)
    allocated_before = torch.cuda.memory_allocated(dev)
    reserved_before = torch.cuda.memory_reserved(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    started = time.perf_counter()
    path = infer_panel(config, manifest, Path(output_dir) if output_dir is not None else Path(campaign) / 'qualification' / 'resident_native_profile',
        population='ASSESS', endpoint='SOURCE', device=device, profiling_first_condition=True)
    torch.cuda.synchronize(dev)
    free_after, _ = torch.cuda.mem_get_info(dev)
    peak = torch.cuda.max_memory_reserved(dev)
    profile = json.loads(path.read_text())
    if (profile.get('schema') != 'round8_one_condition_profile_v1'
            or profile.get('status') != 'profiled' or len(profile['conditions']) != 1
            or len(profile['conditions'][0]['initial_field_sha256']) != 2):
        raise ValueError('native J2 resident qualification incomplete')
    return {'status':'completed', 'device_uuid':os.environ.get('ROUND8_PHYSICAL_GPU_UUID'),
            'training_and_optimizer_resident':True, 'second_source_model':True,
            'device_name':properties.name, 'native_sites':40300, 'members':2, 'conditions':1,
            'manifest_sha256':file_hash(manifest), 'profile_path':str(path),
            'profile_sha256':file_hash(path), 'allocated_before_bytes':allocated_before,
            'reserved_before_bytes':reserved_before, 'peak_allocated_bytes':torch.cuda.max_memory_allocated(dev),
            'peak_reserved_bytes':peak, 'device_total_bytes':properties.total_memory, 'device_free_before_bytes':free_before,
            'device_free_after_bytes':free_after, 'device_free_query_total_bytes':total, 'seconds':time.perf_counter()-started,
            'scope':'same process, live qualification trainer/optimizer and one additional SOURCE model'}


def qualify(retained_config, *, case_dir, artifacts, campaign, charge_attempt, manifest):
    """Fourteen disposable real-model attempts, including an epoch-10 resume.

    A qualification epoch has one update. These disposable states have no
    learning-efficacy interpretation and never initialize the formal warmup.
    """
    config = resolved_config(retained_config, arm="qualification")
    config["optimization"]["steps_per_epoch"] = 1
    restored = {}
    def observe(saved, live):
        for key in ("model", "optimizer", "external_policy", "rng_state", "round8_runtime"):
            if not equal_state(saved[key], live[key]):
                raise ValueError("qualification resume differs: " + key)
        restored.update(exact_loaded_state=True, keys=["model", "optimizer", "external_policy", "rng_state", "round8_runtime"])
    started = time.perf_counter()
    warmup = execute_training(
        config, case_dir=case_dir, artifacts=artifacts, phase="qualification",
        charge_attempt=charge_attempt, max_steps=8)
    memory_profile = {}
    def profile_at_boundary(run_dir, checkpoint, step):
        if step == 10:
            memory_profile.update(profile_resident_inference(retained_config, campaign=campaign, manifest=manifest,
                output_dir=Path(run_dir) / 'artifacts' / 'resident_native_profile'))
    execute_training(
        config, case_dir=case_dir, artifacts=artifacts, phase="qualification",
        charge_attempt=charge_attempt, max_steps=2, resume=warmup,
        resume_state_callback=observe, checkpoint_callback=profile_at_boundary)
    if memory_profile.get("status") != "completed":
        raise ValueError("qualification did not profile live-training native J2 inference")
    if not restored.get("exact_loaded_state"):
        raise ValueError("qualification did not verify actual loaded state")
    fork = fork_warmup(warmup, case_dir=case_dir, retained_config=retained_config,
                       qualification=True)
    segments = {}
    import yaml
    for arm, mode in (("B8", "control"), ("C8", "candidate")):
        run = Path(fork["arms"][arm]["run"])
        child_config = yaml.safe_load((run / "resolved_config.yaml").read_text())
        start = time.perf_counter()
        execute_training(
            child_config, case_dir=case_dir, artifacts=artifacts, phase="qualification",
            charge_attempt=charge_attempt, max_steps=2, resume=run, mode=mode)
        rows = [json.loads(line) for line in (run / "logs/full_updates.jsonl").read_text().splitlines()]
        measured = [float(row["update_seconds"]) for row in rows if row["step"] > 10]
        if len(measured) != 2 or load_project_checkpoint(run / "checkpoints/last.pt")["global_step"] != 12:
            raise ValueError("qualification child cursor/history differs")
        segments[arm] = {"run": str(run), "seconds": time.perf_counter() - start,
                         "update_seconds": measured, "seconds_per_update": sum(measured) / 2}
    result = {"schema": "round8_disposable_qualification_v1", "status": "completed",
              "runtime_version": RUNTIME_VERSION,
              "common_stream": Round8Runtime.stream_protocol(artifacts),
              "warmup": str(warmup), "fork": fork, "resume": restored,
              "optimizer_attempts": 14, "seconds": time.perf_counter() - started,
              "segments": segments, "callback_memory_qualification": memory_profile, "efficacy_claim": False,
              "epoch_definition": "one update per disposable qualification epoch; formal epochs retain 13 updates"}
    return publish_qualification_receipt(Path(campaign) / "artifacts/qualified.json", result)
