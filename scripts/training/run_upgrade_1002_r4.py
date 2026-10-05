#!/usr/bin/env python
"""Independent R4 pilot envelope and exposure ledger; TEST and formal forbidden."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "cases/turbulent_combustion"
AUDIT = CASE / "runs/Test_1002/_audit/R4_campaign"
LEDGER = AUDIT / "campaign_ledger.json"
SOURCE = CASE / "runs/tc_gl_rbf_cq_cached_kv_5000ep/20260826T212231Z_7b68c461"
SOURCE_SHA = "03f2f45b966a00850cc45c011b4cf1cd644b9408de355324fec8dc434a84a810"
STEPS = 38
GPU_UUID = "GPU-233fcd85-5c6a-6212-3f44-655252afec70"
sys.path.insert(0, str(ROOT / "src"))


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024 * 1024),b""):
            h.update(b)
    return h.hexdigest()


def validate(config, *, max_steps=None, resume_epoch=0):
    if config.get("stage") != "post_training" or config.get("case") != "turbulent_combustion":
        raise ValueError("R4 only accepts turbulent-combustion post-training")
    output = Path(config["output"]["experiment_name"])
    if output.is_absolute() or ".." in output.parts or len(output.parts) != 2 or output.parts[0] != "Test_1002" or not output.parts[1].startswith("R4_"):
        raise ValueError("R4 outputs must be Test_1002/R4_* direct children")
    if any(word in output.name.lower() for word in ("5000ep", "formal")):
        raise ValueError("formal names and runs are forbidden")
    epochs = config["optimization"]["epochs"]
    if not isinstance(epochs, int) or isinstance(epochs,bool) or not 1 <= epochs < 250:
        raise ValueError("every R4 lineage remains strictly below 250 epochs")
    opt = config["optimization"]
    if opt["batch_size"] != 32 or not math.isclose(opt["train_fraction"], .15, abs_tol=1e-12, rel_tol=0):
        raise ValueError("R4 exposure requires batch32/TRAINfraction0.15")
    if any(k in opt for k in ("steps_per_epoch", "training_subset")) or opt.get("sampling") == "full_pass":
        raise ValueError("R4 historical epoch cannot be overridden")
    if config["runtime"]["device"] != "cuda:0" or os.getenv("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("R4 requires physical GPU0 via CUDA_VISIBLE_DEVICES=0 and logical cuda:0")
    ev, ck = config["evaluation"], config.get("checkpointing",{})
    if ev.get("split", "validation") != "validation" or ev.get("preview",{}).get("split","validation") != "validation" or ck.get("selection_split","validation") != "validation" or "test" in str(ck.get("selection_metric","")).lower():
        raise ValueError("TEST is locked")
    if Path(config["source_run"]).resolve() != SOURCE.resolve() or config["source_checkpoint"] != "last.pt" or config["model"].get("model_ema_eval") is not False:
        raise ValueError("R4 requires original LIVE SOURCE last.pt")
    if sha(SOURCE / "checkpoints/last.pt") != SOURCE_SHA:
        raise ValueError("SOURCE checksum changed")
    additional_steps = epochs * STEPS - round(resume_epoch * STEPS)
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        additional_steps = min(additional_steps, max_steps)
    if additional_steps <= 0:
        raise ValueError("resume has no new exposure")
    return {"epochs": epochs,"resume_epoch":resume_epoch,"reserved_exposures":additional_steps / STEPS,"output":str(CASE / "runs" / output),"SOURCE_sha256":SOURCE_SHA,"TEST_locked":True}


def load_ledger():
    return json.loads(LEDGER.read_text()) if LEDGER.is_file() else {"schema":"R4_campaign_ledger_v1","epoch_exposure_cap":620,"gpu0_wall_seconds_cap":86400,"offline_gpu_seconds_cap":3600,"entries":[]}


def save_ledger(ledger):
    tmp = LEDGER.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger,indent=2)+"\n")
    tmp.replace(LEDGER)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True,type=Path)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--resume",type=Path)
    p.add_argument("--max-steps",type=int)
    p.add_argument("--role",required=True,choices=("profile","simple_reference","adaptive","continuation","confirmation","targeted_repair","sibling_smoke"))
    args=p.parse_args()
    from phycoflow_reconstruction.cli import _load_case_config
    from phycoflow_reconstruction.training.run_store import load_project_checkpoint
    c = _load_case_config(args.config.resolve(), CASE, "turbulent_combustion", [])
    resume_epoch=0
    if args.resume:
        args.resume.resolve().relative_to((CASE / "runs/Test_1002").resolve())
        if not args.resume.is_dir() or not args.resume.resolve().parent.name.startswith("R4_"):
            raise ValueError("only an unchanged R4 lineage can resume")
        resume_epoch=load_project_checkpoint(args.resume / "checkpoints/last.pt")["global_step"] / STEPS
    contract=validate(c,max_steps=args.max_steps,resume_epoch=resume_epoch)
    command=[sys.executable,"-u",str(CASE / "run.py"),"post-train","--config",str(args.config.resolve())]
    if args.resume: command += ["--resume",str(args.resume.resolve())]
    if args.max_steps: command += ["--max-steps",str(args.max_steps)]
    print(json.dumps({**contract,"execute":args.execute,"role":args.role,"command":command},indent=2),flush=True)
    if not args.execute:return 0
    AUDIT.mkdir(parents=True,exist_ok=True)
    cache=CASE / "runs/Test_1002/R4_runtime_cache"
    for name in ("keops","mpl","xdg","tmp","pycache"):(cache / name).mkdir(parents=True,exist_ok=True)
    alias=Path("/tmp/phycoflow_r4_tmp")
    if not alias.exists():alias.symlink_to(cache / "tmp",target_is_directory=True)
    if alias.resolve() != (cache / "tmp").resolve():raise ValueError("R4 IPC alias points outside its cache")
    env={**os.environ,"KEOPS_CACHE_FOLDER":str(cache / "keops"),"MPLCONFIGDIR":str(cache / "mpl"),"XDG_CACHE_HOME":str(cache / "xdg"),"TMPDIR":str(alias),"PYTHONPYCACHEPREFIX":str(cache / "pycache")}
    with (AUDIT / "campaign.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger=load_ledger()
        reserved=sum(e.get("actual_exposures",e["reserved_exposures"]) for e in ledger["entries"])
        elapsed=sum(e.get("gpu0_wall_seconds",0) for e in ledger["entries"])
        if reserved + contract["reserved_exposures"] > 620:raise ValueError("620 epoch exposure cap exceeded")
        if elapsed >= 86400:raise ValueError("24 GPU0 hours exhausted")
        if args.role == "profile" and sum(e.get("actual_exposures",e["reserved_exposures"]) for e in ledger["entries"] if e["role"] == "profile") + contract["reserved_exposures"] > 10:raise ValueError("initial scratch exposure cap10 exceeded")
        uuid=subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader","-i","0"],text=True).strip()
        if uuid != GPU_UUID:raise ValueError("physical GPU0 identity changed")
        compute=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid","--format=csv,noheader"],text=True)
        occupied=[line for line in compute.splitlines() if GPU_UUID in line]
        if occupied:raise ValueError("physical GPU0 already has a compute owner: " + "; ".join(occupied))
        entry={**contract,"role":args.role,"started_utc":utc(),"status":"running","command":command,"config_sha256":sha(args.config)}
        ledger["entries"].append(entry);save_ledger(ledger)
        root=Path(contract["output"]);before=set(root.iterdir()) if root.is_dir() else set()
        t=time.monotonic();child=subprocess.Popen(command,cwd=ROOT,env=env)
        try:
            while child.poll() is None:
                if time.monotonic()-t >= 86400-elapsed:
                    child.send_signal(signal.SIGINT)
                    break
                time.sleep(10)
            code=child.wait()
        except KeyboardInterrupt:
            child.send_signal(signal.SIGINT);code=child.wait()
        finally:
            entry["gpu0_wall_seconds"]=time.monotonic()-t;entry["ended_utc"]=utc();entry["status"]="finished" if child.poll()==0 else "stopped_or_failed"
            children=[x for x in root.iterdir() if x.is_dir() and (x not in before or args.resume)] if root.is_dir() else []
            if len(children)==1:
                run=children[0];entry["run"]=str(run)
                updates=run / "metrics/coherence_updates.jsonl"
                if updates.is_file():
                    last=max((json.loads(l).get("step",0) for l in updates.read_text().splitlines() if l.strip()),default=0)
                    entry["actual_exposures"]=max(0,last/STEPS-resume_epoch)
            save_ledger(ledger)
        return code


if __name__=="__main__":raise SystemExit(main())
