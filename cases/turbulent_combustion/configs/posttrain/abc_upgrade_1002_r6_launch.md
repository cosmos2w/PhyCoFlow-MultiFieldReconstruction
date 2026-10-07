# R6 beta 0.10 SOURCE profile and launch guide

**Status: `recommended_bounded_recipe` for the tested 150–200-epoch protocol; no formal run has been launched, queued or scheduled.** SOURCE beta 0.10 seed 42 completed 150 and unchanged own-state 200, and independent SOURCE seed 43 completed 150. Required mature 32-panel windows and expanded 256-panel native checks pass alongside useful stable refinement and the engineering endpoint corridor. Historical individual failures, local field/time drift and strict pressure endpoint exceptions remain explicit. The canonical 5000 profile's output-only public SOURCE 1→own 2 sibling passed physical recovery; 5000-epoch stability remains unmeasured.

## Canonical profile and evidence boundary

The tracked profile is [abc_upgrade_1002_r6_5000ep.yaml](abc_upgrade_1002_r6_5000ep.yaml). It derives from the selected 200-epoch recipe [P2_beta010_seed42_200ep.yaml](../../runs/Test_1002/R6_configs/P2_beta010_seed42_200ep.yaml), SHA-256 `758d3e94c63ebd9ca0b2fefb4b0641bb9024749c7264c0e55c495c36eae7ec61`. The selected root record is [P2_150_selection.json](../../runs/Test_1002/R6_root_reviews/P2_150_selection.json), SHA-256 `afcd710f6e3e3fd5d99b6fd4cf98c9d597679833a25788712483b54e0c0f4428`; it selects beta `0.10` at age 150 and pins checkpoint SHA-256 `69929cdc9a7ec5c09a795919c5d23d0ddde7ea8032eb870861f0423df47e8d59`.

The canonical YAML differs from the selected 200-epoch YAML in exactly two parsed values: `optimization.epochs` changes from `200` to `5000`, and `output.experiment_name` changes from `Test_1002/R6_campaign/P2_beta010_seed42` to `tc_abc_upgrade_1002_r6_beta010_5000ep`. Four leading YAML comments record provenance; they do not affect resolved values. The canonical file SHA-256 is `90db16cf7791fd4ddff2aaa2beaeadac518bac8401c57ca24db6f131e740fd85`. CUDA-hidden `load_config`, `validate_config`, and an independent recursive YAML comparison passed; KeOps reported unavailable driver access and fell back to CPU. No `RunStore`, formal run directory, queue, or training process was created.

The source identity remains the original SOURCE run `/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction/cases/turbulent_combustion/runs/tc_gl_rbf_cq_cached_kv_5000ep/20260826T212231Z_7b68c461`, checkpoint `last.pt`, SHA-256 `03f2f45b966a00850cc45c011b4cf1cd644b9408de355324fec8dc434a84a810`. The 5,000-epoch config is a requested horizon, not a completion or usefulness claim.

The root selection record reports 25- and 50-epoch native validation window ratios of `1.040663` and `1.035667`, while individual fixed native checks at epochs 100 and 150 were `1.060623` and `1.062218`. These individual 1.05 failures remain failures and visible; the final bounded recommendation uses the plan's required mature 25/50 windows and expanded paired audit, not an invented requirement that every single monitor pass. A/C descriptive family gains and endpoint-corridor evidence supported continuing the beta 0.10 recipe to its next bounded reviews. The [mature epoch-200 root review](../../runs/Test_1002/R6_root_reviews/P2_200_review.json), SHA-256 `a3efc31f349abf333a0cec8c0949d152c76928fac25fe59f9130fbb5cbfde79c`, records final 25/final 50 native validation ratios `1.032062/1.031660` and useful finite topology with diminishing A gains and mixed B. Raw TRAIN has plateaued; strict endpoint pressure ratios `1.065094/1.061399` fail 1.05 although the engineering corridor holds. Independent SOURCE seed 43 final 25/final 50 native ratios are 1.015735/1.035266. Expanded paired ratios are 0.990552 (selected 200) and 1.029957 (confirmation 150), both<=1.05; local bins and the last confirmation chronological quarter 1.063124 remain adverse descriptive evidence. The [final scientific decision](../../runs/Test_1002/R6_root_reviews/final_scientific_decision.json) scopes the recommendation to the bounded tested protocol.

The objective is a fixed scalar weighted sum: `0.1 * native_RF_loss + 0.10 * calibrated_ABC`. Keep the outer beta fixed from epoch 1; `ConFIG=false`, fidelity controller `false`, and parameter retention disabled. Require startup `training_policy.json` to report the actual beta, native coefficient `0.1`, these dispatch flags, LIVE evaluation selection, SOURCE initialization identity, and own-run resume identity truthfully. Preserve `weighted_sum`, `legacy`, `r4_exact`, `matched_native_v1`, the selected recipe's model, optimizer, sampling, precision, calibration definitions, and schedule. The existing source-only calibration runs at startup using its two fixed batches, calibration seed `700043`, first rollout seed `1700046`, then remains frozen. Never import candidate-trained calibration or initialize from trained candidate weights. Family calibration scales are fixed numerical normalization, not online outer-objective balancing. Evaluation uses LIVE weights (`model_ema_eval: false`), two-step Euler rollout, and validation data.

## Environment setup

Terminal display can be enabled for a new launch or an own-state resume with `export PHYCOFLOW_PROGRESS=batch`. This reporting-only environment override retains the immutable YAML and resume hash: tqdm shows the current batch out of 38, live losses, learning rate and timing, then leaves an average-loss summary at every epoch boundary. `PHYCOFLOW_PROGRESS=epoch` selects the compact epoch display; `PHYCOFLOW_PROGRESS=off` suppresses terminal progress. Unsetting the variable restores the profile's display settings. Segment size, plots, validation, checkpoints, optimizer, RNG and sampling are unchanged. The override must be exported before launching the public segmented module so its trainer subprocess inherits it.

Use the reviewed GPU 1 environment only when the user elects a future bounded check or formal launch. This setup writes runtime caches under `Test_1002/R6_runtime_cache` and the short temporary-directory symlink; it does not create a training output. It does not source an ignored local helper.

~~~bash
export R6_REPO='/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction'
export R6_CASE="$R6_REPO/cases/turbulent_combustion"
export R6_TEST="$R6_CASE/runs/Test_1002"
cd "$R6_REPO"
source /home/wanglz/miniconda3/etc/profile.d/conda.sh
conda activate phycoflow_env

export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PHYCOFLOW_TOPOLOGY_WORKERS=4 PHYCOFLOW_TOPOLOGY_REFERENCE_CACHE=4096
export R6_CACHE="$R6_TEST/R6_runtime_cache"
export R6_TMP_ALIAS=/tmp/phycoflow_r6_tmp
rtk proxy mkdir -p "$R6_CACHE"/{keops,mpl,xdg,tmp,pycache,cuda}
export KEOPS_CACHE_FOLDER="$R6_CACHE/keops" MPLCONFIGDIR="$R6_CACHE/mpl"
export XDG_CACHE_HOME="$R6_CACHE/xdg" PYTHONPYCACHEPREFIX="$R6_CACHE/pycache"
export CUDA_CACHE_PATH="$R6_CACHE/cuda"
if rtk proxy test -e "$R6_TMP_ALIAS" || rtk proxy test -L "$R6_TMP_ALIAS"; then
  rtk proxy test "$(rtk proxy readlink -f "$R6_TMP_ALIAS")" = "$R6_CACHE/tmp"
else
  rtk proxy ln -s "$R6_CACHE/tmp" "$R6_TMP_ALIAS"
fi
export TMPDIR="$R6_TMP_ALIAS"
export PYTHONPATH="$R6_REPO/src"

# Prepare the installed KeOps device cache before importing the training package.
rtk proxy python -c 'import os,platform,sys; from pathlib import Path; name="_".join(platform.uname()[:3])+"_p"+sys.version.split()[0]+"_CUDA_VISIBLE_DEVICES_"+os.environ["CUDA_VISIBLE_DEVICES"]; (Path(os.environ["KEOPS_CACHE_FOLDER"])/name).mkdir(parents=True,exist_ok=True)'
export R6_SOURCE="$R6_CASE/runs/tc_gl_rbf_cq_cached_kv_5000ep/20260826T212231Z_7b68c461/checkpoints/last.pt"
export R6_SOURCE_SHA256='03f2f45b966a00850cc45c011b4cf1cd644b9408de355324fec8dc434a84a810'
rtk proxy python - "$R6_SOURCE" "$R6_SOURCE_SHA256" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
hasher = hashlib.sha256()
with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1 << 20), b""):
        hasher.update(block)
assert hasher.hexdigest() == sys.argv[2], "original SOURCE checkpoint SHA-256 mismatch"
print(f"PASS original SOURCE checkpoint: {path}")
PY
~~~

## Bounded public SOURCE recovery check

The actual output-only sibling physical check passed on GPU1: SOURCE epoch 1 marker-stop at global 38 and unchanged-argv own-state resume to epoch 2/global 76, both return 0, still configured 5000. Its [attempt receipt](../../runs/Test_1002/R6_reports/public_epoch1_own2_attempts/src_ep1_own2_beta010_20261007_a/attempt_receipt.json) and [root recovery review](../../runs/Test_1002/R6_root_reviews/public_physical_recovery_root_review.json) bind actual optimizer/RNG/sampler/calibration/normalization continuity. The 214.637-second outer interval is charged once; the 210.789-second nested wrapper sum is not added. This is execution evidence, not two-epoch learning or 5000 completion. This user-facing procedure derives a sibling YAML from the canonical profile, changing only `output.experiment_name`. It then uses the public module twice with the same `--segment-epochs 1 --until-epoch 2 --stop-file` arguments. A watcher touches the marker after the epoch-1 SOURCE checkpoint, manifest, status, checkpoint-status file, and initialization policy are durable. The first wrapper must return at global step 38 before epoch 2; after removing only that marker, the same command must resume the sibling's own state and return at global step 76 under the unchanged 5,000-epoch profile.

Use a fresh timestamped sibling name under `Test_1002/R6_*`. The sibling config remains in `Test_1002/R6_configs`; its run output remains under `Test_1002/R6_public_source_5000ep_sibling_*`. The check is separate from the ordinary formal output and uses the public segmented module directly, not the internal audit observer.

~~~bash
set -eo pipefail
export R6_FORMAL_CFG="$R6_REPO/cases/turbulent_combustion/configs/posttrain/abc_upgrade_1002_r6_5000ep.yaml"
export R6_FORMAL_CFG_SHA256='90db16cf7791fd4ddff2aaa2beaeadac518bac8401c57ca24db6f131e740fd85'
export R6_CHECK_ID="$(rtk proxy date -u +%Y%m%dT%H%M%SZ)"
case "$R6_CHECK_ID" in *[!A-Za-z0-9_.-]*) printf 'Unsafe sibling ID.\n' >&2; exit 2 ;; esac
export R6_SIBLING_EXPERIMENT="Test_1002/R6_public_source_5000ep_sibling_beta010_$R6_CHECK_ID"
export R6_SIBLING_ROOT="$R6_CASE/runs/$R6_SIBLING_EXPERIMENT"
export R6_SIBLING_CFG="$R6_TEST/R6_configs/R6_public_source_5000ep_sibling_beta010_$R6_CHECK_ID.yaml"
export R6_STOP="$R6_SIBLING_ROOT/_launcher/STOP_AFTER_EPOCH1"
export R6_STOP_LOG="$R6_TEST/R6_reports/qa_tmp/$R6_CHECK_ID-epoch1-stop.log"
export R6_RESUME_LOG="$R6_TEST/R6_reports/qa_tmp/$R6_CHECK_ID-own2-resume.log"
rtk proxy test -f "$R6_FORMAL_CFG"
rtk proxy test ! -e "$R6_SIBLING_CFG"
rtk proxy test ! -e "$R6_SIBLING_ROOT"
rtk proxy test ! -e "$R6_STOP_LOG"
rtk proxy test ! -e "$R6_RESUME_LOG"
rtk proxy mkdir -p "$R6_TEST/R6_configs" "$R6_TEST/R6_reports/qa_tmp"
rtk proxy python - "$R6_FORMAL_CFG" "$R6_SIBLING_CFG" "$R6_SIBLING_EXPERIMENT" "$R6_FORMAL_CFG_SHA256" <<'PY'
from copy import deepcopy
import hashlib
import sys
from pathlib import Path

import yaml

formal_path, sibling_path = map(Path, sys.argv[1:3])
formal = yaml.safe_load(formal_path.read_text(encoding="utf-8"))
digest = hashlib.sha256(formal_path.read_bytes()).hexdigest()
assert digest == sys.argv[4], f"canonical profile SHA mismatch: {digest}"
assert formal["optimization"]["epochs"] == 5000
assert float(formal["objectives"]["coherence"]["weight"]) == 0.10
assert formal["objectives"]["data_retention"] == {"enabled": True, "weight": 0.1}
assert formal["objectives"]["parameter_retention"]["enabled"] is False
assert formal["optimization"]["gradient_balance"] == "weighted_sum"
assert formal["optimization"]["update_policy"] == "legacy"
assert formal["runtime"]["execution_mode"] == "r4_exact"
assert formal["runtime"]["random_stream_policy"] == "matched_native_v1"
assert formal["model"]["model_ema_eval"] is False
assert formal["source_run"].endswith("/tc_gl_rbf_cq_cached_kv_5000ep/20260826T212231Z_7b68c461")
assert formal["source_checkpoint"] == "last.pt"
assert formal["output"]["experiment_name"] == "tc_abc_upgrade_1002_r6_beta010_5000ep"
sibling = deepcopy(formal)
sibling["output"]["experiment_name"] = sys.argv[3]
with sibling_path.open("x", encoding="utf-8") as stream:
    yaml.safe_dump(sibling, stream, sort_keys=False)
saved = yaml.safe_load(sibling_path.read_text(encoding="utf-8"))
changed = set()

def compare(left, right, path=()):
    if isinstance(left, dict) and isinstance(right, dict):
        for key in left.keys() | right.keys():
            compare(left.get(key), right.get(key), path + (key,))
    elif left != right:
        changed.add(path)

compare(saved, formal)
assert changed == {("output", "experiment_name")}, sorted(changed)
print(f"PASS output-only sibling config: {sibling_path}")
PY
export R6_SOURCE="$R6_CASE/runs/tc_gl_rbf_cq_cached_kv_5000ep/20260826T212231Z_7b68c461/checkpoints/last.pt"
export R6_SOURCE_SHA256='03f2f45b966a00850cc45c011b4cf1cd644b9408de355324fec8dc434a84a810'
run_sibling_segment() {
  rtk proxy python -u -m phycoflow_reconstruction.training.segmented \
    --config "$R6_SIBLING_CFG" --case-dir "$R6_CASE" \
    --segment-epochs 1 --until-epoch 2 --stop-file "$R6_STOP"
}

# Start a watcher that refuses a late marker and touches it only after durable epoch-1 SOURCE state.
rtk proxy python -u - "$R6_SIBLING_ROOT" "$R6_STOP" "$R6_SOURCE_SHA256" <<'PY' &
import json
import sys
import time
from pathlib import Path

root, marker, source_sha = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
deadline = time.monotonic() + 1800
while time.monotonic() < deadline:
    manifests = list(root.glob("*/run_manifest.json"))
    if len(manifests) > 1:
        raise RuntimeError(f"multiple sibling runs appeared: {manifests}")
    if manifests:
        run = manifests[0].parent
        try:
            manifest = json.loads(manifests[0].read_text())
            status = json.loads((run / "status.json").read_text())
            checkpoint_status = json.loads((run / "evaluation/checkpoint_status.json").read_text())
            policy = json.loads((run / "training_policy.json").read_text())
        except (OSError, json.JSONDecodeError):
            time.sleep(0.02)
            continue
        step = int(status.get("global_step", -1))
        saved = int(manifest.get("last_checkpoint_step", -1))
        if step > 38 or saved > 38:
            raise RuntimeError("missed the durable epoch-1 window; refusing a late stop marker")
        ready = (
            manifest.get("source_hashes", {}).get("checkpoint") == source_sha
            and manifest.get("last_checkpoint_epoch") == 1.0
            and saved == step == 38
            and status.get("status") == "running"
            and status.get("checkpoint_epoch") == 1.0
            and (run / "checkpoints/last.pt").is_file()
            and checkpoint_status.get("global_step") == 38
            and checkpoint_status.get("training_epoch") == 1.0
            and checkpoint_status.get("last_updated") is True
            and policy.get("initialization", {}).get("source_checkpoint_sha256") == source_sha
            and policy.get("execution", {}).get("start_global_step") == 0
        )
        if ready:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch(exist_ok=False)
            print(f"Created marker after durable SOURCE epoch 1: {run}", flush=True)
            break
    time.sleep(0.02)
else:
    raise TimeoutError("did not observe durable SOURCE epoch-1 state")
PY
R6_WATCHER_PID=$!
run_sibling_segment >"$R6_STOP_LOG" 2>&1 &
R6_WRAPPER_PID=$!
wait "$R6_WRAPPER_PID"
wait "$R6_WATCHER_PID"
rtk proxy cat "$R6_STOP_LOG"
rtk proxy python - "$R6_SIBLING_ROOT" "$R6_STOP_LOG" <<'PY'
import json
import sys
from pathlib import Path

root, log = Path(sys.argv[1]), Path(sys.argv[2])
summary = json.loads((root / "_launcher/segmented.json").read_text())
runs = list(root.glob("*/run_manifest.json"))
assert len(runs) == 1, runs
run = runs[0].parent
manifest = json.loads(runs[0].read_text())
assert summary["execution_limit_epoch"] == 2
assert summary["steps_per_epoch"] == 38
assert summary["global_step"] == 38
assert summary["configured_steps"] == 5000 * 38
assert summary["completed"] is False
assert manifest["last_checkpoint_epoch"] == 1.0
assert manifest["last_checkpoint_step"] == 38
assert "Stopped at a recovery boundary" in log.read_text(encoding="utf-8")
assert (root / "_launcher/STOP_AFTER_EPOCH1").is_file()
print(f"PASS marker-stop returned before epoch 2: {run}")
PY

# Preserve epoch-1 bytes without another checkpoint copy; checkpoint saves use atomic replacement.
export R6_RUN_DIR="$(rtk proxy python - "$R6_SIBLING_ROOT" <<'PY'
import sys
from pathlib import Path
runs = list(Path(sys.argv[1]).glob("*/run_manifest.json"))
assert len(runs) == 1, runs
print(runs[0].parent)
PY
)"
export R6_EPOCH1_CHECKPOINT="$R6_RUN_DIR/checkpoints/last.pt"
export R6_EPOCH1_PROOF="$R6_RUN_DIR/checkpoints/.r6_epoch1_resume_proof.pt"
rtk proxy test ! -e "$R6_EPOCH1_PROOF"
export R6_EPOCH1_SHA256="$(rtk proxy sha256sum "$R6_EPOCH1_CHECKPOINT" | rtk proxy awk '{print $1}')"
rtk proxy ln "$R6_EPOCH1_CHECKPOINT" "$R6_EPOCH1_PROOF"
rtk proxy test "$(rtk proxy sha256sum "$R6_EPOCH1_PROOF" | rtk proxy awk '{print $1}')" = "$R6_EPOCH1_SHA256"
rtk proxy rm -- "$R6_STOP"
rtk proxy test ! -e "$R6_STOP"

# Same function and exact argv: public own-state recovery to epoch 2, still on the 5,000-epoch config.
run_sibling_segment >"$R6_RESUME_LOG" 2>&1
rtk proxy cat "$R6_RESUME_LOG"
rtk proxy python - "$R6_SIBLING_ROOT" "$R6_RUN_DIR" "$R6_EPOCH1_PROOF" "$R6_EPOCH1_SHA256" "$R6_SOURCE_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root, run, proof = map(Path, sys.argv[1:4])
proof_sha, source_sha = sys.argv[4:6]
summary = json.loads((root / "_launcher/segmented.json").read_text())
manifest = json.loads((run / "run_manifest.json").read_text())
status = json.loads((run / "status.json").read_text())
policy = json.loads((run / "training_policy.json").read_text())
assert summary["execution_limit_epoch"] == 2
assert summary["steps_per_epoch"] == 38
assert summary["global_step"] == 76
assert summary["configured_steps"] == 5000 * 38
assert summary["completed"] is False
assert manifest["last_checkpoint_epoch"] == 2.0
assert manifest["last_checkpoint_step"] == 76
assert manifest["source_hashes"]["checkpoint"] == source_sha
assert status["global_step"] == 76
assert policy["execution"]["kind"] == "own_run_resume"
assert policy["execution"]["start_global_step"] == 38
assert policy["execution"]["resume_checkpoint_sha256"] == proof_sha
h = hashlib.sha256(proof.read_bytes()).hexdigest()
assert h == proof_sha, h
print(f"PASS same-config own-state resume at epoch 2: {run}")
PY
~~~

Record the exact sibling run path, config/resolved-config SHA, epoch-1 proof and epoch-2 checkpoint hashes, policy/calibration hashes, optimizer step counters and parameter groups, finite optimizer moments, and RNG/sampler identity after the physical check. Those measurements are not available yet; the commands above are instructions, not reported results. A clean two-epoch resume confirms the recovery path only, not learning quality or suitability of the 5,000-epoch horizon.

## User-only 5,000-epoch launch, clean stop, and manual resume

The formal experiment name is `tc_abc_upgrade_1002_r6_beta010_5000ep`; its output is the ordinary case root `cases/turbulent_combustion/runs/tc_abc_upgrade_1002_r6_beta010_5000ep`, outside `Test_1002`. The public wrapper's receipt and lock live under `cases/turbulent_combustion/runs/long_jobs`. The command below is a future user action; it has not been run, and the output root does not exist as a result of this preparation.

Use `--segment-epochs 25` and at most 24 hours per allocation. Epoch 1 is a separate saved milestone, followed by a 24-epoch first interval to epoch 25. Later stop requests take effect at segment boundaries; a full 25-epoch segment is about 27 minutes of synchronized update work at the measured envelope of roughly 65 seconds per epoch, plus initialization, batch-source, monitoring, checkpoint, and finalization overhead. The full 5,000-epoch horizon projects to roughly 90+ hours, so unchanged manual resumes across allocations are expected. These update-work timings do not include all outer-process overhead.

Before any launch, repeat the canonical-config SHA and inert validation check, verify the original SOURCE checkpoint SHA, and confirm the user-selected profile remains the intended experiment. The tracked profile resolves with the expected beta `0.10`, native coefficient `0.1`, disabled parameter retention, fixed scalar update, original SOURCE identity, and 5,000 configured epochs.

~~~bash
set -eo pipefail
export R6_FORMAL_CFG="$R6_REPO/cases/turbulent_combustion/configs/posttrain/abc_upgrade_1002_r6_5000ep.yaml"
export R6_FORMAL_EXPERIMENT='tc_abc_upgrade_1002_r6_beta010_5000ep'
export R6_FORMAL_ROOT="$R6_CASE/runs/$R6_FORMAL_EXPERIMENT"
export R6_JOB_ROOT="$R6_CASE/runs/long_jobs"
export R6_STOP="$R6_JOB_ROOT/$R6_FORMAL_EXPERIMENT.STOP_AFTER_CURRENT_SEGMENT"
export R6_LAUNCHER_JSON="$R6_JOB_ROOT/$R6_FORMAL_EXPERIMENT.json"
export R6_FORMAL_CFG_SHA256='90db16cf7791fd4ddff2aaa2beaeadac518bac8401c57ca24db6f131e740fd85'
export R6_BETA='0.10'
rtk proxy mkdir -p "$R6_JOB_ROOT"
rtk proxy test -f "$R6_FORMAL_CFG"
rtk proxy test ! -e "$R6_STOP"
rtk proxy python - "$R6_FORMAL_CFG" "$R6_BETA" "$R6_FORMAL_CFG_SHA256" <<'PY'
import hashlib
import sys
import yaml

config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
digest = hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest()
assert digest == sys.argv[3], f"canonical profile SHA mismatch: {digest}"
assert config["optimization"]["epochs"] == 5000
assert float(config["objectives"]["coherence"]["weight"]) == float(sys.argv[2]) == 0.10
assert config["objectives"]["data_retention"] == {"enabled": True, "weight": 0.1}
assert config["objectives"]["parameter_retention"]["enabled"] is False
assert config["optimization"]["gradient_balance"] == "weighted_sum"
assert config["optimization"]["update_policy"] == "legacy"
assert config["runtime"]["execution_mode"] == "r4_exact"
assert config["runtime"]["random_stream_policy"] == "matched_native_v1"
assert config["model"]["model_ema_eval"] is False
assert config["source_checkpoint"] == "last.pt"
assert config["output"]["experiment_name"] == "tc_abc_upgrade_1002_r6_beta010_5000ep"
assert not config["output"]["experiment_name"].startswith("Test_1002/")
print("PASS selected beta, source profile, scalar policy, and formal output identity")
PY
# User-only: launch only after the user decides to start the formal run.
rtk proxy python -u -m phycoflow_reconstruction.training.segmented \
  --config "$R6_FORMAL_CFG" --case-dir "$R6_CASE" \
  --segment-epochs 25 --allocation-hours 24 --stop-file "$R6_STOP"
~~~

To stop cleanly, create the marker with `rtk proxy touch "$R6_STOP"`; the active segment finishes and the wrapper returns before starting the next one. Verify the exact retained run path and epoch boundary in `$R6_LAUNCHER_JSON`. After an allocation boundary or clean stop, remove only that marker and rerun the unchanged public command to resume the same config's own recovery checkpoint. Do not add `--until-epoch 5000`, a manual checkpoint override, requeue, scheduler automation, or a daemon.

After the wrapper returns at a clean boundary or allocation limit, inspect the receipt and exact retained run path before resuming. In a fresh terminal, repeat Environment setup first; then remove the stop marker if it exists and issue the same public command below. This is a manual own-state resume, not a new profile.

~~~bash
set -eo pipefail
export R6_FORMAL_CFG="$R6_REPO/cases/turbulent_combustion/configs/posttrain/abc_upgrade_1002_r6_5000ep.yaml"
export R6_FORMAL_CFG_SHA256='90db16cf7791fd4ddff2aaa2beaeadac518bac8401c57ca24db6f131e740fd85'
export R6_FORMAL_EXPERIMENT='tc_abc_upgrade_1002_r6_beta010_5000ep'
export R6_JOB_ROOT="$R6_CASE/runs/long_jobs"
export R6_STOP="$R6_JOB_ROOT/$R6_FORMAL_EXPERIMENT.STOP_AFTER_CURRENT_SEGMENT"
rtk proxy python - "$R6_FORMAL_CFG" "$R6_FORMAL_CFG_SHA256" <<'PY'
import hashlib
import sys
from pathlib import Path
p = Path(sys.argv[1])
digest = hashlib.sha256(p.read_bytes()).hexdigest()
assert digest == sys.argv[2], f"canonical profile SHA mismatch: {digest}"
print(f"PASS canonical profile identity: {p}")
PY
if rtk proxy test -e "$R6_STOP"; then
  rtk proxy rm -- "$R6_STOP"
fi
rtk proxy test ! -e "$R6_STOP"
rtk proxy test -f "$R6_FORMAL_CFG"
rtk proxy python -u -m phycoflow_reconstruction.training.segmented \
  --config "$R6_FORMAL_CFG" --case-dir "$R6_CASE" \
  --segment-epochs 25 --allocation-hours 24 --stop-file "$R6_STOP"
~~~

## Selected checkpoint and LIVE validation

The current mature continuation checkpoint is `cases/turbulent_combustion/runs/Test_1002/R6_campaign/P2_beta010_seed42/20261007T065425Z_936405fa/checkpoints/epoch_200.pt`, SHA-256 `580b8a5e93a9c8cbf20285c6deb43db07758b6bfbc30a3ee48b8b5fbdcac5c1d`; the [independent mature review](../../runs/Test_1002/R6_root_reviews/P2_200.json) pins this exact path and digest. Epoch 100/150 remain historical learning evidence; they were not replaced by a best-named checkpoint. Never substitute `best.pt`, `last.pt`, a latest glob, or an inferred path.

The validation command evaluates that retained checkpoint with LIVE weights and validation data; it does not require the formal 5,000-epoch horizon to finish and does not impose an eligible-selected-record gate before evaluation. The command is future work and was not executed during profile preparation.

~~~bash
export R6_RUN="$R6_TEST/R6_campaign/P2_beta010_seed42/20261007T065425Z_936405fa"
export R6_CHECKPOINT="$R6_RUN/checkpoints/epoch_200.pt"
export R6_CHECKPOINT_SHA256='580b8a5e93a9c8cbf20285c6deb43db07758b6bfbc30a3ee48b8b5fbdcac5c1d'
rtk proxy python - "$R6_RUN" "$R6_CHECKPOINT" "$R6_SOURCE_SHA256" "$R6_CHECKPOINT_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

run, checkpoint = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
source_sha, expected_sha = sys.argv[3:5]
manifest = json.loads((run / "run_manifest.json").read_text())
assert manifest.get("source_hashes", {}).get("checkpoint") == source_sha
assert checkpoint.is_file(), checkpoint
h = hashlib.sha256()
with checkpoint.open("rb") as stream:
    for block in iter(lambda: stream.read(1 << 20), b""):
        h.update(block)
assert h.hexdigest() == expected_sha, f"checkpoint hash mismatch: {h.hexdigest()}"
print(f"PASS retained mature checkpoint identity: {checkpoint}")
PY
rtk proxy python -u "$R6_CASE/run.py" evaluate-run \
  --run "$R6_RUN" --checkpoint "$R6_CHECKPOINT" --split validation \
  --sample-index 0 --max-samples 64 --query-points 4096 \
  --generation-steps 2 --device cuda:0 --weight-selection live \
  --report-name R6_beta010_epoch200_live_validation
~~~

Review the saved raw native TRAIN history, fixed validation endpoint fields, and active A/B/C components at each mature boundary. A plateau is acceptable; individual objectives need not fall monotonically. Sustained raw native deterioration or endpoint deterioration warrants a clean boundary stop and review of retained checkpoints. Report implementation readiness, useful stable refinement, and strict native preservation as separate conclusions. If native preservation remains unsupported, describe the recipe as an experimental native trade-off and report the failed checks.
