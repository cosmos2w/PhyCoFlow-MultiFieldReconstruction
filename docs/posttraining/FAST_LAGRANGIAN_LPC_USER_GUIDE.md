# Fast Lagrangian and exploratory ordered refinement

Use a completed, immutable, original pretrained SOURCE. These turbulent-combustion profiles retain the existing CH4/CO/T/U_1/p descriptors and two-Euler reconstruction contract. The endpoint-policy interface supports deterministic and generative adapters; this study's learning evidence is one generative SOURCE/seed. See [the measured report](../../FAST_LAGRANGIAN_LPC_REPORT.md) before choosing a profile.

| Profile | Configuration below `cases/turbulent_combustion/configs/posttrain/fidelity_budget/` | Evidence |
|---|---|---|
| Efficient scalar candidate | `fast_lagrangian_scalar.yaml` | P3, dense T family, dense endpoint policy; lower cost with mixed 100-epoch history stability |
| ConFIG comparison | `fast_lagrangian_config.yaml` | P3 with fresh ConFIG each update; periodic reuse is unsupported |
| Exploratory ordered preference | `lpc_ordered.yaml` | T/T0 primary; exact SOURCE A/B bounds and +5% endpoint bounds; changed objective |

All defaults are 100 reporting epochs. A reporting epoch here is 38 batches of 32, not a full TRAIN dataset pass. Evaluation always runs the full existing descriptors; training P3 samples complete filtrations with the established weights. The randomized-family profile is retained in the study evidence but is not another portable default because of its late routine-panel B regression.

## Environment and new storage

Run from your repository checkout. Activate the environment containing the repository's tested dependencies; this machine calls it `phycoflow_env`. Supply your own absolute SOURCE path. The commands below prepare a separate user study; they do not launch anything until the explicit foreground training command.

```bash
conda activate phycoflow_env
cd /absolute/path/to/PhyCoFlow-MultiFieldReconstruction
export LPC_SOURCE=/absolute/path/to/original_pretrained_source_run
export LPC_PROFILE=fast_lagrangian_scalar
export LPC_EPOCHS=100

# Confirm the intended /data2 filesystem before creating any payload.
mountpoint -q /data2 || { echo '/data2 is not mounted'; return 1 2>/dev/null || exit 1; }
LPC_PARENT="/data2/$USER/PhyCoFlow-MultiFieldReconstruction/experiments"
mkdir -p "$LPC_PARENT"
export LPC_PAYLOAD
LPC_PAYLOAD=$(mktemp -d "$LPC_PARENT/UserLPC_XXXXXXXX")
mkdir -p "$LPC_PAYLOAD"/{runs,caches/keops,caches/matplotlib,caches/pycache,tmp,evaluation}
export LPC_ID
LPC_ID=$(basename "$LPC_PAYLOAD")
mkdir -p cases/turbulent_combustion/runs/_LPCUser
LPC_LINK="cases/turbulent_combustion/runs/_LPCUser/$LPC_ID"
if test -e "$LPC_LINK" || test -L "$LPC_LINK"; then
  echo 'Refusing to replace an existing run path'
  return 1 2>/dev/null || exit 1
fi
ln -s "$LPC_PAYLOAD/runs" "$LPC_LINK"
export LPC_EXPERIMENT="_LPCUser/$LPC_ID/$LPC_PROFILE"
export LPC_CONFIG="configs/posttrain/fidelity_budget/$LPC_PROFILE.yaml"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PHYCOFLOW_TOPOLOGY_WORKERS=4
export TMPDIR="$LPC_PAYLOAD/tmp" PYTHONPYCACHEPREFIX="$LPC_PAYLOAD/caches/pycache"
export KEOPS_CACHE_FOLDER="$LPC_PAYLOAD/caches/keops" MPLCONFIGDIR="$LPC_PAYLOAD/caches/matplotlib"
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv
```

Use an idle physical device and leave other jobs alone. `CUDA_VISIBLE_DEVICES=0` makes physical GPU 0 logical `cuda:0`. The standard public trainer does not apply this study's admission lock or bounded campaign ledger. The existing bounded campaign wrapper remains the operator for its original runs; the portable commands operate your separate study.

This small shell function invokes the existing case CLI and gives multiprocessing only a short `/tmp` socket directory; numerical temporaries and caches stay in the mounted payload. It is the same bootstrap used by the measured campaign, rather than a separate trainer.

```bash
pcase() {
  python -c 'import multiprocessing,os,runpy,shutil,sys,tempfile
short_tmp=tempfile.mkdtemp(prefix="lpc-",dir="/tmp")
multiprocessing.current_process()._config["tempdir"]=short_tmp
script=sys.argv[1]; sys.path.insert(0,os.path.dirname(os.path.abspath(script))); sys.argv=sys.argv[1:]
try: runpy.run_path(script,run_name="__main__")
finally: shutil.rmtree(short_tmp,ignore_errors=True)' cases/turbulent_combustion/run.py "$@"
}
```

## Launch, stop and own-state resume

The command runs one foreground child from your original SOURCE. Its printed run directory identifies the child; save that exact path as `LPC_OWN_RUN`. Keep these variables or save the exact command with your study receipt. Do not invoke the dataset `validate` command as a configuration-only check: its structural probes can read split boundaries.

```bash
pcase post-train --config "$LPC_CONFIG" \
  --override "source_run=$LPC_SOURCE" \
  --override "optimization.epochs=$LPC_EPOCHS" \
  --override "output.experiment_name=$LPC_EXPERIMENT" \
  --override "runtime.keops_cache_folder=$LPC_PAYLOAD/caches/keops"
```

To stop this foreground process, press **Ctrl-C** in its terminal. Recovery uses the last atomic rolling checkpoint, which may precede the signal by almost five reporting epochs with the ordinary profile. Discarded work must be counted. The ordered study's real signal test used one-epoch checkpoints and verified restoration; it does not promise a bitwise-identical CUDA/AdamW trajectory.

```bash
export LPC_OWN_RUN=/absolute/path/to/the/printed/child_run
pcase post-train --config "$LPC_CONFIG" \
  --override "source_run=$LPC_SOURCE" \
  --override "optimization.epochs=$LPC_EPOCHS" \
  --override "output.experiment_name=$LPC_EXPERIMENT" \
  --override "runtime.keops_cache_folder=$LPC_PAYLOAD/caches/keops" \
  --resume "$LPC_OWN_RUN"
```

Resume requires the same source, resolved configuration, implementation and horizon; it restores the child's own optimizer/duals/RNG/cursors. It does not initialize an independent arm. For an explicitly chosen future 1,000-epoch study, set `LPC_EPOCHS=1000` **before a fresh launch** in a new payload. That is a separate user-operated study, with the unchanged first-100-epoch workload/warm-up, not an automatic extension of a completed 100-epoch campaign. No such long command was executed or queued here.

## Full-descriptor validation

The portable evaluator writes immutable numerical JSON only. It compares the original SOURCE and the selected child under the same post-training raw-weight/two-Euler configuration, uses uniform validation samples in exact groups of 32, disables temporal derivatives, and executes all six topology filtrations. B is the mean of B(32) group scores, not one B(128). Run both selected and final checks and retain complete training histories. There is no TEST option.

```bash
python scripts/posttraining/evaluate_full_abt.py --run "$LPC_OWN_RUN" \
  --checkpoint best --max-samples 128 --device cuda:0 \
  --output-dir "$LPC_PAYLOAD/evaluation"
python scripts/posttraining/evaluate_full_abt.py --run "$LPC_OWN_RUN" \
  --checkpoint last --max-samples 128 --device cuda:0 \
  --output-dir "$LPC_PAYLOAD/evaluation"
```

The portable evaluator refuses historical `_LPCAccel` runs so their numerical-stage binding cannot be bypassed. For the completed study, use the frozen-stage campaign command, substituting the actual run and arm from the report; it performs a locked device check and charges evaluation time. No training successor is queued.

```bash
export LPC_CAMPAIGN=/data2/wanglz/PhyCoFlow-MultiFieldReconstruction/experiments/LPCAccel_20261022_01a16674
python scripts/posttraining/fast_lagrangian_campaign.py --root "$LPC_CAMPAIGN" status
python scripts/posttraining/fast_lagrangian_campaign.py --root "$LPC_CAMPAIGN" evaluate G1-S \
  --run "$LPC_CAMPAIGN/runs/G1-S/20261023T125652Z_482d2d98" \
  --checkpoint last --max-samples 128 --gpu 0 --execute
```
