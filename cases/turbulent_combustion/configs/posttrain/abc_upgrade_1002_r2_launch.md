# PhyCoFlow R2 candidate launch

**Preparation status: `not_recommended` pending mature evidence.** This is an executable candidate profile, not a completed scientific recommendation. Required epoch-100/150 comparisons, finalist/second-seed evidence, and the final sibling smoke remain in progress. This status will be updated from actual results; it is not an early-stop decision for the pilots.

No formal run has been launched or queued. The command below is for the user to execute later after reviewing the final R2 report. All profile semantics match the native+endpoint ConFIG candidate currently being studied; a different sustained winner will replace this profile before final delivery. Artifact policy bounds preview history and milestone count.

```bash
source /home/wanglz/miniconda3/etc/profile.d/conda.sh
conda activate phycoflow_env
PFC_R2_ROOT=/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction
mkdir -p "$PFC_R2_ROOT/cases/turbulent_combustion/runs/Test_1002/_audit/R2_campaign/ipc"
ln -sfn "$PFC_R2_ROOT/cases/turbulent_combustion/runs/Test_1002/_audit/R2_campaign/ipc" /tmp/pfcR2-formal
CUDA_VISIBLE_DEVICES=0 TMPDIR=/tmp/pfcR2-formal KEOPS_CACHE_FOLDER="$PFC_R2_ROOT/cases/turbulent_combustion/runs/Test_1002/_audit/R2_campaign/keops_cache" python -u "$PFC_R2_ROOT/cases/turbulent_combustion/run.py" post-train --config "$PFC_R2_ROOT/cases/turbulent_combustion/configs/posttrain/abc_upgrade_1002_r2_5000ep.yaml"
```
