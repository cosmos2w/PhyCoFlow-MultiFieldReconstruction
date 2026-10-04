# PhyCoFlow R2 candidate launch

**Final readiness: `not_recommended`.** The controls completed 100 epochs, the three primary corrected arms and second-seed confirmation completed 150 epochs, and the two-epoch formal sibling smoke passed. The locked primary ConFIG checkpoint at epoch 130 failed the reserved validation pressure gate: pressure MSE increased 8.712% against the frozen SOURCE, above the 5% allowance. Its eight matched native draws were all worse than SOURCE. The second seed also showed native validation degradation at epoch 150. These results do not support a 5,000-epoch scientific recommendation.

No formal run has been launched or queued. The command below is supplied for user review with the validated executable candidate. It starts from the original SOURCE checkpoint, not the selected pilot checkpoint. Thirty-two CPU/static checks and the actual two-epoch sibling establish configuration and software validity; they do not establish long-horizon scientific safety. The single locked TEST evaluation was completed without recipe adaptation or reselection. See [the final R2 report](../../../../MODEL_UPGRADE_1002_R2.md) for mature checkpoint, per-field, native-risk, topology and runtime evidence. Artifact policy bounds preview history and milestone count.

```bash
source /home/wanglz/miniconda3/etc/profile.d/conda.sh
conda activate phycoflow_env
PFC_R2_ROOT=/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction
mkdir -p "$PFC_R2_ROOT/cases/turbulent_combustion/runs/Test_1002/_audit/R2_campaign/ipc"
ln -sfn "$PFC_R2_ROOT/cases/turbulent_combustion/runs/Test_1002/_audit/R2_campaign/ipc" /tmp/pfcR2-formal
CUDA_VISIBLE_DEVICES=0 TMPDIR=/tmp/pfcR2-formal KEOPS_CACHE_FOLDER="$PFC_R2_ROOT/cases/turbulent_combustion/runs/Test_1002/_audit/R2_campaign/keops_cache" python -u "$PFC_R2_ROOT/cases/turbulent_combustion/run.py" post-train --config "$PFC_R2_ROOT/cases/turbulent_combustion/configs/posttrain/abc_upgrade_1002_r2_5000ep.yaml"
```
