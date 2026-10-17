# Run 11: bounded AdamW LR warm-up check

The isolated research branch `codex/run11-lr-warmup-20ep` inherits `60ce91aa53fd5079a3b79aa0348a09fc17538ab3`. Its worktree is `/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction_run11w`. The candidate is one fresh 20-reporting-epoch run, segmented into 38 and 722 updates, with no extension. Per the user's placement clarification it uses physical GPU 1; the separate Kolmogorov job runs on GPU 0, Run 12 is paused at its step-35720 checkpoint, and Run 13 remains paused. The completed original Run 11 and its artifacts are read-only.

## Configuration and meaning

The profile is `cases/turbulent_combustion/configs/posttrain/fidelity_budget/run11w_config_lagrangian_lr_warmup_20ep.yaml`, inheriting the original Run 11 profile. Fully resolved configuration equality was checked against the original run: only `optimization.epochs`, the optional `optimization.lr_warmup`, and `output.experiment_name` differ. In particular, internal A/B/T ConFIG, full-model scope, AdamW base LR `5e-5`, weight decay `1e-6`, clip `1`, batch 32, 4096 fixed queries, two Euler steps, unit family scales, full CO/T 32×128 H0/H1 self/mutual topology, seeds, TRAIN calibration 8 batches ×2 draws, fixed validation and +5% per-field budgets remain unchanged. The Lagrangian dual LR stays `0.01`, with fresh zero multipliers.

```yaml
optimization:
  epochs: 20
  lr: 5.0e-5
  lr_warmup: {enabled: true, start_lr: 5.0e-6, updates: 76}
output: {experiment_name: _MultiTask/11w_lagrangian_lr_warmup_20ep}
```

For one-based model update n, the applied LR is `5e-6 + (5e-5 - 5e-6) * (n-1)/75` through update 76, then exactly `5e-5`. Thus update 1 uses `5e-6`, update 38 uses `2.72e-5`, update 39 uses `2.78e-5`, and updates 76/77 use `5e-5`. The trainer derives n from the restored global cursor and sets only the model optimizer's LR before the existing update. There is no scheduler state and no coherence-weight or dual-LR warm-up. Missing or disabled `lr_warmup` leaves optimizer LR unchanged. Actual LR is recorded in epoch history, terminal progress, checkpoint progress and final status.

The physical SOURCE is the original `/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction/cases/turbulent_combustion/runs/tc_gl_rbf_cq_cached_kv_5000ep/20260826T212231Z_7b68c461/checkpoints/last.pt`; the dataset remains `/home/wanglz/Desktop/src/PhyCoFlow/0_demo_TurbulentCombustion/Dataset/Merged_CH4COTU1P.h5`. A local ignored SOURCE-directory symlink lets the inherited recipe resolve to that same physical SOURCE. No SOURCE or dataset files are copied or modified. Candidate checkpoints and evidence reside in the isolated worktree.

## Verification and recorded execution

Prelaunch checks passed 17 focused warm-up tests, 15 unchanged formal-configuration tests and 46 existing fidelity-policy/direction/checkpoint/history regressions: 78 distinct CPU tests in `phycoflow_env`, with the worktree's `PYTHONPATH=src`. Tests cover LR endpoints, plateau, disabled behavior, global-count continuation, AdamW/controller restoration, one model step and model-before-dual ordering, and trimming uncommitted/torn warm-up telemetry after recovery. Targeted Ruff and `git diff --check` passed. No additional scientific candidate or sweep was launched by these tests.

The code/configuration commit will be pushed before the fresh candidate starts. Execution evidence and the historical comparison will replace this paragraph after the single candidate has stopped naturally at update 760; no scientific result is asserted by this preparation note.

Recorded command form, from the isolated worktree (the same exact profile is used for both segments):

```bash
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PHYCOFLOW_TOPOLOGY_WORKERS=4
export KEOPS_CACHE_FOLDER=/home/wanglz/.cache/phycoflow_formal_11_13/keops_gpu1
/home/wanglz/miniconda3/envs/phycoflow_env/bin/python cases/turbulent_combustion/run.py post-train \
  --config configs/posttrain/fidelity_budget/run11w_config_lagrangian_lr_warmup_20ep.yaml --max-steps 38
# Resume that child's run directory, without a fresh SOURCE initialization:
/home/wanglz/miniconda3/envs/phycoflow_env/bin/python cases/turbulent_combustion/run.py post-train \
  --config configs/posttrain/fidelity_budget/run11w_config_lagrangian_lr_warmup_20ep.yaml \
  --resume "$RUN11W" --max-steps 722
```

`metrics/lr_warmup_updates.jsonl` preserves the first 76 updates' actual LR, per-field endpoint risk and normalized violation, multipliers before/after, scientific A/B/T/C, norms and clipping. Risk and coherence entries describe the shared prediction **before** that update; the first row is therefore the unmodified SOURCE prediction, not an optimization-induced overshoot. Ordinary execution retains one shared differentiable reconstruction and one AdamW step, without native training-loss replay, teacher forwards or extra ordinary rollouts.

## Comparison protocol and limits

The historical comparator is `/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction/cases/turbulent_combustion/runs/_MultiTask/11_config_lagrangian_1000ep/20261016T010804Z_e5999942`, using only its existing first 20 epochs. Scalar epoch means are weighted by their finite update counts; its first epoch combines the original 1+37 partial rows. Nested field-risk/violation/multiplier entries are final-batch snapshots and are never presented as epoch means. Direct fixed-panel comparisons use common steps 190/380/570/760 and the independent SOURCE baseline; the historical step-1 panel and candidate's forced step-38 panel are different observation points.

The original first-76-update series is incomplete: only updates 1–4 plus epoch-end anchors are recoverable. The candidate's dense transient maxima cannot be treated as matched historical maxima. Calibration TRAIN risks, stochastic minibatch risks and independent validation SOURCE metrics are distinct quantities. One historical comparison cannot establish statistical superiority, guaranteed fidelity, or general efficacy. The pre-update SOURCE coherence value is reported separately from any error increase after optimization begins.
