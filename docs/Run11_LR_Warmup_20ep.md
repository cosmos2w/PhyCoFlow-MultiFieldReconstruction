# Run 11 AdamW LR warm-up: bounded 20-epoch result

**Outcome: mixed.** The first-20-epoch TRAIN endpoint-MSE peak fell from 0.0341414 to 0.0304423 (10.83% lower), with its peak moving from epoch 1 to epoch 5. Scientific coherence improvement was retained: epoch-20 C was 0.0090536 versus 0.0091416 historically (0.96% lower). Fixed validation was inconsistent: the epoch-5 overshoot was larger, and epoch-20 MSE was higher than the original, although epochs 10/15/20 passed the budgets. This supports a reduced TRAIN error peak, not a reliable held-out fidelity improvement.

## Identity and exact configuration

Branch `codex/run11-lr-warmup-20ep` inherits **`60ce91aa53fd5079a3b79aa0348a09fc17538ab3`**. The isolated worktree is `/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction_run11w`; implementation/configuration commit **`d1338c2463efcd5078bab9408ceee9f260b437df`** was pushed before launch. The only candidate is `/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction_run11w/cases/turbulent_combustion/runs/_MultiTask/11w_lagrangian_lr_warmup_20ep/20261017T153407Z_6bb62fdf`, completed on October 17, 2026, at 11:57:50 a.m. EDT. Physical GPU **1** was explicitly authorized after the GPU-0 placement clarification; the Kolmogorov job was restored from its existing step-4400 checkpoint on GPU 0. Run 12 was paused at its step-35720 checkpoint, and Run 13 remains paused. The completed original Run 11 was read-only throughout.

The profile is `cases/turbulent_combustion/configs/posttrain/fidelity_budget/run11w_config_lagrangian_lr_warmup_20ep.yaml`. Resolved equality against Run 11 was verified except epochs, warm-up and output name. Preserved settings include internal A/B/T ConFIG, full-model training, AdamW weight decay `1e-6`/clip `1`, batch 32, 4096 queries, two Euler steps, full CO/T 32×128 H0/H1 self/mutual topology, unit family scales, seeds, 8×2 TRAIN calibration, fixed validation and +5% per-field budgets. AdamW and multipliers start fresh/zero; live weights and EMA behavior are unchanged.

```yaml
defaults: [run11_config_lagrangian_1000ep.yaml]
optimization:
  epochs: 20
  lr_warmup: {enabled: true, start_lr: 5.0e-6, updates: 76}
output: {experiment_name: _MultiTask/11w_lagrangian_lr_warmup_20ep}
```

The inherited model LR is `5e-5`; dual LR stays `0.01`. For one-based update n, LR is `5e-6 + (5e-5 - 5e-6)*(n-1)/75` through update 76, then `5e-5`. The restored global count determines n. Only the model optimizer LR changes before its existing step; moments/dual updates continue normally. There is no coherence-weight warm-up or scheduler state. Missing/disabled warm-up preserves behavior. Actual LR appears in telemetry and progress.

The physical SOURCE is `/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction/cases/turbulent_combustion/runs/tc_gl_rbf_cq_cached_kv_5000ep/20260826T212231Z_7b68c461/checkpoints/last.pt`, with existing checkpoint SHA256 `03f2f45b966a00850cc45c011b4cf1cd644b9408de355324fec8dc434a84a810`. Dataset `/home/wanglz/Desktop/src/PhyCoFlow/0_demo_TurbulentCombustion/Dataset/Merged_CH4COTU1P.h5` and original normalization statistics are reused directly. The worktree has an ignored SOURCE-directory symlink; no SOURCE or dataset files were copied or modified. The final standard SOURCE immutability check passed.

## Execution and correctness evidence

Exactly **38 + 722 = 760 AdamW updates / 20 reporting epochs** completed (24,320 samples / 3.04 nominal dataset passes). At step 38, inspection confirmed 145 AdamW states, controller 38, EMA 315038, intact RNG/reference and LR `2.72e-5`. The same child resumed at update 39 with `2.78e-5` and counters 38→39. All 76 recorded LRs match the schedule exactly; projected dual updates match the unchanged `0.01` formula within `7e-18`.

Final `last.pt` has **145 AdamW states at 760**, controller **760**, model LR `5e-5`, dual LR `0.01`, EMA **315760**, intact RNG and the original bit-identical risk reference. `best.pt`/`best_fidelity.pt` are at 570; all three pass their embedded guards. Measured totals: **760 shared reconstructions, 1520 Flow velocity calls, 3035 prediction-backward traversals, 760 AdamW steps, zero native-loss calls and zero teacher calls**. Constraint traversal is omitted when multipliers are zero. Natural exit was successful; no extra rollout, extension, second candidate or sweep occurred.

Prelaunch checks passed **78 CPU tests** in `phycoflow_env` with local `PYTHONPATH=src`: 17 new warm-up tests, 15 formal-config tests and 46 existing fidelity-policy/direction/checkpoint/history regressions. Coverage includes endpoints/plateau, disabled behavior, restore continuity, one model step, model-before-dual ordering and torn telemetry recovery. Targeted Ruff and whitespace checks passed. Test modules: `test_posttraining_lr_warmup`, `test_fidelity_budget_formal_config`, `test_fidelity_budget`, `test_fidelity_budget_config_direction`, `test_checkpointing`, `test_coherence_history`. Training-loop wall time was 1321.60 seconds; peak allocation 15.330 GB.

## Historical comparison

Comparator: original Run 11 `20261016T010804Z_e5999942` under `/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction/cases/turbulent_combustion/runs/_MultiTask/11_config_lagrangian_1000ep/`, restricted to its first 20 epochs. D is mean per-field TRAIN endpoint MSE. A/B/T are global-distribution/cross-spectrum/topology; C is their scientific sum, not a scalar optimized by ConFIG. Scalar means use finite-update weighting, including the original first epoch’s **1+37** partial rows. Every entry has 38 updates. Nested field-risk/violation/multiplier arrays are last-batch snapshots.

| Epoch | Run | D | A | B | T | C |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Run11 | 0.0341414 | 0.0091383 | 0.0013494 | 0.0047063 | 0.0151940 |
| 1 | LR warmup | 0.0302211 | 0.0095316 | 0.0010399 | 0.0054947 | 0.0160662 |
| 2 | Run11 | 0.0306121 | 0.0063419 | 0.0010946 | 0.0042345 | 0.0116711 |
| 2 | LR warmup | 0.0298126 | 0.0062208 | 0.0009939 | 0.0043197 | 0.0115344 |
| 5 | Run11 | 0.0295178 | 0.0055593 | 0.0010954 | 0.0039392 | 0.0105938 |
| 5 | LR warmup | 0.0304423 | 0.0055885 | 0.0011259 | 0.0039007 | 0.0106151 |
| 10 | Run11 | 0.0282278 | 0.0049348 | 0.0009066 | 0.0036689 | 0.0095104 |
| 10 | LR warmup | 0.0284148 | 0.0049157 | 0.0009115 | 0.0037026 | 0.0095298 |
| 20 | Run11 | 0.0285295 | 0.0047234 | 0.0009062 | 0.0035120 | 0.0091416 |
| 20 | LR warmup | 0.0282458 | 0.0046078 | 0.0008933 | 0.0035524 | 0.0090536 |

The declared TRAIN calibration mean is 0.0279311. Peak D above that mean decreases from **22.23%** to **8.99%**, with the warm-up peak delayed to epoch 5. Both runs stay below the descriptive scalar `1.05*mean(E_SOURCE)` line from epoch 6; neither reaches the calibration mean. That scalar line is not the actual per-field constraint. C20 is 59.96% below calibrated SOURCE C0=0.0226093 (historical: 59.57%).

Fixed validation uses common SOURCE MSE **0.489732**, identical sensor/query manifests, seed 2027 and two generation steps. Exact-step `metrics/topology_validation.jsonl` gives percentage MSE changes below; eligibility requires total and every field ≤+5%. Unmatched original step 1/candidate step 38 are excluded.

| Step | Run | Eligible | Total | CH4 | CO | T | U_1 | p |
|---:|---|:---:|---:|---:|---:|---:|---:|---:|
| 190 | Run11 | no | +3.24% | -2.63% | +3.52% | -0.46% | +2.62% | +5.63% |
| 190 | LR warmup | no | +12.35% | -1.55% | +5.08% | +0.85% | +10.67% | +23.04% |
| 380 | Run11 | yes | -7.33% | -3.78% | +1.68% | -0.89% | -5.39% | -16.77% |
| 380 | LR warmup | yes | -3.65% | -2.28% | +3.33% | +0.59% | -1.74% | -11.21% |
| 570 | Run11 | yes | -0.95% | +0.80% | +4.85% | +3.05% | -2.16% | -3.21% |
| 570 | LR warmup | yes | -5.86% | -3.07% | +2.42% | -0.06% | -6.70% | -10.48% |
| 760 | Run11 | yes | -5.38% | -0.91% | +3.16% | +2.35% | -4.59% | -13.19% |
| 760 | LR warmup | yes | -0.84% | -2.86% | +1.09% | +0.09% | -1.43% | -0.77% |

Matched total MSEs at steps 190/380/570/760 were historical `0.505602/0.453817/0.485057/0.463362` and candidate `0.550221/0.471841/0.461011/0.485640`. Both fail the first common panel and pass the later three. Warm-up is worse at 190/380/760 and better at 570; the step-190 p regression is +23.04% versus +5.63% historically. Epoch-20 candidate MSE remains 4.81% higher than historical Run 11 despite passing its SOURCE-relative guard. Eligibility and best-checkpoint identity are therefore distinct from error improvement.

## Artifacts, reproduction and limits

`metrics/lr_warmup_updates.jsonl` preserves all 76 updates’ actual LR, field errors/violations, multipliers, A/B/T/C, norms and clipping. Values describe the prediction **before** that step. The initial SOURCE row has C=0.0223159 and mean risk=0.0289587, matching history; it is not an optimization-induced increase. Original dense coverage exists only at updates 1–4, plus sparse anchors at 38/76, so no matched first-76 maximum/recovery curve exists.

Exact scripts, logs, `launch_record.json`, compact `comparison_final.json`, the extractor `compare_warmup_runs.py` and tables are under `/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction_run11w/cases/turbulent_combustion/runs/_lr_warmup_ops`. Logs are `run11w_20261017T153349Z.segment1.log` and `run11w_20261017T153349Z.segment2.log`. The child preserves `artifacts/step38_checkpoint_audit.json`, `step39_resume_audit.json`, `warmup_trace_audit.json`, `final_checkpoint_audit.json` and both segment instrumentation records. These normal ignored artifacts remain local; no prediction dataset or new hashing infrastructure was added.

Recorded CLI form, from the isolated worktree (the recorded scripts additionally wrap only mechanical instrumentation):

```bash
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PHYCOFLOW_TOPOLOGY_WORKERS=4
export KEOPS_CACHE_FOLDER=/home/wanglz/.cache/phycoflow_formal_11_13/keops_gpu1
export RUN11W=/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction_run11w/cases/turbulent_combustion/runs/_MultiTask/11w_lagrangian_lr_warmup_20ep/20261017T153407Z_6bb62fdf
/home/wanglz/miniconda3/envs/phycoflow_env/bin/python cases/turbulent_combustion/run.py post-train \
  --config configs/posttrain/fidelity_budget/run11w_config_lagrangian_lr_warmup_20ep.yaml --max-steps 38
/home/wanglz/miniconda3/envs/phycoflow_env/bin/python cases/turbulent_combustion/run.py post-train \
  --config configs/posttrain/fidelity_budget/run11w_config_lagrangian_lr_warmup_20ep.yaml --resume "$RUN11W" --max-steps 722
```

This is one historical comparison with noisy TRAIN batches and finite SOURCE calibration, without replicates or confidence intervals. GPU placement and segmentation differ; fixed validation retains identical draws, so its regressions reflect model changes. No scientific superiority, generative-calibration or population-fidelity guarantee follows. **Mixed:** smaller initial TRAIN overshoot and retained coherence improvement, without consistent fixed-validation fidelity gain. No further run or evaluation was started.