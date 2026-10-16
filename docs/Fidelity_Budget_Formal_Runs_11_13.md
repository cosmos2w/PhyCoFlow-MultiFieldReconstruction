# Internal ConFIG with external fidelity budgets: Runs 11–13

This continuation starts from `8d69dee556387c2087cb9f478bde72b4dab1dd06` on `codex/fidelity-budget-posttraining`, which contains the required flexible-loss feature `096237ffb7c4a5c0ad90d869bdd9831cf0de105d`. The research branch is `codex/fidelity-budget-formal-11-13`. The task is to verify the composition and sustained execution of three matched 1,000-epoch runs; final scientific ranking and broad validation remain user-owned.

## Mathematical composition

The opt-in selector `optimization.multitask.coherence_aggregation: {method: torchjd_config}` requires the existing `family_scalarization: {method: none}`, `outer_aggregation: null`, an external fidelity policy, and all three coherence families. It preserves the previous scalar, hierarchical and joint routes. The trainer explicitly orders the calibrated losses as A=`global_distribution`, B=`cross_spectrum`, T=`topology` and extracts three separate parameter-gradient rows from the same differentiable reconstruction:

\[
g_k=\nabla_\theta(s_k w_k L_k),\quad G_C=[g_A^T;g_B^T;g_T^T],\quad
d_C=\operatorname{TorchJD\ ConFIG}(G_C),\quad c_0=\max(C^0_{\mathrm{scientific}},10^{-8}).
\]

The scientific value is still \(C_{\mathrm{scientific}}=A+B+T\). It is not a scalar surrogate whose gradient equals ConFIG. The new direction route therefore reports scientific coherence and direction diagnostics separately, leaves `total` null, and records `total_semantics: parameter_direction_without_scalar_objective`. Checkpoint selection uses independent fixed validation metrics, rather than this absent scalar.

| Run | Policy | Model update direction before global clipping |
| --- | --- | --- |
| 11 | `lagrangian` | \(d_C/c_0+\sum_f\lambda_f\nabla h_f\) |
| 12 | `rectify` | \(d_C/c_0\) if all batch constraints hold; otherwise \(\nabla h_{\arg\max h}\) |
| 13 | `extended_log_barrier` | \(d_C/c_0+\sum_f b'_t(h_f)\nabla h_f\) |

The existing SOURCE-relative per-field constraints, Cooper 1.0.1 multiplier projection/update, rectification tie rule, published barrier and checkpoint state are preserved. Lagrangian multipliers advance only after a successful model step and use pre-update detached violations. No new state is needed for stateless ConFIG. All arms use +5% relative allowance, zero absolute allowance and denominator floor `1e-8`; the floor never raises the acceptance bound.

This composition is no longer a one-backward algorithm. Each ordinary update has one shared reconstruction; a coherence update extracts three family gradients plus at most one combined constraint-penalty gradient. Rectification repair skips all three family-gradient evaluations and the ConFIG invocation, extracting only the selected field gradient. Each update clips globally once and takes one AdamW step. There is no native `training_loss()` replay, SOURCE teacher, extra risk rollout or per-field constraint Jacobian. Family forwards remain shared for scientific reporting.

TorchJD 0.17.1 uses `torch.linalg.pinv` with PyTorch's default tolerance. At 5,461,817 FP32 coordinates, default `rtol=max(m,n)*eps` is about 0.6511. A measured full-width analytic example with rows beginning `[1,0,0]`, `[-0.8,0.6,0]`, `[0,0,1]` produced directional products `[-2.98e-8,3.28e-8,1]` in FP32, despite a feasible common descent direction. Applying the same TorchJD aggregator to the same gradients in FP64 produced `[0.27272728,0.27272729,0.27272728]`, with tolerance about `1.21e-9`. The new route therefore lifts its existing gradient matrix to FP64 for ConFIG linear algebra, then uses the established parameter-gradient assignment and clipping. It does not implement another ConFIG algorithm or change legacy aggregation precision. Diagnostics include source/aggregation dtypes, row names, norms/cosines, directional products, matrix rank, ConFIG and combined direction norms, clip coefficient and traversal counts. Zero/nonfinite directions fail explicitly without sum fallback; a rank-deficient input with a valid nonzero direction is recorded.

ConFIG directional products describe its coherence vector before constraint composition and AdamW preconditioning/momentum. They do not guarantee that each scientific family decreases after a finite optimizer step. Likewise, finite barrier pressure, projected dual updates and CRPO-inspired switching on stochastic TRAIN risks do not enforce the held-out validation bounds. Endpoint MSE alone does not preserve the original Flow vector-field training objective or certify generative calibration. The measured validation regressions below must be assessed separately from numerical execution.

## Actual Run 00 protocol and SOURCE

The authoritative reference is `cases/turbulent_combustion/runs/_MultiTask/00_plain_sum_1000ep/20261015T023430Z_40d8a6e5`, independently inspected as completed at 38,000 updates / 1,000 epochs, including restricted checkpoint loading. Its original SOURCE is `cases/turbulent_combustion/runs/tc_gl_rbf_cq_cached_kv_5000ep/20260826T212231Z_7b68c461/checkpoints/last.pt`, a completed 315,000-update Flow model. Run 00's post-trained checkpoint is never used to initialize these arms. Source hashes continue to come from existing provenance machinery.

The three formal profiles preserve Run 00's inherited model, CH4/CO/T/U_1/p field order, normalization, observations, full-model scope, AdamW LR `5e-5`, weight decay `1e-6`, clip `1`, batch/coherence batch `32`, train fraction `0.15`, 38 updates/reporting epoch, two Euler steps, observation consistency, 4,096 fixed-shared queries and query seed `100045`. Runtime seed is `42`; evaluation seed is `2027`. A/B/T weights and family scales remain one (`family_balance.mode: none`). Full topology uses CO/T, raster 32×128, H0/H1, sublevel/superlevel, 32 projections, self and mutual CO–T with one mutual line. The historical fixed 32-validation-snapshot `topology_with_fidelity` panel retains total and every-field +5% bounds and its five-epoch cadence. Evaluation resets the generation seed to `2027` at every panel; changing panel scores reflect updated model weights, rather than newly redrawn rollout noise. This validation baseline is separate from the TRAIN calibration reference. SOURCE and evaluation use live weights while EMA state advances and is saved.

Run 00 is a protocol reference, not a method-only ablation: it optimized native RF plus a sum direction. The three new arms disable native retention and differ only in their declared external policy. Required operational changes are zero recurring family-gradient diagnostics, disabled native/reconstruction previews, and history-plot interval 25→1000. Historical preview cadence keys remain inert in the resolved configuration; fixed-panel validation/checkpointing still runs every five epochs plus the ordinary initial/final boundaries. Inherited legacy gradient-scale keys remain present but unused. No descriptors, targets, source weights or trainable architecture were changed.

## Fixed TRAIN reference and SOURCE-noise diagnostic

Before launch, a no-update diagnostic used the actual full scientific geometry and SOURCE. It measured cumulative 4- and 8-batch references with two generative draws per batch, then evaluated 16 separately seeded TRAIN batches using one risk-only reconstruction each. No calibration draw was reused for these diagnostic trials. Independent batch-selection seed was `704204`; calibration seed was `700103`, with existing rollout seed formula `seed+1000003+batch_index*2+draw`, and independent rollout formula `seed+2000003+trial_index`. The diagnostic created no optimizer and performed zero model updates, native losses or teacher forwards. Its local [source_noise.json](../cases/turbulent_combustion/runs/_MultiTask/11_13_preparation/source_noise.json) contains full observations and settings.

| Field | SOURCE E0, 4 batches | SOURCE E0, selected 8 batches | Selected +5% bound |
| --- | ---: | ---: | ---: |
| CH4 | 0.00962798 | 0.00984443 | 0.01033665 |
| CO | 0.05783514 | 0.05885196 | 0.06179456 |
| T | 0.01637168 | 0.01645091 | 0.01727346 |
| U_1 | 0.04374214 | 0.04276262 | 0.04490076 |
| p | 0.01165468 | 0.01174564 | 0.01233292 |

`C0_scientific` changed from `0.02302901` to `0.02260933`. Field reference drift reached 2.25%, material relative to a 5% allowance, while adding four calibration batches cost about eight seconds. The declared formal reference therefore uses one shared increase to **8 TRAIN batches ×2 draws**, identically across all arms. No budget, dual learning rate, barrier schedule or switching tolerance was tuned. SOURCE repair would fire on 9/16 diagnostic batches with the four-batch reference and 10/16 with eight; this selection was for reference precision rather than a favorable repair rate. Mean field violations remained negative; this is noisy batch switching, rather than evidence of forgetting or almost-all repair. No smoothing estimator was added.

Reference uncertainty remains substantial: with the two draws grouped by data batch, estimated relative standard errors at eight batches were roughly CH4 2.77%, CO 1.78%, T 1.38%, U_1 1.75%, p 2.80%. These are descriptive estimates from eight batches, not population-confidence guarantees. The diagnostic took 16.22 seconds for eight full-coherence calibration batches and 3.12 seconds for the 16 independent risk-only rollouts, with peak CUDA allocation 4.73 GB on physical GPU 0. Two wrapper-only serialization/console issues were corrected; the final complete JSON is retained, and none of those invocations performed an optimizer update.

## Reproduce and operate the same children

Tracked profiles are `configs/posttrain/fidelity_budget/run11_config_lagrangian_1000ep.yaml`, `run12_config_rectify_1000ep.yaml` and `run13_config_log_barrier_1000ep.yaml`, each inheriting `_formal_1000ep_common.yaml`. All declare 1,000 epochs = 38,000 model updates. Policy defaults remain dual LR `0.01` with zero multipliers, maximum normalized batch violation for rectification, and barrier `t_initial=1`, `growth_factor=1.1`, interval `100` optimizer updates, cap `100`.

Run from the repository root with `phycoflow_env`. Bind physical GPUs 0/1/2 to Runs 11/12/13, respectively; each masked process uses internal `runtime.device=cuda:0`. Use independent KeOps caches and tmux sessions, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, and `PHYCOFLOW_TOPOLOGY_WORKERS=4`, preserving unrelated CPU/GPU jobs. Existing rolling checkpoint storage is about 473 MiB per completed reference run; no 200-file milestone schedule is introduced.

```bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PHYCOFLOW_TOPOLOGY_WORKERS=4 KEOPS_CACHE_FOLDER=/home/wanglz/.cache/phycoflow_formal_11_13/keops_gpu0
/home/wanglz/miniconda3/envs/phycoflow_env/bin/python cases/turbulent_combustion/run.py post-train \
  --config configs/posttrain/fidelity_budget/run11_config_lagrangian_1000ep.yaml \
  --max-steps 760
```

Use the corresponding profile/GPU/cache for Runs 12 and 13. The actual launch sequence ended the same child after one mechanical full-topology update and resumed for 759 additional updates; its configured horizon remains 1,000. Once at update 760/epoch 20, resume that exact child with the same profile and `--max-steps 1520` to reach update 2280/epoch 60. Do not restart from SOURCE or reset controller/AdamW/EMA:

```bash
# Example: resume the existing Run 11 child, with the same environment as above.
/home/wanglz/miniconda3/envs/phycoflow_env/bin/python cases/turbulent_combustion/run.py post-train \
  --config configs/posttrain/fidelity_budget/run11_config_lagrangian_1000ep.yaml \
  --resume /home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction/cases/turbulent_combustion/runs/_MultiTask/11_config_lagrangian_1000ep/20261016T010804Z_e5999942 --max-steps 1520

# After the substantive readiness assessment, continue that same child to 38,000.
/home/wanglz/miniconda3/envs/phycoflow_env/bin/python cases/turbulent_combustion/run.py post-train \
  --config configs/posttrain/fidelity_budget/run11_config_lagrangian_1000ep.yaml \
  --resume /home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction/cases/turbulent_combustion/runs/_MultiTask/11_config_lagrangian_1000ep/20261016T010804Z_e5999942
```

Inspect actual processes, GPU identity and step counters rather than treating a session name or `running` status file as liveness. Monitoring records epochs 1/5/10/20 and approximately 40/60, reviews every five-epoch validation, and can extend the same jobs to at most 100 observed epochs if warranted. A real numerical or sustained scientific failure stops the affected arm with evidence, rather than retuning it or silently falling back. Healthy children continue detached toward 1,000; their final completion must later be established from terminal status and checkpoint counters.

The barrier schedule is update-based: at 20 epochs/760 completed updates, the next `t` is `1.1^7≈1.9487`; at 60/2280 it is `1.1^22≈8.1403`; at 100/3800 it is `1.1^38≈37.4043`. The first update using the cap is 4901 (about epoch 129), and `t` stays 100 through 38,000 updates. Tensor checks cover the join, infeasible values and high-t plateau; observing the first 60 epochs does not empirically validate that late regime.

## Measured checks and live run record

Prelaunch focused suites passed 215 tests, plus three genuine fixture-model composition/recovery integration tests and six additional late-schedule tensor cases: 224 distinct tests. They cover analytic policy algebra, separate gradient ordering, FP64 pseudoinverse input, repair traversal omission, no scalar-gradient mislabeling, shared reconstruction, native/teacher omission, ordinary state recovery, old joint/hierarchical behavior, topology and numerical failures. Repository Ruff and `git diff --check` passed. The full scientific SOURCE diagnostic above is separate from optimizer-step evidence.

The new composition integration fixture trains its tiny SOURCE for one epoch, then declares two post-training epochs for Lagrangian (two one-update segments) and one for each other policy (one update). These genuine cubical H0/H1 self/mutual tests are distinct from the full scientific formal children. The SOURCE-noise diagnostic has zero optimizer updates. No disposable 20/60/100-epoch campaign was launched.

Focused test commands, run before formal launch in `phycoflow_env` with the established GPU-0 cache and four CPU threads, were:

```bash
python -m pytest tests/test_fidelity_budget.py tests/test_fidelity_budget_config_direction.py \
  tests/test_fidelity_budget_formal_config.py tests/test_fidelity_budget_config.py \
  tests/test_fidelity_budget_integration.py tests/test_multitask.py tests/test_multitask_joint.py \
  tests/test_multitask_config.py tests/test_multitask_integration.py \
  tests/test_coherence_calibration.py tests/test_coherence_history.py tests/test_checkpointing.py \
  tests/test_stable_gradients.py tests/test_topology_persistence.py -x
# 215 passed in 41.35 s, before six additional late-schedule tensor cases.
python -m pytest tests/test_fidelity_budget_composition_integration.py -x
# 3 passed in 65.22 s.
python -m pytest tests/test_fidelity_budget_config_direction.py -x
# 14 passed in 4.86 s: eight overlap the first command, six new late-schedule cases.
```

Implementation/configuration commit `b8d85f691fea7dd3389de8a018c7893a3be4851d` was pushed before any formal child started. All three children were created at `2026-10-16T01:08:04Z` and declare 1,000 epochs / 38,000 updates. Their original SOURCE checkpoint SHA256 from the existing manifests is `03f2f45b966a00850cc45c011b4cf1cd644b9408de355324fec8dc434a84a810`.

| Run | Child under `cases/turbulent_combustion/runs/_MultiTask/` | Physical GPU / tmux session |
| --- | --- | --- |
| 11 | `11_config_lagrangian_1000ep/20261016T010804Z_e5999942` | 0 / `phycoflow_run11_gpu0` |
| 12 | `12_config_rectify_1000ep/20261016T010804Z_21515e36` | 1 / `phycoflow_run12_gpu1` |
| 13 | `13_config_log_barrier_1000ep/20261016T010804Z_4a8a3190` | 2 / `phycoflow_run13_gpu2` |

Physical device identities are NVIDIA RTX 6000 Ada Generation, 49,140 MiB each: GPU 0 `GPU-233fcd85-5c6a-6212-3f44-655252afec70`, GPU 1 `GPU-3ceda40c-fd5c-4b88-6c47-b3301711571e`, GPU 2 `GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39`. Every masked process uses internal CUDA device 0.

Each first segment exited zero after one genuine full-topology update (`integration_truncated`, source immutability verified). All had one shared differentiable `[32,4096,5]` reconstruction, two actual Flow velocity calls, no native loss or teacher calls, and one AdamW step. Runs 11/13 aggregated rank-three `3×5461817` matrices in FP64 with positive family directional products; their backward traversals were three/ four respectively (zero initial Lagrangian multipliers omit its constraint traversal). Run 12 repaired CH4 with one constraint traversal and zero family traversals; after resume, its natural improve step at global update 4 used ConFIG. No policy branch was forced for this evidence.

| Run | First ordinary forward/backward/step seconds | Policy gradient/step seconds | Peak allocated GB | Pre-clip norm / clip coefficient |
| --- | ---: | ---: | ---: | --- |
| 11 | 2.860 | 1.620 | 15.012 | 35.725 / 0.027991 |
| 12 | 1.648 | 0.390 | 14.968 | 18.574 / 0.053837 |
| 13 | 3.061 | 1.842 | 15.274 | 76.504 / 0.013071 |

These are synchronized first-update timings with all three arms on the workstation, not steady-state throughput. Segment training-loop wall times including boundary evaluation/checkpointing were 10.515, 8.912 and 10.752 seconds; setup and compilation are additional. Restricted checkpoint inspection confirmed all 145 AdamW states at step 1, external-policy completed updates 1, EMA updates 315001 from SOURCE 315000, matching saved references and unchanged SOURCE hashes. First instrumentation is preserved in each child's `artifacts/first_update_instrumentation.json`.

Each same child resumed for 759 additional updates to epoch 20 and then 1,520 additional updates to epoch 60; no new child or SOURCE initialization was created. The first resumed policy/reference state matches the prior saved state and AdamW begins at the preceding completed step. All three closed segments ended naturally with `integration_truncated`, meaning the requested segment ended rather than the 1,000-epoch campaign completed. The completed 20/40/60 observations are below. The same children subsequently resumed without `--max-steps` toward their original 38,000-update horizons.

## Observed readiness window

Values below are raw epoch means for the unchanged A/B/T scientific objectives on TRAIN. The epoch-1 mean combines its 1+37 partial rows. The panel column identifies the actual independent fixed-panel attempt: epoch 1 uses the mechanical step-1 panel, while later scheduled panels use their stated step. Eligibility requires the total and every field to satisfy their validation bounds; these observations are not a method ranking.

| Run | Epoch / TRAIN updates | A | B | T | C scientific | Panel step / eligible |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 11 | 1 / 38 | 0.009138 | 0.001349 | 0.004706 | 0.015194 | 1 / no |
| 11 | 5 / 190 | 0.005559 | 0.001095 | 0.003939 | 0.010594 | 190 / no |
| 11 | 10 / 380 | 0.004935 | 0.000907 | 0.003669 | 0.009510 | 380 / yes |
| 11 | 20 / 760 | 0.004723 | 0.000906 | 0.003512 | 0.009142 | 760 / yes |
| 11 | 40 / 1520 | 0.004168 | 0.000891 | 0.003374 | 0.008434 | 1520 / no |
| 11 | 60 / 2280 | 0.004071 | 0.000855 | 0.003234 | 0.008161 | 2280 / no |
| 12 | 1 / 38 | 0.011494 | 0.001406 | 0.005964 | 0.018864 | 1 / yes |
| 12 | 5 / 190 | 0.008030 | 0.001114 | 0.004691 | 0.013835 | 190 / no |
| 12 | 10 / 380 | 0.006759 | 0.001070 | 0.004325 | 0.012153 | 380 / yes |
| 12 | 20 / 760 | 0.006691 | 0.001166 | 0.004257 | 0.012114 | 760 / yes |
| 12 | 40 / 1520 | 0.006554 | 0.001081 | 0.004147 | 0.011782 | 1520 / no |
| 12 | 60 / 2280 | 0.006295 | 0.001023 | 0.004316 | 0.011634 | 2280 / no |
| 13 | 1 / 38 | 0.008145 | 0.000923 | 0.005693 | 0.014761 | 1 / yes |
| 13 | 5 / 190 | 0.004841 | 0.000728 | 0.004763 | 0.010332 | 190 / no |
| 13 | 10 / 380 | 0.004291 | 0.000602 | 0.004433 | 0.009327 | 380 / no |
| 13 | 20 / 760 | 0.004605 | 0.000533 | 0.004377 | 0.009515 | 760 / no |
| 13 | 40 / 1520 | 0.004068 | 0.000529 | 0.003947 | 0.008545 | 1520 / yes |
| 13 | 60 / 2280 | 0.004208 | 0.000669 | 0.003648 | 0.008524 | 2280 / no |

Every attempted scheduled fixed-panel guard is retained below. Positive changes mean increased endpoint MSE relative to the fixed validation SOURCE baseline. The closest field is the largest per-field relative increase, including when every field decreases. Step 1 is the mechanical update; SOURCE step 0 is omitted from this table.

| Run | Validation step | Total MSE change | Failed guard entries | Largest field change |
| --- | ---: | ---: | --- | --- |
| 11 | 1 | +0.59% | p | p +6.50% |
| 11 | 190 | +3.24% | p | p +5.63% |
| 11 | 380 | -7.33% | none | CO +1.68% |
| 11 | 570 | -0.95% | none | CO +4.85% |
| 11 | 760 | -5.38% | none | CO +3.16% |
| 11 | 950 | +6.33% | total, CO, p | p +11.93% |
| 11 | 1140 | +6.62% | total, U_1, p | p +9.45% |
| 11 | 1330 | +10.12% | total, U_1, p | p +20.17% |
| 11 | 1520 | +11.22% | total, CO, T, U_1, p | p +18.78% |
| 11 | 1710 | +8.12% | total, U_1, p | p +15.75% |
| 11 | 1900 | +2.29% | none | p +4.98% |
| 11 | 2090 | +0.64% | none | CO +4.13% |
| 11 | 2280 | +13.83% | total, CO, U_1, p | p +24.64% |
| 12 | 1 | -5.00% | none | T +0.91% |
| 12 | 190 | +3.10% | p | p +10.60% |
| 12 | 380 | -1.28% | none | CO +2.76% |
| 12 | 570 | -8.97% | none | T +0.77% |
| 12 | 760 | -5.73% | none | CO -0.24% |
| 12 | 950 | -3.39% | none | T -0.80% |
| 12 | 1140 | +9.98% | total, U_1, p | p +20.61% |
| 12 | 1330 | +7.30% | total, U_1, p | p +13.72% |
| 12 | 1520 | +3.98% | p | p +11.64% |
| 12 | 1710 | +16.14% | total, U_1, p | p +33.52% |
| 12 | 1900 | +3.20% | p | p +9.84% |
| 12 | 2090 | -7.92% | none | CO +2.30% |
| 12 | 2280 | +2.26% | p | p +11.71% |
| 13 | 1 | -0.10% | none | p +4.81% |
| 13 | 190 | +4.90% | p | p +14.43% |
| 13 | 380 | -7.59% | T | T +5.45% |
| 13 | 570 | +2.71% | T, p | T +6.81% |
| 13 | 760 | +2.36% | T, p | T +6.37% |
| 13 | 950 | +9.47% | total, T, p | p +23.90% |
| 13 | 1140 | +9.63% | total, T, p | p +23.32% |
| 13 | 1330 | +17.09% | total, T, U_1, p | p +32.98% |
| 13 | 1520 | -0.97% | none | T +2.91% |
| 13 | 1710 | +5.77% | total, T, p | p +12.02% |
| 13 | 1900 | -0.06% | T | T +5.88% |
| 13 | 2090 | +2.53% | T | T +6.31% |
| 13 | 2280 | +4.28% | T, p | p +9.32% |

The following TRAIN values are the final batch at each milestone, expressed as risk / declared upper bound. They are not epoch means or validation values; a ratio above one is a noisy batch violation.

| Run | Epoch | CH4 | CO | T | U_1 | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 20 | 1.007 | 1.020 | 0.751 | 1.013 | 1.001 |
| 11 | 40 | 0.975 | 1.003 | 0.730 | 0.958 | 0.884 |
| 11 | 60 | 1.004 | 1.040 | 0.749 | 1.051 | 1.117 |
| 12 | 20 | 0.993 | 1.016 | 0.775 | 1.027 | 0.981 |
| 12 | 40 | 1.024 | 1.039 | 0.779 | 0.947 | 0.927 |
| 12 | 60 | 0.970 | 0.998 | 0.793 | 1.031 | 1.078 |
| 13 | 20 | 0.764 | 0.808 | 0.560 | 0.851 | 0.855 |
| 13 | 40 | 0.741 | 0.800 | 0.551 | 0.775 | 0.764 |
| 13 | 60 | 0.732 | 0.826 | 0.549 | 0.893 | 0.931 |

Controller snapshots below use completed model updates. Multiplier order is CH4/CO/T/U_1/p; rectification counts accumulate from update 1; barrier t is the value used on the final update of the epoch.

| Run | Epoch / updates | Controller state |
| --- | --- | --- |
| 11 | 20 / 760 | lambda=[0.00008,0.31500,0.00000,0.06795,0.00024] |
| 11 | 40 / 1520 | lambda=[0.00000,0.26899,0.00000,0.07585,0.00000] |
| 11 | 60 / 2280 | lambda=[0.00205,0.24684,0.00000,0.07475,0.00799] |
| 12 | 20 / 760 | 510 repair / 250 improve (67.11% repair) |
| 12 | 40 / 1520 | 997 repair / 523 improve (65.59% repair) |
| 12 | 60 / 2280 | 1453 repair / 827 improve (63.73% repair) |
| 13 | 20 / 760 | t=1.948717 |
| 13 | 40 / 1520 | t=4.177248 |
| 13 | 60 / 2280 | t=8.140275 |

At the closed 20-epoch boundary, each arm had global/AdamW/controller count 760 and EMA count 315760, with unchanged SOURCE references. Cumulative model-gradient traversals were 3039 / 1260 / 3040 for Runs 11 / 12 / 13, respectively; all had 760 shared reconstructions, 1520 ordinary velocity calls, zero native-loss calls and zero teacher calls. Run 12 used 510 repairs (67.11%) and 250 ConFIG improve updates. Run 11 multipliers in CH4/CO/T/U_1/p order were `[0.0000761,0.3150013,0,0.0679498,0.0002380]`; Run 13 used `t=1.9487171`.

Training-loop wall time through those 20 epochs, summing the one-update segment and its 759-update continuation and including boundary validation/checkpointing, was 2157.07 / 1540.30 / 2164.22 seconds. Setup and compilation are additional. Ordinary update time is recorded separately in `metrics/history.jsonl` and instrument reports.

Runs 11 and 12 passed the epoch-20 fixed-panel guard. Run 13 failed T (+6.37%) and p (+5.01%); its eligible topology best then remained step 1 and its aggregate-MSE best was step 380. Its direction, controller and recovery state were finite, so the same child continued to the authorized 60-epoch assessment with the policy, budget and calibration unchanged.

At the closed 60-epoch boundary, each arm had 2,280 model/AdamW/controller updates, all 145 AdamW parameter states at step 2,280, and EMA count 317,280. Each had 2,280 ordinary shared reconstructions, 4,560 ordinary Flow velocity calls, zero native-loss calls and zero teacher calls. References remained identical to their first saved calibration and every segment verified SOURCE immutability. Restricted checkpoint loads and the first three updates after each resume establish state recovery; this is measured update evidence, rather than launch evidence. Each arm had seen 72,960 training samples, or 9.12 nominal passes at 250 batches/pass.

| Run | Gradient traversals through 60 epochs | Mean ordinary update seconds | Training-loop wall seconds through 60 epochs | Peak allocated GB |
| --- | ---: | ---: | ---: | ---: |
| 11 | 9,119 | 2.760 | 6,424.40 | 15.332 |
| 12 | 3,934 | 1.968 | 4,619.44 | 15.027 |
| 13 | 9,120 | 2.750 | 6,401.35 | 15.329 |

Mean ordinary update times are batch-weighted history measurements over the 2,280 updates. Loop wall times sum the 1-, 759- and 1,520-update segments, including their validation/checkpoint boundaries; setup, compilation and time waiting between segments are additional. They are observed workstation timings, not controlled method-speed comparisons. Run 12 executed 1,453 repair and 827 improve updates (63.73% repair); selected repair fields were CH4 193, CO 429, T 0, U_1 487 and p 344. Both branches remained active. Run 11's multipliers remained finite and bounded in the observed window; its largest final multiplier was CO 0.24684. Run 13's final `t=8.140275` is far below the later cap.

All three final fixed-panel attempts failed at least one +5% bound. At epoch 60, Run 11 had total +13.83%, CO +5.79%, U_1 +12.31% and p +24.64%; Run 12 failed p at +11.71% (total +2.26%); Run 13 failed T at +7.50% and p at +9.32% (total +4.28%). These are real regressions on the fixed validation panel. Earlier passes, including Run 11 at epochs 50/55, Run 12 at 55 and Run 13 at 40, do not establish sustained fidelity. Lower TRAIN A/B/T values likewise do not establish held-out coherence gains.

| Run | `last.pt` step at review | Eligible topology `best.pt` step | Aggregate-MSE `best_fidelity.pt` step | Final guard eligible |
| --- | ---: | ---: | ---: | --- |
| 11 | 2,280 | 380 | 380 | no |
| 12 | 2,280 | 380 | 570 | no |
| 13 | 2,280 | 1,520 | 380 (fails T) | no |

The continuation decision concerns execution readiness for the authorized experiment. Through 60 epochs there were no nonfinite updates, exploding dual state, repair-only collapse or broken checkpoint eligibility; all arms had post-training eligible checkpoints, and recovery remained intact. The observed fidelity failures are substantive scientific concerns, but this window does not establish an irrecoverable controller failure under the work plan's stopping criteria. The same children therefore continue with no policy, learning-rate, budget, SOURCE or calibration changes. No extension of active assessment to 100 epochs was needed. This decision does not certify safety, efficacy or eventual satisfaction of the fidelity budget.

## Autonomous continuation and evidence locations

At closeout on 2026-10-16 at approximately 03:05 UTC, the following processes were live in their original tmux sessions and on their requested physical GPU UUIDs. Each resumed from its own update-2,280 checkpoint without `--max-steps`; mechanical inspection confirmed updates 2,281–2,283, one AdamW step each, exact controller/reference recovery, and no native-loss or teacher calls. Scientific assessment stops at epoch 60; those three resumed updates only verify recovery. The configured target remains 1,000 epochs / 38,000 updates, and no completed 1,000-epoch result is claimed.

| Run | Physical GPU | PID | tmux session | Process state |
| --- | ---: | ---: | --- | --- |
| 11 | 0 | 472981 | `phycoflow_run11_gpu0` | running autonomously |
| 12 | 1 | 457154 | `phycoflow_run12_gpu1` | running autonomously |
| 13 | 2 | 472986 | `phycoflow_run13_gpu2` | running autonomously |

All three execute implementation/configuration commit `b8d85f691fea7dd3389de8a018c7893a3be4851d`; only this operating note changed during execution. The local operations directory is [11_13_operations](../cases/turbulent_combustion/runs/_MultiTask/11_13_operations/), with exact launch/resume scripts, `launch_record.json`, per-arm monitor records and `run11.phase_c.log`, `run12.phase_c.log`, `run13.phase_c.log`. Each child preserves `artifacts/first_update_instrumentation.json`, `phase_a_instrumentation.json`, `phase_b_instrumentation.json`, `readiness60_checkpoint_audit.json` and `continuation_start_instrumentation.json`. These normal ignored operational artifacts remain local; no weight snapshots, prediction datasets or new hashing machinery were added. Recovery uses the ordinary rolling checkpoints. PIDs and rolling slots will change as the jobs progress; the tables above describe the recorded review boundary.

Use `status.json`, `progress.json`, `metrics/history.jsonl`, `metrics/topology_validation.jsonl`, fixed evaluation reports, manifests and restricted-loaded checkpoints together. Epoch histories average scalar telemetry and retain latest nested diagnostics, so sampled ranks/cosines are not a distribution over every update: `field_risk` and `field_violation` are the final batch, not per-field epoch means. The first epoch is represented by two partial history rows (1+37 updates) after the mechanical boundary; use batch-weighted scalar averages and summed mode counts for that epoch. Exact completed model/optimizer/EMA/controller counters come from instrumented state and restricted checkpoints; `status.json` follows saved checkpoint cadence and can lag the live update count. The latest fixed-panel guard is in `metrics/topology_validation.jsonl`; `evaluation/fidelity_guard.json` can retain an earlier selected-checkpoint report and must not be treated as the latest attempted panel. `update_mode_counts` and rectification-field fractions describe switching over every update, while multiplier lists and policy state provide controller trajectories. `last.pt` is the recoverable live state, `best.pt` is validation-selected and may remain the step-0 SOURCE fallback, and `best_fidelity.pt` tracks the lowest finite aggregate fixed-panel MSE under the inherited rule, independently of every-field eligibility; it can contain a candidate that failed a field guard. Check its embedded `topology_selection` before treating it as feasible. Do not conflate these identities or compare different policies by direction norm or an absent surrogate scalar.

Later user-owned evaluation should keep these SOURCEs and scientific settings fixed, distinguish TRAIN estimates from the validation baseline and locked TEST data, assess field fidelity and A/B/T separately, measure stochastic reference uncertainty, and report selected/last checkpoint identity. These runs are not a scientific ranking or a certification of +5% population fidelity, generative calibration, diversity or physical coupling. Full scientific composed-policy execution here measures the specified Flow SOURCE; the earlier standalone Senseiver smoke in the companion guide is not a measured full-topology Senseiver ConFIG campaign.

## Changed files

- Runtime: [fidelity_budget.py](../src/phycoflow_reconstruction/training/fidelity_budget.py), [post_training.py](../src/phycoflow_reconstruction/training/post_training.py), [validate.py](../src/phycoflow_reconstruction/config/validate.py).
- Profiles: [_formal_1000ep_common.yaml](../cases/turbulent_combustion/configs/posttrain/fidelity_budget/_formal_1000ep_common.yaml), [Run 11](../cases/turbulent_combustion/configs/posttrain/fidelity_budget/run11_config_lagrangian_1000ep.yaml), [Run 12](../cases/turbulent_combustion/configs/posttrain/fidelity_budget/run12_config_rectify_1000ep.yaml), [Run 13](../cases/turbulent_combustion/configs/posttrain/fidelity_budget/run13_config_log_barrier_1000ep.yaml).
- Tests: [composition integration](../tests/test_fidelity_budget_composition_integration.py), [direction algebra](../tests/test_fidelity_budget_config_direction.py), [formal configuration](../tests/test_fidelity_budget_formal_config.py).
- Notes: this file and [Fidelity_Budget_Posttraining.md](Fidelity_Budget_Posttraining.md).
