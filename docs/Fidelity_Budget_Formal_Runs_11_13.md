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

## Actual Run 00 protocol and SOURCE

The authoritative reference is `cases/turbulent_combustion/runs/_MultiTask/00_plain_sum_1000ep/20261015T023430Z_40d8a6e5`, independently inspected as completed at 38,000 updates / 1,000 epochs, including restricted checkpoint loading. Its original SOURCE is `cases/turbulent_combustion/runs/tc_gl_rbf_cq_cached_kv_5000ep/20260826T212231Z_7b68c461/checkpoints/last.pt`, a completed 315,000-update Flow model. Run 00's post-trained checkpoint is never used to initialize these arms. Source hashes continue to come from existing provenance machinery.

The three formal profiles preserve Run 00's inherited model, CH4/CO/T/U_1/p field order, normalization, observations, full-model scope, AdamW LR `5e-5`, weight decay `1e-6`, clip `1`, batch/coherence batch `32`, train fraction `0.15`, 38 updates/reporting epoch, two Euler steps, observation consistency, 4,096 fixed-shared queries and query seed `100045`. Runtime seed is `42`; evaluation seed is `2027`. A/B/T weights and family scales remain one (`family_balance.mode: none`). Full topology uses CO/T, raster 32×128, H0/H1, sublevel/superlevel, 32 projections, self and mutual CO–T with one mutual line. The historical fixed 32-validation-snapshot `topology_with_fidelity` panel retains total and every-field +5% bounds and its five-epoch cadence. SOURCE and evaluation use live weights while EMA state advances and is saved.

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
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PHYCOFLOW_TOPOLOGY_WORKERS=4 KEOPS_CACHE_FOLDER=/home/wanglz/.cache/phycoflow_formal_11_13/keops_gpu0 \
  /home/wanglz/miniconda3/envs/phycoflow_env/bin/python cases/turbulent_combustion/run.py post-train \
  --config configs/posttrain/fidelity_budget/run11_config_lagrangian_1000ep.yaml \
  --max-steps 760
```

Use the corresponding profile/GPU/cache for Runs 12 and 13. The actual launch sequence may first end the same child after one mechanical full-topology update and resume for 759 additional updates; its configured horizon remains 1,000. Once at update 760/epoch 20, resume that exact child with the same profile and `--max-steps 1520` to reach update 2280/epoch 60. Do not restart from SOURCE or reset controller/AdamW/EMA:

```bash
# Example: resume the existing Run 11 child, with the same environment as above.
python cases/turbulent_combustion/run.py post-train \
  --config configs/posttrain/fidelity_budget/run11_config_lagrangian_1000ep.yaml \
  --resume /absolute/path/to/existing/run11-child --max-steps 1520

# After the substantive readiness assessment, continue that same child to 38,000.
python cases/turbulent_combustion/run.py post-train \
  --config configs/posttrain/fidelity_budget/run11_config_lagrangian_1000ep.yaml \
  --resume /absolute/path/to/existing/run11-child
```

Inspect actual processes, GPU identity and step counters rather than treating a session name or `running` status file as liveness. Monitoring records epochs 1/5/10/20 and approximately 40/60, reviews every five-epoch validation, and can extend the same jobs to at most 100 observed epochs if warranted. A real numerical or sustained scientific failure stops the affected arm with evidence, rather than retuning it or silently falling back. Healthy children continue detached toward 1,000; their final completion must later be established from terminal status and checkpoint counters.

The barrier schedule is update-based: at 20 epochs/760 completed updates, the next `t` is `1.1^7≈1.9487`; at 60/2280 it is `1.1^22≈8.1403`; at 100/3800 it is `1.1^38≈37.4043`. The first update using the cap is 4901 (about epoch 129), and `t` stays 100 through 38,000 updates. Tensor checks cover the join, infeasible values and high-t plateau; observing the first 60 epochs does not empirically validate that late regime.

## Measured checks and live run record

Prelaunch focused suites passed 215 tests, plus three genuine fixture-model composition/recovery integration tests and six additional late-schedule tensor cases: 224 distinct tests. They cover analytic policy algebra, separate gradient ordering, FP64 pseudoinverse input, repair traversal omission, no scalar-gradient mislabeling, shared reconstruction, native/teacher omission, ordinary state recovery, old joint/hierarchical behavior, topology and numerical failures. Repository Ruff and `git diff --check` passed. The full scientific SOURCE diagnostic above is separate from optimizer-step evidence.

The formal child identities, one-update full-topology results and 20/40/60-epoch observations will be recorded here as they become available. No 1,000-epoch completion is claimed by a preparation check or a live process.

Use `status.json`, `progress.json`, `metrics/history.jsonl`, `metrics/topology_validation.jsonl`, fixed evaluation reports, manifests and restricted-loaded checkpoints together. Epoch histories average scalar telemetry and retain latest nested diagnostics; exact completed model/optimizer/EMA/controller counters come from status/checkpoints. `update_mode_counts` and rectification-field fractions describe switching over every update, while multiplier lists and policy state provide controller trajectories. `last.pt` is the recoverable live state, `best.pt` is validation-selected and may remain the step-0 SOURCE fallback, and `best_fidelity.pt` tracks the existing fixed-panel fidelity selection. Do not conflate these identities or compare different policies by direction norm or an absent surrogate scalar.

Later user-owned evaluation should keep these SOURCEs and scientific settings fixed, distinguish TRAIN estimates from the validation baseline and locked TEST data, assess field fidelity and A/B/T separately, measure stochastic reference uncertainty, and report selected/last checkpoint identity. These runs are not a scientific ranking or a certification of +5% population fidelity, generative calibration, diversity or physical coupling.
