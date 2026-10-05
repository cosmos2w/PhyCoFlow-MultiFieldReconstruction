# MODEL_UPGRADE_1002_R4 — measured execution and training-dynamics repair

**Status: execution repair measured; mandatory mature comparisons pending. No formal run launched or queued.**

The repository started at `c70010d636601857427b0d27ee0c900cc4f42c14` on `codex/phycoflow-upgrade-1002`. The user's existing report deletions and R3 YAML edits are preserved. This report is being completed from the bounded R4 campaign; it does not yet recommend a 5,000-epoch run.

## Verified evidence and limits

The stopped formal child is `cases/turbulent_combustion/runs/abc_upgrade_1002_r3_candidate_5000ep_gpu0/20261005T120002Z_f5e4939d`. Its status file still says running, but live process/GPU inspection confirmed it stopped. Its journal contains 145 complete epochs; its recovery checkpoint is epoch 140 (5,320 updates). Retained states are epochs 1, 50, 60, 110, 130 and 140; no checkpoint near the early minimum at epochs 30–35 exists. The missing early state is not reconstructed or substituted.

The [explicit stopped/pilot comparison](cases/turbulent_combustion/runs/Test_1002/_audit/R4_campaign/stopped_first100_identity_comparison.json) verifies the same immutable LIVE SOURCE, identical TRAIN calibration draws/query identities, and exact retained epoch-1 descriptor tensor and non-tensor states. Their calibration objects differ slightly; actual epoch-1 learned model tensors differ by up to 2.30307e−4 despite equal retained CPU/CUDA/index RNG states. First-100 trajectories are compared as observations, not assumed identical from YAML. Neither dirty historical startup retains a complete source-file snapshot; the stopped metadata names c70010d, but its exact dirty startup bytes cannot be independently reconstructed. The replay therefore reproduces the stopped resolved workload on the pinned oracle, rather than claiming an exact historical source snapshot.

SOURCE remains original LIVE `tc_gl_rbf_cq_cached_kv_5000ep/20260826T212231Z_7b68c461/checkpoints/last.pt`, SHA256 `03f2f45b966a00850cc45c011b4cf1cd644b9408de355324fec8dc434a84a810`. Fields remain `[CH4, CO, T, U_1, p]`, sparse T sensors, TRAIN mean/std, 100×403 domain, 4,096 shared coherence queries, optimizer batch 32, TRAIN fraction 0.15, and 38 updates per historical epoch. Two Euler stages, observation correction, model architecture, source prior and cached-KV semantics are preserved. TEST remains locked.

### Reporting bugs: source-confirmed and recovered from the stopped arrays

The legacy validation history combines source-normalized selector scores (epoch 20: approximately 0.927) with a single-snapshot native objective (epoch 25: approximately 0.381). At coincident epoch 50 the selector replaces the preview in the overloaded legacy record. This produces incompatible alternating scales, not sudden validation improvements. New histories retain separate typed native-preview, fixed-native-panel, endpoint and selector observations, including coincident events. Unknown old records remain unknown; missing native observations are not interpolated.

The repaired reference's actual epoch-50 journal preserves nine separate records at the same event: fixed native panel (1.03030×SOURCE), native single-sample preview (raw 0.48104), selector score (0.84882), total endpoint and all five field endpoints. This verifies the coincident-cadence repair during real training, in addition to the focused tests.

The adaptive legacy `total` scalar includes coherence but omits the fidelity gradient contribution. It cannot represent a scalar potential for ConFIG plus retention and AdamW. The renderer now labels coherence and update ingredients honestly, and distinguishes parameter pre-clipping gradient norm from actual parameter displacement. Topology component roles are explicit: zero direct weight of an evaluation-only leaf does not disable trained C.

### Same-device execution measurements

All full-workload timings below used physical GPU0, RTX 6000 Ada UUID `GPU-233fcd85-5c6a-6212-3f44-655252afec70`, `phycoflow_env`, logical `cuda:0`, and unchanged epoch exposure. Production training invokes its existing four persistence workers. Original implementations were replayed from an immutable local archive of c70010d; profiler-enabled times are excluded from final speed measurements.

| Path | Warm training seconds/epoch | Short all-inclusive loop seconds/epoch | Interpretation |
|---|---:|---:|---|
| Historical descriptors and retention | 60.098 | 67.080 | Same-GPU speed floor; original historical record is about 52 seconds on its former device |
| Stopped R3 resolved workload on pinned c70010d oracle | 97.242 | 104.178 | Reproduced regression; historical dirty source snapshot unavailable |
| R3 with optional reporting/diagnostics disabled | 96.919 (one epoch) | 107.684 | Ordinary training machinery dominates; this is not a warmed speed pass |
| Modern ABC, historical native0.1/aggregate-ConFIG route, repaired execution | 64.270 | 72.155 | 33.91% lower training cost than R3; 1.0694× historical control |
| Epoch-controlled scalar adaptive route, fully matched teacher | 71.224 | 78.416 | 26.76% lower training cost than R3; `partial_speed_recovery` because it misses the absolute target |

Short final-selector/checkpoint overhead is deliberately visible. The two-epoch simple profile has 12.27% loop overhead; it does not establish the mature-cadence ≤10% monitoring criterion. The 100/150/200-epoch pilots will measure its amortization.

Training code is frozen at `a4981db`. The simple reference completed its 100-epoch stage with the immutable 200-epoch configuration: exactly 3,800 updates and 121,600 training samples. Mean training time is **63.9046 seconds/epoch**, median 63.696, and the mean reduction relative to the pinned R3 replay is **34.283%**. Ordinary epochs average 63.6536 seconds; diagnostic epochs average 66.1640 seconds and exceed the absolute 65-second target. Amortized training meets 60–65 seconds with unchanged exposure. Whole-process time is 6,546.336 seconds, or 65.4634 seconds/epoch. The 155.872-second residual (2.381% of process time) combines startup, data wait, evaluation, logging, checkpointing and PDF work; it is not an isolated monitoring measurement.

The reference fixes execution and retains useful coherence gains through 100 epochs, but fails the unchanged native VALIDATION corridor. Its favorable current endpoint checkpoint does not qualify a formal recipe. The required adaptive150, finalist200 and independent-seed150 comparisons remain pending.

R3's synchronized warm timers attribute approximately 38.93 seconds/epoch to C forward and 37.94 to backward/optimizer, versus 5.57 A, 0.98 B, 5.51 live rollout, 4.92 source rollout and 2.28 matched native work. Those components describe the original synchronized path; asynchronous repaired host-enqueue timers are not mislabeled device timings. A/B operation microbenchmarks are supporting attribution, not epoch-speed claims.

### Retention diagnosis: measured local evidence, not a universal causal claim

Matched TRAIN probes use actual retained epochs 50, 60 and 140, identical source/native draws, and disposable optimizer-state copies. Native-retention/coherence norm ratios are 15.77, 30.73 and 1.86. At epoch 60 removing native pressure improves the local calibrated-coherence rescore by approximately 2.39× (−0.000672 versus −0.000281); bounding native pressure also helps. At epoch 140 removal is worse. All three sampled original directions have positive A/B/C gradient dots and improving actual AdamW displacement dots. Thus the evidence supports excessive pressure limiting progress in the rebound regime; it does not prove that every original update reverses coherence.

The stopped trajectory also improves again around epoch 118. That later recovery is retained in the analysis. The purple metric-identity bug does not explain genuine training A/B rebound. R4's slower controller must demonstrate a sustained improvement in the authorized mature comparisons.

## Changes and classification

| Class | Exact change | Scientific boundary |
|---|---|---|
| Report-only | Separate validation identities and metadata; honest objective/gradient labels; explicit epoch render cadence; monitoring-only rebound alarm | No new acceptance corridor or model inputs |
| Exact execution | Diagnostic-only graph traversals sampled once per 10 epochs; ordinary family timing synchronization removed; context requests separated from matched teacher requests | Required native/endpoint matching retained |
| Exact execution | One scalar backward for the existing weighted-sum route; wide clipping multiplication/unused-complex parameter behavior preserved | Floating-point summation tolerance documented; ConFIG is not called scalar-equivalent |
| Exact execution | A batched eta roots with independent stopping and one packed CPU transfer; ordinary exact-rank evaluation diagnostics deferred | Own-CDF/smooth-CVaR values and envelope gradients preserved |
| Exact execution | B indices hoisted and diagnostic payload/scalars deferred | Signed second-order same-/cross-band covariance and symmetric floors preserved |
| Tested prototype, excluded from production | Small-Gram ConFIG | Numerically matched but slower on GPU; production retains installed combiner |
| Optimizer change | One epoch PI controller with a native floor and TRAIN-calibrated caps; positive scalar ABC combination for the adaptive arm | Retained native/endpoint risk equations and budgets unchanged |
| Estimator approximation | None adopted | Frozen teacher remains fully matched on every adaptive batch |

GPU A/B value and gradient tests matched bitwise; measured A forward/backward 0.1832→0.0868 seconds and B 0.0397→0.0289 seconds. Scalar mixed real/complex/unused AdamW updates differed by at most 4.77e−7. These are operation checks, not sustained-training evidence. Extreme float32/complex64 clipping tests preserve the oracle's wide multiplication semantics.

The simple route records epoch-mean pre-clipping parameter norms but omits per-update clipping flags and actual displacement. A clipped-batch fraction cannot be inferred from that mean. Four predeclared matched TRAIN batches were independently restored to the actual epoch-100 model and AdamW state. All followed the existing `weighted_sum_aligned` branch; pre-clipping parameter norms were 0.183–0.221 and clipping was **0/4 sampled batches**. Every sampled family, native and aggregate endpoint gradient had a negative dot product with the measured actual AdamW displacement. These dots are first-order local predictions; no post-update loss rescore was performed. They show no reversal in these samples, not a whole-trajectory or validation guarantee. The source and child checkpoint hashes remained unchanged. The four disposable updates consume 4/38 campaign epochs; the initial ten-epoch profiling cap remains separate as explicitly scoped in the plan.

### Completed 100-epoch reference: fixed-window evidence

The [stage100 assessment](cases/turbulent_combustion/runs/Test_1002/R4_reports/stage100_reference_review.json) binds SOURCE, estimator/panel/normalizer identity and raw numerators. Windows contain the actual scheduled observations in `(100−window,100]`: epochs 80/90/100 for 25 epochs and 60/70/80/90/100 for 50. There is no interpolation or post-hoc smoothing.

| SOURCE-relative quantity | Trailing 25 epochs | Trailing 50 epochs |
|---|---:|---:|
| Equal-family coherence score | 0.878054 | 0.875601 |
| A own-CDF/tail family | 0.883106 | 0.877200 |
| B grouped32 family | 1.011334 | 1.006931 |
| B pooled64 diagnostic | 0.985421 | 0.986461 |
| C finite-primary family | 0.739721 | 0.742673 |
| C normalized finite contribution | 0.756222 | 0.758364 |
| C raw finite distance | 0.738469 | 0.741052 |
| C normalized essential contribution | 0.519231 | 0.532989 |
| Native TRAIN aggregate | 1.011731 | 1.007412 |
| Native VALIDATION aggregate | **1.081831 — fails** | **1.080574 — fails** |
| Total endpoint MSE | 1.024240 | 1.024110 |
| CO endpoint MSE | 1.056939 | 1.059606 |

A and C improve; grouped B's small regression remains visible rather than being replaced by its favorable pooled diagnostic. Finite persistence improves independently of essential gains. All endpoint fields satisfy the exploratory +10% corridor in both windows, while CO fails the separately reported stricter +5% standard. Native TRAIN settles close to SOURCE, but native VALIDATION does not meet +5%; local TRAIN descent cannot certify that generalization gap.

Coherence slopes are −0.00069935 and −0.00004744 per epoch over 25/50 epochs; native VALIDATION ratio slopes are −0.00265667 and −0.00053032. Both window means remain below SOURCE for coherence, and there is no persistent rebound alarm. The 50-epoch coherence window retains 92.11% of the best declared 20-epoch rolling gain, or approximately 82.28% relative to the unsmoothed epoch-50 minimum. These are distinct denominators, not interchangeable retention claims.

Current and endpoint-selected epoch100 have coherence 0.868025, A/B/C 0.894339/0.962536/0.747200, total endpoint 1.025582, and per-field ratios `[0.974102, 1.049547, 0.998297, 1.023635, 1.027996]` in the contracted order. All current endpoint fields pass +5%, but current matched native VALIDATION is 1.056360 and still fails. Earlier selected epoch20 and the favorable native epoch50 remain in the record; neither overrides the final windows.

Final-save records repeat the epoch100 selector and eight typed metric identities. The read-only assessment reconciles these only after exact scientific/identity comparison, validating nine deterministic SOURCE-derived fields and excluding only `inference.seconds`. Raw rows and hashes remain preserved; conflicting observations remain unresolved. Nineteen focused reader fixtures verify identical and conflicting values, panels, SOURCE, normalizer and noise identities.

### Exact epoch-controller equation and calibration

The adaptive comparator uses fixed positive TRAIN-fitted family scales:

`d_e = grad(sum_k s_k L_k + sum_j a_j,e g_j)`.

`g_native = (D_live - D_source)/s_native - 0.02`; each endpoint risk retains its source-relative +5% training allowance. All matched source terms remain constants in model gradients. Coefficients are fixed throughout each historical epoch:

`gbar_e = 0.8*gbar_(e-1) + 0.2*mean_TRAIN(g_e)`

`a_(e+1) = clip(a_e + 0.01*gbar_e + 0.02*(gbar_e-gbar_(e-1)), floor, cap)`.

EMA starts at zero. The projected coefficient is the integrator state, so saturation cannot store a hidden windup reservoir. There is no old per-update dual accumulator and no instantaneous `rho*g` term. The native floor inherits raw native weight 0.1; other floors are zero. Complete partial-epoch sums/counts and the historical epoch length are recovered exactly; incompatible versions/exposures fail.

One fixed TRAIN SOURCE batch gives reference calibrated-ConFIG norm 0.302082. Native cap allocates one reference-direction norm, and the six endpoint constraints together allocate one, uniformly in norm. Caps are native 0.0525027, total 0.00539309, CH4 0.00289060, CO 0.00394373, T 0.00232732, U_1 0.00339480 and p 0.00172994. The native floor gives approximately 0.261× that source reference direction. These are conservative engineering limits fitted once on TRAIN, not a bound on future gradient norms or a feasibility guarantee. No validation hyperparameter sweep is authorized.

The policy is an explicit engineering PI variant motivated by [Sohrabi et al., ICML 2024](https://proceedings.mlr.press/v235/sohrabi24a.html); it is not claimed to reproduce νPI exactly or inherit its theory for this field model.

## Bounded campaign and unresolved work

The ledger is `cases/turbulent_combustion/runs/Test_1002/_audit/R4_campaign/campaign_ledger.json`. Every lineage remains below 250 epochs; total budget is 620 epoch exposures and 24 physical-GPU0 hours, including profiling/replay/smoke. Main arms start from immutable LIVE SOURCE: simple reference100; one adaptive repair150; unchanged finalist continuation200; fresh independent seed150; eligible formal sibling2. No 5,000-epoch child is launched or queued.

The fixed fidelity corridors remain total endpoint≤1.05×SOURCE, each field≤1.10×SOURCE, matched-native mean≤1.05×SOURCE; the stricter each-field+5% standard is reported separately. Mature results must include unsmoothed observations, fixed 25/50-epoch trailing windows and slopes, finite/essential C separately, grouped/pooled B separately, and fixed native/source numerators. An early minimum or a best checkpoint alone cannot qualify the formal recipe.

The rebound alarm averages scheduled fixed-panel observations in `(epoch−20,epoch]`; after a useful≥5% gain, return above0.98×SOURCE or ≥80% erasure persisting for two scheduled observations requests one sparse TRAIN diagnostic. It does not change LR, reset weights or switch objectives.

**Mature learning, second seed, field audits, final decision, tested formal configuration and user launch/stop/resume command remain pending.**

## Evidence and code

- [Forensic resolved-config comparison](cases/turbulent_combustion/runs/Test_1002/_audit/R4_campaign/source_and_run_diff.json)
- [Runtime attribution](cases/turbulent_combustion/runs/Test_1002/_audit/R4_campaign/runtime_attribution.json)
- [Matched retained-state counterfactuals](cases/turbulent_combustion/runs/Test_1002/_audit/R4_campaign/controller_counterfactuals.json)
- [Fixed TRAIN controller design](cases/turbulent_combustion/runs/Test_1002/_audit/R4_campaign/controller_design.json)
- [GPU operation parity](cases/turbulent_combustion/runs/Test_1002/R4_00_exact_math/gpu_math_parity.json)

Code maps: `training/{checkpointing,preview,monitoring,coherence_history,post_training,fidelity_controller,gradient_balance,rebound_monitor}.py`, A `components/{tail_risk,cross_copula}.py`, B `covariance_blocks.py`, and `config/validate.py`. New temporary/profiling/analysis outputs remain under `Test_1002/R4_*` or `_audit/R4_campaign`; meaningful SOURCE/HIST/stopped-formal/R1–R3 artifacts are preserved. Main figures will be PDF-only under `Test_1002/R4_reports`, with a short deletion manifest only for disposable new intermediates.

The final full CPU regression passed **726 tests with nine expected GPU/opt-in skips and no failures**, in 773.09 seconds, after source stabilization. All 250 implementation/test Python hashes matched before and after the run. The earlier run's 711 passes, nine skips and one concurrent-source-digest resume failure remain in the evidence; the guard was preserved. Relevant full-workload GPU checks are reported separately. Evidence is [CPU validation summary](cases/turbulent_combustion/runs/Test_1002/R4_tests/cpu_validation_summary.json) and the [frozen regression receipt](cases/turbulent_combustion/runs/Test_1002/R4_tests/frozen_cpu_regression_receipt.json).

One disposable new 3.44-GB profiler trace was removed after digest/link/open-handle checks. Compact streamed counts, local counterfactuals and calibration JSON are retained with hashes in the [deletion receipt](cases/turbulent_combustion/runs/Test_1002/_audit/R4_campaign/R4_disposable_trace_deletion_receipt.json). No historical run evidence was deleted.

Early cache isolation required a correction. The initial oracle generated four kernel/header files in the shared KeOps cache. Birth timestamps and hashes identified them; their original inodes now reside under `R4_runtime_cache/early_global_keops_originals`, with atomic compatibility symlinks preserving the old paths and identical contents for other readers. Six preexisting lookup dictionaries were modified by early imports; they are retained, with provenance, rather than overwritten or deleted. Production training and subsequent preparation imports use R4 caches. The final CPU suite lacked three cache environment overrides, but its conservative post-run inventory found no shared cache/Python-bytecode files modified in that suite window. Details are in the [cache relocation receipt](cases/turbulent_combustion/runs/Test_1002/_audit/R4_campaign/early_cache_relocation_receipt.json).
