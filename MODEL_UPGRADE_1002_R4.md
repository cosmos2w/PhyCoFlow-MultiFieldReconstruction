# MODEL_UPGRADE_1002_R4 — measured execution and training-dynamics repair

**Status: reference100 and adaptive150 completed; unchanged adaptive200 continuation and SOURCE seed43 confirmation remain required. No formal run launched or queued.**

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

The reference fixes execution and retains useful coherence gains through 100 epochs, but fails the unchanged native VALIDATION corridor. Its favorable current endpoint checkpoint does not qualify a formal recipe. The completed adaptive150 comparison now passes the native corridor; finalist200 and independent-seed150 remain required.

The adaptive arm completed its SOURCE-initialized150 stage normally in `R4_20_fast_adaptive/20261005T190116Z_279c9107`:5,700 updates and182,400 samples, with the original200-epoch configuration unchanged. The [verified stage comparison](cases/turbulent_combustion/runs/Test_1002/R4_reports/stage150_comparison_verified.json) records mean training **71.1688 seconds/epoch**, median70.6757, ordinary135 mean70.8273 and diagnostic15 mean74.2431. This improves26.813% over measured R3 but fails the absolute65-second target. Whole-process time is10,890.544 seconds, or72.6036 seconds/epoch; the215.217-second residual combines all non-step work and is not an isolated monitor fraction. The [timer audit](cases/turbulent_combustion/runs/Test_1002/R4_reports/training_timer_provenance.json) keeps in-step diagnostics distinct from external previews, native panels, selectors, PDFs and checkpoint callbacks.

| Adaptive150 SOURCE-relative quantity | Trailing25 epochs | Trailing50 epochs |
|---|---:|---:|
| Equal-family coherence |0.891027|0.893400|
| A / grouped B / finite-primary C |0.892696 /1.022950 /0.757435|0.901137 /1.017954 /0.761108|
| Pooled B |1.008141|1.006127|
| Normalized finite / essential C |0.774733 /0.526288|0.776654 /0.553372|
| Native TRAIN aggregate |1.009331|1.009285|
| Native VALIDATION aggregate |1.043734|1.038822|
| Total endpoint / CO endpoint |1.029395 /1.059996|1.023503 /1.053924|

Both native VALIDATION windows pass the unchanged1.05 corridor, and every endpoint field passes the existing1.10 engineering corridor. CO fails the separately reported strict1.05 standard in both windows. The adaptive arm retains90.263% of its best trailing20-epoch coherence gain; both late coherence slopes are negative, TRAIN native slopes negative, and VALIDATION native slopes negative over25 but positive over50 epochs. Exactly67/5,700 accepted updates clip (1.1754%); no SOURCE-calibrated caps saturate. At the sampled epoch150 batch, native pressure/coherence norm ratio is3.5703 despite unsaturated caps: SOURCE calibration is not a live-norm bound. All15 sampled actual AdamW family displacement dots are negative; these are local first-order observations, not population or post-update rescoring evidence. Separate25-epoch component files retain raw aggregate endpoint norms absent from the compact journal.

[B materiality evidence](cases/turbulent_combustion/runs/Test_1002/R4_reports/B_materiality_actual150_review.md) retains the negative third-family observations: adaptive same-frequency B rises4.49% in the50-epoch window, including6.50% in one fixed group. Current150 pooled same-frequency rises7.90%, while cross-frequency improves. Eight saved-coefficient regroupings span0.99146–1.05141; they are grouping sensitivity, not independent-data confidence intervals. No new materiality threshold is used to declare preservation.

The [reviewed continuation choice](cases/turbulent_combustion/runs/Test_1002/R4_reports/stage150_decision.json) selects **unchanged adaptive150→200 plus fresh SOURCE seed43→150 for evidence**, because adaptive native windows pass while simple100 windows fail. Simple is faster and retains more coherence gain. Unequal stage ages and the common100 comparison prevent attributing the late native improvement to PI alone. Neither recipe currently supports combined fast scientific qualification; B preservation, strict CO and adaptive absolute speed remain unresolved. The actual seed43 YAML is prepared and CPU-validated; it has not started. Its only recipe changes are independent TRAIN seed42→43 and a fresh R4 output name.

At [common age100](cases/turbulent_combustion/runs/Test_1002/R4_reports/common_age100_recipe_comparison.md), both recipes have exactly 3,800 updates and identical fixed SOURCE denominators, panels and native draws. Simple/adaptive trailing50 scores are **0.875601/0.896058**; A/B/C ratios are `[0.877200, 1.006931, 0.742673]` and `[0.911580, 1.010136, 0.766458]`. Native VALIDATION aggregates **1.080574/1.062115 both fail** the unchanged 1.05 limit; native TRAIN is 1.007412/0.986760. Adaptive passes strict per-field 5% in both trailing windows, while simple fails strict CO; both preserve the engineering field corridor. 

Adaptive training averages **71.0802 seconds/epoch** (ordinary 70.7387, diagnostic 74.1534), 26.9042% below R3, with 32/3,800 clips (0.8421%) and no cap saturation. These are actual first100 training statistics; the completed150 process wall is reported above. Simple/adaptive retain 92.1%/88.0% of their best rolling gains, although adaptive's 50-epoch coherence slope is positive while its native validation slope is negative. These are recipe comparisons from one seed, not proof of causality of the PI controller or mature formal qualification; the chosen200-epoch continuation and independent-seed150 test remain required.

R3's synchronized warm timers attribute approximately 38.93 seconds/epoch to C forward and 37.94 to backward/optimizer, versus 5.57 A, 0.98 B, 5.51 live rollout, 4.92 source rollout and 2.28 matched native work. Those components describe the original synchronized path; asynchronous repaired host-enqueue timers are not mislabeled device timings. A/B operation microbenchmarks are supporting attribution, not epoch-speed claims.

A late [copied-checkpoint diagnostic audit](cases/turbulent_combustion/runs/Test_1002/R4_reports/adaptive_diagnostic_reuse_gpu_probe.json) measures the retained150 state with one predeclared TRAIN batch and restored optimizer/RNG. The extra25-epoch component function performs32 actual autograd requests and takes6.7707 seconds; compact diagnostic reuse reduces requests12→6 and instrumented update time2.4736→1.1975 seconds. These are first-call, synchronized single-batch scopes, not epoch-speed measurements. The prototype is **unapplied**. Raw losses, controller, descriptor states and RNG agree exactly, but strict gradient/AdamW bitwise parity fails at small differences (preclip max4.19e−9 in the compact pair); model maxdelta1.02e−7 satisfies the existing1e−6 tolerance. The strict failed receipt is preserved. The [identical-frozen-code repeat](cases/turbulent_combustion/runs/Test_1002/R4_reports/adaptive_frozen_repeat_gpu_probe.json) also fails strict bitwise parity: preclip maxdelta6.52e−9, model1.60e−7 and AdamW6.98e−10, with exact raw losses/controller/descriptor/RNG states. Its preclip relative L2 error is1.95e−7. This demonstrates nonbitwise repeatability on the unchanged path; the responsible kernel is unidentified, and it does not establish strict prototype equivalence. No threshold or failed receipt is changed. The original4 and repeat2 copied updates are charged6/38 to the campaign, separately from initial profiling. No prototype source is adopted during the frozen matrix.

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

Full teacher matching remains enabled. A [CPU finite-population calculation](cases/turbulent_combustion/runs/Test_1002/R4_reports/tooling/teacher_subsampling_cpu_formulas.md) uses the adaptive arm's saved complete TRAIN epochs 1/5/10/15/20 to examine optional SOURCE subsampling. Uniform sampling of 10 of 38 batches has zero conditional bias for the affine raw risk signal, but 20,000 subsets per epoch give native false-feasible signal rates of approximately 18–27%. These are raw-signal Monte Carlo estimates, not smoothed-controller disagreement. With independent subset choices across epochs, the exact conditional EMA covariance satisfies `V_e=0.8² V_(e−1)+0.2² Sigma_e`; at epoch20 native estimation-error SD falls from 0.045735 raw to 0.015965 after smoothing, against full native EMA 0.029158. The original changing LIVE trajectory is held fixed in this calculation. Projected PI coefficients and future model trajectories need not be unbiased; neither estimator suitability nor speed is established. No estimator change has been adopted.

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

Code maps: `training/{checkpointing,preview,monitoring,coherence_history,post_training,fidelity_controller,gradient_balance,rebound_monitor}.py`, A `components/{tail_risk,cross_copula}.py`, B `covariance_blocks.py`, and `config/validate.py`. New temporary/profiling/analysis outputs remain under `Test_1002/R4_*` or `_audit/R4_campaign`; meaningful SOURCE/HIST/stopped-formal/R1–R3 artifacts are preserved.

The current figure index contains four PDF masters, 11 pages in total:

- [Runtime comparison, two pages](cases/turbulent_combustion/runs/Test_1002/R4_reports/runtime_comparison.pdf): identical-workload GPU0 profiles and reference100 ordinary/diagnostic/amortized timing. Process residual combines several non-step costs; it is not an isolated monitoring fraction.
- [Corrected losses and controller, three pages](cases/turbulent_combustion/runs/Test_1002/R4_reports/corrected_losses_and_controller.pdf): distinct native/selector series, retained-state local pressure evidence and sparse actual-displacement diagnostics.
- [Coherence and fidelity, three pages](cases/turbulent_combustion/runs/Test_1002/R4_reports/coherence_and_fidelity.pdf): reference100 raw and SOURCE-relative families, grouped/pooled B, finite/essential C, native fidelity failures and endpoint corridors.
- [Field contours, three pages](cases/turbulent_combustion/runs/Test_1002/R4_reports/fields.pdf): actual SOURCE/epoch50/epoch100 GT, SOURCE and child fields with fixed scales and signed errors for all five fields. Values use the physical normalizer; units are undeclared. Epoch100 is a comparison stage, not the pending mature200 finalist.

All 11 pages were visually reviewed. Exact canonical CSV/window cross-checks, PDF hashes and removal of the disposable QA rasters are recorded in the [stage100 figure receipt](cases/turbulent_combustion/runs/Test_1002/R4_reports/_qa/stage100_review.json). The masters and numerical field arrays are retained; mature and second-seed evidence will replace the pending final assessment without adding extra main figures.

The final full CPU regression passed **726 tests with nine expected GPU/opt-in skips and no failures**, in 773.09 seconds, after source stabilization. All 250 implementation/test Python hashes matched before and after the run. The earlier run's 711 passes, nine skips and one concurrent-source-digest resume failure remain in the evidence; the guard was preserved. Relevant full-workload GPU checks are reported separately. Evidence is [CPU validation summary](cases/turbulent_combustion/runs/Test_1002/R4_tests/cpu_validation_summary.json) and the [frozen regression receipt](cases/turbulent_combustion/runs/Test_1002/R4_tests/frozen_cpu_regression_receipt.json).

One disposable new 3.44-GB profiler trace was removed after digest/link/open-handle checks. Compact streamed counts, local counterfactuals and calibration JSON are retained with hashes in the [deletion receipt](cases/turbulent_combustion/runs/Test_1002/_audit/R4_campaign/R4_disposable_trace_deletion_receipt.json). No historical run evidence was deleted.

The final fixture inventory also identified 67 disposable PNG/SVG files from completed synthetic CPU tests under two explicit R4 fixture roots. Their [cleanup receipt](cases/turbulent_combustion/runs/Test_1002/R4_reports/_qa/disposable_fixture_image_cleanup_receipt.json) records birth/hash/link/process checks and preserves all 323 nonimage fixture artifacts, including 27 PDFs, with unchanged identities. Independent post-checks confirm those 67 files are absent and no raster/SVG images remain under R4 paths at this snapshot. Scientific outputs, numerical arrays, model/checkpoint/config/script evidence and all historical roots remain retained.

Early cache isolation required a correction. The initial oracle generated four kernel/header files in the shared KeOps cache. Birth timestamps and hashes identified them; their original inodes now reside under `R4_runtime_cache/early_global_keops_originals`, with atomic compatibility symlinks preserving the old paths and identical contents for other readers. Six preexisting lookup dictionaries were modified by early imports; they are retained, with provenance, rather than overwritten or deleted. Production training and subsequent preparation imports use R4 caches. The final CPU suite lacked three cache environment overrides, but its conservative post-run inventory found no shared cache/Python-bytecode files modified in that suite window. Details are in the [cache relocation receipt](cases/turbulent_combustion/runs/Test_1002/_audit/R4_campaign/early_cache_relocation_receipt.json).
