# MODEL_UPGRADE_1002_R4 — measured execution and training-dynamics repair

**Status: execution repair measured; mandatory mature comparisons pending. No formal run launched or queued.**

The repository started at `c70010d636601857427b0d27ee0c900cc4f42c14` on `codex/phycoflow-upgrade-1002`. The user's existing report deletions and R3 YAML edits are preserved. This report is being completed from the bounded R4 campaign; it does not yet recommend a 5,000-epoch run.

## Verified evidence and limits

The stopped formal child is `cases/turbulent_combustion/runs/abc_upgrade_1002_r3_candidate_5000ep_gpu0/20261005T120002Z_f5e4939d`. Its status file still says running, but live process/GPU inspection confirmed it stopped. Its journal contains 145 complete epochs; its recovery checkpoint is epoch 140 (5,320 updates). Retained states are epochs 1, 50, 60, 110, 130 and 140; no checkpoint near the early minimum at epochs 30–35 exists. The missing early state is not reconstructed or substituted.

SOURCE remains original LIVE `tc_gl_rbf_cq_cached_kv_5000ep/20260826T212231Z_7b68c461/checkpoints/last.pt`, SHA256 `03f2f45b966a00850cc45c011b4cf1cd644b9408de355324fec8dc434a84a810`. Fields remain `[CH4, CO, T, U_1, p]`, sparse T sensors, TRAIN mean/std, 100×403 domain, 4,096 shared coherence queries, optimizer batch 32, TRAIN fraction 0.15, and 38 updates per historical epoch. Two Euler stages, observation correction, model architecture, source prior and cached-KV semantics are preserved. TEST remains locked.

### Reporting bugs: source-confirmed and recovered from the stopped arrays

The legacy validation history combines source-normalized selector scores (epoch 20: approximately 0.927) with a single-snapshot native objective (epoch 25: approximately 0.381). At coincident epoch 50 the selector replaces the preview in the overloaded legacy record. This produces incompatible alternating scales, not sudden validation improvements. New histories retain separate typed native-preview, fixed-native-panel, endpoint and selector observations, including coincident events. Unknown old records remain unknown; missing native observations are not interpolated.

The adaptive legacy `total` scalar includes coherence but omits the fidelity gradient contribution. It cannot represent a scalar potential for ConFIG plus retention and AdamW. The renderer now labels coherence and update ingredients honestly, and distinguishes parameter pre-clipping gradient norm from actual parameter displacement. Topology component roles are explicit: zero direct weight of an evaluation-only leaf does not disable trained C.

### Same-device execution measurements

All full-workload timings below used physical GPU0, RTX 6000 Ada UUID `GPU-233fcd85-5c6a-6212-3f44-655252afec70`, `phycoflow_env`, logical `cuda:0`, and unchanged epoch exposure. Production training invokes its existing four persistence workers. Original implementations were replayed from an immutable local archive of c70010d; profiler-enabled times are excluded from final speed measurements.

| Path | Warm training seconds/epoch | Short all-inclusive loop seconds/epoch | Interpretation |
|---|---:|---:|---|
| Historical descriptors and retention | 60.098 | 67.080 | Same-GPU speed floor; original historical record is about 52 seconds on its former device |
| R3 exactly as stopped | 97.242 | 104.178 | Reproduced regression |
| R3 with optional reporting/diagnostics disabled | 96.919 (one epoch) | 107.684 | Ordinary training machinery dominates; this is not a warmed speed pass |
| Modern ABC, historical native0.1/aggregate-ConFIG route, repaired execution | 64.270 | 72.155 | 33.91% lower training cost than R3; 1.0694× historical control |
| Epoch-controlled scalar adaptive route, fully matched teacher | 71.224 | 78.416 | 26.76% lower training cost than R3; `partial_speed_recovery` because it misses the absolute target |

Short final-selector/checkpoint overhead is deliberately visible. The two-epoch simple profile has 12.27% loop overhead; it does not establish the mature-cadence ≤10% monitoring criterion. The 100/150/200-epoch pilots will measure its amortization.

Training code is frozen at `a4981db`. The simple reference is running with the immutable 200-epoch configuration and a 100-epoch stage limit. Its first 16 complete epochs have median training time 63.645 seconds, 34.55% below the measured R3 warm epoch. This establishes early sustained runtime evidence only. At fixed-panel epoch 10, equal-family coherence is 0.90830×SOURCE, total endpoint MSE 1.02823×, and matched validation native loss 1.07572×. Native fidelity is outside the corridor at this early observation; no mature qualification is claimed.

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

The simple route records epoch-mean pre-clipping parameter norms but omits per-update clipping flags and actual displacement. A clipped-batch fraction cannot be inferred from that mean. Those quantities remain unavailable in its ordinary journal; four predeclared, disposable matched TRAIN probes will provide explicitly sampled mature diagnostics. They do not substitute for whole-trajectory clipping telemetry.

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

The full CPU suite initially recorded 711 passes, nine expected skips and one resume failure: a concurrent `post_training.py` edit correctly triggered the source-digest guard. Both physical reconstruction roundtrips then passed with source frozen, and the current-revision focused group passed 181 tests with one expected skip. The guard was preserved; this is not reported as a clean full-suite pass. Evidence is [CPU validation summary](cases/turbulent_combustion/runs/Test_1002/R4_tests/cpu_validation_summary.json).

One disposable new 3.44-GB profiler trace was removed after digest/link/open-handle checks. Compact streamed counts, local counterfactuals and calibration JSON are retained with hashes in the [deletion receipt](cases/turbulent_combustion/runs/Test_1002/_audit/R4_campaign/R4_disposable_trace_deletion_receipt.json). No historical run evidence was deleted.
