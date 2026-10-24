<!-- CostAwareLPC owned export v1; root=/data2/wanglz/CostAwareLPC/CostAwareLPC_20261023T185427Z_01a16a15 -->
<!-- Source SHA-256: 69fcf672a22e76d139858e305a137bfbe4b3ceb25f419282d39ecb0f595104a7 -->

# Cost-Aware Lagrangian Execution Strategy: Evidence Report

> **Study status:** All mandatory 100-epoch horizons complete; quality assessments available.

Prepared 2026-10-24T05:27:29+00:00 from the owned campaign root `/data2/wanglz/CostAwareLPC/CostAwareLPC_20261023T185427Z_01a16a15`. The report reads campaign artifacts and analyzer output only; it does not open datasets, checkpoints, or TEST. The science source binding records commit `571180e8e4e8ba06cd545ff1572293f9fbf506af`.

## Reviewed findings

All six mandatory SOURCE-initialized 100-epoch runs are complete. None confirmed the joint target in consecutive mature 40-epoch windows. Time to quality is right-censored at epoch 100 for every arm; retention is unestablished. This study does not demonstrate faster quality attainment or establish impossibility at another horizon.

The fixed P3/whole-T heuristic reduced measured command time by 31.4% on flow (G: 6,414.6 to 4,399.3 s) and 33.6% on DeepONet (D: 5,556.3 to 3,688.0 s). These are latency improvements without confirmed matched quality. Strategy cold-horizon costs including retained profile acquisition were 6,810.7 s for G and 6,196.1 s for D: 6.2% and 11.5% above dense. These exclude subsequent endpoint audits; D includes a separately identified one-second CPU estimate.

At (60,100], G additive ratios were 0.9281/0.9428/0.9437 for dense/heuristic/strategy, all above 0.90 while pooled family and endpoint guards passed. D additive ratios were 0.7064/0.7070/0.7078, but U₁ MSE ratios were 1.0853/1.0859/1.0842, above 1.05. Every mature window failed the full target. Raw target occupancy was 2/10, 1/10 and 0/10 for G, and 0/10 for every D arm; isolated favorable checks cannot substitute for joint attainment.

The default-off layer preserves original scalar coefficients, dense risks and projected dual updates. G selected full execution; D used 380 nested updates before its epoch-10 TRAIN fidelity guard selected full execution for the remaining 3,420 updates. Finite gradient oracles support pre-clipping estimator identities, not trajectories or convergence. The G full-control checkpoint differs from dense by 1.29% relative L2; the first retained loss difference is 1.21e-7 at epoch 1. Its cause remains unresolved, so this is not bitwise replay ([audit](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/independent_validation_G_strategy.json)). Failed periodic ConFIG reuse stays quarantined.

The real D recovery probe preserved a byte-identical cursor-76 checkpoint, stopped only the owned child and resumed the same 100-epoch lineage. Updates 77/78 matched execution identities exactly and numeric diagnostics within the predeclared relative 1e-6/absolute 1e-10 tolerance ([receipt](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/recovery_probe_result.json), [CPU restored decision](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/recovery_runtime_restore.json)). This supports the tested boundary, not bitwise CUDA recovery or future Python/NumPy RNG consumers. Twelve supplementary validation audits retain selected/uniform and final/disjoint results with exact four-group B(32) scope. D selected checkpoints remain at epoch 0 under the unchanged selector; point audits do not replace scheduled histories.

The ledger records 600 science epochs plus 6.6579 auxiliary epoch charges and 33,473.1 s (9.298 h) of admitted process lifetime, including 146.1 s of final audits. Profiling/calibration is a nested 1,032.3 s subscope. Two one-second CPU readiness estimates remain separate; D’s is already allocated in its strategy cold cost. All ceilings hold, original SOURCEs and numerical bindings remain unchanged ([resources](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/resource_summary.json), [protection](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/final_source_protection.json)). The predeclared X pair was omitted: 24,273.0 s remained versus the 26,171.9 s both-full fallback envelope including closeout reserve. No substitute was selected. Cross-model execution is demonstrated; faster quality attainment, retention and physical learning transfer are unestablished. Partial profiler attribution, finite cache/noise probes, sequential one-seed timing, fixed-panel reuse and disclosed HDF5 I/O scope limit broader claims. No TEST feature evaluation or tuning was performed.


## Decision and frozen scope

The execution planner compares applicable measured whole-plan costs after profile acquisition, setup, scheduled refresh, and full-gradient audit overhead, and it admits only TRAIN-calibrated estimator variance under the predeclared noise ceiling and nested importance limit. A no-action full execution is a valid choice when accelerated candidates fail support, noise, amortization, or break-even gates.

| Task | Selected plan | Projected cost | Full-reference forecast | Projected saving | Added variance / intrinsic TRAIN estimate | Maximum joint weight |
|---|---|---:|---:|---:|---:|---:|
| G | full (no action) | 2.04 h | 1.93 h | 0.0% | 0.000 / 257.214 | 1.0 |
| D | nested_topology | 1.10 h | 1.74 h | 37.0% | 22.740 / 168.560 | 4.0 |
| X | nested_topology | 1.81 h | 3.13 h | 42.2% | 1.671 / 2032.065 | 4.0 |

Selection rationale: G: Selected exact full execution; no accelerated candidate passed applicability, support, TRAIN-noise, multiplier, and cold-start break-even gates. D: Selected measured plan 'nested_topology' at 37.0% conservative net savings; added TRAIN execution variance 22.7399 is within the 168.56 ceiling. X: Selected measured plan 'nested_topology' at 42.2% conservative net savings; added TRAIN execution variance 1.67079 is within the 2032.07 ceiling..

Profile lineage: the warm-reference v1 profiles, plans, and failed/superseded attempts remain preserved under `analysis/profiles_warm_reference_v1`; the current forecasts use the complete-retained-profile-lifetime v2 revision. Ledger totals retain every admitted profile lifetime, including failures and earlier warm-reference attempts.

Forecasts are model-based projections from a finite TRAIN profile and are not realized campaign savings or statistical upper bounds. Prediction error is computed only after the matching 100-epoch strategy arm is complete; partial charges remain censored. The full-reference forecast excludes profile acquisition while the selected-plan forecast includes it. For full/no action, the recorded zero saving means no acceleration was adopted; its inclusive forecast still includes profiling and monitoring and can exceed unprofiled dense execution.

### Separate evidence axes

| Evidence axis | Current status |
|---|---|
| Correct execution | 6/6 launched mandatory histories report native-history integrity `ok`; recovery probe is **verified**. D/strategy recovery probe matches the predeclared cursor, replay, binding, and lineage checks. |
| Parameter-estimator expectation | G: max relative error 9.087e-07 (within 5e-4 across four fixed TRAIN batches); D: max relative error 3.951e-07 (within 5e-4 across four fixed TRAIN batches); X: max relative error 7.357e-07 (within 5e-4 across four fixed TRAIN batches). This is a finite estimator-expectation software correctness check, not a confidence bound, convergence result, or trajectory-identity claim; X is software-only and does not establish learning transfer. |
| Latency benefit | paired runs complete; quality-matched cold time still depends on confirmed mature attainment |
| Target attainment | G/dense right_censored; G/heuristic right_censored; G/strategy right_censored; D/dense right_censored; D/heuristic right_censored; D/strategy right_censored |
| Target retention | not established until an arm confirms mature attainment |
| Transfer / scope | D: all three mandatory D 100-epoch model histories complete; cross-model learning comparison is available, with quality efficacy assessed separately; X: The optional X dense/strategy pair was not run: resource gate; physical transfer is unestablished. After all six mandatory 100-epoch histories, the receipt recorded 24273.0 s available versus 26171.9 s required, including the 3600.0 s closeout reserve; reason: After mandatory600, remaining process budget is below both-full fallback planning envelope plus the preserved3600-second closeout reserve, even before added common startup/validation/checkpoint allowances. Do not admit an incomplete/unpaired X lineage or substitute another seed/case based on observed quality. ([validated resource decision](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/optional_pair_decision.json), SHA-256 `5807d35230a72a6bd6ab34bdf826ca92e8b94061d80fe294cc142b52bb7870e4`). |

### Mature attainment and retention

| Task / arm | Execution evidence | Accepted end | Confirmed time or censor | Retention | Raw whole-target occupancy | Worst raw per-check excursions |
|---|---|---:|---|---|---|---|
| G / dense | completed_native_history_valid; recovery verified | 100 | right-censored at E100; 1.78 h accounted cold | not established (no confirmed attainment) | 2/10 (20.0%) | coherence_sum: max 0.9832 vs 0.900 (+0.0832; 8/10 checks); endpoint_field:U_1: max 1.0561 vs 1.050 (+0.0061; 2/10 checks); endpoint_field:p: max 1.0705 vs 1.050 (+0.0205; 2/10 checks); family_B: max 1.0678 vs 1.050 (+0.0178; 1/10 checks) |
| G / heuristic | completed_native_history_valid; recovery verified | 100 | right-censored at E100; 1.22 h accounted cold | not established (no confirmed attainment) | 1/10 (10.0%) | coherence_sum: max 1.0180 vs 0.900 (+0.1180; 8/10 checks); endpoint_aggregate: max 1.0547 vs 1.050 (+0.0047; 2/10 checks); endpoint_field:U_1: max 1.0645 vs 1.050 (+0.0145; 3/10 checks); endpoint_field:p: max 1.0939 vs 1.050 (+0.0439; 3/10 checks) |
| G / strategy | completed_native_history_valid; recovery verified | 100 | right-censored at E100; 1.89 h accounted cold | not established (no confirmed attainment) | 0/10 (0.0%) | coherence_sum: max 0.9795 vs 0.900 (+0.0795; 10/10 checks); endpoint_field:U_1: max 1.0533 vs 1.050 (+0.0033; 2/10 checks); endpoint_field:p: max 1.0563 vs 1.050 (+0.0063; 2/10 checks); family_B: max 1.0642 vs 1.050 (+0.0142; 1/10 checks) |
| D / dense | completed_native_history_valid; recovery verified | 100 | right-censored at E100; 1.54 h accounted cold | not established (no confirmed attainment) | 0/10 (0.0%) | endpoint_aggregate: max 1.1102 vs 1.050 (+0.0602; 7/10 checks); endpoint_field:CO: max 1.0616 vs 1.050 (+0.0116; 1/10 checks); endpoint_field:U_1: max 1.1831 vs 1.050 (+0.1331; 10/10 checks); endpoint_field:p: max 1.1019 vs 1.050 (+0.0519; 3/10 checks) |
| D / heuristic | completed_native_history_valid; recovery verified | 100 | right-censored at E100; 1.02 h accounted cold | not established (no confirmed attainment) | 0/10 (0.0%) | endpoint_aggregate: max 1.1125 vs 1.050 (+0.0625; 7/10 checks); endpoint_field:CO: max 1.0542 vs 1.050 (+0.0042; 1/10 checks); endpoint_field:U_1: max 1.2077 vs 1.050 (+0.1577; 10/10 checks); endpoint_field:p: max 1.0990 vs 1.050 (+0.0490; 3/10 checks) |
| D / strategy | completed_native_history_valid; recovery verified | 100 | right-censored at E100; 1.72 h accounted cold | not established (no confirmed attainment) | 0/10 (0.0%) | endpoint_aggregate: max 1.1135 vs 1.050 (+0.0635; 5/10 checks); endpoint_field:CO: max 1.0731 vs 1.050 (+0.0231; 1/10 checks); endpoint_field:U_1: max 1.1886 vs 1.050 (+0.1386; 10/10 checks); endpoint_field:p: max 1.0954 vs 1.050 (+0.0454; 3/10 checks) |

| Task / arm | E60 | E70 | E80 | E90 | E100 |
|---|---|---|---|---|---|
| G / dense | fail (0/4 (0.0%)) | fail (1/4 (25.0%)) | fail (1/4 (25.0%)) | fail (2/4 (50.0%)) | fail (2/4 (50.0%)) |
| G / heuristic | fail (0/4 (0.0%)) | fail (0/4 (0.0%)) | fail (0/4 (0.0%)) | fail (0/4 (0.0%)) | fail (1/4 (25.0%)) |
| G / strategy | fail (0/4 (0.0%)) | fail (0/4 (0.0%)) | fail (0/4 (0.0%)) | fail (0/4 (0.0%)) | fail (0/4 (0.0%)) |
| D / dense | fail (0/4 (0.0%)) | fail (0/4 (0.0%)) | fail (0/4 (0.0%)) | fail (0/4 (0.0%)) | fail (0/4 (0.0%)) |
| D / heuristic | fail (0/4 (0.0%)) | fail (0/4 (0.0%)) | fail (0/4 (0.0%)) | fail (0/4 (0.0%)) | fail (0/4 (0.0%)) |
| D / strategy | fail (0/4 (0.0%)) | fail (0/4 (0.0%)) | fail (0/4 (0.0%)) | fail (0/4 (0.0%)) | fail (0/4 (0.0%)) |

Window cells show assessor-native pass/fail/censored status and raw whole-target check occupancy; the linked `analysis/mature_window_metrics.csv` retains each metric’s pooled numerator/counts, pooled ratio, per-check mean/median, slope, range, raw violations, and limit. Raw occupancy and per-metric worst excursions above are computed from each observed native check separately and remain visible even when a pooled window passes. If no qualifying consecutive window pair is observed by E100, confirmation is right-censored at the horizon and retention is not established. If confirmation first occurs at E100, subsequent retention is right-censored because no later window exists.

The table below copies assessor-native post-confirmation excursion durations and maximum metric excesses, and lists censored spans without recomputation. The compact JSON/CSV also preserve post-confirmation window/check counts and fractions, final-window pooled ratios, and complete native excursion objects; raw whole-history maxima and counts remain in the attainment table.

| Task / arm | Assessor-native retention excursions (start–end; duration; maximum excess by metric) | Censored spans |
|---|---|---|
| G / dense | not assessed | not assessed |
| G / heuristic | not assessed | not assessed |
| G / strategy | not assessed | not assessed |
| D / dense | not assessed | not assessed |
| D / heuristic | not assessed | not assessed |
| D / strategy | not assessed | not assessed |

### Latest native quality checks

| Task / arm | Run status | Accepted end | Latest native epoch | Accounted cold | Wall elapsed | Additive | A | B | T | Endpoint aggregate MSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G / dense | complete | 100 | 100 | 1.78 h | 1.78 h | 0.9337 ! | 0.9507 | 0.9036 | 0.4754 | 1.0264 |
| G / heuristic | complete | 100 | 100 | 1.22 h | 1.22 h | 0.8771 | 0.8911 | 0.8706 | 0.4693 | 1.0112 |
| G / strategy | complete | 100 | 100 | 1.89 h | 1.78 h | 0.9139 ! | 0.9250 | 0.9966 | 0.4435 | 1.0126 |
| D / dense | complete | 100 | 100 | 1.54 h | 1.54 h | 0.7315 | 0.7959 | 0.5130 | 0.3707 | 1.0423 |
| D / heuristic | complete | 100 | 100 | 1.02 h | 1.02 h | 0.7327 | 0.7946 | 0.5210 | 0.3898 | 1.0422 |
| D / strategy | complete | 100 | 100 | 1.72 h | 1.65 h | 0.7331 | 0.7972 | 0.5128 | 0.3789 | 1.0423 |

| Task / arm | Latest endpoint-field model-unit MSE ratios |
|---|---|
| G / dense | CH4 1.0223, CO 1.0249, T 1.0443, U_1 1.0243, p 1.0288 |
| G / heuristic | CH4 1.0190, CO 1.0181, T 1.0368, U_1 1.0218, p 0.9768 |
| G / strategy | CH4 1.0002, CO 1.0153, T 1.0200, U_1 1.0201, p 0.9970 |
| D / dense | CH4 1.0170, CO 1.0111, T 1.0094, U_1 1.0679 !, p 1.0542 ! |
| D / heuristic | CH4 1.0160, CO 1.0127, T 1.0054, U_1 1.0689 !, p 1.0517 ! |
| D / strategy | CH4 1.0168, CO 1.0108, T 1.0086, U_1 1.0678 !, p 1.0557 ! |

A ! marker denotes a full-precision raw ratio above its limit before display rounding: 0.90 for the additive sum and 1.05 for families and endpoint risks. The compact `analysis/decision_cost_quality.json` and `.csv` preserve one row per task/arm with status, cost attribution, ratios, occupancy, excursions, and window states.

### Portable full A+B+T endpoint audits

Every analyzer-validated portable uniform audit and disjoint supplementary audit is listed; arms without an artifact remain explicitly pending. These endpoint audits do not enter native scheduled histories, mature-window assessment, or time-to-quality. Checkpoint labels follow the existing anchor-bound `topology_with_fidelity` selector, separate from the additive primary optimization and assessment objective; displayed validation ratios do not select or revise a checkpoint.
For D/X, the portable evaluator's legacy two-Euler inference metadata records unused common configuration fields: the deterministic adapter executes its native direct reconstruction and unchanged endpoint observation consistency, with no flow solver or generative loss. G executes the actual two-Euler flow path. The [adapter-scope clarification](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/deterministic_inference_metadata_scope.json) binds this distinction to the frozen source.

| Task / arm | Audit status | Checkpoint / epoch | Panel | Full validation scope | Additive ratio | A ratio | B ratio | T ratio | Endpoint aggregate MSE ratio | Endpoint-field model-unit MSE ratios | Artifact |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| G / dense | valid_metrics | best / E90 | uniform (portable) | validation; full A+B+T; 128 snapshots; 4 groups of 32; 4096 points; six topology filtrations | 0.8980 | 0.9060 | 1.0263 | 0.4421 | 1.0081 | CH4 1.0020, CO 1.0190, T 1.0289, U_1 1.0121, p 0.9875 | [artifact](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/G/dense/20261023T195806Z_89a1e3bc/full_abt_best/full_abt_best_n128_d6869f4018a7_20261024T050828452923Z_validation.json) |
| G / dense | supplementary_disjoint_panel | last / E100 | disjoint (supplementary) | validation; full A+B+T; 128 snapshots; 4 groups of 32; 4096 points; six topology filtrations | 0.9127 ! | 0.9220 | 1.0185 | 0.4641 | 1.0211 | CH4 1.0239, CO 1.0206, T 1.0428, U_1 1.0327, p 0.9947 | [artifact](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/G/dense/20261023T195806Z_89a1e3bc/full_abt_last_disjoint/full_abt_last_n128_f4b7aef3399e_20261024T050842162556Z_disjoint_validation.json) |
| G / heuristic | valid_metrics | best / E100 | uniform (portable) | validation; full A+B+T; 128 snapshots; 4 groups of 32; 4096 points; six topology filtrations | 0.8771 | 0.8911 | 0.8706 | 0.4693 | 1.0112 | CH4 1.0190, CO 1.0181, T 1.0368, U_1 1.0218, p 0.9768 | [artifact](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/G/heuristic/20261023T231858Z_33d84d43/full_abt_best/full_abt_best_n128_f3411ac3e1c5_20261024T050855339156Z_validation.json) |
| G / heuristic | supplementary_disjoint_panel | last / E100 | disjoint (supplementary) | validation; full A+B+T; 128 snapshots; 4 groups of 32; 4096 points; six topology filtrations | 0.9087 ! | 0.9163 | 1.0421 | 0.4714 | 1.0136 | CH4 1.0120, CO 1.0084, T 1.0279, U_1 1.0326, p 0.9800 | [artifact](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/G/heuristic/20261023T231858Z_33d84d43/full_abt_last_disjoint/full_abt_last_n128_bc245bcb008e_20261024T051038364218Z_disjoint_validation.json) |
| G / strategy | valid_metrics | best / E80 | uniform (portable) | validation; full A+B+T; 128 snapshots; 4 groups of 32; 4096 points; six topology filtrations | 0.9285 ! | 0.9434 | 0.9533 | 0.4420 | 1.0280 | CH4 1.0143, CO 1.0300, T 1.0359, U_1 1.0340, p 1.0163 | [artifact](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/G/strategy/20261024T013621Z_5d3b18ae/full_abt_best/full_abt_best_n128_57687156872b_20261024T051101328804Z_validation.json) |
| G / strategy | supplementary_disjoint_panel | last / E100 | disjoint (supplementary) | validation; full A+B+T; 128 snapshots; 4 groups of 32; 4096 points; six topology filtrations | 0.9337 ! | 0.9446 | 1.0374 | 0.4432 | 1.0250 | CH4 1.0098, CO 1.0195, T 1.0248, U_1 1.0394, p 1.0068 | [artifact](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/G/strategy/20261024T013621Z_5d3b18ae/full_abt_last_disjoint/full_abt_last_n128_8c6ca14ccb7a_20261024T051124241878Z_disjoint_validation.json) |
| D / dense | valid_metrics | best / E0 | uniform (portable) | validation; full A+B+T; 128 snapshots; 4 groups of 32; 4096 points; six topology filtrations | 1.0000 ! | 1.0000 | 1.0000 | 1.0000 | 1.0000 | CH4 1.0000, CO 1.0000, T 1.0000, U_1 1.0000, p 1.0000 | [artifact](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/D/dense/20261023T214553Z_7b5c4cf4/full_abt_best/full_abt_best_n128_285ffd91404a_20261024T051145596162Z_validation.json) |
| D / dense | supplementary_disjoint_panel | last / E100 | disjoint (supplementary) | validation; full A+B+T; 128 snapshots; 4 groups of 32; 4096 points; six topology filtrations | 0.7253 | 0.7903 | 0.5022 | 0.3801 | 1.0423 | CH4 1.0174, CO 1.0106, T 1.0096, U_1 1.0686 !, p 1.0536 ! | [artifact](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/D/dense/20261023T214553Z_7b5c4cf4/full_abt_last_disjoint/full_abt_last_n128_e870bd544c0c_20261024T051206990129Z_disjoint_validation.json) |
| D / heuristic | valid_metrics | best / E0 | uniform (portable) | validation; full A+B+T; 128 snapshots; 4 groups of 32; 4096 points; six topology filtrations | 1.0000 ! | 1.0000 | 1.0000 | 1.0000 | 1.0000 | CH4 1.0000, CO 1.0000, T 1.0000, U_1 1.0000, p 1.0000 | [artifact](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/D/heuristic/20261024T003254Z_b544c209/full_abt_best/full_abt_best_n128_25b39bcd263b_20261024T051228118272Z_validation.json) |
| D / heuristic | supplementary_disjoint_panel | last / E100 | disjoint (supplementary) | validation; full A+B+T; 128 snapshots; 4 groups of 32; 4096 points; six topology filtrations | 0.7263 | 0.7889 | 0.5105 | 0.3947 | 1.0423 | CH4 1.0166, CO 1.0122, T 1.0057, U_1 1.0696 !, p 1.0511 ! | [artifact](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/D/heuristic/20261024T003254Z_b544c209/full_abt_last_disjoint/full_abt_last_n128_9103f0a7251a_20261024T051249345196Z_disjoint_validation.json) |
| D / strategy | valid_metrics | best / E0 | uniform (portable) | validation; full A+B+T; 128 snapshots; 4 groups of 32; 4096 points; six topology filtrations | 1.0000 ! | 1.0000 | 1.0000 | 1.0000 | 1.0000 | CH4 1.0000, CO 1.0000, T 1.0000, U_1 1.0000, p 1.0000 | [artifact](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/D/strategy/20261024T032529Z_304cee10/full_abt_best/full_abt_best_n128_502e37803fad_20261024T051310350278Z_validation.json) |
| D / strategy | supplementary_disjoint_panel | last / E100 | disjoint (supplementary) | validation; full A+B+T; 128 snapshots; 4 groups of 32; 4096 points; six topology filtrations | 0.7269 | 0.7918 | 0.5018 | 0.3868 | 1.0424 | CH4 1.0172, CO 1.0103, T 1.0088, U_1 1.0685 !, p 1.0553 ! | [artifact](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/D/strategy/20261024T032529Z_304cee10/full_abt_last_disjoint/full_abt_last_n128_3763789c5268_20261024T051331782738Z_disjoint_validation.json) |
| X / dense | not run: resource gate | not recorded | not run: resource gate | not applicable: X pair omitted (128 snapshots; 4 groups of 32; 4096 points; six topology filtrations) | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded |
| X / strategy | not run: resource gate | not recorded | not run: resource gate | not applicable: X pair omitted (128 snapshots; 4 groups of 32; 4096 points; six topology filtrations) | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded |

### Strategy runtime policy and audit coverage

[Runtime-policy summary](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/runtime_policy_summary.json), SHA-256 `5f880a61cba586561340e8716d95d607d3d077847ae9c33629beff0d421f69c6`.

| Task / arm | Status | Checkpoint step / SHA-256 | Plan initial → final | Accepted cursor | Observed coverage | Monitor / training | Fallback | Recovery intervals | Fixed-TRAIN audit epochs | Full-TRAIN audit artifact |
|---|---|---|---|---:|---|---|---|---|---|---|
| G / strategy | recorded | `3800` / `8c6ca14ccb7a…` | `full` → `full` | `3800` | plans `{"full":3800}`; topology updates `3800`; max importance `1.0` | 24.98 s / 6194.03 s (0.4%) | `sustained TRAIN fidelity violations for fields: CO, U_1` | `[]` | train `[10,20,30,40,50,60,70,80,90,100]`; noise `[20,40,60,80,100]` | [`runs/G/strategy/20261024T013621Z_5d3b18ae/metrics/execution_strategy_audits.jsonl`](<reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/runs/G/strategy/20261024T013621Z_5d3b18ae/metrics/execution_strategy_audits.jsonl>) |
| D / strategy | recorded | `3800` / `3763789c5268…` | `nested_topology` → `full` | `3800` | plans `{"full":3420,"nested_topology":380}`; topology updates `3604`; max importance `4.0` | 23.39 s / 5652.64 s (0.4%) | `sustained TRAIN fidelity violations for fields: CO, U_1, p` | `{"observed_full_plan_coverage":[{"end_epoch":100,"end_step":3800,"plan_id":"full","start_epoch":11,"start_step":381,"update_count":3420}],"scope_note":"Observed execution coverage is derived from accepted update rows and is steps 381-3800 (epochs 11-100). The checkpoint serialized interval [10,20] is preserved as runtime audit/recovery metadata and does not describe the actual duration of full-plan updates.","serialized_state_audit_interval":[{"end_epoch":20,"start_epoch":10}]}` | train `[10,20,30,40,50,60,70,80,90,100]`; noise `[20,40,60,80,100]` | [`runs/D/strategy/20261024T032529Z_304cee10/metrics/execution_strategy_audits.jsonl`](<reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/runs/D/strategy/20261024T032529Z_304cee10/metrics/execution_strategy_audits.jsonl>) |
| X / strategy | not run: resource gate | not recorded / not recorded | not recorded → not recorded | not recorded | plans not recorded; topology updates not recorded; max importance not recorded | not recorded s / not recorded s | not recorded | not recorded | train not recorded; noise not recorded | not recorded |

The monitor percentage is computed from that record as `monitor_overhead_seconds / accepted_training_seconds`; the other plan counts, included-update counts, importance maximum, fallback text, recovery intervals, and audit epochs are copied from the record. Missing strategy records remain pending.

The strategy runtime summary is checkpoint-bound actual run metadata. Sampled optimized Lagrangian training logs describe updates executed under the sampled plan; fixed full-TRAIN audit logs describe the separate full-objective training audits recorded in the summary; scheduled validation histories are a third evidence stream. Their epochs, coverage counts, or outcomes are not substituted for one another.

### TRAIN calibration bindings

| Task | SOURCE TRAIN C0 | Contract C0 | Exact match | Calibration identity | Source-calibration artifact | SHA-256 prefix |
|---|---:|---:|---|---|---|---|
| G | 0.02260933 | 0.02260933 | yes | `52fefbad3cf5b8dd665cf50f0dbf1f6f894de3b5d5a85e8136dc62dcda61319e` | [artifact](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/profiles/G_source_calibration.json) | ee7ca32dfdf1b2c2 |
| D | 0.62622895 | 0.62622895 | yes | `dcbfb9fe33114bc09b7da5ea22331ee5d6eba0c3fa7f324c86800790c17edd73` | [artifact](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/profiles/D_source_calibration.json) | 443dec15a442ddd3 |
| X | 0.40744392 | 0.40744392 | yes | `e0c90ec3a7ad3705fd7d9254bce5f3f173cf0172c0a69fe3ab0e79aecfd01707` | [artifact](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/profiles/X_source_calibration.json) | 9f06509fb8814f33 |

These C0 values are bound from SOURCE TRAIN calibration for training-side eligibility; validation ratios use the paired validation SOURCE baseline from the frozen panel and are a separate population.

The native JSONL omits valid-mask counts, so 4,096 valid queries per validation snapshot are reconstructed from the frozen equal-shape HDF5 loader and all-valid query-mask invariant, not read from serialized per-check counts ([attestation](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/fixed_query_count_attestation.json)). This code-derived count applies only to the current fixed-shape loader; masked or variable-length future loaders require serialized valid counts. The attestation opened no dataset bytes or validators.

Preserved negative evidence: the signed ConFIG coefficient oracle recovered its declared FP64 reference, but both periodic reuse cadences failed the fixed-SOURCE direction-cosine threshold 0.9: the four-update sequence reached 0.891 and the two-update reuse pair reached 0.800 ([retained FAST report](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/manifests/FAST_LAGRANGIAN_LPC_REPORT_at_6021c62.md#L11), SHA-256 `21f997ee655dd4e5207701f5aa6ad353ddb5acdc2345c8539aabdb90a29d813c`). The periodic arm stayed quarantined. The earlier Goal 1 and ordered 100-epoch histories did not establish sustained routine-history retention; the ordered study remains historical and was not continued.

Profile metadata correction ([erratum](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/profile_proxy_label_erratum.json), SHA-256 `a60a6fba3fa85af6104becf51194702469270158dc9cf55b7dfb25853c48a363`): `batches[].output_space_proxy_norms` are unnormalized gradients with respect to reconstructed endpoint predictions, not parameter-gradient norms; the retained `proxy_label` text was wrong. Selection uses the separate weighted/C0-normalized model-parameter gradient arrays (`parameter_execution_variances` and `intrinsic_parameter_variance`); numerical arrays and frozen source were unchanged.

The frozen mandatory set is G and D, each with dense, heuristic, and strategy arms at a 100-epoch horizon. X’s cost forecast is plan-only. The optional X dense/strategy pair was not run: resource gate; physical transfer is unestablished. After all six mandatory 100-epoch histories, the receipt recorded 24273.0 s available versus 26171.9 s required, including the 3600.0 s closeout reserve; reason: After mandatory600, remaining process budget is below both-full fallback planning envelope plus the preserved3600-second closeout reserve, even before added common startup/validation/checkpoint allowances. Do not admit an incomplete/unpaired X lineage or substitute another seed/case based on observed quality. ([validated resource decision](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/optional_pair_decision.json), SHA-256 `5807d35230a72a6bd6ab34bdf826ca92e8b94061d80fe294cc142b52bb7870e4`). A partial or single-arm X history does not establish transfer. A missing, failed, interrupted, under-horizon, multiple-lineage, or time-ambiguous run is not promoted to a completed result.

## Cost and quality evidence

![Figure 1. Plan forecasts, noise screening, ledger-charged execution, and profile cache scope.](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/reports/figures/figure_1_cost_decision.pdf)
Figure 1. Measured whole-update alternatives and TRAIN-noise screening are shown beside the planner's 100-epoch forecast. Panel C bars are terminal training command charges plus allocated profile acquisition, including failed and resumed process lifetimes; final evaluation commands are excluded from these bars and shown separately in the per-arm cost table. The strategy bar includes all task profile attempts once, while dense and fixed-heuristic bars have zero profile allocation. Planner markers cover measured profile cost, whole-optimizer-update extrapolation, and scheduled refresh/audit monitors, but omit native validation, checkpoint and I/O, and fresh-training-calibration startup that remain in actual cold charges. The forecast-to-command gap therefore combines unmodeled common cost and prediction error; it is not pure prediction error. Full-reference and selected-plan markers appear only where measured; no fixed-heuristic forecast is imputed. Fresh TRAIN reference calls across the finite batches (instrumented oracle/input-gradient and support-enumeration scope): G: 42 hits / 726 misses of 768 calls (5.5%); D: 42 hits / 726 misses of 768 calls (5.5%); X: 18 hits / 750 misses of 768 calls (2.3%). Full-update four-batch means, matched pre-oracle / repeated warm: G 1.75/1.36 s; D 1.55/1.29 s; X 1.38/1.07 s. The hit fractions do not describe the timed production-cache path; neither these call counts nor the finite timing means establish stationary-cache or virgin-cache performance.

![Figure 2. Complete native scheduled quality histories for G and D, including endpoint aggregate and fields, A/B/T, and paired self/mutual diagnostics.](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/reports/figures/figure_2_quality_history.pdf)
Figure 2. Each task’s first page compares the same additive coherence ratio against epoch and numeric accounted cold check-completion hours; these clocks stop at each native check and include profile allocation split into measured and recorded estimated amounts in the companion table. Terminal command charges include later closeout and are reported separately in the cost table. Lower panels show A/B/T and endpoint aggregate plus all five endpoint-field model-unit MSE ratios against epoch. Additional pages show positively weighted direct self/mutual component leaves separately: their values are pooled only on the same canonical path as sum(child)/sum(SOURCE) across four fixed 32-snapshot groups, with B independently computed in each; this is a source-weighted mean, not a 128-snapshot covariance claim. Zero-weight diagnostics remain in the raw component CSV, no distinct metric paths or aliases are combined, and self/mutual paths are not summed. Every history point is a primary native scheduled validation check paired to validation SOURCE; no epoch-0 completion time is imputed. Calendar wall completion times are separate in the machine-readable table, and duplicate rows and disagreements remain in analyzer provenance. Final portable evaluations are endpoint audits only and cannot replace scheduled histories or mature-window completion times.

![Figure 3. Mature-window confirmation and retention; optional X pair not run under the validated resource gate.](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/reports/figures/figure_3_mature_windows_transfer.pdf)
Figure 3. Mature windows use the frozen 40-epoch interval `(e-40,e]` and are marked complete only when every scheduled check is present; retention and confirmation are copied from the assessor-native output. X not run: resource gate; physical transfer unestablished. The optional X dense/strategy pair was not run: resource gate; physical transfer is unestablished. After all six mandatory 100-epoch histories, the receipt recorded 24273.0 s available versus 26171.9 s required, including the 3600.0 s closeout reserve; reason: After mandatory600, remaining process budget is below both-full fallback planning envelope plus the preserved3600-second closeout reserve, even before added common startup/validation/checkpoint allowances. Do not admit an incomplete/unpaired X lineage or substitute another seed/case based on observed quality. ([validated resource decision](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/optional_pair_decision.json), SHA-256 `5807d35230a72a6bd6ab34bdf826ca92e8b94061d80fe294cc142b52bb7870e4`).

### Resource accounting

| Task | Action | Attempts | Completed/failed state counts | Charged process time |
|---|---|---:|---|---:|
| D | evaluate | 6 | completed: 6 | 0.02 h |
| D | launch | 3 | completed: 2, failed_or_stopped: 1 | 2.59 h |
| D | profile | 3 | completed: 2, failed_or_stopped: 1 | 0.09 h |
| D | resume | 1 | completed: 1 | 1.61 h |
| G | evaluate | 6 | completed: 6 | 0.02 h |
| G | launch | 3 | completed: 3 | 4.79 h |
| G | profile | 3 | completed: 2, failed_or_stopped: 1 | 0.10 h |
| X | profile | 2 | completed: 2 | 0.07 h |

The ledger reports 9.30 h in terminal process charges across all recorded attempts. 0 attempts are still marked running and their reservations are not counted as realized charges. Profile attempts, including failed and retained warm-reference attempts, remain charged once under their task. Validation evaluation commands are reported as separate ledger actions and are not silently folded into the strategy forecast.

### Per-arm observed cold cost

| Task / arm | Terminal training (s) | Profile measured (s) | Profile estimate (s) | Terminal evaluate (s) | Observed cold total (s) |
|---|---:|---:|---:|---:|---:|
| G / dense | 6414.6 | 0.0 | 0.0 | 27.5 | 6442.1 |
| G / heuristic | 4399.3 | 0.0 | 0.0 | 26.5 | 4425.8 |
| G / strategy | 6436.8 | 373.9 | 0.0 | 25.5 | 6836.3 |
| D / dense | 5556.3 | 0.0 | 0.0 | 22.4 | 5578.6 |
| D / heuristic | 3688.0 | 0.0 | 0.0 | 22.3 | 3710.3 |
| D / strategy | 5873.0 | 322.1 | 1.0 | 21.9 | 6218.0 |
| X / dense | not recorded | 0.0 | 0.0 | 0.0 (none recorded) | not recorded |
| X / strategy | not recorded | not recorded | not recorded | 0.0 (none recorded) | not recorded |

The evaluation column sums terminal `evaluate` ledger charges for all recorded commands. A zero with no recorded evaluation means only that no evaluation command is currently in the ledger; final-audit completeness is separate and is never inferred from absence. An observed total is emitted only when every training and recorded evaluation charge is terminal and measured and the allocated profile total is available; it includes any explicit profile-time estimate. Native TTQ completion times remain unchanged and do not receive future evaluation charges retroactively.

## Methods and limits

For a finite-support unbiased estimator, let `I_i` denote inclusion of scalar contribution `a_i`, with marginal probability `p_i` and joint inclusion probability `π_ij`; the Horvitz–Thompson scalar is `L̂ = Σ_i I_i a_i / p_i`, and its finite-support variance is `Σ_i Σ_j (π_ij − p_i p_j) a_i a_j / (p_i p_j)`. The implemented audits enumerate accepted joint support or use the explicitly declared independent family streams; the report does not substitute marginal-only or assumed-independent covariance for supported correlated draws. Nested family and within-family probabilities multiply: for topology with `p_family=0.5` and conditional polarity inclusion `0.5`, each unit has inclusion `0.25` and inverse-probability multiplier `4`, the predeclared ceiling. Original additive coefficients remain in `a_i`.

Targeted unary reorder and subset-oracle checks verify the fixed descriptor-index/`self_ids` mapping in the opt-in generalized path. Measured CO and T orders already coincide, so these checks make no historical measured-error claim. Legacy identity-ordered v1 estimator state remains accepted; version 2 binds real descriptor IDs and rejects incompatible older reordered state. Existing P3/S2 scientific weights are retained, and correlated/nested variance uses joint inclusion support. The generalized path is gated by these descriptor-index oracle checks and versioned estimator-state bindings.

For parameter gradients `g_i = ∇θ a_i`, the allocation statistic is `V_exec = E||ĝ − g||² = Σ_ij (π_ij/(p_i p_j) − 1) g_iᵀg_j`. For one whole family with gradient `G`, outer inclusion probability `p`, and an unbiased conditional inner estimator of variance `V_inner`, the exact nested variance is `(1/p − 1)||G||² + V_inner/p`; the inner and outer variances cannot simply be added. Enumerating the polarity support gives same-polarity joint inclusion 1/2 and cross-polarity joint inclusion zero; nesting an outer probability 1/2 changes these to 1/4 and zero. Tests also enumerate correlated outer-family support and unequal scientific weights.

For independent work with `v_i = ||g_i||²` and additive marginal costs `c_i`, minimize `Σ_i c_i p_i` subject to `Σ_i v_i(1/p_i − 1) ≤ V_max` and the permitted positive inclusion floor. The Lagrangian derivative is `c_i − λ v_i/p_i² = 0`, giving `p_i = clip(sqrt(λ v_i/c_i), p_min, 1)`; λ enforces the active noise constraint. This independent-work KKT rule motivates allocation but is not applied to correlated support or overlapping costs. The implemented planner compares six measured whole plans because shared reconstruction, caches, batching, backward and dense fidelity make costs nonadditive; it introduces no optimizer.

For a smooth fixed-dual objective with L-Lipschitz gradient and an unbiased minibatch-plus-execution estimator, an unclipped SGD step obeys `E[F(θ_next)] ≤ F(θ) − η(1 − Lη/2)||∇F(θ)||² + (Lη²/2)(σ_train² + V_exec)`, using zero-mean execution noise conditional on the current batch; here `V_exec` is averaged over minibatches and `σ_train²` concerns the full fixed-dual gradient. The measured coherence-only variance is not an estimate of that full-gradient term. This elementary bound motivates noise control only. Persistence nonsmoothness, global clipping, AdamW, evolving projected duals and held-out feasibility lie outside its assumptions, so it supplies no convergence or time-to-quality guarantee for this study.

The sampled scalar and gradient are importance-weighted before global gradient clipping and AdamW. Unbiasedness at that pre-clipping estimator does not establish trajectory equivalence: clipping, adaptive optimizer state, dual updates, checkpoint selection, and later validation are nonlinear and path-dependent. The finite four-batch parameter-gradient variance is coherence-only with SOURCE duals zero; the two-fixed-TRAIN-batch refresh audits are also coherence-gradient estimates, not the full projected dual/AdamW stochastic variance or formal upper bounds. The maximum observed batch duration is an engineering envelope, not a confidence interval.

The cost profile uses four fixed TRAIN batches and matched pre-oracle reference-cache state; the full candidate also has a repeated warm measurement. Instrumented cache counters cover oracle/input-gradient and support-enumeration calls, while whole-update timing uses the production cache without accounting scans. Compiled KeOps cache construction and all admitted profile attempt lifetimes are charged; dense science commands begin after profiling, so the report labels this baseline cache condition rather than claiming a virgin-cache dense comparison. Profiler self-time and inclusive scope durations are not summed because scopes overlap.

The one-update trace is diagnostic only ([scope limitations](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/bottleneck_scope_limitations.json)): family CPU scope values are zero but do not imply zero host work, public reconstruction bypasses the module-forward hook, and data preparation is outside candidate update ticks. Those traces identify no exclusive reconstruction, data-preparation, I/O, CPU, or GPU phase share and no numeric Amdahl affected fraction. Production whole-update timings separately cover four batches but exclude prebuilt-batch I/O; actual command charges include setup, data I/O, validation, and checkpoint work.

C0 and the per-field fidelity denominators are bound from SOURCE TRAIN calibration and are used for training-side eligibility. Validation quality ratios use the paired validation SOURCE baseline from the frozen four-group panel; these are separate populations and are not interchanged.

Quality monitoring uses the fixed 128 uniformly spread validation snapshots in four ordered groups of 32 and the full six-filtration A/B/T path. The disjoint panel, if present, is supplementary only and never enters the primary quality assessment. No TEST path is authorized or opened by this reporting flow. Dataset identity uses the retained fingerprint and metadata/stat attestation; this report does not claim a new full-file byte rehash or prove absence of every operating-system-level read beyond the analyzer's declared access scope.

Resume state binds the accepted cursor and estimator-specific counter RNG, but Python and NumPy global RNG streams are not a byte-for-byte restart contract. Process time is the sum of charged command lifetimes; wall gaps between commands are reported separately by analyzer-native fields when available and are not counted as GPU-active time. One seed per arm and shared-host timings do not establish significance, universal convergence, or causal speedup.

### Artifact scope and rerun commands

This report is generated from the owned profile, raw-profile, plan, attempt-ledger, predeclaration, source-binding, task-inventory, contract-check, and analyzer JSON artifacts under the campaign root; companion analyzer CSVs and preserved profile-revision manifests are fingerprinted for drill-down. The analyzer and report generator are read-only with respect to datasets and checkpoints; they write only analysis/report outputs.

```bash
ROOT='/data2/wanglz/CostAwareLPC/CostAwareLPC_20261023T185427Z_01a16a15'
CUDA_VISIBLE_DEVICES='' python scripts/posttraining/analyze_cost_aware_campaign.py --root "$ROOT"
CUDA_VISIBLE_DEVICES='' python scripts/posttraining/build_cost_aware_report.py --root "$ROOT"
python scripts/posttraining/cost_aware_campaign.py --root "$ROOT" status
```

The campaign controller supports `inspect TASK`, `profile TASK --execute`, `plan TASK`, `launch TASK ARM --execute`, `resume TASK ARM --run RUN_DIR --execute`, `evaluate TASK ARM --run RUN_DIR --checkpoint best --max-samples 128 --panel uniform --execute`, and `stop`; execution commands accept explicit physical GPU and bounded `--max-seconds` options. Inspection, analysis and reporting may write owned metadata or derived outputs and do not train models; status reads records. GPU work requires `--execute`. Stop signals only the active owned process group.


For the G contract check, TRAIN C0 is `0.02260933` and the paired epoch-0 validation SOURCE coherence sum is `0.45129994`. These values serve different populations and are never used interchangeably.

| Evidence artifact | SHA-256 (prefix) |
|---|---|
| [manifests/science_source_binding_v1.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/manifests/science_source_binding_v1.json) | 0de7953603169fd0 |
| [manifests/science_predeclaration.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/manifests/science_predeclaration.json) | 1ee73efb4860ed39 |
| [tasks/task_inventory.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/tasks/task_inventory.json) | 16dd3bfabaa02d5a |
| [manifests/campaign_runner_ledger.jsonl](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/manifests/campaign_runner_ledger.jsonl) | e07aacba7273d7c7 |
| [analysis/first_science_contract_check.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/first_science_contract_check.json) | df71b98875eeb63d |
| [analysis/quality_history.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/quality_history.json) | 43e36c99fe22331a |
| [analysis/final_interpretation.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/final_interpretation.json) | adbf1f04661e9f1c |
| [analysis/runtime_policy_summary.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/runtime_policy_summary.json) | 5f880a61cba58656 |
| [analysis/optional_pair_decision.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/optional_pair_decision.json) | 5807d35230a72a6b |
| [analysis/task_contracts.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/task_contracts.json) | 68e07bf03b1c0176 |
| [analysis/action_registry.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/action_registry.json) | b5e4ddb01b3aa227 |
| [analysis/profiles.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/profiles.json) | 7793f570eac551a6 |
| [analysis/plan_decisions.jsonl](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/plan_decisions.jsonl) | dd6867513b1c3f76 |
| [analysis/resource_summary.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/resource_summary.json) | 322a3a85e337dff5 |
| [analysis/final_source_protection.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/final_source_protection.json) | d235314fee44b807 |
| [analysis/final_endpoint_audit_validation.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/final_endpoint_audit_validation.json) | 160fe9220ea0e563 |
| [analysis/recovery_runtime_restore.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/recovery_runtime_restore.json) | ecc661f6093272ea |
| [analysis/quality_history.csv](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/quality_history.csv) | e4915f1e041cabcf |
| [analysis/quality_components.csv](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/quality_components.csv) | 995832258c04639c |
| [analysis/decision_cost_quality.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/decision_cost_quality.json) | 1a2804097552d843 |
| [analysis/decision_cost_quality.csv](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/decision_cost_quality.csv) | d51dc5876ef51e80 |
| [analysis/mature_window_metrics.csv](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/mature_window_metrics.csv) | 2e99e8224cf15c77 |
| [analysis/fixed_query_count_attestation.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/fixed_query_count_attestation.json) | 93c4693c9adf263d |
| [analysis/bottleneck_scope_limitations.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/bottleneck_scope_limitations.json) | 365ab9251d992a4e |
| [analysis/profile_proxy_label_erratum.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/profile_proxy_label_erratum.json) | a60a6fba3fa85af6 |
| [analysis/deterministic_inference_metadata_scope.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/deterministic_inference_metadata_scope.json) | 04632e7b3bf31671 |
| [manifests/FAST_LAGRANGIAN_LPC_REPORT_at_6021c62.md](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/manifests/FAST_LAGRANGIAN_LPC_REPORT_at_6021c62.md) | 21f997ee655dd4e5 |
| [analysis/profiles_warm_reference_v1/revision.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/profiles_warm_reference_v1/revision.json) | 76c8221812e26e63 |
| [analysis/profile_v2_complete_cost_revision/revision.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/analysis/profile_v2_complete_cost_revision/revision.json) | 869c6c35354a682b |
| [profiles/G_profile.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/profiles/G_profile.json) | c702fce7db88180b |
| [profiles/G_raw_profile.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/profiles/G_raw_profile.json) | 6ce5f48821b1a3b9 |
| [profiles/D_profile.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/profiles/D_profile.json) | 01cfc5c4419af3b3 |
| [profiles/D_raw_profile.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/profiles/D_raw_profile.json) | a3c341ac556ced08 |
| [profiles/X_profile.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/profiles/X_profile.json) | 2bda9ae88419e20d |
| [profiles/X_raw_profile.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/profiles/X_raw_profile.json) | a46f9bb977237732 |
| [plans/G_plan_decision.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/plans/G_plan_decision.json) | 4e4332f6d8e83bdd |
| [plans/D_plan_decision.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/plans/D_plan_decision.json) | ce23bc24a4266eb5 |
| [plans/X_plan_decision.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/plans/X_plan_decision.json) | 23286ff1f6955440 |
| [profiles/G_source_calibration.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/profiles/G_source_calibration.json) | ee7ca32dfdf1b2c2 |
| [profiles/D_source_calibration.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/profiles/D_source_calibration.json) | 443dec15a442ddd3 |
| [profiles/X_source_calibration.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/profiles/X_source_calibration.json) | 9f06509fb8814f33 |
| [tasks/G_task_contract.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/tasks/G_task_contract.json) | e5f4eaaba4fa91a1 |
| [tasks/D_task_contract.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/tasks/D_task_contract.json) | b00d1a018679a50a |
| [tasks/X_task_contract.json](reports/CostAwareLPC_20261023T185427Z_01a16a15_evidence/tasks/X_task_contract.json) | 7b560cbe797ec8c9 |

The main-reviewed interpretation above is bound to the current quality history and campaign ledger; this report does not auto-select a winner.
