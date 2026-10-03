# PhyCoFlow 1002 R2: native-risk protection and sustained comparison

**Execution record in progress.** This document follows [the R2 plan](UpdatePlans/PHYCOFLOW_WORK_PLAN_1002_R2.md). Scientific results and formal readiness remain pending until the declared mature comparisons finish. The [R1 report](MODEL_UPGRADE_1002.md) remains the record of the earlier short negative study.

## Scope and provenance

The repair adds native-risk protection to the existing endpoint controller and broadens the experimental protocol. A/B/C definitions, model adapters, source prior, normalization, sensor semantics, cached K/V, streamed reconstruction, and two-Euler-stage rollout remain unchanged. Implementation commit: `b22006ef74e6176d8330b9f4d35898ec0fb906b4` on `codex/phycoflow-upgrade-1002`, following the verified local/remote HEAD `31e06a52ccdbe120ae1473f6ddc565fb6e83a6bd`.

| Contract | Verified value |
|---|---|
| Immutable LIVE source | `cases/turbulent_combustion/runs/tc_gl_rbf_cq_cached_kv_5000ep/20260826T212231Z_7b68c461/checkpoints/last.pt` |
| Source SHA-256 | `03f2f45b966a00850cc45c011b4cf1cd644b9408de355324fec8dc434a84a810` |
| Historical protocol | `cases/turbulent_combustion/runs/coherence_fix_ABC_sliced_persistence_formal_5000ep_gpu1/20260924T030724Z_4208231c` |
| Historical state at this audit | Completed 5,000 epochs; the R1-era intermediate status is stale |
| Dataset | `/home/wanglz/Desktop/src/PhyCoFlow/0_demo_TurbulentCombustion/Dataset/Merged_CH4COTU1P.h5` |
| Fields / units | `[CH4, CO, T, U_1, p]`; units unspecified |
| Data roles | One chronological trajectory; TRAIN 8,000, validation 1,000, locked test 1,000 snapshots |
| Geometry / observations | Native lattice 100 × 403; shared 4,096-query coherence geometry; sparse T observations 192–384 |
| Optimization | Historical epoch definition; batch 32, TRAIN fraction 0.15, initial learning rate 5e-5, AdamW decay 1e-6, gradient clipping 1 |
| GPU / environment | Physical GPU 0, RTX 6000 Ada UUID `GPU-233fcd85-5c6a-6212-3f44-655252afec70`; `phycoflow_env` |
| Actual outputs | Fresh R2 directories under `cases/turbulent_combustion/runs/Test_1002` |

The source configuration enables EMA evaluation; the matched audits explicitly use LIVE source and LIVE child weights and restore the prior evaluation state. No historical source-EMA/child-LIVE headline is reused as a matched baseline. Eighty-five protected model/coherence files were hashed in the [frozen provenance](cases/turbulent_combustion/runs/Test_1002/_audit/R2_campaign/frozen_provenance.json); their definitions were not edited.

## Forensic findings before sustained training

The v1 adaptive route computes the native objective under `no_grad`, detaches it, and combines only A/B/C gradients with endpoint fidelity pressure. Its native value is a monitor. A focused regression test confirms that contract; R2 does not reinterpret old v1 checkpoints as native-protected experiments.

The [forensic record](cases/turbulent_combustion/runs/Test_1002/_reports/R2_diagnosis/R2_diagnosis.md) reconstructs the saved R1 epoch histories without inventing missing diagnostics. R1 stochastic native-loss medians rose from epochs 1–5 to 16–20 by 80.77% / 61.10% / 75.94% for ConFIG / weighted / CAGrad. These are unmatched stochastic histories, not source-matched native ratios. The original selected pressure regressions of +24.74%, +10.94%, and +11.49% on the old disjoint block remain failures under their original protocol.

The archived R1 ConFIG epoch-5 pressure example has source/child normalized MSE 0.328600/0.414574 (+26.16%). Mean-offset squared error rises 0.111573 → 0.219080, while centered error falls 0.217027 → 0.195493. This example demonstrates mean-level drift despite improved centered error; it does not authorize pressure-gauge subtraction. See [the PDF decomposition](cases/turbulent_combustion/runs/Test_1002/_reports/R2_diagnosis/pressure_error_decomposition_R1.pdf).

Aligned probes use the same evaluator, LIVE weights, source normalizer, sensors, query IDs, noise, and solver. The source TRAIN/validation gap persists:

| Frozen panel | Source endpoint MSE | Source p MSE | Source native objective |
|---|---:|---:|---:|
| Representative TRAIN 32 | 0.0280801 | 0.0174614 | 0.127859 |
| Spread-out selection validation 64 | 0.382514 | 0.433990 | 1.05931 |

These values establish a distribution/protocol difference under aligned inference. They do not establish leakage, a normalizer defect, or a generalization guarantee. Native objective and endpoint MSE have different definitions and scales. The single trajectory limits independence claims.

## Versioned native protection

The opt-in `endpoint_native_primal_dual_v2` controller adds the unchanged adapter native objective with a live autograd graph. The frozen source receives the same native randomness and no gradients. The native constraint is

\[
g_D=(D(\theta;\mathcal B,\xi)-D(\theta_0;\mathcal B,\xi))/s_D-\delta_D,
\qquad \delta_D=0.
\]

Existing total and five-field endpoint constraints retain their +5% allowance and distinct TRAIN risk scales. All seven constraints use the existing inequality augmented-Lagrangian pressure and nonnegative EMA/dual update. A predeclared 2% native reporting tolerance is separate from the zero-increase optimization budget; repeated-measurement variability must be reported before interpreting it.

Torch CPU/CUDA, Python, and NumPy states are captured around the live native evaluation, replayed for the frozen source, and restored to the post-live state. The public training random stream is preserved. Controller risk order, fixed native scale, multiplier/EMA state, teacher identity, and complete recovery state are versioned. NumPy RNG recovery is serialized with built-in types compatible with the project's restricted checkpoint loader. v1 settings and recovery keep their historical mathematics; cross-version artifacts are rejected.

The proposal remains `Combine(A, B, C) + fidelity_pressure`. Native/endpoint fidelity never enters the A/B/C combiner. Raw ConFIG, weighted, and CAGrad implementations are preserved. A fixed positive TRAIN-only combined-direction calibration makes their source median direction norms comparable to weighted sum; raw and calibrated quantities are reported separately. On the common source calibration, the native scale is 0.1370930830 and method scalars are ConFIG 1.00680663, weighted 1, CAGrad 2.55061131. Actual displacement dots are diagnostics, not finite-epoch guarantees.

## Prospective protocol and verification

The [panel declaration](cases/turbulent_combustion/runs/Test_1002/_reports/R2_diagnosis/r2_panel_declaration.json) freezes selection 64 over 16 chronological strata, nested extended validation 128, and fresh disjoint audit 64 with additional sensor/noise realization. Exact 32-snapshot group membership is saved. A/C group means use equal sample weights; B retains grouped reports and separately pools covariance sufficient statistics. C evaluation uses the full saved 16-line bank; training uses its unchanged four-line subset. Previously inspected validation indices 0–95 and 643 are recorded as development evidence. Reserved audit snapshots are excluded from periodic development native-grid audits. Test remains locked.

Frozen native TRAIN 32 and validation 32 panels use common random seeds 2027 and 3027 every five epochs. They never calibrate or update the controller. Ordinary stochastic TRAIN native epoch means, frozen native ratios, endpoint risks, descriptor components, penalties, multipliers, clipping, and displacement diagnostics remain separate.

The actual config-resolver [comparison audit](cases/turbulent_combustion/runs/Test_1002/_audit/R2_campaign/comparison_contract_review/summary.json) found zero unexpected differences across the five planned arms; active R2_20 exactly matches its planned configuration. Endpoint TRAIN calibration uses the same effective seed 700103 in v1 and v2. Legacy ABC and upgraded ABC retain their distinct definitions, so raw family losses are not interchangeable across those controls. A reviewer-derived validation32 index list was corrected against the authoritative saved run declaration in a [superseding panel record](cases/turbulent_combustion/runs/Test_1002/_audit/R2_campaign/epoch100_boundary_review/native32_panel_contract_correction.json); the actual panels and scientific settings did not change. Exact frozen32 noise checks reproduce saved sensors and queries separately from the broader grouped-native audit.

The frozen implementation milestone passed **608 CPU tests, eight skipped**; focused tests cover matched RNG/source parity, live native gradients, restricted recovery, legacy v1 compatibility, epoch budgeting, panel roles, and PDF export. See [the full-suite log](cases/turbulent_combustion/runs/Test_1002/_audit/R2_campaign/full_cpu.log). The one-epoch implementation smoke passed on physical GPU 0 with the actual model and dataset. At source initialization, native values match exactly on both frozen panels. Native pressure was active over 94.7% of the smoke epoch. After epoch 1, matched native ratios are TRAIN 0.990611 and spread-out selection 1.037615. This is software/forensic evidence only, not a mature scientific comparison or formal readiness result.

A subsequent R2-only archive refinement retains the global best two eligible checkpoints plus the best eligible mature checkpoint, bounded to three. Global best/source-fallback selection stays unchanged. Guarded recovery can retain the current committed mature candidate when an older R2 archive contains only early candidates, without pruning files before a new atomic recovery save. The focused checkpoint/native/recovery/lifecycle suite passed **66 tests, one CUDA-only test skipped**, with [the result manifest](cases/turbulent_combustion/runs/Test_1002/_audit/R2_campaign/cpu_checkpoint_native_recovery.result.json). The CUDA RNG contract passed at the epoch-100 GPU0 boundary. The first ConFIG segment loaded the prior checkpoint manager; its explicit epoch-100 snapshot preserves the current mature state for review and guarded continuation. This changes artifact availability, not optimization mathematics.

Audit/report utilities additionally support full native-grid validation field comparisons, exact 32-sample native audit groups, separate additional-noise/replay checks, and distinct extended/fresh-audit markers. Their focused CPU checks and actual sparse-query software PDF geometry/visual checks passed. No unseen mature panel is inferred from these software checks.

After the consequential retention, epoch-budget and protocol guard changes, the full CPU suite passed once again: **633 passed, nine GPU-specific tests skipped, zero failures/errors**. All 243 source/script/test hashes were unchanged during that run. The [frozen CPU result](cases/turbulent_combustion/runs/Test_1002/_audit/R2_campaign/R2_full_cpu_post_protocol/20261003T074942Z_992572b1/result.json) records exact invocation, source snapshots and skip reasons. A subsequent narrow legacy-control report reader/metadata correction is undergoing its focused checks; it does not alter optimization or selection mathematics.

The legacy reader/metadata correction subsequently passed **26 focused CPU checks** and Ruff, including source-baseline identity for both selectors. Legacy topology scores remain the actual selector; reporting derives separately labeled within-definition A/B/C ratios and records the fixed three-line bank and absence of pooled legacy B. Modern selection scores, full 16-line evaluation and pooled B mathematics are unchanged. At the epoch-100 boundary, **all five targeted CUDA checks passed on physical GPU 0**, including mixed CPU/CUDA/Python/NumPy native RNG replay and A/B numerical contracts; [the result manifest](cases/turbulent_combustion/runs/Test_1002/_audit/R2_campaign/R2_cuda_epoch100/20261003T081911Z_f46a7342/result.json) identifies them. Four historical release fixtures requiring GPU 1 remain intentionally unexecuted.

## Sustained campaign — results pending

| Arm | Declared horizon | Current scientific evidence |
|---|---:|---|
| R2_00 historical legacy ABC/manual-retention control | 100 epochs | Not yet observed |
| R2_10 upgraded ABC/endpoint-only control | 100 epochs | Not yet observed |
| R2_20 native+endpoint ConFIG | 150 epochs; unchanged finalist may reach 200 | Completed 100; broad audit and unchanged continuation pending |
| R2_21 native+endpoint weighted | 150 epochs | Not yet observed |
| R2_22 native+endpoint CAGrad | 150 epochs | Not yet observed |
| R2_30 frozen winning recipe, second training seed | 150 epochs | Pending recipe choice |
| R2_40 one targeted fallback if supported | 150 epochs | Not invoked |

All lineages remain strictly below 250 cumulative epochs. The separate R2 campaign ledger charges attempted work, including lost/replayed work, against 1,100 aggregate epochs. Healthy arms continue through ordinary early fidelity violations. The prospective emergency trigger is broad total endpoint risk >1.5× source together with native risk >1.25× source over a sustained ten-epoch window, plus genuine numerical/provenance emergencies.

Early ConFIG observations remain diagnostic: frozen native TRAIN/validation ratios are 0.992710/1.021606 at epoch 5, 1.023944/1.005458 at epoch 10, and 1.136431/1.026596 at epoch 15. All five fields pass broad selection endpoint fidelity at those boundaries. The epoch-15 TRAIN rise is a concern to follow through mature windows; it is not hidden by endpoint eligibility and does not meet the emergency conjunction. At epoch 5, the two validation native draw ratios differ (0.999165 and 1.044975), so 2% cannot yet be presented as exceeding Monte Carlo variability. A separately saved developmental noise-check declaration fixes additional seeds and exact replay checks at mature boundaries; it never changes the zero native budget or calibrates from validation.

At epoch 35, the selection panel is ineligible because p MSE is +9.483% over source (total +2.089%; other four fields remain within budget). A/B/C ratios are 1.071831 / 0.985411 / 0.856933, and frozen native TRAIN/validation ratios are 0.988506 / 1.041873. The ordinary fidelity/family failures are retained and do not trigger an emergency or change the declared horizon. Through epochs 1–25, stochastic TRAIN native-loss median is 0.127069; the partial epochs 26–35 median is 0.126196. These early values cannot answer the sustained comparison. Native gradient norms are sampled at the diagnostic cadence; activity fractions and coefficient/multiplier means cover every update.

At epoch 50, selection total MSE is −2.104% and p is −5.831% versus source; all five fields pass, with A/B/C ratios 0.911695 / 0.985867 / 0.895225. Frozen native TRAIN/validation ratios are 0.988811 / 0.995763. Stochastic native epoch medians are 0.127069 over epochs 1–25 and 0.126534 over 26–50. Frozen-panel medians for those windows are TRAIN 0.992710 / 0.988811 and validation 1.021606 / 1.015318 (five observations per window). These remain early development results; minimum-horizon and causal control comparisons are pending.

The epoch-50 native-grid audit of two development snapshots reports total endpoint and p ratios 0.966482 / 0.858158, with total C ratio 0.928412. This does not imply uniform topological improvement: T H0 total is 1.339012 and its essential component 1.377698, while T H0/H1 finite components are 0.940582 / 0.933407. Aggregate finite/essential loss reductions are 2.59% / 7.25%; roughly 99.3% of the absolute total C reduction comes from essential components. The two-snapshot native estimator, full-bank selection estimator and four-line TRAIN estimator remain distinct. Saved bar trends support that decomposition without changing C.

At epoch 65, selection p MSE rises 12.73% over source (total +3.66%), and A also regresses (ratio 1.102924). Frozen native TRAIN/validation ratios are 0.994249 / 1.041351. At epoch 70, all five fields regain selection eligibility: total/p ratios are 0.993614 / 0.996519, A/B/C are 0.966598 / 0.989650 / 0.794175, and frozen native TRAIN/validation are 0.956660 / 0.998378. Both boundaries remain in the numerical history. This fluctuation reinforces the need for late-window and nearby-checkpoint review; it neither meets the emergency conjunction nor establishes mature fidelity.

At epoch 75, selection total/p ratios are 0.990092 / 0.970438 and all five fields pass. A/B/C ratios are 0.940287 / 1.011699 / 0.853311; B's 1.17% regression remains explicit. Frozen native TRAIN/validation ratios are 0.977179 / 1.014252. On the same two native-grid development snapshots, total/p ratios are 0.951378 / 0.841095 and C is 0.901202. T H0 essential ratio is now 0.947061, reversing that component's epoch-50 regression on this small panel. Whole C finite/essential ratios are 0.976823 / 0.899660; essential components account for 99.53% of the absolute reduction. The [epoch-75 numerical review](cases/turbulent_combustion/runs/Test_1002/_audit/R2_campaign/epoch75_review.json) binds the audit identity and the [saved component tables](cases/turbulent_combustion/runs/Test_1002/_reports/R2_diagnosis/native_bar_trends/native_bar_family_decomposition.csv) retain the decomposition. These are development results before the minimum scientific horizon.

Normalized MSE and decoded relative L2 remain separate: on those two epoch-75 native snapshots, CO MSE ratio is 0.997862 while mean decoded relative L2 rises slightly, 0.474072 → 0.474655. Passing the MSE gate does not mean every field metric improves.

The [epoch-80 window evidence](cases/turbulent_combustion/runs/Test_1002/_audit/R2_campaign/epoch80_window_evidence.json) separates ordinary stochastic training variation from matched-source drift. Ordinary native epoch-mean medians rise in epochs 51–75, alongside the matched source; this alone does not establish persistent child degradation. Each frozen-panel window has five observations, while each stochastic window has 25 completed historical epochs.

| Epoch window | Stochastic native median | Matched source median | Median paired epoch risk ratio | Frozen TRAIN ratio median | Frozen validation ratio median |
|---|---:|---:|---:|---:|---:|
| 1–25 | 0.127069 | 0.125556 | 1.010676 | 0.992710 | 1.021606 |
| 26–50 | 0.126534 | 0.126007 | 1.004185 | 0.988811 | 1.015318 |
| 51–75 | 0.128997 | 0.128151 | 1.005552 | 0.965046 | 0.998378 |

At epoch 80, the selection endpoint panel remains eligible, but frozen native validation ratio 1.044113 exceeds the 2% descriptive reporting tolerance. Frozen TRAIN is 0.953772. The ordinary native latest-50-epoch Theil–Sen slope is positive (2.0114e-5 objective units per epoch); matched-source and frozen-panel histories must accompany that statistic. These developmental windows do not replace the unobserved mature controls, full noise review, or second seed, and the zero native optimization budget remains unchanged.

At epoch 90, selection remains eligible but B regresses 4.12%, and native validation is +3.43% over source. At epoch 95, selection pressure regresses 10.86% (total +2.18%) and A regresses 8.88%; the state is ineligible. Frozen native TRAIN/validation ratios are 0.950022 / 0.990828 at epoch 95. These ordinary failures remain in the history and do not shorten the epoch-100/150 horizon.

The first segment completed **100 epochs** normally. Its epoch-100 checkpoint is ineligible on selection64: total MSE ratio 1.025868 and p 1.107272; the other four fields remain within budget. A/B/C ratios are 1.098551 / 0.944990 / 0.885895. Frozen native TRAIN/validation ratios are 1.036426 / 1.007317, so TRAIN exceeds the descriptive 2% tolerance at this boundary. Seven of ten selection evaluations over epochs 55–100 pass; failures occur at 65, 95 and 100. The retained selector/archive is still early (selected 45; archive 45/70/60), with no eligible mature state yet. Broad matched epoch-100 audits and unchanged continuation to epoch 150 are pending; this is neither readiness nor a completed optimizer comparison.

On the two native-grid development snapshots at epoch 100, total/p MSE ratios are 0.968039 / 0.917853, while CO is 1.016953 and its decoded relative L2 rises 0.474072 → 0.478981. Native C is 0.854481; finite/essential ratios are 0.981834 / 0.851883, and essential terms account for 99.75% of total reduction. This contrast with selection64 emphasizes panel sensitivity and estimator scope. Sixty saved descriptor reconstruction checks through epoch 100 agree within 4.89e-9; they validate the decomposition, not physical causality or uniform field/topology improvement.

The predeclared mature review preserves rolling states at epochs 140 and 145 for each corrected primary arm (190 and 195 only for an authorized finalist continuation). This supplements global-best/mature retention with two distinct nearby late-window states regardless of their eligibility. Each CPU-only copy verifies epoch, source, config and file hash without changing optimization or selection.

Final tables will report windows 1–25, 26–50, 51–100, 101–150, and 151–200 where observed; 25-epoch medians, ten-epoch smoothing, and latest-50-epoch robust slopes. Checkpoint comparison includes source, best eligible mature candidate, latest, neighboring mature snapshots, late-window eligibility, every family, all five fields, pressure decomposition, and matched contours. Missing windows stay unobserved.

## Formal candidate and storage — pending sustained evidence

The [candidate 5,000-epoch configuration](cases/turbulent_combustion/configs/posttrain/abc_upgrade_1002_r2_5000ep.yaml) and [exact user launch instructions](cases/turbulent_combustion/configs/posttrain/abc_upgrade_1002_r2_launch.md) are prepared. Current preparation status is `not_recommended`, pending mature comparisons and second-seed confirmation; it is not a final sustained-failure classification. Thirty-two CPU/static checks passed using the actual case config resolver, source identity, A/B/C/panel/native settings, scientific sibling parity, bounded PDF/checkpoint policy, shell syntax and CLI help. Neither formal config nor its two-epoch sibling has been executed. The final tested recipe, readiness and sibling result will be finalized after the mature comparison and recipe lock. **No formal run has been launched or queued.**

All new report figures are PDF only. Final retention will preserve sources, essential numerical evidence, calibration/recovery artifacts, selected/latest checkpoints and bounded mature snapshots. A cleanup manifest will identify genuinely disposable new intermediates and the evidence retained in their place.
