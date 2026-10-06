# MODEL_UPGRADE_1002_R4 — measured repair, unresolved formal readiness

**Not recommended for a formal run.** The bounded SOURCE-initialized matrix is complete: historical-retention reference at 100 epochs, unchanged adaptive finalist at 200, and independent TRAIN seed43 confirmation at 150. Typed histories and measured execution repairs are implemented and regression-tested. Modern A/B and finite-primary C retain useful coherence gains, but no tested route satisfies both the requested speed and sustained native fidelity. The four PDFs are reviewed and published, and final preservation checks pass. No formal run was launched or queued.

## Verified evidence and limits

The checkout began at reviewed `c70010d636601857427b0d27ee0c900cc4f42c14` on `codex/phycoflow-upgrade-1002`; legitimate newer work and the user's existing report deletions/R3 YAML edits were preserved. Scientific training remained frozen at `a4981db525a46ef47634aac6f23fd013c588b408`. Diagnostic execution cleanup was applied only after all three lineages returned normally.

The actual stopped R3 child is `abc_upgrade_1002_r3_candidate_5000ep_gpu0/20261005T120002Z_f5e4939d`. Live inspection confirmed no training process despite its stale running status. It contains145 completed epochs and recovery at140; retained ages are1/50/60/110/130/140. Its missing20–35 minimum checkpoint remains unavailable. No stopped-run state was resumed or overwritten. The exact dirty startup source is unavailable: profiling replays its resolved workload on the pinned c70010d oracle. Equal YAML does not establish trajectory identity; retained pilot/formal epoch1 models differ by up to2.303e−4 and calibration scales differ slightly. [Forensic comparison](cases/turbulent_combustion/runs/Test_1002/_audit/R4_campaign/source_and_run_diff.json).

Original LIVE SOURCE remains `tc_gl_rbf_cq_cached_kv_5000ep/20260826T212231Z_7b68c461/checkpoints/last.pt`, SHA256 `03f2f45b966a00850cc45c011b4cf1cd644b9408de355324fec8dc434a84a810`. Contracts remain five fields `[CH4,CO,T,U_1,p]`, sparse T sensors192–384, TRAIN mean/std,100×403 domain, batch32, TRAIN fraction0.15,4,096 coherence queries and **38 updates/1,216 samples per historical epoch**. Model, prior, cached-KV semantics, observation correction and two Euler stages are unchanged. C retains four TRAIN lines from the fixed16-line master and32×128 geometry. TEST remains locked. Seed43 actually began with148/148 LIVE tensors bitwise equal to SOURCE and empty AdamW; tiny frozen B/C calibration differences are recorded rather than called bitwise identity.

## What was wrong, and what caused the regression

**Source-confirmed reporting defects.** The old validation curve combines selector scores (epoch20≈0.927) and single-snapshot native loss (epoch25≈0.381); at coincident50 the selector overwrites native. Those scale changes are not sudden learning improvements. New journals distinguish fixed-native, native-preview, selector and endpoint observations, with epoch/panel/source/noise/normalization identities. Real epoch50 preserves nine separate records. Unrecoverable old coincident native values remain unknown. Adaptive `total` omitted the fidelity gradient and duplicated coherence. Plots now show coherence and update ingredients, parameter pre-clipping norm and actual displacement separately; ConFIG is not assigned a fictitious scalar potential.

**Measured execution cost.** Synchronized R3 timers assign38.93seconds/epoch to C and37.94 to backward/optimizer, versus A5.57,B0.98, live rollout5.51, SOURCE rollout4.92 and native pair2.28. Removing optional reporting alone leaves96.92seconds, so plotting is not the dominant slowdown. Repaired asynchronous host-enqueue component timers are not presented as device attribution. [Runtime evidence](cases/turbulent_combustion/runs/Test_1002/_audit/R4_campaign/runtime_attribution.json).

**Measured local retention effect, limited causal conclusion.** Matched TRAIN counterfactuals at retained50/60/140 give native-pressure/coherence norm ratios15.77/30.73/1.86. At60, removing native pressure improves the local coherence rescore about2.39×; at140 it worsens it. Sampled original actual AdamW displacement dots remain favorable to A/B/C. Thus excess pressure limits progress in the rebound regime, but these probes do not prove universal reversal or explain the unavailable early minimum. Genuine training coherence rebounds independently of the plotting bug, then improves again around118. [Matched counterfactuals](cases/turbulent_combustion/runs/Test_1002/_audit/R4_campaign/controller_counterfactuals.json).

## Speed with unchanged workloads

Controlled short profiles use physical GPU0, RTX6000 Ada UUID `GPU-233fcd85-5c6a-6212-3f44-655252afec70`, logicalcuda0, `phycoflow_env`, identical full exposure, OMP/MKL4 and persistence workers4. Profiler-enabled times are excluded.

| Controlled path | Warm training seconds/epoch | Inclusive short loop seconds/epoch |
|---|---:|---:|
| Historical descriptors/policy |60.098|67.080|
| Stopped R3 workload, pinned oracle |97.242|104.178|
| Modern ABC + historical native0.1/aggregate route |64.270|72.155|
| Modern ABC + epoch PI scalar route |71.224|78.416|

The controlled modern-simple repair is33.91% faster than R3 and1.069× HIST. Adaptive is26.76% faster and1.185× HIST, but misses65seconds: **partial_speed_recovery**. Original HIST's approximately51–52seconds on its former GPU remains an operational reference; device/environment differences do not erase the measured regression.

| Completed lineage | Ordinary training | Diagnostic training | All training mean | Whole-process seconds/epoch |
|---|---:|---:|---:|---:|
| Simple42,100 epochs |63.654|66.164|63.905|65.463|
| Adaptive42,200 epochs |70.822|74.330|71.173|72.620|
| Adaptive43,150 epochs |70.771|74.192|71.113|72.537|

Whole-process non-step residuals are2.381%,1.993% and1.963%, respectively, combining startup/data wait/evaluation/checkpoint/logging/PDF work; these are not isolated monitor fractions. Training includes scheduled in-step diagnostics. GPU audits are charged separately and excluded from lineage wall.

**CPU-envelope limit:** science commands did not supply generic thread overrides; only the live seed43 environment directly proves OMP/MKL/OpenBLAS/NumExpr absent. Closed early environments were not serialized. Long timings are measured, but their roughly26.8% reductions versus short-profile R3 are descriptive rather than isolated identical-CPU comparisons. [Thread provenance](cases/turbulent_combustion/runs/Test_1002/_audit/R4_campaign/runtime_thread_provenance.json).

## Sustained learning and fidelity

All entries below are SOURCE-relative fixed-panel50-epoch window results; complete25/50 windows, raw numerators and slopes are in the [numerical tables](cases/turbulent_combustion/runs/Test_1002/R4_reports/three_actual_lineage_numerical_tables.json). Windows use actual ten-epoch probes (three/five observations), not interpolation. Native ratios aggregate matched raw live/SOURCE numerators; TRAIN epoch coverage is complete.

| Actual lineage | Score | A | Grouped B | Finite-primary C | Native TRAIN | Native VALIDATION25 /50 | Endpoint total50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Simple42,100 |.875601|.877200|1.006931|.742673|1.007412|**1.081831 /1.080574**|1.024110|
| Adaptive42,200 |.911978|.962868|1.014972|.758094|1.007592|**1.052604** /1.040599|1.039728|
| Adaptive43,150 |.894163|.913815|1.006221|.762453|1.010499|**1.071141 /1.076228**|1.017142|

The existing corridors remain total endpoint≤1.05, each field≤1.10 and native mean≤1.05; strict each-field1.05 remains separately reported. Both adaptive lineages preserve the engineering endpoint corridor, but AD200 strict CO/p ratios are1.05975/1.07697 over50 epochs, and seed43 CO is1.05091. Favorable selected110/90 checkpoints are endpoint-selected and do not certify native fidelity.

A and genuine finite C improve. In the mature 50-epoch windows, normalized finite/essential C ratios are .77324/.55565 for AD200 and .77523/.59170 for seed43; essential gains do not substitute for finite evidence. B does not support an unqualified preservation claim: AD200 grouped/pooled total ratios are 1.01497/1.02398, same-frequency 1.04120/1.05568 and cross-frequency .98814/.99079. Seed43 grouped/pooled total 1.00622/.99521 masks same-frequency 1.03942/1.03618 versus cross .97225/.95231. Cached regroupings are grouping sensitivity, not confidence intervals. No new B materiality threshold is introduced.

AD200 retains74.53% and seed15083.80% of their best declared20-epoch rolling gains. Neither triggers the predeclared persistent rebound alarm; near-complete gain erasure is not observed. This does not establish future stability. AD200 coherence slopes are negative over25/50 while native slopes rise. Seed43 coherence/native-TRAIN50 slopes are both small positive (+.0001406/+.0002084 per epoch); native VALIDATION50 slope is negative, while its25 slope is positive. These adverse observations are not excused by a new significance threshold.

Exact clipping is118/7,600 AD200 and45/5,700 seed150; all caps remain unsaturated. Sampled actual AdamW A/B/C dots are negative, but local first-order diagnostics are not population or post-update guarantees. Simple whole-trajectory clipping is unavailable; its matched four-batch probe has0/4 clips. At common age100 both recipes fail native validation, so unequal150/200 horizons do not establish PI-only causality. The optional fixed-native-only route was ineligible because TRAIN endpoint pressure was active; no additional optimizer tournament was opened.

## Changes, mathematics and validation

| Class | Implemented change |
|---|---|
| Report-only | Typed metric identity, honest loss/gradient/component labels, explicit epoch cadence and monitoring-only rebound alarm |
| Exact execution | Profile-only synchronization; shared forward contexts; one scalar backward for a declared weighted-sum objective; batched A eta roots; hoisted B indices/deferred scalars; bounded target persistence reuse with fresh generated diagrams |
| Exact execution, after matrix | Remove redundant25-epoch component hook in R4; reuse cached simple/fidelity gradients; skip zero-coefficient constraints; pack diagnostic Gram/displacement products; retain legacy path and zero-weight fallback |
| Optimizer change | One epoch PI version, positive scalar ABC/fidelity route, native floor and fixed TRAIN-calibrated caps |
| Estimator approximation | None adopted; full matched SOURCE teacher retained |

Adaptive pre-clipping direction is `grad(sum_k s_k L_k + sum_j a_j,e g_j)`. Native risk is `(D_live−D_source)/s_native−.02`; endpoint allowances remain.05. Coefficients are held for all38 updates:

```text
m_e = .8*m_(e−1) + .2*mean_TRAIN(g_e)
a_(e+1) = clip(a_e + .01*m_e + .02*(m_e−m_(e−1)), floor, cap)
```

Projected coefficients themselves store the integral state; there is no hidden windup reservoir or old per-update rho term. Native floor inherits raw weight.1; other floors are zero. Caps use retained R3 SOURCE TRAIN gradient calibration, not validation fitting, and do not bound future gradient norms. Actual adaptive direction scale is1.0. Partial-epoch sums/counts, private RNG and exposure contracts recover exactly; incompatible resumes fail. Simple retains the historical two-objective aligned-sum/ConFIG conflict policy, not an unconditional new scalar optimizer.

A/B GPU value/gradient checks are bitwise equal; scalar mixed real/complex/unused AdamW delta≤4.77e−7. Small-Gram ConFIG was slower and was not adopted. SOURCE subsampling was excluded: ten-of38 conditional raw-risk estimates can be unbiased while producing18–27% false-feasible signals; projected control and future trajectories need not be unbiased.

The late copied-TRAIN audit measures32 redundant autograd requests taking6.7707seconds; compact reuse reduces12→6 requests and instrumented single-batch optimizer time2.4736→1.1975seconds. CPU gradient/model/AdamW/RNG proofs are bitwise; strict GPU bitwise checks fail, as does an unchanged-code repeat. Prototype model delta1.02e−7 satisfies the existing1e−6 tolerance, which was not relaxed. The responsible kernel remains unisolated. Those failures are retained. **Post-matrix cleanup has no measured full-epoch speed result** and is not used to claim65seconds. Production zero/nonfinite-weight fallback and final regression results are recorded separately.

Code maps are `training/{post_training,gradient_balance,fidelity_controller,checkpointing,monitoring,preview,coherence_history,rebound_monitor}.py`, A `components/{tail_risk,cross_copula}.py`, B `covariance_blocks.py`, `config/validate.py` and `scripts/training/run_upgrade_1002_r4.py`. The frozen matrix regression passed 726 tests with 9 expected skips. Applied cleanup code `de58583` passed all 39 focused production tests, then the [full CPU regression](cases/turbulent_combustion/runs/Test_1002/R4_tests/post_adoption_full_cpu_retry_01/receipt.json): **744 passed, 9 expected skips, zero failures**, in 837.00 seconds. All 252 Python-file hashes remained unchanged; strict GPU failure receipts were preserved, and 266 audited shared-cache files had no metadata changes. An earlier SIGTERM interruption is retained as incomplete evidence, not a pass. These software checks do not qualify the scientific trajectory.

## Formal draft, artifacts and preservation

The [actual SOURCE5000 draft](cases/turbulent_combustion/runs/Test_1002/R4_configs/abc_upgrade_1002_r4_5000ep_not_recommended.yaml) is **not_recommended**. Only horizon, future output name and archive milestones after 200 differ from the tested adaptive recipe; the first 200 scientific schedule is identical. CPU configuration, metadata, TRAIN loader construction and retained TRAIN-normalizer validation passed on applied code `de58583`. General CLI validation was skipped because its validator reads a TEST scalar; the safe adapter reads no field payload. The conditional two-epoch SOURCE sibling was skipped because the long finalist fails intended checks. Neither a 5,000-epoch trajectory nor whole-epoch timing of the late cleanup is certified.

[Exact user launch/clean-stop/resume commands](cases/turbulent_combustion/runs/Test_1002/R4_reports/tooling/actual_formal_draft_user_commands_NOT_RECOMMENDED.md) unset generic thread overrides and retain tested persistence workers4/LRU4096. They are documentation only. The future outside-Test output directory does not exist. Clean stopping uses the unchanged5000 configuration with `--max-steps7600` from fresh SOURCE; resume uses that same actual child/config/code, never an R3 or200-epoch pilot. Resume max-steps counts additional updates. Ctrl-C is not a graceful checkpoint handler. Recovery every10 epochs, bounded mature/feasible/exploratory archives, external original SOURCE fallback, PDF25 and previews100 are documented. Cache entry limits do not certify5000-epoch RAM use. At measured inclusive72.62seconds/epoch,5000 epochs project about100.9hours; this is an extrapolation, not readiness evidence.

Final campaign use is **460.263 epoch exposures,9.453 GPU0 hours and1,011.23 offline GPU seconds**. Every new scientific lineage is below250; initial profiles total10. All70 protected SOURCE/HIST/R3 content/metadata checks pass. The complete recursive scientific checkpoint inventory covers43 regular files/three symlinks,3.722GB unique payloads and exactly three extra evidence copies. No scientific checkpoint was deleted. [Final budget/preservation receipt](cases/turbulent_combustion/runs/Test_1002/_audit/R4_campaign/final_campaign_budget_reconciliation.json).

Four PDF masters provide12 pages, with raw CSV/JSON retained:

- [Runtime comparison —2 pages](cases/turbulent_combustion/runs/Test_1002/R4_reports/runtime_comparison.pdf): controlled timings and mature ordinary/diagnostic/inclusive costs.
- [Corrected histories/controller —3 pages](cases/turbulent_combustion/runs/Test_1002/R4_reports/corrected_losses_and_controller.pdf): separated metric identities and scoped pressure/displacement evidence.
- [Coherence/fidelity —3 pages](cases/turbulent_combustion/runs/Test_1002/R4_reports/coherence_and_fidelity.pdf): all three lineages, finite/essential C, grouped/pooled B and unchanged fidelity corridors.
- [Fields —4 pages](cases/turbulent_combustion/runs/Test_1002/R4_reports/fields.pdf): SOURCE0/AD50 early minimum/AD100 transition/AD200 mature, all five GT/SOURCE/child fields and signed errors on shared scales. Values are normalizer-decoded; physical units are undocumented.

Only exact disposable new profiler/fixture/QA intermediates were removed with receipts. Final cleanup removed 822 fixture/approved-QA images (151.28 MB), preserved all 5,847 other files/symlinks in that inventory, and reverified all 70 protected historical/SOURCE checks and all 46 scientific checkpoint files/symlinks. Numerical artifacts, PDF masters, model/config/code evidence and historical roots remain intact. Early shared KeOps import writes are documented: four new payloads were relocated with compatibility links; six preexisting lookup dictionaries were retained rather than overwritten. Final branch synchronization and artifact checks complete this bounded negative outcome; no fidelity limit, descriptor scope or epoch exposure was loosened to claim success.
