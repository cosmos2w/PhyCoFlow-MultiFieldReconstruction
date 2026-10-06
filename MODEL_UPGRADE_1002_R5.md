# MODEL_UPGRADE_1002_R5 — usable scalar refinement, unresolved native fidelity

The bounded R5 campaign is complete: native-only 100, matched scalar F150/S150, unchanged S continuation to 200, fresh SOURCE seed43 to 100, and one SOURCE-initialized two-epoch sibling with its own recovery. Modern A and finite-primary C retain useful gains at mature epochs. The cheap SOURCE parameter penalty improves the F/S native-validation trade-off at 150, but neither it nor SOURCE/child interpolation preserves the declared native fidelity across the required monitors. The tested 5,000-epoch profile is technically available as an experimental configuration; it is not recommended under the existing native-fidelity requirement. No formal run was launched or queued.

## Verified scope and what earlier rounds established

The reviewed starting HEAD was `07ad86a6ea49f047e06ef78956bec546c16cba06`. Its actual cleanup parent is `de5858306ef3a65521c03339ef8e8eed0a494be6`; the plan's expanded parent hash was inaccurate. All R5 science and final scalar benchmarks ran on frozen `d9a1ce1bba600189e2db583627fcf3f4174305a7`. The final SOURCE sibling ran on `4fc07a8f065f1df1cc1baeb9dfbb85630cd5f03d`, whose changes add a bounded invocation validation context, focused tests and the public profile/calibration; they do not change the scalar training math. Pre-existing user deletions and the modified R3 configuration were preserved and excluded from commits.

R1 introduced useful descriptor versions and recovery but made native loss monitor-only and stopped integrated studies at20 epochs. R2 restored native gradients and mature comparison, while extra teacher/gradient work did not identify held-out fidelity causes. R3's finite-primary C made shape improvement more interpretable, but provisional formal preparation was not stable-trajectory evidence. R4 repaired execution and typed histories, reached roughly64 seconds per simple historical epoch and avoided near-total coherence rebound; its later cleanup had not been benchmarked, native-only continuation was missing, and PI feedback did not establish native generalization. Those historical conclusions and failed fidelity measurements remain unchanged. The [R4 report](MODEL_UPGRADE_1002_R4.md) remains the prior evidence, rather than being reclassified under new limits.

Original LIVE SOURCE is `tc_gl_rbf_cq_cached_kv_5000ep/20260826T212231Z_7b68c461/checkpoints/last.pt`, SHA256 `03f2f45b966a00850cc45c011b4cf1cd644b9408de355324fec8dc434a84a810`. Every independent N/F/S/seed43/sibling arm initializes all 148 LIVE tensors exactly from it with a fresh optimizer. Its 145 trainable parameter tensors have identity `8671099362c51eeb4e07f38e667c5923669f01fdf93345fff43ff5bf69b1141e`; three frozen tensors remain excluded from parameter retention. Own continuations restore their optimizer, sampler, Python/NumPy/Torch/CUDA RNG and original SOURCE reference.

The fields remain CH4/CO/T/U_1/p, chronological8,000/1,000/1,000 splits, TRAIN mean/std normalization, T-only sensors192–384, native100×403 geometry, batch32, TRAIN fraction0.15,4,096 shared coherence queries, two Euler stages, observation correction, cached K/V, the original spatial prior and LIVE evaluation. One historical epoch remains 38 updates/1,216 sampled exposures. TEST values were used zero times. Fixed development selection64 and native TRAIN32/validation32 panels reuse chronological evidence; three/five sparse scheduled points in final25/50 windows are correlated, not independent replications.

The user overrode the plan's GPU0 with physical GPU1 (`GPU-3ceda40c-fd5c-4b88-6c47-b3301711571e`) for matched N/F/S, timing, selected continuation and sibling. After authorization for GPUs0/2 assistance, fresh seed43 used GPU2 (`GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39`) and the locked selected200 offline audit used GPU0 (`GPU-233fcd85-5c6a-6212-3f44-655252afec70`). Each is logical `cuda:0`; GPU2 is a different-device seed confirmation, not a same-device speed comparison. The common envelope is `phycoflow_env`, four persistence/OMP/MKL/OpenBLAS/NumExpr workers/threads and reference cache4096. CPU checks hide CUDA. The unchanged continuation and independent confirmation ran concurrently after the actual150 decision, with combined device time charged once per attempt.

## Implemented objective and one frozen coefficient

For normalized field u, unchanged spatial-prior draw z and one artificial generative coordinate t_i~Uniform(0,1) per sample, the native adapter uses x_t=(1−t)z+tu and target velocity u−z:

\[
D(\theta)=\mathbb E\frac{1}{BQn_f}\sum_{i,q,c}\left[v_\theta(t_i,x_{t_i};y,\mathrm{coords})_{iqc}-(u-z)_{iqc}\right]^2,
\quad B=32,\ Q=4096,\ n_f=5.
\]

Each query microbatch contributes its squared-error sum divided by the same full tensor element count. This t is not combustion time; the configured `sigma_min` does not add a term to this implemented bridge. Native risk and reconstructed endpoint error \(E_c=\mathbb E\|G_\theta(z,y)^c-u^c\|_{\rm model}^2\) are distinct evaluations. A passing endpoint does not replace a failed native monitor.

\[
\mathcal C(\theta)=s_A L_A(\theta)+s_B L_B(\theta)+s_C L_C(\theta),\qquad
J_N=0.1D,\quad J_F=0.1D+\mathcal C,
\]
\[
\Omega(\theta,\theta_0)=\tfrac12\sum_{p\in\mathcal P_{\rm train}}\|\theta_p-\theta_{0,p}\|_2^2,\qquad
J_S=J_F+\lambda_{\rm SP}\Omega.
\]

The unchanged SOURCE-fitted training scales are approximately1/1.232288/0.01407067; evaluation reports unweighted within-definition SOURCE ratios, not substituted training scales. Independent source fits have small disclosed floating roundoff. Modern A=`marginal_copula_v2`, B=`second_order_blocks_v4` and C=`finite_primary_v1` are unchanged. B grouped32 is primary and pooled64 covariance is a separate estimator. C samples four lines in TRAIN and evaluates all16; normalized finite/essential diagnostics have direct weights0/0 and finite-primary coefficients0.9/0.1, while the active composite has weight1. Ratios of that composite are not weighted averages of component ratios with different SOURCE denominators. Three genuinely zero-SOURCE essential terms remain raw0 with undefined ratios.

All arms retain AdamW lr5e−5, decay1e−6 and clip1. N optimizes no coherence, teacher rollout, controller or parameter penalty; scheduled descriptors remain available. F/S use one ordinary scalar backward. R4 Simple used aligned-sum/ConFIG, so R5 F is a declared optimizer-policy comparison, not asserted trajectory-equivalence. SOURCE-centered Omega uses native complex modulus and excludes buffers/frozen tensors. Its detached original SOURCE snapshot is captured once before child recovery, is not EMA-updated, and requires no additional ordinary forward or parameter-gradient traversal. Its value/gradient are zero at SOURCE and its restoring gradient is lambda*(theta−theta0).

The sole coefficient was frozen from eight predeclared TRAIN32 probes at SOURCE and the predeclared R4 Simple42 epoch100 displacement:

\[
\Delta_{100}=3.7823128074994115,\qquad
G_D=\operatorname{median}\|\nabla_\theta(0.1\ell_{\rm native})\|_2=0.1009768145498743,
\qquad\lambda_{\rm SP}=G_D/\max(\Delta_{100},10^{-12})=0.026697108274509104.
\]

No optimizer update, validation fit, coefficient sweep or subsequent retune occurred. The [public frozen calibration](cases/turbulent_combustion/configs/posttrain/calibration/r5_source_parameter_l2_v1.json) is byte-identical to the tested artifact: raw file SHA256 `9672a6ba42f678fd77b91f199f0bd5bfdfeb9c52369213dbcdcfaac2e3100493`; its canonical JSON identity is `7e0d526e308147910ec3eef6053923debeefc931d16bce11a05281f0e4f2d559`. These identify parameter calibration, not each run's separate coherence calibration. Probe IDs/seeds/layout and reference hashes are retained. This is a scale-calibrated engineering prior, not an optimal value or physical-law/KL guarantee.

## Complete-epoch speed and actual validation

| Measurement on physical GPU1 | Complete training timing | Scope |
|---|---:|---|
| Historical cleanup oracle,07ad86a | 60.676131s | One complete warmup-containing epoch |
| Scalar cleanup oracle,07ad86a | 64.070569s | Same workload/envelope;1.055944× historical |
| Final D9 F profile | Ordinary median64.257669s;63.914690–64.600647 | Two ordinary observations including warmup; third epoch diagnostic |
| Final D9 S profile | Ordinary median64.387751s;64.343867–64.431634 | Matched F inputs; incremental+0.202438% |
| Complete F150 | Median63.946120s;63.205982–65.617621 | Mean64.036833s; inclusive65.399174s/epoch |
| Complete S150 | Median65.001724s;64.288110–68.764598 | Mean65.176003s: marginal preferred65s result |
| Own S150→200 | Median64.563904s;61.939848–66.109627 |50 NEW epochs; inclusive3,284.789571s |

The controlled incremental penalty cost meets the<3% target in a small sample; it is not a precise general estimate. Profile diagnostic calls take1.215421s(F)/1.201146s(S). Actual diagnostic records, rather than the10-epoch monitor cadence, partition timing. Science diagnostics occur every25 epochs. Whole S200 training-step mean is64.929469s; its two disjoint inclusive attempts total13,268.857909s,66.344290s/epoch. Loader work, evaluation, serialization, startup and finalization remain outside the step timer where appropriate, and inclusive residuals are not labeled diagnostic-only. GPU2 seed43 median64.768059s is separately recorded and not pooled into the controlled GPU1 comparison. Native-only timing is not an ABC speed benchmark.

Frozen D9 passed 796 CPU tests with 9 explicit CUDA/opt-in skips and no failures, in 904.247675s. The interrupted earlier suite exceeded the multiprocessing socket-path limit and is preserved as an invalid full-suite attempt; a short temporary path resolved it without source changes. Final startup-guard/config/retention/native focused checks passed 126 with one CUDA replay skip. Scalar value/gradient, complex retention, frozen SOURCE, default-off behavior, RNG/recovery, interpolation, typed histories and old descriptor oracles remain checked. Bare long Test configs still reject; explicit absolute until1/2 permits the unchanged5,000 horizon only for the bounded sibling. Normal CLI metadata validation attempted zero `/fields` reads and initialized no CUDA.

## Matched learning evidence and causes supported by it

All quantities below are SOURCE-relative. Final50 endpoint/coherence/native windows pool raw numerators and matched SOURCE denominators before division; VAL25 uses the final25 window. The exact windows contain3/5 scheduled observations at10-epoch cadence.

| Stage | Native TRAIN50 | Native VAL25 / VAL50 | A50 | B32-50 | C50 | Endpoint50 |
|---|---:|---:|---:|---:|---:|---:|
| N100 | 0.970697 | 1.013387 / 0.998260 | 0.974037 | 0.994713 | 1.000549 | 0.996491 |
| F100 | 1.005486 | 1.074791 / 1.078245 | 0.849426 | 1.001028 | 0.742034 | 1.020359 |
| S100 | 1.014645 | 1.070834 / 1.069836 | 0.913332 | 1.001871 | 0.812470 | 1.019817 |
| F150 | 1.025452 | 1.085660 / 1.084998 | 0.855130 | 0.993666 | 0.727588 | 1.017529 |
| S150 | 1.028821 | 1.063407 / 1.063722 | 0.901729 | 1.003816 | 0.810619 | 1.015619 |
| S200 | 1.025016 | 1.069095 / 1.070667 | 0.926984 | 0.992482 | 0.808388 | 1.023062 |
| S43-100 | 1.009819 | 1.070220 / 1.068301 | 0.903492 | 1.002788 | 0.814492 | 1.013744 |

At common100, native-only continuation reduces TRAIN risk and does not reproduce the F/S validation deterioration. The additional drift is therefore associated with structural fine-tuning under this matched protocol, rather than an inevitable consequence of continuing the native objective. This inference is limited to100: N150/200 was not run. It does not identify a universal physical cause or certify chronological generalization. N/F/S first-batch sample/query/observation/native-seed identities match at all 100 recorded epochs; F/S match through150. Those records are not an audit of every GPU operation.

At150, S improves native validation relative to F by0.021276 SOURCE-ratio points in the final50 window and retains meaningful A/finite-C gains, although weaker than F. Both still fail1.05. Root therefore locked unchanged S as the less harmful useful stabilization check, not an endorsement. Its own recovery adds50 epochs; independent seed43 starts original SOURCE and stops at 100, with no extension or seed selection. A gains persist at7.3%(S200) and9.7%(seed43_100), and finite-primary C gains at19.2%/18.6%. A weakens modestly after150; near-total coherence rebound is not observed. Native validation remains unsupported in both seeds.

B remains mixed, including opposite same/cross trends. These final50 component comparisons use their declared estimator and normalization:

| Stage | B grouped same / cross | B pooled same / cross | C normalized finite / essential |
|---|---:|---:|---:|
| F150 | 1.031398 / 0.955053 | 1.027295 / 0.926351 | 0.748216 / 0.451952 |
| S150 | 1.026301 / 0.980806 | 1.020165 / 0.970842 | 0.818609 / 0.703846 |
| S200 | 1.013626 / 0.970843 | 1.010976 / 0.972035 | 0.817553 / 0.685916 |
| S43-100 | 1.023557 / 0.981534 | 1.026844 / 0.987324 | 0.822733 / 0.704372 |

All listed mature endpoint totals meet1.05 and all fields meet engineering1.10. CO fails the separately retained strict field1.05 throughout the coherence arms; at S200 final50 CO=1.059669 and pressure=1.025997, and at seed43 final50 CO=1.058812/pressure=1.003195. Isolated snapshots have their own field values and do not replace windows. Near-one total B does not establish same-band preservation, while a small B regression is not elevated into a new global prohibition.

CPU-only retained tensor evidence further rejects a simple displacement-magnitude explanation: N100 displacement2.917577 has stable mature native validation; F150 displacement4.468719 has worse validation; S150/S200/seed43_100 displacements0.431683/0.422881/0.428646 remain small but fail native fidelity. S200 post-checkpoint Omega0.08941408 gives penalty0.00238710 and restoring norm0.111805 times frozen G_D. Logged epoch means are pre-update states and are not equated with post-checkpoint values. Parameter proximity alone does not guarantee the held-out native function. No stronger coefficient/controller is introduced in response.

Root locked the exact S epoch200 archive SHA256 `e031e9dc317053ce1c3889d75fa4dc69720d6c32c462924548ad6835a7bde3a4`, before one offline primary/alternate check on GPU0. Primary native validation1.078699 and alternate1.076684 both fail1.05; alternate endpoint1.021411 passes engineering limits. Alternate A/B/C=0.917741/0.946278/0.815474. There was no reselection or repeated alternate. Its26.262157s consumed zero training epochs. Fresh full-grid SOURCE replay differs by at most7.15e−7 in normalized units, inside the existing output tolerance2e−5/2e−6; GT/query/observation coordinates and normalizer identity match exactly, permitting reuse of original GT/SOURCE presentation arrays.

## SOURCE interpolation: available inference snapshot, failed fidelity confirmation

The only children were predeclared R4 Simple42 epoch100 (checkpoint `0dc26beed69b26651c86002182adbcff8c0fddc050da0101e95a04ec7835fff3`) and Adaptive42 epoch200 (`3b9482cf54f4b9a6943f082f0f85f38efd126aa2aa1659d91c7d4138f236786b`), each at global alpha0/0.25/0.5/0.75/1. Nine unique states reuse SOURCE once; no additional alpha search occurred. Only trainable LIVE parameters blend; frozen state, normalizer, prior and semantic identities stay fixed. EMA, Adam, controllers and calibration are not averaged.

Simple100 alpha0.75 was locked from the primary development panel: endpoint1.013513, native TRAIN/VAL0.962937/1.041006, A/B/C0.909694/0.973852/0.809479. Its sole alternate confirms endpoint1.006541 and A/B/C0.895590/0.989968/0.821338, but native VAL1.077517 fails. The primary-eligible inference-only export remains available with that failure explicit; it carries no native-fidelity endorsement and is rejected for training initialization/resume. Export SHA256 is `a71e96ad6b66e81e73253a945d34365f95d1fb5daf3ff54483d143d05daf39d9`.

Actual normal LIVE `evaluate-run` succeeded for validationcase0/full40,300 queries/Euler2:22.894118ms,384,069,120bytes peak, normalized MSE0.05455209. Three synchronized SOURCE/blend repetitions give medians22.494284/22.462483ms (ratio0.998586), supporting comparable cost in that small sample. Interpolation diagnostics took66.071108s with zero training epochs. A single blended model is not a two-model ensemble, training convergence, successful native confirmation or a formal initializer.

## Tested final configuration and honest readiness

The [SOURCE-initialized public profile](cases/turbulent_combustion/configs/posttrain/abc_upgrade_1002_r5_5000ep.yaml) preserves seed42, the scalar S objective, fixed coefficient, optimizer, model/rollout/normalizer and early schedule. Relative dependency paths do not rewrite historical absolute paths inside SOURCE metadata. Its tested companion differs only in output, keeps the configured5,000 horizon and executes absolute until1, then its own until2. One child returns normally after 2 epochs/76 updates on4fc07a8; SOURCE148 initialization, all 145 optimizer states, private/public RNG, calibration/original SOURCE reference and the epoch001 archive are verified. Actual first-batch identities at1/2 match the pilot. This physically tests direct bounded CLI recovery; the full segmented GPU wrapper and5,000-epoch stability were not executed.

An optional comparison of independent learned epoch001 parameter trajectories failed when output-replay tolerances were applied to trained weights; this remains a failed diagnostic, not a passed GPU test or an established parameter-trajectory gate. The [CPU forensic audit](cases/turbulent_combustion/runs/Test_1002/R5_reports/sibling_epoch001_parameter_forensics/report.md) finds comparable nonidentity between two unchanged-d9 runs: difference norm0.0990403/max0.000411547 versus sibling/science norm0.0957237/max0.000475582, with 138 trainable tensors outside the borrowed tolerance in both comparisons. SOURCE initialization, frozen tensors, saved RNG states and recorded first-batch identities match. Independent SOURCE family-scale measurements have tiny nonzero differences. No material objective, optimizer or descriptor change was found in the final startup-guard patch; the cause of trajectory differences is not isolated. These checks support the recorded startup/recovery contracts without establishing exact learned-parameter reproducibility.

| Readiness field | Result | Evidence boundary |
|---|---|---|
| implementation_ready | true | Math/source/config/recovery tests, with explicit skips |
| execution_ready | true | Complete historical epochs and actual SOURCE1→own2 recovery |
| inference_candidate_available | true | Tested alpha0.75 export; failed alternate native check disclosed |
| sustained_training_supported | true for A/finite-C gains | S200 and independent seed43_100; mixed B |
| native_fidelity_supported | false | Both mature validation windows and locked alternates fail |
| all_three_families_improved | false | B grouped/pooled and same/cross results remain mixed |
| formal_recommendation | technical experimental candidate only | Existing native-fidelity condition remains unmet |

[README's R5 commands](README.md#r5-source-parameter-retention) provide environment/cache setup, physicalGPU1 launch, clean stop, unchanged-config resume and explicit saved-archive evaluation through the standard public module with25-epoch segments. A stop is checked after the current segment and can take up to25 epochs. The inference-only export has its separately tested single-case full-grid evaluation command. No campaign-private helper is required in the formal user flow. No formal launch, queue,5,000-epoch stability claim or native-fidelity recommendation follows from the sibling smoke.

## Figures, implementation map and artifact preservation

The presentation consists of three final presentation PDF masters/eight pages. Field panels are TRAIN-normalizer-decoded original dataset values; their SI units are undocumented. The same fixed validationcase0, observations, prior draw, native grid and common GT/SOURCE scales support SOURCE/interpolation/S200 comparisons. Temporary QA rasters are removed after inspection; no PNG/SVG presentation copies are retained.

- [Runtime and interpolation —2 pages](cases/turbulent_combustion/runs/Test_1002/R5_reports/runtime_and_tradeoff.pdf): observed complete-epoch ranges and disjoint inclusive attempts by physical device/code, plus locked primary/alternate interpolation and its failed native check.
- [Training and fidelity —3 pages](cases/turbulent_combustion/runs/Test_1002/R5_reports/training_and_fidelity.pdf): separate TRAIN objectives/penalty, fixed native TRAIN/validation windows, endpoint fields and mature A/B/C histories.
- [Fields and coherence —3 pages](cases/turbulent_combustion/runs/Test_1002/R5_reports/fields_and_coherence.pdf): matched decoded fields and signed errors, grouped/pooled B and finite/essential C with inactive terms kept explicit.

| File/symbol under src/phycoflow_reconstruction | Responsibility |
|---|---|
| training/parameter_retention.py:SourceParameterRetention | Frozen TRAIN coefficient/layout, complex SOURCE-centered penalty and resume identity |
| training/post_training.py:run_post_training | Scalar N/F/S, private streams, typed histories and full recovery |
| training/parameter_interpolation.py | Compatible LIVE blending and inference-only training rejection |
| training/source.py / training/segmented.py / training/update_budget.py | SOURCE/recovery identity and exact epoch boundaries |
| config/validate.py / cli.py / data/validation.py | Default-off contracts, bounded sibling context and structural validation without TEST values |
| evaluation/checkpoint.py | Provenance-bound LIVE inference runtime |

All 550 scientific exposures, 8 scratch timing epochs and the 2-epoch sibling consume exactly 560 exposures; every lineage stays below 250. Combined all-device inclusive GPU time is31150.483114s (8.652912h), below 14h. Offline/interpolation caps remain satisfied. The end inventory verifies all 1,289 protected SOURCE/historical files unchanged. The [final boundary audit](cases/turbulent_combustion/runs/Test_1002/R5_reports/final_campaign_boundary_audit.json) verifies 16 normally returned attempts, no active R5 training, no GPU compute owners and no formal output. The [PDF visual QA receipt](cases/turbulent_combustion/runs/Test_1002/R5_reports/final_PDF_visual_QA_and_cleanup_receipt.json) records inspection of all 8 pages and immediate removal of exactly 8 temporary rasters after layout corrections; numerical bindings remain unchanged. Prior PDF bytes are retained only as layout-QA evidence. Numerical tables, raw sums/denominators, own family calibrations, checkpoint identities, runtime scopes and receipts remain under `Test_1002/R5_*`. The [full trade-off table](cases/turbulent_combustion/runs/Test_1002/R5_reports/final_tradeoff_tables.md) and [root readiness review](cases/turbulent_combustion/runs/Test_1002/R5_reports/final_completion_readiness_root_review.json) retain details beyond this compact report. Only disposable new intermediates were removed; no scientific checkpoint, SOURCE/R1–R4 evidence, code or numerical array was deleted.

The small penalty is motivated by [Li, Grandvalet and Davoine's L2-SP study](https://proceedings.mlr.press/v80/li18a.html), and interpolation by [Wortsman et al.,WiSE-FT](https://arxiv.org/abs/2109.01903), in different tasks. [Flow Matching](https://arxiv.org/abs/2210.02747) and [PyTorch numerical-accuracy guidance](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html) support the distinction between native vector-field risk, endpoint generation and numerical equivalence. They supply no transferred combustion, KL or long-horizon guarantee; the campaign's failed native checks determine that boundary.
