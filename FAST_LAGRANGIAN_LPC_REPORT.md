# Fast Lagrangian refinement and LPC ordering: measured bounded study

Study dates: 2026-10-22–23, America/New_York. Branch: `cosmos2w/fast-lagrangian-lpc`. Payload root throughout this report: `/data2/wanglz/PhyCoFlow-MultiFieldReconstruction/experiments/LPCAccel_20261022_01a16674` (called ROOT below). The original no-long-launch boundary was superseded by the user's explicit three-GPU 100-epoch authorization; all three eligible Goal-1 arms have finished and are assessed. Subsequent work uses physical GPU 0. No lineage is automatically extended beyond 100 epochs.

## Result and scope

Complete-filtration P3 sampling reduced matched ten-epoch user-command time by **18.3%** with dense ConFIG. Scalarization saved another **34.8%** against P3 ConFIG; scalar topology-family importance sampling saved **16.4%** against dense-family scalar training. The combined 55.5% short-run saving changes both descriptor work and nonlinear optimization behavior. These are cost measurements, not a claim of equal optimization trajectories or retained-quality speedup.

All three independent Goal-1 arms completed 100 reporting epochs from the original pretrained SOURCE. Their final uniform-128 full-descriptor audit passes the +5% aggregate/every-field endpoint guards and reduces T by 46–54%. Complete routine histories give a more qualified result: only G1-R reaches the common useful target at two consecutive checks, and it later regresses. No arm demonstrates stable target retention through epoch 100. The efficient scalar route shows useful bounded learning, with mixed stability and a selected-checkpoint pressure-field miss on the broader panel. It is a candidate for further controlled study, not a demonstrated universal replacement for dense ConFIG.

Signed ConFIG coefficient recovery reconstructs the FP64 oracle, but periodic reuse fails the declared staleness probe at both tested cadences. The periodic timing/100-epoch arm remains excluded; no replacement arm was added. Goal 1 is complete. The subsequent read-only Goal-2 gradient diagnostic and real SIGINT/own-state recovery smoke pass their declared gates. The single authorized ordered 100-epoch arm is running on physical GPU 0; its quality result is pending completion and full evaluation.

**TEST lock and disclosed exception:** no TEST model evaluation, metric, selection, or estimator tuning was performed. An earlier readiness attempt mistakenly invoked the existing dataset validator, which structurally read one final HDF5 scalar in the TEST frame range; the scalar value was not retained. `manifests/test_access_exception.json` records this incident. Later validation is config-only, and the broader/native-map/gradient evaluators disable temporal derivatives where necessary. This study cannot be described as having zero TEST access.

## Verified baseline and immutable inputs

Remote `main` was verified with `git ls-remote` at `716f48397dbd40301ecd9d72c09a24a405936ca7`. The starting working HEAD is `a1c02200c06e48c15e528d7c0f10636bba976f54`; its common ancestor with the reviewed merge is `04b38070516947514dbdbd732900740b8abc8a49`. The material tree difference is the newer Run 11S summed-coherence configuration, its external-policy validator correction, and its regression test. That work was retained. The older local `main` reference was not used to reset the working tree.

The protocol anchor is `cases/turbulent_combustion/runs/_MultiTask/11f_lagrangian_lr_warmup_dense_fast_100ep/20261017T170131Z_615167aa/resolved_config.yaml`. It resolves to 1,000 historical reporting epochs, despite its directory label. The historical optimizer cursor is 38,000 and the epoch exposure is 38 minibatches of 32 samples, or 1,216 samples per reporting epoch. The TRAIN split contains 8,000 snapshots. A reporting epoch is not a complete dataset pass.

Every independent scientific arm starts from `/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction/cases/turbulent_combustion/runs/tc_gl_rbf_cq_cached_kv_5000ep/20260826T212231Z_7b68c461/checkpoints/last.pt`, SHA256 `03f2f45b966a00850cc45c011b4cf1cd644b9408de355324fec8dc434a84a810`. It is the original pretrained SOURCE, not Run 11f's post-trained state. The resolved anchor SHA256 is `7709a76f3a265c37d3522c36f9b14ff3fa8d9f3826745e95f6b32572ca348771`.

| Contract | Verified value |
|---|---|
| Fields and normalization | CH4, CO, T, U_1, p; original TRAIN mean/std artifact |
| Model and reconstruction | GL_rbf_ENH_CQ / `gl_rbf_cq`, cached condition K/V, top-k 32, two Euler steps, same endpoint-smooth correction and final clamp |
| Training exposure | Batch 32; 4,096 fixed shared queries; 38 minibatches/reporting epoch |
| Topology | CO and T; 32×128; H0/H1; sublevel and superlevel; one CO+T joint line; 32 distance projections; original essential handling and standardization |
| Model optimizer | AdamW, LR 5e-5, weight decay 1e-6, clip 1; original warm-up from 5e-6 across 76 minibatches/two epochs |
| Dense fidelity | Every batch measures all five endpoint MSEs and applies active primal fidelity contributions; successful model action precedes projected dual action; dual LR 0.01; +5% training bounds |
| Calibration | Full descriptors; eight TRAIN batches; two generative draws; seed 700103; fixed C0 and field baselines |
| Timing validation | Original fixed 32-snapshot validation panel, full descriptors; same sensor/query/noise contract across arms |
| TEST | No evaluation/selection; one incidental validator scalar read, disclosed above |

The fixed C0 is `0.022609325824305415`. T0–T3 reproduce the saved historical Run 11f reference object exactly, including all field denominators, floors and bounds; `manifests/protocol_identity.json` retains that comparison and the numerical source-file hashes. The trials executed this study working tree; the delivered branch commit records those files.

The installed environment is `phycoflow_env`: Torch 2.5.1+cu121, TorchJD 0.17.1, cooper-optim 1.0.1, GUDHI 3.13.0, PyKeOps 2.3, NumPy 2.0.1, and h5py 3.14.0. Physical GPU 1 is NVIDIA RTX 6000 Ada, UUID `GPU-3ceda40c-fd5c-4b88-6c47-b3301711571e`. It was idle before reservation; unrelated jobs on GPUs 0 and 2 were left running. CPU threads and persistence workers are fixed at four. The GPU is exclusive to each trial; the host CPU/storage are shared with those existing jobs.

## What is exact, estimated, approximate, or changed

A is the existing raw-state distribution discrepancy. B retains magnitude-squared/squared-band-energy calculations, including a fourth-order cross-band contribution in the original fields. T is the existing persistence-v3 shape discrepancy. These are state, scale, and shape **proxies**; they are not a pure Local/Pair/Collective partition or a decomposition of field-law KL. `T` is not synonymous with the theoretical `C_LPC` sector.

For complete filtration q, write `T = sum_q w_q t_q`. Each `t_q` keeps all homology dimensions and finite/essential operations. The four unary units each carry the original self-category mean weight; each of the two joint units additionally carries its fixed positive-line minimum-direction factor. They are not six equally weighted terms.

P3 samples one polarity with probability 1/2, shared by the unary and joint categories. Its Horvitz–Thompson coefficients reproduce the original two-sign category means; there is no second factor of two applied to an already reduced sampled mean. S2 samples one of four unary units and one of two joint units, applying each unit's original weight divided by its inclusion probability. Generated and reference sides always use identical sampled identities. Generated pairings are recomputed; target pairing reuse uses the existing content-keyed primitive. The historical hash-bound topology implementation is untouched.

Paired dense TRAIN references are required; omitting a native model loss does not make this method data-free.

The exact finite-support expectation of loss and input gradient is checked against the original full objective. On the real SOURCE TRAIN endpoint, P3 mean loss relative error is 5.43e-8 and gradient relative error 8.67e-6; S2 gives 2.71e-8 and 7.81e-6. These FP32 reductions are numerical checks on a fixed pairing stratum, not bitwise identities or guarantees at nonsmooth pairing changes. The separate real-endpoint profile measures full/unary/joint forward-plus-backward scopes at 0.589/0.375/0.200 seconds, with six/four/two complete filtrations. In the full case, pairing/index-transfer host wall is 0.346 seconds, distance host dispatch is 0.080 seconds, and synchronized backward is 0.157 seconds; these overlapping/dispatch scopes must not be summed as independent complete costs. All 192 reference pairing calls hit the already warm fixed-endpoint cache, while 192 generated pairings are fresh. This is not an observed production cache hit rate; `analysis/work_unit_profile.json` records the scopes and limitations. P3 relative gradient variance is 0.998, versus 1.180 for S2. Their individual direction cosines vary substantially. P3 was chosen conservatively before learning outcomes were examined.

The scalar family estimator is `(A + B + I_T*T_hat/p_T)/C0`, with `p_T=1/2`, plus the unchanged dense fidelity term. When T is omitted, its family callable, raster, filtrations, and GUDHI pairing are not executed. A skipped T is logged as missing, with its execution identity/probability retained. Both the inner `T_estimate` and the outer importance contribution are named explicitly. Calibration and evaluation never receive the training execution plan. Independent counter-seeded family/filtration streams depend on the accepted cursor and do not consume data, sensor, or model-noise RNG state.

Unbiasedness applies to the scalar gradient estimator before clipping and AdamW. It does not make the nonlinear ConFIG map of sampled family gradients unbiased, and it does not equate stochastic, clipped, or AdamW trajectories. All sampled-topology arms share that approximation for their matched comparisons.

ConFIG coefficient extraction uses a reduced QR/SVD on normalized full-width FP64 gradients and the full-width pseudoinverse rank threshold. It avoids normal-equation rank recovery. The new oracle rank diagnostic is authoritative for this check; the older aggregator's Gram-based rank flag remains diagnostic only and does not drive the update. Real SOURCE reconstruction errors are below 7e-16; signed-coefficient fixtures and an AdamW displacement comparison also pass. Actual SOURCE probe coefficients happened to be positive; that does not justify forcing other coefficients positive.

In the fixed-SOURCE P3 replay, holding the batch-0 coefficients gives direction cosines 1.000, 0.961, 0.971, and 0.891 across four fresh batches. Reducing the refresh interval to two still produces a 0.800 cosine on the second refresh/reuse pair. These probes include minibatch, reconstruction-noise, and filtration changes, but no optimizer updates; they do not establish moving-model staleness bounds. Both cadences fail the engineering audit threshold 0.9 for all declared reuse pairs, so the periodic arm is excluded. The default-off implementation has sparse current-gradient audits and requests a fresh oracle refresh when severe staleness is detected; those safeguards do not turn the failed pretest into a supported acceleration.

## Short timing and quality evidence

| Arm / calculation | Command wall, min | Coordinated process, min | Trainer, min | Epoch 1 / 2 wall, s | Epochs 3–10 wall median [range], s | Peak allocated, GiB | T-active batches |
|---|---:|---:|---:|---:|---:|---:|---:|
| T0: Full T + dense ConFIG | 17.55 | 17.39 | 16.78 | 103 / 97 | 98.0 [97–122] | 14.28 | 380/380 |
| T1: P3 + dense ConFIG | 14.33 | 14.25 | 13.71 | 83 / 79 | 80.0 [79–101] | 14.25 | 380/380 |
| T2: P3 + scalar sum | 9.34 | 9.26 | 8.71 | 53 / 49 | 50.5 [50–68] | 12.91 | 380/380 |
| T3: P3 + scalar sum + family p=1/2 | 7.81 | 7.68 | 7.13 | 44 / 39 | 42.5 [37–57] | 12.91 | 192/380 |

Command wall includes launcher preflight. T0/T1 command receipts resolve to whole seconds; T2/T3 additionally have external monotonic timing. T0 includes its failed bootstrap attempt (five command seconds, 0.179 coordinated seconds, zero training exposure). Coordinated process time begins after campaign admission and includes child initialization/calibration, training, routine full validation, and finalization; the separate 64-snapshot command audit is excluded from timing comparisons and included in the resource ledger. Trainer time starts later and is not the inclusive cost. Epoch wall times include reporting/checkpoint/validation work; the final epoch includes finalization. The retained JSON also contains the corresponding training-only epoch times.

The first two epochs include the unchanged warm-up. Epoch wall times come from the trainer's completed-epoch lines, rounded to seconds, including its checkpoint/validation/reporting scope. The 10-epoch inclusive process timer additionally covers SOURCE loading/calibration, initial evaluation, and finalization. Mean update timers are retained separately and are not called complete training cost. Initialization, cache warm-up, refresh/skip work, and failures are charged rather than removed from the budget.

All timing children are fresh SOURCE children. Target pairing caches use the same capacity (4,096 entries) and worker policy. Each process starts its own in-memory pairing cache; compiled KeOps kernels are shared through the owned campaign cache, warmed by the recovery smoke before T0. All profiles use the same current code and scientific descriptor primitives. Timing is a single sequential pass per profile on one physical GPU, with a shared host and no repeated-arm uncertainty estimate. Differences below the practical 15% inclusive saving target remain inconclusive for deployment screening.

| Arm | Epoch | Full A / SOURCE | Full B / SOURCE | Full T / SOURCE | Aggregate endpoint / SOURCE | Worst field ratio | Strict / engineering corridor | Useful target at check |
|---|---:|---:|---:|---:|---:|---:|---|---|
| T0 | 5 | 0.9145 | 0.8683 | 0.4774 | 0.9562 | 1.0092 | pass / pass | yes |
| T0 | 10 | 0.6698 | 1.1120 | 0.4345 | 0.9005 | 1.0454 | pass / pass | no |
| T1 | 5 | 1.2415 | 0.9688 | 0.4910 | 1.1250 | 1.2293 | fail / fail | no |
| T1 | 10 | 0.9186 | 0.9073 | 0.4997 | 1.0008 | 1.0448 | pass / pass | yes |
| T2 | 5 | 1.1055 | 0.9166 | 0.4220 | 1.0796 | 1.1289 | fail / fail | no |
| T2 | 10 | 0.7950 | 0.9297 | 0.4101 | 0.9406 | 1.0346 | pass / pass | yes |
| T3 | 5 | 0.9144 | 0.9115 | 0.5031 | 0.9670 | 1.0509 | fail / pass | no |
| T3 | 10 | 0.8372 | 0.9177 | 0.4305 | 0.9477 | 1.0425 | pass / pass | yes |

All ratios above use the same fixed first-32 validation SOURCE baseline; epoch zero is 1.0000 in every column. Duplicate identical final validation rows are reduced by accepted epoch. The useful-target flag is only a check-level observation, requiring strict feasibility, at least 10% T reduction, and A/B within +5%; it does not establish mature-window retention.

| Arm | Epoch | CH4 | CO | T | U_1 | p |
|---|---:|---:|---:|---:|---:|---:|
| T0 | 5 | 0.9592 | 1.0092 | 0.9831 | 0.9401 | 0.9470 |
| T0 | 10 | 0.9930 | 1.0454 | 1.0249 | 0.9203 | 0.7568 |
| T1 | 5 | 0.9857 | 1.0465 | 1.0063 | 1.1124 | 1.2293 |
| T1 | 10 | 0.9795 | 1.0448 | 1.0098 | 1.0071 | 0.9698 |
| T2 | 5 | 0.9728 | 1.0275 | 0.9950 | 1.0869 | 1.1289 |
| T2 | 10 | 0.9655 | 1.0346 | 1.0037 | 0.9530 | 0.8568 |
| T3 | 5 | 0.9841 | 1.0509 | 1.0029 | 0.9369 | 0.9582 |
| T3 | 10 | 0.9871 | 1.0425 | 1.0174 | 0.9365 | 0.8956 |

Endpoint ratios are model-unit field MSE, child/SOURCE. T0 selects epoch 10; T1 selects epoch 10; T2 selects epoch 10; T3 selects epoch 10. Full selected and final metrics are retained separately. T1’s epoch-5 U_1/p excursion is visible; its separate best-fidelity checkpoint remains at SOURCE epoch zero, whereas its topology-selected and final states are epoch 10. No ten-epoch checkpoint is promoted into a 100-epoch result.

| Arm | Exact child path below payload root |
|---|---|
| T0 | `runs/T0/20261023T021811Z_f7a4ad72` |
| T1 | `runs/T1/20261023T024006Z_433b002f` |
| T2 | `runs/T2/20261023T030742Z_9cadd70e` |
| T3 | `runs/T3/20261023T031859Z_ffea7252` |

No arm meets the useful target at both saved nonzero checks. T3’s epoch-5 CO ratio is 1.0509: a strict miss that remains within the separate engineering corridor. These ten-epoch outcomes check gross instability and provide full-metric context for timing. They do not select a scientific winner. Training estimates are not substituted for full validation metrics, and a small fidelity miss is kept visible. Strict +5% per-field/aggregate feasibility and the separate +5% aggregate/+10% per-field engineering corridor retain their distinct meanings.

## Goal 1: complete 100-epoch evidence

The three children started together at approximately 12:56:52 UTC on 2026-10-23, after all three physical GPUs were idle. Each completed 3,800 accepted updates, 121,600 sample exposures, or 15.2 TRAIN dataset passes. A reporting epoch remains 38 batches of 32 (1,216 samples); it is not an 8,000-snapshot dataset pass. No ten-epoch prefix or post-trained Run-11 checkpoint was used for initialization.

| Arm | Change relative to matched predecessor | Physical GPU | Complete coordinated process, s (min) | Exact run below ROOT |
|---|---|---:|---:|---|
| G1-C | P3, dense topology family, dense ConFIG | 1 | 5558.311 (92.64) | `runs/G1-C/20261023T125652Z_6467aba9` |
| G1-S | P3, dense topology family, scalar sum | 2 | 3316.947 (55.28) | `runs/G1-S/20261023T125652Z_482d2d98` |
| G1-R | P3, scalar sum, entire T family sampled with p=1/2 | 0 | 2754.535 (45.91) | `runs/G1-R/20261023T125652Z_47715345` |

These long-run wall costs are observational across different physical devices and shared CPU/storage. The prior sequential GPU-1 ten-epoch trials provide the matched idle-device latency evidence. Complete coordinated time includes startup/calibration, training, routine full evaluation, checkpoints and finalization; update timers alone are not end-to-end cost. All three completed before GPU 0 became the main working device. Later unrelated jobs on GPUs 1/2 were left untouched.

Every routine check uses the fixed first-32 validation IDs, sensors, shared queries, noise and SOURCE comparison. Existing full A/B/T, six deterministic topology filtrations and all five model-unit endpoint risks are evaluated at epochs 0,10,…,100. Selected means the trainer's existing topology-with-fidelity selection, not a new post-hoc selection rule. The following ratios use their own matched SOURCE panel; strict means aggregate and every field at most 1.05. The engineering corridor allows every field at most 1.10 while retaining aggregate at most 1.05.

| Arm / role | Epoch | A / SOURCE | B / SOURCE | T / SOURCE | Aggregate endpoint / SOURCE | CH4 | CO | T field | U_1 | p | Strict endpoint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C selected = final | 100 | .8682 | 1.0086 | .4153 | .9382 | .9853 | 1.0276 | 1.0171 | .9221 | .8935 | pass |
| S selected | 20 | .9265 | .9315 | .4225 | .9882 | .9987 | 1.0427 | 1.0325 | .9765 | .9676 | pass |
| S final | 100 | .9421 | 1.0553 | .4988 | .9873 | .9745 | 1.0305 | 1.0245 | .9781 | .9743 | pass |
| R selected | 60 | .9181 | 1.0314 | .4089 | 1.0076 | .9960 | 1.0220 | 1.0280 | 1.0280 | .9680 | pass |
| R final | 100 | 1.0488 | 1.3105 | .5049 | 1.0288 | 1.0310 | 1.0501 | 1.0619 | 1.0002 | 1.0550 | fail (CO, T, p) |

Rounded table entries do not decide feasibility; the audit uses unrounded values. S's final routine B exceeds its +5% common-target limit by approximately 0.53 percentage points. R's final B increase is dominated by the legacy cross-frequency squared-band-energy coupling (1.545× SOURCE, versus same-frequency 1.022×); it remains visible rather than being hidden by a favorable T value. Its final engineering endpoint corridor passes, while strict endpoint feasibility fails (CO is only 0.00770 percentage points above its +5% boundary).

The predeclared common useful target requires strict endpoint feasibility, T≤0.90, A≤1.05, B≤1.05, sustained at two adjacent routine checks. R first confirms it at epochs 10+20; confirmation-row visibility bounds are 548.158–558.180 seconds after child start. S has isolated passes at 20/40/80, and C at 40/80/100; neither has two consecutive passing checks. R later leaves the target. There is therefore no successful stable-through-100 retained-quality time ranking. R's early time-to-target is a limited observed event, not a convergence guarantee.

| Mature window, epochs 60–100 | Strict endpoint checks / 5 | Engineering checks / 5 | Common-target checks / 5 | Median A / B / T | Median aggregate / worst-field endpoint |
|---|---:|---:|---:|---|---|
| C | 3 | 4 | 2 | .9889 / 1.0175 / .4327 | 1.0129 / 1.0363 |
| S | 2 | 3 | 1 | .9421 / 1.0332 / .4519 | .9873 / 1.0552 |
| R | 3 | 5 | 1 | .9620 / 1.0665 / .4796 | 1.0076 / 1.0363 |

### Broader uniform-128 audit

Six immutable selected/final audits use identical 128 uniform validation IDs and identical observation/query hashes, in four complete groups of 32 with generation seeds 2027,102030,202033,302036. SOURCE group metrics match exactly across every audit (maximum numeric difference zero). Each SOURCE and child group executes all six topology filtrations. B is the mean of B(32) ensemble scores, not B computed once on 128 snapshots. Endpoint errors pool by valid entry counts; worst-field ratios are computed from pooled field scores. This broader panel partly overlaps the routine panel and remains validation data, not independent TEST generalization.

| Arm / role | Epoch | A / SOURCE | B / SOURCE | T / SOURCE | Aggregate endpoint | CH4 | CO | T field | U_1 | p | Strict endpoint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C selected = final | 100 | .9294 | .9177 | .4622 | 1.0085 | 1.0041 | 1.0237 | 1.0172 | 1.0057 | .9996 | pass |
| S selected | 20 | .9889 | .8812 | .4838 | 1.0380 | .9836 | 1.0187 | 1.0003 | 1.0483 | 1.0571 | fail (p) |
| S final | 100 | .9576 | .9859 | .5379 | 1.0158 | .9997 | 1.0037 | 1.0293 | 1.0187 | 1.0215 | pass |
| R selected | 60 | .8692 | .9628 | .4900 | 1.0077 | .9966 | 1.0022 | 1.0097 | 1.0284 | .9747 | pass |
| R final | 100 | .9097 | .8503 | .5106 | 1.0116 | 1.0213 | 1.0045 | 1.0305 | 1.0199 | .9950 | pass |

The broader panel reverses some routine-panel impressions: R final's B improves there, while S selected fails pressure at +5.71%. These are genuine panel differences, not permission to discard the unfavorable history or choose checkpoints on the broader result. Broader checks are single-check observations and do not establish the two-adjacent-check target. `analysis/goal1_uniform128_comparison.json` retains all family components, endpoint vectors, exact paths/hashes and sampling assertions; companion history audits keep selected and final inputs separate.

Two predeclared native-grid qualitative panels use validation index 0/CO and index 500/T, fixed sensor seed 42, generation seed 2027 and all 100×403 native queries. SOURCE and every child use the same G1-C runtime configuration (raw weights, two Euler steps, same endpoint correction); the SOURCE's own historical EMA/four-step defaults are not used. Values are decoded original-dataset units; physical field units were not specified, so no Kelvin or other units are invented. Main visual review found no gross collapse; remaining front errors are visible. These two panels cannot establish general field fidelity. Numerical maps for distinct selected states, all final states and selected-minus-final differences remain in `analysis/native_grid_field_maps/`.

### Training integrity and measurement limits

The complete per-batch audit verifies 3,800 accepted cursors per arm, exact sampler replay, all five dense current risks, projected dual formula/continuity, unchanged SOURCE references/C0 and full calibration/evaluation. C/S execute T on every update. R executes it on 1,906 and omits it on 1,894; skipped values are missing, and importance factors are correct. Maximum dual-formula error is below 5.6e-17. Minibatch violation frequencies and five multiplier trajectories are retained in `analysis/g1_dense_training_diagnostics.json`; they are training feedback, not proof of full-population feasibility.

R and S training children and launch-ledger rows exited successfully, but their original c073eba coordinators subsequently returned CLI code 2 because live sibling DataLoader descendants were misclassified as orphans during final status collection. No training was repeated. The actual completed status/checkpoints, child return code 0, source hashes, elapsed time and exact process ancestry are preserved in the two `G1-{R,S}_post_success_cli_error.json` receipts. Commit a30e07e corrects owned-descendant verification while still rejecting wrong-config/wrong-GPU/reparented processes; its focused suite passed 42 checks. This operational defect does not turn their successfully completed learning histories into failed or missing experiments.

A read-only ten-second visibility monitor stopped unexpectedly after poll 110 at 13:16:18 UTC; the cause is not established. One replacement finished normally. The observation gap is approximately 26.3 seconds; no newly visible validation row crossed that gap. In-session maximum polling interval was 10.041 seconds. SOURCE-at-zero rows were already visible on the first scan and are left-censored. Row visibility/process wall bounds are not exact GPU-kernel times. One broader R-last preflight briefly refused a busy GPU 0 without starting a child, then idle snapshots resolved the block in about four minutes and the single retry succeeded. This remained below the machine's 30-minute blocked-GPU stop rule; no other job was disturbed.

Goal-1 closeout and evidence hashes are in `manifests/goal1_completion_assessment.json`. All 142 numerical source files remained unchanged through its training, broader audit and map extraction. The immutable source snapshot `source_snapshots/goal1_c073eba` binds those files plus the original campaign/evaluation scripts; its manifest SHA256 is `3e05b001318b16876c33c759a8014db2559afd4f5f949704bd9c21b4e1d66fd9`. Later evaluation of old arms explicitly uses this verified snapshot rather than changed Goal-2 code.

## Goal 2: residual diagnostic and ordered prototype

The user explicitly retained the **+5% endpoint budget** for Goal 2 (`manifests/goal2_constraint_clarification_20261023.json`). A/B bounds are the exact fixed SOURCE calibration means of the full descriptors, with zero added slack; denominator floors affect scaling only. Calibration retains Run 11f's eight TRAIN batches and two draws, rather than estimating a mean over all 8,000 TRAIN snapshots. These empirical bounds do not establish population constraints. No conditional-density estimator, descriptor regressor, new descriptor definition, production projector/QP, or flow-only method was introduced.

The read-only probe compares original SOURCE and G1-S final100 on four paired fixed TRAIN batches of 32, with 4,096 shared queries, identical sensors/noise and the full six-filtration A/B/T definitions. Each batch supplies gA,gB,gT and five field-MSE gradients from one differentiable endpoint. Normalized A/B gradient rows form H; the pinned FP64 QR/SVD helper evaluates rT=gT−Hᵀ(HHᵀ)†HgT without a parameter-sized projector. Only compact Gram/norm/rank/cosine/manifests are saved, not parameter vectors. The engineering threshold was declared before results: finite nonzero T gradient, relative tangent orthogonality≤1e-8 and median residual fraction≥0.10 at both states.

| Fixed state | Residual fraction median [range] | Maximum relative tangent orthogonality error | Engineering gate |
|---|---|---:|---|
| SOURCE | .9753 [.9382,.9832] | 4.23e-15 | pass |
| G1-S final100 | .9959 [.9937,1.0000] | 2.24e-15 | pass |

The admitted physical-GPU-0 process took 38.670 seconds inclusive (32.755 seconds recorded computational scope), made zero optimizer steps, and verified unchanged parameters/SOURCE. `analysis/goal2_residual_probe.result.json` records the eight full-filtration evaluations, per-field fidelity-gradient interference and input/code SHA bindings. The result establishes a nontrivial local Euclidean parameter-gradient direction tangent to A/B equalities on these batches. It depends on parameterization/metric, does not residualize descriptor laws, and does not guarantee behavior after AdamW, clipping, weight decay, active endpoint constraints or finite steps.

The default-off ordered prototype minimizes T/T0 with dense A/B and endpoint inequalities through the existing scalar external Lagrangian. For hA=(A−A0)/max(A0,1e-8), hB=(B−B0)/max(B0,1e-8), it supplies the signed surrogate C0·[T/max(T0,1e-8)+μA hA+μB hB]; ExternalPolicy divides by C0 exactly once and adds its unchanged field terms. Detached μA/μB start at zero and update by projected SGD with LR .01 only after an accepted model/field-dual action. Negative signed scalar values remain valid. Full SOURCE A0/B0/T0 means are collected during the existing eight-by-two calibration, with no extra forward or RNG change; C0 and the five field reference objects remain identical. Raw scientific A+B+T, T estimates, ordered scalar and all dual trajectories have separate log names.

Checkpoint recovery binds the ordered reference artifact, source, exact integration/helper implementation, multipliers and accepted cursor. Speculative ordered JSONL rows are truncated at the atomic checkpoint. Tiny flow and deterministic Senseiver integration tests verify one endpoint/one backward, unchanged legacy calibration/RNG, nonzero parameter updates and reference/dual/log recovery; separate fixtures verify signed/C0/floor math and invalid combinations. These tests establish interface/correctness, not deterministic-model efficacy.

The real ordered recovery smoke `runs/recovery_smoke2ep/20261023T152631Z_a028a667` used an unchanged two-epoch horizon and one-epoch checkpoints. Actual SIGINT interrupted its first process after 46 durable accepted updates; recovery restored the atomic epoch-1 checkpoint at update 38 and finished at 76. The first resumed loss/risks/dual action and all eight repeated sampler decisions match their preserved counterparts. Optimizer steps, CPU/CUDA RNG state, index-RNG advance, ordered/fidelity references, multiplier continuity and unique recovered logs are verified in `manifests/goal2_smoke_recovery_verification.json`. Individual sample IDs were not separately logged; next indices derive from saved RNG state. The first repeated gradient norm differs by 5.77e-7 and later repeated ordered scalars by up to 2.48e-6. This is successful state/cursor recovery, not bitwise CUDA trajectory equivalence. cuDNN deterministic settings do not make every CUDA operation deterministic.

The interrupted segment's coarse status still reported epoch zero because ordinary validation had not reached epoch 10. The bound atomic epoch-1 checkpoint and hashed pre-resume log inventory establish actual progress. The original ledger row is retained; the recovery gate explicitly reconciles this evidence instead of rewriting elapsed work. The smoke charges three conservative epoch exposures, including eight durable repeated minibatches and at most one additional in-flight minibatch. Its two processes took 81.376 and 64.250 seconds.

**Ordered training status:** the single fresh original-SOURCE arm `runs/G2-O/20261023T154059Z_df6c299b` started at 15:40:55 UTC on 2026-10-23 with a 100-epoch horizon and two-hour watchdog. It matches G1-S in exposure, P3/family probability 1, optimizer/warm-up and model. Only objective roles/T0 normalization change. Full calibration reproduces G1-S's C0 and complete five-field reference object exactly; ordered A0/B0/T0 are `0.012916506850160658`, `0.0014809288622927852`, and `0.008211889740778133`. Its receipt is `manifests/goal2_execution_receipt.json`. No tolerance sweep, second ordered arm or automatic continuation is allowed. Quality conclusions await all 100 epochs, complete histories and selected/final uniform-128 checks.

## Resource and recovery accounting

| Resource before the running ordered arm | Actual / conservative charge | Limit |
|---|---:|---:|
| Ten-epoch timing trials | 4 × 10 = 40 epochs | ≤5 × 10 |
| Three Goal-1 arms | 3 × 100 = 300 epochs | ≤4 × 100 |
| All adapter/recovery smokes | 5.2105–5.2368 actual; 6 conservative epochs | ≤6 total public smokes |
| Actual training before ordered arm | 345.2105–345.2368 epochs | 556 |
| True conservative total before ordered arm | 346 epochs | 556 |
| Offline GPU diagnosis | 180 s old conservative reservation + 38.670 s new process | 2700 s |
| Combined GPU process charge before ordered arm | 4.4154 h; exact locked ledger retained | 16 h |
| Optional ordered arm | One 100-epoch arm running | ≤1 × 100 |

The original real deterministic adapter smoke uses the DeepONet SOURCE `cases/turbulent_combustion/runs/tc_deeponet_5000ep/20260829T125329Z_2ebc837b`. It proves adapter compatibility, not efficacy. The original two-epoch flow recovery used bounded max-steps interruption/resume, while the new ordered smoke exercised actual SIGINT. Rolling checkpoint intervals can discard up to almost five attempted reporting epochs in the ordinary profile. Discarded/repeated work is charged, and no smoke extends the scientific lineage. The controller additionally reserves the six smoke epochs on top of its new smoke rows, overcharging by three epochs: it reports 349 before ordered training, versus 346 true conservative and approximately 345.22 actual. This extra reserve is retained and must not be called measured exposure. Completion of the single ordered arm would give approximately 445.22 actual, 446 true conservative, or 449 launcher-reserved epochs.

All numerical outputs, compiler/Python/plot caches and temporary test data remain on verified mounted ext4 `/dev/nvme0n1p1` under ROOT. Repository links resolve there: `cases/turbulent_combustion/runs/_LPCAccel/LPCAccel_20261022_01a16674` → ROOT/runs; `reports/LPCAccel_20261022_01a16674` → ROOT/reports. Only short Unix-domain multiprocessing socket directories use `/tmp` to avoid Linux AF_UNIX path limits. Historical SOURCE, Run11f/11S and previous runs remain in place and untouched.

The campaign uses per-device lifetime locks, shared admission locking, owned-PID/config/GPU ancestry checks, locked idle rechecks and watchdogs. Launch/stop/resume are explicit foreground actions, never a queue of successor runs. The combined ledger counts process lifetime, live reservations, evaluation and interrupted exposure; six-epoch smoke and finalization capacity stay protected. A required GPU blocked continuously for 30 minutes pauses the goal and stops owned launch-capable retries; no such sustained block occurred here.

## Code, tests and reproducibility paths

| Purpose | Repository mapping |
|---|---|
| Complete filtration weights, P3/S2 and raster dispatch | `src/phycoflow_reconstruction/coherence/families/topology/filtration_sampling.py` |
| Independent family/filtration execution streams, opt-in bindings | `src/phycoflow_reconstruction/training/fast_lagrangian.py` |
| Signed FP64 coefficient recovery/fresh-loss reuse; rank-aware residual helper | `src/phycoflow_reconstruction/training/periodic_config.py` |
| Existing dense field-risk primal/dual policy | `src/phycoflow_reconstruction/training/fidelity_budget.py` |
| Ordered reference, signed scalar and projected A/B dual state | `src/phycoflow_reconstruction/training/lpc_ordered.py` |
| Shared endpoint, full calibration, ordered integration and raw/optimized logs | `src/phycoflow_reconstruction/training/post_training.py` |
| Strict opt-in validation; speculative-log recovery | `src/phycoflow_reconstruction/config/validate.py`; `training/run_store.py` |
| Bounded device/storage/source-stage/launch/evaluation control | `scripts/posttraining/fast_lagrangian_campaign.py` |
| Full existing-family campaign evaluation, exact groups/manifests | `scripts/posttraining/fast_lagrangian_evaluate.py` |
| Portable P3 scalar / ConFIG / exploratory ordered profiles | `cases/turbulent_combustion/configs/posttrain/fidelity_budget/{fast_lagrangian_scalar,fast_lagrangian_config,lpc_ordered}.yaml` |

The earlier short-stage relevant CPU suite passed 251 tests (one CUDA-only case skipped), and that CUDA oracle was subsequently run on idle GPU1 and passed. Later independent rounds passed 42 ownership/evaluator checks, 47 native-map checks, 149 production-policy/estimator/recovery regressions, and 55 source-stage/residual controls. The final ordered source passed 14 unit/profile/flow/deterministic cases plus its added direct-Python rejection test; the latter initially expected the wrong exception class, was corrected in the test only, and passed. Ruff and diff checks passed. CPU tests are distinct from scientific dataset execution and physical GPU timing.

Goal-2 numerical source is independently pinned at commit `11d9167` in `manifests/goal2_source_binding_ready.json` and `source_snapshots/goal2_11d9167` (143 numerical Python files, plus tested controller/evaluator snapshots). The original Goal-1 142-file pin is preserved. Run checkpoints and stage receipts retain exact configurations, reference/family/normalizer/source hashes, sampler cursors, RNG/optimizer/dual state and paths. New code-revision rounds are committed and pushed on the study branch; generated numerical/figure payloads and this machine-local UpdatePlans file are not placed in Git.

Important retained ROOT paths: `analysis/g1_complete_history_audit.json`, `analysis/g1_dense_training_diagnostics.json`, `analysis/goal1_uniform128_comparison.json`, `analysis/g1_validation_observations.jsonl`, `analysis/native_grid_field_maps/`, `analysis/goal2_residual_probe.{plan,binding,result}.json`, `manifests/goal1_completion_assessment.json`, `manifests/launch_ledger.jsonl`, `manifests/protocol_identity.json`, `manifests/preflight.json`, and per-run configuration/history/evaluation/checkpoint files. Original ten-epoch numerical evidence remains in `analysis/timing_summary.json`, `analysis/pretest_full.json` and `analysis/pretest_p3.json`.

## Portable operation and figures

Three tested scientific profiles are available: dense-family P3 scalar, dense-family P3 ConFIG, and the default-off exploratory ordered profile. The scalar profile trades lower cost for the mixed history described above; periodic reuse is unsupported. [Portable user commands](docs/posttraining/FAST_LAGRANGIAN_LPC_USER_GUIDE.md) cover explicit original SOURCE, new mounted payload/safe link, foreground launch, Ctrl-C, unchanged-config own-state resume, and full-descriptor selected/final validation. The public `scripts/posttraining/evaluate_full_abt.py` writes immutable JSON, uses the child's raw-weight/two-Euler contract for both models, groups B in exact batches of 32, and refuses the historical campaign layout so its frozen-stage wrapper cannot be bypassed. Campaign evaluation additionally verifies the live controller parent, exact child/admission receipt and held physical-GPU lock. The final control/campaign/portable evaluator suite passes 104 CPU tests. Its real synthetic-dataset end-to-end fixture uses a scoped synchronous DataLoader adapter after a bounded multiprocessing stall; it still exercises the compact dataset, sampler/collator, SOURCE EMA/four-step setup and raw child/two-step full A/B/T comparison, but does not validate asynchronous workers. Actual study training separately exercises the production GPU path. Three profile tests check the measured checkpoint/validation schedule. No larger horizon is launched, queued or inherited from this campaign.

The final figure set remains limited to three PDF masters. Existing short-stage vector pages will be retained and combined with complete Goal-1 histories, time/field evidence, native maps and Goal-2 diagnostics. Current new pages are review drafts under ROOT/tmp/figure_drafts; final PDF paths, regeneration inputs, render QA and the exact owned-disposable cleanup receipt will be updated at closeout. Numerical evidence/checkpoints/source snapshots remain protected.
