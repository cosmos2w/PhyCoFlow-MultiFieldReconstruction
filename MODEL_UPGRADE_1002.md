# Model upgrade 1002: implementation and bounded validation

**Scientific release blocked; implementation validation complete.**

Versioned A/B/C definitions, separate coherence combiners, adaptive fidelity, recovery, selection, and diagnostics are implemented. The bounded study has a **blocked scientific release**: every integrated selected checkpoint fails the independent validation fidelity audit. The first-100-epoch optimizer comparison, second training seed, locked test audit, and formal-profile smoke are incomplete or not run. No 5,000-epoch run was launched. This is the honest negative-study outcome permitted by work-plan §12; formal preparation under work-plan §14 requires passing release gates.

## 1. Repository, source identity, and environment

Repository `cosmos2w/PhyCoFlow-MultiFieldReconstruction`, remote `git@github.com:cosmos2w/PhyCoFlow-MultiFieldReconstruction.git`, branch `codex/phycoflow-upgrade-1002`; reviewed base `f785c673e2c7d05f18ac846d09b06f522294192c`. Coherent pushed milestones:

| Commit | Implementation |
|---|---|
| `879c98e263628a8bb419423d694aa34e48c5dcaf` | Opt-in A/B/C, K-objective/controller, strict evaluation, budget launcher, histories |
| `549a3d470d3be1f48fba3b6a5257af2af323d981` | B symmetric floors, JSON selection membership, invocation provenance |
| `e78c39ec855bb5d52a0bfe88d448e071aa5d7cc1` | Durable off-panel selector serialization |
| `80e17c97d3365c7d5fce9f800f8e7ff6549b0230` | Accepted-update telemetry and run-step plot axes |

The implementation was frozen at `80e17c9` for the final training windows and four-panel GPU audits. Later optional native-audit integration is documented separately; it did not generate the pilot results.

All tests used `/home/wanglz/miniconda3/envs/phycoflow_env/bin/python`: Python3.10.19, Torch2.5.1+cu121, NumPy2.0.1, SciPy1.15.3, GUDHI3.13.0, conflictfree0.1.8, h5py3.14.0, PyKeOps2.3, pytest9.0.3, Ruff0.16.3, Matplotlib3.10.8. Optional dependency contracts are `conflictfree==0.1.8`, `gudhi>=3.13,<4`; installed versions stayed fixed. Actual GPU work used **physical GPU0**, NVIDIA RTX6000 Ada, UUID `GPU-233fcd85-5c6a-6212-3f44-655252afec70`, 49,140MiB, with `CUDA_VISIBLE_DEVICES=0`. Other GPUs/jobs were untouched.

Every independent scientific lineage starts from LIVE weights at:

```text
/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction/cases/turbulent_combustion/runs/tc_gl_rbf_cq_cached_kv_5000ep/20260826T212231Z_7b68c461/checkpoints/last.pt
SHA256 03f2f45b966a00850cc45c011b4cf1cd644b9408de355324fec8dc434a84a810
```

This checkpoint has global step315000 and **no explicit epoch field**. Run manifest/status and resolved config establish the completed 5,000-epoch base run; those are separate evidence. Source configured EMA evaluation was explicitly disabled for live/live comparisons; historical EMA tracking/evaluation contracts remain supported.

Protocol reference is the actual resolved config/history of:

```text
/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction/cases/turbulent_combustion/runs/coherence_fix_ABC_sliced_persistence_formal_5000ep_gpu1/20260924T030724Z_4208231c
```

Its batch32, fraction0.15, **38-update epochs**, sensors and two-Euler-step protocol were retained. Its epoch3025 endpoint and mixed configured-source-EMA/post-live headlines were not pilot initializers or matched numerical baselines.

All evidence is saved under the exact **ignored, machine-local** root:

```text
T = /home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction/cases/turbulent_combustion/runs/Test_1002
```

`T/` paths below expand to that directory. Evidence/checkpoints are not uploaded as Git data. Initial identity and environment are `T/_audit/source_contract.json`, `T/_audit/environment.json`; final preservation is `T/_audit/source_immutability_final.json`.

## 2. Scope preserved and data roles

The 5,461,817-parameter `GL_rbf_CQ` / historical `GL_rbf_ENH_CQ` architecture, decoder, source prior, strict tensor-key loading, cached K/V/cached-streamed execution, RFF256, two-step rollout, observation correction, normalizer/sensors, shared endpoint, RunStore, PH core, and historical definitions were retained. A/B/C remain parallel Local/Pair/Collective observables, without an orthogonal/residualized decomposition or new PDE/physical-time objective.

Dataset `/home/wanglz/Desktop/src/PhyCoFlow/0_demo_TurbulentCombustion/Dataset/Merged_CH4COTU1P.h5`: 8,060,571,792bytes, fingerprint `8c49936567eced7ab94887c336b9b35aaf7ec70dea7479aee83434ff970455d5`, field shape `[1,10000,40300,1,1,5]`, native grid100×403. This is **one trajectory with snapshots**, not 10,000 independent trajectories. Fields **[CH4,CO,T,U_1,p]**, sparseT192–384 sensors; units are unknown in metadata. Decoded plots use dataset-scale values/raw coordinates with unspecified units. No missing velocity component, SI label, or incompressibility residual was invented.

Chronological split8000/1000/1000 TRAIN/validation/test. Frozen TRAIN normalizer offsets `[.0293445941,.0071263295,1118.654296875,-3.4435703754,114098.046875]`, scales `[.0266994927,.0094787497,775.4280395508,37.2216835022,7144.0668945313]`. Model-unit MSE treats fields equally; each field and unobserved-entry risk is exposed separately rather than relying on one raw-unit average.

TRAIN alone supplies canonicalizer/tail calibration, B floors/masks, family gradient scales, and source-risk normalizers. Runtime/observation seed42, family sampling seed700043, fixed shared4096 query seed100045, selection noise2027. Validation selection IDs8000–8031; audits repeat them with noise3027/sensor offset1000 and use disjoint8032–8063 (noise2027) plus8064–8095 (noise3027/offset1000). These are64 distinct disjoint **snapshots**, chronologically correlated, without an independence/CI claim. No test-data tuning, calibration, or inference occurred.

Matched LIVE fixed-panel source total model-unit MSE **.4897319973**; CH4 **.1277378052**, CO **.3917801082**, T **.1122818887**, U_1 **1.0921852589**, p **.7246747613**. Legacy raw A/B/C **.7141329050/.0585604459/.0132849850**; integrated upgraded raw A/B/C **.1336572021/.2551212907/.0137608368**. Cross-definition percentage comparisons are not meaningful.

## 3. A: marginal W2, own-CDF surrogate, calibrated smooth tail

For aligned valid rows X,Y∈R^(N×C), the preserved model-unit marginal estimator is

\[
L_{marg}=\sum_cw_c\frac1N\sum_n(X_{(n)c}-Y_{(n)c})^2,\qquad\sum_cw_c=1.
\]

Five equal weights are used. This term does not independently standardize X/Y, and its empirical query-row measure is not a claimed physical-volume measure. Omitted `definition` keeps `legacy_v1`: raw-value marginal W2, pairwise fixed-bank sliced W2², and all-field hard top-fraction projection costs. Legacy source components .1322593689/.1345590055/.4473145306 sum to .7141329050.

Opt-in `marginal_copula_v2` estimates generated/reference marginals **separately**. Exact detached audit coordinates use deterministic average ties:

\[
U^X_{nc}=(midrank(X_{:c})_n-1/2)/N,\quad U^Y_{nc}=(midrank(Y_{:c})_n-1/2)/N.
\]

Rows stay jointly aligned. Training does not pretend hard ranks have useful derivatives. Its explicit own-CDF approximation is

\[
z=(x-\bar x)/\sqrt{mean((x-\bar x)^2)+s_{min}^2},\quad
q_k=Q_z((k+1/2)/M),\quad\widetilde U_n=M^{-1}\sum_k\sigma((z_n-q_k)/h).
\]

M64, h.1, s_min1e-6, chunk512. Gradients pass through generated centering/scale/quantile landmarks/sigmoids; target coordinates are detached. Work is O(NM) comparisons with bounded chunks plus quantile sorting, rather than an N×N matrix. This local alternative to an external soft-rank dependency is a surrogate, with measured nonlinear-monotone departure.

Using32 serialized mixed Sobol unit directions, seed1234/axes excluded, let d_r be projected empirical W2² in smooth own-CDF coordinates. The dependence score is

\[
L_{dep}=.75\bar d+.25[V_{\rho,\tau}(d)-V_{\rho,\tau}(0)],\quad
V_{\rho,\tau}(d)=\min_\eta\{\eta+\tau(\rho R)^{-1}\sum_rsoftplus((d_r-\eta)/\tau)\}.
\]

rho.1; V(0)=tauH(rho)/rho; rho1 returns mean. Detached deterministic CPUfloat64 bisection solves `mean(sigmoid((d-eta)/tau))=rho`, residual≤1e-6; envelope derivative is `sigmoid((d_r-eta)/tau)/(rhoR)`. There is no persistent eta. TRAIN32×4096 source endpoint gives positive-cost median7.5960721006e-5, frozen tau=max(1e-6,.05median)=**3.7980360503e-6**. Source upgraded joint loss .0013978309. Pairwise training weight is0; optional pairwise/exact-midrank diagnostics are separate, avoiding hidden double counting.

TRAIN bank mean mixture changes16→32:1.314e-5;32→64:1.228e-6; mean absolute snapshot32→64:1.021e-5. Tested exact float64 monotone invariance holds; float32 tie merging moves ranks by up to1/4096. Smooth coordinates change by mean.026662/max.138436 under the tested nonlinear monotone map. Reference duplicate fraction averages27.2%, maximum82.0%; no constant TRAIN snapshot field. Finite directions, atomic marginals and approximation bias remain limitations.

Code under `src/phycoflow_reconstruction/`: `coherence/families/global_distribution/components/{self_marginal,copula,cross_copula,tail_risk,point_masks}.py`; symbols `exact_midrank_uniform`, `quantile_landmark_smooth_cdf`, `identity_calibrated_smooth_cvar`; `GlobalDistributionFamily` serializes discriminator/bank/config. Evidence `T/_audit/A/training_endpoint_bank_diagnostics.{json,npz}`; distribution and evaluation acceptance tests.

## 4. B: signed covariance of linear graph coefficients

Unchanged `legacy_v3` same-mode statistic is

\[
\kappa_{ij}(k)=\frac{|mean_b(a_{bki}\overline{a_{bkj}})|^2}{mean_b|a_{bki}|^2\,mean_b|a_{bkj}|^2+\epsilon}.
\]

Legacy cross-band forms e_bli=Σ_(k∈l)|a_bki|², centers those energies, computes covariance with divisorB, and compares its magnitude squared normalized by energy variances+epsilon. The energy covariance is fourth-order in the fields and discards sign/phase. Its saved `cross_frequency` semantics remain intact.

Opt-in `second_order_blocks_v4` forms centered linear coefficients in a fixed retained graph basis:

\[
\widehat C_{ij}[k,q]=(B-1)^{-1}\sum_b(a_{bki}-\bar a_{ki})\overline{(a_{bqj}-\bar a_{qj})}.
\]

PilotB32/K48, three16-mode bands with intervals `[0,.016293895430862904)`, `[.016293895430862904,.034264786168932915)`, `[.034264786168932915,2)`. Geometry eigengaps define boundaries without splitting near-degenerate clusters. Same-frequency means complete same-band blocks including k≠q; cross-frequency means distinct bands. Both descriptors depend on second moments, although normalization and squared discrepancy are nonlinear functions of those moments.

For E_li=tr(C_ii^(ll)), frozen TRAIN Ecal supplies eligibility (energy fraction≥1e-8), floors and masks:

\[
F_{li}=max(10^{-6}E^{cal}_{li},tiny),\quad\widetilde E_{li}=max(E_{li},F_{li}),\quad
\delta_{ij}^{lm}=10^{-12}+10^{-6}\sqrt{E^{cal}_{li}E^{cal}_{mj}},\quad
\Gamma_{ij}^{lm}=C_{ij}^{lm}/(\sqrt{\widetilde E_{li}\widetilde E_{mj}}+\delta_{ij}^{lm}).
\]

L_B is the sum of uniform eligible same-band and cross-band means of `||GammaX-GammaY||_F²`. Entries are **summed** within each block, not entrywise averaged. Pairs CO/T,T/CH4,T/U_1,CH4/U_1 yield12 same/24 cross blocks. p is fidelity-protected but not a B target; no optimized self-spectrum term is added.

Initial v4 generated-only floors fail low-energy identity (scale1e-4:loss1.051e-6,gradient≈.00664). The explicit repair `energy_floor_policy: symmetric_calibrated`, **artifact4.1**, floors both generated and reference normalization, giving zero low-energy identity value/gradient. Original v4 (`generated_only_legacy`) artifacts and four smokes are quarantined/retained, not silently upgraded or resumed.

Signed/complex blocks preserve phase/sign; paired phase/sign changes, within-band joint unitary rotations, point and ensemble permutations have tested invariances. Floors limit low-amplitude invariance; covariance rank≤31 for B32. Generated collapse cannot disable the frozen mask, but complete covariance collapse may have positive discrepancy with zero field gradient. Means, normalized-out amplitude and fourth-order coupling remain deliberate blind spots. Evaluation pools sufficient statistics before normalization and separately reports aligned32 groups; snapshot ensembles are not posterior samples. Off-frequency covariance describes spatial inhomogeneity/nonstationarity, not nonlinear energy transfer/causality.

Common R1 basis SHA `ab70c2d89d0adb5e47b7f28a0240b10d0d2338e5a23394903a562649eaba5372`; Ecal SHA `4f13ad4e27ecbfa9bad6ec324dba66508e910d8d8b94c7e6739b9d0f2d537f4c`; stored hashes match across arms. CPU recomputation max relative2.1243e-7 versus GPUfloat32 fits5e-7 tolerance, while stored-hash checks remain exact. Complete signed blocks, floors/bands and singular-value/rank diagnostics are saved.

Code: `coherence/families/cross_spectrum/{statistics,basis,covariance_blocks,family}.py`, `evaluation/coherence_set.py`; `T/_audit/B/pilot_monitor/` and spectrum acceptance tests. Covariance coefficients are linear; loss squaring is not a claim of a quadratic field loss.

## 5. C: retained cubical PH, subset bank, native geometry

GUDHI cubical pairings, positive finite bars, essential births, diagonal augmentation, fixed-angle slicedW1 and live critical-value gathers remain. Pair indices are detached; gradients are exact for fixed pairings and nonsmooth at ties/changes. `persistence.py` was not replaced by a surrogate.

Detached GT per-field mean/std is applied to both paired descriptor fields, with a scale floor. For signσ and a fixed positive line(a_l,b_l),

\[
f_{l,\sigma}(x)=max_j[(\sigma z_j(x)-b_{lj})/a_{lj}],\quad w_l=min_ja_{lj}.
\]

CO/T mutual uses a serialized16-line Sobol master(seed1729); training uniformly averages4 selected lines. Groupg/committed zero-based steps uses seed `(1729+1000003s+97409g) mod(2^63-1)` and sorted chosen IDs. Calibration/evaluation use all16; resume recreates the same subset. The subset mean estimates the fixed-bank mean, not an infinite matching distance.

Finite diagrams use cross-diagonal augmentation, sorted projectedL1 sums, mean32 midpoint-angle directions, then division byHW. Essential births use a sortedL1 **sum** times.1, **without HW division**. Existing nested means over H0/H1, self fields/signs and mutual lines/signs/component weights remain; line weights multiply mutual distances. Legacy Betti/spatial definitions remain explicit alternatives.

Complete Cartesian sets use verified coordinate-permutation gather/reshape; integer-factor complete-grid coarsening supports area averaging. Actual100×403→32×128 ratios are noninteger; coarse and sparse4096 maps use IDW. They are not asserted PH-equivalent to native grids. Reference caching is target-only and bank/content/geometry/backend/version bound; generated pairings are recomputed.

Scalar/packed values and input gradients, CPU/GPU0 unique9×9 critical gradients, native100×403 gather/gradients: max error0. Uniform subset-gradient relative error .0652 at16 subsets→.0133 at256. TRAINframes355/1831 uncached PH costs≈1.24s **per complete snapshot of36 filtrations**, coarse/sparse≈.128s/snapshot. These are GT-only costs, not full updates or per-field timings.

Code: `coherence/families/topology/{family,geometry,persistence_objective,persistence}.py`; tests `test_upgrade_1002_topology.py`; `T/_audit/C/native_grid_acceptance.{json,log}`. The default-off periodic native hook is documented with actual integration checks in §11.2.

## 6. Adaptive fidelity and K-objective updates

`optimization.update_policy: coherence_primal_dual` and `coherence_gradient_method: config|weighted_sum|cagrad` expose individually calibrated A/B/C gradients on one endpoint. Frozen TRAIN `initial_grad_norm` family calibration is separate from risk scales. Legacy update/manual-retention restrictions remain.

For j=total,CH4,CO,T,U_1,p, R_j is aggregate model-unit endpointMSE against GT; a frozen LIVE teacher uses the same noise. Define

\[
g_j=[R_j(\theta)-1.05R_j(\theta_0)-a_j]/s_j,\quad
P=\sum_j([\lambda_j+\rho g_j]_+^2-\lambda_j^2)/(2\rho).
\]

Defaults: absoluteallowance0, riskfloor1e-8, rho1, duallr.05, EMA.9, cap100,8 TRAINcalibration batches, matched-noise live source, cadence20 diagnostics. T50 source-risk scales `[total,CH4,CO,T,U_1,p]`=`[.02810287606,.00985535444,.05895649036,.01645670715,.04344174592,.01180408243]`, not the validation risks in§2. T50 family scales A1/B1.2322878007/C.2886464243. T51/T52 share protocol/banks but differ≈1e-8 in gradient scales and have distinct full calibration-file hashes; exact values are in pilot inventory/artifacts.

Weighted baseline sums calibrated family gradients. ConFIG uses conflictfree0.1.8 equal weights/dtype-correct pseudoinverse; only a finite direction with strictly positive dot against each active family is accepted, otherwise a weighted-sum fallback/reason is recorded. Zero/disconnected gradients and undefined cosines are explicit. CAGrad follows the author MEAN convention, alpha.5, rescale1 division by1+alpha², SLSQPftol1e-10/max200; no hidden×K. WeightedSUM and CAGradMEAN therefore differ in algorithmic scaling despite common controller/lr.

Final gradient=`coherence_direction+Σ_j[lambda_j+rho*g_j]_+ grad(g_j)`. Clip once at norm1, take one AdamW(lr5e-5,decay1e-6), then advance bias-corrected EMA of detached **pre-step** violations and capped multipliers. Rejection does not advance state. Teacher deepcopy precedes resumed-child loading. Nativeflow objective is monitor-only; inherited manual retention0; adaptive source-anchor constraint is unsupported/rejected. Parameter/optimizer/dual/EMA/count/RNG/bank/floor/calibration/selector state recover without recalibration.

Actual `grad(L)^T(theta_after-theta_before)` includes AdamW moments/decay; negative predicts local decrease on that batch. Real/imaginary coordinates are consistently interleaved. Raw ConFIG descent does not guarantee descent after fidelity pressure/clipping/preconditioning, nor does the controller guarantee held-out/pointwise/nonconvex convergence. Disjoint pressure failures expose that limitation.

Code: `training/{fidelity_controller,gradient_balance,gradients,coherence_calibration,post_training,checkpointing,run_store}.py`. CAGrad75-reference-vector parity max absolute.0003916/relative norm.0003054, below declared.003 (`T/_audit/cagrad_reference_parity.json`).

## 7. Selection, recovery, and diagnostics

The existing `topology_with_fidelity` selector remains available. New `coherence_with_fidelity` requires total **and every field** to stay within +5% of the matched LIVE source. Among eligible checkpoints it minimizes

\[
S=\frac13\sum_{F=A,B,C}\frac{L_F^{val}(\theta)}{\max(L_F^{val}(\theta_0),10^{-12})}.
\]

Family weights, source scales, query geometry, sensors/noise, 32 validation snapshots, B ensemble membership, and full C bank are frozen. The source at child step0 is an explicit fallback, not a trained improvement. A top-three feasible archive supplements the scalar selector. Source, selected and latest are retained and audited separately; an infeasible latest checkpoint cannot replace an eligible archived one.

Resume restores teacher identity, model/optimizer and auxiliary states, calibration, risk normalizers, dual/EMA counters, line selection, sampling/RNG, and selector/archive without recalibration. JSON family membership is checked as a set with canonical order rather than relying on dictionary insertion order. Missing selector state is rejected instead of inventing a fresh selection history. An actual off-panel recovery retained selector step0 and controller step38 (`T/_audit/checkpoint_off_panel_real.json`). Synthetic interrupted/resumed parity checks the complete state, including a two-update interrupted segment and partial-epoch accounting. The two failed original T10 resume attempts remain in the ledger and are not claimed successful.

`metrics/coherence_updates.jsonl` now writes one strict JSON row per accepted update. Recovery truncates torn/non-durable tails to the checkpoint boundary. Scalar batch risks/violations, pre-step pressure, post-step multipliers/EMA, family losses/scales, combiner/fallback, norms/clipping, and runtime are logged. At cadence20, gradient/cosine/rank diagnostics and actual AdamW displacement dots are recorded. Existing historical summaries are preserved.

Exact per-update records exist for **steps381–760** in each integrated R1 arm; earlier epoch summaries and selected diagnostic records cannot certify every earlier update. In that 380-update window, clipping occurred on T50 352, T51 181, T52 245 updates; T50 ConFIG truthfully fell back 49 times. No recorded multiplier hit cap100. Pressure usually drove the p multiplier; T52 ended with zero multipliers while held-out fidelity still failed. None of these training diagnostics certifies validation eligibility.

Public adaptive plots now use actual **child run-step coordinates**, not parent lifetime steps or rewritten epochs. Each final history has21 indexed summary rows, selected steps190/380/380, and latest760. Ledger accounting separately retains lost work. Code maps: `training/{checkpointing,coherence_history,run_store,post_training}.py`, `scripts/visualization/coherence_posttraining_report.py`; tests `test_upgrade_1002_{lifecycle,update_telemetry}.py`, `test_adaptive_coherence_history.py`.

## 8. Executed validation and commands

All executable checks use `/home/wanglz/miniconda3/envs/phycoflow_env/bin/python` (abbreviated **P**) from the repository root. **T** is the absolute Test_1002 path in §1. CPU tests use `CUDA_VISIBLE_DEVICES=`; GPU checks use `CUDA_VISIBLE_DEVICES=0` with verified physical UUID. Common thread settings are `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PHYCOFLOW_TOPOLOGY_WORKERS=4`; plot/compiler caches reside in `T/_audit`.

| Check and reproducible command/scope | Measured outcome | Exact evidence under T |
|---|---|---|
| Frozen full CPU suite, `P -m pytest`, milestone879 | 544 passed,7 skipped,705.70s | `_audit/frozen_cpu_suite.log`, `frozen_cpu_results.json` |
| Full CPU suite, `P -m pytest`, e78 selector recovery | 548 passed,8 skipped,713.09s | `_audit/final_cpu_suite.log`, `final_cpu_results.json` |
| Full CPU suite, `P -m pytest`, 80e telemetry | 554 passed,8 skipped,775.75s | `_audit/telemetry_cpu_suite.log`, `telemetry_cpu_results.json` |
| Telemetry/history/lifecycle focused checks | 53 passed,110.16s | `_audit/telemetry_final_focused_tests.log` |
| Native periodic audit units and actual interrupted/resumed lifecycle, `P -m pytest tests/test_upgrade_1002_native_audit.py tests/test_upgrade_1002_native_audit_lifecycle.py` | 11 passed,30.56s | `_audit/native_integrated_focused_tests.log` |
| Final complete CPU suite at7e, exact command recorded by helper below | **565 passed,8 skipped,815.61s**, exit0; wrapper817.13s | `_audit/native_final_cpu_short_results.{json,xml}`, `native_final_cpu_short_suite.log` |
| Actual GPU0 A/B focused acceptance | 4 passed,38 deselected,3.75s; focused scope only | `_audit/final_upgrade_cuda_tests.log` |
| Actual GPU0 C oracle/native raster and line-gradient acceptance | Unique critical gradients/native gather error0; line sampling convergence in §5 | `_audit/C/native_grid_acceptance.{json,log}` |
| Pinned legacy values/input gradients, N4096/B32, CPU/GPU0 | CPU error0; GPU error below1e-7 | `_audit/legacy_source_value_gradient_parity.json` |
| Production versus differentiable cached rollout, GPU0/B1/N4096/two Euler steps | Endpoint error0;145 parameter-gradient comparisons maxabs5.7742e-8/maxrel1.04797e-6 | `_audit/source_rollout_gradient_parity.{json,log}` |
| `P T/_audit/audit_cagrad_reference.py` | 75 author-reference vector cases pass declared .003 tolerance | `_audit/cagrad_reference_parity.json` |
| `P T/_audit/validate_tracked_configs.py` against pinned validator | 61 compositions,39 valid;22 unchanged source/template restrictions,0 new regressions | `_audit/tracked_config_validation_summary.json` |
| Bounded launcher/config gates and actual pilot configurations | 9 initial profiles/4 R1 profiles validate; malformed paths, budgets, sources and resume identities reject | `_audit/resolved_pilot_validation.json`, `_audit/{budget_final_gates,budget_console_final_gates,integrated_to20_dryrun}.log`, `planned_runs.json` |
| `P -m ruff check src tests scripts`; `git diff --check` | Pass at7e | `_audit/native_integrated_ruff.log` |
| Final source/reference protection and dataset contract | 18/18 initially protected files byte-identical; original dataset fingerprint unchanged | `_audit/source_immutability_final.json` |

The rollout-gradient audit removes three no-grad wrappers only in the isolated audit; production inference remains unchanged. B tests cover signed/complex covariance, low-energy identity, collapse/masks, complete same-/cross-band blocks, basis invariance, saved calibration and pool/group evaluation. A tests cover ties/constants/masks, finite differences through generated landmarks, CVaR envelope gradients/identity/temperature, permutations, serialization and exact-rank diagnostics. C tests preserve diagonal/essential handling, gradients, replayed subset sequence, native permutation and supported antialiasing.

Earlier failed/interrupted checks remain evidence: an initial full suite raced an in-progress source edit and failed one hash check (541pass/7skip/1fail), then the frozen suite passed. The549 full attempt terminated at64%/exit143; it is not a pass. Focused fixture/import issues were corrected and rerun before the frozen milestones. No thresholds or existing regression tests were weakened to force acceptance.

Two7e full attempts stalled after29 tests in the **unchanged** asynchronous base-loader path and were deliberately terminated (exit143). A 180s forced-spawn diagnostic with the same long temp path timed out (exit124), so it does not justify changing worker start method. The root's 126-byte absolute `TMPDIR` reproduces `OSError: AF_UNIX path too long` in a multiprocessing Listener. A short task-specific symlink points to the **same Test_1002 payload directory**; its53-byte Listener path succeeds. With a short path and default fork, the unchanged target-free source/child lifecycle passes1/1 in16.33s. No loader/source/test edits were made for this environmental correction. Receipts: `_audit/native_final_cpu{,_trace}_stall_receipt.json`, `_audit/ipc_temp{_path_diagnostic,_alias_receipt,_short_path_diagnostic}.json`, `_audit/C/{async_loader_spawn_diag,short_tmp_defaultfork_diag}/`.

The final complete-suite command is:

```bash
rtk proxy env PYTHONDONTWRITEBYTECODE=1 \
  /home/wanglz/miniconda3/envs/phycoflow_env/bin/python \
  cases/turbulent_combustion/runs/Test_1002/_audit/run_native_final_cpu.py
```

That helper records the exact subprocess arguments/environment, preserves the JUnit XML and full log, leaves GPU visibility empty, and requires `/tmp/pfc1002-01a0fe4d` to be a symlink to `T/_audit/launcher_tmp`. The alias is only for a short IPC pathname; every actual test payload remains under T. CPU-only GPU-dependent skips retain their original reasons; no physical GPU1 test ran.

Other executed audit/report entry points are recorded with their arguments in reports/logs: `T/_audit/{audit_inputs,audit_source_rollout_gradients,audit_legacy_descriptors,audit_validation_panels,summarize_pilots}.py`; `T/_audit/B/pilot_monitor/legacy_v3_b_score.py`; `T/_audit/A/{legacy_ac_cached_score,create_pareto_and_topology_reports}.py`; `T/_audit/C/native_periodic_gpu_verification.py`; `T/_audit/render_native_raw_coordinate_fields.py`; the public case `render-history` command and report exporter. Validation-panel execution used `audit_validation_panels.py --candidate-run <exact R1 directory> --checkpoint epoch_010.pt --output-name <T50/T51/T52_R1_to20_source_selected_last_epoch010> --execute`; the earlier10-window calls additionally saved endpoints. These are completed read-only comparisons, not extra training.

## 9. Bounded pilots, exact directories, and budget

Historical semantics remain batch32, trainfraction.15, **38 accepted updates per epoch**. Every initial scientific smoke has3 updates, then reviewed5/10/15/20-epoch segments where warranted. All independent lineages—including scientific fixes—start freshly from the LIVE source SHA in §1. Unchanged arms resume their own child; a changed B definition uses a new versioned lineage. No epoch redefinition or parent post-training endpoint is used.

The ledger has **13 lineages,29 segment records,3432 accepted/attempted updates**, **3280 durable** and **152 lost accepted updates**. Lost/replayed work still consumes the global40000 budget. At batch32 that is109824 presentations (not unique samples) and90.315789 aggregate epoch-equivalents. The longest lineage has760 updates=20 epochs, below249; global headroom36568 was not used for an uncontrolled search. T53 was planned but never executed. Synthetic unit fixtures are explicitly separate from this scientific ledger.

The matrix's maximum initial budgets were T00 75,T10 100,T20/T30/T40 50,T50 175,T51/T52 100,T53 150, with at most two targeted fixes/retries per failing stage; any extension remains evidence-gated and below the cumulative lineage/global limits. Actual stage totals: T00 190,T10 380,T20 190,T30 193,T40 190,T50/T51/T52 763 each. Initial v4 B smokes T30/T50/T51/T52 (3 updates each) are quarantined because asymmetric floors fail identity. Their preserved results are not pooled with v4.1 R1 arms.

### 9.1 Latest fixed-panel measurements for all lineages

Ratios below are each arm's **own versioned raw A/B/C loss divided by its matched source**; never compare old/new raw denominators. Runtime is the median of epoch-mean update times for available epochs1–5; smoke-only entries have no epoch runtime. Peak is process CUDA bytes/2^30. Selected step0 means source fallback. For the failed original T10, the latest *measured* validation is step3; step38 is only its last durable checkpoint, while190 accepted updates include lost work.

| Arm | Accepted/durable updates | Accepted epochs/config cap | Selected step | Measured step | A/B/C source ratios | Epoch-mean median seconds | Peak GiB |
|---|---:|---:|---:|---:|---|---:|---:|
| T00 | 190/190 | 5/75 | 0 | 190 | 1.415227/0.814621/0.518489 | 1.565 | 27.38 |
| T10 | 190/38 | 5/100 | 0 | 3 | 1.376530/1.091048/0.702195 | 2.131 | 23.49 |
| T10 R1 | 190/190 | 5/100 | 0 | 190 | 1.148488/0.950260/0.496299 | 2.179 | 23.86 |
| T20 | 190/190 | 5/50 | 190 | 190 | 0.965470/0.998810/0.531969 | 2.227 | 23.93 |
| T30 | 3/3 | 0.078947/50 | 0 | 3 | 1.204051/0.980990/0.745098 | — | 23.67 |
| T30 R1 | 190/190 | 5/50 | 0 | 190 | 1.045378/1.073859/0.533316 | 2.207 | 23.86 |
| T40 | 190/190 | 5/50 | 0 | 190 | 1.085432/1.010101/0.504595 | 2.396 | 23.91 |
| T50 | 3/3 | 0.078947/175 | 0 | 3 | 1.109681/0.992832/0.735787 | — | 23.77 |
| T50 R1 | 760/760 | 20/175 | 190 | 760 | 1.204470/0.916930/0.482059 | 2.437 | 23.98 |
| T51 | 3/3 | 0.078947/100 | 0 | 3 | 1.064819/0.992816/0.771422 | — | 23.77 |
| T51 R1 | 760/760 | 20/100 | 380 | 760 | 1.189884/0.864617/0.446469 | 2.431 | 23.98 |
| T52 | 3/3 | 0.078947/100 | 0 | 3 | 1.070643/0.999495/0.765121 | — | 23.77 |
| T52 R1 | 760/760 | 20/100 | 380 | 760 | 0.959603/0.993884/0.456274 | 2.449 | 23.99 |

Every latest model-unit MSE relative change, in percent (positive is worse):

| Arm | Measured step | Total | CH4 | CO | T | U_1 | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| T00 | 190 | +20.4192 | +4.4548 | +6.8894 | +3.8938 | +17.0341 | +38.2103 |
| T10 | 3 | +16.7121 | +2.6074 | +5.4981 | +2.5891 | +12.6572 | +33.5604 |
| T10 R1 | 190 | +5.7427 | -3.9113 | +0.0704 | -2.2554 | +5.5759 | +12.0016 |
| T20 | 190 | -2.3354 | -3.4406 | -0.2502 | -0.7395 | -3.7707 | -1.3518 |
| T30 | 3 | +8.0818 | -0.1834 | +3.5557 | +0.6043 | +5.2333 | +17.4372 |
| T30 R1 | 190 | +2.2446 | -2.5837 | +2.2911 | -0.0084 | +1.1419 | +5.0815 |
| T40 | 190 | +0.7833 | -1.1582 | +1.9296 | +2.4477 | -2.6029 | +5.3515 |
| T50 | 3 | +7.9697 | -0.7362 | +3.4275 | +0.1650 | +7.0664 | +14.5306 |
| T50 R1 | 760 | +14.8518 | +1.1882 | +4.9877 | +3.4304 | +12.1485 | +28.4371 |
| T51 | 3 | +4.9992 | -1.2958 | +1.5944 | -0.4567 | +4.0050 | +10.2934 |
| T51 R1 | 760 | +12.6818 | +0.6073 | +5.4710 | +4.1945 | +9.2381 | +25.2136 |
| T52 | 3 | +5.1793 | -1.6911 | +1.8052 | -0.9554 | +4.2684 | +10.5377 |
| T52 R1 | 760 | +0.1774 | +2.8153 | +7.2503 | +5.8931 | -3.7115 | +0.8642 |

T20's single feasible5-epoch point is useful isolated A feasibility, without seed/audit robustness. T30 R1 p+5.0815% and T40 p+5.3515% narrowly fail the predeclared5% gate; the gate was not relaxed. Nine of13 lineages select the source fallback; only T20 and integrated R1 T50/T51/T52 select trained candidates.

### 9.2 Exact run-local configurations and checkpoint identities

The following directories are relative to **T**. Each has `resolved_config.yaml` (file SHA below), `run_manifest.json`, `status.json`, `artifacts/{normalization.pt,global_distribution_family.pt,cross_spectrum_family.pt,topology_family.pt,coherence_calibration.json,evaluation_query_indices.pt,evaluation_sensor_manifest.json}`, `metrics/history.jsonl`, and `evaluation/{before.json,selected.json,after.json}` where durable completion permits. The failed T10 status is stale `running`/38; the ledger and process evidence establish failure, not a live job. Its step3 selection data remains valid and no unavailable final190 result is fabricated.

| Arm | Exact run directory | Resolved YAML file SHA256 |
|---|---|---|
| T00 | `T00_legacy_control/20261002T220726Z_753954bd` | `4191546d47b1b82381ee9a6d262611d7161bc55dd8ef9025202fdaa927736a82` |
| T10 | `T10_adaptive_controller/20261002T221009Z_439dc0b4` | `bb5c66804923370e4f40df4d1534ba12fa765eae2c16e7459e8bdb18596ef459` |
| T10 R1 | `T10_adaptive_controller_r1_recovery/20261002T231232Z_5e1cf75a` | `6d3b7ef18e69722b6c4bd849fe703bb894af5c7d5a2f0b4457059f2b6f87ae2a` |
| T20 | `T20_A_copula/20261002T221244Z_f1671fa1` | `3a2c05b190d258fca8c05c143cd08bc8b18bbed20a2ec9951a3c5dca9c8b000d` |
| T30 | `T30_B_second_order/20261002T221454Z_a0fb59ff` | `92d88a7f37be94cd0a9495ca665c1611ac6961a04c6523dcdc687441a3b7d427` |
| T30 R1 | `T30_B_second_order_r1_symmetric_floor/20261002T231316Z_63838948` | `f85c17f34c357feb7a5df7828a1480eee312e3ee68a118d78f7206a0bf9be9f3` |
| T40 | `T40_C_budgeted_lines/20261002T221702Z_248e207c` | `0f308bfb9a9d67feb1305e7479ad05baf9f61433913e048f1e3d6cfc742165cd` |
| T50 | `T50_ABC_config/20261002T221954Z_879791c5` | `fbc222d2f2451179927b26b95dd50b7326a8afce39a2238a69f7c289e8cc9635` |
| T50 R1 | `T50_ABC_config_r1_symmetric_floor/20261002T231401Z_11d0898d` | `a2eb9457f25edbe62ec5677e876877499f662ba63e78efbc5e4912e73e10a6c6` |
| T51 | `T51_ABC_weighted/20261002T222147Z_b12ed3ee` | `2cc71e86ff0245b00d840c26648aa8b3d3f3a2aa745e24c6d56562f79cbd928e` |
| T51 R1 | `T51_ABC_weighted_r1_symmetric_floor/20261002T231455Z_03c36d0f` | `6848e490cdc368e45e94e60261348cca46aa7919b73cf26a6319ddaf20084e07` |
| T52 | `T52_ABC_cagrad/20261002T222639Z_7df08007` | `2e22cae2eed043c488b54e8a8993d24ca7b2eaa57a6b1704bb5eddef2a35e4a4` |
| T52 R1 | `T52_ABC_cagrad_r1_symmetric_floor/20261002T231548Z_2f456de6` | `830719d6c694dd4b2844bd68a5345a3d357d4ed61e57a1135d09355dad580f28` |

Selected model files are `checkpoints/best.pt` in those directories, explicitly bound by manifest/selected report; latest files are `checkpoints/last.pt`. Complete selected file SHAs are:

| Arm | Child step | Interpretation | Raw A/B/C source ratios | Selected SHA256 |
|---|---:|---|---|---|
| T00 | 0 | source fallback | 1.000000/1.000000/1.000000 | `c462a3141263efc46b779195dc6fd64f6c36cf3065e3b8a6895d442a439b58f5` |
| T10 | 0 | source fallback | 1.000000/1.000000/1.000000 | `b3b5941ca5506f3fbc65420e6bcea8a00e113e6b106e7603cbe66ab593be4750` |
| T10 R1 | 0 | source fallback | 1.000000/1.000000/1.000000 | `e01b6aa3815d353eaf1d8c705adfbef4152c49a7f6182b694e62a384150d82b2` |
| T20 | 190 | trained, fixed-panel eligible | 0.965470/0.998810/0.531969 | `80760ee0c214fc693cfc58a9cbbb0fe1f1be16efd8f6e6ae44184b26cbb0525b` |
| T30 | 0 | source fallback | 1.000000/1.000000/1.000000 | `060640be0ae85c637aa7eb9001c10d49ab32411386c842dbdf7944c12f0ad1c8` |
| T30 R1 | 0 | source fallback | 1.000000/1.000000/1.000000 | `8243b1a82e2ce2c9a01dbd1fc2f672553db11274658aa78ba156c47153a29dc2` |
| T40 | 0 | source fallback | 1.000000/1.000000/1.000000 | `f56fec18100abb460a26134c8332b745894e512ce8e2d94f0145681921927251` |
| T50 | 0 | source fallback | 1.000000/1.000000/1.000000 | `b49f7fcd2ca385d07e8dc8017ee5cb08763afb0baa0bf8eaba508346785b91c5` |
| T50 R1 | 190 | trained, fixed-panel eligible | 0.892543/0.997583/0.558204 | `045303026bae3e4760dbf4dc81c24a1d6f90d965ebb8a74a233cf4a2888e9e11` |
| T51 | 0 | source fallback | 1.000000/1.000000/1.000000 | `6dd34778e7cfe27351cd59981b780ff0d15377cfbc4ce77c1498e0203971bbd3` |
| T51 R1 | 380 | trained, fixed-panel eligible | 0.820693/1.052920/0.520713 | `7375708f87b54e3426d1d27dc26f321c939e6d9cb7389865ae578f3bb6fc88ca` |
| T52 | 0 | source fallback | 1.000000/1.000000/1.000000 | `7a677f80752bd2972c8731e4b15cc852284a67497f107a0ed3f92fec4850b583` |
| T52 R1 | 380 | trained, fixed-panel eligible | 0.857700/1.104266/0.646930 | `b0601f3358083ab624b7830ab09ecbdf84e93e78a99f75329fa29fef0dee2216` |

`T/_reports/pilot_inventory.{json,csv,md}` supplies all source/last hashes, exact runtime/observation/family/query seeds, resolved optimizer/controller fields, calibrations, revision types and29 segment boundaries. The latest source fallback is a child wrapper around original tensors; wrapper serialization hashes differ from the immutable original source file. Identical saved four-panel metrics for selected and epoch010 roles in T51/T52 do **not** imply byte-identical checkpoints; their separately serialized files have different hashes, and final20 prediction arrays were not cached.

Experimental input profiles remain `T/_configs/{T00,T10,T20,T30,T40,T50,T51,T52,T53,T10_R1,T30_R1,T50_R1,T51_R1,T52_R1}.yaml`; T53 is unexecuted. The bounded wrapper is `scripts/training/run_upgrade_1002_pilot.py --config <profile> --max-steps <additional segment updates> [--resume <same child>]`, with `--dry-run` for validation. It refuses epochs≥250 even with a tiny max-steps, escapes/incorrect source/device identity, changed resume config and per-lineage/stage/global cap excess. It records accepted lost/replay work conservatively. The final queue ended; no training remains queued.

## 10. Fluctuation, independent validation, and runtime

All integrated R1 arms stop at20 epochs. The first100-epoch optimizer comparison is **INCONCLUSIVE / NOT COMPLETED**, not a matched100-epoch result. Continuation was stopped after repeated fixed-panel infeasibility and failed disjoint audits, using the plan's early-stop condition.

| R1 arm | Selected epoch/step | Fixed total MSE change | Selected raw A/B/C ratios | S | Unique feasible / all non-source checks | Median S over feasible checks |
|---|---|---:|---|---:|---:|---:|
| T50 ConFIG | 5/190 | −6.4055% | .892543/.997583/.558204 | .816110 | 2/5 | .856781 |
| T51 weighted | 10/380 | −7.4487% | .820693/1.052920/.520713 | .798109 | 2/5 | .803764 |
| T52 CAGrad | 10/380 | −7.4985% | .857700/1.104266/.646930 | .869632 | 1/5 | .869632 |

Unique checks are3/190/380/570/760; regular5/10/15/20 feasible counts are2/4,2/4,1/4. Duplicate periodic/terminal rows are removed by run+step:47 stored rows across the study become40 unique checks (7 duplicates). Source0 is excluded from trained-feasibility fractions. Two or one feasible points do not establish a stable window, seed sensitivity, or confidence interval. B regressions **+5.2920% (T51)** and **+10.4266% (T52)** remain exposed despite A/C reductions; no post-hoc family guard or weight change was added.

Each final source/selected/last/epoch010 audit has four matched32-snapshot panels: fixed8000–8031, same IDs with alternate sensors/noise, disjoint8032–8063, disjoint8064–8095. Every selected candidate passes the first two and final panel but fails8032–8063:

| Selected R1 arm | Disjoint8032 total MSE change | Disjoint8032 p change | Other failed fields | Audit decision |
|---|---:|---:|---|---|
| T50@190 | +6.877% | **+24.7408%** | total,CH4+5.1662%,CO+5.9279%,T+5.0552% | FAIL |
| T51@380 | +4.183% | **+10.9390%** | CO+5.0958%; total itself passes | FAIL |
| T52@380 | +4.40% | **+11.4899%** | none; total itself passes | FAIL |

Full vectors/raw losses are `T/_reports/final20_four_panel_A/final20_paired_validation_rows.csv` (48 role/panel rows) and its JSON summary. These are LIVE/LIVE paired checks, not historic EMA comparisons.

Latest step760 also demonstrates panel dependence: T50 fails fixed total/U_1/p and alternate, passes8032, fails8064 on U_1+5.53%; T51 fails fixed total/CO/U_1/p and alternate, passes8032, fails8064 on U_1+5.60%; T52 fails fixed **CO/T** (total+.1774% and p+.8642% pass), passes alternate, fails8032 total/CO/T/p, passes8064. A low latest S never overrides fidelity failure. No all-panel feasible integrated recipe was found.

Matched-form runtime uses median **epoch-mean** update times in both numerator and denominator. T00 epochs1–5 median1.5647079199s; R1 epochs1–5 T50/T51/T52=2.4365600/2.4313603/2.4491854s (1.5572/1.5539/1.5653×). R1 epochs11–20=2.4438218/2.4738764/2.4801228s (1.5618/1.5810/1.5850×). All exceed the rough1.5× target. Different training phases/windows limit causal overhead attribution. Late individual-update medians2.39319/2.41529/2.42039s are separately descriptive and not divided by an epoch-mean baseline to claim the target passed. No OOM occurred; roughly24.0GiB R1 peaks are lower than T00 27.38GiB. Scientific fidelity gains do not justify accepting the measured overhead because the robust fidelity gate failed.

## 11. Legacy continuity and native topology evidence

Cached5/10-epoch endpoints were rescored with explicitly frozen legacy definitions, independently of new scores. B uses the pinned f785 v3 basis/statistics. Fixed-panel selected legacy same-/cross-frequency ratios: T50 **1.0968/1.2127**, T51 **1.0099/1.0843**, T52 **1.0712/1.2378**. New coefficient blocks and old squared-energy covariance are different targets; old B regressions are not hidden. B CPU source replay relative differences4.97e-7/2.42e-6 satisfy declared rtol5e-6/atol1e-8. Exact tables/plots: `T/_audit/B/pilot_monitor/legacy_v3_all_arms_all_panels.{json,csv,md}` and `legacy_v3_{same_frequency,cross_frequency}_ratios.{pdf,png}`.

Legacy A is pinned f785 v1; legacy C uses strict T00 v3 artifact identity under80e. The initial f785 C artifact loader rejection is retained; the separately attested f785→80 source scalar/input-gradient parity is zero, without bypassing the implementation hash. Cached scoring reproduces fixed source A .7141329050/C .0132849850; 816 component/raw/ratio rows and111 unchanged input hashes are in `T/_reports/legacy_AC_cached_validation/legacy_AC_cached_validation.{json,csv,md}`. An attestation's C scalar differs by9.31e-10, below the declared tolerance, and is recorded explicitly.

Those earlier endpoint caches capture the generation-time child-manifest SHA, but the original complete mutable manifest bytes were not snapshotted. Continuation changed that manifest. Replay binds the captured SHA through unchanged cache/audit/endpoint metadata and immutable ledger invocation snapshots; it does not pretend the current manifest is the old file. Legacy latest760 scoring was **not run** because final20 audits do not duplicate endpoint caches. It is not inferred from the10-window latest.

### 11.1 Archived T50@190 generated native fields

`T/_audit/C/native_generated_T50_R1_first5/` preserves full generated/reference fields, coordinates/sensors, original native/coarse/sparse PH, bars and PDF masters on four validation snapshots8032–8035. All40300 native queries come from one joint model rollout per role, not independently generated tiles. Selected and then-current latest are both190 and have identical archived fields; they are not two independent candidates and not the final760 checkpoint.

Four-snapshot normalized total MSE changes−2.85%, but **p+26.164% fails**. One warmup and one timed GPU batch4 per role gives source/selected/latest≈.047927/.047427/.047446s, peak1.34GiB; those are native inference timings excluding PH. Uncached CPU36-filtration snapshot PH≈1.226–1.293s native and≈.125/.129s coarse/sparse. GT-only reference costs differ (≈.358/.0459/.0511s per snapshot); they are not full training step costs.

The later GT-relative bar audit uses the **already archived fields and bars**, with no new inference or model-bar recomputation: `reference_ph_audit/{reference_ph_audit.json,ground_truth_reference_bars.npz,model_to_reference_per_filtration.csv,model_to_reference_group_summary.csv,reference_ph_summary.md,manifest.json}`. It stores1728 GT bar arrays and2592 role/reference comparisons; selected/latest576 bar arrays coincide. Eight samples for self group means are4 snapshots×2 signs;128 for mutual are4×16 lines×2 signs, with dependence rather than128 independent snapshots. Finite distance is normalized separately from unscaled .1 essential distance. The display diagnostic uses a common **native40300 denominator** for finite distances across representations and unweighted group means; it is not the line-weighted training objective. Its18 selected/source group ratios span.390498–1.012018, with sparseT H0 regression. Coarse/sparse PH is not native-equivalent.

New raw-coordinate panels are `T/_reports/native_fields_raw_coordinates/validation_{8032,8033,8034,8035}_raw_coordinates.{pdf,png}`. Raw x bounds [.0003868075,.9971275330], y[−.0994711965,.1491659880], unknown units; equal coordinate aspect, shared per-field value and symmetric error scales across all roles/states, T-only sensor overlays. Source NPZ SHA `43a3f6dda312f67dca5547b33bdade122c5ee1e7d910a227ce222ee130ad330b` remains unchanged. V1 exports are preserved in `layout_trial_v1`; V2 corrects PDF title/y-label clipping. All four final PDFs have437 text boxes and0 outside; `qa_v2_compact.json` verifies bounds, scales, sensors and hashes.

### 11.2 Default-off, recoverable periodic native audit

Commit7e adds `evaluation.native_topology_audit` with strict `enabled`, `every_steps`, `max_samples` (2–4), `seed`; omitted means OFF. It requires paired cubical2D/fixed-shared semantics and checkpoint-compatible cadence/full saved master bank. It restores the child's exact topology artifact, verifies source/model/config/family/bank/geometry identities, selects only2–4 **disjoint validation** snapshots before loading their dense fields, evaluates all native queries and all16 lines without smoothing, and records normalized/decoded reconstruction plus per-field/observed/unobserved diagnostics where the evaluator provides them, H0/H1 finite/essential bars, and actual timing/memory scope.

Audits run after accepted durable checkpoints; segment-final audits wait for forced last saves. Resume replays a missing due audit before further updates. A bound same-step report replays with zero forwards; conflicting identity rejects. Audit restores Python/NumPy/Torch CPU/CUDA RNG, all module train/eval modes and EMA-selection flag, preserving training/optimizer/selector/controller state. Code: `training/native_topology_audit.py`, hook in `post_training.py`, strict schema in `config/validate.py`. It leaves PH pairing/evaluator/RunStore cores intact.

Actual physicalGPU0 helper checked source LIVE and T51 selected380 on validation IDs**8039/8643**,40300 points each/full16 bank, two evaluation forwards, **zero optimizer updates**, zero additional replay forwards, identical original model/RNG/mode/checkpoint hashes. Complete audit wall before serialization **5.1921s**; source evaluation1.7596s/candidate.8176s and bounded bar diagnostic2.4623s (cold-cache asymmetry limits comparison). Process peak.708GiB is helper allocation scope, not cadence-on-training peak. Two-snapshot total MSE ratio.976758 and all field ratios≤1.05 are an execution smoke, not release evidence. Exact report: `T/_audit/C/native_periodic_gpu_verification/verification_summary.json`; command/log `_audit/C/native_periodic_gpu_verification.py`, `_audit/native_periodic_gpu_verification.log`.

Actual training-on/off/resume parity is demonstrated in the CPU lifecycle fixture (step2 periodic/step3 terminal, resume to4). No scientific GPU pilot used this late optional hook. Its training concurrency/long-run memory/cadence overhead remain unmeasured, so the helper alone does not validate formal readiness.

## 12. Key figures and machine-readable evidence

A short figure index: [fidelity panels](#figure-1-four-panel-fidelity), [family scores](#figure-2-raw-family-ratios), [Pareto trajectories](#figure-3-fidelitycoherence-pareto), [native fields](#figure-4-native-generated-fields), [topology representations](#figure-5-nativecoarsesparse-topology), [gradient diagnostics](#figure-6-controller-and-gradient-history). Every raster below has a retained PDF master next to it; numerical tables remain separate from rendering.

### Figure 1: Four-panel fidelity

![Four-panel source-relative fidelity ratios](/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction/cases/turbulent_combustion/runs/Test_1002/_reports/final20_four_panel_A/final20_fidelity_source_ratios.png)

Selected190/380/380 preserve fixed-panel total and five-field gates, but all fail the first disjoint panel on p (+24.74/+10.94/+11.49%). Shared.8–1.4 axes include all ratios (.84037–1.34591), exposing rather than clipping negative results. These32-snapshot panel means show validation sensitivity; they do not establish independent-trajectory uncertainty. PDF and48-row CSV: `T/_reports/final20_four_panel_A/final20_fidelity_source_ratios.pdf`, `final20_paired_validation_rows.csv`.

### Figure 2: Raw family ratios

![A/B/C source-relative family ratios](/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction/cases/turbulent_combustion/runs/Test_1002/_reports/final20_four_panel_A/final20_family_source_ratios.png)

The selected fixed-panel C ratios.5582/.5207/.6469 improve, but T51/T52 B ratios1.0529/1.1043 regress. Latest coherence improvement can coexist with field failure. Each panel has its own matched source; all ratios use the same new definitions within an arm. PDF/raws: `final20_family_source_ratios.pdf`, same JSON/CSV directory.

### Figure 3: Fidelity–coherence Pareto

![Frozen balanced scores versus worst total or field fidelity](/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction/cases/turbulent_combustion/runs/Test_1002/_reports/pilot_pareto_topology_A/abc_fidelity_pareto_steps.png)

This plot uses15 unique trained arm-step points plus coincident source records, with x=max(total/five-field source ratio)−1 and the predeclared5% boundary. Ten points fail; selected worst-risk changes1.5182/.6605/.5250% have S.8161/.7981/.8696. The envelope summarizes saved points without a confidence interval or superiority claim. PDF/CSV/hash QA: `T/_reports/pilot_pareto_topology_A/{abc_fidelity_pareto_steps.pdf,abc_fidelity_pareto_points.csv,output_manifest.json,visual_qa.json}`.

### Figure 4: Native generated fields

![Raw-coordinate native fields and signed errors on validation8032](/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction/cases/turbulent_combustion/runs/Test_1002/_reports/native_fields_raw_coordinates/validation_8032_raw_coordinates.png)

GT/source/selected190/latest190 and signed errors share fixed field/color scales. The four-state full-grid comparison has total−2.85% while p+26.164% fails, so a visually improved scalar field or lower aggregate cannot certify fidelity. Values and coordinate units follow dataset metadata (unknown physical unit labels). Four PDF masters, manifest and all remaining state panels are in the same directory; source numerical archive is the NPZ in §11.1.

### Figure 5: Native/coarse/sparse topology

![Archived ground-truth topology ratios by representation](/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction/cases/turbulent_combustion/runs/Test_1002/_reports/pilot_pareto_topology_A/native_topology_selected_source_ratios.png)

Across18 H0/H1 self/mutual groups, selected/source descriptive ratios.3905–1.0120 reveal one sparseT H0 regression. The common native40300 finite denominator and unscaled .1 essential term make the display a separate descriptive diagnostic; nested line-weighted training C differs. Four snapshots, two signs and shared banks are correlated replicates, without CI. PDF/tables: `native_topology_selected_source_ratios.pdf`, `native_topology_selected_source_ratios.csv`, byte-identical54-row `native_topology_group_summary_input.csv`.

### Figure 6: Controller and gradient history

![Actual gradient geometry and AdamW displacement for T50](/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction/cases/turbulent_combustion/runs/Test_1002/_reports/T50_R1_epoch20_history_80e17c9/adaptive_gradient_geometry.png)

![Fidelity controller history for T50](/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction/cases/turbulent_combustion/runs/Test_1002/_reports/T50_R1_epoch20_history_80e17c9/adaptive_controller_state.png)

Late T50 records49/380 explicit ConFIG fallbacks and352/380 clipped updates; diagnostic descent geometry does not prevent independent pressure failure. These panels use child update coordinates and distinguish pre-step risks from post-step multipliers. Parallel T51/T52 reports and all PDF/SVG masters: `T/_reports/T{50,51,52}_R1_epoch20_history_80e17c9/`; exported JSON/hash manifest `T/_reports/final20_public_history_reports_manifest.json`.

Final20 detailed audit directories are `T/_reports/validation_audits/T{50,51,52}_R1_to20_source_selected_last_epoch010/`. Their `audit.json.gz` and16 role/panel metric JSON.gz files per arm are **losslessly compressed**, with original/gzip SHA256, byte counts and verified decompression in each compression manifest. Sensors, hashes, histories and numerical content remain recoverable (`gzip -dk <file.json.gz>`). Older10-window caches remain unchanged. Completed synthetic pytest fixture directories also have verified recovery archives/manifests under `_audit/completed_synthetic_fixture_archives`; no source or actual pilot checkpoint/history was deleted. Only three reproducible compiler .so cache files were removed after no-open-handle/hash checks (`compiler_cache_cleanup_receipt.json`). Shared-volume free space varied independently; no payload was relocated outside T.

Additional complete indexes: `T/_reports/pilot_inventory_final_inputs/` contains the frozen JSON/CSV/Markdown inventory, release-gate review, manifest and SHA256SUMS; live-path copies are `T/_reports/pilot_inventory.{json,csv,md}` and `release_gate_review.md`. The frozen inventory JSON SHA256 is `6acf671f49d024f35a8436a403a5f3dc8e8550f2b735dc94eda8e9e450c9d6b6`; bundle manifest SHA256 is `f48c8740c50e99cece16f2957e7d62a091b279a46d50c1058b314edeb77427a2`. The budget and stopped-study state are `T/_audit/stage_ledger.json` and `study_final_state.json`. Evidence is machine-local/ignored; the branch contains source/tests/documentation, not large run data.

The final Pareto/topology figures read frozen input copies under `T/_reports/pilot_pareto_topology_A/inputs/`. An earlier derived inventory JSON was replaced when final test-status metadata was corrected; its original SHA256 (`387da03677ace830745b6530a9a394e84aaf0cdf983b98823d7228f36f8eb37b`) is recorded, but its exact bytes were not archived and are unavailable. Earlier exports/metadata remain in `superseded_metadata_v1/`, and the subsequent unpadded frozen-input batch remains in `superseded_layout_v2/`; no historical-byte recovery is claimed. All three scientific CSV exports and all plotted-data hashes/axis limits match the preceding batch. Final PDF padding is0.18in: both PDFs have zero out-of-page word boxes, with title minimum y=+1.729475pt. After cropping the added white border, topology pixels are identical and Pareto differences are confined to y-axis tick-label antialiasing (0.0638% of interior pixels). Exact checks are `frozen_input_rerender_qa.json` (SHA256 `278c3b1328a28d8a23fbb8f2f2b759d0a78c151891b7fa8346c2cf50e78d8184`); final `output_manifest.json` SHA256 is `28ac5aae20f5f44b8a97c4f2edcc855152d27a47a5b40bfe4ccb83a3f4b8378d`. This export repair added no training or inference.

## 13. Release decision, conditional formal preparation, and limits

| Plan gate | Status | Evidence boundary |
|---|---|---|
| Legacy compatibility | PASS | Final7e full565/8 pass, source/value/gradient parity; physicalGPU1 tests retained and skipped under the GPU0-only boundary |
| A definition/gradient tests | PASS | Own-rank/soft-CDF/tail math tests and actualGPU0 focused checks |
| B second-order/block/invariance tests | PASS for4.1; original4 FAIL/quarantined | Symmetric-floor identity and signed/block tests; no reinterpretation |
| C native/raster/subset tests | PASS | Core/parity/native gradient/subset convergence; optional cadence GPU-training overhead unmeasured |
| Controller/resume correctness | PASS | State/RNG/optimizer/teacher/selector lifecycle, off-panel recovery; no generalization guarantee |
| GPU0 runtime/memory | FAIL runtime target; PASS execution/memory | No OOM; matched-form ratios1.56–1.59× exceedrough1.5× |
| Bounded stage feasibility | FAIL for observed integrated recipe |20-epoch isolated feasible points do not survive the disjoint fidelity gate;100-epoch optimizer comparison remains INCONCLUSIVE |
| Final fidelity eligibility | FAIL | Every integrated selected candidate fails disjoint8032 pressure |
| Joint coherence improvement/family regressions | INCONCLUSIVE for release | Lower selectedS and C; new/old B regressions exposed; held-out fidelity fails |
| Second-seed/audit stability | NOT RUN for second seed; validation stability FAIL | Four-panel rechecks fail; no qualifying recipe to confirm |
| Formal launch preparation/validation | BLOCKED / NOT RUN | Conditional work-plan§14 gates do not pass |

There is **no locked release recipe**. T51@380 has the lowest fixed-panel selected S among these short pilots, but B worsens5.292% and disjoint p worsens10.939%; this is not a recommendation to formalize weighted sum or an optimizer-superiority result. T50 and T52 also fail independent fidelity. No field budget, descriptor definition or selection rule was changed after observing the failed audit.

Per work-plan§12, the authorized outcome is an honest bounded negative-study report and **blocked scientific release**. Work-plan§14 explicitly conditions formal-profile creation/smoke/launch validation on all release gates passing. Therefore **no new5,000-epoch formal configuration or launch command is certified or provided**, and no formal job, T53, locked test-set audit or background queue was launched. Experimental profiles stay available for inspection, clearly separated from release-ready configuration. A5000 profile would not repair the missing feasible recipe; writing an apparently approved command would misrepresent the gates. Any further scientific fixes/100-epoch or second-seed work needs a new bounded scope decision.

Limits: one chronologically correlated trajectory and single training seed; at most20 epochs; only one/two feasible evaluations per integrated arm; no all-panel feasible upgrade; no robust optimizer ranking/5000 convergence proof; nonlinear smooth-copula bias and atom ties; finite directions/landmarks/truncated spectral modes/low-energy floors/rank31; covariance mean/amplitude/fourth-order blind spots; PH nonsmooth ties/finite slices/interpolation error; unknown field/coordinate physical units; no PDE/divergence or physical-causality certificate; imperfect earlier mutable-manifest archival; late-only exact per-update telemetry; optional native audit tested in actualGPU0 inference and CPU resume, not ongoing scientific GPU training. Exact residualized LPC, State→Scale canonicalization, temporal/physical coherence, new architectures and source retraining remain explicitly deferred.

## 14. Changed-code inventory and external provenance

Through implementation7e,48 tracked files changed (13133 insertions/157 deletions including extensive focused tests/docs); existing modules/packages remain in place. The main code boundaries are:

| Existing boundary | Change/symbol mapping |
|---|---|
| `coherence/families/global_distribution/` | Opt-in v2, own-CDF/tail helpers, masks and saved banks; v1 preserved |
| `coherence/families/cross_spectrum/{basis,family,covariance_blocks}.py` | Complete bands, signed centered coefficient covariance, TRAIN floors/masks/version4.1 |
| `coherence/families/topology/{family,geometry,persistence_objective}.py` | Deterministic master/subset lines, packed critical gradients/native gather and supported area averaging; `persistence.py` pairing core unchanged |
| `training/{gradient_balance,gradients,fidelity_controller,post_training}.py` | K methods/common layout, same-noise LIVE teacher, calibrated six-risk primal–dual route, accepted-step ordering/recovery |
| `training/{checkpointing,run_store,coherence_history}.py` | General selector/archive, strict accepted-update stream, adaptive child-step histories |
| `training/native_topology_audit.py` | Optional disjoint full-grid durable native audit with state preservation/idempotent replay |
| `config/validate.py`, `coherence/registry.py` | Strict versioned opt-in keys/defaults/artifact recovery; old restrictions retained |
| `evaluation/{checkpoint,coherence_set,explanatory_coherence,reconstruction_set,topology_set}.py` | Explicit LIVE auxiliary restoration, old/new components, own-rank audits, signed/pool/group covariance diagnostics, persisted portable plot arrays |
| `scripts/training/run_upgrade_1002_pilot.py` | GPU/source/path/config/cumulative stage/global guard and conservative accepted-work ledger |
| `scripts/visualization/coherence_posttraining_report.py` | Reuse public rendering/export for opt-in adaptive histories |
| `tests/test_upgrade_1002_*.py`, `test_adaptive_coherence_history.py` | Mathematical, serialization/resume/provenance/budget/telemetry/native lifecycle coverage |
| `README.md`, `ModelExplain.md`, cross-spectrum README | Versioned opt-in documentation; historical recipe/report unchanged |

No model adapter/backbone/decoder, rollout, base-training, dataset loader, optimizer implementation/RunStore framework, legacy numerical definition, or source run was replaced. The launcher and audit/report additions support the stated experiment/provenance contracts rather than establishing a separate modeling framework.

Primary references checked for this work support the individual mechanisms, not a theorem about the complete algorithm:

- [Blondel et al., ICML2020](https://proceedings.mlr.press/v119/blondel20a.html): rank/sort differentiability; the chosen quantile-landmark smooth-CDF is our explicit permitted fallback, not copied fast-soft-sort code.
- [Rockafellar–Uryasev, Journal of Risk2000](https://doi.org/10.21314/JOR.2000.038): variational CVaR; softplus smoothing/identity offset are explicitly tested computational choices.
- [Marques et al., Stationary Graph Processes](https://arxiv.org/abs/1603.04667), [Kim–Oh, Cross-Spectral Analysis](https://arxiv.org/abs/2408.05961): graph spectral covariance context; off-frequency finite blocks are not nonlinear transfer evidence.
- [Carrière et al., ICML2017](https://proceedings.mlr.press/v70/carriere17a.html), [Scoccola et al., ICML2024](https://proceedings.mlr.press/v235/scoccola24a.html), [GUDHI official cubical documentation](https://gudhi.inria.fr/python/latest/cubical_complex_ref.html): existing sliced persistence/differentiation/backend; installedGUDHI3.13 APIs checked, core retained.
- [ConFIG, ICLR2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/94e85561a342de88b559b72c9b29f638-Abstract-Conference.html): installed `conflictfree==0.1.8`; individual active-family descent check/fallback is explicit.
- [CAGrad, NeurIPS2021](https://proceedings.nips.cc/paper_files/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html): author mean/rescale convention. The checked author repo revision is `dc3d48152b6196945cfd56144879b9d42353b095`; original `nyuv2/utils.py` SHA256 `caa8832ed145ce4c3f2e7b18fe711158e6ebe6f7ba093a058201262ff36a0293`. Original/reference excerpts and75-case parity are retained under `T/_audit/{cagrad_upstream,cagrad_nyuv2,audit_cagrad_reference}.py`, `cagrad_reference_parity.json`; no unverified×K scale.
- [Chamon–Ribeiro, NeurIPS2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/c291b01517f3e6797c774c306591cc32-Abstract.html): constrained-risk motivation; it does not certify nonconvex multigradient AdamW convergence.
- [Xiao et al., September2026 preprint](https://arxiv.org/abs/2609.01558): actual-displacement diagnostics motivation only; no full GUA optimizer imported.

Implementation commits are the milestones in§1 plus native audit `7e328089c7bda9ceb0a5a68dd0806e2e49594f0a`. Later documentation-only commits do not alter the measured scientific code. Branch remains `codex/phycoflow-upgrade-1002`; final local/origin synchronization is verified when the document is committed.
