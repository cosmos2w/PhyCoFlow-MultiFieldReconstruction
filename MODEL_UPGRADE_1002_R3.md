# PhyCoFlow 1002 R3 — bounded coherence refinement

**`provisional_exploratory`.** The 500-epoch training campaign and all seven robustness audits are complete: control 150, finite 150→200 and independent LIVE SOURCE seed 43-150. Coherence gains replicate, while seed-43 exact/final 50 and robustness 150 native stability fail. Selected 110 and final 200 native-field reviews are complete. The fresh SOURCE two-epoch sibling smoke completed, and all four final PDF masters are rendered; final visual/numeric verification passed; the formal 5,000-epoch run has neither launched nor been queued.

## Access and provenance

Repository `cosmos2w/PhyCoFlow-MultiFieldReconstruction`, branch `codex/phycoflow-upgrade-1002`: implementation `90c74137b4f84e5fcafd988e6b042f8f4dd86a0d` was checked against plan baseline `57dad043c2ae9268f1200770ae7645c5e1008370`, committed and pushed; four frozen YAML-only profiles were committed/pushed as `b6f855c`. [Preflight](cases/turbulent_combustion/runs/Test_1002/_audit/R3_campaign/R3_preflight.json) binds protected evidence. Scientific source, kernels and scripts remained frozen during training; after all pilots and smoke finished, only the R3 all-record report reader, essential label and all-three-case contour presentation were corrected. Six new/legacy focused checks and Ruff pass; the historical R2 reader remains unchanged.

Every independent arm starts immutable **LIVE SOURCE** `tc_gl_rbf_cq_cached_kv_5000ep/20260826T212231Z_7b68c461/checkpoints/last.pt` under the case runs tree, SHA256 `03f2f45b966a00850cc45c011b4cf1cd644b9408de355324fec8dc434a84a810`. Historical protocol is `coherence_fix_ABC_sliced_persistence_formal_5000ep_gpu1/20260924T030724Z_4208231c/resolved_config.yaml`: fields `[CH4,CO,T,U_1,p]`, native 100×403 grid, sparse T sensors, TRAIN mean/std normalization, batch 32, fraction 0.15, historical epoch exposure, 4,096 coherence queries, cached K/V, two-step Euler generation and observation correction. Execution uses `phycoflow_env`, physical GPU 0 only; TEST stays locked. Native field units are unknown; field comparisons use the recorded model-unit errors.

[R2](MODEL_UPGRADE_1002_R2.md) remains `not_recommended`, byte-identical to its protected SHA256 `fca4022e7b80bd16cdc858b55b7b298450bbd546d412946c6bca3ab09e8dc457`. Its 100 pressure failures, 150 grouped-B regressions, positive late native slopes, seed 43/local field-PH exceptions, mixed replication, endpoint-only drift and shortened lower-LR comparison retain their original interpretation. Reserved pressure 1.087122 still fails its original +5% rule despite total 1.016597. Historical passing TEST evidence does not override it or become fresh R3 confirmation; reused panels are development evidence.

## Minimal code changes

A retains own-CDF `marginal_copula_v2` and smooth tails. B retains signed second-order same-/cross-band blocks, complete eigenspaces, frozen masks/floors and grouped/pooled reporting. Model/source/rollout/recovery contracts, native-loss gradient, GUDHI pairings and live critical values remain. ConFIG stays primary; no optimizer tournament or residualized LPC pipeline.

Opt-in finite-primary aggregation preserves old topology scores and the persistence engine:

`C_old,g=F_g+0.1E_g`, with finite diagram distance F divided by HW and essential birth distance E **without HW division**. Separately frozen positive TRAIN SOURCE scales are `s_F,g=max(mean_TRAIN F_SOURCE,g,epsilon_F,g)` and similarly E. The new objective is `C_new=Σ_g w_g[0.9F_g/s_F,g+0.1E_g/s_E,g]`, `Σ_g w_g=1`. Category/group/line weights and fixed floors/zero-essential handling are recorded; historical aggregation retains its original category-weight sum 2. Raw F/E, normalized F/E, legacy C and new C remain distinct outputs. Training uses 32×128 raster/four sampled lines; validation full 16 lines; native 100×403 audits have independent TRAIN calibration.

Existing `endpoint_native_primal_dual_v2` uses `g_D=(D_live−D_source)/s_D−0.02`. Frozen TRAIN `s_D=0.13709308300167322` resolves absolute allowance **0.0027418616600334645**, rather than 2% of each batch SOURCE loss. Endpoint training budgets remain +5% total/per field. Feasible SOURCE with zero multipliers has pressure gradient 0 while raw gradients remain live: native/endpoint norms 1.153075/.290574 are means of two fixed TRAIN diagnostic batches, distinct from stochastic startup.

[Stable regression](cases/turbulent_combustion/runs/Test_1002/_audit/R3_campaign/stable_full_regression_result.json) passed 665, expected-skipped 11, failed/errored 0 with CUDA hidden and 148 source hashes unchanged. Four historical GPU 1-only checks were deliberately skipped; six relevant [GPU0 checks](cases/turbulent_combustion/runs/Test_1002/_audit/R3_campaign/topology_gpu_acceptance.json) passed separately, including GUDHI gradient/value parity and native replay. Software acceptance is not convergence.

## Prospective policy

[Frozen policy](cases/turbulent_combustion/runs/Test_1002/_audit/R3_campaign/planned_runs.json) and [panels](cases/turbulent_combustion/runs/Test_1002/_audit/R3_campaign/panel_declaration.json) predate training. Strict descriptive targets: total/each field≤1.05, every family median<1, source-like fluctuating native risk. Exploratory screening: total≤1.05, each field≤1.10, pooled matched validation-native ratio≤1.05. These are engineering screens, not guarantees or revised R2 criteria.

Candidates age ≥100 use the same selection 64 score `S=(r_A+r_B+r_Cnew)/3`; old C never drives selection. Every saved preceding 50 observation contributes: useful coherence requires two family medians≤.97 or their mean≤.95, none>1.10. Native evidence requires all declared 25-epoch observations and both frozen draws; missing coverage fails. Exact mature native, neighboring tensors, fields and replication remain separate review requirements.

Robustness 64 uses alternate sensors/noise 4027/5027 after each declared training horizon releases GPU, and never reselects. Fixed native 32 uses 2027/3027; aggregate ratios pool raw matched losses, never tiny-denominator stochastic ratios. Native-PH development snapshots are 32/437/999 (sample 8032/8437/8999), independently SOURCE-TRAIN-fitted at 0/2671/5342/7999, within the four-snapshot bound.

Routine +5% failures or local/component/noise exceptions do not stop healthy arms. After 40, only sustained 20-epoch catastrophic development patterns (total/native>1.20, field>1.35), gross collapse or software faults warrant a pause. At most one evidence-triggered 100-epoch repair is allowed; none has been selected. No broad sweep or automatic controller/descriptor tuning.

## Completed epoch horizons

[Control](cases/turbulent_combustion/runs/Test_1002/R3_10_corridor_control/20261004T145443Z_ffdf2f0b) completed 150; independent [finite](cases/turbulent_combustion/runs/Test_1002/R3_20_finite_primary/20261004T191607Z_254fbc47) completed 150 then resumed unchanged latest 150→200, with clean releases and no replay. Selected 110 was not restarted. Fresh [SOURCE seed 43](cases/turbulent_combustion/runs/Test_1002/R3_30_confirmation/20261005T013624Z_afa006a8) completed 150 from original LIVE, with clean release. [Saved 150 diagnosis](cases/turbulent_combustion/runs/Test_1002/R3_analysis/confirmation_epoch150_diagnosis.json) preserves all source/draw identities, immutable checkpoint/provenance and every saved final-window observation. Useful A/C coherence replication coexists with B regression and failed native stability; no checkpoint reselection or optional repair follows. Pressure 20/40 failures 1.126114/1.227932 remain. Actual seed-43 robustness audits and the fresh SOURCE sibling smoke are complete. [Independent 150 CPU review](cases/turbulent_combustion/runs/Test_1002/R3_analysis/confirmation_epoch150_CPU_mature_review.json) verifies LIVE/config/SOURCE/TRAIN-calibration and whole buffer/recovery parity. Scientific classification is provisional, not a native-stability success.

[Final sibling boundary](cases/turbulent_combustion/runs/Test_1002/R3_analysis/candidate_sibling2_boundary.json) and [ledger](cases/turbulent_combustion/runs/Test_1002/_audit/R3_campaign/stage_ledger.json) reconcile **502 completed campaign epochs: 500 scientific + 2 smoke**, without replay. Physical-use upper is **14.312570349 h**; evaluation upper is **1757.501276 s** (29.29 min). Native 110 56.574692 s and conservative native 200 85 s are each charged once; inline evaluation is already included in training wall and subset timers are not added again. Caps remain 610 attempted epochs including replay/smokes, every lineage <250, GPU ≤24 h and evaluation ≤2 h. The two-epoch sibling completed normally from original LIVE SOURCE, with sampled nonzero native gradients in both epochs and ConFIG primary without fallback. This validates execution, not convergence. No formal execution/queue.

## Main metric table and C breakdown

All scalar ratios below use matched LIVE SOURCE. Main rows use selection 64/group32 B/full16 raster; fixed native DEV32 is a distinct panel. SOURCE nonzero ratios 1 are reference arithmetic. [Full derived table](cases/turbulent_combustion/runs/Test_1002/R3_analysis/prospective_final_table.json) retains intermediate states and provenance.

| State | Epoch | Total; CH4/CO/T/U1/p MSE ratios | A/B/Cnew; S | Fixed native DEV32 | OldC; rawF/E | Window; adoption |
|---|---:|---|---|---:|---|---|
| SOURCE | 0 | 1;1/1/1/1/1 | 1/1/1;1 | 1 | 1;1/1 | reference |
| Control | 150 | 0.999394; 0.9858 / 1.0065 / 0.9871 / 1.0179 / 0.9612 | 0.8946 / 1.0028 / 0.9262; 0.9412 | 1.007910 | 0.7587; 0.9365 / 0.7360 | pass; comparison |
| Finite selected | 110 | 1.006442; 0.9865 / 1.0300 / 0.9843 / 0.9983 / 1.0131 | 0.9313 / 1.0023 / 0.8367; 0.9234 | 1.032422 | 0.7054; 0.8363 / 0.6887 | pass; recipe lock |
| Finite matched | 150 | 1.024611; 1.0196 / 1.0645 / 1.0258 / 1.0082 / 1.0253 | 0.9123 / 0.9932 / 0.7604; 0.8886 | 1.074943 | 0.5536; 0.7622 / 0.5270 | fail |
| Finite extended | 200 | 1.005955; 0.9521 / 1.0110 / 0.9589 / 1.0110 / 1.0154 | 0.9547 / 0.9951 / 0.8955; 0.9484 | 1.003490 | 0.7850; 0.9033 / 0.7699 | pass; provisional replication |
| SOURCE seed 43 | 150 | 1.015024; 0.9931 / 1.0705 / 1.0032 / 1.0130 / 0.9797 | 0.8573 / 1.0510 / 0.7678; 0.8920 | 1.074516 | 0.5660; 0.7652 / 0.5406 | native failure; provisional |

Frozen ranking led to exact native finite 120 1.084948 and control 110 1.085981 failures, then finite 110 1.032422 pass; its neighboring 100/120 tensors, native topology and fields were reviewed before recipe lock. Finite 120’s cadence-window pass did not override its exact failure. Locked science continued unchanged; subsequent poorer 150 results remain material.

| Horizon | TRAIN20/50 native | Previous25→final 25 TRAIN | Fixed native TRAIN/DEV | Preceding50 fixed TRAIN/DEV | All-record A/B/Cnew medians |
|---|---|---|---|---|---|
| Control 150 | 1.036712/1.024365 | 1.014650→1.034005 | 1.046407/1.007910 | 1.027901/1.037282 | .900016/.997401/.867157 |
| Finite 150 | 1.019108/1.017619 | 1.014980→1.020237 | .992794/1.074943 | .999837/1.089683 | .897457/.993166/.779970 |
| Finite 200 | 1.031384/1.035277 | 1.035505→1.035051 | 1.002287/1.003490 | 1.015508/1.033415 | .950820/1.021751/.884809 |
| Seed 43-150 | 1.019768/1.015436 | 1.010209→1.020678 | 1.003421/1.074516 | 1.038262/1.086019 | 0.879374/1.040697/0.767772 |

Every saved record is included: control final 50 has 110/120/130/140/150/150; finite extension has 160/170/180/190/200/200; confirmation has 110/120/130/140/150/150. Terminal records repeat measured values and add metadata, not independent evidence. Original diagnosis JSONs stay unchanged. Finite 180 pressure 1.118829 and recovery 190 remain included. Individual DEV 125/150/175 ratios 1.104423/1.074943/1.063339 are disclosed; finite 150 window fails despite useful coherence, 200 window passes. TRAIN last two 25 ratio at 200 falls only .045423 percentage points; mean violations +.012945/+.012733, multiplier .114878→1.039498, final 25 clipping 100%, no saturation. Confirmation final 50 DEV 1.086019 fails, with exact 125/150 1.097521/1.074516. Its TRAIN two25 risk rises 1.046810 percentage points while mean violation remains slightly negative and multiplier/clipping relax; sparse first 150 after 149 native pressure 2.0097×ConFIG (endpoint 0), own weightedF/E norms .079735559/.087606202. This is not safely-feasible cycling proof or an optional-repair decision.

Robustness results are generalization evidence; selection criteria are not rerun against them:

| State | Epoch | Total; largest field ratio | A/B/Cnew | Native64 | B eight-regroup median; range; pooled |
|---|---:|---|---|---:|---|
| Control | 100 | 1.0604; p 1.2149 | 1.1648 / 1.0217 / 0.8968 | 1.0976 | 1.0296; 0.9997–1.0786; 1.0613 |
| Control | 150 | 1.0123; p 1.0182 | 0.9471 / 0.9789 / 0.9263 | 1.0151 | 1.0161; 0.9732–1.0328; 1.0116 |
| Finite | 100 | 1.0451; p 1.1346 | 1.1031 / 1.0455 / 0.8905 | 1.0465 | 1.0601; 1.0353–1.1088; 1.1001 |
| Finite | 150 | 1.0426; p 1.0749 | 0.9675 / 1.0156 / 0.7909 | 1.1194 | 1.0229; 0.9880–1.0902; 1.0590 |
| Finite | 200 | 1.0434; p 1.0786 | 1.0500 / 0.9484 / 0.9034 | 1.0457 | 0.9592; 0.9186–0.9939; 0.9169 |
| Seed 43 | 100 | 1.0143; p 1.0341 | 0.9811 / 0.9965 / 0.8952 | 1.0363 | 0.9996; 0.9584–1.0322; 0.9941 |
| Seed 43 | 150 | 1.0289; CO 1.0706 | 0.9592 / 1.0047 / 0.7983 | 1.1003 | 1.0473; 1.0145–1.1147; 1.0813 |

[Seed-43 robustness100](cases/turbulent_combustion/runs/Test_1002/R3_analysis/R3_30_epoch100_robustness/audit.json) passes approximate endpoint/native fidelity. [Robustness150](cases/turbulent_combustion/runs/Test_1002/R3_analysis/R3_30_epoch150_robustness/audit.json) native 1.100271 fails with both draws 1.083135/1.118747, despite total 1.028932/CO 1.070560 and useful A/C gains; B median 1.047348/pool 1.081321 regresses. All these results are retained without reselection or optional repair. Control 100 pressure/native and finite 100 pressure/A/B failures remain. Finite 150 robustness native 1.119392 contrasts with control 1.015071; finite 200 aggregate native 1.045727 passes with pressure +7.86%. Regroupings reuse cached 64 coefficients without regenerating fields; group sensitivity is not an independent-data CI or a requirement that every group improve. Grouped 32 and pooled 64 remain separate estimators. Pressure retains `MSE=mean-error²+centered MSE`, without pressure-gauge subtraction; native field units remain unknown.

[TRAIN diagnosis](cases/turbulent_combustion/runs/Test_1002/R3_analysis/TRAIN_calibration_diagnosis.json) measures A marginal/copula norms .153233/.008027 and B same/cross .077429/.058197. Norm/ConFIG-direction ratios are nonadditive and may exceed 1. Control hypothetical new C uses legacy outer .2886464254; finite actual outer .0140706730 gives SOURCE weighted F/E norms .100375177/.104086256. First 200 after 199 (not post-epoch 200) has ConFIG norm .332413165, native pressure 20.9294×, endpoint pressure 0, own weighted F/E norms .110232797/.117593326. TRAIN raster legacy SOURCE weighted-essential share 85.667% differs from reused R2 native ≈98%; essential-dominated gains alone do not establish finite-shape repair.

Independently calibrated native topology covers **CO/T self H0/H1 and mutual CO+T H0/H1**; contours show all five fields:

| Native audit | Raw F/E ratios | OldC | NormalizedF/E | NewC | Local self T H0 essential |
|---|---|---:|---|---:|---|
| Selected 110 | .937818/.634869 | .645053 | .934298/.874898 | .930563 | +18.76% |
| Final200 | .968604/.766372 | .773170 | .965684/.858458 | .958942 | +2.71% |

Raw finite gain is 6.22% at 110 and 3.14% at 200; H1 essential remains 0/null ratio. Native SOURCE calibration/components and SOURCE/GT/sensor/query arrays match exactly across audits. [Resume semantic proof](cases/turbulent_combustion/runs/Test_1002/R3_analysis/finite_resume_topology_artifact_semantic_review.json) verifies scientific state/bank/scales despite changed serialized bytes: current robustness/native caches are freshly measured; old cache keys were never forced. [Full 200 parity](cases/turbulent_combustion/runs/Test_1002/R3_analysis/finite_epoch200_checkpoint_parity_review.json) binds final milestone e92a096b… and earlier buffer 8e79e346… with identical learned/recovery payload except report metadata; originals preserved.

[Native 200 float64 accounting](cases/turbulent_combustion/runs/Test_1002/R3_analysis/native_epoch200_accounting_review.json) gives three-case mean CH4/CO/T/U1/p ratios .985354/1.025175/.995292/.981946/.829600. DEV 999 ratios **1.094980/1.111110/1.085309/1.260634/.902145** retain U1 +26.06% and CO +11.11%; DEV 437 CO 1.054242 also regresses. Selected 110 includes U1 +9.51% and CH4/T/CO exceptions. The final field PDF shows source-like major structures/no gross collapse, with a stronger localized negative U1 residual 999. Aggregate gains do not imply universal component, field or local improvement.

## Limitations and readiness

**Scientific status: `provisional_exploratory`.** The executable tested recipe is promising and A/C gains replicate, but native stability does not: seed 43 final 50 native DEV 1.086019, exact 150 1.074516 and robustness 150 1.100271 fail the declared 1.05 ceiling. Neither favorable TRAIN averages nor better topology cancel those failures. B remains mixed; the modest native finite-shape gain and localized field regressions constrain the claim. The candidate is an explicitly provisional research experiment, not certified native fidelity or universal improvement. No evidence-triggered optional branch or robustness-based reselection occurred. Reused development panels, stochastic draws and correlated chronological cases limit independence. The completed SOURCE sibling provides technical execution evidence only; final PDF visual/numeric verification passed. No formal run or queue.

Four PDF-only campaign masters are rendered and verified:

- [Training and coherence — 2 pages](cases/turbulent_combustion/runs/Test_1002/R3_final_figures/R3_training_and_coherence.pdf): complete histories, pooled native rolling windows and all saved coherence observations.
- [Finite/essential topology — 3 pages](cases/turbulent_combustion/runs/Test_1002/R3_final_figures/R3_finite_essential_topology.pdf): raw/normalized terms and separate raster/native SOURCE calibration; essential distance has no HW division.
- [Fidelity and Pareto — 1 page](cases/turbulent_combustion/runs/Test_1002/R3_final_figures/R3_fidelity_and_pareto.pdf): black stars denote saved DEV robustness/native-grid audits; exact qualification is documented separately in the table/proofs.
- [Fields — 6 pages](cases/turbulent_combustion/runs/Test_1002/R3_final_figures/R3_fields.pdf): all three fixed DEV cases at 110 and 200, including worst-case 999, shared GT/SOURCE/child scales, symmetric signed errors and actual sensors.

[Numeric/hash review](cases/turbulent_combustion/runs/Test_1002/R3_analysis/final_figures_numerical_review.json) and [all-12-page visual review](cases/turbulent_combustion/runs/Test_1002/R3_analysis/final_figures_visual_review.json) pass. No PNG copies. [Completed cleanup](cases/turbulent_combustion/runs/Test_1002/R3_analysis/cleanup_final_receipt.json) removed 18 disposable directories and 4 inspection texts (355.73 MB apparent); all 23 review buffers, 46 recovery checkpoints, four PDF masters, scientific arrays/calibration/SOURCE caches and R1/R2 evidence remain protected.

## User-only formal command

The [5,000-epoch candidate](cases/turbulent_combustion/configs/posttrain/abc_upgrade_1002_r3_5000ep.yaml) and [saved command](cases/turbulent_combustion/runs/Test_1002/_audit/R3_campaign/user_formal_command.sh) initialize original SOURCE and preserve the tested science. [Final inert candidate validation](cases/turbulent_combustion/runs/Test_1002/R3_analysis/candidate_final_inert_validation.json) passes SOURCE/hashes/science resolution and shell syntax without GPU/launch calls. The fresh SOURCE two-epoch sibling completed with the same science: only horizon, checkpoint ages and output name differ from the candidate. The actual formal/sibling YAML hashes still match the reviewed preflight; no pilot checkpoint becomes formal initialization. The command below is prepared user-only execution; it has **not run or been queued**.

```bash
source /home/wanglz/miniconda3/etc/profile.d/conda.sh
conda activate phycoflow_env
PFC_R3_ROOT=/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction
export CUDA_VISIBLE_DEVICES=0
export KEOPS_CACHE_FOLDER="$PFC_R3_ROOT/cases/turbulent_combustion/runs/Test_1002/_audit/R3_campaign/keops_cache"
export TMPDIR=/tmp/pfcR3
cd "$PFC_R3_ROOT"
python -u "$PFC_R3_ROOT/scripts/training/run_upgrade_1002_r3_formal.py" \
  --config "$PFC_R3_ROOT/cases/turbulent_combustion/configs/posttrain/abc_upgrade_1002_r3_5000ep.yaml" \
  --stop-on-sustained-emergency --execute
```
