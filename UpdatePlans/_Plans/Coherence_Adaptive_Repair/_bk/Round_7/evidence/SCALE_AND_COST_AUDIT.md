# Round 7 evidence: Scale component tradeoffs and execution cost

Date: 25 November 2026. Audited checkout: `Coherence_Adaptive_Repair`, commit `b5bbf8f`. This note separates recorded learning evidence, a new arithmetic readout of existing cached results, and the proposed interpretation. The audit performed no model inference, optimizer actions, training or GPU work. Historical reports and raw results were not modified.

## 1. Evidence and the missing TRAIN component readout

The [original Round 6 report](../../../../../_Reports/Coherence_Adaptive_Repair/Round_6/report.md) and [Scale alignment supplement](../../../../../_Reports/Coherence_Adaptive_Repair/Round_6/Scale_Alignment_Followup/report.md) retain SOURCE/C8. The supplement established that the original epoch-500 model improves the complete Scale composite on TRAIN800 but worsens it on the reused VALIDATION100 panel, even when both splits use the exact trained coefficients and complete query population. Definition mismatch was real, but correcting the scoring definition did not remove the transfer failure.

The following additional table was reconstructed from already recorded `component_paired_summary` values, using the saved positive weights and the original frozen TRAIN unit `u = 0.25978530736085925`. No endpoints or metrics were regenerated. For each semantic block, its recorded raw attraction, separation and score changes were multiplied by its exact training weight and divided by `u`.

| Original epoch 500 minus SOURCE | TRAIN800 | VALIDATION100 |
| --- | ---: | ---: |
| Complete expanded prefit Scale | -0.071984 | +0.273984 |
| Total attraction A | -0.353160 | -0.071135 |
| Total positive separation R | -0.281175 | -0.345118 |
| Weighted joint score | +0.005252 | +0.168917 |
| Weighted signed score | +0.005484 | +0.073199 |
| Weighted log-power score | -0.083610 | +0.018834 |
| Weighted correlation score | +0.001461 | +0.014473 |
| Weighted triple score | -0.000572 | -0.001440 |

Negative score changes favor the child; the fair score is `S = A - R`, so a negative change in positive separation contributes to a larger score. Rows use the same endpoints and frozen representation. They are additive block contributions, not independent experiments.

The TRAIN component intervals confirm that aggregate improvement hides real tradeoffs. At epoch 500, the weighted joint change is `+0.005252 [ +0.000041, +0.010225 ]`, signed is `+0.005484 [ +0.001030, +0.009730 ]`, correlation is `+0.001461 [ +0.000022, +0.002850 ]`, and log-power is `-0.083610 [ -0.089900, -0.077758 ]`. These are the existing paired 95% condition-bootstrap intervals, scaled by the same positive constants. At epoch 100, joint, signed, correlation and triples all worsen on TRAIN, while the log-power improvement makes the complete expanded composite improve by `-0.028294`.

**The sharper diagnosis is therefore twofold: the TRAIN composite permits joint-relation deterioration, and held-out conditions amplify the unfavorable tradeoff.** It is inaccurate to imply that every Scale component improved on TRAIN and only subsequently failed to transfer. These results do not establish which parameter directions cause the tradeoff or prove that a different objective will generalize.

The input files are local-only raw evidence under the case-owned campaign `cases/turbulent_combustion/runs/coherence_adaptive_repair/round_6/scale_alignment_followup_20261125/assessment/`, backed by the DATA2 project root. They are not portable publication links:

| Relative raw evidence path | SHA256 |
| --- | --- |
| `TRAIN800_ExhaustiveScale_20261125_v1/decomposition.json` | `cc6813d632e1be4ea5818f04f6195d2b87cd32dba62edb648ef3cb51b41ab0a7` |
| `Validation100_ExhaustiveScale_20261125_v1/decomposition.json` | `5e1d7c408dcc9475b64b70e2c666bc48dc39854e305aa2176e8a010fd4f9a4d1` |

Both hashes were checked during this audit. The [decomposition implementation](../../../../../../scripts/analysis/car_scale_cached_decomposition.py) reconstructs the published score from cached outputs before producing component summaries. Its complete-prefit weights are `2` for signed, `0.25 * 1.2835023021293974` for joint, `0.5` for log-power, `0.8190801090444172` for correlation, and `0.5` for the mean of eight frozen triple queries. “Complete” covers the declared query population on fixed Q4096, not all 40,300 native points. Both cached comparisons use four independent members and fixed T256 observations. Their condition bootstrap describes these reused development panels, not independent training trajectories.

## 2. What the other diagnostics establish

On VALIDATION100, the original epoch-500 complete-expanded change is `+0.273984`; attraction changes by approximately `-0.0711`, with an interval crossing zero, while separation changes by `-0.3451`, with an interval entirely below zero. The joint block contributes `+0.1689`; its attraction changes by `+0.023490` and separation by `-0.145427`. Weighted CO attribution is `+0.3030`, including band 3 `+0.1129` and piece means `+0.0988`. These norm allocations reconstruct the score but are not independent proper field scores or causal field effects.

Physical member spread increases in all five fields in the original epoch-500 assessment, although Scale representation separation decreases. This is not evidence of uniform raw-field variance collapse. Increasing raw spread or enforcing SOURCE separation would introduce another objective assumption; SOURCE spread is not automatically calibrated, and a valid fair-score improvement can reduce separation.

The supplement's 13 independently reset mature-Adam TRAIN shadows lower the sampled objective and all three complete Scale definitions in every batch. Complete-expanded Scale decreases by `0.0081475` raw units in the sample-weighted summary. Four repeated query-seed replays preserve the TRAIN-benefit/VALIDATION-harm reversal. These results weaken an explanation based solely on bad local TRAIN actions, sampled-query noise or inconsistent reporting definitions. They do not establish long-run held-out descent.

Rotating native-complement field feedback has a narrower demonstrated benefit. At matched additional age 100 it improves CH4/T signed risks and CH4/CO/T member MSE relative to unchanged continuation, but provides no distinguishable Scale advantage. Its original SOURCE risk reference R0 was fitted with selected-support J2 training-observation draws; it is not an exact native-support SOURCE risk. Cached native J4/T256 comparisons have their own paired SOURCE bank and must not be substituted into that training reference without an explicit statistical change.

## 3. Why a joint-only Scale objective is a meaningful candidate

The existing [joint-score implementation](../../../../../../src/phycoflow_reconstruction/coherence/families/multiscale_spectrum/losses.py), functions `joint_spectral_energy_score` and `_joint_spectral_embeddings` at audited lines 285–386, contains the required full representation. With coefficients `a[k,f]`, frozen filters `g[s,k]`, positive band scales `s[s,f]` and total mass `M`, it embeds

\[
z_{kf}=a_{kf}\sqrt{\frac{1}{M}\sum_s\frac{g_{sk}^2}{s_{sf}^2}},\qquad z^\mu_{pf}=\mu_{pf}\frac{\sqrt{w_\mu\pi_p}}{s^\mu_{pf}}.
\]

The second vector appends every retained connected-piece mean and field to the same joint vector. The coefficient embedding has exactly the distance of the stacked normalized signed-band vectors because the frozen basis is mass-orthonormal. A coefficient with positive root is recoverable from its coordinate; if the root is zero, all corresponding filters are zero and every current band-derived auxiliary ignores that coefficient too. This is a mathematical information statement, not a claim that numerically inverting tiny roots is well-conditioned or required in implementation.

The current auxiliary descriptors are deterministic functions of this representation and frozen artifacts:

- Signed self/pair projections and rho use the same filtered coefficients and positive power floors in [sampled Scale](../../../../../../src/phycoflow_reconstruction/coherence/scale_compute.py), audited lines 101–164.
- Log-power is `log1p(power/power_floor)` divided by the frozen relation standard deviation and square root of descriptor count in [shared Scale statistics](../../../../../../src/phycoflow_reconstruction/coherence/query_runtime.py), lines 21–32.
- Regional triples reconstruct the same filtered coefficient fields through [spectral_filter](../../../../../../src/phycoflow_reconstruction/coherence/adaptive_fitting.py), lines 333–362, then apply frozen regional means/scales and triple products in `third_order_scores` in the Scale module.

The frozen experiment uses model units, a FEM basis on Q4096 with at most 160 modes, wavelengths `[1.5, 0.45, 0.15]`, minimum wavelength `0.1`, field-unit floor `0.01`, mean weight `1`, float64 and fair alpha `1`. A proposed joint-only candidate should reuse these fitted artifacts, rather than refit geometry or discard means. For deterministic models the corresponding existing joint point distance uses one prediction; it makes no distributional claim.

**Joint-only changes the learning objective and metric emphasis, while retaining the information represented by the current Scale construction.** It does not preserve the original gradient, finite-sample sensitivity or trajectory. Joint scoring does not apply the projected score's normalized band averaging or the log-power descriptor's separate normalization. Auxiliary scores can emphasize difficult features even when they add no represented information. The finite joint representation still excludes unresolved modes and native locations outside its support. Keep all historical complete Scale definitions and semantic components in assessment so improvement cannot be manufactured by changing the reported yardstick.

A bounded action comparison can reuse State, Shape and the weighted field gradient, add the composite Scale and proposed joint rows, then compare two actual AdamW shadow actions from identical states. Full component attraction/separation and field changes are the informative outputs. One extra parameter VJP is sufficient for this comparison; a new per-field backward suite is unnecessary. Short actions qualify execution and reveal local tradeoffs; the subsequent matched learning experiment must answer generalization.

## 4. Endpoints and cost boundaries

The original Round 6 model completed 500 epochs. Its published Scale change is `+0.22354`, Shape is `-0.31061`, and State is `-0.06141` with uncertain State improvement; four native signed-field allowances fail. The supplement's selected phase-225 endpoint is cumulative age 725, whereas the unchanged control stops at phase 100/cumulative age 600. The later candidate comparison is unequal exposure.

At that selected endpoint, published Scale is `+0.121449`, complete-expanded prefit Scale is `+0.199467 [ +0.071843, +0.320631 ]`, Shape is `-0.324422`, and State is `+0.290467` versus SOURCE. Signed risk change divided by allowance is CH4 `0.37572`, CO `1.0826`, T `3.8086`, U1 `8.6289`, and pressure `24.871`; only CH4 passes. Late recovery is preserved as evidence, but neither native retention nor Scale generalization is solved.

| Recorded cost comparison | Result | Scope |
| --- | --- | --- |
| Separate versus batched family VJPs, shared queries | 127.557 versus 122.949 seconds/epoch; 3.612% lower | Matched original Round 6 qualification |
| Shared versus eight-group queries, batched VJP | 123.490 versus 128.600 seconds/epoch; 4.137% higher | Separate matched qualification block |
| Original separate/shared versus selected batched/grouped package | 127.557 versus 128.600 seconds/epoch; 0.818% higher | Cross-block package readout; common control differed by 0.440% |
| Package peak allocation | 19.02 versus 37.25 GB | Substantial memory increase; no overall speedup established |
| Unchanged versus rotating-complement feedback | 167.315 versus 191.284 seconds/epoch; 14.326% higher | Matched supplement qualification |
| Original formal epoch mean | 153.685 seconds | 499 complete epochs; split epoch 439 has incomplete timing |
| Supplement candidate formal epoch mean | 140.7 seconds | Different execution context; not a matched acceleration claim |
| Original Round 6 charge | 25.226897 occupied GPU-hours | Includes conservative bounds for incompletely timed occupation |
| Supplement charge | 13.940997 occupied GPU-hours | Includes diagnosis, both arms, failures, assessment and stopped tail |

The supplement stopped at phase 227.25 and assessed the last complete recovery checkpoint at phase 225; the remaining 2.25 epochs are preserved and charged. Its final prediction worker used 32.29 GPU-seconds and 146.41 total wall seconds; CPU scoring is recorded separately. These costs support remeasuring backward scheduling and checkpoint boundaries on GPU2. They do not identify batched VJP as a permanent net benefit or demonstrate that the LPC–SSS framework itself is invalid.
