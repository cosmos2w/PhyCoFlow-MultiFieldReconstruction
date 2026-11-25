# Scale alignment diagnosis and selected repair

Date: 25 November 2026. This records the completed diagnostic decision under [the authorized followup](SCALE_ALIGNMENT_FOLLOWUP.md). The separate followup allowance is at most 30 summed occupied GPU-hours, including diagnosis, unchanged control, candidate and assessment. Its 500-epoch maximum includes the 100-epoch review. The original Round 6 endpoint and closed rounds remain preserved; TEST is locked.

## Measurements and interpretation

All changes below are relative to the unchanged original SOURCE and divided by the same frozen TRAIN Scale unit, 0.25978530736085925. TRAIN uses the matched random 800 of 8000 conditions; VALIDATION uses the existing first 100 of 1000 conditions. Both use the same complete query definitions, frozen Q4096 geometry, native cached J4 members, and matched observation/member manifests within each split. The validation panel is reused development data.

| Split and checkpoint | Published Core Default | Complete Core Prefit | Complete Expanded Prefit |
| --- | ---: | ---: | ---: |
| TRAIN, original epoch 100 | -0.111686 | -0.029695 | -0.028294 |
| TRAIN, original epoch 500 | -0.158602 | -0.071412 | -0.071984 |
| VALIDATION, original epoch 100 | +0.060983 | +0.118555 | +0.117651 |
| VALIDATION, original epoch 500 | +0.223544 | +0.275424 | +0.273984 |

The paired condition-bootstrap intervals for each epoch-500 TRAIN change are negative; each corresponding VALIDATION interval is positive. Four paired grouped-query replays preserve the improvement direction on TRAIN and deterioration direction on VALIDATION. Consequently, the definition and query differences do not explain away the held-out failure. This is measured failure to transfer under the examined definitions, not evidence that the two split populations are otherwise equivalent.

On VALIDATION, the expanded epoch-500 Scale change separates into joint +0.168917, signed +0.073199, log-power +0.018834, correlation +0.014473 and third-order -0.001440. Attraction decreases by 0.07113 units at the point estimate, while positive member separation decreases by 0.34512 units; the latter dominates the fair score change. The attraction interval spans zero. CO is the largest positive exact field attribution, especially band 3 and piece means, partly offset by pressure. Joint-norm attributions reconstruct the complete norm and are not independently proper field scores. These observations do not by themselves establish a cause of generative collapse.

The actual saved epoch-500 model, AdamW moments, current PI multipliers and accepted cursor were independently restored for every one of 13 TRAIN shadow batches covering the same 800 conditions. All 13 reduce the same-query and independently queried training Scale, all three complete Scale definitions, and the complete-core first-order product with actual parameter displacement. Sample-weighted raw finite changes are -0.008687 for same-query training Scale, -0.008190 for independent-query training Scale, -0.007988 for Published Core Default, -0.007943 for Complete Core Prefit and -0.008147 for Complete Expanded Prefit. These are descriptive results of independently reset J2 training-law shadows on Q4096, not a sequential continuation or native J4 inference. CH4, CO, T and U_1 signed field risks decrease in all 13 shadows; pressure increases in 12 of 13. No intrinsic local opposition between the two Scale definitions was found in this audit.

Cached native field risks expose a separate feedback coverage problem. At original epoch 500, TRAIN signed-risk changes in units of each original allowance are [1.5269, 1.3925, 0.9610, 1.1642, 1.3021] for [CH4, CO, T, U_1, p]. The fixed Q4096 region has changes [0.8839, 0.9500, 0.4179, 0.9425, 1.2843]; its complement has [1.6004, 1.4431, 1.0230, 1.1895, 1.3041]. Direct paired complement-minus-selected intervals are positive for CH4, CO, T and U_1. VALIDATION deterioration is large on both regions, and all five analogous localization intervals span zero. Rotating field feedback addresses measured TRAIN spatial undercoverage; it is not a demonstrated solution to the entire held-out gap.

The descriptive condition audit finds all examined VALIDATION features within TRAIN marginal ranges, with shifts including about +0.82 TRAIN standard deviations in CO mean/RMS. The last 20% of TRAIN does not have a closer feature centroid than the complete TRAIN panel. These retrospective summaries neither identify a causal shift nor justify arbitrary recency weighting. They are not used to fit validation-dependent weights.

## One selected learning experiment

Keep the existing SSS Q4096 support, positive inner anchors and prefit coefficients, eight condition-query groups, three fresh parameter-space ConFIG gradients, SOURCE coherence scale C0, and raw output policy. Change only the spatial estimator used by the signed Lagrangian field feedback. Let S be the fixed 4096-point SSS support, N the 40300 native points, M their total physical mass, and C_t a fresh uniform sample without replacement of m = 512 points from the exact complement of S. For each field, estimate the native signed risk with

$$\widehat R_t=\frac1M\left[\sum_{i\in S}w_i\ell_i+\frac{N-|S|}{m}\sum_{i\in C_t}w_i\ell_i\right],\qquad \ell_i=(\overline Y_i-y_i)^2-\frac{s_i^2}{J},\quad J=2.$$

The fixed stratum has inclusion probability one and the complementary stratum has probability m/(N-|S|). The denominator remains native total mass; normalizing by the random sampled mass or treating the union as a uniform sample would define a different estimator. Keep the original SOURCE risk references, allowances, risk scales, PI gains and smoothing state. Retain all model/AdamW/controller state at the common parent. No native training loss or additional MSE objective is introduced.

Complement selection uses a separate deterministic stream indexed by the absolute accepted cursor. Preserve the exact original SSS rollout when practical by evaluating the additional field points separately with the same per-member RFF coefficients and trusted observations. A union implementation requires a focused output/gradient/action parity check with the old field formula before attributing its intentional new field action to the estimator. Either implementation must retain global physical indices, masks, masses and the correct finite-member correction. Assess actual complete-epoch time and peak memory; query-count ratios are not cost measurements.

Fork `FFM_SSS_Unchanged_Scale_Continuation` and `FFM_SSS_RotatingComplement_FieldFeedback` from the identical original epoch-500 model, actual AdamW/PI state and data/RNG cursor. Each new run starts phase epoch zero at absolute action 6000 and has its own empty logs and typed recording definitions. Run the two arms to phase epoch 100, with first and ten-epoch loss curves and an affected real-model fresh-process save/config-reload/resume/append qualification. Qualification establishes valid execution, not efficacy. Ordinary strict resume and the original run remain unchanged.

## Review, conditional endpoint and evidence

At the matched phase-100 review, cache one raw VALIDATION100/J4 prediction bank per arm and reuse it for original native fields, complete State/Scale/Shape, and the three aligned Scale definitions. Compare each arm with the common start, SOURCE and the same-age control; report fieldwise retention and Shape separately. No extra inference is needed for the aligned Scale decomposition. Preserve real component improvement even when a model does not qualify for promotion.

Continue the repaired arm toward phase epoch 500 if learning is valid and the joint trajectories leave a useful recovery/retention question within the remaining allowance. After adequate learning, persistently uncompetitive Scale and fields without a recovery trend can close this route; a fixture missing an anticipated margin cannot. Before launch, replace the provisional timing with measured complete epochs and forecast the whole remaining training, generation, CPU scoring and queue tail. The hard budget remains binding. No second 500-epoch control is presumed, so any final comparison with its phase-100 endpoint discloses unequal exposure.

The DATA2-owned Round 6 continuation is `cases/turbulent_combustion/runs/coherence_adaptive_repair/round_6/scale_alignment_followup_20261125/`. Execution originally used the sibling path outside `round_6`; frozen raw records preserve that historical prefix, with the verified relocation map retained under Round 6 `operations/relocations/ScaleAlignment_Followup_20261125_v1/`. Its compact diagnostic evidence is:

| Evidence | Campaign-relative artifact | SHA256 |
| --- | --- | --- |
| VALIDATION complete decomposition | `assessment/Validation100_ExhaustiveScale_20261125_v1/decomposition.json` | `5e1d7c408dcc9475b64b70e2c666bc48dc39854e305aa2176e8a010fd4f9a4d1` |
| TRAIN complete decomposition | `assessment/TRAIN800_ExhaustiveScale_20261125_v1/decomposition.json` | `cc6813d632e1be4ea5818f04f6195d2b87cd32dba62edb648ef3cb51b41ab0a7` |
| VALIDATION query replay | `assessment/Validation100_GroupedScaleEstimator_20261125_v1/query_estimator.json` | `df5c3e961821113ab91c03f510b307d9627896a6b5ab19403cc244c2b61c00d5` |
| TRAIN query replay | `assessment/TRAIN800_GroupedScaleEstimator_20261125_v1/query_estimator.json` | `0e926a59a12c8b62e23c2e2a4280bb41ec8477047feda59933c4998607defebf` |
| Actual Adam shadows | `assessment/Epoch500_ActualAdam_TRAIN800_20261125_v3/receipt.json` | `4f04daf02907b9705dc5b25ca3262f8696e36a1147056d014db0b5d1679f0c39` |
| Native field/support decomposition | `assessment/CachedFieldSupport_TRAIN800_VALID100_20261125_v1/field_support.json` | `e89d56b1d76f14112f38b1f055fd9759a41ff409641534a5dac73194fb165d82` |
| Descriptive condition support | `assessment/ConditionSupport_TRAIN800_VALID100_20261125_v1/condition_support.json` | `95ec9f87bc85bf3d7e4553829f225e728876806b2de6ee71c878bbe7d582d6b0` |

Apply the [mandatory monitoring policy](WORKPLAN.md#10-device-use-and-mandatory-monitoring-policy): `round6_luna_monitor`, GPT-6 Luna with low reasoning, checks at predeclared useful times and pushes terminal/actionable results. The unchanged fork qualification launched at 06:09:22 UTC with a 1200-second cap and one planned observation at 06:21:22 UTC. Future qualification and learning schedules use their measured timing. GPU0 remains the main training GPU; GPU1 is used only for independent work after verifying it is available.
