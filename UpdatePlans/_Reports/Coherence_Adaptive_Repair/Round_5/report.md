# Coherence Adaptive Repair: Round 5

Publication snapshot from the accepted campaign report. Figure previews and PDF masters are copied unchanged; the [evidence index](evidence/INDEX.md) identifies the original report and local research evidence. [Publication PDF](report.pdf).

**Decision:** retain SOURCE/C8. Reject both continuation recipes after their substantive 100-epoch review. Both violate four primary field allowances and the Scale guard; native continuation also worsens the reused event score by 0.766 frozen units. Direct native-plus-SSS supervision improves State and Shape by 0.108 and 0.178 units and the regional task by 0.127 units, but these gains do not compensate for failed field retention and Scale. Its regional gain is smaller than ordinary continuation's 0.271 units. Retain SOURCE/C8 and close Round 5 without invoking the conditional 500-epoch maximum or adding a trial.

Both routes stopped at the 100-epoch review instead of invoking the conditional 500-epoch maximum because of material multi-field and Scale failures. In the safeguarded candidate's final ten epochs, mean pre-proposal CH4 and CO excesses were 1.021883 and 1.113320 times their allowances; 96/120 proposals were pre-infeasible and 34/120 were post-proposal infeasible. Recent sampled field risks and Scale showed no recovery trend, leaving no evidence-based unchanged-recipe recovery question. This rejects the routes within the executed budget; it does not establish that later recovery is impossible.

TEST remains locked. SOURCE/C8 and the historical Round 3 and Round 4 artifacts remain unchanged; those rounds remain closed. The SSS package compares a complete optimization package against native continuation, rather than isolating the contribution of each component.

## Engineering disposition

Optional CQ context reuse is rejected for this round. CUDA endpoint values were identical and all 145 trainable-gradient tensor comparisons passed, but two attention-bias tensors failed the predeclared post-AdamW parameter tolerance (maximum absolute difference 5.334e-7; absolute tolerance 2e-7, relative tolerance 2e-5). Retain context_cache: none; no production speedup is inferred from the synthetic timing.

Eight condition-query groups are rejected; retain shared queries. At fixed SOURCE on TRAIN800, four paired query-seed full-parameter replays gave grouped/shared centered variance ratios 0.39343 for Shape and 0.41979 for the direct scalar, with mean agreement within the declared paired replay convention. This measures finite query-law variation with observations, native draws and J2 endpoints fixed, not total training variance. The separate eight-seed fixed-256-T endpoint replay reduced scalar score SD from 0.04534 to 0.01991 but had endpoint-gradient variance ratio 0.9923; its conditioning regime differs from the fresh-observation parameter replay. Both query laws accepted 26/26 finite proposals. Grouping improved independent-query median scalar decrease from 0.04903096 to 0.05114261, but lowered fixed-query median decrease from 0.05390337 to 0.05349235, failing the required no-worse gate. Paired epoch-cost measurements were omitted after this necessary action gate failed.

Initial direct-supervision startup is qualified without a correction. All 24 candidate attempts accepted full displacement, there were zero ascending proposals, and native input identities matched the ordinary control. An actual saved-configuration fresh-process GPU resume and the independent CPU forced-rejection/resume check passed. The full TRAIN800 shared-query replay had 23/26 fixed-query descents beyond numerical precision, with positive fixed-query and independent-query median decreases. Neither startup nor replay establishes learning efficacy.

The acceleration claim is rejected against the predeclared 25% target: complete candidate epoch time was 88.589 seconds versus 113.852 seconds for the matched historical route on the same GPU, a 22.19% reduction. Ordinary native continuation took 4.404 seconds per complete epoch. Candidate timing includes SOURCE/post-proposal finite field checks, PH work, backward/action work and callbacks. The two-arm accuracy comparison subsequently closed at its 100-epoch learning review; the conditional 500-epoch maximum was not invoked. No teacher, extra arm or automatic extension was introduced.

Matched VALIDATION100 comprises 100 of 1000 validation conditions (10.0%); each prediction uses J4 members. This previously accessed same-trajectory validation panel is used for development selection and is not independent confirmation.

## Method and training exposure

Both arms start from the EMA-materialized SOURCE with child EMA disabled. Both use the original native bridge-velocity objective divided by the same fixed native unit. The safeguarded candidate adds (State / u_state + Scale / u_scale + Shape / u_shape) / 3, accumulated in the order native, State, Scale, Shape, and applies a paired local finite field-action filter.

The fixed SOURCE TRAIN800 units are native=0.11407, state=0.055472, scale=0.26363, shape=1.3179. The native unit is the mean native loss; the three SSS units are means of absolute signed per-condition scores using fresh observations. Each reporting epoch contains 12 attempted batches of 64 conditions, or 768 attempted visits. Candidate training uses J2 endpoints on selected Q4096 support; assessment uses the native VALIDATION100 panel and J4 members.

## Assessed learning ages and exposure

| Path | Assessed epoch | Attempted condition visits | Accepted objective-bearing visits |
| --- | ---: | ---: | ---: |
| Native continuation | 100 | 76800 | 76800 |
| Native + SSS safeguard | 100 | 76800 | 76800 |

The terminal outputs use the same condition and member panel and share the same assessed epoch; recorded attempted and accepted exposure is shown separately.

## Sampled finite actions

The safeguarded path attempted 1200 actions: 1200 accepted, including 0 half-displacement actions, and 0 rejected. It recorded 1200 SOURCE and 1200 post-proposal field ensemble checks, including probes of rejected proposals.

These sampled checks are part of the complete attempted-epoch training cost reported below; they are not standalone profiling or a held-out feasibility guarantee. Half actions are recovered from recorded mean displacement fractions and acceptance counts.

## Field retention and scientific relations

Field values are native-unit squared mass-weighted statistics; signed-risk changes are relative to the matched VALIDATION SOURCE and use the frozen TRAIN allowance converted to native units.

### Field table 1

| Field | Path | Member MSE | Ensemble mean MSE | Member variance | Signed risk change | Allowance | Sensor MSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CH4 | SOURCE | 8.5883e-05 | 8.2485e-05 | 4.53e-06 | unavailable | unavailable | unavailable |
| CH4 | Native continuation | 8.9386e-05 | 8.6266e-05 | 4.1604e-06 | 3.8735e-06 | 7.2357e-07 | unavailable |
| CH4 | Native + SSS safeguard | 8.9365e-05 | 8.692e-05 | 3.2607e-06 | 4.7521e-06 | 7.2357e-07 | unavailable |
| CO | SOURCE | 4.2179e-05 | 3.9719e-05 | 3.2793e-06 | unavailable | unavailable | unavailable |
| CO | Native continuation | 4.3449e-05 | 4.1167e-05 | 3.0428e-06 | 1.5068e-06 | 6.0674e-07 | unavailable |
| CO | Native + SSS safeguard | 4.5598e-05 | 4.38e-05 | 2.3985e-06 | 4.3007e-06 | 6.0674e-07 | unavailable |
| T | SOURCE | 74409 | 71911 | 3330.6 | unavailable | unavailable | 0 |
| T | Native continuation | 77051 | 74766 | 3046.9 | 2926 | 646.4 | 0 |
| T | Native + SSS safeguard | 78656 | 76744 | 2549.1 | 5028.7 | 646.4 | 0 |

### Field table 2

| Field | Path | Member MSE | Ensemble mean MSE | Member variance | Signed risk change | Allowance | Sensor MSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| U_1 | SOURCE | 1183.3 | 1158.6 | 32.83 | unavailable | unavailable | unavailable |
| U_1 | Native continuation | 1193.2 | 1170.5 | 30.347 | 12.442 | 6.1 | unavailable |
| U_1 | Native + SSS safeguard | 1210.7 | 1190.4 | 26.948 | 33.274 | 6.1 | unavailable |
| p | SOURCE | 3.1801e+07 | 3.1637e+07 | 2.1889e+05 | unavailable | unavailable | unavailable |
| p | Native continuation | 3.1848e+07 | 3.1692e+07 | 2.0815e+05 | 58131 | 74097 | unavailable |
| p | Native + SSS safeguard | 3.1894e+07 | 3.1654e+07 | 3.2075e+05 | -8733.1 | 74097 | unavailable |

| Path | State change | Scale change | Shape change | Field / sensor gate | Member gate | Regional improvement | Event change |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| Native continuation | 0.007938 | 0.17611 | 0.0024754 | False | True | 0.2712 | 0.76588 |
| Native + SSS safeguard | -0.10791 | 0.11847 | -0.1776 | False | True | 0.12656 | 0.012807 |

SSS and task changes use frozen TRAIN SOURCE units; negative SSS and event changes improve scores, while positive regional improvement is better. The event and regional tasks are reused development tasks, not independent confirmation.

The paired 95% percentile intervals resample conditions on this finite development panel; they do not cover temporal dependence or training-seed uncertainty.

| Comparison | Score | Mean change | Paired 95% interval |
| --- | --- | ---: | ---: |
| Native minus SOURCE | State | 0.007938 | [-0.25745, 0.26529] |
| Native minus SOURCE | Scale | 0.17611 | [0.10311, 0.24583] |
| Native minus SOURCE | Shape | 0.0024754 | [-0.015109, 0.020381] |
| Native minus SOURCE | Event score change | 0.76588 | [0.61986, 0.91343] |
| Native minus SOURCE | Regional score change | -0.2712 | [-0.63523, 0.071656] |
| Safeguarded minus SOURCE | State | -0.10791 | [-0.3711, 0.13504] |
| Safeguarded minus SOURCE | Scale | 0.11847 | [0.039964, 0.19626] |
| Safeguarded minus SOURCE | Shape | -0.1776 | [-0.20252, -0.15228] |
| Safeguarded minus SOURCE | Event score change | 0.012807 | [-0.077849, 0.10291] |
| Safeguarded minus SOURCE | Regional score change | -0.12656 | [-0.50737, 0.24907] |
| Safeguarded minus native | State | -0.11585 | [-0.28485, 0.046506] |
| Safeguarded minus native | Scale | -0.057636 | [-0.12528, 0.012217] |
| Safeguarded minus native | Shape | -0.18008 | [-0.20396, -0.1566] |
| Safeguarded minus native | Event score change | -0.75307 | [-0.90113, -0.60292] |
| Safeguarded minus native | Regional score change | 0.14464 | [-0.19016, 0.48501] |

Fields and reconstruction errors are decoded to native dataset values. The saved case configuration labels all five physical units as unknown, and the HDF5 attributes supply no unit labels; figures therefore state native dataset units without assigning SI units.

## Training and assessment cost

| Path | Seconds / completed epoch | Occupied GPU s | Update work s | Data wait and loop overhead s | Callback / reporting s |
| --- | --- | --- | --- | --- | --- |
| Native continuation | 3.0673 | 317.59 | 182.57 | 113.18 | 10.978 |
| Native + SSS safeguard | 82.688 | 8277.6 | 8145.4 | 95.201 | 28.21 |

Campaign occupied GPU seconds including failed attempts: 10251. Assessment GPU seconds: 64.233; assessment child CPU seconds: 267.88; full assessment elapsed seconds: 1735.2; assessment tail after last GPU generation seconds: 1669.8.

CPU work inside a GPU lease is included in occupied GPU time and is not added again to wall time. Missing measured values are marked unavailable.

Native continuation recorded child CPU time inside its GPU leases: 472.5 seconds; this CPU time overlaps the occupied lease and is not additional wall time.

Native + SSS safeguard recorded child CPU time inside its GPU leases: 11129 seconds; this CPU time overlaps the occupied lease and is not additional wall time.

The full assessment path elapsed 1735.23 seconds (20.2% of the completed training wall); the summed GPU-lease and CPU-job wall times were 137.83 seconds (1.6%). The 1669.81-second tail includes 1596.00 seconds between the review and terminal CPU jobs for the scheduled monitoring and stop decision; it is not CPU computation. The 20% assessment-wall target is a planning target, not an adoption gate.

Preparation occupied GPU seconds: 1591.3; failed-attempt GPU seconds across the campaign: 22.787; separately recorded preparation CPU seconds: 956.73.

Reporting CPU is separate from scientific assessment costs. Earlier report render attempts:
- [First render](evidence/INDEX.md#evidence-01): 61.824 CPU seconds; 56.545 wall seconds.
- [Second render](evidence/INDEX.md#evidence-02): 62.136 CPU seconds; 56.723 wall seconds.
- [Third render](evidence/INDEX.md#evidence-03): 63.125 CPU seconds; 57.608 wall seconds.
- [Fourth render](evidence/INDEX.md#evidence-04): 62.668 CPU seconds; 57.492 wall seconds.

The current report render has a separate companion CPU receipt.

## Evidence links

- [Final decision](evidence/INDEX.md#evidence-05)
- [Task-free learning review and stop decision](evidence/INDEX.md#evidence-06)
- [Terminal assessment](evidence/INDEX.md#evidence-07)
- [Measured costs](evidence/INDEX.md#evidence-08)

## Figure index

- [Figure query engineering](#figure-query-engineering)
- [Figure learning](#figure-learning)
- [Figure spatial case 1](#figure-spatial-case-1)
- [Figure spatial case 2](#figure-spatial-case-2)
- [Figure spatial case 3](#figure-spatial-case-3)
- [Figure regional relationship](#figure-regional-relationship)

## Figure query engineering

![Finite query replay at fixed SOURCE on TRAIN800: the parameter-gradient comparison uses four paired query seeds with matched observations, native draws and J2 endpoints; the finite-action comparison uses 26 actions from 13 batches at two learning rates. Grouped queries reduce parameter-gradient variance and improve independent-query behavior, but their lower fixed-query median decrease fails the predeclared no-worse gate. Shared queries are retained; full attempted-epoch cost was not measured after that failure. All 26 actions were accepted in each query law; no confidence intervals are inferred.](figures/query_engineering.png)

Finite query replay at fixed SOURCE on TRAIN800: the parameter-gradient comparison uses four paired query seeds with matched observations, native draws and J2 endpoints; the finite-action comparison uses 26 actions from 13 batches at two learning rates. Grouped queries reduce parameter-gradient variance and improve independent-query behavior, but their lower fixed-query median decrease fails the predeclared no-worse gate. Shared queries are retained; full attempted-epoch cost was not measured after that failure. All 26 actions were accepted in each query law; no confidence intervals are inferred. [PDF master](figures/query_engineering.pdf)

## Figure learning

![Raw sampled epoch histories and ten-epoch means for native and SSS training quantities, learning rate, actual action dot product and proposal acceptance. Historical ConFIG appears only for the same frozen State, Scale, Shape and learning-rate definitions on the matching epoch window; query grouping, when adopted, changes cross-condition coupling. These are training diagnostics, not held-out scientific metrics. Assessed epochs: Native continuation 100, Native + SSS safeguard 100.](figures/learning.png)

Raw sampled epoch histories and ten-epoch means for native and SSS training quantities, learning rate, actual action dot product and proposal acceptance. Historical ConFIG appears only for the same frozen State, Scale, Shape and learning-rate definitions on the matching epoch window; query grouping, when adopted, changes cross-condition coupling. These are training diagnostics, not held-out scientific metrics. Assessed epochs: Native continuation 100, Native + SSS safeguard 100. [PDF master](figures/learning.pdf)

## Figure spatial case 1

![Condition trajectory_000000:8000: reference and ensemble-mean reconstructions at common fieldwise color scales; annotations are mass-weighted RMSE on the displayed positive-mass native support in each field’s dataset units. Assessed epochs: Native continuation 100, Native + SSS safeguard 100.](figures/spatial_case_1.png)

Condition trajectory_000000:8000: reference and ensemble-mean reconstructions at common fieldwise color scales; annotations are mass-weighted RMSE on the displayed positive-mass native support in each field’s dataset units. Assessed epochs: Native continuation 100, Native + SSS safeguard 100. [PDF master](figures/spatial_case_1.pdf)

## Figure spatial case 2

![Condition trajectory_000000:8049: reference and ensemble-mean reconstructions at common fieldwise color scales; annotations are mass-weighted RMSE on the displayed positive-mass native support in each field’s dataset units. Assessed epochs: Native continuation 100, Native + SSS safeguard 100.](figures/spatial_case_2.png)

Condition trajectory_000000:8049: reference and ensemble-mean reconstructions at common fieldwise color scales; annotations are mass-weighted RMSE on the displayed positive-mass native support in each field’s dataset units. Assessed epochs: Native continuation 100, Native + SSS safeguard 100. [PDF master](figures/spatial_case_2.pdf)

## Figure spatial case 3

![Condition trajectory_000000:8099: reference and ensemble-mean reconstructions at common fieldwise color scales; annotations are mass-weighted RMSE on the displayed positive-mass native support in each field’s dataset units. Assessed epochs: Native continuation 100, Native + SSS safeguard 100.](figures/spatial_case_3.png)

Condition trajectory_000000:8099: reference and ensemble-mean reconstructions at common fieldwise color scales; annotations are mass-weighted RMSE on the displayed positive-mass native support in each field’s dataset units. Assessed epochs: Native continuation 100, Native + SSS safeguard 100. [PDF master](figures/spatial_case_3.pdf)

## Figure regional relationship

![Matched condition-level reference versus mean-member regional CO–U1 covariance and corresponding signed condition differences, computed on the frozen right-half mass-weighted task from saved predictions. Assessed epochs: Native continuation 100, Native + SSS safeguard 100.](figures/regional_relationship.png)

Matched condition-level reference versus mean-member regional CO–U1 covariance and corresponding signed condition differences, computed on the frozen right-half mass-weighted task from saved predictions. Assessed epochs: Native continuation 100, Native + SSS safeguard 100. [PDF master](figures/regional_relationship.pdf)
