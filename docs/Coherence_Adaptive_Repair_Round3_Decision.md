# Coherence Adaptive Repair Round 3 decision

Completed 2026-11-21. **Reject both learned outputs and retain SOURCE/C8.** Both original FFM lineages reached the maximum authorized epoch 500 after the task-free epoch-100 learning review. They shared exact State/Scale reuse, a 50-epoch warmup, uniform allocation and the fixed physical-field allowances. RawRisk used unit risk scales and dual step 0.01; SourceScaledRisk used positive TRAIN SOURCE member-MSE scales and dual step 0.0001. This compares two feedback rules and does not isolate denominator scaling.

The raw-output assessment used the frozen matched VALIDATION100 panel and four independent members. Changes below use the frozen TRAIN SOURCE normalizers; negative core changes improve scores, while positive regional improvement is desirable.

| Learning path | State change | Scale change | Shape change | Event change | Regional improvement | Signed-field failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FFM_RawRisk_Uniform_Repair | +0.15316 | +0.14413 | -0.33226 | -0.48860 | -0.63442 | 5 of 5 |
| FFM_SourceScaledRisk_Uniform_Repair | -0.15662 | +0.16925 | -0.30526 | -0.25405 | -0.47563 | 4 of 5 |

SOURCE-scaled feedback improves State and pressure retention, but the other four signed-field conditions, the Scale guard and regional utility still fail. RawRisk also fails the State guard. Both fail the separate member-risk guard and pass observed-sensor checks. Their Shape and event improvements do not qualify replacement. Close this comparison at epoch 500; no extension or new learning arm follows from this result.

**Retain exact State/Scale reuse:** real-input values, gradients, parameter-space direction, optimizer action and fresh-process resume qualified; measured peak allocation fell by 0.468 GiB (2.57%). No training-speedup claim is established. **Retain the Shape diagnostic finding:** eight paired fixed-SOURCE query replays reduced sampled score SD from 0.200689 to 0.031079 with condition-specific queries. This supports a later estimator question, not parameter-gradient variance or learning benefit; neither training query law changed.

The conservative campaign cost is **40.124 GPU-hours**, within 45. The two CPU scoring passes occupied 285 seconds enclosing wall and 556.83 worker CPU-seconds; the full assessment-stage spans including generation and coordination total 1581.02 seconds, about 2.25% of one training path's wall time. The independent Shape diagnostic's all-attempt worker upper bound is 3457.29 seconds within 7200. Unknown failed-subattempt costs remain covered by the enclosing setup bound; pure data-wait time was not isolated.

The [final report](../Coherence_Adaptive_Repair_Round3_Final_Review/report.md) contains native fieldwise errors, member/spread separation, paired task effects, complete loss curves, PDF figure masters and evidence hashes. The [campaign index](../cases/turbulent_combustion/runs/coherence_adaptive_repair/round_3/README.md) links complete histories and attempts. VALIDATION and both reused tasks are development evidence, not independent confirmation. TEST stayed locked; deployment profiling was excluded. Closed Round 2 artifacts, the original user revisions and existing branch history were preserved; current main at study initialization was already an ancestor.
