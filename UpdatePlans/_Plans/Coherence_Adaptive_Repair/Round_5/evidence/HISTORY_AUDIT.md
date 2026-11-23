> **Historical planning snapshot:** Round 5 has concluded. This document is preserved for reference; see the final report for its outcome.

# Saved-history audit for Round 5 planning

This bounded audit reads the complete saved classic coverage-11f history, Round 2 FFM full-model history, and both Round 3 histories. It uses CPU arithmetic only: no model loading, GPU operation, inference, new scientific scoring, optimizer action, or changes to historical runs. The original PNG companions were visually inspected. The complete histories remain in their case-owned locations; this directory contains only the audit, its compact machine summary and its reproduction script.

## Outcome

The newer runs show a real early increase in sampled Scale and field-risk proxies, but their learning-rate ramps executed and no changing calibration explains the increase. A longer ramp delayed the rise without eliminating it. Shape improved modestly while retaining substantial sampling variability. The near-identical jagged Shape sequences across differently trained runs follow exactly matched stochastic inputs; they do not by themselves demonstrate unstable parameter trajectories, absent gradients, or exhausted learning capacity. Existing native assessments still reject the recent learned outputs.

## Learning schedule, calibration and early change

All runs start at learning rate `5e-6` and reach `5e-5`. Classic coverage-11f reaches the target at the end of 2 reporting epochs; Round 2 reaches it after 6.333 reporting epochs. Round 3 uses 50 elapsed reporting epochs: its last warmup-log rate is `4.9925e-5`, and epoch 51 onward uses the target. The historical filename containing `20ep` is not evidence of a twenty-epoch ramp. The machine summary distinguishes the declared ramp duration from the first complete epoch entirely at the target rate.

Each complete history has one fixed family-calibration definition. Round 2 and both Round 3 histories also contain one fixed selected coefficient vector, with scalable calibration frozen from the first recorded action. The plot's vertical “freeze at epoch 1” marker does not establish a later calibration transition. These checks concern recorded scale/weight definitions, not a claim that every internal model statistic is constant.

| Recorded sampled TRAIN quantity | Initial/early value | Intermediate value | Late value |
|---|---:|---:|---:|
| Round 2 weighted Scale | Epoch 1: 0.259078 | Epochs 26–50: 0.285825 | Epochs 729–750: 0.254645 |
| Round 2 Scale joint | Epoch 1: 0.274816 | Epochs 26–50: 0.313819 | Epochs 729–750: 0.284030 |
| Round 3 SourceScaledRisk weighted State | Epochs 11–25: 0.047514 | Epochs 51–100: 0.049877 | Epochs 451–500: 0.044754 |
| Round 3 SourceScaledRisk Scale joint | Epoch 1: 0.273837 | Epochs 51–100: 0.306265 | Epochs 451–500: 0.281573 |

The Round 3 State trace first descends, then rises, then descends again. The Scale joint rise continues beyond the warmup interval in both recent rounds. This rejects “missing ramp” and “a longer ramp is sufficient” as explanations of the observed failure; it does not isolate the causal contribution of learning rate from the changed feedback and stochastic optimization.

All quantities above are saved sampled training diagnostics. In the recent external-policy route, `data_loss` is the sampled signed mean-risk proxy, not the original native training objective or native member MSE. Complete histories contain 5,000, 778, 500 and 500 reporting epochs for classic, Round 2, Round 3 SourceScaledRisk and Round 3 RawRisk respectively, with contiguous complete epoch records and accepted-update fractions of one. Round 2's assessed scheduled checkpoint is epoch 750, although its complete recorded history reaches 778.

## Shape learning and the common stochastic sequence

| Run/window | Weighted Shape mean ± across-epoch SD | Self mean ± SD | Mutual mean ± SD |
|---|---:|---:|---:|
| Round 2, epochs 11–25 | 1.13476 ± 0.110 | 0.61715 ± 0.127 | 0.49181 ± 0.0491 |
| Round 2, epochs 729–750 | 1.00436 ± 0.110 | 0.52869 ± 0.129 | 0.44661 ± 0.0278 |
| Round 3 SourceScaledRisk, epochs 11–25 | 1.14062 ± 0.112 | 0.62190 ± 0.129 | 0.49334 ± 0.0491 |
| Round 3 SourceScaledRisk, epochs 451–500 | 1.03517 ± 0.117 | 0.56713 ± 0.121 | 0.44598 ± 0.0367 |

Shape is not wholly unchanged. Its improvement is modest relative to the remaining variability, particularly in the self component. These SDs span changing model parameters, conditions, members and queries; they are neither fixed-model variance decompositions nor independent-replicate uncertainty estimates. The weighted family is not the unweighted sum of the displayed self/mutual values.

R2 and each R3 arm have zero differences in their first 500 epochs of saved condition, observation, spatial-query and member-coupling receipts: 6,000 recorded actions per comparison. The scalable query exposures, coefficient vectors and frozen flags also match for that entire prefix. Their trajectories nevertheless use different ramps and, for SourceScaledRisk, different constraint feedback.

| Comparison against Round 2 over epochs 1–500 | Shape first-difference correlation | Self first-difference correlation | Mutual first-difference correlation |
|---|---:|---:|---:|
| Round 3 RawRisk | 0.998913 | 0.999267 | 0.998336 |
| Round 3 SourceScaledRisk | 0.999011 | 0.999335 | 0.998584 |

All three recent runs have their first-100-epoch Shape maximum at epoch 43. The epoch-42-to-43 increase is 0.4843 for Round 2, 0.4945 for Round 3 RawRisk and 0.4923 for SourceScaledRisk. This near-identical sequence strongly associates the jagged display with common stochastic inputs. It does not isolate query variation from condition/member variation or prove that parameter-gradient variance has the same structure. The already completed fixed-SOURCE TRAIN800/J2 eight-seed query replay supplies the narrower scalar-query evidence: shared-query SD 0.200689 versus condition-specific SD 0.031079. No repetition of that scalar diagnostic is needed merely to recover the same fact.

## Constraint feedback and clipping

| Run/window | Mean constraint-gradient norm | Mean coherence-direction norm | Ratio of those mean norms | Mean clipping coefficient |
|---|---:|---:|---:|---:|
| Classic coverage-11f, epochs 11–25 | 5.77094 | 22.6407 | 25.49% | 0.04289 |
| Round 2, epochs 11–25 | 0.016802 | 7.18478 | 0.234% | 0.14732 |
| Round 3 SourceScaledRisk, epochs 11–25 | 0.001674 | 6.50821 | 0.0257% | 0.16620 |
| Round 3 RawRisk, epochs 451–500 | 0.64657 | 6.81738 | 9.48% | 0.15150 |
| Round 3 SourceScaledRisk, epochs 451–500 | 1.74696 | 6.83740 | 25.55% | 0.13785 |

The recent feedback is weak early and grows substantially. Source scaling changes that growth and some final field/State outcomes, but the final native assessment still fails. Norm ratios do not determine directional alignment. Clipping coefficients are averages of pre-optimizer diagnostics; they are not actual AdamW displacements and do not certify descent for every family. The requested saved histories do not provide an actual-step/shadow trajectory that would resolve that distinction.

## Why classic coverage-11f is not a matched control

The SOURCE checkpoint SHA-256 is the same in all records, `2d1da713538f6e9613560f3a7eb4a9df2a95d1ad1f56dd11a1a8a4f9ef65050c`, but the endpoint and scientific contracts differ. Classic declares `model_ema_eval: false` and a live evaluation path with two Euler reconstruction steps; its manifest does not explicitly record an EMA-materialized SOURCE view. The newer runs explicitly use an EMA-materialized SOURCE and four Euler steps. Equal checkpoint bytes therefore do not establish an identical numerical endpoint.

Classic uses older State/cross-spectrum definitions, CO/T-only topology, paired field MSE with a 5% allowance and SOURCE-risk denominators, B32 fractional random batches, runtime seed 42 and 38 recorded batches per reporting epoch. The newer runs use fair conditional State/Scale/Shape over all five fields, physical masses, a selected group/triple bank, signed mean-risk feedback with allowances derived from 10% SOURCE member MSE, B64 continuous full-pass sampling, seed 47 and 12 batches per reporting epoch. Raw loss magnitudes and equal epoch labels cannot establish matched scientific difficulty or equal exposure across these contracts.

Classic TRAIN Shape means fall from 0.00415722 over epochs 11–25 to 0.00316417 over 451–500 and 0.00224186 over 4976–5000. Its displayed horizontal axis spans 5,000 epochs and its coherence vertical axis is logarithmic; both compress visible noise relative to the newer 500/778-epoch linear plots. At epochs 451–500, Shape coefficient of variation is still 9.22% in classic versus 11.28% in SourceScaledRisk. Visual smoothness is not a matched estimator comparison.

The classic saved first-32-condition/Q4096 VALIDATION before/after records show final epoch-5000 State/spectrum/topology reductions of 26.42%/20.67%/54.54%, but temperature endpoint MSE ratio 1.08020 fails its historical 1.05 allowance. Its saved selected checkpoint is epoch 1210, marked eligible with maximum ratio 1.04498 for temperature. This is distinct from the older 1,000-epoch Run11f whose selected checkpoint was epoch 595. Preserve this positive historical selection example without equating smooth TRAIN descent with an acceptable final endpoint or cross-version superiority.

Round 3's existing native VALIDATION100 endpoint rejects both raw paths: SourceScaledRisk State/Scale/Shape changes are −0.15662/+0.16925/−0.30526 in frozen TRAIN SOURCE units, four signed-field conditions fail, and regional normalized improvement is −0.47563. RawRisk changes are +0.15316/+0.14413/−0.33226, all five conditions fail, and regional improvement is −0.63442. Those are saved assessments, not newly generated predictions or new task scoring.

## Known limits and a bounded follow-up question

The audit establishes the actual schedule, absence of a recorded calibration switch, early sampled-score nonmonotonicity, weak early field feedback and a strong common-sampling signature in Shape. It does not establish saturation, vanished Shape gradients, an LR implementation defect, a unique cause of the early rise, or a causal parameter-gradient benefit from condition-specific queries.

A later authorized estimator claim would need a few matched TRAIN minibatches with the same query marginals, inverse weights, duplicate handling, members and reference, recording family/constraint directional products and the actual AdamW action alongside same-query and independently queried pre/post scores. This is a targeted unresolved check, not an experiment executed by this audit or permission for a broad diagnostic campaign. Round 3 has scheduled saved checkpoints only at epochs 0, 100 and 500; the epoch-10/25 window summaries come from history rows, not recoverable model checkpoints. Do not propose retrospective epoch-10/25 model probes as if those weights exist.

## Source index and reproduction

All run-relative paths and SHA-256 identities used by the script are in [history_summary.json (local-only)](../../../../Coherence_Adaptive_Repair/Round_5/evidence/history_summary.json). The summary also records exact history-to-column mapping through the script, warmup evidence, fixed-definition counts, scheduled checkpoint filenames, compact windows, coupling comparisons and the classic saved assessment. No full configuration, full CSV history or plot is copied here.

| Source | History/configuration | Original figure |
|---|---|---|
| Classic coverage-11f | [History (local-only)](../../../../../cases/turbulent_combustion/runs/_MultiTask/11f_P3_coverage_freshConFIG_5000ep_GPU2_20261101T035441Z/20261101T035631Z_989cecd1/metrics/history.jsonl), [resolved configuration (local-only)](../../../../../cases/turbulent_combustion/runs/_MultiTask/11f_P3_coverage_freshConFIG_5000ep_GPU2_20261101T035441Z/20261101T035631Z_989cecd1/resolved_config.yaml) | [PDF (local-only)](../../../../../cases/turbulent_combustion/runs/_MultiTask/11f_P3_coverage_freshConFIG_5000ep_GPU2_20261101T035441Z/20261101T035631Z_989cecd1/loss_history.pdf) |
| Round 2 selected full model | [History (local-only)](../../../../../cases/turbulent_combustion/runs/coherence_adaptive_repair/round_2/runs/formal/FFM_Selected_Bank_Uniform_FullModel/20261118T001357Z_f8b517be/logs/history.jsonl), [resolved configuration (local-only)](../../../../../cases/turbulent_combustion/runs/coherence_adaptive_repair/round_2/runs/formal/FFM_Selected_Bank_Uniform_FullModel/20261118T001357Z_f8b517be/resolved_config.yaml) | [PDF (local-only)](../../../../../cases/turbulent_combustion/runs/coherence_adaptive_repair/round_2/runs/formal/FFM_Selected_Bank_Uniform_FullModel/20261118T001357Z_f8b517be/logs/coherence_history.pdf) |
| Round 3 SOURCE-scaled risk | [History (local-only)](../../../../../cases/turbulent_combustion/runs/coherence_adaptive_repair/round_3/runs/formal/FFM_SourceScaledRisk_Uniform_Repair/20261120T144800Z_5ccb187e/logs/history.jsonl), [resolved configuration (local-only)](../../../../../cases/turbulent_combustion/runs/coherence_adaptive_repair/round_3/runs/formal/FFM_SourceScaledRisk_Uniform_Repair/20261120T144800Z_5ccb187e/resolved_config.yaml) | [PDF (local-only)](../../../../../cases/turbulent_combustion/runs/coherence_adaptive_repair/round_3/runs/formal/FFM_SourceScaledRisk_Uniform_Repair/20261120T144800Z_5ccb187e/logs/coherence_history.pdf) |
| Round 3 raw risk | [History (local-only)](../../../../../cases/turbulent_combustion/runs/coherence_adaptive_repair/round_3/runs/formal/FFM_RawRisk_Uniform_Repair/20261120T144800Z_e7bfffed/logs/history.jsonl), [resolved configuration (local-only)](../../../../../cases/turbulent_combustion/runs/coherence_adaptive_repair/round_3/runs/formal/FFM_RawRisk_Uniform_Repair/20261120T144800Z_e7bfffed/resolved_config.yaml) | [PDF (local-only)](../../../../../cases/turbulent_combustion/runs/coherence_adaptive_repair/round_3/runs/formal/FFM_RawRisk_Uniform_Repair/20261120T144800Z_e7bfffed/logs/coherence_history.pdf) |

The [Round 3 terminal report](../../../../_Reports/Coherence_Adaptive_Repair/_bk/Round_3/report.md) supplies final native outcomes and the [completed cached Shape diagnostic (local-only)](../../../../../cases/turbulent_combustion/runs/coherence_adaptive_repair/round_3/assessment/shape_queries_recovery_attempt3/disposition.json) supplies the fixed-endpoint scalar replay. The schedule implementation is `_scheduled_model_lr` in [post_training.py](../../../../../src/phycoflow_reconstruction/training/post_training.py).

Run from the repository root with Python 3.10+ and PyYAML. Output defaults to the disposable `/tmp/car_round5_history_audit` directory; `--output-dir` can select another scratch destination. Historical payloads must be available through their existing aliases. The script writes only `history_summary.json` in the requested output directory and does not copy histories or load checkpoints.

```bash
python UpdatePlans/Coherence_Adaptive_Repair/Round_5/evidence/analyze_saved_histories.py --repo-root . --output-dir /tmp/car_round5_history_audit
```
