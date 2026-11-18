# Round 2 design discussion

Date: 2026-11-16. Participants: the primary agent and the user-requested GPT-6.1 Terra max subagent, `round2_terra_review`. Scope: read-only source/evidence review and preparation of the [Round 2 workplan](WORKPLAN.md); no training, source changes or branch mutation. The subagent independently read the branch roadmap, Round 1 plan/design/report and actual training, model, allocator and source artifacts.

Execution authorization received later on 2026-11-16: the user's Round 2 goal approves execution within the reviewed 130 GPU-hour ceiling and explicitly permits GPUs 0 and 1 in parallel. This supersedes the preparatory GPU1/GPU2 recommendations below. The historical design discussion remains preserved; live execution provenance is in [execution state](EXECUTION_STATE.md).

Scheduling correction authorized on 2026-11-17: source-local qualification, allocation audits and mature admission proceed independently when their actual inputs and the whole-round budget permit. Earlier wording requiring all four execution qualifications before either source can advance is superseded by the amended [workplan](WORKPLAN.md) and [scheduling review](SCHEDULING_REVIEW.md). The two-family nonuniform-allocation gate, immutable scientific identities, four mature horizons and TRAIN-selection-before-VALIDATION barrier remain unchanged.

User amendment on 2026-11-18: cancel further GPU deployment-latency measurements, release GPU1, and use GPU0 for remaining necessary GPU work. The amended [workplan](WORKPLAN.md) governs execution; historical deployment proposals below remain the design record. Preserve completed measurements, report missing latency explicitly, and continue the fixed scientific comparison without reopening qualification. Latency-dependent routes remain unavailable where their measurement or priority is absent; the scientific gates, authorized horizons and information barrier are unchanged.

## Joint recommendations

- Preserve the Round 1 expanded representation and measured event gain, while keeping its negative allocation/output-promotion disposition intact. New Scale retention, complete costs and actual deployment address the unfinished branch question.
- Keep all four requested optimization values. They mean 768 visits per reporting epoch, not 8000, and require `coherence.compute_budget.batch_size: 64` as well. Record real objective-bearing IDs; exact microbatch accumulation precedes ConFIG and AdamW.
- Use Senseiver as a real point-model case with J1, not duplicated generator outputs. Pin the completed epoch-5000 `last.pt` literally and preserve native unclamped sensor predictions; sensor-null correction retains those predictions.
- Use the same selected bank, effective loss coefficients and explicit fidelity contract within each source's matched pair. Four mature runs are sufficient for an integrated-package decision. Historical C8/expanded checkpoints provide context without a new causal C8 comparison or cost ratio.
- Exercise the signed mean-risk contract in both mature FFM arms. Freeze fieldwise absolute allowances from positive TRAIN SOURCE member risks and keep signed SOURCE mean-risk estimates out of denominators. Legacy member-MSE qualification remains an additional operational-replacement requirement.
- Learn a compact SOURCE-conditioned residual first. Round 1's teacher does not have the needed Scale behavior, so distilling it is not a solution to that defect. The residual uses inference-available inputs, member-specific SOURCE output, zero initialization, bounded correction and query-chunk invariant evaluation.
- Validate allocation through actual uncentered parameter-gradient estimators and full charged cost on both real models. The cost-aware second-moment proposal is a heuristic; equal-time MSE or MSE-times-cost decides its usefulness. Do not equate source/endpoint sketches with parameter gradients.
- Freeze a new task excluded from selection and tuning. The old event is reused development evidence, and same-trajectory VALIDATION does not become independent physical data merely because a query was reserved.
- Begin complete timers before preparation, use runtime GPU1/GPU2 selection and the mandatory named Luna monitor, and preserve all old artifacts. Treat the total GPU envelope as a new proposal, with real B64 timing required before mature launch.

## Concrete code findings incorporated

| Finding | Consequence |
| --- | --- |
| `training/update_budget.py` returns the explicit epoch length; `data/topology_subset.py` reshuffles independently at full-dataset boundaries. | B64/12 at 1000 reporting epochs gives 12000 updates, 768000 visits and 96 TRAIN8000 passes. |
| `training/post_training.py::_coherence_objective` slices by a separate coherence batch cap. | Changing optimization B32 to B64 alone does not establish increased loss-bearing coverage. |
| Senseiver capabilities are deterministic, and native historical continuation disables clamping/smoothing. | Preserve J1 and SOURCE sensor semantics, with separate point-law interpretations. |
| `_expanded` scores every added group/triple; the current adaptive sampler targets core pairs. | Added-leaf allocation requires real implementation; charged enumeration may be the better small-bank decision. |
| Ten-epoch internal calibration measures each arm's trainable parameter gradients. | A backbone and zero-init residual head would fit different coefficients despite sharing a bank. Prefit one common vector per source, freeze it before either formal run and check equal weighted endpoint scores. |
| `finish_measurements` uses second moments, but runtime sketches and frozen pair costs are narrower than a complete native parameter audit. | Audit actual parameter-space estimators and selection/exposure correction, including residual parameters. |
| The mean-risk controller supports vector allowances, but its training integration repeats one scalar and fits one statistic. | Implement a fieldwise artifact and member/mean statistics from the same reference panel, then qualify the real lifecycle. |
| Current online mean risk uses a frozen reference rather than a same-batch paired SOURCE rollout. | Preserve and label this design; any proposed pairing needs extra measured compute and implementation. |
| Member dispatch depends on the wrapper's generative capability. | Preserve the FFM SOURCE capability behind the deterministic head, including independent members and both-member head gradients; Senseiver remains J1. |
| The old campaign driver freezes FFM/B32/13 and an FFM/EDM source catalog. | Build a distinct Round 2 entry point using reusable pieces; do not bypass Round 1 admission rules. |
| Current main and the existing exact-named Round 1 branch diverged, and generic CAR code is absent from main. | Preserve the old tip, integrate from current main with a reviewed history-preserving merge, and avoid force-pushing. |

## Choices made to keep the comparison bounded

The two full-model controls and two residual candidates share a selected target within each source. A qualified sampler may differ between uniform control and residual candidate; mature findings then describe the integrated package. The allocator has a separate estimator decision, and the bank has a task-conditioned representation decision. No causal decomposition of learner, representation, sampling and fidelity effects is promised.

The FFM allowance is `0.10 * TRAIN SOURCE member_MSE_f` in squared field units. This is an explicit design scale, not a theorem about uncertainty or a relative mean-risk percentage. Mean-risk feasibility, proper represented/task scores, spread diagnostics and legacy operational member-risk eligibility remain distinct. Senseiver keeps its own point-fidelity contract.

The shared Scale intervention multiplies only the `scale.joint` block coefficient by a bounded TRAIN-selected ratio, with the means still inside that complete norm and its `mean_weight` fixed. The common coefficient fit has a specified SOURCE-gradient RMS statistic and clipped median-reference formula; the ratio probe has fixed seeds, response/feasibility criteria and ratio-one fallback. A whole-family positive scalar may cancel in normalized ConFIG directions, so it is not a credible Scale-directed remedy by itself. Actual proposed-step and finite-response checks qualify this choice; the mature result still decides whether it works. The separate common assessment observer retains fixed unit internal coefficients independent of training weights. Allocation audits use the head's actual trainable Jacobian when the head is the deployed learner, with no automatic transfer from a backbone-gradient audit.

Final Terra review removed accidental adoption blockers: either pre-nominated full-model or residual output that earns task eligibility, complete cost accounting and latency within 1.20 times SOURCE can qualify relative to SOURCE, independently of whether the other method qualifies. A 20% saving is claimed only against an eligible measured comparator. It also replaced ambiguous post-validation output choice with TRAIN-frozen amplitude nominations and method priorities followed by fixed accept/fallback gates. This operational cohort decision is not independent post-selection confirmation.

The reserved regional CO–U_1 covariance is selected for signed species/velocity organization and its separation from the old hot/CO/CH4 excursion witness. Its exact definition is frozen from coordinates and TRAIN truths; model gains on it must remain unavailable during method selection. It earns only the stated regional-task claim.

The proposed 130 GPU-hour envelope starts from approximately 93.24 hours for four linearly scaled Round 1 learning equivalents and allows the rest for qualification, selection/audits, assessment, deployment and protected closeout. This is not a measured Senseiver/residual forecast or execution authorization. Real complete-update timing may lower the forecast or show that the obligation does not fit; that is resolved before formal launch without shortening the final comparison by stealth.

## Measurements needed during execution

Resolve source/hash/weight-view and native sensor parity once; qualify all four B64 paths through fresh-process resume; measure complete updates/callbacks, memory and real objective-bearing coverage; fit and freeze selected banks/fieldwise risk artifacts; qualify or reject allocation under its charged scope; and seal the forecast, manifest, reserved task and cost allocation before mature training. These are enabling measurements with a bounded implementation allowance, not new efficacy gates that terminate learning after a short score.

All final adoption claims require the workplan's actual mature/held-out evidence. The preparatory discussion establishes a concrete executable question and its boundaries, not that any proposed mechanism already works.
