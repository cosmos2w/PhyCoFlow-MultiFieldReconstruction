> **Historical record:** Round 4 is closed. This document is preserved for reference and is not a current launch instruction.

# Design discussion after the completed Round 3

Date: 2026-11-21. Participants: primary agent; the user-requested **GPT-6.1 sol max** reviewer `sol_next_round_design`; an independent read-only implementation audit `implementation_generality`. Source snapshot: branch `Coherence_Adaptive_Repair`, `440545c`. No model, training, inference, GPU reservation, or code change was performed. This review supports the proposed [Round 4 workplan](WORKPLAN.md). Round 3 remains closed.

## Recommendation

Use one bounded strategy change: **construct a field-constrained SSS teacher in endpoint space, then amortize it through native training**, compared with ordinary native supervised continuation. Gate student training on whether the offline teacher has a useful feasible result. Require the final student to retain field accuracy and improve the unoptimized regional relationship beyond both SOURCE and the native control. Charge target generation and teacher computation to the candidate's cost.

This is more consequential than reducing minibatches or another round of loss-weight repair. It also has a real possibility of failure: dense truth already supplies excellent native supervision, and a teacher may add cost without useful information. The proposed control makes that failure decidable. The target is a useful, materially cheaper procedure, not preservation of every earlier optimizer choice.

## What the discussion changed

| Question or challenge | Resolution incorporated in the workplan |
|---|---|
| Does 98.5% update time prove the backbone backward is the bottleneck? | No. That timer also includes descriptor and CPU persistence work. Use one bounded scoped profile; claim only the actual reduction in complete and inclusive work. |
| Would exact ordinary-CQ context reuse be the safer next speed change? | It is a concrete opportunity: the physical wrapper already supports live context reuse, while ordinary CQ repeats conditioning across stages. Defer it here because it does not by itself answer failed scientific utility, and the selected native learner no longer performs those repeated rollout stages. Do not discard the engineering finding or advertise an unmeasured speedup. |
| Could one fixed scalar backward or periodic ConFIG solve this cheaply? | Those routes already exist. Older scalar runs were faster but did not establish sustained joint quality; periodic reuse failed directional probes. They remain available research tools, but are not a fresh missing mechanism. No optimizer tournament is added. |
| Can jointly repaired K4 particles use the ordinary fair SSS estimator? | No independence claim survives joint target optimization. Define the teacher as an explicit finite empirical law and use its V-score separation coefficient. Keep independent J4 fair student evaluation unchanged. |
| Should the teacher's mean error use the signed bias-corrected estimator? | No. The empirical law's mean is its exact particle average. Use the squared error of that average for teacher feasibility; retain the signed unbiased estimator for independent student draws. |
| Can clipping a proposed step enforce constraints without a new solver? | Use scalar quadratic feasibility along a ray anchored at immutable SOURCE. A ray from the current point can lock at an active boundary. The SOURCE-anchored radial map is feasible but neither an exact projection nor a descent guarantee. |
| Should both students mix truth with source/teacher targets? | Omit a new mixing coefficient. Use pure teacher targets versus pure truth targets under matched native fitting. This tests the full repair package, with source regularization and SSS effects not separately identified. |
| Does the existing TRAIN800 cache cover the learner's training set? | No. Preserve full declared TRAIN membership for label preparation and both learners. Approximately-10% evaluation remains separate. Do not relabel a small-bank pilot as full-membership acceleration. |
| Is snapshot ID enough to identify a teacher target? | No. Freeze observations, values, support, SOURCE, rollout and member seeds. Use the same conditioning for both learners. A teacher conditioned on yesterday's sensor set is not a valid target for a resampled sensor set. |
| Can native dense arrays be passed directly into the existing adaptive bank? | Only through the frozen support operator. The bank's exact physical-support contract and current expanded two-member path need explicit handling; a K4 tensor alone proves neither correct support nor four-atom participation. |
| Can the base-training command perform the proposed native continuation unchanged? | No. It currently creates a fresh model, defines full-pass epochs and lacks this loop's warmup. Require SOURCE materialization, the twelve-batch reporting epoch, fixed observations/targets and the explicit warmup through the smallest existing lifecycle integration. |
| Is there a second frozen absolute member-error allowance for teacher projection? | No. Declare the teacher's member-error slack equal to the existing absolute mean-risk allowance. Preserve the student's separate relative member guard. |

These corrections are substantive. They prevent apparent speed gains from being obtained by changing data coverage, using stale conditioning, weakening the observer, or incorrectly claiming an unbiased generative score.

## Why this is the selected local mechanism

The [Round 3 report](../../../../_Reports/Coherence_Adaptive_Repair/_bk/Round_3/report.md) supports closing the feedback-conditioning question: its final SourceScaledRisk model still fails four fields, Scale, and the regional task. The retained exact reuse saves memory, not measured time. Its Shape replay supports a query-estimator hypothesis, but a lower scalar-query variance has no demonstrated ability to repair the field failure. A standalone query-only comparison would leave the user's main concern unresolved.

The independent implementation audit favored exact context integration as the first engineering opportunity, and identified one-backward scalarization as an existing option if algorithm changes were allowed. The sol reviewer favored moving SSS outside repeated neural updates. The final choice follows the user's request for a strategy-level improvement and the retrospective's limit of one additional local mechanism study: test amortization as one package, with an early teacher construction decision and a native supervision control. Avoid combining both recommendations into another large campaign.

The proposed computational difference is explicit. Current training differentiates SSS and field feedback through a multi-stage generative endpoint. Teacher construction differentiates SSS only with respect to stored physical fields; native learner updates use the existing random-time RF regression. The distance definitions, complete anchors and final scientific observer remain intact. The teacher reduction and training algorithm change and receive explicit names. No theorem converts this substitution into exact ConFIG, posterior preservation, or guaranteed held-out improvement.

The general idea is a reusable split between a physical target-construction stage and the model's native learning stage. A deterministic model could fit the supplied target by regression; a flow model uses its native bridge; diffusion would require its own correctly implemented native objective. Only the present FFM adapter is in scope. This is a capability boundary, not a promise that every reconstruction model accepts the same code path or benefits from it.

## The three scientific outcomes must stay separate

1. **Teacher construction:** Can the finite representations find an attainable useful correction under explicit native field/sensor bounds? This is a TRAIN optimization result. It may fail because the constrained observable objective is unhelpful or because the declared bounded solver is ineffective; do not diagnose a universal impossibility from that result.
2. **Amortization:** Does native fitting transfer the teacher's useful structure into a reconstruction model while retaining fields? Target feasibility does not guarantee this. A teacher that improves while its student fails closes this proposed amortization recipe.
3. **Unoptimized utility:** Does the student improve the reused regional relation and retain the event beyond SOURCE and ordinary supervised continuation? Better fitted SSS alone does not answer this. These tasks are excluded from optimization but have already been exposed in prior rounds, so they are development evidence rather than untouched confirmation.

A positive result must survive all three distinctions and its inclusive cost. A negative result still makes a concrete decision: retain SOURCE/C8 or a qualifying native continuation, and stop this local recipe rather than enlarge it with more loss terms.

## Bounded workload and communication

The detailed workplan proposes 100 teacher reporting epochs, two student paths with a 100-epoch learning review and conditional 500-epoch maximum, 24 occupied GPU-hours for the whole campaign, and explicit CPU/storage limits. These are prospective limits, not measured speed forecasts. Full bank preparation is training work and is charged once to the candidate. GPU0 is primary; GPU1 may run the independent native control after teacher acceptance. Runtime device selection is not a frozen scientific restriction.

Teacher learning uses the same reporting-epoch visit convention rather than calling a full dataset pass an epoch after the fact. No student horizon is shortened below a meaningful review to satisfy a speed headline. The required timing check can close an obviously uneconomic route before expensive work. It cannot establish learning quality. Long-run monitoring, if execution is later authorized, belongs to the named sparse Luna monitor under [AGENTS.md](../../../../../AGENTS.md#mandatory-run-monitoring-policy).

The final visual story should show the physical field bounds and unoptimized relationships together, then distinguish teacher construction from held-out student behavior, and finish with complete learning/cost evidence. A few fixed cases and all-field quantitative tables are more informative than a large favorable atlas. Reconstruction panels must carry their own fieldwise errors.

## Evidence inspected and claim limits

The review read the Round 3 final report, decision/cost JSONs, saved configuration and implementation paths; the overall [retrospective](../../../../Project_Retrospective_20261120/REPORT.md); the [earlier FAST study](../../../../_Reports/FAST_LAGRANGIAN_LPC_REPORT.md); and current native RF, SSS, risk, rollout and optimizer code. The current code-to-proposal mapping is in the workplan. Historical scalar/periodic results use older SOURCE/observer versions and inform what not to call new; their timing is not a current speed prediction.

The native training rationale is consistent with [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747), which formulates vector-field regression without simulating a learned trajectory during training. That reference supplies no evidence that the proposed repaired labels improve scientific utility. The design has not yet been implemented or benchmarked, and no new empirical success is claimed.
