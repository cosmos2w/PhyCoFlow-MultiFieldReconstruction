> **Historical planning snapshot:** Round 5 has concluded. This document is preserved for reference; see the final report for its outcome.

# Round 5 design discussion and decision record

Date: 2026-11-23. Participants: primary agent and explicitly requested GPT-6.1 sol, maximum reasoning effort, subagent `round5_sol_review`; independent saved-history and implementation audits were delegated. The team inspected current code at `de86b1f`, terminal Round 3/4 evidence, the project retrospective and LPC–SSS documents, full recorded histories, and the requested figures through their PNG companions. No model inference or training was launched. The executable proposal is [WORKPLAN.md](WORKPLAN.md); the compact handoff is [EXECUTION_PROMPT.md](EXECUTION_PROMPT.md).

## Decision

Return to native ground-truth supervision plus SSS. Compare ordinary native continuation with one direct enhancement package. Before that comparison, jointly diagnose early-action validity, exact repeated computation and Shape query covariance within a small preparation budget. Retain SOURCE/C8 unless the final field, relationship and task evidence jointly qualify a replacement. Teacher–student learning remains closed.

The owner's criticism of Round 4 is correct. A feasibility-preserving transform can undo or exceed an intended descent, and 100 reporting epochs provided only 9.6 visits per condition. Tests of score algebra, gradients, physical constraints and resume did not answer whether complete real updates actually lowered the objective. The fixed recipe restriction then prevented a bounded correction. The new plan explicitly checks complete finite updates, allows one evidence-directed startup correction before formal freeze, and distinguishes execution failure from a learning verdict. It does not attempt to justify an imperfect teacher merely because it might be cheaper.

## What the historical review actually established

| Concern | Established evidence | Consequence |
|---|---|---|
| Warmup disappeared | False: Round 3 used a 50-epoch ramp, longer than Round 2's approximately 6.333 epochs. Scale still rose later. | Inspect the actual optimizer action and supervision route; do not repeat warmup-only tuning. |
| A calibration switch caused the jump | Not supported: the recent runs loaded fixed prefit coefficients from their first action. | Do not spend the round repairing an absent epoch-10 transition. |
| Shape is wholly flat and parameters oscillate | Shape improves modestly; first-difference correlations across three runs are about 0.999 under exactly matching recorded stochastic inputs. | Separate stochastic query/condition effects from learning. Fixed-SOURCE Round 3 replay motivates a parameter-gradient estimator test, not a learning claim. |
| Classic 11f proves the new recipe should work | Classic uses different scores, source materialization, rollout and field feedback. Its selected epoch1210 qualified historically; its final epoch5000 temperature error did not. | Preserve its positive evidence without equating visual smoothness with acceptable final reconstruction. |
| Field constraints actively protected early updates | Round 2/3 native loss is skipped by external policy; dual multipliers start at zero and early feedback is weak. Risk logging is pre-action. | Restore native supervision and inspect actual finite field changes from the outset. |
| Exact State/Scale reuse already accelerated training | Round 3 measured 0.468 GiB (2.57%) memory saving, without a speedup. | Retain it; pursue demonstrated repeated context/backward work and measure complete epochs. |

Full source identities, numeric windows and reproduction instructions are in [the CPU history audit](evidence/HISTORY_AUDIT.md). The classic and recent metric magnitudes must not be compared as one definition. No actual AdamW displacement trace exists in the saved histories, so optimizer conflict is a testable mechanism, not a proven historical root cause.

## Alternatives considered with sol

**Retain multi-family ConFIG and add five field-gradient projection constraints.** Rejected for this round: it adds derivative work to an already expensive route, and the user needs one practical improvement. ConFIG's direction property concerns the constructed direction; Adam's preconditioning/momentum and the finite action require their own checks. The original [ConFIG paper](https://arxiv.org/abs/2408.11104) describes conflict-free gradient construction. The later [Cautious ConFIG paper](https://arxiv.org/abs/2610.31855) explicitly studies optimizer-induced conflicts. Those papers support inspecting actual actions; neither establishes that this repository's historical early jump was caused by that mechanism. We are not adding another ConFIG variant as a mature arm.

**A single fixed scalar with native supervision and SSS.** Selected. One summed SSS backward removes repeated full-model family VJPs, permits a cheap scalar-gradient versus actual-displacement test, and answers the paired-GT problem directly. The preferred path also combines native supervision into that backward; a measured joint-memory failure permits exact separate native-gradient accumulation, whose cost must be reported. Its limitation is deliberate: there is no per-family descent guarantee, so final familywise guards remain essential. The [Flow Matching paper](https://arxiv.org/abs/2210.02747) describes supervised regression of conditional-path vector fields; the actual local FFM adapter defines the bridge and masking used here. A duplicated ground-truth endpoint MSE would be a different objective with a spread penalty.

**Give both mature arms the same endpoint filter.** Not selected. It would provide cleaner component attribution but force an ordinary single-velocity native update to pay for repeated J2/four-Euler rollouts. The chosen control answers whether the entire enhancement package earns its cost over ordinary native continuation. Describe this comparison honestly; do not infer that SSS alone caused any difference.

**Use a global TRAIN SOURCE number to filter every minibatch.** Rejected. A difficult minibatch can exceed that global number even at unchanged SOURCE. The selected rule pays for same-input, same-noise SOURCE pairing, checks every finite proposal and permits strictly improving restoration from an already violating local state. Sol identified a boundary issue in the first draft: aggregate restoration must trigger only above numerical tolerance, otherwise a tiny positive violation cannot decrease by more than the demanded tolerance. The workplan includes that correction.

**Replace the model optimizer whenever a step conflicts.** Rejected. Candidate-owned traces may justify one global zero-first-moment setting before freeze, or one lower LR schedule when finite overshoot is demonstrated. A shrinking action cannot repair a positive first-order directional derivative. Formal execution has no per-action solver switching, unbounded line search or hidden restart search.

**Treat the old fixed-SOURCE Shape SD result as proof of an estimator improvement.** Rejected. Shared queries create cross-condition covariance, so grouping is plausible, but parameter gradients and finite actions must be checked. The plan tests eight groups with unchanged per-condition marginals and query-condition work, preserving weights and Scale/Shape coupling. It keeps the old estimator if the bounded test does not justify replacement.

**Precompute a complete topology teacher/cache or replace persistence.** Rejected. Prediction pairings remain parameter dependent; continuously sampled reference query identities may rarely repeat. Only a cheap hit/cost check for exact immutable reference reuse is allowed. The current training arithmetic and tie gradients remain protected. A lower scalar SD, fewer nominal diagrams or less allocated memory is not a training speed result.

## Code-to-design audit

Paths in this table are under `src/phycoflow_reconstruction/` unless stated otherwise. Line references are audit locations at `de86b1f`; future execution should follow the named functions if code moves.

| Existing implementation | Finding | Proposed bounded change |
|---|---|---|
| `training/post_training.py:2905`; `config/validate.py:1824` | External policy skips native supervised loss and constrains compatible settings. | Explicit direct-supervision policy with a test that native loss produces gradients. |
| `training/fidelity_budget.py:491–625` | Three family parameter VJPs plus constraint VJP, clipped composition then AdamW; risk measured before action. | One fixed scalar backward; actual-displacement dot check and finite paired field filter. |
| `models/flows/pointcloud/adapters/gl_rbf_cq_adapter.py:346`; `models/flows/pointcloud/core/portable_core.py:2678` | Native flow-matching objective is already available. | Reuse its intended semantics, rather than per-member endpoint MSE. |
| `training/rollout.py:137`; `models/flows/pointcloud/core/portable_core.py:2457` | Ordinary CQ reconstructs condition context per velocity call. | Port qualified within-graph reuse from `models/compatibility/physical_gl_rbf_cq.py`; preserve checkpointing/ties. |
| `coherence/query_sampling.py:143`; `training/scalable_runtime.py:425` | Shared query batches and explicit Scale/Shape exposure coupling. | Add bounded condition-group assignment with unchanged per-condition law. |
| `coherence/shape_compute.py:341`; `coherence/families/topology/persistence.py:143` | J2 prediction/reference diagram banks include CPU pairing and live gradient regathering. | Preserve exact kernel; only qualify narrowly useful immutable reference reuse. |
| `training/scalable_runtime.py:402`; `training/post_training.py:3305` | Existing rejection restores the sampler and raises an exception. | Explicit consumed-but-rejected attempt semantics, separate from numerical failure and accepted optimizer counters. |
| `training/round2_virtual_probe.py`; `training/topology_constraints.py:337` | Transaction and finite-action mechanics exist. | Reuse mechanics and extend necessary sampler/generator/EMA rollback, without the old full controller. |

## Claim boundaries and paper use

LPC supplies the reason to examine mixed relations in addition to field errors. SSS supplies declared finite observables. Native supervision supplies the original reconstruction task. The finite filter supplies a sampled action safeguard. Independent-of-training task scoring supplies evidence of scientific usefulness. These are different links; none substitutes for the next.

The hoped-for result is better unoptimized relationship recovery than ordinary continuation at acceptable field accuracy and measured training cost. No such result exists yet. Round 5 can also end decisively with ordinary continuation preferred, with exact execution improvements adopted but both models rejected, or with the proposed safeguarded scalar optimizer found impractical. That outcome should close the current mechanism question rather than trigger another indefinite local search.
