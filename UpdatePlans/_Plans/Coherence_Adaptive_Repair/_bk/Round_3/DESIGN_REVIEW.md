> **Historical record:** Round 3 is closed. This document is preserved for reference and is not a current launch instruction.

# Design review after Round 2: what to compute, what to learn

Date: 2026-11-20. Participants: primary agent and the explicitly requested GPT-6.1 Terra max subagent `round2_terra_review`. This review used the closed Round 2 report, saved configurations, complete recorded histories, the actual loss figure and current mathematical/code definitions. It launched no training, inference or deployment benchmark. The next workplan is [WORKPLAN.md](WORKPLAN.md); closed Round 2 files remain intact.

## Recommendation

Make one matched FFM comparison of the current field-feedback rule with a SOURCE-scaled rule, with a shared fifty-epoch warmup. Remove demonstrated repeated State/Scale transforms in both paths. Investigate shared-query Shape noise independently on saved TRAIN endpoints. Review learning at epoch 100 and compare at a provisional endpoint of epoch 500. Retain SOURCE/C8 unless field, coherence and unoptimized scientific-task evidence jointly support replacement.

The reviewer initially favored a Senseiver-residual paired-field anchoring experiment. Discussion changed that recommendation: the user's concrete concern is FFM learning, the FFM field/Scale failure is material, and changing the objective or model family before resolving its weak feedback would obscure this question. A query-only mature comparison was also considered and deferred: less Shape noise has no demonstrated ability to repair the current field failure. There is no extra residual, anchoring, query-learning or Senseiver arm in this round.

## What the curve does and does not say

Original run: `cases/turbulent_combustion/runs/coherence_adaptive_repair/round_2/runs/formal/FFM_Selected_Bank_Uniform_FullModel/20261118T001357Z_f8b517be`. The primary agent inspected the PNG companion to the specified PDF and computed the following descriptive summaries from its complete saved `logs/history.jsonl`; the reviewer independently inspected the PDF/history.

| Recorded sampled TRAIN quantity | Epochs 11–25, mean | Epochs 729–750, mean | Across-epoch SD, early / late |
|---|---:|---:|---:|
| State weighted family | 0.0520 | 0.0442 | 0.0008 / 0.0006 |
| Scale weighted family | 0.2835 | 0.2546 | 0.0073 / 0.0063 |
| Shape weighted family | 1.135 | 1.004 | 0.110 / 0.110 |
| Shape self | 0.617 | 0.529 | 0.127 / 0.129 |
| Shape mutual | 0.492 | 0.447 | 0.049 / 0.028 |

Shape has learned something; describing it as wholly unchanged would be incorrect. Its persistent variability is concentrated in the self term and can mask later modest improvement. These windows change model parameters, conditions, latent members and queries, so their SD is not a fixed-model variance decomposition or evidence that every parameter trajectory oscillates. The Shape sampling mechanism supplies a concrete hypothesis: a single unary field and sampled sign is shared by the whole condition batch, with three sampled pair draws and their signs/lines shared likewise. Duplicate pairs share an inner query. Larger condition batches do not remove this common query component.

Scale joint rises from approximately 0.275 at epoch 1 to 0.314 over epochs 26–50 before declining to approximately 0.284 over epochs 729–750. It therefore does not simply descend from its starting value. The history is not an independently measured native field curve. In the external-policy route, `data_loss` is the sampled signed mean-risk mean, not the native pretraining loss or native member MSE (`training/fidelity_budget.py`, endpoint diagnostics; `training/post_training.py`, external-policy row recording).

Three-family ConFIG constructs a parameter direction, not a scalar objective minimized monotonically by AdamW. Clipping, Adam preconditioning, field feedback and stochastic inputs further separate a per-batch direction claim from population score descent. The recorded native VALIDATION failure decides replacement eligibility; the training trace cannot reverse it.

## Warmup: an execution-unit omission, not a missing ramp

The saved `resolved_config.yaml`, `metrics/lr_warmup_updates.jsonl` and `training/post_training.py::_scheduled_model_lr` agree: the rate ramps from `5e-6` to `5e-5` over about **6.3 reporting epochs**, then stays at the target. No evidence supports claiming the ramp was disabled. The file named `run11w_config_lagrangian_lr_warmup_20ep.yaml` also sets a trial horizon; its filename is not evidence of a twenty-epoch warmup.

The concrete execution omission was carrying a low-level scheduling count without expressing and verifying it under the user's epoch convention. The earlier explanation also repeated that internal unit in user-facing training descriptions. New schedules must be explicit in epochs, including warmup and resume-aware scheduler position. Historical records are preserved rather than renamed or rewritten. The user's subsequent preference is a 50–100-epoch ramp; the plan chooses **50 epochs**, shared between the compared feedback rules. It is not a claim that the optimal ramp has been established. A 100-epoch learning review then includes fifty epochs at the target rate.

The early Scale rise continues beyond the old ramp, and the field proxy rises while its constraint direction is extremely weak. A longer ramp is a sensible common startup choice, but no current evidence identifies it as a sufficient fix or the unique cause. Do not add a warmup tournament to obtain that attribution.

## Field feedback: a demonstrated imbalance worth testing

The saved `artifacts/fidelity_budget_reference.json` has positive risk scales equal to one for all fields. Its SOURCE member risks are roughly 0.010, 0.068, 0.011, 0.044 and 0.015 in the fixed model units. Over epochs 11–25 the mean constraint-gradient norm is about 0.017 versus 7.18 for the coherence direction; near epoch 750 it is about 0.91 versus 6.65. These norms do not prove the constraint is ineffectual in every direction, but they establish weak early corrective feedback. The held-out result independently establishes that the final constraint outcome was inadequate.

This is an allowed but poorly scaled design choice to investigate, not a newly proven gradient implementation defect. The proposed violation divides the same signed excess by positive SOURCE member risk and leaves the physical allowance unchanged. `MeanRiskDifferenceBudget` permits exactly this construction. The candidate's different dual step size is explicit because projected dual ascent makes early feedback depend approximately on its step size divided by the squared normalization scale. Calling this merely an equivalent numerical rescaling would be false. Conversely, choosing fieldwise step sizes that exactly reparameterize the old dynamics would not test the intended stronger/balanced feedback.

The SOURCE-scaled rule with dimensionless dual step size `1e-4` is one predeclared practical candidate. The TRAIN receipt is physical-mass weighted in **model field units squared**, not native field units squared. Native VALIDATION uses its matched SOURCE comparison and a fieldwise conversion of the fixed TRAIN allowance. A candidate reference artifact copies the existing SOURCE evidence but binds the changed risk scales/settings; it requires no SOURCE refit. The short setup check protects finite execution and resume; it does not establish convergence or justify early efficacy rejection. A failed candidate closes that route, not all signed-risk or all physical-coherence methods.

## Concrete computational opportunities

| Term | Actual current computation | Bounded correction and claim boundary |
|---|---|---|
| State | `coherence/state_compute.py::group_projection_scores` calls `exact_common_embeddings` separately for added directions; `prepare_endpoint` sorts the field marginals again even though the group score discards them. | Pack distinct projections and avoid repeated marginal sorts/CDF work. Reuse only matching directions, units and support. Keep the core joint norm and every separate group norm, with their own direction normalization. This changes execution only if values and live gradients match. |
| Scale | `coherence/query_runtime.py::_scale_statistics` computes centered coefficients; `training/scalable_runtime.py::_expanded` invokes `adaptive_fitting.py::spectral_filter`, which computes them again for triples. | Reuse the current live coefficients and reference transforms. Keep all original bands, piece means, signs and triple functions. No new retained information or improved model quality follows merely from this reuse. |
| Shape | `coherence/shape_compute.py::score` already builds each endpoint's selected diagrams once and shares them across attraction/separation. The observer has compiled-reference support in `evaluation/frozen_coherence.py`. | No claim of repeated per-member diagrams for every pair. Consider only measured reference-only training reuse; do not enumerate the entire query bank or build another caching framework. |

The stable sorting/two-term projection correction from Round 2 remains essential: identical scalar distances can conceal different tie gradients. Reuse the corrected implementation and qualify only the graph/packing actually changed, including the pre-clipping parameter-space direction because AdamW/clipping can mask a change. A demonstrated duplicate calculation justifies trying a refactor; passing its tests alone does not justify retaining it without useful measured cost or memory benefit.

With endpoint Jacobian $J$ and family endpoint cotangents $v_f$, ConFIG needs $g_f=J^\top v_f$ and parameter Gram entries $v_f^\top JJ^\top v_g$. Endpoint dot products $v_f^\top v_g$ generally differ. A single combined backward before the parameter-space coefficients are known changes the method; lagged coefficients are an approximation already represented by the existing periodic route. Neither is presented as an exact speedup or added to this plan.

## Why Shape query diversity is a diagnostic first

For fixed endpoints, shared-query batch gradients average $g_b(Q)$ over conditions; independent-query gradients average $g_b(Q_b)$ with the same marginals. Their variance difference contains cross-condition query covariances. Positive common-mode covariance makes diversity promising, while negative covariance prevents a universal variance-reduction guarantee. Duplicate-query coalescing and CPU persistence packing also affect actual cost, so counting nominal queries alone is insufficient.

A small matched query replay on cached TRAIN800 endpoints estimates score variation without another model rollout. It is independent of the feedback comparison and has a bounded CPU allowance. Scalar or endpoint-gradient variation is not parameter-gradient variation: different conditions' endpoint cotangents occupy different coordinate blocks, whereas their parameter Jacobians can align. Nonlinear ConFIG and AdamW invalidate any automatic claim that an unbiased family estimator yields an unbiased optimizer direction. The plan therefore does not reopen the rejected large gradient-MSE/cost qualification programme. A favorable diagnostic earns a concrete later learning question; it does not automatically launch that question.

## Rigor retained and work deliberately omitted

Keep physical masses and masks, native fieldwise assessment, FFM independent members, signed mean risk versus member/spread separation, sensor behavior, fixed bank/coefficients, corrected topology gradients, matched source identities, raw-output selection, complete histories and the locked TEST boundary. The reused regional task stays outside optimization and current candidate task inspection waits until the endpoint. Its previous publication prevents calling it fresh blind confirmation.

Omit new representation searches, allocation audits, full-population evaluation, nine-amplitude scans, per-checkpoint deployment timing, an extra model family, cross-arm/global admission barriers, exhaustive historical serial-cost reconstruction, new provenance/controllers and decorative plan PDFs. None is needed to decide whether this specific feedback rule produces a field-preserving, task-useful repair. Cached Shape diagnostics and optional reference-cache improvements must not delay ready training.

The two 500-epoch paths forecast approximately 36 GPU-hours from actual Round 2 epoch timing, within a proposed 45-hour total ceiling. The initial learning review is at epoch 100, with no automatic horizon extension. GPU0 is the latest primary runtime preference and GPU1 is optional. Current `main` is an ancestor of the existing exact-named branch; preserve that history and local user work. Preparation of these Markdown documents authorizes no new campaign.
