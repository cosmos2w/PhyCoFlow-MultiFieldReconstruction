# Round 7 design review: faster exact backward execution and a clearer Scale objective

Date: 25 November 2026. Reviewed branch: `Coherence_Adaptive_Repair`, HEAD `b5bbf8f`. This is planning, not a new training result. The primary agent discussed the design with the explicitly requested GPT-6.1 sol at maximum reasoning effort and separate backward/Scale auditors. Sources include the completed [Round 6 report](../../../../_Reports/Coherence_Adaptive_Repair/Round_6/report.md), [Scale supplement](../../../../_Reports/Coherence_Adaptive_Repair/Round_6/Scale_Alignment_Followup/report.md), current code and existing numerical receipts. The [evidence audit](evidence/SCALE_AND_COST_AUDIT.md) records additional arithmetic from those receipts; no new model inference was performed. The [workplan](WORKPLAN.md) makes the proposed execution bounded and reviewable.

## 1. What survives the recent failures?

**Retain LPC–SSS as the organizing framework and retain SSS-only, fresh parameter-space ConFIG with Lagrangian field fidelity as the research route.** Current evidence rejects specific learned packages, not those entire constructions. The substantial missing result is a worthwhile intervention under the current finite definitions that transfers to held-out conditions while respecting field budgets at acceptable cost.

LPC's finite-grid mixed-cumulant partition, under its stated exponential-moment assumption, organizes local, pair and collective information. State, Scale and Shape are complementary, overlapping finite views rather than that partition itself. Weighted occupation quantiles, signed band coefficients/piece means and declared excursion persistence can be built for N fields with explicit units, geometry and masks. None guarantees physical-law satisfaction or universal cross-model efficacy. Shape's raster geometry and finite filtration bank remain applicability limits. The earlier [theory/history audit](../../Round_6/evidence/HISTORICAL_AUDIT.md) and the local manuscript foundation `UpdatePlans/Coherence_Manuscript_Foundation/` give the detailed argument and prior successful instances.

Post-training remains well motivated: a capable SOURCE provides a meaningful fidelity reference, expensive relational supervision is concentrated into refinement, and endpoint constraints avoid dependence on RF, diffusion or another native training objective. Jensen-type conditional-mean examples support the nonlinear-relation motivation, but do not establish that early physics supervision is meaningless or that post-training must win every scratch comparison. Paired targets remain required; no-native-loss does not mean target-free.

The current ConFIG composition has a clear, limited meaning. Its three fresh parameter rows coordinate the three current family directions; inner-family tradeoffs are still controlled by each family definition. Field feedback, clipping, AdamW moments and finite steps require separate reasoning. A combined scalar is not the objective of this vector construction, and finite held-out feasibility does not follow from positive pre-optimizer products. The [ConFIG paper](https://arxiv.org/abs/2408.11104) establishes its own geometric/convergence results under its stated construction, not a theorem for this entire stochastic constrained application.

## 2. Training cost is an execution problem with a genuine structural component

The user is right about the principal cost, but high cost is not evidence that LPC or the scientific objective is false. Three fresh parameter gradients require three independent reverse signals through the shared model. An active weighted field-constraint sum adds another signal, not one signal per field. Batching these rows changes execution and memory, not their mathematical number.

The early complete-epoch profile measured 75.658 seconds of inclusive backward inside 109.503 seconds wall time, about 69.1%; pseudoinverse and data wait were only 0.364 and 0.786 seconds. Crucially, that profile contained 36 family backward calls and only one field-constraint backward. It does not separate descriptor adjoints, model adjoints and checkpoint recomputation, or represent mature active-constraint/complement training. The next profile must fill that specific gap.

The isolated batched family VJP comparison saved 3.61% complete epoch time while increasing allocation from about 19.02 to 37.15 GB. The selected batched/eight-group package was 0.82% slower than separate/shared execution. Its lower Shape gradient variance is useful evidence, but does not turn it into a net acceleration. Retain those historical decisions within their scope; select the Round 7 execution afresh from measured time and memory.

The proposed exact change factors the chain rule. With all model-dependent endpoints collected in Y(theta), define descriptor cotangents v_k = partial F_k / partial Y and v_c = partial(sum lambda_f e_f) / partial Y. Then

$$g_k=J_\theta(Y)^\top v_k,\qquad h=J_\theta(Y)^\top v_c,\qquad d=\operatorname{ConFIG}(g_1,g_2,g_3)/C_0+h.$$

Compute descriptor cotangents on detached leaf copies, release descriptor graphs, and send the resulting rows through the original model graph in selected chunks. ConFIG still sees the three full parameter rows. This is **not output-space ConFIG**. It lets model activation retention/checkpointing be selected independently of descriptor vectorization, and allows structurally zero branches to be skipped: SSS has no complement cotangent, whereas field feedback does. It does not promise to replace the reverse work by one ordinary backward.

Checkpointing trades saved activations for recomputation. PyTorch 2.5.1+cu121 and TorchJD 0.17.1 are installed in the project environment. Use the pinned behavior, including explicit non-reentrant checkpointing and preserved randomness; the [PyTorch 2.5 checkpoint documentation](https://docs.pytorch.org/docs/2.5/checkpoint.html) explains the recomputation and state-equivalence obligations. Do not upgrade packages as a shortcut during this round.

First test saved-graph factoring with row chunks of one or two and one profile-selected lighter checkpoint policy. If repeated model recomputation dominates and activation retention cannot fit, a single conditional member-replay variant may obtain all endpoint values first, compute cotangents, then replay one independent member at a time with its original randomness and accumulate the same parameter rows. Count the extra forward, preserve stateful buffers, and qualify actual optimizer actions. This applies to deterministic J1 as well as generative J2. It is not a changed rollout horizon, reduced ensemble or stale gradient method.

## 3. Scale has an additional, more specific problem inside TRAIN

The supplement correctly established that definition mismatch does not explain away validation deterioration: three complete definitions improve on TRAIN800 and worsen on VALIDATION100. The stored decomposition adds a more precise observation: the improved TRAIN composite already conceals regressions in joint, signed and correlation scores.

| Original epoch 500 minus SOURCE, frozen TRAIN Scale units | TRAIN800 | VALIDATION100 |
| --- | ---: | ---: |
| Complete expanded prefit Scale | -0.071984 | +0.273984 |
| Weighted joint block | +0.005252 | +0.168917 |
| Weighted signed block | +0.005484 | +0.073199 |
| Weighted log-power block | -0.083610 | +0.018834 |
| Weighted correlation block | +0.001461 | +0.014473 |
| Attraction A | -0.353160 | -0.071135 |
| Positive separation R | -0.281175 | -0.345118 |

The remaining small contribution is the frozen triple block. In both splits, the fair score is S = A - R; separation is positive and subtracted. The [evidence audit](evidence/SCALE_AND_COST_AUDIT.md) gives the exact hashes and paired intervals. These are stored-data decompositions, not new trials.

The conclusion is not that log-power is wrong or that it causes every failure. The present weighted family permits auxiliary improvements to compensate for deterioration of the full joint score. ConFIG sees that already-combined family gradient and cannot enforce improvement of its individual blocks. Parameter-gradient/action measurements are needed before making a causal competition claim.

Likewise, reduced Scale separation is not global ensemble collapse. Native physical spread increased in every field in the original run. A deterministic transform can reduce pair distances while raw field variance rises. Increasing raw variance, forcing SOURCE separation, or multiplying the whole Scale family would not isolate the observed issue. A fair score may legitimately improve with decreasing separation when attraction improves enough; separation is a diagnostic, not an objective to maximize independently.

The mature actual-Adam shadows lowered all three complete TRAIN Scale definitions in all 13 batches. Thus there is no evidence that the mature optimizer simply moves the complete TRAIN score uphill. Those local checks cannot establish generalization, and aggregate descent cannot settle the internal-block tradeoff.

## 4. One scientifically motivated Scale change

Select **Joint_Anchor_Scale** as the initial new learning candidate, against **Composite_Scale_Control**. The candidate retains the existing complete, mean-augmented signed joint embedding psi and its fair energy score:

$$F_{\rm Scale}^{\rm joint}=\kappa\left[\frac1J\sum_j\|\psi(X_j)-\psi(Z)\|_2-\frac1{J(J-1)}\sum_{j<l}\|\psi(X_j)-\psi(X_l)\|_2\right].$$

All J2 members retain gradients and the correct fair coefficient. For deterministic J1, use the existing paired norm, with no artificial ensemble or separation. Fit the single positive scale kappa once on the SOURCE TRAIN800 parameter-gradient norms to preserve the initial family magnitude; do not tune it against validation. State, Shape, C0 and the outer ConFIG/field policy remain common.

Why is this a principled simplification? The existing joint embedding contains every retained signed coefficient with a fixed modal root and every connected-piece mean with a positive fixed root. Where a modal root is positive, the coefficient is recoverable. A zero root means all relevant band filters vanish there, so the current band-derived auxiliaries ignore that mode as well. Signed projections, normalized log-power, correlation and filtered regional triples are deterministic functions of those retained coordinates and fixed geometry. Removing their **separate loss terms** does not remove that represented information.

This is not equivalence of loss, gradient, or finite-sample sensitivity. Nonlinear auxiliaries can emphasize small but useful features that a high-dimensional joint norm underweights. Neither a small joint energy score nor information retention guarantees useful auxiliary task errors or native-field reconstruction. Therefore the original complete expanded prefit Scale remains a primary outcome, alongside the joint score and every block's A/R decomposition. The candidate cannot win by changing the reporting yardstick to its new loss. Its potential reduction in descriptor work is a changed-objective cost difference, reported separately from exact execution speedups.

The study deliberately does not add SOURCE-law shrinkage, an arbitrary separation floor, another Scale weight search or inner parameter-space ConFIG with extra reverse rows. Those are different hypotheses. The observed within-family tradeoff justifies testing the simpler joint-anchor geometry first.

## 5. Preserve the spatial-feedback finding without preserving a mismatched reference

The rotating 512-point complement produced real local field benefits and remains a useful option. Both new arms use it from fresh SOURCE, with identical streams. They also share a newly fitted R0/M0 reference under the **same training observation, member/RFF, selected-plus-complement reconstruction and quadrature law**. The supplement retained a selected-support R0 while changing the quadrature; its native cached J4/T256 baseline is not that training-law expectation. A fresh study should not hide this offset.

Do not silently replace this reference by full-native cached inference: query-support invariance has not been established by matching sample IDs alone. The primary estimator is an inclusion-weighted native integral only under its declared endpoint consistency assumptions. Preserve the native held-out assessment as a separate paired observation contract and check the selected/complement/full-native endpoint consistency on a bounded fixture. SOURCE-relative feedback must be labeled by its actual law if a difference remains.

This common correction means a fresh control is scientifically necessary. The old continuation is useful historical evidence, not an age-matched control for the new reference. Both new arms begin from SOURCE with fresh optimizer/controller state. No failed child is used as the new initialization.

## 6. Decision and budget

The [workplan](WORKPLAN.md) initially allocates five GPU-hours to joint preparation, then two matched 100-epoch trajectories, and normally one selected continuation to epoch 500. The composite control may be selected if it is better; the new candidate has no entitlement to the extension. The total ceiling is **40 occupied GPU-hours on GPU2**, including failures and assessment. At historical 140–191 seconds per epoch, 600 total substantive epochs alone cost about 23.3–31.8 GPU-hours. A lower promise would assume an unmeasured speedup. Most of that expenditure is conditional after the matched review.

The user's additional instruction explicitly rejects brittle gates and premature aborts. Preparation allocations and the proposed executor screen are adjustable within the total budget. A failed memory layout, noisy pilot or missed final margin calls for diagnosis, not automatic campaign closure. Permit one evidence-led revised learning phase in place of unchanged extension, preserving the initial matched comparison and recording changed recipes/cumulative ages. Use TRAIN/action evidence for choices and keep the reporting yardstick and physical safeguards fixed; exploratory recovery does not become an isolated causal comparison merely because it improves. The workplan defines this discretion without opening an unlimited hyperparameter search.

The round can adopt an exact execution improvement without adopting a model, establish a better Scale tradeoff without claiming universal physics, or reject the tested geometry while retaining the broader framework. Final decisions report fieldwise effects, complete Scale/joint/A/R changes, State/Shape, unoptimized scientific tasks and whole cost. Physical field/member/sensor safeguards remain required for model replacement; old aggregate score margins remain visible references, not automatic reasons to erase measured component progress.

The present task did not launch training. The local device-inventory command could not communicate with the NVIDIA driver, so current GPU2 capacity/availability was not established here; this is an execution-time resource check, not evidence that the scientific plan is blocked or that GPU2 is faulty. Historical timings remain forecasts until measured on GPU2.
