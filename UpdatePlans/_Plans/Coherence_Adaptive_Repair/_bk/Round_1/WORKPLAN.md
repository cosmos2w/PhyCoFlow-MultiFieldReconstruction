> **Historical record:** Round 1 is closed. This document is preserved for reference and is not a current launch instruction.

# Coherence_Adaptive_Repair — Round 1 workplan

Date: 2026-11-13. Status: proposed execution contract, prepared from main `a18d2bd27cac0c72a902242477e98eb8f511f03f` with three GPT-6.1 Terra max reviews. This planning task creates no branch, model or GPU job. When execution is requested, start from current main and create the Git branch **`Coherence_Adaptive_Repair`**, using the owner's exact name. Physical GPU0 is primary; physical GPU1 may accelerate independent work when available. The [branch roadmap](../../../_bk/Legacy_Local_Plans/CoherenceRepair_Branch_Roadmap_20261113.md) supplies the wider program; this workplan selects its first substantive round.

## 1. Objective, scope and decisions

Implement a generic adaptive coherence engine that retains the accepted C8 core, can discover a small useful group/higher-order query bank on TRAIN, allocates complete queries using measured signal and cost, and compiles immutable observer work. Establish its mathematical contracts and exercise genuine conditional generation through both requested sources. Complete a mature FFM learning comparison and make separate decisions about representation, allocation, exact observer reuse and the integrated learned output.

The question is: **can a physically justified, sparsely enriched query representation and cost/variance-aware execution reach useful joint-state repair more efficiently than the current C8 recipe, while preserving each field's fidelity and conditional-member semantics?** This is not a claim that LPC equals three finite losses. LPC inspires their physical meanings; the round must justify the particular retained queries and simplification chain.

The four required products are:

1. A definition/code/task ledger and executable mathematical witnesses covering the C8 core, added group projections and centered third-order queries, representation sensitivity, proper conditional scoring and physical measures.
2. A full-grid EDM and selected-query FFM source/rollout contract, plus typed point/member/mean-risk primitives and a qualified opt-in mean-risk controller path. Formal learning retains the historical member-risk contract.
3. A versioned hierarchical sampler, bounded TRAIN discovery, cost/gradient-variation allocation, and exact compiled observer with matched values, gradients, coupling and resume behavior.
4. A mature C8-versus-integrated-candidate FFM comparison, a full native EDM execution qualification, and an English report with explicit adopt/reject decisions and code extraction scope.

The execution agent is encouraged to revise an ineffective design from TRAIN evidence. Section 9 defines that authority. Implementation and meaningful learning validation remain required; diagnosis alone does not close a viable route. Matrix-free spectral solvers, general 3-D persistence, residual/student distillation, a new optimizer tournament and mature law-fidelity comparisons belong to later rounds unless a narrowly justified enabling fix is necessary here.

## 2. Pin the requested sources literally

| Source | Requested directory, relative to repository root | Verified saved endpoint | Native reconstruction contract |
| --- | --- | --- | --- |
| EDM | `cases/turbulent_combustion/runs/tc_diffusion_pde_5000ep/20261027T222333Z_187a0271` | `checkpoints/last.pt`, epoch **1400**, step 88,200; SHA256 `0bcaa0cf7d1082536fea8fce698646018912d1e8b9f76004c9480ad4a9c3f753` | Conditional U-Net, complete 100×403 grid, 32-step Heun, 63 denoiser calls/member, EMA available. |
| FFM | `cases/turbulent_combustion/runs/tc_gl_rbf_cq_coverage_5000ep/20261029T210910Z_cf9c2276` | `checkpoints/last.pt`, completed epoch **5000**, step 315,000; SHA256 `2d1da713538f6e9613560f3a7eb4a9df2a95d1ad1f56dd11a1a8a4f9ef65050c` | Point-cloud conditional FFM, selected queries supported, four Euler steps, checkpointed RFF prior buffers, EMA available. |

The EDM directory name is not proof of 5,000 completed epochs. Its status is an old `running` snapshot, and the preserved last checkpoint is at 1,400. Admit it transparently as a frozen historical checkpoint; do not resume it, replace it, label it completed or use an integration-only bypass for formal work. Implement a generic frozen-checkpoint source-admission mode with a provenance receipt. At execution, perform one writer/ownership and file-integrity check, copy the pinned artifact into an immutable case-owned snapshot, verify its hash and strict-load metadata, and leave the original run untouched. A changed hash is a source-identity issue to resolve before use.

Both sources use random T-only point observations, with native counts 192–384. FFM's coverage encoding is not additional observed fields. Use matched condition IDs and one frozen T256 observation panel for evaluation; keep the native variable-count protocol for training and record its stream. No direct ranking of pretrained architectures is required. Pair SOURCE/child/control/candidate random inputs **within each model**; the two model families need not share incompatible latent spaces.

Choose one explicit weight view: restore the complete source and auxiliary EMA, materialize its configured EMA shadow as live initial weights, preserve every buffer, and freeze an identical SOURCE anchor. For new post-training children, disable child EMA and train/evaluate those same live weights. Implement this as a tested initialization policy, not a flag change that prevents loading the stored EMA or silently changes the initial function. Hash the resulting parameter/buffer view and record zero-update SOURCE/child parity. This is a new child from a selected function, not continuation of the old pretraining optimizer.

Historical Round 8 C8 used raw FFM weights and two Euler steps. Its outcomes remain prior evidence. The new **C8 control recipe** uses the source view and native four-step solver fixed above; regenerate its endpoints and timings. Do not present the old alpha=0.5 endpoint as a matched new baseline.

## 3. Scientific target and generative fidelity

Use C for fields, Q for scoring sites, J for members, and M for retained spectral modes. Follow the [foundation](../../../../Coherence_Manuscript_Foundation/README.md) and [adaptive design](ADAPTIVE_DESIGN.md) for equations and implementation details.

Retain the C8 State core with all marginals and its eight fixed hard/diverse directions, complete signed joint Scale with piece means, complete log-power-self anchors, signed/rho additive leaves, and pooled-TRAIN unary/pair Shape semantics. Do not sample inside a complete nonlinear State or Scale norm.

Use the qualified combustion geometry as the starting case contract: ordered fields CH4/CO/T/U_1/p; declared native physical nodal masses; FEM/raw-coordinate Scale with up to 160 modes, wavelengths 1.5/0.45/0.15 and minimum 0.1 in recorded raw coordinate units, a 0.01 model-unit field floor, float64 spectral reductions, joint weight 0.25 and mean weight one; Shape uses the nonperiodic 32×128 physical-overlap raster, 32 diagram projections and explicit finite/essential floors. These are case choices, not generic defaults or known SI units. Cross-check the archived Round 8 resolved configuration and current implementation, then freeze the complete resolved case configuration before comparison. Refit source-dependent scales, direction hardness and calibration on the new declared TRAIN/source view; do not copy fitted coefficients from the old raw/two-step endpoint or silently substitute the generic square-raster example.

The candidate adds at most four distinct signed group/all-field projection leaves to State and at most eight regional centered field-band triple leaves to Scale. Each is a **complete fair score**, additive outside the existing core norms. A bounded TRAIN/SOURCE discovery stage chooses and freezes the bank, units and initial weights before formal learning. Existing third-order features provide same-node filtered products; offset/lag triples and higher-order persistence are not required. The new bank defines a changed finite target, not an unbiased estimate of every possible higher-order relationship. Selection must show incremental sensitivity or practical task relevance beyond the existing full joint anchor; parity controls alone do not prove that anchor is inadequate.

For a fixed represented metric d and conditionally IID members independent of truth given observations,

\[
\widehat S_J=\frac1J\sum_j d(X_j,Z)-\frac1{J(J-1)}\sum_{j<k}d(X_j,X_k).
\]

Use J2 in training, with both member gradients. Use J4 for fixed SOURCE/final assessment outputs, extending common scoring to all supplied members. Negative type supports propriety; strict represented-law identification needs strong negative type and a distinguishing representation. An arbitrary composed trainer inherits no universal propriety/convergence theorem. In a Hilbert representation, squaring the distance scores only the mean. Include the scalar J2 flat-interval witness and the distinction between spatial occupation, dataset mixture and conditional law.

Keep the formal per-field bound `physical_member_mse <= 1.10 * SOURCE_member_mse`, plus sensor consistency. Adoption computes this guard on decoded native-support quadrature over the selected condition panel, with matched SOURCE/child members. Label decoded units unknown when the dataset does not declare them. The online Q4096/model-unit, inclusion-corrected TRAIN risk is a separate stochastic support proxy, not exact native risk conditional on one fixed spatial sample. This operational bound can suppress legitimate conditional spread; do not call it neutral to law recovery. Implement and qualify a separate signed mean-risk primitive:

\[
\widehat R_{\rm mean}=\|\bar X-Z\|^2_{\mathbf M}-\frac1{J(J-1)}\sum_j\|X_j-\bar X\|^2_{\mathbf M}
=\widehat R_{\rm member}-\widehat V_X.
\]

For J2 this is the physical inner product of the two member errors. Preserve its sign and both gradients. Expose an opt-in controller using a SOURCE-relative mean-risk **difference** and a declared absolute allowance in field-risk units, with a positive independent optimization scale. Do not divide by a noisy signed SOURCE estimate, clamp individual estimates or manufacture allowance through a floor. Fit/qualify its reference on TRAIN and test its actual lifecycle. It is not active in the mature C8/candidate comparison, and this round cannot claim a learned advantage from changing fidelity contracts.

Separate SOURCE-risk fitting from C0/coherence-amplitude fitting. Replace the implicit 32-condition risk cap with an explicit approximately 10% TRAIN reference, normally 800 of TRAIN8000. Record each population, support, draw set and units independently. Use the same risk reference for both FFM arms, and the same C0 fitted from the SOURCE **C8-core** observer and fixed J2 TRAIN panel/support. Do not recompute A1's C0 from the expanded loss. Added-block coefficients have their own versioned calibration. Artifact fitting, discovery and model-evaluation diagnostics use the declared approximately 10% TRAIN panel; stream/microbatch rather than silently shrink the population. Local derivative microcases are numerical qualification, not population performance estimates.

The theoretical deliverable must calculate the actual retained-Scale conditioning constants/unresolved residual and a physical-task sensitivity or margin bound. It should explain which simplifications preserve a declared task, with no blanket sufficiency assertion. A privileged output perturbation is a diagnostic, not an inference input or deployable correction.

## 4. Implement the adaptive computation package

The current C8 sampler already mixes uniform exploration with EMA difficulty. The candidate must therefore add meaningful hierarchy, signal/cost estimates and execution reuse.

| Component | Required behavior |
| --- | --- |
| Query representation | Stable versioned IDs for unary fields, pairs, discovered group projections and sparse triple/band/region queries; physical measure, scale, target weight and actual selection probability travel with the query. |
| Discovery | Bounded physical/data-driven groups plus reproducible exploration; frozen TRAIN-only bank; no exhaustive C-choose-3 catalogue or exponentially many filtrations. |
| Hierarchical proposal | Known probability for every eligible leaf, a positive uniform component, and correct treatment of multiple paths, duplicate IDs and Shape inner draws. |
| Allocation | Measured incremental bundle cost and lagged gradient-variation/progress evidence; bounded sparse state and at least one valid estimator for each active stratum. Full gradients drive learning. |
| Exact execution | Reuse reference transforms, endpoint features and complete anchors, and deduplicate fixed observer queries while preserving weights/multiplicities. |
| Lifecycle | Commit policy/calibration/cursors only with accepted updates; checkpoint all scientific identities and random streams; exact unchanged resume and fail-fast stale artifacts. |

Use a 25% uniform proposal floor initially and refresh statistics every ten completed epochs. Treat these as preformal starting choices, not universal constants. Keep policy overhead at or below 5% of cumulative training occupied time; bound/disable an ineffective proxy before it consumes the mature comparison. A cheap representation-gradient sketch may guide priorities, but it is not a parameter-gradient variance measurement. Sparse full-parameter or Jacobian-aware audits must validate the claimed relationship and include their cost.

For fixed additive leaves, importance correction estimates the target before nonlinear aggregation. Maintain exact complete anchors, conditional member independence and actual outer/inner probabilities. If a cyclic pair is used, include it as an exact outer stratum and sample the complement correctly; never add a cyclic score to an already complete importance estimate. No unbiasedness claim extends automatically through ConFIG, clipping, rejection or AdamW.

Compile the common frozen observer and the factored differentiable training kernel. Reuse reference quantile sorts/cumulative masses, Scale coefficients/piece means, and persistence diagrams. Construct merged CDF breakpoints for the actual endpoints. Compute endpoint features once and each complete energy anchor once per condition/ensemble. Preserve mutation/stale-cache protection. Do not cache live outputs or graphs across optimizer updates. Measure cold setup, warm callbacks, CPU pairing, transfer, memory and disk cost.

## 5. Make EDM support real

EDM requires the complete logical grid. Fix the actual chain: source admission/weight view, full-grid observation batch, source prefit and geometry calibration, differentiable reconstruction, gather to a declared scoring support, exact shared scoring, constraint update, checkpoint/resume and assessment. Current `scalable_prefit.py` rejects structured sources; `_fit_training_references` also guards full-grid calibration. Removing one exception does not implement the chain.

Use complete 40,300-site model execution, then gather Q4096 scoring sites only through a qualified frozen native-basis/support bridge, or score the native support when that is the justified route. Record those as different support observables. A complete model rollout is not made cheap by scoring 4,096 sites. The current fixed Shape raster remains a declared 2-D physical-overlap target.

Native EDM already hard-clamps observations. The generic differentiable path can ignore the solver label and apply extra endpoint smoothing. Add an explicit native-hook/hard-clamp-only path and verify endpoint and gradient parity with 32-step native EDM on identical noise and sensors. Likewise qualify native FFM Euler and clamping semantics. Do not shorten solvers, lower J, detach early steps or switch sources to obtain a convenient result.

If the full native graph exceeds memory, implement exact segmented activation recomputation or staged observer-vector/rollout-Jacobian backpropagation with explicit every-member/parameter-gradient parity. Random states, dropout/mode, parameter identity and buffers must be unchanged during recomputation. Numerical dtype tolerance is declared; no truncated-gradient approximation is admitted under an “exact” label.

EDM executes a **12-reporting-epoch qualification**, one accepted B1/J2 update per epoch, crossing calibration freeze and policy refresh. Exercise uninterrupted versus pause-at-9/resume-to-12 equivalence, and evaluate both risk primitives/gradients on the same native generated members. Additional tiny opt-in mean-controller qualification is permitted within its allocation. These are execution/numerical products, **not EDM convergence or efficacy evidence**. Report actual accepted updates, visits and recomputation cost; the reporting-epoch size differs explicitly from FFM.

## 6. Evidence design and mature learning

### 6.1 Frozen-endpoint qualification on both sources

Generate immutable matched SOURCE endpoints on approximately 10% TRAIN IDs with fixed observation/member manifests. Use J2 subsets of the J4 TRAIN bank for estimator tests. Assess the C8 core and expanded target through the same compiled representation for each source. Within a source, uniform and adaptive estimators must target **the identical frozen expanded objective** and use equal charged compute or an error–cost frontier. Discovery, policy selection and redesign never use VALIDATION endpoints. Generate/unseal VALIDATION endpoints only for the final frozen assessment procedure.

Compare estimator means and every-member gradients with exhaustive small-bank/reference calculations, plus independently replicated query draws at fixed model/optimizer state. Measure gradient-estimation mean-squared error at equal total time, or MSE times cost per independent estimator. Dividing variance by elapsed time is not the relevant efficiency criterion. Separate output/representation sketches from parameter-space audits. Charge screening, probes, policy fitting and refresh.

Demonstrate bounded policy state for C=1/2/5/20/100 and implicit index/memory behavior at C=1000 without claiming learned large-system transfer. Include unequal masses, masks, missing/inactive groups, duplicate queries, sign-equivalent projections, correlated inner draws, zero-distance subgradients, small populations and accepted/rejected update restoration. Fixed source outputs establish computation and sensitivity, not learned repair.

### 6.2 Two matched FFM trajectories

| Arm | Scientific target | Compute policy | Horizon |
| --- | --- | --- | --- |
| `FFM_B1` | Accepted C8 core | Existing C8 difficulty/uniform policy | 1000 reporting epochs |
| `FFM_A1` | C8 core plus the frozen discovered group/triple bank | Qualified hierarchical signal/cost policy, or the explicitly chosen preformal uniform fallback for that same expanded target | 1000 reporting epochs |

Both use the same EMA-materialized FFM source, four Euler steps, J2, observations/prior stream, Q4096 physical scoring, existing three-family ConFIG, per-field member-risk bound, optimizer and calibration schedule. Start with AdamW learning rate 5e-5 and the retained C8 scheduler; freeze the fully resolved schedule before either formal arm. Do not quietly restart dual/optimizer state when adapting query priorities.

Use the existing C8 reporting-epoch convention: 13 accepted B32 updates per epoch, 13,000 updates and approximately 416,000 condition visits per arm, around 52 TRAIN8000 pass-equivalents. Record actual short-batch/stream counts. This is not 1,000 complete dataset passes. Freeze identical batch/sensor/member schedules independently of the query RNGs. Preserve ten-epoch internal calibration; the candidate has separately versioned added semantic blocks and coefficients.

This comparison estimates the **integrated package effect**. Fixed-target tests isolate allocation behavior; the mature arms do not causally separate representation enrichment, allocation and compilation. No mature six-arm factorial or mature EDM comparison is promised. A positive FFM result plus EDM execution qualification establishes one learned result and a second generative backend, not cross-physics or EDM efficacy.

Save TRAIN core common-observer readings and checkpoints at epochs 0/100/250/500/750/1000, with full training histories/curves. After both final candidates and checkpoint-wise TRAIN alpha rules are frozen, score the saved checkpoints retrospectively on the fixed VALIDATION core/field panel for the learning/time-to-core-quality curves. The reserved physical task is scored only on the final frozen epoch-1000 output policies, not at intermediate checkpoints or for their selection. Charge all retrospective inference/scoring. A common compiled assessment can be shared by both arms; accounting must show whether its cost is shared research infrastructure or per-method repeated deployment cost. Both learning arms receive the exact engine improvements; do not disable correct code in the control merely to manufacture a speed ratio.

### 6.3 Independent scientific assessment

Use reproducibly selected approximately TRAIN800 and VALIDATION100 samples, matched across models/arms; leave TEST locked. Full native spatial support inside those selected samples is allowed and is distinct from full dataset-population evaluation. The dataset fractions must be recomputed from actual splits. Reuse fixed outputs for derived metrics and figures.

Primary independent task: native physical area fraction of the co-excursion T > TRAIN q75, CO > TRAIN q90, CH4 < TRAIN q25, with thresholds fitted from the declared physical-measure TRAIN truths. Report unary areas, event support and unnormalized spatial first moments so an empty event is meaningful. Its fair scalar area CRPS with J4 is the primary represented-law witness; paired overlap/missing/excess areas are separate realization diagnostics. Do not call this a reaction-rate or heat-release measure. If physical units are absent, label model/recorded units explicitly.

Freeze this task outside discovery, gradients, priority estimation and candidate/alpha selection. A predefined regional signed CO/T relation can be a secondary assessment query; it cannot be promoted after seeing results. If the proposed event is degenerate in TRAIN, replace/adjust it from TRAIN support evidence before freezing the formal manifest and preserve the decision. A genuinely rare event is not repaired by choosing an attractive validation threshold.

The common fixed observer contains all C8 channels and **complete** Scale leaves/anchors for the five-field case, plus fixed Shape queries/raster and separate expanded-bank results. Document any sampled Shape estimator and uncertainty. Extend the observer to J4 without using two members under a J4 label. Use common unit internal coefficients, frozen positive SOURCE denominators and the same bank for every endpoint within a comparison. Source-specific training banks can differ; architecture rankings are not the claim.

Report raw and TRAIN-qualified anchored outputs. Retain fixed alpha=0.5 as a declared comparator when TRAIN-feasible. Fit one global coefficient using a finite predeclared TRAIN risk-feasible frontier and complete core scores, never the reserved task; use the exact member-risk quadratic and score actual blends. Freeze the frontier, utility and tie-break rule before formal learning; do not claim continuous optimality. Apply the same procedure to B1/A1, freeze nominations before final VALIDATION assessment, and include both-model inference cost. Anchoring does not establish that the raw learned child qualified.

## 7. Resource forecast, horizons and monitoring

**Proposed Round 1 ceiling: 80 additive occupied GPU-hours**, covering both devices, failed qualifications, pilots, fitting, source inference, recomputation, callbacks, final scoring and closeout. This is a new proposed round budget, not inherited branch-wide authorization. Execution begins only when requested under the agreed contract.

| Work | Initial planning allowance |
| --- | ---: |
| Two mature FFM arms, scheduled callbacks included | 38 GPU-hours |
| Native EDM execution/gradient/resume qualification | 8 GPU-hours |
| Common SOURCE banks and final native scoring | 12 GPU-hours |
| Preflight, discovery, estimator/policy profiling | 6 GPU-hours |
| Contingency/redesign and closeout | 16 GPU-hours |
| **Total** | **80 GPU-hours** |

At least six hours of the reserve remain protected for final scoring, figures, integrity checks and clean completion. The remaining ten can fund one bounded redesign/retry. These allocations can move within the total while retaining completion obligations. Historical two-step FFM timing is only a reference: the native four-step path and extra probes need measured complete-update/callback timings. Native EDM training timings are not a differentiable 63-call rollout forecast.

Before mature launch, measure representative complete native J2 updates, cold fitting/compilation, peak memory and callbacks, then forecast both fixed horizons, source/assessment cost and reserve. Use a conservative upper observed steady-time estimate. Resolve a shortfall through exact reuse, microbatch/accumulation, faithful recomputation, reduced redundant callbacks/inference and bounded policy cost without changing the scientific evaluation coverage or required curve cadence. The prescribed observer checkpoints remain. Do not silently reduce horizon, source solver, member count, valid support or final evidence. If a genuine shortfall persists after those fixes, report the concrete cost and preserve the qualified state rather than exceed the cap.

GPU0 is the principal device for preflight, native same-device benchmarks and one mature arm. GPU1 may run the other mature arm or independent cached scoring/EDM qualification after ownership/UUID/idleness checks. Parallel execution reduces elapsed time, not occupied time. If GPU1 is busy, use GPU0 sequentially and update the wall-time forecast. Preserve every unrelated job and original checkpoint. A device-specific writable KeOps cache is set before imports.

Follow [AGENTS.md monitoring policy](../../../../../AGENTS.md). Name the executing `gpt-6-luna`/low monitor **`car_round1_luna_monitor`**. Its launch receipt records run IDs, device UUIDs, estimated completion and scheduled checks: one permitted startup verification, calibration/qualification completion where actionable, mature epoch 500, epoch 1000/completion, and forecast-based budget warnings. It waits until those estimated times and pushes actionable messages. The primary never repeatedly polls runs or the monitor. If the required GPU0 remains unavailable for 30 continuous minutes, stop owned waiters, preserve pending/resume state and report the resource block; do not displace a job.

Loss curves first appear with history and refresh every ten completed epochs, validation, pause and completion, including resumed and qualification stages. Set `runtime.plot_every_steps: 10`. Keep complete `logs/` histories and readable family/component PNG/PDF curves with individual scales and modest-change visibility. Monitoring cadence is unrelated to scientific logging cadence.

## 8. Acceptance, rejection and claim boundaries

Freeze the metric manifest and numerical rules after TRAIN qualification and before VALIDATION scoring. Numerical gates qualify infrastructure; learned gates decide the package. Suggested starting rules below are part of the proposed contract and may be made stricter from physical TRAIN tolerance evidence before that freeze, never relaxed from assessment outcomes.

| Decision | Rule |
| --- | --- |
| Exact observer/kernel adoption | Per-sample/block values and every-member/parameter gradients agree within justified dtype tolerances; stale artifacts fail; full callback/total cost improves or a supported model path is added. Record cold/warm storage/memory. |
| Adaptive allocator adoption | Correct fixed-target expectations; at least 20% lower gradient MSE–cost product, or corresponding error reduction at equal total cost, on both source endpoint families without unstable tails or >5% policy overhead. A narrower one-backend result is labeled as such. |
| Learned output eligibility | Every physical member-risk field <=1.10 SOURCE and sensor parity passes. Core State/Shape may worsen by at most 0.02 positive SOURCE score units versus B1; complete Scale at most 0.005. Event-area CRPS may worsen by at most 0.05 SOURCE event-score units. A near-zero denominator requires a frozen absolute tolerance. |
| Efficiency route | At least 20% less total occupied time to the same preregistered core/field-quality level, with the final epoch-1000 output preserving independent-task utility and all guards. If either method never reaches that level, do not invent a time-to-quality ratio. This is time-to-core-quality, not a claim about the unmeasured intermediate physical task. |
| Quality route | At least 0.05 SOURCE event-score units of independent-event gain over B1, while preserving every guard and costing no more than 10% extra total occupied time. Report tradeoffs, raw/blended identities and inference latency. |

For retrospective time-to-core-quality, require at least 1% complete Scale and 5% fair Shape gain versus SOURCE, no State regression beyond 0.02 SOURCE units, and native field eligibility at two consecutive scheduled VALIDATION core checkpoints. All checkpoints, coefficient-selection rules and the final output are already fixed from TRAIN before these scores are computed. Use the second checkpoint's charged cumulative time; do not use the reserved event to select or qualify intermediate outputs. Report the final epoch-1000 independent-task decision separately. This descriptive fixed-trajectory crossing statistic does not select a new deployment checkpoint or supply independent post-selection confirmation. Cost includes per-method setup/probes/callbacks and a declared allocation of common infrastructure. Report training, scientific scoring and deployment cost separately; no unspecified amortization.

A candidate may qualify as a numerical infrastructure improvement while its learned output is rejected. If adaptive priorities fail but a fixed uniform expanded-bank method succeeds, adopt only that representation package and state that allocation failed. If the independent task does not improve over SOURCE, an efficiency result is not a new scientific-repair claim. Only a separately mature mean-risk study can support improved learned law-fidelity control. Native EDM qualification cannot support EDM efficacy, deterministic transfer or a different physical-system claim.

Retain the owner-selected C8 recommendation when neither learned route earns adoption. Extract qualified exact reuse/EDM/contract capabilities as optional generic modules where appropriate; do not change main by default during this research round.

## 9. Authorized reasoning and adjustment during execution

The agent must think, compare predictions with observations and fix setbacks. It may independently improve kernels/cache ownership, correct probability algebra, change microbatch/recomputation mechanics with parity, refine TRAIN grouping/discovery, tune preformal proposal statistics, and rebalance phase allocations within 80 hours. It must explain the evidence and update design/receipts rather than follow a failing draft mechanically.

Before formal freeze, discovery/policy options can be compared on fixed TRAIN endpoints within the six-hour profiling allowance and flexible reserve. Choose one integrated candidate, preserve rejected attempts and forecast the full remaining work. Do not impose an arbitrary attempt-count gate that prevents fixing a concrete defect within budget.

After freeze, routine priority/cost EMA refreshes are allowed by the checkpointed policy. One material TRAIN-only redesign is allowed at or before candidate epoch 250, using at most 250 disposable candidate epochs and the flexible reserve. If it changes bank, target, coefficients/normalization rule or update policy, create a new identity and restart the final 1,000-epoch candidate from the pinned source; preserve the pilot and charge it. An actual code defect invalidates affected training from its first impact. Do not splice incompatible histories or claim a 750-epoch replacement completed the final horizon.

The control continues unchanged unless a shared correctness fix invalidates both arms; in that case audit/replay scope and cost explicitly. No redesign may use VALIDATION/TEST or the reserved physical witness. Final candidate, alpha rule and assessment criteria are frozen before held-out scores. Do not add mature arms, change sources/solvers, waive scientific guards, access TEST or exceed the cap without new authorization. If an exact path is genuinely infeasible, close the specific infeasibility with evidence and continue other required deliverables within budget; never turn the remaining round into diagnosis only.

## 10. Delivery

Place experiment payloads under `cases/turbulent_combustion/runs/coherence_adaptive_repair/round_1/`, with immutable source, bank, coupling, budget, launch and completion receipts. Keep campaign planning/evidence under this directory and generic code/tests on `Coherence_Adaptive_Repair`. Preserve original runs, archives and local user changes. At execution start inspect status, synchronize main with `git pull --ff-only` when needed, create the exact named branch, and record the actual base; if main advanced, audit affected contracts before launch.

Deliver qualified generic code with appropriate focused tests, audited commits/pushes on the non-main branch, readable loss histories, and an English report. The report contains the mathematical definition/code ledger, source/EMA/solver identities, discovered banks and rejected alternatives, fixed-target estimator evidence, EDM qualification boundary, mature FFM trajectories, native scientific/field/spread results, raw/anchored decisions, full resource accounting and independent adopt/reject decisions. Formal scientific figures retain PDF masters and relative PNG previews with link/decode/render checks; every reconstruction panel carries its computed fieldwise error on the plotted physical mask/units.

Do not create a merge PR or merge to main merely because this workplan finishes. State which capabilities are ready for main, which remain opt-in, and whether the candidate output should replace the current recipe for this case. The round ends with those decisions and preserved evidence.
