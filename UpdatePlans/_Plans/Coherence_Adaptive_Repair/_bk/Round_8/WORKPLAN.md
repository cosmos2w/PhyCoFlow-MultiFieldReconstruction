# Round 8: coherent fidelity, spatial observations and scientific evaluation

26 November 2026. Prospective execution plan, prepared before Round 7 closeout. This planning task launched no training. Read [PROJECT_REVIEW.md](PROJECT_REVIEW.md) for the scientific reasoning and [README.md](README.md) for evidence and the later reconciliation point. Do not alter the active Round 7 campaign or apply new criteria to its original verdict.

## 1. Intended outcome, scope and authority

Deliver a tested general endpoint-to-SSS-to-field-feedback path with explicit member-MSE retention, genuine deterministic J1 and independent generative JN support; one inexpensive positional Shape observation; and cached quantitative/visual evidence separating field accuracy, relationship discrepancy and conditional feature scores. Retain the SSS-only, fresh parameter-space ConFIG plus multiplier-feedback mainline. No model-native training loss, teacher/student route, new architecture, new dataset campaign, deployment benchmark or broad coefficient search is required.

The **default completion is an engineering and cached-evidence round**, not a promise to train. Formal learning is conditionally included when it answers the concrete question in Section 8. Complete useful independent work even if one optional feature or learned recipe is rejected. The executor may correct diagnosed failures, simplify implementations, substitute a better justified narrow variant and reallocate budget without another approval loop. Record the reason, changed scientific definition and affected comparison before evaluating the variant. Preserve historical artifacts and all user/concurrent work.

**Resource proposal:** at most **6 occupied GPU-hours** for preparation, qualifications and any missing admitted prediction banks. If a useful formal learning question remains, the same round may expand to **24 occupied GPU-hours total**, including the first six, all failures, assessment and reserve. This is a ceiling for a later execution instruction, not a command to spend it. GPU2 is the primary runtime device; select by runtime configuration, check ownership/capacity and record actual hardware. No physical index/UUID is a scientific gate. Matched speed comparisons use the same device. No other GPU is needed by default.

Keep approximately 10% matched conditions per required split: reuse TRAIN800 for calibration/transfer questions and VALIDATION100 for development assessment. TEST stays locked. Do not expand to full populations. Complete histories and first/every-ten-epoch/review/pause/final curves are mandatory for any learning used as evidence, including a pilot. All schedules below are reporting epochs.

## 2. Freeze the question, not unnecessary implementation details

Keep immutable SOURCE/C8 identities, physical measures/masks, field order, observation law, independent member law and endpoint live/EMA choice explicit. Reuse valid Round 7 references and execution qualifications only when their scientific content matches the new path. A changed risk or observation begins a named new protocol/phase; it is not an ordinary resume. No migration of old risk columns or historical verdicts is authorized.

After Round 7 closes, make one short reconciliation record containing its selected endpoint, field/direct/fair/task results, cost and retained executor. Its terminal report is not a dependency for reading code, testing generic risk arithmetic, building cached direct readouts or implementing the optional Shape feature. Before any new formal comparison, choose one common frozen Scale definition from the evidence: retain the Round 7 selected definition if its final evidence supports it; otherwise use the explicit existing definition that answers the next question, with rationale. Do not infer joint-anchor benefit from epoch-100 point differences with intervals spanning zero, compare Joint500 against Composite100 as a matched treatment effect, or begin a third Scale-objective tournament.

Use descriptive paths/names, for example `MemberMSE_BaseShape`, `MemberMSE_SpatialShape`, `Cached_Contract_Reassessment`, `Generic_Risk_Qualification` and `Spatial_Shape_Qualification`. Historical identifiers are preserved. The `round8_lazy_pooled_fair_shape_v1` string already present in current source predates this plan; it is not evidence that this round's proposal exists and should not be renamed for cosmetics.

## 3. Scientific contracts

### 3.1 Endpoint and field fidelity

An endpoint provider supplies the declared physical fields, coordinates, mask/masses and either one deterministic output or independent conditional members. Models remain free to use different native training objectives; this workflow does not request them. Observation consistency, solver and latent randomness are part of the endpoint law. Cases specify observed-field roles and units; shared code must not infer temperature from the literal string `T`.

For field f and condition c use the physical member risk

\[
\widehat M_{cf}=\frac1J\sum_{j=1}^J\frac{\sum_i w_i(X_{cjif}-Z_{cif})^2}{W},\quad W=\sum_iw_i,
\]

then pool conditions with declared weights, giving each state equal weight when that is the protocol. Average repeated masks/members within the state first. The primary bound on each split/protocol D is

\[
G_{f,D}(\theta)=M_{f,D}(\theta)-(1+\epsilon_f)M_{f,D}(\mathrm{SOURCE})-\delta_f\le0,
\quad e_{f,D}=G_{f,D}/s_f,
\quad s_f=\max(M_{f,D}(\mathrm{SOURCE}),s_{\min,f})>0.
\]

Default `epsilon_f=0.10` and `delta_f=0`. A fieldwise physical absolute tolerance may be specified for a real near-zero-risk problem, but must not be smuggled in through the normalization floor. Record squared units and distinguish MSE from RMSE allowances. Use ratios of pooled risks for display, normalized calibrated risk differences for training feedback, and paired absolute margins for assessment uncertainty. The TRAIN baseline calibrates training only; VALIDATION uses its own same-law SOURCE baseline. This aligns the retention meaning, without claiming the same numerical absolute threshold transports across distributions.

Default training uses one frozen approximately-10% TRAIN SOURCE calibration under the actual training observation/member/spatial law. The existing Round 7 calibration already records member MSE and may be reusable if that law is retained; changing the protected risk alone does not force new SOURCE inference. Finite calibration error is reported, not denied. A paired online SOURCE estimator is allowed only when a measured reference-variance problem justifies its additional forward cost. Do not require it automatically, divide by a random batch SOURCE risk or differentiate SOURCE.

Preserve signed population-mean-risk, finite-K mean-output error and model spread as distinct diagnostics where available. The default protects average generated-field error; it does not promise calibration or retention of every draw. For an application explicitly delivering a K-member mean, that mean-output risk is a legitimate separate prospective contract, with fixed K. Do not combine these alternatives into extra simultaneous primary constraints without a task-specific need.

### 3.2 Spatial sampling and SSS geometry

Separate `SSS_geometry_support`, `field_risk_quadrature`, `descriptor_query_law`, `observation_law` and `member_law`. Retain the existing fixed Q4096 Scale/Shape geometry initially. General field-risk sampling has two ordinary implementations: use a native output already available, or integrate sampled query outputs with their actual probabilities and native total mass. For reuse of selected sites S and a fresh uniform complement C of size c,

\[
\widehat M_f=\frac1{WJ}\sum_j\left[\sum_{i\in S}w_i e_{jif}^2+\frac{n-|S|}{c}\sum_{i\in C}w_i e_{jif}^2\right].
\]

This is unbiased in the sampling design for fixed endpoint values. A wholly fresh uniform risk subset uses inclusion probability `q/n`; nonuniform/with-replacement designs supply their actual probabilities and normalization. Do not self-normalize by random sampled mass while claiming native-risk unbiasedness. Preserve each member's random field and original observations across support calls. A query-dependent architecture can change values with the query set; qualify the affected provider and report that law rather than treating one fixture as a general theorem.

Remove literal Q4096/J2 restrictions from generic risk integration, while keeping the combustion case's chosen counts in its config. J1 uses ordinary physical MSE directly. J>=2 can additionally compute the signed correction. Do not implement deterministic support by duplicating outputs. Do not change nonlinear SSS geometry by simply enabling `random_per_sample`; current cached operators would no longer match. A support library/native-basis redesign is deferred unless Stage 1 identifies a material omission and a narrow correction fits the remaining preparation allowance.

### 3.3 Scoring, Scale and family balance

For J>=2 keep regular fair scoring `S=A-R`, with `A=mean_j d(X_j,Z)` and `R=sum_(j<l)d(X_j,X_l)/[J(J-1)]`. Both pair endpoints are live. Genuine J1 uses its paired distance. Preserve the exact State quantile integration and current PH/SW1 definitions. Keep complete training-aligned Scale, historical published Scale and raw joint Scale separately named if more than one is shown. Cache transforms and expose existing A/R outputs; do not generate predictions for each metric.

Within-family weights define the scientific observable and gradient direction. Keep them positive, fixed and TRAIN-calibrated where needed; disclose their units and query inclusion weights. Equal numerical loss values are not a goal. Continue three fresh parameter-space rows and the pinned ConFIG implementation. One weighted sum of current field violations supplies the field row, followed by the recorded clipping/AdamW/PI rules. No new scalar total loss should be presented as the optimized objective. Reuse one weighted field VJP, not one per field.

Keep the current fair coefficient at one. An optional cached J4-to-J2 subset/cancellation readout can determine whether a later targeted gradient check is informative; it is not a mandatory gate. AIFS almost-fair scoring changes the target and is not adopted from analogy alone. Preserve current sampled-query stochasticity/epoch exposure unless a specifically named estimator experiment is warranted.

### 3.4 Positional Shape candidate

Expose the scalar filtrations already produced by `LazyPooledShape._diagram_bank`. For fixed TRAIN thresholds and temperature, construct

\[
B_{q,t,i}=\sigma((\tau_{q,t}-F_q(X)_i)/T_q),\qquad H_{q,k,t}=\frac{\sum_i a_i\chi_k(r_i)B_{q,t,i}}{\sum_i a_i\chi_k(r_i)}.
\]

Here a is valid raster physical mass and chi is a smooth, positive geometry-only window. Start with eight overlapping windows and three fixed thresholds per declared filtration category, using the concrete initial recipe and stable window calculation in [the support audit, Section 6](evidence/SUPPORT_AND_SHAPE_AUDIT.md#6-concrete-minimal-location-aware-representation). Thresholds use pooled TRAIN reference filtration quantiles 0.2/0.5/0.8; temperature starts at one quarter of its 0.9–0.1 span, with minimum 0.10 in standardized filtration units. Use actual physical cell centers, a declared axis metric and periodic windows where required. Empty geometric windows are removed and recorded; near-empty excursions remain finite because the denominator is geometry mass, not excursion mass.

The location distance is `||H(X)-H(Z)||_2/sqrt(dim(H))`, implemented with a finite zero-distance subgradient. Use the existing fixed positive PH normalization unit, `u_PH=1` in the current normalized Shape units; do not estimate it from possibly zero PH discrepancy. Calibrate positive `u_loc` from pooled SOURCE-reference location distances on the existing TRAIN panel. Use the declared unary/mutual query target weights after the current PH leaf/category normalization, rather than adaptive draw frequencies or signed fair-score values. The proposed combined distance is

\[
d_{\mathrm{Shape,new}}=\tfrac12\left[d_{\mathrm{PH}}+(u_{\mathrm{PH}}/u_{\mathrm{loc}})d_{\mathrm{loc}}\right].
\]

Apply the ordinary paired/fair scorer to this fixed positive sum and keep its two components visible. This initial unit matching is a scientific design choice, not a claim of equal gradient effects. Record row norms and actual actions. Permit one bounded TRAIN-only threshold/temperature/window correction for demonstrated saturation or positional blindness. A zero location calibration scale is a reason to repair sensitivity or retain PH alone, not use a huge arbitrary multiplier or terminate the round. Any later change to relative weight must be motivated, frozen and described as part of the candidate package, without a VALIDATION weight sweep.

Reuse the filtration tensor and precomputed window masses; contract in threshold chunks without creating a batch×query×threshold×window×pixel tensor. No additional PH call, field-pair enumeration, fourth family or family VJP is needed. The new descriptor still has arithmetic/memory cost, which must be measured. General N-field unary/pair sampling remains unchanged. This is a coarse raster-position observation, not native topology, injective spatial identification or a full N-way statistic. Existing scientific task regions/thresholds must remain outside this optimized feature definition.

## 4. Stage 1: cached reassessment and focused contract audit

Finish this stage without model inference where compatible banks exist. Start with SOURCE and sealed Round 7 Joint100; reuse Composite100 or one Round 6 endpoint only for a specific mechanism comparison. Incorporate the Round 7 terminal nominated endpoint once sealed. Do not conduct an unrestricted checkpoint search.

1. Reproduce the old signed contract and new prospective pooled member-MSE table from saved quantities. Show point ratios, paired margins, saved intervals, mean-output risk/spread and actual observation/unit laws. Verify member versus mean reduction, final partial-batch weighting, affine unit conversion and split-specific SOURCE identity. Preserve old verdicts in a separate column. Investigate the large native TRAIN/VALIDATION SOURCE-risk gap through existing metadata and cached field/condition strata; do not infer its cause from the ratio or launch a data-curation project.
2. Add direct A, R and S for current State/Scale/Shape. Reuse `car_round7_scale_decompose.py`, existing rho/power calculations and compiled Shape components. State gradient probes are not direct distances. Display actual signed pair/band alignment errors and low-energy coverage; keep current fixed floors and weights. Do not change training weights according to held-out diagnostic denominators.
3. Compare native, fixed-support and complement contributions to field error from the same cached native endpoints. Compare sparse-to-raster and native-to-raster reference/prediction on the same output grid using raster errors, initially without extra PH. Complete these independent discrepancies first; add location features after freezing their TRAIN recipe. Disclose that slicing a native prediction need not reproduce a separately queried model endpoint. Determine whether the difference changes a practical ranking or localization interpretation; numerical difference alone does not justify a new geometry project.
4. Inspect the supplied Shape translation/saturation witnesses and existing Round 7 actual-action/early-history evidence. Close already answered hypotheses, including fixed-prefit versus epoch-ten switches. Use existing J4 cached transforms for a small J2 cancellation readout only if it helps distinguish Scale mechanisms. Do not count overlapping member pairs as independent cases.

Target three hours of CPU wall work for the minimal cached readout, including figures, with the existing bounded lane (initially two scoring workers and controlled threads). A justified TRAIN800 transfer/raster diagnostic may use up to three additional CPU wall hours. These are planning allocations, not scientific abort gates. Record actual CPU/GPU/wall costs separately. Cached-analysis work is not a training speedup. Missing essential banks use the preparation GPU allowance and exactly the admitted approximately-10% panels; record this change before generation.

## 5. Stage 2: minimal code and documentation upgrade

| Change | Existing seam | Completion evidence |
| --- | --- | --- |
| Member/J1/JN physical risk and dynamic quadrature | `coherence/physical_risk.py`, `training/fidelity_quadrature.py`, relevant trainer override | Correct native/inclusion-weighted arithmetic, gradients and genuine J1 behavior; retained signed path |
| Member propagation through every scored block | `coherence/query_runtime.py`, `training/scalable_runtime.py`, endpoint/context construction | Declared member counts reach Shape, grouped queries and expanded blocks without J2 truncation or silently omitting features in deterministic mode |
| Budget-independent normalized PI feedback | `training/fidelity_budget.py`, `config/validate.py`, recording definitions | The existing PI mechanism accepts declared budget semantics without a `MeanRiskDifferenceBudget` type lock; old behavior remains identifiable |
| Source calibration and generic case roles | `training/scalable_prefit.py`, existing calibration/manifest code | Correct sample pooling and matching law; observed-field role comes from case metadata; no compulsory historical manifest migration |
| Positional Shape observation | `coherence/shape_compute.py`, existing raster geometry and frozen observer | Unchanged old PH values/calls when disabled, correct new gradients and frozen feature reload; no extra family row |
| Direct scientific readouts | `evaluation/frozen_coherence.py`, existing assessment/decomposition tools | A/R/S, actual rho/power errors, member/mean fidelity distinction from one saved bank |
| Cost and action instrumentation | Existing Round 7 executor/history tooling | Complete-epoch time/memory, weighted field-row use and changed-path actual actions, with no repeated unrelated qualification |

Paths in this table inherit `src/phycoflow_reconstruction/` except campaign scripts. Keep these small interfaces within existing modules; do not build a new provider framework, evaluator scheduler or provenance system. A geometry-general 3D/unstructured PH implementation is not included. N-field arithmetic and model independence should be qualified honestly on currently supported geometry.

The member-count work is concrete: current Shape routing requests `_draws(..., 2)`, larger grouped nonteacher inputs can be rejected, and expanded/shared paths contain two-member assumptions. Generic risk arithmetic alone does not complete JN training. Propagate the declared count through these callers and validate a real J3 affected action/reload as well as the ordinary J1 and J2 paths. Formal FFM learning still uses J2; this qualification does not authorize a member-count learning sweep. If a path remains unsupported, report its precise capability boundary rather than claiming complete JN execution.

Update the current computation and post-training explanations under `UpdatePlans/Coherence_Manuscript_Foundation/02_Computation/` and `03_Posttraining/`, plus the relevant portable `docs/` contracts. Preserve dated predecessors and align them with the revised `00_Overall/` and `01_Theory/` without rewriting LPC. Explicitly document scores versus gradient probes, paired versus conditional modes, risk choice, sampling versus geometry, actual optimizer composition, and physical/task claim limits. This legacy foundation is DATA2-backed: its paths are local research evidence, and edits need the appropriate host write authorization; independent source/tests/portable docs must not wait for an unavailable documentation mount. Publish the selected new scientific summary as ordinary files under the maintained report library.

## 6. Minimum informative verification

Use tests for the changed mathematical contracts, not a broad synthetic programme. The supplied [CPU script](evidence/support_shape_cpu_diagnostic.py) is a starting witness, not production qualification.

- Risk: exact finite-design native/complement expectation on nonuniform masses; J1/J2/J4 member arithmetic; signed correction retained for J>=2; masks, heterogeneous field scales, near-zero baseline and absolute tolerance versus denominator floor; unequal final batch/member counts.
- Shape: identical endpoints, integer/subpixel displacement, hard-bin blind spot, saturation, invalid cells/holes, declared physical axes and supported periodic seams; fixed threshold/window/scale reload; chunked value/gradient equivalence and PH-call count. Do not claim unsupported geometry was qualified.
- Generality: field permutation and modest field counts such as 1, 3, 5 and 7; permute field metadata, normalizers, query identities and artifacts together. N=1 handles absent pair queries explicitly and does not fabricate fields. A real existing deterministic adapter and the FFM independent-member path each complete an affected action/save/reload/append test. Use actual endpoint capabilities, not artificial repeated deterministic members. Missing model weights should not prevent tensor/adapter fixes; report the remaining real-model qualification and use the reserve only if available.
- Optimizer: from identical state, inspect new risk and optional Shape finite actions on TRAIN fixtures, including row norms, lambda/PI state, clipping, Adam displacement and physical finite changes. Differentiate a valid noisy/tradeoff action from wrong arithmetic, detached gradients or wrong units. Reuse established warmup and frozen coefficients initially; repair demonstrated response problems with one reasoned adjustment instead of an optimizer tournament.
- Recovery: for changed paths, compare uninterrupted training with a real fresh-process save, saved-config reload, resume and append. Save ordered columns/types/units/meanings and loss accumulation order. Device changes/formatting alone do not migrate histories. Reuse unaffected existing qualifications.
- Cost: use the retained Round 7 execution package if it fits GPU2, otherwise its valid lower-memory option. Measure changed-path full reporting epochs with active constraints, not only SOURCE's inactive row. Measure optional Shape increment and peak memory; no unsupported exact-trajectory or end-to-end speed claim. Only a demonstrated bottleneck justifies one further local optimization, such as avoiding already-discarded descriptor work.

A 10-epoch pilot, if needed, tests recording, warmup and response, not efficacy. It is included in preparation, has curves and can be counted as the initial phase of a subsequently continued scientifically unchanged candidate. A new contract/optimizer change is a named phase and cannot be hidden in a nominal fresh-SOURCE comparison. If a pilot is used solely for qualification, restart formal matched arms from the declared SOURCE and record the discarded exposure/cost.

## 7. Decisions after engineering

Adopt the generic risk/PI/sampling implementation when its scientific arithmetic, real affected path and recovery are sound; learned improvement is not required for a truthful reusable interface. Adopt direct readouts and explicit units/support reporting when they reproduce the declared quantities. Adopt the positional Shape implementation as an opt-in observable if it supplies useful position sensitivity at acceptable measured cost; do not make it the proven best training default without learning evidence. A failed location recipe leaves the existing Shape and the rest of the round usable.

At this point publish a concrete disposition: the risk semantics are aligned or identify the remaining discrepancy; support discrepancy/coverage deficit is material or not decision-changing on the admitted panel; direct and conditional Scale trends agree or diverge, with the responsible components; the new position feature has usable gradients or is deferred; the complete J1/JN path is qualified or has one named outstanding limitation. A finite support comparison alone does not establish expectation bias. None of these requires final-model improvement merely to close the engineering round.

## 8. Conditional learning, continuation and reasonable variants

Launch only if the implementation leaves a useful unanswered learning question: **under the explicit member-MSE contract, does adding spatial Shape information improve unoptimized positional/scientific relationships while maintaining field retention and a worthwhile overall SSS tradeoff?** The minimum attribution comparison is two fresh-SOURCE arms, `MemberMSE_BaseShape` and `MemberMSE_SpatialShape`, with identical Scale definition, observations, members, physical risk, PI, optimizer/warmup, query streams and exposure. They differ only in the declared Shape package. This comparison isolates that package under the new contract; it does not isolate member-MSE feedback versus the old signed contract. Compare historical signed endpoints descriptively, without claiming a matched risk-policy effect. A third signed-risk rerun is not the default.

If cached/engineering evidence rejects or defers the spatial feature and the only useful question is whether the generic member-MSE route learns at all, run one explicitly exploratory baseline instead. It can establish its own tradeoff, not a causal policy advantage. If neither question remains useful, close the round without training. These are evidence-based scope choices, not mandatory trials to exhaust the budget.

Initial substantive horizon: **100 reporting epochs per admitted arm**. Default maximum: **500 for one selected arm**, at most **600 substantive epochs total** for the two-arm route. Corrective phases and pilots continued as substantive learning share these same epoch and GPU ceilings; they do not create an additional trajectory allowance. Preserve the current twelve successful actions and 768 objective-bearing condition visits per reporting epoch if using the inherited FFM setup; derive and record actual exposure from resolved configuration. Report 100 epochs as 9.6 TRAIN8000 pass equivalents only when those values hold. Do not silently equate these epochs to the October or a different adapter's epoch.

At epoch 100 use the declared matched VALIDATION panel once, plus available TRAIN histories and existing needed TRAIN readouts. Continue a selected arm when field margins and relation trends leave a specific recovery question, the measured controller is acting coherently, and the remaining forecast fits. A noisy early constraint violation, one unresolved diagnostic or an interval crossing a proposed allowance is not an automatic stop. A short pilot need not satisfy final retention. Conversely, do not extend a persistently deteriorating, uninformative route solely to reach 500.

Matched Shape-package attribution belongs to the two epoch-100 endpoints. A selected epoch-500 child versus a stopped epoch-100 control has unequal exposure and supports its SOURCE-relative utility assessment, not a matched 500-epoch treatment effect. The package changes the relative PH contribution as well as adding a positional observation; do not attribute its entire effect to position information alone.

Within the total budget, one supported corrective phase may replace the default long extension: for example a diagnosed saturation correction, physical-weight error fix or inadequate controller-response adjustment. Reuse the strongest valid baseline where appropriate, state whether this is continuation or a fresh start, and report unequal exposure. Give a substantive revised candidate an adequate planned review, normally 100 epochs, instead of cycling through tiny runs. Multiple unrelated changes cannot be attributed individually. No repeated VALIDATION-guided coefficient or checkpoint search, and no reopening teacher/student or native supervision.

Stopping rules protect concrete things: wrong estimator/units/member law or corrupt resume requires repair of the affected path; nonfinite actions require diagnosis before continued updates; uncompetitive learning after an adequate review closes that recipe; exhaustion of the stated total budget closes with available evidence and recovery saved. A hard memory issue selects a valid lower-memory executor. Optional acceleration or Shape rejection does not cancel the main integration/evaluation work. Do not turn arbitrary gradient-cosine, speed or short-loss thresholds into additional campaign gates.

## 9. Whole-cost forecast and monitoring

| Allocation | Planned scope | Initial GPU allowance |
| --- | --- | ---: |
| Risk/J1/JN/Shape affected actions, calibration reuse or missing reference generation | Only changed paths and admitted panels | 2.0 h |
| Real recovery, active full-epoch timing and optional 10-epoch pilot | Reuse Round 7 qualifications where unchanged | 2.0 h |
| Missing cached endpoints, one diagnosed repair and preparation reserve | No broader evaluation | 2.0 h |
| Optional substantive learning plus its assessment/reserve | Within the expanded total, not additional to it | Up to 18.0 h |

At 86.011 seconds/epoch, two 100-epoch arms plus one additional 400 epochs cost about 14.34 GPU-hours of training. Adding six hours of preparation, 0.75 hour of generation/assessment and 1.25 hours of failure/tail reserve gives about 22.34 total. At 120 seconds/epoch, the same 600 epochs cost 20 hours before preparation and cannot fit that allocation. At 143 seconds/epoch they cost about 23.83 hours alone. These are forecasts from prior measurements, not GPU2 promises. Measure the current setup, include actual preparation spent and reserve, then declare the feasible maximum before formal launch. If 500 cannot fit, retain the informative matched100 comparison and choose a justified smaller extension or no extension; do not shrink required evaluation coverage or raise the ceiling silently.

Existing two-thread cached scoring receipts are approximately 68–156 seconds per VALIDATION100 bank and 539–1268 seconds per TRAIN800 bank, excluding new figures/diagnostics. Forecast GPU generation, data I/O, CPU scoring and final queue tail separately. Aim for added assessment wall time below 20% of planned training time; if essential scoring exceeds it, simplify redundant checkpoints/metrics and budget the remaining cost explicitly. This is a planning target, not an adoption gate. Preparation-only rounds report their CPU work directly, without forcing a meaningless training-overhead percentage.

Follow the [mandatory monitoring policy](../../../../../AGENTS.md#mandatory-run-monitoring-policy). Any long job uses **round8_luna_monitor**, `gpt-6-luna`, reasoning `low`. Give it launch identity, expected duration, saved receipt and failure/completion signals. Schedule a pilot completion check near its predicted finish; for formal learning, one useful budget check near epoch 50 and the epoch-100 decision; for an extension, one check near epoch 300 and endpoint. Adjust scheduled times from measured epoch durations. These are sparse observations, not extra scientific evaluations or per-epoch probes. Luna sends actionable notifications; the primary agent does independent work and does not poll the run or monitor. If Luna is unavailable, report that limitation and do not substitute another polling model.

## 10. Assessment, visualization and adoption

Cache one prediction bank per endpoint/panel/member law and reuse it for all applicable readouts. Native field/task evidence and frozen selected-support SSS evidence retain their own identities. Freeze one raw endpoint per assessment checkpoint; no amplitude/frontier sweep. Keep unoptimized tasks out of PI tuning and continuation nomination where the current protocol permits. VALIDATION is a development panel, not independent final confirmation.

Score SOURCE and every compared child with both common frozen Shape observers: the unchanged PH-only observer and the new PH-plus-location observer. Record A/R/S and component distances for each; never compare each arm only under its own training scalar. Reuse compatible PH transforms while preserving the distinct observer identities.

The final report must show: every field's pooled member-MSE ratio and paired margin, mean-risk/spread diagnostics and declared sensor/observation compatibility; direct A, separation R and fair S for each SSS family; signed Scale pair/band errors and amplitude context; old PH versus new location contributions; one joint-event and one regional scientific task; complete histories and measured training/assessment costs. When SOURCE sensor error is zero, retain its declared absolute numerical tolerance rather than an undefined ratio. Reuse SOURCE-selected representative/difficult IDs with shared plot color scales. Every reconstruction/error panel displays its own masked physical RMSE or relative L2; distinguish mean maps, member maps and event probabilities. Match condition IDs, masks, observations and member identity throughout. If physical units are unavailable, state dataset units rather than inventing Pa/K or physical wavelengths.

Use point estimates and uncertainty together. A point ratio below 1.10 is empirical feasibility; an interval crossing it means population noninferiority is not certified. Do not demand five upper-confidence passes unless that inferential contract is explicitly chosen. The single-trajectory bank supports frame-level development conclusions, not independent trajectory transfer. Use paired frame uncertainty with this limitation; add a bounded temporal-block sensitivity only for a decision it can change. Do not select new features to optimize the supposedly unoptimized tasks.

Final decisions are separate: retain/reject each engineering change; retain/defer positional Shape as a training default; describe direct relation gains and conditional-score tradeoffs; and retain SOURCE/C8 or nominate a raw child. Replacement requires all field point allowances under the new declared contract, preserved declared observation compatibility and a useful coherent pattern of relational and unoptimized-task benefit at measured cost. It need not improve every diagnostic, but a persistent material direct or fair Scale deterioration must be explained and is grounds to retain SOURCE for the broad enhancement claim. Larger separation alone is not a benefit, and a lower direct discrepancy alone is not calibrated uncertainty. At the endpoint a failed replacement can coexist with worthwhile reusable improvements.

## 11. Artifacts and closeout

Follow [Experiment_Artifacts.md](../../../../../docs/Experiment_Artifacts.md). Store generated/resumable payloads under the existing DATA2 project root with a case-owned `round_8` alias, one campaign index, named immutable attempts and preserved failed work. Verify target/capacity before writing and never fall back to home storage. Small disposable scratch belongs in a task-owned temporary directory; preserve required evidence before cleaning only confirmed disposable files. Do not relocate active Round 7 material.

Publish the selected report under `UpdatePlans/_Reports/Coherence_Adaptive_Repair/Round_8/` as ordinary Markdown/PDF and relative PNG/PDF figure pairs. A formal research report needs rendered page/panel inspection; this plan does not need a PDF. Save all curves/histories in case-owned run logs, refresh every ten completed epochs and at review/pause/final, without extra inference. Keep recording schemas and actual failed-attempt costs.

Validate portable links, a relocated or clean export and workspace layout. For eventual execution closeout, review the outgoing diff, preserve concurrent/user edits, commit/push only the authorized completed work and verify remote HEAD; do not absorb active Round 7 edits into an unrelated commit. The planning task itself performs no commit or push. Final delivery states actual adopted/rejected changes, whether learning occurred, exact limitations and any remaining question; no blanket claim that the entire project is solved.
