> **Historical record:** Round 4 is closed. This document is preserved for reference and is not a current launch instruction.

# Round 4: constrained SSS target construction and native amortization

Date: 2026-11-21. Status: proposed, not launched. Branch inspected: `Coherence_Adaptive_Repair`, HEAD `440545c`. This is the next round after the completed Round 3; it does not reopen or rename either rejected Round 3 run. The primary agent developed this proposal with the explicitly requested GPT-6.1 sol max reviewer. See [design discussion](DESIGN_REVIEW.md), [execution handoff](EXECUTION_PROMPT.md), and the [project retrospective](../../../../Project_Retrospective_20261120/REPORT.md).

## 1. Outcome and scope

**Decide whether computing a bounded SSS repair once in physical field space, then learning those targets with the reconstruction model's native training loss, can deliver a useful field-preserving repair at substantially lower total training cost.** Compare it with ordinary native supervised continuation from the identical SOURCE. The intended improvement is in scientific relationships excluded from optimization, not merely a lower SSS training score.

There is one mechanism and at most two student paths. First construct and qualify a finite TRAIN teacher bank. Only a useful, feasible teacher earns the student comparison. Review both students at epoch 100; their provisional and maximum endpoint is epoch 500. The proposed whole-round ceiling is **24 occupied GPU-hours**, with explicit CPU and storage budgets below. No automatic extension, second teacher recipe, amplitude sweep, extra model, or new query-estimator arm follows a failure. This document authorizes no execution by itself.

The strategy changes where optimization occurs: the expensive SSS derivatives act on stored endpoint fields, while repeated learner updates use a native supervised loss. It is a new training algorithm, not an exact acceleration of ConFIG. A successful result would support this repair package for the tested SOURCE and dataset. It would not establish universal transfer, posterior correctness, or a causal attribution to SSS alone.

## 2. Evidence that motivates this choice

The [closed Round 3 report](../../../../_Reports/Coherence_Adaptive_Repair/_bk/Round_3/report.md) rejects both learned outputs and retains SOURCE/C8. RawRisk failed all five signed-field constraints. SourceScaledRisk improved State and pressure but failed four field constraints, Scale, and regional utility. Both improved Shape and the reused event. The stronger feedback therefore did not solve the scientific problem at epoch 500.

| Verified finding | Consequence for this round |
|---|---|
| RawRisk / SourceScaledRisk cost 140.71 / 141.23 seconds per reporting epoch; recorded update work was approximately 98.5% of training wall time | Target model/descriptor/update computation. Loader and reporting changes cannot provide a large gain. The update timer does not isolate backbone backward from CPU persistence. |
| Exact State/Scale reuse saved 0.468 GiB, or 2.57%; no speedup established | Keep the implementation, but do not count it as the proposed acceleration. |
| Shared versus condition-specific Shape queries had score SD 0.200689 versus 0.031079 in eight fixed-SOURCE replays | Preserve the finding. It does not establish better parameter gradients, field retention, or learning; no simultaneous query-law experiment. |
| Scalarization, periodic coefficients, adaptive allocation, and frozen residual heads already have implementations and mixed or unsuccessful historical evidence | Do not present them as missing inventions or reopen a method tournament. |

The current full-model route shares one J2 endpoint bank but traverses the model graph separately for three family gradients and usually one field-feedback gradient. Native rectified-flow training instead evaluates a random-time velocity regression; it need not differentiate a four-stage endpoint or evaluate SSS on every learner update. This is the credible source of a large possible saving. Its magnitude remains to be measured, including target preparation. See [code correspondence](#8-code-correspondence-and-bounded-implementation).

## 3. Frozen scientific inputs and training membership

Keep the immutable FFM SOURCE, physical fields/units/masks/masses, observation protocol, four-stage evaluation rollout, existing TRAIN-fitted representations, expanded query bank, coefficients, and final observer versions. Keep uniform query sampling and the qualified State/Scale reuse. Preserve complete joint State/Scale anchors and corrected Shape gradients. Do not fit the reused event or regional covariance task.

Resolve the exact original TRAIN membership from the saved split before execution. The expected combustion split is TRAIN8000; the existing TRAIN800 cache covers approximately 10%, not the entire learner stream. Both students must use the same full declared TRAIN membership. Build one fixed observation manifest per training condition, shared by SOURCE generation, teacher construction, and both students: for this case use 256 T observations under the existing seeded uniform protocol, preserving values from the original data. The field/count choice belongs to the case manifest, not generic teacher code. Round 3's training configuration allowed 192–384 observations; freezing 256 is an explicit common training-protocol change. Do not resample observations while looking up targets conditioned on a different sensor set. Sample native training query points from the stored physical support with the existing model convention.

The teacher bank requires four SOURCE endpoint atoms per training condition, generated without gradients. Reuse cached atoms only when SOURCE, conditioning, coordinates, units, rollout, and member identity match. Generate missing atoms once. **Full TRAIN bank generation is an explicit proposed training-target preparation operation, not a full-population model assessment.** Subsequent diagnostic model evaluation remains approximately 10% TRAIN and the existing VALIDATION100 panel. If full target preparation is excluded at execution authorization, revise the proposal before launch; do not silently reduce training membership or mix repaired and unrepaired target definitions.

The teacher operates on native support; apply the frozen training support/representation operator before SSS scoring, and propagate its endpoint gradient back to the native arrays. A bank bound to the existing Q4096 support must not receive an unannounced native Q40300 replacement; the physical-mass assertion in the runtime is meaningful. Sites outside the operator's influence remain SOURCE unless another declared operator moves them. Do not replace native field constraints by a small spatial sample. The teacher and truth targets must enter the native loss in its expected model units, with normalization applied exactly once. The dense reference and original observations remain immutable; repaired targets live in a separate bank. Preserve declared dataset units; where the saved metadata says a physical unit is unknown, do not invent an SI label.

## 4. Teacher: a finite empirical law with explicit field feasibility

### 4.1 Probability and score contract

For condition $c_n$, reference field $z_n$, and fixed SOURCE atoms $x^0_{nj}$, use $K=4$ and optimize fields $y_{nj}$ initialized to $x^0_{nj}$. The teacher distribution is explicitly

$$
P^T_n=K^{-1}\sum_{j=1}^{K}\delta_{y_{nj}}.
$$

For each existing State, Scale, or Shape distance $d_{f,q}$, retain its transform, physical measure, complete joint construction, frozen calibration, and query law. The score of this finite law is

$$
S^V_{f,q}(Y_n,z_n)=\frac1K\sum_j d_{f,q}(y_{nj},z_n)-\frac1{K^2}\sum_{j<k}d_{f,q}(y_{nj},y_{nk}).
$$

This is the empirical-law energy-score form, including the zero self-distances through the $K^2$ denominator. Jointly optimized atoms are not independent draws from an unknown conditional model. Therefore do not use the fair U-statistic denominator $K(K-1)$ and call the teacher estimate unbiased. The final student observer continues to use its existing fair scores on genuinely independent J4 model draws. Teacher V scores and student fair scores must have different labels and must not share an unlabeled history curve.

Use one scalar teacher objective, the mean of the three positively SOURCE-normalized family V scores. Fit the three positive V-score normalizers once on the fixed approximately-10% TRAIN SOURCE panel using the current calibration floor convention. Keep the existing inner-family weights and transforms. An independent fixed query replay assesses teacher progress; it does not refit these normalizers or select new representations. This scalarization defines the new teacher objective; it is not claimed to reproduce parameter-space ConFIG.

### 4.2 Feasible set

Let $\|u\|_w^2=\sum_i w_i u_i^2$ on the declared valid native support, with normalized physical masses. For every condition and field, constrain the teacher's finite-law mean error, member error, and displacement:

$$
\|\bar y_c-z_c\|_w^2\leq\|\bar x^0_c-z_c\|_w^2+\epsilon_c,
$$

$$
K^{-1}\sum_j\|y_{jc}-z_c\|_w^2\leq K^{-1}\sum_j\|x^0_{jc}-z_c\|_w^2+b_c,
$$

$$
K^{-1}\sum_j\|y_{jc}-x^0_{jc}\|_w^2\leq\epsilon_c.
$$

Here $\epsilon_c$ is the existing positive absolute mean-risk allowance, expressed in the bank's field units. Set $b_c=\epsilon_c$ as an explicit teacher design choice, also using $\epsilon_c$ for displacement; no extra tuned radius is introduced. The existing student member guard is relative and remains unchanged. In particular, the configuration's `member_allowance_fraction` derives the signed-mean allowance; it does not supply a second absolute member-error budget. Copy the actual $\epsilon_c$ values and unit conversion from the frozen reference artifact. Require zero correction at observed point/field entries, preserving the SOURCE observation policy exactly. Mask invalid coordinates out of both scores and norms.

The first constraint concerns the actual mean of the finite teacher law. No finite-member variance subtraction belongs in that quantity. This is distinct from the unbiased signed mean-risk estimator used to evaluate independent student draws. Neither teacher feasibility nor a teacher's lower TRAIN score guarantees student or held-out feasibility. Report member spread separately so contraction is visible.

### 4.3 One simple feasible update

Work in dimensionless correction coordinates $u_{njc}=(y_{njc}-x^0_{njc})/\sqrt{\epsilon_c}$. Use the physical-mass inner product, averaged over atoms and fields, to define the endpoint gradient. Compute each condition's gradient without an accidental batch-size factor; zero its observed/invalid entries. Clip its norm to at most one in this metric and use a fixed dimensionless step of 1.0. This is one declared solver setting, with no step-size sweep. Positive-mass support is required for converting the autodiff derivative to the weighted gradient.

Precisely, with the declared weighted gradient $g_u=\operatorname{grad}_{w,K,C}L_T$, form $\widetilde g_u=g_u/\max(1,\|g_u\|_{w,K,C})$ and $v=x^0+\sqrt{\epsilon}\,(u-\widetilde g_u)$, using fieldwise multiplication by $\sqrt{\epsilon_c}$. Then use the SOURCE-anchored ray $y(\alpha)=x^0+\alpha(v-x^0)$, $0\leq\alpha\leq1$. Every field constraint above is a scalar quadratic along this ray and is feasible at $\alpha=0$. Select the largest $\alpha$ in the feasible interval containing zero, intersected across fields and constraints, separately for each condition. Evaluate the resulting inequalities directly in FP64; round down at a numerical boundary if necessary. If the proposal is nonfinite, preserve the saved state and diagnose once rather than accepting it.

Anchoring the ray at immutable SOURCE on every visit permits the direction to rotate after a boundary is reached. Clipping only a ray from the current boundary point can repeatedly return zero and falsely suggest no repair is possible. This radial map is a feasibility operation, not an exact Euclidean projection onto the intersection, a proximal minimizer, or a monotonic-descent theorem. The finite teacher learning result decides whether it is useful.

All expensive SSS differentiation ends at the endpoint arrays: there is no SOURCE/backbone backward in this stage. Use the existing sampled query law, fresh at its ordinary cadence, without enumerating the entire query bank on each visit. Cache immutable reference transforms and the SOURCE atoms; keep current teacher-dependent transforms live and refreshed.

## 5. Native amortization and the necessary control

| Path | Initialization and target | Purpose |
|---|---|---|
| `FFM_SSS_Target_Amortization` | Identical immutable SOURCE; one teacher atom sampled uniformly per condition visit | Test the complete constrained-SSS-target package |
| `FFM_Native_Supervised_Continuation` | Identical immutable SOURCE; original dense reference for the same visit | Establish whether ordinary additional supervised fitting is sufficient |

Use identical condition/order, observations, query-selection law, native prior law, random-time law, optimizer settings, and reporting-epoch definition. Separate RNG streams keep teacher atom selection from changing the common prior/time stream. Use the existing native loss, not a new SSS loss during learner updates. For the present rectified-flow adapter this is

$$
\xi\sim p_0,\quad t\sim U[0,1],\quad x_t=(1-t)\xi+t y,\qquad L_{\mathrm{RF}}=\mathbb E\|v_\theta(x_t,t,c)-(y-\xi)\|^2,
$$

where $y$ is the sampled teacher atom or original reference. Preserve the adapter's actual native reduction/units. This objective uses a fresh native prior; it does not preserve the SOURCE atom's generation seed or its transport coupling. Inference still uses the unchanged four-stage reconstruction and independent J4 members. Do not claim one-stage inference, deployment acceleration, or teacher-index-conditioned predictions.

Retain the recorded common 50-epoch learning-rate warmup, $5\times10^{-6}$ to $5\times10^{-5}$, AdamW weight decay $10^{-6}$ and clip norm 1 as the initial fixed recipe. No source/teacher mixture coefficient, residual head, dual-rate search, or teacher refresh during student fitting. If native training at these settings fails, report that outcome; do not silently convert it into a new tuning campaign.

This control answers whether the whole package adds value over native continuation. It does not isolate SSS information from the teacher's SOURCE proximity or finite-law regularization. A third source-only distillation arm would address that attribution later only if a useful candidate exists and the manuscript requires it.

## 6. Stages, horizons, continuation and adoption

**Preparation:** one bounded current-route timing/profile and native-loss timing on GPU0; at most three numerical teacher checks using the fixed approximately-10% TRAIN panel, solely for units, gradients, feasibility, memory and cost. These checks do not establish learning benefit or select among recipes. Reuse compatible qualifications. Stop a demonstrated incompatible implementation route rather than building a new training framework.

**Teacher learning:** initial and maximum horizon 100 reporting epochs, with the same nominal condition-visit definition as Round 3: twelve batches of up to 64 conditions per reporting epoch. This is approximately 768 visits, not a full TRAIN8000 pass; record the actual stream, partial batches, per-condition visit counts and dataset-pass equivalents. Under 8000 conditions, 100 such epochs represent about 9.6 passes. SOURCE bank preparation is not an epoch. Keep complete per-family V-score, feasibility, spread, displacement and accepted-ray histories; save curves first, every ten completed epochs, and at completion.

At the teacher endpoint, use the matched TRAIN800 diagnostic panel and a frozen independent query replay. Require all native teacher field/sensor constraints, no family regression exceeding 0.02 of its frozen SOURCE V-score normalizer, and at least one family improvement of 0.05 units. Also report how frequently the correction is zero, but do not invent another acceptance threshold. This TRAIN-only construction check earns a student experiment, not a scientific model claim. Do not inspect event/regional task scores or optimize those tasks here. If the teacher misses the margins after the authorized horizon, close the mechanism without students or a replacement teacher design.

**Students:** both paths start with a 100-epoch learning review and a maximum/provisional endpoint of 500 reporting epochs under the same nominal visit definition. Cache one raw output per review/endpoint. At epoch 100 inspect fields, sensors, State/Scale/Shape and learning trends; keep event and regional tasks withheld until the terminal assessment. Continue both paths to 500 only if a useful field-preserving repair question remains and the measured inclusive cost forecast fits. A single noisy score is not a stop rule. Persistent multi-field failure with no recovery, no core benefit after substantial native fitting, or a measured failure of the acceleration budget closes the package. If it closes at 100, evaluate the terminal scientific tasks once from those saved outputs and finish; no 500-epoch or extra-arm launch follows automatically.

**Final scientific acceptance:** retain the Round 3 student observer, fixed allowances and practical margins. Require every signed-field and observation constraint and the separately reported legacy member guard. Require no core regression exceeding 0.02 frozen TRAIN SOURCE score units and an improvement of at least 0.05 in one core. Require regional-task improvement of at least 0.05 units versus SOURCE, with event worsening no greater than 0.05 units. The repaired student must additionally improve the regional task over native continuation by at least 0.02 units and must not obtain that gain through failure of a field/core condition. These are decision margins, not significance thresholds. A tie favors the simpler native continuation if it independently meets the scientific acceptance criteria; otherwise retain SOURCE/C8.

**Acceleration acceptance:** target at least a twofold reduction in inclusive candidate GPU work relative to the completed SourceScaledRisk route at the same endpoint age, charging the candidate all necessary SOURCE target generation, teacher construction, native fitting and its assessment. Report both complete-epoch learner speed and this inclusive ratio. Report CPU worker time and the elapsed preparation/scoring tail separately; a large CPU delay cannot be hidden by a lower GPU count. Because the algorithm changes member/target exposure, this is a practical matched-horizon cost comparison, not trajectory equivalence. No time-to-success claim is available against a baseline that never passed the scientific gates.

The recipe needs both useful output and worthwhile cost to become the recommended repair. If it is faster but fails utility, retain only reusable computation and reject the recipe. If it is useful but misses the acceleration target, record the scientific result without promoting it as the requested fast method. End this local mechanism study after that decision; consolidation or a collaborator transfer is the next discussion, not another automatic combustion repair round.

## 7. Resource forecast and runtime devices

Round 3 SourceScaledRisk occupied 70,634.7 GPU-seconds through epoch 500, approximately 19.62 GPU-hours. Its whole-campaign cost is not a same-device controlled speed experiment; the initial GPU0 timing anchors the new implementation comparison. A twofold inclusive candidate target at epoch 500 is approximately 9.81 GPU-hours. If $C_{\mathrm{prep}}$ includes target generation, teacher fitting and candidate assessment, and $\tau_N$ is measured native seconds per epoch, the necessary condition is

$$
C_{\mathrm{prep}}+500\tau_N\leq 35{,}317\ \mathrm{GPU\ seconds}.
$$

No measured native-epoch or teacher timing exists for this proposal. Therefore the following are **caps and planning allowances, not observed runtime predictions**. Before formal work, substitute measured timing from the bounded qualification and forecast the full path. Do not launch a horizon that the forecast cannot support, and do not shrink the 100-epoch learning review to manufacture an efficacy result.

| Work | Proposed GPU allowance | CPU / wall treatment |
|---|---:|---|
| Implementation qualification and same-device timing | 2 hours | Focused affected checks, not a full source-admission programme |
| Full TRAIN SOURCE target bank preparation | 2 hours | Include reads/writes and actual conditioning; matched cache reuse only |
| 100-epoch endpoint teacher | 4 hours | Up to 6 CPU worker-hours for descriptor work; measure nested CPU wall separately |
| Two native students through at most epoch 500 | 10 hours combined | Requires measured mean native epoch cost no greater than about 36 seconds to fit this allocation |
| Required student inference/scoring and closeout | 1 GPU-hour | Up to 2 CPU worker-hours for scoring; forecast the enclosing CPU tail |
| Variation, failures, durable checkpoint/closeout reserve | 5 GPU-hours | This reserve permits no extra candidate, recipe, or epoch extension |
| **Total ceiling** | **24 occupied GPU-hours** | CPU worker limits are separate; overlapping work is not added as elapsed wall |

The inclusive twofold candidate target can be stricter than the 24-hour experiment ceiling, because the latter includes a necessary control. Charge candidate preparation once, never erase it by amortizing over hypothetical future applications. If teacher CPU work or target I/O exceeds the declared limits, preserve the attempt and close or revise the plan before dependent launch; do not quietly move work to an uncharged resource. During a run, stop at a durable epoch boundary if the measured remaining forecast would exceed the authorized ceiling.

At $8000\times4\times100\times403\times5$ float32 values, one full bank is approximately 25.79 GB (24.02 GiB), before metadata/compression. SOURCE and repaired banks alone are approximately 51.58 GB. Plan at least 100 GB free DATA2 capacity for these arrays, recovery and required caches, then replace the estimate with the resolved shape and checkpoint sizes. Stream blocks; do not load the whole bank into GPU memory. Store only necessary versions, preserve SOURCE and the last complete recoverable teacher state, and never overwrite the original dataset.

**GPU0 is primary; GPU1 is optional.** Run the target preparation/teacher and same-device timing on GPU0. After teacher acceptance, GPU1 may run the matched control in parallel if free and compatible. Device choice stays in runtime configuration; record actual hardware and occupancy, without immutable index/UUID restrictions. Compare methods on the same device for hardware speed claims or label the difference. Retain checkpointing where needed; the older uncheckpointed B64/J2 rollout actually failed with OOM. No device reservation or job is created by this plan.

Follow the [monitoring policy](../../../../../AGENTS.md#mandatory-run-monitoring-policy). A future `car_round4_amortization_luna_monitor` must use `gpt-6-luna`, low reasoning, read-only push notifications. Set sparse checks at the forecast teacher endpoint, each student epoch-100 review, and the authorized terminal epoch/budget boundary, including the measured assessment tail. At launch fill in actual times from $E\tau$; revise only after a scheduled observation or actionable failure. The primary agent does not repeatedly poll runs or the monitor. Follow the existing 30-minute blocked-device wait boundary where applicable; preserve exact resume state rather than queue indefinitely.

## 8. Code correspondence and bounded implementation

| Requirement | Existing implementation to use / precise gap |
|---|---|
| Physical risk and units | [physical_risk.py](../../../../../src/phycoflow_reconstruction/coherence/physical_risk.py), `physical_field_mse` and `physical_ensemble_risks`; keep student signed-risk semantics. Teacher finite-mean error is a separately named quantity. |
| All-atom SSS distances | [state_compute.py](../../../../../src/phycoflow_reconstruction/coherence/state_compute.py), [scale_compute.py](../../../../../src/phycoflow_reconstruction/coherence/scale_compute.py), [shape_compute.py](../../../../../src/phycoflow_reconstruction/coherence/shape_compute.py); reuse distance construction and introduce an explicit teacher empirical-law reduction. |
| Member-count trap | [scalable_runtime.py](../../../../../src/phycoflow_reconstruction/training/scalable_runtime.py) `_expanded` currently constructs the conditional extra leaves from the first two members. Passing K4 to this wrapper does not establish that all four participate. Teacher code must explicitly consume all K4 atoms in every relevant leaf. |
| Shape gradient boundary | Keep the qualified differentiable per-pack path. Compiled assessment-only batching may change finite-precision tie gradients; do not quietly enable it for teacher optimization. |
| Native learner | [gl_rbf_cq_adapter.py](../../../../../src/phycoflow_reconstruction/models/flows/pointcloud/adapters/gl_rbf_cq_adapter.py) `training_loss` / `training_backward`; [portable_core.py](../../../../../src/phycoflow_reconstruction/models/flows/pointcloud/core/portable_core.py) random-time RF bridge. Supply the selected target through the existing batch target interface, retaining original observations. Reuse the smaller compatible portion of [post_training.py](../../../../../src/phycoflow_reconstruction/training/post_training.py) or [base_training.py](../../../../../src/phycoflow_reconstruction/training/base_training.py), with the lifecycle corrections below. |
| Current cost comparator | [fidelity_budget.py](../../../../../src/phycoflow_reconstruction/training/fidelity_budget.py), separate family/constraint gradients; [rollout.py](../../../../../src/phycoflow_reconstruction/training/rollout.py), four-stage graph. Use [bottleneck.py](../../../../../src/phycoflow_reconstruction/training/execution_strategy/bottleneck.py) for one bounded scoped profile where helpful. |
| Cached scoring/reporting | Existing Round 3 final scorer, immutable SOURCE/reference cache and physical figure utilities; no new scheduler or provenance service. |

The implementation footprint is one target-bank reader, one endpoint-teacher driver/reduction, and a bounded integration of the existing native loss with a suitable training lifecycle. **The base trainer cannot be invoked unchanged:** it constructs a fresh model, uses full-pass epochs, and does not apply this warmup in its present loop. Explicitly materialize the identical SOURCE/weight view, preserve the shuffled condition stream, implement twelve-batch reporting epochs and the epoch-valued warmup, and bind fixed observations and target selection. Reusing the post-training lifecycle with the native loss may require fewer changes than extending base training. Choose the smaller correct integration; verify the first recorded epoch and initialization against these contracts before formal learning. A full TRAIN8000 pass must not silently become the intended 768-visit reporting epoch.

Preserve default fair scoring and historical checkpoint behavior. Focus tests on the new scientific contracts: empirical-law score value/gradient versus explicit independent sampling of atom indices; participation of all four atoms; radial feasibility including boundary/zero-direction cases; sensor/physical-unit alignment; and no teacher labels in inference. Verify a real fresh-process teacher/student save, configuration reload, resume and append against uninterrupted execution for affected state. Save ordered recording columns with types, units and meanings and the accumulation order. JSON object order alone is not scientific incompatibility.

An exact ordinary-CQ live-context cache is a credible separate engineering opportunity: the physical wrapper already reuses differentiable conditioning/static-query context, while the ordinary rollout rebuilds it. It is **deferred**, because native amortization already removes the repeated rollout from learner updates. Do not combine that refactor, batched VJPs, periodic coefficients, condition-specific Shape queries, and this teacher study in one round. If the chosen route cannot use existing native training without a broad rewrite, stop and report that concrete obstacle.

## 9. Quantitative assessment and figures

Use the same approximately-10% VALIDATION100 development panel, native masks/units and independent J4 seeds for SOURCE and both raw students. TEST remains locked. Generate predictions once per checkpoint/panel and reuse them for every metric and figure. Compare fixed final outputs; no blend search or retrospective best-checkpoint selection. Event and right-half physical-mass-weighted CO–U1 covariance remain outside every loss, teacher check, parameter choice and epoch-100 continuation decision. They have been exposed in prior rounds and are reused unoptimized development tasks, not fresh independent confirmation.

| Evidence panel | Content and interpretation |
|---|---|
| Field retention and relationships | All five fieldwise RMSE/relative L2, signed-risk difference with allowance, member MSE/spread, and sensors; separate State/Scale/Shape and two task changes with their actual frozen units. Never hide a failing field in an average. |
| Repair transfer chain | SOURCE → TRAIN teacher → learned student. Teacher row is labeled in-sample empirical-law construction; held-out student row uses the independent fair observer. Show whether teacher benefit survives amortization without implying the scales are interchangeable. |
| Matched reconstruction atlas | Freeze a representative and a SOURCE-defined tail condition before candidate inspection. Show reference, SOURCE, both students, residuals, and the regional CO–U1 relationship on the same support. Every plotted field/error panel visibly annotates relative L2 or RMSE in declared units. |
| Relationship explanation | On those same cases show joint occupation, signed scale organization and excursion/topology evidence, paired with the unoptimized regional statistic. Avoid selecting a favorable pair/region after seeing the candidate. |
| Learning and inclusive cost | Separate teacher V losses, native RF losses, held-out fair scores, and field constraints. Show complete histories and stacked target-generation/teacher/student/assessment GPU work, CPU work and elapsed tail. No invented scalar ConFIG curve. |

Use paired condition-level differences and descriptive contiguous-block summaries for the same-trajectory validation panel. Do not report IID frame confidence as independent evidence, or single-training-seed variation as replicated training uncertainty. The practical fixed margins decide adoption. A more sophisticated uncertainty study is not a prerequisite for closing an uncompetitive route.

Refresh curves from existing histories first, every ten completed epochs, at validation, pause and completion; no extra inference follows that cadence. For the final research report use Python figures, PDF masters and relative PNG previews with concise captions, a compact figure index, and rendered/decoded/link validation. Planning remains plain Markdown. Assessment added wall time should normally stay within 20% of training wall; if essential scoring does not fit, explicitly budget it rather than reducing required coverage.

## 10. Deliverables, stopping point and manuscript consequence

Store generated payloads under the prescribed DATA2 project root through a new case-owned `coherence_adaptive_repair/round_4` alias, following [Experiment_Artifacts.md](../../../../../docs/Experiment_Artifacts.md). Verify target, ownership and capacity at launch. Keep one campaign index, named immutable attempts, SOURCE/config identities, complete histories, failed-attempt costs and a concise terminal decision. Preserve all Round 3 artifacts and the existing branch history. Durable implementation and final decision documents follow the repository's normal commit/push practice after a completed execution round; this planning task neither changes scientific code nor pushes a branch.

The final recommendation must be one of: adopt a scientifically useful and materially cheaper amortized repair; retain a useful native continuation when the extra teacher fails to add value; or retain SOURCE/C8 and close this local mechanism. Faster unsuccessful learning is a negative repair result. A useful teacher with an unsuccessful student is an amortization failure, not demonstrated model enhancement. A successful student with costly preparation is a scientific finding that misses the intended acceleration claim.

For the manuscript, keep LPC and finite SSS as motivation/observable design. Describe this new algorithm as constrained target construction followed by native amortization; its empirical-law teacher contract and final fair observer are explicit. The generic interface needs a model that can learn supplied dense targets through a native loss: deterministic regression, flow matching, or a suitably implemented diffusion objective. Only FFM is tested here; adapter availability is not evidence of effectiveness on those other capabilities or datasets. Prefer a real collaborator transfer after a retained recipe over another sequence of combustion-specific repairs.

## Primary methodological references

The native flow objective's simulation-free vector-field regression is supported by [Lipman et al., Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) and [Liu et al., Flow Straight and Fast](https://openreview.net/forum?id=gWxpdtQpiYV). They do not establish the proposed teacher's scientific benefit or convergence of this finite-budget adaptation. The distinction from fresh parameter-gradient aggregation follows [Liu et al., ConFIG](https://arxiv.org/abs/2408.11104) and the repository's actual optimizer composition. The teacher feasibility construction and study margins above are this proposal's design choices, not guarantees imported from those papers.
