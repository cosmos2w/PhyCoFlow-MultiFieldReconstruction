> **Historical record:** Round 1 is closed. This document is preserved for reference and is not a current launch instruction.

# Round 1 adaptive computation and scientific contracts

This technical design accompanies [WORKPLAN.md](WORKPLAN.md). It specifies a recommended starting algorithm and the proof obligations. TRAIN evidence may justify a simpler or better implementation within the workplan's redesign rules. Equivalence, unbiased estimation and changed scientific targets are separate claims.

## 1. Representation ledger and mathematical deliverables

| View | Retained C8 information | Round 1 addition | Outside the claim |
| --- | --- | --- | --- |
| State | All unary occupation channels and eight hard/diverse mixed channels in one fixed weighted Hilbert norm; fair scoring of independent occupation representations | A small frozen group/all-field projection bank, each with a complete W2-distance fair leaf outside the core square root | Positions and unqueried directions; laws of scalar projections separately need not identify the joint law of a random spatial occupation measure. |
| Scale | Signed retained bands and piece means in the full joint norm; complete log-power-self; additive signed and rho queries | A small bank of fixed regional centered field-band triple descriptors, with scalar fair scores | Excluded/zero-weight modes, omitted regions/triples, causal transfer or a full cumulant reconstruction. |
| Shape | Frozen pooled-TRAIN unary/pair signs/lines and explicit 2-D H0/H1 raster semantics | Execution/compiler improvements; no additional triple-persistence catalogue | Arbitrary spatial resolution, absolute position, 3-D or complete multiparameter topology. |

### Representation sensitivity

Let P be a physical-mass orthogonal projector retaining the signed modes and connected-piece means. Let A be the actual linear joint signed embedding with A = A P. On positive-mass support calculate c_min such that ||A v||² >= c_min ||v||²_M for every retained v. Under a consistent normalized mass convention its modal components use sum_b filters[b,k]^2 / scales[b,f]^2, with the actual piece-mean weights/scales. Separate projected-leaf `band_weights` do not belong to this joint constant. Include every retained component. If c_min=0, the inverse bound below is unavailable: report no retained-space coercivity, or explicitly redefine P to the positive-weight subspace and put every removed mode into the unresolved residual. A cutoff mode must not disappear silently.

For an L-Lipschitz physical task on the declared admissible class,

\[
|\mathcal Q(x)-\mathcal Q(z)|\le L\{c_{\min}^{-1/2}\|A(x)-A(z)\|_2+\|(I-P)(x-z)\|_{\mathbf M}\}.
\]

Derive the mass/field-unit factors against the implementation and verify small nonuniform/disconnected examples and real fitted spectra. This is a pointwise representation-distance bound, not a bound obtained by substituting the fair energy score for that distance. A finite map can be injective; do not assert an automatic nullspace.

For a threshold intersection, use a margin bound. With normalized physical masses, thresholds t_a and positive margins gamma_a, event disagreement area is bounded by the reference mass within gamma_a of any relevant cut, plus sum_a ||x_a-z_a||²_M / gamma_a². Derive this via the union bound and Markov's inequality, and report margin populations. Unnormalized spatial first-moment error is then bounded using the coordinate magnitude. Avoid unstable normalized centroids of empty/rare events.

Use a parity control to expose pair-only blind spots, and sensor-preserving local perturbations to compare C8-core sensitivity with added-query sensitivity at the same displacement. The full joint Scale anchor may already detect the control. Qualify a new query for interpretable/incremental sensitivity, not for repeating that toy demonstration. A compact decoder/task-sufficiency argument may replace an unnecessarily expensive query if its assumptions and tested errors are explicit.

### Conditional scoring and fidelity

For each added scalar descriptor phi_q, use the unsquared scalar energy/CRPS leaf

\[
\ell_q(X_{1:J},Z)=\frac1J\sum_j|\phi_q(X_j)-\phi_q(Z)|-
\frac1{J(J-1)}\sum_{j<k}|\phi_q(X_j)-\phi_q(X_k)|.
\]

A projected State leaf uses the same formula with complete weighted W2 distance between its one-dimensional occupation measures. Compute that metric before sampling the leaf. Both members receive gradients; handle exact coincidence consistently. Fixed transformations and finite moments support the appropriate proper represented-law target. Nonlinear ConFIG/AdamW composition is not an expected scalar proper score minimizer by theorem.

Write out the population identity R_member = E[V_truth + mean_error² + V_model] and R_mean = E[V_truth + mean_error²]. A calibrated true-law generator can violate a tight member bound relative to a mean predictor. Also test scalar J2 bracketing: the fair absolute-distance score is zero and locally flat while truth lies between members. These distinguish contract incompatibility, estimator weakness and representation blindness.

Implement signed corrected mean risk with physical masses and actual sample-inclusion/proposal correction. Check equality to member risk minus sample variance and, for J2, the inner product of member errors. The opt-in mean-difference constraint uses matched frozen SOURCE outputs, an explicit field-unit allowance and a positive independent controller scale. SOURCE extra inference/storage, if required for varying observations, is part of its cost. No claim of a new learned contract follows from CPU/short-run qualification.

## 2. Bounded TRAIN discovery

Fit a common physical/field normalization from TRAIN truths and declared floors. Use approximately 10% TRAIN condition IDs with independent SOURCE members for both native models. Discovery uses only these endpoints and any declared development probes. It never sees the reserved event or held-out conditions. Within each source retain a common comparison observer; source-specific C8 hardness banks are recorded rather than silently mixed.

Build a deterministic balanced binary hierarchy over fields from explicit physics groups or small TRAIN field sketches. For learned groups, use balanced median splits of fixed low-dimensional standardized field/band sketches rather than quadratic agglomerative clustering or every pair correlation. Record field order, sketches, tree and seed. The tree is a computational proposal structure, not a causal interaction graph.

Generate at most min(8C,128) total candidate projections/triples, using physical groups plus a reproducible random exploration share. Multiple bands/regions count as distinct candidates. Round 1 triple queries require **three distinct fields**; repeated-field cross-band products are valid possible future queries but are not this population. Do not enumerate C-choose-3 or every sign/line product. For C<3, triples are inactive; for C=1, mixed projections are inactive. Deduplicate sign-equivalent projections and complete descriptor IDs.

Retain up to four projection and eight triple leaves, or fewer if support, redundancy and sensitivity tests do not justify them. Preserve all C8 core channels regardless. Rank candidates with source discrepancy, repeatability, complementary sensitivity and measured observer cost. Evaluate redundancy using fixed sketches or a bounded selected set; do not hide a quadratic full-catalogue search. Ranking is a heuristic for useful TRAIN queries, not an optimal sparse-recovery theorem.

One candidate triple is

\[
\phi_q(x)=\operatorname{asinh}\left[\sum_i\bar m_{r i}
\prod_{j=1}^3\frac{(F_{b_j}x_{c_j})(i)-\overline{F_{b_j}x_{c_j}}_r}{s_{r,b_j,c_j}}\right],
\]

where regional masses normalize over declared nonempty physical support, centers use that same endpoint support, and positive scales freeze from TRAIN. The asinh transform is fixed, monotone and versioned; compare the untransformed option on TRAIN if saturation harms sensitivity. Existing `third_order_product_moments` is the starting point, but its current squared-error registry is not the required fair conditional score. Regional masks, filtering and descriptor floors are explicit artifacts. No lag extension is silently inferred.

Freeze selected IDs, representation scales and population weights before formal training. Added-block calibration follows the existing ordinary-batch gradient probes at epochs 1/3/6/8/10 and freezes after epoch 10; attraction and separation remain together. Use new semantic names such as `state.group_projection` and `scale.third_order`. Inactive blocks remain absent. A newly selected bank is a new target even if its size is unchanged.

## 3. A fixed target with an evolving proposal

For each family f define

\[
L_f=A_f+\sum_h W_{fh}\sum_{q\in\mathcal U_h}\pi_h(q)\ell_{fq}.
\]

A_f holds complete C8 anchors, U_h is a fixed unary/pair/discovered-bank stratum and pi_h is its declared target distribution, normally uniform. Every nonzero term has positive, fixed population weight. Internal coefficients change only during the declared common calibration period, then freeze. Proposal updates do not alter these target weights.

An allocation samples complete leaves with detached pre-draw probabilities p_h. For n_h independent draws,

\[
\widehat L_h=\frac{W_{fh}}{n_h}\sum_j\frac{\pi_h(q_j)}{p_h(q_j)}\ell_{fq_j}.
\]

The same identity holds for pathwise gradients under the usual integrability assumptions. If probabilities depend on parameters, stop gradients through them here or derive the required sampling-law correction. Counts n_h, like proposals, are fixed from past information before current draws. Do not stop the current sample mean early based on realized query cost/gradients without a valid sequential-inclusion estimator; plan the batch and enforce the campaign cap at safe update boundaries. Every active stratum receives at least one sample, unless a separately corrected stratum-inclusion design is implemented. Zero allocation to a nonzero target term is not unbiased.

Coalesce repeated complete outer IDs only with multiplicities preserved. For Shape, multiply by the actual sign/line conditional correction as well; a shared inner draw for duplicate outer IDs changes covariance, not the mean correction. Record joint/conditional probabilities, not a product of marginals that was never the actual sampling design.

### Pair hierarchy and positive support

For unordered field pair {a,b}, let v be its unique lowest common ancestor in the frozen binary tree. A group proposal may choose v with probability rho_v, then one field from each of its two disjoint children with known conditional probabilities. This gives a canonical ownership path and an evaluable p_group({a,b}); no duplicate ordered path is omitted. Mix it with uniform implicit-pair sampling and a bounded cache proposal. If an implementation instead uses overlapping groups, sum every path that can produce that pair.

Start with epsilon=0.25 global uniform mass. Empty learned/cache components return their probability mass to a valid normalized component. Maintain O(C) hierarchy/state plus a bounded sparse cache; use implicit pair rank/unrank and never materialize all pairs just to draw. Full all-field model/anchor work still scales with C and support. Explicitly test C=1,2,5,20,100 and probability/index behavior at C=1000.

An optional cyclic pair c can be an exact outer stratum. For target weights w_q, evaluate w_c times its complete or unbiased inner score, then sample other pairs from p_minus(q)=p(q)/(1-p(c)), q != c:

\[
\widehat L_{\rm pair}=w_c\widehat\ell_c+
\frac1k\sum_{j=1}^k\frac{w_{q_j}}{p_{\rm minus}(q_j)}\widehat\ell_{q_j}.
\]

The singleton population needs no complement. Advance the cycle only on accepted updates and retain its RNG/cursor. One outer pair per accepted update gives a full cycle after C(C-1)/2 accepted updates, not a field-count-independent coverage guarantee. Do not add this contribution to an estimator that already includes c in its full target. Coverage is useful large-C engineering, not the established cause of the five-field training bottleneck.

## 4. Signal/cost allocation without expensive self-deception

At fixed parameters and optimizer state, separate query noise from condition/member variability by holding or resampling each deliberately. Compute query-gradient variation on shared endpoints. On selected ordinary graphs, use autograd only to endpoint fields or pre-reduction band/filtration inputs, then project/gather a small fixed seeded sketch. This avoids repeated full-model backpropagation, but is only a proxy. The derivative with respect to the final scalar descriptor alone can be constant/zero under J2 absolute CRPS and omits the descriptor Jacobian; do not use it as the signal. Correct sampling when estimating stratum variation from observed queries. Audit the proxy with sparse true parameter-gradient or Jacobian-aware calculations on representative TRAIN graph batches. Full gradients, not sketches, are passed to ConFIG.

Under independent strata, fixed weights and additive costs, the ideal fixed-cost count is

\[
n_h\propto W_h\sigma_h/\sqrt{c_h}.
\]

Here sigma_h is the query-sampling standard deviation conditional on shared condition/member endpoints, averaged over the declared TRAIN panel. Shared batch/member variability does not decrease as 1/n_h merely by adding queries. Derive this variance/cost allocation, impose integer minima and measure whether the assumptions approximate the actual workload. Signed Scale and Shape share endpoints and may be a joint bundle; account for covariance/shared cost instead of assigning the same feature cost twice. Complete-anchor/model cost is a fixed per-update term, not an optional sampled query cost.

Within a stratum a priority proportional to target weight times estimated gradient second-moment square root divided by square-root incremental cost is a useful starting upper-bound heuristic. It is not a proof of optimality for ConFIG under shared costs. Shrink low-count estimates toward a stratum prior, use detached EMA statistics and a bounded cache, and retain exploration. A loss-only “difficulty” rename does not qualify as the new method.

Refresh every ten completed epochs, commit statistics only after accepted updates, and journal cost quantiles, visitation, proposal tails, effective weights, clipping/rejections and gradient audit agreement. Profile synchronized bundles sparsely and cheaply; do not synchronize every leaf or add a full model forward to every priority update. Include all probes and policy time in the 5% overhead target.

Benchmark uniform, the C8-style difficulty proposal where applicable, and the candidate on the identical frozen expanded target and endpoint bank. Use exact small-population expectations and replicated estimates with a fixed seed manifest. Compare gradient MSE at equal charged compute or MSE times estimator cost. Adaptive sampling is useful only when its saved variance/progress exceeds screening and update overhead. If not, freeze uniform for the expanded target and report the allocator rejection separately.

## 5. Compile immutable observation work

Factor a deep endpoint-feature/score kernel shared by training and the no-gradient evaluator. Reference compilation binds sample/support identity, fields, units, geometry/masses, fitted normalizations, query bank, backend/dtype and source/observer version. Cache immutable reference sorts and cumulative masses, spectral coefficients/means, raster maps and persistence diagrams. For predictions, cache only within the current endpoint evaluation/live graph.

Do not cache a reference-only final quantile grid: merged breakpoints depend on all actual compared endpoint CDFs. Deduplicate frozen Shape pack queries while preserving summed multiplicity. Endpoint features are computed once; complete joint/log-power scores couple truth and all members and are computed once per condition/ensemble. Extend all-member fair scoring to J4.

The frozen observer currently runs under no_grad and detaches results; prove value parity there, and every-member/parameter-gradient parity through the differentiable kernel. Preserve mutation detection unless private immutable ownership replaces it with equivalent integrity. Reference caches need mask/boundary/raster/calibration identities, not a filename-only key. No stale trainable tensor survives an optimizer update.

Benchmark full cold/warm callbacks, model-independent scoring, host pairing/transfer, allocations and persistent storage. An exact observer improvement may be adopted independently of a failed learned candidate. Both mature arms use the same correct compiled implementation; report the separate old-versus-new engine benchmark rather than attributing all engine benefit to the new objective.

## 6. Qualify the two real model contracts

EDM full-grid rollout comes before score-support gathering. Source fitting, Scale geometry/full-grid calibration guards, observation consistency and inference precision must follow the chosen native function. A source-admission snapshot is a generic immutable-checkpoint feature, not a false completed-run flag. Materialize EMA consistently, then disable child EMA explicitly after strict auxiliary loading.

For exact low-memory backpropagation, either checkpoint pure denoising segments with preserved random/mode state, or calculate each family's endpoint vectors and recompute each member's native rollout to accumulate that family's full parameter gradient. Sum member contributions **before** applying parameter-space ConFIG/fidelity correction, clipping and AdamW; applying ConFIG to endpoint vectors and backpropagating once generally changes norms/angles and is a different update. Risk gradients remain separate as required by the policy. Recompute the same member, not a new independent draw. Compare output/family/parameter gradients, actual optimizer displacements and unchanged resume; charge replayed work.

RFF frequencies/phases and EDM noise schedule/initial noise are scientific coupling artifacts. The requested GLRbfCQ source's actual checkpoint buffers are `prior.omega` and `prior.phase`; hash those tensors rather than assuming another adapter's names. Matching seeds is insufficient if buffers, query order, dtype, solver or clamping differ. Across compared models pair the inputs within member j; members j and k remain independent conditional on observations. The frozen SOURCE, initial child and native reconstruction must agree before any accepted update.

Use CPU analytic controls plus both real backends for conditional scoring, physical risk and live gradients. The typed deterministic J1 path remains valid and receives focused contract tests; no third mature model is added. The specified EDM qualification establishes backend execution, not a deterministic or cross-physics transfer claim.

## 7. Evidence, proofs and implementation seams

The workplan's required proof/check set is: physical-mass/fair-score identities; retained-Scale sensitivity/margin bounds; complete-leaf and hierarchical/cyclic expectations; proposal/inner-duplicate gradient treatment; bounded state/indexing; exact cache values/gradients; source/native parity and accepted-update/resume behavior. Include zero/one/small populations, unequal masses, inactive blocks, finite signed mean-risk values and near-zero source references.

Implement through `coherence/query_sampling.py`, `coherence/query_runtime.py`, `coherence/state_compute.py`, `coherence/scale_compute.py`, `coherence/shape_compute.py`, the existing multiscale/topology feature providers, `training/scalable_prefit.py`, `training/scalable_fitting.py`, `training/scalable_runtime.py`, `training/fidelity_budget.py`, `training/rollout.py` and `evaluation/frozen_coherence.py`. Factor real shared boundaries rather than add a parallel training framework. Configuration/version names should distinguish changed targets and exact execution optimizations.

Primary methodological references: [proper scoring rules](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) motivate represented-law scoring; [importance sampling with gradient-norm proxies](https://proceedings.mlr.press/v80/katharopoulos18a.html) motivates cheap signal tests and overhead accounting; [online variance reduction with mixtures](https://proceedings.mlr.press/v97/borsos19a.html) motivates adaptive mixture weights; [EDM](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract.html) motivates the native sampling design. None establishes efficacy for the current scientific task, topology composition or constrained ConFIG trainer.
