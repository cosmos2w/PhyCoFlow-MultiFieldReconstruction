# Project reflection: a general multi-field reconstruction enhancement method

26 November 2026. Review snapshot: `Coherence_Adaptive_Repair`, HEAD `b5bbf8f`, plus the inspected uncommitted Round 7 implementation. Round 7 terminal assessment is not part of this review. This document accompanies the prospective [Round 8 workplan](WORKPLAN.md); it reports source inspection, sealed-result arithmetic and small synthetic CPU checks, not a new learned model. Three GPT-6.1 sol agents at maximum reasoning effort independently examined scoring/theory, support/Shape, and fidelity/evaluation, and their findings were reconciled here.

## 1. Decision and purpose

**Retain LPC as the theoretical organization, finite SSS as the observation design, and SSS-only parameter-space ConFIG with multiplier-based field feedback as the research route. Upgrade the contracts connecting those pieces before commissioning another long repair run.** The next useful work is explicit member-accuracy semantics, complete deterministic/generative execution, honest spatial support, one inexpensive positional Shape observation, and interpretable relationship evaluation. The framework has a defensible scientific basis. Reliable enhancement under the current comprehensive observables remains an empirical objective, rather than an established consequence of that basis.

The strongest criticism is not that the project lacks enough mathematical machinery. It is that several different objects have been called fidelity, query, Scale improvement or physical consistency. Their distinctions affect the optimizer, assessment and interpretation. A fixed TRAIN absolute signed-mean-risk allowance is not a 10% VALIDATION member-MSE allowance. Random descriptor queries are not random spatial nodes. A proper feature-distribution score is not the error of a particular physical relationship. Diagram similarity is not positional accuracy. These are consequential design choices, not merely naming problems.

The next round can finish successfully without formal training: a coherent, tested implementation and cached scientific reassessment are useful bounded outcomes. If those changes leave a concrete learning question, the plan permits a small matched study and an evidence-led extension. An optional component failing does not close independent work or the entire method.

## 2. What the history establishes

The October fidelity-budget route deliberately removed the model-native RF loss. Run 11/11f used three coherence directions and separate endpoint-risk feedback; omission of native supervision was an achievement in model-independent design, not a missing term to restore. The [historical results record](../../../../../docs/Posttraining_Results.md) preserves useful instances: Run 11 at epoch 1000 passed the broader panel's field bounds with 67.00% legacy topology improvement, although its original smaller panel failed; Run 11f's selected epoch 595 passed both panels with 66.29% topology improvement, while its epoch-1000 endpoint failed CO. These are scoped successes with selection and panel limitations. Today's Shape and risk definitions differ, so their percentages cannot be compared directly.

| Stage | What remains useful | What it does not establish |
| --- | --- | --- |
| October Run 11/11f and the retained C8 wrapper | Constrained relation improvement is possible in particular recorded protocols; native training loss need not be replayed | Sustained feasibility, universal improvement or success of every later observable |
| Adaptive Repair Rounds 1–3 | Expanded relation diagnostics, deterministic-adapter experience, field-feedback and memory lessons | A replacement for SOURCE/C8; SSS improvement alone can coexist with worse scientific tasks |
| Round 4 | Feasible teacher construction can still optimize poorly | Failure of all teachers or of learning a good teacher; students were never tested |
| Round 5 | Native-supervision-plus-SSS/local-filter recipes were uncompetitive | Rejection of the SSS-only Lagrangian mainline |
| Round 6 and Scale supplement | Aligned complete scoring, actual-action checks and spatial-feedback local benefits | Solved Scale transfer or restored all field guarantees |
| Sealed Round 7 preparation/matched100 | Corrected training-law SOURCE reference, a useful time-memory execution option, controlled joint/composite Scale comparison | A final epoch-500 verdict or proven joint-anchor superiority |

The revised `MANUSCRIPT_EVIDENCE.md` is the main chronology; [the earlier historical audit](../../Round_6/evidence/HISTORICAL_AUDIT.md) supplies the October provenance. Historical decisions remain intact. A prospective risk redesign must not retroactively turn rejected checkpoints into accepted models.

Round 7 also changes the efficiency diagnosis materially. Its balanced mature-epoch comparison measured 143.273 seconds for separate execution and 86.011 seconds for the factored/chunk-two/static-cache/checkpoint package, a 39.967% reduction on the same RTX 6000 Ada. Peak allocation increased from 17.85 to 37.02 GiB; reservation increased from 21.76 to 41.97 GiB. This is a useful package-level option where capacity permits. It is not a claim that three mathematical reverse signals became one backward, that factoring alone produced the gain, or that floating-point training trajectories are identical. Next work should reuse this result and measure its changed paths, rather than restart a generic executor search.

## 3. Scientific foundation and manuscript argument

### 3.1 Why the problem is real

At finite reconstruction error, fieldwise MSE fixes diagonal residual energies but leaves cross-field residual alignment undetermined. A joint scientific query can depend on those cross terms. Two methods with similar field errors can therefore disagree substantially on a scientifically meaningful relationship. Conversely, convergent paired fields do control sufficiently regular fixed queries; there is no universal opposition between accuracy and coherence.

The conditional-mean example remains helpful: a true state `(S,S²)` with `S=±1` has a nonlinear realization relation that its squared-error-optimal mean `(0,1)` does not preserve. This motivates distinguishing accurate point estimation from plausible joint realizations. It does not prove that early relation supervision is useless or that training from scratch necessarily fails. The defensible post-training argument is practical: start from a capable endpoint, spend expensive relation computations during a focused refinement, and express retention through outputs rather than RF/diffusion/architecture-specific losses. Prior scratch-training difficulties are relevant experience, but a stronger causal comparison needs its own identifiable matched evidence.

### 3.2 What LPC and SSS contribute

The revised LPC partition is an exact organization of a locally convergent mixed-cumulant expansion under its exponential-moment assumptions. Local includes mixed information at one site; Pair contains mixed second order at two sites; Collective contains the remaining higher-order multisite information. Those signed generating-function contributions are not themselves nonnegative losses, causal mechanisms or separately realizable physical laws.

SSS supplies finite, overlapping observations chosen to expose useful information: weighted value occupation, signed spatial organization and excursion structure. There is no one-to-one Local→State, Pair→Scale, Collective→Shape theorem. Matching the finite observations does not establish the full field law, all N-way interactions, governing equations, temporal consistency or causal coupling. These limitations are already handled well in the four revised chapters. The immediate documentation gap lies in propagating their exact definitions into the older computation and training chapters.

Use three distinct probability objects throughout the paper: the conditional reconstruction law at fixed observations; variation across conditions; and spatial occupation within one field. Signed spectral alignment within a snapshot must not be called posterior correlation. A law-level score cannot by itself establish correctness of one particular generated structure.

### 3.3 Established scoring theory and deterministic reduction

For a fixed representation and admissible distance, the lower-is-better score is

\[
S_d(q,z)=\mathbb E d(X,z)-\tfrac12\mathbb E d(X,X'),\qquad X,X'\overset{\mathrm{iid}}\sim q.
\]

Its independent-member estimator is

\[
\widehat S_{d,J}=A_{d,J}-R_{d,J},\quad A_{d,J}=\frac1J\sum_jd(X_j,z),\quad R_{d,J}=\frac1{J(J-1)}\sum_{j<\ell}d(X_j,X_\ell).
\]

This is established energy/CRPS and fair-ensemble scoring, not a new PhyCoFlow principle. The unordered-pair coefficient makes R half the mean pair distance. The relevant foundations are [Gneiting and Raftery (2007)](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf), [Ferro (2014)](https://empslocal.ex.ac.uk/people/staff/ferro/Publications/ferro2013.pdf), and transformed/aggregated scoring in [Pic et al. (2025)](https://ascmo.copernicus.org/articles/11/23/2025/). Fixed Euclidean/Hilbert norm representations have the required scoring basis; arbitrary dissimilarities, target-dependent normalization or squared norm substitutions do not automatically inherit the same identification result.

For a deterministic endpoint `q=δ_x`, the population separation is zero and the score is exactly `d(x,z)`. Implement a genuine J1 branch, rather than duplicate one prediction to satisfy a J2 assertion. A stochastic generator observed with one random member still does not provide an unbiased estimate of its full fair score. Current State/Scale/Shape already contain deterministic scoring branches; the budget/controller and complement adapter need narrow generalization before claiming a common complete training path.

| Current representation | What its score can identify | Important omission |
| --- | --- | --- |
| State: exact integrated weighted quantile functions with a joint Hilbert norm | Law of the retained occupation representation | Positions and unqueried field-space directions |
| Joint Scale: signed retained coefficients and connected-piece means | Law of the fixed joint vector | Omitted modes/support and different physical scales |
| Shape: fixed-angle cross-diagonal SW1 and essential births | Proper assessment of declared SW1/essential-birth features; joint diagram-law identification is not established | Physical position and unobserved diagram relationships |

State is not the earlier attachment's finite quantile grid, and current Shape is not a smoothed Betti vector. The [theory audit](evidence/THEORY_AND_SCORE_AUDIT.md) verifies these code distinctions and their primary references. The supplied Chinese discussion and the earlier three-page PDF are different inputs, with separate hashes.

### 3.4 What the paper should claim next

The paper's argument should run from finite-error scientific ambiguity, through fixed multi-field observations, to a model-independent constrained intervention, then demonstrate useful held-out relationships at retained field accuracy and measured cost. The contribution is the complete specified observation/repair workflow and its empirical effectiveness; established scoring theory supports it. Generic tensor interfaces and mathematical eligibility support applicability, while broad efficacy requires an actually different model and physical system. Round 8 should qualify the deterministic path, not pretend that a short adapter check supplies that efficacy evidence.

## 4. Fidelity: choose the output whose error is protected

Three risks answer different questions. With physical squared norm, model mean m and conditional model variance V, member risk is `M=E||X-Z||²=R_mean+V`; a K-member mean has risk `R_mean+V/K`; population-mean risk is `R_mean=E||m-Z||²`. The current signed independent-member correction estimates the last quantity. At J2 it is the inner product of the two member errors and can be negative on a finite sample. Its implementation is not erroneous merely because its verdict differs from ordinary MSE.

For the owner's stated reconstruction promise, the recommended prospective default is **expected physical member MSE per field**, which naturally includes deterministic models. It protects average member error over the declared condition/observation law, not every draw or condition. Preserve mean-output error and spread as diagnostics. Member MSE penalizes variance: if SOURCE were the exact conditional mean, an ideal posterior sampler could have twice its member risk. Thus this is an explicit accuracy-versus-distribution tradeoff, not a universal posterior-fidelity theorem.

The new constraint should mean exactly what the report says:

\[
M_{f,D}(\theta)\le(1+\epsilon_f)M_{f,D}(\mathrm{SOURCE})+\delta_f.
\]

Use the same risk definition and relative allowance on TRAIN and VALIDATION, with each split/protocol's own matched SOURCE denominator. TRAIN alone supplies controller calibration; VALIDATION remains assessment/development evidence. Default `epsilon=0.10`, `delta=0`, unless a meaningful fieldwise absolute tolerance is explicitly specified. A denominator floor conditions feedback; it does not enlarge the scientific bound. Ten percent MSE is approximately 4.88% RMSE, not ten percent RMSE. Report ratios of pooled risks and paired absolute margins, not means of per-condition ratios.

The sealed Round 7 Joint100 example exposes the distinction:

| Field | Member-MSE increase | Signed-risk change / Round 7 allowance | TRAIN allowance as % of VALIDATION SOURCE member MSE |
| --- | ---: | ---: | ---: |
| CH4 | 4.027% | 2.134 | 0.8699% |
| CO | 9.076% | 3.662 | 1.4252% |
| T | 8.623% | 6.404 | 0.8842% |
| U_1 | 2.695% | 5.394 | 0.5027% |
| p | 2.746% | 8.853 | 0.2179% |

All member point ratios meet a prospective 1.10 rule; CO/T intervals cross it, and State/Scale remain unfavorable. This is not retrospective promotion. More importantly, native matched-protocol VALIDATION SOURCE risks are 6.7–44.0 times native TRAIN SOURCE risks. Moving from training-law calibration to native TRAIN changes means only about 3–5%. The large gap is already present between the two native panels; the audit does not identify its cause. Stored unit metadata is `unknown`, so values must be described as dataset-unit squared until authoritative metadata supplies physical units. The full [fidelity audit](evidence/FIDELITY_AND_EVALUATION_AUDIT.md) contains checked sources, observation laws and uncertainty.

## 5. Spatial support: confirmed behavior and proportionate correction

Q4096 really is one fixed subset shared across conditions and epochs. The current eight groups vary field/pair/sign/line descriptors, not coordinates. The fixed-support mechanism predates recent repair rounds. It enables cached spectral and raster operators; LPC itself does not require it. The separate 10%-of-dataset assessment rule concerns conditions, not these spatial nodes.

Round 7 already supplements fixed SSS support with fresh C512 field-risk points per optimizer action and has corrected its SOURCE reference to that training law. Preserve this accomplishment. Generalize the dynamic field-risk interface to J1/JN and member MSE. Structured models can integrate or gather from an already generated native field; query models can reuse fixed SSS sites and sample the complement with inclusion weights. A wholly fresh field-risk subset is valid where the endpoint provider supports it. If all required field-risk nodes were already refreshed coherently, a special complement mechanism would indeed be unnecessary.

Changing only the spatial sampler flag cannot randomize current Scale/Shape safely: their fitted basis, interpolation and coordinate bindings would become inconsistent. Nor does unbiased point sampling imply unbiased nonlinear PH, quantile or feature-norm scores. Keep fixed honest SSS geometry initially. Use cached native outputs to measure fixed-support versus native risk, and sparse-to-raster versus native-to-raster discrepancies; separate sampling error from query-dependent model output changes. Only a consequential observed omission should justify one improved support or a small support library. A native spectral basis and every-epoch geometry rebuild are outside the default round.

## 6. Shape: add a small observation of position

Persistence deliberately forgets position. Existing creator-location matching is not a reliable minimal training fix: creator indices are discrete, are not always structure centers, and additive location costs can have no positional derivative under fixed pairing. Spatial matching literature confirms that the concern is real, including [Stucki et al. (2023)](https://proceedings.mlr.press/v202/stucki23a.html) and [Wen et al. (2025)](https://openaccess.thecvf.com/content/ICCV2025W/BIC/html/Wen_Topology-Preserving_Image_Segmentation_with_Spatial-Aware_Persistent_Feature_Matching_ICCVW_2025_paper.html), but their segmentation algorithms do not automatically transfer to this N-field scoring contract.

The proposed minimal addition reuses each selected scalar filtration F. At fixed TRAIN thresholds, form soft excursions and integrate them in a few smooth overlapping geometry-only windows. This measures where joint structures occur, adds no PH invocation and joins the existing Shape gradient row. Its finite Euclidean feature supports deterministic distance or the same fair conditional score. Keep the PH component separately visible. No full N-parameter persistence, fourth coherence family or field-specific chemistry rule is needed.

The [support/Shape audit](evidence/SUPPORT_AND_SHAPE_AUDIT.md) contains a completed CPU witness: two structures translated eight columns have PH distance exactly zero; an eight-window, three-threshold feature has distance 0.01235395 and a displacement derivative of 0.00129520 toward correction. Quarter-pixel motion is also detected, while two hard region bins miss the translation. Mask invariance, equality subgradient and saturation were checked. These establish a blind spot and a locally useful representation, not learned efficacy or general positional identifiability. Saturation makes one bounded TRAIN sensitivity check necessary.

## 7. Scale and training: retain the method, expose the actual mechanisms

Do not reduce Scale's history to a reporting mismatch. Under aligned definitions, held-out performance still deteriorated. In the original Round 6 epoch-500 expanded score, attraction changed about -0.0711, separation about -0.3451 and the net score about +0.2740. TRAIN aggregate improvement also concealed worsening joint/signed/correlation terms compensated by log-power improvement. Physical spread increased while selected feature separation decreased, so “global posterior collapse” is too strong a diagnosis.

Report three complementary quantities: direct paired relationship discrepancy A; positive separation R; and fair score S=A-R. Direct A is interpretable but can favor collapsed predictions and need not vanish for a perfect uncertain generator. For Scale, reuse signed band-alignment matrices, power errors and joint signed coefficient/mean discrepancies. Rho measures within-snapshot filtered alignment, not posterior covariance. Weak-band floors and actual pair/band weights must be visible. One aggregate pass/fail flag is inadequate.

Small-member cancellation is a real mathematical possibility: for J2, collinear endpoints bracketing the reference give zero norm score. It is not yet a diagnosis of these high-dimensional learned gradients. Existing J4 banks can support a cheap two-member-subset diagnostic; six subsets are correlated, not six independent replications. [AIFS-CRPS](https://www.nature.com/articles/s44387-026-00073-7) provides a relevant almost-fair precedent, but reducing separation changes the target and should not be the default fix. Preserve regular fair scoring unless specific evidence warrants a separately named experiment.

ConFIG separates directional conflict handling from loss units. In its ideal normalized-row geometry, positive rescaling can leave the conflict direction unchanged while affecting output length and subsequent field feedback; changing terms inside a family generally rotates its row. Fixed semantic scales and query weights still matter. Use declared physical/representation units, positive TRAIN calibration and transparent internal composition rather than requiring every scalar loss to have the same numerical size. Avoid a new continuously adapting weight scheme without a demonstrated need.

The executed method is three fresh parameter gradients, ConFIG, one multiplier-weighted field gradient, clipping/AdamW and projected PI. It is not gradient descent on a simple sum of SSS losses, and adding the field row or Adam moments can alter a nominal descent direction. PI is expectation feedback, not hard per-action feasibility. Actual displacement and finite field changes are the right diagnostics. The old epoch-ten-switch explanation is unsupported for the frozen-prefit R2/R3 implementations; warmup helped historical runs but never proved universal stability. Reuse Round 7's action/history audit and check only newly changed risk/Shape paths before blaming an inherited coefficient switch.

## 8. Evaluation that makes scientific improvement visible

Use one cached matched prediction bank per required endpoint/panel, with separate native-field and frozen-SSS support identities. The minimum scientific display has four parts: fieldwise MSE retention and uncertainty; direct/fair SSS with Scale components; registered reference/SOURCE/child fields and residuals; and an unoptimized joint event plus regional relationship. Select examples by SOURCE/reference rules, include a difficult example, and put a computed physical-mask RMSE or relative L2 on every reconstruction/error panel. Distinguish members, ensemble means and event probabilities.

New Shape windows/thresholds become optimized observations. The scientific task must use held-out-from-optimization definitions, not the identical regional feature under a new name. Existing regional covariance and joint-event tasks can remain useful development outcomes if excluded from tuning. They are not governing-equation certificates or untouched independent confirmation. No density-weighted flux or conservation claim should be inferred from a covariance plot.

Both reused TRAIN800 and VALIDATION100 panels contain one trajectory identifier. Condition-bootstrap intervals describe the saved frames/bank; they do not provide unseen-trajectory or training-seed replication. A trajectory bootstrap cannot manufacture independent trajectories. Report point margins and existing uncertainty; use a bounded temporal-block sensitivity only when dependence could change a decision. Do not turn five interval upper limits into unrequested early gates, or call point feasibility a population guarantee.

## 9. Recommended next round and practical endpoint

Round 8 should complete the cached reinterpretation, explicit risk/support contracts, deterministic/generative plumbing, positional Shape implementation and meaningful scientific readouts. It should propagate the revised theory into current computation/post-training documentation and preserve historical definitions. The default is a bounded engineering and evidence round; conditional learning asks whether the new accuracy contract and spatial observation produce useful relations, without changing Scale, controller, learning rate and geometry simultaneously.

The [workplan](WORKPLAN.md) gives the mathematical definitions, code seams, minimum checks, a six-GPU-hour preparation ceiling and an optional expansion to 24 total occupied GPU-hours only for a useful learning question. GPU2 is the runtime preference. Independent work proceeds before Round 7 closes; its terminal evidence changes only the endpoint comparison, common Scale choice and cost forecast. The final decision separates retained engineering, observed scientific gains and model replacement. SOURCE/C8 remains the operational reference until a newly specified evaluation demonstrates a worthwhile field/relationship/task tradeoff.

## Evidence and reading index

- [Theory and scoring audit](evidence/THEORY_AND_SCORE_AUDIT.md): exact score provenance, deterministic reduction, current code correspondence and primary references.
- [Support and Shape audit](evidence/SUPPORT_AND_SHAPE_AUDIT.md): fixed-node history, quadrature, raster/geometry limits and positional feature proposal.
- [Fidelity and evaluation audit](evidence/FIDELITY_AND_EVALUATION_AUDIT.md): checked risk tables, actual observation laws, pooled estimands, uncertainty and assessment costs.
- [Input identities](evidence/INPUT_IDENTITIES.json): the four revised manuscripts, supplied text and inspected working-code hashes. Manuscript paths are local evidence locators into the existing DATA2-backed collection, not portable dependencies of this bundle.
- [CPU reproduction](evidence/support_shape_cpu_diagnostic.py) and [saved result](evidence/support_shape_cpu_diagnostic.json): synthetic calculations only, with no model or dataset evaluation.
