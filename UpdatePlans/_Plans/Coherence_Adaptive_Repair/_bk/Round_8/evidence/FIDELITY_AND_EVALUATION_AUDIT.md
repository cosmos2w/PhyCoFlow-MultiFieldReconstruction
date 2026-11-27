# Round 8 fidelity and evaluation audit

Date: 26 November 2026. Scope: planning and read-only review of the supplied discussion, revised manuscript foundation, present shared implementation and sealed Round 7 matched-100 evidence. This audit authorizes no training, model inference, metric evaluation, active-run inspection or change to a historical decision. Only this new audit file is written. Code line numbers refer to the inspected working tree at HEAD `b5bbf8fce710746295853841473596c9832e290a`, including the concurrent uncommitted Round 7 implementation; they are not a claim that all inspected code belongs to that commit.

## 1. Recommended decision

Use **physical-measure-weighted member MSE** as the prospective default fidelity estimand when the requested promise is “the reconstructed fields' MSE may worsen by no more than x%.” For each field, constrain its pooled risk relative to SOURCE under the same condition, observation, reconstruction and measure law. A deterministic model supplies its genuine single output, so the same definition naturally supports J1. Keep corrected conditional-mean risk, finite-ensemble-mean MSE, member spread and proper conditional feature scores as separately named diagnostics. A task delivering an ensemble mean can choose its corresponding mean-output risk prospectively, with its member count stated.

Use a frozen, approximately 10% TRAIN calibration for affordable training feedback and a separately frozen, matched SOURCE baseline for each assessment split/protocol. Do not feed VALIDATION risks into the training controller. A SOURCE forward on every training action is an optional paired-estimation alternative, not a default prerequisite. Preserve the original signed-risk allowances, verdicts and field safeguards for Rounds 1–7. A new risk contract is a prospective methodological choice, not grounds for retroactively promoting an existing child.

Expose direct SSS **reference discrepancies** beside attraction, positive separation and fair scores. These answer complementary questions; a lower direct discrepancy does not certify a calibrated posterior or intrinsic physical validity. Retain unoptimized regional and joint-event tasks, native field errors and training costs as the practical adoption evidence. Do not make every diagnostic an additional pass/fail gate or stop useful learning because an early noisy estimate violates a constraint.

## 2. What the present implementation actually protects

The revised guide, `UpdatePlans/Coherence_Manuscript_Foundation/00_Overall/MANUSCRIPT_GUIDE.md`, lines 88–104, and theory, `UpdatePlans/Coherence_Manuscript_Foundation/01_Theory/LPC_THEORY.md`, lines 445–483, correctly distinguish the current mean-risk contract from ordinary member MSE. These are local locators into the legacy DATA2-backed manuscript foundation, not portable document dependencies. The present code does not contain an arithmetic mistake merely because those contracts yield different verdicts.

| Current object | Inspected implementation | Consequence |
| --- | --- | --- |
| Physical field MSE | [physical_risk.py](../../../../../../src/phycoflow_reconstruction/coherence/physical_risk.py), lines 9–39 and 60–106 | Integrates squared errors against physical masses; sampled support uses actual proposal/inclusion probabilities and the native total mass, rather than the sampled mass sum. |
| Member and corrected mean risks | Same file, lines 42–57 | Returns member MSE and a signed finite-member estimate of population conditional-mean risk. The helper currently rejects J1 because its second return value needs at least two independent members; ordinary MSE itself does not require J2. |
| Nonnegative risk budget | [fidelity_budget.py](../../../../../../src/phycoflow_reconstruction/training/fidelity_budget.py), lines 149–197 | Implements `upper=(1+epsilon)*reference+delta` and a separate denominator floor. The floor already does not enlarge the accepted upper bound. |
| Current signed budget | Same file, lines 231–272 | Implements `(risk-reference-absolute_allowance)/risk_scale`; signed samples may be negative. |
| Current PI type restriction | Same file, lines 341–353 | Explicitly requires `MeanRiskDifferenceBudget`, although its feedback calculation needs normalized violations rather than a mathematically signed-risk-specific operation. |
| Current primal composition | Same file, lines 572–615 and 759–777 | Three parameter-space ConFIG rows are followed by one multiplier-weighted field row, clipping/AdamW and projected PI. This is not one scalar SSS optimization objective or a hard feasibility projection. |
| Current rotating field adapter | [fidelity_quadrature.py](../../../../../../src/phycoflow_reconstruction/training/fidelity_quadrature.py), lines 27–47, 59–64 and 99–116 | Fixes Q4096/J2 and selects the signed-risk return. It already preserves original observations and paired member realizations across the selected and complement rollouts. |
| Metric override in the trainer | [post_training.py](../../../../../../src/phycoflow_reconstruction/training/post_training.py), lines 3197–3215 | The complement branch replaces the earlier risk selection. Changing only `metric` in configuration cannot make that branch a generic member-MSE/J1 path. |
| Training calibration | [car_round7_training_reference.py](../../../../../../scripts/qualification/car_round7_training_reference.py), lines 81–102 and 159–191 | Measures both risks using the same selected/complement quadrature, sums condition risks and divides by the number of conditions; the final partial batch is correctly sample weighted. |
| Held-out field assessment | [assess_car_round7.py](../../../../../../scripts/posttraining/assess_car_round7.py), lines 444–480 | Uses a matched native SOURCE bank, carries the fixed TRAIN-derived absolute signed allowance into physical units, and separately reports the ratio of pooled member MSEs. |

Round 7 has already corrected the previous reference-law mismatch. Its sealed TRAIN800 calibration uses training observations, independent J2 members, selected support plus a rotating 512-point complement and model-field units squared. The saved `risk_reference_panel.observation_law` is `random_uniform`, seed 42, with temperature T as the only observed field and a count range of 192–384; it is not a richer all-field observation regime. It separately keeps native J4/T256 assessment denominators. This correction must not be proposed again as an unperformed repair or asserted to explain the remaining held-out result. See [Round 7 workplan](../../../_bk/Round_7/WORKPLAN.md), lines 17–31, and the local locator `UpdatePlans/Coherence_Manuscript_Foundation/00_Overall/MANUSCRIPT_EVIDENCE.md`, lines 91–101.

## 3. Three legitimate risks, three different questions

Let Y contain the supplied observations and conditioning information, Z the paired reference and X_j independent conditional model draws, independent of Z given Y. For one field define the physical norm `||u||_w² = sum_i w_i u_i² / W`, with `W=sum_i w_i>0` on the declared mask. Let `m_theta(Y)=E[X|Y]` and `V_theta=E||X-m_theta(Y)||_w²`. All expectations below include the declared condition and observation law, not just latent randomness.

| Estimand | Definition and identity | Intended output |
| --- | --- | --- |
| Member risk M | `E[J^-1 sum_j ||X_j-Z||_w²] = R_mean + V_theta` | Accuracy of a generated field, averaged over conditions and draws. J changes estimator precision, not this population target. |
| Finite-J mean risk R_mean,J | `E||J^-1 sum_j X_j-Z||_w² = R_mean + V_theta/J` | Accuracy of the actual J-member average. J is part of the scientific contract. |
| Population conditional-mean risk R_mean | `E||m_theta(Y)-Z||_w²` | Accuracy of the model's ideal conditional mean; it is not the error of one generated field. |

The signed estimator subtracts `sum_j ||X_j-Xbar||_w²/[J(J-1)]` from `||Xbar-Z||_w²`. Its expectation is R_mean, and at J2 it is the physical inner product of the two member errors. It can be negative on a finite condition/member sample. Taking its positive part before averaging changes the estimand and biases the feedback; dividing by a signed SOURCE estimate can be unstable or reverse interpretation. Keep a positive independent normalization when this risk is used.

Member MSE has a real tradeoff: it penalizes generated spread as well as mean error. With SOURCE equal to the deterministic true conditional mean and conditional truth variance V, SOURCE member risk is V whereas an independent exact posterior draw has risk 2V. A member-MSE allowance below 100% excludes that ideal posterior in this particular example. The current signed mean risk equals V for both and avoids this variance penalty. These identities justify offering explicit task-dependent contracts; they do not establish that current SOURCE is a true conditional mean or that current spread is calibrated. The local theory cited above proves the example, so it should be retained with its assumptions.

For a deterministic endpoint, member risk, population mean risk and finite-J mean risk all coincide with its ordinary MSE. A deterministic model must not be represented by pretending that duplicated outputs are independent posterior members. A stochastic J1 realization still estimates member MSE; it supplies neither a variance correction nor a finite-member fair-score correction.

For context, proper conditional distribution scoring is a separate established construction. Euclidean feature energy scores target the corresponding feature law under their moment assumptions; fair ensemble estimation corrects the finite-member pair coefficient. See [Gneiting and Raftery (2007), Section 4.3](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf) and [Ferro (2014)](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.2270). These results do not turn MSE retention, direct reference agreement or the composed ConFIG/AdamW update into posterior-calibration guarantees.

## 4. Quantified sealed examples: why the distinction matters

No predictions were generated or rescored for these tables. The first two tables use scalar arithmetic on the sealed Round 7 calibration and matched-100 assessment JSON. Their source calibration, reference, uncertainty and assessment hashes were checked. The last example copies the frozen Round 6 supplement's published field table and calculates percentages from its rounded values.

### 4.1 Same matched-100 child, different fidelity questions

Reused VALIDATION100, native physical support, J4/T256; Joint Anchor at epoch 100 versus matched SOURCE. Field order is CH4, CO, T, U_1, p. “Signed/allowance” uses the fixed new Round 7 TRAIN allowance. Member intervals below are the already saved paired condition-bootstrap intervals, not newly computed intervals.

| Field | Member MSE increase | Saved 95% interval for member-MSE ratio | Signed-risk difference / allowance |
| --- | ---: | ---: | ---: |
| CH4 | +4.027% | [1.02267, 1.05917] | 2.134 |
| CO | +9.076% | [1.07470, 1.10658] | 3.662 |
| T | +8.623% | [1.06745, 1.10719] | 6.404 |
| U_1 | +2.695% | [1.01381, 1.03954] | 5.394 |
| p | +2.746% | [0.99149, 1.06372] | 8.853 |

All five member-MSE point ratios are below 1.10, while all five signed point allowances fail. CO and T member-ratio intervals cross 1.10, so “the sampled point estimates meet the prospective member rule” is supportable; “all population risks are certified within 10%” is not. The signed failures also coexist with unchanged observed-temperature sensor MSE. Member-MSE feasibility alone would not promote this child: complete expanded Scale is +0.1526 SOURCE units, raw joint Scale +0.3049, State +0.07553 and Shape -0.2923 on the same development panel. The sealed matched-100 interpretation at local locator `UpdatePlans/Coherence_Manuscript_Foundation/00_Overall/MANUSCRIPT_EVIDENCE.md`, lines 80–89, remains applicable.

### 4.2 The TRAIN absolute allowance is much smaller than 10% of held-out SOURCE risk

| Field | TRAIN SOURCE member MSE, model-unit² | Native VALIDATION SOURCE member MSE, dataset-unit² | Carried signed allowance, dataset-unit² | Allowance as % of native VALIDATION SOURCE member MSE |
| --- | ---: | ---: | ---: | ---: |
| CH4 | 0.01048034 | 8.588254e-5 | 7.471045e-7 | 0.8699% |
| CO | 0.06690807 | 4.217865e-5 | 6.011469e-7 | 1.4252% |
| T | 0.01094128 | 74408.64 | 657.8866 | 0.8842% |
| U_1 | 0.04293542 | 1183.267 | 5.948504 | 0.5027% |
| p | 0.01357748 | 31801104 | 69296.30 | 0.2179% |

Using the saved conversion, native VALIDATION member risk is approximately 11.50, 7.02, 11.31, 19.89 and 45.89 times the corresponding TRAIN calibration member risk. That comparison combines split, observation and support laws. A more informative decomposition is available from the existing native TRAIN800 SOURCE bank:

| Field | Native TRAIN800 SOURCE member MSE / training-law calibration member MSE | Native VALIDATION100 SOURCE member MSE / native TRAIN800 SOURCE member MSE |
| --- | ---: | ---: |
| CH4 | 1.0338 | 11.119 |
| CO | 1.0462 | 6.706 |
| T | 1.0326 | 10.953 |
| U_1 | 1.0292 | 19.327 |
| p | 1.0428 | 44.010 |

Both native panels have fixed T256, independent J4, native support, the same SOURCE checkpoint, EMA-materialized view, normalizer and field order. The training-law calibration uses only T192–384, so T256 is inside its count range; it does not change observed physical field types. J2 versus J4 changes member-MSE Monte Carlo precision, not its population target. On the observed means, moving from calibration to native TRAIN adds only about 3–5%, whereas the large SOURCE-risk difference is already present between native TRAIN and VALIDATION panels. This is a measured panel SOURCE-error gap, not a causal diagnosis of why the validation frames are harder. It does not prove that counts, support or finite member sampling have no effect. The signed target also differs from member MSE, so the preceding allowance percentages are a scale illustration, not a conversion between identical constraints.

The checked native SOURCE receipt uses checkpoint `2d1da713538f6e9613560f3a7eb4a9df2a95d1ad1f56dd11a1a8a4f9ef65050c` and normalizer digest `50c5e65e563fee03f4c15df336e064753981f8555a06e7465a4722553b3746f1`, both matching the TRAIN reference binding. Recomputing each candidate/SOURCE ratio from the sealed pooled physical member risks exactly reproduces the saved ratios in Section 4.1. The comparison does not substitute ensemble-mean MSE for member MSE. The conversion path is explicit in [assess_car_round7.py](../../../../../../scripts/posttraining/assess_car_round7.py), lines 265–274, and the member reduction in [round2_assessment.py](../../../../../../src/phycoflow_reconstruction/evaluation/round2_assessment.py), lines 36–52.

Changing only the feedback denominator cannot change these accepted absolute budgets. A prospective “same-split SOURCE member MSE plus 10%” contract changes the protected risk and its allowed excess; both must be named explicitly. Current native-field records use denormalized dataset values, but the sealed ordered unit metadata is `unknown` for all five fields. Retain the computed quantities, label them dataset-unit² and repair the metadata from authoritative dataset definitions before making physical-unit claims; do not infer units from a familiar field name.

### 4.3 The supplied phase-225 example is numerically consistent

From the [Round 6 Scale supplement](../../../../../_Reports/Coherence_Adaptive_Repair/Round_6/Scale_Alignment_Followup/report.md), lines 124–130, prospective member-MSE changes are approximately +3.95%, +7.63%, +7.37%, +5.50% and +6.69%, while signed difference/allowance values are 0.376, 1.083, 3.809, 8.629 and 24.871. The discussion's arithmetic is consistent with the rounded published data. Pressure exceeding a small signed allowance nearly 25-fold and member MSE rising only about 6.7% can both be true. The historical result still had unfavorable State, complete Scale and regional-task effects and remains unpromoted.

## 5. Prospective generic fidelity contract

### 5.1 Primary estimand and allowance

For each protocol p and field f define

`M_f^p(theta) = E_{condition, observation, latent} [sum_i w_i (X_theta,if-Z_if)^2 / W]`.

Declare the condition weights, observation law, field units, physical mask/mass, reconstruction solver, observation-consistency operation, latent/member law and live/EMA endpoint. Keep SOURCE fixed under the same protocol. The desired constraint is

`G_f^p(theta) = M_f^p(theta) - (1+epsilon_f) M_f^p(SOURCE) - delta_f <= 0`.

Here epsilon is the stated relative MSE allowance and delta is an explicitly justified absolute allowance in that field's squared physical units. Use delta=0 when no nonzero absolute tolerance is authorized. A 10% MSE allowance corresponds to about 4.88% RMSE increase, so figures and prose must not exchange those percentages. No cross-field averaging may conceal a failed field condition.

For numerical conditioning use `e_f=G_f/s_f`, with frozen positive `s_f=max(M0_f, scale_floor_f)`. The floor affects feedback units only. At exactly zero SOURCE risk, a strict delta=0 contract permits zero child risk; an arbitrary numerical floor must not silently relax that scientific requirement. If a meaningful nonzero absolute tolerance is needed, declare it before evaluating candidates, retain its units and report it independently of the numerical floor.

TRAIN and VALIDATION use the same risk definition and epsilon semantics but their own SOURCE baselines. This is a relative-retention question under each declared distribution, not a promise that absolute training budgets transport unchanged. Training references use only TRAIN. A diagnostic seeking to isolate split transfer should match TRAIN and VALIDATION observation/solver/support laws; otherwise label the combined shift. Do not add full-population evaluation or TEST access to make these definitions convenient.

### 5.2 Estimate pooled risks, not averages of random ratios

Average each condition's mass-normalized MSE over its members, then pool conditions using the declared weights. For an equal-condition panel, `Mhat=sum_c Mhat_c/n`. A final smaller batch contributes its actual condition count. Multiple observation masks should be averaged within a state when the desired target weights states equally; pooling every mask as a separate state changes the target if mask counts differ. Do not let a condition with more sampled members receive more condition weight.

The reported retention ratio is `Mhat_child/Mhat_SOURCE`, a ratio of pooled risks. It is not `mean_c(Mhat_child,c/Mhat_SOURCE,c)`, which instead overweights easy conditions and may be undefined. A ratio of noisy pooled estimates is generally biased in finite samples; that does not make it an invalid descriptive estimate, but it is not an unbiased stochastic constraint. For training use a difference with a frozen scale. For assessment also report the paired absolute margin `Ghat=mean_c[Mhat_child,c-(1+epsilon)Mhat_SOURCE,c]-delta`; it avoids a random denominator and supports meaningful near-zero interpretation.

Apply spatial sampling correction to squared errors before pooling. Uniform sampling without replacement needs inclusion probability Q/N; nonuniform designs need their actual inclusion probabilities. With-replacement draws need proposal probabilities and draw-count normalization. Preserve the native W, and do not replace the integral by a self-normalized sampled-mass ratio while claiming it is the same unbiased risk. Current physical MSE already implements these alternatives. They are unbiased for a coherent fixed endpoint field under the stated sample design; if the prediction changes with the full query set, the target includes that query-support reconstruction law. A single support-consistency fixture is not a population invariance theorem.

For affine constant field normalization `Z_physical=sigma_f*Z_model+mu_f`, MSE and absolute allowances convert by sigma_f², while same-contract SOURCE ratios are invariant. If normalization varies spatially or is nonlinear, compute physical residuals before integration rather than applying one fieldwise conversion. Coupling across models may share random seeds to reduce comparison variance; members within each generative model must retain their declared conditional independence, and model draws must not be coupled to hidden reference information.

### 5.3 Recommended training estimator and affordable alternative

**Default: frozen TRAIN calibration.** Use one reproducible approximately 10% TRAIN panel under the actual training observation/member/spatial law, then freeze its ordered field means, uncertainty and source identity. Train with `(Mhat_batch(theta)-(1+epsilon)M0_calibration-delta)/s`. This keeps the existing one-current-model forward path and one weighted field reverse pass. It controls a calibrated empirical reference; finite calibration error means it is not an exact guarantee against the unknown full TRAIN expectation. The existing Round 7 calibration pattern already provides the needed sample weighting and provenance. Reuse a valid matching calibration; recompute only a changed scientific contract.

**Optional: paired SOURCE on the same training conditions.** Use `(Mhat_batch(theta)-(1+epsilon)Mhat_batch(SOURCE)-delta)/s` with fixed s. Freeze SOURCE and detach its outputs; share observations, support, masks and optionally latent seeds across the two models. This estimates the expected relative excess without dividing by a noisy batch baseline. It can reduce variance of model differences and removes a fixed calibration-mean mismatch, but SOURCE generation adds forward compute and dynamic observations usually prevent reuse of an old bank. It is not a free execution-equivalent change. Use it only if a measured calibration mismatch matters enough to justify the cost; it must not become mandatory infrastructure before the ordinary member-MSE trial.

Neither route uses a batch risk ratio, differentiates through SOURCE or requires one backward pass per field. With fixed multipliers, one scalar `sum_f lambda_f e_f` gives one current-model weighted field VJP. Member MSE already uses the available live members; its output derivative is the ordinary `2 w_i (X_j-Z)/(J W)`, followed by the declared sampling factors. Preserve gradients through all supplied live members. Keep fresh SSS parameter rows and the same optimizer composition unless a separate measured failure justifies changing them.

## 6. Minimum implementation work, if execution is later authorized

1. Expose an explicit `member_mse` versus `mean_risk` risk choice from the existing physical quadrature. Support genuine J1 member MSE without evaluating the J>=2 correction; retain signed-risk behavior under its own identity. Decouple generic measure/sampling semantics from Q4096/J2 case-specific guards. A structured model may generate the declared grid then gather sampled risk sites; a query model must preserve observation/member identity across its requested sites.
2. Let the PI policy depend on the risk-budget interface (`violations`, ordered fields, frozen normalization/reference state) rather than the concrete `MeanRiskDifferenceBudget` class. Both budget types can initialize their normalized SOURCE violation consistently. Do not rewrite historical checkpoints or reinterpret saved columns. A changed scientific risk definition starts a named new run or explicit new phase with its own recording semantics; a permitted device change does not.
3. Carry ordered per-field absolute tolerances and unit metadata through the contract. The existing `FidelityBudget` has the correct floor-versus-bound distinction but a scalar absolute allowance; a generic physical contract can require different dimensional deltas per field. The fixed-source-sensor branch in [scalable_prefit.py](../../../../../../src/phycoflow_reconstruction/training/scalable_prefit.py), lines 110–122, also hard-codes field `T`. New generic use should resolve its observed-field role from explicit case/protocol metadata, retaining T256 as this case's declared scientific choice. Preserve historical manifests and avoid a mandatory backward migration of frozen artifacts. Keep these as small contract extensions, not a new optimizer framework.
4. Record the chosen risk, calibration or paired estimator, normalized violation, member MSE, signed diagnostic when available, multiplier and field-row norm. Separate batch estimates from assessment risks. Preserve ordered recording definitions, loss accumulation, every-ten-epoch curve refresh and the existing save/reload/append contract. Test only the changed J1/member sampling, budget/PI and real fresh-process resume paths against uninterrupted behavior; reuse unaffected Round 7 qualifications.

PI gains fit to the old risk response are not guaranteed to provide the same response for member MSE. Start from the existing affordable controller, using the same dimensional normalization, and inspect its actual successful-action response during qualification. Recalibrate only if the changed derivative produces a demonstrated inadequate response. Positive early violations, a noisy batch or weak short-horizon SSS movement are learning evidence to interpret, not correctness failures that automatically cancel the study. Nonfinite updates, wrong units/weights or broken resume state do require repair before continuing the affected path.

## 7. Direct SSS quality and unoptimized scientific evaluation

### 7.1 Keep three quantities visible

For every fixed family/semantic distance d_k report `A_k=J^-1 sum_j d_k(X_j,Z)`, positive `S_k=[J(J-1)]^-1 sum_{j<l} d_k(X_j,X_l)` and `F_k=A_k-S_k`. A_k is the direct paired reference discrepancy; S_k is half the average unordered-pair distance under this convention. If reporting the full mean pair distance, call it `2*S_k` explicitly. J1 has A_k only, separation zero and the corresponding deterministic distance. Do not substitute distance of the ensemble mean for the mean of member distances, because nonlinear representations do not commute with averaging.

Do not train only on A_k by default: linear expected paired costs can favor collapsed output distributions. Do not infer calibration from more spread or demand SOURCE spread as a universal target. Fair scores test represented conditional-distribution quality under their assumptions; direct discrepancies make scientific changes interpretable. [Pic et al. (2025)](https://ascmo.copernicus.org/articles/11/23/2025/index.html) provide the relevant transformation/aggregation framework and explain why propriety alone is insufficient for an informative diagnostic.

| Layer | Minimum direct readout | Reuse and interpretation |
| --- | --- | --- |
| State | Direct occupation-quantile discrepancy, marginal versus mixed directions where already defined | Reuse frozen weighted quantile transforms and A. Keep its root-norm versus squared-distance convention explicit. Spatial pooling leaves locations unobserved. |
| Self Scale | Band power/log-power differences and piece-mean errors per field | Reuse fixed spectral coefficients. Keep amplitude-sensitive quantities alongside normalized correlations. |
| Mutual Scale | Mean member absolute signed band-coherence error `mean_j,sum_(a<b),s omega_(ab,s)*abs(rho_ab,s(X_j)-rho_ab,s(Z))` | Use frozen power floors and existing `band_coherence`/`selected_band_coherence`. Keep actual pair/band weights; sampled queries need population inclusion weighting. A sampled pair matrix is not a complete N-field matrix. |
| Joint Scale | A, S and F for the retained joint coefficient/mean representation; preserve historical complete Scale definitions | Field/band/piece norm attributions are explanations of one joint norm, not separate proper scores or causal mechanisms. Fixed Q4096 is not the full native spectrum. |
| Shape | Self/mutual, finite/essential A, S and F | Existing compiled Shape constructs each endpoint diagram once and can return components. Global barcode similarity is insensitive to some locations; interpret spatial task errors separately. |
| Spatial scientific task | Fixed joint-excursion location/area and regional relation errors outside optimization | Native support and declared thresholds/masks; one existing regional covariance and one existing joint event suffice initially. Avoid a large new task catalogue. |

The existing code supplies the key computations: [state_compute.py](../../../../../../src/phycoflow_reconstruction/coherence/state_compute.py), lines 112–139; [query_runtime.py](../../../../../../src/phycoflow_reconstruction/coherence/query_runtime.py), lines 21–32; [multiscale losses](../../../../../../src/phycoflow_reconstruction/coherence/families/multiscale_spectrum/losses.py), lines 418–456; [scale_compute.py](../../../../../../src/phycoflow_reconstruction/coherence/scale_compute.py), lines 125–143; and [shape_compute.py](../../../../../../src/phycoflow_reconstruction/coherence/shape_compute.py), lines 284–328. Returning A/S for blocks that currently return F only is a small scorer change; do not regenerate model predictions for that reason.

The original epoch-500 expanded Scale example already shows why the decomposition matters: attraction changed -0.0711 SOURCE units while positive separation changed -0.3451, giving fair-score change +0.2740. Its A improvement interval includes zero, while its S reduction is clearer. This is not proof of uniform raw-field variance collapse: physical spread increased in all fields. Reuse the frozen supplement's interpretation rather than inferring a posterior property from one feature-space score.

### 7.2 Minimal assessment and figure package

Use the same reproducible approximately 10% VALIDATION sample IDs, observations, member manifest and native masks for SOURCE and the nominated raw child. Retain TEST lock. Add an approximately 10% TRAIN assessment only for a stated fitting-versus-transfer question, preferably reusing the already compatible TRAIN800 banks. Use genuine deterministic J1 or the declared generative member count. All models need the same physical risk and task semantics even when their conditional-score estimators use different genuine member counts.

Generate each necessary bank once; reuse native outputs for fields, tasks and native spatial displays. Reuse separately reconstructed selected-support banks for the frozen SSS geometry when native slicing is not equivalent. Cache SOURCE/reference transforms. For a first cached reappraisal, use SOURCE and one relevant sealed child; include an older phase only to answer a concrete historical contract question, not as an unrestricted checkpoint search. Complete query-population scoring on a 10% condition panel does not authorize evaluation of the full condition population.

| Figure/readout | Content | Scientific purpose |
| --- | --- | --- |
| Field retention | Per-field pooled member-MSE ratio and absolute paired margin, with mean-output MSE/spread separate | Shows the requested accuracy allowance in recognizable units. |
| Direct and conditional SSS | Matched A/S/F changes by family and Scale block, plus signed band pair matrices | Distinguishes reference agreement from distribution-spread changes and identifies affected field pairs. |
| Spatial fields | Reference, SOURCE, raw child and residual for fixed panel IDs; one explicitly labelled failure selected by a frozen median/worst rule | Shows whether scalar gains correspond to useful physical structures. Fix shared color limits; state member versus ensemble-mean output. |
| Unoptimized event and regional task | Native joint-event mask overlay and member event areas, plus regional covariance/reference scatter | Tests usefulness outside optimized descriptors. Use fair scalar scores for generative outputs and direct task errors for all models. |

Every reconstruction/error panel must display its own fieldwise relative L2 or RMSE on the displayed physical mask and declared units; a mean-output badge must not be presented as member MSE. Joint-event figures should state reference support and the empty-union convention. Use per-member masks for member-based scores rather than thresholding an ensemble mean. Keep these tasks outside controller tuning and checkpoint selection; reused historical tasks are unoptimized development evidence, not newly independent confirmation. Existing task machinery is in [assess_car_round7.py](../../../../../../scripts/posttraining/assess_car_round7.py), lines 510–549. No new deployment benchmark is needed.

## 8. Statistical interpretation and bounded adoption

Approximately 10% is an evaluation-coverage rule, not a 90%, 95% or universal generalization guarantee. Equal condition weighting estimates the declared panel/distribution average, not worst-case per-condition retention. Pair SOURCE and child on identical IDs and observations; resample those units together when summarizing uncertainty. Include latent/member Monte Carlo limitations rather than counting J members as independent physical conditions. Freeze the endpoint before final task readout.

Both sealed TRAIN800 and VALIDATION100 panels contain only `trajectory_000000` sample-ID prefixes. Their saved condition-bootstrap intervals describe variation across the reused frames, conditional on the fitted endpoint and fixed generated bank. They do not establish independence of neighboring frames, unseen-trajectory transfer or training-seed robustness. A trajectory bootstrap cannot manufacture multiple independent trajectories from one. If temporal dependence affects the conclusion, a bounded time-block sensitivity on the saved panel can be informative, but do not invent a block length or an effective independent sample size. Reusing VALIDATION for development further limits confirmation claims.

For descriptive retention, report the point member-MSE ratio, paired excess margin and interval. If a formal one-sided noninferiority claim is requested, state the intended simultaneous/per-field confidence rule and sampling assumptions in advance; an interval crossing the allowance means the data do not certify that claim. Do not silently replace the user's pointwise empirical adoption rule by five stricter upper-confidence gates. Conversely, do not describe point feasibility as a population theorem. A train expectation constraint, successful PI feedback and a fixed-parameter unbiased estimator provide no automatic held-out feasibility or adaptive optimization guarantee.

At the planned learning review, assess whether field margins, direct/fair SSS and training trends leave a useful bounded recovery question. A 100-epoch learning review and provisional 500-epoch endpoint remain reasonable for this kind of pretrained repair; preserve the main plan's explicit budget and continuation rule. Do not require final success during the execution pilot, stop on one noisy field violation or extend a tied/uncompetitive route indefinitely. At the endpoint adopt only a worthwhile held-out field/coherence/task tradeoff, otherwise retain SOURCE and close the tested recipe. A useful executor, a specific direct-relation gain and raw-model replacement receive separate decisions.

## 9. Cost and evidence identity

The sealed matched-100 scorer used two CPU threads. VALIDATION100 scoring receipts record approximately 68–156 s wall and 97–208 CPU-s per child; TRAIN800 records approximately 539–1268 s wall and 767–1681 CPU-s. These ranges include the existing frozen coherence/Scale calculations and reads, not new native-basis reconstruction or an unlimited new diagnostic suite. Different jobs experienced different costs; these observations are a forecast range, not a scientific method-speed comparison. Cached field reductions alone were 0.7–1.6 s on VALIDATION100. A small direct-A/rho readout should reuse those computations, while location tasks reuse native banks.

For planning, reserve at most one CPU-hour of wall time for a minimal SOURCE-plus-child cached VALIDATION reappraisal, including figures, and two CPU-hours if a justified TRAIN800 matched-law diagnostic is added. These are proposed ceilings, not measured new execution. No model-inference cost is needed when compatible banks exist. If essential banks do not exist, disclose the required GPU generation separately before launching; do not silently claim cached-only work or expand to the full population. Forecast the final queue tail and aim for assessment-added wall time below 20% of planned training wall time, while preserving required coverage. Use the existing bounded CPU lane rather than constructing a scheduler.

Keep future generated outputs on the DATA2 campaign, selected ordinary reading copies under the plan/report libraries and disposable scratch in a task-owned temporary directory. Keep every-ten-epoch curves independent of inference assessment cadence. Any future long-running job must follow the [Luna monitoring policy](../../../../../../AGENTS.md#mandatory-run-monitoring-policy); this audit neither launches nor monitors one.

Local raw evidence locators below are based at `cases/turbulent_combustion/runs/coherence_adaptive_repair/round_7/`. They identify DATA2-backed evidence, not portable publication dependencies.

| Sealed evidence | Checked identity/use |
| --- | --- |
| `assessment/TrainingLaw_Reference_Readout_v1/calibration_readout.json` | SHA256 `ccf189aff101fc41f9db252a55f50333a9d81e123ab70abb909a54459574844b`; ordered TRAIN risks, allowance and units. |
| `artifacts/TrainingLaw_SOURCE_Reference_v2/calibration.json` | SHA256 `44f4c9470cdf0bb8061b9c2e25fd106f3b61fef7bb064f9b7595775166912606`; underlying calibration bytes verified. |
| `artifacts/TrainingLaw_SOURCE_Reference_v2/reference.json` | SHA256 `a1061425a358eabc835e07b91fcb166890f0ea2f6b1a193beb3d9b38048712af`; frozen risk budget verified. |
| `artifacts/TrainingLaw_SOURCE_Reference_v2/uncertainty.json` | SHA256 `9a01f6110259f37e8532176b17cd2c787e62b6b31a232ad7e55da667da25cb16`; calibration uncertainty artifact verified. |
| `assessment/Matched100_Decision_Readout_v1/readout.json` | SHA256 `0a67ef203fd26c324dffcac26b07398563292217ebf98808e03fc1934c41606d`; ratios, saved intervals and matched panel identities. |
| `assessment/JointAnchor_Matched100_CPUScore_v1/validation/assessment.json` | SHA256 `2333a52dab88a757e15c15dee9e6631396c4a34bb762cf877d53ea0ed0106d55`; matched SOURCE/child physical means and dataset-unit metadata verified. |
| `assessment/JointAnchor_Matched100_CPUScore_v1/train/assessment.json` | SHA256 checked against the sealed matched-100 readout; native TRAIN SOURCE means used only for the contract-scale decomposition, with no new evaluation. |
| Matched-100 `Composite_Matched100_CPUScore_v1` and `JointAnchor_Matched100_CPUScore_v1`, each split's `receipt.json` | All four receipt hashes matched their readout references; measured two-thread scoring ranges above. |

The reused native SOURCE receipts are local Round 2 evidence locators: `cases/turbulent_combustion/runs/coherence_adaptive_repair/round_2/assessment/source_train_ffm_v1/receipt.json`, SHA256 `53c2062421c1d63ce0d81dc33c0af735527e8dd5ee8a221d6c436e89148ffbdd`, and `cases/turbulent_combustion/runs/coherence_adaptive_repair/round_2/assessment/early_closeout/validation/source_ffm/receipt.json`, SHA256 `471bb91325888546ca897cf79d3703973220cb557077b1467e378105a695211d`. Both receipt hashes and the hash-bound panel/spec references were verified during this audit. They carry J4, native support and the literal SOURCE identities above.

Inspected implementation SHA256: `physical_risk.py` = `5caa65fc2d469407eb3236ca55ef4509fe8899b5a9537b34b707126bd3816b93`; `fidelity_budget.py` = `24394387c1804add2552df789532cb9264d09a84052992db4382aa1d8d38ce2f`; `fidelity_quadrature.py` = `2583e0944dd2ee186a5c87f4089252fed0e9df33bbbab93ee9dcd07bfb9973b0`; `assess_car_round7.py` = `dc0fdf98fb232365cd7b58679eef98c7be6d3ec48eaf3d6e8ef6da760d1a20a5`. These identify this review snapshot while Round 7 closes concurrently; subsequent code changes should retain the semantic findings and refresh only affected line locators.
