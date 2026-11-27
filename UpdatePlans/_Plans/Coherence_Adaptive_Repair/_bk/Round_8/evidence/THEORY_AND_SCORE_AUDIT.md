# Theory and scoring audit for Round 8

26 November 2026. Planning and research review only. This audit reads the four revised foundation chapters, the supplied five-question note and the current scorer implementations. It introduces no model evaluation, training, runtime modification or experiment decision. Round 7's terminal evidence belongs to its owners; no ongoing-run status was inspected for this audit.

**Recommendation.** Retain the revised LPC and conditional-scoring foundations. Adopt an interpretable readout of paired feature discrepancy, member separation and fair score, together with explicit fieldwise accuracy and selected scientific queries. Preserve genuine deterministic J1 scoring and stochastic independent-member scoring in one framework. Make any next-round fidelity or spatial-observation change explicit in its new protocol. No new scoring theorem, optimizer redesign or rewrite of the revised LPC chapter is needed.

## 1. Inspected inputs and the meaning of the theory

The scientific reading inputs are `UpdatePlans/Coherence_Manuscript_Foundation/00_Overall/MANUSCRIPT_GUIDE.md`, `MANUSCRIPT_EVIDENCE.md` in the same directory, and `UpdatePlans/Coherence_Manuscript_Foundation/01_Theory/LPC_THEORY.md`, `LPC_EVIDENCE.md`. Their respective SHA-256 values at inspection are `83af3229362f18ff66e3d0f7c46612f6e971b98cf2900e8e104415046a02696a`, `d9a4e7d638bc0fc069ee8224120e9ef03a8cf74f2169ce91c307c9d915697e86`, `0fc1faaa88757d552d3f23800684cb66c7ce6679d9b85e89e8a1102dc17a8de9`, and `abc318538a6176cfa577dbeddc774e9529ec53448878fe8edbd8b9c941a49d31`. Repository HEAD is `b5bbf8fce710746295853841473596c9832e290a`, with concurrent uncommitted Round 7 work. Code pointers below describe the inspected working files, not an assertion that those bytes all belong to HEAD.

The supplied Chinese five-question note is `/home/wanglz/.codex/attachments/7dd00335-b37d-4154-b5b5-64009594dee9/Pasted text.txt`, SHA-256 `bc02939fe6d2ccaa51f21edce31177e26a0e787ec8100f7998306a0957d91f31`. It is distinct from the earlier three-page `method.pdf`, SHA-256 `9d18198bc35584631be3276f94a54e88b1617605d47ec0c844571732009ec7a6`, discussed in the foundation's attachment review. The external note's historical numerical claims are not newly measured results of this audit.

The revised foundations are scientifically sound at their stated scope. `LPC_THEORY.md:89–179` proves an exact partition of a locally convergent mixed-cumulant expansion under exponential moments. Local includes all mixed orders at one site; Pair is mixed second order at two sites; Collective contains the remaining mixed higher-order, multisite terms. These are signed generating-function contributions. They supply neither a nonnegative coherence objective nor a completeness or governing-equation theorem for SSS.

The finite-error premise is stronger and more useful than a universal accuracy-versus-coherence slogan: fieldwise MSE fixes the diagonal of the residual Gram matrix, while a linear joint query depends on its off-diagonal entries (`LPC_THEORY.md:291–323`). Continuous scientific queries can still become accurate as fields converge; threshold queries additionally need margin/support assumptions. SSS is a finite observation design with overlapping sensitivities, not a one-to-one implementation of LPC sectors (`LPC_THEORY.md:183–253`). The revised chapters already state these distinctions correctly.

Three probability objects remain separate: the conditional law at fixed observations, variation across conditions, and occupation of locations within one snapshot (`LPC_THEORY.md:255–285`). A snapshot's signed spectral inner product is not a posterior covariance estimate. A spatial conditional-mean curve is not a conditional reconstruction mean given sensors. These distinctions should be carried into the computation and post-training chapters rather than reopening the LPC partition.

## 2. Attraction–separation: provenance and exact contract

Use lower-is-better scores. Let $Y$ include sensor locations, observed channels and values, masks, and any declared observation noise. Conditional on $Y=y$, generated members are independent draws from $q_{\theta,y}$ and independent of reference $Z\sim\mu_y$. For a fixed observation map and distance $d$, write

$$
S_d(q_y,z)=\mathbb E d(X,z)-\tfrac12\mathbb E d(X,X'),\qquad X,X'\overset{\mathrm{iid}}\sim q_y.
$$

The finite independent-member estimator is

$$
\widehat S_{d,J}=A_{d,J}-R_{d,J},\qquad
A_{d,J}=\frac1J\sum_jd(X_j,Z),\qquad
R_{d,J}=\frac1{J(J-1)}\sum_{j<\ell}d(X_j,X_\ell),\quad J\ge2.
$$

Here $R$ is **half the average unordered-pair distance**. Calling it simply ensemble spread without its coefficient invites a factor-of-two error. Both endpoints of every pair retain gradients. Common random numbers between SOURCE and child are compatible with a matched comparison; duplicated or coupled members within one forecast do not become independent by naming them members.

This is established energy/CRPS and fair-ensemble scoring. The statistical contribution available to PhyCoFlow is its specified physical observations and measured constrained intervention, not this formula. Euclidean norm energy scores are strictly proper on their represented laws under finite first moments; arbitrary dissimilarities require a negative-type argument, and strict distribution identification requires more. Squaring a Hilbert distance changes the target to a mean-type score. These distinctions follow from [Gneiting and Raftery (2007), Sections 4.3 and 5.1](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf), [Ferro (2014), fair random-sample ensembles](https://empslocal.ex.ac.uk/people/staff/ferro/Publications/ferro2013.pdf), and [Sejdinovic et al. (2013), Sections 2, 5 and 6](https://arxiv.org/html/1207.6076v3). Ferro's author copy bears its 2013 online-publication metadata; the 2014 journal citation refers to the same DOI, `10.1002/qj.2270`.

The expected score excess is half the corresponding energy-distance expression. It is a population statement about the chosen representation. Its optimum need not be zero because the true conditional law can contain uncertainty. Properness does not guarantee useful finite-sample discrimination, low gradient variance, an attainable model, convergence of the training algorithm, or independent generalization after repeated VALIDATION selection.

One reference for each observation is sufficient to estimate population conditional scores when observation–state pairs are representative and member randomness has the stated independence. It does not reveal every individual posterior. Nearby trajectory frames can be dependent, and multiple masks of one state remain a state cluster. These are the conditions already recorded in `LPC_THEORY.md:493–523,579–591`; they should accompany the empirical interpretation without demanding repeated truth for every unique sensor vector.

## 3. What the implemented family scores actually score

Code references use `src/phycoflow_reconstruction/` as their base unless a full repository-relative path is shown. Bare State/Scale/Shape/query filenames and `families/` paths inherit its `coherence/` directory.

| Object and code pointer | Inspected semantics | Supported interpretation |
| --- | --- | --- |
| `state_compute.py:179–275,322–356` | Exact weighted quantile functions for field axes and frozen mixed directions, integrated on the merged CDF intervals. The common distance takes the square root of the weighted sum of projected W2-squared terms. | A Hilbert-norm energy score of the joint retained quantile-function representation. Spatial labels and unqueried value directions remain absent. |
| `families/multiscale_spectrum/losses.py:19–48,285–386` | Euclidean norm of a joint signed coefficient/connected-piece-mean vector with fixed mass and TRAIN scales; fair unordered-pair correction. | Strictly proper for the retained joint vector law. Excluded modes/nodes remain absent; this is more than a collection of unary power spectra. |
| `scale_compute.py:101–164,168–225` | Signed unary/pair projections, scalar floored signed band correlation, joint anchor, log-power vector, and optional centered regional triple descriptors have distinct scores. | Auxiliaries change sensitivity and gradient geometry. A scalar rho score identifies a scalar descriptor law, not the whole spatial covariance. |
| `shape_compute.py:133–191,284–329,331–379` and `families/topology/persistence.py:202–236` | Fixed unary/pair filtrations, cubical persistence, finite-angle cross-diagonal SW1, fixed-count essential births, and positive pooled TRAIN leaf scales. | A proper composite score under the fixed geometry/transform/count conventions and integrability. Strict identification of a complete joint diagram law is not established. |
| `query_runtime.py:21–32,101–106,114–153` | Frozen spectral transforms/scales, explicit deterministic versus conditional branch, and `almost_fair_alpha == 1` on the scalable path. | The regular path uses fair scoring. Other public legacy/empirical modes are separate contracts. |
| `evaluation/frozen_coherence.py:265–345` | One frozen panel/reference/query pack, explicit mode, all supplied stochastic members, reusable reference transforms. | Training and assessment can share the same observable definitions without repeating inference for each metric. |

The State implementation is not the finite list of quantile levels illustrated by the earlier `method.pdf`. Its common merged integration grid is a way to evaluate exact distances between fixed quantile functions; it does not make the scientific map depend on the comparison endpoint. A finite product of weighted $L^2(0,1)$ spaces is separable Hilbert. [Lyons (2013), Theorem 3.16](https://rdlyons.pages.iu.edu/pdf/dcov-published.pdf) supplies the strong-negative-type justification for the norm version. Strictness concerns the law of this represented tuple, not the original spatial field. This is a useful citation to add when the current computation chapter is aligned.

For Shape, finite positive sums of scalar absolute-difference features preserve negative type. The signed cumulative-feature argument in `LPC_THEORY.md:616–630` matches the inspected p=1 cross-augmentation; fixed-angle approximation preserves the needed negative-type property. [Carrière, Cuturi and Oudot (2017), Lemma 3.2 and the finite-angle approximation in Section 3](https://proceedings.mlr.press/v70/carriere17a/carriere17a.pdf) is the direct source. It does not authorize arbitrary persistence assignment costs or target-dependent normalizations. The [November 2026 projected/reflected-kernel preprint](https://arxiv.org/html/2611.06619v1) independently warns that analogous higher-order cross-augmented constructions can lose the relevant properties; no new metric is required here.

Current Shape is also not the earlier attachment's smoothed Betti vector. Smoothed Betti curves lose birth–death pairing even before threshold discretization, as the revised chapter's two-bar counterexample shows. Spatially marked features and soft excursion anchors would be further declared observations. Their fixed Euclidean distances support energy scoring, but adding them does not automatically make the sum strictly identify their coupling with the persistence features.

The regular U-statistic and the finite optimized-atom V-statistic must retain separate identities. The latter uses a $J^2$ denominator and appears explicitly under `empirical=True` in State, Scale and Shape. An optimized finite teacher atom set is not an IID ensemble merely because its tensor has a member axis. Historical teacher results should keep their original score interpretation.

One reporting trap is especially concrete: `state_compute.py:343–346,381–386` returns marginal/mixed **gradient probes** using a detached factor proportional to $1/(2d)$. Their values are not the two direct distances, and they do not form an ordinary additive decomposition of the family score. Reuse the actual common distance for direct State error; do not relabel these telemetry probes as coherence-quality metrics.

The inspected State/Scale/Shape core hashes are respectively `eaa050e46880110bf6b5a021725fe7bf95be78b3a17e2cff6c20a48c423bfdf4`, `a2a0f6c74b0fe6ca321f885ca072ee306476d5130edc21efb46bfc9704fa9490`, and `17befdc10b806ef5e29ff81bc3003bb61c41f21cd546f04034e99a4fbe7792fb`. The spectral-loss and persistence-file hashes are `9178cafacdba82c2e5d4ebed3f83cb730245d91651d1dd99f1d450da52dd1f1f` and `c4b1d2101dead36455986d67a714def593f92a829c82b3da5d3a108996206818`.

## 4. Deterministic reduction is valid; the complete training path needs one modest repair

For a genuine deterministic endpoint, $q_y=\delta_{\widehat z(y)}$ and $d(\widehat z,\widehat z)=0$, so

$$
S_d(\delta_{\widehat z},z)=d(\widehat z,z).
$$

This is a separate Dirac branch, not substitution of $J=1$ into a zero-denominator estimator. It applies to the same fixed observations, measures and units. State enforces J1 for paired mode (`state_compute.py:304–311`); Scale supplies paired norms/absolute scalar errors (`scale_compute.py:87–99,133–152`); Shape omits separation for one member (`shape_compute.py:298–306`). `FrozenCoherenceObserver` routes the modes explicitly (`evaluation/frozen_coherence.py:280–293`). A stochastic generator evaluated with only one random draw is not thereby a deterministic model, and its paired distance does not estimate its full fair score.

This supports a shared scorer interface, not an already demonstrated general J1 training pipeline. `training/post_training.py:2338` chooses the scorer mode from model capabilities, but `training/fidelity_budget.py:341–350` explicitly restricts epoch-calibrated PI to `MeanRiskDifferenceBudget`. `coherence/physical_risk.py:42–57` requires at least two members for corrected mean-risk; its ordinary `physical_field_mse()` already supports a single field endpoint. A next implementation should let the existing feedback mechanism consume the declared member-MSE budget and J1 risk, with coherent recorded semantics. It should not duplicate a deterministic prediction to satisfy a J2 requirement. This is limited budget/risk plumbing, not a reason to redesign ConFIG.

For a stochastic model, scoring the arithmetic ensemble mean is another endpoint choice. Nonlinear occupation, threshold and persistence features of the mean need not equal the mean feature. Its results should be labeled ensemble-mean reconstruction, with no implied recovery of the generator's conditional law.

## 5. Direct coherence readouts and proper scores answer different questions

Adopt $A_{d,J}$ as **paired-reference feature discrepancy** for all models. Report $R_{d,J}$ and $\widehat S_{d,J}$ beside it for stochastic models. $A$ is nonnegative for the current distances and gives a useful direct account of reference similarity. It is not intrinsic physical validity, conditional calibration, or an unbiased discrepancy that vanishes for a perfect uncertain generator.

For an exact illustration, let the reference feature be Bernoulli with probability $1/4$ and use absolute distance. A perfect independent generator has expected $A=3/8$ and expected fair score $3/16$. A collapsed constant-zero model has $A=1/4$ and score $1/4$. Direct discrepancy favors collapse here, while the fair score favors the correct law. This is exact two-point arithmetic, not a model experiment. Expected member MSE has the analogous variance penalty already derived in `LPC_THEORY.md:445–483`.

Also, an expected member-MSE limit protects average member error per field. It does not guarantee acceptable error for every member or every observation. For a round whose delivered object is an individual reconstructed member, it is a clear primary accuracy contract; mean-risk and conditional spread can remain separately informative. A study whose main target is posterior calibration can need a different fidelity choice. The output object decides the contract; mathematical properness does not choose it automatically.

Keep the following minimum useful readout, using the same saved members and matched approximately 10% condition panel on each needed split:

| Readout | Physical/statistical question | Interpretation boundary |
| --- | --- | --- |
| Per-field physical member MSE or RMSE, SOURCE-relative change | How accurate are delivered member reconstructions on the declared mask and physical measure? | An average accuracy statement; neither conditional calibration nor every-member protection. |
| State $A,R,S$ | How similar are retained joint value occupations, and how does their conditional distribution change? | No location information; marginal changes can contribute to mixed errors. |
| Scale $A,R,S$, with self power, joint signed vector and pairwise rho visible | Which represented amplitudes, signs and alignments improve, and is diversity changing? | These are training-aligned diagnostics, not independent scientific validation. |
| Shape $A,R,S$, plus one declared spatial joint-region diagnostic | Are persistent structures/reference regions closer, and where are they misplaced? | Persistence and local occupancy retain different information; neither certifies dynamics or conservation. |
| A small selected scientific query outside the optimized score | Is the observed change useful for the intended scientific operation? | Its own units, support and interpretation must be explicit; not every possible query must improve. |

For Scale, use the existing signed band quantity

$$
\rho_{ab,s}(z)=\frac{\sum_m f_{sm}^2 c_{ma}(z)c_{mb}(z)}
{\sqrt{(\sum_m f_{sm}^2c_{ma}(z)^2+\delta_{sa})
(\sum_m f_{sm}^2c_{mb}(z)^2+\delta_{sb})}},
\qquad
D_{\rho}=\frac1J\sum_j\frac1{|\mathcal P|}\sum_{(a,b)\in\mathcal P}\sum_s\omega_s
|\rho_{ab,s}(X_j)-\rho_{ab,s}(Z)|.
$$

This is a normalized **within-snapshot spectral inner product**. It is bounded and sign-sensitive; its fixed power floors matter for weak bands. It is not full two-site cross-covariance or posterior correlation. Declare the pair set and band weights; reproducing the current rho block requires its existing uniform band average, rather than silently applying the signed-field block's possibly different band weights (`scale_compute.py:125–143`). Pair/band matrices can expose the physical relation behind the number.

Much of this readout exists already. `scripts/analysis/car_round7_scale_decompose.py:32–76` computes attraction/separation and `rho_pair_band` from cached endpoint transforms. `shape_compute.py:319–328` exposes compiled Shape attraction/separation components. `families/multiscale_spectrum/losses.py:418–456` exposes signed rho. Reuse these computations. A new broad diagnostic framework, repeated inference, or a covariance-block study is not required to explain the current scores.

The motivation for an interpretable diagnostic alongside a proper score is established. [Scheuerer and Hamill (2015)](https://repository.library.noaa.gov/view/noaa/22327/noaa_22327_DS1.pdf) demonstrate weak practical energy-score sensitivity to some correlation errors. [Pic et al. (2025), Propositions 1–2 and Section 5](https://ascmo.copernicus.org/articles/11/23/2025/) explain transformed/aggregated proper scores and the distinction between properness and discrimination. These precedents justify reporting targeted physical observations; they do not establish that direct rho or a particular new Shape feature will improve training.

## 6. Critical disposition of the five-question note

| Note proposal | Audit decision |
| --- | --- |
| Attribute attraction–separation to energy/CRPS and fair scoring | Adopt. The cited Gneiting–Raftery, Ferro, Pacchiardi and Pic sources are appropriate. Do not present the scoring principle as new. |
| Use paired J1 for deterministic reconstruction and fair conditional scoring for independent members | Adopt as the generic scoring contract. Complete the narrow PI/risk support gap before claiming the same end-to-end training works across both. |
| Treat small-ensemble cancellation as a possible mechanism | Retain as a diagnostic hypothesis. For J2 and a collinear reference between two endpoints, triangle equality makes the norm score zero even when both members are distant. This scalar/geometric witness does not diagnose current high-dimensional Scale or actual parameter gradients. |
| Change the almost-fair coefficient to fix that possibility | Do not adopt by default. The current alpha-one path is deliberate. Alpha below one adds a positive spread penalty and changes the population target for a random-sample generator. A fixed diagnostic can inform a later choice without becoming a new prelaunch gate. |
| Add spatial information cheaply to selected Shape filtrations | A reasonable bounded candidate. Prefer a fixed soft excursion/region map as a clearly named spatial observation if useful positional errors remain; marked-creator summaries are descriptive approximations. No theorem says a creator is a structure's center or that its discrete location supplies translation gradients. |
| Use member MSE as a clearer main fidelity measure | A legitimate prospective contract for member reconstruction. It is not an arithmetic correction to mean-risk, cannot retroactively qualify an older run, and does not protect every draw. Preserve the mean-risk/spread interpretation and each protocol's SOURCE law. |
| Replace a single aggregate Scale number with direct errors and $A,R,S$ | Adopt the reporting improvement. Do not relabel existing gradient attributions as direct distances or declare improved calibration from larger separation alone. |
| Reassess cached endpoints before more training | Adopt within a bounded existing panel and the actual open question. Reuse current decomposition outputs before recomputing them. A complete field/pair query population on Q4096 is still Q4096 support, not a native all-frequency assessment. |
| Summarize the method as minimizing a sum of SSS scores plus multipliers | Keep as a conceptual scalar alternative only. Current fresh parameter-space ConFIG plus a field row, clipping, AdamW and PI is not generally the gradient of that scalar. The revised theory already distinguishes them (`LPC_THEORY.md:632–645`). |

Two cited applications were checked directly. [Kochkov et al., NeuralGCM (2024), Methods and Supplement G.6](https://www.nature.com/articles/s41586-024-07744-y) does use two independent members and grid/spherical-harmonic CRPS. [Lang et al., AIFS-CRPS](https://arxiv.org/abs/2412.15832), subsequently [published in 2026](https://www.nature.com/articles/s44387-026-00073-7), motivates almost-fair CRPS through finite-member degeneracy. These are useful precedents, not evidence of failure or a remedy for this reconstruction model. The Nature AIFS page's indexed primary text was available, while a direct full-page fetch redirected to an inaccessible identity endpoint; the author preprint confirms the stated motivation.

The note's broad claim that fixed SSS support also implies fixed field-risk support must be read historically. The revised evidence explicitly records corrected Round 7 training-law references and rotating complement feedback (`MANUSCRIPT_EVIDENCE.md:91–95`). The earlier supplement's mismatched reference law is not a still-unaddressed defect. A proposal to add genuinely fresh field queries remains an implementation choice; it cannot be justified by ignoring the repair already present. Coordinate/support consistency of the generated endpoint is a separate requirement.

## 7. Minimal general multi-field design and bounded next decisions

Use $N_f$ for physical-field count and $n_x$ for spatial sites. Keep only three shared contracts: an endpoint provider returning one deterministic field or independent conditional members; fixed observable geometry/physical measures/units/calibration and a budgeted field/query list; and a declared scorer/risk readout. Cases supply field metadata, masks, boundaries, admissible thresholds and the scientific utility query. No combustion field name belongs in the generic scoring rule.

State can retain a finite bank of physical field-space directions and exact weighted occupation integration. Scale can retain a shared fixed basis, signed coefficients, piece means and selected pairs. Shape can reuse selected unary/pair filtrations and optionally attach a small fixed spatial observation. Computational budget can be controlled by sampled queries with their correct weights. Neither $N_f$-parameter persistent homology nor enumeration of all higher-order field combinations is needed to claim applicability to variable field counts. Conversely, unary/pair coverage does not establish all $N_f$-way relations; higher-order queries should answer a named task rather than serve as a completeness ritual.

Separate family scores can be proper for their respective targets while leaving cross-family coupling unchecked. A strict score on a concatenated representation would change that target and the training geometry. It is unnecessary unless an actual scientific question needs the missing coupling. Likewise, a sum of separate local spatial scores can be proper while missing long-range dependence. [Pacchiardi et al. (2024), Section 3.2 and Appendix B.3](https://jmlr.org/papers/volume25/23-0038/23-0038.pdf) makes this limitation explicit for patched scores.

The immediate bounded decisions are therefore: choose the delivered output and its primary fidelity risk; select one intelligible direct Scale readout using existing rho/attraction results; decide whether one positional Shape observation answers a demonstrated blind spot; and specify any new field-sampling law separately from SSS geometry. Preserve the regular fair score while these are resolved. An inconclusive short cancellation diagnostic is not permission to terminate a substantive learning study or to add a new optimizer tournament.

Learning success still needs a worthwhile measured tradeoff among physical field accuracy, held-out relational/task utility and training cost. It does not require every possible metric to improve, and mathematical eligibility should not be promoted into a collection of incidental early stopping gates. For a future similar repair study, keep the project's substantive 100-epoch review and provisional 500-epoch endpoint unless actual curves/question justify another horizon; small execution or cached-score checks establish only their own limited fact. This audit authorizes no execution.

Theoretical and interface claims can be retained independently of that eventual model decision. Effective enhancement, broad cross-system efficacy, governing-equation validity, causal coupling, temporal coherence and training-cost advantage each require their own matching evidence. The current revised foundations make these boundaries clear; the useful follow-up is to align current computation/post-training descriptions and the next protocol with them.

One recent close precedent for this empirical distinction is [Lang, Leutbecher and Hatfield (2026), a preliminary scale-aware AIFS study](https://arxiv.org/abs/2607.19161): the authors report differing multivariate-score behavior and emphasize effective scale weighting. This supports testing sensitivity and utility rather than inferring them from information retention; its forecasting results are not transferred to PhyCoFlow.

## 8. Checks performed and limits

All four revised chapters and the complete supplied note were read. The source paths and line pointers above were inspected, and the score formulas were compared with primary literature; State's Hilbert and Shape's finite-angle justifications were checked separately. The Bernoulli example and deterministic/J2 reductions are exact analytic calculations. The portable Markdown checker passed for this document using the existing `phycoflow_env` interpreter, and the workspace-navigation checker reported zero issues. No GPU work, dataset iteration, new prediction/metric evaluation, training, polling, checkpoint access, run-status inspection, or modification of existing source/plans/reports was performed. This is an evidence audit supporting a new plan, not a new efficacy result or a reclassification of historical outcomes.
