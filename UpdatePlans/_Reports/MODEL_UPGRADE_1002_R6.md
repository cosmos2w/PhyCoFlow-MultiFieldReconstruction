# PhyCoFlow R6: measured outer-strength trade-off

**Status: `recommended_bounded_recipe`.** The fixed SOURCE beta 0.10 recipe supports useful stable refinement, the engineering endpoint corridor and required bounded native-RF preservation checks at selected seed 42/epoch 200 and independent SOURCE seed 43/epoch 150. Implementation readiness includes the physical public SOURCE 1→own 2 recovery check. Individual historical failures, local native drift and the pressure strict-field exception remain explicit. The SOURCE-initialized 5000 profile is prepared and its two-epoch public sibling is tested; 5000-epoch stability is unmeasured. No formal run was launched, queued or scheduled.

## Operating summary

1. **Active method:** fixed `J_beta = 0.1*native_RF_loss + beta*calibrated_ABC`, one scalar backward, clipping and AdamW (`lr=5e-5`, decay `1e-6`). Selected beta is 0.10, native coefficient 0.1, parameter retention off; no ConFIG or adaptive fidelity controller is active.

2. **Change from R5:** R5 used the fixed scalar route plus a fixed SOURCE-parameter L2 penalty. R6 removes that penalty and measures the global coherence strength, while preserving A/B/C descriptors and SOURCE family-calibration meaning, sampling, learning rate, precision, two-step Euler endpoint sampler and zero warmup. All independent science runs initialize the original SOURCE; the selected 150→200 extension resumes its own state.

3. **Raw data-loss behavior:** at selected 150, preceding/final 25 raw native TRAIN means are 0.130237/0.127398 and final 50 slope is −8.66345e−5 per epoch; fixed native TRAIN ratios are 0.971223/0.974291 in final 25/final 50. At 200 raw TRAIN plateaus (preceding/final 25 means 0.127458/0.127660; final 50 slope +1.63196e−5). SOURCE seed 43 at 150 decreases 0.130178→0.127087 (final 50 slope −8.78807e−5), with fixed native TRAIN ratios 0.999084/0.993788.

4. **Useful and mixed components:** selected 150 final 50 A/B/C family ratio medians are 0.906198/1.008033/0.809367. A marginal and copula discrepancies improve; C self/mutual active totals are 0.808808/0.813527. At 200 A/B/C medians become 0.995268/1.013189/0.825914 and self/mutual active-C totals 0.826910/0.830828; A gains diminish and B remains mixed. Confirmation 150 final 50 A/B/C medians are 0.967542/1.009396/0.819802, with active-C self/mutual totals 0.817177/0.821896. These are descriptor comparisons, without a physical-causality or complete-dependence guarantee.

5. **Endpoint and native risk:** selected 150 passes the engineering endpoint corridor and strict final-window field 1.05 checks. Native validation final 25/final 50 ratios are 1.040663/1.035667, but individual epoch 100/150 ratios 1.060623/1.062218 fail 1.05. At 200 native windows 1.032062/1.031660 pass, but pressure window ratios 1.065094/1.061399 fail strict 1.05 while satisfying the engineering 1.10 corridor. Historical failures remain visible. Expanded 256 ratios are 0.990552 (selected 200) and 1.029957 (confirmation 150); required bounded native checks pass, while local exceptions remain explicit.

6. **Cost and user operation:** 500 science+2 execution-check epochs used 33,862.956 charged GPU seconds (9.406 h), including 843.026 seconds offline diagnosis (720 conservative initial+123.026 measured follow-up). Selected 200 synchronized update work averaged 63.8624 seconds/epoch and inclusive cost 65.2487 seconds/epoch. SOURCE 43→150 returned normally (64.5622 synchronized update seconds and 65.9366 inclusive seconds/epoch); the public two-epoch execution sibling passed. The [actual SOURCE beta0.10 profile](../../cases/turbulent_combustion/configs/posttrain/abc_upgrade_1002_r6_5000ep.yaml) and [exact launch/stop/own-resume commands](../../cases/turbulent_combustion/configs/posttrain/abc_upgrade_1002_r6_launch.md) are prepared; public SOURCE 1→own 2 recovery passed, and 5,000-epoch stability is unmeasured. Do not launch the formal profile as part of R6 execution.

## PDF masters and figure index

| Master | Pages | Evidence |
|---|---:|---|
| [R6_policy_and_cost.pdf](../../cases/turbulent_combustion/runs/Test_1002/R6_reports/R6_policy_and_cost.pdf) | 2 | Literal R5/R6 objective, fixed coefficients, separate readiness/usefulness/native conclusions and measured cost |
| [R6_learning_and_preservation.pdf](../../cases/turbulent_combustion/runs/Test_1002/R6_reports/R6_learning_and_preservation.pdf) | 2 | Unsmoothed learning traces, labeled mature summaries, raw/native/endpoint risk and A/B/C self/mutual components |
| [R6_fields_and_diagnosis.pdf](../../cases/turbulent_combustion/runs/Test_1002/R6_reports/R6_fields_and_diagnosis.pdf) | 4 | Three predeclared GT/SOURCE/selected cases with sensors, shared scales and relative-L2; paired native field/time risk and historical sampler/diversity checks |

The three PDF masters contain eight pages. Ratios use the same saved SOURCE definitions and separately labeled estimators. Signed native-grid residuals retain the original physical normalization, unknown units and pressure gauge. Local adverse native cells and historical failures remain visible. The final [render receipt](../../cases/turbulent_combustion/runs/Test_1002/R6_reports/r6_pdf_render_receipt.json) binds the numerical inputs and PDF hashes; the [root visual inspection receipt](../../cases/turbulent_combustion/runs/Test_1002/R6_root_reviews/final_pdf_visual_qa.json) records all eight inspected pages. Temporary page rasters are removed after inspection; these PDF masters are the presentation outputs.

## Literal method and scope

R5 N used `0.1 * native_RF_loss`; F used `0.1 * native_RF_loss + calibrated_ABC`; S added the fixed SOURCE parameter penalty `lambda_SP * Omega`, with `lambda_SP = 0.026697108274509104` and `Omega = 0.5 * sum_trainable |theta - theta_SOURCE|^2`. The selected R5 scalar path executed one scalar backward, gradient clipping, and AdamW. Its resolved dispatch was `weighted_sum / legacy / r4_exact / matched_native_v1`. ConFIG and adaptive fidelity control were inactive. Frozen SOURCE gradient-norm calibration made A/B/C scales comparable within the structural sum; it did not select or adapt the collective data/coherence trade-off.

The sole new scientific variable is beta in `J_beta = 0.1 * native_RF_loss + beta * calibrated_ABC`. Parameter retention is disabled. SOURCE initialization, native RF bridge, descriptor definitions, family-calibration meaning, sampling, learning rate `5e-5`, AdamW decay `1e-6`, clipping `1`, precision, two-stage endpoint sampler, and zero-warmup schedule are preserved. Each complete scientific epoch contains 38 batches of 32 sampled TRAIN presentations, with 4,096 shared spatial queries. The public 5,000-epoch profile must reproduce the selected recipe's first 200 scientific epochs; its longer configured horizon is not measured learning evidence.

For normalized target `u`, prior `z`, observations `y`, and artificial generative coordinate `t`, the implemented RF bridge is `x_t = (1-t)z + t*u`, and the native residual is `r_theta = v_theta(t, x_t; y) - (u-z)`. The configured `sigma_min` adds no term to this bridge. Artificial `t` is not combustion time. No bridge, PDE, gauge, or smoothing change was introduced. The five fields are CH4/CO/T/U_1/p; physical units remain unknown.

## Descriptor interpretation

A retains `marginal_copula_v2` on all five fields: the model-unit marginal-quantile discrepancy measures self-consistency, while the smooth own-CDF joint-copula discrepancy measures cross-field dependence. An A-total improvement does not by itself identify which of these improved, nor does a smaller copula discrepancy measure a numerical increase in mutual information. Exact-midrank evaluation, where available, is a separate estimator from the smooth coordinates used for training.

B retains `second_order_blocks_v4`: signed centered covariance of linear graph coefficients with the existing symmetric calibrated floors, bands, masks, and normalizations. Its four active pairs are CO/T, T/CH4, T/U_1, and CH4/U_1; p is not an active B pair. The grouped-32 result is primary; pooled-64 covariance is separately labeled. Same-frequency and cross-frequency blocks both describe cross-field second-order dependence. The same-frequency block is not an active self-spectrum objective, and covariance across snapshots is not conditional posterior covariance across repeated draws at one observation.

C retains `finite_primary_v1` on CO/T, the existing 32-by-128 cubical rasterization and fresh generated pairings, target-only reuse, H0/H1, sublevel/superlevel, 16 master lines with four training lines, separately frozen SOURCE scales, and finite/essential coefficients 0.9/0.1. Active self and mutual finite/essential contributions are reconstructed from the exact group weights and denominators. Self CO/T H0/H1 groups each carry weight 1/8; mutual CO+T H0/H1 groups each carry weight 1/4. Legacy `topology.self.persistence` and `topology.mutual.persistence` are not the active category totals. H1 essential terms with zero SOURCE denominator are NA. The 0.9/0.1 coefficients do not guarantee the same gradient split; raster PH is not native-grid PH or a complete multiparameter invariant. The existing topology raster's antialiasing is preserved; the representative native-grid field arrays receive no new smoothing.

## Retained native-RF diagnosis

Before reading losses, the campaign declared 256 validation snapshots uniformly spanning the chronological validation segment and eight fixed native draws. Every snapshot was evaluated under every draw, with 4,096 spatial queries. SOURCE and each child used identical observations, queries, prior, and artificial time. Five artificial-time bins are `[0,.2)`, `[.2,.4)`, `[.4,.6)`, `[.6,.8)`, and `[.8,1]`. Pooled results divide raw squared-error sums by raw element counts; ratios are ratios of pooled means, not averages of cell ratios. Original 32-case monitors are preserved and reported separately. The expanded panel is development evidence on correlated snapshots, and repeated draws are not independent trajectories or pristine TEST evaluation. Training, diagnosis and recipe selection use TRAIN/development validation only; no TEST metrics or TEST-based selection were computed. The existing preload can read the whole HDF5 into CPU memory, including TEST rows. Receipt fields named `no_TEST_reads` refer to excluded TEST computation, not a byte-level HDF5 access audit.

The independent raw-array review gives expanded SOURCE-relative native-RF ratios N100 `1.015098`, F150 `1.052539`, and S200 `1.063787`. F150 field ratios are CH4 `1.059532`, CO `1.066601`, T `1.122550`, U_1 `1.113832`, and p `0.994332`; S200 ratios are `1.116158`, `1.134507`, `1.121015`, `1.097907`, and `1.007290`. These are expanded endpoint-checkpoint results, not historical final-window estimates. The unchanged original-32 final-50 validation ratios F150 `1.084998`, S150 `1.063722`, S200 `1.070667`, and S43-100 `1.068301` remain failures of the historical `<=1.05` reference.

On the identical bridge state, `Delta v = v_child - v_SOURCE` gives the exact accounting identity `D_child - D_SOURCE = E mean |Delta v|^2 + 2 E mean <r_SOURCE, Delta v>`. The explicit bridge evaluation matched production `training_loss` within `1.19e-7`; double-precision accumulation before velocity subtraction closed the accounting identity within `2.64e-11`. For F150 the loss change `0.052167` decomposes into `0.085155 - 0.032988`; for S200 `0.063335` decomposes into `0.046650 + 0.016686`. This measures function change and its residual alignment, not parameter distance or a causal attribution to a particular layer. Field/time raw arrays retain the same identity and count closure.

## Retained coherence, sampler, and diversity diagnosis

F150 final-50 active C ratios are self finite `0.753752`, self essential `0.448636`, self total `0.731288`, mutual finite `0.743067`, mutual essential `0.455427`, mutual total `0.724119`, and family total `0.727588`. The self/mutual totals use their own SOURCE denominators and are not unweighted means of component ratios. S200 family C is `0.808388`. Reconstructed active category contributions close each raw C row within `1.08e-7` and the weighted family within `1.5e-9`. R6 component extraction must use each run's own saved SOURCE calibration and denominators, whose numerical differences are checked within the established tolerance; it must never substitute or refit the R5 state.

On 16 fixed development observations with the same initialization draw per model and solver resolution, normalized endpoint-MSE child/SOURCE ratios at 2/4/8 Euler stages are F150 `1.046711 / 1.004651 / 0.994921` and S200 `1.009453 / 0.986493 / 0.976455`. Absolute normalized MSE increases with more stages for SOURCE and both children. The change in relative ordering therefore indicates sampler sensitivity without establishing that a higher-stage sampler improves reconstruction. The production two-stage sampler is unchanged, and different solver workloads are not used for a same-workload timing claim.

Eight production-sampler draws on the same 16 observations give F150 predictive-spread ratios CH4/CO/T/U_1/p `0.387 / 0.374 / 0.402 / 0.463 / 0.490`, with marginal ensemble CRPS ratios `1.048 / 1.065 / 0.960 / 1.084 / 1.048`. S200 spread ratios are `0.601 / 0.563 / 0.694 / 0.604 / 0.755`, with CRPS `1.036 / 1.072 / 0.956 / 1.041 / 0.994`. Reduced spread alone does not prove collapse; poorer predictive scoring in several fields is relevant adverse evidence. One target per observation does not identify the conditional posterior. These are historical-arm measurements, not measurements of R6 diversity.

For conditionally independent generated/reference descriptor vectors Z/H given y, `E[||Z-H||^2 | y] = ||E[Z|y]-E[H|y]||^2 + tr Cov(Z|y) + tr Cov(H|y)`. Thus a paired squared-descriptor term can favor reduced generated variance, but this illustrative identity does not establish that all current A/B/C terms collapse the generator. No ensemble, adversarial, or replacement loss was introduced.

Eight fixed TRAIN gradient batches at SOURCE and F150 reused each rollout graph. Median `||grad calibrated_ABC|| / ||0.1 grad native_RF_loss||` is approximately `2.560` at SOURCE and `1.269` at F150; mean native/coherence cosine is `0.147` and `0.216`. This does not establish held-out descent, and the norm ratio is not inferred from scalar loss heights. All 145 selected trainable tensors, containing 5,461,817 scalars, are real float32. The generic `.double()` complex-gradient portability concern cannot explain these selected-backbone results; complex-backbone migration is outside this campaign.

## Mathematical definitions and code mapping

Let generated/reference model-unit fields on the shared query set be $\widehat u,u\in\mathbb R^{B\times Q\times5}$, with $B=32$ and $Q=4096$. The native objective averages the squared RF residual over its sampled batch, spatial queries and fields: $D(\theta)=\mathbb E[\|v_\theta(t,(1-t)z+tu;y)-(u-z)\|_2^2/(Q\,5)]$. This native velocity risk is distinct from endpoint reconstruction MSE and from physical relative-L2. Artificial RF time $t$ is independent of chronological snapshot position.

### A: marginal and own-CDF joint dependence

For each snapshot and field, the active self term is the empirical squared Wasserstein discrepancy $W_2^2(\widehat u_f,u_f)=Q^{-1}\sum_j(\widehat u_{(j),f}-u_{(j),f})^2$, averaged with equal field weights. The generated and reference sides of the joint term each use their own smooth CDF: center and divide by $\sqrt{\operatorname{mean}(x-\bar x)^2+10^{-12}}$, construct 64 quantile landmarks at $(k+1/2)/64$, and average $\sigma((x-q_k)/0.1)$ over landmarks. Generated centering, scaling and landmarks remain differentiable; reference values are detached. The smooth coordinates approximate empirical ranks and are not exactly invariant to arbitrary nonlinear monotone transformations.

Project those five-dimensional coordinates onto 32 fixed mixed directions; for direction $r$, $d_r$ is the empirical projected $W_2^2$. The joint term is $A_{\mathrm{copula}}=0.75\operatorname{mean}_r d_r+0.25\mathcal T_{0.1,\tau}(d)$, with fixed $\tau=3.7980360502842816\times10^{-6}$ and

$$
\mathcal T_{\rho,\tau}(d)=\min_\eta\left[\eta+\frac{\tau}{\rho R}\sum_{r=1}^{R}\log(1+e^{(d_r-\eta)/\tau})\right]-b_{\rho,\tau},\quad b_{\rho,\tau}=-\tau\log\rho-\tau\frac{1-\rho}{\rho}\log(1-\rho).
$$

The baseline makes zero directional discrepancy have zero smooth-tail score. Each eta root is solved independently, to tolerance`1e-6` or 80 iterations; it is an internal scalar tail calculation, not a persistent fidelity controller. The active family is $A=A_{\mathrm{marginal}}+A_{\mathrm{copula}}$; the separate pairwise mutual objective is disabled. [Self marginal](../../src/phycoflow_reconstruction/coherence/families/global_distribution/components/self_marginal.py), [smooth copula](../../src/phycoflow_reconstruction/coherence/families/global_distribution/components/copula.py), [joint objective](../../src/phycoflow_reconstruction/coherence/families/global_distribution/components/cross_copula.py) and [tail calculation](../../src/phycoflow_reconstruction/coherence/families/global_distribution/components/tail_risk.py) implement these definitions. A smaller dependence discrepancy does not estimate a numerical increase in mutual information.

### B: cross-field covariance within and across graph bands

Let $a_{b,k,f}$ be the linear coefficients of field $f$ in the fixed retained graph basis. Production uses the unbiased centered sample covariance $\Sigma_{kq,fg}=(B-1)^{-1}\sum_b(a_{b,k,f}-\bar a_{k,f})\overline{(a_{b,q,g}-\bar a_{q,g})}$; with real selected fields the conjugation is immaterial. For an eligible field-pair/band-pair block, the cost is $\|\widehat\Sigma_{fg}^{\ell m}/d_{\widehat u}-\Sigma_{fg}^{\ell m}/d_u\|_F^2$. Each denominator uses the corresponding generated/reference band energies with the same calibrated positive energy floors, plus shared $\delta=10^{-12}+10^{-6}\sqrt{E^{\mathrm{cal}}_{\ell f}E^{\mathrm{cal}}_{mg}}$. Each energy floor is $\max(10^{-6}E^{\mathrm{cal}},\mathrm{tiny})$; reference-calibration band fractions below`1e-8` are excluded. No spatial $Q\times Q$ covariance is formed.

$B=B_{\mathrm{same}}+B_{\mathrm{cross}}$ averages eligible blocks separately for $\ell=m$ and $\ell\ne m$. The four active field pairs are CO/T, T/CH4, T/U_1 and CH4/U_1. Both terms measure cross-field dependence; self-spectrum is disabled. A grouped 32 covariance estimate and a pooled 64 estimate are different estimators because centering and covariance are nonlinear in the grouping. The [covariance-block implementation](../../src/phycoflow_reconstruction/coherence/families/cross_spectrum/covariance_blocks.py) and [family composition](../../src/phycoflow_reconstruction/coherence/families/cross_spectrum/family.py) preserve the original bands and eligible-block reductions.

### C: finite-primary self and mutual shape

CO/T are rasterized on the preserved 32×128 geometry and standardized using each reference snapshot's mean and population standard deviation, floored at`1e-4`; generated fields use those same reference statistics. Self filtrations are each signed field. A positive mutual line $(a,b)$ uses $\max_f((s x_f-b_f)/a_f)$ with $s\in\{+1,-1\}$ and distance weight $\min_f a_f$. Sixteen fixed SOURCE-independent master lines are retained; ordinary training samples four and takes the uniform mean of their weighted distances, without an extra 16/4 multiplier.

Finite diagram distance uses 32 equally spaced midpoint angles, cross-diagonal augmentation, sorted projected absolute differences, summation over diagram points and averaging over angles, divided by the fixed raster size 4096. Essential distance compares sorted essential births with absolute differences and no raster-size divisor; essential classes cannot match the diagonal. Neither quantity is a squared Wasserstein distance or exact diagram assignment. Let $F_g,E_g$ denote these raw distances after polarity/line averaging for group $g$. The positive frozen TRAIN SOURCE scale is $s_{g,p}=\max(\mu^{\mathrm{SOURCE}}_{g,p},10^{-4}+0.1\operatorname{median}_{h:\mu_{h,p}>0}\mu_{h,p})$ separately for $p\in\{F,E\}$, with an empty-positive median taken as zero. The active family is

$$
C=\sum_g w_g\left(0.9\frac{F_g}{s_{g,F}}+0.1\frac{E_g}{s_{g,E}}\right),\quad w_{\mathrm{self},f,h}=\frac18,\quad w_{\mathrm{mutual},CO+T,h}=\frac14,\quad h\in\{0,1\}.
$$

Self and mutual category totals sum their actual weighted groups. Reported category ratios divide by the corresponding SOURCE category total; they are not unweighted averages of component ratios. A zero SOURCE H1-essential denominator yields NA, not a successful zero ratio. [Filtration, normalization and grouping](../../src/phycoflow_reconstruction/coherence/families/topology/persistence_objective.py) and [diagram distances](../../src/phycoflow_reconstruction/coherence/families/topology/persistence.py) implement this finite-positive-line surrogate; it does not establish native-grid topology or a complete multiparameter invariant.

### Frozen family calibration and outer strength

For family $i$, let $n_i$ be the median unweighted family-gradient norm over the same two predeclared SOURCE TRAIN calibration batches, and $n_*=\operatorname{median}_i n_i$. [Family calibration](../../src/phycoflow_reconstruction/training/coherence_calibration.py) sets $\gamma_i=\operatorname{clip}(n_*/\max(n_i,10^{-12}),10^{-5},100)$ once. The calibrated structural sum is $L_{ABC}=\gamma_A A+\gamma_B B+\gamma_C C$, with approximately $(\gamma_A,\gamma_B,\gamma_C)=(1,1.232288,0.01407067)$ and unit configured family weights. Every run saves its own SOURCE-derived calibration; small numerical differences are audited rather than replacing it with R5's state. Native-gradient diagnostics do not enter this scale formula. The scalar [training composition](../../src/phycoflow_reconstruction/training/post_training.py) optimizes $0.1D+\beta L_{ABC}$, with retention disabled and constant beta from the first epoch. Equalizing initial family norms leaves the outer data/coherence strength question to the measured beta comparison; it provides no native-preservation guarantee.

## Implementation and budget evidence already complete

Core implementation is committed and pushed at `b9ccf1784476c53f92f2ed25d1ce6e00aeecbc43`. It exposes the existing scalar outer-strength path, adds source-bound resolved-policy receipts and immutable own-resume checks, and provides an absolute bounded epoch limit for the public segmented launcher without changing its configured horizon. Focused tests cover coefficient application, native gradients, retention-disabled dispatch, calibration meaning, invalid policy combinations, and recovery. The one final CPU regression completed with 845 passed and 9 skipped. These results establish implementation checks, not useful mature learning.

Production SOURCE family calibration records 7.244 seconds for P1 and 7.493 seconds for P2; SOURCE 43 records 7.284 seconds at startup. Each uses two fixed TRAIN batches, then frozen family scales; the own 150→200 continuation reuses the saved calibration. These wall durations are already contained in the external attempt charges and are not added again. Other initialization cost was not isolated from monitors, checkpoint IO and teardown, so their combined residual is not labeled pure initialization.

Initial offline diagnostic stages and failed setup/audit attempts consumed 285.850 seconds of internal measured stage time. Exact initial outer process envelopes were not captured; a conservative inclusive wall-envelope charge of 720 seconds is therefore used once against both the 45-minute diagnosis ceiling and 11-hour overall ceiling. It is not presented as measured GPU compute. Final process launch-to-teardown envelopes are added once; nested timers are not added again. Final science/check exposures are 500/2; the combined inclusive charge is 33,862.956 seconds, below 39,600. The final budget and compact retention are recorded below.

## Preliminary equal-age P1 review at epoch 100

P1 beta 0.25/seed 42 reached its saved epoch 100 checkpoint, SHA256 `5357136171d44a11520c2b4a581873efe0f66df55334f8412687c173930b456d`, while continuing unchanged to its authorized epoch 150 target. The final 50 window contains the five fixed-panel observations at epochs 60/70/80/90/100; final 25 contains epochs 80/90/100. This intermediate review did not select a recipe; the completed mature results below control the final conclusion.

| SOURCE-relative ratio of raw means, final 50 | N100 | F100, beta 1 | P1_100, beta 0.25 |
|---|---:|---:|---:|
| Native RF validation | 0.998260 | 1.078245 | 1.049340 |
| Fixed native RF TRAIN monitor | 0.970697 | 1.005486 | 0.974975 |
| A | 0.974037 | 0.849426 | 0.875819 |
| B, grouped 32 primary | 0.994713 | 1.001028 | 1.002949 |
| C, active finite-primary | 1.000549 | 0.742034 | 0.788582 |
| Endpoint aggregate MSE | 0.996491 | 1.020359 | 1.012991 |

P1 retains useful A/C improvement at 100; B remains mixed, with grouped same/cross-frequency ratios `1.036072 / 0.969052`. A marginal/copula discrepancies are `0.874620 / 0.925851`; reduced copula discrepancy is not a measured increase in mutual information. All five endpoint fields pass the engineering 1.10 corridor and strict 1.05 reference in both windows at 100. The native final 50 ratio narrowly passes 1.05, but final 25 is `1.061143` and fails. The older F100 native failure remains a failure. The smaller beta improves this matched early dose response without establishing native preservation or long-horizon stability.

The single saved epoch 100 endpoint panel has aggregate MSE ratio `1.03285` and CO ratio `1.06004`; its strict field-fidelity selector is ineligible because CO exceeds 1.05, although it remains within the engineering 1.10 corridor. This isolated-panel exception is retained separately from the passing 25/50-window endpoint summaries. A selector label does not replace the complete mature learning evidence or authorize a different checkpoint lineage.

CPU extraction from P1's own saved SOURCE group scales and step 0 validation denominators yields final 50 active-C self finite/essential/total ratios `0.802609 / 0.612636 / 0.788623` and mutual `0.800419 / 0.620153 / 0.788544`. Independent reconstruction of all six group contributions, using their fixed weights summing to 1 and the 0.9/0.1 finite/essential coefficients, closes to recorded active C within `7.12e-8`; its combined SOURCE ratio is `0.788582`. Both self and mutual descriptors contribute retained improvement. This is saved raster-PH discrepancy evidence, not native-grid topology, physical coupling, a complete multiparameter invariant, or native-fidelity preservation. No calibration was refit, no checkpoint forward was added, and historical extraction artifacts remained byte-identical.

P1 raw native TRAIN preceding/final 25 means are `0.128543 / 0.129820`, with final 50 slope `4.08466e-5` per epoch and descriptive epoch standard deviation `0.004722`. N100 also rises between those same windows (`0.127051 / 0.128146`, slope `3.45822e-5`); P1-minus-N mean excess changes from `0.001492` to `0.001673`. These correlated matched-stream observations motivate continuing the complete trajectory and fixed monitors rather than interpreting every positive sampled slope as sustained deterioration. The completed 150 decision below uses its mature windows.

P1 averaged `64.384` seconds of synchronized 38-update work per complete epoch through 100, versus `64.089` for F100 on the verified same physical GPU1 envelope (ratio `1.004593`). System load was not controlled. These timers exclude data-source, monitor, checkpoint, startup and teardown overhead; its completed 150 inclusive envelope is reported below. No additional calibration, controller, optimizer, descriptor, learning-rate or warmup change was made.

### Mature P1 age 150 and the conditional beta 0.10 test

P1 completed all 150 scientific epochs (5,700 updates) from SOURCE with beta 0.25 and retention disabled. Final 25/50 native-validation ratios are 1.056538/1.057457: both fail the historical 1.05 reference. Final 50 A/grouped-B/finite-primary-C ratios are 0.886406/1.016210/0.769608, with marginal/copula 0.885628/0.918885 and B same/cross 1.051591/0.980002. Thus A and C gains remain useful while B is mixed. Endpoint aggregate ratios 1.019721/1.014825 pass the engineering corridor, but CO 1.052862/1.052706 remains a strict 1.05 field exception. The late raw native TRAIN loss falls from 0.131341 in the previous 25 epochs to 0.128424 in the final 25; final 50 slope is -8.955e-5 per epoch. This is stable late training with a held-out native trade-off, not automatic fidelity control. The allowed single SOURCE beta 0.10 arm is justified by useful structural refinement and adverse complete native windows. No other optimizer, retention, descriptor, solver, or schedule change is proposed.

P1 mean complete-epoch synchronized update work is 64.307 seconds (total 9,646.071 seconds). The measured external SOURCE initialization-through-return envelope is 9,865.064 seconds, including calibration, monitoring, checkpointing and terminal evaluation. The 218.992-second difference is a combined non-update residual, not separately measured initialization time. The optional strength arm was launched only after reserving the unchanged continuation, confirmation, final diagnosis and public-check budget. Completed campaign costs appear in the final budget table.

At P1 age 150, reconstructed active-C final 50 SOURCE-relative self finite/essential/total ratios are 0.790265/0.511056/0.769708, and mutual ratios 0.786369/0.530519/0.769515. The final 25 totals are 0.767687 self and 0.765991 mutual. These categories use the active frozen finite-primary group scales and weights, not legacy persistence category totals. Root independently reconstructed every category from saved raw group means, 0.9/0.1 part weights and 1/8 self versus 1/4 mutual group weights; category error is 1.11e-16 and active objective closure 5.69e-8. SOURCE-zero H1 essential ratios remain undefined. No new GPU evaluation or calibration was used.

### Equal-age 100 strength response: developmental evidence before selection

Root independently reviewed all 100 complete epoch histories and the exact P2 epoch 100 archive (`83aab377fb5169d94895fa93dc3b4a3394356c725d03715b9ea6b6b9d3c1b260`). The comparison is bound in `equal_age100_strength_response.json`. N/F provide the existing beta 0/beta 1 references; both new arms disable parameter retention and preserve the same native coefficient, descriptors, calibration meaning, sampling, learning rate and zero warmup.

| SOURCE arm at 100 | Beta | Native VAL final 25 / final 50 | A / grouped B / active C final 50 | Endpoint aggregate final 50 |
|---|---:|---|---|---:|
| R5 N100 | 0 | 1.013387 / 0.998260 | 0.974037 / 0.994713 / 1.000549 | 0.996491 |
| R5 F100 | 1 | 1.074791 / 1.078245 | 0.849426 / 1.001028 / 0.742034 | 1.020359 |
| R6 P1 | 0.25 | 1.061143 / 1.049340 | 0.875819 / 1.002949 / 0.788582 | 1.012991 |
| R6 P2 | 0.10 | 1.043066 / 1.028491 | 0.902624 / 0.992892 / 0.827376 | 1.007724 |

At this common age, reducing the outer strength reduces the average native-validation offset while retaining A/C improvement. P2 final 25/50 marginal ratios are 0.934037/0.901433, joint-copula 0.966825/0.952311, grouped B same-frequency 1.018170/1.022294 and cross-frequency 0.961883/0.962803. B therefore remains mixed; separately pooled B ratios 0.958766/0.966178 use another covariance estimator. Reconstructed active-C final 25/50 self total ratios are 0.814341/0.824636 and mutual total ratios 0.820781/0.829946. Their finite ratios are self 0.829130/0.836138 and mutual 0.832825/0.839389; essential ratios are self 0.628258/0.679913 and mutual 0.649999/0.696036. The own SOURCE total denominator is 1.174633739. Root independently reconstructed these contributions from raw group means, frozen scales and active part/group weights, with zero category mismatch and 7.03e-8 maximum active-C closure error; the captured live-history byte prefix also rehashed exactly. These are active finite-primary quantities, not legacy category averages. H1 essential SOURCE-zero ratios remain undefined, and raster PH does not establish native-grid topology or a physical causal relationship.

Both P2 rolling native windows pass 1.05 at 100, but the single epoch 100 native-validation ratio is 1.060623 and fails that reference. This failure and its individual draw variation remain visible; rolling means do not erase it. Endpoint final 25/50 aggregates 1.018299/1.007724 and all endpoint-field window means pass 1.05 (CO1.038535/1.034546). These are fixed correlated development panels, with three/five monitor observations in the final 25/50 epoch windows, not independent experiments or expanded 256-panel evidence.

P2 raw native TRAIN preceding/final 25 means are 0.127412/0.128705, with final 50 slope +4.14109e-5 per epoch. N100 also rises 0.127051 to 0.128146 (slope +3.45822e-5), with similar positive slopes in F100 and P1. This supports reading the complete trajectory and mature late windows rather than declaring sustained deterioration from this intermediate sampled rise. Complete-epoch update work through 100 averages 64.342 seconds. The completed mature 150 and own 200 results below supersede this intermediate developmental review; no formal launch is claimed.

## Mature beta 0.10 at 150 epochs: selected for unchanged continuation

The selected beta 0.10 SOURCE seed 42 lineage completed all 150 epochs normally. Both final 25/final 50 windows pass the descriptive useful-family screen and engineering endpoint corridor. Raw native TRAIN loss decreased from 0.130237 in the preceding 25 epochs to 0.127398 in the final 25, with a final 50 slope of -8.66345e-5 per epoch; the fixed native TRAIN ratios are 0.971223/0.974291. This supports continuing the unchanged recipe, not a 5,000-epoch stability claim.

Final 50 family ratios of raw means A/B/C are 0.905871/1.010662/0.811243; family ratio medians are 0.906198/1.008033/0.809367. A marginal/copula ratios are 0.905070/0.939335. Grouped B same/cross ratios are 1.041588/0.979014; pooled B total 0.977074 is a different estimator. Active C own-SOURCE self and mutual total ratios are 0.808808/0.813527, including finite ratios 0.825691/0.827909 and essential ratios 0.596373/0.609575. Both self and mutual descriptor discrepancies improve, without proving physical causality or all-field dependence preservation.

Endpoint aggregate final 25/final 50 ratios are 1.013107/1.006972; all final-window fields are below strict 1.05, including CO at 1.043657/1.041920. Native validation window ratios are 1.040663/1.035667, but the exact individual epoch 100 and epoch 150 checks fail at 1.060623 and 1.062218. These failures remain visible. Expanded native field/time confirmation is reported below.

Complete synchronized update work averaged 64.2314 seconds per epoch (9,634.715 seconds for 150), while the actual inclusive outer charge was 9,840.116 seconds (65.6008 seconds per epoch). The residual 205.401 seconds includes all non-update work and is not a measured pure initialization time.

Root selected beta 0.10 from the mature 150 evidence and authorized an unchanged own 150→200 continuation, fifty new science epochs. Startup policy verifies own-run resume from global 5700 with the approved rolling-checkpoint digest and unchanged objective/calibration. The independent SOURCE seed 43, paired diagnosis and physical public SOURCE 1→own 2 results below complete that bounded review.

## Unchanged beta 0.10 continuation to epoch 200

The own 150→200 continuation returned normally at epoch 200/global 7600, adding fifty science epochs and 3,209.630 inclusive charged seconds. The exact retained epoch 200 checkpoint is SHA256 `580b8a5e93a9c8cbf20285c6deb43db07758b6bfbc30a3ee48b8b5fbdcac5c1d`; learning is judged from the complete trajectory and both mature windows. Root and campaign native-window pooling agree within 2e-6.

Final 25/final 50 native TRAIN ratios are 0.993249/0.990410 and validation ratios 1.032062/1.031660. The individual epoch 200 native validation ratio is 1.030417. Previous individual epoch 100/150 failures 1.060623/1.062218 remain failures. Raw TRAIN preceding/final 25 means are 0.127458/0.127660, with final 50 slope +1.63196e-5 per epoch: a late plateau with a small sampled rise, not continued monotonic improvement.

Final 25/final 50 A/B/C medians are 0.995268/0.997467/0.842928 and 0.995268/1.013189/0.825914. Their means 0.945221/0.944790 pass the existing descriptive useful-coherence screen. The earlier A improvement has diminished near SOURCE, while finite-primary C remains useful; this is a family trade-off. Ratios of raw means at final 50 are A0.983339, grouped B1.009699 and C0.828932. A marginal/copula ratios are 0.984617/0.930019; grouped B same/cross ratios 1.037441/0.981309 and pooled 64 same/cross ratios 1.026704/0.965592 expose the mixed B result.

Endpoint aggregate final 25/final 50 ratios 1.025825/1.026004 satisfy the engineering corridor. Pressure ratios 1.065094/1.061399 exceed the strict field 1.05 reference while remaining below 1.10. The exact epoch 200 endpoint aggregate is 1.021460, with CH4/CO/T/U_1/p ratios 0.976005/1.026882/0.980280/1.004573/1.072493. These aggregate and field results are not interchangeable with native vector-field regression. SOURCE/seed 43 confirmation is reported below; expanded paired audits are reported below.

Active-C extraction from the unchanged complete history independently reconstructs the weighted family within 8.1e-8 per row. Final 50 self finite/essential/total ratios to their own SOURCE categories are 0.837381/0.695164/0.826910; mutual ratios are 0.838063/0.728246/0.830828. Thus both self and mutual finite discrepancy improve by about 16%, while essential terms also decrease. These are reference-specific raster descriptor comparisons, not physical causality or full multiparameter topology. H1 essential ratios with zero SOURCE terms remain undefined.

Complete synchronized update work averages 63.862 seconds/epoch over all 200, versus 65.249 inclusive charged seconds/epoch for the two combined P2 attempts. The final campaign totals 500 science epochs plus 2 public execution-check epochs and 33,862.956 inclusive charged seconds, including both offline envelopes. The unchanged SOURCE seed 43 recipe completed 150 epochs; no new strength, loss, controller or sampler is introduced.

## Independent SOURCE seed 43 confirmation at 150

The same beta 0.10 recipe independently initialized the original SOURCE and completed 150 epochs normally (5,700 updates), rather than continuing the seed 42 pilot. Its own SOURCE calibration remained frozen. Raw native TRAIN preceding/final 25 means are 0.130178/0.127087 and final 50 slope −8.78807e−5 per epoch. Native TRAIN final 25/final 50 ratios are 0.999084/0.993788; native validation ratios 1.015735/1.035266 satisfy the 1.05 window reference. Individual monitor failures remain separately visible in the complete traces.

Final 25/final 50 A/B/C medians are 0.965088/1.009396/0.816904 and 0.967542/1.009396/0.819802, passing the descriptive useful-family screen. Final 50 raw-mean A marginal/copula ratios 0.961055/0.966912 improve, while grouped B same/cross 1.037201/0.987132 remains mixed (pooled 64 same/cross 1.017227/0.950865 is a different estimator). Endpoint aggregates 1.015113/1.019079 pass the engineering corridor, and every field-window mean is below 1.05, including pressure ratios 1.034974/1.044865. Own-SOURCE active-C final 50 self finite/essential/total ratios are 0.830127/0.654230/0.817177; mutual ratios 0.830289/0.702882/0.821896. Root independently reconstructed every weighted group from its saved raw means and fixed scales, closing the active objective within 6.5e−8. This supports useful bounded confirmation without an all-family or all-field superiority claim.

Synchronized update work averaged 64.5622 seconds/epoch. Ordinary epochs averaged 64.5357 seconds of update work and diagnostic-boundary epochs 64.8003; these exclude evaluation and checkpoint I/O. The measured charged external envelope is 9,890.483 seconds (65.9366 seconds/epoch), with recorded calibration 7.284 seconds already included. Peak runner CUDA allocation was 20,411,220,480 bytes; sparse device-used samples reached 21,259 MiB, which is not a continuously observed peak. The exact retained 150 archive, rather than last.pt alone, is used for final paired diagnosis.

## Final paired native-RF validation

Both retained mature candidates were evaluated against SOURCE on the unchanged predeclared 256-snapshot validation panel, each under eight native draws and 4,096 queries per field. Each role therefore contributes 8,388,608 error elements per field (41,943,040 overall). Root independently reduced raw sums/counts, verified bin closure, production-loss parity and the residual accounting identity. The expanded pooled ratios are 0.990552 for selected seed 42/epoch 200 and 1.029957 for independent SOURCE seed 43/epoch 150, both below 1.05. Their unchanged original 32 final 25/final 50 native windows also pass. This supports native preservation for the required bounded protocol; it does not mean every checkpoint, field/time cell, draw or chronological stratum passes. Historical R5/P1 failures and isolated new failures remain visible.

| Field | Selected 200 expanded ratio | SOURCE 43/150 expanded ratio |
|---|---:|---:|
| CH4 | 1.038110 | 1.049875 |
| CO | 1.020812 | 1.029985 |
| T | 1.049160 | 1.048141 |
| U_1 | 1.017358 | 1.027113 |
| p | 0.951311 | 1.028311 |

Pressure improvement contributes strongly to selected 200 pooled risk while T is close to 1.05; aggregate preservation is not uniform improvement. Selected 200 chronological-quarter ratios are 0.994507/0.969071/0.973971/1.025541, and confirmation ratios 1.039473/1.013068/1.003019/1.063124. The last confirmation quarter exceeds 1.05 and is retained as descriptive heterogeneity, not a newly introduced all-strata veto. Draw ratios span 0.977632–1.002446 (selected) and 1.018844–1.039010 (confirmation), with descriptive standard deviations 0.008091/0.006915. Snapshot q 05/median/q 95 ratios are 0.803296/1.001822/1.229196 and 0.855525/1.023377/1.345852. These repeated correlated cases do not support independent-sample confidence or statistical significance.

| Field | Selected 200 ratios in artificial-time bins 0–.2 / .2–.4 / .4–.6 / .6–.8 / .8–1 | Confirmation 150, same bins |
|---|---|---|
| CH4 | 1.0112 / 1.0282 / 1.0567 / 1.0465 / 1.0367 | 1.0071 / 1.0304 / 1.0623 / 1.0729 / 1.0609 |
| CO | 1.0653 / 1.0448 / 1.0389 / 1.0108 / 0.9704 | 1.0574 / 1.0373 / 1.0447 / 1.0257 / 0.9996 |
| T | 1.0243 / 1.0460 / 1.0903 / 1.0742 / 1.0184 | 1.0101 / 1.0424 / 1.0853 / 1.0829 / 1.0190 |
| U_1 | 1.0258 / 1.0594 / 1.0556 / 1.0003 / 0.9616 | 1.0161 / 1.0481 / 1.0691 / 1.0218 / 0.9814 |
| p | 1.0015 / 1.0350 / 1.0161 / 0.9787 / 0.8912 | 1.0402 / 1.0682 / 1.0810 / 1.0558 / 0.9844 |

The accounting identity closes each saved raw sum within 1.82e−11 (selected) and 1.10e−11 (confirmation). Selected 200 loss change−0.009381 equals function-change energy 0.080556 plus residual-alignment term−0.089937; confirmation change+0.029745 equals 0.065521−0.035776. Local middle-time T/U_1 and confirmation pressure bins worsen while later selected pressure improves; this is exact squared-error accounting, not causality or an additional loss. All field/bin terms and raw counts remain in the compact [selected native arrays](../../cases/turbulent_combustion/runs/Test_1002/R6_diagnosis/native_validation_P2_beta010_seed42_age200_seedoffset_0_arrays.npz) and [confirmation native arrays](../../cases/turbulent_combustion/runs/Test_1002/R6_diagnosis/native_validation_P3_beta010_seed43_age150_seedoffset_0_arrays.npz), with independent reduction receipts under R6_root_reviews.

Confirmation original 32 pooled monitor failures occur at epochs 20/30/110/120 (1.087427/1.077151/1.058540/1.070583). Draw-level exceptions also occur at 70/80/100; they are labeled draw aggregates, not per-ID errors. Epoch 20 is an operational observation and historical visibility item, not a learning-selection gate. No strength, seed, checkpoint or sampler was changed to erase a failure.

Pooled raw squared-error sums are shown below; each entry has 8,388,608 elements in normalized model units. Each artificial-time bin divides by its own retained raw count, rather than assuming equal bin populations.

| Field | SOURCE raw sum | Selected200 raw sum | Confirmation150 raw sum |
|---|---:|---:|---:|
| CH4 | 1482656.886140 | 1539161.622144 | 1556604.511515 |
| CO | 5383805.724850 | 5495855.834854 | 5545239.964810 |
| T | 2225536.470255 | 2334944.039492 | 2332676.898238 |
| U_1 | 13832387.345320 | 14072490.698619 | 14207428.726628 |
| p | 18721464.398072 | 17809937.617625 | 19251487.333175 |

Fieldwise function-change energy / twice residual alignment / observed native-loss change are:

| Field | Selected200 energy / alignment / delta | Confirmation150 energy / alignment / delta |
|---|---|---|
| CH4 | +0.022435 / -0.015699 / +0.006736 | +0.019824 / -0.011009 / +0.008815 |
| CO | +0.069446 / -0.056089 / +0.013357 | +0.055348 / -0.036104 / +0.019244 |
| T | +0.028947 / -0.015905 / +0.013042 | +0.025910 / -0.013138 / +0.012772 |
| U_1 | +0.118419 / -0.089796 / +0.028623 | +0.092800 / -0.048092 / +0.044708 |
| p | +0.163535 / -0.272198 / -0.108662 | +0.133724 / -0.070540 / +0.063184 |

The [selected reduction receipt](../../cases/turbulent_combustion/runs/Test_1002/R6_root_reviews/native_validation_P2_beta010_seed42_age200_seedoffset_0_independent_review.json) and [confirmation reduction receipt](../../cases/turbulent_combustion/runs/Test_1002/R6_root_reviews/native_validation_P3_beta010_seed43_age150_seedoffset_0_independent_review.json) report both terms for all 25 field/time cells; the linked NPZ arrays preserve their unrounded raw sums and counts.

## Research context and limits

[ConFIG, ICLR2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/94e85561a342de88b559b72c9b29f638-Abstract-Conference.html) describes a constructed conflict-free gradient direction, which is inactive in this scalar recipe and supplies no guarantee for its AdamW displacement or held-out risk. [Xin et al., NeurIPS2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/580c4ec4738ff61d5862a122cdf139b6-Abstract-Conference.html) motivates tuned scalar comparisons without proving scalar weighting universally superior. [Flow Matching, ICLR2023](https://arxiv.org/abs/2210.02747) motivates vector-field regression; measured native RF risk and finite-sampler endpoint MSE remain different quantities. [L2-SP, ICML2018](https://proceedings.mlr.press/v80/li18a.html) motivated R5's optional source-distance prior, whose failures on this data remain controlling evidence. The paired residual and conditional squared-descriptor identities above are elementary accounting derivations, not new results attributed to these papers.

## Three predeclared native-grid cases

The fixed cases 8000/8499/8999 span the chronological validation segment. All 40300 query coordinates, physical GT/SOURCE values, normalization and sensor identities are preserved; T has 286/212/369 valid sensors, padded only for storage. Selected 200 normalized total MSE/SOURCE ratios are 0.845062/1.036116/0.957578, but individual fields are mixed. Physical relative-L2 is recalculated on unchanged uncentered physical values; units remain unknown, and pressure receives no new gauge subtraction. These three cases illustrate spatial behavior, not an independent all-case fidelity gate.

| Case | CH4 SOURCE→selected | CO SOURCE→selected | T SOURCE→selected | U_1 SOURCE→selected | p SOURCE→selected |
|---|---|---|---|---|---|
|8000|0.1381→0.1219|0.3548→0.3270|0.1367→0.1210|0.4029→0.3686|0.01208→0.01226|
|8499|0.2129→0.2190|0.4183→0.4709|0.1834→0.1840|1.0561→1.0284|0.02160→0.02360|
|8999|0.2090→0.2173|0.5140→0.5123|0.1813→0.1875|1.1445→1.2492|0.04331→0.03359|

The middle-case CO error and late-case U_1 error increase visibly; relatively small pressure error on an uncentered field does not prove small fluctuating-pressure error. The figures retain shared physical scales and signed residual scales across all three cases with directly displayed relative-L2 values. Root verified preserved-array parity and recalculated the reported physical errors from the retained raw arrays. A failed first export used an incorrect extra sensor-index dimension; a one-line diagnostic-helper correction restored the existing flattened-index contract, and the failed 6.721-second envelope is charged. Production training, descriptors and both completed native audits are unchanged.

## Exact user workflow

After the environment setup and SOURCE/config checks in the [operating guide](../../cases/turbulent_combustion/configs/posttrain/abc_upgrade_1002_r6_launch.md), the following is the prepared future user command. It was not executed by R6; the actual physical check instead used an output-only sibling with segment 1 and an absolute execution limit of 2.

```bash
export R6_REPO='/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction'
export R6_CASE="$R6_REPO/cases/turbulent_combustion"
export R6_CFG="$R6_CASE/configs/posttrain/abc_upgrade_1002_r6_5000ep.yaml"
export R6_STOP="$R6_CASE/runs/long_jobs/tc_abc_upgrade_1002_r6_beta010_5000ep.STOP_AFTER_CURRENT_SEGMENT"
rtk proxy python -u -m phycoflow_reconstruction.training.segmented \
  --config "$R6_CFG" --case-dir "$R6_CASE" \
  --segment-epochs 25 --allocation-hours 24 --stop-file "$R6_STOP"
```

To stop cleanly, use `rtk proxy touch "$R6_STOP"`; the current recovery segment finishes before normal return. After verifying the durable boundary, remove only that marker with `rtk proxy rm -- "$R6_STOP"` and rerun the identical command for own-state recovery. A 25-epoch segment takes about 27 minutes of update work plus overhead; configurable segment 1 offers shorter stop latency and is physically tested by the bounded sibling. The guide gives full environment setup, the exact bounded sibling procedure, retained LIVE epoch 200 checkpoint selection and manual unchanged-config resume commands. No daemon, scheduler, requeue or formal launcher was started.

Monitor raw native TRAIN stability, retained C self/mutual gains, A marginal/copula, mixed B blocks and endpoint field errors. A plateau is permitted; sustained deterioration warrants a clean boundary stop and review of retained checkpoints. A requested 5000 horizon does not imply every loss should keep decreasing or that 5000-epoch stability has been demonstrated.

## Tested public recovery and final resource accounting

The actual canonical 5000 YAML SHA is `90db16cf7791fd4ddff2aaa2beaeadac518bac8401c57ca24db6f131e740fd85`. An output-only sibling retained that 5000 horizon with `--segment-epochs 1 --until-epoch 2`; both public-module invocations used identical argv. The first initialized SOURCE, wrote the epoch 1 recovery state and returned 0 at global 38 after the marker stop. The unchanged second invocation loaded that own state and returned 0 at global 76/epoch 2. Both boundary statuses are correctly `integration_truncated`, not 5000 completion. The [physical attempt receipt](../../cases/turbulent_combustion/runs/Test_1002/R6_reports/public_epoch1_own2_attempts/src_ep1_own2_beta010_20261007_a/attempt_receipt.json) and [independent root review](../../cases/turbulent_combustion/runs/Test_1002/R6_root_reviews/public_physical_recovery_root_review.json) bind the actual execution.

The retained epoch 1 rolling proof SHA is `6425a0e1423624ee0b43a59036c3fd10cb70a485c36727a8fa7949df05bac428`; final epoch 2 last SHA is `89d6d73384c7f339818e1a7b8e394c23a684fac691e23a81a391114cf7b69787`. The proof was a hardlink to rolling last before atomic replacement; separately serialized epoch_001 is not presumed an alias. Root loaded both actual payloads on CPU: all 145 AdamW parameter counters advance 38→76; 290 floating moment tensors are finite, and model/optimizer layouts match the selected recipe. Normalization, family states/scales/calibration, data/config/SOURCE identities remain equal. Both contain one CUDA RNG state and valid CPU/Python/NumPy/sampler cursor states. Native/coherence private streams are counter-based from fixed seeds/offsets plus global step; no nonexistent private generator blob or cross-boundary equality of consumed RNG bytes is claimed. Together with the executed strict restore path and own-resume receipt, this verifies physical recovery; it adds no mature learning evidence.

| Work | New epochs | Inclusive charged seconds |
|---|---:|---:|
|SOURCE beta 0.25/seed 42→150|150|9865.064|
|SOURCE beta 0.10/seed 42→150|150|9840.116|
|Unchanged own 150→200|50|3209.630|
|Independent SOURCE beta 0.10/seed 43→150|150|9890.483|
|Retained diagnosis, initial conservative envelope|0|720.000|
|Final native/field diagnosis, four outer processes including failed export|0|123.026|
|Public SOURCE 1→own 2 observer, one outer envelope|2|214.637|
|**Total**|**502**|**33862.956 =9.406 hours**|

The public observer outer 214.637-second interval contains the 117.996/92.793-second wrapper intervals (210.789 combined); those inner timers are not charged again. Offline diagnosis totals 843.026 seconds (14.050 minutes), below 45 minutes. Science totals 500 epochs and checks 2, below 500/8/508 caps; lineage ages 150/200/150 and sibling 2 remain below 250. Every GPU process used physicalGPU1 with phycoflow_env. The initial 720-second estimate is intentionally conservative because original outer timing was unavailable; internal 285.850 seconds is diagnostic only. Exact charged categories and envelopes remain in the [final ledger](../../cases/turbulent_combustion/runs/Test_1002/R6_campaign/ledger.json).

| Science lineage | Ordinary update work, seconds/epoch | Diagnostic-boundary update work, seconds/epoch | Inclusive seconds/epoch |
|---|---:|---:|---:|
|beta 0.25/42,150|64.276|64.586|65.767|
|beta 0.10/42,200|63.815|64.287|65.249|
|beta 0.10/43,150|64.536|64.800|65.937|

These complete 38-update timers measure synchronized update work, not the full diagnostic-epoch wall duration. Monitor/checkpoint/loader/startup/teardown overhead is included only in external intervals; it was not independently timed per diagnostic epoch. SOURCE family calibration costs 7.244/7.493/7.284 seconds are already included. The same-GPU historical F100 versus new P1_100 update-work ratio 1.004593 is the qualified matched-envelope speed comparison; system load was not controlled. The approximately 51-second original historical run remains a separate operational reference and is not equated with 65 seconds. Workers remain 4 topology/4 numerical threads and 1 HDF5 loader, with batch 32, 38 updates/epoch and 4,096 queries preserved. No reduced workload is credited as speedup.

## Compact artifacts and preservation

Science retains rolling recovery states, exact 100/150/200 milestones where reached, and at most one useful additional feasible archive per lineage. Terminal selector-only compaction preserved model/optimizer/RNG/calibration payloads recursively, numerical histories and mature checkpoint bytes; exact inventories and receipts identify the pruned new-R6 paths. The public sibling retains the eight original recovery/selector payloads needed to preserve its unmodified tested state and epoch 1→own 2 evidence, rather than applying an untested post-check state rewrite. Those include the epoch 1 rolling proof and a separately serialized epoch_001; duplicate audit copies of model payloads were not created.

Final inode-deduplicated full model payloads are 21 files/1,795,735,060 bytes; including small reference/cache `.pt` files gives 1,820,065,630 bytes, below 2,147,483,648. The conservative bounded public all-pt transient forecast was 2,014,854,849 bytes; this is a forecast bound, not continuous physical peak measurement. The recorded P3 atomic compaction transient was 1,991,043,582 bytes. Root rehashed all 33 protected original/SOURCE model files and verified seven unrelated user changes remain untouched. The [final storage inventory](../../cases/turbulent_combustion/runs/Test_1002/R6_root_reviews/final_unique_pt_inventory.json), [final preservation audit](../../cases/turbulent_combustion/runs/Test_1002/R6_root_reviews/final_preservation_before_QA_cleanup.json) and [single cleanup manifest](../../cases/turbulent_combustion/runs/Test_1002/R6_root_reviews/R6_cleanup_manifest.json) retain the evidence. PDF masters and raw scientific arrays remain protected; disposable presentation QA rasters are removed after inspection. Test_1002 evidence and the three PDF masters remain local research artifacts; the durable code, report, profile and guide are versioned.

## Final interpretation

Implementation readiness, useful stable bounded reconstruction and required bounded native-RF preservation are supported separately. The selected recipe is SOURCE beta 0.10, no source-parameter retention, unchanged calibration meaning/descriptors/sampling/LR/zero warmup. The response to reduced global strength is measured: beta 0.25 retained A/C improvement but failed mature native windows, whereas beta 0.10 retained useful C and passed the required late-window/expanded native checks through selected 200 and independent SOURCE 150. This does not relabel any historical failure, require every diagnostic component to improve, or predict unexecuted 5000 learning. The public 5000 profile is accurately prepared and its bounded public recovery sibling is tested; the formal run remains unlaunched.
