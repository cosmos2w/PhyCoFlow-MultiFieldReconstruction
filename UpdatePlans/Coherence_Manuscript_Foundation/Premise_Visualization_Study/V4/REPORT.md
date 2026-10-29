# Coupled-field relationships beyond reconstruction error: a combustion evidence library

29 October 2026 · V4 · Paper-premise materials, complete candidate banks and reusable figure sources

## What this library establishes

Multi-field reconstruction has two distinct evaluation targets: individual fields should approach their references, and the reconstructed fields should retain the observed organization of their joint state. A vector of fieldwise L2 errors does not determine that organization. This library combines verified constructive controls, native-coordinate coupled-region views, and a complete 6-model comparison census to make that distinction inspectable.

The strongest new field-first examples are frames **9544** and **9111**: the model with higher error in both relevant fields nevertheless preserves the selected joint region better. In frame 9544, FFM has CO/T native-area-weighted L2 of 47.8%/15.5%, versus DiffusionPDE's 42.8%/14.0%, but high-CO/hot-T joint IoU is 0.199 versus 0.077. The unary CO and T region IoUs remain close. The point is a concrete disagreement between fieldwise accuracy and a coupled-state query, rather than a claim that either reconstruction is nearly exact.

Across the complete 100-frame cohort, 3,421/6,834 comparisons matching both fieldwise Euclidean L2 values within 20% have a conditional-relationship error contrast of at least 0.05 reference Y-field standard deviations; 3,271 also retain the model ordering across all five bin resolutions. A separate rank-based diagnostic finds 601/6,834 with at least 0.10 difference in rank-copula histogram TV error. These descriptive comparisons share frames, field pairs and models. Their counts characterize this fixed cohort, not independent trials or universal architecture failure rates.

The deliverable is deliberately broader than a final manuscript figure: five controlled galleries, seven coupled-region comparisons, eight detailed State comparisons, full-cohort diagnostics, and estimator/attribution audits. Every displayed preview has a PDF master. Exact arrays and selection decisions remain separate from plotting, so the evidence can be re-ranked and redrawn.

## 1. The paper argument: LPC motivates observable questions, and SSS makes them testable

The target is a conditional joint physical state inferred from sparse partial observations. Under the analyticity and existence assumptions in the theory document, LPC partitions mixed cumulants by order, distinct sites and distinct fields: Local collects one-site mixed terms; Pair collects two-field/two-site second-order terms; Collective collects the remaining mixed terms spanning multiple sites. This is an exact organization under its assumptions. The operational State–Scale–Shape closure selects finite observations of that richer law; it is not an invertible replacement for the law.

State asks which values occur together. Scale asks how amplitudes and signed alignment organize in spatial modes. Shape asks how field-defined regions intersect, connect and persist. A physically useful query should therefore point back to a recognizable field configuration: the CO-rich band relative to a CH4 transition, CO-rich material inside a hot region, or the continuity of a joint excursion. The present study assesses static snapshot organization. It does not identify reaction rates, causal coupling or satisfaction of governing equations.

A simple mechanism makes the premise more than a metric preference. If two unresolved fields satisfy V = U², their squared-error-optimal conditional means satisfy E[V|Y] − E[U|Y]² = Var(U|Y). With nonzero conditional variance, the pair of mean estimates leaves that nonlinear relation. This establishes a possible failure under explicit conditions; a generative model's single draw is not presumed to be a conditional mean. The strict controls below independently establish matched-error counterexamples without requiring that training-mechanism assumption.

Three distinctions prevent the diagnostics from becoming labels without content. A raw joint-distribution error includes marginal changes; the rank-copula audit separately probes rank association. A unary spectrum does not determine cross-field phase. A joint mask defect can inherit unary defects, so unary masks and their errors must remain visible. Perfect reconstruction would preserve suitable continuous coherence queries; the premise concerns finite and often appreciable errors, not an inconsistency at zero error.

Sources: [LPC theory](../../01_Theory/LPC_THEORY.md), [SSS computation](../../02_Computation/SSS_COMPUTATION.md), [post-training interface](../../03_Posttraining/POSTTRAINING_STRATEGY.md).

## 2. Constructive evidence: hold fieldwise error fixed, then inspect relationships

The strict controls progressively add preserved quantities. The nonlinear State construction matches each field MSE and shares the same reconstructed first field; one candidate satisfies v = u² while the other does not. The periodic Scale construction additionally preserves unary Fourier power exactly while relative phase changes. The Shape construction retains unary Betti numbers across fifteen thresholds while joint connectivity is one versus two. State and Shape do not claim identical marginals. Exact periodic Fourier identities do not transfer to the nonuniform combustion grid.

The real-field permutation controls preserve both unary empirical distributions and equal per-field Euclidean errors algebraically, while common and opposed operations change same-site pairing differently. They intentionally include cyclic index seams and unequal physical displacements on this nonuniform grid. The four calibrated noise/convolution/phase/warp families are output sensitivity tests. Their selective and sometimes weak responses are retained rather than treated as universal failures.

![Strict synthetic controls hold each field MSE fixed while nonlinear State, signed cross-spectrum, and joint connectivity differ. The Scale toy preserves unary power; the Shape toy preserves unary Betti numbers. These are separate contracts.](figures/controls_strict_toys.png)

**Figure 1.** Strict synthetic controls hold each field MSE fixed while nonlinear State, signed cross-spectrum, and joint connectivity differ. The Scale toy preserves unary power; the Shape toy preserves unary Betti numbers. These are separate contracts. [PDF master](figures/controls_strict_toys.pdf).

![A common versus opposed cyclic rearrangement of frame 9575 preserves both unary empirical distributions and identical field errors while changing CH4–CO pairing. The wrap seam and substantial CO error are explicit constructed features, not physical transport.](figures/controls_real_state.png)

**Figure 2.** A common versus opposed cyclic rearrangement of frame 9575 preserves both unary empirical distributions and identical field errors while changing CH4–CO pairing. The wrap seam and substantial CO error are explicit constructed features, not physical transport. [PDF master](figures/controls_real_state.pdf).

![Ten uniformly sampled saved frames, three field pairs and five cyclic shift sizes show how identical field errors and unchanged unary distributions permit different rank-binned joint occupation and signed correlation. Faint curves are frames and bold curves medians; shaded MSE ranges are not confidence intervals.](figures/controls_permutation_sweep.png)

**Figure 3.** Ten uniformly sampled saved frames, three field pairs and five cyclic shift sizes show how identical field errors and unchanged unary distributions permit different rank-binned joint occupation and signed correlation. Faint curves are frames and bold curves medians; shaded MSE ranges are not confidence intervals. [PDF master](figures/controls_permutation_sweep.pdf).

![A phase-amplitude sweep retains unary Fourier power and shared/opposed field-error equality while cross-products diverge. The independent Shape construction retains unary Betti numbers but different joint connectivity over fifteen thresholds. Large sweep amplitudes have large field errors.](figures/controls_strict_sweep.png)

**Figure 4.** A phase-amplitude sweep retains unary Fourier power and shared/opposed field-error equality while cross-products diverge. The independent Shape construction retains unary Betti numbers but different joint connectivity over fifteen thresholds. Large sweep amplitudes have large field errors. [PDF master](figures/controls_strict_sweep.pdf).

![Four calibrated synthetic intervention families on frame 9575 match all five normalized MSE values and restore observed temperature exactly. Their correlation responses differ and can be small. A visibly different error texture alone does not prove a large coupled-state defect.](figures/controls_real_families.png)

**Figure 5.** Four calibrated synthetic intervention families on frame 9575 match all five normalized MSE values and restore observed temperature exactly. Their correlation responses differ and can be small. A visibly different error texture alone does not prove a large coupled-state defect. [PDF master](figures/controls_real_families.pdf).

Control source data and checked contracts: [controls.npz](data/controls.npz), [numerical ledger](metadata/controls.json). The sweep retains chart-ready measurements rather than every shifted field.

## 3. Frozen model and data contracts

All candidate weights are the requested **last** checkpoint, frozen before the new extraction. A nominal `5000ep` directory does not establish completed training. DON, MIMONet and Senseiver are at epoch 5000; the requested GeoFNO is at epoch 2580 and the requested DiffusionPDE at epoch 1400. FFM is retained as the original reference endpoint. Identical previous predictions are reused only after matching checkpoint, configuration and prediction hashes. The three new extractions are DON, GeoFNO and DiffusionPDE.

The common cohort is the original 100 uniformly spaced frames in chronological TEST global IDs 9000–9999, at native nonuniform 100×403 sites and field order (CH4, CO, T, U_1, p). Each frame uses the same 256 saved temperature observations, original TRAIN normalization, full ordered native queries and original deterministic seed protocol. MIMONet retains ascending point-index sensor packing. Generative endpoints use one fixed draw, so this is not a conditional-diversity assessment.

| Model | Checkpoint epoch | Updates | Weights | Endpoint source | Record |
| --- | --- | --- | --- | --- | --- |
| FFM | 5000 | 315000 | EMA | verified prior arrays | [identity](metadata/gl_rbf_cq.json) |
| DON | 5000 | 315000 | live | new extraction | [identity](metadata/deeponet.json) |
| EDM | 1400 | 88200 | EMA | new extraction | [identity](metadata/diffusion_pde.json) |
| GeoFNO | 2580 | 322500 | live | new extraction | [identity](metadata/geofno.json) |
| MIMONet | 5000 | 315000 | live | verified prior arrays | [identity](metadata/mimonet.json) |
| Senseiver | 5000 | 315000 | live | verified prior arrays | [identity](metadata/senseiver.json) |

The model names describe the actual local adapters. DON uses branch/trunk basis reconstruction. MIMONet uses multiplicative branch/trunk merging and its T256 point-index interface. Senseiver uses latent cross-attention. The local GeoFNO adapter uses a rasterized neuraloperator FNO; it does not introduce a learned geometry warp. This requested DiffusionPDE run is explicitly **EDM**, with conditional U-Net, EMA, bf16 and 32-step Heun inference; it is different from the older DDPM/DDIM run discussed in earlier audits. The FFM endpoint uses EMA and four Euler steps. The source audit does not assert equivalence between a local model label and every part of an external original method.

Native observation treatment remains part of the contract: FFM retains endpoint consistency plus its final T clamp; EDM conditions and clamps observed T during denoising/final output; deterministic regressors receive no added clamp. Equal measurement sets therefore do not imply identical hard observation enforcement. Dense target fields are absent from new model inference inputs. [Source/code audit](metadata/model_audit.json), [frozen checkpoint identities](metadata/checkpoints.json).

### Actual fieldwise fidelity of the cohort

| Model | CH4 median L2 | CO median L2 | T median L2 | U_1 median L2 | p median L2 |
| --- | --- | --- | --- | --- | --- |
| FFM (GL-RBF-CQ) | 22.53% | 50.44% | 19.57% | 104.11% | 3.72% |
| DON (DeepONet) | 23.41% | 48.72% | 20.82% | 87.90% | 2.78% |
| DiffusionPDE (EDM) | 17.10% | 42.36% | 15.39% | 86.38% | 2.75% |
| GeoFNO | 20.97% | 49.47% | 18.34% | 87.27% | 2.15% |
| MIMONet | 27.05% | 52.17% | 24.10% | 96.56% | 3.26% |
| Senseiver | 23.58% | 52.45% | 20.62% | 96.90% | 3.28% |

These are unweighted native-node decoded Euclidean relative L2 values. CO and velocity errors remain substantial. Pressure has a large mean, so its small relative L2 is not evidence of accurate pressure fluctuations. No headline example is selected on pressure L2 alone. Dataset field/coordinate units have not been independently established; numeric temperature levels are dataset values.

## 4. Field-first atlas: where coupled states are retained, lost or invented

The event atlas starts from four empirical reference questions: CO-rich material in low CH4; CO-rich material inside the CH4 transition; CO-rich material in a hot region; and hot material in low CH4. CH4 cuts use 25% and 75% of the GT value range, avoiding tied-floor quantile artifacts. CO-rich is above the area-weighted GT 90th percentile and hot is above the area-weighted GT 75th percentile. Every state uses these same GT values; predicted quantiles are not used to force equal occupied area. Actual support is recorded.

Read each page from continuous fields to partner contours and then the joint footprint. Solid white contours show that column's partner region at the common GT cut; dashed gray contours retain the reference partner boundary. Green retains reference joint area, orange identifies missing joint area, and purple identifies extra predicted joint area. These are coupled-region geometry queries, not topology or a complete chemical constraint.

This atlas deliberately reports native-area-weighted decoded L2 alongside native-area region metrics. The full cohort and State atlas use unweighted decoded Euclidean L2. Both conventions are valid declared queries, but their values and match gates must not be substituted for each other.

| Frame | Coupled query | Models A / B | A field L2 | B field L2 | Joint IoU A/B |
| --- | --- | --- | --- | --- | --- |
| 9393 | Low CH4 / high CO | Senseiver / MIMONet | 19.2/42.3% | 21.9/44.7% | 0.480/0.000 |
| 9898 | CH4 transition / high CO | DON / EDM | 22.4/48.5% | 18.1/39.0% | 0.015/0.250 |
| 9454 | High CO / hot T | DON / EDM | 48.2/22.7% | 43.1/19.2% | 0.026/0.421 |
| 9817 | Low CH4 / hot T | FFM / MIMONet | 20.6/15.9% | 24.6/19.5% | 0.731/0.130 |
| 9948 | Low CH4 / hot T | GeoFNO / DON | 21.0/20.0% | 24.3/23.9% | 0.749/0.202 |
| 9544 | High CO / hot T | FFM / EDM | 47.8/15.5% | 42.8/14.0% | 0.199/0.077 |
| 9111 | Low CH4 / high CO | GeoFNO / EDM | 17.6/44.6% | 18.5/44.8% | 0.283/0.385 |

![Frame 9544, highCO_hotT, FFM (GL-RBF-CQ) versus DiffusionPDE (EDM). Native-area field L2 is 47.8%/15.5% versus 42.8%/14.0%; joint IoU 0.199 versus 0.077. Unary region IoUs are 0.408/0.810 versus 0.391/0.830; GT joint support is 3.69% of native area. All displayed model pairs meet the 20% per-field area-L2 matching gate. Solid white is the current partner boundary; dashed gray is its GT reference. The higher-L2 model has higher joint IoU in both-field dominance, while unary IoUs are close; they are not exactly controlled.](figures/events_06_highCO_hotT.png)

**Figure 6.** Frame 9544, highCO_hotT, FFM (GL-RBF-CQ) versus DiffusionPDE (EDM). Native-area field L2 is 47.8%/15.5% versus 42.8%/14.0%; joint IoU 0.199 versus 0.077. Unary region IoUs are 0.408/0.810 versus 0.391/0.830; GT joint support is 3.69% of native area. All displayed model pairs meet the 20% per-field area-L2 matching gate. Solid white is the current partner boundary; dashed gray is its GT reference. The higher-L2 model has higher joint IoU in both-field dominance, while unary IoUs are close; they are not exactly controlled. [PDF master](figures/events_06_highCO_hotT.pdf).

![Frame 9111, lowCH4_highCO, GeoFNO versus DiffusionPDE (EDM). Native-area field L2 is 17.6%/44.6% versus 18.5%/44.8%; joint IoU 0.283 versus 0.385. Unary region IoUs are 0.870/0.430 versus 0.904/0.431; GT joint support is 8.83% of native area. All displayed model pairs meet the 20% per-field area-L2 matching gate. Solid white is the current partner boundary; dashed gray is its GT reference. The higher-L2 model has higher joint IoU in both-field dominance, while unary IoUs are close; they are not exactly controlled.](figures/events_07_lowCH4_highCO.png)

**Figure 7.** Frame 9111, lowCH4_highCO, GeoFNO versus DiffusionPDE (EDM). Native-area field L2 is 17.6%/44.6% versus 18.5%/44.8%; joint IoU 0.283 versus 0.385. Unary region IoUs are 0.870/0.430 versus 0.904/0.431; GT joint support is 8.83% of native area. All displayed model pairs meet the 20% per-field area-L2 matching gate. Solid white is the current partner boundary; dashed gray is its GT reference. The higher-L2 model has higher joint IoU in both-field dominance, while unary IoUs are close; they are not exactly controlled. [PDF master](figures/events_07_lowCH4_highCO.pdf).

![Frame 9393, lowCH4_highCO, Senseiver versus MIMONet. Native-area field L2 is 19.2%/42.3% versus 21.9%/44.7%; joint IoU 0.480 versus 0.000. Unary region IoUs are 0.913/0.501 versus 0.843/0.000; GT joint support is 9.25% of native area. All displayed model pairs meet the 20% per-field area-L2 matching gate. Solid white is the current partner boundary; dashed gray is its GT reference. MIMONet nearly loses the CO-rich unary region, so this dramatic joint failure has a clear unary contribution.](figures/events_01_lowCH4_highCO.png)

**Figure 8.** Frame 9393, lowCH4_highCO, Senseiver versus MIMONet. Native-area field L2 is 19.2%/42.3% versus 21.9%/44.7%; joint IoU 0.480 versus 0.000. Unary region IoUs are 0.913/0.501 versus 0.843/0.000; GT joint support is 9.25% of native area. All displayed model pairs meet the 20% per-field area-L2 matching gate. Solid white is the current partner boundary; dashed gray is its GT reference. MIMONet nearly loses the CO-rich unary region, so this dramatic joint failure has a clear unary contribution. [PDF master](figures/events_01_lowCH4_highCO.pdf).

![Frame 9898, transitionCH4_highCO, DON (DeepONet) versus DiffusionPDE (EDM). Native-area field L2 is 22.4%/48.5% versus 18.1%/39.0%; joint IoU 0.015 versus 0.250. Unary region IoUs are 0.135/0.177 versus 0.336/0.526; GT joint support is 0.94% of native area. All displayed model pairs meet the 20% per-field area-L2 matching gate. Solid white is the current partner boundary; dashed gray is its GT reference. The reference transition intersection is relatively small and should not be presented as a domain-wide event.](figures/events_02_transitionCH4_highCO.png)

**Figure 9.** Frame 9898, transitionCH4_highCO, DON (DeepONet) versus DiffusionPDE (EDM). Native-area field L2 is 22.4%/48.5% versus 18.1%/39.0%; joint IoU 0.015 versus 0.250. Unary region IoUs are 0.135/0.177 versus 0.336/0.526; GT joint support is 0.94% of native area. All displayed model pairs meet the 20% per-field area-L2 matching gate. Solid white is the current partner boundary; dashed gray is its GT reference. The reference transition intersection is relatively small and should not be presented as a domain-wide event. [PDF master](figures/events_02_transitionCH4_highCO.pdf).

![Frame 9454, highCO_hotT, DON (DeepONet) versus DiffusionPDE (EDM). Native-area field L2 is 48.2%/22.7% versus 43.1%/19.2%; joint IoU 0.026 versus 0.421. Unary region IoUs are 0.246/0.533 versus 0.517/0.771; GT joint support is 6.36% of native area. All displayed model pairs meet the 20% per-field area-L2 matching gate. Solid white is the current partner boundary; dashed gray is its GT reference.](figures/events_03_highCO_hotT.png)

**Figure 10.** Frame 9454, highCO_hotT, DON (DeepONet) versus DiffusionPDE (EDM). Native-area field L2 is 48.2%/22.7% versus 43.1%/19.2%; joint IoU 0.026 versus 0.421. Unary region IoUs are 0.246/0.533 versus 0.517/0.771; GT joint support is 6.36% of native area. All displayed model pairs meet the 20% per-field area-L2 matching gate. Solid white is the current partner boundary; dashed gray is its GT reference. [PDF master](figures/events_03_highCO_hotT.pdf).

![Frame 9817, lowCH4_hotT, FFM (GL-RBF-CQ) versus MIMONet. Native-area field L2 is 20.6%/15.9% versus 24.6%/19.5%; joint IoU 0.731 versus 0.130. Unary region IoUs are 0.910/0.731 versus 0.869/0.130; GT joint support is 25.00% of native area. All displayed model pairs meet the 20% per-field area-L2 matching gate. Solid white is the current partner boundary; dashed gray is its GT reference.](figures/events_04_lowCH4_hotT.png)

**Figure 11.** Frame 9817, lowCH4_hotT, FFM (GL-RBF-CQ) versus MIMONet. Native-area field L2 is 20.6%/15.9% versus 24.6%/19.5%; joint IoU 0.731 versus 0.130. Unary region IoUs are 0.910/0.731 versus 0.869/0.130; GT joint support is 25.00% of native area. All displayed model pairs meet the 20% per-field area-L2 matching gate. Solid white is the current partner boundary; dashed gray is its GT reference. [PDF master](figures/events_04_lowCH4_hotT.pdf).

![Frame 9948, lowCH4_hotT, GeoFNO versus DON (DeepONet). Native-area field L2 is 21.0%/20.0% versus 24.3%/23.9%; joint IoU 0.749 versus 0.202. Unary region IoUs are 0.881/0.749 versus 0.739/0.202; GT joint support is 25.00% of native area. All displayed model pairs meet the 20% per-field area-L2 matching gate. Solid white is the current partner boundary; dashed gray is its GT reference.](figures/events_05_lowCH4_hotT.png)

**Figure 12.** Frame 9948, lowCH4_hotT, GeoFNO versus DON (DeepONet). Native-area field L2 is 21.0%/20.0% versus 24.3%/23.9%; joint IoU 0.749 versus 0.202. Unary region IoUs are 0.881/0.749 versus 0.739/0.202; GT joint support is 25.00% of native area. All displayed model pairs meet the 20% per-field area-L2 matching gate. Solid white is the current partner boundary; dashed gray is its GT reference. [PDF master](figures/events_05_lowCH4_hotT.pdf).

Frame 9544 is a useful opening because the two CO and T region IoUs are similar yet their intersection overlaps the reference differently. Frame 9111 supplies a second model pairing and reverses the model role: DiffusionPDE is worse in both field errors but better for the low-CH4/high-CO footprint. Together with the preserved-marginal controls and rank-copula audit, these examples support a relational premise beyond comparing field error magnitudes. They still do not isolate every unary contribution.

Other atlas pages are retained for different purposes. Frame 9393 shows why a spectacular joint-mask loss can partly arise from unary amplitude failure. Frame 9898 makes the narrow transition query visible. The lower-CH4/hot-T pages supply lower-error alternatives and model coverage. The selections are exploratory, explicitly indexed, and accompanied by all 2,400 event-state rows and 6,000 event/model-pair rows. Across the four declared events there are 232 support-qualified, 20%-matched comparisons where a model worse in both area-L2 values has joint IoU at least 0.10 better. These comparisons overlap and are not independent repetitions. [Event ledger](metadata/events_summary.json), [all event states](data/events_metrics.csv), [all event comparisons](data/events_pairs.csv).

## 5. State atlas: connect visible field changes to conditional and joint occupation

The State pages place the continuous fields and their common-cut partner contours before the conditional curves. Their missing/excess joint maps use a different declared 70th-percentile/range-midpoint slice from the event atlas, so the two masks should not be compared as if they were identical. The conditional query is E[Y | X in bin], with each state assigned by its own X values, fixed GT bin edges, common minimum 64-node bin support, and GT occupancy weights. Unsupported bins are omitted. Full bin counts remain in retained arrays. Five resolutions (8, 10, 12, 16, 20) assess ordering stability.

The raw 2D histogram and its marginal histograms respond to absolute states. The empirical-rank copula uses each state's own marginal ranks; it adds a dependence audit, while tied values and finite bins remain part of the estimator. The model-minus-GT joint occupation panels retain a common scale within each comparison. Neither a raw conditional difference nor a joint histogram alone isolates dependence.

![Frame 9655, CO-T, FFM (GL-RBF-CQ) versus DON (DeepONet). Unweighted Euclidean field L2 is 40.9%/15.9% versus 44.5%/18.3%. Conditional discrepancy is 0.01774 versus 0.14519 normalized Y units, common GT support 95.14%; the ordering is stable over all five tested bin counts. Rank-copula TV-to-GT is 0.323 versus 0.518. Clouds use fixed per-field shared limits and white current-partner contours; footprint and occupation differences identify the affected coupled configuration. Selection is exploratory.](figures/example_01.png)

**Figure 13.** Frame 9655, CO-T, FFM (GL-RBF-CQ) versus DON (DeepONet). Unweighted Euclidean field L2 is 40.9%/15.9% versus 44.5%/18.3%. Conditional discrepancy is 0.01774 versus 0.14519 normalized Y units, common GT support 95.14%; the ordering is stable over all five tested bin counts. Rank-copula TV-to-GT is 0.323 versus 0.518. Clouds use fixed per-field shared limits and white current-partner contours; footprint and occupation differences identify the affected coupled configuration. Selection is exploratory. [PDF master](figures/example_01.pdf).

![Frame 9090, CO-T, FFM (GL-RBF-CQ) versus DiffusionPDE (EDM). Unweighted Euclidean field L2 is 47.0%/16.2% versus 41.6%/14.4%. Conditional discrepancy is 0.07916 versus 0.03936 normalized Y units, common GT support 99.76%; the ordering is stable over all five tested bin counts. Rank-copula TV-to-GT is 0.463 versus 0.422. Clouds use fixed per-field shared limits and white current-partner contours; footprint and occupation differences identify the affected coupled configuration. Selection is exploratory.](figures/example_02.png)

**Figure 14.** Frame 9090, CO-T, FFM (GL-RBF-CQ) versus DiffusionPDE (EDM). Unweighted Euclidean field L2 is 47.0%/16.2% versus 41.6%/14.4%. Conditional discrepancy is 0.07916 versus 0.03936 normalized Y units, common GT support 99.76%; the ordering is stable over all five tested bin counts. Rank-copula TV-to-GT is 0.463 versus 0.422. Clouds use fixed per-field shared limits and white current-partner contours; footprint and occupation differences identify the affected coupled configuration. Selection is exploratory. [PDF master](figures/example_02.pdf).

![Frame 9645, CH4-CO, FFM (GL-RBF-CQ) versus GeoFNO. Unweighted Euclidean field L2 is 21.2%/44.8% versus 20.7%/46.3%. Conditional discrepancy is 0.08240 versus 0.01497 normalized Y units, common GT support 100.00%; the ordering is stable over all five tested bin counts. Rank-copula TV-to-GT is 0.730 versus 0.715. Clouds use fixed per-field shared limits and white current-partner contours; footprint and occupation differences identify the affected coupled configuration. Selection is exploratory.](figures/example_03.png)

**Figure 15.** Frame 9645, CH4-CO, FFM (GL-RBF-CQ) versus GeoFNO. Unweighted Euclidean field L2 is 21.2%/44.8% versus 20.7%/46.3%. Conditional discrepancy is 0.08240 versus 0.01497 normalized Y units, common GT support 100.00%; the ordering is stable over all five tested bin counts. Rank-copula TV-to-GT is 0.730 versus 0.715. Clouds use fixed per-field shared limits and white current-partner contours; footprint and occupation differences identify the affected coupled configuration. Selection is exploratory. [PDF master](figures/example_03.pdf).

![Frame 9666, CH4-CO, FFM (GL-RBF-CQ) versus MIMONet. Unweighted Euclidean field L2 is 23.3%/47.2% versus 25.8%/47.2%. Conditional discrepancy is 0.07776 versus 0.19778 normalized Y units, common GT support 100.00%; the ordering is stable over all five tested bin counts. Rank-copula TV-to-GT is 0.692 versus 0.714. Clouds use fixed per-field shared limits and white current-partner contours; footprint and occupation differences identify the affected coupled configuration. Selection is exploratory.](figures/example_04.png)

**Figure 16.** Frame 9666, CH4-CO, FFM (GL-RBF-CQ) versus MIMONet. Unweighted Euclidean field L2 is 23.3%/47.2% versus 25.8%/47.2%. Conditional discrepancy is 0.07776 versus 0.19778 normalized Y units, common GT support 100.00%; the ordering is stable over all five tested bin counts. Rank-copula TV-to-GT is 0.692 versus 0.714. Clouds use fixed per-field shared limits and white current-partner contours; footprint and occupation differences identify the affected coupled configuration. Selection is exploratory. [PDF master](figures/example_04.pdf).

![Frame 9514, CO-T, FFM (GL-RBF-CQ) versus Senseiver. Unweighted Euclidean field L2 is 46.9%/16.9% versus 42.5%/15.8%. Conditional discrepancy is 0.02944 versus 0.07329 normalized Y units, common GT support 99.36%; the ordering is stable over all five tested bin counts. Rank-copula TV-to-GT is 0.375 versus 0.416. Clouds use fixed per-field shared limits and white current-partner contours; footprint and occupation differences identify the affected coupled configuration. Selection is exploratory.](figures/example_05.png)

**Figure 17.** Frame 9514, CO-T, FFM (GL-RBF-CQ) versus Senseiver. Unweighted Euclidean field L2 is 46.9%/16.9% versus 42.5%/15.8%. Conditional discrepancy is 0.02944 versus 0.07329 normalized Y units, common GT support 99.36%; the ordering is stable over all five tested bin counts. Rank-copula TV-to-GT is 0.375 versus 0.416. Clouds use fixed per-field shared limits and white current-partner contours; footprint and occupation differences identify the affected coupled configuration. Selection is exploratory. [PDF master](figures/example_05.pdf).

![Frame 9363, CH4-CO, DON (DeepONet) versus DiffusionPDE (EDM). Unweighted Euclidean field L2 is 21.7%/43.4% versus 18.0%/39.2%. Conditional discrepancy is 0.22173 versus 0.11534 normalized Y units, common GT support 100.00%; the ordering is stable over all five tested bin counts. Rank-copula TV-to-GT is 0.711 versus 0.668. Clouds use fixed per-field shared limits and white current-partner contours; footprint and occupation differences identify the affected coupled configuration. Selection is exploratory.](figures/example_06.png)

**Figure 18.** Frame 9363, CH4-CO, DON (DeepONet) versus DiffusionPDE (EDM). Unweighted Euclidean field L2 is 21.7%/43.4% versus 18.0%/39.2%. Conditional discrepancy is 0.22173 versus 0.11534 normalized Y units, common GT support 100.00%; the ordering is stable over all five tested bin counts. Rank-copula TV-to-GT is 0.711 versus 0.668. Clouds use fixed per-field shared limits and white current-partner contours; footprint and occupation differences identify the affected coupled configuration. Selection is exploratory. [PDF master](figures/example_06.pdf).

![Frame 9625, CH4-CO, DON (DeepONet) versus GeoFNO. Unweighted Euclidean field L2 is 23.3%/44.7% versus 23.6%/44.7%. Conditional discrepancy is 0.18372 versus 0.04703 normalized Y units, common GT support 100.00%; the ordering is stable over all five tested bin counts. Rank-copula TV-to-GT is 0.700 versus 0.660. Clouds use fixed per-field shared limits and white current-partner contours; footprint and occupation differences identify the affected coupled configuration. Selection is exploratory.](figures/example_07.png)

**Figure 19.** Frame 9625, CH4-CO, DON (DeepONet) versus GeoFNO. Unweighted Euclidean field L2 is 23.3%/44.7% versus 23.6%/44.7%. Conditional discrepancy is 0.18372 versus 0.04703 normalized Y units, common GT support 100.00%; the ordering is stable over all five tested bin counts. Rank-copula TV-to-GT is 0.700 versus 0.660. Clouds use fixed per-field shared limits and white current-partner contours; footprint and occupation differences identify the affected coupled configuration. Selection is exploratory. [PDF master](figures/example_07.pdf).

![Frame 9978, CH4-CO, DON (DeepONet) versus MIMONet. Unweighted Euclidean field L2 is 22.5%/46.2% versus 20.7%/45.4%. Conditional discrepancy is 0.10144 versus 0.15586 normalized Y units, common GT support 100.00%; the ordering is stable over all five tested bin counts. Rank-copula TV-to-GT is 0.740 versus 0.739. Clouds use fixed per-field shared limits and white current-partner contours; footprint and occupation differences identify the affected coupled configuration. Selection is exploratory.](figures/example_08.png)

**Figure 20.** Frame 9978, CH4-CO, DON (DeepONet) versus MIMONet. Unweighted Euclidean field L2 is 22.5%/46.2% versus 20.7%/45.4%. Conditional discrepancy is 0.10144 versus 0.15586 normalized Y units, common GT support 100.00%; the ordering is stable over all five tested bin counts. Rank-copula TV-to-GT is 0.740 versus 0.739. Clouds use fixed per-field shared limits and white current-partner contours; footprint and occupation differences identify the affected coupled configuration. Selection is exploratory. [PDF master](figures/example_08.pdf).

## 6. Cohort evidence: how often does the accuracy ordering fail for these queries?

Every unordered field pair, frame and model pair is retained: 10 × 100 × 15 = 15,000 comparisons. The new screen removes the old requirement that both models lie below their own cohort medians. Instead it reports nested 10%, 20% and 40% per-field L2 match strata, preserving poorer-fidelity cases with their literal errors. A match gap is |eA−eB|/max(eA,eB) for each field. A model is L2-dominant only if its error is higher in both fields; an ordering reversal occurs when its descriptor error is lower.

The table below separates any-order reversals from material contrasts. A conditional contrast of 0.05 means five percent of the GT Y-field spatial standard deviation; copula contrast 0.10 is a difference in bounded rank-histogram TV errors. These thresholds organize an exploratory evidence bank and are not significance tests. The normalized conditional effect is the difference of two errors-to-GT, rather than the direct distance between their conditional curves.

| L2 gate | Eligible n | State gap >=0.05 SD | Also bin-stable | Copula gap >=0.10 | Material inversions / dominant n |
| --- | --- | --- | --- | --- | --- |
| all | 15000 | 9231 | 8973 | 1729 | 1085/10268 |
| matched_40 | 12672 | 7276 | 7025 | 1344 | 1012/8508 |
| matched_20 | 6834 | 3421 | 3271 | 601 | 631/4298 |
| matched_10 | 2548 | 1128 | 1077 | 210 | 248/1487 |

![Fraction of comparisons where the model with worse L2 in both fields has lower descriptor error. Labels show eligible comparison counts across frames and field pairs; they are not independent samples. The inversion heatmap counts any strict ordering reversal; material-effect counts are reported separately above.](figures/inversion_heatmaps.png)

**Figure 21.** Fraction of comparisons where the model with worse L2 in both fields has lower descriptor error. Labels show eligible comparison counts across frames and field pairs; they are not independent samples. The inversion heatmap counts any strict ordering reversal; material-effect counts are reported separately above. [PDF master](figures/inversion_heatmaps.pdf).

![Absolute model-pair discrepancy differences across nested fieldwise L2 match gates. Boxes summarize all supported comparisons; n counts are descriptive and share frames. The inversion heatmap counts any strict ordering reversal; material-effect counts are reported separately above.](figures/fidelity_strata.png)

**Figure 22.** Absolute model-pair discrepancy differences across nested fieldwise L2 match gates. Boxes summarize all supported comparisons; n counts are descriptive and share frames. The inversion heatmap counts any strict ordering reversal; material-effect counts are reported separately above. [PDF master](figures/fidelity_strata.pdf).

![Conditional mean discrepancy versus paired-field relative L2, on support shared by GT and the two compared models. Points may repeat as comparison support changes. The inversion heatmap counts any strict ordering reversal; material-effect counts are reported separately above.](figures/cohort_state.png)

**Figure 23.** Conditional mean discrepancy versus paired-field relative L2, on support shared by GT and the two compared models. Points may repeat as comparison support changes. The inversion heatmap counts any strict ordering reversal; material-effect counts are reported separately above. [PDF master](figures/cohort_state.pdf).

![Joint occupation TV versus mean paired-field relative L2 for ten pairs and models. All frames are included. Each point is a paired snapshot; comparisons share frames. The inversion heatmap counts any strict ordering reversal; material-effect counts are reported separately above.](figures/cohort_joint_tv.png)

**Figure 24.** Joint occupation TV versus mean paired-field relative L2 for ten pairs and models. All frames are included. Each point is a paired snapshot; comparisons share frames. The inversion heatmap counts any strict ordering reversal; material-effect counts are reported separately above. [PDF master](figures/cohort_joint_tv.pdf).

![Rank-copula TV versus mean paired-field relative L2 for ten pairs and models. All frames are included. Each point is a paired snapshot; comparisons share frames. The inversion heatmap counts any strict ordering reversal; material-effect counts are reported separately above.](figures/cohort_copula_tv.png)

**Figure 25.** Rank-copula TV versus mean paired-field relative L2 for ten pairs and models. All frames are included. Each point is a paired snapshot; comparisons share frames. The inversion heatmap counts any strict ordering reversal; material-effect counts are reported separately above. [PDF master](figures/cohort_copula_tv.pdf).

The breadth claim is specific: accuracy–relationship disagreement is repeatedly observed across this declared model/frame/query collection, rather than appearing only in the original 9867 example. The data do not establish that all architectures fail, that every plausible reconstruction is physically inconsistent, or that the numerical rate transfers to other datasets. Chronologically related snapshots, repeated field pairs and repeated model comparisons preclude interpreting the row count as independent sample size. No checkpoint was selected using these relationship scores.

Complete outputs: [candidate CSV](data/metrics/candidates.csv), [frame descriptors](data/metrics/frame_metrics.csv), [material contrasts by pair](data/metrics/material_prevalence.csv), [cohort table](data/metrics/cohort.csv), [all definitions and hashes](data/metrics/definitions.json), [exploratory candidate index](data/metrics/candidate_index.json).

## 7. Scale and Shape: complementary probes with explicit attribution

The visualization Scale estimator is a native-coordinate, trapezoid-area NUDFT with mean removal, Hann taper and fixed physical octave bands. Signed complex cross-products are normalized by unary powers, and bands require at least 1% power in each reference and compared field. Unary spectra remain alongside cross-products. The selected comparison is independently recomputed with a rectangular window and common two-window support. The visible ordering changes by band/window: it supplies a sensitivity audit, not a robust model ranking or an isolated mutual-Scale failure.

This estimator differs from the training code's graph-basis, ensemble same-mode coherence and cross-band energy statistics. In particular, the latter magnitude-squared statistics can lose sign, and cross-band energy covariance is not the full second-order LPC Pair block. The stronger signed diagnostic is useful here because the premise study is testing information that a descriptor may omit; it is not silently replacing the historical training objective.

![Frame 9655, CO-T, gl_rbf_cq versus deeponet: unary spectral fractions and signed complex cross ratios over bands with common 1% support. Hann-window native-coordinate quadrature; rectangular-window sensitivity uses bands supported under both windows. Model ordering is window/band dependent; unary spectra also differ. This is an estimator-sensitivity illustration.](figures/example_scale.png)

**Figure 26.** Frame 9655, CO-T, gl_rbf_cq versus deeponet: unary spectral fractions and signed complex cross ratios over bands with common 1% support. Hann-window native-coordinate quadrature; rectangular-window sensitivity uses bands supported under both windows. Model ordering is window/band dependent; unary spectra also differ. This is an estimator-sensitivity illustration. [PDF master](figures/example_scale.pdf).

![Signed complex cross-band discrepancy versus paired-field relative L2, on support shared by GT and the two compared models. Points may repeat as comparison support changes. The cohort census uses Hann only; two-window robustness is assessed for the displayed selected case, not every row.](figures/cohort_scale.png)

**Figure 27.** Signed complex cross-band discrepancy versus paired-field relative L2, on support shared by GT and the two compared models. Points may repeat as comparison support changes. The cohort census uses Hann only; two-window robustness is assessed for the displayed selected case, not every row. [PDF master](figures/cohort_scale.pdf).

Shape retains geometric overlap separately from topology. The cohort's topology summary is the absolute discrepancy in four-connected joint excursion component counts, over three GT level settings and three native-area floors (0.003, 0.006, 0.010 of domain area). It is not persistent homology. CH4–CO and CH4–T use CH4-low and partner-high slices; other pairs use high–high. Actual thresholds, GT support and predicted occupancy are saved; supported cohort summaries require nontrivial GT joint area across slices.

The real-field pages establish coupled-region organization and its relation to fieldwise accuracy. Component-count differences may arise from unary amplitude, unary connectivity, thresholding or joint arrangement. Their attribution controls are shown rather than asserting a pure mutual-topology positive. The strict Shape toy remains the independently controlled proof that equal unary topology can coexist with different joint topology. V3's unary-inheritance negative findings remain valid and are not overwritten.

![Frame 9090, CO-T, gl_rbf_cq versus diffusion_pde: joint-region geometry, unary controls, and joint component discrepancies across GT thresholds and native-area floors; GT joint area is at least 0.001 of domain at every level. Component counts are not persistence.](figures/example_shape.png)

**Figure 28.** Frame 9090, CO-T, gl_rbf_cq versus diffusion_pde: joint-region geometry, unary controls, and joint component discrepancies across GT thresholds and native-area floors; GT joint area is at least 0.001 of domain at every level. Component counts are not persistence. [PDF master](figures/example_shape.pdf).

![Joint mask Jaccard distance versus mean paired-field relative L2 for ten pairs and models. Shape points require GT joint area at least 0.001 of domain at every threshold. Each point is a paired snapshot; comparisons share frames.](figures/cohort_shape_geometry_mean.png)

**Figure 29.** Joint mask Jaccard distance versus mean paired-field relative L2 for ten pairs and models. Shape points require GT joint area at least 0.001 of domain at every threshold. Each point is a paired snapshot; comparisons share frames. [PDF master](figures/cohort_shape_geometry_mean.pdf).

![Joint component count discrepancy versus mean paired-field relative L2 for ten pairs and models. Shape points require GT joint area at least 0.001 of domain at every threshold. Each point is a paired snapshot; comparisons share frames.](figures/cohort_shape_component_abs_mean.png)

**Figure 30.** Joint component count discrepancy versus mean paired-field relative L2 for ten pairs and models. Shape points require GT joint area at least 0.001 of domain at every threshold. Each point is a paired snapshot; comparisons share frames. [PDF master](figures/cohort_shape_component_abs_mean.pdf).

## 8. What this evidence motivates for post-training

A common differentiable reconstructed endpoint permits the same relational queries to be applied to different native predictors. Paired TRAIN references supply relationship targets, and SOURCE-relative per-field risks express the allowed fidelity change. This separates the scientific query from architecture-specific pretraining objectives. Scalarization and ConFIG are solver choices inside that interface, rather than the scientific premise itself.

The present V4 work performs inference and diagnostics only. It supplies candidate queries and failure examples that can motivate subsequent refinement and its independent evaluation. It does not add a new post-training efficacy result. Existing Premise_Add endpoint results, complete routine histories and execution-cost probes retain their own observation/solver/selection contracts; they must not be concatenated with this T256 TEST cohort as a before/after experiment. Paired target use, selected versus final checkpoints, full history and total cost remain separate reporting questions.

**Paper-facing premise wording:** “Across pretrained multi-field reconstruction models, similar fieldwise error can coexist with different thermochemical associations and coupled-region organization. Controlled examples preserve fieldwise errors and selected unary statistics while changing cross-field relationships; a fixed combustion cohort exhibits repeated accuracy–relationship ordering disagreements. These observations motivate complementary data-derived State, Scale and Shape queries for fidelity-constrained refinement.”

“Physical coherence” here denotes agreement with the declared empirical coupled-state observations. A governing-equation, causal or cross-dataset claim requires additional evidence.

## 9. Execution, preservation and reuse

The original GPU2 EDM training was interrupted with SIGINT after process ownership, command, working directory and physical UUID were verified. Its last checkpoint remains epoch 1400; the completed train history extends to epoch 1403, and available validation/preview history to epoch 1400. The following curve labels these distinct histories. No training was resumed, and model extraction used only physical GPU2. Historical run files and all V1/V2/V3/Add evidence remain in place.

![Paused DiffusionPDE EDM history: completed sampled TRAIN loss through epoch 1403, saved validation checks through epoch 1400, and last checkpoint at epoch 1400. Native denoising losses and reconstruction validation metrics remain separately labelled; a low denoising loss is not a reconstruction-quality certificate.](figures/diffusion_paused_loss.png)

**Figure 31.** Paused DiffusionPDE EDM history: completed sampled TRAIN loss through epoch 1403, saved validation checks through epoch 1400, and last checkpoint at epoch 1400. Native denoising losses and reconstruction validation metrics remain separately labelled; a low denoising loss is not a reconstruction-quality certificate. [PDF master](figures/diffusion_paused_loss.pdf).

[Pause record](metadata/training_pause.json), [loss history record](metadata/paused_loss_curve.json), [verification](metadata/verification.json), [model source audit](metadata/model_audit.json).

The retained model arrays are normalized full-grid endpoints, not checkpoint copies. Matching old archives use relative links; new models store one compressed endpoint archive. No completed per-frame inference cache remains. Small selected maps, conditional curves, joint occupation, event masks/thresholds and spectral band summaries support redraw. Complete candidate tables intentionally preserve both accepted and rejected possibilities. They are evidence catalogs, not discarded optimization intermediates.

For a new model, add its run to the registry, freeze its last checkpoint with `inference.py extend`, and extract only that model with the same cohort and protocol. Then recompute the model-pair census and selection, redraw figures, rebuild this report and verify the rendered result. Existing frozen identities must not be replaced. Changing an existing model's last checkpoint requires a new study/version identity. Detailed commands and the inference environment are in [README](README.md).

## Figure index

[Complete figure atlas: all PDF masters at their original page sizes](FIGURE_ATLAS.pdf). The atlas preserves vector text and axes and is convenient for inspecting dense panels or selecting future manuscript figures.

| Figure | Retained preview and PDF master |
| --- | --- |
| 1 | [controls_strict_toys](figures/controls_strict_toys.png) · [PDF](figures/controls_strict_toys.pdf) |
| 2 | [controls_real_state](figures/controls_real_state.png) · [PDF](figures/controls_real_state.pdf) |
| 3 | [controls_permutation_sweep](figures/controls_permutation_sweep.png) · [PDF](figures/controls_permutation_sweep.pdf) |
| 4 | [controls_strict_sweep](figures/controls_strict_sweep.png) · [PDF](figures/controls_strict_sweep.pdf) |
| 5 | [controls_real_families](figures/controls_real_families.png) · [PDF](figures/controls_real_families.pdf) |
| 6 | [events_06_highCO_hotT](figures/events_06_highCO_hotT.png) · [PDF](figures/events_06_highCO_hotT.pdf) |
| 7 | [events_07_lowCH4_highCO](figures/events_07_lowCH4_highCO.png) · [PDF](figures/events_07_lowCH4_highCO.pdf) |
| 8 | [events_01_lowCH4_highCO](figures/events_01_lowCH4_highCO.png) · [PDF](figures/events_01_lowCH4_highCO.pdf) |
| 9 | [events_02_transitionCH4_highCO](figures/events_02_transitionCH4_highCO.png) · [PDF](figures/events_02_transitionCH4_highCO.pdf) |
| 10 | [events_03_highCO_hotT](figures/events_03_highCO_hotT.png) · [PDF](figures/events_03_highCO_hotT.pdf) |
| 11 | [events_04_lowCH4_hotT](figures/events_04_lowCH4_hotT.png) · [PDF](figures/events_04_lowCH4_hotT.pdf) |
| 12 | [events_05_lowCH4_hotT](figures/events_05_lowCH4_hotT.png) · [PDF](figures/events_05_lowCH4_hotT.pdf) |
| 13 | [example_01](figures/example_01.png) · [PDF](figures/example_01.pdf) |
| 14 | [example_02](figures/example_02.png) · [PDF](figures/example_02.pdf) |
| 15 | [example_03](figures/example_03.png) · [PDF](figures/example_03.pdf) |
| 16 | [example_04](figures/example_04.png) · [PDF](figures/example_04.pdf) |
| 17 | [example_05](figures/example_05.png) · [PDF](figures/example_05.pdf) |
| 18 | [example_06](figures/example_06.png) · [PDF](figures/example_06.pdf) |
| 19 | [example_07](figures/example_07.png) · [PDF](figures/example_07.pdf) |
| 20 | [example_08](figures/example_08.png) · [PDF](figures/example_08.pdf) |
| 21 | [inversion_heatmaps](figures/inversion_heatmaps.png) · [PDF](figures/inversion_heatmaps.pdf) |
| 22 | [fidelity_strata](figures/fidelity_strata.png) · [PDF](figures/fidelity_strata.pdf) |
| 23 | [cohort_state](figures/cohort_state.png) · [PDF](figures/cohort_state.pdf) |
| 24 | [cohort_joint_tv](figures/cohort_joint_tv.png) · [PDF](figures/cohort_joint_tv.pdf) |
| 25 | [cohort_copula_tv](figures/cohort_copula_tv.png) · [PDF](figures/cohort_copula_tv.pdf) |
| 26 | [example_scale](figures/example_scale.png) · [PDF](figures/example_scale.pdf) |
| 27 | [cohort_scale](figures/cohort_scale.png) · [PDF](figures/cohort_scale.pdf) |
| 28 | [example_shape](figures/example_shape.png) · [PDF](figures/example_shape.pdf) |
| 29 | [cohort_shape_geometry_mean](figures/cohort_shape_geometry_mean.png) · [PDF](figures/cohort_shape_geometry_mean.pdf) |
| 30 | [cohort_shape_component_abs_mean](figures/cohort_shape_component_abs_mean.png) · [PDF](figures/cohort_shape_component_abs_mean.pdf) |
| 31 | [diffusion_paused_loss](figures/diffusion_paused_loss.png) · [PDF](figures/diffusion_paused_loss.pdf) |

## Reading and artifact map

[Original V1](../REPORT.md) · [V2](../REPORT_V2.md) · [V3 story](../V3_PRESENTATION_20MIN.md) · [V3 evidence](../V3_EVIDENCE_LEDGER.md) · [post-training addition](../Premise_Add/SPEAKER_NOTES.md) · [latest presentation review](../Premise_Add/Presentation_Review_20261029/SPEAKER_SCRIPT_AND_REVIEW.md) · [reproduction and extension](README.md).

This report is generated from saved evidence by `scripts/build_report.py`. Its source paragraphs and captions remain unwrapped. Figure files are independent Python-generated PDF masters with relative PNG previews.
