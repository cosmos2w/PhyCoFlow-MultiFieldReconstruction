# Explicit endpoint, fidelity and spatial contracts

Round 8 specifies the output contract for the existing SSS-only repair route. Historical runs retain their original risk definitions, allowances and verdicts. The execution scope is in the [archived Round 8 workplan](../UpdatePlans/_Plans/Coherence_Adaptive_Repair/_bk/Round_8/WORKPLAN.md); completed measurements and decisions are in the [Round 8 report](../UpdatePlans/_Reports/Coherence_Adaptive_Repair/Round_8/report.md).

## Endpoint and risk

A deterministic endpoint supplies one genuine prediction. A conditional generator supplies its declared independent members. For each condition, average physical squared errors over members and valid spatial mass, then pool conditions with the declared condition weights. A shorter final batch receives its actual condition count. Do not substitute ensemble-mean MSE, an average of conditionwise risk ratios, or duplicated deterministic predictions.

For field `f` on split/protocol `D`, member fidelity means `M_child <= (1 + epsilon_f) M_SOURCE + delta_f`. Round 8 uses `epsilon=0.10` and `delta=0` unless a physical absolute tolerance is explicitly declared. TRAIN calibrates the controller; each assessment split uses its own matched SOURCE denominator under the declared law. The positive denominator floor conditions feedback and never enlarges the accepted bound. A 10% MSE allowance is approximately a 4.88% RMSE allowance.

The configured physical metrics are `physical_member_mse_model_units` and the historical `physical_mean_risk_model_units`. The generic risk arithmetic distinguishes `member_mse` from `mean_risk`. J1 supports member MSE directly; the signed population-mean correction requires at least two independent members. Preserve finite-ensemble-mean error and member variance as separate diagnostics. Member MSE penalizes variance; it is an average reconstruction promise, not posterior calibration or protection of every draw.

Affine field decoding multiplies MSE and dimensional absolute allowances by the field scale squared. SOURCE-relative ratios cancel that scale only when support, weighting and endpoint laws match. If authoritative unit metadata is unavailable, report dataset-unit squared. Field names do not establish units or observed-field roles.

## Five independent laws

| Contract | What it determines |
| --- | --- |
| SSS geometry support | Fixed coordinates, spectral basis, raster map, mask, boundaries and calibrated transforms |
| Field-risk quadrature | Native total mass and the spatial sampling/inclusion estimator |
| Descriptor queries | Selected fields, pairs, signs and filtration lines, with their target and inclusion weights |
| Observation law | Observed fields, sites, values, counts and consistency operation |
| Member law | Deterministic or independent conditional realizations, solver and endpoint weight view |

Round 8 initially retains Q4096 SSS geometry. Rotating field feedback includes selected sites exactly and a uniform complement with inverse inclusion probability, divided by native total mass. A wholly fresh risk subset or native-grid gather uses its own explicit sampling law. The integral applies to the declared endpoint law: a query-set-dependent model can change its prediction when queried on another support. These linear-risk estimators do not make sampled PH, quantile or feature-norm scores unbiased native observables. Slicing a native prediction also need not reproduce a model separately queried on those sites; label endpoint/support differences explicitly.

## Scores and positions

For a fixed represented distance, report attraction `A=mean_j d(X_j,Z)`, positive separation `R=sum_(j<l) d(X_j,X_l)/[J(J-1)]`, and fair score `S=A-R`. R is half the mean unordered-pair distance. Fairness removes finite-ensemble bias for the declared score; propriety depends on the distance and representation. Genuine deterministic J1 has separation zero and its paired distance. A stochastic J1 draw does not estimate a complete fair conditional score. State marginal/mixed gradient probes are telemetry, not direct distances.

The positional Shape candidate reuses selected PH filtrations. Fixed TRAIN quantiles and temperatures create soft sublevel excursions, integrated in eight overlapping geometry-only windows using valid raster mass. Its finite Euclidean feature supplies a paired/fair location component; the combined distance is half the sum of unchanged normalized PH distance and TRAIN-unit-matched location distance. The optional payload freezes thresholds, temperatures, windows, mass/mask, category/query weighting and positive location scale. Disabled position retains the historical PH observer. Keep PH and location contributions visible and use common observers for compared endpoints.

The location feature observes coarse raster position. It supplies neither native topology nor injective spatial registration. Scientific event and regional tasks must use definitions excluded from optimized features. Signed Scale alignment is a within-snapshot spectral inner product; it is not posterior correlation, causality or a conservation law.

## Optimizer, recovery and evidence

The maintained method computes three fresh parameter-space State/Scale/Shape rows, applies ConFIG, adds one multiplier-weighted field row, and uses the declared clipping, AdamW and projected PI feedback. It does not replay model-native supervision. Its direction is generally not the gradient of a scalar sum of family losses; no total objective should be invented for its curves. PI consumes normalized violations under the chosen budget and is expectation feedback rather than a hard feasibility projection.

The qualified factored static-feature executor and query-gradient boundary replay remain the production schedule. Changes to fidelity or Shape must preserve fresh family parameter rows and the complete parameter action. Executor equivalence and acceleration are separate claims: compare affected gradients/actions and recovery for correctness, and use complete epochs on the same device and objective for a new speedup claim. Report allocated and reserved memory with their measurement scopes; existing executor gains do not imply unchanged memory or a measured speedup for every new objective.

A changed risk or positional observation starts a named protocol or phase. Ordinary resume reloads saved scientific recording definitions, ordered columns with types/units/meanings, loss accumulation order, controller, optimizer and random state; formatting, permitted GPU changes and authorized horizon extensions do not migrate histories. The affected path must pass a real fresh-process save, saved-config reload, resume and append comparison. Curves refresh from complete recorded histories at first availability, every ten completed epochs, and review/pause/completion, independently of inference assessment.

For coherence post-training, `runtime.strict_algorithms: true` explicitly requests Torch deterministic algorithms and prepares a valid cuBLAS workspace before CUDA initialization. It requires deterministic runtime and is saved with the numerical configuration. Common-prefit runs retain their existing strict mode; other existing configurations retain their previous defaults. A new numerical choice uses a named attempt and an affected-path recovery check, without rewriting historical configurations or histories.

Use matched approximately-10% conditions on the required splits, preserve TEST lock, generate each admitted prediction bank once and reuse it. Repeated VALIDATION development is not independent confirmation. Frame uncertainty from the existing single-trajectory panels does not establish unseen-trajectory or training-seed transfer. Separate engineering qualification, observed relationship changes and model-replacement decisions.
