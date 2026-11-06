# Versioned State, Scale and Shape queries

The current complete combustion profile selects weighted State, signed multiscale Scale and finite-primary Shape. Legacy equal-node State, graph cross-spectrum and persistence metrics retain their own identities for replay. A package comparison on one case would not isolate each family's learning effect or establish governing-equation validity. The earlier sparse-versus-native study measured query sensitivity; its all-leaf admission gate is superseded by the current bounded evaluation protocol. No new learned benefit is claimed here.

## State: marginal and mixed occupation

`configs/coherence/state_mixed_weighted.yaml` selects `global_distribution.definition: marginal_mixed_weighted_v3`. Each endpoint is an occupation law with explicit, nonnegative physical nodal masses. For every field and each of 32 fixed scrambled-Sobol-to-Gaussian mixed unit directions, sort values and their own endpoint masses together; merge the two cumulative-mass breakpoints and integrate the squared quantile gap. Both endpoints use the same frozen TRAIN model normalization; prediction and reference masses are supplied independently. A one-field profile executes its marginal alone. Zero masses and masks are supported; stable sorting defines a deterministic branch at ties, with value gradients away from changes in ordering. TRAIN-bank references are unsupported until they provide their own masses.

For the complete nonperiodic combustion grid, native masses are products of trapezoidal nodal widths derived from raw rectilinear coordinates; the outermost nodes bound the domain. The declared raw coordinate and field units are unknown. A fixed uniform-without-replacement Q4096 panel uses its selected native masses in a self-normalized inclusion estimator. It approximates the full-native physical occupation; normalization of retained masses does not make the sparse panel exact. The native weights, support, provider and fingerprint are frozen in the family artifact.

`configs/coherence/state_mixed_mean.yaml` retains `marginal_mixed_mean_v2`, which asks the same marginal and mixed-bank questions with equal selected-node mass. Uniform v3 masses reproduce v2. These queries describe occupation separately per snapshot and do not localize arrangements. Independent-channel permutations and three-field parity can alter the mixed score without changing marginals or all pair occupations; fieldwise fidelity remains an independent requirement. Named raw-value and own-rank pair occupations are available independently of pair training. The historical component name `cross.joint_topk_swd` remains in saved schemas; `score_definition` distinguishes the mean-bank query. Pooled joint histograms are occupation diagnostics, not the average per-snapshot State loss.

## Scale: signed conditional fields, including means

The new `multiscale_spectrum` family is independent of legacy `cross_spectrum`. Its FEM basis is orthonormal in the lumped physical mass M. For each connected piece it removes the mass-weighted field mean and applies frozen smooth spectral windows to the remaining coefficients. The geometry artifact records coordinates/metric, support, pieces, mode cutoff, windows and TRAIN calibration. Graph fallback has graph scales and uniform initial node mass, not physical wavelengths.

For IID conditional draws, the fair energy score is

```text
ES = mean_j d(X_j, target)
     - sum_{j<k} d(X_j, X_k) / [J*(J-1)] .
```

The candidate uses two differentiable target-free draws, alpha 1, axes and signed pair projections, a positive full joint score weight of 0.25, and no coherence-CRPS auxiliary. State and Shape reuse the primary draw; an independently seeded second reconstruction cannot advance data, sensor or primary latent streams. A deterministic source must not pretend to have two independent draws. The historical almost-fair default and mean-free `signed_spectral_v3` artifact semantics remain available.

`signed_spectral_mean_augmented_v4` adds scaled piece means to the same joint energy-score embedding:

```text
d_aug^2 = d_spec^2
          + mean_weight * sum_piece,field(piece_mass_fraction * delta_mean^2 / mean_scale^2).
```

The initial mean weight is 1. Mean scales are the square root of TRAIN population variance plus a declared positive field-unit tolerance squared. The same task-unit tolerance supplies positive band floors for eligible weak or constant TRAIN fields. Calibration never excludes a field or changes a denominator based on generated power. Physical-offset conversion and spectral arithmetic use the configured precision; the supplied new fragment explicitly uses float64. The fragment's illustrative tolerance must be frozen for the case before comparison; the bounded combustion qualification used 0.01 model units in every field.

Signed coefficient-space and synthesized-field joint distances have matching values and derivatives. The positive joint score covers dependence of means and retained fluctuations within this finite embedding. Removed support and unresolved modes remain outside it. Properness of this Scale score does not make a sum with paired State/Shape penalties proper, and ConFIG need not optimize any scalar score. Report spread, skill, signed co-spectra, absolute and unresolved power, and piece means separately.

New v4 artifacts bind scientific implementation/backend identity as well as geometry, normalization, scales and config. Exact recovery must use the recorded implementation. Legacy v3 score values, state and default behavior remain unchanged.

## Shape: finite and essential information with a fixed sign/line bank

`topology.strategy: cubical_persistence` with `persistence.definition: finite_primary_v1` selects version 4. Its exact bank and calibration API are documented in the [topology family](../src/phycoflow_reconstruction/coherence/families/topology/README.md). Each pair uses four sign combinations and these three lines:

| Direction | Offset |
|---|---|
| `(1,1)/sqrt(2)` | `(0,0)` |
| `(2,1)/sqrt(5)` | `(0.5,-0.5)` |
| `(1,2)/sqrt(5)` | `(-0.5,0.5)` |

A pair filtration is `max_a((sign_a*f_a-offset_a)/direction_a)`. Line weights are normalized `min(direction)`. CO/T has four unary and twelve joint filtrations, each containing H0/H1. Full direct execution is required; legacy/P3 filtration enumeration cannot score this bank.

Prediction and reference share the reference raster's per-field mean/std and a declared descriptor-unit floor. Raw finite F and essential-birth E leaves remain separate by field/pair, sign, line and homology. Finite bars divide by raster vertex count; essential births do not. The new loss uses `0.9*F/frozen_scale_F + 0.1*E/frozen_scale_E` with component and line weights, calibrated once on paired TRAIN/SOURCE data and positive task floors. Structurally absent classes are omitted; a zero SOURCE discrepancy is retained with a floor. Integer essential counts are frozen and must match on every score and restore, including disconnected domains. Diagram symmetry is not positional fidelity.

The generic trainer calibrates new Shape leaves on 32 temporally spaced TRAIN SOURCE endpoints, preserves RNG/model mode, then serializes the frozen artifact before optimization. Resume checks the exact frozen new-family buffers and SOURCE provenance; missing or changed checkpoint state raises an error. Legacy topology artifacts retain their existing strict source/backend binding. Historical artifacts therefore require their recorded code for exact resume/rescore; base/child model weights still load through the strict model-state interface.

A fixed native nonperiodic mask excludes missing vertices and their incident cells. It does not support mixed periodic masks, changing masks or arbitrary 3D complexes. IDW over a rectangle does not preserve holes. Raw finite `/vertex` scores at different raster resolutions are discrete quantities; they are not automatically a common continuum error. Physical quadrature and equal raster-vertex measures remain distinct.

## Matched evaluation and saved figures

`evaluation/canonical_coherence.py` accepts TRAIN-fitted families and saved `[J,B,N,C]` members. The `evaluate-native-coherence` CLI additionally streams the full ordered native support, saves matched member and ensemble-mean field-risk evidence, and can rescore the members with one TRAIN-fitted native metric bundle. Use four recorded latent seeds, the same field order/normalizer, conditioned samples, endpoints, support and bundle for SOURCE and child. State/Shape use paired members; conditional Scale uses genuine conditional draws. Full-native support does not remove finite spectral mode truncation. The [strategy guide](Posttraining_Strategies.md#current-state-scale-and-shape-profiles) shows the CLI contract.

The comparison fingerprint binds actual endpoint path, solver, steps, guidance/final clamp, effective live/EMA weights, query/chunk settings, conditioning, latent seeds, geometry, scales and score definitions. Changed sampler, support or normalization rejects comparison. Family artifacts are checked before and after scoring. Saved NPZ/JSON scores preserve all leaves and model/checkpoint provenance.

Uncertainty grouping must be case-declared. A single correlated trajectory has descriptive paired time-block sensitivity; sample-ID prefixes, pixels and latent members cannot manufacture independent clusters or a 95% interval. Initialization, all draws, scoring, backward/update, persistence/transfers, checkpointing, wall time, CPU time and peak device memory need separate accounting.

Reconstruction NPZs may declare `physical_mask`, optional positive `physical_weights` and `physical_measure`. The writer accepts case `auxiliary.physical_weights`; absence explicitly means equal valid nodes. Renderers compute displayed-support fieldwise relative L2, or label an unavailable denominator. Any non-full 2D mask uses node rendering so isolated valid support is visible. Save PDF masters and PNG previews. Legacy set topology renderers explicitly reject new definitions they cannot faithfully represent; use canonical raw leaves and dedicated saved-data views for those definitions.

## Bounded evaluation and decision

The earlier Q1024 and Q2048 experiments measured sparse-query sensitivity and did not test learning. Their all-leaf 10% gate no longer decides whether the current profile can train. Compare native and sparse leaves on fixed TRAIN anchors, retain the observed discrepancies, and evaluate learning against the declared native validation yardstick. The current selection rule requires the 10% per-field native physical-risk allowance and a reproducible coherence improvement on the selection panel, followed by the disjoint validation audit. Report each family, field, selected and final checkpoint, complete runtime and memory separately. A sparse-work count or a lower training loss alone is not evidence of reconstruction improvement.
