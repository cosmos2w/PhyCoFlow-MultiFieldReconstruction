# Versioned State, Scale and Shape queries

These definitions are opt-in experimental queries. Legacy global-distribution, graph cross-spectrum and persistence metrics retain their own identities. A package comparison on one case would not isolate each family's learning effect or establish governing-equation validity. The proposed light combustion profile did not qualify in the bounded 2026-11-06 TRAIN/SOURCE study; no formal comparison was launched and the case default remains unchanged.

## State: marginal and mixed occupation

`configs/coherence/state_mixed_mean.yaml` selects `global_distribution.definition: marginal_mixed_mean_v2`. It uses each field's empirical marginal squared W2 and the mean squared W2 over a fixed scrambled-Sobol-to-Gaussian unit bank: 32 mixed directions, seed 1234, no inserted axes, no hard top selection. Both endpoints use the same frozen TRAIN model normalization. Automatic pair training is disabled. Named legacy pair queries remain available independently. A one-field profile executes its marginal alone.

This describes spatial occupation under equal selected-node mass, separately per snapshot; it is invariant to a common spatial permutation. It does not localize arrangements. Independent-channel permutations and three-field parity can alter the mixed score without changing marginals or all pair occupations. Mixed projections can also push already-correct fields, so fieldwise fidelity remains an independent requirement. An independent evaluation bank can use 64 directions and seed 5051 with the same definition.

The historical component name `cross.joint_topk_swd` remains in saved schemas; `score_definition` and the full config distinguish the mean-bank query. Disabled components have unavailable values rather than fabricated zero losses. Pooled joint histograms are occupation diagnostics, not the average per-snapshot State loss. Nonuniform physical quadrature needs a measure-correct adapter; unsorted weights must never multiply sorted residuals.

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

`evaluation/canonical_coherence.py` accepts a frozen bundle of TRAIN-fitted families and saved `[J,B,N,C]` members. It never fits or restores arm-specific geometry. Use distinct legacy/next names, four recorded latent seeds, the same field order/normalizer, same conditioned samples and query panel, and fixed B32 groups for legacy spectral moments. State/Shape average paired member scores; conditional Scale uses all four genuine draws. Field MSE of a member and of an ensemble mean remain separate.

The comparison fingerprint binds actual endpoint path, solver, steps, guidance/final clamp, effective live/EMA weights, query/chunk settings, conditioning, latent seeds, geometry, scales and score definitions. Changed sampler, support or normalization rejects comparison. Family artifacts are checked before and after scoring. Saved NPZ/JSON scores preserve all leaves and model/checkpoint provenance.

Uncertainty grouping must be case-declared. A single correlated trajectory has descriptive paired time-block sensitivity; sample-ID prefixes, pixels and latent members cannot manufacture independent clusters or a 95% interval. Initialization, all draws, scoring, backward/update, persistence/transfers, checkpointing, wall time, CPU time and peak device memory need separate accounting.

Reconstruction NPZs may declare `physical_mask`, optional positive `physical_weights` and `physical_measure`. The writer accepts case `auxiliary.physical_weights`; absence explicitly means equal valid nodes. Renderers compute displayed-support fieldwise relative L2, or label an unavailable denominator. Any non-full 2D mask uses node rendering so isolated valid support is visible. Save PDF masters and PNG previews. Legacy set topology renderers explicitly reject new definitions they cannot faithfully represent; use canonical raw leaves and dedicated saved-data views for those definitions.

## Bounded admission and default decision

A light query/raster is admitted only when its primary TRAIN/SOURCE queries and worst supported leaf agree with the fixed native/common-resolvable anchor within the predeclared tolerance. Aggregate agreement cannot excuse a failed leaf. If neither light panel qualifies, close the launch instead of increasing density, tuning tolerances or reducing the training horizon.

The bounded study retained live coverage SOURCE weights, two Euler steps, endpoint-smooth guidance and final clamp, five fields and T-only 192–384 sensors. Both Q1024 and Q2048 failed the 10% gate. The intended 500-epoch legacy/enhanced comparison and conditional replication were therefore not launched. There is no measured learning gain, complete training runtime, or seed-robustness claim; qualified code remains experimental and the established SOURCE/default is retained.
