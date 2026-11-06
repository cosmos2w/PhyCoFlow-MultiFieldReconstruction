# Topology coherence: efficient sliced persistence

The current post-training method is `strategy: cubical_persistence` with
`persistence.distance: sliced_wasserstein`. The [shared defaults](../../../../../configs/defaults/topology_post_training.yaml)
and [active-emulsion case](../../../../../cases/active_emulsion/README.md)
provide the post-training recipe and launch commands.

## Mathematical definition

GUDHI computes exact lower-star cubical pairings on the configured raster,
including periodic boundaries. Only pairing indices are detached; critical
values are gathered from live generated tensors so the loss has piecewise
parameter gradients. All positive-persistence finite bars are retained, along
with essential births. Ties and changes in pairing remain nonsmooth.

Finite diagrams use a fixed-angle sliced Wasserstein-1 approximation with
cross-diagonal augmentation. This is not an exact Wasserstein assignment.
Essential births are compared separately. Reference-derived scaling and fixed
normalization keep diagrams comparable without using the generated sample to
set its own reference.

Mutual persistence restricts joint filtrations to fixed positive lines
`b + t a`, using `max_i((f_i - b_i) / a_i)` for each signed sublevel field.
Finite slices summarize a multi-parameter module; they do not fully characterize
it. The active-emulsion recipe uses both filtration directions, H0/H1, self
phi/vx/vy and phi × signed vorticity as its only mutual group.

### Finite-primary pair definition (version 4)

`persistence.definition: finite_primary_v1` selects a separate paired
definition. The default remains `legacy_v3`; old scores retain their existing
values. Version 4 uses both unary signs and, for each named pair, the signs
`(+,+)`, `(-,-)`, `(+,-)`, `(-,+)`. Its three fixed lines are
`v=(1,1)/sqrt(2), beta=(0,0)`, `v=(2,1)/sqrt(5), beta=(0.5,-0.5)`, and
`v=(1,2)/sqrt(5), beta=(-0.5,0.5)`. A pair filtration is
`max_a((sign_a * f_a - beta_a) / v_a)` and line weights are `min(v)`
normalized to sum to one. A two-field, one-pair bank has 16 scalar
filtrations, each with H0 and H1. It uses all of them on every evaluation;
the legacy filtration sampler does not support this version.

Each descriptor is standardized using the paired reference raster mean and
population standard deviation, clamped by the declared positive
`persistence.scale_floor`. This is per snapshot, shared by prediction and
reference, and is distinct from the frozen score calibration. Raw finite
distance F is divided by the number of valid raster vertices; raw
essential-birth distance E is not. For every field/pair, sign, line and
homology leaf, calibration takes the mean raw F and E over the declared paired
TRAIN/SOURCE states, independently clamping them to positive
`calibration.finite_floor` and `calibration.essential_floor`. It freezes the
result before optimization. A leaf contributes `0.9*F/scale_F +
0.1*E/scale_E`. Structurally absent essential classes are omitted, with no
redistribution of their 0.1 weight. A zero SOURCE discrepancy is instead
retained with its positive floor. Self terms average over field, sign and
homology; mutual terms average over pair, sign and homology and sum the
normalized line-weighted leaves. Component weights are then applied.

`fit_calibration(prediction, target, coordinates=..., provenance=...)` takes
aligned `[B,N,C]` TRAIN/SOURCE tensors and fixes the scales once. Provenance
must declare `split: train`, one `sample_ids` entry per row, and the SOURCE
checkpoint SHA-256 as `source_checkpoint_sha256`. The family artifact records
the calibration, line bank, geometry and source binding; a score before fit,
an attempted refit, or an inconsistent restored artifact raises an error.
`TermResult`s retain raw finite and essential leaves, their frozen scales and
structural-presence flags. They are diagnostic leaves, not extra scalar terms.

The optional `geometry.native_masked: true` is a fixed, nonperiodic native
vertex-cell complex. It requires zero smoothing, no downsampling or derived
descriptors and a compact provider whose returned nodes are all valid. A full
grid containing invalid land nodes is rejected. With
`geometry.structured_indices: <auxiliary-name>`, train
geometry fitting consumes integer row/column indices for a curvilinear grid;
otherwise the coordinates must form the declared regular lattice. Missing
vertices and all incident cells are excluded from the complex, rather than
filled by interpolation. The same mask applies to every snapshot and its
exact essential counts are frozen at calibration and checked during scoring.
Periodic masks, changing masks, arbitrary
3-D domains and physical-area quadrature are not implemented. A full-rectangle
IDW raster cannot represent holes; use the supported native mask for such a
domain.

## Execution and integration

`persistence.py` batches CPU transfer and pairing work, uploads packed indices
and batches sliced distances. The bounded in-memory reference cache is keyed by exact content.
`persistence_objective.py` owns descriptors, reference preparation and aggregation.
`family.py` supplies raster mapping, unit conversion and the common
`FamilyResult`/`TermResult` contract.

The canonical training recipe uses `optimization.gradient_balance:
topology_regularized`. One Adam update combines calibrated batch-average topology
with source-relative endpoint and anchor penalties. Shared rollout context is
owned by the physical CQ adapter and remains differentiable. Strict held-out
checkpoint selection is separate from stochastic training updates.

Scientific-source hashes bind family artifacts to their implementation.
Saved runs must use their recorded source for exact recovery; changing the
active recipe does not reinterpret historical metrics or checkpoints.

## Historical compatibility

The old `betti_curves`, `spatial_self_mutual`, spatial Wasserstein matching and
component-constrained optimizer paths remain explicitly addressable for saved
configurations, provenance and comparative evaluation. Configurations without
a strategy retain the historical Betti-curve interpretation; every current
recipe names sliced persistence explicitly. This preserves old scientific
meaning while removing the old active-emulsion launch recipes from the current
workflow.
