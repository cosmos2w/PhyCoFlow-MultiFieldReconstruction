# Topology coherence: efficient sliced persistence

The current post-training method is `strategy: cubical_persistence` with
`persistence.distance: sliced_wasserstein`. The active recipe and launcher are
in [TOPOLOGY_POSTTRAINING.md](../../../../../TOPOLOGY_POSTTRAINING.md).

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
