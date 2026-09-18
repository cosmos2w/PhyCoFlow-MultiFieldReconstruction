# Topology Persistence Coherence

This package is a port of the historical `persistence_satloss_cubical` (self) and
`fibered_satloss_cubical` (mutual) topology terms as a coherence family. It is registered as
`topology` and owns the fixed point-set-to-raster map and Gaussian smoothing
(`geometry.py`), the device merge-tree reductions (`merge_tree.py`) and the host thread pool and
backend switch (`execution.py`). Only the cubical-complex losses are ported; the Alpha-complex
variants, the fixed evenly spaced line fan and the periodic complex are deliberately absent.

**This is the topology term.** It compares full persistence diagrams through an optimal partial
matching, so its gradient transports a bar toward its partner, where a level-set count comparison
can only shrink or grow a count; that is why it descended in the synthetic recovery study where the
Betti curves did not. Its config contract is `schema.py`, read by `config/validate.py` through the
registry. The Betti-curve family it superseded survives as an optional detached package under
`../topology_betti/`, which imports the shared pieces from here and registers as `topology_betti`
only when it is present. The family remains experimental and is not exposed through
`visualize-run --eval-coherence`.

`persistence.py` holds the primitives: cubical persistence with vertex generators (GUDHI on the
host, or the merge trees of `merge_tree.py` on an accelerator, agreeing bar for bar), a
persistence floor applied at the reduction, padded batched diagram assembly, the device-built cost
matrices of the spatially-aware matching, SciPy's exact Hungarian solve per unit on the shared
thread pool, the batched differentiable re-costing, reference min-max normalization, and seeded
draws of monotone slice lines. GUDHI is required; construction fails with a clear import error
without the `topology` extra.

The registered components are:

- `topology.self.persistence_matching`: per field, sublevel and/or superlevel
  lower-star diagrams (essential bars capped at the larger field maximum), order-`p`
  Wasserstein distance with L1 ground metric and bar-to-bar costs modified by the L1 separation
  of the two bars' birth vertices over the bounding-box diagonal, in `multiplicative`
  (`1 + λ s`) or `additive` (`+ λ s`) mode. `lambda_spatial: 0` recovers plain matching exactly.
- `topology.mutual.fibered_matching`: per configured field pair, the two-parameter
  lower-star bifiltration restricted to `lines` monotone lines through the lower-left corner of
  the reference-normalized value box, with angles drawn uniformly from
  `(angle_margin, 1 - angle_margin) * pi/2`; each line's barcode is matched as above and the
  per-line distances are averaged. Lines are re-drawn on every call, per sample, direction and
  pair, from a NumPy generator seeded by `mutual.seed` (default 0), so the term is a stochastic
  estimate of the sliced matching distance. The generator state is stored in the family artifact
  and restored on reload, so a resumed family continues the same stream.

Both fields are min-max normalized by the *reference* field's statistics per sample, direction
and field. Finite bars shorter than `matching.min_persistence` (default 0.01, a fraction of that
range) are dropped at the reduction on both sides: a smoothed field's diagram is mostly such bars,
they only ever match the diagonal, and on real rasters a floor of 0.01 keeps about a third of the
bars and over 90% of the total persistence while shrinking the cubic assignment several-fold. Zero
keeps every bar of positive persistence. The per-sample cost of each component is the plain mean
over directions, fields or pairs, and homology degrees. Field selection, raster shape, coordinate
axes, smoothing, degrees, directions, matching order, spatial strength and mode, persistence floor,
field pairs, line count, line sampling, angle margin and line seed are explicit configuration. The
point set/order must remain fixed and is checked by SHA-256.

The matching and the generators are computed on detached values; the reported distance is exact
and its gradient is the subgradient of the piecewise-constant matching, as in the source modes.
One convention is added to the source modes: at an exactly zero distance, or an exactly zero
term under a fractional power, the gradient is the zero subgradient rather than the NaN that
autograd produces from a root's infinite slope, so `order > 1` is safe on identical fields.

Execution runs in three phases over every unit of a call at once (per sample, direction and field
for the self term; per sample, direction, pair and slice line for the mutual term). Phase 1 builds
the compared fields as one live block per side: one normalization, and for the mutual term one
draw of all slice lines from the seeded stream (`mutual.line_sampling`: `stratified`, the default,
draws one angle per equal bin and needs fewer lines for the same estimate; `uniform` draws them
independently as the historical modes did) and one broadcast line reduction. Phase 2 reduces both
blocks with the backend `PHYCOFLOW_TOPOLOGY_PAIRING` selects (`auto` runs the merge trees on the
device wherever the fields live on an accelerator; the host backend feeds GUDHI vertex ranks so
both agree bar for bar, and a batch the device declines falls back to it; the choice is reported as
`reduction_backend`), leaving cached reference rows out of the block, builds every unit's reduced
cost matrices as padded batched tensors on the device (`min(c_ij - p_i - p_j, 0)`, the exact
rectangular form of the partial matching with the diagonal), copies each size-sorted chunk to the
host once, and solves each unit exactly with SciPy on the thread pool of `execution.py`
(`PHYCOFLOW_TOPOLOGY_WORKERS`). Phase 3 contracts the solved plans against the live
diagrams of the whole call in one set of gathers and masked sums, whose backward is equally small.
Reference generators for the self term are cached against a digest of the normalized reference
row; the `reference_cache` diagnostics report hits, misses and occupancy.
