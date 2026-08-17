# Topology Coherence

This package implements v1 differentiable topology coherence for fixed 2-D
rasterizable point sets.

`geometry.py` precomputes a linear inverse-distance point-to-grid map; periodic
grids use tiled coordinate images so nearest-neighbor interpolation does not
break at the seam. `betti_curves.py` computes exact hard-forward H0 persistence
counts, cubical Euler curves, periodic H2, and H1 via
`b1 = b0 - chi + b2`. Straight-through sigmoid surrogates carry gradients while
the reported forward counts remain integer-valued.

The registered components are:

- `topology.self.betti_curves`: selected-field H0/H1 curve matching over
  reference-quantile super- and/or sub-level filtrations;
- `topology.mutual.fibered_betti_curves`: H0/H1 curve matching after restricting
  a two-field filtration to configured positive-slope lines.

Field selection, raster shape, coordinate axes, smoothing, periodicity,
quantiles, dimensions, directions, field pairs, and fibered-line count are
explicit configuration. The point set/order must remain fixed and is checked by
SHA-256. H0 pairings are computed on detached orderings and reused across H0/H1;
the gradients are biased straight-through surrogates, as declared in the family
artifact.

Case PDE terms, RCC evaluation profiles, retired legacy modes, and unavailable
spatial-lifted matching are not part of the v1 topology training family.
