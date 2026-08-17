# Cross-Spectrum Coherence

This package implements graph cross-spectrum coherence for one fixed point set
shared by every state in the generated/reference ensembles.

`basis.py` builds a Gaussian k-nearest-neighbor graph, the symmetric normalized
Laplacian, a deterministic low-frequency eigensystem, equal-mode bands, and a
coordinate fingerprint. `statistics.py` provides the differentiable graph
Fourier and ensemble statistics. `family.py` exposes:

- `cross_spectrum.same_frequency.magnitude_squared`: MSE between field-pair
  magnitude-squared coherence spectra;
- `cross_spectrum.cross_frequency.band_energy_coupling`: MSE between
  off-diagonal normalized band-energy covariance cells;
- `cross_spectrum.band_energy.log_power`: optional log mean-band-power error.

The first two preserve the co-worker objective. They are ensemble statistics,
so `per_sample_cost` is intentionally absent. Same-frequency mode requires at
least two states; cross-frequency mode requires at least three and is still
statistically noisy at that minimum. Formal work should normally use a larger
ensemble.

Geometry-dependent runs must use `query_policy: fixed_shared`. Changing the
coordinate set/order after basis construction is an error. The family artifact
stores the graph tensors, bands, field mapping, scientific config, coordinate
hash, resolved sigma, version, and source revision.
