# Cross-Spectrum Coherence

This package compares fields on one fixed point set shared by every state in the generated and reference ensembles. Fields enter the graph transform in model units or through the configured fixed affine conversion to physical units. The family does not apply a copula transform.

## Legacy v3 definition

Omitting `cross_spectrum.definition` preserves the existing v3 objectives and result paths:

- `cross_spectrum.self_spectrum.auto_spectrum`: optional modewise per-field auto-spectrum MSE;
- `cross_spectrum.same_frequency.magnitude_squared`: MSE between field-pair magnitude-squared coherence at each individual graph mode;
- `cross_spectrum.cross_frequency.band_energy_coupling`: MSE between normalized covariance entries of squared band energies;
- `cross_spectrum.band_energy.log_power`: optional log mean-band-power error.

The v3 cross-frequency statistic is based on squared linear coefficients and is fourth-order in the underlying fields. Its values are not comparable to the v4 cross-frequency objective below. `self_spectrum` remains opt-in and the v4 definition rejects both `self_spectrum` and `band_energy` as optimized terms.

## Opt-in second-order v4 definition

Select `cross_spectrum.definition: second_order_blocks_v4`. The two public component keys remain `same_frequency` and `cross_frequency`; their result paths end in `second_order_covariance_blocks` and mean same-band and cross-band second-order cross-field covariance, respectively.

For graph coefficient (a^X_{bki}), where (b) indexes the ensemble, (k) the retained graph mode, and (i) the field, v4 centers over the ensemble and computes

\[
\Sigma^X_{ij}[k,q] = \frac{1}{B-1}\sum_b
  (a^X_{bki}-\bar a^X_{ki})\,\overline{(a^X_{bqj}-\bar a^X_{qj})}.
\]

For band mode sets (I_l,I_m), the block is `Sigma[i,j][I_l,I_m]`. A diagonal-band block (`l == m`) is the `same_frequency` term; all ordered off-diagonal band blocks (`l != m`) form `cross_frequency`. Each signed or complex block is normalized by the corresponding generated/reference band energies, then compared with the squared Frobenius norm of the **block difference**. The implementation sums within each block and averages over eligible blocks. It retains O(C²K²) coefficient-space covariance and never forms a dense spatial N-by-N covariance or projector.

`graph.band_intervals` is required for v4. It is an ordered list of `{name, lower, upper}` intervals in normalized-Laplacian eigenvalue units. Intervals are contiguous and half-open except the final upper endpoint, and each must contain at least one retained mode. The basis builder rejects a boundary that splits adjacent eigenvalues whose gap is at most `graph.degeneracy_tolerance`. The realized mode IDs and interval definitions are recorded in diagnostics and the family artifact. Choose cutoffs for the actual point geometry and retained spectrum; the legacy equal-count `graph.bands` partition is not an implicit v4 default.

V4 uses unbiased sample covariance with denominator `B-1`; `minimum_ensemble_size` defaults to 32. Before training, call `freeze_reference_calibration(reference, coordinates)` once on a fixed reference panel. It stores each reference band energy and checksum. Repeating an identical calibration is safe; changing it requires `replace=True`. A training forward call with `context.phase == "train"` requires frozen calibration. Direct utility calls without a frozen panel use current-batch reference energies and label that fallback in diagnostics.

For block `(l,m,i,j)`, the calibration defines the eligibility mask from each reference channel's band-energy fraction of its total retained-band energy. The common additive covariance floor is `absolute_floor + relative_floor * sqrt(Ecal[l,i] * Ecal[m,j])`. The generated denominator clamps each generated band energy below by `max(relative_floor * Ecal, finfo.tiny)` before taking the square root; the reference denominator uses its current target energy with the same additive floor. This keeps masks and floors independent of generated amplitude and gives exact generated collapse a finite derivative. Defaults are `relative_floor=1e-6`, `absolute_floor=1e-12`, and `minimum_reference_band_fraction=1e-8`. The absolute floor uses the product units of the configured model or physical field units; calibrate it for the selected unit system.

`LinearCoefficientCovarianceAccumulator` stores pooled coefficient sums and cross-products. Merging its sufficient statistics matches one covariance over the combined panel; averaging separately normalized minibatch scores is a different estimator. Auto-covariance band traces are internal scales and masks only, never an optimized v4 self-spectrum term. Under ideal joint graph stationarity, cross-mode covariance is expected to vanish outside repeated-eigenvalue subspaces; observed cross-band covariance describes departures from that second-order assumption, not energy transfer or nonlinear triadic coupling. Nonlinear energy dependence such as `z` versus `z²-1` can remain even when the v4 linear covariance is zero.

Routine training diagnostics contain eligible/skipped block counts, mask fractions, calibration source, floors, and per-block scalar discrepancies. Passing `context.phase` as `evaluation`, `calibration`, or `audit` also exports the small signed/complex normalized blocks as JSON-safe real/imaginary arrays. The v4 family artifact binds the definition, exact config, graph-basis hash, realized bands, and optional frozen calibration checksum; a v3 artifact cannot be loaded as v4.

Set evaluation restores the hashed `artifacts/cross_spectrum_family.pt` from the training run (or the child run selected for a paired source comparison); it never fits a basis or mask on the evaluated split. Reports include two separate estimators: `pooled` uses all selected snapshots in one centered covariance, while `training_aligned` preserves deterministic fixed-size ensemble membership and reports its dropped remainder. The pooled report uses mergeable coefficient sufficient statistics. Versioned `covariance_metrics_v4.csv` stores per-block losses and `covariance_blocks_v4.json` stores the signed normalized blocks with sample membership. V4 plots use squared Frobenius loss labels and do not publish the legacy bounded coherence score.

All geometry-dependent runs must use `query_policy: fixed_shared`. Reordering or changing coordinates after the basis has been built is rejected. The linear coefficient transform is equivariant to a common point permutation when coordinates are permuted with fields, and block Frobenius scores are invariant to a common orthogonal/unitary rotation within each band.
