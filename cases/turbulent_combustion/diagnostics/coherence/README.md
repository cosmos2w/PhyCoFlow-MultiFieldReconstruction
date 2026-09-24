# Coherence diagnostics

This folder contains set-level coherence outputs, a pinned single-snapshot
topology example, and a matched A/B cross-spectrum score comparison. PNG
figures have matching PDF and SVG files for editing and publication layout.

## Topology example

The example is rerendered from the immutable epoch-800 preview saved under
`source/`; it does not read the live `latest_reconstruction.npz` and does not
run a model. Source run `20260924T030724Z_4208231c` is
`coherence_fix_ABC_sliced_persistence_formal_5000ep_gpu1`. The sample is
`trajectory_000000:8000` from the validation split at epoch 800, global step
30400, using the configured evaluation weights. SHA-256 digests for the saved
payload, preview metrics, resolved config, and normalization artifact are in
`source/provenance.json`.

The scored fields are CO and T in `model_units` (mean/std normalized). The
training topology contract uses 4,096 `fixed_shared` points selected with seed
100045 and rasterizes them onto the configured nonperiodic 32×128 grid. This
example reuses the 4,096 query coordinates saved by the preview (preview seed
2027), so it is an **auxiliary single-snapshot estimate**, not an exact replay
of the training query selector or a set-level benchmark. The raster is
configured for the topology estimator; it is not the dataset's native
100×403 grid. Antialias downsampling is configured but was not applied to the
sparse preview point cloud.

Render on CPU from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src conda run -n phycoflow_env python cases/turbulent_combustion/diagnostics/coherence/render_epoch800_topology.py
```

The script verifies the pinned payload checksum, limits Torch to one CPU
thread, and records the environment and runtime in
`topology/render_manifest.json`. The recorded render used Python 3.10.19,
Torch 2.5.1+cu121, and GUDHI 3.13.0; it ran on CPU with one Torch thread and
took about 5.5 seconds wall time including Conda startup (the exact renderer
time is recorded in the manifest). It writes:

- `topology/persistence_term_distributions.png` — per-snapshot configured
  sliced-Wasserstein persistence distances on their natural linear scale. The
  component-weighted total is before the outer topology family weight.
- `topology/betti_curves.png` — exact H0/H1 counts on the same signed,
  reference-standardized sublevel/superlevel scalar and mutual-line filtrations
  used for the persistence score. Counts are evaluated at the configured
  reference quantiles; the persistence distance uses full diagrams.
- `topology/configured_grid_topology.png` — an interpretation view of paired
  q50 level-set geometry and mask disagreement on the configured 32×128
  raster. It is not a persistence diagram or an H0/H1 score. Column headings
  appear once; each field row gives its reference q50 threshold. The title
  states the normalized model units.
- `topology/metrics.npz`, `topology/metrics.csv`, and `topology/report.json` —
  raw values, exact counts, query/raster provenance, and estimator details.

`--stat-scale` does not change topology plots: distances and counts use their
natural linear scale. Whole-split topology evaluation replays the repository's
`fixed_query_indices` selector on full-grid prediction/reference arrays and
records the query indices, coordinate digest, and whether the contract matches
the training configuration. For this run, that set-level estimator still
uses a 100×403 source array sampled to 4,096 points and mapped to the
configured 32×128 raster; it must not be described as native-grid topology.

## Matched A/B cross-spectrum example

`ab_balanced_train_last/cross_spectrum_source_post_coherence.png` is rerendered
from pinned source and post-training metric payloads under
`source/ab_balanced_train_last/`. It compares train-last results from source
run `20260826T212231Z_7b68c461` and post-training run
`20260829T235221Z_b3b586c4`. The comparison used the same 200 selected
snapshots (192 retained in twelve matched ensembles of 16), evaluation seed
2027, and 4,096 `fixed_shared` graph queries with seed 100045. Graph-Fourier
scores use the recorded 16-neighbor graph and 48 nonzero modes. The plot shows
only the configured `same_frequency` and `cross_frequency` terms; points are
mean bounded agreement scores with ±1 sample standard deviation across the
twelve ensembles, on one shared focused linear score axis. Exact checkpoint
paths, estimator settings, IDs, input hashes, and the pinned
`resolved_config.yaml` are recorded beside the metrics.

There is a provenance discrepancy in the historical evaluation artifacts:
the resolved post-training config enables `same_frequency` and
`cross_frequency`, has `band_energy` disabled, and contains no `self_spectrum`
entry. The old set-level evaluation report and NPZ nevertheless include
`self_spectrum` at weight 1.0 and a family aggregate that mixes that term with
the other scores. The plot excludes both `self_spectrum` and the mixed family
aggregate. Do not interpret those historical evaluation fields as the trained
B objective. This discrepancy and the configured/reported terms are explicit
in `source/ab_balanced_train_last/provenance.json`.

Rerender from the repository root with one CPU thread:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src conda run -n phycoflow_env python cases/turbulent_combustion/diagnostics/coherence/render_ab_train_last_cross_spectrum.py
```

This is a matched score-bar view, not a graph-band energy profile. The saved
A/B evaluation disabled `band_energy` and retained no per-snapshot graph-band
fractions or graph-Fourier coefficients, so a measured source/post band profile
requires a fresh evaluation. The `render_cross_spectrum_band_profiles` plotting
API has a synthetic contract test, but no A/B band profile is included here.

## Coherence figure set

- Keep the global-distribution marginal, pairwise, and joint/top-tail views;
  the opt-in joint-PDF grids remain a separate, more detailed view.
- Keep cross-spectrum score bars and add the per-field graph-band energy
  profile when that evaluation retains or recomputes matched band fractions.
- Keep the topology persistence-distance distributions, exact Betti curves on
  scored filtrations, and configured-raster q50 interpretation view.

For the historic A/B run, cross-spectrum score bars can be regenerated as a
matched pair from retained metrics. Standard global-distribution output lacks a
retained baseline metrics payload, so its existing source/post PNGs must not be
presented as a newly rerendered matched visual. The train joint-PDF extra
payload does retain both source and post bin metrics and can support those
specific panels.
