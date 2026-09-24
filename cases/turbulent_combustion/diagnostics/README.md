# Turbulent-combustion figure gallery

This directory holds versioned, reviewable examples and the figure decisions for
the turbulent-combustion case. Runtime figures under `runs/` are generated
artifacts; the source run, checkpoint, split, sample selection, and plotting
command belong with each example here. The live A+B+C formal run is a source of
example data, not a publication result while training is in progress. The
running Python process retains the plotting code it imported at launch; these
revised renderers apply to new processes and to the offline examples here. The
examples do not restart or write into that run.

## Figure roles

| Suite | Figure | Question it answers | Interpretation boundary |
| --- | --- | --- | --- |
| Training monitor | Objective history | Is optimization numerically stable and is fixed-panel validation improving? | Batch objectives are noisy and are not reconstruction accuracy. |
| Training monitor | Coherence family and component history | Which weighted A, B, and C objectives are being optimized, and how do their raw components evolve? | Objective height alone does not measure gradient influence. |
| Training monitor | Fixed-sample reconstruction preview | Is there a visible local field failure during training? | One fixed sample is a diagnostic, not a set-level estimate. |
| Reconstruction | Snapshot target, reconstruction, and error | Where are field structures preserved or misplaced? | Shared target/reconstruction limits are required within each field; error has its own nonnegative scale. |
| Reconstruction | Relative-$L_2$ distribution | How does physical-field accuracy vary over matched snapshots? | Report split, checkpoint, sample count, and source/child pairing. |
| Coherence quantity | Global-distribution discrepancies | How well do marginal, pairwise, and joint-tail quantities match reference fields? | Raw discrepancies and weighted totals have different meanings. |
| Coherence quantity | Cross-spectrum scores | How well do selected spectral terms agree across matched ensembles? | Scores and whiskers depend on the specified ensemble aggregation. |
| Coherence quantity | Topology H0/H1 persistence discrepancy | How close are the trained topology descriptors to their matched references? | A small descriptor distance does not establish physical causality or topological correctness outside the evaluated fields and raster. |
| Coherence interpretation | Joint field-density maps | Which joint-state regions are lost, shifted, or overrepresented? | Compare with shared bins, axes, and density normalization. |
| Coherence interpretation | Exact Betti curves and level-set geometry | How do connected-component and loop counts change across thresholds, and where do median-level boundaries differ? | State the scored raster, filtration direction, and reference-defined thresholds; a level-set image alone is not a persistence diagram. |

## Display rules

1. Put the scientific quantity and units on an axis or colorbar. State whether
   a value is raw, calibrated, weighted, normalized, or a bounded score.
2. For paired images, use one color scale for target and reconstruction of a
   field. For paired source/child plots, use the same samples and limits.
3. Use a distinct, zero-based scale for absolute error. Do not use a diverging
   palette for an unsigned quantity.
4. Keep titles short enough to survive a two-column page. Put run identity,
   checkpoint, split, sample count, and estimator details in the caption or
   sidecar report rather than a long figure title.
5. Keep operational histories separate from concise figures selected for a
   manuscript. Retain CSV/NPZ/JSON arrays so figures can be re-rendered.
6. A Betti or diagram plot must name H0 or H1, filtration direction, field or
   field pair, and its threshold or bar convention. Label a level-set image as
   geometry, not as a persistence diagram. Explain that the training distance
   uses fixed-angle sliced Wasserstein-1 on finite cubical diagrams plus a
   separate comparison of essential births.

The run history records family weighted objectives and the **combined** data
and coherence gradient norms and cosine. It does not record a separate
gradient for each coherence family. Family objective curves therefore show
what the optimizer was asked to minimize, while only the aggregate gradient
diagnostic can speak to update balance. A chart must not label family loss
fractions as each family's true parameter-update fraction.

## Audit of the cited figures

| Existing output | Observed problem | Figure decision |
| --- | --- | --- |
| A+B+C `loss_history.png` | The first axis repeats three lower panels, the legend covers data, and a title repeats “objective”. Its common logarithmic axis makes training-data and coherence scales hard to compare. | Keep as a concise operational overview with separate labeled quantities and fixed-panel validation. |
| A+B+C `coherence_history.png` | One row per low-level component makes the PNG exceptionally tall; small component captions are hard to read at page width. | Keep a compact family summary in the main monitor; move low-level component details to a separate diagnostic page or data table. |
| A+B+C `training_preview/latest_reconstruction.png` | Five field rows each repeat target and prediction colorbars, while dense black zero-error points obscure the error structure. | Keep the fixed-sample preview, share field color limits and a single target/prediction colorbar per row, and use a visibly legible error encoding. |
| A+B `relative_l2_violin.png` | The long provenance title and broad white space reduce legibility when placed in a two-column article. | Keep the set-level distribution with source/child samples aligned; move long provenance to its report or caption. |
| A+B coherence violins | Component densities, weighted total, and family total are shown together without enough separation of raw and weighted meaning. | Keep quantitative distributions, but label the aggregation level and units explicitly. |
| A+B cross-spectrum bars | The focused score axis helps resolve small differences, but the chart must remain visibly labeled as a focused 65–100% view. | Keep bounded scores and ensemble spread, with source/child plots on one scale. |
| A+B joint PDFs | The paired density panels communicate physical-state support well. Their claims depend on matched bins and normalization, currently documented in the report. | Keep as the principal intuitive distribution view; use shorter labels and shared scales. |
| A+B+C topology | No dedicated set-level H0/H1 figure or intuitive filtration view exists. | Add paired quantitative persistence scores and matched topology interpretation views, with the configured reduced raster named. |

## Recommended use

- **Main scientific narrative:** one matched field-reconstruction panel, one
  matched set-level fidelity figure, concise family-specific A/B/C quantitative
  panels with their own units, and one topology filtration/persistence
  explanation. These should identify
  source versus post-training checkpoint and distinguish descriptive agreement
  from a physical-mechanism claim.
- **Supplementary diagnostics:** all individual marginal/pairwise/joint-tail
  distributions, spectral terms and profiles, H0/H1 term breakdowns, and
  additional joint PDFs or topology examples. Keep the data and matching
  contract beside every figure.
- **Run monitoring:** objective history, component detail, gradient balance,
  fixed-sample preview, and checkpoint eligibility. These are operational
  diagnostics; a descending training objective is not itself an evaluation
  result.

## Suites

- [`history/`](history/README.md): compact training-objective, component,
  gradient-balance, and checkpoint-gate examples re-rendered from a read-only
  epoch-800 snapshot of the active A+B+C history.
- [`reconstruction/`](reconstruction/README.md): pinned full-grid field,
  sparse training-preview, and set-level accuracy examples.
- [`coherence/`](coherence/README.md): family-specific statistical and
  interpretive examples, with the topology preview's query limitation stated.
- `20260831_cross_spectrum_run_comparison/`: earlier reproducible comparison,
  retained as a historical baseline.

The [project README](../../../README.md) explains the plotting commands and
runtime output layout. Example-specific manifests and captions in each suite
are authoritative for the data actually rendered.

## Selected review examples

| Use | Example | Evidence boundary |
| --- | --- | --- |
| Optimization diagnosis | [Combined/data/coherence objectives](history/active_run_epoch_0800/loss_history.png), [gradient balance](history/active_run_epoch_0800/optimization_diagnostics.png), and [checkpoint gates](history/active_run_epoch_0800/checkpoint_fidelity.png) | The read-only snapshot ends at epoch 800; the live run continues. |
| Main field-fidelity view | [Senseiver full-grid test snapshot](reconstruction/senseiver_best_sample9000.png) and [200-sample test relative-$L_2$ distribution](reconstruction/senseiver_test_best_relative_l2.png) | The snapshot illustrates one case; the distribution summarizes saved test samples. |
| A+B+C qualitative monitor | [Fixed epoch-800 validation preview](reconstruction/abc_formal_epoch800_preview.png) | It contains 4,096 preview query points for one sample, not a split-level estimate. |
| Topology explanation | [Persistence distances](coherence/topology/persistence_term_distributions.png), [exact H0/H1 Betti curves](coherence/topology/betti_curves.png), and [q50 level-set geometry](coherence/topology/configured_grid_topology.png) | One pinned validation preview uses its saved seed-2027 queries; it is an auxiliary example, not the training-selector set benchmark. |
| A+B spectral comparison | [Matched configured cross-spectrum terms](coherence/ab_balanced_train_last/cross_spectrum_source_post_coherence.png) | The historical evaluator's self-spectrum and aggregate conflict with the resolved training configuration and are excluded. |
| A+B field fidelity | [Post-training 200-sample distribution](reconstruction/ab_balanced_train_last_relative_l2.png) | Source-set metric arrays were not retained, so this refreshed plot is post-only. |
