# A+B+C formal-run example gallery

These examples use a stable, SHA-256-verified copy of the formal run's
`last.pt` at **epoch 1440, step 54720**. The checkpoint came from
`runs/coherence_fix_ABC_sliced_persistence_formal_5000ep_gpu1/20260924T030724Z_4208231c`.
The live 5000-epoch run was read only. See [snapshot provenance](snapshot_provenance.json)
for its exact hash and capture time. This is a diagnostic snapshot, not a
completed formal-run result. The source checkpoint is the base checkpoint
recorded in the A+B+C run lineage.

All set-level comparisons use the same 64 evenly spaced **validation**
snapshots, sensor selections, generation settings, and sample IDs. Cross-spectrum
uses two complete 32-snapshot ensembles. Standard `visualize-run` post-processing
was run on CPU against the frozen copy with `--eval-coherence
global_distribution cross_spectrum topology --extraview-coherence`.
The [standard report](postprocessing/report.json),
[comparison report](postprocessing/comparison_report.json), and saved CSV/NPZ
files record the exact evaluation contract. Paths inside those generated reports
point to the local ignored `.work/run_snapshot`; the copied figures and numerical
artifacts below are versioned here. The 88 MB checkpoint copy stays ignored,
while the saved metrics and source plotting payloads allow figure-only rerenders.

## Training history and reconstruction

| View | Example | What to read |
| --- | --- | --- |
| Coherence objectives | [History](history/coherence_history.png) | First row separates total coherence at left from three weighted family contributions at right, each with its own vertical scale. Raw components follow below. |
| Data and total objectives | [Loss history](history/loss_history.png) | Training curves and fixed-panel validation are operational diagnostics. |
| Update balance | [Gradient diagnostics](history/optimization_diagnostics.png) | The saved gradient comparison is combined data versus combined coherence, not one gradient per family. |
| Checkpoint gate | [Fidelity history](history/checkpoint_fidelity.png) | Selection state through the frozen epoch-1440 history. |
| Full-grid fields | [Reconstruction](reconstruction/fullgrid_validation_frame8000.png) | One validation frame on the dataset's physical x/y range, with shared truth/prediction color scales for each field. |
| Sparse demonstration | [4,096-query view](reconstruction/sparse_preview_from_fullgrid.png) | Fixed seed-2027 subset of the **same full-grid inference**. This is an illustrative sparse view, not a separate trainer preview. |
| Set-level fidelity | [Relative L2 distribution](postprocessing/relative_l2_violin.png) and [source](postprocessing/relative_l2_violin-base.png) | Same 64 snapshots; see [per-sample CSV](postprocessing/relative_l2.csv). |

The reconstruction [manifest](reconstruction/manifest.json) and
[full-grid NPZ](source/fullgrid_validation_frame8000.npz) retain coordinates,
field values, observations, and sample identity. The [history manifest](history/manifest.json)
describes the frozen, compact JSONL snapshot.

## Complete coherence suite

The `postprocessing/coherence/` directories are the repository's standard
post-processing outputs. Each configured family has current and `-base` source
figures, numerical metrics, and reports. The `explanatory/` views add direct
interpretation from those saved outputs; they do not evaluate a new model.

| Family | Standard quantitative figures | Explanatory views |
| --- | --- | --- |
| Global distribution | [Marginal fields](postprocessing/coherence/global_distribution/marginal_field_distributions.png), [pairwise fields](postprocessing/coherence/global_distribution/pairwise_field_distributions.png), [joint/top-tail](postprocessing/coherence/global_distribution/joint_top_tail_distributions.png), and [all 10 matched joint PDFs](postprocessing/coherence/global_distribution/global_distribution_extra/) | [Weighted terms](explanatory/global_weighted_terms.png) and [three-way CO–T density](explanatory/global_CO_T_joint_density.png) show reference, source, and A+B+C on shared bins. |
| Cross spectrum | [Same-frequency scores](postprocessing/coherence/cross_spectrum/same_frequency_coherence.png), [cross-frequency scores](postprocessing/coherence/cross_spectrum/cross_frequency_coherence.png), and [band profiles](postprocessing/coherence/cross_spectrum/spectral_band_profiles.png) | [Every configured field pair](explanatory/cross_pair_scores.png), [reference/source/A+B+C band energy](explanatory/cross_band_three_way.png), and [percentage-point deviations](explanatory/cross_band_error.png). Self spectrum and spectral-band-energy loss are disabled in this run. |
| Topology | [Sliced-persistence component distances](postprocessing/coherence/topology/persistence_term_distributions.png), [exact H0/H1 Betti curves](postprocessing/coherence/topology/betti_curves.png), and [matched raster geometry](postprocessing/coherence/topology/configured_grid_topology.png) | [Physical-domain CO/T maps](explanatory/topology_field_maps.png), [CO](explanatory/topology_filtration_CO.png) and [T](explanatory/topology_filtration_T.png) reference-threshold sublevel masks, [all three mutual CO–T lines](explanatory/topology_mutual_CO_T.png), and [CO](explanatory/topology_diagrams_CO.png) / [T](explanatory/topology_diagrams_T.png) finite H0/H1 birth–death diagrams in both filtration directions. |

PNG and PDF forms are provided for explanatory views and reconstructions. The
standard evaluator also writes SVG. The [explanation manifest](explanatory/manifest.json)
records the representative validation sample, physical raster extent, and a
check that the recomputed self-diagram sliced distance agrees with the saved
score. Diagram scatter plots show finite bars; text reports essential-class
counts. The topology claim is limited to the configured 32×128 raster, fields
CO/T, and reference-defined filtration thresholds.

**Interpretation:** Weighted objective size is not a family's fraction of the
parameter update. The history contains an aggregate data/coherence gradient
comparison, while the set figures measure descriptive agreement. The plotted
source and A+B+C samples are matched, but this 64-snapshot validation subset
does not establish final held-out efficacy.

## Re-render

From the repository root, with `phycoflow_env` active:

```bash
PYTHONPATH=src python cases/turbulent_combustion/diagnostics/ExampleVisual/render_history.py \
  --run-dir cases/turbulent_combustion/runs/coherence_fix_ABC_sliced_persistence_formal_5000ep_gpu1/20260924T030724Z_4208231c \
  --through-epoch 1440
PYTHONPATH=src python cases/turbulent_combustion/diagnostics/ExampleVisual/render_reconstruction.py
PYTHONPATH=src python cases/turbulent_combustion/diagnostics/ExampleVisual/render_coherence.py
```

The first command reads only the live run's saved history. The latter two use
versioned payloads and metrics, so they require no checkpoint or live process.
For a new full standard evaluation, copy a **stable** checkpoint and its config
to an isolated run directory, then run `visualize-run` there; use the command
recorded above with your chosen split, sample count, and checkpoint. Never point
these example-generation outputs into the running training directory.
