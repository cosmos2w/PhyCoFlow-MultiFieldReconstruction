# Multiscale spectral coherence for post-training

`multiscale_spectrum` compares conditional reconstructions at smooth spatial
scales using a fixed operator and scales fitted on TRAIN references. For a
generative model, use `signed_band_energy_score`, fair alpha `1`, and a positive
joint energy-score weight. The [mathematical reference](../src/phycoflow_reconstruction/coherence/families/multiscale_spectrum/README.md)
defines the operators, distances, descriptors and artifact contract.

## Definition and interpretation

For each field and connected component, remove the mass-weighted mean. Filter
this fluctuation field into overlapping bands, then divide each band by its
frozen training-reference scale. The objective is

```text
L = sum_b w_b mean_p ES_alpha(project_p(scaled signed band_b), target)
    + eta ES_fair(full scaled signed-band vector plus scaled piece means, target)
    + beta CRPS_alpha(normalized band coherence, target)
```

For J independent reconstructions conditioned on the same observations,

```text
ES_alpha = mean_j distance(X_j, Y)
           - [1 - (1-alpha)/J] sum_{j<k} distance(X_j, X_k) / [J(J-1)]
```

The first part rewards agreement with the reference; the second permits
conditional variation instead of forcing every draw toward one reconstruction.
All draws retain gradients. Two draws suffice for an unbiased fair score
estimate, although more draws reduce Monte Carlo noise.

At alpha `1`, the expected score is proper. Alpha below one adds a dispersion
penalty and is not proper at finite J. Separate band scores with finitely many
field projections do not identify arbitrary dependence across fields and bands.
A positive fair joint score closes that identification gap **within the retained
representation**, assuming fixed geometry/scales and finite first moments.
The new `signed_spectral_mean_augmented_v4` appends scaled piece means to the
same joint vector and calibrates their scales with a declared positive task-unit
floor. The preserved `signed_spectral_v3` score omits these means. Excluded nodes
and unresolved modes remain outside both claims. Propriety of the Scale score
does not establish propriety of paired State/Shape penalties or the combined
ConFIG direction, and does not guarantee successful optimization.

Band labels such as 16, 8, 4 and 2 km are center wavelengths of overlapping
filters: progressively finer spatial structure, not separate grids or hard
frequency bins. Physical labels require a physical FEM geometry. With
`coordinate_scaling: raw_extent`, lengths use the dataset's raw coordinate
units; `lonlat_degrees` uses an equirectangular approximation in kilometres.
The `knn_graph` fallback has graph scales, not physical wavelengths.

## Configuration

Merge [the portable spectral fragment](../configs/coherence/multiscale_spectrum.yaml)
into a valid case post-training configuration. It follows all dataset fields by
default. The case supplies the source run, sensors, coherence batch/point counts
and matching `evaluation.query_points`. The FEM operator requires a suitable
one- or two-dimensional geometry. Inspect mesh exclusions, connected pieces,
cutoff and effective modes in the saved geometry record.

The fragment explicitly selects:

```yaml
objective:
  kind: signed_band_energy_score
  score_definition_version: signed_spectral_mean_augmented_v4
  mean_weight: 1.0
  conditional_draws: 2
  almost_fair_alpha: 1.0
  projections: axes_and_pairs
  joint_energy_score_weight: 0.25
  coherence_crps_weight: 0.0
```

The fragment also declares `calibration.field_unit_floor: 0.001` in model units;
this value is a case-specific tolerance and must be reviewed for each field.
The joint weight `0.25` is a starting configuration, not a universal optimum.
The optional coherence-CRPS auxiliary is disabled and requires at least two
fields when enabled. It measures normalized band dependence, not pointwise
reconstruction CRPS.

Defaults preserve the earlier experimental configuration contract:

| Setting | Omitted value | Portable generative fragment |
|---|---|---|
| Objective | `paired_descriptor_mse` | `signed_band_energy_score` |
| Conditional draws for a generative objective | `2` | `2` |
| Signed-score alpha | `0.95` | `1.0` |
| Joint score definition | `signed_spectral_v3` | `signed_spectral_mean_augmented_v4` |
| Joint score weight | `0` | `0.25` |
| Coherence-CRPS weight | `0` | `0` |

Set the generative objective and alpha explicitly. A positive joint weight
requires alpha `1`. The deterministic descriptor-MSE objective and descriptor
energy-score objective remain available for ablations. Descriptor errors are
report-only when training signed fields.

## Shared trainer and recovery

The shared trainer fits geometry, band scales and floors on TRAIN references
before evaluating the source, preserving the model's random stream. It saves
`artifacts/multiscale_spectrum_family.pt`. Calibration and scoring use the same
fixed shared query identities. Structured-grid sources currently require all
native grid points for spectral calibration.

Each update reconstructs J independent samples from identical observations.
Dense targets enter the loss only; they never enter these reconstructions.
Distribution and topology reuse the first draw. Spectral scoring receives the
additional draws. Family-gradient diagnostics reuse the existing graph.
Evaluation also supplies the required independent draws and reports band power
and ensemble diagnostics; ordinary reconstruction metrics still describe the
first draw. The reported inference duration describes that primary draw, not the
extra spectral evaluation work.

The family composes through the existing weighted-sum and multitask routes,
including SOURCE-relative external fidelity policies. Internal ConFIG may use
`multiscale_spectrum` in place of `cross_spectrum` alongside distribution and
topology; every enabled family enters its gradient combination. The historical
`lpc_ordered` algorithm defines fixed A/B/T losses and does not accept this family.
Use initial gradient-norm calibration or explicit fixed weights as appropriate;
equal gradient norms at initialization do not imply equal influence later.

Resume restores the saved family artifact, model, optimizer, calibration and
random states. Do not refit scales using validation data or edit a run's resolved
configuration. Artifact version is `1`; reports record the selected score
definition. The legacy mean-free score remains `signed_spectral_v3` and the
mean-augmented score has its own version and component name. Existing model
state-dict layouts and other coherence-family defaults are unchanged. Older experimental optimizer
checkpoints are not interchangeable with the current upstream fidelity-policy
checkpoints.

## Evaluation and visualization

Compare source, native-continuation control and candidate checkpoints on identical
states, sensors, geometry, scales and latent seeds. Match production integration
steps, solver, observation guidance, live/EMA weights and query execution. Use
more evaluation draws than the two used for training and report paired
uncertainty across independent conditions, not across spatial nodes or draws.

For each band, show the reference, ensemble mean, one matched draw, and that draw
minus its ensemble mean with shared colour limits. Pair these maps with:

- Signed-band and joint scores, per field and band where available.
- Member power, shared power and varying power, compared with reference power.
- Ensemble variance and debiased ensemble-mean error, retaining both quantities
  when their ratio is undefined.
- Reconstruction and native-loss changes, so mean or unresolved-scale failures
  are visible despite a better spectral score.

The ensemble report uses `null` for ratios with nonpositive debiased error and
includes valid-state counts. Sharper maps or increased spectral power alone do
not establish calibrated variation. Check sampler-step convergence, spatial
refinement, sensitivity to band/weight choices, and conditional learnability on a
synthetic process with known observation-dependent uncertainty. Reserve an
untouched final test panel for the final claim.

### Frozen common yardstick

The run-owned recovery evaluator restores that run's original geometry and
scales; it is not a common cross-arm score merely by changing its query count.
For comparison, fit a bundle of legacy and next-definition families once on
TRAIN references and SOURCE, and freeze a single 4096-query panel with seed
100047, independent of training's query seed. Construct
`FrozenCanonicalEvaluator` from `evaluation.canonical_coherence` with unique
family names and aggregation modes: `paired` for State/Shape,
`ensemble_group` for the legacy Scale score, and `conditional` for the new
Scale score. The default contract scores four draws and complete 32-condition
groups. It averages paired scores over members, preserves each raw leaf and
group result, and computes separate per-member and ensemble-mean field MSE.
No group is dropped and no family is fitted or restored by this evaluator.

Call `evaluate` with the saved `[J,B,N,C]` members, the same `ObservationBatch`,
normalizer, effective sampler and ordered reference IDs for each checkpoint.
The sampler can be constructed with `evaluation.spectral.effective_sampler`,
with live weights and the actual number of draws. Distinct latent seeds, solver,
steps, observation guidance/clamp, query execution, weights, observation and
query tensors, normalization, reference IDs and family artifact contents enter
the comparison fingerprint. Checkpoint identities and each saved-ensemble
hash are recorded separately. `assert_comparable` rejects mismatched reports;
`save_canonical_score` writes numeric NPZ leaves and JSON provenance. The caller
must ensure the declared seeds and sampler describe how each saved member was
actually generated, and must load the frozen family bundle *after* checkpoint
loading if a checkpoint loader also restores training-family states.

The spectral `arms` utility requires the same frozen run-owned spectral
artifact; it checks sampler settings and reports sampling, scoring/transfers,
checkpoint and payload costs separately. It is not a replacement for the
common canonical bundle. Its optional `--groups-json` has a declared `unit`
(`independent_cluster` or `time_block`) and `labels` mapping every sample ID to
a case-supplied group. One cluster has no 95% interval; time-block ratios are
descriptive. Without a group file, only point changes are shown.

The reconstruction renderer accepts saved `physical_mask`, `physical_weights`
and `physical_measure` arrays and annotates relative L2 on that exact support.
Without these keys it explicitly reports equal valid nodes, preserving legacy
payloads. The current standalone legacy topology set renderer rejects new
finite-primary or native-mask families because its Betti and diagram paths have
not been qualified for those definitions; score the frozen next Shape family
through the canonical interface. Its existing standalone representative
figures are separate from the matched explanatory view.

## Verification

With development and relevant optional dependencies installed:

```bash
ruff check src tests scripts cases benchmarks
pytest
```

Focused tests are `tests/test_multiscale_*.py` and
`tests/test_spectral_score_contract.py`, `tests/test_spectral_evaluation.py`, and
`tests/test_canonical_coherence.py`. They cover mesh spectra and invariances,
masked/disconnected geometry, TRAIN-only calibration, artifact round trips,
score/gradient identities, projection blind spots, calibrated versus collapsed
ensembles, independent target-free draws, and shared training/resume behavior.
