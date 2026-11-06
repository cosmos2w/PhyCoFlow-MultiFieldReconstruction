# Multiscale spectral coherence

`multiscale_spectrum` scores conditional reconstructions on a fixed spatial filter
bank. The recommended generative objective is the fair signed-band score plus a
positive fair joint score. This reference defines the mathematics and serialized
contracts; the [post-training guide](../../../../../docs/spectral_post_training.md)
covers configuration, shared training and evaluation.

For each selected field, subtract its mass-weighted mean on each connected
component, filter it into smooth bands, and divide each band field by its frozen
training-reference scale. The resulting vector retains spatial arrangement and
sign at the represented scales. Geometry, retained modes, scales and projections
are independent of the predictions being scored.

## Signed fields and joint coherence

`signed_band_energy_score` scores signed band fields instead of only aggregate descriptors. Each field is divided by its frozen band scale; scalar spatial fields are obtained with fixed axes and optionally `(e_f ± e_g)/sqrt(2)` projections. `objective.projection_pairs` can list selected field-name pairs; omitting it scores all pairs. The norm uses normalized analysis mass. Scores are averaged across projections and then positive normalized `band_weights`.

For independent draws, `ES = mean_j d(X_j,target) - r_J sum_{j<k} d(X_j,X_k)/(J(J-1))`, where `r_J = 1-(1-alpha)/J`. `almost_fair_alpha: 1.0` gives the exact fair U-statistic. The legacy default `.95` adds a positive dispersion penalty; its finite-J expectation is **not** a proper distributional score. Finite field projections and separate band scores also do not identify arbitrary joint dependence.

`joint_energy_score_weight: eta > 0` adds a fair energy score of the entire signed-band vector, with distance

```text
d_joint(x,y)^2 = sum_{b,n,f} q_n ((F_b(x_f)_n - F_b(y_f)_n) / s_bf)^2
              = sum_{k,f} (a_kf(x)-a_kf(y))^2 sum_b g_b(k)^2/s_bf^2 / sum_n M_n
```

The second form uses the existing M-orthonormal coefficients and avoids synthesizing another spatial tensor. The default `score_definition_version: signed_spectral_v3` keeps exactly this mean-free score and its existing component name, `joint_spectral_energy_score.conditional`. Configuration requires `almost_fair_alpha: 1.0` when the joint weight is positive.

The opt-in `score_definition_version: signed_spectral_mean_augmented_v4` appends one scaled mean for every connected piece and field to the **same** joint vector:

```text
d_aug(x,target)^2 = d_joint(x,target)^2
  + mean_weight * sum_{piece,field} pi_piece * ((mean_x - mean_target) / mean_scale)^2
mean_scale[piece,field] = sqrt(var_TRAIN(piece_mean) + field_unit_floor[field]^2)
pi_piece = sum(M on piece) / sum(M on all retained pieces)
```

Its component name is `joint_spectral_mean_energy_score.conditional`. `mean_weight` defaults to 1 and must be positive; a positive joint weight and an explicit `calibration.field_unit_floor` are required. The variance is the population variance across the selected TRAIN states, accumulated in float64 with a stable running algorithm. The floor is a positive scalar in the common declared field units or a mapping containing every selected field's positive tolerance. Specify case-meaningful tolerances before training; the example's 0.001 is a starting tolerance in normalized model units, not a universal physical threshold. The calibration record and tensor state freeze the exact field floors, mean scales, piece weights and score version. Diagnostics separately report first-draw piece-mean bias and absolute error in the declared units.

With positive joint weight, fixed geometry/scales, independent conditional draws and finite first moment, the fair score identifies the joint law of its represented vector. Version v4 includes the retained piece means and their dependence on fluctuations; excluded nodes and unresolved modes still lie outside that representation. This is not a finite-data learning or calibration guarantee. Adding paired State/Shape objectives generally changes the proper-scoring interpretation of the combined training update; ConFIG need not optimize a scalar objective at all.

`coherence_crps_weight: beta` optionally adds CRPS of `rho_bfg = G_bfg / sqrt((G_bff+delta_bf)(G_bgg+delta_bg))`. Fixed floors make gain invariance approximate only at power well above the floors. A positive weight requires two or more fields; one-field runs return a zero, unavailable diagnostic at weight zero. This auxiliary uses the same alpha as the signed-band score.

Every draw keeps its gradient. Projected-band and joint distances are computed directly from filtered coefficients using mass orthonormality and the same total-mass normalization. Differences are formed before projection, preserving the zero subgradient at coincidence without smoothing. Spatial band fields are synthesized for requested ensemble reports or nonlinear moment diagnostics. Field-space and coefficient-space values and all draw gradients are regression-tested. Descriptor errors remain diagnostics for signed objectives. The joint term is registered only when enabled, preserving legacy component manifests.

For a deterministic model, `paired_signed_band_mean_distance` with definition `paired_signed_spectral_mean_v1` uses exactly one prediction. It sums the normalized, band-weighted projected point distances and `joint_distance_weight` times the joint signed-band/piece-mean point distance. Set a positive joint weight, positive `mean_weight`, and an explicit TRAIN task-unit floor. Its component paths are `paired_signed_band_distance.paired` and `paired_joint_spectral_mean_distance.paired`; signed Gram descriptors remain evaluation diagnostics. This paired point query makes no conditional-law or proper-score claim. The complete combustion example is [coherence_deterministic.yaml](../../../../../cases/turbulent_combustion/configs/posttrain/coherence_deterministic.yaml), with an explicit deterministic source distinct from the stochastic experiment source.

## Geometry and bands

- **Operator.** `fem_p1` assembles the P1 finite-element stiffness (cotangent weights) and a mixed-Voronoi lumped mass. It solves `K phi = lambda M phi` with `M`-orthonormal eigenvectors, which are mean-free on every piece. Without periodic axes this is the Neumann Laplacian, and `sqrt(lambda)` is a wavenumber in the declared length unit. `coordinate_scaling` defaults to `raw_extent`, which undoes the loader's per-axis normalization so non-square domains keep their aspect ratio. `lonlat_degrees` gives equirectangular km, and `none` keeps normalized coordinates. `knn_graph` is a fallback for any dimension; its bands are graph scales, not wavelengths.
- **Bands.** Bands are given as center wavelengths, or as a `count` that is log-spaced up to the cutoff. Adjacent bands cross over with cos/sin windows in log wavenumber. The finest band crosses over in the same way into a discarded unresolved band that ends at the resolution cutoff, `min_wavelength` (default: `resolution_factor` x p90 nearest-neighbor query spacing). The bank is therefore tight: `sum_m g_m^2 = 1` up to the finest center. It then decays smoothly to zero, so the least accurate, most mesh-sensitive modes near the cutoff carry little weight. The missing part is reported as unresolved energy.

## Descriptor diagnostics

Each state `X` (fields in the declared units) is reduced to two parts:

1. **The band Gram matrices** `G^m_ij = <F_m q_i, F_m q_j>_M`, where `q` is `X` with each connected piece's mean removed. One positive semidefinite matrix per band holds every second-order quantity at band resolution:
   - `log_band_power`: the diagonal (the auto, or self, spectrum), `log(1 + P_mi / delta_mi)`;
   - `signed_band_cospectrum`: the off-diagonal (same-scale cross spectra), `asinh(G^m_ij / (s_mi s_mj))`, with the sign kept. All field pairs are used by default.
2. **Cross-scale coupling** (`third_order_product_moments`, opt-in). The smooth band filters overlap, so cross-band linear inner products need not vanish. This descriptor adds nonlinear information using `asinh` of the third central moment of three scaled band fields; globally this is `<f g, h>_M`. Tuples are an explicit list of `[field, band]` triples and are symmetric in their factors. The useful shapes are:
   - coarse × coarse → mid, a quadratic self-interaction;
   - coarse × fine × fine, the organization of fine-scale intensity by a coarse structure.

   Optional regional versions use fixed raised-cosine windows.

Legacy floors are `delta_mi = rho_band Pbar_mi + rho_abs Vbar_i` and scales are `s_mi = sqrt(Pbar_mi + delta_mi)`. `Pbar` is the mean training band power and `Vbar` the mean training fluctuation power. With an explicit `field_unit_floor`, add its square to each field's power floor, including the corresponding regional factor floors. This enables genuinely constant or weak TRAIN fields while preserving a positive, prediction-independent denominator; it never excludes a field because the prediction has no power. Without this opt-in, the legacy featureless-field rejection and scale formula are unchanged. All quantities are fitted once from training references by `fit_training_reference`, before the source evaluation, and frozen in the family artifact. No denominator depends on a prediction or on validation/test data.

The band fields and all Gram/moment descriptors remain insensitive to constant offsets per connected piece; v4 includes those means only in its joint representation. Check piece-mean and fieldwise reconstruction errors separately. For v4 in `physical_units`, `numerics.dtype: float64` is required so physical scaling, offset addition and mean removal run in double precision. Casting cannot recover precision already lost in the supplied input or normalizer.

## Descriptor objectives

- `paired_descriptor_mse`: `sum_c lambda_c mean_{descriptors in c} (Phi_c(Xhat) - Phi_c(Y))^2` per state. This is a deterministic descriptor objective and a single-draw ablation for flow models; the signed point objective above retains spatial arrangement and piece means.
- `conditional_energy_score` (`conditional_draws: J >= 2`): `mean_j d_W(z_j, y) - sum_{j != l} d_W(z_j, z_l) / (2 J (J-1))`, where `d_W` is the weighted Euclidean norm with `W = lambda_c / d_c` per block. With `standardize_descriptors: true` each entry is further divided by that descriptor's variance across the training calibration states (floored at 1e-3 of its block's mean), so every descriptor counts in units of its natural spread rather than of its transform; well-matched, high-variance descriptors then stop dominating the distance.
  - The post-trainer draws the `J-1` extra reconstructions from the same target-free observations with independent latent noise, and they stay differentiable.
  - At coincident vectors the norm takes the zero subgradient; no smoothing is added.
  - The per-block errors become evaluation-only diagnostics.
  - The source model must be generative.

Block normalization means correlated descriptors within a block share the block's weight rather than adding to it.

## Irregular and disconnected domains

- **`support_factor`.** Drops Delaunay triangles whose centroid or edge midpoints lie farther than `support_factor` times the local `support_neighbors`-th-neighbor spacing from the dataset's full point set. This handles masked domains sampled by a query subset.
- **`triangulation: structured_indices`** with `structured_indices: <auxiliary array>`. Builds elements from per-point logical `(i, j)` grid indices: full cells are split along the shorter diagonal, and three-corner cells add one triangle. This is a specific masked-cell convention, which must be checked against the case's native mesh; every grid point must be in the query set.
- **`connectivity: components`.** Keeps each connected piece holding at least `min_component_fraction` of the points, each with its own constant null mode. The default, `require_connected`, raises instead.

`geometry_record` (in `artifacts/multiscale_spectrum_family.pt`) records the mesh, the pieces, the excluded points and area, the cutoff, band centers, and effective modes per band. Read it before trusting a geometry.

## Limits

- **Descriptor objectives:** aggregate Gram/moment descriptors lose spatial phase and can miss sign-symmetric dependence. Signed-field scoring preserves spatial arrangements within the retained representation; finitely projected scores still have joint-law blind spots unless the joint term is enabled.
- **Not physics diagnostics:** third-order moments are not energy-transfer rates.
- **Resolution:** scales below the cutoff are only reported. Nonlinear products need resolution beyond their factors, so check spatial refinement before interpreting them.
- **Calibration reporting:** `spread_skill` is a retained alias for variance / debiased mean error. `rms_spread_over_rmse_pooled` is its pooled square root. Nonpositive debiased denominators are undefined (`null`), and valid-state counts accompany per-state ratios.

## Implementation map

| Module | Responsibility |
|---|---|
| [config.py](config.py) | One parser for family construction and strict project validation; fields, bands, objective options and legacy defaults |
| [operators.py](operators.py) | Physical/graph geometry, connected components, eigenbasis, filters and regional measures |
| [statistics.py](statistics.py) | Mean removal, spectral coefficients, signed fields, Gram matrices and moments |
| [features.py](features.py) | Descriptor layout, training-reference accumulators and fixed scales |
| [losses.py](losses.py) | Differentiable pairwise energy-score/CRPS kernels and coefficient-space joint distance |
| [objectives.py](objectives.py) | Signed-score component assembly, weights and stable report identifiers |
| [reporting.py](reporting.py) | Power/coherence summaries, ensemble decomposition and undefined-ratio serialization |
| [family.py](family.py) | Shared family contract, reference fitting, scoring orchestration and artifact lifecycle |

`MultiscaleSpectrumFamily.forward(generated, reference, coordinates=..., context=...)`
accepts `[batch, points, fields]` tensors. `generated` is the first draw;
`context["conditional_draws"]` supplies the remaining independent draws, each with
the same shape and conditioning. An evaluator can set
`context["score_all_conditional_draws"] = True` to score all supplied independent
draws with the same frozen family; diagnostics then report the actual draw count.
Without that explicit override, the configured draw count governs training.
`context["report_detail"]` enables detailed
reports, and `context["ensemble_draws"]` supplies additional members for ensemble
power diagnostics. Those report-only members do not change the training loss.
The family returns a `FamilyResult`; the shared trainer applies the outer family
weight and calibration scale.

Artifact version is still `1`. The artifact saves the family config,
selected fields, geometry/calibration provenance and tensor state. Loading requires
matching family identity, version and config and uses strict tensor-state loading.
Signed-score diagnostics use `score_definition_version`, a separate identifier for
score semantics. Legacy `signed_spectral_v3` component paths and tensor names remain
stable. The opt-in v4 adds the `piece_mean_scale` buffer and its distinct component
path. V4 artifacts also bind SHA256 digests of the scientific package modules and
the Torch, NumPy and SciPy versions; strict loading requires exact configuration
and implementation/backend identity. V4 also records the full input field order
and units, so loading a selected-field score against reordered channels is rejected.
The v3 artifact contract remains unchanged. The paired mean-point definition also binds the input field contract and implementation/backend identity. Because source digests include the score implementation, a v4 artifact made before an implementation change is rejected; a controlled new artifact must be created from verified matching geometry and TRAIN calibration rather than weakening the check.

`visualize-run --eval-coherence multiscale_spectrum` requires the run's frozen TRAIN family artifact and scores four independently seeded draws for conditional objectives. The deterministic point objective uses one prediction. For a SOURCE comparison, the child artifact is reused as the common Scale yardstick; incompatible geometry, configuration or artifact identity causes an explicit failure. This fixed-query report is not full-native evaluation unless its frozen artifact actually covers that support.

## Statistical background

- [Gneiting and Raftery: proper scoring rules and energy scores](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).
- [AIFS: finite-ensemble almost-fair scoring](https://www.nature.com/articles/s44387-026-00073-7).
