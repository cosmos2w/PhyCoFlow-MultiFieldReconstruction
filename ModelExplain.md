# Model and Coherence Conventions

## Reconstruction contract

A snapshot task maps sparse scalar observation tokens
`(coordinate, value, field_id)` to all requested fields at query coordinates.
KS uses the same interface after flattening its logical `[time, x, field]`
state into `(t, x)` query points; it is interpolation/reconstruction, not future
forecasting.

All models consume the same padded observation/query batch. Point models use
tokens directly. Grid models use only the value and mask rasters constructed
from those tokens; unobserved target values never enter conditioning.

## Initial model families

- Deterministic: coordinate MLP, MLP-RBF, DeepONet, Senseiver, and Geo-FNO.
- Physics-informed: PINN uses the coordinate regressor plus an active
  case-provided PDE/boundary loss; without that provider it is not called PINN.
- Generative: grid DiffusionPDE-style denoising and latent flow matching.
- Main realization: PointCloudFFM with `GL_rbf_ENH/topk_rbf`; an FNO flow
  backbone is optional for regular grids.

The isolated Demo 50 compatibility model exists only to load the historical
`topk_rbf_glres` checkpoint. It is not a selectable new-training mode.

Latent flow is intentionally two-stage. Stage 1 trains the convolutional
autoencoder. Stage 2 requires a Stage-1 checkpoint, strictly loads and freezes
that autoencoder, and trains only the sparse-conditioned latent velocity and
conditioning modules. This prevents a nominal Stage-2 run from silently using
a random latent representation.

PointCloudFFM follows the 1-rectified-flow bridge
`x_t=(1-t)x_0+t*x_1`, with target velocity `x_1-x_0`. Its RFF prior and all
generative reconstruction methods accept explicit random generators. The
GL-RBF implementation chunks query-to-sensor distance and latent-readout work,
so full combustion clouds and KS space-time states need not allocate a single
`Q × M` distance tensor.

## Data-driven coherence taxonomy

The first implemented family is global distribution coherence:

```text
global_distribution
├── self.marginal_w2
├── mutual.pairwise_swd
└── cross.joint_topk_swd
```

`self`, `mutual`, and `cross` are components of that family, not top-level
coherence categories. Cross-spectrum and topological coherence are future peer
families with their own component structures. Observation consistency is a
reconstruction constraint, and PDE coherence comes from a case physics
provider; neither is a data-driven coherence family.

Every future family/component must document target use, units, normalization,
differentiability, reference provenance, required geometry, estimator, and
failure cases. General calculations belong in the package; field groups,
wavenumber bands, thresholds, and weights belong in case configs.

### Global-distribution estimators

Let generated and reference empirical states be `X,Y in R^(N×C)`. Inputs must
have equal point count, finite values, and an explicitly declared field subset.
They are evaluated in normalized model units unless `units: physical_units`
requests the run's serialized training normalization inverse.

- `self.marginal_w2` sorts each channel independently and computes
  `W2_c^2 = mean_i (sort(X_c)_i-sort(Y_c)_i)^2`. Its sample cost is the
  configured non-negative weighted mean over fields.
- `mutual.pairwise_swd` selects every configured field pair, projects its 2-D
  empirical values onto a fixed seeded bank of unit directions, computes the
  exact sorted 1-D empirical `W2^2` along each direction, then averages over
  directions and pairs.
- `cross.joint_topk_swd` projects the configured joint field vector onto a
  fixed seeded Sobol-normal bank, optionally prefixed by channel axes. It
  averages the largest configured fraction of per-direction `W2^2` values.

All three implementations are batched and differentiable with respect to the
generated state. Projection directions, component settings, field-name/index
mapping, normalization, version, and weights are serialized in
`global_distribution_family.pt`. The migrated estimators are numerically
checked against the historical turbulent-combustion implementation.

For a one-field case such as KS, `self.marginal_w2` is valid while mutual and
cross components are explicitly disabled or return an unavailable reason. No
multi-field score is manufactured.

### Reference and target-use modes

`target_use: training_reference` fits or loads a versioned empirical bank from
the training split only. The artifact records the dataset fingerprint, split
strategy, exact sample IDs and point indices, seed, fields, units, and
normalization. During the coherence path, the paired dense target is removed
from the batch before query selection and rollout; references come only from
this frozen bank. Validation/test dense states are used later for evaluation
metrics, never for fitting the bank or the target-free training loss.

`target_use: paired_supervised` compares the generated endpoint with the dense
target from the current training sample. This reproduces the original Demo50
direct-coherence semantics, but is labeled paired-supervised in configs,
histories, checkpoints, and evaluation files. It must not be reported as
target-free refinement.

### Differentiable post-training

The common post-trainer accepts one differentiable-reconstruction contract.
PointCloudFFM and the isolated Demo50 adapter use their native Euler/Heun
rectified-flow rollout; coordinate regressors, MLP-RBF, DeepONet, Senseiver,
and GeoFNO use their direct differentiable prediction; DiffusionPDE uses its
DDIM path; and latent FM stage 2 uses latent rectified-flow sampling followed
by differentiable decoding. Latent FM stage 1 is only an autoencoder
prerequisite, while PINN has no plain base checkpoint and remains in the
separate direct-physics stage.

`endpoint_smooth` forms a chunked Gaussian map from sparse sensor coordinates.
For rectified flow it guides the estimated clean endpoint during integration;
for other adapters it is the shared differentiable endpoint blend. Exact final
clamping is applied only where query and sensor index mappings coincide. This
observation constraint remains outside the data-driven coherence registry.

The optimizer supports scheduled weighted sums and optional two-objective
ConFIG updates through `conflictfree`. Histories record component losses,
reference IDs, objective gradient norms/cosines/conflicts, the actual update
mode, and any fallback. Every post-training run is a child of an immutable
source run, records all inherited keys and source hashes, and saves fixed-
manifest before/after evaluation without overwriting its parent.

Phase-6 comparisons additionally require the same dataset fingerprint,
training-reference-bank hash, evaluation sensor-manifest hash, comparison-query
hash, training/evaluation seeds, split, and target-use label across every pair.
The report includes absolute and relative MSE/coherence changes, parameter
count, post-training time, stable warmed-up inference time, and peak CUDA
memory. Complex spectral parameters are represented as real/imaginary pairs
inside gradient balancing and restored without changing their dtype.

## Training stages

1. `base_training`: native supervised/diffusion/flow data objective only.
2. `post_training`: immutable source checkpoint plus an explicitly labeled
   training-reference or paired-supervised coherence objective, written as a
   child run.
3. `direct_physics`: data and PDE objectives active from initialization.

Resolved configs, dataset/sensor hashes, normalization, code state, and model
versions are stored with every run.

### Verified PDE paths

The initial differentiable physics provider is case-owned by Brusselator. It
decodes predictions to physical units, reshapes the complete 192×192 periodic
grid, applies spectral Laplacians, and evaluates both reaction-diffusion
equations. Temporal derivatives are paired finite differences from adjacent
stored frames and are labeled with that provenance; they are never described
as autograd-in-time derivatives. Positivity is a separately weighted term.

`direct_physics` constructs a PINN coordinate regressor with data and physics
losses active from initialization. Physics `post_training` strictly loads a
plain differentiable checkpoint, creates an immutable child, and balances data
retention against the same provider. These paths do not share keys with the
data-driven coherence configuration and are validated separately.

Kolmogorov currently exposes periodic divergence, pressure-gauge, vorticity,
energy, enstrophy, and paired-time momentum diagnostics; it does not yet claim
a training PhysicsProvider. KS reports full `(t,x)` layout checks and separate
space/time derivative errors. Mass transport uses nonperiodic derivatives and
boundary-normal flux diagnostics and remains integration-only.
