# Model, Training, and Coherence Logic

This document describes the mathematical contract implemented by
`phycoflow_reconstruction`: how sparse measurements become a full-field
reconstruction, how each registered model is trained and sampled, and how
data-driven or equation-based physical constraints modify a trained model.
The limitations listed here describe the current implementation and are
intended to guide later model updates.

## 1. Reconstruction problem and notation

For sample $b$, let the complete normalized physical state be

$$
\widetilde{\mathbf X}_b\in\mathbb R^{N_b\times C},
$$

where $N_b$ is the number of spatial or space-time points and $C$ is the
number of physical fields. A physical field value $X_{bnc}$ is normalized
channel-wise as

$$
\widetilde X_{bnc}=\frac{X_{bnc}-\mu_c}{s_c},
\qquad
X_{bnc}=s_c\widetilde X_{bnc}+\mu_c,
$$

with serialized offsets $\mu_c$ and positive scales $s_c$. Depending on the
dataset, $s_c$ is a training standard deviation, a robust scale, or one for
identity normalization.

The sparse observation set is

$$
\mathcal O_b
=\left\{\left(\mathbf r_{bm},y_{bm},c_{bm}\right)\right\}_{m=1}^{M_b},
$$

where $\mathbf r_{bm}\in\mathbb R^D$ is a normalized coordinate,
$y_{bm}\in\mathbb R$ is one observed scalar, and
$c_{bm}\in\{0,\ldots,C-1\}$ identifies its field. The requested query set is

$$
\mathcal Q_b=\left\{\mathbf q_{bq}\right\}_{q=1}^{Q_b}.
$$

Every reconstruction adapter implements the map

$$
f_\theta:\left(\mathcal O_b,\mathcal Q_b\right)
\longmapsto
\widehat{\mathbf X}_b\in\mathbb R^{Q_b\times C}.
$$

The common `ObservationBatch` pads observations and queries to tensors with
shapes $B\times M\times D$ and $B\times Q\times D$ and supplies Boolean masks
$m^{\mathrm{obs}}_{bm}$ and $m^{\mathrm{qry}}_{bq}$. It also carries original
point indices when a sensor or query corresponds to a known grid entry.
Point models consume tokens directly. Grid models rasterize only observed
values and binary support:

$$
V_{bcn}=
\begin{cases}
y_{bm},&\text{if observation }m\text{ measures field }c\text{ at point }n,\\
0,&\text{otherwise},
\end{cases}
\qquad
M_{bcn}=\mathbb 1[(c,n)\text{ is observed}].
$$

The separate mask makes an unobserved zero distinguishable from a measured
zero. Dense targets never enter this raster.

For direct supervised models, the shared masked data loss is

$$
\mathcal L_{\mathrm{MSE}}
=\frac{
\sum_{b,q,c}m^{\mathrm{qry}}_{bq}
\left(\widehat X_{bqc}-\widetilde X_{bqc}\right)^2
}{C\sum_{b,q}m^{\mathrm{qry}}_{bq}}.
$$

Kuramoto--Sivashinsky uses exactly the same interface after flattening its
$T\times X\times C$ state into queries with coordinates $(t,x)$. It is joint
space-time reconstruction, not autoregressive forecasting.

## 2. Shared coordinate and observation features

Several point models use fixed Fourier features. With linearly spaced
frequencies $f_\ell\in[1,32]$, the coordinate embedding is

$$
\phi(\mathbf r)
=\left[
\sin(\pi f_\ell r_d),
\cos(\pi f_\ell r_d)
\right]_{d=1,\ldots,D;\ \ell=1,\ldots,L}.
$$

The coordinate MLP and MLP-RBF models also use a compact per-field observation
summary. For field $c$,

$$
n_{bc}=\sum_m m^{\mathrm{obs}}_{bm}\mathbb 1[c_{bm}=c],
\qquad
\overline y_{bc}
=\frac{\sum_m m^{\mathrm{obs}}_{bm}\mathbb 1[c_{bm}=c]y_{bm}}
{\max(n_{bc},1)},
$$

and

$$
\mathbf h_b
=\left[
\overline y_{b1},\ldots,\overline y_{bC},
\mathbb 1[n_{b1}>0],\ldots,\mathbb 1[n_{bC}>0]
\right].
$$

This summary is derived only from observed entries and cannot expose an
unobserved dense target.

## 3. Deterministic reconstruction models

### 3.1 Coordinate MLP

The coordinate MLP broadcasts $\mathbf h_b$ to every query and predicts

$$
\widehat{\mathbf X}_{bq}
=\operatorname{MLP}_\theta
\left([\phi(\mathbf q_{bq}),\mathbf h_b]\right).
$$

The implemented network has three GELU hidden layers followed by a linear
$C$-channel head. Base training minimizes $\mathcal L_{\mathrm{MSE}}$.

Current drawbacks:

- $\mathbf h_b$ retains only each observed field's mean and presence flag; it
  discards sensor locations, value spread, count, and cross-sensor structure.
- The fixed sinusoidal embedding omits raw coordinates and can alias locations
  under its imposed periodic features.
- The model has no native mechanism for exact sensor interpolation, calibrated
  uncertainty, or grid-aware spatial coupling.

### 3.2 MLP-RBF

For every query and field, the model constructs Gaussian RBF weights

$$
w_{bqmc}
=m^{\mathrm{obs}}_{bm}\mathbb 1[c_{bm}=c]
\exp\left(
-\frac{\lVert\mathbf q_{bq}-\mathbf r_{bm}\rVert_2^2}{2\sigma^2}
\right),
$$

then local value and support features

$$
\ell^{\mathrm{value}}_{bqc}
=\frac{\sum_m w_{bqmc}y_{bm}}
{\max\left(\sum_m w_{bqmc},10^{-8}\right)},
\qquad
\ell^{\mathrm{support}}_{bqc}=\sum_m w_{bqmc}.
$$

The prediction is

$$
\widehat{\mathbf X}_{bq}
=\operatorname{MLP}_\theta
\left([
\phi(\mathbf q_{bq}),
\mathbf h_b,
\boldsymbol\ell^{\mathrm{value}}_{bq},
\boldsymbol\ell^{\mathrm{support}}_{bq}
]\right).
$$

Base training again uses $\mathcal L_{\mathrm{MSE}}$.

Current drawbacks:

- One fixed $\sigma$ is shared across fields, samples, and spatial regions;
  it cannot adapt to nonuniform sensor density or different correlation
  lengths.
- Computing the full distance tensor costs $\mathcal O(BQM)$ memory and time;
  unlike the PointCloudFFM backbone, this implementation is not chunked.
- Raw support magnitude varies with sensor count. If a field has no nearby
  effective support, its local value collapses toward zero and the network
  must infer the missing information from global summaries.

### 3.3 Sparse DeepONet

Each sensor token combines coordinate, scalar value, and one-hot field
identity:

$$
\mathbf z_{bm}
=\left[
\mathbf r_{bm},y_{bm},\operatorname{onehot}(c_{bm})
\right].
$$

A token branch produces $CP$ features and masked mean-pools them:

$$
\mathbf B_b
=\operatorname{reshape}_{C\times P}
\left(
\frac{\sum_m m^{\mathrm{obs}}_{bm}
\operatorname{Branch}_\theta(\mathbf z_{bm})}
{\max\left(\sum_m m^{\mathrm{obs}}_{bm},1\right)}
\right).
$$

The trunk produces a field-specific basis at every query,

$$
\mathbf T_{bq}
=\operatorname{reshape}_{C\times P}
\left(\operatorname{Trunk}_\theta(\phi(\mathbf q_{bq}))\right),
$$

and the reconstruction is the branch--trunk contraction

$$
\widehat X_{bqc}
=\sum_{p=1}^{P}B_{bcp}T_{bqcp}+a_c.
$$

Current drawbacks:

- Mean pooling is permutation invariant but represents interactions among
  sensors only indirectly through a sum of independently encoded tokens.
- The fixed basis dimension $P$ imposes a low-rank bottleneck that may
  underrepresent sharp or multiscale fields.
- All output fields share the same pooled observation set, and the model does
  not enforce observations exactly.

### 3.4 Senseiver regressor

Sensor tokens are embedded as

$$
\mathbf s_{bm}
=g_\theta\left([
\phi(\mathbf r_{bm}),y_{bm},\operatorname{onehot}(c_{bm})
]\right).
$$

Starting from learned latent vectors $\mathbf L^{(0)}$, the encoder performs
masked cross-attention from latents to sensors,

$$
\mathbf L^{(1)}
=\mathbf L^{(0)}
+\operatorname{MHA}
\left(\mathbf L^{(0)},\mathbf S,\mathbf S;m^{\mathrm{obs}}\right),
$$

followed by residual latent self-attention/feed-forward blocks. Query tokens
$\mathbf u_{bq}=h_\theta(\phi(\mathbf q_{bq}))$ read from the final latent set:

$$
\mathbf d_{bq}
=\mathbf u_{bq}
+\operatorname{MHA}(\mathbf u_{bq},\mathbf L,\mathbf L),
\qquad
\widehat{\mathbf X}_{bq}=\operatorname{Head}_\theta(\mathbf d_{bq}).
$$

Current drawbacks:

- A fixed number of learned latents creates an information bottleneck as
  sensor count and field complexity grow.
- Output attention costs $\mathcal O(BQL)$ for $L$ latents, and the current
  Senseiver implementation does not chunk large query sets.
- The compact architecture has no explicit local interpolation branch or
  exact sensor constraint.

### 3.5 GeoFNO regressor

GeoFNO receives the regular-grid tensor

$$
\mathbf Z^{(0)}=[\mathbf V,\mathbf M]
\in\mathbb R^{B\times 2C\times N_1\times\cdots\times N_d}.
$$

The maintained `neuraloperator` FNO applies layers whose essential spectral
operation is

$$
\mathbf Z^{(\ell+1)}
=\sigma\left(
\mathcal F^{-1}
\left(
\mathbf R_\ell(\mathbf k)\odot
\mathcal F(\mathbf Z^{(\ell)})(\mathbf k)
\right)
+\mathbf W_\ell\mathbf Z^{(\ell)}
\right),
$$

where only the configured low-frequency modes $\mathbf k$ are learned. The
$C$ output grids are flattened to $B\times Q\times C$ and trained with
$\mathcal L_{\mathrm{MSE}}$.

Current drawbacks:

- The adapter supports only one- or two-dimensional logical grids and requires
  a complete, consistently ordered grid output.
- Sparse observations are inserted without interpolation; very sparse rasters
  can be difficult for the operator to propagate.
- The forward path follows logical grid order and does not use arbitrary query
  coordinates. Spectral truncation can smooth discontinuities, and FFT-based
  layers impose a strong regular-grid/periodicity bias.
- It depends on the optional `neuraloperator` package.

## 4. Generative and flow reconstruction models

### 4.1 Conditional DiffusionPDE model

Let $\mathbf X_0$ denote the complete normalized target grid. A cosine schedule
defines cumulative signal levels $\overline\alpha_t$. Training samples a
timestep and Gaussian noise $\boldsymbol\epsilon\sim\mathcal N(0,\mathbf I)$:

$$
\mathbf X_t
=\sqrt{\overline\alpha_t}\,\mathbf X_0
+\sqrt{1-\overline\alpha_t}\,\boldsymbol\epsilon.
$$

The convolutional denoiser is conditioned on noisy state, sparse value raster,
mask raster, and normalized scalar time:

$$
\widehat{\boldsymbol\epsilon}_\theta
=\epsilon_\theta(\mathbf X_t,\mathbf V,\mathbf M,t),
\qquad
\mathcal L_{\mathrm{diff}}
=\mathbb E_{t,\boldsymbol\epsilon}
\left[
\lVert\widehat{\boldsymbol\epsilon}_\theta-oldsymbol\epsilon\rVert_2^2
\right].
$$

Reconstruction starts from Gaussian noise and uses a deterministic DDIM-style
subsequence. At step $t$,

$$
\widehat{\mathbf X}_0
=\frac{
\mathbf X_t-sqrt{1-\overline\alpha_t}
\widehat{\boldsymbol\epsilon}_\theta
}{\sqrt{\overline\alpha_t}},
$$

and for the next selected timestep $s<t$,

$$
\mathbf X_s
=\sqrt{\overline\alpha_s}\,\widehat{\mathbf X}_0
+\sqrt{1-\overline\alpha_s}
\widehat{\boldsymbol\epsilon}_\theta.
$$

After each update, observed grid entries are replaced exactly:

$$
\mathbf X_s\leftarrow
(\mathbf 1-\mathbf M)\odot\mathbf X_s
+\mathbf M\odot\mathbf V.
$$

Current drawbacks:

- The denoiser is a shallow three-convolution network rather than a multiscale
  U-Net; it has limited receptive-field and conditioning capacity.
- It supports only complete two-dimensional grids and uses a scalar time map,
  with no learned time embedding or attention.
- The training loss conditions on value/mask channels but does not explicitly
  clamp the noisy training state at sensors, whereas inference clamps after
  each DDIM update.
- `reconstruct` returns one draw and no stacked ensemble, so the advertised
  stochastic capability does not currently produce uncertainty samples for
  the common evaluator.

### 4.2 Two-stage latent flow matching

Stage 1 trains a convolutional autoencoder. With encoder $E_\psi$ and decoder
$D_\varphi$,

$$
\mathbf z_1=E_\psi(\mathbf X),
\qquad
\widehat{\mathbf X}=D_\varphi(\mathbf z_1),
\qquad
\mathcal L_{\mathrm{AE}}
=\lVert\widehat{\mathbf X}-\mathbf X\rVert_2^2.
$$

Stage 2 strictly loads and freezes the Stage-1 autoencoder. A condition encoder
maps $[\mathbf V,\mathbf M]$ to a latent-resolution tensor $\mathbf c$. With
$\mathbf z_0\sim\mathcal N(0,\mathbf I)$ and $t\sim\mathcal U(0,1)$, the
rectified bridge is

$$
\mathbf z_t=(1-t)\mathbf z_0+t\mathbf z_1,
\qquad
\mathbf v^\star=\mathbf z_1-\mathbf z_0,
$$

and the velocity objective is

$$
\mathcal L_{\mathrm{latent\ flow}}
=\mathbb E
\left[
\lVert
v_\theta(\mathbf z_t,\mathbf c,t)
-(\mathbf z_1-\mathbf z_0)
\rVert_2^2
\right].
$$

Sampling uses explicit Euler integration,

$$
\mathbf z^{(j+1)}
=\mathbf z^{(j)}+\Delta t\,
v_\theta(\mathbf z^{(j)},\mathbf c,t_j),
\qquad
\Delta t=\frac{1}{S},
$$

then decodes and exactly inserts observed values on the grid.

Current drawbacks:

- Stage 1 instantiates condition and velocity modules even though its loss
  trains only the autoencoder; those unused parameters remain in the model and
  optimizer state.
- Stage 2 freezes the autoencoder completely, so reconstruction bias in the
  latent representation cannot adapt to sparse conditioning.
- Stage-2 reporting computes the autoencoder MSE only as a detached diagnostic;
  it does not contribute to the optimized flow loss.
- The native sampler is first-order Euler on a two-dimensional grid. Downsample
  and transposed-convolution cropping can be restrictive for odd grid sizes.

### 4.3 PointCloudFFM

PointCloudFFM learns a rectified-flow velocity directly at query points. Its
source can be IID Gaussian noise or a smooth random-Fourier-function prior. For
the latter,

$$
\Phi_f(\mathbf q)
=\sqrt{\frac{2}{F}}
\cos(\boldsymbol\omega_f^\top\mathbf q+\beta_f),
\qquad
X_{0,bqc}=\sum_{f=1}^{F}\Phi_f(\mathbf q_{bq})W_{bfc},
$$

with $W_{bfc}\sim\mathcal N(0,1)$. Given target $\mathbf X_1$, training uses

$$
\mathbf X_t=(1-t)\mathbf X_0+t\mathbf X_1,
\qquad
\mathbf v^\star=\mathbf X_1-\mathbf X_0,
$$

and

$$
\mathcal L_{\mathrm{RF}}
=\frac{
\sum_{b,q,c}m^{\mathrm{qry}}_{bq}
\left(
v_\theta(\mathbf X_t,t,\mathcal O,\mathcal Q)_{bqc}
-(X_{1,bqc}-X_{0,bqc})
\right)^2
}{C\sum_{b,q}m^{\mathrm{qry}}_{bq}}.
$$

The native reconstruction solves

$$
\frac{d\mathbf X_t}{dt}
=v_\theta(\mathbf X_t,t,\mathcal O,\mathcal Q),
\qquad
\mathbf X_{t=0}\sim p_0,
$$

with Euler steps. The common post-training rollout can instead select Euler or
Heun while retaining gradients.

#### GL-RBF enhanced point backbone

Sensors are embedded from coordinate, value, and learned field identity. A
learned latent array cross-attends to sensors, passes through self-attention,
and receives sensor reinjection after each latent block. Queries read the
global latents. In parallel, the $K$ nearest valid sensors receive learned-width
Gaussian weights

$$
w_{bqk}
=\exp\left(-\frac{d_{bqk}^2}{2\sigma_\theta^2}\right),
$$

and their enriched tokens are aggregated as

$$
\mathbf l_{bq}
=\frac{\sum_{k=1}^{K}w_{bqk}\mathbf s'_{bqk}}
{\max\left(\sum_{k=1}^{K}w_{bqk},10^{-8}\right)}.
$$

The head fuses current-state/time features, query-to-latent readout, local RBF
features, and the global latent mean. Distance and query-attention work are
chunked to avoid allocating one global $Q\times M$ matrix.

#### FNO flow backbone

For a complete grid, the alternative velocity model applies an FNO to

$$
[\mathbf X_t,\mathbf V,\mathbf M,t]
\in\mathbb R^{B\times(3C+1)\times N_1\times\cdots\times N_d}
$$

and returns a grid velocity.

Current drawbacks:

- Rectified-flow accuracy depends on the chosen prior, number of integration
  steps, and the straight-line bridge assumption.
- The GL-RBF path still performs $\mathcal O(BQM)$ distance arithmetic even
  though chunking bounds peak memory; top-$K$ selection is piecewise smooth.
- Nearest sensors are selected across all fields. Field identity is learned in
  each token, but the neighbor search itself is not field-aware.
- Exact clamping in the native sampler uses index lookups and Python loops and
  is available only when sensor/query point IDs coincide.
- The FNO backbone requires complete regular-grid queries and inherits spectral
  truncation and optional-dependency constraints.

## 5. Physics-informed coordinate model

`PINNRegressor` is the coordinate MLP with an active case-owned physics
provider. It is constructed only by the direct-physics route. Its objective is

$$
\mathcal L_{\mathrm{direct}}
=\mathcal L_{\mathrm{MSE}}
+\lambda_{\mathrm{model\ physics}}
\mathcal L_{\mathrm{physics}}.
$$

This is a physics-regularized supervised regressor: the current implementation
does not sample independent, target-free collocation points and should not be
interpreted as a classical continuous PINN solely constrained by initial and
boundary data.

### Brusselator residual

Brusselator is the case with an active differentiable physics provider. In
physical units, its reaction terms are

$$
R_u=a-(b+1)u+u^2v,
\qquad
R_v=bu-u^2v.
$$

Using a periodic spectral Laplacian and paired finite-difference temporal
derivatives supplied by the dataset, the residuals are

$$
r_u=u_t-D_u\nabla^2u-R_u,
\qquad
r_v=v_t-D_v\nabla^2v-R_v.
$$

With detached RMS scales

$$
s_u=\max\left(\sqrt{\mathbb E[u_t^2]},10^{-6}\right),
\qquad
s_v=\max\left(\sqrt{\mathbb E[v_t^2]},10^{-6}\right),
$$

the provider returns

$$
\mathcal L_{\mathrm{physics}}
=w_u\mathbb E\left[\left(\frac{r_u}{s_u}\right)^2\right]
+w_v\mathbb E\left[\left(\frac{r_v}{s_v}\right)^2\right]
+w_+\mathbb E\left[\operatorname{ReLU}(-[u,v])^2\right].
$$

Current drawbacks:

- The residual requires a complete $192\times192$ grid in fixed $u,v$ order.
- $u_t$ and $v_t$ come from adjacent stored frames, not differentiation of a
  continuous time-coordinate network. Their truncation/noise error becomes
  part of the training signal.
- Residual normalization depends on target-derived temporal-derivative context.
- Only Brusselator currently exposes a trainable physics provider. Other cases
  expose evaluation diagnostics but not equation-loss training.

## 6. Base-training objectives

The base trainer calls each adapter's native `training_loss`:

$$
\mathcal L_{\mathrm{base}}=
\begin{cases}
\mathcal L_{\mathrm{MSE}},
&\text{coordinate MLP, MLP-RBF, DeepONet, Senseiver, GeoFNO},\\
\mathcal L_{\mathrm{diff}},
&\text{DiffusionPDE},\\
\mathcal L_{\mathrm{AE}},
&\text{latent flow Stage 1},\\
\mathcal L_{\mathrm{latent\ flow}},
&\text{latent flow Stage 2},\\
\mathcal L_{\mathrm{RF}},
&\text{PointCloudFFM}.
\end{cases}
$$

Thus “data loss” in later coherence post-training means the source adapter's
native loss. For a flow or diffusion model it is not endpoint reconstruction
MSE. Each optimizer step uses AdamW, optional gradient clipping, deterministic
sample indexing, and a step-dependent sensor seed. Checkpoints serialize model,
optimizer, normalization, data specification, resolved-config hash, and random
state required for supported resume paths.

## 7. Data-driven coherence model

### 7.1 Unified objective

The implemented data-driven families are

$$
\mathfrak F=
\{\text{global distribution},\text{cross-spectrum},\text{topology}\}.
$$

For enabled family $F$ and component $k$, let $w_F$ and $w_{Fk}$ be configured
nonnegative weights. A family produces

$$
\mathcal L_F
=\sum_{k\in\mathcal K_F}w_{Fk}\mathcal L_{Fk},
$$

and the combined coherence objective is

$$
\mathcal L_{\mathrm{coh}}
=\sum_{F\in\mathfrak F_{\mathrm{enabled}}}w_F\mathcal L_F.
$$

The post-training objective at epoch $e$ is

$$
\mathcal L_{\mathrm{post}}(e)
=\lambda_{\mathrm{data}}\mathcal L_{\mathrm{native}}
+\lambda_{\mathrm{coh}}(e)\mathcal L_{\mathrm{coh}}.
$$

After its configured start epoch, a linear warmup of length $E_w$ gives

$$
\lambda_{\mathrm{coh}}(e)
=\lambda_{\mathrm{coh}}^{\max}
\min\left(1,\frac{e-e_0+1}{E_w}\right),
$$

with the maximum weight used directly when $E_w=0$. If coherence is evaluated
only every $K$ optimizer steps, optional interval rescaling multiplies the
sampled coherence loss by $K$.

One differentiable reconstructed endpoint is reused by every enabled family in
an update. `self`, `mutual`, and `cross` name components inside a family; they
are not separate top-level coherence families. Observation consistency is a
reconstruction constraint, and PDE loss is case physics, so neither belongs in
$\mathfrak F$.

### 7.2 Global-distribution coherence

Let generated and reference empirical states be

$$
\mathbf X,\mathbf Y\in\mathbb R^{N\times C}.
$$

They must have equal point counts, finite values, and an explicitly resolved
field subset.

#### Marginal Wasserstein term

For each field $c$, sorting produces the exact equal-mass empirical
one-dimensional quadratic Wasserstein distance

$$
W_{2,c}^2(\mathbf X,\mathbf Y)
=\frac{1}{N}\sum_{i=1}^{N}
\left(X_{(i)c}-Y_{(i)c}\right)^2.
$$

`self.marginal_w2` is the configured weighted mean over fields.

#### Pairwise sliced Wasserstein term

For field pair $(i,j)$ and fixed unit direction
$\boldsymbol\theta_r\in\mathbb S^1$,

$$
p^{X}_{nr}=\boldsymbol\theta_r^\top[X_{ni},X_{nj}],
\qquad
p^{Y}_{nr}=\boldsymbol\theta_r^\top[Y_{ni},Y_{nj}],
$$

and

$$
\operatorname{SW}_2^2
=\frac{1}{R}\sum_{r=1}^{R}
\frac{1}{N}\sum_{n=1}^{N}
\left(p^X_{(n)r}-p^Y_{(n)r}\right)^2.
$$

`mutual.pairwise_swd` averages this value over configured pairs.

#### Joint top-tail sliced Wasserstein term

For the selected $C'$-field vector, fixed seeded Sobol-normal directions
$\boldsymbol\theta_r\in\mathbb S^{C'-1}$ produce per-direction distances
$d_r$. If $\mathcal T_\rho$ contains the largest fraction $\rho$ of those
distances, then

$$
\mathcal L_{\mathrm{joint\ top}K}
=\frac{1}{|\mathcal T_\rho|}\sum_{r\in\mathcal T_\rho}d_r.
$$

Optional axis directions ensure individual channels are represented.

Current drawbacks:

- Marginal matching discards all cross-field and spatial dependence.
- Pairwise slicing cannot represent interactions that appear only jointly
  among three or more fields.
- Joint projections approximate multivariate transport; sorting and top-tail
  selection are only piecewise differentiable.
- Every term ignores where values occur, so spatially permuted fields can have
  perfect distribution coherence.
- Equal empirical point counts are required; the current estimator does not
  solve unequal-weight or unequal-cardinality optimal transport.

### 7.3 Graph cross-spectrum coherence

All states must share identical coordinates and node ordering. A Gaussian
$k$-nearest-neighbor graph uses

$$
W_{ij}
=\exp\left(-\frac{\lVert\mathbf r_i-\mathbf r_j\rVert_2^2}{2\sigma_g^2}\right)
$$

on selected neighbor edges, followed by symmetrization. With degree matrix
$D_{ii}=\sum_jW_{ij}$, the normalized graph Laplacian is

$$
\mathbf L
=\mathbf I-\mathbf D^{-1/2}\mathbf W\mathbf D^{-1/2}.
$$

Its low-frequency eigenvectors $\mathbf U\in\mathbb R^{N\times K}$ define
graph Fourier coefficients

$$
A_{bkc}=\sum_{n=1}^{N}U_{nk}X_{bnc}.
$$

For an ensemble of $B$ states,

$$
P_i(k)=\frac{1}{B}\sum_b|A_{bki}|^2,
\qquad
P_{ij}(k)=\frac{1}{B}\sum_bA_{bki}A_{bkj}^{\ast}.
$$

The same-frequency magnitude-squared coherence is

$$
\gamma_{ij}^2(k)
=\frac{|P_{ij}(k)|^2}{P_i(k)P_j(k)+\varepsilon}.
$$

`same_frequency.magnitude_squared` compares generated and reference curves for
configured field pairs.

For graph-frequency band $\mathcal B_m$, per-state energy is

$$
E_{bmc}=\sum_{k\in\mathcal B_m}|A_{bkc}|^2.
$$

After centering across samples, cross-band coupling is

$$
Q_{mnij}
=\frac{
|\operatorname{Cov}(E_{mi},E_{nj})|^2
}{
\operatorname{Var}(E_{mi})\operatorname{Var}(E_{nj})+\varepsilon
}.
$$

`cross_frequency.band_energy_coupling` compares cells with $m\ne n$.
The optional `band_energy.log_power` also compares

$$
\log\left(\mathbb E_b[E_{bmc}]+\varepsilon\right),
$$

because normalized coherence alone discards absolute spectral power.

Current drawbacks:

- The graph, point order, and eigenbasis are geometry-specific; a changed
  coordinate set invalidates the artifact.
- Magnitude-squared coherence loses cross-spectrum phase and sign.
- Only retained eigenmodes are measured. Results depend on graph neighbor
  count, kernel scale, zero-mode policy, and band boundaries.
- Degenerate or nearly degenerate Laplacian eigenvalues can make individual
  eigenvectors basis-dependent even when their invariant subspace is stable.
- These are ensemble estimators: same-frequency mode requires $B\ge2$ and
  cross-frequency mode requires $B\ge3$, while small batches yield noisy
  covariance estimates.
- Graph construction/eigendecomposition is preprocessing overhead, and the
  stored basis scales with $NK$.

### 7.4 Topology coherence

Topology v1 projects coordinates to two dimensions and precomputes an
inverse-distance linear raster map

$$
\mathcal R:\mathbb R^{B\times N\times C}
\longrightarrow\mathbb R^{B\times C\times H\times W}.
$$

Periodic cases tile coordinate images across seams before interpolation.
Optional Gaussian smoothing is applied on the raster.

For threshold $\tau$, a superlevel filtration uses

$$
K_\tau^{+}=\{p:g(p)\ge\tau\},
$$

and a sublevel filtration uses

$$
K_\tau^{-}=\{p:g(p)\le\tau\}.
$$

A detached union-find ordering obtains exact hard-forward connected-component
births and deaths for $\beta_0(\tau)$. On the cubical grid,

$$
\chi(\tau)=V(\tau)-E(\tau)+F(\tau),
$$

so

$$
\beta_1(\tau)=\beta_0(\tau)-\chi(\tau)+\beta_2(\tau),
$$

with the periodic full-domain $\beta_2$ correction where applicable.
Straight-through sigmoid indicators preserve hard counts in the forward pass
while supplying approximate gradients.

For selected dimensions $d\in\mathcal D$ and thresholds $\tau_j$,
`self.betti_curves` uses

$$
\mathcal L_{\mathrm{Betti}}
=\frac{1}{|\mathcal D|J}
\sum_{d\in\mathcal D}\sum_{j=1}^{J}
\left(\beta_d^{X}(\tau_j)-\beta_d^{Y}(\tau_j)\right)^2.
$$

For field pair $(i,j)$, `mutual.fibered_betti_curves` first standardizes both
axes with detached reference moments. For a positive-slope line with origin
$(s_0,t_0)$ and direction $(v_1,v_2)$, it reduces the two-parameter filtration
to

$$
h(p)
=\min\left(
\frac{g_i(p)-s_0}{v_1},
\frac{g_j(p)-t_0}{v_2}
\right),
$$

then compares the induced $\beta_0$ and $\beta_1$ curves.

Current drawbacks:

- Rasterization can change topology, especially on irregular clouds, coarse
  grids, boundaries, and sparsely supported regions.
- The implementation is two-dimensional and measures only configured
  $\beta_0/\beta_1$ curves; it does not localize features or compare full
  persistence diagrams.
- Union-find order is detached and straight-through indicators have biased
  gradients. A loss decrease need not correspond to the true derivative of
  hard topology.
- Thresholds are reference-quantile dependent, and mutual topology samples
  only a configured line fan through a two-parameter filtration.
- Exact topology calculations over many thresholds, fields, and line slices
  can be expensive.

### 7.5 Reference policies and target leakage boundary

With `target_use: training_reference`, a frozen empirical bank is fitted only
from the training split. It records dataset fingerprint, split policy, sample
IDs, point indices, seed, field order, normalization, and units. Geometry-aware
families require a fixed shared query set and verify its point indices against
the bank.

With `target_use: paired_supervised`, the reference is the dense target of the
current training sample. This is supervised structural regularization, not
target-free physical refinement.

In both modes, dense targets are removed from the batch passed into the model's
differentiable rollout:

$$
\widehat{\mathbf X}
=f_\theta(\mathcal O,\mathcal Q),
\qquad
\mathbf X_{\mathrm{ref}}
\notin \operatorname{inputs}(f_\theta).
$$

The reference is supplied only after reconstruction to compute
$\mathcal L_{\mathrm{coh}}$. Validation and test states are never used to fit a
training reference bank.

## 8. Differentiable reconstruction and observation consistency

Post-training dispatches by capability rather than model name:

- PointCloudFFM and the legacy combustion adapter expose source sampling and
  velocity hooks for differentiable Euler/Heun integration.
- Direct regressors and GeoFNO use their direct prediction.
- DiffusionPDE uses differentiable DDIM.
- Latent flow Stage 2 uses differentiable latent integration and decoding.
- Latent flow Stage 1 cannot be a sparse-reconstruction source.
- PINN belongs to direct physics training and has no plain base source.

For a flow state $\mathbf x_t$, velocity $\mathbf v_t$, and remaining time
$r=1-t$, the estimated endpoint is

$$
\mathbf e_t=\mathbf x_t+r\mathbf v_t.
$$

`endpoint_smooth` constructs Gaussian sensor value and confidence maps
$\mathbf V_s$ and $\mathbf M_s$. With strength $\eta$ and schedule power $p$,

$$
\mathbf G_t=\eta r^p\mathbf M_s,
\qquad
\widetilde{\mathbf e}_t
=(\mathbf 1-\mathbf G_t)\odot\mathbf e_t
+\mathbf G_t\odot\mathbf V_s,
$$

and replaces the velocity by

$$
\widetilde{\mathbf v}_t
=\frac{\widetilde{\mathbf e}_t-\mathbf x_t}{r}.
$$

For a non-flow endpoint $\widehat{\mathbf X}$, the smooth blend is

$$
\widehat{\mathbf X}'
=(\mathbf 1-\mathbf B)\odot\widehat{\mathbf X}
+\mathbf B\odot\mathbf V_s,
\qquad
\mathbf B=\operatorname{clip}(\eta\mathbf M_s,0,1).
$$

`hard` and `endpoint` modes operate only where exact sensor/query index matches
exist. Optional final clamping replaces those entries with measured values.

Current drawbacks:

- Gaussian maps use one coordinate-space bandwidth and are built under
  `no_grad`; their construction and bandwidth are not learned.
- Smooth blending can bias unobserved neighbors toward a kernel interpolant and
  can be physically inappropriate across discontinuities or non-Euclidean
  boundaries.
- Exact clamping is index-based, so it does not apply when sensors and queries
  represent coincident coordinates with unrelated point IDs.

## 9. Physics post-training

Physics post-training strictly loads a native base checkpoint, selects either
the full model or named modules, and creates a child run. Its differentiable
endpoint objective is

$$
\mathcal L_{\mathrm{physics\ post}}
=\lambda_{\mathrm{retain}}
\mathbb E\left[\lVert\widehat{\mathbf X}-\mathbf X\rVert_2^2\right]
+\lambda_{\mathrm{physics}}
\mathcal L_{\mathrm{physics}}(\widehat{\mathbf X}).
$$

Unlike coherence post-training, the retention term here is always endpoint
MSE, not the adapter's native diffusion or flow objective. The source run is
hashed before and after refinement and is never overwritten. A post-training
config selects either data-driven coherence or physics; it cannot enable both
within one child run.

Current drawbacks:

- Physics and data-driven coherence cannot yet be jointly optimized in one
  post-training configuration.
- Physics post-training resume is not implemented.
- For stochastic models, one seeded endpoint per update estimates the loss and
  can have high gradient variance.

## 10. Gradient combination

For post-training objectives, define weighted flattened gradients

$$
\mathbf g_d=\nabla_\theta(\lambda_d\mathcal L_d),
\qquad
\mathbf g_c=\nabla_\theta(\lambda_c\mathcal L_c).
$$

The default update uses

$$
\mathbf g=\mathbf g_d+\mathbf g_c.
$$

The trainer records

$$
\cos\varphi
=\frac{\mathbf g_d^\top\mathbf g_c}
{\lVert\mathbf g_d\rVert_2\lVert\mathbf g_c\rVert_2+\varepsilon}.
$$

When optional ConFIG mode is selected and $\cos\varphi<0$, the
`conflictfree` package proposes a combined gradient. It is accepted only if it
is finite and is a descent direction for both component gradients:

$$
\mathbf g^\top\mathbf g_d>0,
\qquad
\mathbf g^\top\mathbf g_c>0.
$$

Otherwise the trainer falls back according to configuration. Complex
parameters are flattened as real/imaginary pairs and restored without changing
their dtype.

Current drawbacks:

- Weighted sums are sensitive to loss units, scaling, and schedule choices.
- Computing two separate flattened gradients increases memory and backward
  cost.
- The two-objective combiner treats all coherence families as one aggregate;
  conflicts among individual families or components are not resolved
  separately.

## 11. Evaluation definitions

For valid query entries, normalized reconstruction MSE is

$$
\operatorname{MSE}_{\mathrm{norm}}
=\frac{1}{C\sum_{b,q}m^{\mathrm{qry}}_{bq}}
\sum_{b,q,c}m^{\mathrm{qry}}_{bq}
\left(\widehat X_{bqc}-\widetilde X_{bqc}\right)^2.
$$

Physical-unit MSE decodes both tensors before comparison:

$$
\operatorname{MSE}_{\mathrm{phys}}
=\frac{1}{BQC}
\sum_{b,q,c}
\left[s_c(\widehat X_{bqc}-\widetilde X_{bqc})\right]^2.
$$

The evaluator also reports per-field MSE, observed-entry MSE,
unobserved-entry MSE, inference time, and peak CUDA memory. If a model returns
an ensemble $\{\widehat{\mathbf X}^{(s)}\}_{s=1}^{S}$, it reports

$$
\overline{\mathbf X}
=\frac{1}{S}\sum_s\widehat{\mathbf X}^{(s)},
\qquad
\operatorname{Var}(\widehat{\mathbf X})
=\frac{1}{S}\sum_s
\left(\widehat{\mathbf X}^{(s)}-\overline{\mathbf X}\right)^2.
$$

Case-owned diagnostics are evaluated in addition to these shared metrics.
Brusselator reports differentiable PDE residuals; Kolmogorov, KS, turbulent
combustion, and mass transport currently provide diagnostic-only physical
checks. Reproducible evaluation stores the checkpoint hash, resolved config
hash, dataset fingerprint, sensor manifest, query indices, sample IDs, metrics,
and portable plotting arrays.

## 12. Cross-cutting limitations

- Most base objectives are pointwise or noise/velocity regression objectives;
  none alone guarantees conservation, correct topology, or correct multiscale
  cross-field statistics.
- Data-driven coherence matches selected empirical descriptors, not governing
  equations. A low coherence loss is necessary only relative to the chosen
  descriptor and reference population, not proof of physical validity.
- Physics losses are only as reliable as their discretization, boundary model,
  normalization, parameters, and temporal context.
- Fixed normalization, fixed sensor protocols, and fixed geometry artifacts can
  limit transfer to shifted regimes or meshes.
- Current uncertainty reporting is incomplete because standard reconstruction
  adapters generally return a single endpoint even when their sampling process
  is stochastic.
- Fair model comparison requires the same dataset split, normalization, sensor
  manifest, query indices, generation seed/steps, and reference artifacts.
