# DiffusionPDE source training

DiffusionPDE source training uses conditional EDM. Import `configs/models/diffusion_pde.yaml`; the existing `_edm`-suffixed filenames resolve to the same model and case settings. The supplied case launch profiles use 32-step Heun reconstruction and EMA evaluation.

The model supports any ordered field list and any observed subset on a complete two-dimensional logical grid. It uses point observations. Keep spatial query subsampling disabled, and ensure the grid order, field order and TRAIN normalization agree across targets and observations. The implementation applies EDM equations to a sparse-conditioned U-Net; case-specific PDE guidance is outside this generic model.

## Configure your case

Create `cases/<case>/configs/base/diffusion_pde.yaml` after registering the dataset and sensor protocol:

```yaml
defaults:
  - ../../../../configs/defaults/base_training.yaml
  - ../dataset.yaml
  - ../sensors/<protocol>.yaml
  - ../../../../configs/models/diffusion_pde.yaml

stage: base_training
case: <case>
dataset: {normalization: mean_std}
model: {train_microbatch_size: 4}
optimization: {epochs: 200, batch_size: 4, lr: 1.0e-4, weight_decay: 1.0e-6, grad_clip: 1.0}
evaluation:
  split: validation
  max_samples: 16
  query_points: null
  generation_steps: 32
  seed: 2027
  preview:
    enabled: true
    split: validation
    sample_index: 0
    query_points: null
    generation_steps: 32
    loss_every_epochs: 10
    reconstruct_every_epochs: 100
    seed: 2027
  reconstruction_panel:
    sample_indices: [0]
    seeds: [2027, 2028]
    generation_steps: 32
checkpointing:
  enabled: true
  every_epochs: 10
  validation_every_epochs: 50
  selection_metric: reconstruction_mse
output: {experiment_name: diffusion_pde}
```

Set `epochs` to the intended training horizon. Choose the effective batch and microbatch sizes for available memory; microbatches accumulate complete snapshots into one optimizer update. The shared runtime refreshes loss curves every 10 completed epochs. The shared model profile uses FP32; `model.precision: bf16` enables CUDA BF16 autocast in the U-Net while keeping loss arithmetic, parameters and optimizer state in FP32.

Use TRAIN mean/std statistics stored in the dataset or supply `dataset.normalization_stats_path` for a verified TRAIN statistics artifact. Set `evaluation.max_samples` to approximately 10% of validation conditions, and choose the fixed panel indices from those conditions. The one-index template is a starting point; use several matched conditions and seeds for model selection.

```bash
python cases/<case>/run.py validate --config configs/base/diffusion_pde.yaml
python cases/<case>/run.py train-base --config configs/base/diffusion_pde.yaml
```

The complete turbulent-combustion profile is `cases/turbulent_combustion/configs/base/diffusion_pde.yaml`.

## Objective and sampling

For normalized target $y$, sample Gaussian noise $\epsilon$ and $\log\sigma\sim\mathcal N(P_{mean},P_{std}^2)$, then form $x=y+\sigma\epsilon$. With sparse values $V$, mask $M$ and data scale $s_d$, the denoiser estimates the clean field:

$$D_\theta(x,\sigma,V,M)=c_{skip}x+c_{out}F_\theta(c_{in}x,\log(\sigma)/4,V/s_d,M),$$

$$c_{skip}=\frac{s_d^2}{\sigma^2+s_d^2},\qquad c_{out}=\frac{\sigma s_d}{\sqrt{\sigma^2+s_d^2}},\qquad c_{in}=\frac{1}{\sqrt{\sigma^2+s_d^2}}.$$

Replace observed clean entries by $V$. For sample $b$, let $U_b=1-M_b$. The training objective is

$$\mathcal L=\frac1B\sum_b\frac{\sum_i U_{bi}\,w(\sigma_b)\,[D_{bi}-y_{bi}]^2}{\max(1,\sum_i U_{bi})},\qquad w(\sigma)=\frac{\sigma^2+s_d^2}{(\sigma s_d)^2}.$$

The shared profile uses `sigma_data: 1`, `p_mean: -0.5`, `p_std: 1.2` and `high_noise_fraction: 0.2`. The last setting replaces 20% of sampled noise scales by `sigma_max: 80`, exposing training to inputs close to the reconstruction prior. This setting is part of the default conditional recipe.

Reconstruction uses a rho-spaced schedule from $\sigma_{max}=80$ to $\sigma_{min}=0.002$, followed by zero. For $d=(x-D_\theta(x,\sigma))/\sigma$, an Euler prediction is $x'=x+(\sigma'-\sigma)d$; Heun correction averages this direction with the direction evaluated at $(x',\sigma')$. The last update ends at zero noise without a second denoiser call. Thus 32 steps use 63 calls. A final projection matches observed point values exactly.

The U-Net accepts arbitrary field counts, injects value/mask context at every encoder scale, and pads/crops odd grid sizes. Its default widths are `[64,128,256,512]` with two residual blocks per scale and attention on coarse levels. No field names or combustion equations are hard-coded.

## Evaluate and resume

The training curve measures weighted denoising error from a noisy true field. Reconstruction starts from random noise and is assessed separately. Inspect `evaluation/training_preview/latest_metrics.json` and the fixed panel's reconstruction MSE alongside `loss_history.png`. The fixed panel selects `best.pt`; `last.pt` stores recovery state. Compare normalized and physical errors per field, since a pooled physical MSE mixes units and scales.

The project evaluator restores model-owned EMA state and applies the configured evaluation weights:

```bash
python cases/<case>/run.py evaluate-run --run runs/<experiment>/<run-id> \
  --checkpoint best --split validation --max-samples 16 --generation-steps 32 \
  --weight-selection configured --report-name diffusion_check
```

Choose `--max-samples` to cover approximately 10% of validation conditions. Keep matched sample IDs, observations and seeds when comparing models. `reconstruct` uses observations without reading the hidden target.

Resume with the run's saved resolved configuration, preserving its normalization, optimizer, EMA and random state:

```bash
python cases/<case>/run.py train-base --config /path/to/run/resolved_config.yaml \
  --resume /path/to/run
```

Archived DDPM runs retain their original loading and resume contract. Their weights are not converted to EDM; new SOURCE training uses the profile above.

## Mathematical provenance

EDM preconditioning, weighted denoising and Heun integration follow the mathematical references in [DiffusionPDE](https://github.com/generativepde/DiffusionPDE/tree/1e2bc8b9e312f3a936630a30d2f49aedabf0cea7) and NVIDIA EDM. The conditional U-Net is independently implemented. See [provenance](provenance.md) for the pinned reference and implementation scope.
