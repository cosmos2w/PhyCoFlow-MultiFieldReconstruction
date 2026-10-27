# Conditional EDM reconstruction and fresh-training recipe

The historical combustion `tc_diffusion_pde_5000ep/20260829T140426Z_e2c328ff` run used a local epsilon-prediction DDPM and DDIM sampler. Its full-field reconstruction failed even when increasing sampling to 1000 steps. The opt-in `formulation: edm` path replaces that poorly scaled parameterization for fresh training while retaining the project's common sparse-reconstruction and post-training interfaces. Old configurations still select DDPM, and old checkpoints keep their strict state namespace. Never resume a DDPM checkpoint into EDM.

## Source and scope

The source reference is Huang et al., [DiffusionPDE: Generative PDE-Solving Under Partial Observation, NeurIPS 2024](https://arxiv.org/abs/2406.17763), with accessible official repository [generativepde/DiffusionPDE](https://github.com/generativepde/DiffusionPDE/tree/1e2bc8b9e312f3a936630a30d2f49aedabf0cea7). The reviewed commit is `1e2bc8b9e312f3a936630a30d2f49aedabf0cea7`. Its [EDM preconditioning](https://github.com/generativepde/DiffusionPDE/blob/1e2bc8b9e312f3a936630a30d2f49aedabf0cea7/training/networks.py), [weighted denoising loss](https://github.com/generativepde/DiffusionPDE/blob/1e2bc8b9e312f3a936630a30d2f49aedabf0cea7/training/loss.py), and [guided Heun sampler](https://github.com/generativepde/DiffusionPDE/blob/1e2bc8b9e312f3a936630a30d2f49aedabf0cea7/scripts/generate_darcy.py) informed the local equations. No upstream source or unsafe pickle is imported.

The original implementation learns an unconditional coefficient/solution prior and guides inference with observation and PDE gradients. This implementation learns a conditional prior using sparse values and masks. It supports variable point-sensor counts and arbitrary observed field subsets on a shared complete two-dimensional logical grid. It does not assume T-only input, five combustion fields, a particular sensor count, or a particular case. Block averages are rejected rather than interpreted as point values. Point-query subsampling and unstructured off-grid sensors are not supported by this grid adapter. Case physics remains outside the generic model and can use its differentiable reconstruction hook through the existing post-training interface.

## Objective, architecture and sampling

For normalized complete fields $y$, sample Gaussian noise $\epsilon$ and $\log\sigma\sim\mathcal N(P_{mean},P_{std}^2)$. The input is $x=y+\sigma\epsilon$. With sparse values $V$, masks $M$, and data scale $s_d$, the denoiser is

$$D_\theta(x,\sigma,V,M)=c_{skip}x+c_{out}F_\theta(c_{in}x,\log(\sigma)/4,V/s_d,M),$$

$$c_{skip}=\frac{s_d^2}{\sigma^2+s_d^2},\qquad c_{out}=\frac{\sigma s_d}{\sqrt{\sigma^2+s_d^2}},\qquad c_{in}=\frac1{\sqrt{\sigma^2+s_d^2}}.$$

Known entries of this clean-field estimate are replaced by their observed values. The objective averages $w(\sigma)(D-y)^2$ over unobserved entries separately for each sample, then averages samples, with $w(\sigma)=(\sigma^2+s_d^2)/(\sigma s_d)^2$. Thus sensor density does not alter sample weighting, and a fully observed sample has zero unknown-entry loss. Diagnostics retain the weighted EDM objective and the unweighted clean-field error as separate components.

The reusable defaults use `sigma_data: 1.0` and `p_mean: -0.5` for per-field mean/std data. Official EDM defaults use `sigma_data: 0.5` and `p_mean: -1.2`; the approximately log(2) shift preserves comparable noise relative to the data scale. `p_std: 1.2`, `rho: 7`, and sigma endpoints `80 → 0.002` follow upstream choices. `high_noise_fraction: 0.2` is an explicit conditional-task adaptation: 20% of training examples use sigma_max to require reconstruction from near-prior inputs. Setting it to zero restores log-normal-only noise sampling. This mixture is a proposed design choice, not an isolated, mature ablation result.

The local U-Net keeps the previous channel widths `[64,128,256,512]`, two residual blocks per resolution, time embedding 256, and attention at levels 2 and 3. Sparse value/mask context enters each encoder scale; masked pooling divides by observed support so rare sensors do not vanish through averaging. Residual branches use small output initialization and residual scaling; the final residual projection starts at zero. Odd grids are padded to multiples of eight and cropped back, so the combustion grid uses `104×408` internally and returns `100×403`. Learned convolutions follow nearest-neighbor upsampling. This architecture is independently implemented and is not the official DDPM++ SongUNet.

Sampling follows a rho-spaced sigma schedule with Euler prediction and Heun correction, requiring `2*steps-1` denoiser calls. The proposed 32 steps therefore mean 63 calls, not 32. Observed noisy entries start as `V + sigma_max*noise`; projecting the clean estimate provides consistent observed-state evolution without inserting clean values into a high-noise state. A final point-observation projection enforces exact data consistency. `reconstruct` never reads target values; `differentiable_reconstruct` retains gradients for downstream post-training.

## Training and validation contracts

The existing base trainer still performs one AdamW update per effective batch. Model-owned `training_backward` draws a single effective batch's noise and sigma, accumulates complete-sample microbatches with exact last-chunk weighting, and returns detached diagnostics. No point queries are dropped. Clipping occurs once after accumulation; BF16 autocast applies to the U-Net, while sigma arithmetic, loss accumulation, parameters, gradients and Adam state remain FP32. FP16/GradScaler is not introduced. This microbatch execution applies to base training; the existing combined-objective post-training path uses the differentiable loss directly and does not inherit the base-training memory bound.

EMA uses the existing checkpointable `ModelEMA` implementation. It is created on the model device after the first optimizer update and uses a short warmup before approaching decay 0.999. It is saved under `training_aux_state`; all project evaluation entrypoints and the new panel load this state and use EMA through the existing evaluation context. Evaluation temporarily swaps weights and restores live training weights. Resuming restores optimizer, RNG, EMA update count, normalization identity, and full history.

`evaluation.reconstruction_panel` defines fixed split-relative validation indices and initial-noise seeds. The combustion recipe evaluates frames `[0,250,500,750]` and seeds `[2027,2028]`, eight reconstructions total. Sensor manifests are deterministic, saved per frame, and verified on resume. Targets are removed before calling `reconstruct`. Pooled normalized MSE selects `best.pt`; per-field normalized and physical metrics and per-frame/seed results remain available. On intervening epochs a one-frame preview cannot replace the configured panel selection metric. `last.pt` remains the recovery state.

Noise validation and reconstruction validation are different evidence. `loss_history.{png,pdf}` records training/native denoising objectives. `evaluation/reconstruction_panel/reconstruction_mse.{png,pdf}` independently records generation error and retains all panel history across resume. Reconstruction previews preserve physical target/prediction/error panels. No TEST data is used for checkpoint selection.

## Proposed combustion recipe

Use [diffusion_pde_edm_5000ep.yaml](../cases/turbulent_combustion/configs/base/diffusion_pde_edm_5000ep.yaml). The long run is a proposal; bounded qualification is not completion of this recipe.

| Setting | Proposed value | Relation to mature project runs |
|---|---|---|
| Data and split | Existing combustion HDF5, chronological train/validation, verified train mean/std | Same dataset and statistics |
| Observations | Random T-only, 192–384 points | Same variable-count protocol as Senseiver/GL-RBF-CQ |
| Horizon | 5000 epochs from scratch | Same proposed epoch horizon |
| Effective batch | 128 | Same as mature Senseiver/GL-RBF-CQ profiles; old diffusion used 64 |
| GPU execution | BF16, two microbatches of 64 on physical GPU2 | Measured full-grid optimizer probes; no query subsampling |
| Optimizer | AdamW, LR 1e-4, weight decay 1e-6, clip 1 | Same mature project baseline settings |
| Reproducibility/loading | Seed 42, deterministic cuDNN, four workers, async CPU | Same project conventions |
| Recovery and selection | Save epoch 1, every 10 epochs; fixed reconstruction panel every 10 epochs | Adds generation-based selection and EMA |
| Milestones | 100, 500, 1000, 2000, 5000 epochs | Same retained milestones |
| Sampling | 32-step Heun during monitoring | Must be qualified for fidelity; not copied from other models' 4-step setting |

Do not enlarge the U-Net to fill VRAM. Adjust complete-sample microbatch execution instead. GPU2 optimizer probes at effective batch 128 measured FP32 microbatch 16/32/48/56 peak allocations of 13.01/25.68/37.76/43.81 GiB; BF16 microbatch 48/64 measured 31.51/40.93 GiB. BF16 microbatch 64 reserved 46.59 GiB on the 48 GiB device in its initial probe. These are optimizer/EMA probes rather than end-to-end timing claims; complete-epoch qualification must determine sustained memory and monitoring overhead. Use the measured FP32 microbatch-32 or -48 configuration if a full-precision comparison is required.

Run from the repository root after confirming index 2 has UUID `GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39` and is free. Exposing that verified index makes logical `cuda:0` below physical GPU2:

```bash
nvidia-smi -i 2 --query-gpu=index,uuid,memory.used --format=csv
CUDA_VISIBLE_DEVICES=2 \
  /home/wanglz/miniconda3/envs/phycoflow_env/bin/python \
  cases/turbulent_combustion/run.py train-base \
  --config configs/base/diffusion_pde_edm_5000ep.yaml \
  --override runtime.device=cuda:0 \
  --max-steps 6300
```

The first command caps execution at 100 epochs (63 optimizer updates per epoch for the current 8000-frame split and batch 128), retaining the proposed 5000-epoch configuration. Review panel and fieldwise curves at that point before any continuation. Resume only the new EDM run using the same resolved settings, omit `--max-steps`, and add `--resume /absolute/path/to/the/new/run` to continue toward its configured horizon. Do not pass the historical DDPM run as the resume path. Formal readiness requires complete generation-validation history, per-field fidelity across the validation panel, seed stability, and convergence evidence; a few successful epochs establish execution and resume functionality only.

## Bounded GPU2 qualification, 2026-10-27

The fresh three-epoch test completed 189 optimizer updates, with a trainer restart after epoch 2. EMA update counts restored from 126 and reached 189. All three training and reconstruction-validation epochs remain in their histories and PDF/PNG curves. Effective batch 128, BF16 and microbatch 64 ran without OOM. The model has 50,164,677 parameters; the largest measured training allocation across the test was 43.06 GiB. Total qualification wall time was 551.2 seconds, including validation, plotting, checkpoints, endpoint integration checks and the restart; native training epoch times were approximately 110 seconds. This does not establish an end-to-end speedup over another architecture or objective.

| Epoch | Mean sampled EDM weighted training loss | Fixed 4-frame × 2-seed reconstruction MSE |
|---|---:|---:|
| 1 | 0.7681 | 1.6764 |
| 2 | 0.6142 | 1.4649 |
| 3 | 0.5207 | 1.1960 |

For a separate epoch-3, frame-8000/seed-2027 sensitivity check, conditioned FP32 reconstruction MSE was 1.1356 and no-condition MSE was 2.2803. Removing all conditions is outside the training protocol, so this demonstrates condition dependence rather than a controlled architectural ablation. BF16 MSE for that same generated example was 1.1366, and mean squared BF16/FP32 prediction difference was 1.82e-5. This is a limited numerical check, not a general precision-fidelity guarantee. The training-mean predictor's MSE on that frame was 0.9629: generation quality is not yet qualified, and the short run cannot justify a mature accuracy claim. The 100-epoch gate above is therefore part of the proposed recipe.

Legacy epoch-500 DDPM weights still strict-load and reproduce the archived physical prediction with maximum absolute difference zero. Nine original-run checkpoints, histories, configuration and preview artifacts were hash-verified unchanged. No TEST evaluation or 5000-epoch training was performed. Focused numerical, interface, configuration, checkpoint, monitoring and telemetry tests passed (124 tests in total).

The bounded resume also exposed an existing telemetry bug that overwrote pre-resume timing history. The implementation now reloads prior telemetry and discards entries newer than the recovery checkpoint; a focused resume regression passes. The first two qualification epoch summaries were captured during live inspection, while their original per-step telemetry was overwritten before the fix. Training/validation loss and reconstruction histories are complete; the performance trace for this initial qualification has that stated limitation. The out-of-checkout test also lacks the native environment's Git capture; source hashes and its base-commit identity were recorded separately with local artifacts.
