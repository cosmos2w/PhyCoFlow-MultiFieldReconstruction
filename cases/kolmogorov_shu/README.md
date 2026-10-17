# Kolmogorov Shu CQ single-field correction

This case copies the B recipe from FlowReconBenchmark commit `7192fd1d109c1eeaec29085d013483182ad2c657` onto its pinned upstream source commit `1b29b55e82c0b010d940365663a017d28f13a1e6`. This isolated checkout leaves FlowReconBenchmark, its pinned submodule worktree, and the separate DMF-Gen peer-review repository unchanged.

The model correction skips the measurement/support LayerNorm only when `n_fields == 1`. RBF weights sum to one for a valid single-field neighborhood, so the shortcut is `[measurement, 1]`; LayerNorm across that pair nearly turns it into a sign comparison against one. The measurement has already been normalized with training-only dataset statistics. Passing this pair directly preserves its amplitude. Multi-field execution still applies the existing LayerNorm. The constructor and all state keys/shapes remain unchanged, although the old LayerNorm parameters are intentionally unused on the corrected single-field path. Strict loading is possible, but predictions from old single-field checkpoints intentionally change under the corrected code.

The fixed config inherits the original B recipe unchanged except for a distinct experiment name. It retains width 296, condition width 296, 64 latent tokens, 3 blocks, 4 heads, CQ width 128/rank 64, batch 128, 4096 queries with 2048-query microbatches, 65–6554 training sensors, 4000 epochs, constant AdamW learning rate 0.0001, EMA 0.999, seed 42, KeOps, the original checkpoint/preview schedule, and `cuda:0`. Start fresh rather than resuming the earlier DMF-Gen local run, which used a different configuration.

## Dataset preparation

The source package is `/data/data_transfer/forLinzheng/kolmogorov2d_shu`, also used by `/home/wanglz/Desktop/src/PhyCoFlow/DMF-Gen/runs/kolmogorov_cq_1pct_b64_q8192_gpu0_20261010T193710Z`. Its float32 `vorticity.npy` is `(40, 80, 256, 256)` and is already temporally strided. Do not apply stride four again. The preparation script verifies the package and creates a small HDF5 virtual view for B's existing loader. The view reads vorticity and coordinate values directly from the original NPY files using external storage and virtual mappings; it does not copy their numerical payloads. Keep the original package at its existing path while using this view.

From `/home/wanglz/Desktop/src/PhyCoFlow-MFR-kolm-cq-fix`:

```bash
/home/wanglz/miniconda3/envs/phycoflow_env/bin/python cases/kolmogorov_shu/prepare_dataset_view.py \
  --source /data/data_transfer/forLinzheng/kolmogorov2d_shu \
  --out datasets/kolmogorov_shu/kolmogorov_shu.h5
```

The script refuses to overwrite an existing output. Training uses the same 32 trajectories and 2560 frames as the earlier local run. B's native validation uses all 80 frames from the first four held-out trajectories, and native test uses all 80 frames from the remaining four. The earlier local interface instead divides odd/even frames across all eight held-out trajectories. This preparation preserves B's native behavior; benchmark checkpoint/NFE selection on the separate odd-frame TUNE panel remains an external evaluation step, not a change to native training validation. Do not compare the two native validation scores as if their samples were identical.

## Validate without training

```bash
cd /home/wanglz/Desktop/src/PhyCoFlow-MFR-kolm-cq-fix/cases/kolmogorov_shu
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/wanglz/miniconda3/envs/phycoflow_env/bin/python run.py validate \
  --config configs/base/gl_rbf_cq_w296_range_singlefield_fix.yaml
```

## Future training command — not launched during preparation

```bash
cd /home/wanglz/Desktop/src/PhyCoFlow-MFR-kolm-cq-fix/cases/kolmogorov_shu
CUDA_VISIBLE_DEVICES=0 /home/wanglz/miniconda3/envs/phycoflow_env/bin/python run.py train-base \
  --config configs/base/gl_rbf_cq_w296_range_singlefield_fix.yaml
```

The upstream trainer writes beneath `cases/kolmogorov_shu/runs/kolm_shu_gl_rbf_cq_w296_range_singlefield_fix/`. The original DMF-Gen run is not resumed or modified. Unit forward/backward checks establish that the shortcut keeps amplitude and that cached/microbatched execution agrees; they do not establish a lower trained reconstruction error or verify GPU memory fit for the full B recipe.
