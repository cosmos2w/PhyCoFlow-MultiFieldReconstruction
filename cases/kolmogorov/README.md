# Kolmogorov Flow Case

This formal initial benchmark reconstructs periodic `u,v,p` snapshots on a
verified `256×256` grid. The primary protocol observes sparse colocated `u,v`
velocity values and evaluates pressure as an entirely unmeasured field.
Single-component `u`-only and `v`-only protocols expose directional inference
asymmetry. Auxiliary vorticity is available to later physics and evaluation
providers, but is not an implicit model input.

Validate and perform a deliberately truncated integration update from this
directory:

```bash
python run.py validate --config configs/dataset.yaml
CUDA_VISIBLE_DEVICES=1 python run.py train-base \
  --config configs/base/pointcloud_ffm.yaml --max-steps 1
```

Point models train against deterministic 4,096-point query subsets by default;
their scientific evaluation must reconstruct the full field in chunks. Grid
models use the full verified raster. A `--max-steps` run validates mechanics
only and is never a performance result.

The current data holds Reynolds number, forcing amplitude, and forcing
wavenumber fixed, so experiments test cross-trajectory reconstruction rather
than condition generalization. A future `physics.py` should implement periodic
incompressible Navier–Stokes residuals, divergence, the forcing, and the
zero-mean pressure gauge before PINN or PDE-informed training is enabled.
