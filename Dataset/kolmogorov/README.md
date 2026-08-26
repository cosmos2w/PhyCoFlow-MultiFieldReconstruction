# Kolmogorov Flow Dataset

## Physical system and reconstruction task

This dataset contains two-dimensional incompressible Navier–Stokes flow on the
periodic square `[0, 2*pi)^2`, driven by the Kolmogorov body force
`sin(4y) e_x`. The canonical Reynolds number is 40. The solver evolves
vorticity and the processing pipeline derives velocity `(u,v)` and zero-mean
pressure `p`.

The reconstruction target is the three-channel state `(u,v,p)` at one saved
time. The primary sparse-sensing task conditions on velocity and infers the
complete velocity and pressure fields. Solver-native vorticity is auxiliary
physics/evaluation context; it is not silently supplied as an observation or
treated as a fourth reconstruction target.

## Source and local link

- Optional local-only source: `datagen/data/processed/kolmogorov.h5`
- Local project payload: `Dataset/kolmogorov/kolmogorov.h5`
- Source dataset ID: `kolmogorov_canonical`
- Schema: `1.0`
- SHA-256 recorded by the producer:
  `b78471e94138de57cb39653fed6d88b618ccc8be4e16ba0f2647181a4a89e976`
- Approximate size: 16.2 GB (15,418.8 MiB)

From this directory, create the local-only relative link with:

```bash
ln -s ../../../datagen/data/processed/kolmogorov.h5 kolmogorov.h5
```

The source was generated and postprocessed by the separately retained local
`datagen/4_navier_stokes/` workflow. Full generation parameters and the source
raw directory are embedded in `metadata/json`. Before distributing the data,
add the final citation and redistribution license supplied by the dataset
owner; the current file records project provenance but no external release
license.

## Stored structure

- `fields [100,201,65536,1,1,3]`, `float32`, channels `u,v,p`
- `coordinates [65536,1,1,3]`, physical `(x,y,z)` coordinates
- `time [201]`, from 0 through 10 at spacing 0.05
- `conditions [100,3]`: Reynolds number, forcing amplitude, forcing wavenumber
- `auxiliary/vorticity [100,201,65536,1,1,1]`
- `trajectory_id [100]` and `seed [100]`
- `statistics/train_mean [3]` and `statistics/train_std [3]`
- trajectory splits: 80 train, 10 validation, 10 test

The verified logical grid is `256×256` in C order with `x` varying fastest.
Both axes are periodic. Pressure uses a zero-spatial-mean gauge. The current
conditions are fixed at `(40,1,4)`, so this file supports reconstruction and
cross-trajectory generalization, not held-out-condition claims.

Diagnostics include finiteness, divergence, pressure-gauge error, momentum
residuals, kinetic energy, enstrophy, and temporal vorticity variability.

## Validation and appropriate use

Run from the project root:

```bash
python scripts/validate_dataset.py Dataset/kolmogorov/kolmogorov.h5
```

Expected structural results are 100 trajectories, 201 frames, 65,536 points,
three fields, disjoint 80/10/10 trajectory splits, and no errors. Validation
samples only endpoint values and metadata rather than scanning the 16.2 GB
payload.

This is suitable for sparse multi-field reconstruction, velocity-to-pressure
inference, regular-grid operator comparisons, and later coherence/physics
evaluation. Formal condition generalization, nonperiodic-flow claims, or use of
vorticity as a measured input require a separately named protocol or dataset.
