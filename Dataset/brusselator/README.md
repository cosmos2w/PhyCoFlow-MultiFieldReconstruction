# Brusselator Dataset

This canonical 2-D periodic reaction-diffusion dataset contains coupled
dimensionless fields `u, v` on a 192 by 192 domain of length 20.

- Shape: `[100,241,36864,1,1,2]` (`float32`, approximately 4.4 GiB)
- Conditions: `A, B, diffusivity_u, diffusivity_v`
- Splits: 80 train, 10 validation, 10 test trajectories
- Schema: `1.0`
- SHA-256: `973907181a043471a2bdd40e11596a3403b966fa44af7c7dc8b50a611655fdac`

The source is the canonical generator under `datagen/3_brusselator/`; metadata
and training-only mean/std are embedded. The local link targets
`../../../datagen/data/processed/brusselator.h5`.

The primary task infers the full `u,v` state from sparse `u` and includes the
reverse `v -> u` experiment. Periodic PDE residuals are case diagnostics and
are not silently included in plain base training.
