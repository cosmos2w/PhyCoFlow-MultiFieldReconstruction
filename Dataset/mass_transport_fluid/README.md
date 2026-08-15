# Mass Transport–Fluid Demo

This small nonperiodic Elder-type density-coupled Darcy transport file is an
integration fixture, not a formal benchmark.

- Shape: `[1,3,1024,1,1,3]`, 32 by 32 grid
- Fields: `u_x [m/s], u_y [m/s], concentration [mol/m^3]`
- Auxiliary: pressure and source field
- Conditions: `A, x0, y0, s`
- Split: one training trajectory; no validation/test trajectories
- Schema: `1.0`
- SHA-256: `d5a49688b0d1b9309119f1ce8189b6bf76029d9b8cd8ad8b64273d6dde604a15`

It validates physical units, robust offset/99th-percentile normalization,
auxiliary data, boundaries, and visualizations. Formal comparisons require a
larger named production variant. The local link targets
`../../../datagen/data/processed/mass_transport_fluid_demo.h5`.
