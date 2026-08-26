# Kuramoto–Sivashinsky Dataset

This canonical 1-D periodic dataset solves
`u_t + u u_x + u_xx + u_xxxx = 0` on a dimensionless domain of length 60.

- Shape: `[1000,401,256,1,1,1]` (`float32`, approximately 390 MiB)
- Field: `u`
- Conditions: advection, second-order, and fourth-order coefficients
- Splits: 80 train, 10 validation, 10 test trajectories
- Schema: `1.0`
- SHA-256: `92c122300788352c36f934e1e58508da66245fecd011ad1b4e36ed9cd1bbfd2c`

The reconstruction unit is the complete `[401,256,1]` saved space-time state.
The default observation keeps every fourth time and space point independently
and asks the model to reconstruct all omitted points. This is interpolation,
not future forecasting. The local link targets
`../../../datagen/data/processed/ks.h5`; the root `datagen/` tree is optional
local-only material and is not shipped with this branch.
