# CoastalBench inputs

Payloads and links remain local. The [case workflow](../../cases/coastal_bench/README.md)
accepts canonical point-cloud HDF5 inputs with fields `[u, v, temp, salt]` and
stored trajectory-level train/validation/test splits.

| Prepared profile | Expected local path | Normalization |
|---|---|---|
| Original heavy export | `coastal_bench.h5` | Mean/std recomputed on TRAIN by the case preparation script |
| Calendar hourly crop | `calendar_576days_hourly/coastal_bench.h5` | Stored TRAIN median and p99–p1, `robust_99` |

The original profile has 30,542 ocean points. The calendar recipe expects 30,389
fixed wet points, 24 hourly frames per day, 576 TRAIN days and 48 days in each
held-out split. Its native raster extent is `[201, 381]`; the model's logical
shape remains `[30389]`. Integer `auxiliary/native_grid_indices` identify each
point's native row and column. This array is required for physical FEM geometry
and masked cubical topology on the curvilinear ocean grid.

Use the declared metadata and units of the selected payload. The original export
uses unknown units; the prepared calendar contract declares `m/s`, `m/s`, `degC`
and `PSU`. The case does not infer the physical depth of source layer zero.
Preserve stored splits and temporal cadence; disjoint days do not by themselves
establish independent observations. Never compute normalization from holdouts.
