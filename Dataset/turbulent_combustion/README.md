# Turbulent Combustion Dataset

The two legacy HDF5 files contain one long 10,000-frame trajectory over 40,300
points (`float32`, approximately 7.6 GiB each), with physical time step
`dt=1e-4`. They are used for sparse multi-field combustion reconstruction.

- `Merged_CH4COTU1P.h5`: `CH4, CO, T, U_1, p`.
- `Merged_COTU0U1P.h5`: `CO, T, U_0, U_1, p`.

Both have shape `[1,10000,40300,1,1,5]`. The mesh is stored as a point list;
the optional 403 by 100 interpretation must be validated before grid models are
enabled. The standard split is frames `[0,8000)` train, `[8000,9000)`
validation, and `[9000,10000)` test.

The merged legacy `time` array resets to zero at frame 4000 because its second
source segment retained a local clock. The canonical item identity and split
therefore use saved frame order/index. The loader preserves the raw stored time
and does not silently rewrite it; any future corrected physical-time axis must
be a documented case transform derived from verified segment provenance.

Local links target `../../../0_demo_TurbulentCombustion/Dataset/`. The files do
not contain complete modern provenance, units, or checksums; this limitation is
recorded by validation rather than guessed. Dataset statistics must come from
the training frames or a verified legacy checkpoint/sidecar.

Demo 50 uses `Merged_COTU0U1P.h5`. Its checkpoint contains stale field-name
metadata, so compatibility requires the explicit `CO,T,U_0,U_1,p` mapping.
