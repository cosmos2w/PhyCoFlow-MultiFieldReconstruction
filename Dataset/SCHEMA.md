# Dataset Schema

HDF5 is preferred. The canonical dense state is
`fields [B,T,Nx,Ny,Nz,C]`; coordinates are `[Nx,Ny,Nz,3]`, with size-one
unused dimensions. Flattened grids use `[B,T,N,1,1,C]` plus `grid_shape` and
`coordinate_order` metadata.

Required core datasets are `fields`, `coordinates`, `time`, and `conditions`.
Canonical multi-trajectory files also provide trajectory IDs, seeds, stored
train/validation/test trajectory indices, and training-only statistics.

The loader exposes either:

- `snapshot`: one `[N,C]` sample per `(trajectory,time)`; or
- `space_time_trajectory`: one `[T*N,C]` sample with `(t,x)` coordinates.

For a single long trajectory without stored splits, the first 80% of frames are
training, the next 10% validation, and final 10% test. Boundaries are computed
before `time_stride` and are never shuffled.

Trusted `.pt` files must contain a plain tensor/mapping representation of the
same logical fields. Pickled custom dataset classes are not accepted.
