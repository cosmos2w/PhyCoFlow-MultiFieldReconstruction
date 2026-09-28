# Active-emulsion dataset contract

Provide a canonical HDF5 payload following [the shared schema](../SCHEMA.md).
The example path is `generated/fields.h5`; a native source run supplies its actual
path when post-training inherits the source configuration.

- Physical fields are ordered `[phi, vx, vy]` on a structured 128×128 grid.
- Each stored trajectory contains one snapshot. Original simulation identity,
  frame, time, and split are retained in `metadata/json.samples`.
- The case's subset mapping expects `regime`, `m`, `file`, `frame`, `time`, and
  `split` columns. Regime labels describe generator morphology, not measured
  Betti classes. Other metadata schemas can configure the subset column names.
- Stored train, validation, and test splits must keep simulations disjoint.
- Coordinate order and periodic lengths must agree with the case geometry.
- Fields remain in physical units. Source normalization is applied by the model's
  physical-field adapter; the HDF5 normalizer is identity.

Online training augmentation applies periodic translations and right-angle
rotations, including the corresponding velocity-vector transformation. It never
changes the stored payload. Payloads, source hashes, generated manifests, and
checkpoint-specific conversion records remain local and are ignored by Git.
