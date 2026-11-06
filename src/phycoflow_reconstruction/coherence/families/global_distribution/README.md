# State occupation definitions

`marginal_mixed_mean_v2` retains equal mass per selected node. Its objective is
the mean of the per-field empirical squared 1-D Wasserstein distances plus the
mean over its fixed bank of mixed unit directions. Historical defaults and
artifacts retain their existing definitions.

`marginal_mixed_weighted_v3` asks the same marginal and mixed-projection
questions for an explicitly supplied physical occupation measure. Every
projected endpoint is sorted with its own masses; the implementation integrates
the squared quantile gap across the merged endpoint CDF breakpoints. Masses are
fixed geometry inputs, not trainable parameters. Value gradients are piecewise
smooth; stable sorting chooses a deterministic branch at equal-value ties.
Physical and model units are selected exactly as in v2; in the standard recipe
the fields are in the fixed dataset-normalized model units.

The family context must include both `generated_weights` and
`reference_weights`, each `[N]` or `[B,N]` aligned with that endpoint. Optional
`generated_mask` and `reference_mask` are Boolean tensors of the same shapes;
invalid or zero-mass entries do not contribute. Each endpoint must retain
positive finite mass. Paired targets may use the same physical masses; reference
banks are unsupported until they save their own endpoint masses.

`coherence.measures.rectilinear_nodal_weights` constructs nonperiodic
trapezoidal, or equivalently lumped multilinear finite-element, masses on a
complete strictly monotone rectilinear grid, using actual coordinates and the
rectangle bounded by the extreme nodes. A case must declare coordinate units,
metric and boundaries; unknown units remain unknown. Masks, holes, periodic
domains and unstructured meshes need their own validated physical provider.
`subset_inclusion_weights` implements known inclusion-probability correction
without replacing or duplicating nodes. For a uniformly sampled sparse panel,
normalizing retained native masses is the same ratio estimator because the
inclusion probabilities are common; it approximates full-native occupation and
is not an exact quadrature rule. Compare the sparse and full-native queries on
fixed anchors before making a learning claim.

Use `measure_fingerprint` and `bind_measure_artifact(receipt, weights=...)` to
freeze the native mass vector, support identity and declared provider metadata
with the family. Runtime `measure_metadata` can carry `native_measure_sha256`
to check the panel against the frozen native identity. Selected query IDs and
their sampling policy belong in the run's evaluation/training provenance.

`pair_diagnostics` is independent of mutual-pair training. It returns named
two-field raw-value occupation histograms with shared endpoint ranges and
within-endpoint weighted rank-occupation histograms. Equal values receive the
same weighted midrank. These are separate diagnostic queries and do not alter
the optimized State score. Runtime callers can also request them explicitly via
`context["include_pair_occupations"]`; generic training detail does not create
these larger histogram payloads automatically.
