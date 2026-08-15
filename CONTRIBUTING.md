# Contributing

Keep contributions within one clear boundary: a model adapter, coherence
family/component, dataset description, or case provider/config.

- General modules must not import named cases or the legacy demo.
- Dataset payloads, checkpoints, manifests, and run outputs stay out of Git.
- Public interfaces are typed and reject unknown/incompatible configuration.
- Randomness is explicit and seeded; evaluation observations are persisted.
- External code needs an upstream revision, license note, and attribution.
- Add one focused contract test and one small real-case config where relevant.
- Do not add a bespoke trainer or smoke script when the shared registry/CLI can
  exercise the contribution.

New coherence work registers a top-level family and its own components. Do not
force cross-spectrum or topology contributions into the global-distribution
`self/mutual/cross` structure.

Before changing a released benchmark, regenerate its results through
`scripts/aggregate_benchmark.py` and verify that the frozen sensor digest,
dataset checksum, config/checkpoint hashes, and code snapshot remain explicit.
Never add the mass-transport demo to a formal summary: its single trajectory
and three frames make it an integration fixture only.
