# Mass Transport–Fluid Case

The current three-frame file validates units, auxiliary fields, boundaries,
normalization, and code integration. It is explicitly not benchmark-eligible.
Any integration training must acknowledge this with
`--override dataset.allow_nonbenchmark=true`; the demo cannot produce a formal
validation/test comparison.

The integration path uses robust offset/99th-percentile normalization and
exposes auxiliary pressure and source arrays only as diagnostic context. Its
case diagnostics use nonperiodic derivatives, divergence, boundary-normal
flux, concentration extrema, pressure gauge, and source integral. Generated
figures and reports must retain the `benchmark_eligible: false` label.
