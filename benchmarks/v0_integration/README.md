# One-update integration protocol

This fixture defines a small, traceable workflow check for the shared CLI and
case contracts. It is not a performance result and must not be used to rank
models: one update and one validation trajectory cannot support uncertainty or
significance claims.

`suite.yaml` and `configs/` describe the four routes covered (plain base
training, data-driven coherence post-training, physics post-training, and
direct PINN). Run them from the standalone repository root with the local
Brusselator payload, keeping generated reports and manifests under ignored
paths. `results.md` and `AUDIT.md` record the intended scope and caveats; the
previous generated report payloads were removed from the source tree.

For the complete local smoke entry point, use
`bash scripts/smoke/reproduce_brusselator_integration.sh`.
