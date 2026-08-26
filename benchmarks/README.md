# Benchmarks

Benchmarks in this repository are reproducible protocols, not a warehouse for
run output. Keep source configs, scripts, small immutable fixtures, and
concise validation contracts in Git. Store checkpoints, telemetry, manifests,
reports, plots, and run summaries under local ignored output directories.

## Integration contract

`v0_integration/` documents the one-update CPU integration matrix for base,
data-driven coherence post-training, physics post-training, and direct PINN
paths. Its `suite.yaml`, canonical configs, `results.md`, and `AUDIT.md` are
the durable contract. Reports and the fixed sensor manifest are generated
locally before aggregation; they are intentionally not source artifacts.

Run the source benchmark after producing the required local reports:

```bash
python scripts/benchmarks/aggregate_benchmark.py \
  --suite benchmarks/v0_integration/suite.yaml \
  --output /tmp/phycoflow-integration-results.yaml
```

The GL-RBF/CQ protocol and compatibility-sensitive fixtures are documented in
[`gl_rbf_cq_migration_200ep/`](gl_rbf_cq_migration_200ep/).

No benchmark command belongs in normal CI if it requires a large dataset,
checkpoint, GPU, or generated report. Use `pytest` for the lightweight
contracts and `scripts/smoke/models.py` for the local GPU-0 matrix.
