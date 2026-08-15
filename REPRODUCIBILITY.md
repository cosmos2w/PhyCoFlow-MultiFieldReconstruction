# Reproducibility and release trace

## Fresh collaborator workflow

From this project directory:

```bash
conda env create -f environment.yml
conda activate phycoflow_reconstruction
python -m pip install -e .
python scripts/link_dataset.py \
  --case brusselator --source /absolute/path/to/brusselator.h5
bash scripts/reproduce_brusselator_integration.sh
```

The reproduction script launches from `Cases/brusselator`, performs one plain
CPU update, reconstructs one validation trajectory on the fixed `u`-only
protocol, and prints the new run directory. `--max-steps 1` is intentionally an
integration budget, not a research result.

## Frozen Phase-8 artifact

`benchmarks/v0_integration/` contains:

- `suite.yaml`: reviewed entries, matched one-update budgets, allowed and
  forbidden claims, full dataset SHA-256, and relevant source-file list;
- `brusselator_u128_validation.json`: portable version-3 sensor indices and
  their self-checking digest;
- `results.yaml`: aggregate reconstruction, global-distribution coherence where
  available, PDE diagnostics, uncertainty availability, compute metrics, and a
  trajectory-aware statistics schema;
- `configs/` and `reports/`: exact resolved configurations and common evaluator
  reports referenced by the aggregate rows;
- `results.md`: compact human-readable integration table;
- `AUDIT.md`: scope, lineage, license, and limitation audit.

Each result row links to its evaluation report. That report records the resolved
config hash, checkpoint hash, sensor digest, query-index hash, dataset content
fingerprint, sample IDs, and plotting-payload hash. The release additionally
records the full dataset SHA-256 and hashes of the code files responsible for
training and evaluation.

Run products and checkpoints remain ignored because they are large and local.
When sharing a scientific result, publish the referenced checkpoint bundle in
an external artifact store without altering its hash. A row whose checkpoint
cannot be obtained is traceable but not independently replayable.

## Statistical rule

Independent trajectories are the statistical unit whenever available. Means,
standard deviations, and standard errors are computed across trajectory-level
values, not adjacent frames. The present release has one validation trajectory
per row, so standard deviation and standard error are explicitly unavailable.
No significance, uncertainty, or method-ranking claim is permitted from it.
