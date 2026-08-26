# Reproducibility and release trace

## Fresh collaborator workflow

From the standalone repository root:

```bash
conda env create -f environment.yml
conda activate phycoflow_reconstruction
python -m pip install -e '.[dev]'
python scripts/data/link_dataset.py \
  --case brusselator --source /absolute/path/to/brusselator.h5
bash scripts/smoke/reproduce_brusselator_integration.sh
```

The reproduction script launches from `cases/brusselator`, performs one plain
CPU update, reconstructs one validation trajectory on the fixed `u`-only
protocol, and prints the new run directory. `--max-steps 1` is intentionally an
integration budget, not a research result.

## Dataset and split rules

Keep normal payloads and links under local `datasets/` directories. Validate
them before training. Split at the declared trajectory/sample unit before
fitting normalization or coherence reference banks; never fit either from
validation or test data. Preserve field order, units, logical shape, and
coordinate interpretation in every serialized run contract.

Evaluation comparisons should reuse a fixed sensor manifest and query-index
set, and should report generation steps, seed, checkpoint choice, and device.
The evaluator records data/config/checkpoint hashes and sample identities so a
result can be traced even when its large payload is stored externally.

## Integration benchmark contract

`benchmarks/v0_integration/` contains a reviewed source-level integration
contract:

- `suite.yaml` records entries, one-update budgets, allowed/forbidden claims,
  dataset hash, and source files;
- the small fixed sensor manifest records portable indices and its digest;
- canonical configs and `results.md` describe the reproducible workflow;
- `AUDIT.md` records scope, lineage, license, and limitations.

The benchmark aggregation script is
`scripts/benchmarks/aggregate_benchmark.py`. Its output is a generated local
report; do not commit routine telemetry, HTML, repeated JSON/CSV reports,
large manifests, plots, or run summaries.

## Run lineage and external artifacts

A run stores its resolved config, manifest, status, checkpoint aliases, and
optional evaluation payloads under `cases/<case>/runs/`. Post-training stores
the source run/checkpoint identity and hashes before and after refinement. The
source checkpoint must remain immutable.

If a formal result needs to be shared, publish the complete checkpoint/data
bundle in an external artifact store without changing its recorded hash. A
traceable result whose payload is unavailable is not independently replayable.

## Statistical rule

Independent trajectories are the statistical unit whenever available. Compute
means, standard deviations, and standard errors across trajectory-level values,
not adjacent frames. If a release has one trajectory per row, report spread and
uncertainty as unavailable and make no significance or method-ranking claim.
