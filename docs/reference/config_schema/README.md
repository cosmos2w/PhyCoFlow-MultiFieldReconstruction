# Configuration schema snapshots

The YAML files in this directory are descriptive reference snapshots for reviewers and documentation. They are not loaded by the runtime. The authoritative stage and nested-key validation lives in `src/phycoflow_reconstruction/config/schema.py` and `validate.py`; changes to those Python validators must be covered by the configuration contract tests.

Keep these snapshots synchronized when the public stage contract changes, but do not use them as a second parser or validation implementation.
