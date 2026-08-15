#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}/Cases/brusselator"

python run.py validate --config configs/base/coordinate_mlp.yaml
run_dir="$(python run.py train-base \
  --config configs/base/coordinate_mlp.yaml \
  --override runtime.device=cpu \
  --override output.experiment_name=reproduce_phase8_coordinate_mlp \
  --max-steps 1 | tail -n 1)"
python run.py evaluate-run \
  --run "${run_dir}" \
  --sensor-config configs/sensors/u_only_random.yaml \
  --split validation \
  --max-samples 1 \
  --device cpu \
  --report-name reproduced_phase8
printf '%s\n' "${run_dir}"
