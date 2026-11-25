#!/bin/bash
set -euo pipefail
# Set CUDA_VISIBLE_DEVICES at runtime after host ownership admission.
campaign="${1:?campaign path required}"
epoch="${2:?epoch 100 or 500 required}"
attempt="${3:?unique output directory under followup assessment required}"
driver="${4:?immutable reviewed diagnostic driver path required}"
case "$epoch" in 100|500) ;; *) exit 2;; esac
: "${CUDA_VISIBLE_DEVICES:?one admitted runtime GPU required}"
code="$campaign/../round_6/operations/code/2a177eed1a3526b8ec451acb37075e6715503a46097754ccc809ab563da7a9a5"
python=/home/wanglz/miniconda3/envs/phycoflow_env/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CAR_REPO_ROOT=/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction
export KEOPS_CACHE_FOLDER="$campaign/cache/keops_grouped_pilot" MPLCONFIGDIR="$campaign/cache/matplotlib"
export PYTHONPATH="$code/src:$code"
exec "$python" "$code/scripts/posttraining/run_car_round6.py" --campaign "$campaign" lease --phase assessment --max-seconds 600 -- "$python" "$driver" --campaign "$campaign" --epoch "$epoch" --output "$attempt"
