#!/bin/bash
# Base-train gl_rbf_cq on turbulent combustion in mit_preemptable.
#
# Preemption is expected here, so the job is requeueable and each attempt
# continues the same run directory from checkpoints/last.pt. Slurm keeps the
# job id across a requeue, which is what lets an attempt find the run its
# predecessor started.
#
#   sbatch scripts/slurm/train_tc_gl_rbf_cq_preemptable.sh
#
#SBATCH --job-name=tc_glrbfcq
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PYTHON="${PYTHON:-$HOME/.conda/envs/tda/bin/python}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASE_DIR="$REPO/cases/turbulent_combustion"
CONFIG="configs/base/gl_rbf_cq_cached_kv_5000ep.yaml"
EXPERIMENT="tc_gl_rbf_cq_cached_kv_5000ep"

JOB_ID="${SLURM_JOB_ID:-manual}"
STATE_FILE="$CASE_DIR/runs/.slurm/${JOB_ID}.run_dir"

# Stable across requeues of this job id, so compiled KeOps kernels survive a
# preemption instead of being rebuilt on every attempt.
export PYKEOPS_CACHE_FOLDER="$HOME/.cache/keops/$JOB_ID"

mkdir -p "$PYKEOPS_CACHE_FOLDER" "$(dirname "$STATE_FILE")"
cd "$CASE_DIR"

echo "=== attempt $(date -u +%FT%TZ) job=$JOB_ID restarts=${SLURM_RESTART_COUNT:-0} node=$(hostname) ==="

list_runs() {
    # Only runs whose directory name carries this config's digest are ours.
    find "runs/$EXPERIMENT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort
}

RUN_DIR=""
if [[ -s "$STATE_FILE" ]]; then
    CANDIDATE="$(cat "$STATE_FILE")"
    if [[ -f "$CANDIDATE/checkpoints/last.pt" ]]; then
        RUN_DIR="$CANDIDATE"
    else
        # Preempted before the first checkpoint landed; nothing to continue.
        echo "no checkpoint in $CANDIDATE; starting a fresh run"
        rm -f "$STATE_FILE"
    fi
fi

if [[ -n "$RUN_DIR" ]]; then
    echo "resuming $RUN_DIR"
    exec "$PYTHON" run.py train-base --config "$CONFIG" --resume "$RUN_DIR"
fi

# Fresh start: the run directory name embeds a UTC timestamp, so it cannot be
# predicted. Record whatever directory the trainer creates, then wait on it.
BEFORE="$(mktemp)"
trap 'rm -f "$BEFORE"' EXIT
list_runs > "$BEFORE"

"$PYTHON" run.py train-base --config "$CONFIG" &
TRAIN_PID=$!

# Forward Slurm's preemption signal so the trainer exits promptly.
trap 'kill -TERM "$TRAIN_PID" 2>/dev/null || true' TERM INT

for _ in $(seq 1 900); do
    NEW="$(list_runs | comm -13 "$BEFORE" - | head -n 1)"
    if [[ -n "$NEW" ]]; then
        printf '%s\n' "$CASE_DIR/${NEW#./}" > "$STATE_FILE"
        echo "run directory: $(cat "$STATE_FILE")"
        break
    fi
    kill -0 "$TRAIN_PID" 2>/dev/null || break
    sleep 2
done

wait "$TRAIN_PID"
