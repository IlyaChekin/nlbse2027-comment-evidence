#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
exec 9>verification/gpu_queue.lock
flock -w 60 9
run() {
  local name="$1"
  shift
  mkdir -p "nlbse/runs/$name"
  if test -s "nlbse/runs/$name/metrics.json"; then return; fi
  .venv/bin/python -u nlbse/cuda_launch.py nlbse/pipeline.py "$@" --run "$name" >"nlbse/runs/$name/run.log" 2>&1
}
model=nlbse/runs/final_fit/checkpoint
run compact6_kd_pilot train --pilot --save-checkpoint --weighted --lr 2e-5 --train-attention sdpa --bucket-window 2048 \
  --init-from "$model" --retain-layers 1 3 5 7 9 11 --teacher-from "$model"
run compact6_kd_profile_tune evaluate --internal --pilot --profile-reps 1 --bucket-window 2048 \
  --checkpoint nlbse/runs/compact6_kd_pilot/checkpoint --thresholds nlbse/runs/compact6_kd_pilot/thresholds.json
