#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
PYTHON=.venv/bin/python
exec 9>verification/gpu_queue.lock
flock -n 9
run() {
  local name="$1"
  shift
  mkdir -p "nlbse/runs/$name"
  if test -s "nlbse/runs/$name/metrics.json"; then return; fi
  "$PYTHON" -u nlbse/cuda_launch.py nlbse/pipeline.py "$@" --run "$name" >"nlbse/runs/$name/run.log" 2>&1
}
model=nlbse/runs/final_fit/checkpoint
threshold=nlbse/runs/final_fit/thresholds.json
run packing_control_tune evaluate --internal --pilot --profile-reps 1 --checkpoint "$model" --thresholds "$threshold"
run packing_sorted_tune evaluate --internal --pilot --profile-reps 1 --bucket-window 2048 --checkpoint "$model" --thresholds "$threshold"
run compact6_pilot train --pilot --save-checkpoint --weighted --lr 2e-5 --train-attention sdpa --bucket-window 2048 \
  --init-from "$model" --retain-layers 1 3 5 7 9 11
run compact6_profile_tune evaluate --internal --pilot --profile-reps 1 --bucket-window 2048 \
  --checkpoint nlbse/runs/compact6_pilot/checkpoint --thresholds nlbse/runs/compact6_pilot/thresholds.json
